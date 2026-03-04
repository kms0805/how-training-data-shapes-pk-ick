#!/usr/bin/env python3
# ==========================================
# 20-sample Probe (First-Token Accuracy / Exact-Match)
# - train: param / in_ctx / pert_ctx_orig / pert_ctx_pert
# - unknown: ood_in_ctx
# - additional: multi_in_ctx / multi_ood_in_ctx
# ------------------------------------------
# Notes:
#   - pert modes (pert_ctx_orig, pert_ctx_pert) use ATTR_KEYS_PERT only
#   - other modes (param, in_ctx, ood_in_ctx, multi_in_ctx, multi_ood_in_ctx) use ATTR_KEYS_ALL
#   - single seed, single run, single output CSV
# ==========================================
import os, random, re, json, logging, gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("probe")

# ---------- user config ----------
MODEL_ROOT = Path(os.environ.get("PROBE_MODEL_ROOT", "./checkpoints"))
# (optional) limit step range for specific runs. Leave empty to use all steps.
# e.g.: CKPT_RANGE = {"false_concat_10": (1000, 32000)}
CKPT_RANGE: Dict[str, Tuple[int, int]] = {}

# (optional) filter run directory names by substring. None = use all.
FILTER_RUN_NAME_SUBSTR: Optional[str] = None

TRAIN_JSON = "data/bioS_train.json"
TRAIN_PERT_JSON = "data/bioS_pert.json"

UNKNOWN_JSON = "data/bioS_unknown.json"

# ----- attribute keys -----
ATTR_KEYS_PERT = ["birth_date", "university"]
ATTR_KEYS_ALL  = ["birth_city", "birth_date", "major", "university"]

SAMPLE_N_TRAIN   = 200                     # number of train samples
SAMPLE_N_UNKNOWN = 200                     # number of unknown samples

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- utilities ----------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_run_dirs(root: Path) -> List[Path]:
    """Return only run directories that contain checkpoint-* subdirectories."""
    runs = []
    if not root.exists():
        log.warning(f"MODEL_ROOT does not exist: {root.resolve()}")
        return runs
    for p in root.iterdir():
        if p.is_dir():
            try:
                has_ckpt = any(child.is_dir() and child.name.startswith("checkpoint") for child in p.iterdir())
            except PermissionError:
                has_ckpt = False
            if has_ckpt:
                runs.append(p)
    runs.sort(key=lambda x: x.name)
    return runs

def list_ckpts_in_range(model_dir: Path, lo: int = 0, hi: int = 10**12) -> List[Path]:
    """Return sorted checkpoint directories where lo <= step <= hi and step is a multiple of 100."""
    cand = []
    for p in model_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("checkpoint"):
            continue
        m = re.findall(r"\d+", p.name)
        if not m:
            continue
        step = int(m[0])
        if lo <= step <= hi and step % 100 == 0:
            cand.append((step, p))
    return [p for step, p in sorted(cand, key=lambda x: x[0])]

def _get_first_present_key(d: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return k
    return None

def split_train_into_o_p_lists(train_raw, fallback_pert_json: Optional[str]) -> Tuple[List[dict], List[dict]]:
    """
    Extract original (o) and perturbed (p) lists from train_raw.
    - Format A: {"orig":[...], "pert":[...]}  -> use directly
    - Format B: [ { ..., "pert": {"test_corpus":..., "probes":...}, ... }, ... ] -> extract from each entry
    - Format C: [ { "test_corpus":..., "probes":..., "test_corpus_pert":..., "probes_pert":... }, ... ] -> extract by keys
    - Fallback: read from fallback_pert_json and pair by index
    """
    # Format A: dict with "orig" and "pert"
    if isinstance(train_raw, dict):
        if "orig" in train_raw and "pert" in train_raw:
            o_raw, p_raw = train_raw["orig"], train_raw["pert"]
            assert len(o_raw) == len(p_raw), "orig/pert lengths differ."
            return o_raw, p_raw

    # Format B/C: list
    if isinstance(train_raw, list):
        o_list, p_list = [], []
        all_have_pert = True
        for e in train_raw:
            # original keys
            tc_o_key = _get_first_present_key(e, ["test_corpus", "orig_test_corpus", "ctx", "ctx_o"])
            pr_o_key = _get_first_present_key(e, ["probes", "orig_probes"])

            # perturbed: nested 'pert' or flat *_pert
            p_block = None
            if isinstance(e.get("pert"), dict):
                p_block = e["pert"]
            else:
                # flat key candidates
                tc_p_key = _get_first_present_key(e, ["test_corpus_pert", "pert_test_corpus", "p_test_corpus", "ctx_p"])
                pr_p_key = _get_first_present_key(e, ["probes_pert", "pert_probes", "p_probes"])
                if tc_p_key and pr_p_key:
                    p_block = {"test_corpus": e[tc_p_key], "probes": e[pr_p_key]}

            if tc_o_key and pr_o_key:
                o_entry = {"test_corpus": e[tc_o_key], "probes": e[pr_o_key]}
                o_list.append(o_entry)
            else:
                all_have_pert = False  # can't pair without original
                break

            if p_block and "test_corpus" in p_block and "probes" in p_block:
                p_list.append({"test_corpus": p_block["test_corpus"], "probes": p_block["probes"]})
            else:
                all_have_pert = False

        if all_have_pert and len(o_list) == len(p_list) and len(o_list) > 0:
            return o_list, p_list
        # else: fall through to fallback

    # fallback: read from separate pert JSON
    if fallback_pert_json is None:
        raise ValueError(
            "Could not find perturbation info in train JSON. "
            "Either include orig/pert in one file, or set TRAIN_PERT_JSON path."
        )
    pert_raw = load_json(fallback_pert_json)
    if isinstance(train_raw, list) and isinstance(pert_raw, list):
        assert len(train_raw) == len(pert_raw), "train/pert lengths differ."
        return train_raw, pert_raw
    elif isinstance(train_raw, dict) and isinstance(pert_raw, dict):
        return train_raw.get("orig", []), pert_raw.get("pert", [])
    else:
        raise ValueError("Fallback pert format is unexpected.")

# ---------- example construction ----------
def _make_row(tok, mode: str, attr: str, prefix: str, tgt: str, full: Optional[str] = None):
    """
    prefix = (concatenated contexts) + ' ' + prompt  (or just ' ' + prompt)
    full   = prefix + tgt
    Computes prompt_len and target_len safely, accounting for BPE merge boundaries.
    """
    if full is None:
        full = prefix + tgt

    # length must be computed from the diff between prefix and full (BPE-merge safe)
    prompt_len = len(tok(prefix)["input_ids"])
    full_len   = len(tok(full)["input_ids"])
    target_len = full_len - prompt_len

    return {
        "mode": mode,
        "attr": attr,
        "text": full,
        "prompt_len": prompt_len,
        "target_len": target_len
    }

def _safe_get_probe(d: dict, attr: str) -> Optional[Tuple[str, str]]:
    """Return (prompt, tgt) from d['probes'][attr] if present, else None."""
    probes = d.get("probes", {})
    pair = probes.get(attr, None)
    if pair is None:
        return None
    # validate pair format
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return pair[0], pair[1]
    raise ValueError(f"probes['{attr}'] is not a (prompt, tgt) pair: {type(pair)}")

def build_examples_train(o_raw: List[dict], p_raw: List[dict], tok,
                         attr_keys_nonpert: List[str], attr_keys_pert: List[str],
                         sample_n: int) -> Dataset:
    """
    Generate 4 modes from train data:
      - non-pert (param / in_ctx): uses attr_keys_nonpert
      - pert (pert_ctx_orig / pert_ctx_pert): uses attr_keys_pert
    """
    n = min(len(o_raw), len(p_raw))
    if n == 0:
        raise ValueError("Train data is empty.")
    k = min(sample_n, n)
    idxs = random.sample(range(n), k)

    rows = []
    for i in idxs:
        o, p = o_raw[i], p_raw[i]
        ctx_o, ctx_p = o["test_corpus"], p["test_corpus"]

        # ----- non-pert modes -----
        for attr in attr_keys_nonpert:
            pair = _safe_get_probe(o, attr)
            if pair is None:
                continue
            prompt_o, tgt_o = pair

            # param
            prefix = " " + prompt_o
            rows.append(_make_row(tok, "param", attr, prefix, tgt_o))

            # in_ctx
            prefix = " " + ctx_o + " " + prompt_o
            rows.append(_make_row(tok, "in_ctx", attr, prefix, tgt_o))

        # ----- pert modes -----
        for attr in attr_keys_pert:
            pair_o = _safe_get_probe(o, attr)
            pair_p = _safe_get_probe(p, attr)
            if pair_o is None or pair_p is None:
                continue
            prompt_o, tgt_o = pair_o
            prompt_p, tgt_p = pair_p

            # pert_ctx_orig
            prefix = " " + ctx_p + " " + prompt_o
            rows.append(_make_row(tok, "pert_ctx_orig", attr, prefix, tgt_o))

            # pert_ctx_pert
            prefix = " " + ctx_p + " " + prompt_p
            rows.append(_make_row(tok, "pert_ctx_pert", attr, prefix, tgt_p))

    return Dataset.from_list(rows)

def build_examples_ood(u_raw: List[dict], tok, attr_keys: List[str], sample_n: int) -> Dataset:
    """Generate ood_in_ctx from unknown data (uses ATTR_KEYS_ALL, includes target_len)."""
    n = len(u_raw)
    if n == 0:
        raise ValueError("Unknown data is empty.")
    k = min(sample_n, n)
    idxs = random.sample(range(n), k)

    rows = []
    for i in idxs:
        o = u_raw[i]
        ctx_o = o["test_corpus"]
        for attr in attr_keys:
            pair = _safe_get_probe(o, attr)
            if pair is None:
                continue
            prompt_o, tgt_o = pair
            prefix = " " + ctx_o + " " + prompt_o
            rows.append(_make_row(tok, "ood_in_ctx", attr, prefix, tgt_o))
    return Dataset.from_list(rows)

def _sample_others(n_total: int, self_idx: int, num_others: int) -> List[int]:
    """Sample up to num_others indices excluding self_idx. Safe when data is small."""
    pool = [j for j in range(n_total) if j != self_idx]
    k = min(len(pool), num_others)
    return random.sample(pool, k) if k > 0 else []

def build_examples_train_multi(o_raw: List[dict], tok, attr_keys: List[str],
                               sample_n: int, num_others: int = 2, sep: str = " ") -> Dataset:
    """
    Train-based multiple in-ctx:
    - Concatenate own paragraph + num_others other paragraphs (shuffled)
    - Then prompt_o + tgt_o
    - Uses ATTR_KEYS_ALL
    """
    n = len(o_raw)
    if n == 0:
        raise ValueError("Train data is empty.")
    k = min(sample_n, n)
    idxs = random.sample(range(n), k)

    rows = []
    for i in idxs:
        others = _sample_others(n, i, num_others)
        ctxs = [o_raw[i]["test_corpus"]] + [o_raw[j]["test_corpus"] for j in others]
        random.shuffle(ctxs)
        mixed_ctx = sep.join(ctxs)

        for attr in attr_keys:
            pair = _safe_get_probe(o_raw[i], attr)
            if pair is None:
                continue
            prompt_o, tgt_o = pair
            prefix = " " + mixed_ctx + " " + prompt_o
            rows.append(_make_row(tok, "multi_in_ctx", attr, prefix, tgt_o))
    return Dataset.from_list(rows)

def build_examples_ood_multi(u_raw: List[dict], tok, attr_keys: List[str],
                             sample_n: int, num_others: int = 2, sep: str = " ") -> Dataset:
    """
    Unknown-based multiple ood-in-ctx:
    - Concatenate own paragraph + num_others other paragraphs (shuffled)
    - Then prompt_o + tgt_o
    - Uses ATTR_KEYS_ALL
    """
    n = len(u_raw)
    if n == 0:
        raise ValueError("Unknown data is empty.")
    k = min(sample_n, n)
    idxs = random.sample(range(n), k)

    rows = []
    for i in idxs:
        others = _sample_others(n, i, num_others)
        ctxs = [u_raw[i]["test_corpus"]] + [u_raw[j]["test_corpus"] for j in others]
        random.shuffle(ctxs)
        mixed_ctx = sep.join(ctxs)

        for attr in attr_keys:
            pair = _safe_get_probe(u_raw[i], attr)
            if pair is None:
                continue
            prompt_o, tgt_o = pair
            prefix = " " + mixed_ctx + " " + prompt_o
            rows.append(_make_row(tok, "multi_ood_in_ctx", attr, prefix, tgt_o))
    return Dataset.from_list(rows)

# ---------- evaluation ----------
@torch.inference_mode()
def compute_metrics_for_example(model, tok, ex, device: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns: (acc1, em)
      - acc1: First-token accuracy (0/1) or None (not computable)
      - em  : Exact-match        (0/1) or None (not computable)
    """
    max_len = getattr(model.config, "n_ctx", getattr(model.config, "n_positions", 512))
    ids = tok(ex["text"], return_tensors="pt",
              truncation=True, max_length=max_len).input_ids.to(device)
    logits = model(ids).logits.squeeze(0)  # [seq_len, vocab]

    seq_len = ids.shape[1]
    pl = int(ex["prompt_len"])
    tl = int(ex["target_len"])

    acc1 = None
    em   = None

    # First token
    if pl < seq_len and pl - 1 >= 0:
        pred1 = torch.argmax(logits[pl - 1]).item()
        true1 = ids[0, pl].item()
        acc1 = int(pred1 == true1)

    # Exact match: only evaluate when full target fits within input
    if (tl > 0) and (pl + tl <= seq_len) and (pl - 1 >= 0):
        pred_seq = torch.argmax(logits[pl - 1: pl + tl - 1], dim=-1)
        true_seq = ids[0, pl: pl + tl]
        em = int(torch.equal(pred_seq, true_seq))

    return acc1, em

def _init_score_bucket():
    return {"acc1_sum": 0, "acc1_n": 0, "em_sum": 0, "em_n": 0}

def _finalize_scores(buckets: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    out = {}
    for mode, b in buckets.items():
        acc1 = 100.0 * b["acc1_sum"] / b["acc1_n"] if b["acc1_n"] > 0 else float("nan")
        em   = 100.0 * b["em_sum"]   / b["em_n"]   if b["em_n"]   > 0 else float("nan")
        out[f"acc1/{mode}"] = acc1
        out[f"em/{mode}"]   = em
    return out

def probe_ckpt(ckpt_path: Path, tok, eval_ds: Dataset, device: str) -> Dict[str, float]:
    """Compute per-mode First-Token Accuracy / Exact-Match (%) for a single checkpoint."""
    log.info(f"Evaluating: {ckpt_path}")
    model = GPT2LMHeadModel.from_pretrained(ckpt_path).to(device).eval()

    buckets: Dict[str, Dict[str, int]] = {}
    for ex in eval_ds:
        mode = ex["mode"]
        if mode not in buckets:
            buckets[mode] = _init_score_bucket()

        acc1, em = compute_metrics_for_example(model, tok, ex, device)
        if acc1 is not None:
            buckets[mode]["acc1_sum"] += acc1
            buckets[mode]["acc1_n"]   += 1
        if em is not None:
            buckets[mode]["em_sum"] += em
            buckets[mode]["em_n"]   += 1

    # free memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return _finalize_scores(buckets)

# ---------- tokenizer ----------
tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

# ---------- load data ----------
log.info("Loading data...")
train_raw = load_json(TRAIN_JSON)
o_raw_full, p_raw_full = split_train_into_o_p_lists(train_raw, TRAIN_PERT_JSON)

unknown_raw_full = load_json(UNKNOWN_JSON)
# only list format supported (extend if needed)
if isinstance(unknown_raw_full, dict):
    if "data" in unknown_raw_full:
        unknown_raw_full = unknown_raw_full["data"]
    else:
        raise ValueError("Unknown JSON is a dict. Expected a list.")

# ---------- set seed & build eval dataset ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

log.info("Building evaluation samples...")
eval_train       = build_examples_train(o_raw_full, p_raw_full, tok, ATTR_KEYS_ALL, ATTR_KEYS_PERT, SAMPLE_N_TRAIN)
eval_ood         = build_examples_ood(unknown_raw_full, tok, ATTR_KEYS_ALL, SAMPLE_N_UNKNOWN)
eval_train_multi = build_examples_train_multi(o_raw_full, tok, ATTR_KEYS_ALL, SAMPLE_N_TRAIN)
eval_ood_multi   = build_examples_ood_multi(unknown_raw_full, tok, ATTR_KEYS_ALL, SAMPLE_N_UNKNOWN)

eval_all = Dataset.from_list(
    [ex for ex in eval_train] +
    [ex for ex in eval_ood] +
    [ex for ex in eval_train_multi] +
    [ex for ex in eval_ood_multi]
)

# ---------- scan run directories ----------
run_dirs_master = list_run_dirs(MODEL_ROOT)
if FILTER_RUN_NAME_SUBSTR is not None:
    run_dirs_master = [p for p in run_dirs_master if FILTER_RUN_NAME_SUBSTR in p.name]
if not run_dirs_master:
    log.warning(f"No run directories with checkpoints found under {MODEL_ROOT}.")

# ---------- evaluate all checkpoints ----------
all_records = []

for run_dir in run_dirs_master:
    run_name = run_dir.name
    lo, hi = CKPT_RANGE.get(run_name, (0, 10**12))
    ckpts = list_ckpts_in_range(run_dir, lo, hi)
    if not ckpts:
        log.warning(f"{run_name}: no checkpoints in range ({lo}, {hi}).")
        continue

    for ckpt in tqdm(ckpts, desc=f"Probing {run_name}"):
        step = int(re.findall(r"\d+", ckpt.name)[0])
        rec  = probe_ckpt(ckpt, tok, eval_all, DEVICE)
        rec.update({"model": run_name, "step": step})
        all_records.append(rec)

# ---------- save results ----------
if all_records:
    df = pd.DataFrame(all_records).sort_values(["model", "step"])
    metric_cols = sorted([c for c in df.columns if "/" in c])
    df = df[["model", "step"] + metric_cols]

    out_path = Path("probe_results.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"Results saved: {out_path.resolve()}")

    try:
        from IPython.display import display
        log.info("Preview (top 20 rows):")
        display(df.head(20))
    except Exception:
        print(df.head(20))
else:
    log.warning("No results collected.")
