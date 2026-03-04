#!/usr/bin/env python3
# ==========================================
# Knowledge Probing (§2.3)
#
# Evaluates 200 randomly sampled entities per split using exact-match.
#
# Output modes (= CSV column suffixes):
#   pku     → Acc_PKU   : Parametric Knowledge Utilization (E_train, no context)
#   icku    → Acc_ICKU  : In-Context Knowledge Utilization (E_unseen, multi-entity context)
#   pref_pk → Pref_PK   : Parametric preference under knowledge conflict
#   pref_ick→ Pref_ICK  : In-context preference under knowledge conflict
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

# ---------- config ----------
MODEL_ROOT = Path(os.environ.get("PROBE_MODEL_ROOT", "./checkpoints"))
CKPT_RANGE: Dict[str, Tuple[int, int]] = {}
FILTER_RUN_NAME_SUBSTR: Optional[str] = None

TRAIN_JSON      = "data/bioS_train.json"
TRAIN_PERT_JSON = "data/bioS_pert.json"
UNSEEN_JSON     = "data/bioS_unseen.json"

# Attributes probed for conflict resolution (§2.3)
ATTR_KEYS_PERT = ["birth_date", "university"]
# All attributes probed for PKU / ICKU
ATTR_KEYS_ALL  = ["birth_city", "birth_date", "major", "university"]

N_SAMPLES = 200   # entities sampled per evaluation (§2.3)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- utilities ----------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_run_dirs(root: Path) -> List[Path]:
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
    """Extract original and perturbed lists from train data."""
    if isinstance(train_raw, dict):
        if "orig" in train_raw and "pert" in train_raw:
            o_raw, p_raw = train_raw["orig"], train_raw["pert"]
            assert len(o_raw) == len(p_raw), "orig/pert lengths differ."
            return o_raw, p_raw

    if isinstance(train_raw, list):
        o_list, p_list = [], []
        all_have_pert = True
        for e in train_raw:
            tc_o_key = _get_first_present_key(e, ["test_corpus", "orig_test_corpus", "ctx", "ctx_o"])
            pr_o_key = _get_first_present_key(e, ["probes", "orig_probes"])

            p_block = None
            if isinstance(e.get("pert"), dict):
                p_block = e["pert"]
            else:
                tc_p_key = _get_first_present_key(e, ["test_corpus_pert", "pert_test_corpus", "p_test_corpus", "ctx_p"])
                pr_p_key = _get_first_present_key(e, ["probes_pert", "pert_probes", "p_probes"])
                if tc_p_key and pr_p_key:
                    p_block = {"test_corpus": e[tc_p_key], "probes": e[pr_p_key]}

            if tc_o_key and pr_o_key:
                o_list.append({"test_corpus": e[tc_o_key], "probes": e[pr_o_key]})
            else:
                all_have_pert = False
                break

            if p_block and "test_corpus" in p_block and "probes" in p_block:
                p_list.append({"test_corpus": p_block["test_corpus"], "probes": p_block["probes"]})
            else:
                all_have_pert = False

        if all_have_pert and len(o_list) == len(p_list) and len(o_list) > 0:
            return o_list, p_list

    if fallback_pert_json is None:
        raise ValueError("Could not find perturbation info in train JSON.")
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
    if full is None:
        full = prefix + tgt
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
    probes = d.get("probes", {})
    pair = probes.get(attr, None)
    if pair is None:
        return None
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return pair[0], pair[1]
    raise ValueError(f"probes['{attr}'] is not a (prompt, tgt) pair: {type(pair)}")

def _sample_others(n_total: int, self_idx: int, num_others: int) -> List[int]:
    pool = [j for j in range(n_total) if j != self_idx]
    k = min(len(pool), num_others)
    return random.sample(pool, k) if k > 0 else []

def build_examples_pku(o_raw: List[dict], tok, sample_n: int) -> Dataset:
    """Acc_PKU: parametric recall without context (E_train)."""
    n = len(o_raw)
    k = min(sample_n, n)
    idxs = random.sample(range(n), k)
    rows = []
    for i in idxs:
        for attr in ATTR_KEYS_ALL:
            pair = _safe_get_probe(o_raw[i], attr)
            if pair is None:
                continue
            prompt, tgt = pair
            rows.append(_make_row(tok, "pku", attr, " " + prompt, tgt))
    return Dataset.from_list(rows)

def build_examples_icku(u_raw: List[dict], tok, sample_n: int,
                        num_others: int = 2, sep: str = " ") -> Dataset:
    """Acc_ICKU: in-context knowledge utilization on E_unseen with distractor context (§2.3)."""
    n = len(u_raw)
    k = min(sample_n, n)
    idxs = random.sample(range(n), k)
    rows = []
    for i in idxs:
        others = _sample_others(n, i, num_others)
        ctxs = [u_raw[i]["test_corpus"]] + [u_raw[j]["test_corpus"] for j in others]
        random.shuffle(ctxs)
        mixed_ctx = sep.join(ctxs)
        for attr in ATTR_KEYS_ALL:
            pair = _safe_get_probe(u_raw[i], attr)
            if pair is None:
                continue
            prompt, tgt = pair
            rows.append(_make_row(tok, "icku", attr, " " + mixed_ctx + " " + prompt, tgt))
    return Dataset.from_list(rows)

def build_examples_conflict(o_raw: List[dict], p_raw: List[dict], tok,
                            sample_n: int) -> Dataset:
    """Pref_PK / Pref_ICK: knowledge conflict resolution (§2.3)."""
    n = min(len(o_raw), len(p_raw))
    k = min(sample_n, n)
    idxs = random.sample(range(n), k)
    rows = []
    for i in idxs:
        o, p = o_raw[i], p_raw[i]
        ctx_p = p["test_corpus"]
        for attr in ATTR_KEYS_PERT:
            pair_o = _safe_get_probe(o, attr)
            pair_p = _safe_get_probe(p, attr)
            if pair_o is None or pair_p is None:
                continue
            prompt_o, tgt_o = pair_o
            prompt_p, tgt_p = pair_p
            # Pref_PK: model answers with original (parametric) value
            rows.append(_make_row(tok, "pref_pk", attr, " " + ctx_p + " " + prompt_o, tgt_o))
            # Pref_ICK: model answers with perturbed (in-context) value
            rows.append(_make_row(tok, "pref_ick", attr, " " + ctx_p + " " + prompt_p, tgt_p))
    return Dataset.from_list(rows)

# ---------- evaluation ----------
@torch.inference_mode()
def compute_metrics_for_example(model, tok, ex, device: str) -> Tuple[Optional[int], Optional[int]]:
    max_len = getattr(model.config, "n_ctx", getattr(model.config, "n_positions", 512))
    ids = tok(ex["text"], return_tensors="pt",
              truncation=True, max_length=max_len).input_ids.to(device)
    logits = model(ids).logits.squeeze(0)

    seq_len = ids.shape[1]
    pl = int(ex["prompt_len"])
    tl = int(ex["target_len"])

    acc1 = None
    em   = None

    if pl < seq_len and pl - 1 >= 0:
        pred1 = torch.argmax(logits[pl - 1]).item()
        true1 = ids[0, pl].item()
        acc1 = int(pred1 == true1)

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

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return _finalize_scores(buckets)

# ---------- main ----------
tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

log.info("Loading data...")
train_raw = load_json(TRAIN_JSON)
o_raw, p_raw = split_train_into_o_p_lists(train_raw, TRAIN_PERT_JSON)

unseen_raw = load_json(UNSEEN_JSON)
if isinstance(unseen_raw, dict):
    if "data" in unseen_raw:
        unseen_raw = unseen_raw["data"]
    else:
        raise ValueError("Unseen JSON is a dict. Expected a list.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

log.info("Building evaluation samples...")
eval_pku      = build_examples_pku(o_raw, tok, N_SAMPLES)
eval_icku     = build_examples_icku(unseen_raw, tok, N_SAMPLES)
eval_conflict = build_examples_conflict(o_raw, p_raw, tok, N_SAMPLES)

eval_all = Dataset.from_list(
    [ex for ex in eval_pku] +
    [ex for ex in eval_icku] +
    [ex for ex in eval_conflict]
)

run_dirs = list_run_dirs(MODEL_ROOT)
if FILTER_RUN_NAME_SUBSTR is not None:
    run_dirs = [p for p in run_dirs if FILTER_RUN_NAME_SUBSTR in p.name]
if not run_dirs:
    log.warning(f"No run directories with checkpoints found under {MODEL_ROOT}.")

all_records = []
for run_dir in run_dirs:
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

if all_records:
    df = pd.DataFrame(all_records).sort_values(["model", "step"])
    metric_cols = sorted([c for c in df.columns if "/" in c])
    df = df[["model", "step"] + metric_cols]

    out_path = Path("probe_results.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"Results saved: {out_path.resolve()}")
    print(df.head(20))
else:
    log.warning("No results collected.")
