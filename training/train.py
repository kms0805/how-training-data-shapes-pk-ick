#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On-the-fly dataset builder with Zipf sampling (no sentence shuffle)
Modes:
  - base                 : single paragraph from one entity (original paragraph as-is)
  - context(+noise)      : two paragraphs of one entity; first paragraph may be noised
  - multiple-context(+noise): 3 entities × 2 paras; each entity's first occurrence may be noised
Notes:
  - Noise applies only to the first paragraph occurrence per entity.
  - 'name' is excluded from noise (ATTR_KEYS does not contain it).
"""

import os, re, json, argparse, random, wandb
from datetime import datetime
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from trl import SFTTrainer, SFTConfig

# --------------------------
# 1) Constants / defaults
# --------------------------
# 'name' is excluded
ATTR_KEYS = ["birth_date", "birth_city", "university", "major"]
DEFAULT_SEP = " "

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------
# 2) Shared utilities
# --------------------------
def load_profiles(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_attr_pools(profiles: List[Dict]) -> Dict[str, List[str]]:
    pools = {k: [] for k in ATTR_KEYS}
    for p in profiles:
        for k in ATTR_KEYS:
            pools[k].append(p[k])
    return pools

def mutate_paragraph(paragraph: str, prof_idx: int, profiles: List[Dict],
                     attr_pools: Dict[str, List[str]], prob: float) -> str:
    """
    Passage noise: for each attribute in ATTR_KEYS, replace with another person's
    value with probability `prob` (at most once per attribute).
    'name' is not a target (not in ATTR_KEYS).
    """
    if prob <= 0.0:
        return paragraph

    out = paragraph
    prof = profiles[prof_idx]
    n_profiles = len(profiles)

    for attr in ATTR_KEYS:
        if random.random() >= prob:
            continue
        old_val = prof[attr]
        if old_val not in out:
            continue
        for _ in range(50):
            j = random.randrange(n_profiles)
            if j != prof_idx:
                new_val = attr_pools[attr][j]
                if new_val != old_val:
                    out = out.replace(old_val, new_val, 1)
                    break
    return out

# --------------------------
# 3) Zipf sampler
# --------------------------
class ZipfSampler:
    """p(i) proportional to 1/(i+1)^s for i in 0..N-1"""
    def __init__(self, N: int, s: float = 1.1):
        assert N > 0 and s > 0
        ranks = np.arange(1, N + 1, dtype=np.float64)
        w = 1.0 / np.power(ranks, s)
        self.p = w / w.sum()
        self.N = N

    def sample_one(self) -> int:
        return int(np.random.choice(self.N, p=self.p))

    def sample_distinct(self, k: int, exclude: Optional[Iterable[int]] = None) -> List[int]:
        exclude = set(exclude or [])
        mask = np.ones(self.N, dtype=bool)
        if exclude:
            idx = np.fromiter(exclude, dtype=int)
            mask[idx] = False
        valid_idx = np.nonzero(mask)[0]
        if k > len(valid_idx):
            raise ValueError("Not enough candidates after exclusion.")
        p_valid = self.p[valid_idx]
        p_valid = p_valid / p_valid.sum()
        chosen_local = np.random.choice(len(valid_idx), size=k, replace=False, p=p_valid)
        return valid_idx[chosen_local].tolist()

# --------------------------
# 4) Paragraph selection & sample generation (no sentence shuffle)
# --------------------------
def pick_two_paras_of(p: Dict) -> Tuple[str, str]:
    paras = p["train_corpora"]
    if len(paras) >= 2:
        a, b = random.sample(paras, 2)
    else:
        a = b = paras[0]
    # Use original text as-is without shuffling
    return a.strip(), b.strip()

def sample_base_one(profiles: List[Dict], idx: int,
                    attr_pools: Dict[str, List[str]],
                    noise_prob: float) -> str:
    """
    Base mode: single paragraph from one person. No sentence shuffle.
    If noise_prob > 0, noise is applied to that paragraph only.
    """
    paras = profiles[idx]["train_corpora"]
    para = random.choice(paras).strip()
    if noise_prob > 0.0:
        para = mutate_paragraph(para, idx, profiles, attr_pools, prob=noise_prob)
    return para

def sample_context_one(profiles: List[Dict], idx: int,
                       attr_pools: Dict[str, List[str]],
                       noise_prob: float, sep: str) -> str:
    # Two paragraphs: only the first may be noised
    p1, p2 = pick_two_paras_of(profiles[idx])
    if noise_prob > 0.0:
        p1 = mutate_paragraph(p1, idx, profiles, attr_pools, prob=noise_prob)
    return f"{p1}{sep}{p2}"

def interleaved_two_each_order(n_persons: int = 3) -> List[int]:
    labels = []
    for i in range(n_persons):
        labels.extend([i, i])
    random.shuffle(labels)
    return labels

def sample_multiple_context_one(profiles: List[Dict], idx_anchor: int,
                                idx_partners: List[int],
                                attr_pools: Dict[str, List[str]],
                                noise_prob: float, sep: str) -> str:
    """
    3 persons (anchor + 2) x 2 paragraphs, [0,0,1,1,2,2] shuffled.
    Only each person's first-appearing paragraph may be noised; the second is clean.
    """
    assert len(idx_partners) == 2
    triplet = [idx_anchor] + idx_partners
    per_first, per_second = {}, {}

    for local, ref_idx in enumerate(triplet):
        first_p, second_p = pick_two_paras_of(profiles[ref_idx])
        if noise_prob > 0.0:
            first_p = mutate_paragraph(first_p, ref_idx, profiles, attr_pools, prob=noise_prob)
        per_first[local] = first_p
        per_second[local] = second_p

    order = interleaved_two_each_order(3)
    seen = {0: False, 1: False, 2: False}
    out_paras = []
    for tag in order:
        if not seen[tag]:
            out_paras.append(per_first[tag])   # First appearance -> may be noisy
            seen[tag] = True
        else:
            out_paras.append(per_second[tag])  # Second appearance -> clean

    return sep.join(out_paras)

# --------------------------
# 5) IterableDataset
# --------------------------
class DynamicZipfIterableDataset(IterableDataset):
    """
    mode in {base, context, multiple-context}.
    If noise_prob=0.0, no noise is applied.
    """
    def __init__(
        self,
        profiles: List[Dict],
        samples_per_epoch: int,
        mode: str,                       # "base" | "context" | "multiple-context"
        zipf_s: float = 1.1,
        noise_prob: float = 0.01,
        sep: str = DEFAULT_SEP,
        seed: int = 42,
    ):
        super().__init__()
        assert mode in ("base", "context", "multiple-context")
        self.profiles = profiles
        self.N = len(profiles)
        self.samples_per_epoch = int(samples_per_epoch)
        self.mode = mode
        self.noise_prob = float(noise_prob)
        self.sep = sep
        self.seed = int(seed)

        self.attr_pools = build_attr_pools(self.profiles)
        self.zipf = ZipfSampler(self.N, s=zipf_s)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        base_seed = self.seed + (worker_info.id if worker_info else 0)
        random.seed(base_seed + 12345)
        np.random.seed(base_seed + 23456)

        for _ in range(self.samples_per_epoch):
            idx_anchor = self.zipf.sample_one()

            if self.mode == "base":
                text = sample_base_one(
                    self.profiles, idx_anchor, self.attr_pools, self.noise_prob
                )
            elif self.mode == "context":
                text = sample_context_one(
                    self.profiles, idx_anchor, self.attr_pools, self.noise_prob, self.sep
                )
            else:  # "multiple-context"
                partners = self.zipf.sample_distinct(2, exclude={idx_anchor})
                text = sample_multiple_context_one(
                    self.profiles, idx_anchor, partners,
                    self.attr_pools, self.noise_prob, self.sep
                )

            yield {"text": text}

# --------------------------
# 6) Training script
# --------------------------
def stem(path: str):
    s = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._\\-]+", "_", s) or "dataset"

def main():
    p = argparse.ArgumentParser(description="GPT-2 training with on-the-fly Zipf sampling (no sentence shuffle).")
    # Data / sampling
    p.add_argument("--profiles_json", type=str, required=True, help="Path to profiles JSON (bioS_* format)")
    p.add_argument("--samples_per_epoch", type=int, default=100_000, help="Number of samples the IterableDataset yields per epoch")
    p.add_argument("--mode", type=str, required=True,
                   choices=["base", "context", "multiple-context"],
                   help="Training sample generation mode")
    p.add_argument("--zipf_s", type=float, default=0.1, help="Zipf distribution exponent s (> 0)")
    p.add_argument("--noise_prob", type=float, default=0.0, help="Passage noise probability (0 = disabled)")
    p.add_argument("--sep", type=str, default=DEFAULT_SEP, help="Paragraph separator (context/multiple-context)")
    # Training parameters
    p.add_argument("--output_root", type=str, default="./checkpoints")
    p.add_argument("--max_steps", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    wandb.init(
        project="training-data-shapes-pk-ick",
        name=f"dyn_zipf:{os.path.basename(args.profiles_json)}|{args.mode}|s={args.zipf_s}|noise={args.noise_prob}",
        config=vars(args)
    )
    os.makedirs(args.output_root, exist_ok=True)
    set_seed(args.seed)

    profiles = load_profiles(args.profiles_json)
    if len(profiles) < 3 and args.mode == "multiple-context":
        raise ValueError("multiple-context mode requires at least 3 profiles.")

    train_ds = DynamicZipfIterableDataset(
        profiles=profiles,
        samples_per_epoch=args.samples_per_epoch,
        mode=args.mode,
        zipf_s=args.zipf_s,
        noise_prob=args.noise_prob,
        sep=args.sep,
        seed=args.seed,
    )

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    cfg = GPT2Config(
        n_embd=512, n_layer=8, n_head=8, n_inner=2048,
        n_positions=args.max_seq_len, n_ctx=args.max_seq_len,
        pad_token_id=tok.pad_token_id, vocab_size=len(tok),
    )
    model = GPT2LMHeadModel(cfg)
    model.resize_token_embeddings(len(tok))

    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    out_dir = os.path.join(
        args.output_root,
        f"dyn_{stem(args.profiles_json)}_{args.mode}_zipf{args.zipf_s}_noise{args.noise_prob}_{stamp}"
    )
    os.makedirs(out_dir, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=out_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="no",
        report_to=["wandb"],
        packing=True,
        max_seq_length=args.max_seq_len,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_text_field="text",   # Explicitly set since IterableDataset yields {"text": ...}
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        tokenizer=tok,
        data_collator=None,
    )

    print("Start training...")
    trainer.train()
    print("Training done.")

    final_dir = os.path.join(out_dir, "_final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    print(f"Saved model to: {final_dir}")

    wandb.finish()

if __name__ == "__main__":
    main()
