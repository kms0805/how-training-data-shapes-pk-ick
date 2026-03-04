#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8-layer decoder-only Transformer (GPT-2 style) pretraining on pre-generated corpus.

Expects a JSON file produced by generate_corpus.py:
  - Format: ["text1", "text2", ...]  (list of strings)

Uses SFTTrainer with packing for efficient training.
See Table 3 & 4 in the paper for architecture and hyperparameters.
"""

import os, re, json, argparse, random, wandb
from datetime import datetime
from typing import List

import numpy as np
import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from trl import SFTTrainer, SFTConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_texts(path: str) -> List[str]:
    """Load corpus JSON. Supports: ["text1", ...] or [{"text": "..."}, ...]"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict) and "text" in data[0]:
            return [str(x["text"]) for x in data]
        return [str(x) for x in data]
    raise ValueError("Corpus JSON must be a list of strings or dicts with 'text' key.")


def stem(path: str):
    s = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._\\-]+", "_", s) or "dataset"


def main():
    p = argparse.ArgumentParser(description="GPT-2 pretraining on pre-generated corpus.")
    p.add_argument("--data", type=str, required=True, help="Path to corpus JSON (from generate_corpus.py)")
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
        name=f"train:{stem(args.data)}",
        config=vars(args),
    )
    os.makedirs(args.output_root, exist_ok=True)
    set_seed(args.seed)

    # Load corpus (with Arrow cache for fast re-runs)
    cache_dir = os.path.join(os.path.dirname(args.data), ".cache", f"{stem(args.data)}_seed{args.seed}")
    if os.path.exists(cache_dir):
        print(f"Loading cached dataset from {cache_dir} ...")
        train_ds = Dataset.load_from_disk(cache_dir)
    else:
        print(f"Loading corpus from {args.data} ...")
        texts = load_texts(args.data)
        random.shuffle(texts)
        train_ds = Dataset.from_dict({"text": texts})
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        train_ds.save_to_disk(cache_dir)
        print(f"  Saved cache to {cache_dir}")
    print(f"  Loaded {len(train_ds):,} samples.")

    # Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    # Model (8-layer GPT-2)
    cfg = GPT2Config(
        n_embd=512, n_layer=8, n_head=8, n_inner=2048,
        n_positions=args.max_seq_len, n_ctx=args.max_seq_len,
        pad_token_id=tok.pad_token_id, vocab_size=len(tok),
    )
    model = GPT2LMHeadModel(cfg)
    model.resize_token_embeddings(len(tok))

    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    out_dir = os.path.join(args.output_root, f"{stem(args.data)}_{stamp}")
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
        max_length=args.max_seq_len,
        dataset_text_field="text",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tok,
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
