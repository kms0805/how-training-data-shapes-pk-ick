import argparse
import os, json, random
from math import ceil
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
from transformers import GPT2TokenizerFast
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# ==== Argparse configuration ====
def parse_args():
    parser = argparse.ArgumentParser(description="Training corpus generator with controlled repetition, noise, and frequency distribution")

    parser.add_argument("--profiles_json", type=str, default="data/bioS_train.json",
                        help="Path to profiles JSON")
    parser.add_argument("--out_dir", type=str, default="data/corpora",
                        help="Output directory")
    parser.add_argument("--out_json", type=str, default="corpus.json",
                        help="Output JSON filename")
    parser.add_argument("--mode", type=str, choices=["single", "repeated"],
                        default="repeated",
                        help="Corpus mode: 'single' = one paragraph per entity (§3.1 SINGLE), "
                             "'repeated' = two paraphrased paragraphs per entity, "
                             "mixed with other entities (§3.1 REPEATED).")
    parser.add_argument("--zipf_alpha", type=float, default=1.0,
                        help="Zipf distribution α parameter (§3.3). 0 = uniform.")
    parser.add_argument("--noise_prob", type=float, default=0.01,
                        help="Within-document inconsistency probability (§3.2)")
    parser.add_argument("--sep", type=str, default=" ",
                        help="Sentence separator")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2",
                        help="HuggingFace tokenizer name")

    parser.add_argument("--n_preview", type=int, default=5,
                        help="Number of preview samples")
    parser.add_argument("--target_tokens_mode", type=str, choices=["samples", "training"],
                        default="training", help="Token target calculation mode")
    parser.add_argument("--n_samples_goal", type=int, default=0,
                        help="Target number of samples in 'samples' mode")

    # Training mode parameters
    parser.add_argument("--max_steps", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--do_build_dataset", action="store_true",
                        help="Whether to build the dataset")
    parser.add_argument("--max_samples_hard_cap", type=int, default=0,
                        help="Hard cap on number of samples (0 = unlimited)")

    # Performance parameters
    parser.add_argument("--chunk_size_for_tokenize", type=int, default=512,
                        help="Number of texts to tokenize per batch")
    parser.add_argument("--num_tokenizer_workers", type=int, default=min(8, (os.cpu_count() or 4)),
                        help="Number of tokenizer workers")
    parser.add_argument("--max_inflight", type=int, default=None,
                        help="Max number of in-flight task batches")
    parser.add_argument("--log_every_samples", type=int, default=10000,
                        help="Progress log interval (in samples)")

    # preprocessing
    parser.add_argument("--preprocess", action="store_true", default=True,
                        help="Prepend a leading space to each text (preprocessing, default)")
    parser.add_argument("--no-preprocess", dest="preprocess", action="store_false",
                        help="Disable leading space preprocessing")

    return parser.parse_args()

# Usage in main
if __name__ == "__main__":
    args = parse_args()

    PROFILES_JSON = args.profiles_json
    OUT_DIR = args.out_dir
    OUT_JSON = args.out_json
    MODE = args.mode
    ZIPF_ALPHA = args.zipf_alpha
    NOISE_PROB = args.noise_prob
    SEP = args.sep
    SEED = args.seed
    TOKENIZER_NAME = args.tokenizer_name

    N_PREVIEW = args.n_preview
    TARGET_TOKENS_MODE = args.target_tokens_mode
    N_SAMPLES_GOAL = args.n_samples_goal

    MAX_STEPS = args.max_steps
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM = args.grad_accum
    WORLD_SIZE = args.world_size
    MAX_SEQ_LEN = args.max_seq_len

    DO_BUILD_DATASET = args.do_build_dataset
    MAX_SAMPLES_HARD_CAP = args.max_samples_hard_cap

    CHUNK_SIZE_FOR_TOKENIZE = args.chunk_size_for_tokenize
    NUM_TOKENIZER_WORKERS = args.num_tokenizer_workers
    MAX_INFLIGHT = args.max_inflight
    LOG_EVERY_SAMPLES = args.log_every_samples
    PREPROCESS = args.preprocess

    # Run the rest of the pipeline
    # ==== Utilities ====
    ATTR_KEYS = ["birth_date", "birth_city", "university", "major"]

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def load_profiles(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_attr_pools(profiles):
        pools = {k: [] for k in ATTR_KEYS}
        for p in profiles:
            for k in ATTR_KEYS:
                pools[k].append(p[k])
        return pools

    def mutate_paragraph(paragraph, prof_idx, profiles, attr_pools, prob):
        if prob <= 0.0:
            return paragraph
        out = paragraph
        prof = profiles[prof_idx]
        n_profiles = len(profiles)
        for attr in ATTR_KEYS:
            if random.random() >= prob: continue
            old_val = prof[attr]
            if old_val not in out: continue
            for _ in range(50):
                j = random.randrange(n_profiles)
                if j != prof_idx:
                    new_val = attr_pools[attr][j]
                    if new_val != old_val:
                        out = out.replace(old_val, new_val, 1)
                        break
        return out

    class ZipfSampler:
        def __init__(self, N, s=1.1):
            assert N > 0 and s > 0
            ranks = np.arange(1, N + 1, dtype=np.float64)
            w = 1.0 / np.power(ranks, s)
            self.p = w / w.sum()
            self.N = N
        def sample_one(self):
            return int(np.random.choice(self.N, p=self.p))
        def sample_one_excluding(self, exclude: set):
            while True:
                x = int(np.random.choice(self.N, p=self.p))
                if x not in exclude:
                    return x
        def sample_two_partners(self, anchor: int):
            b = self.sample_one_excluding({anchor})
            c = self.sample_one_excluding({anchor, b})
            return [b, c]

    def pick_two_paras_of(p):
        paras = p["train_corpora"]
        if len(paras) >= 2:
            a, b = random.sample(paras, 2)
        else:
            a = b = paras[0]
        return a.strip(), b.strip()

    def sample_single_one(profiles, idx, attr_pools, noise_prob):
        para = random.choice(profiles[idx]["train_corpora"]).strip()
        if noise_prob > 0.0:
            para = mutate_paragraph(para, idx, profiles, attr_pools, prob=noise_prob)
        return para

    def interleaved_two_each_order(n_persons=3):
        labels = []
        for i in range(n_persons):
            labels.extend([i, i])
        random.shuffle(labels)
        return labels

    def sample_repeated_one(profiles, idx_anchor, idx_partners, attr_pools, noise_prob, sep):
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
                out_paras.append(per_first[tag])
                seen[tag] = True
            else:
                out_paras.append(per_second[tag])
        return sep.join(out_paras)

    def stream_write_text_list_header(f):
        f.write("[\n")

    def stream_write_text_list_item(f, text, is_first, preprocess=True):
        if not is_first:
            f.write(",\n")
        if preprocess:
            text = " " + text
        f.write(json.dumps(text, ensure_ascii=False))

    def stream_write_text_list_footer(f):
        f.write("\n]\n")

    def build_until_tokens_streaming_mt(
        profiles, out_dir, out_json, mode, zipf_alpha, noise_prob, sep,
        target_tokens, tokenizer_name, seed,
        max_samples_cap=0, chunk_size=512, num_workers=4, max_inflight=None,
        log_every=10000, preprocess=True
    ):
        """
        Multi-threaded batch tokenization + streaming write:
        - Main thread: generates sample chunks and submits tasks
        - Worker threads: perform batch tokenization (length calculation)
        - Main thread: receives completed batches and writes sequentially to JSON
        """
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_json)

        pools = build_attr_pools(profiles)
        zipf  = ZipfSampler(len(profiles), s=zipf_alpha)
        tok_shared = GPT2TokenizerFast.from_pretrained(tokenizer_name)  # Pre-load (shared by workers)

        if max_inflight is None:
            max_inflight = max(2, num_workers * 2)

        def generate_chunk():
            """Generate a text chunk (single thread)"""
            buf = []
            for _ in range(chunk_size):
                idx_anchor = zipf.sample_one()
                if mode == "single":
                    text = sample_single_one(profiles, idx_anchor, pools, noise_prob)
                else:
                    partners = zipf.sample_two_partners(idx_anchor)
                    text = sample_repeated_one(profiles, idx_anchor, partners, pools, noise_prob, sep)
                buf.append(text)
            return buf

        def tokenize_lengths(texts):
            """Worker function: batch tokenize and compute lengths only"""
            enc = tok_shared(
                texts, add_special_tokens=False, return_attention_mask=False,
                return_length=True, padding=False, truncation=False
            )
            lens = enc.get("length", None)
            if lens is None:
                lens = [len(ids) for ids in enc["input_ids"]]
            return texts, [int(x) for x in lens]

        total_tokens = 0
        n_samples = 0
        first_item = True

        with open(out_path, "w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=num_workers) as ex:

            stream_write_text_list_header(f)
            inflight = set()

            # Pre-fill the task queue
            while len(inflight) < max_inflight:
                buf = generate_chunk()
                fut = ex.submit(tokenize_lengths, buf)
                inflight.add(fut)

            while total_tokens < target_tokens and inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)

                # Process completed batches
                for fut in done:
                    texts, lengths = fut.result()
                    for text, n_tok in zip(texts, lengths):
                        stream_write_text_list_item(f, text, is_first=first_item, preprocess=preprocess)
                        if first_item: first_item = False
                        total_tokens += n_tok
                        n_samples += 1

                        if (n_samples % log_every) == 0:
                            print(f"[progress] samples={n_samples:,}, tokens≈{total_tokens:,}")

                        if (max_samples_cap and n_samples >= max_samples_cap) or (total_tokens >= target_tokens):
                            break

                    if (max_samples_cap and n_samples >= max_samples_cap) or (total_tokens >= target_tokens):
                        break

                # Replenish tasks if target not yet reached
                while (total_tokens < target_tokens) and (len(inflight) < max_inflight):
                    buf = generate_chunk()
                    fut = ex.submit(tokenize_lengths, buf)
                    inflight.add(fut)

            # Remaining tasks are not processed (auto-cancel not supported, but file writing has already stopped)
            stream_write_text_list_footer(f)

        print(f"[OK] Wrote list JSON: {out_path}")
        print(f" - samples: {n_samples:,}, approx_tokens: {total_tokens:,}")

    # ==== Execution ====
    set_seed(SEED)
    profiles = load_profiles(PROFILES_JSON)
    if len(profiles) < 3 and MODE == "repeated":
        raise ValueError("repeated mode requires at least 3 profiles")

    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
    pools = build_attr_pools(profiles)
    zipf = ZipfSampler(len(profiles), s=ZIPF_ALPHA)

    # Preview 5 samples
    print(f"[preview] mode={MODE}, noise_prob={NOISE_PROB}, zipf_alpha={ZIPF_ALPHA}")
    tok_lens = []
    for i in range(N_PREVIEW):
        idx_anchor = zipf.sample_one()
        if MODE == "single":
            text = sample_single_one(profiles, idx_anchor, pools, NOISE_PROB)
        else:
            partners = zipf.sample_two_partners(idx_anchor)
            text = sample_repeated_one(profiles, idx_anchor, partners, pools, NOISE_PROB, SEP)
        n_tok = len(tok.encode(text, add_special_tokens=False))
        tok_lens.append(n_tok)
        print(f"  sample#{i}: tokens={n_tok} | {text[:100]}...")

    avg_tok = float(np.mean(tok_lens))
    print(f"[preview-stats] avg={avg_tok:.1f}, min={min(tok_lens)}, max={max(tok_lens)}")

    # Estimate target_tokens (1.2x margin)
    if TARGET_TOKENS_MODE == "training":
        tokens_needed = MAX_STEPS * BATCH_SIZE * GRAD_ACCUM * WORLD_SIZE * MAX_SEQ_LEN
        print(f"[estimate(training)] steps={MAX_STEPS}, B={BATCH_SIZE}, G={GRAD_ACCUM}, W={WORLD_SIZE}, S={MAX_SEQ_LEN}")
    else:
        tokens_needed = avg_tok * N_SAMPLES_GOAL
        print(f"[estimate(samples)] N_SAMPLES_GOAL={N_SAMPLES_GOAL:,}, avg_tok≈{avg_tok:.1f}")
    target_tokens = int(ceil(tokens_needed * 1.2))
    print(f"[target_tokens] ≈{target_tokens:,}")

    if DO_BUILD_DATASET:
        build_until_tokens_streaming_mt(
            profiles=profiles,
            out_dir=OUT_DIR,
            out_json=OUT_JSON,
            mode=MODE,
            zipf_alpha=ZIPF_ALPHA,
            noise_prob=NOISE_PROB,
            sep=SEP,
            target_tokens=target_tokens,
            tokenizer_name=TOKENIZER_NAME,
            seed=SEED,
            max_samples_cap=MAX_SAMPLES_HARD_CAP,
            chunk_size=CHUNK_SIZE_FOR_TOKENIZE,
            num_workers=NUM_TOKENIZER_WORKERS,
            max_inflight=MAX_INFLIGHT,
            log_every=LOG_EVERY_SAMPLES,
            preprocess=PREPROCESS,
        )
    else:
        print("[build] DO_BUILD_DATASET=False -> skipping dataset generation")