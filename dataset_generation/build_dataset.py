"""Build structured biography dataset with train/unseen/pert splits.

Converts raw profiles (from generate_profiles.py) into the bioS format
used for training and evaluation (§2.1).  Each profile gets:
  - train_corpora : 5 paragraphs (each = 4 shuffled attribute sentences)
  - test_corpus   : 1 paragraph  (evaluation in-context paragraph)
  - probes        : per-attribute [prefix, target] pairs (7th template)

Splits:
  - bioS_train   : E_train  (|E_train| = 50k, for training + PKU/conflict eval)
  - bioS_unseen : E_unseen (|E_unseen| = 50k, for ICKU evaluation)
  - bioS_pert    : perturbed version of train (for knowledge conflict eval)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Any

from sentence_templates import (
    birth_date_templates,
    birth_city_templates,
    major_templates,
    university_templates,
)

# Attribute keys in a fixed canonical order.
ATTR_KEYS = ["birth_date", "birth_city", "university", "major"]

# Attributes to perturb in the pert split.
PERT_ATTR_KEYS = ["birth_date", "university"]

# Map from attribute key -> template list.
TEMPLATES = {
    "birth_date": birth_date_templates,
    "birth_city": birth_city_templates,
    "university": university_templates,
    "major": major_templates,
}


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def sample_7_templates(
    template_list: list[str], rng: random.Random
) -> dict[str, Any]:
    """Sample 7 distinct templates from a list of 20.

    Returns {"train": [5 templates], "test": 1 template, "probe": 1 template}.
    """
    chosen = rng.sample(template_list, 7)
    return {
        "train": chosen[:5],
        "test": chosen[5],
        "probe": chosen[6],
    }


def fill_template(template_str: str, profile: dict) -> str:
    """Fill all placeholders in a template with profile values.

    Handles the mapping from profile key 'name' -> template placeholder '{person}'.
    """
    result = template_str
    result = result.replace("{person}", profile["name"])
    for key in ATTR_KEYS:
        result = result.replace("{" + key + "}", profile[key])
    return result


def make_probe_parts(
    template_str: str, profile: dict, attr_key: str
) -> list[str]:
    """Create a [prefix, target] probe pair from a template.

    - Fills all placeholders except attr_key.
    - Splits at the attr_key placeholder position.
    - prefix = text before the placeholder (stripped).
    - target = " " + attr_value + text after the placeholder.
    """
    # Fill everything except the target attribute
    partially_filled = template_str
    partially_filled = partially_filled.replace("{person}", profile["name"])
    for key in ATTR_KEYS:
        if key != attr_key:
            partially_filled = partially_filled.replace(
                "{" + key + "}", profile[key]
            )

    placeholder = "{" + attr_key + "}"
    idx = partially_filled.index(placeholder)
    before = partially_filled[:idx].rstrip()
    after = partially_filled[idx + len(placeholder) :]

    target = " " + profile[attr_key] + after
    return [before, target]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def create_dataset(
    profiles: list[dict], rng: random.Random
) -> list[dict]:
    """Convert raw profiles into structured bioS entries.

    For each profile:
      - Sample 7 templates per attribute (5 train, 1 test, 1 probe)
      - Build 5 train paragraphs: each has 4 sentences (one per attr), shuffled
      - Build 1 test paragraph: same structure
      - Build probes: per attribute, a [prefix, target] pair
    """
    dataset = []

    for profile in profiles:
        # Sample templates for each attribute
        attr_templates = {}
        for attr in ATTR_KEYS:
            attr_templates[attr] = sample_7_templates(TEMPLATES[attr], rng)

        # ---- Train corpora (5 paragraphs) ----
        train_corpora = []
        for para_idx in range(5):
            sentences = []
            for attr in ATTR_KEYS:
                tmpl = attr_templates[attr]["train"][para_idx]
                sentences.append(fill_template(tmpl, profile))
            rng.shuffle(sentences)
            train_corpora.append(" ".join(sentences))

        # ---- Test corpus (1 paragraph) ----
        test_sentences = []
        for attr in ATTR_KEYS:
            tmpl = attr_templates[attr]["test"]
            test_sentences.append(fill_template(tmpl, profile))
        rng.shuffle(test_sentences)
        test_corpus = " ".join(test_sentences)

        # ---- Probes ----
        probes = {}
        for attr in ATTR_KEYS:
            tmpl = attr_templates[attr]["probe"]
            probes[attr] = make_probe_parts(tmpl, profile, attr)

        entry = {
            "name": profile["name"],
            "birth_date": profile["birth_date"],
            "birth_city": profile["birth_city"],
            "university": profile["university"],
            "major": profile["major"],
            "train_corpora": train_corpora,
            "test_corpus": test_corpus,
            "probes": probes,
        }
        dataset.append(entry)

    return dataset


def build_perturbation(
    dataset: list[dict],
    all_profiles: list[dict],
    pert_attrs: list[str] | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """Create a perturbed version of the dataset.

    For each entry, mutate test_corpus and probes:
      - Replace birth_date and university values with random other values.
      - Sync probes so target values reflect the mutations.

    Parameters
    ----------
    dataset : list[dict]
        The structured dataset entries (e.g., bioS_train).
    all_profiles : list[dict]
        The full raw profiles list (to draw replacement values from).
    pert_attrs : list[str]
        Attributes to perturb (default: ["birth_date", "university"]).
    rng : random.Random
        Random number generator.
    """
    if pert_attrs is None:
        pert_attrs = PERT_ATTR_KEYS
    if rng is None:
        rng = random.Random()

    # Build pools of unique values for each perturbed attribute
    attr_pools = {attr: list({p[attr] for p in all_profiles}) for attr in pert_attrs}

    pert_dataset = []

    for entry in dataset:
        pert_entry = copy.deepcopy(entry)

        # For each perturbed attribute, pick a different value
        mutations: dict[str, str] = {}
        for attr in pert_attrs:
            original = entry[attr]
            candidates = [v for v in attr_pools[attr] if v != original]
            if candidates:
                new_val = rng.choice(candidates)
            else:
                new_val = original  # no alternative available
            mutations[attr] = new_val

        # Mutate test_corpus: replace original values with new values
        mutated_corpus = pert_entry["test_corpus"]
        for attr in pert_attrs:
            mutated_corpus = mutated_corpus.replace(
                entry[attr], mutations[attr]
            )
        pert_entry["test_corpus"] = mutated_corpus

        # Sync probes: update target values for perturbed attributes
        for attr in pert_attrs:
            prefix, old_target = pert_entry["probes"][attr]
            # old_target is " " + old_value + trailing_text
            new_target = old_target.replace(entry[attr], mutations[attr])
            pert_entry["probes"][attr] = [prefix, new_target]

        # Store the mutations for reference
        pert_entry["mutations"] = mutations

        pert_dataset.append(pert_entry)

    return pert_dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build bioS dataset from raw profiles."
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="data/profiles.json",
        help="Path to raw profiles JSON (default: data/profiles.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Output directory for bioS JSON files (default: data/)",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=50000,
        help="Number of profiles for the train split (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ---- Load profiles ----
    print(f"Loading profiles from {args.profiles} ...")
    with open(args.profiles, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    print(f"  Loaded {len(profiles):,} profiles.")

    n_total_needed = args.n_train * 2
    if len(profiles) < n_total_needed:
        raise ValueError(
            f"Need at least {n_total_needed:,} profiles "
            f"(n_train * 2 for train + unknown), "
            f"but only {len(profiles):,} available."
        )

    # ---- Split into train / unknown raw profiles ----
    train_profiles = profiles[: args.n_train]
    unknown_profiles = profiles[args.n_train : args.n_train * 2]

    # ---- Build structured datasets ----
    print(f"Building train dataset ({len(train_profiles):,} profiles) ...")
    train_dataset = create_dataset(train_profiles, rng)

    print(f"Building unknown dataset ({len(unknown_profiles):,} profiles) ...")
    unknown_dataset = create_dataset(unknown_profiles, rng)

    print(f"Building perturbation dataset ({len(train_dataset):,} profiles) ...")
    pert_dataset = build_perturbation(train_dataset, profiles, rng=rng)

    # ---- Write output ----
    os.makedirs(args.output_dir, exist_ok=True)

    def write_json(data: list, filename: str):
        path = os.path.join(args.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Wrote {len(data):,} entries -> {path} ({size_mb:.1f} MB)")

    write_json(train_dataset, "bioS_train.json")
    write_json(unknown_dataset, "bioS_unseen.json")
    write_json(pert_dataset, "bioS_pert.json")

    print("Done.")


if __name__ == "__main__":
    main()
