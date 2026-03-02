"""Generate synthetic person profiles from seed data files.

Replaces profile generation logic from 241009_dataset_generation.ipynb.
Generates unique 3-part names ("First Middle Last") and assigns random
attributes (birth_date, birth_city, university, major) to each.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys


MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def load_lines(filepath: str) -> list[str]:
    """Load a text file and return non-empty stripped lines."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def generate_birth_date(rng: random.Random) -> str:
    """Generate a random birth date string like 'August 25, 1963'."""
    month = rng.choice(MONTHS)
    day = rng.randint(1, 28)
    year = rng.randint(1900, 2099)
    return f"{month} {day}, {year}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic person profiles from seed data."
    )
    parser.add_argument(
        "--seed_data_dir",
        type=str,
        default="dataset_generation/seed_data",
        help="Directory containing seed data text files (default: dataset_generation/seed_data)",
    )
    parser.add_argument(
        "--n_profiles",
        type=int,
        default=200000,
        help="Number of profiles to generate (default: 200000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/profiles.json",
        help="Output JSON file path (default: data/profiles.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # --- Load seed data ---
    seed_dir = args.seed_data_dir
    print(f"Loading seed data from {seed_dir}/ ...")

    first_names_all = load_lines(os.path.join(seed_dir, "first_names.txt"))
    surnames_all = load_lines(os.path.join(seed_dir, "surnames.txt"))
    cities = load_lines(os.path.join(seed_dir, "cities.txt"))
    universities = load_lines(os.path.join(seed_dir, "universities.txt"))
    majors = load_lines(os.path.join(seed_dir, "majors.txt"))

    print(
        f"  first_names: {len(first_names_all)}, surnames: {len(surnames_all)}, "
        f"cities: {len(cities)}, universities: {len(universities)}, majors: {len(majors)}"
    )

    # --- Sample name pools (matching original notebook logic) ---
    # 800 first names sampled, split into 400 first + 400 middle; 1000 surnames
    n_first_sample = 800
    n_surname_sample = 1000

    if len(first_names_all) < n_first_sample:
        print(
            f"Warning: only {len(first_names_all)} first names available "
            f"(need {n_first_sample}). Using all of them.",
            file=sys.stderr,
        )
        first_name_sample = list(first_names_all)
        rng.shuffle(first_name_sample)
    else:
        first_name_sample = rng.sample(first_names_all, n_first_sample)

    if len(surnames_all) < n_surname_sample:
        print(
            f"Warning: only {len(surnames_all)} surnames available "
            f"(need {n_surname_sample}). Using all of them.",
            file=sys.stderr,
        )
        surname_pool = list(surnames_all)
        rng.shuffle(surname_pool)
    else:
        surname_pool = rng.sample(surnames_all, n_surname_sample)

    first_name_pool = first_name_sample[: n_first_sample // 2]  # 400
    middle_name_pool = first_name_sample[n_first_sample // 2 :]  # 400

    max_unique = len(first_name_pool) * len(middle_name_pool) * len(surname_pool)
    print(
        f"  Name pools: {len(first_name_pool)} first x {len(middle_name_pool)} middle "
        f"x {len(surname_pool)} surname = {max_unique:,} max unique names"
    )
    if args.n_profiles > max_unique:
        print(
            f"Error: requested {args.n_profiles:,} profiles but only {max_unique:,} "
            f"unique names are possible.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Generate unique names ---
    print(f"Generating {args.n_profiles:,} unique names ...")
    generated_names: set[str] = set()
    while len(generated_names) < args.n_profiles:
        first = rng.choice(first_name_pool)
        middle = rng.choice(middle_name_pool)
        last = rng.choice(surname_pool)
        generated_names.add(f"{first} {middle} {last}")

        # Progress reporting every 50k
        if len(generated_names) % 50000 == 0:
            print(f"  ... {len(generated_names):,} unique names so far")

    names_list = list(generated_names)
    print(f"  Generated {len(names_list):,} unique names.")

    # --- Build profiles ---
    print("Assigning attributes to profiles ...")
    profiles = []
    for i, name in enumerate(names_list):
        profile = {
            "name": name,
            "birth_date": generate_birth_date(rng),
            "birth_city": rng.choice(cities),
            "university": rng.choice(universities),
            "major": rng.choice(majors),
        }
        profiles.append(profile)

        if (i + 1) % 50000 == 0:
            print(f"  ... {i + 1:,} / {args.n_profiles:,} profiles created")

    # --- Write output ---
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved {len(profiles):,} profiles to {args.output} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
