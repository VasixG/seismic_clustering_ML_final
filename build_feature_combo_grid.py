import argparse
import itertools
import json
from pathlib import Path

from tqdm.auto import tqdm

from two_stg_clust import ATTRIBUTE_FEATURES


def parse_args():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build a JSON grid of feature combinations.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=ATTRIBUTE_FEATURES,
        choices=ATTRIBUTE_FEATURES,
        help="Feature files to combine.",
    )
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--max-size", type=int, default=7)
    parser.add_argument("--output-path", default=str(root_dir / "feature_combos.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_size < 1:
        raise ValueError("--min-size must be >= 1")
    if args.max_size < args.min_size:
        raise ValueError("--max-size must be >= --min-size")

    features = list(dict.fromkeys(args.features))
    combinations = []
    for size in tqdm(
        range(args.min_size, min(args.max_size, len(features)) + 1),
        desc="Building feature combinations",
        unit="size",
    ):
        combinations.extend([list(combo) for combo in itertools.combinations(features, size)])

    output_path = Path(args.output_path).resolve()
    output_path.write_text(json.dumps(combinations, indent=2), encoding="utf-8")
    print(f"Saved {len(combinations)} feature combinations to {output_path}")


if __name__ == "__main__":
    main()
