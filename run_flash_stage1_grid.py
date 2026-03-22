import argparse
import json
import subprocess
from pathlib import Path


def parse_args():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Flash-KMeans stage-1 for a grid of feature combinations and k values.")
    parser.add_argument("--combo-file", required=True, help="JSON file with feature combinations.")
    parser.add_argument("--k-values", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--use-spatial", default="only_twt", choices=["all", "only_twt", "none"])
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--script-path", default=str(root_dir / "flash_kmeans_script.py"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    combo_file = Path(args.combo_file).resolve()
    combinations = json.loads(combo_file.read_text(encoding="utf-8"))
    if not isinstance(combinations, list):
        raise ValueError("--combo-file must contain a JSON list of combinations.")

    for k in args.k_values:
        if k < 1 or (k & (k - 1)) != 0:
            raise ValueError(f"k must be a power of two, got {k}")

    for k in args.k_values:
        for combo in combinations:
            if not isinstance(combo, list) or not combo:
                raise ValueError(f"Invalid feature combination: {combo}")
            cmd = [
                args.python_bin,
                str(Path(args.script_path).resolve()),
                "--feature-files",
                *combo,
                "--use-spatial",
                args.use_spatial,
                "--n-clusters",
                str(k),
                "--sample-size",
                str(args.sample_size),
                "--dtype",
                args.dtype,
                "--tol",
                str(args.tol),
            ]
            if args.verbose:
                cmd.append("--verbose")
            print("RUN:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
