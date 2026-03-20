import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from two_stg_clust import TwoStageClust


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def get_feature_list(attrib_config: str, use_spatial: str) -> list[str]:
    helper = TwoStageClust(data_folder=".")
    return helper.take_features(use_spatial=use_spatial, attrib_config=attrib_config)


def inspect_dataset(data_dir: Path, attrib_config: str, use_spatial: str) -> dict:
    feature_names = get_feature_list(attrib_config, use_spatial)
    arrays = {}
    stats = []
    n_rows = None
    feature_bytes = 0

    for name in feature_names:
        arr = np.load(data_dir / name, mmap_mode="r")
        arrays[name] = arr
        feature_bytes += arr.nbytes
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError(f"Inconsistent lengths: {name} has {arr.shape[0]}, expected {n_rows}")
        stats.append(
            {
                "file": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "bytes": int(arr.nbytes),
                "human_size": format_bytes(int(arr.nbytes)),
            }
        )

    coord_files = ["iline.npy", "xline.npy", "twt.npy", "cdp_x.npy", "cdp_y.npy"]
    for name in coord_files:
        path = data_dir / name
        if path.exists() and name not in arrays:
            arr = np.load(path, mmap_mode="r")
            stats.append(
                {
                    "file": name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "bytes": int(arr.nbytes),
                    "human_size": format_bytes(int(arr.nbytes)),
                }
            )

    feature_matrix_bytes = n_rows * len(feature_names) * np.dtype(np.float32).itemsize
    return {
        "n_rows": int(n_rows),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "source_feature_bytes": int(feature_bytes),
        "source_feature_human": format_bytes(int(feature_bytes)),
        "feature_matrix_bytes_float32": int(feature_matrix_bytes),
        "feature_matrix_human_float32": format_bytes(int(feature_matrix_bytes)),
        "files": stats,
    }


def load_sample_matrix(
    data_dir: Path,
    attrib_config: str,
    use_spatial: str,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    feature_names = get_feature_list(attrib_config, use_spatial)
    first = np.load(data_dir / feature_names[0], mmap_mode="r")
    n_rows = first.shape[0]
    sample_size = min(sample_size, n_rows)
    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(n_rows, size=sample_size, replace=False))

    columns = []
    for name in feature_names:
        arr = np.load(data_dir / name, mmap_mode="r")[sample_idx].astype(np.float32, copy=False)
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        columns.append(arr[:, None])

    return np.concatenate(columns, axis=1)


def benchmark_gmm(
    sample_matrix: np.ndarray,
    components: list[int],
    repeats: int,
    random_state: int,
) -> list[dict]:
    results = []
    n_rows = sample_matrix.shape[0]
    n_features = sample_matrix.shape[1]

    for n_comp in components:
        times = []
        for repeat_idx in range(repeats):
            model = GaussianMixture(
                n_components=n_comp,
                random_state=random_state + repeat_idx,
                covariance_type="full",
            )
            start = time.perf_counter()
            model.fit_predict(sample_matrix)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        sec_per_sample = sum(times) / len(times) / n_rows
        sec_per_sample_feature_comp = sec_per_sample / max(n_features * n_comp, 1)
        results.append(
            {
                "method": f"gmm{n_comp}",
                "components": n_comp,
                "sample_rows": n_rows,
                "sample_features": n_features,
                "runs_sec": times,
                "mean_sec": float(np.mean(times)),
                "median_sec": float(np.median(times)),
                "sec_per_sample": sec_per_sample,
                "sec_per_sample_feature_component": sec_per_sample_feature_comp,
            }
        )

    return results


def extrapolate_runtime(benchmarks: list[dict], full_rows: int, full_features: int) -> list[dict]:
    estimates = []
    for row in benchmarks:
        est_seconds = row["sec_per_sample_feature_component"] * full_rows * full_features * row["components"]
        estimates.append(
            {
                "method": row["method"],
                "estimated_seconds": est_seconds,
                "estimated_minutes": est_seconds / 60.0,
                "estimated_hours": est_seconds / 3600.0,
                "note": "Rough linear extrapolation from sampled fit time. Real full-run time may differ due to memory pressure and EM convergence.",
            }
        )
    return estimates


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inspect dataset size and estimate GMM runtime.")
    parser.add_argument("--data-dir", default=str(root / "data"))
    parser.add_argument(
        "--attrib-config",
        default="only_geom",
        choices=["only_spectr", "no_spectr", "all", "only_geom"],
    )
    parser.add_argument(
        "--use-spatial",
        default="only_twt",
        choices=["all", "only_twt", "none"],
    )
    parser.add_argument("--sample-size", type=int, default=200000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--components", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable text.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    dataset = inspect_dataset(data_dir, args.attrib_config, args.use_spatial)
    sample_matrix = load_sample_matrix(
        data_dir=data_dir,
        attrib_config=args.attrib_config,
        use_spatial=args.use_spatial,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    benchmarks = benchmark_gmm(
        sample_matrix=sample_matrix,
        components=args.components,
        repeats=args.repeats,
        random_state=args.seed,
    )
    estimates = extrapolate_runtime(
        benchmarks=benchmarks,
        full_rows=dataset["n_rows"],
        full_features=dataset["n_features"],
    )

    payload = {
        "dataset": dataset,
        "benchmark_sample_shape": list(sample_matrix.shape),
        "benchmarks": benchmarks,
        "runtime_estimates": estimates,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("Dataset summary")
    print(f"Rows: {dataset['n_rows']:,}")
    print(f"Features used: {dataset['n_features']}")
    print(f"Feature list: {', '.join(dataset['feature_names'])}")
    print(f"Raw selected feature files: {dataset['source_feature_human']}")
    print(f"Dense float32 feature matrix in memory: {dataset['feature_matrix_human_float32']}")
    print()
    print("Benchmark sample")
    print(f"Sample shape: {sample_matrix.shape[0]:,} x {sample_matrix.shape[1]}")
    print()
    print("GMM timings")
    for row in benchmarks:
        print(
            f"{row['method']}: mean={row['mean_sec']:.2f}s, median={row['median_sec']:.2f}s "
            f"on {row['sample_rows']:,} rows"
        )
    print()
    print("Estimated full runtime")
    for row in estimates:
        print(
            f"{row['method']}: ~{row['estimated_minutes']:.1f} min "
            f"({row['estimated_hours']:.2f} h)"
        )


if __name__ == "__main__":
    main()
