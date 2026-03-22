import argparse
import math
from pathlib import Path

import numpy as np

from two_stg_clust import ATTRIBUTE_FEATURES, TwoStageClust


def parse_args():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Flash-KMeans on selected seismic features.")
    parser.add_argument("--data-folder", default=str(root_dir / "data"))
    parser.add_argument("--output-dir", default=str(root_dir / "results" / "flash_kmeans"))
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
    parser.add_argument("--feature-files", nargs="+", default=None, choices=ATTRIBUTE_FEATURES)
    parser.add_argument("--n-clusters", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-pad-to-power2", action="store_true")
    return parser.parse_args()


def _feature_tag(feature_files: list[str] | None, attrib_config: str) -> str:
    if not feature_files:
        return attrib_config
    stems = [Path(name).stem for name in feature_files]
    return "feat_" + "__".join(stems)


def build_feature_matrix(
    data_folder: Path,
    attrib_config: str,
    use_spatial: str,
    feature_files: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    loader = TwoStageClust(data_folder=str(data_folder))
    loader.load_features(
        use_spatial=use_spatial,
        attrib_config=attrib_config,
        weights=None,
        feature_files=feature_files,
    )
    return (
        loader.features.astype(np.float32, copy=False),
        loader.take_features(use_spatial, attrib_config, feature_files=feature_files),
    )


def maybe_subsample(features: np.ndarray, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_rows = features.shape[0]
    if sample_size <= 0 or sample_size >= n_rows:
        return features, np.arange(n_rows, dtype=np.int64)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
    return features[idx], idx


def pad_feature_dim_to_power_of_two(features: np.ndarray) -> tuple[np.ndarray, int]:
    n_features = features.shape[1]
    target_dim = max(16, 1 << math.ceil(math.log2(max(n_features, 1))))
    if target_dim == n_features:
        return features, n_features

    padded = np.zeros((features.shape[0], target_dim), dtype=features.dtype)
    padded[:, :n_features] = features
    return padded, n_features


def main():
    args = parse_args()
    data_folder = Path(args.data_folder).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for Flash-KMeans.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Flash-KMeans requires CUDA. Current machine does not expose a CUDA device. "
            "Run this script on Linux/Windows with an NVIDIA GPU."
        )

    try:
        from flash_kmeans import batch_kmeans_Euclid
    except Exception as exc:
        raise RuntimeError(
            "flash-kmeans is not installed or failed to import. Install it with: pip install flash-kmeans"
        ) from exc

    features, feature_names = build_feature_matrix(
        data_folder=data_folder,
        attrib_config=args.attrib_config,
        use_spatial=args.use_spatial,
        feature_files=args.feature_files,
    )
    features, selected_idx = maybe_subsample(features, args.sample_size, args.seed)

    original_feature_dim = features.shape[1]
    if not args.no_pad_to_power2:
        features, _ = pad_feature_dim_to_power_of_two(features)
        if features.shape[1] != original_feature_dim:
            print(
                f"Padded feature dimension from {original_feature_dim} to {features.shape[1]} "
                "for Triton power-of-two requirement."
            )

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    x = torch.from_numpy(features).to(device="cuda", dtype=torch_dtype).unsqueeze(0)

    cluster_ids, centers, _ = batch_kmeans_Euclid(
        x,
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=args.verbose,
    )

    cluster_ids_np = cluster_ids.squeeze(0).detach().cpu().numpy()
    centers_np = centers.squeeze(0).detach().cpu().numpy()

    run_name = (
        f"flashkmeans_k{args.n_clusters}_"
        f"{_feature_tag(args.feature_files, args.attrib_config)}_{args.use_spatial}_n{features.shape[0]}"
    )
    np.save(output_dir / f"{run_name}_labels.npy", cluster_ids_np)
    np.save(output_dir / f"{run_name}_centers.npy", centers_np)
    np.save(output_dir / f"{run_name}_row_index.npy", selected_idx)

    meta = {
        "run_name": run_name,
        "n_rows": int(features.shape[0]),
        "n_features": int(original_feature_dim),
        "n_features_padded": int(features.shape[1]),
        "n_clusters": int(args.n_clusters),
        "max_iter": int(args.max_iter),
        "feature_names": feature_names,
        "feature_files": args.feature_files,
        "use_spatial": args.use_spatial,
        "attrib_config": args.attrib_config,
        "dtype": args.dtype,
    }
    (output_dir / f"{run_name}_meta.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in meta.items()),
        encoding="utf-8",
    )

    print(f"Saved labels to {output_dir / f'{run_name}_labels.npy'}")
    print(f"Saved centers to {output_dir / f'{run_name}_centers.npy'}")
    print(f"Saved sampled row index to {output_dir / f'{run_name}_row_index.npy'}")


if __name__ == "__main__":
    main()
