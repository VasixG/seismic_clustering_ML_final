import argparse
import json
from pathlib import Path

import numpy as np

from src.point_cloud_visualization import save_point_cloud_sections
from torque_clustering.admm_cns_torque_module import admm_cns_torque


def parse_args():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run torque clustering on Flash-KMeans centers and project snapshots back to the full point cloud."
    )
    parser.add_argument("--data-folder", default=str(root_dir / "data"))
    parser.add_argument("--flash-dir", default=str(root_dir / "results" / "flash_kmeans"))
    parser.add_argument("--flash-run-name", required=True)
    parser.add_argument("--output-dir", default=str(root_dir / "results" / "flash_torque"))
    parser.add_argument("--torque-k", type=int, required=True, help="Target number of torque clusters.")
    parser.add_argument("--n-neighbors", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--rho1", type=float, default=1.0)
    parser.add_argument("--rho2", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-every", type=int, default=2)
    return parser.parse_args()


def _trim_zero_padding(centers: np.ndarray) -> tuple[np.ndarray, int]:
    nonzero_cols = np.where(np.any(np.abs(centers) > 1e-12, axis=0))[0]
    if len(nonzero_cols) == 0:
        return centers, centers.shape[1]
    last_idx = int(nonzero_cols[-1]) + 1
    trimmed = centers[:, :last_idx]
    return trimmed, last_idx


def _load_stage1_outputs(data_folder: Path, flash_dir: Path, run_name: str):
    labels_path = flash_dir / f"{run_name}_labels.npy"
    centers_path = flash_dir / f"{run_name}_centers.npy"
    row_index_path = flash_dir / f"{run_name}_row_index.npy"

    if not labels_path.exists():
        raise FileNotFoundError(f"Stage-1 labels not found: {labels_path}")
    if not centers_path.exists():
        raise FileNotFoundError(f"Stage-1 centers not found: {centers_path}")
    if not row_index_path.exists():
        raise FileNotFoundError(f"Stage-1 row index not found: {row_index_path}")

    stage1_labels = np.load(labels_path, mmap_mode="r")
    stage1_centers = np.load(centers_path)
    row_index = np.load(row_index_path, mmap_mode="r")
    n_points = int(np.load(data_folder / "twt.npy", mmap_mode="r").shape[0])

    if stage1_labels.shape[0] != n_points:
        raise ValueError(
            "Stage-1 labels length does not match full point cloud. "
            "Run flash_kmeans_script.py with --sample-size 0 for the full-data pipeline."
        )
    if row_index.shape[0] != n_points:
        raise ValueError(
            "Stage-1 row index length does not match full point cloud. "
            "Run flash_kmeans_script.py with --sample-size 0 for the full-data pipeline."
        )
    return stage1_labels, stage1_centers, row_index


def _save_snapshot_projection(
    output_dir: Path,
    snapshot_name: str,
    center_labels: np.ndarray,
    stage1_labels: np.ndarray,
    data_folder: Path,
) -> Path:
    full_dtype = np.uint16 if np.max(center_labels) < np.iinfo(np.uint16).max else np.uint32
    full_labels = center_labels[np.asarray(stage1_labels, dtype=np.int64)].astype(full_dtype, copy=False)

    labels_path = output_dir / f"{snapshot_name}_labels.npy"
    np.save(labels_path, full_labels)

    pdf_dir = output_dir / snapshot_name / "cross_secs"
    save_point_cloud_sections(
        data_folder=data_folder,
        labels_path=labels_path,
        output_dir=pdf_dir,
    )
    return labels_path


def main():
    args = parse_args()
    data_folder = Path(args.data_folder).resolve()
    flash_dir = Path(args.flash_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stage1_labels, stage1_centers, _ = _load_stage1_outputs(
        data_folder=data_folder,
        flash_dir=flash_dir,
        run_name=args.flash_run_name,
    )

    centers_trimmed, trimmed_dim = _trim_zero_padding(stage1_centers)
    print(
        f"Stage-1 centers loaded: {stage1_centers.shape[0]} centers, "
        f"feature dim {stage1_centers.shape[1]} -> {trimmed_dim}"
    )

    snapshot_iters = tuple(range(0, args.max_iter + 1, args.snapshot_every))
    torque_result = admm_cns_torque(
        X=centers_trimmed.astype(float),
        k=args.torque_k,
        n_neighbors=args.n_neighbors,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lam=args.lam,
        rho1=args.rho1,
        rho2=args.rho2,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        snapshot_iters=snapshot_iters,
    )

    run_dir = output_root / (
        f"{args.flash_run_name}__torque_k{args.torque_k}_nn{args.n_neighbors}_iter{args.max_iter}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_index = {}
    for iteration, center_labels in sorted(torque_result.snapshots.items()):
        snapshot_name = f"iter_{int(iteration):03d}"
        center_labels = np.asarray(center_labels, dtype=np.int32)

        np.save(run_dir / f"{snapshot_name}_center_labels.npy", center_labels)
        labels_path = _save_snapshot_projection(
            output_dir=run_dir,
            snapshot_name=snapshot_name,
            center_labels=center_labels,
            stage1_labels=stage1_labels,
            data_folder=data_folder,
        )
        snapshot_index[snapshot_name] = {
            "iteration": int(iteration),
            "center_labels_path": str(run_dir / f"{snapshot_name}_center_labels.npy"),
            "full_labels_path": str(labels_path),
            "pdf_dir": str(run_dir / snapshot_name / "cross_secs"),
        }
        print(f"Saved snapshot {snapshot_name}")

    final_center_labels = np.asarray(torque_result.labels, dtype=np.int32)
    np.save(run_dir / "final_center_labels.npy", final_center_labels)
    final_labels_path = _save_snapshot_projection(
        output_dir=run_dir,
        snapshot_name="final",
        center_labels=final_center_labels,
        stage1_labels=stage1_labels,
        data_folder=data_folder,
    )

    np.save(run_dir / "objective.npy", np.asarray(torque_result.objective, dtype=float))
    np.save(run_dir / "objective_aug.npy", np.asarray(torque_result.objective_aug, dtype=float))
    np.save(run_dir / "primal_r1.npy", np.asarray(torque_result.primal_r1, dtype=float))
    np.save(run_dir / "primal_r2.npy", np.asarray(torque_result.primal_r2, dtype=float))
    np.save(run_dir / "dual_s1.npy", np.asarray(torque_result.dual_s1, dtype=float))
    np.save(run_dir / "dual_s2.npy", np.asarray(torque_result.dual_s2, dtype=float))

    meta = {
        "flash_run_name": args.flash_run_name,
        "torque_k": args.torque_k,
        "n_neighbors": args.n_neighbors,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "lam": args.lam,
        "rho1": args.rho1,
        "rho2": args.rho2,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seed": args.seed,
        "snapshot_every": args.snapshot_every,
        "n_stage1_centers": int(stage1_centers.shape[0]),
        "stage1_feature_dim": int(stage1_centers.shape[1]),
        "trimmed_feature_dim": int(trimmed_dim),
        "final_labels_path": str(final_labels_path),
        "final_pdf_dir": str(run_dir / "final" / "cross_secs"),
        "snapshots": snapshot_index,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
