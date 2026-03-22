import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from fast_kmeans_torque_pipeline import (
    _load_stage1_outputs,
    _save_snapshot_projection,
    _trim_zero_padding,
)
from src.surface_projection_metric import SurfaceProjectionEvaluator
from torque_clustering.admm_cns_torque_module import admm_cns_torque


def parse_args():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Automatic parameter search for Flash-KMeans -> torque clustering."
    )
    parser.add_argument("--data-folder", default=str(root_dir / "data"))
    parser.add_argument("--flash-dir", default=str(root_dir / "results" / "flash_kmeans"))
    parser.add_argument("--flash-run-name", required=True)
    parser.add_argument("--output-dir", default=str(root_dir / "results" / "flash_torque_auto"))
    parser.add_argument("--bolvanka-path", default=str(root_dir / "data" / "bolvanka.nc"))
    parser.add_argument("--surface-indices-path", default=str(root_dir / "data" / "surface_indices_init_resol.npz"))
    parser.add_argument("--facies-mask-path", default=str(root_dir / "data" / "big_polyg_mask_init_resolut.npy"))
    parser.add_argument("--torque-k-values", nargs="+", type=int, required=True)
    parser.add_argument("--n-neighbors-values", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--gamma-values", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    parser.add_argument("--lam-values", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rho1", type=float, default=1.0)
    parser.add_argument("--rho2", type=float, default=1.0)
    parser.add_argument("--search-max-iter", type=int, default=12)
    parser.add_argument("--final-max-iter", type=int, default=40)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-every", type=int, default=2)
    return parser.parse_args()


def _run_torque(
    centers: np.ndarray,
    torque_k: int,
    n_neighbors: int,
    alpha: float,
    beta: float,
    gamma: float,
    lam: float,
    rho1: float,
    rho2: float,
    max_iter: int,
    tol: float,
    seed: int,
    snapshot_every: int | None,
):
    snapshot_iters = () if snapshot_every is None else tuple(range(0, max_iter + 1, snapshot_every))
    return admm_cns_torque(
        X=centers.astype(float),
        k=torque_k,
        n_neighbors=n_neighbors,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lam=lam,
        rho1=rho1,
        rho2=rho2,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        snapshot_iters=snapshot_iters,
    )


def _best_snapshot_by_surface_ari(
    result,
    evaluator: SurfaceProjectionEvaluator,
    stage1_labels: np.ndarray,
):
    best_iteration = None
    best_labels = None
    best_score = -np.inf
    best_surface_map = None
    per_iteration = {}

    snapshots = result.snapshots if result.snapshots else {result.labels.shape[0]: result.labels}
    for iteration, center_labels in sorted(snapshots.items()):
        center_labels = np.asarray(center_labels, dtype=np.int32)
        score, surface_map = evaluator.score(
            center_labels=center_labels,
            stage1_labels=stage1_labels,
        )
        per_iteration[int(iteration)] = float(score)
        if score > best_score:
            best_score = score
            best_iteration = int(iteration)
            best_labels = center_labels
            best_surface_map = surface_map

    return best_iteration, best_labels, best_score, best_surface_map, per_iteration


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
    evaluator = SurfaceProjectionEvaluator(
        data_folder=data_folder,
        bolvanka_path=Path(args.bolvanka_path).resolve(),
        surface_indices_path=Path(args.surface_indices_path).resolve(),
        facies_mask_path=Path(args.facies_mask_path).resolve(),
    )
    centers_trimmed, trimmed_dim = _trim_zero_padding(stage1_centers)
    print(
        f"Stage-1 centers loaded: {stage1_centers.shape[0]} centers, "
        f"feature dim {stage1_centers.shape[1]} -> {trimmed_dim}"
    )
    print(f"Surface evaluator: {evaluator.describe()}")

    search_results = []
    combo_iter = itertools.product(
        args.torque_k_values,
        args.n_neighbors_values,
        args.gamma_values,
        args.lam_values,
    )

    best = None
    best_score = -np.inf
    trial_idx = 0
    for torque_k, n_neighbors, gamma, lam in combo_iter:
        trial_idx += 1
        print(
            f"Trial {trial_idx}: torque_k={torque_k}, n_neighbors={n_neighbors}, "
            f"gamma={gamma}, lam={lam}"
        )
        result = _run_torque(
            centers=centers_trimmed,
            torque_k=torque_k,
            n_neighbors=n_neighbors,
            alpha=args.alpha,
            beta=args.beta,
            gamma=gamma,
            lam=lam,
            rho1=args.rho1,
            rho2=args.rho2,
            max_iter=args.search_max_iter,
            tol=args.tol,
            seed=args.seed,
            snapshot_every=1,
        )

        best_iteration, best_labels, score_raw, _, per_iteration_scores = _best_snapshot_by_surface_ari(
            result=result,
            evaluator=evaluator,
            stage1_labels=stage1_labels,
        )
        row = {
            "trial": trial_idx,
            "torque_k": torque_k,
            "n_neighbors": n_neighbors,
            "gamma": gamma,
            "lam": lam,
            "metric": "surface_ari",
            "score_for_optimization": score_raw,
            "score_raw": score_raw,
            "best_iteration": int(best_iteration),
            "n_unique_labels": int(len(np.unique(best_labels))),
            "iteration_scores": per_iteration_scores,
        }
        search_results.append(row)
        print(
            f"  best_surface_ari={score_raw:.6f}, "
            f"best_iteration={best_iteration}, unique_labels={row['n_unique_labels']}"
        )

        if score_raw > best_score:
            best_score = score_raw
            best = row

    if best is None:
        raise RuntimeError("No valid clustering configuration found during search.")

    run_dir = output_root / (
        f"{args.flash_run_name}__metric_surface_ari__best_torque_k{best['torque_k']}"
        f"_nn{best['n_neighbors']}_g{best['gamma']}_lam{best['lam']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "search_results.json").write_text(json.dumps(search_results, indent=2), encoding="utf-8")
    (run_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Best params: {best}")

    best_result = _run_torque(
        centers=centers_trimmed,
        torque_k=int(best["torque_k"]),
        n_neighbors=int(best["n_neighbors"]),
        alpha=args.alpha,
        beta=args.beta,
        gamma=float(best["gamma"]),
        lam=float(best["lam"]),
        rho1=args.rho1,
        rho2=args.rho2,
        max_iter=args.final_max_iter,
        tol=args.tol,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
    )

    snapshot_index = {}
    snapshot_surface_ari = {}
    best_snapshot_name = None
    best_snapshot_score = -np.inf
    for iteration, center_labels in sorted(best_result.snapshots.items()):
        snapshot_name = f"iter_{int(iteration):03d}"
        center_labels = np.asarray(center_labels, dtype=np.int32)
        snapshot_ari, snapshot_surface_map = evaluator.score(
            center_labels=center_labels,
            stage1_labels=stage1_labels,
        )
        np.save(run_dir / f"{snapshot_name}_center_labels.npy", center_labels)
        labels_path = _save_snapshot_projection(
            output_dir=run_dir,
            snapshot_name=snapshot_name,
            center_labels=center_labels,
            stage1_labels=stage1_labels,
            data_folder=data_folder,
        )
        np.save(run_dir / f"{snapshot_name}_surface_map.npy", snapshot_surface_map)
        evaluator.save_plot(
            z_map=snapshot_surface_map,
            output_path=run_dir / f"{snapshot_name}_surface_ari_map.pdf",
            title=f"Reflecting horizon cluster map, ARI={snapshot_ari:.3f}",
        )
        snapshot_index[snapshot_name] = {
            "iteration": int(iteration),
            "surface_ari": float(snapshot_ari),
            "center_labels_path": str(run_dir / f"{snapshot_name}_center_labels.npy"),
            "full_labels_path": str(labels_path),
            "pdf_dir": str(run_dir / snapshot_name / "cross_secs"),
            "surface_map_path": str(run_dir / f"{snapshot_name}_surface_map.npy"),
            "surface_map_pdf": str(run_dir / f"{snapshot_name}_surface_ari_map.pdf"),
        }
        snapshot_surface_ari[snapshot_name] = float(snapshot_ari)
        if snapshot_ari > best_snapshot_score:
            best_snapshot_score = float(snapshot_ari)
            best_snapshot_name = snapshot_name
        print(f"Saved snapshot {snapshot_name}")

    final_center_labels = np.asarray(best_result.labels, dtype=np.int32)
    np.save(run_dir / "final_center_labels.npy", final_center_labels)
    final_ari, final_surface_map = evaluator.score(
        center_labels=final_center_labels,
        stage1_labels=stage1_labels,
    )
    final_labels_path = _save_snapshot_projection(
        output_dir=run_dir,
        snapshot_name="final",
        center_labels=final_center_labels,
        stage1_labels=stage1_labels,
        data_folder=data_folder,
    )
    np.save(run_dir / "final_surface_map.npy", final_surface_map)
    evaluator.save_plot(
        z_map=final_surface_map,
        output_path=run_dir / "final_surface_ari_map.pdf",
        title=f"Reflecting horizon cluster map, ARI={final_ari:.3f}",
    )

    if best_snapshot_name is not None:
        best_snapshot_meta = snapshot_index[best_snapshot_name]
    else:
        best_snapshot_meta = None

    np.save(run_dir / "objective.npy", np.asarray(best_result.objective, dtype=float))
    np.save(run_dir / "objective_aug.npy", np.asarray(best_result.objective_aug, dtype=float))
    np.save(run_dir / "primal_r1.npy", np.asarray(best_result.primal_r1, dtype=float))
    np.save(run_dir / "primal_r2.npy", np.asarray(best_result.primal_r2, dtype=float))
    np.save(run_dir / "dual_s1.npy", np.asarray(best_result.dual_s1, dtype=float))
    np.save(run_dir / "dual_s2.npy", np.asarray(best_result.dual_s2, dtype=float))
    (run_dir / "snapshot_surface_ari.json").write_text(
        json.dumps(snapshot_surface_ari, indent=2),
        encoding="utf-8",
    )

    meta = {
        "flash_run_name": args.flash_run_name,
        "metric": "surface_ari",
        "best": best,
        "final_surface_ari": final_ari,
        "best_snapshot_name": best_snapshot_name,
        "best_snapshot_surface_ari": best_snapshot_score,
        "best_snapshot": best_snapshot_meta,
        "alpha": args.alpha,
        "beta": args.beta,
        "rho1": args.rho1,
        "rho2": args.rho2,
        "search_max_iter": args.search_max_iter,
        "final_max_iter": args.final_max_iter,
        "snapshot_every": args.snapshot_every,
        "n_stage1_centers": int(stage1_centers.shape[0]),
        "stage1_feature_dim": int(stage1_centers.shape[1]),
        "trimmed_feature_dim": int(trimmed_dim),
        "final_labels_path": str(final_labels_path),
        "final_pdf_dir": str(run_dir / "final" / "cross_secs"),
        "snapshots": snapshot_index,
        "data_used": {
            "stage1_labels": str(flash_dir / f"{args.flash_run_name}_labels.npy"),
            "stage1_centers": str(flash_dir / f"{args.flash_run_name}_centers.npy"),
            "point_cloud_coords": [
                str(data_folder / "iline.npy"),
                str(data_folder / "xline.npy"),
                str(data_folder / "twt.npy"),
                str(data_folder / "cdp_x.npy"),
                str(data_folder / "cdp_y.npy"),
            ],
            "surface_indices": str(Path(args.surface_indices_path).resolve()),
            "facies_mask": str(Path(args.facies_mask_path).resolve()),
            "reflecting_horizon_csv": str(data_folder / "reflecting_horizon.csv"),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Auto-search artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
