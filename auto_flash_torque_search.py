import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from flash_kmeans_script import (
    _feature_tag,
    build_feature_matrix,
    maybe_subsample,
    pad_feature_dim_to_power_of_two,
)
from fast_kmeans_torque_pipeline import (
    _load_stage1_outputs,
    _save_snapshot_projection,
    _trim_zero_padding,
)
from src.surface_projection_metric import SurfaceProjectionEvaluator
from torque_clustering.admm_cns_torque_module import admm_cns_torque
from two_stg_clust import ATTRIBUTE_FEATURES


def parse_args():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Automatic parameter search for Flash-KMeans -> torque clustering."
    )
    parser.add_argument("--data-folder", default=str(root_dir / "data"))
    parser.add_argument("--flash-dir", default=str(root_dir / "results" / "flash_kmeans"))
    parser.add_argument("--flash-run-name", default=None)
    parser.add_argument("--flash-run-names", nargs="+", default=None)
    parser.add_argument("--flash-k-values", nargs="+", type=int, default=None)
    parser.add_argument(
        "--flash-attrib-config-values",
        nargs="+",
        default=["only_geom", "only_spectr", "no_spectr", "all"],
        choices=["only_spectr", "no_spectr", "all", "only_geom"],
    )
    parser.add_argument("--flash-skip-attrib-config-grid", action="store_true")
    parser.add_argument(
        "--flash-single-feature-values",
        nargs="+",
        default=None,
        choices=ATTRIBUTE_FEATURES,
    )
    parser.add_argument(
        "--flash-feature-combination",
        action="append",
        nargs="+",
        default=None,
        choices=ATTRIBUTE_FEATURES,
        help="Explicit stage-1 feature combinations. Repeat the flag for multiple combinations.",
    )
    parser.add_argument(
        "--flash-feature-combinations-file",
        default=None,
        help="JSON file with a list of feature combinations, e.g. [[\"dip_dev.npy\", \"FF.npy\"], ...].",
    )
    parser.add_argument("--flash-iterate-all-single-features", action="store_true")
    parser.add_argument(
        "--flash-use-spatial-values",
        nargs="+",
        default=None,
        choices=["all", "only_twt", "none"],
    )
    parser.add_argument("--flash-sample-size", type=int, default=0)
    parser.add_argument("--flash-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--flash-tol", type=float, default=1e-4)
    parser.add_argument("--flash-verbose", action="store_true")
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


def _flash_run_name(
    n_clusters: int,
    attrib_config: str,
    use_spatial: str,
    n_rows: int,
    feature_files: list[str] | None = None,
) -> str:
    return f"flashkmeans_k{n_clusters}_{_feature_tag(feature_files, attrib_config)}_{use_spatial}_n{n_rows}"


def _ensure_flash_run(
    data_folder: Path,
    flash_dir: Path,
    n_clusters: int,
    attrib_config: str,
    use_spatial: str,
    feature_files: list[str] | None,
    sample_size: int,
    seed: int,
    dtype: str,
    tol: float,
    verbose: bool,
) -> str:
    first_feature_matrix, _ = build_feature_matrix(
        data_folder=data_folder,
        attrib_config=attrib_config,
        use_spatial=use_spatial,
        feature_files=feature_files,
    )
    total_rows = first_feature_matrix.shape[0]
    effective_rows = total_rows if sample_size <= 0 or sample_size >= total_rows else sample_size
    run_name = _flash_run_name(n_clusters, attrib_config, use_spatial, effective_rows, feature_files=feature_files)

    labels_path = flash_dir / f"{run_name}_labels.npy"
    centers_path = flash_dir / f"{run_name}_centers.npy"
    row_index_path = flash_dir / f"{run_name}_row_index.npy"
    if labels_path.exists() and centers_path.exists() and row_index_path.exists():
        print(f"Reuse existing stage-1 run: {run_name}")
        return run_name

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for Flash-KMeans stage-1 generation.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate stage-1 Flash-KMeans runs automatically.")

    try:
        from flash_kmeans import batch_kmeans_Euclid
    except Exception as exc:
        raise RuntimeError("flash-kmeans is not installed or failed to import.") from exc

    features = first_feature_matrix
    features, selected_idx = maybe_subsample(features, sample_size, seed)
    original_feature_dim = features.shape[1]
    features, _ = pad_feature_dim_to_power_of_two(features)

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    x = torch.from_numpy(features).to(device="cuda", dtype=torch_dtype).unsqueeze(0)
    cluster_ids, centers, _ = batch_kmeans_Euclid(
        x,
        n_clusters=n_clusters,
        tol=tol,
        verbose=verbose,
    )

    cluster_ids_np = cluster_ids.squeeze(0).detach().cpu().numpy()
    centers_np = centers.squeeze(0).detach().cpu().numpy()

    flash_dir.mkdir(parents=True, exist_ok=True)
    np.save(labels_path, cluster_ids_np)
    np.save(centers_path, centers_np)
    np.save(row_index_path, selected_idx)

    meta = {
        "run_name": run_name,
        "n_rows": int(features.shape[0]),
        "n_features": int(original_feature_dim),
        "n_features_padded": int(features.shape[1]),
        "n_clusters": int(n_clusters),
        "use_spatial": use_spatial,
        "attrib_config": attrib_config,
        "feature_files": feature_files,
        "dtype": dtype,
    }
    (flash_dir / f"{run_name}_meta.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in meta.items()),
        encoding="utf-8",
    )
    print(f"Created stage-1 run: {run_name}")
    return run_name


def main():
    args = parse_args()
    data_folder = Path(args.data_folder).resolve()
    flash_dir = Path(args.flash_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    evaluator = SurfaceProjectionEvaluator(
        data_folder=data_folder,
        bolvanka_path=Path(args.bolvanka_path).resolve(),
        surface_indices_path=Path(args.surface_indices_path).resolve(),
        facies_mask_path=Path(args.facies_mask_path).resolve(),
    )
    print(f"Surface evaluator: {evaluator.describe()}")

    flash_run_names = []
    if args.flash_run_names:
        flash_run_names.extend(args.flash_run_names)
    if args.flash_run_name:
        flash_run_names.append(args.flash_run_name)
    single_feature_values = list(args.flash_single_feature_values or [])
    if args.flash_iterate_all_single_features:
        single_feature_values = list(ATTRIBUTE_FEATURES)
    feature_combinations = []
    if args.flash_feature_combination:
        feature_combinations.extend([list(combo) for combo in args.flash_feature_combination])
    if args.flash_feature_combinations_file:
        combos_from_file = json.loads(Path(args.flash_feature_combinations_file).read_text(encoding="utf-8"))
        if not isinstance(combos_from_file, list):
            raise ValueError("--flash-feature-combinations-file must contain a JSON list.")
        for combo in combos_from_file:
            if not isinstance(combo, list) or not combo:
                raise ValueError("Each feature combination must be a non-empty JSON list.")
            invalid = [feature for feature in combo if feature not in ATTRIBUTE_FEATURES]
            if invalid:
                raise ValueError(f"Unknown feature files in combinations file: {invalid}")
            feature_combinations.append(list(combo))
    deduped_combinations = []
    seen_combinations = set()
    for combo in feature_combinations:
        combo_key = tuple(combo)
        if combo_key in seen_combinations:
            continue
        seen_combinations.add(combo_key)
        deduped_combinations.append(combo)
    feature_combinations = deduped_combinations
    if args.flash_k_values and args.flash_use_spatial_values:
        print(
            "Stage-1 search grid:",
            {
                "flash_k_values": args.flash_k_values,
                "flash_attrib_config_values": [] if args.flash_skip_attrib_config_grid else args.flash_attrib_config_values,
                "flash_single_feature_values": single_feature_values,
                "flash_feature_combinations": feature_combinations,
                "flash_use_spatial_values": args.flash_use_spatial_values,
            },
        )
        stage1_specs = []
        if not args.flash_skip_attrib_config_grid:
            for n_clusters, attrib_config, use_spatial in itertools.product(
                args.flash_k_values,
                args.flash_attrib_config_values,
                args.flash_use_spatial_values,
            ):
                stage1_specs.append((n_clusters, attrib_config, use_spatial, None))
        for n_clusters, feature_file, use_spatial in itertools.product(
            args.flash_k_values,
            single_feature_values,
            args.flash_use_spatial_values,
        ):
            stage1_specs.append((n_clusters, "all", use_spatial, [feature_file]))
        for n_clusters, feature_files, use_spatial in itertools.product(
            args.flash_k_values,
            feature_combinations,
            args.flash_use_spatial_values,
        ):
            stage1_specs.append((n_clusters, "all", use_spatial, list(feature_files)))

        for n_clusters, attrib_config, use_spatial, feature_files in stage1_specs:
            flash_run_names.append(
                _ensure_flash_run(
                    data_folder=data_folder,
                    flash_dir=flash_dir,
                    n_clusters=n_clusters,
                    attrib_config=attrib_config,
                    use_spatial=use_spatial,
                    feature_files=feature_files,
                    sample_size=args.flash_sample_size,
                    seed=args.seed,
                    dtype=args.flash_dtype,
                    tol=args.flash_tol,
                    verbose=args.flash_verbose,
                )
            )
    flash_run_names = list(dict.fromkeys(flash_run_names))
    if not flash_run_names:
        raise ValueError("Provide --flash-run-name or --flash-run-names.")

    search_results = []
    combo_grid = list(itertools.product(
        args.torque_k_values,
        args.n_neighbors_values,
        args.gamma_values,
        args.lam_values,
    ))

    best = None
    best_score = -np.inf
    trial_idx = 0
    for flash_run_name in flash_run_names:
        stage1_labels, stage1_centers, _ = _load_stage1_outputs(
            data_folder=data_folder,
            flash_dir=flash_dir,
            run_name=flash_run_name,
        )
        centers_trimmed, trimmed_dim = _trim_zero_padding(stage1_centers)
        print(
            f"Stage-1 centers loaded for {flash_run_name}: "
            f"{stage1_centers.shape[0]} centers, feature dim {stage1_centers.shape[1]} -> {trimmed_dim}"
        )
        for torque_k, n_neighbors, gamma, lam in combo_grid:
            trial_idx += 1
            print(
                f"Trial {trial_idx}: flash_run={flash_run_name}, torque_k={torque_k}, "
                f"n_neighbors={n_neighbors}, gamma={gamma}, lam={lam}"
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
                "flash_run_name": flash_run_name,
                "stage1_n_centers": int(stage1_centers.shape[0]),
                "stage1_feature_dim": int(stage1_centers.shape[1]),
                "trimmed_feature_dim": int(trimmed_dim),
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
        f"{best['flash_run_name']}__metric_surface_ari__best_torque_k{best['torque_k']}"
        f"_nn{best['n_neighbors']}_g{best['gamma']}_lam{best['lam']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "search_results.json").write_text(json.dumps(search_results, indent=2), encoding="utf-8")
    (run_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Best params: {best}")

    stage1_labels, stage1_centers, _ = _load_stage1_outputs(
        data_folder=data_folder,
        flash_dir=flash_dir,
        run_name=best["flash_run_name"],
    )
    centers_trimmed, trimmed_dim = _trim_zero_padding(stage1_centers)

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
        best_snapshot_labels = np.load(run_dir / f"{best_snapshot_name}_center_labels.npy")
        best_surface_map = np.load(run_dir / f"{best_snapshot_name}_surface_map.npy")
        np.save(run_dir / "best_center_labels.npy", best_snapshot_labels)
        np.save(run_dir / "best_surface_map.npy", best_surface_map)
        evaluator.save_plot(
            z_map=best_surface_map,
            output_path=run_dir / "best_surface_ari_map.pdf",
            title=f"Best surface snapshot {best_snapshot_name}, ARI={best_snapshot_score:.3f}",
        )
        best_full_labels_path = _save_snapshot_projection(
            output_dir=run_dir,
            snapshot_name="best",
            center_labels=np.asarray(best_snapshot_labels, dtype=np.int32),
            stage1_labels=stage1_labels,
            data_folder=data_folder,
        )
    else:
        best_full_labels_path = None

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
        "flash_run_name": best["flash_run_name"],
        "flash_run_names_considered": flash_run_names,
        "metric": "surface_ari",
        "best": best,
        "final_surface_ari": final_ari,
        "best_snapshot_name": best_snapshot_name,
        "best_snapshot_surface_ari": best_snapshot_score,
        "best_snapshot": best_snapshot_meta,
        "best_full_labels_path": str(best_full_labels_path) if best_full_labels_path is not None else None,
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
