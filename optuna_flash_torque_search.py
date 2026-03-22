import argparse
import gc
import json
import hashlib
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from auto_flash_torque_search import _best_snapshot_by_surface_ari, _ensure_flash_run, _run_torque
from fast_kmeans_torque_pipeline import _load_stage1_outputs, _save_snapshot_projection, _trim_zero_padding
from src.surface_projection_metric import SurfaceProjectionEvaluator


def parse_args():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Optuna over existing Flash-KMeans stage-1 runs and torque parameters.")
    parser.add_argument("--data-folder", default=str(root_dir / "data"))
    parser.add_argument("--flash-dir", default=str(root_dir / "results" / "flash_kmeans"))
    parser.add_argument("--output-dir", default=str(root_dir / "results" / "flash_torque_optuna"))
    parser.add_argument("--bolvanka-path", default=str(root_dir / "data" / "bolvanka.nc"))
    parser.add_argument("--surface-indices-path", default=str(root_dir / "data" / "surface_indices_init_resol.npz"))
    parser.add_argument("--facies-mask-path", default=str(root_dir / "data" / "big_polyg_mask_init_resolut.npy"))
    parser.add_argument("--flash-run-names", nargs="+", default=None)
    parser.add_argument("--flash-run-pattern", default="flashkmeans_k*_feat_*_only_twt_n*.npy")
    parser.add_argument("--combo-file", default=None, help="JSON file with stage-1 feature combinations.")
    parser.add_argument("--flash-k-values", nargs="+", type=int, default=None)
    parser.add_argument("--flash-use-spatial", default="only_twt", choices=["all", "only_twt", "none"])
    parser.add_argument("--flash-sample-size", type=int, default=0)
    parser.add_argument("--flash-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--flash-max-iter", type=int, default=50)
    parser.add_argument("--flash-tol", type=float, default=1e-4)
    parser.add_argument("--flash-verbose", action="store_true")
    parser.add_argument("--torque-k-values", nargs="+", type=int, required=True)
    parser.add_argument("--n-neighbors-values", nargs="+", type=int, default=[8, 10, 12])
    parser.add_argument("--gamma-low", type=float, default=0.1)
    parser.add_argument("--gamma-high", type=float, default=0.35)
    parser.add_argument("--lam-low", type=float, default=0.5)
    parser.add_argument("--lam-high", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rho1", type=float, default=1.0)
    parser.add_argument("--rho2", type=float, default=1.0)
    parser.add_argument("--search-max-iter", type=int, default=6)
    parser.add_argument("--final-max-iter", type=int, default=12)
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--study-name", default="flash_torque_surface_ari")
    parser.add_argument(
        "--auto-study-name",
        action="store_true",
        help="Append a hash of the search space to study name (avoids Optuna dynamic categorical errors).",
    )
    parser.add_argument("--storage", default=None, help="Optuna storage URL, e.g. sqlite:///optuna_flash_torque.db")
    parser.add_argument("--sampler-seed", type=int, default=42)
    return parser.parse_args()


def _discover_run_names(flash_dir: Path, pattern: str) -> list[str]:
    run_names = []
    for path in sorted(flash_dir.glob(pattern)):
        name = path.name
        if not name.endswith("_labels.npy"):
            continue
        run_name = name[:-len("_labels.npy")]
        if (flash_dir / f"{run_name}_centers.npy").exists() and (flash_dir / f"{run_name}_row_index.npy").exists():
            run_names.append(run_name)
    return run_names


def _hash_search_space(
    flash_run_names: list[str],
    torque_k_values: list[int],
    n_neighbors_values: list[int],
    gamma_low: float,
    gamma_high: float,
    lam_low: float,
    lam_high: float,
) -> str:
    payload = {
        "flash_run_names": sorted(flash_run_names),
        "torque_k_values": list(torque_k_values),
        "n_neighbors_values": list(n_neighbors_values),
        "gamma_low": float(gamma_low),
        "gamma_high": float(gamma_high),
        "lam_low": float(lam_low),
        "lam_high": float(lam_high),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def _save_best_run(
    output_root: Path,
    best_trial: dict,
    evaluator: SurfaceProjectionEvaluator,
    data_folder: Path,
    flash_dir: Path,
    args,
):
    stage1_labels, stage1_centers, _ = _load_stage1_outputs(
        data_folder=data_folder,
        flash_dir=flash_dir,
        run_name=best_trial["flash_run_name"],
    )
    centers_trimmed, trimmed_dim = _trim_zero_padding(stage1_centers)
    result = _run_torque(
        centers=centers_trimmed,
        torque_k=int(best_trial["torque_k"]),
        n_neighbors=int(best_trial["n_neighbors"]),
        alpha=args.alpha,
        beta=args.beta,
        gamma=float(best_trial["gamma"]),
        lam=float(best_trial["lam"]),
        rho1=args.rho1,
        rho2=args.rho2,
        max_iter=args.final_max_iter,
        tol=args.tol,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
    )

    run_dir = output_root / (
        f"{best_trial['flash_run_name']}__optuna_best_torque_k{best_trial['torque_k']}"
        f"_nn{best_trial['n_neighbors']}_g{best_trial['gamma']:.4f}_lam{best_trial['lam']:.4f}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_index = {}
    snapshot_surface_ari = {}
    best_snapshot_name = None
    best_snapshot_score = -np.inf
    for iteration, center_labels in sorted(result.snapshots.items()):
        snapshot_name = f"iter_{int(iteration):03d}"
        center_labels = np.asarray(center_labels, dtype=np.int32)
        snapshot_ari, snapshot_surface_map = evaluator.score(center_labels=center_labels, stage1_labels=stage1_labels)
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

    final_center_labels = np.asarray(result.labels, dtype=np.int32)
    np.save(run_dir / "final_center_labels.npy", final_center_labels)
    final_ari, final_surface_map = evaluator.score(center_labels=final_center_labels, stage1_labels=stage1_labels)
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

    best_full_labels_path = None
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

    (run_dir / "snapshot_surface_ari.json").write_text(json.dumps(snapshot_surface_ari, indent=2), encoding="utf-8")
    np.save(run_dir / "objective.npy", np.asarray(result.objective, dtype=float))
    np.save(run_dir / "objective_aug.npy", np.asarray(result.objective_aug, dtype=float))
    np.save(run_dir / "primal_r1.npy", np.asarray(result.primal_r1, dtype=float))
    np.save(run_dir / "primal_r2.npy", np.asarray(result.primal_r2, dtype=float))
    np.save(run_dir / "dual_s1.npy", np.asarray(result.dual_s1, dtype=float))
    np.save(run_dir / "dual_s2.npy", np.asarray(result.dual_s2, dtype=float))

    meta = {
        "best_trial": best_trial,
        "final_surface_ari": float(final_ari),
        "best_snapshot_name": best_snapshot_name,
        "best_snapshot_surface_ari": float(best_snapshot_score),
        "final_labels_path": str(final_labels_path),
        "best_full_labels_path": str(best_full_labels_path) if best_full_labels_path is not None else None,
        "snapshots": snapshot_index,
        "n_stage1_centers": int(stage1_centers.shape[0]),
        "stage1_feature_dim": int(stage1_centers.shape[1]),
        "trimmed_feature_dim": int(trimmed_dim),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def main():
    args = parse_args()
    data_folder = Path(args.data_folder).resolve()
    flash_dir = Path(args.flash_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("optuna is required for this script. Install it with: pip3 install optuna") from exc

    if args.flash_run_names:
        flash_run_names = list(dict.fromkeys(args.flash_run_names))
    else:
        flash_run_names = _discover_run_names(flash_dir, args.flash_run_pattern)
    if args.combo_file and args.flash_k_values:
        combo_file = Path(args.combo_file).resolve()
        combinations = json.loads(combo_file.read_text(encoding="utf-8"))
        if not isinstance(combinations, list):
            raise ValueError("--combo-file must contain a JSON list of combinations.")
        generated_run_names = []
        total_stage1_runs = len(args.flash_k_values) * len(combinations)
        stage1_progress = tqdm(total=total_stage1_runs, desc="Stage-1 generation", unit="run")
        for k in args.flash_k_values:
            if k < 1 or (k & (k - 1)) != 0:
                raise ValueError(f"k must be a power of two, got {k}")
            for combo in combinations:
                if not isinstance(combo, list) or not combo:
                    raise ValueError(f"Invalid feature combination: {combo}")
                stage1_progress.set_postfix({"k": k, "n_feat": len(combo)})
                generated_run_names.append(
                    _ensure_flash_run(
                        data_folder=data_folder,
                        flash_dir=flash_dir,
                        n_clusters=k,
                        attrib_config="all",
                        use_spatial=args.flash_use_spatial,
                        feature_files=list(combo),
                        sample_size=args.flash_sample_size,
                        seed=args.seed,
                        dtype=args.flash_dtype,
                        max_iter=args.flash_max_iter,
                        tol=args.flash_tol,
                        verbose=args.flash_verbose,
                    )
                )
                stage1_progress.update(1)
        stage1_progress.close()
        flash_run_names.extend(generated_run_names)
        flash_run_names = list(dict.fromkeys(flash_run_names))
    if not flash_run_names:
        raise ValueError("No stage-1 runs found for Optuna search.")

    evaluator = SurfaceProjectionEvaluator(
        data_folder=data_folder,
        bolvanka_path=Path(args.bolvanka_path).resolve(),
        surface_indices_path=Path(args.surface_indices_path).resolve(),
        facies_mask_path=Path(args.facies_mask_path).resolve(),
    )
    print(f"Surface evaluator: {evaluator.describe()}")
    print(f"Optuna stage-1 candidates: {len(flash_run_names)} runs")

    cache = {}
    trial_rows = []
    fail_score = -1e9

    def get_stage1(run_name: str):
        if run_name not in cache:
            stage1_labels, stage1_centers, _ = _load_stage1_outputs(
                data_folder=data_folder,
                flash_dir=flash_dir,
                run_name=run_name,
            )
            centers_trimmed, trimmed_dim = _trim_zero_padding(stage1_centers)
            cache[run_name] = (stage1_labels, stage1_centers, centers_trimmed, trimmed_dim)
        return cache[run_name]

    def objective(trial):
        flash_run_name = trial.suggest_categorical("flash_run_name", flash_run_names)
        torque_k = trial.suggest_categorical("torque_k", args.torque_k_values)
        n_neighbors = trial.suggest_categorical("n_neighbors", args.n_neighbors_values)
        gamma = trial.suggest_float("gamma", args.gamma_low, args.gamma_high)
        lam = trial.suggest_float("lam", args.lam_low, args.lam_high)

        try:
            stage1_labels, stage1_centers, centers_trimmed, trimmed_dim = get_stage1(flash_run_name)
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
                "trial": int(trial.number),
                "flash_run_name": flash_run_name,
                "stage1_n_centers": int(stage1_centers.shape[0]),
                "stage1_feature_dim": int(stage1_centers.shape[1]),
                "trimmed_feature_dim": int(trimmed_dim),
                "torque_k": int(torque_k),
                "n_neighbors": int(n_neighbors),
                "gamma": float(gamma),
                "lam": float(lam),
                "metric": "surface_ari",
                "score_for_optimization": float(score_raw),
                "best_iteration": int(best_iteration),
                "n_unique_labels": int(len(np.unique(best_labels))),
                "iteration_scores": per_iteration_scores,
            }
            trial_rows.append(row)
            trial.set_user_attr("result", row)
            return float(score_raw)
        except Exception as exc:
            msg = f"Trial {trial.number} failed: {exc}"
            print(msg)
            row = {
                "trial": int(trial.number),
                "flash_run_name": flash_run_name,
                "torque_k": int(torque_k),
                "n_neighbors": int(n_neighbors),
                "gamma": float(gamma),
                "lam": float(lam),
                "metric": "surface_ari",
                "score_for_optimization": float(fail_score),
                "error": str(exc),
            }
            trial_rows.append(row)
            trial.set_user_attr("result", row)
            trial.set_user_attr("error", str(exc))
            return float(fail_score)
        finally:
            gc.collect()

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study_name = args.study_name
    if args.auto_study_name:
        study_name = f"{args.study_name}_{_hash_search_space(
            flash_run_names,
            args.torque_k_values,
            args.n_neighbors_values,
            args.gamma_low,
            args.gamma_high,
            args.lam_low,
            args.lam_high,
        )}"
        print(f"Optuna study name: {study_name}")
    if args.storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_trial_result = study.best_trial.user_attrs["result"]
    study_dir = output_root / f"optuna_{args.study_name}"
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "all_trials.json").write_text(json.dumps(trial_rows, indent=2), encoding="utf-8")
    (study_dir / "best_trial.json").write_text(json.dumps(best_trial_result, indent=2), encoding="utf-8")
    (study_dir / "study_summary.json").write_text(
        json.dumps(
            {
                "best_value": float(study.best_value),
                "best_params": study.best_params,
                "n_trials": len(study.trials),
                "stage1_candidates": flash_run_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    run_dir = _save_best_run(
        output_root=study_dir,
        best_trial=best_trial_result,
        evaluator=evaluator,
        data_folder=data_folder,
        flash_dir=flash_dir,
        args=args,
    )
    print(f"Optuna artifacts saved to {study_dir}")
    print(f"Best run artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
