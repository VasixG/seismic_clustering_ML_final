import argparse
import gc
import os
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from src.save_p_cloud_as_cube import SavePredictionPCloud
from two_stg_clust import TwoStageClust
from visualization import save_sec_interest


DEFAULT_WEIGHTS = {
    "cdp_x.npy": 0.25,
    "cdp_y.npy": 0.25,
    "twt.npy": 0.5,
}

CLUST_METHODS = {
    "gmm3": 3,
    "gmm4": 4,
    "gmm5": 5,
}


def parse_args():
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    results_dir = root_dir / "results"

    parser = argparse.ArgumentParser(description="Run seismic clustering experiments.")
    parser.add_argument("--data-folder", default=str(data_dir), help="Folder with .npy features.")
    parser.add_argument("--bolvanka-path", default=str(data_dir / "bolvanka.nc"), help="Template cube path.")
    parser.add_argument("--results-dir", default=str(results_dir), help="Output directory for clustering results.")
    parser.add_argument(
        "--attrib-config",
        default="only_geom",
        choices=["only_spectr", "no_spectr", "all", "only_geom"],
        help="Feature configuration to use.",
    )
    parser.add_argument(
        "--use-spatial",
        nargs="+",
        default=["all", "only_twt", "none"],
        choices=["all", "only_twt", "none"],
        help="Spatial feature modes to evaluate.",
    )
    parser.add_argument(
        "--twt-weights",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
        help="Weights for twt.npy when spatial features are enabled.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(CLUST_METHODS.keys()),
        choices=list(CLUST_METHODS.keys()),
        help="Clustering methods to run.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for GaussianMixture.",
    )
    return parser.parse_args()


def check_required_files(data_folder: Path, attrib_config: str):
    required = {
        "cdp_x.npy",
        "cdp_y.npy",
        "twt.npy",
        "iline.npy",
        "xline.npy",
        "bolvanka.nc",
    }

    if attrib_config == "only_geom":
        required.update(
            {
                "loc_struct_azim_cos.npy",
                "loc_struct_azim_sin.npy",
                "dip_dev.npy",
                "dip_usual.npy",
            }
        )
    elif attrib_config == "only_spectr":
        required.update({"2019_15Hz.npy", "2019_30Hz.npy", "2019_45Hz.npy"})
    else:
        required.update(
            {
                "2019_15Hz.npy",
                "2019_30Hz.npy",
                "2019_45Hz.npy",
                "2019_SUMM.npy",
                "FF.npy",
                "offset_0950.npy",
                "offset_1600.npy",
                "offset_2500.npy",
                "offset_ENV1600.npy",
                "offset_ENV2500.npy",
                "offset_ENV950.npy",
            }
        )
        if attrib_config == "no_spectr":
            required.difference_update({"2019_15Hz.npy", "2019_30Hz.npy", "2019_45Hz.npy"})

    missing = [name for name in sorted(required) if not (data_folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required data files in {data_folder}: {missing}")


def run_experiment(
    data_folder: Path,
    bolvanka_path: Path,
    results_dir: Path,
    alg_name: str,
    n_comp: int,
    attrib_config: str,
    spatial_mode: str,
    twt_weight: float | None,
    random_state: int,
):
    weights = DEFAULT_WEIGHTS.copy()
    if twt_weight is not None:
        weights["twt.npy"] = twt_weight

    if spatial_mode == "none":
        exp_name = f"{alg_name}_{attrib_config}_no_spat"
        active_weights = None
    else:
        exp_name = f"{alg_name}_{attrib_config}_twt_w_{twt_weight}"
        if spatial_mode == "all":
            exp_name += "_0.25xy"
        active_weights = weights

    print(f"Start fit-predict for {exp_name}")
    two_stg_clust = TwoStageClust(data_folder=str(data_folder))
    two_stg_clust.load_features(
        use_spatial=spatial_mode,
        attrib_config=attrib_config,
        weights=active_weights,
    )

    clust_alg = GaussianMixture(n_components=n_comp, random_state=random_state)
    result = two_stg_clust.fit_predict(clust_alg, use_kmeans_centr=False)

    save_dir = results_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    segment_res_path = save_dir / f"{exp_name}.npy"
    np.save(segment_res_path, result)

    saver = SavePredictionPCloud(
        str(save_dir),
        exp_name,
        str(bolvanka_path),
        segment_res_path=str(segment_res_path),
        iline_path=str(data_folder / "iline.npy"),
        xline_path=str(data_folder / "xline.npy"),
        twt_path=str(data_folder / "twt.npy"),
    )
    save_sec_interest(saver.results_xr, str(save_dir))
    print(f"Finished {exp_name}")


def main():
    args = parse_args()
    data_folder = Path(args.data_folder).resolve()
    bolvanka_path = Path(args.bolvanka_path).resolve()
    results_dir = Path(args.results_dir).resolve()

    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")
    if not bolvanka_path.exists():
        raise FileNotFoundError(f"Bolvanka file does not exist: {bolvanka_path}")

    check_required_files(data_folder, args.attrib_config)
    results_dir.mkdir(parents=True, exist_ok=True)

    for method_name in args.methods:
        n_comp = CLUST_METHODS[method_name]
        for spatial_mode in args.use_spatial:
            weights_to_try = args.twt_weights if spatial_mode != "none" else [None]
            for twt_weight in weights_to_try:
                try:
                    run_experiment(
                        data_folder=data_folder,
                        bolvanka_path=bolvanka_path,
                        results_dir=results_dir,
                        alg_name=method_name,
                        n_comp=n_comp,
                        attrib_config=args.attrib_config,
                        spatial_mode=spatial_mode,
                        twt_weight=twt_weight,
                        random_state=args.random_state,
                    )
                except (ValueError, RuntimeError, MemoryError, TypeError, AttributeError, FileNotFoundError) as exc:
                    print(f"Experiment failed for method={method_name}, spatial={spatial_mode}, twt_weight={twt_weight}: {exc}")
                finally:
                    gc.collect()


if __name__ == "__main__":
    main()
