from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.metrics import adjusted_rand_score

from src.surface_extractor import SeismicMapExtractor


def _compose_point_keys(iline: np.ndarray, xline: np.ndarray, twt: np.ndarray) -> np.ndarray:
    iline64 = iline.astype(np.int64, copy=False)
    xline64 = xline.astype(np.int64, copy=False)
    twt64 = np.rint(twt).astype(np.int64, copy=False)
    return (iline64 << 32) | (xline64 << 16) | twt64


class SurfaceProjectionEvaluator:
    """
    Fast projection of point-cloud labels to the reflecting horizon and ARI computation.
    The logic mirrors the older Inference + SeismicMapExtractor flow, but avoids
    rebuilding a full 3D cube for every trial.
    """

    def __init__(
        self,
        data_folder: str | Path,
        bolvanka_path: str | Path,
        surface_indices_path: str | Path,
        facies_mask_path: str | Path,
    ) -> None:
        self.data_folder = Path(data_folder)
        self.bolvanka_path = Path(bolvanka_path)
        self.surface_indices_path = Path(surface_indices_path)
        self.facies_mask_path = Path(facies_mask_path)
        self.cache_path = self.data_folder / (
            f".surface_cache_{self.surface_indices_path.stem}_{self.facies_mask_path.stem}.npz"
        )

        self.facies_mask = np.load(self.facies_mask_path)
        self._load_or_build_cache()

    def _load_or_build_cache(self) -> None:
        if self.cache_path.exists():
            cache = np.load(self.cache_path, allow_pickle=True)
            self.row_index_grid = cache["row_index_grid"]
            self.valid_grid_mask = cache["valid_grid_mask"]
            self.surface_point_rows = cache["surface_point_rows"]
            self.cdp_x_grid = cache["cdp_x_grid"]
            self.cdp_y_grid = cache["cdp_y_grid"]
            return

        surf_idx = np.load(self.surface_indices_path)
        iline_indices = surf_idx["iline_indices"]
        xline_indices = surf_idx["xline_indices"]
        valid_mask = surf_idx["valid_mask"]
        twt_values_actual = surf_idx["twt_values_actual"]
        grid_shape = tuple(surf_idx["grid_shape"])

        ds = xr.load_dataset(self.bolvanka_path)
        iline_values = ds["iline"].values
        xline_values = ds["xline"].values
        self.cdp_x_grid = ds["cdp_x"].values
        self.cdp_y_grid = ds["cdp_y"].values

        point_iline = np.load(self.data_folder / "iline.npy", mmap_mode="r")
        point_xline = np.load(self.data_folder / "xline.npy", mmap_mode="r")
        point_twt = np.load(self.data_folder / "twt.npy", mmap_mode="r")

        point_keys = _compose_point_keys(point_iline, point_xline, point_twt)
        sort_idx = np.argsort(point_keys)
        sorted_keys = point_keys[sort_idx]

        target_iline_vals = iline_values[iline_indices[valid_mask]]
        target_xline_vals = xline_values[xline_indices[valid_mask]]
        target_twt_vals = twt_values_actual[valid_mask]
        target_keys = _compose_point_keys(target_iline_vals, target_xline_vals, target_twt_vals)

        pos = np.searchsorted(sorted_keys, target_keys)
        in_bounds = pos < len(sorted_keys)
        matched = np.zeros_like(pos, dtype=bool)
        matched[in_bounds] = sorted_keys[pos[in_bounds]] == target_keys[in_bounds]
        matched_rows = np.full(target_keys.shape[0], -1, dtype=np.int64)
        matched_rows[matched] = sort_idx[pos[matched]]

        self.row_index_grid = np.full(grid_shape, -1, dtype=np.int64)
        valid_iline = iline_indices[valid_mask]
        valid_xline = xline_indices[valid_mask]
        self.row_index_grid[valid_iline[matched], valid_xline[matched]] = matched_rows[matched]
        self.valid_grid_mask = self.row_index_grid >= 0
        self.surface_point_rows = self.row_index_grid[self.valid_grid_mask]

        np.savez_compressed(
            self.cache_path,
            row_index_grid=self.row_index_grid,
            valid_grid_mask=self.valid_grid_mask,
            surface_point_rows=self.surface_point_rows,
            cdp_x_grid=self.cdp_x_grid,
            cdp_y_grid=self.cdp_y_grid,
        )

    @staticmethod
    def calculate_ari(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        valid_mask = (ground_truth != -1) & (~np.isnan(ground_truth))
        gt_valid = ground_truth[valid_mask].flatten()
        pred_valid = prediction[valid_mask].flatten()

        valid_idx = ~np.isnan(pred_valid)
        gt_valid = gt_valid[valid_idx]
        pred_valid = pred_valid[valid_idx]
        return adjusted_rand_score(gt_valid, pred_valid)

    def project_center_labels(
        self,
        center_labels: np.ndarray,
        stage1_labels: np.ndarray,
    ) -> np.ndarray:
        surface_labels = center_labels[np.asarray(stage1_labels[self.surface_point_rows], dtype=np.int64)]
        z_map = np.full(self.row_index_grid.shape, np.nan, dtype=float)
        z_map[self.valid_grid_mask] = surface_labels
        return z_map

    def score(
        self,
        center_labels: np.ndarray,
        stage1_labels: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        z_map = self.project_center_labels(center_labels=center_labels, stage1_labels=stage1_labels)
        ari = self.calculate_ari(self.facies_mask, z_map)
        return float(ari), z_map

    def save_plot(
        self,
        z_map: np.ndarray,
        output_path: str | Path,
        title: str,
    ) -> None:
        extractor = SeismicMapExtractor(max_distance=50.0, use_dask=True)
        fig, _ = extractor.plot_cluster_map_as_image(
            self.cdp_x_grid,
            self.cdp_y_grid,
            z_map,
            title=title,
        )
        fig.savefig(output_path)
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass

    def describe(self) -> dict:
        return {
            "surface_indices_path": str(self.surface_indices_path),
            "facies_mask_path": str(self.facies_mask_path),
            "bolvanka_path": str(self.bolvanka_path),
            "cache_path": str(self.cache_path),
            "grid_shape": list(self.row_index_grid.shape),
            "n_valid_surface_points": int(self.surface_point_rows.shape[0]),
        }
