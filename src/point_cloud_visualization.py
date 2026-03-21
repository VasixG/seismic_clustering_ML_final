from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SECTIONS = [
    {"twt": 2480, "iline": 5694, "title": "Channels_A"},
    {"twt": 2498, "iline": 5784, "title": "Soil_slip_A"},
    {"twt": 2508, "title": "Fans_channels_A"},
    {"twt": 2528, "title": "Transition_2528"},
    {"twt": 2548, "title": "Transition_2548"},
    {"twt": 2568, "title": "Fans_channels_B"},
    {"twt": 2588, "title": "Transition_2588"},
    {"twt": 2608, "xline": 2693, "title": "Soil_slip_B"},
    {"twt": 2628, "title": "Deep_2628"},
    {"twt": 2498, "xline": 2693, "title": "Soil_slip_xline_2693_shallow"},
    {"twt": 2568, "iline": 5784, "title": "Soil_slip_iline_5784_mid"},
    {"twt": 2608, "iline": 5784, "title": "Soil_slip_iline_5784_deep"},
    {"twt": 2480, "xline": 2693, "title": "Channels_xline_2693"},
    {"twt": 2508, "iline": 5694, "title": "Channels_iline_5694"},
    {"twt": 2568, "xline": 2693, "title": "Fans_channels_xline_2693"},
]


def _nearest_unique_value(values: np.ndarray, target: float) -> float:
    unique_vals = np.unique(values)
    idx = int(np.argmin(np.abs(unique_vals - target)))
    return float(unique_vals[idx])


def _resolve_section_values(data_folder: Path, sections: list[dict]) -> list[dict]:
    iline = np.load(data_folder / "iline.npy", mmap_mode="r")
    xline = np.load(data_folder / "xline.npy", mmap_mode="r")
    twt = np.load(data_folder / "twt.npy", mmap_mode="r")

    resolved = []
    for section in sections:
        cur = dict(section)
        cur["twt"] = _nearest_unique_value(twt, section["twt"])
        if "iline" in section:
            cur["iline"] = int(_nearest_unique_value(iline, section["iline"]))
        if "xline" in section:
            cur["xline"] = int(_nearest_unique_value(xline, section["xline"]))
        resolved.append(cur)
    return resolved


def _save_horizontal_slice(
    cdp_x: np.ndarray,
    cdp_y: np.ndarray,
    twt: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    twt_value: float,
    title: str,
    figsize: tuple[int, int],
) -> None:
    mask = twt == twt_value
    if not np.any(mask):
        return

    fig, ax = plt.subplots(1, figsize=figsize)
    im = ax.scatter(
        cdp_x[mask],
        cdp_y[mask],
        c=labels[mask],
        s=mpl.rcParams["lines.markersize"] / 4,
        rasterized=True,
    )
    fig.colorbar(im)
    ax.set_title(f"Twt: {twt_value}, {title}")
    fig.savefig(out_path)
    plt.close(fig)


def _save_horizontal_vertical_slice(
    iline: np.ndarray,
    xline: np.ndarray,
    twt: np.ndarray,
    cdp_x: np.ndarray,
    cdp_y: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    section: dict,
    figsize: tuple[int, int],
) -> None:
    twt_value = section["twt"]
    title = section["title"]
    vert_sec_name = "iline" if "iline" in section else "xline"
    vert_sec_value = section[vert_sec_name]

    mask_h = twt == twt_value
    if vert_sec_name == "iline":
        mask_v = iline == vert_sec_value
        x_vert = xline[mask_v]
        xlabel = "xline"
    else:
        mask_v = xline == vert_sec_value
        x_vert = iline[mask_v]
        xlabel = "iline"

    if (not np.any(mask_h)) or (not np.any(mask_v)):
        return

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    im_h = ax[0].scatter(
        cdp_x[mask_h],
        cdp_y[mask_h],
        c=labels[mask_h],
        s=mpl.rcParams["lines.markersize"] / 4,
        rasterized=True,
    )
    fig.colorbar(im_h)
    ax[0].set_title(f"Twt: {twt_value}, {title}")

    im_v = ax[1].scatter(
        x_vert,
        twt[mask_v],
        c=labels[mask_v],
        s=mpl.rcParams["lines.markersize"] / 4,
        rasterized=True,
    )
    fig.colorbar(im_v)
    ax[1].invert_yaxis()
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("twt")
    ax[1].set_title(f"{vert_sec_name}: {vert_sec_value}, {title}")

    fig.savefig(out_path)
    plt.close(fig)


def save_point_cloud_sections(
    data_folder: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    sections: Iterable[dict] | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    data_folder = Path(data_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections_list = list(sections) if sections is not None else list(DEFAULT_SECTIONS)
    sections_list = _resolve_section_values(data_folder, sections_list)

    iline = np.load(data_folder / "iline.npy", mmap_mode="r")
    xline = np.load(data_folder / "xline.npy", mmap_mode="r")
    twt = np.load(data_folder / "twt.npy", mmap_mode="r")
    cdp_x = np.load(data_folder / "cdp_x.npy", mmap_mode="r")
    cdp_y = np.load(data_folder / "cdp_y.npy", mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    if labels.shape[0] != twt.shape[0]:
        raise ValueError(
            f"Labels length {labels.shape[0]} does not match point cloud length {twt.shape[0]}"
        )

    for section in sections_list:
        twt_value = section["twt"]
        title = section["title"]
        if ("iline" not in section) and ("xline" not in section):
            out_path = output_dir / f"twt_{int(twt_value)}_{title}.pdf"
            _save_horizontal_slice(
                cdp_x=cdp_x,
                cdp_y=cdp_y,
                twt=twt,
                labels=labels,
                out_path=out_path,
                twt_value=twt_value,
                title=title,
                figsize=figsize,
            )
        else:
            vert_sec_name = "iline" if "iline" in section else "xline"
            out_path = output_dir / (
                f"twt_{int(twt_value)}_{vert_sec_name}_{section[vert_sec_name]}_{title}.pdf"
            )
            _save_horizontal_vertical_slice(
                iline=iline,
                xline=xline,
                twt=twt,
                cdp_x=cdp_x,
                cdp_y=cdp_y,
                labels=labels,
                out_path=out_path,
                section=section,
                figsize=figsize,
            )
