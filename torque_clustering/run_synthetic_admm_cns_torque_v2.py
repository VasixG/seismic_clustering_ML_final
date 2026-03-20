from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from admm_cns_torque_v2_module import admm_cns_torque_v2


def make_dataset_gaussians(seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1, n2, n3 = 60, 55, 50
    c1 = rng.normal(loc=[-4.2, 0.0], scale=[0.7, 0.9], size=(n1, 2))
    c2 = rng.normal(loc=[0.8, 3.4], scale=[0.75, 0.65], size=(n2, 2))
    c3 = rng.normal(loc=[4.9, -2.3], scale=[0.85, 0.7], size=(n3, 2))
    X = np.vstack([c1, c2, c3])
    y = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int), np.full(n3, 2, dtype=int)])
    return X, y


def make_dataset_moons(seed: int = 321) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 90
    t1 = rng.uniform(0.0, np.pi, size=n)
    moon1 = np.column_stack([np.cos(t1), np.sin(t1)]) + rng.normal(scale=0.06, size=(n, 2))
    t2 = rng.uniform(0.0, np.pi, size=n)
    moon2 = np.column_stack([1.0 - np.cos(t2), -np.sin(t2) - 0.5]) + rng.normal(scale=0.06, size=(n, 2))
    X = np.vstack([moon1, moon2])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def make_dataset_circles(seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_outer, n_inner = 90, 80
    t_outer = rng.uniform(0.0, 2.0 * np.pi, size=n_outer)
    outer = np.column_stack([np.cos(t_outer), np.sin(t_outer)]) + rng.normal(scale=0.05, size=(n_outer, 2))
    t_inner = rng.uniform(0.0, 2.0 * np.pi, size=n_inner)
    inner = 0.5 * np.column_stack([np.cos(t_inner), np.sin(t_inner)]) + rng.normal(scale=0.045, size=(n_inner, 2))
    X = np.vstack([outer, inner])
    y = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])
    return X, y


def make_dataset_spiral(seed: int = 77) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 90
    t = np.linspace(0.5, 3.7 * np.pi, n)
    r = np.linspace(0.25, 2.1, n)
    s1 = np.column_stack([r * np.cos(t), r * np.sin(t)]) + rng.normal(scale=0.06, size=(n, 2))
    s2 = np.column_stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)]) + rng.normal(scale=0.06, size=(n, 2))
    X = np.vstack([s1, s2])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd


def _plot_input(X: np.ndarray, y: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=16, edgecolors="k", linewidths=0.2)
    ax.set_title("Stage 1: Input")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_graph(X: np.ndarray, edges: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], s=11, color="tab:blue")
    m = edges.shape[0]
    if m > 1400:
        pick = np.linspace(0, m - 1, 1400, dtype=int)
        edges = edges[pick]
    for i, j in edges:
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="black", alpha=0.08, linewidth=0.35)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_outer_snapshots(X: np.ndarray, y_true: np.ndarray, snapshots: dict[str, np.ndarray], out: Path) -> None:
    keys = [k for k in snapshots.keys() if k.startswith("outer") and "removed" not in k]
    keys = sorted(keys, key=lambda s: int(s.replace("outer", "")))
    cols = 3
    rows = int(np.ceil((len(keys) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=12, edgecolors="k", linewidths=0.15)
    axes[0].set_title("True labels")

    for idx, key in enumerate(keys, start=1):
        axes[idx].scatter(X[:, 0], X[:, 1], c=snapshots[key], cmap="tab20", s=12, edgecolors="k", linewidths=0.15)
        axes[idx].set_title(key)

    for i in range(len(keys) + 1, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_inner_objectives(inner_obj: list[list[float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, arr in enumerate(inner_obj, start=1):
        if len(arr) == 0:
            continue
        t = np.arange(1, len(arr) + 1)
        ax.plot(t, arr, linewidth=1.7, label=f"outer {i}")
    ax.set_title("Stage 4: Inner ADMM objective")
    ax.set_xlabel("inner iteration")
    ax.set_ylabel("J")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_vals_k(vals_hist: list[np.ndarray], out: Path) -> None:
    if len(vals_hist) == 0:
        return
    max_k = max(v.shape[0] for v in vals_hist)
    M = np.full((len(vals_hist), max_k), np.nan)
    for i, v in enumerate(vals_hist):
        M[i, : v.shape[0]] = v

    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(M, aspect="auto", cmap="cividis")
    ax.set_title("Stage 5: CNS-like vals(k) per outer iteration")
    ax.set_xlabel("k")
    ax.set_ylabel("outer iteration")
    ax.set_xticks(range(max_k))
    ax.set_xticklabels(range(1, max_k + 1))
    ax.set_yticks(range(len(vals_hist)))
    ax.set_yticklabels(range(1, len(vals_hist) + 1))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_final(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out: Path, subtitle: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=16, edgecolors="k", linewidths=0.2)
    axes[0].set_title("True labels")
    axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap="tab20", s=16, edgecolors="k", linewidths=0.2)
    axes[1].set_title(subtitle)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_outer_objective(vals: list[float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    t = np.arange(1, len(vals) + 1)
    ax.plot(t, vals, marker="o", linewidth=1.8)
    ax.set_title("Stage 6: Outer objective per cycle")
    ax.set_xlabel("outer cycle")
    ax.set_ylabel("J_outer")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def run_one(name: str, X_raw: np.ndarray, y: np.ndarray, out_root: Path) -> None:
    X = _zscore(X_raw)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_input(X, y, out_dir / "stage_01_input.png")

    res = admm_cns_torque_v2(
        X=X,
        k_init=max(4, len(np.unique(y)) + 2),
        n_neighbors=6,
        alpha=0.8,
        beta=0.6,
        gamma=0.2,
        lam=1.2,
        rho1=1.0,
        rho2=2.0,
        inner_iters=45,
        outer_iters=4,
        tol=1e-4,
        cns_lam_for_k=0.3,
        prune_eta=1.0,
        prune_max_remove_frac=0.08,
        seed=42,
    )

    _plot_graph(X, res.edge_index_final, out_dir / "stage_02_graph_after_pruning.png", "Stage 2: Graph after CNS+TC pruning")
    _plot_outer_snapshots(X, y, res.snapshots, out_dir / "stage_03_outer_snapshots.png")
    _plot_inner_objectives(res.inner_objective, out_dir / "stage_04_inner_objectives.png")
    _plot_vals_k(res.vals_k_history, out_dir / "stage_05_vals_k.png")
    _plot_outer_objective(res.outer_objective, out_dir / "stage_06_outer_objective.png")
    _plot_final(
        X,
        y,
        res.labels,
        out_dir / "stage_07_final_clusters.png",
        f"Final clusters (k={res.selected_k}, L={res.selected_L})",
    )


if __name__ == "__main__":
    out_root = Path("torque_clustering/outputs/admm_cns_torque_v2")
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = {
        "gaussians": make_dataset_gaussians(),
        "moons": make_dataset_moons(),
        "circles": make_dataset_circles(),
        "spiral": make_dataset_spiral(),
    }

    for name, (X, y) in datasets.items():
        run_one(name, X, y, out_root)

    print(f"Saved v2 visualizations to: {out_root.resolve()}")
