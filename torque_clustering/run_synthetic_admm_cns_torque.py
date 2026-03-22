from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from admm_cns_torque_module import admm_cns_torque


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
    moon1 = np.column_stack([np.cos(t1), np.sin(t1)])
    moon1 += rng.normal(scale=0.06, size=moon1.shape)

    t2 = rng.uniform(0.0, np.pi, size=n)
    moon2 = np.column_stack([1.0 - np.cos(t2), -np.sin(t2) - 0.5])
    moon2 += rng.normal(scale=0.06, size=moon2.shape)

    X = np.vstack([moon1, moon2])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def make_dataset_circles(seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_outer, n_inner = 90, 80

    t_outer = rng.uniform(0.0, 2.0 * np.pi, size=n_outer)
    outer = np.column_stack([np.cos(t_outer), np.sin(t_outer)])
    outer += rng.normal(scale=0.05, size=outer.shape)

    t_inner = rng.uniform(0.0, 2.0 * np.pi, size=n_inner)
    inner = 0.5 * np.column_stack([np.cos(t_inner), np.sin(t_inner)])
    inner += rng.normal(scale=0.045, size=inner.shape)

    X = np.vstack([outer, inner])
    y = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])
    return X, y


def make_dataset_spiral(seed: int = 77) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 90

    t = np.linspace(0.5, 3.7 * np.pi, n)
    r = np.linspace(0.25, 2.1, n)

    s1 = np.column_stack([r * np.cos(t), r * np.sin(t)])
    s2 = np.column_stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)])
    s1 += rng.normal(scale=0.06, size=s1.shape)
    s2 += rng.normal(scale=0.06, size=s2.shape)

    X = np.vstack([s1, s2])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd


def _build_features(X: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode == "xy":
        return _zscore(X)
    if feature_mode == "xy_polar":
        x = X[:, 0]
        y = X[:, 1]
        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)
        X_ext = np.column_stack([x, y, r, np.sin(theta), np.cos(theta)])
        return _zscore(X_ext)
    raise ValueError("feature_mode must be 'xy' or 'xy_polar'")


def _plot_stage01_input(X: np.ndarray, y: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=16, edgecolors="k", linewidths=0.2)
    ax.set_title("Stage 1: Input data")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_stage02_graph(
    X: np.ndarray, edge_index: np.ndarray, out: Path, max_edges: int = 1200
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], s=10, color="tab:blue")

    m = edge_index.shape[0]
    if m > max_edges:
        pick = np.linspace(0, m - 1, max_edges, dtype=int)
        edges = edge_index[pick]
    else:
        edges = edge_index

    for i, j in edges:
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="black", linewidth=0.35, alpha=0.09)

    ax.set_title("Stage 2: kNN graph")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_stage03_init_soft(P: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(P, aspect="auto", cmap="viridis")
    ax.set_title("Stage 3: Initial soft assignment P")
    ax.set_xlabel("cluster")
    ax.set_ylabel("sample")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_stage04_snapshots(
    X: np.ndarray, y_true: np.ndarray, snapshots: dict[int, np.ndarray], out: Path
) -> None:
    steps = sorted(snapshots.keys())[:6]
    cols = 3
    rows = int(np.ceil((len(steps) + 1) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=12, edgecolors="k", linewidths=0.15)
    axes[0].set_title("True labels")

    for idx, step in enumerate(steps, start=1):
        axes[idx].scatter(
            X[:, 0],
            X[:, 1],
            c=snapshots[step],
            cmap="tab20",
            s=12,
            edgecolors="k",
            linewidths=0.15,
        )
        axes[idx].set_title(f"Iter {step}")

    for i in range(len(steps) + 1, len(axes)):
        axes[i].axis("off")

    for ax in axes[: len(steps) + 1]:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_stage05_convergence(
    objective: list[float],
    objective_aug: list[float],
    r1: list[float],
    r2: list[float],
    s1: list[float],
    s2: list[float],
    out: Path,
) -> None:
    t = np.arange(1, len(objective) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(t, objective, color="tab:blue", linewidth=2, label="primal objective")
    axes[0].plot(
        t, objective_aug, color="tab:green", linewidth=1.6, alpha=0.9, label="augmented objective"
    )
    axes[0].set_title("Stage 5A: Objective")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("J")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, r1, label="primal r1=||P-Z||", color="tab:red")
    axes[1].plot(t, r2, label="primal r2=||V-BZ||", color="tab:orange")
    axes[1].plot(t, s1, label="dual s1", color="tab:green")
    axes[1].plot(t, s2, label="dual s2", color="tab:purple")
    axes[1].set_yscale("log")
    axes[1].set_title("Stage 5B: ADMM residuals")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("log residual")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_stage06_final_soft(Z: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Z, aspect="auto", cmap="magma")
    ax.set_title("Stage 6: Final consensus assignment Z")
    ax.set_xlabel("cluster")
    ax.set_ylabel("sample")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_stage07_final_clusters(
    X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=16, edgecolors="k", linewidths=0.2)
    axes[0].set_title("True labels")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")

    axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap="tab20", s=16, edgecolors="k", linewidths=0.2)
    axes[1].set_title("Stage 7: Final ADMM CNS+Torque clusters")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_stage08_torque_edges(
    X: np.ndarray, edge_index: np.ndarray, edge_w: np.ndarray, out: Path, top_q: float = 0.9
) -> None:
    thr = np.quantile(edge_w, top_q)
    mask = edge_w >= thr
    edges = edge_index[mask]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], s=11, color="lightgray")

    for i, j in edges:
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="red", alpha=0.35, linewidth=0.7)

    ax.set_title("Stage 8: Top torque-weighted edges")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def run_one(
    name: str,
    X_raw: np.ndarray,
    y: np.ndarray,
    out_root: Path,
    feature_mode: str = "xy",
    n_neighbors: int = 8,
    gamma: float = 0.10,
    alpha: float = 1.0,
    beta: float = 0.01,
    lam: float = 2.0,
) -> None:
    X = _build_features(X_raw, feature_mode=feature_mode)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_stage01_input(X, y, out_dir / "stage_01_input.png")

    result = admm_cns_torque(
        X=X,
        k=len(np.unique(y)),
        n_neighbors=n_neighbors,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lam=lam,
        rho1=1.0,
        rho2=1.0,
        max_iter=120,
        tol=1e-4,
        seed=42,
        snapshot_iters=(1, 2, 5, 10, 20, 60, 100),
    )

    _plot_stage02_graph(X, result.edge_index, out_dir / "stage_02_knn_graph.png")
    _plot_stage03_init_soft(result.P_init, out_dir / "stage_03_init_P.png")
    _plot_stage04_snapshots(X, y, result.snapshots, out_dir / "stage_04_iter_snapshots.png")
    _plot_stage05_convergence(
        result.objective,
        result.objective_aug,
        result.primal_r1,
        result.primal_r2,
        result.dual_s1,
        result.dual_s2,
        out_dir / "stage_05_convergence.png",
    )
    _plot_stage06_final_soft(result.Z, out_dir / "stage_06_final_Z.png")
    _plot_stage07_final_clusters(X, y, result.labels, out_dir / "stage_07_final_clusters.png")
    _plot_stage08_torque_edges(
        X, result.edge_index, result.torque_edge_weights, out_dir / "stage_08_torque_edges.png"
    )


if __name__ == "__main__":
    out_root = Path("torque_clustering/outputs/admm_cns_torque")
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = {
        "gaussians": make_dataset_gaussians(),
        "moons": make_dataset_moons(),
        "circles": make_dataset_circles(),
        "spiral": make_dataset_spiral(),
    }

    run_one(
        "gaussians", *datasets["gaussians"], out_root, feature_mode="xy", n_neighbors=8, gamma=0.10
    )
    run_one("moons", *datasets["moons"], out_root, feature_mode="xy", n_neighbors=8, gamma=0.10)
    run_one(
        "circles",
        *datasets["circles"],
        out_root,
        feature_mode="xy",
        n_neighbors=3,
        gamma=0.20,
        alpha=0.1,
        beta=0.01,
        lam=1.0,
    )
    run_one(
        "spiral",
        *datasets["spiral"],
        out_root,
        feature_mode="xy",
        n_neighbors=3,
        gamma=0.16,
        alpha=0.1,
        beta=0.01,
        lam=1.0,
    )

    print(f"Saved visualizations to: {out_root.resolve()}")
