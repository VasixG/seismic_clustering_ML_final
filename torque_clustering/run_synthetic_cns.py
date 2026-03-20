from __future__ import annotations

import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cns_module import cns_debug_run


def make_dataset_gaussians(seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1, n2, n3 = 40, 36, 34

    c1 = rng.normal(loc=[-4.0, 0.0], scale=[0.7, 0.8], size=(n1, 2))
    c2 = rng.normal(loc=[1.5, 3.0], scale=[0.8, 0.6], size=(n2, 2))
    c3 = rng.normal(loc=[4.8, -2.5], scale=[0.9, 0.7], size=(n3, 2))

    X = np.vstack([c1, c2, c3])
    y = np.concatenate(
        [np.zeros(n1, dtype=int), np.ones(n2, dtype=int), np.full(n3, 2, dtype=int)]
    )
    return X, y


def make_dataset_moons(seed: int = 321) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 60

    t1 = rng.uniform(0.0, np.pi, size=n)
    moon1 = np.column_stack([np.cos(t1), np.sin(t1)])
    moon1 += rng.normal(scale=0.06, size=moon1.shape)

    t2 = rng.uniform(0.0, np.pi, size=n)
    moon2 = np.column_stack([1.0 - np.cos(t2), -np.sin(t2) - 0.45])
    moon2 += rng.normal(scale=0.06, size=moon2.shape)

    X = np.vstack([moon1, moon2])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def make_dataset_anisotropic(seed: int = 999) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1, n2, n3 = 44, 42, 40

    c1 = rng.normal(size=(n1, 2)) @ np.array([[1.6, 0.9], [0.0, 0.25]]) + np.array([-3.5, 2.0])
    c2 = rng.normal(size=(n2, 2)) @ np.array([[0.3, -1.0], [1.2, 0.3]]) + np.array([0.5, -1.8])
    c3 = rng.normal(size=(n3, 2)) @ np.array([[0.7, 0.0], [0.5, 0.9]]) + np.array([3.8, 2.2])

    X = np.vstack([c1, c2, c3])
    y = np.concatenate(
        [np.zeros(n1, dtype=int), np.ones(n2, dtype=int), np.full(n3, 2, dtype=int)]
    )
    return X, y


def make_dataset_circles(seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_outer, n_inner, n_noise = 80, 70, 20

    t_outer = rng.uniform(0.0, 2.0 * np.pi, size=n_outer)
    outer = np.column_stack([np.cos(t_outer), np.sin(t_outer)])
    outer += rng.normal(scale=0.06, size=outer.shape)

    t_inner = rng.uniform(0.0, 2.0 * np.pi, size=n_inner)
    inner = 0.45 * np.column_stack([np.cos(t_inner), np.sin(t_inner)])
    inner += rng.normal(scale=0.05, size=inner.shape)

    noise = rng.uniform(low=[-1.4, -1.4], high=[1.4, 1.4], size=(n_noise, 2))

    X = np.vstack([outer, inner, noise])
    y = np.concatenate(
        [
            np.zeros(n_outer, dtype=int),
            np.ones(n_inner, dtype=int),
            np.full(n_noise, 2, dtype=int),
        ]
    )
    return X, y


def make_dataset_spiral(seed: int = 77) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 80

    t = np.linspace(0.5, 3.5 * np.pi, n)
    r = np.linspace(0.2, 2.0, n)

    s1 = np.column_stack([r * np.cos(t), r * np.sin(t)])
    s2 = np.column_stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)])
    s1 += rng.normal(scale=0.06, size=s1.shape)
    s2 += rng.normal(scale=0.06, size=s2.shape)

    X = np.vstack([s1, s2])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def _zscore(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def build_features(X: np.ndarray, feature_mode: str = "xy") -> np.ndarray:
    if feature_mode == "xy":
        return _zscore(X)
    if feature_mode == "xy_polar":
        x = X[:, 0]
        y = X[:, 1]
        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)
        X_ext = np.column_stack([x, y, r, np.sin(theta), np.cos(theta)])
        return _zscore(X_ext)
    raise ValueError("feature_mode must be 'xy' or 'xy_polar'.")


def _plot_input(X: np.ndarray, y: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=18, edgecolors="k", linewidths=0.2)
    ax.set_title("Stage 1: input data")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_knn_graph(X: np.ndarray, nns: np.ndarray, nn_cur: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], color="tab:blue", s=12, alpha=0.9)

    for i in range(X.shape[0]):
        for j in range(nn_cur):
            nb = int(nns[i, j])
            ax.plot([X[i, 0], X[nb, 0]], [X[i, 1], X[nb, 1]], color="black", alpha=0.06, linewidth=0.4)

    ax.set_title(f"Stage 2: kNN graph (nn={nn_cur})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_local_maxima(X: np.ndarray, uix: np.ndarray, k_best: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], color="lightgray", s=14)
    sel = uix[:k_best]
    ax.scatter(X[sel, 0], X[sel, 1], color="red", s=70, marker="x")
    for i, idx in enumerate(sel):
        ax.text(X[idx, 0], X[idx, 1], str(i + 1), fontsize=8)

    ax.set_title(f"Stage 3: local maxima centers (k={k_best})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_q_heatmap(Q: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Q, aspect="auto", cmap="viridis")
    ax.set_title("Stage 4: Q matrix (best nn, best lambda)")
    ax.set_xlabel("candidate cluster index")
    ax.set_ylabel("sample index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_vals(vals: np.ndarray, nn: np.ndarray, lams: np.ndarray, out: Path) -> None:
    best_k = np.argmax(vals.reshape(vals.shape[0], -1), axis=0)
    best_k = best_k.reshape(vals.shape[1], vals.shape[2]) + 1
    best_val = np.max(vals, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(best_k, aspect="auto", cmap="plasma")
    axes[0].set_title("Stage 5A: argmax_k vals[k, nn, lambda]")
    axes[0].set_xlabel("lambda index")
    axes[0].set_ylabel("nn index")
    axes[0].set_xticks(range(len(lams)))
    axes[0].set_xticklabels([f"{v:.3f}" for v in lams], rotation=45, ha="right")
    axes[0].set_yticks(range(len(nn)))
    axes[0].set_yticklabels([str(v) for v in nn])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(best_val, aspect="auto", cmap="cividis")
    axes[1].set_title("Stage 5B: max_k vals[k, nn, lambda]")
    axes[1].set_xlabel("lambda index")
    axes[1].set_ylabel("nn index")
    axes[1].set_xticks(range(len(lams)))
    axes[1].set_xticklabels([f"{v:.3f}" for v in lams], rotation=45, ha="right")
    axes[1].set_yticks(range(len(nn)))
    axes[1].set_yticklabels([str(v) for v in nn])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_probabilities(prob: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(prob, aspect="auto", cmap="magma")
    ax.set_title("Stage 6: membership probabilities")
    ax.set_xlabel("cluster")
    ax.set_ylabel("sample")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_final_clusters(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=18, edgecolors="k", linewidths=0.2)
    axes[0].set_title("True labels")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")

    axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap="tab20", s=18, edgecolors="k", linewidths=0.2)
    axes[1].set_title("Stage 7: CNS final clusters")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def run_one_dataset(
    name: str, X: np.ndarray, y: np.ndarray, out_root: Path, feature_mode: str = "xy"
) -> None:
    out_dir = out_root / f"{name}_{feature_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_feat = build_features(X, feature_mode=feature_mode)
    res = cns_debug_run(X_feat, kmax=18, iters=np.inf)
    k_best, nn_idx_1b, lam_idx_1b = res["best_parms"]

    nn_idx = nn_idx_1b - 1
    lam_idx = lam_idx_1b - 1
    nn_cur = int(res["nn"][nn_idx])

    _plot_input(X_feat[:, :2], y, out_dir / "stage_01_input.png")
    _plot_knn_graph(res["X"][:, :2], res["nns_index"], nn_cur, out_dir / "stage_02_knn_graph.png")
    _plot_local_maxima(
        res["X"][:, :2], res["uix_all"][nn_idx], int(k_best), out_dir / "stage_03_local_maxima.png"
    )
    _plot_q_heatmap(res["Qs"][:, : int(k_best), nn_idx, lam_idx], out_dir / "stage_04_Q_heatmap.png")
    _plot_vals(res["vals"], res["nn"], res["lams"], out_dir / "stage_05_vals_selection.png")
    _plot_probabilities(res["probabilities"], out_dir / "stage_06_probabilities.png")
    _plot_final_clusters(res["X"][:, :2], y, res["clusters"], out_dir / "stage_07_final_clusters.png")


if __name__ == "__main__":
    out_root = Path("torque_clustering/outputs/cns")
    out_root.mkdir(parents=True, exist_ok=True)

    Xg, yg = make_dataset_gaussians()
    Xm, ym = make_dataset_moons()
    Xa, ya = make_dataset_anisotropic()
    Xc, yc = make_dataset_circles()
    Xs, ys = make_dataset_spiral()

    run_one_dataset("gaussians", Xg, yg, out_root, feature_mode="xy")
    run_one_dataset("moons", Xm, ym, out_root, feature_mode="xy")
    run_one_dataset("anisotropic", Xa, ya, out_root, feature_mode="xy")

    run_one_dataset("circles", Xc, yc, out_root, feature_mode="xy")
    run_one_dataset("circles", Xc, yc, out_root, feature_mode="xy_polar")

    run_one_dataset("spiral", Xs, ys, out_root, feature_mode="xy")
    run_one_dataset("spiral", Xs, ys, out_root, feature_mode="xy_polar")

    print(f"Saved CNS stage visualizations to: {out_root.resolve()}")
