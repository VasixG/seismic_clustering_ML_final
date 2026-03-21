from __future__ import annotations

from pathlib import Path

import numpy as np

from tc_module import torque_clustering

TGAP_MODE = "auto_tgap"
ROBUST_MIN_TAU_NEXT = 1e-10
ROBUST_MAX_RANK_FRACTION = 0.95


def _partition_from_l(result, n: int, l_value: int) -> list[list[int]]:
    abnormal = set(result.tscl_indices[:l_value])
    kept = set(range(len(result.connections))) - abnormal

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for idx, conn in enumerate(result.connections):
        if idx not in kept:
            continue
        merged = list(conn.src_samples) + list(conn.nn_samples)
        if not merged:
            continue
        root = merged[0]
        for s in merged[1:]:
            union(root, s)

    groups: dict[int, list[int]] = {}
    for s in range(n):
        groups.setdefault(find(s), []).append(s)
    return list(groups.values())


def _auto_score_from_partition(partition: list[list[int]], n: int) -> tuple[float, int, float]:
    sizes = np.array([len(c) for c in partition], dtype=float)
    k = int(len(sizes))
    largest_ratio = float(np.max(sizes) / n)
    singleton_ratio = float(np.sum(sizes == 1.0) / n)
    tiny_ratio = float(np.sum(sizes <= 2.0) / n)
    k_cap = max(2, int(np.sqrt(n) * 0.60))
    k_pen = float(max(0, k - k_cap) / k_cap)

    score = 0.0
    score -= 2.00 * singleton_ratio
    score -= 0.90 * tiny_ratio
    score -= 1.50 * k_pen
    score -= 0.50 * max(0.0, largest_ratio - 0.70)
    score -= 0.80 * abs(k - k_cap) / k_cap
    return score, k, singleton_ratio


def save_score_curve_plot(dataset_name: str, X_proc: np.ndarray, result, out_root: Path) -> None:
    import matplotlib.pyplot as plt

    n = X_proc.shape[0]
    m = len(result.tscl_indices)
    tau_all = np.array([c.tau for c in result.connections], dtype=float)

    valid_pairs = [(i, v) for i, v in enumerate(result.tgap_values) if v is not None]
    cutoff = max(1, int(np.floor((m - 1) * ROBUST_MAX_RANK_FRACTION)))
    robust_pairs = []
    for i, v in valid_pairs:
        tau_next = tau_all[result.tscl_indices[i + 1]]
        if (i < cutoff) and (tau_next >= ROBUST_MIN_TAU_NEXT):
            robust_pairs.append((i, v))
    if not robust_pairs:
        robust_pairs = valid_pairs

    candidate_i = sorted({i for i, _ in robust_pairs})
    if len(candidate_i) > 60:
        picks = np.linspace(0, len(candidate_i) - 1, 60, dtype=int)
        candidate_i = [candidate_i[p] for p in picks]

    if not candidate_i:
        return

    l_values: list[int] = []
    scores: list[float] = []
    k_values: list[int] = []
    singleton_values: list[float] = []
    for i in candidate_i:
        l_value = i + 1
        partition = _partition_from_l(result, n, l_value)
        score, k_value, singleton_ratio = _auto_score_from_partition(partition, n)
        l_values.append(l_value)
        scores.append(score)
        k_values.append(k_value)
        singleton_values.append(singleton_ratio)

    l_arr = np.array(l_values)
    score_arr = np.array(scores)
    k_arr = np.array(k_values)
    singleton_arr = np.array(singleton_values)

    strict_l = 0
    if valid_pairs:
        strict_scores = np.array([v for _, v in valid_pairs], dtype=float)
        strict_l = int(np.argmax(strict_scores) + 1)

    best_l = int(l_arr[int(np.argmax(score_arr))])

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(l_arr, score_arr, color="tab:blue", linewidth=2, label="auto score(L)")
    ax1.set_xlabel("L (top-L abnormal edges removed)")
    ax1.set_ylabel("auto score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(l_arr, k_arr, color="tab:green", linewidth=1.4, alpha=0.85, label="k(L)")
    ax2.plot(
        l_arr,
        singleton_arr,
        color="tab:red",
        linewidth=1.4,
        alpha=0.85,
        label="singleton_ratio(L)",
    )
    ax2.set_ylabel("k / singleton_ratio", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    ax1.axvline(best_l, color="tab:blue", linestyle="--", alpha=0.9)
    ax1.axvline(result.selected_L, color="black", linestyle="-.", alpha=0.8)
    if strict_l > 0:
        ax1.axvline(strict_l, color="tab:orange", linestyle=":", alpha=0.9)

    ax1.set_title(f"{dataset_name}: score(L) vs threshold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    extra = [
        plt.Line2D([0], [0], color="tab:blue", linestyle="--", label=f"best L on curve = {best_l}"),
        plt.Line2D(
            [0], [0], color="black", linestyle="-.", label=f"selected_L_for_abnormal = {result.selected_L}"
        ),
        plt.Line2D([0], [0], color="tab:orange", linestyle=":", label=f"strict reference L = {strict_l}"),
    ]
    ax1.legend(lines1 + lines2 + extra, labels1 + labels2 + [h.get_label() for h in extra], loc="upper left")

    out = out_root / dataset_name / "score_vs_threshold_L.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def preprocess_normalize_and_deduplicate(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0.0] = 1.0
    X_norm = (X - mean) / std

    X_unique, first_idx = np.unique(X_norm, axis=0, return_index=True)
    order = np.argsort(first_idx)
    keep_idx = first_idx[order]

    removed = int(X.shape[0] - X_unique.shape[0])
    return X_norm[keep_idx], y[keep_idx], removed


def make_dataset_a(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1, n2, n3, n_noise = 22, 20, 18, 8

    c1 = rng.normal(loc=[-4.0, -1.0], scale=[0.55, 0.75], size=(n1, 2))
    c2 = rng.normal(loc=[0.5, 4.0], scale=[0.7, 0.6], size=(n2, 2))
    c3 = rng.normal(loc=[4.8, -2.5], scale=[0.8, 0.5], size=(n3, 2))
    noise = rng.uniform(low=[-7.5, -5.5], high=[7.5, 6.5], size=(n_noise, 2))

    X = np.vstack([c1, c2, c3, noise])
    y = np.concatenate(
        [
            np.zeros(n1, dtype=int),
            np.ones(n2, dtype=int),
            np.full(n3, 2, dtype=int),
            np.full(n_noise, 3, dtype=int),
        ]
    )
    return X, y


def make_dataset_b(seed: int = 13) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_per = 24

    t1 = rng.uniform(0.0, np.pi, size=n_per)
    moon1 = np.column_stack([np.cos(t1), np.sin(t1)])
    moon1 += rng.normal(scale=0.06, size=moon1.shape)

    t2 = rng.uniform(0.0, np.pi, size=n_per)
    moon2 = np.column_stack([1.0 - np.cos(t2), -np.sin(t2) - 0.45])
    moon2 += rng.normal(scale=0.06, size=moon2.shape)

    bridge = rng.normal(loc=[0.45, 0.05], scale=[0.12, 0.06], size=(8, 2))

    X = np.vstack([moon1, moon2, bridge])
    y = np.concatenate(
        [
            np.zeros(n_per, dtype=int),
            np.ones(n_per, dtype=int),
            np.full(8, 2, dtype=int),
        ]
    )
    return X, y


def make_dataset_c_big_gaussian(seed: int = 101) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1, n2, n3, n4, n_noise = 80, 75, 70, 65, 30

    c1 = rng.normal(loc=[-8.0, -1.5], scale=[0.8, 0.7], size=(n1, 2))
    c2 = rng.normal(loc=[-2.0, 6.0], scale=[0.9, 1.0], size=(n2, 2))
    c3 = rng.normal(loc=[4.5, 4.5], scale=[1.0, 0.9], size=(n3, 2))
    c4 = rng.normal(loc=[7.5, -3.5], scale=[0.9, 0.7], size=(n4, 2))
    noise = rng.uniform(low=[-10.0, -7.0], high=[10.0, 9.0], size=(n_noise, 2))

    X = np.vstack([c1, c2, c3, c4, noise])
    y = np.concatenate(
        [
            np.zeros(n1, dtype=int),
            np.ones(n2, dtype=int),
            np.full(n3, 2, dtype=int),
            np.full(n4, 3, dtype=int),
            np.full(n_noise, 4, dtype=int),
        ]
    )
    return X, y


def make_dataset_d_big_moons(seed: int = 202) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_per = 110
    n_bridge = 35

    t1 = rng.uniform(0.0, np.pi, size=n_per)
    moon1 = np.column_stack([np.cos(t1), np.sin(t1)])
    moon1 += rng.normal(scale=0.05, size=moon1.shape)

    t2 = rng.uniform(0.0, np.pi, size=n_per)
    moon2 = np.column_stack([1.0 - np.cos(t2), -np.sin(t2) - 0.5])
    moon2 += rng.normal(scale=0.05, size=moon2.shape)

    bridge = rng.normal(loc=[0.45, 0.05], scale=[0.14, 0.08], size=(n_bridge, 2))

    X = np.vstack([moon1, moon2, bridge])
    y = np.concatenate(
        [
            np.zeros(n_per, dtype=int),
            np.ones(n_per, dtype=int),
            np.full(n_bridge, 2, dtype=int),
        ]
    )
    return X, y


def run() -> None:
    out_root = Path("outputs")
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("synthetic_a", make_dataset_a()),
        ("synthetic_b", make_dataset_b()),
        ("synthetic_c_big_gaussian", make_dataset_c_big_gaussian()),
        ("synthetic_d_big_moons", make_dataset_d_big_moons()),
    ]

    for name, (X, y) in datasets:
        X_proc, y_proc, removed_duplicates = preprocess_normalize_and_deduplicate(X, y)

        result = torque_clustering(
            X=X_proc,
            true_labels=y_proc,
            save_dir=out_root,
            dataset_name=name,
            tgap_mode=TGAP_MODE,
            robust_min_tau_next=ROBUST_MIN_TAU_NEXT,
            robust_max_rank_fraction=ROBUST_MAX_RANK_FRACTION,
        )

        valid_tgaps = [v for v in result.tgap_values if v is not None]
        if valid_tgaps:
            max_tgap = float(np.max(valid_tgaps))
            best_i = int(np.argmax(np.array(valid_tgaps)))
            L = best_i + 1
        else:
            max_tgap = float("nan")
            L = 0

        final_sizes = sorted((len(c) for c in result.final_partition), reverse=True)
        halo_sizes = sorted((len(c) for c in result.halo_partition), reverse=True)

        summary_path = out_root / name / "summary.txt"
        summary = [
            f"dataset: {name}",
            f"n_samples: {X.shape[0]}",
            f"n_samples_after_preprocess: {X_proc.shape[0]}",
            f"duplicates_removed: {removed_duplicates}",
            f"tgap_mode: {result.tgap_mode}",
            f"n_connections: {len(result.connections)}",
            f"L_from_max_tgap_strict_reference: {L}",
            f"selected_L_for_abnormal: {result.selected_L}",
            f"max_tgap: {max_tgap:.8f}",
            f"abnormal_count: {len(result.abnormal_connection_indices)}",
            f"halo_count: {len(result.halo_connection_indices)}",
            f"final_partition_count: {len(result.final_partition)}",
            f"halo_partition_count: {len(result.halo_partition)}",
            f"final_top10_component_sizes: {final_sizes[:10]}",
            f"halo_top10_component_sizes: {halo_sizes[:10]}",
            f"mean_tau: {result.mean_tau:.8f}",
            f"mean_M: {result.mean_M:.8f}",
            f"mean_D: {result.mean_D:.8f}",
            f"mean_D_over_M: {result.mean_D_over_M:.8f}",
        ]
        summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
        save_score_curve_plot(name, X_proc, result, out_root)


if __name__ == "__main__":
    run()
