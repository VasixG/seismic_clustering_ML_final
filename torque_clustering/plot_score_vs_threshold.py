from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_synthetic_tc import make_dataset_d_big_moons, preprocess_normalize_and_deduplicate
from tc_module import torque_clustering


def main() -> None:
    X, y = make_dataset_d_big_moons()
    X, y, _ = preprocess_normalize_and_deduplicate(X, y)

    res = torque_clustering(
        X=X,
        true_labels=y,
        save_dir=None,
        dataset_name="synthetic_d_big_moons",
        tgap_mode="auto_tgap",
        robust_min_tau_next=1e-10,
        robust_max_rank_fraction=0.95,
    )

    n = X.shape[0]
    m = len(res.tscl_indices)
    tau_all = np.array([c.tau for c in res.connections], dtype=float)

    valid_pairs = [(i, v) for i, v in enumerate(res.tgap_values) if v is not None]
    cutoff = max(1, int(np.floor((m - 1) * 0.95)))
    robust_pairs = []
    for i, v in valid_pairs:
        tau_next = tau_all[res.tscl_indices[i + 1]]
        if (i < cutoff) and (tau_next >= 1e-10):
            robust_pairs.append((i, v))
    if not robust_pairs:
        robust_pairs = valid_pairs

    cand_i = sorted({i for i, _ in robust_pairs})
    if len(cand_i) > 60:
        picks = np.linspace(0, len(cand_i) - 1, 60, dtype=int)
        cand_i = [cand_i[p] for p in picks]

    def partition_from_l(l_value: int) -> list[list[int]]:
        abnormal = set(res.tscl_indices[:l_value])
        kept = set(range(len(res.connections))) - abnormal

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

        for idx, conn in enumerate(res.connections):
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

    ls: list[int] = []
    scores: list[float] = []
    ks: list[int] = []
    singleton_ratios: list[float] = []

    k_cap = max(2, int(np.sqrt(n)))
    for i in cand_i:
        l_value = i + 1
        part = partition_from_l(l_value)
        sizes = np.array([len(c) for c in part], dtype=float)

        k = int(len(sizes))
        largest_ratio = float(np.max(sizes) / n)
        singleton_ratio = float(np.sum(sizes == 1.0) / n)
        k_pen = float(max(0, k - k_cap) / k_cap)

        score = 0.0
        score -= 1.20 * singleton_ratio
        score -= 0.60 * k_pen
        score -= 0.40 * max(0.0, largest_ratio - 0.70)
        score -= 0.20 * abs(k - k_cap) / k_cap

        ls.append(l_value)
        scores.append(score)
        ks.append(k)
        singleton_ratios.append(singleton_ratio)

    ls_arr = np.array(ls)
    scores_arr = np.array(scores)
    ks_arr = np.array(ks)
    sing_arr = np.array(singleton_ratios)

    best_idx = int(np.argmax(scores_arr))
    best_l = int(ls_arr[best_idx])

    strict_l = 0
    if valid_pairs:
        strict_values = np.array([v for _, v in valid_pairs], dtype=float)
        strict_l = int(np.argmax(strict_values) + 1)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(ls_arr, scores_arr, color="tab:blue", linewidth=2, label="auto score(L)")
    ax1.set_xlabel("L (top-L abnormal edges removed)")
    ax1.set_ylabel("auto score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(ls_arr, ks_arr, color="tab:green", linewidth=1.5, alpha=0.85, label="k(L)")
    ax2.plot(
        ls_arr, sing_arr, color="tab:red", linewidth=1.5, alpha=0.85, label="singleton_ratio(L)"
    )
    ax2.set_ylabel("k / singleton_ratio", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")

    ax1.axvline(best_l, color="tab:blue", linestyle="--", alpha=0.9)
    ax1.axvline(res.selected_L, color="black", linestyle="-.", alpha=0.8)
    if strict_l > 0:
        ax1.axvline(strict_l, color="tab:orange", linestyle=":", alpha=0.9)

    ax1.set_title("synthetic_d_big_moons: score(L) vs threshold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    extra = [
        plt.Line2D([0], [0], color="tab:blue", linestyle="--", label=f"best L on curve = {best_l}"),
        plt.Line2D(
            [0], [0], color="black", linestyle="-.", label=f"selected_L_for_abnormal = {res.selected_L}"
        ),
        plt.Line2D([0], [0], color="tab:orange", linestyle=":", label=f"strict reference L = {strict_l}"),
    ]
    ax1.legend(lines1 + lines2 + extra, labels1 + labels2 + [h.get_label() for h in extra], loc="upper left")

    out = Path("outputs/synthetic_d_big_moons/score_vs_threshold_L.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)

    print(out.resolve())


if __name__ == "__main__":
    main()
