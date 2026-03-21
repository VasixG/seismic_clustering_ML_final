from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class Connection:
    idx: int
    iteration: int
    src_cluster_id: int
    nn_cluster_id: int
    src_samples: Tuple[int, ...]
    nn_samples: Tuple[int, ...]
    min_pair: Tuple[int, int]
    mass_src: int
    mass_nn: int
    M: float
    D: float
    tau: Optional[float] = None


@dataclass
class TCResult:
    final_partition: List[List[int]]
    halo_partition: List[List[int]]
    abnormal_connection_indices: List[int]
    halo_connection_indices: List[int]
    connections: List[Connection]
    tgap_values: List[Optional[float]]
    tscl_indices: List[int]
    mean_tau: float
    mean_M: float
    mean_D: float
    mean_D_over_M: float
    selected_L: int
    tgap_mode: str


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _single_link_distance(
    dist_matrix: np.ndarray, cluster_a: Sequence[int], cluster_b: Sequence[int]
) -> Tuple[float, Tuple[int, int]]:
    a_idx = np.array(cluster_a, dtype=int)
    b_idx = np.array(cluster_b, dtype=int)
    sub = dist_matrix[np.ix_(a_idx, b_idx)]
    flat_pos = int(np.argmin(sub))
    ra, rb = np.unravel_index(flat_pos, sub.shape)
    pair = (int(a_idx[ra]), int(b_idx[rb]))
    return float(sub[ra, rb]), pair


def _components_from_edges(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
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

    for a, b in edges:
        union(a, b)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def _components_cluster_level(
    cluster_ids: Sequence[int], edges: Sequence[Tuple[int, int]]
) -> List[List[int]]:
    idx_of = {cid: i for i, cid in enumerate(cluster_ids)}
    parent = list(range(len(cluster_ids)))
    rank = [0] * len(cluster_ids)

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

    for ca, cb in edges:
        union(idx_of[ca], idx_of[cb])

    groups: Dict[int, List[int]] = {}
    for i, cid in enumerate(cluster_ids):
        r = find(i)
        groups.setdefault(r, []).append(cid)
    return list(groups.values())


def _plot_state(
    X: np.ndarray,
    true_labels: np.ndarray,
    partition: Sequence[Sequence[int]],
    out_path: Path,
    title_right: str,
    edges: Optional[Sequence[Tuple[int, int]]] = None,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_labels = np.full(X.shape[0], -1, dtype=int)
    for cid, comp in enumerate(partition):
        cluster_labels[np.array(comp, dtype=int)] = cid

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap="tab10", s=28, edgecolors="k", linewidths=0.2)
    axes[0].set_title("True classes")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")

    axes[1].scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="tab20", s=28, edgecolors="k", linewidths=0.2)
    if edges:
        for a, b in edges:
            axes[1].plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]], color="black", linewidth=0.6, alpha=0.6)
    axes[1].set_title(title_right)
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def torque_clustering(
    X: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    dataset_name: str = "dataset",
    tgap_mode: str = "paper_strict",
    robust_min_tau_next: float = 1e-10,
    robust_max_rank_fraction: float = 0.95,
) -> TCResult:
    if X is None and S is None:
        raise ValueError("Provide either X or S.")
    if X is not None and S is not None:
        raise ValueError("Provide only one of X or S.")
    if X is not None:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
    else:
        n = int(np.asarray(S).shape[0])

    if X is None and save_dir is not None:
        raise ValueError("Visualization requires X.")
    if tgap_mode not in {"paper_strict", "robust_tgap", "auto_tgap"}:
        raise ValueError("tgap_mode must be 'paper_strict', 'robust_tgap', or 'auto_tgap'.")
    if robust_max_rank_fraction <= 0.0 or robust_max_rank_fraction > 1.0:
        raise ValueError("robust_max_rank_fraction must be in (0, 1].")
    if robust_min_tau_next < 0.0:
        raise ValueError("robust_min_tau_next must be >= 0.")

    if true_labels is None:
        true_labels = np.zeros(n, dtype=int)
    else:
        true_labels = np.asarray(true_labels)

    if S is None:
        dist_matrix = _pairwise_distances(X)
    else:
        dist_matrix = np.asarray(S, dtype=float)
        if dist_matrix.shape != (n, n):
            raise ValueError("S must be shape (n, n).")

    # 2) Initialize clusters Γ = {{x1}, {x2}, ..., {xn}}.
    cluster_counter = n
    clusters: Dict[int, Set[int]] = {i: {i} for i in range(n)}

    # 3) Initialize accumulators.
    M_all: List[float] = []
    D_all: List[float] = []
    C_all: List[Connection] = []

    if save_dir is not None:
        _plot_state(
            X=X,  # type: ignore[arg-type]
            true_labels=true_labels,
            partition=[sorted(list(v)) for v in clusters.values()],
            out_path=Path(save_dir) / dataset_name / "step_02_init_clusters.png",
            title_right="Step 2: initial clusters",
            edges=None,
        )

    iteration = 0
    # 4) Main loop: while |Γ| > 2.
    while len(clusters) > 2:
        iteration += 1

        # 4.1) Masses.
        theta: Dict[int, int] = {cid: len(samples) for cid, samples in clusters.items()}

        # 4.2) 1-nearest cluster for each cluster.
        nearest: Dict[int, Tuple[int, float, Tuple[int, int]]] = {}
        ids = list(clusters.keys())
        sorted_samples: Dict[int, List[int]] = {
            cid: sorted(list(samples)) for cid, samples in clusters.items()
        }
        pair_dist: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ci, cj = ids[i], ids[j]
                pair_dist[(ci, cj)] = _single_link_distance(
                    dist_matrix, sorted_samples[ci], sorted_samples[cj]
                )

        for cid in ids:
            best_nid: Optional[int] = None
            best_dist = np.inf
            best_pair = (-1, -1)
            for nid in ids:
                if nid == cid:
                    continue
                key = (cid, nid) if cid < nid else (nid, cid)
                d, pair = pair_dist[key]
                if d < best_dist:
                    best_dist = d
                    best_nid = nid
                    best_pair = pair if cid < nid else (pair[1], pair[0])
            if best_nid is not None:
                nearest[cid] = (best_nid, best_dist, best_pair)

        # 4.3-4.4) Connections by Eq.1 + properties Eq.3/4.
        iteration_edges_samples: List[Tuple[int, int]] = []
        iteration_edges_clusters: List[Tuple[int, int]] = []
        for cid in ids:
            if cid not in nearest:
                continue
            nn_id, d_cluster, pair = nearest[cid]
            if theta[cid] <= theta[nn_id]:
                conn = Connection(
                    idx=len(C_all),
                    iteration=iteration,
                    src_cluster_id=cid,
                    nn_cluster_id=nn_id,
                    src_samples=tuple(sorted(list(clusters[cid]))),
                    nn_samples=tuple(sorted(list(clusters[nn_id]))),
                    min_pair=pair,
                    mass_src=theta[cid],
                    mass_nn=theta[nn_id],
                    M=float(theta[cid] * theta[nn_id]),
                    D=float(d_cluster * d_cluster),
                )
                C_all.append(conn)
                M_all.append(conn.M)
                D_all.append(conn.D)
                iteration_edges_samples.append(pair)
                iteration_edges_clusters.append((cid, nn_id))

        # 4.5) Γ := Φ(G) from connected components.
        components = _components_cluster_level(ids, iteration_edges_clusters)
        new_clusters: Dict[int, Set[int]] = {}
        for comp in components:
            merged_samples: Set[int] = set()
            for old_id in comp:
                merged_samples.update(clusters[old_id])
            new_clusters[cluster_counter] = merged_samples
            cluster_counter += 1
        clusters = new_clusters

        if save_dir is not None:
            _plot_state(
                X=X,  # type: ignore[arg-type]
                true_labels=true_labels,
                partition=[sorted(list(v)) for v in clusters.values()],
                out_path=Path(save_dir) / dataset_name / f"step_04_iter_{iteration:02d}.png",
                title_right=f"Step 4.5: clusters after iter {iteration}",
                edges=iteration_edges_samples,
            )

    # 5) Torque τi = Mi * Di for all connections.
    Tau_all: List[float] = []
    for i, conn in enumerate(C_all):
        tau = M_all[i] * D_all[i]
        conn.tau = float(tau)
        Tau_all.append(float(tau))

    # 6) TSCL sorted by τ descending.
    tscl_indices = list(np.argsort(np.array(Tau_all))[::-1]) if Tau_all else []

    # 7) TGap via Eq.6-9.
    if Tau_all:
        mean_tau = float(np.mean(Tau_all))
        mean_M = float(np.mean(M_all))
        mean_D = float(np.mean(D_all))
    else:
        mean_tau = 0.0
        mean_M = 0.0
        mean_D = 0.0

    large_c: Set[int] = set()
    for j in range(len(C_all)):
        if Tau_all[j] >= mean_tau and M_all[j] >= mean_M and D_all[j] >= mean_D:
            large_c.add(j)

    tgap_values: List[Optional[float]] = []
    m = len(tscl_indices)
    for i in range(m - 1):
        top_i = set(tscl_indices[: i + 1])
        if len(large_c) == 0:
            omega_i = 0.0
        else:
            omega_i = float(len(large_c.intersection(top_i))) / float(len(large_c))
        tau_i = Tau_all[tscl_indices[i]]
        tau_next = Tau_all[tscl_indices[i + 1]]
        if tau_next != 0:
            tgap_values.append(float(omega_i * (tau_i / tau_next)))
        else:
            tgap_values.append(None)

    # 8) Max TGap => abnormal top-L connections.
    def _partition_from_kept_indices(kept_conn_indices: Set[int]) -> List[List[int]]:
        parent_loc = list(range(n))
        rank_loc = [0] * n

        def find_loc(x: int) -> int:
            while parent_loc[x] != x:
                parent_loc[x] = parent_loc[parent_loc[x]]
                x = parent_loc[x]
            return x

        def union_loc(a: int, b: int) -> None:
            ra = find_loc(a)
            rb = find_loc(b)
            if ra == rb:
                return
            if rank_loc[ra] < rank_loc[rb]:
                parent_loc[ra] = rb
            elif rank_loc[ra] > rank_loc[rb]:
                parent_loc[rb] = ra
            else:
                parent_loc[rb] = ra
                rank_loc[ra] += 1

        for idx, conn in enumerate(C_all):
            if idx not in kept_conn_indices:
                continue
            merged = list(conn.src_samples) + list(conn.nn_samples)
            if not merged:
                continue
            root = merged[0]
            for s in merged[1:]:
                union_loc(root, s)

        groups_loc: Dict[int, List[int]] = {}
        for s in range(n):
            groups_loc.setdefault(find_loc(s), []).append(s)
        return list(groups_loc.values())

    valid_pairs = [(i, v) for i, v in enumerate(tgap_values) if v is not None]
    cutoff = int(np.floor((m - 1) * robust_max_rank_fraction))
    if cutoff < 1:
        cutoff = 1
    robust_pairs = []
    for i, v in valid_pairs:
        tau_next = Tau_all[tscl_indices[i + 1]]
        if (i < cutoff) and (tau_next >= robust_min_tau_next):
            robust_pairs.append((i, v))
    if not robust_pairs:
        robust_pairs = valid_pairs

    if tgap_mode == "paper_strict":
        selected_pairs = valid_pairs
        if selected_pairs:
            best_i = max(selected_pairs, key=lambda x: x[1])[0]
            L = best_i + 1
        else:
            L = 0
    elif tgap_mode == "robust_tgap":
        selected_pairs = robust_pairs
        if selected_pairs:
            best_i = max(selected_pairs, key=lambda x: x[1])[0]
            L = best_i + 1
        else:
            L = 0
    else:
        # auto_tgap: choose L from robust candidates by partition-quality score.
        candidate_i = sorted({i for i, _ in robust_pairs})
        if not candidate_i:
            L = 0
        else:
            # Use up to 60 evenly spaced candidates for speed.
            if len(candidate_i) > 60:
                picks = np.linspace(0, len(candidate_i) - 1, 60, dtype=int)
                candidate_i = [candidate_i[p] for p in picks]

            best_score = -np.inf
            best_L = 0
            k_cap = max(2, int(np.sqrt(n) * 0.60))

            for i in candidate_i:
                cand_L = i + 1
                abnormal_tmp = set(tscl_indices[:cand_L])
                kept_tmp = set(range(len(C_all))) - abnormal_tmp
                part = _partition_from_kept_indices(kept_tmp)
                sizes = np.array([len(c) for c in part], dtype=float)
                k = int(len(sizes))
                largest_ratio = float(np.max(sizes) / n)
                singleton_ratio = float(np.sum(sizes == 1.0) / n)
                tiny_ratio = float(np.sum(sizes <= 2.0) / n)
                k_pen = float(max(0, k - k_cap) / k_cap)

                score = 0.0
                score -= 2.00 * singleton_ratio
                score -= 0.90 * tiny_ratio
                score -= 1.50 * k_pen
                score -= 0.50 * max(0.0, largest_ratio - 0.70)
                score -= 0.80 * abs(k - k_cap) / k_cap

                if score > best_score:
                    best_score = score
                    best_L = cand_L
            L = best_L

    abnormal_indices = tscl_indices[:L]

    # 9) Final partition after removing abnormal connections.
    abnormal_set = set(abnormal_indices)
    final_edges_for_plot: List[Tuple[int, int]] = []
    parent = list(range(n))
    rank = [0] * n

    def find_sample(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_sample(a: int, b: int) -> None:
        ra = find_sample(a)
        rb = find_sample(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i, conn in enumerate(C_all):
        if i in abnormal_set:
            continue
        merged = list(conn.src_samples) + list(conn.nn_samples)
        if not merged:
            continue
        root = merged[0]
        for s in merged[1:]:
            union_sample(root, s)
        final_edges_for_plot.append(conn.min_pair)

    groups: Dict[int, List[int]] = {}
    for s in range(n):
        groups.setdefault(find_sample(s), []).append(s)
    final_partition = list(groups.values())

    if save_dir is not None:
        _plot_state(
            X=X,  # type: ignore[arg-type]
            true_labels=true_labels,
            partition=final_partition,
            out_path=Path(save_dir) / dataset_name / "step_09_final_partition.png",
            title_right="Step 9: final partition (abnormal removed)",
            edges=final_edges_for_plot,
        )

    # 10) Halo_C by Eq.10.
    ratios: List[float] = []
    for i in range(len(C_all)):
        if M_all[i] == 0:
            ratios.append(np.inf)
        else:
            ratios.append(float(D_all[i] / M_all[i]))
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0

    halo_indices: List[int] = []
    for k in range(len(C_all)):
        cond = (
            (M_all[k] <= mean_M)
            and (D_all[k] >= mean_D)
            and (ratios[k] >= mean_ratio)
        )
        if cond:
            halo_indices.append(k)

    # 11) Cluster halo after additionally removing Halo_C.
    halo_set = set(halo_indices)
    halo_edges_for_plot: List[Tuple[int, int]] = []
    parent_h = list(range(n))
    rank_h = [0] * n

    def find_h(x: int) -> int:
        while parent_h[x] != x:
            parent_h[x] = parent_h[parent_h[x]]
            x = parent_h[x]
        return x

    def union_h(a: int, b: int) -> None:
        ra = find_h(a)
        rb = find_h(b)
        if ra == rb:
            return
        if rank_h[ra] < rank_h[rb]:
            parent_h[ra] = rb
        elif rank_h[ra] > rank_h[rb]:
            parent_h[rb] = ra
        else:
            parent_h[rb] = ra
            rank_h[ra] += 1

    for i, conn in enumerate(C_all):
        if i in abnormal_set or i in halo_set:
            continue
        merged = list(conn.src_samples) + list(conn.nn_samples)
        if not merged:
            continue
        root = merged[0]
        for s in merged[1:]:
            union_h(root, s)
        halo_edges_for_plot.append(conn.min_pair)

    groups_h: Dict[int, List[int]] = {}
    for s in range(n):
        groups_h.setdefault(find_h(s), []).append(s)
    halo_partition = list(groups_h.values())

    if save_dir is not None:
        _plot_state(
            X=X,  # type: ignore[arg-type]
            true_labels=true_labels,
            partition=halo_partition,
            out_path=Path(save_dir) / dataset_name / "step_11_halo_partition.png",
            title_right="Step 11: halo partition",
            edges=halo_edges_for_plot,
        )

    return TCResult(
        final_partition=[sorted(comp) for comp in final_partition],
        halo_partition=[sorted(comp) for comp in halo_partition],
        abnormal_connection_indices=abnormal_indices,
        halo_connection_indices=halo_indices,
        connections=C_all,
        tgap_values=tgap_values,
        tscl_indices=tscl_indices,
        mean_tau=mean_tau,
        mean_M=mean_M,
        mean_D=mean_D,
        mean_D_over_M=mean_ratio,
        selected_L=L,
        tgap_mode=tgap_mode,
    )
