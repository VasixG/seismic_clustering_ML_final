from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as spla

from tc_module import TCResult, torque_clustering


@dataclass
class ADMMCNSV2Result:
    Z: np.ndarray
    P: np.ndarray
    U: np.ndarray
    labels: np.ndarray
    selected_k: int
    selected_L: int
    outer_objective: List[float]
    inner_objective: List[List[float]]
    vals_k_history: List[np.ndarray]
    snapshots: Dict[str, np.ndarray]
    tc_result: TCResult
    W_final: sparse.csr_matrix
    edge_index_final: np.ndarray


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _project_simplex_rows(M: np.ndarray) -> np.ndarray:
    out = np.zeros_like(M)
    for i in range(M.shape[0]):
        v = M[i]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0]
        if len(rho) == 0:
            theta = 0.0
        else:
            r = rho[-1]
            theta = (cssv[r] - 1.0) / (r + 1)
        out[i] = np.maximum(v - theta, 0.0)
    return out


def _build_knn_graph(X: np.ndarray, n_neighbors: int, sigma: float | None = None) -> sparse.csr_matrix:
    n = X.shape[0]
    D = _pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    nn_idx = np.argsort(D, axis=1)[:, :n_neighbors]

    rows = np.repeat(np.arange(n), n_neighbors)
    cols = nn_idx.reshape(-1)
    dvals = D[rows, cols]

    if sigma is None:
        sigma = float(np.median(dvals[dvals > 0]))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

    wvals = np.exp(-(dvals**2) / (sigma**2))
    W = sparse.coo_matrix((wvals, (rows, cols)), shape=(n, n)).tocsr()
    W = 0.5 * (W + W.T)
    return W.tocsr()


def _edge_index_from_W(W: sparse.csr_matrix) -> np.ndarray:
    upper = sparse.triu(W, k=1).tocoo()
    if upper.nnz == 0:
        return np.zeros((0, 2), dtype=int)
    return np.column_stack([upper.row, upper.col]).astype(int)


def _build_incidence(n: int, edge_index: np.ndarray) -> sparse.csr_matrix:
    m = edge_index.shape[0]
    rows = np.repeat(np.arange(m), 2)
    cols = np.column_stack([edge_index[:, 0], edge_index[:, 1]]).reshape(-1)
    vals = np.tile(np.array([1.0, -1.0]), m)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr()


def _init_soft_assignments(X: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    center_idx = rng.choice(n, size=k, replace=False)
    centers = X[center_idx].copy()
    U = centers.T.copy()

    D = _pairwise_distances(np.vstack([X, centers]))[:n, n:]
    tau = max(1e-6, float(np.median(D)))
    S = np.exp(-(D**2) / (tau**2))
    P = S / np.sum(S, axis=1, keepdims=True)
    return P, U


def _objective(
    X: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    L: sparse.csr_matrix,
    edge_weights: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    lam: float,
) -> float:
    cns = alpha * float(np.sum(P * (L @ P)))
    fit = beta * float(np.sum((X - Z @ U.T) ** 2))
    torque = gamma * float(np.sum(edge_weights * np.sqrt(np.sum(V * V, axis=1))))
    agree = lam * float(np.sum((P - Z) ** 2))
    return cns + fit + torque + agree


def _admm_inner(
    X: np.ndarray,
    W: sparse.csr_matrix,
    P: np.ndarray,
    Z: np.ndarray,
    U: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    lam: float,
    rho1: float,
    rho2: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    n, k = Z.shape
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = sparse.diags(deg) - W

    edge_index = _edge_index_from_W(W)
    B = _build_incidence(n, edge_index)

    if edge_index.shape[0] == 0:
        return P, Z, U, [_objective(X, P, Z, U, np.zeros((0, k)), L, np.zeros(0), alpha, beta, gamma, lam)]

    D_full = _pairwise_distances(X)
    edge_dist = D_full[edge_index[:, 0], edge_index[:, 1]]

    deg_i = deg[edge_index[:, 0]]
    deg_j = deg[edge_index[:, 1]]
    mass = (deg_i * deg_j) / (deg_i + deg_j + 1e-12)
    edge_weights = mass * edge_dist

    V = np.zeros((edge_index.shape[0], k), dtype=float)
    Lambda = np.zeros((n, k), dtype=float)
    Y = np.zeros((edge_index.shape[0], k), dtype=float)

    A_p = (2.0 * alpha) * L + (2.0 * lam + rho1) * sparse.eye(n, format="csr")
    BtB = (B.T @ B).toarray()

    inner_obj: List[float] = []

    for _ in range(max_iter):
        rhs = (2.0 * lam + rho1) * Z - rho1 * Lambda
        P_new = np.zeros_like(P)
        for c in range(k):
            P_new[:, c] = spla.spsolve(A_p, rhs[:, c])
        P = _project_simplex_rows(P_new)

        A_mat = (2.0 * lam + rho1) * np.eye(n) + rho2 * BtB
        C_mat = 2.0 * beta * (U.T @ U)
        F_mat = (2.0 * lam + rho1) * P + rho1 * Lambda + rho2 * (B.T @ (V + Y)) + 2.0 * beta * (X @ U)
        Z = linalg.solve_sylvester(A_mat, C_mat, F_mat)
        Z = _project_simplex_rows(Z)

        U = (X.T @ Z) @ np.linalg.inv(Z.T @ Z + 1e-8 * np.eye(k))

        BZ = B @ Z
        Q = BZ - Y
        norms = np.sqrt(np.sum(Q * Q, axis=1))
        thr = gamma * edge_weights / rho2
        scale = np.maximum(1.0 - thr / (norms + 1e-12), 0.0)
        V = Q * scale[:, None]

        Lambda = Lambda + (P - Z)
        Y = Y + (V - BZ)

        obj = _objective(X, P, Z, U, V, L, edge_weights, alpha, beta, gamma, lam)
        inner_obj.append(obj)

        r1 = np.linalg.norm(P - Z)
        r2 = np.linalg.norm(V - BZ)
        if max(r1, r2) < tol:
            break

    return P, Z, U, inner_obj


def _oq_like(Q: np.ndarray, q: int) -> np.ndarray:
    cs = np.sum(Q, axis=0)
    sim = Q.T @ Q
    np.fill_diagonal(sim, np.inf)

    ord_idx = np.zeros(q, dtype=int)
    ord_idx[0] = int(np.argmax(cs))
    ds = sim[ord_idx[0], :].copy()
    ds[ord_idx[0]] = np.inf

    for k in range(1, q):
        ord_idx[k] = int(np.argmin(ds / (cs**2 + 1e-12)))
        sim_row = sim[ord_idx[k], :]
        ds = ds * (ds > sim_row) + sim_row * (ds <= sim_row)

    return Q[:, ord_idx]


def _cns_like_select_k(Z: np.ndarray, nn: int, lam_cns: float) -> Tuple[int, np.ndarray, np.ndarray]:
    n, kmax = Z.shape
    Q = _oq_like(Z, kmax)

    vals = np.zeros(kmax, dtype=float)
    vals[0] = 0.0

    ref = (1.0 - lam_cns) * (1.0 / max(nn, 1) + 1.0 / n - 2.0 / np.sqrt(n * max(nn, 1)))
    if abs(ref) < 1e-12:
        ref = 1.0

    for k in range(2, kmax + 1):
        qk = Q[:, :k]
        prob = 1.0 / k + qk - np.sum(qk, axis=1, keepdims=True) / k
        top = np.mean(np.max(prob, axis=1)) - (n - k + k * k) / n / k
        vals[k - 1] = top / ref

    k_best = int(np.argmax(vals) + 1)
    if k_best > 1:
        qk = Q[:, :k_best]
        prob = 1.0 / k_best + qk - np.sum(qk, axis=1, keepdims=True) / k_best
    else:
        prob = np.ones((n, 1), dtype=float)
    return k_best, vals, prob


def _prune_graph_with_tc(
    X: np.ndarray,
    Z: np.ndarray,
    W: sparse.csr_matrix,
    eta: float,
    max_remove_frac: float,
) -> Tuple[sparse.csr_matrix, TCResult, int]:
    n = X.shape[0]
    D = _pairwise_distances(X)
    S = Z @ Z.T
    D_hybrid = D * (1.0 + eta * (1.0 - S))
    np.fill_diagonal(D_hybrid, 0.0)

    tc_res = torque_clustering(S=D_hybrid, tgap_mode="robust_tgap")
    abnormal_idx = tc_res.abnormal_connection_indices
    L = tc_res.selected_L

    if len(abnormal_idx) == 0 or L == 0:
        return W, tc_res, 0

    target = min(len(abnormal_idx), max(1, int(max_remove_frac * W.nnz / 2)))

    W_lil = W.tolil(copy=True)
    removed = 0
    for conn_idx in abnormal_idx[:target]:
        a, b = tc_res.connections[conn_idx].min_pair
        if a < 0 or b < 0:
            continue
        if W_lil[a, b] != 0:
            W_lil[a, b] = 0.0
            W_lil[b, a] = 0.0
            removed += 1

    W_new = W_lil.tocsr()
    W_new.eliminate_zeros()
    return W_new, tc_res, removed


def admm_cns_torque_v2(
    X: np.ndarray,
    k_init: int,
    n_neighbors: int = 8,
    alpha: float = 0.8,
    beta: float = 0.8,
    gamma: float = 0.2,
    lam: float = 1.2,
    rho1: float = 1.0,
    rho2: float = 2.0,
    inner_iters: int = 50,
    outer_iters: int = 4,
    tol: float = 1e-4,
    cns_lam_for_k: float = 0.3,
    prune_eta: float = 1.0,
    prune_max_remove_frac: float = 0.10,
    seed: int = 42,
) -> ADMMCNSV2Result:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    W = _build_knn_graph(X, n_neighbors=n_neighbors)

    P, U = _init_soft_assignments(X, k=k_init, seed=seed)
    Z = P.copy()

    outer_objective: List[float] = []
    inner_objective: List[List[float]] = []
    vals_k_history: List[np.ndarray] = []
    snapshots: Dict[str, np.ndarray] = {"outer0": np.argmax(Z, axis=1)}

    tc_last = torque_clustering(S=_pairwise_distances(X), tgap_mode="robust_tgap")
    selected_k = k_init
    selected_L = 0

    for out in range(1, outer_iters + 1):
        P, Z, U, inner_obj = _admm_inner(
            X=X,
            W=W,
            P=P,
            Z=Z,
            U=U,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lam=lam,
            rho1=rho1,
            rho2=rho2,
            max_iter=inner_iters,
            tol=tol,
        )
        inner_objective.append(inner_obj)

        k_best, vals_k, prob_k = _cns_like_select_k(Z, nn=n_neighbors, lam_cns=cns_lam_for_k)
        vals_k_history.append(vals_k)
        selected_k = k_best

        if k_best != Z.shape[1]:
            Z = prob_k
            P = Z.copy()
            U = (X.T @ Z) @ np.linalg.inv(Z.T @ Z + 1e-8 * np.eye(Z.shape[1]))

        W, tc_last, removed = _prune_graph_with_tc(
            X=X,
            Z=Z,
            W=W,
            eta=prune_eta,
            max_remove_frac=prune_max_remove_frac,
        )
        selected_L = tc_last.selected_L

        deg = np.asarray(W.sum(axis=1)).ravel()
        Lmat = sparse.diags(deg) - W
        obj_outer = float(alpha * np.sum(P * (Lmat @ P)) + beta * np.sum((X - Z @ U.T) ** 2) + lam * np.sum((P - Z) ** 2))
        outer_objective.append(obj_outer)

        snapshots[f"outer{out}"] = np.argmax(Z, axis=1)
        snapshots[f"outer{out}_removed_edges"] = np.array([removed], dtype=int)

        if W.nnz == 0:
            break

    labels = np.argmax(Z, axis=1)
    edge_index_final = _edge_index_from_W(W)

    return ADMMCNSV2Result(
        Z=Z,
        P=P,
        U=U,
        labels=labels,
        selected_k=selected_k,
        selected_L=selected_L,
        outer_objective=outer_objective,
        inner_objective=inner_objective,
        vals_k_history=vals_k_history,
        snapshots=snapshots,
        tc_result=tc_last,
        W_final=W,
        edge_index_final=edge_index_final,
    )
