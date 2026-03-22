from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass
class ADMMCNSResult:
    P_init: np.ndarray
    Z_init: np.ndarray
    P: np.ndarray
    Z: np.ndarray
    U: np.ndarray
    labels: np.ndarray
    objective: List[float]
    objective_aug: List[float]
    primal_r1: List[float]
    primal_r2: List[float]
    dual_s1: List[float]
    dual_s2: List[float]
    snapshots: Dict[int, np.ndarray]
    W: sparse.csr_matrix
    L: sparse.csr_matrix
    edge_index: np.ndarray
    edge_dist: np.ndarray
    torque_edge_weights: np.ndarray


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


def _build_knn_graph(
    X: np.ndarray,
    n_neighbors: int,
    sigma: float | None = None,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n < 2:
        raise ValueError("kNN graph requires at least 2 points.")
    n_neighbors = max(1, min(int(n_neighbors), n - 1))
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
    W = W.tocsr()

    deg = np.asarray(W.sum(axis=1)).ravel()
    L = sparse.diags(deg) - W

    upper = sparse.triu(W, k=1).tocoo()
    edge_index = np.column_stack([upper.row, upper.col])
    edge_dist = D[upper.row, upper.col]

    return W, L.tocsr(), edge_index, edge_dist


def _build_incidence(n: int, edge_index: np.ndarray) -> sparse.csr_matrix:
    m = edge_index.shape[0]
    rows = np.repeat(np.arange(m), 2)
    cols = np.column_stack([edge_index[:, 0], edge_index[:, 1]]).reshape(-1)
    vals = np.tile(np.array([1.0, -1.0]), m)
    B = sparse.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr()
    return B


def _init_soft_assignments(X: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    center_idx = rng.choice(n, size=k, replace=False)
    centers = X[center_idx].copy()  # (k, d)
    U = centers.T.copy()  # (d, k), consistent with ADMM updates

    D = _pairwise_distances(np.vstack([X, centers]))[:n, n:]
    tau = max(1e-6, float(np.median(D)))
    S = np.exp(-(D**2) / (tau**2))
    P0 = S / np.sum(S, axis=1, keepdims=True)
    return P0, U


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


def _objective_augmented(
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
    rho1: float,
    rho2: float,
    Lambda: np.ndarray,
    Y: np.ndarray,
    B: sparse.csr_matrix,
) -> float:
    base = _objective(X, P, Z, U, V, L, edge_weights, alpha, beta, gamma, lam)
    r1 = P - Z + Lambda
    r2 = V - (B @ Z) + Y
    return base + 0.5 * rho1 * float(np.sum(r1 * r1)) + 0.5 * rho2 * float(np.sum(r2 * r2))


def admm_cns_torque(
    X: np.ndarray,
    k: int,
    n_neighbors: int = 12,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    lam: float = 2.0,
    rho1: float = 1.0,
    rho2: float = 1.0,
    max_iter: int = 120,
    tol: float = 1e-4,
    seed: int = 42,
    snapshot_iters: Tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50, 100),
) -> ADMMCNSResult:
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if n < 2:
        raise ValueError("Torque clustering requires at least 2 points.")
    if k < 1:
        raise ValueError("Number of clusters k must be >= 1.")
    if k > n:
        k = n
    if n_neighbors >= n:
        n_neighbors = max(1, n - 1)

    W, L, edge_index, edge_dist = _build_knn_graph(X, n_neighbors=n_neighbors)
    B = _build_incidence(n, edge_index)

    deg = np.asarray(W.sum(axis=1)).ravel()
    deg_i = deg[edge_index[:, 0]]
    deg_j = deg[edge_index[:, 1]]
    mass = (deg_i * deg_j) / (deg_i + deg_j + 1e-12)
    torque_edge_weights = mass * edge_dist

    P, U = _init_soft_assignments(X, k=k, seed=seed)
    Z = P.copy()
    P_init = P.copy()
    Z_init = Z.copy()

    m = edge_index.shape[0]
    V = np.zeros((m, k), dtype=float)
    Lambda = np.zeros((n, k), dtype=float)
    Y = np.zeros((m, k), dtype=float)

    A_p = (2.0 * alpha) * L + (2.0 * lam + rho1) * sparse.eye(n, format="csr")

    objective: List[float] = []
    objective_aug: List[float] = []
    primal_r1: List[float] = []
    primal_r2: List[float] = []
    dual_s1: List[float] = []
    dual_s2: List[float] = []
    snapshots: Dict[int, np.ndarray] = {0: np.argmax(Z, axis=1)}

    Z_prev = Z.copy()
    V_prev = V.copy()

    BtB = (B.T @ B).toarray()

    for it in range(1, max_iter + 1):
        # P-update
        A_rhs = (2.0 * lam + rho1) * Z - rho1 * Lambda
        P_new = np.zeros_like(P)
        for c in range(k):
            P_new[:, c] = spla.spsolve(A_p, A_rhs[:, c])
        P = _project_simplex_rows(P_new)

        # Z-update via Sylvester equation: A Z + Z C = F
        A_mat = (2.0 * lam + rho1) * np.eye(n) + rho2 * BtB
        C_mat = 2.0 * beta * (U.T @ U)
        F_mat = (2.0 * lam + rho1) * P + rho1 * Lambda + rho2 * (B.T @ (V + Y)) + 2.0 * beta * (X @ U)

        Z = linalg.solve_sylvester(A_mat, C_mat, F_mat)
        Z = _project_simplex_rows(Z)

        # U-update
        U = (X.T @ Z) @ np.linalg.inv(Z.T @ Z + 1e-8 * np.eye(k))

        # V-update (group shrinkage per edge)
        Q = (B @ Z) - Y
        norms = np.sqrt(np.sum(Q * Q, axis=1))
        thr = gamma * torque_edge_weights / rho2
        scale = np.maximum(1.0 - thr / (norms + 1e-12), 0.0)
        V = Q * scale[:, None]

        # Dual updates
        Lambda = Lambda + (P - Z)
        BZ = B @ Z
        Y = Y + (V - BZ)

        # Residuals
        r1 = float(np.linalg.norm(P - Z))
        r2 = float(np.linalg.norm(V - BZ))
        s1 = float(rho1 * np.linalg.norm(Z - Z_prev))
        s2 = float(rho2 * np.linalg.norm((B.T @ (V - V_prev))))

        primal_r1.append(r1)
        primal_r2.append(r2)
        dual_s1.append(s1)
        dual_s2.append(s2)

        obj = _objective(
            X=X,
            P=P,
            Z=Z,
            U=U,
            V=V,
            L=L,
            edge_weights=torque_edge_weights,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lam=lam,
        )
        objective.append(obj)
        obj_aug = _objective_augmented(
            X=X,
            P=P,
            Z=Z,
            U=U,
            V=V,
            L=L,
            edge_weights=torque_edge_weights,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lam=lam,
            rho1=rho1,
            rho2=rho2,
            Lambda=Lambda,
            Y=Y,
            B=B,
        )
        objective_aug.append(obj_aug)

        if it in snapshot_iters:
            snapshots[it] = np.argmax(Z, axis=1)

        Z_prev = Z.copy()
        V_prev = V.copy()

        if max(r1, r2, s1, s2) < tol:
            break

    if max_iter not in snapshots:
        snapshots[max_iter] = np.argmax(Z, axis=1)

    labels = np.argmax(Z, axis=1)

    return ADMMCNSResult(
        P_init=P_init,
        Z_init=Z_init,
        P=P,
        Z=Z,
        U=U,
        labels=labels,
        objective=objective,
        objective_aug=objective_aug,
        primal_r1=primal_r1,
        primal_r2=primal_r2,
        dual_s1=dual_s1,
        dual_s2=dual_s2,
        snapshots=snapshots,
        W=W,
        L=L,
        edge_index=edge_index,
        edge_dist=edge_dist,
        torque_edge_weights=torque_edge_weights,
    )
