from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass
class CNSResult:
    probabilities: np.ndarray
    clusters: np.ndarray
    vals: np.ndarray
    nn: np.ndarray
    lams: np.ndarray
    X: np.ndarray


def _get_knn_indices_distances(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-NN via full distance matrix to mirror FNN::get.knn behavior."""
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nn_index = np.argsort(dist, axis=1)[:, :k]
    nn_dist = np.take_along_axis(dist, nn_index, axis=1)
    return nn_index, nn_dist


def oQ(Q: np.ndarray, q: int | None = None) -> np.ndarray:
    if q is None:
        q = Q.shape[1]
    cs = np.sum(Q, axis=0)
    sim = Q.T @ Q
    np.fill_diagonal(sim, np.inf)

    ord_idx = np.zeros(q, dtype=int)
    ord_idx[0] = int(np.argmax(cs))

    ds = sim[ord_idx[0], :].copy()
    ds[ord_idx[0]] = np.inf

    for k in range(1, q):
        ord_idx[k] = int(np.argmin(ds / (cs**2)))
        sim_row = sim[ord_idx[k], :]
        ds = ds * (ds > sim_row) + sim_row * (ds <= sim_row)

    return Q[:, ord_idx]


def appr_solv(W: sparse.spmatrix, F0: sparse.spmatrix, lams: np.ndarray, iter: int = 100) -> np.ndarray:
    out = np.zeros((F0.shape[0], F0.shape[1], len(lams)), dtype=float)
    add = F0.tocsc().copy()

    for it in range(0, iter + 1):
        add_dense = add.toarray()
        for j in range(len(lams)):
            out[:, :, j] = out[:, :, j] + add_dense * (lams[j] ** it)
        add = W @ add

    add_dense = add.toarray()
    for j in range(len(lams)):
        out[:, :, j] = out[:, :, j] + add_dense * (lams[j] ** (it + 1)) / (1.0 - lams[j])

    return out


def _row_maxs(a: np.ndarray) -> np.ndarray:
    return np.max(a, axis=1)


def CNS(
    X: np.ndarray,
    kmax: int = 50,
    nn: np.ndarray | None = None,
    lams: np.ndarray | None = None,
    distance: str = "Euclidean",
    iters: float = np.inf,
) -> CNSResult:
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    n = X.shape[0]

    if nn is None:
        nn = np.ceil(np.log(n)).astype(int) * np.arange(1, 5, dtype=int)
    else:
        nn = np.asarray(nn, dtype=int)

    if lams is None:
        lams = (1.0 / np.sqrt(n)) * np.arange(1, 6, dtype=float)
    else:
        lams = np.asarray(lams, dtype=float)

    if distance == "cosine":
        X = X - np.tile(np.mean(X, axis=0), (n, 1))
        X = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))

    max_nn = int(np.max(nn))
    nn_index, nn_dist = _get_knn_indices_distances(X, max_nn - 1)

    nns_index = np.column_stack((np.arange(n, dtype=int), nn_index))
    nns_dist = np.column_stack((np.zeros(n, dtype=float), nn_dist))
    _ = nns_dist
    nns = nns_index

    vals = np.zeros((kmax, len(nn), len(lams)), dtype=float)
    Qs = np.zeros((n, kmax, len(nn), len(lams)), dtype=float)

    if np.isinf(iters):
        for ni in range(len(nn)):
            nn_cur = int(nn[ni])

            row_idx = np.repeat(np.arange(n, dtype=int), nn_cur)
            col_idx = nns[:, :nn_cur].reshape(-1)
            data = np.ones_like(row_idx, dtype=float)
            L = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n, n)).tocsr()

            cs = np.asarray(L.sum(axis=0)).ravel()

            uix = np.zeros(kmax, dtype=int)
            uix[0] = int(np.argmax(cs))
            ds = np.sqrt(np.sum((X - X[uix[0], :]) ** 2, axis=1))
            for k in range(1, kmax):
                uix[k] = int(np.argmax(ds * cs))
                dnew = np.sqrt(np.sum((X - X[uix[k], :]) ** 2, axis=1))
                ds = ds * (ds < dnew) + dnew * (dnew <= ds)

            pix = sparse.coo_matrix(
                (
                    np.ones(kmax, dtype=float),
                    (uix, np.arange(kmax, dtype=int)),
                ),
                shape=(n, kmax),
            ).tocsr()

            for li in range(len(lams)):
                lam = float(lams[li])
                I = sparse.eye(n, format="csc") - ((1.0 - lam) * L / nn_cur).tocsc()

                solved = spla.spsolve(I, pix.toarray())
                Qcur = oQ(lam * np.asarray(solved), min(kmax, len(uix)))
                qcols = min(kmax, len(uix))
                Qs[:, :qcols, ni, li] = np.asarray(Qcur)

                vals[0, ni, li] = 0.0
                for k in range(2, kmax + 1):
                    ref = (1.0 - lam) * (1.0 / nn_cur + 1.0 / n - 2.0 / np.sqrt(n * nn_cur))
                    qslice = Qs[:, :k, ni, li]
                    expr = 1.0 / k + qslice - np.sum(qslice, axis=1, keepdims=True) / k
                    top = np.mean(_row_maxs(expr)) - (n - k + k**2) / n / k
                    vals[k - 1, ni, li] = top / ref
    else:
        for ni in range(len(nn)):
            nn_cur = int(nn[ni])

            row_idx = np.repeat(np.arange(n, dtype=int), nn_cur)
            col_idx = nns[:, :nn_cur].reshape(-1)
            data = np.ones_like(row_idx, dtype=float)
            L = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n, n)).tocsr()

            cs = np.asarray(L.sum(axis=0)).ravel()

            uix = np.zeros(kmax, dtype=int)
            uix[0] = int(np.argmax(cs))
            ds = np.sqrt(np.sum((X - X[uix[0], :]) ** 2, axis=1))
            for k in range(1, kmax):
                uix[k] = int(np.argmax(ds * cs))
                dnew = np.sqrt(np.sum((X - X[uix[k], :]) ** 2, axis=1))
                ds = ds * (ds < dnew) + dnew * (dnew <= ds)

            pix = sparse.coo_matrix(
                (
                    np.ones(kmax, dtype=float),
                    (uix, np.arange(kmax, dtype=int)),
                ),
                shape=(n, kmax),
            ).tocsr()

            qs = appr_solv(L / nn_cur, pix, 1.0 - lams, iter=int(iters))

            for li in range(len(lams)):
                Qcur = oQ(lams[li] * qs[:, :, li])
                qcols = min(kmax, len(uix))
                Qs[:, :qcols, ni, li] = Qcur[:, :qcols]

                vals[0, ni, li] = 0.0
                for k in range(2, kmax + 1):
                    ref = (1.0 - lams[li]) * (1.0 / nn_cur + 1.0 / n - 2.0 / np.sqrt(n * nn_cur))
                    qslice = Qs[:, :k, ni, li]
                    expr = 1.0 / k + qslice - np.sum(qslice, axis=1, keepdims=True) / k
                    top = np.mean(_row_maxs(expr)) - (n - k + k**2) / n / k
                    vals[k - 1, ni, li] = top / ref

    vals[np.isnan(vals)] = -1.0

    flat_idx = int(np.argmax(vals))
    parms = np.unravel_index(flat_idx, vals.shape)
    k_best = parms[0] + 1

    if k_best > 1:
        qslice = Qs[:, :k_best, parms[1], parms[2]]
        prob = 1.0 / k_best + qslice - np.sum(qslice, axis=1, keepdims=True) / k_best
    else:
        prob = np.ones((n, 1), dtype=float)

    clusters = np.argmax(prob, axis=1) + 1

    return CNSResult(
        probabilities=prob,
        clusters=clusters,
        vals=vals,
        nn=nn,
        lams=lams,
        X=X,
    )


def cns_debug_run(
    X: np.ndarray,
    kmax: int = 50,
    nn: np.ndarray | None = None,
    lams: np.ndarray | None = None,
    distance: str = "Euclidean",
    iters: float = np.inf,
) -> dict[str, Any]:
    """Same CNS computations with stage artifacts for plotting."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if nn is None:
        nn = np.ceil(np.log(n)).astype(int) * np.arange(1, 5, dtype=int)
    else:
        nn = np.asarray(nn, dtype=int)

    if lams is None:
        lams = (1.0 / np.sqrt(n)) * np.arange(1, 6, dtype=float)
    else:
        lams = np.asarray(lams, dtype=float)

    if distance == "cosine":
        X = X - np.tile(np.mean(X, axis=0), (n, 1))
        X = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))

    max_nn = int(np.max(nn))
    nn_index, nn_dist = _get_knn_indices_distances(X, max_nn - 1)
    nns = np.column_stack((np.arange(n, dtype=int), nn_index))

    vals = np.zeros((kmax, len(nn), len(lams)), dtype=float)
    Qs = np.zeros((n, kmax, len(nn), len(lams)), dtype=float)
    uix_all: list[np.ndarray] = []
    cs_all: list[np.ndarray] = []

    if np.isinf(iters):
        for ni in range(len(nn)):
            nn_cur = int(nn[ni])
            row_idx = np.repeat(np.arange(n, dtype=int), nn_cur)
            col_idx = nns[:, :nn_cur].reshape(-1)
            data = np.ones_like(row_idx, dtype=float)
            L = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n, n)).tocsr()
            cs = np.asarray(L.sum(axis=0)).ravel()
            cs_all.append(cs)

            uix = np.zeros(kmax, dtype=int)
            uix[0] = int(np.argmax(cs))
            ds = np.sqrt(np.sum((X - X[uix[0], :]) ** 2, axis=1))
            for k in range(1, kmax):
                uix[k] = int(np.argmax(ds * cs))
                dnew = np.sqrt(np.sum((X - X[uix[k], :]) ** 2, axis=1))
                ds = ds * (ds < dnew) + dnew * (dnew <= ds)
            uix_all.append(uix.copy())

            pix = sparse.coo_matrix(
                (np.ones(kmax, dtype=float), (uix, np.arange(kmax, dtype=int))), shape=(n, kmax)
            ).tocsr()

            for li in range(len(lams)):
                lam = float(lams[li])
                I = sparse.eye(n, format="csc") - ((1.0 - lam) * L / nn_cur).tocsc()
                solved = spla.spsolve(I, pix.toarray())
                Qcur = oQ(lam * np.asarray(solved), min(kmax, len(uix)))
                qcols = min(kmax, len(uix))
                Qs[:, :qcols, ni, li] = Qcur

                vals[0, ni, li] = 0.0
                for k in range(2, kmax + 1):
                    ref = (1.0 - lam) * (1.0 / nn_cur + 1.0 / n - 2.0 / np.sqrt(n * nn_cur))
                    qslice = Qs[:, :k, ni, li]
                    expr = 1.0 / k + qslice - np.sum(qslice, axis=1, keepdims=True) / k
                    top = np.mean(_row_maxs(expr)) - (n - k + k**2) / n / k
                    vals[k - 1, ni, li] = top / ref
    else:
        raise ValueError("Debug run currently supports exact branch only (iters=Inf).")

    vals[np.isnan(vals)] = -1.0
    flat_idx = int(np.argmax(vals))
    parms = np.unravel_index(flat_idx, vals.shape)
    k_best = parms[0] + 1

    if k_best > 1:
        qslice = Qs[:, :k_best, parms[1], parms[2]]
        prob = 1.0 / k_best + qslice - np.sum(qslice, axis=1, keepdims=True) / k_best
    else:
        prob = np.ones((n, 1), dtype=float)

    clusters = np.argmax(prob, axis=1) + 1

    return {
        "probabilities": prob,
        "clusters": clusters,
        "vals": vals,
        "nn": nn,
        "lams": lams,
        "X": X,
        "nns_index": nns,
        "nns_dist": np.column_stack((np.zeros(n), nn_dist)),
        "uix_all": uix_all,
        "cs_all": cs_all,
        "Qs": Qs,
        "best_parms": np.array([k_best, parms[1] + 1, parms[2] + 1], dtype=int),
    }
