from __future__ import annotations

import numpy as np

from cns_module import CNS, appr_solv, oQ


def _toy_dataset(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[-1.0, -1.0], scale=[0.25, 0.25], size=(12, 2))
    b = rng.normal(loc=[1.2, 1.0], scale=[0.25, 0.25], size=(12, 2))
    c = rng.normal(loc=[2.8, -1.5], scale=[0.25, 0.25], size=(12, 2))
    return np.vstack([a, b, c])


def test_oq_reorders_columns_and_preserves_shape() -> None:
    Q = np.array(
        [
            [1.0, 0.1, 0.2],
            [0.8, 0.1, 0.3],
            [0.0, 0.9, 0.1],
            [0.0, 0.7, 0.2],
        ]
    )
    out = oQ(Q)
    assert out.shape == Q.shape
    assert np.isclose(np.sum(out[:, 0]), np.max(np.sum(Q, axis=0)))


def test_appr_solv_output_shape() -> None:
    from scipy import sparse

    W = sparse.eye(4, format="csr") * 0.1
    F0 = sparse.eye(4, 3, format="csr")
    lams = np.array([0.2, 0.5, 0.8])

    out = appr_solv(W, F0, lams, iter=5)
    assert out.shape == (4, 3, 3)


def test_cns_exact_branch_basic_contract() -> None:
    X = _toy_dataset()
    res = CNS(X, kmax=10, nn=np.array([3, 6]), lams=np.array([0.2, 0.4]), iters=np.inf)

    assert res.probabilities.shape[0] == X.shape[0]
    assert res.probabilities.shape[1] >= 1
    assert res.clusters.shape == (X.shape[0],)
    assert np.all(res.clusters >= 1)
    assert res.vals.shape == (10, 2, 2)


def test_cns_approx_branch_basic_contract() -> None:
    X = _toy_dataset(seed=19)
    res = CNS(X, kmax=8, nn=np.array([3, 5]), lams=np.array([0.2, 0.4]), iters=10)

    assert res.probabilities.shape[0] == X.shape[0]
    assert res.clusters.shape == (X.shape[0],)
    assert res.vals.shape == (8, 2, 2)


def test_cns_cosine_mode_runs() -> None:
    X = _toy_dataset(seed=27)
    res = CNS(X, kmax=8, nn=np.array([3, 5]), lams=np.array([0.2, 0.4]), distance="cosine")
    assert res.probabilities.shape[0] == X.shape[0]
    assert res.X.shape == X.shape
