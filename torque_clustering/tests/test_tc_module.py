from __future__ import annotations

import numpy as np

from tc_module import torque_clustering


def _big_gaussian_dataset(seed: int = 101) -> tuple[np.ndarray, np.ndarray]:
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


def _bigger_moons_dataset(seed: int = 202) -> tuple[np.ndarray, np.ndarray]:
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


def _assert_partition_valid(partition: list[list[int]], n_samples: int) -> None:
    flat = [idx for comp in partition for idx in comp]
    assert len(flat) == n_samples
    assert sorted(flat) == list(range(n_samples))


def test_tc_big_gaussian_partition_integrity() -> None:
    X, y = _big_gaussian_dataset()
    result = torque_clustering(X=X, true_labels=y)

    assert len(result.connections) > 0
    assert len(result.final_partition) >= 2
    assert len(result.halo_partition) >= len(result.final_partition)
    _assert_partition_valid(result.final_partition, X.shape[0])
    _assert_partition_valid(result.halo_partition, X.shape[0])


def test_tc_accepts_distance_matrix_input() -> None:
    X, y = _bigger_moons_dataset()
    diff = X[:, None, :] - X[None, :, :]
    S = np.sqrt(np.sum(diff * diff, axis=2))

    result = torque_clustering(S=S, true_labels=y)

    assert len(result.connections) > 0
    assert result.mean_M > 0.0
    assert result.mean_D >= 0.0
    _assert_partition_valid(result.final_partition, X.shape[0])


def test_tc_connection_mass_constraint_and_torque_identity() -> None:
    X, y = _big_gaussian_dataset(seed=404)
    result = torque_clustering(X=X, true_labels=y)

    for conn in result.connections:
        assert conn.mass_src <= conn.mass_nn
        assert conn.tau is not None
        assert np.isclose(conn.tau, conn.M * conn.D)


def test_tc_tscl_is_sorted_by_descending_tau() -> None:
    X, y = _bigger_moons_dataset(seed=505)
    result = torque_clustering(X=X, true_labels=y)

    tau_sorted = [result.connections[i].tau for i in result.tscl_indices]
    assert all(tau_sorted[i] >= tau_sorted[i + 1] for i in range(len(tau_sorted) - 1))
