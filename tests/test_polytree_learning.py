"""Tests for the polytree (singly connected) structure learners."""

import numpy as np
import pytest

from bayes_nets import (
    BayesianNetwork,
    CausalPolytreeLearner,
    ChowLiuTreeLearner,
    PolytreeLPALearner,
    RebanePearlPolytreeLearner,
)

LEARNERS = [
    ChowLiuTreeLearner,
    RebanePearlPolytreeLearner,
    PolytreeLPALearner,
    CausalPolytreeLearner,
]

METHOD_NAMES = ["chow_liu", "rebane_pearl", "lpa", "lpa_marginal", "causal_polytree"]


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _sample_collider(n_samples: int, seed: int = 0):
    """X0 -> X2 <- X1 with X3 <- X2, plus an independent X4.

    A genuine polytree with one head-to-head node, so orientation is
    identifiable from the data.  The collider is a noisy OR rather than a
    XOR: under XOR each parent is *marginally* independent of the child, so
    no pairwise-MI skeleton could ever find the edges (Dasgupta 1999, Ex. 2).
    """
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n_samples)
    x1 = rng.integers(0, 2, n_samples)
    noise = rng.random(n_samples) < 0.05
    x2 = np.where(noise, rng.integers(0, 2, n_samples), x0 | x1)
    x3 = np.where(rng.random(n_samples) < 0.1, 1 - x2, x2)
    x4 = rng.integers(0, 2, n_samples)
    data = np.column_stack([x0, x1, x2, x3, x4]).astype(int)

    true_adj = np.zeros((5, 5), dtype=int)
    true_adj[0, 2] = true_adj[1, 2] = true_adj[2, 3] = 1
    return data, true_adj


def _sample_chain(n_samples: int, seed: int = 0):
    """X0 -> X1 -> X2 -> X3: a chain, whose orientation is not identifiable."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2, n_samples)]
    for _ in range(3):
        prev = cols[-1]
        cols.append(np.where(rng.random(n_samples) < 0.1, 1 - prev, prev))
    data = np.column_stack(cols).astype(int)

    true_adj = np.zeros((4, 4), dtype=int)
    for i in range(3):
        true_adj[i, i + 1] = 1
    return data, true_adj


def _is_polytree(adjacency: np.ndarray) -> bool:
    """The skeleton has no undirected cycle, i.e. edges <= nodes - components."""
    n = adjacency.shape[0]
    skeleton = ((adjacency + adjacency.T) > 0).astype(int)
    n_edges = int(skeleton.sum()) // 2

    seen = set()
    n_components = 0
    for start in range(n):
        if start in seen:
            continue
        n_components += 1
        stack = [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            for u in np.flatnonzero(skeleton[v]):
                if u not in seen:
                    seen.add(int(u))
                    stack.append(int(u))
    return n_edges == n - n_components


def _is_dag(adjacency: np.ndarray) -> bool:
    n = adjacency.shape[0]
    indeg = adjacency.sum(axis=0).copy()
    queue = [v for v in range(n) if indeg[v] == 0]
    visited = 0
    while queue:
        v = queue.pop()
        visited += 1
        for u in np.flatnonzero(adjacency[v]):
            indeg[u] -= 1
            if indeg[u] == 0:
                queue.append(int(u))
    return visited == n


def _skeleton_set(adjacency: np.ndarray) -> set:
    n = adjacency.shape[0]
    return {
        (min(u, v), max(u, v))
        for u in range(n)
        for v in range(n)
        if adjacency[u, v] or adjacency[v, u]
    }


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("learner_cls", LEARNERS)
@pytest.mark.parametrize("generator", [_sample_collider, _sample_chain])
def test_output_is_a_polytree(learner_cls, generator):
    data, _ = generator(2000, seed=1)
    n_vars = data.shape[1]
    cardinality = np.full(n_vars, 2)

    adjacency = learner_cls().learn(data, n_vars, cardinality)

    assert _is_dag(adjacency), "learned structure is not acyclic"
    assert _is_polytree(adjacency), "skeleton contains an undirected cycle"


@pytest.mark.parametrize("learner_cls", LEARNERS)
def test_random_data_stays_a_polytree(learner_cls):
    """Independent variables must not induce cycles through spurious edges."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 2, size=(500, 8))
    cardinality = np.full(8, 2)

    adjacency = learner_cls().learn(data, 8, cardinality)

    assert _is_dag(adjacency)
    assert _is_polytree(adjacency)


def _sample_mixed_cardinality_collider(n_samples: int, seed: int = 0):
    """X0 -> X2 <- X1 with X3 <- X2, over mixed cardinalities [3, 4, 5, 2].

    X2 is a noisy deterministic function of both parents taking 5 values, so
    the collider is pairwise detectable and the parents stay marginally
    independent.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 3, n_samples)
    x1 = rng.integers(0, 4, n_samples)
    clean = (x0 + x1) % 5
    x2 = np.where(rng.random(n_samples) < 0.05, rng.integers(0, 5, n_samples), clean)
    x3 = (x2 % 2)
    x3 = np.where(rng.random(n_samples) < 0.1, 1 - x3, x3)
    data = np.column_stack([x0, x1, x2, x3]).astype(int)

    true_adj = np.zeros((4, 4), dtype=int)
    true_adj[0, 2] = true_adj[1, 2] = true_adj[2, 3] = 1
    return data, true_adj, np.array([3, 4, 5, 2])


@pytest.mark.parametrize("learner_cls", LEARNERS)
def test_mixed_cardinality_skeleton_recovered(learner_cls):
    data, true_adj, cardinality = _sample_mixed_cardinality_collider(6000, seed=13)

    adjacency = learner_cls().learn(data, 4, cardinality)

    assert _is_polytree(adjacency)
    assert _is_dag(adjacency)
    assert _skeleton_set(adjacency) == _skeleton_set(true_adj)


@pytest.mark.parametrize(
    "learner_cls", [RebanePearlPolytreeLearner, PolytreeLPALearner, CausalPolytreeLearner]
)
def test_mixed_cardinality_collider_oriented(learner_cls):
    data, _, cardinality = _sample_mixed_cardinality_collider(6000, seed=13)

    adjacency = learner_cls().learn(data, 4, cardinality)

    assert adjacency[0, 2] == 1 and adjacency[1, 2] == 1


@pytest.mark.parametrize("learner_cls", LEARNERS)
def test_high_cardinality_variable(learner_cls):
    """A cardinality-9 node must not break the CI tests or the thresholds."""
    rng = np.random.default_rng(14)
    n = 4000
    x0 = rng.integers(0, 9, n)
    x1 = np.where(rng.random(n) < 0.15, rng.integers(0, 9, n), x0)  # depends on x0
    x2 = rng.integers(0, 3, n)                                       # independent
    data = np.column_stack([x0, x1, x2]).astype(int)
    cardinality = np.array([9, 9, 3])

    adjacency = learner_cls().learn(data, 3, cardinality)

    assert _is_polytree(adjacency)
    assert adjacency[0, 1] or adjacency[1, 0], "dependent high-cardinality pair missed"
    assert not (adjacency[0, 2] or adjacency[2, 0]), "spurious edge to independent node"


@pytest.mark.parametrize("learner_cls", LEARNERS)
def test_non_binary_cardinality(learner_cls):
    rng = np.random.default_rng(3)
    n = 1500
    x0 = rng.integers(0, 3, n)
    x1 = np.where(rng.random(n) < 0.15, rng.integers(0, 3, n), x0)
    x2 = rng.integers(0, 2, n)
    data = np.column_stack([x0, x1, x2]).astype(int)
    cardinality = np.array([3, 3, 2])

    adjacency = learner_cls().learn(data, 3, cardinality)

    assert _is_polytree(adjacency)
    assert adjacency[0, 1] or adjacency[1, 0], "dependent pair X0-X1 not recovered"


# ---------------------------------------------------------------------------
# Recovery quality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("learner_cls", LEARNERS)
def test_skeleton_recovered_on_polytree_data(learner_cls):
    data, true_adj = _sample_collider(4000, seed=2)
    cardinality = np.full(5, 2)

    adjacency = learner_cls().learn(data, 5, cardinality)

    assert _skeleton_set(adjacency) == _skeleton_set(true_adj)


@pytest.mark.parametrize(
    "learner_cls", [RebanePearlPolytreeLearner, PolytreeLPALearner, CausalPolytreeLearner]
)
def test_collider_is_oriented(learner_cls):
    """X0 and X1 are marginally independent but dependent given X2, so both
    must be oriented into X2.  A branching cannot express this, hence
    ChowLiuTreeLearner is excluded."""
    data, _ = _sample_collider(4000, seed=2)
    cardinality = np.full(5, 2)

    adjacency = learner_cls().learn(data, 5, cardinality)

    assert adjacency[0, 2] == 1 and adjacency[1, 2] == 1


def test_chow_liu_is_a_branching():
    data, _ = _sample_collider(2000, seed=4)
    adjacency = ChowLiuTreeLearner().learn(data, 5, np.full(5, 2))
    assert adjacency.sum(axis=0).max() <= 1


def test_lpa_marginal_mode_matches_signature():
    data, true_adj = _sample_collider(3000, seed=5)
    adjacency = PolytreeLPALearner(dep_mode="marginal").learn(data, 5, np.full(5, 2))
    assert _is_polytree(adjacency)
    assert _skeleton_set(adjacency) == _skeleton_set(true_adj)


def test_invalid_dep_mode_rejected():
    with pytest.raises(ValueError):
        PolytreeLPALearner(dep_mode="bogus")


# ---------------------------------------------------------------------------
# Common learner API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("learner_cls", LEARNERS)
def test_max_parents_is_respected(learner_cls):
    data, _ = _sample_collider(2000, seed=6)
    adjacency = learner_cls(max_parents=1).learn(data, 5, np.full(5, 2))
    assert adjacency.sum(axis=0).max() <= 1
    assert _is_polytree(adjacency)


@pytest.mark.parametrize("learner_cls", LEARNERS)
def test_interaction_matrix_is_respected(learner_cls):
    data, _ = _sample_collider(2000, seed=8)
    n_vars = 5
    allowed = np.ones((n_vars, n_vars), dtype=int)
    allowed[2, 3] = allowed[3, 2] = 0  # forbid the X2-X3 edge

    adjacency = learner_cls().learn(data, n_vars, np.full(n_vars, 2),
                                    interaction_matrix=allowed)

    assert adjacency[2, 3] == 0 and adjacency[3, 2] == 0


@pytest.mark.parametrize("learner_cls", LEARNERS)
def test_sample_weights_accepted(learner_cls):
    data, _ = _sample_collider(1500, seed=9)
    weights = np.full(data.shape[0], 1.0 / data.shape[0])

    adjacency = learner_cls().learn(data, 5, np.full(5, 2), sample_weights=weights)

    assert _is_polytree(adjacency)


@pytest.mark.parametrize("learner_cls", LEARNERS)
def test_permutation_orients_free_edges(learner_cls):
    """On chain data no orientation is identifiable, so the permutation decides:
    reversing it must reverse the arcs."""
    data, _ = _sample_chain(3000, seed=10)
    cardinality = np.full(4, 2)
    forward = np.arange(4)

    adj_fwd = learner_cls().learn(data, 4, cardinality, permutation=forward)
    adj_rev = learner_cls().learn(data, 4, cardinality, permutation=forward[::-1])

    assert _skeleton_set(adj_fwd) == _skeleton_set(adj_rev)
    assert np.array_equal(adj_fwd, adj_rev.T)


# ---------------------------------------------------------------------------
# Integration with BayesianNetwork
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_learn_structure_dispatch(method):
    data, _ = _sample_collider(2000, seed=11)
    bn = BayesianNetwork(n_vars=5, cardinality=np.full(5, 2))

    bn.learn_structure(data, method=method)

    assert bn.is_dag()
    assert _is_polytree(bn.adjacency)


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_fitted_polytree_can_sample(method):
    data, _ = _sample_collider(2000, seed=12)
    bn = BayesianNetwork(n_vars=5, cardinality=np.full(5, 2))
    bn.learn_structure(data, method=method)
    bn.learn_parameters(data, alpha=1.0)

    samples = bn.sample(50)

    assert samples.shape == (50, 5)
    assert samples.min() >= 0 and samples.max() <= 1
