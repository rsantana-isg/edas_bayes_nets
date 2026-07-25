"""
Tests for the DG-search speed-ups and EBNA warm-start additions.

Covers the acceptance criteria of
``docs/bayes_nets_implementation_instructions.md``:

* §2.1 ``initial_structure`` warm-start for the add/delete/reverse hill-climbers
* §2.2 ``k2_pen`` penalized-K2 scorer and the Etxeberria automatic parent bound
* A2 ``candidate_parents="mi:<k>"`` MI candidate pruning
* A3 ``Fast*Scorer`` cached-statistics scorers + ``fast_local_scoring``
* A4 ``max_leaves`` + ``split_score`` bounded local structure
* A5 ``DecisionGraphNDGLearner`` / ``method="dg_ndg"`` one-shot construction
* Priority-3 entry points (DTSL, GSP, PDG)
"""

import time

import numpy as np
import pytest

from bayes_nets import (
    BayesianNetwork,
    K2ScoringMethod,
    K2PenScoringMethod,
    etxeberria_max_parents,
    DecisionTreeMDLScorer,
    DecisionGraphBayesianScorer,
    FastDecisionTreeMDLScorer,
    FastDecisionGraphBayesianScorer,
    StableHillClimbLearner,
    TabuHillClimbLearner,
    DecisionGraphNDGLearner,
    BICScoringMethod,
    mi_candidate_mask,
    learn_local_cpd,
    learn_markov_structure,
    decision_tree_to_features,
    learn_graph_smoothness,
    ProbabilisticDecisionGraph,
    bn_to_pdg,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chain_data():
    """A noisy binary chain X0 → X1 → ... → X_{nv-1}."""
    rng = np.random.default_rng(1234)
    n, nv = 1200, 8
    data = np.zeros((n, nv), dtype=int)
    data[:, 0] = rng.integers(0, 2, n)
    for j in range(1, nv):
        data[:, j] = np.where(rng.random(n) < 0.1, 1 - data[:, j - 1], data[:, j - 1])
    return data, np.full(nv, 2)


@pytest.fixture
def mixed_data():
    """Small mixed-cardinality dataset with context-specific structure."""
    rng = np.random.default_rng(99)
    n = 800
    card = np.array([2, 3, 2, 4, 2, 3])
    data = np.column_stack([rng.integers(0, c, n) for c in card])
    # inject dependencies
    data[:, 1] = np.where(rng.random(n) < 0.2, data[:, 1], data[:, 0] % 3)
    data[:, 2] = np.where(rng.random(n) < 0.15, data[:, 2], data[:, 0])
    return data, card


def _weights(n, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.random(n)
    return w / w.sum()


# ---------------------------------------------------------------------------
# §2.1 warm-start
# ---------------------------------------------------------------------------


class TestWarmStart:
    def test_none_identical_to_default(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        learner = StableHillClimbLearner(BICScoringMethod(alpha=0.0), max_parents=3)
        a_default = learner.learn(data, nv, card)
        a_none = learner.learn(data, nv, card, initial_structure=None)
        assert np.array_equal(a_default, a_none)

    def test_true_dag_unchanged(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        learner = StableHillClimbLearner(BICScoringMethod(alpha=0.0), max_parents=3)
        optimum = learner.learn(data, nv, card)
        warm = learner.learn(data, nv, card, initial_structure=optimum)
        assert np.array_equal(warm, optimum)
        assert learner.last_n_iter_ == 0  # no improving move from the optimum

    def test_warm_start_fewer_iterations(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        learner = StableHillClimbLearner(BICScoringMethod(alpha=0.0), max_parents=3)
        learner.learn(data, nv, card)
        iters_scratch = learner.last_n_iter_
        optimum = learner.learn(data, nv, card)
        learner.learn(data, nv, card, initial_structure=optimum)
        iters_warm = learner.last_n_iter_
        assert iters_warm <= iters_scratch

    def test_cyclic_initial_structure_raises(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        learner = StableHillClimbLearner(BICScoringMethod(alpha=0.0), max_parents=3)
        cyc = np.zeros((nv, nv), dtype=int)
        cyc[0, 1] = cyc[1, 2] = cyc[2, 0] = 1
        with pytest.raises(ValueError):
            learner.learn(data, nv, card, initial_structure=cyc)

    def test_invalid_shape_raises(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        learner = StableHillClimbLearner(BICScoringMethod(alpha=0.0), max_parents=3)
        with pytest.raises(ValueError):
            learner.learn(data, nv, card, initial_structure=np.zeros((nv, nv + 1)))

    def test_tabu_accepts_initial_structure(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        learner = TabuHillClimbLearner(BICScoringMethod(alpha=0.0), max_parents=3, max_iter=50)
        seed = np.zeros((nv, nv), dtype=int)
        seed[0, 1] = 1
        out = learner.learn(data, nv, card, initial_structure=seed)
        assert out.shape == (nv, nv)

    def test_fit_stable_hc_warm_start(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        bn = BayesianNetwork(nv, card).fit(data, method="bic", max_parents=2, alpha=0.5)
        prev = bn.to_adjacency_matrix()
        bn.fit(data, method="stable_hc", max_parents=2, alpha=0.5, initial_structure=prev)
        assert bn.is_dag()


# ---------------------------------------------------------------------------
# §2.2 k2_pen + Etxeberria
# ---------------------------------------------------------------------------


class TestK2Pen:
    def test_zero_penalty_equals_k2(self, mixed_data):
        data, card = mixed_data
        k2 = K2ScoringMethod(alpha=1.0)
        kp = K2PenScoringMethod(alpha=1.0, penalty=0.0)
        for var in range(data.shape[1]):
            parents = [p for p in range(data.shape[1]) if p != var][:2]
            assert kp.local_score(var, parents, data, card) == pytest.approx(
                k2.local_score(var, parents, data, card)
            )

    def test_higher_penalty_sparser(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        counts = []
        for f in (0.0, 0.5, 2.0, 8.0):
            bn = BayesianNetwork(nv, card).fit(
                data, method="k2_pen", max_parents=4, alpha=1.0, penalty=f
            )
            counts.append(int(bn.to_adjacency_matrix().sum()))
        assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))

    def test_etxeberria_paper_example(self):
        # n=20: 17 threes and three 4's; X8 (index 7) is a 4-state variable.
        card = np.array([3] * 7 + [4] + [3] * 5 + [4] + [3] * 5 + [4])
        bound = etxeberria_max_parents(card, 422, 1.0)  # AIC penalty
        assert bound[7] == 5

    def test_k2_pen_none_max_parents_uses_bound(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        bn = BayesianNetwork(nv, card).fit(data, method="k2_pen", alpha=1.0)
        assert bn.is_dag()


# ---------------------------------------------------------------------------
# A2 candidate_parents
# ---------------------------------------------------------------------------


class TestCandidateParents:
    def test_mask_symmetric_and_weighted(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        w = _weights(data.shape[0], seed=5)
        mask = mi_candidate_mask(data, nv, card, 3, sample_weights=w)
        assert np.array_equal(mask, mask.T)
        assert set(np.unique(mask)).issubset({0, 1})

    def test_candidate_parents_recovers_chain_edges(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        bn = BayesianNetwork(nv, card).fit(
            data, method="dg", local_structure="dg", max_parents=4,
            alpha=1.0, candidate_parents="mi:3",
        )
        adj = bn.to_adjacency_matrix()
        # each consecutive chain pair should be connected in some direction
        connected = sum(
            1 for j in range(nv - 1) if adj[j, j + 1] or adj[j + 1, j]
        )
        assert connected >= nv - 2

    def test_candidate_parents_speedup(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        t = time.time()
        BayesianNetwork(nv, card).fit(data, method="dg", local_structure="dg",
                                      max_parents=4, alpha=1.0)
        t_full = time.time() - t
        t = time.time()
        BayesianNetwork(nv, card).fit(data, method="dg", local_structure="dg",
                                      max_parents=4, alpha=1.0, candidate_parents="mi:2")
        t_mi = time.time() - t
        # pruning should not be slower than the unrestricted search
        assert t_mi <= t_full * 1.5

    def test_bad_candidate_parents_raises(self, chain_data):
        data, card = chain_data
        with pytest.raises(ValueError):
            BayesianNetwork(data.shape[1], card).fit(data, method="dg",
                                                     candidate_parents="foo")


# ---------------------------------------------------------------------------
# A3 fast scorers
# ---------------------------------------------------------------------------


class TestFastScorers:
    @pytest.mark.parametrize("weighted", [False, True])
    def test_fast_dt_equals_exact(self, mixed_data, weighted):
        data, card = mixed_data
        sw = _weights(data.shape[0], seed=2) if weighted else None
        exact = DecisionTreeMDLScorer(alpha=1.0, sample_weights=sw)
        fast = FastDecisionTreeMDLScorer(alpha=1.0, sample_weights=sw)
        for var in range(data.shape[1]):
            parents = [p for p in range(data.shape[1]) if p != var][:4]
            assert fast.local_score(var, parents, data, card) == pytest.approx(
                exact.local_score(var, parents, data, card)
            )

    @pytest.mark.parametrize("weighted", [False, True])
    def test_fast_dg_equals_exact(self, mixed_data, weighted):
        data, card = mixed_data
        sw = _weights(data.shape[0], seed=3) if weighted else None
        exact = DecisionGraphBayesianScorer(alpha=1.0, sample_weights=sw)
        fast = FastDecisionGraphBayesianScorer(alpha=1.0, sample_weights=sw)
        for var in range(data.shape[1]):
            parents = [p for p in range(data.shape[1]) if p != var][:4]
            assert fast.local_score(var, parents, data, card) == pytest.approx(
                exact.local_score(var, parents, data, card)
            )

    def test_fit_fast_local_scoring_matches_exact_structure(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        a_exact = BayesianNetwork(nv, card).fit(
            data, method="dg", max_parents=3, alpha=1.0
        ).to_adjacency_matrix()
        a_fast = BayesianNetwork(nv, card).fit(
            data, method="dg", max_parents=3, alpha=1.0, fast_local_scoring=True
        ).to_adjacency_matrix()
        assert np.array_equal(a_exact, a_fast)


# ---------------------------------------------------------------------------
# A4 max_leaves + split_score
# ---------------------------------------------------------------------------


class TestBoundedLocalStructure:
    def test_max_leaves_one_is_marginal(self, mixed_data):
        data, card = mixed_data
        cpd = learn_local_cpd(0, [1, 2, 3], data, card, method="dt",
                              alpha=1.0, max_leaves=1)
        assert cpd.n_distinct_leaves == 1

    def test_max_leaves_monotone(self, mixed_data):
        data, card = mixed_data
        distinct = []
        for ml in (1, 2, 4, 8, None):
            cpd = learn_local_cpd(1, [0, 2, 3, 4, 5], data, card, method="dt",
                                  alpha=1.0, max_leaves=ml)
            distinct.append(cpd.n_distinct_leaves)
        unbounded = distinct[-1]
        assert all(d <= unbounded for d in distinct[:-1])
        assert all(distinct[i] <= distinct[i + 1] for i in range(len(distinct) - 2))

    def test_split_score_mdl_dg_is_samplable(self, mixed_data):
        data, card = mixed_data
        cpd = learn_local_cpd(0, [1, 2, 3], data, card, method="dg",
                              alpha=1.0, split_score="mdl")
        rng = np.random.default_rng(0)
        parents = data[:, [1, 2, 3]]
        out = cpd.sample_rows(parents, rng)
        assert out.shape[0] == data.shape[0]
        assert out.max() < card[0]

    def test_fit_max_leaves_split_score(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        bn = BayesianNetwork(nv, card).fit(
            data, method="dt", local_structure="dt", max_parents=4,
            alpha=1.0, max_leaves=8, split_score="mdl",
        )
        assert bn.has_local_structure()
        for v in bn.cpds:
            assert bn.cpds[v]["local"].n_distinct_leaves <= 8


# ---------------------------------------------------------------------------
# A5 dg_ndg
# ---------------------------------------------------------------------------


class TestDGNdg:
    def test_dg_ndg_returns_dag(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        learner = DecisionGraphNDGLearner(max_parents=4, alpha=1.0)
        adj = learner.learn(data, nv, card)
        bn = BayesianNetwork(nv, card)
        bn.adjacency = adj
        assert bn.is_dag()

    def test_dg_ndg_populates_local_cpds_and_samples(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        bn = BayesianNetwork(nv, card).fit(
            data, method="dg_ndg", local_structure="dg", max_parents=4, alpha=1.0
        )
        assert all("local" in bn.cpds[v] for v in bn.cpds)
        s = bn.sample(50, rng=np.random.default_rng(0))
        assert s.shape == (50, nv)
        assert (s < card).all()

    def test_dg_ndg_faster_than_dg(self):
        rng = np.random.default_rng(7)
        n, nv = 1500, 16
        data = np.zeros((n, nv), dtype=int)
        data[:, 0] = rng.integers(0, 2, n)
        for j in range(1, nv):
            data[:, j] = np.where(rng.random(n) < 0.1, 1 - data[:, j - 1], data[:, j - 1])
        card = np.full(nv, 2)
        t = time.time()
        BayesianNetwork(nv, card).fit(data, method="dg", local_structure="dg",
                                      max_parents=4, alpha=1.0)
        t_dg = time.time() - t
        t = time.time()
        BayesianNetwork(nv, card).fit(data, method="dg_ndg", local_structure="dg",
                                      max_parents=4, alpha=1.0)
        t_ndg = time.time() - t
        assert t_ndg < t_dg

    def test_dg_ndg_honours_sample_weights(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        w = _weights(data.shape[0], seed=8)
        bn = BayesianNetwork(nv, card).fit(
            data, method="dg_ndg", local_structure="dg", max_parents=4,
            alpha=1.0, sample_weights=w,
        )
        assert bn.is_dag()


# ---------------------------------------------------------------------------
# Priority-3 entry points
# ---------------------------------------------------------------------------


class TestPriority3:
    def test_learn_markov_structure_dtsl(self, chain_data):
        data, card = chain_data
        ms = learn_markov_structure(data, card, method="dtsl")
        assert np.array_equal(ms.adjacency, ms.adjacency.T)  # undirected
        assert isinstance(ms.features, list)

    def test_decision_tree_to_features(self, mixed_data):
        data, card = mixed_data
        cpd = learn_local_cpd(0, [1, 2], data, card, method="dt", alpha=1.0)
        feats = decision_tree_to_features(cpd)
        for conj in feats:
            for var, val in conj:
                assert var in (1, 2)
                assert 0 <= val < card[var]

    def test_learn_graph_smoothness_laplacian(self, mixed_data):
        data, card = mixed_data
        L = learn_graph_smoothness(data.astype(float), alpha=1.0, beta=1.0)
        assert L.shape == (data.shape[1], data.shape[1])
        assert np.allclose(L.sum(axis=1), 0.0)  # valid Laplacian rows sum to 0
        assert np.allclose(L, L.T)

    def test_bn_to_pdg(self, chain_data):
        data, card = chain_data
        nv = data.shape[1]
        bn = BayesianNetwork(nv, card).fit(
            data, method="dg_ndg", local_structure="dg", max_parents=3, alpha=1.0
        )
        pdg = bn_to_pdg(bn)
        assert isinstance(pdg, ProbabilisticDecisionGraph)
        assert pdg.n_parameter_nodes >= nv
        assert np.isfinite(pdg.log_probability(data[0]))
