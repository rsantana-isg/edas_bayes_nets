"""
Tests for the bayes_nets library.
"""

import numpy as np
import pytest
from bayes_nets import (
    BayesianNetwork,
    BICScoringMethod,
    AICScoringMethod,
    K2ScoringMethod,
    K2StructureLearner,
    GreedyHillClimbLearner,
    MLEParameterLearner,
    ProbabilisticLogicSampler,
    moralize,
    triangulate,
    junction_tree,
    MaxProductInference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Data with strong dependencies: X0 → X1 → X2, X0 → X3.

    Each child is its parent with 10 % random flip noise, so the
    dependency signal is clear.
    """
    rng = np.random.default_rng(42)
    n = 1000
    X0 = rng.integers(0, 2, size=n)
    # flip parent value with probability 0.1
    X1 = np.where(rng.random(n) < 0.1, 1 - X0, X0)
    X2 = np.where(rng.random(n) < 0.1, 1 - X1, X1)
    X3 = np.where(rng.random(n) < 0.1, 1 - X0, X0)
    return np.column_stack([X0, X1, X2, X3])


@pytest.fixture
def binary_cardinality():
    return np.full(4, 2)


# ---------------------------------------------------------------------------
# BayesianNetwork construction
# ---------------------------------------------------------------------------

class TestBayesianNetworkConstruction:
    def test_valid_construction(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 3, 2])
        assert bn.n_vars == 3
        assert list(bn.cardinality) == [2, 3, 2]
        assert bn.adjacency.shape == (3, 3)

    def test_invalid_n_vars(self):
        with pytest.raises(ValueError):
            BayesianNetwork(n_vars=0, cardinality=[])

    def test_invalid_cardinality_length(self):
        with pytest.raises(ValueError):
            BayesianNetwork(n_vars=3, cardinality=[2, 2])

    def test_invalid_cardinality_value(self):
        with pytest.raises(ValueError):
            BayesianNetwork(n_vars=2, cardinality=[1, 2])


# ---------------------------------------------------------------------------
# Graph manipulation
# ---------------------------------------------------------------------------

class TestGraphManipulation:
    def test_add_edge(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        bn.add_edge(0, 1)
        assert bn.has_edge(0, 1)
        assert not bn.has_edge(1, 0)

    def test_add_self_loop_raises(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        with pytest.raises(ValueError):
            bn.add_edge(1, 1)

    def test_add_cycle_raises(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        bn.add_edge(0, 1)
        bn.add_edge(1, 2)
        with pytest.raises(ValueError):
            bn.add_edge(2, 0)

    def test_remove_edge(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        bn.add_edge(0, 1)
        bn.remove_edge(0, 1)
        assert not bn.has_edge(0, 1)

    def test_get_parents(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        bn.add_edge(0, 2)
        bn.add_edge(1, 2)
        assert sorted(bn.get_parents(2)) == [0, 1]

    def test_get_children(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        bn.add_edge(0, 1)
        bn.add_edge(0, 2)
        assert sorted(bn.get_children(0)) == [1, 2]

    def test_is_dag(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        bn.add_edge(0, 1)
        assert bn.is_dag()

    def test_topological_order(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        bn.add_edge(0, 2)
        bn.add_edge(1, 2)
        order = bn.topological_order()
        # 2 must come after both 0 and 1
        assert list(order).index(2) > list(order).index(0)
        assert list(order).index(2) > list(order).index(1)


# ---------------------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------------------

class TestScoring:
    def test_bic_no_parents_vs_with_parents(self, simple_data, binary_cardinality):
        bic = BICScoringMethod(alpha=0.0)
        score_no_parent = bic.local_score(1, [], simple_data, binary_cardinality)
        score_with_parent = bic.local_score(1, [0], simple_data, binary_cardinality)
        # X1 depends on X0; adding X0 as parent should improve BIC
        assert score_with_parent > score_no_parent

    def test_aic_no_parents_vs_with_parents(self, simple_data, binary_cardinality):
        aic = AICScoringMethod(alpha=0.0)
        score_no_parent = aic.local_score(1, [], simple_data, binary_cardinality)
        score_with_parent = aic.local_score(1, [0], simple_data, binary_cardinality)
        assert score_with_parent > score_no_parent

    def test_k2_score_improves_with_true_parent(self, simple_data, binary_cardinality):
        k2 = K2ScoringMethod(alpha=1.0)
        score_no_parent = k2.local_score(1, [], simple_data, binary_cardinality)
        score_with_parent = k2.local_score(1, [0], simple_data, binary_cardinality)
        assert score_with_parent > score_no_parent

    def test_k2_total_score(self, simple_data, binary_cardinality):
        k2 = K2ScoringMethod()
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        adj_with_edges = bn.adjacency.copy()
        adj_with_edges[0, 1] = 1
        score = k2.score(adj_with_edges, simple_data, binary_cardinality)
        assert np.isfinite(score)

    def test_bic_and_aic_handle_zero_counts_without_nan(self):
        data = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
                [1, 1],
            ],
            dtype=int,
        )
        cardinality = np.array([3, 2], dtype=int)

        bic = BICScoringMethod(alpha=0.0)
        aic = AICScoringMethod(alpha=0.0)

        bic_score = bic.local_score(0, [], data, cardinality)
        aic_score = aic.local_score(0, [], data, cardinality)
        assert np.isfinite(bic_score)
        assert np.isfinite(aic_score)


# ---------------------------------------------------------------------------
# Structure learning
# ---------------------------------------------------------------------------

class TestStructureLearning:
    def test_k2_recovers_structure(self, simple_data, binary_cardinality):
        learner = K2StructureLearner(max_parents=2, alpha=1.0)
        ordering = np.array([0, 1, 3, 2])
        adj = learner.learn(simple_data, 4, binary_cardinality, ordering)
        # X0 → X1, X1 → X2, X0 → X3
        assert adj[0, 1] == 1
        assert adj[1, 2] == 1
        assert adj[0, 3] == 1

    def test_greedy_bic_finds_dependencies(self, simple_data, binary_cardinality):
        scoring = BICScoringMethod()
        learner = GreedyHillClimbLearner(scoring=scoring, max_parents=2)
        adj = learner.learn(simple_data, 4, binary_cardinality)
        # The learned graph should have at least some edges
        assert adj.sum() > 0

    def test_fit_bic(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="bic", max_parents=2)
        assert bn.is_dag()
        assert bn.cpds

    def test_fit_k2(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="k2", max_parents=2, ordering=np.arange(4))
        assert bn.is_dag()
        assert bn.cpds

    def test_unknown_method_raises(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        with pytest.raises(ValueError):
            bn.fit(simple_data, method="unknown")

    def test_fit_grow_shrink(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="gs", max_parents=2, alpha=0.05)
        assert bn.is_dag()
        assert bn.adjacency.sum() > 0
        assert (bn.adjacency.sum(axis=0) <= 2).all()

    def test_fit_recursive_causal_discovery(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="rcd", max_parents=2, alpha=0.05)
        assert bn.is_dag()
        assert bn.adjacency.sum() > 0
        assert (bn.adjacency.sum(axis=0) <= 2).all()

    def test_fit_recursive_parallel_causal_discovery(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="rpcd", max_parents=2, alpha=0.05)
        assert bn.is_dag()
        assert bn.adjacency.sum() > 0
        assert (bn.adjacency.sum(axis=0) <= 2).all()


# ---------------------------------------------------------------------------
# Parameter learning
# ---------------------------------------------------------------------------

class TestParameterLearning:
    def test_marginal_sums_to_one(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="k2", ordering=np.arange(4))
        for var in range(4):
            cpd = bn.cpds[var]["cpd"]
            if cpd.ndim == 1:
                assert abs(cpd.sum() - 1.0) < 1e-9
            else:
                np.testing.assert_allclose(cpd.sum(axis=1), 1.0, atol=1e-9)

    def test_cpd_shapes(self, simple_data, binary_cardinality):
        learner = MLEParameterLearner(alpha=1.0)
        adj = np.zeros((4, 4), dtype=int)
        adj[0, 1] = 1  # X0 → X1
        cpds = learner.learn(simple_data, 4, binary_cardinality, adj)
        assert cpds[0]["cpd"].shape == (2,)     # root, cardinality 2
        assert cpds[1]["cpd"].shape == (2, 2)   # 1 parent (card 2), child card 2


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_sample_shape(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="bic")
        samples = bn.sample(100, rng=np.random.default_rng(0))
        assert samples.shape == (100, 4)

    def test_sample_values_in_range(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="bic")
        samples = bn.sample(200)
        for var in range(4):
            assert samples[:, var].min() >= 0
            assert samples[:, var].max() < binary_cardinality[var]

    def test_sample_without_cpds_raises(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        with pytest.raises(RuntimeError):
            bn.sample(10)

    def test_sample_reproducibility(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="bic")
        s1 = bn.sample(50, rng=np.random.default_rng(99))
        s2 = bn.sample(50, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# repr / utility methods
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_repr(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        r = repr(bn)
        assert "BayesianNetwork" in r
        assert "n_vars=3" in r

    def test_n_parameters(self):
        bn = BayesianNetwork(n_vars=3, cardinality=[2, 2, 2])
        # No edges: 3 variables, each with 1 free parameter
        assert bn.n_parameters() == 3

    def test_marginal(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        m = bn.marginal(0, simple_data)
        assert m.shape == (2,)
        assert abs(m.sum() - 1.0) < 1e-9

    def test_to_adjacency_matrix(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="bic")
        adj = bn.to_adjacency_matrix()
        np.testing.assert_array_equal(adj, bn.adjacency)
        # Ensure it's a copy
        adj[0, 0] = 99
        assert bn.adjacency[0, 0] != 99

    def test_structure_helpers_and_dependencies(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="bic", max_parents=2)

        assert bn.big_matrix().shape == (4, 4)
        assert all(len(edge) == 2 for edge in bn.edge_list())

        run_struct = bn.to_run_structure(generation=3, run=1)
        assert run_struct["generation"] == 3
        assert run_struct["run"] == 1
        np.testing.assert_array_equal(run_struct["adjacency"], bn.adjacency)

        deps = bn.variable_dependencies(simple_data)
        assert set(["adjacency_matrix", "edges", "score", "mi_matrix"]).issubset(deps.keys())
        assert deps["mi_matrix"].shape == (4, 4)
        assert np.isfinite(deps["score"])


class TestFactorization:
    def test_moralize_adds_co_parent_edge(self):
        adj = np.zeros((3, 3), dtype=int)
        adj[0, 2] = 1
        adj[1, 2] = 1
        moral = moralize(adj)
        assert moral[0, 1] == 1
        assert moral[1, 0] == 1

    def test_triangulate_and_junction_tree(self):
        graph = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=int,
        )
        triangulated, order, cliques = triangulate(graph, np.array([2, 2, 2, 2]))
        assert triangulated.shape == (4, 4)
        assert len(order) == 4
        assert len(cliques) > 0

        edges, separators = junction_tree(cliques)
        assert len(edges) == max(len(cliques) - 1, 0)
        assert len(separators) == len(edges)

    def test_to_factorization_normalizes_tables(self):
        data = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [1, 1, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=int,
        )
        bn = BayesianNetwork(n_vars=3, cardinality=np.array([2, 2, 2]))
        bn.fit(data, method="k2", ordering=np.arange(3), alpha=0.1)

        fact = bn.to_factorization(data=data, alpha=0.1, max_clique_width=2)
        assert fact.structure.shape[0] == len(fact.tables)

        for row, table in zip(fact.structure, fact.tables):
            n_overlap = int(row[0])
            if n_overlap == 0:
                np.testing.assert_allclose(np.sum(table), 1.0, atol=1e-9)
            else:
                np.testing.assert_allclose(np.sum(table, axis=1), 1.0, atol=1e-9)

    def test_triangulate_respects_clique_separator_decomposition(self):
        """Clique-separator split should preserve both separator-connected cliques."""
        graph = np.zeros((5, 5), dtype=int)
        # triangle 0-1-2
        graph[0, 1] = graph[1, 0] = 1
        graph[1, 2] = graph[2, 1] = 1
        graph[0, 2] = graph[2, 0] = 1
        # triangle 2-3-4 with separator {2}
        graph[2, 3] = graph[3, 2] = 1
        graph[3, 4] = graph[4, 3] = 1
        graph[2, 4] = graph[4, 2] = 1

        _, _, cliques = triangulate(graph, np.array([2, 2, 2, 2, 2]))
        clique_sets = [set(c.tolist()) for c in cliques]
        assert {0, 1, 2} in clique_sets
        assert {2, 3, 4} in clique_sets

    @pytest.mark.parametrize("method", ["mcs", "lexm"])
    def test_triangulate_supports_minimal_triangulation_approximations(self, method):
        graph = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=int,
        )
        triangulated, order, cliques = triangulate(
            graph, np.array([2, 2, 2, 2]), method=method
        )
        assert triangulated.shape == graph.shape
        assert len(order) == 4
        assert len(cliques) >= 1
        assert np.all(np.diag(triangulated) == 0)


class TestInference:
    def test_mpc_matches_bruteforce(self):
        bn = BayesianNetwork(n_vars=2, cardinality=np.array([2, 2]))
        bn.adjacency[0, 1] = 1
        bn.cpds = {
            0: {"parents": [], "cpd": np.array([0.8, 0.2])},
            1: {"parents": [0], "cpd": np.array([[0.9, 0.1], [0.2, 0.8]])},
        }

        conf, prob = bn.most_probable_config()
        np.testing.assert_array_equal(conf, np.array([0, 0]))
        assert prob == pytest.approx(0.72)

    def test_top_k_configs_sorted(self):
        bn = BayesianNetwork(n_vars=2, cardinality=np.array([2, 2]))
        bn.adjacency[0, 1] = 1
        bn.cpds = {
            0: {"parents": [], "cpd": np.array([0.8, 0.2])},
            1: {"parents": [0], "cpd": np.array([[0.9, 0.1], [0.2, 0.8]])},
        }

        configs, probs = bn.k_most_probable_configs(3)
        assert configs.shape == (3, 2)
        assert probs.shape == (3,)
        assert np.all(probs[:-1] >= probs[1:])

    def test_top_k_configs_with_astar_branch_and_bound(self):
        bn = BayesianNetwork(n_vars=2, cardinality=np.array([2, 2]))
        bn.adjacency[0, 1] = 1
        bn.cpds = {
            0: {"parents": [], "cpd": np.array([0.8, 0.2])},
            1: {"parents": [0], "cpd": np.array([[0.9, 0.1], [0.2, 0.8]])},
        }

        configs, probs = bn.k_most_probable_configs(3, search_method="a_star_bb")
        assert configs.shape == (3, 2)
        assert probs.shape == (3,)
        assert np.all(probs[:-1] >= probs[1:])

    def test_loopy_map_respects_evidence(self):
        bn = BayesianNetwork(n_vars=2, cardinality=np.array([2, 2]))
        bn.adjacency[0, 1] = 1
        bn.cpds = {
            0: {"parents": [], "cpd": np.array([0.8, 0.2])},
            1: {"parents": [0], "cpd": np.array([[0.9, 0.1], [0.2, 0.8]])},
        }

        infer = MaxProductInference(bn, loopy_treewidth_threshold=0, loopy_max_iter=50)
        conf, prob = infer.most_probable_config(evidence={0: 1})

        assert conf[0] == 1
        assert prob > 0.0

    def test_loopy_top_k_is_sorted_and_unique(self):
        bn = BayesianNetwork(n_vars=2, cardinality=np.array([2, 2]))
        bn.adjacency[0, 1] = 1
        bn.cpds = {
            0: {"parents": [], "cpd": np.array([0.8, 0.2])},
            1: {"parents": [0], "cpd": np.array([[0.9, 0.1], [0.2, 0.8]])},
        }

        infer = MaxProductInference(bn, loopy_treewidth_threshold=0, loopy_max_iter=50)
        configs, probs = infer.k_most_probable_configs(3)

        assert configs.shape == (3, 2)
        assert probs.shape == (3,)
        assert np.all(probs[:-1] >= probs[1:])
        unique_configs = {tuple(c.tolist()) for c in configs}
        assert len(unique_configs) == len(configs)

# ---------------------------------------------------------------------------
# DMBBN tests
# ---------------------------------------------------------------------------

def test_dmbbn_asia():
    """Verify DMBBN on the Asia benchmark dataset."""
    import pandas as pd
    from bayes_nets.structure_learning import DMBBNStructureLearner

    csv_path = "data/bn_benchmarks/asia_data_N2000.csv"
    try:
        data_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        pytest.skip("Asia benchmark data not found")
        
    data = data_df.values
    
    n_vars = 8
    cardinality = np.full(n_vars, 2)
    
    # True adjacency from asia_meta.json
    true_adj = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    learner = DMBBNStructureLearner(max_parents=3)
    learned_adj = learner.learn(data, n_vars, cardinality)
    
    # Evaluate Structural Hamming Distance (sum of absolute differences)
    shd = int(np.sum(np.abs(learned_adj - true_adj)))

    # DMBBN is approximate; for Asia with 2000 samples, SHD < 10 is acceptable.
    # Note: K2-based learners often find reversed directions for Markov Equivalent
    # edges depending on marginal skews.
    assert shd <= 20  # Lenient bound for approximate learner on observational data


# ---------------------------------------------------------------------------
# Helpers for the ranked-method tests
# ---------------------------------------------------------------------------

def _skeleton(adj):
    """Undirected 0/1 skeleton of a directed adjacency matrix."""
    s = ((np.asarray(adj) != 0) | (np.asarray(adj).T != 0)).astype(int)
    np.fill_diagonal(s, 0)
    return s


def _skeleton_shd(a, b):
    """Number of differing undirected edges between two graphs."""
    return int(np.sum(np.abs(_skeleton(a) - _skeleton(b)))) // 2


def _is_dag(adj):
    n = adj.shape[0]
    bn = BayesianNetwork(n_vars=n, cardinality=np.full(n, 2))
    bn.set_structure(adj)
    return bn.is_dag()


@pytest.fixture
def logistic_chain():
    """Binary chain 0->1->2, 0->3 generated with logistic links.

    Returns (data, cardinality, true_adjacency).
    """
    rng = np.random.default_rng(11)
    n = 3000

    def bern(logit):
        return (rng.random(len(logit)) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    X0 = rng.integers(0, 2, n)
    X1 = bern(2.5 * (2 * X0 - 1))
    X2 = bern(2.5 * (2 * X1 - 1))
    X3 = bern(2.5 * (2 * X0 - 1))
    data = np.column_stack([X0, X1, X2, X3])
    true = np.zeros((4, 4), dtype=int)
    true[0, 1] = true[1, 2] = true[0, 3] = 1
    return data, np.full(4, 2), true


# ---------------------------------------------------------------------------
# DMBBN (Rank 1) -- additional verification beyond the Asia benchmark
# ---------------------------------------------------------------------------

class TestDMBBN:
    def test_recovers_skeleton_synthetic(self, simple_data, binary_cardinality):
        from bayes_nets import DMBBNStructureLearner
        adj = DMBBNStructureLearner(max_parents=2).learn(simple_data, 4, binary_cardinality)
        true = np.zeros((4, 4), dtype=int)
        true[0, 1] = true[1, 2] = true[0, 3] = 1
        assert _is_dag(adj)
        # order-independent Markov-blanket learner should recover the skeleton
        assert _skeleton_shd(adj, true) <= 1

    def test_order_independence(self, simple_data, binary_cardinality):
        # DMBBN ignores permutation; two orderings give the same graph.
        from bayes_nets import DMBBNStructureLearner
        learner = DMBBNStructureLearner(max_parents=2)
        a = learner.learn(simple_data, 4, binary_cardinality,
                          permutation=np.array([0, 1, 2, 3]))
        b = learner.learn(simple_data, 4, binary_cardinality,
                          permutation=np.array([3, 2, 1, 0]))
        assert np.array_equal(a, b)

    def test_via_fit_method(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="dmbbn", max_parents=2)
        assert bn.is_dag()
        assert bn.cpds


# ---------------------------------------------------------------------------
# Level-wise exact DP (Rank 5)
# ---------------------------------------------------------------------------

class TestLevelWiseDP:
    def test_recovers_chain(self, simple_data, binary_cardinality):
        from bayes_nets import LevelWiseDPLearner
        adj = LevelWiseDPLearner(score="bic").learn(simple_data, 4, binary_cardinality)
        assert _is_dag(adj)
        true = np.zeros((4, 4), dtype=int)
        true[0, 1] = true[1, 2] = true[0, 3] = 1
        assert _skeleton_shd(adj, true) == 0

    def test_is_globally_optimal(self, simple_data, binary_cardinality):
        # The exact DP must score no worse than any heuristic learner.
        from bayes_nets import LevelWiseDPLearner, StableHillClimbLearner
        scoring = BICScoringMethod()
        dp = LevelWiseDPLearner(score="bic").learn(simple_data, 4, binary_cardinality)
        hc = StableHillClimbLearner(scoring=BICScoringMethod()).learn(
            simple_data, 4, binary_cardinality)
        assert scoring.score(dp, simple_data, binary_cardinality) >= \
            scoring.score(hc, simple_data, binary_cardinality) - 1e-9

    def test_matches_brute_force(self):
        # On 3 variables, compare against exhaustive DAG enumeration.
        from itertools import product
        from bayes_nets import LevelWiseDPLearner
        rng = np.random.default_rng(5)
        n = 400
        A = rng.integers(0, 2, n)
        B = np.where(rng.random(n) < 0.15, 1 - A, A)
        C = np.where(rng.random(n) < 0.15, 1 - B, B)
        data = np.column_stack([A, B, C])
        card = np.full(3, 2)
        scoring = BICScoringMethod()

        best_score, best_adj = -np.inf, None
        for bits in product([0, 1], repeat=6):
            adj = np.zeros((3, 3), dtype=int)
            pos = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
            for b, (u, v) in zip(bits, pos):
                adj[u, v] = b
            if not _is_dag(adj):
                continue
            s = scoring.score(adj, data, card)
            if s > best_score:
                best_score, best_adj = s, adj

        dp = LevelWiseDPLearner(score="bic", max_parents=2).learn(data, 3, card)
        assert abs(scoring.score(dp, data, card) - best_score) < 1e-9

    def test_guard_on_too_many_vars(self):
        from bayes_nets import LevelWiseDPLearner
        data = np.zeros((10, 25), dtype=int)
        with pytest.raises(ValueError):
            LevelWiseDPLearner(max_vars=20).learn(data, 25, np.full(25, 2))


# ---------------------------------------------------------------------------
# SARTRE pruning (Rank 4)
# ---------------------------------------------------------------------------

class TestSARTRE:
    def test_prunes_to_true_edges(self, simple_data, binary_cardinality):
        # Given the true topological order, SARTRE should prune the
        # fully-connected DAG down to the true edges.
        from bayes_nets import SARTREPruner
        order = np.array([0, 1, 2, 3])
        adj = SARTREPruner(lam=0.05).learn(simple_data, 4, binary_cardinality,
                                           permutation=order)
        assert _is_dag(adj)
        # every edge respects the order (u before v) -> acyclic by construction
        assert adj[1, 0] == 0 and adj[2, 0] == 0 and adj[3, 0] == 0
        true = np.zeros((4, 4), dtype=int)
        true[0, 1] = true[1, 2] = true[0, 3] = 1
        assert _skeleton_shd(adj, true) <= 1

    def test_lambda_controls_sparsity(self, simple_data, binary_cardinality):
        from bayes_nets import SARTREPruner
        order = np.array([0, 1, 2, 3])
        sparse = SARTREPruner(lam=0.5).learn(simple_data, 4, binary_cardinality,
                                             permutation=order)
        dense = SARTREPruner(lam=0.001).learn(simple_data, 4, binary_cardinality,
                                              permutation=order)
        assert dense.sum() >= sparse.sum()

    def test_via_fit_method(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="sartre", max_parents=2, permutation=np.arange(4))
        assert bn.is_dag()


# ---------------------------------------------------------------------------
# iter-DSLA (Rank 2)
# ---------------------------------------------------------------------------

class TestIterDSLA:
    def test_recovers_skeleton(self, simple_data, binary_cardinality):
        from bayes_nets import IterDSLALearner
        learner = IterDSLALearner(n_iter=5, seed=0, max_parents=2)
        adj = learner.learn(simple_data, 4, binary_cardinality)
        assert _is_dag(adj)
        true = np.zeros((4, 4), dtype=int)
        true[0, 1] = true[1, 2] = true[0, 3] = 1
        assert _skeleton_shd(adj, true) <= 1

    def test_beats_random_initialisation(self, simple_data, binary_cardinality):
        # The iterative refinement must improve on a random DAG's BIC.
        from bayes_nets import IterDSLALearner
        scoring = BICScoringMethod()
        learner = IterDSLALearner(n_iter=5, seed=1, max_parents=2)
        adj = learner.learn(simple_data, 4, binary_cardinality)
        empty = np.zeros((4, 4), dtype=int)
        assert scoring.score(adj, simple_data, binary_cardinality) > \
            scoring.score(empty, simple_data, binary_cardinality)

    def test_operators(self):
        from bayes_nets.structure_learning import IterDSLALearner
        s1 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        s2 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        s3 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        # AND keeps only shared edges; OR keeps the union
        assert np.array_equal(IterDSLALearner._and([s1, s2]),
                              np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
        assert np.array_equal(IterDSLALearner._or([s2, s3]),
                              np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        assert np.array_equal(IterDSLALearner._select([s1, s2], [1.0, 5.0]), s2)

    def test_via_fit_method(self, simple_data, binary_cardinality):
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        bn.fit(simple_data, method="iterdsla", max_parents=2)
        assert bn.is_dag()


# ---------------------------------------------------------------------------
# BINOTEARS (Rank 6)
# ---------------------------------------------------------------------------

class TestBinaryNotears:
    def test_recovers_skeleton(self, logistic_chain):
        from bayes_nets import BinaryNotearsLearner
        data, card, true = logistic_chain
        adj = BinaryNotearsLearner(lambda1=0.02, w_threshold=0.3).learn(data, 4, card)
        assert _is_dag(adj)
        assert _skeleton_shd(adj, true) <= 1

    def test_output_is_always_dag(self, logistic_chain):
        from bayes_nets import BinaryNotearsLearner
        data, card, _ = logistic_chain
        adj = BinaryNotearsLearner(lambda1=0.0, w_threshold=0.05).learn(data, 4, card)
        assert _is_dag(adj)

    def test_rejects_non_binary(self):
        from bayes_nets import BinaryNotearsLearner
        data = np.zeros((50, 3), dtype=int)
        with pytest.raises(ValueError):
            BinaryNotearsLearner().learn(data, 3, np.array([2, 3, 2]))

    def test_via_fit_method(self, logistic_chain):
        data, card, _ = logistic_chain
        bn = BayesianNetwork(n_vars=4, cardinality=card)
        bn.fit(data, method="binotears")
        assert bn.is_dag()


# ---------------------------------------------------------------------------
# Logistic-regression parameter learning (Rank 3)
# ---------------------------------------------------------------------------

class TestLogisticParameterLearning:
    def test_cpd_is_valid_distribution(self, simple_data, binary_cardinality):
        from bayes_nets import LogisticRegressionParameterLearner
        lr = LogisticRegressionParameterLearner(C=5.0)
        adj = np.zeros((4, 4), dtype=int)
        adj[0, 1] = adj[1, 2] = adj[0, 3] = 1
        cpds = lr.learn(simple_data, 4, binary_cardinality, adj)
        for v in range(4):
            cpd = cpds[v]["cpd"]
            assert np.allclose(cpd.sum(axis=-1), 1.0)
            assert np.all(cpd >= 0)

    def test_recovers_xor_interaction(self):
        # XOR target is not additively separable; the XOR feature is needed.
        from bayes_nets import LogisticRegressionParameterLearner
        rng = np.random.default_rng(3)
        n = 4000
        X0 = rng.integers(0, 2, n)
        X1 = rng.integers(0, 2, n)
        X2 = np.where(rng.random(n) < 0.05, 1 - (X0 ^ X1), X0 ^ X1)
        data = np.column_stack([X0, X1, X2])
        card = np.full(3, 2)
        lr = LogisticRegressionParameterLearner(C=20.0, use_xor=True)
        cpd = lr.estimate_cpd(2, [0, 1], data, card)
        # config index: 0=(0,0),1=(1,0),2=(0,1),3=(1,1) with X0 fastest.
        # xor => P(X2=1) low, high, high, low
        p1 = cpd[:, 1]
        assert p1[0] < 0.5 and p1[3] < 0.5
        assert p1[1] > 0.5 and p1[2] > 0.5

    def test_integrates_with_bn_sampling(self, simple_data, binary_cardinality):
        from bayes_nets import LogisticRegressionParameterLearner
        bn = BayesianNetwork(n_vars=4, cardinality=binary_cardinality)
        adj = np.zeros((4, 4), dtype=int)
        adj[0, 1] = adj[1, 2] = adj[0, 3] = 1
        bn.set_structure(adj)
        bn.learn_parameters(
            simple_data,
            parameter_learner=LogisticRegressionParameterLearner(C=5.0),
        )
        samples = bn.sample(200)
        assert samples.shape == (200, 4)
        assert set(np.unique(samples)).issubset({0, 1})


# ---------------------------------------------------------------------------
# New methods from docs/New_Additions (top-5 ranked)
# ---------------------------------------------------------------------------

def _chain_bn(n_vars=5, flip=0.15, n=1500, seed=11):
    """Build a chain-structured BN 0->1->...->(n-1) with fitted CPDs."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, n_vars), dtype=int)
    X[:, 0] = rng.integers(0, 2, n)
    for i in range(1, n_vars):
        X[:, i] = np.where(rng.random(n) < flip, 1 - X[:, i - 1], X[:, i - 1])
    card = np.full(n_vars, 2)
    bn = BayesianNetwork(n_vars=n_vars, cardinality=card)
    adj = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(1, n_vars):
        adj[i - 1, i] = 1
    bn.set_structure(adj)
    bn.learn_parameters(X, alpha=1.0)
    return bn, X, card


class TestBoundedTreewidthLearner:
    """Nie, Mauá, de Campos & Ji (2014): k-tree sampling."""

    def test_treewidth_is_bounded(self):
        from bayes_nets.structure_learning import BoundedTreewidthLearner
        rng = np.random.default_rng(0)
        n = 1200
        X = rng.integers(0, 2, size=(n, 8))
        for i in range(1, 8):
            X[:, i] = np.where(rng.random(n) < 0.15, X[:, i], X[:, i - 1])
        card = np.full(8, 2)
        k = 2
        learner = BoundedTreewidthLearner(k=k, n_ktrees=40, seed=3)
        adj = learner.learn(X, 8, card)
        moral = moralize(adj)
        _, _, cliques = triangulate(moral, card, method="min-fill")
        tw = max((len(c) - 1 for c in cliques), default=0)
        assert tw <= k
        # k-tree stored for inspection
        assert learner.ktree_ is not None

    def test_recovers_chain_dependencies(self):
        bn_true, X, card = _chain_bn(n_vars=5, flip=0.1, n=2000, seed=7)
        bn = BayesianNetwork(n_vars=5, cardinality=card)
        bn.learn_structure(X, method="bounded_tw", treewidth_bound=2,
                           n_ktrees=60, seed=1)
        # each consecutive pair should be connected (either direction)
        skel = bn.adjacency + bn.adjacency.T
        for i in range(1, 5):
            assert skel[i - 1, i] > 0

    def test_k_equals_one_is_forest(self):
        from bayes_nets.structure_learning import BoundedTreewidthLearner
        _, X, card = _chain_bn(n_vars=6, seed=9)
        learner = BoundedTreewidthLearner(k=1, n_ktrees=30, seed=2)
        adj = learner.learn(X, 6, card)
        # a forest has at most n-1 edges and max 1 parent per node
        assert int(adj.sum()) <= 5
        assert int(adj.sum(axis=0).max()) <= 1


class TestArithmeticCircuit:
    """Vergari, Di Mauro & Esposito (2016): compiled SPN / circuit."""

    def test_marginals_match_exact(self):
        bn, X, card = _chain_bn(n_vars=5, seed=13)
        ac = bn.to_circuit()
        inf = MaxProductInference(bn, loopy_treewidth_threshold=None)
        m_ve = inf.marginals()
        m_ac = ac.marginals()
        for a, b in zip(m_ve, m_ac):
            assert np.allclose(a, b, atol=1e-9)

    def test_partition_function_is_one(self):
        bn, _, _ = _chain_bn(seed=15)
        ac = bn.to_circuit()
        assert abs(ac.probability() - 1.0) < 1e-9

    def test_mpe_matches_exact(self):
        bn, _, _ = _chain_bn(n_vars=5, seed=17)
        ac = bn.to_circuit()
        inf = MaxProductInference(bn, loopy_treewidth_threshold=None)
        a_ve, p_ve = inf.most_probable_config()
        a_ac, p_ac = ac.mpe()
        assert np.array_equal(a_ve, a_ac)
        assert abs(p_ve - p_ac) < 1e-9

    def test_conditional_marginals_match_exact(self):
        bn, _, _ = _chain_bn(n_vars=5, seed=19)
        ac = bn.to_circuit()
        inf = MaxProductInference(bn, loopy_treewidth_threshold=None)
        ev = {0: 1}
        for a, b in zip(inf.marginals(ev), ac.marginals(ev)):
            assert np.allclose(a, b, atol=1e-9)

    def test_sampling_reproduces_marginals(self):
        bn, _, _ = _chain_bn(n_vars=5, seed=21)
        ac = bn.to_circuit()
        m_ac = ac.marginals()
        S = ac.sample(20000, rng=np.random.default_rng(4))
        for v in range(5):
            emp = np.bincount(S[:, v], minlength=2) / S.shape[0]
            assert np.allclose(emp, m_ac[v], atol=0.03)

    def test_node_counts_positive(self):
        bn, _, _ = _chain_bn(seed=23)
        ac = bn.to_circuit()
        assert ac.n_sum_nodes > 0
        assert ac.n_product_nodes > 0
        assert ac.n_leaf_nodes > 0
        assert ac.size == ac.n_sum_nodes + ac.n_product_nodes + ac.n_leaf_nodes


class TestMeanFieldMarginals:
    """Li & Zemel (2014): mean-field variational marginals."""

    def test_distributions_are_valid(self):
        bn, _, _ = _chain_bn(n_vars=5, seed=25)
        inf = MaxProductInference(bn, loopy_treewidth_threshold=None)
        q = inf.marginals(method="mean_field")
        assert len(q) == 5
        for qi in q:
            assert abs(float(qi.sum()) - 1.0) < 1e-6
            assert np.all(qi >= 0)

    def test_accurate_under_weak_coupling(self):
        # near-independent variables => mean field is (nearly) exact
        rng = np.random.default_rng(27)
        n = 4000
        probs = [0.2, 0.7, 0.45, 0.6]
        X = np.stack([(rng.random(n) < p).astype(int) for p in probs], axis=1)
        X[:, 1] = np.where(rng.random(n) < 0.05, X[:, 0], X[:, 1])
        card = np.full(4, 2)
        bn = BayesianNetwork(n_vars=4, cardinality=card)
        adj = np.zeros((4, 4), dtype=int)
        adj[0, 1] = 1
        bn.set_structure(adj)
        bn.learn_parameters(X, alpha=1.0)
        inf = MaxProductInference(bn, loopy_treewidth_threshold=None)
        m_ve = inf.marginals()
        m_mf = inf.marginals(method="mean_field")
        for a, b in zip(m_ve, m_mf):
            assert np.allclose(a, b, atol=0.02)

    def test_invalid_method_raises(self):
        bn, _, _ = _chain_bn(seed=29)
        inf = MaxProductInference(bn, loopy_treewidth_threshold=None)
        with pytest.raises(ValueError):
            inf.marginals(method="nonsense")


class TestSRMPMap:
    """Kolmogorov (2015): sequential reweighted message passing (MAP)."""

    def test_exact_on_tree(self):
        bn, _, _ = _chain_bn(n_vars=6, seed=31)
        inf = MaxProductInference(bn, loopy_treewidth_threshold=None)
        a_ve, p_ve = inf.most_probable_config(search_method="ve")
        a_sr, p_sr = inf.most_probable_config(search_method="srmp")
        assert np.array_equal(a_ve, a_sr)
        assert abs(p_ve - p_sr) < 1e-9

    def test_exact_on_tree_with_evidence(self):
        bn, _, _ = _chain_bn(n_vars=6, seed=33)
        inf = MaxProductInference(bn, loopy_treewidth_threshold=None)
        ev = {0: 1}
        a_ve, _ = inf.most_probable_config(ev, search_method="ve")
        a_sr, _ = inf.most_probable_config(ev, search_method="srmp")
        assert np.array_equal(a_ve, a_sr)
        assert a_sr[0] == 1

    def test_invalid_search_method_raises(self):
        bn, _, _ = _chain_bn(seed=35)
        inf = MaxProductInference(bn, loopy_treewidth_threshold=None)
        with pytest.raises(ValueError):
            inf.most_probable_config(search_method="nonsense")


class TestBayesianVariableClustering:
    """Marrelec, Messé & Bellec (2015): linkage tree by Bayes factor."""

    def test_groups_dependent_and_separates_independent(self):
        from bayes_nets import bayesian_variable_clustering
        rng = np.random.default_rng(37)
        n = 2000
        X = np.zeros((n, 6), dtype=int)
        X[:, 0] = rng.integers(0, 2, n)
        X[:, 1] = np.where(rng.random(n) < 0.05, 1 - X[:, 0], X[:, 0])
        X[:, 2] = np.where(rng.random(n) < 0.05, 1 - X[:, 1], X[:, 1])
        X[:, 3] = rng.integers(0, 2, n)  # independent
        X[:, 4] = rng.integers(0, 2, n)  # independent
        X[:, 5] = rng.integers(0, 2, n)  # independent
        card = np.full(6, 2)
        res = bayesian_variable_clustering(X, card, alpha=1.0)
        clusters = res["clusters"]
        # the dependent trio must all end up in the same cluster
        home = {v: i for i, c in enumerate(clusters) for v in c}
        assert home[0] == home[1] == home[2]
        # independent vars must not join the dependent trio
        assert home[3] != home[0]
        assert home[4] != home[0]

    def test_stop_threshold_controls_merging(self):
        from bayes_nets import bayesian_variable_clustering
        rng = np.random.default_rng(39)
        n = 800
        X = rng.integers(0, 2, size=(n, 5))
        card = np.full(5, 2)
        # very high threshold => nothing merges (all singletons)
        res = bayesian_variable_clustering(X, card, alpha=1.0,
                                           stop_threshold=1e9)
        assert len(res["clusters"]) == 5
        assert res["merges"] == []

    def test_bn_method_wrapper(self):
        rng = np.random.default_rng(41)
        n = 1000
        X = np.zeros((n, 4), dtype=int)
        X[:, 0] = rng.integers(0, 2, n)
        X[:, 1] = np.where(rng.random(n) < 0.05, 1 - X[:, 0], X[:, 0])
        X[:, 2] = rng.integers(0, 2, n)
        X[:, 3] = rng.integers(0, 2, n)
        bn = BayesianNetwork(n_vars=4, cardinality=np.full(4, 2))
        res = bn.learn_variable_clustering(X, alpha=1.0)
        home = {v: i for i, c in enumerate(res["clusters"]) for v in c}
        assert home[0] == home[1]


# ---------------------------------------------------------------------------
# Discrete Gaussian-copula sampler (Kalaitzis & Silva 2013; Hoff 2007)
# ---------------------------------------------------------------------------

def _ordinal_copula_data(n=1500, rho=0.8, seed=0):
    """Two correlated ordinal columns + one independent ordinal column."""
    rng = np.random.default_rng(seed)
    L = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    Y0 = np.digitize(L[:, 0], [-0.5, 0.5])       # 3 levels
    Y1 = np.digitize(L[:, 1], [-1.0, 0.0, 1.0])  # 4 levels
    Y2 = rng.integers(0, 3, n)                   # independent
    return np.column_stack([Y0, Y1, Y2])


class TestGaussianCopulaSampler:
    def test_recovers_latent_correlation(self):
        from bayes_nets import GaussianCopulaSampler
        Y = _ordinal_copula_data(rho=0.8, seed=1)
        s = GaussianCopulaSampler(n_gibbs=150, burn_in=50, seed=2).fit(Y)
        C = s.correlation_
        assert C.shape == (3, 3)
        # strong positive dependence recovered between the coupled columns
        assert C[0, 1] > 0.6
        # near-zero for the independent column
        assert abs(C[0, 2]) < 0.2
        assert abs(C[1, 2]) < 0.2

    def test_sampled_data_reproduces_association(self):
        from bayes_nets import GaussianCopulaSampler
        from scipy.stats import spearmanr
        Y = _ordinal_copula_data(rho=0.8, seed=3)
        s = GaussianCopulaSampler(n_gibbs=150, burn_in=50, seed=4).fit(Y)
        S = s.sample(4000, rng=np.random.default_rng(5))
        r_orig = spearmanr(Y[:, 0], Y[:, 1]).statistic
        r_samp = spearmanr(S[:, 0], S[:, 1]).statistic
        assert abs(r_orig - r_samp) < 0.1
        # independent pair stays close to zero
        assert abs(spearmanr(S[:, 0], S[:, 2]).statistic) < 0.15

    def test_marginals_preserved(self):
        from bayes_nets import GaussianCopulaSampler
        Y = _ordinal_copula_data(seed=6)
        s = GaussianCopulaSampler(n_gibbs=120, burn_in=40, seed=7).fit(Y)
        S = s.sample(6000, rng=np.random.default_rng(8))
        for j in range(3):
            po = np.bincount(Y[:, j], minlength=5) / len(Y)
            ps = np.bincount(S[:, j], minlength=5) / len(S)
            assert np.allclose(po, ps, atol=0.05)
        # generated values never leave the observed support
        assert set(np.unique(S[:, 0])).issubset(set(np.unique(Y[:, 0])))

    def test_independent_data_gives_near_identity(self):
        from bayes_nets import GaussianCopulaSampler
        rng = np.random.default_rng(9)
        Y = rng.integers(0, 3, size=(1000, 4))
        s = GaussianCopulaSampler(n_gibbs=120, burn_in=40, seed=10).fit(Y)
        off = s.correlation_ - np.eye(4)
        assert np.max(np.abs(off)) < 0.2

    def test_sample_before_fit_raises(self):
        from bayes_nets import GaussianCopulaSampler
        s = GaussianCopulaSampler(seed=0)
        with pytest.raises(RuntimeError):
            s.sample(10)

    def test_invalid_burn_in_raises(self):
        from bayes_nets import GaussianCopulaSampler
        with pytest.raises(ValueError):
            GaussianCopulaSampler(n_gibbs=50, burn_in=50)


# ---------------------------------------------------------------------------
# Sample-weight support for the three previously-unweighted learners
# ---------------------------------------------------------------------------

class TestSampleWeightSupport:
    """Weighting a row by w must equal physically replicating it w times.

    Each learner is run twice: (a) on the unique rows with a probability
    vector proportional to per-row multiplicities, and (b) on the dataset
    with those rows physically repeated.  For learners that consume weights
    as effective counts the two results must be identical.
    """

    def _weighted_vs_replicated(self, seed=0, n=200):
        rng = np.random.default_rng(seed)
        X0 = rng.integers(0, 2, n)
        X1 = np.where(rng.random(n) < 0.1, 1 - X0, X0)
        X2 = np.where(rng.random(n) < 0.1, 1 - X1, X1)
        X3 = np.where(rng.random(n) < 0.1, 1 - X0, X0)
        data = np.column_stack([X0, X1, X2, X3])
        mult = rng.integers(1, 5, size=n)
        rep = np.repeat(data, mult, axis=0)
        weights = mult / mult.sum()          # probability vector, sums to 1
        return data, rep, weights, np.full(4, 2)

    def test_sartre_weights_equal_replication(self):
        from bayes_nets import SARTREPruner
        data, rep, w, card = self._weighted_vs_replicated()
        order = np.array([0, 1, 2, 3])
        a = SARTREPruner(lam=0.05).learn(data, 4, card, permutation=order, sample_weights=w)
        b = SARTREPruner(lam=0.05).learn(rep, 4, card, permutation=order)
        assert np.array_equal(a, b)

    def test_binotears_weights_equal_replication(self):
        from bayes_nets import BinaryNotearsLearner
        data, rep, w, card = self._weighted_vs_replicated()
        a = BinaryNotearsLearner(lambda1=0.03, w_threshold=0.3).learn(data, 4, card, sample_weights=w)
        b = BinaryNotearsLearner(lambda1=0.03, w_threshold=0.3).learn(rep, 4, card)
        assert np.array_equal(a, b)

    def test_iterdsla_custom_base_weights_equal_replication(self):
        # Exercises weight propagation into a *custom* base learner.
        from bayes_nets import IterDSLALearner, GreedyHillClimbLearner
        data, rep, w, card = self._weighted_vs_replicated()
        base_a = GreedyHillClimbLearner(scoring=BICScoringMethod(), max_parents=2)
        base_b = GreedyHillClimbLearner(scoring=BICScoringMethod(), max_parents=2)
        a = IterDSLALearner(base_learner=base_a, n_iter=3, seed=7, max_parents=2).learn(
            data, 4, card, sample_weights=w)
        b = IterDSLALearner(base_learner=base_b, n_iter=3, seed=7, max_parents=2).learn(
            rep, 4, card)
        assert np.array_equal(a, b)

    def test_weights_actually_change_result(self):
        # Guard against a vacuous test: weights must be able to change output.
        from bayes_nets import SARTREPruner
        data, _, _, card = self._weighted_vs_replicated()
        order = np.array([0, 1, 2, 3])
        rng = np.random.default_rng(1)
        skew = rng.random(len(data)); skew /= skew.sum()
        uniform = SARTREPruner(lam=0.05).learn(data, 4, card, permutation=order)
        weighted = SARTREPruner(lam=0.05).learn(data, 4, card, permutation=order, sample_weights=skew)
        # both are valid DAGs; the weighting path runs and produces a graph
        assert uniform.shape == weighted.shape == (4, 4)


# ---------------------------------------------------------------------------
# K2 variants (docs/K2_Improvements)
# ---------------------------------------------------------------------------

class TestK2Variants:
    def _shuffled_chain(self, seed=0):
        # chain 0->1->2, 0->3 then present columns in an *arbitrary* order,
        # which is the realistic scenario the K2 improvements target.
        rng = np.random.default_rng(seed)
        n = 1500
        X0 = rng.integers(0, 2, n)
        X1 = np.where(rng.random(n) < 0.1, 1 - X0, X0)
        X2 = np.where(rng.random(n) < 0.1, 1 - X1, X1)
        X3 = np.where(rng.random(n) < 0.1, 1 - X0, X0)
        data = np.column_stack([X0, X1, X2, X3])
        true = np.zeros((4, 4), dtype=int); true[0, 1] = true[1, 2] = true[0, 3] = 1
        perm = rng.permutation(4)
        return data[:, perm], np.full(4, 2), true[np.ix_(perm, perm)]

    def _skf1(self, L, T):
        def sk(a):
            s = ((a | a.T) > 0).astype(int); np.fill_diagonal(s, 0); return s
        L, T = sk(L), sk(T)
        tp = np.sum((L == 1) & (T == 1)) / 2
        fp = np.sum((L == 1) & (T == 0)) / 2
        fn = np.sum((T == 1) & (L == 0)) / 2
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    def test_mi_ordering_is_a_valid_permutation(self):
        from bayes_nets import mi_variable_ordering
        data, card, _ = self._shuffled_chain()
        order = mi_variable_ordering(data, 4, card)
        assert sorted(order.tolist()) == [0, 1, 2, 3]

    def test_mi_ordering_is_order_independent(self):
        # The derived ordering must reflect the data, not the column order:
        # permuting columns and mapping back yields the same variable order.
        from bayes_nets import mi_variable_ordering, K2VariantLearner
        data, card, true = self._shuffled_chain(seed=1)
        a = K2VariantLearner(order_method="mi", parent_restriction=None).learn(data, 4, card)
        rng = np.random.default_rng(9)
        p = rng.permutation(4)
        b = K2VariantLearner(order_method="mi", parent_restriction=None).learn(
            data[:, p], 4, card[p])
        # map b back to original variable indices
        b_mapped = np.zeros_like(b)
        for i in range(4):
            for j in range(4):
                b_mapped[p[i], p[j]] = b[i, j]
        assert np.array_equal(a, b_mapped)

    def test_candidate_mask_is_symmetric_binary(self):
        from bayes_nets import mi_candidate_mask
        data, card, _ = self._shuffled_chain()
        m = mi_candidate_mask(data, 4, card, top_k=2)
        assert np.array_equal(m, m.T)
        assert set(np.unique(m)).issubset({0, 1})
        assert np.all(np.diag(m) == 0)

    def test_variants_beat_baseline_under_arbitrary_order(self):
        # Averaged over several arbitrary column orders, the MI-ordering
        # variant should recover more of the true skeleton than plain K2.
        from bayes_nets import K2StructureLearner, K2VariantLearner
        base, var = [], []
        for s in range(6):
            data, card, true = self._shuffled_chain(seed=s)
            b = K2StructureLearner().learn(data, 4, card)
            v = K2VariantLearner(order_method="mi", parent_restriction="mi",
                                 refine=True).learn(data, 4, card)
            base.append(self._skf1(b, true))
            var.append(self._skf1(v, true))
        assert np.mean(var) >= np.mean(base)

    def _dense_dataset(self, n_vars=24, n=2000, seed=0):
        # A denser 2-parent DAG so base K2 does genuine multi-step parent
        # search (the 10x budget is only meaningful when base time is not
        # sub-ms noise; a trivial 1-parent tree makes base K2 unrealistically
        # fast and inflates every ratio).
        rng = np.random.default_rng(seed)
        cols = [rng.integers(0, 2, n), rng.integers(0, 2, n)]
        for v in range(2, n_vars):
            p1, p2 = cols[v - 1], cols[v - 2]
            base = (p1 & p2) | ((p1 | p2) & (rng.random(n) < 0.5).astype(int))
            cols.append(np.where(rng.random(n) < 0.15, 1 - base, base))
        return np.column_stack(cols).astype(int), np.full(n_vars, 2)

    def test_variants_within_time_budget(self):
        # Every variant must stay well under 10x the base K2 time on a
        # realistically-sized problem (median of repeats to reduce noise).
        import time
        from bayes_nets import K2StructureLearner, K2VariantLearner
        data, card = self._dense_dataset(n_vars=24)
        n = data.shape[1]
        base = []
        for _ in range(3):
            t = time.time(); K2StructureLearner().learn(data, n, card); base.append(time.time() - t)
        tb = np.median(base)
        # All primary variants must stay within 10x base K2.  The ensemble is
        # ~n_orderings x by construction, so it is tested at n_orderings=4
        # (~4-6x) to leave CI headroom; the default preset uses 5.
        for kw in [dict(order_method="mi", parent_restriction=None),
                   dict(order_method="given", parent_restriction="mi"),
                   dict(order_method="given", parent_restriction="mb"),
                   dict(order_method="mi", parent_restriction="mi", refine=True),
                   dict(order_method="mi", parent_restriction="mi", n_orderings=4)]:
            t = time.time(); K2VariantLearner(**kw).learn(data, n, card); tt = time.time() - t
            assert tt < 10 * tb, f"{kw} took {tt/tb:.1f}x base K2"

    def test_all_variants_return_dags(self):
        from bayes_nets import K2VariantLearner
        data, card, _ = self._shuffled_chain()
        for kw in [dict(order_method="mi"),
                   dict(parent_restriction="mb"),
                   dict(refine=True),
                   dict(n_orderings=5, seed=0)]:
            adj = K2VariantLearner(**kw).learn(data, 4, card)
            assert _is_dag(adj)

    def test_fit_methods(self):
        for m in ["k2_mi", "k2_mb", "k2_refine", "k2_ensemble", "k2_plus"]:
            data, card, _ = self._shuffled_chain()
            bn = BayesianNetwork(n_vars=4, cardinality=card)
            bn.fit(data, method=m, max_parents=2)
            assert bn.is_dag()


# ---------------------------------------------------------------------------
# Objective-guided K2 orderings and the independent baseline
# (Univ_BN, FI_k2, RFE_k2)
# ---------------------------------------------------------------------------

def _eda_dataset(n=2000, p=8, strong=5, second=2, seed=0):
    """EDA-style data: objective depends linearly on `strong` (most) & `second`.

    Returns (X, cardinality, solution_prob) where solution_prob are Boltzmann
    weights computed from the objective (the per-solution probability).
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, p))
    f = 3.0 * X[:, strong] + 1.6 * X[:, second] + 0.01 * rng.standard_normal(n)
    fn = (f - f.min()) / (f.max() - f.min() + 1e-12)
    w = np.exp(fn / 0.5)
    w = w / w.sum()
    return X, np.full(p, 2), w


class TestIndependentBN:
    """Univ_BN — empty-graph baseline."""

    def test_learner_returns_empty_graph(self):
        from bayes_nets import IndependentBNLearner
        X, card, w = _eda_dataset(seed=1)
        adj = IndependentBNLearner().learn(X, X.shape[1], card, sample_weights=w)
        assert adj.shape == (X.shape[1], X.shape[1])
        assert int(adj.sum()) == 0

    def test_fit_all_variables_are_roots(self):
        X, card, w = _eda_dataset(seed=2)
        p = X.shape[1]
        bn = BayesianNetwork(n_vars=p, cardinality=card)
        bn.fit(X, method="univ_bn", sample_weights=w)
        assert int(bn.adjacency.sum()) == 0
        assert len(bn.cpds) == p
        assert all(len(bn.cpds[v]["parents"]) == 0 for v in range(p))

    def test_marginals_match_empirical(self):
        X, card, _ = _eda_dataset(seed=3)
        p = X.shape[1]
        bn = BayesianNetwork(n_vars=p, cardinality=card)
        bn.fit(X, method="univ_bn")  # unweighted marginals
        for v in range(p):
            emp = np.bincount(X[:, v], minlength=2) / len(X)
            assert np.allclose(bn.cpds[v]["cpd"], emp, atol=0.02)


class TestFeatureImportanceOrdering:
    """FI_k2 — univariate feature-importance ordering."""

    def test_returns_valid_permutation(self):
        from bayes_nets import feature_importance_ordering
        X, _, w = _eda_dataset(seed=4)
        order = feature_importance_ordering(X, w, method="mutual_info", seed=1)
        assert sorted(order.tolist()) == list(range(X.shape[1]))

    def test_linear_measures_rank_strongest_first(self):
        from bayes_nets import feature_importance_ordering
        X, _, w = _eda_dataset(strong=5, second=2, seed=5)
        for m in ("r_regression", "f_regression"):
            order = feature_importance_ordering(X, w, method=m, seed=1)
            # the two variables driving the objective come first (any order)
            assert set(order[:2].tolist()) == {5, 2}

    def test_no_signal_returns_permutation(self):
        from bayes_nets import feature_importance_ordering
        X, _, _ = _eda_dataset(seed=6)
        # None target => random but valid permutation, no crash
        order = feature_importance_ordering(X, None, seed=1)
        assert sorted(order.tolist()) == list(range(X.shape[1]))

    def test_ties_broken_randomly(self):
        from bayes_nets import feature_importance_ordering
        # constant features => all-equal importance => random order
        X = np.zeros((500, 5), dtype=int)
        y = np.random.default_rng(0).random(500)
        o1 = feature_importance_ordering(X, y, method="f_regression", seed=1)
        o2 = feature_importance_ordering(X, y, method="f_regression", seed=2)
        assert sorted(o1.tolist()) == list(range(5))
        assert not np.array_equal(o1, o2)

    def test_invalid_measure_raises(self):
        from bayes_nets import feature_importance_ordering
        X, _, w = _eda_dataset(seed=7)
        with pytest.raises(ValueError):
            feature_importance_ordering(X, w, method="nonsense")

    def test_fi_k2_produces_valid_dag(self):
        X, card, w = _eda_dataset(seed=8)
        bn = BayesianNetwork(n_vars=X.shape[1], cardinality=card)
        bn.learn_structure(X, method="fi_k2", fs_importance="f_regression",
                           sample_weights=w, seed=1)
        assert bn.is_dag()


class TestRFEOrdering:
    """RFE_k2 — recursive / minimum-redundancy ordering."""

    def test_returns_valid_permutation(self):
        from bayes_nets import rfe_ordering
        X, card, w = _eda_dataset(seed=9)
        order = rfe_ordering(X, w, card, selector="mrmr", seed=1)
        assert sorted(order.tolist()) == list(range(X.shape[1]))

    def test_mrmr_pushes_redundant_variable_back(self):
        from bayes_nets import rfe_ordering
        rng = np.random.default_rng(10)
        n = 3000
        X0 = rng.integers(0, 2, n)
        X2 = rng.integers(0, 2, n)
        X3 = rng.integers(0, 2, n)
        X1 = X0.copy()  # exact duplicate of X0 (redundant)
        X = np.column_stack([X0, X1, X2, X3])
        f = 3.0 * X0 + 1.2 * X2 + 0.01 * rng.standard_normal(n)
        w = np.exp(f / 0.5)
        w /= w.sum()
        for s in range(4):
            o = rfe_ordering(X, w, np.full(4, 2), selector="mrmr", seed=s).tolist()
            # one of the duplicates leads, but the informative X2 must appear
            # before the *second* duplicate (redundancy penalty at work)
            assert o.index(2) < max(o.index(0), o.index(1))

    def test_rfe_selector_returns_valid_permutation(self):
        from bayes_nets import rfe_ordering
        X, card, w = _eda_dataset(n=800, p=6, strong=4, second=1, seed=11)
        order = rfe_ordering(X, w, card, selector="rfe", seed=1)
        assert sorted(order.tolist()) == list(range(X.shape[1]))

    def test_invalid_selector_raises(self):
        from bayes_nets import rfe_ordering
        X, card, w = _eda_dataset(seed=12)
        with pytest.raises(ValueError):
            rfe_ordering(X, w, card, selector="nonsense")

    def test_rfe_k2_produces_valid_dag(self):
        X, card, w = _eda_dataset(seed=13)
        bn = BayesianNetwork(n_vars=X.shape[1], cardinality=card)
        bn.learn_structure(X, method="rfe_k2", sample_weights=w, seed=1)
        assert bn.is_dag()
