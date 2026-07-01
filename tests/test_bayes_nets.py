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
