"""
Core BayesianNetwork class.

Represents a discrete Bayesian network as a directed acyclic graph (DAG)
with conditional probability distributions (CPDs) for each variable.

The BN is parameterised by:
  - n_vars     : number of random variables
  - cardinality: 1-D integer array of length n_vars giving the number of
                 states for each variable.

Typical workflow
----------------
>>> import numpy as np
>>> from bayes_nets import BayesianNetwork
>>> data = np.random.randint(0, 2, size=(200, 4))
>>> bn = BayesianNetwork(n_vars=4, cardinality=np.array([2, 2, 2, 2]))
>>> bn.fit(data, method="bic")
>>> samples = bn.sample(100)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class BayesianNetwork:
    """
    Discrete Bayesian Network.

    Parameters
    ----------
    n_vars : int
        Number of random variables in the network.
    cardinality : array-like of int
        Number of discrete states for each variable.

    Attributes
    ----------
    n_vars : int
    cardinality : np.ndarray
    adjacency : np.ndarray, shape (n_vars, n_vars)
        Adjacency matrix.  ``adjacency[i, j] == 1`` means there is a
        directed edge i → j (i is a parent of j).
    cpds : dict
        Maps variable index to a dict ``{"parents": [...], "cpd": np.ndarray}``.
        For a root variable the CPD is a 1-D probability vector of length
        ``cardinality[var]``.  For a variable with parents it is a 2-D array
        of shape ``(n_parent_configs, cardinality[var])`` where each row is a
        conditional probability distribution.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, n_vars: int, cardinality) -> None:
        if n_vars < 1:
            raise ValueError("n_vars must be >= 1")
        self.n_vars: int = n_vars
        self.cardinality: np.ndarray = np.asarray(cardinality, dtype=int)
        if self.cardinality.shape != (n_vars,):
            raise ValueError("cardinality must have length n_vars")
        if np.any(self.cardinality < 2):
            raise ValueError("every variable must have cardinality >= 2")

        self.adjacency: np.ndarray = np.zeros((n_vars, n_vars), dtype=int)
        self.cpds: Dict[int, Dict] = {}

    # ------------------------------------------------------------------
    # Graph manipulation
    # ------------------------------------------------------------------

    def add_edge(self, parent: int, child: int) -> None:
        """Add a directed edge parent → child.

        Raises
        ------
        ValueError
            If the edge would introduce a cycle.
        """
        if parent == child:
            raise ValueError("Self-loops are not allowed")
        if self._would_create_cycle(parent, child):
            raise ValueError(
                f"Adding edge {parent} → {child} would create a cycle"
            )
        self.adjacency[parent, child] = 1

    def remove_edge(self, parent: int, child: int) -> None:
        """Remove the directed edge parent → child (if it exists)."""
        self.adjacency[parent, child] = 0

    def has_edge(self, parent: int, child: int) -> bool:
        """Return True if the edge parent → child exists."""
        return bool(self.adjacency[parent, child])

    def get_parents(self, var: int) -> List[int]:
        """Return list of parent indices for *var*."""
        return list(np.where(self.adjacency[:, var] > 0)[0])

    def get_children(self, var: int) -> List[int]:
        """Return list of child indices for *var*."""
        return list(np.where(self.adjacency[var, :] > 0)[0])

    def is_dag(self) -> bool:
        """Return True if the current structure is a valid DAG."""
        try:
            self._topological_sort()
            return True
        except ValueError:
            return False

    def topological_order(self) -> np.ndarray:
        """Return variable indices in topological order."""
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        method: str = "bic",
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
        ordering: Optional[np.ndarray] = None,
        limit_table_size: bool = True,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "BayesianNetwork":
        """Learn structure **and** parameters from *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
            Observed discrete data.  Values must be integers in
            ``[0, cardinality[j])`` for each column j.
        method : {"bic", "aic", "k2", "stable_hc", "tabu", "gs", "rcd", "rpcd"}
            Scoring / search algorithm.
        max_parents : int or None
            Maximum parents per variable.  ``None`` → rule of thumb
            ``max(1, floor(10·log2/log(max_cardinality)))``
            (10 for binary variables).
        alpha : float
            Dirichlet/Laplace smoothing parameter (>= 0).
        ordering : array-like of int, optional
            Legacy K2 ordering parameter.  Use ``permutation`` instead.
        limit_table_size : bool
            Skip parent sets whose joint table exceeds n_samples.
        permutation : array-like of int, optional
            Permutation σ of [0 … n_vars-1].  Parents of σ(j) are
            restricted to {σ(i) : i < j}.  Applies to all methods.
            ``None`` → unconstrained (cycle detection used for HC).
        interaction_matrix : np.ndarray, shape (n_vars, n_vars), optional
            Symmetric binary matrix.  Edge u → v considered only when
            ``interaction_matrix[u, v] == 1``.  ``None`` → all allowed.
        sample_weights : array of float, shape (n_samples,), optional
            Probability vector (must sum to 1).  Weighted counts replace
            raw counts during structure and parameter learning.
            ``None`` → uniform 1/N.

        Returns
        -------
        self
        """
        data = np.asarray(data, dtype=int)
        self._validate_data(data)
        self._validate_new_params(data.shape[0], permutation, interaction_matrix, sample_weights)

        self.learn_structure(
            data,
            method=method,
            max_parents=max_parents,
            alpha=alpha,
            ordering=ordering,
            limit_table_size=limit_table_size,
            permutation=permutation,
            interaction_matrix=interaction_matrix,
            sample_weights=sample_weights,
        )
        self.learn_parameters(data, alpha=alpha, sample_weights=sample_weights)
        return self

    def learn_structure(
        self,
        data: np.ndarray,
        method: str = "bic",
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
        ordering: Optional[np.ndarray] = None,
        limit_table_size: bool = True,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "BayesianNetwork":
        """Learn the DAG structure from *data*.

        Resets any existing structure before learning.  See :meth:`fit`
        for full parameter documentation.
        """
        from bayes_nets.structure_learning import (
            K2StructureLearner,
            GreedyHillClimbLearner,
            StableHillClimbLearner,
            TabuHillClimbLearner,
            GrowShrinkLearner,
            RecursiveCDLearner,
            RPCDLearner,
        )
        from bayes_nets.scoring import BICScoringMethod, AICScoringMethod

        data = np.asarray(data, dtype=int)

        # permutation overrides legacy ordering for K2
        eff_perm = permutation if permutation is not None else (
            np.asarray(ordering, dtype=int) if ordering is not None else None
        )

        method = method.lower()

        learn_kwargs = dict(
            permutation=eff_perm,
            interaction_matrix=interaction_matrix,
            sample_weights=sample_weights,
        )

        if method == "k2":
            learner = K2StructureLearner(
                max_parents=max_parents,
                alpha=alpha,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(data, self.n_vars, self.cardinality, **learn_kwargs)

        elif method in ("bic", "aic"):
            scoring_cls = BICScoringMethod if method == "bic" else AICScoringMethod
            scoring = scoring_cls(alpha=alpha, sample_weights=sample_weights)
            learner = GreedyHillClimbLearner(
                scoring=scoring,
                max_parents=max_parents,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                # sample_weights already embedded in scoring object
            )

        elif method in ("stable_hc", "tabu"):
            scoring = BICScoringMethod(alpha=alpha, sample_weights=sample_weights)
            cls = StableHillClimbLearner if method == "stable_hc" else TabuHillClimbLearner
            learner = cls(
                scoring=scoring,
                max_parents=max_parents,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
            )

        elif method in ("gs", "rcd", "rpcd"):
            if method == "gs":
                learner_cls = GrowShrinkLearner
            elif method == "rcd":
                learner_cls = RecursiveCDLearner
            else:
                learner_cls = RPCDLearner
            learner = learner_cls(
                alpha_ci=alpha if alpha > 0 else 0.05,
                max_parents=max_parents,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose 'bic', 'aic', 'k2', 'stable_hc', 'tabu', 'gs', 'rcd', or 'rpcd'."
            )

        self.cpds = {}
        return self

    def learn_parameters(
        self,
        data: np.ndarray,
        alpha: float = 1.0,
        *,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "BayesianNetwork":
        """Estimate CPDs from *data* given the current structure.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        alpha : float
            Dirichlet/Laplace smoothing.
        sample_weights : array of float, shape (n_samples,), optional
            Probability vector (must sum to 1).
        """
        from bayes_nets.parameter_learning import MLEParameterLearner

        data = np.asarray(data, dtype=int)
        learner = MLEParameterLearner(alpha=alpha)
        self.cpds = learner.learn(
            data, self.n_vars, self.cardinality, self.adjacency,
            sample_weights=sample_weights,
        )
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw *n_samples* i.i.d. samples from the BN.

        Uses probabilistic logic sampling (ancestral sampling).

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray, shape (n_samples, n_vars)
        """
        if not self.cpds:
            raise RuntimeError(
                "CPDs have not been estimated yet. Call learn_parameters() or fit() first."
            )
        from bayes_nets.sampling import ProbabilisticLogicSampler

        sampler = ProbabilisticLogicSampler()
        return sampler.sample(
            n_samples=n_samples,
            n_vars=self.n_vars,
            cardinality=self.cardinality,
            adjacency=self.adjacency,
            cpds=self.cpds,
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Visualise the BN structure using matplotlib/networkx.

        Keyword arguments are forwarded to
        :func:`bayes_nets.visualization.plot_bayesian_network`.
        """
        from bayes_nets.visualization import plot_bayesian_network

        return plot_bayesian_network(self, **kwargs)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def to_adjacency_matrix(self) -> np.ndarray:
        """Return a copy of the adjacency matrix."""
        return self.adjacency.copy()

    def big_matrix(self) -> np.ndarray:
        """Alias for the adjacency matrix used by visualization pipelines."""
        return self.to_adjacency_matrix()

    def edge_list(self) -> List[tuple[int, int]]:
        """Return directed edges as (parent, child) tuples."""
        edges = np.argwhere(self.adjacency > 0)
        return [(int(parent), int(child)) for parent, child in edges]

    def to_run_structure(self, generation: int, run: int = 0) -> Dict:
        """Return a run-structure entry compatible with EDA visualizers."""
        return {
            "adjacency": self.big_matrix(),
            "generation": int(generation),
            "run": int(run),
        }

    def set_structure(self, adjacency: np.ndarray) -> "BayesianNetwork":
        """Set the graph structure from an adjacency matrix.

        Parameters
        ----------
        adjacency : np.ndarray, shape (n_vars, n_vars)
            ``adjacency[i, j] == 1`` ⟺ edge i → j.
        """
        adj = np.asarray(adjacency, dtype=int)
        if adj.shape != (self.n_vars, self.n_vars):
            raise ValueError("adjacency must have shape (n_vars, n_vars)")
        self.adjacency = adj.copy()
        self.cpds = {}
        return self

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def n_parameters(self) -> int:
        """Total number of free parameters in the CPDs."""
        total = 0
        for var in range(self.n_vars):
            parents = self.get_parents(var)
            k = int(self.cardinality[var])
            n_parent_configs = int(
                np.prod([self.cardinality[p] for p in parents]) if parents else 1
            )
            total += n_parent_configs * (k - 1)
        return total

    def marginal(self, var: int, data: np.ndarray) -> np.ndarray:
        """Empirical marginal distribution of *var* in *data*.

        Parameters
        ----------
        var : int
        data : np.ndarray, shape (n_samples, n_vars)

        Returns
        -------
        np.ndarray, shape (cardinality[var],)
        """
        k = int(self.cardinality[var])
        counts = np.bincount(data[:, var], minlength=k).astype(float)
        return counts / counts.sum()

    def structure_score(self, data: np.ndarray, score: str = "bic", alpha: float = 0.0) -> float:
        """Score the current structure on data using BIC/AIC/K2."""
        from bayes_nets.scoring import BICScoringMethod, AICScoringMethod, K2ScoringMethod

        method = score.lower()
        if method == "bic":
            scorer = BICScoringMethod(alpha=alpha)
        elif method == "aic":
            scorer = AICScoringMethod(alpha=alpha)
        elif method == "k2":
            scorer = K2ScoringMethod(alpha=max(alpha, 1e-12))
        else:
            raise ValueError("score must be 'bic', 'aic', or 'k2'")
        return float(scorer.score(self.adjacency, np.asarray(data, dtype=int), self.cardinality))

    def markov_blanket(self, var: int) -> List[int]:
        """Return Markov blanket: parents, children, and children's other parents."""
        parents = set(self.get_parents(var))
        children = set(self.get_children(var))
        spouses: set[int] = set()
        for child in children:
            spouses.update(self.get_parents(child))
        spouses.discard(var)
        blanket = sorted(parents.union(children).union(spouses))
        return [int(v) for v in blanket]

    def variable_dependencies(self, data: np.ndarray, score: str = "bic", alpha: float = 0.0) -> Dict:
        """Return dependency analysis summary compatible with EDA consumers."""
        data = np.asarray(data, dtype=int)
        n_vars = data.shape[1]
        mi = np.zeros((n_vars, n_vars), dtype=float)

        for i in range(n_vars):
            xi = data[:, i]
            card_i = int(self.cardinality[i])
            p_i = np.bincount(xi, minlength=card_i).astype(float)
            p_i /= p_i.sum()
            for j in range(i + 1, n_vars):
                xj = data[:, j]
                card_j = int(self.cardinality[j])

                pair_idx = xi + card_i * xj
                p_ij = np.bincount(pair_idx, minlength=card_i * card_j).astype(float)
                p_ij = p_ij.reshape(card_j, card_i).T
                p_ij /= p_ij.sum()

                p_j = np.bincount(xj, minlength=card_j).astype(float)
                p_j /= p_j.sum()

                val = 0.0
                for ai in range(card_i):
                    for aj in range(card_j):
                        if p_ij[ai, aj] > 0 and p_i[ai] > 0 and p_j[aj] > 0:
                            val += p_ij[ai, aj] * np.log2(p_ij[ai, aj] / (p_i[ai] * p_j[aj]))
                mi[i, j] = val
                mi[j, i] = val

        return {
            "adjacency_matrix": self.big_matrix(),
            "edges": self.edge_list(),
            "score": self.structure_score(data, score=score, alpha=alpha),
            "mi_matrix": mi,
            "parents": {var: self.get_parents(var) for var in range(self.n_vars)},
            "markov_blankets": {var: self.markov_blanket(var) for var in range(self.n_vars)},
            "degree": {
                var: int(self.adjacency[var, :].sum() + self.adjacency[:, var].sum())
                for var in range(self.n_vars)
            },
        }

    def to_factorization(
        self,
        data: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        max_clique_width: Optional[int] = None,
        width_control: str = "split",
        triangulation_method: str = "min-fill",
    ):
        """Convert the BN into a clique factorization for FDA-style sampling."""
        from bayes_nets.factorization import bn_to_factorization

        return bn_to_factorization(
            adjacency=self.adjacency,
            cardinality=self.cardinality,
            cpds=self.cpds,
            data=data,
            alpha=alpha,
            max_clique_width=max_clique_width,
            width_control=width_control,
            triangulation_method=triangulation_method,
        )

    def most_probable_config(self, evidence: Optional[Dict[int, int]] = None):
        """Return the most probable assignment and its probability."""
        from bayes_nets.inference import MaxProductInference

        return MaxProductInference(self).most_probable_config(evidence=evidence)

    def k_most_probable_configs(
        self,
        k: int,
        evidence: Optional[Dict[int, int]] = None,
        search_method: str = "nilsson",
    ):
        """Return top-k assignments sorted by decreasing probability."""
        from bayes_nets.inference import MaxProductInference

        return MaxProductInference(self).k_most_probable_configs(
            k=k,
            evidence=evidence,
            search_method=search_method,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_edges = int(self.adjacency.sum())
        return (
            f"BayesianNetwork(n_vars={self.n_vars}, "
            f"cardinality={self.cardinality.tolist()}, "
            f"n_edges={n_edges})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_data(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError("data must be a 2-D array (n_samples, n_vars)")
        if data.shape[1] != self.n_vars:
            raise ValueError(
                f"data has {data.shape[1]} columns but n_vars={self.n_vars}"
            )

    def _validate_new_params(
        self,
        n_samples: int,
        permutation: Optional[np.ndarray],
        interaction_matrix: Optional[np.ndarray],
        sample_weights: Optional[np.ndarray],
    ) -> None:
        if permutation is not None:
            p = np.asarray(permutation, dtype=int)
            if p.shape != (self.n_vars,) or set(p.tolist()) != set(range(self.n_vars)):
                raise ValueError(
                    "permutation must be a valid permutation of [0, ..., n_vars-1]"
                )
        if interaction_matrix is not None:
            im = np.asarray(interaction_matrix)
            if im.shape != (self.n_vars, self.n_vars):
                raise ValueError(
                    f"interaction_matrix must have shape ({self.n_vars}, {self.n_vars})"
                )
            if not np.array_equal(im, im.T):
                raise ValueError("interaction_matrix must be symmetric")
        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=float)
            if sw.shape != (n_samples,):
                raise ValueError(
                    f"sample_weights must have shape ({n_samples},)"
                )
            if sw.min() < 0:
                raise ValueError("sample_weights must be non-negative")
            total = sw.sum()
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(
                    f"sample_weights must sum to 1 (got {total:.6f})"
                )

    def _would_create_cycle(self, parent: int, child: int) -> bool:
        """Return True if adding edge parent → child would create a cycle."""
        n = self.n_vars
        visited = np.zeros(n, dtype=bool)
        stack = [child]
        while stack:
            node = stack.pop()
            if node == parent:
                return True
            if visited[node]:
                continue
            visited[node] = True
            for nxt in range(n):
                if self.adjacency[node, nxt] and not visited[nxt]:
                    stack.append(nxt)
        return False

    def _topological_sort(self) -> np.ndarray:
        """Kahn's algorithm for topological sorting."""
        n = self.n_vars
        in_degree = self.adjacency.sum(axis=0).copy()
        queue = list(np.where(in_degree == 0)[0])
        order: List[int] = []
        while queue:
            var = queue.pop(0)
            order.append(var)
            for child in range(n):
                if self.adjacency[var, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        if len(order) != n:
            raise ValueError("Graph contains a cycle – not a valid DAG")
        return np.array(order, dtype=int)
