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

from typing import Dict, List, Optional, Tuple
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
        max_parents: int = 3,
        alpha: float = 1.0,
        ordering: Optional[np.ndarray] = None,
        limit_table_size: bool = True,
    ) -> "BayesianNetwork":
        """Learn structure **and** parameters from *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
            Observed discrete data. Values must be integers in
            ``[0, cardinality[j])`` for each column j.
        method : {"bic", "aic", "k2"}
            Scoring metric used for structure learning.
            ``"k2"`` uses the K2 algorithm with the given *ordering*;
            ``"bic"`` and ``"aic"`` use greedy hill-climbing.
        max_parents : int
            Maximum number of parents per variable.
        alpha : float
            Dirichlet/Laplace smoothing parameter (>= 0).
        ordering : array-like of int, optional
            Variable ordering for the K2 algorithm.  Ignored when
            ``method`` is ``"bic"`` or ``"aic"``.  Defaults to the
            natural order ``[0, 1, ..., n_vars-1]``.
        limit_table_size : bool
            If True, skip candidate parent sets whose joint table size
            exceeds the number of samples (avoids over-fitting).

        Returns
        -------
        self
        """
        data = np.asarray(data, dtype=int)
        self._validate_data(data)

        self.learn_structure(
            data,
            method=method,
            max_parents=max_parents,
            alpha=alpha,
            ordering=ordering,
            limit_table_size=limit_table_size,
        )
        self.learn_parameters(data, alpha=alpha)
        return self

    def learn_structure(
        self,
        data: np.ndarray,
        method: str = "bic",
        max_parents: int = 3,
        alpha: float = 1.0,
        ordering: Optional[np.ndarray] = None,
        limit_table_size: bool = True,
    ) -> "BayesianNetwork":
        """Learn the DAG structure from *data*.

        Resets any existing structure before learning.
        """
        from bayes_nets.structure_learning import (
            K2StructureLearner,
            GreedyHillClimbLearner,
        )
        from bayes_nets.scoring import (
            BICScoringMethod,
            AICScoringMethod,
            K2ScoringMethod,
        )

        data = np.asarray(data, dtype=int)

        method = method.lower()
        if method == "k2":
            if ordering is None:
                ordering = np.arange(self.n_vars)
            learner = K2StructureLearner(
                max_parents=max_parents,
                alpha=alpha,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality, np.asarray(ordering, dtype=int)
            )
        elif method in ("bic", "aic"):
            scoring_cls = BICScoringMethod if method == "bic" else AICScoringMethod
            scoring = scoring_cls(alpha=alpha)
            learner = GreedyHillClimbLearner(
                scoring=scoring,
                max_parents=max_parents,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(data, self.n_vars, self.cardinality)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'bic', 'aic', or 'k2'.")

        self.cpds = {}
        return self

    def learn_parameters(
        self, data: np.ndarray, alpha: float = 1.0
    ) -> "BayesianNetwork":
        """Estimate CPDs from *data* given the current structure.

        Uses maximum-likelihood estimation with optional Dirichlet
        (Laplace) smoothing controlled by *alpha*.
        """
        from bayes_nets.parameter_learning import MLEParameterLearner

        data = np.asarray(data, dtype=int)
        learner = MLEParameterLearner(alpha=alpha)
        self.cpds = learner.learn(data, self.n_vars, self.cardinality, self.adjacency)
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
