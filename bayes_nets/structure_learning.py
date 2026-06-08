"""
Structure learning algorithms for Bayesian networks.

K2StructureLearner
    Greedy search using a fixed variable ordering (Cooper & Herskovits 1992).
    Guarantees acyclicity via the ordering constraint.

GreedyHillClimbLearner
    Unconstrained greedy hill-climbing with explicit cycle detection.
    Suitable for BIC/AIC scoring where no ordering is required.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from bayes_nets.scoring import ScoringMethod, K2ScoringMethod


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------


def _joint_table_size(cardinality: np.ndarray, variables: List[int]) -> int:
    """Number of cells in the joint table of *variables*."""
    if not variables:
        return 1
    return int(np.prod(cardinality[np.asarray(variables, dtype=int)]))


def _would_create_cycle(adjacency: np.ndarray, parent: int, child: int) -> bool:
    """Return True if adding edge parent → child would create a cycle."""
    n = adjacency.shape[0]
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
            if adjacency[node, nxt] and not visited[nxt]:
                stack.append(nxt)
    return False


# ---------------------------------------------------------------------------
# K2 algorithm
# ---------------------------------------------------------------------------


class K2StructureLearner:
    """Learn a BN structure using the K2 algorithm.

    The K2 algorithm (Cooper & Herskovits, 1992) requires a *fixed variable
    ordering* and greedily adds parents to each variable (respecting the
    ordering) as long as the score improves.

    Parameters
    ----------
    max_parents : int
        Maximum number of parents per variable.
    alpha : float
        Prior equivalent sample size for the K2 score.
    limit_table_size : bool
        Skip candidate parent sets whose joint table would exceed the
        number of training samples.
    """

    def __init__(
        self,
        max_parents: int = 3,
        alpha: float = 1.0,
        limit_table_size: bool = True,
    ) -> None:
        self.max_parents = max_parents
        self.alpha = alpha
        self.limit_table_size = limit_table_size
        self._scoring = K2ScoringMethod(alpha=alpha)

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        ordering: np.ndarray,
    ) -> np.ndarray:
        """Return adjacency matrix learned from *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        n_vars : int
        cardinality : np.ndarray, shape (n_vars,)
        ordering : np.ndarray
            Variable ordering – each variable may only have parents that
            appear *earlier* in this ordering.

        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
            Adjacency matrix (``adj[i, j] == 1`` ⟺ i → j).
        """
        n_samples = data.shape[0]
        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        for pos, var in enumerate(ordering):
            possible_parents: List[int] = list(ordering[:pos])
            current_parents: List[int] = []
            current_score = self._scoring.local_score(
                var, current_parents, data, cardinality
            )

            improved = True
            while improved and len(current_parents) < self.max_parents:
                improved = False
                best_parent = -1
                best_score = current_score

                for candidate in possible_parents:
                    if candidate in current_parents:
                        continue
                    test_parents = current_parents + [candidate]
                    if self.limit_table_size:
                        if _joint_table_size(cardinality, [var] + test_parents) > n_samples:
                            continue
                    score = self._scoring.local_score(
                        var, test_parents, data, cardinality
                    )
                    if score > best_score:
                        best_score = score
                        best_parent = candidate
                        improved = True

                if improved:
                    current_parents.append(best_parent)
                    current_score = best_score

            for parent in current_parents:
                adjacency[parent, var] = 1

        return adjacency


# ---------------------------------------------------------------------------
# Greedy hill-climbing
# ---------------------------------------------------------------------------


class GreedyHillClimbLearner:
    """Learn a BN structure using greedy hill-climbing.

    For each variable the algorithm greedily adds parents while the
    chosen scoring metric improves, detecting cycles explicitly so that
    no variable ordering is required.

    Parameters
    ----------
    scoring : ScoringMethod
        An instance of :class:`~bayes_nets.scoring.ScoringMethod`
        (e.g. :class:`~bayes_nets.scoring.BICScoringMethod`).
    max_parents : int
        Maximum number of parents per variable.
    limit_table_size : bool
        Skip candidate parent sets whose joint table would exceed the
        number of training samples.
    """

    def __init__(
        self,
        scoring: ScoringMethod,
        max_parents: int = 3,
        limit_table_size: bool = True,
    ) -> None:
        self.scoring = scoring
        self.max_parents = max_parents
        self.limit_table_size = limit_table_size

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """Return adjacency matrix learned from *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        n_vars : int
        cardinality : np.ndarray, shape (n_vars,)

        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
        """
        n_samples = data.shape[0]
        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        for var in range(n_vars):
            current_parents: List[int] = []
            current_score = self.scoring.local_score(
                var, current_parents, data, cardinality
            )

            for _ in range(min(self.max_parents, n_vars - 1)):
                best_parent = -1
                best_score = current_score

                for candidate in range(n_vars):
                    if candidate == var or candidate in current_parents:
                        continue
                    if _would_create_cycle(adjacency, candidate, var):
                        continue
                    test_parents = current_parents + [candidate]
                    if self.limit_table_size:
                        if _joint_table_size(cardinality, [var] + test_parents) > n_samples:
                            continue
                    score = self.scoring.local_score(
                        var, test_parents, data, cardinality
                    )
                    if score > best_score:
                        best_score = score
                        best_parent = candidate

                if best_parent >= 0:
                    current_parents.append(best_parent)
                    current_score = best_score
                    adjacency[best_parent, var] = 1
                else:
                    break

        return adjacency
