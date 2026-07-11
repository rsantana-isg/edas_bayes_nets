"""
Structure learning algorithms for Bayesian networks.

All learners accept four common optional parameters in their ``learn()``
method (described in detail below):

- ``permutation``      – variable ordering that restricts parent search
- ``interaction_matrix`` – symmetric binary matrix of allowed interactions
- ``max_parents``      – maximum parents per variable (rule-of-thumb if None)
- ``sample_weights``   – probability vector over data rows

K2StructureLearner
    Greedy search using a fixed variable ordering (Cooper & Herskovits 1992).

GreedyHillClimbLearner
    Greedy add-only hill-climbing with explicit cycle detection.

StableHillClimbLearner
    HC over add / delete / reverse with deterministic tie-breaking
    (HC-Stable, Kitson & Constantinou 2023).

TabuHillClimbLearner
    HC-Stable extended with a tabu list (Kitson & Constantinou 2023).

GrowShrinkLearner
    Grow-Shrink Markov-blanket structure induction
    (Margaritis & Thrun 1999).

RecursiveCDLearner
    Recursive causal discovery with conditional-independence tests.

PCLearner
    PC algorithm for constraint-based skeleton + orientation
    (Spirtes & Glymour 1991).

StablePCLearner
    Order-independent (Stable-PC) variant: all CI tests at a given
    conditioning-set size are collected before any edge is removed,
    eliminating the dependence on variable ordering.

DecisionTreeLearner
    HC structure search scored with a decision-tree MDL metric.
    Local structure in CPDs is captured by growing a CART-BIC tree
    over parent variables (Friedman & Goldszmidt 1996).

DecisionGraphLearner
    HC structure search scored with a decision-graph Bayesian metric.
    Extends decision trees with a leaf-merging step that detects and
    exploits parameter equalities (Chickering, Heckerman & Meek 1997).

Common parameters
-----------------
permutation : array-like of int, optional
    A permutation σ of [0 … n_vars-1].  Parents of σ(j) are restricted to
    {σ(i) : i < j}.  This guarantees acyclicity for the ADD operation and
    makes explicit cycle detection unnecessary.  When ``None`` (default)
    no ordering constraint is imposed and cycle detection is used as before.
    Reversal operations are skipped whenever the permutation constraint would
    prevent the reversed edge (including all permutation-constrained cases).

interaction_matrix : np.ndarray of shape (n_vars, n_vars), optional
    Symmetric binary matrix.  An edge u → v is considered only when
    ``interaction_matrix[u, v] == 1``.  ``None`` means all pairs are
    allowed (equivalent to an all-ones matrix).

max_parents : int or None
    Maximum parents per variable.  When ``None``, a rule of thumb is used:
    ``max(1, floor(10 · log(2) / log(max_cardinality)))``.
    For all-binary variables this gives 10; higher cardinalities reduce it.

sample_weights : array of float, shape (n_samples,), optional
    Probability distribution over rows (must sum to 1).  Used to compute
    weighted counts for the scoring function.  ``None`` uses uniform 1/N.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2 as chi2_dist

from bayes_nets.scoring import ScoringMethod, K2ScoringMethod


# ---------------------------------------------------------------------------
# Shared utilities
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
# Common-parameter helpers
# ---------------------------------------------------------------------------


def _default_max_parents(cardinality: np.ndarray) -> int:
    """Rule-of-thumb: keep table sizes comparable to binary with mP=10.

    Returns ``max(1, floor(10 · log2 / log(k_max)))`` where k_max is the
    maximum cardinality.  Examples: k=2 → 10, k=3 → 6, k=4 → 5, k=10 → 3.
    """
    k_max = max(2, int(np.max(cardinality)))
    return max(1, int(10 * np.log(2) / np.log(k_max)))


def _compute_perm_pos(n_vars: int, permutation: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return position-in-permutation for each variable index.

    ``perm_pos[v]`` = j such that permutation[j] = v.
    Returns ``None`` when ``permutation`` is None (no constraint).
    """
    if permutation is None:
        return None
    perm_pos = np.empty(n_vars, dtype=int)
    for j, v in enumerate(permutation):
        perm_pos[int(v)] = j
    return perm_pos


def _build_allowed_parents(
    n_vars: int,
    perm_pos: Optional[np.ndarray],
    interaction_matrix: Optional[np.ndarray],
) -> Dict[int, List[int]]:
    """Compute the set of allowed parents for every variable.

    An edge u → v is allowed when:
    1. permutation constraint: perm_pos[u] < perm_pos[v]  (or no constraint)
    2. interaction constraint: interaction_matrix[u, v] == 1  (or no matrix)
    """
    allowed: Dict[int, List[int]] = {}
    for v in range(n_vars):
        parents = []
        for u in range(n_vars):
            if u == v:
                continue
            if perm_pos is not None and perm_pos[u] >= perm_pos[v]:
                continue
            if interaction_matrix is not None and interaction_matrix[u, v] == 0:
                continue
            parents.append(u)
        allowed[v] = parents
    return allowed


# ---------------------------------------------------------------------------
# K2 algorithm
# ---------------------------------------------------------------------------


class K2StructureLearner:
    """Learn a BN structure using the K2 algorithm.

    Greedy search over a fixed variable ordering (Cooper & Herskovits 1992).

    Parameters
    ----------
    max_parents : int or None
        Maximum parents per variable.  ``None`` → rule of thumb.
    alpha : float
        Prior equivalent sample size for the K2 score.
    limit_table_size : bool
        Skip candidate parent sets whose joint table would exceed the
        number of training samples.
    """

    def __init__(
        self,
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
        limit_table_size: bool = True,
    ) -> None:
        self.max_parents = max_parents
        self.alpha = alpha
        self.limit_table_size = limit_table_size

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        ordering: Optional[np.ndarray] = None,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return adjacency matrix learned from *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        n_vars : int
        cardinality : np.ndarray, shape (n_vars,)
        ordering : array-like of int, optional
            Legacy positional argument; use ``permutation`` instead.
            If both are given, ``permutation`` takes precedence.
        permutation : array-like of int, optional
            Variable ordering σ.  Defaults to [0, 1, ..., n_vars-1].
        interaction_matrix : np.ndarray, optional
        sample_weights : array of float, optional

        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
        """
        n_samples = data.shape[0]
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        # Resolve ordering: permutation > ordering > natural
        if permutation is not None:
            eff_order = np.asarray(permutation, dtype=int)
        elif ordering is not None:
            eff_order = np.asarray(ordering, dtype=int)
        else:
            eff_order = np.arange(n_vars, dtype=int)

        scoring = K2ScoringMethod(alpha=self.alpha, sample_weights=sample_weights)
        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        for pos, var in enumerate(eff_order):
            # Candidate parents: earlier in ordering AND allowed by interaction_matrix
            possible: List[int] = [
                int(eff_order[i]) for i in range(pos)
                if interaction_matrix is None or interaction_matrix[int(eff_order[i]), int(var)] != 0
            ]
            current_parents: List[int] = []
            current_score = scoring.local_score(var, current_parents, data, cardinality)

            improved = True
            while improved and len(current_parents) < mp:
                improved = False
                best_parent = -1
                best_score = current_score

                for candidate in possible:
                    if candidate in current_parents:
                        continue
                    test_parents = current_parents + [candidate]
                    if self.limit_table_size:
                        if _joint_table_size(cardinality, [var] + test_parents) > n_samples:
                            continue
                    s = scoring.local_score(var, test_parents, data, cardinality)
                    if s > best_score:
                        best_score = s
                        best_parent = candidate
                        improved = True

                if improved:
                    current_parents.append(best_parent)
                    current_score = best_score

            for parent in current_parents:
                adjacency[parent, var] = 1

        return adjacency


# ---------------------------------------------------------------------------
# DMBBN algorithm (Dâmaso et al. 2026)
# ---------------------------------------------------------------------------


class DMBBNStructureLearner:
    """Learn a BN structure using the DMBBN algorithm.

    DMBBN (Dynamic Markov Blanket Bayesian Network) induces local structures
    for each variable independently using a Markov-blanket heuristic, then
    combines them into a global DAG using an adapted Kruskal's algorithm
    (Dâmaso et al. 2026).

    Parameters
    ----------
    max_parents : int or None
        Maximum parents per variable.  ``None`` → rule of thumb.
    alpha : float
        Prior equivalent sample size for the K2 score.
    limit_table_size : bool
        Skip candidate parent sets whose joint table would exceed the
        number of training samples.
    """

    def __init__(
        self,
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
        limit_table_size: bool = True,
    ) -> None:
        self.max_parents = max_parents
        self.alpha = alpha
        self.limit_table_size = limit_table_size

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return adjacency matrix learned from *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        n_vars : int
        cardinality : np.ndarray, shape (n_vars,)
        permutation : array-like of int, optional
            Ignored by DMBBN as it is designed to be order-independent.
        interaction_matrix : np.ndarray, optional
        sample_weights : array of float, optional

        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
        """
        n_samples = data.shape[0]
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)
        scoring = K2ScoringMethod(alpha=self.alpha, sample_weights=sample_weights)

        # 1. Induce n local structures (one for each variable as root)
        # adj_counts[u, v] stores how many times edge u -> v appeared in local DAGs
        adj_counts = np.zeros((n_vars, n_vars), dtype=int)

        for root in range(n_vars):
            local_edges = self._induce_local_structure(
                root, data, n_vars, cardinality, scoring, mp, n_samples, interaction_matrix
            )
            for u, v in local_edges:
                adj_counts[u, v] += 1

        # 2. Pre-process edges: resolve direction ties and select dominant direction
        candidates: List[Tuple[int, int, int, int]] = []
        for u in range(n_vars):
            for v in range(u + 1, n_vars):
                c_uv = adj_counts[u, v]
                c_vu = adj_counts[v, u]

                if c_uv == 0 and c_vu == 0:
                    continue
                
                if c_uv == c_vu:
                    continue  # Remove tied edges

                if c_uv > c_vu:
                    # Edge u -> v
                    candidates.append((u, v, c_uv + c_vu, c_uv - c_vu))
                else:
                    # Edge v -> u
                    candidates.append((v, u, c_uv + c_vu, c_vu - c_uv))
        
        # 3. Build final DAG using adapted Kruskal's algorithm
        # Sort by weight (sum) descending, then priority (diff) descending
        candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

        final_adj = np.zeros((n_vars, n_vars), dtype=int)
        for u, v, _, _ in candidates:
            # Check max_parents constraint for the child v
            if np.sum(final_adj[:, v]) >= mp:
                continue
            # Check acyclicity
            if not _would_create_cycle(final_adj, u, v):
                final_adj[u, v] = 1

        return final_adj

    def _induce_local_structure(
        self,
        root: int,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        scoring: K2ScoringMethod,
        mp: int,
        n_samples: int,
        interaction_matrix: Optional[np.ndarray],
    ) -> List[Tuple[int, int]]:
        """Induce local structure for *root* using Modified-DMBC logic."""
        local_adj = np.zeros((n_vars, n_vars), dtype=int)
        
        # We follow a node ordering where 'root' is processed first.
        # This ensures that for other nodes, we can check if 'root' is a parent.
        nodes_order = [root] + [v for v in range(n_vars) if v != root]

        for target in nodes_order:
            initial_parents = []
            if target != root:
                # Modified-DMBC line 15 check: 
                # Only proceed if root is a suitable parent for this target.
                if interaction_matrix is not None and interaction_matrix[root, target] == 0:
                    continue
                
                if self.limit_table_size:
                    if _joint_table_size(cardinality, [target, root]) > n_samples:
                        continue

                base_score = scoring.local_score(target, [], data, cardinality)
                with_root_score = scoring.local_score(target, [root], data, cardinality)
                
                if with_root_score > base_score:
                    initial_parents = [root]
                else:
                    # Not a child of root, skip.
                    continue
            
            # Greedily find other parents for target (could be root's parents or spouses)
            current_parents = list(initial_parents)
            current_score = scoring.local_score(target, current_parents, data, cardinality)

            while len(current_parents) < mp:
                improved = False
                best_p = -1
                best_s = current_score

                for candidate in range(n_vars):
                    if candidate == target or candidate in current_parents:
                        continue
                    if interaction_matrix is not None and interaction_matrix[candidate, target] == 0:
                        continue
                    
                    # Ensure the local structure remains a DAG
                    if _would_create_cycle(local_adj, candidate, target):
                        continue

                    test_parents = current_parents + [candidate]
                    if self.limit_table_size:
                        if _joint_table_size(cardinality, [target] + test_parents) > n_samples:
                            continue
                    
                    s = scoring.local_score(target, test_parents, data, cardinality)
                    if s > best_s:
                        best_s = s
                        best_p = candidate
                        improved = True

                if improved:
                    current_parents.append(best_p)
                    current_score = best_s
                else:
                    break
            
            # Commit to local structure
            for p in current_parents:
                local_adj[p, target] = 1

        # Extract edges from local DAG
        edges: List[Tuple[int, int]] = []
        u_idx, v_idx = np.where(local_adj > 0)
        for u, v in zip(u_idx, v_idx):
            edges.append((int(u), int(v)))
            
        return edges


# ---------------------------------------------------------------------------
# Greedy hill-climbing (add-only)
# ---------------------------------------------------------------------------


class GreedyHillClimbLearner:
    """Learn a BN structure using greedy add-only hill-climbing.

    Considers adding one parent at a time per variable, using cycle
    detection when no permutation is given.

    Parameters
    ----------
    scoring : ScoringMethod
    max_parents : int or None
    limit_table_size : bool
    """

    def __init__(
        self,
        scoring: ScoringMethod,
        max_parents: Optional[int] = None,
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
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return adjacency matrix learned from *data*."""
        n_samples = data.shape[0]
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        perm_pos = _compute_perm_pos(n_vars, permutation)
        allowed = _build_allowed_parents(n_vars, perm_pos, interaction_matrix)
        perm_constrained = perm_pos is not None

        scorer = self.scoring if sample_weights is None else self.scoring.with_weights(sample_weights)

        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        for var in range(n_vars):
            current_parents: List[int] = []
            current_score = scorer.local_score(var, current_parents, data, cardinality)

            for _ in range(mp):
                best_parent = -1
                best_score = current_score

                for candidate in allowed[var]:
                    if candidate in current_parents:
                        continue
                    if not perm_constrained and _would_create_cycle(adjacency, candidate, var):
                        continue
                    test_parents = current_parents + [candidate]
                    if self.limit_table_size:
                        if _joint_table_size(cardinality, [var] + test_parents) > n_samples:
                            continue
                    s = scorer.local_score(var, test_parents, data, cardinality)
                    if s > best_score:
                        best_score = s
                        best_parent = candidate

                if best_parent >= 0:
                    current_parents.append(best_parent)
                    current_score = best_score
                    adjacency[best_parent, var] = 1
                else:
                    break

        return adjacency


# ---------------------------------------------------------------------------
# HC-Stable helpers
# ---------------------------------------------------------------------------

_OP_PRIORITY = {"add": 0, "del": 1, "rev": 2}


def _op_key(delta: float, op: str, u: int, v: int) -> tuple:
    """Comparison key: higher delta wins; ties broken by op type then (u, v)."""
    return (delta, -_OP_PRIORITY[op], -u, -v)


# ---------------------------------------------------------------------------
# HC-Stable (Kitson & Constantinou 2023)
# ---------------------------------------------------------------------------


class StableHillClimbLearner:
    """Learn a BN structure via stable greedy hill-climbing.

    Considers add / delete / reverse operations globally in each iteration
    and breaks score ties deterministically (HC-Stable).  Reversal is
    skipped when the permutation constraint makes it impossible.

    Parameters
    ----------
    scoring : ScoringMethod
    max_parents : int or None
    max_iter : int
    limit_table_size : bool
    """

    def __init__(
        self,
        scoring: ScoringMethod,
        max_parents: Optional[int] = None,
        max_iter: int = 500,
        limit_table_size: bool = True,
    ) -> None:
        self.scoring = scoring
        self.max_parents = max_parents
        self.max_iter = max_iter
        self.limit_table_size = limit_table_size

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return adjacency matrix learned from *data*."""
        n_samples = data.shape[0]
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        perm_pos = _compute_perm_pos(n_vars, permutation)
        allowed = _build_allowed_parents(n_vars, perm_pos, interaction_matrix)
        perm_constrained = perm_pos is not None

        scorer = self.scoring if sample_weights is None else self.scoring.with_weights(sample_weights)
        cache: dict = {}

        def local(var: int, parents: List[int]) -> float:
            key = (var, tuple(sorted(parents)))
            if key not in cache:
                cache[key] = scorer.local_score(var, list(parents), data, cardinality)
            return cache[key]

        def parents_of(var: int) -> List[int]:
            return list(np.where(adjacency[:, var] > 0)[0])

        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        for _ in range(self.max_iter):
            best_key = None
            best_op: Optional[tuple] = None

            for u in range(n_vars):
                for v in range(n_vars):
                    if u == v:
                        continue

                    # ADD u → v
                    if adjacency[u, v] == 0 and adjacency[v, u] == 0 and u in allowed[v]:
                        pa_v = parents_of(v)
                        if len(pa_v) < mp:
                            if perm_constrained or not _would_create_cycle(adjacency, u, v):
                                new_pa = pa_v + [u]
                                if not self.limit_table_size or _joint_table_size(cardinality, [v] + new_pa) <= n_samples:
                                    delta = local(v, new_pa) - local(v, pa_v)
                                    k = _op_key(delta, "add", u, v)
                                    if best_key is None or k > best_key:
                                        best_key, best_op = k, ("add", u, v)

                    # DELETE u → v
                    if adjacency[u, v] == 1:
                        pa_v = parents_of(v)
                        new_pa = [p for p in pa_v if p != u]
                        delta = local(v, new_pa) - local(v, pa_v)
                        k = _op_key(delta, "del", u, v)
                        if best_key is None or k > best_key:
                            best_key, best_op = k, ("del", u, v)

                    # REVERSE u → v  becomes  v → u
                    # Only possible if v is an allowed parent of u.
                    if adjacency[u, v] == 1 and v in allowed[u]:
                        pa_u = parents_of(u)
                        pa_v = parents_of(v)
                        if len(pa_u) < mp:
                            can_rev: bool
                            if perm_constrained:
                                can_rev = True  # ordering already guaranteed by allowed[u]
                            else:
                                adjacency[u, v] = 0
                                can_rev = not _would_create_cycle(adjacency, v, u)
                                adjacency[u, v] = 1
                            if can_rev:
                                new_pa_u = pa_u + [v]
                                new_pa_v = [p for p in pa_v if p != u]
                                if not self.limit_table_size or _joint_table_size(cardinality, [u] + new_pa_u) <= n_samples:
                                    delta = (
                                        local(v, new_pa_v) + local(u, new_pa_u)
                                        - local(v, pa_v) - local(u, pa_u)
                                    )
                                    k = _op_key(delta, "rev", u, v)
                                    if best_key is None or k > best_key:
                                        best_key, best_op = k, ("rev", u, v)

            if best_op is None or best_key[0] <= 0.0:
                break

            op, u, v = best_op
            if op == "add":
                adjacency[u, v] = 1
            elif op == "del":
                adjacency[u, v] = 0
            else:  # rev
                adjacency[u, v] = 0
                adjacency[v, u] = 1
            for key in [k for k in cache if k[0] in (u, v)]:
                cache.pop(key, None)

        return adjacency


# ---------------------------------------------------------------------------
# Tabu-Stable (Kitson & Constantinou 2023)
# ---------------------------------------------------------------------------


class TabuHillClimbLearner:
    """Learn a BN structure via Tabu-stable hill-climbing.

    Extends HC-Stable with a tabu list to escape local optima.  An
    aspiration criterion overrides tabu status when the move strictly
    improves the global best score.

    Parameters
    ----------
    scoring : ScoringMethod
    max_parents : int or None
    max_iter : int
    tabu_length : int
        Number of recent operations kept in the tabu list.
    limit_table_size : bool
    """

    def __init__(
        self,
        scoring: ScoringMethod,
        max_parents: Optional[int] = None,
        max_iter: int = 1000,
        tabu_length: int = 10,
        patience: Optional[int] = None,
        limit_table_size: bool = True,
    ) -> None:
        self.scoring = scoring
        self.max_parents = max_parents
        self.max_iter = max_iter
        self.tabu_length = tabu_length
        # patience: stop after this many consecutive iterations without improving
        # the global best.  None → 5 * tabu_length (enough to escape local optima).
        self.patience = patience
        self.limit_table_size = limit_table_size

    @staticmethod
    def _reverse_op(op: str, u: int, v: int) -> tuple:
        if op == "add":
            return ("del", u, v)
        if op == "del":
            return ("add", u, v)
        return ("rev", v, u)

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return adjacency matrix learned from *data*."""
        n_samples = data.shape[0]
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        perm_pos = _compute_perm_pos(n_vars, permutation)
        allowed = _build_allowed_parents(n_vars, perm_pos, interaction_matrix)
        perm_constrained = perm_pos is not None

        scorer = self.scoring if sample_weights is None else self.scoring.with_weights(sample_weights)
        cache: dict = {}

        def local(var: int, parents: List[int]) -> float:
            key = (var, tuple(sorted(parents)))
            if key not in cache:
                cache[key] = scorer.local_score(var, list(parents), data, cardinality)
            return cache[key]

        def parents_of(var: int) -> List[int]:
            return list(np.where(adjacency[:, var] > 0)[0])

        def total_score() -> float:
            return sum(local(var, parents_of(var)) for var in range(n_vars))

        adjacency = np.zeros((n_vars, n_vars), dtype=int)
        tabu: List[tuple] = []
        # Track total score incrementally: BIC is decomposable, so the delta
        # returned by each operation is the exact change in total score.
        # This avoids calling total_score() (O(n × scoring)) inside the loop.
        current_score = total_score()   # computed once at initialisation
        best_score = current_score
        best_adj = adjacency.copy()
        patience = self.patience if self.patience is not None else max(self.tabu_length * 5, 50)
        no_improve = 0

        for _ in range(self.max_iter):
            best_nontabu_key = None
            best_nontabu_op: Optional[tuple] = None
            best_tabu_key = None
            best_tabu_op: Optional[tuple] = None

            for u in range(n_vars):
                for v in range(n_vars):
                    if u == v:
                        continue

                    # ADD u → v
                    if adjacency[u, v] == 0 and adjacency[v, u] == 0 and u in allowed[v]:
                        pa_v = parents_of(v)
                        if len(pa_v) < mp:
                            if perm_constrained or not _would_create_cycle(adjacency, u, v):
                                new_pa = pa_v + [u]
                                if not self.limit_table_size or _joint_table_size(cardinality, [v] + new_pa) <= n_samples:
                                    delta = local(v, new_pa) - local(v, pa_v)
                                    k = _op_key(delta, "add", u, v)
                                    if ("add", u, v) in tabu:
                                        if best_tabu_key is None or k > best_tabu_key:
                                            best_tabu_key, best_tabu_op = k, ("add", u, v)
                                    else:
                                        if best_nontabu_key is None or k > best_nontabu_key:
                                            best_nontabu_key, best_nontabu_op = k, ("add", u, v)

                    # DELETE u → v
                    if adjacency[u, v] == 1:
                        pa_v = parents_of(v)
                        new_pa = [p for p in pa_v if p != u]
                        delta = local(v, new_pa) - local(v, pa_v)
                        k = _op_key(delta, "del", u, v)
                        if ("del", u, v) in tabu:
                            if best_tabu_key is None or k > best_tabu_key:
                                best_tabu_key, best_tabu_op = k, ("del", u, v)
                        else:
                            if best_nontabu_key is None or k > best_nontabu_key:
                                best_nontabu_key, best_nontabu_op = k, ("del", u, v)

                    # REVERSE u → v  becomes  v → u
                    if adjacency[u, v] == 1 and v in allowed[u]:
                        pa_u = parents_of(u)
                        pa_v = parents_of(v)
                        if len(pa_u) < mp:
                            can_rev: bool
                            if perm_constrained:
                                can_rev = True
                            else:
                                adjacency[u, v] = 0
                                can_rev = not _would_create_cycle(adjacency, v, u)
                                adjacency[u, v] = 1
                            if can_rev:
                                new_pa_u = pa_u + [v]
                                new_pa_v = [p for p in pa_v if p != u]
                                if not self.limit_table_size or _joint_table_size(cardinality, [u] + new_pa_u) <= n_samples:
                                    delta = (
                                        local(v, new_pa_v) + local(u, new_pa_u)
                                        - local(v, pa_v) - local(u, pa_u)
                                    )
                                    k = _op_key(delta, "rev", u, v)
                                    if ("rev", u, v) in tabu:
                                        if best_tabu_key is None or k > best_tabu_key:
                                            best_tabu_key, best_tabu_op = k, ("rev", u, v)
                                    else:
                                        if best_nontabu_key is None or k > best_nontabu_key:
                                            best_nontabu_key, best_nontabu_op = k, ("rev", u, v)

            # ----------------------------------------------------------------
            # Move selection
            #
            # Unlike HC-Stable, Tabu accepts non-improving moves to escape
            # local optima; the tabu list prevents cycling back immediately.
            #
            # Aspiration criterion: a tabu move overrides its tabu status when
            # it is strictly better than the best non-tabu candidate AND the
            # resulting score would exceed the global best.  The comparison uses
            # current_score (tracked incrementally) — no extra scoring calls.
            # ----------------------------------------------------------------
            chosen_key = best_nontabu_key
            chosen_op = best_nontabu_op

            if best_tabu_op is not None:
                nontabu_delta = best_nontabu_key[0] if best_nontabu_key is not None else float('-inf')
                tabu_delta = best_tabu_key[0]
                if tabu_delta > nontabu_delta and current_score + tabu_delta > best_score:
                    chosen_key, chosen_op = best_tabu_key, best_tabu_op

            if chosen_op is None:
                break  # No moves available at all

            # Apply move (delta is the change in total BIC; no extra score call needed)
            chosen_delta = chosen_key[0]
            op, u, v = chosen_op
            if op == "add":
                adjacency[u, v] = 1
            elif op == "del":
                adjacency[u, v] = 0
            else:  # rev
                adjacency[u, v] = 0
                adjacency[v, u] = 1

            for key in [k for k in cache if k[0] in (u, v)]:
                cache.pop(key, None)

            current_score += chosen_delta

            tabu.append(self._reverse_op(op, u, v))
            if len(tabu) > self.tabu_length:
                tabu.pop(0)

            # Track global best; early-stop after 'patience' non-improving steps
            if current_score > best_score:
                best_score = current_score
                best_adj = adjacency.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_adj


# ---------------------------------------------------------------------------
# Grow-Shrink (Margaritis & Thrun 1999)
# ---------------------------------------------------------------------------


def _weighted_contingency(
    x: np.ndarray,
    y: np.ndarray,
    card_x: int,
    card_y: int,
    weights: np.ndarray,
) -> np.ndarray:
    """Build a weighted X-by-Y contingency table."""
    table = np.zeros((card_x, card_y), dtype=float)
    np.add.at(table, (x, y), weights)
    return table


def _chi_square_conditional_independence(
    data: np.ndarray,
    x: int,
    y: int,
    cond: List[int],
    cardinality: np.ndarray,
    alpha: float,
    weights: np.ndarray,
) -> bool:
    """Return True when X ⟂ Y | cond under a stratified chi-square test."""
    card_x = int(cardinality[x])
    card_y = int(cardinality[y])

    total_stat = 0.0
    total_dof = 0

    if len(cond) == 0:
        table = _weighted_contingency(data[:, x], data[:, y], card_x, card_y, weights)
        row_sum = table.sum(axis=1, keepdims=True)
        col_sum = table.sum(axis=0, keepdims=True)
        n = table.sum()
        if n <= 0:
            return True
        expected = (row_sum @ col_sum) / n
        valid = expected > 0
        if not np.any(valid):
            return True
        safe_exp = np.where(valid, expected, 1.0)  # avoid div-by-zero outside valid
        total_stat = float(np.sum(np.where(valid, (table - expected) ** 2 / safe_exp, 0.0)))
        total_dof = (card_x - 1) * (card_y - 1)
        if total_dof <= 0:
            return True
        p_value = float(chi2_dist.sf(total_stat, total_dof))
        return p_value > alpha

    mult = 1
    cond_idx = np.zeros(data.shape[0], dtype=int)
    for var in cond:
        cond_idx += data[:, var] * mult
        mult *= int(cardinality[var])

    for cfg in range(mult):
        mask = cond_idx == cfg
        if not np.any(mask):
            continue
        table = _weighted_contingency(
            data[mask, x],
            data[mask, y],
            card_x,
            card_y,
            weights[mask],
        )
        n = table.sum()
        if n <= 0:
            continue
        row_sum = table.sum(axis=1, keepdims=True)
        col_sum = table.sum(axis=0, keepdims=True)
        expected = (row_sum @ col_sum) / n
        valid = expected > 0
        active_rows = int(np.sum(row_sum[:, 0] > 0))
        active_cols = int(np.sum(col_sum[0, :] > 0))
        dof = (active_rows - 1) * (active_cols - 1)
        if dof <= 0 or not np.any(valid):
            continue
        safe_exp = np.where(valid, expected, 1.0)
        total_stat += float(np.sum(np.where(valid, (table - expected) ** 2 / safe_exp, 0.0)))
        total_dof += dof

    if total_dof <= 0:
        return True
    p_value = float(chi2_dist.sf(total_stat, total_dof))
    return p_value > alpha


class GrowShrinkLearner:
    """Learn a BN structure via Grow-Shrink Markov-blanket induction.

    Parameters
    ----------
    alpha_ci : float
        Chi-square significance level.
    max_parents : int or None
    max_conditioning_set_size : int or None
        Cap the size of the conditioning set used in CI tests.  Limits the
        depth of the shrink phase and prevents the quadratic blow-up that
        occurs on dense or high-cardinality networks.  ``None`` → full
        blanket conditioning (original GS; can be slow on large networks).
    """

    def __init__(
        self,
        alpha_ci: float = 0.05,
        max_parents: Optional[int] = None,
        max_conditioning_set_size: Optional[int] = None,
    ) -> None:
        self.alpha_ci = alpha_ci
        self.max_parents = max_parents
        self.max_conditioning_set_size = max_conditioning_set_size

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=int)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        if sample_weights is None:
            weights = np.ones(data.shape[0], dtype=float)
        else:
            weights = np.asarray(sample_weights, dtype=float)
            total_w = float(np.sum(weights))
            if total_w > 0:
                weights = weights * (len(weights) / total_w)

        max_cond = self.max_conditioning_set_size
        markov_blankets: List[Set[int]] = [set() for _ in range(n_vars)]

        for target in range(n_vars):
            blanket: Set[int] = set()
            changed = True
            while changed:
                changed = False
                for var in range(n_vars):
                    if var == target or var in blanket:
                        continue
                    if interaction_matrix is not None and interaction_matrix[var, target] == 0:
                        continue
                    cond = sorted(blanket)
                    if max_cond is not None:
                        cond = cond[:max_cond]
                    independent = _chi_square_conditional_independence(
                        data, target, var, cond, cardinality, self.alpha_ci, weights
                    )
                    if not independent:
                        blanket.add(var)
                        changed = True

            for var in sorted(blanket):
                cond = sorted(list(blanket - {var}))
                if max_cond is not None:
                    cond = cond[:max_cond]
                independent = _chi_square_conditional_independence(
                    data, target, var, cond, cardinality, self.alpha_ci, weights
                )
                if independent:
                    blanket.remove(var)

            markov_blankets[target] = blanket

        skeleton = np.zeros((n_vars, n_vars), dtype=int)
        for u in range(n_vars):
            for v in range(u + 1, n_vars):
                if v in markov_blankets[u] and u in markov_blankets[v]:
                    skeleton[u, v] = 1
                    skeleton[v, u] = 1

        if permutation is None:
            order = np.arange(n_vars, dtype=int)
        else:
            order = np.asarray(permutation, dtype=int)
        pos = np.empty(n_vars, dtype=int)
        for idx, var in enumerate(order):
            pos[int(var)] = idx

        adjacency = np.zeros((n_vars, n_vars), dtype=int)

        for u in range(n_vars):
            for v in range(u + 1, n_vars):
                if skeleton[u, v] == 0:
                    continue
                if interaction_matrix is not None and interaction_matrix[u, v] == 0:
                    continue

                if pos[u] < pos[v]:
                    parent, child = u, v
                else:
                    parent, child = v, u

                if np.sum(adjacency[:, child]) >= mp:
                    continue
                adjacency[parent, child] = 1

        return adjacency


class RecursiveCDLearner:
    """Learn a BN structure via recursive causal discovery (RCD-style)."""

    def __init__(
        self,
        alpha_ci: float = 0.05,
        max_parents: Optional[int] = None,
        max_conditioning_set: int = 2,
    ) -> None:
        self.alpha_ci = alpha_ci
        self.max_parents = max_parents
        self.max_conditioning_set = max(0, int(max_conditioning_set))

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=int)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        if sample_weights is None:
            weights = np.ones(data.shape[0], dtype=float)
        else:
            weights = np.asarray(sample_weights, dtype=float)
            total_w = float(np.sum(weights))
            if total_w > 0:
                weights = weights * (len(weights) / total_w)

        skeleton = np.zeros((n_vars, n_vars), dtype=int)

        def _is_independent(x: int, y: int, cond: List[int]) -> bool:
            return _chi_square_conditional_independence(
                data, x, y, cond, cardinality, self.alpha_ci, weights
            )

        def _connected_components(nodes: List[int]) -> List[List[int]]:
            node_set = set(nodes)
            seen: Set[int] = set()
            comps: List[List[int]] = []

            for start in sorted(nodes):
                if start in seen:
                    continue
                stack = [start]
                comp: List[int] = []
                while stack:
                    v = stack.pop()
                    if v in seen:
                        continue
                    seen.add(v)
                    comp.append(v)
                    neigh = np.where(skeleton[v] > 0)[0]
                    for nxt in neigh:
                        n_i = int(nxt)
                        if n_i in node_set and n_i not in seen:
                            stack.append(n_i)
                comps.append(sorted(comp))
            return comps

        def _discover(nodes: List[int]) -> None:
            if len(nodes) <= 1:
                return

            for i in range(len(nodes)):
                x = int(nodes[i])
                for j in range(i + 1, len(nodes)):
                    y = int(nodes[j])
                    if interaction_matrix is not None and interaction_matrix[x, y] == 0:
                        continue

                    candidates = [z for z in nodes if z != x and z != y]
                    max_k = min(self.max_conditioning_set, len(candidates))
                    independent = False
                    for k in range(max_k + 1):
                        for cond in combinations(candidates, k):
                            if _is_independent(x, y, list(cond)):
                                independent = True
                                break
                        if independent:
                            break

                    if not independent:
                        skeleton[x, y] = 1
                        skeleton[y, x] = 1

            comps = _connected_components(nodes)
            if len(comps) > 1:
                for comp in comps:
                    _discover(comp)

        _discover(list(range(n_vars)))

        if permutation is None:
            order = np.arange(n_vars, dtype=int)
        else:
            order = np.asarray(permutation, dtype=int)
        pos = np.empty(n_vars, dtype=int)
        for idx, var in enumerate(order):
            pos[int(var)] = idx

        adjacency = np.zeros((n_vars, n_vars), dtype=int)
        for u in range(n_vars):
            for v in range(u + 1, n_vars):
                if skeleton[u, v] == 0:
                    continue
                if pos[u] < pos[v]:
                    parent, child = u, v
                else:
                    parent, child = v, u
                if np.sum(adjacency[:, child]) >= mp:
                    continue
                adjacency[parent, child] = 1

        return adjacency


class RPCDLearner:
    """Learn a BN structure via recursive parallel causal discovery (RPCD-style)."""

    def __init__(
        self,
        alpha_ci: float = 0.05,
        max_parents: Optional[int] = None,
        max_conditioning_set: int = 2,
        max_workers: Optional[int] = None,
        min_parallel_pairs: int = 8,
        ci_test_timeout: float = 60.0,
    ) -> None:
        if max_conditioning_set < 0:
            raise ValueError("max_conditioning_set must be >= 0")
        self.alpha_ci = alpha_ci
        self.max_parents = max_parents
        self.max_conditioning_set = int(max_conditioning_set)
        self.max_workers = max_workers
        # Keep threshold at >=1 so sequential fallback is still available.
        self.min_parallel_pairs = max(1, int(min_parallel_pairs))
        self.ci_test_timeout = float(ci_test_timeout)

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=int)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        if sample_weights is None:
            weights = np.ones(data.shape[0], dtype=float)
        else:
            weights = np.asarray(sample_weights, dtype=float)
            total_w = float(np.sum(weights))
            if total_w > 0:
                weights = weights * (len(weights) / total_w)

        skeleton = np.zeros((n_vars, n_vars), dtype=int)

        def _is_dependent(x: int, y: int, nodes: List[int]) -> bool:
            candidates = [z for z in nodes if z != x and z != y]
            max_k = min(self.max_conditioning_set, len(candidates))
            for k in range(max_k + 1):
                for cond in combinations(candidates, k):
                    independent = _chi_square_conditional_independence(
                        data, x, y, list(cond), cardinality, self.alpha_ci, weights
                    )
                    if independent:
                        return False
            return True

        def _connected_components(nodes: List[int]) -> List[List[int]]:
            node_set = set(nodes)
            seen: Set[int] = set()
            comps: List[List[int]] = []
            for start in sorted(nodes):
                if start in seen:
                    continue
                stack = [start]
                comp: List[int] = []
                while stack:
                    v = stack.pop()
                    if v in seen:
                        continue
                    seen.add(v)
                    comp.append(v)
                    for nxt in np.where(skeleton[v] > 0)[0]:
                        n_i = int(nxt)
                        if n_i in node_set and n_i not in seen:
                            stack.append(n_i)
                comps.append(sorted(comp))
            return comps

        def _evaluate_pairs(nodes: List[int]) -> List[Tuple[int, int, bool]]:
            pairs: List[Tuple[int, int]] = []
            for i in range(len(nodes)):
                x = int(nodes[i])
                for j in range(i + 1, len(nodes)):
                    y = int(nodes[j])
                    if interaction_matrix is not None and interaction_matrix[x, y] == 0:
                        continue
                    pairs.append((x, y))

            if len(pairs) < self.min_parallel_pairs:
                return [(x, y, _is_dependent(x, y, nodes)) for x, y in pairs]

            max_workers = self.max_workers
            if max_workers is None:
                max_workers = min(32, max(1, len(pairs)))
            max_workers = max(1, int(max_workers))
            if max_workers == 1:
                return [(x, y, _is_dependent(x, y, nodes)) for x, y in pairs]

            results: List[Tuple[int, int, bool]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_is_dependent, x, y, nodes): (x, y) for x, y in pairs}
                for future in as_completed(futures):
                    x, y = futures[future]
                    try:
                        dependent = bool(future.result(timeout=self.ci_test_timeout))
                    except TimeoutError:
                        dependent = False
                    results.append((x, y, dependent))
            return results

        def _discover(nodes: List[int]) -> None:
            if len(nodes) <= 1:
                return

            for x, y, dependent in _evaluate_pairs(nodes):
                if dependent:
                    skeleton[x, y] = 1
                    skeleton[y, x] = 1

            comps = _connected_components(nodes)
            if len(comps) > 1:
                for comp in comps:
                    _discover(comp)

        _discover(list(range(n_vars)))

        if permutation is None:
            order = np.arange(n_vars, dtype=int)
        else:
            order = np.asarray(permutation, dtype=int)
        pos = np.empty(n_vars, dtype=int)
        for idx, var in enumerate(order):
            pos[int(var)] = idx

        adjacency = np.zeros((n_vars, n_vars), dtype=int)
        for u in range(n_vars):
            for v in range(u + 1, n_vars):
                if skeleton[u, v] == 0:
                    continue
                if pos[u] < pos[v]:
                    parent, child = u, v
                else:
                    parent, child = v, u
                if np.sum(adjacency[:, child]) >= mp:
                    continue
                adjacency[parent, child] = 1

        return adjacency


# ---------------------------------------------------------------------------
# PC algorithm shared orientation helpers
# ---------------------------------------------------------------------------


def _orient_v_structures(
    n_vars: int,
    skeleton: np.ndarray,
    sep_sets: Dict[Tuple[int, int], List[int]],
) -> np.ndarray:
    """Orient colliders (v-structures): a -b- c with b not in sep(a,c) -> a->b<-c."""
    directed = np.zeros((n_vars, n_vars), dtype=int)
    for b in range(n_vars):
        neighbors = [v for v in range(n_vars) if skeleton[b, v] == 1]
        for ai in range(len(neighbors)):
            a = neighbors[ai]
            for c in neighbors[ai + 1:]:
                if skeleton[a, c] == 1:
                    continue  # shielded triple - skip
                sep_ac = sep_sets.get((a, c), sep_sets.get((c, a)))
                if sep_ac is not None and b not in sep_ac:
                    directed[a, b] = 1
                    directed[c, b] = 1
    return directed


def _apply_meek_rules(
    n_vars: int,
    skeleton: np.ndarray,
    directed: np.ndarray,
) -> np.ndarray:
    """Propagate orientations with Meek's core rules R1, R2, R3."""
    changed = True
    while changed:
        changed = False
        for u in range(n_vars):
            for v in range(n_vars):
                if skeleton[u, v] == 0 or directed[u, v] == 1 or directed[v, u] == 1:
                    continue
                # R1: b->u and b not adj v  ->  u->v
                for b in range(n_vars):
                    if directed[b, u] == 1 and skeleton[b, v] == 0:
                        directed[u, v] = 1; changed = True; break
                if directed[u, v] == 1:
                    continue
                # R2: u->c->v  ->  u->v
                for c in range(n_vars):
                    if directed[u, c] == 1 and directed[c, v] == 1:
                        directed[u, v] = 1; changed = True; break
                if directed[u, v] == 1:
                    continue
                # R3: c->v, d->v, u-c, u-d, c not adj d  ->  u->v
                parents_v = [p for p in range(n_vars) if directed[p, v] == 1]
                for i_c, c in enumerate(parents_v):
                    if skeleton[u, c] == 0:
                        continue
                    for d in parents_v[i_c + 1:]:
                        if skeleton[u, d] == 0:
                            continue
                        if skeleton[c, d] == 0:
                            directed[u, v] = 1; changed = True; break
                    if directed[u, v] == 1:
                        break
    return directed


def _skeleton_to_adjacency(
    n_vars: int,
    skeleton: np.ndarray,
    directed: np.ndarray,
    permutation: Optional[np.ndarray],
    mp: int,
) -> np.ndarray:
    """Directed edges keep orientation; remaining use permutation order."""
    if permutation is None:
        order = np.arange(n_vars, dtype=int)
    else:
        order = np.asarray(permutation, dtype=int)
    pos = np.empty(n_vars, dtype=int)
    for idx, var in enumerate(order):
        pos[int(var)] = idx

    adjacency = np.zeros((n_vars, n_vars), dtype=int)
    for u in range(n_vars):
        for v in range(u + 1, n_vars):
            if skeleton[u, v] == 0:
                continue
            if directed[u, v] == 1:
                parent, child = u, v
            elif directed[v, u] == 1:
                parent, child = v, u
            else:
                parent, child = (u, v) if pos[u] < pos[v] else (v, u)
            if np.sum(adjacency[:, child]) < mp:
                adjacency[parent, child] = 1
    return adjacency


# ---------------------------------------------------------------------------
# PC algorithm  (Spirtes & Glymour 1991)
# ---------------------------------------------------------------------------


class PCLearner:
    """Learn BN structure using the PC algorithm.

    Phase 1 -- Skeleton: for each adjacent pair (X,Y) test conditioning subsets
    of adj(X) of increasing size d; remove edge on first separating set found.
    Phase 2 -- V-structures: orient colliders from the recorded separating sets.
    Phase 3 -- Meek rules R1-R3: propagate orientations to avoid cycles and
    new v-structures.

    Parameters
    ----------
    alpha_ci : float
        Significance level for the chi-square CI test.
    max_cond_set_size : int or None
        Cap on conditioning set size.  None -> unbounded.
    max_parents : int or None

    References
    ----------
    Spirtes & Glymour (1991). "An Algorithm for Fast Recovery of Sparse
    Causal Graphs." Social Science Computer Review 9(1).
    """

    def __init__(
        self,
        alpha_ci: float = 0.05,
        max_cond_set_size: Optional[int] = None,
        max_parents: Optional[int] = None,
    ) -> None:
        self.alpha_ci = alpha_ci
        self.max_cond_set_size = max_cond_set_size
        self.max_parents = max_parents

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=int)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)
        weights = self._make_weights(data.shape[0], sample_weights)
        skeleton, sep_sets = self._find_skeleton(data, n_vars, cardinality, weights, interaction_matrix)
        directed = _orient_v_structures(n_vars, skeleton, sep_sets)
        directed = _apply_meek_rules(n_vars, skeleton, directed)
        return _skeleton_to_adjacency(n_vars, skeleton, directed, permutation, mp)

    def _make_weights(self, n: int, sw: Optional[np.ndarray]) -> np.ndarray:
        if sw is None:
            return np.ones(n, dtype=float)
        w = np.asarray(sw, dtype=float)
        total = float(w.sum())
        return w * (n / total) if total > 0 else np.ones(n, dtype=float)

    def _ci_test(self, data, x, y, cond, cardinality, weights):
        return _chi_square_conditional_independence(
            data, x, y, cond, cardinality, self.alpha_ci, weights
        )

    def _find_skeleton(self, data, n_vars, cardinality, weights, interaction_matrix):
        adj: Dict[int, Set[int]] = {v: set(range(n_vars)) - {v} for v in range(n_vars)}
        if interaction_matrix is not None:
            for v in range(n_vars):
                adj[v] = {u for u in adj[v] if interaction_matrix[u, v] != 0}

        sep_sets: Dict[Tuple[int, int], List[int]] = {}
        max_d = self.max_cond_set_size
        d = 0
        while True:
            removed_any = False
            for x in range(n_vars):
                for y in sorted(adj[x]):
                    if y <= x or (x, y) in sep_sets:
                        continue
                    candidates = sorted(adj[x] - {y})
                    if len(candidates) < d:
                        continue
                    if max_d is not None and d > max_d:
                        continue
                    for cond in combinations(candidates, d):
                        if self._ci_test(data, x, y, list(cond), cardinality, weights):
                            adj[x].discard(y); adj[y].discard(x)
                            sep_sets[(x, y)] = sep_sets[(y, x)] = list(cond)
                            removed_any = True
                            break
            d += 1
            if not removed_any or (max_d is not None and d > max_d):
                break

        skeleton = np.zeros((n_vars, n_vars), dtype=int)
        for x in range(n_vars):
            for y in adj[x]:
                skeleton[x, y] = 1
        return skeleton, sep_sets


# ---------------------------------------------------------------------------
# Stable-PC  (Colombo & Maathuis 2014)
# ---------------------------------------------------------------------------


class StablePCLearner(PCLearner):
    """Order-independent PC algorithm.

    At each level d, all CI tests are evaluated before any edge is removed,
    making the skeleton independent of variable ordering.

    References
    ----------
    Colombo & Maathuis (2014). "Order-Independent Constraint-Based Causal
    Structure Learning." JMLR 15.
    """

    def _find_skeleton(self, data, n_vars, cardinality, weights, interaction_matrix):
        adj: Dict[int, Set[int]] = {v: set(range(n_vars)) - {v} for v in range(n_vars)}
        if interaction_matrix is not None:
            for v in range(n_vars):
                adj[v] = {u for u in adj[v] if interaction_matrix[u, v] != 0}

        sep_sets: Dict[Tuple[int, int], List[int]] = {}
        max_d = self.max_cond_set_size
        d = 0
        while True:
            to_remove: List[Tuple[int, int, List[int]]] = []
            for x in range(n_vars):
                for y in sorted(adj[x]):
                    if y <= x or (x, y) in sep_sets:
                        continue
                    candidates = sorted(adj[x] - {y})
                    if len(candidates) < d:
                        continue
                    if max_d is not None and d > max_d:
                        continue
                    for cond in combinations(candidates, d):
                        if self._ci_test(data, x, y, list(cond), cardinality, weights):
                            to_remove.append((x, y, list(cond)))
                            break

            if not to_remove:
                break
            for x, y, cond in to_remove:
                adj[x].discard(y); adj[y].discard(x)
                if (x, y) not in sep_sets:
                    sep_sets[(x, y)] = sep_sets[(y, x)] = cond
            d += 1
            if max_d is not None and d > max_d:
                break

        skeleton = np.zeros((n_vars, n_vars), dtype=int)
        for x in range(n_vars):
            for y in adj[x]:
                skeleton[x, y] = 1
        return skeleton, sep_sets


# ---------------------------------------------------------------------------
# Decision-tree HC learner  (Friedman & Goldszmidt 1996)
# ---------------------------------------------------------------------------


class DecisionTreeLearner:
    """HC structure search with decision-tree MDL scoring for local CPD structure.

    Uses DecisionTreeMDLScorer inside StableHillClimbLearner.  Each CPD is
    compressed via a CART-BIC tree: splits are accepted only when the BIC
    gain is positive.  Effective parameters = n_leaves*(k-1) <= tabular count.

    References
    ----------
    Friedman & Goldszmidt (1996). "Learning Bayesian Networks with Local
    Structure." UAI-96.
    """

    def __init__(
        self,
        max_parents: Optional[int] = None,
        max_iter: int = 500,
        limit_table_size: bool = True,
        alpha: float = 1.0,
        max_tree_depth: Optional[int] = None,
    ) -> None:
        self.max_parents = max_parents
        self.max_iter = max_iter
        self.limit_table_size = limit_table_size
        self.alpha = alpha
        self.max_tree_depth = max_tree_depth

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        from bayes_nets.scoring import DecisionTreeMDLScorer

        scoring = DecisionTreeMDLScorer(
            alpha=self.alpha,
            sample_weights=sample_weights,
            max_tree_depth=self.max_tree_depth,
        )
        return StableHillClimbLearner(
            scoring=scoring,
            max_parents=self.max_parents,
            max_iter=self.max_iter,
            limit_table_size=self.limit_table_size,
        ).learn(data, n_vars, cardinality,
                permutation=permutation, interaction_matrix=interaction_matrix)


# ---------------------------------------------------------------------------
# Scorer factory (shared by the exact / decomposition learners below)
# ---------------------------------------------------------------------------


def _make_scorer(
    score: str,
    alpha: float,
    sample_weights: Optional[np.ndarray] = None,
) -> ScoringMethod:
    """Return a decomposable :class:`ScoringMethod` given a short name."""
    from bayes_nets.scoring import (
        BICScoringMethod,
        AICScoringMethod,
        K2ScoringMethod,
    )

    name = score.lower()
    if name == "bic":
        return BICScoringMethod(alpha=alpha, sample_weights=sample_weights)
    if name == "aic":
        return AICScoringMethod(alpha=alpha, sample_weights=sample_weights)
    if name in ("k2", "bde", "bdeu"):
        a = alpha if alpha > 0 else 1.0
        return K2ScoringMethod(alpha=a, sample_weights=sample_weights)
    raise ValueError(f"Unknown score '{score}'. Use 'bic', 'aic', or 'k2'.")


# ---------------------------------------------------------------------------
# Decision-graph HC learner  (Chickering, Heckerman & Meek 1997)
# ---------------------------------------------------------------------------


class DecisionGraphLearner:
    """HC structure search with decision-graph Bayesian scoring.

    Uses DecisionGraphBayesianScorer inside StableHillClimbLearner.  After
    growing a greedy K2 tree, pairs of leaves whose data is pooled by K2 are
    merged (parameter sharing), the defining property of decision graphs.

    References
    ----------
    Chickering, Heckerman & Meek (1997). "A Bayesian Approach to Learning
    Bayesian Networks with Local Structure." UAI-97.
    """

    def __init__(
        self,
        max_parents: Optional[int] = None,
        max_iter: int = 500,
        limit_table_size: bool = True,
        alpha: float = 1.0,
        max_tree_depth: Optional[int] = None,
    ) -> None:
        self.max_parents = max_parents
        self.max_iter = max_iter
        self.limit_table_size = limit_table_size
        self.alpha = alpha
        self.max_tree_depth = max_tree_depth

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        from bayes_nets.scoring import DecisionGraphBayesianScorer

        scoring = DecisionGraphBayesianScorer(
            alpha=self.alpha,
            sample_weights=sample_weights,
            max_tree_depth=self.max_tree_depth,
        )
        return StableHillClimbLearner(
            scoring=scoring,
            max_parents=self.max_parents,
            max_iter=self.max_iter,
            limit_table_size=self.limit_table_size,
        ).learn(data, n_vars, cardinality,
                permutation=permutation, interaction_matrix=interaction_matrix)


# ---------------------------------------------------------------------------
# Memory-efficient level-wise exact DP  (Huang & Suzuki 2026)
# ---------------------------------------------------------------------------


class LevelWiseDPLearner:
    """Exact BN structure learning via a level-wise dynamic program.

    Finds the *globally optimal* DAG for a decomposable score by dynamic
    programming over the subset lattice.  The traversal is organised
    level-by-level (by subset cardinality) so that computing level ``k``
    only requires the results from level ``k-1``; parent-set optimisation
    and sink-node identification are fused into a single pass.  This is the
    memory-efficient reformulation of the Silander & Myllymäki (2012) DP
    proposed by Huang & Suzuki (2026); it returns the same optimum but with
    a reduced peak-memory footprint (``O(sqrt(p) 2^p)`` vs ``O(p 2^p)``).

    The algorithm is exact and therefore exponential in the number of
    variables; it is intended as a "gold-standard" learner for small
    problems (``n_vars`` up to ~20).

    Parameters
    ----------
    score : str
        Decomposable score: ``"bic"`` (default), ``"aic"`` or ``"k2"``.
    alpha : float
        Prior / smoothing parameter passed to the score.
    max_parents : int or None
        Maximum parents per variable.  ``None`` -> rule of thumb.  Bounding
        the in-degree dramatically reduces the parent-set search.
    limit_table_size : bool
        Skip parent sets whose joint table would exceed the sample count.
    max_vars : int
        Hard guard: raise ``ValueError`` when ``n_vars`` exceeds this
        (default 20) to avoid accidentally launching an intractable run.

    References
    ----------
    Huang & Suzuki (2026). "Memory-efficient exact Bayesian network
    structure learning: a single-pass level-wise dynamic program."
    Behaviormetrika.
    """

    def __init__(
        self,
        score: str = "bic",
        alpha: float = 1.0,
        max_parents: Optional[int] = None,
        limit_table_size: bool = True,
        max_vars: int = 20,
    ) -> None:
        self.score = score
        self.alpha = alpha
        self.max_parents = max_parents
        self.limit_table_size = limit_table_size
        self.max_vars = max_vars

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the globally optimal adjacency matrix for the score.

        ``permutation`` is ignored (the exact DP searches over all orders).
        ``interaction_matrix`` restricts the candidate parents when given.
        """
        if n_vars > self.max_vars:
            raise ValueError(
                f"LevelWiseDPLearner is exact/exponential; n_vars={n_vars} "
                f"exceeds max_vars={self.max_vars}. Raise max_vars to force it."
            )

        data = np.asarray(data)
        n_samples = data.shape[0]
        cardinality = np.asarray(cardinality, dtype=int)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)
        scorer = _make_scorer(self.score, self.alpha, sample_weights)

        # Allowed parents per variable (interaction constraint only; the DP
        # explores every ordering itself).
        allowed_mask = np.ones((n_vars, n_vars), dtype=bool)
        if interaction_matrix is not None:
            allowed_mask = np.asarray(interaction_matrix) != 0
        np.fill_diagonal(allowed_mask, False)

        full = (1 << n_vars) - 1

        # --- Step 1+2: best parent set of each var, drawn from every subset -
        # best_ps_score[v][C] = best local score of v using parents P subset of C
        # best_ps_set[v][C]   = the argmax parent set P (as a bitmask)
        # Computed level-wise over C by cardinality.
        local_cache: Dict[Tuple[int, int], float] = {}

        def local(v: int, pa_mask: int) -> float:
            key = (v, pa_mask)
            hit = local_cache.get(key)
            if hit is not None:
                return hit
            parents = [u for u in range(n_vars) if (pa_mask >> u) & 1]
            if len(parents) > mp or any(not allowed_mask[u, v] for u in parents):
                val = -np.inf
            elif self.limit_table_size and _joint_table_size(cardinality, [v] + parents) > n_samples:
                val = -np.inf
            else:
                val = scorer.local_score(v, parents, data, cardinality)
            local_cache[key] = val
            return val

        # best_ps over candidate sets C (subsets of V \ {v}).
        best_ps_score = [dict() for _ in range(n_vars)]
        best_ps_set = [dict() for _ in range(n_vars)]
        for v in range(n_vars):
            cand_all = full & ~(1 << v)
            bs = best_ps_score[v]
            bp = best_ps_set[v]
            bs[0] = local(v, 0)
            bp[0] = 0
            # Enumerate subsets C of cand_all in increasing cardinality.
            for C in _subsets_by_level(cand_all, n_vars):
                if C == 0:
                    continue
                best_val = local(v, C)      # use *all* of C as parents
                best_set = C
                # or drop one element and reuse the smaller optimum
                cc = C
                while cc:
                    low = cc & (-cc)
                    Cwithout = C & ~low
                    cand_val = bs.get(Cwithout, -np.inf)
                    if cand_val > best_val:
                        best_val = cand_val
                        best_set = bp.get(Cwithout, 0)
                    cc ^= low
                bs[C] = best_val
                bp[C] = best_set

        # --- Step 3+4: sink-node DP over all subsets S of V ------------------
        # g[S]    = best total score achievable over the sub-DAG on nodes S
        # sink[S] = the sink variable achieving g[S]
        g = {0: 0.0}
        sink: Dict[int, int] = {}
        for S in _subsets_by_level(full, n_vars):
            if S == 0:
                continue
            best_total = -np.inf
            best_sink = -1
            s = S
            while s:
                low = s & (-s)
                v = low.bit_length() - 1
                rest = S & ~low
                val = g[rest] + best_ps_score[v].get(rest, -np.inf)
                if val > best_total:
                    best_total = val
                    best_sink = v
                s ^= low
            g[S] = best_total
            sink[S] = best_sink

        # --- Reconstruct optimal DAG by peeling sinks -----------------------
        adjacency = np.zeros((n_vars, n_vars), dtype=int)
        S = full
        while S:
            v = sink[S]
            rest = S & ~(1 << v)
            pa_mask = best_ps_set[v].get(rest, 0)
            for u in range(n_vars):
                if (pa_mask >> u) & 1:
                    adjacency[u, v] = 1
            S = rest

        return adjacency


def _subsets_by_level(mask: int, n_vars: int) -> List[int]:
    """Return all sub-masks of *mask* ordered by increasing popcount.

    Enumerating subsets by cardinality realises the level-wise traversal:
    every subset of size ``k`` precedes those of size ``k+1``.
    """
    members = [i for i in range(n_vars) if (mask >> i) & 1]
    result: List[int] = [0]
    from itertools import combinations as _combos
    for k in range(1, len(members) + 1):
        for combo in _combos(members, k):
            sub = 0
            for i in combo:
                sub |= (1 << i)
            result.append(sub)
    return result


# ---------------------------------------------------------------------------
# SARTRE pruning  (Kanamori et al. 2026)
# ---------------------------------------------------------------------------


def _group_lasso_bcd(
    phi: np.ndarray,
    y: np.ndarray,
    groups: List[np.ndarray],
    lam: float,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> np.ndarray:
    """Solve group-lasso  ½‖y-Φβ‖² + λ Σ_g ‖β_g‖₂  by block-coord descent.

    ``groups`` is a list of index arrays partitioning the columns of ``phi``.
    Returns the coefficient vector β.  Each block update solves the group
    sub-problem in closed form using the block's Gram matrix.
    """
    n, p = phi.shape
    beta = np.zeros(p, dtype=float)
    residual = y - phi @ beta
    grams = [phi[:, g].T @ phi[:, g] for g in groups]
    # Ridge-stabilised inverse for the "keep" branch.
    inv = []
    for G, gidx in zip(grams, groups):
        d = G.shape[0]
        inv.append(np.linalg.pinv(G + 1e-8 * np.eye(d)))

    for _ in range(max_iter):
        max_change = 0.0
        for gi, g in enumerate(groups):
            phg = phi[:, g]
            # partial residual excluding this group
            residual += phg @ beta[g]
            z = phg.T @ residual
            new_beta_g = np.zeros_like(beta[g])
            if np.linalg.norm(z) > lam:
                # Solve (G + λ/‖β_g‖ I) β_g = z  via a few fixed-point steps.
                bg = inv[gi] @ z
                for _inner in range(25):
                    nrm = np.linalg.norm(bg)
                    if nrm < 1e-12:
                        break
                    Gr = grams[gi] + (lam / nrm) * np.eye(len(g))
                    bg_new = np.linalg.solve(Gr, z)
                    if np.linalg.norm(bg_new - bg) < 1e-9:
                        bg = bg_new
                        break
                    bg = bg_new
                new_beta_g = bg
            change = np.linalg.norm(new_beta_g - beta[g])
            max_change = max(max_change, change)
            beta[g] = new_beta_g
            residual -= phg @ beta[g]
        if max_change < tol:
            break
    return beta


class SARTREPruner:
    """SARTRE: order-based edge pruning by group-sparse regression.

    Given a topological order (``permutation``), SARTRE builds the
    fully-connected DAG induced by that order and then prunes spurious
    candidate parents of each variable.  Each candidate parent contributes a
    *group* of one-hot indicator features (the discrete analogue of the
    randomized-tree-embedding intervals used for continuous data by Kanamori
    et al.).  A group-lasso regression of the child on all candidate-parent
    feature groups drives whole groups to zero; a parent whose group vanishes
    is pruned.  This avoids the repeated hypothesis testing of CAM-pruning.

    The pruner needs an ordering.  Pass one via ``permutation``; if none is
    given it falls back to the natural order ``[0, 1, ..., n_vars-1]``.  It is
    designed to be chained after an order-estimating learner such as
    :class:`K2StructureLearner` or :class:`PCLearner`.

    Parameters
    ----------
    lam : float
        Group-lasso regularisation strength (per-sample scaled).  Larger
        values prune more aggressively.
    max_parents : int or None
        Cap on retained parents per variable (rule of thumb if ``None``).
    tol : float
        A parent group with L2 norm below ``tol`` is pruned.

    References
    ----------
    Kanamori, Takagi, Kobayashi (2026). "Sparse Additive Model Pruning for
    Order-Based Causal Structure Learning."
    """

    def __init__(
        self,
        lam: float = 0.05,
        max_parents: Optional[int] = None,
        tol: float = 1e-4,
    ) -> None:
        self.lam = lam
        self.max_parents = max_parents
        self.tol = tol

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=int)
        n_samples = data.shape[0]
        cardinality = np.asarray(cardinality, dtype=int)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        order = (np.arange(n_vars, dtype=int) if permutation is None
                 else np.asarray(permutation, dtype=int))
        perm_pos = _compute_perm_pos(n_vars, order)

        # Per-row effective weights.  ``sample_weights`` is a probability
        # vector over rows (sums to 1); scaling by N gives effective counts
        # summing to N, so the group-lasso penalty (lam * N) stays calibrated.
        # A weighted least-squares fit is obtained by centring on the weighted
        # mean and scaling every row of phi and y by sqrt(effective weight);
        # then ||sqrt(w)·(y - phi·b)||^2 == sum_i w_i (y_i - phi_i·b)^2.
        if sample_weights is None:
            eff_w = np.ones(n_samples, dtype=float)
        else:
            eff_w = np.asarray(sample_weights, dtype=float) * n_samples
        w_total = eff_w.sum()
        sw_sqrt = np.sqrt(eff_w)

        def _wmean(col: np.ndarray) -> float:
            return float((eff_w * col).sum() / w_total)

        # One-hot encoding of each variable (drop-first to avoid collinearity),
        # weighted-centred and row-scaled so all regressions share the same
        # whitening.
        encoders: List[np.ndarray] = []
        for v in range(n_vars):
            k = int(cardinality[v])
            if k <= 2:
                cols = data[:, v].reshape(-1, 1).astype(float)
            else:
                cols = np.zeros((n_samples, k - 1), dtype=float)
                for s in range(1, k):
                    cols[:, s - 1] = (data[:, v] == s).astype(float)
            for j in range(cols.shape[1]):
                cols[:, j] -= _wmean(cols[:, j])
            cols *= sw_sqrt[:, None]
            encoders.append(cols)

        adjacency = np.zeros((n_vars, n_vars), dtype=int)
        lam_eff = self.lam * n_samples

        for v in range(n_vars):
            candidates = [
                u for u in range(n_vars)
                if perm_pos[u] < perm_pos[v]
                and (interaction_matrix is None or interaction_matrix[u, v] != 0)
            ]
            if not candidates:
                continue

            blocks = [encoders[u] for u in candidates]
            phi = np.hstack(blocks)
            groups, start = [], 0
            for b in blocks:
                groups.append(np.arange(start, start + b.shape[1]))
                start += b.shape[1]

            y_raw = data[:, v].astype(float)
            y = sw_sqrt * (y_raw - _wmean(y_raw))

            beta = _group_lasso_bcd(phi, y, groups, lam_eff)

            # rank surviving parents by group norm, keep up to mp
            norms = [(np.linalg.norm(beta[g]), u) for g, u in zip(groups, candidates)]
            survivors = [(nrm, u) for nrm, u in norms if nrm > self.tol]
            survivors.sort(reverse=True)
            for nrm, u in survivors[:mp]:
                adjacency[u, v] = 1

        return adjacency


# ---------------------------------------------------------------------------
# iter-DSLA: iterative structure decomposition learning  (Jia & Li 2026)
# ---------------------------------------------------------------------------


def _to_skeleton(adjacency: np.ndarray) -> np.ndarray:
    """Undirected skeleton of a DAG (symmetric 0/1 matrix, no self-loops)."""
    skel = ((adjacency != 0) | (adjacency.T != 0)).astype(int)
    np.fill_diagonal(skel, 0)
    return skel


def _community_fitness(skeleton: np.ndarray, community: Set[int], alpha: float) -> float:
    """Lancichinetti fitness  f(C) = k_in / (k_in + k_out)^alpha."""
    if not community:
        return 0.0
    nodes = list(community)
    sub = skeleton[np.ix_(nodes, nodes)]
    k_in = float(sub.sum())                       # 2 * internal edges
    deg = float(skeleton[nodes, :].sum())         # total degree of community
    k_out = deg - k_in
    denom = (k_in + k_out) ** alpha
    return k_in / denom if denom > 0 else 0.0


def _decompose_communities(
    skeleton: np.ndarray,
    n_vars: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    max_community: Optional[int] = None,
) -> List[List[int]]:
    """Locally-extended overlapping community detection (iter-DSLA Alg. 6).

    Expansion: grow a community from a seed by repeatedly adding the
    neighbour that most improves the fitness function, until no improvement.
    Merging: fuse two communities whose node overlap exceeds ``beta``.
    """
    if max_community is None:
        max_community = n_vars
    neighbors = [set(np.where(skeleton[v] != 0)[0].tolist()) for v in range(n_vars)]

    covered: Set[int] = set()
    groups: List[Set[int]] = []
    # seed from highest-degree uncovered node first (deterministic)
    order = sorted(range(n_vars), key=lambda v: -len(neighbors[v]))

    for seed in order:
        # skip only if this node and all its neighbours are already covered
        if seed in covered and all(nb in covered for nb in neighbors[seed]):
            continue
        community = {seed}
        cur_fit = _community_fitness(skeleton, community, alpha)
        while len(community) < max_community:
            frontier = set()
            for u in community:
                frontier |= neighbors[u]
            frontier -= community
            if not frontier:
                break
            best_gain, best_node = 0.0, -1
            for cand in frontier:
                f = _community_fitness(skeleton, community | {cand}, alpha)
                if f - cur_fit > best_gain:
                    best_gain, best_node = f - cur_fit, cand
            if best_node < 0:
                break
            community.add(best_node)
            cur_fit += best_gain
        groups.append(community)
        covered |= community

    # any never-covered node (isolated) becomes its own singleton community
    for v in range(n_vars):
        if v not in covered:
            groups.append({v})

    # merge overly-overlapping communities
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if not a or not b:
                    continue
                overlap = len(a & b)
                if overlap and overlap >= beta * min(len(a), len(b)):
                    groups[i] = a | b
                    groups[j] = set()
                    merged = True
            if merged:
                break
        groups = [g for g in groups if g]

    return [sorted(g) for g in groups]


def _combine_subdags(
    sub_adjs: List[Tuple[List[int], np.ndarray]],
    data: np.ndarray,
    n_vars: int,
    cardinality: np.ndarray,
    scorer: ScoringMethod,
    mp: int,
) -> np.ndarray:
    """Combine per-community sub-DAGs into one global DAG.

    Conflicting / overlapping edges are resolved greedily: every proposed
    directed edge is weighted by its local-score delta and inserted in
    descending order while preserving acyclicity and the parent cap.
    """
    # collect candidate directed edges (union over communities)
    proposed: Set[Tuple[int, int]] = set()
    for nodes, adj in sub_adjs:
        idx = np.array(nodes)
        us, vs = np.where(adj != 0)
        for u, v in zip(us, vs):
            proposed.add((int(idx[u]), int(idx[v])))

    # weight each edge by the score gain of adding u as a parent of v
    scored = []
    base_cache: Dict[int, float] = {}
    for (u, v) in proposed:
        gain = (scorer.local_score(v, [u], data, cardinality)
                - scorer.local_score(v, [], data, cardinality))
        scored.append((gain, u, v))
    scored.sort(key=lambda t: (t[0], -t[1], -t[2]), reverse=True)

    adjacency = np.zeros((n_vars, n_vars), dtype=int)
    for gain, u, v in scored:
        if int(adjacency[:, v].sum()) >= mp:
            continue
        if adjacency[v, u] == 1:          # opposite edge already placed
            continue
        if not _would_create_cycle(adjacency, u, v):
            adjacency[u, v] = 1
    return adjacency


class IterDSLALearner:
    """iter-DSLA: iterative structure decomposition learning.

    A divide-and-conquer learner for large / complex Bayesian networks
    (Jia & Li 2026).  Each iteration:

    1. **Construct** three undirected initial graphs (from the top-3 networks
       of the previous round via the SELECT / AND / OR operators).
    2. **Decompose** each into overlapping communities (Algorithm 6).
    3. **Learn** a sub-DAG for every community with a global base learner.
    4. **Combine** the sub-DAGs into a full DAG, resolving conflicts by score.
    5. **Update** the top-3 highest-scoring (distinct) networks.
    6. **Iterate** until ``n_iter`` is reached; return the best network.

    The three mutation operators keep the search out of local optima:
    SELECT keeps the skeleton of the best network, AND intersects the three
    skeletons (conservative), OR unions them (exploratory).

    Parameters
    ----------
    base_learner : object or None
        Any learner exposing ``learn(data, n_vars, cardinality,
        interaction_matrix=...)``.  Defaults to a
        :class:`StableHillClimbLearner` with a BIC score.
    n_iter : int
        Number of iterations (paper default 10; converges in ~4-5).
    alpha_comm, beta_comm : float
        Community-detection size controls (fitness exponent / merge overlap).
    score : str
        Decomposable score used to rank whole networks.
    alpha : float
        Score smoothing parameter.
    max_parents : int or None
    seed : int or None
        Seed for the random initial networks (reproducibility).

    References
    ----------
    Jia & Li (2026). "An iterative structure decomposition learning method
    for complex Bayesian networks." Complex & Intelligent Systems.
    """

    def __init__(
        self,
        base_learner=None,
        n_iter: int = 10,
        alpha_comm: float = 1.0,
        beta_comm: float = 0.5,
        score: str = "bic",
        alpha: float = 1.0,
        max_parents: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.base_learner = base_learner
        self.n_iter = n_iter
        self.alpha_comm = alpha_comm
        self.beta_comm = beta_comm
        self.score = score
        self.alpha = alpha
        self.max_parents = max_parents
        self.seed = seed

    # -- operators ------------------------------------------------------
    @staticmethod
    def _select(skels: List[np.ndarray], scores: List[float]) -> np.ndarray:
        return skels[int(np.argmax(scores))].copy()

    @staticmethod
    def _and(skels: List[np.ndarray]) -> np.ndarray:
        out = skels[0].copy()
        for s in skels[1:]:
            out &= s
        return out

    @staticmethod
    def _or(skels: List[np.ndarray]) -> np.ndarray:
        out = skels[0].copy()
        for s in skels[1:]:
            out |= s
        return out

    # -- core -----------------------------------------------------------
    def _learn_from_skeleton(
        self, skeleton, data, n_vars, cardinality, base, scorer, mp, global_inter,
        sample_weights=None,
    ) -> np.ndarray:
        """Decompose the skeleton, learn each community, and recombine.

        The input skeleton is used only to *decompose* the problem into
        communities (Algorithm 6).  Within each community the global base
        learner searches freely over all node pairs (subject to the global
        ``interaction_matrix`` if one was supplied), so genuinely new edges
        can be discovered from one iteration to the next.

        ``sample_weights`` (a probability vector over rows) is forwarded to the
        base learner unchanged: communities only subset *columns*, so the same
        per-row weights apply to every sub-problem.  The default base learner
        already carries the weighted scorer, but a user-supplied learner needs
        the weights passed through here.
        """
        communities = _decompose_communities(
            skeleton, n_vars, self.alpha_comm, self.beta_comm
        )
        sub_adjs: List[Tuple[List[int], np.ndarray]] = []
        for nodes in communities:
            if len(nodes) == 1:
                sub_adjs.append((nodes, np.zeros((1, 1), dtype=int)))
                continue
            sub_data = data[:, nodes]
            sub_card = cardinality[nodes]
            inter = None
            if global_inter is not None:
                inter = global_inter[np.ix_(nodes, nodes)]
            sub_adj = base.learn(
                sub_data, len(nodes), sub_card,
                interaction_matrix=inter, sample_weights=sample_weights,
            )
            sub_adjs.append((nodes, sub_adj))
        return _combine_subdags(sub_adjs, data, n_vars, cardinality, scorer, mp)

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = np.asarray(data)
        cardinality = np.asarray(cardinality, dtype=int)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)
        scorer = _make_scorer(self.score, self.alpha, sample_weights)
        rng = np.random.default_rng(self.seed)

        base = self.base_learner
        if base is None:
            base = StableHillClimbLearner(
                scoring=_make_scorer(self.score, self.alpha, sample_weights),
                max_parents=mp,
            )

        def total_score(adj: np.ndarray) -> float:
            return scorer.score(adj, data, cardinality)

        # ---- Step 1: three random initial DAGs -> skeletons ---------------
        def random_dag() -> np.ndarray:
            perm = rng.permutation(n_vars)
            adj = np.zeros((n_vars, n_vars), dtype=int)
            for a in range(n_vars):
                for b in range(a + 1, n_vars):
                    u, v = perm[a], perm[b]
                    if interaction_matrix is not None and interaction_matrix[u, v] == 0:
                        continue
                    if rng.random() < 0.3 and int(adj[:, v].sum()) < mp:
                        adj[u, v] = 1
            return adj

        top_adjs = [random_dag() for _ in range(3)]
        top_scores = [total_score(a) for a in top_adjs]

        best_adj = top_adjs[int(np.argmax(top_scores))].copy()
        best_score = max(top_scores)

        for _ in range(self.n_iter):
            skels = [_to_skeleton(a) for a in top_adjs]
            # three mutation operators -> three initial undirected graphs
            init_graphs = [
                self._select(skels, top_scores),
                self._and(skels),
                self._or(skels),
            ]
            if interaction_matrix is not None:
                im = (np.asarray(interaction_matrix) != 0).astype(int)
                im = ((im | im.T) > 0).astype(int)
                np.fill_diagonal(im, 0)
                init_graphs = [g & im for g in init_graphs]

            new_adjs, new_scores = [], []
            for g in init_graphs:
                adj = self._learn_from_skeleton(
                    g, data, n_vars, cardinality, base, scorer, mp, interaction_matrix,
                    sample_weights=sample_weights,
                )
                new_adjs.append(adj)
                new_scores.append(total_score(adj))

            # ---- Step 5: keep top-3 distinct among old + new --------------
            pool = list(zip(top_scores, top_adjs)) + list(zip(new_scores, new_adjs))
            pool.sort(key=lambda t: t[0], reverse=True)
            chosen_adjs, chosen_scores, seen = [], [], set()
            for sc, adj in pool:
                key = adj.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                chosen_adjs.append(adj)
                chosen_scores.append(sc)
                if len(chosen_adjs) == 3:
                    break
            while len(chosen_adjs) < 3:            # pad if fewer than 3 distinct
                chosen_adjs.append(chosen_adjs[-1].copy())
                chosen_scores.append(chosen_scores[-1])
            top_adjs, top_scores = chosen_adjs, chosen_scores

            if top_scores[0] > best_score:
                best_score = top_scores[0]
                best_adj = top_adjs[0].copy()

        return best_adj


# ---------------------------------------------------------------------------
# Bounded-treewidth BN learning via k-tree sampling  (Nie et al. 2014)
# ---------------------------------------------------------------------------


def _sample_k_tree(
    n_vars: int,
    k: int,
    rng: np.random.Generator,
    order: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Sample a random k-tree and its induced candidate-parent sets.

    A k-tree on ``n_vars`` nodes is a maximal graph of treewidth ``k``,
    built by starting from a ``(k+1)``-clique and repeatedly attaching each
    new node to a randomly chosen existing ``k``-clique (Nie et al. 2014,
    Corollary 1).  Nodes are introduced in the order ``order`` (a random
    permutation when ``None``).  Because every new node's earlier neighbours
    are exactly the ``k``-clique it attached to, those neighbours form a
    clique; hence any DAG whose parents are drawn from these candidate sets
    has a moral graph that is a subgraph of the k-tree, guaranteeing
    treewidth ``≤ k``.

    Returns
    -------
    ktree : np.ndarray, shape (n_vars, n_vars)
        Symmetric adjacency of the sampled k-tree.
    candidate_parents : dict[int, list[int]]
        For every variable, the earlier-introduced nodes it may take as
        parents (a clique in the k-tree).
    """
    if order is None:
        pi = rng.permutation(n_vars)
    else:
        pi = np.asarray(order, dtype=int)

    ktree = np.zeros((n_vars, n_vars), dtype=int)
    candidate_parents: Dict[int, List[int]] = {int(v): [] for v in range(n_vars)}

    n_init = min(k + 1, n_vars)
    init_nodes = [int(pi[j]) for j in range(n_init)]
    # Fully connect the initial clique; candidate parents follow the order.
    for a in range(n_init):
        for b in range(a):
            ktree[init_nodes[a], init_nodes[b]] = 1
            ktree[init_nodes[b], init_nodes[a]] = 1
        candidate_parents[init_nodes[a]] = init_nodes[:a]

    # List of current k-cliques (each a sorted tuple of k node indices).
    k_cliques: List[Tuple[int, ...]] = []
    if n_init >= k and k > 0:
        for combo in combinations(init_nodes, k):
            k_cliques.append(tuple(sorted(combo)))
    elif k == 0:
        k_cliques = []

    for j in range(n_init, n_vars):
        u = int(pi[j])
        if k_cliques:
            clique = k_cliques[int(rng.integers(len(k_cliques)))]
        else:
            clique = tuple(init_nodes[:k])
        parents = list(clique)
        candidate_parents[u] = parents
        for w in parents:
            ktree[u, w] = 1
            ktree[w, u] = 1
        # Register the k new k-cliques created by adding u.
        for w in parents:
            new_clique = tuple(sorted(set(parents) - {w} | {u}))
            if len(new_clique) == k:
                k_cliques.append(new_clique)

    return ktree, candidate_parents


class BoundedTreewidthLearner:
    """Learn a Bayesian network whose moral graph has treewidth ≤ ``k``.

    Implements the approximate k-tree sampling method of Nie, Mauá, de Campos
    & Ji (2014): sample several random k-trees, and for each pick the
    highest-scoring DAG whose families lie inside the k-tree, then keep the
    best DAG over all samples.  Bounding treewidth at learning time
    guarantees that subsequent junction-tree inference and sampling stay
    cheap (cost exponential in ``k`` only) — the natural structure learner
    for repeated use inside Estimation of Distribution Algorithms.

    Parameters
    ----------
    k : int
        Treewidth bound (``k=1`` recovers a tree/forest).
    n_ktrees : int
        Number of random k-trees sampled; the best DAG is returned.
    score : str
        Decomposable score name (``"bic"``, ``"aic"`` or ``"k2"``).
    alpha : float
        Score smoothing / equivalent-sample-size parameter.
    max_parents : int or None
        Extra cap on parents (the effective cap is ``min(k, max_parents)``).
    limit_table_size : bool
        Skip candidate parent sets whose joint table exceeds the sample size.
    seed : int or None
        Seed for the k-tree sampler (reproducibility).

    References
    ----------
    Nie, S., Mauá, D. D., de Campos, C. P. & Ji, Q. (2014).
    "Advances in Learning Bayesian Networks of Bounded Treewidth."
    Advances in Neural Information Processing Systems (NeurIPS) 27.
    arXiv:1406.1411.
    """

    def __init__(
        self,
        k: int = 2,
        n_ktrees: int = 100,
        score: str = "bic",
        alpha: float = 1.0,
        max_parents: Optional[int] = None,
        limit_table_size: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if k < 1:
            raise ValueError("k (treewidth bound) must be >= 1")
        self.k = int(k)
        self.n_ktrees = int(n_ktrees)
        self.score = score
        self.alpha = alpha
        self.max_parents = max_parents
        self.limit_table_size = limit_table_size
        self.seed = seed
        # Populated after learn(): the k-tree behind the returned DAG.
        self.ktree_: Optional[np.ndarray] = None

    def _best_dag_for_ktree(
        self,
        candidate_parents: Dict[int, List[int]],
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        scorer: ScoringMethod,
        cap: int,
        interaction_matrix: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """Greedy score-maximising DAG within one k-tree (K2-style)."""
        n_samples = data.shape[0]
        adjacency = np.zeros((n_vars, n_vars), dtype=int)
        total = 0.0
        for v in range(n_vars):
            possible = [
                p for p in candidate_parents[v]
                if interaction_matrix is None or interaction_matrix[p, v] != 0
            ]
            current: List[int] = []
            current_score = scorer.local_score(v, current, data, cardinality)
            improved = True
            while improved and len(current) < cap:
                improved = False
                best_parent = -1
                best_score = current_score
                for cand in possible:
                    if cand in current:
                        continue
                    trial = current + [cand]
                    if self.limit_table_size and _joint_table_size(
                        cardinality, [v] + trial
                    ) > n_samples:
                        continue
                    s = scorer.local_score(v, trial, data, cardinality)
                    if s > best_score:
                        best_score = s
                        best_parent = cand
                        improved = True
                if improved:
                    current.append(best_parent)
                    current_score = best_score
            for p in current:
                adjacency[p, v] = 1
            total += current_score
        return adjacency, total

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = np.asarray(data)
        cardinality = np.asarray(cardinality, dtype=int)
        rng = np.random.default_rng(self.seed)
        scorer = _make_scorer(self.score, self.alpha, sample_weights)

        k = min(self.k, max(1, n_vars - 1))
        cap = k if self.max_parents is None else min(k, self.max_parents)

        # A fixed permutation (if supplied) fixes the node-introduction order;
        # otherwise every sampled k-tree draws its own random order.
        fixed_order = None if permutation is None else np.asarray(permutation, dtype=int)

        best_adj = np.zeros((n_vars, n_vars), dtype=int)
        best_score = -np.inf
        best_ktree = np.zeros((n_vars, n_vars), dtype=int)

        for _ in range(max(1, self.n_ktrees)):
            ktree, candidate_parents = _sample_k_tree(n_vars, k, rng, fixed_order)
            adj, total = self._best_dag_for_ktree(
                candidate_parents, data, n_vars, cardinality,
                scorer, cap, interaction_matrix,
            )
            if total > best_score:
                best_score = total
                best_adj = adj
                best_ktree = ktree

        self.ktree_ = best_ktree
        return best_adj


# ---------------------------------------------------------------------------
# Bayesian hierarchical clustering of variables  (Marrelec et al. 2015)
# ---------------------------------------------------------------------------


def _dirichlet_multinomial_log_ml(
    data: np.ndarray,
    variables: List[int],
    cardinality: np.ndarray,
    alpha: float,
    weights: np.ndarray,
) -> float:
    """Log marginal likelihood of *variables* under a Dirichlet-multinomial.

    Treats the joint configuration of ``variables`` as a single multinomial
    with a symmetric ``Dirichlet(alpha)`` prior (total pseudo-count
    ``alpha`` spread over all configurations).  This is the BDeu/K2 family
    marginal likelihood and provides the discrete analogue of the Gaussian
    evidence used by Marrelec et al. (2015).
    """
    n_configs = int(np.prod(cardinality[np.asarray(variables, dtype=int)]))
    mult = 1
    idx = np.zeros(data.shape[0], dtype=int)
    for v in variables:
        idx += data[:, v] * mult
        mult *= int(cardinality[v])
    counts = np.bincount(idx, weights=weights, minlength=n_configs)
    n = float(counts.sum())
    a_c = alpha / n_configs
    ll = gammaln(alpha) - gammaln(alpha + n)
    ll += float(np.sum(gammaln(a_c + counts) - gammaln(a_c)))
    return ll


def bayesian_variable_clustering(
    data: np.ndarray,
    cardinality: np.ndarray,
    *,
    sample_weights: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    stop_threshold: float = 0.0,
    max_config: Optional[int] = None,
) -> Dict[str, object]:
    """Agglomerative hierarchical clustering of *variables* (linkage tree).

    Merges the pair of variable clusters with the largest **log Bayes
    factor** in favour of dependence, i.e.

        logBF(A, B) = logML(A ∪ B) − logML(A) − logML(B),

    where ``logML`` is the Dirichlet-multinomial marginal likelihood.  A
    positive logBF means the joint model beats the independent (product of
    marginals) model, so the clusters are dependent and worth merging.  The
    procedure stops automatically when no merge exceeds ``stop_threshold``
    (default 0), removing the need for an arbitrary cut height.  Unlike raw
    mutual-information linkage, the Bayes factor corrects for the
    dimensionality of the merged configuration space.

    The resulting flat clustering is directly usable as a marginal-product
    (linkage-tree) factorization for EDA sampling, or as a prior for the
    community-based structure learners (:class:`DMBBNStructureLearner`,
    :class:`IterDSLALearner`).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_vars)
    cardinality : np.ndarray, shape (n_vars,)
    sample_weights : array of float, optional
        Probability weights over rows (defaults to uniform).
    alpha : float
        Dirichlet-multinomial equivalent sample size.
    stop_threshold : float
        Stop merging once the best log Bayes factor falls at or below this
        value.  ``-inf`` forces a full dendrogram down to a single cluster.
    max_config : int or None
        Skip (disallow) any merge whose joint configuration count would
        exceed this cap, preventing combinatorial blow-up.  Defaults to
        ``max(n_samples, 1000)``.

    Returns
    -------
    dict with keys
        ``clusters`` : list of list of int
            The flat clustering at the automatic cut (marginal-product groups).
        ``merges`` : list of (list, list, float)
            Merge history as ``(cluster_a, cluster_b, log_bayes_factor)``.

    References
    ----------
    Marrelec, G., Messé, A. & Bellec, P. (2015). "A Bayesian alternative to
    mutual information for the hierarchical clustering of dependent random
    variables." PLoS ONE 10(9): e0137278.
    """
    data = np.asarray(data, dtype=int)
    cardinality = np.asarray(cardinality, dtype=int)
    n_samples, n_vars = data.shape
    if sample_weights is None:
        weights = np.ones(n_samples, dtype=float)
    else:
        weights = np.asarray(sample_weights, dtype=float) * n_samples
    if max_config is None:
        max_config = max(n_samples, 1000)

    clusters: List[List[int]] = [[v] for v in range(n_vars)]
    log_ml: List[float] = [
        _dirichlet_multinomial_log_ml(data, c, cardinality, alpha, weights)
        for c in clusters
    ]
    config_size: List[int] = [int(cardinality[v]) for v in range(n_vars)]
    merges: List[Tuple[List[int], List[int], float]] = []

    while len(clusters) > 1:
        best = (-np.inf, -1, -1)
        best_joint_ml = 0.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                joint_size = config_size[i] * config_size[j]
                if joint_size > max_config:
                    continue
                joint_vars = clusters[i] + clusters[j]
                joint_ml = _dirichlet_multinomial_log_ml(
                    data, joint_vars, cardinality, alpha, weights
                )
                log_bf = joint_ml - log_ml[i] - log_ml[j]
                if log_bf > best[0]:
                    best = (log_bf, i, j)
                    best_joint_ml = joint_ml

        log_bf, i, j = best
        if i < 0 or log_bf <= stop_threshold:
            break

        merges.append((list(clusters[i]), list(clusters[j]), float(log_bf)))
        merged = clusters[i] + clusters[j]
        merged_size = config_size[i] * config_size[j]
        # Remove j then i (j > i) and append the merged cluster.
        for idx in (j, i):
            clusters.pop(idx)
            log_ml.pop(idx)
            config_size.pop(idx)
        clusters.append(merged)
        log_ml.append(best_joint_ml)
        config_size.append(merged_size)

    return {"clusters": clusters, "merges": merges}


# ===========================================================================
# K2 variants  (docs/K2_Improvements — see K2_Improvements_Exploration.md)
# ===========================================================================
#
# The classic K2 (``K2StructureLearner``) is a single greedy pass over a
# *fixed* variable ordering.  Its accuracy is dominated by that ordering and
# by the pool of candidate parents it considers.  The helpers and the
# ``K2VariantLearner`` below bundle four cheap, high-leverage improvements
# distilled from the papers in ``docs/K2_Improvements/``:
#
#   A. data-derived variable ordering  (order_method="mi")
#   B. candidate-parent restriction    (parent_restriction="mi" | "mb")
#   C. post-K2 refinement in DAG space (refine=True)
#   D. order ensembling by edge voting (n_orderings>1)
#
# Every option keeps K2's low cost: none multiplies the base running time by
# more than a small constant, well within the 10x budget.


def _pairwise_mutual_information(
    data: np.ndarray,
    cardinality: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the symmetric pairwise mutual-information matrix (nats).

    ``MI[i, j] = sum_{a,b} p(a,b) log( p(a,b) / (p(a) p(b)) )`` using the
    (optionally weighted) empirical distribution.  Diagonal is zero.
    """
    data = np.asarray(data, dtype=int)
    n_samples, n_vars = data.shape
    if sample_weights is None:
        w = np.ones(n_samples, dtype=float)
    else:
        w = np.asarray(sample_weights, dtype=float) * n_samples
    wsum = w.sum()

    # marginal distributions
    marg = []
    for v in range(n_vars):
        c = np.bincount(data[:, v], weights=w, minlength=int(cardinality[v])).astype(float)
        marg.append(c / wsum)

    mi = np.zeros((n_vars, n_vars), dtype=float)
    for i in range(n_vars):
        ci = int(cardinality[i])
        for j in range(i + 1, n_vars):
            cj = int(cardinality[j])
            joint_idx = data[:, i] * cj + data[:, j]
            joint = np.bincount(joint_idx, weights=w, minlength=ci * cj).astype(float)
            joint = (joint / wsum).reshape(ci, cj)
            outer = np.outer(marg[i], marg[j])
            mask = joint > 0
            val = float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))
            mi[i, j] = mi[j, i] = max(val, 0.0)
    return mi


def _tarjan_scc(adj: np.ndarray) -> List[List[int]]:
    """Tarjan's strongly-connected-components of a directed 0/1 matrix.

    Returns the SCCs in reverse topological order of the condensation
    (a component appears before its predecessors), matching Tarjan's
    natural output order.
    """
    n = adj.shape[0]
    index = [0]
    idx = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: List[int] = []
    comps: List[List[int]] = []
    succ = [np.where(adj[u] != 0)[0].tolist() for u in range(n)]

    def strongconnect(v: int) -> None:
        # iterative DFS to avoid recursion limits
        work = [(v, 0)]
        idx[v] = low[v] = index[0]; index[0] += 1
        stack.append(v); on_stack[v] = True
        while work:
            node, pi = work[-1]
            if pi < len(succ[node]):
                work[-1] = (node, pi + 1)
                w = succ[node][pi]
                if idx[w] == -1:
                    idx[w] = low[w] = index[0]; index[0] += 1
                    stack.append(w); on_stack[w] = True
                    work.append((w, 0))
                elif on_stack[w]:
                    low[node] = min(low[node], idx[w])
            else:
                if low[node] == idx[node]:
                    comp = []
                    while True:
                        w = stack.pop(); on_stack[w] = False
                        comp.append(w)
                        if w == node:
                            break
                    comps.append(comp)
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])

    for v in range(n):
        if idx[v] == -1:
            strongconnect(v)
    return comps


def mi_variable_ordering(
    data: np.ndarray,
    n_vars: int,
    cardinality: np.ndarray,
    alpha: float = 1.0,
    sample_weights: Optional[np.ndarray] = None,
    top_m: Optional[int] = None,
) -> np.ndarray:
    """Derive a K2 variable ordering from data (Behjati & Beigy 2020).

    1. Build a *sparse parent graph*: for each node keep the single best
       parent (by K2 local-score gain) chosen among its highest-MI
       candidates -> a directed graph of "best cause" edges.
    2. Contract strongly-connected components (Tarjan).
    3. Topologically sort the condensation (parents before children).
    4. Order nodes inside each SCC by decreasing total MI (hubs first).

    The result is a topological-ish ordering that puts likely root causes
    early, which is exactly what K2 needs.  Cost is ``O(n^2 N)`` for the MI
    matrix plus ``O(n * top_m)`` cheap score evaluations.
    """
    data = np.asarray(data, dtype=int)
    mi = _pairwise_mutual_information(data, cardinality, sample_weights)
    scoring = K2ScoringMethod(alpha=alpha, sample_weights=sample_weights)
    if top_m is None:
        top_m = min(n_vars - 1, max(5, n_vars // 2))

    # 1. sparse parent graph: best single parent per node
    directed = np.zeros((n_vars, n_vars), dtype=int)
    for v in range(n_vars):
        cand = np.argsort(-mi[v])
        cand = [int(u) for u in cand if u != v][:top_m]
        base = scoring.local_score(v, [], data, cardinality)
        best_gain, best_u = 0.0, -1
        for u in cand:
            gain = scoring.local_score(v, [u], data, cardinality) - base
            if gain > best_gain:
                best_gain, best_u = gain, u
        if best_u >= 0:
            directed[best_u, v] = 1          # best_u is a parent (cause) of v

    # 2-3. SCC condensation + topological order (Tarjan yields reverse topo)
    comps = _tarjan_scc(directed)
    comps = list(reversed(comps))            # now parents-before-children

    # 4. within-SCC: hubs (high total MI) first
    total_mi = mi.sum(axis=1)
    order: List[int] = []
    for comp in comps:
        order.extend(sorted(comp, key=lambda x: -total_mi[x]))
    return np.asarray(order, dtype=int)


def mi_candidate_mask(
    data: np.ndarray,
    n_vars: int,
    cardinality: np.ndarray,
    top_k: int,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Symmetric candidate-parent mask keeping each node's top-k MI neighbours.

    Restricting K2's parent search to a small, data-driven neighbourhood
    (the "brave/careful" candidate idea of BigBraveBN and the MI pre-screen
    of Behjati 2020) removes spurious candidates -> better *and* faster.
    """
    mi = _pairwise_mutual_information(data, cardinality, sample_weights)
    mask = np.zeros((n_vars, n_vars), dtype=int)
    for v in range(n_vars):
        nbrs = [int(u) for u in np.argsort(-mi[v]) if u != v and mi[v, u] > 0][:top_k]
        for u in nbrs:
            mask[v, u] = mask[u, v] = 1
    return mask


def elasticnet_candidate_mask(
    data: np.ndarray,
    n_vars: int,
    cardinality: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    l1_ratio: float = 0.5,
    C: float = 0.5,
) -> np.ndarray:
    """Candidate-parent mask from elastic-net Markov blankets (Tabar 2025).

    For each target, fit an elastic-net-penalised (L1+L2) multinomial
    logistic regression on all other variables; predictors with a non-zero
    coefficient group form the estimated Markov blanket.  The union
    (symmetrised) is returned as an interaction mask for K2.  Falls back to
    the MI mask if scikit-learn is unavailable.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:  # pragma: no cover
        return mi_candidate_mask(data, n_vars, cardinality,
                                 top_k=max(5, n_vars // 2),
                                 sample_weights=sample_weights)
    import warnings
    data = np.asarray(data, dtype=int)
    n_samples = data.shape[0]
    w = None if sample_weights is None else np.asarray(sample_weights, float) * n_samples

    # one-hot design for all variables once
    blocks, span = [], []
    start = 0
    for v in range(n_vars):
        k = int(cardinality[v])
        cols = np.zeros((n_samples, k - 1), dtype=float)
        for s in range(1, k):
            cols[:, s - 1] = (data[:, v] == s)
        blocks.append(cols)
        span.append((start, start + k - 1))
        start += k - 1
    X_all = np.hstack(blocks) if blocks else np.zeros((n_samples, 0))

    mask = np.zeros((n_vars, n_vars), dtype=int)
    for v in range(n_vars):
        y = data[:, v]
        if np.unique(y).size < 2:
            continue
        cols_keep = [c for u in range(n_vars) if u != v
                     for c in range(span[u][0], span[u][1])]
        Xv = X_all[:, cols_keep]
        model = LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio,
                                   solver="saga", C=C, max_iter=200, tol=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xv, y, sample_weight=w)
        coef = np.abs(model.coef_).sum(axis=0)   # aggregate over classes
        pos = 0
        for u in range(n_vars):
            if u == v:
                continue
            width = span[u][1] - span[u][0]
            if width > 0 and coef[pos:pos + width].sum() > 1e-8:
                mask[v, u] = mask[u, v] = 1
            pos += width
    return mask


class K2VariantLearner:
    """K2 with data-driven ordering, candidate pruning, refinement, ensembling.

    A configurable superset of :class:`K2StructureLearner` gathering the most
    feasible, high-impact improvements from ``docs/K2_Improvements/`` (see
    ``K2_Improvements_Exploration.md``).  Every enhancement preserves K2's
    speed to within a small constant factor.

    Parameters
    ----------
    order_method : {"given", "mi"}
        ``"given"`` uses the supplied ``permutation`` (classic K2).
        ``"mi"`` derives a topological-ish ordering from data via
        :func:`mi_variable_ordering` (Behjati & Beigy 2020; the single most
        impactful change, since ordering dominates K2 accuracy).
    parent_restriction : {None, "mi", "mb"}
        Restrict candidate parents to a data-driven neighbourhood.
        ``"mi"`` keeps each node's top-``mi_top_k`` mutual-information
        neighbours; ``"mb"`` uses elastic-net Markov blankets (Tabar 2025).
        Combined with any externally supplied ``interaction_matrix``.
    refine : bool
        After the K2 pass, polish the DAG with a bounded stable
        hill-climb over add/delete/reverse moves (switch from ordering
        space to DAG space, Xiang et al. 2024) to fix reversed/missing edges.
    n_orderings : int
        When > 1, run K2 from several orderings (the MI ordering plus
        perturbations) and keep the edges that appear in a majority of the
        resulting DAGs, breaking ties toward acyclicity — an order-robust
        ensemble (Kitson & Constantinou 2024).
    mi_top_k : int or None
        Neighbourhood size for ``parent_restriction="mi"``.  ``None`` ->
        ``max(2*max_parents, 10)``.
    max_parents, alpha, limit_table_size, refine_max_iter, seed
        Standard K2 / refinement controls.
    """

    def __init__(
        self,
        order_method: str = "mi",
        parent_restriction: Optional[str] = "mi",
        refine: bool = False,
        n_orderings: int = 1,
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
        limit_table_size: bool = True,
        mi_top_k: Optional[int] = None,
        refine_max_iter: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.order_method = order_method
        self.parent_restriction = parent_restriction
        self.refine = refine
        self.n_orderings = n_orderings
        self.max_parents = max_parents
        self.alpha = alpha
        self.limit_table_size = limit_table_size
        self.mi_top_k = mi_top_k
        self.refine_max_iter = refine_max_iter
        self.seed = seed

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        ordering: Optional[np.ndarray] = None,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        data = np.asarray(data)
        cardinality = np.asarray(cardinality, dtype=int)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)
        rng = np.random.default_rng(self.seed)

        # ---- candidate-parent restriction (combine with any user mask) ----
        inter = None if interaction_matrix is None else (np.asarray(interaction_matrix) != 0).astype(int)
        if self.parent_restriction == "mi":
            top_k = self.mi_top_k if self.mi_top_k is not None else max(2 * mp, 10)
            top_k = min(top_k, n_vars - 1)
            m = mi_candidate_mask(data, n_vars, cardinality, top_k, sample_weights)
            inter = m if inter is None else (inter & m)
        elif self.parent_restriction == "mb":
            m = elasticnet_candidate_mask(data, n_vars, cardinality, sample_weights)
            inter = m if inter is None else (inter & m)

        # ---- base ordering ------------------------------------------------
        if self.order_method == "mi":
            base_order = mi_variable_ordering(
                data, n_vars, cardinality, self.alpha, sample_weights,
                top_m=self.mi_top_k,
            )
        elif permutation is not None:
            base_order = np.asarray(permutation, dtype=int)
        elif ordering is not None:
            base_order = np.asarray(ordering, dtype=int)
        else:
            base_order = np.arange(n_vars, dtype=int)

        k2 = K2StructureLearner(max_parents=mp, alpha=self.alpha,
                                limit_table_size=self.limit_table_size)

        def run(order):
            return k2.learn(data, n_vars, cardinality, permutation=order,
                            interaction_matrix=inter, sample_weights=sample_weights)

        # ---- single order or order-ensemble -------------------------------
        if self.n_orderings <= 1:
            adjacency = run(base_order)
        else:
            orders = [base_order]
            for _ in range(self.n_orderings - 1):
                orders.append(rng.permutation(n_vars))
            votes = np.zeros((n_vars, n_vars), dtype=float)
            for o in orders:
                votes += run(o)
            adjacency = self._vote_to_dag(votes, len(orders), mp)

        # ---- optional DAG-space refinement --------------------------------
        if self.refine:
            adjacency = self._refine(adjacency, data, n_vars, cardinality,
                                     mp, inter, sample_weights)
        return adjacency

    # ------------------------------------------------------------------
    @staticmethod
    def _vote_to_dag(votes: np.ndarray, n_runs: int, mp: int) -> np.ndarray:
        """Keep majority-voted directed edges, inserting greedily & acyclically."""
        n = votes.shape[0]
        cand = []
        for u in range(n):
            for v in range(n):
                if u != v and votes[u, v] > 0:
                    cand.append((votes[u, v], u, v))
        cand.sort(key=lambda t: (t[0], -t[1], -t[2]), reverse=True)
        adj = np.zeros((n, n), dtype=int)
        half = n_runs / 2.0
        for cnt, u, v in cand:
            if cnt < half:                    # majority rule
                continue
            if adj[v, u] == 1:                # opposite already placed
                continue
            if int(adj[:, v].sum()) >= mp:
                continue
            if not _would_create_cycle(adj, u, v):
                adj[u, v] = 1
        return adj

    def _refine(self, adjacency, data, n_vars, cardinality, mp, inter, sample_weights):
        """Seed a stable hill-climb with the K2 graph and polish it."""
        scorer = K2ScoringMethod(alpha=self.alpha, sample_weights=sample_weights)
        hc = StableHillClimbLearner(scoring=scorer, max_parents=mp,
                                    max_iter=self.refine_max_iter,
                                    limit_table_size=self.limit_table_size)
        # StableHillClimbLearner starts from empty; emulate a warm start by
        # scoring add/del/reverse from the K2 graph via a short local search.
        return _hill_climb_from(adjacency, data, n_vars, cardinality, scorer,
                                mp, inter, self.limit_table_size,
                                self.refine_max_iter)


def _hill_climb_from(
    adjacency: np.ndarray,
    data: np.ndarray,
    n_vars: int,
    cardinality: np.ndarray,
    scorer: ScoringMethod,
    mp: int,
    interaction_matrix: Optional[np.ndarray],
    limit_table_size: bool,
    max_iter: int,
) -> np.ndarray:
    """Warm-started stable hill-climb (add/delete/reverse) from *adjacency*.

    Deterministic tie-breaking (Kitson & Constantinou 2024) via ``_op_key``.
    """
    n_samples = data.shape[0]
    adj = adjacency.copy()
    cache: dict = {}

    def local(var, parents):
        key = (var, tuple(sorted(parents)))
        if key not in cache:
            cache[key] = scorer.local_score(var, list(parents), data, cardinality)
        return cache[key]

    def parents_of(v):
        return list(np.where(adj[:, v] > 0)[0])

    def allowed(u, v):
        return interaction_matrix is None or interaction_matrix[u, v] != 0

    for _ in range(max_iter):
        best_key, best_op = None, None
        for u in range(n_vars):
            for v in range(n_vars):
                if u == v:
                    continue
                pa_v = parents_of(v)
                if adj[u, v] == 0 and adj[v, u] == 0 and allowed(u, v):
                    if len(pa_v) < mp and not _would_create_cycle(adj, u, v):
                        new_pa = pa_v + [u]
                        if not limit_table_size or _joint_table_size(cardinality, [v] + new_pa) <= n_samples:
                            delta = local(v, new_pa) - local(v, pa_v)
                            k = _op_key(delta, "add", u, v)
                            if best_key is None or k > best_key:
                                best_key, best_op = k, ("add", u, v)
                if adj[u, v] == 1:
                    new_pa = [p for p in pa_v if p != u]
                    delta = local(v, new_pa) - local(v, pa_v)
                    k = _op_key(delta, "del", u, v)
                    if best_key is None or k > best_key:
                        best_key, best_op = k, ("del", u, v)
                    # reverse u->v  =>  v->u
                    pa_u = parents_of(u)
                    if allowed(v, u) and len(pa_u) < mp:
                        tmp = adj.copy(); tmp[u, v] = 0
                        if not _would_create_cycle(tmp, v, u):
                            new_pa_v = [p for p in pa_v if p != u]
                            new_pa_u = pa_u + [v]
                            if not limit_table_size or _joint_table_size(cardinality, [u] + new_pa_u) <= n_samples:
                                delta = (local(v, new_pa_v) - local(v, pa_v)
                                         + local(u, new_pa_u) - local(u, pa_u))
                                k = _op_key(delta, "rev", u, v)
                                if best_key is None or k > best_key:
                                    best_key, best_op = k, ("rev", u, v)
        if best_op is None or best_key[0] <= 1e-10:
            break
        op, u, v = best_op
        if op == "add":
            adj[u, v] = 1
        elif op == "del":
            adj[u, v] = 0
        else:
            adj[u, v] = 0; adj[v, u] = 1
    return adj


# ===========================================================================
# Objective-guided K2 orderings and an independent baseline
# ===========================================================================
#
# In an EDA the Bayesian network is learned from a *selected* population whose
# rows carry a per-solution probability (``sample_weights``), derived from the
# objective value.  The three learners below exploit (or deliberately ignore)
# that signal:
#
#   * ``IndependentBNLearner`` (Univ_BN) — empty graph baseline: every
#     variable is marginally independent, only the univariate tables are fit.
#   * ``FeatureImportanceK2Learner`` (FI_k2) — order K2's variables by their
#     univariate predictive power for the solution probability.
#   * ``RFEK2Learner`` (RFE_k2) — order K2's variables by a recursive /
#     minimum-redundancy criterion so that each new variable adds information
#     about the solution probability beyond the variables already placed.


class IndependentBNLearner:
    """Univ_BN — the fully independent (empty-graph) baseline.

    Learns *no* structure: the returned adjacency matrix is all zeros, so
    every variable is a root and :meth:`BayesianNetwork.learn_parameters`
    fits only the univariate marginal probability tables.  Useful as a
    lower-bound reference against which structure-learning methods are
    compared (a product-of-marginals / univariate model, as used by the
    simplest EDAs such as UMDA/PBIL).
    """

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.zeros((n_vars, n_vars), dtype=int)


def _solution_probability_target(
    sample_weights: Optional[np.ndarray],
    n_samples: int,
) -> Optional[np.ndarray]:
    """Return the per-solution probability used as the feature-selection target.

    The target is the ``sample_weights`` vector (the probability computed from
    each solution's objective value).  Returns ``None`` when no usable signal
    is available (weights missing or all equal), so callers can fall back to a
    neutral ordering.
    """
    if sample_weights is None:
        return None
    y = np.asarray(sample_weights, dtype=float).reshape(-1)
    if y.shape[0] != n_samples:
        raise ValueError("sample_weights length must match the number of rows")
    if not np.any(np.abs(y - y[0]) > 1e-15):      # constant target => no signal
        return None
    return y


def _rank_desc_with_random_ties(
    importance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Argsort *importance* descending, breaking ties uniformly at random."""
    importance = np.asarray(importance, dtype=float)
    importance = np.where(np.isfinite(importance), importance, -np.inf)
    tie_break = rng.random(importance.shape[0])
    # np.lexsort: last key is primary.  Primary = -importance (ascending on the
    # negative == descending on importance); secondary = random tie-break.
    return np.lexsort((tie_break, -importance))


def feature_importance_ordering(
    data: np.ndarray,
    solution_prob: Optional[np.ndarray],
    method: str = "mutual_info",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Rank variables by univariate predictive power for the solution probability.

    Computes, for every variable independently, an importance score measuring
    how well it predicts ``solution_prob`` and returns the permutation that
    places the most important variable first (ties broken at random).  This is
    the ordering used by **FI_k2**.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_vars)
        The (integer-coded) solutions.
    solution_prob : np.ndarray or None
        Per-solution probability target (``sample_weights``).  If ``None`` or
        constant, a uniformly random permutation is returned.
    method : {"mutual_info", "f_regression", "r_regression"}
        scikit-learn univariate importance measure.
        ``"mutual_info"`` uses :func:`sklearn.feature_selection.mutual_info_regression`
        with ``discrete_features=True`` (captures non-linear dependence);
        ``"f_regression"`` uses the F-statistic; ``"r_regression"`` uses the
        absolute Pearson correlation.
    seed : int or None
        Seed for tie-breaking (and the MI estimator's internal randomness).

    Returns
    -------
    np.ndarray
        A permutation of ``[0 … n_vars-1]``.
    """
    data = np.asarray(data)
    n_samples, n_vars = data.shape
    rng = np.random.default_rng(seed)

    y = _solution_probability_target(solution_prob, n_samples)
    if y is None:
        return rng.permutation(n_vars)

    X = data.astype(float)
    name = method.lower()
    if name in ("mutual_info", "mutual_info_regression", "mi"):
        from sklearn.feature_selection import mutual_info_regression
        importance = mutual_info_regression(
            X, y, discrete_features=True, random_state=int(rng.integers(1 << 31))
        )
    elif name in ("f_regression", "f"):
        from sklearn.feature_selection import f_regression
        f_stat, _ = f_regression(X, y)
        importance = np.nan_to_num(f_stat, nan=0.0, posinf=0.0, neginf=0.0)
    elif name in ("r_regression", "r", "pearson"):
        from sklearn.feature_selection import r_regression
        importance = np.abs(np.nan_to_num(r_regression(X, y), nan=0.0))
    else:
        raise ValueError(
            "method must be 'mutual_info', 'f_regression', or 'r_regression'"
        )

    return _rank_desc_with_random_ties(importance, rng)


def rfe_ordering(
    data: np.ndarray,
    solution_prob: Optional[np.ndarray],
    cardinality: np.ndarray,
    selector: str = "mrmr",
    seed: Optional[int] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Recursive / minimum-redundancy ordering of variables for **RFE_k2**.

    Unlike :func:`feature_importance_ordering`, the importance of a variable
    here is judged *conditionally* on the variables already placed, so
    redundant variables are pushed back.

    Two strategies are provided:

    ``selector="mrmr"`` (default)
        Greedy minimum-Redundancy-Maximum-Relevance forward selection.  The
        first variable is the one with the highest mutual information with the
        solution probability (relevance).  Each subsequent variable maximises
        ``relevance − mean redundancy``, where redundancy is the mean
        mutual information with the variables already selected.  This directly
        realises "add the most informative variable, then the one that adds
        most while overlapping least with those already chosen".

    ``selector="rfe"``
        scikit-learn :class:`~sklearn.feature_selection.RFE` with a random
        forest estimator predicting the solution probability.  Variables are
        ordered by RFE's elimination ranking: the last variable kept comes
        first, the first variable eliminated comes last.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_vars)
    solution_prob : np.ndarray or None
        Per-solution probability target.  If ``None``/constant a uniformly
        random permutation is returned.
    cardinality : np.ndarray
        Needed for the feature–feature mutual-information (redundancy) matrix.
    selector : {"mrmr", "rfe"}
    seed : int or None
    sample_weights : np.ndarray or None
        Weights for the feature–feature MI matrix (defaults to ``solution_prob``).

    Returns
    -------
    np.ndarray
        A permutation of ``[0 … n_vars-1]``.
    """
    data = np.asarray(data)
    n_samples, n_vars = data.shape
    cardinality = np.asarray(cardinality, dtype=int)
    rng = np.random.default_rng(seed)

    y = _solution_probability_target(solution_prob, n_samples)
    if y is None:
        return rng.permutation(n_vars)

    name = selector.lower()
    if name == "rfe":
        return _rfe_sklearn_ordering(data, y, rng)
    if name != "mrmr":
        raise ValueError("selector must be 'mrmr' or 'rfe'")

    # ---- mRMR greedy forward selection -------------------------------------
    from sklearn.feature_selection import mutual_info_regression
    relevance = mutual_info_regression(
        data.astype(float), y, discrete_features=True,
        random_state=int(rng.integers(1 << 31)),
    )
    # feature-feature MI (redundancy); reuse the weighted MI matrix helper.
    ff_mi = _pairwise_mutual_information(
        data, cardinality,
        sample_weights if sample_weights is not None else None,
    )

    remaining = list(range(n_vars))
    # first variable: maximum relevance (random tie-break)
    first = int(_rank_desc_with_random_ties(relevance, rng)[0])
    order = [first]
    remaining.remove(first)

    while remaining:
        rem = np.array(remaining)
        redundancy = ff_mi[np.ix_(rem, order)].mean(axis=1)
        score = relevance[rem] - redundancy
        # rank remaining by score desc with random ties, take the best
        best_local = int(_rank_desc_with_random_ties(score, rng)[0])
        nxt = int(rem[best_local])
        order.append(nxt)
        remaining.remove(nxt)

    return np.asarray(order, dtype=int)


def _rfe_sklearn_ordering(
    data: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permutation from scikit-learn RFE (kept-longest first, eliminated last)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFE

    est = RandomForestRegressor(
        n_estimators=100, random_state=int(rng.integers(1 << 31)), n_jobs=1
    )
    rfe = RFE(est, n_features_to_select=1, step=1)
    rfe.fit(data.astype(float), y)
    # ranking_ == 1 for the last-surviving feature; larger == eliminated earlier.
    # Ascending ranking => most important (kept longest) first.
    ranking = np.asarray(rfe.ranking_, dtype=float)
    tie_break = rng.random(ranking.shape[0])
    return np.lexsort((tie_break, ranking))


class FeatureImportanceK2Learner:
    """FI_k2 — K2 seeded with a univariate feature-importance ordering.

    Ranks the variables by how well each one predicts the per-solution
    probability (``sample_weights``) using a scikit-learn univariate measure
    (:func:`feature_importance_ordering`), then runs classic K2 with that
    permutation.  The most predictive variable is placed first so it may act
    as a parent of the rest.

    Parameters
    ----------
    importance : {"mutual_info", "f_regression", "r_regression"}
        Univariate importance measure.
    max_parents, alpha, limit_table_size
        Standard K2 controls.
    seed : int or None
        Seed for random tie-breaking.
    """

    def __init__(
        self,
        importance: str = "mutual_info",
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
        limit_table_size: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.importance = importance
        self.max_parents = max_parents
        self.alpha = alpha
        self.limit_table_size = limit_table_size
        self.seed = seed
        self.ordering_: Optional[np.ndarray] = None

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        order = feature_importance_ordering(
            data, sample_weights, method=self.importance, seed=self.seed
        )
        self.ordering_ = order
        return K2StructureLearner(
            max_parents=self.max_parents,
            alpha=self.alpha,
            limit_table_size=self.limit_table_size,
        ).learn(
            data, n_vars, cardinality,
            permutation=order,
            interaction_matrix=interaction_matrix,
            sample_weights=sample_weights,
        )


class RFEK2Learner:
    """RFE_k2 — K2 seeded with a recursive / min-redundancy ordering.

    Ranks variables by their *conditional* contribution to predicting the
    per-solution probability (:func:`rfe_ordering`): each variable added to
    the ordering is the one that best explains the solution probability while
    overlapping least with the variables already placed.  K2 is then run with
    that permutation.

    Parameters
    ----------
    selector : {"mrmr", "rfe"}
        ``"mrmr"`` (default) uses greedy minimum-redundancy-maximum-relevance
        forward selection; ``"rfe"`` uses scikit-learn recursive feature
        elimination with a random-forest estimator.
    max_parents, alpha, limit_table_size
        Standard K2 controls.
    seed : int or None
        Seed for random tie-breaking / estimator randomness.
    """

    def __init__(
        self,
        selector: str = "mrmr",
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
        limit_table_size: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.selector = selector
        self.max_parents = max_parents
        self.alpha = alpha
        self.limit_table_size = limit_table_size
        self.seed = seed
        self.ordering_: Optional[np.ndarray] = None

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        order = rfe_ordering(
            data, sample_weights, cardinality,
            selector=self.selector, seed=self.seed,
            sample_weights=sample_weights,
        )
        self.ordering_ = order
        return K2StructureLearner(
            max_parents=self.max_parents,
            alpha=self.alpha,
            limit_table_size=self.limit_table_size,
        ).learn(
            data, n_vars, cardinality,
            permutation=order,
            interaction_matrix=interaction_matrix,
            sample_weights=sample_weights,
        )
