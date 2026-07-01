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
