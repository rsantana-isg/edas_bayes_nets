"""
Structure learning algorithms for the class of *polytree* (singly connected)
Bayesian networks.

A polytree is a DAG whose underlying undirected graph (the *skeleton*) has no
cycles -- i.e. at most one undirected path connects any two variables.  Chains,
trees and forests are special cases.  Polytrees admit exact linear-time
inference and have low sample complexity, which makes them attractive as the
probabilistic model of a Factorized Distribution Algorithm (see PADA / FDA-SC).

All learners here share the ``learn()`` signature of
:mod:`bayes_nets.structure_learning` (``permutation``, ``interaction_matrix``,
``max_parents``, ``sample_weights``) and return an adjacency matrix.

ChowLiuTreeLearner
    Maximum-weight spanning forest over pairwise mutual information, oriented
    away from a root (Chow & Liu 1968).  Every node has at most one parent, so
    the result is a *branching*.  Dasgupta (1999) proves the optimal branching
    is within a bounded factor of the optimal polytree in log-likelihood, which
    makes this both a fast learner in its own right and the natural baseline
    for the polytree learners below.

RebanePearlPolytreeLearner
    Chow-Liu skeleton plus the Rebane-Pearl orientation step: for a non-adjacent
    pair (a, b) sharing a neighbour c, marginal independence of a and b implies
    the collider a -> c <- b.  Remaining edges are oriented without introducing
    further colliders.

PolytreeLPALearner
    The LPA algorithm used by PADA / FDA-SC (Ochoa, Muehlenbein & Soto 2000).
    Edges are ranked by a *global* dependency degree
    ``DepG(a,b) = min(Dep(a,b), min_z Dep(a,b|z))`` and inserted greedily while
    the skeleton stays singly connected.  Orientation compares the dependency
    before and after instantiating the middle node: ``Dep(a,b|c) > Dep(a,b)``
    signals the head-to-head pattern ``a -> c <- b``.

CausalPolytreeLearner
    The sheaf-based causal polytree recovery of Huete & de Campos (1993).
    Variables are inserted one at a time into a partial structure; the position
    of a new node is located by walking the *sheaf* (set of directly connected
    nodes) using only marginal and first-order conditional independence tests.
    Runs in O(n^2) tests.

Thresholds
----------
The dependency thresholds ``e0`` (marginal) and ``e1`` (conditional) of the LPA
algorithm depend on the sample size.  When not supplied explicitly they are
derived from the chi-square distribution of the likelihood-ratio statistic
``2 N I(X;Y)``, which gives the sample-size-adaptive behaviour the original
paper asks for:  ``e0 = chi2.isf(alpha, dof) / (2 N)``.

References
----------
Chow & Liu (1968). "Approximating discrete probability distributions with
dependence trees." IEEE Trans. Information Theory 14(3).

Rebane & Pearl (1987). "The recovery of causal poly-trees from statistical
data." Uncertainty in Artificial Intelligence 3.

Huete & de Campos (1993). "Learning causal polytrees." ECSQARU, LNCS 747.

Dasgupta (1999). "Learning polytrees." UAI 1999.

Ochoa, Muehlenbein & Soto (2000). "A Factorized Distribution Algorithm Using
Single Connected Bayesian Networks." PPSN VI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import chi2 as chi2_dist

from bayes_nets.structure_learning import (
    _chi_square_conditional_independence,
    _default_max_parents,
    _pairwise_mutual_information,
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


class _UnionFind:
    """Disjoint-set forest used to keep the skeleton singly connected."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]  # path halving
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of *a* and *b*; False when already connected."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _normalize_weights(n: int, sample_weights: Optional[np.ndarray]) -> np.ndarray:
    """Row weights scaled so that they sum to *n* (as the CI tests expect)."""
    if sample_weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    total = float(w.sum())
    return w * (n / total) if total > 0 else np.ones(n, dtype=float)


def _conditional_mutual_information(
    data: np.ndarray,
    x: int,
    y: int,
    z: int,
    cardinality: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted ``I(X;Y|Z) = sum_z p(z) I(X;Y | Z=z)`` in nats."""
    card_x = int(cardinality[x])
    card_y = int(cardinality[y])
    card_z = int(cardinality[z])

    idx = (data[:, z] * card_x + data[:, x]) * card_y + data[:, y]
    joint = np.bincount(idx, weights=weights, minlength=card_z * card_x * card_y)
    joint = joint.astype(float).reshape(card_z, card_x, card_y)
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint /= total

    p_z = joint.sum(axis=(1, 2))
    cmi = 0.0
    for zv in range(card_z):
        if p_z[zv] <= 0:
            continue
        cond = joint[zv] / p_z[zv]
        px = cond.sum(axis=1)
        py = cond.sum(axis=0)
        outer = np.outer(px, py)
        mask = (cond > 0) & (outer > 0)
        if not np.any(mask):
            continue
        cmi += p_z[zv] * float(np.sum(cond[mask] * np.log(cond[mask] / outer[mask])))
    return max(cmi, 0.0)


def _mi_threshold(
    n_samples: float,
    cardinality: np.ndarray,
    x: int,
    y: int,
    alpha: float,
    n_cond_configs: int = 1,
) -> float:
    """Mutual-information value at which the chi-square test rejects independence.

    The likelihood-ratio statistic ``2 N I`` is asymptotically chi-square with
    ``(|X|-1)(|Y|-1)`` degrees of freedom (times the number of conditioning
    configurations), so the critical MI scales as ``1 / N`` -- exactly the
    sample-size dependence the LPA thresholds require.
    """
    dof = (int(cardinality[x]) - 1) * (int(cardinality[y]) - 1) * max(n_cond_configs, 1)
    if dof <= 0 or n_samples <= 0:
        return 0.0
    return float(chi2_dist.isf(alpha, dof)) / (2.0 * n_samples)


def _orient_remaining_no_collider(
    n_vars: int,
    skeleton: np.ndarray,
    directed: np.ndarray,
    permutation: Optional[np.ndarray],
) -> np.ndarray:
    """Orient the still-undirected skeleton edges without creating new colliders.

    In a polytree skeleton no two neighbours of a node are adjacent, so *any*
    second parent of a node forms a collider.  Hence the rule: once a node has
    an incoming arc, every remaining edge at that node must point away from it.
    This is propagated to a fixed point; whatever is left belongs to components
    with no orientation constraint at all and is oriented away from a root,
    which by construction introduces no head-to-head pattern.
    """
    directed = directed.copy()

    def undirected_at(v: int) -> List[int]:
        return [
            u for u in range(n_vars)
            if skeleton[v, u] == 1 and directed[v, u] == 0 and directed[u, v] == 0
        ]

    def propagate() -> None:
        """A node that already has a parent cannot acquire a second one."""
        changed = True
        while changed:
            changed = False
            for v in range(n_vars):
                if not np.any(directed[:, v] == 1):
                    continue
                for u in undirected_at(v):
                    directed[v, u] = 1
                    changed = True

    propagate()

    # Whatever is left is unconstrained: orient it away from a root chosen by
    # the permutation, which by construction creates no head-to-head node.
    if permutation is None:
        order = list(range(n_vars))
    else:
        order = [int(v) for v in np.asarray(permutation, dtype=int)]

    for root in order:
        if not undirected_at(root):
            continue
        stack = [root]
        while stack:
            v = stack.pop()
            for u in undirected_at(v):
                directed[v, u] = 1
                stack.append(u)
        propagate()
    return directed


def _finalize_adjacency(
    n_vars: int,
    skeleton: np.ndarray,
    directed: np.ndarray,
    strength: np.ndarray,
    max_parents: int,
) -> np.ndarray:
    """Assemble the adjacency matrix, capping the in-degree at *max_parents*.

    Excess parents are dropped weakest-first (by dependency *strength*) rather
    than reversed, since reversing could break the singly-connected property.
    """
    adjacency = np.zeros((n_vars, n_vars), dtype=int)
    for u in range(n_vars):
        for v in range(n_vars):
            if skeleton[u, v] == 1 and directed[u, v] == 1 and directed[v, u] == 0:
                adjacency[u, v] = 1

    if max_parents >= 1:
        for v in range(n_vars):
            parents = np.flatnonzero(adjacency[:, v])
            if len(parents) <= max_parents:
                continue
            keep = sorted(parents, key=lambda p: -strength[p, v])[:max_parents]
            adjacency[:, v] = 0
            adjacency[keep, v] = 1
    return adjacency


def _spanning_forest(
    n_vars: int,
    ranked_edges: List[Tuple[float, int, int]],
    interaction_matrix: Optional[np.ndarray],
) -> np.ndarray:
    """Greedily add the highest-weight edges that keep the graph acyclic.

    The result is a maximum-weight spanning forest -- the skeleton of a
    polytree, with at most ``n_vars - 1`` edges.
    """
    skeleton = np.zeros((n_vars, n_vars), dtype=int)
    uf = _UnionFind(n_vars)
    n_edges = 0
    for _, u, v in ranked_edges:
        if n_edges >= n_vars - 1:
            break
        if interaction_matrix is not None and interaction_matrix[u, v] == 0:
            continue
        if uf.union(u, v):
            skeleton[u, v] = skeleton[v, u] = 1
            n_edges += 1
    return skeleton


# ---------------------------------------------------------------------------
# Chow-Liu branching  (Chow & Liu 1968; Dasgupta 1999 approximation guarantee)
# ---------------------------------------------------------------------------


class ChowLiuTreeLearner:
    """Maximum-weight spanning forest over pairwise mutual information.

    Edges whose mutual information falls below the (sample-size adaptive)
    independence threshold are never inserted, so the result is in general a
    forest rather than a single tree.  Orientation is away from a root, giving
    each node at most one parent.

    Dasgupta (1999) shows the cost (negative log-likelihood) of the optimal
    branching is at most a bounded factor away from that of the optimal
    polytree, so this cheap O(n^2) learner is a principled polytree
    approximation as well as a useful baseline.

    Parameters
    ----------
    alpha_ci : float
        Significance level of the independence threshold on MI.  Set to 1.0 to
        keep every edge of the spanning tree regardless of strength.
    max_parents : int or None
        Capped at 1 for a branching; kept for signature compatibility.
    """

    def __init__(self, alpha_ci: float = 0.05, max_parents: Optional[int] = None) -> None:
        self.alpha_ci = alpha_ci
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
        n_samples = data.shape[0]
        cardinality = np.asarray(cardinality, dtype=int)

        mi = _pairwise_mutual_information(data, cardinality, sample_weights)
        ranked = _rank_pairs(n_vars, mi, cardinality, n_samples, self.alpha_ci)
        skeleton = _spanning_forest(n_vars, ranked, interaction_matrix)
        directed = _orient_remaining_no_collider(
            n_vars, skeleton, np.zeros((n_vars, n_vars), dtype=int), permutation
        )
        return _finalize_adjacency(n_vars, skeleton, directed, mi, max_parents=1)


def _rank_pairs(
    n_vars: int,
    weight: np.ndarray,
    cardinality: np.ndarray,
    n_samples: int,
    alpha_ci: float,
) -> List[Tuple[float, int, int]]:
    """Pairs above the independence threshold, sorted by decreasing weight."""
    edges: List[Tuple[float, int, int]] = []
    for u in range(n_vars):
        for v in range(u + 1, n_vars):
            w = float(weight[u, v])
            if alpha_ci < 1.0:
                if w <= _mi_threshold(n_samples, cardinality, u, v, alpha_ci):
                    continue
            elif w <= 0.0:
                continue
            edges.append((w, u, v))
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))
    return edges


# ---------------------------------------------------------------------------
# Rebane-Pearl polytree recovery
# ---------------------------------------------------------------------------


class RebanePearlPolytreeLearner:
    """Chow-Liu skeleton with Rebane-Pearl collider orientation.

    Phase 1 -- Skeleton: maximum-weight spanning forest over mutual information.
    Phase 2 -- Colliders: for every node ``c`` and every pair of its neighbours
    ``a, b`` (necessarily non-adjacent in a polytree), a *marginal* independence
    ``a ⟂ b`` means the unique path a - c - b contains a head-to-head node, so
    orient ``a -> c <- b``.
    Phase 3 -- The remaining edges are oriented without creating new colliders;
    those orientations are not identifiable from the data and any of them
    encodes the same independence model.

    Parameters
    ----------
    alpha_ci : float
        Significance level for the chi-square independence tests.
    max_parents : int or None
    """

    def __init__(self, alpha_ci: float = 0.05, max_parents: Optional[int] = None) -> None:
        self.alpha_ci = alpha_ci
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
        n_samples = data.shape[0]
        cardinality = np.asarray(cardinality, dtype=int)
        weights = _normalize_weights(n_samples, sample_weights)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        mi = _pairwise_mutual_information(data, cardinality, sample_weights)
        ranked = _rank_pairs(n_vars, mi, cardinality, n_samples, self.alpha_ci)
        skeleton = _spanning_forest(n_vars, ranked, interaction_matrix)

        directed = np.zeros((n_vars, n_vars), dtype=int)
        for c in range(n_vars):
            neighbors = list(np.flatnonzero(skeleton[c]))
            for i, a in enumerate(neighbors):
                for b in neighbors[i + 1:]:
                    # a and b are non-adjacent (tree skeleton); marginal
                    # independence identifies the head-to-head node c.
                    if _chi_square_conditional_independence(
                        data, a, b, [], cardinality, self.alpha_ci, weights
                    ):
                        directed[a, c] = 1
                        directed[b, c] = 1

        directed = _orient_remaining_no_collider(n_vars, skeleton, directed, permutation)
        return _finalize_adjacency(n_vars, skeleton, directed, mi, mp)


# ---------------------------------------------------------------------------
# LPA -- the polytree learner of PADA / FDA-SC
# ---------------------------------------------------------------------------


class PolytreeLPALearner:
    """Learn a polytree with the LPA algorithm used by PADA / FDA-SC.

    Follows the five steps of Ochoa, Muehlenbein & Soto (2000):

    0. Start from an empty graph and an empty edge list ``L``.
    1. Insert ``<a,b>`` into ``L`` when ``Dep(a,b) > e0``.
    2. Remove ``<a,b>`` from ``L`` when ``Dep(a,b|c) < e1`` for some ``c``
       (a third variable explains away the dependency).
    3. Score each surviving pair by the global dependency degree
       ``DepG(a,b) = min(Dep(a,b), min_c Dep(a,b|c))`` and rank ``L``
       by decreasing ``DepG``.
    4. Add edges in that order, up to ``n-1`` edges and skipping any edge that
       would close a cycle -- this keeps the skeleton singly connected.
    5. Orient: for every path ``a - c - b`` in the skeleton, instantiating a
       head-to-head node raises the dependency between its parents while
       instantiating any other middle node lowers it, so
       ``Dep(a,b|c) > Dep(a,b)`` implies ``a -> c <- b``.  Edges left
       unoriented are directed without introducing new head-to-head patterns.

    ``Dep`` is the Kullback-Leibler dependency measure, i.e. (conditional)
    mutual information, computed on the (optionally weighted) empirical
    distribution of the selected population.

    Parameters
    ----------
    alpha_ci : float
        Significance level used to derive ``e0`` and ``e1`` when they are not
        given explicitly.  Both thresholds then scale as ``1/N``, reproducing
        the population-size dependence described in the paper.
    e0, e1 : float or None
        Explicit marginal / conditional dependency thresholds.  ``None``
        derives them from ``alpha_ci`` and the sample size.
    dep_mode : {"global", "marginal"}
        ``"global"`` uses ``DepG`` (steps 2-3, cubic in the number of
        independence tests).  ``"marginal"`` ranks by ``Dep(a,b)`` alone and
        skips step 2, giving the quadratic variant of the algorithm; the
        orientation step is unchanged, so the result is still a polytree.
    n_cond_candidates : int or None
        Restrict the conditioning variables ``c`` of steps 2-3 to the
        ``n_cond_candidates`` variables most strongly dependent on ``a`` or
        ``b``.  This keeps LPA tractable on large problems.  ``None`` conditions
        on every variable (the literal cubic algorithm).
    max_parents : int or None
    """

    def __init__(
        self,
        alpha_ci: float = 0.05,
        e0: Optional[float] = None,
        e1: Optional[float] = None,
        dep_mode: str = "global",
        n_cond_candidates: Optional[int] = 5,
        max_parents: Optional[int] = None,
    ) -> None:
        if dep_mode not in ("global", "marginal"):
            raise ValueError("dep_mode must be 'global' or 'marginal'")
        self.alpha_ci = alpha_ci
        self.e0 = e0
        self.e1 = e1
        self.dep_mode = dep_mode
        self.n_cond_candidates = n_cond_candidates
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
        n_samples = data.shape[0]
        cardinality = np.asarray(cardinality, dtype=int)
        weights = _normalize_weights(n_samples, sample_weights)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        dep = _pairwise_mutual_information(data, cardinality, sample_weights)

        # Steps 1-3: build and rank the candidate list L.
        ranked, dep_global = self._ranked_list(
            data, n_vars, cardinality, weights, n_samples, dep, interaction_matrix
        )
        # Step 4: greedy insertion keeping the skeleton singly connected.
        skeleton = _spanning_forest(n_vars, ranked, interaction_matrix)
        # Step 5: orientation.
        directed = self._orient(data, n_vars, skeleton, cardinality, weights, dep)
        directed = _orient_remaining_no_collider(n_vars, skeleton, directed, permutation)
        return _finalize_adjacency(n_vars, skeleton, directed, dep_global, mp)

    # -- step 1-3 ----------------------------------------------------------

    def _ranked_list(
        self, data, n_vars, cardinality, weights, n_samples, dep, interaction_matrix
    ) -> Tuple[List[Tuple[float, int, int]], np.ndarray]:
        dep_global = dep.copy()
        edges: List[Tuple[float, int, int]] = []

        for u in range(n_vars):
            for v in range(u + 1, n_vars):
                if interaction_matrix is not None and interaction_matrix[u, v] == 0:
                    continue
                e0 = self.e0 if self.e0 is not None else _mi_threshold(
                    n_samples, cardinality, u, v, self.alpha_ci
                )
                # Step 1.2: keep only marginally dependent pairs.
                if dep[u, v] <= e0:
                    continue

                if self.dep_mode == "marginal":
                    edges.append((float(dep[u, v]), u, v))
                    continue

                # Steps 2 and 3 share the same sweep over conditioning nodes.
                worst = float(dep[u, v])
                dropped = False
                for c in self._cond_candidates(n_vars, u, v, dep):
                    e1 = self.e1 if self.e1 is not None else _mi_threshold(
                        n_samples, cardinality, u, v, self.alpha_ci,
                        n_cond_configs=int(cardinality[c]),
                    )
                    cmi = _conditional_mutual_information(
                        data, u, v, c, cardinality, weights
                    )
                    if cmi < e1:
                        # c explains away the dependency -- remove from L.
                        dropped = True
                        break
                    worst = min(worst, cmi)
                dep_global[u, v] = dep_global[v, u] = worst
                if not dropped:
                    edges.append((worst, u, v))

        edges.sort(key=lambda e: (-e[0], e[1], e[2]))
        return edges, dep_global

    def _cond_candidates(self, n_vars: int, u: int, v: int, dep: np.ndarray) -> List[int]:
        others = [c for c in range(n_vars) if c != u and c != v]
        if self.n_cond_candidates is None or len(others) <= self.n_cond_candidates:
            return others
        others.sort(key=lambda c: -max(dep[u, c], dep[v, c]))
        return others[: self.n_cond_candidates]

    # -- step 5 ------------------------------------------------------------

    def _orient(self, data, n_vars, skeleton, cardinality, weights, dep) -> np.ndarray:
        directed = np.zeros((n_vars, n_vars), dtype=int)
        for c in range(n_vars):
            neighbors = list(np.flatnonzero(skeleton[c]))
            for i, a in enumerate(neighbors):
                for b in neighbors[i + 1:]:
                    cmi = _conditional_mutual_information(
                        data, a, b, c, cardinality, weights
                    )
                    # Instantiating a head-to-head node increases the
                    # dependency between its parents; any other middle node
                    # decreases it.
                    if cmi > float(dep[a, b]):
                        directed[a, c] = 1
                        directed[b, c] = 1
        return directed


# ---------------------------------------------------------------------------
# Sheaf-based causal polytree recovery  (Huete & de Campos 1993)
# ---------------------------------------------------------------------------


class CausalPolytreeLearner:
    """Recover a causal polytree by incremental sheaf insertion.

    The *sheaf* of a node ``x``, written ``S(x)``, is the set of nodes directly
    connected to ``x`` in the partial structure ``T`` -- its direct causes and
    direct effects.  Starting from an empty ``T``, nodes are added one at a
    time; the position of a new node ``z`` relative to a node ``x`` is found by
    walking the partial structure:

    * If ``x ⟂ z | y`` for some ``y in S(x)`` then ``y`` lies on the path
      between ``x`` and ``z``: move to ``y`` and repeat (Theorem 4).
    * Otherwise ``z`` belongs in ``S(x)`` (Theorem 3).  Every ``y in S(x)``
      with ``x ⟂ y | z`` moves behind ``z``: the edge ``x - y`` is replaced by
      ``x - z`` and ``z - y``.  If no such ``y`` exists, ``x - z`` is simply
      added.

    Only marginal and first-order conditional independence tests are used, and
    the whole procedure takes O(n^2) tests.  Edges that would close a cycle are
    skipped so the skeleton stays singly connected under noisy tests.

    Orientation applies the same head-to-head criterion as Rebane-Pearl: two
    marginally independent variables that become dependent given a third are
    its parents.

    Parameters
    ----------
    alpha_ci : float
        Significance level for the chi-square independence tests.
    max_parents : int or None
    """

    def __init__(self, alpha_ci: float = 0.05, max_parents: Optional[int] = None) -> None:
        self.alpha_ci = alpha_ci
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
        n_samples = data.shape[0]
        cardinality = np.asarray(cardinality, dtype=int)
        weights = _normalize_weights(n_samples, sample_weights)
        mp = self.max_parents if self.max_parents is not None else _default_max_parents(cardinality)

        mi = _pairwise_mutual_information(data, cardinality, sample_weights)

        def indep(x: int, y: int, cond: List[int]) -> bool:
            return _chi_square_conditional_independence(
                data, x, y, cond, cardinality, self.alpha_ci, weights
            )

        # Step 1: the marginal dependency sets A_x.
        adjacent: Dict[int, Set[int]] = {v: set() for v in range(n_vars)}
        for u in range(n_vars):
            for v in range(u + 1, n_vars):
                if interaction_matrix is not None and interaction_matrix[u, v] == 0:
                    continue
                if not indep(u, v, []):
                    adjacent[u].add(v)
                    adjacent[v].add(u)

        skeleton = np.zeros((n_vars, n_vars), dtype=int)

        def sheaf(x: int) -> List[int]:
            return list(np.flatnonzero(skeleton[x]))

        def connected(a: int, b: int) -> bool:
            """True when a path already links *a* and *b* in the skeleton."""
            stack, seen = [a], {a}
            while stack:
                v = stack.pop()
                if v == b:
                    return True
                for u in np.flatnonzero(skeleton[v]):
                    u = int(u)
                    if u not in seen:
                        seen.add(u)
                        stack.append(u)
            return False

        def add_edge(a: int, b: int) -> bool:
            """Add a - b unless it would close a cycle.  False when refused.

            Union-find is not usable here: the sheaf procedure also *deletes*
            edges, so connectivity is recomputed from the skeleton itself.
            """
            if a == b or skeleton[a, b] == 1:
                return True
            # The sheaf procedure links pairs that were never in A_x, so the
            # allowed-interaction check has to be enforced here too.
            if interaction_matrix is not None and interaction_matrix[a, b] == 0:
                return False
            if connected(a, b):
                return False
            skeleton[a, b] = skeleton[b, a] = 1
            return True

        def drop_edge(a: int, b: int) -> None:
            skeleton[a, b] = skeleton[b, a] = 0

        def has_edge(a: int, b: int) -> bool:
            return bool(skeleton[a, b])

        # Steps 2-3: grow T one node at a time.
        in_tree = [False] * n_vars
        visited = [False] * n_vars
        expanded = [False] * n_vars

        for start in range(n_vars):
            if in_tree[start]:
                continue
            in_tree[start] = True
            expanded[start] = True

            pending = [start]
            while pending:
                x_root = pending.pop(0)
                if visited[x_root]:
                    continue
                visited[x_root] = True

                for z in sorted(adjacent[x_root]):
                    if expanded[z]:
                        continue
                    expanded[z] = True
                    in_tree[z] = True
                    self._insert_node(
                        x_root, z, sheaf, indep, add_edge, drop_edge, has_edge
                    )
                    pending.append(z)

        # Orientation: head-to-head detection, then constraint propagation.
        directed = np.zeros((n_vars, n_vars), dtype=int)
        for c in range(n_vars):
            neighbors = sheaf(c)
            for i, a in enumerate(neighbors):
                for b in neighbors[i + 1:]:
                    if indep(a, b, []) and not indep(a, b, [c]):
                        directed[a, c] = 1
                        directed[b, c] = 1

        directed = _orient_remaining_no_collider(n_vars, skeleton, directed, permutation)
        return _finalize_adjacency(n_vars, skeleton, directed, mi, mp)

    @staticmethod
    def _insert_node(x, z, sheaf, indep, add_edge, drop_edge, has_edge) -> None:
        """Place *z* in the partial structure, entering it through *x*."""
        # Walk towards z: y in S(x) with x ⟂ z | y lies on the x-z path.
        seen = {x}
        while True:
            moved = False
            for y in sheaf(x):
                if y in seen or y == z:
                    continue
                if indep(x, z, [y]):
                    seen.add(y)
                    x = y
                    moved = True
                    break
            if not moved:
                break

        # z joins S(x); any y in S(x) separated from x by z moves behind z.
        inserted = False
        for y in sheaf(x):
            if y == z:
                continue
            if not indep(x, y, [z]):
                continue
            had_xz = has_edge(x, z)
            drop_edge(x, y)
            if not (add_edge(x, z) and add_edge(z, y)):
                # Re-linking would close a cycle: restore the original edges.
                if not had_xz:
                    drop_edge(x, z)
                add_edge(x, y)
                continue
            inserted = True
        if not inserted:
            add_edge(x, z)
