"""
Local-structure CPDs: decision-tree and decision-graph representations.

A tabular conditional probability distribution (CPD) stores one distribution
per parent configuration, i.e. ``prod(cardinality[parents])`` rows.  When the
distribution of ``X_i`` depends on its parents only through *context-specific*
patterns, most of those rows are identical and the table is wasteful.  A
**decision tree** captures this by splitting on parent variables and storing a
distribution only at the leaves; a **decision graph** goes further and lets
several leaves *share* the same distribution (parameter tying).

This module provides:

* :class:`LocalStructureCPD` -- a compact, reusable representation of a single
  local CPD that can (i) evaluate ``P(X_i | pa_i)`` for any parent
  configuration, (ii) expand to the equivalent dense table, (iii) **sample**
  ``X_i`` directly by routing a parent configuration to its leaf, and (iv)
  report its (compact) parameter count;
* :func:`learn_local_cpd` -- greedy builder of a decision tree (BIC/MDL gain,
  Friedman & Goldszmidt 1996) or a decision graph (K2 gain plus leaf merging,
  Chickering, Heckerman & Meek 1997) from (weighted) data;
* :class:`LocalStructureParameterLearner` -- a drop-in parameter learner that
  fits a :class:`LocalStructureCPD` per variable of a given DAG, so that any
  base skeleton learner can be *composed* with decision-tree / decision-graph
  CPDs (see :meth:`bayes_nets.BayesianNetwork.learn_local_structure`).

References
----------
Friedman & Goldszmidt (1996). "Learning Bayesian Networks with Local
Structure." UAI-96.
Chickering, Heckerman & Meek (1997). "A Bayesian Approach to Learning Bayesian
Networks with Local Structure." UAI-97.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


class _Node:
    """A node of a decision tree / graph over the parents of a variable.

    Internal node: ``split`` is the position (in the ``parents`` list) of the
    variable tested, and ``children[v]`` is the sub-node for value ``v`` of that
    variable.  Leaf node: ``split == -1`` and ``dist`` indexes the distribution
    in :attr:`LocalStructureCPD.leaf_probs`.
    """

    __slots__ = ("split", "children", "dist")

    def __init__(self, split: int = -1, children: Optional[List["_Node"]] = None,
                 dist: int = -1) -> None:
        self.split = split
        self.children = children
        self.dist = dist

    @property
    def is_leaf(self) -> bool:
        return self.split < 0


# ---------------------------------------------------------------------------
# Compact CPD
# ---------------------------------------------------------------------------


class LocalStructureCPD:
    """Compact decision-tree / decision-graph CPD for one variable.

    Parameters
    ----------
    var : int
        Index of the child variable this CPD models.
    parents : list of int
        Parent variable indices (order matters: ``route`` expects the parent
        columns in this order).
    var_card : int
        Number of states of ``var``.
    parent_card : list of int
        Number of states of each parent (same order as ``parents``).
    root : _Node
        Root of the routing tree.
    leaf_probs : np.ndarray, shape (n_distinct_leaves, var_card)
        The distinct leaf distributions; a leaf node's ``dist`` indexes a row.
    kind : {"dt", "dg"}
        Whether the CPD is a decision tree or a (leaf-merged) decision graph.
    """

    def __init__(self, var: int, parents: List[int], var_card: int,
                 parent_card: List[int], root: _Node, leaf_probs: np.ndarray,
                 kind: str) -> None:
        self.var = int(var)
        self.parents = list(parents)
        self.var_card = int(var_card)
        self.parent_card = [int(c) for c in parent_card]
        self.root = root
        self.leaf_probs = np.asarray(leaf_probs, dtype=float)
        self.kind = kind

    # ------------------------------------------------------------------
    # Sizes
    # ------------------------------------------------------------------

    @property
    def n_distinct_leaves(self) -> int:
        """Number of distinct leaf distributions (parameter blocks)."""
        return int(self.leaf_probs.shape[0])

    def n_leaves(self) -> int:
        """Number of tree leaves (>= n_distinct_leaves for a decision graph)."""
        count = [0]

        def walk(node: _Node) -> None:
            if node.is_leaf:
                count[0] += 1
            else:
                for ch in node.children:
                    walk(ch)

        walk(self.root)
        return count[0]

    @property
    def n_parameters(self) -> int:
        """Free parameters: ``n_distinct_leaves * (var_card - 1)``."""
        return self.n_distinct_leaves * (self.var_card - 1)

    def full_table_parameters(self) -> int:
        """Free parameters of the equivalent dense table (for comparison)."""
        n_cfg = int(np.prod(self.parent_card)) if self.parent_card else 1
        return n_cfg * (self.var_card - 1)

    # ------------------------------------------------------------------
    # Routing / evaluation
    # ------------------------------------------------------------------

    def route(self, parent_values: np.ndarray) -> np.ndarray:
        """Return the leaf-distribution index for every row.

        Parameters
        ----------
        parent_values : np.ndarray, shape (m, n_parents)
            Parent values (columns in the order of :attr:`parents`).  If
            ``n_parents == 0`` an ``(m, 0)`` array (or any m-length array) is
            accepted and every row maps to the single leaf.

        Returns
        -------
        np.ndarray, shape (m,) of int
        """
        pv = np.asarray(parent_values)
        if pv.ndim == 1:
            pv = pv.reshape(-1, 1) if self.parents else pv.reshape(-1, 0)
        m = pv.shape[0]
        out = np.empty(m, dtype=int)

        def descend(node: _Node, rows: np.ndarray) -> None:
            if node.is_leaf:
                out[rows] = node.dist
                return
            col = pv[rows, node.split]
            for v, child in enumerate(node.children):
                sel = rows[col == v]
                if sel.size:
                    descend(child, sel)

        descend(self.root, np.arange(m))
        return out

    def prob_matrix(self, parent_values: np.ndarray) -> np.ndarray:
        """Return ``P(X_i = k | pa)`` for every row: shape ``(m, var_card)``."""
        return self.leaf_probs[self.route(parent_values)]

    def to_table(self) -> np.ndarray:
        """Expand to the equivalent dense CPD table.

        Rows are ordered with the library convention (first parent varies
        fastest): ``config = sum_j val_j * prod(card_0..card_{j-1})``.  For a
        parent-less variable a 1-D probability vector is returned.
        """
        if not self.parents:
            return self.leaf_probs[self.root.dist].copy()
        n_cfg = int(np.prod(self.parent_card))
        # enumerate configurations in first-parent-fastest order
        configs = np.empty((n_cfg, len(self.parents)), dtype=int)
        mult = 1
        for j, c in enumerate(self.parent_card):
            reps = np.tile(np.repeat(np.arange(c), mult), n_cfg // (c * mult))
            configs[:, j] = reps
            mult *= c
        return self.prob_matrix(configs)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_rows(self, parent_values: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
        """Sample ``X_i`` for every row given its parent values.

        Routes each row to its leaf and draws from the leaf distribution.  This
        never materialises the dense table, so it stays cheap even when the
        parent set is large.
        """
        m = (np.asarray(parent_values).shape[0]
             if self.parents else int(parent_values))
        if not self.parents:
            probs = self.leaf_probs[self.root.dist]
            return rng.choice(self.var_card, size=m, p=probs)
        leaf_ids = self.route(parent_values)
        out = np.empty(m, dtype=int)
        u = rng.random(m)
        # group rows by leaf so each unique distribution is used once
        for lid in np.unique(leaf_ids):
            sel = np.where(leaf_ids == lid)[0]
            cdf = np.cumsum(self.leaf_probs[lid])
            cdf[-1] = 1.0
            out[sel] = np.searchsorted(cdf, u[sel], side="right")
        return np.clip(out, 0, self.var_card - 1)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"LocalStructureCPD(var={self.var}, parents={self.parents}, "
                f"kind='{self.kind}', leaves={self.n_leaves()}, "
                f"distinct={self.n_distinct_leaves}, params={self.n_parameters})")


# ---------------------------------------------------------------------------
# Leaf statistics (weighted)
# ---------------------------------------------------------------------------


def _weighted_counts(var_col: np.ndarray, idx: np.ndarray, k: int,
                     w: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        return np.zeros(k, dtype=float)
    return np.bincount(var_col[idx], weights=w[idx], minlength=k).astype(float)


def _leaf_distribution(counts: np.ndarray, alpha: float) -> np.ndarray:
    k = counts.shape[0]
    smoothed = counts + alpha / k
    total = smoothed.sum()
    return smoothed / total if total > 0 else np.full(k, 1.0 / k)


def _leaf_ll(counts: np.ndarray, alpha_k: float) -> float:
    """MDL/BIC leaf log-likelihood (smoothed multinomial)."""
    c = counts + alpha_k
    total = c.sum()
    if total <= 0:
        return 0.0
    nz = c > 0
    return float(np.sum(c[nz] * np.log(c[nz] / total)))


def _k2_leaf(counts: np.ndarray, alpha: float, alpha_k: float) -> float:
    """K2 (Dirichlet-Multinomial) log-marginal likelihood at a leaf."""
    n_eff = float(counts.sum())
    score = float(gammaln(alpha) - gammaln(n_eff + alpha))
    score += float(np.sum(gammaln(counts + alpha_k) - gammaln(alpha_k)))
    return score


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def learn_local_cpd(
    var: int,
    parents: List[int],
    data: np.ndarray,
    cardinality: np.ndarray,
    method: str = "dt",
    alpha: float = 1.0,
    sample_weights: Optional[np.ndarray] = None,
    max_depth: Optional[int] = None,
) -> LocalStructureCPD:
    """Learn a decision-tree (``method='dt'``) or decision-graph
    (``method='dg'``) CPD for *var* given its *parents*.

    The tree is grown greedily: a split on the parent with the best gain is
    accepted while the gain is positive.  For ``'dt'`` the gain is the BIC/MDL
    gain; for ``'dg'`` it is the K2 gain, after which pairs of leaves are merged
    whenever pooling their statistics improves the total K2 score (parameter
    tying).  ``sample_weights`` is a probability vector over the rows (as
    elsewhere in the library); it is turned into effective counts.
    """
    if method not in ("dt", "dg"):
        raise ValueError("method must be 'dt' (decision tree) or 'dg' (decision graph)")
    if method == "dg" and alpha <= 0:
        # The decision-graph gains and leaf merging use the K2
        # (Dirichlet-Multinomial) marginal likelihood, which is undefined for a
        # zero Dirichlet prior.  Fail clearly instead of degenerating silently.
        raise ValueError("decision-graph CPDs ('dg') require alpha > 0 "
                         "(the K2 score is undefined at alpha=0); use 'dt' for "
                         "an MDL tree with no smoothing.")
    data = np.asarray(data, dtype=int)
    n = data.shape[0]
    k = int(cardinality[var])
    parents = list(parents)
    parent_card = [int(cardinality[p]) for p in parents]
    var_col = data[:, var]
    parent_cols = data[:, parents] if parents else np.empty((n, 0), dtype=int)

    if sample_weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(sample_weights, dtype=float) * n
    n_eff = float(w.sum())
    alpha_k = alpha / k
    use_k2 = (method == "dg")

    # collected leaves: list of (node, counts, idx)
    leaves: List[Tuple[_Node, np.ndarray, np.ndarray]] = []

    def counts_of(idx: np.ndarray) -> np.ndarray:
        return _weighted_counts(var_col, idx, k, w)

    def build(idx: np.ndarray, avail: List[int], depth: int) -> _Node:
        node_counts = counts_of(idx)
        if (len(avail) == 0 or idx.size == 0
                or (max_depth is not None and depth >= max_depth)):
            leaf = _Node()
            leaves.append((leaf, node_counts, idx))
            return leaf

        if use_k2:
            base = _k2_leaf(node_counts, alpha, alpha_k)
        else:
            base = _leaf_ll(node_counts, alpha_k)

        best_gain, best_pos = 0.0, -1
        for pos in avail:
            cp = parent_card[pos]
            pv = parent_cols[idx, pos]
            child_score = 0.0
            for v in range(cp):
                cidx = idx[pv == v]
                cc = counts_of(cidx)
                if use_k2:
                    child_score += _k2_leaf(cc, alpha, alpha_k)
                else:
                    child_score += _leaf_ll(cc, alpha_k)
            if use_k2:
                gain = child_score - base
            else:
                # BIC/MDL penalty for the extra leaves introduced by the split
                extra = (cp - 1) * (k - 1)
                gain = child_score - base - extra * np.log(max(n_eff, 1.0)) / 2.0
            if gain > best_gain:
                best_gain, best_pos = gain, pos

        if best_pos < 0:
            leaf = _Node()
            leaves.append((leaf, node_counts, idx))
            return leaf

        cp = parent_card[best_pos]
        pv = parent_cols[idx, best_pos]
        remaining = [q for q in avail if q != best_pos]
        children = [build(idx[pv == v], remaining, depth + 1) for v in range(cp)]
        return _Node(split=best_pos, children=children)

    root = build(np.arange(n), list(range(len(parents))), 0)

    # ---- assign leaf distributions --------------------------------------
    if method == "dg" and len(leaves) > 1:
        groups = _merge_leaves([lc for _, lc, _ in leaves], alpha, alpha_k)
    else:
        groups = [[i] for i in range(len(leaves))]

    leaf_probs = np.zeros((len(groups), k), dtype=float)
    for g_id, members in enumerate(groups):
        pooled = np.zeros(k, dtype=float)
        for m_i in members:
            pooled += leaves[m_i][1]
            leaves[m_i][0].dist = g_id
        leaf_probs[g_id] = _leaf_distribution(pooled, alpha)

    return LocalStructureCPD(var, parents, k, parent_card, root, leaf_probs, method)


def _merge_leaves(leaf_counts: List[np.ndarray], alpha: float,
                  alpha_k: float) -> List[List[int]]:
    """Greedily merge leaf groups while the pooled K2 score improves.

    Returns a partition of ``range(len(leaf_counts))`` (list of index groups).
    """
    groups: List[List[int]] = [[i] for i in range(len(leaf_counts))]
    counts: List[np.ndarray] = [c.copy() for c in leaf_counts]
    changed = True
    while changed and len(groups) > 1:
        changed = False
        best = None
        for i in range(len(groups)):
            si = _k2_leaf(counts[i], alpha, alpha_k)
            for j in range(i + 1, len(groups)):
                merged = counts[i] + counts[j]
                gain = (_k2_leaf(merged, alpha, alpha_k)
                        - si - _k2_leaf(counts[j], alpha, alpha_k))
                if gain > 0.0 and (best is None or gain > best[0]):
                    best = (gain, i, j, merged)
        if best is not None:
            _, i, j, merged = best
            groups[i] = groups[i] + groups[j]
            counts[i] = merged
            del groups[j]
            del counts[j]
            changed = True
    return groups


# ---------------------------------------------------------------------------
# Parameter learner (composes any DAG with DT / DG local CPDs)
# ---------------------------------------------------------------------------


class LocalStructureParameterLearner:
    """Fit decision-tree / decision-graph CPDs for every variable of a DAG.

    Drop-in replacement for :class:`bayes_nets.MLEParameterLearner` in
    :meth:`bayes_nets.BayesianNetwork.learn_parameters`: its :meth:`learn`
    returns the usual ``{var: {"parents", "cpd"}}`` dictionary, augmented with a
    ``"local"`` entry holding the compact :class:`LocalStructureCPD`.  The dense
    ``"cpd"`` table is materialised only when it has at most
    ``max_table_configs`` rows (so the representation stays compact for
    high-in-degree nodes, where it is most useful); otherwise ``"cpd"`` is
    ``None`` and downstream code must use the ``"local"`` CPD (the
    local-structure sampler does).

    Parameters
    ----------
    method : {"dt", "dg"}
        Decision tree or decision graph.
    alpha : float
        Laplace / Dirichlet smoothing.
    max_depth : int or None
        Maximum tree depth (``None`` -> until no split improves the score).
    max_table_configs : int
        Upper bound on the dense table size to materialise (default ``2**16``).
    """

    def __init__(self, method: str = "dt", alpha: float = 1.0,
                 max_depth: Optional[int] = None,
                 max_table_configs: int = 1 << 16) -> None:
        if method not in ("dt", "dg"):
            raise ValueError("method must be 'dt' (decision tree) or 'dg' (decision graph)")
        self.method = method
        self.alpha = alpha
        self.max_depth = max_depth
        self.max_table_configs = max_table_configs

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        adjacency: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict]:
        data = np.asarray(data, dtype=int)
        cardinality = np.asarray(cardinality, dtype=int)
        cpds: Dict[int, Dict] = {}
        for var in range(n_vars):
            parents = list(np.where(adjacency[:, var] > 0)[0])
            local = learn_local_cpd(
                var, parents, data, cardinality, method=self.method,
                alpha=self.alpha, sample_weights=sample_weights,
                max_depth=self.max_depth,
            )
            n_cfg = int(np.prod([int(cardinality[p]) for p in parents])) if parents else 1
            table = local.to_table() if n_cfg <= self.max_table_configs else None
            cpds[var] = {"parents": parents, "cpd": table, "local": local}
        return cpds
