"""
Scoring metrics for Bayesian network structure learning.

All metrics evaluate how well a candidate DAG structure fits a dataset.
Higher scores are better.

Available metrics
-----------------
BICScoringMethod
    Bayesian Information Criterion.
    BIC = log P(D | θ_ML, G) - (k / 2) · log(n)

AICScoringMethod
    Akaike Information Criterion.
    AIC = log P(D | θ_ML, G) - k

K2ScoringMethod
    Bayesian (Dirichlet) scoring metric used by the K2 algorithm.
    Score = log P(D | G) with Dirichlet prior.

DecisionTreeMDLScorer
    MDL (BIC-based) scoring where each CPD is compressed into a decision
    tree over parent variables (Friedman & Goldszmidt 1996).  The tree
    is grown top-down using a BIC gain criterion: a split on parent p is
    accepted when the likelihood improvement exceeds the BIC cost of the
    extra leaf parameters.  Effective parameter count = n_leaves×(k-1)
    instead of n_parent_configs×(k-1), allowing richer parent sets.

DecisionGraphBayesianScorer
    Bayesian (K2-based) scoring where each CPD is represented as a
    decision graph (Chickering, Heckerman & Meek 1997).  Decision graphs
    generalise decision trees by allowing leaf merging (parameter sharing).
    After greedy tree growing, pairs of leaves whose data is pooled by the
    K2 score are merged, yielding sparser representations.

All scoring methods accept an optional ``sample_weights`` vector (a
probability distribution over data rows).  When provided, counts for each
configuration are computed as ``N · Σ_i p_i · I[row i matches config]``,
and Laplace smoothing is then applied on top of those weighted counts.
If ``sample_weights`` is None, uniform weights 1/N are assumed (standard
MLE from counts).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from scipy.special import gammaln  # type: ignore


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ScoringMethod(ABC):
    """Abstract base class for BN scoring metrics."""

    @abstractmethod
    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        """Return the local score for *var* given its *parents*.

        Parameters
        ----------
        var : int
            Variable index.
        parents : list of int
            Indices of parent variables.
        data : np.ndarray, shape (n_samples, n_vars)
            Observed data.
        cardinality : np.ndarray, shape (n_vars,)
            Number of discrete states for each variable.

        Returns
        -------
        float
            Local score (higher is better).
        """

    def score(
        self,
        adjacency: np.ndarray,
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        """Return the total score for a complete DAG (sum of local scores)."""
        n_vars = adjacency.shape[0]
        return sum(
            self.local_score(var, list(np.where(adjacency[:, var] > 0)[0]), data, cardinality)
            for var in range(n_vars)
        )

    def with_weights(self, sample_weights: Optional[np.ndarray]) -> "ScoringMethod":
        """Return a copy of this scorer with new sample weights."""
        return type(self)(alpha=self.alpha, sample_weights=sample_weights)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Internal helpers (shared by BIC / AIC)
# ---------------------------------------------------------------------------


def _effective_weights(n_samples: int, sample_weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Scale a probability vector to effective counts (multiply by N).

    Returns None for uniform weights (allows np.bincount to use faster
    unweighted path).
    """
    if sample_weights is None:
        return None
    return np.asarray(sample_weights, dtype=float) * n_samples


def _log_likelihood(
    var: int,
    parents: List[int],
    data: np.ndarray,
    cardinality: np.ndarray,
    alpha: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
) -> float:
    """Compute log P(D_var | pa(var)) under maximum-likelihood + smoothing.

    Parameters
    ----------
    alpha : float
        Laplace / Dirichlet smoothing added to every cell count.
    sample_weights : array of float, shape (n_samples,), optional
        Probability vector (sums to 1).  Weighted counts replace raw
        counts: ``ñ_c = N · Σ_i p_i · I[row i matches config c]``.
    """
    k = int(cardinality[var])
    n_samples = data.shape[0]
    w = _effective_weights(n_samples, sample_weights)

    if not parents:
        counts = np.bincount(data[:, var], weights=w, minlength=k).astype(float) + alpha
        total = counts.sum()
        nz = counts > 0
        return float(np.sum(counts[nz] * np.log(counts[nz] / total)))

    parent_card = [int(cardinality[p]) for p in parents]
    n_parent_configs = int(np.prod(parent_card))
    parent_configs = _parent_config_indices(data, parents, parent_card, n_samples)

    ll = 0.0
    for pc in range(n_parent_configs):
        mask = parent_configs == pc
        w_pc = w[mask] if w is not None else None
        n_pc = float(w_pc.sum()) if w_pc is not None else float(mask.sum())
        if n_pc == 0 and alpha == 0.0:
            continue
        counts = np.bincount(data[mask, var], weights=w_pc, minlength=k).astype(float) + alpha
        total = counts.sum()
        nz = counts > 0
        ll += float(np.sum(counts[nz] * np.log(counts[nz] / total)))

    return ll


def _parent_config_indices(
    data: np.ndarray,
    parents: List[int],
    parent_card: List[int],
    n_samples: int,
) -> np.ndarray:
    """Encode parent configurations as integer indices (row-major, first parent varies fastest)."""
    configs = np.zeros(n_samples, dtype=int)
    mult = 1
    for j, p in enumerate(parents):
        configs += data[:, p] * mult
        mult *= parent_card[j]
    return configs


def _n_parameters(var: int, parents: List[int], cardinality: np.ndarray) -> int:
    """Return the number of free parameters in the CPD of *var*."""
    k = int(cardinality[var])
    parent_card = [int(cardinality[p]) for p in parents]
    n_parent_configs = int(np.prod(parent_card)) if parent_card else 1
    return n_parent_configs * (k - 1)


# ---------------------------------------------------------------------------
# BIC
# ---------------------------------------------------------------------------


class BICScoringMethod(ScoringMethod):
    """Bayesian Information Criterion (BIC).

    BIC = log P(D | θ_ML, G) - (k / 2) · log(n)

    where *k* is the number of free parameters and *n* is the number of
    data rows (regardless of sample_weights).

    Parameters
    ----------
    alpha : float
        Laplace smoothing applied when estimating the log-likelihood.
    sample_weights : array of float, shape (n_samples,), optional
        Probability distribution over rows (must sum to 1).
    """

    def __init__(self, alpha: float = 0.0, sample_weights: Optional[np.ndarray] = None) -> None:
        self.alpha = alpha
        self.sample_weights = sample_weights

    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        n = data.shape[0]
        ll = _log_likelihood(var, parents, data, cardinality, self.alpha, self.sample_weights)
        k = _n_parameters(var, parents, cardinality)
        return ll - 0.5 * k * np.log(n)


# ---------------------------------------------------------------------------
# AIC
# ---------------------------------------------------------------------------


class AICScoringMethod(ScoringMethod):
    """Akaike Information Criterion (AIC).

    AIC = log P(D | θ_ML, G) - k

    Parameters
    ----------
    alpha : float
        Laplace smoothing applied when estimating the log-likelihood.
    sample_weights : array of float, shape (n_samples,), optional
        Probability distribution over rows (must sum to 1).
    """

    def __init__(self, alpha: float = 0.0, sample_weights: Optional[np.ndarray] = None) -> None:
        self.alpha = alpha
        self.sample_weights = sample_weights

    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        ll = _log_likelihood(var, parents, data, cardinality, self.alpha, self.sample_weights)
        k = _n_parameters(var, parents, cardinality)
        return ll - k


# ---------------------------------------------------------------------------
# K2
# ---------------------------------------------------------------------------


class K2ScoringMethod(ScoringMethod):
    """K2 Bayesian scoring metric.

    Computes the log marginal likelihood under a symmetric Dirichlet prior:

        score(X_i, Pa_i) = Σ_{j=1}^{q_i} [
            ln Γ(α) - ln Γ(N̂_{ij} + α)
            + Σ_{k=1}^{r_i} ln Γ(N̂_{ijk} + α/r_i) - ln Γ(α/r_i)
        ]

    where N̂ denotes weighted counts when ``sample_weights`` is provided:
    N̂_{ijk} = N · Σ_l p_l · I[pa_l = j, X_l = k].

    Parameters
    ----------
    alpha : float
        Prior equivalent sample size (default 1.0).
    sample_weights : array of float, shape (n_samples,), optional
        Probability distribution over rows (must sum to 1).
    """

    def __init__(self, alpha: float = 1.0, sample_weights: Optional[np.ndarray] = None) -> None:
        self.alpha = alpha
        self.sample_weights = sample_weights

    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        k = int(cardinality[var])
        n_samples = data.shape[0]
        alpha = self.alpha
        alpha_k = alpha / k
        w = _effective_weights(n_samples, self.sample_weights)

        if not parents:
            counts = np.bincount(data[:, var], weights=w, minlength=k).astype(float)
            n_total = counts.sum()
            score = gammaln(alpha) - gammaln(n_total + alpha)
            score += float(np.sum(gammaln(counts + alpha_k) - gammaln(alpha_k)))
            return float(score)

        parent_card = [int(cardinality[p]) for p in parents]
        n_parent_configs = int(np.prod(parent_card))
        parent_configs = _parent_config_indices(data, parents, parent_card, n_samples)

        score = 0.0
        for pc in range(n_parent_configs):
            mask = parent_configs == pc
            w_pc = w[mask] if w is not None else None
            n_pc = float(w_pc.sum()) if w_pc is not None else float(mask.sum())
            counts = np.bincount(data[mask, var], weights=w_pc, minlength=k).astype(float)
            score += gammaln(alpha) - gammaln(n_pc + alpha)
            score += float(np.sum(gammaln(counts + alpha_k) - gammaln(alpha_k)))

        return score


# ---------------------------------------------------------------------------
# Decision-tree MDL scorer  (Friedman & Goldszmidt 1996)
# ---------------------------------------------------------------------------


class DecisionTreeMDLScorer(ScoringMethod):
    """MDL (BIC-based) local score using decision-tree CPT representations.

    A decision tree is grown greedily top-down over the parent variables.
    At each node a split is accepted when the BIC gain is positive:

        gain(split on p) = ΔLL_children  –  (card_p – 1)·(k – 1)·log(N)/2

    Effective parameters = n_leaves·(k–1) ≤ n_parent_configs·(k–1), so
    the score rewards local independencies that are invisible to tabular BIC.

    Parameters
    ----------
    alpha : float
        Laplace smoothing (same role as in BICScoringMethod).
    sample_weights : array of float, shape (n_samples,), optional
    max_tree_depth : int or None
        Maximum depth of the decision tree.  None → unbounded (stops when
        no split improves BIC).

    References
    ----------
    Friedman & Goldszmidt (1996). "Learning Bayesian Networks with Local
    Structure." UAI-96.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        sample_weights: Optional[np.ndarray] = None,
        max_tree_depth: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.sample_weights = sample_weights
        self.max_tree_depth = max_tree_depth

    def with_weights(self, sample_weights: Optional[np.ndarray]) -> "DecisionTreeMDLScorer":
        return DecisionTreeMDLScorer(
            alpha=self.alpha,
            sample_weights=sample_weights,
            max_tree_depth=self.max_tree_depth,
        )

    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        n = data.shape[0]
        w = _effective_weights(n, self.sample_weights)
        if w is None:
            w = np.ones(n, dtype=float)
        k = int(cardinality[var])
        alpha_k = self.alpha / k
        idx = np.arange(n)
        return self._tree_score(var, list(parents), idx, data, cardinality, w, n, k, alpha_k, 0)

    # ------------------------------------------------------------------
    # Internal: greedy top-down tree building
    # ------------------------------------------------------------------

    def _leaf_ll(
        self,
        var: int,
        idx: np.ndarray,
        data: np.ndarray,
        k: int,
        alpha_k: float,
        weights: np.ndarray,
    ) -> float:
        if len(idx) == 0:
            return 0.0
        counts = np.bincount(data[idx, var], weights=weights[idx], minlength=k).astype(float) + alpha_k
        total = counts.sum()
        nz = counts > 0
        return float(np.sum(counts[nz] * np.log(counts[nz] / total)))

    def _tree_score(
        self,
        var: int,
        parents: List[int],
        idx: np.ndarray,
        data: np.ndarray,
        cardinality: np.ndarray,
        weights: np.ndarray,
        n_total: int,
        k: int,
        alpha_k: float,
        depth: int,
    ) -> float:
        """Score for the subtree rooted at this node (greedy CART-BIC)."""
        leaf_penalty = (k - 1) * np.log(n_total) / 2

        if len(idx) == 0:
            return -leaf_penalty  # empty branch: just the leaf penalty

        leaf_ll = self._leaf_ll(var, idx, data, k, alpha_k, weights)
        leaf_score = leaf_ll - leaf_penalty

        if not parents or (self.max_tree_depth is not None and depth >= self.max_tree_depth):
            return leaf_score

        # Evaluate immediate gain from each candidate split
        best_p = -1
        best_gain = 0.0
        p_vals_cache: dict = {}

        for p in parents:
            cp = int(cardinality[p])
            pv = data[idx, p]
            p_vals_cache[p] = pv
            children_ll = 0.0
            for val in range(cp):
                cmask = pv == val
                if not np.any(cmask):
                    continue
                children_ll += self._leaf_ll(var, idx[cmask], data, k, alpha_k, weights)
            extra_params = (cp - 1) * (k - 1)
            gain = children_ll - leaf_ll - extra_params * np.log(n_total) / 2
            if gain > best_gain:
                best_gain = gain
                best_p = p

        if best_p < 0:
            return leaf_score

        # Accept split on best_p; recurse on each child
        remaining = [q for q in parents if q != best_p]
        cp = int(cardinality[best_p])
        pv = p_vals_cache[best_p]
        total = 0.0
        for val in range(cp):
            child_idx = idx[pv == val]
            total += self._tree_score(
                var, remaining, child_idx, data, cardinality, weights, n_total, k, alpha_k, depth + 1
            )
        return total


# ---------------------------------------------------------------------------
# Decision-graph Bayesian scorer  (Chickering, Heckerman & Meek 1997)
# ---------------------------------------------------------------------------


class DecisionGraphBayesianScorer(ScoringMethod):
    """Bayesian (K2-based) local score using decision-graph CPT representations.

    Decision graphs generalise decision trees by allowing parameter sharing:
    after growing a greedy tree (using K2 gain at each split), pairs of
    leaves are merged whenever the pooled K2 score exceeds the sum of the
    individual leaf scores.  The merge step represents the "equal parameters"
    constraints that define a decision graph.

    Parameters
    ----------
    alpha : float
        Dirichlet prior equivalent sample size (K2 prior).
    sample_weights : array of float, shape (n_samples,), optional
    max_tree_depth : int or None

    References
    ----------
    Chickering, Heckerman & Meek (1997). "A Bayesian Approach to Learning
    Bayesian Networks with Local Structure." UAI-97.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        sample_weights: Optional[np.ndarray] = None,
        max_tree_depth: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.sample_weights = sample_weights
        self.max_tree_depth = max_tree_depth

    def with_weights(self, sample_weights: Optional[np.ndarray]) -> "DecisionGraphBayesianScorer":
        return DecisionGraphBayesianScorer(
            alpha=self.alpha,
            sample_weights=sample_weights,
            max_tree_depth=self.max_tree_depth,
        )

    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        n = data.shape[0]
        w = _effective_weights(n, self.sample_weights)
        if w is None:
            w = np.ones(n, dtype=float)
        k = int(cardinality[var])
        alpha_k = self.alpha / k
        idx = np.arange(n)
        leaves = self._build_leaves(var, list(parents), idx, data, cardinality, w, n, k, alpha_k, 0)
        return sum(self._k2_leaf(var, leaf_idx, data, k, alpha_k, w) for leaf_idx in leaves)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _k2_leaf(
        self,
        var: int,
        idx: np.ndarray,
        data: np.ndarray,
        k: int,
        alpha_k: float,
        weights: np.ndarray,
    ) -> float:
        """K2 (Dirichlet-Multinomial) log-marginal at one leaf."""
        if len(idx) == 0:
            return 0.0
        counts = np.bincount(data[idx, var], weights=weights[idx], minlength=k).astype(float)
        n_eff = float(counts.sum())
        score = float(gammaln(self.alpha) - gammaln(n_eff + self.alpha))
        score += float(np.sum(gammaln(counts + alpha_k) - gammaln(alpha_k)))
        return score

    def _build_leaves(
        self,
        var: int,
        parents: List[int],
        idx: np.ndarray,
        data: np.ndarray,
        cardinality: np.ndarray,
        weights: np.ndarray,
        n_total: int,
        k: int,
        alpha_k: float,
        depth: int,
    ) -> List[np.ndarray]:
        """Grow a tree greedily using K2 gain; return list of leaf index arrays."""
        if len(idx) == 0 or not parents or (
            self.max_tree_depth is not None and depth >= self.max_tree_depth
        ):
            return [idx]

        base = self._k2_leaf(var, idx, data, k, alpha_k, weights)

        best_p = -1
        best_gain = 0.0

        for p in parents:
            cp = int(cardinality[p])
            pv = data[idx, p]
            children_score = sum(
                self._k2_leaf(var, idx[pv == val], data, k, alpha_k, weights)
                for val in range(cp)
                if np.any(pv == val)
            )
            gain = children_score - base
            if gain > best_gain:
                best_gain = gain
                best_p = p

        if best_p < 0:
            return [idx]

        remaining = [q for q in parents if q != best_p]
        cp = int(cardinality[best_p])
        pv = data[idx, best_p]
        all_leaves: List[np.ndarray] = []
        for val in range(cp):
            child_idx = idx[pv == val]
            all_leaves.extend(
                self._build_leaves(var, remaining, child_idx, data, cardinality, weights, n_total, k, alpha_k, depth + 1)
            )

        # Merge step: pool any pair of leaves whose merged K2 score is higher
        return self._merge_leaves(var, all_leaves, data, k, alpha_k, weights)

    def _merge_leaves(
        self,
        var: int,
        leaves: List[np.ndarray],
        data: np.ndarray,
        k: int,
        alpha_k: float,
        weights: np.ndarray,
    ) -> List[np.ndarray]:
        """Greedily merge leaf pairs that improve the total K2 score."""
        changed = True
        while changed and len(leaves) > 1:
            changed = False
            for i in range(len(leaves)):
                for j in range(i + 1, len(leaves)):
                    merged = np.concatenate([leaves[i], leaves[j]])
                    gain = (
                        self._k2_leaf(var, merged, data, k, alpha_k, weights)
                        - self._k2_leaf(var, leaves[i], data, k, alpha_k, weights)
                        - self._k2_leaf(var, leaves[j], data, k, alpha_k, weights)
                    )
                    if gain > 0.0:
                        leaves = [leaves[l] for l in range(len(leaves)) if l not in (i, j)] + [merged]
                        changed = True
                        break
                if changed:
                    break
        return leaves
