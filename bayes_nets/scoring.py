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
# Penalized K2  (Larrañaga et al. 2000; Etxeberria et al. 1997)
# ---------------------------------------------------------------------------


class K2PenScoringMethod(ScoringMethod):
    """Penalized-K2 score of Larrañaga et al. (2000), ``EBNA_K2+pen``.

    ``local_score = K2_local_score  −  f(N) · dim_local``

    where ``dim_local = n_parent_configs · (k − 1)`` is the number of free
    parameters of the local CPD and ``f(N)`` is the complexity-penalty weight:

    * ``penalty="bic"`` → ``f(N) = 0.5 · log(N)``  (Schwarz / BIC weight);
    * ``penalty="aic"`` → ``f(N) = 1``             (Akaike weight);
    * ``penalty=<float>`` → ``f(N) = that constant``.

    The marginal-likelihood term is the plain K2 (Cooper & Herskovits 1992)
    Dirichlet-multinomial score; subtracting the explicit dimension penalty
    turns the (score-equivalent-unbounded) K2 metric into the penalized metric
    that the original EBNA uses to depart from LFDA.  With ``f(N) = 0`` the
    score reduces to plain K2.

    Parameters
    ----------
    alpha : float
        Dirichlet prior equivalent sample size (K2 prior).
    sample_weights : array of float, shape (n_samples,), optional
        Probability distribution over rows (must sum to 1).
    penalty : {"bic", "aic"} or float
        Penalty weight ``f(N)`` (see above).

    References
    ----------
    Larrañaga, Etxeberria, Lozano & Peña (2000). "Combinatorial Optimization
    by Learning and Simulation of Bayesian Networks." UAI-2000.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        sample_weights: Optional[np.ndarray] = None,
        penalty="bic",
    ) -> None:
        self.alpha = alpha
        self.sample_weights = sample_weights
        self.penalty = penalty
        self._k2 = K2ScoringMethod(alpha=alpha, sample_weights=sample_weights)

    def with_weights(self, sample_weights: Optional[np.ndarray]) -> "K2PenScoringMethod":
        return K2PenScoringMethod(
            alpha=self.alpha, sample_weights=sample_weights, penalty=self.penalty
        )

    def f_penalty(self, n_samples: int) -> float:
        """Return the penalty weight ``f(N)`` for a dataset of ``n_samples`` rows."""
        pen = self.penalty
        if isinstance(pen, str):
            name = pen.lower()
            if name == "bic":
                return 0.5 * float(np.log(n_samples))
            if name == "aic":
                return 1.0
            raise ValueError(
                f"Unknown penalty '{pen}'. Use 'bic', 'aic', or a float constant."
            )
        return float(pen)

    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        base = self._k2.local_score(var, parents, data, cardinality)
        dim_local = _n_parameters(var, parents, cardinality)
        return base - self.f_penalty(data.shape[0]) * dim_local


def etxeberria_max_parents(
    cardinality: np.ndarray,
    n_samples: int,
    f_N: float,
) -> np.ndarray:
    """Per-variable automatic parent-count bound (Etxeberria et al. 1997, Th. 1).

    For the penalized metric ``log P(D|S) − f(N)·Σ_i (r_i−1) q_i`` the maximum
    number of parents of each variable is bounded automatically.  For variable
    ``X_i`` with ``r_i`` states and a database of ``N`` cases, write
    ``N = r_i·m + l`` with ``0 ≤ l < r_i``.  Equation (9) of Larrañaga et al.
    (2000) gives the threshold

    ``T_i = (1 / ((r_i−1) f(N))) · [ ln N! + ln (r_i+l−1)! − ln (N+r_i−1)!``
    ``       + m·( ln (2r_i−1)! − ln (r_i−1)! ) ]``

    and ``X_i`` will not have more than ``pa`` parents, where ``pa`` is the
    smallest count such that the product of the ``pa`` smallest *other* variable
    cardinalities exceeds ``T_i``.

    Worked example (paper §5): ``n=20``, cardinalities seventeen 3's and three
    4's, ``N=422``, AIC penalty ``f(N)=1`` ⇒ ``X_8`` (a 4-state variable) has
    bound ``5``.

    Parameters
    ----------
    cardinality : array-like of int, shape (n_vars,)
    n_samples : int
        Number of cases ``N`` in the database.
    f_N : float
        The penalty weight ``f(N)`` (e.g. ``1`` for AIC, ``0.5·log N`` for BIC).

    Returns
    -------
    np.ndarray of int, shape (n_vars,)
        Per-variable maximum parent count.

    References
    ----------
    Etxeberria, Larrañaga & Picaza (1997a). "Reducing Bayesian Networks
    Complexity while Learning from Data." Proc. Causal Models and Statistical
    Learning, 151-168 (Theorem 1).
    Larrañaga, Etxeberria, Lozano & Peña (2000). "Combinatorial Optimization
    by Learning and Simulation of Bayesian Networks." UAI-2000, eq. (9)
    (restates the theorem and gives the worked X_8 example).
    """
    cardinality = np.asarray(cardinality, dtype=int)
    n = cardinality.shape[0]
    N = int(n_samples)
    bounds = np.empty(n, dtype=int)
    for i in range(n):
        ri = int(cardinality[i])
        if ri < 2 or f_N <= 0 or N <= 0:
            bounds[i] = n - 1
            continue
        m = N // ri
        l = N % ri
        threshold = (
            gammaln(N + 1)
            + gammaln(ri + l)
            - gammaln(N + ri)
            + m * (gammaln(2 * ri) - gammaln(ri))
        ) / ((ri - 1) * f_N)
        others = np.sort(np.delete(cardinality, i))
        prod = 1.0
        bound = n - 1
        for pa in range(1, n):
            prod *= float(others[pa - 1])
            if prod > threshold:
                bound = pa
                break
        bounds[i] = bound
    return bounds


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
        max_leaves: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.sample_weights = sample_weights
        self.max_tree_depth = max_tree_depth
        self.max_leaves = max_leaves

    def with_weights(self, sample_weights: Optional[np.ndarray]) -> "DecisionTreeMDLScorer":
        return DecisionTreeMDLScorer(
            alpha=self.alpha,
            sample_weights=sample_weights,
            max_tree_depth=self.max_tree_depth,
            max_leaves=self.max_leaves,
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
        budget = {"count": 1}  # running leaf count for the max_leaves cap
        return self._tree_score(var, list(parents), idx, data, cardinality, w, n, k, alpha_k, 0, budget)

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
        budget: Optional[dict] = None,
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
            # Respect the leaf budget: a split on p turns this leaf into cp leaves.
            if (self.max_leaves is not None and budget is not None
                    and budget["count"] + (cp - 1) > self.max_leaves):
                continue
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
        if budget is not None:
            budget["count"] += cp - 1
        pv = p_vals_cache[best_p]
        total = 0.0
        for val in range(cp):
            child_idx = idx[pv == val]
            total += self._tree_score(
                var, remaining, child_idx, data, cardinality, weights, n_total, k, alpha_k, depth + 1, budget
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
        max_leaves: Optional[int] = None,
        split_score: str = "k2",
    ) -> None:
        self.alpha = alpha
        self.sample_weights = sample_weights
        self.max_tree_depth = max_tree_depth
        self.max_leaves = max_leaves
        split_score = (split_score or "k2").lower()
        if split_score not in ("k2", "mdl", "bic"):
            raise ValueError("split_score must be 'k2', 'mdl', or 'bic'")
        self.split_score = split_score

    def with_weights(self, sample_weights: Optional[np.ndarray]) -> "DecisionGraphBayesianScorer":
        return DecisionGraphBayesianScorer(
            alpha=self.alpha,
            sample_weights=sample_weights,
            max_tree_depth=self.max_tree_depth,
            max_leaves=self.max_leaves,
            split_score=self.split_score,
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
        budget = {"count": 1}  # running leaf count for the max_leaves cap
        leaves = self._build_leaves(var, list(parents), idx, data, cardinality, w, n, k, alpha_k, 0, budget)
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

    def _mdl_leaf(
        self,
        var: int,
        idx: np.ndarray,
        data: np.ndarray,
        k: int,
        alpha_k: float,
        weights: np.ndarray,
    ) -> float:
        """MDL/BIC (smoothed multinomial) log-likelihood at one leaf."""
        if len(idx) == 0:
            return 0.0
        counts = np.bincount(data[idx, var], weights=weights[idx], minlength=k).astype(float) + alpha_k
        total = counts.sum()
        nz = counts > 0
        return float(np.sum(counts[nz] * np.log(counts[nz] / total)))

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
        budget: Optional[dict] = None,
    ) -> List[np.ndarray]:
        """Grow a tree greedily using the split-score gain; return leaf index arrays.

        ``split_score='k2'`` grows with the K2 (Bayesian) gain; ``'mdl'``/``'bic'``
        grow with the cheaper BIC/MDL gain.  In every case the resulting leaves
        are passed through the K2 leaf-merge step (the defining DG operation).
        """
        if len(idx) == 0 or not parents or (
            self.max_tree_depth is not None and depth >= self.max_tree_depth
        ):
            return [idx]

        use_k2 = (self.split_score == "k2")
        if use_k2:
            base = self._k2_leaf(var, idx, data, k, alpha_k, weights)
        else:
            base = self._mdl_leaf(var, idx, data, k, alpha_k, weights)

        best_p = -1
        best_gain = 0.0

        for p in parents:
            cp = int(cardinality[p])
            if (self.max_leaves is not None and budget is not None
                    and budget["count"] + (cp - 1) > self.max_leaves):
                continue
            pv = data[idx, p]
            if use_k2:
                children_score = sum(
                    self._k2_leaf(var, idx[pv == val], data, k, alpha_k, weights)
                    for val in range(cp) if np.any(pv == val)
                )
                gain = children_score - base
            else:
                children_score = sum(
                    self._mdl_leaf(var, idx[pv == val], data, k, alpha_k, weights)
                    for val in range(cp) if np.any(pv == val)
                )
                extra = (cp - 1) * (k - 1)
                gain = children_score - base - extra * np.log(max(n_total, 1)) / 2
            if gain > best_gain:
                best_gain = gain
                best_p = p

        if best_p < 0:
            return [idx]

        if budget is not None:
            budget["count"] += int(cardinality[best_p]) - 1
        remaining = [q for q in parents if q != best_p]
        cp = int(cardinality[best_p])
        pv = data[idx, best_p]
        all_leaves: List[np.ndarray] = []
        for val in range(cp):
            child_idx = idx[pv == val]
            all_leaves.extend(
                self._build_leaves(var, remaining, child_idx, data, cardinality, weights, n_total, k, alpha_k, depth + 1, budget)
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


# ---------------------------------------------------------------------------
# Fast scorers with cached sufficient statistics  (Su & Zhang 2006)
# ---------------------------------------------------------------------------
#
# The exact DT/DG scorers rescan ``data[idx, p]`` for **every** candidate
# parent at **every** node and mask once per parent value.  The "fast" variants
# below replace that inner double loop by a single weighted 2-D histogram over
# ``(parent_value, var_value)`` per candidate parent — the cached sufficient
# statistics of Su & Zhang (2006, *A Fast Decision Tree Learning Algorithm*).
# All child leaf scores are then read off the histogram rows in one vectorised
# pass.  The numbers are *identical* to the exact scorers (same weighted counts,
# same per-leaf formula, same greedy choice and tie-breaking); only the
# bookkeeping is cheaper.  Empty child branches contribute 0, exactly as the
# exact scorers skip them.


def _dt_ll_rows(hist: np.ndarray, alpha_k: float) -> np.ndarray:
    """MDL/BIC leaf log-likelihood for every row of a ``(cp, k)`` count histogram.

    Rows whose count sum is zero (absent parent value) return 0, matching the
    exact scorer which skips empty children.
    """
    present = hist.sum(axis=1) > 0
    c = hist + alpha_k
    totals = c.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = c * np.log(c / totals)
    ll = contrib.sum(axis=1)
    ll[~present] = 0.0
    return ll


def _k2_ll_rows(hist: np.ndarray, alpha: float, alpha_k: float) -> np.ndarray:
    """K2 (Dirichlet-multinomial) leaf log-marginal for every histogram row.

    Rows whose count sum is zero return 0 (exact scorer skips empty children).
    """
    present = hist.sum(axis=1) > 0
    n_eff = hist.sum(axis=1)
    c = hist + alpha_k
    row = (gammaln(alpha) - gammaln(n_eff + alpha)
           + np.sum(gammaln(c) - gammaln(alpha_k), axis=1))
    row[~present] = 0.0
    return row


def _weighted_joint_hist(
    parent_vals: np.ndarray,
    var_vals: np.ndarray,
    weights: np.ndarray,
    cp: int,
    k: int,
) -> np.ndarray:
    """Weighted ``(cp, k)`` histogram of ``(parent_value, var_value)`` in one pass."""
    flat = parent_vals * k + var_vals
    return np.bincount(flat, weights=weights, minlength=cp * k).astype(float).reshape(cp, k)


class FastDecisionTreeMDLScorer(DecisionTreeMDLScorer):
    """Cached-statistics variant of :class:`DecisionTreeMDLScorer`.

    Produces the identical local score (asserted equal in the test suite) but
    scores candidate splits from a cached weighted joint histogram instead of
    per-value masking (Su & Zhang 2006).  Constructor and public API match the
    exact scorer, plus the inherited ``max_leaves`` bound.

    References
    ----------
    Su & Zhang (2006). "A Fast Decision Tree Learning Algorithm." AAAI-06.
    """

    def with_weights(self, sample_weights: Optional[np.ndarray]) -> "FastDecisionTreeMDLScorer":
        return FastDecisionTreeMDLScorer(
            alpha=self.alpha,
            sample_weights=sample_weights,
            max_tree_depth=self.max_tree_depth,
            max_leaves=self.max_leaves,
        )

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
        budget: Optional[dict] = None,
    ) -> float:
        leaf_penalty = (k - 1) * np.log(n_total) / 2
        if len(idx) == 0:
            return -leaf_penalty

        leaf_ll = self._leaf_ll(var, idx, data, k, alpha_k, weights)
        leaf_score = leaf_ll - leaf_penalty
        if not parents or (self.max_tree_depth is not None and depth >= self.max_tree_depth):
            return leaf_score

        var_vals = data[idx, var]
        w_idx = weights[idx]
        best_p = -1
        best_gain = 0.0
        for p in parents:
            cp = int(cardinality[p])
            if (self.max_leaves is not None and budget is not None
                    and budget["count"] + (cp - 1) > self.max_leaves):
                continue
            hist = _weighted_joint_hist(data[idx, p], var_vals, w_idx, cp, k)
            children_ll = float(_dt_ll_rows(hist, alpha_k).sum())
            extra_params = (cp - 1) * (k - 1)
            gain = children_ll - leaf_ll - extra_params * np.log(n_total) / 2
            if gain > best_gain:
                best_gain = gain
                best_p = p

        if best_p < 0:
            return leaf_score

        remaining = [q for q in parents if q != best_p]
        cp = int(cardinality[best_p])
        if budget is not None:
            budget["count"] += cp - 1
        pv = data[idx, best_p]
        total = 0.0
        for val in range(cp):
            child_idx = idx[pv == val]
            total += self._tree_score(
                var, remaining, child_idx, data, cardinality, weights, n_total, k, alpha_k, depth + 1, budget
            )
        return total


class FastDecisionGraphBayesianScorer(DecisionGraphBayesianScorer):
    """Cached-statistics variant of :class:`DecisionGraphBayesianScorer`.

    Produces the identical local score (asserted equal in the test suite) but
    scores candidate splits from cached weighted joint histograms (Su & Zhang
    2006).  Constructor and public API match the exact scorer, plus the
    inherited ``max_leaves`` and ``split_score`` options.  The K2 leaf-merge
    step is unchanged.

    References
    ----------
    Su & Zhang (2006). "A Fast Decision Tree Learning Algorithm." AAAI-06.
    Chickering, Heckerman & Meek (1997). "A Bayesian Approach to Learning
    Bayesian Networks with Local Structure." UAI-97.
    """

    def with_weights(self, sample_weights: Optional[np.ndarray]) -> "FastDecisionGraphBayesianScorer":
        return FastDecisionGraphBayesianScorer(
            alpha=self.alpha,
            sample_weights=sample_weights,
            max_tree_depth=self.max_tree_depth,
            max_leaves=self.max_leaves,
            split_score=self.split_score,
        )

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
        budget: Optional[dict] = None,
    ) -> List[np.ndarray]:
        if len(idx) == 0 or not parents or (
            self.max_tree_depth is not None and depth >= self.max_tree_depth
        ):
            return [idx]

        use_k2 = (self.split_score == "k2")
        if use_k2:
            base = self._k2_leaf(var, idx, data, k, alpha_k, weights)
        else:
            base = self._mdl_leaf(var, idx, data, k, alpha_k, weights)

        var_vals = data[idx, var]
        w_idx = weights[idx]
        best_p = -1
        best_gain = 0.0
        for p in parents:
            cp = int(cardinality[p])
            if (self.max_leaves is not None and budget is not None
                    and budget["count"] + (cp - 1) > self.max_leaves):
                continue
            hist = _weighted_joint_hist(data[idx, p], var_vals, w_idx, cp, k)
            if use_k2:
                children_score = float(_k2_ll_rows(hist, self.alpha, alpha_k).sum())
                gain = children_score - base
            else:
                children_score = float(_dt_ll_rows(hist, alpha_k).sum())
                extra = (cp - 1) * (k - 1)
                gain = children_score - base - extra * np.log(max(n_total, 1)) / 2
            if gain > best_gain:
                best_gain = gain
                best_p = p

        if best_p < 0:
            return [idx]

        if budget is not None:
            budget["count"] += int(cardinality[best_p]) - 1
        remaining = [q for q in parents if q != best_p]
        cp = int(cardinality[best_p])
        pv = data[idx, best_p]
        all_leaves: List[np.ndarray] = []
        for val in range(cp):
            child_idx = idx[pv == val]
            all_leaves.extend(
                self._build_leaves(var, remaining, child_idx, data, cardinality, weights, n_total, k, alpha_k, depth + 1, budget)
            )
        return self._merge_leaves(var, all_leaves, data, k, alpha_k, weights)
