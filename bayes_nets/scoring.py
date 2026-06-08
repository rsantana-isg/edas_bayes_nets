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
