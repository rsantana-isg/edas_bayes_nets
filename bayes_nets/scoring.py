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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

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
        """Return the total score for a complete DAG.

        Computed as the sum of local scores.
        """
        n_vars = adjacency.shape[0]
        total = 0.0
        for var in range(n_vars):
            parents = list(np.where(adjacency[:, var] > 0)[0])
            total += self.local_score(var, parents, data, cardinality)
        return total


# ---------------------------------------------------------------------------
# Likelihood helpers (shared by BIC / AIC)
# ---------------------------------------------------------------------------


def _log_likelihood(
    var: int,
    parents: List[int],
    data: np.ndarray,
    cardinality: np.ndarray,
    alpha: float = 0.0,
) -> float:
    """Compute log P(D_var | pa(var)) under maximum-likelihood (+ smoothing).

    Parameters
    ----------
    alpha : float
        Laplace / Dirichlet smoothing parameter.
    """
    k = int(cardinality[var])
    n_samples = data.shape[0]

    if not parents:
        counts = np.bincount(data[:, var], minlength=k).astype(float) + alpha
        total = counts.sum()
        ll = np.sum(counts * np.log(counts / total))
        return float(ll)

    parent_card = [int(cardinality[p]) for p in parents]
    n_parent_configs = int(np.prod(parent_card))

    # Parent configuration index for every sample
    parent_configs = _parent_config_indices(data, parents, parent_card, n_samples)

    ll = 0.0
    for pc in range(n_parent_configs):
        mask = parent_configs == pc
        n_pc = mask.sum()
        if n_pc == 0 and alpha == 0.0:
            continue
        counts = np.bincount(data[mask, var], minlength=k).astype(float) + alpha
        total = counts.sum()
        ll += np.sum(counts * np.log(counts / total))

    return float(ll)


def _parent_config_indices(
    data: np.ndarray,
    parents: List[int],
    parent_card: List[int],
    n_samples: int,
) -> np.ndarray:
    """Encode parent configurations as integer indices (row-major)."""
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

    where *k* is the number of free parameters and *n* is the sample size.

    Parameters
    ----------
    alpha : float
        Laplace smoothing applied when estimating the log-likelihood.
    """

    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        n = data.shape[0]
        ll = _log_likelihood(var, parents, data, cardinality, self.alpha)
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
    """

    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def local_score(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        ll = _log_likelihood(var, parents, data, cardinality, self.alpha)
        k = _n_parameters(var, parents, cardinality)
        return ll - k


# ---------------------------------------------------------------------------
# K2
# ---------------------------------------------------------------------------


class K2ScoringMethod(ScoringMethod):
    """K2 Bayesian scoring metric.

    Computes the log marginal likelihood under a symmetric Dirichlet prior:

        score(X_i, Pa_i) = Σ_{j=1}^{q_i} [
            ln Γ(α) - ln Γ(N_{ij} + α)
            + Σ_{k=1}^{r_i} ln Γ(N_{ijk} + α/r_i) - ln Γ(α/r_i)
        ]

    where:
      - q_i = number of parent configurations
      - r_i = cardinality of X_i
      - N_{ij} = number of samples with parent configuration j
      - N_{ijk} = number of samples with parent config j and X_i = k
      - α = equivalent sample size (prior strength)

    Parameters
    ----------
    alpha : float
        Prior equivalent sample size (default 1.0).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

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
        alpha_k = alpha / k  # Per-state pseudo-count

        if not parents:
            counts = np.bincount(data[:, var], minlength=k).astype(float)
            score = gammaln(alpha) - gammaln(n_samples + alpha)
            score += np.sum(gammaln(counts + alpha_k) - gammaln(alpha_k))
            return float(score)

        parent_card = [int(cardinality[p]) for p in parents]
        n_parent_configs = int(np.prod(parent_card))
        parent_configs = _parent_config_indices(data, parents, parent_card, n_samples)

        score = 0.0
        for pc in range(n_parent_configs):
            mask = parent_configs == pc
            n_pc = int(mask.sum())
            counts = np.bincount(data[mask, var], minlength=k).astype(float)
            score += gammaln(alpha) - gammaln(n_pc + alpha)
            score += float(np.sum(gammaln(counts + alpha_k) - gammaln(alpha_k)))

        return score
