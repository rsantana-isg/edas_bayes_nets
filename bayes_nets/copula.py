"""
Gaussian-copula modelling and sampling of discrete data.

This module implements the *extended rank likelihood* Gaussian copula for
discrete / ordinal data: the joint dependence between variables is captured
by a latent Gaussian correlation matrix, while the marginal distributions are
left unspecified.  Only the within-column ordering of the observations
constrains the latent Gaussian variables, so the dependence structure can be
estimated **without a model for the marginals** (Hoff 2007).  Kalaitzis &
Silva (2013) build on this construction to sample correlations of discrete
data efficiently.

The estimator here is the Gibbs sampler of the extended rank likelihood:

1.  **Extended rank mapping** — each column's observed values impose ordering
    constraints on a latent Gaussian column ``Z[:, j]`` (larger observed value
    ⇒ larger latent value).
2.  **Gibbs sampling** — alternate between drawing the latent ``Z`` from
    truncated normals that respect those constraints (given the current
    correlation), and drawing the correlation matrix from its
    inverse-Wishart full conditional given ``Z``.
3.  **Generation** — draw fresh latent Gaussian vectors from the fitted
    correlation and map them back to discrete values through the empirical
    marginals of the training data, yielding new discrete samples that
    reproduce both the learned dependence and the observed marginals.

This is offered as an alternative sampling backend to
:class:`bayes_nets.sampling.ProbabilisticLogicSampler` for settings where a
copula proposal over correlated discrete variables is preferable to an
explicit Bayesian-network factorisation (e.g. copula-style EDAs).

References
----------
Kalaitzis, A. & Silva, R. (2013). "Flexible sampling of discrete data
correlations without the marginal distributions." Advances in Neural
Information Processing Systems (NeurIPS) 26.

Hoff, P. D. (2007). "Extending the rank likelihood for semiparametric copula
estimation." The Annals of Applied Statistics 1(1), 265-283.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import invwishart


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(d, d)
    # numerical clean-up
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _truncated_normal(
    mean: np.ndarray,
    sd: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from ``N(mean, sd^2)`` truncated to ``(lower, upper)``.

    ``mean`` is a vector; ``lower``/``upper`` are shared scalar bounds.
    Uses the inverse-CDF method with numerical guards for extreme tails.
    """
    if sd <= 0:
        return np.clip(mean, lower, upper)
    a = (lower - mean) / sd if np.isfinite(lower) else np.full_like(mean, -np.inf)
    b = (upper - mean) / sd if np.isfinite(upper) else np.full_like(mean, np.inf)
    lo = ndtr(a)
    hi = ndtr(b)
    # Degenerate intervals: fall back to the clipped mean.
    degenerate = hi - lo < 1e-12
    u = lo + (hi - lo) * rng.random(mean.shape)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    z = mean + sd * ndtri(u)
    if np.any(degenerate):
        fallback = np.clip(
            mean,
            lower if np.isfinite(lower) else mean - sd,
            upper if np.isfinite(upper) else mean + sd,
        )
        z = np.where(degenerate, fallback, z)
    return z


class GaussianCopulaSampler:
    """Discrete Gaussian-copula model via the extended rank likelihood.

    Parameters
    ----------
    n_gibbs : int
        Total number of Gibbs sweeps.
    burn_in : int
        Number of initial sweeps discarded before averaging the correlation.
    prior_df : int or None
        Inverse-Wishart prior degrees of freedom ``ν0``.  ``None`` uses
        ``p + 2`` (a weakly-informative default whose prior mean is the
        identity correlation).
    jitter : float
        Diagonal jitter added before matrix inversions for stability.
    seed : int or None
        Seed for the internal random generator (reproducibility).

    Attributes
    ----------
    correlation_ : np.ndarray, shape (p, p)
        Posterior-mean latent correlation matrix (available after ``fit``).
    """

    def __init__(
        self,
        n_gibbs: int = 200,
        burn_in: int = 50,
        prior_df: Optional[int] = None,
        jitter: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        if burn_in >= n_gibbs:
            raise ValueError("burn_in must be smaller than n_gibbs")
        self.n_gibbs = int(n_gibbs)
        self.burn_in = int(burn_in)
        self.prior_df = prior_df
        self.jitter = float(jitter)
        self.seed = seed
        self.correlation_: Optional[np.ndarray] = None
        # Sorted training columns, kept for empirical-quantile generation.
        self._sorted_columns: List[np.ndarray] = []
        self._n_vars = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        cardinality: Optional[np.ndarray] = None,
    ) -> "GaussianCopulaSampler":
        """Estimate the latent correlation from discrete *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
            Integer-coded discrete/ordinal observations.
        cardinality : np.ndarray, optional
            Unused for fitting (the ordering is read directly from the data);
            accepted for API symmetry with the rest of the library.
        """
        data = np.asarray(data)
        n, p = data.shape
        rng = np.random.default_rng(self.seed)
        self._n_vars = p
        self._sorted_columns = [np.sort(data[:, j]) for j in range(p)]

        nu0 = self.prior_df if self.prior_df is not None else p + 2
        prior_scale = nu0 * np.eye(p)

        # Precompute, per column, the row groups for each observed level.
        level_rows: List[List[np.ndarray]] = []
        for j in range(p):
            levels = np.unique(data[:, j])
            level_rows.append([np.where(data[:, j] == lv)[0] for lv in levels])

        # Initialise Z from normal scores of the ranks (respects ordering).
        Z = np.empty((n, p), dtype=float)
        for j in range(p):
            ranks = np.argsort(np.argsort(data[:, j], kind="mergesort"))
            Z[:, j] = ndtri((ranks + 1.0) / (n + 1.0))

        corr = _cov_to_corr(np.cov(Z, rowvar=False) + self.jitter * np.eye(p))
        acc = np.zeros((p, p))
        n_acc = 0

        for sweep in range(self.n_gibbs):
            omega = np.linalg.inv(corr + self.jitter * np.eye(p))

            # --- sample latent Z | corr, ordering constraints -------------
            for j in range(p):
                omega_jj = omega[j, j]
                sd = float(np.sqrt(1.0 / omega_jj))
                # Full-conditional mean: -(Σ_{k≠j} Ω_jk Z_ik) / Ω_jj.
                mean = -(Z @ omega[:, j] - omega_jj * Z[:, j]) / omega_jj

                old_zj = Z[:, j].copy()
                rows_by_level = level_rows[j]
                # Upper bounds come from the minimum latent value of the
                # (still old) higher levels; lower bound is the running max
                # of the freshly sampled lower levels (guarantees ordering).
                level_min_old = np.array(
                    [old_zj[rows].min() for rows in rows_by_level]
                )
                n_levels = len(rows_by_level)
                suffix_min = np.full(n_levels, np.inf)
                for idx in range(n_levels - 2, -1, -1):
                    suffix_min[idx] = min(level_min_old[idx + 1], suffix_min[idx + 1])

                running_lower = -np.inf
                for idx, rows in enumerate(rows_by_level):
                    upper = suffix_min[idx]
                    lower = running_lower
                    samples = _truncated_normal(mean[rows], sd, lower, upper, rng)
                    Z[rows, j] = samples
                    running_lower = max(running_lower, float(samples.max()))

            # --- sample correlation | Z -----------------------------------
            scale = prior_scale + Z.T @ Z
            cov = invwishart.rvs(df=nu0 + n, scale=scale, random_state=rng)
            cov = np.atleast_2d(cov)
            corr = _cov_to_corr(cov + self.jitter * np.eye(p))

            if sweep >= self.burn_in:
                acc += corr
                n_acc += 1

        self.correlation_ = acc / max(n_acc, 1)
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw new discrete samples from the fitted copula.

        Latent Gaussian vectors are drawn from ``N(0, correlation_)`` and
        mapped back to discrete values through the empirical marginals of the
        training data, so both the learned dependence and the observed
        marginals are reproduced.
        """
        if self.correlation_ is None:
            raise RuntimeError("Call fit() before sample().")
        if rng is None:
            rng = np.random.default_rng()

        p = self._n_vars
        corr = self.correlation_
        latent = rng.multivariate_normal(np.zeros(p), corr, size=int(n_samples))

        out = np.empty((int(n_samples), p), dtype=int)
        for j in range(p):
            u = ndtr(latent[:, j])  # marginal is standard normal (unit variance)
            sorted_col = self._sorted_columns[j]
            m = len(sorted_col)
            idx = np.clip((u * m).astype(int), 0, m - 1)
            out[:, j] = sorted_col[idx]
        return out
