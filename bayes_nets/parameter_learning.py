"""
Parameter learning for Bayesian networks.

MLEParameterLearner
    Estimates CPDs from data using maximum-likelihood estimation with
    optional Dirichlet (Laplace) smoothing.

When a ``sample_weights`` vector is provided, counts for each configuration
are computed as ``N · Σ_i p_i · I[row i matches config]``, then Laplace
smoothing is applied on top of those weighted counts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class MLEParameterLearner:
    """Estimate CPDs via MLE with Laplace / Dirichlet smoothing.

    For each variable X_i with parent set Pa_i the CPD is estimated as:

        P(X_i = v | Pa_i = pa) = (N̂_{i,pa,v} + α/r_i) / (N̂_{i,pa} + α)

    where:
      - N̂_{i,pa,v}  is the (possibly weighted) count of rows with
                     Pa_i = pa and X_i = v
      - r_i         is the cardinality of X_i
      - α           is the smoothing parameter (0 → pure MLE)

    When ``sample_weights`` p is provided:
        N̂_{i,pa,v} = N · Σ_l p_l · I[Pa_l = pa, X_l = v]

    Parameters
    ----------
    alpha : float
        Dirichlet equivalent sample size.  Set to 0 for pure MLE.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        adjacency: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict]:
        """Estimate all CPDs from *data* given a DAG *adjacency*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        n_vars : int
        cardinality : np.ndarray, shape (n_vars,)
        adjacency : np.ndarray, shape (n_vars, n_vars)
        sample_weights : array of float, shape (n_samples,), optional
            Probability distribution over rows (must sum to 1).

        Returns
        -------
        dict
            Maps ``var`` → ``{"parents": [...], "cpd": np.ndarray}``.
        """
        cpds: Dict[int, Dict] = {}
        for var in range(n_vars):
            parents = list(np.where(adjacency[:, var] > 0)[0])
            cpd = self._estimate_cpd(var, parents, data, cardinality, sample_weights)
            cpds[var] = {"parents": parents, "cpd": cpd}
        return cpds

    def estimate_cpd(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate the CPD for a single variable.

        Parameters
        ----------
        var : int
        parents : list of int
        data : np.ndarray, shape (n_samples, n_vars)
        cardinality : np.ndarray, shape (n_vars,)
        sample_weights : array of float, shape (n_samples,), optional

        Returns
        -------
        np.ndarray
            1-D array of shape ``(cardinality[var],)`` when *parents* is
            empty, otherwise 2-D array of shape
            ``(n_parent_configs, cardinality[var])``.
        """
        return self._estimate_cpd(var, parents, data, cardinality, sample_weights)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _estimate_cpd(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        k = int(cardinality[var])
        alpha_k = self.alpha / k
        n_samples = data.shape[0]

        # Effective per-row weights
        w: Optional[np.ndarray]
        if sample_weights is None:
            w = None
        else:
            w = np.asarray(sample_weights, dtype=float) * n_samples

        if not parents:
            counts = np.bincount(data[:, var], weights=w, minlength=k).astype(float) + alpha_k
            return counts / counts.sum()

        parent_card = [int(cardinality[p]) for p in parents]
        n_parent_configs = int(np.prod(parent_card))

        # Parent configuration index for every row
        configs = np.zeros(n_samples, dtype=int)
        mult = 1
        for j, p in enumerate(parents):
            configs += data[:, p] * mult
            mult *= parent_card[j]

        cpd = np.zeros((n_parent_configs, k))
        for pc in range(n_parent_configs):
            mask = configs == pc
            w_pc = w[mask] if w is not None else None
            counts = np.bincount(data[mask, var], weights=w_pc, minlength=k).astype(float) + alpha_k
            total = counts.sum()
            cpd[pc, :] = counts / total if total > 0 else 1.0 / k

        return cpd
