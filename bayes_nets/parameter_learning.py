"""
Parameter learning for Bayesian networks.

MLEParameterLearner
    Estimates CPDs from data using maximum-likelihood estimation with
    optional Dirichlet (Laplace) smoothing.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class MLEParameterLearner:
    """Estimate CPDs via MLE with Laplace / Dirichlet smoothing.

    For each variable X_i with parent set Pa_i the CPD is estimated as:

        P(X_i = v | Pa_i = pa) = (N_{i,pa,v} + α/r_i) / (N_{i,pa} + α)

    where:
      - N_{i,pa,v}  is the count of samples with Pa_i = pa and X_i = v
      - N_{i,pa}    is the count of samples with Pa_i = pa
      - r_i         is the cardinality of X_i
      - α           is the smoothing parameter (0 → pure MLE)

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
    ) -> Dict[int, Dict]:
        """Estimate all CPDs from *data* given a DAG *adjacency*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        n_vars : int
        cardinality : np.ndarray, shape (n_vars,)
        adjacency : np.ndarray, shape (n_vars, n_vars)

        Returns
        -------
        dict
            Maps ``var`` → ``{"parents": [...], "cpd": np.ndarray}``.
        """
        cpds: Dict[int, Dict] = {}
        for var in range(n_vars):
            parents = list(np.where(adjacency[:, var] > 0)[0])
            cpd = self._estimate_cpd(var, parents, data, cardinality)
            cpds[var] = {"parents": parents, "cpd": cpd}
        return cpds

    def estimate_cpd(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """Estimate the CPD for a single variable.

        Parameters
        ----------
        var : int
        parents : list of int
        data : np.ndarray, shape (n_samples, n_vars)
        cardinality : np.ndarray, shape (n_vars,)

        Returns
        -------
        np.ndarray
            1-D array of shape ``(cardinality[var],)`` when *parents* is
            empty, otherwise 2-D array of shape
            ``(n_parent_configs, cardinality[var])``.
        """
        return self._estimate_cpd(var, parents, data, cardinality)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _estimate_cpd(
        self,
        var: int,
        parents: List[int],
        data: np.ndarray,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        k = int(cardinality[var])
        alpha_k = self.alpha / k  # Per-state pseudo-count

        if not parents:
            counts = np.bincount(data[:, var], minlength=k).astype(float) + alpha_k
            return counts / counts.sum()

        parent_card = [int(cardinality[p]) for p in parents]
        n_parent_configs = int(np.prod(parent_card))
        n_samples = data.shape[0]

        # Parent configuration index for every sample
        configs = np.zeros(n_samples, dtype=int)
        mult = 1
        for j, p in enumerate(parents):
            configs += data[:, p] * mult
            mult *= parent_card[j]

        cpd = np.zeros((n_parent_configs, k))
        for pc in range(n_parent_configs):
            mask = configs == pc
            counts = np.bincount(data[mask, var], minlength=k).astype(float) + alpha_k
            total = counts.sum()
            if total > 0:
                cpd[pc, :] = counts / total
            else:
                cpd[pc, :] = 1.0 / k  # Uniform fallback

        return cpd
