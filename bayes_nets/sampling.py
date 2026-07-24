"""
Sampling methods for Bayesian networks.

ProbabilisticLogicSampler
    Generates samples from a BN using probabilistic logic sampling
    (a.k.a. forward / ancestral sampling).  Variables are sampled in
    topological order, conditioning on already-sampled parent values.

LocalStructureSampler
    Ancestral sampler that draws each variable directly from its
    decision-tree / decision-graph CPD by *routing* the parent configuration
    to the corresponding leaf, without materialising the dense CPD table.  This
    is what lets the compact context-specific structure be exploited at
    sampling time, not only during structure selection.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class ProbabilisticLogicSampler:
    """Generate samples via probabilistic logic sampling (ancestral sampling).

    This is the standard forward-sampling method for Bayesian networks:

    1. Topologically sort the variables.
    2. For each variable in order, sample from P(X_i | Pa_i = pa_i)
       where pa_i are the already-sampled values of the parents.

    All variables are discrete; the CPD for a root variable is a 1-D
    probability vector and for a non-root variable it is a 2-D array
    indexed by the parent configuration.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        n_samples: int,
        n_vars: int,
        cardinality: np.ndarray,
        adjacency: np.ndarray,
        cpds: Dict[int, Dict],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw *n_samples* samples from the BN.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        n_vars : int
            Number of variables.
        cardinality : np.ndarray, shape (n_vars,)
            Cardinality of each variable.
        adjacency : np.ndarray, shape (n_vars, n_vars)
            Adjacency matrix of the DAG.
        cpds : dict
            Maps variable index → ``{"parents": [...], "cpd": array}``.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray, shape (n_samples, n_vars)
            Sampled discrete observations.
        """
        if rng is None:
            rng = np.random.default_rng()

        order = self._topological_sort(adjacency, n_vars)
        population = np.zeros((n_samples, n_vars), dtype=int)

        for var in order:
            var_info = cpds[var]
            parents: List[int] = var_info["parents"]
            cpd: np.ndarray = var_info["cpd"]
            k = int(cardinality[var])

            if not parents:
                # Root variable: sample from marginal probability vector
                probs = self._ensure_valid_probs(cpd)
                population[:, var] = rng.choice(k, size=n_samples, p=probs)
            else:
                # Non-root: sample conditionally on parent values
                parent_card = [int(cardinality[p]) for p in parents]
                mult = 1
                configs = np.zeros(n_samples, dtype=int)
                for j, p in enumerate(parents):
                    configs += population[:, p] * mult
                    mult *= parent_card[j]

                # Vectorised sampling via cumulative sum trick
                for i in range(n_samples):
                    probs = self._ensure_valid_probs(cpd[configs[i], :])
                    population[i, var] = rng.choice(k, p=probs)

        return population

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _topological_sort(adjacency: np.ndarray, n_vars: int) -> np.ndarray:
        """Kahn's algorithm for topological order."""
        in_degree = adjacency.sum(axis=0).copy()
        queue = list(np.where(in_degree == 0)[0])
        order: List[int] = []
        while queue:
            var = queue.pop(0)
            order.append(var)
            for child in range(n_vars):
                if adjacency[var, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        if len(order) != n_vars:
            raise ValueError("BN adjacency matrix contains a cycle – not a valid DAG")
        return np.array(order, dtype=int)

    @staticmethod
    def _ensure_valid_probs(probs: np.ndarray) -> np.ndarray:
        """Normalise *probs* and guard against numerical noise."""
        p = np.asarray(probs, dtype=float)
        p = np.clip(p, 0.0, None)
        total = p.sum()
        if total == 0.0:
            return np.ones_like(p) / len(p)
        return p / total


class LocalStructureSampler:
    """Ancestral sampler for BNs whose CPDs use decision-tree / -graph structure.

    Each ``cpds[var]`` must carry a ``"local"`` entry holding a
    :class:`bayes_nets.local_structure.LocalStructureCPD`.  Variables are
    sampled in topological order; for each one the parent columns already
    sampled are routed through its decision tree/graph to obtain, per row, the
    leaf distribution to draw from.  The dense conditional table is never built,
    so sampling stays cheap even for high-in-degree nodes.
    """

    def sample(
        self,
        n_samples: int,
        n_vars: int,
        cardinality: np.ndarray,
        adjacency: np.ndarray,
        cpds: Dict[int, Dict],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        order = ProbabilisticLogicSampler._topological_sort(adjacency, n_vars)
        population = np.zeros((n_samples, n_vars), dtype=int)

        for var in order:
            info = cpds[var]
            local = info.get("local")
            if local is None:
                raise ValueError(
                    f"Variable {var} has no local-structure CPD; use "
                    "ProbabilisticLogicSampler for dense tabular CPDs."
                )
            parents: List[int] = info["parents"]
            if not parents:
                population[:, var] = local.sample_rows(n_samples, rng)
            else:
                parent_values = population[:, parents]
                population[:, var] = local.sample_rows(parent_values, rng)

        return population
