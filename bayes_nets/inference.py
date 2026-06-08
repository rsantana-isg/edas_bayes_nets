from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from bayes_nets.bayesian_network import BayesianNetwork


class MaxProductInference:
    """Exact MPC and marginals for discrete Bayesian networks."""

    def __init__(self, bn: BayesianNetwork):
        if not bn.cpds:
            raise RuntimeError("CPDs are required for inference. Call learn_parameters() or fit() first.")
        self.bn = bn

    def _normalise_evidence(self, evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]]) -> Dict[int, int]:
        if evidence is None:
            return {}
        if isinstance(evidence, dict):
            ev = {int(k): int(v) for k, v in evidence.items()}
        else:
            ev = {int(k): int(v) for k, v in evidence}

        for var, value in ev.items():
            if var < 0 or var >= self.bn.n_vars:
                raise ValueError(f"Evidence variable out of range: {var}")
            if value < 0 or value >= int(self.bn.cardinality[var]):
                raise ValueError(f"Evidence value out of range for variable {var}: {value}")
        return ev

    def _all_assignments(self) -> np.ndarray:
        n_vars = self.bn.n_vars
        cards = self.bn.cardinality.astype(int)
        n_states = int(np.prod(cards))

        assignments = np.zeros((n_states, n_vars), dtype=int)
        mults = np.ones(n_vars, dtype=int)
        for i in range(1, n_vars):
            mults[i] = mults[i - 1] * cards[i - 1]

        for idx in range(n_states):
            rem = idx
            for i in range(n_vars - 1, -1, -1):
                assignments[idx, i] = rem // mults[i]
                rem %= mults[i]

        return assignments

    def _joint_probabilities(self, evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
        ev = self._normalise_evidence(evidence)
        assignments = self._all_assignments()

        if ev:
            mask = np.ones(assignments.shape[0], dtype=bool)
            for var, value in ev.items():
                mask &= assignments[:, var] == value
            assignments = assignments[mask]

        probs = np.ones(assignments.shape[0], dtype=float)
        for var in range(self.bn.n_vars):
            info = self.bn.cpds[var]
            parents = [int(p) for p in info["parents"]]
            cpd = np.asarray(info["cpd"], dtype=float)

            if len(parents) == 0:
                probs *= cpd[assignments[:, var]]
                continue

            p_idx = np.zeros(assignments.shape[0], dtype=int)
            mult = 1
            for p in parents:
                p_idx += assignments[:, p] * mult
                mult *= int(self.bn.cardinality[p])
            probs *= cpd[p_idx, assignments[:, var]]

        return assignments, probs

    def most_probable_config(
        self,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, float]:
        assignments, probs = self._joint_probabilities(evidence)
        if len(assignments) == 0:
            raise ValueError("No assignments satisfy the provided evidence")

        idx = int(np.argmax(probs))
        return assignments[idx].copy(), float(probs[idx])

    def k_most_probable_configs(
        self,
        k: int,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if k < 1:
            raise ValueError("k must be >= 1")

        assignments, probs = self._joint_probabilities(evidence)
        if len(assignments) == 0:
            raise ValueError("No assignments satisfy the provided evidence")

        k = min(k, len(assignments))
        order = np.argsort(-probs)[:k]
        return assignments[order].copy(), probs[order].astype(float)

    def marginals(
        self,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
    ) -> list[np.ndarray]:
        assignments, probs = self._joint_probabilities(evidence)
        total = probs.sum()
        if total <= 0:
            raise ValueError("Joint probability under evidence is zero")
        probs = probs / total

        marginals: list[np.ndarray] = []
        for var in range(self.bn.n_vars):
            k = int(self.bn.cardinality[var])
            counts = np.zeros(k, dtype=float)
            for state in range(k):
                counts[state] = probs[assignments[:, var] == state].sum()
            marginals.append(counts)
        return marginals
