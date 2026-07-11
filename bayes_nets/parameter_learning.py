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

from itertools import combinations, product
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


class LogisticRegressionParameterLearner:
    """Estimate CPDs with (regularised) logistic regression + artificial features.

    Instead of storing a full conditional probability table (whose size grows
    exponentially with the number of parents), each node's conditional
    distribution ``P(X | Pa)`` is fit with an L1-regularised multinomial
    logistic regression over features derived from the parents:

    * **dummy** features -- one indicator per non-baseline parent state
      (the ``T_d`` transformation), whose count grows *linearly* with the
      number of parents; and
    * optional **XOR** interaction features between pairs of binary parents
      (a Rijmen-style transformation) that let the model capture higher-order
      dependencies without a full table.

    L1 regularisation (via scikit-learn's SAGA solver) implicitly selects the
    relevant features, mitigating over-fitting in dense networks.  The learner
    still returns a standard dense CPD table (same format as
    :class:`MLEParameterLearner`) so it is a drop-in replacement: the compact
    logistic model is evaluated at every parent configuration to fill the
    table.  The benefit is estimation quality / far fewer effective
    parameters, not table storage.

    Parameters
    ----------
    C : float
        Inverse L1 regularisation strength (smaller -> sparser).
    use_xor : bool
        Add pairwise XOR interaction features between binary parents.
    alpha : float
        Laplace smoothing used for the parent-less marginal and as a floor
        that keeps every probability strictly positive.
    max_iter : int
        Maximum SAGA iterations.

    References
    ----------
    Moral, Moral-García, Cano et al. (2026). "Computing conditional
    probabilities in Bayesian networks using logistic regression."
    Applied Soft Computing 198.
    """

    def __init__(
        self,
        C: float = 1.0,
        use_xor: bool = True,
        alpha: float = 1.0,
        max_iter: int = 200,
    ) -> None:
        self.C = C
        self.use_xor = use_xor
        self.alpha = alpha
        self.max_iter = max_iter
        self._mle = MLEParameterLearner(alpha=alpha)

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        adjacency: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict]:
        """Estimate all CPDs from *data* given a DAG *adjacency*."""
        cpds: Dict[int, Dict] = {}
        for var in range(n_vars):
            parents = list(np.where(adjacency[:, var] > 0)[0])
            cpd = self.estimate_cpd(var, parents, data, cardinality, sample_weights)
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
        """Estimate the CPD for a single variable via logistic regression."""
        k = int(cardinality[var])
        if not parents:
            # no parents -> marginal; logistic regression adds nothing
            return self._mle.estimate_cpd(var, parents, data, cardinality, sample_weights)

        parent_card = [int(cardinality[p]) for p in parents]
        n_parent_configs = int(np.prod(parent_card))

        y = data[:, var].astype(int)
        classes_present = np.unique(y)
        if classes_present.size < 2:
            # degenerate: variable constant given data -> fall back to MLE
            return self._mle.estimate_cpd(var, parents, data, cardinality, sample_weights)

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:  # pragma: no cover - sklearn is a declared dep
            return self._mle.estimate_cpd(var, parents, data, cardinality, sample_weights)

        parent_vals = data[:, parents].astype(int)
        X = self._transform(parent_vals, parent_card)

        w = None
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=float) * data.shape[0]

        model = self._make_logreg(LogisticRegression)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y, sample_weight=w)

        # Evaluate the fitted model at every parent configuration.
        configs = np.array(list(product(*[range(c) for c in parent_card])), dtype=int)
        # `product` iterates the last parent fastest; the library indexes
        # configs as sum_j val_j * prod(card_0..card_{j-1}) (first parent
        # fastest).  Reorder rows to match that convention.
        Xc = self._transform(configs, parent_card)
        proba = model.predict_proba(Xc)          # (n_configs, n_classes_present)

        cpd = np.full((n_parent_configs, k), self.alpha / k, dtype=float)
        for col, cls in enumerate(model.classes_):
            cpd[:, cls] += proba[:, col]
        cpd /= cpd.sum(axis=1, keepdims=True)

        # Map `product` row order -> library config index.
        row_index = np.zeros(n_parent_configs, dtype=int)
        for r, cfg in enumerate(configs):
            idx, mult = 0, 1
            for j, val in enumerate(cfg):
                idx += int(val) * mult
                mult *= parent_card[j]
            row_index[r] = idx
        ordered = np.empty_like(cpd)
        ordered[row_index] = cpd
        return ordered

    def _make_logreg(self, LogisticRegression):
        """Build an L1 logistic model, tolerant of scikit-learn API changes.

        Pure L1 is requested via ``penalty='l1'`` on older scikit-learn and
        via the ``elasticnet`` / ``l1_ratio=1.0`` route on newer versions
        where ``penalty`` is deprecated.
        """
        common = dict(solver="saga", C=self.C, max_iter=self.max_iter, tol=1e-3)
        try:
            return LogisticRegression(penalty="elasticnet", l1_ratio=1.0, **common)
        except TypeError:  # pragma: no cover - very old sklearn
            return LogisticRegression(penalty="l1", **common)

    # ------------------------------------------------------------------
    # Feature transformation
    # ------------------------------------------------------------------

    def _transform(self, parent_vals: np.ndarray, parent_card: List[int]) -> np.ndarray:
        """Map integer parent configs to logistic-regression features.

        Dummy (one-hot, drop-first) encoding of each parent, plus optional
        pairwise XOR interactions between binary parents.
        """
        m, n_parents = parent_vals.shape
        cols: List[np.ndarray] = []
        for j in range(n_parents):
            c = parent_card[j]
            for s in range(1, c):                       # drop baseline state 0
                cols.append((parent_vals[:, j] == s).astype(float))

        if self.use_xor:
            binary = [j for j in range(n_parents) if parent_card[j] == 2]
            for a, b in combinations(binary, 2):
                cols.append((parent_vals[:, a] ^ parent_vals[:, b]).astype(float))

        if not cols:
            return np.zeros((m, 1), dtype=float)
        return np.column_stack(cols)
