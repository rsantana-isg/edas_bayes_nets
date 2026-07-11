"""
Differentiable structure learning for binary data (BINOTEARS).

BinaryNotearsLearner
    Continuous-optimisation structure learning for **binary** Bayesian
    networks.  It adapts the NOTEARS framework (Zheng et al. 2018) to binary
    data by replacing the Gaussian least-squares loss with the multivariate
    Bernoulli / logistic log-likelihood, following Deng & Aragam (2025).

Each node is modelled by a logistic regression on its parents,

    P(X_j = 1 | X) = sigmoid( sum_i W_ij X_i + c_j ),

with a continuous weighted adjacency matrix ``W`` (``W_jj = 0``).  The graph
is forced to be acyclic through the smooth constraint of Zheng et al.,

    h(W) = trace(exp(W ∘ W)) - p = 0,

which is enforced with an augmented-Lagrangian scheme.  An L1 penalty on
``W`` promotes sparsity, and the final weighted matrix is thresholded to a
binary DAG.

This is the tractable *first-order* instantiation of BINOTEARS
(Assumption A of Deng & Aragam 2025): only main effects of the parents are
modelled, which keeps the parameter count at ``p x p`` and the optimisation
practical for moderate ``p``.  Higher-order interaction terms (the fully
general multivariate-Bernoulli model) are left as an extension.

References
----------
Deng & Aragam (2025). "Differentiable Structure Learning and Causal
Discovery for General Binary Data." NeurIPS 2025.
Zheng, Aragam, Ravikumar & Xing (2018). "DAGs with NO TEARS: Continuous
Optimization for Structure Learning." NeurIPS 2018.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable logistic
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class BinaryNotearsLearner:
    """Differentiable (NOTEARS-style) structure learning for binary data.

    Parameters
    ----------
    lambda1 : float
        L1 penalty on the weighted adjacency matrix (larger -> sparser).
    max_iter : int
        Number of augmented-Lagrangian outer iterations.
    h_tol : float
        Stop when the acyclicity value ``h(W)`` falls below this.
    rho_max : float
        Cap on the penalty coefficient ``rho``.
    w_threshold : float
        Absolute-value threshold applied to ``W`` to read off edges.
    max_parents : int or None
        Optional cap on in-degree; the highest-weight parents are kept.

    Notes
    -----
    Only binary variables (cardinality 2) are supported.
    """

    def __init__(
        self,
        lambda1: float = 0.05,
        max_iter: int = 40,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
        max_parents: Optional[int] = None,
    ) -> None:
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.max_parents = max_parents

    # ------------------------------------------------------------------
    # Loss, acyclicity, and their gradients
    # ------------------------------------------------------------------

    def _loss(self, W: np.ndarray, X: np.ndarray, bias: np.ndarray,
              weights: Optional[np.ndarray] = None):
        """Weighted logistic (cross-entropy) loss and its gradient.

        For each node j:  logits = X W[:, j] + bias[j].
        loss = (1/n) * sum_i w_i * (softplus(logit_i) - X_ij * logit_i),
        where ``weights`` are per-row effective weights summing to n (so a
        uniform vector of ones reproduces the plain mean).  ``None`` -> ones.
        """
        n = X.shape[0]
        logits = X @ W + bias                       # (n, p)
        sp = np.logaddexp(0.0, logits)              # stable softplus
        per_row = sp - X * logits                   # (n, p)
        probs = _sigmoid(logits)                    # (n, p)
        resid = probs - X                           # d loss / d logit
        if weights is None:
            loss = np.sum(per_row) / n
            G_W = (X.T @ resid) / n
            G_bias = resid.sum(axis=0) / n
        else:
            w = weights[:, None]                    # (n, 1)
            loss = np.sum(w * per_row) / n
            G_W = (X.T @ (w * resid)) / n
            G_bias = (w * resid).sum(axis=0) / n
        np.fill_diagonal(G_W, 0.0)
        return loss, G_W, G_bias

    def _h(self, W: np.ndarray):
        """Acyclicity h(W) = tr(exp(W∘W)) - p and its gradient."""
        p = W.shape[0]
        M = W * W
        E = expm(M)
        h = np.trace(E) - p
        G_h = E.T * 2.0 * W
        return h, G_h

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn(
        self,
        data: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a binary adjacency matrix learned from *data*.

        ``permutation`` is ignored.  ``interaction_matrix`` masks disallowed
        edges to zero.  ``sample_weights`` is a probability vector over rows
        (must sum to 1); it re-weights the logistic likelihood so that a
        weighted row counts like ``weight * n`` unit-weight rows.  ``None``
        uses uniform weights.
        """
        cardinality = np.asarray(cardinality, dtype=int)
        if np.any(cardinality != 2):
            raise ValueError("BinaryNotearsLearner supports binary variables only.")

        X = np.asarray(data, dtype=float)
        p = n_vars
        n = X.shape[0]

        # Per-row effective weights summing to n (uniform ones -> plain mean).
        if sample_weights is None:
            w_eff = None
        else:
            w_eff = np.asarray(sample_weights, dtype=float) * n

        mask = None
        if interaction_matrix is not None:
            mask = (np.asarray(interaction_matrix) != 0).astype(float)
            np.fill_diagonal(mask, 0.0)

        rho, alpha_lag, h = 1.0, 0.0, np.inf
        W_est = np.zeros((p, p))
        bias = np.zeros(p)
        # fit biases to (weighted) marginals as a warm start
        if w_eff is None:
            marg = X.mean(axis=0)
        else:
            marg = (w_eff[:, None] * X).sum(axis=0) / n
        marg = np.clip(marg, 1e-3, 1 - 1e-3)
        bias = np.log(marg / (1 - marg))

        def unpack(v):
            W = v[: p * p].reshape(p, p)
            b = v[p * p:]
            if mask is not None:
                W = W * mask
            return W, b

        def objective(v):
            W, b = unpack(v)
            loss, G_W, G_b = self._loss(W, X, b, w_eff)
            hval, G_h = self._h(W)
            obj = loss + 0.5 * rho * hval * hval + alpha_lag * hval \
                + self.lambda1 * np.sum(np.abs(W))
            grad_W = G_W + (rho * hval + alpha_lag) * G_h + self.lambda1 * np.sign(W)
            if mask is not None:
                grad_W = grad_W * mask
            np.fill_diagonal(grad_W, 0.0)
            grad = np.concatenate([grad_W.ravel(), G_b])
            return obj, grad

        v = np.concatenate([W_est.ravel(), bias])
        for _ in range(self.max_iter):
            # solve the unconstrained subproblem for the current (rho, alpha)
            sol = minimize(objective, v, jac=True, method="L-BFGS-B")
            v = sol.x
            W_new, bias = unpack(v)
            h_new, _ = self._h(W_new)
            W_est = W_new
            # augmented-Lagrangian updates
            if h_new > 0.25 * h:
                rho = min(rho * 10, self.rho_max)
            alpha_lag += rho * h_new
            h = h_new
            if h <= self.h_tol or rho >= self.rho_max:
                break

        # threshold to a binary DAG
        A = (np.abs(W_est) > self.w_threshold).astype(int)
        np.fill_diagonal(A, 0)

        if self.max_parents is not None:
            A = self._cap_parents(A, np.abs(W_est), self.max_parents)

        A = self._break_cycles(A, np.abs(W_est))
        return A

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cap_parents(A: np.ndarray, W_abs: np.ndarray, mp: int) -> np.ndarray:
        p = A.shape[0]
        for v in range(p):
            parents = np.where(A[:, v] > 0)[0]
            if len(parents) > mp:
                keep = parents[np.argsort(W_abs[parents, v])[::-1][:mp]]
                A[:, v] = 0
                A[keep, v] = 1
        return A

    @staticmethod
    def _break_cycles(A: np.ndarray, W_abs: np.ndarray) -> np.ndarray:
        """Remove weakest edges until the graph is acyclic (safety net)."""
        A = A.copy()

        def has_cycle(adj):
            p = adj.shape[0]
            colour = np.zeros(p, dtype=int)

            def dfs(u):
                colour[u] = 1
                for w in np.where(adj[u] > 0)[0]:
                    if colour[w] == 1:
                        return True
                    if colour[w] == 0 and dfs(w):
                        return True
                colour[u] = 2
                return False

            return any(colour[i] == 0 and dfs(i) for i in range(p))

        while has_cycle(A):
            edges = np.argwhere(A > 0)
            # drop the edge with the smallest |W|
            weakest = min(edges, key=lambda e: W_abs[e[0], e[1]])
            A[weakest[0], weakest[1]] = 0
        return A
