"""
Inference for discrete Bayesian networks.

MaxProductInference
    MAP and k-best inference via variable elimination (max-product).
    Replaces the previous brute-force O(prod(cardinalities)) enumeration
    with an algorithm polynomial in the junction-tree clique sizes.

    MAP (most probable config):
        Variable elimination in the min-fill elimination order.
        Complexity: O(n * exp(treewidth)).

    k-best (k most probable configs):
        Nilsson / Lawler priority-queue search.  Each step runs one
        VE-MAP call with partial evidence.  Exact and duplicate-free.

    Marginals:
        Sum-product VE: one elimination pass per query variable.

References
----------
Nilsson (1998) "An efficient algorithm for finding the M most probable
configurations in probabilistic expert systems."
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from bayes_nets.bayesian_network import BayesianNetwork


# ---------------------------------------------------------------------------
# Factor algebra helpers
# ---------------------------------------------------------------------------


def _factor_from_cpd(
    var: int,
    parents: List[int],
    cpd: np.ndarray,
    cardinality: np.ndarray,
) -> Tuple[List[int], np.ndarray]:
    """Convert a CPD to a factor with *sorted* variable scope.

    Parameters
    ----------
    var : int
    parents : list of int   (in the order stored in the CPD)
    cpd : np.ndarray
        Shape ``(cardinality[var],)`` when no parents;
        ``(n_parent_configs, cardinality[var])`` otherwise.
        Parent config encoding: parents[0] varies fastest.
    cardinality : np.ndarray

    Returns
    -------
    scope : list of int   (sorted)
    table : np.ndarray    (shape = tuple(cardinality[v] for v in scope))
    """
    scope = sorted([var] + list(parents))

    if not parents:
        return scope, np.asarray(cpd, dtype=float).copy()

    parent_cards = [int(cardinality[p]) for p in parents]
    # CPD row index: parents[0] varies fastest (Fortran/column-major across parent dims).
    # Reshape using C-major with reversed parent order so numpy's row-major matches.
    cpd_shape = tuple(reversed(parent_cards)) + (int(cardinality[var]),)
    cpd_arr = np.asarray(cpd, dtype=float).reshape(cpd_shape)
    # Axes after reshape: [parents[-1], ..., parents[0], var]
    current_vars = list(reversed(parents)) + [var]
    perm = [current_vars.index(v) for v in scope]
    return scope, np.ascontiguousarray(np.transpose(cpd_arr, perm))


def _multiply_factors(
    factors: List[Tuple[List[int], np.ndarray]],
    cardinality: np.ndarray,
) -> Tuple[List[int], np.ndarray]:
    """Multiply a list of factors into one via broadcasting.

    Parameters
    ----------
    factors : list of (scope, table)
    cardinality : np.ndarray

    Returns
    -------
    scope : list of int   (sorted union of all scopes)
    table : np.ndarray
    """
    if not factors:
        return [], np.array(1.0)

    all_vars = sorted({v for scope, _ in factors for v in scope})

    if not all_vars:
        prod = 1.0
        for _, t in factors:
            prod *= float(np.asarray(t).flat[0])
        return [], np.array(prod)

    var_to_pos = {v: i for i, v in enumerate(all_vars)}
    cards = tuple(int(cardinality[v]) for v in all_vars)
    result = np.ones(cards, dtype=float)

    for scope, table in factors:
        table = np.asarray(table, dtype=float)
        if not scope:
            result *= float(table.flat[0])
            continue

        # Permute table axes into all_vars order
        perm = sorted(range(len(scope)), key=lambda i: var_to_pos[scope[i]])
        if perm != list(range(len(scope))):
            table = np.transpose(table, perm)
            scope_s = [scope[p] for p in perm]
        else:
            scope_s = list(scope)

        # Build broadcast shape: card for vars in scope, 1 for missing vars
        expanded: List[int] = []
        ptr = 0
        for v in all_vars:
            if ptr < len(scope_s) and scope_s[ptr] == v:
                expanded.append(int(cardinality[v]))
                ptr += 1
            else:
                expanded.append(1)

        result = result * table.reshape(expanded)

    return list(all_vars), result


# ---------------------------------------------------------------------------
# MaxProductInference
# ---------------------------------------------------------------------------


class MaxProductInference:
    """Exact MAP and k-best inference for discrete Bayesian networks.

    Uses variable elimination (max-product) rather than brute-force
    joint enumeration.  The junction-tree elimination order is computed
    once from the moral graph using the min-fill heuristic.

    Parameters
    ----------
    bn : BayesianNetwork
        Must have CPDs estimated (call ``bn.fit()`` or
        ``bn.learn_parameters()`` first).
    """

    def __init__(self, bn: BayesianNetwork) -> None:
        if not bn.cpds:
            raise RuntimeError(
                "CPDs are required for inference. "
                "Call learn_parameters() or fit() first."
            )
        self.bn = bn
        self._elim_order: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def most_probable_config(
        self,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Return the MAP assignment and its probability.

        Parameters
        ----------
        evidence : dict or iterable of (var, value) pairs, optional

        Returns
        -------
        assignment : np.ndarray, shape (n_vars,)
        probability : float
        """
        ev = self._normalize_evidence(evidence)
        return self._ve_map(ev)

    def k_most_probable_configs(
        self,
        k: int,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the *k* most probable assignments (Nilsson 1998).

        Uses a Nilsson/Lawler priority-queue search.  Each candidate is
        found by running VE-MAP with partial evidence, guaranteeing exact
        results with no duplicates.

        Parameters
        ----------
        k : int
        evidence : optional

        Returns
        -------
        assignments : np.ndarray, shape (k', n_vars)   k' ≤ k
        probabilities : np.ndarray, shape (k',)
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        ev = self._normalize_evidence(evidence)
        bn = self.bn

        map_assign, map_prob = self._ve_map(ev)
        result_assigns = [map_assign.copy()]
        result_probs = [map_prob]

        if k == 1:
            return np.array(result_assigns), np.array(result_probs)

        topo = list(bn._topological_sort())
        seen = {tuple(map_assign)}
        counter = [0]
        heap: list = []

        def push(assign: np.ndarray, prob: float, fixed_ev: Dict[int, int], split_pos: int) -> None:
            key = tuple(assign)
            if key not in seen and prob > 0.0:
                seen.add(key)
                heapq.heappush(heap, (-prob, counter[0], assign.copy(), fixed_ev, split_pos))
                counter[0] += 1

        # Initial partition: for each topo position i and alternative value v,
        # fix topo[0..i-1] to MAP values and topo[i] = v, rest free.
        for i, split_var in enumerate(topo):
            if split_var in ev:
                continue
            prefix = dict(ev)
            for j in range(i):
                pv = topo[j]
                if pv not in ev:
                    prefix[pv] = int(map_assign[pv])
            for v in range(int(bn.cardinality[split_var])):
                if v == int(map_assign[split_var]):
                    continue
                branch_ev = {**prefix, split_var: v}
                branch_assign, branch_prob = self._ve_map(branch_ev)
                push(branch_assign, branch_prob, branch_ev, i)

        # Expand until k configs collected
        while heap and len(result_assigns) < k:
            neg_p, _, assign, fixed_ev, split_i = heapq.heappop(heap)
            prob = -neg_p
            result_assigns.append(assign.copy())
            result_probs.append(prob)

            # Generate children: fix variables from split_i+1 to next branch point
            for j in range(split_i + 1, len(topo)):
                split_var = topo[j]
                if split_var in ev:
                    continue
                # Fix topo[split_i+1 .. j-1] to this assignment's values
                new_ev = dict(fixed_ev)
                for jj in range(split_i + 1, j):
                    pv = topo[jj]
                    if pv not in ev:
                        new_ev[pv] = int(assign[pv])
                for v in range(int(bn.cardinality[split_var])):
                    if v == int(assign[split_var]):
                        continue
                    branch_ev = {**new_ev, split_var: v}
                    branch_assign, branch_prob = self._ve_map(branch_ev)
                    push(branch_assign, branch_prob, branch_ev, j)

        return np.array(result_assigns), np.array(result_probs)

    def marginals(
        self,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
    ) -> List[np.ndarray]:
        """Return marginal distributions for all variables (sum-product VE).

        For each query variable, all other free variables are eliminated
        using sum-product.

        Parameters
        ----------
        evidence : optional

        Returns
        -------
        list of np.ndarray
            ``result[i]`` has shape ``(cardinality[i],)`` and sums to 1.
        """
        ev = self._normalize_evidence(evidence)
        bn = self.bn
        elim_base = self._get_elim_order()

        out: List[np.ndarray] = []
        for var in range(bn.n_vars):
            k = int(bn.cardinality[var])
            if var in ev:
                marg = np.zeros(k)
                marg[ev[var]] = 1.0
                out.append(marg)
                continue

            # Sum-product: eliminate all variables except var (and evidence vars)
            factors = self._build_factors()
            factors = self._apply_evidence(factors, ev)
            elim_order = [v for v in elim_base if v not in ev and v != var]

            current = list(factors)
            for elim_var in elim_order:
                vf = [(s, t) for s, t in current if elim_var in s]
                other = [(s, t) for s, t in current if elim_var not in s]
                if not vf:
                    continue
                comb_scope, comb_table = _multiply_factors(vf, bn.cardinality)
                if elim_var not in comb_scope:
                    current = other + [(comb_scope, comb_table)]
                    continue
                ax = comb_scope.index(elim_var)
                new_scope = [v for v in comb_scope if v != elim_var]
                new_table = comb_table.sum(axis=ax)
                current = other + ([(new_scope, new_table)] if new_scope else [])

            final_scope, final_table = _multiply_factors(current, bn.cardinality)
            if var in final_scope:
                ax = final_scope.index(var)
                axes_to_sum = [i for i in range(len(final_scope)) if i != ax]
                marg = final_table
                for axis in sorted(axes_to_sum, reverse=True):
                    marg = marg.sum(axis=axis)
            else:
                marg = np.ones(k)

            total = marg.sum()
            out.append(marg / total if total > 0 else np.ones(k) / k)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_evidence(
        self,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]],
    ) -> Dict[int, int]:
        if evidence is None:
            return {}
        if isinstance(evidence, dict):
            ev = {int(k): int(v) for k, v in evidence.items()}
        else:
            ev = {int(k): int(v) for k, v in evidence}
        for var, val in ev.items():
            if var < 0 or var >= self.bn.n_vars:
                raise ValueError(f"Evidence variable out of range: {var}")
            if val < 0 or val >= int(self.bn.cardinality[var]):
                raise ValueError(f"Evidence value out of range for variable {var}: {val}")
        return ev

    def _get_elim_order(self) -> List[int]:
        if self._elim_order is None:
            from bayes_nets.factorization import moralize, triangulate
            moral = moralize(self.bn.adjacency)
            _, order, _ = triangulate(moral, self.bn.cardinality, method="min-fill")
            self._elim_order = list(order)
        return self._elim_order

    def _build_factors(self) -> List[Tuple[List[int], np.ndarray]]:
        factors = []
        for var in range(self.bn.n_vars):
            info = self.bn.cpds[var]
            parents = [int(p) for p in info["parents"]]
            cpd = np.asarray(info["cpd"], dtype=float)
            scope, table = _factor_from_cpd(var, parents, cpd, self.bn.cardinality)
            factors.append((scope, table))
        return factors

    @staticmethod
    def _apply_evidence(
        factors: List[Tuple[List[int], np.ndarray]],
        ev: Dict[int, int],
    ) -> List[Tuple[List[int], np.ndarray]]:
        if not ev:
            return list(factors)
        result = []
        for scope, table in factors:
            curr_scope = list(scope)
            curr_table = np.asarray(table, dtype=float)
            for ev_var, ev_val in ev.items():
                if ev_var not in curr_scope:
                    continue
                ax = curr_scope.index(ev_var)
                sl: List = [slice(None)] * len(curr_scope)
                sl[ax] = int(ev_val)
                curr_table = curr_table[tuple(sl)]
                curr_scope = [v for v in curr_scope if v != ev_var]
            if curr_scope:
                result.append((curr_scope, curr_table))
        return result

    def _joint_prob(self, assignment: np.ndarray) -> float:
        """Compute P(assignment) from CPDs in O(n)."""
        prob = 1.0
        for var in range(self.bn.n_vars):
            info = self.bn.cpds[var]
            parents = [int(p) for p in info["parents"]]
            cpd = np.asarray(info["cpd"], dtype=float)
            xi = int(assignment[var])
            if not parents:
                prob *= float(cpd[xi])
            else:
                pc = 0
                mult = 1
                for p in parents:
                    pc += int(assignment[p]) * mult
                    mult *= int(self.bn.cardinality[p])
                prob *= float(cpd[pc, xi])
        return prob

    def _ve_map(self, ev: Dict[int, int]) -> Tuple[np.ndarray, float]:
        """Variable elimination for MAP (max-product).

        Returns
        -------
        assignment : np.ndarray, shape (n_vars,)
        probability : float
        """
        bn = self.bn

        factors = self._build_factors()
        factors = self._apply_evidence(factors, ev)

        elim_order = [v for v in self._get_elim_order() if v not in ev]

        current = list(factors)
        argmax_records: List[Tuple[int, List[int], np.ndarray]] = []

        for var in elim_order:
            vf = [(s, t) for s, t in current if var in s]
            other = [(s, t) for s, t in current if var not in s]

            if not vf:
                # Variable absent from all remaining factors; use value 0.
                argmax_records.append((var, [], np.array(0, dtype=int)))
                continue

            comb_scope, comb_table = _multiply_factors(vf, bn.cardinality)

            if var not in comb_scope:
                current = other + [(comb_scope, comb_table)]
                continue

            ax = comb_scope.index(var)
            argmax_scope = [v for v in comb_scope if v != var]
            argmax_t = np.argmax(comb_table, axis=ax)
            max_t = np.max(comb_table, axis=ax)

            argmax_records.append((var, argmax_scope, argmax_t))
            current = other + ([(argmax_scope, max_t)] if argmax_scope else [])

        # Traceback: process in reverse elimination order
        assignment = np.zeros(bn.n_vars, dtype=int)
        for ev_var, val in ev.items():
            assignment[ev_var] = val

        for var, argmax_scope, argmax_t in reversed(argmax_records):
            if not argmax_scope:
                assignment[var] = int(np.asarray(argmax_t).flat[0])
            else:
                idx = tuple(int(assignment[v]) for v in argmax_scope)
                assignment[var] = int(argmax_t[idx])

        return assignment, self._joint_prob(assignment)
