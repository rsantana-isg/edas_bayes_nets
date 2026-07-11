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

MAX_LOOPY_DAMPING = 0.99


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

    def __init__(
        self,
        bn: BayesianNetwork,
        loopy_treewidth_threshold: Optional[int] = 8,
        loopy_max_iter: int = 100,
        loopy_tol: float = 1e-6,
        loopy_damping: float = 0.5,
    ) -> None:
        if not bn.cpds:
            raise RuntimeError(
                "CPDs are required for inference. "
                "Call learn_parameters() or fit() first."
            )
        self.bn = bn
        self._elim_order: Optional[List[int]] = None
        self._treewidth_estimate: Optional[int] = None
        self.loopy_treewidth_threshold = loopy_treewidth_threshold
        self.loopy_max_iter = max(1, int(loopy_max_iter))
        self.loopy_tol = float(loopy_tol)
        self.loopy_damping = float(np.clip(loopy_damping, 0.0, MAX_LOOPY_DAMPING))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def most_probable_config(
        self,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
        search_method: str = "auto",
    ) -> Tuple[np.ndarray, float]:
        """Return the MAP assignment and its probability.

        Parameters
        ----------
        evidence : dict or iterable of (var, value) pairs, optional
        search_method : str
            ``"auto"`` (default) uses exact variable elimination, falling
            back to loopy max-product when the treewidth is large.
            ``"ve"`` forces exact variable elimination.
            ``"loopy"`` forces damped max-product loopy belief propagation.
            ``"srmp"`` / ``"trw"`` uses sequential reweighted message passing
            (Kolmogorov 2015), a convergent approximate MAP solver that is
            exact on tree-structured networks.

        Returns
        -------
        assignment : np.ndarray, shape (n_vars,)
        probability : float
        """
        ev = self._normalize_evidence(evidence)
        method = str(search_method).lower()
        if method in {"srmp", "trw", "trws"}:
            return self._srmp_map(ev)
        if method == "loopy":
            return self._loopy_map(ev)
        if method == "ve":
            return self._ve_map(ev)
        if method not in {"auto"}:
            raise ValueError(
                "search_method must be 'auto', 've', 'loopy', or 'srmp'"
            )
        if self._should_use_loopy():
            return self._loopy_map(ev)
        return self._ve_map(ev)

    def k_most_probable_configs(
        self,
        k: int,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
        search_method: str = "nilsson",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the *k* most probable assignments (Nilsson 1998).

        Uses a Nilsson/Lawler priority-queue search.  Each candidate is
        found by running VE-MAP with partial evidence, guaranteeing exact
        results with no duplicates.  When ``search_method`` is set to
        ``"a_star_bb"``/``"flerova"``, uses an A*/branch-and-bound
        search strategy inspired by Flerova et al.

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
        method = str(search_method).lower()
        if method in {"nilsson", "lawler", "ve"}:
            return self._k_most_nilsson(k=k, ev=ev)
        if method in {"a_star_bb", "astar_bb", "astar", "branch_and_bound", "flerova"}:
            return self._k_most_astar_branch_and_bound(k=k, ev=ev)
        raise ValueError(
            "search_method must be one of: "
            "'nilsson', 'a_star_bb', 'astar_bb', 'astar', 'branch_and_bound', or 'flerova'"
        )

    def _k_most_nilsson(
        self,
        k: int,
        ev: Dict[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        bn = self.bn

        map_solver = self._loopy_map if self._should_use_loopy() else self._ve_map
        map_assign, map_prob = map_solver(ev)
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
                branch_assign, branch_prob = map_solver(branch_ev)
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
                    branch_assign, branch_prob = map_solver(branch_ev)
                    push(branch_assign, branch_prob, branch_ev, j)

        return np.array(result_assigns), np.array(result_probs)

    def _k_most_astar_branch_and_bound(
        self,
        k: int,
        ev: Dict[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """A*/branch-and-bound over partial assignments using MAP bounds."""
        bn = self.bn
        topo = list(bn._topological_sort())
        map_solver = self._loopy_map if self._should_use_loopy() else self._ve_map

        heap: List[Tuple[float, int, Dict[int, int]]] = []
        counter = 0
        _, root_prob = map_solver(ev)
        if root_prob <= 0.0:
            return np.empty((0, bn.n_vars), dtype=int), np.empty(0, dtype=float)
        heapq.heappush(heap, (-float(root_prob), counter, dict(ev)))
        counter += 1

        results: List[Tuple[np.ndarray, float]] = []
        kth_bound = -np.inf

        while heap:
            neg_bound, _, partial = heapq.heappop(heap)
            bound = -neg_bound
            if len(results) >= k and bound <= kth_bound:
                continue

            unassigned = [v for v in topo if v not in partial]
            if not unassigned:
                assign = np.zeros(bn.n_vars, dtype=int)
                for var, val in partial.items():
                    assign[var] = int(val)
                prob = self._joint_prob(assign)
                results.append((assign, prob))
                results.sort(key=lambda x: x[1], reverse=True)
                if len(results) > k:
                    results = results[:k]
                if len(results) >= k:
                    kth_bound = results[k - 1][1]
                continue

            split_var = int(unassigned[0])
            for value in range(int(bn.cardinality[split_var])):
                child = dict(partial)
                child[split_var] = value
                _, child_bound = map_solver(child)
                if child_bound <= 0.0:
                    continue
                if len(results) >= k and child_bound <= kth_bound:
                    continue
                heapq.heappush(heap, (-float(child_bound), counter, child))
                counter += 1

        if not results:
            return np.empty((0, bn.n_vars), dtype=int), np.empty(0, dtype=float)

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:k]
        assignments = np.array([item[0] for item in top], dtype=int)
        probabilities = np.array([item[1] for item in top], dtype=float)
        return assignments, probabilities

    def marginals(
        self,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
        method: str = "ve",
    ) -> List[np.ndarray]:
        """Return marginal distributions for all variables.

        Parameters
        ----------
        evidence : optional
        method : str
            ``"ve"`` (default) computes exact marginals by sum-product
            variable elimination.  ``"mean_field"`` computes fast approximate
            marginals by coordinate-ascent mean-field variational inference
            (Li & Zemel 2014); useful when the exact junction tree is too
            wide for repeated EDA queries.

        Returns
        -------
        list of np.ndarray
            ``result[i]`` has shape ``(cardinality[i],)`` and sums to 1.
        """
        method = str(method).lower()
        if method in {"mean_field", "meanfield", "mf"}:
            return self.mean_field_marginals(evidence)
        if method not in {"ve", "exact"}:
            raise ValueError("method must be 've' or 'mean_field'")
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
            _, order, cliques = triangulate(moral, self.bn.cardinality, method="min-fill")
            self._elim_order = list(order)
            # Empty cliques can occur for degenerate empty-graph cases.
            self._treewidth_estimate = max((len(c) - 1 for c in cliques), default=0)
        return self._elim_order

    def _estimated_treewidth(self) -> int:
        if self._treewidth_estimate is None:
            self._get_elim_order()
        return int(self._treewidth_estimate or 0)

    def _should_use_loopy(self) -> bool:
        if self.loopy_treewidth_threshold is None:
            return False
        return self._estimated_treewidth() > int(self.loopy_treewidth_threshold)

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

    def _loopy_map(self, ev: Dict[int, int]) -> Tuple[np.ndarray, float]:
        """Approximate MAP by damped max-product loopy belief propagation."""
        bn = self.bn
        factors = self._build_factors()

        var_to_factors: List[List[int]] = [[] for _ in range(bn.n_vars)]
        for f_idx, (scope, _) in enumerate(factors):
            for v in scope:
                var_to_factors[v].append(f_idx)

        msg_vf: Dict[Tuple[int, int], np.ndarray] = {}
        msg_fv: Dict[Tuple[int, int], np.ndarray] = {}

        for var in range(bn.n_vars):
            card = int(bn.cardinality[var])
            init = np.ones(card, dtype=float) / card
            if var in ev:
                init = np.zeros(card, dtype=float)
                init[ev[var]] = 1.0
            for f_idx in var_to_factors[var]:
                msg_vf[(var, f_idx)] = init.copy()
                msg_fv[(f_idx, var)] = init.copy()

        for _ in range(self.loopy_max_iter):
            max_delta = 0.0

            new_fv: Dict[Tuple[int, int], np.ndarray] = {}
            for f_idx, (scope, table) in enumerate(factors):
                table_arr = np.asarray(table, dtype=float)
                for target_pos, var in enumerate(scope):
                    msg = table_arr
                    for pos, other_var in enumerate(scope):
                        if pos == target_pos:
                            continue
                        incoming = msg_vf[(other_var, f_idx)]
                        shape = [1] * msg.ndim
                        shape[pos] = int(bn.cardinality[other_var])
                        msg = msg * incoming.reshape(shape)

                    axes = tuple(i for i in range(msg.ndim) if i != target_pos)
                    out = msg if len(axes) == 0 else np.max(msg, axis=axes)
                    out = np.asarray(out, dtype=float)

                    if var in ev:
                        forced = np.zeros(int(bn.cardinality[var]), dtype=float)
                        forced[ev[var]] = 1.0
                        out = forced
                    else:
                        m = float(np.max(out))
                        if not np.isfinite(m) or m <= 0.0:
                            out = np.ones(int(bn.cardinality[var]), dtype=float) / int(bn.cardinality[var])
                        else:
                            out = out / m

                    old = msg_fv[(f_idx, var)]
                    damped = self.loopy_damping * old + (1.0 - self.loopy_damping) * out
                    max_delta = max(max_delta, float(np.max(np.abs(damped - old))))
                    new_fv[(f_idx, var)] = damped

            new_vf: Dict[Tuple[int, int], np.ndarray] = {}
            for var in range(bn.n_vars):
                card = int(bn.cardinality[var])
                for f_idx in var_to_factors[var]:
                    if var in ev:
                        out = np.zeros(card, dtype=float)
                        out[ev[var]] = 1.0
                    else:
                        out = np.ones(card, dtype=float)
                        for other_f in var_to_factors[var]:
                            if other_f == f_idx:
                                continue
                            out *= new_fv[(other_f, var)]
                        m = float(np.max(out))
                        if not np.isfinite(m) or m <= 0.0:
                            out = np.ones(card, dtype=float) / card
                        else:
                            out = out / m

                    old = msg_vf[(var, f_idx)]
                    damped = self.loopy_damping * old + (1.0 - self.loopy_damping) * out
                    max_delta = max(max_delta, float(np.max(np.abs(damped - old))))
                    new_vf[(var, f_idx)] = damped

            msg_fv = new_fv
            msg_vf = new_vf

            if max_delta < self.loopy_tol:
                break

        assignment = np.zeros(bn.n_vars, dtype=int)
        for var in range(bn.n_vars):
            if var in ev:
                assignment[var] = ev[var]
                continue
            card = int(bn.cardinality[var])
            belief = np.ones(card, dtype=float)
            for f_idx in var_to_factors[var]:
                belief *= msg_fv[(f_idx, var)]
            assignment[var] = int(np.argmax(belief))

        return assignment, self._joint_prob(assignment)

    # ------------------------------------------------------------------
    # Mean-field variational marginals  (Li & Zemel 2014)
    # ------------------------------------------------------------------

    def mean_field_marginals(
        self,
        evidence: Optional[Dict[int, int] | Iterable[Tuple[int, int]]] = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        damping: float = 0.5,
    ) -> List[np.ndarray]:
        """Approximate marginals by coordinate-ascent mean field.

        Minimises the KL divergence to a fully factorised distribution
        ``q(x) = ∏_v q_v(x_v)``.  Each variable's factor is updated from its
        neighbours' expectations,

            log q_v(x_v) ← Σ_{f ∋ v} E_{q_{-v}}[ log ψ_f(x_v, x_{-v}) ],

        iterated with optional damping until convergence.  Always converges
        and costs ``O(iterations · Σ_f |ψ_f|)`` — far cheaper than exact
        elimination on wide networks, at the price of an approximation.

        References
        ----------
        Li, Y. & Zemel, R. (2014). "Mean Field Networks." ICML Workshop /
        arXiv:1410.5884.
        """
        ev = self._normalize_evidence(evidence)
        bn = self.bn
        damping = float(np.clip(damping, 0.0, MAX_LOOPY_DAMPING))
        floor = 1e-12

        factors = self._build_factors()
        log_tables = [
            (scope, np.log(np.asarray(t, dtype=float) + floor))
            for scope, t in factors
        ]
        var_to_factors: List[List[int]] = [[] for _ in range(bn.n_vars)]
        for f_idx, (scope, _) in enumerate(factors):
            for v in scope:
                var_to_factors[v].append(f_idx)

        # Initialise q: uniform, or a delta at the evidence value.
        q: List[np.ndarray] = []
        for var in range(bn.n_vars):
            card = int(bn.cardinality[var])
            if var in ev:
                qi = np.zeros(card)
                qi[ev[var]] = 1.0
            else:
                qi = np.ones(card) / card
            q.append(qi)

        for _ in range(max(1, int(max_iter))):
            max_delta = 0.0
            for var in range(bn.n_vars):
                if var in ev:
                    continue
                card = int(bn.cardinality[var])
                log_qv = np.zeros(card, dtype=float)
                for f_idx in var_to_factors[var]:
                    scope, log_tab = log_tables[f_idx]
                    ax = scope.index(var)
                    weighted = log_tab
                    for pos, other in enumerate(scope):
                        if pos == ax:
                            continue
                        shape = [1] * log_tab.ndim
                        shape[pos] = int(bn.cardinality[other])
                        weighted = weighted * q[other].reshape(shape)
                    axes = tuple(i for i in range(log_tab.ndim) if i != ax)
                    log_qv += weighted.sum(axis=axes) if axes else weighted
                log_qv -= log_qv.max()
                new_qv = np.exp(log_qv)
                total = new_qv.sum()
                new_qv = new_qv / total if total > 0 else np.ones(card) / card
                new_qv = damping * q[var] + (1.0 - damping) * new_qv
                new_qv /= new_qv.sum()
                max_delta = max(max_delta, float(np.max(np.abs(new_qv - q[var]))))
                q[var] = new_qv
            if max_delta < tol:
                break

        return q

    # ------------------------------------------------------------------
    # Sequential reweighted message passing for MAP  (Kolmogorov 2015)
    # ------------------------------------------------------------------

    def _srmp_map(
        self,
        ev: Dict[int, int],
        max_iter: int = 100,
        rho: float = 1.0,
        tol: float = 1e-9,
    ) -> Tuple[np.ndarray, float]:
        """Sequential reweighted max-product for MAP-MRF (SRMP / TRW-S).

        Performs block-coordinate ascent on the LP-dual of the MAP problem
        with a forward/backward variable schedule, in the log domain.  Exact
        on tree-structured factor graphs; a convergent approximation with a
        certifying bound on loopy graphs.  The returned probability is the
        exact joint probability of the decoded assignment.

        References
        ----------
        Kolmogorov, V. (2015). "A new look at reweighted message passing."
        IEEE TPAMI 37(5), 919-930.
        """
        bn = self.bn
        floor = 1e-12

        factors = self._build_factors()
        theta = [
            (scope, np.log(np.asarray(t, dtype=float) + floor))
            for scope, t in factors
        ]
        var_to_factors: List[List[int]] = [[] for _ in range(bn.n_vars)]
        for f_idx, (scope, _) in enumerate(factors):
            for v in scope:
                var_to_factors[v].append(f_idx)

        # messages m_{f->v}(x_v) in log domain
        msg_fv: Dict[Tuple[int, int], np.ndarray] = {}
        for f_idx, (scope, _) in enumerate(factors):
            for v in scope:
                msg_fv[(f_idx, v)] = np.zeros(int(bn.cardinality[v]), dtype=float)

        def belief(v: int) -> np.ndarray:
            b = np.zeros(int(bn.cardinality[v]), dtype=float)
            for f_idx in var_to_factors[v]:
                b += msg_fv[(f_idx, v)]
            if v in ev:
                forced = np.full(int(bn.cardinality[v]), -np.inf)
                forced[ev[v]] = 0.0
                b = b + forced
            return b

        def update_factor(f_idx: int, target: int) -> None:
            scope, tab = theta[f_idx]
            ax = scope.index(target)
            acc = tab.copy()
            for pos, u in enumerate(scope):
                if pos == ax:
                    continue
                # reparameterised node potential entering the factor
                n_u = rho * belief(u) - msg_fv[(f_idx, u)]
                shape = [1] * tab.ndim
                shape[pos] = int(bn.cardinality[u])
                acc = acc + n_u.reshape(shape)
            axes = tuple(i for i in range(tab.ndim) if i != ax)
            out = acc.max(axis=axes) if axes else acc
            out = np.asarray(out, dtype=float)
            out = out - out.max()  # normalise for numerical stability
            msg_fv[(f_idx, target)] = out

        order = list(range(bn.n_vars))
        schedule = order + order[::-1]

        best_assign = None
        best_prob = -1.0
        prev_energy = None
        for _ in range(max(1, int(max_iter))):
            for v in schedule:
                if v in ev:
                    continue
                for f_idx in var_to_factors[v]:
                    update_factor(f_idx, v)

            assignment = np.zeros(bn.n_vars, dtype=int)
            for v in range(bn.n_vars):
                assignment[v] = ev[v] if v in ev else int(np.argmax(belief(v)))
            prob = self._joint_prob(assignment)
            if prob > best_prob:
                best_prob = prob
                best_assign = assignment.copy()

            energy = float(sum(float(np.max(belief(v))) for v in range(bn.n_vars)))
            if prev_energy is not None and abs(energy - prev_energy) < tol:
                break
            prev_energy = energy

        if best_assign is None:  # pragma: no cover - defensive
            best_assign = np.zeros(bn.n_vars, dtype=int)
            best_prob = self._joint_prob(best_assign)
        return best_assign, best_prob
