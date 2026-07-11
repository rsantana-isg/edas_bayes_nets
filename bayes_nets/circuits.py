"""
Compile a discrete Bayesian network into a tractable arithmetic circuit.

A Bayesian network of bounded treewidth compiles into an arithmetic circuit
(equivalently a Sum-Product Network, SPN) whose sum/product structure follows
a rooted junction tree.  Once compiled, the circuit answers marginal,
most-probable-explanation (MPE) and sampling queries **exactly** in time
linear in the circuit size — which is ``O(n · exp(treewidth))``.  This is the
canonical "transform the PGM into a more informative structure useful for
optimization": inside an Estimation of Distribution Algorithm the compiled
circuit yields exact marginals, exact best-configuration (MPE) and exact
linear-time sampling from a single compiled representation.

The compilation follows the standard junction-tree → arithmetic-circuit
construction (Darwiche 2003); the SPN reading of the resulting sum/product
DAG follows Poon & Domingos (2011) and the structural analysis of Vergari,
Di Mauro & Esposito (2016).

References
----------
Vergari, A., Di Mauro, N. & Esposito, F. (2016). "Visualizing and
Understanding Sum-Product Networks." Machine Learning 108, 551-573.
arXiv:1608.08266.

Darwiche, A. (2003). "A differential approach to inference in Bayesian
networks." Journal of the ACM 50(3), 280-305.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from bayes_nets.bayesian_network import BayesianNetwork
from bayes_nets.inference import _factor_from_cpd, _multiply_factors

Table = Tuple[List[int], np.ndarray]


# ---------------------------------------------------------------------------
# Table algebra (scope = sorted list of variable indices; axes follow scope)
# ---------------------------------------------------------------------------


def _mul(a: Table, b: Table, cardinality: np.ndarray) -> Table:
    """Multiply two factors, returning a factor over the union of scopes."""
    return _multiply_factors([a, b], cardinality)


def _reduce(table: Table, keep: List[int], op: str) -> Table:
    """Reduce *table* to the ``keep`` sub-scope by ``sum`` or ``max``."""
    scope, arr = table
    keep_set = set(keep)
    axes = tuple(i for i, v in enumerate(scope) if v not in keep_set)
    if axes:
        arr = arr.sum(axis=axes) if op == "sum" else arr.max(axis=axes)
    new_scope = [v for v in scope if v in keep_set]
    return new_scope, np.asarray(arr, dtype=float)


def _restrict(table: Table, fixed: Dict[int, int]) -> Table:
    """Slice *table* at the fixed variable values, dropping those axes."""
    scope, arr = table
    new_scope: List[int] = []
    index: List = []
    for i, v in enumerate(scope):
        if v in fixed:
            index.append(int(fixed[v]))
        else:
            index.append(slice(None))
            new_scope.append(v)
    return new_scope, np.asarray(arr, dtype=float)[tuple(index)]


def _apply_evidence(table: Table, ev: Dict[int, int]) -> Table:
    """Zero out entries inconsistent with evidence (keeps the scope)."""
    scope, arr = table
    arr = np.asarray(arr, dtype=float)
    for v, val in ev.items():
        if v not in scope:
            continue
        ax = scope.index(v)
        masked = np.zeros_like(arr)
        idx: List = [slice(None)] * arr.ndim
        idx[ax] = int(val)
        masked[tuple(idx)] = arr[tuple(idx)]
        arr = masked
    return scope, arr


# ---------------------------------------------------------------------------
# Arithmetic circuit
# ---------------------------------------------------------------------------


class ArithmeticCircuit:
    """Junction-tree arithmetic circuit / SPN compiled from a Bayesian network.

    Build with :meth:`from_bayesian_network`.  The circuit is a rooted clique
    tree whose per-clique potentials are the product of the assigned CPDs;
    all queries are single passes over that tree.

    Attributes
    ----------
    treewidth : int
        Width of the underlying junction tree (max clique size − 1).
    n_sum_nodes, n_product_nodes, n_leaf_nodes : int
        Node counts of the equivalent SPN (for structural inspection, in the
        spirit of Vergari et al. 2016).
    """

    def __init__(
        self,
        cardinality: np.ndarray,
        cliques: List[List[int]],
        psi: List[Table],
        parent: List[int],
        separator: List[List[int]],
        children: List[List[int]],
        order: List[int],
        treewidth: int,
    ) -> None:
        self.cardinality = np.asarray(cardinality, dtype=int)
        self.n_vars = len(self.cardinality)
        self.cliques = cliques
        self.psi = psi
        self.parent = parent
        self.separator = separator
        self.children = children
        self.order = order
        self.roots = [c for c in range(len(cliques)) if parent[c] < 0]
        self.treewidth = int(treewidth)
        # Home clique for each variable (first clique in tree order holding it).
        self._home: Dict[int, int] = {}
        for c in order:
            for v in cliques[c]:
                self._home.setdefault(int(v), c)
        self._count_nodes()

    # -- construction ---------------------------------------------------

    @classmethod
    def from_bayesian_network(
        cls, bn: BayesianNetwork, method: str = "min-fill"
    ) -> "ArithmeticCircuit":
        """Compile *bn* (which must have CPDs) into an arithmetic circuit."""
        if not bn.cpds:
            raise RuntimeError(
                "CPDs are required to compile a circuit. "
                "Call learn_parameters() or fit() first."
            )
        from bayes_nets.factorization import (
            moralize,
            triangulate,
            junction_tree,
            _order_cliques_for_sampling,
        )

        card = np.asarray(bn.cardinality, dtype=int)
        moral = moralize(bn.adjacency)
        _, _, clique_arrs = triangulate(moral, card, method=method)
        cliques = [sorted(int(v) for v in c) for c in clique_arrs]
        if not cliques:
            cliques = [[]]

        tree_edges, _ = junction_tree([np.asarray(c, dtype=int) for c in cliques])
        order, parent_map = _order_cliques_for_sampling(
            [np.asarray(c, dtype=int) for c in cliques], tree_edges
        )
        n_cl = len(cliques)
        parent = [int(parent_map.get(c, -1)) for c in range(n_cl)]
        children: List[List[int]] = [[] for _ in range(n_cl)]
        for c in range(n_cl):
            if parent[c] >= 0:
                children[parent[c]].append(c)
        separator: List[List[int]] = []
        for c in range(n_cl):
            if parent[c] < 0:
                separator.append([])
            else:
                shared = sorted(set(cliques[c]) & set(cliques[parent[c]]))
                separator.append(shared)

        treewidth = max((len(c) - 1 for c in cliques), default=0)

        # Initialise clique potentials to all-ones tables.
        psi: List[Table] = []
        for c in range(n_cl):
            scope = cliques[c]
            shape = tuple(int(card[v]) for v in scope) if scope else ()
            arr = np.ones(shape, dtype=float) if scope else np.array(1.0)
            psi.append((list(scope), arr))

        # Assign each variable's family (var + parents) to a covering clique.
        clique_sets = [set(c) for c in cliques]
        for var in range(bn.n_vars):
            info = bn.cpds[var]
            parents = [int(p) for p in info["parents"]]
            family = set([var] + parents)
            home = next(
                (c for c in order if family <= clique_sets[c]),
                None,
            )
            if home is None:  # pragma: no cover - guaranteed by moralization
                raise RuntimeError(
                    f"No clique covers the family of variable {var}; "
                    "triangulation is inconsistent."
                )
            fam_scope, fam_table = _factor_from_cpd(
                var, parents, np.asarray(info["cpd"], dtype=float), card
            )
            psi[home] = _mul(psi[home], (fam_scope, fam_table), card)

        return cls(card, cliques, psi, parent, separator, children, order, treewidth)

    def _count_nodes(self) -> None:
        n_sum = 0
        n_prod = 0
        for c in range(len(self.cliques)):
            sep_cfg = int(np.prod([int(self.cardinality[v]) for v in self.separator[c]])) if self.separator[c] else 1
            clq_cfg = int(np.prod([int(self.cardinality[v]) for v in self.cliques[c]])) if self.cliques[c] else 1
            n_sum += sep_cfg
            n_prod += clq_cfg
        self.n_sum_nodes = n_sum
        self.n_product_nodes = n_prod
        self.n_leaf_nodes = int(sum(int(self.cardinality[v]) for v in range(self.n_vars)))
        self.size = n_sum + n_prod + self.n_leaf_nodes

    # -- shared upward pass --------------------------------------------

    def _upward(self, op: str, ev: Dict[int, int]) -> Tuple[List[Table], List[Table]]:
        """Collect messages from leaves to roots.

        Returns per-clique ``local`` tables (potential × child messages) and
        ``up`` messages (``local`` reduced to the parent separator).
        """
        card = self.cardinality
        n_cl = len(self.cliques)
        local: List[Optional[Table]] = [None] * n_cl
        up: List[Optional[Table]] = [None] * n_cl

        for c in reversed(self.order):
            table = _apply_evidence(self.psi[c], ev)
            for d in self.children[c]:
                table = _mul(table, up[d], card)
            local[c] = table
            up[c] = _reduce(table, self.separator[c], op)

        return local, up  # type: ignore[return-value]

    # -- queries --------------------------------------------------------

    def probability(self, evidence: Optional[Dict[int, int]] = None) -> float:
        """Return ``P(evidence)`` (the partition function when no evidence)."""
        ev = self._normalize_evidence(evidence)
        _, up = self._upward("sum", ev)
        z = 1.0
        for r in self.roots:
            z *= float(np.asarray(up[r][1]).reshape(-1)[0])
        return z

    def marginals(
        self, evidence: Optional[Dict[int, int]] = None
    ) -> List[np.ndarray]:
        """Return exact marginal distributions for every variable."""
        ev = self._normalize_evidence(evidence)
        card = self.cardinality
        local, up = self._upward("sum", ev)

        down: List[Optional[Table]] = [None] * len(self.cliques)
        belief: List[Optional[Table]] = [None] * len(self.cliques)
        for r in self.roots:
            down[r] = ([], np.array(1.0))

        for c in self.order:
            if down[c] is None:
                down[c] = ([], np.array(1.0))
            belief[c] = _mul(local[c], down[c], card)
            for d in self.children[c]:
                # Message to child d: potential × down × siblings' up messages.
                msg = _apply_evidence(self.psi[c], ev)
                msg = _mul(msg, down[c], card)
                for e in self.children[c]:
                    if e != d:
                        msg = _mul(msg, up[e], card)
                down[d] = _reduce(msg, self.separator[d], "sum")

        out: List[np.ndarray] = []
        for v in range(self.n_vars):
            if v in ev:
                marg = np.zeros(int(card[v]))
                marg[ev[v]] = 1.0
                out.append(marg)
                continue
            c = self._home[v]
            _, arr = _reduce(belief[c], [v], "sum")
            arr = np.asarray(arr, dtype=float).reshape(-1)
            total = arr.sum()
            out.append(arr / total if total > 0 else np.ones(int(card[v])) / int(card[v]))
        return out

    def mpe(
        self, evidence: Optional[Dict[int, int]] = None
    ) -> Tuple[np.ndarray, float]:
        """Return the most probable explanation and its probability."""
        ev = self._normalize_evidence(evidence)
        local, _ = self._upward("max", ev)

        assignment = np.zeros(self.n_vars, dtype=int)
        for v, val in ev.items():
            assignment[v] = int(val)

        for c in self.order:
            fixed = {v: int(assignment[v]) for v in self.separator[c]}
            rem_scope, arr = _restrict(local[c], fixed)
            arr = np.asarray(arr, dtype=float)
            if rem_scope:
                flat = int(np.argmax(arr))
                idx = np.unravel_index(flat, arr.shape)
                for v, val in zip(rem_scope, idx):
                    if v not in ev:
                        assignment[v] = int(val)

        return assignment, self.probability({int(v): int(x) for v, x in enumerate(assignment)})

    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
        evidence: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """Draw exact i.i.d. samples from the compiled distribution."""
        if rng is None:
            rng = np.random.default_rng()
        ev = self._normalize_evidence(evidence)
        local, _ = self._upward("sum", ev)

        out = np.zeros((n_samples, self.n_vars), dtype=int)
        for s in range(n_samples):
            assignment: Dict[int, int] = dict(ev)
            for c in self.order:
                fixed = {v: int(assignment[v]) for v in self.separator[c] if v in assignment}
                rem_scope, arr = _restrict(local[c], fixed)
                arr = np.asarray(arr, dtype=float)
                if not rem_scope:
                    continue
                flat = arr.reshape(-1)
                total = flat.sum()
                if total <= 0:
                    probs = np.ones_like(flat) / flat.size
                else:
                    probs = flat / total
                choice = int(rng.choice(flat.size, p=probs))
                idx = np.unravel_index(choice, arr.shape)
                for v, val in zip(rem_scope, idx):
                    assignment[v] = int(val)
            for v in range(self.n_vars):
                out[s, v] = int(assignment.get(v, 0))
        return out

    # -- helpers --------------------------------------------------------

    def _normalize_evidence(
        self, evidence: Optional[Dict[int, int]]
    ) -> Dict[int, int]:
        if evidence is None:
            return {}
        ev = {int(k): int(v) for k, v in evidence.items()}
        for var, val in ev.items():
            if var < 0 or var >= self.n_vars:
                raise ValueError(f"Evidence variable out of range: {var}")
            if val < 0 or val >= int(self.cardinality[var]):
                raise ValueError(f"Evidence value out of range for variable {var}: {val}")
        return ev
