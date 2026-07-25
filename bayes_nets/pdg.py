"""
Probabilistic Decision Graphs (PDG) — representation skeleton and BN compiler.

This module provides the *interface* requested as a Priority-3 (Part B) item of
``docs/Fast_DG_Learning.md``: a :class:`ProbabilisticDecisionGraph` container and
a :func:`bn_to_pdg` compiler that turns a :class:`bayes_nets.BayesianNetwork`
into a PDG.  It is deliberately a **skeleton**: it fixes the class shape,
attributes and constructor semantics so downstream code and later work can rely
on them, while the heavier inference operations (variable elimination on the
PDG, structure learning of PDGs) are left as clearly marked extension points.

A PDG (Jaeger 2004; Jaeger, Nielsen & Silander 2006) represents a joint
distribution by attaching, to each variable of an underlying forest of variable
nodes, a set of *parameter nodes*; every parameter node stores a distribution
over the variable and each of its outgoing edges (one per value) points to a
parameter node of the next variable.  When compiled from a Bayesian network,
each variable's parameter nodes correspond to the distinct rows of its
(local-structure) CPD, so context-specific independence already captured by a
decision-tree / decision-graph CPD carries over directly.

References
----------
Jaeger (2004). "Probabilistic Decision Graphs — Combining Verification and AI
Techniques for Probabilistic Inference." Int. J. Uncertainty, Fuzziness and
Knowledge-Based Systems.
Jaeger, Nielsen & Silander (2006). "Learning probabilistic decision graphs."
Int. J. Approximate Reasoning.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class PDGParameterNode:
    """One parameter node of a PDG: a distribution over a single variable.

    Attributes
    ----------
    variable : int
        Index of the variable this node parameterises.
    dist : np.ndarray, shape (var_card,)
        The conditional distribution stored at this node.
    """

    __slots__ = ("variable", "dist")

    def __init__(self, variable: int, dist: np.ndarray) -> None:
        self.variable = int(variable)
        self.dist = np.asarray(dist, dtype=float)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PDGParameterNode(variable={self.variable}, dist={self.dist})"


class ProbabilisticDecisionGraph:
    """Probabilistic Decision Graph representation (skeleton).

    Parameters
    ----------
    n_vars : int
        Number of variables.
    cardinality : array-like of int, shape (n_vars,)
        States per variable.
    order : list of int, optional
        A topological variable order (parents before children).  Defaults to
        ``range(n_vars)``.

    Attributes
    ----------
    parameter_nodes : dict[int, list[PDGParameterNode]]
        Per-variable list of parameter nodes (the distinct CPD rows).
    parents : dict[int, list[int]]
        Parent variables of each variable (from the source BN).
    reach : dict[int, object]
        Per-variable routing information used to select a parameter node from a
        parent configuration (here, the compact ``LocalStructureCPD`` when the
        source CPD had local structure, else ``None`` for a dense table).

    Notes
    -----
    This is the agreed **class skeleton**.  Constructing and inspecting a PDG is
    fully supported; :meth:`log_probability` is implemented for complete
    assignments, while :meth:`marginal` and :meth:`sample` are extension points
    that raise :class:`NotImplementedError`.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality,
        order: Optional[List[int]] = None,
    ) -> None:
        self.n_vars = int(n_vars)
        self.cardinality = np.asarray(cardinality, dtype=int)
        self.order = list(order) if order is not None else list(range(n_vars))
        self.parameter_nodes: Dict[int, List[PDGParameterNode]] = {v: [] for v in range(n_vars)}
        self.parents: Dict[int, List[int]] = {v: [] for v in range(n_vars)}
        self.reach: Dict[int, object] = {v: None for v in range(n_vars)}

    # ------------------------------------------------------------------
    # Sizes
    # ------------------------------------------------------------------

    @property
    def n_parameter_nodes(self) -> int:
        """Total number of parameter nodes across all variables."""
        return sum(len(nodes) for nodes in self.parameter_nodes.values())

    def _row_distribution(self, var: int, assignment: np.ndarray) -> np.ndarray:
        """Return P(var | parents) for a complete *assignment* (1-D int array)."""
        local = self.reach[var]
        parents = self.parents[var]
        if local is not None:
            pv = assignment[parents].reshape(1, -1) if parents else np.zeros((1, 0), dtype=int)
            return local.prob_matrix(pv)[0]
        # Dense fallback: a single parameter node (marginal) when no routing.
        nodes = self.parameter_nodes[var]
        return nodes[0].dist

    def log_probability(self, assignment) -> float:
        """Return ``log P(assignment)`` for a complete variable assignment.

        Parameters
        ----------
        assignment : array-like of int, shape (n_vars,)
        """
        x = np.asarray(assignment, dtype=int)
        if x.shape[0] != self.n_vars:
            raise ValueError(f"assignment must have length {self.n_vars}.")
        logp = 0.0
        for var in self.order:
            dist = self._row_distribution(var, x)
            p = float(dist[x[var]])
            logp += np.log(p) if p > 0 else -np.inf
        return logp

    def marginal(self, *args, **kwargs):  # pragma: no cover - extension point
        raise NotImplementedError(
            "PDG marginal inference is an extension point; compile to a "
            "junction tree or use BayesianNetwork inference for now."
        )

    def sample(self, *args, **kwargs):  # pragma: no cover - extension point
        raise NotImplementedError(
            "PDG ancestral sampling is an extension point; sample the source "
            "BayesianNetwork (which the PDG was compiled from) instead."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ProbabilisticDecisionGraph(n_vars={self.n_vars}, "
                f"n_parameter_nodes={self.n_parameter_nodes})")


def bn_to_pdg(bn) -> ProbabilisticDecisionGraph:
    """Compile a :class:`bayes_nets.BayesianNetwork` into a PDG.

    Each variable's parameter nodes are the distinct rows of its CPD.  When the
    BN uses decision-tree / decision-graph local structure (``bn.cpds[v]`` has a
    ``"local"`` entry), the compact ``LocalStructureCPD`` is reused as the
    routing structure and its distinct leaf distributions become the parameter
    nodes — so context-specific independence transfers to the PDG directly.
    For dense CPDs the distinct table rows are used.

    Parameters
    ----------
    bn : BayesianNetwork
        A fitted network (structure **and** parameters learned).

    Returns
    -------
    ProbabilisticDecisionGraph
    """
    if not getattr(bn, "cpds", None):
        raise ValueError("bn_to_pdg requires a BayesianNetwork with learned CPDs.")

    order = bn.topological_order() if hasattr(bn, "topological_order") else list(range(bn.n_vars))
    pdg = ProbabilisticDecisionGraph(bn.n_vars, bn.cardinality, order=order)

    for var in range(bn.n_vars):
        entry = bn.cpds[var]
        parents = list(entry.get("parents", []))
        pdg.parents[var] = parents
        local = entry.get("local")
        if local is not None:
            pdg.reach[var] = local
            for row in local.leaf_probs:
                pdg.parameter_nodes[var].append(PDGParameterNode(var, row))
        else:
            table = entry.get("cpd")
            if table is None:
                raise ValueError(
                    f"variable {var} has neither a dense CPD nor local structure."
                )
            table = np.asarray(table, dtype=float)
            rows = table.reshape(-1, table.shape[-1]) if table.ndim > 1 else table.reshape(1, -1)
            # distinct rows become parameter nodes
            seen = set()
            for row in rows:
                key = tuple(np.round(row, 12))
                if key not in seen:
                    seen.add(key)
                    pdg.parameter_nodes[var].append(PDGParameterNode(var, row))
    return pdg
