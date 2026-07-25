"""
Core BayesianNetwork class.

Represents a discrete Bayesian network as a directed acyclic graph (DAG)
with conditional probability distributions (CPDs) for each variable.

The BN is parameterised by:
  - n_vars     : number of random variables
  - cardinality: 1-D integer array of length n_vars giving the number of
                 states for each variable.

Typical workflow
----------------
>>> import numpy as np
>>> from bayes_nets import BayesianNetwork
>>> data = np.random.randint(0, 2, size=(200, 4))
>>> bn = BayesianNetwork(n_vars=4, cardinality=np.array([2, 2, 2, 2]))
>>> bn.fit(data, method="bic")
>>> samples = bn.sample(100)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class BayesianNetwork:
    """
    Discrete Bayesian Network.

    Parameters
    ----------
    n_vars : int
        Number of random variables in the network.
    cardinality : array-like of int
        Number of discrete states for each variable.

    Attributes
    ----------
    n_vars : int
    cardinality : np.ndarray
    adjacency : np.ndarray, shape (n_vars, n_vars)
        Adjacency matrix.  ``adjacency[i, j] == 1`` means there is a
        directed edge i → j (i is a parent of j).
    cpds : dict
        Maps variable index to a dict ``{"parents": [...], "cpd": np.ndarray}``.
        For a root variable the CPD is a 1-D probability vector of length
        ``cardinality[var]``.  For a variable with parents it is a 2-D array
        of shape ``(n_parent_configs, cardinality[var])`` where each row is a
        conditional probability distribution.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, n_vars: int, cardinality) -> None:
        if n_vars < 1:
            raise ValueError("n_vars must be >= 1")
        self.n_vars: int = n_vars
        self.cardinality: np.ndarray = np.asarray(cardinality, dtype=int)
        if self.cardinality.shape != (n_vars,):
            raise ValueError("cardinality must have length n_vars")
        if np.any(self.cardinality < 2):
            raise ValueError("every variable must have cardinality >= 2")

        self.adjacency: np.ndarray = np.zeros((n_vars, n_vars), dtype=int)
        self.cpds: Dict[int, Dict] = {}
        # ``None`` for dense tabular CPDs, or "dt"/"dg" when the CPDs use
        # decision-tree / decision-graph local structure.
        self.local_structure: Optional[str] = None

    # ------------------------------------------------------------------
    # Graph manipulation
    # ------------------------------------------------------------------

    def add_edge(self, parent: int, child: int) -> None:
        """Add a directed edge parent → child.

        Raises
        ------
        ValueError
            If the edge would introduce a cycle.
        """
        if parent == child:
            raise ValueError("Self-loops are not allowed")
        if self._would_create_cycle(parent, child):
            raise ValueError(
                f"Adding edge {parent} → {child} would create a cycle"
            )
        self.adjacency[parent, child] = 1

    def remove_edge(self, parent: int, child: int) -> None:
        """Remove the directed edge parent → child (if it exists)."""
        self.adjacency[parent, child] = 0

    def has_edge(self, parent: int, child: int) -> bool:
        """Return True if the edge parent → child exists."""
        return bool(self.adjacency[parent, child])

    def get_parents(self, var: int) -> List[int]:
        """Return list of parent indices for *var*."""
        return list(np.where(self.adjacency[:, var] > 0)[0])

    def get_children(self, var: int) -> List[int]:
        """Return list of child indices for *var*."""
        return list(np.where(self.adjacency[var, :] > 0)[0])

    def is_dag(self) -> bool:
        """Return True if the current structure is a valid DAG."""
        try:
            self._topological_sort()
            return True
        except ValueError:
            return False

    def topological_order(self) -> np.ndarray:
        """Return variable indices in topological order."""
        return self._topological_sort()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        method: str = "bic",
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
        ordering: Optional[np.ndarray] = None,
        limit_table_size: bool = True,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        local_structure: Optional[str] = None,
        max_leaves: Optional[int] = None,
        split_score: Optional[str] = None,
        **learn_kwargs,
    ) -> "BayesianNetwork":
        """Learn structure **and** parameters from *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
            Observed discrete data.  Values must be integers in
            ``[0, cardinality[j])`` for each column j.
        method : str
            Scoring / search algorithm.  Choices: ``"bic"``, ``"aic"``,
            ``"k2"``, ``"stable_hc"``, ``"tabu"``, ``"gs"``, ``"rcd"``,
            ``"rpcd"``, ``"pc"``, ``"stable_pc"``, ``"dt"``/
            ``"decision_tree"``, ``"dg"``/``"decision_graph"``, ``"sartre"``,
            ``"dmbbn"``, ... (see :meth:`learn_structure`).
        max_parents : int or None
            Maximum parents per variable.  ``None`` → rule of thumb
            ``max(1, floor(10·log2/log(max_cardinality)))``
            (10 for binary variables).
        alpha : float
            Dirichlet/Laplace smoothing parameter (>= 0).
        ordering : array-like of int, optional
            Legacy K2 ordering parameter.  Use ``permutation`` instead.
        limit_table_size : bool
            Skip parent sets whose joint table exceeds n_samples.
        permutation : array-like of int, optional
            Permutation σ of [0 … n_vars-1].  Parents of σ(j) are
            restricted to {σ(i) : i < j}.  Applies to all methods.
            ``None`` → unconstrained (cycle detection used for HC).
        interaction_matrix : np.ndarray, shape (n_vars, n_vars), optional
            Symmetric binary matrix.  Edge u → v considered only when
            ``interaction_matrix[u, v] == 1``.  ``None`` → all allowed.
        sample_weights : array of float, shape (n_samples,), optional
            Probability vector (must sum to 1).  Weighted counts replace
            raw counts during structure and parameter learning.
            ``None`` → uniform 1/N.
        local_structure : {None, "dt", "dg"}, optional
            When ``"dt"`` or ``"dg"``, the CPDs are represented with
            **decision-tree** or **decision-graph** local structure instead of
            dense tables.  Any base skeleton learner (``method``) can thus be
            composed with a compact, context-specific CPD representation; the
            learned decision graphs are also exploited when sampling.  ``None``
            (default) keeps the classical dense tabular CPDs.
        **learn_kwargs
            Extra keyword arguments forwarded to :meth:`learn_structure`
            (e.g. ``treewidth_bound``, ``seed``, ``fs_importance``).

        Returns
        -------
        self
        """
        data = np.asarray(data, dtype=int)
        self._validate_data(data)
        self._validate_new_params(data.shape[0], permutation, interaction_matrix, sample_weights)

        self.learn_structure(
            data,
            method=method,
            max_parents=max_parents,
            alpha=alpha,
            ordering=ordering,
            limit_table_size=limit_table_size,
            permutation=permutation,
            interaction_matrix=interaction_matrix,
            sample_weights=sample_weights,
            max_leaves=max_leaves,
            split_score=split_score,
            **learn_kwargs,
        )
        if local_structure is not None:
            self.learn_local_structure(
                data, structure=local_structure, alpha=alpha,
                sample_weights=sample_weights,
                max_leaves=max_leaves, split_score=split_score,
            )
        else:
            self.learn_parameters(data, alpha=alpha, sample_weights=sample_weights)
        return self

    def learn_structure(
        self,
        data: np.ndarray,
        method: str = "bic",
        max_parents: Optional[int] = None,
        alpha: float = 1.0,
        ordering: Optional[np.ndarray] = None,
        limit_table_size: bool = True,
        *,
        permutation: Optional[np.ndarray] = None,
        interaction_matrix: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        treewidth_bound: int = 2,
        n_ktrees: int = 100,
        seed: Optional[int] = None,
        fs_importance: str = "mutual_info",
        rfe_selector: str = "mrmr",
        initial_structure: Optional[np.ndarray] = None,
        candidate_parents=None,
        fast_local_scoring: bool = False,
        max_leaves: Optional[int] = None,
        split_score: Optional[str] = None,
        penalty="bic",
    ) -> "BayesianNetwork":
        """Learn the DAG structure from *data*.

        Resets any existing structure before learning.  See :meth:`fit`
        for full parameter documentation.

        Extra keyword-only parameters (all no-ops at their defaults, so existing
        callers are unaffected):

        initial_structure : (n_vars, n_vars) 0/1 array, optional
            Warm-start adjacency for ``method in {"stable_hc", "tabu"}`` — the
            add/delete/reverse search begins from this DAG instead of the empty
            graph (used by the EBNA family to warm-start each generation).
            Ignored (with a warning) for methods that cannot use it.
        candidate_parents : None or ``"mi:<k>"``
            When ``"mi:<k>"``, restrict the parent search to each variable's
            top-``k`` mutual-information neighbours (built with
            :func:`~bayes_nets.mi_candidate_mask` and AND-ed with any supplied
            ``interaction_matrix``).
        fast_local_scoring : bool
            For ``method in {"dt", "dg"}``, use the cached-statistics fast
            scorers (identical scores, lower wall-clock).
        max_leaves : int or None
            Cap on the local-structure leaf count (``dt``/``dg``/``dg_ndg``).
        split_score : {"mdl", "bic", "k2"} or None
            Split-gain criterion for the ``dg``/``dg_ndg`` local structure.
        penalty : {"bic", "aic"} or float
            Complexity-penalty weight ``f(N)`` for ``method="k2_pen"``.
        """
        from bayes_nets.structure_learning import (
            K2StructureLearner,
            GreedyHillClimbLearner,
            StableHillClimbLearner,
            TabuHillClimbLearner,
            GrowShrinkLearner,
            RecursiveCDLearner,
            RPCDLearner,
            mi_candidate_mask,
        )
        from bayes_nets.scoring import BICScoringMethod, AICScoringMethod

        data = np.asarray(data, dtype=int)

        # permutation overrides legacy ordering for K2
        eff_perm = permutation if permutation is not None else (
            np.asarray(ordering, dtype=int) if ordering is not None else None
        )

        method = method.lower()

        # Resolve candidate_parents="mi:<k>" into an interaction mask, AND-ed
        # with any user-supplied interaction_matrix.
        if candidate_parents is not None:
            if isinstance(candidate_parents, str) and candidate_parents.startswith("mi:"):
                try:
                    top_k = int(candidate_parents.split(":", 1)[1])
                except (ValueError, IndexError):
                    raise ValueError(
                        f"candidate_parents must be 'mi:<k>' with integer k; got "
                        f"'{candidate_parents}'."
                    )
                mi_mask = mi_candidate_mask(
                    data, self.n_vars, self.cardinality, top_k, sample_weights
                )
                if interaction_matrix is None:
                    interaction_matrix = mi_mask
                else:
                    interaction_matrix = (
                        (np.asarray(interaction_matrix) != 0).astype(int) & mi_mask
                    )
            else:
                raise ValueError(
                    "candidate_parents must be None or a string 'mi:<k>'."
                )

        # Warn when initial_structure is supplied to a method that ignores it.
        if initial_structure is not None and method not in ("stable_hc", "tabu"):
            import warnings
            warnings.warn(
                f"initial_structure is only used by methods 'stable_hc'/'tabu'; "
                f"ignored for method '{method}'.",
                stacklevel=2,
            )

        learn_kwargs = dict(
            permutation=eff_perm,
            interaction_matrix=interaction_matrix,
            sample_weights=sample_weights,
        )

        if method == "k2":
            learner = K2StructureLearner(
                max_parents=max_parents,
                alpha=alpha,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(data, self.n_vars, self.cardinality, **learn_kwargs)

        elif method in ("k2_mi", "k2_mb", "k2_refine", "k2_ensemble", "k2_plus"):
            from bayes_nets.structure_learning import K2VariantLearner
            # Named presets that layer the K2 improvements (see
            # K2_Improvements_Exploration.md).  All stay within ~5x base K2 time.
            presets = {
                "k2_mi":       dict(order_method="mi", parent_restriction=None),
                "k2_mb":       dict(order_method="given", parent_restriction="mb"),
                "k2_refine":   dict(order_method="mi", parent_restriction="mi", refine=True),
                "k2_ensemble": dict(order_method="mi", parent_restriction="mi", n_orderings=5),
                "k2_plus":     dict(order_method="mi", parent_restriction="mi", refine=True),
            }
            learner = K2VariantLearner(
                max_parents=max_parents, alpha=alpha,
                limit_table_size=limit_table_size, **presets[method],
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm, interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("bic", "aic"):
            scoring_cls = BICScoringMethod if method == "bic" else AICScoringMethod
            scoring = scoring_cls(alpha=alpha, sample_weights=sample_weights)
            learner = GreedyHillClimbLearner(
                scoring=scoring,
                max_parents=max_parents,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                # sample_weights already embedded in scoring object
            )

        elif method in ("stable_hc", "tabu"):
            scoring = BICScoringMethod(alpha=alpha, sample_weights=sample_weights)
            cls = StableHillClimbLearner if method == "stable_hc" else TabuHillClimbLearner
            learner = cls(
                scoring=scoring,
                max_parents=max_parents,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                initial_structure=initial_structure,
            )

        elif method in ("k2_pen", "ebna_k2"):
            from bayes_nets.scoring import K2PenScoringMethod, etxeberria_max_parents
            k2pen = K2PenScoringMethod(
                alpha=alpha, sample_weights=sample_weights, penalty=penalty
            )
            # Etxeberria automatic per-variable parent bound when max_parents is
            # left unspecified (the faithful EBNA_K2+pen behaviour).
            mp_arg = max_parents
            if mp_arg is None:
                f_N = k2pen.f_penalty(data.shape[0])
                mp_arg = etxeberria_max_parents(self.cardinality, data.shape[0], f_N)
            learner = GreedyHillClimbLearner(
                scoring=k2pen,
                max_parents=mp_arg,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
            )

        elif method in ("gs", "rcd", "rpcd"):
            alpha_ci = alpha if 0 < alpha < 1 else 0.05
            if method == "gs":
                learner = GrowShrinkLearner(
                    alpha_ci=alpha_ci,
                    max_parents=max_parents,
                    # Default cap: prevents O(n²) blow-up on dense/large graphs
                    max_conditioning_set_size=5,
                )
            else:
                learner_cls = RecursiveCDLearner if method == "rcd" else RPCDLearner
                learner = learner_cls(
                    alpha_ci=alpha_ci,
                    max_parents=max_parents,
                )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("pc", "stable_pc"):
            from bayes_nets.structure_learning import PCLearner, StablePCLearner
            cls_pc = PCLearner if method == "pc" else StablePCLearner
            learner = cls_pc(
                alpha_ci=alpha if 0 < alpha < 1 else 0.05,
                max_parents=max_parents,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("chow_liu", "tree", "branching",
                        "rebane_pearl", "polytree",
                        "lpa", "pada", "lpa_marginal",
                        "causal_polytree", "sheaf"):
            from bayes_nets.polytree_learning import (
                ChowLiuTreeLearner,
                RebanePearlPolytreeLearner,
                PolytreeLPALearner,
                CausalPolytreeLearner,
            )
            alpha_ci = alpha if 0 < alpha < 1 else 0.05
            if method in ("chow_liu", "tree", "branching"):
                learner = ChowLiuTreeLearner(alpha_ci=alpha_ci, max_parents=max_parents)
            elif method in ("rebane_pearl", "polytree"):
                learner = RebanePearlPolytreeLearner(
                    alpha_ci=alpha_ci, max_parents=max_parents
                )
            elif method in ("lpa", "pada", "lpa_marginal"):
                learner = PolytreeLPALearner(
                    alpha_ci=alpha_ci,
                    dep_mode="marginal" if method == "lpa_marginal" else "global",
                    max_parents=max_parents,
                )
            else:
                learner = CausalPolytreeLearner(
                    alpha_ci=alpha_ci, max_parents=max_parents
                )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("dt", "decision_tree"):
            from bayes_nets.structure_learning import DecisionTreeLearner
            learner = DecisionTreeLearner(
                max_parents=max_parents,
                alpha=alpha,
                max_leaves=max_leaves,
                fast_local_scoring=fast_local_scoring,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("dg", "decision_graph"):
            from bayes_nets.structure_learning import DecisionGraphLearner
            learner = DecisionGraphLearner(
                max_parents=max_parents,
                alpha=alpha,
                max_leaves=max_leaves,
                split_score=(split_score or "k2"),
                fast_local_scoring=fast_local_scoring,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("dg_ndg", "ndg"):
            from bayes_nets.structure_learning import DecisionGraphNDGLearner
            learner = DecisionGraphNDGLearner(
                max_parents=max_parents,
                alpha=alpha,
                max_leaves=max_leaves,
                split_score=(split_score or "mdl"),
                seed=seed,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method == "dmbbn":
            from bayes_nets.structure_learning import DMBBNStructureLearner
            learner = DMBBNStructureLearner(
                max_parents=max_parents,
                alpha=alpha,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("levelwise", "exact"):
            from bayes_nets.structure_learning import LevelWiseDPLearner
            learner = LevelWiseDPLearner(
                score="bic",
                alpha=alpha,
                max_parents=max_parents,
                limit_table_size=limit_table_size,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method == "sartre":
            from bayes_nets.structure_learning import SARTREPruner
            learner = SARTREPruner(max_parents=max_parents)
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("iterdsla", "iter_dsla"):
            from bayes_nets.structure_learning import IterDSLALearner
            learner = IterDSLALearner(
                score="bic",
                alpha=alpha,
                max_parents=max_parents,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method == "binotears":
            from bayes_nets.notears import BinaryNotearsLearner
            learner = BinaryNotearsLearner(max_parents=max_parents)
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("bounded_tw", "bounded_treewidth", "ktree"):
            from bayes_nets.structure_learning import BoundedTreewidthLearner
            learner = BoundedTreewidthLearner(
                k=treewidth_bound,
                n_ktrees=n_ktrees,
                score="bic",
                alpha=alpha,
                max_parents=max_parents,
                limit_table_size=limit_table_size,
                seed=seed,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("univ_bn", "univ", "independent"):
            from bayes_nets.structure_learning import IndependentBNLearner
            learner = IndependentBNLearner()
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                permutation=eff_perm,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("fi_k2", "fik2"):
            from bayes_nets.structure_learning import FeatureImportanceK2Learner
            learner = FeatureImportanceK2Learner(
                importance=fs_importance,
                max_parents=max_parents,
                alpha=alpha,
                limit_table_size=limit_table_size,
                seed=seed,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        elif method in ("rfe_k2", "rfek2"):
            from bayes_nets.structure_learning import RFEK2Learner
            learner = RFEK2Learner(
                selector=rfe_selector,
                max_parents=max_parents,
                alpha=alpha,
                limit_table_size=limit_table_size,
                seed=seed,
            )
            self.adjacency = learner.learn(
                data, self.n_vars, self.cardinality,
                interaction_matrix=interaction_matrix,
                sample_weights=sample_weights,
            )

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose 'bic', 'aic', 'k2', 'k2_pen', "
                "'k2_mi'/'k2_mb'/'k2_refine'/'k2_ensemble'/'k2_plus', "
                "'stable_hc', 'tabu', "
                "'gs', 'rcd', 'rpcd', 'pc', 'stable_pc', "
                "'dt'/'decision_tree', 'dg'/'decision_graph', 'dg_ndg', "
                "'dmbbn', 'levelwise'/'exact', 'sartre', "
                "'iterdsla', 'binotears', 'bounded_tw', "
                "'univ_bn', 'fi_k2', or 'rfe_k2'."
            )

        self.cpds = {}
        return self

    def learn_parameters(
        self,
        data: np.ndarray,
        alpha: float = 1.0,
        *,
        sample_weights: Optional[np.ndarray] = None,
        parameter_learner=None,
    ) -> "BayesianNetwork":
        """Estimate CPDs from *data* given the current structure.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        alpha : float
            Dirichlet/Laplace smoothing (used by the default MLE learner).
        sample_weights : array of float, shape (n_samples,), optional
            Probability vector (must sum to 1).
        parameter_learner : object, optional
            A parameter learner exposing ``learn(data, n_vars, cardinality,
            adjacency, sample_weights=...)`` and returning the CPD dict.
            Defaults to :class:`MLEParameterLearner`.  Pass a
            :class:`LogisticRegressionParameterLearner` to use the
            regression-based CPD estimator for dense networks.
        """
        from bayes_nets.parameter_learning import MLEParameterLearner

        data = np.asarray(data, dtype=int)
        learner = parameter_learner if parameter_learner is not None else MLEParameterLearner(alpha=alpha)
        self.cpds = learner.learn(
            data, self.n_vars, self.cardinality, self.adjacency,
            sample_weights=sample_weights,
        )
        return self

    def learn_local_structure(
        self,
        data: np.ndarray,
        structure: str = "dt",
        *,
        alpha: float = 1.0,
        max_depth: Optional[int] = None,
        sample_weights: Optional[np.ndarray] = None,
        max_leaves: Optional[int] = None,
        split_score: Optional[str] = None,
    ) -> "BayesianNetwork":
        """Fit decision-tree / decision-graph CPDs for the current structure.

        Given the DAG already stored in :attr:`adjacency` (learned by any base
        skeleton learner), this estimates a compact **decision-tree**
        (``structure="dt"``) or **decision-graph** (``structure="dg"``)
        conditional probability distribution for every variable, instead of a
        dense table.  Each ``cpds[var]`` gains a ``"local"`` entry holding a
        :class:`bayes_nets.local_structure.LocalStructureCPD`; the dense
        ``"cpd"`` table is still filled when it is small enough, so existing
        table-based code keeps working, while :meth:`sample` exploits the
        compact local structure directly.

        This is the composable primitive behind
        ``fit(method=..., local_structure="dg")``: it decouples the choice of
        skeleton learner from the CPD representation.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_vars)
        structure : {"dt", "dg"}
            Decision tree or decision graph.
        alpha : float
            Laplace / Dirichlet smoothing.
        max_depth : int or None
            Maximum decision-tree depth (``None`` → grow until no split helps).
        sample_weights : array of float, shape (n_samples,), optional
            Probability vector over rows (must sum to 1).

        References
        ----------
        Friedman & Goldszmidt (1996); Chickering, Heckerman & Meek (1997).
        """
        from bayes_nets.local_structure import LocalStructureParameterLearner

        data = np.asarray(data, dtype=int)
        learner = LocalStructureParameterLearner(
            method=structure, alpha=alpha, max_depth=max_depth,
            max_leaves=max_leaves, split_score=split_score,
        )
        self.cpds = learner.learn(
            data, self.n_vars, self.cardinality, self.adjacency,
            sample_weights=sample_weights,
        )
        self.local_structure = structure
        return self

    def has_local_structure(self) -> bool:
        """True if the CPDs use decision-tree / decision-graph local structure."""
        return bool(self.cpds) and any(
            "local" in self.cpds[v] for v in self.cpds
        )

    def learn_variable_clustering(
        self,
        data: np.ndarray,
        *,
        alpha: float = 1.0,
        stop_threshold: float = 0.0,
        max_config: Optional[int] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> dict:
        """Cluster variables into a linkage tree via Bayesian model comparison.

        Returns the flat marginal-product clustering and the merge history
        (see :func:`bayes_nets.structure_learning.bayesian_variable_clustering`).
        This does not modify the network structure; it produces an informative
        variable grouping useful for EDA linkage / marginal-product sampling.

        References
        ----------
        Marrelec, Messé & Bellec (2015), PLoS ONE 10(9): e0137278.
        """
        from bayes_nets.structure_learning import bayesian_variable_clustering

        data = np.asarray(data, dtype=int)
        return bayesian_variable_clustering(
            data, self.cardinality,
            sample_weights=sample_weights,
            alpha=alpha,
            stop_threshold=stop_threshold,
            max_config=max_config,
        )

    def to_circuit(self, method: str = "min-fill"):
        """Compile the network into a tractable arithmetic circuit / SPN.

        The returned :class:`bayes_nets.circuits.ArithmeticCircuit` answers
        marginal, MPE and sampling queries exactly in time linear in the
        circuit size.  Requires CPDs (call :meth:`fit` /
        :meth:`learn_parameters` first).

        References
        ----------
        Vergari, Di Mauro & Esposito (2016), Machine Learning 108, 551-573.
        """
        from bayes_nets.circuits import ArithmeticCircuit

        return ArithmeticCircuit.from_bayesian_network(self, method=method)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw *n_samples* i.i.d. samples from the BN.

        Uses probabilistic logic sampling (ancestral sampling).

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray, shape (n_samples, n_vars)
        """
        if not self.cpds:
            raise RuntimeError(
                "CPDs have not been estimated yet. Call learn_parameters() or fit() first."
            )
        # When the CPDs carry decision-tree / decision-graph local structure,
        # sample directly from it (routing parent configs to leaves), which
        # exploits the compact representation and also works when a dense table
        # was too large to materialise.
        if self.has_local_structure():
            from bayes_nets.sampling import LocalStructureSampler

            return LocalStructureSampler().sample(
                n_samples=n_samples,
                n_vars=self.n_vars,
                cardinality=self.cardinality,
                adjacency=self.adjacency,
                cpds=self.cpds,
                rng=rng,
            )

        from bayes_nets.sampling import ProbabilisticLogicSampler

        sampler = ProbabilisticLogicSampler()
        return sampler.sample(
            n_samples=n_samples,
            n_vars=self.n_vars,
            cardinality=self.cardinality,
            adjacency=self.adjacency,
            cpds=self.cpds,
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Visualise the BN structure using matplotlib/networkx.

        Keyword arguments are forwarded to
        :func:`bayes_nets.visualization.plot_bayesian_network`.
        """
        from bayes_nets.visualization import plot_bayesian_network

        return plot_bayesian_network(self, **kwargs)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def to_adjacency_matrix(self) -> np.ndarray:
        """Return a copy of the adjacency matrix."""
        return self.adjacency.copy()

    def big_matrix(self) -> np.ndarray:
        """Alias for the adjacency matrix used by visualization pipelines."""
        return self.to_adjacency_matrix()

    def edge_list(self) -> List[tuple[int, int]]:
        """Return directed edges as (parent, child) tuples."""
        edges = np.argwhere(self.adjacency > 0)
        return [(int(parent), int(child)) for parent, child in edges]

    def to_run_structure(self, generation: int, run: int = 0) -> Dict:
        """Return a run-structure entry compatible with EDA visualizers."""
        return {
            "adjacency": self.big_matrix(),
            "generation": int(generation),
            "run": int(run),
        }

    def set_structure(self, adjacency: np.ndarray) -> "BayesianNetwork":
        """Set the graph structure from an adjacency matrix.

        Parameters
        ----------
        adjacency : np.ndarray, shape (n_vars, n_vars)
            ``adjacency[i, j] == 1`` ⟺ edge i → j.
        """
        adj = np.asarray(adjacency, dtype=int)
        if adj.shape != (self.n_vars, self.n_vars):
            raise ValueError("adjacency must have shape (n_vars, n_vars)")
        self.adjacency = adj.copy()
        self.cpds = {}
        return self

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def n_parameters(self) -> int:
        """Total number of free parameters in the CPDs.

        When the CPDs use decision-tree / decision-graph local structure the
        count reflects the *compact* representation (distinct leaves per
        variable), which is what makes local structure more parameter-efficient
        than dense tables.
        """
        total = 0
        for var in range(self.n_vars):
            local = self.cpds.get(var, {}).get("local") if self.cpds else None
            if local is not None:
                total += local.n_parameters
                continue
            parents = self.get_parents(var)
            k = int(self.cardinality[var])
            n_parent_configs = int(
                np.prod([self.cardinality[p] for p in parents]) if parents else 1
            )
            total += n_parent_configs * (k - 1)
        return total

    def marginal(self, var: int, data: np.ndarray) -> np.ndarray:
        """Empirical marginal distribution of *var* in *data*.

        Parameters
        ----------
        var : int
        data : np.ndarray, shape (n_samples, n_vars)

        Returns
        -------
        np.ndarray, shape (cardinality[var],)
        """
        k = int(self.cardinality[var])
        counts = np.bincount(data[:, var], minlength=k).astype(float)
        return counts / counts.sum()

    def structure_score(self, data: np.ndarray, score: str = "bic", alpha: float = 0.0) -> float:
        """Score the current structure on data using BIC/AIC/K2."""
        from bayes_nets.scoring import BICScoringMethod, AICScoringMethod, K2ScoringMethod

        method = score.lower()
        if method == "bic":
            scorer = BICScoringMethod(alpha=alpha)
        elif method == "aic":
            scorer = AICScoringMethod(alpha=alpha)
        elif method == "k2":
            scorer = K2ScoringMethod(alpha=max(alpha, 1e-12))
        else:
            raise ValueError("score must be 'bic', 'aic', or 'k2'")
        return float(scorer.score(self.adjacency, np.asarray(data, dtype=int), self.cardinality))

    def markov_blanket(self, var: int) -> List[int]:
        """Return Markov blanket: parents, children, and children's other parents."""
        parents = set(self.get_parents(var))
        children = set(self.get_children(var))
        spouses: set[int] = set()
        for child in children:
            spouses.update(self.get_parents(child))
        spouses.discard(var)
        blanket = sorted(parents.union(children).union(spouses))
        return [int(v) for v in blanket]

    def variable_dependencies(self, data: np.ndarray, score: str = "bic", alpha: float = 0.0) -> Dict:
        """Return dependency analysis summary compatible with EDA consumers."""
        data = np.asarray(data, dtype=int)
        n_vars = data.shape[1]
        mi = np.zeros((n_vars, n_vars), dtype=float)

        for i in range(n_vars):
            xi = data[:, i]
            card_i = int(self.cardinality[i])
            p_i = np.bincount(xi, minlength=card_i).astype(float)
            p_i /= p_i.sum()
            for j in range(i + 1, n_vars):
                xj = data[:, j]
                card_j = int(self.cardinality[j])

                pair_idx = xi + card_i * xj
                p_ij = np.bincount(pair_idx, minlength=card_i * card_j).astype(float)
                p_ij = p_ij.reshape(card_j, card_i).T
                p_ij /= p_ij.sum()

                p_j = np.bincount(xj, minlength=card_j).astype(float)
                p_j /= p_j.sum()

                val = 0.0
                for ai in range(card_i):
                    for aj in range(card_j):
                        if p_ij[ai, aj] > 0 and p_i[ai] > 0 and p_j[aj] > 0:
                            val += p_ij[ai, aj] * np.log2(p_ij[ai, aj] / (p_i[ai] * p_j[aj]))
                mi[i, j] = val
                mi[j, i] = val

        return {
            "adjacency_matrix": self.big_matrix(),
            "edges": self.edge_list(),
            "score": self.structure_score(data, score=score, alpha=alpha),
            "mi_matrix": mi,
            "parents": {var: self.get_parents(var) for var in range(self.n_vars)},
            "markov_blankets": {var: self.markov_blanket(var) for var in range(self.n_vars)},
            "degree": {
                var: int(self.adjacency[var, :].sum() + self.adjacency[:, var].sum())
                for var in range(self.n_vars)
            },
        }

    def to_factorization(
        self,
        data: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        max_clique_width: Optional[int] = None,
        width_control: str = "split",
        triangulation_method: str = "min-fill",
    ):
        """Convert the BN into a clique factorization for FDA-style sampling."""
        from bayes_nets.factorization import bn_to_factorization

        return bn_to_factorization(
            adjacency=self.adjacency,
            cardinality=self.cardinality,
            cpds=self.cpds,
            data=data,
            alpha=alpha,
            max_clique_width=max_clique_width,
            width_control=width_control,
            triangulation_method=triangulation_method,
        )

    def most_probable_config(self, evidence: Optional[Dict[int, int]] = None):
        """Return the most probable assignment and its probability."""
        from bayes_nets.inference import MaxProductInference

        return MaxProductInference(self).most_probable_config(evidence=evidence)

    def k_most_probable_configs(
        self,
        k: int,
        evidence: Optional[Dict[int, int]] = None,
        search_method: str = "nilsson",
    ):
        """Return top-k assignments sorted by decreasing probability."""
        from bayes_nets.inference import MaxProductInference

        return MaxProductInference(self).k_most_probable_configs(
            k=k,
            evidence=evidence,
            search_method=search_method,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_edges = int(self.adjacency.sum())
        return (
            f"BayesianNetwork(n_vars={self.n_vars}, "
            f"cardinality={self.cardinality.tolist()}, "
            f"n_edges={n_edges})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_data(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError("data must be a 2-D array (n_samples, n_vars)")
        if data.shape[1] != self.n_vars:
            raise ValueError(
                f"data has {data.shape[1]} columns but n_vars={self.n_vars}"
            )

    def _validate_new_params(
        self,
        n_samples: int,
        permutation: Optional[np.ndarray],
        interaction_matrix: Optional[np.ndarray],
        sample_weights: Optional[np.ndarray],
    ) -> None:
        if permutation is not None:
            p = np.asarray(permutation, dtype=int)
            if p.shape != (self.n_vars,) or set(p.tolist()) != set(range(self.n_vars)):
                raise ValueError(
                    "permutation must be a valid permutation of [0, ..., n_vars-1]"
                )
        if interaction_matrix is not None:
            im = np.asarray(interaction_matrix)
            if im.shape != (self.n_vars, self.n_vars):
                raise ValueError(
                    f"interaction_matrix must have shape ({self.n_vars}, {self.n_vars})"
                )
            if not np.array_equal(im, im.T):
                raise ValueError("interaction_matrix must be symmetric")
        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=float)
            if sw.shape != (n_samples,):
                raise ValueError(
                    f"sample_weights must have shape ({n_samples},)"
                )
            if sw.min() < 0:
                raise ValueError("sample_weights must be non-negative")
            total = sw.sum()
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(
                    f"sample_weights must sum to 1 (got {total:.6f})"
                )

    def _would_create_cycle(self, parent: int, child: int) -> bool:
        """Return True if adding edge parent → child would create a cycle."""
        n = self.n_vars
        visited = np.zeros(n, dtype=bool)
        stack = [child]
        while stack:
            node = stack.pop()
            if node == parent:
                return True
            if visited[node]:
                continue
            visited[node] = True
            for nxt in range(n):
                if self.adjacency[node, nxt] and not visited[nxt]:
                    stack.append(nxt)
        return False

    def _topological_sort(self) -> np.ndarray:
        """Kahn's algorithm for topological sorting."""
        n = self.n_vars
        in_degree = self.adjacency.sum(axis=0).copy()
        queue = list(np.where(in_degree == 0)[0])
        order: List[int] = []
        while queue:
            var = queue.pop(0)
            order.append(var)
            for child in range(n):
                if self.adjacency[var, child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        if len(order) != n:
            raise ValueError("Graph contains a cycle – not a valid DAG")
        return np.array(order, dtype=int)
