# ROADMAP

This document describes planned future development for the `bayes_nets` library.

---

## Completed (v0.1.0)

- [x] Core `BayesianNetwork` class with discrete variable support
- [x] BIC, AIC, and K2 scoring metrics for structure learning
- [x] K2 algorithm (ordering-based greedy search)
- [x] Greedy hill-climbing structure learning
- [x] MLE parameter estimation with Dirichlet smoothing
- [x] Probabilistic logic sampling (ancestral/forward sampling)
- [x] BN structure visualisation (`matplotlib` + `networkx`)
- [x] Marginal probability bar-chart visualisation

---

## Near-term (v0.2.0)

### Belief-propagation sampling
Implement sampling and inference based on the junction-tree / message-passing
algorithm.  This will enable:
- Exact marginal and conditional inference
- Evidence propagation (soft and hard evidence)
- Foundation for likelihood-weighted sampling

### Most probable configuration (MPC)
- Viterbi-style max-product algorithm to find the single most probable
  assignment to all variables.
- Extension to *k*-MPC: enumerate the k highest-probability assignments.

### Additional structure-learning metrics
- **BGe / BDe**: score variants that handle incomplete data more gracefully.
- **Mutual information (MI) tests**: independence-test-based constraint
  approach (PC algorithm skeleton).
- **Log-likelihood ratio / χ² conditional independence tests**.

---

## Medium-term (v0.3.0)

### Triangulation and junction-tree factorisation
- Methods to triangulate (moralise + triangulate) a BN and produce a
  chordal graph.
- Extract a junction tree (clique tree) from the triangulated graph.
- Represent the joint distribution as a product of clique potentials.
- Prune the junction tree if necessary to keep the maximum clique size
  within a user-specified bound.

### Constrained tree-width BNs
- Learn BN structures whose tree-width is bounded by a parameter *w*.
- Adapt structure-learning search to enforce the tree-width constraint at
  each step.
- Leverage the connection between tree-width and tractable inference.
- Reference: Korhonen & Parviainen (2013), "Exact Learning of Bounded
  Tree-width Bayesian Networks."

### Hierarchical Bayesian networks
- Add support for latent (hidden) variables and hierarchical plate
  notation.
- Parameter learning for hierarchical models via EM.
- Useful for modelling multi-level dependencies in EDA populations.

---

## Longer-term (v0.4.0+)

### Advanced visualisation
- Interactive graph viewer (e.g. via `pyvis` or `plotly`).
- Conditional probability heatmaps for variables with many parent
  configurations.
- Visualise inference results (posterior marginals) overlaid on the BN
  structure.
- Export to GraphViz `.dot` format for publication-quality diagrams.

### Parallel and GPU-accelerated scoring
- Vectorise local-score computations across all variables for faster
  structure learning on large problems.
- Optional GPU backend (CuPy / PyTorch) for count aggregation.

### Incremental / online learning
- Update CPDs incrementally as new data arrive without full retraining.
- Sliding-window structure adaptation for non-stationary EDA runs.

### Model persistence
- Save / load BNs in a portable format (JSON, HDF5, or BIF – the Bayesian
  Interchange Format used by many BN tools).

### Integration with EDA framework
- Provide `LearningMethod` and `SamplingMethod` adapters compatible with
  the `pateda` component interface so that `bayes_nets` can be used as a
  direct plug-in replacement for pgmpy within `pateda`-based EDAs.

---

## References

* Korhonen, J., & Parviainen, P. (2013). Exact learning of bounded tree-width
  Bayesian networks. *UAI 2013*.
* Jensen, F. V. (2001). *Bayesian Networks and Decision Graphs*. Springer.
* Lauritzen, S. L., & Spiegelhalter, D. J. (1988). Local computations with
  probabilities on graphical structures and their application to expert
  systems. *JRSS-B*, 50(2), 157–194.
* Pelikan, M. (2005). *Hierarchical Bayesian Optimization Algorithm*.
  Springer.
* Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*.
  MIT Press.
