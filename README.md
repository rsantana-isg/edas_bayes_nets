# bayes_nets

A lightweight, standalone Python library for **learning**, **sampling**, and **visualizing** discrete Bayesian networks (BNs).

Designed as a drop-in replacement for [pgmpy](https://pgmpy.org/) within estimation-of-distribution algorithm (EDA) workflows, while remaining fully usable as a general-purpose BN toolkit.

---

## Goals

* **Discrete representation** – all variables take a finite number of states; each variable's cardinality is specified at construction time.
* **Multiple structure-learning algorithms** – BIC, AIC, and K2 scoring with greedy hill-climbing or the K2 algorithm.
* **Probabilistic logic sampling** – forward (ancestral) sampling from a learned BN.
* **EDA integration** – the library is designed to work seamlessly with the `eda_code` learning and sampling modules as a replacement for pgmpy.
* **Visualization** – plot BN structures and marginal/conditional probability distributions.

---

## Installation

```bash
# Clone the repository and install the package
git clone https://github.com/rsantana-isg/edas_bayes_nets.git
cd edas_bayes_nets
pip install -e .
```

**Core dependencies**

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `scipy` | `gammaln` for K2 scoring |

**Optional dependencies** (needed for visualization)

| Package | Purpose |
|---------|---------|
| `matplotlib` | Plotting |
| `networkx` | Graph layout |
| `pygraphviz` | Graphviz-based layout (`dot` programme) |

---

## Quick start

```python
import numpy as np
from bayes_nets import BayesianNetwork

# ── 1. Create a BN for 5 binary variables ──────────────────────────────
bn = BayesianNetwork(n_vars=5, cardinality=np.array([2, 2, 2, 2, 2]))

# ── 2. Simulate some data ──────────────────────────────────────────────
rng = np.random.default_rng(42)
data = rng.integers(0, 2, size=(500, 5))

# ── 3. Learn structure and parameters with BIC ──────────────────────────
bn.fit(data, method="bic", max_parents=2)

# ── 4. Inspect the learned structure ───────────────────────────────────
print(bn)
# BayesianNetwork(n_vars=5, cardinality=[2, 2, 2, 2, 2], n_edges=3)

print("Parents of X3:", bn.get_parents(3))

# ── 5. Draw samples from the BN ────────────────────────────────────────
samples = bn.sample(n_samples=200)
print(samples.shape)   # (200, 5)

# ── 6. Visualise ───────────────────────────────────────────────────────
fig = bn.plot(title="Learned BN (BIC)")
fig.savefig("bn_structure.png")
```

---

## Scoring metrics

### BIC (Bayesian Information Criterion)

Balances goodness-of-fit against model complexity:

```
BIC = log P(D | θ_ML, G)  −  (k / 2) · log(n)
```

where *k* is the number of free parameters and *n* is the sample size.
The penalty term grows with *n*, making BIC more conservative for large datasets.

### AIC (Akaike Information Criterion)

Uses a lighter penalty:

```
AIC = log P(D | θ_ML, G)  −  k
```

### K2

Bayesian scoring metric based on the Dirichlet-multinomial marginal likelihood:

```
K2(X_i, Pa_i) = Σ_j [  Γ(α)  /  Γ(N_ij + α)
                       ·  Π_k  Γ(N_ijk + α/r_i) / Γ(α/r_i)  ]
```

where *α* is the equivalent sample size of the Dirichlet prior,
*r_i* is the cardinality of X_i, *N_ij* is the count of samples
matching parent configuration *j*, and *N_ijk* is the joint count for
X_i = k and parent config j.

---

## Structure learning algorithms

### `K2StructureLearner`

Uses the K2 algorithm (Cooper & Herskovits, 1992).  A **variable ordering** must be provided; each variable may only have parents that appear earlier in the ordering, which guarantees acyclicity.

```python
from bayes_nets import BayesianNetwork
import numpy as np

bn = BayesianNetwork(n_vars=4, cardinality=np.full(4, 3))
bn.learn_structure(data, method="k2", ordering=np.array([0, 2, 1, 3]))
```

### `GreedyHillClimbLearner`

Unconstrained greedy hill-climbing with BIC or AIC scoring.  No ordering needed; cycle detection is performed explicitly.

```python
bn.learn_structure(data, method="bic", max_parents=3)
```

### Polytree (singly connected) learners

A **polytree** is a DAG whose skeleton has no undirected cycle, so at most one
path connects any two variables.  Polytrees admit exact linear-time inference
and need far fewer samples than general BNs, which is why they are the model
class of the Polytree Approximation Distribution Algorithm (PADA / FDA-SC).
These learners live in `bayes_nets/polytree_learning.py` and were implemented
from the papers in `docs/PADA/`:

| Method | Class / `method=` | Paper |
|--------|-------------------|-------|
| **Chow-Liu branching** — max-weight spanning forest over mutual information; a provably bounded approximation to the optimal polytree | `ChowLiuTreeLearner` / `"chow_liu"` | Chow & Liu (1968); Dasgupta (1999) |
| **Rebane-Pearl** — Chow-Liu skeleton + collider orientation from marginal independence | `RebanePearlPolytreeLearner` / `"rebane_pearl"` | Rebane & Pearl (1987) |
| **LPA** — edges ranked by the global dependency degree `DepG(a,b) = min(Dep(a,b), min_c Dep(a,b|c))`, oriented by comparing dependency before/after instantiating the middle node | `PolytreeLPALearner` / `"lpa"`, `"lpa_marginal"` | Ochoa, Mühlenbein & Soto (2000) |
| **Sheaf-based causal polytree** — incremental node insertion using only marginal and first-order CI tests, O(n²) | `CausalPolytreeLearner` / `"causal_polytree"` | Huete & de Campos (1993) |

```python
# LPA, the learner used inside PADA / FDA-SC
bn.learn_structure(data, method="lpa", alpha=0.05)

# quadratic variant: rank by marginal dependency only
bn.learn_structure(data, method="lpa_marginal")
```

All four guarantee a singly connected result. `alpha` sets the significance
level of the independence tests; the LPA thresholds `e0`/`e1` are derived from
it and scale as `1/N`, reproducing the population-size dependence the PADA
paper requires.  `"chow_liu"` restricts every node to a single parent
(a *branching*); the other three can represent head-to-head patterns.

Note that a pure XOR-style collider is invisible to any of these learners:
each parent is then *marginally* independent of the child, so no pairwise
dependency measure can place the edge (Dasgupta 1999, Example 2).

### Advanced learners from the recent literature

These learners were implemented from the papers surveyed in
[`Other_BN_Papers_Ranked.md`](Other_BN_Papers_Ranked.md) (sources in
`docs/Other_BN_Structure_Learning_Methods/`).  Each is available both as a
class and through `learn_structure(method=...)`:

| Method | Class / `method=` | Paper |
|--------|-------------------|-------|
| **DMBBN** — order-free Markov-blanket learning + Kruskal combination | `DMBBNStructureLearner` / `"dmbbn"` | Carvalho Dâmaso et al. (2026) |
| **iter-DSLA** — iterative divide-and-conquer with community decomposition and SELECT/AND/OR mutation operators | `IterDSLALearner` / `"iterdsla"` | Jia & Li (2026) |
| **SARTRE** — order-based edge pruning via group-lasso sparse regression | `SARTREPruner` / `"sartre"` | Kanamori et al. (2026) |
| **Level-wise DP** — memory-efficient *exact* structure learning over the subset lattice | `LevelWiseDPLearner` / `"levelwise"` | Huang & Suzuki (2026) |
| **BINOTEARS** — differentiable (NOTEARS-style) structure learning for binary data | `BinaryNotearsLearner` / `"binotears"` | Deng & Aragam (2025) |

```python
# order-free Markov-blanket learner
bn.learn_structure(data, method="dmbbn", max_parents=3)

# iterative decomposition learner for large networks
bn.learn_structure(data, method="iterdsla", max_parents=3)

# exact optimum for small problems (n_vars <= ~18)
bn.learn_structure(data, method="levelwise")

# prune a fully-connected DAG given a topological order
bn.learn_structure(data, method="sartre", permutation=order)

# differentiable learner for binary data
bn.learn_structure(binary_data, method="binotears")
```

---

## Parameter learning

Conditional probability distributions (CPDs) are estimated by **maximum-likelihood with optional Dirichlet smoothing** (`alpha` parameter):

```python
bn.learn_parameters(data, alpha=1.0)   # Laplace smoothing
```

For a root variable the CPD is a 1-D probability vector.  For a variable with parents it is a 2-D array of shape `(n_parent_configs, cardinality[var])`.

### `LogisticRegressionParameterLearner`

An alternative CPD estimator that fits each node with an **L1-regularised
multinomial logistic regression** over parent-derived features (dummy
encodings plus optional pairwise XOR interactions), following Moral et al.
(2026).  The number of effective parameters grows *linearly* with the number
of parents instead of exponentially, which helps in dense networks; it is a
drop-in replacement for the MLE learner:

```python
from bayes_nets import LogisticRegressionParameterLearner

bn.learn_parameters(data, parameter_learner=LogisticRegressionParameterLearner(C=5.0))
```

---

## Sampling

**Probabilistic logic sampling** (forward/ancestral sampling):

```python
samples = bn.sample(n_samples=1000, rng=np.random.default_rng(0))
```

Variables are sampled in topological order; each variable is drawn from its CPD conditioned on the already-sampled parent values.

---

## EDA integration

The library is designed to work alongside the `eda_code` modules.  The learned BN is represented with a plain `numpy` adjacency matrix and a Python `dict` of CPDs – the same data structures used by `eda_code/learning/` and `eda_code/sampling/`.

Example in an EDA learning step:

```python
from bayes_nets import BayesianNetwork
import numpy as np

def learn_bn_model(data: np.ndarray, cardinality: np.ndarray, **kwargs):
    bn = BayesianNetwork(n_vars=data.shape[1], cardinality=cardinality)
    bn.fit(data, method="bic", **kwargs)
    return bn
```

---

## API reference

### `BayesianNetwork`

| Method / Property | Description |
|-------------------|-------------|
| `__init__(n_vars, cardinality)` | Create an empty BN |
| `fit(data, method, ...)` | Learn structure **and** parameters |
| `learn_structure(data, method, ...)` | Learn structure only |
| `learn_parameters(data, alpha)` | Estimate CPDs given current structure |
| `sample(n_samples, rng)` | Draw samples via probabilistic logic sampling |
| `add_edge(parent, child)` | Add a DAG edge |
| `remove_edge(parent, child)` | Remove a DAG edge |
| `get_parents(var)` | List of parents |
| `get_children(var)` | List of children |
| `is_dag()` | Validate DAG property |
| `topological_order()` | Kahn's topological sort |
| `n_parameters()` | Total free parameters |
| `marginal(var, data)` | Empirical marginal of a variable |
| `plot(**kwargs)` | Visualise structure |
| `adjacency` | Adjacency matrix (n_vars × n_vars) |
| `cpds` | Dict of CPD tables |

---

## References

* Cooper, G. F., & Herskovits, E. (1992). A Bayesian method for the induction of probabilistic networks from data. *Machine Learning*, 9(4), 309–347.
* Etxeberria, R., & Larrañaga, P. (1999). Global optimization using Bayesian networks. *CIMAF-99*, pp. 332–339.
* Pelikan, M., Goldberg, D. E., & Cantú-Paz, E. (1999). BOA: The Bayesian Optimization Algorithm. *GECCO 1999*, pp. 525–532.
* Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461–464.
* Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.

### Recent structure- and parameter-learning methods

* Carvalho Dâmaso, A., et al. (2026). Learning Bayesian Network Structures Without Variable Ordering Influence: A Markov Blanket-Based Approach. *Computational Intelligence*. — `DMBBNStructureLearner`
* Jia, X., & Li, Z. (2026). An iterative structure decomposition learning method for complex Bayesian networks. *Complex & Intelligent Systems*, 12:164. — `IterDSLALearner`
* Moral, S., Moral-García, S., Cano, A., et al. (2026). Computing conditional probabilities in Bayesian networks using logistic regression. *Applied Soft Computing*, 198, 115284. — `LogisticRegressionParameterLearner`
* Kanamori, T., Takagi, S., & Kobayashi, K. (2026). Sparse Additive Model Pruning for Order-Based Causal Structure Learning. *AAAI 2026*. — `SARTREPruner`
* Huang, Z., & Suzuki, J. (2026). Memory-efficient exact Bayesian network structure learning: a single-pass level-wise dynamic program. *Behaviormetrika*. — `LevelWiseDPLearner`
* Deng, C., & Aragam, B. (2025). Differentiable Structure Learning and Causal Discovery for General Binary Data. *NeurIPS 2025*. — `BinaryNotearsLearner`
* Silander, T., & Myllymäki, P. (2012). A simple approach for finding the globally optimal Bayesian network structure. *UAI 2006*. (basis for the level-wise DP)
* Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. *NeurIPS 2018*. (basis for BINOTEARS)
