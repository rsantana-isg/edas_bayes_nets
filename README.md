# bayes_nets

A lightweight, standalone Python library for **learning**, **sampling**, and **visualising** discrete Bayesian networks (BNs).

Designed as a drop-in replacement for [pgmpy](https://pgmpy.org/) within estimation-of-distribution algorithm (EDA) workflows, while remaining fully usable as a general-purpose BN toolkit.

---

## Goals

* **Discrete representation** – all variables take a finite number of states; each variable's cardinality is specified at construction time.
* **Multiple structure-learning algorithms** – BIC, AIC, and K2 scoring with greedy hill-climbing or the K2 algorithm.
* **Probabilistic logic sampling** – forward (ancestral) sampling from a learned BN.
* **EDA integration** – the library is designed to work seamlessly with the `eda_code` learning and sampling modules as a replacement for pgmpy.
* **Visualisation** – plot BN structures and marginal/conditional probability distributions.

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

**Optional dependencies** (needed for visualisation)

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

---

## Parameter learning

Conditional probability distributions (CPDs) are estimated by **maximum-likelihood with optional Dirichlet smoothing** (`alpha` parameter):

```python
bn.learn_parameters(data, alpha=1.0)   # Laplace smoothing
```

For a root variable the CPD is a 1-D probability vector.  For a variable with parents it is a 2-D array of shape `(n_parent_configs, cardinality[var])`.

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
