# Ranking of Extension Methods for `bayes_nets`

Evaluated against the papers in `docs/extensions/` as of 2026-06-08.
Three criteria (in decreasing priority):

1. **Efficiency** — impact on BN learning/inference speed when applied intensively inside EDAs with large numbers of variables.
2. **Consistency** — degree of alignment with the existing package structure (`structure_learning.py`, `inference.py`, `factorization.py`, `scoring.py`). Less disruptive is better.
3. **Ease** — implementation effort for an LLM-based agent (Claude).

Neural-network-based methods designed exclusively for continuous data
(GAE-Autoencoder, Neural Autoregressive Flows/FANS, Gradient-based GCFS)
are excluded: they assume continuous variables and require deep-learning
infrastructure architecturally incompatible with the discrete EDA setting.

---

## Tier 1 — Implement First

### Rank 1 · HC-Stable / Tabu-Stable
*Kitson & Constantinou (2023) — `Structure_Learning/Stable structure learning with HC-Stable and Tabu-Stable algorithms.pdf`*

| Criterion | Assessment |
|-----------|-----------|
| Efficiency | Tabu search consistently escapes local optima in score-based hill-climbing; eliminates non-determinism that forces repeated runs to compensate for ordering artifacts, directly cutting EDA per-generation overhead. Tabu also considers add/delete/reverse simultaneously, which can find better structures in fewer iterations than greedy-add-only search. |
| Consistency | **Near-zero disruption.** `TabuHillClimbLearner` is a natural extension of the existing `GreedyHillClimbLearner` pattern, reuses `ScoringMethod.local_score()` unchanged, and slots into `BayesianNetwork.learn_structure()` as `method="tabu"` / `method="stable_hc"`. |
| Ease | The full algorithm is: maintain a tabu deque of recently visited operations, break score ties deterministically (smallest node index), accept no-improvement moves only if not tabu. All primitives already exist in the codebase. |

---

### Rank 2 · Junction-Tree VE-MAP (Nilsson 1998)
*Nilsson (1998) — `Most_Probable_Configurations/An efficient algorithm for finding the M most probable configurations in probabilistic expert systems.pdf`*

| Criterion | Assessment |
|-----------|-----------|
| Efficiency | **Critical.** The current `MaxProductInference._all_assignments()` enumerates every joint state — O(∏ᵢ kᵢ) — completely unusable beyond ~20 binary variables. Nilsson's variable-elimination MAP replaces this with a cost polynomial in the clique sizes: O(n · exp(treewidth)). With max_parents=3 and binary variables, treewidth ≤ 4. |
| Consistency | `inference.py` can import `bn_to_factorization` and `moralize`/`triangulate` from `factorization.py`. The public API (`most_probable_config`, `k_most_probable_configs`, `marginals`) stays unchanged on `BayesianNetwork`; only the internals of `MaxProductInference` change. |
| Ease | The core is a two-pass algorithm (forward max-product VE + backward argmax traceback) on a tree — structurally similar to the junction-tree ordering already implemented in `_order_cliques_for_sampling`. k-best uses a Nilsson/Lawler priority-queue search, each expansion requiring one VE-MAP call with added evidence. |

---

## Tier 2 — High Value, Moderate Effort

### Rank 3 · Grow-Shrink (GS) Markov-Blanket BN Induction
*Margaritis & Thrun (1999) — `Structure_Learning/NIPS-1999-bayesian-network-induction-via-local-neighborhoods-Paper.pdf`*

| Criterion | Assessment |
|-----------|-----------|
| Efficiency | GS runs O(n · |D|) conditional independence tests — fundamentally better than the O(n² · scoring) cost of `GreedyHillClimbLearner` at large n. In EDA contexts with hundreds of variables and repeated fitting per generation, this asymptotic advantage dominates. |
| Consistency | Adds `GrowShrinkLearner` to `structure_learning.py` with the same `.learn(data, n_vars, cardinality) → adjacency` signature. The only new component is a chi-square CI test (~10 lines using scipy). |
| Ease | The Grow/Shrink/Orient three-phase algorithm is completely specified in the paper and has no hidden numerical subtleties. |

---

### Rank 4 · Tarjan's Clique-Separator Decomposition
*Tarjan (1985) — `Triangulations_and_Decompositions/Decomposition by Clique Separators.pdf`*

| Criterion | Assessment |
|-----------|-----------|
| Efficiency | Clique separators partition a graph into independently triangulable subgraphs. For sparse BN structures (typical in EDAs), this yields cliques smaller than min-fill alone — directly reducing exponential-in-clique-size table costs in `factorization.py`. |
| Consistency | Adds directly to `factorization.py` as a pre-processing step before `triangulate()`. No API changes. |
| Ease | O(nm) graph algorithm, well-defined, operates on the moral adjacency matrix already present. |

---

## Tier 3 — Worth Implementing After Tiers 1–2

### Rank 5 · Recursive Causal Discovery (RCD)
*Mokhtarian et al. — `Structure_Learning/Recursive Causal Discovery.pdf`*

| Criterion | Assessment |
|-----------|-----------|
| Efficiency | Worst-case O(n² + nΔₘ 2^Δₘ) where Δₘ is max Markov blanket size; better than PC when sparse. Recursive removal reduces conditioning set sizes, lowering sample-complexity per generation. |
| Consistency | Moderate. New `RecursiveCDLearner` in `structure_learning.py` fits the existing learner pattern. Shares the chi-square CI test helper needed by Rank 3. Best implemented after GS. |
| Ease | Moderate. A reference Python package (RCD) exists. |

---

### Rank 6 · Loopy Belief Propagation for M-Best Configs
*Yanover & Weiss (2003) — `Most_Probable_Configurations/NIPS-2003-finding-the-m-most-probable-configurations-using-loopy-belief-propagation-Paper.pdf`*

| Criterion | Assessment |
|-----------|-----------|
| Efficiency | Enables approximate M-best inference on graphs too large for exact JT inference (high treewidth). Relevant when the BN has many parents and junction-tree cliques become intractable. |
| Consistency | Lower — requires full loopy BP infrastructure not currently present. Best as a fallback inside `MaxProductInference` when treewidth is too high. |
| Ease | Moderate. Max-product loopy BP is well understood but requires damping/scheduling choices and convergence monitoring. |

---

### Rank 7 · Flerova et al. M-Best A\*/Branch-and-Bound
*Flerova et al. — `Most_Probable_Configurations/Searching for the M Best Solutions in Graphical Models.pdf`*

| Criterion | Assessment |
|-----------|-----------|
| Efficiency | AND/OR best-first search provides anytime behavior with bounded memory when k is small. However, Rank 2 already gives polynomial exact M-best for moderate treewidth, making this mainly useful on very dense graphs. |
| Consistency | Low — AND/OR search graphs are a new data structure not present anywhere in the library. |
| Ease | Hard. AND/OR graph construction and heuristic bounds from bucket elimination are non-trivial. |

---

### Rank 8 · Heggernes Minimal Triangulations
*Heggernes — `Triangulations_and_Decompositions/Minimal triangulations of graphs: A survey.pdf`*

| Criterion | Assessment |
|-----------|-----------|
| Efficiency | Minimum-width triangulations minimize maximum clique size, directly reducing table sizes in the factorization. However, minimum-treewidth computation is NP-hard; only approximations (e.g., LexBFS, MCS) are polynomial. |
| Consistency | Extends `factorization.py`'s `triangulate()` function; compatible with existing interface. |
| Ease | Hard — the survey covers many algorithms; choosing and implementing one correctly takes significant effort. |

---

### Rank 9 · RPCD (Recursive Parallel Causal Discovery)
*Mondal et al. — `Structure_Learning/A Fast Algorithm for High-Dimensional Causal Discovery.pdf`*

| Criterion | Assessment |
|-----------|-----------|
| Efficiency | Designed for n > 1000 (gene expression). Recursive partitioning with parallel CI tests. |
| Consistency | Moderate — new learner class; requires discrete CI test adaptation (primary paper targets continuous data). |
| Ease | Moderate-hard — parallelism and discrete adaptation are underspecified. |

---

## Not Ranked

| Paper | Reason |
|---|---|
| Echegoyen et al. — *Analyzing k Most Probable Solutions in EDAs* | Analysis paper (co-authored by project PI); no standalone algorithm to implement |
| hBOA paper (Hauschild et al.) | Empirical analysis; hBOA would be an EDA variant, not a `bayes_nets` primitive |
| Chen et al. — *M Most Probable Modes* | Introduces "modes" (local maxima) concept; secondary to Rank 2 |
| GAE, FANS, GCFS | Continuous variables + PyTorch; architecturally incompatible with discrete EDA setting |
| Truncated-name files | Could not be fully read; *Decimation strategies/template recombination*, *Diverse M-Best in MRFs*, *Message passing review in EDAs* appear directly relevant and warrant manual review before finalizing roadmap |

---

## Summary Table

| Rank | Method | Category | Efficiency gain | Consistency | Ease |
|------|--------|----------|----------------|-------------|------|
| 1 | HC-Stable / Tabu-Stable | Structure learning | Medium | ★★★ | ★★★ |
| 2 | Nilsson VE-MAP | Inference | **Critical** | ★★★ | ★★ |
| 3 | Grow-Shrink (GS) | Structure learning | High | ★★ | ★★ |
| 4 | Tarjan Clique Separators | Factorization | High (sparse BNs) | ★★★ | ★★ |
| 5 | RCD | Structure learning | Medium-high | ★★ | ★★ |
| 6 | Loopy BP M-best | Inference | High (dense BNs) | ★ | ★★ |
| 7 | M-best A\*/BB | Inference | Medium | ★ | ★ |
| 8 | Minimal Triangulations | Factorization | Medium | ★★ | ★ |
| 9 | RPCD | Structure learning | High (very large n) | ★★ | ★ |

**Implementation priority:** Fix the brute-force inference (Rank 2) first — it is broken for any
realistic EDA problem size. Then add Tabu-Stable structure learning (Rank 1). GS (Rank 3) and
Tarjan (Rank 4) follow naturally once the foundations are solid.
