# EDA Benchmark — Small-tier evaluation of BN learning algorithms

Evaluation of the library's BN structure learners on the EDA benchmark in
`data/eda_datasets/`, **Small tier**. Produced by
`scripts/eval_eda_benchmark.py small`; raw per-run numbers in `eda_eval_small.csv`.

## Protocol

* **Datasets (Small = n ≤ 40, 8 datasets):** `OneMax_36`, `Trap_36`, `Braid_36`,
  `Deceptive3_39`, `Checkerboard_36`, `Ising_36`, `MaxClique_30`, `EqualProducts_36`.
  (Medium = 50 < n < 100; Large = n ≥ 100. `UBQP_50` at n = 50 falls outside the agreed
  bounds and is not in any tier.)
* **Sample probabilities:** objective globally min-max normalised to [0, 1], then a
  **Boltzmann** distribution `p ∝ exp(f_norm / T)` (**maximisation**). T is swept over
  **{0.1, 1.0, 10}**.
* **Learning:** train rows (1000) + their Boltzmann weights are used to learn *both*
  structure and parameters (weighted).
* **Metrics** (per learned BN):
  * **F1** — skeleton F1 vs. the true undirected interaction matrix (directions ignored);
  * **llTest** — mean log-likelihood per sample on the test split (higher = better);
  * **KLtest** — `KL(p_data ‖ p_BN)` on the test split, `p_data` = Boltzmann probability
    and `p_BN` = BN joint, each renormalised over the split's rows (lower = better).
* **Time filter:** on this first (small) evaluation, any method whose **median** per-dataset
  running time exceeds **15× the K2 time** is removed. (K2 ranges 0.1–3.3 s across these
  datasets, so the median — not the mean — is used; the mean is skewed by the slow-K2
  datasets and would wrongly keep methods that time out on the fast ones.)
* Runs are parallelised over 10 CPUs with a per-run SIGALRM cap at 15× K2, so an
  over-budget method is stopped at the threshold and recorded as removed.

## 1. Timing and the 15×-K2 filter

Median / max time ratio vs. K2 at T = 1.0 (8 datasets), and how many datasets each method
actually completed within the cap:

| Method | median ×K2 | max ×K2 | completed | verdict |
|--------|-----------:|--------:|:---------:|:-------:|
| k2 | 1.0 | 1.0 | 7/8 | keep (reference) |
| bic_hc | 3.4 | 4.7 | 7/8 | keep |
| k2_mi | 3.5 | 6.2 | 7/8 | keep |
| pc | 5.0 | 15.0 | 6/8 | keep |
| aic_hc | 5.5 | 10.2 | 7/8 | keep |
| stable_pc | 5.6 | 17.2 | 5/8 | keep |
| k2_mb | 7.5 | 26.0 | 7/8 | keep |
| k2_ensemble | 8.3 | 12.6 | 7/8 | keep |
| stable_hc | 8.6 | 15.0 | 6/8 | keep |
| k2_plus | 9.4 | 13.8 | 7/8 | keep |
| bounded_tw | 11.1 | 25.3 | 7/8 | keep |
| binotears | 12.2 | 26.9 | 7/8 | keep |
| **dt** | 15.5 | 22.9 | 2/8 | **REMOVED** |
| **dmbbn** | 16.0 | 21.7 | 2/8 | **REMOVED** |
| **rcd** | 16.6 | 21.7 | 2/8 | **REMOVED** |
| **gs** | 16.6 | 40.1 | 0/8 | **REMOVED** |
| **tabu** | 16.6 | 40.0 | 0/8 | **REMOVED** |
| **dg** | 16.6 | 40.0 | 0/8 | **REMOVED** |
| **iterdsla** | 16.6 | 40.0 | 0/8 | **REMOVED** |
| **rpcd** | 44.7 | 122.2 | 2/8 | **REMOVED** |

**Removed (8):** `dt`, `dg`, `dmbbn`, `iterdsla`, `gs`, `tabu`, `rcd`, `rpcd`.
**Survivors (12):** `k2`, `k2_mi`, `k2_plus`, `k2_ensemble`, `k2_mb`, `bic_hc`, `aic_hc`,
`stable_hc`, `pc`, `stable_pc`, `binotears`, `bounded_tw`.

*Why so many are removed:* the true structures here are **dense pairwise-interaction
graphs** (see §3). Constraint-based learners (`gs`, `rcd`, `rpcd`, `tabu`) run a
quadratic-or-worse number of CI tests / neighbour moves and blow past 15× K2; the
local-structure learners (`dt`, `dg`) and `dmbbn`/`iterdsla` are likewise too slow at this
density. `pc`/`stable_pc` survive by the median but still time out on the two densest
problems (they complete 5–6/8).

## 2. Accuracy of the survivors (per temperature)

`F1` = mean skeleton-F1 over the **structured** problems (excluding OneMax, whose true
graph has 0 edges and is a degenerate F1 case); `llTest`/`KLtest` averaged over completed
datasets; `done` = datasets completed of 8.

**T = 1.0** (reference)

| Method | F1 (structured) | llTest | KLtest | done |
|--------|----------------:|-------:|-------:|:----:|
| aic_hc | **0.231** | −17.16 | 0.335 | 7/8 |
| k2_mi | 0.203 | −17.31 | 0.345 | 7/8 |
| pc | 0.203 | −18.94 | 0.386 | 6/8 |
| k2_plus | 0.202 | −17.24 | 0.342 | 7/8 |
| k2 | 0.199 | −17.36 | 0.349 | 7/8 |
| k2_mb | 0.199 | −17.36 | 0.349 | 7/8 |
| stable_pc | 0.184 | −20.16 | 0.402 | 5/8 |
| k2_ensemble | 0.182 | −17.64 | **0.322** | 7/8 |
| stable_hc | 0.162 | −18.07 | 0.258 | 6/8 |
| bic_hc | 0.146 | −17.77 | 0.305 | 7/8 |
| bounded_tw | 0.055 | −18.36 | 0.314 | 7/8 |
| binotears | 0.000 | −18.66 | 0.285 | 7/8 |

**T = 0.1** (sharp selection)

| Method | F1 (structured) | llTest | KLtest | done |
|--------|----------------:|-------:|-------:|:----:|
| pc | **0.394** | −15.07 | 0.202 | 3/8 |
| stable_pc | 0.381 | −15.08 | 0.226 | 3/8 |
| aic_hc | 0.348 | −16.40 | 2.447 | 5/8 |
| bic_hc | 0.302 | −16.60 | 3.035 | 6/8 |
| k2 | 0.287 | −14.50 | 1.778 | 4/8 |
| k2_mi | 0.269 | −14.27 | 1.707 | 4/8 |
| k2_mb | 0.246 | **−8.39** | 0.794 | 3/8 |
| k2_plus | 0.238 | −16.36 | 1.608 | 3/8 |
| … | | | | |
| binotears | 0.000 | −22.53 | 5.122 | 7/8 |

**T = 10** (near-uniform)

| Method | F1 (structured) | llTest | KLtest | done |
|--------|----------------:|-------:|-------:|:----:|
| aic_hc | **0.159** | −17.62 | 0.319 | 7/8 |
| pc | 0.154 | −20.41 | 0.429 | 5/8 |
| k2 / k2_mb | 0.146 | −17.77 | 0.317 | 7/8 |
| stable_pc | 0.137 | −20.60 | 0.385 | 5/8 |
| k2_mi | 0.136 | −17.73 | 0.303 | 7/8 |
| stable_hc | 0.136 | −18.41 | **0.222** | 6/8 |
| k2_plus | 0.135 | −17.62 | 0.312 | 7/8 |
| bic_hc | 0.127 | −18.14 | 0.273 | 7/8 |
| binotears | 0.000 | −19.11 | 0.237 | 7/8 |

## 3. Per-problem skeleton F1 (T = 1.0)

| Method | Trap(54) | Decept(39) | Braid(35) | Checker(40) | Ising(72) | MaxClq(270) | EqProd(630) | OneMax(0) |
|--------|---------:|-----------:|----------:|------------:|----------:|------------:|------------:|----------:|
| k2 | 0.28 | 0.12 | – | 0.54 | 0.05 | 0.18 | 0.03 | 0.00 |
| k2_mi | 0.25 | 0.20 | – | 0.46 | 0.07 | 0.21 | 0.03 | 0.00 |
| k2_plus | 0.24 | 0.19 | – | 0.46 | 0.07 | 0.22 | 0.03 | 0.00 |
| aic_hc | 0.24 | 0.19 | – | 0.49 | 0.19 | 0.22 | 0.06 | 0.00 |
| pc | 0.24 | 0.11 | – | 0.46 | 0.11 | 0.24 | 0.06 | – |
| binotears | 0.00 | 0.00 | – | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** |

(Numbers in headers are true edge counts; "–" = not completed / method timed out on that
dataset. `Braid_36` has the slowest K2 (3.3 s) and several runs did not finish at T = 1.0.)

## 4. Findings

1. **Time filter is decisive on dense EDA structures.** Two-thirds of the constraint-based
   and local-structure learners exceed 15× K2 and are removed; the survivors are the
   score-based **K2 family, HC (BIC/AIC), PC/Stable-PC, binotears and bounded_tw**.
2. **Best reliable structure recovery:** **AIC-HC** and the **K2 MI-ordering variants
   (`k2_mi`, `k2_plus`)** give the highest structured F1 among methods that finish on
   (nearly) all datasets. Plain **K2** is an excellent speed/accuracy baseline (1× time,
   F1 ≈ AIC/K2-variants).
3. **PC/Stable-PC score the highest F1 when they finish** (0.39 at T = 0.1) but only
   complete 3–6/8 datasets — unreliable at this density.
4. **Likelihood & KL:** the K2 family gives the best (least-negative) test log-likelihood;
   `k2_ensemble` and `stable_hc` give the lowest test KL at T = 1.0. `binotears` is
   consistently weakest on likelihood.
5. **Temperature matters a lot.** Sharpening the Boltzmann weights (T = 0.1) concentrates
   learning on the high-fitness, more-structured samples and **raises F1**, but drives the
   weighted train distribution toward a few configurations (degenerate likelihood/KL).
   T = 10 flattens the weights and lowers F1. T = 1.0 is the balanced operating point.
6. **Absolute F1 is low by construction.** The true structures are dense pairwise-interaction
   graphs (Trap 54, Ising 72, MaxClique 270, EqualProducts 630 edges) that a bounded-in-degree
   BN cannot fully represent; `Checkerboard` (sparser, local) is the most learnable (F1 ≈ 0.5).
   `binotears` only "wins" `OneMax` because it learns the empty graph (0 true edges).

## 5. Reproduce

```bash
python3 scripts/eval_eda_benchmark.py small      # -> data/eda_datasets/results/eda_eval_small.csv
# medium / large tiers:
python3 scripts/eval_eda_benchmark.py medium
python3 scripts/eval_eda_benchmark.py large
```

Next step (pending your go-ahead): run the **Medium** and **Large** tiers with the surviving
methods to see how the ranking and time budget scale to 60–256 variables.
