# Ranking of BN Structure Learning and Sampling Algorithms

This document evaluates research papers from `docs/Other_BN_Structure_Learning_Methods/` for implementation in the `bayes_nets` library. The primary goal is to enhance the library's performance and capability within the context of Estimation of Distribution Algorithms (EDAs), where Bayesian Networks (BNs) are learned and sampled intensively.

## Evaluation Criteria
1. **Impact on Efficiency**: Capability to handle a large number of variables and high-dimensional data efficiently, specifically for iterative applications in EDAs.
2. **Consistency**: Alignment with the current package architecture (discrete BNs, modular structure, pgmpy-like API).
3. **Ease of Implementation**: Complexity of the algorithm relative to current Python best models and library dependencies.

---

## Ranked Algorithms

### Rank 1: DMBBN (Dynamic Markov Blanket Bayesian Network)
*   **Paper**: *Learning Bayesian Network Structures Without Variable Ordering Influence: A Markov Blanket-Based Approach* (Dâmaso et al., 2026)
*   **Description**: A score-based algorithm that uses Markov Blankets to learn local structures independently for each variable, then combines them into a global DAG using an adapted Kruskal's algorithm to ensure acyclicity.
*   **Evaluation**:
    *   **Efficiency**: Extremely high. By decomposing the problem into local Markov Blanket searches, it scales effectively to large datasets and many variables without requiring a pre-defined ordering (a major bottleneck for K2).
    *   **Consistency**: High. Uses the K2 scoring function already established in the library.
    *   **Implementation**: High. The local search and Kruskal-based combination are straightforward to implement within the existing `structure_learning.py` framework.

### Rank 2: iter-DSLA (Iterative Structure Decomposition Learning)
*   **Paper**: *An iterative structure decomposition learning method for complex Bayesian networks* (Jia & Li, 2026)
*   **Description**: A divide-and-conquer framework that decomposes a large network into subgraphs using community detection, learns them in parallel, and iteratively refines the global structure using mutation operators (SELECT, AND, OR).
*   **Evaluation**:
    *   **Efficiency**: Very high. Specifically designed for "complex" BNs with hundreds or thousands of nodes. The parallelization potential is perfect for intensive EDA workflows.
    *   **Consistency**: Moderate. Fits the "hybrid" approach and can wrap existing learners (like Hill Climbing or K2).
    *   **Implementation**: Moderate. Requires integrating overlapping community detection and iterative loop logic.

### Rank 3: Enhanced Logistic Regression for Parameters
*   **Paper**: *Computing conditional probabilities in Bayesian networks using logistic regression* (Moral et al., 2026)
*   **Description**: Proposes using logistic regression with artificial feature variables (XOR combinations, LDA-based discretization) to estimate CPTs. This drastically reduces the number of parameters for nodes with many parents.
*   **Evaluation**:
    *   **Efficiency**: High impact on memory and estimation quality when BNs are dense. It addresses the "CPT explosion" problem in EDAs.
    *   **Consistency**: High. Directly integrates into `parameter_learning.py`.
    *   **Implementation**: High. Leverages standard tools like `scikit-learn` for logistic regression and LDA.

### Rank 4: SARTRE (Sparse Additive Randomized Tree Ensemble)
*   **Paper**: *Sparse Additive Model Pruning for Order-Based Causal Structure Learning* (Kanamori et al., 2026)
*   **Description**: A pruning method for order-based BNSL. It uses randomized tree embedding and group lasso regression to identify and prune spurious edges from a fully-connected DAG induced by a topological order.
*   **Evaluation**:
    *   **Efficiency**: High. Significantly faster than the standard CAM-pruning while maintaining accuracy.
    *   **Consistency**: Moderate. Best used in conjunction with order-based learners like the library's `K2StructureLearner` or `PCLearner`.
    *   **Implementation**: Moderate. Requires group lasso and randomized tree logic.

### Rank 5: Memory-Efficient Level-Wise DP
*   **Paper**: *Memory-efficient exact bayesian network structure learning: a single-pass level-wise dynamic program* (Huang & Suzuki, 2026)
*   **Description**: An exact structure learning algorithm that uses a single-pass traversal of the subset lattice to minimize memory usage from $O(p 2^p)$ to $O(\sqrt{p} 2^p)$.
*   **Evaluation**:
    *   **Efficiency**: Moderate. While "efficient" for exact learning, it is still limited to ~30 variables, which may be too low for some EDA applications.
    *   **Consistency**: High. Provides a "Gold Standard" exact learner for the library.
    *   **Implementation**: Moderate. Requires efficient bit-masking and lattice traversal.

### Rank 6: BINOTEARS (Binary NOTEARS)
*   **Paper**: *Differentiable Structure Learning and Causal Discovery for General Binary Data* (Deng & Aragam, 2025)
*   **Description**: Adapts the continuous optimization approach (NOTEARS) to binary data using the Multivariate Bernoulli distribution, capturing higher-order interactions.
*   **Evaluation**:
    *   **Efficiency**: Moderate. Continuous optimization is powerful but can be slower than heuristic/greedy methods for very large $N$.
    *   **Consistency**: Low to Moderate. Introducing differentiable optimization (PyTorch/NumPy-based) is a shift from the current combinatorial focus.
    *   **Implementation**: Moderate. Requires a differentiable acyclicity constraint and a continuous optimizer.

---

## Other Evaluated Papers (Lower Rank)

*   **SALAD** (Ng et al., 2024): Focuses on latent variables. While theoretically advanced, it might be too specialized for the core EDA-focused `bayes_nets` library unless latent variable modeling becomes a priority.
*   **Chordal PAC-Learner** (Bhattacharyya et al., 2025): Highly theoretical approach for chordal skeletons. Implementation would be complex and the "known skeleton" assumption is often too strong.
*   **CSBHC / ALDAGs** (Varando et al., 2025): Focuses on staged trees and asymmetry. Very interesting but expands the scope beyond standard DAGs into more complex graphical models.
*   **Unreliable Oracle** (Harviainen et al., 2026): Theoretical study of error tolerance; less about a practical high-performance algorithm for EDAs.
*   **Decomposable Context-Specific Models** (Alexandr et al., 2024): Primarily algebraic characterization; lacks a scalable learning algorithm for immediate implementation.

## Summary Recommendation
The implementation of **DMBBN** and **iter-DSLA** should be prioritized to solve scalability issues for large variable sets in EDAs. Simultaneously, the **Enhanced Logistic Regression** method from Moral et al. (2026) should be added to `parameter_learning.py` to handle the parameter explosion in dense networks.
