from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _as_int_array(values: Iterable[int]) -> np.ndarray:
    return np.array(sorted({int(v) for v in values}), dtype=int)


def _config_indices(data: np.ndarray, variables: Sequence[int], cardinality: np.ndarray) -> np.ndarray:
    """Mixed-radix encoding where the first variable varies fastest."""
    if len(variables) == 0:
        return np.zeros(data.shape[0], dtype=int)

    indices = np.zeros(data.shape[0], dtype=int)
    mult = 1
    for var in variables:
        indices += data[:, var] * mult
        mult *= int(cardinality[var])
    return indices


def _joint_distribution_from_bn(
    adjacency: np.ndarray,
    cpds: Dict[int, Dict],
    cardinality: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (assignments, probabilities) over the full joint implied by CPDs."""
    n_vars = adjacency.shape[0]
    n_states = int(np.prod(cardinality))

    assignments = np.zeros((n_states, n_vars), dtype=int)
    mults = np.ones(n_vars, dtype=int)
    for i in range(1, n_vars):
        mults[i] = mults[i - 1] * int(cardinality[i - 1])

    for idx in range(n_states):
        rem = idx
        for i in range(n_vars - 1, -1, -1):
            assignments[idx, i] = rem // mults[i]
            rem %= mults[i]

    probs = np.ones(n_states, dtype=float)
    for var in range(n_vars):
        info = cpds[var]
        parents = [int(p) for p in info["parents"]]
        cpd = np.asarray(info["cpd"], dtype=float)
        if len(parents) == 0:
            probs *= cpd[assignments[:, var]]
            continue

        parent_mult = 1
        parent_idx = np.zeros(n_states, dtype=int)
        for p in parents:
            parent_idx += assignments[:, p] * parent_mult
            parent_mult *= int(cardinality[p])
        probs *= cpd[parent_idx, assignments[:, var]]

    return assignments, probs


def moralize(adjacency: np.ndarray) -> np.ndarray:
    """Return the undirected moral graph of a DAG adjacency matrix."""
    adj = np.asarray(adjacency, dtype=int)
    n = adj.shape[0]
    moral = np.zeros((n, n), dtype=int)

    # Drop directions.
    directed_edges = np.where(adj > 0)
    moral[directed_edges] = 1
    moral[(directed_edges[1], directed_edges[0])] = 1

    # Marry parents of each child.
    for child in range(n):
        parents = np.where(adj[:, child] > 0)[0]
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                p1 = int(parents[i])
                p2 = int(parents[j])
                moral[p1, p2] = 1
                moral[p2, p1] = 1

    np.fill_diagonal(moral, 0)
    return moral


def _add_fill_edges(work: np.ndarray, node: int) -> Tuple[np.ndarray, np.ndarray]:
    neighbors = np.where(work[node] > 0)[0]
    if len(neighbors) > 1:
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u = int(neighbors[i])
                v = int(neighbors[j])
                work[u, v] = 1
                work[v, u] = 1
    clique = _as_int_array(np.concatenate(([node], neighbors)))
    work[node, :] = 0
    work[:, node] = 0
    return work, clique


def _selection_cost(work: np.ndarray, node: int, method: str) -> Tuple[int, int, int]:
    neighbors = np.where(work[node] > 0)[0]
    degree = len(neighbors)
    if method == "min-degree":
        return degree, degree, node

    # min-fill default
    fill_in = 0
    for i in range(degree):
        for j in range(i + 1, degree):
            if work[neighbors[i], neighbors[j]] == 0:
                fill_in += 1
    return fill_in, degree, node


def _maximal_cliques(cliques: List[np.ndarray]) -> List[np.ndarray]:
    uniq: List[np.ndarray] = []
    for clique in cliques:
        c = _as_int_array(clique)
        if len(c) == 0:
            continue
        if not any(np.array_equal(c, existing) for existing in uniq):
            uniq.append(c)

    maximal: List[np.ndarray] = []
    for i, ci in enumerate(uniq):
        si = set(ci.tolist())
        is_subset = False
        for j, cj in enumerate(uniq):
            if i == j:
                continue
            if si.issubset(set(cj.tolist())) and len(cj) >= len(ci):
                if len(cj) > len(ci) or j < i:
                    is_subset = True
                    break
        if not is_subset:
            maximal.append(ci)

    maximal.sort(key=lambda c: (-len(c), tuple(c.tolist())))
    return maximal


def _split_clique(clique: np.ndarray, max_width: int) -> List[np.ndarray]:
    if len(clique) <= max_width:
        return [clique]

    parts: List[np.ndarray] = []
    start = 0
    while start < len(clique):
        end = min(start + max_width, len(clique))
        part = clique[start:end]
        if start > 0:
            # Ensure overlap with previous segment to preserve connectivity.
            part = np.concatenate(([clique[start - 1]], part))
            part = _as_int_array(part)
            if len(part) > max_width:
                part = part[:max_width]
        parts.append(_as_int_array(part))
        start = end
    return parts


def _induced_subgraph(adjacency: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    return np.asarray(adjacency[np.ix_(nodes, nodes)], dtype=int)


def _connected_components_nodes(adjacency: np.ndarray, nodes: np.ndarray) -> List[np.ndarray]:
    node_set = set(int(v) for v in nodes.tolist())
    seen: set[int] = set()
    components: List[np.ndarray] = []

    for start in sorted(node_set):
        if start in seen:
            continue
        stack = [start]
        comp: List[int] = []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            neigh = np.where(adjacency[v] > 0)[0]
            for nxt in neigh:
                nxt_i = int(nxt)
                if nxt_i in node_set and nxt_i not in seen:
                    stack.append(nxt_i)
        components.append(_as_int_array(comp))

    return components


def _is_clique(adjacency: np.ndarray, nodes: np.ndarray) -> bool:
    if len(nodes) <= 1:
        return True
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if adjacency[int(nodes[i]), int(nodes[j])] == 0:
                return False
    return True


def _find_clique_separator(adjacency: np.ndarray, nodes: np.ndarray) -> Optional[np.ndarray]:
    """Find a clique separator in the induced graph, if one exists."""
    node_set = set(int(v) for v in nodes.tolist())

    # Singleton clique separators (articulation vertices).
    for v in sorted(node_set):
        rest = _as_int_array([u for u in node_set if u != v])
        if len(rest) <= 1:
            continue
        comps = _connected_components_nodes(adjacency, rest)
        if len(comps) > 1:
            return np.array([v], dtype=int)

    # Multi-node candidates via common-neighbor cliques of adjacent pairs.
    candidates: List[np.ndarray] = []
    node_list = sorted(node_set)
    for i in range(len(node_list)):
        u = node_list[i]
        neigh_u = set(int(x) for x in np.where(adjacency[u] > 0)[0] if int(x) in node_set)
        for j in range(i + 1, len(node_list)):
            v = node_list[j]
            if adjacency[u, v] == 0:
                continue
            neigh_v = set(int(x) for x in np.where(adjacency[v] > 0)[0] if int(x) in node_set)
            inter = _as_int_array(neigh_u.intersection(neigh_v))
            if len(inter) <= 1:
                continue
            if _is_clique(adjacency, inter):
                candidates.append(inter)

    candidates.sort(key=lambda s: (len(s), tuple(s.tolist())))
    for sep in candidates:
        sep_set = set(int(v) for v in sep.tolist())
        rest = _as_int_array([u for u in node_set if u not in sep_set])
        if len(rest) == 0:
            continue
        comps = _connected_components_nodes(adjacency, rest)
        if len(comps) > 1:
            return sep

    return None


def _decompose_by_clique_separators(adjacency: np.ndarray) -> List[np.ndarray]:
    """Recursively decompose graph into atoms using clique separators."""
    n = adjacency.shape[0]
    all_nodes = _as_int_array(range(n))

    atoms: List[np.ndarray] = []

    def recurse(nodes: np.ndarray) -> None:
        if len(nodes) <= 2:
            atoms.append(nodes)
            return

        sep = _find_clique_separator(adjacency, nodes)
        if sep is None:
            atoms.append(nodes)
            return

        node_set = set(int(v) for v in nodes.tolist())
        sep_set = set(int(v) for v in sep.tolist())
        rest = _as_int_array([u for u in node_set if u not in sep_set])
        comps = _connected_components_nodes(adjacency, rest)
        if len(comps) <= 1:
            atoms.append(nodes)
            return

        progressed = False
        for comp in comps:
            block = _as_int_array(np.concatenate((comp, sep)))
            if len(block) < len(nodes):
                progressed = True
                recurse(block)

        if not progressed:
            atoms.append(nodes)

    recurse(all_nodes)

    uniq: List[np.ndarray] = []
    for atom in atoms:
        a = _as_int_array(atom)
        if not any(np.array_equal(a, b) for b in uniq):
            uniq.append(a)
    uniq.sort(key=lambda a: (len(a), tuple(a.tolist())))
    return uniq


def _triangulate_block(
    moral_adjacency: np.ndarray,
    block_nodes: np.ndarray,
    method: str,
) -> Tuple[List[int], List[np.ndarray]]:
    work = np.zeros_like(moral_adjacency)
    work[np.ix_(block_nodes, block_nodes)] = moral_adjacency[np.ix_(block_nodes, block_nodes)]

    remaining = set(int(v) for v in block_nodes.tolist())
    order: List[int] = []
    cliques: List[np.ndarray] = []

    while remaining:
        node = min(remaining, key=lambda x: _selection_cost(work, x, method))
        order.append(node)
        work, clique = _add_fill_edges(work, node)
        cliques.append(clique)
        remaining.remove(node)

    return order, cliques


def triangulate(
    moral_adjacency: np.ndarray,
    cardinality: np.ndarray,
    method: str = "min-fill",
    max_clique_width: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Triangulate a moral graph and return adjacency, elimination order, cliques."""
    method = method.lower()
    if method not in {"min-fill", "min-degree"}:
        raise ValueError("method must be 'min-fill' or 'min-degree'")

    work = np.asarray(moral_adjacency, dtype=int).copy()
    if work.shape[0] != work.shape[1]:
        raise ValueError("moral_adjacency must be square")

    n = work.shape[0]
    np.fill_diagonal(work, 0)

    atoms = _decompose_by_clique_separators(work)
    order: List[int] = []
    elimination_cliques: List[np.ndarray] = []

    for atom in atoms:
        local_order, local_cliques = _triangulate_block(work, atom, method)
        order.extend(local_order)
        elimination_cliques.extend(local_cliques)

    cliques = _maximal_cliques(elimination_cliques)

    if max_clique_width is not None and max_clique_width > 0:
        processed: List[np.ndarray] = []
        for clique in cliques:
            processed.extend(_split_clique(clique, int(max_clique_width)))
        cliques = _maximal_cliques(processed)

    triangulated = np.zeros((n, n), dtype=int)
    for clique in cliques:
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u = int(clique[i])
                v = int(clique[j])
                triangulated[u, v] = 1
                triangulated[v, u] = 1

    np.fill_diagonal(triangulated, 0)
    return triangulated, np.array(order, dtype=int), cliques


def junction_tree(cliques: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """Build a clique tree (maximum spanning tree by separator size)."""
    n = len(cliques)
    if n <= 1:
        return [], []

    edges: List[Tuple[int, int, int, np.ndarray]] = []
    for i in range(n):
        set_i = set(cliques[i].tolist())
        for j in range(i + 1, n):
            inter = _as_int_array(set_i.intersection(cliques[j].tolist()))
            if len(inter) > 0:
                edges.append((len(inter), i, j, inter))

    edges.sort(key=lambda x: (-x[0], x[1], x[2]))

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    tree_edges: List[Tuple[int, int]] = []
    separators: List[np.ndarray] = []
    for _, i, j, sep in edges:
        if union(i, j):
            tree_edges.append((i, j))
            separators.append(sep)
            if len(tree_edges) == n - 1:
                break

    # If disconnected (should be rare), connect components with empty separators.
    roots = {find(i) for i in range(n)}
    root_list = list(roots)
    for i in range(len(root_list) - 1):
        a = root_list[i]
        b = root_list[i + 1]
        union(a, b)
        tree_edges.append((a, b))
        separators.append(np.array([], dtype=int))

    return tree_edges, separators


def _order_cliques_for_sampling(
    cliques: List[np.ndarray],
    tree_edges: List[Tuple[int, int]],
) -> Tuple[List[int], Dict[int, int]]:
    n = len(cliques)
    if n == 0:
        return [], {}

    # Root on largest clique for stable ordering.
    root = max(range(n), key=lambda i: (len(cliques[i]), -i))

    neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in tree_edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    parent: Dict[int, int] = {root: -1}
    order: List[int] = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in sorted(neighbors[node]):
            if nxt in parent:
                continue
            parent[nxt] = node
            queue.append(nxt)

    for i in range(n):
        if i not in parent:
            parent[i] = -1
            order.append(i)

    return order, parent


def _compute_table_from_data(
    data: np.ndarray,
    overlap_vars: np.ndarray,
    new_vars: np.ndarray,
    cardinality: np.ndarray,
    alpha: float,
) -> np.ndarray:
    data = np.asarray(data, dtype=int)
    overlap_vars = np.asarray(overlap_vars, dtype=int)
    new_vars = np.asarray(new_vars, dtype=int)

    n_new_configs = int(np.prod(cardinality[new_vars])) if len(new_vars) else 1

    if len(overlap_vars) == 0:
        idx = _config_indices(data, new_vars, cardinality)
        counts = np.bincount(idx, minlength=n_new_configs).astype(float)
        if alpha > 0:
            counts += alpha / max(n_new_configs, 1)
        total = counts.sum()
        if total == 0:
            return np.full(n_new_configs, 1.0 / n_new_configs)
        return counts / total

    n_overlap_configs = int(np.prod(cardinality[overlap_vars]))
    overlap_idx = _config_indices(data, overlap_vars, cardinality)
    new_idx = _config_indices(data, new_vars, cardinality)

    table = np.zeros((n_overlap_configs, n_new_configs), dtype=float)
    if alpha > 0:
        table += alpha / max(n_new_configs, 1)

    np.add.at(table, (overlap_idx, new_idx), 1.0)

    row_sums = table.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] == 0
    row_sums[zero_rows] = 1.0
    table = table / row_sums
    if np.any(zero_rows):
        table[zero_rows, :] = 1.0 / n_new_configs
    return table


def _compute_table_from_joint(
    assignments: np.ndarray,
    probabilities: np.ndarray,
    overlap_vars: np.ndarray,
    new_vars: np.ndarray,
    cardinality: np.ndarray,
) -> np.ndarray:
    n_new_configs = int(np.prod(cardinality[new_vars])) if len(new_vars) else 1

    if len(overlap_vars) == 0:
        idx = _config_indices(assignments, new_vars, cardinality)
        table = np.bincount(idx, weights=probabilities, minlength=n_new_configs).astype(float)
        total = table.sum()
        if total == 0:
            return np.full(n_new_configs, 1.0 / n_new_configs)
        return table / total

    n_overlap_configs = int(np.prod(cardinality[overlap_vars]))
    overlap_idx = _config_indices(assignments, overlap_vars, cardinality)
    new_idx = _config_indices(assignments, new_vars, cardinality)

    table = np.zeros((n_overlap_configs, n_new_configs), dtype=float)
    np.add.at(table, (overlap_idx, new_idx), probabilities)

    row_sums = table.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] == 0
    row_sums[zero_rows] = 1.0
    table = table / row_sums
    if np.any(zero_rows):
        table[zero_rows, :] = 1.0 / n_new_configs
    return table


@dataclass
class CliqueFactorization:
    structure: np.ndarray
    tables: List[np.ndarray]
    cliques: List[np.ndarray]
    separators: List[np.ndarray]


def bn_to_factorization(
    adjacency: np.ndarray,
    cardinality: np.ndarray,
    cpds: Dict[int, Dict],
    data: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    max_clique_width: Optional[int] = None,
    width_control: str = "split",
    triangulation_method: str = "min-fill",
) -> CliqueFactorization:
    """Convert a BN into FDA-style clique factorization."""
    if width_control not in {"split", "mi"}:
        raise ValueError("width_control must be 'split' or 'mi'")

    moral = moralize(adjacency)
    triangulated, _, cliques = triangulate(
        moral,
        cardinality,
        method=triangulation_method,
        max_clique_width=max_clique_width if width_control == "split" else None,
    )

    # Lightweight fallback for MI mode: currently uses the split policy when needed.
    if width_control == "mi" and max_clique_width is not None:
        new_cliques: List[np.ndarray] = []
        for clique in cliques:
            new_cliques.extend(_split_clique(clique, int(max_clique_width)))
        cliques = _maximal_cliques(new_cliques)

        triangulated = np.zeros_like(triangulated)
        for clique in cliques:
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u = int(clique[i])
                    v = int(clique[j])
                    triangulated[u, v] = 1
                    triangulated[v, u] = 1

    tree_edges, separators = junction_tree(cliques)
    order, parent = _order_cliques_for_sampling(cliques, tree_edges)

    adjacency_list: Dict[int, List[int]] = {i: [] for i in range(len(cliques))}
    for i, j in tree_edges:
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)

    if len(cliques) == 0:
        cliques = [_as_int_array([var]) for var in range(adjacency.shape[0])]
        tree_edges, separators = junction_tree(cliques)
        order, parent = _order_cliques_for_sampling(cliques, tree_edges)

    max_clique_size = max((len(c) for c in cliques), default=1)
    structure = np.zeros((len(order), 2 + 2 * max_clique_size), dtype=int)
    tables: List[np.ndarray] = []

    sampled_vars: set[int] = set()

    if data is None:
        if not cpds:
            raise RuntimeError("CPDs are required when data is None")
        assignments, probabilities = _joint_distribution_from_bn(adjacency, cpds, cardinality)
    else:
        assignments, probabilities = np.empty((0, adjacency.shape[0]), dtype=int), np.empty(0)

    for row_idx, clique_idx in enumerate(order):
        clique = cliques[clique_idx]

        p = parent.get(clique_idx, -1)
        if p >= 0:
            overlap = _as_int_array(set(clique.tolist()).intersection(cliques[p].tolist()))
        else:
            overlap = np.array([], dtype=int)

        # Ensure every clique contributes at least one variable.
        new = np.array([v for v in clique if v not in sampled_vars], dtype=int)
        if len(new) == 0:
            # fallback to clique-parent difference
            new = np.array([v for v in clique if v not in overlap], dtype=int)
            if len(new) == 0:
                new = np.array([int(clique[0])], dtype=int)

        overlap = np.array([v for v in overlap if v not in new], dtype=int)

        n_overlap = int(len(overlap))
        n_new = int(len(new))
        structure[row_idx, 0] = n_overlap
        structure[row_idx, 1] = n_new
        structure[row_idx, 2 : 2 + n_overlap] = overlap
        structure[row_idx, 2 + n_overlap : 2 + n_overlap + n_new] = new

        if data is not None:
            table = _compute_table_from_data(data, overlap, new, cardinality, alpha)
        else:
            table = _compute_table_from_joint(assignments, probabilities, overlap, new, cardinality)

        tables.append(table)
        sampled_vars.update(new.tolist())

    return CliqueFactorization(
        structure=structure,
        tables=tables,
        cliques=cliques,
        separators=separators,
    )
