"""
Visualization utilities for Bayesian networks.

Requires matplotlib and networkx (optional dependencies).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from bayes_nets.bayesian_network import BayesianNetwork


def plot_bayesian_network(
    bn: "BayesianNetwork",
    node_labels: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
    node_color: str = "#4C9BE8",
    edge_color: str = "#333333",
    font_color: str = "white",
    font_size: int = 14,
    node_size: int = 1200,
    title: Optional[str] = None,
    node_order: Optional[List[int]] = None,
    pos: Optional[Dict[int, tuple[float, float]]] = None,
    ax=None,
    **layout_kwargs: Any,
):
    """Draw the BN structure as a directed graph.

    Parameters
    ----------
    bn : BayesianNetwork
        The Bayesian network to visualize.
    node_labels : list of str, optional
        Human-readable labels for each variable.  Defaults to
        ``["X0", "X1", ...]``.
    figsize : tuple
        Figure size ``(width, height)`` in inches.
    node_color : str
        Matplotlib colour for nodes.
    edge_color : str
        Matplotlib colour for edges.
    font_color : str
        Font colour for node labels.
    font_size : int
        Font size for node labels.
    node_size : int
        Node size (passed to networkx draw).
    title : str, optional
        Plot title.
    node_order : list of int, optional
        Preferred node ordering used for deterministic fallback layouts.
    pos : dict, optional
        Explicit node positions to reuse across plots.
    ax : matplotlib Axes, optional
        Existing axes to draw on.  If *None* a new figure is created.
    **layout_kwargs
        Extra keyword arguments forwarded to
        ``networkx.drawing.nx_agraph.graphviz_layout`` if available,
        otherwise to ``networkx.spring_layout``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "matplotlib and networkx are required for visualization. "
            "Install them with: pip install matplotlib networkx"
        ) from exc

    if node_labels is None:
        node_labels = [f"X{i}" for i in range(bn.n_vars)]

    # Build a NetworkX DiGraph
    G = nx.DiGraph()
    G.add_nodes_from(range(bn.n_vars))
    for parent in range(bn.n_vars):
        for child in range(bn.n_vars):
            if bn.adjacency[parent, child]:
                G.add_edge(parent, child)

    label_map = {i: node_labels[i] for i in range(bn.n_vars)}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Choose layout
    if pos is None:
        if node_order is not None:
            ordered = [int(v) for v in node_order]
            if sorted(ordered) != list(range(bn.n_vars)):
                raise ValueError("node_order must contain every node exactly once")
            local_kwargs = {"nlist": [ordered], "scale": 1.0}
            local_kwargs.update(layout_kwargs)
            pos = nx.shell_layout(G, **local_kwargs)
        else:
            try:
                from networkx.drawing.nx_agraph import graphviz_layout  # type: ignore

                pos = graphviz_layout(G, prog="dot", **layout_kwargs)
            except (ImportError, Exception):
                pos = nx.spring_layout(G, seed=42, **layout_kwargs)

    nx.draw_networkx(
        G,
        pos=pos,
        labels=label_map,
        ax=ax,
        node_color=node_color,
        edge_color=edge_color,
        font_color=font_color,
        font_size=font_size,
        node_size=node_size,
        arrows=True,
        arrowsize=20,
    )

    if title:
        ax.set_title(title)

    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_marginals(
    bn: "BayesianNetwork",
    node_labels: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    color: str = "#4C9BE8",
):
    """Plot the marginal CPD distributions stored in the BN.

    For root nodes the 1-D marginal probability is plotted; for
    non-root nodes the CPD rows (one per parent configuration) are
    shown as grouped bar charts.

    Parameters
    ----------
    bn : BayesianNetwork
    node_labels : list of str, optional
    figsize : tuple, optional
        If *None*, auto-sized based on the number of variables.
    color : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization.") from exc

    if not bn.cpds:
        raise RuntimeError("CPDs have not been learned yet.")

    if node_labels is None:
        node_labels = [f"X{i}" for i in range(bn.n_vars)]

    n_vars = bn.n_vars
    cols = min(4, n_vars)
    rows = (n_vars + cols - 1) // cols

    if figsize is None:
        figsize = (cols * 4, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for var in range(n_vars):
        ax = axes[var]
        info = bn.cpds[var]
        cpd = info["cpd"]
        k = int(bn.cardinality[var])
        x = range(k)

        if cpd.ndim == 1:
            ax.bar(x, cpd, color=color)
        else:
            # Multiple rows: show mean ± std across parent configs
            mean = cpd.mean(axis=0)
            std = cpd.std(axis=0)
            ax.bar(x, mean, yerr=std, color=color, capsize=3)

        ax.set_title(node_labels[var])
        ax.set_xlabel("State")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_xticks(list(x))

    # Hide empty subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    return fig
