"""GRN network visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from deepscenic.tl._model import DeepSCENICModel

from .._utils import COLORS, savefig_or_show, setup_axes


def network_grn(
    model: DeepSCENICModel,
    *,
    tfs: list[str] | None = None,
    genes: list[str] | None = None,
    top_k: int = 20,
    threshold: float = 0.1,
    layout: str = "spring",
    node_size_tf: int = 800,
    node_size_gene: int = 300,
    tf_color: str = "#E64B35",
    gene_color: str = "#4DBBD5",
    edge_cmap: str = "Greys",
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (12, 12),
    **kwargs,
) -> Axes | Figure | None:
    """
    Plot GRN as a network graph.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    tfs
        TFs to include.
    genes
        Target genes to include.
    top_k
        Number of top edges to show per TF.
    threshold
        Minimum edge weight to include.
    layout
        Network layout: 'spring', 'circular', 'kamada_kawai'.
    node_size_tf
        Size of TF nodes.
    node_size_gene
        Size of gene nodes.
    tf_color
        Color for TF nodes.
    gene_color
        Color for gene nodes.
    edge_cmap
        Colormap for edge weights.
    ax
        Pre-existing axes.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure instead of Axes.
    figsize
        Figure size.
    **kwargs
        Passed to nx.draw_networkx.

    Returns
    -------
    Axes, Figure, or None

    Examples
    --------
    >>> import deepscenic as ds
    >>> model = ds.tl.load_model("model.pt")
    >>> ds.pl.network_grn(model, top_k=15)
    """
    from deepscenic.tl import get_tf_targets

    # Build network
    G = nx.DiGraph()

    # Get TFs to include
    tf_list = tfs if tfs is not None else model.tf_names[:20]

    for tf in tf_list:
        targets = get_tf_targets(model, tf, e1_threshold=threshold, top_k=top_k)
        if len(targets) == 0:
            continue

        G.add_node(tf, node_type="tf")

        for _, row in targets.iterrows():
            gene = row["gene"]
            if genes is not None and gene not in genes:
                continue
            G.add_node(gene, node_type="gene")
            G.add_edge(tf, gene, weight=row["combined_weight"])

    if len(G) == 0:
        raise ValueError("No edges found with given parameters")

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Separate node types
    tf_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "tf"]
    gene_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "gene"]

    # Edge weights for coloring
    edges = G.edges(data=True)
    edge_weights = [d["weight"] for _, _, d in edges]

    # Create plot
    fig, ax = setup_axes(ax, figsize=figsize)

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_weights,
        edge_cmap=plt.cm.get_cmap(edge_cmap),
        alpha=0.6,
        arrows=True,
        arrowsize=10,
        width=1.5,
    )

    # Draw TF nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=tf_nodes,
        node_color=tf_color,
        node_size=node_size_tf,
        node_shape="s",  # square for TFs
    )

    # Draw gene nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=gene_nodes,
        node_color=gene_color,
        node_size=node_size_gene,
        node_shape="o",
    )

    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    ax.set_title("Gene Regulatory Network")
    ax.axis("off")

    savefig_or_show("network_grn", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None


def network_tf_targets(
    model: DeepSCENICModel,
    tf: str,
    *,
    top_k: int = 30,
    threshold: float = 0.0,
    show_regions: bool = False,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (10, 10),
) -> Axes | Figure | None:
    """
    Plot network of a single TF's targets.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    tf
        TF name to visualize.
    top_k
        Number of top targets.
    threshold
        Minimum weight threshold.
    show_regions
        Include region nodes in network.
    ax
        Pre-existing axes.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure instead of Axes.
    figsize
        Figure size.

    Returns
    -------
    Axes, Figure, or None

    Examples
    --------
    >>> import deepscenic as ds
    >>> model = ds.tl.load_model("model.pt")
    >>> ds.pl.network_tf_targets(model, "SOX2", top_k=20)
    """
    from deepscenic.tl import get_tf_targets

    targets = get_tf_targets(model, tf, e1_threshold=threshold, top_k=top_k)

    if len(targets) == 0:
        raise ValueError(f"No targets found for TF '{tf}'")

    G = nx.DiGraph()
    G.add_node(tf, node_type="tf")

    if show_regions:
        for _, row in targets.iterrows():
            region = row["region"]
            gene = row["gene"]
            G.add_node(region, node_type="region")
            G.add_node(gene, node_type="gene")
            G.add_edge(tf, region, weight=row["E1_weight"])
            G.add_edge(region, gene, weight=row["E2_weight"])
    else:
        for _, row in targets.iterrows():
            gene = row["gene"]
            G.add_node(gene, node_type="gene")
            G.add_edge(tf, gene, weight=row["combined_weight"])

    # Layout
    pos = nx.spring_layout(G, k=2)

    fig, ax = setup_axes(ax, figsize=figsize)

    # Node colors by type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        ntype = G.nodes[node].get("node_type", "gene")
        if ntype == "tf":
            node_colors.append(COLORS["tf"])
            node_sizes.append(1000)
        elif ntype == "region":
            node_colors.append(COLORS["region"])
            node_sizes.append(400)
        else:
            node_colors.append(COLORS["gene"])
            node_sizes.append(500)

    # Draw
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=8,
        arrows=True,
    )

    ax.set_title(f"Targets of {tf}")
    ax.axis("off")

    savefig_or_show(f"network_tf_{tf}", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None


def network_gene_regulators(
    model: DeepSCENICModel,
    gene: str,
    *,
    top_k: int = 20,
    threshold: float = 0.0,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
    figsize: tuple[float, float] = (10, 10),
) -> Axes | Figure | None:
    """
    Plot network of TFs regulating a gene.

    Parameters
    ----------
    model
        Trained DeepSCENICModel.
    gene
        Gene name to visualize.
    top_k
        Number of top regulators.
    threshold
        Minimum weight threshold.
    ax
        Pre-existing axes.
    show
        Display figure.
    save
        Save figure.
    return_fig
        Return Figure instead of Axes.
    figsize
        Figure size.

    Returns
    -------
    Axes, Figure, or None

    Examples
    --------
    >>> import deepscenic as ds
    >>> model = ds.tl.load_model("model.pt")
    >>> ds.pl.network_gene_regulators(model, "NANOG")
    """
    from deepscenic.tl import get_gene_regulators

    regulators = get_gene_regulators(model, gene, e1_threshold=threshold, top_k=top_k)

    if len(regulators) == 0:
        raise ValueError(f"No regulators found for gene '{gene}'")

    G = nx.DiGraph()
    G.add_node(gene, node_type="gene")

    for _, row in regulators.iterrows():
        tf = row["tf"]
        G.add_node(tf, node_type="tf")
        G.add_edge(tf, gene, weight=row["combined_weight"])

    pos = nx.spring_layout(G, k=2)

    fig, ax = setup_axes(ax, figsize=figsize)

    node_colors = [COLORS["tf"] if G.nodes[n].get("node_type") == "tf" else COLORS["gene"] for n in G.nodes()]
    node_sizes = [800 if G.nodes[n].get("node_type") == "tf" else 1000 for n in G.nodes()]

    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=8,
        arrows=True,
    )

    ax.set_title(f"Regulators of {gene}")
    ax.axis("off")

    savefig_or_show(f"network_gene_{gene}", show=show, save=save)

    if return_fig:
        return fig
    if show is False:
        return ax
    return None
