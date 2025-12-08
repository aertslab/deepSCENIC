"""PPI (Protein-Protein Interaction) network preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from mudata import MuData


def build_ppi_network(
    mdata: MuData,
    ppi_data: pd.DataFrame,
    confidence_threshold: float = 0.5,
    species: str = "mouse",
    source_col: str = "Source",
    target_col: str = "Target",
    confidence_col: str | None = "Conn",
    inplace: bool = True,
) -> MuData | None:
    r"""
    Build PPI network from interaction data and store in MuData.

    Filters interactions by confidence, matches genes to expression data,
    and stores the PyG-compatible edge index in mdata.uns.

    Parameters
    ----------
    mdata : MuData
        Input multimodal data with 'rna' modality.
    ppi_data : DataFrame
        Protein-protein interaction data with source/target columns.
        If confidence_col is provided, interactions below threshold are filtered.
    confidence_threshold : float, default=0.5
        Minimum confidence score for interactions (if confidence_col provided).
    species : {'mouse', 'human'}
        Species for gene name formatting.
        - 'mouse': Capitalizes first letter only (e.g., 'SOX2' -> 'Sox2')
        - 'human': Keeps uppercase (e.g., 'SOX2' -> 'SOX2')
    source_col : str, default='Source'
        Column name for source gene in ppi_data.
    target_col : str, default='Target'
        Column name for target gene in ppi_data.
    confidence_col : str or None, default='Conn'
        Column name for confidence score. If None, no filtering is applied
        (useful for BioGRID data which has no confidence scores).
    inplace : bool, default=True
        Whether to modify mdata in-place.

    Returns
    -------
    MuData or None
        If inplace=False, returns modified MuData.

    Notes
    -----
    Stores the following in mdata.uns:
    - 'ppi_edge_index': PyG-compatible edge index (2, n_edges)
    - 'ppi_genes_idx': Indices of genes in PPI network
    - 'ppi_tfs_idx_keys': TF indices within PPI genes
    - 'ppi_tfs_idx_values': Target TF indices for reordering
    - 'ppi_gene_order': Gene names in PPI node order

    Examples
    --------
    >>> # Load STRING PPI data
    >>> ppi_df = pd.read_csv("string_ppi.tsv", sep="\t")
    >>> ds.pp.build_ppi_network(mdata, ppi_df, species="mouse")

    >>> # BioGRID (no confidence scores)
    >>> biogrid_df = pd.read_csv("biogrid.tsv", sep="\t")
    >>> ds.pp.build_ppi_network(
    ...     mdata, biogrid_df, confidence_col=None, species="human"
    ... )
    """
    if not inplace:
        mdata = mdata.copy()

    if "rna" not in mdata.mod:
        raise ValueError("MuData must contain 'rna' modality")

    rna = mdata.mod["rna"]

    # Prepare PPI data
    ppi = ppi_data.copy()
    ppi = ppi.rename(columns={source_col: "Source", target_col: "Target"})

    # Filter by confidence if applicable
    if confidence_col is not None and confidence_col in ppi_data.columns:
        ppi["Conn"] = ppi_data[confidence_col]
        ppi = ppi[ppi["Conn"] >= confidence_threshold]

    # Format gene names based on species
    if species == "mouse":
        # Capitalize first letter only: SOX2 -> Sox2
        ppi["Source"] = ppi["Source"].apply(
            lambda x: x[0].upper() + x[1:].lower() if len(x) > 1 else x.upper()
        )
        ppi["Target"] = ppi["Target"].apply(
            lambda x: x[0].upper() + x[1:].lower() if len(x) > 1 else x.upper()
        )
    # human: keep as-is (uppercase)

    # Get all genes in dataset
    all_genes = set(rna.var_names)

    # Remove self-loops and filter to genes in data
    ppi = ppi[ppi["Source"] != ppi["Target"]]
    ppi = ppi[ppi["Source"].isin(all_genes)]
    ppi = ppi[ppi["Target"].isin(all_genes)]

    if len(ppi) == 0:
        raise ValueError("No PPI edges remain after filtering to genes in data")

    # Build networkx graph
    G = nx.from_pandas_edgelist(ppi, "Source", "Target")

    # Add isolated genes (genes with no interactions but in dataset)
    G.add_nodes_from(all_genes)  # Adds isolated nodes

    # Create node mapping (gene name -> index)
    node_list = list(G.nodes())
    node_to_idx = {gene: i for i, gene in enumerate(node_list)}

    # Convert to PyG edge index
    edge_index = _nx_to_pyg_edge_index(G, node_to_idx)

    # Get indices of PPI genes in the RNA var
    ppi_genes_idx = np.array([
        rna.var_names.get_loc(gene) for gene in node_list
        if gene in rna.var_names
    ])

    # Get TF information
    if "is_tf" not in rna.var.columns:
        raise ValueError(
            "TFs must be marked first. Run ds.pp.mark_tfs() before build_ppi_network()"
        )

    tf_names = rna.var_names[rna.var["is_tf"]].tolist()

    # Map TFs to their positions in PPI node list and original TF order
    ppi_tfs_idx_keys = []  # Index in PPI node list
    ppi_tfs_idx_values = []  # Index in TF order

    tf_order = rna.uns.get("tf_order", tf_names)
    tf_to_order_idx = {tf: i for i, tf in enumerate(tf_order)}

    for ppi_node_idx, gene in enumerate(node_list):
        if gene in tf_to_order_idx:
            ppi_tfs_idx_keys.append(ppi_node_idx)
            ppi_tfs_idx_values.append(tf_to_order_idx[gene])

    # Store in mdata.uns
    mdata.uns["ppi_edge_index"] = edge_index.numpy()
    mdata.uns["ppi_genes_idx"] = ppi_genes_idx
    mdata.uns["ppi_tfs_idx_keys"] = np.array(ppi_tfs_idx_keys)
    mdata.uns["ppi_tfs_idx_values"] = np.array(ppi_tfs_idx_values)
    mdata.uns["ppi_gene_order"] = node_list

    n_nodes = len(node_list)
    n_edges = edge_index.shape[1]
    n_tfs_in_ppi = len(ppi_tfs_idx_keys)

    print(f"Built PPI network: {n_nodes} nodes, {n_edges} edges")
    print(f"  {n_tfs_in_ppi} / {len(tf_order)} TFs found in PPI network")

    if not inplace:
        return mdata
    return None


def _nx_to_pyg_edge_index(
    G: nx.Graph,
    node_mapping: dict[str, int],
) -> torch.Tensor:
    """
    Convert networkx graph to PyTorch Geometric edge index.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph (will be converted to directed)
    node_mapping : dict
        Mapping from node names to indices

    Returns
    -------
    torch.Tensor
        Edge index tensor of shape (2, n_edges)
    """
    # Convert to directed graph (PyG expects directed)
    G_directed = G.to_directed() if not nx.is_directed(G) else G

    n_edges = G_directed.number_of_edges()
    edge_index = torch.empty((2, n_edges), dtype=torch.long)

    for i, (src, dst) in enumerate(G_directed.edges()):
        edge_index[0, i] = node_mapping[src]
        edge_index[1, i] = node_mapping[dst]

    return edge_index


def load_string_ppi(
    filepath: str,
    score_threshold: int = 400,
) -> pd.DataFrame:
    """
    Load STRING PPI data from file.

    Parameters
    ----------
    filepath : str
        Path to STRING protein links file.
    score_threshold : int, default=400
        Minimum combined score (STRING scores are 0-1000).

    Returns
    -------
    pd.DataFrame
        DataFrame with Source, Target, Conn columns.

    Notes
    -----
    STRING files typically have columns: protein1, protein2, combined_score.
    Gene names may need species prefix removed (e.g., '9606.ENSP...' for human).
    """
    df = pd.read_csv(filepath, sep=" ")

    # Rename columns
    df = df.rename(columns={
        "protein1": "Source",
        "protein2": "Target",
        "combined_score": "Conn",
    })

    # Filter by score
    df = df[df["Conn"] >= score_threshold]

    # Normalize score to 0-1
    df["Conn"] = df["Conn"] / 1000.0

    return df[["Source", "Target", "Conn"]]
