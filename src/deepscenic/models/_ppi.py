"""Protein-Protein Interaction network for TF activity modulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from torch_geometric.data import Batch


class PPIModule(ABC):
    """Abstract base class for PPI modules (pluggable interface)."""

    @abstractmethod
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute TF modulation weights from gene expression."""
        pass


class PPIgnn(nn.Module, PPIModule):
    """3-layer Graph Attention Network for PPI modeling.

    Models post-transcriptional TF regulation through protein interactions.
    Takes gene expression + PPI graph → TF activity modulation weights.

    Architecture:
        GATConv(1, hidden, heads=2) → ELU
        → GATConv(hidden*2, hidden, heads=2) → ELU
        → GATConv(hidden*2, 1, heads=1)
        → sigmoid → (batch, n_tfs)

    Parameters
    ----------
    hidden_channels
        Hidden dimension per attention head (default: 16)
    heads
        Number of attention heads (default: 2)

    Examples
    --------
    >>> ppi = PPIgnn(hidden_channels=16, heads=2)
    >>> # For batch of cells, use torch_geometric Batch
    >>> output = ppi(x, edge_index)  # (n_nodes_total, 1)
    """

    def __init__(self, hidden_channels: int = 16, heads: int = 2) -> None:
        super().__init__()
        # Import here to make torch_geometric optional at import time
        from torch_geometric.nn import GATConv

        self.conv1 = GATConv(
            in_channels=1,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            add_self_loops=True,
        )
        self.conv3 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=1,
            heads=1,
            concat=False,
            add_self_loops=True,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass through GAT layers.

        Parameters
        ----------
        x
            Node features (n_nodes, 1)
        edge_index
            Graph connectivity (2, n_edges)
        edge_weight
            Optional edge weights

        Returns
        -------
        Tensor
            Node outputs (n_nodes, 1)
        """
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index, edge_weight)
        return x


class NoPPI(nn.Module, PPIModule):
    """Dummy PPI module that returns ones (no modulation)."""

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return ones (no modulation effect)."""
        return torch.ones_like(x)


def build_ppi_batch(
    x_rna: Tensor,
    ppi_genes_idx: Tensor,
    edge_index: Tensor,
    device: torch.device,
) -> Batch:
    """Build PyG Batch from cell expression for PPI forward pass.

    Creates a batch of graphs where each graph represents one cell's
    PPI network with gene expression as node features.

    Parameters
    ----------
    x_rna
        Full RNA expression (n_cells, n_genes)
    ppi_genes_idx
        Indices of genes in PPI network
    edge_index
        PPI graph connectivity (2, n_edges)
    device
        Target device

    Returns
    -------
    Batch
        PyTorch Geometric batch for parallel processing
    """
    from torch_geometric.data import Batch, Data

    n_cells = x_rna.shape[0]
    data_list = []

    for i in range(n_cells):
        node_features = x_rna[i, ppi_genes_idx].unsqueeze(-1)
        data_list.append(Data(x=node_features, edge_index=edge_index).to(device))

    return Batch.from_data_list(data_list)


def extract_tf_weights(
    ppi_output: Tensor,
    n_cells: int,
    n_ppi_genes: int,
    ppi_tfs_idx_keys: Tensor,
    ppi_tfs_idx_values: Tensor,
    device: torch.device,
) -> Tensor:
    """Extract and reorder TF modulation weights from PPI output.

    Parameters
    ----------
    ppi_output
        Raw PPI network output (n_cells * n_ppi_genes, 1)
    n_cells
        Number of cells in batch
    n_ppi_genes
        Number of genes in PPI network
    ppi_tfs_idx_keys
        Indices of TFs within PPI gene order
    ppi_tfs_idx_values
        Target indices for reordering TFs
    device
        Target device

    Returns
    -------
    Tensor
        TF modulation weights (n_cells, n_tfs)
    """
    # Reshape to (n_cells, n_ppi_genes)
    ppi_out = ppi_output.view(n_cells, n_ppi_genes)

    # Extract TF subset and apply sigmoid
    x_rna_ppi = torch.sigmoid(ppi_out[:, ppi_tfs_idx_keys])

    # Reorder to match TF order expected by VAE
    x_rna_ppi = x_rna_ppi[:, ppi_tfs_idx_values].to(device)

    return x_rna_ppi
