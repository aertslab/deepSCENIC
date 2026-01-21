"""Main VAE model for deepSCENIC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ._decoder import GenerativeNet, GenerativeNetATAC
from ._encoder import InferenceNet
from ._ppi import PPIgnn, build_ppi_batch, extract_tf_weights

if TYPE_CHECKING:
    pass


@dataclass
class VAEOutput:
    """Container for VAE forward pass outputs."""

    x_rna_rec: Tensor  # Reconstructed RNA (n_cells, n_genes)
    x_atac_rec: Tensor  # Reconstructed ATAC (n_cells, n_regions)
    z_tf: Tensor  # Latent TF activity (n_cells, n_tfs)
    mu: Tensor  # Encoder mean (n_cells, n_tfs)
    logvar: Tensor  # Encoder log variance (n_cells, n_tfs)
    enh_act: Tensor  # Region activity (n_cells, n_regions)
    z_rna: Tensor  # Gene regulatory signal (n_cells, n_genes)
    x_rna_ppi: Tensor  # PPI modulation weights (n_cells, n_tfs)


class DeepSCENICVAE(nn.Module):
    """Variational Autoencoder for GRN inference.

    Combines:
    - InferenceNet: TF expression → latent TF activity
    - GenerativeNet: Gene signal → RNA reconstruction
    - GenerativeNetATAC: Region activity → ATAC reconstruction
    - PPIgnn (optional): PPI-based TF modulation

    The GRN matrices (E1: TF→region, E2: region→gene) are passed during
    forward rather than stored, as E1 comes from TF2rNet and is cached.

    Parameters
    ----------
    n_tfs
        Number of transcription factors
    n_genes
        Number of genes for reconstruction
    n_regions
        Number of chromatin regions
    r2g_indices
        Sparse indices for region→gene links (2, n_links)
    r2g_distances
        Distance penalties for region→gene links (n_links,)
    tf_indices
        Indices of TFs in the gene expression matrix
    gene_indices
        Indices of genes to reconstruct
    ppi_edge_index
        PPI graph edges (2, n_edges), optional
    ppi_genes_idx
        Indices of genes in PPI network, optional
    ppi_tfs_idx_keys
        TF indices within PPI genes, optional
    ppi_tfs_idx_values
        Target TF indices for reordering, optional
    n_hidden
        MLP hidden dimension (default: 128)
    use_ppi
        Whether to use PPI network for TF modulation
    binary_atac
        Whether ATAC is binary (adds bias to decoder)
    n_batches
        Number of batches for batch correction (0 = no correction)

    Attributes
    ----------
    adj_E2
        Learnable region→gene weights (n_links,)
    """

    def __init__(
        self,
        n_tfs: int,
        n_genes: int,
        n_regions: int,
        r2g_indices: Tensor,
        r2g_distances: Tensor,
        tf_indices: Tensor,
        gene_indices: Tensor,
        ppi_edge_index: Tensor | None = None,
        ppi_genes_idx: Tensor | None = None,
        ppi_tfs_idx_keys: Tensor | None = None,
        ppi_tfs_idx_values: Tensor | None = None,
        n_hidden: int = 128,
        use_ppi: bool = True,
        binary_atac: bool = False,
        n_batches: int = 0,
    ) -> None:
        super().__init__()

        self.n_tfs = n_tfs
        self.n_genes = n_genes
        self.n_regions = n_regions
        self.n_hidden = n_hidden
        self.use_ppi = use_ppi and ppi_edge_index is not None
        self.n_batches = n_batches

        # Store indices as buffers (not parameters)
        self.register_buffer("tf_indices", tf_indices)
        self.register_buffer("gene_indices", gene_indices)
        self.register_buffer("r2g_indices", r2g_indices)
        self.register_buffer("r2g_distances", r2g_distances)

        # Learnable E2 weights (region→gene)
        n_links = r2g_indices.shape[1]
        self.adj_E2 = nn.Parameter(torch.zeros(n_links) + 1e-4)

        # Core networks
        self.encoder = InferenceNet(n_hidden=n_hidden)
        self.decoder_rna = GenerativeNet(n_hidden=n_hidden)
        self.decoder_atac = GenerativeNetATAC(n_hidden=n_hidden, use_bias=binary_atac)

        # PPI network (pluggable)
        if self.use_ppi:
            self.ppi = PPIgnn(hidden_channels=n_hidden, heads=2)
            self.register_buffer("ppi_edge_index", ppi_edge_index)
            self.register_buffer("ppi_genes_idx", ppi_genes_idx)
            self.register_buffer("ppi_tfs_idx_keys", ppi_tfs_idx_keys)
            self.register_buffer("ppi_tfs_idx_values", ppi_tfs_idx_values)
        else:
            self.ppi = None

        # Batch correction layers
        if n_batches > 0:
            self.batch_layer_atac = nn.Sequential(
                nn.Linear(n_batches + n_regions, 128),
                nn.Tanh(),
                nn.Linear(128, n_regions),
            )
            self.batch_layer_rna = nn.Sequential(
                nn.Linear(n_batches + n_genes, 128),
                nn.Tanh(),
                nn.Linear(128, n_genes),
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for Linear/Conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.xavier_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _region_to_gene(self, enh_act: Tensor) -> Tensor:
        """Compute region→gene signal using scatter_add (memory efficient).

        Equivalent to enh_act @ E2 where E2 is a sparse (n_regions, n_genes) matrix,
        but avoids materializing any dense matrices during forward or backward.

        Parameters
        ----------
        enh_act
            Region activity tensor (n_cells, n_regions)

        Returns
        -------
        Tensor
            Gene regulatory signal (n_cells, n_genes)
        """
        n_cells = enh_act.shape[0]
        device = enh_act.device

        # r2g_indices: (2, n_links) - row 0 = region indices, row 1 = gene indices
        region_idx = self.r2g_indices[0]  # (n_links,)
        gene_idx = self.r2g_indices[1]  # (n_links,)
        weights = self.adj_E2.abs()  # (n_links,)

        # Gather region activities for each link and weight them
        # enh_act[:, region_idx]: (n_cells, n_links)
        # weights: (n_links,) broadcasts to (n_cells, n_links)
        link_values = enh_act[:, region_idx] * weights

        # Scatter-add to accumulate weighted contributions per gene
        z_rna = torch.zeros(n_cells, self.n_genes, device=device, dtype=enh_act.dtype)
        gene_idx_expanded = gene_idx.unsqueeze(0).expand(n_cells, -1)
        z_rna.scatter_add_(1, gene_idx_expanded, link_values)

        return z_rna

    def forward(
        self,
        x_rna: Tensor,
        x_atac: Tensor,
        adj_E1: Tensor,
        use_ppi: bool = True,
        use_mean: bool = False,
        ppi_device: torch.device | None = None,
        batch_id: Tensor | None = None,
    ) -> VAEOutput:
        """
        Forward pass through VAE.

        Parameters
        ----------
        x_rna
            RNA expression (n_cells, n_genes)
        x_atac
            ATAC accessibility (n_cells, n_regions)
        adj_E1
            TF→region matrix from TF2rNet (n_regions, n_tfs)
        use_ppi
            Whether to apply PPI modulation this forward pass
        use_mean
            If True, use encoder mean instead of sampling
        ppi_device
            Device for PPI computation (can differ from main device)
        batch_id
            One-hot batch identifiers (n_cells, n_batches) for batch correction

        Returns
        -------
        VAEOutput
            Container with all intermediate and final outputs
        """
        device = x_rna.device
        n_cells = x_rna.shape[0]

        # Extract TF expression
        x_rna_tfs = x_rna[:, self.tf_indices]

        # PPI modulation (optional)
        if self.use_ppi and use_ppi and self.ppi is not None:
            ppi_dev = ppi_device or device
            batch = build_ppi_batch(x_rna, self.ppi_genes_idx, self.ppi_edge_index, ppi_dev)
            ppi_out = self.ppi(batch.x, batch.edge_index)
            x_rna_ppi = extract_tf_weights(
                ppi_out,
                n_cells,
                len(self.ppi_genes_idx),
                self.ppi_tfs_idx_keys,
                self.ppi_tfs_idx_values,
                device,
            )
            x_rna_tfs = x_rna_tfs * x_rna_ppi
        else:
            x_rna_ppi = torch.ones_like(x_rna_tfs)

        # Encoder: TF expression → latent TF activity
        z_tf, mu, logvar = self.encoder(x_rna_tfs, use_mean=use_mean)

        # TF activity → region activity via E1
        enh_act = z_tf @ adj_E1.T

        # Region activity → gene signal via E2 (scatter_add for memory efficiency)
        z_rna = self._region_to_gene(enh_act)

        # Decoders
        x_rna_rec = self.decoder_rna(z_rna)
        x_atac_rec = self.decoder_atac(enh_act)

        # Apply batch correction if enabled
        if self.n_batches > 0 and batch_id is not None:
            x_atac_rec = self.batch_layer_atac(torch.cat([batch_id, x_atac_rec], dim=1))
            x_rna_rec = self.batch_layer_rna(torch.cat([batch_id, x_rna_rec], dim=1))

        return VAEOutput(
            x_rna_rec=x_rna_rec,
            x_atac_rec=x_atac_rec,
            z_tf=z_tf,
            mu=mu,
            logvar=logvar,
            enh_act=enh_act,
            z_rna=z_rna,
            x_rna_ppi=x_rna_ppi,
        )
