"""Decoder networks for deepSCENIC."""

from __future__ import annotations

from torch import Tensor, nn

from ._layers import PositiveLinear


class GenerativeNet(nn.Module):
    """Per-gene MLP decoder for RNA reconstruction.

    Decodes gene regulatory signal to gene expression. Each gene is processed
    independently through the same MLP.

    Architecture:
        (n_cells, n_genes, 1)
        → PositiveLinear(1, hidden) → Tanh
        → PositiveLinear(hidden, hidden) → Tanh
        → PositiveLinear(hidden, 1)
        → x_rec (n_cells, n_genes)

    Parameters
    ----------
    n_hidden
        Hidden layer dimension (default: 128)
    """

    def __init__(self, n_hidden: int = 128) -> None:
        super().__init__()
        self.n_hidden = n_hidden

        self.mlp = nn.Sequential(
            PositiveLinear(1, n_hidden, bias=False),
            nn.Tanh(),
            PositiveLinear(n_hidden, n_hidden, bias=False),
            nn.Tanh(),
            PositiveLinear(n_hidden, 1, bias=False),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Decode gene signal to expression.

        Parameters
        ----------
        z
            Gene regulatory signal (n_cells, n_genes)

        Returns
        -------
        Tensor
            Reconstructed expression (n_cells, n_genes)
        """
        # Reshape: (cells, genes) → (cells, genes, 1)
        z = z.unsqueeze(-1)
        x_rec = self.mlp(z)
        return x_rec.squeeze(-1)


class GenerativeNetATAC(nn.Module):
    """Per-region MLP decoder for ATAC reconstruction.

    Decodes region activity to chromatin accessibility.

    Architecture:
        (n_cells, n_regions, 1)
        → PositiveLinear(1, hidden) → Tanh
        → PositiveLinear(hidden, hidden) → Tanh
        → PositiveLinear(hidden, 1)
        → x_rec (n_cells, n_regions)

    Parameters
    ----------
    n_hidden
        Hidden layer dimension (default: 128)
    use_bias
        Whether to use bias in ALL layers (for binary accessibility).
        When True, all PositiveLinear layers have bias enabled.
    """

    def __init__(self, n_hidden: int = 128, use_bias: bool = False) -> None:
        super().__init__()
        self.n_hidden = n_hidden

        self.mlp = nn.Sequential(
            PositiveLinear(1, n_hidden, bias=use_bias),
            nn.Tanh(),
            PositiveLinear(n_hidden, n_hidden, bias=use_bias),
            nn.Tanh(),
            PositiveLinear(n_hidden, 1, bias=use_bias),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Decode region activity to accessibility.

        Parameters
        ----------
        z
            Region activity (n_cells, n_regions)

        Returns
        -------
        Tensor
            Reconstructed accessibility (n_cells, n_regions)
        """
        z = z.unsqueeze(-1)
        x_rec = self.mlp(z)
        return x_rec.squeeze(-1)
