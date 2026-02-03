"""Decoder networks for deepSCENIC."""

from __future__ import annotations

from torch import Tensor, nn

from ._layers import PositiveLinear


class GenerativeNet(nn.Module):
    """Per-feature MLP decoder.

    Used for both RNA and ATAC reconstruction. Each feature (gene or region)
    is processed independently through the same MLP.

    Architecture:
        (n_cells, n_features, 1)
        → PositiveLinear(1, hidden) → Tanh
        → PositiveLinear(hidden, hidden) → Tanh
        → PositiveLinear(hidden, 1)
        → x_rec (n_cells, n_features)

    Parameters
    ----------
    n_hidden
        Hidden layer dimension (default: 128)
    use_bias
        Whether to use bias in layers (default: False).
        Set to True for binary ATAC accessibility.
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
        Decode latent signal to reconstruction.

        Parameters
        ----------
        z
            Latent signal (n_cells, n_features)

        Returns
        -------
        Tensor
            Reconstructed values (n_cells, n_features)
        """
        z = z.unsqueeze(-1)
        x_rec = self.mlp(z)
        return x_rec.squeeze(-1)
