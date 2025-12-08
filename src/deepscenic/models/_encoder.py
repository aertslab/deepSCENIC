"""TF encoder network for deepSCENIC."""

from __future__ import annotations

from torch import Tensor, nn

from ._layers import GaussianSampler, PositiveLinear


class InferenceNet(nn.Module):
    """Per-TF MLP encoder with VAE reparameterization.

    Transforms TF expression to latent TF activity. Each TF is processed
    independently through the same MLP (weight sharing via reshaping).

    Architecture:
        (n_cells, n_tfs, 1)
        → PositiveLinear(1, hidden) → Tanh
        → PositiveLinear(hidden, hidden) → Tanh
        → GaussianSampler(hidden, 1)
        → z_tf (n_cells, n_tfs)

    Parameters
    ----------
    n_hidden
        Hidden layer dimension (default: 128)

    Examples
    --------
    >>> encoder = InferenceNet(n_hidden=128)
    >>> x_tf = torch.randn(64, 1390)  # (cells, TFs)
    >>> z_tf, mu, logvar = encoder(x_tf)
    >>> z_tf.shape  # (64, 1390)
    """

    def __init__(self, n_hidden: int = 128) -> None:
        super().__init__()
        self.n_hidden = n_hidden

        self.mlp = nn.Sequential(
            PositiveLinear(1, n_hidden, bias=False),
            nn.Tanh(),
            PositiveLinear(n_hidden, n_hidden, bias=False),
            nn.Tanh(),
        )
        self.gaussian = GaussianSampler(n_hidden, 1)

    def forward(
        self,
        x: Tensor,
        use_mean: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Encode TF expression to latent activity.

        Parameters
        ----------
        x
            TF expression matrix (n_cells, n_tfs)
        use_mean
            If True, return mu instead of sampled z (for inference)

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            z_tf: Latent TF activity (n_cells, n_tfs)
            mu: Mean (n_cells, n_tfs)
            logvar: Log variance (n_cells, n_tfs)
        """
        # Reshape for per-TF processing: (cells, tfs) → (cells, tfs, 1)
        x = x.unsqueeze(-1)

        # MLP forward
        h = self.mlp(x)  # (cells, tfs, hidden)

        # Gaussian sampling
        z, mu, logvar = self.gaussian(h, use_mean=use_mean)

        return z, mu, logvar
