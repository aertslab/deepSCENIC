"""TF2rNet: DNA sequence to TF binding prediction."""

from __future__ import annotations

from torch import Tensor, nn


class MotifNet(nn.Module):
    """Context head for Enformer embeddings → TF binding predictions.

    Lightweight architecture that processes Enformer's output embeddings
    to predict TF binding strength per region.

    Architecture:
        (batch, bottleneck_size * emb_len) [flattened Enformer output]
        → reshape to (batch, 1, bottleneck_size * emb_len)
        → Conv1d(1, n_tfs, kernel=bottleneck_size)
        → reshape → Linear(emb_len, 1)
        → (batch, n_tfs)

    Parameters
    ----------
    n_tfs
        Number of transcription factors to predict
    bottleneck_size
        Enformer embedding dimension (default: 3072)
    emb_len
        Enformer output sequence length (default: 5)

    Examples
    --------
    >>> motifnet = MotifNet(n_tfs=1390, bottleneck_size=3072, emb_len=5)
    >>> emb = torch.randn(1000, 3072 * 5)  # Flattened Enformer output
    >>> tf_pred = motifnet(emb)
    >>> tf_pred.shape  # (1000, 1390)
    """

    def __init__(
        self,
        n_tfs: int,
        bottleneck_size: int = 3072,
        emb_len: int = 5,
    ) -> None:
        super().__init__()
        self.n_tfs = n_tfs
        self.bottleneck_size = bottleneck_size
        self.emb_len = emb_len

        # Conv1d that spans full bottleneck dimension
        # Input: (batch, 1, bottleneck_size * emb_len) → Output: (batch, n_tfs, emb_len)
        self.ctx_conv = nn.Conv1d(
            in_channels=1,
            out_channels=n_tfs,
            kernel_size=bottleneck_size,
            padding="valid",
            bias=False,
        )

        # Linear to collapse emb_len dimension
        self.ctx_linear = nn.Linear(emb_len, 1)

    def forward(self, emb: Tensor) -> Tensor:
        """
        Predict TF binding from Enformer embeddings.

        Parameters
        ----------
        emb
            Enformer embeddings, either:
            - (batch, bottleneck_size * emb_len) - flattened format (typical)
            - (batch, bottleneck_size, emb_len) - 3D format

        Returns
        -------
        Tensor
            TF binding predictios (batch, n_tfs)
        """
        batch_size = emb.shape[0]

        # Ensure float32 dtype for compatibility with model weights
        emb = emb.float()

        # Reshape to process each position separately
        # (batch, bottleneck * emb_len) → (batch * emb_len, 1, bottleneck)
        emb = emb.reshape(-1, 1, self.bottleneck_size)

        # Conv1d: (batch * emb_len, 1, bottleneck) → (batch * emb_len, n_tfs, 1)
        h = self.ctx_conv(emb)

        # Reshape to group positions back together
        # (batch * emb_len, n_tfs, 1) → (batch, emb_len, n_tfs) → (batch, n_tfs, emb_len)
        h = h.reshape(batch_size, self.emb_len, self.n_tfs).transpose(1, 2)

        # Linear: (batch, n_tfs, emb_len) → (batch, n_tfs, 1) → (batch, n_tfs)
        out = self.ctx_linear(h).squeeze(-1)

        return out.float()
