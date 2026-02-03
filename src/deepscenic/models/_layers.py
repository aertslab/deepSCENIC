"""Core neural network layers for deepSCENIC."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_uniform_


def _pos_xavier_uniform(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Initialize with positive Xavier uniform values.

    Parameters
    ----------
    tensor
        Tensor to initialize
    gain
        Scaling factor

    Returns
    -------
    Tensor
        Initialized tensor with values in [0, a] where a = sqrt(6 / (fan_in + fan_out))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, 0, a)


class PositiveLinear(nn.Module):
    """Linear layer with positive-constrained weights.

    Forward pass uses weight.abs() to ensure all weights are positive.

    Parameters
    ----------
    in_features
        Size of each input sample
    out_features
        Size of each output sample
    bias
        If True, adds learnable bias. Default: False
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with positive Xavier uniform."""
        _pos_xavier_uniform(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with positive weight constraint."""
        return F.linear(x, self.weight.abs(), self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class GaussianSampler(nn.Module):
    """VAE reparameterization trick: sample z ~ N(mu, sigma).

    Outputs mu, logvar and samples z = mu + std * eps where eps ~ N(0, 1).

    Parameters
    ----------
    in_features
        Input dimension
    out_features
        Output dimension (latent dimension)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.mu_layer = PositiveLinear(in_features, out_features, bias=False)
        self.logvar_layer = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, use_mean: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with reparameterization.

        Parameters
        ----------
        x
            Input tensor of shape (batch, features, hidden)
        use_mean
            If True, return mu instead of sampled z (for inference)

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            z: Sampled latent (batch, features)
            mu: Mean (batch, features)
            logvar: Log variance (batch, features)
        """
        mu = self.mu_layer(x).squeeze(-1)
        logvar = self.logvar_layer(x).squeeze(-1)

        if use_mean:
            # non-training forward passes
            return mu, mu, logvar

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        return z, mu, logvar
