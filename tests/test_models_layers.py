"""Tests for core neural network layers."""

import torch

from deepscenic.models._layers import GaussianSampler, PositiveLinear, _pos_xavier_uniform


class TestPositiveXavierUniform:
    """Tests for positive Xavier uniform initialization."""

    def test_values_are_positive(self):
        """All initialized values should be non-negative."""
        tensor = torch.empty(100, 50)
        _pos_xavier_uniform(tensor)
        assert (tensor >= 0).all()

    def test_values_are_bounded(self):
        """Values should be bounded by Xavier uniform bounds."""
        tensor = torch.empty(100, 50)
        _pos_xavier_uniform(tensor)
        # a = sqrt(3) * sqrt(2 / (fan_in + fan_out))
        expected_bound = (3 * 2 / (100 + 50)) ** 0.5
        assert (tensor <= expected_bound * 1.01).all()  # Small tolerance


class TestPositiveLinear:
    """Tests for PositiveLinear layer."""

    def test_output_shape(self):
        """Output shape should be correct."""
        layer = PositiveLinear(128, 64)
        x = torch.randn(32, 128)
        out = layer(x)
        assert out.shape == (32, 64)

    def test_weights_are_positive_in_forward(self):
        """Effective weights in forward pass should be positive."""
        layer = PositiveLinear(10, 5)
        # Set some negative weights
        with torch.no_grad():
            layer.weight.fill_(-1.0)
        x = torch.ones(1, 10)
        out = layer(x)
        # With abs() applied, output should be positive
        assert (out > 0).all()

    def test_bias_when_requested(self):
        """Bias should exist when requested."""
        layer = PositiveLinear(10, 5, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (5,)


class TestGaussianSampler:
    """Tests for GaussianSampler layer."""

    def test_output_shape(self):
        """Output shapes should be correct."""
        sampler = GaussianSampler(128, 1)
        x = torch.randn(32, 10, 128)  # (batch, features, hidden)
        z, mu, logvar = sampler(x)
        assert z.shape == (32, 10)
        assert mu.shape == (32, 10)
        assert logvar.shape == (32, 10)

    def test_use_mean_returns_mu(self):
        """When use_mean=True, z should equal mu."""
        sampler = GaussianSampler(64, 1)
        x = torch.randn(16, 20, 64)
        z, mu, logvar = sampler(x, use_mean=True)
        assert torch.allclose(z, mu)

    def test_sampling_is_stochastic(self):
        """Multiple samples should differ when use_mean=False."""
        sampler = GaussianSampler(64, 1)
        x = torch.randn(16, 20, 64)
        z1, _, _ = sampler(x, use_mean=False)
        z2, _, _ = sampler(x, use_mean=False)
        # With high probability, samples should differ
        assert not torch.allclose(z1, z2)
