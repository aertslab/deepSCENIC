"""Tests for decoder networks."""

import torch

from deepscenic.models._decoder import GenerativeNet, GenerativeNetATAC


class TestGenerativeNet:
    """Tests for GenerativeNet (RNA decoder)."""

    def test_small_input(self):
        """Should work with small inputs."""
        decoder = GenerativeNet(n_hidden=32)  # (cells, genes)
        z = torch.randn(4, 100)
        x_rec = decoder(z)
        assert x_rec.shape == (4, 100)

    def test_different_hidden_sizes(self):
        """Should work with different hidden dimensions."""
        for n_hidden in [32, 64, 128]:
            decoder = GenerativeNet(n_hidden=n_hidden)
            z = torch.randn(4, 100)
            x_rec = decoder(z)
            assert x_rec.shape == (4, 100)

    def test_batch_size_one(self):
        """Should work with batch size 1."""
        decoder = GenerativeNet(n_hidden=64)
        z = torch.randn(1, 1000)
        x_rec = decoder(z)
        assert x_rec.shape == (1, 1000)

    def test_single_gene(self):
        """Should work with single gene."""
        decoder = GenerativeNet(n_hidden=64)
        z = torch.randn(32, 1)
        x_rec = decoder(z)
        assert x_rec.shape == (32, 1)


class TestGenerativeNetATAC:
    """Tests for GenerativeNetATAC (ATAC decoder)."""

    def test_output_shape(self):
        """Output shape should match input region dimension."""
        decoder = GenerativeNetATAC(n_hidden=32)
        z = torch.randn(8, 200)
        x_rec = decoder(z)
        assert x_rec.shape == (8, 200)

    def test_small_input(self):
        """Should work with small inputs."""
        decoder = GenerativeNetATAC(n_hidden=32)
        z = torch.randn(4, 100)
        x_rec = decoder(z)
        assert x_rec.shape == (4, 100)

    def test_use_bias(self):
        """Bias option should work for binary accessibility."""
        decoder_no_bias = GenerativeNetATAC(n_hidden=64, use_bias=False)
        decoder_with_bias = GenerativeNetATAC(n_hidden=64, use_bias=True)

        z = torch.randn(4, 50)
        out_no_bias = decoder_no_bias(z)
        out_with_bias = decoder_with_bias(z)

        assert out_no_bias.shape == out_with_bias.shape

    def test_different_hidden_sizes(self):
        """Should work with different hidden dimensions."""
        for n_hidden in [32, 64, 128]:
            decoder = GenerativeNetATAC(n_hidden=n_hidden)
            z = torch.randn(4, 100)
            x_rec = decoder(z)
            assert x_rec.shape == (4, 100)

    def test_uses_standard_linear(self):
        """Should use standard Linear layers (not PositiveLinear)."""
        decoder = GenerativeNetATAC(n_hidden=64)
        # Check that the first layer is nn.Linear, not PositiveLinear
        from deepscenic.models._layers import PositiveLinear

        first_layer = decoder.mlp[0]
        assert isinstance(first_layer, torch.nn.Linear)
        assert not isinstance(first_layer, PositiveLinear)
