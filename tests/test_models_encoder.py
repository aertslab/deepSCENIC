"""Tests for the encoder network."""

import torch

from deepscenic.models._encoder import InferenceNet


class TestInferenceNet:
    """Tests for InferenceNet encoder."""

    def test_output_shape(self):
        """Output shape should match input TF dimension."""
        encoder = InferenceNet(n_hidden=128)
        x_tf = torch.randn(64, 1390)  # (cells, TFs)
        z_tf, mu, logvar = encoder(x_tf)
        assert z_tf.shape == (64, 1390)
        assert mu.shape == (64, 1390)
        assert logvar.shape == (64, 1390)

    def test_small_input(self):
        """Should work with small inputs."""
        encoder = InferenceNet(n_hidden=32)
        x_tf = torch.randn(4, 10)
        z_tf, mu, logvar = encoder(x_tf)
        assert z_tf.shape == (4, 10)

    def test_use_mean_deterministic(self):
        """When use_mean=True, output should be deterministic."""
        encoder = InferenceNet(n_hidden=64)
        x_tf = torch.randn(16, 50)
        z1, mu1, _ = encoder(x_tf, use_mean=True)
        z2, mu2, _ = encoder(x_tf, use_mean=True)
        assert torch.allclose(z1, z2)
        assert torch.allclose(z1, mu1)

    def test_sampling_stochastic(self):
        """When use_mean=False, sampling should be stochastic."""
        encoder = InferenceNet(n_hidden=64)
        x_tf = torch.randn(16, 50)
        z1, _, _ = encoder(x_tf, use_mean=False)
        z2, _, _ = encoder(x_tf, use_mean=False)
        # With high probability, samples should differ
        assert not torch.allclose(z1, z2)

    def test_different_hidden_sizes(self):
        """Should work with different hidden dimensions."""
        for n_hidden in [32, 64, 128]:
            encoder = InferenceNet(n_hidden=n_hidden)
            x_tf = torch.randn(4, 10)
            z_tf, mu, logvar = encoder(x_tf)
            assert z_tf.shape == (4, 10)

    def test_batch_size_one(self):
        """Should work with batch size 1."""
        encoder = InferenceNet(n_hidden=64)
        x_tf = torch.randn(1, 100)
        z_tf, mu, logvar = encoder(x_tf)
        assert z_tf.shape == (1, 100)

    def test_single_tf(self):
        """Should work with single TF."""
        encoder = InferenceNet(n_hidden=64)
        x_tf = torch.randn(32, 1)
        z_tf, mu, logvar = encoder(x_tf)
        assert z_tf.shape == (32, 1)
