"""Tests for TF2rNet (MotifNet)."""

import torch

from deepscenic.models._tf2rnet import MotifNet


class TestMotifNet:
    """Tests for MotifNet context head."""

    def test_output_shape_flattened_input(self):
        """Output shape should be (batch, n_tfs) for flattened input."""
        motifnet = MotifNet(n_tfs=10, bottleneck_size=32, emb_len=3)
        emb = torch.randn(8, 32 * 3)  # Flattened input
        tf_pred = motifnet(emb)
        assert tf_pred.shape == (8, 10)

    def test_output_shape_3d_input(self):
        """Output shape should be (batch, n_tfs) for 3D input."""
        motifnet = MotifNet(n_tfs=10, bottleneck_size=32, emb_len=3)
        emb = torch.randn(8, 32, 3)  # 3D input
        tf_pred = motifnet(emb)
        assert tf_pred.shape == (8, 10)

    def test_different_emb_len(self):
        """Should work with different embedding lengths."""
        for emb_len in [1, 3, 5]:
            motifnet = MotifNet(n_tfs=5, bottleneck_size=16, emb_len=emb_len)
            emb = torch.randn(4, 16 * emb_len)
            tf_pred = motifnet(emb)
            assert tf_pred.shape == (4, 5)

    def test_half_precision_input(self):
        """MotifNet should handle float16 input and return float32 output."""
        motifnet = MotifNet(n_tfs=5, bottleneck_size=16, emb_len=2)
        emb = torch.randn(4, 16 * 2).half()
        tf_pred = motifnet(emb)
        assert tf_pred.dtype == torch.float32
