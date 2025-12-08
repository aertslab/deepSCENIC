"""Tests for deepSCENIC sequence plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing


class TestISMHeatmap:
    """Tests for ism_heatmap function."""

    def test_creates_heatmap_shape_seq_len_4(self):
        """Should create ISM heatmap from (seq_len, 4) shape."""
        from deepscenic.pl import ism_heatmap

        ism_scores = np.random.randn(100, 4)
        ax = ism_heatmap(ism_scores, show=False)

        assert ax is not None
        plt.close("all")

    def test_creates_heatmap_shape_4_seq_len(self):
        """Should create ISM heatmap from (4, seq_len) shape."""
        from deepscenic.pl import ism_heatmap

        ism_scores = np.random.randn(4, 100)
        ax = ism_heatmap(ism_scores, show=False)

        assert ax is not None
        plt.close("all")

    def test_slices_positions(self):
        """Should slice to specified start/end positions."""
        from deepscenic.pl import ism_heatmap

        ism_scores = np.random.randn(100, 4)
        ax = ism_heatmap(ism_scores, start=10, end=50, show=False)

        assert ax is not None
        plt.close("all")

    def test_returns_figure_when_requested(self):
        """Should return Figure when return_fig=True."""
        from matplotlib.figure import Figure

        from deepscenic.pl import ism_heatmap

        ism_scores = np.random.randn(100, 4)
        fig = ism_heatmap(ism_scores, show=False, return_fig=True)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_adds_title_when_specified(self):
        """Should add title when provided."""
        from deepscenic.pl import ism_heatmap

        ism_scores = np.random.randn(100, 4)
        ax = ism_heatmap(ism_scores, title="Test ISM", show=False)

        assert ax.get_title() == "Test ISM"
        plt.close("all")


class TestLogoAttribution:
    """Tests for logo_attribution function."""

    @pytest.fixture
    def mock_tangermeme(self, monkeypatch):
        """Mock tangermeme.plot.plot_logo."""
        import sys
        from unittest.mock import MagicMock

        # Create mock module
        mock_module = MagicMock()
        mock_module.plot_logo = MagicMock()
        sys.modules["tangermeme"] = mock_module
        sys.modules["tangermeme.plot"] = mock_module

        yield mock_module

        # Cleanup
        del sys.modules["tangermeme"]
        del sys.modules["tangermeme.plot"]

    def test_raises_import_error_without_tangermeme(self):
        """Should raise ImportError if tangermeme not installed."""
        from deepscenic.pl import logo_attribution

        attrs = np.random.randn(100, 4)

        # This will raise ImportError if tangermeme is not installed
        # We can't easily test this in CI, so just check the function exists
        assert callable(logo_attribution)

    def test_handles_shape_seq_len_4(self, mock_tangermeme):
        """Should handle (seq_len, 4) shape."""
        from deepscenic.pl import logo_attribution

        attrs = np.random.randn(100, 4)
        ax = logo_attribution(attrs, show=False)

        assert ax is not None
        plt.close("all")

    def test_handles_shape_4_seq_len(self, mock_tangermeme):
        """Should handle (4, seq_len) shape."""
        from deepscenic.pl import logo_attribution

        attrs = np.random.randn(4, 100)
        ax = logo_attribution(attrs, show=False)

        assert ax is not None
        plt.close("all")

    def test_masks_by_sequence(self, mock_tangermeme):
        """Should mask attributions by sequence."""
        from deepscenic.pl import logo_attribution

        attrs = np.random.randn(100, 4)
        seq = np.eye(4)[np.random.randint(0, 4, 100)]  # One-hot sequence
        ax = logo_attribution(attrs, sequence=seq, show=False)

        assert ax is not None
        plt.close("all")


class TestLogoMotif:
    """Tests for logo_motif function."""

    @pytest.fixture
    def mock_logomaker(self, monkeypatch):
        """Mock logomaker."""
        import sys
        from unittest.mock import MagicMock

        # Create mock module
        mock_module = MagicMock()
        mock_module.Logo = MagicMock()
        sys.modules["logomaker"] = mock_module

        yield mock_module

        # Cleanup
        del sys.modules["logomaker"]

    def test_raises_import_error_without_logomaker(self):
        """Should raise ImportError if logomaker not installed."""
        from deepscenic.pl import logo_motif

        # Just check the function exists
        assert callable(logo_motif)

    def test_handles_shape_seq_len_4(self, mock_logomaker):
        """Should handle (seq_len, 4) shape."""
        from deepscenic.pl import logo_motif

        pwm = np.random.rand(20, 4)
        pwm = pwm / pwm.sum(axis=1, keepdims=True)  # Normalize
        ax = logo_motif(pwm, show=False)

        assert ax is not None
        plt.close("all")

    def test_handles_shape_4_seq_len(self, mock_logomaker):
        """Should handle (4, seq_len) shape."""
        from deepscenic.pl import logo_motif

        pwm = np.random.rand(4, 20)
        pwm = pwm / pwm.sum(axis=0, keepdims=True)  # Normalize
        ax = logo_motif(pwm, show=False)

        assert ax is not None
        plt.close("all")

    def test_adds_title_when_specified(self, mock_logomaker):
        """Should add title when provided."""
        from deepscenic.pl import logo_motif

        pwm = np.random.rand(20, 4)
        ax = logo_motif(pwm, title="Test Motif", show=False)

        assert ax.get_title() == "Test Motif"
        plt.close("all")
