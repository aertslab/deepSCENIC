"""Tests for GRN plotting functions."""

import inspect

import matplotlib
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

import deepscenic as ds


class TestHeatmapE1Removed:
    """Verify heatmap_e1 has been removed from the public API."""

    def test_not_in_pl_namespace(self):
        """heatmap_e1 should not be importable from ds.pl."""
        assert not hasattr(ds.pl, "heatmap_e1")

    def test_not_in_all(self):
        """heatmap_e1 should not be in ds.pl.__all__."""
        assert "heatmap_e1" not in ds.pl.__all__


class TestHeatmapE2:
    """Tests for heatmap_e2 function."""

    def test_returns_figure(self, mock_deepscenic_model):
        """Should return a figure when return_fig=True."""
        fig = ds.pl.heatmap_e2(mock_deepscenic_model, top_k=5, show=False, return_fig=True)
        assert fig is not None

    def test_returns_axes(self, mock_deepscenic_model):
        """Should return axes when show=False."""
        ax = ds.pl.heatmap_e2(mock_deepscenic_model, top_k=5, show=False)
        assert ax is not None

    def test_with_gene_list(self, mock_deepscenic_model):
        """Should work with explicit gene list."""
        genes = mock_deepscenic_model.gene_names[:3]
        fig = ds.pl.heatmap_e2(mock_deepscenic_model, genes=genes, show=False, return_fig=True)
        assert fig is not None


class TestHeatmapGRN:
    """Tests for heatmap_grn function."""

    def test_returns_figure_zscore(self, mock_deepscenic_model):
        """Should return a figure with default zscore normalization."""
        fig = ds.pl.heatmap_grn(mock_deepscenic_model, top_k=5, show=False, return_fig=True)
        assert fig is not None

    def test_returns_figure_raw(self, mock_deepscenic_model):
        """Should return a figure with normalize=None."""
        fig = ds.pl.heatmap_grn(mock_deepscenic_model, top_k=5, normalize=None, show=False, return_fig=True)
        assert fig is not None

    def test_invalid_normalize_raises(self, mock_deepscenic_model):
        """Should raise ValueError for unknown normalize method."""
        with pytest.raises(ValueError, match="Unknown normalize method"):
            ds.pl.heatmap_grn(mock_deepscenic_model, normalize="invalid", show=False, return_fig=True)

    def test_normalize_param_default(self):
        """Default normalize should be 'zscore'."""
        sig = inspect.signature(ds.pl.heatmap_grn)
        assert sig.parameters["normalize"].default == "zscore"

    def test_with_tf_and_gene_lists(self, mock_deepscenic_model):
        """Should work with explicit TF and gene lists."""
        tfs = mock_deepscenic_model.tf_names[:2]
        genes = mock_deepscenic_model.gene_names[:3]
        fig = ds.pl.heatmap_grn(mock_deepscenic_model, tfs=tfs, genes=genes, show=False, return_fig=True)
        assert fig is not None


class TestNetworkGRN:
    """Tests for network_grn function."""

    def test_returns_figure(self, mock_deepscenic_model):
        """Should return a figure when return_fig=True."""
        fig = ds.pl.network_grn(mock_deepscenic_model, top_k=5, show=False, return_fig=True)
        assert fig is not None

    def test_with_tf_list(self, mock_deepscenic_model):
        """Should work with explicit TF list."""
        tfs = mock_deepscenic_model.tf_names[:2]
        fig = ds.pl.network_grn(mock_deepscenic_model, tfs=tfs, top_k=5, show=False, return_fig=True)
        assert fig is not None


class TestNetworkTFTargets:
    """Tests for network_tf_targets function."""

    def test_returns_figure(self, mock_deepscenic_model):
        """Should return a figure when return_fig=True."""
        tf = mock_deepscenic_model.tf_names[0]
        fig = ds.pl.network_tf_targets(mock_deepscenic_model, tf, top_k=5, show=False, return_fig=True)
        assert fig is not None

    def test_invalid_tf_raises(self, mock_deepscenic_model):
        """Should raise ValueError for unknown TF."""
        with pytest.raises(ValueError):
            ds.pl.network_tf_targets(mock_deepscenic_model, "NONEXISTENT_TF", top_k=5, show=False, return_fig=True)


class TestNetworkGeneRegulators:
    """Tests for network_gene_regulators function."""

    def test_returns_figure(self, mock_deepscenic_model):
        """Should return a figure when return_fig=True."""
        gene = mock_deepscenic_model.gene_names[0]
        fig = ds.pl.network_gene_regulators(mock_deepscenic_model, gene, top_k=5, show=False, return_fig=True)
        assert fig is not None

    def test_invalid_gene_raises(self, mock_deepscenic_model):
        """Should raise ValueError for unknown gene."""
        with pytest.raises(ValueError):
            ds.pl.network_gene_regulators(
                mock_deepscenic_model, "NONEXISTENT_GENE", top_k=5, show=False, return_fig=True
            )


class TestHeatmapCelltypeActivity:
    """Tests for heatmap_celltype_activity defaults and behavior."""

    def test_defaults_match_legacy(self):
        """Default parameters should match legacy visualization."""
        sig = inspect.signature(ds.pl.heatmap_celltype_activity)
        assert sig.parameters["cmap"].default == "bwr"
        assert sig.parameters["center"].default == 0
        assert sig.parameters["cluster_rows"].default is False
        assert sig.parameters["metric"].default == "correlation"

    def test_returns_figure(self):
        """Should return a figure with sample data."""
        import pandas as pd

        activity = pd.DataFrame(
            {"TypeA": [0.1, 0.5, 0.8], "TypeB": [0.9, 0.2, 0.3]},
            index=["TF1", "TF2", "TF3"],
        )
        fig = ds.pl.heatmap_celltype_activity(activity, top_k=None, show=False, return_fig=True)
        assert fig is not None
