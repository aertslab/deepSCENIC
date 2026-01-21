"""Tests for deepSCENIC GRN plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing


class TestHeatmapE1:
    """Tests for heatmap_e1 function."""

    def test_creates_heatmap(self, mock_deepscenic_model):
        """Should create E1 heatmap from model."""
        from deepscenic.pl import heatmap_e1

        ax = heatmap_e1(mock_deepscenic_model, show=False, top_k=5)

        assert ax is not None
        plt.close("all")

    def test_filters_by_tfs(self, mock_deepscenic_model):
        """Should filter by specific TFs."""
        from deepscenic.pl import heatmap_e1

        tfs = mock_deepscenic_model.tf_names[:2]
        ax = heatmap_e1(mock_deepscenic_model, tfs=tfs, show=False)

        assert ax is not None
        plt.close("all")

    def test_returns_figure_when_requested(self, mock_deepscenic_model):
        """Should return Figure when return_fig=True."""
        from matplotlib.figure import Figure

        from deepscenic.pl import heatmap_e1

        fig = heatmap_e1(mock_deepscenic_model, show=False, return_fig=True, top_k=5)

        assert isinstance(fig, Figure)
        plt.close("all")


class TestHeatmapE2:
    """Tests for heatmap_e2 function."""

    def test_creates_heatmap(self, mock_deepscenic_model):
        """Should create E2 heatmap from model."""
        from deepscenic.pl import heatmap_e2

        ax = heatmap_e2(mock_deepscenic_model, show=False, top_k=5)

        assert ax is not None
        plt.close("all")

    def test_filters_by_genes(self, mock_deepscenic_model):
        """Should filter by specific genes."""
        from deepscenic.pl import heatmap_e2

        genes = mock_deepscenic_model.gene_names[:3]
        ax = heatmap_e2(mock_deepscenic_model, genes=genes, show=False)

        assert ax is not None
        plt.close("all")


class TestHeatmapGRN:
    """Tests for heatmap_grn function."""

    def test_creates_combined_heatmap(self, mock_deepscenic_model):
        """Should create combined TF->gene heatmap."""
        from deepscenic.pl import heatmap_grn

        ax = heatmap_grn(mock_deepscenic_model, show=False, top_k=5)

        assert ax is not None
        plt.close("all")

    def test_filters_by_tfs_and_genes(self, mock_deepscenic_model):
        """Should filter by TFs and genes."""
        from deepscenic.pl import heatmap_grn

        tfs = mock_deepscenic_model.tf_names[:2]
        genes = mock_deepscenic_model.gene_names[:5]
        ax = heatmap_grn(mock_deepscenic_model, tfs=tfs, genes=genes, show=False)

        assert ax is not None
        plt.close("all")


class TestNetworkGRN:
    """Tests for network_grn function."""

    def test_creates_network(self, mock_deepscenic_model):
        """Should create GRN network plot."""
        from deepscenic.pl import network_grn

        # Use very low threshold to ensure some edges are found
        ax = network_grn(
            mock_deepscenic_model,
            tfs=mock_deepscenic_model.tf_names[:2],
            threshold=0.0,
            top_k=5,
            show=False,
        )

        assert ax is not None
        plt.close("all")

    def test_different_layouts(self, mock_deepscenic_model):
        """Should support different layouts."""
        from deepscenic.pl import network_grn

        for layout in ["spring", "circular", "kamada_kawai"]:
            ax = network_grn(
                mock_deepscenic_model,
                tfs=mock_deepscenic_model.tf_names[:2],
                threshold=0.0,
                top_k=5,
                layout=layout,
                show=False,
            )
            assert ax is not None
            plt.close("all")


class TestNetworkTFTargets:
    """Tests for network_tf_targets function."""

    def test_creates_tf_targets_network(self, mock_deepscenic_model):
        """Should create TF targets network."""
        from deepscenic.pl import network_tf_targets

        tf = mock_deepscenic_model.tf_names[0]
        ax = network_tf_targets(mock_deepscenic_model, tf, threshold=0.0, top_k=5, show=False)

        assert ax is not None
        plt.close("all")

    def test_shows_regions_when_requested(self, mock_deepscenic_model):
        """Should show region nodes when show_regions=True."""
        from deepscenic.pl import network_tf_targets

        tf = mock_deepscenic_model.tf_names[0]
        ax = network_tf_targets(mock_deepscenic_model, tf, threshold=0.0, top_k=5, show_regions=True, show=False)

        assert ax is not None
        plt.close("all")

    def test_raises_on_invalid_tf(self, mock_deepscenic_model):
        """Should raise error for non-existent TF."""
        from deepscenic.pl import network_tf_targets

        with pytest.raises(ValueError, match="not found"):
            network_tf_targets(mock_deepscenic_model, "NONEXISTENT_TF", show=False)


class TestNetworkGeneRegulators:
    """Tests for network_gene_regulators function."""

    def test_creates_gene_regulators_network(self, mock_deepscenic_model):
        """Should create gene regulators network."""
        from deepscenic.pl import network_gene_regulators

        gene = mock_deepscenic_model.gene_names[0]
        ax = network_gene_regulators(mock_deepscenic_model, gene, threshold=0.0, top_k=5, show=False)

        assert ax is not None
        plt.close("all")

    def test_raises_on_invalid_gene(self, mock_deepscenic_model):
        """Should raise error for non-existent gene."""
        from deepscenic.pl import network_gene_regulators

        with pytest.raises(ValueError, match="not found"):
            network_gene_regulators(mock_deepscenic_model, "NONEXISTENT_GENE", show=False)
