"""Tests for deepSCENIC genomics plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing


@pytest.fixture
def sample_links():
    """Create sample region-gene links."""
    return pd.DataFrame(
        {
            "region": [
                "chr1:100000-100640",
                "chr1:150000-150640",
                "chr1:200000-200640",
                "chr1:250000-250640",
            ],
            "gene": ["GeneA", "GeneB", "GeneA", "GeneC"],
            "weight": [0.8, 0.5, 0.3, 0.9],
        }
    )


@pytest.fixture
def sample_genes():
    """Create sample gene annotations."""
    return pd.DataFrame(
        {
            "gene": ["GeneA", "GeneB", "GeneC"],
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [120000, 180000, 280000],
            "end": [125000, 185000, 285000],
            "strand": ["+", "-", "+"],
        }
    )


class TestArcPlot:
    """Tests for arc_plot function."""

    def test_creates_arc_plot(self, sample_links):
        """Should create arc plot from links."""
        from deepscenic.pl import arc_plot

        ax = arc_plot(sample_links, chrom="chr1", show=False)

        assert ax is not None
        plt.close("all")

    def test_filters_by_region(self, sample_links):
        """Should filter links by genomic region."""
        from deepscenic.pl import arc_plot

        ax = arc_plot(sample_links, chrom="chr1", start=100000, end=200000, show=False)

        assert ax is not None
        plt.close("all")

    def test_returns_figure_when_requested(self, sample_links):
        """Should return Figure when return_fig=True."""
        from matplotlib.figure import Figure

        from deepscenic.pl import arc_plot

        fig = arc_plot(sample_links, chrom="chr1", show=False, return_fig=True)

        assert isinstance(fig, Figure)
        plt.close("all")

    def test_uses_gene_positions(self, sample_links):
        """Should use provided gene positions."""
        from deepscenic.pl import arc_plot

        gene_positions = {"GeneA": 120000, "GeneB": 180000, "GeneC": 280000}
        ax = arc_plot(sample_links, chrom="chr1", gene_positions=gene_positions, show=False)

        assert ax is not None
        plt.close("all")

    def test_raises_on_empty_region(self, sample_links):
        """Should raise error when no links in region."""
        from deepscenic.pl import arc_plot

        with pytest.raises(ValueError, match="No links"):
            arc_plot(sample_links, chrom="chr2", show=False)


class TestGenomeBrowser:
    """Tests for genome_browser function."""

    def test_creates_browser_with_regions(self):
        """Should create browser with region accessibility."""
        from deepscenic.pl import genome_browser

        # Create sample data
        atac_data = np.random.rand(50, 10)
        region_names = [f"chr1:{i*10000}-{i*10000+640}" for i in range(10)]

        fig = genome_browser(
            chrom="chr1",
            start=0,
            end=100000,
            atac_data=atac_data,
            region_names=region_names,
            show=False,
            return_fig=True,
        )

        assert fig is not None
        plt.close("all")

    def test_creates_browser_with_genes(self, sample_genes):
        """Should create browser with gene annotations."""
        from deepscenic.pl import genome_browser

        atac_data = np.random.rand(50, 10)
        region_names = [f"chr1:{i*30000}-{i*30000+640}" for i in range(10)]

        fig = genome_browser(
            chrom="chr1",
            start=100000,
            end=300000,
            atac_data=atac_data,
            region_names=region_names,
            genes=sample_genes,
            show=False,
            return_fig=True,
        )

        assert fig is not None
        plt.close("all")

    def test_creates_browser_with_links(self, sample_links, sample_genes):
        """Should create browser with region-gene links."""
        from deepscenic.pl import genome_browser

        atac_data = np.random.rand(50, 10)
        region_names = [f"chr1:{i*30000}-{i*30000+640}" for i in range(10)]

        fig = genome_browser(
            chrom="chr1",
            start=100000,
            end=300000,
            atac_data=atac_data,
            region_names=region_names,
            genes=sample_genes,
            links=sample_links,
            show=False,
            return_fig=True,
        )

        assert fig is not None
        plt.close("all")

    def test_creates_browser_with_cell_groups(self):
        """Should create browser with cell group tracks."""
        from deepscenic.pl import genome_browser

        atac_data = np.random.rand(50, 10)
        region_names = [f"chr1:{i*10000}-{i*10000+640}" for i in range(10)]
        cell_groups = {"Group1": np.arange(25), "Group2": np.arange(25, 50)}
        group_colors = {"Group1": "#E64B35", "Group2": "#4DBBD5"}

        fig = genome_browser(
            chrom="chr1",
            start=0,
            end=100000,
            atac_data=atac_data,
            region_names=region_names,
            cell_groups=cell_groups,
            group_colors=group_colors,
            show=False,
            return_fig=True,
        )

        assert fig is not None
        plt.close("all")
