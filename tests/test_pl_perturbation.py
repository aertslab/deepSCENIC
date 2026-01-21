"""Tests for deepSCENIC perturbation plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing


@pytest.fixture
def sample_perturbation_results():
    """Create sample perturbation results DataFrame."""
    np.random.seed(42)
    n_genes = 100

    return pd.DataFrame(
        {
            "gene": [f"Gene_{i}" for i in range(n_genes)],
            "log2fc": np.random.randn(n_genes),
            "pvalue": np.random.uniform(0, 1, n_genes),
        }
    )


@pytest.fixture
def sample_multi_tf_results():
    """Create sample results for multiple TFs."""
    np.random.seed(42)
    tfs = ["TF1", "TF2", "TF3"]
    genes = [f"Gene_{i}" for i in range(20)]

    data = []
    for tf in tfs:
        for gene in genes:
            data.append(
                {
                    "tf": tf,
                    "gene": gene,
                    "log2fc": np.random.randn(),
                    "pvalue": np.random.uniform(0, 1),
                }
            )

    return pd.DataFrame(data)


class TestVolcanoPerturbation:
    """Tests for volcano_perturbation function."""

    def test_creates_volcano_plot(self, sample_perturbation_results):
        """Should create volcano plot from results."""
        from deepscenic.pl import volcano_perturbation

        ax = volcano_perturbation(sample_perturbation_results, show=False)

        assert ax is not None
        plt.close("all")


class TestHeatmapPerturbation:
    """Tests for heatmap_perturbation function."""

    def test_creates_heatmap(self, sample_multi_tf_results):
        """Should create perturbation heatmap."""
        from deepscenic.pl import heatmap_perturbation

        # This returns None when return_fig=False
        result = heatmap_perturbation(sample_multi_tf_results, show=False, cluster_rows=False, cluster_cols=False)

        assert result is None
        plt.close("all")

    def test_filters_by_tfs(self, sample_multi_tf_results):
        """Should filter by TFs."""
        from deepscenic.pl import heatmap_perturbation

        result = heatmap_perturbation(
            sample_multi_tf_results, tfs=["TF1", "TF2"], show=False, cluster_rows=False, cluster_cols=False
        )

        assert result is None  # Returns None when return_fig=False
        plt.close("all")

    def test_filters_by_genes(self, sample_multi_tf_results):
        """Should filter by genes."""
        from deepscenic.pl import heatmap_perturbation

        result = heatmap_perturbation(
            sample_multi_tf_results,
            genes=["Gene_0", "Gene_1", "Gene_2"],
            show=False,
            cluster_rows=False,
            cluster_cols=False,
        )

        assert result is None
        plt.close("all")


class TestDotplotPerturbation:
    """Tests for dotplot_perturbation function."""

    def test_creates_dotplot(self, sample_multi_tf_results):
        """Should create perturbation dotplot."""
        from deepscenic.pl import dotplot_perturbation

        ax = dotplot_perturbation(sample_multi_tf_results, show=False)

        assert ax is not None
        plt.close("all")

    def test_filters_by_tfs_and_genes(self, sample_multi_tf_results):
        """Should filter by TFs and genes."""
        from deepscenic.pl import dotplot_perturbation

        ax = dotplot_perturbation(
            sample_multi_tf_results, tfs=["TF1"], genes=["Gene_0", "Gene_1", "Gene_2"], show=False
        )

        assert ax is not None
        plt.close("all")
