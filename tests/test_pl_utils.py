"""Tests for deepSCENIC plotting utilities."""

import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing


class TestSetupAxes:
    """Tests for setup_axes function."""

    def test_creates_new_figure_when_ax_is_none(self):
        """Should create new figure and axes when ax is None."""
        from deepscenic.pl import setup_axes

        fig, ax = setup_axes()

        assert fig is not None
        assert ax is not None
        assert ax.get_figure() is fig
        plt.close(fig)

    def test_creates_figure_with_custom_figsize(self):
        """Should create figure with specified figsize."""
        from deepscenic.pl import setup_axes

        fig, ax = setup_axes(figsize=(10, 5))

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 5
        plt.close(fig)

    def test_uses_existing_axes(self):
        """Should use existing axes when provided."""
        from deepscenic.pl import setup_axes

        existing_fig, existing_ax = plt.subplots()
        fig, ax = setup_axes(ax=existing_ax)

        assert fig is existing_fig
        assert ax is existing_ax
        plt.close(existing_fig)

    def test_returns_correct_types(self):
        """Should return Figure and Axes objects."""
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        from deepscenic.pl import setup_axes

        fig, ax = setup_axes()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)


class TestSavefigOrShow:
    """Tests for savefig_or_show function."""

    def test_saves_figure_when_save_is_true(self, tmp_path):
        """Should save figure when save=True."""
        import scanpy as sc

        from deepscenic.pl import savefig_or_show

        # Set scanpy figdir to temp path
        sc.settings.figdir = str(tmp_path)

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        savefig_or_show("test_plot", show=False, save=True)
        assert (tmp_path / "test_plot.pdf").exists()

    def test_saves_figure_with_custom_suffix(self, tmp_path):
        """Should save figure with custom suffix."""
        import scanpy as sc

        from deepscenic.pl import savefig_or_show

        sc.settings.figdir = str(tmp_path)

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        savefig_or_show("test_plot", show=False, save="_custom")
        assert (tmp_path / "test_plot_custom.pdf").exists()

    def test_closes_figure_when_show_is_false(self):
        """Should close figure when show=False."""
        from deepscenic.pl import savefig_or_show

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        savefig_or_show("test_plot", show=False, save=False)
        # Figure should be closed
        assert not plt.fignum_exists(fig.number)


class TestGetCmapColors:
    """Tests for get_cmap_colors function."""

    def test_returns_correct_number_of_colors(self):
        """Should return exactly n colors."""
        from deepscenic.pl._utils import get_cmap_colors

        colors = get_cmap_colors(5)
        assert len(colors) == 5

        colors = get_cmap_colors(10)
        assert len(colors) == 10

    def test_returns_hex_strings(self):
        """Should return valid hex color strings."""
        from deepscenic.pl._utils import get_cmap_colors

        colors = get_cmap_colors(5)

        for color in colors:
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format

    def test_uses_specified_cmap(self):
        """Should use the specified colormap."""
        from deepscenic.pl._utils import get_cmap_colors

        colors_tab20 = get_cmap_colors(5, cmap="tab20")
        colors_viridis = get_cmap_colors(5, cmap="viridis")

        # Different colormaps should produce different colors
        assert colors_tab20 != colors_viridis

    def test_handles_single_color(self):
        """Should handle n=1 without division by zero."""
        from deepscenic.pl._utils import get_cmap_colors

        colors = get_cmap_colors(1)
        assert len(colors) == 1


class TestSetColors:
    """Tests for set_colors function."""

    def test_stores_colors_in_mdata_uns(self, sample_mdata):
        """Should store colors in mdata.uns."""
        from deepscenic.pl import set_colors

        # Add a categorical column
        sample_mdata.obs["cell_type"] = pd.Categorical(
            ["A"] * 50 + ["B"] * 50, categories=["A", "B"]
        )

        set_colors(sample_mdata, "cell_type")

        assert "cell_type_colors" in sample_mdata.uns
        assert len(sample_mdata.uns["cell_type_colors"]) == 2

    def test_uses_provided_color_list(self, sample_mdata):
        """Should use provided color list."""
        from deepscenic.pl import set_colors

        sample_mdata.obs["cell_type"] = pd.Categorical(
            ["A"] * 50 + ["B"] * 50, categories=["A", "B"]
        )

        set_colors(sample_mdata, "cell_type", colors=["#FF0000", "#00FF00"])

        assert sample_mdata.uns["cell_type_colors"][0] == "#FF0000"
        assert sample_mdata.uns["cell_type_colors"][1] == "#00FF00"

    def test_uses_provided_color_dict(self, sample_mdata):
        """Should use provided color dictionary."""
        from deepscenic.pl import set_colors

        sample_mdata.obs["cell_type"] = pd.Categorical(
            ["A"] * 50 + ["B"] * 50, categories=["A", "B"]
        )

        set_colors(sample_mdata, "cell_type", colors={"A": "#FF0000", "B": "#00FF00"})

        colors = sample_mdata.uns["cell_type_colors"]
        assert colors[0] == "#FF0000"  # A
        assert colors[1] == "#00FF00"  # B

    def test_raises_on_missing_key(self, sample_mdata):
        """Should raise KeyError for non-existent key."""
        from deepscenic.pl import set_colors

        with pytest.raises(KeyError, match="not found"):
            set_colors(sample_mdata, "nonexistent_key")

    def test_raises_on_wrong_color_count(self, sample_mdata):
        """Should raise ValueError for wrong number of colors."""
        from deepscenic.pl import set_colors

        sample_mdata.obs["cell_type"] = pd.Categorical(
            ["A"] * 50 + ["B"] * 50, categories=["A", "B"]
        )

        with pytest.raises(ValueError, match="Need 2 colors"):
            set_colors(sample_mdata, "cell_type", colors=["#FF0000"])


class TestGetColors:
    """Tests for get_colors function."""

    def test_retrieves_stored_colors(self, sample_mdata):
        """Should retrieve colors from mdata.uns."""
        from deepscenic.pl import get_colors, set_colors

        sample_mdata.obs["cell_type"] = pd.Categorical(
            ["A"] * 50 + ["B"] * 50, categories=["A", "B"]
        )
        set_colors(sample_mdata, "cell_type", colors=["#FF0000", "#00FF00"])

        colors = get_colors(sample_mdata, "cell_type")

        assert colors["A"] == "#FF0000"
        assert colors["B"] == "#00FF00"

    def test_generates_colors_if_not_stored(self, sample_mdata):
        """Should generate colors if not already stored."""
        from deepscenic.pl import get_colors

        sample_mdata.obs["cell_type"] = pd.Categorical(
            ["A"] * 50 + ["B"] * 50, categories=["A", "B"]
        )

        colors = get_colors(sample_mdata, "cell_type")

        assert "A" in colors
        assert "B" in colors
        # Should also store them
        assert "cell_type_colors" in sample_mdata.uns

    def test_returns_dict(self, sample_mdata):
        """Should return dict mapping categories to colors."""
        from deepscenic.pl import get_colors

        sample_mdata.obs["cell_type"] = pd.Categorical(
            ["A"] * 50 + ["B"] * 50, categories=["A", "B"]
        )

        colors = get_colors(sample_mdata, "cell_type")

        assert isinstance(colors, dict)
        assert len(colors) == 2

    def test_raises_on_missing_key(self, sample_mdata):
        """Should raise KeyError for non-existent key."""
        from deepscenic.pl import get_colors

        with pytest.raises(KeyError, match="not found"):
            get_colors(sample_mdata, "nonexistent_key")
