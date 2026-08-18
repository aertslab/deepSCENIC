"""Tests for perturbation PCA plotting."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.sparse import csr_matrix

matplotlib.use("Agg")

from deepscenic.pl import perturbation_pca


@pytest.fixture
def paired_matrices():
    """Create paired original and perturbed data matrices for testing."""
    rng = np.random.default_rng(42)
    original = rng.normal(size=(20, 6))
    perturbed = original.copy()
    perturbed[:, :2] += 1.0
    return original, perturbed


def test_returns_axes_and_draws_paired_data(paired_matrices):
    """Test that function returns Axes object with correct plot elements."""
    original, perturbed = paired_matrices

    ax = perturbation_pca(original, perturbed, show=False)

    assert isinstance(ax, Axes)
    assert len(ax.collections) == 3  # Two scatters and one quiver collection
    assert "PC1" in ax.get_xlabel()
    assert "PC2" in ax.get_ylabel()
    assert ax.collections[0].get_alpha() == 0.35
    assert ax.collections[1].get_alpha() == 1.0


def test_accepts_separate_point_opacities(paired_matrices):
    """Test that function accepts and applies separate opacity values."""
    original, perturbed = paired_matrices

    ax = perturbation_pca(
        original,
        perturbed,
        original_alpha=0.2,
        perturbed_alpha=0.9,
        show=False,
    )

    assert ax.collections[0].get_alpha() == 0.2
    assert ax.collections[1].get_alpha() == 0.9


def test_returns_figure(paired_matrices):
    """Test that function can return Figure object when requested."""
    original, perturbed = paired_matrices

    fig = perturbation_pca(original, perturbed, show=False, return_fig=True)

    assert isinstance(fig, Figure)


def test_accepts_sparse_matrices(paired_matrices):
    """Test that function handles sparse matrix inputs."""
    original, perturbed = paired_matrices

    ax = perturbation_pca(csr_matrix(original), csr_matrix(perturbed), show=False)

    assert isinstance(ax, Axes)


def test_can_disable_arrows(paired_matrices):
    """Test that function can disable arrow rendering."""
    original, perturbed = paired_matrices

    ax = perturbation_pca(original, perturbed, max_arrows=0, show=False)

    assert len(ax.collections) == 2


def test_colors_by_categorical_annotation(paired_matrices):
    """Test that function correctly colors points by categorical annotation."""
    original, perturbed = paired_matrices
    obs = pd.DataFrame({"cell_state": pd.Categorical(["A"] * 10 + ["B"] * 10)})

    ax = perturbation_pca(
        original,
        perturbed,
        obs=obs,
        color="cell_state",
        palette={"A": "red", "B": "blue"},
        show=False,
    )

    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_labels == ["Original", "Perturbed", "A", "B"]


def test_colors_by_numeric_annotation(paired_matrices):
    """Test that function correctly colors points by numeric annotation."""
    original, perturbed = paired_matrices
    obs = pd.DataFrame({"score": np.linspace(0, 1, len(original))})

    fig = perturbation_pca(
        original,
        perturbed,
        obs=obs,
        color="score",
        show=False,
        return_fig=True,
    )

    assert len(fig.axes) == 2  # PCA axes and annotation colorbar


def test_annotation_length_must_match(paired_matrices):
    """Test that function raises error when annotation length doesn't match data."""
    original, perturbed = paired_matrices
    obs = pd.DataFrame({"cell_state": ["A"] * (len(original) - 1)})

    with pytest.raises(ValueError, match="obs has"):
        perturbation_pca(original, perturbed, obs=obs, color="cell_state", show=False)


def test_rejects_shape_mismatch(paired_matrices):
    """Test that function rejects matrices with mismatched shapes."""
    original, perturbed = paired_matrices

    with pytest.raises(ValueError, match="Shape mismatch"):
        perturbation_pca(original, perturbed[:, :-1], show=False)


@pytest.mark.parametrize("n_components", [0, 1, 7])
def test_rejects_invalid_component_count(paired_matrices, n_components):
    """Test that function rejects invalid n_components values."""
    original, perturbed = paired_matrices

    with pytest.raises(ValueError, match="n_components"):
        perturbation_pca(
            original,
            perturbed,
            n_components=n_components,
            show=False,
        )
