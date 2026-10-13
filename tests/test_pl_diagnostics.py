"""Tests for training diagnostic plots."""

import matplotlib
import pytest
from matplotlib.axes import Axes
from scipy.sparse import csr_matrix

matplotlib.use("Agg")

from deepscenic.pl import enhancer_activity_histogram
from tests.conftest import MINIMAL_DIMS


def _bar_heights(ax: Axes) -> list[float]:
    return [patch.get_height() for patch in ax.patches]


def test_enhancer_activity_histogram_counts_every_cell_region_pair(mock_deepscenic_model, mock_mdata_for_model):
    """With zeros kept, the histogram holds one value per cell and region."""
    ax = enhancer_activity_histogram(
        mock_deepscenic_model, mock_mdata_for_model, exclude_zeros=False, log_scale=False, show=False
    )

    assert isinstance(ax, Axes)
    assert sum(_bar_heights(ax)) == mock_mdata_for_model.n_obs * MINIMAL_DIMS["n_regions"]


@pytest.mark.parametrize("exclude_zeros", [True, False])
def test_enhancer_activity_histogram_sparse_matches_dense(mock_deepscenic_model, mock_mdata_for_model, exclude_zeros):
    """Sparse RNA input draws exactly the same histogram as dense input."""
    dense_ax = enhancer_activity_histogram(
        mock_deepscenic_model, mock_mdata_for_model, exclude_zeros=exclude_zeros, show=False
    )
    mock_mdata_for_model.mod["rna"].X = csr_matrix(mock_mdata_for_model.mod["rna"].X)
    sparse_ax = enhancer_activity_histogram(
        mock_deepscenic_model, mock_mdata_for_model, exclude_zeros=exclude_zeros, show=False
    )

    assert _bar_heights(sparse_ax) == _bar_heights(dense_ax)
