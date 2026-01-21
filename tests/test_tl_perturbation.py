"""Tests for perturbation functions (_perturbation.py)."""

import numpy as np
import pytest

from deepscenic.tl._perturbation import simulate_multi_perturbation, simulate_perturbation


class TestSimulatePerturbation:
    """Tests for simulate_perturbation function."""

    def test_returns_numpy_array(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should return numpy array."""
        result = simulate_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_name="TF0", level=0.0, n_iter=2, batch_size=4, device="cpu"
        )
        assert isinstance(result, np.ndarray)

    def test_output_shape(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Output should have shape (n_cells, n_genes)."""
        result = simulate_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_name="TF0", level=0.0, n_iter=2, batch_size=4, device="cpu"
        )
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        assert result.shape == (n_cells, d["n_genes"])

    def test_invalid_tf_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should raise error for invalid TF name."""
        with pytest.raises(ValueError, match="not found"):
            simulate_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                tf_name="INVALID_TF",
                level=0.0,
                n_iter=2,
                batch_size=4,
                device="cpu",
            )

    def test_knockout_level_zero(self, mock_deepscenic_model, mock_mdata_for_model):
        """Knockout (level=0) should work."""
        result = simulate_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_name="TF0", level=0.0, n_iter=2, batch_size=4, device="cpu"
        )
        assert result is not None

    def test_overexpression_level_positive(self, mock_deepscenic_model, mock_mdata_for_model):
        """Overexpression (level>0) should work."""
        result = simulate_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_name="TF0", level=2.0, n_iter=2, batch_size=4, device="cpu"
        )
        assert result is not None

    def test_different_n_iter(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should work with different iteration counts."""
        for n_iter in [1, 5, 10]:
            result = simulate_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                tf_name="TF0",
                level=0.0,
                n_iter=n_iter,
                batch_size=4,
                device="cpu",
            )
            d = minimal_dims
            assert result.shape == (d["n_cells"] * 2, d["n_genes"])


class TestSimulateMultiPerturbation:
    """Tests for simulate_multi_perturbation function."""

    def test_returns_numpy_array(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should return numpy array."""
        result = simulate_multi_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_names=["TF0", "TF1"], n_iter=2, batch_size=4, device="cpu"
        )
        assert isinstance(result, np.ndarray)

    def test_output_shape(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Output should have shape (n_cells, n_genes)."""
        result = simulate_multi_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_names=["TF0", "TF1"], n_iter=2, batch_size=4, device="cpu"
        )
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        assert result.shape == (n_cells, d["n_genes"])

    def test_invalid_tf_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should raise error for invalid TF name."""
        with pytest.raises(ValueError, match="not found"):
            simulate_multi_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                tf_names=["TF0", "INVALID"],
                n_iter=2,
                batch_size=4,
                device="cpu",
            )

    def test_levels_default_to_zero(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should default to knockout (level=0) for all TFs."""
        result = simulate_multi_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_names=["TF0", "TF1"], n_iter=2, batch_size=4, device="cpu"
        )
        assert result is not None

    def test_custom_levels(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should accept custom levels per TF."""
        result = simulate_multi_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_names=["TF0", "TF1"],
            levels=[0.0, 2.0],
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        assert result is not None

    def test_mismatched_levels_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should raise error if levels length doesn't match tf_names."""
        with pytest.raises(ValueError, match="same length"):
            simulate_multi_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                tf_names=["TF0", "TF1"],
                levels=[0.0],
                n_iter=2,
                batch_size=4,
                device="cpu",
            )
