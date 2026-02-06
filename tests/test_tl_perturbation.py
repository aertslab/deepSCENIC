"""Tests for perturbation functions (_perturbation.py)."""

import numpy as np
import pandas as pd
import pytest

from deepscenic.tl._perturbation import (
    process_perturbation_results,
    simulate_multi_perturbation,
    simulate_perturbation,
)


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

    def test_return_intermediate(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should return dict of iterations when return_intermediate=True."""
        n_iter = 3
        result = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            level=0.0,
            n_iter=n_iter,
            batch_size=4,
            device="cpu",
            return_intermediate=True,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {1, 2, 3}
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        for i in range(1, n_iter + 1):
            assert result[i].shape == (n_cells, d["n_genes"])


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

    def test_return_intermediate(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should return dict of iterations when return_intermediate=True."""
        n_iter = 3
        result = simulate_multi_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_names=["TF0", "TF1"],
            n_iter=n_iter,
            batch_size=4,
            device="cpu",
            return_intermediate=True,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {1, 2, 3}
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        for i in range(1, n_iter + 1):
            assert result[i].shape == (n_cells, d["n_genes"])


class TestProcessPerturbationResults:
    """Tests for process_perturbation_results function."""

    def test_returns_dataframe(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should return pandas DataFrame."""
        logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0")

        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should have expected columns."""
        logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0")

        assert "gene" in result.columns
        assert "tf" in result.columns
        assert "log2fc" in result.columns
        assert "abs_log2fc" in result.columns
        assert "pvalue" in result.columns

    def test_output_rows(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should have one row per gene."""
        logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0")

        assert len(result) == minimal_dims["n_genes"]

    def test_no_pvalues(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should work without computing pvalues."""
        logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0", compute_pvalues=False)

        assert "pvalue" not in result.columns

    def test_gene_subset(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should filter to gene subset."""
        logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )

        gene_names = list(mock_mdata_for_model.mod["rna"].var_names[:3])
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0", gene_subset=gene_names)

        assert len(result) == 3
        assert set(result["gene"]) == set(gene_names)

    def test_multi_tf_label(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should combine multiple TF names with '+'."""
        logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, ["TF0", "TF1"])

        assert result["tf"].iloc[0] == "TF0+TF1"

    def test_mismatched_shape_raises(self, mock_mdata_for_model):
        """Should raise error if logFC shape doesn't match mdata."""
        logFC = np.zeros((10, 5))  # Wrong number of genes

        with pytest.raises(ValueError, match="genes"):
            process_perturbation_results(logFC, mock_mdata_for_model, "TF0")
