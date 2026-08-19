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
        perturbed, logFC = simulate_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_name="TF0", level=0.0, n_iter=2, batch_size=4, device="cpu"
        )
        assert isinstance(perturbed, np.ndarray)
        assert isinstance(logFC, np.ndarray)

    def test_output_shape(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Output should have shape (n_cells, n_genes)."""
        perturbed, logFC = simulate_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_name="TF0", level=0.0, n_iter=2, batch_size=4, device="cpu"
        )
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        assert perturbed.shape == (n_cells, d["n_genes"])
        assert logFC.shape == (n_cells, d["n_genes"])

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
        perturbed_trace, logFC_trace = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            level=0.0,
            n_iter=n_iter,
            batch_size=4,
            device="cpu",
            return_intermediate=True,
        )
        assert isinstance(perturbed_trace, dict)
        assert isinstance(logFC_trace, dict)
        assert set(perturbed_trace.keys()) == {1, 2, 3}
        assert set(logFC_trace.keys()) == {1, 2, 3}
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        for i in range(1, n_iter + 1):
            assert perturbed_trace[i].shape == (n_cells, d["n_genes"])
            assert logFC_trace[i].shape == (n_cells, d["n_genes"])

    def test_multiple_tfs(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should accept multiple TFs and one level per TF."""
        perturbed, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name=["TF0", "TF1"],
            level=[0.0, 2.0],
            n_iter=2,
            batch_size=4,
            device="cpu",
        )

        tf0_idx = int(mock_deepscenic_model.vae.tf_indices[0])
        tf1_idx = int(mock_deepscenic_model.vae.tf_indices[1])
        assert np.all(perturbed[:, tf0_idx] == 0.0)
        assert np.all(perturbed[:, tf1_idx] == 2.0)
        assert perturbed.shape == logFC.shape

    def test_mismatched_levels_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """Test that function raises error when levels and TF names have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            simulate_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                tf_name=["TF0", "TF1"],
                level=[0.0],
                device="cpu",
            )


class TestSimulateMultiPerturbation:
    """Tests for simulate_multi_perturbation function."""

    def test_returns_numpy_array(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should return numpy array."""
        perturbed, logFC = simulate_multi_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_names=["TF0", "TF1"], n_iter=2, batch_size=4, device="cpu"
        )
        assert isinstance(perturbed, np.ndarray)
        assert isinstance(logFC, np.ndarray)

    def test_output_shape(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Output should have shape (n_cells, n_genes)."""
        perturbed, logFC = simulate_multi_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_names=["TF0", "TF1"], n_iter=2, batch_size=4, device="cpu"
        )
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        assert perturbed.shape == (n_cells, d["n_genes"])
        assert logFC.shape == (n_cells, d["n_genes"])

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
        assert all(value is not None for value in result)

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
        perturbed_trace, logFC_trace = simulate_multi_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_names=["TF0", "TF1"],
            n_iter=n_iter,
            batch_size=4,
            device="cpu",
            return_intermediate=True,
        )
        assert isinstance(perturbed_trace, dict)
        assert isinstance(logFC_trace, dict)
        assert set(perturbed_trace.keys()) == {1, 2, 3}
        assert set(logFC_trace.keys()) == {1, 2, 3}
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        for i in range(1, n_iter + 1):
            assert perturbed_trace[i].shape == (n_cells, d["n_genes"])
            assert logFC_trace[i].shape == (n_cells, d["n_genes"])


class TestProcessPerturbationResults:
    """Tests for process_perturbation_results function."""

    def test_returns_dataframe(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should return pandas DataFrame."""
        _, logFC = simulate_perturbation(
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
        _, logFC = simulate_perturbation(
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

    def test_output_rows(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should have one row per gene."""
        _, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0")

        assert len(result) == minimal_dims["n_genes"]

    def test_no_pvalue_columns(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should not include pvalue/padj columns (removed from API)."""
        _, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0")

        assert "pvalue" not in result.columns
        assert "padj" not in result.columns

    def test_gene_subset(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should filter to gene subset."""
        _, logFC = simulate_perturbation(
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
        _, logFC = simulate_perturbation(
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
