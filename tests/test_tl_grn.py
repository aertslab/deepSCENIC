"""Tests for GRN extraction functions (_grn.py)."""

import pandas as pd
import pytest

from deepscenic.tl._grn import (
    extract_e1_matrix,
    extract_e2_matrix,
    extract_grn,
    get_gene_regulators,
    get_tf_targets,
)


class TestExtractGRN:
    """Tests for extract_grn function."""

    def test_returns_dict(self, mock_deepscenic_model):
        """Should return a dictionary."""
        result = extract_grn(mock_deepscenic_model)
        assert isinstance(result, dict)

    def test_contains_e1_and_e2(self, mock_deepscenic_model):
        """Should contain E1 and E2 keys."""
        result = extract_grn(mock_deepscenic_model)
        assert "E1" in result
        assert "E2" in result

    def test_e1_is_dataframe(self, mock_deepscenic_model):
        """E1 should be a DataFrame."""
        result = extract_grn(mock_deepscenic_model)
        assert isinstance(result["E1"], pd.DataFrame)

    def test_e2_is_dataframe(self, mock_deepscenic_model):
        """E2 should be a DataFrame."""
        result = extract_grn(mock_deepscenic_model)
        assert isinstance(result["E2"], pd.DataFrame)

    def test_e1_shape(self, mock_deepscenic_model, minimal_dims):
        """E1 should have shape (n_regions, n_tfs)."""
        result = extract_grn(mock_deepscenic_model)
        d = minimal_dims
        assert result["E1"].shape == (d["n_regions"], d["n_tfs"])

    def test_e1_index_is_regions(self, mock_deepscenic_model):
        """E1 index should be region names."""
        result = extract_grn(mock_deepscenic_model)
        assert list(result["E1"].index) == mock_deepscenic_model.region_names

    def test_e1_columns_are_tfs(self, mock_deepscenic_model):
        """E1 columns should be TF names."""
        result = extract_grn(mock_deepscenic_model)
        assert list(result["E1"].columns) == mock_deepscenic_model.tf_names

    def test_e2_has_expected_columns(self, mock_deepscenic_model):
        """E2 should have region, gene, weight columns."""
        result = extract_grn(mock_deepscenic_model)
        assert "region" in result["E2"].columns
        assert "gene" in result["E2"].columns
        assert "weight" in result["E2"].columns


class TestExtractE1Matrix:
    """Tests for extract_e1_matrix function."""

    def test_returns_dataframe(self, mock_deepscenic_model):
        """Should return a DataFrame."""
        result = extract_e1_matrix(mock_deepscenic_model)
        assert isinstance(result, pd.DataFrame)

    def test_shape(self, mock_deepscenic_model, minimal_dims):
        """Should have correct shape."""
        result = extract_e1_matrix(mock_deepscenic_model)
        d = minimal_dims
        assert result.shape == (d["n_regions"], d["n_tfs"])


class TestExtractE2Matrix:
    """Tests for extract_e2_matrix function."""

    def test_sparse_format(self, mock_deepscenic_model):
        """Sparse format should return DataFrame with 3 columns."""
        result = extract_e2_matrix(mock_deepscenic_model, as_sparse=True)
        assert isinstance(result, pd.DataFrame)
        assert "region" in result.columns
        assert "gene" in result.columns
        assert "weight" in result.columns

    def test_dense_format(self, mock_deepscenic_model, minimal_dims):
        """Dense format should return (n_regions, n_genes) DataFrame."""
        result = extract_e2_matrix(mock_deepscenic_model, as_sparse=False)
        d = minimal_dims
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (d["n_regions"], d["n_genes"])


class TestGetTFTargets:
    """Tests for get_tf_targets function."""

    def test_returns_dataframe(self, mock_deepscenic_model):
        """Should return a DataFrame."""
        result = get_tf_targets(mock_deepscenic_model, "TF0")
        assert isinstance(result, pd.DataFrame)

    def test_invalid_tf_raises(self, mock_deepscenic_model):
        """Should raise error for invalid TF."""
        with pytest.raises(ValueError, match="not found"):
            get_tf_targets(mock_deepscenic_model, "INVALID_TF")

    def test_has_expected_columns(self, mock_deepscenic_model):
        """Should have expected columns."""
        result = get_tf_targets(mock_deepscenic_model, "TF0", threshold=0.0)
        if len(result) > 0:
            expected_cols = ["tf", "region", "E1_weight", "gene", "E2_weight", "combined_weight"]
            for col in expected_cols:
                assert col in result.columns

    def test_top_k(self, mock_deepscenic_model):
        """Should limit results with top_k."""
        result = get_tf_targets(mock_deepscenic_model, "TF0", threshold=0.0, top_k=5)
        assert len(result) <= 5

    def test_threshold_filter(self, mock_deepscenic_model):
        """Should filter by threshold."""
        high_thresh = get_tf_targets(mock_deepscenic_model, "TF0", threshold=1.0)
        low_thresh = get_tf_targets(mock_deepscenic_model, "TF0", threshold=0.0)
        assert len(high_thresh) <= len(low_thresh)


class TestGetGeneRegulators:
    """Tests for get_gene_regulators function."""

    def test_returns_dataframe(self, mock_deepscenic_model):
        """Should return a DataFrame."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0")
        assert isinstance(result, pd.DataFrame)

    def test_invalid_gene_raises(self, mock_deepscenic_model):
        """Should raise error for invalid gene."""
        with pytest.raises(ValueError, match="not found"):
            get_gene_regulators(mock_deepscenic_model, "INVALID_GENE")

    def test_has_expected_columns(self, mock_deepscenic_model):
        """Should have expected columns."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", threshold=0.0)
        if len(result) > 0:
            expected_cols = ["gene", "tf", "region", "E1_weight", "E2_weight", "combined_weight"]
            for col in expected_cols:
                assert col in result.columns

    def test_top_k(self, mock_deepscenic_model):
        """Should limit results with top_k."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", threshold=0.0, top_k=5)
        assert len(result) <= 5
