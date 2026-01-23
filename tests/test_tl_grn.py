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

    def test_e1_index_is_regions(self, mock_deepscenic_model):
        """E1 index should be region names."""
        result = extract_grn(mock_deepscenic_model)
        assert list(result["E1"].index) == mock_deepscenic_model.region_names

    def test_e1_columns_are_tfs(self, mock_deepscenic_model):
        """E1 columns should be TF names."""
        result = extract_grn(mock_deepscenic_model)
        assert list(result["E1"].columns) == mock_deepscenic_model.tf_names


class TestExtractE1Matrix:
    """Tests for extract_e1_matrix function."""

    def test_shape(self, mock_deepscenic_model, minimal_dims):
        """Should have correct shape."""
        result = extract_e1_matrix(mock_deepscenic_model)
        d = minimal_dims
        assert result.shape == (d["n_regions"], d["n_tfs"])


class TestExtractE2Matrix:
    """Tests for extract_e2_matrix function."""

    def test_dense_format(self, mock_deepscenic_model, minimal_dims):
        """Dense format should return (n_regions, n_genes) DataFrame."""
        result = extract_e2_matrix(mock_deepscenic_model, as_edgelist=False)
        d = minimal_dims
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (d["n_regions"], d["n_genes"])


class TestGetTFTargets:
    """Tests for get_tf_targets function."""

    def test_invalid_tf_raises(self, mock_deepscenic_model):
        """Should raise error for invalid TF."""
        with pytest.raises(ValueError, match="not found"):
            get_tf_targets(mock_deepscenic_model, "INVALID_TF")

    def test_threshold_filter(self, mock_deepscenic_model):
        """Should filter by E1 threshold."""
        high_thresh = get_tf_targets(mock_deepscenic_model, "TF0", e1_threshold=1.0)
        low_thresh = get_tf_targets(mock_deepscenic_model, "TF0", e1_threshold=0.0)
        assert len(high_thresh) <= len(low_thresh)


class TestGetGeneRegulators:
    """Tests for get_gene_regulators function."""

    def test_invalid_gene_raises(self, mock_deepscenic_model):
        """Should raise error for invalid gene."""
        with pytest.raises(ValueError, match="not found"):
            get_gene_regulators(mock_deepscenic_model, "INVALID_GENE")

    def test_top_k(self, mock_deepscenic_model):
        """Should limit results with top_k."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", e1_threshold=0.0, top_k=5)
        assert len(result) <= 5
