"""Tests for GRN extraction functions (_grn.py)."""

import numpy as np
import pandas as pd
import pytest

from deepscenic.tl._grn import (
    _get_otsu_threshold,
    build_grn_for_tfs,
    compute_celltype_enhancer_activity,
    compute_tf_activity_scores,
    extract_e1_matrix,
    extract_e2_matrix,
    extract_grn,
    get_gene_regulators,
    get_tf_targets,
    identify_active_enhancers,
    identify_key_tfs,
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

    def test_edgelist_format_default(self, mock_deepscenic_model, minimal_dims):
        """Edge list format (default) should return correct structure."""
        result = extract_e2_matrix(mock_deepscenic_model)  # as_edgelist=True by default

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["region", "gene", "weight"]
        assert len(result) == minimal_dims["n_links"]

    def test_edgelist_region_names_valid(self, mock_deepscenic_model):
        """Edge list regions should be valid region names."""
        result = extract_e2_matrix(mock_deepscenic_model)
        assert all(r in mock_deepscenic_model.region_names for r in result["region"])

    def test_edgelist_gene_names_valid(self, mock_deepscenic_model):
        """Edge list genes should be valid gene names."""
        result = extract_e2_matrix(mock_deepscenic_model)
        assert all(g in mock_deepscenic_model.gene_names for g in result["gene"])

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

    def test_result_columns(self, mock_deepscenic_model):
        """Result should have correct columns."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", e1_threshold=0.0)
        expected_columns = ["gene", "tf", "region", "E1_weight", "E2_weight", "combined_weight"]
        assert list(result.columns) == expected_columns

    def test_gene_column_matches_input(self, mock_deepscenic_model):
        """Gene column should match input gene name."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", e1_threshold=0.0)
        if len(result) > 0:
            assert all(result["gene"] == "GENE0")

    def test_sorted_by_absolute_weight(self, mock_deepscenic_model):
        """Results should be sorted by absolute combined_weight descending."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", e1_threshold=0.0)
        if len(result) > 1:
            abs_weights = np.abs(result["combined_weight"].values)
            assert all(abs_weights[i] >= abs_weights[i + 1] for i in range(len(abs_weights) - 1))

    def test_e2_threshold_filter(self, mock_deepscenic_model):
        """E2 threshold should filter results."""
        no_thresh = get_gene_regulators(mock_deepscenic_model, "GENE0", e1_threshold=0.0, e2_threshold=0.0)
        high_thresh = get_gene_regulators(mock_deepscenic_model, "GENE0", e1_threshold=0.0, e2_threshold=100.0)
        assert len(high_thresh) <= len(no_thresh)


class TestOtsuThreshold:
    """Tests for _get_otsu_threshold helper."""

    def test_otsu_success(self):
        """Otsu should compute threshold for bimodal distribution."""
        # Create bimodal distribution
        rng = np.random.default_rng(42)
        values = np.concatenate([rng.normal(0.2, 0.05, 100), rng.normal(0.8, 0.05, 100)])
        thresh, used_otsu = _get_otsu_threshold(values, fallback=0.5)

        assert used_otsu is True
        assert 0.3 < thresh < 0.7  # Should be between modes

    def test_otsu_fallback_single_value(self):
        """Otsu should fallback when single positive value."""
        values = np.array([0.5])
        thresh, used_otsu = _get_otsu_threshold(values, fallback=0.1)

        assert used_otsu is False
        assert thresh == 0.1

    def test_otsu_all_identical_returns_value(self):
        """Otsu returns threshold for identical values (skimage behavior)."""
        values = np.array([0.5, 0.5, 0.5, 0.5])
        thresh, used_otsu = _get_otsu_threshold(values, fallback=0.2)

        # skimage's threshold_otsu doesn't raise for identical values
        # It returns the value itself as the threshold
        assert used_otsu is True
        assert thresh == 0.5

    def test_otsu_fallback_no_positive(self):
        """Otsu should fallback when no positive values."""
        values = np.array([-1.0, -0.5, 0.0])
        thresh, used_otsu = _get_otsu_threshold(values, fallback=0.3)

        assert used_otsu is False
        assert thresh == 0.3


class TestEmptyResults:
    """Tests for empty result handling."""

    def test_get_tf_targets_empty_has_columns(self, mock_deepscenic_model):
        """Empty TF targets result should have correct columns."""
        # Use very high threshold to get empty results
        result = get_tf_targets(mock_deepscenic_model, "TF0", e1_threshold=1000.0)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        expected = ["tf", "region", "E1_weight", "gene", "E2_weight", "combined_weight"]
        assert list(result.columns) == expected

    def test_get_gene_regulators_empty_has_columns(self, mock_deepscenic_model):
        """Empty gene regulators result should have correct columns."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", e1_threshold=1000.0)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        expected = ["gene", "tf", "region", "E1_weight", "E2_weight", "combined_weight"]
        assert list(result.columns) == expected


class TestGetTFTargetsExtended:
    """Extended tests for get_tf_targets function."""

    def test_result_columns(self, mock_deepscenic_model):
        """Result should have correct columns."""
        result = get_tf_targets(mock_deepscenic_model, "TF0", e1_threshold=0.0)
        expected_columns = ["tf", "region", "E1_weight", "gene", "E2_weight", "combined_weight"]
        assert list(result.columns) == expected_columns

    def test_tf_column_matches_input(self, mock_deepscenic_model):
        """TF column should match input TF name."""
        result = get_tf_targets(mock_deepscenic_model, "TF0", e1_threshold=0.0)
        if len(result) > 0:
            assert all(result["tf"] == "TF0")

    def test_sorted_by_absolute_weight(self, mock_deepscenic_model):
        """Results should be sorted by absolute combined_weight descending."""
        result = get_tf_targets(mock_deepscenic_model, "TF0", e1_threshold=0.0)
        if len(result) > 1:
            abs_weights = np.abs(result["combined_weight"].values)
            assert all(abs_weights[i] >= abs_weights[i + 1] for i in range(len(abs_weights) - 1))

    def test_e2_threshold_filter(self, mock_deepscenic_model):
        """E2 threshold should filter results."""
        no_thresh = get_tf_targets(mock_deepscenic_model, "TF0", e1_threshold=0.0, e2_threshold=0.0)
        high_thresh = get_tf_targets(mock_deepscenic_model, "TF0", e1_threshold=0.0, e2_threshold=100.0)
        assert len(high_thresh) <= len(no_thresh)


# =============================================================================
# Cell-Type Aware GRN Extraction Tests
# =============================================================================


class TestComputeCelltypeEnhancerActivity:
    """Tests for compute_celltype_enhancer_activity."""

    def test_output_shape(self, mock_deepscenic_model, mock_mdata_for_model):
        """Output should have regions as index, celltypes as columns."""
        mock_mdata_for_model.obs["celltype"] = ["A", "B"] * (len(mock_mdata_for_model.obs) // 2)
        result = compute_celltype_enhancer_activity(mock_deepscenic_model, mock_mdata_for_model, "celltype")

        assert list(result.index) == mock_deepscenic_model.region_names
        assert set(result.columns) == {"A", "B"}

    def test_output_is_dataframe(self, mock_deepscenic_model, mock_mdata_for_model):
        """Output should be a DataFrame."""
        mock_mdata_for_model.obs["celltype"] = ["TypeA"] * len(mock_mdata_for_model.obs)
        result = compute_celltype_enhancer_activity(mock_deepscenic_model, mock_mdata_for_model, "celltype")

        assert isinstance(result, pd.DataFrame)


class TestIdentifyActiveEnhancers:
    """Tests for identify_active_enhancers."""

    def test_returns_dict(self):
        """Should return dict mapping celltype to region list."""
        activity = pd.DataFrame(
            {
                "A": [0.1, 0.2, 0.8, 0.9],
                "B": [0.5, 0.6, 0.1, 0.2],
            },
            index=["r1", "r2", "r3", "r4"],
        )

        result = identify_active_enhancers(activity)

        assert isinstance(result, dict)
        assert "A" in result and "B" in result
        assert all(isinstance(v, list) for v in result.values())

    def test_invalid_method_raises(self):
        """Should raise error for invalid method."""
        activity = pd.DataFrame({"A": [0.1, 0.2]}, index=["r1", "r2"])

        with pytest.raises(ValueError, match="Unknown method"):
            identify_active_enhancers(activity, method="invalid")


class TestComputeTFActivityScores:
    """Tests for compute_tf_activity_scores."""

    def test_output_shape(self, mock_deepscenic_model, mock_mdata_for_model):
        """Output should have TFs as index, celltypes as columns."""
        mock_mdata_for_model.obs["celltype"] = ["A", "B"] * (len(mock_mdata_for_model.obs) // 2)

        # First compute enhancer activity and identify active enhancers
        enh_activity = compute_celltype_enhancer_activity(mock_deepscenic_model, mock_mdata_for_model, "celltype")
        active_enh = identify_active_enhancers(enh_activity)

        result = compute_tf_activity_scores(mock_deepscenic_model, mock_mdata_for_model, "celltype", active_enh)

        assert list(result.index) == mock_deepscenic_model.tf_names
        assert set(result.columns) == {"A", "B"}


class TestIdentifyKeyTFs:
    """Tests for identify_key_tfs."""

    def test_returns_sorted_list(self):
        """Should return sorted unique TF list."""
        scores = pd.DataFrame(
            {
                "A": [0.1, 0.2, 0.8],
                "B": [0.9, 0.1, 0.2],
            },
            index=["TF1", "TF2", "TF3"],
        )

        result = identify_key_tfs(scores)

        assert isinstance(result, list)
        assert result == sorted(result)  # Should be sorted

    def test_invalid_method_raises(self):
        """Should raise error for invalid method."""
        scores = pd.DataFrame({"A": [0.1, 0.2]}, index=["TF1", "TF2"])

        with pytest.raises(ValueError, match="Unknown method"):
            identify_key_tfs(scores, method="invalid")


class TestBuildGRNForTFs:
    """Tests for build_grn_for_tfs."""

    def test_output_columns(self, mock_deepscenic_model):
        """Output should have legacy-compatible columns."""
        result = build_grn_for_tfs(mock_deepscenic_model, ["TF0", "TF1"])

        expected_columns = ["TF", "region", "gene", "tf2r_score", "r2g_score", "tf2g_score"]
        assert list(result.columns) == expected_columns

    def test_filters_to_specified_tfs(self, mock_deepscenic_model):
        """Output should only contain specified TFs."""
        result = build_grn_for_tfs(mock_deepscenic_model, ["TF0"])

        if len(result) > 0:
            assert all(result["TF"] == "TF0")

    def test_empty_tf_list(self, mock_deepscenic_model):
        """Empty TF list should return empty DataFrame with columns."""
        result = build_grn_for_tfs(mock_deepscenic_model, [])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        expected_columns = ["TF", "region", "gene", "tf2r_score", "r2g_score", "tf2g_score"]
        assert list(result.columns) == expected_columns

    def test_invalid_tf_skipped(self, mock_deepscenic_model):
        """Invalid TF names should be skipped with warning."""
        result = build_grn_for_tfs(mock_deepscenic_model, ["INVALID_TF", "TF0"])

        # Should still work, just skip invalid TF
        if len(result) > 0:
            assert all(result["TF"] == "TF0")


class TestCelltypeWorkflowIntegration:
    """Integration tests for cell-type aware GRN workflow."""

    def test_full_workflow(self, mock_deepscenic_model, mock_mdata_for_model):
        """Full cell-type workflow should produce valid GRN."""
        # Add cell type annotations
        n_cells = len(mock_mdata_for_model.obs)
        mock_mdata_for_model.obs["celltype"] = ["TypeA", "TypeB"] * (n_cells // 2)

        # Step 1: Compute enhancer activity
        enh_activity = compute_celltype_enhancer_activity(mock_deepscenic_model, mock_mdata_for_model, "celltype")
        assert enh_activity.shape[0] > 0
        assert set(enh_activity.columns) == {"TypeA", "TypeB"}

        # Step 2: Identify active enhancers
        active_enh = identify_active_enhancers(enh_activity)
        assert isinstance(active_enh, dict)
        assert "TypeA" in active_enh and "TypeB" in active_enh

        # Step 3: Compute TF scores
        tf_scores = compute_tf_activity_scores(mock_deepscenic_model, mock_mdata_for_model, "celltype", active_enh)
        assert tf_scores.shape[0] > 0
        assert set(tf_scores.columns) == {"TypeA", "TypeB"}

        # Step 4: Identify key TFs
        key_tfs = identify_key_tfs(tf_scores)
        assert isinstance(key_tfs, list)

        # Step 5: Build GRN (only if we have key TFs)
        if key_tfs:
            grn = build_grn_for_tfs(mock_deepscenic_model, key_tfs)
            expected_cols = ["TF", "region", "gene", "tf2r_score", "r2g_score", "tf2g_score"]
            assert list(grn.columns) == expected_cols

    def test_workflow_with_single_celltype(self, mock_deepscenic_model, mock_mdata_for_model):
        """Workflow should work with single cell type."""
        mock_mdata_for_model.obs["celltype"] = ["SingleType"] * len(mock_mdata_for_model.obs)

        enh_activity = compute_celltype_enhancer_activity(mock_deepscenic_model, mock_mdata_for_model, "celltype")
        assert "SingleType" in enh_activity.columns

        active_enh = identify_active_enhancers(enh_activity)
        assert "SingleType" in active_enh
