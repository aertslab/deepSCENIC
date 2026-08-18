"""Tests for GRN extraction functions (_grn.py)."""

import numpy as np
import pandas as pd
import pytest

from deepscenic.tl._grn import (
    _get_otsu_threshold,
    _normalize_r2g_per_gene,
    build_grn_for_tfs,
    compute_celltype_r2g,
    compute_celltype_tf2g,
    compute_celltype_tf2r,
    compute_tf_activity_scores,
    extract_grn,
    extract_r2g_matrix,
    extract_tf2r_matrix,
    get_gene_regulators,
    get_region_info,
    get_tf_targets,
    identify_active_enhancers,
)


class TestExtractGRN:
    """Tests for extract_grn function."""

    def test_e1_index_is_tfs(self, mock_deepscenic_model):
        """tf2r index should be TF names."""
        result = extract_grn(mock_deepscenic_model)
        assert list(result["tf2r"].index) == mock_deepscenic_model.tf_names

    def test_e1_columns_are_regions(self, mock_deepscenic_model):
        """tf2r columns should be region names."""
        result = extract_grn(mock_deepscenic_model)
        assert list(result["tf2r"].columns) == mock_deepscenic_model.region_names


class TestExtractTf2rMatrix:
    """Tests for extract_tf2r_matrix function."""

    def test_shape(self, mock_deepscenic_model, minimal_dims):
        """Should have correct shape."""
        result = extract_tf2r_matrix(mock_deepscenic_model)
        d = minimal_dims
        assert result.shape == (d["n_tfs"], d["n_regions"])


class TestExtractR2gMatrix:
    """Tests for extract_r2g_matrix function."""

    def test_edgelist_format_default(self, mock_deepscenic_model, minimal_dims):
        """Edge list format (default) should return correct structure."""
        result = extract_r2g_matrix(mock_deepscenic_model)  # as_edgelist=True by default

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["region", "gene", "weight"]
        assert len(result) == minimal_dims["n_links"]

    def test_edgelist_region_names_valid(self, mock_deepscenic_model):
        """Edge list regions should be valid region names."""
        result = extract_r2g_matrix(mock_deepscenic_model)
        assert all(r in mock_deepscenic_model.region_names for r in result["region"])

    def test_edgelist_gene_names_valid(self, mock_deepscenic_model):
        """Edge list genes should be valid gene names."""
        result = extract_r2g_matrix(mock_deepscenic_model)
        assert all(g in mock_deepscenic_model.gene_names for g in result["gene"])

    def test_dense_format(self, mock_deepscenic_model, minimal_dims):
        """Dense format should return (n_regions, n_genes) DataFrame."""
        result = extract_r2g_matrix(mock_deepscenic_model, as_edgelist=False)
        d = minimal_dims
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (d["n_regions"], d["n_genes"])


class TestComputeCelltypeTf2r:
    """Tests for TF-activity-weighted cell-type tf2r scores."""

    @staticmethod
    def _add_activity(mdata, n_tfs):
        n_cells = len(mdata.obs)
        activity = np.arange(n_cells * n_tfs, dtype=float).reshape(n_cells, n_tfs)
        mdata.obsm["X_deepscenic_z_tf"] = activity
        mdata.obs["celltype"] = ["Oligo"] * 2 + ["Other"] * (n_cells - 2)
        return activity

    def test_dense_matches_legacy_calculation(self, mock_deepscenic_model, mock_mdata_for_model):
        """Dense output should equal mean TF activity[:, None] * E1."""
        n_tfs = len(mock_deepscenic_model.tf_names)
        activity = self._add_activity(mock_mdata_for_model, n_tfs)
        raw_tf2r = extract_tf2r_matrix(mock_deepscenic_model)

        result = compute_celltype_tf2r(
            mock_deepscenic_model,
            mock_mdata_for_model,
            class_key="celltype",
            class_label="Oligo",
        )

        expected = activity[:2].mean(axis=0)[:, None] * raw_tf2r.values
        np.testing.assert_allclose(result.values, expected)
        assert result.index.tolist() == mock_deepscenic_model.tf_names
        assert result.columns.tolist() == mock_deepscenic_model.region_names

    def test_edgelist_scores(self, mock_deepscenic_model, mock_mdata_for_model):
        """Edge-list score should be TF activity times raw tf2r weight."""
        n_tfs = len(mock_deepscenic_model.tf_names)
        self._add_activity(mock_mdata_for_model, n_tfs)

        result = compute_celltype_tf2r(
            mock_deepscenic_model,
            mock_mdata_for_model,
            class_key="celltype",
            class_label="Oligo",
            as_edgelist=True,
        )

        assert list(result.columns) == [
            "tf",
            "region",
            "tf2r_weight",
            "tf_activity",
            "celltype_tf2r_weight",
        ]
        np.testing.assert_allclose(
            result["celltype_tf2r_weight"],
            result["tf_activity"] * result["tf2r_weight"],
        )

    def test_missing_class_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """An absent class label should produce an informative error."""
        n_tfs = len(mock_deepscenic_model.tf_names)
        self._add_activity(mock_mdata_for_model, n_tfs)

        with pytest.raises(ValueError, match="No cells found"):
            compute_celltype_tf2r(
                mock_deepscenic_model,
                mock_mdata_for_model,
                class_key="celltype",
                class_label="Missing",
            )


class TestComputeCelltypeR2g:
    """Tests for enhancer-activity-weighted cell-type r2g scores."""

    @staticmethod
    def _add_activity(mdata, n_regions):
        n_cells = len(mdata.obs)
        activity = np.arange(n_cells * n_regions, dtype=float).reshape(n_cells, n_regions)
        mdata.obsm["X_deepscenic_enh_act"] = activity
        mdata.obs["celltype"] = ["Vip"] * 2 + ["Other"] * (n_cells - 2)
        return activity

    def test_dense_matches_legacy_calculation(self, mock_deepscenic_model, mock_mdata_for_model):
        """Dense output should equal mean enhancer activity[:, None] * E2."""
        n_regions = len(mock_deepscenic_model.region_names)
        activity = self._add_activity(mock_mdata_for_model, n_regions)
        raw_r2g = extract_r2g_matrix(mock_deepscenic_model, as_edgelist=False, normalize=False)

        result = compute_celltype_r2g(
            mock_deepscenic_model,
            mock_mdata_for_model,
            class_key="celltype",
            class_label="Vip",
        )

        expected = raw_r2g.values * activity[:2].mean(axis=0)[:, None]
        np.testing.assert_allclose(result.values, expected)
        assert result.index.tolist() == mock_deepscenic_model.region_names
        assert result.columns.tolist() == mock_deepscenic_model.gene_names

    def test_edgelist_scores(self, mock_deepscenic_model, mock_mdata_for_model):
        """Edge-list score should be enhancer activity times raw r2g weight."""
        n_regions = len(mock_deepscenic_model.region_names)
        self._add_activity(mock_mdata_for_model, n_regions)

        result = compute_celltype_r2g(
            mock_deepscenic_model,
            mock_mdata_for_model,
            class_key="celltype",
            class_label="Vip",
            as_edgelist=True,
        )

        assert list(result.columns) == [
            "region",
            "gene",
            "r2g_weight",
            "enhancer_activity",
            "celltype_r2g_weight",
        ]
        np.testing.assert_allclose(
            result["celltype_r2g_weight"],
            result["enhancer_activity"] * result["r2g_weight"],
        )

    def test_missing_class_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """An absent class label should produce an informative error."""
        n_regions = len(mock_deepscenic_model.region_names)
        self._add_activity(mock_mdata_for_model, n_regions)

        with pytest.raises(ValueError, match="No cells found"):
            compute_celltype_r2g(
                mock_deepscenic_model,
                mock_mdata_for_model,
                class_key="celltype",
                class_label="Missing",
            )


class TestComputeCelltypeTf2g:
    """Tests for TF-activity-weighted cell-type tf2g scores."""

    @staticmethod
    def _add_activity(mdata, n_tfs):
        n_cells = len(mdata.obs)
        activity = np.arange(n_cells * n_tfs, dtype=float).reshape(n_cells, n_tfs)
        mdata.obsm["X_deepscenic_z_tf"] = activity
        mdata.obs["celltype"] = ["Oligo"] * 2 + ["Other"] * (n_cells - 2)
        return activity

    def test_dense_matches_legacy_calculation(self, mock_deepscenic_model, mock_mdata_for_model):
        """Dense output should equal mean TF activity[:, None] * (E1 @ E2)."""
        n_tfs = len(mock_deepscenic_model.tf_names)
        activity = self._add_activity(mock_mdata_for_model, n_tfs)
        tf2r = extract_tf2r_matrix(mock_deepscenic_model)
        r2g = extract_r2g_matrix(mock_deepscenic_model, as_edgelist=False, normalize=False)

        result = compute_celltype_tf2g(
            mock_deepscenic_model,
            mock_mdata_for_model,
            class_key="celltype",
            class_label="Oligo",
        )

        expected = activity[:2].mean(axis=0)[:, None] * (tf2r.values @ r2g.values)
        np.testing.assert_allclose(result.values, expected)
        assert result.index.tolist() == mock_deepscenic_model.tf_names
        assert result.columns.tolist() == mock_deepscenic_model.gene_names

    def test_edgelist_scores(self, mock_deepscenic_model, mock_mdata_for_model):
        """Edge-list score should be TF activity times combined tf2g weight."""
        n_tfs = len(mock_deepscenic_model.tf_names)
        self._add_activity(mock_mdata_for_model, n_tfs)

        result = compute_celltype_tf2g(
            mock_deepscenic_model,
            mock_mdata_for_model,
            class_key="celltype",
            class_label="Oligo",
            as_edgelist=True,
        )

        assert list(result.columns) == [
            "tf",
            "gene",
            "tf2g_weight",
            "tf_activity",
            "celltype_tf2g_weight",
        ]
        np.testing.assert_allclose(
            result["celltype_tf2g_weight"],
            result["tf_activity"] * result["tf2g_weight"],
        )

    def test_missing_class_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """An absent class label should produce an informative error."""
        n_tfs = len(mock_deepscenic_model.tf_names)
        self._add_activity(mock_mdata_for_model, n_tfs)

        with pytest.raises(ValueError, match="No cells found"):
            compute_celltype_tf2g(
                mock_deepscenic_model,
                mock_mdata_for_model,
                class_key="celltype",
                class_label="Missing",
            )


class TestGetTFTargets:
    """Tests for get_tf_targets function."""

    def test_invalid_tf_raises(self, mock_deepscenic_model):
        """Should raise error for invalid TF."""
        with pytest.raises(ValueError, match="not found"):
            get_tf_targets(mock_deepscenic_model, "INVALID_TF")

    def test_threshold_filter(self, mock_deepscenic_model):
        """Should filter by tf2r threshold."""
        high_thresh = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=1.0)
        low_thresh = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=0.0)
        assert len(high_thresh) <= len(low_thresh)


class TestGetGeneRegulators:
    """Tests for get_gene_regulators function."""

    def test_invalid_gene_raises(self, mock_deepscenic_model):
        """Should raise error for invalid gene."""
        with pytest.raises(ValueError, match="not found"):
            get_gene_regulators(mock_deepscenic_model, "INVALID_GENE")

    def test_top_k(self, mock_deepscenic_model):
        """Should limit results with top_k."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", tf2r_threshold=0.0, top_k=5)
        assert len(result) <= 5

    def test_result_columns(self, mock_deepscenic_model):
        """Result should have correct columns."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", tf2r_threshold=0.0)
        expected_columns = ["gene", "tf", "region", "tf2r_weight", "r2g_weight", "combined_weight"]
        assert list(result.columns) == expected_columns

    def test_gene_column_matches_input(self, mock_deepscenic_model):
        """Gene column should match input gene name."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", tf2r_threshold=0.0)
        if len(result) > 0:
            assert all(result["gene"] == "GENE0")

    def test_sorted_by_absolute_weight(self, mock_deepscenic_model):
        """Results should be sorted by absolute combined_weight descending."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", tf2r_threshold=0.0)
        if len(result) > 1:
            abs_weights = np.abs(result["combined_weight"].values)
            assert all(abs_weights[i] >= abs_weights[i + 1] for i in range(len(abs_weights) - 1))

    def test_r2g_threshold_filter(self, mock_deepscenic_model):
        """r2g threshold should filter results."""
        no_thresh = get_gene_regulators(mock_deepscenic_model, "GENE0", tf2r_threshold=0.0, r2g_threshold=0.0)
        high_thresh = get_gene_regulators(mock_deepscenic_model, "GENE0", tf2r_threshold=0.0, r2g_threshold=100.0)
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
        result = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=1000.0)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        expected = ["tf", "region", "tf2r_weight", "gene", "r2g_weight", "combined_weight"]
        assert list(result.columns) == expected

    def test_get_gene_regulators_empty_has_columns(self, mock_deepscenic_model):
        """Empty gene regulators result should have correct columns."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", tf2r_threshold=1000.0)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        expected = ["gene", "tf", "region", "tf2r_weight", "r2g_weight", "combined_weight"]
        assert list(result.columns) == expected


class TestGetTFTargetsExtended:
    """Extended tests for get_tf_targets function."""

    def test_result_columns(self, mock_deepscenic_model):
        """Result should have correct columns."""
        result = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=0.0)
        expected_columns = ["tf", "region", "tf2r_weight", "gene", "r2g_weight", "combined_weight"]
        assert list(result.columns) == expected_columns

    def test_tf_column_matches_input(self, mock_deepscenic_model):
        """TF column should match input TF name."""
        result = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=0.0)
        if len(result) > 0:
            assert all(result["tf"] == "TF0")

    def test_sorted_by_absolute_weight(self, mock_deepscenic_model):
        """Results should be sorted by absolute combined_weight descending."""
        result = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=0.0)
        if len(result) > 1:
            abs_weights = np.abs(result["combined_weight"].values)
            assert all(abs_weights[i] >= abs_weights[i + 1] for i in range(len(abs_weights) - 1))

    def test_r2g_threshold_filter(self, mock_deepscenic_model):
        """r2g threshold should filter results."""
        no_thresh = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=0.0, r2g_threshold=0.0)
        high_thresh = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=0.0, r2g_threshold=100.0)
        assert len(high_thresh) <= len(no_thresh)


# =============================================================================
# Cell-Type Aware GRN Extraction Tests
# =============================================================================


class TestIdentifyActiveEnhancers:
    """Tests for identify_active_enhancers."""

    def test_returns_dict(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should return dict mapping celltype to region list."""
        mock_mdata_for_model.obs["celltype"] = ["A", "B"] * (len(mock_mdata_for_model.obs) // 2)
        result = identify_active_enhancers(mock_deepscenic_model, mock_mdata_for_model, class_key="celltype")
        assert isinstance(result, dict)
        assert "A" in result and "B" in result
        assert all(isinstance(v, list) for v in result.values())

    def test_top_n_limits_results(self, mock_deepscenic_model, mock_mdata_for_model):
        """top_n should cap the number of active enhancers returned per cell type."""
        mock_mdata_for_model.obs["celltype"] = ["A", "B"] * (len(mock_mdata_for_model.obs) // 2)
        result = identify_active_enhancers(mock_deepscenic_model, mock_mdata_for_model, "celltype", top_n=2)
        for regions in result.values():
            assert len(regions) <= 2

    def test_pval_filtering(self, mock_deepscenic_model, mock_mdata_for_model):
        """Stricter pval threshold should return fewer or equal enhancers."""
        mock_mdata_for_model.obs["celltype"] = ["A", "B"] * (len(mock_mdata_for_model.obs) // 2)
        strict = identify_active_enhancers(
            mock_deepscenic_model, mock_mdata_for_model, "celltype", pval_threshold=1e-10
        )
        loose = identify_active_enhancers(mock_deepscenic_model, mock_mdata_for_model, "celltype", pval_threshold=1.0)
        for ct in strict:
            assert len(strict[ct]) <= len(loose[ct])


class TestComputeTFActivityScores:
    """Tests for compute_tf_activity_scores."""

    def test_output_shape(self, mock_deepscenic_model, mock_mdata_for_model):
        """Output should have TFs as index, celltypes as columns."""
        mock_mdata_for_model.obs["celltype"] = ["A", "B"] * (len(mock_mdata_for_model.obs) // 2)

        active_enh = identify_active_enhancers(mock_deepscenic_model, mock_mdata_for_model, "celltype")
        result = compute_tf_activity_scores(
            mock_deepscenic_model,
            mock_mdata_for_model,
            class_key="celltype",
            active_enhancers=active_enh,
        )

        assert list(result.index) == mock_deepscenic_model.tf_names
        assert set(result.columns) == {"A", "B"}

    def test_preserves_signed_scores(self, mock_deepscenic_model, mock_mdata_for_model):
        """TF scores should retain the sign of TF activity times tf2r."""
        n_cells = len(mock_mdata_for_model.obs)
        n_tfs = len(mock_deepscenic_model.tf_names)
        mock_mdata_for_model.obs["celltype"] = ["A"] * n_cells

        tf_activity = np.ones((n_cells, n_tfs))
        tf_activity[:, 0] = -2.0
        mock_mdata_for_model.obsm["X_deepscenic_z_tf"] = tf_activity

        region = mock_deepscenic_model.region_names[0]
        result = compute_tf_activity_scores(
            mock_deepscenic_model,
            mock_mdata_for_model,
            class_key="celltype",
            active_enhancers={"A": [region]},
        )

        tf2r = mock_deepscenic_model.adj_tf2r[0].cpu().numpy()
        expected = tf_activity.mean(axis=0) * tf2r
        np.testing.assert_allclose(result["A"].values, expected)
        assert result["A"].iloc[0] < 0

    def test_zscores_each_tf_across_classes(self, mock_deepscenic_model, mock_mdata_for_model):
        """Optional z-scoring should normalize each TF row across classes."""
        n_cells = len(mock_mdata_for_model.obs)
        n_tfs = len(mock_deepscenic_model.tf_names)
        classes = np.array(["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2))
        mock_mdata_for_model.obs["celltype"] = classes

        tf_activity = np.ones((n_cells, n_tfs))
        tf_activity[classes == "A", 0] = 1.0
        tf_activity[classes == "B", 0] = 3.0
        mock_mdata_for_model.obsm["X_deepscenic_z_tf"] = tf_activity

        region = mock_deepscenic_model.region_names[0]
        result = compute_tf_activity_scores(
            mock_deepscenic_model,
            mock_mdata_for_model,
            class_key="celltype",
            active_enhancers={"A": [region], "B": [region]},
            zscore=True,
        )

        np.testing.assert_allclose(result.mean(axis=1).values, 0.0, atol=1e-7)
        variable = result.std(axis=1, ddof=0) > 0
        np.testing.assert_allclose(result.loc[variable].std(axis=1, ddof=0).values, 1.0)
        assert (result.loc[~variable] == 0).all().all()


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

        # Step 1: Identify active enhancers
        active_enh = identify_active_enhancers(mock_deepscenic_model, mock_mdata_for_model, "celltype")
        assert isinstance(active_enh, dict)
        assert "TypeA" in active_enh and "TypeB" in active_enh

        # Step 2: Compute TF scores
        tf_scores = compute_tf_activity_scores(mock_deepscenic_model, mock_mdata_for_model, "celltype", active_enh)
        assert tf_scores.shape[0] > 0
        assert set(tf_scores.columns) == {"TypeA", "TypeB"}

        # Step 3: Build a focused GRN for explicitly selected TFs
        selected_tfs = list(mock_deepscenic_model.tf_names[:2])
        grn = build_grn_for_tfs(mock_deepscenic_model, selected_tfs)
        expected_cols = ["TF", "region", "gene", "tf2r_score", "r2g_score", "tf2g_score"]
        assert list(grn.columns) == expected_cols

    def test_workflow_with_single_celltype(self, mock_deepscenic_model, mock_mdata_for_model):
        """Workflow should work with single cell type."""
        mock_mdata_for_model.obs["celltype"] = ["SingleType"] * len(mock_mdata_for_model.obs)

        active_enh = identify_active_enhancers(mock_deepscenic_model, mock_mdata_for_model, "celltype")
        assert "SingleType" in active_enh


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalizeR2gPerGene:
    """Tests for _normalize_r2g_per_gene helper."""

    def test_weights_sum_to_one_per_gene(self):
        """Normalized weights for each gene should sum to 1."""
        # Two genes: gene 0 has 3 links, gene 1 has 2 links
        r2g_vals = np.array([0.5, 0.3, 0.2, 0.8, 0.2], dtype=np.float32)
        r2g_indices = np.array(
            [
                [0, 1, 2, 3, 4],  # region indices
                [0, 0, 0, 1, 1],  # gene indices
            ]
        )
        result = _normalize_r2g_per_gene(r2g_vals, r2g_indices)

        # Gene 0: sum = 0.5 + 0.3 + 0.2 = 1.0
        assert abs(result[0] + result[1] + result[2] - 1.0) < 1e-5
        # Gene 1: sum = 0.8 + 0.2 = 1.0
        assert abs(result[3] + result[4] - 1.0) < 1e-5

    def test_proportions_correct(self):
        """Relative proportions should be preserved."""
        r2g_vals = np.array([0.4, 0.6], dtype=np.float32)
        r2g_indices = np.array([[0, 1], [0, 0]])
        result = _normalize_r2g_per_gene(r2g_vals, r2g_indices)

        assert abs(result[0] - 0.4) < 1e-5
        assert abs(result[1] - 0.6) < 1e-5

    def test_zero_total_gene_unchanged(self):
        """Gene with all-zero weights should not produce NaN."""
        r2g_vals = np.array([0.0, 0.0], dtype=np.float32)
        r2g_indices = np.array([[0, 1], [0, 0]])
        result = _normalize_r2g_per_gene(r2g_vals, r2g_indices)

        assert np.all(np.isfinite(result))
        assert np.all(result == 0.0)

    def test_single_gene_single_link(self):
        """Single link for a gene should normalize to 1."""
        r2g_vals = np.array([0.7], dtype=np.float32)
        r2g_indices = np.array([[0], [0]])
        result = _normalize_r2g_per_gene(r2g_vals, r2g_indices)
        assert abs(result[0] - 1.0) < 1e-5


class TestExtractR2gMatrixNormalization:
    """Tests for normalization in extract_r2g_matrix."""

    def test_default_is_normalized(self, mock_deepscenic_model):
        """Default should return normalized weights (per gene sum to 1)."""
        result = extract_r2g_matrix(mock_deepscenic_model, as_edgelist=True)
        # Each gene's weights should sum to ~1
        gene_sums = result.groupby("gene")["weight"].sum()
        for gene, s in gene_sums.items():
            assert abs(s - 1.0) < 1e-4, f"Gene '{gene}' weights sum to {s}, expected ~1.0"

    def test_normalize_false_returns_raw(self, mock_deepscenic_model):
        """normalize=False should return raw (unnormalized) weights."""
        normalized = extract_r2g_matrix(mock_deepscenic_model, as_edgelist=True, normalize=True)
        raw = extract_r2g_matrix(mock_deepscenic_model, as_edgelist=True, normalize=False)
        # Raw weights should differ from normalized (unless there's only one link per gene)
        # At minimum, they should be the same non-negative values, just possibly different scale
        assert all(raw["weight"] >= 0)
        assert len(raw) == len(normalized)

    def test_normalized_weights_in_unit_interval(self, mock_deepscenic_model):
        """Normalized weights should be in [0, 1]."""
        result = extract_r2g_matrix(mock_deepscenic_model, as_edgelist=True, normalize=True)
        assert all(result["weight"] >= 0)
        assert all(result["weight"] <= 1.0 + 1e-6)


class TestGetTFTargetsNormalization:
    """Tests for normalization in get_tf_targets."""

    def test_default_r2g_normalized(self, mock_deepscenic_model):
        """By default, r2g_weight values should be in [0, 1]."""
        result = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=0.0)
        if len(result) > 0:
            assert all(result["r2g_weight"] >= 0)
            assert all(result["r2g_weight"] <= 1.0 + 1e-6)

    def test_normalize_false_returns_raw(self, mock_deepscenic_model):
        """normalize_r2g=False should return raw weights."""
        normed = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=0.0, normalize_r2g=True)
        raw = get_tf_targets(mock_deepscenic_model, "TF0", tf2r_threshold=0.0, normalize_r2g=False)
        if len(raw) > 0:
            assert all(raw["r2g_weight"] >= 0)
        # Both should have same rows (same regions/genes selected), just different r2g_weight scale
        assert len(normed) == len(raw)


class TestGetGeneRegulatorsNormalization:
    """Tests for normalization in get_gene_regulators."""

    def test_default_r2g_normalized(self, mock_deepscenic_model):
        """By default, r2g_weight values should be in [0, 1]."""
        result = get_gene_regulators(mock_deepscenic_model, "GENE0", tf2r_threshold=0.0)
        if len(result) > 0:
            assert all(result["r2g_weight"] >= 0)
            assert all(result["r2g_weight"] <= 1.0 + 1e-6)


class TestGetRegionInfo:
    """Tests for get_region_info function."""

    def test_invalid_region_raises(self, mock_deepscenic_model):
        """Should raise for unknown region names."""
        with pytest.raises(ValueError, match="not found"):
            get_region_info(mock_deepscenic_model, "INVALID_REGION")

    def test_string_input_accepted(self, mock_deepscenic_model):
        """Single string region name should work."""
        region = mock_deepscenic_model.region_names[0]
        result = get_region_info(mock_deepscenic_model, region, top_k_tfs=3, top_k_genes=3)
        assert isinstance(result, pd.DataFrame)

    def test_list_input_accepted(self, mock_deepscenic_model):
        """List of region names should work."""
        regions = mock_deepscenic_model.region_names[:2]
        result = get_region_info(mock_deepscenic_model, list(regions), top_k_tfs=3, top_k_genes=3)
        assert isinstance(result, pd.DataFrame)

    def test_result_columns(self, mock_deepscenic_model):
        """Result should have the expected columns."""
        region = mock_deepscenic_model.region_names[0]
        result = get_region_info(mock_deepscenic_model, region)
        expected = ["region", "tf", "tf2r_weight", "gene", "r2g_weight"]
        assert list(result.columns) == expected

    def test_top_k_tfs_limits(self, mock_deepscenic_model):
        """top_k_tfs should cap number of TFs per region."""
        region = mock_deepscenic_model.region_names[0]
        result = get_region_info(mock_deepscenic_model, region, tf2r_threshold=0.0, top_k_tfs=2, top_k_genes=None)
        if len(result) > 0:
            n_tfs = result["tf"].nunique()
            assert n_tfs <= 2

    def test_top_k_genes_limits(self, mock_deepscenic_model):
        """top_k_genes should cap number of genes per region."""
        region = mock_deepscenic_model.region_names[0]
        result = get_region_info(mock_deepscenic_model, region, tf2r_threshold=0.0, top_k_tfs=None, top_k_genes=2)
        if len(result) > 0:
            n_genes = result["gene"].nunique()
            assert n_genes <= 3  # +1 for possible None gene

    def test_region_column_matches_input(self, mock_deepscenic_model):
        """Region column should match queried region."""
        region = mock_deepscenic_model.region_names[0]
        result = get_region_info(mock_deepscenic_model, region, tf2r_threshold=0.0)
        if len(result) > 0:
            assert all(result["region"] == region)

    def test_empty_result_has_columns(self, mock_deepscenic_model):
        """Very high threshold should return empty DataFrame with correct columns."""
        region = mock_deepscenic_model.region_names[0]
        result = get_region_info(mock_deepscenic_model, region, tf2r_threshold=1e9)
        assert isinstance(result, pd.DataFrame)
        expected = ["region", "tf", "tf2r_weight", "gene", "r2g_weight"]
        assert list(result.columns) == expected

    def test_normalize_r2g_default(self, mock_deepscenic_model):
        """r2g_weight should be in [0, 1] when normalized."""
        region = mock_deepscenic_model.region_names[0]
        result = get_region_info(mock_deepscenic_model, region, tf2r_threshold=0.0, normalize_r2g=True)
        valid = result[result["r2g_weight"].notna()]
        if len(valid) > 0:
            assert all(valid["r2g_weight"] >= 0)
            assert all(valid["r2g_weight"] <= 1.0 + 1e-6)
