"""Tests for schema validation."""

import mudata as md
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from deepscenic.data.schema import SchemaError, is_valid_schema, validate_schema


def test_validate_missing_rna():
    """Test error when RNA modality is missing."""
    atac = sc.AnnData(np.random.rand(10, 20))
    mdata = md.MuData({"atac": atac})

    issues = validate_schema(mdata, strict=False)
    assert "Missing 'rna' modality" in issues


def test_validate_missing_atac_training():
    """Test error when ATAC is missing in training mode."""
    rna = sc.AnnData(np.random.randn(10, 20))
    mdata = md.MuData({"rna": rna})

    issues = validate_schema(mdata, mode="training", strict=False)
    assert any("atac" in issue.lower() for issue in issues)


def test_validate_missing_atac_inference_ok():
    """Test that missing ATAC is OK in inference mode."""
    rna = sc.AnnData(np.random.randn(10, 20))
    rna.var["is_tf"] = True
    rna.uns["tf_order"] = []
    mdata = md.MuData({"rna": rna})
    mdata.obs["split"] = pd.Categorical(["train"] * 10)
    mdata.uns["deepscenic_version"] = "0.1.0"

    issues = validate_schema(mdata, mode="inference", strict=False)
    # Should not complain about missing ATAC
    assert not any("atac" in issue.lower() for issue in issues)


def test_validate_missing_tf_columns(minimal_mdata):
    """Test error when TF columns are missing."""
    issues = validate_schema(minimal_mdata, strict=False)
    assert "rna.var missing 'is_tf' column" in issues


def test_validate_missing_region_columns(minimal_mdata):
    """Test error when region coordinate columns are missing."""
    issues = validate_schema(minimal_mdata, strict=False)
    assert any("chromosome" in issue for issue in issues)


def test_validate_missing_split_column(minimal_mdata):
    """Test error when split column is missing."""
    issues = validate_schema(minimal_mdata, strict=False)
    assert "mdata.obs missing 'split' column" in issues


def test_validate_missing_r2g_training(minimal_mdata):
    """Test error when r2g is missing in training mode."""
    issues = validate_schema(minimal_mdata, mode="training", strict=False)
    assert any("r2g" in issue for issue in issues)


def test_validate_r2g_missing_ordering(sample_mdata):
    """Test error when r2g ordering arrays are missing."""
    # Remove ordering arrays
    del sample_mdata.uns["r2g"]["region_order_train"]

    issues = validate_schema(sample_mdata, mode="training", strict=False)
    assert any("region_order_train" in issue for issue in issues)


def test_validate_strict_raises(minimal_mdata):
    """Test that strict mode raises SchemaError."""
    with pytest.raises(SchemaError):
        validate_schema(minimal_mdata, mode="training", strict=True)


def test_validate_invalid_mode():
    """Test error on invalid mode."""
    rna = sc.AnnData(np.random.randn(10, 20))
    mdata = md.MuData({"rna": rna})

    with pytest.raises(ValueError, match="mode must be"):
        validate_schema(mdata, mode="invalid")


def test_is_valid_schema_false(minimal_mdata):
    """Test is_valid_schema returns False for incomplete schema."""
    assert is_valid_schema(minimal_mdata, mode="training") is False


def test_is_valid_schema_true(sample_mdata):
    """Test is_valid_schema returns True for complete schema."""
    # sample_mdata should have all required fields
    assert is_valid_schema(sample_mdata, mode="training") is True


def test_validate_complete_schema(sample_mdata):
    """Test that a complete schema passes validation."""
    issues = validate_schema(sample_mdata, mode="training", strict=False)
    assert len(issues) == 0
