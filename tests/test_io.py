"""Tests for I/O functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import deepscenic as ds


def test_write_read_roundtrip(sample_mdata):
    """Test that write then read produces identical data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5mu"

        # Write without validation (sample_mdata may not pass strict validation)
        ds.write(sample_mdata, path, validate=False)

        # Read back
        loaded = ds.read(path, validate=False)

        # Check shapes
        assert loaded.n_obs == sample_mdata.n_obs
        assert loaded.mod["rna"].n_vars == sample_mdata.mod["rna"].n_vars
        assert loaded.mod["atac"].n_vars == sample_mdata.mod["atac"].n_vars

        # Check RNA data
        np.testing.assert_array_almost_equal(
            loaded.mod["rna"].X,
            sample_mdata.mod["rna"].X,
        )

        # Check ATAC data
        np.testing.assert_array_almost_equal(
            loaded.mod["atac"].X,
            sample_mdata.mod["atac"].X,
        )


def test_read_nonexistent_file():
    """Test error on nonexistent file."""
    with pytest.raises(FileNotFoundError):
        ds.read("nonexistent.h5mu")


def test_read_invalid_extension():
    """Test error on invalid file extension."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        with pytest.raises(ValueError, match="Expected .h5mu file"):
            ds.read(f.name)


def test_write_creates_directory(sample_mdata):
    """Test that write creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "nested" / "test.h5mu"

        ds.write(sample_mdata, path, validate=False)

        assert path.exists()
        assert path.parent.exists()


def test_read_with_validation_warnings(minimal_mdata):
    """Test that read with validation emits warnings for incomplete schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5mu"

        # Write minimal data
        minimal_mdata.write(path)

        # Read with validation - should warn but not raise
        with pytest.warns(ds.data.SchemaWarning):
            loaded = ds.read(path, validate=True)

        assert loaded is not None
