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

        ds.write(sample_mdata, path, validate=True)
        loaded = ds.read(path, validate=True)

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


def test_write_read_sparse_r2g_matrix(sample_mdata):
    """Test that sparse r2g matrix is correctly serialized and restored."""
    from scipy.sparse import issparse

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5mu"

        # Verify original has sparse matrix
        assert "r2g" in sample_mdata.uns
        assert "matrix" in sample_mdata.uns["r2g"]
        original_matrix = sample_mdata.uns["r2g"]["matrix"]
        assert issparse(original_matrix)

        # Write and read
        ds.write(sample_mdata, path, validate=True)
        loaded = ds.read(path, validate=True)

        # Verify restored has sparse matrix
        assert "r2g" in loaded.uns
        assert "matrix" in loaded.uns["r2g"]
        loaded_matrix = loaded.uns["r2g"]["matrix"]
        assert issparse(loaded_matrix)

        # Verify data is identical
        assert loaded_matrix.shape == original_matrix.shape
        np.testing.assert_array_almost_equal(
            loaded_matrix.toarray(),
            original_matrix.toarray(),
        )

        # Verify config dict is restored
        assert "config" in loaded.uns["r2g"]
        assert loaded.uns["r2g"]["config"]["max_distance"] == 1000000
        assert loaded.uns["r2g"]["config"]["method"] == "gaussian"

        # Verify region/gene names are restored (may be list or numpy array)
        assert "region_names" in loaded.uns["r2g"]
        assert "gene_names" in loaded.uns["r2g"]
        # AnnData converts lists to numpy arrays, which is semantically equivalent
        np.testing.assert_array_equal(
            loaded.uns["r2g"]["region_names"],
            sample_mdata.uns["r2g"]["region_names"],
        )
        np.testing.assert_array_equal(
            loaded.uns["r2g"]["gene_names"],
            sample_mdata.uns["r2g"]["gene_names"],
        )


def test_write_read_nullable_string_columns():
    """Test roundtrip with nullable string columns containing pd.NA."""
    import pandas as pd
    import scanpy as sc

    rna = sc.AnnData(np.random.rand(10, 5).astype(np.float32))
    rna.var_names = [f"Gene_{i}" for i in range(5)]
    rna.obs_names = [f"Cell_{i}" for i in range(10)]

    # Nullable string column with pd.NA
    rna.var["chromosome"] = pd.array(["chr1", "chr1", pd.NA, "chr7", pd.NA], dtype="string")

    atac = sc.AnnData(np.random.rand(10, 3).astype(np.float32))
    atac.var_names = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(3)]
    atac.obs_names = [f"Cell_{i}" for i in range(10)]

    import mudata as md

    mdata = md.MuData({"rna": rna, "atac": atac})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5mu"
        ds.write(mdata, path, validate=False)
        loaded = ds.read(path, validate=False)

        # Check NA values preserved
        assert pd.isna(loaded.mod["rna"].var["chromosome"].iloc[2])
        assert pd.isna(loaded.mod["rna"].var["chromosome"].iloc[4])
        # Check non-NA values
        assert loaded.mod["rna"].var["chromosome"].iloc[0] == "chr1"
        assert loaded.mod["rna"].var["chromosome"].iloc[3] == "chr7"


def test_write_read_nullable_integer_columns():
    """Test roundtrip with nullable integer columns containing pd.NA."""
    import pandas as pd
    import scanpy as sc

    rna = sc.AnnData(np.random.rand(10, 5).astype(np.float32))
    rna.var_names = [f"Gene_{i}" for i in range(5)]
    rna.obs_names = [f"Cell_{i}" for i in range(10)]

    # Nullable integer column with pd.NA
    rna.var["tss"] = pd.array([1000, 2000, pd.NA, 4000, pd.NA], dtype="Int64")

    atac = sc.AnnData(np.random.rand(10, 3).astype(np.float32))
    atac.var_names = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(3)]
    atac.obs_names = [f"Cell_{i}" for i in range(10)]

    import mudata as md

    mdata = md.MuData({"rna": rna, "atac": atac})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5mu"
        ds.write(mdata, path, validate=False)
        loaded = ds.read(path, validate=False)

        # Check NA values preserved
        assert pd.isna(loaded.mod["rna"].var["tss"].iloc[2])
        assert pd.isna(loaded.mod["rna"].var["tss"].iloc[4])
        # Check non-NA values
        assert loaded.mod["rna"].var["tss"].iloc[0] == 1000
        assert loaded.mod["rna"].var["tss"].iloc[3] == 4000


def test_write_read_nested_dict_in_uns():
    """Test roundtrip with nested dicts in uns."""
    import scanpy as sc

    rna = sc.AnnData(np.random.rand(10, 5).astype(np.float32))
    atac = sc.AnnData(np.random.rand(10, 3).astype(np.float32))

    import mudata as md

    mdata = md.MuData({"rna": rna, "atac": atac})
    mdata.uns["r2g"] = {
        "config": {"max_distance": 1000000, "sigma": 100000, "method": "gaussian"},
        "other_key": "value",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5mu"
        ds.write(mdata, path, validate=False)
        loaded = ds.read(path, validate=False)

        # Check nested dict preserved
        assert "r2g" in loaded.uns
        assert "config" in loaded.uns["r2g"]
        assert loaded.uns["r2g"]["config"]["max_distance"] == 1000000
        assert loaded.uns["r2g"]["config"]["sigma"] == 100000
        assert loaded.uns["r2g"]["config"]["method"] == "gaussian"
        assert loaded.uns["r2g"]["other_key"] == "value"


def test_write_read_none_in_uns():
    """Test roundtrip with None values in uns."""
    import scanpy as sc

    rna = sc.AnnData(np.random.rand(10, 5).astype(np.float32))
    rna.uns["log1p"] = {"base": None}

    atac = sc.AnnData(np.random.rand(10, 3).astype(np.float32))

    import mudata as md

    mdata = md.MuData({"rna": rna, "atac": atac})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5mu"
        ds.write(mdata, path, validate=False)
        loaded = ds.read(path, validate=False)

        # Check None preserved
        assert "log1p" in loaded.mod["rna"].uns
        assert loaded.mod["rna"].uns["log1p"]["base"] is None


def test_write_read_object_dtype_with_na():
    """Test roundtrip converts object dtype with pd.NA to proper nullable dtypes."""
    import pandas as pd
    import scanpy as sc

    rna = sc.AnnData(np.random.rand(10, 5).astype(np.float32))
    rna.var_names = [f"Gene_{i}" for i in range(5)]
    rna.obs_names = [f"Cell_{i}" for i in range(10)]

    # Object dtype column with pd.NA (what add_gene_annotation produces)
    chromosomes = ["chr1", "chr1", pd.NA, "chr7", pd.NA]
    rna.var["chromosome"] = pd.Series(chromosomes, index=rna.var_names)
    tss_values = [1000, 2000, pd.NA, 4000, pd.NA]
    rna.var["tss"] = pd.Series(tss_values, index=rna.var_names)

    # Verify object dtype before write
    assert rna.var["chromosome"].dtype == object
    assert rna.var["tss"].dtype == object

    atac = sc.AnnData(np.random.rand(10, 3).astype(np.float32))
    atac.var_names = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(3)]
    atac.obs_names = [f"Cell_{i}" for i in range(10)]

    import mudata as md

    mdata = md.MuData({"rna": rna, "atac": atac})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5mu"
        ds.write(mdata, path, validate=False)
        loaded = ds.read(path, validate=False)

        # Check NA values preserved
        assert pd.isna(loaded.mod["rna"].var["chromosome"].iloc[2])
        assert pd.isna(loaded.mod["rna"].var["tss"].iloc[2])
        # Check non-NA values
        assert loaded.mod["rna"].var["chromosome"].iloc[0] == "chr1"
        assert loaded.mod["rna"].var["tss"].iloc[0] == 1000


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

        # Read with validation - should warn but not raise with strict False
        with pytest.warns(ds.SchemaWarning):
            loaded = ds.read(path, validate=True, strict=False)

        assert loaded is not None


class TestReadBed:
    """Tests for read_bed function."""

    def test_basic_reading(self, tmp_path):
        """Test basic BED file reading without validation."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text("chr1\t1000\t2000\nchr1\t3000\t4000\nchr2\t5000\t6000\n")

        regions = ds.read_bed(bed_file)

        assert len(regions) == 3
        assert list(regions.columns) == ["chromosome", "start", "end", "region"]

        # Check first region
        assert regions.iloc[0]["chromosome"] == "chr1"
        assert regions.iloc[0]["start"] == 1000
        assert regions.iloc[0]["end"] == 2000
        assert regions.iloc[0]["region"] == "chr1:1000-2000"

        # Check last region
        assert regions.iloc[2]["chromosome"] == "chr2"
        assert regions.iloc[2]["region"] == "chr2:5000-6000"

    def test_extra_columns_ignored(self, tmp_path):
        """Test that extra BED columns are ignored."""
        bed_file = tmp_path / "regions.bed"
        # Standard BED6 format: chrom, start, end, name, score, strand
        bed_file.write_text("chr1\t1000\t2000\tpeak_1\t500\t+\nchr1\t3000\t4000\tpeak_2\t300\t-\n")

        regions = ds.read_bed(bed_file)

        # Should only use first 3 columns
        assert len(regions) == 2
        assert regions.iloc[0]["region"] == "chr1:1000-2000"
        assert regions.iloc[1]["region"] == "chr1:3000-4000"

    def test_region_string_format(self, tmp_path):
        """Test region string format."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text("chr10\t12345\t67890\n")

        regions = ds.read_bed(bed_file)

        assert regions.iloc[0]["region"] == "chr10:12345-67890"

    def test_file_not_found(self, tmp_path):
        """Test error when BED file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.bed"

        with pytest.raises(FileNotFoundError, match="BED file not found"):
            ds.read_bed(nonexistent)

    def test_empty_bed_file(self, tmp_path):
        """Test error on empty BED file."""
        bed_file = tmp_path / "empty.bed"
        bed_file.write_text("")

        with pytest.raises(ValueError, match="BED file is empty"):
            ds.read_bed(bed_file)

    def test_basic_coordinate_validation_negative_start(self, tmp_path):
        """Test filtering regions with negative start position."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text(
            "chr1\t1000\t2000\n"
            "chr1\t-100\t500\n"  # Invalid: negative start
            "chr2\t3000\t4000\n"
        )

        regions = ds.read_bed(bed_file)

        # Invalid region should be filtered
        assert len(regions) == 2
        assert "chr1:1000-2000" in regions["region"].values
        assert "chr2:3000-4000" in regions["region"].values
        assert "chr1:-100-500" not in regions["region"].values

    def test_basic_coordinate_validation_start_after_end(self, tmp_path):
        """Test filtering regions where start >= end."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text(
            "chr1\t1000\t2000\n"
            "chr1\t5000\t5000\n"  # Invalid: start == end
            "chr1\t6000\t5000\n"  # Invalid: start > end
            "chr2\t3000\t4000\n"
        )

        regions = ds.read_bed(bed_file)

        # Only valid regions should remain
        assert len(regions) == 2
        assert "chr1:1000-2000" in regions["region"].values
        assert "chr2:3000-4000" in regions["region"].values

    def test_validate_with_chromsizes_dataframe(self, tmp_path):
        """Test validation with chromsizes DataFrame."""
        import pandas as pd

        bed_file = tmp_path / "regions.bed"
        bed_file.write_text(
            "chr1\t1000\t2000\n"
            "chr2\t3000\t4000\n"
            "chrZ\t5000\t6000\n"  # Invalid: unknown chromosome
            "chr1\t100000\t200000\n"  # Invalid: end > chr length
        )

        # Create chromsizes DataFrame
        chromsizes = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr2"],
                "Start": [0, 0],
                "End": [50000, 100000],
            }
        )

        regions = ds.read_bed(bed_file, chromsizes=chromsizes)

        # Only regions on chr1 and chr2 within bounds should remain
        assert len(regions) == 2
        assert "chr1:1000-2000" in regions["region"].values
        assert "chr2:3000-4000" in regions["region"].values
        assert "chrZ:5000-6000" not in regions["region"].values
        assert "chr1:100000-200000" not in regions["region"].values

    def test_validate_with_chromsizes_file(self, tmp_path):
        """Test validation with chromsizes file path."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text(
            "chr1\t1000\t2000\nchr2\t3000\t4000\nchr3\t5000\t6000\n"  # Invalid: not in chromsizes
        )

        # Create chromsizes file
        chromsizes_file = tmp_path / "chromsizes.txt"
        chromsizes_file.write_text("chr1\t50000\nchr2\t100000\n")

        regions = ds.read_bed(bed_file, chromsizes=chromsizes_file)

        # Only chr1 and chr2 regions should remain
        assert len(regions) == 2
        assert "chr1:1000-2000" in regions["region"].values
        assert "chr2:3000-4000" in regions["region"].values
        assert "chr3:5000-6000" not in regions["region"].values

    def test_no_validation_without_chromsizes(self, tmp_path):
        """Test that validation is skipped when chromsizes not provided."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text(
            "chr1\t1000\t2000\nchrZ\t5000\t6000\n"  # Would be invalid if chromsizes provided
        )

        # No chromsizes provided, so no chromosome validation
        regions = ds.read_bed(bed_file)

        # Both regions should be present (validation skipped)
        assert len(regions) == 2
        assert "chrZ:5000-6000" in regions["region"].values

    def test_all_regions_filtered_raises_error(self, tmp_path):
        """Test error when all regions are filtered out."""
        import pandas as pd

        bed_file = tmp_path / "regions.bed"
        # BED with Ensembl-style chromosomes (1, 2, 3)
        bed_file.write_text("1\t1000\t2000\n2\t3000\t4000\n3\t5000\t6000\n")

        # Chromsizes with UCSC-style chromosomes (chr1, chr2, chr3)
        chromsizes = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr2", "chr3"],
                "Start": [0, 0, 0],
                "End": [100000, 100000, 100000],
            }
        )

        with pytest.raises(ValueError, match="All regions.*filtered out"):
            ds.read_bed(bed_file, chromsizes=chromsizes)

    def test_chromsizes_file_not_found(self, tmp_path):
        """Test error when chromsizes file doesn't exist."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text("chr1\t1000\t2000\n")

        nonexistent = tmp_path / "nonexistent_chromsizes.txt"

        with pytest.raises(FileNotFoundError, match="Chromsizes file not found"):
            ds.read_bed(bed_file, chromsizes=nonexistent)

    def test_get_region_list(self, tmp_path):
        """Test converting DataFrame to list of region strings."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text("chr1\t1000\t2000\nchr1\t3000\t4000\n")

        regions_df = ds.read_bed(bed_file)
        regions_list = regions_df["region"].tolist()

        assert isinstance(regions_list, list)
        assert len(regions_list) == 2
        assert regions_list[0] == "chr1:1000-2000"
        assert regions_list[1] == "chr1:3000-4000"
