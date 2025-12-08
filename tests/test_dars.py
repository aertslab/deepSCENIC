"""Tests for DAR handling functionality."""

import numpy as np
import pytest
import scanpy as sc
from mudata import MuData

import deepscenic as ds


@pytest.fixture
def sample_mdata_for_dars():
    """Create sample MuData with ATAC regions for DAR testing."""
    # RNA data
    rna = sc.AnnData(np.random.randn(10, 20).astype(np.float32))
    rna.obs_names = [f"Cell_{i}" for i in range(10)]
    rna.var_names = [f"Gene_{i}" for i in range(20)]

    # ATAC data with realistic region names
    atac = sc.AnnData(np.random.rand(10, 15).astype(np.float32))
    atac.obs_names = [f"Cell_{i}" for i in range(10)]
    atac.var_names = [
        "chr1:1000-2000",
        "chr1:3000-4000",
        "chr1:5000-6000",
        "chr2:1000-2000",
        "chr2:3000-4000",
        "chr2:5000-6000",
        "chr7:1000-2000",
        "chr7:3000-4000",
        "chr7:5000-6000",
        "chr11:1000-2000",
        "chr11:3000-4000",
        "chr11:5000-6000",
        "chr18:1000-2000",
        "chr18:3000-4000",
        "chr19:1000-2000",
    ]

    mdata = MuData({"rna": rna, "atac": atac})
    return mdata


class TestMarkDars:
    """Tests for mark_dars function."""

    def test_mark_dars_from_dict(self, sample_mdata_for_dars):
        """Test marking DARs from dictionary."""
        dar_dict = {
            "Astro": ["chr1:1000-2000", "chr1:3000-4000"],
            "Oligo": ["chr2:1000-2000", "chr7:1000-2000"],
        }

        ds.pp.mark_dars(sample_mdata_for_dars, dar_dict=dar_dict)

        # Check is_dar column exists and is boolean
        assert "is_dar" in sample_mdata_for_dars.mod["atac"].var.columns
        assert sample_mdata_for_dars.mod["atac"].var["is_dar"].dtype == bool

        # Check correct regions are marked
        assert sample_mdata_for_dars.mod["atac"].var.loc["chr1:1000-2000", "is_dar"]
        assert sample_mdata_for_dars.mod["atac"].var.loc["chr1:3000-4000", "is_dar"]
        assert sample_mdata_for_dars.mod["atac"].var.loc["chr2:1000-2000", "is_dar"]
        assert sample_mdata_for_dars.mod["atac"].var.loc["chr7:1000-2000", "is_dar"]

        # Check non-DAR regions are False
        assert not sample_mdata_for_dars.mod["atac"].var.loc["chr1:5000-6000", "is_dar"]
        assert not sample_mdata_for_dars.mod["atac"].var.loc["chr11:1000-2000", "is_dar"]

        # Check count
        assert sample_mdata_for_dars.mod["atac"].var["is_dar"].sum() == 4

    def test_mark_dars_from_dir(self, sample_mdata_for_dars, tmp_path):
        """Test marking DARs from directory of BED files."""
        # Create BED files
        astro_bed = tmp_path / "Astro.bed"
        astro_bed.write_text("chr1\t1000\t2000\nchr1\t3000\t4000\n")

        oligo_bed = tmp_path / "Oligo.bed"
        oligo_bed.write_text("chr2\t1000\t2000\nchr7\t1000\t2000\n")

        ds.pp.mark_dars(sample_mdata_for_dars, dar_dir=tmp_path)

        # Check correct regions are marked
        assert sample_mdata_for_dars.mod["atac"].var["is_dar"].sum() == 4

    def test_mark_dars_inplace_false(self, sample_mdata_for_dars):
        """Test mark_dars with inplace=False."""
        dar_dict = {"Astro": ["chr1:1000-2000"]}

        result = ds.pp.mark_dars(sample_mdata_for_dars, dar_dict=dar_dict, inplace=False)

        assert result is not None
        assert result is not sample_mdata_for_dars
        assert "is_dar" in result.mod["atac"].var.columns

    def test_mark_dars_missing_regions_handled(self, sample_mdata_for_dars):
        """Test that DAR regions not in data are ignored gracefully."""
        dar_dict = {
            "Astro": ["chr1:1000-2000", "chr99:9999-10000"],  # chr99 doesn't exist
        }

        ds.pp.mark_dars(sample_mdata_for_dars, dar_dict=dar_dict)

        # Only chr1:1000-2000 should be marked
        assert sample_mdata_for_dars.mod["atac"].var["is_dar"].sum() == 1

    def test_mark_dars_no_source_raises(self, sample_mdata_for_dars):
        """Test error when neither dar_dir nor dar_dict provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            ds.pp.mark_dars(sample_mdata_for_dars)

    def test_mark_dars_no_atac_raises(self):
        """Test error when ATAC modality missing."""
        rna = sc.AnnData(np.random.randn(10, 20))
        mdata = MuData({"rna": rna})

        with pytest.raises(ValueError, match="atac"):
            ds.pp.mark_dars(mdata, dar_dict={"Astro": []})


class TestLoadDarsFromBed:
    """Tests for load_dars_from_bed function."""

    def test_load_single_bed(self, tmp_path):
        """Test loading regions from single BED file."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text("chr1\t1000\t2000\nchr1\t3000\t4000\nchr2\t5000\t6000\n")

        regions = ds.pp.load_dars_from_bed(bed_file)

        assert len(regions) == 3
        assert regions[0] == "chr1:1000-2000"
        assert regions[1] == "chr1:3000-4000"
        assert regions[2] == "chr2:5000-6000"

    def test_load_bed_with_extra_columns(self, tmp_path):
        """Test loading BED file that has extra columns (e.g., name, score)."""
        bed_file = tmp_path / "regions.bed"
        bed_file.write_text("chr1\t1000\t2000\tpeak_1\t500\nchr1\t3000\t4000\tpeak_2\t300\n")

        regions = ds.pp.load_dars_from_bed(bed_file)

        # Should only use first 3 columns
        assert len(regions) == 2
        assert regions[0] == "chr1:1000-2000"
        assert regions[1] == "chr1:3000-4000"
