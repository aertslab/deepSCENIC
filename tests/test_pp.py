"""Tests for preprocessing functions."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import deepscenic as ds


class TestRemoveZeroVarianceGenes:
    """Tests for remove_zero_variance_genes function."""

    def test_removes_zero_variance(self):
        """Test that zero-variance genes are removed."""
        adata = sc.AnnData(np.random.randn(100, 50).astype(np.float32))
        adata.var_names = [f"Gene_{i}" for i in range(50)]

        # Set first 5 genes to constant value (zero variance)
        adata.X[:, :5] = 1.0

        original_n_genes = adata.n_vars
        ds.pp.remove_zero_variance_genes(adata)

        assert adata.n_vars == original_n_genes - 5
        assert adata.n_vars == 45

    def test_keeps_all_if_no_zero_variance(self):
        """Test that all genes kept if none have zero variance."""
        adata = sc.AnnData(np.random.randn(100, 50).astype(np.float32))
        adata.var_names = [f"Gene_{i}" for i in range(50)]

        ds.pp.remove_zero_variance_genes(adata)

        assert adata.n_vars == 50

    def test_inplace_false(self):
        """Test inplace=False returns new object."""
        adata = sc.AnnData(np.random.randn(100, 50).astype(np.float32))
        adata.X[:, :5] = 1.0  # Zero variance

        result = ds.pp.remove_zero_variance_genes(adata, inplace=False)

        assert result is not None
        assert result is not adata
        assert result.n_vars == 45
        assert adata.n_vars == 50  # Original unchanged


class TestComputeR2GPenalty:
    """Tests for compute_r2g_penalty function."""

    @pytest.fixture
    def mdata_with_regions(self, sample_rna, sample_atac):
        """Create MuData with parsed region coordinates."""
        mdata = ds.pp.create_mudata(sample_rna, sample_atac)
        ds.pp.parse_region_coordinates(mdata)
        return mdata

    def test_compute_r2g_penalty_basic(self, mdata_with_regions):
        """Test basic R2G computation stores in uns."""
        genes = pd.DataFrame(
            {
                "tss": [5000, 15000, 25000],
                "chromosome": ["chr1"] * 3,
            },
            index=["GeneA", "GeneB", "GeneC"],
        )

        ds.pp.compute_r2g_penalty(
            mdata_with_regions,
            genes,
            max_distance=50000,
            sigma=10000,
        )

        # Check results stored in uns
        assert "r2g" in mdata_with_regions.uns
        r2g_data = mdata_with_regions.uns["r2g"]
        assert "matrix" in r2g_data
        assert "config" in r2g_data
        assert "region_names" in r2g_data
        assert "gene_names" in r2g_data

        # Check matrix properties
        assert r2g_data["matrix"].nnz > 0
        assert r2g_data["config"]["n_links"] > 0

    def test_compute_r2g_penalty_requires_parsed_coords(self, sample_rna, sample_atac):
        """Test error when coordinates not parsed."""
        mdata = ds.pp.create_mudata(sample_rna, sample_atac)

        # Remove parsed coordinates
        del mdata.mod["atac"].var["chromosome"]

        genes = pd.DataFrame(
            {"tss": [100], "chromosome": ["chr1"]},
            index=["GeneA"],
        )

        with pytest.raises(ValueError, match="parse_region_coordinates"):
            ds.pp.compute_r2g_penalty(mdata, genes)

    def test_compute_r2g_penalty_copy(self, mdata_with_regions):
        """Test copy=True returns new MuData."""
        genes = pd.DataFrame(
            {"tss": [5000], "chromosome": ["chr1"]},
            index=["GeneA"],
        )

        result = ds.pp.compute_r2g_penalty(
            mdata_with_regions, genes, copy=True
        )

        assert result is not None
        assert result is not mdata_with_regions
        assert "r2g" in result.uns
        assert "r2g" not in mdata_with_regions.uns

    def test_compute_r2g_penalty_key_added(self, mdata_with_regions):
        """Test custom key_added parameter."""
        genes = pd.DataFrame(
            {"tss": [5000], "chromosome": ["chr1"]},
            index=["GeneA"],
        )

        ds.pp.compute_r2g_penalty(
            mdata_with_regions, genes, key_added="custom_r2g"
        )

        assert "custom_r2g" in mdata_with_regions.uns
        assert "r2g" not in mdata_with_regions.uns

    def test_compute_r2g_penalty_no_cross_chromosome(self, sample_rna):
        """Test that R2G doesn't link across chromosomes."""
        # Create ATAC with 2 chromosomes
        n_cells = sample_rna.n_obs
        atac = sc.AnnData(np.random.rand(n_cells, 2))
        atac.obs_names = sample_rna.obs_names
        atac.var_names = ["chr1:0-640", "chr2:0-640"]

        mdata = ds.pp.create_mudata(sample_rna, atac)
        ds.pp.parse_region_coordinates(mdata)

        genes = pd.DataFrame(
            {
                "tss": [100, 100],
                "chromosome": ["chr1", "chr2"],
            },
            index=["GeneA", "GeneB"],
        )

        ds.pp.compute_r2g_penalty(mdata, genes, max_distance=1000)

        # Should only have 2 links (chr1-chr1 and chr2-chr2), not 4
        assert mdata.uns["r2g"]["matrix"].nnz == 2

    def test_compute_r2g_penalty_gaussian(self, sample_rna):
        """Test Gaussian penalty values."""
        # Create ATAC with single region
        n_cells = sample_rna.n_obs
        atac = sc.AnnData(np.random.rand(n_cells, 1))
        atac.obs_names = sample_rna.obs_names
        atac.var_names = ["chr1:0-640"]

        mdata = ds.pp.create_mudata(sample_rna, atac)
        ds.pp.parse_region_coordinates(mdata)

        genes = pd.DataFrame(
            {
                "tss": [320],  # Region center
                "chromosome": ["chr1"],
            },
            index=["GeneA"],
        )

        ds.pp.compute_r2g_penalty(
            mdata, genes, max_distance=10000, sigma=1000
        )

        # Distance = 0, so penalty should be ~0
        assert mdata.uns["r2g"]["matrix"][0, 0] < 0.01


class TestSplitR2GByChromosome:
    """Tests for split_r2g_by_chromosome function."""

    def test_split_r2g_basic(self):
        """Test basic R2G splitting."""
        from scipy.sparse import csr_matrix

        # Create full R2G matrix
        regions_df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr1", "chr7", "chr7"],
            },
            index=["r1", "r2", "r3", "r4"],
        )
        genes_df = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr1", "chr7"],
            },
            index=["g1", "g2", "g3"],
        )

        r2g = csr_matrix(np.random.rand(4, 3))

        r2g_train, r2g_test, split_info = ds.pp.split_r2g_by_chromosome(
            r2g, regions_df, genes_df, test_chromosomes=["chr7"]
        )

        # Train should have chr1 regions/genes
        assert r2g_train.shape == (2, 2)
        # Test should have chr7 regions/genes
        assert r2g_test.shape == (2, 1)
        assert split_info["n_train_regions"] == 2
        assert split_info["n_test_regions"] == 2


class TestCreateMuData:
    """Tests for create_mudata function."""

    def test_create_mudata_basic(self, sample_rna, sample_atac):
        """Test basic MuData creation."""
        mdata = ds.pp.create_mudata(sample_rna, sample_atac)

        assert "rna" in mdata.mod
        assert "atac" in mdata.mod
        assert mdata.n_obs == 100

    def test_create_mudata_mismatched_cells(self):
        """Test error on mismatched cell names."""
        rna = sc.AnnData(np.random.randn(10, 20))
        rna.obs_names = [f"Cell_{i}" for i in range(10)]

        atac = sc.AnnData(np.random.rand(10, 15))
        atac.obs_names = [f"DifferentCell_{i}" for i in range(10)]

        with pytest.raises(ValueError, match="Cell names must match"):
            ds.pp.create_mudata(rna, atac)


class TestMarkTFs:
    """Tests for mark_tfs function."""

    def test_mark_tfs_from_list(self, sample_rna, sample_atac):
        """Test marking TFs from a list."""
        mdata = ds.pp.create_mudata(sample_rna, sample_atac)

        # Remove existing TF marks
        del mdata.mod["rna"].var["is_tf"]
        del mdata.mod["rna"].var["tf_index"]
        del mdata.mod["rna"].uns["tf_order"]

        tf_list = ["Gene_0", "Gene_1", "Gene_2", "NonExistent"]
        ds.pp.mark_tfs(mdata, tf_list)

        # Check TFs are marked
        assert mdata.mod["rna"].var["is_tf"].sum() == 3
        assert len(mdata.mod["rna"].uns["tf_order"]) == 3
        assert "NonExistent" not in mdata.mod["rna"].uns["tf_order"]


class TestSplitCells:
    """Tests for split_cells function."""

    def test_split_cells_basic(self, sample_rna, sample_atac):
        """Test basic cell splitting."""
        mdata = ds.pp.create_mudata(sample_rna, sample_atac)

        ds.pp.split_cells(mdata, test_fraction=0.2, seed=42)

        assert "split" in mdata.obs.columns
        train_count = (mdata.obs["split"] == "train").sum()
        test_count = (mdata.obs["split"] == "test").sum()

        assert train_count == 80
        assert test_count == 20

    def test_split_cells_reproducible(self, sample_rna, sample_atac):
        """Test that splitting is reproducible with same seed."""
        mdata1 = ds.pp.create_mudata(sample_rna, sample_atac)
        mdata2 = ds.pp.create_mudata(sample_rna, sample_atac)

        ds.pp.split_cells(mdata1, test_fraction=0.2, seed=42)
        ds.pp.split_cells(mdata2, test_fraction=0.2, seed=42)

        assert (mdata1.obs["split"] == mdata2.obs["split"]).all()


class TestParseRegionCoordinates:
    """Tests for parse_region_coordinates function."""

    def test_parse_coordinates(self, sample_rna, sample_atac):
        """Test parsing region coordinates."""
        mdata = ds.pp.create_mudata(sample_rna, sample_atac)

        # Remove existing coordinates
        del mdata.mod["atac"].var["chromosome"]
        del mdata.mod["atac"].var["start"]
        del mdata.mod["atac"].var["end"]

        ds.pp.parse_region_coordinates(mdata)

        assert "chromosome" in mdata.mod["atac"].var.columns
        assert "start" in mdata.mod["atac"].var.columns
        assert "end" in mdata.mod["atac"].var.columns

        # Check first region
        assert mdata.mod["atac"].var.iloc[0]["chromosome"] == "chr1"
        assert mdata.mod["atac"].var.iloc[0]["start"] == 0
        assert mdata.mod["atac"].var.iloc[0]["end"] == 640
