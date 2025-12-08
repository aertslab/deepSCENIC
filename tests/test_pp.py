"""Tests for preprocessing functions."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import deepscenic as ds


class TestFilterGenes:
    """Tests for filter_genes function."""

    def test_filter_genes_removes_lowly_expressed(self):
        """Test that genes with few cells are removed."""
        adata = sc.AnnData(np.random.randn(100, 50).astype(np.float32))
        adata.var_names = [f"Gene_{i}" for i in range(50)]

        # Set first 10 genes to be expressed in very few cells
        adata.X[:, :10] = 0
        adata.X[:5, :10] = 1  # Only 5 cells

        original_n_genes = adata.n_vars
        ds.pp.filter_genes(adata, min_cells=10)

        # Should have removed the first 10 genes
        assert adata.n_vars < original_n_genes
        assert adata.n_vars == 40

    def test_filter_genes_inplace_false(self):
        """Test filter_genes with inplace=False returns new object."""
        adata = sc.AnnData(np.random.randn(100, 50).astype(np.float32))
        adata.var_names = [f"Gene_{i}" for i in range(50)]

        result = ds.pp.filter_genes(adata, min_cells=1, inplace=False)

        assert result is not None
        assert result is not adata


class TestNormalizeRNA:
    """Tests for normalize_rna function."""

    def test_normalize_rna_basic(self):
        """Test basic normalization."""
        adata = sc.AnnData(np.abs(np.random.randn(100, 50)).astype(np.float32))
        adata.var_names = [f"Gene_{i}" for i in range(50)]

        ds.pp.normalize_rna(adata, log=True)

        # After log1p, values should be smaller
        assert adata.X.max() < 20  # Reasonable upper bound after log

    def test_normalize_rna_inplace_false(self):
        """Test normalize_rna with inplace=False."""
        adata = sc.AnnData(np.abs(np.random.randn(100, 50)).astype(np.float32))

        result = ds.pp.normalize_rna(adata, inplace=False)

        assert result is not None
        assert result is not adata


class TestComputeR2GPenalty:
    """Tests for compute_r2g_penalty function."""

    def test_compute_r2g_penalty_basic(self):
        """Test basic R2G computation."""
        regions = pd.DataFrame(
            {
                "chromosome": ["chr1"] * 5,
                "start": [0, 10000, 20000, 30000, 40000],
                "end": [640, 10640, 20640, 30640, 40640],
            },
            index=[f"chr1:{i * 10000}-{i * 10000 + 640}" for i in range(5)],
        )

        genes = pd.DataFrame(
            {
                "tss": [5000, 15000, 25000],
                "chromosome": ["chr1"] * 3,
            },
            index=["GeneA", "GeneB", "GeneC"],
        )

        r2g, config = ds.pp.compute_r2g_penalty(
            regions,
            genes,
            max_distance=50000,
            sigma=10000,
        )

        # All should be linked (within 50kb)
        assert r2g.shape == (5, 3)
        assert r2g.nnz > 0
        assert config["n_links"] > 0

    def test_compute_r2g_penalty_no_cross_chromosome(self):
        """Test that R2G doesn't link across chromosomes."""
        regions = pd.DataFrame(
            {
                "chromosome": ["chr1", "chr2"],
                "start": [0, 0],
                "end": [640, 640],
            },
            index=["chr1:0-640", "chr2:0-640"],
        )

        genes = pd.DataFrame(
            {
                "tss": [100, 100],
                "chromosome": ["chr1", "chr2"],
            },
            index=["GeneA", "GeneB"],
        )

        r2g, config = ds.pp.compute_r2g_penalty(regions, genes, max_distance=1000)

        # Should only have 2 links (chr1-chr1 and chr2-chr2), not 4
        assert r2g.nnz == 2

    def test_compute_r2g_penalty_gaussian(self):
        """Test Gaussian penalty values."""
        regions = pd.DataFrame(
            {
                "chromosome": ["chr1"],
                "start": [0],
                "end": [640],
            },
            index=["chr1:0-640"],
        )

        genes = pd.DataFrame(
            {
                "tss": [320],  # Region center
                "chromosome": ["chr1"],
            },
            index=["GeneA"],
        )

        r2g, _ = ds.pp.compute_r2g_penalty(
            regions, genes, max_distance=10000, sigma=1000
        )

        # Distance = 0, so penalty should be ~0
        assert r2g[0, 0] < 0.01


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
