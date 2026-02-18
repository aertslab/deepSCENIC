"""Tests for preprocessing functions."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import deepscenic as ds


class TestMarkDars:
    """Tests for mark_dars function."""

    def test_mark_dars_from_dict(self, sample_atac):
        """Test mark dars from dictionary."""
        dars = {"celltypeA": [f"chr1:{i * 1000}-{i * 1000 + 640}" for i in range(20)]}  # 20/30 regions in sample_atac
        ds.pp.mark_dars(sample_atac, dar_dict=dars)
        assert sample_atac.var["is_dar"].sum() == 20

    def test_mark_dars_from_dir(self, sample_atac, tmp_path):
        """Test marking DARs from directory of BED files."""
        # Create BED files
        astro_bed = tmp_path / "Astro.bed"
        astro_bed.write_text("chr1\t1000\t1640\nchr1\t3000\t3640\n")

        oligo_bed = tmp_path / "Oligo.bed"
        oligo_bed.write_text("chr7\t1000\t1640\nchr7\t2000\t2640\n")

        ds.pp.mark_dars(sample_atac, dar_dir=tmp_path)

        # Check correct regions are marked
        assert sample_atac.var["is_dar"].sum() == 4

    def test_mark_dars_inplace_false(self, sample_atac):
        """Test mark_dars with inplace=False."""
        dar_dict = {"Astro": ["chr1:1000-1640"]}

        # Extract ATAC modality
        result = ds.pp.mark_dars(sample_atac, dar_dict=dar_dict, inplace=False)

        assert result is not None
        assert result is not sample_atac
        assert "is_dar" in result.var.columns

    def test_mark_dars_missing_regions_handled(self, sample_atac):
        """Test that DAR regions not in data are ignored gracefully."""
        dar_dict = {
            "Astro": ["chr1:1000-1640", "chr99:9999-10000"],  # chr99 doesn't exist
        }

        # Extract ATAC modality
        ds.pp.mark_dars(sample_atac, dar_dict=dar_dict)

        # Only chr1:1000-2000 should be marked
        assert sample_atac.var["is_dar"].sum() == 1

    def test_mark_dars_no_source_raises(self, sample_atac):
        """Test error when neither dar_dir nor dar_dict provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            ds.pp.mark_dars(sample_atac)


class TestRemoveZeroVarianceGenes:
    """Tests for remove_zero_variance_genes function."""

    def test_removes_zero_variance(self):
        """Test that zero-variance genes are removed."""
        adata = sc.AnnData(np.random.randn(100, 50).astype(np.float32))
        adata.var_names = [f"Gene_{i}" for i in range(50)]

        # Set first 5 genes to constant value (zero variance)
        assert adata.X is not None
        assert isinstance(adata.X, np.ndarray)
        adata.X[:, :5] = 1.0

        original_n_genes = adata.n_vars
        ds.pp.remove_zero_variance_genes(adata)

        assert adata.n_vars == original_n_genes - 5

    def test_keeps_all_if_no_zero_variance(self):
        """Test that all genes kept if none have zero variance."""
        adata = sc.AnnData(np.random.randn(100, 50).astype(np.float32))
        adata.var_names = [f"Gene_{i}" for i in range(50)]

        ds.pp.remove_zero_variance_genes(adata)

        assert adata.n_vars == 50

    def test_inplace_false(self):
        """Test inplace=False returns new object."""
        adata = sc.AnnData(np.random.randn(100, 50).astype(np.float32))
        assert adata.X is not None
        assert isinstance(adata.X, np.ndarray)
        adata.X[:, :5] = 1.0  # Zero variance

        result = ds.pp.remove_zero_variance_genes(adata, inplace=False)

        assert result is not None
        assert result is not adata
        assert result.n_vars == 45
        assert adata.n_vars == 50


class TestComputeR2GPenalty:
    """Tests for compute_r2g_penalty function."""

    @pytest.fixture
    def mdata_with_regions(self, sample_rna, sample_atac):
        """Create MuData with parsed region coordinates."""
        # create_mudata automatically parses region coordinates
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)
        return mdata

    def test_compute_r2g_penalty_basic(self, mdata_with_regions):
        """Test basic R2G computation stores in uns."""
        genes = pd.DataFrame(
            {"tss": [1000, 5000, 6000], "chromosome": ["chr1"] * 3, "genename": ["Gene_1", "Gene_2", "Gene_3"]},
        )
        genes.set_index("genename", inplace=True)

        ds.pp.compute_r2g_penalty(
            mdata_with_regions,
            genes,
            max_distance=50000,
            sigma=10000,
            filter_to_rna_genes=True,
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
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        # Remove parsed coordinates
        del mdata.mod["atac"].var["chromosome"]

        genes = pd.DataFrame(
            {"tss": [100], "chromosome": ["chr1"], "genenames": ["GeneA"]},
        )
        genes.set_index("genenames", inplace=True)

        with pytest.raises(ValueError, match="create_mudata"):
            ds.pp.compute_r2g_penalty(mdata, genes)

    def test_compute_r2g_penalty_copy(self, mdata_with_regions):
        """Test inplace=False returns new MuData."""
        genes = pd.DataFrame(
            {"tss": [5000], "chromosome": ["chr1"], "genenames": ["GeneA"]},
        )
        genes.set_index("genenames", inplace=True)
        result = ds.pp.compute_r2g_penalty(mdata_with_regions, genes, inplace=False, filter_to_rna_genes=False)

        assert result is not None
        assert result is not mdata_with_regions
        assert "r2g" in result.uns
        assert "r2g" not in mdata_with_regions.uns

    def test_compute_r2g_penalty_key_added(self, mdata_with_regions):
        """Test custom key_added parameter."""
        genes = pd.DataFrame(
            {"tss": [5000], "chromosome": ["chr1"], "genenames": ["GeneA"]},
        )
        genes.set_index("genenames", inplace=True)
        ds.pp.compute_r2g_penalty(mdata_with_regions, genes, key_added="custom_r2g", filter_to_rna_genes=False)

        assert "custom_r2g" in mdata_with_regions.uns
        assert "r2g" not in mdata_with_regions.uns

    def test_compute_r2g_penalty_no_cross_chromosome(self, sample_rna):
        """Test that R2G doesn't link across chromosomes."""
        # Create ATAC with 2 chromosomes
        n_cells = sample_rna.n_obs
        atac = sc.AnnData(np.random.rand(n_cells, 2))
        atac.obs_names = sample_rna.obs_names
        atac.var_names = ["chr1:0-640", "chr2:0-640"]

        # Create RNA with genes that match the annotation
        rna = sc.AnnData(np.random.randn(n_cells, 2))
        rna.obs_names = sample_rna.obs_names
        rna.var_names = ["GeneA", "GeneB"]

        mdata = ds.pp.create_mudata(rna=rna, atac=atac)

        genes = pd.DataFrame(
            {
                "tss": [100, 100],
                "chromosome": ["chr1", "chr2"],
                "genenames": ["GeneA", "GeneB"],
            },
        )
        genes.set_index("genenames", inplace=True)

        ds.pp.compute_r2g_penalty(mdata, genes, max_distance=1000, filter_to_rna_genes=False)

        # Should only have 2 links (chr1-chr1 and chr2-chr2), not 4
        assert mdata.uns["r2g"]["matrix"].nnz == 2
        assert mdata.uns["r2g"]["config"]["n_links"] == 2

    def test_compute_r2g_penalty_gaussian(self, sample_rna):
        """Test Gaussian penalty values."""
        # Create ATAC with one close region and one far region
        n_cells = sample_rna.n_obs
        atac = sc.AnnData(np.random.rand(n_cells, 2))
        atac.obs_names = sample_rna.obs_names
        atac.var_names = ["chr1:0-640", "chr1: 999000-999640"]

        # Create RNA with the gene we're testing
        rna = sc.AnnData(np.random.randn(n_cells, 1))
        rna.obs_names = sample_rna.obs_names
        rna.var_names = ["geneA"]

        mdata = ds.pp.create_mudata(rna=rna, atac=atac)

        genes = pd.DataFrame(
            {
                "tss": [320],  # Region center
                "chromosome": ["chr1"],
                "genenames": ["geneA"],
            },
        )
        genes.set_index("genenames", inplace=True)

        ds.pp.compute_r2g_penalty(mdata, genes, max_distance=1000000, sigma=100000, filter_to_rna_genes=False)

        # Distance = 0, so penalty should be ~0
        assert mdata.uns["r2g"]["matrix"][0, 0] < 0.01

        # distance = 1Mb, so penalty should be ~1
        assert mdata.uns["r2g"]["matrix"][1, 0] > 0.99

    def test_compute_r2g_penalty_excludes_distant_links(self, sample_rna):
        """Test that region-gene pairs beyond max_distance are NOT stored in sparse matrix.

        This is important because:
        - Stored link with penalty ≈ 0 (very close) returns 0 when accessed
        - Non-stored link (too far) ALSO returns 0 when accessed
        So we must check the sparse structure (indices), not just values.
        """
        n_cells = sample_rna.n_obs
        # Create 3 regions at different distances from gene at position 50000
        # Region 0: center at 320 -> distance = 49680 (within 100k)
        # Region 1: center at 100320 -> distance = 50320 (within 100k)
        # Region 2: center at 200320 -> distance = 150320 (BEYOND 100k)
        atac = sc.AnnData(np.random.rand(n_cells, 3))
        atac.obs_names = sample_rna.obs_names
        atac.var_names = ["chr1:0-640", "chr1:100000-100640", "chr1:200000-200640"]

        # Create RNA with the gene we're testing
        rna = sc.AnnData(np.random.randn(n_cells, 1))
        rna.obs_names = sample_rna.obs_names
        rna.var_names = ["TestGene"]

        mdata = ds.pp.create_mudata(rna=rna, atac=atac)

        genes = pd.DataFrame(
            {
                "tss": [50000],
                "chromosome": ["chr1"],
                "genename": ["TestGene"],
            },
        )
        genes.set_index("genename", inplace=True)

        # Use max_distance=100000 so region 2 is excluded
        ds.pp.compute_r2g_penalty(mdata, genes, max_distance=100000, filter_to_rna_genes=False)

        r2g = mdata.uns["r2g"]["matrix"]

        # Should only have 2 links stored (regions 0 and 1), not 3
        assert r2g.nnz == 2, f"Expected 2 stored links, got {r2g.nnz}"

        # Check sparse structure - convert to COO to inspect indices
        coo = r2g.tocoo()
        stored_region_indices = set(coo.row)

        # Regions 0 and 1 should be stored
        assert 0 in stored_region_indices, "Region 0 (within max_distance) should be stored"
        assert 1 in stored_region_indices, "Region 1 (within max_distance) should be stored"

        # Region 2 should NOT be stored (beyond max_distance)
        assert 2 not in stored_region_indices, "Region 2 (beyond max_distance) should NOT be stored"

        # Note: accessing r2g[2, 0] would return 0, but that's misleading!
        # The 0 means "not stored" not "zero penalty". This test verifies the distinction.

    def test_compute_r2g_penalty_filter_to_rna_genes_default(self, sample_rna):
        """Test that filter_to_rna_genes=True (default) filters annotation to RNA genes."""
        n_cells = sample_rna.n_obs
        atac = sc.AnnData(np.random.rand(n_cells, 3))
        atac.obs_names = sample_rna.obs_names
        atac.var_names = ["chr1:0-640", "chr1:1000-1640", "chr1:2000-2640"]

        mdata = ds.pp.create_mudata(rna=sample_rna, atac=atac)

        # Annotation has 4 genes, but only 2 are in RNA data (Gene_0, Gene_1)
        # sample_rna has genes like Gene_0, Gene_1, ..., Gene_49
        genes = pd.DataFrame(
            {
                "tss": [320, 1320, 2320, 3320],
                "chromosome": ["chr1", "chr1", "chr1", "chr1"],
                "genenames": ["Gene_0", "Gene_1", "NotInRNA_A", "NotInRNA_B"],
            },
        )
        genes.set_index("genenames", inplace=True)
        ds.pp.compute_r2g_penalty(mdata, genes, max_distance=50000)

        # Legacy behavior: TFs are kept (10), non-TFs only if annotated (2)
        # sample_rna has 10 TFs (Gene_0-9), but only Gene_0 and Gene_1 have annotation
        # Non-TF genes without annotation are removed
        n_tfs = 10  # All TFs kept
        n_annotated_non_tfs = 0  # Gene_0, Gene_1 are TFs, no non-TF genes with annotation
        expected_genes = n_tfs + n_annotated_non_tfs
        assert len(mdata.uns["r2g"]["gene_names"]) == expected_genes

        # Genes in RNA with annotation should be in r2g
        assert "Gene_0" in mdata.uns["r2g"]["gene_names"]
        assert "Gene_1" in mdata.uns["r2g"]["gene_names"]

        # Genes NOT in RNA should NOT be in r2g (even if in annotation)
        assert "NotInRNA_A" not in mdata.uns["r2g"]["gene_names"]
        assert "NotInRNA_B" not in mdata.uns["r2g"]["gene_names"]

        # 6 links should be stored (3 regions × 2 annotated genes within distance)
        assert mdata.uns["r2g"]["matrix"].nnz == 6

    def test_compute_r2g_penalty_reorders_rna_genes(self, sample_rna):
        """Test that non-TF genes without annotation are removed (legacy behavior)."""
        n_cells = sample_rna.n_obs
        atac = sc.AnnData(np.random.rand(n_cells, 3))
        atac.obs_names = sample_rna.obs_names
        atac.var_names = ["chr1:0-640", "chr1:1000-1640", "chr1:2000-2640"]

        mdata = ds.pp.create_mudata(rna=sample_rna, atac=atac)

        # Remove is_tf column to test behavior without TFs
        del mdata.mod["rna"].var["is_tf"]

        # Annotation only covers 2 of the 50 genes in sample_rna
        genes = pd.DataFrame(
            {
                "tss": [320, 1320],
                "chromosome": ["chr1", "chr1"],
                "genenames": ["Gene_0", "Gene_1"],
            },
        )
        genes.set_index("genenames", inplace=True)

        ds.pp.compute_r2g_penalty(mdata, genes, max_distance=50000)

        # Legacy behavior: Without TFs, only genes with annotation are kept
        assert mdata.mod["rna"].n_vars == 2

        # Only annotated genes should be present
        assert set(mdata.mod["rna"].var_names) == {"Gene_0", "Gene_1"}

        # R2G gene_names should match RNA var_names exactly
        assert mdata.uns["r2g"]["gene_names"] == list(mdata.mod["rna"].var_names)

        # 6 links stored (3 regions × 2 annotated genes within distance)
        assert mdata.uns["r2g"]["matrix"].nnz == 6

    def test_compute_r2g_penalty_no_matching_genes_raises(self, sample_rna):
        """Test error when no annotation genes match RNA genes."""
        n_cells = sample_rna.n_obs
        atac = sc.AnnData(np.random.rand(n_cells, 1))
        atac.obs_names = sample_rna.obs_names
        atac.var_names = ["chr1:0-640"]

        mdata = ds.pp.create_mudata(rna=sample_rna, atac=atac)

        # Annotation has no genes that match RNA var_names
        genes = pd.DataFrame(
            {
                "tss": [320],
                "chromosome": ["chr1"],
                "genenames": ["CompletelyDifferentGeneName"],
            },
        )
        genes.set_index("genenames", inplace=False)

        with pytest.raises(ValueError, match="No genes from annotation found"):
            ds.pp.compute_r2g_penalty(mdata, genes)

    def test_compute_r2g_penalty_reorders_tfs_first(self, sample_rna):
        """Test that TFs come first and are kept even without annotation (legacy behavior)."""
        n_cells = sample_rna.n_obs
        atac = sc.AnnData(np.random.rand(n_cells, 3))
        atac.obs_names = sample_rna.obs_names
        atac.var_names = ["chr1:0-640", "chr1:1000-1640", "chr1:2000-2640"]

        mdata = ds.pp.create_mudata(rna=sample_rna, atac=atac)

        # Mark some genes as TFs (including ones that won't have annotation)
        # sample_rna has genes Gene_0, Gene_1, ..., Gene_49
        # Mark Gene_0, Gene_1, Gene_2 as TFs
        mdata.mod["rna"].var["is_tf"] = False
        mdata.mod["rna"].var.loc[["Gene_0", "Gene_1", "Gene_2"], "is_tf"] = True

        # Annotation only covers Gene_0 and Gene_1 (not Gene_2)
        # Gene_2 is a TF but has no annotation
        genes = pd.DataFrame(
            {
                "tss": [320, 1320],
                "chromosome": ["chr1", "chr1"],
                "genenames": ["Gene_0", "Gene_1"],
            },
        )
        genes.set_index("genenames", inplace=True)

        ds.pp.compute_r2g_penalty(mdata, genes, max_distance=50000)

        # Legacy behavior: TFs are kept (even without annotation), non-TFs only if annotated
        # 3 TFs + 0 non-TF genes with annotation = 3 genes
        assert mdata.mod["rna"].n_vars == 3

        # All TFs should be present and come first
        assert "Gene_0" in mdata.mod["rna"].var_names
        assert "Gene_1" in mdata.mod["rna"].var_names
        assert "Gene_2" in mdata.mod["rna"].var_names

        # TFs should be at the beginning (first 3 positions)
        tf_genes = list(mdata.mod["rna"].var_names[:3])
        assert set(tf_genes) == {"Gene_0", "Gene_1", "Gene_2"}

        # R2G gene_names should include TFs (even without annotation)
        assert "Gene_0" in mdata.uns["r2g"]["gene_names"]
        assert "Gene_1" in mdata.uns["r2g"]["gene_names"]
        assert "Gene_2" in mdata.uns["r2g"]["gene_names"]  # TF without annotation still in r2g
        assert len(mdata.uns["r2g"]["gene_names"]) == 3

        # tf_order should be stored in uns
        assert "tf_order" in mdata.mod["rna"].uns
        assert set(mdata.mod["rna"].uns["tf_order"]) == {"Gene_0", "Gene_1", "Gene_2"}

        # 6 links stored (3 regions × 2 annotated genes within distance)
        assert mdata.uns["r2g"]["matrix"].nnz == 6


class TestCreateMuData:
    """Tests for create_mudata function."""

    def test_create_mudata_basic(self, sample_rna, sample_atac):
        """Test basic MuData creation."""
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        assert "rna" in mdata.mod
        assert "atac" in mdata.mod
        assert mdata.n_obs == 100

    def test_create_mudata_mismatched_cells(self):
        """Test error on mismatched cell names."""
        rna = sc.AnnData(np.random.randn(10, 20))
        rna.obs_names = [f"Cell_{i}" for i in range(10)]

        atac = sc.AnnData(np.random.rand(10, 15))
        atac.obs_names = [f"DifferentCell_{i}" for i in range(10)]

        with pytest.raises(ValueError, match="Observation names must match"):
            ds.pp.create_mudata(rna=rna, atac=atac)


class TestMarkTFs:
    """Tests for mark_tfs function."""

    def test_mark_tfs_from_list(self, sample_rna):
        """Test marking TFs from a list."""
        # Remove existing TF marks
        del sample_rna.var["is_tf"]

        tf_list = ["Gene_0", "Gene_1", "Gene_2", "NonExistent"]
        ds.pp.mark_tfs(sample_rna, tf_list)

        # Check TFs are marked
        assert sample_rna.var["is_tf"].sum() == 3
        # Verify the correct genes are marked
        tf_names = sample_rna.var_names[sample_rna.var["is_tf"]].tolist()
        assert "Gene_0" in tf_names
        assert "Gene_1" in tf_names
        assert "Gene_2" in tf_names
        assert "NonExistent" not in tf_names


class TestSplitCells:
    """Tests for split_cells function."""

    def test_split_cells_basic(self, sample_rna, sample_atac):
        """Test basic cell splitting."""
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        ds.pp.split_cells(mdata, test_fraction=0.2, seed=42)

        assert "split" in mdata.obs.columns
        train_count = (mdata.obs["split"] == "train").sum()
        test_count = (mdata.obs["split"] == "test").sum()

        assert train_count == 80
        assert test_count == 20

    def test_split_cells_reproducible(self, sample_rna, sample_atac):
        """Test that splitting is reproducible with same seed."""
        mdata1 = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)
        mdata2 = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        ds.pp.split_cells(mdata1, test_fraction=0.2, seed=42)
        ds.pp.split_cells(mdata2, test_fraction=0.2, seed=42)

        assert mdata1.obs["split"].equals(mdata2.obs["split"])

    def test_different_seeds_different_split(self, sample_rna, sample_atac):
        """Test that seeding the function differently leads to different splits."""
        mdata1 = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)
        mdata2 = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        ds.pp.split_cells(mdata1, test_fraction=0.2, seed=1)
        ds.pp.split_cells(mdata2, test_fraction=0.2, seed=2)

        assert not (mdata1.obs["split"].equals(mdata2.obs["split"]))


class TestSplitFeaturesByChromosome:
    """Tests for split_features_by_chromosome function."""

    def test_split_features_basic(self, sample_rna, sample_atac):
        """Test basic feature splitting by chromosome."""
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        # Remove existing split columns to test fresh
        del mdata.mod["atac"].var["split"]
        del mdata.mod["rna"].var["split"]

        ds.pp.split_features_by_chromosome(mdata, test_chromosomes=["chr7"])

        # Check ATAC splits: 20 chr1 (train), 10 chr7 (test)
        atac_train = (mdata.mod["atac"].var["split"] == "train").sum()
        atac_test = (mdata.mod["atac"].var["split"] == "test").sum()
        assert atac_train == 20
        assert atac_test == 10

        # Check RNA splits — all genes (including TFs) split by chromosome:
        # - Genes 0-9 (10): TFs on chr1 -> train
        # - Genes 10-24 (15) chr1: train
        # - Genes 25-29 (5) NA: train (unannotated default)
        # - Genes 30-39 (10) chr7: test
        # - Genes 40-49 (10) chr11: train
        rna_var = mdata.mod["rna"].var
        rna_train = (rna_var["split"] == "train").sum()
        rna_test = (rna_var["split"] == "test").sum()

        assert rna_train == 40  # 10 (TFs chr1) + 15 (chr1 non-TF) + 5 (NA) + 10 (chr11)
        assert rna_test == 10  # chr7 only
        assert "both" not in rna_var["split"].cat.categories

    def test_split_features_multiple_test_chromosomes(self, sample_rna, sample_atac):
        """Test splitting with multiple test chromosomes."""
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        del mdata.mod["atac"].var["split"]
        del mdata.mod["rna"].var["split"]

        ds.pp.split_features_by_chromosome(mdata, test_chromosomes=["chr7", "chr11"])

        # RNA: chr7 (10) + chr11 (10) = 20 test genes (TFs are all on chr1, so train)
        rna_test = (mdata.mod["rna"].var["split"] == "test").sum()
        assert rna_test == 20

    def test_split_features_inplace_false(self, sample_rna, sample_atac):
        """Test inplace=False returns a copy."""
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        del mdata.mod["atac"].var["split"]
        del mdata.mod["rna"].var["split"]

        result = ds.pp.split_features_by_chromosome(mdata, test_chromosomes=["chr7"], inplace=False)

        assert result is not None
        assert result is not mdata
        assert "split" in result.mod["atac"].var.columns
        assert "split" in result.mod["rna"].var.columns
        assert "split" not in mdata.mod["atac"].var.columns

    def test_split_features_missing_chromosome_raises(self, sample_rna, sample_atac):
        """Test error when RNA lacks chromosome annotation."""
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        # Remove chromosome column from RNA
        del mdata.mod["rna"].var["chromosome"]

        with pytest.raises(ValueError, match="chromosome"):
            ds.pp.split_features_by_chromosome(mdata, test_chromosomes=["chr7"])

    def test_split_features_with_na_chromosomes(self, sample_rna, sample_atac):
        """Test that non-TF genes without chromosome annotation are assigned to train."""
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        # Remove existing split columns
        del mdata.mod["atac"].var["split"]
        del mdata.mod["rna"].var["split"]

        # Set some chromosomes to NA for NON-TF genes (genes 10-14)
        # Note: genes 0-9 are TFs on chr1, they get "train" (chr1 is train chromosome)
        na_genes = mdata.mod["rna"].var_names[10:15]
        mdata.mod["rna"].var.loc[na_genes, "chromosome"] = pd.NA

        # Should not raise an error
        ds.pp.split_features_by_chromosome(mdata, test_chromosomes=["chr7"])

        # NA chromosomes should be assigned to train (not test) for non-TFs
        assert (mdata.mod["rna"].var.loc[na_genes, "split"] == "train").all()

        # chr7 genes (30-39) are non-TFs, should be split="test"
        chr7_genes = mdata.mod["rna"].var[mdata.mod["rna"].var["chromosome"] == "chr7"].index
        assert (mdata.mod["rna"].var.loc[chr7_genes, "split"] == "test").all()

    def test_split_features_tfs_follow_chromosome_split(self, sample_rna, sample_atac):
        """Test that TFs are split by chromosome like all other genes."""
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        del mdata.mod["atac"].var["split"]
        del mdata.mod["rna"].var["split"]

        # Mark gene 30 (on chr7) as a TF to test TF on test chromosome
        mdata.mod["rna"].var.loc["Gene_30", "is_tf"] = True

        ds.pp.split_features_by_chromosome(mdata, test_chromosomes=["chr7"])

        # Gene_30 is on chr7 (test) and IS a TF -> split="test"
        assert mdata.mod["rna"].var.loc["Gene_30", "split"] == "test"

        # Gene_31 is on chr7 (test) and is NOT a TF -> split="test"
        assert mdata.mod["rna"].var.loc["Gene_31", "split"] == "test"

        # Gene_0 is on chr1 (train) and is a TF -> split="train"
        assert mdata.mod["rna"].var.loc["Gene_0", "split"] == "train"

        # Gene_40 is on chr11 (train) and is NOT a TF -> split="train"
        assert mdata.mod["rna"].var.loc["Gene_40", "split"] == "train"

        # No "both" category
        assert "both" not in mdata.mod["rna"].var["split"].cat.categories


class TestFilterRegionsByCelltype:
    """Tests for filter_regions_by_celltype function."""

    def test_filter_regions_basic(self, sample_atac):
        """Test basic filtering keeps regions present in enough cells."""
        # sample_atac has random values 0-1, most should be > 0
        original_n_regions = sample_atac.n_vars

        ds.pp.filter_regions_by_celltype(sample_atac, celltype_key="celltype", min_fraction=0.1)

        # Most regions should be kept since random values are mostly nonzero
        assert sample_atac.n_vars == original_n_regions

    def test_filter_regions_removes_sparse_regions(self):
        """Test that regions with few nonzero values are removed."""
        n_cells = 100
        n_regions = 10

        adata = sc.AnnData(np.zeros((n_cells, n_regions), dtype=np.float32))
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
        adata.var_names = [f"region_{i}" for i in range(n_regions)]
        adata.obs["celltype"] = ["TypeA"] * 50 + ["TypeB"] * 50

        # Region 0-4: present in 20% of TypeA cells (should be kept at min_fraction=0.1)
        assert adata.X is not None
        adata.X[:10, :5] = 1.0  # 10 cells out of 50 TypeA = 20% # type: ignore[index]

        # Region 5-9: present in only 8% of cells (should be removed at min_fraction=0.1)
        adata.X[:4, 5:] = 1.0  # 4 cells out of 50 = 8% # type: ignore[index]

        ds.pp.filter_regions_by_celltype(adata, celltype_key="celltype", min_fraction=0.1)

        # Only first 5 regions should remain
        assert adata.n_vars == 5
        assert all(f"region_{i}" in adata.var_names for i in range(5))

    def test_filter_regions_keeps_if_any_celltype_passes(self):
        """Test that region is kept if ANY cell type passes threshold."""
        n_cells = 100
        n_regions = 3

        adata = sc.AnnData(np.zeros((n_cells, n_regions), dtype=np.float32))
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
        adata.var_names = ["region_A", "region_B", "region_C"]
        adata.obs["celltype"] = ["TypeA"] * 50 + ["TypeB"] * 50

        assert adata.X is not None
        # region_A: passes in TypeA only
        adata.X[:10, 0] = 1.0  # 10/50 = 20% in TypeA # type: ignore[index]

        # region_B: passes in TypeB only
        adata.X[50:60, 1] = 1.0  # 10/50 = 20% in TypeB # type: ignore[index]

        # region_C: fails in both
        adata.X[0, 2] = 1.0  # 1/50 = 2% in TypeA # type: ignore[index]

        ds.pp.filter_regions_by_celltype(adata, celltype_key="celltype", min_fraction=0.1)

        # region_A and region_B should be kept, region_C removed
        assert adata.n_vars == 2
        assert "region_A" in adata.var_names
        assert "region_B" in adata.var_names
        assert "region_C" not in adata.var_names

    def test_filter_regions_inplace_false(self, sample_atac):
        """Test inplace=False returns a copy."""
        original_n_vars = sample_atac.n_vars

        result = ds.pp.filter_regions_by_celltype(sample_atac, celltype_key="celltype", min_fraction=0.1, inplace=False)

        assert result is not None
        assert result is not sample_atac
        # Original should be unchanged
        assert sample_atac.n_vars == original_n_vars


class TestCreateMuDataParsesCoordinates:
    """Tests that create_mudata automatically parses region coordinates."""

    def test_create_mudata_parses_coordinates(self, sample_rna, sample_atac):
        """Test that create_mudata automatically parses region coordinates."""
        mdata = ds.pp.create_mudata(rna=sample_rna, atac=sample_atac)

        # Coordinates should be automatically parsed
        assert "chromosome" in mdata.mod["atac"].var.columns
        assert "start" in mdata.mod["atac"].var.columns
        assert "end" in mdata.mod["atac"].var.columns

        # Check first region (sample_atac has regions like "chr1:0-640")
        assert mdata.mod["atac"].var.iloc[0]["chromosome"] == "chr1"
        assert mdata.mod["atac"].var.iloc[0]["start"] == 0
        assert mdata.mod["atac"].var.iloc[0]["end"] == 640


class TestAddGeneAnnotation:
    """Tests for add_gene_annotation function."""

    def test_add_gene_annotation_basic(self):
        """Test basic gene annotation addition."""
        adata = sc.AnnData(np.random.randn(10, 5).astype(np.float32))
        adata.var_names = ["Gene_A", "Gene_B", "Gene_C", "Gene_D", "Gene_E"]

        annotation = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr2", "chr3", "chr4", "chr5"],
                "Transcription_Start_Site": [1000, 2000, 3000, 4000, 5000],
                "genenames": ["Gene_A", "Gene_B", "Gene_C", "Gene_D", "Gene_E"],
            },
        )
        annotation.set_index("genenames", inplace=True)

        ds.pp.add_gene_annotation(adata, annotation)

        assert "chromosome" in adata.var.columns
        assert "tss" in adata.var.columns
        assert adata.var.loc["Gene_A", "chromosome"] == "chr1"  # type: ignore
        assert adata.var.loc["Gene_C", "tss"] == 3000  # type: ignore

    def test_add_gene_annotation_partial_match(self):
        """Test when annotation has more genes than data."""
        adata = sc.AnnData(np.random.randn(10, 3).astype(np.float32))
        adata.var_names = ["Gene_A", "Gene_B", "Gene_C"]

        annotation = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr2", "chr3", "chr4", "chr5"],
                "Transcription_Start_Site": [1000, 2000, 3000, 4000, 5000],
                "genenames": ["Gene_A", "Gene_B", "Gene_C", "Gene_D", "Gene_E"],
            },
        )
        annotation.set_index("genenames", inplace=True)

        ds.pp.add_gene_annotation(adata, annotation)

        # All 3 genes should have annotations
        assert adata.var["chromosome"].notna().sum() == 3

    def test_add_gene_annotation_missing_genes(self):
        """Test when data has genes not in annotation."""
        adata = sc.AnnData(np.random.randn(10, 5).astype(np.float32))
        adata.var_names = ["Gene_A", "Gene_B", "Gene_C", "Unknown_1", "Unknown_2"]

        annotation = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr2", "chr3"],
                "Transcription_Start_Site": [1000, 2000, 3000],
                "genenames": ["Gene_A", "Gene_B", "Gene_C"],
            },
        )
        annotation.set_index("genenames", inplace=True)

        ds.pp.add_gene_annotation(adata, annotation)

        # Only 3 genes should have annotations
        assert adata.var["chromosome"].notna().sum() == 3
        assert pd.isna(adata.var.loc["Unknown_1", "chromosome"])  # type: ignore

    def test_add_gene_annotation_no_match_raises(self):
        """Test error when no genes match."""
        adata = sc.AnnData(np.random.randn(10, 3).astype(np.float32))
        adata.var_names = ["Gene_X", "Gene_Y", "Gene_Z"]

        annotation = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr2"],
                "Transcription_Start_Site": [1000, 2000],
                "genenames": ["Gene_A", "Gene_B"],
            },
        )
        annotation.set_index("genenames", inplace=True)

        with pytest.raises(ValueError, match="No genes from annotation"):
            ds.pp.add_gene_annotation(adata, annotation)

    def test_add_gene_annotation_custom_columns(self):
        """Test adding custom columns."""
        adata = sc.AnnData(np.random.randn(10, 3).astype(np.float32))
        adata.var_names = ["Gene_A", "Gene_B", "Gene_C"]

        annotation = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr2", "chr3"],
                "Transcription_Start_Site": [1000, 2000, 3000],
                "Start": [900, 1900, 2900],
                "End": [1100, 2100, 3100],
                "Strand": ["+", "-", "+"],
                "genenames": ["Gene_A", "Gene_B", "Gene_C"],
            },
        )
        annotation.set_index("genenames", inplace=True)

        ds.pp.add_gene_annotation(adata, annotation, columns=["chromosome", "tss", "start", "end", "strand"])

        assert "chromosome" in adata.var.columns
        assert "tss" in adata.var.columns
        assert "start" in adata.var.columns
        assert "end" in adata.var.columns
        assert "strand" in adata.var.columns

    def test_add_gene_annotation_inplace_false(self):
        """Test inplace=False returns copy."""
        adata = sc.AnnData(np.random.randn(10, 3).astype(np.float32))
        adata.var_names = ["Gene_A", "Gene_B", "Gene_C"]

        annotation = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr2", "chr3"],
                "Transcription_Start_Site": [1000, 2000, 3000],
                "genenames": ["Gene_A", "Gene_B", "Gene_C"],
            },
        )
        annotation.set_index("genenames", inplace=True)

        result = ds.pp.add_gene_annotation(adata, annotation, inplace=False)

        assert result is not None
        assert result is not adata
        assert "chromosome" in result.var.columns
        assert "chromosome" not in adata.var.columns

    def test_add_gene_annotation_unknown_column_raises(self):
        """Test error for unknown column name."""
        adata = sc.AnnData(np.random.randn(10, 3).astype(np.float32))
        adata.var_names = ["Gene_A", "Gene_B", "Gene_C"]

        annotation = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr2", "chr3"],
                "Transcription_Start_Site": [1000, 2000, 3000],
                "genenames": ["Gene_A", "Gene_B", "Gene_C"],
            },
        )
        annotation.set_index("genenames", inplace=True)

        with pytest.raises(ValueError, match="Unknown column"):
            ds.pp.add_gene_annotation(adata, annotation, columns=["unknown_col"])
