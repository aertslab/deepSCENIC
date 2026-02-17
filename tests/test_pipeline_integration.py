"""Integration tests following the tutorial pipeline structure.

These tests verify that the outputs of each pipeline stage work
correctly as inputs to the next stage, using minimal mock data
and models for fast execution.

Pipeline flow:
    Tutorial 1: Data Preparation (preprocessing)
    Tutorial 2: Training
    Tutorial 3: Model Diagnosis (embeddings)
    Tutorial 4: GRN Analysis
    Tutorial 5: Perturbation Analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import deepscenic as ds


class TestPreprocessingPipeline:
    """Test Tutorial 1: Data Preparation flow."""

    def test_mark_tfs_creates_is_tf_column(self, raw_rna_adata, pipeline_tf_list):
        """Verify mark_tfs adds boolean is_tf column."""
        ds.pp.mark_tfs(raw_rna_adata, pipeline_tf_list)

        assert "is_tf" in raw_rna_adata.var.columns
        assert raw_rna_adata.var["is_tf"].dtype == bool
        assert raw_rna_adata.var["is_tf"].sum() == len(pipeline_tf_list)

    def test_add_gene_annotation_creates_required_columns(self, raw_rna_adata, pipeline_gene_annotation):
        """Verify add_gene_annotation adds chromosome and tss."""
        ds.pp.add_gene_annotation(raw_rna_adata, pipeline_gene_annotation)

        assert "chromosome" in raw_rna_adata.var.columns
        assert "tss" in raw_rna_adata.var.columns
        # Check nullable dtypes for h5mu compatibility
        assert raw_rna_adata.var["chromosome"].dtype == "string"
        assert raw_rna_adata.var["tss"].dtype == "Int64"

    def test_create_mudata_parses_region_coordinates(
        self, raw_rna_adata, raw_atac_adata, pipeline_tf_list, pipeline_gene_annotation
    ):
        """Verify create_mudata auto-parses region coordinates."""
        ds.pp.mark_tfs(raw_rna_adata, pipeline_tf_list)
        ds.pp.add_gene_annotation(raw_rna_adata, pipeline_gene_annotation)

        mdata = ds.pp.create_mudata(rna=raw_rna_adata, atac=raw_atac_adata)

        assert "chromosome" in mdata["atac"].var.columns
        assert "start" in mdata["atac"].var.columns
        assert "end" in mdata["atac"].var.columns
        # Check first region was parsed correctly
        assert mdata["atac"].var.iloc[0]["chromosome"] == "chr1"
        assert mdata["atac"].var.iloc[0]["start"] == 0
        assert mdata["atac"].var.iloc[0]["end"] == 640

    def test_split_cells_creates_train_test(self, preprocessed_mdata):
        """Verify split_cells creates cell split."""
        assert "split" in preprocessed_mdata.obs.columns
        assert set(preprocessed_mdata.obs["split"].cat.categories) == {"train", "test"}
        # Check we have cells in both splits
        assert (preprocessed_mdata.obs["split"] == "train").sum() > 0
        assert (preprocessed_mdata.obs["split"] == "test").sum() > 0

    def test_split_features_by_chromosome(self, preprocessed_mdata):
        """Verify feature splits are chromosome-based."""
        # ATAC: strict train/test by chromosome
        assert "split" in preprocessed_mdata["atac"].var.columns
        # chr7 regions should be test
        chr7_mask = preprocessed_mdata["atac"].var["chromosome"] == "chr7"
        assert all(preprocessed_mdata["atac"].var.loc[chr7_mask, "split"] == "test")

        # RNA: TFs should be "both"
        assert "split" in preprocessed_mdata["rna"].var.columns
        tf_mask = preprocessed_mdata["rna"].var["is_tf"]
        assert all(preprocessed_mdata["rna"].var.loc[tf_mask, "split"] == "both")

    def test_compute_r2g_creates_penalty_matrix(self, preprocessed_mdata):
        """Verify R2G computation creates valid penalty matrix."""
        assert "r2g" in preprocessed_mdata.uns
        r2g = preprocessed_mdata.uns["r2g"]

        assert "matrix" in r2g
        assert "config" in r2g
        assert "region_names" in r2g
        assert "gene_names" in r2g

        # Matrix shape should be (n_regions, n_genes_final)
        assert r2g["matrix"].shape[0] == preprocessed_mdata["atac"].n_vars
        # Genes should be reordered (TFs first)
        assert r2g["gene_names"][:5] == ["GENE0", "GENE1", "GENE2", "GENE3", "GENE4"]

    def test_preprocessed_mdata_passes_validation(self, preprocessed_mdata):
        """Verify preprocessed MuData passes schema validation."""
        issues = ds.validate_schema(preprocessed_mdata, mode="training")
        assert len(issues) == 0

    def test_write_read_roundtrip_preserves_schema(self, preprocessed_mdata, tmp_path):
        """Verify I/O roundtrip preserves all required fields."""
        path = tmp_path / "test_mdata.h5mu"

        # Write
        ds.write(preprocessed_mdata, path)
        assert path.exists()

        # Read
        loaded = ds.read(path)

        # Verify key fields preserved
        assert "split" in loaded.obs.columns
        assert "is_tf" in loaded["rna"].var.columns
        assert "r2g" in loaded.uns

        # Verify shapes match
        assert loaded.n_obs == preprocessed_mdata.n_obs
        assert loaded["rna"].n_vars == preprocessed_mdata["rna"].n_vars

        # Verify passes validation
        issues = ds.validate_schema(loaded, mode="training")
        assert len(issues) == 0


class TestTrainingPipeline:
    """Test Tutorial 2: Training flow."""

    def test_register_genome_succeeds(self, tmp_fasta_path):
        """Verify genome registration works with test FASTA."""
        ds.register_genome(tmp_fasta_path)

        genome = ds.get_genome()
        assert genome is not None
        assert "chr1" in genome.chromosomes

    def test_train_with_mock_enformer(self, preprocessed_mdata, tmp_fasta_path):
        """Verify training runs with MockEnformer and preprocessed data."""
        from deepscenic.tl._training_state import ModelConfig
        from tests.conftest import MockEnformer

        ds.register_genome(tmp_fasta_path)

        config = ModelConfig(
            sequence_model=MockEnformer(bottleneck_size=16, emb_len=2),
            seq_len=640,
            bottleneck_size=16,
            emb_len=2,
            n_hidden=8,
        )

        model = ds.tl.train(
            preprocessed_mdata,
            config=config,
            epochs=1,
            batch_size=4,
            seq_batch_size=10,
            device="cpu",
        )

        # Verify model structure
        assert hasattr(model, "vae")
        assert hasattr(model, "motifnet")
        assert hasattr(model, "adj_tf2r")
        assert hasattr(model, "tf_names")
        assert len(model.tf_names) == 5  # 5 TFs

    def test_trained_model_has_correct_dimensions(self, pipeline_trained_model, preprocessed_mdata):
        """Verify trained model dimensions match input data."""
        model = pipeline_trained_model

        # TF count
        assert len(model.tf_names) == preprocessed_mdata["rna"].var["is_tf"].sum()

        # Gene count
        assert len(model.gene_names) == preprocessed_mdata["rna"].n_vars

        # tf2r matrix covers all regions (train + test) for all TFs
        # Note: model.region_names may only include train regions, but adj_tf2r includes all
        assert model.adj_tf2r.shape[0] == preprocessed_mdata["atac"].n_vars  # All regions
        assert model.adj_tf2r.shape[1] == len(model.tf_names)  # All TFs

    def test_model_save_load_roundtrip(self, pipeline_trained_model, tmp_path):
        """Verify model save/load preserves weights and config."""
        from tests.conftest import MockEnformer

        model = pipeline_trained_model
        path = tmp_path / "test_model.pt"

        # Save
        model.save(path)
        assert path.exists()

        # Load (need to provide MockEnformer since it's custom)
        mock_seq = MockEnformer(bottleneck_size=16, emb_len=2)
        loaded = ds.tl.load_model(path, device="cpu", sequence_model=mock_seq)

        # Verify structure preserved
        assert len(loaded.tf_names) == len(model.tf_names)
        assert loaded.tf_names == model.tf_names
        assert loaded.gene_names == model.gene_names

    def test_finetune_r2g_with_trained_model(self, pipeline_trained_model, preprocessed_mdata):
        """Verify r2g finetuning works with trained model."""
        model = pipeline_trained_model

        # Finetune on test cells
        model = ds.tl.finetune_r2g(
            model,
            preprocessed_mdata,
            cell_split="test",
            feature_split="train",
            epochs=1,
        )

        # Model should still be valid after finetuning
        assert hasattr(model, "vae")
        assert hasattr(model, "adj_tf2r")
        # tf2r should still have TFs as second dimension
        assert model.adj_tf2r.shape[1] == len(model.tf_names)


class TestAnalysisPipeline:
    """Test Tutorials 3-4: Model Diagnosis and GRN Analysis flow."""

    def test_to_latent_stores_embeddings_in_obsm(self, pipeline_trained_model, preprocessed_mdata):
        """Verify to_latent stores all embedding types in mdata.obsm (Tutorial 3)."""
        ds.tl.to_latent(
            pipeline_trained_model,
            preprocessed_mdata,
            batch_size=8,
        )

        assert "X_deepscenic_z_tf" in preprocessed_mdata.obsm
        assert "X_deepscenic_enh_act" in preprocessed_mdata.obsm
        assert "X_deepscenic_z_rna" in preprocessed_mdata.obsm

        # Check shapes
        n_cells = preprocessed_mdata.n_obs
        n_tfs = len(pipeline_trained_model.tf_names)
        n_genes = len(pipeline_trained_model.gene_names)

        assert preprocessed_mdata.obsm["X_deepscenic_z_tf"].shape == (n_cells, n_tfs)
        assert preprocessed_mdata.obsm["X_deepscenic_enh_act"].shape[0] == n_cells
        assert preprocessed_mdata.obsm["X_deepscenic_z_rna"].shape == (n_cells, n_genes)

    def test_to_latent_custom_key_prefix(self, pipeline_trained_model, preprocessed_mdata):
        """Verify to_latent respects custom key_prefix parameter."""
        ds.tl.to_latent(
            pipeline_trained_model,
            preprocessed_mdata,
            key_prefix="X_custom_",
            batch_size=8,
        )

        assert "X_custom_z_tf" in preprocessed_mdata.obsm
        assert "X_custom_enh_act" in preprocessed_mdata.obsm

    def test_extract_grn_returns_matrices(self, pipeline_trained_model):
        """Verify extract_grn returns tf2r, r2g DataFrames (Tutorial 4)."""
        grn = ds.tl.extract_grn(pipeline_trained_model)

        assert "tf2r" in grn
        assert "r2g" in grn
        assert isinstance(grn["tf2r"], pd.DataFrame)
        assert isinstance(grn["r2g"], pd.DataFrame)

        # tf2r should have region-TF structure
        assert len(grn["tf2r"]) > 0

        # r2g should be edge list with region, gene, weight
        assert "region" in grn["r2g"].columns or len(grn["r2g"].columns) >= 3

    def test_get_tf_targets(self, pipeline_trained_model):
        """Verify get_tf_targets returns target genes for a TF."""
        tf_name = pipeline_trained_model.tf_names[0]  # First TF

        targets = ds.tl.get_tf_targets(
            pipeline_trained_model,
            tf_name=tf_name,
            top_k=5,
        )

        assert isinstance(targets, pd.DataFrame)
        # May be empty if no strong targets, but should have correct columns
        if len(targets) > 0:
            assert "gene" in targets.columns or "tf2r_weight" in targets.columns

    def test_get_gene_regulators(self, pipeline_trained_model):
        """Verify get_gene_regulators returns TFs for a gene."""
        # Use a non-TF gene
        gene_names = pipeline_trained_model.gene_names
        tf_names = set(pipeline_trained_model.tf_names)
        non_tf_genes = [g for g in gene_names if g not in tf_names]

        if non_tf_genes:
            gene_name = non_tf_genes[0]
            regulators = ds.tl.get_gene_regulators(
                pipeline_trained_model,
                gene_name=gene_name,
                top_k=3,
            )

            assert isinstance(regulators, pd.DataFrame)


class TestPerturbationPipeline:
    """Test Tutorial 5: Perturbation Analysis flow."""

    def test_simulate_perturbation_returns_logfc(self, pipeline_trained_model, preprocessed_mdata):
        """Verify simulate_perturbation returns log fold change array."""
        tf_name = pipeline_trained_model.tf_names[0]

        logFC = ds.tl.simulate_perturbation(
            pipeline_trained_model,
            preprocessed_mdata,
            tf_name=tf_name,
            level=0.0,  # Knockout
            n_iter=2,  # Minimal iterations for speed
            batch_size=8,
        )

        assert isinstance(logFC, np.ndarray)
        assert logFC.shape == (
            preprocessed_mdata.n_obs,
            len(pipeline_trained_model.gene_names),
        )
        # Should not contain NaN/Inf
        assert np.isfinite(logFC).all()

    def test_simulate_perturbation_with_split(self, pipeline_trained_model, preprocessed_mdata):
        """Verify perturbation respects cell split."""
        tf_name = pipeline_trained_model.tf_names[0]

        logFC = ds.tl.simulate_perturbation(
            pipeline_trained_model,
            preprocessed_mdata,
            tf_name=tf_name,
            level=0.0,
            n_iter=2,
            split="test",
            batch_size=8,
        )

        n_test_cells = (preprocessed_mdata.obs["split"] == "test").sum()
        assert logFC.shape[0] == n_test_cells

    def test_simulate_perturbation_intermediate(self, pipeline_trained_model, preprocessed_mdata):
        """Verify return_intermediate returns dict with per-iteration results."""
        tf_name = pipeline_trained_model.tf_names[0]

        logFC_dict = ds.tl.simulate_perturbation(
            pipeline_trained_model,
            preprocessed_mdata,
            tf_name=tf_name,
            level=0.0,
            n_iter=3,
            return_intermediate=True,
            batch_size=8,
        )

        assert isinstance(logFC_dict, dict)
        assert len(logFC_dict) == 3  # 3 iterations
        assert 1 in logFC_dict and 2 in logFC_dict and 3 in logFC_dict

    def test_process_perturbation_results(self, pipeline_trained_model, preprocessed_mdata):
        """Verify process_perturbation_results creates summary DataFrame."""
        tf_name = pipeline_trained_model.tf_names[0]

        # Get logFC
        logFC = ds.tl.simulate_perturbation(
            pipeline_trained_model,
            preprocessed_mdata,
            tf_name=tf_name,
            level=0.0,
            n_iter=2,
            batch_size=8,
        )

        # Process results
        results = ds.tl.process_perturbation_results(
            logFC,
            preprocessed_mdata,
            tf_name=tf_name,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(pipeline_trained_model.gene_names)
        # Should have key columns
        assert "gene" in results.columns or results.index.name == "gene"

    def test_simulate_multi_perturbation(self, pipeline_trained_model, preprocessed_mdata):
        """Verify multi-TF perturbation works."""
        tf_names = pipeline_trained_model.tf_names[:2]  # First 2 TFs

        logFC = ds.tl.simulate_multi_perturbation(
            pipeline_trained_model,
            preprocessed_mdata,
            tf_names=tf_names,
            n_iter=2,
            batch_size=8,
        )

        assert isinstance(logFC, np.ndarray)
        assert logFC.shape == (
            preprocessed_mdata.n_obs,
            len(pipeline_trained_model.gene_names),
        )


class TestFullPipeline:
    """End-to-end test covering Tutorials 1-5."""

    def test_complete_pipeline_from_raw_to_perturbation(self, tmp_path):
        """
        Run complete pipeline from raw AnnData to perturbation results.

        This test verifies that all tutorial steps work together:
        1. Data preparation (preprocessing)
        2. Training (with MockEnformer)
        3. Model diagnosis (embedding extraction)
        4. GRN analysis
        5. Perturbation simulation
        """
        import anndata as ad

        from deepscenic.tl._training_state import ModelConfig
        from tests.conftest import MockEnformer

        # ====================================================================
        # Tutorial 1: Data Preparation
        # ====================================================================

        # Create raw RNA data
        n_cells = 12
        n_genes = 15
        rna = ad.AnnData(
            X=np.random.rand(n_cells, n_genes).astype(np.float32),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
            var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
        )
        rna.obs["celltype"] = ["A"] * 8 + ["B"] * 4

        # Create raw ATAC data
        n_regions = 20
        region_names = [f"chr1:{i * 1000}-{i * 1000 + 640}" for i in range(15)] + [
            f"chr7:{i * 1000}-{i * 1000 + 640}" for i in range(5)
        ]
        atac = ad.AnnData(
            X=np.random.rand(n_cells, n_regions).astype(np.float32),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
            var=pd.DataFrame(index=region_names),
        )

        # TF list and gene annotation
        tf_list = ["G0", "G1", "G2"]  # First 3 genes are TFs
        gene_annot = pd.DataFrame(
            {
                "Chromosome": ["chr1"] * 10 + ["chr7"] * 5,
                "Transcription_Start_Site": [i * 2000 for i in range(n_genes)],
            },
            index=[f"G{i}" for i in range(n_genes)],
        )

        # Preprocessing steps
        ds.pp.mark_tfs(rna, tf_list)
        ds.pp.add_gene_annotation(rna, gene_annot)
        mdata = ds.pp.create_mudata(rna=rna, atac=atac)
        mdata.obs["celltype"] = rna.obs["celltype"].copy()
        ds.pp.split_cells(mdata, test_fraction=0.25)
        ds.pp.split_features_by_chromosome(mdata, test_chromosomes=["chr7"])
        ds.pp.compute_r2g_penalty(mdata, max_distance=30000, sigma=5000)

        # Validate preprocessing
        issues = ds.validate_schema(mdata, mode="training")
        assert len(issues) == 0, f"Preprocessing validation failed: {issues}"

        # I/O roundtrip
        mdata_path = tmp_path / "pipeline_test.h5mu"
        ds.write(mdata, mdata_path)
        mdata = ds.read(mdata_path)

        # ====================================================================
        # Tutorial 2: Training
        # ====================================================================

        # Create test FASTA
        fasta_path = tmp_path / "test.fa"
        with open(fasta_path, "w") as f:
            for chrom in ["chr1", "chr7"]:
                f.write(f">{chrom}\n")
                seq = "".join(np.random.choice(list("ACGT"), 100000))
                for i in range(0, len(seq), 80):
                    f.write(seq[i : i + 80] + "\n")

        ds.register_genome(fasta_path)

        # Train with MockEnformer
        config = ModelConfig(
            sequence_model=MockEnformer(bottleneck_size=16, emb_len=2),
            seq_len=640,
            bottleneck_size=16,
            emb_len=2,
            n_hidden=8,
        )
        model = ds.tl.train(
            mdata,
            config=config,
            epochs=2,
            batch_size=4,
            seq_batch_size=10,
            device="cpu",
        )

        assert len(model.tf_names) == 3  # 3 TFs

        # ====================================================================
        # Tutorial 3: Model Diagnosis
        # ====================================================================

        ds.tl.to_latent(model, mdata, batch_size=8)

        assert "X_deepscenic_z_tf" in mdata.obsm
        assert mdata.obsm["X_deepscenic_z_tf"].shape == (n_cells, 3)  # 3 TFs
        assert np.isfinite(mdata.obsm["X_deepscenic_z_tf"]).all()

        # ====================================================================
        # Tutorial 4: GRN Analysis
        # ====================================================================

        grn = ds.tl.extract_grn(model)
        assert "tf2r" in grn
        assert "r2g" in grn

        # Get TF targets
        targets = ds.tl.get_tf_targets(model, tf_name="G0", top_k=5)
        assert isinstance(targets, pd.DataFrame)

        # ====================================================================
        # Tutorial 5: Perturbation Analysis
        # ====================================================================

        logFC = ds.tl.simulate_perturbation(
            model,
            mdata,
            tf_name="G0",
            level=0.0,
            n_iter=2,
            batch_size=8,
        )

        assert logFC.shape == (n_cells, len(model.gene_names))
        assert np.isfinite(logFC).all()

        results = ds.tl.process_perturbation_results(logFC, mdata, tf_name="G0")
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

        # ====================================================================
        # Success: Complete pipeline ran without errors
        # ====================================================================
