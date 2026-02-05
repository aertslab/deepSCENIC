"""Shared fixtures for deepSCENIC tests."""

from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import torch
from scipy.sparse import csr_matrix

MINIMAL_DIMS = {
    "n_tfs": 5,
    "n_genes": 20,
    "n_regions": 30,
    "n_links": 50,
    "n_cells": 8,
    "n_hidden": 8,
    "bottleneck_size": 16,
    "emb_len": 2,
}


@pytest.fixture
def minimal_dims():
    """Shared minimal dimensions for tests."""
    return MINIMAL_DIMS


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Force all tests to run on CPU by disabling CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield


class MockEnformer(torch.nn.Module):
    """Mock Enformer for testing (avoids loading real model).

    Mimics Enformer's interface by accepting return_only_embeddings parameter
    and returning embeddings with shape (batch, emb_len, bottleneck_size).
    """

    def __init__(self, bottleneck_size: int = 16, emb_len: int = 2):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.emb_len = emb_len
        self.linear = torch.nn.Linear(10, bottleneck_size * emb_len)

    def forward(self, x: torch.Tensor, return_only_embeddings: bool = False) -> torch.Tensor:
        """Return mock embeddings with shape (batch, emb_len, bottleneck_size)."""
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.emb_len, self.bottleneck_size)


@pytest.fixture
def sample_rna():
    """Create sample RNA AnnData."""
    n_cells = 100
    n_genes = 50

    adata = sc.AnnData(np.random.randn(n_cells, n_genes).astype(np.float32))
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    adata.var["is_tf"] = [i < 10 for i in range(n_genes)]

    # Simulate add_gene_annotation behavior: some genes have pd.NA for chromosome/tss
    # Use nullable dtypes for h5mu compatibility
    chromosomes = ["chr1"] * 25 + [pd.NA] * 5 + ["chr7"] * 10 + ["chr11"] * 10
    adata.var["chromosome"] = pd.array(chromosomes, dtype="string")

    # tss column with pd.NA mixed with integers (common case from add_gene_annotation)
    tss_values = [i * 10000 for i in range(25)] + [pd.NA] * 5 + [i * 10000 for i in range(25, 45)]
    adata.var["tss"] = pd.array(tss_values, dtype="Int64")

    # Split column: TFs (genes 0-9) get "both", non-TFs get "train" or "test"
    # Gene layout: chr1 (0-24), NA (25-29), chr7 (30-39), chr11 (40-49)
    # TFs: genes 0-9 (all on chr1)
    # Using chr7 as test chromosome: genes 30-39 should be "test" (non-TFs)
    splits = []
    for i in range(n_genes):
        if i < 10:  # TFs
            splits.append("both")
        elif i >= 30 and i < 40:  # chr7 (test chromosome)
            splits.append("test")
        else:  # chr1, NA, chr11 (train chromosomes)
            splits.append("train")
    adata.var["split"] = pd.Categorical(splits, categories=["train", "test", "both"])

    return adata


@pytest.fixture
def sample_atac():
    """Create sample ATAC AnnData."""
    n_cells = 100
    n_regions = 30

    adata = sc.AnnData(np.random.rand(n_cells, n_regions).astype(np.float32))
    adata.var_names = [f"chr1:{i * 1000}-{i * 1000 + 640}" for i in range(20)] + [
        f"chr7:{i * 1000}-{i * 1000 + 640}" for i in range(10)
    ]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    adata.obs["celltype"] = ["Astrocytes" for _ in range(n_cells - 10)] + [
        "Oligo" for _ in range(10)
    ]  # 90 Astros, 10 Oligos

    adata.var["chromosome"] = ["chr1"] * 20 + ["chr7"] * 10
    adata.var["start"] = [i * 1000 for i in range(20)] + [i * 1000 for i in range(10)]
    adata.var["end"] = [i * 1000 + 640 for i in range(20)] + [i * 1000 + 640 for i in range(10)]
    adata.var["split"] = pd.Categorical(["train"] * 20 + ["test"] * 10, categories=["train", "test"])

    return adata


@pytest.fixture
def sample_mdata(sample_rna, sample_atac):
    """Create sample MuData with full schema."""
    mdata = md.MuData({"rna": sample_rna, "atac": sample_atac})

    mdata.obs["split"] = pd.Categorical(["train"] * 80 + ["test"] * 20, categories=["train", "test"])

    mdata["rna"].uns["log1p"] = {"base": None}
    mdata["rna"].layers["log_norm"] = np.random.rand(sample_rna.n_obs, len(sample_rna.var)).astype(np.float32)  # type: ignore

    mdata.uns["deepscenic_version"] = "0.1.0"
    mdata.uns["r2g"] = {
        "matrix": csr_matrix(np.random.rand(30, 50).astype(np.float32)),
        "config": {"max_distance": 1000000, "sigma": 100000, "method": "gaussian"},
        "region_names": sample_atac.var_names.tolist(),
        "gene_names": sample_rna.var_names.tolist(),
    }

    return mdata


@pytest.fixture
def minimal_mdata():
    """Create minimal MuData without full schema (for testing validation)."""
    rna = sc.AnnData(np.random.randn(10, 20).astype(np.float32))
    rna.var_names = [f"Gene_{i}" for i in range(20)]
    rna.obs_names = [f"Cell_{i}" for i in range(10)]

    atac = sc.AnnData(np.random.rand(10, 15).astype(np.float32))
    atac.var_names = [f"chr1:{i * 1000}-{i * 1000 + 640}" for i in range(15)]
    atac.obs_names = [f"Cell_{i}" for i in range(10)]

    return md.MuData({"rna": rna, "atac": atac})


@pytest.fixture
def mock_vae():
    """Create a minimal VAE for testing."""
    from deepscenic.models import DeepSCENICVAE

    d = MINIMAL_DIMS

    r2g_indices = torch.stack(
        [
            torch.randint(0, d["n_regions"], (d["n_links"],)),
            torch.randint(0, d["n_genes"], (d["n_links"],)),
        ]
    )
    r2g_distances = torch.rand(d["n_links"])
    tf_indices = torch.randperm(d["n_genes"])[: d["n_tfs"]]
    gene_indices = torch.arange(d["n_genes"])
    region_indices = torch.arange(d["n_regions"])

    return DeepSCENICVAE(
        n_tfs=d["n_tfs"],
        n_genes=d["n_genes"],
        n_regions=d["n_regions"],
        r2g_indices=r2g_indices,
        r2g_distances=r2g_distances,
        tf_indices=tf_indices,
        gene_indices=gene_indices,
        region_indices=region_indices,
        n_hidden=d["n_hidden"],
        n_batches=0,
    )


@pytest.fixture
def mock_motifnet():
    """Create a minimal MotifNet for testing."""
    from deepscenic.models import MotifNet

    d = MINIMAL_DIMS
    return MotifNet(
        n_tfs=d["n_tfs"],
        bottleneck_size=d["bottleneck_size"],
        emb_len=d["emb_len"],
    )


@pytest.fixture
def mock_adj_E1():
    """Create a mock E1 adjacency matrix with positive values."""
    d = MINIMAL_DIMS
    return torch.abs(torch.randn(d["n_regions"], d["n_tfs"])) + 0.1


@pytest.fixture
def mock_deepscenic_model(mock_vae, mock_motifnet, mock_adj_E1):
    """Create a complete DeepSCENICModel for testing."""
    from deepscenic.tl._model import DeepSCENICModel
    from deepscenic.tl._training_state import ModelConfig, TrainingHistory

    d = MINIMAL_DIMS
    config = ModelConfig(n_hidden=128)

    # Create sample history
    history = TrainingHistory()
    history.log("train", {"total": 1.0, "rna_recon": 0.5})
    history.log("train", {"total": 0.8, "rna_recon": 0.4})
    history.log("val_cells", {"total": 1.1})
    history.log("val_cells", {"total": 0.9})

    return DeepSCENICModel(
        vae=mock_vae,
        motifnet=mock_motifnet,
        sequence_model=MockEnformer(
            bottleneck_size=d["bottleneck_size"],
            emb_len=d["emb_len"],
        ),
        adj_E1=mock_adj_E1,
        config=config,
        tf_names=[f"TF{i}" for i in range(d["n_tfs"])],
        gene_names=[f"GENE{i}" for i in range(d["n_genes"])],
        region_names=[f"chr1:{i * 100}-{i * 100 + 100}" for i in range(d["n_regions"])],
        history=history,
    )


@pytest.fixture
def tmp_fasta(tmp_path):
    """Create a temporary FASTA file for testing."""
    import random

    import pyfaidx

    fasta_path = tmp_path / "test.fa"

    # Generate a deterministic sequence for reproducible tests
    random.seed(42)
    bases = "ACGT"
    seq = "".join(random.choice(bases) for _ in range(10000))

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i : i + 80] + "\n")
        # Add a second chromosome
        f.write(">chr2\n")
        seq2 = "".join(random.choice(bases) for _ in range(5000))
        for i in range(0, len(seq2), 80):
            f.write(seq2[i : i + 80] + "\n")

    # Create the index file using pyfaidx
    pyfaidx.Fasta(str(fasta_path))

    return fasta_path


@pytest.fixture
def mock_mdata_for_model():
    """Create MuData compatible with mock_deepscenic_model dimensions."""
    d = MINIMAL_DIMS
    n_cells = d["n_cells"] * 2  # 16 cells (8 train, 8 test)

    rna = ad.AnnData(X=np.random.rand(n_cells, d["n_genes"]).astype(np.float32))
    rna.var_names = [f"GENE{i}" for i in range(d["n_genes"])]
    rna.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    # Add feature split: first half train, second half test
    rna.var["split"] = ["train"] * (d["n_genes"] // 2) + ["test"] * (d["n_genes"] - d["n_genes"] // 2)

    atac = ad.AnnData(X=np.random.rand(n_cells, d["n_regions"]).astype(np.float32))
    atac.var_names = [f"chr1:{i * 100}-{i * 100 + 100}" for i in range(d["n_regions"])]
    atac.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    # Add feature split: first half train, second half test
    atac.var["split"] = ["train"] * (d["n_regions"] // 2) + ["test"] * (d["n_regions"] - d["n_regions"] // 2)

    mdata = md.MuData({"rna": rna, "atac": atac})
    mdata.obs["split"] = ["train"] * (n_cells // 2) + ["test"] * (n_cells // 2)

    return mdata


# =============================================================================
# Pipeline Integration Test Fixtures
# =============================================================================


@pytest.fixture
def raw_rna_adata() -> ad.AnnData:
    """Create minimal raw RNA data before preprocessing.

    Returns AnnData with:
    - 16 cells (will be split 80/20)
    - 20 genes (5 will be marked as TFs)
    - Random expression values
    """
    n_cells = 16
    n_genes = 20

    # Random log-normalized expression
    X = np.random.rand(n_cells, n_genes).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]

    # Add celltype for stratified splitting
    adata.obs["celltype"] = ["TypeA"] * 10 + ["TypeB"] * 6

    return adata


@pytest.fixture
def raw_atac_adata() -> ad.AnnData:
    """Create minimal raw ATAC data before preprocessing.

    Returns AnnData with:
    - 16 cells (matching RNA)
    - 30 regions across chr1 (20) and chr7 (10, test chromosome)
    - Region names in chr:start-end format for auto-parsing
    """
    n_cells = 16
    n_regions = 30

    # Random accessibility values
    X = np.random.rand(n_cells, n_regions).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Region names: chr1 (train), chr7 (test)
    # Each region is 640bp (matching default seq_len)
    region_names = []
    for i in range(20):  # chr1 regions
        start = i * 10000
        region_names.append(f"chr1:{start}-{start + 640}")
    for i in range(10):  # chr7 regions (test chromosome)
        start = i * 10000
        region_names.append(f"chr7:{start}-{start + 640}")

    adata.var_names = region_names

    return adata


@pytest.fixture
def pipeline_tf_list() -> list[str]:
    """TF names that match raw_rna_adata genes."""
    # First 5 genes are TFs
    return ["GENE0", "GENE1", "GENE2", "GENE3", "GENE4"]


@pytest.fixture
def pipeline_gene_annotation() -> pd.DataFrame:
    """Gene annotation DataFrame for preprocessing.

    Provides chromosome and TSS for each gene.
    - GENE0-9: chr1 (train genes, TFs in 0-4)
    - GENE10-14: chr7 (test genes)
    - GENE15-19: chr1 (train genes)
    """
    data = {
        "Chromosome": (
            ["chr1"] * 10 +   # GENE0-9 on chr1
            ["chr7"] * 5 +    # GENE10-14 on chr7 (test)
            ["chr1"] * 5      # GENE15-19 on chr1
        ),
        "Transcription_Start_Site": [i * 5000 for i in range(20)],
    }
    df = pd.DataFrame(data)
    df.index = [f"GENE{i}" for i in range(20)]
    return df


@pytest.fixture
def preprocessed_mdata(
    raw_rna_adata: ad.AnnData,
    raw_atac_adata: ad.AnnData,
    pipeline_tf_list: list[str],
    pipeline_gene_annotation: pd.DataFrame,
) -> md.MuData:
    """Create MuData through actual preprocessing pipeline.

    Runs Tutorial 1 preprocessing steps:
    1. mark_tfs
    2. add_gene_annotation
    3. create_mudata
    4. split_cells
    5. split_features_by_chromosome
    6. compute_r2g_penalty
    """
    import deepscenic as ds

    # Step 1: Mark TFs (before create_mudata)
    ds.pp.mark_tfs(raw_rna_adata, pipeline_tf_list)

    # Step 2: Add gene annotation
    ds.pp.add_gene_annotation(raw_rna_adata, pipeline_gene_annotation)

    # Step 3: Create MuData
    mdata = ds.pp.create_mudata(rna=raw_rna_adata, atac=raw_atac_adata)

    # Copy celltype from RNA to MuData for stratified splitting
    mdata.obs["celltype"] = raw_rna_adata.obs["celltype"].copy()

    # Step 4: Split cells
    ds.pp.split_cells(mdata, test_fraction=0.25, stratify_key="celltype")

    # Step 5: Split features by chromosome
    ds.pp.split_features_by_chromosome(mdata, test_chromosomes=["chr7"])

    # Step 6: Compute R2G penalty
    ds.pp.compute_r2g_penalty(mdata, max_distance=50000, sigma=10000)

    return mdata


@pytest.fixture
def tmp_fasta_path(tmp_path: Path) -> Path:
    """Create temporary FASTA file with test sequences.

    Provides sequences for chr1 and chr7 regions used in pipeline tests.
    """
    fasta_path = tmp_path / "test_genome.fa"

    # Generate enough sequence for our test regions
    # Each region is 640bp, we need up to 200000bp per chromosome
    seq_len = 200000
    bases = "ACGT"

    with open(fasta_path, "w") as f:
        for chrom in ["chr1", "chr7"]:
            f.write(f">{chrom}\n")
            # Write sequence in 80-char lines
            seq = "".join(np.random.choice(list(bases), seq_len))
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

    return fasta_path


@pytest.fixture
def pipeline_trained_model(
    preprocessed_mdata: md.MuData,
    tmp_fasta_path: Path,
) -> "DeepSCENICModel":
    """Create trained model through actual training API with MockEnformer.

    Runs Tutorial 2 training with minimal epochs and mock sequence model.
    """
    import deepscenic as ds
    from deepscenic.tl._training_state import ModelConfig

    # Register genome
    ds.register_genome(tmp_fasta_path)

    # Create config with MockEnformer
    config = ModelConfig(
        sequence_model=MockEnformer(bottleneck_size=16, emb_len=2),
        seq_len=640,
        bottleneck_size=16,
        emb_len=2,
        n_hidden=8,  # Small hidden size for speed
    )

    # Train with minimal epochs
    model = ds.tl.train(
        preprocessed_mdata,
        config=config,
        epochs=2,
        batch_size=4,
        seq_batch_size=10,
        lr=1e-3,
        device="cpu",
    )

    return model
