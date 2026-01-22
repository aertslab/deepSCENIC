"""Shared fixtures for deepSCENIC tests."""

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
        use_ppi=False,
        n_batches=0,
    )


@pytest.fixture
def mock_tf2rnet():
    """Create a minimal TF2rNet/MotifNet for testing."""
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
def mock_deepscenic_model(mock_vae, mock_tf2rnet, mock_adj_E1):
    """Create a complete DeepSCENICModel for testing."""
    from deepscenic.tl._model import DeepSCENICModel
    from deepscenic.tl._training_state import TrainingConfig, TrainingHistory

    d = MINIMAL_DIMS
    config = TrainingConfig(epochs=10, batch_size=d["n_cells"])

    # Create sample history
    history = TrainingHistory()
    history.log("train", {"loss": 1.0, "rec_rna": 0.5})
    history.log("train", {"loss": 0.8, "rec_rna": 0.4})
    history.log("test", {"loss": 1.1})
    history.log("test", {"loss": 0.9})

    return DeepSCENICModel(
        vae=mock_vae,
        tf2rnet=mock_tf2rnet,
        enformer=MockEnformer(
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
    import anndata as ad

    d = MINIMAL_DIMS
    n_cells = d["n_cells"] * 2  # 16 cells (8 train, 8 test)

    rna = ad.AnnData(X=np.random.rand(n_cells, d["n_genes"]).astype(np.float32))
    rna.var_names = [f"GENE{i}" for i in range(d["n_genes"])]
    rna.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    atac = ad.AnnData(X=np.random.rand(n_cells, d["n_regions"]).astype(np.float32))
    atac.var_names = [f"chr1:{i * 100}-{i * 100 + 100}" for i in range(d["n_regions"])]
    atac.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    mdata = md.MuData({"rna": rna, "atac": atac})
    mdata.obs["split"] = ["train"] * (n_cells // 2) + ["test"] * (n_cells // 2)

    return mdata
