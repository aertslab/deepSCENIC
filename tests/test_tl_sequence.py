"""Tests for sequence interpretation tools."""

from types import SimpleNamespace

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import torch

from deepscenic.tl import in_silico_mutagenesis


class _Genome:
    sequence = "AC"

    def fetch(self, chrom, start, end):
        assert (chrom, start, end) == ("chr1", 10, 12)
        return self.sequence

    def fetch_onehot(self, chrom, start, end):
        encoding = torch.tensor([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=torch.float32)
        return encoding


class _VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("tf_indices", torch.tensor([0, 1]))


class _SequenceModel(torch.nn.Module):
    def forward(self, sequences):
        return sequences.sum(dim=1)


class _MotifNet(torch.nn.Module):
    def forward(self, embeddings):
        return embeddings[:, :2]


@pytest.fixture
def ism_inputs():
    """Create test inputs for in silico mutagenesis testing."""
    obs = pd.DataFrame({"lineState": ["MEL", "MEL", "OTHER"]}, index=["a", "b", "c"])
    mdata = md.MuData({"rna": ad.AnnData(np.zeros((3, 2)), obs=obs.copy())})
    mdata.obs["lineState"] = obs["lineState"]
    mdata.obsm["X_deepscenic_z_tf"] = np.array([[1, 2], [3, 4], [20, 30]])
    model = SimpleNamespace(
        vae=_VAE(),
        sequence_model=_SequenceModel(),
        motifnet=_MotifNet(),
        config=SimpleNamespace(seq_len=2, bottleneck_size=4, emb_len=1),
        tf_names=["TF1", "TF2"],
    )
    return model, mdata


def test_in_silico_mutagenesis_scores_and_metadata(ism_inputs):
    """Test that function returns correct scores and metadata."""
    model, mdata = ism_inputs
    result = in_silico_mutagenesis(
        model,
        mdata,
        "chr1:10-12",
        "lineState",
        "MEL",
        genome=_Genome(),
        batch_size=3,
    )

    expected = np.array([[0, 1, -2, -2], [-1, 0, -3, -3]], dtype=np.float32)
    np.testing.assert_allclose(result.scores, expected)
    assert result.sequence == "AC"
    assert result.sequence_region == "chr1:10-12"
    assert result.baseline_score == pytest.approx(5.0)
    assert result.class_label == "MEL"


def test_in_silico_mutagenesis_rejects_empty_class(ism_inputs):
    """Test that function raises error for empty cell classes."""
    model, mdata = ism_inputs
    with pytest.raises(ValueError, match="No cells"):
        in_silico_mutagenesis(
            model,
            mdata,
            "chr1:10-12",
            "lineState",
            "ABSENT",
            genome=_Genome(),
        )
