"""Tests for aligning MuData features to a model and the mismatch checks."""

import warnings

import anndata as ad
import mudata as md
import numpy as np
import pytest
from scipy import sparse

import deepscenic as ds


def _misaligned(mdata: md.MuData, seed: int = 0) -> md.MuData:
    """Return a copy of mdata with shuffled features plus one extra gene and region."""
    rng = np.random.default_rng(seed)
    rna, atac = mdata["rna"], mdata["atac"]

    extra_gene = ad.AnnData(X=np.ones((rna.n_obs, 1), dtype=np.float32), obs=rna.obs[[]], var={"split": ["train"]})
    extra_gene.var_names = ["EXTRA_GENE"]
    extra_region = ad.AnnData(X=np.ones((atac.n_obs, 1), dtype=np.float32), obs=atac.obs[[]], var={"split": ["train"]})
    extra_region.var_names = ["chr9:0-100"]

    rna = ad.concat([rna, extra_gene], axis=1, merge="same")
    atac = ad.concat([atac, extra_region], axis=1, merge="same")
    rna = rna[:, rng.permutation(rna.n_vars)].copy()
    atac = atac[:, rng.permutation(atac.n_vars)].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = md.MuData({"rna": rna, "atac": atac})
    out.obs["split"] = mdata.obs["split"].to_numpy()
    return out


class TestAlignToModel:
    """Tests for ds.pp.align_to_model."""

    def test_reorders_and_drops_extra_features(self, mock_deepscenic_model, mock_mdata_for_model):
        """Features should follow the model order, with extra features dropped."""
        model = mock_deepscenic_model
        mdata = _misaligned(mock_mdata_for_model)
        expected_rna = mock_mdata_for_model["rna"].X

        ds.pp.align_to_model(mdata, model)

        assert list(mdata["rna"].var_names) == model.gene_names
        assert list(mdata["atac"].var_names) == model.region_names
        np.testing.assert_array_equal(mdata["rna"].X, expected_rna)
        assert mdata["rna"].uns["tf_order"] == model.tf_names
        assert mdata.n_vars == len(model.gene_names) + len(model.region_names)

    def test_inplace_false_leaves_input_untouched(self, mock_deepscenic_model, mock_mdata_for_model):
        """inplace=False should return an aligned copy and keep the input as is."""
        mdata = _misaligned(mock_mdata_for_model)
        original_genes = list(mdata["rna"].var_names)

        result = ds.pp.align_to_model(mdata, mock_deepscenic_model, inplace=False)

        assert result is not None
        assert list(result["rna"].var_names) == mock_deepscenic_model.gene_names
        assert list(mdata["rna"].var_names) == original_genes

    def test_missing_model_features_raise(self, mock_deepscenic_model, mock_mdata_for_model):
        """Data lacking a model gene cannot be aligned."""
        mdata = mock_mdata_for_model
        mdata.mod["rna"] = mdata["rna"][:, 1:].copy()
        mdata.update()

        with pytest.raises(ValueError, match="GENE0"):
            ds.pp.align_to_model(mdata, mock_deepscenic_model)

    def test_reorders_r2g_penalty(self, mock_deepscenic_model, mock_mdata_for_model):
        """The r2g penalty matrix should follow the new gene and region order."""
        model = mock_deepscenic_model
        mdata = _misaligned(mock_mdata_for_model)
        genes, regions = list(mdata["rna"].var_names), list(mdata["atac"].var_names)
        rng = np.random.default_rng(1)
        matrix = sparse.csr_matrix(
            rng.random((len(regions), len(genes))) * (rng.random((len(regions), len(genes))) < 0.3)
        )
        mdata.uns["r2g"] = {"matrix": matrix, "region_names": regions, "gene_names": genes, "config": {}}

        ds.pp.align_to_model(mdata, model)

        r2g = mdata.uns["r2g"]
        assert r2g["region_names"] == model.region_names
        assert r2g["gene_names"] == model.gene_names
        dense = matrix.toarray()
        expected = dense[
            np.ix_([regions.index(r) for r in model.region_names], [genes.index(g) for g in model.gene_names])
        ]
        np.testing.assert_array_equal(r2g["matrix"].toarray(), expected)
        assert r2g["config"]["n_links"] == r2g["matrix"].nnz


class TestModelMismatchErrors:
    """Functions that index RNA by position should refuse misaligned data."""

    def test_to_latent_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """to_latent should refuse misaligned data."""
        mdata = _misaligned(mock_mdata_for_model)
        with pytest.raises(ValueError, match="align_to_model"):
            ds.tl.to_latent(mock_deepscenic_model, mdata)

    def test_simulate_perturbation_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """simulate_perturbation should refuse misaligned data."""
        mdata = _misaligned(mock_mdata_for_model)
        with pytest.raises(ValueError, match="align_to_model"):
            ds.tl.simulate_perturbation(mock_deepscenic_model, mdata, tf_name="TF0", level=0)

    def test_same_genes_different_order_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """Same genes in a different order should also be refused."""
        mdata = mock_mdata_for_model
        mdata.mod["rna"] = mdata["rna"][:, ::-1].copy()
        mdata.update()
        with pytest.raises(ValueError, match="align_to_model"):
            ds.tl.to_latent(mock_deepscenic_model, mdata)

    def test_aligned_data_works(self, mock_deepscenic_model, mock_mdata_for_model):
        """Aligned data should run through to_latent."""
        mdata = _misaligned(mock_mdata_for_model)
        ds.pp.align_to_model(mdata, mock_deepscenic_model)

        ds.tl.to_latent(mock_deepscenic_model, mdata)

        assert mdata.obsm["X_deepscenic_z_tf"].shape == (mdata.n_obs, len(mock_deepscenic_model.tf_names))
