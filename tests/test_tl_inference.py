"""Tests for inference functions (_inference.py)."""

import numpy as np
from scipy.sparse import csr_matrix

from deepscenic.tl._inference import to_latent


class TestToLatent:
    """Tests for to_latent function."""

    def test_returns_expected_keys(self, mock_deepscenic_model, mock_mdata_for_model):
        """to_latent should return all expected embedding keys."""
        result = to_latent(mock_deepscenic_model, mock_mdata_for_model, batch_size=4, device="cpu")
        expected_keys = ["z_tf", "enh_act", "z_rna", "x_rna_rec", "x_atac_rec", "x_rna_ppi"]
        for key in expected_keys:
            assert key in result

    def test_output_shapes(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Output arrays should have correct shapes."""
        result = to_latent(mock_deepscenic_model, mock_mdata_for_model, batch_size=4, device="cpu")

        d = minimal_dims
        n_cells = d["n_cells"] * 2

        assert result["z_tf"].shape == (n_cells, d["n_tfs"])
        assert result["enh_act"].shape == (n_cells, d["n_regions"])
        assert result["z_rna"].shape == (n_cells, d["n_genes"])
        assert result["x_rna_rec"].shape == (n_cells, d["n_genes"])
        assert result["x_atac_rec"].shape == (n_cells, d["n_regions"])
        assert result["x_rna_ppi"].shape == (n_cells, d["n_tfs"])

    def test_returns_numpy_arrays(self, mock_deepscenic_model, mock_mdata_for_model):
        """All values should be numpy arrays."""
        result = to_latent(mock_deepscenic_model, mock_mdata_for_model, batch_size=4, device="cpu")
        for key, value in result.items():
            assert isinstance(value, np.ndarray), f"{key} is not numpy array"

    def test_split_filter(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should filter by split when specified."""
        result = to_latent(mock_deepscenic_model, mock_mdata_for_model, batch_size=4, device="cpu", split="train")
        assert result["z_tf"].shape[0] == minimal_dims["n_cells"]

    def test_deterministic_with_eval_mode(self, mock_deepscenic_model, mock_mdata_for_model):
        """Results should be deterministic (uses mean, not sampling)."""
        mock_deepscenic_model.eval()
        result1 = to_latent(mock_deepscenic_model, mock_mdata_for_model, batch_size=4, device="cpu")
        result2 = to_latent(mock_deepscenic_model, mock_mdata_for_model, batch_size=4, device="cpu")
        np.testing.assert_allclose(result1["z_tf"], result2["z_tf"])

    def test_handles_sparse_input(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should handle sparse matrix inputs."""
        mock_mdata_for_model.mod["rna"].X = csr_matrix(mock_mdata_for_model.mod["rna"].X)
        mock_mdata_for_model.mod["atac"].X = csr_matrix(mock_mdata_for_model.mod["atac"].X)

        result = to_latent(mock_deepscenic_model, mock_mdata_for_model, batch_size=4, device="cpu")
        assert result["z_tf"].shape[0] == minimal_dims["n_cells"] * 2
