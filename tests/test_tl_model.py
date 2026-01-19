"""Tests for model wrapper (_model.py)."""

import tempfile
from pathlib import Path

import pytest
import torch


class TestDeepSCENICModel:
    """Tests for DeepSCENICModel class."""

    def test_has_required_attributes(self, mock_deepscenic_model):
        """Model should have all required attributes."""
        assert hasattr(mock_deepscenic_model, "vae")
        assert hasattr(mock_deepscenic_model, "tf2rnet")
        assert hasattr(mock_deepscenic_model, "enformer")
        assert hasattr(mock_deepscenic_model, "adj_E1")
        assert hasattr(mock_deepscenic_model, "config")
        assert hasattr(mock_deepscenic_model, "tf_names")
        assert hasattr(mock_deepscenic_model, "gene_names")
        assert hasattr(mock_deepscenic_model, "region_names")

    def test_to_device(self, mock_deepscenic_model):
        """Model should move to device correctly."""
        mock_deepscenic_model.to("cpu")
        assert next(mock_deepscenic_model.vae.parameters()).device.type == "cpu"

    def test_eval_mode(self, mock_deepscenic_model):
        """Model should set eval mode correctly."""
        mock_deepscenic_model.train()
        assert mock_deepscenic_model.vae.training
        mock_deepscenic_model.eval()
        assert not mock_deepscenic_model.vae.training

    def test_train_mode(self, mock_deepscenic_model):
        """Model should set train mode correctly."""
        mock_deepscenic_model.eval()
        assert not mock_deepscenic_model.vae.training
        mock_deepscenic_model.train()
        assert mock_deepscenic_model.vae.training


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_creates_file(self, mock_deepscenic_model):
        """Save should create a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            mock_deepscenic_model.save(path)
            assert path.exists()

    def test_save_file_is_loadable(self, mock_deepscenic_model):
        """Saved file should be loadable by torch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            mock_deepscenic_model.save(path)
            data = torch.load(path)
            assert "vae_state_dict" in data
            assert "tf2rnet_state_dict" in data
            assert "adj_E1" in data
            assert "config" in data
            assert "tf_names" in data
            assert "gene_names" in data
            assert "region_names" in data

    def test_save_contains_architecture_info(self, mock_deepscenic_model):
        """Saved file should contain architecture info for reconstruction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            mock_deepscenic_model.save(path)
            data = torch.load(path)
            assert "n_tfs" in data
            assert "n_genes" in data
            assert "n_regions" in data
            assert "n_hidden" in data

    def test_save_contains_buffers(self, mock_deepscenic_model):
        """Saved file should contain buffer tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            mock_deepscenic_model.save(path)
            data = torch.load(path)
            assert "tf_indices" in data
            assert "gene_indices" in data
            assert "r2g_indices" in data
            assert "r2g_distances" in data


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_returns_correct_type(self, mock_deepscenic_model):
        """load_model should return DeepSCENICModel (skip if Enformer needed)."""
        pytest.skip("Requires Enformer model download")
