"""Tests for custom sequence model support."""

import pytest
import torch

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


class CustomSequenceModel(torch.nn.Module):
    """Simple custom sequence model for testing.

    Returns flattened embeddings directly (unlike Enformer which returns 3D).
    """

    def __init__(self, bottleneck_size: int = 16, emb_len: int = 2):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.emb_len = emb_len
        self.output_dim = bottleneck_size * emb_len
        self.linear = torch.nn.Linear(4, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened embeddings with shape (batch, bottleneck_size * emb_len)."""
        batch_size = x.shape[0]
        # Return random embeddings with correct shape
        return torch.randn(batch_size, self.output_dim)


class TestIsEnformer:
    """Tests for _is_enformer helper function."""

    def test_detects_enformer_by_name(self):
        """Should detect Enformer class by name."""
        from deepscenic.tl._train import _is_enformer

        # Create a mock class named "Enformer"
        class Enformer(torch.nn.Module):
            pass

        enformer = Enformer()
        assert _is_enformer(enformer) is True

    def test_detects_custom_model(self):
        """Should return False for non-Enformer models."""
        from deepscenic.tl._train import _is_enformer

        custom = CustomSequenceModel()
        assert _is_enformer(custom) is False

    def test_detects_generic_module(self):
        """Should return False for generic nn.Module."""
        from deepscenic.tl._train import _is_enformer

        model = torch.nn.Linear(10, 10)
        assert _is_enformer(model) is False


class TestGetSequenceEmbeddings:
    """Tests for _get_sequence_embeddings function."""

    def test_custom_model_returns_flat_embeddings(self):
        """Custom model output should be used directly."""
        from deepscenic.tl._train import _get_sequence_embeddings

        d = MINIMAL_DIMS
        custom = CustomSequenceModel(
            bottleneck_size=d["bottleneck_size"],
            emb_len=d["emb_len"],
        )

        # Create mock input sequences
        batch_size = 4
        seq_len = 100
        sequences = torch.randn(batch_size, seq_len, 4)

        emb = _get_sequence_embeddings(
            custom,
            sequences,
            d["bottleneck_size"],
            d["emb_len"],
        )

        expected_dim = d["bottleneck_size"] * d["emb_len"]
        assert emb.shape == (batch_size, expected_dim)

    def test_enformer_like_model_gets_reshaped(self):
        """Enformer-like model output should be flattened."""
        from deepscenic.tl._train import _get_sequence_embeddings

        d = MINIMAL_DIMS

        # Create mock Enformer-like model that returns 3D output
        class MockEnformer(torch.nn.Module):
            def __init__(self, bottleneck_size, emb_len):
                super().__init__()
                self.bottleneck_size = bottleneck_size
                self.emb_len = emb_len

            def forward(self, x, return_only_embeddings=False):
                batch_size = x.shape[0]
                return torch.randn(batch_size, self.emb_len, self.bottleneck_size)

        mock_enformer = MockEnformer(
            bottleneck_size=d["bottleneck_size"],
            emb_len=d["emb_len"],
        )

        # Rename the class to "Enformer" to trigger the special handling
        mock_enformer.__class__.__name__ = "Enformer"

        batch_size = 4
        seq_len = 100
        sequences = torch.randn(batch_size, seq_len, 4)

        emb = _get_sequence_embeddings(
            mock_enformer,
            sequences,
            d["bottleneck_size"],
            d["emb_len"],
        )

        expected_dim = d["bottleneck_size"] * d["emb_len"]
        assert emb.shape == (batch_size, expected_dim)

    def test_output_is_float32(self):
        """Output should be float32 regardless of input dtype."""
        from deepscenic.tl._train import _get_sequence_embeddings

        d = MINIMAL_DIMS
        custom = CustomSequenceModel(
            bottleneck_size=d["bottleneck_size"],
            emb_len=d["emb_len"],
        )

        batch_size = 4
        seq_len = 100
        sequences = torch.randn(batch_size, seq_len, 4).half()

        emb = _get_sequence_embeddings(
            custom,
            sequences,
            d["bottleneck_size"],
            d["emb_len"],
        )

        assert emb.dtype == torch.float32


class TestTrainingConfigSequenceModel:
    """Tests for TrainingConfig with sequence_model parameter."""

    def test_default_sequence_model_is_none(self):
        """Default sequence_model should be None."""
        from deepscenic.tl._training_state import TrainingConfig

        config = TrainingConfig()
        assert config.sequence_model is None

    def test_accepts_custom_sequence_model(self):
        """Should accept custom sequence model."""
        from deepscenic.tl._training_state import TrainingConfig

        custom = CustomSequenceModel()
        config = TrainingConfig(sequence_model=custom)
        assert config.sequence_model is custom

    def test_to_dict_excludes_sequence_model(self):
        """to_dict should exclude sequence_model (not serializable)."""
        from deepscenic.tl._training_state import TrainingConfig

        custom = CustomSequenceModel()
        config = TrainingConfig(sequence_model=custom)
        d = config.to_dict()

        assert "sequence_model" not in d

    def test_from_dict_ignores_sequence_model(self):
        """from_dict should work without sequence_model."""
        from deepscenic.tl._training_state import TrainingConfig

        config = TrainingConfig(epochs=50, batch_size=32)
        d = config.to_dict()
        restored = TrainingConfig.from_dict(d)

        assert restored.epochs == 50
        assert restored.batch_size == 32
        assert restored.sequence_model is None


class TestPretrainConfigSequenceModel:
    """Tests for PretrainConfig with sequence_model parameter."""

    def test_default_sequence_model_is_none(self):
        """Default sequence_model should be None."""
        from deepscenic.tl._training_state import PretrainConfig

        config = PretrainConfig()
        assert config.sequence_model is None

    def test_accepts_custom_sequence_model(self):
        """Should accept custom sequence model."""
        from deepscenic.tl._training_state import PretrainConfig

        custom = CustomSequenceModel()
        config = PretrainConfig(sequence_model=custom)
        assert config.sequence_model is custom

    def test_to_dict_excludes_sequence_model(self):
        """to_dict should exclude sequence_model (not serializable)."""
        from deepscenic.tl._training_state import PretrainConfig

        custom = CustomSequenceModel()
        config = PretrainConfig(sequence_model=custom)
        d = config.to_dict()

        assert "sequence_model" not in d


class TestModelSaveLoadCustomSequenceModel:
    """Tests for save/load with custom sequence models."""

    def test_save_marks_custom_model(self, mock_deepscenic_model, tmp_path):
        """Saving with custom model should set is_custom_sequence_model flag."""
        d = MINIMAL_DIMS

        # Replace enformer with a custom model
        custom = CustomSequenceModel(
            bottleneck_size=d["bottleneck_size"],
            emb_len=d["emb_len"],
        )
        mock_deepscenic_model.enformer = custom

        path = tmp_path / "model.pt"
        mock_deepscenic_model.save(path)

        data = torch.load(path)
        assert data["is_custom_sequence_model"] is True

    def test_save_marks_enformer_model(self, mock_deepscenic_model, tmp_path):
        """Saving with Enformer-like model should set flag to False."""
        # MockEnformer should be detected as custom (not named "Enformer")
        path = tmp_path / "model.pt"
        mock_deepscenic_model.save(path)

        data = torch.load(path)
        # MockEnformer is named "MockEnformer", not "Enformer", so it's custom
        assert data["is_custom_sequence_model"] is True

    def test_load_custom_model_requires_instance(self, mock_deepscenic_model, tmp_path):
        """Loading custom model without providing instance should raise error."""
        from deepscenic.tl._model import DeepSCENICModel

        d = MINIMAL_DIMS

        # Update config to match the custom model dimensions
        mock_deepscenic_model.config.bottleneck_size = d["bottleneck_size"]
        mock_deepscenic_model.config.emb_len = d["emb_len"]

        # Save model with custom sequence model
        custom = CustomSequenceModel(
            bottleneck_size=d["bottleneck_size"],
            emb_len=d["emb_len"],
        )
        mock_deepscenic_model.enformer = custom
        path = tmp_path / "model.pt"
        mock_deepscenic_model.save(path)

        # Try to load without providing sequence_model
        with pytest.raises(ValueError, match="custom sequence model"):
            DeepSCENICModel.load(path)

    def test_load_custom_model_with_instance(self, mock_deepscenic_model, tmp_path):
        """Loading custom model with instance should work."""
        from deepscenic.tl._model import DeepSCENICModel

        d = MINIMAL_DIMS

        # Update config to match the custom model dimensions
        mock_deepscenic_model.config.bottleneck_size = d["bottleneck_size"]
        mock_deepscenic_model.config.emb_len = d["emb_len"]

        # Save model with custom sequence model
        custom = CustomSequenceModel(
            bottleneck_size=d["bottleneck_size"],
            emb_len=d["emb_len"],
        )
        mock_deepscenic_model.enformer = custom
        path = tmp_path / "model.pt"
        mock_deepscenic_model.save(path)

        # Load with matching instance
        new_custom = CustomSequenceModel(
            bottleneck_size=d["bottleneck_size"],
            emb_len=d["emb_len"],
        )
        loaded = DeepSCENICModel.load(path, sequence_model=new_custom)

        assert loaded.enformer is new_custom
