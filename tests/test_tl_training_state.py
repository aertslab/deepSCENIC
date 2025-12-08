"""Tests for training state management."""

import pytest
import torch

from deepscenic.tl._training_state import (
    Checkpoint,
    EarlyStopping,
    TrainingConfig,
    TrainingHistory,
)


class TestCheckpoint:
    """Tests for Checkpoint save/load."""

    @pytest.fixture
    def sample_checkpoint(self):
        """Create sample checkpoint for testing."""
        history = TrainingHistory()
        history.log("train", {"loss": 0.5})
        config = TrainingConfig(epochs=50)

        return Checkpoint(
            epoch=10,
            vae_state_dict={"weight": torch.randn(10, 10)},
            tf2rnet_state_dict={"weight": torch.randn(5, 5)},
            enformer_state_dict=None,
            optimizer_state_dict={"step": 100},
            scheduler_state_dict=None,
            history=history,
            config=config,
            adj_E1=torch.randn(100, 50),
            best_loss=0.3,
        )

    def test_save_load_roundtrip(self, sample_checkpoint, tmp_path):
        """Should save and load correctly."""
        path = tmp_path / "checkpoint.pt"
        sample_checkpoint.save(path)

        loaded = Checkpoint.load(path)

        assert loaded.epoch == sample_checkpoint.epoch
        assert loaded.best_loss == sample_checkpoint.best_loss
        assert loaded.config.epochs == sample_checkpoint.config.epochs
        assert loaded.history.train["loss"] == sample_checkpoint.history.train["loss"]
        assert torch.allclose(loaded.adj_E1, sample_checkpoint.adj_E1)

    def test_load_map_location(self, sample_checkpoint, tmp_path):
        """Should respect map_location."""
        path = tmp_path / "checkpoint.pt"
        sample_checkpoint.save(path)

        loaded = Checkpoint.load(path, map_location="cpu")
        assert loaded.adj_E1.device == torch.device("cpu")


class TestEarlyStopping:
    """Tests for EarlyStopping class."""

    def test_stop_after_patience(self):
        """Should stop after patience epochs without improvement."""
        early_stop = EarlyStopping(patience=3)

        early_stop(1.0)  # Best loss
        early_stop(1.1)  # No improvement, counter=1
        early_stop(1.2)  # No improvement, counter=2
        result = early_stop(1.3)  # No improvement, counter=3, stop!

        assert result
        assert early_stop.should_stop

    def test_improvement_resets_counter(self):
        """Improvement should reset counter."""
        early_stop = EarlyStopping(patience=3)

        early_stop(1.0)
        early_stop(1.1)  # counter=1
        early_stop(1.2)  # counter=2
        early_stop(0.8)  # Improvement! counter=0

        assert not early_stop.should_stop
        assert early_stop.counter == 0
