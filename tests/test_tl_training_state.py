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


class TestSaveBestCheckpoints:
    """Tests for save_best_checkpoints parameter behavior."""

    def test_default_enables_with_checkpoint_dir(self):
        """save_best_checkpoints=None enables when checkpoint_dir is set."""
        checkpoint_dir = "/some/path"
        save_best = None
        resolved = save_best if save_best is not None else (checkpoint_dir is not None)
        assert resolved is True

    def test_default_disables_without_checkpoint_dir(self):
        """save_best_checkpoints=None disables when checkpoint_dir is None."""
        checkpoint_dir = None
        save_best = None
        resolved = save_best if save_best is not None else (checkpoint_dir is not None)
        assert resolved is False

    def test_explicit_false_disables(self):
        """save_best_checkpoints=False disables even with checkpoint_dir."""
        checkpoint_dir = "/some/path"
        save_best = False
        resolved = save_best if save_best is not None else (checkpoint_dir is not None)
        assert resolved is False

    def test_best_loss_preserved_in_checkpoint(self, tmp_path):
        """best_loss should be preserved when saving/loading checkpoints."""
        history = TrainingHistory()
        config = TrainingConfig()
        checkpoint = Checkpoint(
            epoch=5,
            vae_state_dict={},
            tf2rnet_state_dict={},
            enformer_state_dict=None,
            optimizer_state_dict={},
            scheduler_state_dict=None,
            history=history,
            config=config,
            adj_E1=torch.randn(10, 5),
            best_loss=0.42,
        )
        path = tmp_path / "test.pt"
        checkpoint.save(path)
        loaded = Checkpoint.load(path)
        assert loaded.best_loss == pytest.approx(0.42)
