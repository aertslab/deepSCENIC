"""Tests for training state management."""

import pytest
import torch

from deepscenic.tl._training_state import (
    EarlyStopping,
    ModelConfig,
    TrainingHistory,
)


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
