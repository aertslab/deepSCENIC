"""Tests for training state management."""

from deepscenic.tl._training_state import (
    EarlyStopping,
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


class TestCopyToCpu:
    """Tests for the CPU snapshot used by async checkpointing."""

    def test_nested_tensors_are_independent_cpu_copies(self):
        """Tensors at any depth are copied to CPU and no longer share storage with the original."""
        import torch

        from deepscenic.tl._train import _copy_to_cpu

        original = {"w": torch.ones(3), "state": {0: {"exp_avg": torch.zeros(2)}}, "groups": [{"lr": 1e-4}]}
        snapshot = _copy_to_cpu(original)
        original["w"].add_(1)
        original["state"][0]["exp_avg"].add_(1)

        assert snapshot["w"].device.type == "cpu"
        assert torch.equal(snapshot["w"], torch.ones(3))
        assert torch.equal(snapshot["state"][0]["exp_avg"], torch.zeros(2))
        assert snapshot["groups"] == [{"lr": 1e-4}]
        assert snapshot["groups"] is not original["groups"]

    def test_optimizer_state_round_trip(self):
        """An optimizer snapshot loads back into a fresh optimizer."""
        import torch

        from deepscenic.tl._train import _copy_to_cpu

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model(torch.randn(8, 4)).sum().backward()
        optimizer.step()

        restored = torch.optim.Adam(model.parameters(), lr=1e-3)
        restored.load_state_dict(_copy_to_cpu(optimizer.state_dict()))

        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(restored.state_dict()["state"][0][key], optimizer.state_dict()["state"][0][key])
