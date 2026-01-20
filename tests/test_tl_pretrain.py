"""Tests for pretrain functionality."""

import pytest
import torch

from deepscenic.tl._model import PretrainedModel
from deepscenic.tl._training_state import PretrainConfig


class TestPretrainConfig:
    """Tests for PretrainConfig dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        config = PretrainConfig()

        assert config.epochs == 100
        assert config.batch_size == 256
        assert config.lr == 1e-4
        assert config.weight_decay == 0.0
        assert config.alpha == 1e-2
        assert config.device == "cuda"
        assert config.bottleneck_size == 3072
        assert config.emb_len == 5
        assert config.seq_len == 640

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        config = PretrainConfig(epochs=50, lr=1e-5)
        d = config.to_dict()

        assert d["epochs"] == 50
        assert d["lr"] == 1e-5
        assert isinstance(d, dict)

    def test_from_dict_roundtrip(self):
        """Should save and load from dict correctly."""
        config = PretrainConfig(epochs=50, lr=1e-5, alpha=0.05)
        d = config.to_dict()
        loaded = PretrainConfig.from_dict(d)

        assert loaded.epochs == config.epochs
        assert loaded.lr == config.lr
        assert loaded.alpha == config.alpha

    def test_from_dict_ignores_extra_keys(self):
        """Should ignore unknown keys in dictionary."""
        d = {"epochs": 200, "unknown_key": "value"}
        config = PretrainConfig.from_dict(d)

        assert config.epochs == 200
        assert not hasattr(config, "unknown_key")


class TestPretrainedModel:
    """Tests for PretrainedModel dataclass."""

    @pytest.fixture
    def sample_pretrained_model(self, mock_tf2rnet, mock_adj_E1, minimal_dims):
        """Create sample PretrainedModel for testing."""
        import torch

        # Simple mock enformer
        class MockEnformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return x

        config = PretrainConfig(epochs=10, batch_size=32)
        d = minimal_dims

        return PretrainedModel(
            tf2rnet=mock_tf2rnet,
            enformer=MockEnformer(),
            adj_E1=mock_adj_E1,
            config=config,
            tf_names=[f"TF{i}" for i in range(d["n_tfs"])],
            region_names=[f"chr1:{i * 100}-{i * 100 + 100}" for i in range(d["n_regions"])],
        )

    def test_has_required_attributes(self, sample_pretrained_model):
        """Should have all required attributes."""
        model = sample_pretrained_model

        assert hasattr(model, "tf2rnet")
        assert hasattr(model, "enformer")
        assert hasattr(model, "adj_E1")
        assert hasattr(model, "config")
        assert hasattr(model, "tf_names")
        assert hasattr(model, "region_names")

    def test_save_creates_file(self, sample_pretrained_model, tmp_path):
        """Save should create a file."""
        path = tmp_path / "pretrained.pt"
        sample_pretrained_model.save(path)

        assert path.exists()

    def test_save_file_is_loadable(self, sample_pretrained_model, tmp_path):
        """Saved file should be loadable with torch.load."""
        path = tmp_path / "pretrained.pt"
        sample_pretrained_model.save(path)

        data = torch.load(path, map_location="cpu")

        assert "tf2rnet_state_dict" in data
        assert "adj_E1" in data
        assert "config" in data
        assert "tf_names" in data
        assert "region_names" in data

    def test_to_device(self, sample_pretrained_model):
        """Should move model to specified device."""
        model = sample_pretrained_model.to("cpu")

        assert model.adj_E1.device == torch.device("cpu")

    def test_eval_mode(self, sample_pretrained_model):
        """Should set all components to eval mode."""
        model = sample_pretrained_model.eval()

        assert not model.tf2rnet.training

    def test_train_mode(self, sample_pretrained_model):
        """Should set all components to train mode."""
        sample_pretrained_model.eval()
        model = sample_pretrained_model.train_mode()

        assert model.tf2rnet.training


class TestLoadPretrained:
    """Tests for load_pretrained function."""

    def test_function_exists(self):
        """Load pretrained function should be importable."""
        from deepscenic.tl import load_pretrained

        assert callable(load_pretrained)


class TestTrainWithPretrained:
    """Tests for train() with pretrained_model parameter."""

    def test_train_accepts_pretrained_model_param(self):
        """Train function should accept pretrained_model parameter."""
        from inspect import signature

        from deepscenic.tl import train

        sig = signature(train)
        params = list(sig.parameters.keys())

        assert "pretrained_model" in params
