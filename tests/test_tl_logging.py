"""Tests for logging interface."""

import pytest

from deepscenic.tl._logging import (
    CompositeLogger,
    DictLogger,
    TrainingLogger,
    get_logger,
)


class TestDictLogger:
    """Tests for DictLogger."""

    def test_log_metrics(self):
        """Should store metrics in history."""
        logger = DictLogger()
        logger.log_metrics({"loss": 0.5, "acc": 0.9}, step=1)
        logger.log_metrics({"loss": 0.4, "acc": 0.95}, step=2)

        assert logger.history["loss"] == [0.5, 0.4]
        assert logger.history["acc"] == [0.9, 0.95]

    def test_log_hyperparams(self):
        """Should store hyperparameters."""
        logger = DictLogger()
        logger.log_hyperparams({"lr": 0.001, "batch_size": 64})

        assert logger.hyperparams["lr"] == 0.001
        assert logger.hyperparams["batch_size"] == 64

    def test_is_training_logger(self):
        """Should be instance of TrainingLogger."""
        logger = DictLogger()
        assert isinstance(logger, TrainingLogger)


class TestCompositeLogger:
    """Tests for CompositeLogger."""

    def test_forwards_to_all_loggers(self):
        """Should forward metrics to all child loggers."""
        logger1 = DictLogger()
        logger2 = DictLogger()
        composite = CompositeLogger([logger1, logger2])

        composite.log_metrics({"loss": 0.5}, step=1)

        assert logger1.history["loss"] == [0.5]
        assert logger2.history["loss"] == [0.5]

    def test_forwards_hyperparams(self):
        """Should forward hyperparams to all child loggers."""
        logger1 = DictLogger()
        logger2 = DictLogger()
        composite = CompositeLogger([logger1, logger2])

        composite.log_hyperparams({"lr": 0.001})

        assert logger1.hyperparams["lr"] == 0.001
        assert logger2.hyperparams["lr"] == 0.001

    def test_closes_all_loggers(self):
        """Should close all child loggers."""
        logger1 = DictLogger()
        logger2 = DictLogger()
        composite = CompositeLogger([logger1, logger2])

        composite.close()  # Should not raise


class TestGetLogger:
    """Tests for get_logger factory function."""

    def test_dict_backend(self):
        """Should create DictLogger for 'dict' backend."""
        logger = get_logger("dict")
        assert isinstance(logger, DictLogger)

    def test_none_backend(self):
        """Should create logger for 'none' backend."""
        logger = get_logger("none")
        # Should work without errors
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.close()

    def test_invalid_backend(self):
        """Should raise error for invalid backend."""
        with pytest.raises(ValueError, match="Unknown logger backend"):
            get_logger("invalid_backend")

    def test_tensorboard_backend(self, tmp_path):
        """Should create TensorboardLogger for 'tensorboard' backend."""
        pytest.importorskip("torch.utils.tensorboard")
        log_dir = str(tmp_path / "runs")
        logger = get_logger("tensorboard", log_dir=log_dir)

        from deepscenic.tl._logging import TensorboardLogger

        assert isinstance(logger, TensorboardLogger)
        logger.close()


class TestTensorboardLogger:
    """Tests for TensorboardLogger."""

    @pytest.fixture
    def tb_logger(self, tmp_path):
        """Create TensorboardLogger for testing."""
        pytest.importorskip("torch.utils.tensorboard")
        from deepscenic.tl._logging import TensorboardLogger

        log_dir = str(tmp_path / "runs")
        logger = TensorboardLogger(log_dir=log_dir)
        yield logger
        logger.close()

    def test_log_metrics(self, tb_logger):
        """Should log metrics without error."""
        tb_logger.log_metrics({"loss": 0.5}, step=1)
        # No assertion needed - just check it doesn't raise

    def test_log_hyperparams(self, tb_logger):
        """Should log hyperparams without error."""
        tb_logger.log_hyperparams({"lr": 0.001, "batch_size": 64})
        # No assertion needed - just check it doesn't raise

    def test_is_training_logger(self, tb_logger):
        """Should be instance of TrainingLogger."""
        assert isinstance(tb_logger, TrainingLogger)
