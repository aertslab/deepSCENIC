"""Tests for split handling utilities."""

from deepscenic._data import TrainingView, get_split


class TestGetSplit:
    """Tests for get_split function."""

    def test_get_split_rna_train(self, sample_mdata):
        """Test getting RNA train split."""
        result = get_split(sample_mdata, "rna", cells="train", features="train")

        # 80 train cells, 30 train genes
        assert result.n_obs == 80
        assert result.n_vars == 30

    def test_get_split_rna_test(self, sample_mdata):
        """Test getting RNA test split."""
        result = get_split(sample_mdata, "rna", cells="test", features="test")

        # 20 test cells, 20 test genes
        assert result.n_obs == 20
        assert result.n_vars == 20

    def test_get_split_rna_e2(self, sample_mdata):
        """Test getting RNA E2 split (train cells, test features)."""
        result = get_split(sample_mdata, "rna", cells="train", features="test")

        assert result.n_obs == 80
        assert result.n_vars == 20

    def test_get_split_atac_train(self, sample_mdata):
        """Test getting ATAC train split."""
        result = get_split(sample_mdata, "atac", cells="train", features="train")

        assert result.n_obs == 80
        assert result.n_vars == 20

    def test_get_split_all_cells(self, sample_mdata):
        """Test getting all cells."""
        result = get_split(sample_mdata, "rna", cells="all", features="train")

        assert result.n_obs == 100

    def test_get_split_all_features(self, sample_mdata):
        """Test getting all features."""
        result = get_split(sample_mdata, "rna", cells="train", features="all")

        assert result.n_vars == 50


class TestTrainingView:
    """Tests for TrainingView class."""

    def test_training_view_rna_train(self, sample_mdata):
        """Test TrainingView.rna_train property."""
        view = TrainingView(sample_mdata)

        result = view.rna_train
        assert result.n_obs == 80
        assert result.n_vars == 30

    def test_training_view_rna_test(self, sample_mdata):
        """Test TrainingView.rna_test property."""
        view = TrainingView(sample_mdata)

        result = view.rna_test
        assert result.n_obs == 20
        assert result.n_vars == 20

    def test_training_view_rna_e2(self, sample_mdata):
        """Test TrainingView.rna_e2 property."""
        view = TrainingView(sample_mdata)

        result = view.rna_e2
        assert result.n_obs == 80
        assert result.n_vars == 20

    def test_training_view_rna_eval(self, sample_mdata):
        """Test TrainingView.rna_eval property."""
        view = TrainingView(sample_mdata)

        result = view.rna_eval
        assert result.n_obs == 20
        assert result.n_vars == 30

    def test_training_view_atac_train(self, sample_mdata):
        """Test TrainingView.atac_train property."""
        view = TrainingView(sample_mdata)

        result = view.atac_train
        assert result.n_obs == 80
        assert result.n_vars == 20

    def test_training_view_r2g_train(self, sample_mdata):
        """Test TrainingView.r2g_train property."""
        view = TrainingView(sample_mdata)

        r2g = view.r2g_train
        assert r2g.shape == (20, 30)

    def test_training_view_r2g_test(self, sample_mdata):
        """Test TrainingView.r2g_test property."""
        view = TrainingView(sample_mdata)

        r2g = view.r2g_test
        assert r2g.shape == (10, 20)

    def test_training_view_tf_names(self, sample_mdata):
        """Test TrainingView.tf_names property."""
        view = TrainingView(sample_mdata)

        tf_names = view.tf_names
        assert len(tf_names) == 10
        assert tf_names[0] == "Gene_0"

    def test_training_view_n_tfs(self, sample_mdata):
        """Test TrainingView.n_tfs property."""
        view = TrainingView(sample_mdata)

        assert view.n_tfs == 10

    def test_training_view_caching(self, sample_mdata):
        """Test that TrainingView caches results."""
        view = TrainingView(sample_mdata)

        # Access twice
        result1 = view.rna_train
        result2 = view.rna_train

        # Should be same object (cached)
        assert result1 is result2

    def test_training_view_clear_cache(self, sample_mdata):
        """Test TrainingView.clear_cache method."""
        view = TrainingView(sample_mdata)

        # Populate cache
        _ = view.rna_train
        assert len(view._cache) > 0

        # Clear cache
        view.clear_cache()
        assert len(view._cache) == 0
