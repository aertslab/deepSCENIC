"""Tests for split handling utilities."""

from deepscenic._data import TrainingView, get_split


class TestGetSplit:
    """Tests for get_split function.

    With the new split='both' for TFs:
    - sample_rna: 50 genes total
      - Genes 0-9 (10): TFs with split='both'
      - Genes 10-29 (20): non-TFs with split='train'
      - Genes 30-39 (10): non-TFs with split='test' (chr7)
      - Genes 40-49 (10): non-TFs with split='train'
    - features='train' includes split in ['train', 'both'] = 10 + 20 + 10 = 40 genes
    - features='test' includes split in ['test', 'both'] = 10 + 10 = 20 genes
    """

    def test_get_split_rna_train(self, sample_mdata):
        """Test getting RNA train split."""
        result = get_split(sample_mdata, "rna", cells="train", features="train")

        # 80 train cells, 40 genes (10 both + 30 train)
        assert result.n_obs == 80
        assert result.n_vars == 40

    def test_get_split_rna_test(self, sample_mdata):
        """Test getting RNA test split."""
        result = get_split(sample_mdata, "rna", cells="test", features="test")

        # 20 test cells, 20 genes (10 both + 10 test)
        assert result.n_obs == 20
        assert result.n_vars == 20

    def test_get_split_rna_e2(self, sample_mdata):
        """Test getting RNA E2 split (train cells, test features)."""
        result = get_split(sample_mdata, "rna", cells="train", features="test")

        # 80 train cells, 20 genes (10 both + 10 test)
        assert result.n_obs == 80
        assert result.n_vars == 20

    def test_get_split_atac_train(self, sample_mdata):
        """Test getting ATAC train split."""
        result = get_split(sample_mdata, "atac", cells="train", features="train")

        # ATAC has no "both" category - 20 train regions
        assert result.n_obs == 80
        assert result.n_vars == 20

    def test_get_split_all_cells(self, sample_mdata):
        """Test getting all cells."""
        result = get_split(sample_mdata, "rna", cells="all", features="train")

        assert result.n_obs == 100
        # 40 genes (10 both + 30 train)
        assert result.n_vars == 40

    def test_get_split_all_features(self, sample_mdata):
        """Test getting all features."""
        result = get_split(sample_mdata, "rna", cells="train", features="all")

        assert result.n_vars == 50

    def test_get_split_includes_both_in_train_and_test(self, sample_mdata):
        """Test that genes with split='both' appear in both train and test views."""
        rna_train = get_split(sample_mdata, "rna", cells="all", features="train")
        rna_test = get_split(sample_mdata, "rna", cells="all", features="test")

        # TFs (Gene_0 to Gene_9) should appear in both
        for i in range(10):
            gene_name = f"Gene_{i}"
            assert gene_name in rna_train.var_names, f"{gene_name} should be in train"
            assert gene_name in rna_test.var_names, f"{gene_name} should be in test"


class TestTrainingView:
    """Tests for TrainingView class.

    With the new split='both' for TFs:
    - rna_train: 40 genes (10 both + 30 train)
    - rna_test: 20 genes (10 both + 10 test)
    - r2g_train: 20 regions × 40 genes
    - r2g_test: 10 regions × 20 genes
    """

    def test_training_view_rna_train(self, sample_mdata):
        """Test TrainingView.rna_train property."""
        view = TrainingView(sample_mdata)

        result = view.rna_train
        assert result.n_obs == 80
        assert result.n_vars == 40  # 10 both + 30 train

    def test_training_view_rna_test(self, sample_mdata):
        """Test TrainingView.rna_test property."""
        view = TrainingView(sample_mdata)

        result = view.rna_test
        assert result.n_obs == 20
        assert result.n_vars == 20  # 10 both + 10 test

    def test_training_view_rna_e2(self, sample_mdata):
        """Test TrainingView.rna_e2 property."""
        view = TrainingView(sample_mdata)

        result = view.rna_e2
        assert result.n_obs == 80
        assert result.n_vars == 20  # 10 both + 10 test

    def test_training_view_rna_eval(self, sample_mdata):
        """Test TrainingView.rna_eval property."""
        view = TrainingView(sample_mdata)

        result = view.rna_eval
        assert result.n_obs == 20
        assert result.n_vars == 40  # 10 both + 30 train

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
        # 20 train regions × 40 genes (10 both + 30 train)
        assert r2g.shape == (20, 40)

    def test_training_view_r2g_test(self, sample_mdata):
        """Test TrainingView.r2g_test property."""
        view = TrainingView(sample_mdata)

        r2g = view.r2g_test
        # 10 test regions × 20 genes (10 both + 10 test)
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
