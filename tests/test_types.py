"""Tests for LazyImpute and LazyImputeAnndata."""

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData

from deepscenic._types import LazyImpute, LazyImputeAnndata


N_CELLS = 10
N_REGIONS = 15
N_TOPICS = 5


@pytest.fixture
def matrices():
    """Random cell-topic and region-topic matrices."""
    rng = np.random.default_rng(0)
    cell_topic = rng.random((N_CELLS, N_TOPICS)).astype(np.float64)
    region_topic = rng.random((N_REGIONS, N_TOPICS)).astype(np.float64)
    return cell_topic, region_topic


@pytest.fixture
def cell_topic_adata(matrices):
    cell_topic, _ = matrices
    adata = sc.AnnData(cell_topic)
    adata.obs_names = [f"Cell_{i}" for i in range(N_CELLS)]
    adata.var_names = [f"Topic_{i}" for i in range(N_TOPICS)]
    return adata


@pytest.fixture
def region_topic_adata(matrices):
    _, region_topic = matrices
    adata = sc.AnnData(region_topic)
    adata.obs_names = [f"chr1:{i * 1000}-{i * 1000 + 640}" for i in range(N_REGIONS)]
    adata.var_names = [f"Topic_{i}" for i in range(N_TOPICS)]
    return adata


@pytest.fixture
def lazy_impute_anndata(cell_topic_adata, region_topic_adata):
    return LazyImputeAnndata.from_topic(
        cell_topic=cell_topic_adata,
        region_topic=region_topic_adata,
    )


class TestLazyImpute:
    """Tests for the LazyImpute proxy."""

    def test_getitem_1d_all_regions(self, matrices):
        """1D index selects cells, returns all regions."""
        cell_topic, region_topic = matrices
        li = LazyImpute(cell_topic, region_topic.T)

        result = li[0]
        expected = cell_topic[0, :] @ region_topic.T
        np.testing.assert_allclose(result, expected)

    def test_getitem_1d_slice(self, matrices):
        """1D slice selects a batch of cells."""
        cell_topic, region_topic = matrices
        li = LazyImpute(cell_topic, region_topic.T)

        result = li[2:5]
        expected = cell_topic[2:5, :] @ region_topic.T
        np.testing.assert_allclose(result, expected)

    def test_getitem_2d(self, matrices):
        """2D index selects cells and regions."""
        cell_topic, region_topic = matrices
        li = LazyImpute(cell_topic, region_topic.T)

        result = li[1:4, 3:8]
        expected = cell_topic[1:4, :] @ region_topic.T[:, 3:8]
        np.testing.assert_allclose(result, expected)

    def test_getitem_invalid_dim(self, matrices):
        """3D index raises ValueError."""
        cell_topic, region_topic = matrices
        li = LazyImpute(cell_topic, region_topic.T)

        with pytest.raises(ValueError, match="3-d"):
            li[0, 1, 2]

    def test_no_copy_of_arrays(self, matrices):
        """LazyImpute holds references, not copies."""
        cell_topic, region_topic = matrices
        li = LazyImpute(cell_topic, region_topic.T)

        assert li.cell_topic is cell_topic
        # .T returns a new view object each call, so use shares_memory instead of `is`
        assert np.shares_memory(li.topic_region, region_topic)


class TestLazyImputeAnndata:
    """Tests for LazyImputeAnndata."""

    def test_from_topic_shape(self, lazy_impute_anndata):
        """Shape is (n_cells, n_regions)."""
        assert lazy_impute_anndata.shape == (N_CELLS, N_REGIONS)

    def test_from_topic_obs_var(self, lazy_impute_anndata, cell_topic_adata, region_topic_adata):
        """obs and var names come from the input AnnData objects."""
        assert list(lazy_impute_anndata.obs_names) == list(cell_topic_adata.obs_names)
        assert list(lazy_impute_anndata.var_names) == list(region_topic_adata.obs_names)

    def test_from_topic_matrices_stored(self, lazy_impute_anndata):
        """Factor matrices are stored in obsm and varm."""
        assert "cell_topic" in lazy_impute_anndata.obsm
        assert "region_topic" in lazy_impute_anndata.varm

    def test_from_topic_mismatched_topics_raises(self, cell_topic_adata, region_topic_adata):
        """Mismatched topic count raises ValueError."""
        # Add an extra topic to region_topic
        extra = sc.AnnData(np.random.rand(N_REGIONS, N_TOPICS + 1))
        extra.obs_names = region_topic_adata.obs_names
        extra.var_names = [f"Topic_{i}" for i in range(N_TOPICS + 1)]

        with pytest.raises(ValueError, match="inconsistent number of topics"):
            LazyImputeAnndata.from_topic(cell_topic=cell_topic_adata, region_topic=extra)

    def test_X_returns_lazy_impute(self, lazy_impute_anndata):
        """Accessing .X returns a LazyImpute instance."""
        assert isinstance(lazy_impute_anndata.X, LazyImpute)

    def test_X_computes_correctly(self, lazy_impute_anndata, matrices):
        """Imputed values match cell_topic @ region_topic.T."""
        cell_topic, region_topic = matrices
        expected = cell_topic @ region_topic.T

        result = lazy_impute_anndata.X[:]
        np.testing.assert_allclose(result, expected)

    def test_X_setter_raises(self, lazy_impute_anndata):
        """Setting .X raises AttributeError."""
        with pytest.raises(AttributeError):
            lazy_impute_anndata.X = np.zeros((N_CELLS, N_REGIONS))

    def test_copy_returns_lazy_impute_anndata(self, lazy_impute_anndata):
        """copy() returns a LazyImputeAnndata, not a plain AnnData."""
        copied = lazy_impute_anndata.copy()
        assert isinstance(copied, LazyImputeAnndata)

    def test_copy_data_correct(self, lazy_impute_anndata, matrices):
        """Copied object produces the same imputed values."""
        cell_topic, region_topic = matrices
        expected = cell_topic @ region_topic.T

        copied = lazy_impute_anndata.copy()
        np.testing.assert_allclose(copied.X[:], expected)

    def test_to_memory_returns_lazy_impute_anndata(self, lazy_impute_anndata):
        """to_memory() on a view returns a LazyImputeAnndata."""
        view = lazy_impute_anndata[:5]
        result = view.to_memory()
        assert isinstance(result, LazyImputeAnndata)

    def test_slice_returns_lazy_impute_anndata(self, lazy_impute_anndata):
        """Slicing returns a LazyImputeAnndata."""
        sliced = lazy_impute_anndata[:5]
        assert isinstance(sliced, LazyImputeAnndata)

    def test_slice_cells_correct(self, lazy_impute_anndata, matrices):
        """Cell slice produces correct imputed values."""
        cell_topic, region_topic = matrices
        expected = cell_topic[2:5, :] @ region_topic.T

        sliced = lazy_impute_anndata[2:5]
        np.testing.assert_allclose(sliced.X[:], expected)

    def test_slice_regions_correct(self, lazy_impute_anndata, matrices):
        """Region slice produces correct imputed values."""
        cell_topic, region_topic = matrices
        region_names = lazy_impute_anndata.var_names[:5]
        expected = cell_topic @ region_topic[:5, :].T

        sliced = lazy_impute_anndata[:, region_names]
        np.testing.assert_allclose(sliced.X[:], expected)

    def test_from_anndata_basic(self, lazy_impute_anndata):
        """from_anndata wraps a plain AnnData with the right slots."""
        plain = AnnData(lazy_impute_anndata)  # plain copy
        result = LazyImputeAnndata.from_anndata(plain)
        assert isinstance(result, LazyImputeAnndata)

    def test_from_anndata_missing_fields_raises(self):
        """from_anndata raises ValueError when slots are missing."""
        plain = sc.AnnData(np.zeros((N_CELLS, N_REGIONS)))
        with pytest.raises(ValueError, match="cell_topic"):
            LazyImputeAnndata.from_anndata(plain)

    def test_from_anndata_computes_correctly(self, lazy_impute_anndata, matrices):
        """from_anndata produces correct imputed values."""
        cell_topic, region_topic = matrices
        expected = cell_topic @ region_topic.T

        plain = AnnData(lazy_impute_anndata)
        result = LazyImputeAnndata.from_anndata(plain)
        np.testing.assert_allclose(result.X[:], expected)
