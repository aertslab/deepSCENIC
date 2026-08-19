"""
Custom AnnData types for on-the-fly ATAC accessibility imputation.

Imputed accessibility is computed as the matrix product of cell-topic scores
and region-topic scores: ``cell_topic @ region_topic.T``. Rather than
materializing this (potentially large) matrix, :class:`LazyImputeAnndata`
stores the two factor matrices and computes slices on demand via
:class:`LazyImpute`.

The factor matrices are stored in the standard AnnData slots so that the
object can be saved and reloaded with ``anndata`` I/O unchanged:

- ``adata.obsm["cell_topic"]``: ``(n_cells, n_topics)``
- ``adata.varm["region_topic"]``: ``(n_regions, n_topics)``
"""

import logging
from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

log = logging.getLogger("deepscenic.types")

DEFAULT_SCALING_FACTOR = 10**6


class LazyImpute:
    """Proxy for the imputed accessibility matrix ``cell_topic @ region_topic.T``.

    Slicing this object computes only the requested submatrix, avoiding
    materializing the full ``(n_cells, n_regions)`` product.

    Parameters
    ----------
    cell_topic:
        Array of shape ``(n_cells, n_topics)``.
    topic_region:
        Array of shape ``(n_topics, n_regions)`` — i.e. ``region_topic.T``.
    scaling_factor:
        Scalar multiplied into the imputed values. Defaults to
        ``DEFAULT_SCALING_FACTOR`` (10^6), giving CPM-like units.
    """

    def __init__(self, cell_topic: np.ndarray, topic_region: np.ndarray, scaling_factor: float):
        self.cell_topic = cell_topic
        self.topic_region = topic_region
        self.scaling_factor = scaling_factor

    def __getitem__(self, index: int | slice | tuple[int | slice] | tuple[int | slice, int | slice]) -> np.ndarray:
        """Return the imputed submatrix for the given cell/region indices."""
        if not isinstance(index, tuple):
            index = (index,)

        region_idx: int | slice
        if len(index) == 1:
            cell_idx = index[0]
            region_idx = slice(None)  # slice across all regions
        elif len(index) == 2:
            cell_idx = index[0]
            region_idx = index[1]
        else:
            raise ValueError("Only 1-d or 2-d slicing is supported " + f"(not {len(index)}-d).")

        return cast(np.ndarray, (self.cell_topic[cell_idx, :] @ self.topic_region[:, region_idx]) * self.scaling_factor)


class LazyImputeAnndata(AnnData):
    """An :class:`~anndata.AnnData` whose ``X`` is imputed on the fly.

    Rather than storing a dense accessibility matrix, this class stores
    low-rank topic factor matrices and computes accessibility as
    ``cell_topic @ region_topic.T`` on demand. Accessing ``adata.X``
    returns a :class:`LazyImpute` proxy that performs the matrix product
    only for the requested slice.

    The object is otherwise a fully valid ``AnnData`` and can be saved and
    loaded with standard ``anndata`` I/O (``write_h5ad`` / ``read_h5ad``).
    After loading, reconstruct with :meth:`from_anndata`.

    Factor matrices
    ---------------
    obsm["cell_topic"]: ndarray of shape ``(n_cells, n_topics)``
    varm["region_topic"]: ndarray of shape ``(n_regions, n_topics)``
    """

    @classmethod
    def from_topic(
        cls,
        cell_topic: AnnData,
        region_topic: AnnData,
        scaling_factor: float = DEFAULT_SCALING_FACTOR,
    ) -> "LazyImputeAnndata":
        """Construct from separate cell-topic and region-topic AnnData objects.

        Parameters
        ----------
        cell_topic:
            AnnData of shape ``(n_cells, n_topics)`` where ``X`` holds
            per-cell topic scores.
        region_topic:
            AnnData of shape ``(n_regions, n_topics)`` where ``X`` holds
            per-region topic scores.
        scaling_factor:
            Scalar multiplied into the imputed values. Defaults to
            ``DEFAULT_SCALING_FACTOR`` (10^6), giving CPM-like units.
        """
        if cell_topic.shape[1] != region_topic.shape[1]:
            raise ValueError(
                f"cell_topic ({cell_topic.shape[1]} topics) and "
                + f"region_topic ({region_topic.shape[1]} topics) "
                + "have inconsistent number of topics!"
            )

        cell_obs = cell_topic.obs
        region_obs = region_topic.obs

        assert isinstance(cell_obs, pd.DataFrame)
        assert isinstance(region_obs, pd.DataFrame)

        X_cell_topic = cell_topic.X
        X_region_topic = region_topic.X

        assert isinstance(X_cell_topic, np.ndarray)
        assert isinstance(X_region_topic, np.ndarray)

        n_cells = cell_topic.shape[0]
        n_regions = region_topic.shape[0]

        # generate a sparse array of all zeros to fake a correct matrix in AnnData
        X_fake = scipy.sparse.csr_array((n_cells, n_regions), dtype=float)

        instance = cls.__new__(cls)
        instance._allow_X_assignment = True

        AnnData.__init__(instance, X=X_fake, obs=cell_obs, var=region_obs)
        instance._allow_X_assignment = False

        instance.obsm["cell_topic"] = X_cell_topic
        instance.varm["region_topic"] = X_region_topic
        instance.uns["scaling_factor"] = scaling_factor

        return instance

    @classmethod
    def from_anndata(cls, adata: AnnData) -> "LazyImputeAnndata":
        """Construct from a plain AnnData that already contains the factor matrices.

        Use this to reconstruct a :class:`LazyImputeAnndata` after loading
        from disk with ``anndata.read_h5ad``.

        Parameters
        ----------
        adata:
            AnnData with ``obsm["cell_topic"]`` and ``varm["region_topic"]``
            populated.
        """
        if "cell_topic" not in adata.obsm or "region_topic" not in adata.varm:
            raise ValueError(
                "adata should contain following fields:\n"
                + "\tadata.obsm['cell_topic']\n"
                + "\tadata.varm['region_topic']"
            )

        instance = cls.__new__(cls)
        instance._allow_X_assignment = True
        AnnData.__init__(instance, adata)
        instance._allow_X_assignment = False
        return instance

    @property
    def X(self) -> LazyImpute:  # type: ignore
        """Imputed accessibility matrix, computed on demand as ``cell_topic @ region_topic.T``."""
        if "cell_topic" not in self.obsm or "region_topic" not in self.varm:
            raise ValueError(
                "adata should contain following fields:\n"
                + "\tadata.obsm['cell_topic']\n"
                + "\tadata.varm['region_topic']"
            )
        if "scaling_factor" not in self.uns:
            log.warning(f"Scaling factor not found in uns field setting to {DEFAULT_SCALING_FACTOR}")
            self.uns["scaling_factor"] = DEFAULT_SCALING_FACTOR

        scaling_factor = self.uns["scaling_factor"]
        assert isinstance(scaling_factor, float) or isinstance(scaling_factor, int)

        return LazyImpute(
            cell_topic=self.obsm["cell_topic"],  # type: ignore
            topic_region=self.varm["region_topic"].T,  # type: ignore
            scaling_factor=scaling_factor,
        )

    @X.setter
    def X(self, value) -> None:  # type: ignore
        if getattr(self, "_allow_X_assignment", False):
            self._X = value
            return
        raise AttributeError(
            "Cannot set X on LazyImputeAnndata. "
            "Accessibility is imputed on the fly from obsm['cell_topic'] and varm['region_topic']."
        )

    def copy(self) -> "LazyImputeAnndata":  # type: ignore
        """Return an in-memory copy as a :class:`LazyImputeAnndata`.

        Cannot delegate to ``super().copy()`` because ``AnnData._mutated_copy``
        calls ``self.X.copy()``, which hits our property and returns a
        :class:`LazyImpute` object. AnnData then tries to store that as ``X``
        on the new object, which fails. We bypass this by reconstructing the
        copy directly from the factor matrices and metadata.
        """
        import copy as _copy

        n_cells, n_regions = self.shape
        X_fake = scipy.sparse.csr_array((n_cells, n_regions), dtype=float)

        assert isinstance(self.obs, pd.DataFrame)
        assert isinstance(self.var, pd.DataFrame)

        instance = LazyImputeAnndata.__new__(LazyImputeAnndata)
        instance._allow_X_assignment = True
        AnnData.__init__(
            instance,
            X=X_fake,
            obs=self.obs.copy(),
            var=self.var.copy(),
            uns=_copy.deepcopy(dict(self.uns)),
            obsm={k: v.copy() for k, v in self.obsm.items()},
            varm={k: v.copy() for k, v in self.varm.items()},
        )
        instance._allow_X_assignment = False
        return instance

    def to_memory(self, copy: bool = False) -> "LazyImputeAnndata":  # type: ignore
        """Convert a view to an in-memory :class:`LazyImputeAnndata`.

        Same issue as :meth:`copy` — ``super().to_memory()`` internally calls
        ``_mutated_copy`` which accesses ``self.X``. We bypass this by
        reconstructing directly from the (already-sliced) factor matrices.
        """
        import copy as _copy

        n_cells, n_regions = self.shape
        X_fake = scipy.sparse.csr_array((n_cells, n_regions), dtype=float)

        instance = LazyImputeAnndata.__new__(LazyImputeAnndata)
        instance._allow_X_assignment = True

        assert isinstance(self.obs, pd.DataFrame)
        assert isinstance(self.var, pd.DataFrame)

        AnnData.__init__(
            instance,
            X=X_fake,
            obs=self.obs.copy(),
            var=self.var.copy(),
            uns=_copy.deepcopy(dict(self.uns)),
            obsm={k: v.copy() for k, v in self.obsm.items()},
            varm={k: v.copy() for k, v in self.varm.items()},
        )
        instance._allow_X_assignment = False
        return instance

    def __getitem__(self, index) -> "LazyImputeAnndata":  # type: ignore
        """Return a sliced view as a :class:`LazyImputeAnndata`.

        Without this override, the base ``AnnData.__getitem__`` returns a plain
        ``AnnData`` view, losing the type and the ``X`` override.
        Since ``obsm`` and ``varm`` are both sliced automatically by AnnData
        along their respective axes, no manual fixup is needed.
        """
        oidx, vidx = self._normalize_indices(index)
        instance = LazyImputeAnndata.__new__(LazyImputeAnndata)
        instance._allow_X_assignment = True
        AnnData.__init__(instance, self, oidx=oidx, vidx=vidx, asview=True)
        instance._allow_X_assignment = False
        return instance

    def to_anndata(self) -> AnnData:
        """Return a plain :class:`~anndata.AnnData` suitable for serialization.

        mudata's HDF5 writer accesses ``adata.X`` directly, which on a
        :class:`LazyImputeAnndata` returns a :class:`LazyImpute` object with
        no registered writer. This method downcasts to a plain ``AnnData``
        with a fake sparse-zero ``X``, preserving ``obsm``, ``varm``, and
        ``uns`` so that :meth:`from_anndata` can reconstruct the
        :class:`LazyImputeAnndata` on load.
        """
        n_cells, n_regions = self.shape
        X_fake = scipy.sparse.csr_array((n_cells, n_regions), dtype=float)

        assert isinstance(self.obs, pd.DataFrame)
        assert isinstance(self.var, pd.DataFrame)

        return AnnData(
            X=X_fake,
            obs=self.obs,
            var=self.var,
            obsm=dict(self.obsm),
            varm=dict(self.varm),
            uns=dict(self.uns),
        )
