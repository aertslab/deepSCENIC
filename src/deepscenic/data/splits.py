"""Train/test split handling utilities for deepSCENIC."""

from typing import Literal

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd


def get_split(
    mdata: md.MuData,
    modality: Literal["rna", "atac"],
    cells: Literal["train", "test", "all"] = "all",
    features: Literal["train", "test", "all"] = "all",
) -> ad.AnnData:
    """
    Get a specific cell×feature split from MuData.

    Parameters
    ----------
    mdata
        Input multimodal data
    modality
        Which modality to extract
    cells
        Which cell split to use
    features
        Which feature split to use

    Returns
    -------
    Subset of the requested modality

    Examples
    --------
    >>> import deepscenic as ds
    >>> mdata = ds.read("dataset.h5mu")
    >>> rna_train = ds.data.get_split(mdata, "rna", cells="train", features="train")
    >>> rna_e2 = ds.data.get_split(mdata, "rna", cells="train", features="test")
    """
    adata = mdata.mod[modality]

    # Cell mask
    if cells == "all":
        cell_mask = np.ones(mdata.n_obs, dtype=bool)
    else:
        cell_mask = mdata.obs["split"] == cells

    # Feature mask
    if features == "all":
        feature_mask = np.ones(adata.n_vars, dtype=bool)
    else:
        feature_mask = adata.var["split"] == features

    return adata[cell_mask][:, feature_mask].copy()


class TrainingView:
    """
    Convenient view into MuData for training access patterns.

    Provides easy access to all 4 cell×feature combinations for each modality,
    plus R2G matrices aligned with the correct ordering.

    Parameters
    ----------
    mdata
        Input multimodal data (must have valid deepSCENIC schema)

    Attributes
    ----------
    rna_train
        Train cells × train genes (main training data)
    rna_test
        Test cells × test genes (held-out evaluation)
    rna_e2
        Train cells × test genes (E2 matrix learning)
    rna_eval
        Test cells × train genes (reconstruction evaluation)
    atac_train
        Train cells × train regions
    atac_test
        Test cells × test regions
    atac_e2
        Train cells × test regions
    atac_eval
        Test cells × train regions
    r2g_train
        R2G penalty matrix for train regions × train genes
    r2g_test
        R2G penalty matrix for test regions × test genes

    Examples
    --------
    >>> import deepscenic as ds
    >>> mdata = ds.read("dataset.h5mu")
    >>> view = ds.data.TrainingView(mdata)
    >>> view.rna_train.shape
    (31576, 12832)
    >>> view.r2g_train.shape
    (438728, 12832)
    """

    def __init__(self, mdata: md.MuData) -> None:
        self._mdata = mdata
        self._cache: dict = {}

    @property
    def mdata(self) -> md.MuData:
        """The underlying MuData object."""
        return self._mdata

    # RNA splits
    @property
    def rna_train(self) -> ad.AnnData:
        """Train cells × train genes."""
        if "rna_train" not in self._cache:
            self._cache["rna_train"] = get_split(self._mdata, "rna", cells="train", features="train")
        return self._cache["rna_train"]

    @property
    def rna_test(self) -> ad.AnnData:
        """Test cells × test genes (held-out)."""
        if "rna_test" not in self._cache:
            self._cache["rna_test"] = get_split(self._mdata, "rna", cells="test", features="test")
        return self._cache["rna_test"]

    @property
    def rna_e2(self) -> ad.AnnData:
        """Train cells × test genes (E2 learning)."""
        if "rna_e2" not in self._cache:
            self._cache["rna_e2"] = get_split(self._mdata, "rna", cells="train", features="test")
        return self._cache["rna_e2"]

    @property
    def rna_eval(self) -> ad.AnnData:
        """Test cells × train genes (reconstruction eval)."""
        if "rna_eval" not in self._cache:
            self._cache["rna_eval"] = get_split(self._mdata, "rna", cells="test", features="train")
        return self._cache["rna_eval"]

    # ATAC splits
    @property
    def atac_train(self) -> ad.AnnData:
        """Train cells × train regions."""
        if "atac_train" not in self._cache:
            self._cache["atac_train"] = get_split(self._mdata, "atac", cells="train", features="train")
        return self._cache["atac_train"]

    @property
    def atac_test(self) -> ad.AnnData:
        """Test cells × test regions (held-out)."""
        if "atac_test" not in self._cache:
            self._cache["atac_test"] = get_split(self._mdata, "atac", cells="test", features="test")
        return self._cache["atac_test"]

    @property
    def atac_e2(self) -> ad.AnnData:
        """Train cells × test regions."""
        if "atac_e2" not in self._cache:
            self._cache["atac_e2"] = get_split(self._mdata, "atac", cells="train", features="test")
        return self._cache["atac_e2"]

    @property
    def atac_eval(self) -> ad.AnnData:
        """Test cells × train regions."""
        if "atac_eval" not in self._cache:
            self._cache["atac_eval"] = get_split(self._mdata, "atac", cells="test", features="train")
        return self._cache["atac_eval"]

    # R2G matrices (computed on-demand from feature splits)
    @property
    def r2g_train(self):
        """R2G penalty matrix for train split (computed on-demand)."""
        if "r2g_train" not in self._cache:
            full_r2g = self._mdata.uns["r2g"]["matrix"]

            # Get boolean masks from feature splits
            train_region_mask = self._mdata["atac"].var["split"] == "train"
            train_gene_mask = self._mdata["rna"].var["split"] == "train"

            # Subset the sparse matrix
            self._cache["r2g_train"] = full_r2g[train_region_mask.values][:, train_gene_mask.values]
        return self._cache["r2g_train"]

    @property
    def r2g_test(self):
        """R2G penalty matrix for test split (computed on-demand)."""
        if "r2g_test" not in self._cache:
            full_r2g = self._mdata.uns["r2g"]["matrix"]

            # Get boolean masks from feature splits
            test_region_mask = self._mdata["atac"].var["split"] == "test"
            test_gene_mask = self._mdata["rna"].var["split"] == "test"

            # Subset the sparse matrix
            self._cache["r2g_test"] = full_r2g[test_region_mask.values][:, test_gene_mask.values]
        return self._cache["r2g_test"]

    # TF information
    @property
    def tf_names(self) -> list[str]:
        """List of TF names (in var_names order)."""
        rna = self._mdata.mod["rna"]
        return rna.var_names[rna.var["is_tf"]].tolist()

    @property
    def tf_mask(self) -> pd.Series:
        """Boolean mask for TFs in RNA var."""
        return self._mdata.mod["rna"].var["is_tf"]

    @property
    def n_tfs(self) -> int:
        """Number of TFs."""
        return len(self.tf_names)

    def clear_cache(self) -> None:
        """Clear cached split views to free memory."""
        self._cache.clear()
