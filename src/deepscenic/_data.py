"""MuData schema validation and train/test split utilities for deepSCENIC."""

import warnings
from typing import Literal

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

__all__ = [
    "validate_schema",
    "is_valid_schema",
    "SchemaError",
    "SchemaWarning",
    "get_split",
    "TrainingView",
]


class SchemaError(Exception):
    """Raised when MuData doesn't conform to deepSCENIC schema."""

    pass


class SchemaWarning(UserWarning):
    """Warning for non-critical schema issues."""

    pass


def validate_schema(
    mdata: md.MuData,
    mode: str = "training",
    strict: bool = False,
) -> list[str]:
    """
    Validate MuData against deepSCENIC schema.

    Parameters
    ----------
    mdata
        Data to validate
    mode
        Validation mode:
        - 'training': require rna + atac + r2g (strict)
        - 'inference': require rna only (flexible)
    strict
        If True, raise errors. If False, collect warnings.

    Returns
    -------
    List of validation issues (empty if valid)

    Raises
    ------
    SchemaError
        If strict=True and validation fails

    Examples
    --------
    >>> import deepscenic as ds
    >>> # Validate for training (strict)
    >>> ds.validate_schema(mdata, mode="training", strict=True)  # doctest: +SKIP

    >>> # Validate for inference (RNA-only OK)
    >>> ds.validate_schema(mdata, mode="inference")  # doctest: +SKIP
    """
    issues = []

    if mode not in ("training", "inference"):
        raise ValueError(f"mode must be 'training' or 'inference', got: {mode}")

    # Check RNA modality (always required)
    if "rna" not in mdata.mod:
        issues.append("Missing 'rna' modality")

    # Check ATAC modality (required for training, optional for inference)
    if "atac" not in mdata.mod:
        if mode == "training":
            issues.append("Missing 'atac' modality (required for training)")

    if issues and strict:
        raise SchemaError(f"Schema validation failed: {issues}")

    # Check RNA modality
    if "rna" in mdata.mod:
        rna = mdata.mod["rna"]

        if "is_tf" not in rna.var.columns:
            issues.append("rna.var missing 'is_tf' column")
        elif rna.var["is_tf"].dtype != bool:
            issues.append("rna.var['is_tf'] should be bool")

    # Check ATAC modality
    if "atac" in mdata.mod:
        atac = mdata.mod["atac"]

        for col in ["chromosome", "start", "end"]:
            if col not in atac.var.columns:
                issues.append(f"atac.var missing '{col}' column")

        if "split" not in atac.var.columns:
            issues.append("atac.var missing 'split' column")

    # Check shared obs
    if "split" not in mdata.obs.columns:
        issues.append("mdata.obs missing 'split' column")

    # Check r2g (required for training, not for inference)
    if mode == "training":
        if "r2g" not in mdata.uns:
            issues.append("mdata.uns missing 'r2g' (required for training)")
        else:
            r2g = mdata.uns["r2g"]
            # Check required keys
            for key in ["config"]:
                if key not in r2g:
                    issues.append(f"mdata.uns['r2g'] missing '{key}'")

    if issues:
        if strict:
            raise SchemaError(f"Schema validation failed: {issues}")
        else:
            for issue in issues:
                warnings.warn(issue, SchemaWarning, stacklevel=2)

    return issues


def is_valid_schema(mdata: md.MuData, mode: str = "training") -> bool:
    """Check if MuData conforms to schema without raising."""
    issues = validate_schema(mdata, mode=mode, strict=False)
    return len(issues) == 0


def get_split(
    mdata: md.MuData,
    modality: Literal["rna", "atac"],
    cells: Literal["train", "test", "all"] = "all",
    features: Literal["train", "test", "all"] = "all",
) -> ad.AnnData:
    """
    Get a specific cell x feature split from MuData.

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
    >>> mdata = ds.read("dataset.h5mu")  # doctest: +SKIP
    >>> rna_train = ds.get_split(mdata, "rna", cells="train", features="train")  # doctest: +SKIP
    >>> rna_e2 = ds.get_split(mdata, "rna", cells="train", features="test")  # doctest: +SKIP
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

    Provides easy access to all 4 cell x feature combinations for each modality,
    plus R2G matrices aligned with the correct ordering.

    Parameters
    ----------
    mdata
        Input multimodal data (must have valid deepSCENIC schema)

    Attributes
    ----------
    rna_train
        Train cells x train genes (main training data)
    rna_test
        Test cells x test genes (held-out evaluation)
    rna_e2
        Train cells x test genes (E2 matrix learning)
    rna_eval
        Test cells x train genes (reconstruction evaluation)
    atac_train
        Train cells x train regions
    atac_test
        Test cells x test regions
    atac_e2
        Train cells x test regions
    atac_eval
        Test cells x train regions
    r2g_train
        R2G penalty matrix for train regions x train genes
    r2g_test
        R2G penalty matrix for test regions x test genes

    Examples
    --------
    >>> import deepscenic as ds
    >>> mdata = ds.read("dataset.h5mu")  # doctest: +SKIP
    >>> view = ds.TrainingView(mdata)  # doctest: +SKIP
    >>> view.rna_train.shape  # doctest: +SKIP
    (31576, 12832)
    >>> view.r2g_train.shape  # doctest: +SKIP
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
        """Train cells x train genes."""
        if "rna_train" not in self._cache:
            self._cache["rna_train"] = get_split(self._mdata, "rna", cells="train", features="train")
        return self._cache["rna_train"]

    @property
    def rna_test(self) -> ad.AnnData:
        """Test cells x test genes (held-out)."""
        if "rna_test" not in self._cache:
            self._cache["rna_test"] = get_split(self._mdata, "rna", cells="test", features="test")
        return self._cache["rna_test"]

    @property
    def rna_e2(self) -> ad.AnnData:
        """Train cells x test genes (E2 learning)."""
        if "rna_e2" not in self._cache:
            self._cache["rna_e2"] = get_split(self._mdata, "rna", cells="train", features="test")
        return self._cache["rna_e2"]

    @property
    def rna_eval(self) -> ad.AnnData:
        """Test cells x train genes (reconstruction eval)."""
        if "rna_eval" not in self._cache:
            self._cache["rna_eval"] = get_split(self._mdata, "rna", cells="test", features="train")
        return self._cache["rna_eval"]

    # ATAC splits
    @property
    def atac_train(self) -> ad.AnnData:
        """Train cells x train regions."""
        if "atac_train" not in self._cache:
            self._cache["atac_train"] = get_split(self._mdata, "atac", cells="train", features="train")
        return self._cache["atac_train"]

    @property
    def atac_test(self) -> ad.AnnData:
        """Test cells x test regions (held-out)."""
        if "atac_test" not in self._cache:
            self._cache["atac_test"] = get_split(self._mdata, "atac", cells="test", features="test")
        return self._cache["atac_test"]

    @property
    def atac_e2(self) -> ad.AnnData:
        """Train cells x test regions."""
        if "atac_e2" not in self._cache:
            self._cache["atac_e2"] = get_split(self._mdata, "atac", cells="train", features="test")
        return self._cache["atac_e2"]

    @property
    def atac_eval(self) -> ad.AnnData:
        """Test cells x train regions."""
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
