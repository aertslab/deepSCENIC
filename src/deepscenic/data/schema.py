"""MuData schema validation for deepSCENIC."""

import warnings

import mudata as md


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
    >>> # Validate for training (strict)
    >>> validate_schema(mdata, mode="training", strict=True)

    >>> # Validate for inference (RNA-only OK)
    >>> validate_schema(mdata, mode="inference")
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
