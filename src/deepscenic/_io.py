"""I/O functions for deepSCENIC data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from ._data import validate_schema

if TYPE_CHECKING:
    pass

__all__ = ["read", "read_bed", "write"]

log = logging.getLogger("deepscenic.io")

md.set_options(pull_on_update=False)  # adopt new mudata behaviour

# Enable nullable string serialization (supported in anndata >= 0.11)
ad.settings.allow_write_nullable_strings = True


def read(
    path: str | Path,
    validate: bool = True,
    strict: bool = False,
    backed: str | bool | None = None,
) -> md.MuData:
    """
    Read a deepSCENIC MuData file.

    Wrapper around mudata.read() with schema validation for required deepSCENIC fields.

    Parameters
    ----------
    path
        Path to .h5mu file.
    validate
        Whether to validate schema after loading.
    strict
        Whether to log warnings (strict: False) or raise Exception when not following schema.
    backed
        Load in backed mode for memory efficiency.
        See anndata/mudata official documentation for more info.

    Returns
    -------
    Loaded and optionally validated MuData.

    Examples
    --------
    >>> import deepscenic as ds
    >>> mdata = ds.read("my_dataset.h5mu")  # doctest: +SKIP
    >>> mdata  # doctest: +SKIP
    MuData object with n_obs x n_vars = 39470 x 455165
      rna: 39470 x 16437
      atac: 39470 x 438728
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix != ".h5mu":
        raise ValueError(f"Expected .h5mu file, got: {path.suffix}")

    mdata = md.read(path, backed=backed)
    if isinstance(mdata, ad.AnnData):
        raise ValueError("Expected MuData format, got Anndata")

    if validate:
        validate_schema(mdata, strict=strict)

    log.info(f"Loaded {path}: {mdata.n_obs} cells, {mdata['rna'].n_vars} genes, {mdata['atac'].n_vars} regions")

    return mdata


def _convert_nullable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object-dtype columns with NA to proper nullable pandas dtypes.

    h5py cannot serialize object-dtype columns containing pd.NA. This function
    converts them to proper nullable dtypes (string, Int64, Float64) which
    AnnData can serialize correctly.

    Parameters
    ----------
    df
        DataFrame to convert.

    Returns
    -------
    DataFrame with converted columns.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if not df[col].isna().any():  # type: ignore
            continue

        non_null = df[col].dropna()
        if len(non_null) == 0:
            # All NA - default to string dtype
            df[col] = df[col].astype("string")
            continue

        first_val = non_null.iloc[0]
        if isinstance(first_val, str):
            df[col] = df[col].astype("string")
        elif isinstance(first_val, (int, np.integer)):
            df[col] = df[col].astype("Int64")
        elif isinstance(first_val, (float, np.floating)):
            df[col] = df[col].astype("Float64")
        elif isinstance(first_val, (bool, np.bool_)):
            df[col] = df[col].astype("boolean")

    return df


def write(
    mdata: md.MuData,
    path: str | Path,
    validate: bool = True,
    strict: bool = True,
    compression: str = "gzip",
) -> None:
    """
    Write MuData to file.

    Parameters
    ----------
    mdata
        Data to write.
    path
        Output path (should end in .h5mu).
    validate
        Validate schema before writing.
    strict
        Use strict validation (raise Exceptions instead of logging warnings).
    compression
        Compression for h5 file.

    Examples
    --------
    >>> import deepscenic as ds
    >>> ds.write(mdata, "processed_dataset.h5mu")  # doctest: +SKIP
    """
    path = Path(path)

    if validate:
        validate_schema(mdata, strict=strict)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Create a copy to avoid modifying the original
    mdata = mdata.copy()

    # Convert object-dtype columns with pd.NA to proper nullable dtypes
    mdata.obs = _convert_nullable_columns(mdata.obs)
    for mod in mdata.mod.values():
        mod.var = _convert_nullable_columns(mod.var)
        mod.obs = _convert_nullable_columns(mod.obs)

    mdata.write(str(path), compression=compression)

    log.info(f"Wrote {path}: {mdata.n_obs} cells")


def read_bed(
    bed_file: str | Path,
    chromsizes: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """
    Read genomic regions from BED file.

    Reads a BED format file (tab-separated) and returns a DataFrame with
    parsed genomic coordinates. Optionally validates regions against
    chromosome sizes to filter invalid regions.

    Parameters
    ----------
    bed_file
        Path to BED file (tab-separated: chrom, start, end).
        Additional columns beyond the first 3 are ignored.
    chromsizes
        Chromosome sizes for validation. Can be:

        - pd.DataFrame with columns: 'Chromosome', 'Start', 'End'
          (e.g., from ``ds.fetch_gene_annotation()``)
        - Path to chromsizes file (tab-separated: chrom, size)
        - None to skip validation (default)

        When provided, regions are validated and filtered if:

        - Chromosome not in chromsizes
        - Start position < 0
        - End position > chromosome length

    Returns
    -------
    pd.DataFrame
        Regions with columns:

        - chromosome (str): Chromosome name
        - start (int): Start position (0-based)
        - end (int): End position (exclusive)
        - region (str): Formatted as "chr:start-end"

    Raises
    ------
    FileNotFoundError
        If BED file doesn't exist.
    ValueError
        If BED file has fewer than 3 columns or all regions are filtered.

    Examples
    --------
    Basic usage without validation:

    >>> import deepscenic as ds
    >>> regions = ds.read_bed("peaks.bed")  # doctest: +SKIP
    >>> regions.head()  # doctest: +SKIP
       chromosome  start    end          region
    0        chr1   1000   2000  chr1:1000-2000
    1        chr1   3000   4000  chr1:3000-4000

    With chromosome validation:

    >>> annot, chromsizes = ds.fetch_gene_annotation("mmusculus")  # doctest: +SKIP
    >>> regions = ds.read_bed("peaks.bed", chromsizes=chromsizes)  # doctest: +SKIP
    # Regions on unknown chromosomes or out of bounds are filtered

    Get list of region strings:

    >>> regions_list = regions["region"].tolist()  # doctest: +SKIP
    >>> len(regions_list)  # doctest: +SKIP
    1234
    """
    bed_file = Path(bed_file)

    # Check file exists
    if not bed_file.is_file():
        raise FileNotFoundError(f"BED file not found: {bed_file}")

    # Read BED file (first 3 columns only)
    try:
        regions = pd.read_csv(
            bed_file,
            sep="\t",
            header=None,
            usecols=(0, 1, 2),  # type: ignore[arg-type]
            dtype={0: str, 1: "Int32", 2: "Int32"},
            names=["chromosome", "start", "end"],
        )
    except Exception as e:
        raise ValueError(f"Error reading BED file {bed_file}: {e}") from e

    if regions.empty:
        raise ValueError(f"BED file is empty: {bed_file}")

    # Create region string column
    regions["region"] = (
        regions["chromosome"].astype(str) + ":" + regions["start"].astype(str) + "-" + regions["end"].astype(str)
    )

    n_total = len(regions)

    # Basic coordinate validation (always performed)
    valid_coords = (regions["start"] >= 0) & (regions["start"] < regions["end"])
    n_invalid_coords = (~valid_coords).sum()

    if n_invalid_coords > 0:
        log.warning(f"Filtered {n_invalid_coords} regions with invalid coordinates (start < 0 or start >= end)")
        regions = regions[valid_coords].copy()

    # Chromsizes validation (optional)
    if chromsizes is not None:
        # Load chromsizes if it's a file path
        if isinstance(chromsizes, str | Path):
            chromsizes = _load_chromsizes_file(Path(chromsizes))

        # Build chromosome size dict
        chromsizes_dict = dict(zip(chromsizes["Chromosome"], chromsizes["End"], strict=False))

        # Filter regions: chromosome must exist in chromsizes
        valid_mask = regions.apply(
            lambda row: (
                row["chromosome"] in chromsizes_dict
                and row["start"] >= 0
                and row["end"] <= chromsizes_dict[row["chromosome"]]
            ),
            axis=1,
        )

        n_filtered = (~valid_mask).sum()
        regions = regions[valid_mask].copy()

        if n_filtered > 0:
            log.warning(f"Filtered {n_filtered} regions (out of {n_total}) that are not within known chromosome bounds")

    # Check if all regions were filtered
    if len(regions) == 0:
        raise ValueError(
            f"All regions in {bed_file} were filtered out. "
            f"This likely indicates a chromosome name mismatch between your BED file "
            f"and chromsizes (e.g., 'chr1' vs '1'). Check your BED file format."
        )

    # Reset index for clean output
    regions = regions.reset_index(drop=True)

    return regions


def _load_chromsizes_file(chromsizes_file: Path) -> pd.DataFrame:
    """
    Load chromosome sizes from file.

    Parameters
    ----------
    chromsizes_file
        Path to chromsizes file (tab-separated: chrom, size).

    Returns
    -------
    pd.DataFrame
        Chromosome sizes with columns: Chromosome, Start (0), End.
    """
    if not chromsizes_file.is_file():
        raise FileNotFoundError(f"Chromsizes file not found: {chromsizes_file}")

    try:
        # Read chromsizes file (2 columns: chrom, size)
        df = pd.read_csv(
            chromsizes_file,
            sep="\t",
            header=None,
            names=["Chromosome", "End"],
            dtype={"Chromosome": str, "End": "Int32"},
        )
        df["Start"] = 0

        # Reorder columns to match fetch_gene_annotation format
        chromsizes = pd.DataFrame(df[["Chromosome", "Start", "End"]])

    except Exception as e:
        raise ValueError(f"Error reading chromsizes file {chromsizes_file}: {e}") from e

    return chromsizes
