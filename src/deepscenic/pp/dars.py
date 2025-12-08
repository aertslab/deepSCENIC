"""DAR (Differentially Accessible Regions) handling."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from mudata import MuData


def mark_dars(
    mdata: MuData,
    dar_dir: str | Path | None = None,
    dar_dict: dict[str, list[str]] | None = None,
    celltype_key: str | None = None,
    inplace: bool = True,
) -> MuData | None:
    """
    Mark Differentially Accessible Regions in ATAC modality.

    DARs can be loaded from a directory of BED files (pyCisTopic output)
    or provided directly as a dictionary. DARs are typically more
    informative for training and may be upweighted during sampling.

    Parameters
    ----------
    mdata : MuData
        Input multimodal data with 'atac' modality.
    dar_dir : path, optional
        Directory containing DAR BED files (one per cell type).
        Files should be named: {celltype}.bed
        BED format: chrom, start, end (tab-separated, no header).
    dar_dict : dict, optional
        Dictionary mapping cell type names to lists of region names.
        Region names should match mdata.mod['atac'].var_names format.
    celltype_key : str, optional
        Key in obs for cell type labels. Currently unused, reserved for
        future per-celltype DAR tracking.
    inplace : bool, default=True
        Whether to modify mdata in-place.

    Returns
    -------
    MuData or None
        If inplace=False, returns modified MuData.

    Notes
    -----
    DARs are typically identified using pyCisTopic's differential
    accessibility analysis. This function simply marks regions as DARs
    based on external files.

    Examples
    --------
    >>> # From pyCisTopic BED files
    >>> ds.pp.mark_dars(mdata, dar_dir="/path/to/DARs/")

    >>> # From dictionary
    >>> dars = {"Astro": ["chr1:1000-2000", ...], "Oligo": [...]}
    >>> ds.pp.mark_dars(mdata, dar_dict=dars)

    >>> # Check result
    >>> mdata.mod['atac'].var['is_dar'].sum()
    5000
    """
    if not inplace:
        mdata = mdata.copy()

    if "atac" not in mdata.mod:
        raise ValueError("MuData must contain 'atac' modality")

    atac = mdata.mod["atac"]

    # Initialize DAR column
    atac.var["is_dar"] = False

    # Get DARs from either source
    if dar_dir is not None:
        dar_dict = _load_dars_from_dir(Path(dar_dir))
    elif dar_dict is None:
        raise ValueError("Must provide either dar_dir or dar_dict")

    # Collect all unique DAR regions
    all_dars = set()
    for _celltype, regions in dar_dict.items():
        all_dars.update(regions)

    # Mark DARs that exist in our data
    dars_in_data = all_dars & set(atac.var_names)
    atac.var.loc[list(dars_in_data), "is_dar"] = True

    n_marked = atac.var["is_dar"].sum()
    n_provided = len(all_dars)
    n_missing = n_provided - len(dars_in_data)

    print(f"Marked {n_marked} / {len(atac.var)} regions as DARs")
    if n_missing > 0:
        print(f"  ({n_missing} DAR regions not found in data)")

    if not inplace:
        return mdata
    return None


def _load_dars_from_dir(dar_dir: Path) -> dict[str, list[str]]:
    """
    Load DARs from directory of BED files.

    Parameters
    ----------
    dar_dir : Path
        Directory containing .bed files.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping celltype name to list of region names.
    """
    dar_dict = {}

    for fname in os.listdir(dar_dir):
        if not fname.endswith(".bed"):
            continue

        celltype = fname.replace(".bed", "")
        fpath = dar_dir / fname

        # Read BED file (chrom, start, end)
        df = pd.read_csv(
            fpath,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["Chromosome", "Start", "End"],
        )

        # Convert to region names (matching ATAC var_names format)
        regions = [f"{row.Chromosome}:{row.Start}-{row.End}" for _, row in df.iterrows()]

        dar_dict[celltype] = regions

    if not dar_dict:
        raise ValueError(f"No .bed files found in {dar_dir}")

    return dar_dict


def load_dars_from_bed(
    bed_file: str | Path,
) -> list[str]:
    """
    Load region names from a single BED file.

    Parameters
    ----------
    bed_file : path
        Path to BED file (chrom, start, end format).

    Returns
    -------
    list[str]
        List of region names in 'chr:start-end' format.

    Examples
    --------
    >>> regions = ds.pp.load_dars_from_bed("my_regions.bed")
    >>> len(regions)
    1234
    """
    df = pd.read_csv(
        bed_file,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["Chromosome", "Start", "End"],
    )

    regions = [f"{row.Chromosome}:{row.Start}-{row.End}" for _, row in df.iterrows()]

    return regions
