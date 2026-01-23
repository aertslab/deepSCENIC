"""Dataset fetching utilities for deepSCENIC."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import pooch
import requests

__all__ = ["fetch_tf_collection", "fetch_gene_annotation", "clear_cache", "get_cache_info"]

log = logging.getLogger("deepscenic.datasets")
_NCBI_MAX_RETRIES = 3

# TF collection URLs (SCENIC+ resources)
TF_COLLECTION_URLS = {
    "human": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt",
    "mouse": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_mm.txt",
    "fly": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_dmel.txt",
}

# Known checksums (None allows download without verification)
TF_COLLECTION_CHECKSUMS = {
    "human": "3953034f84112c60d3d8ef15b0e0c8ac5fce0b40d2c7c0824c2945c70cee2523",
    "mouse": "17a95e142147fb7dc063d7b9e84262746b0b64f622793b3cc5df0eddf2f1194c",
    "fly": "20d7e11540b595dda3ed133f86af559c9fc708810dc39d4bef106961fedf348d",
}


class NCBIError(Exception):
    """Error fetching data from NCBI."""

    pass


def _get_cache_dir() -> Path:
    """Get the deepscenic cache directory."""
    cache_dir: Path = pooch.os_cache("deepscenic")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_key(
    species: str,
    biomart_host: str,
    use_ucsc_chromosome_style: bool,
    transcript_type: str | None,
) -> str:
    """Generate a unique cache key for the query parameters."""
    # Hash the host to keep filename reasonable
    host_hash = hashlib.md5(biomart_host.encode()).hexdigest()[:8]
    ucsc = "ucsc" if use_ucsc_chromosome_style else "ensembl"
    ttype = transcript_type or "all"
    return f"gene_annot_{species}_{host_hash}_{ucsc}_{ttype}"


def fetch_tf_collection(
    species: Literal["human", "mouse", "fly"],
) -> list[str]:
    """
    Fetch SCENIC+ transcription factor collection.

    Downloads and caches the TF list for the specified species from
    the SCENIC+ resources at aertslab.org.

    Parameters
    ----------
    species
        Species to fetch TF list for.

    Returns
    -------
    List of transcription factor gene names.

    Examples
    --------
    >>> import deepscenic as ds
    >>> tfs = ds.fetch_tf_collection(species="mouse")
    >>> len(tfs)
    1390
    >>> tfs[:3]
    ['Adnp', 'Aebp1', 'Aebp2']
    """
    if species not in TF_COLLECTION_URLS:
        raise ValueError(f"Unknown species: {species}. Available: {list(TF_COLLECTION_URLS.keys())}")

    url = TF_COLLECTION_URLS[species]
    known_hash = TF_COLLECTION_CHECKSUMS[species]

    cache_dir = pooch.os_cache("deepscenic")
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = pooch.retrieve(
        url=url,
        known_hash=known_hash,
        path=cache_dir,
        fname=f"allTFs_{species}.txt",
        progressbar=True,
    )

    # Read TF list
    with open(local_path) as f:
        tf_names = [line.strip() for line in f if line.strip()]

    return tf_names


def fetch_gene_annotation(
    species: str = "hsapiens",
    biomart_host: str = "http://www.ensembl.org",
    use_ucsc_chromosome_style: bool = True,
    transcript_type: str = "protein_coding",
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Download gene annotation from Ensembl Biomart.

    Results are cached locally as parquet files for fast subsequent access.
    After the first download, this function loads from cache without
    making network requests.

    Parameters
    ----------
    species
        Species name for Ensembl (e.g., "hsapiens", "mmusculus", "dmelanogaster").
    biomart_host
        Biomart host URL. Use archived hosts for reproducibility:
        - "http://nov2020.archive.ensembl.org/" for GRCm38
        - "http://www.ensembl.org" for latest
    use_ucsc_chromosome_style
        Convert chromosome names to UCSC style (chr1, chr2, etc.).
    transcript_type
        Filter for transcript type. Set to None to include all types.
    force_download
        Force re-download even if cached data exists.

    Returns
    -------
    gene_annotation
        Gene annotation with columns:
        - Chromosome, Start, End, Strand, Transcription_Start_Site, Transcript_type
        Index is Gene name.
    chromsizes
        Chromosome sizes (if available from NCBI):
        - Chromosome, Start (0), End

    Examples
    --------
    >>> import deepscenic as ds
    >>> annot, chromsizes = ds.fetch_gene_annotation(
    ...     species="mmusculus",
    ...     biomart_host="http://nov2020.archive.ensembl.org/",
    ... )
    >>> annot.head()
       Chromosome  Start    End Strand  Transcription_Start_Site
    Gene
    0610005C13Rik  chr7  ...
    """
    cache_dir = _get_cache_dir()
    cache_key = _get_cache_key(species, biomart_host, use_ucsc_chromosome_style, transcript_type)

    annot_path = cache_dir / f"{cache_key}.parquet"
    chromsizes_path = cache_dir / f"{cache_key}_chromsizes.parquet"

    # Try to load from cache
    if not force_download and annot_path.exists():
        log.info(f"Loading cached gene annotation from {annot_path}")
        annot = pd.read_parquet(annot_path)
        chromsizes = pd.read_parquet(chromsizes_path) if chromsizes_path.exists() else None
        log.info(f"Loaded annotation for {len(annot)} genes from cache")
        return annot, chromsizes

    # Fetch from Biomart (this imports pybiomart)
    log.info(f"Fetching gene annotation from {biomart_host}...")
    annot, chromsizes = _fetch_from_biomart(
        species=species,
        biomart_host=biomart_host,
        use_ucsc_chromosome_style=use_ucsc_chromosome_style,
        transcript_type=transcript_type,
    )

    # Cache results as parquet
    log.info(f"Caching gene annotation to {annot_path}")
    annot.to_parquet(annot_path)
    if chromsizes is not None:
        chromsizes.to_parquet(chromsizes_path)

    return annot, chromsizes


def _fetch_from_biomart(
    species: str,
    biomart_host: str,
    use_ucsc_chromosome_style: bool,
    transcript_type: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Fetch gene annotation from Biomart.

    This function imports pybiomart, which creates a .pybiomart.sqlite file.
    It is isolated here so the main fetch_gene_annotation() can avoid
    importing pybiomart when loading from cache.

    The requests_cache is redirected to our cache directory to prevent
    .pybiomart.sqlite appearing in the user's working directory.
    """
    # Redirect pybiomart's cache to our cache directory BEFORE importing
    import requests_cache

    cache_dir = _get_cache_dir()
    _original_install_cache = requests_cache.install_cache

    def _patched_install_cache(cache_name: str = "http_cache", **kwargs):
        if cache_name == ".pybiomart":
            cache_name = str(cache_dir / "pybiomart_requests")
        return _original_install_cache(cache_name, **kwargs)

    requests_cache.install_cache = _patched_install_cache  # type: ignore[assignment]

    # Now import pybiomart (will use our patched install_cache)
    import pybiomart as pbm

    dataset_name = f"{species}_gene_ensembl"

    log.info(f"Connecting to Biomart host: {biomart_host}")
    server = pbm.Server(host=biomart_host, use_cache=False)
    mart = server["ENSEMBL_MART_ENSEMBL"]

    if dataset_name not in mart.list_datasets()["name"].to_numpy():
        raise ValueError(f"Dataset '{dataset_name}' not found. Check species name or Biomart host.")

    dataset = mart[dataset_name]

    # Handle different Biomart attribute names across versions
    external_gene_name_query = (
        "external_gene_name" if "external_gene_name" in dataset.attributes.keys() else "hgnc_symbol"
    )
    tss_query = (
        "transcription_start_site" if "transcription_start_site" in dataset.attributes.keys() else "transcript_start"
    )

    log.info(f"Querying gene annotation for {species}...")
    annot = pd.DataFrame(
        dataset.query(
            attributes=[
                "chromosome_name",
                "start_position",
                "end_position",
                "strand",
                external_gene_name_query,
                tss_query,
                "transcript_biotype",
            ]
        )
    )
    annot.columns = [
        "Chromosome",
        "Start",
        "End",
        "Strand",
        "Gene",
        "Transcription_Start_Site",
        "Transcript_type",
    ]

    # Filter for transcript type
    if transcript_type:
        annot = pd.DataFrame(annot[annot.Transcript_type == transcript_type].copy())

    # Convert strand from numeric to +/-
    annot["Strand"] = ["+" if strand == 1 else "-" for strand in annot["Strand"]]

    # Try to get chromosome sizes from NCBI
    chromsizes = None
    try:
        chromsizes, annot = _fetch_chromsizes_and_filter(annot, dataset, use_ucsc_chromosome_style)
    except NCBIError as e:
        log.warning(f"Could not fetch chromosome info from NCBI: {e}")
        log.warning("Returning annotation without chromosome filtering.")

    # Remove duplicate genes (keep first occurrence)
    annot = annot.drop_duplicates(subset="Gene", keep="first")

    # Remove rows with empty gene names
    annot = annot[annot["Gene"].notna() & (annot["Gene"] != "")]

    # Set Gene as index for easy lookup
    annot = annot.set_index("Gene")

    log.info(f"Downloaded annotation for {len(annot)} genes")

    return annot, chromsizes


def _fetch_chromsizes_and_filter(
    annot: pd.DataFrame,
    dataset,
    use_ucsc_style: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch chromosome sizes from NCBI and filter annotation."""
    import xml.etree.ElementTree as xml_tree

    # Get assembly name from dataset
    regex_display = re.search(r"\((.*?)\)", dataset.display_name)
    if regex_display is None:
        raise NCBIError("Could not find assembly from Biomart display name")

    ncbi_search_term = regex_display.group(1)
    log.info(f"Using genome assembly: {ncbi_search_term}")

    def _get_with_retries(url: str, params: dict | None = None) -> requests.Response:
        for _ in range(_NCBI_MAX_RETRIES):
            resp = requests.get(url, params=params)
            if resp.ok:
                return resp
            time.sleep(0.5)
        raise NCBIError(f"Failed to fetch from {url} after {_NCBI_MAX_RETRIES} retries")

    # Search NCBI assembly database
    esearch_resp = _get_with_retries(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "assembly", "term": f"{ncbi_search_term}[Assembly Name]"},
    )

    id_list = xml_tree.fromstring(esearch_resp.content).find("IdList")
    id_elem = id_list.find("Id") if id_list is not None else None

    if id_elem is None:
        raise NCBIError(f"No assembly found for: {ncbi_search_term}")

    assembly_id = id_elem.text
    log.info(f"Found NCBI assembly ID: {assembly_id}")

    # Get assembly summary
    esummary_resp = _get_with_retries(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "assembly", "id": assembly_id},
    )

    doc_summary = xml_tree.fromstring(esummary_resp.content).find(".//DocumentSummary")
    if doc_summary is None:
        raise NCBIError("No DocumentSummary in NCBI response")

    ftp_path = doc_summary.find("FtpPath_Assembly_rpt")
    if ftp_path is None or ftp_path.text is None:
        raise NCBIError("No FTP path for assembly report")

    log.info(f"Downloading assembly report from: {ftp_path.text}")

    # Load assembly report
    assembly_report = pd.read_csv(
        ftp_path.text,
        comment="#",
        names=[
            "Sequence-Name",
            "Sequence-Role",
            "Assigned-Molecule",
            "Assigned-Molecule-Location/Type",
            "GenBank-Accn",
            "Relationship",
            "RefSeq-Accn",
            "Assembly-Unit",
            "Sequence-Length",
            "UCSC-style-name",
        ],
        sep="\t",
    )

    # Filter to assembled chromosomes only
    assembly_report = assembly_report[assembly_report["Sequence-Role"] == "assembled-molecule"].copy()

    assembled_molecules = assembly_report["Sequence-Name"].tolist()
    log.info(f"Found {len(assembled_molecules)} assembled chromosomes")

    # Filter annotation to assembled chromosomes
    annot = pd.DataFrame(annot[annot["Chromosome"].isin(assembled_molecules)].copy())

    # Build chromsizes DataFrame
    chromsizes = pd.DataFrame(
        {
            "Chromosome": assembly_report["Sequence-Name"].tolist(),
            "Start": 0,
            "End": assembly_report["Sequence-Length"].tolist(),
        }
    )

    # Convert to UCSC style if requested
    if use_ucsc_style:
        ensembl_to_ucsc = dict(
            zip(
                assembly_report["Sequence-Name"],
                assembly_report["UCSC-style-name"],
                strict=False,
            )
        )
        annot["Chromosome"] = [ensembl_to_ucsc.get(c, c) for c in annot["Chromosome"]]
        chromsizes["Chromosome"] = [ensembl_to_ucsc.get(c, c) for c in chromsizes["Chromosome"]]
        log.info("Converted chromosome names to UCSC style")

    return chromsizes, annot


def clear_cache(pattern: str | None = None) -> list[str]:
    """
    Clear cached data files.

    Parameters
    ----------
    pattern
        Glob pattern to match. If None, clears all cache.
        Example: "gene_annot_*" to clear only gene annotations.

    Returns
    -------
    List of deleted file paths.

    Examples
    --------
    >>> import deepscenic as ds
    >>> # Clear only gene annotation caches
    >>> ds.clear_cache("gene_annot_*")
    ['/home/user/.cache/deepscenic/gene_annot_mmusculus_abc123_ucsc_protein_coding.parquet']
    >>> # Clear all cache files
    >>> ds.clear_cache()
    []
    """
    cache_dir = pooch.os_cache("deepscenic")
    if not cache_dir.exists():
        return []

    if pattern is None:
        pattern = "*"

    deleted = []
    for f in cache_dir.glob(pattern):
        if f.is_file():
            f.unlink()
            deleted.append(str(f))

    return deleted


def get_cache_info() -> dict:
    """
    Get information about cached files.

    Returns
    -------
    dict
        Dictionary containing:
        - cache_dir: Path to cache directory
        - files: List of dicts with name and size_mb
        - total_size_mb: Total size of all cached files

    Examples
    --------
    >>> import deepscenic as ds
    >>> info = ds.get_cache_info()
    >>> info['cache_dir']
    '/home/user/.cache/deepscenic'
    >>> info['total_size_mb']
    5.2
    """
    cache_dir = pooch.os_cache("deepscenic")
    files = []
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file():
                files.append(
                    {
                        "name": f.name,
                        "size_mb": f.stat().st_size / (1024 * 1024),
                    }
                )

    return {"cache_dir": str(cache_dir), "files": files, "total_size_mb": sum(f["size_mb"] for f in files)}
