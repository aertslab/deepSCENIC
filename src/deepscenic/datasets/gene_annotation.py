"""Gene annotation download utilities."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

import pandas as pd
import requests

if TYPE_CHECKING:
    pass

# Setup logging
log = logging.getLogger("deepscenic.datasets")

_NCBI_MAX_RETRIES = 3


class NCBIError(Exception):
    """Error fetching data from NCBI."""

    pass


def fetch_gene_annotation(
    species: str = "hsapiens",
    biomart_host: str = "http://www.ensembl.org",
    use_ucsc_chromosome_style: bool = True,
    transcript_type: str = "protein_coding",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Download gene annotation from Ensembl Biomart.

    Parameters
    ----------
    species : str, default="hsapiens"
        Species name for Ensembl (e.g., "hsapiens", "mmusculus", "dmelanogaster").
    biomart_host : str
        Biomart host URL. Use archived hosts for reproducibility:
        - "http://nov2020.archive.ensembl.org/" for GRCm38
        - "http://www.ensembl.org" for latest
    use_ucsc_chromosome_style : bool, default=True
        Convert chromosome names to UCSC style (chr1, chr2, etc.).
    transcript_type : str, default="protein_coding"
        Filter for transcript type. Set to None to include all types.

    Returns
    -------
    gene_annotation : DataFrame
        Gene annotation with columns:
        - Chromosome, Start, End, Strand, Transcription_Start_Site, Transcript_type
        Index is Gene name.
    chromsizes : DataFrame or None
        Chromosome sizes (if available from NCBI):
        - Chromosome, Start (0), End

    Examples
    --------
    >>> import deepscenic as ds
    >>> annot, chromsizes = ds.datasets.fetch_gene_annotation(
    ...     species="mmusculus",
    ...     biomart_host="http://nov2020.archive.ensembl.org/",
    ... )
    >>> annot.head()
       Chromosome  Start    End Strand  Transcription_Start_Site
    Gene
    0610005C13Rik  chr7  ...
    """
    import pybiomart as pbm

    dataset_name = f"{species}_gene_ensembl"

    log.info(f"Connecting to Biomart host: {biomart_host}")
    server = pbm.Server(host=biomart_host, use_cache=False)
    mart = server["ENSEMBL_MART_ENSEMBL"]

    if dataset_name not in mart.list_datasets()["name"].to_numpy():
        raise ValueError(f"Dataset '{dataset_name}' not found. " "Check species name or Biomart host.")

    dataset = mart[dataset_name]

    # Handle different Biomart attribute names across versions
    external_gene_name_query = (
        "external_gene_name" if "external_gene_name" in dataset.attributes.keys() else "hgnc_symbol"
    )
    tss_query = (
        "transcription_start_site" if "transcription_start_site" in dataset.attributes.keys() else "transcript_start"
    )

    log.info(f"Querying gene annotation for {species}...")
    annot = dataset.query(
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
        annot = annot[annot.Transcript_type == transcript_type].copy()

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

    def _get_with_retries(url: str, params: dict = None) -> requests.Response:
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
    annot = annot[annot["Chromosome"].isin(assembled_molecules)].copy()

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
