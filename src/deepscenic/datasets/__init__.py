"""Dataset fetching utilities for deepSCENIC."""

from __future__ import annotations

import pooch

from .gene_annotation import fetch_gene_annotation

__all__ = ["fetch_tf_collection", "fetch_gene_annotation"]

# TF collection URLs (SCENIC+ resources)
TF_COLLECTION_URLS = {
    "human": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt",
    "mouse": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_mm.txt",
    "fly": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_dmel.txt",
}

# Known checksums (None allows download without verification)
TF_COLLECTION_CHECKSUMS = {
    "human": None,
    "mouse": None,
    "fly": None,
}


def fetch_tf_collection(
    species: str = "human",
) -> list[str]:
    """
    Fetch SCENIC+ transcription factor collection.

    Downloads and caches the TF list for the specified species from
    the SCENIC+ resources at aertslab.org.

    Parameters
    ----------
    species : {'human', 'mouse', 'fly'}
        Species to fetch TF list for.

    Returns
    -------
    list[str]
        List of transcription factor gene names.

    Examples
    --------
    >>> import deepscenic as ds
    >>> tfs = ds.datasets.fetch_tf_collection(species="mouse")
    >>> len(tfs)
    1390
    >>> tfs[:3]
    ['Adnp', 'Aebp1', 'Aebp2']
    """
    if species not in TF_COLLECTION_URLS:
        raise ValueError(f"Unknown species: {species}. Available: {list(TF_COLLECTION_URLS.keys())}")

    url = TF_COLLECTION_URLS[species]
    known_hash = TF_COLLECTION_CHECKSUMS[species]

    # Use pooch to download/cache
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
