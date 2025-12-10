"""Dataset fetching utilities for deepSCENIC."""

from __future__ import annotations

import pooch

from .gene_annotation import fetch_gene_annotation

__all__ = ["fetch_tf_collection", "fetch_gene_annotation", "clear_cache", "get_cache_info"]

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


def clear_cache(pattern: str | None = None) -> list[str]:
    """
    Clear cached data files.

    Parameters
    ----------
    pattern : str, optional
        Glob pattern to match. If None, clears all cache.
        Example: "gene_annot_*" to clear only gene annotations.

    Returns
    -------
    list[str]
        List of deleted file paths.

    Examples
    --------
    >>> import deepscenic as ds
    >>> # Clear only gene annotation caches
    >>> ds.datasets.clear_cache("gene_annot_*")
    ['/home/user/.cache/deepscenic/gene_annot_mmusculus_abc123_ucsc_protein_coding.parquet']
    >>> # Clear all cache files
    >>> ds.datasets.clear_cache()
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
    >>> info = ds.datasets.get_cache_info()
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
                files.append({
                    "name": f.name,
                    "size_mb": f.stat().st_size / (1024 * 1024),
                })

    return {
        "cache_dir": str(cache_dir),
        "files": files,
        "total_size_mb": sum(f["size_mb"] for f in files),
    }
