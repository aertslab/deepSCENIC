# Datasets

Functions for fetching external data resources (TF collections, gene annotations).

## TF Collections

```{eval-rst}
.. autofunction:: deepscenic.fetch_tf_collection
```

## Gene Annotations

```{eval-rst}
.. autofunction:: deepscenic.fetch_gene_annotation
```

## Cache Management

```{eval-rst}
.. autofunction:: deepscenic.get_cache_info
.. autofunction:: deepscenic.clear_cache
```

## Usage

```python
import deepscenic as ds

# Fetch mouse TF list from SCENIC+ resources
tfs = ds.fetch_tf_collection(species="mouse")
print(f"Found {len(tfs)} transcription factors")

# Fetch gene annotations from Ensembl
annot, chromsizes = ds.fetch_gene_annotation(
    species="mmusculus",
    biomart_host="http://nov2020.archive.ensembl.org/",
)

# Check cache status
ds.get_cache_info()

# Clear cached data
ds.clear_cache()
```
