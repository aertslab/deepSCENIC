# Datasets (`ds.datasets`)

Functions for fetching external data resources.

## TF Collections

```{eval-rst}
.. autofunction:: deepscenic.datasets.fetch_tf_collection
```

## Gene Annotations

```{eval-rst}
.. autofunction:: deepscenic.datasets.fetch_gene_annotation
```

## Usage Example

```python
import deepscenic as ds

# Fetch mouse TF list from SCENIC+ resources
tfs = ds.datasets.fetch_tf_collection(species="mouse")
print(f"Found {len(tfs)} transcription factors")

# Fetch gene annotations from Ensembl
annot, chromsizes = ds.datasets.fetch_gene_annotation(
    species="mmusculus",
    biomart_host="http://nov2020.archive.ensembl.org/",
)
```
