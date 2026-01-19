---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Tutorial 1: Data Preparation

This tutorial covers loading, preprocessing, and saving multimodal data for deepSCENIC.


## Setup

First, let's import the required packages:

```python
import anndata as ad
import scanpy as sc

import deepscenic as ds

print(f"deepSCENIC version: {ds.__version__}")
```

## Download Sample Data

For this tutorial, we'll use the scenicplus MM_lines dataset (melanoma cell lines).

```{note}
Preprocessed sample datasets are coming soon. For now, provide your own scATAC and scRNA datasets to follow the tutorial.  
```

```python
# For now, set your data directory manually:
import os
data_dir = "../../data/deepSCENIC/MM_lines/"
assert os.path.exists(data_dir), "dir not found"
```

## scRNA and scATAC Preprocessing

deepSCENIC combines RNA expression data, chromatin accessibility data, and genomic sequences to learn gene regulatory networks at the single-cell level. deepSCENIC follows the `scverse` convention for data formatting, and therefore we expect that your data is already preprocessed into the `anndata.AnnData` format. We will further preprocess these data modalities separately and later combine them into one `mudata.MuData` object, which is the format that deepSCENIC uses to run both training and inference.


### scRNA-seq Preprocessing

Load the gene expression matrix:

```python
# Load scRNA-seq data
adata_rna = ad.read_h5ad(f"{data_dir}/raw_exprMat.h5ad")
print(f"RNA data: {adata_rna.n_obs} cells x {adata_rna.n_vars} genes")
```

We perform some basic filtering and normalization using standard `scanpy` functionality.

```python
# Standard scRNA-seq preprocessing with scanpy
# (deepSCENIC expects log-normalized RNA data)
sc.pp.filter_genes(adata_rna, min_cells=10)
sc.pp.normalize_total(adata_rna)
sc.pp.log1p(adata_rna)
```

Additionally, we remove zero-variance genes (if there are any). These are genes that have the same constant expression across all cells (e.g. 0), which means they'll likely be completely uninformative for training.

```python
ds.pp.remove_zero_variance_genes(adata_rna)
```

#### Fetch External Resources

Before further preprocessing, we need to fetch transcription factor lists and gene annotations from external databases.

```python
# Fetch TF list from SCENIC+ resources
tfs = ds.datasets.fetch_tf_collection(species="human")  # also available: "mouse", "fly"
print(f"Found {len(tfs)} transcription factors")
tfs[:10]
```

```python
# Fetch gene annotations from Ensembl
annot, chromsizes = ds.datasets.fetch_gene_annotation(
    species="hsapiens",
    biomart_host = "http://www.ensembl.org", # default
    transcript_type="protein_coding",
)
print(f"Annotations for {len(annot)} genes")
annot.head()
```

#### Mark Transcription Factors

deepSCENIC models gene regulation through transcription factors (TFs). We need to identify which genes in our dataset are TFs so the model knows which genes can act as regulators. This adds an `is_tf` column to `adata_rna.var` and stores the TF order in `adata_rna.uns['tf_order']`.

```python
# Mark TFs in the RNA data
ds.pp.mark_tfs(adata_rna, tfs)
print(f"Marked {adata_rna.var['is_tf'].sum()} TFs in the dataset")
```

#### Add Gene Annotations

Gene annotations (chromosome positions) are needed for two purposes:
1. Computing the region-to-gene search space (which regions can regulate which genes)
2. Splitting features by chromosome for model evaluation

This adds `chromosome` and `tss` (transcription start site) columns to `adata_rna.var`.

```python
# Add gene annotations (chromosome, TSS)
ds.pp.add_gene_annotation(adata_rna, annot)
adata_rna.var[['chromosome', 'tss', 'is_tf']].head()
```

### scATAC-seq Preprocessing

Load the chromatin accessibility matrix. This is required for training a deepSCENIC model, but not required for running inference or gene perturbation analyses using an already-trained model.

```{note}
We strongly recommend using **imputed** accessibility values (instead of raw) from [pycisTopic](https://pycistopic.readthedocs.io/en/latest/). The imputed matrix provides continuous accessibility values that are better suited for the VAE architecture than sparse binary fragment counts.
```

```python
# Load scATAC-seq data (imputed accessibility from pycisTopic)
adata_atac = sc.read_h5ad(f"{data_dir}/fragment_matrix.h5ad")
print(f"ATAC data: {adata_atac.n_obs} cells x {adata_atac.n_vars} regions")
```

#### Mark Differentially Accessible Regions (Optional)

Differentially Accessible Regions (DARs) are cell-type-specific regulatory regions identified through differential accessibility analysis. Marking DARs allows the model to upweight these biologically important regions during training.

```{note}
This step is **optional**. If you have DAR results from (for example) pycisTopic, you can load them here. Otherwise, skip this step.
```

```python
# Optional: Mark DARs if you have pycisTopic differential accessibility results
# ds.pp.mark_dars(adata_atac, dar_dir="path/to/DARs/")

# Alternatively, provide a dictionary mapping cell types to region lists:
# dar_dict = {"CellType1": ["chr1:1000-2000", ...], "CellType2": [...]}
# ds.pp.mark_dars(adata_atac, dar_dict=dar_dict)
```

## Create MuData

deepSCENIC uses MuData to store multimodal data in a single object. The {func}`~deepscenic.pp.create_mudata` function combines the RNA and ATAC modalities and automatically parses region coordinates from the ATAC `var_names`. We expect the same cells in both data modalities. 

```{note}
The RNA and ATAC datasets must have identical cell barcodes (matching `obs_names`).
```

```python
# Create MuData from RNA and ATAC
mdata = ds.pp.create_mudata(rna=adata_rna, atac=adata_atac)
mdata
```

## MuData Processing

Now we apply the preprocessing steps that operate on the combined MuData object.


### Compute Region-to-Gene Search Space

The region-to-gene (R2G) search space defines which genomic regions can potentially regulate which genes. This is based on genomic distance: regions closer to a gene's transcription start site (TSS) are more likely to regulate that gene.

The function computes a distance-weighted penalty matrix where:
- `max_distance`: Maximum distance from TSS to consider (default: 1Mb)
- `sigma`: Gaussian decay parameter (default: 100kb) - regions closer to TSS get lower penalty

**Important** Since we need to know the TSS per gene to deduce region-gene links we filter out genes without TSS annotation in this step. 

```python
# Compute r2g penalty matrix (distance-based)
ds.pp.compute_r2g_penalty(
    mdata,
    max_distance=1_000_000,  # 1Mb
    sigma=100_000,           # 100kb Gaussian sigma for decay penalty
)
```

### Build PPI Network (Optional)

The Protein-Protein Interaction (PPI) network models post-transcriptional regulation of TF activity. When TFs interact with other proteins, their regulatory activity can be modulated. This is captured through a Graph Attention Network (GAT) that learns from PPI data.

```{note}
This step is **optional**. If you don't have PPI data, the model will skip the PPI modulation component.
```

```python
# Optional: Build PPI network for TF activity modulation
# Download STRING PPI data from https://string-db.org/
#
# ppi_df = ds.pp.load_string_ppi("path/to/string_links.txt", score_threshold=400)
# ds.pp.build_ppi_network(mdata, ppi_df, species="mouse")
```

### Split Data for Training

We split the data into training and test sets at two levels:

1. **Cell split**: Random split of cells (e.g., 80% train, 20% test)
2. **Feature split**: Chromosome-based split of genes/regions for proper evaluation

The split creates a 2×2 matrix for each modality:

```
                        GENES (by TSS chromosome)
                    ┌─────────────┬─────────────┐
                    │   TRAIN     │    TEST     │
                    │ (other chr) │ (chr7,11,   │
                    │             │  18,19)     │
     ┌──────────────┼─────────────┼─────────────┤
     │    TRAIN     │ ad_rna_train│ ad_rna_E2   │
CELLS│   (80%)      │ [MAIN]      │ [E2 EVAL]   │
     ├──────────────┼─────────────┼─────────────┤
     │    TEST      │ train_eval  │ ad_rna_test │
     │   (20%)      │ [RECON EVAL]│ [HELD OUT]  │
     └──────────────┴─────────────┴─────────────┘

                       REGIONS (by chromosome)
                    ┌─────────────┬─────────────┐
                    │   TRAIN     │    TEST     │
     ┌──────────────┼─────────────┼─────────────┤
     │    TRAIN     │ad_atac_train│ ad_atac_E2  │
CELLS│   (80%)      │ [MAIN]      │ [E2 EVAL]   │
     ├──────────────┼─────────────┼─────────────┤
     │    TEST      │ ctest_rtrain│ad_atac_test │
     │   (20%)      │ [RECON EVAL]│ [HELD OUT]  │
     └──────────────┴─────────────┴─────────────┘
```

**Split semantics:**
- **Cell split**: Same for both modalities (80/20 random)
- **Feature split**: Independent per modality, but linked by biology
  - RNA genes: split by TSS chromosome
  - ATAC regions: split by region chromosome
  - Since R2G only links within-chromosome pairs, test genes ↔ test regions

**Training uses:**
- `train cells × train features`: Main training data
- `train cells × test features`: E2 matrix evaluation (region→gene weights for held-out chromosomes)

**Evaluation uses:**
- `test cells × train features`: Reconstruction evaluation (same features, new cells)
- `test cells × test features`: Full held-out evaluation (new cells, new features)

```python
# Split cells (80% train, 20% test)
ds.pp.split_cells(mdata, test_fraction=0.2, seed=42)
print(f"Cell split: {(mdata.obs['split'] == 'train').sum()} train, {(mdata.obs['split'] == 'test').sum()} test")
```

```python
# Split features by chromosome
ds.pp.split_features_by_chromosome(
    mdata,
    test_chromosomes=["chr7", "chr11", "chr18", "chr19"],
)
print(f"Gene split: {(mdata['rna'].var['split'] == 'train').sum()} train, {(mdata['rna'].var['split'] == 'test').sum()} test")
print(f"Region split: {(mdata['atac'].var['split'] == 'train').sum()} train, {(mdata['atac'].var['split'] == 'test').sum()} test")
```

## Saving and Loading

deepSCENIC stores all preprocessed data in a single `.h5mu` file. This includes both modalities, all annotations, the R2G matrix, and split information.


### Save Preprocessed Data

Use {func}`~deepscenic.write` to save the preprocessed MuData. The function validates the data against the deepSCENIC schema before saving to ensure all required fields are present.

```python
# Save preprocessed MuData
ds.write(mdata, data_dir + "preprocessed_data.h5mu")
```

### Load Preprocessed Data

In future sessions, use {func}`~deepscenic.read` to load your preprocessed data. The function validates the schema on load to ensure data integrity.

```python
# Load preprocessed MuData in future sessions
# mdata = ds.read(data_dir + "preprocessed_data.h5mu")
```
## Next Steps

In the next tutorial, we'll train the deepSCENIC model:

- {doc}`02_training`
