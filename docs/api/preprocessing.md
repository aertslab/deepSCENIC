# Preprocessing (`ds.pp`)

Functions for preparing multimodal data for GRN learning.

## Data Integration

```{eval-rst}
.. autofunction:: deepscenic.pp.create_mudata
```

## Basic Preprocessing

```{eval-rst}
.. autofunction:: deepscenic.pp.filter_regions_by_celltype
.. autofunction:: deepscenic.pp.remove_zero_variance_genes
```

## TF and Gene Annotation

```{eval-rst}
.. autofunction:: deepscenic.pp.mark_tfs
.. autofunction:: deepscenic.pp.add_gene_annotation
```

## DAR Handling

```{eval-rst}
.. autofunction:: deepscenic.pp.mark_dars
```

## Search Space

```{eval-rst}
.. autofunction:: deepscenic.pp.compute_r2g_penalty
```

## Train/Test Splitting

```{eval-rst}
.. autofunction:: deepscenic.pp.split_cells
.. autofunction:: deepscenic.pp.split_features_by_chromosome
```
