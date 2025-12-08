# Preprocessing (`ds.pp`)

Functions for preparing multimodal data for GRN learning.

## Data Integration

```{eval-rst}
.. autofunction:: deepscenic.pp.create_mudata
```

## Basic Preprocessing

```{eval-rst}
.. autofunction:: deepscenic.pp.filter_genes
.. autofunction:: deepscenic.pp.normalize_rna
.. autofunction:: deepscenic.pp.filter_regions_by_celltype
```

## TF Annotation

```{eval-rst}
.. autofunction:: deepscenic.pp.mark_tfs
.. autofunction:: deepscenic.pp.parse_region_coordinates
```

## DAR Handling

```{eval-rst}
.. autofunction:: deepscenic.pp.mark_dars
.. autofunction:: deepscenic.pp.load_dars_from_bed
```

## Search Space

```{eval-rst}
.. autofunction:: deepscenic.pp.compute_r2g_penalty
.. autofunction:: deepscenic.pp.split_r2g_by_chromosome
```

## PPI Network

```{eval-rst}
.. autofunction:: deepscenic.pp.build_ppi_network
.. autofunction:: deepscenic.pp.load_string_ppi
```

## Train/Test Splitting

```{eval-rst}
.. autofunction:: deepscenic.pp.split_cells
.. autofunction:: deepscenic.pp.split_features_by_chromosome
```

## High-Level Functions

```{eval-rst}
.. autofunction:: deepscenic.pp.prepare_for_training
```
