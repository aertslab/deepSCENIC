# Plotting (`ds.pl`)

Visualization functions for GRN analysis, perturbation results, and model diagnostics.

## Utilities

```{eval-rst}
.. autofunction:: deepscenic.pl.set_colors
.. autofunction:: deepscenic.pl.get_colors
.. autofunction:: deepscenic.pl.setup_axes
.. autofunction:: deepscenic.pl.savefig_or_show
```

## GRN Visualization

### Heatmaps

```{eval-rst}
.. autofunction:: deepscenic.pl.heatmap_r2g
.. autofunction:: deepscenic.pl.heatmap_grn
```

### Networks

```{eval-rst}
.. autofunction:: deepscenic.pl.network_grn
.. autofunction:: deepscenic.pl.network_tf_targets
.. autofunction:: deepscenic.pl.network_gene_regulators
```

## Sequence Analysis

```{eval-rst}
.. autofunction:: deepscenic.pl.logo_attribution
.. autofunction:: deepscenic.pl.logo_motif
.. autofunction:: deepscenic.pl.ism_heatmap
```

## Embedding Visualization

```{eval-rst}
.. autofunction:: deepscenic.pl.embedding_umap
```

## Perturbation Results

```{eval-rst}
.. autofunction:: deepscenic.pl.waterfall_perturbation
.. autofunction:: deepscenic.pl.heatmap_perturbation
.. autofunction:: deepscenic.pl.dotplot_perturbation
.. autofunction:: deepscenic.pl.perturbation_pca
```

## Genomics

```{eval-rst}
.. autofunction:: deepscenic.pl.genome_browser
.. autofunction:: deepscenic.pl.arc_plot
```

## Training Diagnostics

```{eval-rst}
.. autofunction:: deepscenic.pl.loss_curves
.. autofunction:: deepscenic.pl.enhancer_activity_histogram
.. autofunction:: deepscenic.pl.tf_activity_clustermap
```
