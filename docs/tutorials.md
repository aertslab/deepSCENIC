# Tutorials

These tutorials demonstrate the complete deepSCENIC workflow from data loading to perturbation simulation.

## Tutorial Series

```{toctree}
:maxdepth: 1

notebooks/01_data_preparation
notebooks/02_training
notebooks/03_model_diagnosis
notebooks/04_grn_analysis
notebooks/05_perturbation_analysis
notebooks/06_advanced_visualization
notebooks/07_legacy_migration
```

## Prerequisites

Before starting, ensure you have:

1. **deepSCENIC installed**: `pip install deepscenic[enformer]`
2. **Sample data downloaded**: See {doc}`notebooks/01_data_preparation` for instructions
3. **GPU access** (recommended): Training benefits significantly from GPU acceleration

## Tutorial Overview

| Tutorial | Topic | Key Functions |
|----------|-------|---------------|
| 01 | Data Preparation | `ds.pp.create_mudata()`, `ds.pp.mark_tfs()`, `ds.pp.compute_r2g_penalty()` |
| 02 | Training | `ds.tl.train()`, `ds.tl.finetune_e2()` |
| 03 | Model Diagnosis | `ds.tl.to_latent()`, `sc.tl.umap()` |
| 04 | GRN Analysis | `ds.tl.extract_grn()`, `ds.tl.compute_celltype_enhancer_activity()` |
| 05 | Perturbation Analysis | `ds.tl.simulate_perturbation()`, `ds.pl.volcano_perturbation()` |
| 06 | Advanced Visualization | `ds.pl.genome_browser()`, `ds.pl.arc_plot()` |
| 07 | Legacy Migration | Migrating from legacy deepSCENIC format |

## Data Requirements

Tutorials use the MM_lines dataset (melanoma cell lines):
- **scRNA-seq**: ~40,000 cells, ~16,000 genes
- **scATAC-seq**: ~40,000 cells, ~440,000 regions (pyCisTopic imputed)

Download instructions are provided in the first tutorial.
