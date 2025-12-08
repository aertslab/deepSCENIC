# Tutorials

These tutorials demonstrate the complete deepSCENIC workflow from data loading to perturbation simulation.

## Tutorial Series

```{toctree}
:maxdepth: 1

notebooks/01_data_loading
notebooks/02_preprocessing
notebooks/03_training
notebooks/04_grn_analysis
notebooks/05_perturbation_simulation
notebooks/06_visualization
```

## Prerequisites

Before starting, ensure you have:

1. **deepSCENIC installed**: `pip install deepscenic[enformer]`
2. **Sample data downloaded**: See {doc}`notebooks/01_data_loading` for instructions
3. **GPU access** (recommended): Training benefits significantly from GPU acceleration

## Tutorial Overview

| Tutorial | Topic | Key Functions |
|----------|-------|---------------|
| 01 | Data Loading | `ds.read()`, `ds.datasets.*` |
| 02 | Preprocessing | `ds.pp.create_mudata()`, `ds.pp.prepare_for_training()` |
| 03 | Training | `ds.tl.train()`, `ds.pl.loss_curves()` |
| 04 | GRN Analysis | `ds.tl.extract_grn()`, `ds.pl.network_*()` |
| 05 | Perturbation | `ds.tl.simulate_perturbation()`, `ds.pl.volcano_*()` |
| 06 | Visualization | `ds.pl.genome_browser()`, `ds.pl.arc_plot()` |

## Data Requirements

Tutorials use the MM_lines dataset (melanoma cell lines):
- **scRNA-seq**: ~40,000 cells, ~16,000 genes
- **scATAC-seq**: ~40,000 cells, ~440,000 regions (pyCisTopic imputed)

Download instructions are provided in the first tutorial.
