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
notebooks/06_sequence_interpretation
notebooks/06_legacy_migration
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
| 02 | Training | `ds.tl.train()`, `ds.tl.finetune_r2g()` |
| 03 | Model Diagnosis | `ds.tl.to_latent()`, `sc.tl.umap()` |
| 04 | GRN Analysis | `ds.tl.extract_grn()`, `ds.tl.identify_active_enhancers()` |
| 05 | Perturbation Analysis | `ds.tl.simulate_perturbation()`, `ds.pl.waterfall_perturbation()` |
| 06 | Sequence Interpretation | `ds.tl.in_silico_mutagenesis()`, `ds.pl.ism_heatmap()` |
| 07 | Legacy Migration | Migrating from legacy deepSCENIC format |
