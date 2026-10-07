# deepSCENIC

**Deep learning for single-cell Gene Regulatory Networks**

[![Documentation](https://readthedocs.org/projects/deepscenic/badge/?version=latest)](https://deepscenic.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/deepscenic.svg)](https://pypi.org/project/deepscenic)
[![Python](https://img.shields.io/pypi/pyversions/deepscenic.svg)](https://pypi.org/project/deepscenic)

<img src="https://github.com/aertslab/deepSCENIC/blob/main/docs/deepSCENIC.png" width=100%>

deepSCENIC learns hierarchical TF→region→gene regulatory cascades by integrating scRNA-seq, scATAC-seq, and DNA sequence data. It combines a VAE-based multimodal framework with Enformer-derived sequence embeddings to infer cell-type-specific gene regulatory networks.

## Key Features

- **Multimodal integration**: Jointly models RNA expression and chromatin accessibility
- **Sequence-informed**: Uses Enformer to learn TF binding patterns from DNA sequence
- **Perturbation prediction**: Simulates TF knockdown/overexpression effects
- **scverse compatible**: Works with AnnData, MuData, and scanpy workflows

## Installation

```bash
pip install deepscenic
```
If you want to use the [enformer](https://github.com/lucidrains/enformer-pytorch) model as the sequence model (this is the default):

```bash
pip install deepscenic[enformer]
```

**Requirements**: Python ≥3.12, PyTorch ≥2.6

## Documentation

Full documentation including tutorials and API reference: [deepscenic.readthedocs.io](https://deepscenic.readthedocs.io)

## Quick Start

deepSCENIC follows the scanpy-style API with modules for preprocessing (`ds.pp`), tools (`ds.tl`), and plotting (`ds.pl`).

### Data Preparation

```python
import deepscenic as ds

mdata = ds.pp.create_mudata(rna=adata_rna, atac=adata_atac)

# Annotate transcription factors
tfs = ds.fetch_tf_collection(species="human")
ds.pp.mark_tfs(mdata, tf_list=tfs)

# Add gene annotations (for chromosome-based splitting)
annot, chromsizes = ds.fetch_gene_annotation(species="hsapiens")
ds.pp.add_gene_annotation(mdata, annotation=annot)

# Compute region-to-gene search space (regions within 1Mb of gene TSS)
ds.pp.compute_r2g_penalty(mdata)

# Split cells and features for training/evaluation
ds.pp.split_cells(mdata, test_fraction=0.2, stratify_key="cell_type")
ds.pp.split_features_by_chromosome(mdata, test_chromosomes=["chr7", "chr11"])
```

### Model Training

```python
# Register genome for sequence extraction
ds.register_genome(fasta_path="/path/to/hg38.fa")

# Phase 1: Train full model (VAE + MotifNet + Enformer)
model = ds.tl.train(mdata, epochs=100, device="cuda")

# Phase 2: Recompute tf2r and finetune all r2g links
model = ds.tl.finetune_r2g(
    model,
    mdata,
    epochs=500,
)

# Save trained model
model.save("my_model/")
```

### GRN Extraction & Analysis

```python
# Extract gene regulatory network
grn = ds.tl.extract_grn(model, mdata)

# Get targets of a specific TF
targets = ds.tl.get_tf_targets(grn, tf="SOX10", top_n=100)

# Visualize GRN heatmap
ds.pl.heatmap_grn(grn, tfs=["SOX10", "MITF", "PAX3"])
```

### Perturbation Simulation

```python
# Simulate TF knockdown (level=0) or overexpression (level=2)
perturbed, logFC = ds.tl.simulate_perturbation(
    model,
    mdata,
    tf_name="SOX10",
    level=0,  # knockdown
)

# Summarize and visualize perturbation effects
results = ds.tl.process_perturbation_results(logFC, mdata, tf_name="SOX10")
ds.pl.waterfall_perturbation(results)
```

For complete workflows, see the [tutorials](https://deepscenic.readthedocs.io/en/latest/tutorials.html).

## Citation

If you use deepSCENIC in your research, please cite:

> [Citation to be added]

## Contributors

- **Gabriele Partel** (Author)
- Lukas Mahieu

## Links

- [Documentation](https://deepscenic.readthedocs.io)
- [GitHub Repository](https://github.com/aertslab/deepSCENIC)
- [Issue Tracker](https://github.com/aertslab/deepSCENIC/issues)
- [Enformer PyTorch](https://github.com/lucidrains/enformer-pytorch) - Sequence model implementation
