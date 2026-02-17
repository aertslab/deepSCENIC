# API Reference

deepSCENIC follows the scanpy-style API organization with separate modules for preprocessing (`pp`), tools (`tl`), and plotting (`pl`). Core I/O and utility functions are available directly on the package.

```{toctree}
:maxdepth: 2

api/preprocessing
api/tools
api/plotting
api/io
api/datasets
api/data
```

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `ds.pp` | Preprocessing | `create_mudata`, `mark_tfs`, `compute_r2g_penalty`, `split_cells` |
| `ds.tl` | Training & Analysis | `train`, `finetune_r2g`, `extract_grn`, `simulate_perturbation` |
| `ds.pl` | Plotting | `loss_curves`, `heatmap_grn`, `waterfall_perturbation`, `genome_browser` |

## Root-Level Functions

These functions are available directly on `deepscenic`:

```python
import deepscenic as ds

# I/O
mdata = ds.read("data.h5mu")
ds.write(mdata, "output.h5mu")
regions = ds.read_bed("peaks.bed")

# Data fetching
tfs = ds.fetch_tf_collection(species="mouse")
annot, chromsizes = ds.fetch_gene_annotation(species="mmusculus")

# Validation
ds.validate_schema(mdata)
```
