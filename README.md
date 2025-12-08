# deepSCENIC

**Deep learning for single-cell Gene Regulatory Networks**

<img src="https://github.com/aertslab/deepSCENIC/blob/main/docs/deepSCENIC.png" width=100%>

deepSCENIC learns hierarchical TF→region→gene regulatory cascades by integrating scRNA-seq, scATAC-seq, and DNA sequence data.

## Key Features

- **Multimodal integration**: Jointly models RNA expression and chromatin accessibility
- **Sequence-informed**: Uses Enformer to learn TF binding from DNA sequence
- **Perturbation prediction**: Simulates TF knockdown/overexpression effects
- **scverse compatible**: Works with AnnData, MuData, and scanpy workflows

## Installation

```bash
pip install deepscenic
```

For GPU support with Enformer:
```bash
pip install deepscenic[enformer]
```

## Quick Start

```python
import deepscenic as ds

# Load preprocessed data
mdata = ds.read("my_dataset.h5mu")

# Train model
model = ds.tl.train(mdata, epochs=100)

# Extract GRN
grn = ds.tl.extract_grn(model)

# Simulate TF perturbation
logFC = ds.tl.simulate_perturbation(model, mdata, tf_name="SOX10", level=0)
```

## Documentation

For detailed tutorials and API reference, visit [deepscenic.readthedocs.io](https://deepscenic.readthedocs.io).

## Citation

If you use deepSCENIC in your research, please cite:

> [Citation to be added]

## Links

- [Documentation](https://deepscenic.readthedocs.io)
- [GitHub Repository](https://github.com/aertslab/deepSCENIC)
- [Issue Tracker](https://github.com/aertslab/deepSCENIC/issues)
- [SCENIC+ Documentation](https://scenicplus.readthedocs.io/)
