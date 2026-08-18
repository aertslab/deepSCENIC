# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU support)
- 16GB+ RAM (32GB+ recommended for large datasets)

## Basic Installation

```bash
pip install deepscenic
```

## With GPU Support (Enformer)

For full functionality including DNA sequence processing:

```bash
pip install deepscenic[enformer]
```

## Development Installation

```bash
git clone https://github.com/aertslab/deepSCENIC.git
cd deepSCENIC

# Using hatch (recommended)
pip install hatch
hatch env create

# Or using pip directly
pip install -e ".[dev,test,doc]"
```

## Optional Dependencies

### Sequence Analysis
For sequence logos and ISM visualization:
```bash
pip install deepscenic[sequence]
```

### All Optional Dependencies
```bash
pip install deepscenic[enformer,sequence]
```

## Verify Installation

```python
import deepscenic as ds

print(ds.__version__)

# Check GPU availability
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
```

## Common Issues

### CUDA Out of Memory
Reduce batch sizes:
```python
model = ds.tl.train(mdata, batch_size=32, seq_batch_size=500)
```

### Enformer Download Issues
The Enformer model (~1GB) downloads automatically on first use. If download fails:
```python
from enformer_pytorch import Enformer

Enformer.from_pretrained("EleutherAI/enformer-official-rough")
```

### Import Errors
If you encounter import errors, ensure all dependencies are installed:
```bash
pip install deepscenic[enformer,sequence] --upgrade
```
