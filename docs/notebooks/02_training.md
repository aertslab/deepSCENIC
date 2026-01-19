---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Tutorial 2: Training

This tutorial covers training the deepSCENIC model.

## Learning Objectives

- Understand the deepSCENIC architecture
- Configure training parameters
- Train the model
- Monitor training progress
- Save and load models


## Setup

```python
import deepscenic as ds
import torch

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

## Load Preprocessed Data

```python
# Load preprocessed MuData
# mdata = ds.read("preprocessed_data.h5mu")
# mdata
```

## Understanding the Architecture

deepSCENIC consists of:

1. **TF2rNet**: Predicts TF binding from DNA sequence (Enformer + MotifNet)
2. **PPI Network**: Models post-transcriptional TF regulation (GAT)
3. **VAE**: Transforms TF expression to regulatory activity
4. **GRN Matrices**: E1 (TF→region) and E2 (region→gene)

```
TF expression
  ↓ PPI modulation
  ↓ InferenceNet
  ↓ z_tf (latent TF activity)
  ↓ × E1 (TF→region)
  ↓ region_activity
  ├──→ ATAC reconstruction
  ↓ × E2 (region→gene)
  └──→ RNA reconstruction
```


## Configure Training

```python
# Training configuration
config = {
    "epochs": 100,
    "batch_size": 64,
    "seq_batch_size": 1000,
    "learning_rate": 1e-4,
    "warmup_vae": 10,      # Epochs before updating TF2rNet
    "warmup_grn": 50,      # Epochs before enabling PPI
    "beta": 1e-2,          # KL loss weight
    "alpha": 1e-2,         # E1 sparsity weight
    "gamma": 1.0,          # E2 sparsity weight
}
```

## Train the Model

```python
# Train deepSCENIC
# model = ds.tl.train(
#     mdata,
#     epochs=config["epochs"],
#     batch_size=config["batch_size"],
#     seq_batch_size=config["seq_batch_size"],
#     learning_rate=config["learning_rate"],
#     warmup_vae=config["warmup_vae"],
#     warmup_grn=config["warmup_grn"],
#     beta=config["beta"],
#     alpha=config["alpha"],
#     gamma=config["gamma"],
#     device=device,
# )
```

## Monitor Training

Visualize training progress with loss curves:

```python
# Plot loss curves
# ds.pl.loss_curves(model)
```

## Evaluate Model Quality

Check the learned sparsity and latent representations:

```python
# Sparsity histogram of E1 and E2
# ds.pl.sparsity_histogram(model)
```

```python
# UMAP of latent TF activity space
# ds.pl.latent_umap(model, mdata, color="cell_type")
```

## Save the Model

```python
# Save the trained model
# model.save("deepscenic_model.pt")
```

## Load a Saved Model

```python
# Load a previously trained model
# model = ds.tl.load_model("deepscenic_model.pt")
```

## Next Steps

In the next tutorial, we'll extract and analyze the GRN:

- {doc}`03_grn_analysis`
