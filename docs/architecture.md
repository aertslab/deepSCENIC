# Architecture

deepSCENIC is a hierarchical Gene Regulatory Network (GRN) model that learns TF→region→gene regulatory cascades.

## Model Overview

```
TF expression
  ↓ InferenceNet (per-TF MLP)
  ↓ z_tf (latent TF activity)
  ↓ × tf2r (TF→region weights)
  ↓ region_activity
  ├──→ GenerativeNet_ATAC → ATAC reconstruction
  ↓ × r2g (region→gene weights)
  ↓ gene_signal
  └──→ GenerativeNet_RNA → RNA reconstruction
```

## Core Components

### 1. TF2rNet (TF-to-Region Network)

Predicts TF binding potential from DNA sequence:
- **Enformer**: Pretrained DNA sequence transformer (processes ~640bp regions)
- **MotifNet**: Context head that learns TF binding patterns
- **Output**: tf2r matrix (n_regions × n_tfs)

### 2. VAE (Variational Autoencoder)

Transforms TF expression to regulatory activity:
- **InferenceNet**: Per-TF MLP encoder with Gaussian sampling
- **GenerativeNet_RNA**: Per-gene MLP decoder for RNA
- **GenerativeNet_ATAC**: Per-region MLP decoder for ATAC

### 3. GRN Matrices

- **tf2r (TF→region)**: Learned from DNA sequence via TF2rNet
- **r2g (region→gene)**: Sparse learnable matrix, distance-constrained

## Training Strategy

The sequence model, MotifNet, and VAE are trained jointly. Region-to-gene
weights can subsequently be refined with `finetune_r2g`.

## Loss Components

```
Total Loss =
  + loss_rec_rna      # RNA reconstruction
  + loss_rec_atac     # ATAC reconstruction
  + loss_gauss_rna    # KL divergence (VAE regularization)
  + tf2r_sparse         # L1 sparsity on TF→region
  + r2g_sparse         # L1 sparsity on region→gene (distance-weighted)
```

## Perturbation Simulation

```python
# Set TF expression to perturbation level
perturbed_matrix[:, "SOX10"] = 0

# Iterate to steady state
for i in range(n_iter):
    z_tf_pert = InferenceNet(perturbed_TF_expression)
    gene_signal_pert = z_tf_pert @ (tf2r @ r2g)
    logFC = gene_signal_pert - gene_signal_orig
```

## Key Design Choices

### Positive-Constrained Layers
All network weights use absolute values, ensuring positive activations (biologically motivated).

### Distance Penalty
Region-gene links are weighted by distance to TSS:
```
penalty = 1 - gaussian(distance, σ=100kb)
```

### Per-Feature MLPs
InferenceNet and GenerativeNets process each feature independently through shared-weight MLPs, maintaining interpretability.

## Data Flow Summary

```
Input:
  - scRNA-seq: (n_cells, n_genes)
  - scATAC-seq: (n_cells, n_regions)
  - DNA sequences: (n_regions, 640bp)

Processing:
  1. TF2rNet: DNA → tf2r (TF binding predictions)
  2. InferenceNet: TF expression → z_tf (latent activity)
  3. tf2r × r2g: z_tf → region activity → gene signal
  4. Decoders: Reconstruct RNA and ATAC

Output:
  - Trained GRN (tf2r, r2g matrices)
  - Perturbation predictions
  - Latent TF activity space
```
