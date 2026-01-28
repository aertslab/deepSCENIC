"""Loss functions for deepSCENIC training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    loss_type: str = "mse",
    dropout_mask: bool = False,
) -> Tensor:
    """
    Compute reconstruction loss.

    Parameters
    ----------
    prediction
        Model predictions
    target
        Ground truth values
    loss_type
        One of: 'mse', 'mae', 'bce', 'cosine'
    dropout_mask
        If True, only compute loss on non-zero targets

    Returns
    -------
    Tensor
        Scalar loss value
    """
    if loss_type == "mse":
        if dropout_mask:
            mask = target != 0
            # Avoid division by zero
            counts = mask.sum(dim=1).clamp(min=1)
            return (((prediction - target) ** 2 * mask).sum(dim=1) / counts).mean()
        return F.mse_loss(prediction, target)

    elif loss_type == "mae":
        if dropout_mask:
            mask = target != 0
            counts = mask.sum(dim=1).clamp(min=1)
            return (((prediction - target).abs() * mask).sum(dim=1) / counts).mean()
        return F.l1_loss(prediction, target)

    elif loss_type == "bce":
        # Skip cells with all -1 (missing data indicator)
        mask = ~(target == -1).all(dim=1)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=prediction.device)
        return F.binary_cross_entropy_with_logits(prediction[mask], target[mask])

    elif loss_type == "cosine":
        return 1 - F.cosine_similarity(prediction, target, dim=1).mean()

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    KL divergence for VAE: KL(q(z|x) || p(z)) where p(z) = N(0, I).

    Parameters
    ----------
    mu
        Encoder mean
    logvar
        Encoder log variance

    Returns
    -------
    Tensor
        KL divergence (scalar)
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def e1_sparsity_loss(adj_E1: Tensor, seq_idx: Tensor | None = None) -> Tensor:
    """
    L1 sparsity penalty on E1 (TF→region weights).

    Parameters
    ----------
    adj_E1
        E1 weights (full matrix or subset)
    seq_idx
        Indices of regions to compute sparsity on (legacy behavior).
        If None, computes sparsity over entire E1 matrix.

    Returns
    -------
    Tensor
        Sparsity loss (scalar)
    """
    if seq_idx is not None:
        # Legacy behavior: only compute on sampled batch indices
        return adj_E1[seq_idx, :].abs().mean()
    return adj_E1.abs().mean()


def f1_score_binary(prediction: Tensor, target: Tensor) -> Tensor:
    """Binary F1 score for ATAC predictions.

    Only meaningful when binary_atac=True.

    Parameters
    ----------
    prediction
        Model predictions (logits)
    target
        Ground truth binary values

    Returns
    -------
    Tensor
        F1 score (scalar)
    """
    from torchmetrics.classification import BinaryF1Score

    # Mask invalid entries (all -1)
    mask = ~(target == -1).all(dim=1)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=prediction.device)

    f1 = BinaryF1Score().to(prediction.device)
    # Apply sigmoid to convert logits to probabilities
    pred_probs = torch.sigmoid(prediction[mask]).ravel()
    return f1(pred_probs, target[mask].int().ravel())


def e2_sparsity_loss(adj_E2: Tensor, r2g_distances: Tensor) -> Tensor:
    """
    Distance-weighted L1 sparsity on E2 (region→gene weights).

    Combines L1 penalty with distance penalty to encourage nearby links.

    Parameters
    ----------
    adj_E2
        Learnable E2 weights
    r2g_distances
        Distance penalties (higher = farther from TSS)

    Returns
    -------
    Tensor
        Sparsity loss (scalar)
    """
    return (adj_E2.abs() * r2g_distances).mean()


def ppi_activation_loss(x_rna_ppi: Tensor) -> Tensor:
    """
    Encourage PPI network to produce high activation weights.

    Loss = 1 - mean(|ppi_weights|)

    Parameters
    ----------
    x_rna_ppi
        PPI modulation weights (should be close to 1)

    Returns
    -------
    Tensor
        PPI activation loss (scalar)
    """
    return 1 - x_rna_ppi.abs().mean()


def compute_total_loss(
    x_rna: Tensor,
    x_atac: Tensor,
    x_rna_rec: Tensor,
    x_atac_rec: Tensor,
    mu: Tensor,
    logvar: Tensor,
    adj_E1_batch: Tensor,
    adj_E2: Tensor,
    r2g_distances: Tensor,
    x_rna_ppi: Tensor,
    gene_indices: Tensor,
    region_indices: Tensor,
    loss_rna: str = "mse",
    loss_atac: str = "mse",
    dropout_mask_rna: bool = False,
    dropout_mask_atac: bool = False,
    beta: float = 1e-2,
    alpha: float = 1e-2,
    gamma: float = 1.0,
    rna_tau: float = 1.0,
    atac_tau: float = 1.0,
    use_ppi: bool = True,
    include_e1_sparsity: bool = True,
    ppi_phase: bool = False,
    seq_idx: Tensor | None = None,
) -> dict[str, Tensor]:
    """
    Compute all loss components for training.

    Parameters
    ----------
    x_rna
        Input RNA expression
    x_atac
        Input ATAC accessibility
    x_rna_rec
        Reconstructed RNA
    x_atac_rec
        Reconstructed ATAC
    mu
        Encoder mean
    logvar
        Encoder log variance
    adj_E1_batch
        E1 weights being updated this iteration
    adj_E2
        All E2 weights
    r2g_distances
        Distance penalties for E2
    x_rna_ppi
        PPI modulation weights
    gene_indices
        Indices of TRAIN genes to compute RNA reconstruction loss on.
        Only these genes contribute to the reconstruction loss, while E2 links
        to all genes still receive sparsity regularization.
    region_indices
        Indices of TRAIN regions to compute ATAC reconstruction loss on.
        Only these regions contribute to the reconstruction loss.
    loss_rna
        RNA loss type
    loss_atac
        ATAC loss type
    dropout_mask_rna
        Whether to mask RNA loss on zeros
    dropout_mask_atac
        Whether to mask ATAC loss on zeros
    beta
        KL divergence weight
    alpha
        E1 sparsity + PPI weight
    gamma
        E2 sparsity weight
    rna_tau
        RNA reconstruction weight
    atac_tau
        ATAC reconstruction weight
    use_ppi
        Whether PPI is being used
    include_e1_sparsity
        Whether to include E1 sparsity in the total loss.
    ppi_phase
        If True, we're in "PPI training phase" (epoch >= warmup_grn).
        In this phase, sparsity losses (E1, E2) are excluded from the total
        loss. This matches legacy behavior where only reconstruction losses
        are used to train PPInet.
    seq_idx
        Indices of regions being updated this iteration (for E1 sparsity).
        If provided, E1 sparsity is computed only on these regions (legacy behavior).

    Returns
    -------
    dict[str, Tensor]
        Dictionary with all loss components and total
    """
    # Reconstruction losses - filter both prediction and target to TRAIN features
    # This ensures reconstruction loss only evaluates on held-in features,
    # while E2 sparsity loss still applies to ALL links (including to test genes).
    loss_rec_rna = (
        reconstruction_loss(x_rna_rec[:, gene_indices], x_rna[:, gene_indices], loss_rna, dropout_mask_rna) * rna_tau
    )
    loss_rec_atac = (
        reconstruction_loss(x_atac_rec[:, region_indices], x_atac[:, region_indices], loss_atac, dropout_mask_atac)
        * atac_tau
    )

    # KL divergence
    loss_kl = kl_divergence(mu, logvar) * beta

    # Sparsity losses
    # E1 sparsity: only on sampled batch (legacy behavior)
    loss_e1_sparse = e1_sparsity_loss(adj_E1_batch, seq_idx=seq_idx) * alpha
    loss_e2_sparse = e2_sparsity_loss(adj_E2, r2g_distances) * gamma

    # PPI loss (only if using PPI)
    if use_ppi:
        loss_ppi = ppi_activation_loss(x_rna_ppi) * alpha
    else:
        loss_ppi = torch.tensor(0.0, device=x_rna.device)

    # Total loss for backpropagation
    # During PPI phase (epoch >= warmup_grn), sparsity losses are excluded
    # This matches legacy behavior where PPInet is trained only via reconstruction
    if ppi_phase:
        total = loss_rec_rna + loss_rec_atac + loss_kl
    else:
        total = loss_rec_rna + loss_rec_atac + loss_kl + loss_e2_sparse
        if include_e1_sparsity:
            total = total + loss_e1_sparse

    return {
        "total": total,
        "rna_recon": loss_rec_rna.detach(),
        "atac_recon": loss_rec_atac.detach(),
        "kl_div": loss_kl.detach(),
        "e1_l1": loss_e1_sparse.detach(),
        "e2_l1": loss_e2_sparse.detach(),
        "ppi_reg": loss_ppi.detach(),
    }


def compute_test_chromosome_loss(
    x_atac: Tensor,
    x_atac_rec_test: Tensor,
    adj_E1_test: Tensor,
    test_region_indices: Tensor,
    loss_atac: str,
    alpha: float,
    atac_tau: float,
    x_rna: Tensor | None = None,
    x_rna_rec: Tensor | None = None,
    test_gene_indices: Tensor | None = None,
    loss_rna: str = "mse",
    rna_tau: float = 1.0,
) -> dict[str, Tensor]:
    """Compute reconstruction loss on test chromosomes.

    Parameters
    ----------
    x_atac
        Ground truth ATAC (n_cells, n_all_regions)
    x_atac_rec_test
        Reconstructed ATAC for test regions (n_cells, n_test_regions)
    adj_E1_test
        E1 matrix for test regions (n_test_regions, n_tfs)
    test_region_indices
        Global indices of test regions
    loss_atac
        ATAC loss type: 'mse', 'mae', 'bce', 'cosine'
    alpha
        E1 sparsity weight
    atac_tau
        ATAC reconstruction weight
    x_rna
        Ground truth RNA (n_cells, n_all_genes). Optional.
    x_rna_rec
        Reconstructed RNA (n_cells, n_all_genes). Optional.
    test_gene_indices
        Global indices of test genes. Optional.
    loss_rna
        RNA loss type: 'mse', 'mae', 'cosine'
    rna_tau
        RNA reconstruction weight

    Returns
    -------
    dict with keys: total, atac_recon, e1_l1, and optionally rna_recon
    """
    # Extract ground truth for test regions
    x_atac_test = x_atac[:, test_region_indices]

    # ATAC reconstruction loss
    loss_rec_atac = reconstruction_loss(x_atac_rec_test, x_atac_test, loss_atac) * atac_tau

    # E1 sparsity on test E1
    loss_e1_sparse = e1_sparsity_loss(adj_E1_test) * alpha

    result = {
        "total": loss_rec_atac + loss_e1_sparse,
        "atac_recon": loss_rec_atac.detach(),
        "e1_l1": loss_e1_sparse.detach(),
    }

    # Add RNA reconstruction if provided
    if x_rna is not None and x_rna_rec is not None and test_gene_indices is not None:
        x_rna_test = x_rna[:, test_gene_indices]
        x_rna_rec_test = x_rna_rec[:, test_gene_indices]
        loss_rec_rna = reconstruction_loss(x_rna_rec_test, x_rna_test, loss_rna) * rna_tau
        result["rna_recon"] = loss_rec_rna.detach()
        result["total"] = result["total"] + loss_rec_rna

    return result
