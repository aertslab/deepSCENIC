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


def e1_sparsity_loss(adj_E1: Tensor) -> Tensor:
    """
    L1 sparsity penalty on E1 (TF→region weights).

    Parameters
    ----------
    adj_E1
        E1 weights (subset being updated)

    Returns
    -------
    Tensor
        Sparsity loss (scalar)
    """
    return adj_E1.abs().mean()


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
        Indices of genes to reconstruct
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
        Whether to include E1 sparsity in the total loss. Set to False
        before warmup_grn to match legacy behavior where E1 sparsity is
        only applied after the GRN warmup phase.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with all loss components and total
    """
    # Reconstruction losses
    loss_rec_rna = reconstruction_loss(x_rna_rec, x_rna[:, gene_indices], loss_rna, dropout_mask_rna) * rna_tau
    loss_rec_atac = reconstruction_loss(x_atac_rec, x_atac, loss_atac, dropout_mask_atac) * atac_tau

    # KL divergence
    loss_kl = kl_divergence(mu, logvar) * beta

    # Sparsity losses
    loss_e1_sparse = e1_sparsity_loss(adj_E1_batch) * alpha
    loss_e2_sparse = e2_sparsity_loss(adj_E2, r2g_distances) * gamma

    # PPI loss (only if using PPI)
    if use_ppi:
        loss_ppi = ppi_activation_loss(x_rna_ppi) * alpha
    else:
        loss_ppi = torch.tensor(0.0, device=x_rna.device)

    # Total loss for backpropagation
    # Note: PPI loss is computed for logging but excluded from total to match legacy
    # behavior where it was calculated but commented out of the loss sum.
    total = loss_rec_rna + loss_rec_atac + loss_kl + loss_e2_sparse
    if include_e1_sparsity:
        total = total + loss_e1_sparse

    return {
        "total": total,
        "rec_rna": loss_rec_rna.detach(),
        "rec_atac": loss_rec_atac.detach(),
        "kl": loss_kl.detach(),
        "e1_sparse": loss_e1_sparse.detach(),
        "e2_sparse": loss_e2_sparse.detach(),
        "ppi": loss_ppi.detach(),
    }
