"""Perturbation simulation for deepSCENIC."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import torch
from scipy.sparse import issparse
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from typing import Literal

    import mudata as md

    from ._model import DeepSCENICModel


@overload
def simulate_perturbation(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_name: str,
    level: float = ...,
    n_iter: int = ...,
    batch_size: int = ...,
    device: str | torch.device | None = ...,
    split: str | None = ...,
    clip_percentile: float = ...,
    return_intermediate: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def simulate_perturbation(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_name: str,
    level: float = ...,
    n_iter: int = ...,
    batch_size: int = ...,
    device: str | torch.device | None = ...,
    split: str | None = ...,
    clip_percentile: float = ...,
    return_intermediate: Literal[True] = ...,
) -> dict[int, np.ndarray]: ...


def simulate_perturbation(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_name: str,
    level: float = 0.0,
    n_iter: int = 10,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    split: str | None = None,
    clip_percentile: float = 99.9,
    return_intermediate: bool = False,
) -> np.ndarray | dict[int, np.ndarray]:
    """
    Simulate TF knockdown/overexpression.

    Perturbs a TF's expression and iteratively propagates the effect through
    the GRN to predict gene expression changes.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA + ATAC data
    tf_name
        Name of TF to perturb (e.g., 'SOX10')
    level
        Perturbation level (0 = knockout, >0 = overexpression)
    n_iter
        Iterations to reach steady state
    batch_size
        Cells per batch for inference
    device
        Device for computation
    split
        If specified, only process cells from this split
    clip_percentile
        Percentile for clipping perturbed values (default: 99.9)
    return_intermediate
        If True, return logFC at each iteration for convergence analysis

    Returns
    -------
    np.ndarray | dict[int, np.ndarray]
        If return_intermediate=False: Log fold change predictions (n_cells, n_genes)
        If return_intermediate=True: Dict mapping iteration (1-indexed) to logFC array

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> logFC = ds.tl.simulate_perturbation(model, mdata, tf_name="SOX10", level=0)
    >>> # logFC contains predicted expression changes for all genes

    >>> # Check convergence across iterations
    >>> logFC_trace = ds.tl.simulate_perturbation(
    ...     model, mdata, tf_name="SOX10", return_intermediate=True
    ... )
    >>> for i, logFC in logFC_trace.items():
    ...     print(f"Iteration {i}: mean |logFC| = {np.abs(logFC).mean():.4f}")
    """
    model.eval()

    if device is None:
        device = next(model.vae.parameters()).device
    else:
        model.to(device)

    # Find TF index
    if tf_name not in model.tf_names:
        raise ValueError(f"TF '{tf_name}' not found. Available TFs: {model.tf_names[:10]}...")
    tf_idx = model.tf_names.index(tf_name)

    # Get data
    if split is not None:
        mask = mdata.obs["split"] == split
        rna = mdata.mod["rna"][mask].X
    else:
        rna = mdata.mod["rna"].X

    # Convert sparse to dense
    if issparse(rna):
        rna = rna.toarray()

    n_cells = rna.shape[0]

    # Compute clipping threshold from original data
    clip_max = np.percentile(rna, clip_percentile)

    # Output arrays
    logFC_per_iter: dict[int, list[np.ndarray]] = {i: [] for i in range(1, n_iter + 1)}
    logFC_all: list[np.ndarray] = []

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_cells, batch_size), desc="Simulating perturbation"):
            end_idx = min(i + batch_size, n_cells)
            batch_rna = rna[i:end_idx].copy()

            x_rna = torch.FloatTensor(batch_rna).to(device)

            # Get original z_rna (baseline gene regulatory signal)
            output_orig = model.vae(
                x_rna,
                model.adj_E1,
                use_ppi=model.vae.use_ppi,
                use_mean=True,
            )
            z_rna_orig = output_orig.z_rna

            # Create perturbed expression matrix
            perturbed_rna = x_rna.clone()

            # TF index in full gene matrix
            tf_gene_idx = model.vae.tf_indices[tf_idx]  # type: ignore[index]

            # Initialize logFC_decoded for edge case of n_iter=0
            logFC_decoded = torch.zeros_like(x_rna)

            # Iterate to steady state
            for iter_num in range(1, n_iter + 1):
                # Set TF expression to perturbation level
                perturbed_rna[:, tf_gene_idx] = level

                # Forward pass with perturbed input
                output_pert = model.vae(
                    perturbed_rna,
                    model.adj_E1,
                    use_ppi=model.vae.use_ppi,
                    use_mean=True,
                )

                # Compute logFC in latent space, then decode the DIFFERENCE
                # This matches legacy: decode(z_pert - z_orig), NOT decode(z_pert) - decode(z_orig)
                logFC_latent = output_pert.z_rna - z_rna_orig
                logFC_decoded = model.vae.decoder_rna(logFC_latent)

                # Store intermediate if requested
                if return_intermediate:
                    logFC_per_iter[iter_num].append(logFC_decoded.cpu().numpy())

                # Update perturbed matrix: original + decoded fold change
                perturbed_rna = x_rna + logFC_decoded

                # Clip to valid range [0, p99]
                perturbed_rna = torch.clamp(perturbed_rna, 0, clip_max)

                # Re-enforce TF perturbation level after clipping
                perturbed_rna[:, tf_gene_idx] = level

            # Final logFC
            logFC_all.append(logFC_decoded.cpu().numpy())

    if return_intermediate:
        return {i: np.concatenate(logFC_per_iter[i], axis=0) for i in range(1, n_iter + 1)}
    return np.concatenate(logFC_all, axis=0)


@overload
def simulate_multi_perturbation(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_names: list[str],
    levels: list[float] | None = ...,
    n_iter: int = ...,
    batch_size: int = ...,
    device: str | torch.device | None = ...,
    split: str | None = ...,
    clip_percentile: float = ...,
    return_intermediate: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def simulate_multi_perturbation(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_names: list[str],
    levels: list[float] | None = ...,
    n_iter: int = ...,
    batch_size: int = ...,
    device: str | torch.device | None = ...,
    split: str | None = ...,
    clip_percentile: float = ...,
    return_intermediate: Literal[True] = ...,
) -> dict[int, np.ndarray]: ...


def simulate_multi_perturbation(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_names: list[str],
    levels: list[float] | None = None,
    n_iter: int = 10,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    split: str | None = None,
    clip_percentile: float = 99.9,
    return_intermediate: bool = False,
) -> np.ndarray | dict[int, np.ndarray]:
    """
    Simulate simultaneous perturbation of multiple TFs.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA + ATAC data
    tf_names
        List of TF names to perturb
    levels
        Perturbation levels for each TF (default: all 0 = knockout)
    n_iter
        Iterations to reach steady state
    batch_size
        Cells per batch for inference
    device
        Device for computation
    split
        If specified, only process cells from this split
    clip_percentile
        Percentile for clipping perturbed values (default: 99.9)
    return_intermediate
        If True, return logFC at each iteration for convergence analysis

    Returns
    -------
    np.ndarray | dict[int, np.ndarray]
        If return_intermediate=False: Log fold change predictions (n_cells, n_genes)
        If return_intermediate=True: Dict mapping iteration (1-indexed) to logFC array
    """
    model.eval()

    if device is None:
        device = next(model.vae.parameters()).device
    else:
        model.to(device)

    # Default levels
    if levels is None:
        levels = [0.0] * len(tf_names)
    if len(levels) != len(tf_names):
        raise ValueError("levels must have same length as tf_names")

    # Find TF indices
    tf_indices = []
    for tf_name in tf_names:
        if tf_name not in model.tf_names:
            raise ValueError(f"TF '{tf_name}' not found")
        tf_indices.append(model.tf_names.index(tf_name))

    # Get data
    if split is not None:
        mask = mdata.obs["split"] == split
        rna = mdata.mod["rna"][mask].X
    else:
        rna = mdata.mod["rna"].X

    if issparse(rna):
        rna = rna.toarray()

    n_cells = rna.shape[0]

    # Compute clipping threshold from original data
    clip_max = np.percentile(rna, clip_percentile)

    # Output arrays
    logFC_per_iter: dict[int, list[np.ndarray]] = {i: [] for i in range(1, n_iter + 1)}
    logFC_all: list[np.ndarray] = []

    with torch.no_grad():
        for i in tqdm(range(0, n_cells, batch_size), desc="Simulating multi-perturbation"):
            end_idx = min(i + batch_size, n_cells)
            batch_rna = rna[i:end_idx].copy()

            x_rna = torch.FloatTensor(batch_rna).to(device)

            # Get original z_rna (baseline)
            output_orig = model.vae(
                x_rna,
                model.adj_E1,
                use_ppi=model.vae.use_ppi,
                use_mean=True,
            )
            z_rna_orig = output_orig.z_rna

            # Create perturbed expression matrix
            perturbed_rna = x_rna.clone()

            # Get TF gene indices
            tf_gene_indices = [model.vae.tf_indices[tf_idx] for tf_idx in tf_indices]  # type: ignore[index]

            # Initialize logFC_decoded for edge case of n_iter=0
            logFC_decoded = torch.zeros_like(x_rna)

            # Iterate to steady state
            for iter_num in range(1, n_iter + 1):
                # Set all TF expressions to perturbation levels
                for tf_gene_idx, lv in zip(tf_gene_indices, levels, strict=False):
                    perturbed_rna[:, tf_gene_idx] = lv

                output_pert = model.vae(
                    perturbed_rna,
                    model.adj_E1,
                    use_ppi=model.vae.use_ppi,
                    use_mean=True,
                )

                # Compute logFC in latent space, then decode the DIFFERENCE
                logFC_latent = output_pert.z_rna - z_rna_orig
                logFC_decoded = model.vae.decoder_rna(logFC_latent)

                # Store intermediate if requested
                if return_intermediate:
                    logFC_per_iter[iter_num].append(logFC_decoded.cpu().numpy())

                # Update perturbed matrix
                perturbed_rna = x_rna + logFC_decoded

                # Clip to valid range
                perturbed_rna = torch.clamp(perturbed_rna, 0, clip_max)

                # Re-enforce TF perturbation levels
                for tf_gene_idx, lv in zip(tf_gene_indices, levels, strict=False):
                    perturbed_rna[:, tf_gene_idx] = lv

            logFC_all.append(logFC_decoded.cpu().numpy())

    if return_intermediate:
        return {i: np.concatenate(logFC_per_iter[i], axis=0) for i in range(1, n_iter + 1)}
    return np.concatenate(logFC_all, axis=0)
