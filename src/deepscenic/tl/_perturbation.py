"""Perturbation simulation for deepSCENIC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.sparse import issparse
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import mudata as md
    import pandas as pd

    from ._model import DeepSCENICModel


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
            # skip_atac=True: only z_rna is needed, skipping ATAC decoder
            # saves ~37 GB VRAM per MLP layer (281k regions × 128 hidden)
            output_orig = model.vae(
                x_rna,
                model.adj_E1,
                use_mean=True,
                skip_atac=True,
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

                # Forward pass with perturbed input (skip ATAC decoder)
                output_pert = model.vae(
                    perturbed_rna,
                    model.adj_E1,
                    use_mean=True,
                    skip_atac=True,
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
            # skip_atac=True: only z_rna is needed, skipping ATAC decoder
            # saves ~37 GB VRAM per MLP layer (281k regions × 128 hidden)
            output_orig = model.vae(
                x_rna,
                model.adj_E1,
                use_mean=True,
                skip_atac=True,
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

                # Forward pass with perturbed input (skip ATAC decoder)
                output_pert = model.vae(
                    perturbed_rna,
                    model.adj_E1,
                    use_mean=True,
                    skip_atac=True,
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


def process_perturbation_results(
    logFC: np.ndarray,
    mdata: md.MuData,
    tf_name: str | list[str],
    *,
    gene_subset: list[str] | None = None,
) -> pd.DataFrame:
    """
    Process perturbation results into a DataFrame for plotting.

    Computes mean log fold change across cells for each gene. The values
    represent the decoder-space effect of the perturbation (not literal
    log2 fold changes).

    Parameters
    ----------
    logFC
        Log fold change array from simulate_perturbation (n_cells, n_genes)
    mdata
        MuData used for simulation (provides gene names)
    tf_name
        Name of perturbed TF(s) - used for the 'tf' column
    gene_subset
        If provided, only include these genes in output

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'gene', 'tf', 'log2fc', 'abs_log2fc'.
        Ready for waterfall_perturbation, heatmap_perturbation, or
        dotplot_perturbation.

    Examples
    --------
    >>> logFC = ds.tl.simulate_perturbation(model, mdata, "SOX10")
    >>> results = ds.tl.process_perturbation_results(logFC, mdata, "SOX10")
    >>> ds.pl.waterfall_perturbation(results)

    >>> # For multi-TF comparison heatmaps
    >>> results_list = []
    >>> for tf in ["SOX10", "MITF", "PAX3"]:
    ...     logFC = ds.tl.simulate_perturbation(model, mdata, tf)
    ...     results_list.append(ds.tl.process_perturbation_results(logFC, mdata, tf))
    >>> combined = pd.concat(results_list)
    >>> ds.pl.heatmap_perturbation(combined)
    """
    import pandas as pd

    # Get gene names from mdata
    gene_names = list(mdata.mod["rna"].var_names)

    if logFC.shape[1] != len(gene_names):
        raise ValueError(f"logFC has {logFC.shape[1]} genes but mdata has {len(gene_names)} genes")

    # Handle tf_name as string or list
    if isinstance(tf_name, str):
        tf_label = tf_name
    else:
        tf_label = "+".join(tf_name)  # e.g., "SOX10+MITF" for multi-TF

    # Compute mean logFC per gene (across cells)
    mean_logfc = logFC.mean(axis=0)

    # Build result DataFrame
    result = pd.DataFrame(
        {
            "gene": gene_names,
            "tf": tf_label,
            "log2fc": mean_logfc,
            "abs_log2fc": np.abs(mean_logfc),
        }
    )

    # Filter to gene subset if provided
    if gene_subset is not None:
        result = result[result["gene"].isin(gene_subset)].copy()

    return result
