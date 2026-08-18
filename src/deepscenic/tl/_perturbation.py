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
    tf_name: str | list[str],
    level: float | list[float] = 0.0,
    n_iter: int = 10,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    split: str | None = None,
    clip_percentile: float = 99.99,
    return_intermediate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Simulate one or more TF knockdowns/overexpressions.

    Perturbs a TF's expression and iteratively propagates the effect through
    the GRN to predict gene expression changes.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA + ATAC data
    tf_name
        TF name or list of TF names to perturb (e.g., ``"SOX10"`` or
        ``["SOX10", "MITF"]``).
    level
        Perturbation level or one level per TF. A scalar is applied to every
        TF (0 = knockout, >0 = overexpression).
    n_iter
        Iterations to reach steady state
    batch_size
        Cells per batch for inference
    device
        Device for computation
    split
        If specified, only process cells from this split
    clip_percentile
        Percentile for clipping perturbed values (default: 99.99)
    return_intermediate
        If True, return perturbed expression and logFC at each iteration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or tuple[dict, dict]
        If ``return_intermediate=False``, return ``(perturbed_matrix, logFC)``.
        Both arrays have shape ``(n_cells, n_genes)``. If
        ``return_intermediate=True``, return two dictionaries mapping each
        1-indexed iteration to its perturbed matrix and logFC array.

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> perturbed, logFC = ds.tl.simulate_perturbation(
    ...     model, mdata, tf_name="SOX10", level=0
    ... )

    >>> # Check convergence across iterations
    >>> perturbed_trace, logFC_trace = ds.tl.simulate_perturbation(
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

    tf_names = [tf_name] if isinstance(tf_name, str) else list(tf_name)
    if not tf_names:
        raise ValueError("tf_name must contain at least one TF")
    if len(set(tf_names)) != len(tf_names):
        raise ValueError("tf_name contains duplicate TF names")

    if isinstance(level, (int, float)):
        levels = [float(level)] * len(tf_names)
    else:
        levels = [float(value) for value in level]
        if len(levels) != len(tf_names):
            raise ValueError("level must have the same length as tf_name")

    tf_indices = []
    for name in tf_names:
        if name not in model.tf_names:
            raise ValueError(f"TF '{name}' not found. Available TFs: {model.tf_names[:10]}...")
        tf_indices.append(model.tf_names.index(name))
    tf_gene_indices = [model.vae.tf_indices[index] for index in tf_indices]  # type: ignore[index]

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
    perturbed_per_iter: dict[int, list[np.ndarray]] = {i: [] for i in range(1, n_iter + 1)}
    logFC_per_iter: dict[int, list[np.ndarray]] = {i: [] for i in range(1, n_iter + 1)}
    perturbed_all: list[np.ndarray] = []
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
                model.adj_tf2r,
                use_mean=True,
                skip_atac=True,
            )
            z_rna_orig = output_orig.z_rna

            # Create perturbed expression matrix
            perturbed_rna = x_rna.clone()

            # Initialize logFC_decoded for edge case of n_iter=0
            logFC_decoded = torch.zeros_like(x_rna)

            for tf_gene_idx, perturbation_level in zip(tf_gene_indices, levels, strict=True):
                perturbed_rna[:, tf_gene_idx] = perturbation_level

            # Iterate to steady state
            for iter_num in range(1, n_iter + 1):
                # Set TF expressions to their perturbation levels
                for tf_gene_idx, perturbation_level in zip(tf_gene_indices, levels, strict=True):
                    perturbed_rna[:, tf_gene_idx] = perturbation_level

                # Forward pass with perturbed input (skip ATAC decoder)
                output_pert = model.vae(
                    perturbed_rna,
                    model.adj_tf2r,
                    use_mean=True,
                    skip_atac=True,
                )

                # Compute logFC in latent space, then decode the DIFFERENCE
                # This matches legacy: decode(z_pert - z_orig), NOT decode(z_pert) - decode(z_orig)
                logFC_latent = output_pert.z_rna - z_rna_orig
                logFC_decoded = model.vae.decoder_rna(logFC_latent)

                # Update perturbed matrix: original + decoded fold change
                perturbed_rna = x_rna + logFC_decoded

                # Clip to valid range [0, p99]
                perturbed_rna = torch.clamp(perturbed_rna, 0, clip_max)

                # Re-enforce TF perturbation levels after clipping
                for tf_gene_idx, perturbation_level in zip(tf_gene_indices, levels, strict=True):
                    perturbed_rna[:, tf_gene_idx] = perturbation_level

                if return_intermediate:
                    perturbed_per_iter[iter_num].append(perturbed_rna.cpu().numpy().copy())
                    logFC_per_iter[iter_num].append(logFC_decoded.cpu().numpy())

            perturbed_all.append(perturbed_rna.cpu().numpy())
            logFC_all.append(logFC_decoded.cpu().numpy())

    if return_intermediate:
        perturbed_trace = {
            i: np.concatenate(perturbed_per_iter[i], axis=0) for i in range(1, n_iter + 1)
        }
        logFC_trace = {i: np.concatenate(logFC_per_iter[i], axis=0) for i in range(1, n_iter + 1)}
        return perturbed_trace, logFC_trace
    return np.concatenate(perturbed_all, axis=0), np.concatenate(logFC_all, axis=0)


def simulate_multi_perturbation(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_names: list[str],
    levels: list[float] | None = None,
    n_iter: int = 10,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    split: str | None = None,
    clip_percentile: float = 99.99,
    return_intermediate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Simulate multiple TF perturbations.

    This compatibility wrapper delegates to ``simulate_perturbation``.
    New code can pass a list directly through ``tf_name`` instead.
    """
    perturbation_levels = [0.0] * len(tf_names) if levels is None else levels
    return simulate_perturbation(
        model=model,
        mdata=mdata,
        tf_name=tf_names,
        level=perturbation_levels,
        n_iter=n_iter,
        batch_size=batch_size,
        device=device,
        split=split,
        clip_percentile=clip_percentile,
        return_intermediate=return_intermediate,
    )


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
    >>> perturbed, logFC = ds.tl.simulate_perturbation(model, mdata, "SOX10")
    >>> results = ds.tl.process_perturbation_results(logFC, mdata, "SOX10")
    >>> ds.pl.waterfall_perturbation(results)

    >>> # For multi-TF comparison heatmaps
    >>> results_list = []
    >>> for tf in ["SOX10", "MITF", "PAX3"]:
    ...     _, logFC = ds.tl.simulate_perturbation(model, mdata, tf)
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
