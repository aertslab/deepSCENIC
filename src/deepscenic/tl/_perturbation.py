"""Perturbation simulation for deepSCENIC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.sparse import issparse
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import mudata as md

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
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Log fold change predictions (n_cells, n_genes)

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> logFC = ds.tl.simulate_perturbation(model, mdata, tf_name="SOX10", level=0)
    >>> # logFC contains predicted expression changes for all genes
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
        atac = mdata.mod["atac"][mask].X
    else:
        rna = mdata.mod["rna"].X
        atac = mdata.mod["atac"].X

    # Convert sparse to dense
    if issparse(rna):
        rna = rna.toarray()
    if issparse(atac):
        atac = atac.toarray()

    n_cells = rna.shape[0]
    n_genes = len(model.gene_names)

    # Output array
    logFC_all = []

    # Build E2 matrix once
    with torch.no_grad():
        E2 = model.vae._build_e2_dense()

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_cells, batch_size), desc="Simulating perturbation"):
            end_idx = min(i + batch_size, n_cells)
            batch_rna = rna[i:end_idx].copy()
            batch_atac = atac[i:end_idx]

            x_rna = torch.FloatTensor(batch_rna).to(device)
            x_atac = torch.FloatTensor(batch_atac).to(device)

            # Get original z_rna (baseline)
            output_orig = model.vae(
                x_rna,
                x_atac,
                model.adj_E1,
                use_ppi=model.vae.use_ppi,
                use_mean=True,
            )
            z_rna_orig = output_orig.z_rna

            # Create perturbed expression matrix
            perturbed_rna = x_rna.clone()

            # Iterate to steady state
            for _ in range(n_iter):
                # Set TF expression to perturbation level
                # TF index in full gene matrix
                tf_gene_idx = model.vae.tf_indices[tf_idx]  # type: ignore[index]
                perturbed_rna[:, tf_gene_idx] = level

                # Forward pass with perturbed input
                output_pert = model.vae(
                    perturbed_rna,
                    x_atac,
                    model.adj_E1,
                    use_ppi=model.vae.use_ppi,
                    use_mean=True,
                )

                # Compute logFC in gene regulatory signal space
                logFC = output_pert.z_rna - z_rna_orig

                # Decode to get expression changes
                x_rna_rec_orig = model.vae.decoder_rna(z_rna_orig)
                x_rna_rec_pert = model.vae.decoder_rna(output_pert.z_rna)

                # Update perturbed matrix for next iteration
                # Use original + decoded change
                perturbed_rna = x_rna + (x_rna_rec_pert - x_rna_rec_orig)
                perturbed_rna[:, tf_gene_idx] = level

            # Final logFC (in expression space)
            logFC_final = (x_rna_rec_pert - x_rna_rec_orig).cpu().numpy()
            logFC_all.append(logFC_final)

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
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Log fold change predictions (n_cells, n_genes)
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
        atac = mdata.mod["atac"][mask].X
    else:
        rna = mdata.mod["rna"].X
        atac = mdata.mod["atac"].X

    if issparse(rna):
        rna = rna.toarray()
    if issparse(atac):
        atac = atac.toarray()

    n_cells = rna.shape[0]
    logFC_all = []

    with torch.no_grad():
        for i in tqdm(range(0, n_cells, batch_size), desc="Simulating multi-perturbation"):
            end_idx = min(i + batch_size, n_cells)
            batch_rna = rna[i:end_idx].copy()
            batch_atac = atac[i:end_idx]

            x_rna = torch.FloatTensor(batch_rna).to(device)
            x_atac = torch.FloatTensor(batch_atac).to(device)

            # Get original
            output_orig = model.vae(
                x_rna,
                x_atac,
                model.adj_E1,
                use_ppi=model.vae.use_ppi,
                use_mean=True,
            )
            z_rna_orig = output_orig.z_rna

            # Create perturbed expression matrix
            perturbed_rna = x_rna.clone()

            # Iterate to steady state
            for _ in range(n_iter):
                # Set all TF expressions to perturbation levels
                for tf_idx, level in zip(tf_indices, levels, strict=False):
                    tf_gene_idx = model.vae.tf_indices[tf_idx]  # type: ignore[index]
                    perturbed_rna[:, tf_gene_idx] = level

                output_pert = model.vae(
                    perturbed_rna,
                    x_atac,
                    model.adj_E1,
                    use_ppi=model.vae.use_ppi,
                    use_mean=True,
                )

                x_rna_rec_orig = model.vae.decoder_rna(z_rna_orig)
                x_rna_rec_pert = model.vae.decoder_rna(output_pert.z_rna)

                perturbed_rna = x_rna + (x_rna_rec_pert - x_rna_rec_orig)
                for tf_idx, level in zip(tf_indices, levels, strict=False):
                    tf_gene_idx = model.vae.tf_indices[tf_idx]  # type: ignore[index]
                    perturbed_rna[:, tf_gene_idx] = level

            logFC_final = (x_rna_rec_pert - x_rna_rec_orig).cpu().numpy()
            logFC_all.append(logFC_final)

    return np.concatenate(logFC_all, axis=0)
