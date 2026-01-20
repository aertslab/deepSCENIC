"""Inference functions for deepSCENIC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.sparse import issparse
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import mudata as md

    from ._model import DeepSCENICModel


def to_latent(
    model: DeepSCENICModel,
    mdata: md.MuData,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    split: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract embeddings from trained model.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA + ATAC data
    batch_size
        Cells per batch for inference
    device
        Device for inference (None = use model's current device)
    split
        If specified, only process cells from this split ('train' or 'test')

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with embeddings:
        - 'z_tf': TF latent activity (n_cells, n_tfs)
        - 'enh_act': Region activity (n_cells, n_regions)
        - 'z_rna': Gene regulatory signal (n_cells, n_genes)
        - 'x_rna_rec': Reconstructed RNA (n_cells, n_genes)
        - 'x_atac_rec': Reconstructed ATAC (n_cells, n_regions)
        - 'x_rna_ppi': PPI modulation weights (n_cells, n_tfs)

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> embeddings = ds.tl.to_latent(model, mdata)
    >>> z_tf = embeddings['z_tf']  # (n_cells, n_tfs)
    """
    model.eval()

    if device is None:
        device = next(model.vae.parameters()).device
    else:
        model.to(device)

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

    # Initialize output arrays
    outputs: dict[str, list[np.ndarray]] = {
        "z_tf": [],
        "enh_act": [],
        "z_rna": [],
        "x_rna_rec": [],
        "x_atac_rec": [],
        "x_rna_ppi": [],
    }

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_cells, batch_size), desc="Extracting embeddings"):
            end_idx = min(i + batch_size, n_cells)

            x_rna = torch.FloatTensor(rna[i:end_idx]).to(device)
            x_atac = torch.FloatTensor(atac[i:end_idx]).to(device)

            output = model.vae(
                x_rna,
                x_atac,
                model.adj_E1,
                use_ppi=model.vae.use_ppi,
                use_mean=True,  # Deterministic inference
            )

            outputs["z_tf"].append(output.z_tf.cpu().numpy())
            outputs["enh_act"].append(output.enh_act.cpu().numpy())
            outputs["z_rna"].append(output.z_rna.cpu().numpy())
            outputs["x_rna_rec"].append(output.x_rna_rec.cpu().numpy())
            outputs["x_atac_rec"].append(output.x_atac_rec.cpu().numpy())
            outputs["x_rna_ppi"].append(output.x_rna_ppi.cpu().numpy())

    # Concatenate batches
    return {k: np.concatenate(v, axis=0) for k, v in outputs.items()}
