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
    region_chunk_size: int = 50000,
    device: str | torch.device | None = None,
    key_prefix: str = "X_deepscenic_",
) -> None:
    """
    Extract latent embeddings from trained model and store in mdata.obsm.

    Runs the model encoder on RNA expression to extract TF activity,
    enhancer activity, and gene regulatory signals. Only RNA data is
    required - ATAC reconstruction is predicted from the latent space.

    Results are stored in ``mdata.obsm`` with the specified prefix:
    - ``{key_prefix}z_tf``: TF latent activity (n_cells, n_tfs)
    - ``{key_prefix}enh_act``: Region activity (n_cells, n_regions)
    - ``{key_prefix}z_rna``: Gene regulatory signal (n_cells, n_genes)
    - ``{key_prefix}rna_rec``: Reconstructed RNA (n_cells, n_genes)
    - ``{key_prefix}atac_rec``: Reconstructed ATAC (n_cells, n_regions)

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA modality (ATAC not required for inference).
        Embeddings will be stored in mdata.obsm.
    batch_size
        Cells per batch for inference
    region_chunk_size
        Number of regions to decode at once through the ATAC decoder.
        The ATAC decoder creates (batch, n_regions, 128) intermediate
        tensors. With 281k regions this is ~37 GB per layer, exceeding
        most GPUs. Chunking to 50k regions reduces this to ~1.6 GB.
    device
        Device for inference (None = use model's current device)
    key_prefix
        Prefix for keys stored in mdata.obsm. Default: "X_deepscenic_"

    Returns
    -------
    None
        Embeddings are stored in mdata.obsm in-place.

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> ds.tl.to_latent(model, mdata)
    >>>
    >>> # Embeddings now in mdata.obsm:
    >>> z_tf = mdata.obsm["X_deepscenic_z_tf"]  # (n_cells, n_tfs)
    >>>
    >>> # Use scanpy for UMAP visualization
    >>> import scanpy as sc
    >>> sc.pp.neighbors(mdata, use_rep="X_deepscenic_z_tf")
    >>> sc.tl.umap(mdata)
    >>> sc.pl.umap(mdata, color="cell_state")
    """
    model.eval()

    if device is None:
        device = next(model.vae.parameters()).device
    else:
        model.to(device)

    # Get RNA data (ATAC not needed for inference)
    rna = mdata.mod["rna"].X

    # Convert sparse to dense
    if issparse(rna):
        rna = rna.toarray()

    n_cells = rna.shape[0]
    n_regions = len(model.region_names)

    # Initialize output arrays
    outputs: dict[str, list[np.ndarray]] = {
        "z_tf": [],
        "enh_act": [],
        "z_rna": [],
        "rna_rec": [],
        "atac_rec": [],
    }

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_cells, batch_size), desc="Extracting embeddings"):
            end_idx = min(i + batch_size, n_cells)

            x_rna = torch.FloatTensor(rna[i:end_idx]).to(device)

            # Skip ATAC decoder in main forward pass to avoid OOM.
            # The ATAC decoder creates (batch, n_regions, 128) tensors
            # which is ~37 GB for 281k regions — exceeds most GPUs.
            output = model.vae(
                x_rna,
                model.adj_E1,
                use_mean=True,
                skip_atac=True,
            )

            outputs["z_tf"].append(output.z_tf.cpu().numpy())
            outputs["enh_act"].append(output.enh_act.cpu().numpy())
            outputs["z_rna"].append(output.z_rna.cpu().numpy())
            outputs["rna_rec"].append(output.x_rna_rec.cpu().numpy())

            # Decode ATAC in region chunks to avoid OOM
            enh_act = output.enh_act  # (batch, n_regions) — stays on GPU
            atac_chunks = []
            for j in range(0, n_regions, region_chunk_size):
                chunk = enh_act[:, j : j + region_chunk_size]
                atac_chunks.append(model.vae.decoder_atac(chunk).cpu().numpy())
            outputs["atac_rec"].append(np.concatenate(atac_chunks, axis=1))

    # Concatenate batches and store in mdata.obsm
    for key, values in outputs.items():
        mdata.obsm[f"{key_prefix}{key}"] = np.concatenate(values, axis=0)
