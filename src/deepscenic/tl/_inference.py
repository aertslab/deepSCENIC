"""Inference functions for deepSCENIC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.sparse import issparse
from tqdm.auto import tqdm

from ._model import _check_mdata_matches_model

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
        Prefix for keys stored in mdata.obsm. Default: ``"X_deepscenic_"``.

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
    _check_mdata_matches_model(model, mdata)
    model.eval()

    if device is None:
        device = next(model.vae.parameters()).device
    else:
        model.to(device)

    # Get RNA data (ATAC not needed for inference); sparse data is densified per batch
    rna = mdata.mod["rna"].X

    n_cells = rna.shape[0]
    n_regions = len(model.region_names)

    # Output arrays, allocated once the first batch shows their width and dtype
    outputs: dict[str, np.ndarray] = {}

    def _store(key: str, rows: slice, values: np.ndarray, cols: slice = slice(None), width: int | None = None) -> None:
        if key not in outputs:
            outputs[key] = np.empty((n_cells, width or values.shape[1]), dtype=values.dtype)
        outputs[key][rows, cols] = values

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_cells, batch_size), desc="Extracting embeddings"):
            end_idx = min(i + batch_size, n_cells)
            rows = slice(i, end_idx)

            batch_rna = rna[i:end_idx]
            if issparse(batch_rna):
                batch_rna = batch_rna.toarray()
            x_rna = torch.FloatTensor(batch_rna).to(device)

            # Skip ATAC decoder in main forward pass to avoid OOM.
            # The ATAC decoder creates (batch, n_regions, 128) tensors
            # which is ~37 GB for 281k regions — exceeds most GPUs.
            output = model.vae(
                x_rna,
                model.adj_tf2r,
                use_mean=True,
                skip_atac=True,
            )

            _store("z_tf", rows, output.z_tf.cpu().numpy())
            _store("enh_act", rows, output.enh_act.cpu().numpy())
            _store("z_rna", rows, output.z_rna.cpu().numpy())
            _store("rna_rec", rows, output.x_rna_rec.cpu().numpy())

            # Decode ATAC in region chunks to avoid OOM, writing each chunk into place
            enh_act = output.enh_act  # (batch, n_regions) — stays on GPU
            for j in range(0, n_regions, region_chunk_size):
                chunk = enh_act[:, j : j + region_chunk_size]
                atac_chunk = model.vae.decoder_atac(chunk).cpu().numpy()
                _store("atac_rec", rows, atac_chunk, slice(j, j + atac_chunk.shape[1]), width=n_regions)

    if not outputs:
        raise ValueError("need at least one array to concatenate")

    # Store in mdata.obsm
    for key in ["z_tf", "enh_act", "z_rna", "rna_rec", "atac_rec"]:
        mdata.obsm[f"{key_prefix}{key}"] = outputs[key]
