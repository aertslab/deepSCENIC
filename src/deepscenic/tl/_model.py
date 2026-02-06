"""Trained model wrapper for deepSCENIC."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import nn

from ._training_state import ModelConfig, TrainingHistory

log = logging.getLogger("deepscenic.tl")

if TYPE_CHECKING:
    from mudata import MuData

    from ..models import DeepSCENICVAE, MotifNet


@dataclass
class TrainingState:
    """Training state for resuming training."""

    optimizer_state_dict: dict
    scheduler_state_dict: dict | None
    epoch: int
    best_loss: float


@dataclass
class DeepSCENICModel:
    """Trained deepSCENIC model container.

    This is the object returned by `ds.tl.train()` and loaded by `ds.tl.load_model()`.
    Contains all trained components and metadata for self-contained inference.

    Attributes
    ----------
    vae
        Trained VAE model.
    motifnet
        Trained MotifNet context head.
    sequence_model
        Sequence embedding model (Enformer or custom nn.Module).
    adj_E1
        Cached E1 matrix (n_regions, n_tfs).
    config
        Model configuration.
    tf_names
        TF names in order.
    gene_names
        Gene names in order.
    region_names
        Region names in order.
    history
        Training history with loss metrics per epoch. Available when model was
        just trained or loaded from a file that includes history.
    training_state
        Training state for resuming (optimizer, scheduler, epoch, best_loss).
        Only present when loaded from a checkpoint saved with include_training_state=True.

    Examples
    --------
    >>> model = ds.tl.train(mdata, epochs=100)
    >>> model.save("model.pt")
    >>> loaded = ds.tl.load_model("model.pt")
    >>> embeddings = ds.tl.to_latent(loaded, mdata)
    """

    vae: DeepSCENICVAE
    motifnet: MotifNet
    sequence_model: nn.Module
    adj_E1: torch.Tensor
    config: ModelConfig
    tf_names: list[str]
    gene_names: list[str]
    region_names: list[str]
    history: TrainingHistory | None = None
    training_state: TrainingState | None = None

    def save(
        self,
        path: str | Path,
        *,
        include_training_state: bool = False,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        epoch: int | None = None,
        best_loss: float | None = None,
    ) -> None:
        """Save model to file.

        Parameters
        ----------
        path
            Output file path (.pt)
        include_training_state
            If True, include optimizer/scheduler state for resuming training.
        optimizer
            Optimizer to save state from. Required if include_training_state=True.
        scheduler
            Optional LR scheduler to save state from.
        epoch
            Current epoch number.
        best_loss
            Best validation loss.

        Examples
        --------
        Save for inference:

        >>> model.save("model.pt")

        Save with training state:

        >>> model.save(
        ...     "checkpoint.pt",
        ...     include_training_state=True,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     epoch=current_epoch,
        ...     best_loss=best_val_loss,
        ... )
        """
        # Detect if sequence model is Enformer or custom
        is_custom_sequence_model = type(self.sequence_model).__name__ != "Enformer"

        data = {
            "vae_state_dict": self.vae.state_dict(),
            "motifnet_state_dict": self.motifnet.state_dict(),
            "sequence_model_state_dict": self.sequence_model.state_dict(),
            "adj_E1": self.adj_E1,
            "config": self.config.to_dict(),
            "tf_names": self.tf_names,
            "gene_names": self.gene_names,
            "region_names": self.region_names,
            # Store model architecture params for reconstruction
            "n_tfs": self.vae.n_tfs,
            "n_genes": self.vae.n_genes,
            "n_regions": self.vae.n_regions,
            "n_hidden": self.vae.n_hidden,
            "n_batches": self.vae.n_batches,
            # Store buffer tensors for VAE reconstruction
            "tf_indices": self.vae.tf_indices,
            "gene_indices": self.vae.gene_indices,
            "region_indices": self.vae.region_indices,
            "r2g_indices": self.vae.r2g_indices,
            "r2g_distances": self.vae.r2g_distances,
            # PPI buffers (if present)
            "use_ppi": self.vae.use_ppi,
            "ppi_edge_index": getattr(self.vae, "ppi_edge_index", None),
            "ppi_genes_idx": getattr(self.vae, "ppi_genes_idx", None),
            "ppi_tfs_idx_keys": getattr(self.vae, "ppi_tfs_idx_keys", None),
            "ppi_tfs_idx_values": getattr(self.vae, "ppi_tfs_idx_values", None),
            # Custom sequence model flag
            "is_custom_sequence_model": is_custom_sequence_model,
        }
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if self.history is not None:
            data["history"] = self.history.to_dict()

        # Optional training state for resumption
        if include_training_state:
            if optimizer is None:
                raise ValueError("optimizer required when include_training_state=True")
            data["optimizer_state_dict"] = optimizer.state_dict()
            data["scheduler_state_dict"] = scheduler.state_dict() if scheduler else None
            data["epoch"] = epoch
            data["best_loss"] = best_loss

        torch.save(data, path)

        log.info(f"Saved model to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        sequence_model: nn.Module | None = None,
    ) -> DeepSCENICModel:
        """
        Load model from file.

        Parameters
        ----------
        path
            Model file path (.pt)
        device
            Device to load model to
        sequence_model
            Custom sequence model instance. Required when loading a model that
            was trained with a custom sequence model. The model architecture
            must match what was used during training. For default Enformer models,
            leave as None - Enformer will be instantiated and weights loaded.

        Returns
        -------
        DeepSCENICModel
            Loaded model ready for inference. If the file was saved with
            include_training_state=True, model.training_state will contain
            optimizer/scheduler state for resuming training.

        Examples
        --------
        Loading a model for inference:

        >>> model = DeepSCENICModel.load("model.pt")

        Loading a checkpoint and resuming training:

        >>> model = DeepSCENICModel.load("checkpoint.pt", device="cuda")
        >>> if model.training_state:
        ...     optimizer.load_state_dict(model.training_state.optimizer_state_dict)
        ...     start_epoch = model.training_state.epoch + 1
        """
        from ..models import DeepSCENICVAE, MotifNet

        data = torch.load(path, map_location=device, weights_only=False)

        # Reconstruct config (handles both old TrainingConfig and new ModelConfig)
        config = ModelConfig.from_dict(data["config"])

        # Reconstruct VAE (PPI removed for initial release - legacy PPI weights are ignored)
        if data.get("use_ppi", False):
            log.info("  Note: Legacy checkpoint had PPI enabled - PPI is removed in this release")

        # Ensure index tensors have correct dtype (int64) for scatter operations
        # This fixes checkpoints saved before the dtype fix
        r2g_indices = data["r2g_indices"].to(torch.long)
        tf_indices = data["tf_indices"].to(torch.long)
        gene_indices = data["gene_indices"].to(torch.long)
        region_indices = data["region_indices"].to(torch.long)

        vae = DeepSCENICVAE(
            n_tfs=data["n_tfs"],
            n_genes=data["n_genes"],
            n_regions=data["n_regions"],
            r2g_indices=r2g_indices,
            r2g_distances=data["r2g_distances"],
            tf_indices=tf_indices,
            gene_indices=gene_indices,
            region_indices=region_indices,
            n_hidden=data["n_hidden"],
            n_batches=data.get("n_batches", 0),
        )

        # Fix dtypes in state dict before loading (buffers would overwrite our fixed tensors)
        vae_state = data["vae_state_dict"]
        for key in ["r2g_indices", "tf_indices", "gene_indices", "region_indices"]:
            if key in vae_state:
                vae_state[key] = vae_state[key].to(torch.long)
        vae.load_state_dict(vae_state)
        vae.to(device)
        vae.eval()

        # Reconstruct MotifNet
        motifnet = MotifNet(
            n_tfs=data["n_tfs"],
            bottleneck_size=config.bottleneck_size,
            emb_len=config.emb_len,
        )
        motifnet.load_state_dict(data["motifnet_state_dict"])
        motifnet.to(device)
        motifnet.eval()

        # Reconstruct sequence model
        is_custom = data.get("is_custom_sequence_model", False)
        if is_custom:
            if sequence_model is None:
                raise ValueError(
                    "This model was trained with a custom sequence model. "
                    "You must provide the model instance via sequence_model parameter."
                )
            seq_model = sequence_model
        else:
            # Default: instantiate Enformer (weights will be loaded from saved state)
            from enformer_pytorch import Enformer

            seq_model = Enformer.from_pretrained(
                "EleutherAI/enformer-official-rough",
                target_length=-1,
            )

        seq_model.load_state_dict(data["sequence_model_state_dict"])
        seq_model.to(device)
        seq_model.eval()

        log.info(f"Loaded model from {path}: {data['n_genes']} genes, {data['n_tfs']} TFs, {data['n_regions']} regions")

        # Restore history if present
        history = None
        if "history" in data:
            history = TrainingHistory.from_dict(data["history"])

        # Restore training state if present
        training_state = None
        if "optimizer_state_dict" in data:
            training_state = TrainingState(
                optimizer_state_dict=data["optimizer_state_dict"],
                scheduler_state_dict=data.get("scheduler_state_dict"),
                epoch=data.get("epoch", 0),
                best_loss=data.get("best_loss", float("inf")),
            )
            log.info(f"  Training state: epoch={training_state.epoch}, best_loss={training_state.best_loss:.6f}")

        return cls(
            vae=vae,
            motifnet=motifnet,
            sequence_model=seq_model,
            adj_E1=data["adj_E1"].to(device),
            config=config,
            tf_names=data["tf_names"],
            gene_names=data["gene_names"],
            region_names=data["region_names"],
            history=history,
            training_state=training_state,
        )

    def to(self, device: str | torch.device) -> DeepSCENICModel:
        """Move model to device."""
        self.vae.to(device)
        self.motifnet.to(device)
        self.sequence_model.to(device)
        self.adj_E1 = self.adj_E1.to(device)
        return self

    def eval(self) -> DeepSCENICModel:
        """Set all components to eval mode."""
        self.vae.eval()
        self.motifnet.eval()
        self.sequence_model.eval()
        return self

    def train(self) -> DeepSCENICModel:
        """Set all components to train mode."""
        self.vae.train()
        self.motifnet.train()
        self.sequence_model.train()
        return self


def load_model(
    path: str | Path,
    device: str = "cpu",
    sequence_model: nn.Module | None = None,
) -> DeepSCENICModel:
    """
    Load a trained deepSCENIC model.

    This is the unified loading function for both inference-only models and
    checkpoints with training state. If the file was saved with
    include_training_state=True, the model.training_state attribute will
    contain optimizer/scheduler state for resuming training.

    Parameters
    ----------
    path
        Path to saved model file (.pt)
    device
        Device to load model to
    sequence_model
        Custom sequence model instance. Required when loading a model that
        was trained with a custom sequence model.

    Returns
    -------
    DeepSCENICModel
        Loaded model ready for inference. Check model.training_state for
        resumption info if loading a checkpoint.

    Examples
    --------
    Load for inference:

    >>> model = ds.tl.load_model("model.pt")

    Load checkpoint and resume training:

    >>> model = ds.tl.load_model("checkpoint.pt", device="cuda")
    >>> if model.training_state:
    ...     optimizer.load_state_dict(model.training_state.optimizer_state_dict)
    ...     start_epoch = model.training_state.epoch + 1
    """
    return DeepSCENICModel.load(path, device=device, sequence_model=sequence_model)


def load_legacy_data(
    rna_path: str | Path,
    atac_path: str | Path,
    r2g_path: str | Path | None = None,
    tf_path: str | Path | None = None,
    n_tfs: int | None = None,
) -> MuData:
    """
    Load legacy training data files into MuData format for loading legacy models.

    Legacy deepSCENIC models were trained with separate .h5ad and .npz files.
    This function loads those files and creates a MuData with the exact
    structure needed to load the corresponding legacy model.

    Parameters
    ----------
    rna_path
        Path to RNA AnnData file. For model loading, use the training file
        (e.g., ``raw_exprMat_train.h5ad``). For inference on all cells,
        use the full file (e.g., ``raw_exprMat.h5ad``).
    atac_path
        Path to ATAC AnnData file. For model loading, use the training file
        (e.g., ``fragment_matrix_train.h5ad``). For inference on all cells,
        use the full file (e.g., ``fragment_matrix.h5ad``).
    r2g_path
        Path to r2g penalty matrix (e.g., ``r2gpenalty_train.npz``).
        Required for model loading. Can be omitted for inference-only use.
    tf_path
        Path to TF list file (e.g., ``TFs.txt``), one TF name per line.
        If None, uses ``n_tfs`` to mark the first N genes as TFs.
    n_tfs
        Number of TFs (first N genes are TFs). Only used if ``tf_path`` is None.
        Legacy format always has TFs as the first genes in the RNA data.

    Returns
    -------
    MuData with:
        - 'rna' modality with 'is_tf' column in var
        - 'atac' modality with parsed coordinates
        - 'r2g' in uns with matrix and metadata (if r2g_path provided)

    Examples
    --------
    >>> # Load legacy training data (for model loading)
    >>> mdata_train = ds.tl.load_legacy_data(
    ...     rna_path="MM_lines/raw_exprMat_train.h5ad",
    ...     atac_path="MM_lines/fragment_matrix_train.h5ad",
    ...     r2g_path="MM_lines/r2gpenalty_train.npz",
    ...     tf_path="MM_lines/TFs.txt",
    ... )
    >>> model = ds.tl.load_legacy_model(..., mdata=mdata_train, ...)

    >>> # Load full data (for inference on all cells)
    >>> mdata_full = ds.tl.load_legacy_data(
    ...     rna_path="MM_lines/raw_exprMat.h5ad",
    ...     atac_path="MM_lines/fragment_matrix.h5ad",
    ...     tf_path="MM_lines/TFs.txt",
    ...     # r2g_path not needed for inference
    ... )

    See Also
    --------
    load_legacy_model : Load legacy model using the MuData from this function
    """
    import mudata as md
    import scanpy as sc
    from scipy import sparse

    # Load RNA and ATAC data
    log.info(f"Loading RNA data from {rna_path}")
    rna = sc.read_h5ad(rna_path)
    log.info(f"  RNA: {rna.n_obs} cells × {rna.n_vars} genes")

    log.info(f"Loading ATAC data from {atac_path}")
    atac = sc.read_h5ad(atac_path)
    log.info(f"  ATAC: {atac.n_obs} cells × {atac.n_vars} regions")

    # Load r2g penalty matrix (optional)
    r2g_loaded = None
    if r2g_path is not None:
        log.info(f"Loading R2G matrix from {r2g_path}")
        r2g_loaded = sparse.load_npz(r2g_path)
        log.info(f"  R2G: {r2g_loaded.shape[0]} regions × {r2g_loaded.shape[1]} genes, {r2g_loaded.nnz} links")

        # Validate dimensions
        if r2g_loaded.shape[0] != atac.n_vars:
            raise ValueError(
                f"R2G matrix rows ({r2g_loaded.shape[0]}) != ATAC regions ({atac.n_vars}). "
                "Make sure you're using matching legacy files."
            )
        if r2g_loaded.shape[1] != rna.n_vars:
            raise ValueError(
                f"R2G matrix columns ({r2g_loaded.shape[1]}) != RNA genes ({rna.n_vars}). "
                "Make sure you're using matching legacy files."
            )
    else:
        log.info("  R2G matrix not provided (inference-only mode)")

    # Mark TFs
    if tf_path is not None:
        # Load TF list from file
        tf_path = Path(tf_path)
        with open(tf_path) as f:
            tf_names = [line.strip() for line in f if line.strip()]
        log.info(f"  Loaded {len(tf_names)} TFs from {tf_path}")

        # Mark TFs in RNA var
        rna.var["is_tf"] = rna.var_names.isin(tf_names)
        n_tfs_found = rna.var["is_tf"].sum()
        log.info(f"  Marked {n_tfs_found} TFs in RNA data")
    elif n_tfs is not None:
        # Use first N genes as TFs (legacy format)
        log.info(f"  Marking first {n_tfs} genes as TFs")
        is_tf_values = [True] * n_tfs + [False] * (rna.n_vars - n_tfs)
        rna.var["is_tf"] = is_tf_values
    else:
        raise ValueError("Either tf_path or n_tfs must be provided")

    # Parse ATAC region coordinates if not already present
    if "chromosome" not in atac.var.columns:
        log.info("  Parsing ATAC region coordinates...")
        # Parse region names like "chr1:1000-2000"
        coords = []
        for region in atac.var_names:
            if ":" in region and "-" in region:
                chrom, pos = region.split(":")
                start, end = pos.split("-")
                coords.append({"chromosome": chrom, "start": int(start), "end": int(end)})
            else:
                coords.append({"chromosome": None, "start": None, "end": None})
        coord_df = pd.DataFrame(coords, index=atac.var_names)
        atac.var["chromosome"] = coord_df["chromosome"]
        atac.var["start"] = coord_df["start"]
        atac.var["end"] = coord_df["end"]

    # Create MuData
    log.info("Creating MuData...")
    mdata = md.MuData({"rna": rna, "atac": atac})

    # Store r2g in uns (matching format from compute_r2g_penalty)
    if r2g_loaded is not None:
        mdata.uns["r2g"] = {
            "matrix": r2g_loaded.tocsr(),  # Ensure CSR format
            "config": {
                "n_links": r2g_loaded.nnz,
                "source": "legacy",
            },
            "region_names": atac.var_names.tolist(),
            "gene_names": rna.var_names.tolist(),
        }

    # Store TF order in rna.uns (first n_tfs genes)
    tf_mask = rna.var["is_tf"].values
    tf_names_list = rna.var_names[tf_mask].tolist()
    mdata.mod["rna"].uns["tf_order"] = tf_names_list

    log.info(f"Created MuData: {mdata['rna'].n_vars} genes, {mdata['atac'].n_vars} regions, {len(tf_names_list)} TFs")

    return mdata


def load_legacy_model(
    vae_path: str | Path,
    motifnet_path: str | Path,
    enformer_path: str | Path,
    mdata: MuData,
    e1_path: str | Path | None = None,
    *,
    device: str = "cuda",
) -> DeepSCENICModel:
    """
    Load legacy deepSCENIC model (3 separate .pth files) into unified format.

    Legacy models were saved as three separate checkpoint files. This function
    reconstructs the model architecture from preprocessed h5mu data and loads
    the legacy weights.

    Parameters
    ----------
    vae_path
        Path to VAE checkpoint (e.g., best_model.pth)
    motifnet_path
        Path to MotifNet checkpoint (e.g., best_model_tf2r.pth)
    enformer_path
        Path to Enformer checkpoint (e.g., best_model_tf2r_encoder.pth)
    mdata
        Preprocessed MuData with RNA and ATAC modalities. Must have:

        - ``mdata['rna'].var['is_tf']`` column (from ds.pp.mark_tfs())
        - ``mdata.uns['r2g']`` (from ds.pp.compute_r2g_penalty())
    e1_path
        Path to pre-computed E1 matrix (E1.pkl). If None, reconstructs E1 by
        running Enformer + MotifNet inference on all regions. This requires
        a genome to be registered via ``ds.register_genome()``.
    device
        Device to load model to

    Returns
    -------
    DeepSCENICModel
        Loaded model ready for inference

    Examples
    --------
    Load with pre-computed E1:

    >>> mdata = ds.read("preprocessed.h5mu")
    >>> model = ds.tl.load_legacy_model(
    ...     vae_path="results/best_model.pth",
    ...     motifnet_path="results/best_model_tf2r.pth",
    ...     enformer_path="results/best_model_tf2r_encoder.pth",
    ...     mdata=mdata,
    ...     e1_path="results/E1.pkl",
    ... )

    Reconstruct E1 from model weights (slower, requires registered genome):

    >>> ds.register_genome("/path/to/hg38.fa")
    >>> model = ds.tl.load_legacy_model(
    ...     vae_path="results/best_model.pth",
    ...     motifnet_path="results/best_model_tf2r.pth",
    ...     enformer_path="results/best_model_tf2r_encoder.pth",
    ...     mdata=mdata,
    ...     e1_path=None,  # Reconstruct from model
    ... )

    Save in new format:

    >>> model.save("converted_model.pt")

    Notes
    -----
    Legacy checkpoint format:
        {'epoch': int, 'model_state_dict': OrderedDict, 'optimizer_state_dict': ...}

    The state dict keys are identical between legacy and new code:
        - VAE: 'adj_E2', 'inference_rna.*', 'generative_rna.*', 'generative_atac.*'
        - MotifNet: 'ctx_head_layer.weight', 'ctx_lin.weight', 'ctx_lin.bias'
        - Enformer: Full Enformer state dict
        - Note: 'PPInet.*' keys in legacy models are ignored (PPI removed in this release)

    See Also
    --------
    load_model : Load new format model
    extract_grn : Extract GRN matrices from loaded model
    """
    from ..models import DeepSCENICVAE, MotifNet

    # Convert paths to Path objects
    vae_path = Path(vae_path)
    motifnet_path = Path(motifnet_path)
    enformer_path = Path(enformer_path)

    # =========================================================================
    # Step 1: Extract data from h5mu
    # =========================================================================
    log.info("Extracting data from h5mu...")

    # Get gene and region names
    gene_names = mdata["rna"].var_names.tolist()
    region_names = mdata["atac"].var_names.tolist()
    n_genes = len(gene_names)
    n_regions = len(region_names)

    # TF indices from is_tf column
    if "is_tf" not in mdata["rna"].var.columns:
        raise ValueError("mdata missing 'is_tf' column - run ds.pp.mark_tfs() first")
    tf_mask = mdata["rna"].var["is_tf"].values
    tf_indices = torch.tensor(np.where(tf_mask)[0], dtype=torch.long)
    tf_names = mdata["rna"].var_names[tf_mask].tolist()
    n_tfs = len(tf_names)

    log.info(f"  RNA: {n_genes} genes, ATAC: {n_regions} regions, TFs: {n_tfs}")

    # Gene indices (which genes to reconstruct) - use all genes for legacy models
    gene_indices = torch.arange(n_genes, dtype=torch.long)
    # Region indices (which regions to reconstruct) - use all regions for legacy models
    region_indices = torch.arange(n_regions, dtype=torch.long)

    # Get r2g from uns
    if "r2g" not in mdata.uns:
        raise ValueError("mdata missing 'r2g' - run ds.pp.compute_r2g_penalty() first")
    r2g_sparse = mdata.uns["r2g"]["matrix"]
    r2g_coo = r2g_sparse.tocoo()
    r2g_indices = torch.tensor(np.vstack([r2g_coo.row, r2g_coo.col]), dtype=torch.long)
    r2g_distances = torch.tensor(r2g_coo.data, dtype=torch.float32)

    log.info(f"  R2G links: {r2g_indices.shape[1]}")

    # =========================================================================
    # Step 2: Check for PPI in h5mu (PPI removed for initial release)
    # =========================================================================
    if "ppi_edge_index" in mdata.uns:
        log.info("  Note: Legacy data has PPI info - PPI is removed in this release, ignoring")

    # =========================================================================
    # Step 3: Construct VAE and load state dict
    # =========================================================================
    log.info("Loading VAE...")
    vae = DeepSCENICVAE(
        n_tfs=n_tfs,
        n_genes=n_genes,
        n_regions=n_regions,
        r2g_indices=r2g_indices,
        r2g_distances=r2g_distances,
        tf_indices=tf_indices,
        gene_indices=gene_indices,
        region_indices=region_indices,
        n_hidden=128,  # Legacy default
        binary_atac=False,  # Legacy default
        n_batches=0,  # Will detect from state dict
    )

    # Load legacy checkpoint
    vae_ckpt = torch.load(vae_path, map_location=device)
    vae_state = vae_ckpt["model_state_dict"]

    # Check if batch correction was used (detect from state dict)
    has_batch = "batch_layer_rna.0.weight" in vae_state
    if has_batch:
        # Need to reinitialize VAE with batch layers
        # Infer n_batches from weight shape
        n_batches = vae_state["batch_layer_rna.0.weight"].shape[1] - n_genes
        log.info(f"  Detected batch correction with {n_batches} batches")
        vae = DeepSCENICVAE(
            n_tfs=n_tfs,
            n_genes=n_genes,
            n_regions=n_regions,
            r2g_indices=r2g_indices,
            r2g_distances=r2g_distances,
            tf_indices=tf_indices,
            gene_indices=gene_indices,
            region_indices=region_indices,
            n_hidden=128,
            binary_atac=False,
            n_batches=n_batches,
        )

    # Map legacy state dict keys to new format
    vae_state = _map_legacy_vae_state_dict(vae_state)

    # PPI removed for initial release - always remove PPI keys from legacy state dict
    vae_state = {k: v for k, v in vae_state.items() if not k.startswith("ppi.")}

    # Log ignored keys (e.g., generative_rec.* dead code in PPI models)
    expected_keys = set(vae.state_dict().keys())
    ignored_keys = [k for k in vae_state.keys() if k not in expected_keys]
    if ignored_keys:
        log.info(
            f"  Ignoring {len(ignored_keys)} unused legacy keys: {ignored_keys[:3]}{'...' if len(ignored_keys) > 3 else ''}"
        )

    vae.load_state_dict(vae_state, strict=False)
    vae.to(device)
    vae.eval()

    log.info(f"  VAE loaded (epoch {vae_ckpt.get('epoch', 'unknown')})")

    # =========================================================================
    # Step 4: Construct MotifNet and load state dict
    # =========================================================================
    log.info("Loading MotifNet...")

    motifnet = MotifNet(
        n_tfs=n_tfs,
        bottleneck_size=3072,  # Legacy default (Enformer output dim)
        emb_len=5,  # Legacy default (target_length=5)
    )

    motifnet_ckpt = torch.load(motifnet_path, map_location=device)
    motifnet_state = _map_legacy_motifnet_state_dict(motifnet_ckpt["model_state_dict"])
    motifnet.load_state_dict(motifnet_state)
    motifnet.to(device)
    motifnet.eval()

    log.info(f"  MotifNet loaded (epoch {motifnet_ckpt.get('epoch', 'unknown')})")

    # =========================================================================
    # Step 5: Construct Enformer and load state dict
    # =========================================================================
    log.info("Loading Enformer...")

    from enformer_pytorch import Enformer

    enformer = Enformer.from_pretrained(
        "EleutherAI/enformer-official-rough",
        target_length=-1,
    )

    enf_ckpt = torch.load(enformer_path, map_location=device)
    enformer.load_state_dict(enf_ckpt["model_state_dict"])
    enformer.to(device)
    enformer.eval()

    log.info(f"  Enformer loaded (epoch {enf_ckpt.get('epoch', 'unknown')})")

    # =========================================================================
    # Step 6: Load or reconstruct E1 matrix
    # =========================================================================
    if e1_path is not None:
        # Load from pickle file
        e1_path = Path(e1_path)
        if not e1_path.exists():
            raise FileNotFoundError(f"E1 file not found: {e1_path}")

        log.info(f"Loading E1 from {e1_path}...")
        e1_df = pd.read_pickle(e1_path)
        # E1.pkl is (n_tfs, n_regions), we need (n_regions, n_tfs)
        adj_E1 = torch.tensor(e1_df.values.T, dtype=torch.float32, device=device)
        log.info(f"  E1 shape: {adj_E1.shape}")
    else:
        # Reconstruct from model weights
        from tqdm import tqdm

        from .._genome import get_genome
        from ._dataloaders import build_sequence_dataloader

        log.info("Reconstructing E1 matrix from model weights...")
        log.info("  (This may take a while - running all regions through Enformer)")

        # Check genome is registered
        try:
            genome = get_genome()
        except RuntimeError as err:
            raise ValueError(
                "No E1 path provided and no genome registered. Either:\n"
                "  1. Provide e1_path to load pre-computed E1 matrix, OR\n"
                "  2. Register genome with ds.register_genome('/path/to/genome.fa')"
            ) from err

        # Build sequence dataloader (no augmentation for inference)
        seq_loader = build_sequence_dataloader(
            regions=region_names,
            genome=genome,
            batch_size=1000,
            shuffle=False,
            shift_augs=(0, 0),  # No augmentation
            rc_aug=False,
            context_length=640,  # Legacy default
            num_workers=4,
        )

        # Compute E1
        adj_E1 = torch.zeros(n_regions, n_tfs, device=device)
        enformer.eval()
        motifnet.eval()

        with torch.no_grad():
            for sequences, seq_idx in tqdm(seq_loader, desc="Computing E1"):
                sequences = sequences.to(device)
                # Get embeddings from Enformer
                emb = enformer(sequences, return_only_embeddings=True)
                # Flatten: (batch, emb_len, bottleneck) -> (batch, bottleneck * emb_len)
                emb = emb.reshape(emb.shape[0], -1).float()
                # Get TF predictions from MotifNet
                tf_pred = motifnet(emb)
                adj_E1[seq_idx] = tf_pred

        log.info(f"  E1 shape: {adj_E1.shape}")

    # =========================================================================
    # Step 7: Package into DeepSCENICModel
    # =========================================================================
    config = ModelConfig(
        n_hidden=128,
        bottleneck_size=3072,
        emb_len=5,
    )

    model = DeepSCENICModel(
        vae=vae,
        motifnet=motifnet,
        sequence_model=enformer,
        adj_E1=adj_E1,
        config=config,
        tf_names=tf_names,
        gene_names=gene_names,
        region_names=region_names,
    )

    log.info("Legacy model loaded successfully!")
    return model


def _map_legacy_motifnet_state_dict(state_dict: dict) -> dict:
    """Map legacy MotifNet state dict keys to new format.

    Legacy format:
    - ctx_head_layer.weight → ctx_conv.weight
    - ctx_lin.weight → ctx_linear.weight
    - ctx_lin.bias → ctx_linear.bias
    """
    key_mapping = {
        "ctx_head_layer.weight": "ctx_conv.weight",
        "ctx_lin.weight": "ctx_linear.weight",
        "ctx_lin.bias": "ctx_linear.bias",
    }
    return {key_mapping.get(k, k): v for k, v in state_dict.items()}


def _map_legacy_vae_state_dict(state_dict: dict) -> dict:
    """Map legacy VAE state dict keys to new format.

    Legacy format used different naming conventions:
    - inference_rna.inference_qzyx.* → encoder.mlp.* / encoder.gaussian.*
    - generative_rna.generative_pxz.* → decoder_rna.mlp.*
    - generative_atac.generative_pxz.* → decoder_atac.mlp.*
    - PPInet.* → ppi.*

    Returns new state dict with mapped keys.
    """
    key_mapping = {
        # InferenceNet (encoder)
        "inference_rna.inference_qzyx.0.weight": "encoder.mlp.0.weight",
        "inference_rna.inference_qzyx.2.weight": "encoder.mlp.2.weight",
        "inference_rna.inference_qzyx.4.mu.weight": "encoder.gaussian.mu_layer.weight",
        "inference_rna.inference_qzyx.4.var.weight": "encoder.gaussian.logvar_layer.weight",
        "inference_rna.inference_qzyx.4.var.bias": "encoder.gaussian.logvar_layer.bias",
        # GenerativeNet RNA (decoder_rna)
        "generative_rna.generative_pxz.0.weight": "decoder_rna.mlp.0.weight",
        "generative_rna.generative_pxz.2.weight": "decoder_rna.mlp.2.weight",
        "generative_rna.generative_pxz.4.weight": "decoder_rna.mlp.4.weight",
        # GenerativeNet ATAC (decoder_atac)
        "generative_atac.generative_pxz.0.weight": "decoder_atac.mlp.0.weight",
        "generative_atac.generative_pxz.0.bias": "decoder_atac.mlp.0.bias",
        "generative_atac.generative_pxz.2.weight": "decoder_atac.mlp.2.weight",
        "generative_atac.generative_pxz.2.bias": "decoder_atac.mlp.2.bias",
        "generative_atac.generative_pxz.4.weight": "decoder_atac.mlp.4.weight",
        "generative_atac.generative_pxz.4.bias": "decoder_atac.mlp.4.bias",
    }

    # Build new state dict with mapped keys
    new_state_dict = {}

    for old_key, value in state_dict.items():
        # Check for exact mapping
        if old_key in key_mapping:
            new_key = key_mapping[old_key]
            new_state_dict[new_key] = value
        # Handle PPInet.* → ppi.* prefix change
        elif old_key.startswith("PPInet."):
            new_key = "ppi." + old_key[7:]  # Remove "PPInet." and add "ppi."
            new_state_dict[new_key] = value
        # Keep other keys as-is (e.g., adj_E2, batch_layer_*)
        else:
            new_state_dict[old_key] = value

    return new_state_dict
