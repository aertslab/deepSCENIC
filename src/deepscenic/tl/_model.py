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

from ._training_state import PretrainConfig, TrainingConfig, TrainingHistory

log = logging.getLogger("deepscenic.tl")

if TYPE_CHECKING:
    from ..models import DeepSCENICVAE, MotifNet


@dataclass
class DeepSCENICModel:
    """Trained deepSCENIC model container.

    This is the object returned by `ds.tl.train()` and loaded by `ds.tl.load_model()`.
    Contains all trained components and metadata for self-contained inference.

    Attributes
    ----------
    vae
        Trained VAE model.
    tf2rnet
        Trained TF2rNet context head.
    enformer
        Sequence embedding model (Enformer or custom nn.Module).
    adj_E1
        Cached E1 matrix (n_regions, n_tfs).
    config
        Training configuration used.
    tf_names
        TF names in order.
    gene_names
        Gene names in order.
    region_names
        Region names in order.
    history
        Training history with loss metrics per epoch. Available when model was
        just trained or loaded from a file that includes history.

    Examples
    --------
    >>> model = ds.tl.train(mdata, epochs=100)
    >>> model.save("model.pt")
    >>> loaded = ds.tl.load_model("model.pt")
    >>> embeddings = ds.tl.to_latent(loaded, mdata)
    """

    vae: DeepSCENICVAE
    tf2rnet: MotifNet
    enformer: nn.Module
    adj_E1: torch.Tensor
    config: TrainingConfig
    tf_names: list[str]
    gene_names: list[str]
    region_names: list[str]
    history: TrainingHistory | None = None

    def save(self, path: str | Path) -> None:
        """
        Save model to file.

        Includes all components (VAE, TF2rNet, sequence model), metadata, and
        training history if available. File size will be ~1GB+ if using Enformer.

        Parameters
        ----------
        path
            Output file path (.pt)
        """
        # Detect if sequence model is Enformer or custom
        is_custom_sequence_model = type(self.enformer).__name__ != "Enformer"

        data = {
            "vae_state_dict": self.vae.state_dict(),
            "tf2rnet_state_dict": self.tf2rnet.state_dict(),
            "enformer_state_dict": self.enformer.state_dict(),
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
            must match what was used during training.

        Returns
        -------
        DeepSCENICModel
            Loaded model ready for inference

        Notes
        -----
        If the saved model used a custom sequence model (not Enformer), you must
        provide a matching model instance via the ``sequence_model`` parameter.
        The weights will be loaded from the checkpoint.

        Examples
        --------
        Loading a model trained with default Enformer:

        >>> model = DeepSCENICModel.load("model.pt")
        Loading a model trained with custom sequence model:

        >>> custom = MySequenceModel()  # Same architecture as training
        >>> model = DeepSCENICModel.load("model.pt", sequence_model=custom)
        """
        from ..models import DeepSCENICVAE, MotifNet

        data = torch.load(path, map_location=device)

        # Reconstruct config
        config = TrainingConfig.from_dict(data["config"])

        # Reconstruct VAE
        vae = DeepSCENICVAE(
            n_tfs=data["n_tfs"],
            n_genes=data["n_genes"],
            n_regions=data["n_regions"],
            r2g_indices=data["r2g_indices"],
            r2g_distances=data["r2g_distances"],
            tf_indices=data["tf_indices"],
            gene_indices=data["gene_indices"],
            region_indices=data["region_indices"],
            ppi_edge_index=data.get("ppi_edge_index"),
            ppi_genes_idx=data.get("ppi_genes_idx"),
            ppi_tfs_idx_keys=data.get("ppi_tfs_idx_keys"),
            ppi_tfs_idx_values=data.get("ppi_tfs_idx_values"),
            n_hidden=data["n_hidden"],
            use_ppi=data.get("use_ppi", False),
            n_batches=data.get("n_batches", 0),
        )
        vae.load_state_dict(data["vae_state_dict"])
        vae.to(device)
        vae.eval()

        # Reconstruct MotifNet
        tf2rnet = MotifNet(
            n_tfs=data["n_tfs"],
            bottleneck_size=config.bottleneck_size,
            emb_len=config.emb_len,
        )
        tf2rnet.load_state_dict(data["tf2rnet_state_dict"])
        tf2rnet.to(device)
        tf2rnet.eval()

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
            # Default: load Enformer
            from enformer_pytorch import Enformer

            seq_model = Enformer.from_pretrained(
                "EleutherAI/enformer-official-rough",
                target_length=-1,
            )

        seq_model.load_state_dict(data["enformer_state_dict"])
        seq_model.to(device)
        seq_model.eval()

        log.info(f"Loaded model from {path}: {data['n_genes']} genes, {data['n_tfs']} TFs, {data['n_regions']} regions")

        # Restore history if present
        history = None
        if "history" in data:
            history = TrainingHistory.from_dict(data["history"])

        return cls(
            vae=vae,
            tf2rnet=tf2rnet,
            enformer=seq_model,
            adj_E1=data["adj_E1"].to(device),
            config=config,
            tf_names=data["tf_names"],
            gene_names=data["gene_names"],
            region_names=data["region_names"],
            history=history,
        )

    def to(self, device: str | torch.device) -> DeepSCENICModel:
        """Move model to device."""
        self.vae.to(device)
        self.tf2rnet.to(device)
        self.enformer.to(device)
        self.adj_E1 = self.adj_E1.to(device)
        return self

    def eval(self) -> DeepSCENICModel:
        """Set all components to eval mode."""
        self.vae.eval()
        self.tf2rnet.eval()
        self.enformer.eval()
        return self

    def train(self) -> DeepSCENICModel:
        """Set all components to train mode."""
        self.vae.train()
        self.tf2rnet.train()
        self.enformer.train()
        return self


@dataclass
class PretrainedModel:
    """Container for pretrained sequence models.

    This is the object returned by `ds.tl.pretrain()` containing pretrained
    TF2rNet and sequence models with a cached E1 matrix. It can be passed
    to `ds.tl.train()` to initialize the sequence models.

    Attributes
    ----------
    tf2rnet
        Pretrained MotifNet model.
    enformer
        Pretrained sequence embedding model (Enformer or custom nn.Module).
    adj_E1
        Cached E1 matrix (n_regions, n_tfs).
    config
        Pretraining configuration used.
    tf_names
        TF names in order.
    region_names
        Region names in order.

    Examples
    --------
    >>> pretrained = ds.tl.pretrain(mdata, epochs=100)
    >>> pretrained.save("pretrained.pt")
    >>> pretrained = ds.tl.load_pretrained("pretrained.pt")
    >>> model = ds.tl.train(mdata, pretrained_model=pretrained)
    """

    tf2rnet: MotifNet
    enformer: nn.Module
    adj_E1: torch.Tensor
    config: PretrainConfig
    tf_names: list[str]
    region_names: list[str]

    def save(self, path: str | Path) -> None:
        """Save pretrained model to file.

        Parameters
        ----------
        path
            Output file path (.pt).
        """
        is_custom_sequence_model = type(self.enformer).__name__ != "Enformer"

        torch.save(
            {
                "tf2rnet_state_dict": self.tf2rnet.state_dict(),
                "enformer_state_dict": self.enformer.state_dict(),
                "adj_E1": self.adj_E1,
                "config": self.config.to_dict(),
                "tf_names": self.tf_names,
                "region_names": self.region_names,
                "n_tfs": self.tf2rnet.n_tfs,
                "bottleneck_size": self.config.bottleneck_size,
                "emb_len": self.config.emb_len,
                "is_custom_sequence_model": is_custom_sequence_model,
            },
            path,
        )

        log.info(f"Saved pretrained model to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        sequence_model: nn.Module | None = None,
    ) -> PretrainedModel:
        """Load pretrained model from file.

        Parameters
        ----------
        path
            Model file path (.pt).
        device
            Device to load model to.
        sequence_model
            Custom sequence model instance. Required when loading a model that
            was pretrained with a custom sequence model.

        Returns
        -------
        PretrainedModel
            Loaded pretrained model.
        """
        from ..models import MotifNet

        data = torch.load(path, map_location=device)
        config = PretrainConfig.from_dict(data["config"])

        # Reconstruct MotifNet
        tf2rnet = MotifNet(
            n_tfs=data["n_tfs"],
            bottleneck_size=data["bottleneck_size"],
            emb_len=data["emb_len"],
        )
        tf2rnet.load_state_dict(data["tf2rnet_state_dict"])
        tf2rnet.to(device)
        tf2rnet.eval()

        # Reconstruct sequence model
        is_custom = data.get("is_custom_sequence_model", False)
        if is_custom:
            if sequence_model is None:
                raise ValueError(
                    "This model was pretrained with a custom sequence model. "
                    "You must provide the model instance via sequence_model parameter."
                )
            seq_model = sequence_model
        else:
            # Default: load Enformer
            from enformer_pytorch import Enformer

            seq_model = Enformer.from_pretrained(
                "EleutherAI/enformer-official-rough",
                target_length=-1,
            )

        seq_model.load_state_dict(data["enformer_state_dict"])
        seq_model.to(device)
        seq_model.eval()

        log.info(f"Loaded pretrained model from {path}")

        return cls(
            tf2rnet=tf2rnet,
            enformer=seq_model,
            adj_E1=data["adj_E1"].to(device),
            config=config,
            tf_names=data["tf_names"],
            region_names=data["region_names"],
        )

    def to(self, device: str | torch.device) -> PretrainedModel:
        """Move model to device."""
        self.tf2rnet.to(device)
        self.enformer.to(device)
        self.adj_E1 = self.adj_E1.to(device)
        return self

    def eval(self) -> PretrainedModel:
        """Set all components to eval mode."""
        self.tf2rnet.eval()
        self.enformer.eval()
        return self

    def train_mode(self) -> PretrainedModel:
        """Set all components to train mode."""
        self.tf2rnet.train()
        self.enformer.train()
        return self


def load_pretrained(
    path: str | Path,
    device: str = "cpu",
    sequence_model: nn.Module | None = None,
) -> PretrainedModel:
    """Load a pretrained sequence model.

    Parameters
    ----------
    path
        Path to saved pretrained model file (.pt).
    device
        Device to load model to.
    sequence_model
        Custom sequence model instance. Required when loading a model that
        was pretrained with a custom sequence model.

    Returns
    -------
    PretrainedModel
        Loaded pretrained model ready to use with ds.tl.train().

    Examples
    --------
    >>> pretrained = ds.tl.load_pretrained("pretrained.pt")
    >>> model = ds.tl.train(mdata, pretrained_model=pretrained)
    """
    return PretrainedModel.load(path, device=device, sequence_model=sequence_model)


def load_model(
    path: str | Path,
    device: str = "cpu",
    sequence_model: nn.Module | None = None,
) -> DeepSCENICModel:
    """
    Load a trained deepSCENIC model.

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
        Loaded model ready for inference
    """
    return DeepSCENICModel.load(path, device=device, sequence_model=sequence_model)


def load_legacy_model(
    vae_path: str | Path,
    tf2rnet_path: str | Path,
    enformer_path: str | Path,
    *,
    # Data paths for architecture reconstruction
    rna_path: str | Path,
    atac_path: str | Path,
    r2g_path: str | Path,
    tf_list_path: str | Path,
    # Pre-computed E1 matrix (required)
    e1_path: str | Path,
    # Optional: PPI data
    ppi_data: pd.DataFrame | None = None,
    ppi_confidence: float = 0.5,
    species: str = "human",
    # Loading options
    device: str = "cpu",
) -> DeepSCENICModel:
    """
    Load legacy deepSCENIC model (3 separate .pth files) into unified format.

    Legacy models were saved as three separate checkpoint files:
    - best_model.pth: VAE (InferenceNet + Decoders + adj_E2 + PPIgnn)
    - best_model_tf2r.pth: MotifNet context head
    - best_model_tf2r_encoder.pth: Enformer (finetuned)

    This function reconstructs the model architecture from the data files
    and loads the state dicts.

    Parameters
    ----------
    vae_path
        Path to VAE checkpoint (e.g., best_model.pth)
    tf2rnet_path
        Path to MotifNet checkpoint (e.g., best_model_tf2r.pth)
    enformer_path
        Path to Enformer checkpoint (e.g., best_model_tf2r_encoder.pth)
    rna_path
        Path to RNA h5ad file (e.g., raw_exprMat_train.h5ad)
    atac_path
        Path to ATAC h5ad file (e.g., fragment_matrix_train.h5ad)
    r2g_path
        Path to r2g penalty matrix (e.g., r2gpenalty_train.npz)
    tf_list_path
        Path to TF list file (e.g., TFs.txt)
    e1_path
        Path to pre-computed E1 matrix (E1.pkl). This is the TF-to-region
        binding matrix extracted during legacy training.
    ppi_data
        Optional PPI interaction DataFrame. If None, PPI is disabled.
    ppi_confidence
        Confidence threshold for PPI filtering.
    species
        Species for gene name formatting ('human' or 'mouse').
    device
        Device to load model to.

    Returns
    -------
    DeepSCENICModel
        Loaded model ready for inference.

    Examples
    --------
    >>> model = ds.tl.load_legacy_model(
    ...     vae_path="results/best_model.pth",
    ...     tf2rnet_path="results/best_model_tf2r.pth",
    ...     enformer_path="results/best_model_tf2r_encoder.pth",
    ...     rna_path="data/raw_exprMat_train.h5ad",
    ...     atac_path="data/fragment_matrix_train.h5ad",
    ...     r2g_path="data/r2gpenalty_train.npz",
    ...     tf_list_path="data/TFs.txt",
    ...     e1_path="results/E1.pkl",
    ... )

    Notes
    -----
    Legacy checkpoint format:
        {'epoch': int, 'model_state_dict': OrderedDict, 'optimizer_state_dict': ...}

    The state dict keys are identical between legacy and new code:
        - VAE: 'adj_E2', 'inference_rna.*', 'generative_rna.*', 'generative_atac.*', 'PPInet.*'
        - MotifNet: 'ctx_head_layer.weight', 'ctx_lin.weight', 'ctx_lin.bias'
        - Enformer: Full Enformer state dict

    See Also
    --------
    load_model : Load new format model
    extract_grn : Extract GRN matrices from loaded model
    """
    import scanpy as sc
    from scipy.sparse import load_npz

    from ..models import DeepSCENICVAE, MotifNet

    # Convert paths to Path objects
    vae_path = Path(vae_path)
    tf2rnet_path = Path(tf2rnet_path)
    enformer_path = Path(enformer_path)
    rna_path = Path(rna_path)
    atac_path = Path(atac_path)
    r2g_path = Path(r2g_path)
    tf_list_path = Path(tf_list_path)

    # =========================================================================
    # Step 1: Load data to extract dimensions and indices
    # =========================================================================
    log.info("Loading data files...")

    # Load RNA data
    adata_rna = sc.read_h5ad(rna_path)
    gene_names = adata_rna.var_names.tolist()
    n_genes = len(gene_names)

    # Load ATAC data
    adata_atac = sc.read_h5ad(atac_path)
    region_names = adata_atac.var_names.tolist()
    n_regions = len(region_names)

    # Load TF list
    with open(tf_list_path) as f:
        tf_list = [line.strip() for line in f if line.strip()]

    # Get TF indices (TFs that exist in RNA data)
    tf_mask = adata_rna.var_names.isin(tf_list)
    tf_indices = torch.tensor(np.where(tf_mask)[0], dtype=torch.long)
    tf_names = adata_rna.var_names[tf_mask].tolist()
    n_tfs = len(tf_names)

    log.info(f"  RNA: {n_genes} genes, ATAC: {n_regions} regions, TFs: {n_tfs}")

    # Gene indices (which genes to reconstruct) - use all genes for legacy models
    gene_indices = torch.arange(n_genes, dtype=torch.long)
    # Region indices (which regions to reconstruct) - use all regions for legacy models
    region_indices = torch.arange(n_regions, dtype=torch.long)

    # Load r2g sparse matrix
    r2g_sparse = load_npz(r2g_path)
    r2g_coo = r2g_sparse.tocoo()
    r2g_indices = torch.tensor(np.vstack([r2g_coo.row, r2g_coo.col]), dtype=torch.long)
    r2g_distances = torch.tensor(r2g_coo.data, dtype=torch.float32)

    log.info(f"  R2G links: {r2g_indices.shape[1]}")

    # =========================================================================
    # Step 2: Build PPI network (if data provided)
    # =========================================================================
    use_ppi = ppi_data is not None
    ppi_edge_index = None
    ppi_genes_idx = None
    ppi_tfs_idx_keys = None
    ppi_tfs_idx_values = None

    if use_ppi:
        log.info("Building PPI network...")
        ppi_edge_index, ppi_genes_idx, ppi_tfs_idx_keys, ppi_tfs_idx_values = _build_ppi_indices(
            adata_rna,
            ppi_data,
            tf_names,
            ppi_confidence,
            species,
        )
        log.info(f"  PPI: {ppi_edge_index.shape[1]} edges, {len(ppi_genes_idx)} genes")

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
        ppi_edge_index=ppi_edge_index,
        ppi_genes_idx=ppi_genes_idx,
        ppi_tfs_idx_keys=ppi_tfs_idx_keys,
        ppi_tfs_idx_values=ppi_tfs_idx_values,
        n_hidden=128,  # Legacy default
        use_ppi=use_ppi,
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
            ppi_edge_index=ppi_edge_index,
            ppi_genes_idx=ppi_genes_idx,
            ppi_tfs_idx_keys=ppi_tfs_idx_keys,
            ppi_tfs_idx_values=ppi_tfs_idx_values,
            n_hidden=128,
            use_ppi=use_ppi,
            binary_atac=False,
            n_batches=n_batches,
        )

    # Map legacy state dict keys to new format
    vae_state = _map_legacy_vae_state_dict(vae_state)

    # Handle potential key mismatches for PPI
    if not use_ppi:
        # Remove PPI keys from state dict if we're not using PPI
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

    tf2rnet = MotifNet(
        n_tfs=n_tfs,
        bottleneck_size=3072,  # Legacy default (Enformer output dim)
        emb_len=5,  # Legacy default (target_length=5)
    )

    tf2r_ckpt = torch.load(tf2rnet_path, map_location=device)
    tf2r_state = _map_legacy_motifnet_state_dict(tf2r_ckpt["model_state_dict"])
    tf2rnet.load_state_dict(tf2r_state)
    tf2rnet.to(device)
    tf2rnet.eval()

    log.info(f"  MotifNet loaded (epoch {tf2r_ckpt.get('epoch', 'unknown')})")

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
    # Step 6: Load E1 matrix from pickle
    # =========================================================================
    e1_path = Path(e1_path)
    if not e1_path.exists():
        raise FileNotFoundError(f"E1 file not found: {e1_path}")

    log.info(f"Loading E1 from {e1_path}...")
    e1_df = pd.read_pickle(e1_path)
    # E1.pkl is (n_tfs, n_regions), we need (n_regions, n_tfs)
    adj_E1 = torch.tensor(e1_df.values.T, dtype=torch.float32, device=device)
    log.info(f"  E1 shape: {adj_E1.shape}")

    # =========================================================================
    # Step 7: Package into DeepSCENICModel
    # =========================================================================
    config = TrainingConfig(
        n_hidden=128,
        use_ppi=use_ppi,
        bottleneck_size=3072,
        emb_len=5,
    )

    model = DeepSCENICModel(
        vae=vae,
        tf2rnet=tf2rnet,
        enformer=enformer,
        adj_E1=adj_E1,
        config=config,
        tf_names=tf_names,
        gene_names=gene_names,
        region_names=region_names,
    )

    log.info("Legacy model loaded successfully!")
    return model


def _build_ppi_indices(
    adata_rna,
    ppi_data: pd.DataFrame,
    tf_names: list[str],
    confidence_threshold: float,
    species: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build PPI indices from interaction data."""
    import networkx as nx

    ppi = ppi_data.copy()

    # Standardize column names
    if "Source" not in ppi.columns:
        # Try common column names
        for src_col in ["protein1", "source", "gene1", "proteinA"]:
            if src_col in ppi.columns:
                ppi = ppi.rename(columns={src_col: "Source"})
                break
    if "Target" not in ppi.columns:
        for tgt_col in ["protein2", "target", "gene2", "proteinB"]:
            if tgt_col in ppi.columns:
                ppi = ppi.rename(columns={tgt_col: "Target"})
                break

    # Filter by confidence if column exists
    for conf_col in ["Conn", "confidence", "combined_score", "score"]:
        if conf_col in ppi.columns:
            ppi = ppi[ppi[conf_col] >= confidence_threshold]
            break

    # Format gene names based on species
    if species == "mouse":
        ppi["Source"] = ppi["Source"].apply(lambda x: x[0].upper() + x[1:].lower() if len(x) > 1 else x.upper())
        ppi["Target"] = ppi["Target"].apply(lambda x: x[0].upper() + x[1:].lower() if len(x) > 1 else x.upper())

    # Filter to genes in data
    all_genes = set(adata_rna.var_names)
    ppi = ppi[ppi["Source"] != ppi["Target"]]  # Remove self-loops
    ppi = ppi[ppi["Source"].isin(all_genes)]
    ppi = ppi[ppi["Target"].isin(all_genes)]

    if len(ppi) == 0:
        raise ValueError("No PPI edges remain after filtering")

    # Build graph
    G = nx.from_pandas_edgelist(ppi, "Source", "Target")
    G.add_nodes_from(all_genes)  # Add isolated nodes

    node_list = list(G.nodes())
    node_to_idx = {gene: i for i, gene in enumerate(node_list)}

    # Convert to edge index
    G_directed = G.to_directed()
    n_edges = G_directed.number_of_edges()
    edge_index = torch.empty((2, n_edges), dtype=torch.long)
    for i, (src, dst) in enumerate(G_directed.edges()):
        edge_index[0, i] = node_to_idx[src]
        edge_index[1, i] = node_to_idx[dst]

    # Get gene indices in RNA data
    ppi_genes_idx = torch.tensor(
        [adata_rna.var_names.get_loc(gene) for gene in node_list if gene in adata_rna.var_names],
        dtype=torch.long,
    )

    # Map TFs to PPI positions
    tf_to_order = {tf: i for i, tf in enumerate(tf_names)}
    ppi_tfs_idx_keys_list: list[int] = []
    ppi_tfs_idx_values_list: list[int] = []

    for ppi_idx, gene in enumerate(node_list):
        if gene in tf_to_order:
            ppi_tfs_idx_keys_list.append(ppi_idx)
            ppi_tfs_idx_values_list.append(tf_to_order[gene])

    ppi_tfs_idx_keys = torch.tensor(ppi_tfs_idx_keys_list, dtype=torch.long)
    ppi_tfs_idx_values = torch.tensor(ppi_tfs_idx_values_list, dtype=torch.long)

    return edge_index, ppi_genes_idx, ppi_tfs_idx_keys, ppi_tfs_idx_values


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
