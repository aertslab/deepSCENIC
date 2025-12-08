"""Trained model wrapper for deepSCENIC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from ._training_state import TrainingConfig

if TYPE_CHECKING:
    from ..models import DeepSCENICVAE, MotifNet


@dataclass
class DeepSCENICModel:
    """Trained deepSCENIC model container.

    This is the object returned by `ds.tl.train()` and loaded by `ds.tl.load_model()`.
    Contains all trained components and metadata for self-contained inference.

    Attributes
    ----------
    vae : DeepSCENICVAE
        Trained VAE model
    tf2rnet : MotifNet
        Trained TF2rNet context head
    enformer : nn.Module
        Fine-tuned Enformer model
    adj_E1 : Tensor
        Cached E1 matrix (n_regions, n_tfs)
    config : TrainingConfig
        Training configuration used
    tf_names : list[str]
        TF names in order
    gene_names : list[str]
        Gene names in order
    region_names : list[str]
        Region names in order

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

    def save(self, path: str | Path) -> None:
        """
        Save model to file.

        Includes all components (VAE, TF2rNet, Enformer) and metadata.
        File size will be ~1GB+ due to Enformer weights.

        Parameters
        ----------
        path
            Output file path (.pt)
        """
        torch.save(
            {
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
                "r2g_indices": self.vae.r2g_indices,
                "r2g_distances": self.vae.r2g_distances,
                # PPI buffers (if present)
                "use_ppi": self.vae.use_ppi,
                "ppi_edge_index": getattr(self.vae, "ppi_edge_index", None),
                "ppi_genes_idx": getattr(self.vae, "ppi_genes_idx", None),
                "ppi_tfs_idx_keys": getattr(self.vae, "ppi_tfs_idx_keys", None),
                "ppi_tfs_idx_values": getattr(self.vae, "ppi_tfs_idx_values", None),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> DeepSCENICModel:
        """
        Load model from file.

        Parameters
        ----------
        path
            Model file path (.pt)
        device
            Device to load model to

        Returns
        -------
        DeepSCENICModel
            Loaded model ready for inference
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

        # Reconstruct Enformer
        from enformer_pytorch import Enformer

        enformer = Enformer.from_pretrained(
            "EleutherAI/enformer-official-rough",
            target_length=-1,
        )
        enformer.load_state_dict(data["enformer_state_dict"])
        enformer.to(device)
        enformer.eval()

        return cls(
            vae=vae,
            tf2rnet=tf2rnet,
            enformer=enformer,
            adj_E1=data["adj_E1"].to(device),
            config=config,
            tf_names=data["tf_names"],
            gene_names=data["gene_names"],
            region_names=data["region_names"],
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


def load_model(path: str | Path, device: str = "cpu") -> DeepSCENICModel:
    """
    Load a trained deepSCENIC model.

    Parameters
    ----------
    path
        Path to saved model file (.pt)
    device
        Device to load model to

    Returns
    -------
    DeepSCENICModel
        Loaded model ready for inference
    """
    return DeepSCENICModel.load(path, device=device)
