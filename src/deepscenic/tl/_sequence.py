"""Sequence interpretation tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.sparse import issparse

from .._genome import Genome, get_genome
from ._train import _get_sequence_embeddings

if TYPE_CHECKING:
    from mudata import MuData

    from ._model import DeepSCENICModel


@dataclass(frozen=True)
class ISMResult:
    """Result of class-aware in-silico mutagenesis.

    Attributes
    ----------
    scores
        Mutation effects with shape ``(sequence_length, 4)`` in A, C, G, T
        order. Values are mutant minus reference class-specific activity.
    sequence
        Reference DNA sequence used by the model.
    region
        Input genomic region.
    sequence_region
        Genomic interval after centering to the model context length.
    baseline_score
        Predicted class-specific activity of the reference sequence.
    class_key, class_label
        Cell annotation column and selected class.
    """

    scores: np.ndarray
    sequence: str
    region: str
    sequence_region: str
    baseline_score: float
    class_key: str
    class_label: str


def _parse_region(region: str, context_length: int) -> tuple[str, int, int]:
    try:
        chrom, coordinates = region.rsplit(":", 1)
        start_text, end_text = coordinates.replace(",", "").split("-", 1)
        start, end = int(start_text), int(end_text)
    except (ValueError, AttributeError) as e:
        raise ValueError("region must have the form 'chrom:start-end'") from e
    if start < 0 or end <= start:
        raise ValueError("region coordinates must satisfy 0 <= start < end")

    midpoint = (start + end) // 2
    sequence_start = midpoint - context_length // 2
    return chrom, sequence_start, sequence_start + context_length


def _mean_tf_activity(
    model: DeepSCENICModel,
    mdata: MuData,
    mask: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    key_prefix: str,
) -> torch.Tensor:
    latent_key = f"{key_prefix}z_tf"
    if latent_key in mdata.obsm:
        z_tf = np.asarray(mdata.obsm[latent_key])
        if z_tf.shape != (mdata.n_obs, len(model.tf_names)):
            raise ValueError(
                f"mdata.obsm[{latent_key!r}] has shape {z_tf.shape}; expected "
                f"({mdata.n_obs}, {len(model.tf_names)})"
            )
        return torch.as_tensor(z_tf[mask].mean(axis=0), dtype=torch.float32, device=device)

    if "rna" not in mdata.mod:
        raise KeyError("mdata must contain an 'rna' modality when TF latent activity is absent")
    if mdata.mod["rna"].n_obs != mdata.n_obs:
        raise ValueError("The RNA modality must contain the same cells, in the same order, as mdata.obs")

    cell_indices = np.flatnonzero(mask)
    rna = mdata.mod["rna"].X
    total = torch.zeros(len(model.tf_names), dtype=torch.float32, device=device)
    count = 0
    model.vae.eval()
    with torch.no_grad():
        for offset in range(0, len(cell_indices), batch_size):
            values = rna[cell_indices[offset : offset + batch_size]]
            if issparse(values):
                values = values.toarray()
            x_rna = torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)
            x_rna_tfs = x_rna[:, model.vae.tf_indices]
            z_tf, _, _ = model.vae.encoder(x_rna_tfs, use_mean=True)
            total += z_tf.sum(dim=0)
            count += z_tf.shape[0]
    return total / count


def in_silico_mutagenesis(
    model: DeepSCENICModel,
    mdata: MuData,
    region: str,
    class_key: str,
    class_label: str,
    *,
    genome: Genome | None = None,
    context_length: int | None = None,
    batch_size: int = 128,
    cell_batch_size: int = 256,
    device: str | torch.device | None = None,
    key_prefix: str = "X_deepscenic_",
) -> ISMResult:
    """Measure sequence-mutation effects for a selected cell class.

    Every nucleotide at every position is substituted with A, C, G, and T.
    For each sequence, the TF-to-region prediction is weighted by mean latent
    TF activity in the selected cells and summed across TFs. Returned scores
    are ``mutant_score - reference_score``; reference nucleotides are set to
    zero. If ``mdata.obsm['X_deepscenic_z_tf']`` exists it is reused,
    otherwise TF activity is computed from the RNA modality.

    Parameters
    ----------
    model
        Trained deepSCENIC model.
    mdata
        MuData containing cell annotations and either latent TF activity or
        an RNA modality.
    region
        Genomic interval in ``chrom:start-end`` format. It is centered and
        resized to the model's sequence context.
    class_key
        Column in ``mdata.obs`` containing cell annotations.
    class_label
        Cell class whose mean TF activity weights the sequence predictions.
    genome
        Genome to use. By default, use the globally registered genome.
    context_length
        Sequence length. Defaults to ``model.config.seq_len``.
    batch_size
        Mutant sequences evaluated per batch.
    cell_batch_size
        Cells evaluated per batch when latent TF activity must be computed.
    device
        PyTorch device. Defaults to the VAE's current device.
    key_prefix
        Prefix used to find the cached ``z_tf`` embedding in ``mdata.obsm``.

    Returns
    -------
    ISMResult
        Scores and the sequence/region metadata needed for visualization.

    Examples
    --------
    >>> result = ds.tl.in_silico_mutagenesis(
    ...     model, mdata, "chr6:396072-396572", "lineState", "MEL"
    ... )
    >>> ds.pl.ism_heatmap(result.scores, title="IRF4 enhancer — MEL")
    """
    if batch_size <= 0 or cell_batch_size <= 0:
        raise ValueError("batch_size and cell_batch_size must be positive")
    if class_key not in mdata.obs:
        raise KeyError(f"mdata.obs does not contain {class_key!r}")
    mask = np.asarray(mdata.obs[class_key] == class_label)
    if not mask.any():
        raise ValueError(f"No cells have {class_key} == {class_label!r}")

    if context_length is None:
        context_length = int(model.config.seq_len)
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    chrom, start, end = _parse_region(region, context_length)
    genome = get_genome() if genome is None else genome
    sequence = genome.fetch(chrom, start, end)
    reference = genome.fetch_onehot(chrom, start, end).T.float()

    if device is None:
        device = next(model.vae.parameters()).device
    device = torch.device(device)
    model.vae.to(device)
    model.sequence_model.to(device).eval()
    model.motifnet.to(device).eval()
    reference = reference.to(device)
    class_activity = _mean_tf_activity(
        model,
        mdata,
        mask,
        batch_size=cell_batch_size,
        device=device,
        key_prefix=key_prefix,
    )

    def predict(sequences: torch.Tensor) -> torch.Tensor:
        embeddings = _get_sequence_embeddings(
            model.sequence_model,
            sequences,
            model.config.bottleneck_size,
            model.config.emb_len,
        )
        tf2r = model.motifnet(embeddings)
        if tf2r.ndim != 2 or tf2r.shape[1] != len(model.tf_names):
            raise ValueError(
                "motifnet must return shape (batch, n_tfs); got "
                f"{tuple(tf2r.shape)}"
            )
        return (tf2r * class_activity).sum(dim=1)

    n_mutations = context_length * 4
    deltas = torch.empty(n_mutations, dtype=torch.float32, device=device)
    with torch.no_grad():
        baseline = predict(reference.unsqueeze(0))[0]
        for offset in range(0, n_mutations, batch_size):
            flat_indices = torch.arange(
                offset, min(offset + batch_size, n_mutations), device=device
            )
            positions = torch.div(flat_indices, 4, rounding_mode="floor")
            bases = flat_indices.remainder(4)
            mutants = reference.unsqueeze(0).expand(len(flat_indices), -1, -1).clone()
            rows = torch.arange(len(flat_indices), device=device)
            mutants[rows, positions] = 0
            mutants[rows, positions, bases] = 1
            deltas[offset : offset + len(flat_indices)] = predict(mutants) - baseline

    scores = deltas.reshape(context_length, 4).cpu().numpy()
    base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    for position, base in enumerate(sequence):
        if base in base_to_index:
            scores[position, base_to_index[base]] = 0.0

    return ISMResult(
        scores=scores,
        sequence=sequence,
        region=region,
        sequence_region=f"{chrom}:{start}-{end}",
        baseline_score=float(baseline.cpu()),
        class_key=class_key,
        class_label=str(class_label),
    )
