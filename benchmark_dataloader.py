"""Benchmark script for dataloader performance."""

import time

from deepscenic._genome import Genome, register_genome
from deepscenic.tl._dataloaders import build_sequence_dataloader


def benchmark_onehot(genome: Genome, n_iterations: int = 1000) -> None:
    """Benchmark one-hot encoding speed."""
    seq = genome.fetch("chr1", 0, 640)

    start = time.perf_counter()
    for _ in range(n_iterations):
        Genome._seq_to_onehot(seq)
    elapsed = time.perf_counter() - start

    print(f"One-hot encoding: {n_iterations} iterations in {elapsed:.3f}s")
    print(f"  {n_iterations / elapsed:.1f} sequences/second")


def benchmark_dataloader(loader, n_batches: int = 10) -> None:
    """Benchmark dataloader iteration speed."""
    start = time.perf_counter()
    total_samples = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        total_samples += batch[0][0].shape[0]
    elapsed = time.perf_counter() - start

    print(f"Dataloader: {total_samples} samples in {elapsed:.3f}s")
    print(f"  {total_samples / elapsed:.1f} samples/second")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benchmark_dataloader.py <fasta_file>")
        print("  Example: python benchmark_dataloader.py /path/to/hg38.fa")
        sys.exit(1)

    fasta_path = sys.argv[1]
    genome = Genome(fasta_path)

    print("=" * 50)
    print("Dataloader Performance Benchmark")
    print("=" * 50)

    print("\n1. One-hot encoding benchmark:")
    benchmark_onehot(genome)

    print("\n2. Sequence dataloader benchmark:")
    register_genome(genome)

    regions = [f"chr1:{i * 100}-{i * 100 + 640}" for i in range(1000)]
    loader = build_sequence_dataloader(
        regions=regions,
        batch_size=100,
        shuffle=False,
        shift_augs=(0, 0),
        rc_aug=False,
    )
    benchmark_dataloader(loader)

    print("\n" + "=" * 50)
