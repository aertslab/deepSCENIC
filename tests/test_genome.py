"""Tests for genome module."""

import random

import pytest
import torch

from deepscenic.genome import (
    Genome,
    GenomeIntervalDataset,
    clear_genome,
    get_genome,
    register_genome,
)


@pytest.fixture
def tmp_fasta(tmp_path):
    """Create a temporary FASTA file for testing."""
    fasta_path = tmp_path / "test.fa"

    # Generate a deterministic sequence for reproducible tests
    random.seed(42)
    bases = "ACGT"
    seq = "".join(random.choice(bases) for _ in range(10000))

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i : i + 80] + "\n")
        # Add a second chromosome
        f.write(">chr2\n")
        seq2 = "".join(random.choice(bases) for _ in range(5000))
        for i in range(0, len(seq2), 80):
            f.write(seq2[i : i + 80] + "\n")

    # Create the index file using pyfaidx
    import pyfaidx

    pyfaidx.Fasta(str(fasta_path))

    return fasta_path


@pytest.fixture(autouse=True)
def reset_genome():
    """Reset global genome state before each test."""
    clear_genome()
    yield
    clear_genome()


class TestGenome:
    """Tests for Genome class."""

    def test_init_with_valid_file(self, tmp_fasta):
        """Genome should initialize with a valid FASTA file."""
        genome = Genome(tmp_fasta)
        assert genome.fasta_file == tmp_fasta
        assert genome.name == "test"

    def test_init_with_invalid_file(self, tmp_path):
        """Genome should raise error for nonexistent file."""
        with pytest.raises(FileNotFoundError, match="FASTA file not found"):
            Genome(tmp_path / "nonexistent.fa")

    def test_chromosomes(self, tmp_fasta):
        """Should return list of chromosome names."""
        genome = Genome(tmp_fasta)
        chroms = genome.chromosomes
        assert "chr1" in chroms
        assert "chr2" in chroms
        assert len(chroms) == 2

    def test_fetch_sequence(self, tmp_fasta):
        """Test basic sequence fetching."""
        genome = Genome(tmp_fasta)
        seq = genome.fetch("chr1", 0, 10)
        assert len(seq) == 10
        assert set(seq).issubset({"A", "C", "G", "T", "N"})

    def test_fetch_sequence_content(self, tmp_fasta):
        """Fetching same region should return same sequence."""
        genome = Genome(tmp_fasta)
        seq1 = genome.fetch("chr1", 100, 200)
        seq2 = genome.fetch("chr1", 100, 200)
        assert seq1 == seq2

    def test_fetch_onehot_shape(self, tmp_fasta):
        """Test one-hot encoding shape."""
        genome = Genome(tmp_fasta)
        onehot = genome.fetch_onehot("chr1", 0, 100)
        assert onehot.shape == (4, 100)

    def test_fetch_onehot_dtype(self, tmp_fasta):
        """One-hot tensor should be float."""
        genome = Genome(tmp_fasta)
        onehot = genome.fetch_onehot("chr1", 0, 10)
        assert onehot.dtype == torch.float32

    def test_fetch_onehot_values(self, tmp_fasta):
        """One-hot columns should sum to 1."""
        genome = Genome(tmp_fasta)
        onehot = genome.fetch_onehot("chr1", 0, 100)
        # Each column should sum to 1
        assert torch.allclose(onehot.sum(dim=0), torch.ones(100))

    def test_reverse_complement(self):
        """Test reverse complement computation."""
        assert Genome._reverse_complement("ACGT") == "ACGT"
        assert Genome._reverse_complement("AAAA") == "TTTT"
        assert Genome._reverse_complement("CCCC") == "GGGG"
        assert Genome._reverse_complement("AACG") == "CGTT"
        assert Genome._reverse_complement("N") == "N"

    def test_fetch_with_rc(self, tmp_fasta):
        """Reverse complement should reverse the sequence."""
        genome = Genome(tmp_fasta)
        seq_fwd = genome.fetch("chr1", 100, 110)
        seq_rc = genome.fetch("chr1", 100, 110, rc=True)
        # RC should be different (unless palindrome)
        assert len(seq_fwd) == len(seq_rc) == 10

    def test_padding_negative_start(self, tmp_fasta):
        """Should pad with N for negative start coordinates."""
        genome = Genome(tmp_fasta)
        seq = genome.fetch("chr1", -10, 10)
        assert len(seq) == 20
        assert seq[:10] == "N" * 10
        assert "N" not in seq[10:]  # No N in valid region

    def test_padding_beyond_end(self, tmp_fasta):
        """Should pad with N for coordinates beyond chromosome end."""
        genome = Genome(tmp_fasta)
        # chr1 has 10000 bases
        seq = genome.fetch("chr1", 9990, 10010)
        assert len(seq) == 20
        assert seq[-10:] == "N" * 10
        assert "N" not in seq[:10]  # No N in valid region

    def test_onehot_n_bases(self, tmp_fasta):
        """N bases should be encoded as 0.25 for each channel."""
        genome = Genome(tmp_fasta)
        # Fetch region with padding (will have N bases)
        onehot = genome.fetch_onehot("chr1", -5, 5)
        # First 5 positions are N, should be 0.25 each
        n_region = onehot[:, :5]
        assert torch.allclose(n_region, torch.full((4, 5), 0.25))


class TestGenomeRegistration:
    """Tests for global genome registration."""

    def test_register_with_path(self, tmp_fasta):
        """Should register genome from path."""
        register_genome(tmp_fasta)
        genome = get_genome()
        assert genome is not None
        assert genome.fasta_file == tmp_fasta

    def test_register_with_genome_instance(self, tmp_fasta):
        """Should accept existing Genome instance."""
        genome = Genome(tmp_fasta)
        register_genome(genome)
        retrieved = get_genome()
        assert retrieved is genome

    def test_get_without_registration(self):
        """Should raise error when no genome registered."""
        with pytest.raises(RuntimeError, match="No genome registered"):
            get_genome()

    def test_clear_genome(self, tmp_fasta):
        """Clear should remove registered genome."""
        register_genome(tmp_fasta)
        assert get_genome() is not None
        clear_genome()
        with pytest.raises(RuntimeError, match="No genome registered"):
            get_genome()

    def test_register_replaces_existing(self, tmp_fasta, tmp_path):
        """Re-registering should replace existing genome."""
        register_genome(tmp_fasta)
        genome1 = get_genome()

        # Create second FASTA
        fasta2 = tmp_path / "test2.fa"
        with open(fasta2, "w") as f:
            f.write(">chrX\nACGT\n")
        import pyfaidx

        pyfaidx.Fasta(str(fasta2))

        register_genome(fasta2)
        genome2 = get_genome()

        assert genome1 is not genome2
        assert genome2.fasta_file == fasta2


class TestGenomeIntervalDataset:
    """Tests for GenomeIntervalDataset."""

    def test_length(self, tmp_fasta):
        """Dataset length should match number of regions."""
        genome = Genome(tmp_fasta)
        regions = ["chr1:0-640", "chr1:100-740", "chr1:200-840"]
        dataset = GenomeIntervalDataset(regions, genome, context_length=640)
        assert len(dataset) == 3

    def test_getitem_returns_tuple(self, tmp_fasta):
        """Getitem should return tuple for compatibility."""
        genome = Genome(tmp_fasta)
        regions = ["chr1:0-640"]
        dataset = GenomeIntervalDataset(regions, genome, context_length=640)
        result = dataset[0]
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_getitem_shape(self, tmp_fasta):
        """Output should have shape (context_length, 4)."""
        genome = Genome(tmp_fasta)
        regions = ["chr1:0-640"]
        dataset = GenomeIntervalDataset(regions, genome, context_length=640)
        (seq,) = dataset[0]
        assert seq.shape == (640, 4)

    def test_different_context_length(self, tmp_fasta):
        """Should respect custom context_length."""
        genome = Genome(tmp_fasta)
        regions = ["chr1:0-640"]
        dataset = GenomeIntervalDataset(regions, genome, context_length=320)
        (seq,) = dataset[0]
        assert seq.shape == (320, 4)

    def test_shift_augmentation(self, tmp_fasta):
        """Shift augmentation should produce different sequences."""
        genome = Genome(tmp_fasta)
        regions = ["chr1:500-1140"]
        dataset = GenomeIntervalDataset(
            regions, genome, context_length=640, shift_augs=(-50, 50)
        )

        # Get multiple samples - they should sometimes differ
        seqs = [(dataset[0][0]) for _ in range(10)]
        # Not all should be identical due to random shifts
        unique_seqs = {tuple(s.flatten().tolist()) for s in seqs}
        assert len(unique_seqs) > 1

    def test_no_shift_augmentation(self, tmp_fasta):
        """Without shift augmentation, same region gives same sequence."""
        genome = Genome(tmp_fasta)
        regions = ["chr1:500-1140"]
        dataset = GenomeIntervalDataset(
            regions, genome, context_length=640, shift_augs=(0, 0), rc_aug=False
        )

        seq1 = dataset[0][0]
        seq2 = dataset[0][0]
        assert torch.equal(seq1, seq2)

    def test_rc_augmentation(self, tmp_fasta):
        """RC augmentation should produce different sequences."""
        genome = Genome(tmp_fasta)
        regions = ["chr1:500-1140"]
        dataset = GenomeIntervalDataset(
            regions, genome, context_length=640, shift_augs=(0, 0), rc_aug=True
        )

        # Get multiple samples - about half should be RC
        seqs = [(dataset[0][0]) for _ in range(20)]
        unique_seqs = {tuple(s.flatten().tolist()) for s in seqs}
        # With RC, we should get at least 2 different versions
        assert len(unique_seqs) >= 2

    def test_parse_region(self):
        """Region parsing should work correctly."""
        chrom, start, end = GenomeIntervalDataset._parse_region("chr1:1000-2000")
        assert chrom == "chr1"
        assert start == 1000
        assert end == 2000

    def test_parse_region_large_coords(self):
        """Should handle large coordinates."""
        chrom, start, end = GenomeIntervalDataset._parse_region("chr1:100000000-100001000")
        assert chrom == "chr1"
        assert start == 100000000
        assert end == 100001000

    def test_output_dtype(self, tmp_fasta):
        """Output tensor should be float32."""
        genome = Genome(tmp_fasta)
        regions = ["chr1:0-640"]
        dataset = GenomeIntervalDataset(regions, genome, context_length=640)
        (seq,) = dataset[0]
        assert seq.dtype == torch.float32

    def test_multiple_regions(self, tmp_fasta):
        """Should correctly handle multiple regions."""
        genome = Genome(tmp_fasta)
        regions = [
            "chr1:0-640",
            "chr1:1000-1640",
            "chr2:0-640",
        ]
        dataset = GenomeIntervalDataset(
            regions, genome, context_length=640, shift_augs=(0, 0), rc_aug=False
        )

        # Each region should give different sequence
        seq0 = dataset[0][0]
        seq1 = dataset[1][0]
        seq2 = dataset[2][0]

        assert not torch.equal(seq0, seq1)
        assert not torch.equal(seq0, seq2)
