"""Tests for dataset fetching utilities."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import deepscenic as ds
from deepscenic.datasets import TF_COLLECTION_URLS


class TestFetchTFCollection:
    """Tests for fetch_tf_collection function."""

    def test_invalid_species(self):
        """Test error on invalid species."""
        with pytest.raises(ValueError, match="Unknown species"):
            ds.datasets.fetch_tf_collection(species="invalid")

    def test_species_urls_defined(self):
        """Test that all expected species have URLs defined."""
        assert "human" in TF_COLLECTION_URLS
        assert "mouse" in TF_COLLECTION_URLS
        assert "fly" in TF_COLLECTION_URLS

    @patch("pooch.retrieve")
    def test_fetch_returns_list(self, mock_retrieve, tmp_path):
        """Test that fetch returns a list of TF names."""
        # Create mock TF file
        tf_file = tmp_path / "allTFs_mouse.txt"
        tf_file.write_text("Adnp\nAebp1\nAebp2\nAhr\n")

        mock_retrieve.return_value = str(tf_file)

        tfs = ds.datasets.fetch_tf_collection(species="mouse")

        assert isinstance(tfs, list)
        assert len(tfs) == 4
        assert tfs[0] == "Adnp"
        assert tfs[3] == "Ahr"

    @patch("pooch.retrieve")
    def test_fetch_handles_empty_lines(self, mock_retrieve, tmp_path):
        """Test that empty lines are ignored."""
        tf_file = tmp_path / "allTFs_mouse.txt"
        tf_file.write_text("Adnp\n\nAebp1\n  \nAhr\n")

        mock_retrieve.return_value = str(tf_file)

        tfs = ds.datasets.fetch_tf_collection(species="mouse")

        assert len(tfs) == 3
        assert "" not in tfs

    @patch("pooch.retrieve")
    def test_fetch_uses_correct_url(self, mock_retrieve, tmp_path):
        """Test that correct URL is used for each species."""
        tf_file = tmp_path / "allTFs.txt"
        tf_file.write_text("TF1\n")
        mock_retrieve.return_value = str(tf_file)

        ds.datasets.fetch_tf_collection(species="mouse")

        # Check the URL used
        call_kwargs = mock_retrieve.call_args[1]
        assert "allTFs_mm.txt" in call_kwargs["url"]

        mock_retrieve.reset_mock()
        ds.datasets.fetch_tf_collection(species="human")

        call_kwargs = mock_retrieve.call_args[1]
        assert "allTFs_hg38.txt" in call_kwargs["url"]


class TestFetchGeneAnnotation:
    """Tests for fetch_gene_annotation function."""

    @patch("pybiomart.Server")
    def test_invalid_species_raises(self, mock_server):
        """Test error on invalid species."""
        # Setup mock to return empty dataset list
        mock_mart = MagicMock()
        mock_mart.list_datasets.return_value = pd.DataFrame({"name": []})
        mock_server.return_value.__getitem__.return_value = mock_mart

        with pytest.raises(ValueError, match="not found"):
            ds.datasets.fetch_gene_annotation(species="invalid_species_xyz")

    @patch("pybiomart.Server")
    def test_fetch_returns_dataframe(self, mock_server):
        """Test that fetch returns properly formatted DataFrame."""
        # Setup mock dataset
        mock_dataset = MagicMock()
        mock_dataset.display_name = "Mouse genes (GRCm38.p6)"
        mock_dataset.attributes.keys.return_value = ["external_gene_name", "transcription_start_site"]

        # Mock query response
        mock_query_result = pd.DataFrame(
            {
                "Chromosome": ["1", "1", "2"],
                "Start": [1000, 2000, 3000],
                "End": [1500, 2500, 3500],
                "Strand": [1, -1, 1],
                "Gene": ["Gene1", "Gene2", "Gene3"],
                "Transcription_Start_Site": [1000, 2500, 3000],
                "Transcript_type": ["protein_coding"] * 3,
            }
        )
        mock_dataset.query.return_value = mock_query_result

        # Setup mart mock
        mock_mart = MagicMock()
        mock_mart.list_datasets.return_value = pd.DataFrame({"name": ["mmusculus_gene_ensembl"]})
        mock_mart.__getitem__.return_value = mock_dataset

        mock_server.return_value.__getitem__.return_value = mock_mart

        # Mock NCBI calls to fail gracefully
        with patch("deepscenic.datasets.gene_annotation._fetch_chromsizes_and_filter") as mock_ncbi:
            from deepscenic.datasets.gene_annotation import NCBIError

            mock_ncbi.side_effect = NCBIError("Mocked NCBI failure")

            annot, chromsizes = ds.datasets.fetch_gene_annotation(species="mmusculus")

        # Verify result structure
        assert isinstance(annot, pd.DataFrame)
        assert annot.index.name == "Gene"
        assert "Chromosome" in annot.columns
        assert "Transcription_Start_Site" in annot.columns
        assert len(annot) == 3

        # Check strand conversion
        assert annot.loc["Gene1", "Strand"] == "+"
        assert annot.loc["Gene2", "Strand"] == "-"

    @patch("pybiomart.Server")
    def test_fetch_removes_duplicates(self, mock_server):
        """Test that duplicate genes are removed."""
        mock_dataset = MagicMock()
        mock_dataset.display_name = "Mouse genes (GRCm38.p6)"
        mock_dataset.attributes.keys.return_value = ["external_gene_name", "transcription_start_site"]

        # Mock query with duplicate gene names
        mock_query_result = pd.DataFrame(
            {
                "Chromosome": ["1", "1", "2"],
                "Start": [1000, 2000, 3000],
                "End": [1500, 2500, 3500],
                "Strand": [1, -1, 1],
                "Gene": ["Gene1", "Gene1", "Gene2"],  # Gene1 appears twice
                "Transcription_Start_Site": [1000, 2000, 3000],
                "Transcript_type": ["protein_coding"] * 3,
            }
        )
        mock_dataset.query.return_value = mock_query_result

        mock_mart = MagicMock()
        mock_mart.list_datasets.return_value = pd.DataFrame({"name": ["mmusculus_gene_ensembl"]})
        mock_mart.__getitem__.return_value = mock_dataset
        mock_server.return_value.__getitem__.return_value = mock_mart

        with patch("deepscenic.datasets.gene_annotation._fetch_chromsizes_and_filter") as mock_ncbi:
            from deepscenic.datasets.gene_annotation import NCBIError

            mock_ncbi.side_effect = NCBIError("Mocked")

            annot, _ = ds.datasets.fetch_gene_annotation(species="mmusculus")

        # Should only have 2 genes (duplicates removed)
        assert len(annot) == 2
        assert "Gene1" in annot.index
        assert "Gene2" in annot.index
