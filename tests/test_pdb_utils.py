"""
Tests for pipeline/pdb_utils.py

Network calls are mocked with unittest.mock to avoid internet dependency.
get_template_sequence is tested with a minimal synthetic PDB.
"""

import os
import tempfile
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from pipeline.pdb_utils import (
    download_pdb,
    validate_pdb_id,
    get_ligand_smiles,
    get_template_sequence,
)

# ── Minimal valid PDB string (alanine dipeptide – no actual residues, but
#    enough for BioPython to parse chain A with one residue) ──────────────────
MINIMAL_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   3.000   3.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.000   1.000   4.000  1.00  0.00           C
ATOM      6  N   GLY A   2       4.000   2.000   3.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       5.000   2.000   3.000  1.00  0.00           C
ATOM      8  C   GLY A   2       6.000   2.000   3.000  1.00  0.00           C
ATOM      9  O   GLY A   2       6.000   3.000   3.000  1.00  0.00           O
END
"""


# ── download_pdb ──────────────────────────────────────────────────────────────

class TestDownloadPdb:
    @patch("pipeline.pdb_utils.requests.get")
    def test_creates_pdb_file(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(status_code=200, text=MINIMAL_PDB)
        mock_get.return_value.raise_for_status = lambda: None

        path = download_pdb("1PMO", str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith("1PMO.pdb")

    @patch("pipeline.pdb_utils.requests.get")
    def test_file_content_written_correctly(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(status_code=200, text=MINIMAL_PDB)
        mock_get.return_value.raise_for_status = lambda: None

        path = download_pdb("1PMO", str(tmp_path))
        with open(path) as fh:
            content = fh.read()
        assert "ATOM" in content

    @patch("pipeline.pdb_utils.requests.get")
    def test_pdb_id_uppercased_in_filename(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(status_code=200, text=MINIMAL_PDB)
        mock_get.return_value.raise_for_status = lambda: None

        path = download_pdb("1pmo", str(tmp_path))
        assert "1PMO.pdb" in path

    @patch("pipeline.pdb_utils.requests.get")
    def test_cached_file_not_re_downloaded(self, mock_get, tmp_path):
        """If the file already exists, requests should not be called again."""
        existing = tmp_path / "1PMO.pdb"
        existing.write_text(MINIMAL_PDB)

        download_pdb("1PMO", str(tmp_path))
        mock_get.assert_not_called()

    @patch("pipeline.pdb_utils.requests.get")
    def test_creates_output_directory_if_missing(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(status_code=200, text=MINIMAL_PDB)
        mock_get.return_value.raise_for_status = lambda: None

        nested = str(tmp_path / "new" / "subdir")
        download_pdb("1PMO", nested)
        assert os.path.isdir(nested)

    @patch("pipeline.pdb_utils.requests.get")
    def test_calls_correct_rcsb_url(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(status_code=200, text=MINIMAL_PDB)
        mock_get.return_value.raise_for_status = lambda: None

        download_pdb("1PMO", str(tmp_path))
        called_url = mock_get.call_args[0][0]
        assert "1PMO.pdb" in called_url
        assert "rcsb.org" in called_url


# ── validate_pdb_id ───────────────────────────────────────────────────────────

class TestValidatePdbId:
    @patch("pipeline.pdb_utils.requests.get")
    def test_returns_true_for_200_response(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert validate_pdb_id("1PMO") is True

    @patch("pipeline.pdb_utils.requests.get")
    def test_returns_false_for_404_response(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        assert validate_pdb_id("XXXX") is False

    @patch("pipeline.pdb_utils.requests.get")
    def test_returns_false_on_network_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.RequestException("timeout")
        assert validate_pdb_id("1PMO") is False

    @patch("pipeline.pdb_utils.requests.get")
    def test_pdb_id_uppercased_in_url(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        validate_pdb_id("1pmo")
        called_url = mock_get.call_args[0][0]
        assert "1PMO" in called_url


# ── get_template_sequence ─────────────────────────────────────────────────────

class TestGetTemplateSequence:
    def test_extracts_sequence_from_minimal_pdb(self, tmp_path):
        pdb_path = str(tmp_path / "test.pdb")
        with open(pdb_path, "w") as fh:
            fh.write(MINIMAL_PDB)
        seq = get_template_sequence(pdb_path, chain_id="A")
        assert isinstance(seq, str)
        assert len(seq) >= 1
        # ALA = A, GLY = G
        assert "A" in seq or "G" in seq

    def test_returns_string(self, tmp_path):
        pdb_path = str(tmp_path / "test.pdb")
        with open(pdb_path, "w") as fh:
            fh.write(MINIMAL_PDB)
        seq = get_template_sequence(pdb_path)
        assert isinstance(seq, str)

    def test_empty_chain_returns_empty_string(self, tmp_path):
        pdb_path = str(tmp_path / "test.pdb")
        with open(pdb_path, "w") as fh:
            fh.write(MINIMAL_PDB)
        seq = get_template_sequence(pdb_path, chain_id="Z")  # chain Z doesn't exist
        assert seq == ""


# ── get_ligand_smiles ─────────────────────────────────────────────────────────

class TestGetLigandSmiles:
    @patch("pipeline.pdb_utils.requests.get")
    def test_returns_none_when_no_nonpolymer_entities(self, mock_get):
        entry_resp = MagicMock(status_code=200)
        entry_resp.raise_for_status = lambda: None
        entry_resp.json.return_value = {
            "rcsb_entry_info": {"nonpolymer_entity_count": 0}
        }
        mock_get.return_value = entry_resp
        result = get_ligand_smiles("1PMO")
        assert result is None

    @patch("pipeline.pdb_utils.requests.get")
    def test_returns_none_on_entry_fetch_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.RequestException("network error")
        result = get_ligand_smiles("1PMO")
        assert result is None

    @patch("pipeline.pdb_utils.requests.get")
    def test_returns_smiles_when_found(self, mock_get):
        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            if "nonpolymer_entity" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "pdbx_entity_nonpoly": {"comp_id": "ATP"}
                }
            elif "chemcomp" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "rcsb_chem_comp_descriptor": {
                        "smiles": "c1ccccc1"
                    }
                }
            else:
                resp.status_code = 200
                resp.json.return_value = {
                    "rcsb_entry_info": {"nonpolymer_entity_count": 1}
                }
            return resp

        mock_get.side_effect = side_effect
        result = get_ligand_smiles("1PMO")
        assert result == "c1ccccc1"
