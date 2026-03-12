"""
Tests for pipeline/proteinmpnn_runner.py

Subprocess calls are mocked throughout.
update_fixed_positions is tested with real JSONL files.
"""

import os
import json
import tempfile
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock, call
from pipeline.proteinmpnn_runner import (
    update_fixed_positions,
    parse_chains,
    assign_chains,
    make_fixed_positions,
    run_proteinmpnn,
)


# ── update_fixed_positions ────────────────────────────────────────────────────

class TestUpdateFixedPositions:
    def _write_jsonl(self, path, records):
        with open(path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    def test_updates_matching_design_name(self, tmp_path):
        input_jsonl = str(tmp_path / "in.jsonl")
        output_jsonl = str(tmp_path / "out.jsonl")
        records = [{"design1": {"A": [1]}}]
        self._write_jsonl(input_jsonl, records)

        update_fixed_positions(input_jsonl, output_jsonl, {"design1": [5, 6, 7]})

        with open(output_jsonl) as fh:
            updated = json.loads(fh.readline())
        assert updated["design1"]["A"] == [5, 6, 7]

    def test_does_not_update_unmatched_design(self, tmp_path):
        input_jsonl = str(tmp_path / "in.jsonl")
        output_jsonl = str(tmp_path / "out.jsonl")
        records = [{"other": {"A": [1]}}]
        self._write_jsonl(input_jsonl, records)

        update_fixed_positions(input_jsonl, output_jsonl, {"design1": [5, 6, 7]})

        with open(output_jsonl) as fh:
            updated = json.loads(fh.readline())
        # "other" is not in the mapping, so its existing A values are preserved unchanged
        assert updated["other"]["A"] == [1]

    def test_multiple_records_updated_independently(self, tmp_path):
        input_jsonl = str(tmp_path / "in.jsonl")
        output_jsonl = str(tmp_path / "out.jsonl")
        records = [
            {"design1": {"A": [1]}},
            {"design2": {"A": [1]}},
        ]
        self._write_jsonl(input_jsonl, records)
        mapping = {"design1": [10, 11], "design2": [20, 21]}

        update_fixed_positions(input_jsonl, output_jsonl, mapping)

        with open(output_jsonl) as fh:
            r1 = json.loads(fh.readline())
            r2 = json.loads(fh.readline())
        assert r1["design1"]["A"] == [10, 11]
        assert r2["design2"]["A"] == [20, 21]

    def test_empty_fixed_residues_list(self, tmp_path):
        input_jsonl = str(tmp_path / "in.jsonl")
        output_jsonl = str(tmp_path / "out.jsonl")
        records = [{"design1": {"A": [1]}}]
        self._write_jsonl(input_jsonl, records)

        update_fixed_positions(input_jsonl, output_jsonl, {"design1": []})

        with open(output_jsonl) as fh:
            updated = json.loads(fh.readline())
        assert updated["design1"]["A"] == []

    def test_output_file_created(self, tmp_path):
        input_jsonl = str(tmp_path / "in.jsonl")
        output_jsonl = str(tmp_path / "out.jsonl")
        self._write_jsonl(input_jsonl, [{"d": {"A": [1]}}])

        update_fixed_positions(input_jsonl, output_jsonl, {})
        assert os.path.exists(output_jsonl)

    def test_preserves_other_chain_keys(self, tmp_path):
        """Chain B data should not be touched when mapping only sets chain A."""
        input_jsonl = str(tmp_path / "in.jsonl")
        output_jsonl = str(tmp_path / "out.jsonl")
        records = [{"design1": {"A": [1], "B": [99]}}]
        self._write_jsonl(input_jsonl, records)

        update_fixed_positions(input_jsonl, output_jsonl, {"design1": [5]})

        with open(output_jsonl) as fh:
            updated = json.loads(fh.readline())
        assert updated["design1"]["A"] == [5]
        assert updated["design1"]["B"] == [99]


# ── parse_chains (subprocess mocked) ─────────────────────────────────────────

class TestParseChains:
    @patch("pipeline.proteinmpnn_runner.subprocess.run")
    def test_calls_parse_multiple_chains_script(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        jsonl_out = str(tmp_path / "chains.jsonl")
        parse_chains(str(tmp_path / "pdb_dir"), jsonl_out)
        cmd = mock_run.call_args[0][0]
        assert any("parse_multiple_chains.py" in arg for arg in cmd)

    @patch("pipeline.proteinmpnn_runner.subprocess.run")
    def test_raises_on_nonzero_return(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="fail")
        with pytest.raises(RuntimeError):
            parse_chains(str(tmp_path), str(tmp_path / "out.jsonl"))

    @patch("pipeline.proteinmpnn_runner.subprocess.run")
    def test_returns_output_path(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        out = str(tmp_path / "chains.jsonl")
        result = parse_chains(str(tmp_path), out)
        assert result == out


# ── assign_chains (subprocess mocked) ────────────────────────────────────────

class TestAssignChains:
    @patch("pipeline.proteinmpnn_runner.subprocess.run")
    def test_calls_assign_fixed_chains_script(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assign_chains(str(tmp_path / "in.jsonl"), str(tmp_path / "out.jsonl"))
        cmd = mock_run.call_args[0][0]
        assert any("assign_fixed_chains.py" in arg for arg in cmd)

    @patch("pipeline.proteinmpnn_runner.subprocess.run")
    def test_default_chain_is_A(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assign_chains(str(tmp_path / "in.jsonl"), str(tmp_path / "out.jsonl"))
        cmd = mock_run.call_args[0][0]
        assert "A" in cmd


# ── run_proteinmpnn (subprocess + FASTA parsing) ──────────────────────────────

class TestRunProteinmpnn:
    @patch("pipeline.proteinmpnn_runner.subprocess.run")
    def test_calls_protein_mpnn_run_script(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        run_proteinmpnn(
            str(tmp_path / "j.jsonl"),
            str(tmp_path / "a.jsonl"),
            str(tmp_path / "f.jsonl"),
            str(tmp_path / "out"),
        )
        cmd = mock_run.call_args[0][0]
        assert any("protein_mpnn_run.py" in arg for arg in cmd)

    @patch("pipeline.proteinmpnn_runner.subprocess.run")
    def test_parses_fasta_files_in_seqs_dir(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        out_dir = tmp_path / "out"
        seqs_dir = out_dir / "seqs"
        seqs_dir.mkdir(parents=True)

        # Simulate ProteinMPNN FASTA output (first entry = template, rest = designs)
        fasta_content = (
            ">template, score=0.5\n"
            "MKTII\n"
            ">design_0001, score=0.4\n"
            "GAKLV\n"
            ">design_0002, score=0.3\n"
            "EEEKE\n"
        )
        (seqs_dir / "backbone_0.fa").write_text(fasta_content)

        result = run_proteinmpnn(
            str(tmp_path / "j.jsonl"),
            str(tmp_path / "a.jsonl"),
            str(tmp_path / "f.jsonl"),
            str(out_dir),
        )
        assert "backbone_0" in result
        # Template (MKTII) should be excluded; designs should be included
        seqs = result["backbone_0"]
        assert "GAKLV" in seqs
        assert "EEEKE" in seqs
        assert "MKTII" not in seqs

    @patch("pipeline.proteinmpnn_runner.subprocess.run")
    def test_returns_empty_dict_when_no_seqs_dir(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = run_proteinmpnn(
            str(tmp_path / "j.jsonl"),
            str(tmp_path / "a.jsonl"),
            str(tmp_path / "f.jsonl"),
            str(tmp_path / "out_no_seqs"),
        )
        assert result == {}

    @patch("pipeline.proteinmpnn_runner.subprocess.run")
    def test_raises_on_subprocess_failure(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        with pytest.raises(RuntimeError):
            run_proteinmpnn(
                str(tmp_path / "j.jsonl"),
                str(tmp_path / "a.jsonl"),
                str(tmp_path / "f.jsonl"),
                str(tmp_path / "out"),
            )
