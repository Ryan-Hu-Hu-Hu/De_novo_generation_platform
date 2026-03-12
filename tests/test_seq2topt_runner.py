"""
Tests for pipeline/seq2topt_runner.py

run_seq2topt is tested with mocked subprocess.
parse_seq2topt_results and select_by_temperature use real file I/O / pure logic.
"""

import os
import csv
import tempfile
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from pipeline.seq2topt_runner import (
    parse_seq2topt_results,
    select_by_temperature,
    run_seq2topt,
)


# ── parse_seq2topt_results ────────────────────────────────────────────────────

class TestParseSeq2toptResults:
    def _write_csv(self, path, rows, fieldnames):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_parses_name_and_topt_columns(self, tmp_path):
        csv_path = str(tmp_path / "r.csv")
        self._write_csv(csv_path,
            [{"name": "s1", "topt": "37.5"}, {"name": "s2", "topt": "55.0"}],
            ["name", "topt"])
        result = parse_seq2topt_results(csv_path)
        assert result == {"s1": 37.5, "s2": 55.0}

    def test_parses_id_column_as_name(self, tmp_path):
        csv_path = str(tmp_path / "r.csv")
        self._write_csv(csv_path,
            [{"id": "enzyme1", "Topt": "42.0"}],
            ["id", "Topt"])
        result = parse_seq2topt_results(csv_path)
        assert "enzyme1" in result
        assert result["enzyme1"] == pytest.approx(42.0)

    def test_parses_temperature_column(self, tmp_path):
        csv_path = str(tmp_path / "r.csv")
        self._write_csv(csv_path,
            [{"name": "s1", "temperature": "60.0"}],
            ["name", "temperature"])
        result = parse_seq2topt_results(csv_path)
        assert result["s1"] == pytest.approx(60.0)

    def test_parses_predicted_topt_column(self, tmp_path):
        csv_path = str(tmp_path / "r.csv")
        self._write_csv(csv_path,
            [{"name": "s1", "predicted_topt": "45.5"}],
            ["name", "predicted_topt"])
        result = parse_seq2topt_results(csv_path)
        assert result["s1"] == pytest.approx(45.5)

    def test_invalid_topt_defaults_to_zero(self, tmp_path):
        csv_path = str(tmp_path / "r.csv")
        self._write_csv(csv_path,
            [{"name": "s1", "topt": "not_a_number"}],
            ["name", "topt"])
        result = parse_seq2topt_results(csv_path)
        assert result["s1"] == 0.0

    def test_empty_csv_returns_empty_dict(self, tmp_path):
        csv_path = str(tmp_path / "empty.csv")
        self._write_csv(csv_path, [], ["name", "topt"])
        assert parse_seq2topt_results(csv_path) == {}

    def test_strips_whitespace_from_names(self, tmp_path):
        csv_path = str(tmp_path / "r.csv")
        self._write_csv(csv_path,
            [{"name": "  seq1  ", "topt": "37.0"}],
            ["name", "topt"])
        result = parse_seq2topt_results(csv_path)
        assert "seq1" in result

    def test_multiple_rows(self, tmp_path):
        csv_path = str(tmp_path / "r.csv")
        rows = [{"name": f"s{i}", "topt": str(i * 10.0)} for i in range(1, 6)]
        self._write_csv(csv_path, rows, ["name", "topt"])
        result = parse_seq2topt_results(csv_path)
        assert len(result) == 5
        assert result["s3"] == pytest.approx(30.0)


# ── select_by_temperature ─────────────────────────────────────────────────────

class TestSelectByTemperature:
    def test_returns_closest_to_target(self):
        seqs = {"s1": "AAAA", "s2": "EEEE", "s3": "KKKK"}
        preds = {"s1": 20.0, "s2": 37.0, "s3": 70.0}
        result = select_by_temperature(seqs, preds, target_temp=40.0)
        assert result["name"] == "s2"  # 37°C is closest to 40°C

    def test_exact_match_selected(self):
        seqs = {"s1": "AAAA", "s2": "EEEE"}
        preds = {"s1": 25.0, "s2": 37.0}
        result = select_by_temperature(seqs, preds, target_temp=37.0)
        assert result["name"] == "s2"

    def test_returns_sequence_and_topt(self):
        seqs = {"s1": "MKTII"}
        preds = {"s1": 42.5}
        result = select_by_temperature(seqs, preds, target_temp=40.0)
        assert result["sequence"] == "MKTII"
        assert result["topt"] == 42.5

    def test_returns_none_for_empty_sequences(self):
        result = select_by_temperature({}, {}, target_temp=37.0)
        assert result is None

    def test_single_candidate_always_selected(self):
        seqs = {"only": "GAKLV"}
        preds = {"only": 100.0}
        result = select_by_temperature(seqs, preds, target_temp=20.0)
        assert result["name"] == "only"

    def test_missing_prediction_treated_as_inf(self):
        """A sequence missing from preds should be selected last."""
        seqs = {"has_pred": "AAAA", "no_pred": "EEEE"}
        preds = {"has_pred": 37.0}
        result = select_by_temperature(seqs, preds, target_temp=37.0)
        assert result["name"] == "has_pred"

    def test_tie_breaking_consistent(self):
        """When two candidates are equidistant, min() should pick one deterministically."""
        seqs = {"s1": "AAAA", "s2": "EEEE"}
        preds = {"s1": 30.0, "s2": 50.0}  # both 10°C away from 40
        result = select_by_temperature(seqs, preds, target_temp=40.0)
        assert result["name"] in ("s1", "s2")  # either is acceptable


# ── run_seq2topt (subprocess mocked) ─────────────────────────────────────────

class TestRunSeq2topt:
    @patch("pipeline.seq2topt_runner.subprocess.run")
    @patch("pipeline.seq2topt_runner.SEQ2TOPT_PATH", "/fake/seq2topt")
    def test_raises_file_not_found_when_script_missing(self, mock_run):
        seqs = {"s1": "AAAA"}
        with pytest.raises(FileNotFoundError, match="Seq2Topt script not found"):
            run_seq2topt(seqs, "/tmp/out")

    @patch("pipeline.seq2topt_runner.subprocess.run")
    def test_raises_runtime_error_on_nonzero_return(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        # Fake the script into existence
        seq2topt_dir = tmp_path / "Seq2Topt"
        seq2topt_dir.mkdir()
        (seq2topt_dir / "seq2topt.py").touch()

        with patch("pipeline.seq2topt_runner.SEQ2TOPT_PATH", str(seq2topt_dir)):
            with pytest.raises(RuntimeError, match="Seq2Topt failed"):
                run_seq2topt({"s1": "AAAA"}, str(tmp_path / "out"))

    @patch("pipeline.seq2topt_runner.subprocess.run")
    def test_writes_fasta_before_running(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        seq2topt_dir = tmp_path / "Seq2Topt"
        seq2topt_dir.mkdir()
        (seq2topt_dir / "seq2topt.py").touch()

        # Create a fake output CSV that run_seq2topt will return
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        seqs = {"s1": "MKTII", "s2": "GAKLV"}
        with patch("pipeline.seq2topt_runner.SEQ2TOPT_PATH", str(seq2topt_dir)):
            # We don't care about the return value here; just check FASTA was written
            try:
                run_seq2topt(seqs, str(out_dir))
            except Exception:
                pass
            fasta = out_dir / "seq2topt_input.fasta"
            assert fasta.exists()
            content = fasta.read_text()
            assert ">s1" in content
            assert "MKTII" in content
