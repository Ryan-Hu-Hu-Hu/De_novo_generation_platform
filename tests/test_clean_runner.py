"""
Tests for pipeline/clean_runner.py

run_clean is tested with mocked subprocess.
prepare_clean_input, parse_clean_results, filter_by_ec use real file I/O.
"""

import os
import csv
import tempfile
import shutil
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from pipeline.clean_runner import (
    prepare_clean_input,
    parse_clean_results,
    filter_by_ec,
    run_clean,
)


# ── prepare_clean_input ───────────────────────────────────────────────────────

class TestPrepareCleanInput:
    def test_creates_tsv_with_header(self, tmp_path):
        seqs = {"seq1": "MKTII", "seq2": "GAKLV"}
        csv_out   = str(tmp_path / "out.tsv")
        fasta_out = str(tmp_path / "out.fasta")
        prepare_clean_input(seqs, csv_out, fasta_out)

        with open(csv_out) as fh:
            lines = fh.readlines()
        assert lines[0].strip() == "Entry\tEC_number\tSequence"

    def test_tsv_contains_all_sequences(self, tmp_path):
        seqs = {"A": "MKTI", "B": "GAKLV", "C": "EEEKE"}
        prepare_clean_input(seqs, str(tmp_path / "o.tsv"), str(tmp_path / "o.fasta"))
        with open(str(tmp_path / "o.tsv")) as fh:
            content = fh.read()
        for name, seq in seqs.items():
            assert name in content
            assert seq in content

    def test_placeholder_ec_number_used(self, tmp_path):
        seqs = {"s1": "MKTI"}
        prepare_clean_input(seqs, str(tmp_path / "o.tsv"), str(tmp_path / "o.fasta"))
        with open(str(tmp_path / "o.tsv")) as fh:
            content = fh.read()
        assert "1.1.1.1" in content

    def test_creates_fasta_with_all_sequences(self, tmp_path):
        seqs = {"enzyme1": "MKTII", "enzyme2": "GAKLV"}
        prepare_clean_input(seqs, str(tmp_path / "o.tsv"), str(tmp_path / "o.fasta"))
        with open(str(tmp_path / "o.fasta")) as fh:
            content = fh.read()
        assert ">enzyme1\nMKTII" in content
        assert ">enzyme2\nGAKLV" in content

    def test_creates_parent_directories(self, tmp_path):
        nested = str(tmp_path / "deep" / "dir" / "out.tsv")
        nested_fa = str(tmp_path / "deep" / "dir" / "out.fasta")
        prepare_clean_input({"s": "MKTI"}, nested, nested_fa)
        assert os.path.exists(nested)

    def test_empty_sequences(self, tmp_path):
        prepare_clean_input({}, str(tmp_path / "empty.tsv"), str(tmp_path / "empty.fasta"))
        with open(str(tmp_path / "empty.tsv")) as fh:
            lines = fh.readlines()
        assert len(lines) == 1  # only header


# ── parse_clean_results ───────────────────────────────────────────────────────

class TestParseCleanResults:
    def _write_csv(self, path, rows, fieldnames):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_parses_id_and_prediction_columns(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        self._write_csv(csv_path,
            [{"id": "seq1", "prediction": "1.2.3.4"},
             {"id": "seq2", "prediction": "3.4.5.6"}],
            ["id", "prediction"])
        result = parse_clean_results(csv_path)
        assert result == {"seq1": "1.2.3.4", "seq2": "3.4.5.6"}

    def test_parses_entry_and_ec_number_columns(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        self._write_csv(csv_path,
            [{"Entry": "e1", "EC_number": "2.7.1.1"}],
            ["Entry", "EC_number"])
        result = parse_clean_results(csv_path)
        assert result == {"e1": "2.7.1.1"}

    def test_strips_whitespace_from_names(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        self._write_csv(csv_path,
            [{"id": "  seq1  ", "prediction": " 1.2.3.4 "}],
            ["id", "prediction"])
        result = parse_clean_results(csv_path)
        assert "seq1" in result
        assert result["seq1"] == "1.2.3.4"

    def test_empty_csv_returns_empty_dict(self, tmp_path):
        csv_path = str(tmp_path / "empty.csv")
        self._write_csv(csv_path, [], ["id", "prediction"])
        assert parse_clean_results(csv_path) == {}

    def test_skips_rows_with_empty_name(self, tmp_path):
        csv_path = str(tmp_path / "r.csv")
        self._write_csv(csv_path,
            [{"id": "", "prediction": "1.2.3.4"},
             {"id": "good", "prediction": "4.5.6.7"}],
            ["id", "prediction"])
        result = parse_clean_results(csv_path)
        assert "" not in result
        assert "good" in result


# ── filter_by_ec ──────────────────────────────────────────────────────────────

class TestFilterByEc:
    def test_exact_ec_match_passes(self):
        seqs = {"s1": "MKTII"}
        preds = {"s1": "1.2.3.4"}
        kept = filter_by_ec(seqs, preds, "1.2.3.4")
        assert "s1" in kept

    def test_third_level_match_passes(self):
        """Different 4th level but same 3rd level should pass."""
        seqs = {"s1": "MKTII"}
        preds = {"s1": "1.2.3.99"}   # same up to 3rd level
        kept = filter_by_ec(seqs, preds, "1.2.3.4")
        assert "s1" in kept

    def test_second_level_mismatch_fails(self):
        seqs = {"s1": "MKTII"}
        preds = {"s1": "1.3.3.4"}    # 2nd level differs
        kept = filter_by_ec(seqs, preds, "1.2.3.4")
        assert "s1" not in kept

    def test_completely_different_ec_fails(self):
        seqs = {"s1": "MKTII"}
        preds = {"s1": "5.6.7.8"}
        kept = filter_by_ec(seqs, preds, "1.2.3.4")
        assert "s1" not in kept

    def test_missing_prediction_fails(self):
        """Sequences with no prediction entry should be filtered out."""
        seqs = {"s1": "MKTII"}
        kept = filter_by_ec(seqs, {}, "1.2.3.4")
        assert "s1" not in kept

    def test_multiple_sequences_mixed_results(self):
        seqs = {
            "pass1": "AAA",
            "pass2": "EEE",
            "fail":  "GGG",
        }
        preds = {
            "pass1": "1.2.3.1",
            "pass2": "1.2.3.99",
            "fail":  "2.3.4.5",
        }
        kept = filter_by_ec(seqs, preds, "1.2.3.4")
        assert "pass1" in kept
        assert "pass2" in kept
        assert "fail" not in kept

    def test_empty_sequences_returns_empty(self):
        kept = filter_by_ec({}, {"s": "1.2.3.4"}, "1.2.3.4")
        assert kept == {}


# ── run_clean (subprocess mocked) ────────────────────────────────────────────

class TestRunClean:
    @patch("pipeline.clean_runner.subprocess.run")
    def test_copies_csv_to_clean_dir_and_runs_command(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_csv = str(tmp_path / "test_input.tsv")
        with open(input_csv, "w") as fh:
            fh.write("Entry\tEC_number\tSequence\nseq1\t1.1.1.1\tMKTII\n")

        results_dir = str(tmp_path / "results")

        # Mock the CLEAN results file to exist
        with patch("pipeline.clean_runner.CLEAN_APP_PATH", str(tmp_path / "clean")):
            clean_app = tmp_path / "clean"
            clean_app.mkdir()
            (clean_app / "data").mkdir()
            result_file = clean_app / "results" / "test_input_maxsep.csv"
            result_file.parent.mkdir()
            result_file.write_text("id,prediction\nseq1,1.2.3.4\n")

            out = run_clean(input_csv, results_dir)

        mock_run.assert_called_once()
        assert "test_input_maxsep.csv" in out

    @patch("pipeline.clean_runner.subprocess.run")
    def test_raises_runtime_error_on_nonzero_return(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="CLEAN error")

        input_csv = str(tmp_path / "bad.tsv")
        with open(input_csv, "w") as fh:
            fh.write("Entry\tEC_number\tSequence\n")

        with patch("pipeline.clean_runner.CLEAN_APP_PATH", str(tmp_path / "clean")):
            (tmp_path / "clean" / "data").mkdir(parents=True)
            with pytest.raises(RuntimeError, match="CLEAN failed"):
                run_clean(input_csv, str(tmp_path / "results"))
