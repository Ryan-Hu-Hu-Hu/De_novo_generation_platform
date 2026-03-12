"""
Tests for pipeline/unikp_runner.py

run_unikp is tested with mocked subprocess.
parse_unikp_results and filter_by_kinetics use real file I/O / pure logic.
"""

import os
import csv
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from pipeline.unikp_runner import (
    run_unikp,
    parse_unikp_results,
    filter_by_kinetics,
)


# ── parse_unikp_results ───────────────────────────────────────────────────────

class TestParseUnikpResults:
    def _write_csv(self, path, rows):
        fieldnames = ["name", "kcat", "Km", "kcat_Km"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_parses_all_fields(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        self._write_csv(csv_path, [
            {"name": "s1", "kcat": "10.5", "Km": "0.2",  "kcat_Km": "52.5"},
            {"name": "s2", "kcat": "5.0",  "Km": "0.5",  "kcat_Km": "10.0"},
        ])
        result = parse_unikp_results(csv_path)
        assert result["s1"]["kcat"]    == pytest.approx(10.5)
        assert result["s1"]["Km"]      == pytest.approx(0.2)
        assert result["s1"]["kcat_Km"] == pytest.approx(52.5)

    def test_empty_csv_returns_empty_dict(self, tmp_path):
        csv_path = str(tmp_path / "empty.csv")
        self._write_csv(csv_path, [])
        assert parse_unikp_results(csv_path) == {}

    def test_missing_numeric_value_defaults_to_zero(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        self._write_csv(csv_path, [{"name": "s1", "kcat": "", "Km": "0.5", "kcat_Km": ""}])
        result = parse_unikp_results(csv_path)
        assert result["s1"]["kcat"]    == 0.0
        assert result["s1"]["kcat_Km"] == 0.0

    def test_strips_name_whitespace(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        self._write_csv(csv_path, [{"name": "  s1  ", "kcat": "1.0", "Km": "1.0", "kcat_Km": "1.0"}])
        result = parse_unikp_results(csv_path)
        assert "s1" in result

    def test_multiple_sequences(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        rows = [{"name": f"s{i}", "kcat": str(i), "Km": "1.0", "kcat_Km": str(float(i))} for i in range(1, 6)]
        self._write_csv(csv_path, rows)
        result = parse_unikp_results(csv_path)
        assert len(result) == 5


# ── filter_by_kinetics ────────────────────────────────────────────────────────

class TestFilterByKinetics:
    def _kinetics(self, kcat, km, kcat_km):
        return {"kcat": kcat, "Km": km, "kcat_Km": kcat_km}

    def test_all_pass_when_template_is_zero(self):
        seqs = {"s1": "AAAA", "s2": "EEEE"}
        kinetics = {
            "s1": self._kinetics(0.1, 10.0, 0.01),
            "s2": self._kinetics(0.1, 10.0, 0.01),
        }
        kept = filter_by_kinetics(seqs, kinetics, {"kcat": 0, "Km": float("inf"), "kcat_Km": 0})
        assert set(kept.keys()) == {"s1", "s2"}

    def test_low_kcat_filtered_out(self):
        seqs = {"good": "AAAA", "bad": "EEEE"}
        kinetics = {
            "good": self._kinetics(100.0, 0.1, 1000.0),
            "bad":  self._kinetics(1.0,   0.1, 10.0),
        }
        kept = filter_by_kinetics(seqs, kinetics, {"kcat": 50.0, "Km": 1.0, "kcat_Km": 0})
        assert "good" in kept
        assert "bad" not in kept

    def test_high_km_filtered_out(self):
        seqs = {"good": "AAAA", "bad": "EEEE"}
        kinetics = {
            "good": self._kinetics(50.0, 0.05, 1000.0),
            "bad":  self._kinetics(50.0, 5.0,  10.0),
        }
        kept = filter_by_kinetics(seqs, kinetics, {"kcat": 0, "Km": 0.1, "kcat_Km": 0})
        assert "good" in kept
        assert "bad" not in kept

    def test_low_kcat_km_filtered_out(self):
        seqs = {"good": "AAAA", "bad": "EEEE"}
        kinetics = {
            "good": self._kinetics(50.0, 0.1, 500.0),
            "bad":  self._kinetics(50.0, 0.1, 10.0),
        }
        kept = filter_by_kinetics(seqs, kinetics, {"kcat": 0, "Km": float("inf"), "kcat_Km": 100.0})
        assert "good" in kept
        assert "bad" not in kept

    def test_missing_kinetics_entry_filtered(self):
        seqs = {"s1": "AAAA"}
        kept = filter_by_kinetics(seqs, {}, {"kcat": 1.0, "Km": 1.0, "kcat_Km": 1.0})
        assert "s1" not in kept

    def test_empty_sequences_returns_empty(self):
        kept = filter_by_kinetics({}, {}, {"kcat": 1.0})
        assert kept == {}

    def test_boundary_kcat_passes(self):
        """kcat exactly equal to template kcat should pass."""
        seqs = {"s1": "AAAA"}
        kinetics = {"s1": self._kinetics(50.0, 0.1, 500.0)}
        kept = filter_by_kinetics(seqs, kinetics, {"kcat": 50.0, "Km": float("inf"), "kcat_Km": 0})
        assert "s1" in kept

    def test_boundary_km_passes(self):
        """Km exactly equal to template Km should pass."""
        seqs = {"s1": "AAAA"}
        kinetics = {"s1": self._kinetics(50.0, 0.1, 500.0)}
        kept = filter_by_kinetics(seqs, kinetics, {"kcat": 0, "Km": 0.1, "kcat_Km": 0})
        assert "s1" in kept

    def test_all_three_criteria_must_pass(self):
        """All three conditions must be satisfied; failing any one removes the sequence."""
        template = {"kcat": 10.0, "Km": 1.0, "kcat_Km": 20.0}
        seqs = {"fail_kcat": "A", "fail_km": "E", "fail_kcat_km": "K", "pass": "M"}
        kinetics = {
            "fail_kcat":    self._kinetics(5.0,  0.5,  10.0),   # kcat too low
            "fail_km":      self._kinetics(20.0, 5.0,  4.0),    # Km too high
            "fail_kcat_km": self._kinetics(20.0, 0.5,  5.0),    # kcat/Km too low
            "pass":         self._kinetics(20.0, 0.5, 40.0),    # all good
        }
        kept = filter_by_kinetics(seqs, kinetics, template)
        assert "pass" in kept
        assert "fail_kcat"    not in kept
        assert "fail_km"      not in kept
        assert "fail_kcat_km" not in kept


# ── run_unikp (subprocess mocked) ────────────────────────────────────────────

class TestRunUnikp:
    @patch("pipeline.unikp_runner.subprocess.run")
    def test_writes_input_tsv_with_correct_columns(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        seqs = {"s1": "MKTII", "s2": "GAKLV"}

        # Create fake output CSV
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with patch("pipeline.unikp_runner.UNIKP_PATH", str(tmp_path)):
            # Intercept before script is called so we can check the TSV
            try:
                run_unikp(seqs, "CC(O)=O", str(out_dir))
            except Exception:
                pass

        tsv_path = out_dir / "unikp_input.tsv"
        assert tsv_path.exists()
        with open(tsv_path) as fh:
            header = fh.readline().strip().split("\t")
        assert "name" in header
        assert "sequence" in header
        assert "smiles" in header

    @patch("pipeline.unikp_runner.subprocess.run")
    def test_raises_runtime_error_on_failure(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="UniKP error")
        seqs = {"s1": "MKTII"}
        with patch("pipeline.unikp_runner.UNIKP_PATH", str(tmp_path)):
            with pytest.raises(RuntimeError, match="UniKP failed"):
                run_unikp(seqs, "CC(O)=O", str(tmp_path / "out"))

    @patch("pipeline.unikp_runner.subprocess.run")
    def test_smiles_included_in_tsv_rows(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with patch("pipeline.unikp_runner.UNIKP_PATH", str(tmp_path)):
            try:
                run_unikp({"s1": "MKTII"}, "CC(=O)O", str(out_dir))
            except Exception:
                pass

        tsv_path = out_dir / "unikp_input.tsv"
        content = tsv_path.read_text()
        assert "CC(=O)O" in content
