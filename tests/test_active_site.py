"""
Tests for pipeline/active_site.py

run_p2rank is tested with mocks (requires Java binary).
parse_p2rank_output and residues_to_contig use temp files / pure logic.
"""

import os
import csv
import tempfile
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from pipeline.active_site import run_p2rank, parse_p2rank_output, residues_to_contig


# ── residues_to_contig ────────────────────────────────────────────────────────

class TestResiduesToContig:
    def test_empty_fixed_returns_full_diffuse(self):
        result = residues_to_contig([], total_length=50)
        assert result == "[50-50]"

    def test_all_residues_fixed(self):
        result = residues_to_contig([1, 2, 3], total_length=3)
        assert result == "[A1-3]"

    def test_single_fixed_residue_in_middle(self):
        # total=10, fixed=[5] → [4-4/A5-5/5-5]
        result = residues_to_contig([5], total_length=10)
        assert "A5-5" in result
        assert result.startswith("[")
        assert result.endswith("]")

    def test_contiguous_fixed_block(self):
        # fixed=[5,6,7], total=10 → [4-4/A5-7/3-3]
        result = residues_to_contig([5, 6, 7], total_length=10)
        assert "A5-7" in result

    def test_two_separate_fixed_blocks(self):
        # fixed=[3,4,8], total=10 → [2-2/A3-4/3-3/A8-8/2-2]
        result = residues_to_contig([3, 4, 8], total_length=10)
        assert "A3-4" in result
        assert "A8-8" in result

    def test_first_residue_fixed(self):
        result = residues_to_contig([1], total_length=5)
        assert result.startswith("[A1-1")

    def test_last_residue_fixed(self):
        result = residues_to_contig([5], total_length=5)
        assert result.endswith("A5-5]")

    def test_free_segment_count_correct(self):
        """Free segment numbers must sum correctly with fixed regions."""
        fixed = [5, 6, 7]
        total = 10
        result = residues_to_contig(fixed, total)
        # Remove brackets and split
        parts = result.strip("[]").split("/")
        free_total = 0
        fixed_total = 0
        for p in parts:
            if p.startswith("A"):
                lo, hi = p[1:].split("-")
                fixed_total += int(hi) - int(lo) + 1
            else:
                n = int(p.split("-")[0])
                free_total += n
        assert free_total + fixed_total == total

    def test_unsorted_input_sorted_internally(self):
        r1 = residues_to_contig([7, 5, 6], total_length=10)
        r2 = residues_to_contig([5, 6, 7], total_length=10)
        # Both should produce the same contig since the function uses a set
        assert r1 == r2

    def test_custom_chain_label(self):
        result = residues_to_contig([3], total_length=5, chain="B")
        assert "B3-3" in result

    def test_brackets_present(self):
        for fixed in [[], [1], [2, 3], [1, 5, 10]]:
            result = residues_to_contig(fixed, total_length=10)
            assert result.startswith("[")
            assert result.endswith("]")

    def test_large_protein(self):
        """Verify it works for a realistic protein length without error."""
        fixed = list(range(100, 130))  # residues 100–129
        result = residues_to_contig(fixed, total_length=300)
        assert "A100-129" in result


# ── parse_p2rank_output ───────────────────────────────────────────────────────

class TestParseP2rankOutput:
    def _write_residues_csv(self, tmpdir, pdb_stem, rows):
        """Write a synthetic P2Rank residues CSV in the visualizations sub-dir."""
        vis_dir = os.path.join(tmpdir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        csv_path = os.path.join(vis_dir, f"{pdb_stem}.pdb_residues.csv")
        fieldnames = ["residue_label", "pocket_rank", "score"]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return tmpdir

    def test_returns_residues_of_pocket_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"residue_label": "A_5",  "pocket_rank": "1", "score": "0.9"},
                {"residue_label": "A_6",  "pocket_rank": "1", "score": "0.8"},
                {"residue_label": "A_20", "pocket_rank": "2", "score": "0.5"},
            ]
            self._write_residues_csv(tmpdir, "1PMO", rows)
            result = parse_p2rank_output(tmpdir, "1PMO")
            assert result == [5, 6]

    def test_excludes_pocket_2_and_higher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"residue_label": "A_10", "pocket_rank": "1", "score": "0.9"},
                {"residue_label": "A_99", "pocket_rank": "2", "score": "0.4"},
            ]
            self._write_residues_csv(tmpdir, "test", rows)
            result = parse_p2rank_output(tmpdir, "test")
            assert 99 not in result

    def test_sorted_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"residue_label": "A_30", "pocket_rank": "1", "score": "0.7"},
                {"residue_label": "A_5",  "pocket_rank": "1", "score": "0.9"},
                {"residue_label": "A_15", "pocket_rank": "1", "score": "0.8"},
            ]
            self._write_residues_csv(tmpdir, "struct", rows)
            result = parse_p2rank_output(tmpdir, "struct")
            assert result == sorted(result)

    def test_deduplicated(self):
        """Duplicate residues should appear only once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"residue_label": "A_5", "pocket_rank": "1", "score": "0.9"},
                {"residue_label": "A_5", "pocket_rank": "1", "score": "0.9"},
            ]
            self._write_residues_csv(tmpdir, "dup", rows)
            result = parse_p2rank_output(tmpdir, "dup")
            assert result.count(5) == 1

    def test_missing_csv_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                parse_p2rank_output(tmpdir, "nonexistent")

    def test_fallback_path_without_visualizations_subdir(self):
        """Older P2Rank versions write CSV directly in output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "old.pdb_residues.csv")
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["residue_label", "pocket_rank", "score"])
                writer.writeheader()
                writer.writerow({"residue_label": "A_7", "pocket_rank": "1", "score": "0.8"})
            result = parse_p2rank_output(tmpdir, "old")
            assert 7 in result

    def test_invalid_residue_label_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"residue_label": "A_5",   "pocket_rank": "1", "score": "0.9"},
                {"residue_label": "A_BAD", "pocket_rank": "1", "score": "0.9"},
            ]
            self._write_residues_csv(tmpdir, "bad", rows)
            result = parse_p2rank_output(tmpdir, "bad")
            assert 5 in result
            assert len(result) == 1


# ── run_p2rank ────────────────────────────────────────────────────────────────

class TestRunP2rank:
    def test_raises_file_not_found_when_binary_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="P2Rank binary not found"):
            run_p2rank(str(tmp_path / "fake.pdb"), str(tmp_path / "out"))

    @patch("pipeline.active_site.subprocess.run")
    @patch("pipeline.active_site.P2RANK_PATH", new="/fake/p2rank")
    def test_calls_prank_binary(self, mock_run, tmp_path):
        # Make the binary "exist"
        prank_bin = "/fake/p2rank/prank"
        with patch("os.path.isfile", return_value=True):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            run_p2rank(str(tmp_path / "fake.pdb"), str(tmp_path / "out"))
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "prank" in cmd[0]
        assert "predict" in cmd

    @patch("pipeline.active_site.subprocess.run")
    @patch("os.path.isfile", return_value=True)
    def test_raises_on_non_zero_return(self, mock_isfile, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stderr="error text")
        with pytest.raises(RuntimeError, match="P2Rank failed"):
            run_p2rank(str(tmp_path / "fake.pdb"), str(tmp_path / "out"))
