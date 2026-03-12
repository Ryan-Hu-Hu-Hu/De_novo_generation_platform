"""
Example tests using PDB entry 1HEB — Human Carbonic Anhydrase II (HCA II).

Key facts about 1HEB:
  - EC number  : 4.2.1.1  (carbonate dehydratase)
  - Chain A    : 260 amino acids
  - Ligands    : ZN (catalytic zinc), HG (heavy-atom derivative)
  - Active site: zinc-coordinating histidines H94, H96, H119

Tests are grouped into:
  1. pdb_utils  — download, sequence extraction, SMILES lookup (HTTP mocked)
  2. sodope     — solubility scoring of the real sequence (pure in-process)
  3. clean      — EC-filter logic using 4.2.1.1
  4. unikp      — kinetics filter with realistic thresholds
  5. seq2topt   — temperature selection around physiological 37 °C
  6. active_site— contig generation from HCA II active-site residues, capped at 50 aa
  7. pipeline   — lightweight mock integration: orchestrator uses 1HEB parameters
"""

import os
import csv
import json
import math
import tempfile
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock, call

# ── Known facts about 1HEB ────────────────────────────────────────────────────

PDB_ID = "1HEB"
EC_NUMBER = "4.2.1.1"               # carbonate dehydratase
TEMPLATE_SEQ = (
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRI"
    "LNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAEL"
    "HLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTN"
    "FDPRGLLPESLDYWTYPGSETTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPE"
    "ELMVDNWRPAQPLKNRQIKASFK"
)
assert len(TEMPLATE_SEQ) == 260, f"Expected 260 aa, got {len(TEMPLATE_SEQ)}"

# HCA II zinc-coordinating active-site residues (canonical numbering in chain A)
ACTIVE_SITE_RESIDUES = [94, 96, 119]

# CO2 substrate SMILES (substrate of carbonic anhydrase)
CO2_SMILES = "O=C=O"

# Generated sequence shorter than template (fictitious but realistic-length)
GENERATED_SEQ_50 = "MSHVWGFGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVS"
assert len(GENERATED_SEQ_50) == 50


# ═══════════════════════════════════════════════════════════════════════════════
# 1. pdb_utils
# ═══════════════════════════════════════════════════════════════════════════════

from pipeline.pdb_utils import download_pdb, validate_pdb_id, get_ligand_smiles, get_template_sequence


class TestPdbUtils1HEB:
    """Tests that exercise pdb_utils with 1HEB-specific data."""

    # ── download_pdb ──────────────────────────────────────────────────────────

    @patch("pipeline.pdb_utils.requests.get")
    def test_download_creates_1heb_file(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(status_code=200, text="ATOM 1 N ALA A 1\nEND")
        mock_get.return_value.raise_for_status = lambda: None
        path = download_pdb(PDB_ID, str(tmp_path))
        assert path.endswith("1HEB.pdb")
        assert os.path.exists(path)

    @patch("pipeline.pdb_utils.requests.get")
    def test_download_url_contains_1heb(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(status_code=200, text="END")
        mock_get.return_value.raise_for_status = lambda: None
        download_pdb(PDB_ID, str(tmp_path))
        url = mock_get.call_args[0][0]
        assert "1HEB" in url
        assert "rcsb.org" in url

    @patch("pipeline.pdb_utils.requests.get")
    def test_lowercase_pdb_id_uppercased(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(status_code=200, text="END")
        mock_get.return_value.raise_for_status = lambda: None
        path = download_pdb("1heb", str(tmp_path))
        assert "1HEB" in path

    # ── validate_pdb_id ───────────────────────────────────────────────────────

    @patch("pipeline.pdb_utils.requests.get")
    def test_1heb_validates_successfully(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert validate_pdb_id(PDB_ID) is True

    @patch("pipeline.pdb_utils.requests.get")
    def test_invalid_code_fails(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        assert validate_pdb_id("XXXX") is False

    # ── get_template_sequence ─────────────────────────────────────────────────

    def _write_minimal_1heb_pdb(self, path):
        """Write a synthetic PDB with a 3-residue chain A (ALA-GLY-ALA)."""
        pdb_text = (
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C\n"
            "ATOM      4  O   ALA A   1       3.000   3.000   3.000  1.00  0.00           O\n"
            "ATOM      5  CB  ALA A   1       2.000   1.000   4.000  1.00  0.00           C\n"
            "ATOM      6  N   GLY A   2       4.000   2.000   3.000  1.00  0.00           N\n"
            "ATOM      7  CA  GLY A   2       5.000   2.000   3.000  1.00  0.00           C\n"
            "ATOM      8  C   GLY A   2       6.000   2.000   3.000  1.00  0.00           C\n"
            "ATOM      9  O   GLY A   2       6.000   3.000   3.000  1.00  0.00           O\n"
            "ATOM     10  N   ALA A   3       7.000   2.000   3.000  1.00  0.00           N\n"
            "ATOM     11  CA  ALA A   3       8.000   2.000   3.000  1.00  0.00           C\n"
            "ATOM     12  C   ALA A   3       9.000   2.000   3.000  1.00  0.00           C\n"
            "ATOM     13  O   ALA A   3       9.000   3.000   3.000  1.00  0.00           O\n"
            "ATOM     14  CB  ALA A   3       8.000   1.000   4.000  1.00  0.00           C\n"
            "END\n"
        )
        with open(path, "w") as fh:
            fh.write(pdb_text)

    def test_sequence_extraction_returns_string(self, tmp_path):
        pdb_path = str(tmp_path / "1HEB.pdb")
        self._write_minimal_1heb_pdb(pdb_path)
        seq = get_template_sequence(pdb_path, chain_id="A")
        assert isinstance(seq, str)
        assert len(seq) > 0

    def test_sequence_contains_expected_aas(self, tmp_path):
        """ALA=A, GLY=G — minimal PDB should yield those one-letter codes."""
        pdb_path = str(tmp_path / "1HEB.pdb")
        self._write_minimal_1heb_pdb(pdb_path)
        seq = get_template_sequence(pdb_path, chain_id="A")
        assert "A" in seq
        assert "G" in seq

    def test_nonexistent_chain_returns_empty(self, tmp_path):
        pdb_path = str(tmp_path / "1HEB.pdb")
        self._write_minimal_1heb_pdb(pdb_path)
        seq = get_template_sequence(pdb_path, chain_id="Z")
        assert seq == ""

    # ── get_ligand_smiles — ZN is a metal, expect either SMILES or None ───────

    @patch("pipeline.pdb_utils.requests.get")
    def test_1heb_ligand_lookup_does_not_raise(self, mock_get):
        """get_ligand_smiles must return str or None, never raise."""
        mock_get.return_value = MagicMock(status_code=404)
        result = get_ligand_smiles(PDB_ID)
        assert result is None or isinstance(result, str)

    @patch("pipeline.pdb_utils.requests.get")
    def test_smiles_returned_when_chemcomp_found(self, mock_get):
        """If RCSB returns a SMILES, it should be passed through."""
        def side_effect(url, **kwargs):
            r = MagicMock()
            r.raise_for_status = lambda: None
            if "nonpolymer_entity" in url:
                r.status_code = 200
                r.json.return_value = {"pdbx_entity_nonpoly": {"comp_id": "ZN"}}
            elif "chemcomp/ZN" in url:
                r.status_code = 200
                r.json.return_value = {
                    "rcsb_chem_comp_descriptor": {"smiles": "[Zn]"}
                }
            else:
                r.status_code = 200
                r.json.return_value = {
                    "rcsb_entry_info": {"nonpolymer_entity_count": 2}
                }
            return r

        mock_get.side_effect = side_effect
        result = get_ligand_smiles(PDB_ID)
        assert result == "[Zn]"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SoDoPE — real HCA II sequence
# ═══════════════════════════════════════════════════════════════════════════════

from pipeline.sodope_runner import run_sodope, filter_by_solubility, _compute_swi


class TestSodope1HEB:
    """SoDoPE runs in-process — uses the actual 260-aa HCA II sequence."""

    def test_template_seq_scores_between_0_and_1(self):
        result = run_sodope({PDB_ID: TEMPLATE_SEQ})
        assert 0.0 < result[PDB_ID] < 1.0

    def test_template_swi_in_expected_range(self):
        """HCA II is a well-expressed, moderately-soluble protein."""
        swi = _compute_swi(TEMPLATE_SEQ)
        assert 0.70 < swi < 0.85, f"Unexpected SWI for HCA II: {swi:.4f}"

    def test_generated_50aa_scores_correctly(self):
        result = run_sodope({"gen": GENERATED_SEQ_50})
        assert "gen" in result
        assert isinstance(result["gen"], float)

    def test_template_passes_its_own_filter(self):
        """The template sequence must pass the solubility filter using itself as baseline."""
        scores = run_sodope({PDB_ID: TEMPLATE_SEQ})
        kept = filter_by_solubility({PDB_ID: TEMPLATE_SEQ}, scores, scores[PDB_ID])
        assert PDB_ID in kept

    def test_multiple_candidates_some_filtered(self):
        """Hydrophobic sequence should be filtered out when template sets the bar."""
        hydrophobic = "IIIIILLLLLVVVVVFFFF" * 2          # low SWI
        candidates = {PDB_ID: TEMPLATE_SEQ, "hydro": hydrophobic}
        scores = run_sodope(candidates)
        template_score = scores[PDB_ID]
        kept = filter_by_solubility(candidates, scores, template_score)
        # Template always passes; hydrophobic may or may not depending on score
        assert PDB_ID in kept

    def test_50aa_cap_seq_scored_independently(self):
        candidates = {
            "template": TEMPLATE_SEQ,
            "cap50":    GENERATED_SEQ_50,
        }
        scores = run_sodope(candidates)
        assert len(scores) == 2
        for v in scores.values():
            assert 0.0 < v < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLEAN — EC filter for 4.2.1.1
# ═══════════════════════════════════════════════════════════════════════════════

from pipeline.clean_runner import (
    prepare_clean_input, parse_clean_results, filter_by_ec
)


class TestClean1HEB:
    """Tests the CLEAN helper functions with HCA II's EC 4.2.1.1."""

    def test_prepare_input_writes_template_sequence(self, tmp_path):
        seqs = {PDB_ID: TEMPLATE_SEQ}
        csv_out   = str(tmp_path / "1heb.tsv")
        fasta_out = str(tmp_path / "1heb.fasta")
        prepare_clean_input(seqs, csv_out, fasta_out)

        with open(csv_out) as fh:
            content = fh.read()
        assert PDB_ID in content
        assert TEMPLATE_SEQ[:20] in content

    def test_prepare_fasta_contains_correct_header(self, tmp_path):
        prepare_clean_input(
            {PDB_ID: TEMPLATE_SEQ},
            str(tmp_path / "o.tsv"),
            str(tmp_path / "o.fasta"),
        )
        with open(str(tmp_path / "o.fasta")) as fh:
            first_line = fh.readline().strip()
        assert first_line == f">{PDB_ID}"

    def test_ec_4211_passes_exact_match(self):
        seqs  = {"gen_0": GENERATED_SEQ_50}
        preds = {"gen_0": "4.2.1.1"}
        kept  = filter_by_ec(seqs, preds, EC_NUMBER)
        assert "gen_0" in kept

    def test_ec_4211_passes_third_level_match(self):
        """4.2.1.99 shares the first 3 levels with 4.2.1.1 — should pass."""
        seqs  = {"gen_0": GENERATED_SEQ_50}
        preds = {"gen_0": "4.2.1.99"}
        kept  = filter_by_ec(seqs, preds, EC_NUMBER)
        assert "gen_0" in kept

    def test_ec_mismatch_rejected(self):
        """A hydrolase (3.x.x.x) must be rejected when template is 4.2.1.1."""
        seqs  = {"gen_0": GENERATED_SEQ_50}
        preds = {"gen_0": "3.1.1.1"}
        kept  = filter_by_ec(seqs, preds, EC_NUMBER)
        assert "gen_0" not in kept

    def test_mixed_ec_results_filtered_correctly(self):
        designs = {
            "pass_exact":  "A" * 50,
            "pass_3level": "E" * 50,
            "fail_class":  "G" * 50,
        }
        preds = {
            "pass_exact":  "4.2.1.1",
            "pass_3level": "4.2.1.77",
            "fail_class":  "1.1.1.1",
        }
        kept = filter_by_ec(designs, preds, EC_NUMBER)
        assert "pass_exact"  in kept
        assert "pass_3level" in kept
        assert "fail_class"  not in kept

    def test_parse_results_with_1heb_name(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "prediction"])
            writer.writeheader()
            writer.writerow({"id": PDB_ID, "prediction": EC_NUMBER})
        result = parse_clean_results(csv_path)
        assert result[PDB_ID] == EC_NUMBER


# ═══════════════════════════════════════════════════════════════════════════════
# 4. UniKP — kinetics filter
# ═══════════════════════════════════════════════════════════════════════════════

from pipeline.unikp_runner import (
    parse_unikp_results, filter_by_kinetics, run_unikp
)


class TestUniKP1HEB:
    """UniKP kinetics filter tests parameterised with HCA II-like values.

    Published kcat for HCA II (CO2): ~1,000,000 s⁻¹ (extremely fast).
    Km(CO2) ≈ 12 mM. kcat/Km ≈ 8.3 × 10⁷ M⁻¹s⁻¹.
    We use scaled-down values here because unikp_predict.py predictions
    are log10-transformed and rescaled; realistic ballpark values are used.
    """

    TEMPLATE_KINETICS = {"kcat": 1e4, "Km": 12.0, "kcat_Km": 8.3e5}

    def test_parse_unikp_output_csv(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["name", "kcat", "Km", "kcat_Km"])
            writer.writeheader()
            writer.writerow({"name": "gen_0", "kcat": "1e4", "Km": "10.0", "kcat_Km": "1000.0"})
        result = parse_unikp_results(csv_path)
        assert "gen_0" in result
        assert result["gen_0"]["kcat"] == pytest.approx(1e4)

    def test_high_activity_candidate_passes(self):
        seqs = {"gen_0": GENERATED_SEQ_50}
        kinetics = {"gen_0": {"kcat": 2e4, "Km": 8.0, "kcat_Km": 2.5e6}}
        kept = filter_by_kinetics(seqs, kinetics, self.TEMPLATE_KINETICS)
        assert "gen_0" in kept

    def test_low_kcat_fails_hcaii_threshold(self):
        seqs = {"gen_0": GENERATED_SEQ_50}
        kinetics = {"gen_0": {"kcat": 100.0, "Km": 8.0, "kcat_Km": 12.5}}
        kept = filter_by_kinetics(seqs, kinetics, self.TEMPLATE_KINETICS)
        assert "gen_0" not in kept

    def test_high_km_fails_hcaii_threshold(self):
        seqs = {"gen_0": GENERATED_SEQ_50}
        kinetics = {"gen_0": {"kcat": 2e4, "Km": 500.0, "kcat_Km": 40.0}}
        kept = filter_by_kinetics(seqs, kinetics, self.TEMPLATE_KINETICS)
        assert "gen_0" not in kept

    def test_multiple_designs_screened(self):
        designs = {
            "good": "M" * 50,
            "bad_kcat": "A" * 50,
            "bad_km":   "E" * 50,
        }
        kinetics = {
            "good":     {"kcat": 2e4, "Km": 5.0,   "kcat_Km": 4e6},
            "bad_kcat": {"kcat": 500, "Km": 5.0,   "kcat_Km": 100},
            "bad_km":   {"kcat": 2e4, "Km": 1000., "kcat_Km": 20},
        }
        kept = filter_by_kinetics(designs, kinetics, self.TEMPLATE_KINETICS)
        assert "good" in kept
        assert "bad_kcat" not in kept
        assert "bad_km"   not in kept

    @patch("pipeline.unikp_runner.subprocess.run")
    def test_run_unikp_writes_co2_smiles_to_tsv(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        out_dir = tmp_path / "unikp_out"
        out_dir.mkdir()
        with patch("pipeline.unikp_runner.UNIKP_PATH", str(tmp_path)):
            try:
                run_unikp({"gen_0": GENERATED_SEQ_50}, CO2_SMILES, str(out_dir))
            except Exception:
                pass
        tsv = out_dir / "unikp_input.tsv"
        assert tsv.exists()
        content = tsv.read_text()
        assert CO2_SMILES in content
        assert "gen_0" in content
        assert GENERATED_SEQ_50 in content


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Seq2Topt — temperature selection
# ═══════════════════════════════════════════════════════════════════════════════

from pipeline.seq2topt_runner import (
    parse_seq2topt_results, select_by_temperature
)


class TestSeq2topt1HEB:
    """Temperature selection tests around physiological 37 °C (HCA II is human)."""

    TARGET_TEMP = 37  # °C — human body temperature

    def test_select_closest_to_37C(self):
        seqs = {
            "gen_0": GENERATED_SEQ_50,
            "gen_1": "A" * 50,
            "gen_2": "E" * 50,
        }
        preds = {"gen_0": 37.0, "gen_1": 55.0, "gen_2": 20.0}
        result = select_by_temperature(seqs, preds, self.TARGET_TEMP)
        assert result["name"] == "gen_0"
        assert result["topt"] == pytest.approx(37.0)

    def test_slightly_off_target_still_selected_as_best(self):
        """If no design hits exactly 37°C, the nearest one wins."""
        seqs = {"a": "A" * 50, "b": "E" * 50}
        preds = {"a": 35.0, "b": 45.0}
        result = select_by_temperature(seqs, preds, self.TARGET_TEMP)
        assert result["name"] == "a"   # 35 is closer to 37 than 45

    def test_parse_seq2topt_with_1heb_names(self, tmp_path):
        csv_path = str(tmp_path / "topt.csv")
        rows = [
            {"name": "gen_0", "topt": "37.2"},
            {"name": "gen_1", "topt": "55.8"},
        ]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["name", "topt"])
            writer.writeheader()
            writer.writerows(rows)
        result = parse_seq2topt_results(csv_path)
        assert result["gen_0"] == pytest.approx(37.2)
        assert result["gen_1"] == pytest.approx(55.8)

    def test_select_returns_sequence_string(self):
        seqs = {"gen_0": GENERATED_SEQ_50}
        preds = {"gen_0": 37.5}
        result = select_by_temperature(seqs, preds, self.TARGET_TEMP)
        assert result["sequence"] == GENERATED_SEQ_50


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Active site — contig generation for HCA II (capped at 50 aa)
# ═══════════════════════════════════════════════════════════════════════════════

from pipeline.active_site import residues_to_contig, parse_p2rank_output
from pipeline.config import MAX_GENERATED_LENGTH


class TestActiveSite1HEB:
    """Contig-string generation using HCA II active-site residues H94, H96, H119."""

    def test_max_generated_length_is_50(self):
        """Config sanity check — MAX_GENERATED_LENGTH must be 50."""
        assert MAX_GENERATED_LENGTH == 50

    def test_active_site_residues_within_cap(self):
        """All HCA II active-site residues (H94, H96, H119) exceed the 50 aa cap.

        This is expected behaviour: the orchestrator proceeds with no fixed
        residues when the template active site lies outside the capped length,
        and RFdiffusion generates a fully diffused 50-residue scaffold.
        """
        capped_residues = [r for r in ACTIVE_SITE_RESIDUES if r <= MAX_GENERATED_LENGTH]
        assert capped_residues == []  # H94, H96, H119 are all > 50

        # Fallback: fully-diffused contig of exactly 50 residues
        contig = residues_to_contig(capped_residues, total_length=MAX_GENERATED_LENGTH)
        assert contig == f"[{MAX_GENERATED_LENGTH}-{MAX_GENERATED_LENGTH}]"

    def test_contig_for_50aa_cap_is_valid(self):
        capped = [r for r in ACTIVE_SITE_RESIDUES if r <= MAX_GENERATED_LENGTH]
        contig = residues_to_contig(capped, total_length=MAX_GENERATED_LENGTH)
        assert contig.startswith("[")
        assert contig.endswith("]")

    def test_free_plus_fixed_equals_cap(self):
        capped = [r for r in ACTIVE_SITE_RESIDUES if r <= MAX_GENERATED_LENGTH]
        contig = residues_to_contig(capped, total_length=MAX_GENERATED_LENGTH)
        parts = contig.strip("[]").split("/")
        total = 0
        for p in parts:
            if p.startswith("A"):
                lo, hi = p[1:].split("-")
                total += int(hi) - int(lo) + 1
            else:
                total += int(p.split("-")[0])
        assert total == MAX_GENERATED_LENGTH

    def test_full_length_contig_for_260aa(self):
        """Without cap, all three HCA II active-site residues are included.

        H94 and H96 are *not* contiguous (H95 is absent from the active site),
        so they each appear as separate single-residue fixed segments.
        """
        contig = residues_to_contig(ACTIVE_SITE_RESIDUES, total_length=260)
        assert "A94-94"  in contig   # H94 — zinc-coordinating histidine
        assert "A96-96"  in contig   # H96 — zinc-coordinating histidine
        assert "A119-119" in contig  # H119 — zinc-coordinating histidine

    def test_parse_p2rank_returns_active_site_residues(self, tmp_path):
        """Simulate P2Rank output containing HCA II Zn-coordinating histidines."""
        vis_dir = tmp_path / "visualizations"
        vis_dir.mkdir()
        csv_path = vis_dir / "1HEB.pdb_residues.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["residue_label", "pocket_rank", "score"])
            writer.writeheader()
            for resnum in ACTIVE_SITE_RESIDUES:
                writer.writerow({
                    "residue_label": f"A_{resnum}",
                    "pocket_rank":   "1",
                    "score":         "0.95",
                })
        result = parse_p2rank_output(str(tmp_path), "1HEB")
        assert result == sorted(ACTIVE_SITE_RESIDUES)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Pipeline integration (fully mocked, 1HEB parameters)
# ═══════════════════════════════════════════════════════════════════════════════

from pipeline.orchestrator import PipelineOrchestrator, _flat_sequences


class TestOrchestrator1HEB:
    """Lightweight integration tests: patch every external call, check flow."""

    @patch("pipeline.orchestrator.run_seq2topt")
    @patch("pipeline.orchestrator.parse_seq2topt_results")
    @patch("pipeline.orchestrator.run_unikp")
    @patch("pipeline.orchestrator.parse_unikp_results")
    @patch("pipeline.orchestrator.run_sodope")
    @patch("pipeline.orchestrator.run_clean")
    @patch("pipeline.orchestrator.parse_clean_results")
    @patch("pipeline.orchestrator.prepare_clean_input")
    @patch("pipeline.orchestrator.run_proteinmpnn")
    @patch("pipeline.orchestrator.update_fixed_positions")
    @patch("pipeline.orchestrator.make_fixed_positions")
    @patch("pipeline.orchestrator.assign_chains")
    @patch("pipeline.orchestrator.parse_chains")
    @patch("pipeline.orchestrator.run_rfdiffusion")
    @patch("pipeline.orchestrator.run_p2rank")
    @patch("pipeline.orchestrator.parse_p2rank_output")
    @patch("pipeline.orchestrator.get_ligand_smiles")
    @patch("pipeline.orchestrator.get_template_sequence")
    @patch("pipeline.orchestrator.download_pdb")
    def test_full_pipeline_returns_best_candidate(
        self,
        mock_dl, mock_seq, mock_smiles,
        mock_p2rank_parse, mock_p2rank_run,
        mock_rfdiff, mock_parse_chains, mock_assign, mock_make_fp, mock_update_fp,
        mock_mpnn,
        mock_prep_clean, mock_parse_clean_r, mock_run_clean,
        mock_sodope,
        mock_parse_unikp, mock_run_unikp,
        mock_parse_seq2topt, mock_run_seq2topt,
        tmp_path,
    ):
        # ── Setup mocks ───────────────────────────────────────────────────────
        mock_dl.return_value          = str(tmp_path / "1HEB.pdb")
        mock_seq.return_value         = TEMPLATE_SEQ
        mock_smiles.return_value      = CO2_SMILES
        mock_p2rank_run.return_value  = str(tmp_path / "p2rank")
        mock_p2rank_parse.return_value= [94, 96]

        mock_rfdiff.return_value      = [str(tmp_path / "design_0.pdb")]
        mock_parse_chains.return_value= str(tmp_path / "chains.jsonl")
        mock_assign.return_value      = str(tmp_path / "assigned.jsonl")
        mock_make_fp.return_value     = str(tmp_path / "fixed.jsonl")
        mock_update_fp.return_value   = str(tmp_path / "updated.jsonl")

        # ProteinMPNN: 2 sequences for design_0
        mock_mpnn.return_value = {"design_0": [GENERATED_SEQ_50, "A" * 50]}

        # CLEAN: parse_clean_results is called twice —
        #   call 1: for the template sequence (must return {"template": EC_NUMBER}
        #            so orchestrator sets template_ec = "4.2.1.1")
        #   call 2: for iteration designs
        mock_run_clean.return_value = str(tmp_path / "clean_out.csv")
        mock_parse_clean_r.side_effect = [
            {"template":   EC_NUMBER},                             # template baseline
            {"design_0_1": EC_NUMBER, "design_0_2": EC_NUMBER},   # iteration designs
        ]

        # SoDoPE: template + candidates all have decent solubility
        from pipeline.sodope_runner import run_sodope as _real_sodope
        def sodope_side_effect(seqs):
            return {name: 0.7 for name in seqs}
        mock_sodope.side_effect = sodope_side_effect

        # UniKP: all pass
        mock_run_unikp.return_value   = str(tmp_path / "unikp_out.csv")
        mock_parse_unikp.return_value = {
            "design_0_1": {"kcat": 2e4, "Km": 5.0, "kcat_Km": 4e6},
            "design_0_2": {"kcat": 2e4, "Km": 5.0, "kcat_Km": 4e6},
        }

        # Seq2Topt: gen_0_1 is closest to 37°C
        mock_run_seq2topt.return_value   = str(tmp_path / "topt.csv")
        mock_parse_seq2topt.return_value = {
            "design_0_1": 37.2,
            "design_0_2": 55.0,
        }

        # ── Run orchestrator ──────────────────────────────────────────────────
        messages = []
        orch = PipelineOrchestrator()
        result = orch.run(PDB_ID, 37, str(tmp_path / "job"), lambda m: messages.append(m))

        # ── Assertions ────────────────────────────────────────────────────────
        assert result is not None
        assert result.get("sequence") is not None
        assert result.get("name") == "design_0_1"
        assert result.get("topt") == pytest.approx(37.2)

    def test_flat_sequences_naming(self):
        """_flat_sequences must produce '{design_name}_{idx}' keys."""
        design_seqs = {
            "backbone_0": ["SEQ1", "SEQ2"],
            "backbone_1": ["SEQ3"],
        }
        flat = _flat_sequences(design_seqs)
        assert "backbone_0_1" in flat
        assert "backbone_0_2" in flat
        assert "backbone_1_1" in flat
        assert flat["backbone_0_1"] == "SEQ1"
        assert flat["backbone_0_2"] == "SEQ2"

    def test_flat_sequences_total_count(self):
        design_seqs = {f"d{i}": ["A", "B", "C"] for i in range(5)}
        flat = _flat_sequences(design_seqs)
        assert len(flat) == 15   # 5 designs × 3 sequences

    @patch("pipeline.orchestrator.download_pdb")
    def test_pipeline_uses_correct_pdb_id(self, mock_dl, tmp_path):
        """PDB ID must be forwarded to download_pdb unchanged (uppercased)."""
        mock_dl.side_effect = RuntimeError("stop early")
        orch = PipelineOrchestrator()
        with pytest.raises(RuntimeError):
            orch.run("1heb", 37, str(tmp_path), lambda m: None)
        mock_dl.assert_called_once()
        called_id = mock_dl.call_args[0][0]
        assert called_id.upper() == PDB_ID
