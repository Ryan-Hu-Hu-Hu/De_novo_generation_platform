"""
Tests for pipeline/sodope_runner.py

SoDoPE runs entirely in-process (no external binary or conda env required),
so every function is fully testable without mocking.
"""

import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.sodope_runner import (
    _compute_swi,
    _compute_prob_solubility,
    run_sodope,
    filter_by_solubility,
    _WEIGHTS,
    _A,
    _B,
)


# ── _compute_swi ──────────────────────────────────────────────────────────────

class TestComputeSwi:
    def test_single_residue_A(self):
        assert _compute_swi("A") == pytest.approx(_WEIGHTS["A"])

    def test_single_residue_E(self):
        assert _compute_swi("E") == pytest.approx(_WEIGHTS["E"])

    def test_all_same_residues(self):
        seq = "AAAA"
        assert _compute_swi(seq) == pytest.approx(_WEIGHTS["A"])

    def test_average_of_two_residues(self):
        seq = "AE"
        expected = (_WEIGHTS["A"] + _WEIGHTS["E"]) / 2
        assert _compute_swi(seq) == pytest.approx(expected)

    def test_lowercase_input(self):
        """Sequence should be uppercased internally."""
        assert _compute_swi("aaa") == pytest.approx(_WEIGHTS["A"])

    def test_mixed_case(self):
        assert _compute_swi("Ae") == pytest.approx(_compute_swi("AE"))

    def test_empty_sequence_returns_zero(self):
        assert _compute_swi("") == 0.0

    def test_invalid_characters_ignored(self):
        """Non-standard AAs (X, B, Z) should be ignored; only valid ones averaged."""
        swi_valid = _compute_swi("A")
        swi_with_invalid = _compute_swi("AX")   # X is not in weights
        assert swi_with_invalid == pytest.approx(swi_valid)

    def test_all_invalid_returns_zero(self):
        assert _compute_swi("XXX") == 0.0

    def test_range_0_to_1(self):
        """All weight values are between 0 and 1, so SWI must be too."""
        for seq in ["MKTLLLTLVVVTIVCLDLGAVGNGTCVPINNATLAQDQPSLWCQAMGCHPCGESSEVHWPENSTSPIFYNPNRTEILHGQSLDSPGTSVTIPCHYRNGIGYHCKHKEFLIEGQADNCTKKQNYWNSSNYTILKHFSDNLNFTENDLPNLPQDVAHISSSGNASSGTTNTHNISHNGNKTEYHCNLSFEAGTREGDTFPKFKDLLFNQSSYVYELDLPSTQREP"]:
            swi = _compute_swi(seq)
            assert 0.0 <= swi <= 1.0

    def test_highly_soluble_sequence_higher_swi(self):
        """Sequences rich in E/K/D (high weights) should have higher SWI than F/I/L."""
        high_sol = "EEEEEKKKK"  # high-weight residues
        low_sol  = "FFFFIIILL"  # low-weight residues
        assert _compute_swi(high_sol) > _compute_swi(low_sol)


# ── _compute_prob_solubility ──────────────────────────────────────────────────

class TestComputeProbSolubility:
    def test_output_between_0_and_1(self):
        for swi in [0.0, 0.5, 0.75, 0.85, 1.0]:
            prob = _compute_prob_solubility(swi)
            assert 0.0 < prob < 1.0

    def test_higher_swi_means_higher_probability(self):
        assert _compute_prob_solubility(0.85) > _compute_prob_solubility(0.70)

    def test_logistic_formula(self):
        swi = 0.76
        expected = 1.0 / (1.0 + math.exp(-(_A * swi + _B)))
        assert _compute_prob_solubility(swi) == pytest.approx(expected)

    def test_swi_zero_gives_very_low_prob(self):
        """SWI of 0 should give near-zero solubility probability."""
        prob = _compute_prob_solubility(0.0)
        assert prob < 0.01

    def test_swi_one_gives_high_prob(self):
        prob = _compute_prob_solubility(1.0)
        assert prob > 0.99


# ── run_sodope ────────────────────────────────────────────────────────────────

class TestRunSodope:
    def test_returns_dict_with_correct_keys(self):
        seqs = {"seq1": "AAAA", "seq2": "EEEE"}
        result = run_sodope(seqs)
        assert set(result.keys()) == {"seq1", "seq2"}

    def test_all_values_between_0_and_1(self):
        seqs = {"s1": "MKTII", "s2": "FFFF", "s3": "EEEE"}
        result = run_sodope(seqs)
        for v in result.values():
            assert 0.0 < v < 1.0

    def test_empty_input(self):
        result = run_sodope({})
        assert result == {}

    def test_single_sequence(self):
        result = run_sodope({"only": "AKTVEG"})
        assert "only" in result
        assert isinstance(result["only"], float)

    def test_higher_solubility_for_charged_sequence(self):
        """E/K-rich sequences should score higher than hydrophobic ones."""
        seqs = {
            "charged":      "EKEKEKEKEK",
            "hydrophobic":  "ILIVLLIVIL",
        }
        result = run_sodope(seqs)
        assert result["charged"] > result["hydrophobic"]

    def test_scores_match_manual_calculation(self):
        seq = "AEK"
        swi = (_WEIGHTS["A"] + _WEIGHTS["E"] + _WEIGHTS["K"]) / 3
        expected_prob = 1.0 / (1.0 + math.exp(-(_A * swi + _B)))
        result = run_sodope({"AEK": seq})
        assert result["AEK"] == pytest.approx(expected_prob, rel=1e-6)


# ── filter_by_solubility ──────────────────────────────────────────────────────

class TestFilterBySolubility:
    def setup_method(self):
        self.sequences = {
            "high": "EKEKEKEKEK",
            "mid":  "AKTVEGGILS",
            "low":  "IIIIILLLLL",
        }
        self.scores = run_sodope(self.sequences)

    def test_all_pass_when_cutoff_is_zero(self):
        kept = filter_by_solubility(self.sequences, self.scores, template_score=0.0)
        assert set(kept.keys()) == set(self.sequences.keys())

    def test_none_pass_when_cutoff_is_one(self):
        kept = filter_by_solubility(self.sequences, self.scores, template_score=1.0)
        assert kept == {}

    def test_high_solubility_passes(self):
        cutoff = self.scores["mid"]  # mid-level cutoff
        kept = filter_by_solubility(self.sequences, self.scores, cutoff)
        assert "high" in kept

    def test_low_solubility_filtered_out(self):
        cutoff = self.scores["mid"]
        kept = filter_by_solubility(self.sequences, self.scores, cutoff)
        assert "low" not in kept

    def test_threshold_overrides_template_score(self):
        """When threshold is given, it should take priority."""
        high_score = max(self.scores.values()) + 0.01
        # template_score = 0 (everything should pass), but threshold = 1.0 (nothing passes)
        kept = filter_by_solubility(self.sequences, self.scores, 0.0, threshold=high_score)
        assert kept == {}

    def test_exact_boundary_passes(self):
        """Score equal to cutoff should pass (>= comparison)."""
        name = "high"
        exact_cutoff = self.scores[name]
        kept = filter_by_solubility({name: self.sequences[name]}, self.scores, exact_cutoff)
        assert name in kept

    def test_missing_score_treated_as_zero(self):
        """Sequence not in scores dict defaults to 0.0 → filtered out with any cutoff > 0."""
        kept = filter_by_solubility(
            {"ghost": "AAAA"}, {}, template_score=0.01
        )
        assert kept == {}

    def test_empty_sequences(self):
        kept = filter_by_solubility({}, {}, template_score=0.5)
        assert kept == {}
