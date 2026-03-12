"""SoDoPE SWI solubility runner (in-process, no subprocess needed)."""

import os
import sys
import logging
import math
from typing import Dict, Optional

from .config import SODOPE_PATH

logger = logging.getLogger(__name__)

# SWI weights and logistic constants (from SoDoPE_paper_2020/SWI/swi.py)
_WEIGHTS = {
    'A': 0.8356471476582918, 'C': 0.5208088354857734, 'E': 0.9876987431418378,
    'D': 0.9079044671339564, 'G': 0.7997168496420723, 'F': 0.5849790194237692,
    'I': 0.6784124413866582, 'H': 0.8947913996466419, 'K': 0.9267104557513497,
    'M': 0.6296623675420369, 'L': 0.6554221515081433, 'N': 0.8597433107431216,
    'Q': 0.789434648348208,  'P': 0.8235328714705341, 'S': 0.7440908318492778,
    'R': 0.7712466317693457, 'T': 0.8096922697856334, 'W': 0.6374678690957594,
    'V': 0.7357837119163659, 'Y': 0.6112801822947587,
}
_A = 81.0581
_B = -62.7775


def _compute_swi(sequence: str) -> float:
    """Return the Solubility-Weighted Index for a single amino acid sequence."""
    valid_aas = [aa for aa in sequence.upper() if aa in _WEIGHTS]
    if not valid_aas:
        return 0.0
    swi = sum(_WEIGHTS[aa] for aa in valid_aas) / len(valid_aas)
    return swi


def _compute_prob_solubility(swi: float) -> float:
    """Convert SWI to probability of solubility via logistic function."""
    return 1.0 / (1.0 + math.exp(-(_A * swi + _B)))


def run_sodope(sequences: Dict[str, str]) -> Dict[str, float]:
    """
    Compute SWI solubility scores in-process for a dict of sequences.

    Returns a dict mapping ``seq_name → probability_of_solubility``.
    """
    results: Dict[str, float] = {}
    for name, seq in sequences.items():
        swi = _compute_swi(seq)
        prob = _compute_prob_solubility(swi)
        results[name] = prob
        logger.debug("SoDoPE %s: SWI=%.4f, P(sol)=%.4f", name, swi, prob)
    logger.info("SoDoPE scored %d sequences", len(results))
    return results


def filter_by_solubility(
    sequences: Dict[str, str],
    scores: Dict[str, float],
    template_score: float,
    threshold: Optional[float] = None,
) -> Dict[str, str]:
    """
    Keep sequences with solubility score >= *template_score* (or *threshold*).

    If *threshold* is provided it overrides *template_score*.
    """
    cutoff = threshold if threshold is not None else template_score
    kept: Dict[str, str] = {}
    for name, seq in sequences.items():
        if scores.get(name, 0.0) >= cutoff:
            kept[name] = seq
        else:
            logger.debug(
                "Solubility filter removed %s (%.4f < %.4f)",
                name, scores.get(name, 0.0), cutoff,
            )
    logger.info("Solubility filter: %d/%d sequences passed", len(kept), len(sequences))
    return kept
