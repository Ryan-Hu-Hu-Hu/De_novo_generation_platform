"""P2Rank integration for active-site / pocket prediction."""

import os
import csv
import logging
import subprocess
from typing import List, Tuple

from .config import P2RANK_PATH

logger = logging.getLogger(__name__)


def run_p2rank(pdb_path: str, output_dir: str) -> str:
    """
    Run P2Rank pocket prediction on *pdb_path*.

    Returns the output directory path where prediction CSV files are written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Locate the prank executable
    prank_bin = os.path.join(P2RANK_PATH, "prank")
    if not os.path.isfile(prank_bin):
        raise FileNotFoundError(
            f"P2Rank binary not found at {prank_bin}. "
            "Download from https://github.com/rdk/p2rank/releases and extract to tools/p2rank/"
        )

    cmd = [prank_bin, "predict", "-f", pdb_path, "-o", output_dir]
    logger.info("Running P2Rank: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"P2Rank failed (rc={result.returncode}):\n{result.stderr}"
        )
    logger.info("P2Rank finished. Output: %s", output_dir)
    return output_dir


def parse_p2rank_output(output_dir: str, pdb_stem: str) -> List[int]:
    """
    Parse P2Rank predictions CSV and return the residue numbers of the top pocket.

    P2Rank writes `{pdb_stem}.pdb_predictions.csv` and
    `{pdb_stem}.pdb_residues.csv` inside *output_dir*.
    We read the residues CSV and keep residues belonging to pocket rank 1.
    """
    residues_csv = os.path.join(
        output_dir, "visualizations", f"{pdb_stem}.pdb_residues.csv"
    )
    # Fallback path (older P2Rank versions)
    if not os.path.exists(residues_csv):
        residues_csv = os.path.join(output_dir, f"{pdb_stem}.pdb_residues.csv")

    if not os.path.exists(residues_csv):
        raise FileNotFoundError(
            f"P2Rank residues CSV not found: {residues_csv}"
        )

    pocket_residues: List[int] = []
    with open(residues_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # P2Rank CSV has leading spaces in column names — strip them
            stripped = {k.strip(): v.strip() for k, v in row.items()}
            # 'pocket' column: 0 = not in any pocket, 1 = top pocket, etc.
            pocket_num = stripped.get("pocket", "0")
            if pocket_num == "1":
                # residue_label is a plain integer residue number (e.g. "42")
                res_num_str = stripped.get("residue_label", "")
                try:
                    pocket_residues.append(int(res_num_str))
                except ValueError:
                    logger.warning("Could not parse residue number from '%s'", res_num_str)

    pocket_residues = sorted(set(pocket_residues))
    logger.info("Top pocket residues: %s", pocket_residues)
    return pocket_residues


def residues_to_contig(
    fixed_residues: List[int],
    total_length: int,
    chain: str = "A",
) -> str:
    """
    Convert a list of fixed residue numbers to an RFdiffusion contig string.

    Example: fixed=[5,6,7,20], total=100 →
      '[1-4/A5-7/8-19/A20/21-100]'

    The returned string is the value for `contigmap.contigs`, already
    wrapped in square brackets.
    """
    if not fixed_residues:
        return f"[{total_length}-{total_length}]"

    fixed_set = set(fixed_residues)
    segments: List[str] = []
    i = 1
    while i <= total_length:
        if i in fixed_set:
            # Extend to the end of the contiguous run of fixed residues
            j = i
            while j + 1 <= total_length and j + 1 in fixed_set:
                j += 1
            segments.append(f"{chain}{i}-{j}")
            i = j + 1
        else:
            # Extend the free (diffused) segment
            j = i
            while j + 1 <= total_length and j + 1 not in fixed_set:
                j += 1
            segments.append(f"{j - i + 1}-{j - i + 1}")
            i = j + 1

    return "[" + "/".join(segments) + "]"
