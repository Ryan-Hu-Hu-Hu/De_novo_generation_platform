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


def cluster_into_islands(
    residues: List[int],
    total_length: int,
    max_islands: int = 2,
    padding: int = 5,
) -> List[Tuple[int, int]]:
    """
    Consolidate all active-site residues into at most *max_islands* contiguous
    ranges, guaranteeing every residue is covered.

    Algorithm (for max_islands=2):
    1. Sort residues.
    2. Find the largest gap between consecutive residues — this is the natural
       split point that separates the two motif regions.
    3. Island 1: [first_residue, last_residue_before_gap]
       Island 2: [first_residue_after_gap, last_residue]
    4. Expand each island by *padding* residues on each side so the surrounding
       backbone context is included; clamp to [1, total_length].

    If all residues are contiguous (no gap found) or only one island is requested,
    a single island spanning all residues is returned.

    Example: residues=[63,125,126,127,163,165,210,212,...,317,318,465,466],
             total_length=470, padding=5
      → largest gap: 318→465 (size 147)
      → raw islands: [(63,318), (465,466)]
      → after padding: [(58,323), (460,470)]  (clamped)
    """
    if not residues:
        return []

    sorted_res = sorted(set(residues))

    if max_islands == 1 or len(sorted_res) == 1:
        start = max(1, sorted_res[0] - padding)
        end   = min(total_length, sorted_res[-1] + padding)
        islands = [(start, end)]
        logger.info("Motif island: %s ← %d residues", islands, len(sorted_res))
        return islands

    # Find the largest gap between consecutive residues
    max_gap = -1
    split_idx = 0  # index of the last residue in group 1
    for i in range(len(sorted_res) - 1):
        gap = sorted_res[i + 1] - sorted_res[i]
        if gap > max_gap:
            max_gap = gap
            split_idx = i

    group1 = sorted_res[: split_idx + 1]
    group2 = sorted_res[split_idx + 1 :]

    island1 = (
        max(1, group1[0] - padding),
        min(total_length, group1[-1] + padding),
    )
    island2 = (
        max(1, group2[0] - padding),
        min(total_length, group2[-1] + padding),
    )

    # Merge overlapping/touching islands (shouldn't happen with active sites,
    # but guard against it)
    if island1[1] >= island2[0]:
        merged = (island1[0], max(island1[1], island2[1]))
        logger.info("Motif islands merged (overlap): %s ← %d residues", [merged], len(sorted_res))
        return [merged]

    islands = [island1, island2]
    logger.info(
        "Motif islands (split at gap %d, padding %d): %s ← %d residues",
        max_gap, padding, islands, len(sorted_res),
    )
    return islands


def split_islands_at_gaps(
    islands: List[Tuple[int, int]],
    pdb_residues: set,
) -> List[Tuple[int, int]]:
    """
    Validate island ranges against the residues that physically exist in the PDB.

    For each island (start, end):
    - Keep only residues within [start, end] that are present in *pdb_residues*.
    - If missing residues create internal gaps, split the island into separate
      contiguous fragments (each fragment is a valid contig segment).

    Returns a list of (start, end) tuples — all guaranteed to contain only
    residues present in *pdb_residues*.  Empty islands are dropped.
    """
    if not pdb_residues:
        return islands

    fragments: List[Tuple[int, int]] = []
    for start, end in islands:
        present = sorted(r for r in range(start, end + 1) if r in pdb_residues)
        if not present:
            logger.warning("Island (%d, %d) has no present residues in PDB — skipping.", start, end)
            continue

        # Group into contiguous runs
        run_s = run_e = present[0]
        for r in present[1:]:
            if r == run_e + 1:
                run_e = r
            else:
                fragments.append((run_s, run_e))
                run_s = run_e = r
        fragments.append((run_s, run_e))

    if len(fragments) != len(islands):
        logger.info(
            "After PDB gap validation: %d islands → %d fragments: %s",
            len(islands), len(fragments), fragments,
        )
    return fragments


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


def islands_to_contig(
    islands: List[Tuple[int, int]],
    total_length: int,
    chain: str = "A",
) -> str:
    """
    Build an RFdiffusion contig string from a list of (start, end) island ranges.

    Free (diffused) segments between/around islands keep their exact length so
    the total scaffold length matches *total_length*.

    Example: islands=[(5,10),(40,50)], total=100 →
      '[4-4/A5-10/29-29/A40-50/50-50]'
    """
    if not islands:
        return f"[{total_length}-{total_length}]"

    segments: List[str] = []
    prev_end = 0
    for start, end in islands:
        gap = start - prev_end - 1
        if gap > 0:
            segments.append(f"{gap}-{gap}")
        segments.append(f"{chain}{start}-{end}")
        prev_end = end

    remaining = total_length - prev_end
    if remaining > 0:
        segments.append(f"{remaining}-{remaining}")

    return "[" + "/".join(segments) + "]"


def islands_to_fixed_residues(islands: List[Tuple[int, int]]) -> List[int]:
    """Expand (start, end) island ranges to a flat sorted list of all residue numbers."""
    residues: List[int] = []
    for start, end in islands:
        residues.extend(range(start, end + 1))
    return residues
