"""Seq2Topt optimal temperature prediction runner."""

import os
import csv
import logging
import subprocess
from typing import Dict, Optional

from .config import SEQ2TOPT_PATH, SEQ2TOPT_ENV

logger = logging.getLogger(__name__)


def _write_csv(sequences: Dict[str, str], csv_path: str) -> None:
    """Write sequences as CSV with 'sequence' column (required by seq2topt.py)."""
    import csv as csv_mod
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv_mod.writer(fh)
        writer.writerow(["sequence"])
        for seq in sequences.values():
            writer.writerow([seq])


def run_seq2topt(sequences: Dict[str, str], output_dir: str) -> str:
    """
    Run Seq2Topt to predict optimal temperature for each sequence.

    Returns path to the output CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_csv  = os.path.join(output_dir, "seq2topt_input.csv")
    # seq2topt.py appends '.csv' to the --output argument
    output_stem = os.path.join(output_dir, "seq2topt_output")
    output_csv  = output_stem + ".csv"
    _write_csv(sequences, input_csv)

    seq2topt_script = os.path.join(SEQ2TOPT_PATH, "seq2topt.py")
    if not os.path.exists(seq2topt_script):
        raise FileNotFoundError(
            f"Seq2Topt script not found at {seq2topt_script}. "
            "Clone https://github.com/SizheQiu/Seq2Topt into tools/Seq2Topt/"
        )

    cmd = [
        "conda", "run", "--no-capture-output", "-n", SEQ2TOPT_ENV,
        "python", seq2topt_script,
        "--input", input_csv,
        "--output", output_stem,
    ]
    logger.info("Running Seq2Topt: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=SEQ2TOPT_PATH
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Seq2Topt failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    logger.info("Seq2Topt finished. Results at %s", output_csv)
    return output_csv


def parse_seq2topt_results(output_csv: str, seq_names: Optional[list] = None) -> Dict[str, float]:
    """
    Parse Seq2Topt output CSV.

    seq2topt.py uses integer row indices as 'id'; if *seq_names* is provided
    the predictions are mapped back to the original sequence names by position.

    Returns ``{seq_name: predicted_topt}``.
    """
    results: Dict[str, float] = {}
    rows = []
    with open(output_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        topt_col = next(
            (c for c in ("pred_topt", "topt", "Topt", "temperature", "predicted_topt")
             if c in fieldnames),
            fieldnames[-1],
        )
        for row in reader:
            try:
                topt = float(row[topt_col])
            except (ValueError, KeyError):
                topt = 0.0
            rows.append(topt)

    if seq_names and len(seq_names) == len(rows):
        for name, topt in zip(seq_names, rows):
            results[name] = topt
    else:
        for i, topt in enumerate(rows):
            results[str(i)] = topt

    logger.info("Seq2Topt parsed %d predictions", len(results))
    return results


def select_by_temperature(
    sequences: Dict[str, str],
    topt_predictions: Dict[str, float],
    target_temp: float,
) -> Optional[Dict]:
    """
    Return the candidate sequence closest to *target_temp*.

    Returns a dict with keys: name, sequence, topt — or None if empty.
    """
    if not sequences:
        return None

    best_name = min(
        sequences.keys(),
        key=lambda n: abs(topt_predictions.get(n, float("inf")) - target_temp),
    )
    return {
        "name":     best_name,
        "sequence": sequences[best_name],
        "topt":     topt_predictions.get(best_name, None),
    }
