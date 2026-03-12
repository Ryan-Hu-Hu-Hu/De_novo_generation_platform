"""CLEAN EC-number prediction runner."""

import os
import shutil
import logging
import subprocess
from typing import Dict

from .config import CLEAN_APP_PATH, CLEAN_ENV

logger = logging.getLogger(__name__)


def prepare_clean_input(
    sequences_dict: Dict[str, str],
    csv_out: str,
    fasta_out: str,
) -> None:
    """
    Write sequences to a TSV (CSV with tab separator) and a FASTA file.

    *sequences_dict* maps ``seq_name → amino_acid_sequence``.
    The TSV format expected by CLEAN is: Entry\\tEC_number\\tSequence
    (with a placeholder EC number of '1.1.1.1').
    """
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(fasta_out) or ".", exist_ok=True)

    with open(csv_out, "w") as csv_fh, open(fasta_out, "w") as fa_fh:
        csv_fh.write("Entry\tEC_number\tSequence\n")
        for name, seq in sequences_dict.items():
            csv_fh.write(f"{name}\t1.1.1.1\t{seq}\n")
            fa_fh.write(f">{name}\n{seq}\n")

    logger.info("Wrote %d sequences to %s and %s", len(sequences_dict), csv_out, fasta_out)


def run_clean(csv_path: str, results_dir: str) -> str:
    """
    Copy input files to the CLEAN app directory, run inference, and copy results back.

    Returns the path to the results CSV produced by CLEAN.
    """
    os.makedirs(results_dir, exist_ok=True)

    # CLEAN expects its data in <CLEAN_APP_PATH>/data/
    clean_data_dir = os.path.join(CLEAN_APP_PATH, "data")
    os.makedirs(clean_data_dir, exist_ok=True)

    # Derive base name for CLEAN (must match what CLEAN_execute.py uses)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    dest_csv = os.path.join(clean_data_dir, f"{base_name}.csv")
    shutil.copy2(csv_path, dest_csv)

    # Also copy FASTA if it exists next to the CSV
    fasta_path = os.path.splitext(csv_path)[0] + ".fasta"
    if os.path.exists(fasta_path):
        shutil.copy2(fasta_path, os.path.join(clean_data_dir, f"{base_name}.fasta"))

    # Write a temporary execute script tailored to our base_name
    exec_script = os.path.join(CLEAN_APP_PATH, "_pipeline_clean_exec.py")
    with open(exec_script, "w") as fh:
        fh.write(f"""\
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from CLEAN.utils import *

csv_to_fasta("./data/{base_name}.csv", "./data/{base_name}.fasta")
retrive_esm1b_embedding("{base_name}")

from CLEAN.infer import infer_maxsep
train_data = "split100"
test_data = "{base_name}"
infer_maxsep(train_data, test_data, report_metrics=False, pretrained=True)
""")

    cmd = [
        "conda", "run", "--no-capture-output", "-n", CLEAN_ENV,
        "python", "_pipeline_clean_exec.py",
    ]
    logger.info("Running CLEAN: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=CLEAN_APP_PATH)
    if result.returncode != 0:
        raise RuntimeError(
            f"CLEAN failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Copy results back
    clean_results_src = os.path.join(CLEAN_APP_PATH, "results", f"{base_name}_maxsep.csv")
    if not os.path.exists(clean_results_src):
        # Try alternate naming
        import glob
        candidates = glob.glob(os.path.join(CLEAN_APP_PATH, "results", f"{base_name}*.csv"))
        clean_results_src = candidates[0] if candidates else clean_results_src

    dest_results = os.path.join(results_dir, f"{base_name}_maxsep.csv")
    if os.path.exists(clean_results_src):
        shutil.copy2(clean_results_src, dest_results)
    else:
        raise FileNotFoundError(f"CLEAN results not found at {clean_results_src}")

    logger.info("CLEAN results saved to %s", dest_results)
    return dest_results


def parse_clean_results(results_csv: str) -> Dict[str, str]:
    """
    Parse CLEAN maxsep results CSV.

    CLEAN writes headerless lines in the format:
        {seq_id},EC:{ec_number}/{distance}[,EC:{ec2}/{dist2},...]

    Returns a dict mapping ``seq_name → predicted_EC_number`` (best = lowest distance).
    """
    predictions: Dict[str, str] = {}
    with open(results_csv) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            seq_name = parts[0].strip()
            best_ec, best_dist = None, float("inf")
            for part in parts[1:]:
                part = part.strip()
                if part.startswith("EC:"):
                    try:
                        ec, dist = part[3:].split("/")
                        if float(dist) < best_dist:
                            best_dist = float(dist)
                            best_ec = ec
                    except ValueError:
                        pass
            if seq_name and best_ec:
                predictions[seq_name] = best_ec

    logger.info("Parsed %d EC predictions from %s", len(predictions), results_csv)
    return predictions


def filter_by_ec(
    sequences: Dict[str, str],
    ec_predictions: Dict[str, str],
    template_ec: str,
) -> Dict[str, str]:
    """
    Keep only sequences whose predicted EC matches *template_ec* at
    the 3rd-level (e.g., '1.2.3') or full 4th-level.
    """
    template_top3 = ".".join(template_ec.split(".")[:3])
    kept: Dict[str, str] = {}
    for name, seq in sequences.items():
        predicted = ec_predictions.get(name, "")
        predicted_top3 = ".".join(predicted.split(".")[:3])
        if predicted_top3 == template_top3:
            kept[name] = seq
        else:
            logger.debug("EC filter removed %s (%s != %s)", name, predicted, template_ec)

    logger.info("EC filter: %d/%d sequences passed", len(kept), len(sequences))
    return kept
