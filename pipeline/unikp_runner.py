"""UniKP kinetics prediction runner."""

import os
import csv
import logging
import subprocess
from typing import Dict, Optional

from .config import UNIKP_PATH, UNIKP_ENV

logger = logging.getLogger(__name__)

_PREDICT_SCRIPT = os.path.join(UNIKP_PATH, "unikp_predict.py")


def run_unikp(
    sequences: Dict[str, str],
    smiles: str,
    output_dir: str,
) -> str:
    """
    Write an input TSV, call unikp_predict.py in the Uni_test conda env,
    and return the path to the output CSV.

    Parameters
    ----------
    sequences : dict
        Mapping of ``seq_name → amino_acid_sequence``.
    smiles : str
        Canonical SMILES of the substrate ligand.
    output_dir : str
        Directory where the input/output files will be written.

    Returns
    -------
    str
        Path to the output CSV produced by unikp_predict.py.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_tsv = os.path.join(output_dir, "unikp_input.tsv")
    output_csv = os.path.join(output_dir, "unikp_output.csv")

    with open(input_tsv, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["name", "sequence", "smiles"])
        for name, seq in sequences.items():
            writer.writerow([name, seq, smiles])

    cmd = [
        "conda", "run", "--no-capture-output", "-n", UNIKP_ENV,
        "python", _PREDICT_SCRIPT,
        "--input", input_tsv,
        "--output", output_csv,
    ]
    logger.info("Running UniKP: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=UNIKP_PATH
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"UniKP failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    logger.info("UniKP finished. Results at %s", output_csv)
    return output_csv


def parse_unikp_results(output_csv: str) -> Dict[str, Dict[str, float]]:
    """
    Parse unikp_predict.py output CSV.

    Returns ``{seq_name: {kcat, Km, kcat_Km}}``.
    """
    results: Dict[str, Dict[str, float]] = {}
    with open(output_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row["name"].strip()
            results[name] = {
                "kcat":    float(row.get("kcat", 0) or 0),
                "Km":      float(row.get("Km", 0) or 0),
                "kcat_Km": float(row.get("kcat_Km", 0) or 0),
            }
    logger.info("Parsed UniKP results for %d sequences", len(results))
    return results


def filter_by_kinetics(
    sequences: Dict[str, str],
    kinetics: Dict[str, Dict[str, float]],
    template_kinetics: Dict[str, float],
) -> Dict[str, str]:
    """
    Keep sequences with kcat >= template AND Km <= template AND kcat/Km >= template.

    If a criterion value is None/0 it is skipped.
    """
    t_kcat    = template_kinetics.get("kcat", 0)
    t_km      = template_kinetics.get("Km", float("inf"))
    t_kcat_km = template_kinetics.get("kcat_Km", 0)

    kept: Dict[str, str] = {}
    for name, seq in sequences.items():
        k = kinetics.get(name, {})
        kcat    = k.get("kcat", 0)
        km      = k.get("Km", float("inf"))
        kcat_km = k.get("kcat_Km", 0)

        if t_kcat > 0 and kcat < t_kcat:
            logger.debug("Kinetics filter removed %s (kcat %.4f < %.4f)", name, kcat, t_kcat)
            continue
        if t_km < float("inf") and km > t_km:
            logger.debug("Kinetics filter removed %s (Km %.4f > %.4f)", name, km, t_km)
            continue
        if t_kcat_km > 0 and kcat_km < t_kcat_km:
            logger.debug(
                "Kinetics filter removed %s (kcat/Km %.4f < %.4f)",
                name, kcat_km, t_kcat_km,
            )
            continue
        kept[name] = seq

    logger.info("Kinetics filter: %d/%d sequences passed", len(kept), len(sequences))
    return kept
