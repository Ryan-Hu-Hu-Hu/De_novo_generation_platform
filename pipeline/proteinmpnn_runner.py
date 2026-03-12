"""ProteinMPNN sequence design runner.

Implements the 4-step ProteinMPNN workflow:
  1. parse_multiple_chains
  2. assign_fixed_chains
  3. make_fixed_positions_dict  (placeholder "1")
  4. update fixed-position JSONL with real active-site residues
  5. protein_mpnn_run
"""

import os
import json
import logging
import subprocess
from typing import Dict, List, Optional

from .config import PROTEINMPNN_PATH, NUM_SEQ_PER_TARGET

logger = logging.getLogger(__name__)

HELPER = os.path.join(PROTEINMPNN_PATH, "helper_scripts")


def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    logger.info("CMD: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def parse_chains(pdb_dir: str, jsonl_out: str) -> str:
    """Run parse_multiple_chains.py and return the output JSONL path."""
    os.makedirs(os.path.dirname(jsonl_out) or ".", exist_ok=True)
    _run([
        "python",
        os.path.join(HELPER, "parse_multiple_chains.py"),
        "--input_path", pdb_dir,
        "--output_path", jsonl_out,
    ])
    return jsonl_out


def assign_chains(jsonl: str, assigned_jsonl: str, chain_list: str = "A") -> str:
    """Run assign_fixed_chains.py and return the assigned JSONL path."""
    _run([
        "python",
        os.path.join(HELPER, "assign_fixed_chains.py"),
        "--input_path", jsonl,
        "--output_path", assigned_jsonl,
        "--chain_list", chain_list,
    ])
    return assigned_jsonl


def make_fixed_positions(
    jsonl: str,
    fixed_pos_jsonl: str,
    chain_list: str = "A",
    position_list: str = "1",
) -> str:
    """Create fixed-positions JSONL with a placeholder position list."""
    _run([
        "python",
        os.path.join(HELPER, "make_fixed_positions_dict.py"),
        "--input_path", jsonl,
        "--output_path", fixed_pos_jsonl,
        "--chain_list", chain_list,
        "--position_list", position_list,
    ])
    return fixed_pos_jsonl


def update_fixed_positions(
    fixed_pos_jsonl: str,
    updated_jsonl: str,
    fixed_residues_map: Dict[str, List[int]],
) -> str:
    """
    Replace placeholder residue positions in the fixed-positions JSONL.

    *fixed_residues_map* maps design name → list of integer residue positions
    on chain A.  This replicates the logic of Helper_script/Fixposition_replacer.py
    but operates on in-memory data so we can pass per-design residue lists.
    """
    with open(fixed_pos_jsonl) as infile, open(updated_jsonl, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            for design_name in list(data.keys()):
                if design_name in fixed_residues_map:
                    data[design_name]["A"] = fixed_residues_map[design_name]
                else:
                    # If no specific mapping, keep placeholder but convert to int list
                    current = data[design_name].get("A", [])
                    if isinstance(current, list):
                        data[design_name]["A"] = [int(x) for x in current if str(x).isdigit()]
            outfile.write(json.dumps(data) + "\n")
    return updated_jsonl


def run_proteinmpnn(
    jsonl: str,
    assigned_jsonl: str,
    fixed_pos_jsonl: str,
    out_folder: str,
    num_seq: int = NUM_SEQ_PER_TARGET,
) -> Dict[str, List[str]]:
    """
    Run protein_mpnn_run.py and return a dict mapping design name → [sequences].

    The output FASTA files are parsed to extract sequences.
    """
    os.makedirs(out_folder, exist_ok=True)
    _run([
        "python",
        os.path.join(PROTEINMPNN_PATH, "protein_mpnn_run.py"),
        "--jsonl_path", jsonl,
        "--chain_id_jsonl", assigned_jsonl,
        "--fixed_positions_jsonl", fixed_pos_jsonl,
        "--out_folder", out_folder,
        "--num_seq_per_target", str(num_seq),
        "--sampling_temp", "0.1",
        "--seed", "0",
        "--batch_size", "1",
        "--save_score", "0",
        "--save_probs", "0",
    ])

    # Parse generated FASTA files in out_folder/seqs/
    seqs_dir = os.path.join(out_folder, "seqs")
    results: Dict[str, List[str]] = {}
    if not os.path.isdir(seqs_dir):
        logger.warning("ProteinMPNN seqs directory not found: %s", seqs_dir)
        return results

    for fa_file in os.listdir(seqs_dir):
        if not fa_file.endswith(".fa"):
            continue
        design_name = os.path.splitext(fa_file)[0]
        sequences: List[str] = []
        current_seq_lines: List[str] = []
        with open(os.path.join(seqs_dir, fa_file)) as fh:
            first = True
            for line in fh:
                line = line.rstrip()
                if line.startswith(">"):
                    if current_seq_lines:
                        seq = "".join(current_seq_lines)
                        if first:
                            first = False  # skip the template (first entry)
                        else:
                            sequences.append(seq)
                    current_seq_lines = []
                    first = False if not first else True
                else:
                    current_seq_lines.append(line)
            # Last entry
            if current_seq_lines:
                sequences.append("".join(current_seq_lines))
        results[design_name] = sequences

    total = sum(len(v) for v in results.values())
    logger.info("ProteinMPNN produced %d sequences across %d designs", total, len(results))
    return results
