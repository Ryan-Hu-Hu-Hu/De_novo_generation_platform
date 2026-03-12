"""RFdiffusion backbone generation runner."""

import os
import glob
import logging
import subprocess
from typing import List, Optional

from .config import RFDIFFUSION_PATH, RFDIFFUSION_ENV, NUM_DESIGNS

logger = logging.getLogger(__name__)


def run_rfdiffusion(
    pdb_path: str,
    contig_str: str,
    output_prefix: str,
    num_designs: int = NUM_DESIGNS,
    job_dir: Optional[str] = None,
) -> List[str]:
    """
    Run RFdiffusion to generate *num_designs* backbone PDB files.

    Parameters
    ----------
    pdb_path : str
        Input template PDB file path.
    contig_str : str
        Contig string (already bracketed), e.g. ``[1-4/A5-7/8-20]``.
    output_prefix : str
        Prefix for output files (directory + base name).
    num_designs : int
        Number of designs to generate.
    job_dir : str, optional
        Working directory for the subprocess.

    Returns
    -------
    List[str]
        Paths to the generated PDB files.
    """
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    inference_script = os.path.join(RFDIFFUSION_PATH, "scripts", "run_inference.py")
    active_site_ckpt = os.path.join(RFDIFFUSION_PATH, "models", "ActiveSite_ckpt.pt")

    cmd = [
        "conda", "run", "--no-capture-output", "-n", RFDIFFUSION_ENV,
        "python", inference_script,
        f"contigmap.contigs={contig_str}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
        f"inference.input_pdb={pdb_path}",
        f"inference.ckpt_override_path={active_site_ckpt}",
    ]

    logger.info("Running RFdiffusion: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=job_dir or RFDIFFUSION_PATH,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"RFdiffusion failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Collect generated PDB files matching the prefix pattern
    output_dir = os.path.dirname(output_prefix)
    base = os.path.basename(output_prefix)
    pattern = os.path.join(output_dir, f"{base}_*.pdb")
    generated = sorted(glob.glob(pattern))
    logger.info("RFdiffusion generated %d PDB files", len(generated))
    return generated
