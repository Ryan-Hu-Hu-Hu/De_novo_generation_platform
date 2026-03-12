"""Test SMILES extraction strategies for a few known PDB entries."""

import sys
import os
import tempfile
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.pdb_utils import get_ligand_smiles, _ligand_codes_from_pdb, download_pdb

# 1HEB = carbonic anhydrase; no small-molecule ligand in crystal → needs UniProt fallback
# 1PMO = Pseudomonas mendocina lipase; has a ligand
# 1A0T = has inhibitor EI1
TEST_CASES = [
    ("1PMO", "has ligand"),
    ("1A0T", "has inhibitor"),
    ("1HEB", "no crystal ligand → fallback"),
]

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        for pdb_id, desc in TEST_CASES:
            print(f"\n{'─'*60}")
            print(f"Testing {pdb_id} ({desc})")
            pdb_path = download_pdb(pdb_id, tmpdir)
            codes = _ligand_codes_from_pdb(pdb_path)
            print(f"  HETATM ligand codes: {codes}")
            smiles = get_ligand_smiles(pdb_id, pdb_path=pdb_path)
            if smiles:
                print(f"  SMILES: {smiles[:80]}{'…' if len(smiles) > 80 else ''}")
            else:
                print(f"  SMILES: None (no ligand found)")

if __name__ == "__main__":
    main()
