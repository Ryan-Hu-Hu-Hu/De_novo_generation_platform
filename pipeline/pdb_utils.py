"""PDB download utilities and RCSB REST API helpers."""

import os
import logging
import requests
from typing import Optional, List

logger = logging.getLogger(__name__)

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_ENTRY_URL    = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_CHEM_URL     = "https://data.rcsb.org/rest/v1/core/chemcomp/{chem_comp_id}"

# Residue codes to skip when looking for ligands (water, buffers, ions, phasing agents)
_SKIP_HETATM = {
    "HOH", "WAT", "H2O", "DOD",                    # water
    "SO4", "PO4", "EDO", "GOL", "PEG", "MPD",      # buffers/cryo
    "TRS", "MES", "HEPES", "TRIS", "IMD",           # buffer molecules
    "CL",  "NA",  "MG",  "ZN",  "CA",  "FE",       # common ions
    "MN",  "CU",  "K",   "IOD", "BR",  "F",        # ions/halides
    "HG",  "AU",  "PT",  "PB",  "SE",  "XE",       # heavy-atom phasing
    "ACE", "NH2", "MSE", "UNX", "UNL",             # modified AA / unknown
}


def download_pdb(pdb_id: str, out_dir: str) -> str:
    """Download a PDB file from RCSB and return the local file path."""
    os.makedirs(out_dir, exist_ok=True)
    pdb_id = pdb_id.upper()
    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")

    if os.path.exists(out_path):
        logger.info("PDB %s already downloaded at %s", pdb_id, out_path)
        return out_path

    logger.info("Downloading PDB %s from %s", pdb_id, url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(out_path, "w") as fh:
        fh.write(resp.text)
    logger.info("Saved PDB to %s", out_path)
    return out_path


def get_template_sequence(pdb_path: str, chain_id: str = "A") -> str:
    """Extract the FASTA sequence of a given chain from a PDB file using BioPython."""
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import PPBuilder

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", pdb_path)
    ppb = PPBuilder()
    sequences = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for pp in ppb.build_peptides(chain):
                    sequences.append(str(pp.get_sequence()))
    return "".join(sequences)


def validate_pdb_id(pdb_id: str) -> bool:
    """Return True if the PDB ID exists in RCSB."""
    url = RCSB_ENTRY_URL.format(pdb_id=pdb_id.upper())
    try:
        resp = requests.get(url, timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _pubchem_smiles_from_props(props: dict) -> Optional[str]:
    """Extract SMILES from a PubChem PropertyTable Properties entry (handles varying key names)."""
    return (
        props.get("CanonicalSMILES")
        or props.get("ConnectivitySMILES")
        or props.get("IsomericSMILES")
        or props.get("SMILES")
    )


def _smiles_from_uniprot_pubchem(pdb_id: str) -> Optional[str]:
    """
    Fallback SMILES lookup: PDB → UniProt → ChEBI IDs from catalytic activity → PubChem SMILES.
    """
    # Trivial ChEBI IDs to skip (water, proton, O2, etc.)
    _TRIVIAL_CHEBI = {
        "CHEBI:15377",  # water
        "CHEBI:15378",  # H+
        "CHEBI:57945",  # H+ (another form)
        "CHEBI:15379",  # O2
        "CHEBI:16240",  # H2O2
        "CHEBI:30616",  # ATP (too generic)
        "CHEBI:456216", # ADP
    }

    # Step 1: get UniProt accession from RCSB
    try:
        up_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
        resp = requests.get(up_url, timeout=15)
        if resp.status_code != 200:
            return None
        uniprot_ids = (
            resp.json()
            .get("rcsb_polymer_entity_container_identifiers", {})
            .get("uniprot_ids", [])
        )
        if not uniprot_ids:
            return None
        uniprot_id = uniprot_ids[0]
    except (requests.RequestException, ValueError):
        return None

    # Step 2: collect ChEBI IDs and a fallback substrate name from UniProt
    try:
        up_resp = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json", timeout=15
        )
        if up_resp.status_code != 200:
            return None
        up_data = up_resp.json()
        chebi_ids: list = []
        fallback_name: Optional[str] = None
        for comment in up_data.get("comments", []):
            if comment.get("commentType") == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction", {})
                for xref in reaction.get("reactionCrossReferences", []):
                    if xref.get("database") == "ChEBI":
                        cid = xref.get("id", "")
                        if cid and cid not in _TRIVIAL_CHEBI:
                            chebi_ids.append(cid)
                # Fallback: first reactant from reaction name string
                if not fallback_name:
                    rname = reaction.get("name", "")
                    if rname:
                        fallback_name = rname.split("=")[0].split("+")[0].strip()
    except (requests.RequestException, ValueError):
        return None

    # Step 3a: try each ChEBI ID via PubChem xref lookup
    for chebi_id in chebi_ids[:6]:
        try:
            pc_resp = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/"
                f"{chebi_id}/property/CanonicalSMILES/JSON",
                timeout=15,
            )
            if pc_resp.status_code == 200:
                props = pc_resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                smiles = _pubchem_smiles_from_props(props)
                if smiles:
                    logger.info(
                        "Fallback SMILES for %s via UniProt(%s)/ChEBI(%s): %s",
                        pdb_id, uniprot_id, chebi_id, smiles[:60],
                    )
                    return smiles
        except (requests.RequestException, ValueError):
            continue

    # Step 3b: fallback — name search in PubChem
    if fallback_name:
        try:
            pc_resp = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{requests.utils.quote(fallback_name)}/property/CanonicalSMILES/JSON",
                timeout=15,
            )
            if pc_resp.status_code == 200:
                props = pc_resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                smiles = _pubchem_smiles_from_props(props)
                if smiles:
                    logger.info(
                        "Fallback SMILES for %s via UniProt(%s)/name('%s'): %s",
                        pdb_id, uniprot_id, fallback_name, smiles[:60],
                    )
                    return smiles
        except (requests.RequestException, ValueError):
            pass

    logger.info("UniProt/PubChem fallback found no SMILES for %s (UniProt: %s)", pdb_id, uniprot_id)
    return None


def _ligand_codes_from_pdb(pdb_path: str) -> List[str]:
    """
    Parse HETATM records to collect unique non-solvent ligand 3-letter codes,
    preserving first-occurrence order.
    """
    seen: dict = {}
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith("HETATM"):
                    continue
                res_name = line[17:20].strip().upper()
                if res_name and res_name not in _SKIP_HETATM and res_name not in seen:
                    seen[res_name] = True
    except OSError:
        pass
    return list(seen.keys())


def _smiles_from_chemcomp(chem_comp_id: str) -> Optional[str]:
    """Query RCSB chemical component dictionary for canonical SMILES."""
    url = RCSB_CHEM_URL.format(chem_comp_id=chem_comp_id)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        smiles = (
            data.get("rcsb_chem_comp_descriptor", {}).get("smiles")
            or data.get("chem_comp", {}).get("pdbx_smiles_canonical")
        )
        if smiles:
            logger.info("SMILES for %s from chemcomp: %s", chem_comp_id, smiles[:80])
        return smiles
    except (requests.RequestException, ValueError) as exc:
        logger.warning("chemcomp lookup failed for %s: %s", chem_comp_id, exc)
        return None


def get_ligand_smiles(pdb_id: str, pdb_path: Optional[str] = None) -> Optional[str]:
    """
    Retrieve the canonical SMILES of the primary ligand for a PDB entry.

    Strategy (in order):
    1. Parse HETATM records from the local PDB file (most reliable).
    2. Query RCSB nonpolymer_entity API.
    3. Fallback via UniProt catalytic activity → PubChem.

    Returns None if no ligand information is available.
    """
    pdb_id = pdb_id.upper()

    # ── Strategy 1: parse HETATM records from local file ─────────────────────
    if pdb_path and os.path.exists(pdb_path):
        codes = _ligand_codes_from_pdb(pdb_path)
        if codes:
            logger.info("Ligand codes found in PDB file %s: %s", pdb_path, codes)
            for code in codes:
                smiles = _smiles_from_chemcomp(code)
                if smiles:
                    return smiles
            logger.warning("chemcomp lookup returned no SMILES for codes %s", codes)

    # ── Strategy 2: RCSB nonpolymer_entity API ────────────────────────────────
    try:
        resp = requests.get(
            f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/1",
            timeout=15,
        )
        if resp.status_code == 200:
            np_data = resp.json()
            chem_comp_id = (
                np_data.get("pdbx_entity_nonpoly", {}).get("comp_id")
                or np_data.get("nonpolymer_comp", {}).get("chem_comp", {}).get("id")
            )
            if chem_comp_id and chem_comp_id.upper() not in _SKIP_HETATM:
                smiles = _smiles_from_chemcomp(chem_comp_id)
                if smiles:
                    return smiles
    except requests.RequestException as exc:
        logger.warning("RCSB nonpolymer_entity lookup failed for %s: %s", pdb_id, exc)

    # ── Strategy 3: UniProt catalytic activity → PubChem ─────────────────────
    logger.info("Trying UniProt/PubChem fallback for %s…", pdb_id)
    return _smiles_from_uniprot_pubchem(pdb_id)
