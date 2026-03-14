"""Central configuration for the De Novo Protein Generation Platform."""

import os

# ── Root paths ────────────────────────────────────────────────────────────────
PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PLATFORM_ROOT, "data")

# ── Data subdirectories ───────────────────────────────────────────────────────
PDB_DIR                = os.path.join(DATA_DIR, "pdb")
RFDIFFUSION_RESULT_DIR = os.path.join(DATA_DIR, "RFdiffusion_result")
PROTEINMPNN_RESULT_DIR = os.path.join(DATA_DIR, "ProteinMPNN_result")
CLEAN_RESULT_DIR       = os.path.join(DATA_DIR, "CLEAN_result")
SOLUBILITY_RESULT_DIR  = os.path.join(DATA_DIR, "solubility_result")
UNIKP_RESULT_DIR       = os.path.join(DATA_DIR, "unikp_result")
SEQ2TOPT_RESULT_DIR    = os.path.join(DATA_DIR, "seq2topt_result")
FINAL_RESULT_DIR       = os.path.join(DATA_DIR, "final_result")
JSON_DIR               = os.path.join(DATA_DIR, "json")

ALL_DATA_DIRS = [
    DATA_DIR, PDB_DIR, RFDIFFUSION_RESULT_DIR, PROTEINMPNN_RESULT_DIR,
    CLEAN_RESULT_DIR, SOLUBILITY_RESULT_DIR, UNIKP_RESULT_DIR,
    SEQ2TOPT_RESULT_DIR, FINAL_RESULT_DIR, JSON_DIR,
]

# ── Tool paths ────────────────────────────────────────────────────────────────
RFDIFFUSION_PATH  = os.path.join(PLATFORM_ROOT, "RFdiffusion")
PROTEINMPNN_PATH  = os.path.join(PLATFORM_ROOT, "ProteinMPNN")
CLEAN_APP_PATH    = os.path.join(PLATFORM_ROOT, "CLEAN", "app")
SODOPE_PATH       = os.path.join(PLATFORM_ROOT, "SoDoPE_paper_2020", "SWI")
UNIKP_PATH        = os.path.join(PLATFORM_ROOT, "UniKP")
SEQ2TOPT_PATH     = os.path.join(PLATFORM_ROOT, "tools", "Seq2Topt", "code")
P2RANK_PATH       = os.path.join(PLATFORM_ROOT, "tools", "p2rank")

# ── Conda environment names ───────────────────────────────────────────────────
# RFDIFFUSION_ENV = "SE3nv"
# CLEAN_ENV       = "cu117py308"
RFDIFFUSION_ENV = "lin"
CLEAN_ENV       = "lin"
UNIKP_ENV       = "Uni_test"
SEQ2TOPT_ENV    = "lin"
# ProteinMPNN and SoDoPE run in the current 'lin' environment

# ── Pipeline parameters ───────────────────────────────────────────────────────
NUM_DESIGNS        = 2  # RFdiffusion: number of backbone designs per iteration
NUM_SEQ_PER_TARGET = 2   # ProteinMPNN: sequences per backbone
MAX_ITERATIONS     = 1   # Max generation/evaluation loops before giving up

# ── Generation length constraint ─────────────────────────────────────────────
# Caps the diffused backbone to at most this many residues. Keeping this small
# (e.g. 50) reduces GPU memory and wall-clock time while other programs are
# running. Set to None to disable capping.
# MAX_GENERATED_LENGTH = 50   # amino acids
MAX_GENERATED_LENGTH = None   # amino acids

# ── Chatbot / server ──────────────────────────────────────────────────────────
FLASK_PORT   = 5000
JOBS_DB_PATH = os.path.join(PLATFORM_ROOT, "jobs.db")
