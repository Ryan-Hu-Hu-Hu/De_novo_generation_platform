---
license: mit
tags:
  - protein-design
  - de-novo-generation
  - enzyme-engineering
  - rfdiffusion
  - proteinmpnn
  - chatbot
language:
  - en
pretty_name: De Novo Protein Generation Platform
---

# De Novo Protein Generation Platform

An automated pipeline for de novo enzyme design, integrated with a LINE chatbot frontend. Users send a PDB template code and a target reaction temperature via LINE; the platform generates novel protein sequences, evaluates them through a multi-stage screening pipeline, and returns the best candidate.

---

## Architecture Overview

```
LINE user
   │ (LINE Messaging API)
   ▼
Wix Velo webhook  (https://graceng-ncku.com/_functions/lineWebhook)
   │ validates signature + forwards
   ▼
Flask server  (local HPC, exposed via cloudflared tunnel)
   │
   ▼
Pipeline Orchestrator
   ├─ PDB download + SMILES extraction
   ├─ P2Rank      → active-site pocket prediction
   ├─ RFdiffusion → backbone generation
   ├─ ProteinMPNN → sequence design
   ├─ CLEAN       → EC number prediction  (filter)
   ├─ SoDoPE      → solubility prediction (filter)
   ├─ UniKP       → kinetics prediction — kcat, Km  (filter)
   └─ Seq2Topt    → optimal temperature prediction  (selection)
```

---

## Requirements

### System
- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU with CUDA 12+ (RTX 5090 recommended; ≥16 GB VRAM)
- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) — for exposing the local server via tunnel

### Conda environments

| Environment | Python | Used for | Environment file |
|---|---|---|---|
| `lin` | 3.9 | Flask chatbot, ProteinMPNN, SoDoPE, Seq2Topt, RFdiffusion | `envs/lin.yml` |
| `Uni_test` | 3.10 | UniKP kinetics prediction | `envs/Uni_test.yml` |

**Recommended — restore from the provided environment files (exact reproducibility):**
```bash
conda env create -f envs/lin.yml
conda env create -f envs/Uni_test.yml
```

**Alternative — create from scratch:**
```bash
conda create -n lin python=3.9 -y
conda activate lin
pip install -r requirements_chatbot.txt
```

For the UniKP environment see [`UniKP/README.md`](UniKP/README.md).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ryan-Hu-Hu-Hu/De_novo_generation_platform.git
cd De_novo_generation_platform
git submodule update --init --recursive
```

### 2. Download external tools

#### P2Rank (pocket prediction)
```bash
mkdir -p tools/p2rank
# Download the latest release from https://github.com/rdk/p2rank/releases
# Extract and place the binary at:  tools/p2rank/prank
chmod +x tools/p2rank/prank
```

#### Seq2Topt (temperature prediction)
```bash
git clone https://github.com/SizheQiu/Seq2Topt tools/Seq2Topt
mkdir -p tools/large_model_pth
# Download model weights from the Seq2Topt GitHub releases and place at:
#   tools/large_model_pth/model_topt_window=3_r2=0.57.pth
```

#### CLEAN pretrained weights
Download the following files from [Google Drive](https://drive.google.com/drive/folders/1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ) and place them into `CLEAN/app/`:
```
CLEAN/app/weights/split100.pth
CLEAN/app/weights/split70.pth
CLEAN/app/data/split100.csv
CLEAN/app/data/split70.csv
```
Also clone the ESM scripts required by CLEAN:
```bash
git clone https://github.com/facebookresearch/esm CLEAN/app/esm
```

### 3. Configure credentials

Create a `.env` file in the project root (**never commit this file — it is gitignored**):

```bash
# .env
LINE_CHANNEL_SECRET=your_line_channel_secret_here
LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token_here
```

Get these values from the [LINE Developers Console](https://developers.line.biz/console/).

### 4. Start the cloudflared tunnel

```bash
./cloudflared tunnel --url http://localhost:5000
# Note the generated HTTPS URL, e.g.: https://xxx.trycloudflare.com
```

Update `FLASK_BACKEND_URL` in your Wix `http-functions.js` to:
```
https://xxx.trycloudflare.com/callback
```

### 5. Run the platform

```bash
conda activate lin
python main.py
```

---

## LINE Chatbot Usage

1. Add the bot as a LINE friend (QR code from LINE Developers Console)
2. Send any message to start
3. Enter a 4-letter PDB code (e.g. `1HEB`)
4. Select a target reaction temperature (10–100 °C)
5. Wait for the pipeline — progress updates are sent automatically
6. Receive the best candidate sequence with EC number, solubility, and predicted Topt

**Commands available at any time:**

| Command | Action |
|---|---|
| `help` / `?` | Show help message |
| `cancel` / `reset` / `stop` | Abort current job and return to start |

---

## Wix Webhook Setup

The LINE Messaging API webhook URL should point to:
```
https://graceng-ncku.com/_functions/lineWebhook
```

The Wix `http-functions.js` validates the LINE HMAC-SHA256 signature and forwards valid requests to the cloudflared tunnel. See [`docs/LINE_setup_guide.md`](docs/LINE_setup_guide.md) for the full setup walkthrough.

---

## Project Structure

```
De_novo_generation_platform/
├── main.py                    # Entry point — Flask server + DB init
├── requirements_chatbot.txt   # Python dependencies
├── .env                       # Credentials (local only, gitignored)
│
├── pipeline/                  # Core pipeline package
│   ├── config.py              # All paths and tunable parameters
│   ├── orchestrator.py        # Full pipeline loop
│   ├── pdb_utils.py           # PDB download + SMILES extraction (3-strategy)
│   ├── active_site.py         # P2Rank pocket prediction
│   ├── rfdiffusion_runner.py  # RFdiffusion backbone generation
│   ├── proteinmpnn_runner.py  # ProteinMPNN sequence design
│   ├── clean_runner.py        # CLEAN EC number prediction
│   ├── sodope_runner.py       # SoDoPE solubility (SWI) prediction
│   ├── unikp_runner.py        # UniKP kinetics prediction
│   └── seq2topt_runner.py     # Seq2Topt optimal temperature prediction
│
├── chatbot/                   # LINE chatbot
│   ├── app.py                 # Flask /callback webhook endpoint
│   ├── line_handler.py        # State machine (IDLE→PDB→temp→processing)
│   └── job_manager.py         # SQLite job queue + background threads
│
├── tools/
│   ├── p2rank/                # P2Rank binary   (download separately)
│   ├── Seq2Topt/              # Seq2Topt source  (git clone separately)
│   └── large_model_pth/       # Model weights    (download separately)
│
├── RFdiffusion/               # (git submodule)
├── ProteinMPNN/               # (git submodule)
├── CLEAN/                     # (git submodule)
├── SoDoPE_paper_2020/         # (git submodule)
├── UniKP/                     # (git submodule)
│
├── envs/
│   ├── lin.yml                # Exact conda env for lin (main environment)
│   └── Uni_test.yml           # Exact conda env for Uni_test (UniKP)
│
├── docs/
│   └── LINE_setup_guide.md
└── tests/
    ├── test_seq2topt.py
    └── test_smiles_from_pdb.py
```

---

## Configuration (`pipeline/config.py`)

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `NUM_DESIGNS` | `1` | RFdiffusion backbone designs per iteration |
| `NUM_SEQ_PER_TARGET` | `2` | ProteinMPNN sequences per backbone |
| `MAX_ITERATIONS` | `2` | Max generation loops before returning best-so-far |
| `MAX_GENERATED_LENGTH` | `None` | Cap backbone length (e.g. `100` to save VRAM) |
| `RFDIFFUSION_ENV` | `"lin"` | Conda env for RFdiffusion |
| `CLEAN_ENV` | `"lin"` | Conda env for CLEAN |
| `UNIKP_ENV` | `"Uni_test"` | Conda env for UniKP |
| `SEQ2TOPT_ENV` | `"lin"` | Conda env for Seq2Topt |

---

## Security Notes

- **`.env` is gitignored** — LINE credentials are never committed to the repository
- `jobs.db` (SQLite with user job history) is gitignored
- `data/` (all generated PDBs, sequences, prediction results) is gitignored
- Large model weights (`*.pth`, `*.pt`) are gitignored — download them separately per the instructions above

---

## Acknowledgements

This platform integrates the following open-source tools:

- **RFdiffusion** — [Watson et al., Nature 2023](https://www.nature.com/articles/s41586-023-06415-8)
- **ProteinMPNN** — [Dauparas et al., Science 2022](https://www.science.org/doi/10.1126/science.add2187)
- **CLEAN** — [Yu et al., Science 2023](https://www.science.org/doi/10.1126/science.adf2465)
- **SoDoPE** — [van Kempen et al., PLOS CB 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008576)
- **UniKP** — [Yu et al., Nature Comm. 2023](https://www.nature.com/articles/s41467-023-38082-4)
- **Seq2Topt** — [Qiu et al., ACS Synth. Biol. 2023](https://pubs.acs.org/doi/10.1021/acssynbio.2c00416)
- **P2Rank** — [Krivák & Hoksza, J. Cheminformatics 2018](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0285-8)
