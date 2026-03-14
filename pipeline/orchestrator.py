"""Main pipeline orchestrator: backbone generation → sequence design → screening."""

import os
import logging
from typing import Callable, Dict, Optional

from .config import (
    NUM_DESIGNS, NUM_SEQ_PER_TARGET, MAX_ITERATIONS, MAX_GENERATED_LENGTH,
    PDB_DIR, RFDIFFUSION_RESULT_DIR, PROTEINMPNN_RESULT_DIR,
    CLEAN_RESULT_DIR, SOLUBILITY_RESULT_DIR, UNIKP_RESULT_DIR,
    SEQ2TOPT_RESULT_DIR, FINAL_RESULT_DIR, JSON_DIR,
    ALL_DATA_DIRS,
)
from .pdb_utils import download_pdb, get_template_sequence, get_ligand_smiles
from .active_site import (
    run_p2rank, parse_p2rank_output,
    cluster_into_islands, islands_to_contig, islands_to_fixed_residues,
)
from .rfdiffusion_runner import run_rfdiffusion
from .proteinmpnn_runner import (
    parse_chains, assign_chains, make_fixed_positions,
    update_fixed_positions, run_proteinmpnn,
)
from .clean_runner import (
    prepare_clean_input, run_clean, parse_clean_results, filter_by_ec,
)
from .sodope_runner import run_sodope, filter_by_solubility
from .unikp_runner import run_unikp, parse_unikp_results, filter_by_kinetics
from .seq2topt_runner import (
    run_seq2topt, parse_seq2topt_results, select_by_temperature,
)

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]


def _flat_sequences(design_seqs: Dict[str, list]) -> Dict[str, str]:
    """Flatten {design_name: [seq1, seq2, ...]} → {design_name_1: seq1, ...}."""
    flat: Dict[str, str] = {}
    for design_name, seqs in design_seqs.items():
        for idx, seq in enumerate(seqs, start=1):
            flat[f"{design_name}_{idx}"] = seq
    return flat


def _backbone_length(pdb_path: str) -> int:
    """Count CA atoms in chain A of a PDB to determine backbone residue count."""
    count = 0
    seen = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                chain = line[21]
                res_seq = line[22:26].strip()
                key = (chain, res_seq)
                if chain == "A" and key not in seen:
                    seen.add(key)
                    count += 1
    return count


class PipelineOrchestrator:
    """Coordinate the full de-novo protein generation and evaluation loop."""

    def __init__(self):
        # Ensure all data directories exist
        for d in ALL_DATA_DIRS:
            os.makedirs(d, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    def run(
        self,
        pdb_id: str,
        target_temp: int,
        job_dir: str,
        progress_callback: Progress = print,
    ) -> Dict:
        """
        Execute the full pipeline for *pdb_id* at *target_temp* °C.

        Parameters
        ----------
        pdb_id : str
            4-letter RCSB PDB code (e.g. '1PMO').
        target_temp : int
            Desired reaction temperature in °C.
        job_dir : str
            Working directory for this job's intermediate files.
        progress_callback : callable
            Called with a status string at each major step.

        Returns
        -------
        dict
            Keys: name, sequence, ec, solubility, kcat, Km, topt
        """
        os.makedirs(job_dir, exist_ok=True)
        cb = progress_callback

        # ── Step 1 — Template protein ─────────────────────────────────────────
        cb("Downloading template PDB…")
        pdb_path = download_pdb(pdb_id, os.path.join(job_dir, "pdb"))
        template_seq = get_template_sequence(pdb_path)
        cb(f"Template sequence retrieved ({len(template_seq)} aa).")

        smiles = get_ligand_smiles(pdb_id, pdb_path=pdb_path)
        if not smiles:
            cb("No ligand SMILES found; kinetics screening will be skipped.")
        else:
            cb(f"Ligand SMILES: {smiles[:60]}…" if len(smiles) > 60 else f"Ligand SMILES: {smiles}")

        # ── Step 2 — Active site prediction ──────────────────────────────────
        cb("Running P2Rank pocket prediction…")
        p2rank_out = os.path.join(job_dir, "p2rank")
        try:
            run_p2rank(pdb_path, p2rank_out)
            pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
            raw_residues = parse_p2rank_output(p2rank_out, pdb_stem)
            # Consolidate into at most 2 contiguous motif islands
            motif_islands = cluster_into_islands(raw_residues, max_islands=2, gap_tolerance=4)
            fixed_residues = islands_to_fixed_residues(motif_islands)
            cb(
                f"Active site residues: {raw_residues}\n"
                f"Motif islands (≤2): {motif_islands}\n"
                f"Fixed positions: {fixed_residues}"
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("P2Rank failed (%s); using empty contig.", exc)
            cb("P2Rank unavailable; generating without fixed residues.")
            motif_islands = []
            fixed_residues = []

        # ── Step 3 — Template screening baselines ────────────────────────────
        cb("Computing template baselines (EC, solubility, kinetics)…")

        template_sequences = {"template": template_seq}
        template_csv = os.path.join(job_dir, "template_clean_input.tsv")
        template_fasta = os.path.join(job_dir, "template_clean_input.fasta")
        prepare_clean_input(template_sequences, template_csv, template_fasta)

        template_ec_reliable = False
        try:
            template_clean_results_csv = run_clean(template_csv, os.path.join(job_dir, "template_clean_result"))
            template_ec_preds = parse_clean_results(template_clean_results_csv)
            template_ec = template_ec_preds.get("template", "1.1.1.1")
            template_ec_reliable = True
        except Exception as exc:
            logger.warning("Template CLEAN failed: %s", exc, exc_info=True)
            template_ec = "unknown"
        cb(f"Template EC: {template_ec}" + ("" if template_ec_reliable else " (unavailable — EC filter disabled)"))

        template_sol_scores = run_sodope(template_sequences)
        template_sol = template_sol_scores.get("template", 0.5)
        cb(f"Template solubility (SWI prob): {template_sol:.3f}")

        template_kinetics: Dict = {}
        if smiles:
            try:
                unikp_out = run_unikp(template_sequences, smiles, os.path.join(job_dir, "template_unikp"))
                unikp_parsed = parse_unikp_results(unikp_out)
                template_kinetics = unikp_parsed.get("template", {})
                cb(
                    f"Template kinetics — kcat={template_kinetics.get('kcat', 0):.3f} s⁻¹, "
                    f"Km={template_kinetics.get('Km', 0):.3f} mM, "
                    f"kcat/Km={template_kinetics.get('kcat_Km', 0):.3f}"
                )
            except Exception as exc:
                logger.warning("Template UniKP failed: %s; kinetics screening disabled.", exc)
                cb("UniKP unavailable; kinetics screening will be skipped.")

        try:
            s2t_tmpl = run_seq2topt(template_sequences, os.path.join(job_dir, "template_seq2topt"))
            topt_tmpl = parse_seq2topt_results(s2t_tmpl, seq_names=["template"])
            template_topt = topt_tmpl.get("template")
            cb(f"Template Topt: {f'{template_topt:.1f}°C' if template_topt is not None else 'N/A'}")
        except Exception as exc:
            logger.warning("Template Seq2Topt failed: %s", exc)
            template_topt = None
            cb("Seq2Topt unavailable for template.")

        # ── Step 4 — Generation / evaluation iteration loop ──────────────────
        best_so_far: Optional[Dict] = None

        # Cap total backbone length to MAX_GENERATED_LENGTH to save GPU memory
        effective_length = len(template_seq)
        if MAX_GENERATED_LENGTH is not None and effective_length > MAX_GENERATED_LENGTH:
            effective_length = MAX_GENERATED_LENGTH
            # Drop island ranges and fixed residues that fall beyond the cap
            motif_islands = [(s, min(e, effective_length)) for s, e in motif_islands if s <= effective_length]
            fixed_residues = islands_to_fixed_residues(motif_islands)
            cb(
                f"Generation length capped at {MAX_GENERATED_LENGTH} aa "
                f"(template is {len(template_seq)} aa)."
            )

        contig_str = islands_to_contig(motif_islands, effective_length)

        for iteration in range(1, MAX_ITERATIONS + 1):
            cb(f"Iteration {iteration}/{MAX_ITERATIONS}: generating backbones with RFdiffusion…")

            iter_dir = os.path.join(job_dir, f"iter_{iteration}")
            rf_out_prefix = os.path.join(iter_dir, "rfdiffusion", "design")
            mpnn_out_dir  = os.path.join(iter_dir, "proteinmpnn")
            json_dir      = os.path.join(iter_dir, "json")

            try:
                backbone_pdbs = run_rfdiffusion(
                    pdb_path, contig_str, rf_out_prefix,
                    num_designs=NUM_DESIGNS, job_dir=iter_dir,
                )
            except RuntimeError as exc:
                cb(f"RFdiffusion error: {exc}")
                break

            if not backbone_pdbs:
                cb("RFdiffusion produced no backbones. Stopping.")
                break

            cb(f"Generated {len(backbone_pdbs)} backbones. Running ProteinMPNN…")
            rfdiff_pdb_dir = os.path.dirname(backbone_pdbs[0])
            jsonl          = os.path.join(json_dir, "chains.jsonl")
            assigned_jsonl = os.path.join(json_dir, "chains_assigned.jsonl")
            fixed_pos_jsonl  = os.path.join(json_dir, "fixed_pos.jsonl")
            updated_pos_jsonl = os.path.join(json_dir, "updated_fixed_pos.jsonl")

            try:
                parse_chains(rfdiff_pdb_dir, jsonl)
                assign_chains(jsonl, assigned_jsonl)
                make_fixed_positions(jsonl, fixed_pos_jsonl)

                # Build per-design residue map, clamped to each backbone's length
                design_names = [os.path.splitext(os.path.basename(p))[0] for p in backbone_pdbs]
                fixed_map = {}
                for pdb_p, name in zip(backbone_pdbs, design_names):
                    bb_len = _backbone_length(pdb_p)
                    clamped = [r for r in fixed_residues if r <= bb_len]
                    if len(clamped) < len(fixed_residues):
                        logger.warning(
                            "Clamped fixed residues for %s: backbone has %d residues; "
                            "dropped positions %s",
                            name, bb_len,
                            sorted(set(fixed_residues) - set(clamped)),
                        )
                    fixed_map[name] = clamped
                update_fixed_positions(fixed_pos_jsonl, updated_pos_jsonl, fixed_map)

                design_seqs = run_proteinmpnn(
                    jsonl, assigned_jsonl, updated_pos_jsonl,
                    mpnn_out_dir, num_seq=NUM_SEQ_PER_TARGET,
                )
            except RuntimeError as exc:
                cb(f"ProteinMPNN error: {exc}")
                break

            all_seqs = _flat_sequences(design_seqs)
            cb(f"ProteinMPNN generated {len(all_seqs)} sequences. Running CLEAN…")

            # ── CLEAN filter ──────────────────────────────────────────────────
            clean_csv  = os.path.join(iter_dir, "clean_input.tsv")
            clean_fasta = os.path.join(iter_dir, "clean_input.fasta")
            prepare_clean_input(all_seqs, clean_csv, clean_fasta)
            ec_preds = {}
            if not template_ec_reliable:
                cb("EC filter skipped (template EC unavailable). Running SoDoPE…")
                passed_ec = all_seqs
            else:
                try:
                    clean_results_csv = run_clean(clean_csv, os.path.join(iter_dir, "clean_result"))
                    ec_preds = parse_clean_results(clean_results_csv)
                    passed_ec = filter_by_ec(all_seqs, ec_preds, template_ec)
                    cb(f"CLEAN filter: {len(passed_ec)}/{len(all_seqs)} passed. Running SoDoPE…")
                except Exception as exc:
                    logger.warning("CLEAN failed: %s; skipping EC filter.", exc)
                    cb("CLEAN unavailable; skipping EC filter. Running SoDoPE…")
                    passed_ec = all_seqs
            if not passed_ec:
                cb("No candidates passed EC filter. Iterating…")
                if best_so_far is None:
                    # Track best by solubility even if EC fails
                    sol_all = run_sodope(all_seqs)
                    best_name = max(sol_all, key=sol_all.get)
                    best_so_far = {
                        "name": best_name, "sequence": all_seqs[best_name],
                        "ec": ec_preds.get(best_name, template_ec),
                        "solubility": sol_all[best_name],
                    }
                continue

            # ── SoDoPE filter ─────────────────────────────────────────────────
            sol_scores = run_sodope(passed_ec)
            passed_sol = filter_by_solubility(passed_ec, sol_scores, template_sol)
            cb(f"SoDoPE filter: {len(passed_sol)}/{len(passed_ec)} passed.")

            if not passed_sol:
                cb("No candidates passed solubility filter. Iterating…")
                if best_so_far is None:
                    best_name = max(sol_scores, key=sol_scores.get)
                    best_so_far = {
                        "name": best_name,
                        "sequence": passed_ec.get(best_name, all_seqs.get(best_name, "")),
                        "solubility": sol_scores.get(best_name, 0),
                    }
                continue

            # ── UniKP filter ──────────────────────────────────────────────────
            passed_kinetics = passed_sol
            if smiles and template_kinetics:
                cb("Running UniKP kinetics screening…")
                try:
                    unikp_out = run_unikp(
                        passed_sol, smiles,
                        os.path.join(iter_dir, "unikp"),
                    )
                    kinetics = parse_unikp_results(unikp_out)
                    passed_kinetics = filter_by_kinetics(passed_sol, kinetics, template_kinetics)
                    cb(f"UniKP filter: {len(passed_kinetics)}/{len(passed_sol)} passed.")
                except Exception as exc:
                    logger.warning("UniKP failed: %s; skipping kinetics filter.", exc)
                    cb("UniKP unavailable; skipping kinetics filter.")
                    kinetics = {}
            else:
                kinetics = {}

            if not passed_kinetics:
                cb("No candidates passed kinetics filter. Iterating…")
                # Update best_so_far from solubility-passing set
                if not best_so_far or len(passed_sol) > 0:
                    best_name = max(sol_scores, key=lambda n: sol_scores.get(n, 0))
                    best_so_far = {
                        "name": best_name,
                        "sequence": passed_sol.get(best_name, ""),
                        "solubility": sol_scores.get(best_name, 0),
                        **({"kcat": kinetics[best_name]["kcat"],
                            "Km":   kinetics[best_name]["Km"],
                            "kcat_Km": kinetics[best_name]["kcat_Km"]}
                           if best_name in kinetics else {}),
                    }
                continue

            # ── Seq2Topt — select closest to target temperature ───────────────
            cb("Running Seq2Topt temperature prediction…")
            try:
                seq2topt_out = run_seq2topt(
                    passed_kinetics, os.path.join(iter_dir, "seq2topt")
                )
                topt_preds = parse_seq2topt_results(seq2topt_out, seq_names=list(passed_kinetics.keys()))
                best = select_by_temperature(passed_kinetics, topt_preds, target_temp)
            except Exception as exc:
                logger.warning("Seq2Topt failed: %s; selecting by solubility.", exc)
                cb("Seq2Topt unavailable; selecting best by solubility.")
                best_name = max(
                    passed_kinetics,
                    key=lambda n: sol_scores.get(n, 0),
                )
                best = {"name": best_name, "sequence": passed_kinetics[best_name], "topt": None}
                topt_preds = {}

            if best:
                name = best["name"]
                result = {
                    "name":       name,
                    "sequence":   best["sequence"],
                    "ec":         ec_preds.get(name, template_ec),
                    "solubility": sol_scores.get(name, 0),
                    "topt":       best.get("topt"),
                }
                if name in kinetics:
                    result.update({
                        "kcat":    kinetics[name]["kcat"],
                        "Km":      kinetics[name]["Km"],
                        "kcat_Km": kinetics[name]["kcat_Km"],
                    })
                cb(
                    f"Best candidate: {name} | "
                    f"EC={result['ec']} | "
                    f"Solubility={result['solubility']:.3f} | "
                    f"Topt={result['topt']}°C"
                )
                return result

        # ── All iterations exhausted — return best found so far ───────────────
        if best_so_far:
            cb(
                f"All iterations complete. Returning best-so-far: "
                f"{best_so_far.get('name', 'unknown')}"
            )
            # Attempt Seq2Topt on the best-so-far sequence if topt not yet set
            if best_so_far.get("topt") is None and best_so_far.get("sequence"):
                try:
                    seq_map = {best_so_far["name"]: best_so_far["sequence"]}
                    s2t_out = run_seq2topt(seq_map, os.path.join(job_dir, "final_seq2topt"))
                    topt_preds = parse_seq2topt_results(s2t_out, seq_names=list(seq_map.keys()))
                    best_so_far["topt"] = topt_preds.get(best_so_far["name"])
                except Exception as exc:
                    logger.warning("Final Seq2Topt failed: %s", exc)
            return best_so_far

        cb("Pipeline completed with no viable candidates.")
        return {
            "name":     None,
            "sequence": None,
            "message":  "No candidates passed all filters after all iterations.",
        }
