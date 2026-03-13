#!/usr/bin/env python3
"""
recycle_mpnn.py — Iterative coverage recycling protocol for LigandMPNN.

Generates sequences with LigandMPNN, scores peptide library coverage, fixes
covered positions (by threading the best designed sequence onto the backbone
PDB), and redesigns uncovered positions in subsequent rounds.  Repeats until
target coverage is reached, patience is exhausted, or max rounds is hit.

Usage:
    python recycle_mpnn.py \
        --pdb structure.pdb \
        --library 260120_mouse_peptides.txt \
        --output_dir recycle_outputs \
        --num_seqs 1000000 \
        --max_rounds 50
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time

import numpy as np
from prody import parsePDB, writePDB

from data_utils import restype_1to3
from run_mpnn_sweep import CHECKPOINT_FLAG, CHECKPOINT_FILENAME, resolve_checkpoint
from score_kmer_coverage import (
    compute_positional_coverage,
    load_peptide_library,
    parse_mpnn_fasta,
)

logger = logging.getLogger(__name__)

DEFAULT_LIGANDMPNN_DIR = os.path.dirname(os.path.abspath(__file__))


# ── PDB utilities ─────────────────────────────────────────────────────────

def get_residue_ids_from_pdb(pdb_path: str) -> list:
    """
    Extract ordered PDB residue IDs matching run.py's encoding:
    chain_letter + str(resnum) + icode.

    Uses CA atoms to get one entry per residue.
    """
    atoms = parsePDB(pdb_path, subset="calpha")
    chids = atoms.getChids()
    resnums = atoms.getResnums()
    icodes = atoms.getIcodes()
    residue_ids = []
    for ch, rn, ic in zip(chids, resnums, icodes):
        rid = str(ch) + str(rn) + (ic if ic and ic.strip() else "")
        residue_ids.append(rid)
    return residue_ids


def thread_sequence_onto_backbone(pdb_path: str, sequence: str, output_path: str):
    """
    Thread a designed sequence onto a backbone PDB.

    Replaces residue names with the designed sequence's amino acid types.
    Always threads onto the original PDB to avoid artifact accumulation.
    """
    backbone = parsePDB(pdb_path, subset="backbone")
    # backbone has 4 atoms per residue (N, CA, C, O)
    seq_3letter = np.array(
        [restype_1to3.get(aa, "UNK") for aa in sequence]
    )
    # Repeat each 3-letter code for each backbone atom (N, CA, C, O)
    resnames = np.repeat(seq_3letter, 4)
    backbone.setResnames(resnames)
    writePDB(output_path, backbone)


# ── Coverage helpers ──────────────────────────────────────────────────────

def longest_contiguous_covered(covered: np.ndarray) -> tuple:
    """
    Find the longest contiguous run of True values.

    Returns (start, end, length) where end is exclusive.
    """
    if not covered.any():
        return (0, 0, 0)
    # Pad with False to catch runs at boundaries
    padded = np.concatenate(([False], covered, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    lengths = ends - starts
    best_idx = np.argmax(lengths)
    return (int(starts[best_idx]), int(ends[best_idx]), int(lengths[best_idx]))


def contiguous_block_from_position(covered: np.ndarray, start_pos: int) -> tuple:
    """
    Find the contiguous run of True values starting at start_pos.

    Returns (start, end_exclusive, length).  If covered[start_pos] is False,
    returns (start_pos, start_pos, 0).
    """
    if start_pos >= len(covered) or not covered[start_pos]:
        return (start_pos, start_pos, 0)
    end = start_pos
    while end < len(covered) and covered[end]:
        end += 1
    return (start_pos, end, end - start_pos)


def trim_to_contiguous_block(covered: np.ndarray, growth_mode: str) -> np.ndarray:
    """
    Keep only the relevant contiguous block, zeroing everything else.

    Parameters
    ----------
    covered : boolean array
    growth_mode : str
        "none"  — keep the longest contiguous block (wherever it is).
        "n_terminal" — keep only the contiguous block starting at position 1
                       (skip Met at position 0).

    Returns a new boolean array.
    """
    result = np.zeros_like(covered)
    if growth_mode == "n_terminal":
        start, end, length = contiguous_block_from_position(covered, 1)
    else:
        start, end, length = longest_contiguous_covered(covered)
    if length > 0:
        result[start:end] = True
    return result


def select_best_sequence(
    fasta_path: str,
    length_sets: dict,
    cumulative_covered: np.ndarray,
    metric: str = "total_coverage",
    growth_mode: str = "none",
) -> tuple:
    """
    Score all designed sequences and select the best one.

    Parameters
    ----------
    fasta_path : str
        Path to LigandMPNN output FASTA.
    length_sets : dict
        Peptide library grouped by length.
    cumulative_covered : np.ndarray
        Boolean array of positions already covered in previous rounds
        (already trimmed to the relevant contiguous block).
    metric : str
        "total_coverage" — maximize union of cumulative + new coverage.
        "longest_contiguous" — maximize longest contiguous covered block.
        Ignored when growth_mode is "n_terminal".
    growth_mode : str
        "none" — use metric as-is.
        "n_terminal" — score by length of contiguous block from position 1.

    Returns
    -------
    best_sequence : str
    best_covered : np.ndarray
        Boolean coverage array for the best sequence.
    best_score : float
    best_seq_id : str
    """
    entries = parse_mpnn_fasta(fasta_path)

    best_sequence = None
    best_covered = None
    best_score = -1
    best_new_positions = -1
    best_confidence = -1.0
    best_seq_id = ""

    for seq_id, sequence, meta in entries:
        if seq_id == "native":
            continue

        cov, n_cov, n_tot, n_hits, covered = compute_positional_coverage(
            sequence, length_sets,
        )

        # Compute union with cumulative
        union = cumulative_covered | covered
        union_count = int(union.sum())
        new_positions = int(covered.sum()) - int((cumulative_covered & covered).sum())
        confidence = float(meta.get("overall_confidence", 0))

        if growth_mode == "n_terminal":
            # Score = length of contiguous block from position 1 in the union
            _, _, nt_len = contiguous_block_from_position(union, 1)
            score = nt_len
            is_better = (
                score > best_score
                or (score == best_score and new_positions > best_new_positions)
                or (score == best_score and new_positions == best_new_positions
                    and confidence > best_confidence)
            )
        elif metric == "total_coverage":
            score = union_count
            # Tiebreak: more new positions, then higher confidence
            is_better = (
                score > best_score
                or (score == best_score and new_positions > best_new_positions)
                or (score == best_score and new_positions == best_new_positions
                    and confidence > best_confidence)
            )
        elif metric == "longest_contiguous":
            _, _, longest = longest_contiguous_covered(union)
            score = longest
            is_better = (
                score > best_score
                or (score == best_score and union_count > best_new_positions)
            )
        else:
            raise ValueError(f"Unknown selection metric: {metric!r}")

        if is_better:
            best_sequence = sequence
            best_covered = covered
            best_score = score
            best_new_positions = new_positions if metric != "longest_contiguous" else union_count
            best_confidence = confidence
            best_seq_id = seq_id

    return best_sequence, best_covered, best_score, best_seq_id


# ── LigandMPNN runner ────────────────────────────────────────────────────

def run_ligandmpnn(
    pdb_path: str,
    output_dir: str,
    ligandmpnn_dir: str,
    model_type: str,
    noise: str,
    temperature: float,
    bias_AA: str,
    batch_size: int,
    num_seqs: int,
    seed: int,
    fixed_residues: str = "",
) -> str:
    """
    Run LigandMPNN's run.py and return the path to the output FASTA.
    """
    run_script = os.path.join(ligandmpnn_dir, "run.py")
    model_params_dir = os.path.join(ligandmpnn_dir, "model_params")
    checkpoint_path = resolve_checkpoint(model_type, noise, model_params_dir)
    checkpoint_flag = CHECKPOINT_FLAG[model_type]

    number_of_batches = max(1, num_seqs // batch_size)

    cmd = [
        sys.executable, run_script,
        "--pdb_path", pdb_path,
        "--out_folder", output_dir,
        "--model_type", model_type,
        checkpoint_flag, checkpoint_path,
        "--temperature", str(temperature),
        "--batch_size", str(batch_size),
        "--number_of_batches", str(number_of_batches),
        "--seed", str(seed),
        "--save_stats", "0",
        "--skip_pdb", "1",
    ]

    if bias_AA:
        cmd.extend(["--bias_AA", bias_AA])

    if fixed_residues:
        cmd.extend(["--fixed_residues", fixed_residues])

    logger.info(f"Running LigandMPNN: {num_seqs} seqs "
                f"(batch_size={batch_size}, batches={number_of_batches})")
    logger.info(f"  Command: {' '.join(cmd[:6])} ...")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"LigandMPNN failed (exit code {result.returncode})")
        logger.error(f"stderr: {result.stderr[-2000:]}")
        raise RuntimeError(f"LigandMPNN run.py failed with exit code {result.returncode}")

    logger.info(f"  LigandMPNN finished in {elapsed:.0f}s")

    # Save stdout/stderr
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "stdout.log"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(log_dir, "stderr.log"), "w") as f:
        f.write(result.stderr)

    # Find output FASTA
    seqs_dir = os.path.join(output_dir, "seqs")
    import glob as _glob
    fa_files = sorted(_glob.glob(os.path.join(seqs_dir, "*.fa")))
    if not fa_files:
        raise FileNotFoundError(f"No .fa files found in {seqs_dir}")
    return fa_files[0]


# ── Resume from checkpoint ────────────────────────────────────────────────

def _try_resume(output_dir, length_sets, seq_len, growth_mode="none"):
    """
    Attempt to resume a previous run from saved artifacts.

    Scans output_dir for completed rounds (those with both
    best_sequence.fasta and coverage_summary.json).  Reconstructs
    cumulative_covered by re-scoring each round's best sequence against the
    library — fast (<1 ms per sequence) and avoids persisting the boolean
    array.

    Returns
    -------
    dict with keys: start_round, cumulative_covered, best_sequence,
    round_results, rounds_without_improvement, prev_coverage_count.
    Returns None if no checkpoint is found.
    """
    progression_path = os.path.join(output_dir, "progression.json")
    if not os.path.exists(progression_path):
        return None

    with open(progression_path) as f:
        progression = json.load(f)

    saved_rounds = progression.get("rounds", [])
    if not saved_rounds:
        return None

    # Find the last fully completed round (both artifacts present)
    last_complete = -1
    for rinfo in saved_rounds:
        rnum = rinfo["round"]
        rdir = os.path.join(output_dir, f"round_{rnum:03d}")
        fasta_ok = os.path.exists(os.path.join(rdir, "best_sequence.fasta"))
        json_ok = os.path.exists(os.path.join(rdir, "coverage_summary.json"))
        if fasta_ok and json_ok:
            last_complete = rnum
        else:
            break  # stop at first incomplete round

    if last_complete < 0:
        return None

    # Reconstruct cumulative_covered by re-scoring each round's best seq
    cumulative_covered = np.zeros(seq_len, dtype=bool)
    best_sequence = None

    for rnum in range(last_complete + 1):
        rdir = os.path.join(output_dir, f"round_{rnum:03d}")
        fasta_file = os.path.join(rdir, "best_sequence.fasta")
        with open(fasta_file) as f:
            lines = f.readlines()
        # Second line is the sequence
        seq = lines[1].strip()
        _, _, _, _, covered = compute_positional_coverage(seq, length_sets)
        cumulative_covered |= covered
        best_sequence = seq

    # Trim to contiguous block (matching the growth_mode used for this run)
    cumulative_covered = trim_to_contiguous_block(cumulative_covered, growth_mode)

    # Restore round_results up to last_complete
    round_results = [r for r in saved_rounds if r["round"] <= last_complete]

    last_info = round_results[-1]
    rounds_without_improvement = last_info.get("rounds_without_improvement", 0)
    prev_coverage_count = last_info.get("cumulative_covered", 0)

    return {
        "start_round": last_complete + 1,
        "cumulative_covered": cumulative_covered,
        "best_sequence": best_sequence,
        "round_results": round_results,
        "rounds_without_improvement": rounds_without_improvement,
        "prev_coverage_count": prev_coverage_count,
    }


def _save_and_return(output_dir, pdb_path, library_path, model_type, noise,
                     temperature, bias_AA, num_seqs, batch_size, seed,
                     selection_metric, x_mode, target_coverage, patience,
                     max_rounds, lib_stats, round_results, best_sequence,
                     cumulative_covered, seq_len, residue_ids,
                     growth_mode="none"):
    """Save final artifacts and return the progression dict."""
    coverage_frac = cumulative_covered.mean()
    new_total = int(cumulative_covered.sum())

    progression = {
        "pdb": os.path.abspath(pdb_path),
        "library": os.path.abspath(library_path),
        "parameters": {
            "model_type": model_type,
            "noise": noise,
            "temperature": temperature,
            "bias_AA": bias_AA,
            "num_seqs": num_seqs,
            "batch_size": batch_size,
            "base_seed": seed,
            "selection_metric": selection_metric,
            "x_mode": x_mode,
            "target_coverage": target_coverage,
            "patience": patience,
            "max_rounds": max_rounds,
            "growth_mode": growth_mode,
        },
        "library_stats": lib_stats,
        "rounds": round_results,
    }
    with open(os.path.join(output_dir, "progression.json"), "w") as f:
        json.dump(progression, f, indent=2)

    if best_sequence is not None:
        with open(os.path.join(output_dir, "final_sequence.fasta"), "w") as f:
            f.write(f">final_sequence coverage={coverage_frac:.4f} "
                    f"rounds={len(round_results)}\n{best_sequence}\n")

    logger.info(f"\n{'='*70}")
    logger.info(f"RECYCLING COMPLETE")
    logger.info(f"  Rounds: {len(round_results)}")
    logger.info(f"  Final coverage: {new_total}/{seq_len} ({coverage_frac*100:.1f}%)")
    logger.info(f"  Results: {output_dir}")
    logger.info(f"{'='*70}")

    return progression


# ── Main recycling loop ──────────────────────────────────────────────────

def recycle(
    pdb_path: str,
    library_path: str,
    output_dir: str,
    ligandmpnn_dir: str,
    model_type: str,
    noise: str,
    temperature: float,
    bias_AA: str,
    num_seqs: int,
    batch_size: int,
    seed: int,
    max_rounds: int,
    target_coverage: float,
    patience: int,
    x_mode: str,
    selection_metric: str,
    growth_mode: str = "none",
) -> dict:
    """
    Run the iterative coverage recycling protocol.

    Returns a dict with the final progression data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load library
    logger.info(f"Loading peptide library from {library_path} (x_mode={x_mode})")
    length_sets, lib_stats = load_peptide_library(library_path, x_mode)
    logger.info(f"  {lib_stats['n_unique']:,} unique peptides "
                f"(lengths {lib_stats['min_length']}-{lib_stats['max_length']})")

    # Get residue IDs from PDB (invariant across rounds)
    residue_ids = get_residue_ids_from_pdb(pdb_path)
    seq_len = len(residue_ids)
    logger.info(f"PDB has {seq_len} residues")

    # Try to resume from a previous run
    resumed = _try_resume(output_dir, length_sets, seq_len, growth_mode)
    if resumed is not None:
        start_round = resumed["start_round"]
        cumulative_covered = resumed["cumulative_covered"]
        best_sequence = resumed["best_sequence"]
        round_results = resumed["round_results"]
        rounds_without_improvement = resumed["rounds_without_improvement"]
        prev_coverage_count = resumed["prev_coverage_count"]
        logger.info(f"RESUMED from checkpoint — {start_round} rounds completed")
        logger.info(f"  Cumulative coverage: {cumulative_covered.sum()}/{seq_len} "
                     f"({cumulative_covered.mean()*100:.1f}%)")
        logger.info(f"  Rounds without improvement: {rounds_without_improvement}")

        # Check if already converged before resuming
        coverage_frac = cumulative_covered.mean()
        if coverage_frac >= target_coverage:
            logger.info(f"Target coverage already reached — nothing to do")
            return _save_and_return(output_dir, pdb_path, library_path, model_type,
                                    noise, temperature, bias_AA, num_seqs, batch_size,
                                    seed, selection_metric, x_mode, target_coverage,
                                    patience, max_rounds, lib_stats, round_results,
                                    best_sequence, cumulative_covered, seq_len,
                                    residue_ids, growth_mode=growth_mode)
        if rounds_without_improvement >= patience:
            logger.info(f"Patience already exhausted — nothing to do")
            return _save_and_return(output_dir, pdb_path, library_path, model_type,
                                    noise, temperature, bias_AA, num_seqs, batch_size,
                                    seed, selection_metric, x_mode, target_coverage,
                                    patience, max_rounds, lib_stats, round_results,
                                    best_sequence, cumulative_covered, seq_len,
                                    residue_ids, growth_mode=growth_mode)
    else:
        start_round = 0
        cumulative_covered = np.zeros(seq_len, dtype=bool)
        best_sequence = None
        round_results = []
        rounds_without_improvement = 0
        prev_coverage_count = 0

    for round_num in range(start_round, max_rounds):
        round_dir = os.path.join(output_dir, f"round_{round_num:03d}")
        mpnn_output_dir = os.path.join(round_dir, "mpnn_output")
        os.makedirs(round_dir, exist_ok=True)

        logger.info(f"\n{'='*70}")
        logger.info(f"ROUND {round_num} — cumulative coverage: "
                     f"{cumulative_covered.sum()}/{seq_len} "
                     f"({cumulative_covered.mean()*100:.1f}%)")
        logger.info(f"{'='*70}")

        # Thread best sequence onto PDB if not first round
        input_pdb = pdb_path
        fixed_residues_str = ""
        if round_num > 0 and best_sequence is not None:
            threaded_pdb = os.path.join(round_dir, "threaded.pdb")
            logger.info(f"Threading best sequence onto backbone PDB...")
            thread_sequence_onto_backbone(pdb_path, best_sequence, threaded_pdb)
            input_pdb = threaded_pdb

            # Build fixed_residues string for covered positions
            fixed_ids = [
                residue_ids[i] for i in range(seq_len) if cumulative_covered[i]
            ]
            fixed_residues_str = " ".join(fixed_ids)
            logger.info(f"Fixing {len(fixed_ids)} covered positions")

        # Run LigandMPNN
        round_seed = seed + round_num
        fasta_path = run_ligandmpnn(
            pdb_path=input_pdb,
            output_dir=mpnn_output_dir,
            ligandmpnn_dir=ligandmpnn_dir,
            model_type=model_type,
            noise=noise,
            temperature=temperature,
            bias_AA=bias_AA,
            batch_size=batch_size,
            num_seqs=num_seqs,
            seed=round_seed,
            fixed_residues=fixed_residues_str,
        )

        # Score and select best
        logger.info(f"Scoring {num_seqs} sequences...")
        t0 = time.time()
        seq, covered_arr, score, seq_id = select_best_sequence(
            fasta_path, length_sets, cumulative_covered, selection_metric,
            growth_mode=growth_mode,
        )
        scoring_time = time.time() - t0
        logger.info(f"  Scoring completed in {scoring_time:.0f}s")

        if seq is None:
            logger.error("No valid sequences found — stopping")
            break

        # Update cumulative coverage — OR in new coverage, then trim to the
        # relevant contiguous block (scattered positions aren't fixed so
        # tracking them would inflate scores)
        prev_total = int(cumulative_covered.sum())
        raw_union = cumulative_covered | covered_arr
        cumulative_covered = trim_to_contiguous_block(raw_union, growth_mode)
        new_total = int(cumulative_covered.sum())
        n_new = new_total - prev_total
        coverage_frac = cumulative_covered.mean()

        # Contiguous coverage info (after trimming, this IS the block)
        cont_start, cont_end, cont_len = longest_contiguous_covered(cumulative_covered)

        best_sequence = seq

        # Check improvement
        if new_total > prev_coverage_count:
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        prev_coverage_count = new_total

        # Uncovered positions
        uncovered_ids = [
            residue_ids[i] for i in range(seq_len) if not cumulative_covered[i]
        ]

        round_info = {
            "round": round_num,
            "best_seq_id": seq_id,
            "new_positions_covered": n_new,
            "cumulative_covered": new_total,
            "cumulative_coverage": float(coverage_frac),
            "longest_contiguous_block": cont_len,
            "longest_contiguous_range": f"{cont_start}-{cont_end}",
            "uncovered_positions": len(uncovered_ids),
            "scoring_time_s": scoring_time,
            "rounds_without_improvement": rounds_without_improvement,
        }
        round_results.append(round_info)

        logger.info(f"  Best sequence: {seq_id}")
        logger.info(f"  New positions covered: {n_new}")
        logger.info(f"  Cumulative: {new_total}/{seq_len} ({coverage_frac*100:.1f}%)")
        logger.info(f"  Longest contiguous block: {cont_len} "
                     f"(positions {cont_start}-{cont_end})")
        logger.info(f"  Uncovered: {len(uncovered_ids)} positions")

        # Save round artifacts
        with open(os.path.join(round_dir, "best_sequence.fasta"), "w") as f:
            f.write(f">round_{round_num:03d}_best id={seq_id}\n{seq}\n")

        with open(os.path.join(round_dir, "coverage_summary.json"), "w") as f:
            json.dump({
                **round_info,
                "uncovered_residue_ids": uncovered_ids,
                "selection_metric": selection_metric,
            }, f, indent=2)

        # Save progression so far (overwrite each round for resume visibility)
        progression = {
            "pdb": os.path.abspath(pdb_path),
            "library": os.path.abspath(library_path),
            "parameters": {
                "model_type": model_type,
                "noise": noise,
                "temperature": temperature,
                "bias_AA": bias_AA,
                "num_seqs": num_seqs,
                "batch_size": batch_size,
                "base_seed": seed,
                "selection_metric": selection_metric,
                "x_mode": x_mode,
                "target_coverage": target_coverage,
                "patience": patience,
                "max_rounds": max_rounds,
                "growth_mode": growth_mode,
            },
            "library_stats": lib_stats,
            "rounds": round_results,
        }
        with open(os.path.join(output_dir, "progression.json"), "w") as f:
            json.dump(progression, f, indent=2)

        # Check convergence
        if coverage_frac >= target_coverage:
            logger.info(f"\nTarget coverage {target_coverage*100:.0f}% REACHED!")
            break
        if rounds_without_improvement >= patience:
            logger.info(f"\nNo improvement for {patience} rounds — stopping (patience)")
            break

    return _save_and_return(output_dir, pdb_path, library_path, model_type,
                            noise, temperature, bias_AA, num_seqs, batch_size,
                            seed, selection_metric, x_mode, target_coverage,
                            patience, max_rounds, lib_stats, round_results,
                            best_sequence, cumulative_covered, seq_len,
                            residue_ids, growth_mode=growth_mode)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Iterative coverage recycling protocol for LigandMPNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python recycle_mpnn.py \\
        --pdb AF-P01012-F1-model_v6.pdb \\
        --library 260120_mouse_peptides.txt \\
        --output_dir recycle_outputs \\
        --num_seqs 1000000 \\
        --max_rounds 50
        """,
    )
    parser.add_argument("--pdb", required=True, help="Input PDB structure")
    parser.add_argument("--library", required=True, help="Peptide library file (one per line)")
    parser.add_argument("--output_dir", default="./recycle_outputs",
                        help="Output directory (default: ./recycle_outputs)")
    parser.add_argument("--noise", default="0.30",
                        help="Noise level for LigandMPNN (default: 0.30)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1)")
    parser.add_argument("--bias_AA", default="N:3.0",
                        help="Amino acid bias string (default: N:3.0)")
    parser.add_argument("--model_type", default="protein_mpnn",
                        help="Model type (default: protein_mpnn)")
    parser.add_argument("--num_seqs", type=int, default=1000000,
                        help="Sequences per round (default: 1000000)")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for LigandMPNN (default: 1000)")
    parser.add_argument("--auto_batch_size", action="store_true",
                        help="Auto-calibrate batch size to fit GPU memory "
                             "(overrides --batch_size)")
    parser.add_argument("--memory_fraction", type=float, default=0.85,
                        help="Fraction of GPU VRAM to target when using "
                             "--auto_batch_size (default: 0.85)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed, incremented per round (default: 42)")
    parser.add_argument("--max_rounds", type=int, default=50,
                        help="Maximum number of rounds (default: 50)")
    parser.add_argument("--target_coverage", type=float, default=1.0,
                        help="Stop when this coverage fraction is reached (default: 1.0)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Rounds without improvement before stopping (default: 5)")
    parser.add_argument("--x_mode", choices=["replace", "exclude"], default="replace",
                        help="How to handle X in library peptides (default: replace)")
    parser.add_argument("--selection_metric",
                        choices=["total_coverage", "longest_contiguous"],
                        default="total_coverage",
                        help="Metric for selecting best sequence per round "
                             "(default: total_coverage)")
    parser.add_argument("--growth_mode",
                        choices=["none", "n_terminal"],
                        default="none",
                        help="'none': fix longest contiguous block; "
                             "'n_terminal': grow from position 1, skipping Met "
                             "(default: none)")
    parser.add_argument("--ligandmpnn_dir", default=DEFAULT_LIGANDMPNN_DIR,
                        help="Path to LigandMPNN repository")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Auto-calibrate batch size if requested
    if args.auto_batch_size:
        from auto_batch_size import calibrate_batch_size

        model_params_dir = os.path.join(args.ligandmpnn_dir, "model_params")
        checkpoint_path = resolve_checkpoint(
            args.model_type, args.noise, model_params_dir
        )

        logger.info("=" * 70)
        logger.info("Auto batch size calibration")
        logger.info(f"  Memory target: {args.memory_fraction:.0%} of GPU VRAM")

        auto_bs_result = calibrate_batch_size(
            pdb_path=os.path.abspath(args.pdb),
            checkpoint_path=checkpoint_path,
            model_type=args.model_type,
            memory_fraction=args.memory_fraction,
            verbose=True,
            throughput_profile=True,
        )
        args.batch_size = auto_bs_result["batch_size"]

        logger.info(f"  Calibrated batch_size: {args.batch_size}")
        logger.info(f"  GPU: {auto_bs_result['gpu_name']}")
        logger.info(f"  Protein: {auto_bs_result['protein_length']} residues")
        logger.info(f"  Base memory: {auto_bs_result['base_memory_mb']:.0f} MB")
        logger.info(f"  Per-sample: {auto_bs_result['per_sample_mb']:.1f} MB")

        if "throughput" in auto_bs_result:
            tp = auto_bs_result["throughput"]
            logger.info(f"  Throughput-optimal batch_size: "
                        f"{tp['throughput_optimal_batch_size']}")
            logger.info(f"  Peak throughput: "
                        f"{tp['peak_throughput_seqs_per_sec']:.1f} seq/s")
        logger.info("=" * 70)

    recycle(
        pdb_path=args.pdb,
        library_path=args.library,
        output_dir=args.output_dir,
        ligandmpnn_dir=args.ligandmpnn_dir,
        model_type=args.model_type,
        noise=args.noise,
        temperature=args.temperature,
        bias_AA=args.bias_AA,
        num_seqs=args.num_seqs,
        batch_size=args.batch_size,
        seed=args.seed,
        max_rounds=args.max_rounds,
        target_coverage=args.target_coverage,
        patience=args.patience,
        x_mode=args.x_mode,
        selection_metric=args.selection_metric,
        growth_mode=args.growth_mode,
    )


if __name__ == "__main__":
    main()
