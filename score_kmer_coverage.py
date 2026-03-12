#!/usr/bin/env python3
"""
score_kmer_coverage.py — Score ProteinMPNN sequences for peptide library coverage.

For each designed sequence, finds all library peptides that appear as exact
substrings, then computes what fraction of positions are covered by at least
one matching peptide ("1x coverage").

Two library modes:
  --x_mode replace : Replace X with N in all library peptides (glycosylation
                     sites become normal Asn — tests maximum possible coverage)
  --x_mode exclude : Drop any library peptide containing X (conservative —
                     only uses peptides with fully confirmed sequence)

Output:
  - condition_summary.csv : side-by-side comparison of all conditions
  - top_sequences/<condition>_top<N>.csv : best sequences per condition
  - library_metadata.json : library stats and run parameters

Usage:
    python score_kmer_coverage.py \\
        --library 260120_mouse_peptides.txt \\
        --mpnn_dir mpnn_outputs \\
        --output_dir coverage_results_replace \\
        --x_mode replace \\
        --top_n 100

    python score_kmer_coverage.py \\
        --library 260120_mouse_peptides.txt \\
        --mpnn_dir mpnn_outputs \\
        --output_dir coverage_results_exclude \\
        --x_mode exclude \\
        --top_n 100

    # Parallel scoring across conditions (default: uses all CPUs):
    python score_kmer_coverage.py \\
        --library 260120_mouse_peptides.txt \\
        --mpnn_dir mpnn_outputs \\
        --num_workers 16
"""
import argparse
import csv
import glob
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Library loading ────────────────────────────────────────────────────────

def load_peptide_library(
    library_path: str,
    x_mode: str = "replace",
) -> Tuple[Dict[int, Set[str]], dict]:
    """
    Load the peptide library, grouped by peptide length for efficient lookup.

    Parameters
    ----------
    library_path : str
        One peptide per line, no headers.
    x_mode : str
        "replace" — substitute X with N (glycosylation sites → Asn)
        "exclude" — drop any peptide containing X

    Returns
    -------
    length_sets : dict of {int: set of str}
        Library peptides grouped by length. Only lengths with peptides present.
    stats : dict
        Library statistics.
    """
    raw_peptides = []
    with open(library_path) as f:
        for line in f:
            pep = line.strip().upper()
            if pep and not pep.startswith("#") and not pep.startswith(">"):
                raw_peptides.append(pep)

    n_raw = len(raw_peptides)
    n_with_x = sum(1 for p in raw_peptides if "X" in p)
    lengths_raw = Counter(len(p) for p in raw_peptides)

    # Apply X handling
    if x_mode == "replace":
        processed = [p.replace("X", "N") for p in raw_peptides]
        n_dropped = 0
    elif x_mode == "exclude":
        processed = [p for p in raw_peptides if "X" not in p]
        n_dropped = n_with_x
    else:
        raise ValueError(f"Unknown x_mode: {x_mode!r}")

    # Group by length for efficient lookup
    library = set(processed)
    length_sets = {}
    for pep in library:
        k = len(pep)
        if k not in length_sets:
            length_sets[k] = set()
        length_sets[k].add(pep)

    lengths_final = Counter(len(p) for p in library)

    stats = {
        "n_raw_peptides": n_raw,
        "n_with_x": n_with_x,
        "x_mode": x_mode,
        "n_dropped": n_dropped,
        "n_after_processing": len(processed),
        "n_unique": len(library),
        "length_distribution_raw": dict(sorted(lengths_raw.items())),
        "length_distribution_final": dict(sorted(lengths_final.items())),
        "min_length": min(length_sets.keys()) if length_sets else 0,
        "max_length": max(length_sets.keys()) if length_sets else 0,
        "n_distinct_lengths": len(length_sets),
    }
    return length_sets, stats


# ── FASTA parsing ──────────────────────────────────────────────────────────

def parse_mpnn_fasta(fasta_path: str) -> List[Tuple[str, str, dict]]:
    """
    Parse a LigandMPNN output FASTA file.

    Returns list of (seq_id, sequence, metadata).
    seq_id is "native" for the first entry (no id= in header).
    """
    entries = []
    current_header = None
    current_seq_parts = []

    def _flush():
        if current_header is not None:
            seq = "".join(current_seq_parts).upper()
            meta = {}
            parts = current_header.split(",")
            meta["name"] = parts[0].strip()
            for part in parts[1:]:
                part = part.strip()
                if "=" in part:
                    key, val = part.split("=", 1)
                    meta[key.strip()] = val.strip()
            seq_id = meta.get("id", "native")
            entries.append((seq_id, seq, meta))

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                _flush()
                current_header = line[1:]
                current_seq_parts = []
            else:
                current_seq_parts.append(line.strip())
    _flush()

    return entries


# ── Coverage computation ───────────────────────────────────────────────────

def compute_positional_coverage(
    sequence: str,
    length_sets: Dict[int, Set[str]],
) -> Tuple[float, int, int, int]:
    """
    Compute 1x positional coverage of a sequence against the peptide library.

    For every substring of each length present in the library, check if it's
    in the corresponding length set. If so, mark all positions it spans as
    covered. Coverage is the fraction of positions covered by at least one
    matching peptide.

    Parameters
    ----------
    sequence : str
        The designed protein sequence.
    length_sets : dict of {int: set of str}
        Library peptides grouped by length.

    Returns
    -------
    coverage : float
        Fraction of positions with >=1x coverage.
    n_covered_positions : int
        Number of positions covered.
    n_total_positions : int
        Total positions in the sequence.
    n_peptide_hits : int
        Number of distinct library peptides found as substrings.
    """
    seq_len = len(sequence)
    # Boolean array: is position i covered by at least one library peptide?
    covered = np.zeros(seq_len, dtype=bool)
    n_hits = 0

    for k, kmer_set in length_sets.items():
        n_windows = seq_len - k + 1
        if n_windows <= 0:
            continue
        for i in range(n_windows):
            substr = sequence[i:i + k]
            if substr in kmer_set:
                covered[i:i + k] = True
                n_hits += 1
        # Early termination: all positions already covered
        if covered.all():
            break

    n_covered = int(covered.sum())
    coverage = n_covered / seq_len if seq_len > 0 else 0.0
    return coverage, n_covered, seq_len, n_hits


# ── Per-condition processing ───────────────────────────────────────────────

@dataclass
class ConditionSummary:
    """Aggregate statistics for one experimental condition."""
    condition_id: str
    group: str
    fasta_path: str
    noise: str = ""
    temperature: str = ""
    bias_AA: str = ""
    model_type: str = ""
    seed: int = 0
    n_sequences: int = 0
    mean_coverage: float = 0.0
    median_coverage: float = 0.0
    min_coverage: float = 0.0
    max_coverage: float = 0.0
    std_coverage: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    n_above_50: int = 0
    n_above_75: int = 0
    n_above_90: int = 0
    n_above_95: int = 0
    n_above_99: int = 0
    n_perfect: int = 0
    mean_peptide_hits: float = 0.0
    native_coverage: float = 0.0
    native_peptide_hits: int = 0
    elapsed_s: float = 0.0


@dataclass
class SeqResult:
    """Coverage result for a single designed sequence."""
    seq_id: str
    coverage: float
    n_covered: int
    n_total: int
    n_peptide_hits: int
    overall_confidence: float = 0.0
    seq_recovery: float = 0.0


def process_condition(
    fasta_path: str,
    length_sets: Dict[int, Set[str]],
    condition_id: str,
    group: str,
    exp_config: dict,
    top_n: int = 100,
) -> Tuple[ConditionSummary, List[SeqResult]]:
    """Score all sequences in one FASTA file against the peptide library."""
    t0 = time.time()
    entries = parse_mpnn_fasta(fasta_path)

    if not entries:
        logger.warning(f"No sequences found in {fasta_path}")
        return ConditionSummary(condition_id, group, fasta_path), []

    results = []
    native_cov = 0.0
    native_hits = 0
    n_total = len(entries)

    for ix, (seq_id, sequence, meta) in enumerate(entries):
        cov, n_cov, n_tot, n_hits = compute_positional_coverage(
            sequence, length_sets,
        )

        result = SeqResult(
            seq_id=seq_id,
            coverage=cov,
            n_covered=n_cov,
            n_total=n_tot,
            n_peptide_hits=n_hits,
            overall_confidence=float(meta.get("overall_confidence", 0)),
            seq_recovery=float(meta.get("seq_rec", 0)),
        )

        if seq_id == "native":
            native_cov = cov
            native_hits = n_hits
        else:
            results.append(result)

        # Progress logging (only from main process or if logging is configured)
        if (ix + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (ix + 1) / elapsed
            eta = (n_total - ix - 1) / rate
            logger.info(
                f"  [{condition_id}] {ix+1:,}/{n_total:,} seqs "
                f"({rate:.0f}/s, ETA {eta:.0f}s)"
            )

    if not results:
        logger.warning(f"No designed sequences in {fasta_path}")
        return ConditionSummary(condition_id, group, fasta_path), []

    # Compute statistics using numpy for speed on large arrays
    coverages = np.array([r.coverage for r in results])
    hits = np.array([r.n_peptide_hits for r in results])

    summary = ConditionSummary(
        condition_id=condition_id,
        group=group,
        fasta_path=fasta_path,
        noise=str(exp_config.get("noise", "")),
        temperature=str(exp_config.get("temperature", "")),
        bias_AA=str(exp_config.get("bias_AA", "")),
        model_type=str(exp_config.get("model_type", "")),
        seed=int(exp_config.get("seed", 0)),
        n_sequences=len(results),
        mean_coverage=float(np.mean(coverages)),
        median_coverage=float(np.median(coverages)),
        min_coverage=float(np.min(coverages)),
        max_coverage=float(np.max(coverages)),
        std_coverage=float(np.std(coverages)),
        p90=float(np.percentile(coverages, 90)),
        p95=float(np.percentile(coverages, 95)),
        p99=float(np.percentile(coverages, 99)),
        n_above_50=int(np.sum(coverages >= 0.50)),
        n_above_75=int(np.sum(coverages >= 0.75)),
        n_above_90=int(np.sum(coverages >= 0.90)),
        n_above_95=int(np.sum(coverages >= 0.95)),
        n_above_99=int(np.sum(coverages >= 0.99)),
        n_perfect=int(np.sum(coverages >= 1.0)),
        mean_peptide_hits=float(np.mean(hits)),
        native_coverage=native_cov,
        native_peptide_hits=native_hits,
        elapsed_s=time.time() - t0,
    )

    # Top N by coverage, break ties by confidence
    results.sort(key=lambda r: (r.coverage, r.overall_confidence), reverse=True)
    top_results = results[:top_n]

    return summary, top_results


# ── Output ─────────────────────────────────────────────────────────────────

def write_summary_csv(summaries: List[ConditionSummary], output_path: str):
    """Write the cross-condition comparison CSV."""
    fieldnames = [
        "condition_id", "group", "noise", "temperature", "bias_AA",
        "model_type", "seed", "n_sequences",
        "native_coverage", "native_peptide_hits",
        "mean_coverage", "median_coverage", "std_coverage",
        "min_coverage", "max_coverage",
        "p90", "p95", "p99",
        "n_above_50", "n_above_75", "n_above_90",
        "n_above_95", "n_above_99", "n_perfect",
        "mean_peptide_hits", "elapsed_s",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({
                "condition_id": s.condition_id,
                "group": s.group,
                "noise": s.noise,
                "temperature": s.temperature,
                "bias_AA": s.bias_AA,
                "model_type": s.model_type,
                "seed": s.seed,
                "n_sequences": s.n_sequences,
                "native_coverage": f"{s.native_coverage:.6f}",
                "native_peptide_hits": s.native_peptide_hits,
                "mean_coverage": f"{s.mean_coverage:.6f}",
                "median_coverage": f"{s.median_coverage:.6f}",
                "std_coverage": f"{s.std_coverage:.6f}",
                "min_coverage": f"{s.min_coverage:.6f}",
                "max_coverage": f"{s.max_coverage:.6f}",
                "p90": f"{s.p90:.6f}",
                "p95": f"{s.p95:.6f}",
                "p99": f"{s.p99:.6f}",
                "n_above_50": s.n_above_50,
                "n_above_75": s.n_above_75,
                "n_above_90": s.n_above_90,
                "n_above_95": s.n_above_95,
                "n_above_99": s.n_above_99,
                "n_perfect": s.n_perfect,
                "mean_peptide_hits": f"{s.mean_peptide_hits:.1f}",
                "elapsed_s": f"{s.elapsed_s:.1f}",
            })


def write_top_sequences_csv(
    condition_id: str,
    top_results: List[SeqResult],
    output_path: str,
):
    """Write the top N sequences for one condition."""
    fieldnames = [
        "condition_id", "seq_id", "coverage",
        "n_covered", "n_total", "n_peptide_hits",
        "overall_confidence", "seq_recovery",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in top_results:
            writer.writerow({
                "condition_id": condition_id,
                "seq_id": r.seq_id,
                "coverage": f"{r.coverage:.6f}",
                "n_covered": r.n_covered,
                "n_total": r.n_total,
                "n_peptide_hits": r.n_peptide_hits,
                "overall_confidence": f"{r.overall_confidence:.4f}",
                "seq_recovery": f"{r.seq_recovery:.4f}",
            })


# ── Directory discovery ────────────────────────────────────────────────────

def discover_conditions(mpnn_dir: str) -> List[dict]:
    """
    Find all completed MPNN experiment outputs.

    Expects: mpnn_dir/<group>/<condition_id>/seqs/*.fa

    Each condition directory may contain an experiment_config.json with the
    full parameter set (noise, temperature, bias_AA, etc.), written by
    run_mpnn_sweep.py.
    """
    conditions = []
    for group_dir in sorted(glob.glob(os.path.join(mpnn_dir, "*"))):
        if not os.path.isdir(group_dir):
            continue
        group = os.path.basename(group_dir)

        for cond_dir in sorted(glob.glob(os.path.join(group_dir, "*"))):
            if not os.path.isdir(cond_dir):
                continue
            condition_id = os.path.basename(cond_dir)
            seqs_dir = os.path.join(cond_dir, "seqs")
            fa_files = sorted(glob.glob(os.path.join(seqs_dir, "*.fa")))
            if not fa_files:
                continue

            # Load experiment config if available
            exp_config = {}
            config_path = os.path.join(cond_dir, "experiment_config.json")
            if os.path.isfile(config_path):
                try:
                    with open(config_path) as f:
                        exp_config = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            conditions.append({
                "condition_id": condition_id,
                "group": group,
                "fasta_path": fa_files[0],
                "exp_config": exp_config,
            })

    return conditions


# ── Worker wrapper for multiprocessing ─────────────────────────────────────

def _score_condition_worker(args_tuple):
    """Wrapper for process_condition that unpacks a single tuple argument.

    ProcessPoolExecutor.map requires a single-argument callable, so we pack
    all arguments into a tuple and unpack here.
    """
    (fasta_path, length_sets, condition_id, group, exp_config, top_n) = args_tuple
    return process_condition(
        fasta_path=fasta_path,
        length_sets=length_sets,
        condition_id=condition_id,
        group=group,
        exp_config=exp_config,
        top_n=top_n,
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score MPNN sequences for peptide library coverage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # X→N replacement (maximum coverage, treats glyco sites as normal Asn):
  python score_kmer_coverage.py --library 260120_mouse_peptides.txt \\
      --mpnn_dir mpnn_outputs --x_mode replace

  # X exclusion (conservative, only fully confirmed peptides):
  python score_kmer_coverage.py --library 260120_mouse_peptides.txt \\
      --mpnn_dir mpnn_outputs --x_mode exclude \\
      --output_dir coverage_results_exclude

  # Parallel scoring with 16 workers:
  python score_kmer_coverage.py --library 260120_mouse_peptides.txt \\
      --mpnn_dir mpnn_outputs --num_workers 16
        """,
    )
    parser.add_argument(
        "--library", required=True,
        help="Path to the peptide library (one peptide per line).",
    )
    parser.add_argument(
        "--mpnn_dir", required=True,
        help="Base directory of MPNN sweep outputs "
             "(expects <group>/<condition>/seqs/*.fa structure).",
    )
    parser.add_argument(
        "--output_dir", default="./coverage_results",
        help="Where to write results (default: ./coverage_results).",
    )
    parser.add_argument(
        "--x_mode", choices=["replace", "exclude"], default="replace",
        help="How to handle X in library peptides (default: replace).",
    )
    parser.add_argument(
        "--top_n", type=int, default=100,
        help="Number of top sequences to save per condition (default: 100).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of parallel workers for scoring conditions. "
             "Default: min(cpu_count, n_conditions). Use 1 for sequential.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load library ──
    logger.info(f"Loading peptide library from {args.library} (x_mode={args.x_mode})")
    t0 = time.time()
    length_sets, lib_stats = load_peptide_library(args.library, args.x_mode)
    min_len = lib_stats["min_length"]
    max_len = lib_stats["max_length"]
    logger.info(
        f"  {lib_stats['n_raw_peptides']:,} raw peptides "
        f"({lib_stats['n_with_x']:,} contain X)"
    )
    logger.info(
        f"  After {args.x_mode}: {lib_stats['n_unique']:,} unique peptides "
        f"(lengths {min_len}-{max_len}, {lib_stats['n_distinct_lengths']} distinct lengths)"
    )
    logger.info(f"  Length distribution: {lib_stats['length_distribution_final']}")
    logger.info(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Discover conditions ──
    conditions = discover_conditions(args.mpnn_dir)
    if not conditions:
        logger.error(f"No FASTA files found under {args.mpnn_dir}")
        sys.exit(1)

    n_with_config = sum(1 for c in conditions if c["exp_config"])
    logger.info(
        f"Found {len(conditions)} conditions to score "
        f"({n_with_config} with experiment_config.json)"
    )

    # ── Set up output directories ──
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "top_sequences"), exist_ok=True)

    # ── Score conditions (parallel or sequential) ──
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, len(conditions))
    num_workers = max(1, num_workers)

    all_summaries = []
    total_t0 = time.time()

    if num_workers == 1:
        # Sequential mode — simpler logging, easier to debug
        for i, cond in enumerate(conditions):
            cid = cond["condition_id"]
            logger.info(
                f"\n[{i+1}/{len(conditions)}] Scoring {cid} ({cond['group']})..."
            )

            summary, top_results = process_condition(
                fasta_path=cond["fasta_path"],
                length_sets=length_sets,
                condition_id=cid,
                group=cond["group"],
                exp_config=cond["exp_config"],
                top_n=args.top_n,
            )
            all_summaries.append(summary)
            _log_condition_result(summary)

            if top_results:
                top_path = os.path.join(
                    args.output_dir, "top_sequences",
                    f"{cid}_top{args.top_n}.csv",
                )
                write_top_sequences_csv(cid, top_results, top_path)
    else:
        # Parallel mode
        logger.info(f"Scoring with {num_workers} parallel workers")

        work_items = [
            (
                cond["fasta_path"],
                length_sets,
                cond["condition_id"],
                cond["group"],
                cond["exp_config"],
                args.top_n,
            )
            for cond in conditions
        ]

        # Map conditions to their index for ordering
        cond_map = {c["condition_id"]: c for c in conditions}

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_score_condition_worker, item): item[2]  # condition_id
                for item in work_items
            }

            completed = 0
            for future in as_completed(futures):
                cid = futures[future]
                completed += 1
                try:
                    summary, top_results = future.result()
                except Exception as e:
                    logger.error(f"[{cid}] FAILED: {e}")
                    continue

                all_summaries.append(summary)
                logger.info(f"[{completed}/{len(conditions)}] {cid}:")
                _log_condition_result(summary)

                if top_results:
                    top_path = os.path.join(
                        args.output_dir, "top_sequences",
                        f"{cid}_top{args.top_n}.csv",
                    )
                    write_top_sequences_csv(cid, top_results, top_path)

    # ── Write summary ──
    # Sort by group then condition_id for consistent output
    all_summaries.sort(key=lambda s: (s.group, s.condition_id))

    summary_path = os.path.join(args.output_dir, "condition_summary.csv")
    write_summary_csv(all_summaries, summary_path)

    meta_path = os.path.join(args.output_dir, "library_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "library_path": os.path.abspath(args.library),
            "x_mode": args.x_mode,
            "library_stats": lib_stats,
            "n_conditions": len(conditions),
            "num_workers": num_workers,
            "total_elapsed_s": time.time() - total_t0,
        }, f, indent=2)

    # ── Final comparison table ──
    total_elapsed = time.time() - total_t0
    logger.info(f"\n{'='*115}")
    logger.info(f"COVERAGE SCORING COMPLETE — {total_elapsed:.0f}s total")
    logger.info(f"Library: {lib_stats['n_unique']:,} peptides, x_mode={args.x_mode}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Results: {args.output_dir}")
    logger.info(f"{'='*115}")
    logger.info(
        f"{'Condition':<16} {'Group':<10} {'Noise':<6} {'Temp':<5} {'Bias':<8} "
        f"{'Mean':>7} {'Median':>7} {'Max':>7} {'P95':>7} {'P99':>7} "
        f"{'>90%':>7} {'>95%':>7} {'100%':>6} {'Hits':>6}"
    )
    logger.info("-" * 115)
    for s in sorted(all_summaries, key=lambda x: x.mean_coverage, reverse=True):
        bias_str = s.bias_AA if s.bias_AA else "-"
        logger.info(
            f"{s.condition_id:<16} {s.group:<10} "
            f"{s.noise:<6} {s.temperature:<5} {bias_str:<8} "
            f"{s.mean_coverage:>7.4f} {s.median_coverage:>7.4f} "
            f"{s.max_coverage:>7.4f} {s.p95:>7.4f} {s.p99:>7.4f} "
            f"{s.n_above_90:>7,} {s.n_above_95:>7,} {s.n_perfect:>6,} "
            f"{s.mean_peptide_hits:>6.0f}"
        )
    logger.info("=" * 115)


def _log_condition_result(summary: ConditionSummary):
    """Log a single condition's results."""
    logger.info(
        f"  n={summary.n_sequences:,}  "
        f"mean={summary.mean_coverage:.4f}  "
        f"median={summary.median_coverage:.4f}  "
        f"max={summary.max_coverage:.4f}  "
        f"p95={summary.p95:.4f}  "
        f"p99={summary.p99:.4f}  "
        f"hits={summary.mean_peptide_hits:.0f}  "
        f"[{summary.elapsed_s:.1f}s]"
    )
    logger.info(
        f"  native: coverage={summary.native_coverage:.4f}  "
        f"hits={summary.native_peptide_hits}"
    )
    logger.info(
        f"  >90%: {summary.n_above_90:,}  "
        f">95%: {summary.n_above_95:,}  "
        f">99%: {summary.n_above_99:,}  "
        f"perfect: {summary.n_perfect:,}"
    )


if __name__ == "__main__":
    main()
