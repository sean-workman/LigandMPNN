#!/usr/bin/env python3
"""
run_mpnn_sweep.py — Batch runner for LigandMPNN sequence generation experiments.

Reads experiment conditions from a JSON config file and runs LigandMPNN's
run.py for each condition sequentially.  Designed to be run from a working
directory separate from the LigandMPNN repository itself (e.g. inside a
SLURM job on the Fir cluster).

Features:
  - Resume capability: skips conditions whose output directories already
    contain a .fa file (so you can Ctrl-C and restart without re-running
    completed conditions).
  - Per-condition log files: stdout/stderr from each run.py invocation is
    captured to a log file alongside the outputs.
  - Progress tracking: writes a progress.json after each condition with
    timing and status information.
  - Dry-run mode: prints the commands that would be executed without
    actually running them.
  - Group/ID filtering: run only specific groups (e.g., --groups A B) or
    specific experiment IDs (e.g., --ids A01 A02 B05).
  - Automatic checkpoint resolution from model_type + noise level.
  - Auto batch size calibration: profiles GPU memory to find the largest
    batch size that fits, then adjusts number_of_batches to maintain the
    target sequence count.

Usage (inside a SLURM job or interactive session on Fir):
    module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11
    source ~/envs/ligandmpnn/bin/activate

    python run_mpnn_sweep.py \\
        --config experiment_configs.json \\
        --pdb /path/to/structure.pdb \\
        --output_base ./mpnn_outputs

    # Auto-optimize batch size for your GPU:
    python run_mpnn_sweep.py \\
        --config experiment_configs.json \\
        --pdb /path/to/structure.pdb \\
        --auto_batch_size

    # Dry run:
    python run_mpnn_sweep.py \\
        --config experiment_configs.json \\
        --pdb /path/to/structure.pdb \\
        --dry_run

    # Run only Group A:
    python run_mpnn_sweep.py \\
        --config experiment_configs.json \\
        --pdb /path/to/structure.pdb \\
        --groups A

    # Resume after interruption (automatically skips completed runs):
    python run_mpnn_sweep.py \\
        --config experiment_configs.json \\
        --pdb /path/to/structure.pdb
"""
import argparse
import glob
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_LIGANDMPNN_DIR = os.path.join(os.environ.get("HOME", "~"), "software", "LigandMPNN")

# Checkpoint flag for each model type (as expected by run.py)
CHECKPOINT_FLAG = {
    "protein_mpnn": "--checkpoint_protein_mpnn",
    "ligand_mpnn": "--checkpoint_ligand_mpnn",
    "soluble_mpnn": "--checkpoint_soluble_mpnn",
    "per_residue_label_membrane_mpnn": "--checkpoint_per_residue_label_membrane_mpnn",
    "global_label_membrane_mpnn": "--checkpoint_global_label_membrane_mpnn",
}

# Checkpoint filename patterns: (model_type, noise) -> filename
CHECKPOINT_FILENAME = {
    # ProteinMPNN
    ("protein_mpnn", "0.02"): "proteinmpnn_v_48_002.pt",
    ("protein_mpnn", "0.10"): "proteinmpnn_v_48_010.pt",
    ("protein_mpnn", "0.20"): "proteinmpnn_v_48_020.pt",
    ("protein_mpnn", "0.30"): "proteinmpnn_v_48_030.pt",
    # LigandMPNN (atom_context=25)
    ("ligand_mpnn", "0.05"): "ligandmpnn_v_32_005_25.pt",
    ("ligand_mpnn", "0.10"): "ligandmpnn_v_32_010_25.pt",
    ("ligand_mpnn", "0.20"): "ligandmpnn_v_32_020_25.pt",
    ("ligand_mpnn", "0.30"): "ligandmpnn_v_32_030_25.pt",
    # SolubleMPNN
    ("soluble_mpnn", "0.02"): "solublempnn_v_48_002.pt",
    ("soluble_mpnn", "0.10"): "solublempnn_v_48_010.pt",
    ("soluble_mpnn", "0.20"): "solublempnn_v_48_020.pt",
    ("soluble_mpnn", "0.30"): "solublempnn_v_48_030.pt",
    # Membrane models (only 0.20 available)
    ("per_residue_label_membrane_mpnn", "0.20"): "per_residue_label_membrane_mpnn_v_48_020.pt",
    ("global_label_membrane_mpnn", "0.20"): "global_label_membrane_mpnn_v_48_020.pt",
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    """Load and validate the experiment configuration JSON."""
    with open(config_path) as f:
        config = json.load(f)

    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in config file")

    required_fields = {
        "id", "model_type", "noise", "temperature",
        "number_of_batches", "batch_size", "seed",
    }
    for i, exp in enumerate(experiments):
        missing = required_fields - set(exp.keys())
        if missing:
            raise ValueError(
                f"Experiment index {i} (id={exp.get('id', '?')}) "
                f"missing required fields: {missing}"
            )
    return config


def resolve_checkpoint(model_type: str, noise: str, model_params_dir: str) -> str:
    """Resolve the full checkpoint path from model_type and noise level."""
    key = (model_type, noise)
    if key not in CHECKPOINT_FILENAME:
        available = [
            f"{mt} noise={n}"
            for (mt, n) in sorted(CHECKPOINT_FILENAME.keys())
            if mt == model_type
        ]
        raise ValueError(
            f"No checkpoint for model_type={model_type}, noise={noise}. "
            f"Available for {model_type}: {available or 'none (unknown model_type)'}"
        )
    return os.path.join(model_params_dir, CHECKPOINT_FILENAME[key])


def build_command(
    exp: dict,
    pdb_path: str,
    output_dir: str,
    run_script: str,
    model_params_dir: str,
) -> list:
    """
    Construct the LigandMPNN run.py command-line arguments for one experiment.
    """
    model_type = exp["model_type"]
    noise = exp["noise"]

    checkpoint_path = resolve_checkpoint(model_type, noise, model_params_dir)
    checkpoint_flag = CHECKPOINT_FLAG[model_type]

    cmd = [
        sys.executable, run_script,
        "--pdb_path", pdb_path,
        "--out_folder", output_dir,
        "--model_type", model_type,
        checkpoint_flag, checkpoint_path,
        "--temperature", str(exp["temperature"]),
        "--batch_size", str(exp["batch_size"]),
        "--number_of_batches", str(exp["number_of_batches"]),
        "--seed", str(exp["seed"]),
        "--save_stats", "0",
    ]

    # Optional string flags — only added when non-empty
    optional_str_flags = {
        "bias_AA": "--bias_AA",
        "bias_AA_per_residue": "--bias_AA_per_residue",
        "fixed_residues": "--fixed_residues",
        "redesigned_residues": "--redesigned_residues",
        "omit_AA": "--omit_AA",
        "omit_AA_per_residue": "--omit_AA_per_residue",
        "chains_to_design": "--chains_to_design",
        "symmetry_residues": "--symmetry_residues",
        "symmetry_weights": "--symmetry_weights",
    }
    for key, flag in optional_str_flags.items():
        val = exp.get(key, "")
        if val:
            cmd.extend([flag, str(val)])

    # Optional integer flags
    optional_int_flags = {
        "homo_oligomer": "--homo_oligomer",
        "save_stats": "--save_stats",
    }
    for key, flag in optional_int_flags.items():
        val = exp.get(key)
        if val is not None:
            cmd.extend([flag, str(val)])

    return cmd


def is_completed(output_dir: str) -> bool:
    """
    Check whether an experiment has already been completed.
    LigandMPNN writes .fa files to a seqs/ subdirectory under out_folder.
    """
    seqs_dir = os.path.join(output_dir, "seqs")
    if not os.path.isdir(seqs_dir):
        return False
    fa_files = glob.glob(os.path.join(seqs_dir, "*.fa"))
    return len(fa_files) > 0


def count_sequences_in_fasta(fasta_path: str) -> int:
    """Count the number of sequence entries in a FASTA file."""
    count = 0
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def summarize_completed(output_dir: str) -> dict:
    """Gather summary statistics from a completed run."""
    summary = {"n_sequences": 0, "has_stats": False, "fasta_files": []}
    seqs_dir = os.path.join(output_dir, "seqs")
    if os.path.isdir(seqs_dir):
        for fa in sorted(glob.glob(os.path.join(seqs_dir, "*.fa"))):
            n = count_sequences_in_fasta(fa)
            summary["fasta_files"].append(os.path.basename(fa))
            summary["n_sequences"] += n
    stats_dir = os.path.join(output_dir, "stats")
    summary["has_stats"] = os.path.isdir(stats_dir) and bool(
        os.listdir(stats_dir)
    )
    return summary


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


# ── Main execution logic ───────────────────────────────────────────────────
def run_experiment(
    exp: dict,
    pdb_path: str,
    output_base: str,
    logger: logging.Logger,
    dry_run: bool,
    run_script: str,
    model_params_dir: str,
    ligandmpnn_dir: str,
    clean_backbones: bool = True,
) -> dict:
    """
    Run a single experiment condition.
    Returns a result dict with timing and status information.
    """
    exp_id = exp["id"]
    group = exp.get("group", "ungrouped")
    n_seqs = exp["batch_size"] * exp["number_of_batches"]

    # Each experiment gets its own subdirectory: output_base/group/id/
    output_dir = os.path.abspath(os.path.join(output_base, group, exp_id))

    # Check for prior completion (resume support)
    if is_completed(output_dir):
        summary = summarize_completed(output_dir)
        logger.info(
            f"[{exp_id}] SKIPPED (already complete: "
            f"{summary['n_sequences']} sequences found)"
        )
        return {
            "id": exp_id,
            "status": "skipped",
            "n_sequences": summary["n_sequences"],
            "duration_s": 0,
        }

    # Build the command
    cmd = build_command(exp, pdb_path, output_dir, run_script, model_params_dir)

    # Descriptive label for logging
    bias_str = exp.get("bias_AA", "none") or "none"
    label = (
        f"{exp['model_type']} | noise={exp['noise']} | "
        f"T={exp['temperature']} | bias={bias_str} | "
        f"n={n_seqs}"
    )

    if dry_run:
        logger.info(f"[{exp_id}] DRY RUN: {label}")
        logger.info(f"  Command: {' '.join(cmd)}")
        return {"id": exp_id, "status": "dry_run", "duration_s": 0}

    # Create output directory and log file
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{exp_id}_run.log")

    logger.info(f"[{exp_id}] STARTING: {label}")
    logger.info(f"  Output: {output_dir}")

    # Save the experiment config alongside the outputs for traceability
    config_copy_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_copy_path, "w") as f:
        json.dump(exp, f, indent=2)

    # Run LigandMPNN, capturing stdout/stderr to the log file
    t0 = time.time()
    with open(log_path, "w") as log_f:
        log_f.write(f"# Experiment: {exp_id}\n")
        log_f.write(f"# Label: {label}\n")
        log_f.write(f"# Started: {datetime.now().isoformat()}\n")
        log_f.write(f"# Command: {' '.join(cmd)}\n")
        log_f.write(f"{'=' * 70}\n\n")
        log_f.flush()

        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=ligandmpnn_dir,
        )
    duration = time.time() - t0

    if result.returncode != 0:
        logger.error(
            f"[{exp_id}] FAILED (return code {result.returncode}) "
            f"after {format_duration(duration)}. See {log_path}"
        )
        return {
            "id": exp_id,
            "status": "failed",
            "returncode": result.returncode,
            "duration_s": duration,
            "log": log_path,
        }

    # Verify outputs exist
    summary = summarize_completed(output_dir)

    # Clean up backbone PDB files (they're just the input backbone with
    # the new sequence threaded on — not actual structure predictions)
    if clean_backbones:
        backbones_dir = os.path.join(output_dir, "backbones")
        if os.path.isdir(backbones_dir):
            n_pdbs = len(glob.glob(os.path.join(backbones_dir, "*.pdb")))
            shutil.rmtree(backbones_dir)
            logger.info(f"  Cleaned up {n_pdbs} backbone PDBs")

    logger.info(
        f"[{exp_id}] COMPLETE: {summary['n_sequences']} sequences in "
        f"{format_duration(duration)} "
        f"(stats={'yes' if summary['has_stats'] else 'no'})"
    )
    return {
        "id": exp_id,
        "status": "completed",
        "n_sequences": summary["n_sequences"],
        "duration_s": duration,
        "has_stats": summary["has_stats"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for LigandMPNN sequence generation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (activate conda/venv first):
  python run_mpnn_sweep.py --config experiment_configs.json --pdb ./structure.pdb

  # Dry run to check commands:
  python run_mpnn_sweep.py --config experiment_configs.json --pdb ./structure.pdb --dry_run

  # Run only specific groups:
  python run_mpnn_sweep.py --config experiment_configs.json --pdb ./structure.pdb --groups baseline asn_bias

  # Run only specific experiments:
  python run_mpnn_sweep.py --config experiment_configs.json --pdb ./structure.pdb --ids N10_T02 N20_T03_N3

  # Keep backbone PDB files:
  python run_mpnn_sweep.py --config experiment_configs.json --pdb ./structure.pdb --keep_backbones

  # Auto-optimize batch size (profiles GPU, adjusts batches to keep same total seqs):
  python run_mpnn_sweep.py --config experiment_configs.json --pdb ./structure.pdb --auto_batch_size

  # Auto batch size with custom memory fraction:
  python run_mpnn_sweep.py --config experiment_configs.json --pdb ./structure.pdb --auto_batch_size --memory_fraction 0.80
        """,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the experiment configuration JSON file.",
    )
    parser.add_argument(
        "--pdb", required=True,
        help="Path to the target PDB file.",
    )
    parser.add_argument(
        "--output_base", default="./mpnn_outputs",
        help="Base directory for all experiment outputs (default: ./mpnn_outputs).",
    )
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Only run experiments from these groups (e.g., baseline asn_bias).",
    )
    parser.add_argument(
        "--ids", nargs="+", default=None,
        help="Only run these specific experiment IDs (e.g., N10_T02 N20_T03_N3).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--keep_backbones", action="store_true",
        help="Keep backbone PDB files instead of cleaning them up after each run.",
    )
    parser.add_argument(
        "--auto_batch_size", action="store_true",
        help="Auto-calibrate batch size by profiling GPU memory. "
             "Overrides batch_size in all experiments and adjusts "
             "number_of_batches to maintain the same total sequence count.",
    )
    parser.add_argument(
        "--memory_fraction", type=float, default=0.85,
        help="Fraction of GPU memory to target when using --auto_batch_size "
             "(default: 0.85). Lower values are safer for long runs.",
    )
    parser.add_argument(
        "--ligandmpnn_dir", default=DEFAULT_LIGANDMPNN_DIR,
        help=f"Path to the LigandMPNN repository (default: {DEFAULT_LIGANDMPNN_DIR}).",
    )

    args = parser.parse_args()

    # Resolve LigandMPNN paths
    ligandmpnn_dir = os.path.abspath(args.ligandmpnn_dir)
    run_script = os.path.join(ligandmpnn_dir, "run.py")
    model_params_dir = os.path.join(ligandmpnn_dir, "model_params")

    # Validate paths
    if not os.path.isfile(args.pdb):
        print(f"ERROR: PDB file not found: {args.pdb}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(run_script):
        print(f"ERROR: LigandMPNN run.py not found: {run_script}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(model_params_dir):
        print(f"ERROR: model_params directory not found: {model_params_dir}", file=sys.stderr)
        print("  Run: bash get_model_params.sh ./model_params", file=sys.stderr)
        sys.exit(1)

    # Set up logging — both to console and to a master log file
    os.makedirs(args.output_base, exist_ok=True)
    log_path = os.path.join(args.output_base, "sweep_master.log")

    logger = logging.getLogger("mpnn_sweep")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Load config
    config = load_config(args.config)
    experiments = config["experiments"]

    # Filter by group and/or ID
    if args.groups:
        experiments = [e for e in experiments if e.get("group") in args.groups]
    if args.ids:
        experiments = [e for e in experiments if e["id"] in args.ids]

    if not experiments:
        logger.error("No experiments match the specified filters")
        sys.exit(1)

    # Validate all checkpoints exist before starting
    for exp in experiments:
        ckpt = resolve_checkpoint(exp["model_type"], exp["noise"], model_params_dir)
        if not args.dry_run and not os.path.isfile(ckpt):
            logger.error(
                f"Checkpoint not found for {exp['id']}: {ckpt}\n"
                f"  Run: bash get_model_params.sh {model_params_dir}"
            )
            sys.exit(1)

    # Auto batch size calibration
    auto_bs_result = None
    if args.auto_batch_size and not args.dry_run:
        # Add LigandMPNN dir to path so auto_batch_size can import its modules
        if ligandmpnn_dir not in sys.path:
            sys.path.insert(0, ligandmpnn_dir)

        from auto_batch_size import calibrate_batch_size

        # Use the first experiment's model_type/noise for calibration
        # (the model architecture is the same across noise levels, and
        # protein_mpnn vs soluble_mpnn have the same memory footprint)
        first_exp = experiments[0]
        cal_checkpoint = resolve_checkpoint(
            first_exp["model_type"], first_exp["noise"], model_params_dir
        )

        logger.info("=" * 70)
        logger.info("Auto batch size calibration")
        logger.info(f"  Memory target: {args.memory_fraction:.0%} of GPU VRAM")

        auto_bs_result = calibrate_batch_size(
            pdb_path=os.path.abspath(args.pdb),
            checkpoint_path=cal_checkpoint,
            model_type=first_exp["model_type"],
            memory_fraction=args.memory_fraction,
            verbose=True,
        )
        optimal_bs = auto_bs_result["batch_size"]

        logger.info(f"  Calibrated batch_size: {optimal_bs}")
        logger.info(f"  GPU: {auto_bs_result['gpu_name']}")
        logger.info(f"  Protein: {auto_bs_result['protein_length']} residues")
        logger.info(f"  Base memory: {auto_bs_result['base_memory_mb']:.0f} MB")
        logger.info(f"  Per-sample: {auto_bs_result['per_sample_mb']:.1f} MB")

        # Override batch_size in all experiments, adjusting number_of_batches
        # to maintain the same total sequence count
        for exp in experiments:
            target_seqs = exp["batch_size"] * exp["number_of_batches"]
            exp["batch_size"] = optimal_bs
            exp["number_of_batches"] = math.ceil(target_seqs / optimal_bs)
            actual_seqs = exp["batch_size"] * exp["number_of_batches"]
            if actual_seqs != target_seqs:
                logger.info(
                    f"  [{exp['id']}] {target_seqs:,} -> {actual_seqs:,} seqs "
                    f"(batch_size={optimal_bs}, batches={exp['number_of_batches']})"
                )

        logger.info("=" * 70)

    # Compute totals for progress reporting
    total_seqs = sum(e["batch_size"] * e["number_of_batches"] for e in experiments)
    total_exps = len(experiments)

    logger.info("=" * 70)
    logger.info("LigandMPNN Sweep Runner")
    logger.info(f"  Config:       {args.config}")
    logger.info(f"  PDB:          {args.pdb}")
    logger.info(f"  Output base:  {args.output_base}")
    logger.info(f"  LigandMPNN:   {ligandmpnn_dir}")
    logger.info(f"  Experiments:  {total_exps}")
    logger.info(f"  Total seqs:   {total_seqs:,}")
    if auto_bs_result:
        logger.info(f"  Batch size:   {auto_bs_result['batch_size']} (auto-calibrated)")
    logger.info(f"  Dry run:      {args.dry_run}")
    logger.info("=" * 70)

    pdb_abs = os.path.abspath(args.pdb)

    all_results = []
    completed_seqs = 0
    failed = 0
    skipped = 0
    sweep_t0 = time.time()

    progress_path = os.path.join(args.output_base, "progress.json")

    for i, exp in enumerate(experiments):
        elapsed = time.time() - sweep_t0
        if completed_seqs > 0 and elapsed > 0:
            rate = completed_seqs / elapsed
            remaining_seqs = total_seqs - completed_seqs
            eta = remaining_seqs / rate if rate > 0 else 0
            eta_str = format_duration(eta)
        else:
            eta_str = "calculating..."

        logger.info(
            f"\n--- Experiment {i+1}/{total_exps} "
            f"({completed_seqs:,}/{total_seqs:,} seqs done, "
            f"ETA: {eta_str}) ---"
        )

        result = run_experiment(
            exp,
            pdb_abs,
            args.output_base,
            logger,
            dry_run=args.dry_run,
            run_script=run_script,
            model_params_dir=model_params_dir,
            ligandmpnn_dir=ligandmpnn_dir,
            clean_backbones=not args.keep_backbones,
        )
        all_results.append(result)

        if result["status"] == "completed":
            completed_seqs += result.get("n_sequences", 0)
        elif result["status"] == "skipped":
            completed_seqs += result.get("n_sequences", 0)
            skipped += 1
        elif result["status"] == "failed":
            failed += 1

        progress = {
            "last_updated": datetime.now().isoformat(),
            "total_experiments": total_exps,
            "completed": sum(1 for r in all_results if r["status"] == "completed"),
            "skipped": skipped,
            "failed": failed,
            "remaining": total_exps - (i + 1),
            "total_sequences_generated": completed_seqs,
            "elapsed_s": time.time() - sweep_t0,
            "auto_batch_size": auto_bs_result,
            "results": all_results,
        }
        if not args.dry_run:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)

    total_elapsed = time.time() - sweep_t0
    logger.info("\n" + "=" * 70)
    logger.info("SWEEP COMPLETE")
    logger.info(f"  Total time:   {format_duration(total_elapsed)}")
    logger.info(f"  Completed:    {sum(1 for r in all_results if r['status'] == 'completed')}")
    logger.info(f"  Skipped:      {skipped}")
    logger.info(f"  Failed:       {failed}")
    logger.info(f"  Sequences:    {completed_seqs:,}")
    logger.info("=" * 70)

    if failed > 0:
        logger.warning(
            f"{failed} experiments failed. Check individual log files "
            f"in the output directories for details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
