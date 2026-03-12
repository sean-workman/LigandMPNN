#!/usr/bin/env python3
"""
generate_sweep_config.py — Generate experiment configuration JSON files for
run_mpnn_sweep.py by specifying parameter ranges from the command line.

Produces a full combinatorial grid over the specified noise levels,
temperatures, and bias values, with controllable batch size, number of
sequences, and seed strategy.

Examples:
  # Minimal: 3 noise levels x 2 temperatures, 1000 seqs each
  python generate_sweep_config.py \\
      --noise 0.10 0.20 0.30 \\
      --temperature 0.1 0.2 0.3 \\
      --num_seqs 1000 \\
      --batch_size 50 \\
      -o my_sweep.json

  # With Asn bias variations
  python generate_sweep_config.py \\
      --noise 0.10 0.20 \\
      --temperature 0.2 0.3 \\
      --bias_AA "" "N:3.0" "N:5.0" \\
      --num_seqs 100000 \\
      --batch_size 50 \\
      -o asn_sweep.json

  # Temperature range with step size
  python generate_sweep_config.py \\
      --noise 0.20 \\
      --temperature_range 0.1 0.5 0.1 \\
      --num_seqs 10000 \\
      --batch_size 50 \\
      -o temp_scan.json

  # Use ligand_mpnn instead of protein_mpnn
  python generate_sweep_config.py \\
      --model_type ligand_mpnn \\
      --noise 0.10 0.20 \\
      --temperature 0.2 \\
      --num_seqs 5000 \\
      --batch_size 50 \\
      -o ligand_sweep.json

  # Add fixed residues to all experiments
  python generate_sweep_config.py \\
      --noise 0.20 \\
      --temperature 0.2 0.3 \\
      --num_seqs 10000 \\
      --batch_size 50 \\
      --fixed_residues "A1 A2 A3 A10" \\
      -o fixed_sweep.json

  # Preview without writing (prints to stdout)
  python generate_sweep_config.py \\
      --noise 0.10 0.20 \\
      --temperature 0.2 \\
      --num_seqs 1000 \\
      --batch_size 50 \\
      --preview
"""
import argparse
import json
import math
import sys
from itertools import product


def frange(start: float, stop: float, step: float) -> list:
    """Generate a list of floats from start to stop (inclusive) by step."""
    values = []
    n_steps = round((stop - start) / step)
    for i in range(n_steps + 1):
        val = start + i * step
        # Round to avoid floating point drift
        val = round(val, 6)
        if val <= stop + 1e-9:
            values.append(val)
    return values


def noise_label(noise: str) -> str:
    """Convert noise string like '0.20' to short label like 'N20'."""
    # Remove leading '0.' and pad
    parts = noise.split(".")
    if len(parts) == 2:
        return f"N{parts[1]}"
    return f"N{noise.replace('.', '')}"


def temp_label(temp: float) -> str:
    """Convert temperature float to short label like 'T02' or 'T15'."""
    # Multiply by 10 to get a clean integer if possible
    t10 = temp * 10
    if t10 == int(t10):
        return f"T{int(t10):02d}"
    # For finer granularity, multiply by 100
    t100 = temp * 100
    if t100 == int(t100):
        return f"T{int(t100):03d}"
    return f"T{temp:.2f}".replace(".", "")


def bias_label(bias_str: str) -> str:
    """Convert bias string like 'N:3.0' to short label like 'N3'."""
    if not bias_str:
        return ""
    parts = []
    for entry in bias_str.split(","):
        entry = entry.strip()
        if ":" in entry:
            aa, val = entry.split(":", 1)
            # Clean up the value for the label
            val_clean = val.replace(".", "").replace("-", "m")
            parts.append(f"{aa}{val_clean}")
    return "_".join(parts)


def generate_experiments(
    model_types: list,
    noises: list,
    temperatures: list,
    biases: list,
    num_seqs: int,
    batch_size: int,
    base_seed: int,
    groups: dict,
    fixed_residues: str,
    redesigned_residues: str,
    omit_AA: str,
    chains_to_design: str,
) -> list:
    """Generate the combinatorial grid of experiments."""
    experiments = []
    seed_counter = base_seed

    for model_type, noise, temp, bias_aa in product(model_types, noises, temperatures, biases):
        # Compute number_of_batches from desired total sequences
        number_of_batches = math.ceil(num_seqs / batch_size)

        # Build experiment ID
        parts = [noise_label(noise), temp_label(temp)]
        bl = bias_label(bias_aa)
        if bl:
            parts.append(bl)
        if len(model_types) > 1:
            # Prefix with model type abbreviation when using multiple models
            mt_prefix = {"protein_mpnn": "PM", "ligand_mpnn": "LM",
                         "soluble_mpnn": "SM"}.get(model_type, model_type[:2].upper())
            parts.insert(0, mt_prefix)

        exp_id = "_".join(parts)

        # Determine group
        if bias_aa:
            group = groups.get("biased", f"bias_{bias_label(bias_aa).split('_')[0]}")
        else:
            group = groups.get("baseline", "baseline")

        exp = {
            "group": group,
            "id": exp_id,
            "model_type": model_type,
            "noise": noise,
            "temperature": temp,
            "bias_AA": bias_aa,
            "number_of_batches": number_of_batches,
            "batch_size": batch_size,
            "seed": seed_counter,
        }

        # Optional fields — only include if set
        if fixed_residues:
            exp["fixed_residues"] = fixed_residues
        if redesigned_residues:
            exp["redesigned_residues"] = redesigned_residues
        if omit_AA:
            exp["omit_AA"] = omit_AA
        if chains_to_design:
            exp["chains_to_design"] = chains_to_design

        # Auto-generate a human-readable note
        note_parts = [
            f"Noise {noise}",
            f"T={temp}",
        ]
        if bias_aa:
            note_parts.append(f"bias={bias_aa}")
        else:
            note_parts.append("no bias")
        note_parts.append(f"{number_of_batches * batch_size} seqs")
        exp["notes"] = ", ".join(note_parts)

        experiments.append(exp)
        seed_counter += 1

    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment config JSON for run_mpnn_sweep.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter ranges can be specified in two ways:
  --temperature 0.1 0.2 0.3         (explicit list of values)
  --temperature_range 0.1 0.5 0.1   (start stop step, inclusive)

The same applies to noise:
  --noise 0.10 0.20 0.30
  --noise_range 0.10 0.30 0.10

Both can be combined freely with --bias_AA to create the full grid.
        """,
    )

    # Core sweep axes
    parser.add_argument(
        "--model_type", nargs="+", default=["protein_mpnn"],
        choices=["protein_mpnn", "ligand_mpnn", "soluble_mpnn",
                 "per_residue_label_membrane_mpnn", "global_label_membrane_mpnn"],
        help="Model type(s) to sweep over (default: protein_mpnn).",
    )
    parser.add_argument(
        "--noise", nargs="+", default=None,
        help="Noise level(s) as strings, e.g. 0.10 0.20 0.30",
    )
    parser.add_argument(
        "--noise_range", nargs=3, type=float, default=None,
        metavar=("START", "STOP", "STEP"),
        help="Noise range: start stop step (inclusive). E.g. 0.10 0.30 0.10",
    )
    parser.add_argument(
        "--temperature", nargs="+", type=float, default=None,
        help="Temperature value(s), e.g. 0.1 0.2 0.3",
    )
    parser.add_argument(
        "--temperature_range", nargs=3, type=float, default=None,
        metavar=("START", "STOP", "STEP"),
        help="Temperature range: start stop step (inclusive). E.g. 0.1 0.5 0.1",
    )
    parser.add_argument(
        "--bias_AA", nargs="+", default=[""],
        help='Amino acid bias value(s). Use "" for no bias, "N:3.0" for Asn bias, etc. '
             'Multiple biases create additional grid points.',
    )

    # Sequence generation
    parser.add_argument(
        "--num_seqs", type=int, required=True,
        help="Target number of sequences per condition.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size (sequences per batch). When using run_mpnn_sweep.py "
             "with --auto_batch_size, this is overridden by GPU calibration "
             "and only affects number_of_batches in the config. Default: 1 "
             "(gives number_of_batches=num_seqs, auto-calibration adjusts both).",
    )
    parser.add_argument(
        "--base_seed", type=int, default=1001,
        help="Starting seed; increments by 1 for each condition (default: 1001).",
    )

    # Optional residue-level settings (applied to ALL experiments)
    parser.add_argument(
        "--fixed_residues", default="",
        help='Fixed residues applied to all experiments, e.g. "A1 A2 A3".',
    )
    parser.add_argument(
        "--redesigned_residues", default="",
        help='Redesigned residues applied to all experiments (everything else fixed).',
    )
    parser.add_argument(
        "--omit_AA", default="",
        help='Globally omitted amino acids for all experiments, e.g. "CM".',
    )
    parser.add_argument(
        "--chains_to_design", default="",
        help='Chains to design for all experiments, e.g. "A,B".',
    )

    # Group naming
    parser.add_argument(
        "--baseline_group", default="baseline",
        help="Group name for unbiased experiments (default: baseline).",
    )
    parser.add_argument(
        "--biased_group", default="biased",
        help="Group name for biased experiments (default: biased).",
    )

    # Metadata
    parser.add_argument(
        "--description", default="",
        help="Description string for the config metadata.",
    )
    parser.add_argument(
        "--target", default="",
        help="Target PDB filename for the config metadata.",
    )

    # Output
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output JSON file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Print a summary table instead of the full JSON.",
    )

    args = parser.parse_args()

    # Resolve noise values
    noises = []
    if args.noise:
        noises.extend(args.noise)
    if args.noise_range:
        start, stop, step = args.noise_range
        for val in frange(start, stop, step):
            noises.append(f"{val:.2f}")
    if not noises:
        parser.error("Must specify --noise or --noise_range")

    # Deduplicate while preserving order
    seen = set()
    unique_noises = []
    for n in noises:
        if n not in seen:
            seen.add(n)
            unique_noises.append(n)
    noises = unique_noises

    # Resolve temperature values
    temperatures = []
    if args.temperature:
        temperatures.extend(args.temperature)
    if args.temperature_range:
        start, stop, step = args.temperature_range
        temperatures.extend(frange(start, stop, step))
    if not temperatures:
        parser.error("Must specify --temperature or --temperature_range")

    seen = set()
    unique_temps = []
    for t in temperatures:
        if t not in seen:
            seen.add(t)
            unique_temps.append(t)
    temperatures = unique_temps

    # Clean up bias values (handle empty string for "no bias")
    biases = []
    for b in args.bias_AA:
        biases.append(b.strip('"').strip("'"))

    groups = {
        "baseline": args.baseline_group,
        "biased": args.biased_group,
    }

    experiments = generate_experiments(
        model_types=args.model_type,
        noises=noises,
        temperatures=temperatures,
        biases=biases,
        num_seqs=args.num_seqs,
        batch_size=args.batch_size,
        base_seed=args.base_seed,
        groups=groups,
        fixed_residues=args.fixed_residues,
        redesigned_residues=args.redesigned_residues,
        omit_AA=args.omit_AA,
        chains_to_design=args.chains_to_design,
    )

    n_batches = math.ceil(args.num_seqs / args.batch_size)
    total_seqs = len(experiments) * n_batches * args.batch_size

    if args.preview:
        print(f"{'ID':<25} {'Group':<12} {'Model':<15} {'Noise':<6} {'Temp':<5} {'Bias':<12} {'Seqs':>8} {'Seed':>6}")
        print("-" * 95)
        for exp in experiments:
            n = exp["batch_size"] * exp["number_of_batches"]
            bias = exp.get("bias_AA", "") or "-"
            print(
                f"{exp['id']:<25} {exp['group']:<12} {exp['model_type']:<15} "
                f"{exp['noise']:<6} {exp['temperature']:<5.2f} {bias:<12} "
                f"{n:>8,} {exp['seed']:>6}"
            )
        print("-" * 95)
        print(f"Total: {len(experiments)} experiments, {total_seqs:,} sequences")
        return

    config = {
        "_metadata": {
            "description": args.description or f"Sweep config: {len(experiments)} experiments",
            "target": args.target,
            "sequences_per_condition": n_batches * args.batch_size,
            "total_sequences": total_seqs,
            "total_experiments": len(experiments),
            "created": __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
            "generator_command": " ".join(sys.argv),
        },
        "experiments": experiments,
    }

    json_str = json.dumps(config, indent=4)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
            f.write("\n")
        print(f"Wrote {len(experiments)} experiments ({total_seqs:,} total seqs) to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
