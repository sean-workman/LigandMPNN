#!/usr/bin/env python3
"""
auto_batch_size.py — Automatically determine the optimal batch size for
LigandMPNN based on GPU memory and protein size.

Strategy:
  1. Load model and featurize the protein (same as run.py).
  2. Run a single sample at B=1 to measure baseline GPU memory.
  3. Run a sample at B=2 to measure the per-sample memory increment.
  4. Extrapolate the maximum batch size that fits in available VRAM,
     with a configurable safety margin.
  5. Optionally validate by running one batch at the computed size.

Can be used as a library (call calibrate_batch_size()) or as a CLI tool.

Usage:
    # Standalone — prints recommended batch size:
    python auto_batch_size.py \\
        --pdb ./structure.pdb \\
        --model_type protein_mpnn \\
        --checkpoint ./model_params/proteinmpnn_v_48_020.pt

    # With safety margin (default 0.85 = use 85% of free VRAM):
    python auto_batch_size.py \\
        --pdb ./structure.pdb \\
        --model_type protein_mpnn \\
        --checkpoint ./model_params/proteinmpnn_v_48_020.pt \\
        --memory_fraction 0.80

    # JSON output for scripting:
    python auto_batch_size.py \\
        --pdb ./structure.pdb \\
        --model_type protein_mpnn \\
        --checkpoint ./model_params/proteinmpnn_v_48_020.pt \\
        --json
"""
import argparse
import json
import sys

import numpy as np
import torch


def get_gpu_memory_info(device: torch.device) -> dict:
    """Get GPU memory information in bytes."""
    if device.type != "cuda":
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}

    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = total - reserved
    return {
        "total": total,
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "device_name": torch.cuda.get_device_properties(device).name,
    }


def _bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def _bytes_to_gb(b: int) -> float:
    return b / (1024 * 1024 * 1024)


def load_model_and_protein(
    pdb_path: str,
    checkpoint_path: str,
    model_type: str = "protein_mpnn",
    device: torch.device = None,
    ligand_mpnn_use_side_chain_context: int = 0,
    ligand_mpnn_use_atom_context: int = 1,
    ligand_mpnn_cutoff_for_score: float = 8.0,
):
    """
    Load model and parse/featurize a PDB, returning the objects needed
    for a trial sample call. Mirrors the setup in run.py lines 54-401.
    """
    # Import LigandMPNN modules (must be on sys.path or run from repo root)
    from data_utils import featurize, parse_PDB
    from model_utils import ProteinMPNN

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        k_neighbors = checkpoint["num_edges"]
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
        k_neighbors = checkpoint["num_edges"]

    # Build model
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Parse PDB
    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
        pdb_path,
        device=device,
        chains=[],
        parse_all_atoms=(model_type == "ligand_mpnn"),
        parse_atoms_with_zero_occupancy=False,
    )

    # Featurize
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=ligand_mpnn_cutoff_for_score,
        use_atom_context=ligand_mpnn_use_atom_context,
        number_of_ligand_atoms=atom_context_num,
        model_type=model_type,
    )

    # Add required keys (mirrors run.py lines 402-412)
    B, L, _, _ = feature_dict["X"].shape
    feature_dict["temperature"] = 0.1
    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    omit_AA = torch.zeros([21], device=device, dtype=torch.float32)
    feature_dict["bias"] = bias_AA.repeat([1, L, 1])
    feature_dict["symmetry_residues"] = [[]]
    feature_dict["symmetry_weights"] = [[]]

    return model, feature_dict, L, k_neighbors


def _run_trial(model, feature_dict, batch_size: int, device: torch.device):
    """Run a single trial sample and return peak GPU memory allocated."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    feature_dict["batch_size"] = batch_size
    L = feature_dict["mask"].shape[1]
    feature_dict["randn"] = torch.randn([batch_size, L], device=device)

    with torch.no_grad():
        _ = model.sample(feature_dict)

    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)

    # Clean up
    del feature_dict["randn"]
    torch.cuda.empty_cache()

    return peak


def calibrate_batch_size(
    pdb_path: str,
    checkpoint_path: str,
    model_type: str = "protein_mpnn",
    memory_fraction: float = 0.85,
    min_batch_size: int = 1,
    max_batch_size: int = 2048,
    validate: bool = True,
    verbose: bool = True,
    device: torch.device = None,
    ligand_mpnn_use_side_chain_context: int = 0,
    ligand_mpnn_use_atom_context: int = 1,
    ligand_mpnn_cutoff_for_score: float = 8.0,
) -> dict:
    """
    Determine optimal batch size by profiling GPU memory usage.

    Args:
        pdb_path: Path to input PDB file.
        checkpoint_path: Path to model checkpoint.
        model_type: Model type string.
        memory_fraction: Fraction of total GPU VRAM to target (0.0-1.0).
        min_batch_size: Minimum batch size to return.
        max_batch_size: Maximum batch size to return.
        validate: If True, run a validation pass at the computed batch size.
        verbose: Print progress information.
        device: Torch device (auto-detected if None).
        ligand_mpnn_use_side_chain_context: Use side chain context (ligand_mpnn only).
        ligand_mpnn_use_atom_context: Use atom context (ligand_mpnn only).
        ligand_mpnn_cutoff_for_score: Cutoff distance for scoring.

    Returns:
        dict with keys:
            batch_size: Recommended batch size.
            gpu_name: GPU device name.
            gpu_total_mb: Total GPU memory in MB.
            protein_length: Number of residues.
            k_neighbors: Number of edge neighbors from checkpoint.
            base_memory_mb: Memory at B=1 in MB.
            per_sample_mb: Additional memory per sample in MB.
            target_memory_mb: Target memory budget in MB.
            memory_fraction: Fraction used.
            validated: Whether the batch size was validated.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        if verbose:
            print("No CUDA device available. Returning batch_size=1.")
        return {
            "batch_size": 1,
            "gpu_name": "cpu",
            "gpu_total_mb": 0,
            "protein_length": 0,
            "k_neighbors": 0,
            "base_memory_mb": 0,
            "per_sample_mb": 0,
            "target_memory_mb": 0,
            "memory_fraction": memory_fraction,
            "validated": False,
        }

    gpu_info = get_gpu_memory_info(device)
    gpu_name = gpu_info["device_name"]
    gpu_total = gpu_info["total"]

    if verbose:
        print(f"GPU: {gpu_name} ({_bytes_to_gb(gpu_total):.1f} GB)")
        print(f"Loading model and protein...")

    # Load model and protein
    model, feature_dict, L, k_neighbors = load_model_and_protein(
        pdb_path=pdb_path,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
        ligand_mpnn_use_atom_context=ligand_mpnn_use_atom_context,
        ligand_mpnn_cutoff_for_score=ligand_mpnn_cutoff_for_score,
    )

    if verbose:
        print(f"Protein length: {L} residues, K={k_neighbors} neighbors")

    # Trial at B=1 (warmup + baseline measurement)
    if verbose:
        print("Profiling at batch_size=1...")
    # Warmup pass to trigger any lazy CUDA initialization
    _run_trial(model, feature_dict, 1, device)
    mem_b1 = _run_trial(model, feature_dict, 1, device)

    if verbose:
        print(f"  Peak memory at B=1: {_bytes_to_mb(mem_b1):.1f} MB")

    # Trial at B=2 to measure per-sample increment
    if verbose:
        print("Profiling at batch_size=2...")
    mem_b2 = _run_trial(model, feature_dict, 2, device)

    per_sample = mem_b2 - mem_b1  # memory delta for one additional sample

    if verbose:
        print(f"  Peak memory at B=2: {_bytes_to_mb(mem_b2):.1f} MB")
        print(f"  Per-sample increment: {_bytes_to_mb(per_sample):.1f} MB")

    # Handle edge case where per_sample is very small or negative (unlikely)
    if per_sample <= 0:
        if verbose:
            print("  Warning: per-sample increment <= 0, using conservative estimate")
        # Fall back to a conservative estimate: assume B=2 memory / 3
        per_sample = mem_b2 // 3

    # Calculate optimal batch size
    target_memory = int(gpu_total * memory_fraction)
    available_for_batches = target_memory - mem_b1 + per_sample  # B=1 already includes one sample
    optimal_batch_size = max(min_batch_size, int(available_for_batches / per_sample))
    optimal_batch_size = min(optimal_batch_size, max_batch_size)

    if verbose:
        print(f"\nTarget memory budget: {_bytes_to_mb(target_memory):.0f} MB "
              f"({memory_fraction:.0%} of {_bytes_to_gb(gpu_total):.1f} GB)")
        print(f"Estimated optimal batch_size: {optimal_batch_size}")

    # Validate with actual run at computed batch size
    validated = False
    if validate and optimal_batch_size > 2:
        if verbose:
            print(f"Validating at batch_size={optimal_batch_size}...")
        try:
            mem_opt = _run_trial(model, feature_dict, optimal_batch_size, device)
            actual_usage = _bytes_to_mb(mem_opt)
            target_mb = _bytes_to_mb(target_memory)

            if mem_opt > target_memory:
                # Overshot — scale back based on actual measurement
                actual_per_sample = (mem_opt - mem_b1) / (optimal_batch_size - 1)
                available = target_memory - mem_b1 + actual_per_sample
                optimal_batch_size = max(min_batch_size, int(available / actual_per_sample))
                if verbose:
                    print(f"  Actual peak: {actual_usage:.0f} MB > target {target_mb:.0f} MB")
                    print(f"  Adjusted batch_size to {optimal_batch_size}")
            else:
                validated = True
                if verbose:
                    print(f"  Actual peak: {actual_usage:.0f} MB <= target {target_mb:.0f} MB — OK")

        except torch.cuda.OutOfMemoryError:
            # OOM at computed size — binary search downward
            if verbose:
                print(f"  OOM at batch_size={optimal_batch_size}, searching downward...")
            torch.cuda.empty_cache()

            low, high = min_batch_size, optimal_batch_size - 1
            best = min_batch_size
            while low <= high:
                mid = (low + high) // 2
                try:
                    torch.cuda.empty_cache()
                    mem_mid = _run_trial(model, feature_dict, mid, device)
                    if mem_mid <= target_memory:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    high = mid - 1

            optimal_batch_size = best
            validated = True
            if verbose:
                print(f"  Binary search found batch_size={optimal_batch_size}")

    # Clean up
    del model, feature_dict
    torch.cuda.empty_cache()

    result = {
        "batch_size": optimal_batch_size,
        "gpu_name": gpu_name,
        "gpu_total_mb": round(_bytes_to_mb(gpu_total), 1),
        "protein_length": L,
        "k_neighbors": k_neighbors,
        "base_memory_mb": round(_bytes_to_mb(mem_b1), 1),
        "per_sample_mb": round(_bytes_to_mb(per_sample), 1),
        "target_memory_mb": round(_bytes_to_mb(target_memory), 1),
        "memory_fraction": memory_fraction,
        "validated": validated,
    }

    if verbose:
        print(f"\nRecommended batch_size: {optimal_batch_size}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Auto-calibrate LigandMPNN batch size for your GPU and protein.",
    )
    parser.add_argument("--pdb", required=True, help="Path to input PDB file.")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pt)."
    )
    parser.add_argument(
        "--model_type", default="protein_mpnn",
        choices=["protein_mpnn", "ligand_mpnn", "soluble_mpnn",
                 "per_residue_label_membrane_mpnn", "global_label_membrane_mpnn"],
        help="Model type (default: protein_mpnn).",
    )
    parser.add_argument(
        "--memory_fraction", type=float, default=0.85,
        help="Fraction of GPU memory to target (default: 0.85).",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=2048,
        help="Upper limit on batch size (default: 2048).",
    )
    parser.add_argument(
        "--no_validate", action="store_true",
        help="Skip validation pass (faster but less reliable).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output result as JSON (for scripting).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output (implies --json).",
    )

    args = parser.parse_args()

    verbose = not args.quiet
    result = calibrate_batch_size(
        pdb_path=args.pdb,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        memory_fraction=args.memory_fraction,
        max_batch_size=args.max_batch_size,
        validate=not args.no_validate,
        verbose=verbose,
    )

    if args.json or args.quiet:
        print(json.dumps(result, indent=2))
    elif not verbose:
        print(result["batch_size"])


if __name__ == "__main__":
    main()
