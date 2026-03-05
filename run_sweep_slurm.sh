#!/bin/bash
#SBATCH --account=def-caveney-ab
#SBATCH --job-name=mpnn_sweep
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --output=mpnn_sweep_%j.out
#SBATCH --error=mpnn_sweep_%j.err

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11

source ~/envs/ligandmpnn/bin/activate

# --- PATHS (edit these) ---
LIGANDMPNN_DIR="$HOME/software/LigandMPNN"
CONFIG="./experiment_configs.json"    # path to your sweep config JSON
INPUT_PDB="./structure.pdb"           # path to input PDB file
OUTPUT_BASE="./mpnn_outputs"          # base directory for all outputs

# --- RUN SWEEP ---
python "$LIGANDMPNN_DIR/run_mpnn_sweep.py" \
    --config "$CONFIG" \
    --pdb "$INPUT_PDB" \
    --output_base "$OUTPUT_BASE" \
    --ligandmpnn_dir "$LIGANDMPNN_DIR"

    # Optional flags (uncomment and add backslash to previous line):
    #   --groups baseline          # run only specific groups
    #   --ids N10_T02 N20_T03     # run only specific experiment IDs
    #   --dry_run                  # preview commands without running
    #   --keep_backbones           # don't delete backbone PDB files
