#!/bin/bash
#SBATCH --account=def-caveney-ab
#SBATCH --job-name=ligandmpnn
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=3:00:00
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --output=ligandmpnn_%j.out
#SBATCH --error=ligandmpnn_%j.err

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11

source ~/envs/ligandmpnn/bin/activate

# --- PATHS ---

LIGANDMPNN_DIR="$HOME/software/LigandMPNN"

INPUT_PDB=""          # path to input PDB file
OUTPUT_DIR=""         # path to output directory

# Model checkpoint (see $LIGANDMPNN_DIR/model_params/ for available weights)
CHECKPOINT="$LIGANDMPNN_DIR/model_params/proteinmpnn_v_48_002.pt"

# Optional parameters (uncomment and set as needed)
# CHAINS=""           # chains to design, e.g. "A B"
# FIXED_RESIDUES=""   # residues to keep fixed, e.g. "A1 A2 A3"
# NUM_SEQS=8          # number of sequences to generate
# SEED=42
# TEMPERATURE=0.1

# --- RUN ---

python "$LIGANDMPNN_DIR/run.py" \
    --pdb_path "$INPUT_PDB" \
    --out_folder "$OUTPUT_DIR" \
    --checkpoint_protein_mpnn "$CHECKPOINT"
    # To add optional flags, remove the comment and add a backslash (\)
    # to the end of the PREVIOUS line. For example:
    #   --checkpoint_protein_mpnn "$CHECKPOINT" \
    #   --chains_to_design "$CHAINS"
