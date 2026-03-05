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

# --- FILL IN BELOW ---

INPUT_PDB=""          # path to input PDB file
OUTPUT_DIR=""         # path to output directory
MODEL_TYPE=""         # e.g. ligand_mpnn, soluble_mpnn, etc.
CHECKPOINT=""         # path to model checkpoint

# Optional parameters (uncomment and set as needed)
# CHAINS=""           # chains to design, e.g. "A B"
# FIXED_RESIDUES=""   # residues to keep fixed, e.g. "A1 A2 A3"
# NUM_SEQS=8          # number of sequences to generate
# SEED=42
# TEMPERATURE=0.1

# --- RUN ---

python run.py \
    --pdb_path "$INPUT_PDB" \
    --out_folder "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --checkpoint_path "$CHECKPOINT" \
    # --chains_to_design "$CHAINS" \
    # --fixed_residues "$FIXED_RESIDUES" \
    # --number_of_batches "$NUM_SEQS" \
    # --seed "$SEED" \
    # --temperature "$TEMPERATURE"
