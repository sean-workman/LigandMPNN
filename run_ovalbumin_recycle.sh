#!/bin/bash
#SBATCH --account=def-caveney-ab
#SBATCH --job-name=oval_recycle
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --output=oval_recycle_%j.out
#SBATCH --error=oval_recycle_%j.err

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11

source ~/envs/ligandmpnn/bin/activate

# --- PATHS (edit these) ---
LIGANDMPNN_DIR="$HOME/software/LigandMPNN"
INPUT_PDB="./AF-P01012-F1-model_v6.pdb"
LIBRARY="./260120_mouse_peptides.txt"
OUTPUT_DIR="./ovalbumin_recycle_outputs"

echo "=== Iterative coverage recycling protocol ==="
echo "  PDB:     $INPUT_PDB"
echo "  Library: $LIBRARY"
echo "  Output:  $OUTPUT_DIR"
echo "  Resume:  will automatically continue from last checkpoint if present"
echo ""

python "$LIGANDMPNN_DIR/recycle_mpnn.py" \
    --pdb "$INPUT_PDB" \
    --library "$LIBRARY" \
    --output_dir "$OUTPUT_DIR" \
    --noise 0.30 \
    --temperature 0.1 \
    --bias_AA "N:3.0" \
    --num_seqs 1000000 \
    --batch_size 1000 \
    --max_rounds 50 \
    --patience 5 \
    --target_coverage 1.0 \
    --selection_metric total_coverage

echo ""
echo "=== Done ==="
echo "Results in: $OUTPUT_DIR"
echo "Progression: $OUTPUT_DIR/progression.json"
echo ""
echo "If not converged, resubmit to continue from checkpoint:"
echo "  sbatch run_ovalbumin_recycle.sh"
