#!/bin/bash
#SBATCH --account=def-caveney-ab
#SBATCH --job-name=oval_sweep
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --output=oval_sweep_%j.out
#SBATCH --error=oval_sweep_%j.err

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11

source ~/envs/ligandmpnn/bin/activate

# --- PATHS (edit these) ---
LIGANDMPNN_DIR="$HOME/software/LigandMPNN"
INPUT_PDB="./AF-P01012-F1-model_v6.pdb"
OUTPUT_BASE="./ovalbumin_sweep_outputs"
CONFIG="./experiment_configs.json"

# --- GENERATE CONFIG ---
# 48 conditions: 4 noise x 4 temps x 3 biases = 24M total sequences
# batch_size=50 is a placeholder — auto_batch_size overrides it at runtime
echo "=== Generating experiment config ==="
python "$LIGANDMPNN_DIR/generate_sweep_config.py" \
    --noise 0.02 0.10 0.20 0.30 \
    --temperature 0.1 0.2 0.3 0.4 \
    --bias_AA "" "N:1.5" "N:3.0" \
    --num_seqs 500000 \
    --batch_size 50 \
    --base_seed 1001 \
    --baseline_group baseline \
    --biased_group asn_bias \
    --description "ProteinMPNN sweep for chicken ovalbumin (P01012) k-mer coverage exploration" \
    --target "AF-P01012-F1-model_v6.pdb" \
    -o "$CONFIG"

echo ""
echo "=== Preview ==="
python "$LIGANDMPNN_DIR/generate_sweep_config.py" \
    --noise 0.02 0.10 0.20 0.30 \
    --temperature 0.1 0.2 0.3 0.4 \
    --bias_AA "" "N:1.5" "N:3.0" \
    --num_seqs 500000 \
    --batch_size 50 \
    --preview

echo ""
echo "=== Running sweep with auto batch size ==="
python "$LIGANDMPNN_DIR/run_mpnn_sweep.py" \
    --config "$CONFIG" \
    --pdb "$INPUT_PDB" \
    --output_base "$OUTPUT_BASE" \
    --ligandmpnn_dir "$LIGANDMPNN_DIR" \
    --auto_batch_size \
    --memory_fraction 0.85

echo ""
echo "=== Done ==="
echo "Results in: $OUTPUT_BASE"
echo "Progress log: $OUTPUT_BASE/progress.json"
