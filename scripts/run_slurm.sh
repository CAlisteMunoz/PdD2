#!/bin/bash
#SBATCH --job-name=clima_ml
#SBATCH --output=logs/gpu_%j.out
#SBATCH --error=logs/gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --partition=ngen-ko

export TMPDIR=/tmp
PYTHON_EXEC="$HOME/micromamba/envs/clima_ai/bin/python"
PROJECT_DIR="$HOME/Proyecto_Clima_ML"
SCRIPT_DIR="$PROJECT_DIR/scripts"
SRC_DIR="$PROJECT_DIR/src"

cd "$SCRIPT_DIR" || exit 1

echo "=========================================="
echo " JOB GPU A100: $(hostname)"
echo "=========================================="

echo ">>> [1] VERIFICANDO GPU (NVIDIA-SMI):"
nvidia-smi

echo ">>> [2] VERIFICANDO PYTORCH (CUDA):"
"$PYTHON_EXEC" "$SRC_DIR/check_gpu.py"

echo ">>> [3] INICIANDO ENTRENAMIENTO (Batch 32)..."
echo "--- CNN ---"
"$PYTHON_EXEC" -u 03_train.py --model cnn --epochs 30 --batch_size 16 --val_split 0.2

echo "--- MLP ---"
"$PYTHON_EXEC" -u 03_train.py --model mlp --epochs 30 --batch_size 16 --val_split 0.2
