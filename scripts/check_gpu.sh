#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --output=/mnt/beegfs/home/caliste/Proyecto_Clima_ML/outputs/logs/gpu_info.log
#SBATCH --partition=ngen-ko      # Usamos tu partición correcta
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             # ¡IMPORTANTE! Solicitamos 1 GPU
#SBATCH --time=00:05:00          # Solo necesitamos 5 minutos

echo "--- INFORMACIÓN DEL NODO ---"
hostname
echo "----------------------------"

echo "--- ESTADO DE NVIDIA (NVIDIA-SMI) ---"
# Este comando nos da la versión del driver y de CUDA
nvidia-smi

echo "----------------------------"
echo "--- COMPILADOR CUDA (NVCC) ---"
# Vemos si hay un módulo de cuda cargado por defecto
nvcc --version
