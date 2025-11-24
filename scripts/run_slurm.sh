#!/bin/bash
#SBATCH --job-name=clima_future
#SBATCH --output=/mnt/beegfs/home/caliste/Proyecto_Clima_ML/outputs/logs/future_%j.log
#SBATCH --error=/mnt/beegfs/home/caliste/Proyecto_Clima_ML/outputs/logs/future_err_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=ngen-ko

# --- CONFIGURACIÓN ---
PROJECT_ROOT="/mnt/beegfs/home/caliste/Proyecto_Clima_ML"
source ~/.bashrc
micromamba activate clima_ai
cd "$PROJECT_ROOT/scripts"

echo "=== INICIANDO PIPELINE DE PREDICCIÓN FUTURA ==="
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"

# 1. PREPARACIÓN (Genera train.pt y val.pt con ventanas deslizantes)
echo -e "\n>>> [1/3] PREPARANDO DATOS (80/20 SPLIT)..."
python prepare_data.py

if [ $? -ne 0 ]; then echo " Falló preparación"; exit 1; fi

# 2. ENTRENAMIENTO (Genera best_model.pth)
echo -e "\n>>> [2/3] ENTRENANDO MODELO..."
python train.py --epochs 3000 --lr 0.0001

if [ $? -ne 0 ]; then echo " Falló entrenamiento"; exit 1; fi

# 3. PREDICCIÓN FUTURA (Genera prediccion_futuro_2064.nc)
echo -e "\n>>> [3/3] PROYECTANDO EL FUTURO..."
python predict_future.py

if [ $? -ne 0 ]; then echo " Falló predicción"; exit 1; fi

echo -e "\n===  ¡PROCESO TERMINADO! ==="
echo "Revisa el archivo: outputs/prediccion_futuro_2064.nc"
