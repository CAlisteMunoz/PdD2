import sys
import os
import torch
import argparse
from pathlib import Path
from torch.utils.data import random_split

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
from src.dataset import ClimateDataset

def main(args):
    print("--- Preparando Datos ---")
    RAW_DATA = PROJECT_ROOT / "data" / "raw" / "prw_*.nc"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    dataset = ClimateDataset(str(RAW_DATA), window_size=20, lag=50, step=2)
    
    # Split
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    
    # Guardar Training
    print("Guardando training set...")
    X_tr, Y_tr = [], []
    for i in train_set.indices:
        x, y = dataset[i]
        X_tr.append(x); Y_tr.append(y)
        
    torch.save({
        'X': torch.stack(X_tr), 'Y': torch.stack(Y_tr),
        'coords': {'lat': dataset.lat, 'lon': dataset.lon},
        'scaler': {'min': dataset.min_val, 'max': dataset.max_val},
        # GUARDAMOS LA CLIMATOLOGÍA (Importante para volver a valores reales)
        'climatology': torch.from_numpy(dataset.climatology.values).float()
    }, PROCESSED_DIR / "train.pt")
    
    # Guardar Validation
    print("Guardando validation set...")
    X_val, Y_val = [], []
    for i in val_set.indices:
        x, y = dataset[i]
        X_val.append(x); Y_val.append(y)
    
    torch.save({
        'X': torch.stack(X_val), 'Y': torch.stack(Y_val),
        'coords': {'lat': dataset.lat, 'lon': dataset.lon},
        'scaler': {'min': dataset.min_val, 'max': dataset.max_val},
        'climatology': torch.from_numpy(dataset.climatology.values).float()
    }, PROCESSED_DIR / "val.pt")
    
    print(" Datos de anomalías listos.")

if __name__ == "__main__":
    main(None)
