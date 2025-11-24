import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score

# Configurar path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
from src.model import ClimateEmulator

def main(args):
    # Rutas
    TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.pt"
    VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.pt"
    MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
    LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    MODEL_SAVE_PATH = MODELS_DIR / "best_model.pth"

    # Cargar datos
    train_data = torch.load(TRAIN_FILE)
    val_data = torch.load(VAL_FILE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    X_train = train_data['X'].to(device)
    Y_train = train_data['Y'].to(device)
    X_val = val_data['X'].to(device)
    Y_val = val_data['Y'].to(device)
    
    # Modelo
    model = ClimateEmulator(input_size=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    metrics = []
    best_val_loss = float('inf')
    
    print(f"Iniciando entrenamiento de {args.epochs} épocas...")
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, Y_train)
        loss.backward()
        optimizer.step()
        
        # Validación y Guardado (Cada 10 épocas)
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, Y_val)
                
                # Calcular R2 en validación
                r2 = r2_score(Y_val.cpu().numpy().flatten(), val_pred.cpu().numpy().flatten())
            
            # Guardar el mejor modelo basado en validación
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
            # Log en pantalla cada 100 épocas
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | R2: {r2:.4f}")
            
            metrics.append({
                'epoch': epoch+1, 
                'train_loss': loss.item(), 
                'val_loss': val_loss.item(), 
                'r2': r2
            })
            
    # Guardar historial CSV
    pd.DataFrame(metrics).to_csv(LOGS_DIR / "training_metrics.csv", index=False)
    print(f" Entrenamiento finalizado. Mejor Loss: {best_val_loss:.6f}")
    print(f"Modelo guardado en: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()
    main(args)
