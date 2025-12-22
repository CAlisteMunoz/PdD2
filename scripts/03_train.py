import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import ClimateCNN, ClimateMLP

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de downscaling climático")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "mlp"], help="Arquitectura a entrenar: 'cnn' o 'mlp'")
    parser.add_argument("--epochs", type=int, default=30, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Tasa de aprendizaje")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proporción de validación (Hindcast)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuración de directorios
    data_path = BASE_DIR / "data/processed/train_unified.pt"
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Carga de datos
    if not data_path.exists():
        sys.exit(f"Error: No se encuentra el dataset en {data_path}")
        
    print(f"Cargando dataset desde {data_path}...")
    checkpoint = torch.load(data_path)
    X_full = checkpoint['X'] # Tensor formato (Time, Lat, Lon)
    
    # Ajuste de dimensiones (N, C, H, W)
    if X_full.ndim == 3:
        X_full = X_full.unsqueeze(1)
        
    # Preparación de datos para aprendizaje supervisado (t -> t+1)
    # X_input: Mes actual
    # Y_target: Mes siguiente (Ground Truth / ERA5)
    X_input = X_full[:-1]
    Y_target = X_full[1:]
    
    # División cronológica para Hindcast
    n_val = int(len(X_input) * args.val_split)
    n_train = len(X_input) - n_val
    
    train_dataset = TensorDataset(X_input[:n_train], Y_target[:n_train])
    val_dataset = TensorDataset(X_input[n_train:], Y_target[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Arquitectura: {args.model.upper()}")
    print(f"Dispositivo: {device}")
    print(f"Datos de entrenamiento: {len(train_dataset)} muestras")
    print(f"Datos de validación (Hindcast): {len(val_dataset)} muestras")
    
    # Inicialización del modelo
    _, _, H, W = X_full.shape
    if args.model == "cnn":
        model = ClimateCNN(height=H, width=W).to(device)
    else:
        model = ClimateMLP(height=H, width=W).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Bucle de entrenamiento
    best_val_loss = float('inf')
    model_save_path = models_dir / f"best_model_{args.model}.pth"
    
    print("Iniciando entrenamiento...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Guardado del mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            
    print(f"Entrenamiento finalizado. Modelo guardado en {model_save_path}")
    
    # Generación de resultados de Hindcast para visualización
    print("Generando archivo de resultados para análisis posterior...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    preds = []
    targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            p = model(x)
            preds.append(p.cpu())
            targets.append(y.cpu())
            
    results_path = models_dir / f"hindcast_results_{args.model}.pt"
    torch.save({
        'predictions': torch.cat(preds),
        'targets': torch.cat(targets),
        'coords': checkpoint.get('coords'),
        'model_type': args.model
    }, results_path)
    print(f"Resultados guardados en {results_path}")

if __name__ == "__main__":
    main()
