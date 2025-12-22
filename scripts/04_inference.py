import torch
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import ClimateCNN

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/processed/train_unified.pt"
MODEL_PATH = ROOT / "models/best_model_cnn.pth"
OUT_DIR = ROOT / "outputs/predictions"
OUT_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = torch.device("cpu") 

print("--- GENERANDO PROYECCIÓN 2100 ---")

# Cargar
data = torch.load(DATA_PATH)
coords = data['coords']
H, W = len(coords['lat']), len(coords['lon'])

model = ClimateCNN(height=H, width=W).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
except:
    sys.exit(" No hay modelo entrenado.")

# Configurar Simulación
# Último dato real conocido (final de 2014)
last_x = data['X'][-1].unsqueeze(0).unsqueeze(0).to(DEVICE)

# Fechas futuras (2015 - 2100 Mensual)
dates = pd.date_range(start='2015-01-01', end='2100-12-31', freq='MS')
steps = len(dates)

print(f" Simulando {steps} meses (Autoregresivo)...")
results = []

with torch.no_grad():
    current = last_x
    for i in range(steps):
        pred = model(current)
        
        # Guardar valor real (0-80)
        val = pred.squeeze().cpu().numpy() * 80.0
        val = np.clip(val, 0, 90) # Clip de seguridad visual
        results.append(val)
        
        current = pred # Feedback loop
        if i % 120 == 0: print(f"   ... Año {dates[i].year}")

# Guardar NC
ds = xr.Dataset(
    {"prw": (("time", "lat", "lon"), np.array(results))},
    coords={"time": dates, "lat": coords['lat'], "lon": coords['lon']}
)
ds.to_netcdf(OUT_DIR / "proyeccion_2100_final.nc")
print(f" Archivo listo: outputs/predictions/proyeccion_2100_final.nc")
