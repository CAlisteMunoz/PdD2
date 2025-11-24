import sys
import os
import torch
import xarray as xr
import numpy as np
from pathlib import Path

# Configurar rutas
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import ClimateEmulator

def main():
    print("---  PREDICCIÓN DEL FUTURO (AJUSTE FINAL) ---")
    
    # RUTAS (Las mismas de siempre)
    MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
    RAW_DATA = PROJECT_ROOT / "data" / "raw" / "prw_*.nc"
    TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.pt" 
    OUTPUT_NC = PROJECT_ROOT / "outputs" / "prediccion_futuro_2064.nc"

    # 1. CONFIGURACIÓN
    BASE_START = "1994"
    BASE_END = "2014"
    LAG_YEARS = 50
    
    # 2. CARGAR DATOS NECESARIOS
    if not TRAIN_FILE.exists():
        print("❌ Error: Falta train.pt"); sys.exit(1)
    
    # Cargamos todo lo que guardó prepare_data.py
    meta = torch.load(TRAIN_FILE)
    scaler = meta['scaler']
    coords = meta['coords']
    
    # Recuperamos la Climatología (El mapa promedio histórico)
    if 'climatology' in meta:
        climatology_flat = meta['climatology'].numpy().flatten()
        print(" Climatología histórica recuperada.")
    else:
        print(" Error: Tu archivo train.pt es antiguo. Ejecuta 'python scripts/prepare_data.py' una vez más.")
        sys.exit(1)

    # 3. CARGAR MODELO
    models_list = list(MODELS_DIR.glob("*.pth"))
    if not models_list: print(" Faltan modelos."); sys.exit(1)
    LATEST_MODEL = max(models_list, key=os.path.getctime)
    
    model = ClimateEmulator(input_size=len(meta['X'][0]))
    model.load_state_dict(torch.load(LATEST_MODEL))
    model.eval()

    # 4. PREPARAR EL DATO RECIENTE (1994-2014)
    print(f"Usando base: {BASE_START}-{BASE_END}")
    ds = xr.open_mfdataset(str(RAW_DATA), combine='by_coords', engine='netcdf4')
    
    # Promedio absoluto reciente
    base_absolute = ds['prw'].sel(time=slice(BASE_START, BASE_END)).mean(dim='time')
    base_absolute = base_absolute.fillna(0.0)
    
    # Reconstruir mapa de climatología para restar
    n_lat, n_lon = len(coords['lat']), len(coords['lon'])
    climatology_map = climatology_flat.reshape(n_lat, n_lon)
    
    # Calcular Anomalía Reciente (Esto es lo que entra a la IA)
    input_anomaly = base_absolute.values - climatology_map
    
    # Normalizar (0 a 1)
    norm_input = (input_anomaly.flatten() - scaler['min']) / (scaler['max'] - scaler['min'])
    X_future = torch.from_numpy(norm_input).float().unsqueeze(0)

    # 5. PREDECIR
    print(" Ejecutando predicción...")
    with torch.no_grad():
        pred_norm = model(X_future)

    # 6. RECONSTRUIR EL FUTURO ABSOLUTO (La parte nueva)
    print(" Sumando climatología para obtener valores reales...")
    
    # A. Volver a anomalía (kg/m2)
    pred_anomaly = pred_norm.numpy().squeeze() * (scaler['max'] - scaler['min']) + scaler['min']
    
    # B. Anomalía Futura + Historia = Futuro Real
    pred_absolute = pred_anomaly + climatology_flat
    
    # Reshape a mapa
    pred_grid = pred_absolute.reshape(n_lat, n_lon)
    
    # 7. GUARDAR
    ds_out = xr.Dataset(
        {
            "prw_futuro": (("lat", "lon"), pred_grid),
            "prw_base": (("lat", "lon"), base_absolute.values),
            "cambio_neto": (("lat", "lon"), pred_grid - base_absolute.values)
        },
        coords={"lat": coords['lat'], "lon": coords['lon']},
        attrs={"description": f"Proyeccion Absoluta (Base + Anomalia) a +{LAG_YEARS} anos"}
    )
    
    ds_out.to_netcdf(OUTPUT_NC)
    print(f" ¡Listo! Archivo corregido en: {OUTPUT_NC}")

if __name__ == "__main__":
    main()
