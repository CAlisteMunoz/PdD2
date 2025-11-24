import sys
import os
import torch
import xarray as xr
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS DINÁMICA ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import ClimateEmulator

def main():
    print("---  Iniciando Evaluación Histórica (Hindcast) ---")
    
    # RUTAS ACTUALIZADAS
    # Usamos 'train.pt' porque ahí están guardados el Scaler y la Climatología
    DATA_FILE = PROJECT_ROOT / "data" / "processed" / "train.pt"
    RAW_DATA = PROJECT_ROOT / "data" / "raw" / "prw_*.nc"
    MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
    OUTPUT_NC = PROJECT_ROOT / "outputs" / "evaluacion_historica.nc"
    
    # 1. CARGAR METADATOS
    if not DATA_FILE.exists():
        print(f" Error: No se encuentra {DATA_FILE}. Ejecuta prepare_data.py primero.")
        sys.exit(1)
        
    print("Cargando metadatos de entrenamiento...")
    meta = torch.load(DATA_FILE)
    
    scaler = meta['scaler']
    coords = meta['coords']
    
    if 'climatology' not in meta:
        print(" Error: El archivo train.pt no tiene climatología. Actualiza prepare_data.py.")
        sys.exit(1)
    
    climatology_flat = meta['climatology'].numpy().flatten()
    print(" Metadatos cargados.")

    # 2. CARGAR MODELO
    models_list = list(MODELS_DIR.glob("*.pth"))
    if not models_list:
        raise FileNotFoundError(" No hay modelos .pth entrenados.")
    
    LATEST_MODEL = max(models_list, key=os.path.getctime)
    print(f"Usando modelo: {LATEST_MODEL.name}")
    
    # Inicializar arquitectura (usamos el tamaño del climatology como referencia de input)
    input_size = len(climatology_flat)
    model = ClimateEmulator(input_size=input_size)
    model.load_state_dict(torch.load(LATEST_MODEL))
    model.eval()
    
    # 3. PREPARAR DATOS ESPECÍFICOS PARA HINDCAST
    # Queremos probar específicamente: Input(1850-1900) -> Target(1960-2010)
    print("Generando climatologías específicas desde datos crudos...")
    ds = xr.open_mfdataset(str(RAW_DATA), combine='by_coords', engine='netcdf4')
    
    # Periodos
    p_in_start, p_in_end = "1850", "1900"
    p_tgt_start, p_tgt_end = "1960", "2010"
    
    # Promedios temporales absolutos
    input_abs = ds['prw'].sel(time=slice(p_in_start, p_in_end)).mean(dim='time').fillna(0.0)
    target_abs = ds['prw'].sel(time=slice(p_tgt_start, p_tgt_end)).mean(dim='time').fillna(0.0)
    
    # Reconstruir mapa climatología 2D
    n_lat, n_lon = len(coords['lat']), len(coords['lon'])
    clim_map = climatology_flat.reshape(n_lat, n_lon)
    
    # Calcular Anomalías (Input para la IA)
    input_anom = input_abs.values - clim_map
    
    # Normalizar
    norm_input = (input_anom.flatten() - scaler['min']) / (scaler['max'] - scaler['min'])
    X_tensor = torch.from_numpy(norm_input).float().unsqueeze(0)
    
    # 4. PREDECIR
    print(f"Prediciendo cambio climático ({p_in_end} -> {p_tgt_start})...")
    with torch.no_grad():
        pred_norm = model(X_tensor)
        
    # 5. RECONSTRUIR
    print("Reconstruyendo valores físicos...")
    # Des-normalizar anomalía
    pred_anom = pred_norm.numpy().squeeze() * (scaler['max'] - scaler['min']) + scaler['min']
    
    # Sumar Climatología para obtener valor absoluto
    pred_abs = pred_anom + climatology_flat
    
    # Reshape 2D
    pred_grid = pred_abs.reshape(n_lat, n_lon)
    
    # 6. GUARDAR
    ds_out = xr.Dataset(
        {
            "prediccion": (("lat", "lon"), pred_grid),
            "realidad": (("lat", "lon"), target_abs.values),
            "error_bias": (("lat", "lon"), pred_grid - target_abs.values),
        },
        coords={"lat": coords['lat'], "lon": coords['lon']},
        attrs={
            "description": "Validacion Historica (Hindcast)",
            "input_period": f"{p_in_start}-{p_in_end}",
            "target_period": f"{p_tgt_start}-{p_tgt_end}",
            "units": "kg m-2"
        }
    )
    
    ds_out.to_netcdf(OUTPUT_NC)
    print(f" Evaluación guardada exitosamente en: {OUTPUT_NC}")

if __name__ == "__main__":
    main()
