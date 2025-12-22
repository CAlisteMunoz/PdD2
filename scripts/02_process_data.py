import xarray as xr
import torch
import numpy as np
from pathlib import Path
import sys
import warnings

# Ignorar advertencias de metadatos
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw"
PROC_DIR = ROOT / "data/processed"
PROC_DIR.mkdir(exist_ok=True, parents=True)

print("--- INICIANDO PROCESAMIENTO (ETL Mensual Robusto) ---")

# --- 1. FUNCIÓN MAESTRA DE ESTANDARIZACIÓN ---
def standardize_dataset(ds):
    """Limpia nombres de coordenadas y busca la variable correcta"""
    
    # A) Renombrar Coordenadas (Case Insensitive)
    rename_dict = {}
    for c in ds.coords:
        c_lower = c.lower()
        if c_lower in ['latitude', 'lat'] and c != 'lat':
            rename_dict[c] = 'lat'
        elif c_lower in ['longitude', 'lon'] and c != 'lon':
            rename_dict[c] = 'lon'
    
    if rename_dict:
        ds = ds.rename(rename_dict)
        
    # B) Asegurar que lat/lon son coordenadas activas
    if 'lat' in ds.data_vars: ds = ds.set_coords('lat')
    if 'lon' in ds.data_vars: ds = ds.set_coords('lon')
    
    return ds

# --- 2. BUSCAR REFERENCIA (ERA5) ---
era5_files = list(RAW_DIR.glob("*era5*.nc"))
if not era5_files:
    sys.exit(" ERROR: No encuentro archivo ERA5 en data/raw.")

# Cargar referencia
ref_ds = xr.open_dataset(era5_files[0])
ref_ds = standardize_dataset(ref_ds)
print(f" Referencia: ERA5 ({len(ref_ds.lat)}x{len(ref_ds.lon)})")


# --- 3. FUNCIÓN DE PROCESAMIENTO SEGURO ---
def process_file(filepath):
    try:
        # Abrir dataset
        ds = xr.open_dataset(filepath, use_cftime=True)
        ds = standardize_dataset(ds) 
        
        # BUSCAR LA VARIABLE DE DATOS (prw)
        # 1. Intentar por nombre exacto
        if 'prw' in ds:
            var_name = 'prw'
        # 2. Si no, buscar la variable que tenga dimensiones (time, lat, lon)
        else:
            candidates = [v for v in ds.data_vars if 'lat' in ds[v].dims and 'lon' in ds[v].dims]
            if candidates:
                var_name = candidates[0]
            else:
                # Caso extremo: coger la primera variable que no sea coordenadas
                var_name = list(ds.data_vars)[0]

        data = ds[var_name]

        # VALIDAR QUE TENEMOS COORDENADAS
        if not hasattr(data, 'lat') or not hasattr(data, 'lon'):
             # Intento desesperado de asignar coords si existen en el dataset
             if 'lat' in ds and 'lon' in ds:
                 data = data.assign_coords(lat=ds.lat, lon=ds.lon)
             else:
                 print(f"   SALTADO: {filepath.name} - No encuentro coords lat/lon. Tiene: {list(ds.coords)}")
                 return None

        # A) REGRIDDING (Interpolación Espacial)
        # Solo interpolamos si las dimensiones son distintas
        if (data.lat.size != ref_ds.lat.size) or (data.lon.size != ref_ds.lon.size):
            # Usamos kwargs para extrapolación si es necesario
            data = data.interp(lat=ref_ds.lat, lon=ref_ds.lon, method='linear', kwargs={"fill_value": "extrapolate"})

        # B) NORMALIZACIÓN (0 a 80 kg/m²)
        data = (data - 0) / 80.0
        data = data.fillna(0.0) 

        # C) NUMPY
        vals = data.values
        if vals.ndim == 4: vals = vals.squeeze() # Quitar dimensiones extra (ej: altura 1)
        
        ds.close()
        return vals

    except Exception as e:
        print(f"   ERROR CRÍTICO en {filepath.name}: {e}")
        return None

# --- 4. EJECUCIÓN PRINCIPAL ---

# A) Procesar ERA5
print("\n Procesando ERA5...")
era5_data = process_file(era5_files[0])
if era5_data is not None:
    torch.save({
        'X': torch.tensor(era5_data, dtype=torch.float32),
        'coords': {'lat': ref_ds.lat, 'lon': ref_ds.lon},
        'scaler': {'min': 0, 'max': 80}
    }, PROC_DIR / "val_era5.pt")
    print(" ERA5 guardado.")

# B) Procesar CMIP6
print("\n Procesando Modelos CMIP6...")
files = sorted(list(RAW_DIR.glob("*.nc")))
model_files = [f for f in files if "era5" not in f.name and "historical" in f.name]

train_list = []

if not model_files:
    print(" ADVERTENCIA: No encontré modelos 'historical'.")
else:
    for f in model_files:
        print(f"   -> {f.name}")
        processed = process_file(f)
        if processed is not None:
            # Validación extra de forma antes de agregar
            if processed.shape[1:] == (len(ref_ds.lat), len(ref_ds.lon)):
                train_list.append(processed)
            else:
                print(f"      Forma incorrecta tras interp: {processed.shape}. Se descarta.")

    # Fusionar
    if train_list:
        try:
            X_train = np.concatenate(train_list, axis=0)
            torch.save({
                'X': torch.tensor(X_train, dtype=torch.float32),
                'coords': {'lat': ref_ds.lat, 'lon': ref_ds.lon},
                'scaler': {'min': 0, 'max': 80}
            }, PROC_DIR / "train_unified.pt")
            print(f" Dataset de Entrenamiento Creado: {X_train.shape} (Meses, Lat, Lon)")
        except ValueError as e:
            print(f" Error al fusionar: {e}")
    else:
        print(" Error: Ningún modelo pudo ser procesado correctamente.")

print("\n ETL Terminado.")
