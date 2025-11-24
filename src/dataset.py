import xarray as xr
import torch
import numpy as np
from torch.utils.data import Dataset

class ClimateDataset(Dataset):
    def __init__(self, nc_files_pattern, var_name='prw', 
                 start_year=1850, end_year=2014, 
                 window_size=20, lag=50, step=2):
        
        print(f" Cargando datos (Anomalías): {nc_files_pattern}")
        self.ds = xr.open_mfdataset(str(nc_files_pattern), combine='by_coords', engine='netcdf4')
        # Cargar todo a RAM
        self.data_array = self.ds[var_name].load()
        
        # 1. CALCULAR CLIMATOLOGÍA (El mapa "base")
        # Promediamos todo el periodo histórico para tener la referencia
        self.climatology = self.data_array.mean(dim='time')
        print(" Climatología base calculada.")
        
        # Coordenadas
        self.lat = self.data_array.lat.values
        self.lon = self.data_array.lon.values
        
        # 2. CALCULAR ANOMALÍAS (Dato Real - Promedio Histórico)
        self.anomalies = self.data_array - self.climatology
        
        # Normalización de Anomalías
        self.min_val = float(self.anomalies.min())
        self.max_val = float(self.anomalies.max())
        print(f"⚖️ Rango Anomalías: {self.min_val:.3f} a {self.max_val:.3f}")
        
        # Generar ventanas
        self.samples = []
        current_year = start_year
        while True:
            input_end = current_year + window_size
            target_start = current_year + lag
            target_end = target_start + window_size
            
            if target_end > end_year: break
            
            self.samples.append({
                'input': (str(current_year), str(input_end)),
                'target': (str(target_start), str(target_end))
            })
            current_year += step
            
        print(f" Muestras generadas: {len(self.samples)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        period = self.samples[idx]
        
        # Tomamos las ANOMALÍAS, no los datos crudos (raw) 
        # Input: Anomalía promedio del pasado
        inp = self.anomalies.sel(time=slice(*period['input'])).mean(dim='time')
        # Target: Anomalía promedio del futuro
        tgt = self.anomalies.sel(time=slice(*period['target'])).mean(dim='time')
        
        return self._process(inp), self._process(tgt)

    def _process(self, data):
        data = data.fillna(0.0)
        # Normalizar entre -1 y 1 aprox (o 0 y 1)
        norm = (data - self.min_val) / (self.max_val - self.min_val)
        return torch.from_numpy(norm.values.flatten()).float()
