import cdsapi
import os

# Asegurar que la carpeta de destino existe
output_folder = "../data/raw"
os.makedirs(output_folder, exist_ok=True)

c = cdsapi.Client()

print("Iniciando descarga de ERA5 (Vapor de agua mensual 1979-2014)...")

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'total_column_water_vapour',
        'year': [
            '1979', '1980', '1981', '1982', '1983',
            '1984', '1985', '1986', '1987', '1988',
            '1989', '1990', '1991', '1992', '1993',
            '1994', '1995', '1996', '1997', '1998',
            '1999', '2000', '2001', '2002', '2003',
            '2004', '2005', '2006', '2007', '2008',
            '2009', '2010', '2011', '2012', '2013',
            '2014',
        ],
        'month': [
            '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '10', '11', '12',
        ],
        'time': '00:00',
        'format': 'netcdf',
    },
    f'{output_folder}/era5_tcwv_monthly_1979-2014.nc')

print("¡Descarga finalizada exitosamente!")
