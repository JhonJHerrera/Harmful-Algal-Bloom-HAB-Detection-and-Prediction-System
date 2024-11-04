import datetime   
from datetime import datetime             # a lirbary that supports the creation of date objects
import os                                 # a library that allows us access to basic operating system commands like making directories
from pathlib import Path                  # a library that helps construct system path objects
import shutil                             # a library that allows us access to basic operating system commands like copy
import eumdac                             # a tool that helps us download via the eumetsat/data-store
import xarray as xr                       # a library that supports the use of multi-dimensional arrays in Python
import matplotlib.pyplot as plt           # a library that support plotting
import numpy as np                        # a library that provides support for array-based mathematics
import eumartools                         # a EUMETSAT library that support working with Sentinel-3 products
from shapely import geometry, vectorized  # a library that supports the creation of shape objects, like polygons  
import csv
import time
import pandas as pd


def download(start_date, end_date, datastore,  output_dir, collectionID='EO:EUM:DAT:0407', longps = -117.31646, latgps = 32.92993, factor = 0.01802 ):
    #coordinates
    longspf = longps + factor
    longmpf = longps - factor
    latpf = latgps + factor
    latmf = latgps - factor
    roi = [[longspf,latpf],[longmpf,latpf],[longmpf,latmf],[longspf,latmf],[longspf,latpf]]
    lines = []
    total_time = 0
    directories = []
    # Get the data collection
    selected_collection = datastore.get_collection(collectionID)
    
    # Directory to save the products (same directory for CSV)
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'products')

    os.makedirs(output_dir, exist_ok=True)
    
    # Search for products in the ROI and within the date range
    products = selected_collection.search(
        geo='POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in roi])),
        dtstart=start_date,
        dtend=end_date
    )
    if not products:
            print("No products found for the specified date and region.")
    else:
        # Loop through the found products
        for product in products:
            start_time = time.time()                
            # Look for the 'chl_nn.nc' and 'chl_oc4me.nc' files within the product
            for entry in product.entries:
                entry_name = 'entry'
                entry_parts = entry.split('_')
                if len(entry_parts) > 7:
                    entry_name = entry_parts[7]
                    if entry_name not in directories:
                        directories.append(entry_name)
                        print(entry_name)
                # Ensure the directory for the product entry exists
                entry_dir = os.path.join(output_dir, entry_name)
                os.makedirs(entry_dir, exist_ok=True)  # Create directory if it doesn't exist

                if "Oa01_reflectance" in entry:
                    required_entry = entry
                if 'tie_geo_coordinates.nc' in entry:
                    continue
                if 'geo_coordinates.nc' in entry or 'chl' in entry:
                    with product.open(entry=entry) as fsrc, open(os.path.join(output_dir,entry_name, fsrc.name),
                                                                mode='wb') as fdst:
                        
                        print(f'Downloading {fsrc.name}. in {os.path.join(output_dir,entry_name)}')
                        shutil.copyfileobj(fsrc, fdst)
                        print(f'Download of file {fsrc.name} finished.')
                if os.path.exists(os.path.join(output_dir,entry_name, 'geo_coordinates.nc')) and  os.path.exists(os.path.join(output_dir,entry_name, 'chl_nn.nc')) and os.path.exists(os.path.join(output_dir,entry_name, 'chl_oc4me.nc')):
                    break
            end_time = datetime.now()
        
            # Calcular el tiempo de la iteración y formatearlo en inglés
            end_time = time.time()
            elapsed_time = end_time - start_time  # en segundos
            total_time += elapsed_time
            lines.append(f"Iteration {product}: {elapsed_time:.2f} seconds\n")
    lines.append(f"Total time: {total_time:.2f} seconds\n")

    with open(os.path.join(output_dir,'entry','time_download'), 'w') as file:
        file.writelines(lines)
        

    print(f"Tiempos de cada iteración guardados")
    return directories, roi    #coordinates

def process_chlorophyll_data(vec, roi, directory = None):
    """
    Procesa archivos de clorofila en un directorio específico y devuelve un DataFrame
    con valores de clorofila dentro de un ROI.

    Parameters:
    - directory: Ruta del directorio donde están los archivos de clorofila y coordenadas.
    - roi: Lista de coordenadas que define el ROI en formato [[lat1, lon1], [lat2, lon2], ...].

    Returns:
    - DataFrame con latitud, longitud, valores de clorofila y clorofila OC4ME (si existe).
    """
    total_time = 0
    for paths in vec:
        lines = []
        start_time = time.time()
        if directory is None:
            directory = os.path.join(os.getcwd(), 'products', paths)
        os.makedirs(directory, exist_ok=True)

        # Cargar datos de clorofila (chl.nc) y coordenadas (geo_coordinates.nc)
        chl_path = os.path.join(directory, paths, 'chl_nn.nc')
        geo_path = os.path.join(directory, paths, 'geo_coordinates.nc')
        chl_oc4me_path = os.path.join(directory, paths, 'chl_oc4me.nc')

        # Cargar datos de coordenadas
        geo_data = xr.open_dataset(geo_path)
        lat = geo_data['latitude'].data
        lon = geo_data['longitude'].data
        geo_data.close()

        # Crear máscara espacial para el ROI
        polygon = geometry.Polygon(roi)
        point_mask = vectorized.contains(polygon, lon, lat)

        # Cargar valores de clorofila y aplicar la máscara del ROI
        chl_data = xr.open_dataset(chl_path)
        chl_values = chl_data['CHL_NN'].data  # Asegúrate de que 'chl' es la variable de clorofila en el archivo
        chl_data.close()
        chl_roi = chl_values[point_mask]
        lat_roi = lat[point_mask]
        lon_roi = lon[point_mask]

        # Crear DataFrame inicial
        df = pd.DataFrame({
            'latitude': lat_roi,
            'longitude': lon_roi,
            'chlorophyll': chl_roi
        })
#         print(df)
        # Verificar si existe 'chl_oc4me.nc' y agregarlo al DataFrame
        
        if os.path.exists(chl_oc4me_path):
            chl_oc4me_data = xr.open_dataset(chl_oc4me_path)
            chl_oc4me_values = chl_oc4me_data['CHL_OC4ME'].data  # Asegúrate de que 'chl_oc4me' es el nombre de la variable
            chl_oc4me_data.close()
            chl_oc4me_roi = chl_oc4me_values[point_mask]

            # Agregar columna al DataFrame
            df['chlorophyll_oc4me'] = chl_oc4me_roi

        # Guardar el DataFrame en un archivo CSV en el mismo directorio
        output_path = os.path.join(directory, paths, 'chlorophyll_roi.csv')
        df.to_csv(output_path, index=False)
        print(f"DataFrame guardado en: {output_path}")
        # Eliminar los archivos de datos para ahorrar espacio
        if os.path.exists(chl_path):
            os.remove(chl_path)
        if os.path.exists(chl_oc4me_path):
            os.remove(chl_oc4me_path)
        if os.path.exists(geo_path):
            os.remove(geo_path)
            
        end_time = time.time()
        elapsed_time = end_time - start_time  # en segundos
        total_time += elapsed_time
        lines.append(f"Iteration {paths}: {elapsed_time:.2f} seconds\n")
    lines.append(f"Total time: {total_time:.2f} seconds\n")
    with open(os.path.join(directory, 'entry', 'read_time'), 'w') as file:
        file.writelines(lines)
    return df

def main():
    download_dir = os.path.join(Path.home(), "products")
    os.makedirs(download_dir, exist_ok=True)

    # load credentials
    eumdac_credentials_file = Path(Path.home() / '.eumdac' / 'credentials')

    if os.path.exists(eumdac_credentials_file):
        consumer_key, consumer_secret = Path(eumdac_credentials_file).read_text().split(',')
    else:
        # creating authentication file
        consumer_key = 'fNdGKRIUrnD_gUGgOofwyuXzFZAa'
        # consumer_key = input('Enter your consumer key: ')
        consumer_secret = 'ajdq2MjBDqGB5R_u51w1FSXLfKoa'
        # consumer_secret = getpass.getpass('Enter your consumer secret: ')
        try:
            os.makedirs(os.path.dirname(eumdac_credentials_file), exist_ok=True)
            with open(eumdac_credentials_file, "w") as f:
                f.write(f'{consumer_key},{consumer_secret}')
        except:
            pass
            
    token = eumdac.AccessToken((consumer_key, consumer_secret))
    print(f"This token '{token}' expires {token.expiration}")

    # create data store object
    datastore = eumdac.DataStore(token)

    # Pedir al usuario las fechas de inicio y fin para la descarga de productos
    # start_date = input("Ingrese la fecha de inicio (YYYY-MM-DD): ")
    # end_date = input("Ingrese la fecha de fin (YYYY-MM-DD): ")

    # # Validar formato de fechas
    # try:
    #     datetime.strptime(start_date, '%Y-%m-%d')
    #     datetime.strptime(end_date, '%Y-%m-%d')
    # except ValueError:
    #     print("Formato de fecha incorrecto. Por favor, use el formato YYYY-MM-DD.")
    #     return
    start_date = "2022-01-01"
    end_date = "2022-01-02"
    # Definir el ID de la colección
    collectionID = 'EO:EUM:DAT:0407'
    
    # Descargar productos y obtener los directorios de productos y ROI
    directories, roi = download(start_date, end_date, datastore, download_dir, collectionID)

    # Procesar datos de clorofila
    df = process_chlorophyll_data(directories, roi, download_dir)

    # Mostrar el DataFrame final de clorofila
    print("Datos de clorofila procesados:")
    print(df)

if __name__ == "__main__":
    main()
