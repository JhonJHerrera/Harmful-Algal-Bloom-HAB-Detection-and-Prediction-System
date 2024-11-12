import datetime   
from datetime import datetime             
import os                                 
from pathlib import Path                  
import shutil                             
import eumdac                             
import xarray as xr                       
import matplotlib.pyplot as plt           
import numpy as np                        
import eumartools                         
from shapely import geometry, vectorized  
import csv
import time
import pandas as pd


def download(directories, start_date, end_date, datastore, output_dir, roi, collection_ids = ["EO:EUM:DAT:0407", "EO:EUM:DAT:0556"]):
    list_prod =[]
    lines = []
    total_time = 0

    
    
    # Directory to save the products (same directory for CSV)
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'products')

    os.makedirs(output_dir, exist_ok=True)
    
    # Get the data collection
    for collectionID in collection_ids:
        selected_collection = datastore.get_collection(collectionID)
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
                list_prod.append(product)
        for product in list_prod:
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

                        # Append each new directory to 'directories_list.txt' without overwriting
                        with open(os.path.join(output_dir, 'directories_list.txt'), 'a') as file:
                            file.write(f"{entry_name}\n")
                
                # Ensure the directory for the product entry exists
                entry_dir = os.path.join(output_dir, entry_name)
                os.makedirs(entry_dir, exist_ok=True)

                if "Oa01_reflectance" in entry:
                    required_entry = entry
                if 'tie_geo_coordinates.nc' in entry:
                    continue
                if 'geo_coordinates.nc' in entry or 'chl' in entry:
                    with product.open(entry=entry) as fsrc, open(os.path.join(output_dir, entry_name, fsrc.name),
                                                                mode='wb') as fdst:
                        print(f'Downloading {fsrc.name} in {os.path.join(output_dir, entry_name)}')
                        shutil.copyfileobj(fsrc, fdst)
                        print(f'Download of file {fsrc.name} finished.')
                
                # Break if necessary files are downloaded
                if os.path.exists(os.path.join(output_dir, entry_name, 'geo_coordinates.nc')) and \
                   os.path.exists(os.path.join(output_dir, entry_name, 'chl_nn.nc')) and \
                   os.path.exists(os.path.join(output_dir, entry_name, 'chl_oc4me.nc')):
                    break

            # Record iteration time
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            lines.append(f"Iteration {product}: {elapsed_time:.2f} seconds\n")

    lines.append(f"Total time: {total_time:.2f} seconds\n")
    with open(os.path.join(output_dir, 'time_download'), 'w') as file:
        file.writelines(lines)

    print(f"Tiempos de cada iteración guardados")
    return directories


def read_directories_list(output_dir):
    """
    Lee el archivo directories_list.txt y devuelve una lista de directorios.

    Parameters:
    - output_dir: Ruta del directorio donde se encuentra el archivo directories_list.txt.

    Returns:
    - Una lista de nombres de directorios leídos desde el archivo.
    """
    directories = []
    file_path = os.path.join(output_dir, 'directories_list.txt')
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            directories = [line.strip() for line in file.readlines()]
        print(f"Directorio cargado desde {file_path}")
    else:
        print(f"El archivo {file_path} no existe.")
    
    return directories

def process_chlorophyll_data(vec, roi, directory=None):
    total_time = 0
    lines = []
    df = pd.DataFrame()

    for paths in vec:
        start_time = time.time()
        if directory is None:
            directory = os.path.join(os.getcwd(), 'products', paths)
        os.makedirs(directory, exist_ok=True)

        chl_path = os.path.join(directory, paths, 'chl_nn.nc')
        geo_path = os.path.join(directory, paths, 'geo_coordinates.nc')
        chl_oc4me_path = os.path.join(directory, paths, 'chl_oc4me.nc')

        if not os.path.exists(chl_path) or not os.path.exists(geo_path):
            print(f"Required files not found in {directory}. Skipping this directory.")
            continue

        try:
            # Intentar abrir geo_coordinates.nc para confirmar que no esté corrupto
            geo_data = xr.open_dataset(geo_path)
            lat = geo_data['latitude'].data
            lon = geo_data['longitude'].data
            geo_data.close()

            polygon = geometry.Polygon(roi)
            point_mask = vectorized.contains(polygon, lon, lat)

            chl_data = xr.open_dataset(chl_path)
            chl_values = chl_data['CHL_NN'].data
            chl_data.close()
            chl_roi = chl_values[point_mask]
            lat_roi = lat[point_mask]
            lon_roi = lon[point_mask]

            df = pd.DataFrame({
                'latitude': lat_roi,
                'longitude': lon_roi,
                'chlorophyll': chl_roi
            })

            if os.path.exists(chl_oc4me_path):
                chl_oc4me_data = xr.open_dataset(chl_oc4me_path)
                chl_oc4me_values = chl_oc4me_data['CHL_OC4ME'].data
                chl_oc4me_data.close()
                chl_oc4me_roi = chl_oc4me_values[point_mask]
                df['chlorophyll_oc4me'] = chl_oc4me_roi

            output_path = os.path.join(directory, paths, 'chlorophyll_roi.csv')
            df.to_csv(output_path, index=False)
            print(f"DataFrame saved to: {output_path}")

            if os.path.exists(chl_path):
                os.remove(chl_path)
            if os.path.exists(chl_oc4me_path):
                os.remove(chl_oc4me_path)
            if os.path.exists(geo_path):
                os.remove(geo_path)

        except Exception as e:
            print(f"Error processing files in {directory}: {e}")
            continue

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        lines.append(f"Iteration {paths}: {elapsed_time:.2f} seconds\n")

    lines.append(f"Total time: {total_time:.2f} seconds\n")
    with open(os.path.join(directory, 'read_time'), 'w') as file:
        file.writelines(lines)

    return df

def main(longps = -117.31646, latgps = 32.92993, factor = 0.01802):
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
    #coordinates
    longspf = longps + factor
    longmpf = longps - factor
    latpf = latgps + factor
    latmf = latgps - factor
    roi = [[longspf,latpf],[longmpf,latpf],[longmpf,latmf],[longspf,latmf],[longspf,latpf]]
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
    start_date = "2019-06-01"
    end_date = "2019-06-03"
    # Definir el ID de la colección
    collectionID = 'EO:EUM:DAT:0407'
    
    # Descargar productos y obtener los directorios de productos y ROI
    directories = read_directories_list(download_dir)
    download(directories, start_date, end_date, datastore, download_dir, roi)
    print(directories)

    # Procesar datos de clorofila
    df = process_chlorophyll_data(directories, roi, download_dir)

    # Mostrar el DataFrame final de clorofila
    print("Datos de clorofila procesados:")
    print(df)

if __name__ == "__main__":
    main()
