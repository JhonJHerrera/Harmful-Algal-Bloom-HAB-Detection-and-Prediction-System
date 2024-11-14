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

def download(directories, start_date, end_date, datastore, output_dir, roi, collection_ids=["EO:EUM:DAT:0407", "EO:EUM:DAT:0556"]):
    list_prod = []
    lines = []
    total_time = 0

    # Set output directory for products and CSV if not provided
    output_dir = output_dir or os.path.join(os.getcwd(), 'products')
    os.makedirs(output_dir, exist_ok=True)

    for collection_id in collection_ids:
        selected_collection = datastore.get_collection(collection_id)
        
        # Search for products within date range and ROI
        products = selected_collection.search(
            geo='POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in roi])),
            dtstart=start_date,
            dtend=end_date
        )

        if not products:
            print("No products found for the specified date and region.")
            continue

        # Process each found product
        for product in products:
            list_prod.append(product)
            start_time = time.time()

            for entry in product.entries:
                entry_name = entry.split('_')[7] if len(entry.split('_')) > 7 else 'entry'
                
                if entry_name not in directories:
                    directories.append(entry_name)
                    with open(os.path.join(output_dir, 'directories_list.txt'), 'a') as file:
                        file.write(f"{entry_name}\n")

                entry_dir = os.path.join(output_dir, entry_name)
                os.makedirs(entry_dir, exist_ok=True)

                # Check and download specific files
                if "Oa01_reflectance" in entry or 'geo_coordinates.nc' in entry or 'chl' in entry:
                    with product.open(entry=entry) as fsrc, open(os.path.join(entry_dir, fsrc.name), 'wb') as fdst:
                        print(f'Downloading {fsrc.name} to {entry_dir}')
                        shutil.copyfileobj(fsrc, fdst)
                        print(f'Download of file {fsrc.name} completed.')

                # Break if necessary files are downloaded
                if all(os.path.exists(os.path.join(entry_dir, fname)) for fname in ['geo_coordinates.nc', 'chl_nn.nc', 'chl_oc4me.nc']):
                    break

            # Track time per product download
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            lines.append(f"Iteration {product}: {elapsed_time:.2f} seconds\n")

    lines.append(f"Total time: {total_time:.2f} seconds\n")
    with open(os.path.join(output_dir, 'time_download'), 'w') as file:
        file.writelines(lines)

    print("Download times saved")
    return directories


def read_directories_list(output_dir):
    """Reads 'directories_list.txt' to retrieve list of directories."""
    directories = []
    file_path = os.path.join(output_dir, 'directories_list.txt')

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            directories = [line.strip() for line in file.readlines()]
        print(f"Directories loaded from {file_path}")
    else:
        print(f"{file_path} does not exist.")
    
    return directories


def process_chlorophyll_data(vec, roi, directory=None):
    total_time = 0
    lines = []
    df = pd.DataFrame()

    for paths in vec:
        start_time = time.time()
        directory = directory or os.path.join(os.getcwd(), 'products', paths)
        os.makedirs(directory, exist_ok=True)

        chl_path, geo_path, chl_oc4me_path = (
            os.path.join(directory, paths, 'chl_nn.nc'),
            os.path.join(directory, paths, 'geo_coordinates.nc'),
            os.path.join(directory, paths, 'chl_oc4me.nc')
        )

        if not all(os.path.exists(path) for path in [chl_path, geo_path]):
            print(f"Required files missing in {directory}. Skipping this directory.")
            continue

        try:
            # Load and filter geo-coordinates
            geo_data = xr.open_dataset(geo_path)
            lat, lon = geo_data['latitude'].data, geo_data['longitude'].data
            polygon = geometry.Polygon(roi)
            point_mask = vectorized.contains(polygon, lon, lat)
            geo_data.close()

            # Process chlorophyll data
            chl_data = xr.open_dataset(chl_path)
            chl_values = chl_data['CHL_NN'].data
            chl_data.close()
            chl_roi = chl_values[point_mask]
            lat_roi, lon_roi = lat[point_mask], lon[point_mask]

            df = pd.DataFrame({
                'latitude': lat_roi,
                'longitude': lon_roi,
                'chlorophyll': chl_roi
            })

            if os.path.exists(chl_oc4me_path):
                chl_oc4me_data = xr.open_dataset(chl_oc4me_path)
                df['chlorophyll_oc4me'] = chl_oc4me_data['CHL_OC4ME'].data[point_mask]
                chl_oc4me_data.close()

            # Save data to CSV and clean up
            output_path = os.path.join(directory, paths, 'chlorophyll_roi.csv')
            df.to_csv(output_path, index=False)
            print(f"DataFrame saved to: {output_path}")

            for file in [chl_path, chl_oc4me_path, geo_path]:
                if os.path.exists(file):
                    os.remove(file)

        except Exception as e:
            print(f"Error processing files in {directory}: {e}")
            continue

        # Track time per directory processing
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        lines.append(f"Iteration {paths}: {elapsed_time:.2f} seconds\n")

    lines.append(f"Total time: {total_time:.2f} seconds\n")
    with open(os.path.join(directory, 'read_time'), 'w') as file:
        file.writelines(lines)

    return df


def main(longps=-117.31646, latgps=32.92993, factor=0.01802, sd = "2019-06-01", ed = "2019-06-03"):
    download_dir = os.path.join(Path.home(), "products")
    os.makedirs(download_dir, exist_ok=True)

    eumdac_credentials_file = Path(Path.home() / '.eumdac' / 'credentials')

    # Load or create credentials
    if os.path.exists(eumdac_credentials_file):
        consumer_key, consumer_secret = Path(eumdac_credentials_file).read_text().split(',')
    else:
        consumer_key, consumer_secret = 'fNdGKRIUrnD_gUGgOofwyuXzFZAa', 'ajdq2MjBDqGB5R_u51w1FSXLfKoa'
        os.makedirs(os.path.dirname(eumdac_credentials_file), exist_ok=True)
        with open(eumdac_credentials_file, "w") as f:
            f.write(f'{consumer_key},{consumer_secret}')

    token = eumdac.AccessToken((consumer_key, consumer_secret))
    print(f"Token '{token}' expires on {token.expiration}")

    # Define ROI and collection dates
    roi = [[longps + factor, latgps + factor], [longps - factor, latgps + factor],
           [longps - factor, latgps - factor], [longps + factor, latgps - factor],
           [longps + factor, latgps + factor]]
    
    directories = read_directories_list(download_dir)
    download(directories, sd, ed, eumdac.DataStore(token), download_dir, roi)

    df = process_chlorophyll_data(directories, roi, download_dir)
    print("Processed chlorophyll data:")
    print(df)


if __name__ == "__main__":
    main()