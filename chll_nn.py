import argparse
import datetime
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import shutil
import eumdac
import xarray as xr
import numpy as np
from shapely import geometry
import time
import pandas as pd
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET

# Setup logging for tracking events and errors
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_chlorophyll_data(datastore, longps, latgps, factor, start_date, end_date, collection_ids, download_dir, selected_products):
    # Initialize storage and logs
    directories = []
    downloaded = []
    max_retries = 3
    retry_delay = 5
    time_log_file = os.path.join(download_dir, 'time.txt')
    total_time_spent = 0
    directories_list_file = os.path.join(download_dir, 'products.txt')

    # Initialize log file
    with open(time_log_file, 'w') as time_file:
        time_file.write("Product ID,Time Spent (seconds)\n")

    # Load previously downloaded product list
    if os.path.exists(directories_list_file):
        with open(directories_list_file, 'r') as file:
            downloaded = [line.strip() for line in file.readlines()]
    else:
        with open(directories_list_file, 'w') as file:
            pass

    # Remove tie_geo_coordinates from selected products
    selected_products = [product for product in selected_products if product != "tie_geo_coordinates.nc"]

    # Define rectangular region of interest (ROI)
    roi_coords = [(longps + factor, latgps + factor),
                  (longps - factor, latgps + factor),
                  (longps - factor, latgps - factor),
                  (longps + factor, latgps - factor),
                  (longps + factor, latgps + factor)]
    roi_polygon = Polygon(roi_coords)
    roi_wkt = "POLYGON((" + ", ".join([f"{lon} {lat}" for lon, lat in roi_coords]) + "))"

    # Attempt to read a custom masking polygon from a KML file
    kml_path = os.path.join(os.getcwd(), "Polygon", "masking.kml")
    if os.path.isfile(kml_path):
        with open(kml_path, 'r', encoding='utf-8') as file:
            kml_content = file.read()
        namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
        root = ET.fromstring(kml_content)
        coords_text = None
        for placemark in root.findall(".//kml:Placemark", namespace):
            polygon = placemark.find(".//kml:Polygon", namespace)
            if polygon is not None:
                coords_element = polygon.find(".//kml:coordinates", namespace)
                if coords_element is not None:
                    coords_text = coords_element.text.strip()
                    break
        if coords_text:
            coords = []
            for coord in coords_text.split():
                lon, lat, *_ = map(float, coord.split(','))
                coords.append((lon, lat))
            kml_polygon = Polygon(coords)
        else:
            print("No valid polygon found in the KML file.")
    else:
        print("KML file not found. Using default square ROI.")

    for collection_id in collection_ids:
        selected_collection = datastore.get_collection(collection_id)
        try:
            products = selected_collection.search(geo=roi_wkt, dtstart=start_date, dtend=end_date)
        except Exception as e:
            print(f"Error searching products in {collection_id}: {e}")
            continue

        # Iterate day by day from start to end date
        date_cursor = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        while date_cursor <= end_date_dt:
            day_str = date_cursor.strftime("%Y%m%d")
            date_cursor += timedelta(days=1)

            for product in products:
                product_id = product._id
                if day_str not in product_id:
                    continue

                start_time = time.time()

                if product_id in downloaded:
                    continue

                entry_name = product_id.split('_')[7] if len(product_id.split('_')) > 7 else 'entry'
                print(f"Processing product {product_id}")

                downloaded_files = []
                # Download selected product entries
                for entry in product.entries:
                    if any(filename in entry for filename in selected_products):
                        if "tie_geo_coordinates.nc" in entry:
                            print(f"Skipping {entry} as per user request.")
                            continue
                        attempt = 0
                        while attempt < max_retries:
                            try:
                                file_path = os.path.join(download_dir, os.path.basename(entry))
                                with product.open(entry=entry) as fsrc, open(file_path, mode='wb') as fdst:
                                    print(f'Downloading {fsrc.name} from {entry_name}. Attempt {attempt + 1}.')
                                    shutil.copyfileobj(fsrc, fdst)
                                    print(f'Download complete: {fsrc.name}')
                                downloaded_files.append(file_path)
                                break
                            except Exception as e:
                                attempt += 1
                                print(f"Error downloading {entry}: {e}. Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay = min(retry_delay * 2, 60)
                        else:
                            print(f"Persistent error in {entry}. Skipping...")
                            continue

                # Validate downloaded files
                file_map = {os.path.basename(f): f for f in downloaded_files}
                geo_path = file_map.get("geo_coordinates.nc")
                flag_path = file_map.get("wqsf.nc")
                chl_path = file_map.get("chl_nn.nc")

                if not geo_path or not flag_path or not chl_path:
                    print(f"Missing one or more essential files for {entry_name}. Skipping processing.")
                    continue

                directories.append(str(entry_name))
                downloaded.append(product_id)
                with open(directories_list_file, 'a') as file:
                    file.write(f"{product_id}\n")

                try:
                    # Load and mask spatial data
                    geo_data = xr.open_dataset(geo_path)
                    lat, lon = geo_data['latitude'].data, geo_data['longitude'].data
                    geo_data.close()
                    polygon = geometry.Polygon(roi_coords)
                    point_mask = np.array([polygon.contains(geometry.Point(x, y)) for x, y in zip(lon.flatten(), lat.flatten())]).reshape(lon.shape)
                    if lat.shape != point_mask.shape:
                        print(f"Warning: Shape mismatch detected for {entry_name}. Setting flag columns empty.")
                        point_mask = np.zeros_like(lat, dtype=bool)

                    # Create dataframe
                    datetime_obj = datetime.strptime(entry_name, "%Y%m%dT%H%M%S")
                    df = pd.DataFrame({"latitude": lat[point_mask], "longitude": lon[point_mask], "datetime": datetime_obj})

                    # Apply KML polygon filter (secondary filter)
                    if 'kml_polygon' in locals() and isinstance(kml_polygon, Polygon):
                        df['inside_kml'] = [kml_polygon.contains(geometry.Point(lon, lat)) for lon, lat in zip(df['longitude'], df['latitude'])]
                        df = df[df['inside_kml']].drop(columns='inside_kml')
                    else:
                        print("KML polygon is not fully contained within the square ROI.")

                    # Process WQSF flags
                    if flag_path:
                        flag_data = xr.open_dataset(flag_path)
                        wqsf_values = flag_data['WQSF'].data
                        flag_data.close()
                        df['INVALID'] = (wqsf_values[point_mask] & (1 << 0)) > 0
                        df['WATER'] = (wqsf_values[point_mask] & (1 << 1)) > 0
                        df['CLOUD'] = (wqsf_values[point_mask] & (1 << 2)) > 0
                        df['LAND'] = (wqsf_values[point_mask] & (1 << 3)) > 0

                    # Track variables used
                    var_list_file = os.path.join(download_dir, "var_names.txt")
                    existing_vars = set()
                    if os.path.exists(var_list_file):
                        with open(var_list_file, "r") as file:
                            existing_vars = set(line.strip() for line in file.readlines())

                    # Process additional variables from downloaded .nc files
                    for file in downloaded_files:
                        if file.endswith(".nc") and file not in [geo_path, flag_path]:
                            try:
                                with xr.open_dataset(file) as dataset:
                                    for var_name in dataset.data_vars:
                                        var_data = dataset[var_name].data
                                        if var_data.shape == lat.shape:
                                            df[var_name] = var_data[point_mask]
                                            existing_vars.add(var_name)
                                        else:
                                            print(f"Shape mismatch for {var_name} in {file}. Skipping...")
                            except Exception as e:
                                print(f"Error processing {file}: {e}")

                    # Save updated variable list
                    with open(var_list_file, "w") as file:
                        for var in sorted(existing_vars):
                            file.write(var + "\n")

                    # Save the final DataFrame as CSV
                    output_path = os.path.join(download_dir, f"{entry_name}.csv")
                    df.to_csv(output_path, index=False)
                    print(f"DataFrame saved at: {output_path}")

                except Exception as e:
                    logging.error(f"Error processing files for {entry_name}: {e}")
                finally:
                    # Clean temporary files
                    for file in downloaded_files:
                        if os.path.exists(file):
                            os.remove(file)
                    print(f"Cleaned up temporary files for {entry_name}")

                end_time = time.time()
                time_spent = end_time - start_time
                total_time_spent += time_spent
                with open(time_log_file, 'a') as time_file:
                    time_file.write(f"{entry_name} -- {time_spent:.2f} seconds\n")
                logging.info(f"Processing time: {entry_name} -- {time_spent:.2f} seconds")

    logging.info(f"Total processing time: {total_time_spent:.2f} seconds")

def km_to_degrees(km):
    # Convert kilometers to degrees (approximate)
    return km / 111.32

def main(args):
    # Load credentials
    credentials_file = Path.home() / ".eumdac" / "credentials"
    try:
        credentials = credentials_file.read_text().split(",")
        token = eumdac.AccessToken((credentials[0], credentials[1]))
        logging.info(f"Token obtained. Expires on: {token.expiration}")
    except (FileNotFoundError, IndexError):
        logging.error("Error loading credentials.")
        return

    datastore = eumdac.DataStore(token)
    download_dir = Path.home() / args.directory
    download_dir.mkdir(parents=True, exist_ok=True)
    factor = km_to_degrees(args.factor)

    default = ["chl_nn.nc", "chl_oc4me.nc", "wqsf.nc","geo_coordinates.nc"]
    selected_products = default + [p.strip() for p in args.products.split(",")] if args.products else default

    process_chlorophyll_data(
        datastore, args.longps, args.latgps, factor, args.start_date, args.end_date,
        args.collection_ids, str(download_dir), selected_products
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Chlorophyll Data Processor")
    parser.add_argument("--longps", type=float, default=-66.025, help="Longitud del ROI")
    parser.add_argument("--latgps", type=float, default=18.425, help="Latitud del ROI")
    parser.add_argument("--factor", type=float, default=5, help="Factor de expansión del ROI")
    parser.add_argument("--start_date", type=str, default="2019-01-04", help="Fecha de inicio (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2019-01-07", help="Fecha de fin (YYYY-MM-DD)")
    parser.add_argument("--collection_ids", nargs="+", default=["EO:EUM:DAT:0407", "EO:EUM:DAT:0556"], help="Colecciones")
    parser.add_argument("--directory", type=str, default="SanJose_new", help="Directorio de salida")
    parser.add_argument("--products",type=str, default="iop_lsd.nc,iop_nn.nc,iwv.nc,Oa01_reflectance.nc,Oa02_reflectance.nc,Oa03_reflectance.nc,Oa04_reflectance.nc,Oa05_reflectance.nc,Oa06_reflectance.nc,Oa07_reflectance.nc,Oa08_reflectance.nc,Oa09_reflectance.nc,Oa10_reflectance.nc,Oa11_reflectance.nc,Oa12_reflectance.nc,Oa16_reflectance.nc,Oa17_reflectance.nc,Oa18_reflectance.nc,Oa21_reflectance.nc,par.nc,tie_geo_coordinates.nc,tie_meteo.nc,time_coordinates.nc,trsp.nc,tsm_nn.nc,w_aer.nc", help="Comma-separated list of products to download. Example: 'geo_coordinates.nc,wqsf.nc,Oa01_reflectance.nc'.")
    args = parser.parse_args()
    main(args)
