import datetime                 # For handling dates and times
from datetime import datetime   # For precise datetime operations
import os                       # For interacting with the operating system (e.g., file and directory operations)
from pathlib import Path        # For object-oriented file path manipulations
import shutil                   # For high-level file operations (e.g., copying and removing files)
import eumdac                   # For accessing EUMETSAT data services
import xarray as xr             # For handling multi-dimensional labeled data arrays (e.g., NetCDF files)
import matplotlib.pyplot as plt # For creating visualizations and plots
import numpy as np              # For numerical operations and array manipulation
import eumartools               # For tools related to EUMETSAT data (specific functionality required)
from shapely import geometry, vectorized  # For geometric operations, handling polygons and spatial relationships
import csv                      # For reading and writing CSV files
import time                     # For handling delays and measuring execution time
import pandas as pd             # For data manipulation and analysis, especially with tabular data


def process_chlorophyll_data(datastore, longps, latgps, factor, start_date, end_date, collection_ids, download_dir):
    """
    Processes chlorophyll data by downloading, filtering, and analyzing satellite products
    within a specified region of interest (ROI) and date range. Saves the results to CSV files
    and logs processing times for each product.

    Parameters:
        datastore (object): Interface for accessing satellite data collections.
        longps (float): Longitude of the center of the ROI.
        latgps (float): Latitude of the center of the ROI.
        factor (float): Offset to create the ROI boundaries.
        start_date (str): Start date for the product search (ISO format: 'YYYY-MM-DD').
        end_date (str): End date for the product search (ISO format: 'YYYY-MM-DD').
        collection_ids (list): List of collection IDs to search for satellite data.
        download_dir (str): Directory to save downloaded files and outputs.

    Returns:
        list: A list of processed directory names.
    """
    directories = []  # List to store processed directory names
    downloaded = []  # List to keep track of already downloaded products
    max_retries = 3  # Maximum number of retries for download errors
    retry_delay = 5  # Initial delay (in seconds) between retries
    time_log_file = os.path.join(download_dir, 'time.txt')  # File to log processing times
    total_time_spent = 0  # Total time spent on processing all products

    # Path to track downloaded products
    directories_list_file = os.path.join(download_dir, 'products.txt')

    # Initialize or create the time log file
    with open(time_log_file, 'w') as time_file:
        time_file.write("Product ID,Time Spent (seconds)\n")  # Write header to time log file

    # Load previously downloaded products if the tracking file exists
    if os.path.exists(directories_list_file):
        with open(directories_list_file, 'r') as file:
            downloaded = [line.strip() for line in file.readlines()]
    else:
        # Create the tracking file if it doesn't exist
        with open(directories_list_file, 'w') as file:
            pass

    # Define the region of interest (ROI) as a polygon
    roi = [[longps + factor, latgps + factor], [longps - factor, latgps + factor],
           [longps - factor, latgps - factor], [longps + factor, latgps - factor],
           [longps + factor, latgps + factor]]

    # Loop through each collection ID to search for products
    for collection_id in collection_ids:
        selected_collection = datastore.get_collection(collection_id)  # Access collection

        try:
            # Search for products within the ROI and date range
            products = selected_collection.search(
                geo='POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in roi])),
                dtstart=start_date,
                dtend=end_date
            )
        except Exception as e:
            print(f"Error searching products in collection {collection_id}: {e}")
            continue  # Skip to the next collection ID

        # Loop through the found products
        for product in products:
            start_time = time.time()  # Start timing this product's processing
            product_id = product._id  # Extract product's unique identifier
            if product_id in downloaded:
                continue  # Skip already downloaded products

            # Extract a readable name for the product
            entry_name = product_id.split('_')[7] if len(product_id.split('_')) > 7 else 'entry'
            print(f"Processing product {product_id}")

            # Download required entries for the product
            for entry in product.entries:
                if (('geo_coordinates.nc' in entry or 'chl' in entry) and not 'tie_geo_coordinates.nc' in entry):
                    attempt = 0  # Initialize retry attempts
                    while attempt < max_retries:
                        try:
                            # Open the entry and save it to the local directory
                            with product.open(entry=entry) as fsrc, open(os.path.join(download_dir, fsrc.name), mode='wb') as fdst:
                                print(f'Downloading {fsrc.name} of {entry_name}. Attempt {attempt + 1}.')
                                shutil.copyfileobj(fsrc, fdst)
                                print(f'Download of file {fsrc.name} finished.')
                            break  # Exit retry loop on success
                        except Exception as e:
                            attempt += 1
                            print(f"Error downloading {entry}: {e}. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff for retries
                    else:
                        print(f"Failed to download {entry} after {max_retries} attempts. Skipping...")
                        continue

            # Paths to the downloaded files
            chl_path, geo_path, chl_oc4me_path = (
                os.path.join(download_dir, 'chl_nn.nc'),
                os.path.join(download_dir, 'geo_coordinates.nc'),
                os.path.join(download_dir, 'chl_oc4me.nc')
            )

            # Skip if required files are missing
            if not all(os.path.exists(path) for path in [chl_path, geo_path, chl_oc4me_path]):
                print(f"Missing files for product {entry_name}. Skipping...")
                continue

            try:
                # Process geographic data
                geo_data = xr.open_dataset(geo_path)
                lat, lon = geo_data['latitude'].data, geo_data['longitude'].data
                polygon = geometry.Polygon(roi)
                # Create a mask of points within the ROI
                point_mask = np.array([polygon.contains(geometry.Point(x, y)) for x, y in zip(lon.flatten(), lat.flatten())]).reshape(lon.shape)
                geo_data.close()

                # Process chlorophyll data
                chl_data = xr.open_dataset(chl_path)
                chl_values = chl_data['CHL_NN'].data
                chl_data.close()
                chl_roi = chl_values[point_mask]  # Filter data for points within the ROI
                lat_roi, lon_roi = lat[point_mask], lon[point_mask]

                # Create a DataFrame with the filtered data
                df = pd.DataFrame({
                    'latitude': lat_roi,
                    'longitude': lon_roi,
                    'chlorophyll': chl_roi
                })

                # Add additional chlorophyll data if available
                chl_oc4me_data = xr.open_dataset(chl_oc4me_path)
                df['chlorophyll_oc4me'] = chl_oc4me_data['CHL_OC4ME'].data[point_mask]
                chl_oc4me_data.close()

                # Save the DataFrame to a CSV file
                output_path = os.path.join(download_dir, f"{entry_name}.csv")
                df.to_csv(output_path, index=False)
                print(f"DataFrame saved to: {output_path}")

                directories.append(str(entry_name))  # Add entry to processed list
                downloaded.append(product_id)  # Mark product as downloaded

                # Update the tracking file
                with open(directories_list_file, 'a') as file:
                    file.write(f"{product_id}\n")

                # Clean up temporary files
                for file in [chl_path, chl_oc4me_path, geo_path]:
                    if os.path.exists(file):
                        os.remove(file)

            except Exception as e:
                print(f"Error processing files for product {entry_name}: {e}")

            end_time = time.time()  # End timing this product's processing
            time_spent = end_time - start_time
            total_time_spent += time_spent  # Update total processing time
            # Log time spent for this product
            with open(time_log_file, 'a') as time_file:
                time_file.write(f"{entry_name} -- {time_spent:.2f} seconds\n")
            print(f"Time spent on product {entry_name} -- {time_spent:.2f} seconds")

    # Write the total time spent to the time log file
    with open(time_log_file, 'a') as time_file:
        time_file.write(f"\nTotal Time Spent,{total_time_spent:.2f} seconds\n")
    print(f"Total time spent on all products: {total_time_spent:.2f} seconds")

    return directories


def summarize_chlorophyll_data(directories, products_dir, buoy_long=-117.31646, buoy_lat=32.92993, radius=0.01, output_csv="chlorophyll_summary.csv"):
    """
    Summarizes chlorophyll data, calculating average, median, mode, and pixel concentration
    from the 'chlorophyll_roi.csv' files within each product directory.

    Parameters:
    - directories (list): List of directories containing product data.
    - products_dir (str): Base directory where the product files are located.
    - buoy_long (float): Longitude of the buoy (reference point).
    - buoy_lat (float): Latitude of the buoy (reference point).
    - radius (float): Radius (in degrees) around the buoy for filtering data points.
    - output_csv (str): Name of the output CSV file for the summary.

    Returns:
    - summary_df (DataFrame): DataFrame containing the summarized chlorophyll data.
    """

    # Initialize a list to store summary data
    summary_data = []

    # Iterate through each directory listed in the directories parameter
    for directory in directories:
        # Extract the date and time from the directory name (assumes a specific naming convention)
        try:
            datetime_obj = datetime.strptime(directory, "%Y%m%dT%H%M%S")  # Parse the directory name into a datetime object
            date = datetime_obj.strftime("%Y-%m-%d")  # Extract the date in "YYYY-MM-DD" format
            time = datetime_obj.strftime("%H:%M:%S")  # Extract the time in "HH:MM:SS" format
        except ValueError:
            # If the directory name doesn't match the expected format, log an error and skip
            print(f"Invalid date format in directory name: {directory}")
            continue

        # Define the path to the CSV file within the current directory
        csv_file_path = os.path.join(products_dir, f"{directory}.csv")

        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"CSV file not found for directory: {directory}")
            continue

        # Check if the required columns exist in the DataFrame
        if 'latitude' in df.columns and 'longitude' in df.columns and 'chlorophyll' in df.columns:
            # Calculate the distance of each data point to the buoy
            df['distance_to_buoy'] = np.sqrt((df['latitude'] - buoy_lat)**2 + (df['longitude'] - buoy_long)**2)

            # Filter data points within the specified radius
            filtered_df = df[df['distance_to_buoy'] <= radius]
            chlorophyll_values = filtered_df['chlorophyll'].dropna()

            # If no chlorophyll values are found within the radius, log a message and skip
            if chlorophyll_values.empty:
                print(f"No chlorophyll values found near the buoy on {date}.")
                continue

            # Calculate summary statistics
            mean_val = chlorophyll_values.mean()  # Mean chlorophyll value
            median_val = chlorophyll_values.median()  # Median chlorophyll value
            # Mode chlorophyll value (handles empty mode gracefully)
            mode_val = chlorophyll_values.mode().iloc[0] if not chlorophyll_values.mode().empty else np.nan
            pixel_concentration = len(filtered_df)  # Number of pixels within the radius

            # Append the summary data for the current directory
            summary_data.append({
                "date": date,  # Date of the data
                "time": time,  # Time of the data
                "mean_val": mean_val,  # Mean chlorophyll value
                "median_val": median_val,  # Median chlorophyll value
                "mode_val": mode_val,  # Mode chlorophyll value
                "pixel_concentration": pixel_concentration  # Number of pixels within the radius
            })

    # Create a DataFrame from the collected summary data
    summary_df = pd.DataFrame(summary_data)

    # Save the summary DataFrame to a CSV file
    saved_csv_file_path = os.path.join(products_dir, f"r_{radius}_{output_csv}")
    summary_df.to_csv(saved_csv_file_path, index=False)
    print(f"Chlorophyll summary saved to: {saved_csv_file_path}")

    # Return the summary DataFrame
    return summary_df

def main(
    longps=-117.31646, 
    latgps=32.92993, 
    factor=0.01802, 
    start_date="2022-01-23", 
    end_date="2022-01-24", 
    collection_ids=["EO:EUM:DAT:0407", "EO:EUM:DAT:0556"], 
    directory='new_products',  
    buoy_long=-117.31646, 
    buoy_lat=32.92993,
    radii=[1, 0.02, 0.01, 0.005]
):
    """
    Main function to process and summarize chlorophyll data from satellite collections.

    Parameters:
    - longps (float): Longitude of the ROI center.
    - latgps (float): Latitude of the ROI center.
    - factor (float): Offset to define the ROI boundary.
    - start_date (str): Start date for the product search (ISO format: 'YYYY-MM-DD').
    - end_date (str): End date for the product search (ISO format: 'YYYY-MM-DD').
    - collection_ids (list): List of collection IDs to search for data.
    - directory (str): Directory to save downloaded products and results.
    - buoy_long (float): Longitude of the buoy (reference point).
    - buoy_lat (float): Latitude of the buoy (reference point).
    - radii (list): List of radii (in degrees) for summarization.

    Returns:
    - None
    """

    # Load credentials for accessing the datastore
    credentials_file = os.path.join(os.path.expanduser("~"), '.eumdac', 'credentials')
    try:
        credentials = Path(credentials_file).read_text().split(',')
        token = eumdac.AccessToken((credentials[0], credentials[1]))
        print(f"This token '{token}' expires {token.expiration}")
    except (FileNotFoundError, IndexError) as e:
        print("Error loading credentials. Please ensure the credentials file is properly set up.")
        return

    # Create a DataStore object using the token
    datastore = eumdac.DataStore(token)

    # Define the directory for storing downloaded data
    download_dir = os.path.join(Path.home(), directory)

    # Check and create the directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Directory created: {download_dir}")
    else:
        print(f"Using existing directory: {download_dir}")

    # Process chlorophyll data
    directories = process_chlorophyll_data(
        datastore, longps, latgps, factor, start_date, end_date, collection_ids, download_dir
    )

    # Summarize chlorophyll data for each specified radius
    for rs in radii:
        print(f"Summarizing chlorophyll data for radius: {rs}")
        summarize_chlorophyll_data(directories, download_dir, radius=rs)


if __name__ == "__main__":
    main()