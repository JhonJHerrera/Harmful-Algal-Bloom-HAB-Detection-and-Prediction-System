import argparse
import datetime                 # For handling dates and times
from datetime import datetime   # For precise datetime operations
import os                       # For interacting with the operating system (e.g., file and directory operations)
from pathlib import Path        # For object-oriented file path manipulations
import eumdac                   # For accessing EUMETSAT data services
import xarray as xr             # For handling multi-dimensional labeled data arrays (e.g., NetCDF files)
import numpy as np              # For numerical operations and array manipulation
import pandas as pd             # For data manipulation and analysis, especially with tabular data

def load_variable_names(var_list_file):
    """
    Reads the variable names from a file and filters out any that contain '_err'.
    
    Parameters:
    - var_list_file (str): Path to the file containing variable names.
    
    Returns:
    - List of valid variable names.
    """
    if os.path.exists(var_list_file):
        with open(var_list_file, "r") as file:
            variables = [line.strip() for line in file.readlines() if "_err" not in line.strip()]
        return variables
    else:
        print(f"Warning: {var_list_file} not found. Using default variables.")
        return ['CHL_NN']  

def summarize_chlorophyll_data(columns, directories_list_file, products_dir, buoy_long=-117.31646, buoy_lat=32.92993, radius=0.01, output_csv_prefix="chlorophyll_summary"):
    """
    Summarizes multiple columns of chlorophyll data, calculating average, median, mode, and pixel concentration
    from the 'chlorophyll_roi.csv' files within each product directory.

    Parameters:
    - columns (list): List of column names to summarize.
    - directories_list_file (str): Path to the file containing directory names.
    - products_dir (str): Base directory where the product files are located.
    - buoy_long (float): Longitude of the buoy (reference point).
    - buoy_lat (float): Latitude of the buoy (reference point).
    - radius (float): Radius (in degrees) around the buoy for filtering data points.
    - output_csv_prefix (str): Prefix for the output CSV file.

    Returns:
    - A dictionary of summary DataFrames for each column.
    """

    # Initialize storage for summary data
    summary_data_dict = {col: [] for col in columns}
    directories = []

    if os.path.exists(directories_list_file):
        with open(directories_list_file, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  # Check if the line is not empty
                    cropped_value = line.split('_')[7]
                    directories.append(cropped_value)
    else:
        print("File not found!")

    # Iterate through each directory
    for directory in directories:
        print(directory)
        try:
            datetime_obj = datetime.strptime(directory, "%Y%m%dT%H%M%S")
            date = datetime_obj.strftime("%Y-%m-%d")
            time = datetime_obj.strftime("%H:%M:%S")
        except ValueError:
            print(f"Invalid date format in directory name: {directory}")
            continue

        csv_file_path = os.path.join(products_dir, f"{directory}.csv")

        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"CSV file not found for directory: {directory}")
            continue

        # Calculate the distance to the buoy
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['distance_to_buoy'] = np.sqrt((df['latitude'] - buoy_lat)**2 + (df['longitude'] - buoy_long)**2)
            filtered_df = df[df['distance_to_buoy'] <= radius]
        else:
            print(f"Skipping {directory}: Missing latitude/longitude columns.")
            continue

        # Process each column separately
        for column in columns:
            if column in df.columns:
                chlorophyll_values = filtered_df[column].dropna()

                if chlorophyll_values.empty:
                    print(f"No {column} values found near the buoy on {date}.")
                    continue

                # Calculate statistics
                mean_val = chlorophyll_values.mean()
                median_val = chlorophyll_values.median()
                mode_val = chlorophyll_values.mode().iloc[0] if not chlorophyll_values.mode().empty else np.nan
                pixel_concentration = len(filtered_df)

                # Store summary data
                summary_data_dict[column].append({
                    "date": date,
                    "time": time,
                    "mean_val": mean_val,
                    "median_val": median_val,
                    "mode_val": mode_val,
                    "pixel_concentration": pixel_concentration
                })

            else:
                print(f"Skipping {directory}: Column '{column}' not found.")

    # Save each column's summary data
    summary_dfs = {}
    for column, summary_data in summary_data_dict.items():
        summary_df = pd.DataFrame(summary_data)
        summary_dfs[column] = summary_df

        # Save CSV file
        saved_csv_file_path = os.path.join(products_dir, f"{column}_r{radius}_{output_csv_prefix}.csv")
        summary_df.to_csv(saved_csv_file_path, index=False)
        print(f"{column} summary saved to: {saved_csv_file_path}")

    return summary_dfs

def main(
    var_list_file="var_names.txt",
    buoy_long=-117.31646, 
    buoy_lat=32.92993,
    radii=[0.02],
    download_dirct="datos_delmarr_new"
):
    """
    Main function to process and summarize chlorophyll data from satellite collections.

    Parameters:
    - var_list_file (str): File path containing variable names.
    - buoy_long (float): Longitude of the buoy (reference point).
    - buoy_lat (float): Latitude of the buoy (reference point).
    - radii (list): List of radii (in degrees) for summarization.
    - download_dirct (str): Directory where the NetCDF files are stored.

    Returns:
    - None
    """

    # Define the directory for storing downloaded data
    download_dir = os.path.join(Path.home(), download_dirct)

    # Check and create the directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Directory created: {download_dir}")
    else:
        print(f"Using existing directory: {download_dir}")

    # Path to the variable names file
    var_file_path = os.path.join(download_dir, var_list_file)

    # Load variable names from the file
    columns = load_variable_names(var_file_path)
    if not columns:
        print("No valid variable names found. Exiting.")
        return

    # Path to track downloaded products
    directories_list_file = os.path.join(download_dir, 'products.txt')

    # Summarize data for each radius
    for rs in radii:
        print(f"Summarizing data for radius: {rs}")
        summarize_chlorophyll_data(columns, directories_list_file, download_dir, buoy_long, buoy_lat, radius=rs)

if __name__ == "__main__":
    main()
