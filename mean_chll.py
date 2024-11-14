import os
import pandas as pd
from pathlib import Path
from datetime import datetime

def summarize_chlorophyll_data(products_dir, output_csv="chlorophyll_summary_with_datetime.csv"):
    """
    Summarizes chlorophyll data by calculating the mean values from 'chlorophyll_roi.csv' files within each product directory.
    """
    # Path to the file listing directories
    directories_list_file = os.path.join(products_dir, 'directories_list.txt')
    
    # Check if the directories list file exists
    if not os.path.exists(directories_list_file):
        print(f"File not found: {directories_list_file}")
        return
    
    # Read the directories from the file
    with open(directories_list_file, 'r') as file:
        directories = [line.strip() for line in file.readlines()]
    
    # Initialize list to store summary data
    summary_data = []

    # Iterate over each directory listed in the file
    for directory in directories:
        # Extract date and time from directory name
        try:
            datetime_obj = datetime.strptime(directory, "%Y%m%dT%H%M%S")
            date = datetime_obj.strftime("%Y-%m-%d")
            time = datetime_obj.strftime("%H:%M:%S")
        except ValueError:
            print(f"Invalid date format in directory name: {directory}")
            continue

        # Path to CSV file within each directory
        directory_path = os.path.join(products_dir, directory)
        csv_file_path = os.path.join(directory_path, 'chlorophyll_roi.csv')
        
        # Check if CSV file exists
        if os.path.exists(csv_file_path):
            # Read the CSV and calculate means
            df = pd.read_csv(csv_file_path)
            chl_nn_mean = df['chlorophyll'].mean() if 'chlorophyll' in df.columns else None
            chl_oc4me_mean = df['chlorophyll_oc4me'].mean() if 'chlorophyll_oc4me' in df.columns else None
            
            # Append summary data for each directory
            summary_data.append({
                "date": date,
                "time": time,
                "chl_nn_mean": chl_nn_mean,
                "chl_oc4me_mean": chl_oc4me_mean
            })
        else:
            print(f"CSV file not found in directory: {directory_path}")

    # Create a DataFrame from the summary data
    summary_df = pd.DataFrame(summary_data)

    # Save summary DataFrame to a CSV file
    summary_df.to_csv(output_csv, index=False)
    print(f"Chlorophyll summary saved to: {output_csv}")

    return summary_df

def main():
    products_dir = os.path.join(Path.home(), "products")
    summary_df = summarize_chlorophyll_data(products_dir)

    # Display the chlorophyll summary DataFrame
    print("Chlorophyll summary by directory:")
    print(summary_df)

if __name__ == "__main__":
    main()
