# **Harmful Algal Bloom (HAB) Detection Tool**  

This project aims to develop an **early warning system for Harmful Algal Blooms (HABs)** in **San José Lagoon, San Juan, Puerto Rico**. The lagoon frequently experiences **toxic cyanobacterial blooms**, which deplete oxygen levels, harm aquatic life, and negatively impact local recreation and the economy.

This tool **automates satellite data retrieval, processing, and analysis** to detect HABs using geospatial techniques. It integrates real-time satellite data from the **Copernicus Data Store (EUMETSAT API)** and generates CSV summaries of chlorophyll concentrations.

- **Primary focus area:** San José Lagoon, Puerto Rico *(can be customized)*  
- **Data source:** Sentinel-3 OLCI hyperspectral data  
- **Output:** Processed chlorophyll concentration datasets & statistical summaries  

---

## **Features**  

- **Automated Satellite Data Retrieval:** Connects to the **Copernicus API** and downloads Sentinel-3 OLCI hyperspectral data.
- **Flexible Region Selection:** Define the **Region of Interest (ROI)** with latitude, longitude, and an adjustable area expansion factor.
- **Customizable Product Selection:** Users can specify **additional spectral bands or auxiliary products** for analysis.
- **Robust Logging & Error Handling:** Implements **retry mechanisms** for failed downloads and logs processing times.
- **Geospatial Data Processing:** Uses **shapely** and **xarray** to filter chlorophyll data within a defined region.

---

## **Installation**  

### **1. Clone the Repository**  
```bash
git clone [https://github.com/your-username/HAB-Detection-Tool.git](https://github.com/JhonJHerrera/Harmful-Algal-Bloom-HAB-Detection-and-Prediction-System.git)
cd HAB-Detection-Tool
 ```

### **2. Set Up the Environment**  
Ensure you have **Python 3.10+** and Conda installed. Then, create the environment using:  
```bash
conda env create -f environment.yml
conda activate remotesent
```
---

## **Usage**  

### **Running the Tool**  
To run the script, execute the following command:  
```bash
python chll_nn.py \
  --longps -66.070 \
  --latgps 18.427 \
  --factor 5 \
  --start_date 2009-01-01 \
  --end_date 2019-01-03 \
  --collection_ids EO:EUM:DAT:0407 EO:EUM:DAT:0556 \
  --directory datos_SanJose \
  --products geo_coordinates.nc,wqsf.nc,Oa01_reflectance.nc
```

### **Command-Line Arguments**  
| Parameter        | Description                                              | Example Value |
|-----------------|----------------------------------------------------------|--------------|
| `--longps`      | Longitude of the center of the region of interest (ROI)  | `-66.070`    |
| `--latgps`      | Latitude of the center of the region of interest (ROI)   | `18.427`     |
| `--factor`      | ROI expansion factor in km                               | `5`          |
| `--start_date`  | Start date for satellite data retrieval                  | `2009-01-01` |
| `--end_date`    | End date for satellite data retrieval                    | `2019-01-03` |
| `--collection_ids` | Satellite dataset IDs from Copernicus API             | `EO:EUM:DAT:0407` |
| `--directory`   | Output directory for storing results                     | `datos_delmarr_new` |
| `--products`    | Comma-separated list of additional products to download  | `geo_coordinates.nc,wqsf.nc,Oa01_reflectance.nc` |


product_list = [
        "EOPMetadata.xml", "instrument_data.nc",
        "iop_lsd.nc", "iop_nn.nc", "iwv.nc", "Oa01_reflectance.nc", "Oa02_reflectance.nc", 
        "Oa03_reflectance.nc", "Oa04_reflectance.nc", "Oa05_reflectance.nc", "Oa06_reflectance.nc", 
        "Oa07_reflectance.nc", "Oa08_reflectance.nc", "Oa09_reflectance.nc", "Oa10_reflectance.nc",
        "Oa11_reflectance.nc", "Oa12_reflectance.nc", "Oa16_reflectance.nc","Oa17_reflectance.nc",
        "Oa18_reflectance.nc", "Oa21_reflectance.nc","par.nc","tie_geo_coordinates.nc","tie_meteo.nc",
        "time_coordinates.nc", "trsp.nc",  "tsm_nn.nc","w_aer.nc"
    ]
---

## **Example Output**  

### **1. Processed Chlorophyll Data**  
**Output Directory Structure:**  
```
datos_SanJose/
│── 20090101T120000.csv  # Processed chlorophyll data for a specific timestamp
│── 20090102T120000.csv
│── products.txt         # Record of successfully downloaded products
│── time.txt             # Log of download times for each dataset

```

## **How It Works**  

1. **Authenticate:** Securely connects to **Copernicus API** and retrieves an access token.  
2. **Search & Download Data:** Queries and downloads chlorophyll satellite data based on **ROI & time range**.  
3. **Process Data:** Extracts chlorophyll values from `.nc` and other products of the hyperspectral image and converts them into CSV format.  

---

## **Customization**  

You can modify the following parameters to customize the tool:  

- **Geographical Area:** Change `--longps`, `--latgps`, and `--factor`  
- **Data Collection IDs:** Adjust `--collection_ids` to fetch data from different satellite sources  
- **Timestamp:** Modify `--start_date`, `--end_date` to establish time window 
- **Selected Products:** Add or remove specific `.nc` files via the `--products` argument  

---

## **Troubleshooting**  

- **Permission Issues?** Try `chmod +x chll_nn.py` before executing  
- **Copernicus API Authentication Failure?** Check your `~/.eumdac/credentials` file  
- **File Not Found Errors?** Ensure the correct **`--products`** list is provided  

---

## **Contributing**  

Contributions are welcome! Feel free to:  
- Submit a **pull request**  
- Open an **issue** for bug reports or feature requests  
- Fork the repo and experiment with **new features**  

---
