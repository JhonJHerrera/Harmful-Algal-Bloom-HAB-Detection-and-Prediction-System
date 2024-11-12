import os
import pandas as pd
from pathlib import Path      
from datetime import datetime

def summarize_chlorophyll_data(products_dir, output_csv="chlorophyll_summary_with_datetime.csv"):
    # Ruta del archivo de directorios
    directories_list_file = os.path.join(products_dir, 'directories_list.txt')
    
    # Comprobar que el archivo existe
    if not os.path.exists(directories_list_file):
        print(f"No se encontró el archivo: {directories_list_file}")
        return
    
    # Leer los directorios desde el archivo
    with open(directories_list_file, 'r') as file:
        directories = [line.strip() for line in file.readlines()]
    
    # Lista para almacenar los datos resumen
    summary_data = []

    # Iterar sobre cada directorio listado
    for directory in directories:
        # Extraer fecha y hora del nombre del directorio
        try:
            datetime_obj = datetime.strptime(directory, "%Y%m%dT%H%M%S")
            date = datetime_obj.strftime("%Y-%m-%d")
            hours = datetime_obj.strftime("%H:%M:%S")
        except ValueError:
            print(f"Formato de fecha no válido en el nombre del directorio: {directory}")
            continue

        # Construir la ruta del archivo CSV en cada directorio
        directory_path = os.path.join(products_dir, directory)
        csv_file_path = os.path.join(directory_path, 'chlorophyll_roi.csv')
        
        # Comprobar si el archivo CSV existe en el directorio
        if os.path.exists(csv_file_path):
            # Leer el CSV
            df = pd.read_csv(csv_file_path)
            
            # Calcular los promedios
            chl_nn_mean = df['chlorophyll'].mean() if 'chlorophyll' in df.columns else None
            chl_oc4me_mean = df['chlorophyll_oc4me'].mean() if 'chlorophyll_oc4me' in df.columns else None
            
            # Agregar los resultados al resumen
            summary_data.append({
                "date": date,
                "hours": hours,
                "chl_nn_mean": chl_nn_mean,
                "chl_oc4me_mean": chl_oc4me_mean
            })
        else:
            print(f"Archivo CSV no encontrado en el directorio: {directory_path}")

    # Convertir los datos resumen en un DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Guardar el resumen en un nuevo archivo CSV
    summary_df.to_csv(output_csv, index=False)
    print(f"Resumen de clorofila guardado en: {output_csv}")

    return summary_df



def main():
    products_dir = os.path.join(Path.home(), "products")
    summary_df = summarize_chlorophyll_data(products_dir)

    # Mostrar el DataFrame resumen
    print("Resumen de clorofila por directorio:")
    print(summary_df)

if __name__ == "__main__":
    main()