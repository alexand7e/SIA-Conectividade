import pandas as pd
import os

def read_excel_files(path):
    dataframes = {}
    
    # Check if the directory exists
    if not os.path.exists(path):
        print(f"The directory {path} does not exist.")
        return dataframes

    # List all files in the folder
    files = os.listdir(path)
    
    # Read each Excel file and store in a dataframe
    for file in files:
        if file.endswith('.xlsx'):
            file_path = os.path.join(path, file)
            df_name = file.replace('.xlsx', '').replace(' ', '_').lower()
            try:
                dataframes[df_name] = pd.read_excel(file_path)
                print(f"Successfully read {file}")
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")

    return dataframes

# Define the path to the folder
path = os.path.join(os.path.dirname(__file__), "..", "data", "excel")

# Execute the function and store the dataframes
dataframes = read_excel_files(path)

# Example of how to access the dataframes
for name, df in dataframes.items():
    print(f"\nDataFrame: {name}")
    print(df.head())
    print(f"Shape: {df.shape}")