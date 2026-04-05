import xarray as xr
import pandas as pd
import numpy as np
import fsspec
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()

AZURE_ACCOUNT_NAME = 'oceanxdc' 
CONTAINER_RAW = 'noaa-wod-data'                    
CONTAINER_PROCESSED = 'processed-noaa-etl' 
SUBFOLDER = 'Surface_Temperature'    

storage_options = {
    'account_name': AZURE_ACCOUNT_NAME,
    'credential': credential
}

def process_surface_temperature(start_year, end_year):
    for year in range(start_year, end_year + 1):
        # Skip the known bad years from your original script
        if year in [1917, 1918]:
            print(f"Skipping known bad year {year}.")
            continue
            
        raw_file_path = f'abfs://{CONTAINER_RAW}/{year}/wod_osd_{year}.nc'
        processed_file_path = f'abfs://{CONTAINER_PROCESSED}/{SUBFOLDER}/{SUBFOLDER}_{year}.parquet'

        print(f"Attempting to process surface temperature for {year}...")
        
        try:
            with fsspec.open(raw_file_path, **storage_options) as f:
                ds = xr.open_dataset(f, engine='h5netcdf', decode_timedelta=True)
                
                temp_rows = ds['Temperature_row_size'].values
                temp_values = ds['Temperature'].values
                z_values = ds['z'].values
                lat = ds['lat'].values
                lon = ds['lon'].values
                time = ds['time'].values

                rows = []
                cursor = 0

                for i, row_count in enumerate(temp_rows):
                    if np.isnan(row_count) or row_count == 0:
                        continue
                        
                    row_count = int(row_count)
                    temperatures = temp_values[cursor:cursor + row_count]
                    depths = z_values[cursor:cursor + row_count]
                    cursor += row_count

                    # Filter: Must not be NaN, and must be surface level (e.g., < 5 meters)
                    valid = (~np.isnan(temperatures)) & (~np.isnan(depths)) & (depths < 5.0)
                    
                    temperatures = temperatures[valid]
                    valid_depths = depths[valid]

                    if len(temperatures) == 0:
                        continue

                    # Your highly efficient vectorized creation
                    df_temp = pd.DataFrame({
                        'time': np.repeat(time[i], len(temperatures)),
                        'lat': np.repeat(lat[i], len(temperatures)),
                        'lon': np.repeat(lon[i], len(temperatures)),
                        'surface_temperature': temperatures,   # Keeping the raw value
                        'year': year                     # Directly using the loop's year variable
                    })
                    rows.append(df_temp)

                if not rows:
                    print(f"  -> No valid surface temperature data found for {year}. Skipping.")
                    continue

                # Combine all valid rows for this year
                df_year = pd.concat(rows, ignore_index=True)
                df_year['time'] = pd.to_datetime(df_year['time'])
                
                print(f"  -> Writing {len(df_year)} surface temperature rows to Parquet...")
                df_year.to_parquet(
                    processed_file_path,
                    engine='pyarrow',
                    storage_options=storage_options,
                    index=False
                )
                print(f"  -> Successfully processed {year}.")

        except FileNotFoundError:
            print(f"  -> File for {year} not found in Azure. Skipping.")
        except Exception as e:
            print(f"  -> Error processing {year}: {e}")

# Run the backfill loop
process_surface_temperature(1800, 2025)