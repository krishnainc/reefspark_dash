import xarray as xr
import pandas as pd
import numpy as np
import fsspec
from azure.identity import DefaultAzureCredential

# 1. RBAC Authentication (Zero secrets in code)
# This automatically uses the VM's Managed Identity
credential = DefaultAzureCredential()

AZURE_ACCOUNT_NAME = 'oceanxdc' 
CONTAINER_RAW = 'noaa-wod-data'                   
CONTAINER_PROCESSED = 'processed-noaa-etl'
SUBFOLDER = 'Salinity'    

storage_options = {
    'account_name': AZURE_ACCOUNT_NAME,
    'credential': credential
}

def process_salinity_to_parquet(start_year, end_year):
    for year in range(start_year, end_year + 1):
        raw_file_path = f'abfs://{CONTAINER_RAW}/{year}/wod_osd_{year}.nc'
        processed_file_path = f'abfs://{CONTAINER_PROCESSED}/{SUBFOLDER}/{SUBFOLDER}_{year}.parquet'

        print(f"Attempting to process year {year}...")
        
        try:
            # Stream directly from Azure
            with fsspec.open(raw_file_path, **storage_options) as f:
                ds = xr.open_dataset(f, engine='h5netcdf', decode_timedelta=True)
                
                # Load values into memory for this year only
                sal_rows = ds['Salinity_row_size'].values
                sal_values = ds['Salinity'].values
                z_values = ds['z'].values
                lat = ds['lat'].values
                lon = ds['lon'].values
                time = ds['time'].values

                rows = []
                cursor = 0
                
                # Your exact parsing logic
                for i, row_count in enumerate(sal_rows):
                    if np.isnan(row_count) or row_count == 0:
                        continue 

                    row_count = int(row_count)
                    sals = sal_values[cursor:cursor + row_count]
                    depths = z_values[cursor:cursor + row_count]
                    cursor += row_count

                    if len(sals) == 0 or len(depths) == 0:
                        continue

                    for sal, z in zip(sals, depths):
                        if pd.isna(sal) or pd.isna(z):
                            continue
                        rows.append({
                            'time': time[i],
                            'lat': lat[i],
                            'lon': lon[i],
                            'z': z,
                            'salinity': sal
                        })

                # Gracefully skip if the year has the file but no salinity data
                if not rows:
                    print(f"  -> No valid salinity data found for {year}. Skipping.")
                    continue

                # Create DataFrame for just this year
                df = pd.DataFrame(rows)
                df['time'] = pd.to_datetime(df['time'])

                # Save as a distinct Parquet file back to Azure
                print(f"  -> Writing {len(df)} rows to Parquet...")
                df.to_parquet(
                    processed_file_path,
                    engine='pyarrow',
                    storage_options=storage_options,
                    index=False
                )
                print(f"  -> Successfully processed {year}.")
                
        except FileNotFoundError:
            # Handles years where NOAA has no data file at all
            print(f"  -> File for {year} not found in Azure. Skipping.")
        except Exception as e:
            print(f"  -> Error processing {year}: {e}")

# Run the full historical backfill loop
process_salinity_to_parquet(1800, 2025)