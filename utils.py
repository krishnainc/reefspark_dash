"""
Utility functions for data loading and processing via DuckDB + PyArrow + Azure
"""
from flask.cli import load_dotenv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import duckdb
import pyarrow.dataset as ds
from adlfs import AzureBlobFileSystem
import threading
import os

load_dotenv()

_db_lock = threading.Lock()

# ==========================================
# 1. SETUP AZURE FILESYSTEM (Pure Python)
# ==========================================
CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
fs = AzureBlobFileSystem(connection_string=CONNECTION_STRING)

# ==========================================
# 2. CREATE PYARROW DATASETS 
# ==========================================
ds_surface_temp = ds.dataset("processed-noaa-etl/Surface_Temperature/", filesystem=fs, format="parquet")
ds_surface_oxy = ds.dataset("processed-noaa-etl/Surface_Oxygen/", filesystem=fs, format="parquet")
ds_surface_sal = ds.dataset("processed-noaa-etl/Surface_Salinity/", filesystem=fs, format="parquet")

ds_deep_temp = ds.dataset("processed-noaa-etl/Temperature/", filesystem=fs, format="parquet")
ds_deep_oxy = ds.dataset("processed-noaa-etl/Oxygen/", filesystem=fs, format="parquet")
ds_deep_sal = ds.dataset("processed-noaa-etl/Salinity/", filesystem=fs, format="parquet")

# ==========================================
# 3. INITIALIZE DUCKDB
# ==========================================
db = duckdb.connect(':memory:')

# Register datasets for SQL access
db.register("ds_surface_temp", ds_surface_temp)
db.register("ds_surface_oxy", ds_surface_oxy)
db.register("ds_surface_sal", ds_surface_sal)
db.register("ds_deep_temp", ds_deep_temp)
db.register("ds_deep_oxy", ds_deep_oxy)
db.register("ds_deep_sal", ds_deep_sal)

def get_bounds_sql(b, table_alias=""):
    """Helper to generate SQL WHERE clauses for geographic bounds."""
    prefix = f"{table_alias}." if table_alias else ""
    lat_clause = f"{prefix}lat BETWEEN {b['lat_min']} AND {b['lat_max']}"
    if b['lon_min'] < b['lon_max']:
        lon_clause = f"{prefix}lon BETWEEN {b['lon_min']} AND {b['lon_max']}"
    else:
        lon_clause = f"({prefix}lon >= {b['lon_min']} OR {prefix}lon <= {b['lon_max']})"
    return f"{lat_clause} AND {lon_clause}"

# ==========================================
# 4. DATA LAKE QUERIES
# ==========================================

def get_yearly_regional_averages(bounds):
    b_sql = get_bounds_sql(bounds)

    with _db_lock:
        # Surface Temp: confirmed 'surface_temperature'
        df_temp = db.execute(f"SELECT year, avg(surface_temperature) as surface_temp FROM ds_surface_temp WHERE {b_sql} GROUP BY year ORDER BY year").df()
        # Surface Oxy: confirmed 'oxygen_mg_L'
        df_oxy = db.execute(f"SELECT year, avg(oxygen_mg_L) as oxygen_mg_L FROM ds_surface_oxy WHERE {b_sql} GROUP BY year ORDER BY year").df()
        # Surface Sal: confirmed 'surface_salinity'
        df_sal = db.execute(f"SELECT year, avg(surface_salinity) as surface_sal FROM ds_surface_sal WHERE {b_sql} GROUP BY year ORDER BY year").df()
    
    return df_temp, df_oxy, df_sal

def get_temp_oxy_correlation_by_year(bounds):
    b_sql = get_bounds_sql(bounds, "t")
    query = f"""
        SELECT t.year, corr(t.surface_temperature, o.oxygen_mg_L) as correlation
        FROM ds_surface_temp t
        JOIN ds_surface_oxy o 
          ON t.time = o.time AND t.lat = o.lat AND t.lon = o.lon
        WHERE {b_sql}
        GROUP BY t.year ORDER BY t.year
    """
    with _db_lock:
        return db.execute(query).df()

def get_ts_scatter_data(year_val, bounds):
    b_sql = get_bounds_sql(bounds, "t")
    # Deep columns: 'temperature' and 'salinity'
    query = f"""
        SELECT t.temperature as temp, s.salinity as sal, t.z as depth
        FROM ds_deep_temp t
        JOIN ds_deep_sal s 
          ON t.time = s.time AND t.lat = s.lat AND t.lon = s.lon AND t.z = s.z
        WHERE year(t.time) = {year_val} AND {b_sql}
    """
    with _db_lock:
        return db.execute(query).df()

def get_stratification_proxy_data(year_val, bounds):
    b_sql = get_bounds_sql(bounds, "t_surf")
    # Deep column: 'temperature'
    query = f"""
        SELECT t_surf.lat, t_surf.lon, (t_surf.temperature - t_deep.temperature) as delta_T
        FROM ds_deep_temp t_surf
        JOIN ds_deep_temp t_deep
          ON t_surf.time = t_deep.time AND t_surf.lat = t_deep.lat AND t_surf.lon = t_deep.lon
        WHERE year(t_surf.time) = {year_val} 
          AND CAST(t_surf.z AS INT) = 0 
          AND CAST(t_deep.z AS INT) = 200
          AND {b_sql}
    """
    with _db_lock:
        return db.execute(query).df()

def get_hovmoller_data(var_type, bounds):
    dataset_map = {'temp': 'ds_deep_temp', 'oxy': 'ds_deep_oxy', 'sal': 'ds_deep_sal'}
    # Deep metrics: 'temperature', 'oxygen', 'salinity'
    metric_map = {'temp': 'temperature', 'oxy': 'oxygen', 'sal': 'salinity'}
    
    dataset_name = dataset_map.get(var_type)
    metric = metric_map.get(var_type)
    b_sql = get_bounds_sql(bounds)
    
    query = f"""
        SELECT year(time) as year, round(z, -1) as depth, avg({metric}) as value
        FROM {dataset_name}
        WHERE {b_sql}
        GROUP BY year(time), round(z, -1)
    """
    with _db_lock:
        return db.execute(query).df()

def get_climatology_data(bounds):
    b_sql = get_bounds_sql(bounds)
    query = f"""
        SELECT year, month(time) as month, avg(surface_temperature) as surface_temp
        FROM ds_surface_temp
        WHERE {b_sql}
        GROUP BY year, month(time)
    """
    with _db_lock:
        return db.execute(query).df()

def get_vertical_profile_data(year_val, lat, lon, tol, region_bounds=None):
    if region_bounds:
        b_sql = get_bounds_sql(region_bounds, "t")
    else:
        b_sql = f"t.lat BETWEEN {lat-tol} AND {lat+tol} AND t.lon BETWEEN {lon-tol} AND {lon+tol}"

    # Deep metrics: 'temperature' and 'oxygen'
    query = f"""
        SELECT t.z as depth, avg(t.temperature) as temp, avg(o.oxygen) as oxy
        FROM ds_deep_temp t
        JOIN ds_deep_oxy o 
          ON t.time = o.time AND t.lat = o.lat AND t.lon = o.lon AND t.z = o.z
        WHERE year(t.time) = {year_val} AND {b_sql}
        GROUP BY t.z ORDER BY t.z
    """
    with _db_lock:
        return db.execute(query).df()

# ==========================================
# FORECASTING & LITTER (Unchanged)
# ==========================================
def load_litter_df():
    df = pd.read_csv("data/litter_dataset_final_data.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

def generate_forecast(data, periods, window_size=15):
    if len(data) > window_size:
        data = data[-window_size:]
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data.values
    model = LinearRegression().fit(X, y)
    trend, last_val = model.coef_[0], y[-1]
    hist_std = np.std(y)

    forecasts = []
    for i in range(periods):
        next_val = last_val + trend * (i + 1)
        noise = np.random.normal(0, hist_std * 0.1 * (1 - i/periods))
        forecasts.append(next_val + noise)
        last_val = next_val 
    return forecasts

def make_smooth_forecast(values, periods, window=10):
    y = np.array(values, dtype=float)[-window:] if len(values) >= window else np.array(values, dtype=float)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    slope, last_val = float(model.coef_[0]), float(y[-1])
    hist_std = float(np.std(y)) if len(y) > 1 else 0.0

    forecasts = []
    for i in range(periods):
        next_base = last_val + slope
        noise = np.random.normal(0, hist_std * 0.1 * (1 + i / periods))
        forecasts.append(float(np.clip(next_base + noise, 0.0, 1.0)))
        last_val = next_base
    return forecasts