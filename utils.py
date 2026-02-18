"""
Utility functions for data loading and processing
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def load_litter_df():
    """Load and normalize the litter dataset."""
    df = pd.read_csv("data/litter_dataset_final_data.csv")
    # normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_ocean_data(var_type='temp'):
    """Load ocean depth variable data (temp, oxy, etc.)"""
    return pd.read_csv(f"data/vardepth_{var_type}.csv")


def load_surface_data(var_type='temp'):
    """Load surface-level ocean data."""
    return pd.read_csv(f"data/cleaned_surface_{var_type}.csv")


def parse_dates(df, time_col='time'):
    """Parse datetime column and extract year/month."""
    df[time_col] = pd.to_datetime(df[time_col])
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    return df


def filter_region(df, region, region_bounds):
    """Filter dataframe by geographic region."""
    b = region_bounds.get(region, region_bounds.get('global'))
    
    # Handle Pacific dateline wrap-around
    if b['lon_min'] < b['lon_max']:
        return df[(df['lat'].between(b['lat_min'], b['lat_max'])) & 
                  (df['lon'].between(b['lon_min'], b['lon_max']))]
    else:
        return df[(df['lat'].between(b['lat_min'], b['lat_max'])) & 
                  ((df['lon'] >= b['lon_min']) | (df['lon'] <= b['lon_max']))]


def generate_forecast(data, periods, window_size=15):
    """Generate forecast using linear regression with noise."""
    # Prepare data
    if len(data) > window_size:
        historical_data = data[-window_size:]  # Use last window_size points
        X = np.array(range(len(historical_data))).reshape(-1, 1)
        y = historical_data.values
    else:
        X = np.array(range(len(data))).reshape(-1, 1)
        y = data.values

    # Fit model on historical data
    model = LinearRegression()
    model.fit(X, y)

    # Get last historical value and trend
    last_value = y[-1]
    trend = model.coef_[0]

    # Generate future points
    forecast = []
    historical_std = np.std(y)

    for i in range(periods):
        # Calculate base prediction
        next_value = last_value + trend * (i + 1)
        
        # Add decreasing noise based on historical variance
        noise_scale = historical_std * 0.1 * (1 - i/periods)  # Reduce noise over time
        noise = np.random.normal(0, noise_scale)
        
        forecast.append(next_value + noise)
        last_value = next_value  # Update for next iteration

    return forecast


def make_smooth_forecast(values, periods, window=10):
    """Generate smooth forecast from values array."""
    arr = np.array(values, dtype=float)
    if len(arr) >= window:
        y = arr[-window:]
    else:
        y = arr

    # x as 0..n-1
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    slope = float(model.coef_[0])
    last_val = float(y[-1])
    hist_std = float(np.std(y)) if len(y) > 1 else 0.0

    forecasts = []
    current = last_val
    for i in range(periods):
        # predict next by linear step
        next_base = current + slope
        # noise scaled to historical variability and horizon
        noise = np.random.normal(0, hist_std * 0.1 * (1 + i / periods))
        next_val = next_base + noise
        # clip to [0,1]
        next_val = float(np.clip(next_val, 0.0, 1.0))
        forecasts.append(next_val)
        current = next_base

    return forecasts


def select_surface(df):
    """Select surface values from dataframe (z == 0.0 or z <= 5.0)."""
    if (df['z'] == 0.0).any():
        return df[df['z'] == 0.0].copy()
    else:
        return df[df['z'] <= 5.0].copy()


def clean_coordinates(df, precision=2):
    """Round latitude, longitude for merge stability."""
    df['lat'] = df['lat'].round(precision)
    df['lon'] = df['lon'].round(precision)
    return df
