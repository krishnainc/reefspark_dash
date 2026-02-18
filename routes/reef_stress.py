"""
Reef Stress API routes - reef health and stress index endpoints
"""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from utils import (
    load_ocean_data,
    parse_dates,
    make_smooth_forecast,
    select_surface,
    clean_coordinates
)

reef_api = Blueprint('reef_api', __name__, url_prefix='/api')

# Define region bounds for reuse
REGION_BOUNDS = {
    'global':  {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'tropics': {'lat_min': -23, 'lat_max': 23, 'lon_min': -180, 'lon_max': 180},
    'arctic':  {'lat_min': 66,  'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'antarctic': {'lat_min': -90, 'lat_max': -66, 'lon_min': -180, 'lon_max': 180},
    'indian':  {'lat_min': -30, 'lat_max': 30, 'lon_min': 20, 'lon_max': 120},
    'pacific': {'lat_min': -30, 'lat_max': 30, 'lon_min': 120, 'lon_max': -100}
}


@reef_api.route('/reef_stress')
def reef_stress():
    """
    Returns yearly Reef Stress Index (RSI) for a selected region.
    Query params:
      - region (default 'global')
      - start (optional, int year)
      - end (optional, int year)
      - w_tsi, w_hci, w_osi (optional weights, default 0.5,0.3,0.2)
    Response:
      {
        "years": [...],
        "TSI": [...],      # thermal stress index (0-1)
        "HCI": [...],      # stratification / heat content index  (0-1)
        "OSI": [...],      # oxygen stress index (0-1)
        "RSI": [...],      # combined index (0-1)
      }
    """
    # --- user params ---
    region = request.args.get('region', 'global')
    start = request.args.get('start', None)
    end = request.args.get('end', None)
    # weights (must sum to 1 ideally)
    w_tsi = float(request.args.get('w_tsi', 0.5))
    w_hci = float(request.args.get('w_hci', 0.3))
    w_osi = float(request.args.get('w_osi', 0.2))

    # safe normalization of weights
    total_w = (w_tsi + w_hci + w_osi)
    if total_w == 0:
        w_tsi, w_hci, w_osi = 0.5, 0.3, 0.2
    else:
        w_tsi, w_hci, w_osi = w_tsi/total_w, w_hci/total_w, w_osi/total_w

    b = REGION_BOUNDS.get(region, REGION_BOUNDS['global'])

    # --- load data (depth datasets) ---
    df_temp = load_ocean_data('temp')
    df_oxy = load_ocean_data('oxy')

    # basic cleaning/parsing
    for df in (df_temp, df_oxy):
        df.dropna(subset=['time', 'lat', 'lon', 'z'], inplace=True)
        df = parse_dates(df, 'time')
        df.dropna(subset=['time'], inplace=True)

    # filter region (handle pacific dateline wrap)
    if b['lon_min'] < b['lon_max']:
        df_temp = df_temp[(df_temp['lat'].between(b['lat_min'], b['lat_max'])) &
                          (df_temp['lon'].between(b['lon_min'], b['lon_max']))]
        df_oxy  = df_oxy[(df_oxy['lat'].between(b['lat_min'], b['lat_max'])) &
                         (df_oxy['lon'].between(b['lon_min'], b['lon_max']))]
    else:
        df_temp = df_temp[(df_temp['lat'].between(b['lat_min'], b['lat_max'])) &
                          ((df_temp['lon'] >= b['lon_min']) | (df_temp['lon'] <= b['lon_max']))]
        df_oxy  = df_oxy[(df_oxy['lat'].between(b['lat_min'], b['lat_max'])) &
                         ((df_oxy['lon'] >= b['lon_min']) | (df_oxy['lon'] <= b['lon_max']))]

    # optional year window
    if start:
        start = int(start)
        df_temp = df_temp[df_temp['year'] >= start]
        df_oxy  = df_oxy[df_oxy['year'] >= start]
    if end:
        end = int(end)
        df_temp = df_temp[df_temp['year'] <= end]
        df_oxy  = df_oxy[df_oxy['year'] <= end]

    # --- COMPONENT 1: TSI (Thermal Stress Index) ---
    # Use surface values only (z == 0.0) when available. If exact 0 missing, use nearest shallow depth (z <= 5m)
    surf_temp = select_surface(df_temp)
    surf_oxy  = select_surface(df_oxy)

    # compute climatology
    if surf_temp.empty:
        return jsonify({"years": [], "TSI": [], "HCI": [], "OSI": [], "RSI": []})

    # Simpler: compute long-term mean and std at surface across all years
    longmean_temp = surf_temp['temp'].mean()
    longstd_temp  = surf_temp['temp'].std(ddof=0) if surf_temp['temp'].std(ddof=0) > 0 else 1.0

    # compute yearly mean surface temp and anomaly standardized
    yearly_surf_temp = surf_temp.groupby('year')['temp'].mean().reset_index(name='temp_mean')
    yearly_surf_temp['TSI_raw'] = (yearly_surf_temp['temp_mean'] - longmean_temp) / longstd_temp
    # convert TSI_raw to 0-1 by mapping -3..+3 sigma -> 0..1 (clamp)
    yearly_surf_temp['TSI'] = yearly_surf_temp['TSI_raw'].clip(-3, 3).apply(lambda x: (x + 3) / 6.0)

    # --- COMPONENT 2: HCI (Heat / Stratification Index) ---
    # For each cast-date find temp at z==0 and z==200 (or nearest). Then compute delta = temp_surface - temp_200
    # We'll compute yearly mean of delta across casts
    df_temp['z_round'] = df_temp['z'].round(0)
    # surface temp per cast
    surf = df_temp[df_temp['z_round'] == 0].rename(columns={'temp': 'temp_surface'})[['time','lat','lon','year','temp_surface']]
    depth200 = df_temp[df_temp['z_round'] == 200].rename(columns={'temp': 'temp_200'})[['time','lat','lon','year','temp_200']]

    # merge on time/lat/lon/year to get pairs; use inner merge
    merged200 = pd.merge(surf, depth200, on=['time','lat','lon','year'], how='inner')
    if merged200.empty:
        # fallback: compute surface minus mean of deeper layer (e.g., mean temp between 150-250m)
        deeper = df_temp[(df_temp['z'] >= 150) & (df_temp['z'] <= 250)]
        deeper_mean = deeper.groupby(['time','lat','lon','year'])['temp'].mean().reset_index(name='temp_200')
        merged200 = pd.merge(surf, deeper_mean, on=['time','lat','lon','year'], how='inner')

    if merged200.empty:
        # if still empty, HCI zeros
        yearly_hci = pd.DataFrame({'year': yearly_surf_temp['year'], 'HCI': 0.0})
    else:
        merged200['delta_T'] = merged200['temp_surface'] - merged200['temp_200']
        yearly_hci = merged200.groupby('year')['delta_T'].mean().reset_index(name='delta_mean')
        # normalize to 0-1 by clipping to plausible range e.g., 0..10 degC
        ymin, ymax = 0.0, 10.0
        yearly_hci['HCI'] = yearly_hci['delta_mean'].clip(ymin, ymax).apply(lambda x: (x - ymin) / (ymax - ymin))

    # --- COMPONENT 3: OSI (Oxygen Stress Index) ---
    # Use surface oxygen mean per year: low oxygen -> higher stress.
    yearly_surf_oxy = surf_oxy.groupby('year')['oxy'].mean().reset_index(name='oxy_mean')
    # We'll convert oxygen into stress index: high O2 -> low stress; so invert and normalize.
    # Determine realistic bounds (example): oxy 0..10 mg/L (0 worst, 10 best)
    oxy_min, oxy_max = 0.0, 10.0
    yearly_surf_oxy['oxy_clip'] = yearly_surf_oxy['oxy_mean'].clip(oxy_min, oxy_max)
    yearly_surf_oxy['OSI'] = yearly_surf_oxy['oxy_clip'].apply(lambda x: 1.0 - ((x - oxy_min) / (oxy_max - oxy_min)))

    # --- MERGE yearly components into a single table ---
    years = sorted(set(yearly_surf_temp['year'].tolist() +
                       yearly_hci['year'].tolist() +
                       yearly_surf_oxy['year'].tolist()))

    df_yearly = pd.DataFrame({'year': years})
    df_yearly = df_yearly.merge(yearly_surf_temp[['year','TSI']], on='year', how='left')
    df_yearly = df_yearly.merge(yearly_hci[['year','HCI']], on='year', how='left')
    df_yearly = df_yearly.merge(yearly_surf_oxy[['year','OSI']], on='year', how='left')

    # fill missing component values conservatively
    df_yearly['TSI'] = df_yearly['TSI'].fillna(0.5)  # neutral
    df_yearly['HCI'] = df_yearly['HCI'].fillna(0.0)
    df_yearly['OSI'] = df_yearly['OSI'].fillna(0.5)

    # Combined RSI (0 low stress -> 1 high stress)
    df_yearly['RSI_raw'] = w_tsi * df_yearly['TSI'] + w_hci * df_yearly['HCI'] + w_osi * df_yearly['OSI']
    # Clip to [0,1] to be safe
    df_yearly['RSI'] = df_yearly['RSI_raw'].clip(0.0, 1.0)

    # filter start/end if explicitly provided & ensure sorted
    if start:
        df_yearly = df_yearly[df_yearly['year'] >= start]
    if end:
        df_yearly = df_yearly[df_yearly['year'] <= end]
    df_yearly = df_yearly.sort_values('year')

    return jsonify({
        "years": df_yearly['year'].astype(int).tolist(),
        "TSI": df_yearly['TSI'].round(3).tolist(),
        "HCI": df_yearly['HCI'].round(3).tolist(),
        "OSI": df_yearly['OSI'].round(3).tolist(),
        "RSI": df_yearly['RSI'].round(3).tolist(),
        "weights": {"w_tsi": w_tsi, "w_hci": w_hci, "w_osi": w_osi}
    })


@reef_api.route('/forecast/rsi')
def forecast_rsi():
    """Generate forecast for Reef Stress Index (RSI) components."""
    region = request.args.get('region', 'global')
    forecast_years = int(request.args.get('forecast_years', 5))
    
    # Get historical RSI data first
    rsi_response = reef_stress()
    rsi_data = rsi_response.get_json()
    
    # Prepare data for forecasting (use last N points to compute trend)
    years = rsi_data['years']
    rsi_values = rsi_data['RSI']
    tsi_values = rsi_data['TSI']
    osi_values = rsi_data['OSI']

    if not years or not rsi_values or not tsi_values or not osi_values:
        return jsonify({
            'historical': rsi_data,
            'forecast': {
                'years': [],
                'RSI': [],
                'TSI': [],
                'OSI': []
            }
        })

    last_year = int(years[-1])
    future_years = list(range(last_year + 1, last_year + forecast_years + 1))

    rsi_forecast = make_smooth_forecast(rsi_values, forecast_years, window=10)
    tsi_forecast = make_smooth_forecast(tsi_values, forecast_years, window=10)
    osi_forecast = make_smooth_forecast(osi_values, forecast_years, window=10)
    
    return jsonify({
        'historical': {
            'years': rsi_data['years'],
            'RSI': rsi_data['RSI'],
            'TSI': rsi_data['TSI'],
            'OSI': rsi_data['OSI']
        },
        'forecast': {
            'years': future_years,
            'RSI': [round(x, 3) for x in rsi_forecast],
            'TSI': [round(x, 3) for x in tsi_forecast],
            'OSI': [round(x, 3) for x in osi_forecast]
        }
    })
