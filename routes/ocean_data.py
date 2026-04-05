"""
Ocean Data API routes - oceanographic data endpoints
"""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from utils import (
    get_yearly_regional_averages, get_temp_oxy_correlation_by_year,
    get_ts_scatter_data, get_stratification_proxy_data,
    get_hovmoller_data, get_climatology_data, get_vertical_profile_data,
    generate_forecast
)

ocean_api = Blueprint('ocean_api', __name__, url_prefix='/api')

REGION_BOUNDS = {
    'global':     {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'tropics':    {'lat_min': -23, 'lat_max': 23, 'lon_min': -180, 'lon_max': 180},
    'arctic':     {'lat_min': 66,  'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'antarctic':  {'lat_min': -90, 'lat_max': -66,'lon_min': -180, 'lon_max': 180},
    'indian':     {'lat_min': -30, 'lat_max': 30, 'lon_min': 20,   'lon_max': 120},
    'pacific':    {'lat_min': -30, 'lat_max': 30, 'lon_min': 120,  'lon_max': -100}
}

def get_request_bounds():
    region = request.args.get('region')
    if region and region in REGION_BOUNDS:
        return REGION_BOUNDS[region]
    return {
        'lat_min': float(request.args.get('lat_min', -90)),
        'lat_max': float(request.args.get('lat_max', 90)),
        'lon_min': float(request.args.get('lon_min', -180)),
        'lon_max': float(request.args.get('lon_max', 180))
    }

@ocean_api.route('/trends')
def trends():
    bounds = get_request_bounds()
    df_temp, df_oxy, df_sal = get_yearly_regional_averages(bounds)

    return jsonify({
        'labels': df_temp['year'].dropna().tolist(),
        'temp_data': df_temp['surface_temp'].round(2).tolist(),
        'labels2': df_oxy['year'].dropna().tolist(),
        'oxy_data': df_oxy['oxygen_mg_L'].round(2).tolist(),
        'labels3': df_sal['year'].dropna().tolist(),
        'sal_data': df_sal['surface_sal'].round(2).tolist()
    })

@ocean_api.route('/forecast/trends')
def forecast_trends():
    bounds = get_request_bounds()
    forecast_years = int(request.args.get('forecast_years', 5))
    df_temp, df_oxy, df_sal = get_yearly_regional_averages(bounds)

    # Filter to historical (e.g., <= 1940 as per old script)
    df_temp = df_temp[df_temp['year'] <= 1940].sort_values('year')
    df_oxy = df_oxy[df_oxy['year'] <= 1940].sort_values('year')
    df_sal = df_sal[df_sal['year'] <= 1940].sort_values('year')

    last_year = max(df_temp['year'].max(), df_oxy['year'].max(), df_sal['year'].max())
    future_years = list(range(int(last_year) + 1, int(last_year) + forecast_years + 1))

    temp_f = generate_forecast(df_temp['surface_temp'], forecast_years)
    oxy_f = generate_forecast(df_oxy['oxygen_mg_L'], forecast_years)
    sal_f = generate_forecast(df_sal['surface_sal'], forecast_years)

    return jsonify({
        'historical': {
            'years': df_temp['year'].tolist(),
            'temperature': df_temp['surface_temp'].round(2).tolist(),
            'oxygen': df_oxy['oxygen_mg_L'].round(2).tolist(),
            'salinity': df_sal['surface_sal'].round(2).tolist()
        },
        'forecast': {
            'years': future_years,
            'temperature': [round(x, 2) for x in temp_f],
            'oxygen': [round(x, 2) for x in oxy_f],
            'salinity': [round(x, 2) for x in sal_f]
        }
    })

@ocean_api.route('/temp_oxy_correlation')
def temp_oxy_correlation():
    bounds = get_request_bounds()
    df = get_temp_oxy_correlation_by_year(bounds).dropna()
    return jsonify({
        'years': df['year'].tolist(),
        'correlation': df['correlation'].round(2).tolist()
    })

@ocean_api.route('/ts_scatter_by_year')
def ts_scatter_by_year():
    year = int(request.args.get('year'))
    bounds = get_request_bounds()
    df = get_ts_scatter_data(year, bounds).dropna()
    return jsonify({
        'temp': df['temp'].round(2).tolist(),
        'sal': df['sal'].round(2).tolist(),
        'depth': df['depth'].round(1).tolist()  
    })

@ocean_api.route('/strat_proxy_by_year')
def strat_proxy_by_year():
    year = int(request.args.get('year'))
    bounds = get_request_bounds()
    df = get_stratification_proxy_data(year, bounds).dropna()
    return jsonify({
        'lat': df['lat'].round(2).tolist(),
        'lon': df['lon'].round(2).tolist(),
        'delta_T': df['delta_T'].round(2).tolist()
    })

@ocean_api.route('/climatology_anomalies')
def climatology_anomalies():
    bounds = get_request_bounds()
    df_monthly = get_climatology_data(bounds)

    climatology = df_monthly.groupby('month')['surface_temp'].mean().reset_index(name='climatology')
    merged = pd.merge(df_monthly, climatology, on='month').sort_values(['year', 'month'])
    merged['anomaly'] = merged['surface_temp'] - merged['climatology']

    return jsonify({
        "months": climatology['month'].tolist(),
        "climatology": climatology['climatology'].round(2).tolist(),
        "years": merged['year'].tolist(),
        "monthly_temp": merged['surface_temp'].round(2).tolist(),
        "anomaly": merged['anomaly'].round(2).tolist()
    })

@ocean_api.route('/vertical_profile')
def vertical_profile():
    year = int(request.args.get('year'))
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    region = request.args.get('region')
    
    bounds = REGION_BOUNDS.get(region) if region else None
    
    df = get_vertical_profile_data(year, float(lat) if lat else 0, float(lon) if lon else 0, 0.5, bounds).dropna()
    
    return jsonify({
        "depth": df['depth'].tolist(),
        "temp": df['temp'].round(2).tolist(),
        "oxygen": df['oxy'].round(2).tolist()
    })

@ocean_api.route('/hovmoller')
def hovmoller():
    variable = request.args.get('var', 'temp')
    bounds = get_request_bounds()
    
    df = get_hovmoller_data(variable, bounds).dropna()
    grid = df.pivot(index='depth', columns='year', values='value').replace([np.nan, np.inf, -np.inf], None)

    return jsonify({
        "year": [int(y) for y in grid.columns.tolist()],
        "depth": [float(z) for z in grid.index.tolist()],
        "values": [[None if v is None else float(round(v, 2)) for v in row] for row in grid.values]
    })