"""
Ocean Data API routes - oceanographic data endpoints
"""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from utils import (
    load_surface_data, 
    load_ocean_data, 
    parse_dates, 
    filter_region, 
    generate_forecast,
    clean_coordinates,
    select_surface
)

ocean_api = Blueprint('ocean_api', __name__, url_prefix='/api')

# Define region bounds for reuse
REGION_BOUNDS = {
    'global':     {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'tropics':    {'lat_min': -23, 'lat_max': 23, 'lon_min': -180, 'lon_max': 180},
    'arctic':     {'lat_min': 66,  'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    'antarctic':  {'lat_min': -90, 'lat_max': -66,'lon_min': -180, 'lon_max': 180},
    'indian':     {'lat_min': -30, 'lat_max': 30, 'lon_min': 20,   'lon_max': 120},
    'pacific':    {'lat_min': -30, 'lat_max': 30, 'lon_min': 120,  'lon_max': -100}
}


@ocean_api.route('/trends')
def trends():
    """Get temperature, oxygen, and salinity trends by region."""
    df_temp = load_surface_data('temp')
    df_oxy = load_surface_data('oxy')
    df_sal = load_surface_data('sal')

    # Clean data - remove invalid lat/lon values (e.g., -999 fill values)
    for df in [df_temp, df_oxy, df_sal]:
        df = df[(df['lat'] >= -90) & (df['lat'] <= 90) & 
                (df['lon'] >= -180) & (df['lon'] <= 180)]

    # Parse dates
    for df in [df_temp, df_oxy, df_sal]:
        df = parse_dates(df)

    # Get region from query string
    region = request.args.get('region', None)
    
    # Get bounds from region or use custom parameters
    if region and region in REGION_BOUNDS:
        b = REGION_BOUNDS[region]
    else:
        # Fallback to custom lat/lon parameters
        b = {
            'lat_min': float(request.args.get('lat_min', -90)),
            'lat_max': float(request.args.get('lat_max', 90)),
            'lon_min': float(request.args.get('lon_min', -180)),
            'lon_max': float(request.args.get('lon_max', 180))
        }

    # Filter by selected region (handle Pacific dateline wrap-around)
    if b['lon_min'] < b['lon_max']:
        df_temp = df_temp[(df_temp['lat'].between(b['lat_min'], b['lat_max'])) & 
                          (df_temp['lon'].between(b['lon_min'], b['lon_max']))]
        df_oxy = df_oxy[(df_oxy['lat'].between(b['lat_min'], b['lat_max'])) & 
                        (df_oxy['lon'].between(b['lon_min'], b['lon_max']))]
        df_sal = df_sal[(df_sal['lat'].between(b['lat_min'], b['lat_max'])) & 
                        (df_sal['lon'].between(b['lon_min'], b['lon_max']))]
    else:
        # Pacific crosses dateline
        df_temp = df_temp[(df_temp['lat'].between(b['lat_min'], b['lat_max'])) & 
                          ((df_temp['lon'] >= b['lon_min']) | (df_temp['lon'] <= b['lon_max']))]
        df_oxy = df_oxy[(df_oxy['lat'].between(b['lat_min'], b['lat_max'])) & 
                        ((df_oxy['lon'] >= b['lon_min']) | (df_oxy['lon'] <= b['lon_max']))]
        df_sal = df_sal[(df_sal['lat'].between(b['lat_min'], b['lat_max'])) & 
                        ((df_sal['lon'] >= b['lon_min']) | (df_sal['lon'] <= b['lon_max']))]

    yearly_temp = df_temp.groupby('year').agg({'surface_temp': 'mean'}).reset_index()
    yearly_oxy = df_oxy.groupby('year').agg({'oxygen_mg_L': 'mean'}).reset_index()
    yearly_sal = df_sal.groupby('year').agg({'surface_sal': 'mean'}).reset_index()

    return jsonify({
        'labels': yearly_temp['year'].tolist(),
        'temp_data': yearly_temp['surface_temp'].round(2).tolist(),
        'labels2': yearly_oxy['year'].tolist(),
        'oxy_data': yearly_oxy['oxygen_mg_L'].round(2).tolist(),
        'labels3': yearly_sal['year'].tolist(),
        'sal_data': yearly_sal['surface_sal'].round(2).tolist()
    })


@ocean_api.route('/forecast/trends')
def forecast_trends():
    """Generate forecasts for temperature, oxygen, and salinity trends."""
    df_temp = load_surface_data('temp')
    df_oxy = load_surface_data('oxy')
    df_sal = load_surface_data('sal')

    region = request.args.get('region', 'global')
    forecast_years = int(request.args.get('forecast_years', 5))

    # Parse timestamps
    for df in [df_temp, df_oxy, df_sal]:
        df = parse_dates(df)

    # Region filtering based on bounds
    b = REGION_BOUNDS.get(region, REGION_BOUNDS['global'])

    # Filter data by region
    if b['lon_min'] < b['lon_max']:
        df_temp = df_temp[(df_temp['lat'].between(b['lat_min'], b['lat_max'])) &
                          (df_temp['lon'].between(b['lon_min'], b['lon_max']))]
        df_oxy = df_oxy[(df_oxy['lat'].between(b['lat_min'], b['lat_max'])) &
                        (df_oxy['lon'].between(b['lon_min'], b['lon_max']))]
        df_sal = df_sal[(df_sal['lat'].between(b['lat_min'], b['lat_max'])) &
                        (df_sal['lon'].between(b['lon_min'], b['lon_max']))]
    else:
        df_temp = df_temp[(df_temp['lat'].between(b['lat_min'], b['lat_max'])) &
                          ((df_temp['lon'] >= b['lon_min']) | (df_temp['lon'] <= b['lon_max']))]
        df_oxy = df_oxy[(df_oxy['lat'].between(b['lat_min'], b['lat_max'])) &
                        ((df_oxy['lon'] >= b['lon_min']) | (df_oxy['lon'] <= b['lon_max']))]
        df_sal = df_sal[(df_sal['lat'].between(b['lat_min'], b['lat_max'])) &
                        ((df_sal['lon'] >= b['lon_min']) | (df_sal['lon'] <= b['lon_max']))]

    # Work with historical data (1800-1940)
    df_temp = df_temp[(df_temp['year'] >= 1800) & (df_temp['year'] <= 1940)]
    df_oxy = df_oxy[(df_oxy['year'] >= 1800) & (df_oxy['year'] <= 1940)]
    df_sal = df_sal[(df_sal['year'] >= 1800) & (df_sal['year'] <= 1940)]
    
    # Calculate yearly averages
    yearly_temp = df_temp.groupby('year')['surface_temp'].mean().reset_index()
    yearly_oxy = df_oxy.groupby('year')['oxygen_mg_L'].mean().reset_index()
    yearly_sal = df_sal.groupby('year')['surface_sal'].mean().reset_index()

    # Generate forecasts using linear regression
    last_year = max(yearly_temp['year'].max(),
                    yearly_oxy['year'].max(),
                    yearly_sal['year'].max())
    future_years = list(range(last_year + 1, last_year + forecast_years + 1))

    # Create forecasts
    temp_forecast = generate_forecast(yearly_temp['surface_temp'], forecast_years)
    oxy_forecast = generate_forecast(yearly_oxy['oxygen_mg_L'], forecast_years)
    sal_forecast = generate_forecast(yearly_sal['surface_sal'], forecast_years)

    return jsonify({
        'historical': {
            'years': yearly_temp['year'].tolist(),
            'temperature': yearly_temp['surface_temp'].round(2).tolist(),
            'oxygen': yearly_oxy['oxygen_mg_L'].round(2).tolist(),
            'salinity': yearly_sal['surface_sal'].round(2).tolist()
        },
        'forecast': {
            'years': future_years,
            'temperature': [round(x, 2) for x in temp_forecast],
            'oxygen': [round(x, 2) for x in oxy_forecast],
            'salinity': [round(x, 2) for x in sal_forecast]
        }
    })


@ocean_api.route('/temp_oxy_correlation')
def temp_oxy_correlation():
    """Calculate temperature-oxygen correlation."""
    df_temp = load_surface_data('temp')
    df_oxy = load_surface_data('oxy')

    # Region filtering
    lat_min = float(request.args.get('lat_min', -90))
    lat_max = float(request.args.get('lat_max', 90))
    lon_min = float(request.args.get('lon_min', -180))
    lon_max = float(request.args.get('lon_max', 180))

    df_temp = parse_dates(df_temp)
    df_oxy = parse_dates(df_oxy)

    # Filter by region
    df_temp = df_temp[(df_temp['lat'].between(lat_min, lat_max)) & (df_temp['lon'].between(lon_min, lon_max))]
    df_oxy = df_oxy[(df_oxy['lat'].between(lat_min, lat_max)) & (df_oxy['lon'].between(lon_min, lon_max))]

    # Merge & calculate correlation
    df = pd.merge(df_temp, df_oxy, on=['time', 'lat', 'lon'])
    df = df.dropna(subset=['surface_temp', 'oxygen_mg_L'])

    # FIX: Extract year again after merge
    df['year'] = pd.to_datetime(df['time']).dt.year

    corr_by_year = (
        df.groupby('year')
        .apply(lambda g: g['surface_temp'].corr(g['oxygen_mg_L']))
        .dropna()
        .reset_index(name='correlation')
    )

    return jsonify({
        'years': corr_by_year['year'].tolist(),
        'correlation': corr_by_year['correlation'].round(2).tolist()
    })


@ocean_api.route('/ts_scatter_by_year')
def ts_scatter_by_year():
    """Get temperature-salinity scatter plot data for a given year."""
    year = int(request.args.get('year'))
    lat_min = float(request.args.get('lat_min', -90))
    lat_max = float(request.args.get('lat_max', 90))
    lon_min = float(request.args.get('lon_min', -180))
    lon_max = float(request.args.get('lon_max', 180))

    df_temp = load_ocean_data('temp')
    df_sal = load_ocean_data('salinity')

    df_temp = parse_dates(df_temp)
    df_sal = parse_dates(df_sal)

    df_temp = df_temp[(df_temp['year'] == year) & (df_temp['lat'].between(lat_min, lat_max)) & (df_temp['lon'].between(lon_min, lon_max))]
    df_sal = df_sal[(df_sal['year'] == year) & (df_sal['lat'].between(lat_min, lat_max)) & (df_sal['lon'].between(lon_min, lon_max))]

    for df in [df_temp, df_sal]:
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)
        df['z'] = df['z'].round(1)

    df = pd.merge(df_temp, df_sal, on=['time', 'lat', 'lon', 'z'])
    df = df.dropna(subset=['temp', 'salinity'])

    return jsonify({
        'temp': df['temp'].tolist(),
        'sal': df['salinity'].tolist(),
        'depth': df['z'].tolist()  
    })


@ocean_api.route('/strat_proxy_by_year')
def strat_proxy_by_year():
    """Get stratification proxy (temperature difference) by year."""
    year = int(request.args.get('year'))
    lat_min = float(request.args.get('lat_min', -90))
    lat_max = float(request.args.get('lat_max', 90))
    lon_min = float(request.args.get('lon_min', -180))
    lon_max = float(request.args.get('lon_max', 180))

    df = load_ocean_data('temp')
    df = parse_dates(df)
    df = df[df['year'] == year]
    df = df[(df['lat'].between(lat_min, lat_max)) & (df['lon'].between(lon_min, lon_max))]
    df = df.dropna(subset=['temp', 'z'])

    # Round for merge stability
    df['lat'] = df['lat'].round(2)
    df['lon'] = df['lon'].round(2)
    df['z'] = df['z'].round(1)

    # Pivot temperature values by depth
    df_surface = df[df['z'] == 0.0].rename(columns={'temp': 'temp_surface'})
    df_200m = df[df['z'] == 200.0].rename(columns={'temp': 'temp_200m'})

    merged = pd.merge(df_surface, df_200m, on=['time', 'lat', 'lon'], how='inner')
    merged['delta_T'] = merged['temp_surface'] - merged['temp_200m']

    return jsonify({
        'lat': merged['lat'].tolist(),
        'lon': merged['lon'].tolist(),
        'delta_T': merged['delta_T'].round(2).tolist()
    })


@ocean_api.route('/climatology_anomalies')
def climatology_anomalies():
    """Get climatology and temperature anomalies."""
    df_temp = load_surface_data('temp')

    # Parse time
    df_temp = parse_dates(df_temp)

    # Region filtering
    lat_min = float(request.args.get('lat_min', -90))
    lat_max = float(request.args.get('lat_max', 90))
    lon_min = float(request.args.get('lon_min', -180))
    lon_max = float(request.args.get('lon_max', 180))

    # Handle Pacific special case (wrap dateline)
    if lon_min > lon_max:
        df_temp = df_temp[(df_temp['lat'].between(lat_min, lat_max)) &
                        ((df_temp['lon'] >= lon_min) | (df_temp['lon'] <= lon_max))]
    else:
        df_temp = df_temp[(df_temp['lat'].between(lat_min, lat_max)) &
                        (df_temp['lon'].between(lon_min, lon_max))]

    # --- Climatology (mean per month across all years) ---
    climatology = (
        df_temp.groupby('month')['surface_temp']
        .mean()
        .reset_index(name='climatology')
    )

    # --- Monthly means per year (for anomaly calc) ---
    monthly_means = (
        df_temp.groupby(['year', 'month'])['surface_temp']
        .mean()
        .reset_index()
    )

    # Merge with climatology
    merged = pd.merge(monthly_means, climatology, on='month')
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
    """Get vertical temperature and oxygen profiles."""
    year = int(request.args.get('year'))
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    region = request.args.get('region')  # new: "global", "tropics", etc.
    tol = 0.5

    # Load datasets
    df_temp = load_ocean_data('temp')
    df_oxy = load_ocean_data('oxy')

    # Parse time
    for df in [df_temp, df_oxy]:
        df = parse_dates(df)

    # --- REGION FILTERING ---
    if region:
        if region == "global":
            df_temp = df_temp[df_temp['year'] == year]
            df_oxy = df_oxy[df_oxy['year'] == year]
        elif region == "tropics":
            df_temp = df_temp[(df_temp['year'] == year) & (df_temp['lat'].between(-23, 23))]
            df_oxy = df_oxy[(df_oxy['year'] == year) & (df_oxy['lat'].between(-23, 23))]
        elif region == "arctic":
            df_temp = df_temp[(df_temp['year'] == year) & (df_temp['lat'] >= 66)]
            df_oxy = df_oxy[(df_oxy['year'] == year) & (df_oxy['lat'] >= 66)]
        elif region == "antarctic":
            df_temp = df_temp[(df_temp['year'] == year) & (df_temp['lat'] <= -66)]
            df_oxy = df_oxy[(df_oxy['year'] == year) & (df_oxy['lat'] <= -66)]
        elif region == "indian":
            df_temp = df_temp[(df_temp['year'] == year) &
                              (df_temp['lat'].between(-30, 30)) &
                              (df_temp['lon'].between(20, 120))]
            df_oxy = df_oxy[(df_oxy['year'] == year) &
                            (df_oxy['lat'].between(-30, 30)) &
                            (df_oxy['lon'].between(20, 120))]
        elif region == "pacific":
            df_temp = df_temp[(df_temp['year'] == year) &
                              (df_temp['lat'].between(-30, 30)) &
                              (df_temp['lon'].between(120, -100))]
            df_oxy = df_oxy[(df_oxy['year'] == year) &
                            (df_oxy['lat'].between(-30, 30)) &
                            (df_oxy['lon'].between(120, -100))]
    else:
        # --- POINT FILTERING (default) ---
        lat = float(lat)
        lon = float(lon)
        df_temp = df_temp[(df_temp['year'] == year) &
                          (df_temp['lat'].between(lat - tol, lat + tol)) &
                          (df_temp['lon'].between(lon - tol, lon + tol))]
        df_oxy = df_oxy[(df_oxy['year'] == year) &
                        (df_oxy['lat'].between(lat - tol, lat + tol)) &
                        (df_oxy['lon'].between(lon - tol, lon + tol))]

    # Round coords & merge
    for df in [df_temp, df_oxy]:
        if not df.empty:
            df['lat'] = df['lat'].round(2)
            df['lon'] = df['lon'].round(2)
            df['z'] = df['z'].round(1)

    df = pd.merge(df_temp, df_oxy, on=['time','lat','lon','z'], how='inner')
    if df.empty:
        return jsonify({"depth": [], "temp": [], "oxygen": []})

    df = df.dropna(subset=['temp','oxy'])

    # Average profile across casts
    profile = df.groupby('z').agg({
        'temp': 'mean',
        'oxy': 'mean'
    }).reset_index()

    return jsonify({
        "depth": profile['z'].tolist(),
        "temp": profile['temp'].round(2).tolist(),
        "oxygen": profile['oxy'].round(2).tolist()
    })


@ocean_api.route('/hovmoller')
def hovmoller():
    """Get Hovmoller diagram (time-depth) data."""
    variable = request.args.get('var', 'temp')
    region = request.args.get('region', 'global')

    b = REGION_BOUNDS.get(region, REGION_BOUNDS['global'])

    df = load_ocean_data(variable)
    df = df.dropna(subset=['lat', 'lon', 'z', variable])
    df = parse_dates(df)

    # Handle Pacific
    if b['lon_min'] < b['lon_max']:
        df = df[(df['lat'].between(b['lat_min'], b['lat_max'])) &
                (df['lon'].between(b['lon_min'], b['lon_max']))]
    else:
        df = df[(df['lat'].between(b['lat_min'], b['lat_max'])) &
                ((df['lon'] >= b['lon_min']) | (df['lon'] <= b['lon_max']))]

    df_mean = df.groupby(['year', 'z'])[variable].mean().reset_index()
    grid = df_mean.pivot(index='z', columns='year', values=variable)

    # Fill missing with None safely
    grid = grid.replace([np.nan, np.inf, -np.inf], None)

    return jsonify({
        "year": [int(y) for y in grid.columns.tolist()],
        "depth": [float(z) for z in grid.index.tolist()],
        "values": [[None if (pd.isna(v) or v is np.nan) else float(v) for v in row] for row in grid.values]
    })
