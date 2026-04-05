"""
Reef Stress API routes - reef health and stress index endpoints
"""
from flask import Blueprint, request, jsonify
import pandas as pd
from utils import db, get_bounds_sql, _db_lock, get_yearly_regional_averages, make_smooth_forecast  
import duckdb

reef_api = Blueprint('reef_api', __name__, url_prefix='/api')

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
    region = request.args.get('region', 'global')
    start = request.args.get('start', None)
    end = request.args.get('end', None)
    w_tsi = float(request.args.get('w_tsi', 0.5))
    w_hci = float(request.args.get('w_hci', 0.3))
    w_osi = float(request.args.get('w_osi', 0.2))

    total_w = (w_tsi + w_hci + w_osi)
    if total_w == 0:
        w_tsi, w_hci, w_osi = 0.5, 0.3, 0.2
    else:
        w_tsi, w_hci, w_osi = w_tsi/total_w, w_hci/total_w, w_osi/total_w

    bounds = REGION_BOUNDS.get(region, REGION_BOUNDS['global'])
    
    # 1. Get TSI & OSI base data (Surface Averages)
    df_temp, df_oxy, _ = get_yearly_regional_averages(bounds)
    
    # 2. Get HCI base data (Deep Water differences)
    from utils import db, get_bounds_sql # Removed PROCESSED_DATA_PATH
    
    b_sql = get_bounds_sql(bounds, "t_surf")
    query_hci = f"""
        SELECT year(t_surf.time) as year, avg(t_surf.temperature - t_deep.temperature) as delta_mean
        FROM ds_deep_temp t_surf
        JOIN ds_deep_temp t_deep
          ON t_surf.time = t_deep.time AND t_surf.lat = t_deep.lat AND t_surf.lon = t_deep.lon
        WHERE CAST(t_surf.z AS INT) = 0 AND CAST(t_deep.z AS INT) = 200 AND {b_sql}
        GROUP BY year(t_surf.time)
    """
    with _db_lock:
        df_hci = db.execute(query_hci).df()

    # Apply Math
    if df_temp.empty:
         return jsonify({"years": [], "TSI": [], "HCI": [], "OSI": [], "RSI": []})

    # TSI Math
    longmean_temp = df_temp['surface_temp'].mean()
    longstd_temp = df_temp['surface_temp'].std(ddof=0) if df_temp['surface_temp'].std(ddof=0) > 0 else 1.0
    df_temp['TSI_raw'] = (df_temp['surface_temp'] - longmean_temp) / longstd_temp
    df_temp['TSI'] = df_temp['TSI_raw'].clip(-3, 3).apply(lambda x: (x + 3) / 6.0)

    # HCI Math
    if df_hci.empty:
        df_hci = pd.DataFrame({'year': df_temp['year'], 'HCI': 0.0})
    else:
        df_hci['HCI'] = df_hci['delta_mean'].clip(0.0, 10.0).apply(lambda x: x / 10.0)

    # OSI Math
    if not df_oxy.empty:
        df_oxy['OSI'] = df_oxy['oxygen_mg_L'].clip(0.0, 10.0).apply(lambda x: 1.0 - (x / 10.0))
    else:
        df_oxy = pd.DataFrame({'year': df_temp['year'], 'OSI': 0.5})

    # Merge everything
    df_yearly = df_temp[['year', 'TSI']].merge(df_hci[['year', 'HCI']], on='year', how='outer')
    df_yearly = df_yearly.merge(df_oxy[['year', 'OSI']], on='year', how='outer').sort_values('year')
    
    df_yearly['TSI'] = df_yearly['TSI'].fillna(0.5)
    df_yearly['HCI'] = df_yearly['HCI'].fillna(0.0)
    df_yearly['OSI'] = df_yearly['OSI'].fillna(0.5)

    df_yearly['RSI'] = (w_tsi * df_yearly['TSI'] + w_hci * df_yearly['HCI'] + w_osi * df_yearly['OSI']).clip(0.0, 1.0)

    if start: df_yearly = df_yearly[df_yearly['year'] >= int(start)]
    if end: df_yearly = df_yearly[df_yearly['year'] <= int(end)]

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
    forecast_years = int(request.args.get('forecast_years', 5))
    rsi_data = reef_stress().get_json()
    
    if not rsi_data.get('years'):
        return jsonify({'historical': rsi_data, 'forecast': {'years': [], 'RSI': [], 'TSI': [], 'OSI': []}})

    last_year = int(rsi_data['years'][-1])
    future_years = list(range(last_year + 1, last_year + forecast_years + 1))

    return jsonify({
        'historical': rsi_data,
        'forecast': {
            'years': future_years,
            'RSI': [round(x, 3) for x in make_smooth_forecast(rsi_data['RSI'], forecast_years)],
            'TSI': [round(x, 3) for x in make_smooth_forecast(rsi_data['TSI'], forecast_years)],
            'OSI': [round(x, 3) for x in make_smooth_forecast(rsi_data['OSI'], forecast_years)]
        }
    })