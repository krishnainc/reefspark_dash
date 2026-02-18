"""
Litter API routes - beach litter data endpoints
"""
from flask import Blueprint, request, jsonify
import pandas as pd
from utils import load_litter_df

litters_api = Blueprint('litters_api', __name__, url_prefix='/api/litters')


@litters_api.route('/trends')
def litters_trends():
    """Return yearly totals and averages across all beaches.
    Output: { years: [...], total_abund: [...], avg_abund_per_survey: [...] }
    """
    df = load_litter_df()
    
    # optional filters
    country = request.args.get('country')
    search = request.args.get('search')
    if country:
        df = df[df['country'].astype(str).str.contains(country, case=False, na=False)]
    if search:
        mask = (
            df.get('beachname', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False) |
            df.get('beachcode', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]
    
    # collect pairs like 2012_abund / 2012_nbsur ... 2023
    years = []
    totals = []
    avgs = []
    for year in range(2012, 2024):
        abund_col = f"{year}_abund"
        nbsur_col = f"{year}_nbsur"
        if abund_col in df.columns and nbsur_col in df.columns:
            total_abund = pd.to_numeric(df[abund_col], errors='coerce').sum(min_count=1)
            total_survey = pd.to_numeric(df[nbsur_col], errors='coerce').sum(min_count=1)
            avg_per_survey = (total_abund / total_survey) if total_survey and total_survey != 0 else None
            years.append(year)
            totals.append(round(float(total_abund), 2) if pd.notna(total_abund) else None)
            avgs.append(round(float(avg_per_survey), 3) if avg_per_survey is not None and pd.notna(avg_per_survey) else None)
    
    return jsonify({"years": years, "total_abund": totals, "avg_abund_per_survey": avgs})


@litters_api.route('/predictions')
def litters_predictions():
    """Predict next 5 years litter for each beach using slope/intercept columns.
    Query: country, search, sort, order, page, pageSize
    Output: rows with beach info and predicted litter for 2024–2028
    """
    df = load_litter_df()
    country = request.args.get('country')
    search = request.args.get('search')
    sort = request.args.get('sort', 'pred2028')
    order = request.args.get('order', 'desc')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('pageSize', 10))

    # filter
    if country:
        df = df[df['country'].astype(str).str.contains(country, case=False, na=False)]
    if search:
        mask = (
            df.get('beachname', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False) |
            df.get('beachcode', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]

    # predict for 2024–2028 using y = slope * year + intercept
    # If slope/intercept missing, fallback to last available year
    years = list(range(2024, 2029))
    preds = []
    for _, r in df.iterrows():
        slope = r.get('litter_slope')
        intercept = r.get('litter_intercept')
        try:
            slope = float(slope)
            intercept = float(intercept)
        except (TypeError, ValueError):
            slope = None
            intercept = None
        pred_row = {
            'country': str(r.get('country', '')),
            'beachname': str(r.get('beachname', '')),
            'beachcode': str(r.get('beachcode', '')),
            'slope': slope,
            'intercept': intercept,
            'preds': []
        }
        for year in years:
            if slope is not None and intercept is not None:
                pred = slope * year + intercept
            else:
                # fallback: use last available year
                last_year = 2023
                pred = r.get(f'{last_year}_abund', None)
                try:
                    pred = float(pred)
                except (TypeError, ValueError):
                    pred = None
            pred_row['preds'].append(round(pred, 2) if pred is not None else None)
        # for sorting
        pred_row['pred2028'] = pred_row['preds'][-1] if pred_row['preds'] else None
        preds.append(pred_row)

    # sort
    sort_map = {
        'pred2028': 'pred2028',
        'name': 'beachname',
        'code': 'beachcode',
        'country': 'country'
    }
    sort_col = sort_map.get(sort, 'pred2028')
    preds = sorted(preds, key=lambda x: (x.get(sort_col) is None, x.get(sort_col)), reverse=(order=='desc'))

    # paginate
    total_rows = len(preds)
    start = (page - 1) * page_size
    end = start + page_size
    page_preds = preds[start:end]

    return jsonify({
        'total': int(total_rows),
        'page': page,
        'pageSize': page_size,
        'years': years,
        'rows': page_preds
    })


@litters_api.route('/top_beaches')
def litters_top_beaches():
    """Top N beaches by totalLitter (default 10)."""
    N = int(request.args.get('n', 10))
    country = request.args.get('country')
    search = request.args.get('search')
    df = load_litter_df()
    if country:
        df = df[df['country'].astype(str).str.contains(country, case=False, na=False)]
    if search:
        mask = (
            df.get('beachname', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False) |
            df.get('beachcode', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]
    if 'totalLitter' in df.columns:
        df['totalLitter'] = pd.to_numeric(df['totalLitter'], errors='coerce')
        top = df.sort_values('totalLitter', ascending=False).head(N)
        return jsonify({
            "labels": top.get('beachname', top.get('beachcode', top.index)).astype(str).tolist(),
            "values": top['totalLitter'].round(2).fillna(0).tolist()
        })
    # fallback: compute total across yearly columns
    yearly_cols = [c for c in df.columns if c.endswith('_abund')]
    df['computed_total'] = df[yearly_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
    top = df.sort_values('computed_total', ascending=False).head(N)
    return jsonify({
        "labels": top.get('beachname', top.get('beachcode', top.index)).astype(str).tolist(),
        "values": top['computed_total'].round(2).fillna(0).tolist()
    })


@litters_api.route('/countries')
def litters_countries():
    """Get list of all countries in dataset."""
    df = load_litter_df()
    countries = sorted(set(df.get('country', pd.Series(dtype=str)).dropna().astype(str).unique().tolist()))
    return jsonify({"countries": countries})


@litters_api.route('/rows')
def litters_rows():
    """Return paged/sorted rows with computed totals for UI table.
    Query: country, search, sort (total|avg|name|code), order (asc|desc), page, pageSize
    """
    df = load_litter_df()
    country = request.args.get('country')
    search = request.args.get('search')
    sort = request.args.get('sort', 'total')
    order = request.args.get('order', 'desc')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('pageSize', 20))

    # filter
    if country:
        df = df[df['country'].astype(str).str.contains(country, case=False, na=False)]
    if search:
        mask = (
            df.get('beachname', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False) |
            df.get('beachcode', pd.Series(index=df.index, dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]

    # totals/averages
    yearly_abund = [c for c in df.columns if c.endswith('_abund')]
    yearly_nbsur = [c for c in df.columns if c.endswith('_nbsur')]
    df['total_abund'] = df[yearly_abund].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
    df['total_survey'] = df[yearly_nbsur].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
    df['avg_per_survey'] = df.apply(lambda r: (r['total_abund'] / r['total_survey']) if pd.notna(r['total_abund']) and pd.notna(r['total_survey']) and r['total_survey'] != 0 else None, axis=1)

    # sort
    sort_map = {
        'total': 'total_abund',
        'avg': 'avg_per_survey',
        'name': 'beachname',
        'code': 'beachcode'
    }
    sort_col = sort_map.get(sort, 'total_abund')
    df = df.sort_values(sort_col, ascending=(order == 'asc'), na_position='last')

    # paginate
    total_rows = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    rows = []
    for _, r in page_df.iterrows():
        rows.append({
            'country': str(r.get('country', '')),
            'beachname': str(r.get('beachname', '')),
            'beachcode': str(r.get('beachcode', '')),
            'total_abund': None if pd.isna(r.get('total_abund')) else float(r.get('total_abund')),
            'total_survey': None if pd.isna(r.get('total_survey')) else float(r.get('total_survey')),
            'avg_per_survey': None if pd.isna(r.get('avg_per_survey')) else float(r.get('avg_per_survey'))
        })

    return jsonify({
        'total': int(total_rows),
        'page': page,
        'pageSize': page_size,
        'rows': rows
    })
