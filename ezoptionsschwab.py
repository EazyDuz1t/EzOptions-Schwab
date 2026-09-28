from flask import Flask, render_template_string, jsonify, request, Response, stream_with_context
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from bisect import bisect_left
from datetime import datetime, timedelta
import math
import time
import schwabdev
import os
from dotenv import load_dotenv
import pytz
import sqlite3
from contextlib import closing
from scipy.stats import norm
import json
import threading
import queue


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
    <defs>
        <linearGradient id="gold-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#fff1a8"/>
            <stop offset="45%" stop-color="#f4c542"/>
            <stop offset="100%" stop-color="#b77911"/>
        </linearGradient>
    </defs>
    <rect x="2" y="2" width="60" height="60" rx="15" fill="#111111" stroke="#f4c542" stroke-width="2.5"/>
    <text x="32" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" font-style="italic" font-weight="700" fill="url(#gold-gradient)">Ez</text>
</svg>
"""

MAX_RETAINED_SESSION_DATES = 2
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIONS_DB_PATH = os.path.join(_APP_DIR, 'options_data.db')
SETTINGS_PATH = os.path.join(_APP_DIR, 'settings.json')
_DB_BUSY_TIMEOUT_SEC = 30
_db_write_lock = threading.RLock()


def get_options_db_connection():
    """Open options_data.db with WAL mode and busy timeout for concurrent access."""
    conn = sqlite3.connect(OPTIONS_DB_PATH, timeout=_DB_BUSY_TIMEOUT_SEC)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn
# Global error handlers for Flask
@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/') or request.path.startswith('/update') or request.path.startswith('/expirations'):
        return jsonify({'error': 'API endpoint not found'}), 404
    return "404 - Not Found", 404

@app.errorhandler(500)
def internal_error(error):
    # Expose the error message for API-like endpoints so the frontend can show details
    msg = getattr(error, 'description', None) or str(error)
    if request.path.startswith('/api/') or request.path.startswith('/update') or request.path.startswith('/expirations'):
        return jsonify({'error': msg}), 500
    return "500 - Internal Server Error", 500

# Initialize SQLite database
def init_db():
    with _db_write_lock, closing(get_options_db_connection()) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interval_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    price REAL NOT NULL,
                    strike REAL NOT NULL,
                    net_gamma REAL NOT NULL,
                    net_delta REAL NOT NULL,
                    net_vanna REAL NOT NULL,
                    net_charm REAL,
                    net_volume REAL,
                    net_speed REAL,
                    net_vomma REAL,
                    net_color REAL,
                    abs_gex_total REAL,
                    expiry_key TEXT NOT NULL DEFAULT '',
                    date TEXT NOT NULL
                )
            ''')
            # Try to add net_charm column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE interval_data ADD COLUMN net_charm REAL')
            except sqlite3.OperationalError:
                pass 
            try:
                cursor.execute('ALTER TABLE interval_data ADD COLUMN net_volume REAL')
            except sqlite3.OperationalError:
                pass
            # Add abs_gex_total column if it's missing
            try:
                cursor.execute('ALTER TABLE interval_data ADD COLUMN abs_gex_total REAL')
            except sqlite3.OperationalError:
                pass 
            try:
                cursor.execute('ALTER TABLE interval_data ADD COLUMN net_speed REAL')
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute('ALTER TABLE interval_data ADD COLUMN net_vomma REAL')
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute('ALTER TABLE interval_data ADD COLUMN net_color REAL')
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE interval_data ADD COLUMN expiry_key TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interval_session_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    price REAL NOT NULL,
                    expected_move REAL,
                    expected_move_upper REAL,
                    expected_move_lower REAL,
                    expiry_key TEXT NOT NULL DEFAULT '',
                    date TEXT NOT NULL
                )
            ''')
            try:
                cursor.execute("ALTER TABLE interval_session_data ADD COLUMN expiry_key TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            
            # Add centroid data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS centroid_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    price REAL NOT NULL,
                    call_centroid REAL NOT NULL,
                    put_centroid REAL NOT NULL,
                    call_volume INTEGER NOT NULL,
                    put_volume INTEGER NOT NULL,
                    expiry_key TEXT NOT NULL DEFAULT '',
                    date TEXT NOT NULL
                )
            ''')
            try:
                cursor.execute("ALTER TABLE centroid_data ADD COLUMN expiry_key TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass

            # Indices for fast lookups by (ticker, date) and (ticker, date, expiry_key)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interval_data_td ON interval_data(ticker, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interval_data_tde ON interval_data(ticker, date, expiry_key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interval_data_ts ON interval_data(ticker, date, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interval_session_td ON interval_session_data(ticker, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interval_session_tde ON interval_session_data(ticker, date, expiry_key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_centroid_td ON centroid_data(ticker, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_centroid_tde ON centroid_data(ticker, date, expiry_key)')
            conn.commit()

# Regular-session hours per ET date, from Schwab's market-hours endpoint so holidays
# (closed) and half-days (13:00 close) aren't recorded as normal sessions.
_session_hours_cache = {}  # 'YYYY-MM-DD' -> ((open_dt, close_dt) | None, fetched_at_unix_ts)
_SESSION_HOURS_RETRY_SEC = 600


def _default_session_bounds(day):
    """Weekday 9:30-16:00 ET fallback when the market-hours API is unavailable."""
    if day.weekday() >= 5:
        return None
    est = pytz.timezone('US/Eastern')
    base = datetime(day.year, day.month, day.day)
    return (est.localize(base.replace(hour=9, minute=30)), est.localize(base.replace(hour=16)))


def get_regular_session_bounds(day=None):
    """Return (open_dt, close_dt) in ET for the regular equity session on `day`
    (a date; defaults to today ET), or None if the market is closed that day."""
    est = pytz.timezone('US/Eastern')
    if day is None:
        day = datetime.now(est).date()
    key = day.strftime('%Y-%m-%d')
    cached = _session_hours_cache.get(key)
    if cached and (cached[1] is None or time.time() - cached[1] < _SESSION_HOURS_RETRY_SEC):
        return cached[0]

    if day.weekday() >= 5:
        _session_hours_cache[key] = (None, None)
        return None

    api_client = globals().get('client')
    if api_client is None:
        return _default_session_bounds(day)
    try:
        response = api_client.market_hours(['equity'], key)
        if not response.ok:
            raise Exception(f"{response.status_code} {response.reason}")
        markets = (response.json() or {}).get('equity', {}) or {}
        bounds = None
        for info in markets.values():
            if not isinstance(info, dict):
                continue
            if not info.get('isOpen'):
                continue
            regular = (info.get('sessionHours') or {}).get('regularMarket') or []
            if regular:
                start = datetime.fromisoformat(regular[0]['start']).astimezone(est)
                end = datetime.fromisoformat(regular[-1]['end']).astimezone(est)
                bounds = (start, end)
                break
        # A successful answer is final for the day (None = holiday)
        _session_hours_cache[key] = (bounds, None)
        return bounds
    except Exception as e:
        print(f"Market hours lookup failed for {key}, assuming a regular session: {e}")
        bounds = _default_session_bounds(day)
        _session_hours_cache[key] = (bounds, time.time())
        return bounds


def is_market_hours():
    """Return True if the current time is within today's regular session (holiday/half-day aware)."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    bounds = get_regular_session_bounds(now.date())
    if not bounds:
        return False
    return bounds[0] <= now <= bounds[1]


INTERVAL_LEVEL_DISPLAY_NAMES = {
    'GEX': 'GEX',
    'AbsGEX': 'Abs GEX',
    'DEX': 'DEX',
    'VEX': 'Vanna',
    'Charm': 'Charm',
    'Volume': 'Volume',
    'Speed': 'Speed',
    'Vomma': 'Vomma',
    'Color': 'Color',
    'Expected Move': 'Expected Move',
}

INTERVAL_LEVEL_VALUE_KEYS = {
    'GEX': 'net_gamma',
    'AbsGEX': 'abs_gex_total',
    'DEX': 'net_delta',
    'VEX': 'net_vanna',
    'Charm': 'net_charm',
    'Volume': 'net_volume',
    'Speed': 'net_speed',
    'Vomma': 'net_vomma',
    'Color': 'net_color',
}


def normalize_level_type(level_type):
    if level_type in ('Vanna', 'VEX'):
        return 'VEX'
    return level_type


def resolve_level_value_column(level_type):
    normalized_type = normalize_level_type(level_type)
    if normalized_type == 'AbsGEX':
        return 'GEX'
    if normalized_type == 'Volume':
        return 'volume'
    return normalized_type


def combine_level_values(level_type, call_value, put_value):
    normalized_type = normalize_level_type(level_type)
    if normalized_type == 'GEX':
        return call_value - put_value
    if normalized_type == 'AbsGEX':
        return abs(call_value) + abs(put_value)
    if normalized_type == 'Volume':
        return call_value - put_value
    return call_value + put_value


HEATMAP_TITLE_DISPLAY_NAMES = {
    'GEX': 'Gamma Exposure',
    'AbsGEX': 'Absolute Gamma Exposure',
    'DEX': 'Delta Exposure',
    'VEX': 'Vanna Exposure',
    'Charm': 'Charm Exposure',
    'Volume': 'Volume Exposure',
    'Speed': 'Speed Exposure',
    'Vomma': 'Vomma Exposure',
    'Color': 'Color Exposure',
}


def build_expiry_selection_key(expiry_dates):
    if not expiry_dates:
        return ''

    if isinstance(expiry_dates, str):
        normalized_dates = [expiry_dates]
    else:
        normalized_dates = [str(expiry_date) for expiry_date in expiry_dates if expiry_date]

    return '|'.join(sorted(set(normalized_dates)))


def calculate_expected_move_snapshot(calls, puts, spot_price):
    """Return the current ATM straddle-based expected move snapshot."""
    if calls is None or puts is None or calls.empty or puts.empty or not spot_price:
        return None

    strikes_sorted = sorted(calls['strike'].unique())
    if not strikes_sorted:
        return None

    atm_strike = min(strikes_sorted, key=lambda strike: abs(strike - spot_price))

    def _get_mid(df, strike):
        row = df.loc[df['strike'] == strike]
        if row is None or row.empty:
            return None
        bid = row['bid'].values[0]
        ask = row['ask'].values[0]
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        if bid > 0:
            return bid
        if ask > 0:
            return ask
        return None

    call_mid = _get_mid(calls, atm_strike)
    put_mid = _get_mid(puts, atm_strike)
    expected_move = (call_mid or 0) + (put_mid or 0)
    if expected_move <= 0:
        return None

    return {
        'atm_strike': atm_strike,
        'move': float(expected_move),
        'upper': float(spot_price + expected_move),
        'lower': float(spot_price - expected_move),
    }

# Function to store centroid data
def store_centroid_data(ticker, price, calls, puts, expiry_key=''):
    """Store call and put centroid data for 5-minute intervals during market hours only"""
    # Get current time in Eastern Time
    est = pytz.timezone('US/Eastern')
    current_time_est = datetime.now(est)
    
    # Only during today's regular session (skips holidays and half-day afternoons)
    if not is_market_hours():
        return
    
    current_time = int(current_time_est.timestamp())
    current_date = current_time_est.strftime('%Y-%m-%d')
    
    # Round to nearest 5-minute interval (300 seconds)
    interval_timestamp = (current_time // 300) * 300
    
    # Calculate centroids (volume-weighted average strike prices)
    call_centroid = 0
    put_centroid = 0
    call_volume = 0
    put_volume = 0
    
    if not calls.empty:
        # Filter out zero volume options
        calls_with_volume = calls[calls['volume'] > 0]
        if not calls_with_volume.empty:
            call_volume = int(calls_with_volume['volume'].sum())
            # Calculate weighted average strike price
            weighted_strikes = calls_with_volume['strike'] * calls_with_volume['volume']
            call_centroid = weighted_strikes.sum() / call_volume
    
    if not puts.empty:
        # Filter out zero volume options
        puts_with_volume = puts[puts['volume'] > 0]
        if not puts_with_volume.empty:
            put_volume = int(puts_with_volume['volume'].sum())
            # Calculate weighted average strike price
            weighted_strikes = puts_with_volume['strike'] * puts_with_volume['volume']
            put_centroid = weighted_strikes.sum() / put_volume
    
    with _db_write_lock:
        with closing(get_options_db_connection()) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute('''
                    DELETE FROM centroid_data
                    WHERE ticker = ? AND timestamp = ? AND expiry_key = ? AND date = ?
                ''', (ticker, interval_timestamp, expiry_key, current_date))
                if call_volume > 0 or put_volume > 0:
                    cursor.execute('''
                        INSERT INTO centroid_data (
                            ticker, timestamp, price, call_centroid, put_centroid,
                            call_volume, put_volume, expiry_key, date
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker,
                        interval_timestamp,
                        price,
                        call_centroid,
                        put_centroid,
                        call_volume,
                        put_volume,
                        expiry_key,
                        current_date,
                    ))
                conn.commit()

def _market_hours_bounds(date_str):
    """Return (open_ts, close_ts) Unix timestamps for 9:30–16:00 ET on the given date string."""
    est = pytz.timezone('US/Eastern')
    day = datetime.strptime(date_str, '%Y-%m-%d')
    open_dt = est.localize(day.replace(hour=9, minute=30, second=0, microsecond=0))
    close_dt = est.localize(day.replace(hour=16, minute=0, second=0, microsecond=0))
    return int(open_dt.timestamp()), int(close_dt.timestamp())


# Function to get centroid data
def get_centroid_data(ticker, date=None, expiry_key=None):
    """Get centroid data for current trading session only (market hours)"""
    if date is None:
        date = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')

    open_ts, close_ts = _market_hours_bounds(date)

    with closing(get_options_db_connection()) as conn:
        with closing(conn.cursor()) as cursor:
            if expiry_key is None:
                cursor.execute('''
                    SELECT timestamp, price, call_centroid, put_centroid, call_volume, put_volume
                    FROM centroid_data
                    WHERE ticker = ? AND date = ?
                      AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                ''', (ticker, date, open_ts, close_ts))
            else:
                cursor.execute('''
                    SELECT timestamp, price, call_centroid, put_centroid, call_volume, put_volume
                    FROM centroid_data
                    WHERE ticker = ? AND date = ? AND expiry_key = ?
                      AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                ''', (ticker, date, expiry_key, open_ts, close_ts))
            return cursor.fetchall()

# Function to store interval data
def store_interval_data(ticker, price, strike_range, calls, puts, expiry_key=''):
    if not is_market_hours():
        return
    est = pytz.timezone('US/Eastern')
    current_time_est = datetime.now(est)
    current_time = int(current_time_est.timestamp())
    current_date = current_time_est.strftime('%Y-%m-%d')
    
    # Store interval overlays at 1-minute resolution so they can be aggregated
    # to whatever candle timeframe the chart is using.
    interval_timestamp = (current_time // 60) * 60
    # Calculate strike range boundaries
    min_strike = price * (1 - strike_range)
    max_strike = price * (1 + strike_range)
    
    # Filter options within strike range
    range_calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    range_puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

    # Build per-strike net exposures using vectorized groupby instead of iterrows
    _agg_src = ['strike', 'GEX', 'DEX', 'VEX', 'Charm', 'volume', 'Speed', 'Vomma', 'Color']
    _sum_cols = ['call_gamma','put_gamma','net_gamma','net_delta','net_vanna',
                 'net_charm','net_volume','net_speed','net_vomma','net_color']

    if not range_calls.empty:
        c = range_calls[_agg_src].copy()
        c['call_gamma'] = c['GEX']
        c['put_gamma'] = 0.0
        c['net_gamma'] = c['GEX']
        c['net_delta'] = c['DEX']
        c['net_vanna'] = c['VEX']
        c['net_charm'] = c['Charm']
        c['net_volume'] = c['volume']
        c['net_speed'] = c['Speed']
        c['net_vomma'] = c['Vomma']
        c['net_color'] = c['Color']
    else:
        c = pd.DataFrame(columns=['strike'] + _sum_cols)

    if not range_puts.empty:
        p = range_puts[_agg_src].copy()
        p['call_gamma'] = 0.0
        p['put_gamma'] = p['GEX']
        p['net_gamma'] = -p['GEX']
        p['net_delta'] = p['DEX']
        p['net_vanna'] = p['VEX']
        p['net_charm'] = p['Charm']
        p['net_volume'] = -p['volume']
        p['net_speed'] = p['Speed']
        p['net_vomma'] = p['Vomma']
        p['net_color'] = p['Color']
    else:
        p = pd.DataFrame(columns=['strike'] + _sum_cols)

    combined_exp = pd.concat([c[['strike']+_sum_cols], p[['strike']+_sum_cols]], ignore_index=True)
    grouped = combined_exp.groupby('strike', sort=False)[_sum_cols].sum()
    grouped['abs_gex_total'] = grouped['call_gamma'].abs() + grouped['put_gamma'].abs()

    expected_move_snapshot = calculate_expected_move_snapshot(calls, puts, price)

    n = len(grouped)
    strikes_list = grouped.index.tolist()
    rows_to_insert = list(zip(
        [ticker] * n,
        [interval_timestamp] * n,
        [price] * n,
        strikes_list,
        grouped['net_gamma'].tolist(),
        grouped['net_delta'].tolist(),
        grouped['net_vanna'].tolist(),
        grouped['net_charm'].tolist(),
        grouped['net_volume'].tolist(),
        grouped['net_speed'].tolist(),
        grouped['net_vomma'].tolist(),
        grouped['net_color'].tolist(),
        grouped['abs_gex_total'].tolist(),
        [expiry_key] * n,
        [current_date] * n,
    ))

    with _db_write_lock:
        with closing(get_options_db_connection()) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute('''
                    DELETE FROM interval_data
                    WHERE ticker = ? AND timestamp = ? AND expiry_key = ? AND date = ?
                ''', (ticker, interval_timestamp, expiry_key, current_date))
                cursor.execute('''
                    DELETE FROM interval_session_data
                    WHERE ticker = ? AND timestamp = ? AND expiry_key = ? AND date = ?
                ''', (ticker, interval_timestamp, expiry_key, current_date))

                cursor.executemany('''
                    INSERT INTO interval_data (
                        ticker, timestamp, price, strike, net_gamma, net_delta, net_vanna,
                        net_charm, net_volume, net_speed, net_vomma, net_color,
                        abs_gex_total, expiry_key, date
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', rows_to_insert)

                if expected_move_snapshot:
                    cursor.execute('''
                        INSERT INTO interval_session_data (
                            ticker, timestamp, price, expected_move, expected_move_upper,
                            expected_move_lower, expiry_key, date
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker,
                        interval_timestamp,
                        price,
                        expected_move_snapshot['move'],
                        expected_move_snapshot['upper'],
                        expected_move_snapshot['lower'],
                        expiry_key,
                        current_date,
                    ))
                conn.commit()

# Function to get interval data
def get_interval_data(ticker, date=None, expiry_key=None):
    if date is None:
        date = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')

    open_ts, close_ts = _market_hours_bounds(date)

    with closing(get_options_db_connection()) as conn:
        with closing(conn.cursor()) as cursor:
            if expiry_key is None:
                cursor.execute('''
                    SELECT timestamp, price, strike, net_gamma, net_delta, net_vanna, net_charm,
                        abs_gex_total, net_volume, net_speed, net_vomma, net_color
                    FROM interval_data
                    WHERE ticker = ? AND date = ?
                      AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp, strike
                ''', (ticker, date, open_ts, close_ts))
            else:
                cursor.execute('''
                    SELECT timestamp, price, strike, net_gamma, net_delta, net_vanna, net_charm,
                        abs_gex_total, net_volume, net_speed, net_vomma, net_color
                    FROM interval_data
                    WHERE ticker = ? AND date = ? AND expiry_key = ?
                      AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp, strike
                ''', (ticker, date, expiry_key, open_ts, close_ts))
            return cursor.fetchall()


def get_interval_session_data(ticker, date=None, expiry_key=None):
    if date is None:
        date = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')

    open_ts, close_ts = _market_hours_bounds(date)

    with closing(get_options_db_connection()) as conn:
        with closing(conn.cursor()) as cursor:
            if expiry_key is None:
                cursor.execute('''
                    SELECT timestamp, price, expected_move, expected_move_upper, expected_move_lower
                    FROM interval_session_data
                    WHERE ticker = ? AND date = ?
                      AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                ''', (ticker, date, open_ts, close_ts))
            else:
                cursor.execute('''
                    SELECT timestamp, price, expected_move, expected_move_upper, expected_move_lower
                    FROM interval_session_data
                    WHERE ticker = ? AND date = ? AND expiry_key = ?
                      AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                ''', (ticker, date, expiry_key, open_ts, close_ts))
            return cursor.fetchall()

def get_last_session_date(ticker, table='interval_data', expiry_key=None):
    """Return the most recent date that has data for ticker, or None."""
    with closing(get_options_db_connection()) as conn:
        with closing(conn.cursor()) as cursor:
            if expiry_key is None:
                cursor.execute(f'SELECT MAX(date) FROM {table} WHERE ticker = ?', (ticker,))
            else:
                cursor.execute(
                    f'SELECT MAX(date) FROM {table} WHERE ticker = ? AND expiry_key = ?',
                    (ticker, expiry_key),
                )
            row = cursor.fetchone()
            return row[0] if row and row[0] else None

def get_latest_session_expiry_key(ticker, table):
    """Return (date, expiry_key) of the most-sampled selection on the latest date with data."""
    with closing(get_options_db_connection()) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(f'''
                SELECT date, expiry_key
                FROM {table}
                WHERE ticker = ? AND date = (SELECT MAX(date) FROM {table} WHERE ticker = ?)
                GROUP BY expiry_key
                ORDER BY COUNT(*) DESC
                LIMIT 1
            ''', (ticker, ticker))
            row = cursor.fetchone()
            return (row[0], row[1]) if row else None


# Function to clear old data
def clear_old_data():
    """Keep only the most recent session dates in each SQLite history table."""
    tables = ('interval_data', 'centroid_data', 'interval_session_data')
    deleted_rows = {}

    with _db_write_lock:
        with closing(get_options_db_connection()) as conn:
            with closing(conn.cursor()) as cursor:
                for table_name in tables:
                    cursor.execute(f'''
                        SELECT DISTINCT date
                        FROM {table_name}
                        WHERE date IS NOT NULL
                        ORDER BY date DESC
                    ''')
                    all_dates = [row[0] for row in cursor.fetchall() if row[0]]
                    stale_dates = all_dates[MAX_RETAINED_SESSION_DATES:]
                    if not stale_dates:
                        continue

                    placeholders = ','.join('?' for _ in stale_dates)
                    cursor.execute(
                        f'DELETE FROM {table_name} WHERE date IN ({placeholders})',
                        stale_dates,
                    )
                    deleted_rows[table_name] = cursor.rowcount

                conn.commit()

    if deleted_rows:
        print(
            'Pruned database history to the latest '
            f'{MAX_RETAINED_SESSION_DATES} session dates: {deleted_rows}'
        )


# Initialize database
init_db()

# Prune retained history on startup as well as on active writes.
clear_old_data()
_last_prune_date = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')


# Cache for last fetched options data per ticker — used by /update_price
# so the price chart can refresh independently without re-fetching the full chain.
_options_cache = {}  # (ticker, expiry_key) -> {'calls': DataFrame, 'puts': DataFrame, 'S': float}

# Spot price populated by fetch_options_for_date so /update can skip a redundant quote call.
_chain_spot_cache = {}  # ticker -> (price, fetched_at_unix_ts)

# Short-lived cache of full quote payloads for the /update price-info panel. /update is
# polled every second; re-quoting each time adds ~60 Schwab calls/minute on top of the
# option-chain calls, which pushes against the API rate limit.
_QUOTE_CACHE_TTL_SEC = 5
_quote_cache = {}  # symbol -> (quote_json, fetched_at_unix_ts)

# Initialize Schwab client
try:
    client = schwabdev.Client(
        os.getenv('SCHWAB_APP_KEY'),
        os.getenv('SCHWAB_APP_SECRET'),
        os.getenv('SCHWAB_CALLBACK_URL')
    )
except Exception as e:
    print(f"Error initializing Schwab client: {e}")
    client = None

# ── Real-time Price Streamer ─────────────────────────────────────────────────
class PriceStreamer:
    """Manages a single schwabdev streaming websocket for real-time price data.
    Feeds per-ticker queues that are consumed by the /price_stream SSE endpoint.
    """
    def __init__(self):
        self._stream = None
        self._lock = threading.Lock()
        self._queues = {}       # ticker (upper) -> list[queue.Queue]
        self._subscribed = set()  # tickers with active stream subscriptions
        self._started = False

    def _handler(self, message):
        """Parse raw schwabdev stream message and push candle/quote data to queues."""
        try:
            data = json.loads(message) if isinstance(message, str) else message
            # Schwab wraps every data message in {"data": [...]}; unwrap it so
            # we can iterate over individual service messages directly.
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
                else:
                    return  # login/response/notify envelope – nothing to forward
            if not isinstance(data, list):
                data = [data]
            for msg in data:
                service = msg.get('service', '')
                if service == 'CHART_EQUITY':
                    for item in msg.get('content', []):
                        ticker = item.get('key', '').upper()
                        chart_time_ms = item.get('7')
                        if not ticker or chart_time_ms is None:
                            continue
                        # Live CHART_EQUITY messages are key, sequence, O, H, L, C, V, time, day
                        # (the order in Schwab's docs is wrong; schwabdev's translate.py uses
                        # the corrected order). Fall back to the documented order if the
                        # corrected one doesn't form a valid bar.
                        ohlcv = [item.get(k) for k in ('2', '3', '4', '5', '6')]
                        if not self._is_valid_bar(ohlcv):
                            ohlcv = [item.get(k) for k in ('1', '2', '3', '4', '5')]
                            if not self._is_valid_bar(ohlcv):
                                continue
                        self._push(ticker, self._candle_payload(chart_time_ms, ohlcv))
                elif service == 'CHART_FUTURES':
                    for item in msg.get('content', []):
                        ticker = item.get('key', '').upper()
                        # CHART_FUTURES fields: key, time, O, H, L, C, V
                        chart_time_ms = item.get('1')
                        if not ticker or chart_time_ms is None:
                            continue
                        ohlcv = [item.get(k) for k in ('2', '3', '4', '5', '6')]
                        if not self._is_valid_bar(ohlcv):
                            continue
                        self._push(ticker, self._candle_payload(chart_time_ms, ohlcv))
                elif service == 'LEVELONE_EQUITIES':
                    for item in msg.get('content', []):
                        ticker = item.get('key', '').upper()
                        last = item.get('3')  # field 3 = last price
                        if not ticker or last is None:
                            continue
                        payload = json.dumps({'type': 'quote', 'last': float(last)})
                        self._push(ticker, payload)
                elif service == 'LEVELONE_FUTURES':
                    for item in msg.get('content', []):
                        ticker = item.get('key', '').upper()
                        last = item.get('3')  # field 3 = last price
                        if not ticker or last is None:
                            continue
                        payload = json.dumps({'type': 'quote', 'last': float(last)})
                        self._push(ticker, payload)
        except Exception as e:
            print(f"[PriceStreamer] handler error: {e}")

    @staticmethod
    def _is_valid_bar(ohlcv):
        try:
            o, h, l, c = (float(v) for v in ohlcv[:4])
        except (TypeError, ValueError):
            return False
        return l <= min(o, c) and max(o, c) <= h

    @staticmethod
    def _candle_payload(chart_time_ms, ohlcv):
        o, h, l, c, v = ohlcv
        return json.dumps({
            'type': 'candle',
            'time': int(chart_time_ms) // 1000,
            'open': float(o),
            'high': float(h),
            'low': float(l),
            'close': float(c),
            'volume': float(v or 0),
        })

    def _push(self, ticker, payload):
        with self._lock:
            qs = list(self._queues.get(ticker, []))
        for q in qs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass  # drop stale data rather than block

    def _ensure_started(self):
        with self._lock:
            if self._started or client is None:
                return
            try:
                self._stream = schwabdev.Stream(client)
                self._stream.start(self._handler)
                self._started = True
                print("[PriceStreamer] Stream started.")
            except Exception as e:
                print(f"[PriceStreamer] Failed to start stream: {e}")

    def subscribe(self, ticker, q):
        """Register a client SSE queue and ensure ticker is subscribed on the stream."""
        self._ensure_started()
        with self._lock:
            if ticker not in self._queues:
                self._queues[ticker] = []
            self._queues[ticker].append(q)
            needs_sub = ticker not in self._subscribed
        if needs_sub and self._started and self._stream:
            try:
                is_future = ticker.startswith('/')
                if is_future:
                    self._stream.send(self._stream.chart_futures(ticker, "0,1,2,3,4,5,6,7,8"))
                    self._stream.send(self._stream.level_one_futures(ticker, "0,1,2,3"))
                else:
                    self._stream.send(self._stream.chart_equity(ticker, "0,1,2,3,4,5,6,7,8"))
                    self._stream.send(self._stream.level_one_equities(ticker, "0,1,2,3"))
                # Only mark as subscribed once the request was sent, so a failed start
                # or send is retried on the next connection instead of sticking forever.
                with self._lock:
                    self._subscribed.add(ticker)
                print(f"[PriceStreamer] Subscribed to {ticker}")
            except Exception as e:
                print(f"[PriceStreamer] Subscribe error for {ticker}: {e}")

    def unsubscribe_queue(self, ticker, q):
        """Remove a specific client queue (called on SSE disconnect)."""
        with self._lock:
            if ticker in self._queues:
                try:
                    self._queues[ticker].remove(q)
                except ValueError:
                    pass

    def stop(self):
        with self._lock:
            if self._stream and self._started:
                try:
                    self._stream.stop()
                except Exception:
                    pass
                self._stream = None
                self._started = False


price_streamer = PriceStreamer()

# Helper Functions
def format_ticker(ticker):
    if not ticker:
        return ""
    ticker = ticker.upper()
    if ticker.startswith('/'):
        return ticker
    elif ticker in ['SPX', '$SPX']:
        return '$SPX'  # Return $SPX for API calls
    elif ticker in ['NDX', '$NDX']:
        return '$NDX'  # Return $NDX for API calls
    elif ticker in ['VIX', '$VIX']:
        return '$VIX'  # Return $VIX for API calls
    return ticker

def format_display_ticker(ticker):
    """Helper function to format tickers for display and data filtering"""
    if not ticker:
        return []
    ticker = ticker.upper()
    if ticker.startswith('/'):
        return [ticker]
    elif ticker in ['$SPX', 'SPX']:
        # For SPX, return SPXW for options symbols and $SPX for underlying
        return ['SPXW', '$SPX']
    elif ticker in ['$NDX', 'NDX']:
        # For NDX, return NDXP for options symbols and $NDX for underlying
        return ['NDXP', '$NDX']
    elif ticker in ['$VIX', 'VIX']:
        # For VIX, return VIX for options symbols and $VIX for underlying
        return ['VIX', '$VIX']
    elif ticker == 'MARKET2':
        return ['SPY']
    return [ticker]

def format_large_number(num):
    """Format large numbers with suffixes (K, M, B, T)"""
    if num is None:
        return "0"
    
    abs_num = abs(num)
    if abs_num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif abs_num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:,.0f}"


def build_hover_template(title, rows):
    hover_rows = [f"<b>{title}</b>"]
    hover_rows.extend(f"{label}: {value}" for label, value in rows)
    return '<br>'.join(hover_rows) + '<extra></extra>'


def build_time_hover_template(title, rows, time_expr='%{x|%H:%M}'):
    return build_hover_template(title, rows + [('Time', time_expr)])

def get_strike_interval(strikes):
    """Determine the most common strike interval from a list of strikes"""
    if len(strikes) < 2:
        return 1.0
    
    sorted_strikes = sorted(set(strikes))
    intervals = []
    for i in range(1, len(sorted_strikes)):
        diff = sorted_strikes[i] - sorted_strikes[i-1]
        if diff > 0:
            intervals.append(diff)
    
    if not intervals:
        return 1.0
    
    # Return the most common interval
    from collections import Counter
    interval_counts = Counter([round(i, 2) for i in intervals])
    return interval_counts.most_common(1)[0][0]

def round_to_strike(value, strike_interval):
    """Round a value to the nearest strike interval"""
    # Round half up; Python's round() uses banker's rounding, which sends half strikes
    # to alternating neighbours (212.5 -> 210 but 217.5 -> 220).
    return math.floor(value / strike_interval + 0.5) * strike_interval

def aggregate_by_strike(df, value_columns, strike_interval):
    """Aggregate dataframe by rounded strike prices"""
    if df.empty:
        return df
    
    df = df.copy()
    df['rounded_strike'] = df['strike'].apply(lambda x: round_to_strike(x, strike_interval))
    
    # Build aggregation dict for value columns
    agg_dict = {}
    for col in value_columns:
        if col in df.columns:
            agg_dict[col] = 'sum'
    
    if not agg_dict:
        return df
    
    # Group by rounded strike and aggregate
    aggregated = df.groupby('rounded_strike', as_index=False).agg(agg_dict)
    aggregated = aggregated.rename(columns={'rounded_strike': 'strike'})
    
    return aggregated

# CBOE VIX1D-style floor: persist near-term inputs when < 60 minutes remain (see CBOE VIX1D methodology).
T_EXPIRATION_FLOOR_MINUTES = 60
SECONDS_PER_YEAR = 365 * 24 * 3600

def calculate_time_to_expiration(expiry_date):
    """
    Calculate time to expiration in years using Eastern Time.

    Uses a 60-minute floor per CBOE VIX1D near-term handling so Black-Scholes
    Greeks do not blow up in the final hour. Returns 0.0 after expiration
    (4:00 PM ET on expiry day for standard equity/PM-settled options).

    expiry_date: datetime.date object or string 'YYYY-MM-DD'
    Returns: time in years (float), or 0.0 if already expired
    """
    try:
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        if isinstance(expiry_date, str):
            expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        elif isinstance(expiry_date, datetime):
            expiry_date = expiry_date.date()
            
        # Standard equity / PM-settled index options expire at 4:00 PM ET
        expiry_dt = datetime.combine(expiry_date, datetime.min.time()) + timedelta(hours=16)
        expiry_dt = et_tz.localize(expiry_dt)
        
        seconds_left = (expiry_dt - now_et).total_seconds()
        if seconds_left <= 0:
            return 0.0

        minutes_left = seconds_left / 60.0
        if minutes_left < T_EXPIRATION_FLOOR_MINUTES:
            minutes_left = T_EXPIRATION_FLOOR_MINUTES

        return (minutes_left * 60.0) / SECONDS_PER_YEAR
             
    except Exception as e:
        print(f"Error calculating time to expiration: {e}")
        return 0.0

# Raw option-chain responses are cached briefly so /update, /update_price and the
# background collector share one Schwab request instead of each making their own.
_CHAIN_CACHE_TTL_SEC = 1.5
_chain_cache = {}  # (ticker, from_date, to_date) -> (chain_json, fetched_at_unix_ts)
_chain_cache_lock = threading.Lock()


# Index underlyings have had several spellings on Schwab's option endpoints. Try them in
# order and remember the first one that returns a chain (failures aren't cached, so a
# transient error is retried on the next request).
_INDEX_OPTION_SYMBOL_CANDIDATES = {
    '$SPX': ['$SPX', 'SPX', '$SPX.X'],
    '$NDX': ['$NDX', 'NDX', '$NDX.X'],
    '$VIX': ['$VIX', 'VIX', '$VIX.X'],
}
_resolved_option_symbols = {}  # ticker -> symbol Schwab accepts for option endpoints


def resolve_option_symbol(ticker):
    """Return the symbol to send to Schwab's option endpoints for `ticker`."""
    resolved = _resolved_option_symbols.get(ticker)
    if resolved:
        return resolved
    candidates = _INDEX_OPTION_SYMBOL_CANDIDATES.get(ticker)
    if not candidates:
        return ticker
    attempts = []
    for symbol in candidates:
        try:
            response = client.option_chains(symbol=symbol, contractType='CALL', strikeCount=1)
            if response.ok and (response.json() or {}).get('callExpDateMap'):
                _resolved_option_symbols[ticker] = symbol
                if symbol != ticker:
                    print(f"Using option symbol {symbol!r} for {ticker}")
                return symbol
            attempts.append(f"{response.url} -> {response.status_code} {response.text[:150]}")
        except Exception as e:
            attempts.append(f"{symbol}: {e}")
    raise Exception(f"Schwab returned no option chain for any of {candidates}: " + ' | '.join(attempts))


def _fetch_chain_json(ticker, from_date, to_date):
    """Fetch (or reuse a very recent) raw Schwab option chain for [from_date, to_date]."""
    key = (ticker, from_date, to_date)
    with _chain_cache_lock:
        cached = _chain_cache.get(key)
        if cached and (time.time() - cached[1]) < _CHAIN_CACHE_TTL_SEC:
            return cached[0]

    chain_response = client.option_chains(
        symbol=resolve_option_symbol(ticker),
        fromDate=from_date,
        toDate=to_date,
        contractType='ALL'
    )
    if not chain_response.ok:
        # Build the message outside the try so the detailed error isn't swallowed
        try:
            error_data = chain_response.json()
            error_msg = error_data.get('error', 'Unknown API error')
            if 'error_description' in error_data:
                error_msg += f": {error_data['error_description']}"
        except Exception:
            error_msg = f"{chain_response.status_code} {chain_response.reason}"
        raise Exception(f"Schwab API Error: {error_msg}")

    chain = chain_response.json()
    with _chain_cache_lock:
        now = time.time()
        for stale_key in [k for k, (_, ts) in _chain_cache.items() if now - ts > 30]:
            _chain_cache.pop(stale_key, None)
        _chain_cache[key] = (chain, now)
    return chain


def fetch_options_for_date(ticker, date, exposure_metric="Open Interest", delta_adjusted: bool = False, calculate_in_notional: bool = True, S=None, chain=None):
    """Build call/put frames for one expiry. `chain` may be a pre-fetched multi-expiry
    chain response; only the contracts expiring on `date` are used."""
    if client is None:
        raise Exception("Schwab API client not initialized. Check your environment variables.")
    
    if ticker == "MARKET" or ticker == "MARKET2":
        # MARKET: synthetic SPX-centered view blending $SPX + SPY exposures.
        # Raw dollar-notional exposures are mapped onto the SPX strike grid via moneyness.
        base_ticker = "$SPX" if ticker == "MARKET" else "SPY"
        base_price = S if S else get_current_price(base_ticker)
        
        if not base_price:
             return pd.DataFrame(), pd.DataFrame()

        # Fetch base chain to build the SPX strike grid
        base_calls_raw, base_puts_raw = fetch_options_for_date(base_ticker, date, exposure_metric, delta_adjusted, calculate_in_notional)
        
        if base_calls_raw.empty and base_puts_raw.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Bucket size from base chain strike spacing (SPX ~$5, SPY ~$1)
        base_all_strikes = []
        if not base_calls_raw.empty: base_all_strikes.extend(base_calls_raw['strike'].tolist())
        if not base_puts_raw.empty: base_all_strikes.extend(base_puts_raw['strike'].tolist())
        bucket_size = get_strike_interval(base_all_strikes) if base_all_strikes else 5.0

        if ticker == "MARKET":
            component_tickers = ["$SPX", "SPY"]
        else:
            component_tickers = ["SPY"]
        
        calls_list = []
        puts_list = []

        # Columns split across SPX buckets during moneyness interpolation
        weight_cols = ['GEX', 'DEX', 'VEX', 'Charm', 'Speed', 'Vomma', 'Color', 'openInterest', 'volume']

        for comp_tick in component_tickers:
            if comp_tick == base_ticker:
                c, p = base_calls_raw.copy(), base_puts_raw.copy()
                comp_price = base_price
            else:
                comp_price = get_current_price(comp_tick)
                if not comp_price: continue
                c, p = fetch_options_for_date(comp_tick, date, exposure_metric, delta_adjusted, calculate_in_notional)
                c, p = c.copy() if not c.empty else c, p.copy() if not p.empty else p
            
            if c.empty and p.empty: continue
            
            is_base = (comp_tick == base_ticker)

            if not c.empty:
                c = c.copy()
                if is_base:
                    c['strike'] = (c['strike'] / bucket_size).round() * bucket_size
                    calls_list.append(c)
                else:
                    # Map to SPX-equivalent strike via moneyness; interpolate across
                    # adjacent buckets so exposure is conserved (weight_lo + weight_hi = 1).
                    exact = (c['strike'] / comp_price) * base_price
                    exact = exact.round(6)
                    bucket_lo = np.floor(exact / bucket_size) * bucket_size
                    bucket_hi = bucket_lo + bucket_size
                    weight_hi = (exact - bucket_lo) / bucket_size
                    weight_lo = 1.0 - weight_hi
                    
                    c_lo = c.copy()
                    c_hi = c.copy()
                    c_lo['strike'] = bucket_lo
                    c_hi['strike'] = bucket_hi
                    for col in weight_cols:
                        if col in c_lo.columns:
                            c_lo[col] = c_lo[col] * weight_lo
                            c_hi[col] = c_hi[col] * weight_hi
                    
                    calls_list.append(c_lo)
                    calls_list.append(c_hi)

            if not p.empty:
                p = p.copy()
                if is_base:
                    p['strike'] = (p['strike'] / bucket_size).round() * bucket_size
                    puts_list.append(p)
                else:
                    exact = (p['strike'] / comp_price) * base_price
                    exact = exact.round(6)
                    bucket_lo = np.floor(exact / bucket_size) * bucket_size
                    bucket_hi = bucket_lo + bucket_size
                    weight_hi = (exact - bucket_lo) / bucket_size
                    weight_lo = 1.0 - weight_hi
                    
                    p_lo = p.copy()
                    p_hi = p.copy()
                    p_lo['strike'] = bucket_lo
                    p_hi['strike'] = bucket_hi
                    for col in weight_cols:
                        if col in p_lo.columns:
                            p_lo[col] = p_lo[col] * weight_lo
                            p_hi[col] = p_hi[col] * weight_hi
                    
                    puts_list.append(p_lo)
                    puts_list.append(p_hi)

        combined_calls = pd.concat(calls_list, ignore_index=True) if calls_list else pd.DataFrame()
        combined_puts = pd.concat(puts_list, ignore_index=True) if puts_list else pd.DataFrame()

        def aggregate_market_data(df):
            if df.empty: return df
            sum_cols = ['openInterest', 'volume', 'GEX', 'DEX', 'VEX', 'Charm', 'Speed', 'Vomma', 'Color']
            agg_dict = {col: 'sum' for col in sum_cols if col in df.columns}
            for col in df.columns:
                if col not in agg_dict and col != 'strike':
                    agg_dict[col] = 'first'
            return df.groupby('strike', as_index=False).agg(agg_dict)

        combined_calls = aggregate_market_data(combined_calls)
        combined_puts = aggregate_market_data(combined_puts)
        
        return combined_calls, combined_puts

    try:
        expiry = datetime.strptime(date, '%Y-%m-%d').date()
        if chain is None:
            chain = _fetch_chain_json(ticker, expiry.strftime('%Y-%m-%d'), expiry.strftime('%Y-%m-%d'))
        S = float(chain.get('underlyingPrice', 0))
        if S == 0:
            S = get_current_price(ticker)
        if S is None:
            return pd.DataFrame(), pd.DataFrame()

        # Cache spot price so /update can skip a redundant quote API call
        _chain_spot_cache[ticker] = (S, time.time())

        # Calculate time to expiration in years (60-min CBOE floor applied inside)
        t = calculate_time_to_expiration(expiry)
        r = 0.02  # risk-free rate (2% as default to match Yahoo script)
        
        calls_data = []
        puts_data = []
        display_tickers = format_display_ticker(ticker)
        expiry_date_obj = expiry  # datetime.date — same for all options in this call

        # Parse options into plain dicts WITHOUT computing Greeks (vectorized below)
        for exp_date, strikes in chain.get('callExpDateMap', {}).items():
            if exp_date.split(':')[0] != date:
                continue
            for strike, options in strikes.items():
                for option in options:
                    if any(option['symbol'].startswith(dt) for dt in display_tickers):
                        K = float(option['strikePrice'])
                        raw_vol = float(option.get('volatility', -999.0))
                        vol = (raw_vol / 100) if raw_vol > 0 else 0.20
                        last = float(option['last'])
                        bid = float(option['bid'])
                        ask = float(option['ask'])
                        calls_data.append({
                            'contractSymbol': option['symbol'],
                            'strike': K,
                            'lastPrice': last,
                            'bid': bid,
                            'ask': ask,
                            'volume': int(option['totalVolume']),
                            'openInterest': int(option['openInterest']),
                            'impliedVolatility': vol,
                            'inTheMoney': option['inTheMoney'],
                            'expiration': expiry_date_obj,
                            'side': infer_side(last, bid, ask),
                        })

        for exp_date, strikes in chain.get('putExpDateMap', {}).items():
            if exp_date.split(':')[0] != date:
                continue
            for strike, options in strikes.items():
                for option in options:
                    if any(option['symbol'].startswith(dt) for dt in display_tickers):
                        K = float(option['strikePrice'])
                        raw_vol = float(option.get('volatility', -999.0))
                        vol = (raw_vol / 100) if raw_vol > 0 else 0.20
                        last = float(option['last'])
                        bid = float(option['bid'])
                        ask = float(option['ask'])
                        puts_data.append({
                            'contractSymbol': option['symbol'],
                            'strike': K,
                            'lastPrice': last,
                            'bid': bid,
                            'ask': ask,
                            'volume': int(option['totalVolume']),
                            'openInterest': int(option['openInterest']),
                            'impliedVolatility': vol,
                            'inTheMoney': option['inTheMoney'],
                            'expiration': expiry_date_obj,
                            'side': infer_side(last, bid, ask),
                        })

        def _apply_vectorized_greeks(option_list, flag_int):
            """Vectorized Greek + exposure computation for a list of option dicts."""
            if not option_list:
                return pd.DataFrame()
            df = pd.DataFrame(option_list)
            if t <= 0:
                for col in ('delta','gamma','theta','vega','rho','DEX','GEX','VEX','Charm','Speed','Vomma','Color'):
                    df[col] = 0.0
                return df

            K_arr = df['strike'].to_numpy(dtype=float)
            sigma_arr = df['impliedVolatility'].to_numpy(dtype=float)

            # Compute exposure weight per selected metric
            oi_arr = df['openInterest'].to_numpy(dtype=float)
            vol_arr = df['volume'].to_numpy(dtype=float)
            if exposure_metric == 'Volume':
                weight_arr = vol_arr
            elif exposure_metric == 'Max OI vs Volume':
                weight_arr = np.maximum(oi_arr, vol_arr)
            elif exposure_metric == 'OI + Volume':
                weight_arr = oi_arr + vol_arr
            else:
                weight_arr = oi_arr

            results = _compute_all_greeks_and_exposures_vectorized(
                np.full(len(df), flag_int, dtype=np.int8),
                S, K_arr, t, sigma_arr, weight_arr,
                r=r, q=0,
                delta_adjusted=delta_adjusted,
                calculate_in_notional=calculate_in_notional,
            )
            for col, arr in results.items():
                df[col] = arr
            return df

        calls = _apply_vectorized_greeks(calls_data, 0)
        puts = _apply_vectorized_greeks(puts_data, 1)
        return calls, puts
        
    except Exception as e:
        msg = f"Error fetching options chain: {e}"
        print(msg)
        # Propagate so callers (API routes) can return the error to clients
        raise Exception(msg)

def _compute_all_greeks_and_exposures_vectorized(
    flags, S, K_arr, t, sigma_arr, weight_arr, r=0.02, q=0,
    delta_adjusted=False, calculate_in_notional=True,
):
    """
    Vectorized computation of all Greeks and exposures for an array of options.
    flags: numpy int array — 0 = call, 1 = put
    K_arr, sigma_arr, weight_arr: numpy float arrays (same length as flags)
    t: scalar time to expiration in years (same for all options in a single-date call)
    Returns a dict mapping column name -> numpy array.
    """
    t = max(float(t), 1e-5)
    sqrt_t = math.sqrt(t)

    # Guard against zero/negative sigma — replace with 0.20 floor
    sigma_arr = np.where(sigma_arr > 0, sigma_arr, 0.20)

    log_SK = np.log(S / K_arr)
    d1 = (log_SK + (r - q + 0.5 * sigma_arr ** 2) * t) / (sigma_arr * sqrt_t)
    d2 = d1 - sigma_arr * sqrt_t

    exp_qt = math.exp(-q * t)
    exp_rt = math.exp(-r * t)
    norm_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_neg_d1 = 1.0 - cdf_d1
    cdf_neg_d2 = 1.0 - cdf_d2

    # Delta
    call_delta = exp_qt * cdf_d1
    put_delta = exp_qt * (cdf_d1 - 1.0)
    delta = np.where(flags == 0, call_delta, put_delta)

    # Gamma (same for calls and puts)
    gamma = exp_qt * norm_d1 / (S * sigma_arr * sqrt_t)

    # Vega (same)
    vega = S * exp_qt * norm_d1 * sqrt_t

    # Vanna (same)
    vanna = -exp_qt * norm_d1 * d2 / sigma_arr

    # Theta
    term1 = -S * exp_qt * norm_d1 * sigma_arr / (2.0 * sqrt_t)
    call_theta = term1 - r * K_arr * exp_rt * cdf_d2 + q * S * exp_qt * cdf_d1
    put_theta = term1 + r * K_arr * exp_rt * cdf_neg_d2 - q * S * exp_qt * cdf_neg_d1
    theta = np.where(flags == 0, call_theta, put_theta)

    # Rho
    call_rho = K_arr * t * exp_rt * cdf_d2
    put_rho = -K_arr * t * exp_rt * cdf_neg_d2
    rho = np.where(flags == 0, call_rho, put_rho)

    # Charm
    inner_charm = norm_d1 * (2.0 * (r - q) * t - d2 * sigma_arr * sqrt_t) / (2.0 * t * sigma_arr * sqrt_t)
    call_charm = -exp_qt * (inner_charm - q * cdf_d1)
    put_charm = -exp_qt * (inner_charm + q * cdf_neg_d1)
    charm = np.where(flags == 0, call_charm, put_charm)

    # Speed
    speed = -gamma * (d1 / (sigma_arr * sqrt_t) + 1.0) / S

    # Vomma
    vomma = vega * (d1 * d2) / sigma_arr

    # Color: dGamma/dt in calendar time (same sign convention as charm)
    color = exp_qt * (norm_d1 / (2.0 * S * t * sigma_arr * sqrt_t)) * \
            (2.0 * q * t + 1.0 + (2.0 * (r - q) * t - d2 * sigma_arr * sqrt_t) * d1 / (sigma_arr * sqrt_t))

    # --- Exposure calculations ---
    contract_size = 100
    spot_mult = S if calculate_in_notional else 1.0

    dex = delta * weight_arr * contract_size * spot_mult
    gex = gamma * weight_arr * contract_size * S * spot_mult * 0.01
    vex = vanna * weight_arr * contract_size * spot_mult * 0.01
    charm_exp = charm * weight_arr * contract_size * spot_mult / 365.0
    speed_exp = speed * weight_arr * contract_size * S * spot_mult * 0.01
    vomma_exp = vomma * weight_arr * contract_size * 0.01
    color_exp = color * weight_arr * contract_size * S * spot_mult * 0.01 / 365.0

    if delta_adjusted:
        abs_delta = np.abs(delta)
        gex *= abs_delta
        vex *= abs_delta
        charm_exp *= abs_delta
        speed_exp *= abs_delta
        vomma_exp *= abs_delta
        color_exp *= abs_delta

    return {
        'delta': delta, 'gamma': gamma, 'vega': vega, 'vanna': vanna,
        'theta': theta, 'rho': rho,
        'DEX': dex, 'GEX': gex, 'VEX': vex, 'Charm': charm_exp,
        'Speed': speed_exp, 'Vomma': vomma_exp, 'Color': color_exp,
    }


def get_current_price(ticker):
    if client is None:
        raise Exception("Schwab API client not initialized. Check your environment variables.")
        
    if ticker == "MARKET":
        ticker = "$SPX"
    elif ticker == "MARKET2":
        ticker = "SPY"
    try:
        quote_response = client.quotes(ticker)
        if not quote_response.ok:
            raise Exception(f"Failed to fetch quote: {quote_response.status_code} {quote_response.reason}")
        quote = quote_response.json()
        if quote and ticker in quote:
            return quote[ticker]['quote']['lastPrice']
        raise Exception("Malformed quote data returned from Schwab API")
    except Exception as e:
        msg = f"Error fetching price from Schwab API: {e}"
        print(msg)
        raise Exception(msg)

def get_option_expirations(ticker):
    if client is None:
        raise Exception("Schwab API client not initialized. Check your environment variables.")
    
    if ticker == "MARKET":
        ticker = "$SPX"
    elif ticker == "MARKET2":
        ticker = "SPY"
    try:
        expiration_dates = []
        expiration_chain_error = None
        option_symbol = resolve_option_symbol(ticker)
        try:
            response = client.option_expiration_chain(option_symbol)
            if not response.ok:
                raise Exception(f"{response.url} -> {response.status_code} {response.reason}: {response.text[:200]}")
            response_json = response.json() or {}
            for item in response_json.get('expirationList') or []:
                expiration = item.get('expirationDate') or item.get('expiration')
                if expiration:
                    expiration_dates.append(str(expiration)[:10])
        except Exception as e:
            expiration_chain_error = e
            print(f"expirationchain failed for {ticker}: {e}")

        if not expiration_dates:
            # Fallback (e.g. index symbols like $SPX): read the expiry dates from a one-strike
            # option chain. Map keys look like "2026-09-24:0".
            chain_response = client.option_chains(symbol=option_symbol, contractType='CALL', strikeCount=1)
            if not chain_response.ok:
                raise Exception(
                    f"expirationchain: {expiration_chain_error or 'no expirations'}; "
                    f"chains: {chain_response.url} -> {chain_response.status_code} {chain_response.reason}: {chain_response.text[:200]}"
                )
            chain = chain_response.json() or {}
            for key in list((chain.get('callExpDateMap') or {}).keys()) + list((chain.get('putExpDateMap') or {}).keys()):
                expiration_dates.append(key.split(':')[0])

        # SPX lists some dates under both the SPX and SPXW roots; selecting a duplicate
        # would double-count that expiry's exposure. Schwab can keep listing yesterday's
        # expiry for a while after midnight; it has no chain anymore, so drop it.
        today = _today_et_str()
        return sorted(d for d in set(expiration_dates) if d >= today)
    except Exception as e:
        msg = f"Error fetching option expirations: {e}"
        print(msg)
        # Propagate the error so route handlers or Flask error handlers can return it to clients
        raise Exception(msg)

_EXPIRATIONS_CACHE_TTL_SEC = 600
_expirations_cache = {}  # ticker -> (dates, fetched_at_unix_ts)


def _today_et_str():
    return datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')


def get_cached_option_expirations(ticker):
    cached = _expirations_cache.get(ticker)
    if cached and (time.time() - cached[1]) < _EXPIRATIONS_CACHE_TTL_SEC:
        # A list cached just before midnight must not hand out yesterday's expiry
        today = _today_et_str()
        return [d for d in cached[0] if d >= today]
    dates = get_option_expirations(ticker)
    _expirations_cache[ticker] = (dates, time.time())
    return dates


def split_expired_expiries(expiry_dates):
    """Split a selection into (live, expired) by ET date. What replaces an expired
    selection is the page's call (its 0DTE / week rules roll forward, hand-picked dates don't)."""
    today = _today_et_str()
    live = [d for d in expiry_dates if str(d)[:10] >= today]
    expired = [d for d in expiry_dates if str(d)[:10] < today]
    return live, expired


def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color to rgba string with specified alpha."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return f'rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})'

def get_colors(base_color, values, max_val, coloring_mode='Solid'):
    """
    Apply coloring mode to a set of values.
    
    Args:
        base_color: Hex color string (e.g., '#00FF00')
        values: Array/list of numeric values
        max_val: Maximum value for normalization
        coloring_mode: 'Solid', 'Linear Intensity', or 'Ranked Intensity'
    
    Returns:
        Either a single color string (Solid mode) or list of RGBA colors
    """
    # Solid mode: return base color as-is
    if coloring_mode == 'Solid':
        return base_color
    
    # Handle edge case
    if max_val == 0:
        return base_color
    
    # Convert to list if series/array
    vals = values.tolist() if hasattr(values, 'tolist') else list(values)
    
    if coloring_mode == 'Linear Intensity':
        # Linear mapping: opacity from 0.3 to 1.0
        # Formula: 0.3 + 0.7 * (|value| / max_value)
        return [hex_to_rgba(base_color, 0.3 + 0.7 * (abs(v) / max_val)) for v in vals]
    
    elif coloring_mode == 'Ranked Intensity':
        # Exponential mapping: opacity from 0.1 to 1.0 with cubic power
        # Formula: 0.1 + 0.9 * ((|value| / max_value) ^ 3)
        # This aggressively fades lower values, making only top exposures bright
        return [hex_to_rgba(base_color, 0.1 + 0.9 * ((abs(v) / max_val) ** 3)) for v in vals]
    
    else:
        return base_color

def get_net_colors(values, max_val, call_color, put_color, coloring_mode='Solid'):
    """
    Apply coloring mode to net exposure values (can be positive or negative).
    Color is based on sign: positive = call_color, negative = put_color.
    
    Args:
        values: Array/list of numeric values (can be negative)
        max_val: Maximum absolute value for normalization
        call_color: Hex color for positive values
        put_color: Hex color for negative values
        coloring_mode: 'Solid', 'Linear Intensity', or 'Ranked Intensity'
    
    Returns:
        List of colors (either hex or RGBA based on mode)
    """
    vals = values.tolist() if hasattr(values, 'tolist') else list(values)
    
    if coloring_mode == 'Solid':
        return [call_color if v >= 0 else put_color for v in vals]
    
    if max_val == 0:
        return [call_color if v >= 0 else put_color for v in vals]
    
    colors = []
    for val in vals:
        base = call_color if val >= 0 else put_color
        
        if coloring_mode == 'Linear Intensity':
            opacity = 0.3 + 0.7 * (abs(val) / max_val)
            colors.append(hex_to_rgba(base, min(1.0, opacity)))
        
        elif coloring_mode == 'Ranked Intensity':
            opacity = 0.1 + 0.9 * ((abs(val) / max_val) ** 3)
            colors.append(hex_to_rgba(base, min(1.0, opacity)))
        
        else:  # Solid fallback
            colors.append(base)
    
    return colors


def add_current_price_reference(fig, price, horizontal=False, text_color='white'):
    """Add a current-price guide line and keep horizontal labels in chart margin space."""
    if horizontal:
        fig.add_hline(
            y=price,
            line_dash="dash",
            line_color=text_color,
            opacity=0.5,
            line_width=1
        )
        fig.add_annotation(
            x=1.0,
            y=price,
            xref='paper',
            yref='y',
            text=f"{price:.2f}",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            xshift=4,
            font=dict(color=text_color, size=11),
        )
        return

    fig.add_vline(
        x=price,
        line_dash="dash",
        line_color=text_color,
        opacity=0.5,
        annotation_text=f"{price:.2f}",
        annotation_position="top",
        annotation_font_color=text_color,
        line_width=1
    )


def build_chart_title_text(base_title, selected_expiries=None, showing_last_session=False):
    chart_title = base_title
    if selected_expiries and len(selected_expiries) > 1:
        chart_title = f"{base_title} ({len(selected_expiries)} expiries)"
    if showing_last_session:
        chart_title += ' (Last Session)'
    return chart_title


def calc_linear_trend_line(timestamps, values):
    """Fit a least-squares trend through valid points; returns (times, trend_values) or (None, None)."""
    # Regress against elapsed seconds (not sample index) so gaps in sampling
    # don't distort the slope.
    points = [
        (timestamp.timestamp() if hasattr(timestamp, 'timestamp') else float(timestamp), timestamp, value)
        for timestamp, value in zip(timestamps, values)
        if value is not None and math.isfinite(value)
    ]
    if len(points) < 2:
        return None, None
    origin = points[0][0]
    points = [(x - origin, timestamp, value) for x, timestamp, value in points]

    n = len(points)
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[2] for point in points)
    sum_xy = sum(point[0] * point[2] for point in points)
    sum_xx = sum(point[0] * point[0] for point in points)
    denominator = (n * sum_xx) - (sum_x * sum_x)
    slope = 0 if denominator == 0 else ((n * sum_xy) - (sum_x * sum_y)) / denominator
    intercept = (sum_y - (slope * sum_x)) / n

    trend_times = [point[1] for point in points]
    trend_values = [intercept + (slope * point[0]) for point in points]
    return trend_times, trend_values


def compute_iv_stats(calls, puts, _spot=None):
    """Compute volume-weighted IV stats for the price-info bar."""
    if (calls is None or calls.empty) and (puts is None or puts.empty):
        return None

    def _valid_iv(value):
        try:
            iv = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(iv) and iv > 0

    def _volume_weighted_iv(df):
        if df is None or df.empty or 'impliedVolatility' not in df.columns:
            return None
        valid = df[df['impliedVolatility'].apply(_valid_iv)].copy()
        if valid.empty:
            return None
        if 'volume' in valid.columns:
            weighted = valid[valid['volume'] > 0]
            if not weighted.empty:
                total_vol = weighted['volume'].sum()
                return float((weighted['impliedVolatility'] * weighted['volume']).sum() / total_vol)
        return float(valid['impliedVolatility'].mean())

    def _pct(iv):
        return round(iv * 100, 2) if iv is not None else None

    call_vw_iv = _pct(_volume_weighted_iv(calls))
    put_vw_iv = _pct(_volume_weighted_iv(puts))
    if call_vw_iv is None and put_vw_iv is None:
        return None

    return {
        'call_vw_iv': call_vw_iv,
        'put_vw_iv': put_vw_iv,
    }


def build_bar_chart_title(base_title, call_total, put_total, net_total, call_color, put_color,
                          selected_expiries=None):
    return build_chart_title_text(base_title, selected_expiries=selected_expiries)


def build_bar_chart_totals_annotation(call_total, put_total, net_total, call_color, put_color):
    net_color = call_color if net_total >= 0 else put_color
    return dict(
        text=(
            f"<b><span style='color:{call_color}'>{format_large_number(abs(call_total))}</span>"
            f" <span style='color:#8e9cb1'>|</span> "
            f"<span style='color:{net_color}'>Net: {format_large_number(abs(net_total))}</span>"
            f" <span style='color:#8e9cb1'>|</span> "
            f"<span style='color:{put_color}'>{format_large_number(abs(put_total))}</span></b>"
        ),
        x=0.5,
        y=1.04,
        xref='paper',
        yref='paper',
        xanchor='center',
        yanchor='bottom',
        showarrow=False,
        align='center',
        font=dict(size=13, color='#CCCCCC'),
    )



def build_left_aligned_title(title_text, text_color='#CCCCCC', size=16, y=0.98):
    return dict(
        text=f"<b>{title_text}</b>",
        font=dict(color=text_color, size=size),
        x=0.01,
        xanchor='left',
        y=y,
    )


def ensure_bar_text_visibility(fig, horizontal=False):
    """Keep bar labels visible by disabling clipping and padding the value axis."""
    fig.update_traces(cliponaxis=False, selector=dict(type='bar'))

    values = []
    for trace in fig.data:
        if getattr(trace, 'type', None) != 'bar':
            continue
        series_values = trace.x if horizontal else trace.y
        if not series_values:
            continue
        values.extend(float(value) for value in series_values if value is not None)

    if not values:
        return

    min_value = min(values)
    max_value = max(values)
    value_span = max_value - min_value
    padding = value_span * 0.12 if value_span else max(abs(max_value), 1.0) * 0.12
    lower_bound = min(0, min_value - padding)
    upper_bound = max(0, max_value + padding)

    if horizontal:
        fig.update_xaxes(range=[lower_bound, upper_bound])
    else:
        fig.update_yaxes(range=[lower_bound, upper_bound])


def create_exposure_heatmap(calls, puts, S, strike_range=0.02, show_calls=True, show_puts=True,
                            show_net=True, call_color='#00FF00', put_color='#FF0000',
                            selected_expiries=None, heatmap_type='GEX',
                            heatmap_coloring_mode='Global'):
    normalized_type = normalize_level_type(heatmap_type)
    value_column = resolve_level_value_column(normalized_type)
    display_name = INTERVAL_LEVEL_DISPLAY_NAMES.get(normalized_type, normalized_type)
    title_display_name = HEATMAP_TITLE_DISPLAY_NAMES.get(normalized_type, display_name)
    text_color = '#CCCCCC'
    grid_color = '#333333'
    background_color = '#1E1E1E'

    base_title = f'{title_display_name} Heatmap'
    empty_title = build_chart_title_text(base_title, selected_expiries=selected_expiries)

    if (
        value_column not in calls.columns or
        value_column not in puts.columns or
        'expiration' not in calls.columns or
        'expiration' not in puts.columns
    ):
        fig = go.Figure()
        fig.update_layout(
            title=build_left_aligned_title(empty_title, text_color=text_color),
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            font=dict(color=text_color),
            annotations=[dict(
                text='No exposure data available',
                x=0.5,
                y=0.5,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(color=text_color, size=14),
            )],
        )
        return fig.to_json()

    min_strike = S * (1 - strike_range)
    max_strike = S * (1 + strike_range)

    calls_df = calls[['expiration', 'strike', value_column]].copy()
    puts_df = puts[['expiration', 'strike', value_column]].copy()

    calls_df = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
    puts_df = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]

    all_strikes = list(calls_df['strike']) + list(puts_df['strike'])
    strike_interval = get_strike_interval(all_strikes) if all_strikes else 1.0

    def aggregate_heatmap_frame(df):
        if df.empty:
            return pd.DataFrame(columns=['expiration_label', 'strike', value_column])

        working = df.copy()
        working['strike'] = working['strike'].apply(lambda value: round_to_strike(value, strike_interval))
        working['expiration_label'] = working['expiration'].astype(str)
        return working.groupby(['expiration_label', 'strike'], as_index=False)[value_column].sum()

    calls_grouped = aggregate_heatmap_frame(calls_df)
    puts_grouped = aggregate_heatmap_frame(puts_df)

    call_lookup = {
        (row['expiration_label'], row['strike']): row[value_column]
        for _, row in calls_grouped.iterrows()
    }
    put_lookup = {
        (row['expiration_label'], row['strike']): row[value_column]
        for _, row in puts_grouped.iterrows()
    }

    available_expiries = sorted(
        set(calls_grouped['expiration_label'].tolist()) | set(puts_grouped['expiration_label'].tolist())
    )
    if selected_expiries:
        requested_expiries = [str(expiry) for expiry in selected_expiries]
        expiry_labels = [expiry for expiry in requested_expiries if expiry in available_expiries]
        expiry_labels.extend(expiry for expiry in available_expiries if expiry not in expiry_labels)
    else:
        expiry_labels = available_expiries

    strikes = sorted(set(calls_grouped['strike'].tolist()) | set(puts_grouped['strike'].tolist()))

    if not expiry_labels or not strikes:
        fig = go.Figure()
        fig.update_layout(
            title=build_left_aligned_title(empty_title, text_color=text_color),
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            font=dict(color=text_color),
            annotations=[dict(
                text='No exposure data in selected strike range',
                x=0.5,
                y=0.5,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(color=text_color, size=14),
            )],
        )
        return fig.to_json()

    if show_net or (show_calls and show_puts):
        mode_label = 'Net'
    elif show_calls:
        mode_label = 'Call'
    elif show_puts:
        mode_label = 'Put'
    else:
        mode_label = 'Net'

    actual_values = []
    text_values = []
    for strike in strikes:
        value_row = []
        text_row = []
        for expiry in expiry_labels:
            call_value = call_lookup.get((expiry, strike), 0.0)
            put_value = put_lookup.get((expiry, strike), 0.0)

            if mode_label == 'Call':
                exposure_value = call_value
            elif mode_label == 'Put':
                # Show the put's contribution to net: only GEX/Volume subtract puts
                exposure_value = -put_value if normalized_type in ('GEX', 'Volume') else put_value
            else:
                exposure_value = combine_level_values(normalized_type, call_value, put_value)

            value_row.append(exposure_value)
            text_row.append(format_large_number(exposure_value))
        actual_values.append(value_row)
        text_values.append(text_row)

    global_peak = max((abs(value) for row in actual_values for value in row), default=0.0)
    exposure_floor = global_peak * 0.01 if global_peak > 0 else 0.0
    visible_indices = [
        si for si in range(len(strikes))
        if max((abs(actual_values[si][ei]) for ei in range(len(expiry_labels))), default=0.0) > exposure_floor
    ]
    if visible_indices and len(visible_indices) < len(strikes):
        strikes = [strikes[si] for si in visible_indices]
        actual_values = [actual_values[si] for si in visible_indices]
        text_values = [text_values[si] for si in visible_indices]

    if heatmap_coloring_mode == 'Per Expiration':
        column_maxima = []
        for expiry_index in range(len(expiry_labels)):
            column_values = [row[expiry_index] for row in actual_values]
            column_max = max((abs(value) for value in column_values), default=0.0)
            column_maxima.append(column_max if column_max > 0 else 1.0)
        color_values = []
        for row in actual_values:
            color_values.append([
                row[idx] / column_maxima[idx] if column_maxima[idx] else 0.0
                for idx in range(len(row))
            ])
        colorbar_max = 1.0
    else:
        color_values = actual_values
        max_abs_exposure = max((abs(value) for row in actual_values for value in row), default=0.0)
        colorbar_max = max_abs_exposure if max_abs_exposure > 0 else 1.0

    total_call_exposure = calls[value_column].sum() if not calls.empty else 0
    total_put_exposure = puts[value_column].sum() if not puts.empty else 0
    if mode_label == 'Call':
        total_net_exposure = total_call_exposure
    elif mode_label == 'Put':
        total_net_exposure = -total_put_exposure
    else:
        total_net_exposure = combine_level_values(normalized_type, total_call_exposure, total_put_exposure)

    chart_title = build_chart_title_text(
        f'{title_display_name} Heatmap',
        selected_expiries=selected_expiries,
    )

    heatmap_hover_template = build_hover_template(
        mode_label,
        [
            ('Expiration', '%{x}'),
            ('Price', '$%{y:.2f}'),
            (title_display_name, '%{customdata}'),
        ],
    )

    heatmap_kwargs = dict(
        x=expiry_labels,
        y=strikes,
        z=color_values,
        customdata=actual_values,
        colorscale=[
            [0.0, put_color],
            [0.5, '#2A2A2A'],
            [1.0, call_color],
        ],
        zmin=-colorbar_max,
        zmax=colorbar_max,
        zmid=0,
        name='Exposure',
        hoverongaps=False,
        showscale=False,
        xgap=2,
        ygap=2,
        hovertemplate=heatmap_hover_template,
    )

    label_font_size = 11  # JS will recalculate this on every render/resize

    fig = go.Figure(data=[go.Heatmap(**heatmap_kwargs)])

    cell_annotations = []
    for strike_index, strike in enumerate(strikes):
        for expiry_index, expiry_label in enumerate(expiry_labels):
            cell_annotations.append(dict(
                x=expiry_label,
                y=strike,
                xref='x',
                yref='y',
                text=text_values[strike_index][expiry_index],
                showarrow=False,
                xanchor='center',
                yanchor='middle',
                align='center',
                font=dict(color='rgba(255, 255, 255, 0.82)', size=label_font_size, family='Arial, sans-serif'),
            ))

    # Draw a border around the cell with the highest absolute exposure in each expiration column
    highlight_shapes = []
    half_interval = strike_interval / 2
    for expiry_index in range(len(expiry_labels)):
        col_abs = [(abs(actual_values[si][expiry_index]), si) for si in range(len(strikes))]
        if not col_abs:
            continue
        _, max_si = max(col_abs)
        max_strike_val = strikes[max_si]
        highlight_shapes.append(dict(
            type='rect',
            xref='x',
            yref='y',
            x0=expiry_index - 0.5,
            x1=expiry_index + 0.5,
            y0=max_strike_val - half_interval,
            y1=max_strike_val + half_interval,
            line=dict(color='#AA00FF', width=2),
            fillcolor='rgba(0,0,0,0)',
            layer='above',
        ))

    add_current_price_reference(fig, S, horizontal=True, text_color=text_color)

    y_tick_format = '.2f' if strike_interval < 1 else ('.1f' if strike_interval < 5 else '.0f')
    parsed_expiries = []
    for expiry_label in expiry_labels:
        try:
            parsed_expiries.append(datetime.strptime(expiry_label, '%Y-%m-%d'))
        except Exception:
            parsed_expiries.append(None)

    multiple_expiry_years = len({dt.year for dt in parsed_expiries if dt is not None}) > 1
    compact_expiry_labels = []
    for expiry_label, parsed_expiry in zip(expiry_labels, parsed_expiries):
        if parsed_expiry is None:
            compact_expiry_labels.append(expiry_label)
        elif multiple_expiry_years:
            compact_expiry_labels.append(parsed_expiry.strftime('%m/%d/%y'))
        else:
            compact_expiry_labels.append(parsed_expiry.strftime('%m/%d'))

    if len(set(compact_expiry_labels)) != len(compact_expiry_labels):
        compact_expiry_labels = [
            parsed_expiry.strftime('%m/%d/%y') if parsed_expiry is not None else expiry_label
            for expiry_label, parsed_expiry in zip(expiry_labels, parsed_expiries)
        ]

    max_visible_expiry_ticks = 8
    if len(expiry_labels) > max_visible_expiry_ticks:
        visible_tick_step = max(1, math.ceil(len(expiry_labels) / max_visible_expiry_ticks))
        compact_expiry_ticktext = [
            label if idx % visible_tick_step == 0 else ''
            for idx, label in enumerate(compact_expiry_labels)
        ]
    else:
        compact_expiry_ticktext = compact_expiry_labels

    expiry_tick_font_size = 10 if len(expiry_labels) <= 8 else 9
    y_pad = strike_interval * 0.5
    y_range = [strikes[0] - y_pad, strikes[-1] + y_pad]
    fig.update_layout(
        shapes=list(fig.layout.shapes) + highlight_shapes,
        title=build_left_aligned_title(chart_title, text_color=text_color),
        annotations=list(fig.layout.annotations) + cell_annotations + [
            build_bar_chart_totals_annotation(
                total_call_exposure,
                total_put_exposure,
                total_net_exposure,
                call_color,
                put_color,
            )
        ],
        xaxis=dict(
            title='Expiration',
            title_font=dict(color=text_color),
            tickfont=dict(color=text_color, size=expiry_tick_font_size),
            gridcolor=grid_color,
            linecolor=grid_color,
            showgrid=False,
            zeroline=False,
            type='category',
            tickmode='array',
            tickvals=expiry_labels,
            ticktext=compact_expiry_ticktext,
            categoryorder='array',
            categoryarray=expiry_labels,
            tickangle=0,
            constrain='domain',
        ),
        yaxis=dict(
            title='Price',
            title_font=dict(color=text_color),
            tickfont=dict(color=text_color),
            gridcolor=grid_color,
            linecolor=grid_color,
            showgrid=False,
            zeroline=False,
            tickformat=y_tick_format,
            range=y_range,
            autorange=False,
            constrain='domain',
        ),
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font=dict(color=text_color),
        margin=dict(l=44, r=52, t=48, b=32),
        hoverlabel=dict(
            bgcolor=background_color,
            font_size=12,
            font_family='Arial',
        ),
        transition=dict(duration=0, easing='linear'),
    )

    return fig.to_json()

def create_exposure_chart(calls, puts, exposure_type, title, S, strike_range=0.02, show_calls=True, show_puts=True, show_net=True, coloring_mode='Solid', call_color='#00FF00', put_color='#FF0000', selected_expiries=None, horizontal=False, show_abs_gex_area=False, abs_gex_opacity=0.2, highlight_max_level=False, max_level_color='#800080', max_level_mode='Absolute'):
    # Ensure the exposure_type column exists
    if exposure_type not in calls.columns or exposure_type not in puts.columns:
        print(f"Warning: {exposure_type} not found in data")
        return go.Figure().to_json()
    
    # Filter out zero values and create dataframes
    calls_df = calls[['strike', exposure_type]].copy()
    calls_df = calls_df[calls_df[exposure_type] != 0]
    calls_df['OptionType'] = 'Call'
    
    puts_df = puts[['strike', exposure_type]].copy()
    puts_df = puts_df[puts_df[exposure_type] != 0]
    puts_df['OptionType'] = 'Put'
    
    # Calculate range based on percentage of current price
    min_strike = S * (1 - strike_range)
    max_strike = S * (1 + strike_range)
    
    calls_df = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
    puts_df = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]
    
    # Determine strike interval and aggregate by rounded strikes
    all_strikes = list(calls_df['strike']) + list(puts_df['strike'])
    if all_strikes:
        strike_interval = get_strike_interval(all_strikes)
        calls_df = aggregate_by_strike(calls_df, [exposure_type], strike_interval)
        puts_df = aggregate_by_strike(puts_df, [exposure_type], strike_interval)
    
    # Calculate total net exposure from the entire chain (not just strike range)
    total_call_exposure = calls[exposure_type].sum() if not calls.empty and exposure_type in calls.columns else 0
    total_put_exposure = puts[exposure_type].sum() if not puts.empty and exposure_type in puts.columns else 0

    if exposure_type == 'GEX':
        total_net_exposure = total_call_exposure - total_put_exposure
    elif exposure_type == 'DEX':
        total_net_exposure = total_call_exposure + total_put_exposure
    else:
        total_net_exposure = total_call_exposure + total_put_exposure
    
    # Create the main title and net exposure as separate annotations
    fig = go.Figure()
    hover_metric_label = title.replace(' by Strike', '') if title.endswith(' by Strike') else title
    
    # Add Absolute GEX Area Chart if enabled
    if exposure_type == 'GEX' and show_abs_gex_area:
        try:
            # Get all unique strikes in the range
            all_strikes_abs = sorted(list(set(calls_df['strike'].tolist() + puts_df['strike'].tolist())))
            abs_gex_values = []
            
            for strike in all_strikes_abs:
                # Calculate absolute gamma at this strike (Total Gamma)
                c_val = calls_df[calls_df['strike'] == strike][exposure_type].sum() if not calls_df.empty else 0
                p_val = puts_df[puts_df['strike'] == strike][exposure_type].sum() if not puts_df.empty else 0
                
                # Use absolute values to get total magnitude
                total_abs_val = abs(c_val) + abs(p_val)
                abs_gex_values.append(total_abs_val)
                
            # Add the area trace
            if horizontal:
                fig.add_trace(go.Scatter(
                    y=all_strikes_abs,
                    x=abs_gex_values,
                    mode='none',
                    fill='tozerox',
                    name='Abs GEX Total',
                    fillcolor=f'rgba(200, 200, 200, {abs_gex_opacity})',
                    hoverinfo='skip',
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=all_strikes_abs,
                    y=abs_gex_values,
                    mode='none',
                    fill='tozeroy',
                    name='Abs GEX Total',
                    fillcolor=f'rgba(200, 200, 200, {abs_gex_opacity})',
                    hoverinfo='skip',
                    showlegend=False
                ))
        except Exception as e:
            print(f"Error adding Abs GEX area: {e}")

    # Define colors
    grid_color = '#333333'
    text_color = '#CCCCCC'
    background_color = '#1E1E1E'
    
    # Calculate max exposure for normalization across all data (calls, puts, net)
    max_exposure = 1.0
    all_abs_vals = []
    if not calls_df.empty:
        all_abs_vals.extend(calls_df[exposure_type].abs().tolist())
    if not puts_df.empty:
        all_abs_vals.extend(puts_df[exposure_type].abs().tolist())
    if all_abs_vals:
        max_exposure = max(all_abs_vals)
    if max_exposure == 0:
        max_exposure = 1.0  # Prevent division by zero
    
    if show_calls and not calls_df.empty:
        # Apply coloring mode
        call_colors = get_colors(call_color, calls_df[exposure_type], max_exposure, coloring_mode)
        
        if horizontal:
            fig.add_trace(go.Bar(
                y=calls_df['strike'].tolist(),
                x=calls_df[exposure_type].tolist(),
                name='Call',
                marker_color=call_colors,
                text=[format_large_number(val) for val in calls_df[exposure_type]],
                textposition='auto',
                orientation='h',
                hovertemplate=build_hover_template('Call', [('Strike', '$%{y:.2f}'), (hover_metric_label, '%{text}')]),
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=calls_df['strike'].tolist(),
                y=calls_df[exposure_type].tolist(),
                name='Call',
                marker_color=call_colors,
                text=[format_large_number(val) for val in calls_df[exposure_type]],
                textposition='auto',
                hovertemplate=build_hover_template('Call', [('Strike', '$%{x:.2f}'), (hover_metric_label, '%{text}')]),
                marker_line_width=0
            ))
    
    if show_puts and not puts_df.empty:
        # Apply coloring mode
        put_colors = get_colors(put_color, puts_df[exposure_type], max_exposure, coloring_mode)
        # Plot puts in the direction they contribute to net: GEX subtracts puts, every
        # other exposure (DEX, Vanna, Charm, ...) adds the signed put value.
        put_plot_values = (-puts_df[exposure_type] if exposure_type == 'GEX' else puts_df[exposure_type]).tolist()

        if horizontal:
            fig.add_trace(go.Bar(
                y=puts_df['strike'].tolist(),
                x=put_plot_values,
                name='Put',
                marker_color=put_colors,
                text=[format_large_number(val) for val in puts_df[exposure_type]],
                textposition='auto',
                orientation='h',
                hovertemplate=build_hover_template('Put', [('Strike', '$%{y:.2f}'), (hover_metric_label, '%{text}')]),
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=puts_df['strike'].tolist(),
                y=put_plot_values,
                name='Put',
                marker_color=put_colors,
                text=[format_large_number(val) for val in puts_df[exposure_type]],
                textposition='auto',
                hovertemplate=build_hover_template('Put', [('Strike', '$%{x:.2f}'), (hover_metric_label, '%{text}')]),
                marker_line_width=0
            ))
    
    if show_net and not (calls_df.empty and puts_df.empty):
        # Create net exposure by combining calls and puts
        all_strikes = sorted(set(calls_df['strike'].tolist() + puts_df['strike'].tolist()))
        net_exposure = []
        
        for strike in all_strikes:
            call_value = calls_df[calls_df['strike'] == strike][exposure_type].sum() if not calls_df.empty else 0
            put_value = puts_df[puts_df['strike'] == strike][exposure_type].sum() if not puts_df.empty else 0
            
            if exposure_type == 'GEX':
                net_value = call_value - put_value
            elif exposure_type == 'DEX':
                net_value = call_value + put_value
            else:
                net_value = call_value + put_value
            
            net_exposure.append(net_value)
        
        # Calculate max for net exposure normalization
        max_net_exposure = max(abs(min(net_exposure)), abs(max(net_exposure))) if net_exposure else 1.0
        if max_net_exposure == 0:
            max_net_exposure = 1.0
        
        # Apply coloring mode for net values
        net_colors = get_net_colors(net_exposure, max_net_exposure, call_color, put_color, coloring_mode)
        
        if horizontal:
            fig.add_trace(go.Bar(
                y=all_strikes,
                x=net_exposure,
                name='Net',
                base=0,  # overlay from zero instead of stacking on the Call bar
                width=strike_interval * 0.5,
                marker_color=net_colors,
                text=[format_large_number(val) for val in net_exposure],
                textposition='auto',
                orientation='h',
                hovertemplate=build_hover_template('Net', [('Strike', '$%{y:.2f}'), (hover_metric_label, '%{text}')]),
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=all_strikes,
                y=net_exposure,
                name='Net',
                base=0,  # overlay from zero instead of stacking on the Call bar
                width=strike_interval * 0.5,
                marker_color=net_colors,
                text=[format_large_number(val) for val in net_exposure],
                textposition='auto',
                hovertemplate=build_hover_template('Net', [('Strike', '$%{x:.2f}'), (hover_metric_label, '%{text}')]),
                marker_line_width=0
            ))
    
    add_current_price_reference(fig, S, horizontal=horizontal, text_color=text_color)
    
    chart_title = build_bar_chart_title(
        title,
        total_call_exposure,
        total_put_exposure,
        total_net_exposure,
        call_color,
        put_color,
        selected_expiries=selected_expiries,
    )
    
    xaxis_config = dict(
        title='',
        title_font=dict(color=text_color),
        tickfont=dict(color=text_color, size=12),
        gridcolor=grid_color,
        linecolor=grid_color,
        showgrid=False,
        zeroline=True,
        zerolinecolor=grid_color
    )
    
    yaxis_config = dict(
        title='',
        title_font=dict(color=text_color),
        tickfont=dict(color=text_color),
        gridcolor=grid_color,
        linecolor=grid_color,
        showgrid=False,
        zeroline=True,
        zerolinecolor=grid_color
    )
    
    # Configure axes based on orientation
    if horizontal:
        # Strike axis is Y
        yaxis_config.update(dict(
            range=[min_strike, max_strike],
            autorange=False,
            tickformat='.0f',
            showticklabels=True,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor=text_color,
            automargin=True
        ))
        # Value axis is X
        xaxis_config.update(dict(
            showticklabels=True
        ))
    else:
        # Strike axis is X
        xaxis_config.update(dict(
            range=[min_strike, max_strike],
            autorange=False,
            tickangle=45,
            tickformat='.0f',
            showticklabels=True,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor=text_color,
            automargin=True
        ))
        # Value axis is Y
        yaxis_config.update(dict(
            showticklabels=True
        ))

    fig.update_layout(
        title=build_left_aligned_title(chart_title, text_color=text_color),
        annotations=list(fig.layout.annotations) + [
            build_bar_chart_totals_annotation(
                total_call_exposure,
                total_put_exposure,
                total_net_exposure,
                call_color,
                put_color,
            )
        ],
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        barmode='relative',
        hovermode='y unified' if horizontal else 'x unified',
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font=dict(color=text_color),
        showlegend=False,  # Removed legend
        bargap=0.1,
        bargroupgap=0.1,
        margin=dict(l=38, r=56 if horizontal else 68, t=56, b=16),
        hoverlabel=dict(
            bgcolor=background_color,
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100,
        height=500
    )

    ensure_bar_text_visibility(fig, horizontal=horizontal)
    
    # Add hover spikes
    fig.update_xaxes(showspikes=True, spikecolor=text_color, spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor=text_color, spikethickness=1)
    
    # Logic for Highlighting Max Level
    if highlight_max_level:
        try:
            if max_level_mode == 'Net':
                # Compute net exposure for the entire chain (not just plotted range)
                all_chain_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
                chain_net_exposure = []
                for strike in all_chain_strikes:
                    call_value = calls[calls['strike'] == strike][exposure_type].sum() if not calls.empty else 0
                    put_value = puts[puts['strike'] == strike][exposure_type].sum() if not puts.empty else 0
                    if exposure_type == 'GEX':
                        net_value = call_value - put_value
                    elif exposure_type == 'DEX':
                        net_value = call_value + put_value
                    else:
                        net_value = call_value + put_value
                    chain_net_exposure.append(net_value)

                # Find the strike with the max absolute net exposure (or max/min depending on sign of total net)
                if chain_net_exposure:
                    total_chain_net = sum(chain_net_exposure)
                    if total_chain_net >= 0:
                        max_net_val = max(chain_net_exposure)
                    else:
                        max_net_val = min(chain_net_exposure)
                    # Find the strike(s) with this value
                    max_strikes = [s for s, v in zip(all_chain_strikes, chain_net_exposure) if v == max_net_val]
                    # Now, highlight the bar in the plotted Net trace that matches this strike (if present)
                    net_trace_idx = next((i for i, t in enumerate(fig.data) if t.type == 'bar' and t.name == 'Net'), None)
                    if net_trace_idx is not None:
                        plotted_strikes = fig.data[net_trace_idx].y if horizontal else fig.data[net_trace_idx].x
                        # For horizontal, y is strikes; else, x is strikes
                        if plotted_strikes:
                            # Find the index of the strike in the plot that matches max_strikes
                            highlight_idx = None
                            for idx, s in enumerate(plotted_strikes):
                                if s in max_strikes:
                                    highlight_idx = idx
                                    break
                            if highlight_idx is not None:
                                vals = fig.data[net_trace_idx].x if horizontal else fig.data[net_trace_idx].y
                                line_widths = [0] * len(vals)
                                line_widths[highlight_idx] = 5
                                fig.data[net_trace_idx].update(marker=dict(
                                    line=dict(width=line_widths, color=max_level_color)
                                ))
            else:  # 'Absolute' - default behaviour
                max_abs_val = 0
                max_trace_idx = -1
                max_bar_idx = -1
                for i, trace in enumerate(fig.data):
                    if trace.type == 'bar':
                        vals = trace.x if horizontal else trace.y
                        if vals:
                            abs_vals = [abs(v) for v in vals]
                            if abs_vals:
                                local_max = max(abs_vals)
                                if local_max > max_abs_val:
                                    max_abs_val = local_max
                                    max_trace_idx = i
                                    max_bar_idx = abs_vals.index(local_max)
                if max_trace_idx != -1:
                    vals = fig.data[max_trace_idx].x if horizontal else fig.data[max_trace_idx].y
                    line_widths = [0] * len(vals)
                    line_widths[max_bar_idx] = 5
                    fig.data[max_trace_idx].update(marker=dict(
                        line=dict(width=line_widths, color=max_level_color)
                    ))
        except Exception as e:
            print(f"Error highlighting max level: {e}")

    return fig.to_json()

def create_volume_chart(call_volume, put_volume, use_itm=True, call_color='#00FF00', put_color='#FF0000', selected_expiries=None):
    base_title = '% Range Call vs Put Volume Ratio' if use_itm else 'Call vs Put Volume Ratio'
    title = build_chart_title_text(base_title, selected_expiries=selected_expiries)
    fig = go.Figure(data=[go.Pie(
        labels=['Calls', 'Puts'],
        values=[call_volume, put_volume],
        hole=0.3,
        marker_colors=[call_color, put_color]
    )])
    
    fig.update_layout(
        title=build_left_aligned_title(title, text_color='white'),
        showlegend=False,
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white'),
        height=500
    )
    
    return fig.to_json()

def create_options_volume_chart(calls, puts, S, strike_range=0.02, call_color='#00FF00', put_color='#FF0000', coloring_mode='Solid', show_calls=True, show_puts=True, show_net=True, selected_expiries=None, horizontal=False, highlight_max_level=False, max_level_color='#800080', max_level_mode='Absolute'):
    total_call_volume = calls['volume'].sum() if not calls.empty and 'volume' in calls.columns else 0
    total_put_volume = puts['volume'].sum() if not puts.empty and 'volume' in puts.columns else 0
    total_net_volume = total_call_volume - total_put_volume

    # Filter strikes within range
    min_strike = S * (1 - strike_range)
    max_strike = S * (1 + strike_range)
    
    calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
    puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()
    
    # Determine strike interval and aggregate by rounded strikes
    all_strikes = list(calls['strike']) + list(puts['strike'])
    if all_strikes:
        strike_interval = get_strike_interval(all_strikes)
        calls = aggregate_by_strike(calls, ['volume'], strike_interval)
        puts = aggregate_by_strike(puts, ['volume'], strike_interval)
    
    # Create figure
    fig = go.Figure()
    
    # Calculate max volume for normalization across all data
    max_volume = 1.0
    all_abs_vals = []
    if not calls.empty:
        all_abs_vals.extend(calls['volume'].abs().tolist())
    if not puts.empty:
        all_abs_vals.extend(puts['volume'].abs().tolist())
    if all_abs_vals:
        max_volume = max(all_abs_vals)
    if max_volume == 0:
        max_volume = 1.0
    
    # Add call volume bars
    if show_calls and not calls.empty:
        # Apply coloring mode
        call_colors = get_colors(call_color, calls['volume'], max_volume, coloring_mode)
            
        if horizontal:
            fig.add_trace(go.Bar(
                y=calls['strike'].tolist(),
                x=calls['volume'].tolist(),
                name='Call',
                marker_color=call_colors,
                text=calls['volume'].tolist(),
                textposition='auto',
                orientation='h',
                hovertemplate=build_hover_template('Call', [('Strike', '$%{y:.2f}'), ('Volume', '%{x:,.0f}')]),
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=calls['strike'].tolist(),
                y=calls['volume'].tolist(),
                name='Call',
                marker_color=call_colors,
                text=calls['volume'].tolist(),
                textposition='auto',
                hovertemplate=build_hover_template('Call', [('Strike', '$%{x:.2f}'), ('Volume', '%{y:,.0f}')]),
                marker_line_width=0
            ))
    
    # Add put volume bars (as negative values)
    if show_puts and not puts.empty:
        # Apply coloring mode
        put_colors = get_colors(put_color, puts['volume'], max_volume, coloring_mode)
            
        if horizontal:
            fig.add_trace(go.Bar(
                y=puts['strike'].tolist(),
                x=[-v for v in puts['volume'].tolist()],  # Make put volumes negative
                name='Put',
                marker_color=put_colors,
                text=puts['volume'].tolist(),
                textposition='auto',
                orientation='h',
                hovertemplate=build_hover_template('Put', [('Strike', '$%{y:.2f}'), ('Volume', '%{text}')]),
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=puts['strike'].tolist(),
                y=[-v for v in puts['volume'].tolist()],  # Make put volumes negative
                name='Put',
                marker_color=put_colors,
                text=puts['volume'].tolist(),
                textposition='auto',
                hovertemplate=build_hover_template('Put', [('Strike', '$%{x:.2f}'), ('Volume', '%{text}')]),
                marker_line_width=0
            ))
    
    # Add net volume bars if enabled
    if show_net and not (calls.empty and puts.empty):
        # Create net volume by combining calls and puts
        all_strikes_list = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        net_volume = []
        
        for strike in all_strikes_list:
            call_vol = calls[calls['strike'] == strike]['volume'].sum() if not calls.empty else 0
            put_vol = puts[puts['strike'] == strike]['volume'].sum() if not puts.empty else 0
            net_vol = call_vol - put_vol
            
            net_volume.append(net_vol)
        
        # Calculate max for net volume normalization
        max_net_volume = max(abs(min(net_volume)), abs(max(net_volume))) if net_volume else 1.0
        if max_net_volume == 0:
            max_net_volume = 1.0
        
        # Apply coloring mode for net values
        net_colors = get_net_colors(net_volume, max_net_volume, call_color, put_color, coloring_mode)
        
        if horizontal:
            fig.add_trace(go.Bar(
                y=all_strikes_list,
                x=net_volume,
                name='Net',
                base=0,  # overlay from zero instead of stacking on the Call bar
                width=strike_interval * 0.5,
                marker_color=net_colors,
                text=[f"{vol:,.0f}" for vol in net_volume],
                textposition='auto',
                orientation='h',
                hovertemplate=build_hover_template('Net', [('Strike', '$%{y:.2f}'), ('Volume', '%{text}')]),
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=all_strikes_list,
                y=net_volume,
                name='Net',
                base=0,  # overlay from zero instead of stacking on the Call bar
                width=strike_interval * 0.5,
                marker_color=net_colors,
                text=[f"{vol:,.0f}" for vol in net_volume],
                textposition='auto',
                hovertemplate=build_hover_template('Net', [('Strike', '$%{x:.2f}'), ('Volume', '%{text}')]),
                marker_line_width=0
            ))
    
    add_current_price_reference(fig, S, horizontal=horizontal, text_color='white')
    
    chart_title = build_bar_chart_title(
        'Options Volume by Strike',
        total_call_volume,
        total_put_volume,
        total_net_volume,
        call_color,
        put_color,
        selected_expiries=selected_expiries,
    )
    
    xaxis_config = dict(
        title='',
        title_font=dict(color='#CCCCCC'),
        tickfont=dict(color='#CCCCCC'),
        gridcolor='#333333',
        linecolor='#333333',
        showgrid=False,
        zeroline=True,
        zerolinecolor='#333333',
        automargin=True
    )
    
    yaxis_config = dict(
        title='',
        title_font=dict(color='#CCCCCC'),
        tickfont=dict(color='#CCCCCC'),
        gridcolor='#333333',
        linecolor='#333333',
        showgrid=False,
        zeroline=True,
        zerolinecolor='#333333'
    )
    
    if horizontal:
         yaxis_config.update(dict(
            range=[min_strike, max_strike],
            autorange=False
         ))
    else:
        xaxis_config.update(dict(
            range=[min_strike, max_strike],
            autorange=False,
            tickangle=45,
            tickformat='.0f',
            showticklabels=True,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor='#CCCCCC'
        ))

    # Update layout
    fig.update_layout(
        title=build_left_aligned_title(chart_title),
        annotations=list(fig.layout.annotations) + [
            build_bar_chart_totals_annotation(
                total_call_volume,
                total_put_volume,
                total_net_volume,
                call_color,
                put_color,
            )
        ],
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        barmode='relative',
        hovermode='y unified' if horizontal else 'x unified',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='#CCCCCC'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="right",
            x=1,
            font=dict(color='#CCCCCC'),
            bgcolor='#1E1E1E'
        ),
        bargap=0.1,
        bargroupgap=0.1,
        margin=dict(l=38, r=56 if horizontal else 42, t=52, b=28 if horizontal else 72),
        hoverlabel=dict(
            bgcolor='#1E1E1E',
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100,
        showlegend=False,
        height=500
    )

    ensure_bar_text_visibility(fig, horizontal=horizontal)
    
    # Add hover spikes
    fig.update_xaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    
    # Logic for Highlighting Max Level
    if highlight_max_level:
        try:
            if max_level_mode == 'Net':
                net_trace_idx = next((i for i, t in enumerate(fig.data) if t.type == 'bar' and t.name == 'Net'), None)
                if net_trace_idx is not None:
                    raw = fig.data[net_trace_idx].x if horizontal else fig.data[net_trace_idx].y
                    if raw:
                        vals = list(raw)
                        total_net = sum(vals)
                        if total_net >= 0:
                            max_bar_idx = vals.index(max(vals))
                        else:
                            max_bar_idx = vals.index(min(vals))
                        line_widths = [0] * len(vals)
                        line_widths[max_bar_idx] = 5
                        fig.data[net_trace_idx].update(marker=dict(
                            line=dict(width=line_widths, color=max_level_color)
                        ))
            else:
                max_abs_val = 0
                max_trace_idx = -1
                max_bar_idx = -1
                for i, trace in enumerate(fig.data):
                    if trace.type == 'bar':
                        vals = trace.x if horizontal else trace.y
                        if vals:
                            abs_vals = [abs(v) for v in vals]
                            if abs_vals:
                                local_max = max(abs_vals)
                                if local_max > max_abs_val:
                                    max_abs_val = local_max
                                    max_trace_idx = i
                                    max_bar_idx = abs_vals.index(local_max)
                if max_trace_idx != -1:
                    vals = fig.data[max_trace_idx].x if horizontal else fig.data[max_trace_idx].y
                    line_widths = [0] * len(vals)
                    line_widths[max_bar_idx] = 5
                    fig.data[max_trace_idx].update(marker=dict(
                        line=dict(width=line_widths, color=max_level_color)
                    ))
        except Exception as e:
            print(f"Error highlighting max level in options volume: {e}")

    return fig.to_json()



def aggregate_to_hourly(candles):
    """Aggregate sub-hourly candles to 1-hour candles aligned to ET hour boundaries."""
    tz = pytz.timezone('US/Eastern')
    hourly = {}
    for candle in candles:
        et = datetime.fromtimestamp(candle['datetime'] / 1000, tz)
        hour_key = et.replace(minute=0, second=0, microsecond=0)
        if hour_key not in hourly:
            hourly[hour_key] = []
        hourly[hour_key].append(candle)
    result = []
    for hour_key in sorted(hourly.keys()):
        group = hourly[hour_key]
        result.append({
            'datetime': group[0]['datetime'],
            'open': group[0]['open'],
            'high': max(c['high'] for c in group),
            'low': min(c['low'] for c in group),
            'close': group[-1]['close'],
            'volume': sum(c.get('volume', 0) for c in group)
        })
    return result

# Schwab API hard limit: minute-frequency data is only available for the last 10 trading days.
# Requesting a wider date range via startDate/endDate simply returns no extra candles.
SCHWAB_MAX_MINUTE_TRADING_DAYS = 10

# Realistic bars per trading day for each supported timeframe (390 min market session).
_BARS_PER_DAY = {1: 390, 5: 78, 10: 39, 15: 26, 30: 13, 60: 7}


def schwab_max_bars(timeframe):
    """Return the absolute maximum indicator-warmup bars the Schwab API can deliver."""
    bpd = _BARS_PER_DAY.get(int(timeframe), 78)
    return SCHWAB_MAX_MINUTE_TRADING_DAYS * bpd


def _price_history_params(timeframe, lookback_bars=220):
    """
    Return (period_type, period, calendar_span_days) for the Schwab price-history call.

    Schwab only serves up to SCHWAB_MAX_MINUTE_TRADING_DAYS of intraday data.
    We always request that full window so indicators have the most context possible,
    but we never ask for more than the API can deliver.
    """
    bpd = _BARS_PER_DAY.get(int(timeframe), 78)
    max_bars = SCHWAB_MAX_MINUTE_TRADING_DAYS * bpd

    # Clamp the caller's request to what the API can actually return
    effective_bars = max(bpd * 2, min(max_bars, int(lookback_bars)))
    trading_days = max(2, min(SCHWAB_MAX_MINUTE_TRADING_DAYS, math.ceil(effective_bars / bpd) + 1))
    # periodType=day only accepts these period values
    trading_days = next((p for p in (1, 2, 3, 4, 5, 10) if p >= trading_days), 10)

    # Add extra calendar days to absorb weekends and market holidays
    calendar_span = trading_days * 2 + 3
    return 'day', trading_days, calendar_span


def get_price_history(ticker, timeframe=1, lookback_bars=220):
    if ticker == "MARKET":
        ticker = "$SPX"
    elif ticker == "MARKET2":
        ticker = "SPY"
    try:
        # Get current time in EST
        est = datetime.now(pytz.timezone('US/Eastern'))
        current_date = est.date()

        # Schwab API only supports minute frequencies: 1, 5, 10, 15, 30.
        # For 60-min (hourly), fetch 30-min candles and aggregate after.
        api_frequency = 30 if timeframe == 60 else timeframe
        period_type, period, calendar_span_days = _price_history_params(timeframe, lookback_bars=lookback_bars)
        start_date = datetime.combine(current_date - timedelta(days=calendar_span_days), datetime.min.time())
        end_date = datetime.combine(current_date + timedelta(days=1), datetime.min.time())

        # Convert dates to milliseconds since epoch
        response = client.price_history(
            symbol=ticker,
            periodType=period_type,
            period=period,
            frequencyType="minute",
            frequency=api_frequency,
            startDate=int(start_date.timestamp() * 1000),
            endDate=int(end_date.timestamp() * 1000),
            needExtendedHoursData=True
        )
        
        if not response.ok:
            raise Exception(f"Failed to fetch price history: {response.status_code} {response.reason}")

        data = response.json()
        if not data or 'candles' not in data:
            raise Exception("Malformed price history data from Schwab API")

        # Filter for market hours
        candles = filter_market_hours(data['candles'])
        if not candles:
            raise Exception("No market-hour candles returned from Schwab API")

        # Sort candles by timestamp
        candles.sort(key=lambda x: x['datetime'])

        # Aggregate to hourly candles if requested
        if timeframe == 60:
            candles = aggregate_to_hourly(candles)

        # Get previous trading day's close
        prev_day_candles = []
        for candle in reversed(candles):
            candle_time = datetime.fromtimestamp(candle['datetime']/1000, pytz.timezone('US/Eastern'))
            if candle_time.date() < current_date:
                prev_day_candles.append(candle)
                if len(prev_day_candles) >= 30:  # Get at least 30 minutes of data
                    break

        # Get the last candle of the previous trading day (first one found iterating backwards)
        prev_day_close = prev_day_candles[0]['close'] if prev_day_candles else None

        return {
            'candles': candles,
            'prev_day_close': prev_day_close
        }
    except Exception as e:
        msg = f"[DEBUG] Error fetching price history: {e}"
        print(msg)
        raise Exception(msg)

def filter_market_hours(candles):
    """Filter candles to only include regular market hours (9:30 AM - 4:00 PM ET)"""
    filtered_candles = []
    for candle in candles:
        dt = datetime.fromtimestamp(candle['datetime']/1000)
        # Convert to Eastern Time
        et = dt.astimezone(pytz.timezone('US/Eastern'))
        # Check if it's a weekday and within market hours
        if et.weekday() < 5:  # 0-4 is Monday-Friday
            market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = et.replace(hour=16, minute=0, second=0, microsecond=0)
            # Candle times are bar-open times, so the 16:00 bar is after-hours trading
            if market_open <= et < market_close:
                filtered_candles.append(candle)
    return filtered_candles

def convert_to_heikin_ashi(candles):
    """Convert regular OHLC candles to Heikin-Ashi candles"""
    if not candles:
        return []
    
    ha_candles = []
    prev_ha_open = None
    prev_ha_close = None
    
    for candle in candles:
        # Calculate Heikin-Ashi values
        ha_close = (candle['open'] + candle['high'] + candle['low'] + candle['close']) / 4
        
        if prev_ha_open is None:
            # First candle: HA_Open = (Open + Close) / 2
            ha_open = (candle['open'] + candle['close']) / 2
        else:
            # Subsequent candles: HA_Open = (Previous HA_Open + Previous HA_Close) / 2
            ha_open = (prev_ha_open + prev_ha_close) / 2
        
        ha_high = max(candle['high'], ha_open, ha_close)
        ha_low = min(candle['low'], ha_open, ha_close)
        
        # Create new candle with Heikin-Ashi values
        ha_candle = {
            'datetime': candle['datetime'],
            'open': ha_open,
            'high': ha_high,
            'low': ha_low,
            'close': ha_close,
            'volume': candle['volume']
        }
        
        ha_candles.append(ha_candle)
        
        # Store values for next iteration
        prev_ha_open = ha_open
        prev_ha_close = ha_close
    
    return ha_candles



def bucket_timestamp_to_chart_interval(timestamp, chart_times):
    """Map a stored interval timestamp to the chart candle bucket that contains it."""
    if not chart_times:
        return None

    idx = bisect_left(chart_times, timestamp)
    if idx < len(chart_times) and chart_times[idx] == timestamp:
        return chart_times[idx]
    if idx <= 0:
        return None
    return chart_times[idx - 1]



def build_historical_levels_overlay(ticker, display_date, chart_times, latest_price, strike_range,
                                    selected_types, levels_count, call_color, put_color,
                                    selected_expiries=None, highlight_max_level=False, max_level_color='#800080',
                                    coloring_mode='Linear Intensity'):
    """Build historical intraday exposure overlays for the TradingView price chart."""
    if not ticker or not chart_times or not selected_types or not latest_price:
        return [], []

    normalized_types = []
    include_expected_move = False
    for level_type in selected_types:
        normalized = normalize_level_type(level_type)
        if normalized == 'Expected Move':
            include_expected_move = True
            continue
        if normalized in INTERVAL_LEVEL_VALUE_KEYS and normalized not in normalized_types:
            normalized_types.append(normalized)

    min_strike = latest_price * (1 - strike_range)
    max_strike = latest_price * (1 + strike_range)
    expiry_key = build_expiry_selection_key(selected_expiries)
    # Only read rows for the session actually displayed on the chart. Rows from an
    # earlier session would all bucket before the first candle and be dropped anyway.
    interval_rows = get_interval_data(ticker, display_date, expiry_key=expiry_key) if normalized_types else []
    session_rows = get_interval_session_data(ticker, display_date, expiry_key=expiry_key) if include_expected_move else []

    points_by_time = {}
    for row in interval_rows:
        (
            timestamp,
            _stored_price,
            strike,
            net_gamma,
            net_delta,
            net_vanna,
            net_charm,
            abs_gex_total,
            net_volume,
            net_speed,
            net_vomma,
            net_color,
        ) = row
        if strike < min_strike or strike > max_strike:
            continue

        # Samples are stored every minute; map each into the candle that contains it
        # (not the nearest candle) so e.g. 10:03 stays on the 10:00 5m bar.
        snapped_time = bucket_timestamp_to_chart_interval(timestamp, chart_times)
        if snapped_time is None:
            continue

        value_map = {
            'GEX': net_gamma,
            'AbsGEX': abs_gex_total,
            'DEX': net_delta,
            'VEX': net_vanna,
            'Charm': net_charm,
            'Volume': net_volume,
            'Speed': net_speed,
            'Vomma': net_vomma,
            'Color': net_color,
        }
        bucket = points_by_time.setdefault(snapped_time, {'time': snapped_time, 'by_type': {}})
        for level_type in normalized_types:
            value = value_map.get(level_type)
            if value is None or value == 0:
                continue
            # Rows are ordered by timestamp, so the latest sample per strike in the
            # bucket wins instead of one strike being repeated once per minute.
            bucket['by_type'].setdefault(level_type, {})[float(strike)] = float(value)

    selected_points = []
    max_abs_by_type = {}
    for bucket in points_by_time.values():
        snapped_time = bucket['time']
        for level_type, candidates in bucket['by_type'].items():
            top_levels = sorted(candidates.items(), key=lambda item: abs(item[1]), reverse=True)[:levels_count]
            for rank, (strike, value) in enumerate(top_levels, start=1):
                selected_points.append({
                    'time': snapped_time,
                    'price': strike,
                    'value': value,
                    'type': level_type,
                    'rank': rank,
                })
                max_abs_by_type[level_type] = max(max_abs_by_type.get(level_type, 0), abs(value))

    highlight_abs_by_bucket_type = {}
    if highlight_max_level:
        for point in selected_points:
            bucket_key = (point['time'], point['type'])
            highlight_abs_by_bucket_type[bucket_key] = max(
                highlight_abs_by_bucket_type.get(bucket_key, 0),
                abs(point['value'])
            )

    historical_points = []
    for point in selected_points:
        level_type = point['type']
        type_max_value = max_abs_by_type.get(level_type, 0) or 1.0
        normalized_value = min(1.0, abs(point['value']) / type_max_value)
        if coloring_mode == 'Solid':
            intensity = 0.95
        elif coloring_mode == 'Ranked Intensity':
            intensity = 0.15 + 0.85 * (normalized_value ** 3)
        else:
            intensity = 0.35 + 0.65 * normalized_value

        bucket_key = (point['time'], level_type)
        is_max = (
            highlight_max_level
            and highlight_abs_by_bucket_type.get(bucket_key, 0) > 0
            and abs(point['value']) == highlight_abs_by_bucket_type[bucket_key]
        )
        base_color = call_color if point['value'] >= 0 else put_color
        formatted_value = format_large_number(point['value'])
        side_label = 'Call' if point['value'] >= 0 else 'Put'
        historical_points.append({
            'time': point['time'],
            'price': round(point['price'], 4),
            'size': round(6 + (12 * normalized_value) + (2 if is_max else 0), 2),
            'color': hex_to_rgba(base_color, intensity),
            'border_color': max_level_color if is_max else base_color,
            'border_width': 2 if is_max else 1,
            'label': INTERVAL_LEVEL_DISPLAY_NAMES.get(level_type, level_type),
            'rank': point['rank'],
            'side': side_label,
            'value': formatted_value,
            'kind': 'exposure',
        })

    expected_move_by_time = {}
    for row in session_rows:
        timestamp, price, expected_move, expected_move_upper, expected_move_lower = row
        if expected_move is None or expected_move <= 0 or expected_move_upper is None or expected_move_lower is None:
            continue
        snapped_time = bucket_timestamp_to_chart_interval(timestamp, chart_times)
        if snapped_time is None:
            continue
        expected_move_by_time[snapped_time] = {
            'time': snapped_time,
            'price': round(price, 4),
            'move': round(expected_move, 4),
            'upper': round(expected_move_upper, 4),
            'lower': round(expected_move_lower, 4),
        }

    expected_move_rows = [
        expected_move_by_time[time_key]
        for time_key in sorted(expected_move_by_time.keys())
    ]
    max_expected_move = max((row['move'] for row in expected_move_rows), default=0) or 1.0
    for row in expected_move_rows:
        normalized_value = min(1.0, abs(row['move']) / max_expected_move)
        if coloring_mode == 'Solid':
            intensity = 0.95
        elif coloring_mode == 'Ranked Intensity':
            intensity = 0.15 + 0.85 * (normalized_value ** 3)
        else:
            intensity = 0.35 + 0.65 * normalized_value

        bubble_size = round(7 + (10 * normalized_value), 2)
        for direction, bubble_price in (('Upper', row['upper']), ('Lower', row['lower'])):
            historical_points.append({
                'time': row['time'],
                'price': bubble_price,
                'size': bubble_size,
                'color': hex_to_rgba('#036bfc', intensity),
                'border_color': '#81b4ff',
                'border_width': 1,
                'label': 'Expected Move',
                'rank': None,
                'side': direction,
                'value': f"${row['move']:.2f}",
                'kind': 'expected-move',
                'reference_price': f"${row['price']:.2f}",
            })

    overlap_groups = {}
    for point in historical_points:
        overlap_key = (
            point['time'],
            round(point['price'], 4),
            point.get('kind', 'exposure'),
        )
        overlap_groups.setdefault(overlap_key, []).append(point)

    for group_points in overlap_groups.values():
        total_points = len(group_points)
        for index, point in enumerate(group_points):
            point['overlap_slot'] = index
            point['overlap_count'] = total_points

    historical_points.sort(
        key=lambda point: (
            point['time'],
            0 if point.get('kind') == 'expected-move' else 1,
            point.get('rank') or 99,
            point['price'],
        )
    )

    historical_expected_moves = []

    return historical_points, historical_expected_moves


def prepare_price_chart_data(price_data, calls=None, puts=None, exposure_levels_types=[],
                              exposure_levels_count=3, call_color='#00FF00', put_color='#FF0000',
                              strike_range=0.1, use_heikin_ashi=False,
                              highlight_max_level=False, max_level_color='#800080',
                              coloring_mode='Linear Intensity', ticker=None, selected_expiries=None,
                              show_latest_level_lines=True, timeframe=1):
    """Return raw OHLCV + overlay data as JSON for TradingView Lightweight Charts rendering."""
    import json as _json

    # Handle backward compatibility
    if isinstance(exposure_levels_types, str):
        if exposure_levels_types == 'None':
            exposure_levels_types = []
        else:
            exposure_levels_types = [exposure_levels_types]

    if not price_data or 'candles' not in price_data or not price_data['candles']:
        return _json.dumps({'error': 'No price data'})

    candles = filter_market_hours(price_data['candles'])
    if not candles:
        return _json.dumps({'error': 'No market-hour candles'})

    est = pytz.timezone('US/Eastern')
    current_date = datetime.now(est).date()

    # Deduplicate and sort
    unique_candles = {}
    for c in candles:
        t = datetime.fromtimestamp(c['datetime'] / 1000, est)
        unique_candles[t] = c
    sorted_candles = [c for _, c in sorted(unique_candles.items(), key=lambda x: x[0])]

    # Filter to current day
    current_day_candles = [c for c in sorted_candles
                           if datetime.fromtimestamp(c['datetime'] / 1000, est).date() == current_date]
    display_date = current_date
    if not current_day_candles:
        most_recent_date = max(
            datetime.fromtimestamp(c['datetime'] / 1000, est).date() for c in sorted_candles)
        display_date = most_recent_date
        current_day_candles = [c for c in sorted_candles
                               if datetime.fromtimestamp(c['datetime'] / 1000, est).date() == most_recent_date]

    # Apply Heikin-Ashi using all candles as seed, then slice to current day
    if use_heikin_ashi:
        all_ha = convert_to_heikin_ashi(sorted_candles)
        day_start_idx = len(sorted_candles) - len(current_day_candles)
        display_candles = all_ha[day_start_idx:]
    else:
        display_candles = current_day_candles

    # Previous day close
    previous_day_close = None
    for c in reversed(sorted_candles):
        t = datetime.fromtimestamp(c['datetime'] / 1000, est)
        if t.date() < current_date:
            previous_day_close = c['close']
            break

    # Build Lightweight Charts candle data (time in seconds UTC)
    lc_candles = []
    lc_volume = []
    for i, c in enumerate(display_candles):
        ts = int(c['datetime'] / 1000)
        lc_candles.append({'time': ts, 'open': c['open'], 'high': c['high'],
                           'low': c['low'], 'close': c['close']})
        is_up = c['close'] >= c['open'] if i == 0 else c['close'] >= display_candles[i - 1]['close']
        lc_volume.append({'time': ts, 'value': c['volume'],
                          'color': call_color if is_up else put_color})

    # Multi-day raw candles for indicator warmup (FBB/SMA200 need up to 200 prior bars)
    lc_indicator_candles = [
        {'time': int(c['datetime'] / 1000), 'open': c['open'], 'high': c['high'],
         'low': c['low'], 'close': c['close'], 'volume': c.get('volume', 0)}
        for c in sorted_candles
    ]
    current_day_start_time = int(current_day_candles[0]['datetime'] / 1000) if current_day_candles else 0

    # Use the real last trade, not the Heikin-Ashi averaged close, for strike filtering,
    # expected move and the stored interval price.
    current_price = current_day_candles[-1]['close'] if current_day_candles else 0
    last_candle = display_candles[-1] if display_candles else None
    last_candle_up = (last_candle['close'] >= last_candle['open']) if last_candle else True

    display_date_key = display_date.strftime('%Y-%m-%d') if hasattr(display_date, 'strftime') else str(display_date)
    requested_level_types = [normalize_level_type(level_type) for level_type in exposure_levels_types if level_type]
    wants_historical_levels = any(level_type in INTERVAL_LEVEL_VALUE_KEYS for level_type in requested_level_types)
    wants_expected_move_history = 'Expected Move' in requested_level_types
    expiry_key = build_expiry_selection_key(selected_expiries)

    if ticker and current_price and calls is not None and puts is not None and (wants_historical_levels or wants_expected_move_history):
        needs_interval_seed = wants_historical_levels and not get_interval_data(ticker, display_date_key, expiry_key=expiry_key)
        needs_session_seed = wants_expected_move_history and not get_interval_session_data(ticker, display_date_key, expiry_key=expiry_key)
        if needs_interval_seed or needs_session_seed:
            store_interval_data(ticker, current_price, strike_range, calls, puts, expiry_key=expiry_key)

    historical_exposure_levels, historical_expected_moves = build_historical_levels_overlay(
        ticker=ticker,
        display_date=display_date_key,
        chart_times=[c['time'] for c in lc_candles],
        latest_price=current_price,
        strike_range=strike_range,
        selected_types=exposure_levels_types,
        levels_count=exposure_levels_count,
        call_color=call_color,
        put_color=put_color,
        selected_expiries=selected_expiries,
        highlight_max_level=highlight_max_level,
        max_level_color=max_level_color,
        coloring_mode=coloring_mode,
    )
    # Compute exposure levels
    exposure_levels = []
    expected_moves = []

    if exposure_levels_types and calls is not None and puts is not None:
        min_strike = current_price * (1 - strike_range)
        max_strike = current_price * (1 + strike_range)
        range_calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
        range_puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

        dash_map = ['dashed', 'dotted', 'large_dashed', 'dotted', 'dashed']
        all_top_levels = []

        for i, etype in enumerate(exposure_levels_types):
            if etype.lower() == 'expected move':
                expected_move_snapshot = calculate_expected_move_snapshot(calls, puts, current_price)
                if expected_move_snapshot:
                    expected_moves.append({
                        'upper': round(expected_move_snapshot['upper'], 2),
                        'lower': round(expected_move_snapshot['lower'], 2)
                    })
                continue

            col_name = etype
            if etype in ('Vanna', 'VEX'):
                col_name = 'VEX'
            if etype == 'AbsGEX':
                col_name = 'GEX'
            if etype == 'Volume':
                col_name = 'volume'

            if col_name in range_calls.columns and col_name in range_puts.columns:
                call_ex = range_calls.groupby('strike')[col_name].sum().to_dict() if not range_calls.empty else {}
                put_ex = range_puts.groupby('strike')[col_name].sum().to_dict() if not range_puts.empty else {}
                all_strikes_set = set(call_ex.keys()) | set(put_ex.keys())
                levels = {}
                for strike in all_strikes_set:
                    c_val = call_ex.get(strike, 0)
                    p_val = put_ex.get(strike, 0)
                    if etype == 'GEX':
                        net_val = c_val - p_val
                    elif etype == 'AbsGEX':
                        net_val = abs(c_val) + abs(p_val)
                    elif etype == 'Volume':
                        net_val = c_val - p_val
                    else:
                        net_val = c_val + p_val
                    levels[strike] = net_val

                top = sorted(levels.items(), key=lambda x: abs(x[1]), reverse=True)[:exposure_levels_count]
                for strike, val in top:
                    all_top_levels.append((strike, val, etype, i))

        # Max per type for highlight
        max_abs_by_type = {}
        if highlight_max_level:
            for strike, val, etype, tidx in all_top_levels:
                if etype not in max_abs_by_type or abs(val) > max_abs_by_type[etype]:
                    max_abs_by_type[etype] = abs(val)

        type_max_vals = {}
        for strike, val, etype, tidx in all_top_levels:
            if etype not in type_max_vals or abs(val) > type_max_vals[etype]:
                type_max_vals[etype] = abs(val)

        for strike, val, etype, type_index in all_top_levels:
            type_max_val = type_max_vals.get(etype, 1) or 1
            is_max = (highlight_max_level
                      and max_abs_by_type.get(etype, 0) > 0
                      and abs(val) == max_abs_by_type[etype])

            if is_max:
                intensity = 1.0
            elif coloring_mode == 'Solid':
                intensity = 1.0
            elif coloring_mode == 'Ranked Intensity':
                intensity = 0.1 + 0.9 * ((abs(val) / type_max_val) ** 3)
            else:  # Linear Intensity (default)
                intensity = 0.3 + 0.7 * (abs(val) / type_max_val)

            color = max_level_color if is_max else (call_color if val >= 0 else put_color)
            line_width = 2 if is_max else 1

            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            rgba = f'rgba({r},{g},{b},{intensity:.2f})'

            display_name = etype
            if etype == 'VEX':
                display_name = 'Vanna'
            if etype == 'AbsGEX':
                display_name = 'Abs GEX'

            exposure_levels.append({
                'price': float(strike),
                'value': float(val),
                'type': display_name,
                'color': rgba,
                'line_width': line_width,
                'dash_style': dash_map[type_index % len(dash_map)],
                'is_max': is_max,
                'label': f"{display_name}: {format_large_number(val)}"
            })

    return _json.dumps({
        'candles': lc_candles,
        'volume': lc_volume,
        'previous_day_close': previous_day_close,
        'current_price': current_price,
        'call_color': call_color,
        'put_color': put_color,
        'use_heikin_ashi': use_heikin_ashi,
        'last_candle_up': last_candle_up,
        'show_latest_level_lines': bool(show_latest_level_lines),
        'exposure_levels': exposure_levels,
        'expected_moves': expected_moves,
        'historical_exposure_levels': historical_exposure_levels,
        'historical_expected_moves': historical_expected_moves,
        'indicator_candles': lc_indicator_candles,
        'indicator_candles_count': len(lc_indicator_candles),
        'max_indicator_bars': schwab_max_bars(timeframe),
        'current_day_start_time': current_day_start_time,
    })


OPTIONS_CHAIN_CSS = '''
<style>
.oc-wrap { --oc-em: #e8cf86; --oc-em-edge: rgba(232, 207, 134, 0.4); background: var(--chart-bg, #1E1E1E); color: var(--text-primary, #eef2f7); height: 100%; display: flex; flex-direction: column; overflow: hidden; border-radius: 10px; font-variant-numeric: tabular-nums; }
.oc-head { padding: 10px 12px 0; display: flex; flex-direction: column; gap: 8px; }
.oc-tabs { display: flex; gap: 6px; overflow-x: auto; scrollbar-width: none; padding: 1px 0; }
.oc-tabs::-webkit-scrollbar { display: none; }
.oc-tab { flex: 0 0 auto; background: transparent; border: 1px solid var(--border-color, #444); color: var(--text-secondary, #ccc); border-radius: 12px; padding: 3px 10px; font: inherit; font-size: 11px; cursor: pointer; }
.oc-tab:first-child { margin-left: auto; }
.oc-tab:last-child { margin-right: auto; }
.oc-tab small { font-size: 9.5px; color: var(--text-muted, #888); margin-left: 5px; }
.oc-tab:hover { color: var(--text-primary, #fff); border-color: var(--text-muted, #888); }
.oc-tab.active { background: var(--accent-color, #5ab0ff); border-color: var(--accent-color, #5ab0ff); color: var(--chart-bg, #111); font-weight: 600; }
.oc-tab.active small { color: inherit; opacity: 0.75; }
.oc-section { flex: 1; display: flex; flex-direction: column; min-height: 0; }
.oc-section[hidden] { display: none; }
.oc-section-head { padding: 7px 12px 8px; display: flex; flex-direction: column; gap: 7px; border-bottom: 1px solid var(--border-color, #333); }
.oc-title { text-align: center; font-size: 14px; font-weight: 600; color: var(--text-secondary, #ccc); letter-spacing: 0.3px; }
.oc-title .oc-sub { font-size: 11px; font-weight: 400; color: var(--text-muted, #888); margin-left: 6px; }
.oc-stats { display: flex; justify-content: center; flex-wrap: wrap; gap: 6px 16px; font-size: 11px; color: var(--text-secondary, #ccc); }
.oc-stats em { font-style: normal; color: var(--text-muted, #888); margin-right: 4px; text-transform: uppercase; font-size: 9.5px; letter-spacing: 0.6px; }
.oc-split-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted, #888); }
.oc-split-labels b { font-weight: 600; }
.oc-split-labels .c b { color: var(--oc-call); }
.oc-split-labels .p b { color: var(--oc-put); }
.oc-split { display: flex; height: 4px; border-radius: 2px; overflow: hidden; background: var(--grid-color, #2A2A2A); margin-top: 3px; }
.oc-split .c { background: var(--oc-call); opacity: 0.85; }
.oc-split .p { background: var(--oc-put); opacity: 0.85; flex: 1; }
.oc-scroll { flex: 1; overflow: auto; min-height: 0; scrollbar-width: thin; scrollbar-color: var(--border-color, #444) transparent; }
.oc-scroll::-webkit-scrollbar { width: 8px; height: 8px; }
.oc-scroll::-webkit-scrollbar-thumb { background: var(--border-color, #444); border-radius: 4px; }
.oc-scroll::-webkit-scrollbar-track { background: transparent; }
.oc-table { width: 100%; border-collapse: separate; border-spacing: 0; table-layout: fixed; font-size: 11px; }
.oc-table thead { position: sticky; top: 0; z-index: 5; }
.oc-table thead th { background: var(--panel-bg-alt, #2D2D2D); font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.oc-table .oc-group th { padding: 5px 6px 3px; font-size: 10px; letter-spacing: 1.4px; text-transform: uppercase; }
.oc-table .oc-group .c { color: var(--oc-call); text-align: center; border-bottom: 2px solid var(--oc-call); }
.oc-table .oc-group .p { color: var(--oc-put); text-align: center; border-bottom: 2px solid var(--oc-put); }
.oc-table .oc-group .k { color: var(--text-muted, #888); text-align: center; border-bottom: 2px solid transparent; font-size: 9.5px; letter-spacing: 0.6px; }
.oc-table .oc-cols th { padding: 4px 6px; font-size: 9.5px; color: var(--text-muted, #888); text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer; user-select: none; border-bottom: 1px solid var(--border-color, #333); }
.oc-table .oc-cols th:hover { color: var(--text-primary, #fff); }
.oc-table .oc-cols th.c { text-align: right; }
.oc-table .oc-cols th.p { text-align: left; }
.oc-table .oc-cols th.k { text-align: center; }
.oc-table th .oc-sort { display: none; font-size: 8px; margin-left: 3px; opacity: 0.5; }
.oc-table th:hover .oc-sort { display: inline; }
.oc-table th .oc-sort.active { display: inline; opacity: 1; color: var(--accent-color, #5ab0ff); }
.oc-table td { padding: 4px 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid var(--grid-color, #2A2A2A); background-repeat: no-repeat; }
.oc-table td.c { text-align: right; }
.oc-table td.p { text-align: left; }
.oc-table td.c.itm { background-color: var(--oc-call-itm); }
.oc-table td.p.itm { background-color: var(--oc-put-itm); }
.oc-table td.c.bar { background-image: linear-gradient(to left, var(--oc-call-bar) var(--bar), transparent var(--bar)); }
.oc-table td.p.bar { background-image: linear-gradient(to right, var(--oc-put-bar) var(--bar), transparent var(--bar)); }
.oc-table td.peak { font-weight: 700; }
.oc-table td.c.peak { color: var(--oc-call); }
.oc-table td.p.peak { color: var(--oc-put); }
.oc-table td.zero { color: var(--text-muted, #888); opacity: 0.6; }
.oc-table td.iv { color: var(--text-secondary, #ccc); }
.oc-table td.k { text-align: center; font-weight: 700; color: var(--text-primary, #fff); background: var(--panel-bg-alt, #2D2D2D); border-left: 1px solid var(--border-color, #333); border-right: 1px solid var(--border-color, #333); }
.oc-table td.k.atm { color: var(--accent-color, #5ab0ff); box-shadow: inset 3px 0 0 var(--accent-color, #5ab0ff), inset -3px 0 0 var(--accent-color, #5ab0ff); }
.oc-table tbody tr:not(.oc-spot):hover td { box-shadow: inset 0 0 0 999px rgba(255, 255, 255, 0.05); }
.oc-table tbody tr:not(.oc-spot):hover td.k.atm { box-shadow: inset 3px 0 0 var(--accent-color, #5ab0ff), inset -3px 0 0 var(--accent-color, #5ab0ff), inset 0 0 0 999px rgba(255, 255, 255, 0.05); }
.oc-table tr.oc-marker td { padding: 0; height: 18px; border: 0; position: relative; }
.oc-table tr.oc-marker .oc-pill { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); padding: 1px 9px; font-size: 10px; font-weight: 700; line-height: 14px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4); white-space: nowrap; }
.oc-table tr.oc-spot td { background: linear-gradient(var(--accent-color, #5ab0ff), var(--accent-color, #5ab0ff)) center / 100% 1px no-repeat; }
.oc-table tr.oc-spot .oc-pill { color: var(--chart-bg, #111); background: var(--accent-color, #5ab0ff); }
.oc-table tr.oc-marker td { overflow: visible; }
.oc-table tr.oc-marker .oc-pill { z-index: 2; }
.oc-table td.k.in-em, .oc-table tr.oc-marker td.in-em { border-left: 2px dashed var(--oc-em-edge); border-right: 2px dashed var(--oc-em-edge); }
.oc-table .oc-group th.k.em { color: var(--oc-em); font-weight: 700; letter-spacing: 0.3px; }
.oc-stats .oc-em-stat { color: var(--oc-em); font-weight: 600; }
.oc-stats .oc-em-stat small { font-size: 10px; font-weight: 400; color: var(--text-muted, #888); margin-left: 4px; }
.oc-empty { padding: 30px; text-align: center; color: var(--text-muted, #888); font-size: 12px; }
</style>
'''


def _build_chain_section(calls, puts, S, strike_range):
    """Render one expiry's chain: its stats header and the calls | strike | puts table."""
    expected_move = calculate_expected_move_snapshot(calls, puts, S)

    min_strike = S * (1 - strike_range)
    max_strike = S * (1 + strike_range)

    calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

    def collect(df):
        quotes = {}
        for _, row in df.iterrows():
            strike = float(row['strike'])
            quote = {
                'bid': float(row['bid']),
                'ask': float(row['ask']),
                'last': float(row['lastPrice']),
                'volume': int(row['volume']),
                'openInterest': int(row['openInterest']),
                'iv': float(row['impliedVolatility']),
            }
            existing = quotes.get(strike)
            if existing is None:
                quotes[strike] = quote
            else:
                # Same strike from another root (e.g. SPX + SPXW): sum size, keep the busier quote
                volume = existing['volume'] + quote['volume']
                oi = existing['openInterest'] + quote['openInterest']
                if quote['volume'] > existing['volume']:
                    existing.update(quote)
                existing['volume'] = volume
                existing['openInterest'] = oi
        return quotes

    call_quotes = collect(calls)
    put_quotes = collect(puts)
    strikes = sorted(set(call_quotes) | set(put_quotes))

    def fmt_strike(k):
        return f'{k:,.2f}'.rstrip('0').rstrip('.')

    def fmt_count(n):
        if n >= 1_000_000:
            return f'{n / 1_000_000:.1f}M'
        if n >= 100_000:
            return f'{n / 1_000:.0f}K'
        return f'{n:,}'

    total_call_vol = sum(q['volume'] for q in call_quotes.values())
    total_put_vol = sum(q['volume'] for q in put_quotes.values())
    total_call_oi = sum(q['openInterest'] for q in call_quotes.values())
    total_put_oi = sum(q['openInterest'] for q in put_quotes.values())
    pc_vol = f'{total_put_vol / total_call_vol:.2f}' if total_call_vol else '—'
    pc_oi = f'{total_put_oi / total_call_oi:.2f}' if total_call_oi else '—'
    total_vol = total_call_vol + total_put_vol
    call_share = (total_call_vol / total_vol * 100) if total_vol else 50

    all_quotes = list(call_quotes.values()) + list(put_quotes.values())
    max_oi = max((q['openInterest'] for q in all_quotes), default=0) or 1
    max_vol = max((q['volume'] for q in all_quotes), default=0) or 1
    peak_call_oi = max((q['openInterest'] for q in call_quotes.values()), default=0)
    peak_put_oi = max((q['openInterest'] for q in put_quotes.values()), default=0)
    peak_call_vol = max((q['volume'] for q in call_quotes.values()), default=0)
    peak_put_vol = max((q['volume'] for q in put_quotes.values()), default=0)
    atm_strike = min(strikes, key=lambda k: abs(k - S)) if strikes else None

    # Column order mirrors around the strike; indices feed sortTable()
    call_cols = [('OI', 'openInterest'), ('Vol', 'volume'), ('IV', 'iv'), ('Last', 'last'), ('Bid', 'bid'), ('Ask', 'ask')]
    put_cols = [('Bid', 'bid'), ('Ask', 'ask'), ('Last', 'last'), ('IV', 'iv'), ('Vol', 'volume'), ('OI', 'openInterest')]
    strike_col = len(call_cols)

    def header_cell(label, idx, side):
        return (f'<th class="{side}" data-col="{idx}" onclick="sortTable({idx}, \'number\')">'
                f'{label}<span class="oc-sort">▼▲</span></th>')

    col_headers = ''.join(header_cell(label, i, 'c') for i, (label, _) in enumerate(call_cols))
    col_headers += header_cell('Strike', strike_col, 'k')
    col_headers += ''.join(header_cell(label, strike_col + 1 + i, 'p') for i, (label, _) in enumerate(put_cols))

    def data_cell(quote, field, side, itm, peak_oi, peak_vol):
        classes = [side]
        if itm:
            classes.append('itm')
        style = ''
        if quote is None:
            return f'<td class="{" ".join(classes + ["zero"])}" data-sort="0">—</td>'
        value = quote[field]
        if field in ('openInterest', 'volume'):
            text = fmt_count(value)
            peak = peak_oi if field == 'openInterest' else peak_vol
            scale = max_oi if field == 'openInterest' else max_vol
            if value > 0:
                classes.append('bar')
                style = f' style="--bar: {value / scale * 100:.1f}%"'
                if value == peak:
                    classes.append('peak')
            else:
                classes.append('zero')
        elif field == 'iv':
            text = f'{value * 100:.1f}%'
            classes.append('iv')
        else:
            if value > 0:
                text = f'{value:.2f}'
            else:
                text = '—'
                classes.append('zero')
        return f'<td class="{" ".join(classes)}" data-sort="{value}"{style}>{text}</td>'

    # Strikes inside the expected move get dashed gold edges on the strike column
    em_lower = em_upper = None
    if expected_move:
        em_lower, em_upper = expected_move['lower'], expected_move['upper']

    def in_em(price):
        return em_lower is not None and em_lower <= price <= em_upper

    # Divider row for the spot price, placed between the strikes it falls between. The strike
    # cell is kept separate so the expected-move edges run unbroken through it.
    def marker_row(price, cls, label):
        em_cls = ' in-em' if in_em(price) else ''
        return (price, f'<tr class="oc-marker {cls}" data-price="{price}">'
                       f'<td colspan="{len(call_cols)}"></td>'
                       f'<td class="mk{em_cls}"><span class="oc-pill">{label}</span></td>'
                       f'<td colspan="{len(put_cols)}"></td></tr>')

    markers = [marker_row(S, 'oc-spot', f'{S:,.2f}')]

    rows = []
    for strike in strikes:
        while markers and markers[0][0] < strike:
            rows.append(markers.pop(0)[1])
        call_q = call_quotes.get(strike)
        put_q = put_quotes.get(strike)
        cells = [data_cell(call_q, field, 'c', strike < S, peak_call_oi, peak_call_vol) for _, field in call_cols]
        atm_cls = ' atm' if strike == atm_strike else ''
        em_cls = ' in-em' if in_em(strike) else ''
        cells.append(f'<td class="k{atm_cls}{em_cls}" data-sort="{strike}">{fmt_strike(strike)}</td>')
        cells += [data_cell(put_q, field, 'p', strike > S, peak_put_oi, peak_put_vol) for _, field in put_cols]
        rows.append(f'<tr>{"".join(cells)}</tr>')
    if strikes:
        rows.extend(html for _, html in markers)

    if expected_move:
        em_move = expected_move['move']
        em_header = f'<th class="k em" title="Expected move (ATM straddle)">±{em_move:,.2f}</th>'
        em_stat = (f'<span class="oc-em-stat"><em>Exp Move</em>±{em_move:,.2f}'
                   f'<small>({em_move / S * 100:.2f}%)</small>'
                   f'<small>{em_lower:,.2f} – {em_upper:,.2f}</small></span>')
    else:
        em_header = f'<th class="k">{S:,.2f}</th>'
        em_stat = ''

    if strikes:
        colgroup = '<col>' * len(call_cols) + '<col style="width: 11%">' + '<col>' * len(put_cols)
        body = f'''
        <div class="oc-scroll">
            <table class="oc-table">
                <colgroup>{colgroup}</colgroup>
                <thead>
                    <tr class="oc-group"><th class="c" colspan="{len(call_cols)}">Calls</th>{em_header}<th class="p" colspan="{len(put_cols)}">Puts</th></tr>
                    <tr class="oc-cols">{col_headers}</tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        </div>'''
    else:
        body = '<div class="oc-empty">No contracts in the selected strike range</div>'

    return f'''
            <div class="oc-section-head">
                <div class="oc-stats">
                    <span><em>Spot</em>{S:,.2f}</span>
                    <span><em>P/C Vol</em>{pc_vol}</span>
                    <span><em>P/C OI</em>{pc_oi}</span>
                    {em_stat}
                </div>
                <div>
                    <div class="oc-split-labels">
                        <span class="c">Calls <b>{fmt_count(total_call_vol)}</b> vol · {fmt_count(total_call_oi)} OI</span>
                        <span class="p">{fmt_count(total_put_oi)} OI · <b>{fmt_count(total_put_vol)}</b> vol Puts</span>
                    </div>
                    <div class="oc-split"><span class="c" style="width: {call_share:.1f}%"></span><span class="p"></span></div>
                </div>
            </div>
            {body}'''


def create_large_trades_table(calls, puts, S, strike_range, call_color='#00FF00', put_color='#FF0000', selected_expiries=None):
    """Create a T-style options chain: calls on the left, strikes down the middle, puts on the right.

    Each expiry gets its own chain; with several expiries selected, tabs switch between them.
    """
    # Split into one chain per expiry rather than mixing expiries in one table
    sections = []
    if 'expiration' in calls.columns and 'expiration' in puts.columns:
        expiries = sorted(set(calls['expiration'].dropna()) | set(puts['expiration'].dropna()))
        for expiry in expiries:
            sections.append((
                pd.Timestamp(expiry).date(),
                calls[calls['expiration'] == expiry],
                puts[puts['expiration'] == expiry],
            ))
    if not sections:
        sections = [(None, calls, puts)]

    today = datetime.now(pytz.timezone('US/Eastern')).date()
    tabs = []
    section_html = []
    for index, (expiry, exp_calls, exp_puts) in enumerate(sections):
        key = expiry.isoformat() if expiry else 'all'
        if expiry:
            tabs.append(f'<button type="button" class="oc-tab{" active" if index == 0 else ""}" data-expiry="{key}" '
                        f'onclick="selectChainExpiry(this)">{expiry.strftime("%b %d")}'
                        f'<small>{(expiry - today).days}d</small></button>')
        hidden = ' hidden' if index > 0 else ''
        section_html.append(f'<div class="oc-section" data-expiry="{key}"{hidden}>'
                            f'{_build_chain_section(exp_calls, exp_puts, S, strike_range)}</div>')

    if len(sections) > 1:
        head_extra = f'<div class="oc-tabs">{"".join(tabs)}</div>'
        subtitle = f'{len(sections)} expiries'
    else:
        head_extra = ''
        expiry = sections[0][0]
        subtitle = expiry.strftime('%b %d, %Y') if expiry else (str(selected_expiries[0]) if selected_expiries else '')

    wrap_style = (
        f'--oc-call: {call_color}; --oc-put: {put_color}; '
        f'--oc-call-itm: {hex_to_rgba(call_color, 0.07)}; --oc-put-itm: {hex_to_rgba(put_color, 0.07)}; '
        f'--oc-call-bar: {hex_to_rgba(call_color, 0.22)}; --oc-put-bar: {hex_to_rgba(put_color, 0.22)};'
    )

    return f'''{OPTIONS_CHAIN_CSS}
    <div class="oc-wrap" style="{wrap_style}">
        <div class="oc-head">
            <div class="oc-title">Options Chain<span class="oc-sub">{subtitle}</span></div>
            {head_extra}
        </div>
        {"".join(section_html)}
    </div>
    '''



def create_open_interest_chart(calls, puts, S, strike_range=0.02, call_color='#00FF00', put_color='#FF0000', coloring_mode='Solid', show_calls=True, show_puts=True, show_net=True, selected_expiries=None, horizontal=False, highlight_max_level=False, max_level_color='#800080', max_level_mode='Absolute'):
    total_call_oi = calls['openInterest'].sum() if not calls.empty and 'openInterest' in calls.columns else 0
    total_put_oi = puts['openInterest'].sum() if not puts.empty and 'openInterest' in puts.columns else 0
    total_net_oi = total_call_oi - total_put_oi

    # Filter strikes within range
    min_strike = S * (1 - strike_range)
    max_strike = S * (1 + strike_range)
    
    calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
    puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()
    
    # Determine strike interval and aggregate by rounded strikes
    all_strikes = list(calls['strike']) + list(puts['strike'])
    if all_strikes:
        strike_interval = get_strike_interval(all_strikes)
        calls = aggregate_by_strike(calls, ['openInterest'], strike_interval)
        puts = aggregate_by_strike(puts, ['openInterest'], strike_interval)
    
    # Create figure
    fig = go.Figure()
    
    # Calculate max OI for normalization across all data
    max_oi = 1.0
    all_abs_vals = []
    if not calls.empty:
        all_abs_vals.extend(calls['openInterest'].abs().tolist())
    if not puts.empty:
        all_abs_vals.extend(puts['openInterest'].abs().tolist())
    if all_abs_vals:
        max_oi = max(all_abs_vals)
    if max_oi == 0:
        max_oi = 1.0
    
    # Add call OI bars
    if show_calls and not calls.empty:
        call_colors = get_colors(call_color, calls['openInterest'], max_oi, coloring_mode)
            
        if horizontal:
            fig.add_trace(go.Bar(
                y=calls['strike'].tolist(),
                x=calls['openInterest'].tolist(),
                name='Call',
                marker_color=call_colors,
                text=[format_large_number(v) for v in calls['openInterest']],
                textposition='auto',
                orientation='h',
                hovertemplate=build_hover_template('Call', [('Strike', '$%{y:.2f}'), ('Open Interest', '%{text}')]),
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=calls['strike'].tolist(),
                y=calls['openInterest'].tolist(),
                name='Call',
                marker_color=call_colors,
                text=[format_large_number(v) for v in calls['openInterest']],
                textposition='auto',
                hovertemplate=build_hover_template('Call', [('Strike', '$%{x:.2f}'), ('Open Interest', '%{text}')]),
                marker_line_width=0
            ))
    
    # Add put OI bars (as negative values)
    if show_puts and not puts.empty:
        put_colors = get_colors(put_color, puts['openInterest'], max_oi, coloring_mode)
            
        if horizontal:
            fig.add_trace(go.Bar(
                y=puts['strike'].tolist(),
                x=[-v for v in puts['openInterest'].tolist()],
                name='Put',
                marker_color=put_colors,
                text=[format_large_number(v) for v in puts['openInterest']],
                textposition='auto',
                orientation='h',
                hovertemplate=build_hover_template('Put', [('Strike', '$%{y:.2f}'), ('Open Interest', '%{text}')]),
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=puts['strike'].tolist(),
                y=[-v for v in puts['openInterest'].tolist()],
                name='Put',
                marker_color=put_colors,
                text=[format_large_number(v) for v in puts['openInterest']],
                textposition='auto',
                hovertemplate=build_hover_template('Put', [('Strike', '$%{x:.2f}'), ('Open Interest', '%{text}')]),
                marker_line_width=0
            ))
    
    # Add net OI bars if enabled
    if show_net and not (calls.empty and puts.empty):
        all_strikes_list = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        net_oi = []
        
        for strike in all_strikes_list:
            call_val = calls[calls['strike'] == strike]['openInterest'].sum() if not calls.empty else 0
            put_val = puts[puts['strike'] == strike]['openInterest'].sum() if not puts.empty else 0
            net_oi.append(call_val - put_val)
        
        max_net_oi = max(abs(min(net_oi)), abs(max(net_oi))) if net_oi else 1.0
        if max_net_oi == 0:
            max_net_oi = 1.0
        
        net_colors = get_net_colors(net_oi, max_net_oi, call_color, put_color, coloring_mode)
        
        if horizontal:
            fig.add_trace(go.Bar(
                y=all_strikes_list,
                x=net_oi,
                name='Net',
                base=0,  # overlay from zero instead of stacking on the Call bar
                width=strike_interval * 0.5,
                marker_color=net_colors,
                text=[format_large_number(val) for val in net_oi],
                textposition='auto',
                orientation='h',
                hovertemplate=build_hover_template('Net', [('Strike', '$%{y:.2f}'), ('Open Interest', '%{text}')]),
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=all_strikes_list,
                y=net_oi,
                name='Net',
                base=0,  # overlay from zero instead of stacking on the Call bar
                width=strike_interval * 0.5,
                marker_color=net_colors,
                text=[format_large_number(val) for val in net_oi],
                textposition='auto',
                hovertemplate=build_hover_template('Net', [('Strike', '$%{x:.2f}'), ('Open Interest', '%{text}')]),
                marker_line_width=0
            ))
    
    add_current_price_reference(fig, S, horizontal=horizontal, text_color='white')
    
    chart_title = build_bar_chart_title(
        'Open Interest by Strike',
        total_call_oi,
        total_put_oi,
        total_net_oi,
        call_color,
        put_color,
        selected_expiries=selected_expiries,
    )
    
    xaxis_config = dict(
        title='',
        title_font=dict(color='#CCCCCC'),
        tickfont=dict(color='#CCCCCC'),
        gridcolor='#333333',
        linecolor='#333333',
        showgrid=False,
        zeroline=True,
        zerolinecolor='#333333',
        automargin=True
    )
    
    yaxis_config = dict(
        title='',
        title_font=dict(color='#CCCCCC'),
        tickfont=dict(color='#CCCCCC'),
        gridcolor='#333333',
        linecolor='#333333',
        showgrid=False,
        zeroline=True,
        zerolinecolor='#333333'
    )
    
    if horizontal:
         yaxis_config.update(dict(
            range=[min_strike, max_strike],
            autorange=False
         ))
    else:
        xaxis_config.update(dict(
            range=[min_strike, max_strike],
            autorange=False,
            tickangle=45,
            tickformat='.0f',
            showticklabels=True,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor='#CCCCCC'
        ))

    fig.update_layout(
        title=build_left_aligned_title(chart_title),
        annotations=list(fig.layout.annotations) + [
            build_bar_chart_totals_annotation(
                total_call_oi,
                total_put_oi,
                total_net_oi,
                call_color,
                put_color,
            )
        ],
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        barmode='relative',
        hovermode='y unified' if horizontal else 'x unified',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='#CCCCCC'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="right",
            x=1,
            font=dict(color='#CCCCCC'),
            bgcolor='#1E1E1E'
        ),
        bargap=0.1,
        bargroupgap=0.1,
        margin=dict(l=38, r=56 if horizontal else 42, t=52, b=28 if horizontal else 72),
        hoverlabel=dict(
            bgcolor='#1E1E1E',
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100,
        showlegend=False,
        height=500
    )

    ensure_bar_text_visibility(fig, horizontal=horizontal)
    
    fig.update_xaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    
    if highlight_max_level:
        try:
            if max_level_mode == 'Net':
                net_trace_idx = next((i for i, t in enumerate(fig.data) if t.type == 'bar' and t.name == 'Net'), None)
                if net_trace_idx is not None:
                    raw = fig.data[net_trace_idx].x if horizontal else fig.data[net_trace_idx].y
                    if raw:
                        vals = list(raw)
                        total_net = sum(vals)
                        if total_net >= 0:
                            max_bar_idx = vals.index(max(vals))
                        else:
                            max_bar_idx = vals.index(min(vals))
                        line_widths = [0] * len(vals)
                        line_widths[max_bar_idx] = 5
                        fig.data[net_trace_idx].update(marker=dict(
                            line=dict(width=line_widths, color=max_level_color)
                        ))
            else:
                max_abs_val = 0
                max_trace_idx = -1
                max_bar_idx = -1
                for i, trace in enumerate(fig.data):
                    if trace.type == 'bar':
                        vals = trace.x if horizontal else trace.y
                        if vals:
                            abs_vals = [abs(v) for v in vals]
                            if abs_vals:
                                local_max = max(abs_vals)
                                if local_max > max_abs_val:
                                    max_abs_val = local_max
                                    max_trace_idx = i
                                    max_bar_idx = abs_vals.index(local_max)
                if max_trace_idx != -1:
                    vals = fig.data[max_trace_idx].x if horizontal else fig.data[max_trace_idx].y
                    line_widths = [0] * len(vals)
                    line_widths[max_bar_idx] = 5
                    fig.data[max_trace_idx].update(marker=dict(
                        line=dict(width=line_widths, color=max_level_color)
                    ))
        except Exception as e:
            print(f"Error highlighting max level in open interest chart: {e}")

    return fig.to_json()


def create_centroid_chart(ticker, call_color='#00FF00', put_color='#FF0000', selected_expiries=None):
    """Create a chart showing call and put centroids over time with price line"""
    est = pytz.timezone('US/Eastern')
    current_time_est = datetime.now(est)

    def _empty_centroid_chart(title):
        fig = go.Figure()
        fig.update_layout(
            title=build_left_aligned_title(title),
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='#CCCCCC'),
            xaxis=dict(title='Time', title_font=dict(color='#CCCCCC'), tickfont=dict(color='#CCCCCC')),
            yaxis=dict(title='Price/Strike', title_font=dict(color='#CCCCCC'), tickfont=dict(color='#CCCCCC')),
            autosize=True
        )
        return fig.to_json()

    # Get centroid data; fall back to most recent session if today has no data
    showing_last_session = False
    expiry_key = build_expiry_selection_key(selected_expiries)
    centroid_data = get_centroid_data(ticker, expiry_key=expiry_key)

    if not centroid_data:
        last_date = get_last_session_date(ticker, 'centroid_data', expiry_key=expiry_key)
        if last_date:
            centroid_data = get_centroid_data(ticker, last_date, expiry_key=expiry_key)
            showing_last_session = True

    if not centroid_data:
        # Rolling selections such as 0DTE get a new expiry key every day, so the exact key
        # never has a prior session. Fall back to the most-sampled selection of the latest
        # session for this ticker.
        fallback = get_latest_session_expiry_key(ticker, 'centroid_data')
        if fallback:
            last_date, fallback_key = fallback
            centroid_data = get_centroid_data(ticker, last_date, expiry_key=fallback_key)
            showing_last_session = True

    if not centroid_data:
        if current_time_est.weekday() >= 5:
            return _empty_centroid_chart('Call vs Put Centroid Map (Market Closed - Weekend)')
        elif get_regular_session_bounds(current_time_est.date()) is None:
            return _empty_centroid_chart('Call vs Put Centroid Map (Market Closed - Holiday)')
        elif current_time_est.hour < 9 or (current_time_est.hour == 9 and current_time_est.minute < 30):
            return _empty_centroid_chart('Call vs Put Centroid Map (Pre-Market)')
        elif current_time_est.hour >= 16:
            return _empty_centroid_chart('Call vs Put Centroid Map (After Hours)')
        return _empty_centroid_chart('Call vs Put Centroid Map (No Data)')
    
    # Convert data to lists for plotting
    timestamps = []
    prices = []
    call_centroids = []
    put_centroids = []
    call_volumes = []
    put_volumes = []
    
    for row in centroid_data:
        timestamp, price, call_centroid, put_centroid, call_volume, put_volume = row
        # Plot in exchange time regardless of the machine's local timezone
        dt = datetime.fromtimestamp(timestamp, est).replace(tzinfo=None)
        timestamps.append(dt)
        prices.append(price)
        call_centroids.append(call_centroid if call_centroid > 0 else None)
        put_centroids.append(put_centroid if put_centroid > 0 else None)
        call_volumes.append(call_volume)
        put_volumes.append(put_volume)
    
    # Create figure
    fig = go.Figure()
    
    # Add call centroid line (top layer)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=call_centroids,
        mode='lines+markers',
        marker=dict(size=4),
        name='Call Centroid',
        line=dict(color=call_color, width=2),
        hovertemplate=build_time_hover_template('Call', [('Centroid', '$%{y:.2f}'), ('Volume', '%{customdata:,.0f}')]),
        customdata=call_volumes,
        connectgaps=False
    ))

    # Add put centroid line (middle layer)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=put_centroids,
        mode='lines+markers',
        marker=dict(size=4),
        name='Put Centroid',
        line=dict(color=put_color, width=2),
        hovertemplate=build_time_hover_template('Put', [('Centroid', '$%{y:.2f}'), ('Volume', '%{customdata:,.0f}')]),
        customdata=put_volumes,
        connectgaps=False
    ))

    call_trend_times, call_trend_values = calc_linear_trend_line(timestamps, call_centroids)
    if call_trend_times:
        fig.add_trace(go.Scatter(
            x=call_trend_times,
            y=call_trend_values,
            mode='lines',
            name='Call Trend',
            line=dict(color=call_color, width=2, dash='dash'),
            hovertemplate=build_time_hover_template('Call Trend', [('Trend', '$%{y:.2f}')]),
            connectgaps=False
        ))

    put_trend_times, put_trend_values = calc_linear_trend_line(timestamps, put_centroids)
    if put_trend_times:
        fig.add_trace(go.Scatter(
            x=put_trend_times,
            y=put_trend_values,
            mode='lines',
            name='Put Trend',
            line=dict(color=put_color, width=2, dash='dash'),
            hovertemplate=build_time_hover_template('Put Trend', [('Trend', '$%{y:.2f}')]),
            connectgaps=False
        ))

    # Add price line last (bottom layer)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=prices,
        mode='lines+markers',
        marker=dict(size=3),
        name='Price',
        line=dict(color='gold', width=2),
        hovertemplate=build_time_hover_template('Price', [('Price', '$%{y:.2f}')])
    ))
    
    # Add expiry info to title if multiple expiries are selected
    chart_title = build_chart_title_text(
        'Call vs Put Centroid Map',
        selected_expiries=selected_expiries,
        showing_last_session=showing_last_session,
    )
    
    # Update layout to match interval map style
    fig.update_layout(
        title=build_left_aligned_title(chart_title),
        xaxis=dict(
            title='Time',
            title_font=dict(color='#CCCCCC'),
            tickfont=dict(color='#CCCCCC'),
            gridcolor='#333333',
            linecolor='#333333',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#333333',
            tickformat='%H:%M',
            showticklabels=True,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor='#CCCCCC',
            automargin=True
        ),
        yaxis=dict(
            title='Price/Strike',
            title_font=dict(color='#CCCCCC'),
            tickfont=dict(color='#CCCCCC'),
            gridcolor='#333333',
            linecolor='#333333',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#333333'
        ),
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='#CCCCCC'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#CCCCCC'),
            bgcolor='#1E1E1E'
        ),
        margin=dict(l=50, r=50, t=50, b=20),
        showlegend=False,
        autosize=True,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1E1E1E',
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100
    )

    # Add hover spikes
    fig.update_xaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)

    return fig.to_json()

def infer_side(last, bid, ask):
    # If last is closer to ask, it's a buy; if closer to bid, it's a sell
    if abs(last - ask) < abs(last - bid):
        return 1  # buy
    elif abs(last - bid) < abs(last - ask):
        return -1  # sell
    else:
        return 0  # indeterminate

def _range_chain_for_dates(ticker, dates):
    """Return one chain response covering all `dates`, or None when a single range request
    would pull in many unselected expiries (then per-date requests are cheaper)."""
    if ticker in ('MARKET', 'MARKET2') or len(dates) < 2:
        return None
    from_date, to_date = min(dates), max(dates)
    try:
        listed = get_cached_option_expirations(ticker)
        in_range = sum(1 for d in listed if from_date <= d <= to_date)
    except Exception:
        return None
    if in_range > 2 * len(dates) + 2:
        return None
    try:
        return _fetch_chain_json(ticker, from_date, to_date)
    except Exception as e:
        print(f"Range chain fetch failed, falling back to per-expiry requests: {e}")
        return None


def fetch_options_for_multiple_dates(ticker, dates, exposure_metric="Open Interest", delta_adjusted: bool = False, calculate_in_notional: bool = True):
    """Fetch options for multiple expiration dates and combine them. Uses one Schwab
    request for the whole date range when practical, else concurrent per-date requests."""
    results = {}
    exceptions = {}
    range_chain = _range_chain_for_dates(ticker, dates)

    def _fetch(date):
        try:
            calls, puts = fetch_options_for_date(
                ticker, date,
                exposure_metric=exposure_metric,
                delta_adjusted=delta_adjusted,
                calculate_in_notional=calculate_in_notional,
                chain=range_chain,
            )
            results[date] = (calls, puts)
        except Exception as e:
            exceptions[date] = e
            print(f"Error fetching options for {date}: {e}")

    if range_chain is not None:
        for date in dates:
            _fetch(date)
    else:
        threads = [threading.Thread(target=_fetch, args=(date,), daemon=True) for date in dates]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    all_calls = []
    all_puts = []
    for date in dates:
        if date in results:
            calls, puts = results[date]
            if not calls.empty:
                all_calls.append(calls)
            if not puts.empty:
                all_puts.append(puts)

    combined_calls = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    combined_puts = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
    if combined_calls.empty and combined_puts.empty and exceptions:
        raise next(iter(exceptions.values()))
    if exceptions:
        # Flag partial chains so callers don't persist them as a full snapshot
        combined_calls.attrs['partial_fetch'] = True
        combined_puts.attrs['partial_fetch'] = True

    return combined_calls, combined_puts

# ── Background history collector ─────────────────────────────────────────────
# History (price-chart bubbles, centroid map) used to be recorded only while a browser
# was polling /update. The collector keeps recording the selections viewed today, once
# a minute during the regular session, whenever no browser has refreshed them recently.
# Disable with EZOPTIONS_BACKGROUND_COLLECT=0.
BACKGROUND_COLLECT_ENABLED = os.getenv('EZOPTIONS_BACKGROUND_COLLECT', '1') != '0'
BACKGROUND_COLLECT_INTERVAL_SEC = 60
BACKGROUND_COLLECT_MAX_SELECTIONS = 6
_collector_selections = {}  # (ticker, expiry_key) -> selection dict
_collector_lock = threading.Lock()
_collector_thread = None


def register_collector_selection(ticker, expiry_dates, strike_range, exposure_metric,
                                 delta_adjusted, calculate_in_notional, stored=False):
    """Remember a selection the dashboard is showing so the collector can keep recording it."""
    if not BACKGROUND_COLLECT_ENABLED:
        return
    expiry_key = build_expiry_selection_key(expiry_dates)
    now = time.time()
    with _collector_lock:
        entry = _collector_selections.get((ticker, expiry_key))
        if entry is None:
            entry = {'last_stored': 0.0}
            _collector_selections[(ticker, expiry_key)] = entry
        entry.update({
            'ticker': ticker,
            'expiry_dates': list(expiry_dates),
            'expiry_key': expiry_key,
            'strike_range': strike_range,
            'exposure_metric': exposure_metric,
            'delta_adjusted': delta_adjusted,
            'calculate_in_notional': calculate_in_notional,
            'last_seen': now,
            'seen_date': datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d'),
        })
        if stored:
            entry['last_stored'] = now
        # Keep only the most recently viewed selections
        if len(_collector_selections) > BACKGROUND_COLLECT_MAX_SELECTIONS:
            for stale_key, _ in sorted(_collector_selections.items(), key=lambda kv: kv[1]['last_seen'])[
                    :len(_collector_selections) - BACKGROUND_COLLECT_MAX_SELECTIONS]:
                _collector_selections.pop(stale_key, None)
    ensure_collector_started()


def _collect_selection(entry):
    ticker = entry['ticker']
    expiry_dates = entry['expiry_dates']
    if len(expiry_dates) == 1:
        calls, puts = fetch_options_for_date(
            ticker, expiry_dates[0], exposure_metric=entry['exposure_metric'],
            delta_adjusted=entry['delta_adjusted'], calculate_in_notional=entry['calculate_in_notional'])
    else:
        calls, puts = fetch_options_for_multiple_dates(
            ticker, expiry_dates, exposure_metric=entry['exposure_metric'],
            delta_adjusted=entry['delta_adjusted'], calculate_in_notional=entry['calculate_in_notional'])
    if calls.empty and puts.empty:
        return
    if calls.attrs.get('partial_fetch') or puts.attrs.get('partial_fetch'):
        return
    cached_spot = _chain_spot_cache.get(ticker)
    S = cached_spot[0] if cached_spot else get_current_price(ticker)
    if not S:
        return
    store_interval_data(ticker, S, entry['strike_range'], calls, puts, expiry_key=entry['expiry_key'])
    store_centroid_data(ticker, S, calls, puts, expiry_key=entry['expiry_key'])


def _collector_loop():
    while True:
        time.sleep(15)
        try:
            if client is None or not is_market_hours():
                continue
            today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
            now = time.time()
            with _collector_lock:
                due = [
                    dict(entry) for entry in _collector_selections.values()
                    if entry['seen_date'] == today
                    and now - entry['last_stored'] >= BACKGROUND_COLLECT_INTERVAL_SEC
                    # an expired expiry has no chain, so the snapshot would be partial;
                    # the page rolls the selection forward and re-registers it
                    and all(d >= today for d in entry['expiry_dates'])
                ]
            for entry in due:
                try:
                    _collect_selection(entry)
                except Exception as e:
                    print(f"[Collector] {entry['ticker']} {entry['expiry_key']}: {e}")
                with _collector_lock:
                    live = _collector_selections.get((entry['ticker'], entry['expiry_key']))
                    if live is not None:
                        live['last_stored'] = time.time()
        except Exception as e:
            print(f"[Collector] loop error: {e}")


def ensure_collector_started():
    global _collector_thread
    if not BACKGROUND_COLLECT_ENABLED:
        return
    with _collector_lock:
        if _collector_thread is not None and _collector_thread.is_alive():
            return
        _collector_thread = threading.Thread(target=_collector_loop, name='history-collector', daemon=True)
        _collector_thread.start()


@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>EzOptions - Schwab</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml">
    <link rel="shortcut icon" href="/favicon.svg" type="image/svg+xml">
    <link rel="apple-touch-icon" href="/favicon.svg">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <script src="https://cdn.plot.ly/plotly-1.58.5.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        *, *::before, *::after {
            box-sizing: border-box;
        }
        body {
            background-color: #1E1E1E;
            color: white;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            width: 100%;
            overflow-x: hidden;
        }
        /* Token Monitor */
        #token-monitor {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 4px;
        }
        .tm-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }
        .tm-ok   { background: #4CAF50; }
        .tm-warn { background: #ffb300; }
        .tm-err  { background: #ff4444; }
        .tm-neutral { background: #555; }
        .tm-stats {
            font-size: 11px;
            color: #888;
            font-family: monospace;
            letter-spacing: 0.02em;
        }
        .tm-stats span { color: #ccc; }
        .tm-btn-group {
            display: flex;
            gap: 5px;
        }
        .tm-btn {
            background: none;
            border: 1px solid #444;
            color: #777;
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 10px;
            cursor: pointer;
            transition: color 0.15s, border-color 0.15s;
        }
        .tm-btn:hover { color: #ccc; border-color: #777; }
        .tm-btn-del {
            border-color: #444;
            color: #666;
        }
        .tm-btn-del:hover { background: #2a1010; border-color: #883333; color: #cc4444; }
        .container {
            width: 95%;
            max-width: none;
            margin: 0 auto;
            padding: 15px;
        }
        .header {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-bottom: 20px;
            padding: 20px;
            background-color: #2D2D2D;
            border-radius: 10px;
            width: 100%;
            box-sizing: border-box;
        }
        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        .header-bottom {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
            padding-top: 15px;
            border-top: 1px solid #444;
        }
        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        .control-group {
            display: flex;
            gap: 10px;
            align-items: center;
            background-color: #333;
            padding: 8px 12px;
            border-radius: 6px;
        }
        .expiry-dropdown {
            position: relative;
            min-width: 150px;
        }
        .expiry-display {
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #444;
            background-color: #333;
            color: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }
        .expiry-display:hover {
            border-color: #555;
            background-color: #3a3a3a;
        }
        .expiry-display::after {
            content: '▼';
            font-size: 12px;
            color: #888;
        }
        .expiry-options {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background-color: #333;
            border: 1px solid #444;
            border-radius: 6px;
            border-top: none;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            max-height: 200px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
        }
        .expiry-options.open {
            display: block;
        }
        .expiry-option {
            padding: 8px 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background-color 0.2s;
        }
        .expiry-option:hover {
            background-color: #444;
        }
        .expiry-option input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: #00FF00;
        }
        .expiry-buttons {
            padding: 6px 8px;
            border-top: 1px solid #444;
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        .expiry-buttons button {
            padding: 4px 6px;
            font-size: 10px;
            border-radius: 4px;
            border: 1px solid #555;
            background-color: #444;
            color: white;
            cursor: pointer;
            flex: 1;
            min-width: 40px;
        }
        .expiry-buttons button:hover {
            background-color: #555;
        }
        .expiry-buttons .expiry-range-btns {
            display: flex;
            gap: 5px;
            width: 100%;
            flex-wrap: wrap;
        }
        .expiry-buttons .expiry-range-btns button {
            flex: 1;
            min-width: 38px;
            background-color: #3a3a5e;
            border-color: #5555aa;
        }
        .expiry-buttons .expiry-range-btns button:hover {
            background-color: #4a4a7e;
        }
        .levels-dropdown {
            position: relative;
            min-width: 150px;
        }
        .levels-display {
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #444;
            background-color: #333;
            color: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }
        .levels-display:hover {
            border-color: #555;
            background-color: #3a3a3a;
        }
        .levels-display::after {
            content: '▼';
            font-size: 12px;
            color: #888;
        }
        .levels-options {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background-color: #333;
            border: 1px solid #444;
            border-radius: 6px;
            border-top: none;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            max-height: 200px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
        }
        .levels-options.open {
            display: block;
        }
        .levels-option {
            padding: 8px 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background-color 0.2s;
        }
        .levels-option:hover {
            background-color: #444;
        }
        .levels-option input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: #00FF00;
        }
        .control-group label {
            white-space: nowrap;
        }
        input[type="text"], select {
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #444;
            background-color: #333;
            color: white;
            min-width: 120px;
        }

        input[type="range"] {
            width: 150px;
            height: 6px;
            background: #444;
            border-radius: 3px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #00FF00;
            border-radius: 50%;
            cursor: pointer;
        }
        .range-value {
            min-width: 40px;
            text-align: center;
        }
        .chart-selector {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
            width: 100%;
        }
        .chart-checkbox {
            display: flex;
            align-items: center;
            gap: 5px;
            background-color: #333;
            padding: 6px 10px;
            border-radius: 6px;
        }
        .chart-checkbox input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }
        .chart-checkbox label {
            cursor: pointer;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 5px;
            width: 100%;
            min-width: 0;
        }
        
        .price-chart-container {
            /* overridden below by TV chart styles */
        }
        
        
        .chart-container {
            padding: 0;
            height: var(--chart-height, 500px);
            width: 100%;
            min-width: 0;
            position: relative;
            background-color: #2D2D2D;
            border-radius: 10px;
            margin-bottom: 5px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-sizing: border-box;
        }
        
        .chart-container > div {
            flex: 1;
            width: 100%;
            height: 100%;
            min-width: 0;
            min-height: 0;
        }

        #chart-control-staging {
            display: none;
        }

        .heatmap-chart-shell {
            display: grid;
            grid-template-rows: auto minmax(0, 1fr);
            flex: 1 1 0;
            width: 100%;
            height: 100%;
            min-width: 0;
            min-height: 0;
            gap: 8px;
            padding: 8px;
            box-sizing: border-box;
            overflow: hidden;
        }

        .heatmap-chart-toolbar {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 8px;
            flex-wrap: wrap;
            padding: 0 0 0 82px;
            align-content: flex-start;
            grid-row: 1;
        }

        .heatmap-chart-toolbar-controls {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            flex-wrap: wrap;
            gap: 8px;
            width: 100%;
            align-content: flex-start;
        }

        .heatmap-chart-toolbar .control-group {
            margin: 0;
            min-width: 182px;
            padding: 4px 8px;
            gap: 6px;
            background: var(--panel-bg-alt);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            box-shadow: none;
            justify-content: space-between;
        }

        .heatmap-chart-toolbar .control-group label {
            color: var(--text-secondary);
            font-size: 11px;
            font-weight: 500;
        }

        .heatmap-chart-toolbar .control-group select {
            min-width: 104px;
            margin-left: auto;
            padding: 4px 8px;
            border-radius: 8px;
            border-color: var(--border-color);
            background: var(--panel-bg);
            color: var(--text-primary);
            font-size: 12px;
        }

        .heatmap-chart-toolbar .control-group select:hover {
            border-color: var(--accent-color);
        }

        .heatmap-chart-toolbar .control-group select:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px var(--accent-soft);
        }

        .heatmap-plot {
            grid-row: 2;
            width: 100%;
            height: 100%;
            min-width: 0;
            min-height: 0;
            border-radius: 12px;
            overflow: hidden;
            box-sizing: border-box;
            position: relative;
        }

        /* TradingView-style price chart overrides */
        .price-chart-container {
            background: #1a1a1a;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 5px;
            grid-column: 1 / -1;
        }
        #price-chart {
            padding: 0 !important;
            background-color: #1E1E1E !important;
            height: var(--price-chart-height, 680px) !important;
            border-radius: 0 0 0 0;
            overflow: hidden;
            position: relative;
            /* override .chart-container defaults that conflict */
            margin-bottom: 0 !important;
        }
        .tv-historical-overlay {
            position: absolute;
            inset: 0;
            z-index: 20;
            pointer-events: none;
            overflow: hidden;
        }
        .tv-historical-canvas {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .tv-historical-tooltip,
        .chart-hover-tooltip {
            position: absolute;
            z-index: 55;
            display: none;
            width: auto !important;
            height: auto !important;
            min-width: 0;
            max-width: min(300px, calc(100% - 16px));
            padding: 8px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            background: linear-gradient(180deg, rgba(30, 34, 41, 0.96), rgba(16, 18, 23, 0.98));
            color: #eef2f7;
            font-size: 10px;
            line-height: 1.25;
            pointer-events: none;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.38);
            backdrop-filter: blur(10px);
            flex: none !important;
            align-self: flex-start;
            overflow: hidden;
            white-space: normal;
        }
        .tv-historical-tooltip .tt-head,
        .chart-hover-tooltip .tt-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 6px;
        }
        .tv-historical-tooltip .tt-badge,
        .chart-hover-tooltip .tt-badge {
            padding: 2px 6px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: #c9d1db;
            font-size: 9px;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .tv-historical-tooltip .tt-time,
        .chart-hover-tooltip .tt-time {
            color: #8f9baa;
            font-size: 9px;
            margin-bottom: 0;
        }
        .tv-historical-tooltip .tt-list,
        .chart-hover-tooltip .tt-list {
            display: grid;
            gap: 4px;
        }
        .tv-historical-tooltip .tt-row,
        .chart-hover-tooltip .tt-row {
            display: flex;
            align-items: center;
            gap: 6px;
            min-width: 0;
        }
        .tv-historical-tooltip .tt-dot,
        .chart-hover-tooltip .tt-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.12);
            flex: 0 0 auto;
        }
        .tv-historical-tooltip .tt-main,
        .chart-hover-tooltip .tt-main {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 8px;
            min-width: 0;
            width: 100%;
        }
        .tv-historical-tooltip .tt-name,
        .chart-hover-tooltip .tt-name {
            color: #f4f7fb;
            font-weight: 600;
            flex-shrink: 0;
            white-space: nowrap;
        }
        .tv-historical-tooltip .tt-value,
        .chart-hover-tooltip .tt-value {
            color: #9fb0c4;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            flex: 0 1 auto;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: right;
        }
        .tv-historical-tooltip .tt-more,
        .chart-hover-tooltip .tt-more {
            color: #8190a3;
            margin-top: 4px;
            padding-top: 4px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            font-size: 9px;
        }
        .chart-container .hoverlayer {
            opacity: 0 !important;
            pointer-events: none !important;
        }
        .tv-chart-title {
            display: inline-block;
            color: #CCCCCC;
            font-size: 13px;
            font-weight: bold;
            padding: 2px 8px;
            pointer-events: none;
        }
        /* Chart toolbar — sits ABOVE the canvas, normal document flow */
        .tv-toolbar-container {
            background: #1a1a1a;
            border-bottom: 1px solid #333;
            border-radius: 10px 10px 0 0;
            padding: 4px 8px;
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            align-items: center;
        }
        .tv-toolbar-sep {
            width: 1px;
            height: 20px;
            background: #444;
            margin: 0 2px;
        }
        .tv-tb-btn {
            background: #2a2a2a;
            border: 1px solid #444;
            color: #ccc;
            border-radius: 4px;
            padding: 3px 7px;
            font-size: 11px;
            cursor: pointer;
            white-space: nowrap;
            transition: background 0.15s;
            user-select: none;
        }
        .tv-tb-btn:hover  { background: #3a3a3a; color: #fff; }
        .tv-tb-btn.active { background: #1a5fac; border-color: #4b90e2; color: #fff; }
        .tv-tb-btn.danger { background: #5c1a1a; border-color: #c0392b; color: #f88; }
        .tv-indicator-picker {
            position: relative;
        }
        .tv-indicator-summary {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            list-style: none;
        }
        .tv-indicator-summary::-webkit-details-marker {
            display: none;
        }
        .tv-indicator-summary::marker {
            content: '';
        }
        .tv-indicator-badge {
            display: none;
            min-width: 18px;
            padding: 1px 6px;
            border-radius: 999px;
            background: #3b4758;
            color: #eef2f7;
            font-size: 10px;
            line-height: 1.3;
            text-align: center;
        }
        .tv-indicator-menu {
            position: absolute;
            top: calc(100% + 6px);
            left: 0;
            width: min(320px, calc(100vw - 32px));
            padding: 8px;
            border: 1px solid #444;
            border-radius: 10px;
            background: #1c1c1c;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
            z-index: 130;
        }
        .tv-indicator-search {
            width: 100%;
            border: 1px solid #444;
            border-radius: 8px;
            background: #121212;
            color: #eee;
            padding: 7px 10px;
            font-size: 12px;
            outline: none;
        }
        .tv-indicator-search:focus {
            border-color: #4b90e2;
        }
        .tv-indicator-options {
            display: grid;
            gap: 4px;
            margin-top: 8px;
            max-height: 240px;
            overflow: auto;
        }
        .tv-indicator-option {
            width: 100%;
            border: 1px solid #333;
            border-radius: 8px;
            background: #242424;
            color: #ccc;
            padding: 7px 9px;
            text-align: left;
            cursor: pointer;
            display: grid;
            gap: 2px;
        }
        .tv-indicator-option:hover {
            background: #2f2f2f;
            border-color: #4b90e2;
        }
        .tv-indicator-option.active {
            background: #173a63;
            border-color: #4b90e2;
            color: #fff;
        }
        .tv-indicator-option-name {
            font-size: 12px;
            font-weight: 600;
        }
        .tv-indicator-option-desc {
            font-size: 10px;
            color: #9ea7b3;
        }
        .tv-indicator-option.active .tv-indicator-option-desc {
            color: #d7e4f3;
        }
        .tv-indicator-option-empty {
            padding: 8px;
            font-size: 11px;
            color: #888;
            text-align: center;
        }
        .tv-settings-menu {
            width: min(300px, calc(100vw - 32px));
            display: grid;
            gap: 8px;
        }
        .tv-settings-row {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 8px;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
        }
        .tv-settings-row:hover {
            background: #252525;
        }
        .tv-settings-row input[type="checkbox"] {
            margin: 2px 0 0;
        }
        .tv-settings-copy {
            display: grid;
            gap: 2px;
            min-width: 0;
        }
        .tv-settings-name {
            font-size: 12px;
            font-weight: 600;
            color: #f2f2f2;
        }
        .tv-settings-desc {
            font-size: 10px;
            color: #9ea7b3;
            line-height: 1.35;
        }
        /* Indicator option row — wraps toggle button + gear icon */
        .tv-indicator-option-row {
            display: flex;
            align-items: stretch;
            gap: 4px;
        }
        .tv-indicator-option-row .tv-indicator-option {
            flex: 1;
            min-width: 0;
        }
        .tv-indicator-settings-btn {
            flex-shrink: 0;
            width: 28px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--toolbar-button-bg);
            color: var(--text-muted);
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.15s, color 0.15s;
        }
        .tv-indicator-settings-btn:hover {
            background: var(--toolbar-button-hover);
            color: var(--text-primary);
            border-color: var(--accent-color);
        }
        /* Floating indicator settings panel — fixed to viewport, never clipped */
        .tv-ind-settings-panel {
            position: fixed;
            width: 300px;
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.55);
            z-index: 9999;
            overflow: hidden;
        }
        .tv-ind-settings-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 14px;
            background: var(--panel-bg-alt);
            border-bottom: 1px solid var(--border-color);
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }
        .tv-ind-settings-close {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 16px;
            cursor: pointer;
            padding: 0 4px;
            line-height: 1;
        }
        .tv-ind-settings-close:hover { color: var(--text-primary); }
        .tv-ind-settings-body {
            padding: 10px 12px;
            display: grid;
            gap: 8px;
        }
        .tv-ind-param-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 5px 2px;
        }
        .tv-ind-param-label {
            font-size: 12px;
            color: var(--text-secondary);
            flex: 1;
        }
        .tv-ind-param-row input[type="number"] {
            width: 72px;
            background: var(--panel-bg-strong);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-primary);
            padding: 4px 8px;
            font-size: 12px;
            outline: none;
        }
        .tv-ind-param-row input[type="number"]:focus { border-color: var(--accent-color); }
        .tv-ind-param-row input[type="color"] {
            width: 36px; height: 26px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: none;
            cursor: pointer;
            padding: 0;
        }
        .tv-ind-param-row input[type="checkbox"] {
            width: 16px; height: 16px;
            cursor: pointer;
            accent-color: var(--accent-color);
        }
        /* Warmup warning badge on indicator summary */
        .tv-indicator-warn-badge {
            display: none;
            font-size: 11px;
            color: #ffb74d;
            margin-left: 2px;
        }
        /* Warning highlight on individual indicator option when it can't be fully warmed up */
        .tv-indicator-option-row.warmup-warn .tv-indicator-option.active {
            border-color: #ffb74d;
            background: #2a1f0a;
        }
        .tv-indicator-option-row.warmup-warn .tv-indicator-option-name::after {
            content: ' ⚠';
            font-size: 9px;
            color: #ffb74d;
        }
        /* SMC overlay canvas — rendered above chart, pointer-events none */
        .tv-smc-overlay {
            position: absolute;
            inset: 0;
            pointer-events: none;
            z-index: 20;
            overflow: hidden;
            display: none;
        }
        .tv-smc-overlay canvas {
            position: absolute;
            inset: 0;
        }
        /* Indicator legend — inside canvas, pointer-events none so it doesn't block */
        .tv-indicator-legend {
            position: absolute;
            bottom: 34px;  /* clear of the time axis */
            left: 8px;
            display: none;  /* shown by updateIndicatorLegend() when indicators are active */
            flex-wrap: wrap;
            gap: 6px;
            z-index: 15;
            pointer-events: none;
        }
        .tv-legend-item {
            font-size: 10px;
            color: #ccc;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .tv-legend-swatch {
            width: 14px;
            height: 3px;
            border-radius: 2px;
        }
        /* RSI / MACD sub-panes */
        .tv-sub-pane {
            background: #1E1E1E;
            border-top: 1px solid #333;
            position: relative;
            overflow: hidden;
        }
        .tv-sub-pane-header {
            position: absolute;
            top: 4px;
            left: 8px;
            z-index: 5;
            font-size: 10px;
            color: #888;
            font-weight: bold;
            pointer-events: none;
        }
        /* OHLC hover tooltip */
        .tv-ohlc-tooltip {
            position: absolute;
            top: 8px;
            left: 8px;
            z-index: 50;
            font-size: 11px;
            font-family: 'Courier New', monospace;
            color: #ccc;
            pointer-events: none;
            white-space: nowrap;
            width: auto !important;
            height: auto !important;
            max-width: none !important;
            flex: none !important;
            display: none;
            line-height: 1.6;
        }
        .tv-ohlc-tooltip .tt-time { color: #aaa; font-size: 10px; margin-bottom: 2px; }
        .tv-ohlc-tooltip .tt-up   { color: #00FF00; }
        .tv-ohlc-tooltip .tt-dn   { color: #FF4444; }
        /* Candle close timer */
        .candle-close-timer {
            font-size: 11px;
            font-family: 'Courier New', monospace;
            padding: 3px 7px;
            border-radius: 4px;
            background: #2a2a2a;
            border: 1px solid #444;
            color: #ccc;
            white-space: nowrap;
            user-select: none;
            letter-spacing: 0.5px;
        }
        .price-info {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: stretch;
            gap: 6px;
            width: 100%;
            max-width: 100%;
            margin: 0 0 14px;
            padding: 8px;
            background-color: var(--panel-bg);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            box-shadow: var(--button-shadow);
            overflow: hidden;
        }
        .price-info-stats {
            display: flex;
            flex-wrap: wrap;
            align-items: stretch;
            align-content: flex-start;
            gap: 6px;
            min-width: 0;
        }
        .price-info-market-group {
            display: flex;
            align-items: stretch;
            gap: 6px;
            flex: 0 0 auto;
            padding-left: 10px;
            border-left: 1px solid var(--border-color);
        }
        .price-info-item {
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-width: 0;
            padding: 9px 14px;
            border-radius: 10px;
            background-color: var(--panel-bg-alt);
            border: 1px solid var(--border-color);
            transition: background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }
        .price-info-item:hover {
            background-color: var(--panel-hover);
            border-color: var(--border-color);
        }
        .price-info-item strong {
            display: block;
            margin-bottom: 5px;
            font-size: 0.62em;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
            white-space: nowrap;
        }
        .price-info-item > span {
            display: block;
            font-size: 1.02em;
            font-weight: 600;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
            line-height: 1.3;
            white-space: nowrap;
        }
        .price-info-item span span {
            font-size: 0.84em;
            font-weight: 500;
        }
        .price-info-item.price-info-primary > span {
            font-size: 1.18em;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--text-primary);
        }
        .green,
        .price-info-item.green > span {
            color: var(--good-color);
        }
        .red,
        .price-info-item.red > span {
            color: var(--bad-color);
        }
        .price-info-item[title] {
            cursor: help;
        }
        .price-info-item.price-info-market {
            flex-shrink: 0;
            align-self: stretch;
        }
        button {
            padding: 8px 16px;
            border-radius: 6px;
            border: none;
            background-color: #444;
            color: white;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #555;
        }
        .title {
            font-size: 1.5em;
            font-weight: bold;
            color: #800080;
        }
        .stream-control {
            display: inline-flex;
            align-items: center;
            margin-left: 10px;
        }
        .stream-control button {
            padding: 8px 16px;
            border-radius: 4px;
            border: 1px solid #404040;
            background-color: #2d2d2d;
            color: #ffffff;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            height: 36px; /* Match the height of other controls */
        }
        .stream-control button:hover {
            background-color: #3d3d3d;
            transform: translateY(-1px);
        }
        .stream-control button.paused {
            background-color: #2d2d2d;
            color: #ff4444;
        }
        .stream-control button.paused:hover {
            background-color: #3d3d3d;
        }
        .stream-control button::before {
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #4CAF50;
            transition: background-color 0.2s ease;
        }
        .stream-control button.paused::before {
            background-color: #ff4444;
        }
        .stream-control button:active {
            transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        .settings-control {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-left: 10px;
        }
        .settings-control button {
            padding: 8px 16px;
            border-radius: 4px;
            border: 1px solid #404040;
            background-color: #2d2d2d;
            color: #ffffff;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            height: 36px;
        }
        .settings-control button:hover {
            background-color: #3d3d3d;
            transform: translateY(-1px);
        }
        .settings-control button:active {
            transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        .settings-control button.success {
            background-color: #2e7d32;
            border-color: #4caf50;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 5px;
            width: 100%;
        }
        
        /* Add new CSS for the responsive grid layout */
        .charts-grid {
            display: grid;
            gap: 5px;
            width: 100%;
            min-width: 0;
            align-items: stretch;
        }
        
        .charts-grid.one-chart {
            grid-template-columns: 1fr;
        }
        
        .charts-grid.two-charts {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        
        .charts-grid.three-charts {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        
        .charts-grid.four-charts {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        
        .charts-grid.many-charts {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        #error-notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #ff4444;
            color: white;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10000;
            display: none;
            animation: slideIn 0.3s ease-out;
            max-width: 400px;
        }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .error-close {
            position: absolute;
            top: 5px;
            right: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 18px;
        }
        
        /* Mobile responsive styles */
        @media screen and (max-width: 768px) {
            .container {
                width: 100%;
                padding: 10px;
            }
            .header {
                padding: 10px;
            }
            .header-top, .header-bottom {
                flex-direction: column;
                align-items: stretch;
            }
            .controls {
                flex-direction: column;
                width: 100%;
            }
            .control-group {
                width: 100%;
                justify-content: space-between;
                min-height: 44px; /* Touch-friendly height */
            }
            .control-group label {
                font-size: 14px;
            }
            input[type="text"], select {
                min-width: 100px;
                font-size: 16px; /* Prevent zoom on iOS */
                min-height: 44px;
            }
            input[type="range"] {
                width: 100px;
            }
            .expiry-dropdown, .levels-dropdown {
                width: 100%;
            }
            .expiry-display, .levels-display {
                min-height: 44px;
                display: flex;
                align-items: center;
            }
            .chart-selector {
                flex-direction: column;
            }
            .chart-checkbox {
                width: 100%;
                min-height: 44px;
                display: flex;
                align-items: center;
            }
            .chart-checkbox input[type="checkbox"] {
                width: 22px;
                height: 22px;
            }
            .charts-grid.two-charts,
            .charts-grid.three-charts,
            .charts-grid.four-charts,
            .charts-grid.many-charts {
                grid-template-columns: 1fr;
            }
            .chart-container {
                height: 350px;
            }
            .price-info {
                grid-template-columns: 1fr;
                padding: 8px;
            }
            .price-info-stats {
                flex-wrap: wrap;
            }
            .price-info-market-group {
                padding-left: 0;
                border-left: none;
                border-top: 1px solid var(--border-color);
                padding-top: 8px;
            }
            .stream-control button, .settings-control button {
                min-height: 44px;
                padding: 10px 16px;
            }
            button {
                min-height: 44px;
            }
        }
        
        @media screen and (max-width: 480px) {
            .title {
                font-size: 1.2em;
            }
            .chart-container {
                height: 300px;
            }
        }

        /* Fullscreen chart overlay */
        .chart-fullscreen-btn {
            position: absolute;
            top: 8px;
            left: 8px;
            z-index: 200;
            background: rgba(45, 45, 45, 0.85);
            border: 1px solid #555;
            color: #ccc;
            width: 30px;
            height: 30px;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.2s;
            padding: 0;
            line-height: 1;
        }
        .chart-container:not(.fullscreen):hover .chart-fullscreen-btn,
        .price-chart-container:not(.fullscreen) .chart-container:hover .chart-fullscreen-btn,
        .chart-fullscreen-btn:focus-visible {
            opacity: 1;
        }
        .chart-fullscreen-btn:hover {
            background: rgba(80, 80, 80, 0.95);
            color: #fff;
            border-color: #777;
        }
        .chart-container.fullscreen,
        .price-chart-container.fullscreen {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            max-height: 100vh !important;
            z-index: 9999 !important;
            border-radius: 0 !important;
            margin: 0 !important;
            padding: 10px !important;
            background-color: #1E1E1E !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }
        .chart-container.fullscreen {
            display: flex !important;
            flex-direction: column !important;
        }
        .chart-container.fullscreen > div:not(.chart-fullscreen-chrome) {
            flex: 1 1 auto !important;
            width: 100% !important;
            min-height: 0 !important;
            height: auto !important;
            overflow: hidden !important;
        }
        .chart-container.fullscreen.js-plotly-plot {
            min-height: 0 !important;
        }
        .chart-container.fullscreen .plot-container.plotly {
            flex: 1 1 auto !important;
            width: 100% !important;
            min-height: 0 !important;
            height: 100% !important;
        }
        .chart-container.fullscreen .heatmap-chart-shell {
            display: grid !important;
            grid-template-rows: auto minmax(0, 1fr) !important;
            height: 100% !important;
            min-height: 0 !important;
        }
        .chart-container.fullscreen .heatmap-plot {
            min-height: 0 !important;
        }
        .price-chart-container.fullscreen {
            display: flex !important;
            flex-direction: column !important;
            padding: 0 !important;
        }
        .price-chart-container.fullscreen .tv-toolbar-container {
            flex: 0 0 auto;
            border-radius: 0;
            z-index: 10000;
        }
        .price-chart-container.fullscreen #price-chart {
            flex: 1 1 auto !important;
            height: auto !important;
            min-height: 0 !important;
            border-radius: 0 !important;
        }
        .price-chart-container.fullscreen .tv-sub-pane {
            flex: 0 0 auto;
        }
        .chart-fullscreen-chrome {
            display: none !important;
            position: fixed;
            inset: 0;
            width: 0;
            height: 0;
            overflow: visible;
            pointer-events: none;
            flex: 0 0 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
        }
        .chart-container.fullscreen .chart-fullscreen-chrome,
        .price-chart-container.fullscreen .chart-fullscreen-chrome {
            display: block !important;
        }
        .chart-fullscreen-hoverzone {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 48px;
            z-index: 10001;
            pointer-events: auto;
        }
        .chart-fullscreen-logo {
            position: fixed;
            top: 6px;
            left: 10px;
            z-index: 10002;
            font-family: Arial, sans-serif;
            font-size: 11px;
            font-weight: bold;
            color: var(--accent-color);
            opacity: 0.7;
            pointer-events: none;
            letter-spacing: 0.5px;
        }
        .chart-container.fullscreen .chart-fullscreen-btn,
        .price-chart-container.fullscreen .chart-fullscreen-btn {
            opacity: 0;
            position: fixed;
            top: 10px;
            left: 10px;
            right: auto;
            z-index: 10004;
        }
        .chart-container.fullscreen:has(.chart-fullscreen-hoverzone:hover) .chart-fullscreen-btn,
        .price-chart-container.fullscreen:has(.chart-fullscreen-hoverzone:hover) .chart-fullscreen-btn,
        .chart-container.fullscreen .chart-fullscreen-btn:hover,
        .chart-container.fullscreen .chart-fullscreen-btn:focus-visible,
        .price-chart-container.fullscreen .chart-fullscreen-btn:hover,
        .price-chart-container.fullscreen .chart-fullscreen-btn:focus-visible {
            opacity: 1;
        }
        /* Pop-out button */
        .chart-popout-btn {
            position: absolute;
            top: 8px;
            left: 42px;
            z-index: 200;
            background: rgba(45, 45, 45, 0.85);
            border: 1px solid #555;
            color: #ccc;
            width: 30px;
            height: 30px;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.2s;
            padding: 0;
            line-height: 1;
        }
        .chart-container:not(.fullscreen):hover .chart-popout-btn,
        .price-chart-container:not(.fullscreen) .chart-container:hover .chart-popout-btn,
        .chart-popout-btn:focus-visible {
            opacity: 1;
        }
        .chart-popout-btn:hover {
            background: rgba(80, 80, 80, 0.95);
            color: #fff;
            border-color: #777;
        }
        .chart-container.fullscreen .chart-popout-btn,
        .price-chart-container.fullscreen .chart-popout-btn {
            opacity: 0;
            position: fixed;
            top: 10px;
            left: 46px;
            right: auto;
            z-index: 10004;
        }
        .chart-container.fullscreen:has(.chart-fullscreen-hoverzone:hover) .chart-popout-btn,
        .price-chart-container.fullscreen:has(.chart-fullscreen-hoverzone:hover) .chart-popout-btn,
        .chart-container.fullscreen .chart-popout-btn:hover,
        .chart-container.fullscreen .chart-popout-btn:focus-visible,
        .price-chart-container.fullscreen .chart-popout-btn:hover,
        .price-chart-container.fullscreen .chart-popout-btn:focus-visible {
            opacity: 1;
        }

        :root {
            --app-bg: #0f131a;
            --panel-bg: #1a212c;
            --panel-bg-alt: #222b38;
            --panel-bg-strong: #2a3444;
            --panel-hover: #344155;
            --chart-bg: #10161f;
            --chart-bg-alt: #171f2b;
            --border-color: #3a4659;
            --text-primary: #eef3fb;
            --text-secondary: #c7d1e0;
            --text-muted: #8e9cb1;
            --accent-color: #5ab0ff;
            --accent-soft: rgba(90, 176, 255, 0.18);
            --slider-color: #4dd5a6;
            --good-color: #37d67a;
            --bad-color: #ff6675;
            --toolbar-bg: #151d29;
            --toolbar-button-bg: #243142;
            --toolbar-button-hover: #304157;
            --toolbar-button-active: #1d5fa6;
            --toolbar-button-danger: #5f2428;
            --grid-color: #2b3748;
            --crosshair-color: #607188;
            --tooltip-bg: linear-gradient(180deg, rgba(30, 36, 46, 0.97), rgba(14, 18, 24, 0.99));
            --tooltip-border: rgba(255, 255, 255, 0.08);
            --error-bg: #c84d57;
            --success-bg: #2c8f54;
            --button-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
            --overlay-shadow: 0 14px 36px rgba(0, 0, 0, 0.38);
            --match-em-bg: #243142;
            --match-em-text: #9eb3d2;
            --match-em-border: #5c6d86;
            --match-em-active-bg: #123d38;
            --match-em-active-text: #7ef0ca;
            --match-em-active-border: #2e9b82;
            --floating-button-bg: rgba(25, 32, 42, 0.88);
            --floating-button-hover: rgba(58, 72, 92, 0.96);
        }

        body {
            background-color: var(--app-bg);
            color: var(--text-primary);
        }
        body[data-theme="neon"] {
            background-image:
                radial-gradient(circle at top left, rgba(0, 234, 255, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(255, 79, 163, 0.16), transparent 24%),
                radial-gradient(circle at 50% 100%, rgba(143, 255, 61, 0.10), transparent 30%);
            background-attachment: fixed;
        }
        body,
        .header,
        .control-group,
        .chart-checkbox,
        .chart-container,
        .price-chart-container,
        .tv-toolbar-container,
        .tv-sub-pane,
        .price-info,
        .price-info-item,
        input,
        select,
        button,
        .expiry-display,
        .levels-display,
        .expiry-options,
        .levels-options,
        .tv-historical-tooltip,
        .chart-hover-tooltip,
        #error-notification {
            transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .header {
            background-color: var(--panel-bg);
            box-shadow: var(--button-shadow);
        }
        .header-bottom {
            border-top-color: var(--border-color);
        }
        .control-group,
        .chart-checkbox {
            background-color: var(--panel-bg-alt);
            color: var(--text-secondary);
            border: 1px solid transparent;
        }
        .control-group label,
        .chart-checkbox label,
        .tv-chart-title,
        .tv-legend-item,
        .tv-sub-pane-header {
            color: var(--text-secondary);
        }
        .price-info,
        .price-info-item {
            background-color: var(--panel-bg);
            border-color: var(--border-color);
        }
        .price-info-item {
            background-color: var(--panel-bg-alt);
        }
        .price-info-item:hover {
            background-color: var(--panel-hover);
            border-color: var(--border-color);
        }
        .expiry-display,
        .levels-display,
        input[type="text"],
        input[type="number"],
        select {
            background-color: var(--panel-bg-strong);
            color: var(--text-primary);
            border-color: var(--border-color);
        }
        .expiry-display:hover,
        .levels-display:hover,
        .expiry-option:hover,
        .levels-option:hover,
        button:hover,
        .stream-control button:hover,
        .settings-control button:hover {
            background-color: var(--panel-hover);
            border-color: var(--accent-color);
        }
        .expiry-display::after,
        .levels-display::after,
        .tm-stats,
        .tm-label,
        .tm-divider,
        .tv-sub-pane-header {
            color: var(--text-muted);
        }
        .expiry-options,
        .levels-options {
            background-color: var(--panel-bg-strong);
            border-color: var(--border-color);
            box-shadow: var(--button-shadow);
        }
        .expiry-buttons,
        .tv-toolbar-container,
        .tv-sub-pane {
            border-color: var(--border-color);
        }
        .expiry-buttons button,
        .settings-control button,
        .stream-control button,
        button,
        .tm-btn,
        .tv-tb-btn,
        .candle-close-timer {
            background-color: var(--panel-bg-strong);
            color: var(--text-primary);
            border-color: var(--border-color);
        }
        .expiry-buttons .expiry-range-btns button,
        .tv-tb-btn.active {
            background-color: color-mix(in srgb, var(--accent-color) 30%, var(--panel-bg-strong));
            border-color: var(--accent-color);
            color: var(--text-primary);
        }
        .tv-tb-btn.danger {
            background-color: var(--toolbar-button-danger);
            border-color: color-mix(in srgb, var(--bad-color) 70%, var(--toolbar-button-danger));
        }
        .tm-btn:hover,
        .tv-tb-btn:hover {
            color: var(--text-primary);
        }
        .tm-btn-del:hover {
            background: color-mix(in srgb, var(--bad-color) 22%, transparent);
            border-color: var(--bad-color);
            color: var(--bad-color);
        }
        input[type="range"] {
            background: var(--grid-color);
        }
        input[type="range"]::-webkit-slider-thumb {
            background: var(--slider-color);
        }
        input[type="checkbox"] {
            accent-color: var(--accent-color);
        }
        .title {
            color: var(--accent-color);
        }
        .green {
            color: var(--good-color);
        }
        .red {
            color: var(--bad-color);
        }
        .chart-container {
            background-color: var(--panel-bg);
            box-shadow: var(--button-shadow);
        }
        .price-chart-container,
        #price-chart,
        .tv-toolbar-container,
        .tv-sub-pane {
            background: var(--chart-bg);
        }
        .tv-historical-tooltip,
        .chart-hover-tooltip {
            background: var(--tooltip-bg);
            border-color: var(--tooltip-border);
            color: var(--text-primary);
            box-shadow: var(--overlay-shadow);
        }
        .tv-historical-tooltip .tt-badge,
        .chart-hover-tooltip .tt-badge {
            background: color-mix(in srgb, var(--text-primary) 10%, transparent);
            color: var(--text-secondary);
        }
        .tv-historical-tooltip .tt-time,
        .chart-hover-tooltip .tt-time,
        .tv-historical-tooltip .tt-value,
        .chart-hover-tooltip .tt-value,
        .tv-historical-tooltip .tt-more,
        .chart-hover-tooltip .tt-more,
        .tv-ohlc-tooltip .tt-time {
            color: var(--text-muted);
        }
        .tv-historical-tooltip .tt-name,
        .chart-hover-tooltip .tt-name,
        .tv-ohlc-tooltip {
            color: var(--text-primary);
        }
        .stream-control button::before {
            background-color: var(--good-color);
        }
        .stream-control button.paused {
            color: var(--bad-color);
        }
        .stream-control button.paused::before {
            background-color: var(--bad-color);
        }
        .settings-control button.success {
            background-color: var(--success-bg);
            border-color: var(--success-bg);
        }
        #error-notification {
            background-color: var(--error-bg);
            color: var(--text-primary);
        }
        .chart-fullscreen-btn,
        .chart-popout-btn {
            background: var(--floating-button-bg);
            border-color: var(--border-color);
            color: var(--text-secondary);
        }
        .chart-fullscreen-btn:hover,
        .chart-popout-btn:hover {
            background: var(--floating-button-hover);
            border-color: var(--accent-color);
            color: var(--text-primary);
        }
        .chart-container.fullscreen,
        .price-chart-container.fullscreen {
            background-color: var(--chart-bg) !important;
        }
        body[data-theme="neon"] .header {
            border: 1px solid color-mix(in srgb, var(--accent-color) 30%, var(--border-color));
            box-shadow: 0 0 0 1px rgba(0, 234, 255, 0.08), 0 24px 50px rgba(0, 0, 0, 0.46), 0 0 34px rgba(0, 234, 255, 0.10);
        }
        body[data-theme="neon"] .title {
            color: #7df7ff;
            text-shadow: 0 0 10px rgba(0, 234, 255, 0.32), 0 0 22px rgba(255, 79, 163, 0.18);
        }
        body[data-theme="neon"] .control-group,
        body[data-theme="neon"] .chart-checkbox,
        body[data-theme="neon"] .expiry-options,
        body[data-theme="neon"] .levels-options,
        body[data-theme="neon"] .chart-container,
        body[data-theme="neon"] .price-chart-container,
        body[data-theme="neon"] .price-info,
        body[data-theme="neon"] .price-info-item,
        body[data-theme="neon"] .tv-toolbar-container,
        body[data-theme="neon"] .tv-sub-pane {
            border-color: color-mix(in srgb, var(--accent-color) 22%, var(--border-color));
            box-shadow: 0 0 0 1px rgba(0, 234, 255, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.02), var(--button-shadow);
        }
        body[data-theme="neon"] .price-chart-container,
        body[data-theme="neon"] #price-chart,
        body[data-theme="neon"] .tv-toolbar-container,
        body[data-theme="neon"] .tv-sub-pane {
            background-image: linear-gradient(180deg, rgba(0, 234, 255, 0.03), transparent 24%);
        }
        body[data-theme="neon"] .expiry-buttons .expiry-range-btns button,
        body[data-theme="neon"] .tv-tb-btn.active,
        body[data-theme="neon"] .mobile-panel-toggles button.active,
        body[data-theme="neon"] #match_em_range.active {
            box-shadow: 0 0 0 1px rgba(0, 234, 255, 0.22), 0 0 18px rgba(0, 234, 255, 0.18);
        }
        body[data-theme="neon"] .stream-control button::before {
            box-shadow: 0 0 10px currentColor;
        }
        body[data-theme="neon"] ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, rgba(0, 234, 255, 0.78), rgba(255, 79, 163, 0.82));
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08), 0 0 10px rgba(0, 234, 255, 0.18);
        }
        #match_em_range {
            margin-left: 4px;
            padding: 2px 6px;
            font-size: 11px;
            min-height: auto;
            background: var(--match-em-bg);
            color: var(--match-em-text);
            border: 1px solid var(--match-em-border);
            border-radius: 3px;
            box-shadow: none;
        }
        #match_em_range.active {
            background: var(--match-em-active-bg);
            color: var(--match-em-active-text);
            border-color: var(--match-em-active-border);
        }
        .theme-control select {
            min-width: 120px;
        }
        .mobile-panel-toggles {
            display: none;
        }
        .mobile-panel-toggles button.active {
            background-color: color-mix(in srgb, var(--accent-color) 28%, var(--panel-bg-strong));
            border-color: var(--accent-color);
            color: var(--text-primary);
        }
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: color-mix(in srgb, var(--chart-bg) 88%, black);
        }
        ::-webkit-scrollbar-thumb {
            background: color-mix(in srgb, var(--border-color) 85%, var(--accent-color));
            border-radius: 999px;
        }
        @media screen and (max-width: 900px) {
            body.mobile-layout {
                padding-bottom: calc(72px + env(safe-area-inset-bottom, 0px));
            }
            body.mobile-layout .container {
                width: 100%;
                padding: 12px;
            }
            body.mobile-layout .header {
                gap: 12px;
                margin-bottom: 12px;
                padding: 14px;
                border-radius: 16px;
            }
            body.mobile-layout .header-top {
                align-items: stretch;
                gap: 12px;
            }
            body.mobile-layout .header-top .controls {
                width: 100%;
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
                align-items: stretch;
            }
            body.mobile-layout .header-top .controls > * {
                min-width: 0;
                margin-left: 0;
            }
            body.mobile-layout .header-top .control-group,
            body.mobile-layout .header-top .stream-control,
            body.mobile-layout .header-top .settings-control {
                width: 100%;
                min-height: 48px;
                margin-left: 0;
                justify-content: space-between;
            }
            body.mobile-layout .header-top .control-group input[type="text"],
            body.mobile-layout .header-top .control-group select,
            body.mobile-layout .header-top .control-group .expiry-dropdown,
            body.mobile-layout .header-top .control-group .levels-dropdown {
                min-width: 0;
                width: 100%;
            }
            body.mobile-layout .header-top .stream-control button,
            body.mobile-layout .header-top .settings-control button {
                width: 100%;
                min-height: 48px;
                justify-content: center;
            }
            body.mobile-layout .header-top .settings-control {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
            }
            body.mobile-layout .mobile-panel-toggles {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
                position: sticky;
                top: 0;
                z-index: 400;
                margin: 0 0 12px;
                padding-top: 4px;
                background: linear-gradient(180deg, color-mix(in srgb, var(--app-bg) 96%, transparent), transparent);
            }
            body.mobile-layout .mobile-panel-toggles button {
                min-height: 48px;
                border: 1px solid var(--border-color);
                background: var(--panel-bg);
                color: var(--text-primary);
                box-shadow: var(--button-shadow);
                font-weight: 600;
            }
            body.mobile-layout #advanced-controls,
            body.mobile-layout #chart-selector {
                display: none;
            }
            body.mobile-layout.mobile-filters-open #advanced-controls {
                display: block;
                margin-bottom: 12px;
                padding: 14px;
                border: 1px solid var(--border-color);
                border-radius: 16px;
                background: var(--panel-bg);
                box-shadow: var(--button-shadow);
            }
            body.mobile-layout.mobile-filters-open #advanced-controls .controls {
                display: grid;
                grid-template-columns: 1fr;
                width: 100%;
                gap: 10px;
            }
            body.mobile-layout.mobile-filters-open #advanced-controls .control-group {
                width: 100%;
                min-height: 48px;
                justify-content: space-between;
            }
            body.mobile-layout.mobile-filters-open #advanced-controls .control-group input[type="text"],
            body.mobile-layout.mobile-filters-open #advanced-controls .control-group select,
            body.mobile-layout.mobile-filters-open #advanced-controls .control-group .expiry-dropdown,
            body.mobile-layout.mobile-filters-open #advanced-controls .control-group .levels-dropdown {
                min-width: 0;
                width: 100%;
            }
            body.mobile-layout.mobile-filters-open #advanced-controls .expiry-display,
            body.mobile-layout.mobile-filters-open #advanced-controls .levels-display,
            body.mobile-layout.mobile-filters-open #advanced-controls input[type="text"],
            body.mobile-layout.mobile-filters-open #advanced-controls input[type="number"],
            body.mobile-layout.mobile-filters-open #advanced-controls select {
                min-height: 48px;
                font-size: 16px;
            }
            body.mobile-layout.mobile-filters-open #advanced-controls input[type="range"] {
                width: 100%;
            }
            body.mobile-layout.mobile-filters-open #advanced-controls .expiry-options,
            body.mobile-layout.mobile-filters-open #advanced-controls .levels-options {
                max-height: min(42dvh, 320px);
            }
            body.mobile-layout.mobile-charts-open #chart-selector {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
                margin: 0 0 14px;
            }
            body.mobile-layout #chart-selector .chart-checkbox {
                width: 100%;
                min-height: 54px;
                padding: 10px 12px;
                gap: 10px;
            }
            body.mobile-layout #chart-selector .chart-checkbox input[type="checkbox"] {
                width: 22px;
                height: 22px;
            }
            body.mobile-layout .price-info {
                display: flex;
                flex-direction: row;
                flex-wrap: nowrap;
                align-items: stretch;
                gap: 8px;
                overflow-x: auto;
                overflow-y: hidden;
                padding: 8px;
                scroll-snap-type: x proximity;
                -webkit-overflow-scrolling: touch;
            }
            body.mobile-layout .price-info-stats {
                display: flex;
                flex: 0 0 auto;
                gap: 8px;
                flex-wrap: nowrap;
                min-width: 0;
            }
            body.mobile-layout .price-info-market-group {
                display: flex;
                flex: 0 0 auto;
                gap: 8px;
                flex-wrap: nowrap;
                margin-left: auto;
                padding-left: 10px;
                border-left: 1px solid var(--border-color);
                border-top: none;
                padding-top: 0;
            }
            body.mobile-layout .price-info-item {
                min-height: 72px;
                min-width: 128px;
                padding: 10px 12px;
                border-radius: 12px;
                box-sizing: border-box;
                scroll-snap-align: start;
            }
            body.mobile-layout .chart-container {
                height: clamp(320px, 44dvh, 420px);
                border-radius: 14px;
            }
            body.mobile-layout .heatmap-chart-shell {
                padding: 8px;
                gap: 8px;
            }
            body.mobile-layout .heatmap-chart-toolbar {
                padding: 0;
            }
            body.mobile-layout .heatmap-chart-toolbar-controls {
                width: 100%;
                justify-content: stretch;
            }
            body.mobile-layout .heatmap-chart-toolbar .control-group {
                flex: 1 1 100%;
                justify-content: space-between;
                border-radius: 12px;
            }
            body.mobile-layout .heatmap-chart-toolbar .control-group select {
                min-width: 0;
                flex: 1 1 auto;
            }
            body.mobile-layout .price-chart-container {
                border-radius: 14px;
            }
            body.mobile-layout #price-chart {
                height: clamp(380px, 58dvh, 560px) !important;
            }
            body.mobile-layout .tv-toolbar-container {
                flex-wrap: nowrap;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                padding-bottom: 8px;
            }
            body.mobile-layout .tv-tb-btn,
            body.mobile-layout .candle-close-timer {
                flex: 0 0 auto;
                min-height: 38px;
            }
            body.mobile-layout .tv-toolbar-sep,
            body.mobile-layout .tv-chart-title {
                flex: 0 0 auto;
            }
            body.mobile-layout .chart-container.fullscreen {
                padding: 12px 8px 8px !important;
            }
            body.mobile-layout .price-chart-container.fullscreen {
                padding: 0 !important;
            }
            body.mobile-layout .chart-container.fullscreen .chart-fullscreen-btn,
            body.mobile-layout .price-chart-container.fullscreen .chart-fullscreen-btn {
                top: 10px;
                left: 10px;
                right: auto;
            }
        }
        @media screen and (max-width: 640px) {
            body.mobile-layout .header-top .controls,
            body.mobile-layout .header-top .settings-control,
            body.mobile-layout.mobile-charts-open #chart-selector {
                grid-template-columns: 1fr;
            }
            body.mobile-layout.mobile-filters-open #advanced-controls .control-group {
                flex-wrap: wrap;
                align-items: flex-start;
            }
            body.mobile-layout .price-info-item {
                min-width: 148px;
            }
        }
        @media (hover: none), (pointer: coarse) {
            .chart-fullscreen-btn {
                opacity: 1;
                width: 38px;
                height: 38px;
            }
            .chart-container.fullscreen:has(.chart-fullscreen-hoverzone:hover) .chart-fullscreen-btn,
            .price-chart-container.fullscreen:has(.chart-fullscreen-hoverzone:hover) .chart-fullscreen-btn,
            .chart-container.fullscreen .chart-fullscreen-btn:hover,
            .price-chart-container.fullscreen .chart-fullscreen-btn:hover {
                opacity: 1;
            }
            .chart-popout-btn {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div id="error-notification">
        <span class="error-close" onclick="hideError()">&times;</span>
        <div id="error-message"></div>
    </div>
    <div class="container">
        <div class="header">
            <div class="header-top">
                <div>
                    <div class="title">EzDuz1t Options</div>
                    <div id="token-monitor">
                        <span class="tm-dot tm-neutral" id="tm-dot"></span>
                        <span class="tm-stats tm-label">SCHWAB API</span>
                        <span class="tm-stats" id="tm-access-stat" title="">…</span>
                        <span class="tm-stats tm-divider">·</span>
                        <span class="tm-stats" id="tm-refresh-stat" title="">…</span>
                        <div class="tm-btn-group">
                            <button class="tm-btn" onclick="fetchTokenHealth()" title="Refresh token status">&#8635;</button>
                            <button class="tm-btn tm-btn-del" onclick="forceDeleteToken()" title="Clear stored tokens">&#128465; reset</button>
                        </div>
                    </div>
                </div>
                <div class="controls">
                    <div class="control-group">
                        <label for="ticker">Ticker:</label>
                        <input type="text" id="ticker" placeholder="Enter Ticker" value="SPY" title="Enter a ticker symbol (e.g., SPY, AAPL) or special aggregate tickers: 'MARKET' (SPX grid + SPY exposures) or 'MARKET2' (SPY base)">
                    </div>
                    <div class="control-group">
                        <label for="timeframe">Timeframe:</label>
                        <select id="timeframe">
                            <option value="1">1 min</option>
                            <option value="5">5 min</option>
                            <option value="15">15 min</option>
                            <option value="30">30 min</option>
                            <option value="60">1 hour</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Expiry:</label>
                        <div class="expiry-dropdown">
                            <div class="expiry-display" id="expiry-display">
                                <span id="expiry-text">Select expiry dates...</span>
                            </div>
                            <div class="expiry-options" id="expiry-options">
                                <!-- Options will be populated here -->
                                <div class="expiry-buttons">
                                    <div class="expiry-range-btns">
                                        <button type="button" id="expiryToday">Today</button>
                                        <button type="button" id="expiryThisWk">This Wk</button>
                                        <button type="button" id="expiry2Wks">+1 Wk</button>
                                        <button type="button" id="expiry4Wks">+2 Wks</button>
                                        <button type="button" id="expiry1Mo">+1 Mo</button>
                                    </div>
                                    <button type="button" id="selectAllExpiry">All</button>
                                    <button type="button" id="clearAllExpiry">Clear</button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="stream-control">
                        <button id="streamToggle">Auto-Update</button>
                    </div>
                    <div class="settings-control">
                        <button id="saveSettings" title="Save current settings to file">💾 Save</button>
                        <button id="loadSettings" title="Load settings from file">📂 Load</button>
                    </div>
                    <div class="control-group theme-control">
                        <label for="theme_select">Theme:</label>
                        <select id="theme_select" title="Choose a site theme">
                            <option value="dark" selected>Dark</option>
                            <option value="light">Light</option>
                            <option value="ocean">Ocean</option>
                            <option value="midnight">Midnight</option>
                            <option value="arctic">Arctic</option>
                            <option value="cobalt">Cobalt</option>
                            <option value="forest">Forest</option>
                            <option value="sunset">Sunset</option>
                            <option value="ruby">Ruby</option>
                            <option value="amber">Amber</option>
                            <option value="neon">Neon</option>
                        </select>
                    </div>
                </div>
            </div>
            <div class="header-bottom" id="advanced-controls">
                <div class="controls">
                    <div class="control-group">
                        <label for="strike_range">Strike Range (%):</label>
                        <input type="range" id="strike_range" min="0.5" max="20" value="2" step="0.5">
                        <span class="range-value" id="strike_range_value">2%</span>
                        <button id="match_em_range" title="Toggle: auto-sync strike range to Expected Move (ATM straddle) + 0.5% wiggle room">📐 EM</button>
                    </div>
                    <div class="control-group">
                        <label for="exposure_metric">Exposure Metric:</label>
                        <select id="exposure_metric" title="Select the metric used to weight exposure formulas (GEX/DEX/VEX etc)">
                            <option value="Open Interest" selected>Open Interest</option>
                            <option value="Volume">Volume</option>
                            <option value="Max OI vs Volume">Max OI vs Volume</option>
                            <option value="OI + Volume">OI + Volume</option>
                        </select>
                    </div>
                    <div class="control-group" title="When enabled, exposure formulas are adjusted by delta.">
                        <input type="checkbox" id="delta_adjusted_exposures">
                        <label for="delta_adjusted_exposures">Delta-Adjusted Exposures</label>
                    </div>
                    <div class="control-group" title="When enabled, exposures are calculated in notional value (Dollars). When disabled, in share equivalents.">
                        <input type="checkbox" id="calculate_in_notional" checked>
                        <label for="calculate_in_notional">Notional Calc</label>
                    </div>
                    <div class="control-group">
                        <input type="checkbox" id="show_calls">
                        <label for="show_calls">Calls</label>
                    </div>
                    <div class="control-group">
                        <input type="checkbox" id="show_puts">
                        <label for="show_puts">Puts</label>
                    </div>
                    <div class="control-group">
                        <input type="checkbox" id="show_net" checked>
                        <label for="show_net">Net</label>
                    </div>
                    <div class="control-group">
                        <label for="coloring_mode">Coloring Mode:</label>
                        <select id="coloring_mode" title="Solid: All bars same color | Linear: Gradual fade by value | Ranked: Only highest exposures are bright, others heavily muted">
                            <option value="Solid" selected>Solid</option>
                            <option value="Linear Intensity">Linear Intensity</option>
                            <option value="Ranked Intensity">Ranked Intensity</option>
                        </select>
                    </div>
                    <div class="control-group" style="display:none;">
                        <label>Price Levels:</label>
                        <div class="levels-dropdown">
                            <div class="levels-display" id="levels-display">
                                <span id="levels-text">None</span>
                            </div>
                            <div class="levels-options" id="levels-options">
                                <div class="levels-option"><input type="checkbox" value="GEX" id="lvl-GEX"><label for="lvl-GEX">GEX</label></div>
                                <div class="levels-option"><input type="checkbox" value="AbsGEX" id="lvl-AbsGEX"><label for="lvl-AbsGEX">Abs GEX</label></div>
                                <div class="levels-option"><input type="checkbox" value="DEX" id="lvl-DEX"><label for="lvl-DEX">DEX</label></div>
                                <div class="levels-option"><input type="checkbox" value="VEX" id="lvl-VEX"><label for="lvl-VEX">Vanna</label></div>
                                <div class="levels-option"><input type="checkbox" value="Charm" id="lvl-Charm"><label for="lvl-Charm">Charm</label></div>
                                <div class="levels-option"><input type="checkbox" value="Volume" id="lvl-Volume"><label for="lvl-Volume">Volume</label></div>
                                <div class="levels-option"><input type="checkbox" value="Speed" id="lvl-Speed"><label for="lvl-Speed">Speed</label></div>
                                <div class="levels-option"><input type="checkbox" value="Vomma" id="lvl-Vomma"><label for="lvl-Vomma">Vomma</label></div>
                                <div class="levels-option"><input type="checkbox" value="Color" id="lvl-Color"><label for="lvl-Color">Color</label></div>
                                <div class="levels-option"><input type="checkbox" value="Expected Move" id="lvl-ExpectedMove"><label for="lvl-ExpectedMove">Expected Move</label></div>
                            </div>
                        </div>
                    </div>
                    <div class="control-group" style="display:none;">
                        <label for="levels_count">Top #:</label>
                        <input type="number" id="levels_count" min="1" max="10" value="3" style="width: 50px;">
                    </div>
                    <div class="control-group" style="display:none;">
                        <input type="checkbox" id="use_heikin_ashi">
                        <label for="use_heikin_ashi">Heikin-Ashi</label>
                    </div>
                    <div class="control-group">
                        <input type="checkbox" id="horizontal_bars">
                        <label for="horizontal_bars">Horizontal Bars</label>
                    </div>
                    <!-- New Absolute GEX Settings -->
                    <div class="control-group">
                        <input type="checkbox" id="show_abs_gex">
                        <label for="show_abs_gex">Show Abs GEX Area</label>
                    </div>
                    <div class="control-group">
                        <label for="abs_gex_opacity">Abs GEX Opacity:</label>
                        <input type="range" id="abs_gex_opacity" min="0" max="100" value="20" style="width: 80px;">
                    </div>
                    <div class="control-group">
                        <input type="checkbox" id="use_range">
                        <label for="use_range">% Range Volume</label>
                    </div>
                    <div class="control-group">
                        <label for="call_color">Call Color:</label>
                        <input type="color" id="call_color" value="#00FF00">
                    </div>
                    <div class="control-group">
                        <label for="put_color">Put Color:</label>
                        <input type="color" id="put_color" value="#FF0000">
                    </div>
                    <div class="control-group">
                        <input type="checkbox" id="highlight_max_level">
                        <label for="highlight_max_level">Highlight Max Level</label>
                    </div>
                    <div class="control-group">
                        <label for="max_level_mode">Max Level Mode:</label>
                        <select id="max_level_mode" title="Absolute: highlights the single bar with the largest magnitude | Net: highlights the strike where the net (calls minus puts) is largest">
                            <option value="Absolute" selected>Absolute</option>
                            <option value="Net">Net</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="max_level_color">Max Level Color:</label>
                        <input type="color" id="max_level_color" value="#800080">
                    </div>
                </div>
            </div>
        </div>

        <div class="mobile-panel-toggles">
            <button type="button" id="mobile-toggle-filters" aria-controls="advanced-controls" aria-expanded="false">Filters</button>
            <button type="button" id="mobile-toggle-charts" aria-controls="chart-selector" aria-expanded="false">Charts</button>
        </div>
        
        <div class="price-info" id="price-info"></div>
        
        <div class="chart-selector" id="chart-selector">
            <div class="chart-checkbox">
                <input type="checkbox" id="price" checked>
                <label for="price">Price Chart</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="gamma" checked>
                <label for="gamma">Gamma Exposure</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="heatmap">
                <label for="heatmap">Exposure Heatmap</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="delta" checked>
                <label for="delta">Delta Exposure</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="vanna" checked>
                <label for="vanna">Vanna Exposure</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="charm" checked>
                <label for="charm">Charm Exposure</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="speed">
                <label for="speed">Speed Exposure</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="vomma">
                <label for="vomma">Vomma Exposure</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="color">
                <label for="color">Color Exposure</label>
            </div>

            <div class="chart-checkbox">
                <input type="checkbox" id="options_volume" checked>
                <label for="options_volume">Options Volume</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="open_interest">
                <label for="open_interest">Open Interest</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="volume" checked>
                <label for="volume">Volume Ratio</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="large_trades" checked>
                <label for="large_trades">Options Chain</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="centroid" checked>
                <label for="centroid">Call vs Put Centroid Map</label>
            </div>
        </div>

        <div id="chart-control-staging" aria-hidden="true">
            <div class="control-group" id="heatmap-type-control">
                <label for="heatmap_type">Heatmap Type:</label>
                <select id="heatmap_type" title="Choose the metric shown in the exposure heatmap">
                    <option value="GEX" selected>GEX</option>
                    <option value="AbsGEX">Abs GEX</option>
                    <option value="DEX">DEX</option>
                    <option value="VEX">Vanna</option>
                    <option value="Charm">Charm</option>
                    <option value="Volume">Volume</option>
                    <option value="Speed">Speed</option>
                    <option value="Vomma">Vomma</option>
                    <option value="Color">Color</option>
                </select>
            </div>
            <div class="control-group" id="heatmap-coloring-control">
                <label for="heatmap_coloring_mode">Heatmap Colors:</label>
                <select id="heatmap_coloring_mode" title="Global uses one scale across all expirations. Per Expiration rescales each expiration column independently.">
                    <option value="Global" selected>Global</option>
                    <option value="Per Expiration">Per Expiration</option>
                </select>
            </div>
        </div>
        
        <div class="chart-grid" id="chart-grid">
            <div class="price-chart-container">
                <div class="tv-toolbar-container" id="tv-toolbar-container"></div>
                <div class="chart-container" id="price-chart"></div>
                <div class="tv-sub-pane" id="rsi-pane" style="display:none">
                    <div class="tv-sub-pane-header">RSI 14</div>
                    <div id="rsi-chart" style="height:110px"></div>
                </div>
                <div class="tv-sub-pane" id="macd-pane" style="display:none">
                    <div class="tv-sub-pane-header">MACD (12,26,9)</div>
                    <div id="macd-chart" style="height:120px"></div>
                </div>
                <div class="tv-sub-pane" id="arv-pane" style="display:none">
                    <div class="tv-sub-pane-header">ARV 20 (Ann. %)</div>
                    <div id="arv-chart" style="height:110px"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let charts = {};
        let updateInterval;
        let callColor = '#00FF00';
        let putColor = '#FF0000';
        let maxLevelColor = '#800080';
        let lastData = {}; // Store last received data
        let lastPriceData = null; // Price chart data stored separately (fetched via /update_price)
        let updateInProgress = false;
        let pendingFullUpdate = false;
        let pendingHeatmapOnlyUpdate = false;
        let expirationsLoading = false;
        let expirationsTicker = null;  // ticker whose expirations populate the expiry list
        let lastExpirationsAttemptMs = 0;
        const EXPIRATIONS_RETRY_MS = 5000;
        let consecutiveUpdateFailures = 0;
        const UPDATE_FAILURES_BEFORE_BACKOFF = 3;
        let updateBackoffUntilMs = 0;
        let expirationsLoadedEtDate = null;  // ET trading date the expiry list was loaded on
        const EXPIRATIONS_ROLLOVER_RETRY_MS = 30000;
        let latestExpirationsRequestId = 0;

        // How the expiry selection was made. A rule ('0dte', 'thisWk', '1wk', '2wk', '1mo',
        // 'all') is re-applied whenever the list reloads (new day, new ticker), so it rolls
        // forward by itself: 0DTE always shows the nearest expiry, "This Wk" always the rest
        // of the current week. 'fixed' dates were hand-picked: they stay as picked, drop off
        // as they expire, and once all are gone the selection is left empty.
        const EXPIRY_RULE_MODES = ['0dte', 'thisWk', '1wk', '2wk', '1mo', 'all'];
        const EXPIRY_MODE_STORAGE_KEY = 'ezoptions.expirySelectionMode';
        let expirySelectionMode = '0dte';
        try {
            const savedMode = localStorage.getItem(EXPIRY_MODE_STORAGE_KEY);
            if (EXPIRY_RULE_MODES.includes(savedMode)) expirySelectionMode = savedMode;
        } catch (e) {}
        let expiredSelectionNotice = '';  // shown in the dropdown while an expired pick is cleared

        function setExpirySelectionMode(mode) {
            expirySelectionMode = mode;
            try {
                // Only rules survive a page reload; a fresh page starts from 0DTE otherwise
                if (EXPIRY_RULE_MODES.includes(mode)) localStorage.setItem(EXPIRY_MODE_STORAGE_KEY, mode);
                else localStorage.removeItem(EXPIRY_MODE_STORAGE_KEY);
            } catch (e) {}
        }

        function addDaysToDateStr(dateStr, days) {
            const [y, m, d] = dateStr.split('-').map(Number);
            return new Date(Date.UTC(y, m - 1, d + days)).toISOString().slice(0, 10);
        }

        // Friday of this week (next Friday on a weekend) plus `weeksAhead` weeks, ET-based
        function fridayCutoffStr(todayEt, weeksAhead) {
            const [y, m, d] = todayEt.split('-').map(Number);
            const dow = new Date(Date.UTC(y, m - 1, d)).getUTCDay();
            return addDaysToDateStr(todayEt, (5 - dow + 7) % 7 + weeksAhead * 7);
        }

        // Dates a selection rule picks from the listed (sorted, unexpired) expiries
        function expiryDatesForRule(mode, dates) {
            if (!dates.length) return [];
            const todayEt = etTodayStr();
            let cutoff;
            switch (mode) {
                case 'all': return dates.slice();
                case 'thisWk': cutoff = fridayCutoffStr(todayEt, 0); break;
                case '1wk': cutoff = fridayCutoffStr(todayEt, 1); break;
                case '2wk': cutoff = fridayCutoffStr(todayEt, 2); break;
                case '1mo': cutoff = addDaysToDateStr(todayEt, 30); break;
                default: return [dates[0]];  // 0DTE: today's expiry, or the next one when none today
            }
            const picked = dates.filter(date => date <= cutoff);
            return picked.length ? picked : [dates[0]];
        }

        // Expiries are ET dates; the browser's local date can differ near midnight.
        function etTodayStr() {
            return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
        }

        // Auto-update keeps running when polls fail (Schwab maintenance windows, token
        // refreshes, an expiry that just dropped off) but backs off instead of pausing, so
        // a page left open overnight recovers by itself.
        function noteUpdateFailure() {
            consecutiveUpdateFailures++;
            if (consecutiveUpdateFailures < UPDATE_FAILURES_BEFORE_BACKOFF) return;
            const steps = consecutiveUpdateFailures - UPDATE_FAILURES_BEFORE_BACKOFF;
            updateBackoffUntilMs = Date.now() + Math.min(60000, 5000 * Math.pow(2, steps));
            // The failure may be a selected expiry that is no longer listed
            if (!expirationsLoading && Date.now() - lastExpirationsAttemptMs >= EXPIRATIONS_ROLLOVER_RETRY_MS) {
                loadExpirations();
            }
        }

        function autoUpdateTick() {
            if (Date.now() < updateBackoffUntilMs) return;
            updateData();
        }
        let isStreaming = true;
        let savedScrollPosition = 0; // Track scroll position
        let chartContainerCache = {}; // Cache for chart containers to prevent recreation

        // TradingView Lightweight Charts instances for the price chart
        let tvPriceChart = null;
        let tvCandleSeries = null;
        let tvVolumeSeries = null;
        let tvResizeObserver = null;
        // Indicator series references
        let tvIndicatorSeries = {};
        // Sub-pane charts for flow indicators, RSI, MACD, and ARV
        let tvRsiChart = null, tvRsiSeries = null;
        let tvMacdChart = null, tvMacdSeries = {};
        let tvArvChart = null, tvArvSeries = null;
        // Persist active indicators across data refreshes
        let tvActiveInds = new Set();
        // Per-indicator configurable parameters (persisted with settings)
        let tvIndicatorParams = {
            smc:    { swing_len: 5, show_ob: true, show_fvg: true, show_bos: true, show_swings: true, show_eqhl: true },
            bb:     { period: 20, mult: 2 },
            fbb:    { length: 200, mult: 3 },
            rsi:    { period: 14 },
            macd:   { fast: 12, slow: 26, signal: 9 },
            atr:    { period: 14 },
            sma9:   { period: 9 },   sma20:  { period: 20 },  sma50:  { period: 50 },
            sma100: { period: 100 }, sma200: { period: 200 },
            ema9:   { period: 9 },   ema21:  { period: 21 },  ema50:  { period: 50 },
            ema100: { period: 100 }, ema200: { period: 200 },
            wma20:  { period: 20 },  wma50:  { period: 50 },
        };
        // SMC canvas draw state
        let tvSmcData = null;
        let tvSmcDrawPending = false;
        // Max bars the Schwab API can supply for the current timeframe (updated on each price fetch)
        let tvMaxIndicatorBars = 3900;
        // Auto-range: when true, chart fits all data on every update; when false, zoom/pan is preserved
        let tvAutoRange = false;
        // Time-scale sync state
        let tvSyncHandlers = [], tvSyncingTimeScale = false;
        // Drawing state
        let tvDrawMode = null;          // null | 'hline' | 'trendline' | 'rect' | 'text'
        let tvDrawStart = null;         // {price, time, x, y} of first click
        let tvDrawings = [];            // list of drawn series/line objects for undo/clear
        let tvDrawingDefs = [];         // serializable drawing definitions — survive full re-renders
        let tvLastCandles = [];         // current-day display candles (for streaming OHLCV updates)
        let tvIndicatorCandles = [];    // multi-day candles for indicator warmup (SMA200, EMA, etc.)
        let tvCurrentDayStartTime = 0;  // unix seconds of current day's first candle (for daily VWAP)
        let tvLastPriceData = null;     // cache of full priceData for redraw
        // All overlay level prices (exposure, EM, drawn H-lines) — used by autoscaleInfoProvider
        let tvAllLevelPrices = [];
        // References to dynamically-added price lines (exposure levels, expected moves)
        // kept so they can be removed without a full chart rebuild
        let tvExposurePriceLines = [];
        let tvExpectedMovePriceLines = [];
        let tvLatestLevelLabelPriceLines = [];
        let tvLatestLevelLabelSignature = '';
        let tvHistoricalPoints = [];
        let tvHistoricalExpectedMoveSeries = [];
        let tvHistoricalOverlayPending = false;
        let tvHistoricalOverlayDomEventsBound = false;
        let tvHistoricalRenderedPoints = [];
        let tvHistoricalHoverBuckets = new Map();
        let tvShowLatestLevelLines = true;
            let tvPriceAboveBubbles = false;
        const tvHistoricalOverlayMaxVisible = 1200;
        const tvHistoricalHoverBucketSize = 48;
        // Track the active ticker so we can reset chart state on ticker change
        let tvLastTicker = null;
        // When true, the next render will call fitContent() regardless of tvAutoRange
        let tvForceFit = false;
        // EventSource for real-time price streaming from /price_stream/<ticker>
        let priceEventSource = null;
        let priceStreamTicker = null;
        // Debounce timer for indicator refresh on intra-minute quote ticks
        let tvIndicatorRefreshTimer = null;
        // Candle close countdown timer
        let candleCloseTimerInterval = null;
        // Live price from the streamer (null until first quote arrives)
        let livePrice = null;
        let lastPriceInfo = null;
        let marketContextTimer = null;
        // Debounce timer for Plotly price-line updates (avoid flooding relayout calls)
        let plotlyPriceUpdateTimer = null;
        let currentTheme = 'dark';

        const PRICE_LEVEL_EMPTY_FALLBACK_MS = 500;
        let lastStablePriceLevelTypes = [];
        let priceLevelEmptySince = 0;
        let priceLevelClearRefreshTimer = null;

        function readSelectedPriceLevelTypes(root = document) {
            try {
                return Array.from(root.querySelectorAll('.levels-option input[type="checkbox"]:checked')).map(cb => cb.value);
            } catch(e) {
                return [];
            }
        }

        function rememberStablePriceLevelTypes(levels) {
            if (!levels.length) return;
            lastStablePriceLevelTypes = levels.slice();
            priceLevelEmptySince = 0;
            if (priceLevelClearRefreshTimer) {
                clearTimeout(priceLevelClearRefreshTimer);
                priceLevelClearRefreshTimer = null;
            }
        }

        function isPriceLevelEmptyFallbackActive() {
            return !!(
                priceLevelEmptySince &&
                lastStablePriceLevelTypes.length &&
                Date.now() - priceLevelEmptySince < PRICE_LEVEL_EMPTY_FALLBACK_MS
            );
        }

        function getSelectedPriceLevelTypes(options = {}) {
            const root = options.root || document;
            const levels = readSelectedPriceLevelTypes(root);
            if (root === document && levels.length) {
                rememberStablePriceLevelTypes(levels);
            }
            if (root === document && !levels.length && options.allowStableFallback !== false && isPriceLevelEmptyFallbackActive()) {
                return lastStablePriceLevelTypes.slice();
            }
            return levels;
        }
        window.getSelectedPriceLevelTypes = getSelectedPriceLevelTypes;

        function notePriceLevelSelectionChanged() {
            const levels = readSelectedPriceLevelTypes();
            if (levels.length) {
                rememberStablePriceLevelTypes(levels);
            } else if (!priceLevelEmptySince) {
                priceLevelEmptySince = Date.now();
            }
            return levels;
        }

        function schedulePriceLevelEmptyRefresh() {
            if (priceLevelClearRefreshTimer) return;
            const elapsed = priceLevelEmptySince ? Date.now() - priceLevelEmptySince : PRICE_LEVEL_EMPTY_FALLBACK_MS;
            const delay = Math.max(25, PRICE_LEVEL_EMPTY_FALLBACK_MS - elapsed + 25);
            priceLevelClearRefreshTimer = setTimeout(() => {
                priceLevelClearRefreshTimer = null;
                refreshPriceLevelsOnly();
            }, delay);
        }

        function getShowLatestLevelLinesSetting() {
            const toggle = document.getElementById('show_latest_level_lines');
            return toggle ? toggle.checked : tvShowLatestLevelLines !== false;
        }
        window.getShowLatestLevelLinesSetting = getShowLatestLevelLinesSetting;

            function getPriceAboveBubblesSetting() {
                const toggle = document.getElementById('price_above_bubbles');
                return toggle ? toggle.checked : tvPriceAboveBubbles === true;
            }
            window.getPriceAboveBubblesSetting = getPriceAboveBubblesSetting;

        function setShowLatestLevelLinesSetting(enabled) {
            tvShowLatestLevelLines = enabled !== false;
            const toggle = document.getElementById('show_latest_level_lines');
            if (toggle && toggle.checked !== tvShowLatestLevelLines) {
                toggle.checked = tvShowLatestLevelLines;
            }
            if (lastPriceData && typeof lastPriceData === 'object') {
                lastPriceData.show_latest_level_lines = tvShowLatestLevelLines;
            }
            if (tvPriceChart) {
                scheduleTVHistoricalOverlayDraw();
            }
        }

            function setPriceAboveBubblesSetting(enabled, syncPopout = true) {
                tvPriceAboveBubbles = enabled === true;
                const toggle = document.getElementById('price_above_bubbles');
                if (toggle && toggle.checked !== tvPriceAboveBubbles) {
                    toggle.checked = tvPriceAboveBubbles;
                }
                if (lastPriceData && typeof lastPriceData === 'object') {
                    lastPriceData.price_above_bubbles = tvPriceAboveBubbles;
                }
                if (tvPriceChart) {
                    scheduleTVHistoricalOverlayDraw();
                }
                if (syncPopout && lastPriceData && typeof pushDataToPopout === 'function') {
                    pushDataToPopout('price-chart');
                }
            }


        const BASE_THEME = {
            '--app-bg': '#0f131a',
            '--panel-bg': '#1a212c',
            '--panel-bg-alt': '#222b38',
            '--panel-bg-strong': '#2a3444',
            '--panel-hover': '#344155',
            '--chart-bg': '#10161f',
            '--chart-bg-alt': '#171f2b',
            '--border-color': '#3a4659',
            '--text-primary': '#eef3fb',
            '--text-secondary': '#c7d1e0',
            '--text-muted': '#8e9cb1',
            '--accent-color': '#5ab0ff',
            '--accent-soft': 'rgba(90, 176, 255, 0.18)',
            '--slider-color': '#4dd5a6',
            '--good-color': '#37d67a',
            '--bad-color': '#ff6675',
            '--toolbar-bg': '#151d29',
            '--toolbar-button-bg': '#243142',
            '--toolbar-button-hover': '#304157',
            '--toolbar-button-active': '#1d5fa6',
            '--toolbar-button-danger': '#5f2428',
            '--grid-color': '#2b3748',
            '--crosshair-color': '#607188',
            '--tooltip-bg': 'linear-gradient(180deg, rgba(30, 36, 46, 0.97), rgba(14, 18, 24, 0.99))',
            '--tooltip-border': 'rgba(255, 255, 255, 0.08)',
            '--error-bg': '#c84d57',
            '--success-bg': '#2c8f54',
            '--button-shadow': '0 10px 28px rgba(0, 0, 0, 0.18)',
            '--overlay-shadow': '0 14px 36px rgba(0, 0, 0, 0.38)',
            '--match-em-bg': '#243142',
            '--match-em-text': '#9eb3d2',
            '--match-em-border': '#5c6d86',
            '--match-em-active-bg': '#123d38',
            '--match-em-active-text': '#7ef0ca',
            '--match-em-active-border': '#2e9b82',
            '--floating-button-bg': 'rgba(25, 32, 42, 0.88)',
            '--floating-button-hover': 'rgba(58, 72, 92, 0.96)',
        };

        const THEME_PRESETS = {
            dark: {
                '--app-bg': '#000000',
                '--panel-bg': '#0b0b0d',
                '--panel-bg-alt': '#121317',
                '--panel-bg-strong': '#1b1d22',
                '--panel-hover': '#262930',
                '--chart-bg': '#050607',
                '--chart-bg-alt': '#0f1114',
                '--border-color': '#2f333b',
                '--text-primary': '#f6f7f9',
                '--text-secondary': '#d8dbe2',
                '--text-muted': '#8a909b',
                '--accent-color': '#ffb84d',
                '--accent-soft': 'rgba(255, 184, 77, 0.18)',
                '--slider-color': '#41d98a',
                '--good-color': '#41d98a',
                '--bad-color': '#ff6878',
                '--toolbar-bg': '#08090b',
                '--toolbar-button-bg': '#181a20',
                '--toolbar-button-hover': '#252830',
                '--toolbar-button-active': '#8f5a11',
                '--toolbar-button-danger': '#5f2428',
                '--grid-color': '#1e2128',
                '--crosshair-color': '#666c79',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(18, 18, 22, 0.98), rgba(4, 4, 5, 0.99))',
                '--tooltip-border': 'rgba(255, 255, 255, 0.08)',
                '--button-shadow': '0 16px 36px rgba(0, 0, 0, 0.42)',
                '--overlay-shadow': '0 18px 48px rgba(0, 0, 0, 0.55)',
                '--match-em-bg': '#231e14',
                '--match-em-text': '#ffd18b',
                '--match-em-border': '#91641f',
                '--match-em-active-bg': '#0f3524',
                '--match-em-active-text': '#7df0b1',
                '--match-em-active-border': '#2ba262',
                '--floating-button-bg': 'rgba(14, 15, 18, 0.92)',
                '--floating-button-hover': 'rgba(34, 37, 43, 0.98)',
            },
            light: {
                '--app-bg': '#f7f1e6',
                '--panel-bg': '#fffdf8',
                '--panel-bg-alt': '#f1e8d8',
                '--panel-bg-strong': '#e4d6be',
                '--panel-hover': '#dbc9ac',
                '--chart-bg': '#fffefb',
                '--chart-bg-alt': '#fbf6ee',
                '--border-color': '#cbb89b',
                '--text-primary': '#1f1b16',
                '--text-secondary': '#473c30',
                '--text-muted': '#817160',
                '--accent-color': '#2667c9',
                '--slider-color': '#1aa36c',
                '--good-color': '#15885b',
                '--bad-color': '#c95061',
                '--toolbar-bg': '#f2e7d5',
                '--toolbar-button-bg': '#e5d5bb',
                '--toolbar-button-hover': '#d8c4a3',
                '--toolbar-button-active': '#2667c9',
                '--grid-color': '#e4d6bf',
                '--crosshair-color': '#9f8d76',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(255, 252, 246, 0.98), rgba(243, 232, 213, 0.98))',
                '--tooltip-border': 'rgba(31, 27, 22, 0.10)',
                '--button-shadow': '0 12px 30px rgba(76, 54, 28, 0.10)',
                '--overlay-shadow': '0 18px 42px rgba(81, 59, 31, 0.14)',
                '--match-em-bg': '#ebdfca',
                '--match-em-text': '#61523f',
                '--match-em-border': '#b89f78',
                '--match-em-active-bg': '#d9f0e4',
                '--match-em-active-text': '#116e4c',
                '--match-em-active-border': '#37a57a',
                '--floating-button-bg': 'rgba(255, 248, 239, 0.96)',
                '--floating-button-hover': 'rgba(241, 229, 207, 0.98)',
            },
            ocean: {
                '--app-bg': '#03141d',
                '--panel-bg': '#082430',
                '--panel-bg-alt': '#0f3442',
                '--panel-bg-strong': '#154758',
                '--panel-hover': '#1d5b6f',
                '--chart-bg': '#041821',
                '--chart-bg-alt': '#0a212d',
                '--border-color': '#2d6878',
                '--text-primary': '#e7fcff',
                '--text-secondary': '#bde4eb',
                '--text-muted': '#77b3bf',
                '--accent-color': '#2bd4ff',
                '--accent-soft': 'rgba(43, 212, 255, 0.18)',
                '--slider-color': '#1ee8b7',
                '--good-color': '#1ee8b7',
                '--bad-color': '#ff7f88',
                '--toolbar-bg': '#061c28',
                '--toolbar-button-bg': '#123746',
                '--toolbar-button-hover': '#1b4b5f',
                '--toolbar-button-active': '#0e84a3',
                '--grid-color': '#204857',
                '--crosshair-color': '#6ca0aa',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(10, 38, 48, 0.97), rgba(3, 18, 24, 0.99))',
                '--match-em-bg': '#113848',
                '--match-em-text': '#9befff',
                '--match-em-border': '#32a6c8',
                '--match-em-active-bg': '#093833',
                '--match-em-active-text': '#83ffe6',
                '--match-em-active-border': '#1dc6a8',
            },
            midnight: {
                '--app-bg': '#06040d',
                '--panel-bg': '#120b1f',
                '--panel-bg-alt': '#1b1230',
                '--panel-bg-strong': '#271946',
                '--panel-hover': '#35245d',
                '--chart-bg': '#0a0714',
                '--chart-bg-alt': '#130c22',
                '--border-color': '#4a3674',
                '--text-primary': '#f5efff',
                '--text-secondary': '#d8c8ff',
                '--text-muted': '#9f8bc9',
                '--accent-color': '#9f7bff',
                '--accent-soft': 'rgba(159, 123, 255, 0.20)',
                '--slider-color': '#5bd7ff',
                '--toolbar-bg': '#0d0918',
                '--toolbar-button-bg': '#24183d',
                '--toolbar-button-hover': '#342458',
                '--toolbar-button-active': '#5f43c7',
                '--grid-color': '#2d2147',
                '--crosshair-color': '#7f70a5',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(28, 19, 45, 0.98), rgba(8, 5, 16, 0.99))',
                '--match-em-bg': '#2a1f4a',
                '--match-em-text': '#d8c8ff',
                '--match-em-border': '#8266d3',
            },
            arctic: {
                '--app-bg': '#edf9ff',
                '--panel-bg': '#fcfeff',
                '--panel-bg-alt': '#dff2fb',
                '--panel-bg-strong': '#c7e7f4',
                '--panel-hover': '#b1dced',
                '--chart-bg': '#f9fdff',
                '--chart-bg-alt': '#eff9fd',
                '--border-color': '#abd2e1',
                '--text-primary': '#0e2330',
                '--text-secondary': '#23485e',
                '--text-muted': '#6e95a8',
                '--accent-color': '#00a7e1',
                '--accent-soft': 'rgba(0, 167, 225, 0.18)',
                '--slider-color': '#11c68d',
                '--good-color': '#11b57f',
                '--bad-color': '#de6172',
                '--toolbar-bg': '#e7f6fd',
                '--toolbar-button-bg': '#d0ecf8',
                '--toolbar-button-hover': '#bbe2f2',
                '--toolbar-button-active': '#00a7e1',
                '--grid-color': '#d0e9f3',
                '--crosshair-color': '#89b7c7',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(253, 255, 255, 0.98), rgba(226, 244, 251, 0.98))',
                '--tooltip-border': 'rgba(14, 35, 48, 0.10)',
            },
            cobalt: {
                '--app-bg': '#06102b',
                '--panel-bg': '#0d1b46',
                '--panel-bg-alt': '#132867',
                '--panel-bg-strong': '#1b3988',
                '--panel-hover': '#2750ac',
                '--chart-bg': '#071435',
                '--chart-bg-alt': '#0d1e49',
                '--border-color': '#436abf',
                '--text-primary': '#eef4ff',
                '--text-secondary': '#c7d8ff',
                '--text-muted': '#8fa9e7',
                '--accent-color': '#4aa3ff',
                '--accent-soft': 'rgba(74, 163, 255, 0.18)',
                '--slider-color': '#61e6d4',
                '--toolbar-bg': '#0b173d',
                '--toolbar-button-bg': '#183171',
                '--toolbar-button-hover': '#234595',
                '--toolbar-button-active': '#2d78ff',
                '--grid-color': '#24427d',
                '--crosshair-color': '#7ea1ef',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(18, 35, 82, 0.97), rgba(6, 16, 43, 0.99))',
                '--match-em-bg': '#17337a',
                '--match-em-text': '#c8d9ff',
                '--match-em-border': '#628de5',
            },
            forest: {
                '--app-bg': '#08120b',
                '--panel-bg': '#112118',
                '--panel-bg-alt': '#183024',
                '--panel-bg-strong': '#234435',
                '--panel-hover': '#315946',
                '--chart-bg': '#0c1810',
                '--chart-bg-alt': '#122117',
                '--border-color': '#3f6a51',
                '--text-primary': '#eff8f1',
                '--text-secondary': '#cee3d3',
                '--text-muted': '#8daf97',
                '--accent-color': '#8fdc5b',
                '--accent-soft': 'rgba(143, 220, 91, 0.18)',
                '--slider-color': '#4be39c',
                '--good-color': '#4be39c',
                '--bad-color': '#ff7f7a',
                '--toolbar-bg': '#0d1b12',
                '--toolbar-button-bg': '#20382a',
                '--toolbar-button-hover': '#2d4f3a',
                '--toolbar-button-active': '#4c9056',
                '--grid-color': '#263f30',
                '--crosshair-color': '#76967f',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(24, 42, 28, 0.97), rgba(8, 18, 11, 0.99))',
            },
            sunset: {
                '--app-bg': '#24110a',
                '--panel-bg': '#3a1d13',
                '--panel-bg-alt': '#542a1a',
                '--panel-bg-strong': '#6e3821',
                '--panel-hover': '#8a4929',
                '--chart-bg': '#2c150d',
                '--chart-bg-alt': '#381b10',
                '--border-color': '#9b5c35',
                '--text-primary': '#fff2e7',
                '--text-secondary': '#f3cfb7',
                '--text-muted': '#d49d78',
                '--accent-color': '#ff8a3d',
                '--accent-soft': 'rgba(255, 138, 61, 0.20)',
                '--slider-color': '#ffd166',
                '--good-color': '#ffca6b',
                '--bad-color': '#ff7c72',
                '--toolbar-bg': '#31170f',
                '--toolbar-button-bg': '#60311e',
                '--toolbar-button-hover': '#7d4127',
                '--toolbar-button-active': '#c56224',
                '--grid-color': '#61331f',
                '--crosshair-color': '#d69369',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(78, 40, 24, 0.97), rgba(27, 12, 7, 0.99))',
                '--match-em-bg': '#663317',
                '--match-em-text': '#ffd4ad',
                '--match-em-border': '#d47c42',
                '--match-em-active-bg': '#5a4310',
                '--match-em-active-text': '#ffe080',
                '--match-em-active-border': '#e1a62e',
            },
            ruby: {
                '--app-bg': '#17050c',
                '--panel-bg': '#2a0c17',
                '--panel-bg-alt': '#3a1021',
                '--panel-bg-strong': '#531733',
                '--panel-hover': '#682045',
                '--chart-bg': '#1e0710',
                '--chart-bg-alt': '#2b0d18',
                '--border-color': '#7a3553',
                '--text-primary': '#fff0f5',
                '--text-secondary': '#f3c7d6',
                '--text-muted': '#ce94aa',
                '--accent-color': '#ff5c93',
                '--accent-soft': 'rgba(255, 92, 147, 0.18)',
                '--slider-color': '#ff8e5d',
                '--good-color': '#ffb067',
                '--bad-color': '#ff6f8d',
                '--toolbar-bg': '#230913',
                '--toolbar-button-bg': '#45152c',
                '--toolbar-button-hover': '#5d2040',
                '--toolbar-button-active': '#b52f67',
                '--grid-color': '#4c1b32',
                '--crosshair-color': '#bf6f91',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(68, 18, 39, 0.97), rgba(22, 6, 12, 0.99))',
            },
            amber: {
                '--app-bg': '#151006',
                '--panel-bg': '#241b0b',
                '--panel-bg-alt': '#35270f',
                '--panel-bg-strong': '#4c3612',
                '--panel-hover': '#664918',
                '--chart-bg': '#1b1408',
                '--chart-bg-alt': '#251b0b',
                '--border-color': '#866325',
                '--text-primary': '#fff7e4',
                '--text-secondary': '#f0dda7',
                '--text-muted': '#c6a960',
                '--accent-color': '#ffbf2f',
                '--accent-soft': 'rgba(255, 191, 47, 0.18)',
                '--slider-color': '#ffd84a',
                '--good-color': '#ffe169',
                '--bad-color': '#ff8d5c',
                '--toolbar-bg': '#1f1708',
                '--toolbar-button-bg': '#432f11',
                '--toolbar-button-hover': '#5b4217',
                '--toolbar-button-active': '#b17d11',
                '--grid-color': '#4e3915',
                '--crosshair-color': '#b5944d',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(71, 51, 16, 0.97), rgba(18, 13, 5, 0.99))',
            },
            neon: {
                '--app-bg': '#02060c',
                '--panel-bg': '#07111b',
                '--panel-bg-alt': '#0b1724',
                '--panel-bg-strong': '#102133',
                '--panel-hover': '#17314b',
                '--chart-bg': '#030a12',
                '--chart-bg-alt': '#08111d',
                '--border-color': '#1d4565',
                '--text-primary': '#f3feff',
                '--text-secondary': '#c4f5ff',
                '--text-muted': '#76a7bf',
                '--accent-color': '#00eaff',
                '--accent-soft': 'rgba(0, 234, 255, 0.22)',
                '--slider-color': '#8fff3d',
                '--good-color': '#8fff3d',
                '--bad-color': '#ff4fa3',
                '--toolbar-bg': '#050d17',
                '--toolbar-button-bg': '#0d1927',
                '--toolbar-button-hover': '#14263a',
                '--toolbar-button-active': '#5628a8',
                '--toolbar-button-danger': '#4a1530',
                '--grid-color': '#12314a',
                '--crosshair-color': '#4fb6d3',
                '--tooltip-bg': 'linear-gradient(180deg, rgba(13, 27, 39, 0.98), rgba(2, 8, 14, 0.99))',
                '--tooltip-border': 'rgba(0, 234, 255, 0.20)',
                '--button-shadow': '0 0 0 1px rgba(0, 234, 255, 0.14), 0 18px 42px rgba(0, 0, 0, 0.48), 0 0 26px rgba(0, 234, 255, 0.10)',
                '--overlay-shadow': '0 0 0 1px rgba(255, 79, 163, 0.16), 0 22px 54px rgba(0, 0, 0, 0.58), 0 0 32px rgba(0, 234, 255, 0.12)',
                '--match-em-bg': '#221526',
                '--match-em-text': '#ffb8e2',
                '--match-em-border': '#9c4f8b',
                '--match-em-active-bg': '#10290d',
                '--match-em-active-text': '#b8ff8f',
                '--match-em-active-border': '#6ae232',
                '--floating-button-bg': 'rgba(5, 13, 22, 0.94)',
                '--floating-button-hover': 'rgba(16, 31, 48, 0.98)',
            },
        };

        function getThemePreset(themeName) {
            return { ...BASE_THEME, ...(THEME_PRESETS[themeName] || {}) };
        }

        function getThemeValue(name, fallback) {
            const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
            return value || fallback;
        }

        function getThemeColors() {
            return {
                appBg: getThemeValue('--app-bg', '#0f131a'),
                panelBg: getThemeValue('--panel-bg', '#1a212c'),
                panelBgAlt: getThemeValue('--panel-bg-alt', '#222b38'),
                chartBg: getThemeValue('--chart-bg', '#10161f'),
                borderColor: getThemeValue('--border-color', '#3a4659'),
                textPrimary: getThemeValue('--text-primary', '#eef3fb'),
                textSecondary: getThemeValue('--text-secondary', '#c7d1e0'),
                textMuted: getThemeValue('--text-muted', '#8e9cb1'),
                accentColor: getThemeValue('--accent-color', '#5ab0ff'),
                gridColor: getThemeValue('--grid-color', '#2b3748'),
                crosshairColor: getThemeValue('--crosshair-color', '#607188'),
            };
        }

        function buildLightweightThemeOptions() {
            const theme = getThemeColors();
            return {
                layout: {
                    background: { color: theme.chartBg },
                    textColor: theme.textSecondary,
                    fontFamily: 'Arial, sans-serif',
                },
                grid: {
                    vertLines: { color: theme.gridColor },
                    horzLines: { color: theme.gridColor },
                },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal,
                    vertLine: { color: theme.crosshairColor, labelBackgroundColor: theme.panelBgAlt },
                    horzLine: { color: theme.crosshairColor, labelBackgroundColor: theme.panelBgAlt },
                },
                rightPriceScale: { borderColor: theme.borderColor },
                timeScale: { borderColor: theme.borderColor },
            };
        }

        function applyThemeToPlotlyLayout(layout) {
            const theme = getThemeColors();
            layout.plot_bgcolor = theme.chartBg;
            layout.paper_bgcolor = theme.panelBg;
            layout.font = Object.assign({}, layout.font || {}, { color: theme.textSecondary });
            if (layout.title) {
                layout.title.font = Object.assign({}, layout.title.font || {}, { color: theme.textSecondary });
            }
            if (layout.legend) {
                layout.legend.bgcolor = theme.panelBgAlt;
                layout.legend.font = Object.assign({}, layout.legend.font || {}, { color: theme.textSecondary });
            }
            if (layout.hoverlabel) {
                layout.hoverlabel.bgcolor = theme.panelBgAlt;
                layout.hoverlabel.bordercolor = theme.borderColor;
            }
            ['xaxis', 'yaxis', 'xaxis2', 'yaxis2'].forEach(axisKey => {
                const axis = layout[axisKey];
                if (!axis) return;
                axis.gridcolor = theme.gridColor;
                axis.linecolor = theme.borderColor;
                axis.zerolinecolor = theme.gridColor;
                axis.tickcolor = theme.textMuted;
                axis.tickfont = Object.assign({}, axis.tickfont || {}, { color: theme.textSecondary });
                if (axis.title && typeof axis.title === 'object') {
                    axis.title.font = Object.assign({}, axis.title.font || {}, { color: theme.textSecondary });
                }
                if (axis.title_font) {
                    axis.title_font.color = theme.textSecondary;
                }
            });
        }

        function buildPlotlyThemeRelayout(div) {
            const theme = getThemeColors();
            const relayout = {
                plot_bgcolor: theme.chartBg,
                paper_bgcolor: theme.panelBg,
                'font.color': theme.textSecondary,
                'hoverlabel.bgcolor': theme.panelBgAlt,
                'hoverlabel.bordercolor': theme.borderColor,
                'legend.bgcolor': theme.panelBgAlt,
                'legend.font.color': theme.textSecondary,
                'title.font.color': theme.textSecondary,
            };
            if (!div || !div._fullLayout) return relayout;
            ['xaxis', 'yaxis', 'xaxis2', 'yaxis2'].forEach(axisKey => {
                if (!div._fullLayout[axisKey]) return;
                relayout[`${axisKey}.gridcolor`] = theme.gridColor;
                relayout[`${axisKey}.linecolor`] = theme.borderColor;
                relayout[`${axisKey}.zerolinecolor`] = theme.gridColor;
                relayout[`${axisKey}.tickfont.color`] = theme.textSecondary;
                relayout[`${axisKey}.tickcolor`] = theme.textMuted;
                relayout[`${axisKey}.title.font.color`] = theme.textSecondary;
            });
            return relayout;
        }

        function refreshRenderedChartsForTheme() {
            if (tvPriceChart) {
                try { tvPriceChart.applyOptions(buildLightweightThemeOptions()); } catch (e) {}
            }
            if (tvRsiChart) {
                try { tvRsiChart.applyOptions(buildLightweightThemeOptions()); } catch (e) {}
            }
            if (tvMacdChart) {
                try { tvMacdChart.applyOptions(buildLightweightThemeOptions()); } catch (e) {}
            }
            if (tvArvChart) {
                try { tvArvChart.applyOptions(buildLightweightThemeOptions()); } catch (e) {}
            }
            Object.keys(charts).forEach(key => {
                const div = getPlotlyChartElement(key);
                if (!div || !div._fullLayout || key === 'large_trades') return;
                try {
                    Plotly.relayout(div, buildPlotlyThemeRelayout(div));
                    if (key === 'heatmap') {
                        resizeHeatmapPlot();
                    } else {
                        Plotly.Plots.resize(div);
                    }
                } catch (e) {}
            });
            scheduleTVHistoricalOverlayDraw();
        }

        function buildPopoutThemePayload() {
            const vars = {};
            Object.keys(BASE_THEME).forEach(key => {
                vars[key] = getThemeValue(key, BASE_THEME[key]);
            });
            return {
                name: currentTheme,
                vars,
            };
        }
        window.buildPopoutThemePayload = buildPopoutThemePayload;

        function applyTheme(themeName) {
            if (!Object.prototype.hasOwnProperty.call(THEME_PRESETS, themeName)) {
                themeName = 'dark';
            }
            const theme = getThemePreset(themeName);
            const root = document.documentElement;
            Object.keys(theme).forEach(key => root.style.setProperty(key, theme[key]));
            currentTheme = themeName;
            document.body.dataset.theme = themeName;
            const selector = document.getElementById('theme_select');
            if (selector && selector.value !== themeName) {
                selector.value = themeName;
            }
            refreshRenderedChartsForTheme();
            if (typeof emRangeLocked !== 'undefined') {
                setEmRangeLocked(emRangeLocked);
            }
            if (typeof pushAllPopouts === 'function') {
                pushAllPopouts();
            }
        }

        // List of Plotly chart div IDs that carry a current-price line shape
        const PLOTLY_PRICE_LINE_CHARTS = [
            'gamma-chart', 'delta-chart', 'vanna-chart', 'charm-chart',
            'speed-chart', 'vomma-chart', 'color-chart',
            'options_volume-chart', 'open_interest-chart'
        ];

        /**
         * Update the current-price line (shape + annotation) on all visible Plotly charts
         * and refresh the "Current Price" text in the price-info panel.
         */
        function updateAllPlotlyPriceLines(price) {
            const priceStr = price.toFixed(2);

            PLOTLY_PRICE_LINE_CHARTS.forEach(function(id) {
                const div = document.getElementById(id);
                // Off-screen charts catch up when they scroll into view (flushPendingPlotlyChart)
                if (div && isPlotlyContainerVisible(div)) applyPlotlyPriceLine(div, price);
            });

            // Live-update the "Current Price" line in the price-info panel
            const priceInfo = document.getElementById('price-info');
            if (priceInfo) {
                const cpLine = priceInfo.querySelector('[data-live-price]');
                if (cpLine) {
                    cpLine.textContent = '$' + priceStr;
                }
                refreshMarketContextItems(lastPriceInfo, price);
            }
        }

        function applyPlotlyPriceLine(div, price) {
            if (!div || !div._fullLayout || PLOTLY_PRICE_LINE_CHARTS.indexOf(div.id) === -1) return;
            const priceStr = price.toFixed(2);
            // Each relayout redraws the chart; skip when the line already shows this price
            if (div.__livePriceStr === priceStr) return;

            const shapes = div._fullLayout.shapes || [];
            const annotations = div._fullLayout.annotations || [];
            const update = {};

            // Identify and update the price line shape.
            // add_vline produces: xref='x', yref='paper', x0===x1
            // add_hline produces: xref='paper', yref='y', y0===y1
            for (let i = 0; i < shapes.length; i++) {
                const sh = shapes[i];
                if (sh.xref === 'x' && sh.yref === 'paper' && sh.x0 === sh.x1) {
                    update['shapes[' + i + '].x0'] = price;
                    update['shapes[' + i + '].x1'] = price;
                    break;
                } else if (sh.xref === 'paper' && sh.yref === 'y' && sh.y0 === sh.y1) {
                    update['shapes[' + i + '].y0'] = price;
                    update['shapes[' + i + '].y1'] = price;
                    break;
                }
            }

            // Identify and update the price line annotation.
            // add_vline annotation: xref='x', yref='paper'
            // add_hline annotation: xref='paper', yref='y'
            for (let i = 0; i < annotations.length; i++) {
                const ann = annotations[i];
                if (ann.xref === 'x' && ann.yref === 'paper') {
                    update['annotations[' + i + '].x'] = price;
                    update['annotations[' + i + '].text'] = priceStr;
                    break;
                } else if (ann.xref === 'paper' && ann.yref === 'y') {
                    update['annotations[' + i + '].y'] = price;
                    update['annotations[' + i + '].text'] = priceStr;
                    break;
                }
            }

            div.__livePriceStr = priceStr;
            if (Object.keys(update).length > 0) {
                try { Plotly.relayout(div, update); } catch(e) {}
            }
        }

        function isMobileViewport() {
            return window.matchMedia('(max-width: 900px)').matches;
        }

        function resizeAllCharts() {
            Object.keys(charts).forEach(chartKey => {
                const chartElement = getPlotlyChartElement(chartKey);
                if (chartElement && charts[chartKey] && chartKey !== 'large_trades') {
                    try { Plotly.Plots.resize(chartElement); } catch (e) {}
                }
            });
            scheduleTVHistoricalOverlayDraw();
        }

        function syncMobilePanelButtons() {
            const filtersButton = document.getElementById('mobile-toggle-filters');
            const chartsButton = document.getElementById('mobile-toggle-charts');
            if (!filtersButton || !chartsButton) {
                return;
            }

            const filtersOpen = document.body.classList.contains('mobile-filters-open');
            const chartsOpen = document.body.classList.contains('mobile-charts-open');
            const selectedChartsCount = document.querySelectorAll('.chart-checkbox input[type="checkbox"]:checked').length;

            filtersButton.classList.toggle('active', filtersOpen);
            chartsButton.classList.toggle('active', chartsOpen);
            filtersButton.textContent = filtersOpen ? 'Hide Filters' : 'Filters';
            chartsButton.textContent = chartsOpen
                ? `Hide Charts (${selectedChartsCount})`
                : `Charts (${selectedChartsCount})`;
            filtersButton.setAttribute('aria-expanded', filtersOpen ? 'true' : 'false');
            chartsButton.setAttribute('aria-expanded', chartsOpen ? 'true' : 'false');
        }

        function applyMobileLayoutState() {
            const mobileLayout = isMobileViewport();
            document.body.classList.toggle('mobile-layout', mobileLayout);
            if (!mobileLayout) {
                document.body.classList.remove('mobile-filters-open', 'mobile-charts-open');
            }
            syncMobilePanelButtons();
            requestAnimationFrame(resizeAllCharts);
        }

        function toggleMobilePanel(panelName) {
            if (!document.body.classList.contains('mobile-layout')) {
                return;
            }

            const filtersClass = 'mobile-filters-open';
            const chartsClass = 'mobile-charts-open';

            if (panelName === 'filters') {
                document.body.classList.toggle(filtersClass, !document.body.classList.contains(filtersClass));
                document.body.classList.remove(chartsClass);
            } else if (panelName === 'charts') {
                document.body.classList.toggle(chartsClass, !document.body.classList.contains(chartsClass));
                document.body.classList.remove(filtersClass);
            }

            syncMobilePanelButtons();
            requestAnimationFrame(resizeAllCharts);
        }

        // Apply (or re-apply) the autoscaleInfoProvider so the Y-axis always fits levels
        function tvApplyAutoscale() {
            if (!tvCandleSeries) return;
            const levelPrices = tvAllLevelPrices.slice(); // snapshot
            tvCandleSeries.applyOptions({
                autoscaleInfoProvider: (original) => {
                    const res = original();
                    if (!res) return res;
                    if (levelPrices.length === 0) return res;
                    const pad = (res.priceRange.maxValue - res.priceRange.minValue) * 0.05;
                    const minVal = Math.min(res.priceRange.minValue, ...levelPrices) - pad;
                    const maxVal = Math.max(res.priceRange.maxValue, ...levelPrices) + pad;
                    return { priceRange: { minValue: minVal, maxValue: maxVal }, margins: res.margins };
                }
            });
        }

        function tvFitAll() {
            if (!tvPriceChart) return;
            // Use setTimeout so this fires after LightweightCharts finishes its own internal layout pass
            setTimeout(() => {
                try {
                    // Reset X-axis (time scale)
                    tvPriceChart.timeScale().fitContent();
                    // Reset Y-axis: re-enable auto-scaling (user dragging the price axis locks it to manual mode)
                    tvPriceChart.priceScale('right').applyOptions({ autoScale: true });
                    // Re-arm the autoscaleInfoProvider so level lines are included in the Y range
                    tvApplyAutoscale();
                    // Sub-pane charts also need their price axes reset
                    if (tvRsiChart)  tvRsiChart.priceScale('right').applyOptions({ autoScale: true });
                    if (tvMacdChart) tvMacdChart.priceScale('right').applyOptions({ autoScale: true });
                    if (tvArvChart)  tvArvChart.priceScale('right').applyOptions({ autoScale: true });
                } catch(e) {}
            }, 50);
        }

        // ── Real-time price streaming via Server-Sent Events ─────────────────
        function disconnectPriceStream() {
            if (priceEventSource) {
                priceEventSource.close();
                priceEventSource = null;
                priceStreamTicker = null;
            }
        }

        function connectPriceStream(ticker) {
            if (!ticker) return;
            const upperTicker = ticker.toUpperCase();
            // Already connected to the right ticker – nothing to do
            if (priceEventSource && priceStreamTicker === upperTicker &&
                priceEventSource.readyState !== EventSource.CLOSED) {
                return;
            }
            // Disconnect any existing connection first
            disconnectPriceStream();

            priceEventSource = new EventSource('/price_stream/' + encodeURIComponent(upperTicker));
            priceStreamTicker = upperTicker;

            priceEventSource.onmessage = function(event) {
                try {
                    const msg = JSON.parse(event.data);
                    if (!tvCandleSeries || !tvLastCandles.length) return;
                    if (msg.type === 'quote' && typeof msg.last === 'number') {
                        applyRealtimeQuote(msg.last);
                    } else if (msg.type === 'candle' && msg.time) {
                        applyRealtimeCandle(msg);
                    }
                } catch(e) {}
            };

            priceEventSource.onerror = function() {
                // Browser will auto-reconnect on error; just log it quietly
                console.debug('[PriceStream] Connection error – browser will retry.');
            };
        }

        function getTVTimeframeSeconds() {
            const tf = parseInt(document.getElementById('timeframe').value, 10);
            return (Number.isFinite(tf) && tf > 0 ? tf : 1) * 60;
        }

        // Map a timestamp (seconds) to the start of the chart candle that contains it.
        // Buckets are anchored on the last loaded candle so they follow the session-aligned
        // bars from the backend (e.g. 60m bars that start at :30). Returns null for
        // timestamps before the first candle on the chart.
        function getTVBucketStart(tsSec) {
            const tfSec = getTVTimeframeSeconds();
            const lastCandle = tvLastCandles[tvLastCandles.length - 1];
            if (!lastCandle) return Math.floor(tsSec / tfSec) * tfSec;
            const bucket = lastCandle.time + Math.floor((tsSec - lastCandle.time) / tfSec) * tfSec;
            return bucket < tvLastCandles[0].time ? null : bucket;
        }

        // The chart only shows the regular session (9:30-16:00 ET); ignore pre/post-market
        // ticks so they don't append candles the history endpoint filters out.
        function isRegularSessionTime(tsSec) {
            if ((priceStreamTicker || '').startsWith('/')) return true;
            const et = new Date(new Date(tsSec * 1000).toLocaleString('en-US', { timeZone: 'America/New_York' }));
            const day = et.getDay();
            const mins = et.getHours() * 60 + et.getMinutes();
            return day >= 1 && day <= 5 && mins >= 9 * 60 + 30 && mins < 16 * 60;
        }

        // Live data is merged into the raw (non-Heikin-Ashi) bars in tvIndicatorCandles, and the
        // displayed candle is derived from them, so Heikin-Ashi charts stay Heikin-Ashi and
        // indicators keep computing from real prices.
        function tvFindBarIndex(bars, time) {
            for (let i = bars.length - 1; i >= 0; i--) {
                if (bars[i].time === time) return i;
                if (bars[i].time < time) return -1;
            }
            return -1;
        }

        function tvUpsertRawBar(bucketTime, build) {
            const idx = tvFindBarIndex(tvIndicatorCandles, bucketTime);
            if (idx >= 0) {
                tvIndicatorCandles[idx] = build(tvIndicatorCandles[idx]);
                return tvIndicatorCandles[idx];
            }
            const bar = build(null);
            tvIndicatorCandles.push(bar);
            tvIndicatorCandles.sort((a, b) => a.time - b.time);
            return bar;
        }

        function tvDisplayBarFromRaw(raw, prevDisplay) {
            const useHA = !!(tvLastPriceData && tvLastPriceData.use_heikin_ashi);
            if (!useHA) {
                return { time: raw.time, open: raw.open, high: raw.high, low: raw.low, close: raw.close };
            }
            const haClose = (raw.open + raw.high + raw.low + raw.close) / 4;
            const haOpen = prevDisplay ? (prevDisplay.open + prevDisplay.close) / 2 : (raw.open + raw.close) / 2;
            return {
                time: raw.time,
                open: haOpen,
                high: Math.max(raw.high, haOpen, haClose),
                low: Math.min(raw.low, haOpen, haClose),
                close: haClose,
            };
        }

        function tvVolumeBar(displayBar, prevDisplay, volume) {
            const up = prevDisplay ? displayBar.close >= prevDisplay.close : displayBar.close >= displayBar.open;
            const colors = tvLastPriceData || {};
            return { time: displayBar.time, value: volume || 0, color: up ? (colors.call_color || '#00FF00') : (colors.put_color || '#FF0000') };
        }

        // Re-derive the displayed candle for `raw` (and, for Heikin-Ashi, every later candle,
        // since each HA open depends on the previous bar). Returns true if a new bar was added.
        function tvApplyRawBar(raw) {
            let idx = tvFindBarIndex(tvLastCandles, raw.time);
            const isNew = idx < 0;
            if (isNew) {
                const lastDisplay = tvLastCandles[tvLastCandles.length - 1];
                if (lastDisplay && raw.time < lastDisplay.time) return false;  // gap inside history; wait for refetch
                tvLastCandles.push(null);
                idx = tvLastCandles.length - 1;
            }
            const useHA = !!(tvLastPriceData && tvLastPriceData.use_heikin_ashi);
            const lastIdx = useHA ? tvLastCandles.length - 1 : idx;
            for (let i = idx; i <= lastIdx; i++) {
                const rawIdx = tvFindBarIndex(tvIndicatorCandles, i === idx ? raw.time : tvLastCandles[i].time);
                const rawBar = i === idx ? raw : (rawIdx >= 0 ? tvIndicatorCandles[rawIdx] : null);
                if (!rawBar) break;
                tvLastCandles[i] = tvDisplayBarFromRaw(rawBar, tvLastCandles[i - 1]);
            }
            const isLast = idx === tvLastCandles.length - 1 && lastIdx === idx;
            const volumeBar = tvVolumeBar(tvLastCandles[idx], tvLastCandles[idx - 1], raw.volume);
            if (isLast) {
                try { tvCandleSeries.update(tvLastCandles[idx]); } catch (e) {}
                if (tvVolumeSeries) { try { tvVolumeSeries.update(volumeBar); } catch (e) {} }
            } else {
                // series.update() only accepts the newest bar; an older bar needs a full reset
                try { tvCandleSeries.setData(tvLastCandles); } catch (e) {}
                if (tvVolumeSeries) {
                    const volumes = tvLastCandles.map((bar, i) => {
                        const rawIdx = tvFindBarIndex(tvIndicatorCandles, bar.time);
                        return tvVolumeBar(bar, tvLastCandles[i - 1], rawIdx >= 0 ? tvIndicatorCandles[rawIdx].volume : 0);
                    });
                    try { tvVolumeSeries.setData(volumes); } catch (e) {}
                }
            }
            return isNew;
        }

        /**
         * Update or extend the chart's current candle from a real-time last price.
         * Buckets by the selected timeframe so higher-timeframe charts aren't fed 1m bars.
         */
        function applyRealtimeQuote(last) {
            // Track live price and debounce Plotly chart updates
            livePrice = last;
            clearTimeout(plotlyPriceUpdateTimer);
            plotlyPriceUpdateTimer = setTimeout(function() { updateAllPlotlyPriceLines(last); }, 500);

            const nowSec = Math.floor(Date.now() / 1000);
            if (!tvCandleSeries || !tvLastCandles.length || !isRegularSessionTime(nowSec)) return;
            const bucketTime = getTVBucketStart(nowSec);
            if (bucketTime === null || bucketTime < tvLastCandles[tvLastCandles.length - 1].time) return;

            const raw = tvUpsertRawBar(bucketTime, bar => bar
                ? { ...bar, high: Math.max(bar.high, last), low: Math.min(bar.low, last), close: last }
                : { time: bucketTime, open: last, high: last, low: last, close: last, volume: 0 });
            const isNewBar = tvApplyRawBar(raw);

            if (tvActiveInds.size > 0) {
                clearTimeout(tvIndicatorRefreshTimer);
                if (isNewBar) {
                    applyIndicators(tvIndicatorCandles, tvActiveInds);
                } else {
                    // Debounce indicator refresh to at most once every 2 seconds on tick updates
                    tvIndicatorRefreshTimer = setTimeout(() => applyIndicators(tvIndicatorCandles, tvActiveInds), 2000);
                }
            }
            // The price scale may have moved; keep bubbles/level lines pinned to their prices
            scheduleTVHistoricalOverlayDraw();
        }

        // Streamed 1-minute volumes for the current higher-timeframe bucket, so repeated
        // minute bars aren't double-counted. Reset whenever history is reloaded.
        let tvStreamBucketState = null;  // { bucketTime, baseVolume, minutes: {time: volume} }

        /**
         * Apply a completed 1-minute candle from CHART_EQUITY streaming, merged into the
         * candle of the selected timeframe.
         */
        function applyRealtimeCandle(candle) {
            if (!tvCandleSeries || !tvLastCandles.length || !isRegularSessionTime(candle.time)) return;
            const bucketTime = getTVBucketStart(candle.time);
            if (bucketTime === null) return;
            const minuteBars = getTVTimeframeSeconds() <= 60;

            const raw = tvUpsertRawBar(bucketTime, bar => {
                if (minuteBars) {
                    // The official completed bar replaces the one built from quotes
                    return { time: bucketTime, open: candle.open, high: candle.high, low: candle.low,
                             close: candle.close, volume: candle.volume || 0 };
                }
                if (!tvStreamBucketState || tvStreamBucketState.bucketTime !== bucketTime) {
                    tvStreamBucketState = { bucketTime, baseVolume: bar ? (bar.volume || 0) : 0, minutes: {}, lastMinute: 0 };
                }
                tvStreamBucketState.minutes[candle.time] = candle.volume || 0;
                const isLatestMinute = candle.time >= tvStreamBucketState.lastMinute;
                tvStreamBucketState.lastMinute = Math.max(tvStreamBucketState.lastMinute, candle.time);
                const streamedVolume = Object.values(tvStreamBucketState.minutes).reduce((sum, v) => sum + v, 0);
                const volume = tvStreamBucketState.baseVolume + streamedVolume;
                if (!bar) {
                    return { time: bucketTime, open: candle.open, high: candle.high, low: candle.low,
                             close: candle.close, volume };
                }
                return {
                    ...bar,
                    high: Math.max(bar.high, candle.high),
                    low: Math.min(bar.low, candle.low),
                    close: isLatestMinute ? candle.close : bar.close,
                    volume,
                };
            });
            tvApplyRawBar(raw);

            // Refresh indicators with the full multi-day history
            if (tvActiveInds.size > 0) applyIndicators(tvIndicatorCandles, tvActiveInds);
            scheduleTVHistoricalOverlayDraw();
        }

        // --- Fullscreen chart support ---
        const fsExpandSvg = '<svg viewBox="0 0 14 14" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 5V1h4M9 1h4v4M13 9v4H9M5 13H1V9"/></svg>';
        const fsCollapseSvg = '<svg viewBox="0 0 14 14" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M5 1v4H1M9 5h4V1M9 13V9h4M1 9h4v4"/></svg>';

        function getChartFullscreenTarget(container) {
            if (!container) return null;
            if (container.id === 'price-chart') {
                return container.closest('.price-chart-container') || container;
            }
            return container;
        }

        function queryFullscreenCharts() {
            return document.querySelectorAll('.chart-container.fullscreen, .price-chart-container.fullscreen');
        }

        function isChartFullscreenActive(containerId) {
            const container = document.getElementById(containerId);
            if (!container) return false;
            if (container.classList.contains('fullscreen')) return true;
            if (containerId === 'price-chart') {
                const wrapper = container.closest('.price-chart-container');
                return !!(wrapper && wrapper.classList.contains('fullscreen'));
            }
            return false;
        }

        function getChartFullscreenButton(container, target) {
            if (container && container.querySelector('.chart-fullscreen-btn')) {
                return container.querySelector('.chart-fullscreen-btn');
            }
            return target ? target.querySelector('.chart-fullscreen-btn') : null;
        }

        function getPlotlyLayoutSize(containerEl, plotEl) {
            if (!containerEl || !plotEl) return null;
            const plotContainer = plotEl.querySelector('.plot-container') || plotEl;
            const width = plotContainer.clientWidth || containerEl.clientWidth;
            const height = plotContainer.clientHeight || containerEl.clientHeight;
            if (width < 2 || height < 2) return null;
            return { width, height };
        }

        const PLOTLY_BASE_MARGINS = { l: 38, r: 68, t: 56, b: 16 };

        // Size/margins a regular (non-heatmap) Plotly chart should be drawn at. Shared by
        // renderPlotlyChart (so react draws at the final size) and resizePlotlyChart.
        function getPlotlyTargetLayout(containerEl, plotEl) {
            const size = getPlotlyLayoutSize(containerEl, plotEl);
            if (!size) return null;
            const containerId = containerEl.id;
            return {
                width: size.width,
                height: size.height,
                autosize: !(containerId && isChartFullscreenActive(containerId)),
                margin: containerId ? getChartMargins(containerId, PLOTLY_BASE_MARGINS) : PLOTLY_BASE_MARGINS,
            };
        }

        function plotlyLayoutMatches(plotEl, target) {
            const fl = plotEl && plotEl._fullLayout;
            if (!fl || !fl.margin) return false;
            const m = fl.margin;
            return fl.width === target.width && fl.height === target.height &&
                (target.autosize === undefined || fl.autosize === target.autosize) &&
                (!target.margin || (m.l === target.margin.l && m.r === target.margin.r &&
                                    m.t === target.margin.t && m.b === target.margin.b));
        }

        function resizePlotlyChart(containerOrEl) {
            const el = typeof containerOrEl === 'string'
                ? document.getElementById(containerOrEl)
                : containerOrEl;
            if (!el) return;
            if (el.id === 'heatmap-chart') {
                try { resizeHeatmapPlot(); } catch(e) {}
                return;
            }
            const plotEl = el.classList.contains('js-plotly-plot') ? el : el.querySelector('.js-plotly-plot');
            if (!plotEl) return;

            const target = getPlotlyTargetLayout(el, plotEl);
            // Every relayout is a full redraw; skip it when the chart is already this size
            if (!target || plotlyLayoutMatches(plotEl, target)) return;

            try {
                // Explicit width/height makes Plots.resize a no-op, so relayout alone suffices
                Plotly.relayout(plotEl, target);
            } catch (e) {}
        }

        function schedulePlotlyChartResize(containerId) {
            requestAnimationFrame(() => {
                resizePlotlyChart(containerId);
                requestAnimationFrame(() => resizePlotlyChart(containerId));
            });
        }

        function resizeTvPriceCharts() {
            const tvContainer = document.getElementById('price-chart');
            if (!tvContainer) return;
            const width = tvContainer.clientWidth;
            const height = tvContainer.clientHeight;
            if (tvPriceChart && width > 0 && height > 0) {
                try { tvPriceChart.applyOptions({ width, height }); } catch(e) {}
            }
            try { scheduleTVHistoricalOverlayDraw(); } catch(e) {}
        }

        function resizeChartsAfterFullscreen() {
            const doResize = () => {
                document.querySelectorAll('.chart-container').forEach(resizePlotlyChart);
                queryFullscreenCharts().forEach(resizePlotlyChart);
                resizeTvPriceCharts();
            };
            requestAnimationFrame(() => {
                doResize();
                requestAnimationFrame(doResize);
            });
            setTimeout(doResize, 100);
            setTimeout(doResize, 300);
        }

        function exitAllFullscreenCharts() {
            queryFullscreenCharts().forEach(el => {
                el.classList.remove('fullscreen');
                const btnHost = el.querySelector('#price-chart') || el;
                const b = btnHost.querySelector('.chart-fullscreen-btn');
                if (b) b.innerHTML = fsExpandSvg;
            });
            document.body.style.overflow = '';
        }

        function toggleChartFullscreen(container) {
            const target = getChartFullscreenTarget(container);
            if (!target) return;
            const wasFullscreen = target.classList.contains('fullscreen');

            exitAllFullscreenCharts();

            if (!wasFullscreen) {
                target.classList.add('fullscreen');
                document.body.style.overflow = 'hidden';
                const b = getChartFullscreenButton(container, target);
                if (b) b.innerHTML = fsCollapseSvg;
            }

            resizeChartsAfterFullscreen();
        }

        function ensureChartFullscreenChrome(container) {
            const target = getChartFullscreenTarget(container);
            if (!target) return;

            let chrome = target.querySelector('.chart-fullscreen-chrome');
            if (!chrome) {
                chrome = document.createElement('div');
                chrome.className = 'chart-fullscreen-chrome';

                const hoverzone = document.createElement('div');
                hoverzone.className = 'chart-fullscreen-hoverzone';
                hoverzone.setAttribute('aria-hidden', 'true');

                const logo = document.createElement('div');
                logo.className = 'chart-fullscreen-logo';
                logo.textContent = 'EzDuz1t Options';

                chrome.appendChild(hoverzone);
                chrome.appendChild(logo);
                target.appendChild(chrome);
            } else if (!chrome.querySelector('.chart-fullscreen-hoverzone')) {
                const hoverzone = document.createElement('div');
                hoverzone.className = 'chart-fullscreen-hoverzone';
                hoverzone.setAttribute('aria-hidden', 'true');
                chrome.insertBefore(hoverzone, chrome.firstChild);
            }

            // Migrate legacy standalone logo nodes into chrome
            target.querySelectorAll(':scope > .chart-fullscreen-logo').forEach(legacyLogo => {
                if (!chrome.contains(legacyLogo)) {
                    legacyLogo.remove();
                }
            });
        }

        function addFullscreenButton(container) {
            if (!container) return;
            ensureChartFullscreenChrome(container);
            if (container.querySelector('.chart-fullscreen-btn')) return;
            const target = getChartFullscreenTarget(container);
            const btn = document.createElement('button');
            btn.className = 'chart-fullscreen-btn';
            btn.innerHTML = (target && target.classList.contains('fullscreen')) ? fsCollapseSvg : fsExpandSvg;
            btn.title = 'Toggle fullscreen (Esc to exit)';
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                e.preventDefault();
                toggleChartFullscreen(container);
            });
            container.appendChild(btn);
        }

        // ESC key exits fullscreen chart
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && queryFullscreenCharts().length) {
                exitAllFullscreenCharts();
                resizeChartsAfterFullscreen();
            }
        });
        // Helper: returns appropriate Plotly margins depending on whether chart is fullscreen
        function getChartMargins(containerId, defaultMargins) {
            if (isChartFullscreenActive(containerId)) {
                return {
                    l: Math.max(defaultMargins.l || 50, 60),
                    r: Math.max(defaultMargins.r || 50, 130),
                    t: Math.max(defaultMargins.t || 40, 60),
                    b: Math.max(defaultMargins.b || 20, 40)
                };
            }
            return defaultMargins;
        }
        // --- End fullscreen support ---

        // --- Pop-out (Picture-in-Picture) chart support ---
        // Popouts get the main window's helpers injected as source, so each popout runs its
        // own copy (no calls across windows) without the code being duplicated here.
        function popoutHelperSource(fns) {
            return fns.map(fn => fn.toString()).join('\\n');
        }

        const popoutWindows = {}; // Map of chartId -> Window reference
        const popoutSvg = '<svg viewBox="0 0 14 14" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M10 1h3v3M13 1L8 6M5 2H2v10h10V9"/></svg>';

        function openPopoutChart(chartId) {
            // If already open and not closed, focus it
            if (popoutWindows[chartId] && !popoutWindows[chartId].closed) {
                popoutWindows[chartId].focus();
                return;
            }

            // Derive a display name from the chart id
            const displayName = chartId.replace('-chart', '').replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());

            const popup = window.open('', 'popout_' + chartId, 'width=900,height=650,menubar=no,toolbar=no,location=no,status=no,resizable=yes,scrollbars=no');
            if (!popup) {
                showError('Pop-up blocked! Please allow pop-ups for this site.');
                return;
            }

            // Price chart uses TradingView Lightweight Charts — needs a different template
            if (chartId === 'price-chart') {
                popup.document.write(`<!DOCTYPE html>
<html><head><title>Price Chart - EzOptions</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"><\\/script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
    :root { --app-bg:#1E1E1E; --panel-bg:#1a1a1a; --panel-bg-alt:#2D2D2D; --panel-bg-strong:#2a2a2a; --panel-hover:#3a3a3a; --chart-bg:#1E1E1E; --border-color:#333; --text-primary:#eef2f7; --text-secondary:#ccc; --text-muted:#888; --accent-color:#800080; --grid-color:#2A2A2A; --crosshair-color:#555; --tooltip-bg:linear-gradient(180deg,rgba(30,34,41,0.96),rgba(16,18,23,0.98)); --tooltip-border:rgba(255,255,255,0.08); }
    body { background:var(--app-bg); color:var(--text-secondary); font-family:Arial,sans-serif; display:flex; flex-direction:column; height:100vh; overflow:hidden; }
    #popout-logo { position:fixed; top:6px; left:10px; z-index:200; font-size:11px; font-weight:bold; color:var(--accent-color); opacity:0.7; pointer-events:none; letter-spacing:0.5px; }
    #toolbar { background:var(--panel-bg); border-bottom:1px solid var(--border-color); padding:4px 8px; display:flex; flex-wrap:wrap; gap:4px; align-items:center; flex-shrink:0; z-index:100; }
    .tv-tb-sep { width:1px; height:20px; background:var(--border-color); margin:0 2px; }
    .tb-btn { background:var(--panel-bg-strong); border:1px solid var(--border-color); color:var(--text-secondary); border-radius:4px; padding:3px 7px; font-size:11px; cursor:pointer; white-space:nowrap; transition:background 0.15s,color 0.15s,border-color 0.15s; user-select:none; }
        .tb-btn:hover  { background:var(--panel-hover); color:var(--text-primary); border-color:var(--accent-color); }
        .tb-btn.active { background:color-mix(in srgb, var(--accent-color) 30%, var(--panel-bg-strong)); border-color:var(--accent-color); color:var(--text-primary); }
        .tb-btn.danger { background:#5c1a1a; border-color:#c0392b; color:#f88; }
        .tv-indicator-picker { position:relative; }
        .tv-indicator-summary { display:inline-flex; align-items:center; gap:6px; list-style:none; }
        .tv-indicator-summary::-webkit-details-marker { display:none; }
        .tv-indicator-summary::marker { content:''; }
        .tv-indicator-badge { display:none; min-width:18px; padding:1px 6px; border-radius:999px; background:rgba(255,255,255,0.12); color:var(--text-primary); font-size:10px; line-height:1.3; text-align:center; }
        .tv-indicator-menu { position:absolute; top:calc(100% + 6px); left:0; width:min(320px,calc(100vw - 32px)); padding:8px; border:1px solid var(--border-color); border-radius:10px; background:var(--panel-bg); box-shadow:0 12px 28px rgba(0,0,0,0.35); z-index:130; }
        .tv-indicator-search { width:100%; border:1px solid var(--border-color); border-radius:8px; background:var(--panel-bg-strong); color:var(--text-primary); padding:7px 10px; font-size:12px; outline:none; }
        .tv-indicator-search:focus { border-color:var(--accent-color); }
        .tv-indicator-options { display:grid; gap:4px; margin-top:8px; max-height:240px; overflow:auto; }
        .tv-indicator-option { width:100%; border:1px solid var(--border-color); border-radius:8px; background:var(--panel-bg-strong); color:var(--text-secondary); padding:7px 9px; text-align:left; cursor:pointer; display:grid; gap:2px; }
        .tv-indicator-option:hover { background:var(--panel-hover); border-color:var(--accent-color); color:var(--text-primary); }
        .tv-indicator-option.active { background:color-mix(in srgb, var(--accent-color) 24%, var(--panel-bg-strong)); border-color:var(--accent-color); color:var(--text-primary); }
        .tv-indicator-option-name { font-size:12px; font-weight:600; }
        .tv-indicator-option-desc { font-size:10px; color:var(--text-muted); }
        .tv-indicator-option.active .tv-indicator-option-desc { color:var(--text-secondary); }
        .tv-indicator-option-empty { padding:8px; font-size:11px; color:var(--text-muted); text-align:center; }
  #chart-area { flex:1; display:flex; flex-direction:column; min-height:0; position:relative; }
  #price-chart { flex:1; min-height:0; position:relative; }
    .tv-sub-pane { background:var(--chart-bg); border-top:1px solid var(--border-color); flex-shrink:0; position:relative; }
    .tv-sub-pane-hdr { position:absolute; top:4px; left:8px; z-index:5; font-size:10px; color:var(--text-muted); font-weight:bold; pointer-events:none; }
    .ind-legend { position:absolute; bottom:8px; left:8px; display:none; flex-wrap:wrap; gap:6px; z-index:15; pointer-events:none; }
    .ind-item { font-size:10px; color:var(--text-secondary); display:flex; align-items:center; gap:4px; }
  .ind-swatch { width:14px; height:3px; border-radius:2px; }
    .title-el { display:inline-block; color:var(--text-secondary); font-size:13px; font-weight:bold; padding:2px 8px; pointer-events:none; }
    .candle-close-timer { font-size:11px; font-family:'Courier New',monospace; padding:3px 7px; border-radius:4px; background:var(--panel-bg-strong); border:1px solid var(--border-color); color:var(--text-secondary); white-space:nowrap; user-select:none; letter-spacing:0.5px; }
    .tv-ohlc-tooltip { position:absolute; top:8px; left:8px; z-index:50; font-size:11px; font-family:'Courier New',monospace; color:var(--text-secondary); pointer-events:none; white-space:nowrap; width:max-content; display:none; line-height:1.6; }
    .tv-ohlc-tooltip .tt-time { color:var(--text-muted); font-size:10px; margin-bottom:2px; }
  .tv-ohlc-tooltip .tt-up { color:#00FF00; }
  .tv-ohlc-tooltip .tt-dn { color:#FF4444; }
    .tv-historical-overlay { position:absolute; inset:0; z-index:20; pointer-events:none; overflow:hidden; }
    .tv-historical-canvas { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
        .tv-historical-tooltip { position:absolute; z-index:55; display:none; width:auto !important; height:auto !important; min-width:0; max-width:min(240px,calc(100% - 16px)); padding:8px; border:1px solid var(--tooltip-border); border-radius:10px; background:var(--tooltip-bg); color:var(--text-primary); font-size:10px; line-height:1.25; pointer-events:none; box-shadow:0 14px 36px rgba(0,0,0,0.38); backdrop-filter:blur(10px); flex:none !important; align-self:flex-start; overflow:hidden; white-space:normal; }
    .tv-historical-tooltip .tt-head { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:6px; }
        .tv-historical-tooltip .tt-badge { padding:2px 6px; border-radius:999px; background:rgba(255,255,255,0.08); color:var(--text-secondary); font-size:9px; letter-spacing:0.02em; text-transform:uppercase; }
        .tv-historical-tooltip .tt-time { color:var(--text-muted); font-size:9px; margin-bottom:0; }
    .tv-historical-tooltip .tt-list { display:grid; gap:4px; }
    .tv-historical-tooltip .tt-row { display:flex; align-items:center; gap:6px; min-width:0; }
    .tv-historical-tooltip .tt-dot { width:7px; height:7px; border-radius:999px; box-shadow:0 0 0 1px rgba(255,255,255,0.12); flex:0 0 auto; }
    .tv-historical-tooltip .tt-main { display:flex; justify-content:space-between; align-items:baseline; gap:8px; min-width:0; width:100%; }
        .tv-historical-tooltip .tt-name { color:var(--text-primary); font-weight:600; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .tv-historical-tooltip .tt-value { color:var(--text-secondary); font-variant-numeric:tabular-nums; white-space:nowrap; flex:0 0 auto; }
        .tv-historical-tooltip .tt-more { color:var(--text-muted); margin-top:4px; padding-top:4px; border-top:1px solid rgba(255,255,255,0.06); font-size:9px; }
</style></head><body>
<div id="popout-logo">EzDuz1t Options</div>
<div id="toolbar">
  <span class="title-el" id="chart-title">Price Chart</span>
  <div class="tv-tb-sep"></div>
</div>
<div id="chart-area">
  <div id="price-chart"></div>
  <div class="tv-sub-pane" id="rsi-pane" style="display:none"><div class="tv-sub-pane-hdr">RSI 14</div><div id="rsi-chart" style="height:110px"></div></div>
  <div class="tv-sub-pane" id="macd-pane" style="display:none"><div class="tv-sub-pane-hdr">MACD (12,26,9)</div><div id="macd-chart" style="height:120px"></div></div>
</div>
<script>
  // ── State ──────────────────────────────────────────────────────────────────
    var tvChart=null, tvCandle=null, tvVol=null;
    var tvRightScaleMinWidth=88;
  var tvRsiChart=null, tvRsiSeries=null;
  var tvMacdChart=null, tvMacdSeries={};
  var tvIndSeries={};
  var activeInds=new Set();
    var tvPriceLines=[], tvLatestLevelLabelLines=[], tvDrawings=[], tvDrawingDefs=[];
    var tvLatestLevelLabelSignature='';
  var tvAllLevelPrices=[];
    var tvHistoricalPoints=[];
  var tvLastCandles=[];
  var tvIndicatorCandles=[];
  var tvCurrentDayStartTime=0;
  var tvIndicatorRefreshTimer=null;
  var tvDrawMode=null, tvDrawStart=null;
  var tvAutoRange=false;
  var tvSyncHandlers=[], tvSyncingTS=false;
  var drawColor='#FFD700';
  var lineStyleMap={};
  var popoutTimeframe=1;
  var popoutCandleTimerInterval=null;
    var popoutTheme='dark';
    var popoutThemeVars={};
    var historicalDomBound=false;
    var tvHistoricalRenderedPoints=[];
    var tvHistoricalHoverBuckets=new Map();
        var tvShowLatestLevelLines=true;
            var tvPriceAboveBubbles=false;
        var historicalBubbleDrawPending=false;
        var historicalBubbleMaxVisible=1200;
        var historicalBubbleHoverBucketSize=48;
            var tvPriceColors={up:'#00FF00',down:'#FF0000'};

        function setPriceAboveBubblesSetting(enabled){tvPriceAboveBubbles=enabled===true;}

    function buildTVThemeOptions(){
        return {
            layout:{background:{color:getThemeValue('--chart-bg','#1E1E1E')},textColor:getThemeValue('--text-secondary','#CCCCCC'),fontFamily:'Arial, sans-serif'},
            grid:{vertLines:{color:getThemeValue('--grid-color','#2A2A2A')},horzLines:{color:getThemeValue('--grid-color','#2A2A2A')}},
            crosshair:{mode:LightweightCharts.CrosshairMode.Normal,vertLine:{color:getThemeValue('--crosshair-color','#555'),labelBackgroundColor:getThemeValue('--panel-bg-alt','#2D2D2D')},horzLine:{color:getThemeValue('--crosshair-color','#555'),labelBackgroundColor:getThemeValue('--panel-bg-alt','#2D2D2D')}},
            rightPriceScale:{borderColor:getThemeValue('--border-color','#333')},
            timeScale:{borderColor:getThemeValue('--border-color','#333')}
        };
    }
    function applyPopoutTheme(themePayload){
        if(!themePayload||!themePayload.vars)return;
        popoutTheme=themePayload.name||'dark';
        popoutThemeVars=themePayload.vars||{};
        Object.keys(popoutThemeVars).forEach(function(key){document.documentElement.style.setProperty(key,popoutThemeVars[key]);});
        document.body.dataset.theme=popoutTheme;
        if(tvChart){try{tvChart.applyOptions(buildTVThemeOptions());}catch(e){}}
        if(tvRsiChart){try{tvRsiChart.applyOptions(buildTVThemeOptions());}catch(e){}}
        if(tvMacdChart){try{tvMacdChart.applyOptions(buildTVThemeOptions());}catch(e){}}
    }

  // ── Candle close timer (popout) ────────────────────────────────────────────
  function startCandleCloseTimer(){
    if(popoutCandleTimerInterval)clearInterval(popoutCandleTimerInterval);
    function upd(){
      var el=document.getElementById('candle-close-timer');if(!el){clearInterval(popoutCandleTimerInterval);return;}
      var tfSecs=popoutTimeframe*60;
      var now=new Date();
      var fmt=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour:'numeric',minute:'numeric',second:'numeric',hour12:false});
      var h=0,m2=0,s2=0;
      fmt.formatToParts(now).forEach(function(p){if(p.type==='hour')h=parseInt(p.value);else if(p.type==='minute')m2=parseInt(p.value);else if(p.type==='second')s2=parseInt(p.value);});
      var sec=h*3600+m2*60+s2;
      var rem=tfSecs-(sec%tfSecs);
      var mm=Math.floor(rem/60),ss=rem%60;
      el.textContent='\u23F1 '+mm+':'+(ss<10?'0':'')+ss;
      el.className='candle-close-timer';
    }
    upd();
    popoutCandleTimerInterval=setInterval(upd,1000);
  }

  // ── Shared helpers (source injected from the main window) ────────────────
${popoutHelperSource([getThemeValue, calcSMA, calcEMA, calcWMA, calcVWAP, calcBB, calcRollingStdev,
    calcVWMA, calcFBB, calcRSI, calcATR, calcAutoTrendLine, calcMACD, formatLargeNumberCompact,
    formatTVBubbleTime, buildTVHistoricalTooltipHtml, getTVBubblePointKey, applyTVBubbleOverlapMetadata,
    getTVLatestLevelKey, formatTVLatestLevelLineText, tvFindBarIndex])}

    // ── Sub-pane chart factory ─────────────────────────────────────────────────
        function mkSubChart(el,h){return LightweightCharts.createChart(el,Object.assign({},buildTVThemeOptions(),{autoSize:true,height:h,rightPriceScale:{borderColor:getThemeValue('--border-color','#333'),scaleMargins:{top:0.1,bottom:0.1},minimumWidth:tvRightScaleMinWidth},localization:{timeFormatter:function(time){var d=new Date(time*1000);return d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/New_York'});}},timeScale:{borderColor:getThemeValue('--border-color','#333'),timeVisible:true,secondsVisible:false,fixLeftEdge:false,fixRightEdge:false,tickMarkFormatter:function(time){var d=new Date(time*1000);return d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/New_York'});}},handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true},handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false}}));}

  // ── Time-scale sync ────────────────────────────────────────────────────────
    function setupSync(){tvSyncHandlers.forEach(function(h){try{h.chart.timeScale().unsubscribeVisibleLogicalRangeChange(h.handler);}catch(e){}});tvSyncHandlers=[];var all=[tvChart,tvRsiChart,tvMacdChart].filter(Boolean);if(all.length<2)return;all.forEach(function(src){var others=all.filter(function(c){return c!==src;});var h=function(range){if(tvSyncingTS||!range)return;tvSyncingTS=true;others.forEach(function(c){try{c.timeScale().setVisibleLogicalRange(range);}catch(e){}});tvSyncingTS=false;};try{src.timeScale().subscribeVisibleLogicalRangeChange(h);}catch(e){}tvSyncHandlers.push({chart:src,handler:h});});if(tvChart){try{var r=tvChart.timeScale().getVisibleLogicalRange();if(r)[tvRsiChart,tvMacdChart].filter(Boolean).forEach(function(c){try{c.timeScale().setVisibleLogicalRange(r);}catch(e){}});}catch(e){}}}

  // ── Indicators ─────────────────────────────────────────────────────────────
  function applyIndicators(){
    if(!tvChart||!tvCandle)return;
    var candles=tvIndicatorCandles.length?tvIndicatorCandles:tvLastCandles;
    if(!candles.length)return;
    var dayStart=tvCurrentDayStartTime||0;
    function todayOnly(pairs){return pairs.filter(function(p){return p&&(!dayStart||p.time>=dayStart);});}
    var times=candles.map(function(c){return c.time;}),closes=candles.map(function(c){return c.close;});
    function mkLine(col,lw,title){return tvChart.addLineSeries({color:col,lineWidth:lw||1,priceScaleId:'right',lastValueVisible:true,priceLineVisible:false,title:title||''});}
    // Remove deactivated
    Object.keys(tvIndSeries).forEach(function(k){if(!activeInds.has(k)){var s=tvIndSeries[k];if(Array.isArray(s))s.forEach(function(x){try{tvChart.removeSeries(x);}catch(e){}});else{try{tvChart.removeSeries(s);}catch(e){};}delete tvIndSeries[k];}});
        if(activeInds.has('sma9')){if(!tvIndSeries['sma9'])tvIndSeries['sma9']=mkLine('#ffe082',1,'SMA9');tvIndSeries['sma9'].setData(todayOnly(calcSMA(closes,9).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('sma20')){if(!tvIndSeries['sma20'])tvIndSeries['sma20']=mkLine('#f0c040',1,'SMA20');tvIndSeries['sma20'].setData(todayOnly(calcSMA(closes,20).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('sma50')){if(!tvIndSeries['sma50'])tvIndSeries['sma50']=mkLine('#40a0f0',1,'SMA50');tvIndSeries['sma50'].setData(todayOnly(calcSMA(closes,50).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('sma100')){if(!tvIndSeries['sma100'])tvIndSeries['sma100']=mkLine('#7fd1ff',1,'SMA100');tvIndSeries['sma100'].setData(todayOnly(calcSMA(closes,100).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('sma200')){if(!tvIndSeries['sma200'])tvIndSeries['sma200']=mkLine('#e040fb',1,'SMA200');tvIndSeries['sma200'].setData(todayOnly(calcSMA(closes,200).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('ema9')){if(!tvIndSeries['ema9'])tvIndSeries['ema9']=mkLine('#ff9900',1,'EMA9');tvIndSeries['ema9'].setData(todayOnly(calcEMA(closes,9).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('ema21')){if(!tvIndSeries['ema21'])tvIndSeries['ema21']=mkLine('#00e5ff',1,'EMA21');tvIndSeries['ema21'].setData(todayOnly(calcEMA(closes,21).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('ema50')){if(!tvIndSeries['ema50'])tvIndSeries['ema50']=mkLine('#ff7096',1,'EMA50');tvIndSeries['ema50'].setData(todayOnly(calcEMA(closes,50).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('ema100')){if(!tvIndSeries['ema100'])tvIndSeries['ema100']=mkLine('#b388ff',1,'EMA100');tvIndSeries['ema100'].setData(todayOnly(calcEMA(closes,100).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('ema200')){if(!tvIndSeries['ema200'])tvIndSeries['ema200']=mkLine('#00c853',1,'EMA200');tvIndSeries['ema200'].setData(todayOnly(calcEMA(closes,200).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('wma20')){if(!tvIndSeries['wma20'])tvIndSeries['wma20']=mkLine('#ffd166',1,'WMA20');tvIndSeries['wma20'].setData(todayOnly(calcWMA(closes,20).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('wma50')){if(!tvIndSeries['wma50'])tvIndSeries['wma50']=mkLine('#8ecae6',1,'WMA50');tvIndSeries['wma50'].setData(todayOnly(calcWMA(closes,50).map(function(v,i){return v!==null?{time:times[i],value:v}:null;})));}
        if(activeInds.has('vwap')){var todayCandles=dayStart>0?candles.filter(function(c){return c.time>=dayStart;}):candles;if(!tvIndSeries['vwap'])tvIndSeries['vwap']=mkLine('#ffffff',1,'VWAP');var vv=calcVWAP(todayCandles.map(function(c){return{high:c.high,low:c.low,close:c.close,volume:c.volume||0};}));tvIndSeries['vwap'].setData(vv.map(function(v,i){return{time:todayCandles[i].time,value:v};}));}
        if(activeInds.has('bb')){var bb=calcBB(closes);if(!tvIndSeries['bb']){tvIndSeries['bb']=[mkLine('rgba(100,180,255,0.8)',1,'BB U'),mkLine('rgba(100,180,255,0.5)',1,'BB M'),mkLine('rgba(100,180,255,0.8)',1,'BB L')];}var bbSeries=tvIndSeries['bb'];bbSeries[0].setData(todayOnly(bb.map(function(v,i){return v.upper!==null?{time:times[i],value:v.upper}:null;})));bbSeries[1].setData(todayOnly(bb.map(function(v,i){return v.mid!==null?{time:times[i],value:v.mid}:null;})));bbSeries[2].setData(todayOnly(bb.map(function(v,i){return v.lower!==null?{time:times[i],value:v.lower}:null;})));}
        if(activeInds.has('fbb')){var fbb=calcFBB(candles);if(!tvIndSeries['fbb']||tvIndSeries['fbb'].length!==3){if(tvIndSeries['fbb']){tvIndSeries['fbb'].forEach(function(s){try{tvChart.removeSeries(s);}catch(e){}});}tvIndSeries['fbb']=[mkLine('#e040fb',2,'FBB Basis'),mkLine('#ef5350',2,'FBB Upper'),mkLine('#66bb6a',2,'FBB Lower')];}var fbbSeries=tvIndSeries['fbb'];fbbSeries[0].setData(todayOnly(fbb.map(function(v,i){return v.basis!==null?{time:times[i],value:v.basis}:null;})));fbbSeries[1].setData(todayOnly(fbb.map(function(v,i){return v.upper!==null?{time:times[i],value:v.upper}:null;})));fbbSeries[2].setData(todayOnly(fbb.map(function(v,i){return v.lower!==null?{time:times[i],value:v.lower}:null;})));}
        if(activeInds.has('atr')){var atrV=calcATR(candles),e20=calcEMA(closes,20),mult=1.5;if(!tvIndSeries['atr']){tvIndSeries['atr']=[mkLine('rgba(255,152,0,0.8)',1,'ATR U'),mkLine('rgba(255,152,0,0.8)',1,'ATR L')];}var atrSeries=tvIndSeries['atr'];atrSeries[0].setData(todayOnly(e20.map(function(v,i){return(v!==null&&atrV[i]!==null)?{time:times[i],value:v+mult*atrV[i]}:null;})));atrSeries[1].setData(todayOnly(e20.map(function(v,i){return(v!==null&&atrV[i]!==null)?{time:times[i],value:v-mult*atrV[i]}:null;})));}
                if(activeInds.has('auto_trend')){if(!tvIndSeries['auto_trend'])tvIndSeries['auto_trend']=tvChart.addLineSeries({color:'#ffca28',lineWidth:2,lineStyle:LightweightCharts.LineStyle.LargeDashed,priceScaleId:'right',lastValueVisible:true,priceLineVisible:false,title:'Auto Trend'});tvIndSeries['auto_trend'].setData(calcAutoTrendLine(candles,dayStart));}
    if(activeInds.has('rsi'))applyRsiPane(candles,times);else destroyRsiPane();
    if(activeInds.has('macd'))applyMacdPane(candles,times);else destroyMacdPane();
    updateLegend();
  }
  function applyRsiPane(candles,times){
    var pane=document.getElementById('rsi-pane');if(!pane)return;pane.style.display='block';
    var rsiVals=calcRSI(candles.map(function(c){return c.close;}));
    var rsiData=rsiVals.map(function(v,i){return v!==null?{time:times[i],value:v}:null;}).filter(Boolean);
    if(!tvRsiChart){var el=document.getElementById('rsi-chart');if(!el)return;tvRsiChart=mkSubChart(el,110);tvRsiSeries=tvRsiChart.addLineSeries({color:'#e91e63',lineWidth:1.5,lastValueVisible:true,priceLineVisible:false,title:'RSI14'});tvRsiSeries.createPriceLine({price:70,color:'rgba(255,100,100,0.7)',lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:'70'});tvRsiSeries.createPriceLine({price:30,color:'rgba(100,200,100,0.7)',lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:'30'});}
    if(rsiData.length)tvRsiSeries.setData(rsiData);
    setupSync();
  }
  function destroyRsiPane(){var pane=document.getElementById('rsi-pane');if(pane)pane.style.display='none';if(tvRsiChart){tvSyncHandlers=tvSyncHandlers.filter(function(h){return h.chart!==tvRsiChart;});try{tvRsiChart.remove();}catch(e){}tvRsiChart=null;tvRsiSeries=null;}}
  function applyMacdPane(candles,times){
    var pane=document.getElementById('macd-pane');if(!pane)return;pane.style.display='block';
    var md=calcMACD(candles.map(function(c){return c.close;}));
    var hd=md.histogram.map(function(v,i){return v!==null?{time:times[i],value:v,color:v>=0?'rgba(76,175,80,0.8)':'rgba(244,67,54,0.8)'}:null;}).filter(Boolean);
    var ld=md.macd.map(function(v,i){return v!==null?{time:times[i],value:v}:null;}).filter(Boolean);
    var sd=md.signal.map(function(v,i){return v!==null?{time:times[i],value:v}:null;}).filter(Boolean);
    if(!tvMacdChart){var el=document.getElementById('macd-chart');if(!el)return;tvMacdChart=mkSubChart(el,120);tvMacdSeries.hist=tvMacdChart.addHistogramSeries({lastValueVisible:false,priceLineVisible:false});tvMacdSeries.line=tvMacdChart.addLineSeries({color:'#2196f3',lineWidth:1.5,lastValueVisible:true,priceLineVisible:false,title:'MACD'});tvMacdSeries.signal=tvMacdChart.addLineSeries({color:'#ff9800',lineWidth:1,lastValueVisible:true,priceLineVisible:false,title:'Signal'});}
    if(hd.length)tvMacdSeries.hist.setData(hd);if(ld.length)tvMacdSeries.line.setData(ld);if(sd.length)tvMacdSeries.signal.setData(sd);
    setupSync();
  }
  function destroyMacdPane(){var pane=document.getElementById('macd-pane');if(pane)pane.style.display='none';if(tvMacdChart){tvSyncHandlers=tvSyncHandlers.filter(function(h){return h.chart!==tvMacdChart;});try{tvMacdChart.remove();}catch(e){}tvMacdChart=null;tvMacdSeries={
};}}
  function updateLegend(){
    var cont=document.getElementById('price-chart');if(!cont)return;
    var leg=cont.querySelector('.ind-legend');if(!leg){leg=document.createElement('div');leg.className='ind-legend';cont.appendChild(leg);}
    var cols={sma9:'#ffe082',sma20:'#f0c040',sma50:'#40a0f0',sma100:'#7fd1ff',sma200:'#e040fb',ema9:'#ff9900',ema21:'#00e5ff',ema50:'#ff7096',ema100:'#b388ff',ema200:'#00c853',wma20:'#ffd166',wma50:'#8ecae6',vwap:'#ffffff',bb:'rgba(100,180,255,0.8)',fbb:'#e040fb',auto_trend:'#ffca28',rsi:'#e91e63',macd:'#2196f3',atr:'rgba(255,152,0,0.8)'};
    var lbls={sma9:'SMA9',sma20:'SMA20',sma50:'SMA50',sma100:'SMA100',sma200:'SMA200',ema9:'EMA9',ema21:'EMA21',ema50:'EMA50',ema100:'EMA100',ema200:'EMA200',wma20:'WMA20',wma50:'WMA50',vwap:'VWAP',bb:'BB(20,2)',fbb:'FBB(200,3)',auto_trend:'Auto Trend',rsi:'RSI14',macd:'MACD',atr:'ATR Bands'};
    leg.innerHTML=Object.keys(tvIndSeries).map(function(k){return '<div class="ind-item"><div class="ind-swatch" style="background:'+( cols[k]||'#888')+'"></div>'+(lbls[k]||k)+'</div>';}).join('');
  }

  // ── Drawing tools ──────────────────────────────────────────────────────────
  function setDrawMode(mode){tvDrawMode=(tvDrawMode===mode)?null:mode;tvDrawStart=null;document.querySelectorAll('.tb-btn[data-draw]').forEach(function(b){b.classList.toggle('active',b.dataset.draw===tvDrawMode);});}
  function doUndo(){if(!tvChart||tvDrawings.length===0)return;var last=tvDrawings.pop();tvDrawingDefs.pop();if(Array.isArray(last))last.forEach(function(s){try{tvChart.removeSeries(s);}catch(e){}});else if(last&&last._isLine){try{tvCandle.removePriceLine(last);}catch(e){};}else{try{tvChart.removeSeries(last);}catch(e){}}}
  function doClear(){if(!tvChart)return;while(tvDrawings.length>0){var last=tvDrawings.pop();if(Array.isArray(last))last.forEach(function(s){try{tvChart.removeSeries(s);}catch(e){}});else if(last&&last._isLine){try{tvCandle.removePriceLine(last);}catch(e){};}else{try{tvChart.removeSeries(last);}catch(e){}}}tvDrawingDefs=[];}
  function handleClick(param){
    if(!tvDrawMode||!param||!param.point)return;
    var price=tvCandle?tvCandle.coordinateToPrice(param.point.y):null;
    if(price===null||price===undefined)return;
    var LS=LightweightCharts.LineStyle;
    if(tvDrawMode==='hline'){var l=tvCandle.createPriceLine({price:price,color:drawColor,lineWidth:1,lineStyle:LS.Solid,axisLabelVisible:true,title:''});l._isLine=true;tvDrawings.push(l);tvDrawingDefs.push({type:'hline',price:price,color:drawColor});return;}
    var clickTime=param.time;
    if(!clickTime&&tvLastCandles.length){try{clickTime=tvChart.timeScale().coordinateToTime(param.point.x);}catch(e){}if(!clickTime){var idx=Math.max(0,Math.min(Math.round(param.logical!=null?param.logical:tvLastCandles.length-1),tvLastCandles.length-1));clickTime=tvLastCandles[idx].time;}}
    if(tvDrawMode==='trendline'||tvDrawMode==='rect'){if(!clickTime)return;if(!tvDrawStart){tvDrawStart={price:price,time:clickTime};}else{if(tvDrawMode==='trendline'){var t1=tvDrawStart.time,p1=tvDrawStart.price,t2=clickTime,p2=price,tMin=Math.min(t1,t2),tMax=Math.max(t1,t2),vMin=t1<=t2?p1:p2,vMax=t1<=t2?p2:p1;var s=tvChart.addLineSeries({color:drawColor,lineWidth:1,priceScaleId:'right',lastValueVisible:false,priceLineVisible:false});s.setData([{time:tMin,value:vMin},{time:tMax,value:vMax}]);tvDrawings.push(s);tvDrawingDefs.push({type:'trendline',t1:t1,p1:p1,t2:t2,p2:p2,color:drawColor});}else{var top=Math.max(tvDrawStart.price,price),bot=Math.min(tvDrawStart.price,price);var tl=tvCandle.createPriceLine({price:top,color:drawColor,lineWidth:1,lineStyle:LS.Solid,axisLabelVisible:false,title:''});var bl=tvCandle.createPriceLine({price:bot,color:drawColor,lineWidth:1,lineStyle:LS.Solid,axisLabelVisible:false,title:''});tl._isLine=true;bl._isLine=true;tvDrawings.push([tl,bl]);tvDrawingDefs.push({type:'rect',top:top,bot:bot,color:drawColor});}tvDrawStart=null;}return;}
    if(tvDrawMode==='text'){var txt=prompt('Enter label text:');if(!txt)return;var l=tvCandle.createPriceLine({price:price,color:drawColor,lineWidth:1,lineVisible:false,lineStyle:LS.Solid,axisLabelVisible:true,title:txt});l._isLine=true;tvDrawings.push(l);tvDrawingDefs.push({type:'text',price:price,text:txt,color:drawColor});}
  }

  // ── Toolbar ────────────────────────────────────────────────────────────────
  function buildToolbar(candles,upColor,downColor){
    var tb=document.getElementById('toolbar');
    // Remove everything except the title and sep (first 2 children)
    while(tb.children.length>2)tb.removeChild(tb.lastChild);
    function btn(text,title,onClick,extra){var b=document.createElement('button');b.className='tb-btn'+(extra?' '+extra:'');b.textContent=text;b.title=title;b.addEventListener('click',onClick);return b;}
    function sep(){var d=document.createElement('div');d.className='tv-tb-sep';return d;}
        var inds=[{k:'sma9',l:'SMA9',t:'Simple Moving Average (9)'},{k:'sma20',l:'SMA20',t:'Simple Moving Average (20)'},{k:'sma50',l:'SMA50',t:'Simple Moving Average (50)'},{k:'sma100',l:'SMA100',t:'Simple Moving Average (100)'},{k:'sma200',l:'SMA200',t:'Simple Moving Average (200)'},{k:'ema9',l:'EMA9',t:'Exponential Moving Average (9)'},{k:'ema21',l:'EMA21',t:'Exponential Moving Average (21)'},{k:'ema50',l:'EMA50',t:'Exponential Moving Average (50)'},{k:'ema100',l:'EMA100',t:'Exponential Moving Average (100)'},{k:'ema200',l:'EMA200',t:'Exponential Moving Average (200)'},{k:'wma20',l:'WMA20',t:'Weighted Moving Average (20)'},{k:'wma50',l:'WMA50',t:'Weighted Moving Average (50)'},{k:'vwap',l:'VWAP',t:'Volume Weighted Average Price'},{k:'bb',l:'BB',t:'Bollinger Bands (20, 2)'},{k:'fbb',l:'FBB',t:'Fibonacci Bollinger Bands (200, 3)'},{k:'auto_trend',l:'Auto Trend',t:'Automatic linear-regression trend line for the current session'},{k:'rsi',l:'RSI',t:'Relative Strength Index (14) — sub-pane'},{k:'macd',l:'MACD',t:'MACD (12, 26, 9) — sub-pane'},{k:'atr',l:'ATR',t:'Average True Range (14)'},{k:'smc',l:'SMC',t:'Smart Money Concepts — Order Blocks, FVGs, BOS/CHoCH'}];
        var indPicker=document.createElement('details');indPicker.className='tv-indicator-picker';
        var indSummary=document.createElement('summary');indSummary.className='tb-btn tv-indicator-summary';indSummary.title='Search and toggle indicators';
        var indLabel=document.createElement('span');indLabel.textContent='Indicators';
        var indBadge=document.createElement('span');indBadge.className='tv-indicator-badge';
        indSummary.appendChild(indLabel);indSummary.appendChild(indBadge);indPicker.appendChild(indSummary);
        var indMenu=document.createElement('div');indMenu.className='tv-indicator-menu';
        var indSearch=document.createElement('input');indSearch.type='search';indSearch.className='tv-indicator-search';indSearch.placeholder='Search indicators';indSearch.autocomplete='off';
        var indOptions=document.createElement('div');indOptions.className='tv-indicator-options';
        function syncIndicatorSummary(){var count=activeInds.size;indBadge.textContent=String(count);indBadge.style.display=count?'inline-flex':'none';}
        function renderIndicatorOptions(){var query=(indSearch.value||'').trim().toLowerCase();indOptions.innerHTML='';var matches=inds.filter(function(def){return def.l.toLowerCase().indexOf(query)!==-1||def.t.toLowerCase().indexOf(query)!==-1;});if(!matches.length){var empty=document.createElement('div');empty.className='tv-indicator-option-empty';empty.textContent='No matching indicators';indOptions.appendChild(empty);return;}matches.forEach(function(def){var option=document.createElement('button');option.type='button';option.className='tv-indicator-option';if(activeInds.has(def.k))option.classList.add('active');var name=document.createElement('span');name.className='tv-indicator-option-name';name.textContent=def.l;var desc=document.createElement('span');desc.className='tv-indicator-option-desc';desc.textContent=def.t;option.appendChild(name);option.appendChild(desc);option.addEventListener('click',function(){if(activeInds.has(def.k))activeInds.delete(def.k);else activeInds.add(def.k);syncIndicatorSummary();renderIndicatorOptions();applyIndicators();});indOptions.appendChild(option);});}
        indSearch.addEventListener('input',renderIndicatorOptions);
        indPicker.addEventListener('toggle',function(){if(indPicker.open){setTimeout(function(){indSearch.focus();indSearch.select();},0);}else{indSearch.value='';renderIndicatorOptions();}});
        indMenu.appendChild(indSearch);indMenu.appendChild(indOptions);indPicker.appendChild(indMenu);syncIndicatorSummary();renderIndicatorOptions();tb.appendChild(indPicker);
    tb.appendChild(sep());
    // Drawing tools
    var draws=[{k:'hline',l:'— H-Line',t:'Horizontal price line'},{k:'trendline',l:'↗ Trend',t:'Trend line'},{k:'rect',l:'▭ Box',t:'Rectangle'},{k:'text',l:'T Label',t:'Price label'}];
    draws.forEach(function(def){var b=btn(def.l,def.t,function(){setDrawMode(def.k);});b.dataset.draw=def.k;if(tvDrawMode===def.k)b.classList.add('active');tb.appendChild(b);});
    // Color picker
    var cw=document.createElement('span');cw.style.cssText='display:flex;align-items:center;gap:3px;';
    var cp=document.createElement('input');cp.type='color';cp.value=drawColor;cp.style.cssText='width:24px;height:22px;border:none;background:none;cursor:pointer;padding:0;';cp.title='Drawing color';cp.addEventListener('input',function(){drawColor=cp.value;});
    cw.appendChild(cp);tb.appendChild(cw);
    tb.appendChild(sep());
    tb.appendChild(btn('↩ Undo','Undo last drawing',doUndo));
    tb.appendChild(btn('✕ Clear','Clear all drawings',doClear,'danger'));
    var spacer=document.createElement('div');spacer.style.flex='1';tb.appendChild(spacer);
    // Auto-range
    var arBtn=btn(tvAutoRange?'⤢ AR ON':'⤢ AR OFF','Toggle auto-range',function(){tvAutoRange=!tvAutoRange;arBtn.textContent=tvAutoRange?'⤢ AR ON':'⤢ AR OFF';arBtn.classList.toggle('active',tvAutoRange);if(tvChart)fitAll();},tvAutoRange?'active':'');
    tb.appendChild(arBtn);
    tb.appendChild(btn('⟳ Reset','Fit all data',fitAll));
    var timerEl=document.createElement('span');timerEl.id='candle-close-timer';timerEl.className='candle-close-timer';timerEl.title='Time remaining until the current candle closes';timerEl.textContent='\u23F1 --:--';tb.appendChild(timerEl);
    startCandleCloseTimer();
    if(tvChart)tvChart.subscribeClick(handleClick);
  }
  function tvApplyAutoscale(){if(!tvCandle)return;var lp=tvAllLevelPrices.slice();tvCandle.applyOptions({autoscaleInfoProvider:function(original){var res=original();if(!res)return res;if(lp.length===0)return res;var pad=(res.priceRange.maxValue-res.priceRange.minValue)*0.05;var minV=Math.min.apply(null,[res.priceRange.minValue].concat(lp))-pad;var maxV=Math.max.apply(null,[res.priceRange.maxValue].concat(lp))+pad;return{priceRange:{minValue:minV,maxValue:maxV},margins:res.margins};}});}
    function fitAll(){if(!tvChart)return;setTimeout(function(){try{tvChart.timeScale().fitContent();tvChart.priceScale('right').applyOptions({autoScale:true});tvApplyAutoscale();if(tvRsiChart)tvRsiChart.priceScale('right').applyOptions({autoScale:true});if(tvMacdChart)tvMacdChart.priceScale('right').applyOptions({autoScale:true});}catch(e){}},50);}
    function ensureHistOverlay(){var c=document.getElementById('price-chart');if(!c)return null;var o=c.querySelector('.tv-historical-overlay');if(!o){o=document.createElement('div');o.className='tv-historical-overlay';c.appendChild(o);}return o;}
    function ensureHistCanvas(){var o=ensureHistOverlay();if(!o)return null;var canvas=o.querySelector('.tv-historical-canvas');if(!canvas){canvas=document.createElement('canvas');canvas.className='tv-historical-canvas';o.appendChild(canvas);}return canvas;}
    function syncHistCanvas(canvas,overlay){if(!canvas||!overlay)return null;var dpr=window.devicePixelRatio||1,width=Math.max(1,Math.round(overlay.clientWidth)),height=Math.max(1,Math.round(overlay.clientHeight)),pixelWidth=Math.max(1,Math.round(width*dpr)),pixelHeight=Math.max(1,Math.round(height*dpr));if(canvas.width!==pixelWidth||canvas.height!==pixelHeight){canvas.width=pixelWidth;canvas.height=pixelHeight;canvas.style.width=width+'px';canvas.style.height=height+'px';}var ctx=canvas.getContext('2d');if(!ctx)return null;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);return{ctx:ctx,width:width,height:height};}
    function ensureHistTip(){var c=document.getElementById('price-chart');if(!c)return null;var t=c.querySelector('.tv-historical-tooltip');if(!t){t=document.createElement('div');t.className='tv-historical-tooltip';c.appendChild(t);}return t;}
    function snapshotBubblePoints(pd){var candles=pd.candles&&pd.candles.length?pd.candles:tvLastCandles,latest=candles&&candles.length?candles[candles.length-1]:null,t=latest?latest.time:null;if(!t)return[];var pts=[],levels=Array.isArray(pd.exposure_levels)?pd.exposure_levels:[];levels.forEach(function(level,index){var price=Number(level.price),raw=Number(level.value);if(!Number.isFinite(price))return;var label=level.type||'Level',value=level.label?String(level.label).replace(/^.*?:\\\\s*/,''):formatLargeNumberCompact(raw),base=raw>=0?(pd.call_color||'#00FF00'):(pd.put_color||'#FF0000');pts.push({time:t,price:price,size:level.is_max?16:12,color:level.color||base,border_color:level.is_max?(pd.max_level_color||'#800080'):base,border_width:level.is_max?2:Math.max(1,Number(level.line_width)||1),label:label,rank:index+1,side:raw>=0?'Call':'Put',value:value,kind:'exposure'});});var moves=Array.isArray(pd.expected_moves)?pd.expected_moves:[];moves.forEach(function(move,index){var upper=Number(move.upper),lower=Number(move.lower),moveValue=(Number.isFinite(upper)&&Number.isFinite(lower))?'$'+(Math.abs(upper-lower)/2).toFixed(2):'';[['Upper',upper],['Lower',lower]].forEach(function(item){var side=item[0],price=Number(item[1]);if(!Number.isFinite(price))return;pts.push({time:t,price:price,size:13,color:'rgba(3,107,252,0.82)',border_color:'#81b4ff',border_width:1,label:'Expected Move',rank:levels.length+index+1,side:side,value:moveValue,kind:'expected-move'});});});return pts;}
    function priceLevelBubblePoints(pd){var hist=Array.isArray(pd.historical_exposure_levels)?pd.historical_exposure_levels.filter(function(p){return p&&p.time&&Number.isFinite(Number(p.price));}).map(function(p){return Object.assign({},p);}):[],snap=snapshotBubblePoints(pd);if(!hist.length)return applyTVBubbleOverlapMetadata(snap);if(!snap.length)return applyTVBubbleOverlapMetadata(hist);var seen=new Set(hist.map(function(p){return getTVBubblePointKey(p);}));snap.forEach(function(p){var key=getTVBubblePointKey(p);if(!seen.has(key)){seen.add(key);hist.push(p);}});return applyTVBubbleOverlapMetadata(hist);}
    function buildNativeLevelLabelSignature(points){if(!tvShowLatestLevelLines||!points.length)return '';return points.map(function(p){return[Number(p.price),p.border_color||p.color||'#fff',formatTVLatestLevelLineText(p)].join('|');}).join('||');}
    function clearNativeLevelLabels(){tvLatestLevelLabelSignature='';if(tvCandle&&tvLatestLevelLabelLines.length){tvLatestLevelLabelLines.forEach(function(line){try{tvCandle.removePriceLine(line);}catch(e){}});}tvLatestLevelLabelLines=[];}
    function setNativeLevelLabels(points){var signature=buildNativeLevelLabelSignature(points);if(signature&&signature===tvLatestLevelLabelSignature)return;clearNativeLevelLabels();if(!tvShowLatestLevelLines||!tvCandle||!points.length)return;tvLatestLevelLabelSignature=signature;points.forEach(function(p){var price=Number(p.price);if(!Number.isFinite(price))return;var line=tvCandle.createPriceLine({price:price,color:p.border_color||p.color||'#fff',lineWidth:1,lineVisible:false,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:formatTVLatestLevelLineText(p)});tvLatestLevelLabelLines.push(line);});}
    function drawLatestLevelLines(ctx,width,height){if(!tvShowLatestLevelLines||!tvHistoricalRenderedPoints.length){clearNativeLevelLabels();return;}var latestByKey=new Map();tvHistoricalRenderedPoints.forEach(function(p){if(!p||p.x==null||p.y==null)return;var key=getTVLatestLevelKey(p),existing=latestByKey.get(key);if(!existing||p.time>existing.time||(p.time===existing.time&&p.x>existing.x))latestByKey.set(key,p);});var points=Array.from(latestByKey.values()).sort(function(a,b){return a.y-b.y;});if(!points.length){clearNativeLevelLabels();return;}setNativeLevelLabels(points);ctx.save();points.forEach(function(p){var color=p.border_color||p.color||'#fff',radius=(p.size||8)/2,lineStartX=Math.max(0,p.x+radius+3),lineEndX=Math.max(lineStartX,width-2);if(lineEndX<=lineStartX)return;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=p.border_width||1;ctx.globalAlpha=0.92;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(lineStartX,p.y);ctx.lineTo(lineEndX,p.y);ctx.stroke();ctx.restore();});ctx.restore();}
    function posHistTip(t,e){var c=document.getElementById('price-chart');if(!t||!c||!e)return;var b=c.getBoundingClientRect();var l=Math.min(Math.max(8,e.clientX-b.left+12),Math.max(8,b.width-t.offsetWidth-8));var top=Math.min(Math.max(8,e.clientY-b.top+12),Math.max(8,b.height-t.offsetHeight-8));t.style.left=l+'px';t.style.top=top+'px';}
    function addHistHoverPoint(point){var hoverRadius=Math.max(8,(point.size||8)/2+5),minBucketX=Math.floor((point.x-hoverRadius)/historicalBubbleHoverBucketSize),maxBucketX=Math.floor((point.x+hoverRadius)/historicalBubbleHoverBucketSize),minBucketY=Math.floor((point.y-hoverRadius)/historicalBubbleHoverBucketSize),maxBucketY=Math.floor((point.y+hoverRadius)/historicalBubbleHoverBucketSize);for(var bucketX=minBucketX;bucketX<=maxBucketX;bucketX++){for(var bucketY=minBucketY;bucketY<=maxBucketY;bucketY++){var key=bucketX+':'+bucketY,bucket=tvHistoricalHoverBuckets.get(key);if(!bucket){bucket=[];tvHistoricalHoverBuckets.set(key,bucket);}bucket.push(point);}}}
    function getHistHoverCandidates(cx,cy){if(!tvHistoricalHoverBuckets.size)return tvHistoricalRenderedPoints;var minBucketX=Math.floor((cx-32)/historicalBubbleHoverBucketSize),maxBucketX=Math.floor((cx+32)/historicalBubbleHoverBucketSize),minBucketY=Math.floor((cy-32)/historicalBubbleHoverBucketSize),maxBucketY=Math.floor((cy+32)/historicalBubbleHoverBucketSize),seen=new Set(),candidates=[];for(var bucketX=minBucketX;bucketX<=maxBucketX;bucketX++){for(var bucketY=minBucketY;bucketY<=maxBucketY;bucketY++){var bucket=tvHistoricalHoverBuckets.get(bucketX+':'+bucketY);if(!bucket)continue;bucket.forEach(function(point){if(seen.has(point))return;seen.add(point);candidates.push(point);});}}return candidates;}
    function findHistHoverPoints(e){var c=document.getElementById('price-chart');if(!c||!tvHistoricalRenderedPoints.length)return[];var b=c.getBoundingClientRect(),cx=e.clientX-b.left,cy=e.clientY-b.top;return getHistHoverCandidates(cx,cy).filter(function(p){var dx=cx-p.x,dy=cy-p.y,r=Math.max(8,(p.size||8)/2+5);return(dx*dx+dy*dy)<=(r*r);}).sort(function(a,bp){var ad=(cx-a.x)*(cx-a.x)+(cy-a.y)*(cy-a.y),bd=(cx-bp.x)*(cx-bp.x)+(cy-bp.y)*(cy-bp.y);return ad-bd;});}
    function updateHistTip(e){var t=ensureHistTip();if(!t)return;if(e&&e.buttons){t.style.display='none';return;}var pts=findHistHoverPoints(e);if(!pts.length){t.style.display='none';return;}var topPts=pts.slice(0,5),anchorTime=topPts[0].time;t.innerHTML='<div class="tt-head"><span class="tt-badge">'+pts.length+' bubble'+(pts.length===1?'':'s')+'</span><div class="tt-time">'+formatTVBubbleTime(anchorTime)+'</div></div><div class="tt-list">'+topPts.map(function(p){return buildTVHistoricalTooltipHtml(p);}).join('')+'</div>'+(pts.length>topPts.length?'<div class="tt-more">+'+(pts.length-topPts.length)+' more</div>':'');t.style.display='block';posHistTip(t,e);}
    function getVisibleHistoricalBubblePoints(){if(!tvHistoricalPoints.length)return[];var pts=tvHistoricalPoints;try{var range=tvChart.timeScale().getVisibleLogicalRange();if(range&&tvLastCandles.length){var li=Math.max(0,Math.floor(range.from)-2),ri=Math.min(tvLastCandles.length-1,Math.ceil(range.to)+2),left=tvLastCandles[li],right=tvLastCandles[ri];if(left&&right){var span=tvLastCandles.length>1?Math.max(60,tvLastCandles[1].time-tvLastCandles[0].time):60,minTime=left.time-(span*2),maxTime=right.time+(span*2);pts=tvHistoricalPoints.filter(function(p){return p.time>=minTime&&p.time<=maxTime;});}}}catch(e){}if(pts.length<=historicalBubbleMaxVisible)return pts;var priority=[],secondary=[];pts.forEach(function(p){if(p.kind==='expected-move'||p.rank===1)priority.push(p);else secondary.push(p);});if(priority.length>=historicalBubbleMaxVisible){var pStride=Math.ceil(priority.length/historicalBubbleMaxVisible);return priority.filter(function(_,i){return i%pStride===0;});}var slots=Math.max(0,historicalBubbleMaxVisible-priority.length);if(!secondary.length||slots===0)return priority;var stride=Math.ceil(secondary.length/slots);return priority.concat(secondary.filter(function(_,i){return i%stride===0;}));}
        function getVisibleOverlayCandles(){if(!tvLastCandles.length||!tvChart)return[];var candles=tvLastCandles;try{var range=tvChart.timeScale().getVisibleLogicalRange();if(range){var li=Math.max(0,Math.floor(range.from)-2),ri=Math.min(tvLastCandles.length-1,Math.ceil(range.to)+2);candles=tvLastCandles.slice(li,ri+1);}}catch(e){}return candles;}
        function drawPriceAboveBubbles(ctx,width){if(!tvPriceAboveBubbles||!tvChart||!tvCandle||!tvLastCandles.length)return;var candlePoints=getVisibleOverlayCandles().map(function(candle){return{candle:candle,x:tvChart.timeScale().timeToCoordinate(candle.time)};}).filter(function(point){return point.x!=null&&!Number.isNaN(point.x)&&point.x>=-24&&point.x<=width+24;}).sort(function(left,right){return left.x-right.x;});if(!candlePoints.length)return;var minDelta=Infinity;for(var index=1;index<candlePoints.length;index++){var delta=candlePoints[index].x-candlePoints[index-1].x;if(delta>0&&delta<minDelta)minDelta=delta;}var bodyWidth=Number.isFinite(minDelta)?Math.max(3,Math.min(14,Math.round(minDelta*0.68))):5,halfWidth=bodyWidth/2,upColor=tvPriceColors.up||'#00FF00',downColor=tvPriceColors.down||'#FF0000';ctx.save();candlePoints.forEach(function(entry){var candle=entry.candle,x=entry.x,openY=tvCandle.priceToCoordinate(candle.open),highY=tvCandle.priceToCoordinate(candle.high),lowY=tvCandle.priceToCoordinate(candle.low),closeY=tvCandle.priceToCoordinate(candle.close);if([openY,highY,lowY,closeY].some(function(value){return value==null||Number.isNaN(value);})){return;}var color=candle.close>=candle.open?upColor:downColor,wickTop=Math.min(highY,lowY),wickBottom=Math.max(highY,lowY),bodyTop=Math.min(openY,closeY),bodyBottom=Math.max(openY,closeY),bodyHeight=Math.max(1,Math.round(bodyBottom-bodyTop)),left=Math.round(x-halfWidth),centerX=Math.round(x)+0.5;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(centerX,wickTop);ctx.lineTo(centerX,wickBottom);ctx.stroke();ctx.fillRect(left,Math.round(bodyTop),bodyWidth,bodyHeight);});ctx.restore();}
        function drawHistoricalBubbles(){var o=ensureHistOverlay(),canvas=ensureHistCanvas(),t=ensureHistTip();if(!o||!canvas||!tvChart||!tvCandle){clearNativeLevelLabels();return;}tvHistoricalRenderedPoints=[];tvHistoricalHoverBuckets=new Map();if(!tvHistoricalPoints.length){clearNativeLevelLabels();o.style.display='none';if(t)t.style.display='none';return;}o.style.display='block';var canvasState=syncHistCanvas(canvas,o);if(!canvasState){clearNativeLevelLabels();o.style.display='none';if(t)t.style.display='none';return;}var ctx=canvasState.ctx,width=canvasState.width,height=canvasState.height;var points=getVisibleHistoricalBubblePoints();if(!points.length){clearNativeLevelLabels();o.style.display='none';if(t)t.style.display='none';return;}var visible=0;var paneW=width,paneH=height;try{paneW=tvChart.timeScale().width()||width;paneH=Math.max(0,height-(tvChart.timeScale().height()||0));}catch(e){}ctx.save();ctx.beginPath();ctx.rect(0,0,paneW,paneH);ctx.clip();points.forEach(function(p){var x=tvChart.timeScale().timeToCoordinate(p.time),y=tvCandle.priceToCoordinate(p.price);if(x==null||y==null||Number.isNaN(x)||Number.isNaN(y))return;var size=p.size||8,radius=size/2,overlapCount=Math.max(1,p.overlap_count||1),overlapSlot=Math.max(0,Math.min(overlapCount-1,p.overlap_slot||0)),offsetStep=Math.max(4,Math.min(10,radius*0.9)),offsetX=overlapCount>1?(overlapSlot-((overlapCount-1)/2))*offsetStep:0,drawX=x+offsetX,hoverRadius=Math.max(8,radius+5);if(drawX<-hoverRadius||drawX>paneW+hoverRadius||y<-hoverRadius||y>paneH+hoverRadius)return;ctx.save();ctx.globalAlpha=0.95;ctx.fillStyle=p.color||'rgba(255,255,255,0.6)';ctx.beginPath();ctx.arc(drawX,y,radius,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle='rgba(0,0,0,0.25)';ctx.lineWidth=1;ctx.beginPath();ctx.arc(drawX,y,radius+1,0,Math.PI*2);ctx.stroke();ctx.strokeStyle=p.border_color||p.color||'#fff';ctx.lineWidth=p.border_width||1;ctx.beginPath();ctx.arc(drawX,y,Math.max(0.5,radius-((p.border_width||1)/2)),0,Math.PI*2);ctx.stroke();ctx.restore();var renderedPoint=Object.assign({},p,{x:drawX,y:y});tvHistoricalRenderedPoints.push(renderedPoint);addHistHoverPoint(renderedPoint);visible++;});ctx.restore();if(visible>0){drawLatestLevelLines(ctx,paneW,paneH);drawPriceAboveBubbles(ctx,width);}else clearNativeLevelLabels();o.style.display=visible>0?'block':'none';}
    function scheduleHistoricalBubbleDraw(){if(historicalBubbleDrawPending)return;historicalBubbleDrawPending=true;requestAnimationFrame(function(){historicalBubbleDrawPending=false;drawHistoricalBubbles();});}

  // ── Main renderer ──────────────────────────────────────────────────────────
  var isFirstRender=true;
  function renderPriceChart(priceData){
    var upColor=priceData.call_color||'#00FF00',downColor=priceData.put_color||'#FF0000';
        if(priceData.price_above_bubbles!==undefined)setPriceAboveBubblesSetting(priceData.price_above_bubbles);
        tvPriceColors={up:upColor,down:downColor};
        var candles=priceData.candles||[];
    popoutTimeframe=parseInt(priceData.timeframe)||1;
    popoutUseHA=!!priceData.use_heikin_ashi;
    lineStyleMap={dashed:LightweightCharts.LineStyle.Dashed,dotted:LightweightCharts.LineStyle.Dotted,large_dashed:LightweightCharts.LineStyle.LargeDashed};
    if(!tvChart){
      var el=document.getElementById('price-chart');
    tvChart=LightweightCharts.createChart(el,Object.assign({},buildTVThemeOptions(),{autoSize:true,rightPriceScale:{borderColor:getThemeValue('--border-color','#333'),scaleMargins:{top:0.04,bottom:0.15},minimumWidth:tvRightScaleMinWidth},localization:{timeFormatter:function(time){var d=new Date(time*1000);return d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/New_York'});}},timeScale:{borderColor:getThemeValue('--border-color','#333'),timeVisible:true,secondsVisible:false,fixLeftEdge:false,fixRightEdge:false,tickMarkFormatter:function(time){var d=new Date(time*1000);return d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/New_York'});}},handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true},handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false}}));
      tvCandle=tvChart.addCandlestickSeries({upColor:upColor,downColor:downColor,borderVisible:false,wickUpColor:upColor,wickDownColor:downColor});
      tvVol=tvChart.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'volume',lastValueVisible:false,priceLineVisible:false});
      tvChart.priceScale('volume').applyOptions({scaleMargins:{top:0.88,bottom:0}});
      document.getElementById('chart-title').textContent=priceData.use_heikin_ashi?'Price Chart (Heikin-Ashi)':'Price Chart';
      buildToolbar(candles,upColor,downColor);
    ensureHistOverlay();ensureHistTip();tvChart.timeScale().subscribeVisibleLogicalRangeChange(function(){scheduleHistoricalBubbleDraw();});if(!historicalDomBound){historicalDomBound=true;el.addEventListener('wheel',function(){scheduleHistoricalBubbleDraw();},{passive:true});el.addEventListener('mouseup',function(){scheduleHistoricalBubbleDraw();});el.addEventListener('touchend',function(){scheduleHistoricalBubbleDraw();},{passive:true});el.addEventListener('mousemove',function(e){updateHistTip(e);if(e.buttons)scheduleHistoricalBubbleDraw();});el.addEventListener('mouseleave',function(){var t=ensureHistTip();if(t)t.style.display='none';});}
      // ── OHLC hover tooltip ──────────────────────────────────────────────
      var _ptip=document.createElement('div');_ptip.className='tv-ohlc-tooltip';_ptip.id='tv-ohlc-tooltip';el.appendChild(_ptip);
      tvChart.subscribeCrosshairMove(function(param){
        var tip=document.getElementById('tv-ohlc-tooltip');if(!tip)return;
        if(!param||!param.time||!param.seriesData){tip.style.display='none';return;}
        var bar=param.seriesData.get(tvCandle);if(!bar){tip.style.display='none';return;}
        var d=new Date(param.time*1000);
        var ts=d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/New_York'})+' ET';
        var cls=bar.close>=bar.open?'tt-up':'tt-dn';
        var chg=bar.open!==0?((bar.close-bar.open)/bar.open*100).toFixed(2):'0.00';
        var fmt=function(v){return v!=null?v.toFixed(2):'--';};
        var fv=function(v){return v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(0)+'K':(v||0).toString();};
        tip.innerHTML='<div class="tt-time">'+ts+'</div>'
          +'<span class="'+cls+'">O <b>'+fmt(bar.open)+'</b>  H <b>'+fmt(bar.high)+'</b>  L <b>'+fmt(bar.low)+'</b>  C <b>'+fmt(bar.close)+'</b>  '+(chg>=0?'+':'')+chg+'%</span>'
                    +'<br><span style="color:'+getThemeValue('--text-muted','#888')+'">Vol <b>'+fv(bar.volume)+'</b></span>';
        tip.style.display='block';
      });
    } else {
            try{tvChart.applyOptions(buildTVThemeOptions());}catch(e){}
      tvCandle.applyOptions({upColor:upColor,downColor:downColor,wickUpColor:upColor,wickDownColor:downColor});
    }
    tvCandle.setData(candles);
    tvVol.setData(priceData.volume||[]);
    tvLastCandles=candles;
    popStreamBucketState=null;
    tvIndicatorCandles=(priceData.indicator_candles&&priceData.indicator_candles.length>0)?priceData.indicator_candles:candles.map(function(c){return Object.assign({},c);});
    tvCurrentDayStartTime=priceData.current_day_start_time||0;
        tvPriceLines.forEach(function(l){try{tvCandle.removePriceLine(l);}catch(e){}});tvPriceLines=[];tvAllLevelPrices=[];
        tvShowLatestLevelLines=priceData.show_latest_level_lines!==false;
        tvHistoricalPoints=priceLevelBubblePoints(priceData);
        pushLatestLevelPrices();
        scheduleHistoricalBubbleDraw();
    tvApplyAutoscale();
    if(activeInds.size>0)applyIndicators();
    if(isFirstRender||tvAutoRange){fitAll();isFirstRender=false;}
  }

  // ── Real-time quote / candle application ─────────────────────────────────
  // Bucket a timestamp into the candle of the chart timeframe, anchored on the last loaded
  // candle so session-aligned bars (e.g. 60m bars starting at :30) line up.
  function pushLatestLevelPrices(){var mx=0;tvHistoricalPoints.forEach(function(p){if(Number(p.time)>mx)mx=Number(p.time);});tvHistoricalPoints.forEach(function(p){if(Number(p.time)===mx)tvAllLevelPrices.push(p.price);});}
  function popBucketStart(ts){var tf=(popoutTimeframe||1)*60;var lc=tvLastCandles[tvLastCandles.length-1];if(!lc)return Math.floor(ts/tf)*tf;var b=lc.time+Math.floor((ts-lc.time)/tf)*tf;return b<tvLastCandles[0].time?null:b;}
  function popIsSessionTime(ts){if((popoutSseTicker||'').charAt(0)==='/')return true;var et=new Date(new Date(ts*1000).toLocaleString('en-US',{timeZone:'America/New_York'}));var day=et.getDay(),mins=et.getHours()*60+et.getMinutes();return day>=1&&day<=5&&mins>=570&&mins<960;}
  var popStreamBucketState=null,popoutUseHA=false;
  // Live data is merged into raw bars (tvIndicatorCandles); the displayed candle is derived
  // from them so Heikin-Ashi stays Heikin-Ashi and volume bars follow along.
  function popUpsertRaw(bucketTime,build){var i=tvFindBarIndex(tvIndicatorCandles,bucketTime);if(i>=0){tvIndicatorCandles[i]=build(tvIndicatorCandles[i]);return tvIndicatorCandles[i];}var b=build(null);tvIndicatorCandles.push(b);tvIndicatorCandles.sort(function(a,c){return a.time-c.time;});return b;}
  function popDisplayBar(raw,prev){if(!popoutUseHA)return{time:raw.time,open:raw.open,high:raw.high,low:raw.low,close:raw.close};var hc=(raw.open+raw.high+raw.low+raw.close)/4,ho=prev?(prev.open+prev.close)/2:(raw.open+raw.close)/2;return{time:raw.time,open:ho,high:Math.max(raw.high,ho,hc),low:Math.min(raw.low,ho,hc),close:hc};}
  function popVolumeBar(bar,prev,volume){var up=prev?bar.close>=prev.close:bar.close>=bar.open;return{time:bar.time,value:volume||0,color:up?(tvPriceColors.up||'#00FF00'):(tvPriceColors.down||'#FF0000')};}
  function popApplyRaw(raw){
    var idx=tvFindBarIndex(tvLastCandles,raw.time),isNew=idx<0;
    if(isNew){var ld=tvLastCandles[tvLastCandles.length-1];if(ld&&raw.time<ld.time)return false;tvLastCandles.push(null);idx=tvLastCandles.length-1;}
    var lastIdx=popoutUseHA?tvLastCandles.length-1:idx;
    for(var i=idx;i<=lastIdx;i++){var rb=i===idx?raw:tvIndicatorCandles[tvFindBarIndex(tvIndicatorCandles,tvLastCandles[i].time)];if(!rb)break;tvLastCandles[i]=popDisplayBar(rb,tvLastCandles[i-1]);}
    if(idx===tvLastCandles.length-1&&lastIdx===idx){
      try{tvCandle.update(tvLastCandles[idx]);}catch(e){}
      if(tvVol){try{tvVol.update(popVolumeBar(tvLastCandles[idx],tvLastCandles[idx-1],raw.volume));}catch(e){}}
    }else{
      try{tvCandle.setData(tvLastCandles);}catch(e){}
      if(tvVol){try{tvVol.setData(tvLastCandles.map(function(b,i){var ri=tvFindBarIndex(tvIndicatorCandles,b.time);return popVolumeBar(b,tvLastCandles[i-1],ri>=0?tvIndicatorCandles[ri].volume:0);}));}catch(e){}}
    }
    return isNew;
  }
  function applyRealtimeQuote(last){
    if(!tvCandle||!tvLastCandles.length)return;
    var nowSec=Math.floor(Date.now()/1000);
    if(!popIsSessionTime(nowSec))return;
    var bucketTime=popBucketStart(nowSec);
    if(bucketTime===null||bucketTime<tvLastCandles[tvLastCandles.length-1].time)return;
    var raw=popUpsertRaw(bucketTime,function(bar){return bar?Object.assign({},bar,{high:Math.max(bar.high,last),low:Math.min(bar.low,last),close:last}):{time:bucketTime,open:last,high:last,low:last,close:last,volume:0};});
    var isNewBar=popApplyRaw(raw);
    if(activeInds.size>0){clearTimeout(tvIndicatorRefreshTimer);if(isNewBar)applyIndicators();else tvIndicatorRefreshTimer=setTimeout(applyIndicators,2000);}
    scheduleHistoricalBubbleDraw();
  }
  function applyRealtimeCandle(candle){
    if(!tvCandle||!tvLastCandles.length||!popIsSessionTime(candle.time))return;
    var bucketTime=popBucketStart(candle.time);
    if(bucketTime===null)return;
    var minuteBars=(popoutTimeframe||1)<=1;
    var raw=popUpsertRaw(bucketTime,function(bar){
      if(minuteBars)return{time:bucketTime,open:candle.open,high:candle.high,low:candle.low,close:candle.close,volume:candle.volume||0};
      if(!popStreamBucketState||popStreamBucketState.bucketTime!==bucketTime){popStreamBucketState={bucketTime:bucketTime,baseVolume:bar?(bar.volume||0):0,minutes:{},lastMinute:0};}
      popStreamBucketState.minutes[candle.time]=candle.volume||0;
      var latest=candle.time>=popStreamBucketState.lastMinute;popStreamBucketState.lastMinute=Math.max(popStreamBucketState.lastMinute,candle.time);
      var streamed=0;Object.keys(popStreamBucketState.minutes).forEach(function(k){streamed+=popStreamBucketState.minutes[k];});
      var volume=popStreamBucketState.baseVolume+streamed;
      if(!bar)return{time:bucketTime,open:candle.open,high:candle.high,low:candle.low,close:candle.close,volume:volume};
      return Object.assign({},bar,{high:Math.max(bar.high,candle.high),low:Math.min(bar.low,candle.low),close:latest?candle.close:bar.close,volume:volume});
    });
    popApplyRaw(raw);
    if(activeInds.size>0)applyIndicators();
    scheduleHistoricalBubbleDraw();
  }

  // ── SSE price stream ───────────────────────────────────────────────────────
  var popoutEvtSource=null,popoutSseTicker=null;
  function connectPopoutStream(ticker){
    if(!ticker)return;
    var upper=ticker.toUpperCase();
    if(popoutEvtSource&&popoutSseTicker===upper&&popoutEvtSource.readyState!==2)return;
    if(popoutEvtSource){try{popoutEvtSource.close();}catch(e){}}
    popoutEvtSource=new EventSource('/price_stream/'+encodeURIComponent(upper));
    popoutSseTicker=upper;
    popoutEvtSource.onmessage=function(ev){
      try{
        var msg=JSON.parse(ev.data);
        if(msg.type==='quote'&&typeof msg.last==='number'){applyRealtimeQuote(msg.last);}
        else if(msg.type==='candle'&&msg.time){applyRealtimeCandle(msg);}
      }catch(e){}
    };
    popoutEvtSource.onerror=function(){console.debug('[Popout] SSE error – browser will retry.');};
  }

  // ── Initial candle load + settings helpers ────────────────────────────────
  var popoutFetching=false,popoutCurrentTicker=null;
  function getSettingsFromOpener(){
    try{
      var op=window.opener;if(!op||op.closed)return null;
      var d=op.document;
      function val(id){var el=d.getElementById(id);return el?el.value:null;}
    function chk(id,defaultValue){var el=d.getElementById(id);return el?el.checked:!!defaultValue;}
      var ticker=val('ticker');if(!ticker)return null;
            var expiry=[];
      var levelsTypes=[];
            try{expiry=Array.from(d.querySelectorAll('.expiry-option input[type="checkbox"]:checked')).map(function(cb){return cb.value;});}catch(e){}
            try{levelsTypes=op.getSelectedPriceLevelTypes?op.getSelectedPriceLevelTypes():Array.from(d.querySelectorAll('.levels-option input:checked')).map(function(cb){return cb.value;});}catch(e){}
            var themePayload=null;
            try{if(op.buildPopoutThemePayload)themePayload=op.buildPopoutThemePayload();}catch(e){}
                                                return{ticker:ticker,expiry:expiry,timeframe:val('timeframe')||'1',call_color:val('call_color')||'#00ff00',put_color:val('put_color')||'#ff0000',levels_types:levelsTypes,levels_count:parseInt(val('levels_count'))||3,use_heikin_ashi:chk('use_heikin_ashi'),strike_range:parseFloat(val('strike_range'))/100||0.1,highlight_max_level:chk('highlight_max_level'),show_latest_level_lines:op.getShowLatestLevelLinesSetting?op.getShowLatestLevelLinesSetting():chk('show_latest_level_lines',true),price_above_bubbles:op.getPriceAboveBubblesSetting?op.getPriceAboveBubblesSetting():false,max_level_color:val('max_level_color')||'#800080',coloring_mode:val('coloring_mode')||'Linear Intensity',exposure_metric:val('exposure_metric')||'Open Interest',delta_adjusted:chk('delta_adjusted_exposures'),calculate_in_notional:chk('calculate_in_notional',true),theme_payload:themePayload};
    }catch(e){return null;}
  }
  function loadInitialData(){
    if(popoutFetching)return;
    var settings=getSettingsFromOpener();
    if(!settings||!settings.ticker)return;
        if(settings.theme_payload)applyPopoutTheme(settings.theme_payload);
        setPriceAboveBubblesSetting(settings.price_above_bubbles);
    popoutFetching=true;
    fetch('/update_price',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(settings)})
      .then(function(r){return r.json();})
      .then(function(data){
        if(!data.error&&data.price){
          renderPriceChart(typeof data.price==='string'?JSON.parse(data.price):data.price);
        }
        // Connect SSE after we have the initial candle history
        connectPopoutStream(settings.ticker);
      })
      .catch(function(e){console.warn('Popout initial load error:',e);connectPopoutStream(settings.ticker);})
      .finally(function(){popoutFetching=false;});
  }

  // ── Ticker-change watcher (lightweight DOM read only) ─────────────────────
  // Reconnects SSE and reloads candle history whenever the ticker changes.
  // Exposure levels are refreshed periodically since options data changes.
  function tickerWatchLoop(){
    var settings=getSettingsFromOpener();
        if(settings&&settings.theme_payload)applyPopoutTheme(settings.theme_payload);
    var ticker=settings?settings.ticker:null;
    // Only commit the new ticker once a load can actually start; otherwise retry next tick
    if(ticker&&ticker!==popoutCurrentTicker&&!popoutFetching){
      popoutCurrentTicker=ticker;
      loadInitialData();
    }
  }
  // Refresh exposure levels every 60 s (options cache updated by main /update cycle)
  function refreshExposureLevels(){
    if(popoutFetching||!popoutCurrentTicker)return;
    var settings=getSettingsFromOpener();
    if(!settings)return;
        setPriceAboveBubblesSetting(settings.price_above_bubbles);
    popoutFetching=true;
    fetch('/update_price',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(settings)})
      .then(function(r){return r.json();})
      .then(function(data){
        if(!data.error&&data.price){
          var pd=typeof data.price==='string'?JSON.parse(data.price):data.price;
          // Only refresh price lines (exposure levels + expected moves), not candles
          if(tvCandle){
            tvPriceLines.forEach(function(l){try{tvCandle.removePriceLine(l);}catch(e){}});
            tvPriceLines=[];tvAllLevelPrices=[];
                                                tvShowLatestLevelLines=pd.show_latest_level_lines!==false;
                        tvHistoricalPoints=priceLevelBubblePoints(pd);
                        pushLatestLevelPrices();
                        scheduleHistoricalBubbleDraw();
            tvApplyAutoscale();
          }
        }
      })
      .catch(function(e){console.warn('Popout exposure refresh error:',e);})
      .finally(function(){popoutFetching=false;});
  }

  // Kick off: initial load then watch for ticker changes every 3 s
  setTimeout(function(){
    loadInitialData();
    setInterval(tickerWatchLoop,3000);
    setInterval(refreshExposureLevels,60000);
  },300);

  // Entry point kept for compatibility with pushDataToPopout
  window.updatePopoutChart=function(priceDataJSON){
        var themePayload=arguments.length>2?arguments[2]:null;
        try{if(themePayload)applyPopoutTheme(themePayload);var priceData=typeof priceDataJSON==='string'?JSON.parse(priceDataJSON):priceDataJSON;if(!priceData||priceData.error)return;renderPriceChart(priceData);}catch(e){console.error('Popout price chart error:',e);}
  };
    window.addEventListener('resize',function(){scheduleHistoricalBubbleDraw();if(tvChart&&tvAutoRange){try{tvChart.timeScale().fitContent();}catch(e){}}});
  window.addEventListener('beforeunload',function(){if(popoutEvtSource){try{popoutEvtSource.close();}catch(e){}}});
<\\/script></body></html>`);
            } else {
                popup.document.write(`<!DOCTYPE html>
<html><head><title>${displayName} - EzOptions</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<script src="https://cdn.plot.ly/plotly-1.58.5.min.js"><\\/script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { --app-bg:#1E1E1E; --panel-bg:#1a1a1a; --panel-bg-alt:#2D2D2D; --chart-bg:#1E1E1E; --border-color:#333; --text-primary:#eef2f7; --text-secondary:#ccc; --text-muted:#888; --accent-color:#800080; --grid-color:#2A2A2A; --tooltip-bg:linear-gradient(180deg, rgba(30, 36, 46, 0.97), rgba(14, 18, 24, 0.99)); --tooltip-border:rgba(255,255,255,0.08); }
    body { background: var(--app-bg); color: var(--text-secondary); overflow: hidden; position: relative; font-family: Arial, sans-serif; }
    #popout-logo { position: fixed; top: 6px; left: 10px; z-index: 100; font-family: Arial, sans-serif; font-size: 11px; font-weight: bold; color: var(--accent-color); opacity: 0.7; pointer-events: none; letter-spacing: 0.5px; }
  #popout-plot { width: 100vw; height: 100vh; }
    #popout-html { width: 100vw; height: 100vh; overflow: auto; background: var(--chart-bg); color: var(--text-primary); font-family: Arial, sans-serif; }
        #popout-plot .hoverlayer { opacity: 0 !important; visibility: hidden !important; pointer-events: none !important; }
        #popout-plot .hoverlayer *, #popout-plot g.hovertext, #popout-plot g.hovertext * { opacity: 0 !important; visibility: hidden !important; pointer-events: none !important; }
        .chart-hover-tooltip { position:absolute; z-index:55; display:none; width:auto !important; height:auto !important; min-width:0; max-width:min(300px,calc(100% - 16px)); padding:8px; border:1px solid var(--tooltip-border); border-radius:10px; background:var(--tooltip-bg); color:var(--text-primary); font-size:10px; line-height:1.25; font-family:Arial, sans-serif; pointer-events:none; box-shadow:0 14px 36px rgba(0,0,0,0.38); backdrop-filter:blur(10px); overflow:hidden; white-space:normal; }
        .chart-hover-tooltip .tt-head { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:6px; }
        .chart-hover-tooltip .tt-badge { padding:2px 6px; border-radius:999px; background:rgba(255,255,255,0.08); color:var(--text-secondary); font-size:9px; letter-spacing:0.02em; text-transform:uppercase; }
        .chart-hover-tooltip .tt-time, .chart-hover-tooltip .tt-more { color:var(--text-muted); }
        .chart-hover-tooltip .tt-list { display:grid; gap:4px; }
        .chart-hover-tooltip .tt-row { display:flex; align-items:center; gap:6px; min-width:0; }
        .chart-hover-tooltip .tt-dot { width:7px; height:7px; border-radius:999px; box-shadow:0 0 0 1px rgba(255,255,255,0.12); flex:0 0 auto; }
        .chart-hover-tooltip .tt-main { display:flex; justify-content:space-between; align-items:baseline; gap:8px; min-width:0; width:100%; }
        .chart-hover-tooltip .tt-name { color:var(--text-primary); font-weight:600; flex-shrink:0; white-space:nowrap; }
        .chart-hover-tooltip .tt-value { color:var(--text-muted); font-variant-numeric:tabular-nums; white-space:nowrap; flex:0 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; text-align:right; }
        .chart-hover-tooltip .tt-more { margin-top:4px; padding-top:4px; border-top:1px solid rgba(255,255,255,0.06); font-size:9px; }
</style></head><body>
<div id="popout-logo">EzDuz1t Options</div>
<div id="popout-plot"></div>
<div id="popout-html" style="display:none;"></div>
<script>
  let plotInited = false;
    function applyPlotlyThemeToLayout(layout) {
        const chartBg = getThemeValue('--chart-bg', '#1E1E1E');
        const panelBg = getThemeValue('--panel-bg', '#1a1a1a');
        const panelBgAlt = getThemeValue('--panel-bg-alt', '#2D2D2D');
        const textSecondary = getThemeValue('--text-secondary', '#ccc');
        const textMuted = getThemeValue('--text-muted', '#888');
        const borderColor = getThemeValue('--border-color', '#333');
        const gridColor = getThemeValue('--grid-color', '#2A2A2A');
        layout.plot_bgcolor = chartBg;
        layout.paper_bgcolor = panelBg;
        layout.font = Object.assign({}, layout.font || {}, { color: textSecondary });
        if (layout.title) {
            layout.title.font = Object.assign({}, layout.title.font || {}, { color: textSecondary });
        }
        if (layout.legend) {
            layout.legend.bgcolor = panelBgAlt;
            layout.legend.font = Object.assign({}, layout.legend.font || {}, { color: textSecondary });
        }
        // Always force hoverlabel to transparent — custom JS cards replace all native hover display
        layout.hoverlabel = Object.assign({}, layout.hoverlabel || {}, {
            bgcolor: 'rgba(0,0,0,0)',
            bordercolor: 'rgba(0,0,0,0)',
            font: Object.assign({}, (layout.hoverlabel || {}).font || {}, { color: 'rgba(0,0,0,0)', size: 1 }),
            namelength: 0,
        });
        ['xaxis', 'yaxis', 'xaxis2', 'yaxis2'].forEach(axisKey => {
            const axis = layout[axisKey];
            if (!axis) return;
            axis.gridcolor = gridColor;
            axis.linecolor = borderColor;
            axis.zerolinecolor = gridColor;
            axis.tickcolor = textMuted;
            axis.tickfont = Object.assign({}, axis.tickfont || {}, { color: textSecondary });
            if (axis.title && typeof axis.title === 'object') {
                axis.title.font = Object.assign({}, axis.title.font || {}, { color: textSecondary });
            }
            if (axis.title_font) {
                axis.title_font.color = textSecondary;
            }
        });
    }
    function applyPopoutTheme(themePayload) {
        if (!themePayload || !themePayload.vars) return;
        Object.keys(themePayload.vars).forEach(key => document.documentElement.style.setProperty(key, themePayload.vars[key]));
        document.body.dataset.theme = themePayload.name || 'dark';
        const plotDiv = document.getElementById('popout-plot');
        if (plotDiv && plotDiv.classList.contains('js-plotly-plot')) {
            try {
                Plotly.relayout(plotDiv, {
                    plot_bgcolor: getThemeValue('--chart-bg', '#1E1E1E'),
                    paper_bgcolor: getThemeValue('--panel-bg', '#1a1a1a'),
                    'font.color': getThemeValue('--text-secondary', '#ccc'),
                    'hoverlabel.bgcolor': getThemeValue('--panel-bg-alt', '#2D2D2D'),
                    'hoverlabel.bordercolor': getThemeValue('--border-color', '#333'),
                    'legend.bgcolor': getThemeValue('--panel-bg-alt', '#2D2D2D'),
                    'legend.font.color': getThemeValue('--text-secondary', '#ccc'),
                    'title.font.color': getThemeValue('--text-secondary', '#ccc'),
                    'xaxis.gridcolor': getThemeValue('--grid-color', '#2A2A2A'),
                    'xaxis.linecolor': getThemeValue('--border-color', '#333'),
                    'xaxis.zerolinecolor': getThemeValue('--grid-color', '#2A2A2A'),
                    'xaxis.tickfont.color': getThemeValue('--text-secondary', '#ccc'),
                    'yaxis.gridcolor': getThemeValue('--grid-color', '#2A2A2A'),
                    'yaxis.linecolor': getThemeValue('--border-color', '#333'),
                    'yaxis.zerolinecolor': getThemeValue('--grid-color', '#2A2A2A'),
                    'yaxis.tickfont.color': getThemeValue('--text-secondary', '#ccc')
                });
            } catch (e) {}
        }
    }
    // Shared helpers (source injected from the main window)
${popoutHelperSource([getThemeValue, escapeTooltipHtml, stripTooltipHtml, formatTooltipNumber, formatTooltipMoney,
    formatTooltipDateTime, parseTooltipColor, formatTooltipRgb, interpolateTooltipColor, resolveHeatmapPointColor,
    resolvePlotlyPointColor, buildPlotlyTooltipContext, buildPlotlyTooltipRow, ensurePlotlyTooltip,
    hidePlotlyTooltip, positionPlotlyTooltip, attachPlotlyCustomTooltip])}
    // Options chain table header clicks call sortTable(); define it here too since the
    // popped-out table lives in this document.
    var popoutChainSort = null;
    function applyPopoutChainSort() {
        if (!popoutChainSort) return;
        Array.prototype.forEach.call(document.querySelectorAll('table.oc-table'), sortPopoutChainTable);
    }
    function sortPopoutChainTable(table) {
        var col = popoutChainSort.col, type = popoutChainSort.type, dir = popoutChainSort.dir;
        var tbody = table.tBodies[0];
        var markers = Array.prototype.slice.call(tbody.querySelectorAll('tr.oc-marker'));
        var rows = Array.prototype.slice.call(tbody.rows).filter(function(row) { return !row.classList.contains('oc-marker'); });
        function cellValue(row) {
            var cell = row.cells[col];
            if (type === 'number') { var v = parseFloat(cell.getAttribute('data-sort')); return isNaN(v) ? 0 : v; }
            return cell.textContent.toLowerCase();
        }
        rows.sort(function(a, b) { var x = cellValue(a), y = cellValue(b); var c = x < y ? -1 : x > y ? 1 : 0; return dir === 'asc' ? c : -c; });
        rows.forEach(function(row) { tbody.appendChild(row); });
        if (markers.length) {
            var strikeHeader = table.querySelector('th.k[data-col]');
            var byStrike = strikeHeader && parseInt(strikeHeader.getAttribute('data-col'), 10) === col;
            var markerPrice = function(row) { return parseFloat(row.getAttribute('data-price')); };
            markers.sort(function(a, b) { return dir === 'asc' ? markerPrice(a) - markerPrice(b) : markerPrice(b) - markerPrice(a); });
            markers.forEach(function(marker) {
                marker.style.display = byStrike ? '' : 'none';
                if (!byStrike) return;
                var price = markerPrice(marker), next = null;
                for (var i = 0; i < rows.length; i++) {
                    var k = parseFloat(rows[i].cells[col].getAttribute('data-sort'));
                    if (dir === 'asc' ? k > price : k < price) { next = rows[i]; break; }
                }
                tbody.insertBefore(marker, next);
            });
        }
        Array.prototype.forEach.call(table.querySelectorAll('th[data-col]'), function(header) {
            var span = header.querySelector('.oc-sort');
            if (!span) return;
            var active = parseInt(header.getAttribute('data-col'), 10) === col;
            span.textContent = active ? (dir === 'asc' ? '▲' : '▼') : '▼▲';
            span.classList.toggle('active', active);
        });
    }
    var popoutChainExpiry = null;
    var popoutChainScroll = {};
    function rememberPopoutChainScroll(root) {
        Array.prototype.forEach.call(root.querySelectorAll('.oc-section:not([hidden])'), function(section) {
            var scroller = section.querySelector('.oc-scroll');
            if (scroller) popoutChainScroll[section.getAttribute('data-expiry')] = scroller.scrollTop;
        });
    }
    function showPopoutChainSection(root, expiry) {
        var sections = Array.prototype.slice.call(root.querySelectorAll('.oc-section'));
        if (!sections.length) return;
        var target = sections.filter(function(section) { return section.getAttribute('data-expiry') === expiry; })[0] || sections[0];
        var key = target.getAttribute('data-expiry');
        sections.forEach(function(section) { section.hidden = section !== target; });
        Array.prototype.forEach.call(root.querySelectorAll('.oc-tab'), function(tab) {
            tab.classList.toggle('active', tab.getAttribute('data-expiry') === key);
        });
        popoutChainExpiry = key;
        var scroller = target.querySelector('.oc-scroll');
        if (!scroller) return;
        if (popoutChainScroll[key] !== undefined) { scroller.scrollTop = popoutChainScroll[key]; return; }
        var spotRow = scroller.querySelector('tr.oc-spot');
        if (spotRow && spotRow.style.display !== 'none') scroller.scrollTop = Math.max(0, spotRow.offsetTop - scroller.clientHeight / 2);
        popoutChainScroll[key] = scroller.scrollTop;
    }
    window.selectChainExpiry = function(tab) {
        var root = tab.closest('.oc-wrap');
        if (!root) return;
        rememberPopoutChainScroll(root);
        showPopoutChainSection(root, tab.getAttribute('data-expiry'));
    };
    window.sortTable = function(col, type) {
        var dir = popoutChainSort && popoutChainSort.col === col && popoutChainSort.dir === 'asc' ? 'desc' : 'asc';
        popoutChainSort = { col: col, type: type, dir: dir };
        applyPopoutChainSort();
    };
    window.updatePopoutChart = function(chartDataJSON, isHtml, themePayload) {
        if (themePayload) {
            applyPopoutTheme(themePayload);
        }
    if (isHtml) {
      document.getElementById('popout-plot').style.display = 'none';
      const htmlDiv = document.getElementById('popout-html');
      htmlDiv.style.display = 'block';
            rememberPopoutChainScroll(htmlDiv);
            htmlDiv.innerHTML = chartDataJSON || '';
            applyPopoutChainSort();
            showPopoutChainSection(htmlDiv, popoutChainExpiry);
      return;
    }
        if (!chartDataJSON) {
            return;
        }
    document.getElementById('popout-html').style.display = 'none';
    const plotDiv = document.getElementById('popout-plot');
    plotDiv.style.display = 'block';
    try {
      const chartData = JSON.parse(chartDataJSON);
      chartData.layout.autosize = true;
      chartData.layout.width = null;
      chartData.layout.height = null;
    chartData.layout.margin = chartData.layout.margin || { l: 60, r: 60, t: 60, b: 40 };
            applyPlotlyThemeToLayout(chartData.layout);
      const config = { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d'], displaylogo: false, scrollZoom: true };
            let plotPromise;
      if (plotInited) {
                plotPromise = Plotly.react('popout-plot', chartData.data, chartData.layout, config);
      } else {
                plotPromise = Plotly.newPlot('popout-plot', chartData.data, chartData.layout, config);
        plotInited = true;
      }
            Promise.resolve(plotPromise).then(function() { attachPlotlyCustomTooltip(document.getElementById('popout-plot')); });
      updatePopoutHeatmapTextSize();
    } catch(e) { console.error('Popout chart error:', e); }
  };
  function updatePopoutHeatmapTextSize() {
    const div = document.getElementById('popout-plot');
    if (!div || !div._fullLayout) return;
    const fl = div._fullLayout;
    if (!fl.annotations || fl.annotations.length === 0) return;
    const cellAnnotIdxs = [];
    fl.annotations.forEach((a, i) => {
      if (a.xref === 'x' && a.yref === 'y' && !a.showarrow) cellAnnotIdxs.push(i);
    });
    if (cellAnnotIdxs.length === 0) return;
    const trace = div._fullData && div._fullData[0];
    const ncols = (trace && trace.x) ? trace.x.length : 1;
    const nrows = (trace && trace.y) ? trace.y.length : 1;
    const plotW = (fl.width  || div.offsetWidth)  - (fl.margin.l + fl.margin.r);
    const plotH = (fl.height || div.offsetHeight) - (fl.margin.t + fl.margin.b);
    const cellW = plotW / Math.max(1, ncols);
    const cellH = plotH / Math.max(1, nrows);
    const newSize = Math.max(6, Math.min(20, Math.floor(Math.min(cellW, cellH) * 0.38)));
    if (fl.annotations[cellAnnotIdxs[0]].font && fl.annotations[cellAnnotIdxs[0]].font.size === newSize) return;
    const relayoutUpdate = { 'transition.duration': 0 };
    cellAnnotIdxs.forEach(i => { relayoutUpdate['annotations[' + i + '].font.size'] = newSize; });
    try { Plotly.relayout(div, relayoutUpdate); } catch(e) {}
  }
  window.addEventListener('resize', function() {
    const el = document.getElementById('popout-plot');
    if (el && el.classList.contains('js-plotly-plot')) {
      try { Plotly.Plots.resize(el); } catch(e) {}
      setTimeout(updatePopoutHeatmapTextSize, 80);
    }
  });
<\\/script></body></html>`);
            }
            popup.document.close();

            popoutWindows[chartId] = popup;

            // Push initial data after a small delay so the popup's DOM is ready
            setTimeout(() => { pushDataToPopout(chartId); }, 300);

            // Clean up reference when popup closes
            const checkClosed = setInterval(() => {
                if (popup.closed) {
                    clearInterval(checkClosed);
                    delete popoutWindows[chartId];
                }
            }, 1000);
        }

        function pushDataToPopout(chartId) {
            const popup = popoutWindows[chartId];
            if (!popup || popup.closed) { delete popoutWindows[chartId]; return; }
            if (typeof popup.updatePopoutChart !== 'function') return; // not ready yet

            // Determine the data key from chart id  (e.g. 'gamma-chart' -> 'gamma', 'price-chart' -> 'price')
            const dataKey = chartId.replace('-chart', '');
            // Price data is stored separately since it's fetched via /update_price
            // Price data goes over as JSON so the popout gets its own copy of the candle arrays
            // (it pushes live candles into them, which would corrupt the main chart's state).
            const chartPayload = (dataKey === 'price')
                ? (lastPriceData ? JSON.stringify(lastPriceData) : null)
                : lastData[dataKey];

            const isHtml = (dataKey === 'large_trades');
            try {
                popup.updatePopoutChart(chartPayload || null, isHtml, buildPopoutThemePayload());
            } catch(e) {
                // popup may have navigated away or been closed
                console.warn('Could not push to popout:', e);
            }
        }

        function pushAllPopouts() {
            Object.keys(popoutWindows).forEach(chartId => pushDataToPopout(chartId));
        }

        function addPopoutButton(container) {
            if (!container || container.querySelector('.chart-popout-btn')) return;
            const btn = document.createElement('button');
            btn.className = 'chart-popout-btn';
            btn.innerHTML = popoutSvg;
            btn.title = 'Pop out chart to separate window';
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                e.preventDefault();
                openPopoutChart(container.id);
            });
            container.appendChild(btn);
        }

        // Clean up popout windows on page unload
        window.addEventListener('beforeunload', function() {
            Object.values(popoutWindows).forEach(w => { try { w.close(); } catch(e) {} });
        });
        // --- End pop-out support ---

        function showError(message) {
            const notification = document.getElementById('error-notification');
            const messageElement = document.getElementById('error-message');
            messageElement.textContent = message;
            notification.style.display = 'block';
            
            // Auto-hide after 10 seconds unless it's a persistent error
            setTimeout(hideError, 10000);
        }

        function hideError() {
            document.getElementById('error-notification').style.display = 'none';
        }
        
        // Update colors when color pickers change
        document.getElementById('call_color').addEventListener('change', function(e) {
            callColor = e.target.value;
            updateData();
        });
        
        document.getElementById('put_color').addEventListener('change', function(e) {
            putColor = e.target.value;
            updateData();
        });

        document.getElementById('max_level_color').addEventListener('change', function(e) {
            maxLevelColor = e.target.value;
            updateData();
        });

        // highlight_max_level is also covered by the generic .control-group checkbox listener
        document.getElementById('max_level_mode').addEventListener('change', updateData);
        
        // Update strike range value display
        document.getElementById('strike_range').addEventListener('input', function() {
            document.getElementById('strike_range_value').textContent = this.value + '%';
            updateData();
        });

        // EM range lock toggle state
        let emRangeLocked = false;

        function applyEmRange(em, triggerUpdate) {
            if (!em || em.upper_pct == null) return false;
            const emPct = Math.abs(em.upper_pct);
            const withWiggle = emPct + 0.5;
            const stepped = Math.round(withWiggle / 0.5) * 0.5;
            const clamped = Math.min(20, Math.max(0.5, stepped));
            const slider = document.getElementById('strike_range');
            if (parseFloat(slider.value) === clamped) return true; // no change needed
            slider.value = clamped;
            document.getElementById('strike_range_value').textContent = clamped + '%';
            if (triggerUpdate) updateData();
            return true;
        }

        function setEmRangeLocked(locked) {
            emRangeLocked = locked;
            const btn = document.getElementById('match_em_range');
            btn.classList.toggle('active', locked);
            btn.title = locked
                ? 'EM Range Lock ON — click to disable'
                : 'Toggle: auto-sync strike range to Expected Move (ATM straddle) + 0.5% wiggle room';
        }

        // Match EM range button: toggle auto-sync of strike range to EM
        document.getElementById('match_em_range').addEventListener('click', function() {
            if (emRangeLocked) {
                setEmRangeLocked(false);
            } else {
                setEmRangeLocked(true);
                // Apply immediately if EM data is already available
                const em = lastData && lastData.price_info && lastData.price_info.expected_move_range;
                if (!applyEmRange(em, true)) {
                    alert('Expected Move data not yet available. Fetch data first.');
                    setEmRangeLocked(false);
                }
            }
        });

        // Coloring mode listeners
        document.getElementById('timeframe').addEventListener('change', function() {
            updateData();
            startCandleCloseTimer();
        });
        document.getElementById('coloring_mode').addEventListener('change', updateData);
        document.getElementById('exposure_metric').addEventListener('change', updateData);
        document.getElementById('heatmap_type').addEventListener('change', updateHeatmapOnly);
        document.getElementById('heatmap_coloring_mode').addEventListener('change', updateHeatmapOnly);
        document.getElementById('levels_count').addEventListener('input', refreshPriceLevelsOnly);
        document.getElementById('abs_gex_opacity').addEventListener('input', updateData);

        function getSelectedExpiryValues() {
            return Array.from(document.querySelectorAll('.expiry-option input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        }

        function escapeTooltipHtml(value) {
            return String(value == null ? '' : value)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        function stripTooltipHtml(value) {
            return String(value == null ? '' : value).replace(/<[^>]*>/g, ' ').replace(/\\s+/g, ' ').trim();
        }

        function formatTooltipNumber(value, maxFractionDigits = 2) {
            const numericValue = Number(value);
            if (!Number.isFinite(numericValue)) return String(value == null ? '' : value);
            const fractionDigits = Number.isInteger(numericValue) ? 0 : maxFractionDigits;
            return numericValue.toLocaleString('en-US', {
                minimumFractionDigits: 0,
                maximumFractionDigits: fractionDigits,
            });
        }

        function formatLargeNumberCompact(value) {
            const numericValue = Number(value);
            if (!Number.isFinite(numericValue)) return '';
            const abs = Math.abs(numericValue);
            if (abs >= 1e12) return (numericValue / 1e12).toFixed(2) + 'T';
            if (abs >= 1e9) return (numericValue / 1e9).toFixed(2) + 'B';
            if (abs >= 1e6) return (numericValue / 1e6).toFixed(2) + 'M';
            if (abs >= 1e3) return (numericValue / 1e3).toFixed(2) + 'K';
            return Math.round(numericValue).toLocaleString('en-US');
        }

        function formatTooltipMoney(value) {
            const numericValue = Number(value);
            if (!Number.isFinite(numericValue)) return String(value == null ? '' : value);
            return `$${numericValue.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            })}`;
        }

        function formatTooltipDateTime(value) {
            if (value == null || value === '') return '';
            if (typeof value === 'string' && /^\\d{2}:\\d{2}(:\\d{2})?(\\s*[A-Z]{2,4})?$/.test(value.trim())) {
                return value;
            }
            const parsedDate = value instanceof Date ? value : new Date(value);
            if (!Number.isFinite(parsedDate.getTime())) return String(value);
            const sameDay = parsedDate.toDateString() === new Date().toDateString();
            return parsedDate.toLocaleString('en-US', sameDay ? {
                hour: '2-digit',
                minute: '2-digit',
            } : {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
            });
        }

        function parseTooltipColor(colorValue) {
            if (typeof colorValue !== 'string') return null;
            const color = colorValue.trim();
            if (color.startsWith('#')) {
                const hex = color.slice(1);
                if (hex.length === 3) {
                    return {
                        r: parseInt(hex[0] + hex[0], 16),
                        g: parseInt(hex[1] + hex[1], 16),
                        b: parseInt(hex[2] + hex[2], 16),
                    };
                }
                if (hex.length === 6) {
                    return {
                        r: parseInt(hex.slice(0, 2), 16),
                        g: parseInt(hex.slice(2, 4), 16),
                        b: parseInt(hex.slice(4, 6), 16),
                    };
                }
            }
            const rgbMatch = color.match(/rgba?\\(([^)]+)\\)/i);
            if (!rgbMatch) return null;
            const parts = rgbMatch[1].split(',').map(part => Number.parseFloat(part.trim()));
            if (parts.length < 3 || parts.some(part => !Number.isFinite(part))) return null;
            return { r: parts[0], g: parts[1], b: parts[2] };
        }

        function formatTooltipRgb(color) {
            const clamp = value => Math.max(0, Math.min(255, Math.round(value)));
            return `rgb(${clamp(color.r)}, ${clamp(color.g)}, ${clamp(color.b)})`;
        }

        function interpolateTooltipColor(leftColor, rightColor, ratio) {
            const clampedRatio = Math.max(0, Math.min(1, ratio));
            return formatTooltipRgb({
                r: leftColor.r + ((rightColor.r - leftColor.r) * clampedRatio),
                g: leftColor.g + ((rightColor.g - leftColor.g) * clampedRatio),
                b: leftColor.b + ((rightColor.b - leftColor.b) * clampedRatio),
            });
        }

        function resolveHeatmapPointColor(point) {
            const colorscale = point?.fullData?.colorscale;
            if (!Array.isArray(colorscale) || !colorscale.length) return null;

            const zValue = Number(point?.z);
            const zMin = Number(point?.fullData?.zmin);
            const zMax = Number(point?.fullData?.zmax);
            if (!Number.isFinite(zValue) || !Number.isFinite(zMin) || !Number.isFinite(zMax) || zMax === zMin) {
                return null;
            }

            const normalizedValue = Math.max(0, Math.min(1, (zValue - zMin) / (zMax - zMin)));
            const normalizedScale = colorscale
                .map(stop => ({
                    offset: Array.isArray(stop) ? Number(stop[0]) : Number(stop?.offset),
                    color: parseTooltipColor(Array.isArray(stop) ? stop[1] : stop?.color),
                }))
                .filter(stop => Number.isFinite(stop.offset) && stop.color)
                .sort((left, right) => left.offset - right.offset);

            if (!normalizedScale.length) return null;
            if (normalizedValue <= normalizedScale[0].offset) {
                return formatTooltipRgb(normalizedScale[0].color);
            }
            for (let index = 1; index < normalizedScale.length; index += 1) {
                const leftStop = normalizedScale[index - 1];
                const rightStop = normalizedScale[index];
                if (normalizedValue <= rightStop.offset) {
                    const span = rightStop.offset - leftStop.offset || 1;
                    return interpolateTooltipColor(leftStop.color, rightStop.color, (normalizedValue - leftStop.offset) / span);
                }
            }
            return formatTooltipRgb(normalizedScale[normalizedScale.length - 1].color);
        }

        function resolvePlotlyPointColor(point) {
            if (point?.fullData?.type === 'heatmap') {
                const heatmapColor = resolveHeatmapPointColor(point);
                if (heatmapColor) return heatmapColor;
            }
            const marker = point?.fullData?.marker;
            if (marker && marker.color != null) {
                const markerColor = Array.isArray(marker.color) ? marker.color[point.pointNumber] : marker.color;
                if (typeof markerColor === 'string') return markerColor;
            }
            if (marker && marker.colors != null) {
                const markerColors = Array.isArray(marker.colors) ? marker.colors[point.pointNumber] : marker.colors;
                if (typeof markerColors === 'string') return markerColors;
            }
            const lineColor = point?.fullData?.line?.color;
            if (typeof lineColor === 'string') return lineColor;
            return getThemeValue('--accent-color', '#5ab0ff');
        }

        function buildPlotlyTooltipContext(points, plotDiv) {
            const firstPoint = points[0] || {};
            if (firstPoint?.fullData?.type === 'pie') {
                const titleText = stripTooltipHtml(plotDiv?._fullLayout?.title?.text || 'Breakdown');
                return escapeTooltipHtml(titleText || 'Breakdown');
            }
            if (firstPoint?.fullData?.type === 'heatmap') {
                return `Exp ${escapeTooltipHtml(firstPoint.x)} • Strike ${escapeTooltipHtml(formatTooltipMoney(firstPoint.y))}`;
            }
            if (firstPoint?.fullData?.orientation === 'h') {
                return `Strike ${escapeTooltipHtml(formatTooltipMoney(firstPoint.y))}`;
            }
            if (firstPoint?.customdata && Array.isArray(firstPoint.customdata) && firstPoint.customdata.length >= 3) {
                return escapeTooltipHtml(formatTooltipDateTime(firstPoint.x));
            }
            if (firstPoint?.x != null) {
                if (typeof firstPoint.x === 'number') {
                    return `Strike ${escapeTooltipHtml(formatTooltipMoney(firstPoint.x))}`;
                }
                return escapeTooltipHtml(formatTooltipDateTime(firstPoint.x));
            }
            const titleText = stripTooltipHtml(plotDiv?._fullLayout?.title?.text || 'Details');
            return escapeTooltipHtml(titleText || 'Details');
        }

        function buildPlotlyTooltipRow(point, plotDiv) {
            const traceName = stripTooltipHtml(point?.fullData?.name || point?.data?.name || '');
            const isBubblePoint = point?.customdata && Array.isArray(point.customdata) && point.customdata.length >= 3;
            const isPiePoint = point?.fullData?.type === 'pie';
            const isHeatmapPoint = point?.fullData?.type === 'heatmap';
            const isCentroidPoint = /centroid/i.test(traceName);
            const chartTitle = stripTooltipHtml(plotDiv?._fullLayout?.title?.text || '');
            let name = traceName && !/^trace\\s+\\d+$/i.test(traceName) ? traceName : 'Value';
            let value = '';

            if (isPiePoint) {
                name = point.label || name;
                const percentValue = typeof point.percent === 'number'
                    ? `  ${formatTooltipNumber(point.percent * 100)}%`
                    : '';
                value = `${formatTooltipNumber(point.value, 0)}${percentValue}`;
            } else if (isBubblePoint && traceName !== 'Price') {
                name = `${point.customdata[0]} ${name}`.trim();
                value = `${formatTooltipMoney(point.customdata[1])}  ${formatTooltipNumber(point.customdata[2])}`;
            } else if (isHeatmapPoint) {
                name = name === 'Value' ? 'Exposure' : name;
                value = point.customdata != null ? formatTooltipNumber(point.customdata) : formatTooltipNumber(point.z);
            } else if (isCentroidPoint) {
                const centroidValue = formatTooltipMoney(point.y);
                const volumeValue = point.customdata != null ? `Vol ${formatTooltipNumber(point.customdata, 0)}` : '';
                value = volumeValue ? `${centroidValue}  ${volumeValue}` : centroidValue;
            } else if (/price/i.test(name)) {
                value = formatTooltipMoney(point.y != null ? point.y : point.x);
            } else {
                let rawValue = point?.fullData?.orientation === 'h' ? point.x : point.y;
                const isPutSide = /^(Put|Sell)$/i.test(traceName);
                const hasText = point?.text != null && String(point.text).trim() !== '';
                if (isPutSide && typeof rawValue === 'number') {
                    rawValue = Math.abs(rawValue);
                }
                if (isPutSide && hasText && !/premium/i.test(chartTitle)) {
                    // Put bars may be drawn mirrored; the bar text holds the actual signed value
                    value = String(point.text);
                } else if (typeof rawValue === 'number') {
                    value = /premium/i.test(chartTitle) ? formatTooltipMoney(rawValue) : formatTooltipNumber(rawValue);
                } else if (point?.text != null && String(point.text).trim() !== '') {
                    value = String(point.text);
                } else {
                    value = String(rawValue == null ? '' : rawValue);
                }
            }

            return '<div class="tt-row">'
                + `<span class="tt-dot" style="background:${escapeTooltipHtml(resolvePlotlyPointColor(point))}"></span>`
                + '<div class="tt-main">'
                + `<span class="tt-name">${escapeTooltipHtml(name)}</span>`
                + `<span class="tt-value">${escapeTooltipHtml(value)}</span>`
                + '</div>'
                + '</div>';
        }

        function ensurePlotlyTooltip(plotDiv) {
            if (!plotDiv) return null;
            let tooltip = plotDiv.querySelector('.chart-hover-tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.className = 'chart-hover-tooltip';
                plotDiv.appendChild(tooltip);
            }
            return tooltip;
        }

        function hidePlotlyTooltip(plotDiv) {
            const tooltip = plotDiv?.querySelector('.chart-hover-tooltip');
            if (tooltip) {
                tooltip.style.display = 'none';
            }
        }

        function positionPlotlyTooltip(plotDiv, tooltip, event) {
            if (!plotDiv || !tooltip || !event) return;
            const bounds = plotDiv.getBoundingClientRect();
            const left = Math.min(
                Math.max(8, event.clientX - bounds.left + 12),
                Math.max(8, bounds.width - tooltip.offsetWidth - 8)
            );
            const top = Math.min(
                Math.max(8, event.clientY - bounds.top + 12),
                Math.max(8, bounds.height - tooltip.offsetHeight - 8)
            );
            tooltip.style.left = `${left}px`;
            tooltip.style.top = `${top}px`;
        }

        function attachPlotlyCustomTooltip(plotDiv) {
            if (!plotDiv || plotDiv.__customTooltipBound || typeof plotDiv.on !== 'function') {
                return;
            }

            plotDiv.__customTooltipBound = true;
            plotDiv.on('plotly_hover', eventData => {
                const tooltip = ensurePlotlyTooltip(plotDiv);
                const hoverPoints = Array.isArray(eventData?.points) ? eventData.points : [];
                if (!tooltip || !hoverPoints.length) {
                    hidePlotlyTooltip(plotDiv);
                    return;
                }

                const topPoints = hoverPoints.slice(0, 5);
                tooltip.innerHTML = '<div class="tt-head"><div class="tt-time">'
                    + buildPlotlyTooltipContext(hoverPoints, plotDiv) + '</div></div><div class="tt-list">'
                    + topPoints.map(point => buildPlotlyTooltipRow(point, plotDiv)).join('') + '</div>'
                    + (hoverPoints.length > topPoints.length
                        ? `<div class="tt-more">+${hoverPoints.length - topPoints.length} more</div>`
                        : '');

                tooltip.style.display = 'block';
                positionPlotlyTooltip(plotDiv, tooltip, eventData?.event);
            });

            // plotDiv.on('plotly_unhover', () => hidePlotlyTooltip(plotDiv));
            // plotDiv.on('plotly_relayout', () => hidePlotlyTooltip(plotDiv));
            plotDiv.addEventListener('mouseleave', () => hidePlotlyTooltip(plotDiv));
        }

        function mountHeatmapControls(toolbar) {
            const staging = document.getElementById('chart-control-staging');
            const target = toolbar || staging;
            if (!target) {
                return;
            }

            ['heatmap-type-control', 'heatmap-coloring-control'].forEach(controlId => {
                const control = document.getElementById(controlId);
                if (control && control.parentElement !== target) {
                    target.appendChild(control);
                }
            });
        }

        function ensureHeatmapChartShell(container) {
            if (!container) {
                mountHeatmapControls(null);
                return null;
            }

            let shell = container.querySelector('.heatmap-chart-shell');
            if (!shell) {
                shell = document.createElement('div');
                shell.className = 'heatmap-chart-shell';

                const toolbar = document.createElement('div');
                toolbar.className = 'heatmap-chart-toolbar';

                const toolbarControls = document.createElement('div');
                toolbarControls.className = 'heatmap-chart-toolbar-controls';

                const plot = document.createElement('div');
                plot.className = 'heatmap-plot';
                plot.id = 'heatmap-plot';

                toolbar.appendChild(toolbarControls);
                shell.appendChild(toolbar);
                shell.appendChild(plot);
                container.appendChild(shell);
            }

            mountHeatmapControls(shell.querySelector('.heatmap-chart-toolbar-controls'));
            return shell;
        }

        function getPlotlyChartElement(chartKey) {
            const container = document.getElementById(`${chartKey}-chart`);
            if (!container) {
                return null;
            }
            if (chartKey === 'heatmap') {
                const shell = ensureHeatmapChartShell(container);
                return shell ? shell.querySelector('.heatmap-plot') : null;
            }
            return container;
        }

        function renderPlotlyChart(key, rawChartData) {
            const containerId = `${key}-chart`;
            const container = document.getElementById(containerId);
            if (!container || !rawChartData) {
                return false;
            }

            const plotElement = getPlotlyChartElement(key);
            if (!plotElement) {
                return false;
            }

            const alreadyPlotted = !!(charts[key] && plotElement._fullLayout);
            if (alreadyPlotted) {
                // Unchanged payload: the chart on screen is already current
                if (typeof rawChartData === 'string' && plotElement.__lastRawChart === rawChartData) {
                    delete container.__pendingChartData;
                    return true;
                }
                // Off-screen or background tab: keep the payload and draw it once visible
                if (!isPlotlyContainerVisible(container)) {
                    container.__pendingChartData = rawChartData;
                    observePlotlyVisibility(key, container);
                    return true;
                }
            }
            delete container.__pendingChartData;

            const chartData = typeof rawChartData === 'string' ? JSON.parse(rawChartData) : rawChartData;
            const baseMargins = chartData.layout.margin || {l: 44, r: 28, t: 44, b: 28};

            chartData.layout.autosize = true;
            chartData.layout.width = null;
            chartData.layout.height = null;
            chartData.layout.margin = getChartMargins(containerId, key === 'heatmap' ? baseMargins : PLOTLY_BASE_MARGINS);
            // Draw at the size the post-render resize would pick, so it has nothing to redo
            if (alreadyPlotted) {
                const target = key === 'heatmap'
                    ? getHeatmapTargetLayout(plotElement)
                    : getPlotlyTargetLayout(container, plotElement);
                if (target) {
                    chartData.layout.width = target.width;
                    chartData.layout.height = target.height;
                    if (target.autosize !== undefined) chartData.layout.autosize = target.autosize;
                    if (target.margin) chartData.layout.margin = target.margin;
                }
            }

            if (key !== 'heatmap') {
                if (chartData.layout.xaxis) {
                    chartData.layout.xaxis.autorange = true;
                }
                if (chartData.layout.yaxis) {
                    chartData.layout.yaxis.autorange = true;
                }
            }
            applyThemeToPlotlyLayout(chartData.layout);

            // Pre-compute heatmap annotation font size so the first render is correct
            // and no follow-up relayout (which causes a visible size-change flash) is needed.
            if (key === 'heatmap' && chartData.layout && Array.isArray(chartData.layout.annotations)) {
                const trace = chartData.data && chartData.data[0];
                const ncols = (trace && trace.x) ? trace.x.length : 1;
                const nrows = (trace && trace.y) ? trace.y.length : 1;
                const margin = chartData.layout.margin || {l: 44, r: 52, t: 48, b: 32};
                const containerW = plotElement.clientWidth  || container.clientWidth  || 400;
                const containerH = plotElement.clientHeight || container.clientHeight || 300;
                const plotW = Math.max(1, containerW - (margin.l || 0) - (margin.r || 0));
                const plotH = Math.max(1, containerH - (margin.t || 0) - (margin.b || 0));
                const cellW = plotW / Math.max(1, ncols);
                const cellH = plotH / Math.max(1, nrows);
                const preSize = Math.max(6, Math.min(20, Math.floor(Math.min(cellW, cellH) * 0.38)));
                chartData.layout.annotations.forEach(a => {
                    if (a.xref === 'x' && a.yref === 'y' && !a.showarrow) {
                        a.font = Object.assign({}, a.font, { size: preSize });
                    }
                });
            }

            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                displaylogo: false,
                scrollZoom: true
            };

            let plotPromise;
            if (charts[key]) {
                plotPromise = Plotly.react(plotElement, chartData.data, chartData.layout, config);
            } else {
                plotPromise = Plotly.newPlot(plotElement, chartData.data, chartData.layout, config);
                charts[key] = plotPromise;
            }
            plotElement.__lastRawChart = typeof rawChartData === 'string' ? rawChartData : null;
            plotElement.__livePriceStr = null;

            Promise.resolve(plotPromise).then(() => {
                attachPlotlyCustomTooltip(plotElement);
                if (key === 'heatmap') {
                    scheduleHeatmapPlotResize(containerId);
                    attachHeatmapResizeObserver(containerId);
                } else {
                    schedulePlotlyChartResize(containerId);
                    attachPlotlyResizeObserver(containerId);
                }
            });

            addFullscreenButton(container);
            addPopoutButton(container);
            return true;
        }

        // --- Deferred rendering for charts outside the viewport ---
        // Redrawing every chart on each /update is the main source of lag, so charts that
        // are scrolled out of view (or in a background tab) keep their latest payload in
        // container.__pendingChartData and are drawn when they become visible again.
        const _plotlyVisibleContainers = new WeakMap();
        let _plotlyVisibilityObserver = null;

        function isPlotlyContainerVisible(container) {
            if (document.hidden) return false;
            if (container.classList.contains('fullscreen')) return true;
            // Unknown until the IntersectionObserver reports; assume visible
            return _plotlyVisibleContainers.get(container) !== false;
        }

        function flushPendingPlotlyChart(container) {
            if (!container || !container.isConnected) return;
            const key = container.id.replace(/-chart$/, '');
            if (container.__pendingChartData !== undefined) {
                if (!charts[key]) { delete container.__pendingChartData; return; }
                try { renderPlotlyChart(key, container.__pendingChartData); } catch (e) {
                    console.error(`Error rendering deferred ${key} chart:`, e);
                }
            }
            // Price-line updates skip hidden charts, so bring this one up to the live price
            if (typeof livePrice === 'number') applyPlotlyPriceLine(container, livePrice);
        }

        // Stop observing a chart container that is being removed from the page
        function forgetPlotlyContainer(container) {
            if (!container) return;
            if (_plotlyVisibilityObserver) _plotlyVisibilityObserver.unobserve(container);
            container.__visibilityObserved = false;
            const entry = _plotlyResizeObservers.get(container.id);
            if (entry && entry.container === container) {
                entry.observer.disconnect();
                clearTimeout(entry.timer);
                _plotlyResizeObservers.delete(container.id);
            }
        }

        function observePlotlyVisibility(key, container) {
            if (typeof IntersectionObserver === 'undefined' || container.__visibilityObserved) return;
            if (!_plotlyVisibilityObserver) {
                _plotlyVisibilityObserver = new IntersectionObserver(entries => {
                    entries.forEach(entry => {
                        _plotlyVisibleContainers.set(entry.target, entry.isIntersecting);
                        if (entry.isIntersecting) flushPendingPlotlyChart(entry.target);
                    });
                }, { rootMargin: '200px 0px' });
            }
            container.__visibilityObserved = true;
            _plotlyVisibilityObserver.observe(container);
        }

        document.addEventListener('visibilitychange', () => {
            if (document.hidden) return;
            document.querySelectorAll('.chart-container').forEach(container => {
                if (isPlotlyContainerVisible(container)) flushPendingPlotlyChart(container);
            });
        });

        let _heatmapResizeObserver = null;
        let _heatmapResizeObserved = null;
        const _plotlyResizeObservers = new Map();

        function attachPlotlyResizeObserver(containerId) {
            if (containerId === 'heatmap-chart') {
                attachHeatmapResizeObserver(containerId);
                return;
            }
            const container = document.getElementById(containerId);
            if (!container) return;
            observePlotlyVisibility(containerId.replace(/-chart$/, ''), container);
            if (typeof ResizeObserver === 'undefined') return;

            const existing = _plotlyResizeObservers.get(containerId);
            // Re-creating the observer on every render makes it fire (and resize) each time
            if (existing && existing.container === container) return;
            if (existing) {
                existing.observer.disconnect();
                clearTimeout(existing.timer);
            }

            const entry = { observer: null, timer: null, container };
            entry.observer = new ResizeObserver(() => {
                clearTimeout(entry.timer);
                entry.timer = setTimeout(() => resizePlotlyChart(container), 120);
            });
            entry.observer.observe(container);
            _plotlyResizeObservers.set(containerId, entry);
        }

        function getHeatmapTargetLayout(plotEl) {
            if (!plotEl) return null;
            const width = plotEl.clientWidth;
            const height = plotEl.clientHeight;
            if (width < 2 || height < 2) return null;
            return { width, height };
        }

        function resizeHeatmapPlot() {
            const plotEl = getPlotlyChartElement('heatmap');
            const target = getHeatmapTargetLayout(plotEl);
            if (!target || plotlyLayoutMatches(plotEl, target)) return;
            try {
                // Explicit width/height makes Plots.resize a no-op, so relayout alone suffices
                Plotly.relayout(plotEl, target);
            } catch (e) {}
        }

        function scheduleHeatmapPlotResize(containerId) {
            requestAnimationFrame(() => {
                resizeHeatmapPlot();
                requestAnimationFrame(() => {
                    resizeHeatmapPlot();
                    updateHeatmapTextSize(containerId);
                });
            });
        }

        function attachHeatmapResizeObserver(containerId) {
            const plotEl = getPlotlyChartElement('heatmap');
            const containerEl = document.getElementById(containerId);
            if (containerEl) observePlotlyVisibility('heatmap', containerEl);
            const observeEl = plotEl || containerEl;
            // Re-creating the observer on every render makes it fire (and resize) each time
            if (_heatmapResizeObserver && _heatmapResizeObserved === observeEl) return;
            if (_heatmapResizeObserver) {
                _heatmapResizeObserver.disconnect();
                _heatmapResizeObserver = null;
            }
            _heatmapResizeObserved = observeEl;
            if (!observeEl || typeof ResizeObserver === 'undefined') return;
            let _heatmapResizeTimer = null;
            _heatmapResizeObserver = new ResizeObserver(() => {
                clearTimeout(_heatmapResizeTimer);
                _heatmapResizeTimer = setTimeout(() => {
                    resizeHeatmapPlot();
                    updateHeatmapTextSize(containerId);
                }, 120);
            });
            _heatmapResizeObserver.observe(observeEl);
            if (containerEl && containerEl !== observeEl) {
                _heatmapResizeObserver.observe(containerEl);
            }
        }

        function updateHeatmapTextSize(containerId) {
            const div = getPlotlyChartElement(containerId.replace('-chart', ''));
            if (!div || !div._fullLayout) return;
            const fl = div._fullLayout;
            if (!fl.annotations || fl.annotations.length === 0) return;
            // Count heatmap cell annotations (xref='x', yref='y', no arrow)
            const cellAnnotIdxs = [];
            fl.annotations.forEach((a, i) => {
                if (a.xref === 'x' && a.yref === 'y' && !a.showarrow) cellAnnotIdxs.push(i);
            });
            if (cellAnnotIdxs.length === 0) return;
            // Determine grid dims from heatmap trace
            const trace = div._fullData && div._fullData[0];
            const ncols = (trace && trace.x) ? trace.x.length : 1;
            const nrows = (trace && trace.y) ? trace.y.length : 1;
            const plotW = (fl.width  || div.offsetWidth)  - (fl.margin.l + fl.margin.r);
            const plotH = (fl.height || div.offsetHeight) - (fl.margin.t + fl.margin.b);
            const cellW = plotW / Math.max(1, ncols);
            const cellH = plotH / Math.max(1, nrows);
            const newSize = Math.max(6, Math.min(20, Math.floor(Math.min(cellW, cellH) * 0.38)));
            if (fl.annotations[cellAnnotIdxs[0]].font && fl.annotations[cellAnnotIdxs[0]].font.size === newSize) return;
            const relayoutUpdate = { 'transition.duration': 0 };
            cellAnnotIdxs.forEach(i => { relayoutUpdate[`annotations[${i}].font.size`] = newSize; });
            try { Plotly.relayout(div, relayoutUpdate); } catch(e) {}
        }

        function updateHeatmapOnly() {
            let fallbackToFullUpdate = false;

            if (!document.getElementById('heatmap').checked) {
                return;
            }

            if (updateInProgress) {
                pendingHeatmapOnlyUpdate = true;
                return;
            }

            const expiry = getSelectedExpiryValues();
            if (expiry.length === 0) {
                return;
            }

            updateInProgress = true;
            pendingHeatmapOnlyUpdate = false;

            fetch('/update_heatmap', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ticker: document.getElementById('ticker').value,
                    expiry,
                    show_calls: document.getElementById('show_calls').checked,
                    show_puts: document.getElementById('show_puts').checked,
                    show_net: document.getElementById('show_net').checked,
                    strike_range: parseFloat(document.getElementById('strike_range').value) / 100,
                    call_color: callColor,
                    put_color: putColor,
                    heatmap_type: document.getElementById('heatmap_type').value,
                    heatmap_coloring_mode: document.getElementById('heatmap_coloring_mode').value,
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error || !data.heatmap) {
                    console.warn('Heatmap-only update fell back to full update:', data.error || 'missing heatmap payload');
                    fallbackToFullUpdate = true;
                    return;
                }

                lastData = {
                    ...lastData,
                    heatmap: data.heatmap,
                    selected_expiries: data.selected_expiries || lastData.selected_expiries,
                };

                if (!renderPlotlyChart('heatmap', data.heatmap)) {
                    updateCharts(lastData);
                } else {
                    pushAllPopouts();
                }
            })
            .catch(error => {
                console.error('Heatmap-only update failed:', error);
                fallbackToFullUpdate = true;
            })
            .finally(() => {
                updateInProgress = false;
                if (fallbackToFullUpdate) {
                    updateData();
                    return;
                }
                if (pendingHeatmapOnlyUpdate) {
                    pendingHeatmapOnlyUpdate = false;
                    updateHeatmapOnly();
                }
            });
        }

        // Levels dropdown handlers
        function updateLevelsDisplay() {
            const checkedBoxes = document.querySelectorAll('.levels-option input[type="checkbox"]:checked');
            const levelsText = document.getElementById('levels-text');
            
            if (checkedBoxes.length === 0) {
                levelsText.textContent = 'None';
            } else if (checkedBoxes.length === 1) {
                levelsText.textContent = checkedBoxes[0].value;
            } else {
                levelsText.textContent = `${checkedBoxes.length} selected`;
            }
        }

        function refreshPriceLevelsOnly() {
            const rawLevels = notePriceLevelSelectionChanged();
            updateLevelsDisplay();
            syncMobilePanelButtons();
            const hasLastData = !!(lastData && Object.keys(lastData).length);
            if (!rawLevels.length && isPriceLevelEmptyFallbackActive()) {
                schedulePriceLevelEmptyRefresh();
                return;
            }
            if (!rawLevels.length && tvPriceChart) {
                tvHistoricalPoints = [];
                clearTVLatestLevelNativeLabels();
                scheduleTVHistoricalOverlayDraw();
            }
            if (document.getElementById('price').checked) {
                _priceHistoryLastKey = '';
                fetchPriceHistory(true);
            }
            if (!hasLastData && !updateInProgress) {
                updateData();
            }
        }

        document.getElementById('levels-display').addEventListener('click', function(e) {
            e.stopPropagation();
            const options = document.getElementById('levels-options');
            options.classList.toggle('open');
        });
        
        // Add event listeners for level checkboxes
        document.querySelectorAll('.levels-option input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                refreshPriceLevelsOnly();
            });
        });
        
        function updateData() {
            if (updateInProgress || expirationsLoading) {
                pendingFullUpdate = true;
                return; // Skip if an update is already in progress
            }

            // Don't send a half-typed / uncommitted ticker with the previous ticker's
            // expiries; loadExpirations() triggers the update once the ticker is committed.
            const typedTicker = (document.getElementById('ticker').value || '').toUpperCase();
            if (expirationsTicker !== null && typedTicker !== expirationsTicker) {
                // If loading the new ticker's expiries failed, retry instead of stalling
                // (not while the ticker box is still being typed in)
                const tickerInputEl = document.getElementById('ticker');
                if (!expirationsLoading && typedTicker && document.activeElement !== tickerInputEl
                        && Date.now() - lastExpirationsAttemptMs >= EXPIRATIONS_RETRY_MS) {
                    loadExpirations();
                }
                return;
            }

            // A new ET day (page left open overnight) or a selected expiry that has passed:
            // reload the list, which rolls expired selections forward to the nearest expiry.
            if (expirationsTicker !== null && !expirationsLoading) {
                const todayEt = etTodayStr();
                const hasExpiredSelection = getSelectedExpiryValues().some(d => d < todayEt);
                if ((hasExpiredSelection || expirationsLoadedEtDate !== todayEt)
                        && Date.now() - lastExpirationsAttemptMs >= EXPIRATIONS_ROLLOVER_RETRY_MS) {
                    loadExpirations();
                    return;
                }
                // Don't request chains that no longer exist while the reload is pending
                if (hasExpiredSelection) return;
            }

            updateInProgress = true;

            const ticker = document.getElementById('ticker').value;
            const requestTicker = ticker.toUpperCase();
            const tickerChanged = tvLastTicker !== null && ticker.toUpperCase() !== tvLastTicker.toUpperCase();

            // Reset chart state when the ticker changes
            if (tickerChanged) {
                // Clear drawings
                tvClearDrawings();
                tvDrawingDefs = [];
                // Reset zoom on the next render
                tvLastCandles = [];
                tvIndicatorCandles = [];
                tvCurrentDayStartTime = 0;
                tvForceFit = true;
                tvStreamBucketState = null;
                // Drop any in-flight price response for the old ticker
                _priceHistoryDesiredKey = '';
                // Disconnect the price stream so it reconnects on the new ticker
                disconnectPriceStream();
            }
            tvLastTicker = ticker;

            const expiry = getSelectedExpiryValues();
            const requestExpiryKey = expiry.slice().sort().join('|');
            
            // Ensure at least one expiry is selected
            if (expiry.length === 0) {
                updateInProgress = false;
                // A failed/empty /expirations call (e.g. a transient Schwab error at startup)
                // used to leave the list empty forever; keep retrying until it loads.
                const hasExpiryOptions = document.querySelectorAll('.expiry-option input[type="checkbox"]').length > 0;
                // (With a list but nothing selected, e.g. a hand-picked expiry that expired,
                // just wait for a new pick; the dropdown says why it's empty.)
                if (!hasExpiryOptions && !expirationsLoading && Date.now() - lastExpirationsAttemptMs >= EXPIRATIONS_RETRY_MS) {
                    loadExpirations();
                }
                // /update_price works without expiries (candles only, no exposure levels)
                if (document.getElementById('price').checked) {
                    fetchPriceHistory(tickerChanged || !tvLastCandles.length);
                }
                return;
            }
            const showCalls = document.getElementById('show_calls').checked;
            const showPuts = document.getElementById('show_puts').checked;
            const showNet = document.getElementById('show_net').checked;
            const coloringMode = document.getElementById('coloring_mode').value;
            const levelsTypes = getSelectedPriceLevelTypes();
            const horizontalBars = document.getElementById('horizontal_bars').checked;
            const showAbsGex = document.getElementById('show_abs_gex').checked;
            const absGexOpacity = parseInt(document.getElementById('abs_gex_opacity').value) / 100;
            const useRange = document.getElementById('use_range').checked;
            const exposureMetric = document.getElementById('exposure_metric').value;
            const heatmapType = document.getElementById('heatmap_type').value;
            const heatmapColoringMode = document.getElementById('heatmap_coloring_mode').value;
            const deltaAdjusted = document.getElementById('delta_adjusted_exposures').checked;
            const calculateInNotional = document.getElementById('calculate_in_notional').checked;
            const strikeRange = parseFloat(document.getElementById('strike_range').value) / 100;
            const highlightMaxLevel = document.getElementById('highlight_max_level').checked;
            const maxLevelMode = document.getElementById('max_level_mode').value;

            // Get visible charts
            const visibleCharts = {
                show_price: document.getElementById('price').checked,
                show_gamma: document.getElementById('gamma').checked,
                show_heatmap: document.getElementById('heatmap').checked,
                show_delta: document.getElementById('delta').checked,
                show_vanna: document.getElementById('vanna').checked,
                show_charm: document.getElementById('charm').checked,
                show_speed: document.getElementById('speed').checked,
                show_vomma: document.getElementById('vomma').checked,
                show_color: document.getElementById('color').checked,
                show_options_volume: document.getElementById('options_volume').checked,
                show_open_interest: document.getElementById('open_interest').checked,
                show_volume: document.getElementById('volume').checked,
                show_large_trades: document.getElementById('large_trades').checked,
                show_centroid: document.getElementById('centroid').checked
            };

            // Historical bubble overlays depend on /update populating the interval
            // store and options cache first. If level overlays are enabled, defer the
            // price-chart fetch until the /update request completes to avoid rendering
            // a bubble-less chart that only corrects after a full reload.
            const deferPriceHistoryUntilUpdate = levelsTypes.length > 0;

            // Fetch price history immediately only when no historical overlays are requested.
            // Real-time candle ticks come from SSE (connectPriceStream), not from polling.
            if (visibleCharts.show_price && !deferPriceHistoryUntilUpdate) {
                fetchPriceHistory(tickerChanged || !tvLastCandles.length);
            }

            function normalizeFetchError(error) {
                if (!error) {
                    return 'Unknown update error.';
                }
                if (typeof error === 'string') {
                    return error;
                }
                if (error.name === 'AbortError') {
                    return 'Request was cancelled.';
                }
                if (error.name === 'SyntaxError') {
                    return 'The server returned an unreadable response.';
                }
                return error.message || String(error);
            }
            
            fetch('/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    ticker, 
                    expiry,
                    show_calls: showCalls,
                    show_puts: showPuts,
                    show_net: showNet,
                    coloring_mode: coloringMode,
                    horizontal_bars: horizontalBars,
                    show_abs_gex: showAbsGex,
                    abs_gex_opacity: absGexOpacity,
                    use_range: useRange,
                    exposure_metric: exposureMetric,
                    heatmap_type: heatmapType,
                    heatmap_coloring_mode: heatmapColoringMode,
                    delta_adjusted: deltaAdjusted,
                    calculate_in_notional: calculateInNotional,
                    strike_range: strikeRange,
                    call_color: callColor,
                    put_color: putColor,
                    highlight_max_level: highlightMaxLevel,
                    max_level_color: maxLevelColor,
                    max_level_mode: maxLevelMode,
                    ...visibleCharts
                })
            })
            .then(async response => {
                const payload = await response.json().catch(() => {
                    throw new SyntaxError(`Update request returned non-JSON (HTTP ${response.status})`);
                });

                if (!response.ok) {
                    throw new Error(payload?.error || `Update request failed with HTTP ${response.status}`);
                }

                return payload;
            })
            .then(data => {
                const activeTicker = (document.getElementById('ticker').value || '').toUpperCase();
                const activeExpiryKey = getSelectedExpiryValues().slice().sort().join('|');
                if (activeTicker !== requestTicker || activeExpiryKey !== requestExpiryKey) {
                    return;
                }
                // The server dropped expired dates from this selection; reload the list so
                // the selection rule (or the empty hand-picked selection) is applied to it
                if (data.expired_expiries && data.expired_expiries.length && !expirationsLoading) {
                    loadExpirations();
                }
                if (data.error) {
                    showError(data.error);
                    noteUpdateFailure();
                    return;
                }
                consecutiveUpdateFailures = 0;
                updateBackoffUntilMs = 0;
                
                // Only update if data has changed
                if (JSON.stringify(data) !== JSON.stringify(lastData)) {
                    lastData = data;  // Update before rendering so popout windows get fresh data
                    try {
                        updateCharts(data);
                        updatePriceInfo(data.price_info);
                    } catch (renderError) {
                        console.error('Error rendering update payload:', renderError);
                        showError('Render Error: ' + normalizeFetchError(renderError));
                        return;
                    }
                }
            })
            .catch(error => {
                // Ignore failures of requests for a ticker/expiry that is no longer selected
                const activeTicker = (document.getElementById('ticker').value || '').toUpperCase();
                const activeExpiryKey = getSelectedExpiryValues().slice().sort().join('|');
                if (activeTicker !== requestTicker || activeExpiryKey !== requestExpiryKey) {
                    return;
                }
                const normalizedMessage = normalizeFetchError(error);
                const userMessage = /network|failed to fetch/i.test(normalizedMessage)
                    ? 'Network Error: Could not connect to the server.'
                    : normalizedMessage;
                showError(userMessage);
                noteUpdateFailure();
                console.error('Error fetching data:', error);
            })
            .finally(() => {
                updateInProgress = false;
                // The options cache is populated now, so the price chart can pick up levels.
                // Only force a Schwab price-history refetch when the chart needs a fresh
                // load; otherwise let fetchPriceHistory's throttle apply (it still refetches
                // immediately whenever the payload changes). Runs on errors too so a failed
                // /update doesn't leave the deferred price chart empty.
                if (visibleCharts.show_price) {
                    fetchPriceHistory(deferPriceHistoryUntilUpdate && (tickerChanged || !tvLastCandles.length));
                }
                if (pendingFullUpdate) {
                    pendingFullUpdate = false;
                    updateData();
                    return;
                }
                if (pendingHeatmapOnlyUpdate) {
                    pendingHeatmapOnlyUpdate = false;
                    updateHeatmapOnly();
                }
            });
        }

        // ── TradingView Lightweight Charts price chart renderer ───────────────

        // ── Indicator math helpers ────────────────────────────────────────────
        function calcSMA(closes, period) {
            return closes.map((_, i) => {
                if (i < period - 1) return null;
                const slice = closes.slice(i - period + 1, i + 1);
                return slice.reduce((a, b) => a + b, 0) / period;
            });
        }
        function calcEMA(closes, period) {
            const k = 2 / (period + 1);
            const result = [];
            let ema = null;
            for (let i = 0; i < closes.length; i++) {
                if (i < period - 1) { result.push(null); continue; }
                if (ema === null) { ema = closes.slice(0, period).reduce((a,b)=>a+b,0)/period; }
                else              { ema = closes[i] * k + ema * (1 - k); }
                result.push(ema);
            }
            return result;
        }
        function calcWMA(closes, period) {
            const result = [];
            const denominator = period * (period + 1) / 2;
            for (let i = 0; i < closes.length; i++) {
                if (i < period - 1) { result.push(null); continue; }
                let weightedSum = 0;
                for (let weight = 1; weight <= period; weight++) {
                    weightedSum += closes[i - period + weight] * weight;
                }
                result.push(weightedSum / denominator);
            }
            return result;
        }
        function calcVWAP(candles) {
            let cumPV = 0, cumVol = 0;
            return candles.map(c => {
                const typical = (c.high + c.low + c.close) / 3;
                cumPV  += typical * c.volume;
                cumVol += c.volume;
                return cumVol > 0 ? cumPV / cumVol : c.close;
            });
        }
        function calcBB(closes, period=20, mult=2) {
            const sma = calcSMA(closes, period);
            return sma.map((mid, i) => {
                if (mid === null) return { upper: null, mid: null, lower: null };
                const slice = closes.slice(Math.max(0, i - period + 1), i + 1);
                const variance = slice.reduce((a, b) => a + (b - mid) ** 2, 0) / slice.length;
                const sd = Math.sqrt(variance);
                return { upper: mid + mult * sd, mid, lower: mid - mult * sd };
            });
        }
        function calcRollingStdev(values, period) {
            return values.map((_, i) => {
                if (i < period - 1) return null;
                const slice = values.slice(i - period + 1, i + 1);
                const mean = slice.reduce((a, b) => a + b, 0) / period;
                const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / period;
                return Math.sqrt(variance);
            });
        }
        function calcVWMA(values, volumes, period) {
            return values.map((_, i) => {
                if (i < period - 1) return null;
                let sumPV = 0, sumVol = 0;
                const slice = values.slice(i - period + 1, i + 1);
                for (let j = i - period + 1; j <= i; j++) {
                    const vol = volumes[j] || 0;
                    sumPV += values[j] * vol;
                    sumVol += vol;
                }
                if (sumVol > 0) return sumPV / sumVol;
                return slice.reduce((a, b) => a + b, 0) / period;
            });
        }
        function calcFBB(candles, length=200, mult=3.0) {
            const src = candles.map(c => (c.high + c.low + c.close) / 3);
            const volumes = candles.map(c => c.volume || 0);
            const basis = calcVWMA(src, volumes, length);
            const stdev = calcRollingStdev(src, length);
            return candles.map((_, i) => {
                if (basis[i] === null || stdev[i] === null) {
                    return { basis: null, upper: null, lower: null };
                }
                const dev = mult * stdev[i];
                return { basis: basis[i], upper: basis[i] + dev, lower: basis[i] - dev };
            });
        }
        function calcRSI(closes, period=14) {
            const result = [];
            for (let i = 0; i < closes.length; i++) {
                if (i < period) { result.push(null); continue; }
                let gains = 0, losses = 0;
                for (let j = i - period + 1; j <= i; j++) {
                    const diff = closes[j] - closes[j-1];
                    if (diff > 0) gains  += diff;
                    else          losses -= diff;
                }
                const avgGain = gains  / period;
                const avgLoss = losses / period;
                result.push(avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss));
            }
            return result;
        }
        function calcMACD(closes, fast=12, slow=26, signal=9) {
            const emaFast   = calcEMA(closes, fast);
            const emaSlow   = calcEMA(closes, slow);
            const macdLine  = emaFast.map((v, i) => (v !== null && emaSlow[i] !== null) ? v - emaSlow[i] : null);
            const sigLine   = [];
            let emaS = null;
            let validIdx = 0;
            const k = 2 / (signal + 1);
            for (let i = 0; i < macdLine.length; i++) {
                if (macdLine[i] === null) { sigLine.push(null); continue; }
                if (validIdx < signal - 1) { sigLine.push(null); validIdx++; continue; }
                if (emaS === null) {
                    const slice = macdLine.filter(v=>v!==null).slice(0, signal);
                    emaS = slice.reduce((a,b)=>a+b,0)/signal;
                } else {
                    emaS = macdLine[i] * k + emaS * (1 - k);
                }
                sigLine.push(emaS);
                validIdx++;
            }
            return { macd: macdLine, signal: sigLine,
                     histogram: macdLine.map((v,i) => (v!==null && sigLine[i]!==null) ? v-sigLine[i] : null) };
        }

        function calcATR(candles, period=14) {
            const result = [];
            for (let i = 0; i < candles.length; i++) {
                const tr = i === 0
                    ? candles[i].high - candles[i].low
                    : Math.max(
                        candles[i].high - candles[i].low,
                        Math.abs(candles[i].high - candles[i-1].close),
                        Math.abs(candles[i].low  - candles[i-1].close)
                      );
                if (i < period - 1) { result.push(null); continue; }
                if (result.length === 0 || result[result.length-1] === null) {
                    let sum = 0;
                    for (let j = i - period + 1; j <= i; j++) {
                        const t = j === 0
                            ? candles[j].high - candles[j].low
                            : Math.max(
                                candles[j].high - candles[j].low,
                                Math.abs(candles[j].high - candles[j-1].close),
                                Math.abs(candles[j].low  - candles[j-1].close)
                              );
                        sum += t;
                    }
                    result.push(sum / period);
                } else {
                    result.push((result[result.length-1] * (period - 1) + tr) / period);
                }
            }
            return result;
        }

        function calcARV(candles, period=20) {
            const tradingSecondsPerYear = 252 * 6.5 * 60 * 60;
            if (!Array.isArray(candles) || candles.length === 0) return [];

            const barIntervals = [];
            for (let i = 1; i < candles.length; i++) {
                const deltaSeconds = Number(candles[i].time) - Number(candles[i - 1].time);
                if (Number.isFinite(deltaSeconds) && deltaSeconds > 0) {
                    barIntervals.push(deltaSeconds);
                }
            }

            const sortedIntervals = barIntervals.slice().sort((a, b) => a - b);
            const medianInterval = sortedIntervals.length
                ? (sortedIntervals.length % 2 === 0
                    ? (sortedIntervals[(sortedIntervals.length / 2) - 1] + sortedIntervals[sortedIntervals.length / 2]) / 2
                    : sortedIntervals[Math.floor(sortedIntervals.length / 2)])
                : 60;
            const annualizationFactor = Math.sqrt(tradingSecondsPerYear / Math.max(medianInterval, 1));

            const logReturns = candles.map((candle, index) => {
                if (index === 0) return null;
                const prevClose = Number(candles[index - 1].close);
                const close = Number(candle.close);
                if (!(prevClose > 0) || !(close > 0)) return null;
                return Math.log(close / prevClose);
            });

            return candles.map((_, index) => {
                if (index < period) return null;
                const window = logReturns
                    .slice(index - period + 1, index + 1)
                    .filter(value => Number.isFinite(value));
                if (window.length < period) return null;

                const mean = window.reduce((sum, value) => sum + value, 0) / window.length;
                const variance = window.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / Math.max(window.length - 1, 1);
                return Math.sqrt(Math.max(variance, 0)) * annualizationFactor * 100;
            });
        }

        function calcAutoTrendLine(candles, dayStart=0) {
            const dayCandles = dayStart > 0 ? candles.filter(c => c.time >= dayStart) : candles;
            const points = dayCandles
                .map(c => ({ time: c.time, close: Number(c.close) }))
                .filter(point => Number.isFinite(point.close));

            if (points.length < 2) return [];

            let sumX = 0;
            let sumY = 0;
            let sumXY = 0;
            let sumXX = 0;

            points.forEach((point, index) => {
                sumX += index;
                sumY += point.close;
                sumXY += index * point.close;
                sumXX += index * index;
            });

            const n = points.length;
            const denominator = (n * sumXX) - (sumX * sumX);
            const slope = denominator === 0 ? 0 : ((n * sumXY) - (sumX * sumY)) / denominator;
            const intercept = (sumY - (slope * sumX)) / n;

            return points.map((point, index) => ({
                time: point.time,
                value: intercept + (slope * index),
            }));
        }

        // ── Smart Money Concepts calculation ──────────────────────────────────
        function calcSMC(candles, params) {
            const swingLen = Math.max(2, Math.min(20, parseInt(params.swing_len) || 5));
            const n = candles.length;
            const EQ_THRESH = 0.0015; // 0.15% tolerance for equal highs/lows

            // ── 1. Pivot detection ──────────────────────────────────────────
            const swingHighs = [], swingLows = [];
            for (let i = swingLen; i < n - swingLen; i++) {
                const h = candles[i].high, l = candles[i].low;
                let isH = true, isL = true;
                for (let j = i - swingLen; j <= i + swingLen; j++) {
                    if (j === i) continue;
                    if (candles[j].high >= h) isH = false;
                    if (candles[j].low  <= l) isL = false;
                    if (!isH && !isL) break;
                }
                if (isH) swingHighs.push({ index: i, time: candles[i].time, price: h });
                if (isL) swingLows.push({ index: i, time: candles[i].time, price: l });
            }

            // ── 2. HH/LH and HL/LL classification → Strong/Weak ───────────
            // Strong High = Higher High (HH), Weak High = Lower High (LH)
            // Strong Low  = Higher Low  (HL), Weak Low  = Lower Low  (LL)
            for (let i = 0; i < swingHighs.length; i++) {
                const prev = swingHighs[i - 1];
                const isHH = !prev || swingHighs[i].price > prev.price;
                swingHighs[i].label  = isHH ? 'HH' : 'LH';
                swingHighs[i].strong = isHH;
            }
            for (let i = 0; i < swingLows.length; i++) {
                const prev = swingLows[i - 1];
                const isHL = !prev || swingLows[i].price > prev.price;
                swingLows[i].label  = isHL ? 'HL' : 'LL';
                swingLows[i].strong = isHL;
            }

            // ── 3. Equal Highs / Equal Lows ─────────────────────────────────
            const eqHighs = [], eqLows = [];
            if (params.show_eqhl !== false) {
                for (let i = 1; i < swingHighs.length; i++) {
                    const a = swingHighs[i - 1], b = swingHighs[i];
                    if (Math.abs(a.price - b.price) / Math.max(a.price, b.price) <= EQ_THRESH)
                        eqHighs.push({ price: (a.price + b.price) / 2, startTime: a.time, endTime: b.time });
                }
                for (let i = 1; i < swingLows.length; i++) {
                    const a = swingLows[i - 1], b = swingLows[i];
                    if (Math.abs(a.price - b.price) / Math.max(a.price, b.price) <= EQ_THRESH)
                        eqLows.push({ price: (a.price + b.price) / 2, startTime: a.time, endTime: b.time });
                }
            }

            // ── 4. BOS / CHoCH (sliding confirmed-swing window) ─────────────
            const structures = [];
            if (params.show_bos !== false) {
                // Build confirm-index lookup for O(1) insertion
                const shByConfirm = new Map(), slByConfirm = new Map();
                swingHighs.forEach(sh => {
                    const ci = sh.index + swingLen + 1;
                    if (!shByConfirm.has(ci)) shByConfirm.set(ci, []);
                    shByConfirm.get(ci).push(sh);
                });
                swingLows.forEach(sl => {
                    const ci = sl.index + swingLen + 1;
                    if (!slByConfirm.has(ci)) slByConfirm.set(ci, []);
                    slByConfirm.get(ci).push(sl);
                });

                let trend = null;
                let watchH = [], watchL = []; // pending structural levels

                for (let i = swingLen + 1; i < n; i++) {
                    // Add newly confirmed swings to watch lists
                    if (shByConfirm.has(i)) shByConfirm.get(i).forEach(sh => watchH.push(sh));
                    if (slByConfirm.has(i)) slByConfirm.get(i).forEach(sl => watchL.push(sl));

                    const close = candles[i].close, time = candles[i].time;

                    // Bullish break: scan from newest → oldest high
                    for (let k = watchH.length - 1; k >= 0; k--) {
                        const sh = watchH[k];
                        if (close > sh.price) {
                            structures.push({ type: trend === 'up' ? 'bos' : 'choch', dir: 'bull',
                                              price: sh.price, startTime: sh.time, breakTime: time });
                            trend = 'up';
                            watchH = watchH.slice(k + 1); // remove broken + older
                            break;
                        }
                    }
                    // Bearish break: scan from newest → oldest low
                    for (let k = watchL.length - 1; k >= 0; k--) {
                        const sl = watchL[k];
                        if (close < sl.price) {
                            structures.push({ type: trend === 'down' ? 'bos' : 'choch', dir: 'bear',
                                              price: sl.price, startTime: sl.time, breakTime: time });
                            trend = 'down';
                            watchL = watchL.slice(k + 1);
                            break;
                        }
                    }
                }
            }

            // ── 5. Fair Value Gaps ───────────────────────────────────────────
            const fvgs = [];
            if (params.show_fvg !== false) {
                for (let i = 2; i < n; i++) {
                    const c0 = candles[i - 2], c2 = candles[i];
                    if (c0.high < c2.low) {
                        const fvg = { type:'bull', top:c2.low, bottom:c0.high, startTime:c0.time, endTime:null, filled:false };
                        for (let k = i + 1; k < n; k++) {
                            if (candles[k].low <= fvg.top) { fvg.filled = true; fvg.endTime = candles[k].time; break; }
                        }
                        fvgs.push(fvg);
                    }
                    if (c0.low > c2.high) {
                        const fvg = { type:'bear', top:c0.low, bottom:c2.high, startTime:c0.time, endTime:null, filled:false };
                        for (let k = i + 1; k < n; k++) {
                            if (candles[k].high >= fvg.bottom) { fvg.filled = true; fvg.endTime = candles[k].time; break; }
                        }
                        fvgs.push(fvg);
                    }
                }
            }

            // ── 6. Order Blocks ──────────────────────────────────────────────
            const obs = [];
            if (params.show_ob !== false) {
                // Bull OB: last bearish candle before price breaks a swing high
                swingHighs.forEach(sh => {
                    for (let i = sh.index + 1; i < n; i++) {
                        if (candles[i].close > sh.price) {
                            const limit = Math.max(sh.index, i - 30);
                            for (let j = i - 1; j > limit; j--) {
                                const c = candles[j];
                                if (c.close < c.open) {
                                    const ob = { type:'bull', top:c.high, bottom:Math.min(c.open, c.close), startTime:c.time, endTime:null, mitigated:false };
                                    for (let k = i + 1; k < n; k++) {
                                        if (candles[k].low <= ob.bottom) { ob.mitigated = true; ob.endTime = candles[k].time; break; }
                                    }
                                    obs.push(ob); break;
                                }
                            }
                            break;
                        }
                    }
                });
                // Bear OB: last bullish candle before price breaks a swing low
                swingLows.forEach(sl => {
                    for (let i = sl.index + 1; i < n; i++) {
                        if (candles[i].close < sl.price) {
                            const limit = Math.max(sl.index, i - 30);
                            for (let j = i - 1; j > limit; j--) {
                                const c = candles[j];
                                if (c.close > c.open) {
                                    const ob = { type:'bear', top:Math.max(c.open, c.close), bottom:c.low, startTime:c.time, endTime:null, mitigated:false };
                                    for (let k = i + 1; k < n; k++) {
                                        if (candles[k].high >= ob.top) { ob.mitigated = true; ob.endTime = candles[k].time; break; }
                                    }
                                    obs.push(ob); break;
                                }
                            }
                            break;
                        }
                    }
                });
            }

            return { swingHighs, swingLows, fvgs, obs, structures, eqHighs, eqLows };
        }

        // ── Apply/remove indicators on existing chart ─────────────────────────
        function applyIndicators(candles, activeInds) {
            if (!tvPriceChart || !tvCandleSeries) return;
            const times  = candles.map(c => c.time);
            const closes = candles.map(c => c.close);

            // Remove deactivated indicators
            Object.keys(tvIndicatorSeries).forEach(key => {
                if (!activeInds.has(key)) {
                    const series = tvIndicatorSeries[key];
                    if (Array.isArray(series)) series.forEach(s => { try { tvPriceChart.removeSeries(s); } catch(e){} });
                    else                       { try { tvPriceChart.removeSeries(series); } catch(e){} }
                    delete tvIndicatorSeries[key];
                }
            });

            // Create-or-update helper: always setData so streaming updates are reflected
            function mkLineSeries(color, lineWidth=1, priceScaleId='right', title='') {
                return tvPriceChart.addLineSeries({ color, lineWidth, priceScaleId,
                    lastValueVisible: true, priceLineVisible: false, title });
            }

            // Helper: filter computed (time, value) pairs to today only.
            // candles may span multiple days (for warmup); we only plot current-day values.
            const dayStart = tvCurrentDayStartTime || 0;
            function todayOnly(pairs) {
                return pairs.filter(p => p !== null && p.time >= dayStart);
            }

            // Helper to get param with fallback default
            function p(key, field, def) { return (tvIndicatorParams[key]||{})[field] ?? def; }
            function mapTA(vals) { return vals.map((v,i) => v!==null ? {time:times[i], value:v} : null); }

            if (activeInds.has('sma9')) {
                if (!tvIndicatorSeries['sma9']) tvIndicatorSeries['sma9'] = mkLineSeries('#ffe082', 1, 'right', 'SMA9');
                tvIndicatorSeries['sma9'].setData(todayOnly(mapTA(calcSMA(closes, p('sma9','period',9)))));
            }
            if (activeInds.has('sma20')) {
                if (!tvIndicatorSeries['sma20']) tvIndicatorSeries['sma20'] = mkLineSeries('#f0c040', 1, 'right', 'SMA20');
                tvIndicatorSeries['sma20'].setData(todayOnly(mapTA(calcSMA(closes, p('sma20','period',20)))));
            }
            if (activeInds.has('sma50')) {
                if (!tvIndicatorSeries['sma50']) tvIndicatorSeries['sma50'] = mkLineSeries('#40a0f0', 1, 'right', 'SMA50');
                tvIndicatorSeries['sma50'].setData(todayOnly(mapTA(calcSMA(closes, p('sma50','period',50)))));
            }
            if (activeInds.has('sma100')) {
                if (!tvIndicatorSeries['sma100']) tvIndicatorSeries['sma100'] = mkLineSeries('#7fd1ff', 1, 'right', 'SMA100');
                tvIndicatorSeries['sma100'].setData(todayOnly(mapTA(calcSMA(closes, p('sma100','period',100)))));
            }
            if (activeInds.has('sma200')) {
                if (!tvIndicatorSeries['sma200']) tvIndicatorSeries['sma200'] = mkLineSeries('#e040fb', 1, 'right', 'SMA200');
                tvIndicatorSeries['sma200'].setData(todayOnly(mapTA(calcSMA(closes, p('sma200','period',200)))));
            }
            if (activeInds.has('ema9')) {
                if (!tvIndicatorSeries['ema9']) tvIndicatorSeries['ema9'] = mkLineSeries('#ff9900', 1, 'right', 'EMA9');
                tvIndicatorSeries['ema9'].setData(todayOnly(mapTA(calcEMA(closes, p('ema9','period',9)))));
            }
            if (activeInds.has('ema21')) {
                if (!tvIndicatorSeries['ema21']) tvIndicatorSeries['ema21'] = mkLineSeries('#00e5ff', 1, 'right', 'EMA21');
                tvIndicatorSeries['ema21'].setData(todayOnly(mapTA(calcEMA(closes, p('ema21','period',21)))));
            }
            if (activeInds.has('ema50')) {
                if (!tvIndicatorSeries['ema50']) tvIndicatorSeries['ema50'] = mkLineSeries('#ff7096', 1, 'right', 'EMA50');
                tvIndicatorSeries['ema50'].setData(todayOnly(mapTA(calcEMA(closes, p('ema50','period',50)))));
            }
            if (activeInds.has('ema100')) {
                if (!tvIndicatorSeries['ema100']) tvIndicatorSeries['ema100'] = mkLineSeries('#b388ff', 1, 'right', 'EMA100');
                tvIndicatorSeries['ema100'].setData(todayOnly(mapTA(calcEMA(closes, p('ema100','period',100)))));
            }
            if (activeInds.has('ema200')) {
                if (!tvIndicatorSeries['ema200']) tvIndicatorSeries['ema200'] = mkLineSeries('#00c853', 1, 'right', 'EMA200');
                tvIndicatorSeries['ema200'].setData(todayOnly(mapTA(calcEMA(closes, p('ema200','period',200)))));
            }
            if (activeInds.has('wma20')) {
                if (!tvIndicatorSeries['wma20']) tvIndicatorSeries['wma20'] = mkLineSeries('#ffd166', 1, 'right', 'WMA20');
                tvIndicatorSeries['wma20'].setData(todayOnly(mapTA(calcWMA(closes, p('wma20','period',20)))));
            }
            if (activeInds.has('wma50')) {
                if (!tvIndicatorSeries['wma50']) tvIndicatorSeries['wma50'] = mkLineSeries('#8ecae6', 1, 'right', 'WMA50');
                tvIndicatorSeries['wma50'].setData(todayOnly(mapTA(calcWMA(closes, p('wma50','period',50)))));
            }
            if (activeInds.has('vwap')) {
                // VWAP resets daily — always compute from today's candles only
                const todayCandles = dayStart > 0 ? candles.filter(c => c.time >= dayStart) : candles;
                const vwapVals = calcVWAP(todayCandles.map(c => ({
                    time: c.time, high: c.high, low: c.low, close: c.close, volume: c.volume || 0
                })));
                if (!tvIndicatorSeries['vwap']) tvIndicatorSeries['vwap'] = mkLineSeries('#ffffff', 1, 'right', 'VWAP');
                tvIndicatorSeries['vwap'].setData(vwapVals.map((v, i) => ({time: todayCandles[i].time, value: v})));
            }
            if (activeInds.has('bb')) {
                const bb = calcBB(closes, p('bb','period',20), p('bb','mult',2));
                if (!tvIndicatorSeries['bb']) {
                    tvIndicatorSeries['bb'] = [
                        mkLineSeries('rgba(100,180,255,0.8)', 1, 'right', 'BB Upper'),
                        mkLineSeries('rgba(100,180,255,0.5)', 1, 'right', 'BB Mid'),
                        mkLineSeries('rgba(100,180,255,0.8)', 1, 'right', 'BB Lower'),
                    ];
                }
                const [upperS, midS, lowerS] = tvIndicatorSeries['bb'];
                upperS.setData(todayOnly(bb.map((v,i) => v.upper!==null ? {time:times[i],value:v.upper} : null)));
                midS.setData(  todayOnly(bb.map((v,i) => v.mid  !==null ? {time:times[i],value:v.mid}   : null)));
                lowerS.setData(todayOnly(bb.map((v,i) => v.lower!==null ? {time:times[i],value:v.lower}  : null)));
            }
            if (activeInds.has('fbb')) {
                const fbb = calcFBB(candles, p('fbb','length',200), p('fbb','mult',3));
                if (!tvIndicatorSeries['fbb'] || tvIndicatorSeries['fbb'].length !== 3) {
                    if (tvIndicatorSeries['fbb']) {
                        tvIndicatorSeries['fbb'].forEach(s => { try { tvPriceChart.removeSeries(s); } catch(e){} });
                    }
                    tvIndicatorSeries['fbb'] = [
                        mkLineSeries('#e040fb', 2, 'right', 'FBB Basis'),
                        mkLineSeries('#ef5350', 2, 'right', 'FBB Upper'),
                        mkLineSeries('#66bb6a', 2, 'right', 'FBB Lower'),
                    ];
                }
                const [fbbBasis, fbbUpper, fbbLower] = tvIndicatorSeries['fbb'];
                fbbBasis.setData(todayOnly(fbb.map((v, i) => v.basis !== null ? { time: times[i], value: v.basis } : null)));
                fbbUpper.setData(todayOnly(fbb.map((v, i) => v.upper !== null ? { time: times[i], value: v.upper } : null)));
                fbbLower.setData(todayOnly(fbb.map((v, i) => v.lower !== null ? { time: times[i], value: v.lower } : null)));
            }
            if (activeInds.has('atr')) {
                const atrVals = calcATR(candles, p('atr','period',14));
                const ema20   = calcEMA(closes, 20);
                const mult    = 1.5;
                if (!tvIndicatorSeries['atr']) {
                    tvIndicatorSeries['atr'] = [
                        mkLineSeries('rgba(255,152,0,0.8)', 1, 'right', 'ATR Upper'),
                        mkLineSeries('rgba(255,152,0,0.8)', 1, 'right', 'ATR Lower'),
                    ];
                }
                const [atrUpper, atrLower] = tvIndicatorSeries['atr'];
                atrUpper.setData(todayOnly(ema20.map((v,i) => (v!==null && atrVals[i]!==null) ? {time:times[i], value:v + mult*atrVals[i]} : null)));
                atrLower.setData(todayOnly(ema20.map((v,i) => (v!==null && atrVals[i]!==null) ? {time:times[i], value:v - mult*atrVals[i]} : null)));
            }
            if (activeInds.has('auto_trend')) {
                if (!tvIndicatorSeries['auto_trend']) {
                    tvIndicatorSeries['auto_trend'] = tvPriceChart.addLineSeries({
                        color: '#ffca28',
                        lineWidth: 2,
                        lineStyle: LightweightCharts.LineStyle.LargeDashed,
                        priceScaleId: 'right',
                        lastValueVisible: true,
                        priceLineVisible: false,
                        title: 'Auto Trend'
                    });
                }
                tvIndicatorSeries['auto_trend'].setData(calcAutoTrendLine(candles, dayStart));
            }

            // RSI and MACD sub-panes: compute with full history but display today only
            const todayCandles = dayStart > 0 ? candles.filter(c => c.time >= dayStart) : candles;
            if (activeInds.has('rsi')) applyRsiPane(candles, todayCandles, p('rsi','period',14));
            else                       destroyRsiPane();
            if (activeInds.has('macd')) applyMacdPane(candles, todayCandles, p('macd','fast',12), p('macd','slow',26), p('macd','signal',9));
            else                        destroyMacdPane();
            if (activeInds.has('arv')) applyArvPane(candles, todayCandles);
            else                       destroyArvPane();

            // SMC — canvas-based overlay
            if (activeInds.has('smc')) {
                tvSmcData = calcSMC(candles, tvIndicatorParams.smc || {});
                scheduleSmcDraw();
            } else {
                tvSmcData = null;
                const smcOverlay = document.getElementById('price-chart')?.querySelector('.tv-smc-overlay');
                if (smcOverlay) smcOverlay.style.display = 'none';
            }

            // Update legend overlay
            updateIndicatorLegend();
        }

        // ── Indicator legend ─────────────────────────────────────────────────
        function updateIndicatorLegend() {
            const container = document.getElementById('price-chart');
            if (!container) return;
            let legend = container.querySelector('.tv-indicator-legend');
            if (!legend) {
                legend = document.createElement('div');
                legend.className = 'tv-indicator-legend';
                container.appendChild(legend);
            }
            function pLabel(key, field, def) { return (tvIndicatorParams[key]||{})[field] ?? def; }
            const items = {
                sma9:`SMA${pLabel('sma9','period',9)}`,sma20:`SMA${pLabel('sma20','period',20)}`,
                sma50:`SMA${pLabel('sma50','period',50)}`,sma100:`SMA${pLabel('sma100','period',100)}`,sma200:`SMA${pLabel('sma200','period',200)}`,
                ema9:`EMA${pLabel('ema9','period',9)}`,ema21:`EMA${pLabel('ema21','period',21)}`,
                ema50:`EMA${pLabel('ema50','period',50)}`,ema100:`EMA${pLabel('ema100','period',100)}`,ema200:`EMA${pLabel('ema200','period',200)}`,
                wma20:`WMA${pLabel('wma20','period',20)}`,wma50:`WMA${pLabel('wma50','period',50)}`,
                vwap:'VWAP',bb:`BB(${pLabel('bb','period',20)},${pLabel('bb','mult',2)})`,
                fbb:`FBB(${pLabel('fbb','length',200)},${pLabel('fbb','mult',3)})`,auto_trend:'Auto Trend',
                rsi:`RSI${pLabel('rsi','period',14)}`,macd:'MACD',atr:'ATR Bands',smc:'SMC'
            };
            const colors = {
                sma9:'#ffe082',sma20:'#f0c040',sma50:'#40a0f0',sma100:'#7fd1ff',sma200:'#e040fb',
                ema9:'#ff9900',ema21:'#00e5ff',ema50:'#ff7096',ema100:'#b388ff',ema200:'#00c853',
                wma20:'#ffd166',wma50:'#8ecae6',
                vwap:'#ffffff',bb:'rgba(100,180,255,0.8)',fbb:'#e040fb',auto_trend:'#ffca28',
                rsi:'#e91e63',macd:'#2196f3',atr:'rgba(255,152,0,0.8)',smc:'#26a69a'
            };
            const allActive = [...Object.keys(tvIndicatorSeries), ...(tvSmcData ? ['smc'] : [])];
            legend.innerHTML = allActive.map(k => `
                <div class="tv-legend-item">
                    <div class="tv-legend-swatch" style="background:${colors[k]||'#888'}"></div>
                    ${items[k]||k}
                </div>`).join('');
            legend.style.display = allActive.length ? 'flex' : 'none';
        }

        // ── SMC Canvas Overlay ────────────────────────────────────────────────
        function ensureSmcOverlay() {
            const c = document.getElementById('price-chart');
            if (!c) return null;
            let o = c.querySelector('.tv-smc-overlay');
            if (!o) {
                o = document.createElement('div');
                o.className = 'tv-smc-overlay';
                c.appendChild(o);
            }
            return o;
        }

        function drawSmcCanvas() {
            const overlay = ensureSmcOverlay();
            if (!overlay || !tvPriceChart || !tvCandleSeries || !tvSmcData) {
                if (overlay) overlay.style.display = 'none';
                return;
            }
            // Show before measuring — display:none gives clientWidth/Height = 0
            overlay.style.display = 'block';
            let canvas = overlay.querySelector('canvas');
            if (!canvas) {
                canvas = document.createElement('canvas');
                overlay.appendChild(canvas);
            }
            const dpr = window.devicePixelRatio || 1;
            const w = Math.max(1, overlay.clientWidth  || overlay.parentElement?.clientWidth  || 800);
            const h = Math.max(1, overlay.clientHeight || overlay.parentElement?.clientHeight || 600);
            canvas.width  = Math.round(w * dpr);
            canvas.height = Math.round(h * dpr);
            canvas.style.width  = w + 'px';
            canvas.style.height = h + 'px';

            const ctx = canvas.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);

            const ts = tvPriceChart.timeScale();
            const toX = t => { const x = ts.timeToCoordinate(t); return (x == null || isNaN(x)) ? null : x; };
            const toY = p => { const y = tvCandleSeries.priceToCoordinate(p); return (y == null || isNaN(y)) ? null : y; };
            const lastTime = tvLastCandles.length ? tvLastCandles[tvLastCandles.length - 1].time : null;
            const params = tvIndicatorParams.smc || {};
            const { obs, fvgs, structures, swingHighs, swingLows, eqHighs = [], eqLows = [] } = tvSmcData;

            const BULL = '#26a69a', BEAR = '#ef5350';

            // Helper: draw a filled+stroked box, clamping start to left edge if off-screen
            function drawBox(x1, x2end, topPrice, botPrice, fill, stroke, fillA, strokeA) {
                let lx = toX(x1);
                const rx = x2end ? toX(x2end) : (lastTime ? toX(lastTime) : null);
                const ty = toY(topPrice), by = toY(botPrice);
                if (ty == null || by == null) return;
                if (lx == null) lx = 0;
                const rx2 = rx != null ? rx : w;
                if (rx2 < 0) return;
                const bx = Math.min(lx, rx2), bw = Math.abs(rx2 - lx);
                const by2 = Math.min(ty, by), bh = Math.abs(by - ty);
                if (bh < 0.5) return;
                ctx.save();
                ctx.globalAlpha = fillA; ctx.fillStyle = fill;
                ctx.fillRect(bx, by2, bw, bh);
                ctx.globalAlpha = strokeA; ctx.strokeStyle = stroke; ctx.lineWidth = 1.5;
                ctx.strokeRect(bx + 0.5, by2 + 0.5, bw - 1, bh - 1);
                ctx.restore();
            }

            // Helper: small pill label
            function drawLabel(text, cx, cy, col, bgAlpha) {
                ctx.save();
                ctx.font = 'bold 9px sans-serif';
                const tw = ctx.measureText(text).width;
                ctx.globalAlpha = bgAlpha ?? 0.72;
                ctx.fillStyle = '#111';
                ctx.beginPath();
                ctx.roundRect(cx - tw / 2 - 4, cy - 7, tw + 8, 14, 3);
                ctx.fill();
                ctx.globalAlpha = 1;
                ctx.fillStyle = col;
                ctx.textBaseline = 'middle';
                ctx.textAlign = 'center';
                ctx.fillText(text, cx, cy);
                ctx.restore();
            }

            // ── FVGs ─────────────────────────────────────────────────────────
            if (params.show_fvg !== false) {
                fvgs.forEach(fvg => {
                    const col = fvg.type === 'bull' ? BULL : BEAR;
                    drawBox(fvg.startTime, fvg.endTime, fvg.top, fvg.bottom, col, col,
                            fvg.filled ? 0.06 : 0.18, fvg.filled ? 0.18 : 0.55);
                    // Label on unfilled FVGs only
                    if (!fvg.filled) {
                        let lx = toX(fvg.startTime); if (lx == null) lx = 0;
                        const rx2 = fvg.endTime ? toX(fvg.endTime) : (lastTime ? toX(lastTime) : null);
                        const ty = toY(fvg.top), by = toY(fvg.bottom);
                        if (ty != null && by != null) {
                            const midX = lx + (rx2 != null ? (rx2 - lx) * 0.15 : 30);
                            const midY = (ty + by) / 2;
                            ctx.save(); ctx.font = 'bold 8px sans-serif';
                            ctx.globalAlpha = 0.8; ctx.fillStyle = col;
                            ctx.textBaseline = 'middle';
                            ctx.fillText('FVG', midX, midY);
                            ctx.restore();
                        }
                    }
                });
            }

            // ── Order Blocks ──────────────────────────────────────────────────
            if (params.show_ob !== false) {
                obs.forEach(ob => {
                    const col = ob.type === 'bull' ? BULL : BEAR;
                    drawBox(ob.startTime, ob.endTime, ob.top, ob.bottom, col, col,
                            ob.mitigated ? 0.07 : 0.22, ob.mitigated ? 0.22 : 0.85);
                    if (!ob.mitigated) {
                        let lx = toX(ob.startTime); if (lx == null) lx = 4;
                        const ty = toY(ob.top);
                        if (ty != null) {
                            const tag = ob.type === 'bull' ? 'OB ▲' : 'OB ▼';
                            ctx.save(); ctx.font = 'bold 9px sans-serif';
                            ctx.globalAlpha = 0.9; ctx.fillStyle = col;
                            ctx.textBaseline = 'top';
                            ctx.fillText(tag, lx + 4, ty + 3);
                            ctx.restore();
                        }
                    }
                });
            }

            // ── BOS / CHoCH ───────────────────────────────────────────────────
            if (params.show_bos !== false) {
                structures.forEach(st => {
                    let x1 = toX(st.startTime), x2 = toX(st.breakTime), y = toY(st.price);
                    if (y == null || x2 == null) return;
                    if (x1 == null) x1 = 0;
                    const col = st.dir === 'bull' ? BULL : BEAR;
                    const label = st.type === 'choch' ? 'CHoCH' : 'BOS';
                    ctx.save();
                    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
                    ctx.setLineDash(st.type === 'choch' ? [5, 3] : []);
                    ctx.globalAlpha = 0.85;
                    ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.restore();
                    drawLabel(label, (x1 + x2) / 2, y - 9, col);
                });
            }

            // ── Equal Highs / Equal Lows ──────────────────────────────────────
            if (params.show_eqhl !== false) {
                const drawEQ = (eq, col, labelText) => {
                    let x1 = toX(eq.startTime); if (x1 == null) x1 = 0;
                    const x2 = toX(eq.endTime); if (x2 == null) return;
                    const y = toY(eq.price); if (y == null) return;
                    ctx.save();
                    ctx.strokeStyle = col; ctx.lineWidth = 1;
                    ctx.setLineDash([4, 3]); ctx.globalAlpha = 0.75;
                    ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.restore();
                    drawLabel(labelText, x2 + 20, y, col, 0.65);
                };
                eqHighs.forEach(eq => drawEQ(eq, BEAR, 'EQH'));
                eqLows.forEach(eq => drawEQ(eq, BULL, 'EQL'));
            }

            // ── Swing Points with Strong / Weak labels ────────────────────────
            if (params.show_swings !== false) {
                const drawSwing = (sw, isHigh) => {
                    const x = toX(sw.time), y = toY(sw.price);
                    if (x == null || y == null) return;
                    const col = isHigh ? BEAR : BULL;
                    const size = 5;
                    // Triangle marker
                    ctx.save(); ctx.fillStyle = col; ctx.globalAlpha = 0.9;
                    ctx.beginPath();
                    if (isHigh) {
                        // pointing down above the high
                        ctx.moveTo(x, y - 4);
                        ctx.lineTo(x - size, y - 4 - size);
                        ctx.lineTo(x + size, y - 4 - size);
                    } else {
                        // pointing up below the low
                        ctx.moveTo(x, y + 4);
                        ctx.lineTo(x - size, y + 4 + size);
                        ctx.lineTo(x + size, y + 4 + size);
                    }
                    ctx.closePath(); ctx.fill(); ctx.restore();
                    // HH/LH/HL/LL label
                    const labelY = isHigh ? y - 4 - size - 10 : y + 4 + size + 10;
                    drawLabel(sw.label, x, labelY, col, 0.6);
                    // Strong / Weak sub-label
                    const swText = sw.strong ? 'Strong' : 'Weak';
                    ctx.save(); ctx.font = '8px sans-serif'; ctx.globalAlpha = 0.55;
                    ctx.fillStyle = col; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
                    const subY = isHigh ? labelY - 12 : labelY + 12;
                    ctx.fillText(swText, x, subY);
                    ctx.restore();
                };
                swingHighs.forEach(sh => drawSwing(sh, true));
                swingLows.forEach(sl => drawSwing(sl, false));
            }
        }

        function scheduleSmcDraw() {
            if (tvSmcDrawPending) return;
            tvSmcDrawPending = true;
            requestAnimationFrame(() => { tvSmcDrawPending = false; drawSmcCanvas(); });
        }

        // ── Sub-pane chart helper functions ──────────────────────────────────
        const tvRightScaleMinimumWidth = 88;

        function createSubPaneChart(element, height) {
            if (!element) return null;
            return LightweightCharts.createChart(element, Object.assign({}, buildLightweightThemeOptions(), {
                autoSize: true,
                height: height,
                rightPriceScale: { borderColor: getThemeValue('--border-color', '#333333'), scaleMargins: { top: 0.1, bottom: 0.1 }, minimumWidth: tvRightScaleMinimumWidth },
                localization: {
                    timeFormatter: (time) => {
                        const d = new Date(time * 1000);
                        return d.toLocaleTimeString('en-US', {
                            hour: '2-digit', minute: '2-digit',
                            hour12: false, timeZone: 'America/New_York'
                        });
                    }
                },
                timeScale: {
                    borderColor: getThemeValue('--border-color', '#333333'), timeVisible: true, secondsVisible: false,
                    fixLeftEdge: false, fixRightEdge: false,
                    tickMarkFormatter: (time) => {
                        const d = new Date(time * 1000);
                        return d.toLocaleTimeString('en-US', {
                            hour: '2-digit', minute: '2-digit',
                            hour12: false, timeZone: 'America/New_York'
                        });
                    }
                },
                handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
                handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
            }));
        }

        function setupTimeScaleSync() {
            // Remove old subscriptions first
            tvSyncHandlers.forEach(({chart, handler}) => {
                try { chart.timeScale().unsubscribeVisibleLogicalRangeChange(handler); } catch(e){}
            });
            tvSyncHandlers = [];
            const allCharts = [tvPriceChart, tvRsiChart, tvMacdChart, tvArvChart].filter(Boolean);
            if (allCharts.length < 2) return;
            allCharts.forEach(srcChart => {
                const others = allCharts.filter(c => c !== srcChart);
                const handler = (range) => {
                    if (tvSyncingTimeScale || !range) return;
                    tvSyncingTimeScale = true;
                    others.forEach(c => { try { c.timeScale().setVisibleLogicalRange(range); } catch(e){} });
                    tvSyncingTimeScale = false;
                };
                try { srcChart.timeScale().subscribeVisibleLogicalRangeChange(handler); } catch(e){}
                tvSyncHandlers.push({ chart: srcChart, handler });
            });
            // Immediately match current main chart range
            if (tvPriceChart) {
                try {
                    const range = tvPriceChart.timeScale().getVisibleLogicalRange();
                    if (range) [tvRsiChart, tvMacdChart, tvArvChart].filter(Boolean).forEach(c => {
                        try { c.timeScale().setVisibleLogicalRange(range); } catch(e){}
                    });
                } catch(e){}
            }
        }

        function applyRsiPane(allCandles, todayCandles, rsiPeriod=14) {
            const pane = document.getElementById('rsi-pane');
            if (!pane) return;
            pane.style.display = 'block';
            // Compute RSI using full history for warmup, then filter to today for display
            const allTimes   = allCandles.map(c => c.time);
            const rsiVals    = calcRSI(allCandles.map(c => c.close), rsiPeriod);
            const dayStart   = tvCurrentDayStartTime || 0;
            const rsiData    = rsiVals
                .map((v,i) => v!==null ? {time:allTimes[i],value:v} : null)
                .filter(p => p !== null && p.time >= dayStart);
            if (!tvRsiChart) {
                const chartEl = document.getElementById('rsi-chart');
                if (!chartEl) return;
                tvRsiChart = createSubPaneChart(chartEl, 110);
                tvRsiSeries = tvRsiChart.addLineSeries({
                    color: '#e91e63', lineWidth: 1.5,
                    lastValueVisible: true, priceLineVisible: false, title: 'RSI14'
                });
                tvRsiSeries.createPriceLine({ price: 70, color: 'rgba(255,100,100,0.7)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '70' });
                tvRsiSeries.createPriceLine({ price: 50, color: 'rgba(150,150,150,0.4)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: false, title: '' });
                tvRsiSeries.createPriceLine({ price: 30, color: 'rgba(100,200,100,0.7)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '30' });
            }
            if (rsiData.length) tvRsiSeries.setData(rsiData);
            setupTimeScaleSync();
        }

        function destroyRsiPane() {
            const pane = document.getElementById('rsi-pane');
            if (pane) pane.style.display = 'none';
            if (tvRsiChart) {
                tvSyncHandlers = tvSyncHandlers.filter(h => h.chart !== tvRsiChart);
                try { tvRsiChart.remove(); } catch(e){}
                tvRsiChart = null; tvRsiSeries = null;
            }
        }

        function applyMacdPane(allCandles, todayCandles, macdFast=12, macdSlow=26, macdSignal=9) {
            const pane = document.getElementById('macd-pane');
            if (!pane) return;
            pane.style.display = 'block';
            // Compute MACD using full history for warmup, then filter to today for display
            const allTimes = allCandles.map(c => c.time);
            const macdData = calcMACD(allCandles.map(c => c.close), macdFast, macdSlow, macdSignal);
            const dayStart = tvCurrentDayStartTime || 0;
            function todayOnly(pairs) { return pairs.filter(p => p !== null && p.time >= dayStart); }
            const histData = todayOnly(macdData.histogram.map((v,i) => v!==null ? {time:allTimes[i],value:v,color:v>=0?'rgba(76,175,80,0.8)':'rgba(244,67,54,0.8)'} : null));
            const lineData = todayOnly(macdData.macd.map((v,i)    => v!==null ? {time:allTimes[i],value:v} : null));
            const sigData  = todayOnly(macdData.signal.map((v,i)  => v!==null ? {time:allTimes[i],value:v} : null));
            if (!tvMacdChart) {
                const chartEl = document.getElementById('macd-chart');
                if (!chartEl) return;
                tvMacdChart = createSubPaneChart(chartEl, 120);
                tvMacdSeries.hist   = tvMacdChart.addHistogramSeries({ lastValueVisible: false, priceLineVisible: false });
                tvMacdSeries.line   = tvMacdChart.addLineSeries({ color: '#2196f3', lineWidth: 1.5, lastValueVisible: true, priceLineVisible: false, title: 'MACD' });
                tvMacdSeries.signal = tvMacdChart.addLineSeries({ color: '#ff9800', lineWidth: 1,   lastValueVisible: true, priceLineVisible: false, title: 'Signal' });
                tvMacdSeries.line.createPriceLine({ price: 0, color: 'rgba(150,150,150,0.4)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: false, title: '' });
            }
            if (histData.length) tvMacdSeries.hist.setData(histData);
            if (lineData.length) tvMacdSeries.line.setData(lineData);
            if (sigData.length)  tvMacdSeries.signal.setData(sigData);
            setupTimeScaleSync();
        }

        function destroyMacdPane() {
            const pane = document.getElementById('macd-pane');
            if (pane) pane.style.display = 'none';
            if (tvMacdChart) {
                tvSyncHandlers = tvSyncHandlers.filter(h => h.chart !== tvMacdChart);
                try { tvMacdChart.remove(); } catch(e){}
                tvMacdChart = null; tvMacdSeries = {};
            }
        }

        function applyArvPane(allCandles, todayCandles) {
            const pane = document.getElementById('arv-pane');
            if (!pane) return;
            pane.style.display = 'block';

            const allTimes = allCandles.map(c => c.time);
            const dayStart = tvCurrentDayStartTime || 0;
            const arvData = calcARV(allCandles, 20)
                .map((value, index) => Number.isFinite(value) ? { time: allTimes[index], value } : null)
                .filter(point => point !== null && point.time >= dayStart);

            if (!tvArvChart) {
                const chartEl = document.getElementById('arv-chart');
                if (!chartEl) return;
                tvArvChart = createSubPaneChart(chartEl, 110);
                tvArvSeries = tvArvChart.addLineSeries({
                    color: '#26c6da',
                    lineWidth: 1.5,
                    lastValueVisible: true,
                    priceLineVisible: false,
                    title: 'ARV20 %',
                    priceFormat: { type: 'price', precision: 2, minMove: 0.01 }
                });
            }

            tvArvSeries.setData(arvData);
            setupTimeScaleSync();
        }

        function destroyArvPane() {
            const pane = document.getElementById('arv-pane');
            if (pane) pane.style.display = 'none';
            if (tvArvChart) {
                tvSyncHandlers = tvSyncHandlers.filter(h => h.chart !== tvArvChart);
                try { tvArvChart.remove(); } catch(e){}
                tvArvChart = null;
                tvArvSeries = null;
            }
        }


        // ── Drawing tools ─────────────────────────────────────────────────────
        function setDrawMode(mode) {
            tvDrawMode = (tvDrawMode === mode) ? null : mode;  // toggle
            tvDrawStart = null;
            const container = document.getElementById('price-chart');
            if (!container) return;
            // crosshair cursor applied via CSS on canvas child
            Array.from(container.querySelectorAll('canvas')).forEach(c => {
                c.style.cursor = tvDrawMode ? 'crosshair' : '';
            });
            // Sync button states
            document.querySelectorAll('.tv-tb-btn[data-draw]').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.draw === tvDrawMode);
            });
        }

        function tvUndoDrawing() {
            if (!tvPriceChart || tvDrawings.length === 0) return;
            const last = tvDrawings.pop();
            tvDrawingDefs.pop(); // keep defs in sync
            if (Array.isArray(last)) last.forEach(s => {
                if (s && s._isLine) { try { tvCandleSeries.removePriceLine(s); } catch(e){} }
                else                { try { tvPriceChart.removeSeries(s); }      catch(e){} }
            });
            else if (last && last._isLine) { try { tvCandleSeries.removePriceLine(last); } catch(e){} }
            else                           { try { tvPriceChart.removeSeries(last); }     catch(e){} }
        }

        function tvClearDrawings() {
            if (!tvPriceChart) return;
            // Remove all series first, then clear both arrays together
            while (tvDrawings.length > 0) {
                const last = tvDrawings.pop();
                if (Array.isArray(last)) last.forEach(s => {
                    if (s && s._isLine) { try { tvCandleSeries.removePriceLine(s); } catch(e){} }
                    else                { try { tvPriceChart.removeSeries(s); }      catch(e){} }
                });
                else if (last && last._isLine) { try { tvCandleSeries.removePriceLine(last); } catch(e){} }
                else                           { try { tvPriceChart.removeSeries(last); }     catch(e){} }
            }
            tvDrawingDefs = [];
        }

        function tvRestoreDrawings() {
            if (!tvPriceChart || !tvCandleSeries) return;
            tvDrawings = [];
            for (const def of tvDrawingDefs) {
                if (def.type === 'hline') {
                    const line = tvCandleSeries.createPriceLine({
                        price: def.price, color: def.color, lineWidth: 1,
                        lineStyle: LightweightCharts.LineStyle.Solid,
                        axisLabelVisible: true, title: ''
                    });
                    line._isLine = true;
                    tvDrawings.push(line);
                    tvAllLevelPrices.push(def.price);
                } else if (def.type === 'trendline') {
                    const tMin = Math.min(def.t1, def.t2), tMax = Math.max(def.t1, def.t2);
                    const vAtMin = def.t1 <= def.t2 ? def.p1 : def.p2;
                    const vAtMax = def.t1 <= def.t2 ? def.p2 : def.p1;
                    const s = tvPriceChart.addLineSeries({
                        color: def.color, lineWidth: 1, priceScaleId: 'right',
                        lastValueVisible: false, priceLineVisible: false
                    });
                    s.setData([{ time: tMin, value: vAtMin }, { time: tMax, value: vAtMax }]);
                    tvDrawings.push(s);
                } else if (def.type === 'rect') {
                    const topL = tvCandleSeries.createPriceLine({ price: def.top, color: def.color, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: false, title: '' });
                    const botL = tvCandleSeries.createPriceLine({ price: def.bot, color: def.color, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: false, title: '' });
                    topL._isLine = true; botL._isLine = true;
                    tvDrawings.push([topL, botL]);
                } else if (def.type === 'text') {
                    const line = tvCandleSeries.createPriceLine({
                        price: def.price, color: def.color, lineWidth: 1, lineVisible: false,
                        lineStyle: LightweightCharts.LineStyle.Solid,
                        axisLabelVisible: true, title: def.text
                    });
                    line._isLine = true;
                    tvDrawings.push(line);
                }
            }
        }

        function tvHandleChartClick(param) {
            if (!tvDrawMode || !param || !param.point) return;
            const price = tvCandleSeries ? tvCandleSeries.coordinateToPrice(param.point.y) : null;
            if (price === null || price === undefined) return;

            if (tvDrawMode === 'hline') {
                // H-Line only needs the Y coordinate — no need for param.time
                const drawColor = document.getElementById('tv-draw-color') ? document.getElementById('tv-draw-color').value : '#FFD700';
                const line = tvCandleSeries.createPriceLine({
                    price, color: drawColor, lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Solid,
                    axisLabelVisible: true, title: ''
                });
                line._isLine = true;
                tvDrawings.push(line);
                tvDrawingDefs.push({ type: 'hline', price, color: drawColor });
                // Extend autoscale range to include this drawn level
                tvAllLevelPrices.push(price);
                tvApplyAutoscale();
                return;
            }

            // Resolve time — use param.time if available, otherwise snap to nearest candle
            let clickTime = param.time;
            if (!clickTime && tvLastCandles && tvLastCandles.length) {
                // coordinateToTime can be null in empty areas; fall back to snapping to nearest candle
                try { clickTime = tvPriceChart.timeScale().coordinateToTime(param.point.x); } catch(e){}
                if (!clickTime) {
                    // snap to closest candle time by logical index
                    const logical = param.logical != null ? param.logical : tvLastCandles.length - 1;
                    const idx = Math.max(0, Math.min(Math.round(logical), tvLastCandles.length - 1));
                    clickTime = tvLastCandles[idx].time;
                }
            }

            if (tvDrawMode === 'trendline' || tvDrawMode === 'rect') {
                if (!clickTime) return; // still no time — bail
                if (!tvDrawStart) {
                    tvDrawStart = { price, time: clickTime };
                    // Show visual hint that first point is set
                    const container = document.getElementById('price-chart');
                    if (container) { container.title = 'Click second point to complete drawing'; }
                } else {
                    const drawColor = document.getElementById('tv-draw-color') ? document.getElementById('tv-draw-color').value : '#FFD700';
                    if (tvDrawMode === 'trendline') {
                        const t1 = tvDrawStart.time, p1 = tvDrawStart.price;
                        const t2 = clickTime,        p2 = price;
                        const tMin = Math.min(t1, t2), tMax = Math.max(t1, t2);
                        const vAtMin = t1 <= t2 ? p1 : p2;
                        const vAtMax = t1 <= t2 ? p2 : p1;
                        const s = tvPriceChart.addLineSeries({
                            color: drawColor, lineWidth: 1, priceScaleId: 'right',
                            lastValueVisible: false, priceLineVisible: false
                        });
                        s.setData([{ time: tMin, value: vAtMin }, { time: tMax, value: vAtMax }]);
                        tvDrawings.push(s);
                        tvDrawingDefs.push({ type: 'trendline', t1, p1, t2, p2, color: drawColor });
                    } else if (tvDrawMode === 'rect') {
                        const top = Math.max(tvDrawStart.price, price);
                        const bot = Math.min(tvDrawStart.price, price);
                        const topL = tvCandleSeries.createPriceLine({ price: top, color: drawColor, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: false, title: '' });
                        const botL = tvCandleSeries.createPriceLine({ price: bot, color: drawColor, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Solid, axisLabelVisible: false, title: '' });
                        topL._isLine = true; botL._isLine = true;
                        tvDrawings.push([topL, botL]);
                        tvDrawingDefs.push({ type: 'rect', top, bot, color: drawColor });
                    }
                    tvDrawStart = null;
                    const container = document.getElementById('price-chart');
                    if (container) container.title = '';
                }
                return;
            }

            if (tvDrawMode === 'text') {
                const userText = prompt('Enter label text:');
                if (!userText) return;
                const drawColor = document.getElementById('tv-draw-color') ? document.getElementById('tv-draw-color').value : '#FFD700';
                const line = tvCandleSeries.createPriceLine({
                    price, color: drawColor, lineWidth: 1, lineVisible: false,
                    lineStyle: LightweightCharts.LineStyle.Solid,
                    axisLabelVisible: true, title: userText
                });
                line._isLine = true;
                tvDrawings.push(line);
                tvDrawingDefs.push({ type: 'text', price, text: userText, color: drawColor });
            }
        }

        // ── Candle Close Timer ─────────────────────────────────────────────────────
        function startCandleCloseTimer() {
            if (candleCloseTimerInterval) clearInterval(candleCloseTimerInterval);
            function updateTimer() {
                const el = document.getElementById('candle-close-timer');
                if (!el) { clearInterval(candleCloseTimerInterval); return; }
                const tfEl = document.getElementById('timeframe');
                const tf = tfEl ? parseInt(tfEl.value) || 1 : 1;
                const tfSecs = tf * 60;
                // Use ET (America/New_York) time for candle boundary calculation
                const now = new Date();
                const etFormatter = new Intl.DateTimeFormat('en-US', {
                    timeZone: 'America/New_York',
                    hour: 'numeric', minute: 'numeric', second: 'numeric', hour12: false
                });
                const parts = etFormatter.formatToParts(now);
                let h = 0, m2 = 0, s2 = 0;
                for (const p of parts) {
                    if (p.type === 'hour')   h  = parseInt(p.value);
                    if (p.type === 'minute') m2 = parseInt(p.value);
                    if (p.type === 'second') s2 = parseInt(p.value);
                }
                const secondsOfDay = h * 3600 + m2 * 60 + s2;
                const elapsed = secondsOfDay % tfSecs;
                const remaining = tfSecs - elapsed;
                const minutes = Math.floor(remaining / 60);
                const seconds = remaining % 60;
                el.textContent = `⏱ ${minutes}:${seconds.toString().padStart(2, '0')}`;
            }
            updateTimer();
            candleCloseTimerInterval = setInterval(updateTimer, 1000);
        }

        // ── Build the chart toolbar ──────────────────────────────────────────
        function buildTVToolbar(container, candles, upColor, downColor) {
            const toolbarContainer = document.getElementById('tv-toolbar-container');
            if (!toolbarContainer) return;
            toolbarContainer.innerHTML = '';
            const toolbar = toolbarContainer;
            toolbar.className = 'tv-toolbar-container';

            // Use the persistent global set so state survives data refreshes
            // (tvActiveInds is declared at page level)

            function btn(text, title, onClick, extraClass='') {
                const b = document.createElement('button');
                b.className = 'tv-tb-btn' + (extraClass ? ' ' + extraClass : '');
                b.textContent = text;
                b.title = title;
                b.addEventListener('click', onClick);
                return b;
            }

            const indicatorDefs = [
                { key:'sma9',   label:'SMA9',   title:'Simple Moving Average (9)' },
                { key:'sma20',  label:'SMA20',  title:'Simple Moving Average (20)' },
                { key:'sma50',  label:'SMA50',  title:'Simple Moving Average (50)' },
                { key:'sma100', label:'SMA100', title:'Simple Moving Average (100)' },
                { key:'sma200', label:'SMA200', title:'Simple Moving Average (200)' },
                { key:'ema9',   label:'EMA9',   title:'Exponential Moving Average (9)' },
                { key:'ema21',  label:'EMA21',  title:'Exponential Moving Average (21)' },
                { key:'ema50',  label:'EMA50',  title:'Exponential Moving Average (50)' },
                { key:'ema100', label:'EMA100', title:'Exponential Moving Average (100)' },
                { key:'ema200', label:'EMA200', title:'Exponential Moving Average (200)' },
                { key:'wma20',  label:'WMA20',  title:'Weighted Moving Average (20)' },
                { key:'wma50',  label:'WMA50',  title:'Weighted Moving Average (50)' },
                { key:'vwap',   label:'VWAP',   title:'Volume Weighted Average Price' },
                { key:'bb',     label:'BB',     title:'Bollinger Bands (20, 2)' },
                { key:'fbb',    label:'FBB',    title:'Fibonacci Bollinger Bands (200, 3)' },
                { key:'auto_trend', label:'Auto Trend', title:'Automatic linear-regression trend line for the current session' },
                { key:'rsi',    label:'RSI',    title:'Relative Strength Index (14) — sub-pane' },
                { key:'macd',   label:'MACD',   title:'MACD (12, 26, 9) — sub-pane' },
                { key:'arv',    label:'ARV',    title:'Annualized Realized Volatility (20) — sub-pane' },
                { key:'atr',    label:'ATR',    title:'Average True Range (14) — sub-pane' },
                { key:'smc',    label:'SMC',    title:'Smart Money Concepts — Order Blocks, FVGs, BOS/CHoCH' },
            ];
            const indicatorPicker = document.createElement('details');
            indicatorPicker.className = 'tv-indicator-picker';
            const indicatorSummary = document.createElement('summary');
            indicatorSummary.className = 'tv-tb-btn tv-indicator-summary';
            indicatorSummary.title = 'Search and toggle indicators';
            const indicatorLabel = document.createElement('span');
            indicatorLabel.textContent = 'Indicators';
            const indicatorBadge = document.createElement('span');
            indicatorBadge.className = 'tv-indicator-badge';
            indicatorSummary.appendChild(indicatorLabel);
            indicatorSummary.appendChild(indicatorBadge);
            indicatorPicker.appendChild(indicatorSummary);

            const indicatorMenu = document.createElement('div');
            indicatorMenu.className = 'tv-indicator-menu';
            const indicatorSearch = document.createElement('input');
            indicatorSearch.type = 'search';
            indicatorSearch.className = 'tv-indicator-search';
            indicatorSearch.placeholder = 'Search indicators';
            indicatorSearch.autocomplete = 'off';
            const indicatorOptions = document.createElement('div');
            indicatorOptions.className = 'tv-indicator-options';

            // Ensure the warn badge element exists on the summary
            let indicatorWarnBadge = indicatorSummary.querySelector('.tv-indicator-warn-badge');
            if (!indicatorWarnBadge) {
                indicatorWarnBadge = document.createElement('span');
                indicatorWarnBadge.className = 'tv-indicator-warn-badge';
                indicatorWarnBadge.title = 'Some indicators may not have enough history from Schwab API for full warmup at this timeframe';
                indicatorWarnBadge.textContent = '⚠';
                indicatorSummary.appendChild(indicatorWarnBadge);
            }

            function syncIndicatorSummary() {
                const count = tvActiveInds.size;
                indicatorBadge.textContent = String(count);
                indicatorBadge.style.display = count ? 'inline-flex' : 'none';
                const warned = getUnderwarnedIndicators();
                indicatorWarnBadge.style.display = warned.length ? 'inline' : 'none';
            }

            function renderIndicatorOptions() {
                const query = (indicatorSearch.value || '').trim().toLowerCase();
                indicatorOptions.innerHTML = '';
                const warnedSet = new Set(getUnderwarnedIndicators());
                const matches = indicatorDefs.filter(def =>
                    def.label.toLowerCase().includes(query) || def.title.toLowerCase().includes(query)
                );

                if (!matches.length) {
                    const empty = document.createElement('div');
                    empty.className = 'tv-indicator-option-empty';
                    empty.textContent = 'No matching indicators';
                    indicatorOptions.appendChild(empty);
                    return;
                }

                const HAS_SETTINGS = new Set(['smc','bb','fbb','rsi','macd','atr',
                    'sma9','sma20','sma50','sma100','sma200',
                    'ema9','ema21','ema50','ema100','ema200','wma20','wma50']);

                matches.forEach(def => {
                    const isWarned = tvActiveInds.has(def.key) && warnedSet.has(def.key);
                    const row = document.createElement('div');
                    row.className = 'tv-indicator-option-row' + (isWarned ? ' warmup-warn' : '');

                    const option = document.createElement('button');
                    option.type = 'button';
                    option.className = 'tv-indicator-option';
                    if (tvActiveInds.has(def.key)) option.classList.add('active');

                    const name = document.createElement('span');
                    name.className = 'tv-indicator-option-name';
                    name.textContent = def.label;
                    const desc = document.createElement('span');
                    desc.className = 'tv-indicator-option-desc';
                    desc.textContent = def.title;

                    option.appendChild(name);
                    option.appendChild(desc);
                    option.addEventListener('click', () => {
                        const wasActive = tvActiveInds.has(def.key);
                        const prevWarmup = computeRequiredWarmup();
                        if (wasActive) tvActiveInds.delete(def.key);
                        else           tvActiveInds.add(def.key);
                        syncIndicatorSummary();
                        renderIndicatorOptions();
                        if (!wasActive && computeRequiredWarmup() > prevWarmup) {
                            // Newly enabled indicator needs more history — re-fetch
                            fetchPriceHistory(true);
                        } else {
                            applyIndicators(tvIndicatorCandles, tvActiveInds);
                        }
                    });

                    row.appendChild(option);

                    if (HAS_SETTINGS.has(def.key)) {
                        const gear = document.createElement('button');
                        gear.type = 'button';
                        gear.className = 'tv-indicator-settings-btn';
                        gear.textContent = '⚙';
                        gear.title = 'Settings for ' + def.label;
                        gear.addEventListener('click', (e) => {
                            e.stopPropagation();
                            openIndicatorSettings(def.key, def.label, gear);
                        });
                        row.appendChild(gear);
                    }

                    indicatorOptions.appendChild(row);
                });
            }

            indicatorSearch.addEventListener('input', renderIndicatorOptions);
            indicatorPicker.addEventListener('toggle', () => {
                if (indicatorPicker.open) {
                    setTimeout(() => {
                        indicatorSearch.focus();
                        indicatorSearch.select();
                    }, 0);
                } else {
                    indicatorSearch.value = '';
                    renderIndicatorOptions();
                }
            });

            indicatorMenu.appendChild(indicatorSearch);
            indicatorMenu.appendChild(indicatorOptions);
            indicatorPicker.appendChild(indicatorMenu);
            syncIndicatorSummary();
            renderIndicatorOptions();
            toolbar.appendChild(indicatorPicker);

            const heikinAshiInput = document.getElementById('use_heikin_ashi');
            if (heikinAshiInput) {
                const chartSettingsPicker = document.createElement('details');
                chartSettingsPicker.className = 'tv-indicator-picker';
                const chartSettingsSummary = document.createElement('summary');
                chartSettingsSummary.className = 'tv-tb-btn tv-indicator-summary';
                chartSettingsSummary.title = 'Chart display settings';
                const chartSettingsLabel = document.createElement('span');
                chartSettingsLabel.textContent = 'Chart';
                const chartSettingsBadge = document.createElement('span');
                chartSettingsBadge.className = 'tv-indicator-badge';
                chartSettingsSummary.appendChild(chartSettingsLabel);
                chartSettingsSummary.appendChild(chartSettingsBadge);
                chartSettingsPicker.appendChild(chartSettingsSummary);

                const chartSettingsMenu = document.createElement('div');
                chartSettingsMenu.className = 'tv-indicator-menu tv-settings-menu';
                const heikinAshiRow = document.createElement('label');
                heikinAshiRow.className = 'tv-settings-row';
                heikinAshiRow.title = 'Render the price chart with Heikin-Ashi candles';
                const heikinAshiToggle = document.createElement('input');
                heikinAshiToggle.type = 'checkbox';
                const heikinAshiCopy = document.createElement('div');
                heikinAshiCopy.className = 'tv-settings-copy';
                const heikinAshiName = document.createElement('span');
                heikinAshiName.className = 'tv-settings-name';
                heikinAshiName.textContent = 'Heikin-Ashi';
                const heikinAshiDesc = document.createElement('span');
                heikinAshiDesc.className = 'tv-settings-desc';
                heikinAshiDesc.textContent = 'Use smoothed Heikin-Ashi candles in the price chart.';
                heikinAshiCopy.appendChild(heikinAshiName);
                heikinAshiCopy.appendChild(heikinAshiDesc);
                heikinAshiRow.appendChild(heikinAshiToggle);
                heikinAshiRow.appendChild(heikinAshiCopy);
                chartSettingsMenu.appendChild(heikinAshiRow);
                chartSettingsPicker.appendChild(chartSettingsMenu);

                function syncChartSettingsSummary() {
                    heikinAshiToggle.checked = !!heikinAshiInput.checked;
                    chartSettingsBadge.textContent = 'HA';
                    chartSettingsBadge.style.display = heikinAshiToggle.checked ? 'inline-flex' : 'none';
                    chartSettingsSummary.title = heikinAshiToggle.checked
                        ? 'Chart display settings · Heikin-Ashi on'
                        : 'Chart display settings';
                }

                chartSettingsPicker.addEventListener('toggle', () => {
                    if (chartSettingsPicker.open) {
                        syncChartSettingsSummary();
                    }
                });
                heikinAshiToggle.addEventListener('change', () => {
                    if (heikinAshiInput.checked !== heikinAshiToggle.checked) {
                        heikinAshiInput.checked = heikinAshiToggle.checked;
                        heikinAshiInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    syncChartSettingsSummary();
                    chartSettingsPicker.open = false;
                });

                syncChartSettingsSummary();
                toolbar.appendChild(chartSettingsPicker);
            }

            const levelDefs = Array.from(document.querySelectorAll('.levels-option input[type="checkbox"]')).map(input => {
                const label = document.querySelector(`label[for="${input.id}"]`);
                return {
                    value: input.value,
                    label: label ? label.textContent : input.value,
                    input,
                };
            });
            const hiddenLevelsCount = document.getElementById('levels_count');
            const levelsPicker = document.createElement('details');
            levelsPicker.className = 'tv-indicator-picker';
            const levelsSummary = document.createElement('summary');
            levelsSummary.className = 'tv-tb-btn tv-indicator-summary';
            levelsSummary.title = 'Select price levels shown on the TradingView chart';
            const levelsLabel = document.createElement('span');
            levelsLabel.textContent = 'Levels';
            const levelsBadge = document.createElement('span');
            levelsBadge.className = 'tv-indicator-badge';
            levelsSummary.appendChild(levelsLabel);
            levelsSummary.appendChild(levelsBadge);
            levelsPicker.appendChild(levelsSummary);

            const levelsMenu = document.createElement('div');
            levelsMenu.className = 'tv-indicator-menu';
            const levelsSearch = document.createElement('input');
            levelsSearch.type = 'search';
            levelsSearch.className = 'tv-indicator-search';
            levelsSearch.placeholder = 'Search price levels';
            levelsSearch.autocomplete = 'off';
            const levelsOptions = document.createElement('div');
            levelsOptions.className = 'tv-indicator-options';
            const levelsFooter = document.createElement('div');
            levelsFooter.style.cssText = 'display:grid;gap:8px;margin-top:8px;padding-top:8px;border-top:1px solid var(--border-color);';
            const levelsCountRow = document.createElement('div');
            levelsCountRow.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:8px;';
            const levelsCountLabel = document.createElement('label');
            levelsCountLabel.textContent = 'Top #';
            levelsCountLabel.style.cssText = 'font-size:11px;color:var(--text-secondary);';
            const levelsCountInput = document.createElement('input');
            levelsCountInput.type = 'number';
            levelsCountInput.min = '1';
            levelsCountInput.max = '10';
            levelsCountInput.value = hiddenLevelsCount ? hiddenLevelsCount.value : '3';
            levelsCountInput.style.cssText = 'width:58px;border:1px solid var(--border-color);border-radius:6px;background:var(--panel-bg-strong);color:var(--text-primary);padding:6px 8px;font-size:12px;';
            levelsCountRow.appendChild(levelsCountLabel);
            levelsCountRow.appendChild(levelsCountInput);
            const latestLevelLinesLabel = document.createElement('label');
            latestLevelLinesLabel.htmlFor = 'show_latest_level_lines';
            latestLevelLinesLabel.title = 'Draw a short line and label from the latest intraday level bubble to the right side of the price chart';
            latestLevelLinesLabel.style.cssText = 'display:flex;align-items:center;gap:7px;font-size:11px;color:var(--text-secondary);cursor:pointer;';
            const latestLevelLinesInput = document.createElement('input');
            latestLevelLinesInput.type = 'checkbox';
            latestLevelLinesInput.id = 'show_latest_level_lines';
            latestLevelLinesInput.checked = getShowLatestLevelLinesSetting();
            const latestLevelLinesText = document.createElement('span');
            latestLevelLinesText.textContent = 'Latest Level Lines';
            latestLevelLinesLabel.appendChild(latestLevelLinesInput);
            latestLevelLinesLabel.appendChild(latestLevelLinesText);
                const priceAboveBubblesLabel = document.createElement('label');
                priceAboveBubblesLabel.htmlFor = 'price_above_bubbles';
                priceAboveBubblesLabel.title = 'Redraw the visible candles above intraday level bubbles so price stays readable';
                priceAboveBubblesLabel.style.cssText = 'display:flex;align-items:center;gap:7px;font-size:11px;color:var(--text-secondary);cursor:pointer;';
                const priceAboveBubblesInput = document.createElement('input');
                priceAboveBubblesInput.type = 'checkbox';
                priceAboveBubblesInput.id = 'price_above_bubbles';
                priceAboveBubblesInput.checked = getPriceAboveBubblesSetting();
                const priceAboveBubblesText = document.createElement('span');
                priceAboveBubblesText.textContent = 'Price On Top';
                priceAboveBubblesLabel.appendChild(priceAboveBubblesInput);
                priceAboveBubblesLabel.appendChild(priceAboveBubblesText);
            levelsFooter.appendChild(levelsCountRow);
            levelsFooter.appendChild(latestLevelLinesLabel);
                levelsFooter.appendChild(priceAboveBubblesLabel);

            function getSelectedLevels() {
                return levelDefs.filter(def => def.input.checked).map(def => def.value);
            }

            function syncLevelsSummary() {
                const count = getSelectedLevels().length;
                const lineState = getShowLatestLevelLinesSetting() ? 'Lines on' : 'Lines off';
                    const layerState = getPriceAboveBubblesSetting() ? 'Price on top' : 'Bubbles on top';
                levelsBadge.textContent = String(count);
                levelsBadge.style.display = count ? 'inline-flex' : 'none';
                levelsSummary.title = count
                        ? `${count} price level${count === 1 ? '' : 's'} selected · Top ${levelsCountInput.value} · ${lineState} · ${layerState}`
                        : `Select price levels shown on the TradingView chart · ${lineState} · ${layerState}`;
            }

            function renderLevelsOptions() {
                const query = (levelsSearch.value || '').trim().toLowerCase();
                levelsOptions.innerHTML = '';
                const matches = levelDefs.filter(def =>
                    def.label.toLowerCase().includes(query) || def.value.toLowerCase().includes(query)
                );

                if (!matches.length) {
                    const empty = document.createElement('div');
                    empty.className = 'tv-indicator-option-empty';
                    empty.textContent = 'No matching price levels';
                    levelsOptions.appendChild(empty);
                    return;
                }

                matches.forEach(def => {
                    const option = document.createElement('button');
                    option.type = 'button';
                    option.className = 'tv-indicator-option';
                    if (def.input.checked) option.classList.add('active');

                    const name = document.createElement('span');
                    name.className = 'tv-indicator-option-name';
                    name.textContent = def.label;
                    const desc = document.createElement('span');
                    desc.className = 'tv-indicator-option-desc';
                    desc.textContent = `Overlay ${def.value} levels on price`; 

                    option.appendChild(name);
                    option.appendChild(desc);
                    option.addEventListener('click', () => {
                        def.input.checked = !def.input.checked;
                        def.input.dispatchEvent(new Event('change', { bubbles: true }));
                        syncLevelsSummary();
                        renderLevelsOptions();
                    });

                    levelsOptions.appendChild(option);
                });
            }

            levelsSearch.addEventListener('input', renderLevelsOptions);
            levelsPicker.addEventListener('toggle', () => {
                if (levelsPicker.open) {
                    setTimeout(() => {
                        levelsSearch.focus();
                        levelsSearch.select();
                    }, 0);
                } else {
                    levelsSearch.value = '';
                    renderLevelsOptions();
                }
            });
            levelsCountInput.addEventListener('input', () => {
                let nextValue = parseInt(levelsCountInput.value, 10);
                if (!Number.isFinite(nextValue)) return;
                nextValue = Math.max(1, Math.min(10, nextValue));
                levelsCountInput.value = String(nextValue);
                if (hiddenLevelsCount && hiddenLevelsCount.value !== String(nextValue)) {
                    hiddenLevelsCount.value = String(nextValue);
                    hiddenLevelsCount.dispatchEvent(new Event('input', { bubbles: true }));
                }
                syncLevelsSummary();
            });
            latestLevelLinesInput.addEventListener('change', () => {
                setShowLatestLevelLinesSetting(latestLevelLinesInput.checked);
                syncLevelsSummary();
                if (typeof pushDataToPopout === 'function') {
                    pushDataToPopout('price-chart');
                }
            });
                priceAboveBubblesInput.addEventListener('change', () => {
                    setPriceAboveBubblesSetting(priceAboveBubblesInput.checked);
                    syncLevelsSummary();
                });

            levelsMenu.appendChild(levelsSearch);
            levelsMenu.appendChild(levelsOptions);
            levelsMenu.appendChild(levelsFooter);
            levelsPicker.appendChild(levelsMenu);
            syncLevelsSummary();
            renderLevelsOptions();
            toolbar.appendChild(levelsPicker);

            // --- Separator ---
            const sep2 = document.createElement('div'); sep2.className = 'tv-toolbar-sep'; toolbar.appendChild(sep2);

            // Drawing tools
            const drawDefs = [
                { key:'hline',     label:'— H-Line', title:'Draw horizontal price line (single click)' },
                { key:'trendline', label:'↗ Trend',  title:'Draw trend line (click start, click end)' },
                { key:'rect',      label:'▭ Box',    title:'Draw rectangle between two prices (click two points)' },
                { key:'text',      label:'T Label',  title:'Add price label (click to place)' },
            ];
            drawDefs.forEach(def => {
                const b = btn(def.label, def.title, () => setDrawMode(def.key));
                b.dataset.draw = def.key;
                if (tvDrawMode === def.key) b.classList.add('active');
                toolbar.appendChild(b);
            });

            // Draw color picker
            const colorWrap = document.createElement('span');
            colorWrap.style.cssText = 'display:flex;align-items:center;gap:3px;';
            const colorLabel = document.createElement('span');
            colorLabel.style.cssText = 'font-size:10px;color:#aaa;';
            colorLabel.textContent = '🎨';
            const colorPicker = document.createElement('input');
            colorPicker.type = 'color';
            colorPicker.id = 'tv-draw-color';
            colorPicker.value = '#FFD700';
            colorPicker.style.cssText = 'width:24px;height:22px;border:none;background:none;cursor:pointer;padding:0;';
            colorPicker.title = 'Drawing color';
            colorWrap.appendChild(colorLabel);
            colorWrap.appendChild(colorPicker);
            toolbar.appendChild(colorWrap);

            // --- Separator ---
            const sep3 = document.createElement('div'); sep3.className = 'tv-toolbar-sep'; toolbar.appendChild(sep3);

            // Undo / Clear
            toolbar.appendChild(btn('↩ Undo', 'Undo last drawing', tvUndoDrawing));
            toolbar.appendChild(btn('✕ Clear', 'Clear all drawings', tvClearDrawings, 'danger'));

            // Push Fit / Auto-Range to far right
            const spacer = document.createElement('div');
            spacer.style.cssText = 'flex:1';
            toolbar.appendChild(spacer);

            // Auto-Range toggle
            const arBtn = document.createElement('button');
            arBtn.className = 'tv-tb-btn' + (tvAutoRange ? ' active' : '');
            arBtn.title = 'Auto-Range: when ON, the chart fits all candles on every data update. When OFF, your zoom & pan are preserved.';
            arBtn.textContent = tvAutoRange ? '⤢ Auto-Range ON' : '⤢ Auto-Range OFF';
            arBtn.addEventListener('click', () => {
                tvAutoRange = !tvAutoRange;
                arBtn.textContent = tvAutoRange ? '⤢ Auto-Range ON' : '⤢ Auto-Range OFF';
                arBtn.classList.toggle('active', tvAutoRange);
                if (tvPriceChart) tvFitAll();  // always fit immediately when toggling, ON or OFF
            });
            toolbar.appendChild(arBtn);

            toolbar.appendChild(btn('⟳ Reset', 'Reset zoom and pan to fit all data', () => tvFitAll()));

            // Candle close timer
            const timerEl = document.createElement('span');
            timerEl.id = 'candle-close-timer';
            timerEl.className = 'candle-close-timer';
            timerEl.title = 'Time remaining until the current candle closes';
            timerEl.textContent = '⏱ --:--';
            toolbar.appendChild(timerEl);
            startCandleCloseTimer();

            // Wire up click handler for drawing
            if (tvPriceChart) {
                tvPriceChart.subscribeClick(tvHandleChartClick);
            }
        }

        // ── Indicator Settings Panel ──────────────────────────────────────────────
        function getIndicatorParamDefs(key) {
            const map = {
                smc:    [
                    { key:'swing_len',  label:'Swing Length',       type:'number',   default:5,    min:2, max:20 },
                    { key:'show_ob',    label:'Order Blocks',        type:'checkbox', default:true  },
                    { key:'show_fvg',   label:'Fair Value Gaps',     type:'checkbox', default:true  },
                    { key:'show_bos',   label:'BOS / CHoCH',         type:'checkbox', default:true  },
                    { key:'show_swings',label:'Swing Points',        type:'checkbox', default:true  },
                    { key:'show_eqhl',  label:'Equal Highs/Lows',    type:'checkbox', default:true  },
                ],
                bb:     [{ key:'period', label:'Period', type:'number', default:20, min:2,  max:500 },
                         { key:'mult',   label:'Std Dev', type:'number', default:2,  min:0.1,max:10, step:0.1 }],
                fbb:    [{ key:'length', label:'Length',  type:'number', default:200,min:10, max:500 },
                         { key:'mult',   label:'Mult',    type:'number', default:3,  min:0.1,max:10, step:0.1 }],
                rsi:    [{ key:'period', label:'Period', type:'number', default:14, min:2,  max:100 }],
                macd:   [{ key:'fast',   label:'Fast',   type:'number', default:12, min:2,  max:200 },
                         { key:'slow',   label:'Slow',   type:'number', default:26, min:2,  max:500 },
                         { key:'signal', label:'Signal', type:'number', default:9,  min:2,  max:100 }],
                atr:    [{ key:'period', label:'Period', type:'number', default:14, min:2,  max:200 }],
                sma9:   [{ key:'period', label:'Period', type:'number', default:9,   min:1, max:500 }],
                sma20:  [{ key:'period', label:'Period', type:'number', default:20,  min:1, max:500 }],
                sma50:  [{ key:'period', label:'Period', type:'number', default:50,  min:1, max:500 }],
                sma100: [{ key:'period', label:'Period', type:'number', default:100, min:1, max:500 }],
                sma200: [{ key:'period', label:'Period', type:'number', default:200, min:1, max:500 }],
                ema9:   [{ key:'period', label:'Period', type:'number', default:9,   min:1, max:500 }],
                ema21:  [{ key:'period', label:'Period', type:'number', default:21,  min:1, max:500 }],
                ema50:  [{ key:'period', label:'Period', type:'number', default:50,  min:1, max:500 }],
                ema100: [{ key:'period', label:'Period', type:'number', default:100, min:1, max:500 }],
                ema200: [{ key:'period', label:'Period', type:'number', default:200, min:1, max:500 }],
                wma20:  [{ key:'period', label:'Period', type:'number', default:20,  min:1, max:500 }],
                wma50:  [{ key:'period', label:'Period', type:'number', default:50,  min:1, max:500 }],
            };
            return map[key] || [];
        }

        function openIndicatorSettings(key, label, triggerEl) {
            // Toggle off if same key panel already open
            const existingPanel = document.querySelector('.tv-ind-settings-panel');
            if (existingPanel) {
                const wasThisKey = existingPanel.dataset.key === key;
                existingPanel.remove();
                if (wasThisKey) return;
            }

            const params = tvIndicatorParams[key] || {};
            const panel = document.createElement('div');
            panel.className = 'tv-ind-settings-panel';
            panel.dataset.key = key;

            const header = document.createElement('div');
            header.className = 'tv-ind-settings-header';
            const title = document.createElement('span');
            title.textContent = label + ' Settings';
            const closeBtn = document.createElement('button');
            closeBtn.className = 'tv-ind-settings-close';
            closeBtn.textContent = '✕';
            closeBtn.addEventListener('click', () => panel.remove());
            header.appendChild(title);
            header.appendChild(closeBtn);
            panel.appendChild(header);

            const body = document.createElement('div');
            body.className = 'tv-ind-settings-body';

            const apiNote = document.createElement('div');
            apiNote.style.cssText = 'font-size:10px;color:var(--text-muted);padding:2px 2px 6px;border-bottom:1px solid var(--border-color);margin-bottom:4px;';
            apiNote.textContent = `Schwab API: up to ${tvMaxIndicatorBars} bars available at current timeframe (10 trading days).`;
            body.appendChild(apiNote);

            getIndicatorParamDefs(key).forEach(pd => {
                const row = document.createElement('div');
                row.className = 'tv-ind-param-row';
                const lbl = document.createElement('label');
                lbl.className = 'tv-ind-param-label';
                lbl.textContent = pd.label;

                let input;
                if (pd.type === 'checkbox') {
                    input = document.createElement('input');
                    input.type = 'checkbox';
                    input.checked = params[pd.key] !== undefined ? !!params[pd.key] : pd.default !== false;
                } else if (pd.type === 'color') {
                    input = document.createElement('input');
                    input.type = 'color';
                    input.value = params[pd.key] || pd.default;
                } else {
                    input = document.createElement('input');
                    input.type = 'number';
                    input.min = pd.min || 1;
                    input.max = pd.max || 999;
                    input.step = pd.step || 1;
                    input.value = params[pd.key] !== undefined ? params[pd.key] : pd.default;
                }

                input.addEventListener('change', () => {
                    if (!tvIndicatorParams[key]) tvIndicatorParams[key] = {};
                    const prevWarmup = computeRequiredWarmup();
                    if (pd.type === 'checkbox')  tvIndicatorParams[key][pd.key] = input.checked;
                    else if (pd.type === 'color') tvIndicatorParams[key][pd.key] = input.value;
                    else                          tvIndicatorParams[key][pd.key] = Number(input.value);
                    if (tvActiveInds.has(key)) {
                        const newWarmup = computeRequiredWarmup();
                        if (newWarmup > prevWarmup) {
                            fetchPriceHistory(true);
                        } else {
                            applyIndicators(tvIndicatorCandles, tvActiveInds);
                        }
                    }
                });

                row.appendChild(lbl);
                row.appendChild(input);
                body.appendChild(row);
            });

            panel.appendChild(body);
            document.body.appendChild(panel);

            // Position the panel near the trigger element (gear button), clamped to viewport
            const PANEL_W = 300;
            if (triggerEl) {
                const r = triggerEl.getBoundingClientRect();
                let left = r.left;
                let top = r.bottom + 6;
                // Clamp so panel stays within viewport
                if (left + PANEL_W > window.innerWidth - 8) left = window.innerWidth - PANEL_W - 8;
                if (left < 8) left = 8;
                if (top + 200 > window.innerHeight) top = r.top - 10; // flip above if near bottom
                panel.style.left = left + 'px';
                panel.style.top = top + 'px';
            } else {
                panel.style.left = Math.max(8, (window.innerWidth - PANEL_W) / 2) + 'px';
                panel.style.top = '80px';
            }

            // Close when clicking outside
            function onOutsideClick(e) {
                if (!panel.contains(e.target) && e.target !== triggerEl) {
                    panel.remove();
                    document.removeEventListener('mousedown', onOutsideClick, true);
                }
            }
            setTimeout(() => document.addEventListener('mousedown', onOutsideClick, true), 0);
        }

        function ensureTVHistoricalOverlay() {
            const container = document.getElementById('price-chart');
            if (!container) return null;
            let overlay = container.querySelector('.tv-historical-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.className = 'tv-historical-overlay';
                container.appendChild(overlay);
            }
            return overlay;
        }

        function ensureTVHistoricalCanvas() {
            const overlay = ensureTVHistoricalOverlay();
            if (!overlay) return null;
            let canvas = overlay.querySelector('.tv-historical-canvas');
            if (!canvas) {
                canvas = document.createElement('canvas');
                canvas.className = 'tv-historical-canvas';
                overlay.appendChild(canvas);
            }
            return canvas;
        }

        function syncTVHistoricalCanvas(canvas, overlay) {
            if (!canvas || !overlay) return null;
            const pixelRatio = window.devicePixelRatio || 1;
            const width = Math.max(1, Math.round(overlay.clientWidth));
            const height = Math.max(1, Math.round(overlay.clientHeight));
            const pixelWidth = Math.max(1, Math.round(width * pixelRatio));
            const pixelHeight = Math.max(1, Math.round(height * pixelRatio));

            if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
                canvas.width = pixelWidth;
                canvas.height = pixelHeight;
                canvas.style.width = `${width}px`;
                canvas.style.height = `${height}px`;
            }

            const context = canvas.getContext('2d');
            if (!context) return null;
            context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
            context.clearRect(0, 0, width, height);
            return { context, width, height };
        }

        function ensureTVHistoricalTooltip() {
            const container = document.getElementById('price-chart');
            if (!container) return null;
            let tooltip = container.querySelector('.tv-historical-tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.className = 'tv-historical-tooltip';
                container.appendChild(tooltip);
            }
            return tooltip;
        }

        function formatTVBubbleTime(timestamp) {
            return new Date(timestamp * 1000).toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                hour12: false,
                timeZone: 'America/New_York'
            }) + ' ET';
        }

        function buildTVHistoricalTooltipHtml(point) {
            const dotColor = point.border_color || point.color || '#ffffff';
            const name = point.kind === 'expected-move'
                ? point.label + ' ' + point.side
                : point.label + ' ' + point.side.charAt(0);
            const value = point.kind === 'expected-move'
                ? point.value
                : '$' + Number(point.price).toFixed(2) + '  ' + point.value;
            return '<div class="tt-row">'
                + '<span class="tt-dot" style="background:' + dotColor + '"></span>'
                + '<div class="tt-main">'
                + '<span class="tt-name">' + name + '</span>'
                + '<span class="tt-value">' + value + '</span>'
                + '</div>'
                + '</div>';
        }

        function buildTVSnapshotBubblePoints(priceData) {
            const candles = Array.isArray(priceData.candles) && priceData.candles.length
                ? priceData.candles
                : tvLastCandles;
            const latestCandle = candles && candles.length ? candles[candles.length - 1] : null;
            const anchorTime = latestCandle ? latestCandle.time : null;
            if (!anchorTime) return [];

            const snapshotPoints = [];
            const exposureLevels = Array.isArray(priceData.exposure_levels) ? priceData.exposure_levels : [];
            // Levels arrive sorted by magnitude within each type; rank per type so the
            // keys line up with the historical bubbles (which are ranked 1..N per type).
            const rankByType = {};
            exposureLevels.forEach((level, index) => {
                const price = Number(level.price);
                if (!Number.isFinite(price)) return;
                const rawValue = Number(level.value);
                const label = level.type || 'Level';
                const formattedValue = level.label
                    ? String(level.label).replace(/^.*?:\\s*/, '')
                    : formatLargeNumberCompact(rawValue);
                snapshotPoints.push({
                    time: anchorTime,
                    price,
                    size: level.is_max ? 16 : 12,
                    color: level.color || (rawValue >= 0 ? priceData.call_color : priceData.put_color) || '#ffffff',
                    border_color: level.is_max ? maxLevelColor : (rawValue >= 0 ? priceData.call_color : priceData.put_color) || level.color || '#ffffff',
                    border_width: level.is_max ? 2 : Math.max(1, Number(level.line_width) || 1),
                    label,
                    rank: (rankByType[label] = (rankByType[label] || 0) + 1),
                    side: rawValue >= 0 ? 'Call' : 'Put',
                    value: formattedValue,
                    kind: 'exposure',
                });
            });

            const expectedMoves = Array.isArray(priceData.expected_moves) ? priceData.expected_moves : [];
            expectedMoves.forEach((move, index) => {
                const upper = Number(move.upper);
                const lower = Number(move.lower);
                const moveValue = Number.isFinite(upper) && Number.isFinite(lower)
                    ? '$' + (Math.abs(upper - lower) / 2).toFixed(2)
                    : '';
                [
                    ['Upper', upper],
                    ['Lower', lower],
                ].forEach(([side, price]) => {
                    if (!Number.isFinite(price)) return;
                    snapshotPoints.push({
                        time: anchorTime,
                        price,
                        size: 13,
                        color: 'rgba(3,107,252,0.82)',
                        border_color: '#81b4ff',
                        border_width: 1,
                        label: 'Expected Move',
                        rank: exposureLevels.length + index + 1,
                        side,
                        value: moveValue,
                        kind: 'expected-move',
                    });
                });
            });

            return snapshotPoints;
        }

        function getTVBubblePointKey(point) {
            return [
                point.time || '',
                Number(point.price).toFixed(4),
                point.kind || 'exposure',
                point.label || '',
                point.side || '',
            ].join('|');
        }

        function applyTVBubbleOverlapMetadata(points) {
            const groups = new Map();
            points.forEach(point => {
                const key = [
                    point.time || '',
                    Number(point.price).toFixed(4),
                    point.kind || 'exposure',
                ].join('|');
                if (!groups.has(key)) groups.set(key, []);
                groups.get(key).push(point);
            });
            groups.forEach(group => {
                group.forEach((point, index) => {
                    point.overlap_slot = index;
                    point.overlap_count = group.length;
                });
            });
            return points;
        }

        function getTVPriceLevelBubblePoints(priceData) {
            const historicalPoints = Array.isArray(priceData.historical_exposure_levels)
                ? priceData.historical_exposure_levels
                    .filter(point => point && point.time && Number.isFinite(Number(point.price)))
                    .map(point => ({ ...point }))
                : [];
            const snapshotPoints = buildTVSnapshotBubblePoints(priceData);
            if (!historicalPoints.length) return applyTVBubbleOverlapMetadata(snapshotPoints);
            if (!snapshotPoints.length) return applyTVBubbleOverlapMetadata(historicalPoints);

            const seenKeys = new Set(historicalPoints.map(point => getTVBubblePointKey(point)));
            snapshotPoints.forEach(point => {
                const key = getTVBubblePointKey(point);
                if (!seenKeys.has(key)) {
                    seenKeys.add(key);
                    historicalPoints.push(point);
                }
            });
            return applyTVBubbleOverlapMetadata(historicalPoints);
        }

        function getTVLatestLevelKey(point) {
            if ((point.kind || '') === 'expected-move') {
                return ['expected-move', point.label || '', point.side || ''].join('|');
            }
            return ['exposure', point.label || '', point.rank || ''].join('|');
        }

        function formatTVLatestLevelLineText(point) {
            if ((point.kind || '') === 'expected-move') {
                const moveSide = point.side ? ' ' + point.side : '';
                return `${point.label || 'Expected Move'}${moveSide}: ${point.value || ''}`;
            }
            return `${point.label || 'Level'}: ${point.value || ''}`;
        }

        function buildTVLatestLevelLabelSignature(points) {
            if (!tvShowLatestLevelLines || !points.length) return '';
            return points.map(point => [
                Number(point.price),
                point.border_color || point.color || '#ffffff',
                formatTVLatestLevelLineText(point),
            ].join('|')).join('||');
        }

        function clearTVLatestLevelNativeLabels() {
            tvLatestLevelLabelSignature = '';
            if (tvCandleSeries && tvLatestLevelLabelPriceLines.length) {
                tvLatestLevelLabelPriceLines.forEach(line => {
                    try { tvCandleSeries.removePriceLine(line); } catch(e) {}
                });
            }
            tvLatestLevelLabelPriceLines = [];
        }

        function setTVLatestLevelNativeLabels(points) {
            const signature = buildTVLatestLevelLabelSignature(points);
            if (signature && signature === tvLatestLevelLabelSignature) return;
            clearTVLatestLevelNativeLabels();
            if (!tvShowLatestLevelLines || !tvCandleSeries || !points.length) return;
            tvLatestLevelLabelSignature = signature;

            points.forEach(point => {
                const price = Number(point.price);
                if (!Number.isFinite(price)) return;
                const labelLine = tvCandleSeries.createPriceLine({
                    price,
                    color: point.border_color || point.color || '#ffffff',
                    lineWidth: 1, lineVisible: false,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: formatTVLatestLevelLineText(point),
                });
                tvLatestLevelLabelPriceLines.push(labelLine);
            });
        }

        function drawTVLatestLevelLines(context, width, height) {
            if (!tvShowLatestLevelLines || !tvHistoricalRenderedPoints.length) {
                clearTVLatestLevelNativeLabels();
                return;
            }

            const latestByKey = new Map();
            tvHistoricalRenderedPoints.forEach(point => {
                if (!point || point.x == null || point.y == null) return;
                const key = getTVLatestLevelKey(point);
                const existing = latestByKey.get(key);
                if (!existing || point.time > existing.time || (point.time === existing.time && point.x > existing.x)) {
                    latestByKey.set(key, point);
                }
            });

            const latestPoints = Array.from(latestByKey.values()).sort((left, right) => left.y - right.y);
            if (!latestPoints.length) {
                clearTVLatestLevelNativeLabels();
                return;
            }

            setTVLatestLevelNativeLabels(latestPoints);

            context.save();
            latestPoints.forEach(point => {
                const color = point.border_color || point.color || '#ffffff';
                const radius = (point.size || 8) / 2;
                const lineStartX = Math.max(0, point.x + radius + 3);
                const lineEndX = Math.max(lineStartX, width - 2);
                if (lineEndX <= lineStartX) return;

                context.save();
                context.strokeStyle = color;
                context.lineWidth = point.border_width || 1;
                context.globalAlpha = 0.92;
                context.setLineDash([4, 4]);
                context.beginPath();
                context.moveTo(lineStartX, point.y);
                context.lineTo(lineEndX, point.y);
                context.stroke();
                context.restore();
            });

            context.restore();
        }

        function positionTVHistoricalTooltip(tooltip, event) {
            const container = document.getElementById('price-chart');
            if (!tooltip || !container || !event) return;
            const bounds = container.getBoundingClientRect();
            const offsetX = 12;
            const offsetY = 12;
            const left = Math.min(
                Math.max(8, event.clientX - bounds.left + offsetX),
                Math.max(8, bounds.width - tooltip.offsetWidth - 8)
            );
            const top = Math.min(
                Math.max(8, event.clientY - bounds.top + offsetY),
                Math.max(8, bounds.height - tooltip.offsetHeight - 8)
            );
            tooltip.style.left = `${left}px`;
            tooltip.style.top = `${top}px`;
        }

        function indexTVHistoricalHoverPoint(point) {
            const hoverRadius = Math.max(8, (point.size || 8) / 2 + 5);
            const minBucketX = Math.floor((point.x - hoverRadius) / tvHistoricalHoverBucketSize);
            const maxBucketX = Math.floor((point.x + hoverRadius) / tvHistoricalHoverBucketSize);
            const minBucketY = Math.floor((point.y - hoverRadius) / tvHistoricalHoverBucketSize);
            const maxBucketY = Math.floor((point.y + hoverRadius) / tvHistoricalHoverBucketSize);

            for (let bucketX = minBucketX; bucketX <= maxBucketX; bucketX += 1) {
                for (let bucketY = minBucketY; bucketY <= maxBucketY; bucketY += 1) {
                    const key = `${bucketX}:${bucketY}`;
                    let bucket = tvHistoricalHoverBuckets.get(key);
                    if (!bucket) {
                        bucket = [];
                        tvHistoricalHoverBuckets.set(key, bucket);
                    }
                    bucket.push(point);
                }
            }
        }

        function getTVHistoricalHoverCandidates(cursorX, cursorY) {
            if (!tvHistoricalHoverBuckets.size) return tvHistoricalRenderedPoints;

            const minBucketX = Math.floor((cursorX - 32) / tvHistoricalHoverBucketSize);
            const maxBucketX = Math.floor((cursorX + 32) / tvHistoricalHoverBucketSize);
            const minBucketY = Math.floor((cursorY - 32) / tvHistoricalHoverBucketSize);
            const maxBucketY = Math.floor((cursorY + 32) / tvHistoricalHoverBucketSize);
            const seen = new Set();
            const candidates = [];

            for (let bucketX = minBucketX; bucketX <= maxBucketX; bucketX += 1) {
                for (let bucketY = minBucketY; bucketY <= maxBucketY; bucketY += 1) {
                    const bucket = tvHistoricalHoverBuckets.get(`${bucketX}:${bucketY}`);
                    if (!bucket) continue;
                    bucket.forEach(point => {
                        if (seen.has(point)) return;
                        seen.add(point);
                        candidates.push(point);
                    });
                }
            }

            return candidates;
        }

        function findTVHistoricalHoverPoints(event) {
            const container = document.getElementById('price-chart');
            if (!container || !tvHistoricalRenderedPoints.length) return [];
            const bounds = container.getBoundingClientRect();
            const cursorX = event.clientX - bounds.left;
            const cursorY = event.clientY - bounds.top;
            return getTVHistoricalHoverCandidates(cursorX, cursorY)
                .filter(point => {
                    const dx = cursorX - point.x;
                    const dy = cursorY - point.y;
                    const radius = Math.max(8, (point.size || 8) / 2 + 5);
                    return (dx * dx + dy * dy) <= (radius * radius);
                })
                .sort((left, right) => {
                    const leftDist = (cursorX - left.x) ** 2 + (cursorY - left.y) ** 2;
                    const rightDist = (cursorX - right.x) ** 2 + (cursorY - right.y) ** 2;
                    return leftDist - rightDist;
                });
        }

        function updateTVHistoricalTooltip(event) {
            const tooltip = ensureTVHistoricalTooltip();
            if (!tooltip) return;
            if (event && event.buttons) {
                tooltip.style.display = 'none';
                return;
            }
            const hoverPoints = findTVHistoricalHoverPoints(event);
            if (!hoverPoints.length) {
                tooltip.style.display = 'none';
                return;
            }

            const topPoints = hoverPoints.slice(0, 5);
            const anchorTime = topPoints[0].time;
            tooltip.innerHTML = '<div class="tt-head"><span class="tt-badge">' + hoverPoints.length + ' bubble' + (hoverPoints.length === 1 ? '' : 's') + '</span><div class="tt-time">' + formatTVBubbleTime(anchorTime) + '</div></div>'
                + '<div class="tt-list">' + topPoints.map(point => buildTVHistoricalTooltipHtml(point)).join('') + '</div>'
                + (hoverPoints.length > topPoints.length
                    ? '<div class="tt-more">+' + (hoverPoints.length - topPoints.length) + ' more</div>'
                    : '');
            tooltip.style.display = 'block';
            positionTVHistoricalTooltip(tooltip, event);
        }

        function clearTVHistoricalExpectedMoveSeries() {
            if (!tvPriceChart || !tvHistoricalExpectedMoveSeries.length) return;
            tvHistoricalExpectedMoveSeries.forEach(series => {
                try { tvPriceChart.removeSeries(series); } catch(e) {}
            });
            tvHistoricalExpectedMoveSeries = [];
        }

        function getVisibleTVHistoricalPoints() {
            if (!tvHistoricalPoints.length) return [];

            let visiblePoints = tvHistoricalPoints;
            try {
                const visibleRange = tvPriceChart.timeScale().getVisibleLogicalRange();
                if (visibleRange && tvLastCandles.length) {
                    const leftIndex = Math.max(0, Math.floor(visibleRange.from) - 2);
                    const rightIndex = Math.min(tvLastCandles.length - 1, Math.ceil(visibleRange.to) + 2);
                    const leftCandle = tvLastCandles[leftIndex];
                    const rightCandle = tvLastCandles[rightIndex];
                    if (leftCandle && rightCandle) {
                        const candleSpan = tvLastCandles.length > 1
                            ? Math.max(60, tvLastCandles[1].time - tvLastCandles[0].time)
                            : 60;
                        const minTime = leftCandle.time - (candleSpan * 2);
                        const maxTime = rightCandle.time + (candleSpan * 2);
                        visiblePoints = tvHistoricalPoints.filter(point => point.time >= minTime && point.time <= maxTime);
                    }
                }
            } catch(e) {}

            if (visiblePoints.length <= tvHistoricalOverlayMaxVisible) {
                return visiblePoints;
            }

            const priorityPoints = [];
            const secondaryPoints = [];
            visiblePoints.forEach(point => {
                if (point.kind === 'expected-move' || point.rank === 1) priorityPoints.push(point);
                else secondaryPoints.push(point);
            });

            if (priorityPoints.length >= tvHistoricalOverlayMaxVisible) {
                const stride = Math.ceil(priorityPoints.length / tvHistoricalOverlayMaxVisible);
                return priorityPoints.filter((_, index) => index % stride === 0);
            }

            const secondarySlots = Math.max(0, tvHistoricalOverlayMaxVisible - priorityPoints.length);
            if (!secondaryPoints.length || secondarySlots === 0) {
                return priorityPoints;
            }

            const stride = Math.ceil(secondaryPoints.length / secondarySlots);
            return priorityPoints.concat(secondaryPoints.filter((_, index) => index % stride === 0));
        }

        function getVisibleTVOverlayCandles() {
            if (!tvLastCandles.length || !tvPriceChart) return [];

            let visibleCandles = tvLastCandles;
            try {
                const visibleRange = tvPriceChart.timeScale().getVisibleLogicalRange();
                if (visibleRange) {
                    const leftIndex = Math.max(0, Math.floor(visibleRange.from) - 2);
                    const rightIndex = Math.min(tvLastCandles.length - 1, Math.ceil(visibleRange.to) + 2);
                    visibleCandles = tvLastCandles.slice(leftIndex, rightIndex + 1);
                }
            } catch(e) {}

            return visibleCandles;
        }

        function drawTVPriceAboveBubbles(context, width) {
            if (!getPriceAboveBubblesSetting() || !tvPriceChart || !tvCandleSeries || !tvLastCandles.length) {
                return;
            }

            const candlePoints = getVisibleTVOverlayCandles()
                .map(candle => ({
                    candle,
                    x: tvPriceChart.timeScale().timeToCoordinate(candle.time),
                }))
                .filter(point => point.x != null && !Number.isNaN(point.x) && point.x >= -24 && point.x <= width + 24)
                .sort((left, right) => left.x - right.x);

            if (!candlePoints.length) {
                return;
            }

            let minDelta = Number.POSITIVE_INFINITY;
            for (let index = 1; index < candlePoints.length; index += 1) {
                const delta = candlePoints[index].x - candlePoints[index - 1].x;
                if (delta > 0 && delta < minDelta) {
                    minDelta = delta;
                }
            }

            const bodyWidth = Number.isFinite(minDelta)
                ? Math.max(3, Math.min(14, Math.round(minDelta * 0.68)))
                : 5;
            const halfWidth = bodyWidth / 2;
            const upColor = (tvLastPriceData && tvLastPriceData.call_color) || '#00FF00';
            const downColor = (tvLastPriceData && tvLastPriceData.put_color) || '#FF0000';

            context.save();
            candlePoints.forEach(({ candle, x }) => {
                const openY = tvCandleSeries.priceToCoordinate(candle.open);
                const highY = tvCandleSeries.priceToCoordinate(candle.high);
                const lowY = tvCandleSeries.priceToCoordinate(candle.low);
                const closeY = tvCandleSeries.priceToCoordinate(candle.close);
                if ([openY, highY, lowY, closeY].some(value => value == null || Number.isNaN(value))) {
                    return;
                }

                const color = candle.close >= candle.open ? upColor : downColor;
                const wickTop = Math.min(highY, lowY);
                const wickBottom = Math.max(highY, lowY);
                const bodyTop = Math.min(openY, closeY);
                const bodyBottom = Math.max(openY, closeY);
                const bodyHeight = Math.max(1, Math.round(bodyBottom - bodyTop));
                const left = Math.round(x - halfWidth);
                const centerX = Math.round(x) + 0.5;

                context.strokeStyle = color;
                context.fillStyle = color;
                context.lineWidth = 1;
                context.beginPath();
                context.moveTo(centerX, wickTop);
                context.lineTo(centerX, wickBottom);
                context.stroke();
                context.fillRect(left, Math.round(bodyTop), bodyWidth, bodyHeight);
            });
            context.restore();
        }

        function drawTVHistoricalOverlay() {
            const overlay = ensureTVHistoricalOverlay();
            const canvas = ensureTVHistoricalCanvas();
            const tooltip = ensureTVHistoricalTooltip();
            if (!overlay || !canvas || !tvPriceChart || !tvCandleSeries) return;

            tvHistoricalRenderedPoints = [];
            tvHistoricalHoverBuckets = new Map();
            if (!tvHistoricalPoints.length) {
                clearTVLatestLevelNativeLabels();
                overlay.style.display = 'none';
                if (tooltip) tooltip.style.display = 'none';
                return;
            }

            overlay.style.display = 'block';
            const canvasState = syncTVHistoricalCanvas(canvas, overlay);
            if (!canvasState) {
                clearTVLatestLevelNativeLabels();
                overlay.style.display = 'none';
                if (tooltip) tooltip.style.display = 'none';
                return;
            }
            const { context, width, height } = canvasState;

            const pointsToRender = getVisibleTVHistoricalPoints();
            if (!pointsToRender.length) {
                clearTVLatestLevelNativeLabels();
                overlay.style.display = 'none';
                if (tooltip) tooltip.style.display = 'none';
                return;
            }

            let visibleCount = 0;
            // Keep bubbles inside the plotting pane (not over the price/time axes)
            let paneWidth = width, paneHeight = height;
            try {
                paneWidth = tvPriceChart.timeScale().width() || width;
                paneHeight = Math.max(0, height - (tvPriceChart.timeScale().height() || 0));
            } catch (e) {}
            context.save();
            context.beginPath();
            context.rect(0, 0, paneWidth, paneHeight);
            context.clip();
            for (const point of pointsToRender) {
                const x = tvPriceChart.timeScale().timeToCoordinate(point.time);
                const y = tvCandleSeries.priceToCoordinate(point.price);
                if (x == null || y == null || Number.isNaN(x) || Number.isNaN(y)) continue;
                const size = point.size || 8;
                const radius = size / 2;
                const overlapCount = Math.max(1, point.overlap_count || 1);
                const overlapSlot = Math.max(0, Math.min(overlapCount - 1, point.overlap_slot || 0));
                const offsetStep = Math.max(4, Math.min(10, radius * 0.9));
                const offsetX = overlapCount > 1
                    ? (overlapSlot - ((overlapCount - 1) / 2)) * offsetStep
                    : 0;
                const drawX = x + offsetX;
                const hoverRadius = Math.max(8, radius + 5);
                if (drawX < -hoverRadius || drawX > paneWidth + hoverRadius || y < -hoverRadius || y > paneHeight + hoverRadius) {
                    continue;
                }

                context.save();
                context.globalAlpha = 0.95;
                context.fillStyle = point.color || 'rgba(255,255,255,0.6)';
                context.beginPath();
                context.arc(drawX, y, radius, 0, Math.PI * 2);
                context.fill();
                context.globalAlpha = 1;
                context.strokeStyle = 'rgba(0,0,0,0.25)';
                context.lineWidth = 1;
                context.beginPath();
                context.arc(drawX, y, radius + 1, 0, Math.PI * 2);
                context.stroke();
                context.strokeStyle = point.border_color || point.color || '#ffffff';
                context.lineWidth = point.border_width || 1;
                context.beginPath();
                context.arc(drawX, y, Math.max(0.5, radius - ((point.border_width || 1) / 2)), 0, Math.PI * 2);
                context.stroke();
                context.restore();

                const renderedPoint = { ...point, x: drawX, y };
                tvHistoricalRenderedPoints.push(renderedPoint);
                indexTVHistoricalHoverPoint(renderedPoint);
                visibleCount += 1;
            }
            context.restore();

            if (visibleCount > 0) {
                drawTVLatestLevelLines(context, paneWidth, paneHeight);
                drawTVPriceAboveBubbles(context, width);
            } else {
                clearTVLatestLevelNativeLabels();
            }

            overlay.style.display = visibleCount > 0 ? 'block' : 'none';
        }

        function scheduleTVHistoricalOverlayDraw() {
            if (tvHistoricalOverlayPending) return;
            tvHistoricalOverlayPending = true;
            requestAnimationFrame(() => {
                tvHistoricalOverlayPending = false;
                drawTVHistoricalOverlay();
            });
        }

        function renderTVPriceChart(priceData) {
            const container = document.getElementById('price-chart');
            if (!container) return;

            tvLastPriceData = priceData;
                if (priceData && priceData.price_above_bubbles !== undefined) {
                    setPriceAboveBubblesSetting(priceData.price_above_bubbles, false);
                }
            const upColor   = priceData.call_color || '#00FF00';
            const downColor = priceData.put_color  || '#FF0000';
            const candles   = priceData.candles || [];

            const lineStyleMap = {
                dashed:       LightweightCharts.LineStyle.Dashed,
                dotted:       LightweightCharts.LineStyle.Dotted,
                large_dashed: LightweightCharts.LineStyle.LargeDashed,
            };

            // ── First render: create the chart and all series once ────────────
            if (!tvPriceChart) {
                // Remove any leftover overlays
                container.querySelectorAll('.tv-chart-title, .tv-indicator-legend').forEach(el => el.remove());
                const _tc = document.getElementById('tv-toolbar-container');
                if (_tc) _tc.innerHTML = '';

                tvPriceChart = LightweightCharts.createChart(container, Object.assign({}, buildLightweightThemeOptions(), {
                    autoSize: true,
                    rightPriceScale: {
                        borderColor:  getThemeValue('--border-color', '#333333'),
                        scaleMargins: { top: 0.04, bottom: 0.15 },
                        minimumWidth: tvRightScaleMinimumWidth,
                    },
                    localization: {
                        timeFormatter: (time) => {
                            const d = new Date(time * 1000);
                            return d.toLocaleTimeString('en-US', {
                                hour: '2-digit', minute: '2-digit',
                                hour12: false, timeZone: 'America/New_York'
                            });
                        }
                    },
                    timeScale: {
                        borderColor:      getThemeValue('--border-color', '#333333'),
                        timeVisible:      true,
                        secondsVisible:   false,
                        fixLeftEdge:      false,
                        fixRightEdge:     false,
                        tickMarkFormatter: (time) => {
                            const d = new Date(time * 1000);
                            return d.toLocaleTimeString('en-US', {
                                hour: '2-digit', minute: '2-digit',
                                hour12: false, timeZone: 'America/New_York'
                            });
                        }
                    },
                    handleScale:  { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
                    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
                }));

                tvCandleSeries = tvPriceChart.addCandlestickSeries({
                    upColor, downColor,
                    borderVisible: false,
                    wickUpColor:   upColor,
                    wickDownColor: downColor,
                });

                tvVolumeSeries = tvPriceChart.addHistogramSeries({
                    priceFormat:  { type: 'volume' },
                    priceScaleId: 'volume',
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
                tvPriceChart.priceScale('volume').applyOptions({
                    scaleMargins: { top: 0.88, bottom: 0 },
                });

                // Toolbar + title (only built once)
                buildTVToolbar(container, candles, upColor, downColor);
                ensureTVHistoricalOverlay();
                ensureSmcOverlay();
                tvPriceChart.timeScale().subscribeVisibleLogicalRangeChange(() => {
                    scheduleTVHistoricalOverlayDraw();
                    if (tvSmcData) scheduleSmcDraw();
                });
                if (!tvHistoricalOverlayDomEventsBound) {
                    tvHistoricalOverlayDomEventsBound = true;
                    container.addEventListener('wheel', () => {
                        scheduleTVHistoricalOverlayDraw();
                        if (tvSmcData) scheduleSmcDraw();
                    }, { passive: true });
                    container.addEventListener('mouseup', () => {
                        scheduleTVHistoricalOverlayDraw();
                        if (tvSmcData) scheduleSmcDraw();
                    });
                    container.addEventListener('touchend', () => {
                        scheduleTVHistoricalOverlayDraw();
                        if (tvSmcData) scheduleSmcDraw();
                    }, { passive: true });
                    container.addEventListener('mousemove', (event) => {
                        updateTVHistoricalTooltip(event);
                        // Dragging the price/time axis rescales without a range-change event
                        if (event.buttons) {
                            scheduleTVHistoricalOverlayDraw();
                            if (tvSmcData) scheduleSmcDraw();
                        }
                    });
                    container.addEventListener('mouseleave', () => {
                        const tooltip = ensureTVHistoricalTooltip();
                        if (tooltip) tooltip.style.display = 'none';
                    });
                }
                const _tc2 = document.getElementById('tv-toolbar-container');
                if (_tc2) {
                    const titleEl = document.createElement('div');
                    titleEl.className = 'tv-chart-title';
                    titleEl.textContent = priceData.use_heikin_ashi ? 'Price Chart (Heikin-Ashi)' : 'Price Chart';
                    _tc2.insertBefore(titleEl, _tc2.firstChild);
                }

                // ── OHLC hover tooltip ────────────────────────────────────
                container.querySelectorAll('.tv-ohlc-tooltip').forEach(el => el.remove());
                const _tip = document.createElement('div');
                _tip.className = 'tv-ohlc-tooltip';
                _tip.id = 'tv-ohlc-tooltip';
                container.appendChild(_tip);

                tvPriceChart.subscribeCrosshairMove(function(param) {
                    const tip = document.getElementById('tv-ohlc-tooltip');
                    if (!tip) return;
                    if (!param || !param.time || !param.seriesData) {
                        tip.style.display = 'none'; return;
                    }
                    const bar = param.seriesData.get(tvCandleSeries);
                    if (!bar) { tip.style.display = 'none'; return; }
                    const d = new Date(param.time * 1000);
                    const timeStr = d.toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/New_York'}) + ' ET';
                    const isUp = bar.close >= bar.open;
                    const cls = isUp ? 'tt-up' : 'tt-dn';
                    const chg = bar.open !== 0 ? ((bar.close - bar.open) / bar.open * 100).toFixed(2) : '0.00';
                    const fmt = v => v != null ? v.toFixed(2) : '--';
                    const fmtVol = v => v >= 1e6 ? (v/1e6).toFixed(2)+'M' : v >= 1e3 ? (v/1e3).toFixed(0)+'K' : (v||0).toString();
                    tip.innerHTML =
                        '<div class="tt-time">'+timeStr+'</div>'
                        +'<span class="'+cls+'">'
                        +'O <b>'+fmt(bar.open)+'</b>  '
                        +'H <b>'+fmt(bar.high)+'</b>  '
                        +'L <b>'+fmt(bar.low)+'</b>  '
                        +'C <b>'+fmt(bar.close)+'</b>  '
                        +(chg>=0?'+':'')+chg+'%'
                        +'</span>'
                        +'<br><span style="color:'+getThemeValue('--text-muted', '#888')+'">Vol <b>'+fmtVol(bar.volume)+'</b></span>';
                    tip.style.display = 'block';
                });

                if (tvDrawingDefs.length) {
                    tvRestoreDrawings();
                }
            }

            // ── Every render: update data and overlays in place ───────────────

            // Update candle colors in case they changed
            try { tvPriceChart.applyOptions(buildLightweightThemeOptions()); } catch (e) {}
            tvCandleSeries.applyOptions({
                upColor, downColor,
                wickUpColor: upColor, wickDownColor: downColor,
            });

            const isFirstRender = !tvLastCandles.length;

            tvCandleSeries.setData(candles);
            tvLastCandles = candles;
            tvStreamBucketState = null;
            tvVolumeSeries.setData(priceData.volume || []);
            // Use multi-day candles for indicator warmup so SMA200, EMA200, etc. start from day open
            tvIndicatorCandles = (priceData.indicator_candles && priceData.indicator_candles.length > 0)
                ? priceData.indicator_candles : candles.map(c => ({ ...c }));
            tvCurrentDayStartTime = priceData.current_day_start_time || 0;
            tvMaxIndicatorBars = priceData.max_indicator_bars || 3900;

            // Start/maintain real-time streaming for the current ticker
            const streamTicker = (document.getElementById('ticker').value || '').trim();
            if (streamTicker && isStreaming) {
                connectPriceStream(streamTicker);
            }

            // Remove old dynamic price lines before rebuilding current snapshot and historical overlays.
            tvExposurePriceLines.forEach(l => { try { tvCandleSeries.removePriceLine(l); } catch(e){} });
            tvExposurePriceLines = [];
            tvExpectedMovePriceLines.forEach(l => { try { tvCandleSeries.removePriceLine(l); } catch(e){} });
            tvExpectedMovePriceLines = [];
            clearTVHistoricalExpectedMoveSeries();

            tvAllLevelPrices = [];
            setShowLatestLevelLinesSetting(priceData.show_latest_level_lines !== false);

            tvHistoricalPoints = getTVPriceLevelBubblePoints(priceData);
            // Autoscale to the current levels only; including every historical bubble of
            // the session stretches the axis over the whole strike range and squashes candles.
            const latestBubbleTime = tvHistoricalPoints.reduce((mx, point) => Math.max(mx, Number(point.time) || 0), 0);
            tvHistoricalPoints
                .filter(point => Number(point.time) === latestBubbleTime)
                .forEach(point => tvAllLevelPrices.push(point.price));
            tvDrawingDefs.forEach(def => {
                if ((def.type === 'hline' || def.type === 'text') && Number.isFinite(def.price)) {
                    tvAllLevelPrices.push(def.price);
                }
            });
            scheduleTVHistoricalOverlayDraw();

            tvApplyAutoscale();
            if (tvActiveInds.size > 0) applyIndicators(tvIndicatorCandles, tvActiveInds);

            // fitContent on first render, when auto-range is ON, or when explicitly forced (ticker change)
            if (tvAutoRange || isFirstRender || tvForceFit) {
                const _chart = tvPriceChart;
                setTimeout(() => {
                    try {
                        _chart.timeScale().fitContent();
                        _chart.priceScale('right').applyOptions({ autoScale: true });
                        tvApplyAutoscale();
                        if (tvRsiChart)  tvRsiChart.priceScale('right').applyOptions({ autoScale: true });
                        if (tvMacdChart) tvMacdChart.priceScale('right').applyOptions({ autoScale: true });
                        if (tvArvChart)  tvArvChart.priceScale('right').applyOptions({ autoScale: true });
                        scheduleTVHistoricalOverlayDraw();
                    } catch(e) {}
                }, 50);
                tvForceFit = false;
            }
        }
        // ─────────────────────────────────────────────────────────────────────

        // Standalone price chart renderer — called by /update_price without touching other charts.
        function applyPriceData(priceJson) {
            if (!document.getElementById('price').checked) return;
            let priceContainer = document.querySelector('.price-chart-container');
            if (!priceContainer) {
                priceContainer = document.createElement('div');
                priceContainer.className = 'price-chart-container';
                const toolbarContainer = document.createElement('div');
                toolbarContainer.className = 'tv-toolbar-container';
                toolbarContainer.id = 'tv-toolbar-container';
                const chartDiv = document.createElement('div');
                chartDiv.className = 'chart-container';
                chartDiv.id = 'price-chart';
                const rsiPane = document.createElement('div');
                rsiPane.className = 'tv-sub-pane'; rsiPane.id = 'rsi-pane'; rsiPane.style.display = 'none';
                rsiPane.innerHTML = '<div class="tv-sub-pane-header">RSI 14</div><div id="rsi-chart" style="height:110px"></div>';
                const macdPane = document.createElement('div');
                macdPane.className = 'tv-sub-pane'; macdPane.id = 'macd-pane'; macdPane.style.display = 'none';
                macdPane.innerHTML = '<div class="tv-sub-pane-header">MACD (12,26,9)</div><div id="macd-chart" style="height:120px"></div>';
                const arvPane = document.createElement('div');
                arvPane.className = 'tv-sub-pane'; arvPane.id = 'arv-pane'; arvPane.style.display = 'none';
                arvPane.innerHTML = '<div class="tv-sub-pane-header">ARV 20 (Ann. %)</div><div id="arv-chart" style="height:110px"></div>';
                priceContainer.appendChild(toolbarContainer);
                priceContainer.appendChild(chartDiv);
                priceContainer.appendChild(rsiPane);
                priceContainer.appendChild(macdPane);
                priceContainer.appendChild(arvPane);
                document.getElementById('chart-grid').insertBefore(priceContainer, document.getElementById('chart-grid').firstChild);
            }
            priceContainer.style.display = 'block';
            const parsed = typeof priceJson === 'string' ? JSON.parse(priceJson) : priceJson;
            if (!parsed.error) {
                parsed.price_above_bubbles = getPriceAboveBubblesSetting();
                lastPriceData = parsed;
                renderTVPriceChart(parsed);
                pushDataToPopout('price-chart');
            }
        }

        // ── Throttled price history fetcher ───────────────────────────────────
        // Fetches candle history + exposure levels from /update_price.
        // Real-time ticks come from SSE; this only handles the historical snapshot
        // and exposure level overlays, so it runs at most every 30 seconds unless
        // forced (ticker change or visible settings change).
        let _priceHistoryLastMs = 0;
        let _priceHistoryLastKey = '';
        let _priceHistoryInFlight = false;
        let _priceHistoryPendingRefresh = false;
        let _priceHistoryDesiredKey = '';

        function computeRequiredWarmup() {
            // Determine the minimum number of historical bars needed for all active indicators.
            // The backend uses this to fetch enough multi-day history for proper warmup,
            // but is always capped at what Schwab's API can actually deliver (10 trading days).
            function p(key, field, def) { return (tvIndicatorParams[key]||{})[field] ?? def; }
            let bars = 220; // hard floor — always enough for SMA200 / FBB200
            const active = tvActiveInds;
            if (active.has('sma9'))   bars = Math.max(bars, p('sma9','period',9)   + 20);
            if (active.has('sma20'))  bars = Math.max(bars, p('sma20','period',20) + 20);
            if (active.has('sma50'))  bars = Math.max(bars, p('sma50','period',50) + 20);
            if (active.has('sma100')) bars = Math.max(bars, p('sma100','period',100) + 20);
            if (active.has('sma200')) bars = Math.max(bars, p('sma200','period',200) + 20);
            if (active.has('ema9'))   bars = Math.max(bars, p('ema9','period',9)   * 5);
            if (active.has('ema21'))  bars = Math.max(bars, p('ema21','period',21) * 5);
            if (active.has('ema50'))  bars = Math.max(bars, p('ema50','period',50) * 5);
            if (active.has('ema100')) bars = Math.max(bars, p('ema100','period',100) * 5);
            if (active.has('ema200')) bars = Math.max(bars, p('ema200','period',200) * 5);
            if (active.has('wma20'))  bars = Math.max(bars, p('wma20','period',20)  + 20);
            if (active.has('wma50'))  bars = Math.max(bars, p('wma50','period',50)  + 20);
            if (active.has('bb'))     bars = Math.max(bars, p('bb','period',20)     + 20);
            if (active.has('fbb'))    bars = Math.max(bars, p('fbb','length',200)   + 20);
            if (active.has('rsi'))    bars = Math.max(bars, p('rsi','period',14)    * 5);
            if (active.has('macd'))   bars = Math.max(bars, (p('macd','slow',26) + p('macd','signal',9)) * 3);
            if (active.has('atr'))    bars = Math.max(bars, p('atr','period',14)    + 20);
            if (active.has('smc')) {
                const swingLen = p('smc','swing_len',5);
                bars = Math.max(bars, swingLen * 40, 300);
            }
            // Never request more than the Schwab API can deliver for this timeframe
            return Math.min(tvMaxIndicatorBars, Math.ceil(bars));
        }

        function getUnderwarnedIndicators() {
            // Returns labels of active indicators whose warmup need exceeds what the API provides.
            const available = tvIndicatorCandles.length || tvMaxIndicatorBars;
            function p(key, field, def) { return (tvIndicatorParams[key]||{})[field] ?? def; }
            const warnings = [];
            const checks = {
                sma9:   k => p(k,'period',9)   + 20,
                sma20:  k => p(k,'period',20)  + 20,
                sma50:  k => p(k,'period',50)  + 20,
                sma100: k => p(k,'period',100) + 20,
                sma200: k => p(k,'period',200) + 20,
                ema9:   k => p(k,'period',9)   * 5,
                ema21:  k => p(k,'period',21)  * 5,
                ema50:  k => p(k,'period',50)  * 5,
                ema100: k => p(k,'period',100) * 5,
                ema200: k => p(k,'period',200) * 5,
                wma20:  k => p(k,'period',20)  + 20,
                wma50:  k => p(k,'period',50)  + 20,
                bb:     k => p(k,'period',20)  + 20,
                fbb:    k => p(k,'length',200) + 20,
                rsi:    k => p(k,'period',14)  * 5,
                macd:   k => (p(k,'slow',26) + p(k,'signal',9)) * 3,
                atr:    k => p(k,'period',14)  + 20,
                smc:    k => Math.max(p(k,'swing_len',5) * 40, 300),
            };
            tvActiveInds.forEach(key => {
                if (checks[key] && checks[key](key) > available) warnings.push(key);
            });
            return warnings;
        }

        function buildPricePayload() {
            const expiry = Array.from(document.querySelectorAll('.expiry-option input[type="checkbox"]:checked')).map(cb => cb.value);
            return {
                ticker: document.getElementById('ticker').value,
                expiry,
                timeframe: document.getElementById('timeframe').value,
                call_color: callColor,
                put_color: putColor,
                levels_types: getSelectedPriceLevelTypes(),
                levels_count: parseInt(document.getElementById('levels_count').value),
                use_heikin_ashi: document.getElementById('use_heikin_ashi').checked,
                strike_range: parseFloat(document.getElementById('strike_range').value) / 100,
                highlight_max_level: document.getElementById('highlight_max_level').checked,
                show_latest_level_lines: getShowLatestLevelLinesSetting(),
                max_level_color: maxLevelColor,
                coloring_mode: document.getElementById('coloring_mode').value,
                exposure_metric: document.getElementById('exposure_metric').value,
                delta_adjusted: document.getElementById('delta_adjusted_exposures').checked,
                calculate_in_notional: document.getElementById('calculate_in_notional').checked,
                indicator_warmup: computeRequiredWarmup(),
            };
        }

        function fetchPriceHistory(force) {
            if (!document.getElementById('price').checked) return;
            const payload = buildPricePayload();
            const key = JSON.stringify(payload);
            _priceHistoryDesiredKey = key;
            if (_priceHistoryInFlight) {
                _priceHistoryPendingRefresh = _priceHistoryPendingRefresh || !!force || key !== _priceHistoryLastKey;
                return;
            }
            const now = Date.now();
            // Skip if nothing changed and it's been less than 30 seconds
            if (!force && key === _priceHistoryLastKey && now - _priceHistoryLastMs < 30000) return;
            _priceHistoryLastMs = now;
            _priceHistoryLastKey = key;
            _priceHistoryPendingRefresh = false;
            _priceHistoryInFlight = true;
            fetch('/update_price', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: key
            })
            .then(r => r.json())
            .then(priceResp => {
                const currentTicker = (document.getElementById('ticker').value || '').toUpperCase();
                if (key !== _priceHistoryDesiredKey || String(payload.ticker || '').toUpperCase() !== currentTicker) {
                    return;
                }
                if (!priceResp.error && priceResp.price) {
                    applyPriceData(priceResp.price);
                }
            })
            .catch(err => console.error('Error fetching price chart:', err))
            .finally(() => {
                _priceHistoryInFlight = false;
                if (_priceHistoryPendingRefresh || _priceHistoryDesiredKey !== _priceHistoryLastKey) {
                    _priceHistoryPendingRefresh = false;
                    fetchPriceHistory(true);
                }
            });
        }

        function updateCharts(data) {
            // Save scroll position before any DOM changes
            savedScrollPosition = window.scrollY || window.pageYOffset;
            
            const selectedCharts = {
                price: document.getElementById('price').checked,
                gamma: document.getElementById('gamma').checked,
                heatmap: document.getElementById('heatmap').checked,
                delta: document.getElementById('delta').checked,
                vanna: document.getElementById('vanna').checked,
                charm: document.getElementById('charm').checked,
                speed: document.getElementById('speed').checked,
                vomma: document.getElementById('vomma').checked,
                color: document.getElementById('color').checked,
                options_volume: document.getElementById('options_volume').checked,
                open_interest: document.getElementById('open_interest').checked,
                volume: document.getElementById('volume').checked,
                large_trades: document.getElementById('large_trades').checked,
                centroid: document.getElementById('centroid').checked
            };

            // Handle price chart separately (TradingView Lightweight Charts)
            if (selectedCharts.price && data.price) {
                let priceContainer = document.querySelector('.price-chart-container');
                if (!priceContainer) {
                    priceContainer = document.createElement('div');
                    priceContainer.className = 'price-chart-container';
                    const toolbarContainer = document.createElement('div');
                    toolbarContainer.className = 'tv-toolbar-container';
                    toolbarContainer.id = 'tv-toolbar-container';
                    const chartDiv = document.createElement('div');
                    chartDiv.className = 'chart-container';
                    chartDiv.id = 'price-chart';
                    // RSI sub-pane
                    const rsiPane = document.createElement('div');
                    rsiPane.className = 'tv-sub-pane'; rsiPane.id = 'rsi-pane'; rsiPane.style.display = 'none';
                    rsiPane.innerHTML = '<div class="tv-sub-pane-header">RSI 14</div><div id="rsi-chart" style="height:110px"></div>';
                    // MACD sub-pane
                    const macdPane = document.createElement('div');
                    macdPane.className = 'tv-sub-pane'; macdPane.id = 'macd-pane'; macdPane.style.display = 'none';
                    macdPane.innerHTML = '<div class="tv-sub-pane-header">MACD (12,26,9)</div><div id="macd-chart" style="height:120px"></div>';
                    // ARV sub-pane
                    const arvPane = document.createElement('div');
                    arvPane.className = 'tv-sub-pane'; arvPane.id = 'arv-pane'; arvPane.style.display = 'none';
                    arvPane.innerHTML = '<div class="tv-sub-pane-header">ARV 20 (Ann. %)</div><div id="arv-chart" style="height:110px"></div>';
                    priceContainer.appendChild(toolbarContainer);
                    priceContainer.appendChild(chartDiv);
                    priceContainer.appendChild(rsiPane);
                    priceContainer.appendChild(macdPane);
                    priceContainer.appendChild(arvPane);
                    document.getElementById('chart-grid').insertBefore(priceContainer, document.getElementById('chart-grid').firstChild);
                }
                priceContainer.style.display = 'block';

                const priceData = typeof data.price === 'string' ? JSON.parse(data.price) : data.price;
                if (!priceData.error) {
                    renderTVPriceChart(priceData);
                }
            } else if (!selectedCharts.price) {
                const priceContainer = document.querySelector('.price-chart-container');
                if (priceContainer) {
                    priceContainer.style.display = 'none';
                }
                destroyRsiPane();
                destroyMacdPane();
                destroyArvPane();
                if (tvPriceChart) {
                    try { tvPriceChart.unsubscribeClick(tvHandleChartClick); } catch(e){}
                    // Reset label/candle state so a recreated chart redraws labels and refits
                    clearTVLatestLevelNativeLabels();
                    tvLastCandles = [];
                    tvStreamBucketState = null;
                    tvPriceChart.remove();
                    tvPriceChart = null;
                    tvCandleSeries = null;
                    tvVolumeSeries = null;
                    tvIndicatorSeries = {};
                    tvHistoricalPoints = [];
                    tvHistoricalExpectedMoveSeries = [];
                }
                if (tvResizeObserver) {
                    tvResizeObserver.disconnect();
                    tvResizeObserver = null;
                }
            }
            
            // Handle other charts
            let chartsGrid = document.querySelector('.charts-grid');
            if (!chartsGrid) {
                chartsGrid = document.createElement('div');
                chartsGrid.className = 'charts-grid';
                document.getElementById('chart-grid').appendChild(chartsGrid);
            }
            
            // Check if we need to rebuild the grid (enabled charts changed)
            const currentChartIds = Array.from(chartsGrid.querySelectorAll('.chart-container')).map(el => el.id.replace('-chart', ''));
            
            // Count enabled regular charts (excluding price)
            const regularCharts = Object.entries(selectedCharts).filter(([key, selected]) => 
                selected && !['price'].includes(key) && data[key]
            );
            
            const regularChartIds = regularCharts.map(([key]) => key);
            const needsGridRebuild = regularChartIds.length !== currentChartIds.length ||
                                     !regularChartIds.every((id, i) => currentChartIds[i] === id);
            
            // Hide the charts grid if no regular charts are enabled
            if (regularCharts.length === 0) {
                chartsGrid.style.display = 'none';
                mountHeatmapControls(null);
                chartsGrid.querySelectorAll('.chart-container').forEach(forgetPlotlyContainer);
                chartsGrid.innerHTML = '';
            } else {
                chartsGrid.style.display = 'grid';
                
                // Only rebuild if chart selection changed
                if (needsGridRebuild) {
                    mountHeatmapControls(null);
                    // Free Plotly's listeners/WebGL state before dropping the nodes
                    chartsGrid.querySelectorAll('.chart-container').forEach(forgetPlotlyContainer);
                    chartsGrid.querySelectorAll('.js-plotly-plot').forEach(el => { try { Plotly.purge(el); } catch (e) {} });
                    chartsGrid.innerHTML = '';
                    chartsGrid.className = 'charts-grid';
                    
                    // Add appropriate class based on number of enabled charts
                    if (regularCharts.length === 1) {
                        chartsGrid.classList.add('one-chart');
                    } else if (regularCharts.length === 2) {
                        chartsGrid.classList.add('two-charts');
                    } else if (regularCharts.length === 3) {
                        chartsGrid.classList.add('three-charts');
                    } else if (regularCharts.length === 4) {
                        chartsGrid.classList.add('four-charts');
                    } else {
                        chartsGrid.classList.add('many-charts');
                    }
                    
                    regularCharts.forEach(([key, selected]) => {
                        const newContainer = document.createElement('div');
                        newContainer.className = 'chart-container';
                        newContainer.id = `${key}-chart`;
                        chartsGrid.appendChild(newContainer);
                        chartContainerCache[key] = newContainer;
                    });
                }
                
                // Update chart data
                regularCharts.forEach(([key, selected]) => {
                    let container = document.getElementById(`${key}-chart`);
                    if (!container) {
                        container = document.createElement('div');
                        container.className = 'chart-container';
                        container.id = `${key}-chart`;
                        chartsGrid.appendChild(container);
                    }
                    
                    try {
                        // Special handling for options chain (HTML table)
                        if (key === 'large_trades') {
                            // Only update if content changed (compare against the raw HTML
                            // we rendered; innerHTML serialization never matches it exactly)
                            if (container.__rawHtml !== data[key]) {
                                container.__rawHtml = data[key];
                                renderOptionsChainHtml(container, data[key]);
                            }
                        } else {
                            renderPlotlyChart(key, data[key]);
                        }
                    } catch (error) {
                        console.error(`Error rendering ${key} chart:`, error);
                    }
                });
            }
            
            // Clean up disabled regular charts from charts object
            Object.keys(selectedCharts).forEach(key => {
                if (!selectedCharts[key] && !['price'].includes(key)) {
                    const container = document.getElementById(`${key}-chart`);
                    if (container) {
                        if (key === 'heatmap') {
                            mountHeatmapControls(null);
                        }
                        forgetPlotlyContainer(container);
                        container.remove();
                    }
                    delete charts[key];
                    delete chartContainerCache[key];
                }
            });
            
            // Add fullscreen and popout buttons to all chart containers
            document.querySelectorAll('.chart-container').forEach(c => { addFullscreenButton(c); addPopoutButton(c); });

            // Push updated data to any open popout windows
            pushAllPopouts();

            // If a chart is currently fullscreen, ensure it resizes to fill viewport
            if (queryFullscreenCharts().length) {
                resizeChartsAfterFullscreen();
            }

            // Restore scroll position after DOM updates
            requestAnimationFrame(() => {
                window.scrollTo(0, savedScrollPosition);
            });
        }
        
        // Options chain table sorting. Lives here because <script> tags inserted via
        // innerHTML never execute; the sort is re-applied after each data refresh.
        let optionsChainSort = null;  // { columnIndex, dataType, direction }

        function sortOptionsChainTable(table) {
            const { columnIndex, dataType, direction } = optionsChainSort;
            const tbody = table.tBodies[0];
            const markers = Array.from(tbody.querySelectorAll('tr.oc-marker'));
            const rows = Array.from(tbody.rows).filter(row => !row.classList.contains('oc-marker'));
            const cellValue = row => {
                const cell = row.cells[columnIndex];
                if (dataType === 'number') {
                    const v = parseFloat(cell.getAttribute('data-sort') || cell.textContent.replace(/[$,%]/g, ''));
                    return isNaN(v) ? 0 : v;
                }
                return cell.textContent.toLowerCase();
            };
            rows.sort((a, b) => {
                const aVal = cellValue(a);
                const bVal = cellValue(b);
                const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
                return direction === 'asc' ? cmp : -cmp;
            });
            rows.forEach(row => tbody.appendChild(row));
            // The spot-price divider only makes sense when rows are ordered by strike
            if (markers.length) {
                const strikeCol = table.querySelector('th.k[data-col]');
                const byStrike = strikeCol && parseInt(strikeCol.dataset.col, 10) === columnIndex;
                const price = row => parseFloat(row.dataset.price);
                markers.sort((a, b) => direction === 'asc' ? price(a) - price(b) : price(b) - price(a));
                markers.forEach(marker => {
                    marker.style.display = byStrike ? '' : 'none';
                    if (!byStrike) return;
                    const next = rows.find(row => {
                        const k = parseFloat(row.cells[columnIndex].getAttribute('data-sort'));
                        return direction === 'asc' ? k > price(marker) : k < price(marker);
                    });
                    tbody.insertBefore(marker, next || null);
                });
            }
            table.querySelectorAll('th[data-col]').forEach(header => {
                const span = header.querySelector('.oc-sort');
                if (!span) return;
                const active = parseInt(header.dataset.col, 10) === columnIndex;
                span.textContent = active ? (direction === 'asc' ? '▲' : '▼') : '▼▲';
                span.classList.toggle('active', active);
            });
        }

        // Every expiry has its own table; keep them all on the same sort
        function applyOptionsChainSort() {
            if (!optionsChainSort) return;
            document.querySelectorAll('table.oc-table').forEach(sortOptionsChainTable);
        }

        // Selected expiry tab and each expiry's scroll position survive data refreshes
        let optionsChainExpiry = null;
        const optionsChainScroll = {};

        function rememberOptionsChainScroll(root) {
            root.querySelectorAll('.oc-section:not([hidden])').forEach(section => {
                const scroller = section.querySelector('.oc-scroll');
                if (scroller) optionsChainScroll[section.dataset.expiry] = scroller.scrollTop;
            });
        }

        function showOptionsChainSection(root, expiry) {
            const sections = Array.from(root.querySelectorAll('.oc-section'));
            if (!sections.length) return;
            const target = sections.find(section => section.dataset.expiry === expiry) || sections[0];
            sections.forEach(section => { section.hidden = section !== target; });
            root.querySelectorAll('.oc-tab').forEach(tab => {
                tab.classList.toggle('active', tab.dataset.expiry === target.dataset.expiry);
            });
            optionsChainExpiry = target.dataset.expiry;
            const scroller = target.querySelector('.oc-scroll');
            if (!scroller) return;
            const saved = optionsChainScroll[optionsChainExpiry];
            if (saved !== undefined) {
                scroller.scrollTop = saved;
                return;
            }
            // First time this expiry is shown: center it on the spot price
            const spotRow = scroller.querySelector('tr.oc-spot');
            if (spotRow && spotRow.style.display !== 'none') {
                scroller.scrollTop = Math.max(0, spotRow.offsetTop - scroller.clientHeight / 2);
            }
            optionsChainScroll[optionsChainExpiry] = scroller.scrollTop;
        }

        function renderOptionsChainHtml(container, html) {
            rememberOptionsChainScroll(container);
            container.innerHTML = html;
            applyOptionsChainSort();
            showOptionsChainSection(container, optionsChainExpiry);
        }

        window.selectChainExpiry = function(tab) {
            const root = tab.closest('.oc-wrap');
            if (!root) return;
            rememberOptionsChainScroll(root);
            showOptionsChainSection(root, tab.dataset.expiry);
        };

        window.sortTable = function(columnIndex, dataType) {
            const direction = optionsChainSort && optionsChainSort.columnIndex === columnIndex && optionsChainSort.direction === 'asc'
                ? 'desc' : 'asc';
            optionsChainSort = { columnIndex, dataType, direction };
            applyOptionsChainSort();
        };

        function escapeAttr(text) {
            return String(text)
                .replace(/&/g, '&amp;')
                .replace(/"/g, '&quot;')
                .replace(/</g, '&lt;');
        }

        function getEasternNow() {
            return new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
        }

        function formatETTime() {
            return new Intl.DateTimeFormat('en-US', {
                timeZone: 'America/New_York',
                hour: 'numeric',
                minute: '2-digit',
                hour12: true
            }).format(new Date()) + ' ET';
        }

        function getMarketSession() {
            const now = getEasternNow();
            const day = now.getDay();
            const mins = now.getHours() * 60 + now.getMinutes();
            const closeMins = 16 * 60;
            const timeToClose = closeMins - mins;
            const closeLabel = timeToClose > 0
                ? `${Math.floor(timeToClose / 60)}h ${String(timeToClose % 60).padStart(2, '0')}m to close`
                : '';

            if (day === 0 || day === 6) {
                return {
                    short: 'Weekend',
                    tip: 'US equity markets are closed Sat–Sun. Index options may still decay on calendar time; cash session resumes Monday 9:30 AM ET.'
                };
            }
            if (mins >= 4 * 60 && mins < 9 * 60 + 30) {
                return {
                    short: 'Pre-Market',
                    tip: 'Extended hours (4:00–9:30 AM ET). Lower liquidity and wider spreads — overnight news gets priced in before the open.'
                };
            }
            if (mins >= 9 * 60 + 30 && mins < 10 * 60) {
                return {
                    short: 'Market Open',
                    tip: `Opening 30 minutes (9:30–10:00 AM ET) — peak volume and volatility. Opening range often sets the day\u2019s tone. ${closeLabel}.`
                };
            }
            if (mins >= 10 * 60 && mins < 12 * 60) {
                return {
                    short: 'Morning',
                    tip: `Morning session (10:00 AM–12:00 PM ET). Institutional flow dominates; primary intraday trend often forms here. ${closeLabel}.`
                };
            }
            if (mins >= 12 * 60 && mins < 13 * 60 + 30) {
                return {
                    short: 'NYC Lunch',
                    tip: `Midday lull (12:00–1:30 PM ET). Volume typically drops 30–40%; ranges tighten and mean-reversion is common. ${closeLabel}.`
                };
            }
            if (mins >= 13 * 60 + 30 && mins < 15 * 60) {
                return {
                    short: 'Afternoon',
                    tip: `Afternoon session (1:30–3:00 PM ET). Volume picks back up; macro releases often land around 2:00 PM ET. ${closeLabel}.`
                };
            }
            if (mins >= 15 * 60 && mins < 16 * 60) {
                return {
                    short: 'Power Hour',
                    tip: `Final hour (3:00–4:00 PM ET). Elevated volume, MOC imbalances, and pin risk into the close. ${closeLabel}.`
                };
            }
            if (mins >= 16 * 60 && mins < 20 * 60) {
                return {
                    short: 'After Hours',
                    tip: 'After-hours (4:00–8:00 PM ET). Thin liquidity — earnings reactions and headline risk; most options are closed.'
                };
            }
            return {
                short: 'Overnight',
                tip: 'US cash equities closed (8:00 PM–4:00 AM ET). Futures trade overnight; regular session opens 9:30 AM ET.'
            };
        }

        function formatIvPct(value) {
            if (value == null || !Number.isFinite(Number(value))) return '—';
            return `${Number(value).toFixed(1)}%`;
        }

        function buildIvStatsHtml(info) {
            const iv = info && info.iv_stats;
            if (!iv || (iv.call_vw_iv == null && iv.put_vw_iv == null)) return '';

            const tip = `Volume-weighted IV across the selected chain. Calls ${formatIvPct(iv.call_vw_iv)}, Puts ${formatIvPct(iv.put_vw_iv)}. Contracts with more volume count more.`;
            return `<div class="price-info-item" title="${escapeAttr(tip)}"><strong>VW IV</strong><span><span style="color:${callColor}">${formatIvPct(iv.call_vw_iv)}</span>/<span style="color:${putColor}">${formatIvPct(iv.put_vw_iv)}</span></span></div>`;
        }

        function buildMarketContextItemsHtml() {
            const session = getMarketSession();
            const timeStr = formatETTime();
            return `<div class="price-info-market-group">
                <div class="price-info-item price-info-market" id="price-info-session" title="${escapeAttr(session.tip)}">
                    <strong>Session</strong>
                    <span>${session.short} · ${timeStr}</span>
                </div>
            </div>`;
        }

        function refreshMarketContextItems(info, displayPrice) {
            if (!info) return;
            const priceInfo = document.getElementById('price-info');
            if (!priceInfo) return;
            const price = Number.isFinite(displayPrice)
                ? displayPrice
                : ((livePrice !== null) ? livePrice : info.current_price);
            const marketGroup = priceInfo.querySelector('.price-info-market-group');
            if (marketGroup) marketGroup.remove();
            priceInfo.insertAdjacentHTML('beforeend', buildMarketContextItemsHtml());
        }

        function updatePriceInfo(info) {
            if (!info) return;
            lastPriceInfo = info;
            // If EM range lock is active, silently sync the slider without triggering a full re-fetch
            if (emRangeLocked && info.expected_move_range) {
                applyEmRange(info.expected_move_range, false);
            }
            const priceInfo = document.getElementById('price-info');
            const selectedExpiries = lastData.selected_expiries || [];
            const expiryText = selectedExpiries.length > 1 ? 
                `${selectedExpiries.length} expiries selected` : 
                selectedExpiries[0] || 'No expiry selected';

            let expectedMoveHtml = '';
            if (info.expected_move_range && info.expected_move_range.lower && info.expected_move_range.upper) {
                const lowPct  = info.expected_move_range.lower_pct != null ?
                    `${info.expected_move_range.lower_pct >= 0 ? '+' : ''}${info.expected_move_range.lower_pct}%` : '';
                const highPct = info.expected_move_range.upper_pct != null ?
                    `${info.expected_move_range.upper_pct >= 0 ? '+' : ''}${info.expected_move_range.upper_pct}%` : '';
                const emMove = info.expected_move_range.move || ((info.expected_move_range.upper - info.expected_move_range.lower) / 2);
                const emTip = `ATM straddle implied move: ±$${Number(emMove).toFixed(2)} (±${((emMove / info.current_price) * 100).toFixed(2)}%). Market\u2019s priced range for the selected expiry.`;
                // lower bound is below spot -> use putColor, upper bound above spot -> callColor
                const lowColor = putColor;
                const highColor = callColor;
                expectedMoveHtml = `<div class="price-info-item" title="${escapeAttr(emTip)}"><strong>Expected Move</strong><span><span style="color:${lowColor}">$${info.expected_move_range.lower.toFixed(2)} ${lowPct}</span> - <span style="color:${highColor}">$${info.expected_move_range.upper.toFixed(2)} ${highPct}</span></span></div>`;
            }

            // high/low diff coloring (use call/put colors)
            const highDiff = info.high_diff || 0;
            const highDiffPct = info.high_diff_pct || 0;
            const lowDiff = info.low_diff || 0;
            const lowDiffPct = info.low_diff_pct || 0;
            // positive movement uses callColor, negative uses putColor
            const highColor = highDiff >= 0 ? callColor : putColor;
            const lowColor = lowDiff >= 0 ? callColor : putColor;

            // Use the live streamer price if available, otherwise use the fetched price
            const displayPrice = (livePrice !== null) ? livePrice : info.current_price;
            const marketContextHtml = buildMarketContextItemsHtml();
            const ivStatsHtml = buildIvStatsHtml(info);
            const volTip = `Share of today\u2019s options volume: ${info.call_percentage.toFixed(1)}% calls / ${info.put_percentage.toFixed(1)}% puts.`;
            const expiryTip = 'Selected option expiration date(s) driving exposure and expected-move calculations.';
            priceInfo.innerHTML = `
                <div class="price-info-stats">
                    <div class="price-info-item price-info-primary" title="Last traded price${livePrice !== null ? ' (live stream)' : ''}.">
                        <strong>Current Price</strong>
                        <span data-live-price>$${displayPrice.toFixed(2)}</span>
                    </div>
                    <div class="price-info-item" title="Session high and how far current price sits below it.">
                        <strong>Day High</strong>
                        <span>$${info.high.toFixed(2)} <span style="color:${highColor}">(${highDiffPct>=0?'+':''}${highDiffPct.toFixed(2)}%)</span></span>
                    </div>
                    <div class="price-info-item" title="Session low and how far current price sits above it.">
                        <strong>Day Low</strong>
                        <span>$${info.low.toFixed(2)} <span style="color:${lowColor}">(${lowDiffPct>=0?'+':''}${lowDiffPct.toFixed(2)}%)</span></span>
                    </div>
                    <div class="price-info-item ${info.net_change >= 0 ? 'green' : 'red'}" title="Change vs. prior regular-session close.">
                        <strong>Change</strong>
                        <span>${info.net_change >= 0 ? '+' : ''}${info.net_change.toFixed(2)} (${info.net_percent >= 0 ? '+' : ''}${info.net_percent.toFixed(2)}%)</span>
                    </div>
                    <div class="price-info-item" title="${escapeAttr(volTip)}">
                        <strong>Vol Ratio</strong>
                        <span><span style="color: ${callColor}">${info.call_percentage.toFixed(2)}%</span>/<span style="color: ${putColor}">${info.put_percentage.toFixed(2)}%</span></span>
                    </div>
                    ${ivStatsHtml}
                    ${expectedMoveHtml}
                    <div class="price-info-item" title="${escapeAttr(expiryTip)}">
                        <strong>Expiries</strong>
                        <span>${expiryText}</span>
                    </div>
                </div>
                ${marketContextHtml}
            `;
        }
        
        function loadExpirations() {
            const ticker = document.getElementById('ticker').value;
            const requestId = ++latestExpirationsRequestId;
            expirationsLoading = true;
            document.getElementById('expiry-text').textContent = 'Loading expiries...';
            lastExpirationsAttemptMs = Date.now();
            fetch(`/expirations/${encodeURIComponent(ticker)}`)
                .then(async response => {
                    const payload = await response.json().catch(() => null);
                    if (!response.ok) {
                        throw new Error((payload && payload.error) || `HTTP ${response.status}`);
                    }
                    return payload;
                })
                .then(data => {
                    const activeTicker = document.getElementById('ticker').value;
                    if (requestId !== latestExpirationsRequestId || activeTicker !== ticker) {
                        return;
                    }
                    if (!data || data.error) {
                        expirationsLoading = false;
                        console.error('Expirations request failed:', data && data.error);
                        showError('Error loading expirations: ' + ((data && data.error) || 'empty response'));
                        showExpiryLoadFailure('No expiries (retrying...)');
                        return;
                    }
                    const todayEt = etTodayStr();
                    if (Array.isArray(data)) {
                        data = data.filter(date => String(date).slice(0, 10) >= todayEt);
                    }
                    if (!Array.isArray(data) || data.length === 0) {
                        expirationsLoading = false;
                        console.warn('No expirations returned for', ticker);
                        showExpiryLoadFailure('No expiries found (retrying...)');
                        return;
                    }
                    const optionsContainer = document.getElementById('expiry-options');
                    const previousSelections = Array.from(document.querySelectorAll('.expiry-option input[type="checkbox"]:checked')).map(cb => cb.value);
                    const sameTicker = expirationsTicker === String(ticker || '').toUpperCase();

                    // Work out the new selection before rebuilding the list
                    let wanted;
                    if (expirySelectionMode !== 'fixed') {
                        wanted = expiryDatesForRule(expirySelectionMode, data);
                    } else {
                        wanted = previousSelections.filter(date => data.includes(date));
                        if (!wanted.length && sameTicker) {
                            // Hand-picked expiries all expired: leave it empty for a new pick
                            if (previousSelections.length) {
                                expiredSelectionNotice = previousSelections.length === 1
                                    ? `${previousSelections[0]} expired - select an expiry`
                                    : 'Selected expiries expired - select an expiry';
                            }
                        } else if (!wanted.length) {
                            // New ticker without those dates (or first load): start from 0DTE
                            setExpirySelectionMode('0dte');
                            wanted = expiryDatesForRule('0dte', data);
                        }
                    }

                    // Clear existing options but keep the buttons
                    const buttons = optionsContainer.querySelector('.expiry-buttons');
                    optionsContainer.innerHTML = '';
                    
                    data.forEach(date => {
                        const optionDiv = document.createElement('div');
                        optionDiv.className = 'expiry-option';
                        
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.value = date;
                        checkbox.id = 'expiry-' + date;
                        
                        const label = document.createElement('label');
                        label.htmlFor = 'expiry-' + date;
                        label.textContent = date;
                        label.style.cursor = 'pointer';
                        label.style.flex = '1';
                        
                        checkbox.checked = wanted.includes(date);

                        // Picking by hand makes the selection fixed, except picking just
                        // today's expiry, which is 0DTE and should roll to tomorrow's
                        checkbox.addEventListener('change', function() {
                            const picked = getSelectedExpiryValues();
                            setExpirySelectionMode(picked.length === 1 && picked[0] === etTodayStr() ? '0dte' : 'fixed');
                            expiredSelectionNotice = '';
                            updateExpiryDisplay();
                            updateData();
                        });
                        
                        optionDiv.appendChild(checkbox);
                        optionDiv.appendChild(label);
                        optionsContainer.appendChild(optionDiv);
                    });
                    
                    // Re-add the buttons at the top
                    optionsContainer.insertBefore(buttons, optionsContainer.firstChild);
                    
                    if (wanted.length) expiredSelectionNotice = '';

                    expirationsLoading = false;
                    expirationsLoadedEtDate = todayEt;
                    expirationsTicker = String(ticker || '').toUpperCase();
                    updateExpiryDisplay();
                    updateData();
                })
                .catch(error => {
                    const activeTicker = document.getElementById('ticker').value;
                    if (requestId !== latestExpirationsRequestId || activeTicker !== ticker) {
                        return;
                    }
                    expirationsLoading = false;
                    console.error('Expirations request failed:', error);
                    showError('Error loading expirations: ' + error.message);
                    showExpiryLoadFailure('No expiries (retrying...)');
                });
        }

        // A failed reload (e.g. the overnight rollover check) keeps the list it already has
        function showExpiryLoadFailure(message) {
            if (document.querySelectorAll('.expiry-option input[type="checkbox"]').length > 0) {
                updateExpiryDisplay();
            } else {
                document.getElementById('expiry-text').textContent = message;
            }
        }
        
        function updateExpiryDisplay() {
            const checkedBoxes = document.querySelectorAll('.expiry-option input[type="checkbox"]:checked');
            const expiryText = document.getElementById('expiry-text');
            
            if (checkedBoxes.length === 0) {
                expiryText.textContent = expiredSelectionNotice || 'Select expiry dates...';
            } else if (checkedBoxes.length === 1) {
                expiryText.textContent = checkedBoxes[0].value;
            } else {
                expiryText.textContent = `${checkedBoxes.length} expiries selected`;
            }
        }
        
        // Add event listeners for checkboxes
        document.querySelectorAll('.chart-checkbox input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                syncMobilePanelButtons();
                // The price chart isn't part of the /update payload, so the response doesn't
                // change when it's toggled; apply the show/hide immediately.
                if (checkbox.id === 'price' && !checkbox.checked && lastData && Object.keys(lastData).length) {
                    updateCharts(lastData);
                }
                updateData();
            });
        });
        
        // Add event listeners for control checkboxes
        document.querySelectorAll('.control-group input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                if (checkbox.closest('.levels-option')) {
                    return;
                }
                updateData();
            });
        });

        document.getElementById('mobile-toggle-filters').addEventListener('click', function() {
            toggleMobilePanel('filters');
        });

        document.getElementById('mobile-toggle-charts').addEventListener('click', function() {
            toggleMobilePanel('charts');
        });
        
        document.getElementById('ticker').addEventListener('change', loadExpirations);
        
        // Add event listeners for dropdown toggle
        document.getElementById('expiry-display').addEventListener('click', function(e) {
            e.stopPropagation();
            const options = document.getElementById('expiry-options');
            options.classList.toggle('open');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            const dropdown = document.querySelector('.expiry-dropdown');
            const options = document.getElementById('expiry-options');
            if (dropdown && !dropdown.contains(e.target)) {
                options.classList.remove('open');
            }
            
            const levelsDropdown = document.querySelector('.levels-dropdown');
            const levelsOptions = document.getElementById('levels-options');
            if (levelsDropdown && !levelsDropdown.contains(e.target)) {
                levelsOptions.classList.remove('open');
            }
        });
        
        // Add event listeners for expiry selection buttons
        // Quick-select buttons set a rule (see expirySelectionMode) that is re-applied on
        // each new day and ticker change. "+1 Wk" / "+2 Wks" mean through Friday of next
        // week / the week after, not the first N listed expiries.
        function applyExpiryRule(mode) {
            setExpirySelectionMode(mode);
            expiredSelectionNotice = '';
            const checkboxes = Array.from(document.querySelectorAll('.expiry-option input[type="checkbox"]'));
            const wanted = expiryDatesForRule(mode, checkboxes.map(cb => cb.value));
            checkboxes.forEach(cb => { cb.checked = wanted.includes(cb.value); });
            updateExpiryDisplay();
            updateData();
        }

        const EXPIRY_RULE_BUTTONS = {
            selectAllExpiry: 'all',
            clearAllExpiry: '0dte',  // "Clear" leaves just the nearest expiry selected
            expiryToday: '0dte',
            expiryThisWk: 'thisWk',
            expiry2Wks: '1wk',
            expiry4Wks: '2wk',
            expiry1Mo: '1mo',
        };
        Object.entries(EXPIRY_RULE_BUTTONS).forEach(([id, mode]) => {
            document.getElementById(id).addEventListener('click', function(e) {
                e.stopPropagation();
                applyExpiryRule(mode);
            });
        });

        applyTheme('dark');

        applyMobileLayoutState();

        // Initial load - automatically load saved settings, or use defaults
        loadSettings(false);

        // Auto-update every 1 second
        updateInterval = setInterval(autoUpdateTick, 1000);

        // Refresh session clock / market-context items every minute
        marketContextTimer = setInterval(function() {
            if (lastPriceInfo) {
                refreshMarketContextItems(lastPriceInfo, livePrice !== null ? livePrice : lastPriceInfo.current_price);
            }
        }, 60000);
        
        // Handle window resize
        window.addEventListener('resize', () => {
            applyMobileLayoutState();
            if (tvPriceChart) {
                scheduleTVHistoricalOverlayDraw();
            }
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            clearInterval(updateInterval);
            if (marketContextTimer) clearInterval(marketContextTimer);
            disconnectPriceStream();
            Object.values(charts).forEach(chart => {
                Plotly.purge(chart);
            });
        });

        function toggleStreaming() {
            isStreaming = !isStreaming;
            const button = document.getElementById('streamToggle');
            button.textContent = isStreaming ? 'Auto-Update' : 'Paused';
            button.classList.toggle('paused', !isStreaming);
            
            if (isStreaming) {
                consecutiveUpdateFailures = 0;
                updateBackoffUntilMs = 0;
                updateInterval = setInterval(autoUpdateTick, 1000);
                // Reconnect real-time price stream when resuming
                const tickerVal = (document.getElementById('ticker').value || '').trim();
                if (tickerVal) connectPriceStream(tickerVal);
            } else {
                clearInterval(updateInterval);
                // Disconnect real-time price stream when pausing
                disconnectPriceStream();
            }
        }
        
        document.getElementById('streamToggle').addEventListener('click', toggleStreaming);

        // Settings save/load functions
        function gatherSettings() {
            return {
                ticker: document.getElementById('ticker').value,
                theme: document.getElementById('theme_select').value,
                timeframe: document.getElementById('timeframe').value,
                strike_range: document.getElementById('strike_range').value,
                exposure_metric: document.getElementById('exposure_metric').value,
                heatmap_type: document.getElementById('heatmap_type').value,
                heatmap_coloring_mode: document.getElementById('heatmap_coloring_mode').value,
                delta_adjusted_exposures: document.getElementById('delta_adjusted_exposures').checked,
                calculate_in_notional: document.getElementById('calculate_in_notional').checked,
                show_calls: document.getElementById('show_calls').checked,
                show_puts: document.getElementById('show_puts').checked,
                show_net: document.getElementById('show_net').checked,
                coloring_mode: document.getElementById('coloring_mode').value,
                levels_types: getSelectedPriceLevelTypes(),
                levels_count: document.getElementById('levels_count').value,
                use_heikin_ashi: document.getElementById('use_heikin_ashi').checked,
                    price_above_bubbles: getPriceAboveBubblesSetting(),
                horizontal_bars: document.getElementById('horizontal_bars').checked,
                show_abs_gex: document.getElementById('show_abs_gex').checked,
                abs_gex_opacity: document.getElementById('abs_gex_opacity').value,
                use_range: document.getElementById('use_range').checked,
                call_color: document.getElementById('call_color').value,
                put_color: document.getElementById('put_color').value,
                highlight_max_level: document.getElementById('highlight_max_level').checked,
                show_latest_level_lines: getShowLatestLevelLinesSetting(),
                max_level_color: document.getElementById('max_level_color').value,
                max_level_mode: document.getElementById('max_level_mode').value,
                em_range_locked: emRangeLocked,
                // Indicator state
                active_indicators: Array.from(tvActiveInds),
                indicator_params: JSON.parse(JSON.stringify(tvIndicatorParams)),
                // Chart visibility
                charts: {
                    price: document.getElementById('price').checked,
                    gamma: document.getElementById('gamma').checked,
                    heatmap: document.getElementById('heatmap').checked,
                    delta: document.getElementById('delta').checked,
                    vanna: document.getElementById('vanna').checked,
                    charm: document.getElementById('charm').checked,
                    speed: document.getElementById('speed').checked,
                    vomma: document.getElementById('vomma').checked,
                    color: document.getElementById('color').checked,
                    options_volume: document.getElementById('options_volume').checked,
                        open_interest: document.getElementById('open_interest').checked,
                    volume: document.getElementById('volume').checked,
                    large_trades: document.getElementById('large_trades').checked,
                        centroid: document.getElementById('centroid').checked
                }
            };
        }
        
        function applySettings(settings) {
            if (settings.ticker) document.getElementById('ticker').value = settings.ticker;
            applyTheme(settings.theme || 'dark');
            if (settings.timeframe) document.getElementById('timeframe').value = settings.timeframe;
            if (settings.strike_range) {
                document.getElementById('strike_range').value = settings.strike_range;
                document.getElementById('strike_range_value').textContent = settings.strike_range + '%';
            }
            if (settings.exposure_metric) document.getElementById('exposure_metric').value = settings.exposure_metric;
            if (settings.heatmap_type) document.getElementById('heatmap_type').value = settings.heatmap_type;
            if (settings.heatmap_coloring_mode) document.getElementById('heatmap_coloring_mode').value = settings.heatmap_coloring_mode;
            if (settings.delta_adjusted_exposures !== undefined) document.getElementById('delta_adjusted_exposures').checked = settings.delta_adjusted_exposures;
            if (settings.calculate_in_notional !== undefined) document.getElementById('calculate_in_notional').checked = settings.calculate_in_notional;
            if (settings.show_calls !== undefined) document.getElementById('show_calls').checked = settings.show_calls;
            if (settings.show_puts !== undefined) document.getElementById('show_puts').checked = settings.show_puts;
            if (settings.show_net !== undefined) document.getElementById('show_net').checked = settings.show_net;
            // Handle coloring_mode with migration from old color_intensity setting
            if (settings.coloring_mode) {
                document.getElementById('coloring_mode').value = settings.coloring_mode;
            } else if (settings.color_intensity !== undefined) {
                // Migrate old color_intensity boolean to new coloring_mode
                document.getElementById('coloring_mode').value = settings.color_intensity ? 'Linear Intensity' : 'Solid';
            }
            if (settings.levels_types) {
                document.querySelectorAll('.levels-option input').forEach(cb => cb.checked = false);
                settings.levels_types.forEach(type => {
                    const cb = document.getElementById('lvl-' + type.replace(/\\s+/g, ''));
                    if (cb) cb.checked = true;
                });
                updateLevelsDisplay();
            }
            if (settings.levels_count) document.getElementById('levels_count').value = settings.levels_count;
            if (settings.use_heikin_ashi !== undefined) document.getElementById('use_heikin_ashi').checked = settings.use_heikin_ashi;
            if (settings.horizontal_bars !== undefined) document.getElementById('horizontal_bars').checked = settings.horizontal_bars;
            if (settings.show_abs_gex !== undefined) document.getElementById('show_abs_gex').checked = settings.show_abs_gex;
            if (settings.abs_gex_opacity !== undefined) document.getElementById('abs_gex_opacity').value = settings.abs_gex_opacity;
            if (settings.use_range !== undefined) document.getElementById('use_range').checked = settings.use_range;
            if (settings.call_color) {
                document.getElementById('call_color').value = settings.call_color;
                callColor = settings.call_color;
            }
            if (settings.put_color) {
                document.getElementById('put_color').value = settings.put_color;
                putColor = settings.put_color;
            }
            if (settings.highlight_max_level !== undefined) {
                document.getElementById('highlight_max_level').checked = settings.highlight_max_level;
            }
            if (settings.show_latest_level_lines !== undefined) {
                setShowLatestLevelLinesSetting(settings.show_latest_level_lines);
            }
                if (settings.price_above_bubbles !== undefined) {
                    setPriceAboveBubblesSetting(settings.price_above_bubbles);
                }
            if (settings.max_level_color) {
                document.getElementById('max_level_color').value = settings.max_level_color;
                maxLevelColor = settings.max_level_color;
            }
            if (settings.max_level_mode) {
                document.getElementById('max_level_mode').value = settings.max_level_mode;
            }
            if (settings.em_range_locked !== undefined) {
                setEmRangeLocked(settings.em_range_locked);
            }
            // Chart visibility
            if (settings.charts) {
                Object.keys(settings.charts).forEach(chartId => {
                    const checkbox = document.getElementById(chartId);
                    if (checkbox) checkbox.checked = settings.charts[chartId];
                });
            }
            // Restore indicator parameters
            if (settings.indicator_params && typeof settings.indicator_params === 'object') {
                Object.keys(settings.indicator_params).forEach(key => {
                    tvIndicatorParams[key] = Object.assign(tvIndicatorParams[key] || {}, settings.indicator_params[key]);
                });
            }
            // Restore active indicators (an empty saved list means "none active")
            if (Array.isArray(settings.active_indicators)) {
                tvActiveInds.clear();
                settings.active_indicators.forEach(k => tvActiveInds.add(k));
            }
            syncMobilePanelButtons();
            // The price-chart toolbar is built once; rebuild it so its badges, "Top #"
            // input and indicator menu reflect the loaded settings.
            rebuildTVToolbar();
            if (tvPriceChart) {
                applyIndicators(tvIndicatorCandles, tvActiveInds);
            }
        }

        function rebuildTVToolbar() {
            const container = document.getElementById('price-chart');
            if (!tvPriceChart || !container) return;
            buildTVToolbar(container, tvLastCandles, callColor, putColor);
            const toolbarContainer = document.getElementById('tv-toolbar-container');
            if (toolbarContainer) {
                const titleEl = document.createElement('div');
                titleEl.className = 'tv-chart-title';
                titleEl.textContent = document.getElementById('use_heikin_ashi').checked
                    ? 'Price Chart (Heikin-Ashi)' : 'Price Chart';
                toolbarContainer.insertBefore(titleEl, toolbarContainer.firstChild);
            }
        }
        
        function saveSettings() {
            const settings = gatherSettings();
            fetch('/save_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const btn = document.getElementById('saveSettings');
                    btn.classList.add('success');
                    btn.textContent = '✓ Saved';
                    setTimeout(() => {
                        btn.classList.remove('success');
                        btn.textContent = '💾 Save';
                    }, 2000);
                } else {
                    showError('Error saving settings: ' + data.error);
                }
            })
            .catch(error => showError('Error saving settings: ' + error));
        }
        
        function loadSettings(showFeedback = true) {
            fetch('/load_settings')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    if (showFeedback) {
                        showError('Error loading settings: ' + data.error);
                    }
                    // If auto-loading fails, fall back to default initialization
                    if (!showFeedback) {
                        loadExpirations();
                    }
                } else {
                    applySettings(data);
                    if (showFeedback) {
                        const btn = document.getElementById('loadSettings');
                        btn.classList.add('success');
                        btn.textContent = '✓ Loaded';
                        setTimeout(() => {
                            btn.classList.remove('success');
                            btn.textContent = '📂 Load';
                        }, 2000);
                    }
                    // Reload expirations for the new ticker and update
                    loadExpirations();
                }
            })
            .catch(error => {
                if (showFeedback) {
                    showError('Error loading settings: ' + error);
                }
                // If auto-loading fails, fall back to default initialization
                if (!showFeedback) {
                    loadExpirations();
                }
            });
        }
        
        document.getElementById('saveSettings').addEventListener('click', saveSettings);
        document.getElementById('loadSettings').addEventListener('click', loadSettings);
        document.getElementById('theme_select').addEventListener('change', function(e) {
            applyTheme(e.target.value);
        });

        // Add event listener for ticker input
        document.getElementById('ticker').addEventListener('input', function(e) {
            // Stop auto-update when user starts typing
            if (isStreaming) {
                toggleStreaming();
            }
        });

        // Add event listener for ticker enter key
        document.getElementById('ticker').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                // Start auto-update when user hits enter
                if (!isStreaming) {
                    toggleStreaming();
                }
                // A new ticker needs its own expiries first (loadExpirations updates when done)
                const enteredTicker = (this.value || '').toUpperCase();
                if (enteredTicker !== expirationsTicker) {
                    loadExpirations();
                } else {
                    updateData();
                }
            }
        });

    // ── Token Monitor ──────────────────────────────────────────────────────
    function fetchTokenHealth() {
        fetch('/token_health')
            .then(r => r.json())
            .then(d => {
                const dot   = document.getElementById('tm-dot');
                if (!d.db_exists || d.error) {
                    dot.className = 'tm-dot tm-err';
                    const accessStat = document.getElementById('tm-access-stat');
                    const refreshStat = document.getElementById('tm-refresh-stat');
                    if (accessStat) {
                        accessStat.textContent = d.error || 'DB missing';
                        accessStat.title = d.db_path || '';
                    }
                    if (refreshStat) refreshStat.textContent = '';
                    return;
                }

                // Determine overall health
                const apiOk = d.api_ok === true;
                const atOk  = d.access_token_valid === true;
                const rtOk  = d.refresh_token_valid === true;
                const rtWarn = d.refresh_token_age_days !== null && d.refresh_token_age_days > 5;

                if (!atOk || !rtOk || !apiOk) {
                    dot.className = 'tm-dot tm-err';
                } else if (rtWarn) {
                    dot.className = 'tm-dot tm-warn';
                } else {
                    dot.className = 'tm-dot tm-ok';
                }

                const atMins = d.access_token_age_minutes !== null ? d.access_token_age_minutes.toFixed(1) + 'm' : '?';
                const rtDays = d.refresh_token_age_days   !== null ? d.refresh_token_age_days.toFixed(2)   + 'd' : '?';

                const atEl = document.getElementById('tm-access-stat');
                const rtEl = document.getElementById('tm-refresh-stat');

                atEl.textContent = 'access ' + atMins;
                atEl.title = d.access_token_valid
                    ? `Access token is ${atMins} old. Valid for ${(30 - d.access_token_age_minutes).toFixed(1)}m more (expires at 30 min).`
                    : `Access token is ${atMins} old — EXPIRED. Run the token getter script to refresh.`;
                atEl.style.color = d.access_token_valid ? '' : '#ff5252';

                rtEl.textContent = 'refresh ' + rtDays;
                const rtRemain = d.refresh_token_age_days !== null ? (7 - d.refresh_token_age_days).toFixed(2) : '?';
                rtEl.title = d.refresh_token_valid
                    ? `Refresh token is ${rtDays} old. Valid for ${rtRemain}d more (expires at 7 days). `
                      + (d.refresh_token_age_days > 5 ? 'Re-authenticate soon!' : 'Good.')
                    : `Refresh token is ${rtDays} old — EXPIRED. Full browser re-authentication required.`;
                rtEl.style.color = !d.refresh_token_valid ? '#ff5252' : (d.refresh_token_age_days > 5 ? '#ffb300' : '');
            })
            .catch(() => {
                const dot = document.getElementById('tm-dot');
                if (dot) { dot.className = 'tm-dot tm-err'; }
                const atEl = document.getElementById('tm-access-stat');
                if (atEl) { atEl.textContent = 'unreachable'; }
            });
    }

    function forceDeleteToken() {
        if (!confirm('Delete the Schwab token file? You will need to restart the server to re-authenticate.')) return;
        fetch('/token_delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
            .then(r => r.json())
            .then(d => {
                if (d.success) {
                    alert('Tokens cleared from: ' + d.db + '\\n\\n' + d.message);
                    fetchTokenHealth();
                } else {
                    alert('Delete failed: ' + d.error);
                }
            })
            .catch(err => alert('Request failed: ' + err));
    }

    // Fetch on load, then every 2 minutes
    fetchTokenHealth();
    setInterval(fetchTokenHealth, 120000);
    </script>
</body>
</html>
    ''')

@app.route('/favicon.svg')
@app.route('/favicon.ico')
def favicon():
    return Response(FAVICON_SVG, mimetype='image/svg+xml')

@app.route('/expirations/<ticker>')
def get_expirations(ticker):
    try:
        ticker = format_ticker(ticker)
        expirations = get_option_expirations(ticker)
        return jsonify(expirations)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def _safe_chart(name, builder, *args, **kwargs):
    """Run one chart builder; on failure log it and return None instead of failing /update."""
    try:
        return builder(*args, **kwargs)
    except Exception:
        import traceback
        print(f"Error building {name} chart:")
        traceback.print_exc()
        return None


@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    ticker = data.get('ticker')
    expiry = data.get('expiry')  # This can now be a list or single value
    
    ticker = format_ticker(ticker) 
    if not ticker or not expiry:
        return jsonify({'error': 'Missing ticker or expiry'}), 400
    
    # Handle both single expiry and multiple expiries
    if isinstance(expiry, list):
        expiry_dates = expiry
    else:
        expiry_dates = [expiry]

    try:
        # A page left open across midnight can still send yesterday's selection; its chain
        # is gone, so drop it and flag the response so the page reloads its expiry list.
        expiry_dates, expired_expiries = split_expired_expiries(expiry_dates)
        if not expiry_dates:
            return jsonify({'error': 'The selected expiry has expired - select a new expiration',
                            'expired_expiries': expired_expiries})

        # Setting: use volume or OI for exposure weighting
        exposure_metric = data.get('exposure_metric', "Open Interest")
        delta_adjusted = data.get('delta_adjusted', False)
        # Default calculate_in_notional to True if not present, but handle string 'true' just in case
        cin_val = data.get('calculate_in_notional', True)
        if isinstance(cin_val, str):
            calculate_in_notional = cin_val.lower() == 'true'
        else:
            calculate_in_notional = bool(cin_val)

        # Fetch options data for multiple dates
        if len(expiry_dates) == 1:
            calls, puts = fetch_options_for_date(ticker, expiry_dates[0], exposure_metric=exposure_metric, delta_adjusted=delta_adjusted, calculate_in_notional=calculate_in_notional)
        else:
            calls, puts = fetch_options_for_multiple_dates(ticker, expiry_dates, exposure_metric=exposure_metric, delta_adjusted=delta_adjusted, calculate_in_notional=calculate_in_notional)
        
        if calls.empty and puts.empty:
            return jsonify({'error': 'No options data found'})

        # Reuse the spot price cached by fetch_options_for_date (from the chain response)
        # to avoid a redundant separate quote API call.  Fall back to a live quote only if
        # the cache is missing or unexpectedly stale (>30 s).
        _cached_spot = _chain_spot_cache.get(ticker)
        if _cached_spot and (time.time() - _cached_spot[1]) < 30:
            S = _cached_spot[0]
        else:
            S = get_current_price(ticker)
        if S is None:
            return jsonify({'error': 'Could not fetch current price'})

        expiry_key = build_expiry_selection_key(expiry_dates)

        # Cache options data so /update_price can use it without re-fetching
        _options_cache[(ticker, expiry_key)] = {'calls': calls.copy(), 'puts': puts.copy(), 'S': S}
        
        # Get strike range
        strike_range = float(data.get('strike_range', 0.1))
        
        # A chain missing some of the selected expiries would record a fake dip in the
        # history tables, so only persist complete snapshots.
        complete_snapshot = not (calls.attrs.get('partial_fetch') or puts.attrs.get('partial_fetch'))

        # Recording history must never take the charts down with it
        try:
            if complete_snapshot:
                store_interval_data(ticker, S, strike_range, calls, puts, expiry_key=expiry_key)
                store_centroid_data(ticker, S, calls, puts, expiry_key=expiry_key)

            register_collector_selection(
                ticker, expiry_dates, strike_range, exposure_metric,
                delta_adjusted, calculate_in_notional, stored=complete_snapshot,
            )

            # Prune old sessions once per ET trading date
            global _last_prune_date
            today_et = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
            if _last_prune_date != today_et:
                _last_prune_date = today_et
                clear_old_data()
        except Exception:
            import traceback
            print("Error recording history in /update:")
            traceback.print_exc()

        # NOTE: price chart is handled separately by /update_price

        # Calculate volumes and other metrics
        use_range = data.get('use_range', False)  # Rename to use_range for clarity
        strike_range = float(data.get('strike_range', 0.1))
        
        if use_range:
            # Filter for options within the strike range percentage
            min_strike = S * (1 - strike_range)
            max_strike = S * (1 + strike_range)
            range_calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
            range_puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
            call_volume = int(range_calls['volume'].sum()) if not range_calls.empty else 0
            put_volume = int(range_puts['volume'].sum()) if not range_puts.empty else 0
        else:
            # Use all options
            call_volume = int(calls['volume'].sum()) if not calls.empty else 0
            put_volume = int(puts['volume'].sum()) if not puts.empty else 0
            
        total_volume = int(call_volume + put_volume)
        
        # Calculate volume percentages safely
        call_percentage = 0.0
        put_percentage = 0.0
        if total_volume > 0:
            call_percentage = float(round((call_volume / total_volume * 100), 1))
            put_percentage = float(round((put_volume / total_volume * 100), 1))
        
        # Get chart visibility settings
        show_calls = data.get('show_calls', True)
        show_puts = data.get('show_puts', True)
        show_net = data.get('show_net', True)
        # Handle coloring_mode with migration from old color_intensity setting
        coloring_mode = data.get('coloring_mode', 'Linear Intensity')
        call_color = data.get('call_color', '#00ff00')
        put_color = data.get('put_color', '#ff0000')
        heatmap_type = normalize_level_type(data.get('heatmap_type', 'GEX'))
        heatmap_coloring_mode = data.get('heatmap_coloring_mode', 'Global')
        horizontal = data.get('horizontal_bars', False)
        show_abs_gex = data.get('show_abs_gex', False)
        abs_gex_opacity = float(data.get('abs_gex_opacity', 0.2))
        highlight_max_level = data.get('highlight_max_level', False)
        max_level_color = data.get('max_level_color', '#800080')
        max_level_mode = data.get('max_level_mode', 'Absolute')
 
        
        response = {}
        
        # Create charts based on visibility settings
        # NOTE: price chart is handled by /update_price (separate concurrent request)

        if data.get('show_gamma', True):
            response['gamma'] = _safe_chart('gamma', create_exposure_chart, calls, puts, "GEX", "Gamma Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, show_abs_gex_area=show_abs_gex, abs_gex_opacity=abs_gex_opacity, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)

        if data.get('show_heatmap', False):
            response['heatmap'] = _safe_chart('heatmap', create_exposure_heatmap, 
                calls,
                puts,
                S,
                strike_range,
                show_calls,
                show_puts,
                show_net,
                call_color,
                put_color,
                expiry_dates,
                heatmap_type=heatmap_type,
                heatmap_coloring_mode=heatmap_coloring_mode,
            )
        
        if data.get('show_delta', True):
            response['delta'] = _safe_chart('delta', create_exposure_chart, calls, puts, "DEX", "Delta Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_vanna', True):
            response['vanna'] = _safe_chart('vanna', create_exposure_chart, calls, puts, "VEX", "Vanna Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_charm', True):
            response['charm'] = _safe_chart('charm', create_exposure_chart, calls, puts, "Charm", "Charm Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_speed', True):
            response['speed'] = _safe_chart('speed', create_exposure_chart, calls, puts, "Speed", "Speed Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_vomma', True):
            response['vomma'] = _safe_chart('vomma', create_exposure_chart, calls, puts, "Vomma", "Vomma Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)

        if data.get('show_color', True):
            response['color'] = _safe_chart('color', create_exposure_chart, calls, puts, "Color", "Color Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_volume', True):
            response['volume'] = _safe_chart('volume', create_volume_chart, call_volume, put_volume, use_range, call_color, put_color, expiry_dates)
        
        if data.get('show_options_volume', True):
            response['options_volume'] = _safe_chart('options_volume', create_options_volume_chart, calls, puts, S, strike_range, call_color, put_color, coloring_mode, show_calls, show_puts, show_net, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)

        
        if data.get('show_open_interest', True):
            response['open_interest'] = _safe_chart('open_interest', create_open_interest_chart, calls, puts, S, strike_range, call_color, put_color, coloring_mode, show_calls, show_puts, show_net, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_centroid', True):
            response['centroid'] = _safe_chart('centroid', create_centroid_chart, ticker, call_color, put_color, expiry_dates)

        if data.get('show_large_trades', True):
            response['large_trades'] = _safe_chart('large_trades', create_large_trades_table, calls, puts, S, strike_range, call_color, put_color, expiry_dates)

        
        # Drop charts whose builder failed (logged by _safe_chart)
        response = {key: value for key, value in response.items() if value is not None}

        response.update({
            'selected_expiries': expiry_dates,  # Add this to show which expiries are selected
            'expired_expiries': expired_expiries,
        })
        
        iv_stats = compute_iv_stats(calls, puts, S)

        # Get fresh quote data
        try:
            # Use appropriate base ticker for market tickers
            if ticker == "MARKET":
                quote_ticker = "$SPX"
            elif ticker == "MARKET2":
                quote_ticker = "SPY"
            else:
                quote_ticker = ticker

            cached_quote = _quote_cache.get(quote_ticker)
            if cached_quote and (time.time() - cached_quote[1]) < _QUOTE_CACHE_TTL_SEC:
                quote_data = cached_quote[0]
            else:
                quote_response = client.quote(quote_ticker)
                if not quote_response.ok:
                    raise Exception(f"Failed to fetch quote for display: {quote_response.status_code} {quote_response.reason}")
                quote_data = quote_response.json()
                _quote_cache[quote_ticker] = (quote_data, time.time())

            # --- Always Calculate Expected Move Range (same as chart logic) ---
            expected_move_range = None
            strikes_sorted = sorted(calls['strike'].unique()) if not calls.empty else []
            if strikes_sorted:
                atm_strike = min(strikes_sorted, key=lambda x: abs(x - S))
                def get_mid(df, strike):
                    row = df.loc[df['strike'] == strike]
                    if row is not None and not row.empty:
                        bid = row['bid'].values[0]
                        ask = row['ask'].values[0]
                        if bid > 0 and ask > 0:
                            return (bid + ask) / 2
                        elif bid > 0:
                            return bid
                        elif ask > 0:
                            return ask
                    return None
                call_mid_atm = get_mid(calls, atm_strike)
                put_mid_atm = get_mid(puts, atm_strike)
                straddle = (call_mid_atm if call_mid_atm is not None else 0) + (put_mid_atm if put_mid_atm is not None else 0)
                # Expected Move = ATM Straddle (most common market formula)
                expected_move = straddle
                if expected_move > 0:
                    upper = S + expected_move
                    lower = S - expected_move
                    expected_move_range = {'lower': round(lower, 2), 'upper': round(upper, 2), 'move': round(expected_move, 2)}

            if quote_data:
                ticker_data = quote_data.get(quote_ticker, {})
                quote = ticker_data.get('quote', {})

                # compute high/low diffs relative to current price
                high_price = quote.get('highPrice', S)
                low_price  = quote.get('lowPrice', S)
                high_diff = high_price - S
                low_diff  = low_price - S
                high_diff_pct = (high_diff / S * 100) if S else 0
                low_diff_pct  = (low_diff  / S * 100) if S else 0

                # add percentage of expected move boundaries if available (rounded to 2 decimals)
                if expected_move_range:
                    expected_move_range['lower_pct'] = round(((expected_move_range['lower'] - S) / S * 100), 2)
                    expected_move_range['upper_pct'] = round(((expected_move_range['upper'] - S) / S * 100), 2)

                response['price_info'] = {
                    'current_price': S,
                    'high': high_price,
                    'low': low_price,
                    'high_diff': round(high_diff, 2),
                    'high_diff_pct': round(high_diff_pct, 2),
                    'low_diff': round(low_diff, 2),
                    'low_diff_pct': round(low_diff_pct, 2),
                    'net_change': quote.get('netChange', 0),
                    'net_percent': quote.get('netPercentChange', 0),
                    'call_percentage': call_percentage,
                    'put_percentage': put_percentage,
                    'expected_move_range': expected_move_range,
                    'iv_stats': iv_stats,
                }
        except Exception as e:
            print(f"Error fetching quote data: {e}")
            response['price_info'] = {
                'current_price': S,
                'high': S,
                'low': S,
                'high_diff': 0,
                'high_diff_pct': 0,
                'low_diff': 0,
                'low_diff_pct': 0,
                'net_change': 0,
                'net_percent': 0,
                'call_percentage': call_percentage,
                'put_percentage': put_percentage,
                'expected_move_range': None,
                'iv_stats': iv_stats,
            }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/update_heatmap', methods=['POST'])
def update_heatmap():
    data = request.get_json()
    ticker = format_ticker(data.get('ticker'))
    expiry = data.get('expiry')

    if not ticker or not expiry:
        return jsonify({'error': 'Missing ticker or expiry'}), 400

    if isinstance(expiry, list):
        expiry_dates = expiry
    else:
        expiry_dates = [expiry]

    try:
        expiry_key = build_expiry_selection_key(expiry_dates)
        cached = _options_cache.get((ticker, expiry_key), {})
        calls = cached.get('calls')
        puts = cached.get('puts')
        S = cached.get('S')

        if calls is None or puts is None or S is None:
            return jsonify({'error': 'Heatmap cache not ready'}), 409

        heatmap = create_exposure_heatmap(
            calls,
            puts,
            S,
            float(data.get('strike_range', 0.1)),
            data.get('show_calls', True),
            data.get('show_puts', True),
            data.get('show_net', True),
            data.get('call_color', '#00ff00'),
            data.get('put_color', '#ff0000'),
            expiry_dates,
            heatmap_type=normalize_level_type(data.get('heatmap_type', 'GEX')),
            heatmap_coloring_mode=data.get('heatmap_coloring_mode', 'Global'),
        )

        return jsonify({
            'heatmap': heatmap,
            'selected_expiries': expiry_dates,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/update_price', methods=['POST'])
def update_price():
    """Lightweight endpoint that returns only the price chart data.
    Runs concurrently with /update so the price chart is never blocked
    by the heavier options-chain computations.
    """
    data = request.get_json()
    ticker = data.get('ticker')
    expiry = data.get('expiry')
    ticker = format_ticker(ticker)
    if not ticker:
        return jsonify({'error': 'Missing ticker'}), 400
    try:
        if isinstance(expiry, list):
            expiry_dates = expiry
        elif expiry:
            expiry_dates = [expiry]
        else:
            expiry_dates = []
        expiry_key = build_expiry_selection_key(expiry_dates)

        timeframe = int(data.get('timeframe', 1))
        call_color = data.get('call_color', '#00ff00')
        put_color = data.get('put_color', '#ff0000')
        exposure_levels_types = data.get('levels_types', [])
        exposure_levels_count = int(data.get('levels_count', 3))
        strike_range = float(data.get('strike_range', 0.1))
        use_heikin_ashi = data.get('use_heikin_ashi', False)
        highlight_max_level = data.get('highlight_max_level', False)
        show_latest_level_lines = data.get('show_latest_level_lines', True)
        max_level_color = data.get('max_level_color', '#800080')
        coloring_mode = data.get('coloring_mode', 'Linear Intensity')
        exposure_metric = data.get('exposure_metric', 'Open Interest')
        delta_adjusted = data.get('delta_adjusted', False)
        cin_val = data.get('calculate_in_notional', True)
        if isinstance(cin_val, str):
            calculate_in_notional = cin_val.lower() == 'true'
        else:
            calculate_in_notional = bool(cin_val)
        indicator_warmup = int(data.get('indicator_warmup', 220))

        price_data = get_price_history(ticker, timeframe=timeframe, lookback_bars=indicator_warmup)

        # Use the most recently cached options data for exposure overlays.
        # If no cache exists yet the chart renders without overlays and
        # will gain them after the first /update completes.
        cached = _options_cache.get((ticker, expiry_key), {})
        calls = cached.get('calls')
        puts = cached.get('puts')

        requires_options_data = bool(exposure_levels_types) and expiry_dates
        if requires_options_data and (calls is None or puts is None):
            if len(expiry_dates) == 1:
                calls, puts = fetch_options_for_date(
                    ticker,
                    expiry_dates[0],
                    exposure_metric=exposure_metric,
                    delta_adjusted=delta_adjusted,
                    calculate_in_notional=calculate_in_notional,
                )
            else:
                calls, puts = fetch_options_for_multiple_dates(
                    ticker,
                    expiry_dates,
                    exposure_metric=exposure_metric,
                    delta_adjusted=delta_adjusted,
                    calculate_in_notional=calculate_in_notional,
                )

            if not (calls.empty and puts.empty):
                # /update_heatmap needs a spot price from the cache; fall back to a quote
                # when /update hasn't populated one yet.
                spot = cached.get('S')
                if spot is None:
                    cached_spot = _chain_spot_cache.get(ticker)
                    spot = cached_spot[0] if cached_spot else get_current_price(ticker)
                _options_cache[(ticker, expiry_key)] = {
                    'calls': calls.copy(),
                    'puts': puts.copy(),
                    'S': spot,
                }

        price_chart = prepare_price_chart_data(
            price_data=price_data,
            calls=calls,
            puts=puts,
            exposure_levels_types=exposure_levels_types,
            exposure_levels_count=exposure_levels_count,
            call_color=call_color,
            put_color=put_color,
            strike_range=strike_range,
            use_heikin_ashi=use_heikin_ashi,
            highlight_max_level=highlight_max_level,
            max_level_color=max_level_color,
            coloring_mode=coloring_mode,
            ticker=ticker,
            selected_expiries=expiry_dates,
            show_latest_level_lines=show_latest_level_lines,
            timeframe=timeframe,
        )
        # Inject timeframe so the popout candle-close timer knows the selected interval
        try:
            import json as _json
            pc_dict = _json.loads(price_chart)
            pc_dict['timeframe'] = timeframe
            price_chart = _json.dumps(pc_dict)
        except Exception:
            pass
        return jsonify({'price': price_chart})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/save_settings', methods=['POST'])
def save_settings():
    try:
        settings = request.get_json()
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(settings, f, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_settings')
def load_settings():
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r') as f:
                settings = json.load(f)
            return jsonify(settings)
        else:
            return jsonify({'error': 'No settings file found'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/price_stream/<path:ticker>')
def price_stream(ticker):
    """Server-Sent Events endpoint for real-time price candle/quote updates.

    The frontend connects here via EventSource; the backend pushes CHART_EQUITY
    (completed 1-min candles) and LEVELONE_EQUITIES (real-time last price) data
    from the schwabdev websocket stream.
    """
    ticker = format_ticker(ticker)
    # Composite tickers aren't real stream symbols; stream their base instrument instead
    # (same mapping as the quote lookups in /update).
    ticker = {'MARKET': '$SPX', 'MARKET2': 'SPY'}.get(ticker, ticker)
    client_queue = queue.Queue(maxsize=300)
    price_streamer.subscribe(ticker, client_queue)

    def generate():
        try:
            # Initial connection confirmation
            yield 'data: {"type":"connected"}\n\n'
            while True:
                try:
                    payload = client_queue.get(timeout=20)
                    yield f'data: {payload}\n\n'
                except queue.Empty:
                    # Heartbeat keeps the connection alive through proxies/browsers
                    yield 'data: {"type":"heartbeat"}\n\n'
        except GeneratorExit:
            pass
        finally:
            price_streamer.unsubscribe_queue(ticker, client_queue)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',   # disable Nginx buffering if behind a proxy
            'Connection': 'keep-alive',
        }
    )


def _get_token_db_path():
    """Return the path to the TokenManager SQLite database."""
    return os.path.expanduser(os.getenv('SCHWAB_TOKENS_DB', '~/.schwabdev/tokens.db'))


def _read_token_db(db_path):
    """Read token row from the SQLite DB. Returns dict or None."""
    if not os.path.exists(db_path):
        return None
    with closing(sqlite3.connect(db_path)) as conn:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT access_token_issued, refresh_token_issued, access_token FROM schwabdev"
        ).fetchone()
    if not row:
        return None
    return {'access_token_issued': row[0], 'refresh_token_issued': row[1], 'access_token': row[2]}


@app.route('/token_health')
def token_health():
    """Return Schwab token status and API connectivity check as JSON."""
    import datetime as _dt
    import requests as _requests

    db_path = _get_token_db_path()
    result = {
        'db_path': db_path,
        'db_exists': os.path.exists(db_path),
        'access_token_age_minutes': None,
        'access_token_valid': None,
        'refresh_token_age_days': None,
        'refresh_token_valid': None,
        'api_ok': None,
        'api_message': None,
        'error': None,
    }

    try:
        row = _read_token_db(db_path)
        if row is None:
            result['error'] = 'No token row found in DB'
            return jsonify(result), 200

        now = _dt.datetime.now(_dt.timezone.utc)
        at_issued = _dt.datetime.fromisoformat(row['access_token_issued'])
        rt_issued = _dt.datetime.fromisoformat(row['refresh_token_issued'])
        if at_issued.tzinfo is None:
            at_issued = at_issued.replace(tzinfo=_dt.timezone.utc)
        if rt_issued.tzinfo is None:
            rt_issued = rt_issued.replace(tzinfo=_dt.timezone.utc)

        at_age_min = (now - at_issued).total_seconds() / 60
        rt_age_days = (now - rt_issued).total_seconds() / 86400

        result['access_token_age_minutes'] = round(at_age_min, 2)
        result['access_token_valid'] = at_age_min < 30
        result['refresh_token_age_days'] = round(rt_age_days, 4)
        result['refresh_token_valid'] = rt_age_days < 7

        # Live API test using the stored access token
        access_token = row.get('access_token', '')
        if access_token:
            try:
                resp = _requests.get(
                    'https://api.schwabapi.com/trader/v1/accounts/accountNumbers',
                    headers={'Authorization': f'Bearer {access_token}'},
                    timeout=10,
                )
                if resp.ok:
                    result['api_ok'] = True
                    result['api_message'] = f'API OK ({resp.status_code})'
                else:
                    result['api_ok'] = False
                    result['api_message'] = f'API {resp.status_code}: {resp.text[:120]}'
            except Exception as api_err:
                result['api_ok'] = False
                result['api_message'] = f'API error: {str(api_err)[:120]}'
        else:
            result['api_ok'] = False
            result['api_message'] = 'No access token stored'

    except Exception as e:
        result['error'] = str(e)
        return jsonify(result), 500

    return jsonify(result)


@app.route('/token_delete', methods=['POST'])
def token_delete():
    """Clear all rows from the token DB and null out in-memory client tokens (logout)."""
    # Require a JSON request: cross-site HTML forms can't send one, so another website
    # can't silently wipe the tokens via a form POST to localhost.
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Expected a JSON request'}), 400
    db_path = _get_token_db_path()
    if not os.path.exists(db_path):
        return jsonify({'success': False, 'error': f'Token DB not found: {db_path}'})
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("DELETE FROM schwabdev")
            conn.commit()

        # Null out in-memory tokens on the schwabdev client so it can't make API calls
        if client is not None:
            import datetime as _dt
            client.tokens.access_token = None
            client.tokens.refresh_token = None
            client.tokens.id_token = None
            client.tokens._access_token_issued = _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)
            client.tokens._refresh_token_issued = _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)

        return jsonify({
            'success': True,
            'db': db_path,
            'message': 'Logged out. Run your token getter script to re-authenticate.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=os.getenv('EZOPTIONS_DEBUG') == '1', port=5001, threaded=True)