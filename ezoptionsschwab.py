from flask import Flask, render_template_string, jsonify, request, Response, stream_with_context
import pandas as pd
import plotly.graph_objects as go
import numpy as np
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
import warnings
import json
import threading
import queue


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

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
    with closing(sqlite3.connect('options_data.db')) as conn:
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
                    abs_gex_total REAL,
                    date TEXT NOT NULL
                )
            ''')
            # Try to add net_charm column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE interval_data ADD COLUMN net_charm REAL')
            except sqlite3.OperationalError:
                pass 
            # Add abs_gex_total column if it's missing
            try:
                cursor.execute('ALTER TABLE interval_data ADD COLUMN abs_gex_total REAL')
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
                    date TEXT NOT NULL
                )
            ''')
            conn.commit()

def is_market_hours():
    """Return True if the current time is within regular market hours (9:30 AM - 4:00 PM ET, Mon-Fri)."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

# Function to store centroid data
def store_centroid_data(ticker, price, calls, puts):
    """Store call and put centroid data for 5-minute intervals during market hours only"""
    # Get current time in Eastern Time
    est = pytz.timezone('US/Eastern')
    current_time_est = datetime.now(est)
    
    # Check if we're in market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
    if current_time_est.weekday() >= 5:  # Weekend
        return
    
    market_open = current_time_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time_est.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if not (market_open <= current_time_est <= market_close):
        return  # Outside market hours
    
    current_time = int(current_time_est.timestamp())
    current_date = current_time_est.strftime('%Y-%m-%d')
    
    # Round to nearest 5-minute interval (300 seconds)
    interval_timestamp = (current_time // 300) * 300
    
    # Delete existing data for this 5-minute interval to update with most recent data
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                DELETE FROM centroid_data 
                WHERE ticker = ? AND timestamp = ? AND date = ?
            ''', (ticker, interval_timestamp, current_date))
            conn.commit()
    
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
    
    # Only store if we have volume data
    if call_volume > 0 or put_volume > 0:
        with closing(sqlite3.connect('options_data.db')) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute('''
                    INSERT INTO centroid_data (ticker, timestamp, price, call_centroid, put_centroid, call_volume, put_volume, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (ticker, interval_timestamp, price, call_centroid, put_centroid, call_volume, put_volume, current_date))
                conn.commit()

# Function to get centroid data
def get_centroid_data(ticker, date=None):
    """Get centroid data for current trading session only (market hours)"""
    if date is None:
        # Get current date in Eastern Time
        est = pytz.timezone('US/Eastern')
        current_date_est = datetime.now(est).strftime('%Y-%m-%d')
        date = current_date_est
    
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                SELECT timestamp, price, call_centroid, put_centroid, call_volume, put_volume
                FROM centroid_data
                WHERE ticker = ? AND date = ?
                ORDER BY timestamp
            ''', (ticker, date))
            
            # Filter data to only include market hours (9:30 AM - 4:00 PM ET)
            all_data = cursor.fetchall()
            filtered_data = []
            
            for row in all_data:
                timestamp = row[0]
                # Convert timestamp to Eastern Time
                dt_est = datetime.fromtimestamp(timestamp, pytz.timezone('US/Eastern'))
                
                # Check if within market hours
                market_open = dt_est.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = dt_est.replace(hour=16, minute=0, second=0, microsecond=0)
                
                if market_open <= dt_est <= market_close and dt_est.weekday() < 5:
                    filtered_data.append(row)
            
            return filtered_data

# Function to store interval data
def store_interval_data(ticker, price, strike_range, calls, puts):
    if not is_market_hours():
        return
    current_time = int(time.time())
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Round to nearest 5-minute interval (300 seconds)
    interval_timestamp = (current_time // 300) * 300
    
    # Delete existing data for this 5-minute interval to update with most recent data
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                DELETE FROM interval_data 
                WHERE ticker = ? AND timestamp = ? AND date = ?
            ''', (ticker, interval_timestamp, current_date))
            conn.commit()
    
    # Calculate strike range boundaries
    min_strike = price * (1 - strike_range)
    max_strike = price * (1 + strike_range)
    
    # Filter options within strike range
    range_calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    range_puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    # Calculate net gamma, delta, and vanna for each strike
    exposure_by_strike = {}
    for _, row in range_calls.iterrows():
        strike = row['strike']
        gamma = row['GEX']
        delta = row['DEX']
        vanna = row['VEX']
        charm = row['Charm']
        cur = exposure_by_strike.get(strike, {'gamma':0,'delta':0,'vanna':0,'charm':0,'call_gamma':0,'put_gamma':0})
        cur['gamma'] = cur.get('gamma',0) + gamma
        cur['delta'] = cur.get('delta',0) + delta
        cur['vanna'] = cur.get('vanna',0) + vanna
        cur['charm'] = cur.get('charm',0) + charm
        cur['call_gamma'] = cur.get('call_gamma',0) + gamma
        exposure_by_strike[strike] = cur
        
    for _, row in range_puts.iterrows():
        strike = row['strike']
        gamma = row['GEX']
        delta = row['DEX']
        vanna = row['VEX']
        charm = row['Charm']
        cur = exposure_by_strike.get(strike, {'gamma':0,'delta':0,'vanna':0,'charm':0,'call_gamma':0,'put_gamma':0})
        cur['gamma'] = cur.get('gamma',0) - gamma
        cur['delta'] = cur.get('delta',0) + delta
        cur['vanna'] = cur.get('vanna',0) + vanna
        cur['charm'] = cur.get('charm',0) + charm
        cur['put_gamma'] = cur.get('put_gamma',0) + gamma
        exposure_by_strike[strike] = cur
    
    # Store data for each strike
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            for strike, exposure in exposure_by_strike.items():
                abs_gex_total = abs(exposure.get('call_gamma',0)) + abs(exposure.get('put_gamma',0))
                cursor.execute('''
                    INSERT INTO interval_data (ticker, timestamp, price, strike, net_gamma, net_delta, net_vanna, net_charm, abs_gex_total, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (ticker, interval_timestamp, price, strike, exposure['gamma'], exposure['delta'], exposure['vanna'], exposure['charm'], abs_gex_total, current_date))
            conn.commit()

# Function to get interval data
def get_interval_data(ticker, date=None):
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                SELECT timestamp, price, strike, net_gamma, net_delta, net_vanna, net_charm, abs_gex_total
                FROM interval_data
                WHERE ticker = ? AND date = ?
                ORDER BY timestamp, strike
            ''', (ticker, date))
            all_data = cursor.fetchall()

    est = pytz.timezone('US/Eastern')
    filtered = []
    for row in all_data:
        dt = datetime.fromtimestamp(row[0], est)
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        if dt.weekday() < 5 and market_open <= dt <= market_close:
            filtered.append(row)
    return filtered

def get_last_session_date(ticker, table='interval_data'):
    """Return the most recent date that has data for ticker, or None."""
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(f'SELECT MAX(date) FROM {table} WHERE ticker = ?', (ticker,))
            row = cursor.fetchone()
            return row[0] if row and row[0] else None

# Function to clear old data
def clear_old_data():
    """Clear data from previous days, keeping only today's data"""
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).strftime('%Y-%m-%d')
    
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                DELETE FROM interval_data
                WHERE date < ?
            ''', (today,))
            cursor.execute('''
                DELETE FROM centroid_data
                WHERE date < ?
            ''', (today,))
            conn.commit()
            print(f"Cleared old data from database. Kept data from {today}")

# Function to clear centroid data for new session
def clear_centroid_session_data(ticker):
    """Clear centroid data at the start of a new trading session"""
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).strftime('%Y-%m-%d')
    
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                DELETE FROM centroid_data
                WHERE ticker = ? AND date = ?
            ''', (ticker, today))
            conn.commit()
            print(f"Cleared centroid data for new session: {ticker} on {today}")

# Initialize database
init_db()

# Clear old data at the start of the day
est = pytz.timezone('US/Eastern')
current_time_est = datetime.now(est)

# Clear old data at midnight ET
if current_time_est.hour == 0 and current_time_est.minute == 0:
    clear_old_data()

# Clear centroid data at market open (9:30 AM ET) for a fresh session
if current_time_est.hour == 9 and current_time_est.minute == 30 and current_time_est.weekday() < 5:
    # Note: This will clear centroid data for all tickers at market open
    # Individual ticker clearing happens in the update route when first accessed
    pass

# Global variables for streaming
current_chain = {'calls': [], 'puts': []}
last_update_time = 0
UPDATE_INTERVAL = 1  # seconds
current_ticker = None
current_expiry = None

# Cache for last fetched options data per ticker — used by /update_price
# so the price chart can refresh independently without re-fetching the full chain.
_options_cache = {}  # ticker -> {'calls': DataFrame, 'puts': DataFrame, 'S': float}

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
                        payload = json.dumps({
                            'type': 'candle',
                            'time': int(chart_time_ms) // 1000,
                            'open':   item.get('1'),
                            'high':   item.get('2'),
                            'low':    item.get('3'),
                            'close':  item.get('4'),
                            'volume': item.get('5'),
                        })
                        self._push(ticker, payload)
                elif service == 'CHART_FUTURES':
                    for item in msg.get('content', []):
                        ticker = item.get('key', '').upper()
                        chart_time_ms = item.get('3')
                        if not ticker or chart_time_ms is None:
                            continue
                        payload = json.dumps({
                            'type': 'candle',
                            'time': int(chart_time_ms) // 1000,
                            'open':   item.get('4'),
                            'high':   item.get('5'),
                            'low':    item.get('6'),
                            'close':  item.get('7'),
                            'volume': item.get('8'),
                        })
                        self._push(ticker, payload)
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
        needs_sub = False
        with self._lock:
            if ticker not in self._queues:
                self._queues[ticker] = []
            self._queues[ticker].append(q)
            if ticker not in self._subscribed:
                self._subscribed.add(ticker)
                needs_sub = True
        if needs_sub and self._started and self._stream:
            try:
                is_future = ticker.startswith('/')
                if is_future:
                    self._stream.send(self._stream.chart_futures(ticker, "0,1,2,3,4,5,6,7,8"))
                    self._stream.send(self._stream.level_one_futures(ticker, "0,1,2,3"))
                else:
                    self._stream.send(self._stream.chart_equity(ticker, "0,1,2,3,4,5,6,7,8"))
                    self._stream.send(self._stream.level_one_equities(ticker, "0,1,2,3"))
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
    return round(value / strike_interval) * strike_interval

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

def calculate_time_to_expiration(expiry_date):
    """
    Calculate time to expiration in years using Eastern Time.
    expiry_date: datetime.date object or string 'YYYY-MM-DD'
    Returns: time in years (float)
    """
    try:
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        if isinstance(expiry_date, str):
            expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        elif isinstance(expiry_date, datetime):
            expiry_date = expiry_date.date()
            
        # Set expiration to 4:00 PM ET on the expiration date
        expiry_dt = datetime.combine(expiry_date, datetime.min.time()) + timedelta(hours=16)
        expiry_dt = et_tz.localize(expiry_dt)
        
        # Calculate time difference in years
        diff = expiry_dt - now_et
        t = diff.total_seconds() / (365 * 24 * 3600)
        
        return t
             
    except Exception as e:
        print(f"Error calculating time to expiration: {e}")
        return 0

def fetch_options_for_date(ticker, date, exposure_metric="Open Interest", delta_adjusted: bool = False, calculate_in_notional: bool = True, S=None):
    if client is None:
        raise Exception("Schwab API client not initialized. Check your environment variables.")
    
    if ticker == "MARKET" or ticker == "MARKET2":
        # Step 1: Initialize Base
        base_ticker = "$SPX" if ticker == "MARKET" else "SPY"
        base_price = S if S else get_current_price(base_ticker)
        
        if not base_price:
             return pd.DataFrame(), pd.DataFrame()

        # Fetch Base chain to build strike grid
        base_calls_raw, base_puts_raw = fetch_options_for_date(base_ticker, date, exposure_metric, delta_adjusted, calculate_in_notional)
        
        if base_calls_raw.empty and base_puts_raw.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Step 2: Components to combine
        # Calculate bucket size from the base chain's actual strike spacing
        # (e.g. SPX → typically $5, SPY → $1). Avoids hardcoding.
        base_all_strikes = []
        if not base_calls_raw.empty: base_all_strikes.extend(base_calls_raw['strike'].tolist())
        if not base_puts_raw.empty: base_all_strikes.extend(base_puts_raw['strike'].tolist())
        bucket_size = get_strike_interval(base_all_strikes) if base_all_strikes else 5.0

        if ticker == "MARKET":
            component_tickers = ["$SPX", "QQQ"]
        else:
            component_tickers = ["SPY"]
        
        calls_list = []
        puts_list = []

        # Columns that get per-Greek normalization
        exposure_cols = ['GEX', 'DEX', 'VEX', 'Charm', 'Speed', 'Vomma', 'Color']
        activity_cols = ['openInterest', 'volume']

        # First pass: collect data and compute per-Greek total absolute exposure
        # for each component.  This lets us normalize each Greek independently
        # so that e.g. 5 000 OI on IWM is proportionally as loud as 500 000 on SPX.
        component_data = []
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
            
            # Per-Greek total absolute exposure ----------------------------
            totals = {}
            for col in exposure_cols + activity_cols:
                total = 0
                if not c.empty and col in c.columns:
                    total += c[col].abs().sum()
                if not p.empty and col in p.columns:
                    total += p[col].abs().sum()
                totals[col] = total if total > 0 else 1  # avoid /0
            
            component_data.append({
                'ticker': comp_tick,
                'price': comp_price,
                'calls': c,
                'puts': p,
                'totals': totals          # dict keyed by column name
            })
        
        if not component_data:
            return pd.DataFrame(), pd.DataFrame()
        
        # Use the base component's totals as a fixed reference anchor.
        # Base component (SPX) stays untouched (factor = 1.0).
        # Non-base components: factor = base_total / component_total
        # (scaled UP so their total matches SPX's magnitude).
        # Combined total ≈ N × base_total.
        #
        # Key stability property: a change in QQQ's or IWM's totals only
        # affects that component's bars — SPX/QQQ bars stay unchanged.
        base_cd = next((cd for cd in component_data if cd['ticker'] == base_ticker), component_data[0])
        base_totals = base_cd['totals']
        
        # Second pass: per-column normalization anchored to base component,
        # then map strikes to base-equivalent via moneyness.
        # Matches ezoptions.py: base component (SPX) is UNTOUCHED,
        # non-base components are scaled so their total matches SPX's total.
        for cd in component_data:
            comp_tick = cd['ticker']
            comp_price = cd['price']
            c = cd['calls']
            p = cd['puts']
            totals = cd['totals']
            
            is_base = (comp_tick == base_ticker)

            # Build per-column norm factors anchored to base component.
            # Base component: factor = 1.0 (unchanged).
            # Non-base: factor = base_total / component_total (scale up to match SPX magnitude).
            col_norm = {}
            for col in exposure_cols + activity_cols:
                if is_base:
                    col_norm[col] = 1.0
                else:
                    col_norm[col] = base_totals[col] / totals[col]
            
            # Process Calls
            if not c.empty:
                c = c.copy()
                
                # Normalize each column independently (Greeks + OI/Volume)
                # Base component is untouched (factor=1.0), others scaled to match base
                for col in exposure_cols + activity_cols:
                    if col in c.columns and not is_base:
                        c[col] = c[col] * col_norm[col]
                
                if is_base:
                    # Base component: strikes are already native SPX strikes.
                    # No moneyness mapping needed — just snap to nearest bucket
                    # to avoid floating-point ghost rows.
                    c['strike'] = (c['strike'] / bucket_size).round() * bucket_size
                    calls_list.append(c)
                else:
                    # Map strikes to base-equivalent via moneyness with linear
                    # interpolation between the two nearest buckets.  This prevents
                    # "bucket-hopping" where a small price change snaps 100% of a
                    # strike's exposure from one bucket to an adjacent one.
                    # Total exposure is conserved: weight_lo + weight_hi = 1.0.
                    weight_cols = exposure_cols + activity_cols
                    exact = (c['strike'] / comp_price) * base_price
                    # Round to avoid floating-point boundary jitter
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

            # Process Puts
            if not p.empty:
                p = p.copy()
                
                # Normalize each column independently (Greeks + OI/Volume)
                # Base component is untouched (factor=1.0), others scaled to match base
                for col in exposure_cols + activity_cols:
                    if col in p.columns and not is_base:
                        p[col] = p[col] * col_norm[col]
                
                if is_base:
                    # Base component: strikes are already native SPX strikes.
                    p['strike'] = (p['strike'] / bucket_size).round() * bucket_size
                    puts_list.append(p)
                else:
                    # Map strikes to base-equivalent via moneyness with linear interpolation
                    exact = (p['strike'] / comp_price) * base_price
                    # Round to avoid floating-point boundary jitter
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

        # Step 3: Combine and Aggregate by Strike
        combined_calls = pd.concat(calls_list, ignore_index=True) if calls_list else pd.DataFrame()
        combined_puts = pd.concat(puts_list, ignore_index=True) if puts_list else pd.DataFrame()

        def aggregate_market_data(df):
            if df.empty: return df
            sum_cols = ['openInterest', 'volume', 'GEX', 'DEX', 'VEX', 'Charm', 'Speed', 'Vomma', 'Color']
            avg_cols = ['lastPrice', 'bid', 'ask', 'impliedVolatility', 'delta', 'gamma', 'vega', 'theta', 'rho']
            
            agg_dict = {col: 'sum' for col in sum_cols if col in df.columns}
            agg_dict.update({col: 'mean' for col in avg_cols if col in df.columns})
            
            for col in df.columns:
                if col not in agg_dict and col != 'strike':
                    agg_dict[col] = 'first'
                    
            return df.groupby('strike', as_index=False).agg(agg_dict)

        combined_calls = aggregate_market_data(combined_calls)
        combined_puts = aggregate_market_data(combined_puts)
        
        return combined_calls, combined_puts

    try:
        expiry = datetime.strptime(date, '%Y-%m-%d').date()
        chain_response = client.option_chains(
            symbol=ticker,
            fromDate=expiry.strftime('%Y-%m-%d'),
            toDate=expiry.strftime('%Y-%m-%d'),
            contractType='ALL'
        )
        
        if not chain_response.ok:
            try:
                error_data = chain_response.json()
                error_msg = error_data.get('error', 'Unknown API error')
                if 'error_description' in error_data:
                    error_msg += f": {error_data['error_description']}"
                raise Exception(f"Schwab API Error: {error_msg}")
            except:
                raise Exception(f"Schwab API Error: {chain_response.status_code} {chain_response.reason}")
        
        chain = chain_response.json()
        S = float(chain.get('underlyingPrice', 0))
        if S == 0:
            S = get_current_price(ticker)
        if S is None:
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate time to expiration in years
        t = calculate_time_to_expiration(expiry)
        t = max(t, 1e-5)  # Minimum 1 minute
        r = 0.02  # risk-free rate (2% as default to match Yahoo script)
        
        calls_data = []
        puts_data = []
        display_tickers = format_display_ticker(ticker)
        
        for exp_date, strikes in chain.get('callExpDateMap', {}).items():
            for strike, options in strikes.items():
                for option in options:
                    if any(option['symbol'].startswith(t) for t in display_tickers):
                        K = float(option['strikePrice'])
                        raw_vol = float(option.get('volatility', -999.0))
                        vol = (raw_vol / 100) if raw_vol > 0 else 0.20
                        
                        # Calculate Greeks
                        if t > 0 and vol > 0 and K > 0:
                            delta, gamma, vega, vanna = calculate_greeks('c', S, K, t, vol, r, 0)
                            theta = calculate_theta('c', S, K, t, vol, r, 0)
                            rho = calculate_rho('c', S, K, t, vol, r, 0)
                        else:
                            delta = gamma = theta = vega = rho = 0
                        
                        option_data = {
                            'contractSymbol': option['symbol'],
                            'strike': K,
                            'lastPrice': float(option['last']),
                            'bid': float(option['bid']),
                            'ask': float(option['ask']),
                            'volume': int(option['totalVolume']),
                            'openInterest': int(option['openInterest']),
                            'impliedVolatility': vol,
                            'inTheMoney': option['inTheMoney'],
                            'expiration': datetime.strptime(exp_date.split(':')[0], '%Y-%m-%d').date(),
                            'delta': delta,
                            'gamma': gamma,
                            'theta': theta,
                            'vega': vega,
                            'rho': rho
                        }
                        option_data['side'] = infer_side(option_data['lastPrice'], option_data['bid'], option_data['ask'])
                        calls_data.append(option_data)
        
        for exp_date, strikes in chain.get('putExpDateMap', {}).items():
            for strike, options in strikes.items():
                for option in options:
                    if any(option['symbol'].startswith(t) for t in display_tickers):
                        K = float(option['strikePrice'])
                        raw_vol = float(option.get('volatility', -999.0))
                        vol = (raw_vol / 100) if raw_vol > 0 else 0.20
                        
                        # Calculate Greeks
                        if t > 0 and vol > 0 and K > 0:
                            delta, gamma, vega, vanna = calculate_greeks('p', S, K, t, vol, r, 0)
                            theta = calculate_theta('p', S, K, t, vol, r, 0)
                            rho = calculate_rho('p', S, K, t, vol, r, 0)
                        else:
                            delta = gamma = theta = vega = rho = 0
                        
                        option_data = {
                            'contractSymbol': option['symbol'],
                            'strike': K,
                            'lastPrice': float(option['last']),
                            'bid': float(option['bid']),
                            'ask': float(option['ask']),
                            'volume': int(option['totalVolume']),
                            'openInterest': int(option['openInterest']),
                            'impliedVolatility': vol,
                            'inTheMoney': option['inTheMoney'],
                            'expiration': datetime.strptime(exp_date.split(':')[0], '%Y-%m-%d').date(),
                            'delta': delta,
                            'gamma': gamma,
                            'theta': theta,
                            'vega': vega,
                            'rho': rho
                        }
                        option_data['side'] = infer_side(option_data['lastPrice'], option_data['bid'], option_data['ask'])
                        puts_data.append(option_data)
        
        # Calculate exposures with selected metric
        for option_data in calls_data:
            weight = 0
            if exposure_metric == 'Volume':
                weight = option_data['volume']
            elif exposure_metric == 'Max OI vs Volume':
                # Use the greater of OI and volume as the weight
                oi = option_data['openInterest']
                vol = option_data['volume']
                weight = max(oi, vol)
            elif exposure_metric == 'OI + Volume':
                # Use the sum of OI and volume as the weight
                weight = option_data['openInterest'] + option_data['volume']
            else: # Open Interest
                weight = option_data['openInterest']
                
            exposures = calculate_greek_exposures(option_data, S, weight, delta_adjusted=delta_adjusted, calculate_in_notional=calculate_in_notional)
            option_data.update(exposures)

        for option_data in puts_data:
            weight = 0
            if exposure_metric == 'Volume':
                weight = option_data['volume']
            elif exposure_metric == 'Max OI vs Volume':
                # Use the greater of OI and volume as the weight
                oi = option_data['openInterest']
                vol = option_data['volume']
                weight = max(oi, vol)
            elif exposure_metric == 'OI + Volume':
                # Use the sum of OI and volume as the weight
                weight = option_data['openInterest'] + option_data['volume']
            else: # Open Interest
                weight = option_data['openInterest']
                
            exposures = calculate_greek_exposures(option_data, S, weight, delta_adjusted=delta_adjusted, calculate_in_notional=calculate_in_notional)
            option_data.update(exposures)

        calls = pd.DataFrame(calls_data)
        puts = pd.DataFrame(puts_data)
        return calls, puts
        
    except Exception as e:
        msg = f"Error fetching options chain: {e}"
        print(msg)
        # Propagate so callers (API routes) can return the error to clients
        raise Exception(msg)

def calculate_greeks(flag, S, K, t, sigma, r=0.02, q=0):
    """Calculate delta, gamma, vega, vanna."""
    try:
        t = max(t, 1e-5)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        # Delta
        if flag == 'c':
            delta = np.exp(-q * t) * norm.cdf(d1)
        else:
            delta = np.exp(-q * t) * (norm.cdf(d1) - 1)
        
        # Gamma
        gamma = np.exp(-q * t) * norm.pdf(d1) / (S * sigma * np.sqrt(t))
        
        # Vega
        vega = S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)
        
        # Vanna
        vanna = -np.exp(-q * t) * norm.pdf(d1) * d2 / sigma
        
        return delta, gamma, vega, vanna
    except Exception as e:
        return 0, 0, 0, 0

def calculate_theta(flag, S, K, t, sigma, r=0.02, q=0):
    try:
        t = max(t, 1e-5)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        term1 = -S * np.exp(-q * t) * norm.pdf(d1) * sigma / (2 * np.sqrt(t))
        
        if flag == 'c':
            theta = term1 - r * K * np.exp(-r * t) * norm.cdf(d2) + q * S * np.exp(-q * t) * norm.cdf(d1)
        else:
            theta = term1 + r * K * np.exp(-r * t) * norm.cdf(-d2) - q * S * np.exp(-q * t) * norm.cdf(-d1)
        return theta
    except:
        return 0

def calculate_rho(flag, S, K, t, sigma, r=0.02, q=0):
    try:
        t = max(t, 1e-5)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        if flag == 'c':
            rho = K * t * np.exp(-r * t) * norm.cdf(d2)
        else:
            rho = -K * t * np.exp(-r * t) * norm.cdf(-d2)
        return rho
    except:
        return 0

def calculate_charm(flag, S, K, t, sigma, r=0.02, q=0):
    try:
        t = max(t, 1e-5)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        norm_d1 = norm.pdf(d1)
        
        if flag == 'c':
            charm = -np.exp(-q * t) * (norm_d1 * (2*(r-q)*t - d2*sigma*np.sqrt(t)) / (2*t*sigma*np.sqrt(t)) - q * norm.cdf(d1))
        else:
            charm = -np.exp(-q * t) * (norm_d1 * (2*(r-q)*t - d2*sigma*np.sqrt(t)) / (2*t*sigma*np.sqrt(t)) + q * norm.cdf(-d1))
        return charm
    except:
        return 0

def calculate_speed(flag, S, K, t, sigma, r=0.02, q=0):
    try:
        t = max(t, 1e-5)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        gamma = np.exp(-q * t) * norm.pdf(d1) / (S * sigma * np.sqrt(t))
        speed = -gamma * (d1/(sigma * np.sqrt(t)) + 1) / S
        return speed
    except:
        return 0

def calculate_vomma(flag, S, K, t, sigma, r=0.02, q=0):
    try:
        t = max(t, 1e-5)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        vega = S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)
        vomma = vega * (d1 * d2) / sigma
        return vomma
    except:
        return 0

def calculate_color(flag, S, K, t, sigma, r=0.02, q=0):
    try:
        t = max(t, 1e-5)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        norm_d1 = norm.pdf(d1)
        term1 = 2 * (r - q) * t
        term2 = d2 * sigma * np.sqrt(t)
        color = -np.exp(-q*t) * (norm_d1 / (2 * S * t * sigma * np.sqrt(t))) * \
                (1 + (term1 - term2) * d1 / (2 * t * sigma * np.sqrt(t)))
        return color
    except:
        return 0

def calculate_greek_exposures(option, S, weight, delta_adjusted: bool = False, calculate_in_notional: bool = True):
    """Calculate accurate Greek exposures per $1 move, weighted by the provided weight."""
    contract_size = 100
    
    # Recalculate Greeks to ensure consistency with S and t
    vol = option['impliedVolatility']
    
    # Calculate time to expiration in years
    expiry_date = option['expiration']
    t = calculate_time_to_expiration(expiry_date)
    t = max(t, 1e-5)  # Minimum time to prevent division by zero
    
    # Determine flag (c/p) based on symbol if possible, or use parameter
    flag = 'c'
    if 'P' in option['contractSymbol'] and not 'C' in option['contractSymbol']:
         flag = 'p'
    import re
    match = re.search(r'\d{6}([CP])', option['contractSymbol'])
    if match:
        flag = match.group(1).lower()

    r = 0.02  # risk-free rate
    q = 0

    # Re-calculate Greeks using consistent inputs
    K = option['strike']
    delta, gamma, _, vanna = calculate_greeks(flag, S, K, t, vol, r, q)

    # Calculate exposures (per $1 move in underlying)
    # Check if calculation should be in notional (dollars) or standard (shares)
    spot_multiplier = S if calculate_in_notional else 1.0
    
    # DEX: Delta exposure
    # Delta is unitless (shares/contract / 100). 
    # Notional DEX = Delta * 100 * S. (Dollar Value of Delta).
    dex = delta * weight * contract_size * spot_multiplier
    
    # GEX: Gamma exposure
    # GEX (Notional) ~ Gamma * S * S * 0.01
    gex = gamma * weight * contract_size * S * spot_multiplier * 0.01
    
    # VEX: Vanna exposure
    vanna_exposure = vanna * weight * contract_size * spot_multiplier * 0.01

    # Charm
    charm = calculate_charm(flag, S, K, t, vol, r, q)
    charm_exposure = charm * weight * contract_size * spot_multiplier / 365.0
    
    # Speed
    # Speed Exposure (Notional) ~ Speed * S * S * 0.01 
    speed = calculate_speed(flag, S, K, t, vol, r, q)
    speed_exposure = speed * weight * contract_size * S * spot_multiplier * 0.01
    
    # Vomma
    vomma = calculate_vomma(flag, S, K, t, vol, r, q)
    vomma_exposure = vomma * weight * contract_size * 0.01

    # Color
    color = calculate_color(flag, S, K, t, vol, r, q)
    color_exposure = color * weight * contract_size * S * spot_multiplier * 0.01 / 365.0

    # Apply delta adjustment if enabled
    if delta_adjusted:
        abs_delta = abs(delta)
        gex *= abs_delta
        vanna_exposure *= abs_delta
        charm_exposure *= abs_delta
        speed_exposure *= abs_delta
        vomma_exposure *= abs_delta
        color_exposure *= abs_delta

    return {
        'DEX': dex,
        'GEX': gex,
        'VEX': vanna_exposure,
        'Charm': charm_exposure,
        'Speed': speed_exposure,
        'Vomma': vomma_exposure,
        'Color': color_exposure
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
        response = client.option_expiration_chain(ticker)
        if not response.ok:
            raise Exception(f"Failed to fetch expirations: {response.status_code} {response.reason}")
        response_json = response.json()
        if response_json and 'expirationList' in response_json:
            expiration_dates = [item['expirationDate'] for item in response_json['expirationList']]
            return sorted(expiration_dates)
        return []
    except Exception as e:
        msg = f"Error fetching option expirations: {e}"
        print(msg)
        # Propagate the error so route handlers or Flask error handlers can return it to clients
        raise Exception(msg)

def get_color_with_opacity(value, max_value, base_color, color_intensity=True):
    """Get color with opacity based on value. Legacy function for backward compatibility."""
    if not color_intensity:
        opacity = 1.0  # Full opacity when color intensity is disabled
    else:
        # Ensure opacity is between 0.3 and 0.8 for better visibility and less intensity
        opacity = min(max(abs(value / max_value) if max_value != 0 else 0, 0.3), 0.8)
        
    if isinstance(base_color, str) and base_color.startswith('#'):
        # Convert hex to rgb
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        return f'rgba({r}, {g}, {b}, {opacity})'
    return base_color

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
        # Calculate total net volume from the entire chain (not just strike range)
        total_call_volume = calls['volume'].sum() if not calls.empty and 'volume' in calls.columns else 0
        total_put_volume = puts['volume'].sum() if not puts.empty and 'volume' in puts.columns else 0
        total_net_volume = total_call_volume - total_put_volume
    
    # Create the main title and net exposure as separate annotations
    fig = go.Figure()
    
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
                hovertemplate='Strike: %{y}<br>Value: %{text}<extra></extra>',
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
                hovertemplate='Strike: %{x}<br>Value: %{text}<extra></extra>',
                marker_line_width=0
            ))
    
    if show_puts and not puts_df.empty:
        # Apply coloring mode
        put_colors = get_colors(put_color, puts_df[exposure_type], max_exposure, coloring_mode)
            
        if horizontal:
            fig.add_trace(go.Bar(
                y=puts_df['strike'].tolist(),
                x=(-puts_df[exposure_type]).tolist(),
                name='Put',
                marker_color=put_colors,
                text=[format_large_number(val) for val in puts_df[exposure_type]],
                textposition='auto',
                orientation='h',
                hovertemplate='Strike: %{y}<br>Value: %{text}<extra></extra>',
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=puts_df['strike'].tolist(),
                y=(-puts_df[exposure_type]).tolist(),
                name='Put',
                marker_color=put_colors,
                text=[format_large_number(val) for val in puts_df[exposure_type]],
                textposition='auto',
                hovertemplate='Strike: %{x}<br>Value: %{text}<extra></extra>',
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
                marker_color=net_colors,
                text=[format_large_number(val) for val in net_exposure],
                textposition='auto',
                orientation='h',
                hovertemplate='Strike: %{y}<br>Net Value: %{text}<extra></extra>',
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=all_strikes,
                y=net_exposure,
                name='Net',
                marker_color=net_colors,
                text=[format_large_number(val) for val in net_exposure],
                textposition='auto',
                hovertemplate='Strike: %{x}<br>Net Value: %{text}<extra></extra>',
                marker_line_width=0
            ))
    
    if horizontal:
        # Add current price line
        fig.add_hline(
            y=S,
            line_dash="dash",
            line_color=text_color,
            opacity=0.5,
            annotation_text=f"{S:.2f}",
            annotation_position="right",
            annotation_font_color=text_color,
            line_width=1
        )
    else:
        # Add current price line with improved styling
        fig.add_vline(
            x=S,
            line_dash="dash",
            line_color=text_color,
            opacity=0.5,
            annotation_text=f"{S:.2f}",
            annotation_position="top",
            annotation_font_color=text_color,
            line_width=1
        )
    
    # Calculate padding as percentage of price range
    padding = (max_strike - min_strike) * 0.02
    
    # Add expiry info to title if multiple expiries are selected
    chart_title = title
    if selected_expiries and len(selected_expiries) > 1:
        chart_title = f"{title} ({len(selected_expiries)} expiries)"
    
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

    # Update layout with improved styling and split title
    fig.update_layout(
        title=dict(
            text=chart_title,  # Main title with expiry info
            font=dict(color=text_color, size=16),
            x=0.5,
            xanchor='center',
            y=0.98  # Push title higher to avoid collision with price annotation
        ),
        annotations=list(fig.layout.annotations) + [
            dict(
                text=f"Net: {format_large_number(abs(total_net_exposure))}",
                x=0.98,
                y=1.03,
                xref='paper',
                yref='paper',
                xanchor='right',
                yanchor='top',
                showarrow=False,
                font=dict(
                    size=14,
                    color=call_color if total_net_exposure >= 0 else put_color
                )
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
        margin=dict(l=50, r=80, t=60, b=20),
        hoverlabel=dict(
            bgcolor=background_color,
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100,
        height=500
    )
    
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
    title = base_title
    if selected_expiries and len(selected_expiries) > 1:
        title = f"{base_title} ({len(selected_expiries)} expiries)"
    fig = go.Figure(data=[go.Pie(
        labels=['Calls', 'Puts'],
        values=[call_volume, put_volume],
        hole=0.3,
        marker_colors=[call_color, put_color]
    )])
    
    fig.update_layout(
        title_text=title,
        showlegend=True,
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white'),
        height=500
    )
    
    return fig.to_json()

def create_options_volume_chart(calls, puts, S, strike_range=0.02, call_color='#00FF00', put_color='#FF0000', coloring_mode='Solid', show_calls=True, show_puts=True, show_net=True, selected_expiries=None, horizontal=False, highlight_max_level=False, max_level_color='#800080', max_level_mode='Absolute'):
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
                hovertemplate='Strike: %{y}<br>Volume: %{x}<extra></extra>',
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
                hovertemplate='Strike: %{x}<br>Volume: %{y}<extra></extra>',
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
                hovertemplate='Strike: %{y}<br>Volume: %{text}<extra></extra>',  # Show positive value in hover
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
                hovertemplate='Strike: %{x}<br>Volume: %{text}<extra></extra>',  # Show positive value in hover
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
                marker_color=net_colors,
                text=[f"{vol:,.0f}" for vol in net_volume],
                textposition='auto',
                orientation='h',
                hovertemplate='Strike: %{y}<br>Net Volume: %{x}<extra></extra>',
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=all_strikes_list,
                y=net_volume,
                name='Net',
                marker_color=net_colors,
                text=[f"{vol:,.0f}" for vol in net_volume],
                textposition='auto',
                hovertemplate='Strike: %{x}<br>Net Volume: %{y}<extra></extra>',
                marker_line_width=0
            ))
    
    if horizontal:
        # Add current price line
        fig.add_hline(
            y=S,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text=f"{S:.2f}",
            annotation_position="right",
            annotation_font_color="white",
            line_width=1
        )
    else:
        # Add current price line
        fig.add_vline(
            x=S,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text=f"{S:.2f}",
            annotation_position="top",
            annotation_font_color="white",
            line_width=1
        )
    
    # Add expiry info to title if multiple expiries are selected
    chart_title = 'Options Volume by Strike'
    if selected_expiries and len(selected_expiries) > 1:
        chart_title = f"Options Volume by Strike ({len(selected_expiries)} expiries)"
    
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
        title=dict(
            text=chart_title,
            font=dict(color='#CCCCCC', size=16),
            x=0.5,
            xanchor='center'
        ),
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
        margin=dict(l=50, r=50, t=50, b=100),
        hoverlabel=dict(
            bgcolor='#1E1E1E',
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100,
        showlegend=True,
        height=500
    )
    
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

def update_options_chain(ticker, expiration_date=None):
    """Update the options chain by fetching new data from the API"""
    global current_chain, last_update_time, current_ticker, current_expiry
    
    current_time = time.time()
    if current_time - last_update_time < 1.0:  # Enforce 1 second minimum between API calls
        return  # Don't update if less than 1 second has passed
        
    try:
        # Fetch new options chain data (default to OI-weighted exposures for background cache)
        new_chain = fetch_options_for_date(ticker, expiration_date, exposure_metric="Open Interest")
        if new_chain and not new_chain[0].empty and not new_chain[1].empty:
            current_chain = {
                'calls': new_chain[0].to_dict('records'),
                'puts': new_chain[1].to_dict('records')
            }
            last_update_time = current_time
            current_ticker = ticker
            current_expiry = expiration_date
    except Exception as e:
        print(f"Error updating options chain: {e}")

def get_price_history(ticker, timeframe=1):
    if ticker == "MARKET":
        ticker = "$SPX"
    elif ticker == "MARKET2":
        ticker = "SPY"
    try:
        # Get current time in EST
        est = datetime.now(pytz.timezone('US/Eastern'))
        current_date = est.date()
        
        # Calculate start date (5 days ago to ensure we get previous trading day)
        start_date = datetime.combine(current_date - timedelta(days=5), datetime.min.time())
        end_date = datetime.combine(current_date + timedelta(days=1), datetime.min.time())
        
        # Convert dates to milliseconds since epoch
        response = client.price_history(
            symbol=ticker,
            periodType="day",
            period=5,  # Get 5 days of data
            frequencyType="minute",
            frequency=timeframe,
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

        # Get previous trading day's close
        prev_day_candles = []
        for candle in reversed(candles):
            candle_time = datetime.fromtimestamp(candle['datetime']/1000, pytz.timezone('US/Eastern'))
            if candle_time.date() < current_date:
                prev_day_candles.append(candle)
                if len(prev_day_candles) >= 30:  # Get at least 30 minutes of data
                    break

        # Get the last candle of the previous trading day
        prev_day_close = prev_day_candles[-1]['close'] if prev_day_candles else None

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
            if market_open <= et <= market_close:
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

def create_price_chart(price_data, calls=None, puts=None, exposure_levels_types=[], exposure_levels_count=3, call_color='#00FF00', put_color='#FF0000', strike_range=0.02, use_heikin_ashi=False, highlight_max_level=False, max_level_color='#800080', coloring_mode='Linear Intensity'):
    # Handle backward compatibility or empty default
    if isinstance(exposure_levels_types, str):
        if exposure_levels_types == 'None':
            exposure_levels_types = []
        else:
            exposure_levels_types = [exposure_levels_types]
            
    if not price_data or 'candles' not in price_data or not price_data['candles']:
        return go.Figure().to_json()
    
    # Filter for market hours
    candles = filter_market_hours(price_data['candles'])
    if not candles:
        return go.Figure().to_json()
    
    # Get current time in EST
    est = datetime.now(pytz.timezone('US/Eastern'))
    current_date = est.date()
    
    # Sort candles by datetime and remove duplicates
    unique_candles = {}
    for candle in candles:
        candle_time = datetime.fromtimestamp(candle['datetime']/1000, pytz.timezone('US/Eastern'))
        unique_candles[candle_time] = candle
    
    # Convert back to list and sort
    sorted_candles = sorted(unique_candles.items(), key=lambda x: x[0])
    all_candles = [candle for _, candle in sorted_candles]
    
    # Filter for current day's candles only
    current_day_candles = []
    for candle in all_candles:
        candle_time = datetime.fromtimestamp(candle['datetime']/1000, pytz.timezone('US/Eastern'))
        # Convert both dates to EST and compare
        candle_date = candle_time.date()
        if candle_date == current_date:
            current_day_candles.append(candle)
    
    # If no current day candles, use the most recent day's candles
    if not current_day_candles:
        # Get the most recent trading day
        most_recent_day = max(candle['datetime'] for candle in all_candles)
        most_recent_day = datetime.fromtimestamp(most_recent_day/1000, pytz.timezone('US/Eastern')).date()
        
        # Filter candles for most recent trading day
        current_day_candles = []
        for candle in all_candles:
            candle_time = datetime.fromtimestamp(candle['datetime']/1000, pytz.timezone('US/Eastern'))
            if candle_time.date() == most_recent_day:
                current_day_candles.append(candle)
    
    # Use all candles for calculations but current day candles for display
    if use_heikin_ashi:
        ha_candles = convert_to_heikin_ashi(all_candles)  # Use all candles for calculations
        display_candles = convert_to_heikin_ashi(current_day_candles)  # Use current day for display
    else:
        # Use regular candles
        ha_candles = all_candles
        display_candles = current_day_candles
    
    # Get previous day's close
    previous_day_close = None
    for candle in reversed(all_candles):
        candle_time = datetime.fromtimestamp(candle['datetime']/1000, pytz.timezone('US/Eastern'))
        if candle_time.date() < current_date:
            previous_day_close = candle['close']
            break
    
    if previous_day_close is None:
        previous_day_close = display_candles[0]['close'] if display_candles else 0
    
    dates = [datetime.fromtimestamp(candle['datetime']/1000) for candle in display_candles]
    opens = [candle['open'] for candle in display_candles]
    highs = [candle['high'] for candle in display_candles]
    lows = [candle['low'] for candle in display_candles]
    closes = [candle['close'] for candle in display_candles]
    volumes = [candle['volume'] for candle in display_candles]
    

    
    # Calculate price range for proper scaling
    if not lows or not highs:  # Check if lists are empty
        return go.Figure().to_json()
        
    price_min = min(lows)
    price_max = max(highs)
    price_range = price_max - price_min
    padding = price_range * 0.02  # 2% padding
    
    # Get current price for strike range calculation
    current_price = closes[-1] if closes else (price_min + price_max) / 2
    
    # Determine if last candle is up or down
    last_candle_up = closes[-1] >= opens[-1] if len(closes) > 0 else True
    current_price_color = call_color if last_candle_up else put_color
    
    # Calculate strike range boundaries
    min_strike = current_price * (1 - strike_range)
    max_strike = current_price * (1 + strike_range)
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Add candlestick trace to the first subplot
    fig.add_trace(go.Candlestick(
        x=dates,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        name='OHLC',
        increasing_line_color=call_color,
        decreasing_line_color=put_color,
        increasing_fillcolor=call_color,
        decreasing_fillcolor=put_color
    ))
    
    # Modify the volume trace coloring
    volume_colors = []
    for i in range(len(closes)):
        if i == 0:
            # For first candle, compare close to open
            is_up = closes[i] >= opens[i]
        else:
            # For other candles, compare to previous close
            is_up = closes[i] >= closes[i-1]
        # Use call_color for up volume and put_color for down volume
        volume_colors.append(call_color if is_up else put_color)
    
    # Update the volume trace with the new colors
    fig.add_trace(go.Bar(
        x=dates,
        y=volumes,
        name='Volume',
        marker_color=volume_colors,
        marker_line_width=0,
        yaxis='y2',
        opacity=0.7  # Add some transparency
    ))
    

    
    # Update layout with subplots
    chart_title = 'Price Chart (Heikin-Ashi)' if use_heikin_ashi else 'Price Chart'
    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(color='#CCCCCC', size=16),
            x=0.5,
            xanchor='center',
            y=0.98
        ),
        xaxis=dict(
            title='',
            title_font=dict(color='#CCCCCC'),
            tickfont=dict(color='#CCCCCC'),
            gridcolor='#333333',
            linecolor='#333333',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#333333',
            rangeslider=dict(visible=False),
            tickformat='%H:%M',
            showline=True,
            linewidth=1,
            mirror=True,
            domain=[0, 1]
        ),
        yaxis=dict(
            title='Price',
            title_font=dict(color='#CCCCCC'),
            tickfont=dict(color='#CCCCCC'),
            gridcolor='#333333',
            linecolor='#333333',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#333333',
            showline=True,
            linewidth=1,
            mirror=True,
            autorange=True,  # Enable auto-scaling
            domain=[0.25, 1],  # Price takes up 75% of the space
            side='right',  # Move axis to right side
            title_standoff=0,  # Reduce space between title and axis
            automargin=True  # Enable automatic margin adjustment
        ),
        yaxis2=dict(
            title='Volume',
            title_font=dict(color='#CCCCCC'),
            tickfont=dict(color='#CCCCCC'),
            gridcolor='#333333',
            linecolor='#333333',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#333333',
            showline=True,
            linewidth=1,
            mirror=True,
            domain=[0, 0.2],  # Volume takes up 20% of the space
            side='right',  # Move axis to right side
            title_standoff=0,  # Reduce space between title and axis
            automargin=True  # Enable automatic margin adjustment
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
        bargap=0.1,
        bargroupgap=0.1,
        margin=dict(l=50, r=120, t=30, b=20),  # Increased right margin further
        hovermode='x unified',
        showlegend=True,
        height=550,  # Increased height for better visibility
        dragmode='pan',  # Set default tool to pan
        # Add current price annotation
        annotations=[
            dict(
                x=1,
                y=current_price,
                xref="paper",
                yref="y",
                text=f"${current_price:.2f}",
                showarrow=False,
                font=dict(
                    size=10,
                    color=current_price_color
                ),
                bgcolor='#1E1E1E',
                bordercolor=current_price_color,
                borderwidth=1,
                borderpad=2,
                xanchor='left',
                yanchor='middle',
                xshift=1  # Moved left
            )
        ]
    )
    
    # Logic to add Exposure Levels to Price Chart
    if exposure_levels_types and calls is not None and puts is not None:
        # Filter options within strike range for better visualization
        range_calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
        range_puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
        
        # Define dash styles to differentiate if multiple types are selected
        dash_styles = ['dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        
        # Pre-calculate all top levels to find the overall absolute maximum for highlighting
        all_top_levels = [] # List of (strike, value, type_name, type_index)
        
        for i, exposure_levels_type in enumerate(exposure_levels_types):
            # --- Expected Move Chart Level ---
            if exposure_levels_type.lower() == 'expected move':
                # --- Weighted Expected Move Calculation ---
                # Find ATM strike (closest to current price)
                strikes_sorted = sorted(calls['strike'].unique()) if not calls.empty else []
                if not strikes_sorted:
                    continue
                atm_strike = min(strikes_sorted, key=lambda x: abs(x - current_price))
                atm_idx = strikes_sorted.index(atm_strike)
                # Helper to get mid price
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
                # ATM Straddle
                call_mid_atm = get_mid(calls, atm_strike)
                put_mid_atm = get_mid(puts, atm_strike)
                straddle = (call_mid_atm if call_mid_atm is not None else 0) + (put_mid_atm if put_mid_atm is not None else 0)
                # Expected Move = ATM Straddle (most common market formula)
                expected_move = straddle
                if expected_move > 0:
                    upper = current_price + expected_move
                    lower = current_price - expected_move
                    em_color = '#036bfc'
                    # Plot dashed lines for expected move in #036bfc
                    fig.add_hline(y=upper, line_dash='dash', line_color=em_color, line_width=2)
                    fig.add_hline(y=lower, line_dash='dash', line_color=em_color, line_width=2)
                    # Add consistent annotation with value
                    fig.add_annotation(
                        x=1, y=upper, xref="paper", yref="y",
                        text=f"EM + {upper:.2f}", showarrow=False,
                        font=dict(size=10, color=em_color),
                        xanchor='left', yanchor='bottom', xshift=-105, yshift=-5
                    )
                    fig.add_annotation(
                        x=1, y=lower, xref="paper", yref="y",
                        text=f"EM - {lower:.2f}", showarrow=False,
                        font=dict(size=10, color=em_color),
                        xanchor='left', yanchor='top', xshift=-105, yshift=5
                    )
                continue

            # Determine column name based on type
            col_name = exposure_levels_type
            if exposure_levels_type == 'Vanna' or exposure_levels_type == 'VEX': col_name = 'VEX'
            if exposure_levels_type == 'AbsGEX': col_name = 'GEX'
            
            # Check if column exists
            if col_name in range_calls.columns and col_name in range_puts.columns:
                # Calculate aggregated exposure for each strike
                call_ex = range_calls.groupby('strike')[col_name].sum().to_dict() if not range_calls.empty else {}
                put_ex = range_puts.groupby('strike')[col_name].sum().to_dict() if not range_puts.empty else {}
                
                levels = {}
                all_strikes = set(call_ex.keys()) | set(put_ex.keys())
                
                for strike in all_strikes:
                    c_val = call_ex.get(strike, 0)
                    p_val = put_ex.get(strike, 0)
                    
                    # Calculate Net Exposure based on type logic
                    if exposure_levels_type == 'GEX':
                        # GEX is Call - Put (puts are positive in calculation)
                        net_val = c_val - p_val
                    elif exposure_levels_type == 'AbsGEX':
                        # Absolute GEX = |Call GEX| + |Put GEX|
                        net_val = abs(c_val) + abs(p_val)
                    elif exposure_levels_type == 'DEX':
                         # DEX: Call + Put. (Puts have negative delta).
                         net_val = c_val + p_val
                    else: 
                         # Others: Call + Put.
                         net_val = c_val + p_val
                    
                    levels[strike] = net_val

                # Sort by absolute exposure and get top levels
                sorted_levels = sorted(levels.items(), key=lambda x: abs(x[1]), reverse=True)
                top_levels = sorted_levels[:exposure_levels_count]
                
                for strike, val in top_levels:
                    all_top_levels.append((strike, val, exposure_levels_type, i))

        # Find the max level independently for EACH exposure type for highlighting
        max_abs_by_type = {}
        if highlight_max_level and all_top_levels:
            for strike, val, etype, tidx in all_top_levels:
                abs_val = abs(val)
                if etype not in max_abs_by_type or abs_val > max_abs_by_type[etype]:
                    max_abs_by_type[etype] = abs_val

        # Draw all collected levels
        for strike, val, exposure_levels_type, type_index in all_top_levels:
            # Pick dash style
            dash_style = dash_styles[type_index % len(dash_styles)]
            
            # Check if this is the maximum level within its own exposure type
            type_max = max_abs_by_type.get(exposure_levels_type, 0)
            is_max_level = highlight_max_level and type_max > 0 and abs(val) == type_max
            
            if is_max_level:
                color = max_level_color
                intensity = 1.0
            else:
                # Determine color: Green for positive, Red for negative
                color = call_color if val >= 0 else put_color
                
                # Calculate color intensity based on coloring mode
                type_max_val = max(abs(l[1]) for l in all_top_levels if l[2] == exposure_levels_type)
                if type_max_val == 0: type_max_val = 1
                if coloring_mode == 'Solid':
                    intensity = 1.0
                elif coloring_mode == 'Ranked Intensity':
                    intensity = 0.1 + 0.9 * ((abs(val) / type_max_val) ** 3)
                else:  # Linear Intensity (default)
                    intensity = 0.3 + 0.7 * (abs(val) / type_max_val)
            
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            rgba_color = f'rgba({r}, {g}, {b}, {intensity:.2f})'
            
            # Add the horizontal line
            fig.add_hline(
                y=strike,
                line_dash=dash_style,
                line_color=rgba_color,
                line_width=2 if is_max_level else 1
            )
            
            # Add separate annotation for the text
            y_offset_pixels = 5 + (type_index * 15)
            
            # Map type to display name
            display_name = exposure_levels_type
            if exposure_levels_type == 'VEX': display_name = 'Vanna'
            if exposure_levels_type == 'AbsGEX': display_name = 'Abs GEX'
            
            display_text = f"<b>{display_name}: {format_large_number(val)}</b>" if is_max_level else f"{display_name}: {format_large_number(val)}"
            
            fig.add_annotation(
                x=1,
                y=strike,
                xref="paper",
                yref="y",
                text=display_text,
                showarrow=False,
                font=dict(
                    size=10,
                    color=rgba_color,
                ),
                textangle=0,
                xanchor='left',
                yanchor='top',
                xshift=-105,
                yshift=-y_offset_pixels
            )

    return fig.to_json()


def prepare_price_chart_data(price_data, calls=None, puts=None, exposure_levels_types=[],
                              exposure_levels_count=3, call_color='#00FF00', put_color='#FF0000',
                              strike_range=0.1, use_heikin_ashi=False,
                              highlight_max_level=False, max_level_color='#800080',
                              coloring_mode='Linear Intensity'):
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
    if not current_day_candles:
        most_recent_date = max(
            datetime.fromtimestamp(c['datetime'] / 1000, est).date() for c in sorted_candles)
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

    # Multi-day raw candles for indicator warmup (SMA200, EMA, etc. need prior-day history)
    lc_indicator_candles = [
        {'time': int(c['datetime'] / 1000), 'open': c['open'], 'high': c['high'],
         'low': c['low'], 'close': c['close'], 'volume': c.get('volume', 0)}
        for c in sorted_candles
    ]
    current_day_start_time = int(current_day_candles[0]['datetime'] / 1000) if current_day_candles else 0

    current_price = display_candles[-1]['close'] if display_candles else 0
    last_candle = display_candles[-1] if display_candles else None
    last_candle_up = (last_candle['close'] >= last_candle['open']) if last_candle else True

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
                strikes_sorted = sorted(calls['strike'].unique()) if not calls.empty else []
                if not strikes_sorted:
                    continue
                atm_strike = min(strikes_sorted, key=lambda x: abs(x - current_price))

                def _get_mid(df, strike):
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

                c_mid = _get_mid(calls, atm_strike)
                p_mid = _get_mid(puts, atm_strike)
                straddle = (c_mid or 0) + (p_mid or 0)
                if straddle > 0:
                    expected_moves.append({
                        'upper': round(current_price + straddle, 2),
                        'lower': round(current_price - straddle, 2)
                    })
                continue

            col_name = etype
            if etype in ('Vanna', 'VEX'):
                col_name = 'VEX'
            if etype == 'AbsGEX':
                col_name = 'GEX'

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
        'exposure_levels': exposure_levels,
        'expected_moves': expected_moves,
        'indicator_candles': lc_indicator_candles,
        'current_day_start_time': current_day_start_time,
    })


def create_large_trades_table(calls, puts, S, strike_range, call_color='#00FF00', put_color='#FF0000', selected_expiries=None):
    """Create a sortable options chain table showing all options within the strike range"""
    # Calculate strike range boundaries
    min_strike = S * (1 - strike_range)
    max_strike = S * (1 + strike_range)
    
    # Filter options within strike range
    calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    def analyze_options(df, is_put=False):
        options = []
        for _, row in df.iterrows():
            options.append({
                'type': 'Put' if is_put else 'Call',
                'strike': float(row['strike']),
                'bid': float(row['bid']),
                'ask': float(row['ask']),
                'last': float(row['lastPrice']),
                'volume': int(row['volume']),
                'openInterest': int(row['openInterest']),
                'iv': float(row['impliedVolatility'])
            })
        return options
    
    # Get options for both calls and puts
    options_chain = analyze_options(calls) + analyze_options(puts, is_put=True)
    
    # Sort by strike price (default)
    options_chain.sort(key=lambda x: x['strike'])
    
    # Add expiry info to title if multiple expiries are selected
    chart_title = 'Options Chain'
    if selected_expiries and len(selected_expiries) > 1:
        chart_title = f"Options Chain ({len(selected_expiries)} expiries)"
    
    # Create HTML table with sorting functionality
    html_content = f'''
    <div style="background-color: #1E1E1E; padding: 10px; border-radius: 10px; height: 100%; overflow: hidden; display: flex; flex-direction: column;">
        <h3 style="color: #CCCCCC; text-align: center; margin: 0 0 10px 0; font-size: 14px;">{chart_title}</h3>
        <div style="flex: 1; overflow: auto;">
            <table id="optionsChainTable" style="width: 100%; border-collapse: collapse; background-color: #1E1E1E; color: white; font-family: Arial, sans-serif; font-size: 10px; table-layout: fixed;">
                <thead>
                    <tr style="background-color: #2D2D2D; position: sticky; top: 0; z-index: 10;">
                        <th onclick="sortTable(0, 'string')" style="padding: 4px 2px; border: 1px solid #444444; cursor: pointer; user-select: none; font-size: 10px; width: 8%;">
                            Type <span style="font-size: 8px;">▼▲</span>
                        </th>
                        <th onclick="sortTable(1, 'number')" style="padding: 4px 2px; border: 1px solid #444444; cursor: pointer; user-select: none; font-size: 10px; width: 12%;">
                            Strike <span style="font-size: 8px;">▼▲</span>
                        </th>
                        <th onclick="sortTable(2, 'number')" style="padding: 4px 2px; border: 1px solid #444444; cursor: pointer; user-select: none; font-size: 10px; width: 12%;">
                            Bid <span style="font-size: 8px;">▼▲</span>
                        </th>
                        <th onclick="sortTable(3, 'number')" style="padding: 4px 2px; border: 1px solid #444444; cursor: pointer; user-select: none; font-size: 10px; width: 12%;">
                            Ask <span style="font-size: 8px;">▼▲</span>
                        </th>
                        <th onclick="sortTable(4, 'number')" style="padding: 4px 2px; border: 1px solid #444444; cursor: pointer; user-select: none; font-size: 10px; width: 12%;">
                            Last <span style="font-size: 8px;">▼▲</span>
                        </th>
                        <th onclick="sortTable(5, 'number')" style="padding: 4px 2px; border: 1px solid #444444; cursor: pointer; user-select: none; font-size: 10px; width: 14%;">
                            Vol <span style="font-size: 8px;">▼▲</span>
                        </th>
                        <th onclick="sortTable(6, 'number')" style="padding: 4px 2px; border: 1px solid #444444; cursor: pointer; user-select: none; font-size: 10px; width: 22%;">
                            OI <span style="font-size: 8px;">▼▲</span>
                        </th>
                        <th onclick="sortTable(7, 'number')" style="padding: 4px 2px; border: 1px solid #444444; cursor: pointer; user-select: none; font-size: 10px; width: 8%;">
                            IV <span style="font-size: 8px;">▼▲</span>
                        </th>
                    </tr>
                </thead>
                <tbody>
    '''
    
    # Add table rows
    for option in options_chain:
        row_color = call_color if option['type'] == 'Call' else put_color
        html_content += f'''
                    <tr style="border-bottom: 1px solid #333333;" onmouseover="this.style.backgroundColor='#333333'" onmouseout="this.style.backgroundColor='transparent'">
                        <td style="padding: 3px 2px; border: 1px solid #444444; color: {row_color}; font-weight: bold; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{option['type'][0]}</td>
                        <td style="padding: 3px 2px; border: 1px solid #444444; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" data-sort="{option['strike']}">{option['strike']:.0f}</td>
                        <td style="padding: 3px 2px; border: 1px solid #444444; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" data-sort="{option['bid']}">{option['bid']:.2f}</td>
                        <td style="padding: 3px 2px; border: 1px solid #444444; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" data-sort="{option['ask']}">{option['ask']:.2f}</td>
                        <td style="padding: 3px 2px; border: 1px solid #444444; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" data-sort="{option['last']}">{option['last']:.2f}</td>
                        <td style="padding: 3px 2px; border: 1px solid #444444; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" data-sort="{option['volume']}">{option['volume']:,}</td>
                        <td style="padding: 3px 2px; border: 1px solid #444444; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" data-sort="{option['openInterest']}">{option['openInterest']:,}</td>
                        <td style="padding: 3px 2px; border: 1px solid #444444; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" data-sort="{option['iv']}">{option['iv']:.0%}</td>
                    </tr>
        '''
    
    html_content += '''
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
    let sortDirection = {};
    
    function sortTable(columnIndex, dataType) {
        const table = document.getElementById('optionsChainTable');
        const tbody = table.tBodies[0];
        const rows = Array.from(tbody.rows);
        
        // Toggle sort direction
        if (!sortDirection[columnIndex]) {
            sortDirection[columnIndex] = 'asc';
        } else {
            sortDirection[columnIndex] = sortDirection[columnIndex] === 'asc' ? 'desc' : 'asc';
        }
        
        const direction = sortDirection[columnIndex];
        
        rows.sort((a, b) => {
            let aVal, bVal;
            
            if (dataType === 'number') {
                aVal = parseFloat(a.cells[columnIndex].getAttribute('data-sort') || a.cells[columnIndex].textContent.replace(/[$,%]/g, ''));
                bVal = parseFloat(b.cells[columnIndex].getAttribute('data-sort') || b.cells[columnIndex].textContent.replace(/[$,%]/g, ''));
                
                if (isNaN(aVal)) aVal = 0;
                if (isNaN(bVal)) bVal = 0;
            } else {
                aVal = a.cells[columnIndex].textContent.toLowerCase();
                bVal = b.cells[columnIndex].textContent.toLowerCase();
            }
            
            if (direction === 'asc') {
                return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
            } else {
                return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
            }
        });
        
        // Clear tbody and append sorted rows
        while (tbody.firstChild) {
            tbody.removeChild(tbody.firstChild);
        }
        
        rows.forEach(row => tbody.appendChild(row));
        
        // Update header indicators
        const headers = table.querySelectorAll('th');
        headers.forEach((header, index) => {
            const span = header.querySelector('span');
            if (index === columnIndex) {
                span.textContent = direction === 'asc' ? '▲' : '▼';
                span.style.color = '#00FF00';
            } else {
                span.textContent = '▼▲';
                span.style.color = '#666';
            }
        });
    }
    </script>
    '''
    
    return html_content





def create_historical_bubble_levels_chart(ticker, strike_range, call_color='#00FFA3', put_color='#FF3B3B', exposure_type='gamma', absolute=False, highlight_max_level=False, max_level_color='#800080'):
    """Create a chart showing price and exposure (gamma, delta, or vanna) over time for the full session.

    Supports optional highlighting of the max exposure bubble via highlight_max_level and max_level_color.
    If absolute is True and exposure_type == 'gamma', gamma exposures are plotted as absolute values (useful for absolute GEX charts).
    """
    # Get interval data; fall back to most recent session if today has no data
    showing_last_session = False
    interval_data = get_interval_data(ticker)

    if not interval_data:
        last_date = get_last_session_date(ticker, 'interval_data')
        if last_date:
            interval_data = get_interval_data(ticker, last_date)
            showing_last_session = True

    if not interval_data:
        return None
    
    # Get the latest price from the most recent data point to establish strike range
    latest_price = interval_data[-1][1]
    min_strike = latest_price * (1 - strike_range)
    max_strike = latest_price * (1 + strike_range)
    
    # Group data by timestamp (show full session, no time filtering)
    data_by_time = {}
    for row in interval_data:
        timestamp = row[0]
            
        price = row[1]
        strike = row[2]
        net_gamma = row[3]
        net_delta = row[4]
        net_vanna = row[5]
        # Check if net_charm exists (for backward compatibility during readout)
        if len(row) > 6:
            net_charm = row[6] if row[6] is not None else 0
        else:
            net_charm = 0
        # Check if abs_gex_total exists (newer DB schema)
        if len(row) > 7:
            abs_gex_total = row[7] if row[7] is not None else None
        else:
            abs_gex_total = None
        
        # Filter strikes based on fixed strike_range relative to latest price
        if strike < min_strike or strike > max_strike:
            continue  # Skip strikes outside the range
        
        if timestamp not in data_by_time:
            data_by_time[timestamp] = {
                'price': price,
                'strikes': []
            }
        
        # Store the exposure value based on the requested type
        exposure = 0
        if exposure_type == 'gamma':
            exposure = net_gamma
        elif exposure_type == 'delta':
            exposure = net_delta
        elif exposure_type == 'vanna':
            exposure = net_vanna
        elif exposure_type == 'charm':
            exposure = net_charm
        
        if exposure is None:
            exposure = 0
        
        # If absolute flag is set for gamma, prefer stored abs_gex_total (call+put magnitudes)
        if absolute and exposure_type == 'gamma':
            if abs_gex_total is not None:
                exposure = abs_gex_total
            else:
                exposure = abs(exposure)
            
        data_by_time[timestamp]['strikes'].append((strike, exposure))
    
    # Convert to lists for plotting
    timestamps = []
    prices = []
    strikes = []
    exposures = []
    
    # Group exposures by timestamp for per-time scaling
    exposures_by_time = {}
    for timestamp, data in data_by_time.items():
        dt = datetime.fromtimestamp(timestamp)
        for strike, exposure in data['strikes']:
            timestamps.append(dt)
            prices.append(data['price'])
            strikes.append(strike)
            exposures.append(exposure)
            if dt not in exposures_by_time:
                exposures_by_time[dt] = []
            exposures_by_time[dt].append(exposure)
    
    # Calculate max exposure for each time slice
    max_exposure_by_time = {dt: max(abs(e) for e in exposures) for dt, exposures in exposures_by_time.items()}
    
    # Create colors and sizes based on per-time scaling
    colors = []
    bubble_sizes = []
    adjusted_strikes = []  # New list for adjusted strike positions
    
    # Group strikes by timestamp to handle overlaps
    strikes_by_time = {}
    for i, (dt, strike) in enumerate(zip(timestamps, strikes)):
        if dt not in strikes_by_time:
            strikes_by_time[dt] = []
        strikes_by_time[dt].append((i, strike))
    
    # Adjust strike positions to prevent overlap
    for dt, strike_data in strikes_by_time.items():
        # Sort strikes for this timestamp
        strike_data.sort(key=lambda x: x[1])
        
        # Group strikes that are close to each other
        groups = []
        current_group = []
        for idx, strike in strike_data:
            if not current_group:
                current_group.append((idx, strike))
            else:
                # If this strike is close to the last one in the group, add it
                if abs(strike - current_group[-1][1]) < 0.1:  # Adjust this threshold as needed
                    current_group.append((idx, strike))
                else:
                    groups.append(current_group)
                    current_group = [(idx, strike)]
        if current_group:
            groups.append(current_group)
        
        # Adjust positions within each group
        for group in groups:
            if len(group) == 1:
                # Single strike, no adjustment needed
                adjusted_strikes.append(group[0][1])
            else:
                # Multiple strikes, spread them out
                center = sum(s for _, s in group) / len(group)
                spread = 0.1  # Adjust this value to control spread
                for i, (idx, strike) in enumerate(group):
                    # Calculate offset based on position in group
                    offset = (i - (len(group) - 1) / 2) * spread
                    adjusted_strikes.append(strike + offset)
    
    # Create colors and sizes for the adjusted strikes
    hover_sides = []
    formatted_exposures = []
    original_strikes = []
    for i, exposure in enumerate(exposures):
        dt = timestamps[i]
        max_exposure = max_exposure_by_time[dt]
        if max_exposure == 0:
            max_exposure = 1  # Prevent division by zero

        # Calculate color and side label
        if absolute and exposure_type == 'gamma':
            colors.append(get_color_with_opacity(exposure, max_exposure, call_color, True))
            hover_sides.append('Total')
        elif exposure >= 0:
            colors.append(get_color_with_opacity(exposure, max_exposure, call_color, True))
            hover_sides.append('Call')
        else:
            colors.append(get_color_with_opacity(exposure, max_exposure, put_color, True))
            hover_sides.append('Put')

        # Calculate bubble size (scaled to the max exposure for this time slice)
        size = max(4, min(25, abs(exposure) * 20 / max_exposure))
        bubble_sizes.append(size)
        formatted_exposures.append(format_large_number(exposure))
        original_strikes.append(strikes[i])

    # If highlight is enabled, mark the max bubble for each timestamp (historical highlighting)
    if highlight_max_level:
        try:
            # Compute local maximum absolute exposure for each timestamp
            local_max_by_dt = {dt: max(abs(v) for v in vals) for dt, vals in exposures_by_time.items()}

            # Prepare a list of line widths to add an outline to highlighted bubbles
            highlight_line_widths = [0] * len(colors)

            # Iterate through each bubble and mark it if it equals the local max for its timestamp
            for idx, (dt, e) in enumerate(zip(timestamps, exposures)):
                local_max = local_max_by_dt.get(dt, 0)
                if local_max > 0 and abs(e) == local_max:
                    colors[idx] = max_level_color
                    highlight_line_widths[idx] = 4
        except Exception as e:
            print(f"Error computing highlight for historical bubble levels: {e}")

    # Create figure
    fig = go.Figure()

    # Add exposure bubbles for each strike first (bottom layer)
    exposure_name = {
        'gamma': 'Gamma',
        'delta': 'Delta',
        'vanna': 'Vanna',
        'charm': 'Charm'
    }.get(exposure_type, 'Exposure')

    # If absolute gamma is requested, adjust the label
    if absolute and exposure_type == 'gamma':
        exposure_name = 'Gamma (Abs)'

    # Build customdata: [side, original_strike, formatted_exposure]
    bubble_customdata = list(zip(hover_sides, original_strikes, formatted_exposures))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=adjusted_strikes,
        mode='markers',
        name=exposure_name,
        marker=dict(
            size=bubble_sizes,
            color=colors,
            opacity=1.0,
            line=dict(width=0)
        ),
        customdata=bubble_customdata,
        hovertemplate='<b>%{customdata[0]}</b><br>Strike: $%{customdata[1]:.2f}<br>' + exposure_name + ': %{customdata[2]}<br>Time: %{x|%H:%M}<extra></extra>',
        yaxis='y1'
    ))

    # If highlight was computed above, apply marker line widths and color for outline
    if highlight_max_level and 'highlight_line_widths' in locals():
        try:
            # Find the bubble trace and update its marker line widths
            for i, trace in enumerate(fig.data):
                if trace.name == exposure_name and 'markers' in trace.mode:
                    fig.data[i].update(marker=dict(line=dict(width=highlight_line_widths, color=max_level_color)))
                    break
        except Exception as e:
            print(f"Error applying highlight to bubble trace: {e}")

    # Add price line last (top layer)
    unique_times = sorted(set(timestamps))
    unique_prices = [data_by_time[int(t.timestamp())]['price'] for t in unique_times]
    fig.add_trace(go.Scatter(
        x=unique_times,
        y=unique_prices,
        mode='lines',
        name='Price',
        line=dict(color='gold', width=2),
        hovertemplate='<b>Price</b>: $%{y:.2f}<br>Time: %{x|%H:%M}<extra></extra>',
        yaxis='y1'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Historical Bubble Levels - {exposure_name}' + (' (Last Session)' if showing_last_session else ''),
            font=dict(color='#CCCCCC', size=16),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time (Full Session)',
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
        showlegend=True,
        autosize=True,
        hovermode='closest',
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



def create_open_interest_chart(calls, puts, S, strike_range=0.02, call_color='#00FF00', put_color='#FF0000', coloring_mode='Solid', show_calls=True, show_puts=True, show_net=True, selected_expiries=None, horizontal=False, highlight_max_level=False, max_level_color='#800080', max_level_mode='Absolute'):
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
                hovertemplate='Strike: %{y}<br>OI: %{text}<extra></extra>',
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
                hovertemplate='Strike: %{x}<br>OI: %{text}<extra></extra>',
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
                hovertemplate='Strike: %{y}<br>OI: %{text}<extra></extra>',
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
                hovertemplate='Strike: %{x}<br>OI: %{text}<extra></extra>',
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
                marker_color=net_colors,
                text=[format_large_number(val) for val in net_oi],
                textposition='auto',
                orientation='h',
                hovertemplate='Strike: %{y}<br>Net OI: %{text}<extra></extra>',
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=all_strikes_list,
                y=net_oi,
                name='Net',
                marker_color=net_colors,
                text=[format_large_number(val) for val in net_oi],
                textposition='auto',
                hovertemplate='Strike: %{x}<br>Net OI: %{text}<extra></extra>',
                marker_line_width=0
            ))
    
    if horizontal:
        fig.add_hline(
            y=S,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text=f"{S:.2f}",
            annotation_position="right",
            annotation_font_color="white",
            line_width=1
        )
    else:
        fig.add_vline(
            x=S,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text=f"{S:.2f}",
            annotation_position="top",
            annotation_font_color="white",
            line_width=1
        )
    
    base_title = 'Open Interest by Strike'
    chart_title = base_title
    if selected_expiries and len(selected_expiries) > 1:
        chart_title = f"{base_title} ({len(selected_expiries)} expiries)"
    
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
        title=dict(
            text=chart_title,
            font=dict(color='#CCCCCC', size=16),
            x=0.5,
            xanchor='center'
        ),
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
        margin=dict(l=50, r=50, t=50, b=100),
        hoverlabel=dict(
            bgcolor='#1E1E1E',
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100,
        showlegend=True,
        height=500
    )
    
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

def create_premium_chart(calls, puts, S, strike_range=0.02, call_color='#00FF00', put_color='#FF0000', coloring_mode='Solid', show_calls=True, show_puts=True, show_net=True, selected_expiries=None, horizontal=False, highlight_max_level=False, max_level_color='#800080', max_level_mode='Absolute'):
    # Filter strikes within range
    min_strike = S * (1 - strike_range)
    max_strike = S * (1 + strike_range)
    
    calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
    puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()
    
    # Determine strike interval and aggregate by rounded strikes
    all_strikes = list(calls['strike']) + list(puts['strike'])
    if all_strikes:
        strike_interval = get_strike_interval(all_strikes)
        calls = aggregate_by_strike(calls, ['lastPrice'], strike_interval)
        puts = aggregate_by_strike(puts, ['lastPrice'], strike_interval)
    
    # Create figure
    fig = go.Figure()
    
    # Calculate max premium for normalization across all data
    max_premium = 1.0
    all_abs_vals = []
    if not calls.empty:
        all_abs_vals.extend(calls['lastPrice'].abs().tolist())
    if not puts.empty:
        all_abs_vals.extend(puts['lastPrice'].abs().tolist())
    if all_abs_vals:
        max_premium = max(all_abs_vals)
    if max_premium == 0:
        max_premium = 1.0
    
    # Add call premium bars
    if show_calls and not calls.empty:
        # Apply coloring mode
        call_colors = get_colors(call_color, calls['lastPrice'], max_premium, coloring_mode)
            
        if horizontal:
            fig.add_trace(go.Bar(
                y=calls['strike'].tolist(),
                x=calls['lastPrice'].tolist(),
                name='Call',
                marker_color=call_colors,
                text=[f"${price:.2f}" for price in calls['lastPrice']],
                textposition='auto',
                orientation='h',
                hovertemplate='Strike: %{y}<br>Premium: $%{x:.2f}<extra></extra>',
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=calls['strike'].tolist(),
                y=calls['lastPrice'].tolist(),
                name='Call',
                marker_color=call_colors,
                text=[f"${price:.2f}" for price in calls['lastPrice']],
                textposition='auto',
                hovertemplate='Strike: %{x}<br>Premium: $%{y:.2f}<extra></extra>',
                marker_line_width=0
            ))
    
    # Add put premium bars
    if show_puts and not puts.empty:
        # Apply coloring mode
        put_colors = get_colors(put_color, puts['lastPrice'], max_premium, coloring_mode)
            
        if horizontal:
            fig.add_trace(go.Bar(
                y=puts['strike'].tolist(),
                x=puts['lastPrice'].tolist(),
                name='Put',
                marker_color=put_colors,
                text=[f"${price:.2f}" for price in puts['lastPrice']],
                textposition='auto',
                orientation='h',
                hovertemplate='Strike: %{y}<br>Premium: $%{x:.2f}<extra></extra>',
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=puts['strike'].tolist(),
                y=puts['lastPrice'].tolist(),
                name='Put',
                marker_color=put_colors,
                text=[f"${price:.2f}" for price in puts['lastPrice']],
                textposition='auto',
                hovertemplate='Strike: %{x}<br>Premium: $%{y:.2f}<extra></extra>',
                marker_line_width=0
            ))
    
    # Add net premium bars if enabled
    if show_net and not (calls.empty and puts.empty):
        # Create net premium by combining calls and puts
        all_strikes_list = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        net_premium = []
        
        for strike in all_strikes_list:
            call_prem = calls[calls['strike'] == strike]['lastPrice'].sum() if not calls.empty else 0
            put_prem = puts[puts['strike'] == strike]['lastPrice'].sum() if not puts.empty else 0
            net_prem = call_prem - put_prem
            
            net_premium.append(net_prem)
        
        # Calculate max for net premium normalization
        max_net_premium = max(abs(min(net_premium)), abs(max(net_premium))) if net_premium else 1.0
        if max_net_premium == 0:
            max_net_premium = 1.0
        
        # Apply coloring mode for net values
        net_colors = get_net_colors(net_premium, max_net_premium, call_color, put_color, coloring_mode)
        
        if horizontal:
            fig.add_trace(go.Bar(
                y=all_strikes_list,
                x=net_premium,
                name='Net',
                marker_color=net_colors,
                text=[f"${prem:.2f}" for prem in net_premium],
                textposition='auto',
                orientation='h',
                hovertemplate='Strike: %{y}<br>Net Premium: $%{x:.2f}<extra></extra>',
                marker_line_width=0
            ))
        else:
            fig.add_trace(go.Bar(
                x=all_strikes,
                y=net_premium,
                name='Net',
                marker_color=net_colors,
                text=[f"${prem:.2f}" for prem in net_premium],
                textposition='auto',
                hovertemplate='Strike: %{x}<br>Net Premium: $%{y:.2f}<extra></extra>',
                marker_line_width=0
            ))
    
    if horizontal:
        # Add current price line
        fig.add_hline(
            y=S,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text=f"{S:.2f}",
            annotation_position="right",
            annotation_font_color="white",
            line_width=1
        )
    else:
        # Add current price line
        fig.add_vline(
            x=S,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text=f"{S:.2f}",
            annotation_position="top",
            annotation_font_color="white",
            line_width=1
        )
    
    # Add expiry info to title if multiple expiries are selected
    chart_title = 'Option Premium by Strike'
    if selected_expiries and len(selected_expiries) > 1:
        chart_title = f"Option Premium by Strike ({len(selected_expiries)} expiries)"
    
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
        title=dict(
            text=chart_title,
            font=dict(color='#CCCCCC', size=16),
            x=0.5,
            xanchor='center'
        ),
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
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#CCCCCC'),
            bgcolor='#1E1E1E'
        ),
        bargap=0.1,
        bargroupgap=0.1,
        margin=dict(l=50, r=50, t=40, b=20),
        hoverlabel=dict(
            bgcolor='#1E1E1E',
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100,
        showlegend=True,
        height=500
    )
    
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
            print(f"Error highlighting max level in premium chart: {e}")

    return fig.to_json()

def create_centroid_chart(ticker, call_color='#00FF00', put_color='#FF0000', selected_expiries=None):
    """Create a chart showing call and put centroids over time with price line"""
    est = pytz.timezone('US/Eastern')
    current_time_est = datetime.now(est)

    def _empty_centroid_chart(title):
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(color='#CCCCCC', size=16), x=0.5, xanchor='center'),
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
    centroid_data = get_centroid_data(ticker)

    if not centroid_data:
        last_date = get_last_session_date(ticker, 'centroid_data')
        if last_date:
            centroid_data = get_centroid_data(ticker, last_date)
            showing_last_session = True

    if not centroid_data:
        if current_time_est.weekday() >= 5:
            return _empty_centroid_chart('Call vs Put Centroid Map (Market Closed - Weekend)')
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
        dt = datetime.fromtimestamp(timestamp)
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
        mode='lines',
        name='Call Centroid',
        line=dict(color=call_color, width=2),
        hovertemplate='Time: %{x}<br>Call Centroid: $%{y:.2f}<br>Call Volume: %{customdata}<extra></extra>',
        customdata=call_volumes,
        connectgaps=False
    ))

    # Add put centroid line (middle layer)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=put_centroids,
        mode='lines',
        name='Put Centroid',
        line=dict(color=put_color, width=2),
        hovertemplate='Time: %{x}<br>Put Centroid: $%{y:.2f}<br>Put Volume: %{customdata}<extra></extra>',
        customdata=put_volumes,
        connectgaps=False
    ))

    # Add price line last (bottom layer)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='gold', width=2),
        hovertemplate='Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add expiry info to title if multiple expiries are selected
    chart_title = 'Call vs Put Centroid Map'
    if selected_expiries and len(selected_expiries) > 1:
        chart_title = f"Call vs Put Centroid Map ({len(selected_expiries)} expiries)"
    if showing_last_session:
        chart_title += ' (Last Session)'
    
    # Update layout to match interval map style
    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(color='#CCCCCC', size=16),
            x=0.5,
            xanchor='center'
        ),
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
        showlegend=True,
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

def fetch_options_for_multiple_dates(ticker, dates, exposure_metric="Open Interest", delta_adjusted: bool = False, calculate_in_notional: bool = True):
    """Fetch options for multiple expiration dates and combine them"""
    all_calls = []
    all_puts = []
    last_exception = None
    
    for date in dates:
        try:
            calls, puts = fetch_options_for_date(ticker, date, exposure_metric=exposure_metric, delta_adjusted=delta_adjusted, calculate_in_notional=calculate_in_notional)
            if not calls.empty:
                all_calls.append(calls)
            if not puts.empty:
                all_puts.append(puts)
        except Exception as e:
            msg = f"Error fetching options for {date}: {e}"
            print(msg)
            last_exception = e
            continue
    
    # Combine all dataframes
    combined_calls = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    combined_puts = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
    # If we couldn't fetch any data and there was an exception, propagate it
    if combined_calls.empty and combined_puts.empty and last_exception is not None:
        raise last_exception

    return combined_calls, combined_puts

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>EzOptions - Schwab</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
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
        }
        
        .price-chart-container {
            /* overridden below by TV chart styles */
        }
        
        .historical-bubbles-row {
            display: grid;
            gap: 5px;
            width: 100%;
            margin-bottom: 5px;
        }
        
        .historical-bubbles-row.one-bubble {
            grid-template-columns: 1fr;
        }
        
        .historical-bubbles-row.two-bubbles {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .historical-bubbles-row.three-bubbles {
            grid-template-columns: repeat(3, 1fr);
        }
        
        .historical-bubbles-row.four-bubbles {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .historical-bubble-container {
            width: 100%;
            height: 500px;
            background-color: #1E1E1E;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .historical-bubble-container .chart-container {
            height: 100%;
            width: 100%;
        }
        
        .chart-container {
            padding: 5px;
            height: 500px;
            width: 100%;
            min-width: 0;
            position: relative;
            background-color: #2D2D2D;
            border-radius: 10px;
            margin-bottom: 5px;
            display: flex;
            flex-direction: column;
        }
        
        .chart-container > div {
            flex: 1;
            width: 100%;
            height: 100%;
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
            height: 680px !important;
            border-radius: 0 0 0 0;
            overflow: hidden;
            /* override .chart-container defaults that conflict */
            margin-bottom: 0 !important;
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
        .tv-toolbar {
            display: contents; /* children flow directly into container */
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
        /* Indicator legend — inside canvas, pointer-events none so it doesn't block */
        .tv-indicator-legend {
            position: absolute;
            bottom: 8px;
            left: 8px;
            display: flex;
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
        /* Drawing mode cursor */
        #price-chart.draw-mode > canvas { cursor: crosshair !important; }
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
            display: flex;
            gap: 15px;
            align-items: center;
            font-size: 1.2em;
            flex-wrap: wrap;
            width: 100%;
        }
        .green {
            color: #00FF00;
        }
        .red {
            color: #FF0000;
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
        }
        
        .charts-grid.one-chart {
            grid-template-columns: 1fr;
        }
        
        .charts-grid.two-charts {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .charts-grid.three-charts {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .charts-grid.four-charts {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .charts-grid.many-charts {
            grid-template-columns: repeat(2, 1fr);
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
            .historical-bubbles-row.two-bubbles,
            .historical-bubbles-row.three-bubbles,
            .historical-bubbles-row.four-bubbles {
                grid-template-columns: 1fr;
            }
            .chart-container {
                height: 350px;
            }
            .historical-bubble-container {
                height: 350px;
            }
            .price-info {
                flex-direction: column;
                align-items: flex-start;
                font-size: 1em;
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
            .historical-bubble-container {
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
        .chart-container:hover .chart-fullscreen-btn,
        .chart-fullscreen-btn:focus {
            opacity: 1;
        }
        .chart-fullscreen-btn:hover {
            background: rgba(80, 80, 80, 0.95);
            color: #fff;
            border-color: #777;
        }
        .chart-container.fullscreen {
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
            overflow: visible !important;
        }
        .chart-container.fullscreen > div {
            width: 100% !important;
            height: 100% !important;
            overflow: visible !important;
        }
        .chart-container.fullscreen .chart-fullscreen-btn {
            opacity: 1;
            position: fixed;
            top: 14px;
            left: 14px;
            z-index: 10001;
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
        .chart-container:hover .chart-popout-btn,
        .chart-popout-btn:focus {
            opacity: 1;
        }
        .chart-popout-btn:hover {
            background: rgba(80, 80, 80, 0.95);
            color: #fff;
            border-color: #777;
        }
        .chart-container.fullscreen .chart-popout-btn {
            opacity: 1;
            position: fixed;
            top: 14px;
            left: 50px;
            z-index: 10001;
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
                        <span class="tm-stats" style="color:#666;font-size:10px;">SCHWAB API</span>
                        <span class="tm-stats" id="tm-access-stat" title="">…</span>
                        <span class="tm-stats" style="color:#444;">·</span>
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
                        <input type="text" id="ticker" placeholder="Enter Ticker" value="SPY" title="Enter a ticker symbol (e.g., SPY, AAPL) or special aggregate tickers: 'MARKET' (SPX base) or 'MARKET2' (SPY base)">
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
                </div>
            </div>
            <div class="header-bottom">
                <div class="controls">
                    <div class="control-group">
                        <label for="strike_range">Strike Range (%):</label>
                        <input type="range" id="strike_range" min="1" max="20" value="2" step="0.5">
                        <span class="range-value" id="strike_range_value">2%</span>
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
                    <div class="control-group">
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
                                <div class="levels-option"><input type="checkbox" value="Speed" id="lvl-Speed"><label for="lvl-Speed">Speed</label></div>
                                <div class="levels-option"><input type="checkbox" value="Vomma" id="lvl-Vomma"><label for="lvl-Vomma">Vomma</label></div>
                                <div class="levels-option"><input type="checkbox" value="Color" id="lvl-Color"><label for="lvl-Color">Color</label></div>
                                <div class="levels-option"><input type="checkbox" value="Expected Move" id="lvl-ExpectedMove"><label for="lvl-ExpectedMove">Expected Move</label></div>
                            </div>
                        </div>
                    </div>
                    <div class="control-group">
                        <label for="levels_count">Top #:</label>
                        <input type="number" id="levels_count" min="1" max="10" value="3" style="width: 50px;">
                    </div>
                    <div class="control-group">
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
        
        <div class="price-info" id="price-info"></div>
        
        <div class="chart-selector">
            <div class="chart-checkbox">
                <input type="checkbox" id="price" checked>
                <label for="price">Price Chart</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="gex_historical_bubble" checked>
                <label for="gex_historical_bubble">GEX Historical Bubble Levels</label>
            </div>
            <div class="chart-checkbox" style="margin-left:12px;">
                <input type="checkbox" id="gex_absolute">
                <label for="gex_absolute">Absolute GEX (bubble)</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="dex_historical_bubble" checked>
                <label for="dex_historical_bubble">DEX Historical Bubble Levels</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="vanna_historical_bubble" checked>
                <label for="vanna_historical_bubble">Vanna Historical Bubble Levels</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="charm_historical_bubble" checked>
                <label for="charm_historical_bubble">Charm Historical Bubble Levels</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="gamma" checked>
                <label for="gamma">Gamma Exposure</label>
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
                <input type="checkbox" id="premium" checked>
                <label for="premium">Premium by Strike</label>
            </div>
            <div class="chart-checkbox">
                <input type="checkbox" id="centroid" checked>
                <label for="centroid">Call vs Put Centroid Map</label>
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
            </div>
            <div class="historical-bubbles-row" id="historical-bubbles-row">
                <!-- Historical bubble levels will be dynamically inserted here -->
            </div>
        </div>
    </div>

    <script>
        let charts = {};
        let updateInterval;
        let lastUpdateTime = 0;
        let callColor = '#00FF00';
        let putColor = '#FF0000';
        let maxLevelColor = '#800080';
        let lastData = {}; // Store last received data
        let lastPriceData = null; // Price chart data stored separately (fetched via /update_price)
        let updateInProgress = false;
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
        // Sub-pane charts for RSI and MACD
        let tvRsiChart = null, tvRsiSeries = null;
        let tvMacdChart = null, tvMacdSeries = {};
        // Persist active indicators across data refreshes
        let tvActiveInds = new Set();
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
        // Debounce timer for Plotly price-line updates (avoid flooding relayout calls)
        let plotlyPriceUpdateTimer = null;

        // List of Plotly chart div IDs that carry a current-price line shape
        const PLOTLY_PRICE_LINE_CHARTS = [
            'gamma-chart', 'delta-chart', 'vanna-chart', 'charm-chart',
            'speed-chart', 'vomma-chart', 'color-chart',
            'options_volume-chart', 'open_interest-chart', 'premium-chart'
        ];

        /**
         * Update the current-price line (shape + annotation) on all visible Plotly charts
         * and refresh the "Current Price" text in the price-info panel.
         */
        function updateAllPlotlyPriceLines(price) {
            const priceStr = price.toFixed(2);

            PLOTLY_PRICE_LINE_CHARTS.forEach(function(id) {
                const div = document.getElementById(id);
                if (!div || !div._fullLayout) return;

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

                if (Object.keys(update).length > 0) {
                    try { Plotly.relayout(div, update); } catch(e) {}
                }
            });

            // Live-update the "Current Price" line in the price-info panel
            const priceInfo = document.getElementById('price-info');
            if (priceInfo) {
                const cpLine = priceInfo.querySelector('[data-live-price]');
                if (cpLine) {
                    cpLine.textContent = 'Current Price: $' + priceStr;
                }
            }
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

        /**
         * Update or extend the chart's current minute candle from a real-time last price.
         * Uses UTC second-aligned minute boundaries to match the chart's time axis.
         */
        function applyRealtimeQuote(last) {
            // Track live price and debounce Plotly chart updates
            livePrice = last;
            clearTimeout(plotlyPriceUpdateTimer);
            plotlyPriceUpdateTimer = setTimeout(function() { updateAllPlotlyPriceLines(last); }, 500);

            if (!tvCandleSeries || !tvLastCandles.length) return;
            const nowSec = Math.floor(Date.now() / 1000);
            const minuteStart = Math.floor(nowSec / 60) * 60;
            const lastCandle = tvLastCandles[tvLastCandles.length - 1];

            if (lastCandle.time === minuteStart) {
                // Update the existing in-progress candle
                const updated = {
                    time:   lastCandle.time,
                    open:   lastCandle.open,
                    high:   Math.max(lastCandle.high, last),
                    low:    Math.min(lastCandle.low,  last),
                    close:  last,
                    volume: lastCandle.volume || 0,
                };
                try { tvCandleSeries.update(updated); } catch(e) {}
                tvLastCandles[tvLastCandles.length - 1] = updated;
                // Keep multi-day indicator candles in sync
                const icLast = tvIndicatorCandles[tvIndicatorCandles.length - 1];
                if (icLast && icLast.time === updated.time) {
                    tvIndicatorCandles[tvIndicatorCandles.length - 1] = updated;
                }
                // Debounce indicator refresh to at most once every 2 seconds on tick updates
                if (tvActiveInds.size > 0) {
                    clearTimeout(tvIndicatorRefreshTimer);
                    tvIndicatorRefreshTimer = setTimeout(() => applyIndicators(tvIndicatorCandles, tvActiveInds), 2000);
                }
            } else if (minuteStart > lastCandle.time) {
                // New minute – open a new candle and immediately refresh indicators
                const newCandle = { time: minuteStart, open: last, high: last, low: last, close: last, volume: 0 };
                try { tvCandleSeries.update(newCandle); } catch(e) {}
                tvLastCandles.push(newCandle);
                tvIndicatorCandles.push(newCandle);
                if (tvActiveInds.size > 0) {
                    clearTimeout(tvIndicatorRefreshTimer);
                    applyIndicators(tvIndicatorCandles, tvActiveInds);
                }
            }
        }

        /**
         * Apply a completed 1-minute candle from CHART_EQUITY streaming.
         */
        function applyRealtimeCandle(candle) {
            if (!tvCandleSeries) return;
            const c = { time: candle.time, open: candle.open, high: candle.high,
                        low: candle.low, close: candle.close, volume: candle.volume || 0 };
            try { tvCandleSeries.update(c); } catch(e) {}
            // Update display candles (current-day)
            const idx = tvLastCandles.findIndex(x => x.time === c.time);
            if (idx >= 0) { tvLastCandles[idx] = c; }
            else { tvLastCandles.push(c); tvLastCandles.sort((a, b) => a.time - b.time); }
            // Update multi-day indicator candles
            const icIdx = tvIndicatorCandles.findIndex(x => x.time === c.time);
            if (icIdx >= 0) { tvIndicatorCandles[icIdx] = c; }
            else { tvIndicatorCandles.push(c); tvIndicatorCandles.sort((a, b) => a.time - b.time); }
            // Refresh indicators with the full multi-day history
            if (tvActiveInds.size > 0) applyIndicators(tvIndicatorCandles, tvActiveInds);
        }

        // --- Fullscreen chart support ---
        const fsExpandSvg = '<svg viewBox="0 0 14 14" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 5V1h4M9 1h4v4M13 9v4H9M5 13H1V9"/></svg>';
        const fsCollapseSvg = '<svg viewBox="0 0 14 14" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M5 1v4H1M9 5h4V1M9 13V9h4M1 9h4v4"/></svg>';

        function toggleChartFullscreen(container) {
            const isFullscreen = container.classList.contains('fullscreen');

            // Exit any other fullscreen chart first
            document.querySelectorAll('.chart-container.fullscreen').forEach(el => {
                el.classList.remove('fullscreen');
                const b = el.querySelector('.chart-fullscreen-btn');
                if (b) b.innerHTML = fsExpandSvg;
            });

            if (!isFullscreen) {
                container.classList.add('fullscreen');
                document.body.style.overflow = 'hidden';
                const b = container.querySelector('.chart-fullscreen-btn');
                if (b) b.innerHTML = fsCollapseSvg;
            } else {
                document.body.style.overflow = '';
            }

            // Let Plotly know about the size change; also trigger TV chart resize
            requestAnimationFrame(() => {
                document.querySelectorAll('.chart-container').forEach(el => {
                    const plot = el.querySelector('.js-plotly-plot');
                    if (plot) { try { Plotly.Plots.resize(plot); } catch(e) {} }
                });
                // Resize TradingView price chart
                const tvContainer = document.getElementById('price-chart');
                if (tvPriceChart && tvContainer) {
                    tvPriceChart.applyOptions({ width: tvContainer.clientWidth });
                }
            });
        }

        function addFullscreenButton(container) {
            if (!container || container.querySelector('.chart-fullscreen-btn')) return;
            const btn = document.createElement('button');
            btn.className = 'chart-fullscreen-btn';
            btn.innerHTML = container.classList.contains('fullscreen') ? fsCollapseSvg : fsExpandSvg;
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
            if (e.key === 'Escape') {
                const fs = document.querySelector('.chart-container.fullscreen');
                if (fs) {
                    fs.classList.remove('fullscreen');
                    document.body.style.overflow = '';
                    const b = fs.querySelector('.chart-fullscreen-btn');
                    if (b) b.innerHTML = fsExpandSvg;
                    requestAnimationFrame(() => {
                        document.querySelectorAll('.chart-container').forEach(el => {
                            const plot = el.querySelector('.js-plotly-plot');
                            if (plot) { try { Plotly.Plots.resize(plot); } catch(e) {} }
                        });
                        const tvContainer = document.getElementById('price-chart');
                        if (tvPriceChart && tvContainer) {
                            tvPriceChart.applyOptions({ width: tvContainer.clientWidth });
                        }
                    });
                }
            }
        });
        // Helper: returns appropriate Plotly margins depending on whether chart is fullscreen
        function getChartMargins(containerId, defaultMargins) {
            const container = document.getElementById(containerId);
            if (container && container.classList.contains('fullscreen')) {
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
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"><\\/script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#1E1E1E; color:#ccc; font-family:Arial,sans-serif; display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  #popout-logo { position:fixed; top:6px; left:10px; z-index:200; font-size:11px; font-weight:bold; color:#800080; opacity:0.7; pointer-events:none; letter-spacing:0.5px; }
  #toolbar { background:#1a1a1a; border-bottom:1px solid #333; padding:4px 8px; display:flex; flex-wrap:wrap; gap:4px; align-items:center; flex-shrink:0; z-index:100; }
  .tv-tb-sep { width:1px; height:20px; background:#444; margin:0 2px; }
  .tb-btn { background:#2a2a2a; border:1px solid #444; color:#ccc; border-radius:4px; padding:3px 7px; font-size:11px; cursor:pointer; white-space:nowrap; transition:background 0.15s; user-select:none; }
  .tb-btn:hover  { background:#3a3a3a; color:#fff; }
  .tb-btn.active { background:#1a5fac; border-color:#4b90e2; color:#fff; }
  .tb-btn.danger { background:#5c1a1a; border-color:#c0392b; color:#f88; }
  #chart-area { flex:1; display:flex; flex-direction:column; min-height:0; position:relative; }
  #price-chart { flex:1; min-height:0; position:relative; }
  .tv-sub-pane { background:#1E1E1E; border-top:1px solid #333; flex-shrink:0; position:relative; }
  .tv-sub-pane-hdr { position:absolute; top:4px; left:8px; z-index:5; font-size:10px; color:#888; font-weight:bold; pointer-events:none; }
  .ind-legend { position:absolute; bottom:8px; left:8px; display:flex; flex-wrap:wrap; gap:6px; z-index:15; pointer-events:none; }
  .ind-item { font-size:10px; color:#ccc; display:flex; align-items:center; gap:4px; }
  .ind-swatch { width:14px; height:3px; border-radius:2px; }
  .title-el { display:inline-block; color:#ccc; font-size:13px; font-weight:bold; padding:2px 8px; pointer-events:none; }
  .candle-close-timer { font-size:11px; font-family:'Courier New',monospace; padding:3px 7px; border-radius:4px; background:#2a2a2a; border:1px solid #444; color:#ccc; white-space:nowrap; user-select:none; letter-spacing:0.5px; }
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
  var tvRsiChart=null, tvRsiSeries=null;
  var tvMacdChart=null, tvMacdSeries={};
  var tvIndSeries={};
  var activeInds=new Set();
  var tvPriceLines=[], tvDrawings=[], tvDrawingDefs=[];
  var tvAllLevelPrices=[];
  var tvLastCandles=[];
  var tvDrawMode=null, tvDrawStart=null;
  var tvAutoRange=false;
  var tvSyncHandlers=[], tvSyncingTS=false;
  var drawColor='#FFD700';
  var lineStyleMap={};
  var popoutTimeframe=1;
  var popoutCandleTimerInterval=null;

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

  // ── Math helpers ───────────────────────────────────────────────────────────
  function calcSMA(c,p){return c.map(function(_,i){if(i<p-1)return null;var s=c.slice(i-p+1,i+1);return s.reduce(function(a,b){return a+b;},0)/p;});}
  function calcEMA(c,p){var k=2/(p+1),r=[],e=null;for(var i=0;i<c.length;i++){if(i<p-1){r.push(null);continue;}if(e===null){e=c.slice(0,p).reduce(function(a,b){return a+b;},0)/p;}else{e=c[i]*k+e*(1-k);}r.push(e);}return r;}
  function calcVWAP(cs){var cp=0,cv=0;return cs.map(function(c){var t=(c.high+c.low+c.close)/3;cp+=t*c.volume;cv+=c.volume;return cv>0?cp/cv:c.close;});}
  function calcBB(c,p,m){p=p||20;m=m||2;var s=calcSMA(c,p);return s.map(function(mid,i){if(mid===null)return{upper:null,mid:null,lower:null};var sl=c.slice(Math.max(0,i-p+1),i+1),v=sl.reduce(function(a,b){return a+(b-mid)*(b-mid);},0)/sl.length,sd=Math.sqrt(v);return{upper:mid+m*sd,mid:mid,lower:mid-m*sd};});}
  function calcRSI(c,p){p=p||14;var r=[];for(var i=0;i<c.length;i++){if(i<p){r.push(null);continue;}var g=0,l=0;for(var j=i-p+1;j<=i;j++){var d=c[j]-c[j-1];if(d>0)g+=d;else l-=d;}var ag=g/p,al=l/p;r.push(al===0?100:100-100/(1+ag/al));}return r;}
  function calcATR(candles,p){p=p||14;var r=[];for(var i=0;i<candles.length;i++){var tr;if(i===0){tr=candles[i].high-candles[i].low;}else{tr=Math.max(candles[i].high-candles[i].low,Math.abs(candles[i].high-candles[i-1].close),Math.abs(candles[i].low-candles[i-1].close));}if(i<p-1){r.push(null);continue;}if(r.length===0||r[r.length-1]===null){var sum=0;for(var j=i-p+1;j<=i;j++){var t2;if(j===0){t2=candles[j].high-candles[j].low;}else{t2=Math.max(candles[j].high-candles[j].low,Math.abs(candles[j].high-candles[j-1].close),Math.abs(candles[j].low-candles[j-1].close));}sum+=t2;}r.push(sum/p);}else{r.push((r[r.length-1]*(p-1)+tr)/p);}}return r;}
  function calcMACD(c,fast,slow,sig){fast=fast||12;slow=slow||26;sig=sig||9;var ef=calcEMA(c,fast),es=calcEMA(c,slow);var ml=ef.map(function(v,i){return(v!==null&&es[i]!==null)?v-es[i]:null;});var sl=[],es2=null,vi=0,k=2/(sig+1);for(var i=0;i<ml.length;i++){if(ml[i]===null){sl.push(null);continue;}if(vi<sig-1){sl.push(null);vi++;continue;}if(es2===null){var piece=ml.filter(function(v){return v!==null;}).slice(0,sig);es2=piece.reduce(function(a,b){return a+b;},0)/sig;}else{es2=ml[i]*k+es2*(1-k);}sl.push(es2);vi++;}return{macd:ml,signal:sl,histogram:ml.map(function(v,i){return(v!==null&&sl[i]!==null)?v-sl[i]:null;})};}

  // ── Sub-pane chart factory ─────────────────────────────────────────────────
  function mkSubChart(el,h){return LightweightCharts.createChart(el,{autoSize:true,height:h,layout:{background:{color:'#1E1E1E'},textColor:'#CCCCCC',fontFamily:'Arial,sans-serif'},grid:{vertLines:{color:'#2A2A2A'},horzLines:{color:'#2A2A2A'}},crosshair:{mode:LightweightCharts.CrosshairMode.Normal,vertLine:{color:'#555',labelBackgroundColor:'#2D2D2D'},horzLine:{color:'#555',labelBackgroundColor:'#2D2D2D'}},rightPriceScale:{borderColor:'#333',scaleMargins:{top:0.1,bottom:0.1}},timeScale:{borderColor:'#333',timeVisible:false,secondsVisible:false,fixLeftEdge:true,fixRightEdge:false},handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true},handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false}});}

  // ── Time-scale sync ────────────────────────────────────────────────────────
  function setupSync(){tvSyncHandlers.forEach(function(h){try{h.chart.timeScale().unsubscribeVisibleLogicalRangeChange(h.handler);}catch(e){}});tvSyncHandlers=[];var all=[tvChart,tvRsiChart,tvMacdChart].filter(Boolean);if(all.length<2)return;all.forEach(function(src){var others=all.filter(function(c){return c!==src;});var h=function(range){if(tvSyncingTS||!range)return;tvSyncingTS=true;others.forEach(function(c){try{c.timeScale().setVisibleLogicalRange(range);}catch(e){}});tvSyncingTS=false;};try{src.timeScale().subscribeVisibleLogicalRangeChange(h);}catch(e){}tvSyncHandlers.push({chart:src,handler:h});});if(tvChart){try{var r=tvChart.timeScale().getVisibleLogicalRange();if(r)[tvRsiChart,tvMacdChart].filter(Boolean).forEach(function(c){try{c.timeScale().setVisibleLogicalRange(r);}catch(e){}});}catch(e){}}}

  // ── Indicators ─────────────────────────────────────────────────────────────
  function applyIndicators(candles){
    if(!tvChart||!tvCandle)return;
    var times=candles.map(function(c){return c.time;}),closes=candles.map(function(c){return c.close;});
    function mkLine(col,lw,title){return tvChart.addLineSeries({color:col,lineWidth:lw||1,priceScaleId:'right',lastValueVisible:true,priceLineVisible:false,title:title||''});}
    // Remove deactivated
    Object.keys(tvIndSeries).forEach(function(k){if(!activeInds.has(k)){var s=tvIndSeries[k];if(Array.isArray(s))s.forEach(function(x){try{tvChart.removeSeries(x);}catch(e){}});else{try{tvChart.removeSeries(s);}catch(e){};}delete tvIndSeries[k];}});
    // Add activated
    if(activeInds.has('sma20')&&!tvIndSeries['sma20']){var s=mkLine('#f0c040',1,'SMA20');s.setData(calcSMA(closes,20).map(function(v,i){return v!==null?{time:times[i],value:v}:null;}).filter(Boolean));tvIndSeries['sma20']=s;}
    if(activeInds.has('sma50')&&!tvIndSeries['sma50']){var s=mkLine('#40a0f0',1,'SMA50');s.setData(calcSMA(closes,50).map(function(v,i){return v!==null?{time:times[i],value:v}:null;}).filter(Boolean));tvIndSeries['sma50']=s;}
    if(activeInds.has('sma200')&&!tvIndSeries['sma200']){var s=mkLine('#e040fb',1,'SMA200');s.setData(calcSMA(closes,200).map(function(v,i){return v!==null?{time:times[i],value:v}:null;}).filter(Boolean));tvIndSeries['sma200']=s;}
    if(activeInds.has('ema9')&&!tvIndSeries['ema9']){var s=mkLine('#ff9900',1,'EMA9');s.setData(calcEMA(closes,9).map(function(v,i){return v!==null?{time:times[i],value:v}:null;}).filter(Boolean));tvIndSeries['ema9']=s;}
    if(activeInds.has('ema21')&&!tvIndSeries['ema21']){var s=mkLine('#00e5ff',1,'EMA21');s.setData(calcEMA(closes,21).map(function(v,i){return v!==null?{time:times[i],value:v}:null;}).filter(Boolean));tvIndSeries['ema21']=s;}
    if(activeInds.has('vwap')&&!tvIndSeries['vwap']){var vv=calcVWAP(candles.map(function(c,i){return{high:candles[i].high,low:candles[i].low,close:candles[i].close,volume:c.volume||0};}));var s=mkLine('#ffffff',1,'VWAP');s.setData(vv.map(function(v,i){return{time:times[i],value:v};}));tvIndSeries['vwap']=s;}
    if(activeInds.has('bb')&&!tvIndSeries['bb']){var bb=calcBB(closes);var u=mkLine('rgba(100,180,255,0.8)',1,'BB U'),m=mkLine('rgba(100,180,255,0.5)',1,'BB M'),l=mkLine('rgba(100,180,255,0.8)',1,'BB L');u.setData(bb.map(function(v,i){return v.upper!==null?{time:times[i],value:v.upper}:null;}).filter(Boolean));m.setData(bb.map(function(v,i){return v.mid!==null?{time:times[i],value:v.mid}:null;}).filter(Boolean));l.setData(bb.map(function(v,i){return v.lower!==null?{time:times[i],value:v.lower}:null;}).filter(Boolean));tvIndSeries['bb']=[u,m,l];}
    if(activeInds.has('atr')&&!tvIndSeries['atr']){var atrV=calcATR(candles),e20=calcEMA(closes,20),mult=1.5;var au=mkLine('rgba(255,152,0,0.8)',1,'ATR U'),al=mkLine('rgba(255,152,0,0.8)',1,'ATR L');au.setData(e20.map(function(v,i){return(v!==null&&atrV[i]!==null)?{time:times[i],value:v+mult*atrV[i]}:null;}).filter(Boolean));al.setData(e20.map(function(v,i){return(v!==null&&atrV[i]!==null)?{time:times[i],value:v-mult*atrV[i]}:null;}).filter(Boolean));tvIndSeries['atr']=[au,al];}
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
    var cols={sma20:'#f0c040',sma50:'#40a0f0',sma200:'#e040fb',ema9:'#ff9900',ema21:'#00e5ff',vwap:'#ffffff',bb:'rgba(100,180,255,0.8)',rsi:'#e91e63',macd:'#2196f3',atr:'rgba(255,152,0,0.8)'};
    var lbls={sma20:'SMA20',sma50:'SMA50',sma200:'SMA200',ema9:'EMA9',ema21:'EMA21',vwap:'VWAP',bb:'BB(20,2)',rsi:'RSI14',macd:'MACD',atr:'ATR Bands'};
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
    if(tvDrawMode==='text'){var txt=prompt('Enter label text:');if(!txt)return;var l=tvCandle.createPriceLine({price:price,color:drawColor,lineWidth:0,lineStyle:LS.Solid,axisLabelVisible:true,title:txt});l._isLine=true;tvDrawings.push(l);tvDrawingDefs.push({type:'text',price:price,text:txt,color:drawColor});}
  }

  // ── Toolbar ────────────────────────────────────────────────────────────────
  function buildToolbar(candles,upColor,downColor){
    var tb=document.getElementById('toolbar');
    // Remove everything except the title and sep (first 2 children)
    while(tb.children.length>2)tb.removeChild(tb.lastChild);
    function btn(text,title,onClick,extra){var b=document.createElement('button');b.className='tb-btn'+(extra?' '+extra:'');b.textContent=text;b.title=title;b.addEventListener('click',onClick);return b;}
    function sep(){var d=document.createElement('div');d.className='tv-tb-sep';return d;}
    // Indicator toggles
    var inds=[{k:'sma20',l:'SMA20',t:'SMA 20'},{k:'sma50',l:'SMA50',t:'SMA 50'},{k:'sma200',l:'SMA200',t:'SMA 200'},{k:'ema9',l:'EMA9',t:'EMA 9'},{k:'ema21',l:'EMA21',t:'EMA 21'},{k:'vwap',l:'VWAP',t:'VWAP'},{k:'bb',l:'BB',t:'Bollinger Bands (20,2)'},{k:'rsi',l:'RSI',t:'RSI 14 — sub-pane'},{k:'macd',l:'MACD',t:'MACD (12,26,9) — sub-pane'},{k:'atr',l:'ATR',t:'Average True Range 14 — sub-pane'}];
    inds.forEach(function(def){var b=btn(def.l,def.t,function(){if(activeInds.has(def.k))activeInds.delete(def.k);else activeInds.add(def.k);b.classList.toggle('active',activeInds.has(def.k));applyIndicators(tvLastCandles);});if(activeInds.has(def.k))b.classList.add('active');tb.appendChild(b);});
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

  // ── Main renderer ──────────────────────────────────────────────────────────
  var isFirstRender=true;
  function renderPriceChart(priceData){
    var candles=priceData.candles||[];
    var upColor=priceData.call_color||'#00FF00',downColor=priceData.put_color||'#FF0000';
    popoutTimeframe=parseInt(priceData.timeframe)||1;
    lineStyleMap={dashed:LightweightCharts.LineStyle.Dashed,dotted:LightweightCharts.LineStyle.Dotted,large_dashed:LightweightCharts.LineStyle.LargeDashed};
    if(!tvChart){
      var el=document.getElementById('price-chart');
      tvChart=LightweightCharts.createChart(el,{autoSize:true,layout:{background:{color:'#1E1E1E'},textColor:'#CCCCCC',fontFamily:'Arial,sans-serif'},grid:{vertLines:{color:'#2A2A2A'},horzLines:{color:'#2A2A2A'}},crosshair:{mode:LightweightCharts.CrosshairMode.Normal,vertLine:{color:'#555',labelBackgroundColor:'#2D2D2D'},horzLine:{color:'#555',labelBackgroundColor:'#2D2D2D'}},rightPriceScale:{borderColor:'#333',scaleMargins:{top:0.04,bottom:0.15}},localization:{timeFormatter:function(time){var d=new Date(time*1000);return d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/New_York'});}},timeScale:{borderColor:'#333',timeVisible:true,secondsVisible:false,fixLeftEdge:false,fixRightEdge:false,tickMarkFormatter:function(time){var d=new Date(time*1000);return d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/New_York'});}},handleScale:{mouseWheel:true,pinch:true,axisPressedMouseMove:true},handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false}});
      tvCandle=tvChart.addCandlestickSeries({upColor:upColor,downColor:downColor,borderVisible:false,wickUpColor:upColor,wickDownColor:downColor});
      tvVol=tvChart.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'volume',lastValueVisible:false,priceLineVisible:false});
      tvChart.priceScale('volume').applyOptions({scaleMargins:{top:0.88,bottom:0}});
      document.getElementById('chart-title').textContent=priceData.use_heikin_ashi?'Price Chart (Heikin-Ashi)':'Price Chart';
      buildToolbar(candles,upColor,downColor);
    } else {
      tvCandle.applyOptions({upColor:upColor,downColor:downColor,wickUpColor:upColor,wickDownColor:downColor});
    }
    tvCandle.setData(candles);
    tvVol.setData(priceData.volume||[]);
    tvLastCandles=candles;
    // Exposure & expected-move price lines
    tvPriceLines.forEach(function(l){try{tvCandle.removePriceLine(l);}catch(e){}});tvPriceLines=[];tvAllLevelPrices=[];
    (priceData.exposure_levels||[]).forEach(function(lv){tvAllLevelPrices.push(lv.price);tvPriceLines.push(tvCandle.createPriceLine({price:lv.price,color:lv.color,lineWidth:lv.line_width||1,lineStyle:lineStyleMap[lv.dash_style]||LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:lv.label}));});
    (priceData.expected_moves||[]).forEach(function(em){tvAllLevelPrices.push(em.upper,em.lower);tvPriceLines.push(tvCandle.createPriceLine({price:em.upper,color:'#036bfc',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:'EM+'}));tvPriceLines.push(tvCandle.createPriceLine({price:em.lower,color:'#036bfc',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:'EM-'}));});
    tvApplyAutoscale();
    if(activeInds.size>0)applyIndicators(candles);
    if(isFirstRender||tvAutoRange){fitAll();isFirstRender=false;}
  }

  // ── Real-time quote / candle application ─────────────────────────────────
  function applyRealtimeQuote(last){
    if(!tvCandle||!tvLastCandles.length)return;
    var nowSec=Math.floor(Date.now()/1000);
    var minuteStart=Math.floor(nowSec/60)*60;
    var lc=tvLastCandles[tvLastCandles.length-1];
    if(lc.time===minuteStart){
      var updated={time:lc.time,open:lc.open,high:Math.max(lc.high,last),low:Math.min(lc.low,last),close:last,volume:lc.volume||0};
      try{tvCandle.update(updated);}catch(e){}
      tvLastCandles[tvLastCandles.length-1]=updated;
    }else if(minuteStart>lc.time){
      var newC={time:minuteStart,open:last,high:last,low:last,close:last,volume:0};
      try{tvCandle.update(newC);}catch(e){}
      tvLastCandles.push(newC);
    }
  }
  function applyRealtimeCandle(candle){
    if(!tvCandle)return;
    var c={time:candle.time,open:candle.open,high:candle.high,low:candle.low,close:candle.close,volume:candle.volume||0};
    try{tvCandle.update(c);}catch(e){}
    var idx=tvLastCandles.findIndex(function(x){return x.time===c.time;});
    if(idx>=0){tvLastCandles[idx]=c;}else{tvLastCandles.push(c);tvLastCandles.sort(function(a,b){return a.time-b.time;});}
    if(activeInds.size>0)applyIndicators(tvLastCandles);
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
      function chk(id){var el=d.getElementById(id);return el?el.checked:false;}
      var ticker=val('ticker');if(!ticker)return null;
      var levelsTypes=[];
      try{levelsTypes=Array.from(d.querySelectorAll('.levels-option input:checked')).map(function(cb){return cb.value;});}catch(e){}
      return{ticker:ticker,timeframe:val('timeframe')||'1',call_color:val('call_color')||'#00ff00',put_color:val('put_color')||'#ff0000',levels_types:levelsTypes,levels_count:parseInt(val('levels_count'))||3,use_heikin_ashi:chk('use_heikin_ashi'),strike_range:parseFloat(val('strike_range'))/100||0.1,highlight_max_level:chk('highlight_max_level'),max_level_color:val('max_level_color')||'#800080',coloring_mode:val('coloring_mode')||'Linear Intensity'};
    }catch(e){return null;}
  }
  function loadInitialData(){
    if(popoutFetching)return;
    var settings=getSettingsFromOpener();
    if(!settings||!settings.ticker)return;
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
  var popoutExpLevelTimer=null;
  function tickerWatchLoop(){
    var settings=getSettingsFromOpener();
    var ticker=settings?settings.ticker:null;
    if(ticker&&ticker!==popoutCurrentTicker){
      popoutCurrentTicker=ticker;
      loadInitialData();
    }
  }
  // Refresh exposure levels every 60 s (options cache updated by main /update cycle)
  function refreshExposureLevels(){
    if(popoutFetching||!popoutCurrentTicker)return;
    var settings=getSettingsFromOpener();
    if(!settings)return;
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
            (pd.exposure_levels||[]).forEach(function(lv){tvAllLevelPrices.push(lv.price);tvPriceLines.push(tvCandle.createPriceLine({price:lv.price,color:lv.color,lineWidth:lv.line_width||1,lineStyle:lineStyleMap[lv.dash_style]||LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:lv.label}));});
            (pd.expected_moves||[]).forEach(function(em){tvAllLevelPrices.push(em.upper,em.lower);tvPriceLines.push(tvCandle.createPriceLine({price:em.upper,color:'#036bfc',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:'EM+'}));tvPriceLines.push(tvCandle.createPriceLine({price:em.lower,color:'#036bfc',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:'EM-'}));});
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
    try{var priceData=typeof priceDataJSON==='string'?JSON.parse(priceDataJSON):priceDataJSON;if(!priceData||priceData.error)return;renderPriceChart(priceData);}catch(e){console.error('Popout price chart error:',e);}
  };
  window.addEventListener('resize',function(){if(tvChart&&tvAutoRange){try{tvChart.timeScale().fitContent();}catch(e){}}});
  window.addEventListener('beforeunload',function(){if(popoutEvtSource){try{popoutEvtSource.close();}catch(e){}}});
<\\/script></body></html>`);
            } else {
                popup.document.write(`<!DOCTYPE html>
<html><head><title>${displayName} - EzOptions</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"><\\/script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1E1E1E; overflow: hidden; position: relative; }
  #popout-logo { position: fixed; top: 6px; left: 10px; z-index: 100; font-family: Arial, sans-serif; font-size: 11px; font-weight: bold; color: #800080; opacity: 0.7; pointer-events: none; letter-spacing: 0.5px; }
  #popout-plot { width: 100vw; height: 100vh; }
  #popout-html { width: 100vw; height: 100vh; overflow: auto; background: #1E1E1E; color: white; font-family: Arial, sans-serif; }
</style></head><body>
<div id="popout-logo">EzDuz1t Options</div>
<div id="popout-plot"></div>
<div id="popout-html" style="display:none;"></div>
<script>
  let plotInited = false;
  window.updatePopoutChart = function(chartDataJSON, isHtml) {
    if (isHtml) {
      document.getElementById('popout-plot').style.display = 'none';
      const htmlDiv = document.getElementById('popout-html');
      htmlDiv.style.display = 'block';
      htmlDiv.innerHTML = chartDataJSON;
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
      chartData.layout.margin = { l: 60, r: 130, t: 60, b: 40 };
      const config = { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d'], displaylogo: false, scrollZoom: true };
      if (plotInited) {
        Plotly.react('popout-plot', chartData.data, chartData.layout, config);
      } else {
        Plotly.newPlot('popout-plot', chartData.data, chartData.layout, config);
        plotInited = true;
      }
    } catch(e) { console.error('Popout chart error:', e); }
  };
  window.addEventListener('resize', function() {
    const el = document.getElementById('popout-plot');
    if (el && el.querySelector('.js-plotly-plot')) { try { Plotly.Plots.resize(el); } catch(e) {} }
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
            const chartPayload = (dataKey === 'price') ? lastPriceData : lastData[dataKey];
            if (!chartPayload) return;

            const isHtml = (dataKey === 'large_trades');
            try {
                popup.updatePopoutChart(chartPayload, isHtml);
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

        document.getElementById('highlight_max_level').addEventListener('change', updateData);
        document.getElementById('max_level_mode').addEventListener('change', updateData);
        document.getElementById('gex_absolute').addEventListener('change', updateData);
        
        // Helper function to create rgba color with opacity
        function createRgbaColor(hexColor, opacity) {
            const r = parseInt(hexColor.slice(1, 3), 16);
            const g = parseInt(hexColor.slice(3, 5), 16);
            const b = parseInt(hexColor.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${opacity})`;
        }
        
        // Update strike range value display
        document.getElementById('strike_range').addEventListener('input', function() {
            document.getElementById('strike_range_value').textContent = this.value + '%';
            updateData();
        });

        // Coloring mode listeners
        document.getElementById('timeframe').addEventListener('change', function() {
            updateData();
            startCandleCloseTimer();
        });
        document.getElementById('coloring_mode').addEventListener('change', updateData);
        document.getElementById('exposure_metric').addEventListener('change', updateData);
        document.getElementById('levels_count').addEventListener('input', updateData);
        document.getElementById('abs_gex_opacity').addEventListener('input', updateData);

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

        document.getElementById('levels-display').addEventListener('click', function(e) {
            e.stopPropagation();
            const options = document.getElementById('levels-options');
            options.classList.toggle('open');
        });
        
        // Add event listeners for level checkboxes
        document.querySelectorAll('.levels-option input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                updateLevelsDisplay();
                updateData();
            });
        });
        
        function updateData() {
            if (updateInProgress) {
                return; // Skip if an update is already in progress
            }
            
            updateInProgress = true;
            
            const ticker = document.getElementById('ticker').value;
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
                // Disconnect the price stream so it reconnects on the new ticker
                disconnectPriceStream();
            }
            tvLastTicker = ticker;

            const selectedCheckboxes = document.querySelectorAll('.expiry-option input[type="checkbox"]:checked');
            const expiry = Array.from(selectedCheckboxes).map(checkbox => checkbox.value);
            
            // Ensure at least one expiry is selected
            if (expiry.length === 0) {
                console.warn('No expiry selected, skipping update');
                updateInProgress = false;
                return;
            }
            const showCalls = document.getElementById('show_calls').checked;
            const showPuts = document.getElementById('show_puts').checked;
            const showNet = document.getElementById('show_net').checked;
            const coloringMode = document.getElementById('coloring_mode').value;
            const levelsTypes = Array.from(document.querySelectorAll('.levels-option input:checked')).map(cb => cb.value);
            const levelsCount = parseInt(document.getElementById('levels_count').value);
            const useHeikinAshi = document.getElementById('use_heikin_ashi').checked;
            const horizontalBars = document.getElementById('horizontal_bars').checked;
            const showAbsGex = document.getElementById('show_abs_gex').checked;
            const absGexOpacity = parseInt(document.getElementById('abs_gex_opacity').value) / 100;
            const useRange = document.getElementById('use_range').checked;
            const exposureMetric = document.getElementById('exposure_metric').value;
            const deltaAdjusted = document.getElementById('delta_adjusted_exposures').checked;
            const calculateInNotional = document.getElementById('calculate_in_notional').checked;
            const strikeRange = parseFloat(document.getElementById('strike_range').value) / 100;
            const highlightMaxLevel = document.getElementById('highlight_max_level').checked;
            const maxLevelMode = document.getElementById('max_level_mode').value;
            
            // Get visible charts
            const gexAbsolute = document.getElementById('gex_absolute').checked;
            const visibleCharts = {
                show_price: document.getElementById('price').checked,
                show_gex_historical_bubble: document.getElementById('gex_historical_bubble').checked,
                show_dex_historical_bubble: document.getElementById('dex_historical_bubble').checked,
                show_vanna_historical_bubble: document.getElementById('vanna_historical_bubble').checked,
                show_charm_historical_bubble: document.getElementById('charm_historical_bubble').checked,
                show_gamma: document.getElementById('gamma').checked,
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
                show_premium: document.getElementById('premium').checked,
                show_centroid: document.getElementById('centroid').checked
            };

            // Common payload fields shared by both requests
            const sharedPayload = {
                ticker,
                timeframe: document.getElementById('timeframe').value,
                call_color: callColor,
                put_color: putColor,
                levels_types: levelsTypes,
                levels_count: levelsCount,
                use_heikin_ashi: useHeikinAshi,
                strike_range: strikeRange,
                highlight_max_level: highlightMaxLevel,
                max_level_color: maxLevelColor,
                coloring_mode: coloringMode
            };

            // Fetch price history: immediate on ticker/settings change, throttled to 30s otherwise.
            // Real-time candle ticks come from SSE (connectPriceStream), not from polling.
            if (visibleCharts.show_price) {
                fetchPriceHistory(tickerChanged || !tvLastCandles.length);
            }
            
            fetch('/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    ticker, 
                    expiry,
                    timeframe: document.getElementById('timeframe').value,
                    show_calls: showCalls,
                    show_puts: showPuts,
                    show_net: showNet,
                    coloring_mode: coloringMode,
                    levels_types: levelsTypes,
                    levels_count: levelsCount,
                    use_heikin_ashi: useHeikinAshi,
                    horizontal_bars: horizontalBars,
                    show_abs_gex: showAbsGex,
                    abs_gex_opacity: absGexOpacity,
                    use_range: useRange,
                    exposure_metric: exposureMetric,
                    delta_adjusted: deltaAdjusted,
                    calculate_in_notional: calculateInNotional,
                    strike_range: strikeRange,
                    call_color: callColor,
                    put_color: putColor,
                    highlight_max_level: highlightMaxLevel,
                    max_level_color: maxLevelColor,
                    max_level_mode: maxLevelMode,
                    absolute_gex: gexAbsolute,
                    show_price: false,  // price is fetched independently via /update_price
                    ...visibleCharts
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showError(data.error);
                    // Pause streaming on persistent error
                    if (isStreaming) {
                        toggleStreaming();
                    }
                    return;
                }
                
                // Only update if data has changed
                if (JSON.stringify(data) !== JSON.stringify(lastData)) {
                    lastData = data;  // Update before rendering so popout windows get fresh data
                    updateCharts(data);
                    updatePriceInfo(data.price_info);
                }
                // Options cache is now populated — refresh price levels immediately.
                // This fixes the delay where levels were missing right after a ticker change
                // because /update_price fired before the options chain was cached.
                if (document.getElementById('price').checked) {
                    _priceHistoryLastKey = ''; // force cache-miss so fetchPriceHistory re-fetches
                    fetchPriceHistory(true);
                }
            })
            .catch(error => {
                showError('Network Error: Could not connect to the server.');
                if (isStreaming) {
                    toggleStreaming();
                }
                console.error('Error fetching data:', error);
            })
            .finally(() => {
                updateInProgress = false;
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
                const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
                result.push(100 - 100 / (1 + rs));
            }
            return result;
        }
        function calcMACD(closes, fast=12, slow=26, signal=9) {
            const emaFast   = calcEMA(closes, fast);
            const emaSlow   = calcEMA(closes, slow);
            const macdLine  = emaFast.map((v, i) => (v !== null && emaSlow[i] !== null) ? v - emaSlow[i] : null);
            const validMACD = macdLine.filter(v => v !== null);
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

        // ── Apply/remove indicators on existing chart ─────────────────────────
        function applyIndicators(candles, activeInds) {
            if (!tvPriceChart || !tvCandleSeries) return;
            const times  = candles.map(c => c.time);
            const closes = candles.map(c => c.close);
            const highs  = candles.map(c => c.high);
            const lows   = candles.map(c => c.low);

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

            if (activeInds.has('sma20')) {
                if (!tvIndicatorSeries['sma20']) tvIndicatorSeries['sma20'] = mkLineSeries('#f0c040', 1, 'right', 'SMA20');
                tvIndicatorSeries['sma20'].setData(todayOnly(calcSMA(closes, 20).map((v,i) => v!==null ? {time:times[i], value:v} : null)));
            }
            if (activeInds.has('sma50')) {
                if (!tvIndicatorSeries['sma50']) tvIndicatorSeries['sma50'] = mkLineSeries('#40a0f0', 1, 'right', 'SMA50');
                tvIndicatorSeries['sma50'].setData(todayOnly(calcSMA(closes, 50).map((v,i) => v!==null ? {time:times[i], value:v} : null)));
            }
            if (activeInds.has('sma200')) {
                if (!tvIndicatorSeries['sma200']) tvIndicatorSeries['sma200'] = mkLineSeries('#e040fb', 1, 'right', 'SMA200');
                tvIndicatorSeries['sma200'].setData(todayOnly(calcSMA(closes, 200).map((v,i) => v!==null ? {time:times[i], value:v} : null)));
            }
            if (activeInds.has('ema9')) {
                if (!tvIndicatorSeries['ema9']) tvIndicatorSeries['ema9'] = mkLineSeries('#ff9900', 1, 'right', 'EMA9');
                tvIndicatorSeries['ema9'].setData(todayOnly(calcEMA(closes, 9).map((v,i) => v!==null ? {time:times[i], value:v} : null)));
            }
            if (activeInds.has('ema21')) {
                if (!tvIndicatorSeries['ema21']) tvIndicatorSeries['ema21'] = mkLineSeries('#00e5ff', 1, 'right', 'EMA21');
                tvIndicatorSeries['ema21'].setData(todayOnly(calcEMA(closes, 21).map((v,i) => v!==null ? {time:times[i], value:v} : null)));
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
                const bb = calcBB(closes);
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
            if (activeInds.has('atr')) {
                const atrVals = calcATR(candles);
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

            // RSI and MACD sub-panes: compute with full history but display today only
            const todayCandles = dayStart > 0 ? candles.filter(c => c.time >= dayStart) : candles;
            if (activeInds.has('rsi')) applyRsiPane(candles, todayCandles);
            else                       destroyRsiPane();
            if (activeInds.has('macd')) applyMacdPane(candles, todayCandles);
            else                        destroyMacdPane();

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
            const items = {
                sma20:'SMA20',sma50:'SMA50',sma200:'SMA200',
                ema9:'EMA9',ema21:'EMA21',vwap:'VWAP',bb:'BB(20,2)',rsi:'RSI14',macd:'MACD',atr:'ATR Bands'
            };
            const colors = {
                sma20:'#f0c040',sma50:'#40a0f0',sma200:'#e040fb',
                ema9:'#ff9900',ema21:'#00e5ff',vwap:'#ffffff',bb:'rgba(100,180,255,0.8)',
                rsi:'#e91e63',macd:'#2196f3',atr:'rgba(255,152,0,0.8)'
            };
            legend.innerHTML = Object.keys(tvIndicatorSeries).map(k => `
                <div class="tv-legend-item">
                    <div class="tv-legend-swatch" style="background:${colors[k]||'#888'}"></div>
                    ${items[k]||k}
                </div>`).join('');
        }

        // ── Sub-pane chart helper functions ──────────────────────────────────
        function createSubPaneChart(element, height) {
            if (!element) return null;
            return LightweightCharts.createChart(element, {
                autoSize: true,
                height: height,
                layout: { background: { color: '#1E1E1E' }, textColor: '#CCCCCC', fontFamily: 'Arial, sans-serif' },
                grid: { vertLines: { color: '#2A2A2A' }, horzLines: { color: '#2A2A2A' } },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal,
                    vertLine: { color: '#555555', labelBackgroundColor: '#2D2D2D' },
                    horzLine: { color: '#555555', labelBackgroundColor: '#2D2D2D' },
                },
                rightPriceScale: { borderColor: '#333333', scaleMargins: { top: 0.1, bottom: 0.1 } },
                timeScale: {
                    borderColor: '#333333', timeVisible: false, secondsVisible: false,
                    fixLeftEdge: true, fixRightEdge: false,
                },
                handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
                handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
            });
        }

        function setupTimeScaleSync() {
            // Remove old subscriptions first
            tvSyncHandlers.forEach(({chart, handler}) => {
                try { chart.timeScale().unsubscribeVisibleLogicalRangeChange(handler); } catch(e){}
            });
            tvSyncHandlers = [];
            const allCharts = [tvPriceChart, tvRsiChart, tvMacdChart].filter(Boolean);
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
                    if (range) [tvRsiChart, tvMacdChart].filter(Boolean).forEach(c => {
                        try { c.timeScale().setVisibleLogicalRange(range); } catch(e){}
                    });
                } catch(e){}
            }
        }

        function applyRsiPane(allCandles, todayCandles) {
            const pane = document.getElementById('rsi-pane');
            if (!pane) return;
            pane.style.display = 'block';
            // Compute RSI using full history for warmup, then filter to today for display
            const allTimes   = allCandles.map(c => c.time);
            const rsiVals    = calcRSI(allCandles.map(c => c.close));
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

        function applyMacdPane(allCandles, todayCandles) {
            const pane = document.getElementById('macd-pane');
            if (!pane) return;
            pane.style.display = 'block';
            // Compute MACD using full history for warmup, then filter to today for display
            const allTimes = allCandles.map(c => c.time);
            const macdData = calcMACD(allCandles.map(c => c.close));
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
                        price: def.price, color: def.color, lineWidth: 0,
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
                    price, color: drawColor, lineWidth: 0,
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

            // Indicator toggles
            const indicatorDefs = [
                { key:'sma20',  label:'SMA20',  title:'Simple Moving Average (20)' },
                { key:'sma50',  label:'SMA50',  title:'Simple Moving Average (50)' },
                { key:'sma200', label:'SMA200', title:'Simple Moving Average (200)' },
                { key:'ema9',   label:'EMA9',   title:'Exponential Moving Average (9)' },
                { key:'ema21',  label:'EMA21',  title:'Exponential Moving Average (21)' },
                { key:'vwap',   label:'VWAP',   title:'Volume Weighted Average Price' },
                { key:'bb',     label:'BB',     title:'Bollinger Bands (20, 2)' },
                { key:'rsi',    label:'RSI',    title:'Relative Strength Index (14) — sub-pane' },
                { key:'macd',   label:'MACD',   title:'MACD (12, 26, 9) — sub-pane' },
                { key:'atr',    label:'ATR',    title:'Average True Range (14) — sub-pane' },
            ];
            indicatorDefs.forEach(def => {
                const b = btn(def.label, def.title, () => {
                    if (tvActiveInds.has(def.key)) tvActiveInds.delete(def.key);
                    else                           tvActiveInds.add(def.key);
                    b.classList.toggle('active', tvActiveInds.has(def.key));
                    applyIndicators(tvIndicatorCandles, tvActiveInds);
                });
                if (tvActiveInds.has(def.key)) b.classList.add('active');
                toolbar.appendChild(b);
            });

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

        function renderTVPriceChart(priceData) {
            const container = document.getElementById('price-chart');
            if (!container) return;

            tvLastPriceData = priceData;
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

                tvPriceChart = LightweightCharts.createChart(container, {
                    autoSize: true,
                    layout: {
                        background: { color: '#1E1E1E' },
                        textColor:   '#CCCCCC',
                        fontFamily:  'Arial, sans-serif',
                    },
                    grid: {
                        vertLines: { color: '#2A2A2A' },
                        horzLines: { color: '#2A2A2A' },
                    },
                    crosshair: {
                        mode: LightweightCharts.CrosshairMode.Normal,
                        vertLine: { color: '#555555', labelBackgroundColor: '#2D2D2D' },
                        horzLine: { color: '#555555', labelBackgroundColor: '#2D2D2D' },
                    },
                    rightPriceScale: {
                        borderColor:  '#333333',
                        scaleMargins: { top: 0.04, bottom: 0.15 },
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
                        borderColor:      '#333333',
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
                });

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
                const _tc2 = document.getElementById('tv-toolbar-container');
                if (_tc2) {
                    const titleEl = document.createElement('div');
                    titleEl.className = 'tv-chart-title';
                    titleEl.textContent = priceData.use_heikin_ashi ? 'Price Chart (Heikin-Ashi)' : 'Price Chart';
                    _tc2.insertBefore(titleEl, _tc2.firstChild);
                }
            }

            // ── Every render: update data and overlays in place ───────────────

            // Update candle colors in case they changed
            tvCandleSeries.applyOptions({
                upColor, downColor,
                wickUpColor: upColor, wickDownColor: downColor,
            });

            const isFirstRender = !tvLastCandles.length;

            tvCandleSeries.setData(candles);
            tvLastCandles = candles;
            tvVolumeSeries.setData(priceData.volume || []);
            // Use multi-day candles for indicator warmup so SMA200, EMA200, etc. start from day open
            tvIndicatorCandles = (priceData.indicator_candles && priceData.indicator_candles.length > 0)
                ? priceData.indicator_candles : candles;
            tvCurrentDayStartTime = priceData.current_day_start_time || 0;

            // Start/maintain real-time streaming for the current ticker
            const streamTicker = (document.getElementById('ticker').value || '').trim();
            if (streamTicker && isStreaming) {
                connectPriceStream(streamTicker);
            }

            // Remove old dynamic price lines and re-add fresh ones
            tvExposurePriceLines.forEach(l => { try { tvCandleSeries.removePriceLine(l); } catch(e){} });
            tvExposurePriceLines = [];
            tvExpectedMovePriceLines.forEach(l => { try { tvCandleSeries.removePriceLine(l); } catch(e){} });
            tvExpectedMovePriceLines = [];

            tvAllLevelPrices = [];
            for (const level of (priceData.exposure_levels || [])) {
                tvAllLevelPrices.push(level.price);
                tvExposurePriceLines.push(tvCandleSeries.createPriceLine({
                    price:            level.price,
                    color:            level.color,
                    lineWidth:        level.line_width || 1,
                    lineStyle:        lineStyleMap[level.dash_style] || LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title:            level.label,
                }));
            }
            for (const em of (priceData.expected_moves || [])) {
                tvAllLevelPrices.push(em.upper, em.lower);
                tvExpectedMovePriceLines.push(
                    tvCandleSeries.createPriceLine({ price: em.upper, color: '#036bfc', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'EM+' }),
                    tvCandleSeries.createPriceLine({ price: em.lower, color: '#036bfc', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'EM-' })
                );
            }

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
                    } catch(e) {}
                }, 50);
                tvForceFit = false;
            }
        }
        // ─────────────────────────────────────────────────────────────────────

        // Standalone price chart renderer — called by /update_price without touching other charts.
        function applyPriceData(priceJson) {
            if (!document.getElementById('price').checked) return;
            lastPriceData = priceJson; // keep for popout push
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
                priceContainer.appendChild(toolbarContainer);
                priceContainer.appendChild(chartDiv);
                priceContainer.appendChild(rsiPane);
                priceContainer.appendChild(macdPane);
                document.getElementById('chart-grid').insertBefore(priceContainer, document.getElementById('chart-grid').firstChild);
            }
            priceContainer.style.display = 'block';
            const parsed = typeof priceJson === 'string' ? JSON.parse(priceJson) : priceJson;
            if (!parsed.error) {
                renderTVPriceChart(parsed);
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

        function buildPricePayload() {
            return {
                ticker: document.getElementById('ticker').value,
                timeframe: document.getElementById('timeframe').value,
                call_color: callColor,
                put_color: putColor,
                levels_types: Array.from(document.querySelectorAll('.levels-option input:checked')).map(cb => cb.value),
                levels_count: parseInt(document.getElementById('levels_count').value),
                use_heikin_ashi: document.getElementById('use_heikin_ashi').checked,
                strike_range: parseFloat(document.getElementById('strike_range').value) / 100,
                highlight_max_level: document.getElementById('highlight_max_level').checked,
                max_level_color: maxLevelColor,
                coloring_mode: document.getElementById('coloring_mode').value,
            };
        }

        function fetchPriceHistory(force) {
            if (!document.getElementById('price').checked) return;
            if (_priceHistoryInFlight) return;
            const payload = buildPricePayload();
            const key = JSON.stringify(payload);
            const now = Date.now();
            // Skip if nothing changed and it's been less than 30 seconds
            if (!force && key === _priceHistoryLastKey && now - _priceHistoryLastMs < 30000) return;
            _priceHistoryLastMs = now;
            _priceHistoryLastKey = key;
            _priceHistoryInFlight = true;
            fetch('/update_price', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: key
            })
            .then(r => r.json())
            .then(priceResp => {
                if (!priceResp.error && priceResp.price) {
                    applyPriceData(priceResp.price);
                }
            })
            .catch(err => console.error('Error fetching price chart:', err))
            .finally(() => { _priceHistoryInFlight = false; });
        }

        function updateCharts(data) {
            // Save scroll position before any DOM changes
            savedScrollPosition = window.scrollY || window.pageYOffset;
            
            const selectedCharts = {
                price: document.getElementById('price').checked,
                gex_historical_bubble: document.getElementById('gex_historical_bubble').checked,
                dex_historical_bubble: document.getElementById('dex_historical_bubble').checked,
                vanna_historical_bubble: document.getElementById('vanna_historical_bubble').checked,
                charm_historical_bubble: document.getElementById('charm_historical_bubble').checked,
                gamma: document.getElementById('gamma').checked,
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
                premium: document.getElementById('premium').checked,
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
                    priceContainer.appendChild(toolbarContainer);
                    priceContainer.appendChild(chartDiv);
                    priceContainer.appendChild(rsiPane);
                    priceContainer.appendChild(macdPane);
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
                if (tvPriceChart) {
                    try { tvPriceChart.unsubscribeClick(tvHandleChartClick); } catch(e){}
                    tvPriceChart.remove();
                    tvPriceChart = null;
                    tvCandleSeries = null;
                    tvVolumeSeries = null;
                    tvIndicatorSeries = {};
                }
                if (tvResizeObserver) {
                    tvResizeObserver.disconnect();
                    tvResizeObserver = null;
                }
            }
            
            // Handle historical bubble levels and centroid
            const historicalBubbles = ['gex_historical_bubble', 'dex_historical_bubble', 'vanna_historical_bubble', 'charm_historical_bubble', 'centroid'];
            const historicalBubblesRow = document.getElementById('historical-bubbles-row');

            // Count enabled historical bubble levels
            const enabledBubbles = historicalBubbles.filter(bubbleType => selectedCharts[bubbleType]);

            const bubbleConfig = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                displaylogo: false,
                scrollZoom: true
            };

            // Only rebuild containers if the set of enabled bubbles changed
            const currentBubbleIds = Array.from(historicalBubblesRow.querySelectorAll('.historical-bubble-container')).map(el => el.dataset.bubbleType || '');
            const needsRebuild = enabledBubbles.length !== currentBubbleIds.length ||
                                 !enabledBubbles.every((b, i) => currentBubbleIds[i] === b);

            if (needsRebuild) {
                // Purge existing Plotly charts before clearing DOM
                currentBubbleIds.forEach(bt => {
                    const el = document.getElementById(bt + '-chart');
                    if (el) { try { Plotly.purge(el); } catch(e) {} }
                    delete charts[bt];
                });
                historicalBubblesRow.innerHTML = '';
                historicalBubblesRow.className = 'historical-bubbles-row';

                // Hide the row if no bubbles are enabled
                if (enabledBubbles.length === 0) {
                    historicalBubblesRow.style.display = 'none';
                } else {
                    historicalBubblesRow.style.display = 'grid';

                    // Add appropriate class based on number of enabled bubbles
                    if (enabledBubbles.length === 1) {
                        historicalBubblesRow.classList.add('one-bubble');
                    } else if (enabledBubbles.length === 2) {
                        historicalBubblesRow.classList.add('two-bubbles');
                    } else if (enabledBubbles.length === 3) {
                        historicalBubblesRow.classList.add('three-bubbles');
                    } else if (enabledBubbles.length === 4) {
                        historicalBubblesRow.classList.add('four-bubbles');
                    }

                    // Create containers and render Plotly charts
                    enabledBubbles.forEach(bubbleType => {
                        const bubbleContainer = document.createElement('div');
                        bubbleContainer.className = 'historical-bubble-container';
                        bubbleContainer.dataset.bubbleType = bubbleType;

                        const chartDiv = document.createElement('div');
                        chartDiv.className = 'chart-container';
                        chartDiv.id = bubbleType + '-chart';

                        bubbleContainer.appendChild(chartDiv);
                        historicalBubblesRow.appendChild(bubbleContainer);

                        if (data[bubbleType]) {
                            const chartData = JSON.parse(data[bubbleType]);
                            chartData.layout.margin = getChartMargins(chartDiv.id, chartData.layout.margin || {l: 50, r: 50, t: 50, b: 20});
                            charts[bubbleType] = Plotly.newPlot(chartDiv, chartData.data, chartData.layout, bubbleConfig);
                        }
                    });
                }
            } else {
                // Efficiently update existing Plotly charts without rebuilding DOM
                enabledBubbles.forEach(bubbleType => {
                    const chartDiv = document.getElementById(bubbleType + '-chart');
                    if (!chartDiv) return;
                    if (data[bubbleType]) {
                        const chartData = JSON.parse(data[bubbleType]);
                        chartData.layout.margin = getChartMargins(chartDiv.id, chartData.layout.margin || {l: 50, r: 50, t: 50, b: 20});
                        Plotly.react(chartDiv, chartData.data, chartData.layout, bubbleConfig);
                        charts[bubbleType] = true;
                    } else {
                        // No data for new ticker — clear to prevent stale previous ticker chart
                        Plotly.react(chartDiv, [], {
                            plot_bgcolor: '#1E1E1E',
                            paper_bgcolor: '#1E1E1E',
                            font: { color: '#CCCCCC' },
                            annotations: [{
                                text: 'No Data Available',
                                x: 0.5, y: 0.5,
                                xref: 'paper', yref: 'paper',
                                showarrow: false,
                                font: { color: '#CCCCCC', size: 14 }
                            }]
                        }, bubbleConfig);
                        charts[bubbleType] = true;
                    }
                });
            }

            // Clean up disabled historical bubble levels from charts object
            historicalBubbles.forEach(bubbleType => {
                if (!selectedCharts[bubbleType]) {
                    const el = document.getElementById(bubbleType + '-chart');
                    if (el) { try { Plotly.purge(el); } catch(e) {} }
                    delete charts[bubbleType];
                }
            });
            
            // Handle other charts
            let chartsGrid = document.querySelector('.charts-grid');
            if (!chartsGrid) {
                chartsGrid = document.createElement('div');
                chartsGrid.className = 'charts-grid';
                document.getElementById('chart-grid').appendChild(chartsGrid);
            }
            
            // Check if we need to rebuild the grid (enabled charts changed)
            const currentChartIds = Array.from(chartsGrid.querySelectorAll('.chart-container')).map(el => el.id.replace('-chart', ''));
            
            // Count enabled regular charts (excluding price and historical bubble levels)
            const regularCharts = Object.entries(selectedCharts).filter(([key, selected]) => 
                selected && !['price'].includes(key) && !historicalBubbles.includes(key) && data[key]
            );
            
            const regularChartIds = regularCharts.map(([key]) => key);
            const needsGridRebuild = regularChartIds.length !== currentChartIds.length ||
                                     !regularChartIds.every((id, i) => currentChartIds[i] === id);
            
            // Hide the charts grid if no regular charts are enabled
            if (regularCharts.length === 0) {
                chartsGrid.style.display = 'none';
                chartsGrid.innerHTML = '';
            } else {
                chartsGrid.style.display = 'grid';
                
                // Only rebuild if chart selection changed
                if (needsGridRebuild) {
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
                            // Only update if content changed
                            if (container.innerHTML !== data[key]) {
                                container.innerHTML = data[key];
                            }
                        } else {
                            const chartData = JSON.parse(data[key]);
                            
                            // Configure chart sizing to fill container
                            chartData.layout.autosize = true;
                            chartData.layout.width = null;
                            chartData.layout.height = null;
                            chartData.layout.margin = getChartMargins(`${key}-chart`, {l: 50, r: 50, t: 40, b: 20});
                            
                            // Ensure axes auto-scale with new data
                            if (chartData.layout.xaxis) {
                                chartData.layout.xaxis.autorange = true;
                            }
                            if (chartData.layout.yaxis) {
                                chartData.layout.yaxis.autorange = true;
                            }
                            
                            chartData.layout.plot_bgcolor = '#1E1E1E';
                            chartData.layout.paper_bgcolor = '#1E1E1E';
                            
                            const config = {
                                responsive: true,
                                displayModeBar: true,
                                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                                displaylogo: false,
                                useResizeHandler: true,
                                style: {width: "100%", height: "100%"}
                            };
                            
                            if (charts[key]) {
                                Plotly.react(`${key}-chart`, chartData.data, chartData.layout, config);
                            } else {
                                charts[key] = Plotly.newPlot(`${key}-chart`, chartData.data, chartData.layout, config);
                            }
                        }
                    } catch (error) {
                        console.error(`Error rendering ${key} chart:`, error);
                    }
                });
            }
            
            // Clean up disabled regular charts from charts object
            Object.keys(selectedCharts).forEach(key => {
                if (!selectedCharts[key] && !['price'].includes(key) && !historicalBubbles.includes(key)) {
                    const container = document.getElementById(`${key}-chart`);
                    if (container) {
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
            const fsChart = document.querySelector('.chart-container.fullscreen');
            if (fsChart) {
                requestAnimationFrame(() => {
                    const plot = fsChart.querySelector('.js-plotly-plot');
                    if (plot) { try { Plotly.Plots.resize(plot); } catch(e) {} }
                });
            }

            // Restore scroll position after DOM updates
            requestAnimationFrame(() => {
                window.scrollTo(0, savedScrollPosition);
            });
        }
        
        function updatePriceInfo(info) {
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
                // lower bound is below spot -> use putColor, upper bound above spot -> callColor
                const lowColor = putColor;
                const highColor = callColor;
                expectedMoveHtml = `<div>Expected Move: <span style="color:${lowColor}">$${info.expected_move_range.lower.toFixed(2)} ${lowPct}</span> - <span style="color:${highColor}">$${info.expected_move_range.upper.toFixed(2)} ${highPct}</span></div>`;
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
            priceInfo.innerHTML = `
                <div data-live-price>Current Price: $${displayPrice.toFixed(2)}</div>
                <div>High: $${info.high.toFixed(2)} <span style="color:${highColor}">(${highDiffPct>=0?'+':''}${highDiffPct.toFixed(2)}%)</span></div>
                <div>Low:  $${info.low.toFixed(2)}  <span style="color:${lowColor}">(${lowDiffPct>=0?'+':''}${lowDiffPct.toFixed(2)}%)</span></div>
                <div class="${info.net_change >= 0 ? 'green' : 'red'}">
                    <span style="color:white !important">Change:</span> ${info.net_change >= 0 ? '+' : ''}${info.net_change.toFixed(2)} (${info.net_percent >= 0 ? '+' : ''}${info.net_percent.toFixed(2)}%)
                </div>
                <div>Vol Ratio: <span style="color: ${callColor}">${info.call_percentage.toFixed(2)}%</span>/<span style="color: ${putColor}">${info.put_percentage.toFixed(2)}%</span></div>
                ${expectedMoveHtml}
                <div>Expiries: ${expiryText}</div>
            `;
        }
        
        function loadExpirations() {
            const ticker = document.getElementById('ticker').value;
            fetch(`/expirations/${ticker}`)
                .then(response => {
                    if (!response.ok) throw new Error('Failed to fetch expirations');
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        showError(data.error);
                        return;
                    }
                    const optionsContainer = document.getElementById('expiry-options');
                    const previousSelections = Array.from(document.querySelectorAll('.expiry-option input[type="checkbox"]:checked')).map(cb => cb.value);
                    
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
                        
                        // Restore previous selections if they still exist
                        if (previousSelections.includes(date)) {
                            checkbox.checked = true;
                        }
                        
                        // Add change event listener
                        checkbox.addEventListener('change', function() {
                            updateExpiryDisplay();
                            updateData();
                        });
                        
                        optionDiv.appendChild(checkbox);
                        optionDiv.appendChild(label);
                        optionsContainer.appendChild(optionDiv);
                    });
                    
                    // Re-add the buttons at the top
                    optionsContainer.insertBefore(buttons, optionsContainer.firstChild);
                    
                    // If no previous selections or none match, select the first option
                    const checkedBoxes = document.querySelectorAll('.expiry-option input[type="checkbox"]:checked');
                    if (checkedBoxes.length === 0 && data.length > 0) {
                        const firstCheckbox = document.querySelector('.expiry-option input[type="checkbox"]');
                        if (firstCheckbox) {
                            firstCheckbox.checked = true;
                        }
                    }
                    
                    updateExpiryDisplay();
                    updateData();
                })
                .catch(error => {
                    showError('Error loading expirations: ' + error.message);
                });
        }
        
        function updateExpiryDisplay() {
            const checkedBoxes = document.querySelectorAll('.expiry-option input[type="checkbox"]:checked');
            const expiryText = document.getElementById('expiry-text');
            
            if (checkedBoxes.length === 0) {
                expiryText.textContent = 'Select expiry dates...';
            } else if (checkedBoxes.length === 1) {
                expiryText.textContent = checkedBoxes[0].value;
            } else {
                expiryText.textContent = `${checkedBoxes.length} expiries selected`;
            }
        }
        
        // Add event listeners for checkboxes
        document.querySelectorAll('.chart-checkbox input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', updateData);
        });
        
        // Add event listeners for control checkboxes
        document.querySelectorAll('.control-group input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', updateData);
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
        document.getElementById('selectAllExpiry').addEventListener('click', function(e) {
            e.stopPropagation();
            const checkboxes = document.querySelectorAll('.expiry-option input[type="checkbox"]');
            checkboxes.forEach(checkbox => {
                checkbox.checked = true;
            });
            updateExpiryDisplay();
            updateData();
        });
        
        document.getElementById('clearAllExpiry').addEventListener('click', function(e) {
            e.stopPropagation();
            const checkboxes = document.querySelectorAll('.expiry-option input[type="checkbox"]');
            checkboxes.forEach(checkbox => {
                checkbox.checked = false;
            });
            // Select the first option to ensure at least one is selected
            if (checkboxes.length > 0) {
                checkboxes[0].checked = true;
            }
            updateExpiryDisplay();
            updateData();
        });

        function selectExpiriesUpTo(cutoffDate) {
            const checkboxes = document.querySelectorAll('.expiry-option input[type="checkbox"]');
            let anyChecked = false;
            checkboxes.forEach(checkbox => {
                // Parse as local date to avoid UTC offset issues
                const parts = checkbox.value.split('-');
                const d = new Date(parseInt(parts[0]), parseInt(parts[1]) - 1, parseInt(parts[2]));
                checkbox.checked = d <= cutoffDate;
                if (checkbox.checked) anyChecked = true;
            });
            if (!anyChecked && checkboxes.length > 0) {
                checkboxes[0].checked = true;
            }
            updateExpiryDisplay();
            updateData();
        }

        function getFriday(weeksAhead) {
            const today = new Date();
            today.setHours(0, 0, 0, 0);
            const dow = today.getDay(); // 0=Sun,1=Mon,...,5=Fri,6=Sat
            const daysToFriday = (5 - dow + 7) % 7;
            const cutoff = new Date(today);
            cutoff.setDate(today.getDate() + daysToFriday + weeksAhead * 7);
            return cutoff;
        }

        document.getElementById('expiryToday').addEventListener('click', function(e) {
            e.stopPropagation();
            const today = new Date();
            today.setHours(0, 0, 0, 0);
            selectExpiriesUpTo(today);
        });

        document.getElementById('expiryThisWk').addEventListener('click', function(e) {
            e.stopPropagation();
            selectExpiriesUpTo(getFriday(0));
        });

        function selectFirstNExpiries(n) {
            const checkboxes = document.querySelectorAll('.expiry-option input[type="checkbox"]');
            checkboxes.forEach((checkbox, i) => {
                checkbox.checked = i < n;
            });
            if (checkboxes.length > 0 && n === 0) checkboxes[0].checked = true;
            updateExpiryDisplay();
            updateData();
        }

        document.getElementById('expiry2Wks').addEventListener('click', function(e) {
            e.stopPropagation();
            selectFirstNExpiries(7);
        });

        document.getElementById('expiry4Wks').addEventListener('click', function(e) {
            e.stopPropagation();
            selectFirstNExpiries(14);
        });

        document.getElementById('expiry1Mo').addEventListener('click', function(e) {
            e.stopPropagation();
            const cutoff = new Date();
            cutoff.setHours(0, 0, 0, 0);
            cutoff.setDate(cutoff.getDate() + 30);
            selectExpiriesUpTo(cutoff);
        });

        // Initial load - automatically load saved settings, or use defaults
        loadSettings(false);

        // Auto-update every 1 second
        updateInterval = setInterval(updateData, 1000);
        
        // Handle window resize
        window.addEventListener('resize', () => {
            Object.keys(charts).forEach(chartKey => {
                const chartElement = document.getElementById(`${chartKey}-chart`);
                if (chartElement && charts[chartKey]) {
                    Plotly.Plots.resize(chartElement);
                }
            });
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            clearInterval(updateInterval);
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
                updateInterval = setInterval(updateData, 1000);
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
                timeframe: document.getElementById('timeframe').value,
                strike_range: document.getElementById('strike_range').value,
                exposure_metric: document.getElementById('exposure_metric').value,
                delta_adjusted_exposures: document.getElementById('delta_adjusted_exposures').checked,
                calculate_in_notional: document.getElementById('calculate_in_notional').checked,
                show_calls: document.getElementById('show_calls').checked,
                show_puts: document.getElementById('show_puts').checked,
                show_net: document.getElementById('show_net').checked,
                coloring_mode: document.getElementById('coloring_mode').value,
                levels_types: Array.from(document.querySelectorAll('.levels-option input:checked')).map(cb => cb.value),
                levels_count: document.getElementById('levels_count').value,
                use_heikin_ashi: document.getElementById('use_heikin_ashi').checked,
                horizontal_bars: document.getElementById('horizontal_bars').checked,
                show_abs_gex: document.getElementById('show_abs_gex').checked,
                abs_gex_opacity: document.getElementById('abs_gex_opacity').value,
                gex_absolute: document.getElementById('gex_absolute').checked,
                use_range: document.getElementById('use_range').checked,
                call_color: document.getElementById('call_color').value,
                put_color: document.getElementById('put_color').value,
                highlight_max_level: document.getElementById('highlight_max_level').checked,
                max_level_color: document.getElementById('max_level_color').value,
                max_level_mode: document.getElementById('max_level_mode').value,
                // Chart visibility
                charts: {
                    price: document.getElementById('price').checked,
                    gex_historical_bubble: document.getElementById('gex_historical_bubble').checked,
                    dex_historical_bubble: document.getElementById('dex_historical_bubble').checked,
                    vanna_historical_bubble: document.getElementById('vanna_historical_bubble').checked,
                    charm_historical_bubble: document.getElementById('charm_historical_bubble').checked,
                    gamma: document.getElementById('gamma').checked,
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
                    premium: document.getElementById('premium').checked,
                    centroid: document.getElementById('centroid').checked
                }
            };
        }
        
        function applySettings(settings) {
            if (settings.ticker) document.getElementById('ticker').value = settings.ticker;
            if (settings.timeframe) document.getElementById('timeframe').value = settings.timeframe;
            if (settings.strike_range) {
                document.getElementById('strike_range').value = settings.strike_range;
                document.getElementById('strike_range_value').textContent = settings.strike_range + '%';
            }
            if (settings.exposure_metric) document.getElementById('exposure_metric').value = settings.exposure_metric;
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
                    const cb = document.getElementById('lvl-' + type.replace(/\s+/g, ''));
                    if (cb) cb.checked = true;
                });
                updateLevelsDisplay();
            }
            if (settings.levels_count) document.getElementById('levels_count').value = settings.levels_count;
            if (settings.use_heikin_ashi !== undefined) document.getElementById('use_heikin_ashi').checked = settings.use_heikin_ashi;
            if (settings.horizontal_bars !== undefined) document.getElementById('horizontal_bars').checked = settings.horizontal_bars;
            if (settings.show_abs_gex !== undefined) document.getElementById('show_abs_gex').checked = settings.show_abs_gex;
            if (settings.abs_gex_opacity !== undefined) document.getElementById('abs_gex_opacity').value = settings.abs_gex_opacity;
            if (settings.gex_absolute !== undefined) document.getElementById('gex_absolute').checked = settings.gex_absolute;
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
            if (settings.max_level_color) {
                document.getElementById('max_level_color').value = settings.max_level_color;
                maxLevelColor = settings.max_level_color;
            }
            if (settings.max_level_mode) {
                document.getElementById('max_level_mode').value = settings.max_level_mode;
            }
            // Chart visibility
            if (settings.charts) {
                Object.keys(settings.charts).forEach(chartId => {
                    const checkbox = document.getElementById(chartId);
                    if (checkbox) checkbox.checked = settings.charts[chartId];
                });
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
                // Also update the data
                updateData();
            }
        });

    // ── Token Monitor ──────────────────────────────────────────────────────
    function fetchTokenHealth() {
        fetch('/token_health')
            .then(r => r.json())
            .then(d => {
                const dot   = document.getElementById('tm-dot');
                const stats = document.getElementById('tm-stats');

                if (!d.db_exists || d.error) {
                    dot.className = 'tm-dot tm-err';
                    stats.textContent = d.error || 'DB missing';
                    stats.title = d.db_path || '';
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
        fetch('/token_delete', { method: 'POST' })
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

@app.route('/expirations/<ticker>')
def get_expirations(ticker):
    try:
        ticker = format_ticker(ticker)
        expirations = get_option_expirations(ticker)
        return jsonify(expirations)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

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
            
        # Get current price
        S = get_current_price(ticker)
        if S is None:
            return jsonify({'error': 'Could not fetch current price'})

        # Cache options data so /update_price can use it without re-fetching
        _options_cache[ticker] = {'calls': calls.copy(), 'puts': puts.copy(), 'S': S}
        
        # Get strike range
        strike_range = float(data.get('strike_range', 0.1))
        
        # Store interval data
        store_interval_data(ticker, S, strike_range, calls, puts)
        
        # Check if this is the first access of the day for this ticker and clear centroid data if needed
        est = pytz.timezone('US/Eastern')
        current_time_est = datetime.now(est)
        
        # Check if we're in a new trading session (after 9:30 AM ET)
        if (current_time_est.hour == 9 and current_time_est.minute >= 30) or current_time_est.hour > 9:
            if current_time_est.weekday() < 5:  # Weekday
                # Check if we have any centroid data from before 9:30 AM today
                today = current_time_est.strftime('%Y-%m-%d')
                market_open_timestamp = int(current_time_est.replace(hour=9, minute=30, second=0, microsecond=0).timestamp())
                
                with closing(sqlite3.connect('options_data.db')) as conn:
                    with closing(conn.cursor()) as cursor:
                        cursor.execute('''
                            SELECT COUNT(*) FROM centroid_data 
                            WHERE ticker = ? AND date = ? AND timestamp < ?
                        ''', (ticker, today, market_open_timestamp))
                        
                        pre_market_count = cursor.fetchone()[0]
                        if pre_market_count > 0:
                            # Clear pre-market centroid data for a fresh session
                            cursor.execute('''
                                DELETE FROM centroid_data 
                                WHERE ticker = ? AND date = ? AND timestamp < ?
                            ''', (ticker, today, market_open_timestamp))
                            conn.commit()
        
        # Store centroid data
        store_centroid_data(ticker, S, calls, puts)
        
        # Clear centroid data at the end of the day
        current_time = datetime.now()
        if current_time.hour == 23 and current_time.minute == 59:
            clear_old_data()
        
        # Get timeframe from request
        timeframe = int(data.get('timeframe', 1))

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
        coloring_mode = data.get('coloring_mode', None)
        if coloring_mode is None:
            # Migrate from old boolean color_intensity
            old_color_intensity = data.get('color_intensity', True)
            coloring_mode = 'Linear Intensity' if old_color_intensity else 'Solid'
        call_color = data.get('call_color', '#00ff00')
        put_color = data.get('put_color', '#ff0000')
        exposure_levels_types = data.get('levels_types', [])
        exposure_levels_count = int(data.get('levels_count', 3))
        use_heikin_ashi = data.get('use_heikin_ashi', False)
        horizontal = data.get('horizontal_bars', False)
        show_abs_gex = data.get('show_abs_gex', False)
        abs_gex_opacity = float(data.get('abs_gex_opacity', 0.2))
        highlight_max_level = data.get('highlight_max_level', False)
        max_level_color = data.get('max_level_color', '#800080')
        max_level_mode = data.get('max_level_mode', 'Absolute')
 
        
        response = {}
        
        # Create charts based on visibility settings
        # NOTE: price chart is handled by /update_price (separate concurrent request)

        if data.get('show_gex_historical_bubble', True):
            # Accept either 'gex_absolute' (old key) or 'absolute_gex' (JS payload) for compatibility
            absolute_gex = bool(data.get('gex_absolute', data.get('absolute_gex', False)))
            response['gex_historical_bubble'] = create_historical_bubble_levels_chart(ticker, strike_range, call_color, put_color, 'gamma', absolute=absolute_gex, highlight_max_level=highlight_max_level, max_level_color=max_level_color)

        if data.get('show_dex_historical_bubble', True):
            response['dex_historical_bubble'] = create_historical_bubble_levels_chart(ticker, strike_range, call_color, put_color, 'delta', highlight_max_level=highlight_max_level, max_level_color=max_level_color)

        if data.get('show_vanna_historical_bubble', True):
            response['vanna_historical_bubble'] = create_historical_bubble_levels_chart(ticker, strike_range, call_color, put_color, 'vanna', highlight_max_level=highlight_max_level, max_level_color=max_level_color)

        if data.get('show_charm_historical_bubble', True):
            response['charm_historical_bubble'] = create_historical_bubble_levels_chart(ticker, strike_range, call_color, put_color, 'charm', highlight_max_level=highlight_max_level, max_level_color=max_level_color)
        
        if data.get('show_gamma', True):
            response['gamma'] = create_exposure_chart(calls, puts, "GEX", "Gamma Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, show_abs_gex_area=show_abs_gex, abs_gex_opacity=abs_gex_opacity, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_delta', True):
            response['delta'] = create_exposure_chart(calls, puts, "DEX", "Delta Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_vanna', True):
            response['vanna'] = create_exposure_chart(calls, puts, "VEX", "Vanna Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_charm', True):
            response['charm'] = create_exposure_chart(calls, puts, "Charm", "Charm Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_speed', True):
            response['speed'] = create_exposure_chart(calls, puts, "Speed", "Speed Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_vomma', True):
            response['vomma'] = create_exposure_chart(calls, puts, "Vomma", "Vomma Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)

        if data.get('show_color', True):
            response['color'] = create_exposure_chart(calls, puts, "Color", "Color Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, coloring_mode, call_color, put_color, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_volume', True):
            response['volume'] = create_volume_chart(call_volume, put_volume, use_range, call_color, put_color, expiry_dates)
        
        if data.get('show_options_volume', True):
            response['options_volume'] = create_options_volume_chart(calls, puts, S, strike_range, call_color, put_color, coloring_mode, show_calls, show_puts, show_net, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_open_interest', True):
            response['open_interest'] = create_open_interest_chart(calls, puts, S, strike_range, call_color, put_color, coloring_mode, show_calls, show_puts, show_net, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_premium', True):
            response['premium'] = create_premium_chart(calls, puts, S, strike_range, call_color, put_color, coloring_mode, show_calls, show_puts, show_net, expiry_dates, horizontal, highlight_max_level=highlight_max_level, max_level_color=max_level_color, max_level_mode=max_level_mode)
        
        if data.get('show_large_trades', True):
            response['large_trades'] = create_large_trades_table(calls, puts, S, strike_range, call_color, put_color, expiry_dates)
        
        if data.get('show_centroid', True):
            response['centroid'] = create_centroid_chart(ticker, call_color, put_color, expiry_dates)

        
        # Add volume data to response
        response.update({
            'call_volume': call_volume,
            'put_volume': put_volume,
            'total_volume': total_volume,
            'call_percentage': call_percentage,
            'put_percentage': put_percentage,
            'selected_expiries': expiry_dates  # Add this to show which expiries are selected
        })
        
        # Get fresh quote data
        try:
            # Use appropriate base ticker for market tickers
            if ticker == "MARKET":
                quote_ticker = "$SPX"
            elif ticker == "MARKET2":
                quote_ticker = "SPY"
            else:
                quote_ticker = ticker

            quote_response = client.quote(quote_ticker)
            if not quote_response.ok:
                raise Exception(f"Failed to fetch quote for display: {quote_response.status_code} {quote_response.reason}")

            # --- Always Calculate Expected Move Range (same as chart logic) ---
            expected_move_range = None
            strikes_sorted = sorted(calls['strike'].unique()) if not calls.empty else []
            if strikes_sorted:
                atm_strike = min(strikes_sorted, key=lambda x: abs(x - S))
                atm_idx = strikes_sorted.index(atm_strike)
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

            if quote_response.ok:
                quote_data = quote_response.json()
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
                    'call_volume': call_volume,
                    'put_volume': put_volume,
                    'total_volume': total_volume,
                    'call_percentage': call_percentage,
                    'put_percentage': put_percentage,
                    'expected_move_range': expected_move_range
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
                'call_volume': call_volume,
                'put_volume': put_volume,
                'total_volume': total_volume,
                'call_percentage': call_percentage,
                'put_percentage': put_percentage,
                'expected_move_range': None
            }
        
        return jsonify(response)
        
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
    ticker = format_ticker(ticker)
    if not ticker:
        return jsonify({'error': 'Missing ticker'}), 400
    try:
        timeframe = int(data.get('timeframe', 1))
        call_color = data.get('call_color', '#00ff00')
        put_color = data.get('put_color', '#ff0000')
        exposure_levels_types = data.get('levels_types', [])
        exposure_levels_count = int(data.get('levels_count', 3))
        strike_range = float(data.get('strike_range', 0.1))
        use_heikin_ashi = data.get('use_heikin_ashi', False)
        highlight_max_level = data.get('highlight_max_level', False)
        max_level_color = data.get('max_level_color', '#800080')
        coloring_mode = data.get('coloring_mode', 'Linear Intensity')

        price_data = get_price_history(ticker, timeframe=timeframe)

        # Use the most recently cached options data for exposure overlays.
        # If no cache exists yet the chart renders without overlays and
        # will gain them after the first /update completes.
        cached = _options_cache.get(ticker, {})
        calls = cached.get('calls')
        puts = cached.get('puts')

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
            coloring_mode=coloring_mode
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
        with open('settings.json', 'w') as f:
            json.dump(settings, f, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_settings')
def load_settings():
    try:
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
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
    app.run(debug=True, port=5001, threaded=True)