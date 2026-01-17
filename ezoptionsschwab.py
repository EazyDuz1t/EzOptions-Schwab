from flask import Flask, render_template_string, jsonify, request
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
from scipy.optimize import brentq
import base64
import warnings


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

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
                    date TEXT NOT NULL
                )
            ''')
            # Try to add net_charm column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE interval_data ADD COLUMN net_charm REAL')
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

# Function to store centroid data
def store_centroid_data(ticker, price, calls, puts):
    """Store call and put centroid data for 15-minute intervals during market hours only"""
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
    
    # Round to nearest 15-minute interval (900 seconds)
    interval_timestamp = (current_time // 900) * 900
    
    # Check if we already have data for this 15-minute interval
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                SELECT id FROM centroid_data 
                WHERE ticker = ? AND timestamp = ? AND date = ?
            ''', (ticker, interval_timestamp, current_date))
            
            if cursor.fetchone():
                return  # Data already exists for this interval
    
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
    current_time = int(time.time())
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Only store data if it's a new 1-minute interval
    last_data = get_interval_data(ticker)
    if last_data:
        last_timestamp = last_data[-1][0]
        if current_time - last_timestamp < 60:  # Less than 60 seconds since last update
            return
    
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
        exposure_by_strike[strike] = {
            'gamma': exposure_by_strike.get(strike, {}).get('gamma', 0) + gamma,
            'delta': exposure_by_strike.get(strike, {}).get('delta', 0) + delta,
            'vanna': exposure_by_strike.get(strike, {}).get('vanna', 0) + vanna,
            'charm': exposure_by_strike.get(strike, {}).get('charm', 0) + charm
        }
        
    for _, row in range_puts.iterrows():
        strike = row['strike']
        gamma = row['GEX']
        delta = row['DEX']
        vanna = row['VEX']
        charm = row['Charm']
        exposure_by_strike[strike] = {
            'gamma': exposure_by_strike.get(strike, {}).get('gamma', 0) - gamma,
            'delta': exposure_by_strike.get(strike, {}).get('delta', 0) + delta,
            'vanna': exposure_by_strike.get(strike, {}).get('vanna', 0) + vanna,
            'charm': exposure_by_strike.get(strike, {}).get('charm', 0) + charm
        }
    
    # Store data for each strike
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            for strike, exposure in exposure_by_strike.items():
                cursor.execute('''
                    INSERT INTO interval_data (ticker, timestamp, price, strike, net_gamma, net_delta, net_vanna, net_charm, date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (ticker, current_time, price, strike, exposure['gamma'], exposure['delta'], exposure['vanna'], exposure['charm'], current_date))
            conn.commit()

# Function to get interval data
def get_interval_data(ticker, date=None):
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    with closing(sqlite3.connect('options_data.db')) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                SELECT timestamp, price, strike, net_gamma, net_delta, net_vanna, net_charm
                FROM interval_data
                WHERE ticker = ? AND date = ?
                ORDER BY timestamp, strike
            ''', (ticker, date))
            return cursor.fetchall()

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
    if ticker == "MARKET":
        # Get prices for scaling (Base price is $SPX)
        spx_price = S if S else get_current_price("$SPX")
        spy_price = get_current_price("SPY")
        qqq_price = get_current_price("QQQ")
        iwm_price = get_current_price("IWM")
        
        if not spx_price:
             return pd.DataFrame(), pd.DataFrame()
        
        calls_list = []
        puts_list = []
        
        # Helper to fetch and scale by moneyness
        def add_scaled_data(tick, etf_price):
            try:
                # Fetch ETF options
                # Note: We pass exposure_metric and delta_adjusted to recursive calls
                c, p = fetch_options_for_date(tick, date, exposure_metric, delta_adjusted, calculate_in_notional)
                
                if not c.empty:
                    c = c.copy()
                    # Calculate moneyness (strike / spot) for each ETF option
                    c['moneyness'] = c['strike'] / etf_price
                    # Map to SPX strikes by moneyness: SPX_strike = moneyness * SPX_price
                    c['strike'] = (c['moneyness'] * spx_price / 10).round() * 10
                    c.drop(columns=['moneyness'], inplace=True)
                    
                    # Scale prices by price ratio (options are more expensive on higher priced underlyings)
                    price_ratio = spx_price / etf_price
                    for col in ['lastPrice', 'bid', 'ask', 'change']:
                        if col in c.columns:
                            c[col] = c[col] * price_ratio
                    
                    # Scale Greeks (Exposure scales with Spot Price)
                    greek_cols = ['GEX', 'DEX', 'VEX', 'Charm', 'Speed', 'Vomma', 'Color']
                    for col in greek_cols:
                        if col in c.columns:
                            c[col] = c[col] * price_ratio
                    
                    calls_list.append(c)
                
                if not p.empty:
                    p = p.copy()
                    p['moneyness'] = p['strike'] / etf_price
                    p['strike'] = (p['moneyness'] * spx_price / 10).round() * 10
                    p.drop(columns=['moneyness'], inplace=True)
                    
                    price_ratio = spx_price / etf_price
                    for col in ['lastPrice', 'bid', 'ask', 'change']:
                        if col in p.columns:
                            p[col] = p[col] * price_ratio

                    # Scale Greeks (Exposure scales with Spot Price)
                    greek_cols = ['GEX', 'DEX', 'VEX', 'Charm', 'Speed', 'Vomma', 'Color']
                    for col in greek_cols:
                        if col in p.columns:
                            p[col] = p[col] * price_ratio
                    
                    puts_list.append(p)
            except Exception:
                pass 

        # Add scaled data from ETFs
        if spy_price: add_scaled_data("SPY", spy_price)
        if qqq_price: add_scaled_data("QQQ", qqq_price)
        if iwm_price: add_scaled_data("IWM", iwm_price)
        
        combined_calls = pd.concat(calls_list, ignore_index=True) if calls_list else pd.DataFrame()
        combined_puts = pd.concat(puts_list, ignore_index=True) if puts_list else pd.DataFrame()
        
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
            return pd.DataFrame(), pd.DataFrame()
        
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
                        # Calculate implied volatility from mid price if possible
                        bid = float(option['bid'])
                        ask = float(option['ask'])
                        market_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(option['last'])
                        
                        K = float(option['strikePrice'])
                        vol = 0.2
                        if market_price > 0:
                            vol = calculate_implied_volatility(market_price, S, K, t, r, 'c', 0)
                            if vol is None: vol = 0.2
                        
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
                        # Calculate implied volatility from mid price if possible
                        bid = float(option['bid'])
                        ask = float(option['ask'])
                        market_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(option['last'])
                        
                        K = float(option['strikePrice'])
                        vol = 0.2
                        if market_price > 0:
                            vol = calculate_implied_volatility(market_price, S, K, t, r, 'p', 0)
                            if vol is None: vol = 0.2
                        
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
        max_vol = 1
        if exposure_metric == 'OI Weighted by Volume':
            max_vol_calls = max((o['volume'] for o in calls_data), default=0)
            max_vol_puts = max((o['volume'] for o in puts_data), default=0)
            max_vol = max(max_vol_calls, max_vol_puts, 1)

        for option_data in calls_data:
            weight = 0
            if exposure_metric == 'Volume':
                weight = option_data['volume']
            elif exposure_metric == 'OI Weighted by Volume':
                weight = option_data['openInterest'] * (1 + option_data['volume'] / max_vol)
            else: # Open Interest
                weight = option_data['openInterest']
                
            exposures = calculate_greek_exposures(option_data, S, weight, delta_adjusted=delta_adjusted, calculate_in_notional=calculate_in_notional)
            option_data.update(exposures)

        for option_data in puts_data:
            weight = 0
            if exposure_metric == 'Volume':
                weight = option_data['volume']
            elif exposure_metric == 'OI Weighted by Volume':
                weight = option_data['openInterest'] * (1 + option_data['volume'] / max_vol)
            else: # Open Interest
                weight = option_data['openInterest']
                
            exposures = calculate_greek_exposures(option_data, S, weight, delta_adjusted=delta_adjusted, calculate_in_notional=calculate_in_notional)
            option_data.update(exposures)

        calls = pd.DataFrame(calls_data)
        puts = pd.DataFrame(puts_data)
        return calls, puts
        
    except Exception as e:
        print(f"Error fetching options chain: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_bs_price(flag, S, K, t, r, sigma, q=0):
    """Calculate Black-Scholes option price with dividends."""
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        if flag == 'c':
            price = S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-q * t) * norm.cdf(-d1)
        return price
    except:
        return 0.0

def calculate_bs_vega(S, K, t, r, sigma, q=0):
    """Calculate Black-Scholes Vega with dividends."""
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        return S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)
    except:
        return 0.0

def calculate_implied_volatility(price, S, K, t, r, flag, q=0):
    """Calculate Implied Volatility using Newton-Raphson method."""
    sigma = 0.5  # Initial guess
    for i in range(100):
        bs_price = calculate_bs_price(flag, S, K, t, r, sigma, q)
        diff = price - bs_price
        
        if abs(diff) < 1e-5:
            return sigma
            
        vega = calculate_bs_vega(S, K, t, r, sigma, q)
        if abs(vega) < 1e-8:
            return None
            
        sigma = sigma + diff / vega
        
        if sigma <= 0:
            sigma = 0.001 # Reset if negative
        if sigma > 5:
            sigma = 5.0 # Cap if too high
            
    return 0.2 # Default if failed

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
    # Speed Exposure (Notional) ~ Speed * S * S * 0.01 (Matches ezoptions.py Speed * S * spot_multiplier)
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
    if ticker == "MARKET":
        ticker = "$SPX"
    try:
        quote_response = client.quotes(ticker)
        if not quote_response.ok:
            return None
        quote = quote_response.json()
        if quote and ticker in quote:
            return quote[ticker]['quote']['lastPrice']
        return None
    except Exception as e:
        print(f"Error fetching price from Schwab API: {e}")
        return None

def get_option_expirations(ticker):
    if ticker == "MARKET":
        ticker = "$SPX"
    try:
        response = client.option_expiration_chain(ticker)
        if not response.ok:
            return []
        response_json = response.json()
        if response_json and 'expirationList' in response_json:
            expiration_dates = [item['expirationDate'] for item in response_json['expirationList']]
            return sorted(expiration_dates)
        return []
    except Exception as e:
        print(f"Error fetching option expirations: {e}")
        return []

def get_color_with_opacity(value, max_value, base_color, color_intensity=True):
    """Get color with opacity based on value."""
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

def create_exposure_chart(calls, puts, exposure_type, title, S, strike_range=0.02, show_calls=True, show_puts=True, show_net=True, color_intensity=True, call_color='#00FF00', put_color='#FF0000', selected_expiries=None, perspective='Customer'):
    # Ensure the exposure_type column exists
    if exposure_type not in calls.columns or exposure_type not in puts.columns:
        print(f"Warning: {exposure_type} not found in data")  # Fixed f-string syntax
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
    
    # Calculate total net exposure
    total_call_exposure = calls_df[exposure_type].sum() if not calls_df.empty else 0
    total_put_exposure = puts_df[exposure_type].sum() if not puts_df.empty else 0
    
    if exposure_type == 'GEX':
        # For gamma exposure, subtract puts from calls (puts are positive in calculation)
        total_net_exposure = total_call_exposure - total_put_exposure
    elif exposure_type == 'DEX':
        # For delta exposure, sum them (puts are negative in calculation)
        total_net_exposure = total_call_exposure + total_put_exposure
    else:
        # For other exposures, sum them (puts are same sign as calls usually)
        total_net_exposure = total_call_exposure + total_put_exposure
    
    # Apply perspective (Dealer = Short, flip the sign of Net values)
    if perspective == 'Dealer':
        total_net_exposure = total_net_exposure * -1
    
    # Create the main title and net exposure as separate annotations
    fig = go.Figure()
    
    # Define colors
    grid_color = '#333333'
    text_color = '#CCCCCC'
    background_color = '#1E1E1E'
    
    if show_calls and not calls_df.empty:
        if color_intensity:
            # Calculate color intensity for calls with proper scaling
            max_call_value = max(abs(calls_df[exposure_type].max()), abs(calls_df[exposure_type].min()))
            if max_call_value == 0:
                max_call_value = 1  # Prevent division by zero
            call_colors = [get_color_with_opacity(volume, max_call_value, call_color, color_intensity) for volume in calls_df[exposure_type]]
        else:
            call_colors = call_color
            
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
        if color_intensity:
            # Calculate color intensity for puts with proper scaling
            max_put_value = max(abs(puts_df[exposure_type].max()), abs(puts_df[exposure_type].min()))
            if max_put_value == 0:
                max_put_value = 1  # Prevent division by zero
            put_colors = [get_color_with_opacity(volume, max_put_value, put_color, color_intensity) for volume in puts_df[exposure_type]]
        else:
            put_colors = put_color
            
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
            
            # Apply perspective (Dealer = Short, flip the sign of Net values)
            if perspective == 'Dealer':
                net_value = net_value * -1
                
            net_exposure.append(net_value)
        
        if color_intensity:
            # Calculate color intensity for net values with proper scaling
            max_net_value = max(abs(min(net_exposure)), abs(max(net_exposure)))
            if max_net_value == 0:
                max_net_value = 1  # Prevent division by zero
            net_colors = []
            for volume in net_exposure:
                base_color = call_color if volume >= 0 else put_color
                net_colors.append(get_color_with_opacity(volume, max_net_value, base_color, color_intensity))
        else:
            net_colors = [call_color if volume >= 0 else put_color for volume in net_exposure]
        
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
    padding = (max_strike - min_strike) * 0.1
    
    # Add expiry info to title if multiple expiries are selected
    chart_title = title
    if selected_expiries and len(selected_expiries) > 1:
        chart_title = f"{title} ({len(selected_expiries)} expiries)"
    
    # Update layout with improved styling and split title
    fig.update_layout(
        title=dict(
            text=chart_title,  # Main title with expiry info
            font=dict(color=text_color, size=16),
            x=0.5,
            xanchor='center',
            y=0.95  # Adjust title position to make room for net exposure
        ),
        annotations=list(fig.layout.annotations) + [
            dict(
                text=f"Net: {format_large_number(abs(total_net_exposure))}",
                x=1.05,
                y=1.05,  # Adjusted from 1.1 to 1.05 to move it slightly down
                xref='paper',
                yref='paper',
                xanchor='right',
                yanchor='middle',
                showarrow=False,
                font=dict(
                    size=16,
                    color=call_color if total_net_exposure >= 0 else put_color
                )
            )
        ],
        xaxis=dict(
            title='',
            title_font=dict(color=text_color),
            tickfont=dict(color=text_color, size=12),
            gridcolor=grid_color,
            linecolor=grid_color,
            range=[min_strike - padding, max_strike + padding],
            tickmode='linear',
            dtick=math.ceil((max_strike - min_strike) / 10),
            showgrid=False,
            zeroline=True,
            zerolinecolor=grid_color,
            tickangle=45,
            tickformat='.0f',
            showticklabels=True,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor=text_color,
            automargin=True
        ),
        yaxis=dict(
            title='',
            title_font=dict(color=text_color),
            tickfont=dict(color=text_color),
            gridcolor=grid_color,
            linecolor=grid_color,
            showgrid=False,
            zeroline=True,
            zerolinecolor=grid_color
        ),
        barmode='relative',
        hovermode='x unified',
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font=dict(color=text_color),
        showlegend=False,  # Removed legend
        bargap=0.1,
        bargroupgap=0.1,
        margin=dict(l=50, r=50, t=70, b=100),  # Increased top margin for net value
        hoverlabel=dict(
            bgcolor=background_color,
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100
    )
    
    # Add hover spikes
    fig.update_xaxes(showspikes=True, spikecolor=text_color, spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor=text_color, spikethickness=1)
    
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
        font=dict(color='white')
    )
    
    return fig.to_json()

def create_options_volume_chart(calls, puts, S, strike_range=0.02, call_color='#00FF00', put_color='#FF0000', color_intensity=True, show_calls=True, show_puts=True, show_net=True, selected_expiries=None, perspective='Customer'):
    # Filter strikes within range
    min_strike = S * (1 - strike_range)
    max_strike = S * (1 + strike_range)
    
    calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    # Create figure
    fig = go.Figure()
    
    # Add call volume bars
    if show_calls and not calls.empty:
        if color_intensity:
            # Calculate color intensity for calls
            max_call_volume = calls['volume'].max()
            r = int(call_color[1:3], 16)
            g = int(call_color[3:5], 16)
            b = int(call_color[5:7], 16)
            # Ensure opacity is between 0.1 and 1.0
            call_colors = [get_color_with_opacity(volume, max_call_volume, call_color) for volume in calls['volume']]
        else:
            call_colors = call_color
            
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
        if color_intensity:
            # Calculate color intensity for puts
            max_put_volume = puts['volume'].max()
            r = int(put_color[1:3], 16)
            g = int(put_color[3:5], 16)
            b = int(put_color[5:7], 16)
            # Ensure opacity is between 0.1 and 1.0
            put_colors = [get_color_with_opacity(volume, max_put_volume, put_color) for volume in puts['volume']]
        else:
            put_colors = put_color
            
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
        all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        net_volume = []
        
        for strike in all_strikes:
            call_vol = calls[calls['strike'] == strike]['volume'].sum() if not calls.empty else 0
            put_vol = puts[puts['strike'] == strike]['volume'].sum() if not puts.empty else 0
            net_vol = call_vol - put_vol
            
            # Apply perspective (Dealer = Short, flip the sign of Net values)
            if perspective == 'Dealer':
                net_vol = net_vol * -1
                
            net_volume.append(net_vol)
        
        if color_intensity:
            # Calculate color intensity for net values
            max_net_volume = max(abs(min(net_volume)), abs(max(net_volume)))
            net_colors = []
            for volume in net_volume:
                if volume >= 0:
                    net_colors.append(get_color_with_opacity(volume, max_net_volume, call_color))
                else:
                    net_colors.append(get_color_with_opacity(abs(volume), max_net_volume, put_color))
        else:
            net_colors = [call_color if volume >= 0 else put_color for volume in net_volume]
        
        fig.add_trace(go.Bar(
            x=all_strikes,
            y=net_volume,
            name='Net',
            marker_color=net_colors,
            text=[f"{vol:,.0f}" for vol in net_volume],
            textposition='auto',
            hovertemplate='Strike: %{x}<br>Net Volume: %{y}<extra></extra>',
            marker_line_width=0
        ))
    
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
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(color='#CCCCCC', size=16),
            x=0.5,
            xanchor='center'
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
            tickangle=45,
            tickformat='.0f',
            showticklabels=True,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor='#CCCCCC',
            automargin=True
        ),
        yaxis=dict(
            title='',
            title_font=dict(color='#CCCCCC'),
            tickfont=dict(color='#CCCCCC'),
            gridcolor='#333333',
            linecolor='#333333',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#333333'
        ),
        barmode='relative',
        hovermode='x unified',
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
        margin=dict(l=50, r=50, t=50, b=100),
        hoverlabel=dict(
            bgcolor='#1E1E1E',
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100,
        showlegend=True,
        height=400
    )
    
    # Add hover spikes
    fig.update_xaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    
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

def get_price_history(ticker):
    if ticker == "MARKET":
        ticker = "$SPX"
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
            frequency=1,
            startDate=int(start_date.timestamp() * 1000),
            endDate=int(end_date.timestamp() * 1000),
            needExtendedHoursData=True
        )
        
        if not response.ok:
            return None
            
        data = response.json()
        if not data or 'candles' not in data:
            return None
            
        # Filter for market hours
        candles = filter_market_hours(data['candles'])
        if not candles:
            return None
            
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
        print(f"[DEBUG] Error fetching price history: {e}")
        return None

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

def create_price_chart(price_data, calls=None, puts=None, exposure_levels_types=[], exposure_levels_count=3, call_color='#00FF00', put_color='#FF0000', strike_range=0.02, use_heikin_ashi=False):
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
    padding = price_range * 0.1  # 10% padding
    
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
            range=[price_min - padding, price_max + padding],
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
        height=500,  # Reduced height after removing momentum subplot
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
        
        for i, exposure_levels_type in enumerate(exposure_levels_types):
            # Determine column name based on type
            col_name = exposure_levels_type
            if exposure_levels_type == 'Vanna': col_name = 'VEX'
            
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
                
                # Pick dash style
                dash_style = dash_styles[i % len(dash_styles)]
                
                # Add horizontal lines for each level
                for strike, val in top_levels:
                     # Calculate color intensity based on value
                    max_val = max(abs(v) for _, v in top_levels)
                    if max_val == 0: max_val = 1
                    intensity = max(0.1, min(1.0, abs(val) / max_val))
                    
                    # Determine color: Green for positive, Red for negative
                    is_positive = val >= 0
                    color = call_color if is_positive else put_color
                    
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    rgba_color = f'rgba({r}, {g}, {b}, {intensity:.2f})'
                    
                    # Add the horizontal line
                    fig.add_hline(
                        y=strike,
                        line_dash=dash_style,
                        line_color=rgba_color,
                        line_width=1
                    )
                    
                    # Add separate annotation for the text
                    # Offset vertically by index to prevent stacking perfectly
                    y_offset_pixels = 5 + (i * 15)
                    
                    fig.add_annotation(
                        x=1,
                        y=strike,
                        xref="paper",
                        yref="y",
                        text=f"{exposure_levels_type}: {format_large_number(val)}",
                        showarrow=False,
                        font=dict(
                            size=10,
                            color=rgba_color
                        ),
                        xanchor='left',
                        yanchor='top',
                        xshift=-105, # Moved further left to accommodate type name
                        yshift=-y_offset_pixels
                    )

    return fig.to_json()

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





def create_historical_bubble_levels_chart(ticker, strike_range, call_color='#00FFA3', put_color='#FF3B3B', exposure_type='gamma'):
    """Create a chart showing price and exposure (gamma, delta, or vanna) over time for the last hour."""
    # Get interval data from database
    interval_data = get_interval_data(ticker)
    
    if not interval_data:
        return None
    
    # Calculate timestamp for 1 hour ago
    one_hour_ago = int(time.time()) - 3600
    
    # Group data by timestamp and filter for last hour
    data_by_time = {}
    for row in interval_data:
        timestamp = row[0]
        if timestamp < one_hour_ago:
            continue  # Skip data older than 1 hour
            
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
    for i, exposure in enumerate(exposures):
        dt = timestamps[i]
        max_exposure = max_exposure_by_time[dt]
        if max_exposure == 0:
            max_exposure = 1  # Prevent division by zero
            
        # Calculate color
        if exposure >= 0:
            colors.append(get_color_with_opacity(exposure, max_exposure, call_color, True))
        else:
            colors.append(get_color_with_opacity(exposure, max_exposure, put_color, True))
            
        # Calculate bubble size (scaled to the max exposure for this time slice)
        size = max(4, min(25, abs(exposure) * 20 / max_exposure))
        bubble_sizes.append(size)
    
    # Create figure
    fig = go.Figure()
    
    # Add exposure bubbles for each strike first (bottom layer)
    exposure_name = {
        'gamma': 'Gamma',
        'delta': 'Delta',
        'vanna': 'Vanna',
        'charm': 'Charm'
    }.get(exposure_type, 'Exposure')
    
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
        yaxis='y1'
    ))
    
    # Add price line last (top layer)
    unique_times = sorted(set(timestamps))
    unique_prices = [data_by_time[int(t.timestamp())]['price'] for t in unique_times]
    fig.add_trace(go.Scatter(
        x=unique_times,
        y=unique_prices,
        mode='lines',
        name='Price',
        line=dict(color='gold', width=2),
        yaxis='y1'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Historical Bubble Levels - {exposure_name}',
            font=dict(color='#CCCCCC', size=24),  # Increased from 16
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time (Last Hour)',
            title_font=dict(color='#CCCCCC', size=18),  # Added size
            tickfont=dict(color='#CCCCCC', size=16),  # Added size
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
            title_font=dict(color='#CCCCCC', size=18),  # Added size
            tickfont=dict(color='#CCCCCC', size=16),  # Added size
            gridcolor='#333333',
            linecolor='#333333',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#333333'
        ),
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='#CCCCCC', size=16),  # Added size
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#CCCCCC', size=16),  # Added size
            bgcolor='#1E1E1E'
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        height=800,
        width=1200
    )
    
    # Convert figure to image with higher scale factor
    img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
    return img_bytes



def create_premium_chart(calls, puts, S, strike_range=0.02, call_color='#00FF00', put_color='#FF0000', color_intensity=True, show_calls=True, show_puts=True, show_net=True, selected_expiries=None, perspective='Customer'):
    # Filter strikes within range
    min_strike = S * (1 - strike_range)
    max_strike = S * (1 + strike_range)
    
    calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    # Create figure
    fig = go.Figure()
    
    # Add call premium bars
    if show_calls and not calls.empty:
        if color_intensity:
            # Calculate color intensity for calls
            max_call_premium = calls['lastPrice'].max()
            call_colors = [get_color_with_opacity(premium, max_call_premium, call_color) for premium in calls['lastPrice']]
        else:
            call_colors = call_color
            
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
        if color_intensity:
            # Calculate color intensity for puts
            max_put_premium = puts['lastPrice'].max()
            put_colors = [get_color_with_opacity(premium, max_put_premium, put_color) for premium in puts['lastPrice']]
        else:
            put_colors = put_color
            
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
        all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        net_premium = []
        
        for strike in all_strikes:
            call_prem = calls[calls['strike'] == strike]['lastPrice'].sum() if not calls.empty else 0
            put_prem = puts[puts['strike'] == strike]['lastPrice'].sum() if not puts.empty else 0
            net_prem = call_prem - put_prem
            
            # Apply perspective (Dealer = Short, flip the sign of Net values)
            if perspective == 'Dealer':
                net_prem = net_prem * -1
                
            net_premium.append(net_prem)
        
        if color_intensity:
            # Calculate color intensity for net values
            max_net_premium = max(abs(min(net_premium)), abs(max(net_premium)))
            net_colors = []
            for premium in net_premium:
                if premium >= 0:
                    net_colors.append(get_color_with_opacity(premium, max_net_premium, call_color))
                else:
                    net_colors.append(get_color_with_opacity(abs(premium), max_net_premium, put_color))
        else:
            net_colors = [call_color if premium >= 0 else put_color for premium in net_premium]
        
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
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(color='#CCCCCC', size=16),
            x=0.5,
            xanchor='center'
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
            tickangle=45,
            tickformat='.0f',
            showticklabels=True,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor='#CCCCCC',
            automargin=True
        ),
        yaxis=dict(
            title='',
            title_font=dict(color='#CCCCCC'),
            tickfont=dict(color='#CCCCCC'),
            gridcolor='#333333',
            linecolor='#333333',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#333333'
        ),
        barmode='relative',
        hovermode='x unified',
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
        margin=dict(l=50, r=50, t=50, b=100),
        hoverlabel=dict(
            bgcolor='#1E1E1E',
            font_size=12,
            font_family="Arial"
        ),
        spikedistance=1000,
        hoverdistance=100,
        showlegend=True,
        height=400
    )
    
    # Add hover spikes
    fig.update_xaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    
    return fig.to_json()

def create_centroid_chart(ticker, call_color='#00FF00', put_color='#FF0000', selected_expiries=None):
    """Create a chart showing call and put centroids over time with price line"""
    # Check if we're in market hours
    est = pytz.timezone('US/Eastern')
    current_time_est = datetime.now(est)
    
    # Get centroid data from database
    centroid_data = get_centroid_data(ticker)
    
    if not centroid_data:
        # Determine appropriate message based on time
        if current_time_est.weekday() >= 5:  # Weekend
            chart_title = 'Call vs Put Centroid Map (Market Closed - Weekend)'
        elif current_time_est.hour < 9 or (current_time_est.hour == 9 and current_time_est.minute < 30):
            chart_title = 'Call vs Put Centroid Map (Pre-Market)'
        elif current_time_est.hour >= 16:
            chart_title = 'Call vs Put Centroid Map (After Hours)'
        else:
            chart_title = 'Call vs Put Centroid Map (No Data)'
        
        # Return empty chart if no data
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text=chart_title,
                font=dict(color='#CCCCCC', size=24),
                x=0.5,
                xanchor='center'
            ),
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='#CCCCCC', size=16),
            xaxis=dict(title='Time', title_font=dict(color='#CCCCCC', size=18), tickfont=dict(color='#CCCCCC', size=16)),
            yaxis=dict(title='Price/Strike', title_font=dict(color='#CCCCCC', size=18), tickfont=dict(color='#CCCCCC', size=16)),
            height=800,
            width=1200
        )
        # Convert figure to image with higher scale factor
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        return img_bytes
    
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
    
    # Update layout to match interval map style
    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(color='#CCCCCC', size=24),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time',
            title_font=dict(color='#CCCCCC', size=18),
            tickfont=dict(color='#CCCCCC', size=16),
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
            title_font=dict(color='#CCCCCC', size=18),
            tickfont=dict(color='#CCCCCC', size=16),
            gridcolor='#333333',
            linecolor='#333333',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#333333'
        ),
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='#CCCCCC', size=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#CCCCCC', size=16),
            bgcolor='#1E1E1E'
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        height=800,
        width=1200
    )
    
    # Add hover spikes
    fig.update_xaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor='#CCCCCC', spikethickness=1)
    
    # Convert figure to image with higher scale factor
    img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
    return img_bytes

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
    
    for date in dates:
        try:
            calls, puts = fetch_options_for_date(ticker, date, exposure_metric=exposure_metric, delta_adjusted=delta_adjusted, calculate_in_notional=calculate_in_notional)
            if not calls.empty:
                all_calls.append(calls)
            if not puts.empty:
                all_puts.append(puts)
        except Exception as e:
            print(f"Error fetching options for {date}: {e}")
            continue
    
    # Combine all dataframes
    combined_calls = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    combined_puts = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
    
    return combined_calls, combined_puts

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>EzOptions - Schwab</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
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
            padding: 8px;
            border-top: 1px solid #444;
            display: flex;
            gap: 8px;
        }
        .expiry-buttons button {
            padding: 4px 8px;
            font-size: 11px;
            border-radius: 4px;
            border: 1px solid #555;
            background-color: #444;
            color: white;
            cursor: pointer;
            flex: 1;
        }
        .expiry-buttons button:hover {
            background-color: #555;
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
            grid-column: 1 / -1;
            margin-bottom: 5px;
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
            height: 400px;
            background-color: #1E1E1E;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .historical-bubble-container .chart-container {
            height: 100%;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .historical-bubble-container .chart-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }
        
        .chart-container {
            padding: 5px;
            height: 400px;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-top">
                <div class="title">EzDuz1t Options</div>
                <div class="controls">
                    <div class="control-group">
                        <label for="ticker">Ticker:</label>
                        <input type="text" id="ticker" placeholder="Enter Ticker" value="SPY">
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
                                    <button type="button" id="selectAllExpiry">All</button>
                                    <button type="button" id="clearAllExpiry">Clear</button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="stream-control">
                        <button id="streamToggle">Auto-Update</button>
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
                            <option value="OI Weighted by Volume">OI Weighted by Volume</option>
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
                        <label for="perspective">Perspective:</label>
                        <select id="perspective">
                            <option value="Customer" selected>Customer</option>
                            <option value="Dealer">Dealer</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <input type="checkbox" id="color_intensity" checked>
                        <label for="color_intensity">Color Intensity</label>
                    </div>
                    <div class="control-group">
                        <label>Price Levels:</label>
                        <div class="levels-dropdown">
                            <div class="levels-display" id="levels-display">
                                <span id="levels-text">None</span>
                            </div>
                            <div class="levels-options" id="levels-options">
                                <div class="levels-option"><input type="checkbox" value="GEX" id="lvl-GEX"><label for="lvl-GEX">GEX</label></div>
                                <div class="levels-option"><input type="checkbox" value="DEX" id="lvl-DEX"><label for="lvl-DEX">DEX</label></div>
                                <div class="levels-option"><input type="checkbox" value="VEX" id="lvl-VEX"><label for="lvl-VEX">Vanna</label></div>
                                <div class="levels-option"><input type="checkbox" value="Charm" id="lvl-Charm"><label for="lvl-Charm">Charm</label></div>
                                <div class="levels-option"><input type="checkbox" value="Speed" id="lvl-Speed"><label for="lvl-Speed">Speed</label></div>
                                <div class="levels-option"><input type="checkbox" value="Vomma" id="lvl-Vomma"><label for="lvl-Vomma">Vomma</label></div>
                                <div class="levels-option"><input type="checkbox" value="Color" id="lvl-Color"><label for="lvl-Color">Color</label></div>
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
                <div class="chart-container" id="price-chart"></div>
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
        let chartRevisions = {}; // Store chart revisions
        let callColor = '#00FF00';
        let putColor = '#FF0000';
        let lastData = {}; // Store last received data
        let updateInProgress = false;
        let isStreaming = true;
        

        
        // Update colors when color pickers change
        document.getElementById('call_color').addEventListener('change', function(e) {
            callColor = e.target.value;
            updateData();
        });
        
        document.getElementById('put_color').addEventListener('change', function(e) {
            putColor = e.target.value;
            updateData();
        });
        
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

        // Perspective listener
        document.getElementById('perspective').addEventListener('change', updateData);
        document.getElementById('exposure_metric').addEventListener('change', updateData);
        document.getElementById('levels_count').addEventListener('input', updateData);

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
            const colorIntensity = document.getElementById('color_intensity').checked;
            const levelsTypes = Array.from(document.querySelectorAll('.levels-option input:checked')).map(cb => cb.value);
            const levelsCount = parseInt(document.getElementById('levels_count').value);
            const useHeikinAshi = document.getElementById('use_heikin_ashi').checked;
            const useRange = document.getElementById('use_range').checked;
            const exposureMetric = document.getElementById('exposure_metric').value;
            const deltaAdjusted = document.getElementById('delta_adjusted_exposures').checked;
            const calculateInNotional = document.getElementById('calculate_in_notional').checked;
            const strikeRange = parseFloat(document.getElementById('strike_range').value) / 100;
            const perspective = document.getElementById('perspective').value;
            
            // Get visible charts
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
                show_volume: document.getElementById('volume').checked,
                show_large_trades: document.getElementById('large_trades').checked,
                show_premium: document.getElementById('premium').checked,
                show_centroid: document.getElementById('centroid').checked
            };
            
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
                    color_intensity: colorIntensity,
                    levels_types: levelsTypes,
                    levels_count: levelsCount,
                    use_heikin_ashi: useHeikinAshi,
                    use_range: useRange,
                    exposure_metric: exposureMetric,
                    delta_adjusted: deltaAdjusted,
                    calculate_in_notional: calculateInNotional,
                    strike_range: strikeRange,
                    perspective: perspective,
                    call_color: callColor,
                    put_color: putColor,
                    ...visibleCharts
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('Error:', data.error);
                    return;
                }
                
                // Only update if data has changed
                if (JSON.stringify(data) !== JSON.stringify(lastData)) {
                    updateCharts(data);
                    updatePriceInfo(data.price_info);
                    lastData = data;
                }
            })
            .catch(error => {
                console.error('Error fetching data:', error);
            })
            .finally(() => {
                updateInProgress = false;
            });
        }
        
        function updateCharts(data) {
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
                volume: document.getElementById('volume').checked,
                large_trades: document.getElementById('large_trades').checked,
                premium: document.getElementById('premium').checked,
                centroid: document.getElementById('centroid').checked
            };
            
            // Handle price chart separately
            if (selectedCharts.price && data.price) {
                let priceContainer = document.querySelector('.price-chart-container');
                if (!priceContainer) {
                    priceContainer = document.createElement('div');
                    priceContainer.className = 'price-chart-container';
                    const chartDiv = document.createElement('div');
                    chartDiv.className = 'chart-container';
                    chartDiv.id = 'price-chart';
                    priceContainer.appendChild(chartDiv);
                    document.getElementById('chart-grid').insertBefore(priceContainer, document.getElementById('chart-grid').firstChild);
                }
                priceContainer.style.display = 'block';
                
                const chartData = JSON.parse(data.price);
                
                // Preserve user's chart manipulation state
                if (!chartRevisions.price) {
                    chartRevisions.price = Date.now();
                }
                chartData.layout.uirevision = chartRevisions.price;
                
                // Configure chart sizing to fill container
                chartData.layout.autosize = true;
                chartData.layout.width = null;
                chartData.layout.height = null;
                chartData.layout.margin = {l: 50, r: 120, t: 30, b: 20};
                
                // Update chart colors
                if (chartData.data[0].type === 'candlestick') {
                    chartData.data[0].increasing.line.color = callColor;
                    chartData.data[0].increasing.fillcolor = callColor;
                    chartData.data[0].decreasing.line.color = putColor;
                    chartData.data[0].decreasing.fillcolor = putColor;
                }
                
                // Simplified update logic with preserved state
                const config = {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                    displaylogo: false,
                    scrollZoom: true,
                    useResizeHandler: true,
                    style: {width: "100%", height: "100%"}
                };
                
                if (charts.price) {
                    // Get current view state before update
                    const currentView = Plotly.relayout('price-chart', {});
                    
                    // Update the chart while preserving the view
                    Plotly.react('price-chart', chartData.data, {
                        ...chartData.layout,
                        ...currentView
                    }, {
                        ...config,
                        animate: false,
                        transition: {
                            duration: 0
                        }
                    });
                } else {
                    charts.price = Plotly.newPlot('price-chart', chartData.data, chartData.layout, config);
                }
            } else if (!selectedCharts.price) {
                const priceContainer = document.querySelector('.price-chart-container');
                if (priceContainer) {
                    priceContainer.style.display = 'none';
                    delete charts.price;
                    delete chartRevisions.price;
                }
            }
            
            // Handle historical bubble levels and centroid
            const historicalBubbles = ['gex_historical_bubble', 'dex_historical_bubble', 'vanna_historical_bubble', 'charm_historical_bubble', 'centroid'];
            const historicalBubblesRow = document.getElementById('historical-bubbles-row');
            
            // Count enabled historical bubble levels
            const enabledBubbles = historicalBubbles.filter(bubbleType => selectedCharts[bubbleType]);
            
            // Clear existing containers and classes
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
                
                // Add or update selected historical bubble levels
                enabledBubbles.forEach(bubbleType => {
                    const bubbleContainer = document.createElement('div');
                    bubbleContainer.className = 'historical-bubble-container';
                    
                    const chartDiv = document.createElement('div');
                    chartDiv.className = 'chart-container';
                    chartDiv.id = `${bubbleType.replace('_', '-')}-chart`;
                    
                    // Create image element
                    const img = document.createElement('img');
                    img.style.width = '100%';
                    img.style.height = '100%';
                    img.style.objectFit = 'contain';
                    img.style.imageRendering = 'crisp-edges';
                    
                    if (data[bubbleType]) {
                        img.src = data[bubbleType];
                    }
                    
                    chartDiv.appendChild(img);
                    bubbleContainer.appendChild(chartDiv);
                    historicalBubblesRow.appendChild(bubbleContainer);
                });
            }
            
            // Clean up disabled historical bubble levels from charts object
            historicalBubbles.forEach(bubbleType => {
                if (!selectedCharts[bubbleType]) {
                    delete charts[bubbleType];
                    delete chartRevisions[bubbleType];
                }
            });
            
            // Handle other charts
            let chartsGrid = document.querySelector('.charts-grid');
            if (!chartsGrid) {
                chartsGrid = document.createElement('div');
                chartsGrid.className = 'charts-grid';
                document.getElementById('chart-grid').appendChild(chartsGrid);
            } else {
                // Clear existing containers and classes
                chartsGrid.innerHTML = '';
                chartsGrid.className = 'charts-grid';
            }
            
            // Count enabled regular charts (excluding price and historical bubble levels)
            const regularCharts = Object.entries(selectedCharts).filter(([key, selected]) => 
                selected && !['price'].includes(key) && !historicalBubbles.includes(key) && data[key]
            );
            
            // Hide the charts grid if no regular charts are enabled
            if (regularCharts.length === 0) {
                chartsGrid.style.display = 'none';
            } else {
                chartsGrid.style.display = 'grid';
                
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
                    
                    try {
                        // Special handling for options chain (HTML table)
                        if (key === 'large_trades') {
                            newContainer.innerHTML = data[key];
                        } else {
                            const chartData = JSON.parse(data[key]);
                            
                            // Configure chart sizing to fill container
                            chartData.layout.autosize = true;
                            chartData.layout.width = null;
                            chartData.layout.height = null;
                            chartData.layout.margin = {l: 50, r: 50, t: 50, b: 50};
                            
                            if (!chartRevisions[key]) {
                                chartRevisions[key] = Date.now();
                            }
                            chartData.layout.uirevision = chartRevisions[key];
                            
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
                                Plotly.react(`${key}-chart`, chartData.data, chartData.layout, {
                                    ...config,
                                    animate: false,
                                    transition: {
                                        duration: 0
                                    }
                                });
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
                    delete chartRevisions[key];
                }
            });
            

        }
        
        function updatePriceInfo(info) {
            const priceInfo = document.getElementById('price-info');
            const selectedExpiries = lastData.selected_expiries || [];
            const expiryText = selectedExpiries.length > 1 ? 
                `${selectedExpiries.length} expiries selected` : 
                selectedExpiries[0] || 'No expiry selected';
            
            priceInfo.innerHTML = `
                <div>Current Price: $${info.current_price}</div>
                <div>High: $${info.high}</div>
                <div>Low: $${info.low}</div>
                <div class="${info.net_change >= 0 ? 'green' : 'red'}">
                    ${info.net_change >= 0 ? '+' : ''}${info.net_change} (${info.net_percent >= 0 ? '+' : ''}${info.net_percent}%)
                </div>
                <div>Vol Ratio: <span style="color: ${callColor}">${info.call_percentage}%</span>/<span style="color: ${putColor}">${info.put_percentage}%</span></div>
                <div>Expiries: ${expiryText}</div>
            `;
        }
        
        function loadExpirations() {
            const ticker = document.getElementById('ticker').value;
            fetch(`/expirations/${ticker}`)
                .then(response => response.json())
                .then(data => {
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
                    
                    // Re-add the buttons at the end
                    optionsContainer.appendChild(buttons);
                    
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
        
        // Initial load
        loadExpirations();
        
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
            } else {
                clearInterval(updateInterval);
            }
        }
        
        document.getElementById('streamToggle').addEventListener('click', toggleStreaming);

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
    </script>
</body>
</html>
    ''')

@app.route('/expirations/<ticker>')
def get_expirations(ticker):
    ticker = format_ticker(ticker)
    expirations = get_option_expirations(ticker)
    return jsonify(expirations)

@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    ticker = data.get('ticker')
    expiry = data.get('expiry')  # This can now be a list or single value
    
    ticker = format_ticker(ticker) 
    if not ticker or not expiry:
        return jsonify({'error': 'Missing ticker or expiry'})
    
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
        
        # Clear old data at the end of the day
        current_time = datetime.now()
        if current_time.hour == 23 and current_time.minute == 59:
            clear_old_data()
        
        # Get fresh price data
        price_data = get_price_history(ticker)
        
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
        color_intensity = data.get('color_intensity', True)
        call_color = data.get('call_color', '#00ff00')
        put_color = data.get('put_color', '#ff0000')
        exposure_levels_types = data.get('levels_types', [])
        exposure_levels_count = int(data.get('levels_count', 3))
        use_heikin_ashi = data.get('use_heikin_ashi', False)
        perspective = data.get('perspective', 'Customer')
 
        
        response = {}
        
        # Create charts based on visibility settings
        if data.get('show_price', True):
            response['price'] = create_price_chart(
                price_data=price_data,
                calls=calls,
                puts=puts,
                exposure_levels_types=exposure_levels_types,
                exposure_levels_count=exposure_levels_count,
                call_color=call_color,
                put_color=put_color,
                strike_range=strike_range,
                use_heikin_ashi=use_heikin_ashi
            )
        
        if data.get('show_gex_historical_bubble', True):
            img_bytes = create_historical_bubble_levels_chart(ticker, strike_range, call_color, put_color, 'gamma')
            if img_bytes:
                response['gex_historical_bubble'] = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        
        if data.get('show_dex_historical_bubble', True):
            img_bytes = create_historical_bubble_levels_chart(ticker, strike_range, call_color, put_color, 'delta')
            if img_bytes:
                response['dex_historical_bubble'] = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        
        if data.get('show_vanna_historical_bubble', True):
            img_bytes = create_historical_bubble_levels_chart(ticker, strike_range, call_color, put_color, 'vanna')
            if img_bytes:
                response['vanna_historical_bubble'] = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        
        if data.get('show_charm_historical_bubble', True):
            img_bytes = create_historical_bubble_levels_chart(ticker, strike_range, call_color, put_color, 'charm')
            if img_bytes:
                response['charm_historical_bubble'] = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        
        if data.get('show_gamma', True):
            response['gamma'] = create_exposure_chart(calls, puts, "GEX", "Gamma Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, color_intensity, call_color, put_color, expiry_dates, perspective)
        
        if data.get('show_delta', True):
            response['delta'] = create_exposure_chart(calls, puts, "DEX", "Delta Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, color_intensity, call_color, put_color, expiry_dates, perspective)
        
        if data.get('show_vanna', True):
            response['vanna'] = create_exposure_chart(calls, puts, "VEX", "Vanna Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, color_intensity, call_color, put_color, expiry_dates, perspective)
        
        if data.get('show_charm', True):
            response['charm'] = create_exposure_chart(calls, puts, "Charm", "Charm Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, color_intensity, call_color, put_color, expiry_dates, perspective)
        
        if data.get('show_speed', True):
            response['speed'] = create_exposure_chart(calls, puts, "Speed", "Speed Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, color_intensity, call_color, put_color, expiry_dates, perspective)
        
        if data.get('show_vomma', True):
            response['vomma'] = create_exposure_chart(calls, puts, "Vomma", "Vomma Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, color_intensity, call_color, put_color, expiry_dates, perspective)

        if data.get('show_color', True):
            response['color'] = create_exposure_chart(calls, puts, "Color", "Color Exposure by Strike", S, strike_range, show_calls, show_puts, show_net, color_intensity, call_color, put_color, expiry_dates, perspective)
        
        if data.get('show_volume', True):
            response['volume'] = create_volume_chart(call_volume, put_volume, use_range, call_color, put_color, expiry_dates)
        
        if data.get('show_options_volume', True):
            response['options_volume'] = create_options_volume_chart(calls, puts, S, strike_range, call_color, put_color, color_intensity, show_calls, show_puts, show_net, expiry_dates, perspective)
        
        if data.get('show_premium', True):
            response['premium'] = create_premium_chart(calls, puts, S, strike_range, call_color, put_color, color_intensity, show_calls, show_puts, show_net, expiry_dates, perspective)
        
        if data.get('show_large_trades', True):
            response['large_trades'] = create_large_trades_table(calls, puts, S, strike_range, call_color, put_color, expiry_dates)
        
        if data.get('show_centroid', True):
            img_bytes = create_centroid_chart(ticker, call_color, put_color, expiry_dates)
            if img_bytes:
                response['centroid'] = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        
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
            quote_response = client.quote(ticker)
            if quote_response.ok:
                quote_data = quote_response.json()
                ticker_data = quote_data.get(ticker, {})
                quote = ticker_data.get('quote', {})
                
                response['price_info'] = {
                    'current_price': S,
                    'high': quote.get('highPrice', S),
                    'low': quote.get('lowPrice', S),
                    'net_change': quote.get('netChange', 0),
                    'net_percent': quote.get('netPercentChange', 0),
                    'call_volume': call_volume,
                    'put_volume': put_volume,
                    'total_volume': total_volume,
                    'call_percentage': call_percentage,
                    'put_percentage': put_percentage
                }
        except Exception as e:
            print(f"Error fetching quote data: {e}")
            response['price_info'] = {
                'current_price': S,
                'high': S,
                'low': S,
                'net_change': 0,
                'net_percent': 0,
                'call_volume': call_volume,
                'put_volume': put_volume,
                'total_volume': total_volume,
                'call_percentage': call_percentage,
                'put_percentage': put_percentage
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)