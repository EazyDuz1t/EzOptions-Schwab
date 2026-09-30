# EzOptions - Schwab Options Trading Dashboard

A real-time **gamma exposure (GEX)** and options Greeks dashboard for **Charles Schwab** accounts. It pulls live option chains from the Schwab API and charts dealer positioning by strike: GEX, DEX, vanna, charm, speed, vomma and color exposure. It also includes an exposure heatmap, a price-by-time exposure surface, historical level bubbles on a live price chart, and a sortable options chain. It works for stocks and ETFs, SPX / 0DTE, and index futures (/ES, /NQ and more).

<div align="center">
  <a href="https://github.com/EazyDuz1t/EzOptions-Schwab">
    <img src="https://img.shields.io/github/stars/EazyDuz1t/EzOptions-Schwab" alt="GitHub Repo stars"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License: GPL-3.0"/>
  </a>
</div>

## What is gamma exposure (GEX)?

Market makers hedge the options they sell by trading the underlying. Gamma exposure estimates how much hedging a move in price would force on them, strike by strike:

- **Positive GEX:** dealers tend to buy dips and sell rips, which dampens moves. Price often pins near large strikes.
- **Negative GEX:** dealers hedge in the direction of the move, which amplifies it.

The same idea applies to the other Greeks. Vanna shows how hedging changes with volatility, and charm shows how it changes with time, which matters most into expiry for 0DTE. EzOptions computes all of these from the live chain and your chosen weighting (open interest or volume).

## Features

### 📊 Real-Time Data
- Live options chain data from the Schwab API
- Real-time price streaming, with candles updating tick by tick
- Charts update live over a server push stream (no page refreshes)

### 📈 Options Analytics
- **Gamma Exposure (GEX)** and **Absolute GEX**: where dealer hedging dampens or amplifies moves
- **Delta Exposure (DEX)**: directional exposure by strike
- **Vanna Exposure (VEX)**: volatility-price sensitivity
- **Charm, Speed, Vomma and Color**: higher-order Greek exposures
- **Exposure Heatmap**: exposure by strike across every selected expiry
- **Exposure Surface**: smoothed price-by-time heatmap of any exposure, re-priced through the session close, with peak, trough and zero-flip lines
- **Historical Bubble Levels**: exposure levels recorded through the session and drawn on the price chart
- **Call vs Put Centroid Map**: where call and put volume is centered over the session
- **Expected Move**: from the at-the-money straddle

### 🎯 Interactive Charts
- Candlestick or Heikin-Ashi price chart (1 min to 1 hour) with indicators: SMA, EMA, WMA, VWAP, Bollinger Bands, RSI, MACD, ATR, SMC and more
- Drawing tools: horizontal lines, trend lines, boxes and labels
- Exposure levels and expected move drawn on the price chart
- Customizable strike range, multiple expirations and calls / puts / net views
- Pop-out and fullscreen charts, multiple themes

### ⚙️ Flexible Configuration
- Stocks, ETFs and indexes (SPY, QQQ, SPX, NDX, AAPL, ...)
- Index futures (/ES, /MES, /NQ, /MNQ, /RTY, /M2K, /YM, /MYM): exposures come from the cash index's options (SPX, NDX, RUT, DJX), with strikes scaled onto the futures price by the measured futures/index ratio (carry)
- Exposure weighting: Open Interest (default), Volume, Max OI vs Volume, or OI + Volume
- Optional delta-adjusted and notional ($) exposures
- Customizable call, put and max-level colors

## Installation

Requires Python 3 (developed on Python 3.11) and a Schwab brokerage account.

1. **Clone the repository**
   ```bash
   git clone https://github.com/EazyDuz1t/EzOptions-Schwab
   cd EzOptions-Schwab
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   SCHWAB_APP_KEY=your_app_key_here
   SCHWAB_APP_SECRET=your_app_secret_here
   SCHWAB_CALLBACK_URL=https://127.0.0.1
   ```

## Schwab API Setup

1. **Create a Schwab Developer Account**
   - Visit the [Schwab Developer Portal](https://developer.schwab.com/)
   - Register for a developer account
   - Create a new application with the **Market Data** product. Approval can take a few days.

2. **Get API Credentials**
   - App Key (Consumer Key)
   - App Secret (Consumer Secret)
   - Callback URL: use the same one in your Schwab app settings and your `.env` file (e.g. `https://127.0.0.1`)

3. **First login**
   - The first time you start the app, [schwabdev](https://github.com/tylerebowers/Schwabdev) opens the Schwab login page in your browser
   - Log in and approve access. You'll be redirected to your callback URL, which may show a "can't connect" page; that's expected.
   - Copy the full URL from the address bar and paste it into the terminal where the app is running
   - Tokens are saved and refreshed automatically. Schwab requires logging in again about every 7 days.

## Usage

1. **Start the application**
   ```bash
   python ezoptionsschwab.py
   ```

2. **Access the dashboard**
   - Open your browser to `http://localhost:5001`

3. **Using the dashboard**
   - Enter a ticker symbol (e.g., SPY, SPX, /ES, AAPL)
   - Select expiration dates, or use the quick buttons (Today / This Wk / +1 Wk / +2 Wks / +1 Mo / All)
   - Adjust the strike range with the slider
   - Turn charts on and off under **Charts**, and settings under **Filters**
   - Pause or resume live updates with the **Auto-Update** button

## Chart Types

### Price Chart
- Real-time candlestick or Heikin-Ashi charts with volume
- Top exposure levels (GEX, DEX, vanna, charm and more) and expected move as price lines
- Historical level bubbles for the session, plus technical indicators and drawing tools

![Price chart with vanna levels and historical level bubbles](https://i.imgur.com/8qGUMjr.png)

### Exposure Charts
- **Gamma Exposure**: where market maker hedging dampens or amplifies moves
- **Delta Exposure**: directional exposure by strike
- **Vanna Exposure**: volatility-price cross-sensitivity
- **Charm, Speed, Vomma and Color**: higher-order Greek exposures

![Delta, vanna, charm and color exposure by strike](https://i.imgur.com/1mjIyGH.png)

### Exposure Heatmap
- Exposure by strike and expiration, with the largest cell of each expiry highlighted

### Exposure Surface
- Today's positions re-priced across a grid of prices and times through the session close, so you can see where exposure builds or flips as price moves and time passes
- Peak, trough and zero-flip lines, with live candles on top
- Past columns come from what was recorded at the time, not repainted with hindsight

![Charm exposure surface with zero-flip, peak and trough lines and live candles](https://i.imgur.com/ck3OKDq.png)

### Historical Bubble Levels
- Exposure levels recorded about once a minute through the session
- Bubble size and color show exposure intensity over time
- Available for every exposure type

### Volume Analysis
- Options volume and open interest by strike
- Call/put volume ratio
- Call vs put centroid map

### Options Chain
- One sortable table per expiry
- Bid/ask/last, volume, open interest and implied volatility
- Spot-price divider and expected-move edges

![Options chain with volume bars, spot divider and expected-move edges](https://i.imgur.com/NBXG3p3.png)

## Configuration Options

### Strike Range
- Adjustable from 0.5% to 20% of the current price
- Filters options within the specified range

### Chart Toggles
- Show or hide calls, puts, or net exposure
- Coloring modes: Solid, Linear Intensity or Ranked Intensity
- Exposure weighting: Open Interest, Volume, Max OI vs Volume, or OI + Volume. It applies to every exposure formula (GEX/DEX/VEX/etc.).

### Color Customization
- Customizable call, put and max-level colors
- Intensity-based color scaling
- Multiple page themes

### Auto-Update
- Charts refresh every second while auto-update is on
- Pause/resume at any time

## Database

The application uses SQLite to store historical bubble levels data:
- Automatic database initialization
- Stores minute-by-minute exposure data
- Automatic cleanup of old data
- Only regular-session data is recorded (holidays and half-days come from Schwab's market-hours API)
- While the server is running, a background collector keeps recording the ticker/expiries on screen about once a minute, even when paused or with the browser closed. Tickers you switch away from are not recorded
- A page left open overnight reloads its expiry list on the new day. Selections made with the quick buttons (Today / This Wk / +1 Wk / +2 Wks / +1 Mo / All) are rules and re-apply each day and on ticker changes — Today always shows the nearest (0DTE) expiry, This Wk the rest of the current week. Hand-picked dates stay as picked; once they expire the selection is left empty until you pick again (hand-picking only today's expiry counts as 0DTE)
- If updates keep failing (e.g. Schwab maintenance), auto-update backs off and retries on its own instead of pausing

### Optional environment variables
- `EZOPTIONS_BACKGROUND_COLLECT=0`: turn off the background history collector
- `EZOPTIONS_DEBUG=1`: run Flask in debug mode (debugger + auto-reloader)

## Troubleshooting

- **"Schwab API client not initialized"**: check that `.env` has your app key, secret and callback URL, then restart the app.
- **Charts stop updating after about a week**: the Schwab login has expired. The app opens the Schwab login page again; complete the first-login steps (paste the redirect URL into the terminal). If nothing opens, restart the app.
- **Changes to the code don't show up**: stop the running server completely and start it again; a browser refresh isn't enough.

## License

Licensed under the [GNU General Public License v3.0](LICENSE). Please also comply with Schwab's API terms of service and any applicable regulations regarding financial data usage.

## Disclaimer

This software is for informational purposes only. It does not constitute financial advice. Trading options involves significant risk and may not be suitable for all investors. Always consult with a qualified financial advisor before making investment decisions.

## Support

For issues and questions:
- See the Troubleshooting section above
- Open an issue on GitHub
- Review the Schwab API documentation

## Contact
Discord - eazy101
