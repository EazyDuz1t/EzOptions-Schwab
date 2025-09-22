# EzOptions - Schwab Options Trading Dashboard

A real-time options trading dashboard that integrates with the Schwab API to provide comprehensive options analysis and visualization.

<div align="center">
  <a href="https://github.com/EazyDuz1t/EzOptions-Schwab">
    <img src="https://img.shields.io/github/stars/EazyDuz1t/EzOptions-Schwab" alt="GitHub Repo stars"/>
  </a>
</div>

## Features

### 📊 Real-Time Data
- Live options chain data from Schwab API
- Real-time price updates with auto-refresh
- Intraday price charts with volume analysis

### 📈 Advanced Options Analytics
- **Gamma Exposure (GEX)** - Visualize market maker hedging flows
- **Delta Exposure (DEX)** - Track directional exposure
- **Vanna Exposure** - Analyze volatility-price sensitivity
- **Charm, Speed, and Vomma** - Advanced Greek exposures
- **Historical Bubble Levels** - Historical exposure tracking over time

### 🎯 Interactive Charts
- Customizable strike range filtering
- Multiple expiration date support
- Color-coded exposure visualization
- Heikin-Ashi candlestick charts
- Volume analysis and ratios

### ⚙️ Flexible Configuration
- Multiple ticker support (SPY, SPX, etc.)
- Adjustable strike range percentages
- Customizable chart colors
- Toggle between different chart types
- Auto-update streaming data
 - Choose exposure weighting: Open Interest (default) or Volume

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/EazyDuz1t/EzOptions-Schwab
   cd ezoptions-schwab
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
   SCHWAB_CALLBACK_URL=your_callback_url_here
   ```

## Schwab API Setup

1. **Create a Schwab Developer Account**
   - Visit [Schwab Developer Portal](https://developer.schwab.com/)
   - Register for a developer account
   - Create a new application

2. **Get API Credentials**
   - App Key (Consumer Key)
   - App Secret (Consumer Secret)
   - Callback URL (for OAuth)

3. **Configure OAuth**
   - Set up the callback URL in your Schwab app settings
   - Ensure your callback URL matches the one in your `.env` file

## Usage

1. **Start the application**
   ```bash
   python ezoptionsschwab.py
   ```

2. **Access the dashboard**
   - Open your browser to `http://localhost:5001`
   - The application will start on port 5001 by default

3. **Using the Dashboard**
   - Enter a ticker symbol (e.g., SPY, SPX, AAPL)
   - Select expiration dates from the dropdown
   - Adjust strike range using the slider
   - Toggle different chart types and options
   - Enable/disable auto-update streaming

## Chart Types

### Price Chart
- Real-time candlestick or Heikin-Ashi charts
- Volume overlay
- Support for gamma and percentage GEX levels

### Exposure Charts
- **Gamma Exposure**: Shows market maker hedging requirements
- **Delta Exposure**: Directional exposure by strike
- **Vanna Exposure**: Volatility-price cross-sensitivity
- **Advanced Greeks**: Charm, Speed, and Vomma exposures

### Historical Bubble Levels
- Historical exposure tracking over the last hour
- Bubble charts showing exposure intensity over time
- Available for Gamma, Delta, and Vanna

### Volume Analysis
- Options volume by strike
- Call/Put volume ratios
- Premium analysis by strike

### Options Chain
- Sortable options chain table
- Real-time bid/ask/last prices
- Volume and open interest data
- Implied volatility display

## Configuration Options

### Strike Range
- Adjustable from 1% to 20% of current price
- Filters options within the specified range

### Chart Toggles
- Show/hide calls, puts, or net exposure
- Color intensity based on exposure values
- Multiple expiration date support
 - Use Volume for Exposures: When enabled, all exposure formulas (GEX/DEX/VEX/etc.) are weighted by traded volume instead of open interest

### Color Customization
- Customizable call and put colors
- Intensity-based color scaling

### Auto-Update
- Real-time streaming data updates
- Pause/resume functionality
- 1-second update intervals

## Database

The application uses SQLite to store historical bubble levels data:
- Automatic database initialization
- Stores minute-by-minute exposure data
- Automatic cleanup of old data

#

## License

This project is for educational and personal use only. Please comply with Schwab's API terms of service and any applicable regulations regarding financial data usage.

## Disclaimer

This software is for informational purposes only. It does not constitute financial advice. Trading options involves significant risk and may not be suitable for all investors. Always consult with a qualified financial advisor before making investment decisions.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review Schwab API documentation
- Ensure all dependencies are properly installed

## Contact
Discord - eazy101