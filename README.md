# CryptoTrader

A comprehensive Python application for cryptocurrency trading analysis, featuring real-time price tracking, technical analysis, trading signals, backtesting, and visualization.

## Setup

1. Install the required dependencies:

```
pip install -r requirements.txt
```

2. Run one of the available scripts:

```
# For real-time Bitcoin price tracking with SMA calculation and trading signals
python get_price.py

# For real-time trading signals with visualization
python signal_visualizer.py

# For backtesting the SMA crossover strategy
python backtest_strategy.py

# For enhanced real-time tracking with price changes
python enhanced_tracker.py

# For multiple cryptocurrency prices
python multi_price.py

# For Bitcoin price history (last 7 days)
python historical_price.py

# For price alerts (interactive)
python price_alert.py

# For GUI application
python crypto_gui.py
```

## Key Features

### Real-Time Data
- **Live Price Tracking**: Get real-time price updates for any cryptocurrency on Binance.
- **Enhanced Tracker**: An advanced view with price changes, 24-hour stats, and colored output.

### Technical Analysis & Signals
- **Simple Moving Average (SMA)**: Automatically calculates and displays the 20-period SMA alongside the real-time price.
- **Crossover Signals**: Generates BUY/SELL signals when the price crosses above or below the SMA.

### Visualization
- **Live Trading Chart**: Visualizes the price, SMA, and trading signals in a real-time chart.

### Strategy & Backtesting
- **Backtester**: Evaluate the performance of the SMA crossover strategy on historical data.
- **Performance Metrics**: Get key metrics like total return, win rate, and Sharpe ratio.

### Additional Tools
- **Multi-Price Viewer**: Track multiple cryptocurrencies at once.
- **Price Alerts**: Set custom price alerts.
- **GUI**: A simple graphical interface for price tracking.

## Extending the Application

You can modify the `symbol` variable in `get_price.py` to get prices for other cryptocurrencies. For example:

- 'ETHUSDT' for Ethereum
- 'BNBUSDT' for Binance Coin
- 'ADAUSDT' for Cardano

## Requirements

- Python 3.6 or higher
- python-binance library