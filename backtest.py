import pandas as pd
from binance.client import Client
import matplotlib.pyplot as plt

# --- 1. SETUP ---
client = Client()
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1DAY
start_date = "1 year ago UTC"
initial_capital = 10000.0
sma_period = 50
rsi_period = 14

# --- 2. DATA FETCHING AND PREPARATION ---
print("Fetching historical data...")
klines = client.get_historical_klines(symbol, interval, start_date)

data = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
])

# Convert data types
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)
for col in ['open', 'high', 'low', 'close', 'volume']:
    data[col] = pd.to_numeric(data[col])

# Calculate Technical Indicators
print("Calculating indicators...")
data['sma'] = data['close'].rolling(window=sma_period).mean()

# Calculate RSI manually
delta = data['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
rs = gain / loss
data['rsi'] = 100 - (100 / (1 + rs))


# Drop missing values
data.dropna(inplace=True)

# --- 3. STRATEGY SIMULATION ---
print("Running backtest simulation...")
capital = initial_capital
position = None
buy_signals = []
sell_signals = []
portfolio_value = []
asset_quantity = 0

for i, row in data.iterrows():
    # Check for BUY signal
    if row['close'] > row['sma'] and row['rsi'] > 50 and position != 'IN':
        position = 'IN'
        buy_signals.append(i)
        asset_quantity = capital / row['close']
        capital = 0

    # Check for SELL signal
    elif row['close'] < row['sma'] and row['rsi'] < 50 and position == 'IN':
        position = 'OUT'
        sell_signals.append(i)
        capital = asset_quantity * row['close']
        asset_quantity = 0
    
    current_value = capital + (asset_quantity * row['close'] if position == 'IN' else 0)
    portfolio_value.append(current_value)

data['portfolio_value'] = portfolio_value

# --- 4. RESULTS AND VISUALIZATION ---
print("Generating results...")
final_value = data['portfolio_value'].iloc[-1]
total_return = (final_value - initial_capital) / initial_capital * 100
buy_and_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100

print("\n--- Backtest Results (SMA + RSI Strategy) ---")
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Portfolio Value: ${final_value:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Buy and Hold Return: {buy_and_hold_return:.2f}%")
print(f"Number of Trades: {len(buy_signals) + len(sell_signals)}")
print("------------------------------------------\n")

# Plotting the results with two subplots
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('SMA + RSI Strategy Backtest on BTC/USDT', fontsize=16)

# Plot 1: Price, SMA, and Trades
ax1.plot(data.index, data['close'], label='BTC/USDT Price', color='skyblue')
ax1.plot(data.index, data['sma'], label=f'{sma_period}-Day SMA', color='orange', linestyle='--')
ax1.plot(buy_signals, data.loc[buy_signals]['close'], '^', markersize=12, color='lime', label='Buy Signal', markeredgecolor='black')
ax1.plot(sell_signals, data.loc[sell_signals]['close'], 'v', markersize=12, color='red', label='Sell Signal', markeredgecolor='black')
ax1.set_ylabel('Price (USDT)')
ax1.legend()
ax1.grid(True)

# Plot 2: RSI
ax2.plot(data.index, data['rsi'], label='RSI', color='purple')
ax2.axhline(70, linestyle='--', color='red', alpha=0.5)
ax2.axhline(50, linestyle='--', color='gray', alpha=0.7)
ax2.axhline(30, linestyle='--', color='green', alpha=0.5)
ax2.set_ylabel('RSI (0-100)')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()