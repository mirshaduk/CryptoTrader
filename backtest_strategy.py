# Import necessary libraries
from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

# --- SETUP ---
client = Client()
symbol = 'BTCUSDT'
sma_period = 20

# Function to fetch historical data for backtesting
def fetch_historical_data(symbol, interval, start_date, end_date=None):
    print(f"{Fore.CYAN}Fetching historical data for {symbol} from {start_date} to {end_date or 'now'}{Style.RESET_ALL}")
    
    # Convert dates to milliseconds timestamp
    start_ms = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    
    if end_date:
        end_ms = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    else:
        end_ms = int(datetime.now().timestamp() * 1000)
    
    # Fetch the klines (candlestick data)
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_ms,
        end_str=end_ms
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    print(f"{Fore.GREEN}Fetched {len(df)} data points{Style.RESET_ALL}")
    return df

# Function to apply the SMA crossover strategy
def apply_sma_strategy(df, sma_period):
    # Calculate SMA
    df['sma'] = df['close'].rolling(window=sma_period).mean()
    
    # Initialize signal column
    df['signal'] = 'HOLD'
    
    # Initialize position column (1 for long, 0 for no position, -1 for short)
    df['position'] = 0
    
    # Determine initial position
    for i in range(sma_period, len(df)):
        if i == sma_period:  # First valid SMA value
            df.loc[i, 'position'] = 1 if df.loc[i, 'close'] > df.loc[i, 'sma'] else 0
        else:
            # Check for crossover
            if df.loc[i, 'close'] > df.loc[i, 'sma'] and df.loc[i-1, 'close'] <= df.loc[i-1, 'sma']:
                df.loc[i, 'signal'] = 'BUY'
                df.loc[i, 'position'] = 1
            elif df.loc[i, 'close'] < df.loc[i, 'sma'] and df.loc[i-1, 'close'] >= df.loc[i-1, 'sma']:
                df.loc[i, 'signal'] = 'SELL'
                df.loc[i, 'position'] = 0
            else:
                df.loc[i, 'position'] = df.loc[i-1, 'position']
    
    return df

# Function to calculate strategy performance
def calculate_performance(df):
    # Create a copy to avoid modifying the original
    perf_df = df.copy()
    
    # Calculate daily returns of the asset
    perf_df['asset_returns'] = perf_df['close'].pct_change()
    
    # Calculate strategy returns (only earn returns when in position)
    perf_df['strategy_returns'] = perf_df['asset_returns'] * perf_df['position'].shift(1)
    
    # Calculate cumulative returns
    perf_df['cum_asset_returns'] = (1 + perf_df['asset_returns']).cumprod() - 1
    perf_df['cum_strategy_returns'] = (1 + perf_df['strategy_returns']).cumprod() - 1
    
    # Calculate drawdowns
    perf_df['asset_peak'] = perf_df['cum_asset_returns'].cummax()
    perf_df['strategy_peak'] = perf_df['cum_strategy_returns'].cummax()
    perf_df['asset_drawdown'] = perf_df['asset_peak'] - perf_df['cum_asset_returns']
    perf_df['strategy_drawdown'] = perf_df['strategy_peak'] - perf_df['cum_strategy_returns']
    
    # Count trades
    trades = perf_df[perf_df['signal'].isin(['BUY', 'SELL'])]
    num_trades = len(trades)
    
    # Calculate win rate
    if num_trades > 0:
        # A trade is profitable if the price at the next SELL is higher than the price at BUY
        buy_signals = perf_df[perf_df['signal'] == 'BUY'].copy()
        sell_signals = perf_df[perf_df['signal'] == 'SELL'].copy()
        
        if len(buy_signals) > 0 and len(sell_signals) > 0:
            # Match each BUY with the next SELL
            buy_prices = []
            sell_prices = []
            
            buy_idx = buy_signals.index.tolist()
            sell_idx = sell_signals.index.tolist()
            
            for b_idx in buy_idx:
                # Find the next SELL after this BUY
                next_sells = [s for s in sell_idx if s > b_idx]
                if next_sells:
                    buy_prices.append(perf_df.loc[b_idx, 'close'])
                    sell_prices.append(perf_df.loc[next_sells[0], 'close'])
            
            if buy_prices and sell_prices:
                trade_returns = [(sell - buy) / buy for buy, sell in zip(buy_prices, sell_prices)]
                winning_trades = sum(1 for ret in trade_returns if ret > 0)
                win_rate = winning_trades / len(trade_returns) if trade_returns else 0
            else:
                win_rate = 0
        else:
            win_rate = 0
    else:
        win_rate = 0
    
    # Calculate performance metrics
    total_return = perf_df['cum_strategy_returns'].iloc[-1]
    max_drawdown = perf_df['strategy_drawdown'].max()
    sharpe_ratio = (perf_df['strategy_returns'].mean() / perf_df['strategy_returns'].std()) * np.sqrt(252) if perf_df['strategy_returns'].std() != 0 else 0
    
    # Calculate annualized return
    days = (perf_df['timestamp'].iloc[-1] - perf_df['timestamp'].iloc[0]).days
    annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    
    # Return performance metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': num_trades,
        'win_rate': win_rate
    }
    
    return perf_df, metrics

# Function to visualize the backtest results
def visualize_backtest(df, metrics):
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'SMA Crossover Strategy Backtest for {symbol}', fontsize=16)
    
    # Plot price and SMA
    ax1.plot(df['timestamp'], df['close'], label='Price', color='blue')
    ax1.plot(df['timestamp'], df['sma'], label=f'{sma_period}-period SMA', color='red')
    
    # Plot buy and sell signals
    buy_signals = df[df['signal'] == 'BUY']
    sell_signals = df[df['signal'] == 'SELL']
    
    ax1.scatter(buy_signals['timestamp'], buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    ax1.scatter(sell_signals['timestamp'], sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    ax1.set_title('Price and SMA with Buy/Sell Signals')
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot cumulative returns
    ax2.plot(df['timestamp'], df['cum_asset_returns'], label='Buy and Hold', color='blue')
    ax2.plot(df['timestamp'], df['cum_strategy_returns'], label='SMA Strategy', color='green')
    
    ax2.set_title('Cumulative Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add performance metrics as text
    metrics_text = (
        f"Total Return: {metrics['total_return']:.2%}\n"
        f"Annualized Return: {metrics['annualized_return']:.2%}\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Number of Trades: {metrics['num_trades']}\n"
        f"Win Rate: {metrics['win_rate']:.2%}"
    )
    
    plt.figtext(0.15, 0.01, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

# Main function to run the backtest
def run_backtest(symbol, interval, start_date, end_date=None, sma_period=20):
    # Fetch historical data
    df = fetch_historical_data(symbol, interval, start_date, end_date)
    
    # Apply the strategy
    df = apply_sma_strategy(df, sma_period)
    
    # Calculate performance
    perf_df, metrics = calculate_performance(df)
    
    # Print performance metrics
    print(f"\n{Fore.YELLOW}===== BACKTEST RESULTS ====={Style.RESET_ALL}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date or 'now'}")
    print(f"Strategy: {sma_period}-period SMA Crossover")
    print(f"\n{Fore.GREEN}Performance Metrics:{Style.RESET_ALL}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    # Compare to buy and hold
    buy_hold_return = perf_df['cum_asset_returns'].iloc[-1]
    print(f"\n{Fore.BLUE}Buy and Hold Return: {buy_hold_return:.2%}{Style.RESET_ALL}")
    
    if metrics['total_return'] > buy_hold_return:
        print(f"{Fore.GREEN}Strategy OUTPERFORMED Buy and Hold by {metrics['total_return'] - buy_hold_return:.2%}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Strategy UNDERPERFORMED Buy and Hold by {buy_hold_return - metrics['total_return']:.2%}{Style.RESET_ALL}")
    
    # Visualize the results
    visualize_backtest(perf_df, metrics)
    
    return perf_df, metrics

# Run the backtest if this script is executed directly
if __name__ == "__main__":
    # Default parameters
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1DAY
    
    # Calculate date 90 days ago for default start date
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    end_date = None
    
    print(f"{Fore.CYAN}=== SMA Crossover Strategy Backtester ==={Style.RESET_ALL}")
    print(f"Running with default settings: {symbol}, {sma_period}-period SMA, from {start_date} to now")
    
    # Run the backtest with default parameters
    run_backtest(symbol, interval, start_date, end_date, sma_period)
    
    print(f"\n{Fore.CYAN}To run with custom parameters, modify the script or use the interactive mode.{Style.RESET_ALL}")
    print("Interactive mode can be enabled by changing the code in backtest_strategy.py.")