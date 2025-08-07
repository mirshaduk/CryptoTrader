# Import necessary libraries
from binance.client import Client
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

# --- SETUP ---
client = Client()
symbol = 'BTCUSDT'
sma_period = 20

# Setup for data storage
max_data_points = 100  # Store more data points for better visualization
price_history = []
sma_history = []
signal_history = []
time_history = []

# Setup for visualization
plt.style.use('dark_background')  # Dark theme for better visibility
fig, ax = plt.subplots(figsize=(12, 6))
fig.canvas.manager.set_window_title(f'{symbol} Price and SMA Crossover Strategy')

# Initialize lines for the plot
price_line, = ax.plot([], [], 'g-', label='Price', linewidth=2)
sma_line, = ax.plot([], [], 'r-', label='SMA', linewidth=2)
buy_scatter = ax.scatter([], [], color='lime', s=100, marker='^', label='BUY')
sell_scatter = ax.scatter([], [], color='red', s=100, marker='v', label='SELL')

# Configure the plot
ax.set_title(f'{symbol} Real-time Price with SMA Crossover Signals')
ax.set_xlabel('Time')
ax.set_ylabel('Price (USDT)')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')

# Function to fetch historical data
def fetch_historical_data():
    print(f"{Fore.CYAN}--- Fetching historical data for {symbol} ---{Style.RESET_ALL}")
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1MINUTE,
        limit=max_data_points
    )
    
    # Extract closing prices and timestamps
    closing_prices = [float(kline[4]) for kline in klines]
    timestamps = [pd.to_datetime(kline[0], unit='ms') for kline in klines]
    
    # Calculate SMA
    price_series = pd.Series(closing_prices)
    sma_series = price_series.rolling(window=sma_period).mean()
    
    # Initialize our position (above or below SMA)
    last_price = price_series.iloc[-1]
    last_sma = sma_series.iloc[-1]
    position = 'above' if last_price > last_sma else 'below'
    
    # Initialize our data storage
    global price_history, sma_history, time_history, signal_history
    price_history = closing_prices
    sma_history = sma_series.tolist()
    time_history = timestamps
    
    # Initialize signals (all None except where we would have had signals)
    signal_history = [None] * len(closing_prices)
    
    # Backfill signals based on crossovers
    for i in range(sma_period + 1, len(closing_prices)):
        if closing_prices[i] > sma_series[i] and closing_prices[i-1] <= sma_series[i-1]:
            signal_history[i] = 'BUY'
        elif closing_prices[i] < sma_series[i] and closing_prices[i-1] >= sma_series[i-1]:
            signal_history[i] = 'SELL'
    
    print(f"{Fore.GREEN}Initial position: Price is {position} the SMA{Style.RESET_ALL}")
    return position

# Function to update the plot
def update_plot(frame):
    # Update the lines
    price_line.set_data(range(len(time_history)), price_history)
    sma_line.set_data(range(len(time_history)), sma_history)
    
    # Update the buy/sell signals
    buy_x = [i for i, signal in enumerate(signal_history) if signal == 'BUY']
    buy_y = [price_history[i] for i in buy_x]
    sell_x = [i for i, signal in enumerate(signal_history) if signal == 'SELL']
    sell_y = [price_history[i] for i in sell_x]
    
    buy_scatter.set_offsets(list(zip(buy_x, buy_y)))
    sell_scatter.set_offsets(list(zip(sell_x, sell_y)))
    
    # Adjust the plot limits
    ax.set_xlim(0, len(time_history))
    if price_history:
        buffer = (max(price_history) - min(price_history)) * 0.1
        ax.set_ylim(min(price_history) - buffer, max(price_history) + buffer)
    
    # Update x-axis labels to show time
    if time_history:
        # Only show a few time labels to avoid overcrowding
        num_labels = 5
        step = max(1, len(time_history) // num_labels)
        positions = range(0, len(time_history), step)
        labels = [time_history[i].strftime('%H:%M:%S') for i in positions]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45)
    
    return price_line, sma_line, buy_scatter, sell_scatter

# Main function to run the strategy and visualization
def run_strategy():
    # Fetch initial data and determine our starting position
    position = fetch_historical_data()
    
    print(f"{Fore.YELLOW}--- Starting real-time signal generation and visualization for {symbol} ---{Style.RESET_ALL}")
    print(f"Press Ctrl+C to stop the script.")
    
    # Start the animation
    ani = FuncAnimation(fig, update_plot, interval=5000, blit=True)
    plt.show(block=False)  # Show the plot but don't block execution
    
    try:
        while True:
            # Get the latest price
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            current_time = pd.Timestamp.now()
            
            # Update our data
            price_history.append(current_price)
            time_history.append(current_time)
            
            # Calculate the latest SMA
            price_series = pd.Series(price_history)
            sma = price_series.rolling(window=sma_period).mean().iloc[-1]
            sma_history.append(sma)
            
            # Keep our data within the max size
            if len(price_history) > max_data_points:
                price_history.pop(0)
                sma_history.pop(0)
                time_history.pop(0)
                signal_history.pop(0)
            
            # --- STRATEGY LOGIC ---
            signal = None  # Default is no signal
            
            # Check for a BUY signal
            if current_price > sma and position == 'below':
                signal = 'BUY'
                position = 'above'  # Update our position
                print(f"{Fore.GREEN}[{current_time.strftime('%H:%M:%S')}] *** BUY SIGNAL *** at {current_price:,.2f}{Style.RESET_ALL}")
            
            # Check for a SELL signal
            elif current_price < sma and position == 'above':
                signal = 'SELL'
                position = 'below'  # Update our position
                print(f"{Fore.RED}[{current_time.strftime('%H:%M:%S')}] --- SELL SIGNAL --- at {current_price:,.2f}{Style.RESET_ALL}")
            
            # Add the signal to our history
            signal_history.append(signal)
            
            # Display current status
            if signal is None:
                print(f"[{current_time.strftime('%H:%M:%S')}] Price: {current_price:,.2f} | SMA: {sma:,.2f} | Signal: HOLD")
            
            # Wait before the next update
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nSignal generator and visualization stopped.")
        plt.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        plt.close()

# Run the strategy if this script is executed directly
if __name__ == "__main__":
    run_strategy()