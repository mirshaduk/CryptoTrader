import pandas as pd
import pandas_ta as ta
from binance.client import Client
import time
import os
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# --- CONFIGURATION ---
SYMBOL = "BTCUSDT"  # Default symbol to monitor
SMA_PERIOD = 50      # Default SMA period
RSI_PERIOD = 14      # Default RSI period
CHECK_INTERVAL = 60  # Default check interval in seconds

# --- HELPER FUNCTIONS ---
def clear_screen():
    """Clear the terminal screen based on OS."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_live_data(symbol, interval, sma_period, rsi_period):
    """Fetches live market data and calculates indicators."""
    client = Client()
    # Fetch enough data to calculate our indicators
    limit = max(sma_period, rsi_period) * 2
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_asset_volume', 'number_of_trades', 
                                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # Convert data types
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col])
        
    # Calculate indicators
    data.ta.sma(length=sma_period, append=True)
    data.ta.rsi(length=rsi_period, append=True)
    data.rename(columns={f'SMA_{sma_period}': 'sma', f'RSI_{rsi_period}': 'rsi'}, inplace=True)
    
    return data

def print_header():
    """Print the application header."""
    print(Fore.CYAN + "=" * 80)
    print(Fore.CYAN + " " * 25 + "CRYPTO TRADE ALERT SYSTEM" + " " * 25)
    print(Fore.CYAN + "=" * 80)
    print()

def print_settings(symbol, sma_period, rsi_period, check_interval):
    """Print the current monitoring settings."""
    print(Fore.YELLOW + "MONITORING SETTINGS:")
    print(Fore.YELLOW + f"Symbol: {symbol}")
    print(Fore.YELLOW + f"SMA Period: {sma_period}")
    print(Fore.YELLOW + f"RSI Period: {rsi_period}")
    print(Fore.YELLOW + f"Check Interval: {check_interval} seconds")
    print()

def print_alert(signal, position):
    """Print a trading signal alert."""
    if signal == "BUY":
        print(Fore.BLACK + Back.GREEN + " " * 20 + "BUY SIGNAL DETECTED!" + " " * 20)
        # Print ASCII art for BUY signal
        print(Fore.GREEN + """
        ▲▲▲▲▲▲▲▲
        ▲▲▲▲▲▲▲▲
        ▲▲▲▲▲▲▲▲
        ▲▲▲▲▲▲▲▲
        """)
    elif signal == "SELL":
        print(Fore.BLACK + Back.RED + " " * 20 + "SELL SIGNAL DETECTED!" + " " * 20)
        # Print ASCII art for SELL signal
        print(Fore.RED + """
        ▼▼▼▼▼▼▼▼
        ▼▼▼▼▼▼▼▼
        ▼▼▼▼▼▼▼▼
        ▼▼▼▼▼▼▼▼
        """)
    else:
        print(Fore.BLUE + f"HOLDING... Current Position: {position}")

def print_metrics(latest_row, sma_period, rsi_period):
    """Print the current market metrics."""
    print(Fore.WHITE + "CURRENT METRICS:")
    print(Fore.WHITE + f"Current Price: ${latest_row['close']:,.2f}")
    print(Fore.WHITE + f"SMA({sma_period}): ${latest_row['sma']:,.2f}")
    print(Fore.WHITE + f"RSI({rsi_period}): {latest_row['rsi']:.2f}")
    print()

def print_mini_chart(data, rows=10):
    """Print a simple ASCII chart of recent price movements."""
    print(Fore.WHITE + "RECENT PRICE MOVEMENTS:")
    
    # Get the last few data points
    recent_data = data.tail(rows)
    
    # Normalize the data for display in terminal
    max_price = recent_data['close'].max()
    min_price = recent_data['close'].min()
    range_price = max_price - min_price
    
    # Avoid division by zero
    if range_price == 0:
        range_price = 1
    
    # Chart width
    width = 50
    
    # Print the chart
    for _, row in recent_data.iterrows():
        # Calculate position
        position = int(((row['close'] - min_price) / range_price) * width)
        
        # Determine color based on price movement
        if row['close'] > row['sma']:
            color = Fore.GREEN
        else:
            color = Fore.RED
            
        # Print the bar
        print(f"{row['timestamp'].strftime('%H:%M:%S')} {color}{'█' * position} {row['close']:.2f}")
    
    print()

def main():
    """Main function to run the terminal alert system."""
    # Initialize variables
    position = "OUT"  # Can be "IN" or "OUT"
    
    # Get user input for parameters
    clear_screen()
    print_header()
    
    print("Enter monitoring parameters (press Enter for defaults):")
    symbol_input = input(f"Crypto Symbol [{SYMBOL}]: ").strip() or SYMBOL
    sma_input = int(input(f"SMA Period [{SMA_PERIOD}]: ").strip() or SMA_PERIOD)
    rsi_input = int(input(f"RSI Period [{RSI_PERIOD}]: ").strip() or RSI_PERIOD)
    check_interval_input = int(input(f"Check Interval in seconds [{CHECK_INTERVAL}]: ").strip() or CHECK_INTERVAL)
    
    print("\nStarting monitoring...\n")
    time.sleep(2)  # Brief pause before starting
    
    # Main monitoring loop
    while True:
        try:
            # Clear screen for fresh display
            clear_screen()
            print_header()
            print_settings(symbol_input, sma_input, rsi_input, check_interval_input)
            
            # Fetch the latest data and indicators
            df = get_live_data(symbol_input, Client.KLINE_INTERVAL_1MINUTE, sma_input, rsi_input)
            
            # Get the most recent data point
            latest_row = df.iloc[-1]
            
            # --- STRATEGY LOGIC ---
            signal = "HOLD"
            
            # Check for BUY signal
            if latest_row['close'] > latest_row['sma'] and latest_row['rsi'] > 50 and position != "IN":
                signal = "BUY"
                position = "IN"
                
            # Check for SELL signal
            elif latest_row['close'] < latest_row['sma'] and latest_row['rsi'] < 50 and position == "IN":
                signal = "SELL"
                position = "OUT"
            
            # Print the alert and metrics
            print_alert(signal, position)
            print_metrics(latest_row, sma_input, rsi_input)
            print_mini_chart(df)
            
            # Print last updated time
            print(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
            print(Fore.CYAN + "Press Ctrl+C to exit")

        except Exception as e:
            clear_screen()
            print_header()
            print(Fore.RED + f"An error occurred: {e}")
            print(Fore.RED + "Will retry in 60 seconds...")
        
        # Wait for the next check
        time.sleep(check_interval_input)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\nMonitoring stopped. Thank you for using the Crypto Trade Alert System!\n")