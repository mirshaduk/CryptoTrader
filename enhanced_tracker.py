# Import required libraries
from binance.client import Client
import time
import colorama
from colorama import Fore, Style

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# Initialize the client (no API key needed for public data)
client = Client()

# Define the symbol we want to track
symbol = 'BTCUSDT'

def get_price_with_change():
    """Get the current price and calculate change from previous check"""
    try:
        # Get the latest ticker price
        ticker = client.get_symbol_ticker(symbol=symbol)
        
        # Get 24h ticker data for additional information
        ticker_24h = client.get_ticker(symbol=symbol)
        
        # Extract the current price
        current_price = float(ticker['price'])
        
        # Extract 24h change percentage
        change_24h_percent = float(ticker_24h['priceChangePercent'])
        
        # Extract 24h high and low
        high_24h = float(ticker_24h['highPrice'])
        low_24h = float(ticker_24h['lowPrice'])
        
        return {
            'price': current_price,
            'change_24h_percent': change_24h_percent,
            'high_24h': high_24h,
            'low_24h': low_24h
        }
    except Exception as e:
        print(f"Error getting price: {e}")
        return None

def format_change(change_percent):
    """Format the change percentage with color"""
    if change_percent > 0:
        return f"{Fore.GREEN}+{change_percent:.2f}%{Style.RESET_ALL}"
    elif change_percent < 0:
        return f"{Fore.RED}{change_percent:.2f}%{Style.RESET_ALL}"
    else:
        return f"{change_percent:.2f}%"

def main():
    print(f"\n{Fore.CYAN}=== Enhanced Real-Time Price Tracker for {symbol} ==={Style.RESET_ALL}")
    print(f"Press {Fore.YELLOW}Ctrl + C{Style.RESET_ALL} to stop the script.\n")
    
    # Variables to track previous price for change calculation
    previous_price = None
    check_count = 0
    start_price = None
    
    try:
        while True:
            # Get current price data
            price_data = get_price_with_change()
            
            if price_data:
                current_price = price_data['price']
                change_24h_percent = price_data['change_24h_percent']
                high_24h = price_data['high_24h']
                low_24h = price_data['low_24h']
                
                # Set start price on first run
                if start_price is None:
                    start_price = current_price
                
                # Calculate change since previous check
                if previous_price:
                    change_since_prev = ((current_price - previous_price) / previous_price) * 100
                    change_str = format_change(change_since_prev)
                else:
                    change_str = "0.00%"
                
                # Calculate change since start
                if start_price:
                    change_since_start = ((current_price - start_price) / start_price) * 100
                    start_change_str = format_change(change_since_start)
                else:
                    start_change_str = "0.00%"
                
                # Get current time
                current_time = time.strftime('%H:%M:%S', time.localtime())
                
                # Clear line if not first check
                if check_count > 0:
                    print("\033[A\033[K" * 5, end="")
                
                # Print the formatted price and time
                print(f"{Fore.YELLOW}[{current_time}]{Style.RESET_ALL} Current Price: {Fore.CYAN}${current_price:,.2f}{Style.RESET_ALL} USDT")
                print(f"Since last check: {change_str}")
                print(f"Since tracking started: {start_change_str}")
                print(f"24h Change: {format_change(change_24h_percent)}")
                print(f"24h Range: ${low_24h:,.2f} - ${high_24h:,.2f}")
                
                # Update previous price for next iteration
                previous_price = current_price
                check_count += 1
            
            # Wait for 5 seconds before the next check
            time.sleep(5)
    
    except KeyboardInterrupt:
        # This block catches the Ctrl + C command to stop the loop gracefully
        print(f"\n\n{Fore.YELLOW}Tracker stopped.{Style.RESET_ALL}")
    
    except Exception as e:
        # If any other error occurs, print it and wait before retrying
        print(f"\n{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")
        time.sleep(10)

if __name__ == "__main__":
    main()