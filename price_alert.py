# Import required libraries
from binance.client import Client
import time
import os

# Initialize the client
client = Client()

def get_current_price(symbol):
    """Get the current price for a given cryptocurrency symbol"""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error getting price: {e}")
        return None

def set_price_alert(symbol, target_price, check_interval=60, alert_type='above'):
    """Monitor price and alert when it crosses the target threshold"""
    print(f"\nPrice Alert Set for {symbol}")
    print(f"Target: ${target_price:,.2f} USDT ({alert_type.capitalize()})")
    print(f"Checking every {check_interval} seconds...")
    print("Press Ctrl+C to stop monitoring\n")
    
    alert_triggered = False
    
    try:
        while not alert_triggered:
            current_price = get_current_price(symbol)
            
            if current_price is not None:
                print(f"Current price of {symbol}: ${current_price:,.2f} USDT", end='\r')
                
                # Check if price crossed the threshold
                if (alert_type.lower() == 'above' and current_price >= target_price) or \
                   (alert_type.lower() == 'below' and current_price <= target_price):
                    # Alert triggered
                    alert_triggered = True
                    
                    # Clear the current line
                    print(" " * 80, end='\r')
                    
                    # Print alert message
                    print(f"\n\n🚨 ALERT! 🚨")
                    print(f"{symbol} price is now ${current_price:,.2f} USDT")
                    print(f"Target of ${target_price:,.2f} USDT ({alert_type}) has been reached!")
                    
                    # Make a sound alert (beep)
                    for _ in range(3):
                        print('\a', end='', flush=True)  # Bell character
                        time.sleep(0.5)
            
            # Wait for the specified interval before checking again
            if not alert_triggered:
                time.sleep(check_interval)
                
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")

def main():
    # Default values
    symbol = 'BTCUSDT'
    target_price = 115000.0  # Example target price
    check_interval = 10  # Check every 10 seconds (for demo purposes)
    alert_type = 'above'  # Alert when price goes above target
    
    # Get user input (with defaults)
    print("\nCrypto Price Alert Setup\n" + "-" * 25)
    
    # Use input with defaults
    user_symbol = input(f"Enter cryptocurrency symbol (default: {symbol}): ").strip().upper()
    symbol = user_symbol if user_symbol else symbol
    
    # Get current price to help user set a reasonable target
    current_price = get_current_price(symbol)
    if current_price is not None:
        print(f"Current price of {symbol}: ${current_price:,.2f} USDT")
    
    # Get target price
    try:
        user_target = input(f"Enter target price in USDT (default: ${target_price:,.2f}): ").strip()
        target_price = float(user_target) if user_target else target_price
    except ValueError:
        print("Invalid price. Using default.")
    
    # Get alert type
    user_alert_type = input(f"Alert when price goes 'above' or 'below' target (default: {alert_type}): ").strip().lower()
    if user_alert_type in ['above', 'below']:
        alert_type = user_alert_type
    
    # Get check interval
    try:
        user_interval = input(f"Check interval in seconds (default: {check_interval}): ").strip()
        check_interval = int(user_interval) if user_interval else check_interval
    except ValueError:
        print("Invalid interval. Using default.")
    
    # Start the price alert
    set_price_alert(symbol, target_price, check_interval, alert_type)

if __name__ == "__main__":
    main()