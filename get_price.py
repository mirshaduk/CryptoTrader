# Import necessary libraries 
from binance.client import Client 
import pandas as pd 
import time 

# --- SETUP --- 
client = Client() 
symbol = 'BTCUSDT' 
sma_period = 20 

print("--- Setting up Strategy & Signal Engine ---") 

# 1. FETCH HISTORICAL DATA 
klines = client.get_historical_klines( 
    symbol=symbol, 
    interval=Client.KLINE_INTERVAL_1MINUTE, 
    limit=50 
) 
closing_prices = [float(kline[4]) for kline in klines] 
price_series = pd.Series(closing_prices) 

# --- INITIALIZE STRATEGY STATE --- 
# We need to know our starting position (is price above or below the SMA?) 
initial_sma = price_series.rolling(window=sma_period).mean().iloc[-1] 
last_price = price_series.iloc[-1] 
position = 'above' if last_price > initial_sma else 'below' 
print(f"Initial Position: Price is {position} the SMA.") 
print(f"--- Starting real-time signal generation for {symbol} ---") 

# --- MAIN LOOP --- 
while True: 
    try: 
        # Get the latest price 
        ticker = client.get_symbol_ticker(symbol=symbol) 
        current_price = float(ticker['price']) 
        
        # Update our price series 
        price_series = pd.concat([price_series, pd.Series([current_price])], ignore_index=True) 
        if len(price_series) > 50: 
            price_series = price_series.iloc[1:] 

        # Calculate the latest SMA 
        sma = price_series.rolling(window=sma_period).mean().iloc[-1] 
        
        # --- STRATEGY LOGIC --- 
        signal = 'HOLD' # Default signal is to do nothing 
        
        # Check for a BUY signal 
        if current_price > sma and position == 'below': 
            signal = '*** BUY SIGNAL ***' 
            position = 'above' # Update our position 
        
        # Check for a SELL signal 
        elif current_price < sma and position == 'above': 
            signal = '--- SELL SIGNAL ---' 
            position = 'below' # Update our position 

        # --- DISPLAY RESULTS --- 
        print(f"Price: {current_price:,.2f} | SMA: {sma:,.2f} | Signal: {signal}") 
        
        time.sleep(5) 

    except KeyboardInterrupt: 
        print("\nSignal generator stopped.") 
        break 
        
    except Exception as e: 
        print(f"An error occurred: {e}") 
        time.sleep(10)