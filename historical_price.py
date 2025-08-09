# Import required libraries
from binance.client import Client
from datetime import datetime, timedelta

# Initialize the client
client = Client()

def get_historical_prices(symbol, days_back=7):
    """Get historical daily prices for a cryptocurrency"""
    # Calculate the start time (days_back days ago)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    
    # Convert to millisecond timestamps for Binance API
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    try:
        # Get the historical klines (candlestick data)
        # Interval '1d' means daily data
        klines = client.get_historical_klines(
            symbol=symbol,
            interval='1d',
            start_str=start_timestamp,
            end_str=end_timestamp
        )
        
        # Process the data
        historical_data = []
        for kline in klines:
            # Kline format: [Open time, Open, High, Low, Close, Volume, Close time, ...]
            date = datetime.fromtimestamp(kline[0]/1000).strftime('%Y-%m-%d')
            close_price = float(kline[4])  # Close price
            historical_data.append((date, close_price))
            
        return historical_data
    
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return []

def main():
    symbol = 'BTCUSDT'
    days = 7
    
    print(f"\nHistorical Prices for {symbol} (Last {days} days):\n" + "-" * 40)
    
    historical_data = get_historical_prices(symbol, days)
    
    if historical_data:
        # Sort by date (oldest first)
        historical_data.sort(key=lambda x: x[0])
        
        # Print the data
        for date, price in historical_data:
            print(f"{date}: ${price:,.2f} USDT")
        
        # Calculate and print price change
        if len(historical_data) >= 2:
            first_price = historical_data[0][1]
            last_price = historical_data[-1][1]
            price_change = last_price - first_price
            price_change_percent = (price_change / first_price) * 100
            
            print("\nPrice Summary:")
            print(f"Starting Price ({historical_data[0][0]}): ${first_price:,.2f} USDT")
            print(f"Current Price ({historical_data[-1][0]}): ${last_price:,.2f} USDT")
            print(f"Change: ${price_change:,.2f} USDT ({price_change_percent:.2f}%)")
    else:
        print("No historical data available.")

if __name__ == "__main__":
    main()