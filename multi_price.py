# Import the Client class from the python-binance library 
from binance.client import Client 

# We don't need an API key for public data, so we can leave these empty. 
client = Client() 

# Define the symbols we want to get prices for
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']

def get_crypto_price(symbol):
    """Get the current price for a given cryptocurrency symbol"""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        return price
    except Exception as e:
        return f"Error: {e}"

def main():
    print("\nCurrent Cryptocurrency Prices:\n" + "-" * 30)
    
    for symbol in symbols:
        price = get_crypto_price(symbol)
        if isinstance(price, float):
            base_asset = symbol[:-4]  # Remove the 'USDT' part
            print(f"{base_asset}: ${price:,.2f} USDT")
        else:
            print(f"{symbol}: {price}")

if __name__ == "__main__":
    main()