import pandas as pd 
import pandas_ta as ta 
from binance.client import Client 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report 
 
# --- 1. DATA FETCHING AND FEATURE ENGINEERING --- 
print("Fetching data and engineering features...") 
client = Client() 
symbol = 'BTCUSDT' 
interval = Client.KLINE_INTERVAL_1DAY 
start_date = "3 years ago UTC" # We need more data for ML 
 
klines = client.get_historical_klines(symbol, interval, start_date) 
data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']) 
 
# Data Preparation 
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms') 
data.set_index('timestamp', inplace=True) 
for col in ['open', 'high', 'low', 'close', 'volume']: 
    data[col] = pd.to_numeric(data[col]) 
 
# Feature Creation 
print("Creating features...") 
data.ta.rsi(length=14, append=True) 
data.ta.mom(length=10, append=True) # Momentum 
data['price_change_1d'] = data['close'].pct_change(1) # 1-day percentage change 
data['price_change_5d'] = data['close'].pct_change(5) # 5-day percentage change 
 
# Target Variable Creation (This is what we want to predict) 
# If tomorrow's price is higher, target is 1 (UP). Otherwise, it's 0 (DOWN). 
data['target'] = (data['close'].shift(-1) > data['close']).astype(int) 
 
data.dropna(inplace=True) 
 
# Define our features (X) and target (y) 
features = ['RSI_14', 'MOM_10', 'price_change_1d', 'price_change_5d', 'volume'] 
X = data[features] 
y = data['target'] 
 
 
# --- 2. MODEL TRAINING AND PREDICTION --- 
print("Splitting data and training model...") 
# We split the data to train on the past and test on the most recent data 
# IMPORTANT: We must not shuffle time-series data 
train_size = int(len(X) * 0.8) 
X_train, X_test = X[:train_size], X[train_size:] 
y_train, y_test = y[:train_size], y[train_size:] 
 
# Initialize and train the RandomForestClassifier 
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train) 
 
# Make predictions on the test set (the data the model has never seen) 
predictions = model.predict(X_test) 
 
 
# --- 3. MODEL EVALUATION --- 
print("\n--- Model Evaluation ---") 
accuracy = accuracy_score(y_test, predictions) 
print(f"Model Accuracy on Test Data: {accuracy:.4f}") 
print("Classification Report:") 
print(classification_report(y_test, predictions, target_names=['DOWN', 'UP'])) 
print("------------------------\n") 
# An accuracy just over 0.5 is typical for financial markets and does NOT guarantee profitability. 
 
 
# --- 4. BACKTESTING THE ML STRATEGY --- 
print("Running backtest on ML predictions...") 
initial_capital = 10000.0 
capital = initial_capital 
position = None 
asset_quantity = 0 
portfolio_value = [] 
 
test_data = data[train_size:].copy() # The part of data we're testing on 
test_data['prediction'] = predictions 
 
for i, row in test_data.iterrows(): 
    # BUY if model predicts UP and we are not in a position 
    if row['prediction'] == 1 and position != 'IN': 
        position = 'IN' 
        asset_quantity = capital / row['close'] 
        capital = 0 
    # SELL if model predicts DOWN and we are in a position 
    elif row['prediction'] == 0 and position == 'IN': 
        position = 'OUT' 
        capital = asset_quantity * row['close'] 
        asset_quantity = 0 
 
    current_value = capital + (asset_quantity * row['close'] if position == 'IN' else 0) 
    portfolio_value.append(current_value) 
 
test_data['portfolio_value'] = portfolio_value 
 
# --- 5. RESULTS --- 
final_value = test_data['portfolio_value'].iloc[-1] 
total_return = (final_value - initial_capital) / initial_capital * 100 
buy_and_hold_return = (test_data['close'].iloc[-1] - test_data['close'].iloc[0]) / test_data['close'].iloc[0] * 100 
 
print("\n--- ML Strategy Backtest Results ---") 
print(f"Initial Capital: ${initial_capital:,.2f}") 
print(f"Final Portfolio Value: ${final_value:,.2f}") 
print(f"Total Return: {total_return:.2f}%") 
print(f"Buy and Hold Return: {buy_and_hold_return:.2f}%") 
print("------------------------------------\n")