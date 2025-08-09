import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import streamlit as st
import time
import itertools
from datetime import timedelta, datetime
import os
import logging

# Technical Indicator Functions
def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_rsi(data, period):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading.log')
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION & HELPER FUNCTIONS ---
ALERT_SOUND_URL = "https://www.soundjay.com/buttons/sounds/button-16.mp3"
PARAMETER_SEARCH_SPACE = {
    'sma_period': [20, 30, 50], 'rsi_period': [10, 14, 18],
    'tp_input': [3.0, 5.0], 'sl_input': [2.0, 3.0]
}

def validate_parameters(params):
    """Validate and bound-check trading parameters"""
    valid_params = params.copy()
    try:
        valid_params['sma_period'] = max(10, min(200, int(params['sma_period'])))
        valid_params['rsi_period'] = max(5, min(50, int(params['rsi_period'])))
        valid_params['tp_input'] = max(1.0, min(50.0, float(params['tp_input'])))
        valid_params['sl_input'] = max(1.0, min(50.0, float(params['sl_input'])))
    except (ValueError, TypeError) as e:
        st.error(f"Parameter validation error: {e}")
        return None
    return valid_params

def backup_trade_history(suffix=''):
    """Create backup of trade history"""
    try:
        backup_file = f"trade_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}.csv"
        if hasattr(st.session_state, 'trade_history'):
            st.session_state.trade_history.to_csv(backup_file, index=False)
            return True
    except Exception as e:
        st.error(f"Failed to create trade history backup: {e}")
    return False

def validate_position_state():
    """Validate and fix position state if invalid"""
    try:
        if st.session_state.position == 'IN':
            if not st.session_state.entry_price or not st.session_state.entry_time:
                st.warning("Invalid position state detected. Resetting position.")
                st.session_state.position = 'OUT'
                st.session_state.position_type = None
                st.session_state.entry_price = 0.0
                st.session_state.entry_time = None
                return False
        return True
    except Exception as e:
        st.error(f"Position validation error: {e}")
        return False

# Market analysis constants
VOLUME_THRESHOLD = 1.5  # Volume spike threshold
TREND_PERIODS = 20     # Number of periods for trend analysis
RSI_OVERBOUGHT = 70    # RSI overbought level
RSI_OVERSOLD = 30      # RSI oversold level

# Self-evolution parameters
LEARNING_RATE = 0.1    # How fast the system adapts
MIN_TRADES_FOR_EVOLUTION = 5  # Minimum trades before starting evolution
SUCCESS_THRESHOLD = 0.6  # Win rate threshold for parameter adjustment
OPTIMIZATION_INTERVAL = timedelta(hours=4)
OPTIMIZATION_DATA_RANGE = "3 days ago UTC"
TRADE_HISTORY_FILE = "trade_history.csv"

# --- NEW: Real-time Multi-Crypto Price Fetch ---
def place_order(symbol, side, quantity, max_retries=3):
    """Place an order with retry mechanism and validation"""
    for attempt in range(max_retries):
        try:
            client = Client()
            
            # Validate parameters
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Invalid symbol")
            if side not in ['BUY', 'SELL']:
                raise ValueError("Invalid side - must be 'BUY' or 'SELL'")
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                raise ValueError("Invalid quantity - must be positive number")
                
            # Get latest symbol info for precision requirements
            symbol_info = client.get_symbol_info(symbol)
            if not symbol_info:
                raise ValueError(f"Could not get symbol info for {symbol}")
                
            # Check for sufficient balance
            account = client.get_account()
            if side == 'BUY':
                quote_asset = symbol[len(symbol)-4:] if symbol.endswith('USDT') else 'USDT'
                balance = float([b for b in account['balances'] if b['asset'] == quote_asset][0]['free'])
                if balance < quantity:
                    raise ValueError(f"Insufficient {quote_asset} balance: {balance}")
            else:
                base_asset = symbol[:len(symbol)-4] if symbol.endswith('USDT') else symbol[:-4]
                balance = float([b for b in account['balances'] if b['asset'] == base_asset][0]['free'])
                if balance < quantity:
                    raise ValueError(f"Insufficient {base_asset} balance: {balance}")
            
            # Round quantity based on symbol's lot size filter
            lot_size = next(filter(lambda f: f['filterType'] == 'LOT_SIZE', symbol_info['filters']))
            step_size = float(lot_size['stepSize'])
            quantity = round(quantity - (quantity % step_size), len(str(step_size).split('.')[1]))
            
            # Place the order
            order = client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Log success
            logger.info(f"Successfully placed {side} order for {quantity} {symbol}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Order placement failed.")
                raise
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Order placement failed.")
                raise
    return None

def get_multi_prices(symbols):
    # Initialize client with testnet
    client = Client("", "", testnet=True)
    prices = []
    errors = []
    
    try:
        # Get all ticker prices at once for efficiency
        all_tickers = client.get_all_tickers()
        price_dict = {t['symbol']: float(t['price']) for t in all_tickers}
        
        for symbol in symbols:
            try:
                if symbol in price_dict:
                    prices.append({'Symbol': symbol, 'Price (USDT)': price_dict[symbol]})
                else:
                    errors.append(f"No price data available for {symbol}")
            except Exception as e:
                errors.append(f"Error processing {symbol}: {str(e)}")
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {str(e)}")
        errors.append("Unable to fetch prices from Binance")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        errors.append("An unexpected error occurred")
    
    return prices, errors

def play_alert_sound():
    audio_html = f"""<audio autoplay><source src=\"{ALERT_SOUND_URL}\" type=\"audio/mpeg\"></audio>"""
    st.components.v1.html(audio_html, height=0)

@st.cache_data(ttl=60)
def get_klines(symbol, interval, start_str=None, limit=None):
    client = Client("", "", testnet=True)
    return client.get_historical_klines(symbol, interval, start_str=start_str, limit=limit)

@st.cache_data(ttl=30)
def evolve_parameters(symbol, active_params, recent_trades):
    """Evolve trading parameters based on performance"""
    if len(recent_trades) < MIN_TRADES_FOR_EVOLUTION:
        return active_params

    # Calculate recent performance metrics
    win_rate = len(recent_trades[recent_trades['Outcome'] == 'Profit']) / len(recent_trades)
    avg_profit = recent_trades[recent_trades['Outcome'] == 'Profit']['P/L %'].mean()
    avg_loss = abs(recent_trades[recent_trades['Outcome'] == 'Loss']['P/L %'].mean())

    # Initialize evolution stats if needed
    if symbol not in st.session_state.evolution_stats:
        st.session_state.evolution_stats[symbol] = {
            'win_rate_history': [],
            'param_history': []
        }
    
    # Record current stats
    st.session_state.evolution_stats[symbol]['win_rate_history'].append(win_rate)
    st.session_state.evolution_stats[symbol]['param_history'].append(active_params.copy())

    # Evolve parameters based on performance
    new_params = active_params.copy()
    
    if win_rate > SUCCESS_THRESHOLD:
        # If strategy is working well, make minor adjustments
        if avg_profit < 2 * avg_loss:  # Risk/reward optimization
            new_params['tp_input'] = min(50.0, new_params['tp_input'] * (1 + LEARNING_RATE))
        elif avg_loss > avg_profit / 3:
            new_params['sl_input'] = max(1.0, new_params['sl_input'] * (1 - LEARNING_RATE))
    else:
        # If strategy is not performing well, make more significant changes
        if win_rate < 0.4:  # Poor performance
            # Adjust periods to be more conservative
            new_params['sma_period'] = min(200, int(new_params['sma_period'] * (1 + LEARNING_RATE)))
            new_params['rsi_period'] = min(50, int(new_params['rsi_period'] * (1 + LEARNING_RATE)))
        
        # Adjust stop loss if losing too much
        if avg_loss > 2 * avg_profit:
            new_params['sl_input'] = max(1.0, new_params['sl_input'] * (1 - LEARNING_RATE * 2))

    return new_params

def analyze_market_conditions(df):
    """Analyze market conditions to determine trading bias"""
    latest = df.iloc[-1]
    
    # Volume analysis
    avg_volume = df['volume'].rolling(window=TREND_PERIODS).mean()
    volume_spike = latest['volume'] > (avg_volume.iloc[-1] * VOLUME_THRESHOLD)
    
    # Trend analysis
    trend_sma = df['close'].rolling(window=TREND_PERIODS).mean().iloc[-1]
    price_trend = 'UPTREND' if latest['close'] > trend_sma else 'DOWNTREND'
    
    # Price momentum
    price_momentum = df['close'].pct_change().rolling(window=5).mean().iloc[-1]
    
    # Market conditions score (-100 to +100, negative favors short, positive favors long)
    score = 0
    
    # Trend contribution
    score += 30 if price_trend == 'UPTREND' else -30
    
    # RSI contribution
    if latest['rsi'] > RSI_OVERBOUGHT:
        score -= 20  # Overbought, favor short
    elif latest['rsi'] < RSI_OVERSOLD:
        score += 20  # Oversold, favor long
    
    # Volume contribution
    if volume_spike:
        score += 20 if price_momentum > 0 else -20
    
    # SMA position
    score += 30 if latest['close'] > latest['sma'] else -30
    
    return {
        'bias': 'LONG' if score > 20 else 'SHORT' if score < -20 else 'NEUTRAL',
        'score': score,
        'price_trend': price_trend,
        'volume_spike': volume_spike,
        'rsi_condition': 'OVERBOUGHT' if latest['rsi'] > RSI_OVERBOUGHT else 'OVERSOLD' if latest['rsi'] < RSI_OVERSOLD else 'NEUTRAL'
    }

def get_live_data(symbol, interval, sma_period, rsi_period, max_retries=3):
    """Get live market data with retry mechanism"""
    for attempt in range(max_retries):
        try:
            client = Client("", "", testnet=True)
            # Get more data than needed to ensure accurate indicator calculations
            limit = max(sma_period, rsi_period, TREND_PERIODS) * 3  
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if not klines:
                raise Exception("No data received from API")
                
            # Validate data format
            if not all(len(k) >= 12 for k in klines):  # Ensure all columns are present
                raise Exception("Invalid kline data format")
                
            # Convert data to DataFrame with proper column names
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp and numeric columns
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Drop any rows with invalid data
            data.dropna(subset=['close'], inplace=True)
            
            # Calculate technical indicators
            data['sma'] = calculate_sma(data['close'], sma_period)
            data['rsi'] = calculate_rsi(data['close'], rsi_period)
            
            # Drop any NaN values created by indicators and verify data
            data.dropna(subset=['sma', 'rsi'], inplace=True)
            
            if len(data) == 0:
                raise Exception("No valid data after processing")
                
            return data
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Could not fetch market data.")
                raise
    return None  # This line will only be reached if all retries fail

def find_optimal_parameters(data):
    param_combinations = list(itertools.product(
        PARAMETER_SEARCH_SPACE['sma_period'], PARAMETER_SEARCH_SPACE['rsi_period'],
        PARAMETER_SEARCH_SPACE['tp_input'], PARAMETER_SEARCH_SPACE['sl_input']
    ))
    best_pnl, best_params = -100, {}
    progress_bar, status_text = st.progress(0), st.empty()
    for i, params in enumerate(param_combinations):
        sma_p, rsi_p, tp_p, sl_p = params
        df = data.copy()
        df.ta.sma(length=sma_p, append=True); df.ta.rsi(length=rsi_p, append=True); df.dropna(inplace=True)
        entry_price, pnl, position = 0, 0, 'OUT'
        position_type = None
        
        for idx, row in df.iterrows():
            # Long entry conditions
            if position == 'OUT' and row['close'] > row[f'SMA_{sma_p}'] and row[f'RSI_{rsi_p}'] > 50:
                position, position_type, entry_price = 'IN', 'LONG', row['close']
            # Short entry conditions
            elif position == 'OUT' and row['close'] < row[f'SMA_{sma_p}'] and row[f'RSI_{rsi_p}'] < 30:
                position, position_type, entry_price = 'IN', 'SHORT', row['close']
            # Position exit conditions
            elif position == 'IN':
                if position_type == 'LONG':
                    if (row['close'] >= entry_price * (1 + tp_p / 100) or  # Take profit
                        row['close'] <= entry_price * (1 - sl_p / 100) or  # Stop loss
                        (row['close'] < row[f'SMA_{sma_p}'] and row[f'RSI_{rsi_p}'] < 50)):  # Reversal
                        pnl += (row['close'] - entry_price) / entry_price
                        position, position_type = 'OUT', None
                else:  # SHORT position
                    if (row['close'] <= entry_price * (1 - tp_p / 100) or  # Take profit
                        row['close'] >= entry_price * (1 + sl_p / 100) or  # Stop loss
                        (row['close'] > row[f'SMA_{sma_p}'] and row[f'RSI_{rsi_p}'] > 50)):  # Reversal
                        pnl += (entry_price - row['close']) / entry_price  # Note reversed calculation for shorts
                        position, position_type = 'OUT', None
        if pnl > best_pnl:
            best_pnl, best_params = pnl, {'sma_period': sma_p, 'rsi_period': rsi_p, 'tp_input': tp_p, 'sl_input': sl_p}
        progress_bar.progress((i + 1) / len(param_combinations))
        status_text.text(f"Testing combination {i+1}/{len(param_combinations)}...")
    status_text.success(f"Optimization complete! Best P/L on recent data: {best_pnl*100:.2f}%")
    return best_params

# --- STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("🏆 Final Crypto Trading Application")

# --- Real-time Multi-Crypto Price Table ---
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
all_options = ['Show All'] + symbols
prices, errors = get_multi_prices(symbols)
with st.container():
    st.subheader("Live Crypto Prices")
    if prices and 'Symbol' in prices[0] and 'Price (USDT)' in prices[0]:
        st.dataframe(pd.DataFrame(prices), use_container_width=True)
    else:
        st.warning("No price data available.")
    for err in errors:
        st.warning(err)

# --- Initialize Session State ---
if 'position' not in st.session_state: st.session_state.position = 'OUT'
if 'position_type' not in st.session_state: st.session_state.position_type = None  # 'LONG' or 'SHORT'
if 'entry_price' not in st.session_state: st.session_state.entry_price = 0.0
if 'entry_time' not in st.session_state: st.session_state.entry_time = None
if 'evolution_stats' not in st.session_state: st.session_state.evolution_stats = {}  # Track performance per symbol
trade_history_columns = ['Symbol', 'Position Type', 'Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'Outcome', 'P/L %']
if 'trade_history' not in st.session_state:
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            st.session_state.trade_history = pd.read_csv(TRADE_HISTORY_FILE)
            st.info(f"Loaded trade history from {TRADE_HISTORY_FILE}.")
        except Exception as e:
            st.warning(f"Could not read {TRADE_HISTORY_FILE}: {e}")
            st.session_state.trade_history = pd.DataFrame(columns=trade_history_columns)
    else:
        st.session_state.trade_history = pd.DataFrame(columns=trade_history_columns)
if 'optimal_params_per_symbol' not in st.session_state: st.session_state.optimal_params_per_symbol = {}
if 'last_optimization_time_per_symbol' not in st.session_state: st.session_state.last_optimization_time_per_symbol = {}
if 'mode' not in st.session_state: st.session_state.mode = 'Manual'

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Master Controls")
trade_symbol = st.sidebar.selectbox('Select Trading Symbol', all_options, index=1)
st.sidebar.radio(
    "Select Strategy Mode",
    ('Manual', 'Adaptive'),
    index=0 if st.session_state.mode == 'Manual' else 1,
    key='mode'
)

st.sidebar.header("Manual Parameters")
sma_input = st.sidebar.slider('SMA Period', 10, 200, 50, 1, disabled=(st.session_state.mode == 'Adaptive'))
rsi_input = st.sidebar.slider('RSI Period', 5, 50, 14, 1, disabled=(st.session_state.mode == 'Adaptive'))
tp_input = st.sidebar.slider('Take Profit (%)', 1.0, 50.0, 5.0, 0.5, disabled=(st.session_state.mode == 'Adaptive'))
sl_input = st.sidebar.slider('Stop-Loss (%)', 1.0, 50.0, 2.0, 0.5, disabled=(st.session_state.mode == 'Adaptive'))
check_interval_seconds = st.sidebar.slider('Check Interval (Seconds)', 30, 300, 60)

refresh = st.sidebar.button('Refresh Now')
if refresh:
    st.info("Please refresh your browser (F5) to update the data.")

placeholder = st.empty()

try:
    active_params = {}
    now = datetime.now().astimezone()
    with placeholder.container():
        if trade_symbol == 'Show All':
            st.header("Showing All Trade History and Performance for All Coins")
            history_df = st.session_state.trade_history
            total_trades = len(history_df)
            if total_trades > 0:
                win_rate = (len(history_df[history_df['Outcome'] == 'Profit']) / total_trades) * 100
                total_pnl = history_df['P/L %'].sum()
            else:
                win_rate = 0
                total_pnl = 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{win_rate:.2f}%")
            col3.metric("Total P/L", f"{total_pnl:.2f}%")
            st.info("Trading logic is disabled in 'Show All' mode. Please select a specific symbol to trade.")
            with st.expander("Full Trade History for All Coins"):
                if total_trades > 0:
                    def style_pnl(pnl):
                        color = 'green' if pnl > 0 else 'red' if pnl < 0 else 'gray'
                        return f'color: {color}'
                    st.dataframe(
                        history_df.sort_index(ascending=False).style.map(style_pnl, subset=['P/L %']),
                        use_container_width=True
                    )
                else:
                    st.write("No trades have been completed yet in this session.")
            st.stop()
        st.header(f"{'🤖' if st.session_state.mode == 'Adaptive' else '⚙️'} Mode: {st.session_state.mode}")
        if st.session_state.mode == 'Adaptive':
            last_opt_time = st.session_state.last_optimization_time_per_symbol.get(trade_symbol)
            if (trade_symbol not in st.session_state.optimal_params_per_symbol or 
                last_opt_time is None or 
                (now - last_opt_time) > OPTIMIZATION_INTERVAL):
                st.info(f"Recalibrating adaptive strategy for {trade_symbol}...")
                klines_for_opt = get_klines(trade_symbol, Client.KLINE_INTERVAL_5MINUTE, start_str=OPTIMIZATION_DATA_RANGE)
                opt_df = pd.DataFrame(klines_for_opt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                opt_df['timestamp'] = pd.to_datetime(opt_df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']: opt_df[col] = pd.to_numeric(opt_df[col])
                st.session_state.optimal_params_per_symbol[trade_symbol] = find_optimal_parameters(opt_df)
                st.session_state.last_optimization_time_per_symbol[trade_symbol] = now
            active_params = st.session_state.optimal_params_per_symbol[trade_symbol]
            col1, col2 = st.columns(2)
            with col1: st.write("Current Optimal Parameters:"); st.json(active_params)
            with col2: 
                st.write("Optimization Schedule:")
                last_opt_time = st.session_state.last_optimization_time_per_symbol[trade_symbol]
                st.metric("Last Optimization", last_opt_time.strftime('%I:%M %p'))
                next_opt_time = last_opt_time + OPTIMIZATION_INTERVAL
                st.metric("Next Optimization", next_opt_time.strftime('%I:%M %p'))
        else:
            st.info("Using parameters set manually in the sidebar.")
            active_params = {'sma_period': sma_input, 'rsi_period': rsi_input, 'tp_input': tp_input, 'sl_input': sl_input}
        # --- COMMON TRADING & DISPLAY LOGIC ---
        df = get_live_data(trade_symbol, Client.KLINE_INTERVAL_1MINUTE, active_params['sma_period'], active_params['rsi_period'])
        latest_row = df.iloc[-1]
        alert_message = ""
        play_sound = False
        if st.session_state.position == 'IN':
            if st.session_state.position_type == 'LONG':
                tp_price = st.session_state.entry_price * (1 + active_params['tp_input'] / 100)
                sl_price = st.session_state.entry_price * (1 - active_params['sl_input'] / 100)
                exit_reason = ""
                if latest_row['close'] >= tp_price: exit_reason = "Take Profit"
                elif latest_row['close'] <= sl_price: exit_reason = "Stop-Loss"
                elif latest_row['close'] < latest_row['sma'] and latest_row['rsi'] < 50: exit_reason = "Indicator Reversal"
                pnl_multiplier = 1
            else:  # SHORT position
                tp_price = st.session_state.entry_price * (1 - active_params['tp_input'] / 100)
                sl_price = st.session_state.entry_price * (1 + active_params['sl_input'] / 100)
                exit_reason = ""
                if latest_row['close'] <= tp_price: exit_reason = "Take Profit"
                elif latest_row['close'] >= sl_price: exit_reason = "Stop-Loss"
                elif latest_row['close'] > latest_row['sma'] and latest_row['rsi'] > 50: exit_reason = "Indicator Reversal"
                pnl_multiplier = -1  # Reverse P&L calculation for shorts
            if exit_reason:
                exit_price = latest_row['close']
                exit_time = pd.Timestamp.now(tz='Asia/Kolkata')
                pnl_percentage = ((exit_price - st.session_state.entry_price) / st.session_state.entry_price) * 100 * pnl_multiplier
                outcome = "Profit" if pnl_percentage >= 0 else "Loss"
                new_trade = pd.DataFrame([{
                    'Symbol': trade_symbol,
                    'Position Type': st.session_state.position_type,
                    'Entry Time': st.session_state.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Entry Price': st.session_state.entry_price,
                    'Exit Time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit Price': exit_price,
                    'Outcome': outcome,
                    'P/L %': pnl_percentage
                }])
                if not new_trade.empty:
                    st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_trade], ignore_index=True)
                try:
                    st.session_state.trade_history.to_csv(TRADE_HISTORY_FILE, index=False)
                    st.info(f"Trade history saved to {TRADE_HISTORY_FILE}.")
                    
                    # Evolve parameters after trade completion
                    if st.session_state.mode == 'Adaptive':
                        recent_trades = st.session_state.trade_history[
                            st.session_state.trade_history['Symbol'] == trade_symbol
                        ].tail(MIN_TRADES_FOR_EVOLUTION)
                        
                        evolved_params = evolve_parameters(trade_symbol, active_params, recent_trades)
                        if evolved_params != active_params:
                            st.session_state.optimal_params_per_symbol[trade_symbol] = evolved_params
                            st.info("🧬 Strategy evolved based on recent performance!")
                            
                except Exception as e:
                    st.warning(f"Could not save trade history to {TRADE_HISTORY_FILE}: {e}")
                alert_message = f"🔔 EXIT: {exit_reason} | P/L: {pnl_percentage:.2f}%"
                st.session_state.position = 'OUT'
                st.session_state.entry_price = 0.0
                st.session_state.entry_time = None
                play_sound = True
        elif st.session_state.position == 'OUT':
            # Analyze market conditions
            market_analysis = analyze_market_conditions(df)
            current_price = latest_row['close']
            
            # Display market analysis
            st.sidebar.markdown("### Market Analysis")
            st.sidebar.markdown(f"**Market Bias:** {market_analysis['bias']}")
            st.sidebar.markdown(f"**Trend:** {market_analysis['price_trend']}")
            st.sidebar.markdown(f"**RSI Condition:** {market_analysis['rsi_condition']}")
            st.sidebar.markdown(f"**Volume Spike:** {'Yes' if market_analysis['volume_spike'] else 'No'}")
            st.sidebar.progress(int((market_analysis['score'] + 100) / 2))  # Convert score to 0-100 range
            
            # Check for long (buy) signal
            if (market_analysis['bias'] == 'LONG' and 
                latest_row['close'] > latest_row['sma'] and 
                latest_row['rsi'] > 50):
                alert_message = f"✅ NEW LONG (BUY) SIGNAL at ${current_price:,.2f}!"
                st.session_state.position = 'IN'
                st.session_state.position_type = 'LONG'
                st.session_state.entry_price = current_price
                st.session_state.entry_time = pd.Timestamp.now(tz='Asia/Kolkata')
                play_sound = True
            
            # Check for short (sell) signal
            elif (market_analysis['bias'] == 'SHORT' and 
                  latest_row['close'] < latest_row['sma'] and 
                  latest_row['rsi'] < RSI_OVERSOLD):
                alert_message = f"🔻 NEW SHORT (SELL) SIGNAL at ${current_price:,.2f}!"
                st.session_state.position = 'IN'
                st.session_state.position_type = 'SHORT'
                st.session_state.entry_price = current_price
                st.session_state.entry_time = pd.Timestamp.now(tz='Asia/Kolkata')
                play_sound = True
        # --- DASHBOARD ---
        st.header(f"Session Performance for {trade_symbol}")
        history_df = st.session_state.trade_history
        symbol_history = history_df[history_df['Symbol'] == trade_symbol] if not history_df.empty else history_df
        total_trades = len(symbol_history)
        if total_trades > 0:
            win_rate = (len(symbol_history[symbol_history['Outcome'] == 'Profit']) / total_trades) * 100
            total_pnl = symbol_history['P/L %'].sum()
        else:
            win_rate = 0
            total_pnl = 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.2f}%")
        col3.metric("Total P/L", f"{total_pnl:.2f}%")
        st.header("Live Status for " + trade_symbol)
        if st.session_state.position == 'IN':
            position_emoji = "🟢" if st.session_state.position_type == "LONG" else "🔻"
            st.subheader(f"{position_emoji} IN {st.session_state.position_type} TRADE for {trade_symbol}")
            if st.session_state.position_type == 'LONG':
                tp_price = st.session_state.entry_price * (1 + active_params['tp_input'] / 100)
                sl_price = st.session_state.entry_price * (1 - active_params['sl_input'] / 100)
            else:  # SHORT position
                tp_price = st.session_state.entry_price * (1 - active_params['tp_input'] / 100)
                sl_price = st.session_state.entry_price * (1 + active_params['sl_input'] / 100)
            col1_trade, col2_trade, col3_trade = st.columns(3)
            col1_trade.metric("Entry Price", f"${st.session_state.entry_price:,.2f}")
            col2_trade.metric("Take Profit Target", f"${tp_price:,.2f}")
            col3_trade.metric("Stop-Loss Target", f"${sl_price:,.2f}")
        else:
            st.subheader(f"⚪ OUT OF TRADE for {trade_symbol}")
            st.info("Status: Looking for an entry signal based on your parameters.")
        st.write("---")
        with st.expander(f"Full Trade History for {trade_symbol}"):
            if total_trades > 0:
                def style_pnl(pnl):
                    color = 'green' if pnl > 0 else 'red' if pnl < 0 else 'gray'
                    return f'color: {color}'
                st.dataframe(
                    symbol_history.sort_index(ascending=False).style.map(style_pnl, subset=['P/L %']),
                    use_container_width=True
                )
            else:
                st.write(f"                st.write(f"No trades have been completed yet for {trade_symbol} in this session.")
        
        st.write("---")
        st.subheader(f"Live Market Data for {trade_symbol}")")
        st.subheader(f"Live Market Data for {trade_symbol}")
        col_market_1, col_market_2, col_market_3 = st.columns(3)
        col_market_1.metric("Current Price", f"${latest_row['close']:,.2f}")
        col_market_2.metric(f"SMA({active_params['sma_period']})", f"${latest_row['sma']:,.2f}")
        col_market_3.metric(f"RSI({active_params['rsi_period']})", f"{latest_row['rsi']:.2f}")
        st.line_chart(df['close'].tail(100))
        st.write(f"Last updated: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
        if play_sound: play_alert_sound()
except Exception as e:
    with placeholder.container():
        st.error(f"An error occurred: {e}")
        st.error(f"Will retry in {check_interval_seconds} sec# --- NO AUTO-REFRESH: Please use the Refresh Now button or F5 to update manually ---