import pandas as pd
import pandas_ta as ta
from binance.client import Client
import streamlit as st
import time
import itertools
from datetime import timedelta

# --- CONFIGURATION ---
ALERT_SOUND_URL = "https://www.soundjay.com/buttons/sounds/button-16.mp3"

# ** NEW: Define the parameter search space for optimization **
PARAMETER_SEARCH_SPACE = {
    'sma_period': [20, 30, 50],
    'rsi_period': [10, 14, 18],
    'tp_input': [3.0, 5.0],
    'sl_input': [2.0, 3.0]
}
OPTIMIZATION_INTERVAL = timedelta(hours=4) # Recalibrate every 4 hours
OPTIMIZATION_DATA_RANGE = "3 days ago UTC" # Use last 3 days of data for optimization

# --- HELPER FUNCTIONS ---
def play_alert_sound():
    audio_html = f"""<audio autoplay><source src=\"{ALERT_SOUND_URL}\" type=\"audio/mpeg\"></audio>"""
    st.components.v1.html(audio_html, height=0)

@st.cache_data(ttl=60)
def get_klines(symbol, interval, start_str=None, limit=None):
    client = Client()
    return client.get_historical_klines(symbol, interval, start_str=start_str, limit=limit)

@st.cache_data(ttl=30)
def get_live_data(symbol, interval, sma_period, rsi_period):
    client = Client()
    limit = max(sma_period, rsi_period) * 2
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col])
    data.ta.sma(length=sma_period, append=True)
    data.ta.rsi(length=rsi_period, append=True)
    data.rename(columns={f'SMA_{sma_period}': 'sma', f'RSI_{rsi_period}': 'rsi'}, inplace=True)
    return data

# ** NEW: The core optimization function **
def find_optimal_parameters(data):
    """Runs a backtest for every parameter combination and finds the best."""
    param_combinations = list(itertools.product(
        PARAMETER_SEARCH_SPACE['sma_period'],
        PARAMETER_SEARCH_SPACE['rsi_period'],
        PARAMETER_SEARCH_SPACE['tp_input'],
        PARAMETER_SEARCH_SPACE['sl_input']
    ))
    
    best_pnl = -100
    best_params = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, params in enumerate(param_combinations):
        sma_p, rsi_p, tp_p, sl_p = params
        
        # Simple backtest logic for this combination
        df = data.copy()
        df.ta.sma(length=sma_p, append=True)
        df.ta.rsi(length=rsi_p, append=True)
        df.dropna(inplace=True)
        
        entry_price = 0
        pnl = 0
        position = 'OUT'

        for idx, row in df.iterrows():
            if position == 'OUT' and row['close'] > row[f'SMA_{sma_p}'] and row[f'RSI_{rsi_p}'] > 50:
                position = 'IN'
                entry_price = row['close']
            elif position == 'IN':
                if row['close'] >= entry_price * (1 + tp_p / 100) or row['close'] <= entry_price * (1 - sl_p / 100):
                    pnl += (row['close'] - entry_price) / entry_price
                    position = 'OUT'
        
        if pnl > best_pnl:
            best_pnl = pnl
            best_params = {'sma_period': sma_p, 'rsi_period': rsi_p, 'tp_input': tp_p, 'sl_input': sl_p}
        
        progress = (i + 1) / len(param_combinations)
        progress_bar.progress(progress)
        status_text.text(f"Testing combination {i+1}/{len(param_combinations)}...")

    status_text.text(f"Optimization complete! Best P/L on recent data: {best_pnl*100:.2f}%")
    return best_params

# --- STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("🤖 Adaptive Crypto Trading Bot")

# Initialize session state
if 'position' not in st.session_state: st.session_state.position = 'OUT'
if 'entry_price' not in st.session_state: st.session_state.entry_price = 0.0
if 'entry_time' not in st.session_state: st.session_state.entry_time = None
if 'trade_history' not in st.session_state: st.session_state.trade_history = pd.DataFrame()
if 'optimal_params' not in st.session_state: st.session_state.optimal_params = None
if 'last_optimization_time' not in st.session_state: st.session_state.last_optimization_time = None

symbol_input = st.sidebar.text_input('Crypto Symbol', 'BTCUSDT')
check_interval_seconds = st.sidebar.slider('Check Interval (Seconds)', 30, 300, 60)

placeholder = st.empty()

try:
    now = pd.Timestamp.now(tz='UTC')
    # --- 1. OPTIMIZATION LOGIC ---
    if st.session_state.optimal_params is None or (st.session_state.last_optimization_time is None) or (now - st.session_state.last_optimization_time) > OPTIMIZATION_INTERVAL:
        with placeholder.container():
            st.info(f"Recalibrating strategy for {symbol_input}... This may take a minute.")
            klines_for_opt = get_klines(symbol_input, Client.KLINE_INTERVAL_5MINUTE, start_str=OPTIMIZATION_DATA_RANGE)
            opt_df = pd.DataFrame(klines_for_opt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            opt_df['timestamp'] = pd.to_datetime(opt_df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                opt_df[col] = pd.to_numeric(opt_df[col])
            st.session_state.optimal_params = find_optimal_parameters(opt_df)
            st.session_state.last_optimization_time = now

    # --- 2. LIVE TRADING LOGIC (using optimal params) ---
    active_params = st.session_state.optimal_params
    df = get_live_data(symbol_input, Client.KLINE_INTERVAL_1MINUTE, active_params['sma_period'], active_params['rsi_period'])
    latest_row = df.iloc[-1]
    
    alert_message = ""
    play_sound = False

    if st.session_state.position == 'IN':
        tp_price = st.session_state.entry_price * (1 + active_params['tp_input'] / 100)
        sl_price = st.session_state.entry_price * (1 - active_params['sl_input'] / 100)
        exit_reason = ""
        if latest_row['close'] >= tp_price: exit_reason = "Take Profit"
        elif latest_row['close'] <= sl_price: exit_reason = "Stop-Loss"
        elif latest_row['close'] < latest_row['sma'] and latest_row['rsi'] < 50: exit_reason = "Indicator Reversal"
        if exit_reason:
            exit_price = latest_row['close']
            exit_time = pd.Timestamp.now(tz='Asia/Kolkata')
            pnl_percentage = ((exit_price - st.session_state.entry_price) / st.session_state.entry_price) * 100
            outcome = "Profit" if pnl_percentage >= 0 else "Loss"
            new_trade = pd.DataFrame([{
                'Entry Time': st.session_state.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Entry Price': st.session_state.entry_price,
                'Exit Time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Exit Price': exit_price,
                'Outcome': outcome,
                'P/L %': pnl_percentage
            }])
            st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_trade], ignore_index=True)
            alert_message = f"🔔 EXIT: {exit_reason} | P/L: {pnl_percentage:.2f}%"
            st.session_state.position = 'OUT'
            st.session_state.entry_price = 0.0
            st.session_state.entry_time = None
            play_sound = True
    elif st.session_state.position == 'OUT':
        if latest_row['close'] > latest_row['sma'] and latest_row['rsi'] > 50:
            alert_message = f"✅ NEW BUY SIGNAL at ${latest_row['close']:,.2f}!"
            st.session_state.position = 'IN'
            st.session_state.entry_price = latest_row['close']
            st.session_state.entry_time = pd.Timestamp.now(tz='Asia/Kolkata')
            play_sound = True
    
    # --- 3. UPDATE DASHBOARD ---
    with placeholder.container():
        st.header("Adaptive Strategy Status")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Current Optimal Parameters:")
            st.json(active_params)
        with col2:
            st.write("Optimization Schedule:")
            st.metric("Last Optimization", st.session_state.last_optimization_time.strftime('%I:%M %p %Z'))
            next_opt_time = st.session_state.last_optimization_time + OPTIMIZATION_INTERVAL
            st.metric("Next Optimization", next_opt_time.strftime('%I:%M %p %Z'))
        st.write("---")
        st.header("Live Status & Trade History")
        st.info("The live status and trade history display would go here, identical to our previous app.")
        if play_sound: play_alert_sound()
except Exception as e:
    with placeholder.container():
        st.error(f"An error occurred: {e}")
        st.error(f"Will retry in {check_interval_seconds} seconds...")

# --- AUTO-REFRESH ---
st.write(f"""<meta http-equiv='refresh' content='{check_interval_seconds}'>""", unsafe_allow_html=True)