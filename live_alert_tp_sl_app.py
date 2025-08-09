import pandas as pd
import pandas_ta as ta
from binance.client import Client
import streamlit as st
import time

# --- CONFIGURATION ---
ALERT_SOUND_URL = "https://www.soundjay.com/buttons/sounds/button-16.mp3"

# --- HELPER FUNCTIONS ---
def play_alert_sound():
    audio_html = f"""<audio autoplay><source src="{ALERT_SOUND_URL}" type="audio/mpeg"></audio>"""
    st.components.v1.html(audio_html, height=0)

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

# --- STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("📈 The Complete Crypto Trading Tool")

# Sidebar for user inputs
st.sidebar.header("Monitoring Parameters")
symbol_input = st.sidebar.text_input('Crypto Symbol', 'BTCUSDT')
sma_input = st.sidebar.slider('SMA Period', min_value=10, max_value=200, value=50, step=1)
rsi_input = st.sidebar.slider('RSI Period', min_value=5, max_value=50, value=14, step=1)
check_interval_seconds = st.sidebar.slider('Check Interval (Seconds)', min_value=10, max_value=300, value=60, step=10)

st.sidebar.header("Exit Strategy Parameters")
tp_input = st.sidebar.slider('Take Profit (%)', min_value=1.0, max_value=50.0, value=5.0, step=0.5)
sl_input = st.sidebar.slider('Stop-Loss (%)', min_value=1.0, max_value=50.0, value=2.0, step=0.5)

# Initialize session state for trade tracking
if 'position' not in st.session_state:
    st.session_state.position = 'OUT'
if 'entry_price' not in st.session_state:
    st.session_state.entry_price = 0.0
if 'entry_time' not in st.session_state:
    st.session_state.entry_time = None
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=['Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'Outcome', 'P/L %'])

placeholder = st.empty()

try:
    df = get_live_data(symbol_input, Client.KLINE_INTERVAL_1MINUTE, sma_input, rsi_input)
    latest_row = df.iloc[-1]
    
    alert_message = ""
    play_sound = False

    # --- A: CHECK FOR AN EXIT if we are currently IN a trade ---
    if st.session_state.position == 'IN':
        tp_price = st.session_state.entry_price * (1 + tp_input / 100)
        sl_price = st.session_state.entry_price * (1 - sl_input / 100)
        exit_reason = ""

        if latest_row['close'] >= tp_price:
            exit_reason = "Take Profit"
        elif latest_row['close'] <= sl_price:
            exit_reason = "Stop-Loss"
        elif latest_row['close'] < latest_row['sma'] and latest_row['rsi'] < 50:
            exit_reason = "Indicator Reversal"

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

    # --- B: CHECK FOR AN ENTRY if we are currently OUT of a trade ---
    elif st.session_state.position == 'OUT':
        if latest_row['close'] > latest_row['sma'] and latest_row['rsi'] > 50:
            alert_message = f"✅ NEW BUY SIGNAL at ${latest_row['close']:,.2f}!"
            st.session_state.position = 'IN'
            st.session_state.entry_price = latest_row['close']
            st.session_state.entry_time = pd.Timestamp.now(tz='Asia/Kolkata')
            play_sound = True
    
    # --- UPDATE DASHBOARD ---
    with placeholder.container():
        if "BUY SIGNAL" in alert_message: st.success(alert_message)
        if "EXIT SIGNAL" in alert_message: st.warning(alert_message)

        st.header("Session Performance")
        
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
        
        st.header("Live Status")
        if st.session_state.position == 'IN':
            st.subheader("🟢 IN TRADE")
            tp_price = st.session_state.entry_price * (1 + tp_input / 100)
            sl_price = st.session_state.entry_price * (1 - sl_input / 100)
            col1_trade, col2_trade, col3_trade = st.columns(3)
            col1_trade.metric("Entry Price", f"${st.session_state.entry_price:,.2f}")
            col2_trade.metric("Take Profit Target", f"${tp_price:,.2f}")
            col3_trade.metric("Stop-Loss Target", f"${sl_price:,.2f}")
        else:
            st.subheader("⚪ OUT OF TRADE")
            st.info("Status: Looking for an entry signal based on your parameters.")

        st.write("---")
        with st.expander("Full Trade History"):
            if total_trades > 0:
                def style_pnl(pnl):
                    color = 'green' if pnl > 0 else 'red' if pnl < 0 else 'gray'
                    return f'color: {color}'
                st.dataframe(
                    history_df.sort_index(ascending=False).style.applymap(style_pnl, subset=['P/L %']),
                    use_container_width=True
                )
            else:
                st.write("No trades have been completed yet in this session.")

        st.write("---")
        st.subheader("Live Market Data")
        col_market_1, col_market_2, col_market_3 = st.columns(3)
        col_market_1.metric("Current Price", f"${latest_row['close']:,.2f}")
        col_market_2.metric(f"SMA({sma_input})", f"${latest_row['sma']:,.2f}")
        col_market_3.metric(f"RSI({rsi_input})", f"{latest_row['rsi']:.2f}")

        st.line_chart(df['close'].tail(100))
        st.write(f"Last updated: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %I:%M:%S %p %Z')}")

        if play_sound: play_alert_sound()

except Exception as e:
    with placeholder.container():
        st.error(f"An error occurred: {e}")
        st.error(f"Will retry in {check_interval_seconds} seconds...")

# --- AUTO-REFRESH ---
st.write(f"""<meta http-equiv='refresh' content='{check_interval_seconds}'>""", unsafe_allow_html=True)