# main.py - FINAL STABLE VERSION WITH KRAKEN RETRY LOGIC

import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Bot
from flask import Flask, jsonify, render_template_string
import threading
import time
import traceback 

# --- ML Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 

# --- CONFIGURATION LOADING ---
from dotenv import load_dotenv 
load_dotenv() 

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# Asset list (Can be Crypto or Forex Proxies)
SYMBOLS = os.getenv("ASSETS", "BTC/USDT,ETH/USDT,SOL/USDT").split(',')
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
DAILY_TIMEFRAME = '1d'

# Initialize Bot and Exchange (Kraken Public)
bot = Bot(token=TELEGRAM_BOT_TOKEN)
exchange = ccxt.kraken({
    'enableRateLimit': True,
    'rateLimit': 2000, # Increased for stability
})

# Global ML Model and Scaler
ML_MODEL = None
SCALER = None

# --- Status Tracking ---
bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "monitored_assets": SYMBOLS,
    "uptime_start": datetime.now().isoformat()
}

# =========================================================================
# === TECHNICAL ANALYSIS & ML FUNCTIONS ===
# =========================================================================

def train_prediction_model(df):
    global SCALER
    try:
        if len(df) < 100: return None, None
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        df['fast_over_slow'] = np.where(df['fast_sma'] > df['slow_sma'], 1, 0)
        df['close_over_fast'] = np.where(df['close'] > df['fast_sma'], 1, 0)
        returns = df['close'].pct_change()
        df['volatility'] = returns.rolling(20, min_periods=1).std().fillna(0)
        df = df.dropna()
        X = df[['fast_over_slow', 'close_over_fast', 'volatility']]; y = df['target']
        SCALER = StandardScaler()
        X_scaled = SCALER.fit_transform(X)
        model = LogisticRegression(solver='liblinear'); model.fit(X_scaled, y)
        return model, SCALER
    except: return None, None

def calculate_cpr_levels(df_daily):
    if df_daily.empty or len(df_daily) < 2: return None
    prev_day = df_daily.iloc[-2]
    H, L, C = prev_day['high'], prev_day['low'], prev_day['close']
    PP = (H + L + C) / 3.0; BC = (H + L) / 2.0; TC = PP - BC + PP
    R1 = 2*PP - L; S1 = 2*PP - H; R2 = PP + (H - L); S2 = PP - (H - L)
    return {'PP': PP, 'TC': TC, 'BC': BC, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2}

# =========================================================================
# === THE KRAKEN-FIXED DATA FETCHING FUNCTION ===
# =========================================================================

def fetch_and_prepare_data(symbol, timeframe, daily_timeframe='1d', limit=500):
    max_retries = 3
    retry_delay = 5 

    for attempt in range(max_retries):
        try:
            if not exchange.markets:
                exchange.load_markets()

            # Normalize symbol for Kraken (e.g., 'ETH/USDT' -> 'ETHUSDT')
            market_info = exchange.market(symbol)
            kraken_symbol = market_info['id']

            # Fetch OHLCV with explicit timeout to prevent thread hang
            ohlcv = exchange.fetch_ohlcv(kraken_symbol, timeframe, limit=limit, params={'timeout': 20000})
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True); df = df.dropna()

            # Indicators
            df['fast_sma'] = df['close'].rolling(window=9).mean()
            df['slow_sma'] = df['close'].rolling(window=20).mean()
            df = df.dropna()

            if len(df) < 20: return pd.DataFrame(), None

            # Daily data for CPR
            ohlcv_daily = exchange.fetch_ohlcv(kraken_symbol, daily_timeframe, limit=20, params={'timeout': 20000})
            df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_daily.set_index('timestamp', inplace=True)
            
            cpr_levels = calculate_cpr_levels(df_daily)
            return df, cpr_levels

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return pd.DataFrame(), None

# =========================================================================
# === ANALYSIS & SIGNAL LOGIC ===
# =========================================================================

def generate_and_send_signal(symbol):
    global bot_stats
    try:
        async def send_msg(text):
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='HTML')

        df, cpr = fetch_and_prepare_data(symbol, TIMEFRAME)
        if df.empty or cpr is None: return

        latest = df.iloc[-1]; current_price = latest['close']
        fast_sma = latest['fast_sma']; slow_sma = latest['slow_sma']
        
        # Trend & Signal
        trend = "Uptrend" if fast_sma > slow_sma else "Downtrend"
        trend_emoji = "🟢" if trend == "Uptrend" else "🔴"
        
        signal = "HOLD"; emoji = "🟡"
        if trend == "Uptrend" and current_price > cpr['PP']: signal = "STRONG BUY"; emoji = "🚀"
        elif trend == "Downtrend" and current_price < cpr['PP']: signal = "STRONG SELL"; emoji = "🔻"

        message = (
            f"<b>{emoji} {symbol} ANALYSIS</b>\n"
            f"Price: <code>{current_price:,.4f}</code>\n"
            f"Trend: {trend_emoji} {trend}\n"
            f"<b>Signal: {signal}</b>\n\n"
            f"PP: <code>{cpr['PP']:,.4f}</code>\n"
            f"R1: <code>{cpr['R1']:,.4f}</code> | S1: <code>{cpr['S1']:,.4f}</code>"
        )
        
        asyncio.run(send_msg(message))
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"

    except Exception as e:
        print(f"❌ Error in analysis for {symbol}: {e}")

# =========================================================================
# === SCHEDULER & FLASK ===
# =========================================================================

def start_scheduler_thread():
    # Initial ML Training
    global ML_MODEL, SCALER
    try:
        exchange.load_markets()
        ohlcv = exchange.fetch_ohlcv(exchange.market(SYMBOLS[0])['id'], TIMEFRAME, limit=500)
        df_train = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_train['fast_sma'] = df_train['close'].rolling(9).mean()
        df_train['slow_sma'] = df_train['close'].rolling(20).mean()
        ML_MODEL, SCALER = train_prediction_model(df_train.dropna())
    except: pass

    scheduler = BackgroundScheduler()
    for s in SYMBOLS:
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[s.strip()])
    scheduler.start()
    
    # Run once immediately
    for s in SYMBOLS: generate_and_send_signal(s.strip())
    while True: time.sleep(10)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string("<h1>Bot Active</h1><p>Analyses: {{a}}</p><p>Status: {{s}}</p>", 
                                 a=bot_stats['total_analyses'], s=bot_stats['status'])

@app.route('/health')
def health(): return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    threading.Thread(target=start_scheduler_thread, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
