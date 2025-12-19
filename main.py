import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler # Switched to Background for Gunicorn stability
from telegram import Bot
from flask import Flask, jsonify, render_template_string
import threading
import time
import traceback 

# --- ML Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 

# --- CONFIGURATION LOADING ---
from dotenv import load_dotenv 
load_dotenv() 

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# Fixed CRYPTOS formatting to handle spaces/normalization
CRYPTOS = [s.strip() for s in os.getenv("CRYPTOS", "BTC/USDT,ETH/USDT").split(',')]
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
DAILY_TIMEFRAME = '1d' 

# Initialize Bot and Exchange
bot = Bot(token=TELEGRAM_BOT_TOKEN)
exchange = ccxt.kraken({
    'enableRateLimit': True,
    'rateLimit': 2000, # Increased for Kraken stability
})

# Global ML Model and Scaler
ML_MODEL = None
SCALER = None

# ========== FLASK WEB SERVER & STATUS TRACKING ==========
app = Flask(__name__) 

bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "monitored_assets": CRYPTOS,
    "uptime_start": datetime.now().isoformat()
}

@app.route('/')
def home():
    return render_template_string("<h1>Bot Status Page</h1>" + 
                                 f"<p>Status: {bot_stats['status']}</p>" + 
                                 f"<p>Analyses: {bot_stats['total_analyses']}</p>" +
                                 f"<p>Last Analysis: {bot_stats['last_analysis'] or 'N/A'}</p>")

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

# ========== ML & TECHNICAL FUNCTIONS ==========

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
        X = df[['fast_over_slow', 'close_over_fast', 'volatility']].copy()
        y = df['target'].copy()
        split_idx = int(len(X) * 0.9)
        SCALER = StandardScaler()
        X_train_scaled = SCALER.fit_transform(X.iloc[:split_idx])
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X_train_scaled, y.iloc[:split_idx])
        return model, SCALER
    except: return None, None

def calculate_cpr_levels(df_daily):
    if df_daily.empty or len(df_daily) < 2: return None
    prev_day = df_daily.iloc[-2]  
    H, L, C = prev_day['high'], prev_day['low'], prev_day['close']
    PP = (H + L + C) / 3.0
    BC = (H + L) / 2.0
    TC = PP - BC + PP
    return {'PP': PP, 'TC': TC, 'BC': BC, 'R1': 2*PP-L, 'S1': 2*PP-H}

# ========== ROBUST FETCHING WITH RETRIES ==========

def fetch_and_prepare_data(symbol, timeframe, daily_timeframe='1d', limit=500):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not exchange.markets: exchange.load_markets()
            # Normalize symbol (ETH/USDT -> ETHUSDT)
            market_id = exchange.market(symbol)['id']
            
            # Fetch with explicit timeout
            ohlcv = exchange.fetch_ohlcv(market_id, timeframe, limit=limit, params={'timeout': 20000})
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True); df = df.dropna()
            
            df['fast_sma'] = df['close'].rolling(window=9).mean()
            df['slow_sma'] = df['close'].rolling(window=20).mean()
            df = df.dropna()
            
            if len(df) < 20: return pd.DataFrame(), None

            ohlcv_daily = exchange.fetch_ohlcv(market_id, daily_timeframe, limit=20, params={'timeout': 20000})
            df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_daily.set_index('timestamp', inplace=True)
            
            return df, calculate_cpr_levels(df_daily)
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed for {symbol}: {e}")
            if attempt < max_retries - 1: time.sleep(5)
            else: return pd.DataFrame(), None

# ========== SIGNAL GENERATION & SCHEDULER ==========

def generate_and_send_signal(symbol):
    try:
        df, cpr = fetch_and_prepare_data(symbol, TIMEFRAME)
        if df.empty or cpr is None: return

        latest = df.iloc[-1]
        current_price = latest['close']
        fast_sma = latest['fast_sma']
        slow_sma = latest['slow_sma']
        
        # Simple Logic for Signal
        trend = "Uptrend" if fast_sma > slow_sma else "Downtrend"
        signal = "HOLD"
        if trend == "Uptrend" and current_price > cpr['PP']: signal = "STRONG BUY"
        elif trend == "Downtrend" and current_price < cpr['PP']: signal = "STRONG SELL"

        message = (
            f"╔════════════════════════════════╗\n"
            f"  🧠 <b>AI MARKET INTELLIGENCE REPORT</b>\n"
            f"╚════════════════════════════════╝\n\n"
            f"<b>{symbol}</b> | {datetime.now().strftime('%H:%M UTC')}\n\n"
            f"Signal: <b>{signal}</b>\n"
            f"Price: <code>{current_price:,.2f}</code>\n"
            f"Trend: {'🟢' if trend == 'Uptrend' else '🔴'} {trend}\n"
            f"PP: <code>{cpr['PP']:.2f}</code>\n"
        )

        asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML'))
        
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"

    except Exception as e:
        print(f"❌ Error for {symbol}: {e}")

# ========== GUNICORN-SAFE INITIALIZATION ==========

def start_bot():
    global ML_MODEL, SCALER
    print("🤖 Initializing Bot...")
    try:
        # Initial ML Training
        exchange.load_markets()
        ohlcv = exchange.fetch_ohlcv(exchange.market(CRYPTOS[0])['id'], TIMEFRAME, limit=500)
        df_train = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_train['fast_sma'] = df_train['close'].rolling(9).mean()
        df_train['slow_sma'] = df_train['close'].rolling(20).mean()
        ML_MODEL, SCALER = train_prediction_model(df_train.dropna())
    except: pass

    scheduler = BackgroundScheduler()
    for s in CRYPTOS:
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[s.strip()])
    scheduler.start()
    
    # Run once immediately in separate threads to not block boot
    for s in CRYPTOS:
        threading.Thread(target=generate_and_send_signal, args=(s.strip(),)).start()

# Call initialization outside main block so Gunicorn picks it up
start_bot()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
