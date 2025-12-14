# main.py - The FINAL, fully optimized structure for Render Web Service

import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
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
CRYPTOS = os.getenv("CRYPTOS", "BTC/USDT,ETH/USDT").split(',') 
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
DAILY_TIMEFRAME = '1d' 
ANALYSIS_INTERVAL = 30 

# Initialize Bot and Exchange
bot = Bot(token=TELEGRAM_BOT_TOKEN)
exchange = ccxt.kraken({
    'enableRateLimit': True,
    'rateLimit': 1000, 
})

# Global ML Model and Scaler (must be global for prediction)
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

@app.route('/status')
def status():
    return jsonify(bot_stats), 200

# ========== ML TRAINING FUNCTION ==========

def train_prediction_model(df):
    """Trains a Logistic Regression model and returns the model and scaler."""
    global SCALER
    
    if len(df) < 500:
        print("⚠️ Not enough data (need 500+ rows) for robust ML training. Skipping.")
        return None, None

    # 1. Target Definition (y)
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    # 2. Feature Engineering (X)
    df['fast_over_slow'] = np.where(df['fast_sma'] > df['slow_sma'], 1, 0)
    df['close_over_fast'] = np.where(df['close'] > df['fast_sma'], 1, 0)
    df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0) 
    
    df = df.dropna()
    
    X = df[['fast_over_slow', 'close_over_fast', 'volatility']]
    y = df['target']
    
    X_train = X.iloc[:-int(len(X) * 0.1)]
    y_train = y.iloc[:-int(len(y) * 0.1)]

    # 3. Scaling
    SCALER = StandardScaler()
    X_train_scaled = SCALER.fit_transform(X_train)

    # 4. Training
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_scaled, y_train)
    
    print(f"✅ ML Model trained successfully. Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    return model, SCALER

# ========== TECHNICAL ANALYSIS FUNCTIONS ==========

# 1. CPR Calculation Function
def calculate_cpr_levels(df_daily):
    """Calculates Daily Pivot Points (PP, TC, BC, R/S levels) from previous day's data."""
    if df_daily.empty or len(df_daily) < 2:
        return None

    prev_day = df_daily.iloc[-2]  
    H, L, C = prev_day['high'], prev_day['low'], prev_day['close']
    
    PP = (H + L + C) / 3.0
    BC = (H + L) / 2.0
    TC = PP - BC + PP
    
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    R3 = H + 2 * (PP - L)
    S3 = L - 2 * (H - PP)
    
    return {'PP': PP, 'TC': TC, 'BC': BC, 'R1': R1, 'S1': S1,
            'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}

# 2. Data Fetching and Preparation Function
def fetch_and_prepare_data(symbol, timeframe, daily_timeframe='1d', limit=500):
    """Fetches main chart data, prepares for analysis, and calculates SMAs."""
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.dropna()
    
    df['fast_sma'] = df['close'].rolling(window=9).mean()
    df['slow_sma'] = df['close'].rolling(window=20).mean()
    
    df = df.dropna() 
    
    if len(df) < 20: 
        return pd.DataFrame(), None
    
    ohlcv_daily = exchange.fetch_ohlcv(symbol, daily_timeframe, limit=20) 
    df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_daily.set_index('timestamp', inplace=True)
    
    cpr_levels = calculate_cpr_levels(df_daily)
    
    return df, cpr_levels

# 3. Trend and Signal Generation
def get_trend_and_signal(df, cpr_levels):
    """Determines trend via SMA crossover and incorporates ML prediction."""
    
    latest = df.iloc[-1]
    current_price = latest['close']
    fast_sma = latest['fast_sma']
    slow_sma = latest['slow_sma']
    
# --- ML Prediction Block ---
    ml_prediction = "NEUTRAL (No Model)"
    if ML_MODEL is not None and SCALER is not None:
        try:
            # 1. Calculate volatility robustly and safely
            
            # --- CRITICAL FIX: Ensure clean Volatility number ---
            # Use the last 20 close prices to calculate returns and standard deviation
            close_prices_recent = df['close'].iloc[-20:] 
            if len(close_prices_recent) < 20: # Not enough data to calculate a reliable 20-period StdDev
                 current_volatility = 0.0
            else:
                 returns = close_prices_recent.pct_change().dropna()
                 current_volatility = returns.std(skipna=True).fillna(0)
                 # Ensure we take the final scalar value if Pandas returns a Series
                 current_volatility = current_volatility.iloc[-1] if isinstance(current_volatility, pd.Series) and not current_volatility.empty else float(current_volatility)
                 current_volatility = 0.0 if np.isinf(current_volatility) or np.isnan(current_volatility) else current_volatility
            # --- End Volatility Fix ---

            # 2. Build the latest features DataFrame
            is_fast_over_slow = 1 if fast_sma > slow_sma else 0
            is_close_over_fast = 1 if current_price > fast_sma else 0
            
            latest_features = pd.DataFrame({
                'fast_over_slow': [is_fast_over_slow],
                'close_over_fast': [is_close_over_fast],
                'volatility': [current_volatility] 
            })
            
            # 3. Scaling and Prediction
            X_predict_scaled = SCALER.transform(latest_features)
            prediction = ML_MODEL.predict(X_predict_scaled)[0]
            probability = ML_MODEL.predict_proba(X_predict_scaled)[0]
            bullish_prob = probability[1]
            
            # 4. Final Prediction Output
            if prediction == 1 and bullish_prob > 0.55:
                ml_prediction = f"BULLISH ({bullish_prob*100:.0f}%)"
            elif prediction == 0 and probability[0] > 0.55:
                ml_prediction = f"BEARISH ({probability[0]*100:.0f}%)"
            else:
                 ml_prediction = "NEUTRAL (Low Conviction)"

        except Exception as e:
            # If any step in the prediction fails, log the error and skip
            print(f"❌ ML PREDICTION FAILED (Runtime Error): {e}")
            ml_prediction = "NEUTRAL (ML Error)"
            
    # --- End of ML Prediction Block ---

    # --- Trend Assessment (Used for confirmation and fallback) ---
    trend = "Neutral"
    if fast_sma > slow_sma:
        trend = "Uptrend"
        trend_emoji = "🟢"
    elif fast_sma < slow_sma:
        trend = "Downtrend"
        trend_emoji = "🔴"
    else:
        trend_emoji = "🟡"

    # --- Final Signal Generation ---
    pp = cpr_levels.get('PP', 'N/A')
    
    proximity_msg = ""
    if pp != 'N/A':
        distance_to_pp = current_price - pp
        if abs(distance_to_pp / pp) < 0.005: 
            proximity_msg = "Price is near the <b>Central Pivot Point (PP)</b>."
        elif distance_to_pp > 0:
            proximity_msg = f"Price is <b>Above PP</b> ({pp:.2f})."
        else:
            proximity_msg = f"Price is <b>Below PP</b> ({pp:.2f})."
            
    signal = "HOLD"
    signal_emoji = "🟡"
    if "BULLISH" in ml_prediction and current_price > pp:
        signal = "STRONG BUY"
        signal_emoji = "🚀"
    elif "BEARISH" in ml_prediction and current_price < pp:
        signal = "STRONG SELL"
        signal_emoji = "🔻"
        
    return trend, trend_emoji, proximity_msg, signal, signal_emoji, ml_prediction

# 4. ASYNC SCHEDULER FUNCTIONS

async def generate_and_send_signal(symbol):
    """Fetches data, runs analysis, and sends the Telegram message."""
    
    try:
        # Step 1: Fetch and Prepare Data
        df, cpr_levels = fetch_and_prepare_data(symbol, TIMEFRAME)
        
        if df.empty or cpr_levels is None:
            message = f"🚨 Data Fetch/Processing Error for {symbol}. Could not generate signal (Insufficient clean data)."
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            return

        # Step 2: Generate Analysis & Signal
        trend, trend_emoji, proximity_msg, signal, signal_emoji, ml_prediction = get_trend_and_signal(df, cpr_levels)
        current_price = df.iloc[-1]['close']
        
        cpr_text = (
            f"<b>Daily CPR Levels:</b>\n"
            f"  - <b>PP (Pivot Point):</b> <code>{cpr_levels['PP']:.2f}</code>\n"
            f"  - <b>R1/S1:</b> <code>{cpr_levels['R1']:.2f}</code> / <code>{cpr_levels['S1']:.2f}</code>\n"
            f"  - <b>R2/S2:</b> <code>{cpr_levels['R2']:.2f}</code> / <code>{cpr_levels['S2']:.2f}</code>\n"
        )
        
        # --- FINAL PROFESSIONAL HTML MESSAGE TEMPLATE ---
        
        message = (
            f"╔════════════════════════════════╗\n"
            f"  🧠 <b>AI MARKET INTELLIGENCE REPORT</b>\n"
            f"╚════════════════════════════════╝\n\n"
            
            f"** {symbol} | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} **\n\n"
            
            f"---🚨 <b>{signal_emoji} FINAL SIGNAL: {signal}</b> 🚨---\n\n"
            
            f"<b>💰 Current Price:</b> <code>{current_price:,.2f}</code>\n"
            f"<b>⏰ Timeframe:</b> {TIMEFRAME}\n"
            
            f"\n\n<b>🤖 ML PREDICTION</b>\n"
            f"<b>Forecast:</b> {ml_prediction}\n"
            f"<b>Confidence:</b> {ml_prediction.split(' ')[-1].replace(')', '').replace('(', '')}\n"
            
            f"\n\n<b>📊 TECHNICAL & KEY LEVELS</b>\n"
            f"{trend_emoji} <b>Trend (SMA 9/20):</b> {trend}\n"
            f"{proximity_msg.replace('**', '<b>').replace('**', '</b>')}\n\n"
            
            f"{cpr_text}\n"
            
            f"----------------------------------------\n"
            f"<i>Disclaimer: This analysis is for educational purposes only.</i>"
        )

        # Apply global HTML escaping
        message = message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Revert valid HTML tags back from their escaped form
        message = message.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        message = message.replace('&lt;code&gt;', '<code>').replace('&lt;/code&gt;', '</code>')
        message = message.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ Error generating signal for {symbol}: {e}")
        
        diagnostic_message = (
            f"❌ <b>FATAL ANALYSIS ERROR for {symbol}</b> ❌\n\n"
            f"<b>Time:</b> {datetime.now().strftime('%H:%M:%S UTC')}\n"
            f"<b>Issue:</b> The calculation thread crashed.\n\n"
            f"<b>Source Trace:</b>\n<code>{str(e)[:150]}</code>" 
        )
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=diagnostic_message, parse_mode='HTML')


async def start_scheduler_loop():
    """Sets up the scheduler and keeps the asyncio loop running."""
    
    # --- ML Training before starting the loop ---
    global ML_MODEL
    global SCALER

    print("\n⏳ Preparing and training Machine Learning Model...")
    try:
        ohlcv_train = exchange.fetch_ohlcv(CRYPTOS[0].strip(), TIMEFRAME, limit=600)
        df_train = pd.DataFrame(ohlcv_train, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_train['close'] = pd.to_numeric(df_train['close'])
        
        df_train['fast_sma'] = df_train['close'].rolling(window=9).mean()
        df_train['slow_sma'] = df_train['close'].rolling(window=20).mean()
        df_train = df_train.dropna()
        
        ML_MODEL, SCALER = train_prediction_model(df_train)
        
    except Exception as e:
        print(f"❌ ML Model Training Failed: {e}")
        ML_MODEL = None
        SCALER = None


    # --- Start the scheduler loop ---
    scheduler = AsyncIOScheduler()
    
    for symbol in [s.strip() for s in CRYPTOS]:
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[symbol]) 
    
    scheduler.start()
    print("🚀 Scheduler started successfully.")

    # Run initial analysis immediately after scheduler starts
    await generate_and_send_signal(CRYPTOS[0].strip()) 
    if len(CRYPTOS) > 1:
        await generate_and_send_signal(CRYPTOS[1].strip())

    # Keep the main thread running (Worker thread)
    while True:
        await asyncio.sleep(60)


# 5. CRITICAL STARTUP THREAD (Fixes the Gunicorn/Threading Conflict)

def start_asyncio_thread():
    """Target function for the background thread."""
    try:
        asyncio.run(start_scheduler_loop())
    except Exception as e:
        print(f"FATAL SCHEDULER ERROR: {e}")

# This thread starts immediately when Gunicorn loads the 'app' instance
scheduler_thread = threading.Thread(target=start_asyncio_thread, daemon=True)
scheduler_thread.start()

print("✅ Gunicorn loading Flask app. Scheduler thread initialized.")



