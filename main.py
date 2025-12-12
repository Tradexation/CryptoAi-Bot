# main.py - The final working structure for Render Web Service

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
import traceback # Required for error diagnostics

# --- CONFIGURATION LOADING ---
from dotenv import load_dotenv 
load_dotenv() 

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CRYPTOS = os.getenv("CRYPTOS", "BTC/USDT,ETH/USDT").split(',') 
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
DAILY_TIMEFRAME = '1d' 
ANALYSIS_INTERVAL = 30 # Set to 30 minutes

# Initialize Bot and Exchange
bot = Bot(token=TELEGRAM_BOT_TOKEN)
# Switched to KuCoin for stable API access
exchange = ccxt.kucoin({
    'enableRateLimit': True,
    'rateLimit': 1000, 
})

# ========== FLASK WEB SERVER & STATUS TRACKING ==========
app = Flask(__name__) # Gunicorn loads this 'app' instance

bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "monitored_assets": CRYPTOS,
    "uptime_start": datetime.now().isoformat()
}

@app.route('/')
def home():
    # Placeholder for a simple homepage
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

# ========== TECHNICAL ANALYSIS FUNCTIONS (Defined before scheduler calls them) ==========

# 1. CPR Calculation Function
def calculate_cpr_levels(df_daily):
    """Calculates Daily Pivot Points (PP, TC, BC, R/S levels) from previous day's data."""
    # --- FIX: Check for sufficient data ---
    if df_daily.empty or len(df_daily) < 2:
        print("⚠️ CPR calculation failed: Not enough historical daily data (need at least 2 days).")
        return None

    # Get data from the *last completed* daily candle (index -2)
    prev_day = df_daily.iloc[-2]  
    
    H = prev_day['high']
    L = prev_day['low']
    C = prev_day['close']
    
    # CPR Components
    PP = (H + L + C) / 3.0
    BC = (H + L) / 2.0
    TC = PP - BC + PP
    
    # Resistance & Support Levels
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    R3 = H + 2 * (PP - L)
    S3 = L - 2 * (H - PP)
    
    cpr_width = abs(TC - BC)

    return {
        'PP': PP, 'TC': TC, 'BC': BC, 'R1': R1, 'S1': S1,
        'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3, 'CPR_Width': cpr_width
    }

# 2. Data Fetching and Preparation Function
def fetch_and_prepare_data(symbol, timeframe, daily_timeframe='1d', limit=100):
    """Fetches main chart data and daily data for CPR, and calculates SMAs."""
    
    # Fetch main chart data (e.g., 4h data)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # --- FIX: Drop NaNs from the main DataFrame before and after SMA calculation ---
    df = df.dropna()
    
    # Calculate SMAs (9 and 20 periods)
    df['fast_sma'] = df['close'].rolling(window=9).mean()
    df['slow_sma'] = df['close'].rolling(window=20).mean()
    
    # Drop NaNs again after calculating SMAs (the first 20 rows will have NaNs)
    df = df.dropna() 
    
    # Check if we have enough data left for analysis
    if len(df) < 20: # Ensure we have enough clean rows to analyze the trend
        return pd.DataFrame(), None
    
    # Fetch Daily data for CPR calculation
    ohlcv_daily = exchange.fetch_ohlcv(symbol, daily_timeframe, limit=limit)
    df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_daily.set_index('timestamp', inplace=True)
    
    # Calculate CPR levels
    cpr_levels = calculate_cpr_levels(df_daily)
    
    return df, cpr_levels

# 3. Trend and Signal Generation
def get_trend_and_signal(df, cpr_levels):
    """Determines trend via SMA crossover and checks price vs CPR levels."""
    
    # Get the latest data point
    latest = df.iloc[-1]
    current_price = latest['close']
    fast_sma = latest['fast_sma']
    slow_sma = latest['slow_sma']
    
    # Check for SMA Crossover (Trend)
    trend = "Neutral"
    if fast_sma > slow_sma:
        trend = "Uptrend (Fast SMA > Slow SMA)"
        trend_emoji = "🟢"
    elif fast_sma < slow_sma:
        trend = "Downtrend (Fast SMA < Slow SMA)"
        trend_emoji = "🔴"
    else:
        trend_emoji = "🟡"

    # Check for proximity to the Central Pivot Point (PP)
    pp = cpr_levels.get('PP', 'N/A')
    
    proximity_msg = ""
    if pp != 'N/A':
        distance_to_pp = current_price - pp
        if abs(distance_to_pp / pp) < 0.005: # Price is within 0.5% of PP
            proximity_msg = "Price is near the <b>Central Pivot Point (PP)</b>."
        elif distance_to_pp > 0:
            proximity_msg = f"Price is <b>Above PP</b> ({pp:.2f})."
        else:
            proximity_msg = f"Price is <b>Below PP</b> ({pp:.2f})."
            
    # Simple Signal: Buy if Uptrend and Above PP, Sell if Downtrend and Below PP
    signal = "HOLD"
    signal_emoji = "🟡"
    if trend == "Uptrend (Fast SMA > Slow SMA)" and current_price > pp:
        signal = "STRONG BUY"
        signal_emoji = "🚀"
    elif trend == "Downtrend (Fast SMA < Slow SMA)" and current_price < pp:
        signal = "STRONG SELL"
        signal_emoji = "🔻"
        
    return trend, trend_emoji, proximity_msg, signal, signal_emoji


# 4. ASYNC SCHEDULER FUNCTIONS

async def generate_and_send_signal(symbol):
    """Fetches data, runs analysis, and sends the Telegram message."""
    print(f"Generating signal for {symbol}...")
    
    try:
        # Step 1: Fetch and Prepare Data
        df, cpr_levels = fetch_and_prepare_data(symbol, TIMEFRAME)
        
        # --- Handle insufficient data gracefully ---
        if df.empty or cpr_levels is None:
            message = f"🚨 Data Fetch/Processing Error for {symbol}. Could not generate signal (Insufficient clean data)."
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            return

        # Step 2: Generate Analysis & Signal
        trend, trend_emoji, proximity_msg, signal, signal_emoji = get_trend_and_signal(df, cpr_levels)
        current_price = df.iloc[-1]['close']
        
        # Step 3: Format Professional Message (Using HTML to fix parsing errors)
        
        cpr_text = (
            f"<b>Daily CPR Levels:</b>\n"
            f"  - <b>PP (Pivot Point):</b> <code>{cpr_levels['PP']:.2f}</code>\n"
            f"  - <b>R1/S1:</b> <code>{cpr_levels['R1']:.2f}</code> / <code>{cpr_levels['S1']:.2f}</code>\n"
            f"  - <b>R2/S2:</b> <code>{cpr_levels['R2']:.2f}</code> / <code>{cpr_levels['S2']:.2f}</code>\n"
        )
        
        # --- FINAL FIX: Construct the message, then apply HTML escaping ---
        message = (
            f"<b>📈 {symbol} Market Analysis ({TIMEFRAME} Chart)</b>\n"
            f"---🚨 <b>{signal_emoji} AI SIGNAL: {signal}</b> 🚨---\n\n"
            f"💰 <b>Current Price:</b> <code>{current_price:.2f}</code>\n"
            f"{trend_emoji} <b>Trend Analysis (SMA 9/20):</b> {trend}\n\n"
            f"📊 <b>Key Levels Summary</b>\n"
            f"{proximity_msg.replace('**', '<b>').replace('**', '</b>')}\n"
            f"{cpr_text}"
            f"\n<i>Analysis based on Daily CPR and {TIMEFRAME} SMA Crossover.</i>"
        )

        # Apply global HTML escaping to prevent the 'unsupported start tag' error
        # Replace <, >, and & with their HTML entities, but skip valid tags
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
        # --- FIX: Diagnostic error logging ---
        error_trace = traceback.format_exc()
        print(f"❌ Error generating signal for {symbol}: {e}")
        print(error_trace) 

        # Send a simplified, HTML-formatted diagnostic message to the channel
        diagnostic_message = (
            f"❌ <b>FATAL ANALYSIS ERROR for {symbol}</b> ❌\n\n"
            f"<b>Time:</b> {datetime.now().strftime('%H:%M:%S UTC')}\n"
            f"<b>Issue:</b> The calculation thread crashed.\n\n"
            f"<b>Source Trace:</b>\n<code>{str(e)[:150]}</code>" 
        )
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=diagnostic_message, parse_mode='HTML')


async def start_scheduler_loop():
    """Sets up the scheduler and keeps the asyncio loop running."""
    scheduler = AsyncIOScheduler()
    
    # Set up the job for each crypto
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

# This thread starts immediately when Gunicorn loads the 'app' instance, 
# running the scheduler in the background while Gunicorn handles the web server.
scheduler_thread = threading.Thread(target=start_asyncio_thread, daemon=True)
scheduler_thread.start()

print("✅ Gunicorn loading Flask app. Scheduler thread initialized.")
# The Web Service is now running Flask (via Gunicorn) AND the scheduler (via Thread)
