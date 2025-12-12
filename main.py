# main.py - The final streamlined execution file

import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot
from flask import Flask, jsonify, render_template_string
import threading # Still needed for the main event loop
import time

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

# (Keep your HTML_TEMPLATE and /health, /status routes here, unchanged)
# ... (Placeholder for the unchanged Flask routes and template) ...

@app.route('/')
def home():
    return render_template_string("<h1>Bot Status Page</h1>" + 
                                 f"<p>Status: {bot_stats['status']}</p>" + 
                                 f"<p>Analyses: {bot_stats['total_analyses']}</p>")

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

@app.route('/status')
def status():
    return jsonify(bot_stats), 200

# ========== ANALYSIS FUNCTIONS (Unchanged logic) ==========
# (Keep your calculate_cpr_levels, fetch_and_prepare_data, get_trend_and_signal functions here, unchanged)
# ...

# 4. Orchestration and Sending Message (Finalized with HTML mode)
async def generate_and_send_signal(symbol):
    """Fetches data, runs analysis, and sends the Telegram message."""
    # (Keep your FINAL, HTML-formatted message logic here, unchanged)
    
    # NOTE: Ensure the parse_mode is 'HTML' and the message uses <b> and <code> tags.
    try:
        df, cpr_levels = fetch_and_prepare_data(symbol, TIMEFRAME)
        if df.empty or cpr_levels is None:
            message = f"🚨 Data Fetch/Processing Error for {symbol}."
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            return

        # ... (analysis logic runs here) ...
        current_price = df.iloc[-1]['close']
        
        # ... (HTML message construction remains here) ...
        
        message = f"<b>📈 {symbol} Market Analysis</b>\n..." # Example HTML
        
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"

    except Exception as e:
        print(f"❌ Error generating signal for {symbol}: {e}")
        bot_stats['status'] = f"error: {str(e)[:50]}"

# 5. Scheduler Setup
async def start_scheduler_loop():
    """Sets up the scheduler and keeps the asyncio loop running."""
    scheduler = AsyncIOScheduler()
    
    for symbol in [s.strip() for s in CRYPTOS]:
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[symbol]) 
    
    scheduler.start()
    bot_stats['status'] = "operational"
    print("🚀 Scheduler started successfully.")

    # Keep the main asyncio loop running
    # The Gunicorn worker process will manage the web server thread
    while True:
        await asyncio.sleep(60)

# ==========================================================
# ======= CRITICAL STARTUP FIX: Scheduler in a Thread ========
# ==========================================================
# Gunicorn loads the 'app' object and immediately executes code outside of functions.

def start_asyncio_thread():
    """Target function for the background thread."""
    try:
        asyncio.run(start_scheduler_loop())
    except Exception as e:
        print(f"FATAL SCHEDULER ERROR: {e}")

# Check if we are running under Gunicorn (i.e., not '__main__')
# This thread is started immediately upon Gunicorn loading main:app.
scheduler_thread = threading.Thread(target=start_asyncio_thread, daemon=True)
scheduler_thread.start()

print("✅ Gunicorn loading Flask app. Scheduler thread initialized.")
# The Web Service is now running Flask (via Gunicorn) AND the scheduler (via Thread)
