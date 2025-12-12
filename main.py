# main.py - The main execution file
import traceback
import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot
# from sklearn.linear_model import LogisticRegression # Not used for this version

# --- CONFIGURATION LOADING ---
from dotenv import load_dotenv 
load_dotenv() 

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# NOTE: Adjusted the default CRYPTOS format to a string for simpler splitting
CRYPTOS = os.getenv("CRYPTOS", "BTC/USDT,ETH/USDT").split(',') 
TIMEFRAME = os.getenv("TIMEFRAME", "4h") # Main chart timeframe (4-hour)
DAILY_TIMEFRAME = '1d' # For CPR calculation

# Initialize Bot and Exchange
bot = Bot(token=TELEGRAM_BOT_TOKEN)
exchange = ccxt.kucoin({
    'enableRateLimit': True,
    'rateLimit': 1000, # Respect exchange rate limits
})

# --- TRADING LOGIC IMPLEMENTATION ---

# 1. CPR Calculation Function (using high, low, close from previous Daily candle)
def calculate_cpr_levels(df_daily):
    """Calculates Daily Pivot Points (PP, TC, BC, R/S levels) from previous day's data."""
    # --- CRITICAL FIX: Check for sufficient data ---
    if df_daily.empty or len(df_daily) < 2:
        print("⚠️ CPR calculation failed: Not enough historical daily data (need at least 2 days).")
        return None # Return None instead of an empty dict to signal a definite failure

    # Get data from the *last completed* daily candle (index -2)
    prev_day = df_daily.iloc[-2]  
    
    H = prev_day['high']
    L = prev_day['low']
    C = prev_day['close']
    
    # CPR Components
    PP = (H + L + C) / 3.0
    BC = (H + L) / 2.0  # Bottom Central Pivot / Range
    TC = PP - BC + PP   # Top Central Pivot / Range
    
    # Resistance & Support Levels (R1/S1, R2/S2, R3/S3 remain the same)
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    R3 = H + 2 * (PP - L)
    S3 = L - 2 * (H - PP)
    
    # Calculate CPR Width for bias assessment (Optional but useful)
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
    
    # Calculate SMAs (9 and 20 periods, as per the script)
    df['fast_sma'] = df['close'].rolling(window=9).mean()
    df['slow_sma'] = df['close'].rolling(window=20).mean()
    
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
            proximity_msg = "Price is near the **Central Pivot Point (PP)**."
        elif distance_to_pp > 0:
            proximity_msg = f"Price is **Above PP** ({pp:.2f})."
        else:
            proximity_msg = f"Price is **Below PP** ({pp:.2f})."
            
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

# 4. Orchestration and Sending Message
async def generate_and_send_signal(symbol):
    """Fetches data, runs analysis, and sends the Telegram message."""
    print(f"Generating signal for {symbol}...")
    
    try:
        # Step 1: Fetch and Prepare Data
        df, cpr_levels = fetch_and_prepare_data(symbol, TIMEFRAME)
        if df.empty or not cpr_levels:
            message = f"🚨 Data Fetch Error for {symbol}. Could not generate signal."
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            return

        # Step 2: Generate Analysis & Signal
        trend, trend_emoji, proximity_msg, signal, signal_emoji = get_trend_and_signal(df, cpr_levels)
        current_price = df.iloc[-1]['close']
        
        # --- CRITICAL FIX: Switch all Markdown to HTML tags ---
        # <b> for **bold** and <i> for *italic* are supported HTML tags.
        
        # Format the CPR levels with a focus on PP and R1/S1
        cpr_text = (
            f"<b>Daily CPR Levels:</b>\n"
            f"  - <b>PP (Pivot Point):</b> <code>{cpr_levels['PP']:.2f}</code>\n"
            f"  - <b>R1/S1:</b> <code>{cpr_levels['R1']:.2f}</code> / <code>{cpr_levels['S1']:.2f}</code>\n"
            f"  - <b>R2/S2:</b> <code>{cpr_levels['R2']:.2f}</code> / <code>{cpr_levels['S2']:.2f}</code>\n"
        )
        
        # Ensure all price/level values are wrapped in <code> for safe parsing
        message = (
            f"<b>📈 {symbol} Market Analysis ({TIMEFRAME} Chart)</b>\n"
            f"---🚨 <b>{signal_emoji} AI SIGNAL: {signal}</b> 🚨---\n\n"
            f"💰 <b>Current Price:</b> <code>{current_price:.2f}</code>\n"
            f"{trend_emoji} <b>Trend Analysis (SMA 9/20):</b> {trend}\n\n"
            f"📊 <b>Key Levels Summary</b>\n"
            f"{proximity_msg.replace('**', '<b>').replace('**', '</b>')}\n"
            f"{cpr_text}"
            f"\n<i>Analysis based on Daily CPR and {TIMEFRAME} SMA Crossover.</i>" # Changed *italic* to <i>italic</i>
        )

        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        print(f"Signal for {symbol} sent successfully!")

    except Exception as e:
        print(f"Error for {symbol}: {e}")
        # Send a simplified, unformatted message to avoid further parsing errors
        error_message = f"⚠️ Critical Error generating signal for {symbol}. The background analysis thread failed. Please check logs."
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=error_message)

# The rest of the code (start_bot_scheduler, if __name__ == '__main__') remains the same.

# 5. Scheduler Setup
async def start_bot_scheduler():
    """Sets up the scheduler to run the signal function automatically every 30 minutes."""
    scheduler = AsyncIOScheduler()
    
    # Runs the job for each crypto every 30 minutes (:00 and :30)
    for symbol in [s.strip() for s in CRYPTOS]:
        # '*/30' for minute means it runs every 30 minutes
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[symbol]) 
    
    scheduler.start()
    print(f"Scheduler started for {CRYPTOS}. Running every 30 minutes. Press Ctrl+C to exit.")
    
    # Keep the main thread running
    while True:
        await asyncio.sleep(60) 

if __name__ == '__main__':
    # Initial run check (optional, but good for debugging)
    print("Starting bot...")

    asyncio.run(start_bot_scheduler())





