"""
- Pull last 5 trading days of 15-min bars from yfinance (to allow z-normalization),
then restrict decisions to the last 2 days of regular trading hours using Alpaca market calendar like in OHLCV_retriever.py
- Compute TA features with the same generate_features() used in processing
- Build 64-D embeddings using the trained MLP (best_*_model.pt)
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import threading
import numpy as np
from math import floor

import numpy
import pandas as pd
import yaml
import pytz
import requests
import importlib.util
import pickle
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.historical import CryptoHistoricalDataClient
from collections import defaultdict

import torch
from torch import nn
from collections import deque

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")
CONF_DIR = os.path.join(BASE_DIR, "../../conf")
MODEL_PATH = os.path.join(BASE_DIR, "../../models/07_01_feed_forward_checkpoint.pt")

SPLIT_DATA_DIR = os.path.join(BASE_DIR, "../../data/split_data")
IMAGE_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
IMAGE_DIR_ORG = os.path.join(BASE_DIR, "../../images")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR_ORG, exist_ok=True)
DATASET_FILE = os.path.join(LARGE_DATA_DIR, "03_crypto_ml_dataset.parquet")
POST_SPLIT_DIR = os.path.join(BASE_DIR, "../../data/post_split_data")
os.makedirs(POST_SPLIT_DIR, exist_ok=True)

FEATURES_PY_PATH = os.path.join(BASE_DIR, "../../scripts/03_pre_split_prep/features.py")
BUFFER_DIR = "./buffers"
os.makedirs(BUFFER_DIR, exist_ok=True)
FEATURE_COLUMNS_FILE = os.path.join(POST_SPLIT_DIR, "feature_columns.pt")

# create feature_model (when module is not standard, need tto load from another file path)
spec = importlib.util.spec_from_file_location("features_model", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
    spec.loader.exec_module(features_module) 
else:
    raise RuntimeError(f"Could not load features.py from {FEATURES_PY_PATH}")

generate_features = getattr(features_module, "compute_features")

# Load params and keys 
with open(os.path.join(CONF_DIR, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)
with open(os.path.join(CONF_DIR, "keys.yaml"), "r") as f:
    keys = yaml.safe_load(f)

# Alpaca keys (Paper by default)
ALPACA_KEY_ID = keys["APCA_KEYS"]["APCA_API_KEY_ID"]
ALPACA_SECRET = keys["APCA_KEYS"]["APCA_API_SECRET_KEY"]

ALPACA_BASE = "https://paper-api.alpaca.markets"

client = CryptoHistoricalDataClient(api_key=ALPACA_KEY_ID, secret_key=ALPACA_SECRET)

# SYMBOL = "BTC/USDT"
USDT_PAIRS = ["BTC/USDT", "ETH/USDT", "LTC/USDT", "SOL/USDT"]
BTC_PAIRS = ["ETH/BTC", "LTC/BTC", "SOL/BTC"]
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
PAIR_MAP = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "LTC/USDT": "LTCUSDT",
    "SOL/USDT": "SOLUSDT",

    "ETH/BTC": "ETHBTC",
    "LTC/BTC": "LTCBTC",
    "SOL/BTC": "SOLBTC",
}

# LOOKBACK_BARS = {
#     "15m": 200,   # 2 days
#     "1h": 200,    # ~8 days
#     "4h": 200,    # ~33 days
#     "1d": 200     # ~200 days
# }
SLEEP_SECONDS = 60 * 15
TF_MAP = {
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame.Hour,
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame.Day,
}

MAX_POSITION_USD = 5000
ENTRY_THRESHOLD = 0.002
EXIT_THRESHOLD = 0.0005

# Initialize buffer
buffers = defaultdict(pd.DataFrame)

MIN_BARS_REQUIRED = {
    "15m": 100,
    "1h": 120,
    "4h": 120,
    "1d": 120
}
MAX_BUFFER_LEN = 600

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model class
class CryptoMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Deeper network with layer norm, batch norm and dropout for multi-task learning
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Normalize input features
        x = self.input_norm(x)
        return self.net(x)
    
# Load model 
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# debug to see model's key
state_dict = checkpoint["model_state_dict"]
for key in state_dict.keys():
    print(f"- {key}")

    # Print from the checkpoint itself
print("Checkpoint input dim:", checkpoint["model_state_dict"]["net.0.weight"].shape)
print("Checkpoint output dim:", checkpoint["model_state_dict"]["net.12.weight"].shape)

model = CryptoMLP(
    input_dim=checkpoint["model_state_dict"]["net.0.weight"].shape[1],
    output_dim=checkpoint["model_state_dict"]["net.12.weight"].shape[0],
)

print("Model input_dim:", model.net[0].weight.shape[1])
print("Model output_dim:", model.net[12].weight.shape[0])

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

EASTERN = pytz.timezone("US/Eastern")

def bootstrap_buffers(symbols, tf_map, lookback=10000):
    buffers = {}
    now = datetime.now(timezone.utc)
    tf_15m = tf_map["15m"]
    
    for sym in symbols:
        print(f"\n--- DEBUG: Starting {sym} ---")
        req = CryptoBarsRequest(
            symbol_or_symbols=[sym],
            timeframe=tf_15m,
            start=now - timedelta(days=60),
            end = now, 
            limit=lookback
        )

        raw_df = client.get_crypto_bars(req).df
        
        # 1. Debug Raw DataFrame
        print(f"[DEBUG] Raw DF shape: {raw_df.shape}")
        print(f"[DEBUG] Raw DF Index type: {type(raw_df.index)}")
        
        if isinstance(raw_df.index, pd.MultiIndex):
            print(f"[DEBUG] MultiIndex detected. Levels: {raw_df.index.names}")
            try:
                df_15m = raw_df.xs(sym, level='symbol').copy()
                print(f"[DEBUG] .xs selection successful. New shape: {df_15m.shape}")
            except KeyError:
                print(f"[ERROR] Symbol {sym} not found in MultiIndex. Available: {raw_df.index.get_level_values('symbol').unique()}")
                continue
        else:
            df_15m = raw_df.copy()

        # 2. Debug Columns
        print(f"[DEBUG] Columns available: {df_15m.columns.tolist()}")

        # 3. Debug Index Conversion
        df_15m.index = pd.to_datetime(df_15m.index, utc=True).tz_convert(None)
        print(f"[DEBUG] Index converted. First 2 timestamps: {df_15m.index[:2].tolist()}")

        # Store 15m
        buffers[(sym, "15m")] = df_15m.copy()

        # 4. Debug Resampling
        resample_rules = {"1h": "1H", "4h": "4H", "1d": "1D"}
        for tf_str, rule in resample_rules.items():
            try:
                resampled = df_15m.resample(rule).agg({
                    "open": "first", "high": "max", "low": "min", 
                    "close": "last", "volume": "sum", "trade_count": "sum", "vwap": "mean"
                }).dropna()
                
                print(f"[DEBUG] {tf_str} resample done. Rows: {len(resampled)}")
                buffers[(sym, tf_str)] = resampled
            except Exception as e:
                print(f"[ERROR] Failed resampling {tf_str}: {e}")

    return buffers


# header
def alpaca_headers() -> Dict[str, str]:
    if not ALPACA_KEY_ID or not ALPACA_SECRET:
        raise RuntimeError("Alpaca API keys are missing!")
    return {
        "APCA-API-KEY-ID": ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

# position
def get_position_qty(symbol: str = "BTC/USD") -> float:
    url = f"{ALPACA_BASE}/v2/positions"
    try:
        r = requests.get(url, headers=alpaca_headers(), timeout=30)
        if r.status_code == 404:
            return 0.0
        r.raise_for_status()
        
        positions = r.json()
        
        # Normalize the symbol for comparison (e.g., "BTC/USD" -> "BTCUSD")
        search_symbol = symbol.replace("/", "")
        
        for pos in positions:
            # Normalize the returned symbol too, just in case
            pos_symbol = pos["symbol"].replace("/", "")
            if pos_symbol == search_symbol:
                return float(pos["qty"])
                
        return 0.0
    except Exception as e:
        print(f"[ERROR] Could not fetch positions: {e}")
        return 0.0

def submit_market_order(symbol: str, side: str, notional: float = None, qty: float = None) -> dict | None:
    side = side.lower()
    url = f"{ALPACA_BASE}/v2/orders"
    
    # Payload logic: Use notional if provided, otherwise use qty
    payload = {
        "symbol": symbol,
        "side": side,
        "type": "market",
        "time_in_force": "gtc",
    }
    
    if notional:
        # payload["notional"] = str(notional)
        # Alpaca REQUIREMENT: Notional must be max 2 decimal places
        payload["notional"] = str(round(float(notional), 2))
    elif qty:
        # payload["qty"] = str(qty)
        # Alpaca REQUIREMENT: Qty can be more precise, but 8-9 is safe for Crypto
        payload["qty"] = f"{float(qty):.8f}"
    else:
        raise ValueError("Must provide either 'notional' or 'qty'")

    try:
        r = requests.post(url, headers=alpaca_headers(), json=payload, timeout=30)

        if r.status_code != 200:
            print(f"[DEBUG] Status: {r.status_code}")
            print(f"[DEBUG] Response Body: {r.text}")

        r.raise_for_status()
        od = r.json()
        amount = f"${notional}" if notional else f"{qty} units"
        print(f"[order] {side.upper()} {amount} of {symbol} submitted.")
        return od
    except Exception as e:
        print(f"[ERROR] Order failed: {e}")
        return None

def generate_multi_asset_multi_tf_features(
    bars: dict[tuple[str, str], pd.DataFrame],
    base_timeframe: str = "15m"
) -> pd.DataFrame:
    """
    Combines multi-timeframe data into a single feature matrix, 
    aligned to the most recent 'base_timeframe' index.
    """
    feature_blocks = []
    base_index = None

    # --- NEW: SYNTHETIC PAIR CALCULATION ---
    # We create ETH/BTC, LTC/BTC, and SOL/BTC manually from USDT data
    for tf in TIMEFRAMES:
        btc_key = ("BTC/USDT", tf)
        if btc_key not in buffers:
            continue

        base_data = buffers.get(("SOL/USDT", tf))
        btc_data  = buffers.get(btc_key)

        if base_data is None or btc_data is None:
            continue

        # Align timestamps perfectly
        temp = pd.DataFrame(index=base_data.index)

        # 1. Calculate the raw ratio prices
        temp["open"]  = base_data["open"]  / btc_data["open"]
        temp["high"]  = base_data["high"]  / btc_data["high"]
        temp["low"]   = base_data["low"]   / btc_data["low"]
        temp["close"] = base_data["close"] / btc_data["close"]

        # 2. Handle Volume: Use a tiny floor (1e-9) to avoid division by zero in cumulative sums
        temp["volume"] = base_data["volume"].replace(0, 1e-9)

        # 3. Calculate Typical Price
        tp = (temp["high"] + temp["low"] + temp["close"]) / 3

        # 4. Calculate Synthetic VWAP
        # We use the 'typical price' of the ratio and the volume of the base asset (SOL)
        cum_pv = (tp * temp["volume"]).cumsum()
        cum_vol = temp["volume"].cumsum()
        temp["vwap"] = cum_pv / cum_vol

        # 5. Clean up: If volume was truly 0, the trade_count should be 0
        temp["trade_count"] = base_data.get("trade_count", 0)

        bars[("SOL/BTC", tf)] = temp.ffill().dropna()

    # 1. Sort keys: Base timeframe first, then by symbol
    # This ensures our 'base_index' is defined before we try to align 1h or 4h data
    sorted_keys = sorted(
        bars.keys(),
        key=lambda x: (0 if x[1] == base_timeframe else 1, x[0])
    )

    print(f"[INFO] Processing {len(sorted_keys)} data blocks...")

    for (symbol, tf) in sorted_keys:
        df = bars.get((symbol, tf), pd.DataFrame())
        
        # --- ROBUSTNESS CHECK ---
        # Skip empty dataframes or symbols with no data (like your SOL/BTC issue)
        if df.empty or len(df) < 2:
            print(f"[SKIP] {symbol} @ {tf}: No data available.")
            continue

        pair_name = PAIR_MAP.get(symbol, symbol.replace("/", ""))
        
        try:
            # Generate indicators (RSI, EMA, etc.)
            f = generate_features(df, pair_name, tf)
            
            if f.empty:
                print(f"[WARN] {symbol} @ {tf}: Indicators returned empty.")
                continue
                
        except Exception as e:
            print(f"[ERROR] {symbol} @ {tf}: Feature generation failed: {e}")
            continue

        # 2. Establish Base Index (The "Master Timeline")
        # We align everything to the 15m timestamps of the primary asset
        if tf == base_timeframe and base_index is None:
            base_index = f.index
            print(f"[INFO] Master timeline set from {symbol} {tf} (Length: {len(base_index)})")

        # 3. Alignment Logic
        if tf != base_timeframe:
            if base_index is None:
                print(f"[WARN] {symbol} @ {tf}: Skipping - no base index established yet.")
                continue
            
            # Reindex to 15m and 'forward fill' higher TF values
            # Example: A 1h RSI value stays the same for four 15m candles
            f = f.reindex(base_index, method="ffill")

        feature_blocks.append(f)

    # 4. Concatenation and Cleanup
    if not feature_blocks:
        print("[CRITICAL] No feature blocks succeeded. Check data sources.")
        return pd.DataFrame()

    # Join all columns side-by-side
    X = pd.concat(feature_blocks, axis=1)

    # Clean up any gaps (e.g., if a specific pair started trading later)
    # 1. Fill forward (keep the last known value)
    # 2. Fill 0 for any data that never existed at all
    X = X.ffill().fillna(0.0)

    # Keep only the very last row for live prediction
    # Or keep everything if this is for training
    print(f"[SUCCESS] Feature matrix ready. Shape: {X.shape}")
    
    return X

def save_buffers(buffers, path="./buffers", symbol_to_save = None):
    os.makedirs(path, exist_ok=True)

    for (sym, tf), df in buffers.items():
        # If symbol_to_save is provided, skip all other symbols
        if symbol_to_save and sym != symbol_to_save:
            continue
        
        fname = f"{sym.replace('/', '_')}__{tf}.parquet"
        df.to_parquet(os.path.join(path, fname))

# Load buffers
def load_buffers(path="./buffers"):
    buffers = {}

    for f in os.listdir(path):
        sym, tf = f.replace(".parquet", "").split("__")
        sym = sym.replace("_", "/")
        df = pd.read_parquet(os.path.join(path, f))
        buffers[(sym, tf)] = df

    return buffers

def update_all_timeframes(buffers, symbol, tf_map, max_15m_len=5000):

    #  Get existing buffer first
    buf_15m = buffers.get((symbol, "15m"), pd.DataFrame())

    # request recent candle
    now = datetime.now(timezone.utc)

    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=tf_map["15m"],
        start=now - timedelta(hours=2),
        end=now,
        limit=20
    )
    resp = client.get_crypto_bars(req)

    # Check if Alpaca returned anything at all
    if resp.df.empty:
        print(f"[WARN] No data returned for {symbol}.")
        return
    
    raw_df = resp.df
    if isinstance(raw_df.index, pd.MultiIndex):
        df_new = raw_df.xs(symbol, level='symbol').copy()
    else:
        df_new = raw_df.copy()

    df_new.index = pd.to_datetime(df_new.index, utc=True).tz_convert(None)
    df_new = df_new.sort_index()
    
    # If the existing buffer is empty, just use the new data
    if buf_15m.empty:
        buffers[(symbol, "15m")] = df_new.tail(max_15m_len)
        print(f"[INIT] {symbol} buffer initialized with {len(df_new)} rows.")
        return
        
    last_ts = buf_15m.index[-1]
    latest_api_ts = df_new.index[-1]

    # Guard against stale API responses
    if latest_api_ts <= last_ts:
        print(f"[NO NEW CANDLE] {symbol} | last={last_ts}")
        return

    #  Append only closed candles
    new_rows = df_new[df_new.index > last_ts]

    if new_rows.empty:
        print(f"[NO CLOSED CANDLE] {symbol}")
        return
    
    # Append new candles
    updated_15m = pd.concat([buf_15m, new_rows]).tail(max_15m_len)
    buffers[(symbol, "15m")] = updated_15m

    print(
        f"[UPDATE] {symbol} +{len(new_rows)} candle(s) | "
        f"Last={updated_15m.index[-1]}"
    )
    
    # Re-resample higher timeframes
    resample_rules = {"1h": "1H", "4h": "4H", "1d": "1D"}
    for tf_str, rule in resample_rules.items():
        buffers[(symbol, tf_str)] = updated_15m.resample(rule).agg({
            "open": "first", "high": "max", "low": "min", 
            "close": "last", "volume": "sum", "trade_count": "sum", "vwap": "mean"
        }).dropna()
            
def execute_trade(symbol: str, side: str, qty: float):
    try:
        print(f"[TRADE] Executing {side} order for {qty:.6f} units of {symbol}")
        submit_market_order(symbol, side, qty)
        print(f"[TRADE] {side} order for {symbol} completed")
    except Exception as e:
        print(f"[TRADE ERROR] {e}")
        # Optional: stop main loop by raising
        raise


# =========================
# PRE-LOOP (only run once to create buffer)
# =========================
# print("TRAIN_END:", TRAIN_END)
# print("NOW:", datetime.now(timezone.utc))

# if not os.path.exists(BUFFER_DIR) or len(os.listdir(BUFFER_DIR)) == 0:
#     print("[INIT] No buffers found. Bootstrapping historical data...")

#     buffers = bootstrap_buffers(
#         symbols=USDT_PAIRS + BTC_PAIRS,
#         tf_map=TF_MAP
#     )

#     save_buffers(buffers, BUFFER_DIR)

#     print("[INIT] Bootstrap completed and buffers saved.")
# else:
#     print("[INIT] Buffers already exist. Skipping bootstrap.")

# =========================
# MAIN LOOP
# =========================

buffers = load_buffers("./buffers")
symbols = USDT_PAIRS + BTC_PAIRS
feature_columns = torch.load(FEATURE_COLUMNS_FILE)
print(f"[INFO] Loaded {len(feature_columns)} feature columns for live prediction")
# --- Configuration ---
RISK_PER_TRADE_PCT = 0.01  # Risk 1% of available cash per trade
STOP_LOSS_PCT = 0.02       # 2% Stop Loss
entry_price = 0.0          # Track this to calculate stop loss
pred_history = deque(maxlen=100) # keep the last 100 predictions

# Set eval mode
model.eval()

print("LIVE TRADING STARTED")

while True:
    try:
        # 1. Update buffers (Resampling logic)
        for sym in symbols: 
            try:
                update_all_timeframes(buffers, sym, TF_MAP)
                
                # Check memory count
                current_len = len(buffers.get((sym, "15m"), []))
                print(f"Memory Buffer for {sym} is now {current_len} rows.")

                # SAVE the new data to the .parquet files immediately
                save_buffers(buffers, BUFFER_DIR, symbol_to_save=sym)
                
                print(f"[DISK] Updated parquet files for {sym}")

            except Exception as e:
                print(f"[WARN] Skipping {sym} update: {e}")

        # --- FIX: Filter out empty symbols ---
        active_symbols = []
        for sym in symbols:
            # Check if the 15m buffer exists and has data
            if (sym, "15m") in buffers and not buffers[(sym, "15m")].empty:
                active_symbols.append(sym)
            else:
                print(f"[INFO] Removing {sym} from this iteration (no data)")

        if not active_symbols:
            print("[ERROR] No symbols have data. Sleeping...")
            time.sleep(60)
            continue

        # 2. Generate and Align Features
        X_all = generate_multi_asset_multi_tf_features(buffers, base_timeframe="15m")
        X_all = X_all.reindex(columns=feature_columns, fill_value=0.0)
        # X_hist_full = generate_multi_asset_multi_tf_features(buffers, base_timeframe="15m") --> X_all

        # We only need enough to fill our baseline (e.g., last 25-50 samples)
        warmup_samples = min(len(X_all), 50) 
        X_warmup = X_all.iloc[-warmup_samples:]

        # Keep track of the timestamps for our tuples
        timestamps = X_warmup.index

        if X_all.empty:
            print("[WARN] Feature matrix empty, skipping...")
            time.sleep(SLEEP_SECONDS)
            continue

        # X_live = X_all.iloc[-1:].values # Keep as 2D for model
        
        # 3. Model Prediction
        # Align columns to match what the model expects
        X_warmup_aligned = X_warmup.reindex(columns=feature_columns, fill_value=0.0) 

        with torch.no_grad():
            inputs = torch.tensor(X_warmup_aligned.values, dtype=torch.float32).to(DEVICE)
            preds = model(inputs)
        

        # with torch.no_grad():
        #     pred = model(torch.tensor(X_live, dtype=torch.float32).to(DEVICE))
        
        # Extract BTC 4h prediction
        target_order = ["BTCUSDT_target_logret_4h", "BTCUSDT_target_logret_1d", "ETHUSDT_target_logret_4h", "ETHUSDT_target_logret_1d", "LTCUSDT_target_logret_4h", "LTCUSDT_target_logret_1d", "SOLUSDT_target_logret_4h", "SOLUSDT_target_logret_1d"]
        pred_4h = preds[0, target_order.index("BTCUSDT_target_logret_4h")].item()
        # pred_history.append(pred_4h)

        # Identify the correct output index for BTC 4h
        btc_idx = target_order.index("BTCUSDT_target_logret_4h")
        btc_preds = preds[:, btc_idx].cpu().numpy()

        # Add to history with 'live' label and current timestamp
        current_ts = buffers[("BTC/USDT", "15m")].index[-1]
        pred_history.append((current_ts, pred_4h, "live"))

        # 5. Populate the Deque
        for ts, val in zip(timestamps, btc_preds):
            pred_history.append((ts, val, "historical"))

        print(f"[SUCCESS] Experience buffer primed with {len(pred_history)} historical predictions.")

        # 4. Market State
        current_price = buffers[("BTC/USDT", "15m")]["close"].iloc[-1]
        position_qty = get_position_qty("BTC/USD")

        # 5. Trading Logic
        
        # --- STOP LOSS CHECK ---
        if position_qty > 0 and entry_price > 0:
            price_change = (current_price - entry_price) / entry_price
            if price_change <= -STOP_LOSS_PCT:
                print(f"[STOP LOSS] Price dropped {price_change:.2%}. Exiting at {current_price}")
                execute_trade("BTC/USD", "sell", qty=position_qty)
                entry_price = 0.0
                continue # Skip rest of loop to avoid re-entry

        # Calculate the dynamic baseline
        if len(pred_history) >= 20: # Wait for at least 20 samples to be safe
            # raw_values = [p[1] for p in pred_history]
            # rolling_mean = sum(raw_values) / len(raw_values)
            raw_values = np.array([p[1] for p in pred_history])
            rolling_mean = raw_values.mean()
            rolling_std = raw_values.std()

            # rolling_mean = sum(pred_history) / len(pred_history)
            
            # Adjust thresholds relative to the rolling mean
            # dynamic_entry = rolling_mean + 0.0005  # Buy if 0.0005 above average
            # dynamic_exit  = rolling_mean - 0.0002  # Exit if it dips slightly below average

            dynamic_entry = rolling_mean + (rolling_std * 0.75) 
            dynamic_exit  = rolling_mean - (rolling_std * 0.25)
            
            print(f"[LOG] Pred: {pred_4h:.6f} | Mean: {rolling_mean:.6f} | Std: {rolling_std:.6f} | Target entry: {dynamic_entry:.6f} | Target exit: {dynamic_exit:.6f}")

            # --- ENTRY LOGIC (Using Notional to save money) ---
            if pred_4h > dynamic_entry and position_qty == 0:
                # Get account cash balance
                account_data = requests.get(f"{ALPACA_BASE}/v2/account", headers=alpaca_headers()).json()
                available_cash = float(account_data['cash'])
                
                # Use 1% of cash for this trade
                notional_value = round(available_cash * RISK_PER_TRADE_PCT, 2)
                
                if notional_value >= 1.0: # Alpaca minimum $1
                    print(f"[ENTRY] Signal: {pred_4h:.5f}. Buying ${notional_value:.2f} of BTC")
                    # Update execute_trade or call submit_market_order directly with notional
                    submit_market_order("BTC/USD", "buy", notional=notional_value)
                    entry_price = current_price # Record entry for SL check
                else:
                    print(f"[WARN] Notional ${notional_value:.2f} too low to trade.")

            # --- SIGNAL EXIT LOGIC ---
            elif pred_4h < dynamic_exit and position_qty > 0:
                # 1. Round quantity to 8 decimal places (Standard for BTC)
                # 2. Convert to string to prevent scientific notation issues
                clean_qty = floor(position_qty * 10**8) / 10**8

                print(f"[EXIT] Signal: {pred_4h:.5f}. Selling {clean_qty} BTC")
                execute_trade("BTC/USD", "sell", qty=clean_qty)
                entry_price = 0.0
        else: 
            print(f"[INFO] Collecting prediction history... ({len(pred_history)}/20)")
            
        print(f"[STATUS] BTC Price: {current_price:.2f} | Pred: {pred_4h:.5f} | Pos: {position_qty}")

    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        # time.sleep(30) # Cool down before retry
        break

    time.sleep(SLEEP_SECONDS)