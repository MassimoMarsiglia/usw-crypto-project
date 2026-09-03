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
BUFFER_DIR = os.path.join(BASE_DIR, "../../data/live_buffers")
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

LOOKBACK_BARS = {
    "15m": 200,   # 2 days
    "1h": 200,    # ~8 days
    "4h": 200,    # ~33 days
    "1d": 200     # ~200 days
}
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

# fetch data multi x multi
# def fetch_multi_asset_bars(
#     symbols: list[str],
#     timeframes: dict[str, TimeFrame],
#     lookback: dict[str, int],
#     # save_dir: str = "./fetched_bars"
# ) -> dict[tuple[str, str], pd.DataFrame]:
    
#     # os.makedirs(save_dir, exist_ok=True)
#     out = {}
    
#     for tf_str, tf_obj in timeframes.items():
#         # Iterate through symbols individually for more robust fetching
#         for sym in symbols:
#             try:
#                 # 1. Request bars
#                 lb = lookback.get(tf_str, 200)
#                 print(f"[DEBUG] Fetching {sym} at {tf_str} with lookback={lb}")
#                 request = CryptoBarsRequest(
#                     symbol_or_symbols=[sym],
#                     timeframe=tf_obj,
#                     limit=lb
#                 )
                
#                 bars = client.get_crypto_bars(request).df
                
#                 print(f"[DEBUG] {sym}/{tf_str} fetched bars shape: {bars.shape}")

#                 if bars.empty:
#                     print(f"Skipping {sym}/{tf_str}: No data returned.")
#                     continue

                
#                 # --- Save raw fetched bars BEFORE any processing ---
#                 # raw_file_path = os.path.join(save_dir, f"{sym}_{tf_str}_raw.parquet")
#                 # os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
#                 # bars.to_parquet(raw_file_path)
                
#                 # 2. Extract, Clean Index, and Store
#                 # Since we requested only one symbol, the index should be a simple DatetimeIndex (Level 0)
#                 # and the symbol index (Level 1) can be cleanly dropped or doesn't exist.
                
#                 # Fix: Handle the possibility of a MultiIndex, but assume the simpler structure
#                 if bars.index.nlevels > 1:
#                     # Level 0: symbol, Level 1: timestamp
#                     df = bars.xs(sym, level=0).copy()  # select only current symbol
#                 else:
#                     df = bars.copy()


#                 df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

#                 out[(sym, tf_str)] = df[
#                     ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
#                 ]
                
#             except Exception as e:
#                 # Catch API errors for specific symbols
#                 print(f"Error fetching {sym} at {tf_str}: {e}")
#                 continue
                
#     return out

# version 1: fetching data
def bootstrap_buffers(symbols, tf_map, lookback):
    buffers = {}
    now = datetime.now(timezone.utc)

    TF_DELTAS = {
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }

    for tf_str, tf_obj in tf_map.items():
        lb = lookback[tf_str]
        tf_delta = TF_DELTAS[tf_str]

        for sym in symbols:
            # end = min(TRAIN_END, now)
            # start = end - lb * tf_delta

            # if start >= end:
            #     print(f"[SKIP] {sym}/{tf_str}: start >= end")
            #     continue

            # print(
            #     f"[BOOTSTRAP] {sym}/{tf_str} | "
            #     f"start={start} end={end} bars≈{lb}"
            # )

            req = CryptoBarsRequest(
                symbol_or_symbols=[sym],
                timeframe=tf_obj,
                start=TRAIN_END,
                end=now,
            )

            df = client.get_crypto_bars(req).df

            if df.empty:
                print(f"[WARN] No data for {sym}/{tf_str}")
                continue

            if isinstance(df.index, pd.MultiIndex):
            # Alpaca crypto bars: timestamp is ALWAYS one level
                for level in range(df.index.nlevels):
                    try:
                        pd.to_datetime(df.index.get_level_values(level))
                        df = df.droplevel(1 - level)
                        break
                    except Exception:
                        continue

            if df.index.nlevels > 1:
                df = df.droplevel(1)

            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

            buffers[(sym, tf_str)] = df[
                ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
            ]

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

def is_rth(ts: pd.Timestamp, cal_map: Dict[datetime.date, Tuple[datetime, datetime]]) -> bool:
    if ts.tzinfo is None:
        # Assume UTC if tz-naive
        ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)
    else:
        try:
            ts_eastern = ts.tz_convert(EASTERN)  # type: ignore[attr-defined]
        except Exception:
            ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)
    d = ts_eastern.date()
    if d not in cal_map:
        return False
    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt


# position
def get_position_qty(symbol: str = "BTC/USDT") -> float:
    """
    Returns the current position quantity for a symbol.
    Returns 0 if no position exists.
    """
    url = f"{ALPACA_BASE}/v2/positions"
    r = requests.get(url, headers=alpaca_headers(), timeout=30)
    if r.status_code == 404:
        return 0.0
    r.raise_for_status()
    
    positions = r.json()
    for pos in positions:
        if pos["symbol"] == symbol:
            return float(pos["qty"])
    return 0.0

def get_filled_orders_for_symbol(symbol: str, limit: int = 50) -> List[dict]:
    # /v2/orders?status=closed will include filled orders; we also filter by symbol client-side to be safe
    url = f"{ALPACA_BASE}/v2/orders"
    params_q = {
        "status": "closed",
        "limit": str(limit),
        "nested": "false",
        "direction": "desc",
    }
    r = requests.get(url, headers=alpaca_headers(), params=params_q, timeout=30)
    r.raise_for_status()
    orders = r.json()
    # Filter filled and by symbol
    out = []
    for o in orders:
        if str(o.get("status", "")).lower() != "filled":
            continue
        if str(o.get("symbol", "")).upper() != symbol.upper():
            continue
        out.append(o)
    return out

# version 1: submit order 
def submit_market_order(symbol: str, side: str, qty: int | float = 1) -> dict | None:
    # Ensure side is a string and lower case
    if not isinstance(side, str):
        raise ValueError(f"Expected 'side' to be a string, got {type(side)}")
    side = side.lower()

    # Ensure qty is numeric (int or float)
    if not isinstance(qty, (int, float)):
        raise ValueError(f"Expected 'qty' to be a numeric value, got {type(qty)}")

    # Ensure qty is a positive number
    if qty <= 0:
        raise ValueError(f"Expected 'qty' to be a positive number, got {qty}")

    url = f"{ALPACA_BASE}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": float(qty),  # ensure numeric type
        "side": side,
        "type": "market",
        "time_in_force": "gtc",
    }
    
    try:
        r = requests.post(url, headers=alpaca_headers(), json=payload, timeout=30)
        r.raise_for_status()
        od = r.json()
        print(f"[order] {side.upper()} {qty} {symbol}: submitted id={od.get('id')}")
        return od
    except requests.HTTPError as e:
        # Raise exception with status code and response content
        raise RuntimeError(f"Order failed ({r.status_code}): {r.text}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error placing order: {e}") from e



def close_positions_older_than_30m():
    """Check open positions and sell those whose most recent BUY fill is >= 30 minutes ago."""
    try:
        positions = get_position_qty()
    except Exception as e:
        print(f"[pos] Cannot fetch positions: {e}")
        return

    if not positions:
        print("[pos] No open positions.")
        return

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(minutes=30)

    for p in positions:
        symbol = p.get("symbol") or p.get("asset_symbol")
        if not symbol:
            continue
        try:
            qty_str = p.get("qty") or p.get("quantity")
            qty = int(float(qty_str)) if qty_str is not None else None
        except Exception:
            qty = None
        if not qty:
            continue

        # Find last filled BUY order for this symbol
        try:
            orders = get_filled_orders_for_symbol(symbol, limit=50)
        except Exception as e:
            print(f"[pos] get_orders failed for {symbol}: {e}")
            continue

        last_buy_fill = None
        for o in orders:
            side = str(o.get("side", "")).lower()
            if side != "buy":
                continue
            filled_at = o.get("filled_at")
            if not filled_at:
                continue
            try:
                # ISO8601 with Z
                dt = datetime.fromisoformat(str(filled_at).replace("Z", "+00:00"))
            except Exception:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            if (last_buy_fill is None) or (dt > last_buy_fill):
                last_buy_fill = dt

        if last_buy_fill and last_buy_fill <= cutoff:
            print(f"[pos] Closing {symbol}: last BUY fill at {last_buy_fill.isoformat()} (<= {cutoff.isoformat()})")
            submit_market_order(symbol, side="sell", qty=qty)
        else:
            print(f"[pos] Keep {symbol}: last BUY fill at {last_buy_fill}")

# version 1 : generate features
def generate_multi_asset_multi_tf_features(
    bars: dict[tuple[str, str], pd.DataFrame],
    base_timeframe: str = "15m"
) -> pd.DataFrame:
    feature_blocks = []
    base_index = None

    print(f"[INFO] Starting feature generation for {len(bars)} (symbol, tf) pairs")

    # Sort keys so base timeframe comes first for each symbol
    sorted_keys = sorted(
        bars.keys(),
        key=lambda x: (0 if x[1] == base_timeframe else 1, x[0])
    )

    for (symbol, tf) in sorted_keys:
        df = bars[(symbol, tf)]
        pair_name = PAIR_MAP[symbol]

        print(f"[INFO] Generating features for {symbol} at {tf}... df shape: {df.shape}")
        f = generate_features(df, pair_name, tf)

        if f.empty:
            print(f"[WARN] Features for {symbol}/{tf} are empty, skipping.")
            continue

        print(f"[INFO] Features shape for {symbol}/{tf}: {f.shape}")

        # Establish base index from base timeframe
        if tf == base_timeframe and base_index is None:
            base_index = f.index
            print(f"[INFO] Base index established from {symbol}/{tf}, length={len(base_index)}")

        # Align higher timeframes to base_index
        if tf != base_timeframe:
            if base_index is None:
                print(f"[WARN] Base index not set yet. Skipping alignment for {symbol}/{tf}")
            else:
                print(f"[INFO] Aligning {symbol}/{tf} features to base_index (length={len(base_index)})")
                try:
                    f = f.reindex(base_index, method="ffill").loc[base_index]
                    print(f"[INFO] Alignment done. Shape after reindex & loc: {f.shape}")
                except Exception as e:
                    print(f"[ERROR] Failed to align {symbol}/{tf}: {e}")
                    continue

        # Optional: add prefix to distinguish features per pair/timeframe
        # f = f.add_prefix(f"{pair_name}__{tf}__")

        feature_blocks.append(f)
        print(f"[INFO] Feature block appended. Total blocks: {len(feature_blocks)}")

    if not feature_blocks:
        print("[WARN] No feature blocks generated. Returning empty DataFrame.")
        return pd.DataFrame()

    # Combine all feature blocks
    X = pd.concat(feature_blocks, axis=1)
    print(f"[INFO] Shape after concatenation: {X.shape}")

    # Fill missing values
    X = X.fillna(method="ffill").fillna(0)
    print(f"[INFO] Shape after filling NaNs: {X.shape}")

    # Final cleanup
    X = X.dropna()
    print(f"[INFO] Final shape after dropping NaNs: {X.shape}")

    return X


# save buffers to parquet
# def save_buffer_to_parquet(buffers:dict):
#     for (symbol, tf), df in buffers.items():
#         if df.empty:
#             continue
#         safe_symbol = symbol. replace("/", "_")
#         path = os.path.join(BUFFER_DIR, f"{safe_symbol}_{tf}.parquet")
#         df.to_parquet(path)

# save buffer
def save_buffers(buffers, path="./buffers"):
    os.makedirs(path, exist_ok=True)

    for (sym, tf), df in buffers.items():
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

# version 1: update buffer
def update_buffer(
    buffers,
    symbol,
    tf_str,
    tf_obj,
    max_len=500
):
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=tf_obj,
        limit=2,  # only latest bar
    )

    df_new = client.get_crypto_bars(req).df

    if df_new.empty:
        return

    if isinstance(df_new.index, pd.MultiIndex):
        # Alpaca crypto bars: timestamp is ALWAYS one level
            for level in range(df_new.index.nlevels):
                try:
                    pd.to_datetime(df_new.index.get_level_values(level))
                    df_new = df_new.droplevel(1 - level)
                    break
                except Exception:
                    continue

    if df_new.index.nlevels > 1:
        df_new = df_new.droplevel(1)

    df_new.index = pd.to_datetime(df_new.index, utc=True, format="%Y-%m-%dT%H:%M:%SZ").tz_convert(None)

    buf = buffers[(symbol, tf_str)]

    if df_new.index[-1] <= buf.index[-1]:
        return  # no new bar

    buf = pd.concat([buf, df_new]).tail(max_len)
    buffers[(symbol, tf_str)] = buf

            
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
# BUFFER_DIR = "./buffers"
# print("TRAIN_END:", TRAIN_END)
# print("NOW:", datetime.now(timezone.utc))

# if not os.path.exists(BUFFER_DIR) or len(os.listdir(BUFFER_DIR)) == 0:
#     print("[INIT] No buffers found. Bootstrapping historical data...")

#     buffers = bootstrap_buffers(
#         symbols=USDT_PAIRS + BTC_PAIRS,
#         tf_map=TF_MAP,
#         lookback=LOOKBACK_BARS["15m"]
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

# version 1: main loop
print("LIVE TRADING STARTED")

while True:
    try:
        # 1️. Update buffer
        for tf_str, tf_obj in TF_MAP.items():
            for sym in symbols:
                update_buffer(buffers, sym, tf_str, tf_obj)

        # 2️. Generate features
        X_all = generate_multi_asset_multi_tf_features(buffers, base_timeframe="15m")

        # 3️. Align to training columns
        missing = set(feature_columns) - set(X_all.columns)
        if missing:
            print(f"[WARN] Missing features: {len(missing)} columns")
        X_all = X_all.reindex(columns=feature_columns, fill_value=0.0)

        if X_all.shape[0] == 0:
            print("[WARN] Feature matrix empty, skipping iteration")
            time.sleep(SLEEP_SECONDS)
            continue

        X_live = X_all.iloc[-1].values.reshape(1, -1)
        print(f"[INFO] X_live shape: {X_live.shape}")

        # 4️. Model prediction
        with torch.no_grad():
            pred = model(torch.tensor(X_live, dtype=torch.float32).to(DEVICE))

        # 5️. Map output index to target
        target_order = [
            "BTCUSDT_target_logret_4h",
            "BTCUSDT_target_logret_1d",
            "ETHUSDT_target_logret_4h",
            "ETHUSDT_target_logret_1d",
            "LTCUSDT_target_logret_4h",
            "LTCUSDT_target_logret_1d",
            "SOLUSDT_target_logret_4h",
            "SOLUSDT_target_logret_1d",
        ]
        btc_4h_idx = target_order.index("BTCUSDT_target_logret_4h")
        pred_4h = pred[0, btc_4h_idx].item()

        price = buffers[("BTC/USDT", "15m")]["close"].iloc[-1]
        position_qty = get_position_qty("BTC/USDT")
        print(f"[LIVE] BTC/USDT 4h pred: {pred_4h:.5f}, Position: {position_qty}, Price: {price:.2f}")

        # 6️. Trading
        print("[INFO] About to submit order if needed")
        # if pred_4h > ENTRY_THRESHOLD and position_qty == 0:
        #     qty = MAX_POSITION_USD / price
        #     threading.Thread(target=execute_trade, args=("buy", qty), daemon=True).start()
        # elif pred_4h < EXIT_THRESHOLD and position_qty > 0:
        #     threading.Thread(target=execute_trade, args=("sell", position_qty), daemon=True).start()

        if pred_4h > ENTRY_THRESHOLD and position_qty == 0:
            MIN_ORDER_USD = 1.0  # Alpaca minimum order size

            qty = MAX_POSITION_USD / price
            order_value = qty * price

            if order_value < MIN_ORDER_USD:
                print(f"[WARN] Order value ${order_value:.2f} below minimum ${MIN_ORDER_USD}, skipping trade")
            else:
                execute_trade("BTC/USDT", "buy", round(qty, 6))

        elif pred_4h < EXIT_THRESHOLD and position_qty > 0:
            execute_trade("BTC/USDT", "sell", position_qty)

        print("[INFO] Trading logic done, sleeping...")

    except Exception as e:
        print("[ERROR]", e)
        break

    time.sleep(SLEEP_SECONDS)

