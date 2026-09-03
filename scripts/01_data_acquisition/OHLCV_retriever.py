from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import CryptoBarsRequest
import tqdm
import yaml
import os
from datetime import datetime
from typing import List
from dataclasses import dataclass

# ================================
#  Load Configs
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_DIR = os.path.join(BASE_DIR, "../../conf/keys.yaml")
PARAM_DIR = os.path.join(BASE_DIR, "../../conf/params.yaml")

DATA_DIR = os.path.join(BASE_DIR, "../../data")
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LARGE_DATA_DIR, exist_ok=True)

# Load API keys
with open(KEY_DIR, "r") as file:
    keys = yaml.safe_load(file)

API_KEY = keys["APCA_KEYS"]["APCA_API_KEY_ID"]
SECRET_KEY = keys["APCA_KEYS"]["APCA_API_SECRET_KEY"]

# Load Start/End Date
with open(PARAM_DIR, "r") as file:
    params = yaml.safe_load(file)

START_DATE = datetime.strptime(params["DATA_ACQUISITON"]["START_DATE"], "%Y-%m-%d")
END_DATE   = datetime.strptime(params["DATA_ACQUISITON"]["END_DATE"],   "%Y-%m-%d")

client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# ================================
#  Symbols & Timeframes
# ================================

@dataclass
class Pair:
    def __init__(self, coin: str, pairs: List[str]):
        self.coin = coin
        self.pairs = pairs

BTC_PAIR = Pair(
    coin="BTC",
    pairs=[
    "USDT/BTC",
    "BCH/BTC",
    "ETH/BTC",
    "LTC/BTC",
    "SOL/BTC",
    "XRP/BTC",
    "TRX/BTC",
])

USDT_PAIR = Pair(
    coin="USDT",
    pairs=[
    "BTC/USDT",
    "BCH/USDT",
    "ETH/USDT",
    "LTC/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "TRX/USDT",
    ])

PAIRS = [BTC_PAIR, USDT_PAIR]

TIMEFRAMES = [
    ("15 minutes",  TimeFrame(15, TimeFrameUnit.Minute), "15Mins", False),
    ("30 minutes",  TimeFrame(30, TimeFrameUnit.Minute), "30Mins", False),
    ("60 minutes",  TimeFrame(1, TimeFrameUnit.Hour), "60Mins", False),

    ("4 hours",     TimeFrame(4, TimeFrameUnit.Hour), "240Mins", False),
    ("8 hours",     TimeFrame(8, TimeFrameUnit.Hour), "480Mins", False),
    ("12 hours",    TimeFrame(12, TimeFrameUnit.Hour), "720Mins", False),
    ("16 hours",    TimeFrame(16, TimeFrameUnit.Hour), "960Mins", False),
    ("20 hours",    TimeFrame(20, TimeFrameUnit.Hour),"1200Mins", False),

    ("1 day",       TimeFrame(1,  TimeFrameUnit.Day),     "1Day",   True),
]


# ================================
#  Helper functions
# ================================

def tf(minutes: int):
    return TimeFrame(minutes, TimeFrameUnit.Minute)

def build_output_path(label: str, coin: str):
    filename = f"01_{coin}pairs_OHLCV_{label}.csv"
    return os.path.join(LARGE_DATA_DIR, filename)

# ================================
#  Fetch Function
# ================================

def fetch_ohlcv(tf_obj: TimeFrame, label: str, pair: Pair):
    print(f"\nFetching OHLCV for timeframe: {label}")

    request_params = CryptoBarsRequest(
        symbol_or_symbols=pair.pairs,
        timeframe=tf_obj,
        start=START_DATE,
        end=END_DATE,
    )

    bars = client.get_crypto_bars(request_params)
    df = bars.df.reset_index()

    output_path = build_output_path(label, pair.coin)
    df.to_csv(output_path, index=False)

    print(f"Saved → {output_path}")


# ================================
#  MAIN LOOP
# ================================

def main():

    for _, pair in tqdm.tqdm(enumerate(PAIRS), total=len(PAIRS), desc="Pair Groups"):
        for name, tf_obj, label, is_daily in TIMEFRAMES:
            fetch_ohlcv(tf_obj, label, pair)

if __name__ == "__main__":
    main()