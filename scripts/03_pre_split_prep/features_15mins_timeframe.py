import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
import os
import matplotlib.pyplot as plt

# config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")
IMAGE_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
IMAGE_DIR_ORG = os.path.join(BASE_DIR, "../../images")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR_ORG, exist_ok=True)

# save csv
csv_BTC_15mins = os.path.join(LARGE_DATA_DIR, "03_btc_features_15mins.csv")

# save images
btc_ema_15mins_trend = os.path.join(IMAGE_DIR, "03_BTC_EMA_15mins_Trend.png")
btc_rsi_15mins_dist = os.path.join(IMAGE_DIR, "03_BTC_RSI_15mins_Distribution.png")
btc_target_15mins_balance = os.path.join(IMAGE_DIR, "03_BTC_Target_15mins_Balance.png")

# path to csv files
csv_BTCpairs_1day = os.path.join(DATA_DIR, "01_BTCPairs_Alpaca_OHLCV_1Day_data.csv")
csv_BTCpairs_5mins = os.path.join(LARGE_DATA_DIR, "01_BTCPairs_Alpaca_OHLCV_5Mins_data.csv")
csv_BTCpairs_15mins = os.path.join(LARGE_DATA_DIR, "01_BTCPairs_Alpaca_OHLCV_15Mins_data.csv")

csv_USDTpairs_1day = os.path.join(DATA_DIR, "01_USDTPairs_Alpaca_OHLCV_1Day_data.csv")
csv_USDTpairs_5mins = os.path.join(LARGE_DATA_DIR, "01_USDTPairs_Alpaca_OHLCV_5Mins_data.csv")
csv_USDTpairs_15mins = os.path.join(LARGE_DATA_DIR, "01_USDTPairs_Alpaca_OHLCV_15Mins_data.csv")

# choose timeframe to analyze
selected_file = csv_USDTpairs_15mins # change to another timeframe if needed

# load csv
df = pd.read_csv(selected_file)

# convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# set the timestamp as index
df = df.set_index('timestamp')

# pivot to wide format for easier plotting
# eg: from timestamp | symbol | close -> timestamp | BTC/USDT | ETH/USDT....
columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
df_ohlcv = df.pivot_table(index='timestamp', columns='symbol', values=columns)
df_ohlcv = df_ohlcv.sort_index()

df_btc_only = df_ohlcv.xs(key='BTC/USDT', level='symbol', axis=1, drop_level=True)

# df_close: timestamp x symbols with Close prices
df = df_btc_only.copy()

# Example: compute log returns
returns = np.log(df / df.shift(1)).dropna()

# Compute features for BTC/USDT
df_features = pd.DataFrame(index=df.index)

# Technical indicators
df_features['BTC_Close'] = df['close']
df_features['BTC_EMA20'] = EMAIndicator(df['close'], window=20).ema_indicator()
df_features['BTC_EMA50'] = EMAIndicator(df['close'], window=50).ema_indicator()
df_features['BTC_RSI14'] = RSIIndicator(df['close'], window=14).rsi()
df_features['BTC_BB_high'] = BollingerBands(df['close'], window=20, window_dev=2).bollinger_hband()
df_features['BTC_BB_low'] = BollingerBands(df['close'], window=20, window_dev=2).bollinger_lband()
df_features['BTC_ATR14'] = AverageTrueRange(low=df['low'], high=df['high'], close=df['close'], window=14).average_true_range()

# Target: 1-day forward movement (binary)
horizon = 1  # 1 day
df_features['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)

# Drop NaN rows
df_features = df_features.dropna()

# save df_features to csv
df_features.to_csv(csv_BTC_15mins, index=True)

# Features
print(df_features.describe())

# Target distribution
print(df_features['target'].value_counts(normalize=True))

# ------
# Plot for 5min Timeframe
# ------

# Plot future trends
plot_df = df_features[['BTC_Close', 'BTC_EMA20', 'BTC_EMA50']].copy()
plot_df.index
fig1 = plot_df.plot(
    figsize=(14, 6), 
    title='BTC/USDT Trend: Close Price vs. EMAs (15Mins Timeframe)'
).get_figure()
fig1.autofmt_xdate(rotation=45)
fig1.savefig(btc_ema_15mins_trend, bbox_inches='tight', dpi=300)
plt.close(fig1)


# Plot RSI Distribution
fig2, ax2 = plt.subplots(figsize=(8, 5))
df_features['BTC_RSI14'].hist(bins=50, ax=ax2)
ax2.set_title('RSI(14) Distribution (15Mins Timeframe)')
ax2.set_xlabel('RSI Value')
ax2.set_ylabel('Frequency')
fig2.savefig(btc_rsi_15mins_dist, bbox_inches='tight', dpi=300)
plt.close(fig2) 

# Plot target distribution
fig3, ax3 = plt.subplots(figsize=(6, 4))
df_features['target'].value_counts().plot(kind='bar', rot=0, ax=ax3)
ax3.set_title('Target Class Balance 15Mins Timeframe (0: Down/Flat, 1: Up)')
ax3.set_xlabel('Direction')
ax3.set_ylabel('Count')
fig3.savefig(btc_target_15mins_balance, bbox_inches='tight', dpi=300)
plt.close(fig3)

