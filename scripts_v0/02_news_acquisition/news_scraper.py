from typing import List
from tqdm import tqdm
import yaml
from alpha_vantage.alphaintelligence import AlphaIntelligence
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

# Load YAML configuration
def load_config(config_path="config.yaml"):
    """Loads configuration settings from a YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit()

def save_to_parquet(data: pd.DataFrame, name: str):
    """Saves DataFrame to a Parquet file."""
    if not DATA.exists():
        DATA.mkdir(parents=True)
    data.to_parquet(DATA / name, index=False)
    print(f"Data saved to {DATA / name}")

def fetch_news_sentiment(ai: AlphaIntelligence, tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetches news sentiment data for given tickers within a date range."""
    all_data = []
    for ticker in tqdm(tickers, desc="Fetching news sentiment"):
        data, _ = ai.get_news_sentiment(tickers=ticker, time_from=start, time_to=end, limit=1000)
        all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

def main():
    config = load_config("config.yml")
    alphavantage_key = config['keys']['alphavantage']
    start = config['data_source']['start_date']
    end = config['data_source']['end_date']
    
    ai = AlphaIntelligence(key=alphavantage_key, output_format='pandas')

    tickers = config['alphavantage']['news']['tickers']
    data = fetch_news_sentiment(ai, tickers, start, end)
    print(data.head())
    print(f"Total records fetched: {len(data)}")
    
    # Drop duplicates using only hashable columns (exclude list columns like ticker_sentiment)
    # Use url or title as unique identifier
    if 'url' in data.columns:
        data.drop_duplicates(subset=['url'], inplace=True)
    elif 'title' in data.columns:
        data.drop_duplicates(subset=['title'], inplace=True)
    else:
        print("Warning: No suitable column for deduplication found")
    
    print(f"Total records after removing duplicates: {len(data)}")

    save_to_parquet(data, "news_sentiment.parquet")

if __name__ == "__main__":
    main()