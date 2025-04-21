import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from config import TICKERS, START_DATE, END_DATE, PROCESSED_DATA_DIR


def fetch_and_save_data(
    tickers=TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    output_dir=PROCESSED_DATA_DIR,
):
    """
    Fetches daily Close prices for given tickers and saves each as CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    for ticker, filename_prefix in tickers.items():
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data fetched for {ticker}")
        closing = data[["Close"]].rename(columns={"Close": f"{filename_prefix}_close"})
        today = datetime.today().strftime("%Y-%m-%d")
        path = os.path.join(output_dir, f"{today}_{filename_prefix}.csv")
        closing.to_csv(path)
        print(f"[risk.data] saved {ticker} → {path}")
