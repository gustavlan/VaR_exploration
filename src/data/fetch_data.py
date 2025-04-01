import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from config import TICKERS, START_DATE, END_DATE, PROCESSED_DATA_DIR

def fetch_and_save_data(tickers, start_date, end_date, output_dir):
    """
    Fetches historical daily closing prices for given tickers from Yahoo Finance,
    and saves each as a CSV in the specified directory.

    Args:
        tickers (dict): Dictionary with ticker symbols as keys and filename prefixes as values.
                        Example: {'^GSPC': 'sp500', '^IXIC': 'nasdaq'}
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        output_dir (str): Directory path to save CSV files.

    Raises:
        Exception: If fetching data fails or the saving process encounters an issue.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for ticker, filename_prefix in tickers.items():
        try:
            # Fetch data from Yahoo Finance
            print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
            data = yf.download(ticker, start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data fetched for {ticker}.")

            # Extract only the closing prices
            closing_prices = data[['Close']].copy()

            # Rename 'Close' column to ticker-friendly name
            closing_prices.rename(columns={'Close': f'{filename_prefix}_close'}, inplace=True)

            # Set filename with today's date
            today_str = datetime.today().strftime('%Y-%m-%d')
            filename = f"{today_str}_{filename_prefix}.csv"
            filepath = os.path.join(output_dir, filename)

            # Save to CSV
            closing_prices.to_csv(filepath)
            print(f"Data for {ticker} saved successfully at {filepath}.\n")

        except Exception as e:
            print(f"Error fetching or saving data for {ticker}: {e}\n")

if __name__ == "__main__":
    fetch_and_save_data(TICKERS, START_DATE, END_DATE, PROCESSED_DATA_DIR)
