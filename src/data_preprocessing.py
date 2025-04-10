import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os

def download_stock_data(ticker="TSLA", start_date="2023-01-01", end_date="2024-03-24"):
    """Download stock data from Yahoo Finance"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data, save_path="data/preprocessed_stock_data.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = data.reset_index()
    data.to_csv(save_path, index=False)
    print(f"✅ Raw data saved to {save_path}")
    return data

def manual_minmax_scale(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)

def load_guaranteed_numeric_data(filepath="data/preprocessed_stock_data.csv", output_path="data/verified_numeric_only.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df = pd.read_csv(filepath)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        df = pd.read_csv(filepath, header=0)

    expected_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    if not all(col in df.columns for col in expected_columns):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 5:
            rename_map = {numeric_cols[i]: expected_columns[i] for i in range(5)}
            df = df.rename(columns=rename_map)
        else:
            raise ValueError("❌ Could not identify expected columns in data file")

    for col in expected_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    # Add indicators
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['WILLIAMS_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14).williams_r()
    df['STOCH_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['STOCH_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    df = df.dropna()
    df = df.select_dtypes(include=[np.number])

    for col in df.columns:
        df[col] = manual_minmax_scale(df[col])

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed and verified data saved to {output_path}")
    return df
