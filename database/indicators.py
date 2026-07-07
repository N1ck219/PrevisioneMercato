import logging
import pandas as pd
import numpy as np

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Indicators")


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Riceve un DataFrame contenente i prezzi storici di un singolo ticker ordinati cronologicamente
    e restituisce un nuovo DataFrame contenente tutti gli indicatori tecnici calcolati in modo vettoriale.
    
    Il DataFrame in input deve contenere le colonne:
    ['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    # Assicuriamoci che il DataFrame sia ordinato cronologicamente
    df_sorted = df.sort_values("timestamp").copy()
    
    # Estraiamo le serie necessarie
    close = df_sorted["close"]
    high = df_sorted["high"]
    low = df_sorted["low"]

    # 1. Simple Moving Averages (SMA)
    df_sorted["sma_10"] = close.rolling(window=10).mean()
    df_sorted["sma_20"] = close.rolling(window=20).mean()
    df_sorted["sma_50"] = close.rolling(window=50).mean()
    df_sorted["sma_200"] = close.rolling(window=200).mean()

    # 2. Exponential Moving Averages (EMA)
    df_sorted["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df_sorted["ema_21"] = close.ewm(span=21, adjust=False).mean()

    # 3. Relative Strength Index (RSI - 14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's exponential moving average per il guadagno e la perdita media
    # Wilder usa un fattore di lisciamento alpha = 1 / window
    # che equivale a ewm(com=window-1)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    # Evita la divisione per zero
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    df_sorted["rsi_14"] = 100 - (100 / (1 + rs))

    # 4. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df_sorted["macd"] = ema_12 - ema_26
    df_sorted["macd_signal"] = df_sorted["macd"].ewm(span=9, adjust=False).mean()
    df_sorted["macd_hist"] = df_sorted["macd"] - df_sorted["macd_signal"]

    # 5. Bollinger Bands (20, 2)
    df_sorted["bb_middle"] = close.rolling(window=20).mean()
    # Metodo più veloce per deviazione standard rolling
    std_rolling = close.rolling(window=20).std()
    df_sorted["bb_upper"] = df_sorted["bb_middle"] + (2 * std_rolling)
    df_sorted["bb_lower"] = df_sorted["bb_middle"] - (2 * std_rolling)

    # 6. Average True Range (ATR - 14)
    # TR = max(high - low, abs(high - close_prev), abs(low - close_prev))
    high_low = high - low
    high_close_prev = (high - close.shift(1)).abs()
    low_close_prev = (low - close.shift(1)).abs()
    
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    # L'ATR originale di Wilder usa un lisciamento EMA con alpha = 1 / window
    df_sorted["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Prepariamo il DataFrame finale degli indicatori
    # Rimuoviamo i campi OHLCV originali per evitare sovrapposizioni inutili a DB
    cols_to_keep = [
        "ticker", "timestamp", "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_9", "ema_21", "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower", "atr_14"
    ]
    
    # Ritorniamo il DataFrame pulito con gli indicatori
    # Nota: Le prime righe avranno valori NaN a causa delle rolling window (es. SMA 200 necessita di 200 righe per essere calcolata).
    # DuckDB gestisce nativamente i valori NULL per queste righe.
    return df_sorted[cols_to_keep]
