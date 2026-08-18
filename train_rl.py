import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

# Assicura che la directory radice sia nel PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent))

import config
from database.db_manager import DBManager
from models.rl.rl_model import RLTradingModel
from models.rl.trading_env import TradingEnv


def load_all_tickers_data(limit_tickers: int = 0) -> Dict[str, pd.DataFrame]:
    """
    Carica i dati di tutti i ticker presenti nel database SQLite locale (tabella ohlcv).
    Se limit_tickers > 0, limita il numero di ticker caricati.
    """
    db = DBManager()
    query_tickers = "SELECT DISTINCT ticker FROM ohlcv"
    tickers_df = db.execute_query(query_tickers)
    tickers = tickers_df["ticker"].tolist()

    if limit_tickers > 0:
        tickers = tickers[:limit_tickers]

    print(f"[INFO] Caricamento dati per {len(tickers)} ticker dal database SQLite...")

    ticker_dfs = {}
    for ticker in tickers:
        query = f"SELECT timestamp as Date, open as Open, high as High, low as Low, close as Close, volume as Volume FROM ohlcv WHERE ticker='{ticker}' ORDER BY timestamp ASC"
        df = db.execute_query(query)
        if not df.empty and len(df) >= 300:
            df["Date"] = pd.to_datetime(df["Date"])
            ticker_dfs[ticker] = df

    print(f"[INFO] Caricati {len(ticker_dfs)} ticker validi con almeno 300 righe storiche ciascuno.")
    return ticker_dfs


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calcola le feature stazionarie ed indicatori tecnici privi di look-ahead bias.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # 1. Log Returns
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 2. RSI (Relative Strength Index 14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_scaled"] = (rsi - 50.0) / 50.0  # Scalato tra [-1, 1]

    # 3. MACD Relativo
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_rel"] = macd / df["Close"]
    df["macd_signal_rel"] = signal / df["Close"]

    # 4. Bollinger Bands (%B)
    sma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    df["bollinger_b"] = (df["Close"] - lower_bb) / ((upper_bb - lower_bb) + 1e-8) - 0.5

    # 5. Volatilità Rolling 21g
    df["volatility_21"] = df["log_return"].rolling(window=21).std() * np.sqrt(252)

    # 6. ATR Relativo
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    df["atr_rel"] = atr / df["Close"]

    feature_cols = ["log_return", "rsi_scaled", "macd_rel", "macd_signal_rel", "bollinger_b", "volatility_21", "atr_rel"]

    df = df.dropna().reset_index(drop=True)
    return df, feature_cols


def prepare_multi_ticker_splits(
    ticker_dfs: Dict[str, pd.DataFrame], train_ratio: float = 0.70, val_ratio: float = 0.15
):
    """
    Processa e suddivide tutti i ticker in Train/Val/Test in sequenza temporale rigorosa,
    e normalizza le feature calcolando media e std ESCLUSIVAMENTE sui Train splits.
    """
    raw_train_dfs, raw_val_dfs, raw_test_dfs = [], [], []
    feature_cols = []

    for ticker, df_raw in ticker_dfs.items():
        df_feat, feat_cols = engineer_features(df_raw)
        if df_feat.empty or len(df_feat) < 250:
            continue
        feature_cols = feat_cols

        n = len(df_feat)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        raw_train_dfs.append(df_feat.iloc[:train_end].copy().reset_index(drop=True))
        raw_val_dfs.append(df_feat.iloc[train_end:val_end].copy().reset_index(drop=True))
        raw_test_dfs.append(df_feat.iloc[val_end:].copy().reset_index(drop=True))

    # Calcolo Z-Score globali solo sui Train splits per evitare Data Leakage
    all_train_data = pd.concat([df[feature_cols] for df in raw_train_dfs], ignore_index=True)
    means = all_train_data[feature_cols].mean()
    stds = all_train_data[feature_cols].std().replace(0, 1.0)

    # Applicazione Z-score normalizzazione su tutti i DataFrame
    train_dfs, val_dfs, test_dfs = [], [], []

    for df in raw_train_dfs:
        d = df.copy()
        for col in feature_cols:
            d[col] = (d[col] - means[col]) / (stds[col] + 1e-8)
        train_dfs.append(d)

    for df in raw_val_dfs:
        d = df.copy()
        for col in feature_cols:
            d[col] = (d[col] - means[col]) / (stds[col] + 1e-8)
        val_dfs.append(d)

    for df in raw_test_dfs:
        d = df.copy()
        for col in feature_cols:
            d[col] = (d[col] - means[col]) / (stds[col] + 1e-8)
        test_dfs.append(d)

    return train_dfs, val_dfs, test_dfs, feature_cols


def main():
    print("=" * 70)
    print("  TRAINING MULTI-TICKER CROSS-ASSET REINFORCEMENT LEARNING (PPO)")
    print("=" * 70)

    total_timesteps = 1_000_000  # 1 Milione di step per training robusto multi-asset
    n_envs = 8                  # 8 worker paralleli su CPU

    # 1. Caricamento dati di tutti i ticker dal database SQLite
    ticker_dfs = load_all_tickers_data()
    if not ticker_dfs:
        print("[ERROR] Nessun dato trovato nel database locale. Assicurati di aver popolato SQLite.")
        return

    # 2. Ingegnerizzazione feature e split anti-data leakage
    print("\n[INFO] Ingegnerizzazione delle feature e suddivisione per ciascun asset...")
    train_dfs, val_dfs, test_dfs, feature_cols = prepare_multi_ticker_splits(ticker_dfs)

    total_train_rows = sum(len(d) for d in train_dfs)
    print(f"[INFO] Cross-Asset Dataset Pronto:")
    print(f"       - Ticker Attivi: {len(train_dfs)}")
    print(f"       - Totale Righe di Training: {total_train_rows:,} campioni giornalieri di borsa")
    print(f"       - Feature Stazionarie: {feature_cols}")

    # 3. Setup del dispositivo e modello RL
    import torch
    device = "cuda" if torch.cuda.is_available() else "auto"
    print(f"\n[INFO] Dispositivo PyTorch selezionato: {device} | Worker Paralleli: {n_envs}")

    rl_model = RLTradingModel(
        algorithm="PPO",
        window_size=30,
        learning_rate=3e-4,
        ent_coef=0.01,
        total_timesteps=total_timesteps,
        device=device,
        n_envs=n_envs,
    )

    # 4. Avvio Training Multi-Asset
    print(f"\n[INFO] Avvio addestramento PPO su {total_timesteps:,} step totali across {len(train_dfs)} asset...")
    stats = rl_model.train(X_train=train_dfs, feature_cols=feature_cols)

    # 5. Salvataggio Modello
    save_path = config.BASE_DIR / "models" / "rl_ppo_model"
    rl_model.save(str(save_path))

    print("\n" + "=" * 70)
    print(f"  CROSS-ASSET TRAINING COMPLETATO E MODELLO SALVATO IN: {save_path}.zip")
    print("=" * 70)


if __name__ == "__main__":
    main()
