import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Assicura installazione automatica di Optuna se mancante usando uv/pip
try:
    import optuna
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AutoInstall")
    logger.info("Optuna non rilevato nel sistema. Tentativo di installazione in corso...")
    try:
        # Usa uv se disponibile, altrimenti pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
        import optuna
        logger.info("Optuna installato ed importato con successo!")
    except Exception as e:
        logger.error(f"Errore durante l'installazione di Optuna: {e}")
        sys.exit(1)

import numpy as np
import pandas as pd
import torch

# Assicuriamoci che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from database.db_manager import DBManager
from train import prepare_features_and_targets_v4, create_temporal_sequences
from models.rete_neurale.v4.model import NeuralNetworkV4

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HyperTuning")


def objective(trial: optuna.Trial, df_raw: pd.DataFrame, tickers: List[str]) -> float:
    # 1. Prepara i parametri del Trial
    lookback = trial.suggest_int("lookback", 15, 45, step=5)
    
    # d_model deve essere divisibile per nhead (es. nhead=4, d_model in [32, 64, 128])
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
    
    # 2. Ingegneria delle feature
    df_features, y_series, feature_cols = prepare_features_and_targets_v4(df_raw)
    df_features['target'] = y_series
    df_features_sorted = df_features.sort_values('timestamp').reset_index(drop=True)
    
    total_samples = len(df_features_sorted)
    train_end_idx = int(total_samples * 0.70)
    val_end_idx = int(total_samples * 0.85)
    
    # Purged & Embargoed splits
    df_train = df_features_sorted.iloc[:train_end_idx]
    df_val = df_features_sorted.iloc[train_end_idx + lookback : val_end_idx]
    
    X_train_raw = df_train[feature_cols].values
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0.0] = 1e-8
    
    # Creazione sequenze 3D
    train_cutoff_time = pd.Timestamp(df_train['timestamp'].iloc[-1])
    val_cutoff_time = pd.Timestamp(df_val['timestamp'].iloc[-1])
    
    df_scaled = df_features_sorted.copy()
    df_scaled[feature_cols] = (df_scaled[feature_cols] - mean) / std
    
    X_train, y_train, X_val, y_val, _, _ = create_temporal_sequences(
        df_scaled, feature_cols, lookback, train_cutoff_time, val_cutoff_time
    )
    
    # 3. Addestramento del Modello (Epoche limitate per velocità di tuning)
    input_dim = len(feature_cols)
    model = NeuralNetworkV4(
        input_dim=input_dim,
        lookback=lookback,
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        lr=lr
    )
    
    # Impostiamo il dropout dinamico
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
            
    # Addestra per 8 epoche per trial
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=8,
        batch_size=512,
        early_stopping_rounds=3,
        verbose=False
    )
    
    # Metrica da minimizzare: la migliore Val Loss ottenuta
    val_loss = min(history["val_loss"]) if history["val_loss"] else 1e9
    return val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Ottimizzazione automatica degli iperparametri del Transformer v4 con Optuna"
    )
    parser.add_argument(
        "-t", "--tickers",
        type=str,
        default="AAPL,MSFT,NVDA",
        help="Ticker da utilizzare per il tuning veloce (default: AAPL,MSFT,NVDA)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Numero di trial di ottimizzazione da eseguire (default: 10)"
    )
    args = parser.parse_args()
    
    logger.info("=== AVVIO TUNING AUTOMATICO DEGLI IPERPARAMETRI (OPTUNA) ===")
    
    db = DBManager()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    logger.info(f"Caricamento dati per il tuning sui ticker: {tickers}")
    
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
        SELECT 
            o.ticker, o.timestamp, o.open, o.high, o.low, o.close, o.volume,
            i.sma_10, i.sma_20, i.sma_50, i.sma_200, 
            i.ema_9, i.ema_21, i.rsi_14, 
            i.bb_upper, i.bb_middle, i.bb_lower, i.atr_14
        FROM ohlcv o
        INNER JOIN indicators i 
            ON o.ticker = i.ticker 
           AND o.timestamp = i.timestamp
        WHERE o.ticker IN ({placeholders})
    """
    
    df_raw = db.execute_query(query, tuple(tickers))
    if df_raw.empty:
        logger.error("Nessun dato trovato nel DB. Tuning interrotto.")
        sys.exit(1)
        
    logger.info(f"Dati caricati: {len(df_raw)} righe. Avvio dello studio Bayesian con Optuna...")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df_raw, tickers), n_trials=args.trials)
    
    logger.info("=== FINE OTTIMIZZAZIONE ===")
    logger.info(f"Miglior Val Loss: {study.best_value:.6f}")
    logger.info("Migliori Iperparametri Rilevati:")
    for key, value in study.best_params.items():
        logger.info(f"  - {key}: {value}")
        
    # Salva i parametri ottimi in un file JSON nella directory v4 per referenza
    out_dir = config.BASE_DIR / "models" / "rete_neurale" / "v4"
    out_dir.mkdir(exist_ok=True, parents=True)
    import json
    with open(out_dir / "best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    logger.info(f"Migliori parametri salvati in: {out_dir}/best_hyperparameters.json")


if __name__ == "__main__":
    main()
