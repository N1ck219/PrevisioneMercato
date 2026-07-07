import logging
from typing import List, Optional, Tuple, Any
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Assicuriamoci che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from database.db_manager import DBManager
from models.rete_neurale.v2.model import NeuralNetworkV2

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainPipelineV2")


def prepare_features_and_targets_v2(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Riceve il DataFrame grezzo unito di prezzi ed indicatori.
    Crea feature avanzate scala-invarianti v2 prendendo ispirazione dai file legacy.
    """
    # 1. Copia ed ordinamento cronologico per ogni ticker per evitare sfasamenti temporali
    df = df.copy().sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    
    # 2. Creazione del Target: 1 se il prezzo di domani è superiore alla chiusura di oggi, altrimenti 0
    # Utilizziamo groupby per calcolare lo shift a ritroso sul singolo ticker per prevenire contaminazioni
    df['close_tomorrow'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['close_tomorrow'] > df['close']).astype(int)
    
    # 3. Creazione delle feature scala-invarianti
    # Per evitare contaminazioni tra diversi ticker, calcoliamo i rendimenti all'interno del groupby
    df['ret'] = df.groupby('ticker')['close'].pct_change().fillna(0)
    df['vol_ret'] = df.groupby('ticker')['volume'].pct_change().fillna(0)
    
    # Calcolo OBV_ret per ciascun ticker
    def compute_obv_ret(group):
        # OBV = cumsum(sign(ret) * volume)
        obv = (np.sign(group['ret']) * group['volume']).fillna(0).cumsum()
        # OBV_ret = pct_change dell'OBV
        return obv.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df['OBV_ret'] = df.groupby('ticker', group_keys=False).apply(compute_obv_ret)
    
    # Altre feature tecniche dirette dalle colonne esistenti nel DB
    close = df['close']
    df['RSI_14'] = df['rsi_14'] / 100.0
    df['ATRr_14'] = df['atr_14'] / close
    
    # Bollinger Bands
    # Bollinger %B = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    # Bollinger Width = (bb_upper - bb_lower) / (bb_middle + 1e-9)
    # Nota: bb_middle è sma_20
    df['Bollinger_%B'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['Bollinger_Width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
    
    # Distanza dalle medie mobili
    df['Dist_SMA200'] = (close - df['sma_200']) / (df['sma_200'] + 1e-9)
    df['Dist_SMA50'] = (close - df['sma_50']) / (df['sma_50'] + 1e-9)
    
    # 4. Definizione delle colonne delle feature v2
    feature_cols = [
        'ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width',
        'ATRr_14', 'Dist_SMA200', 'Dist_SMA50', 'OBV_ret'
    ]
    
    # Rimuoviamo le righe con valori NaN (dovuti alle medie mobili rolling iniziali e allo shift del target)
    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
    
    return df_clean[feature_cols + ['timestamp', 'ticker']], df_clean['target'], feature_cols


def run_training_pipeline_v2(tickers: Optional[List[str]] = None, save_name: str = "neural_model.pth") -> None:
    """
    Esegue l'intero flusso di caricamento, preprocessing v2, addestramento residuo e salvataggio del modello v2.
    """
    logger.info("=== INIZIO PIPELINE DI ADDESTRAMENTO MODELLO PYTORCH V2 ===")
    
    # 1. Caricamento dati da SQLite tramite DBManager
    db = DBManager()
    
    # Query di caricamento per estrarre OHLCV ed indicatori uniti
    query = """
        SELECT 
            o.ticker, o.timestamp, o.open, o.high, o.low, o.close, o.volume,
            i.sma_10, i.sma_20, i.sma_50, i.sma_200, 
            i.ema_9, i.ema_21, i.rsi_14, 
            i.macd, i.macd_signal, i.macd_hist, 
            i.bb_upper, i.bb_middle, i.bb_lower, i.atr_14
        FROM ohlcv o
        INNER JOIN indicators i 
            ON o.ticker = i.ticker 
           AND o.timestamp = i.timestamp
    """
    
    if tickers:
        placeholders = ",".join(["?"] * len(tickers))
        query += f" WHERE o.ticker IN ({placeholders})"
        logger.info(f"Caricamento dati per {len(tickers)} ticker specificati...")
        df_raw = db.execute_query(query, tuple(tickers))
    else:
        logger.info("Caricamento dati di tutti i ticker disponibili nel database...")
        df_raw = db.execute_query(query)
        
    if df_raw.empty:
        logger.error("Nessun dato trovato nel database. Impossibile procedere con l'addestramento.")
        return
        
    logger.info(f"Dati caricati con successo: {len(df_raw)} righe estratte.")
    
    # 2. Preprocessing e ingegneria delle feature v2
    df_features, y_series, feature_cols = prepare_features_and_targets_v2(df_raw)
    logger.info(f"Preprocessing v2 completato. Campioni puliti utilizzabili: {len(df_features)}.")
    logger.info(f"Feature selezionate per l'addestramento: {feature_cols}")
    
    # 3. Suddivisione temporale cronologica per prevenire lookahead bias
    df_features['target'] = y_series
    df_features_sorted = df_features.sort_values('timestamp').reset_index(drop=True)
    
    total_samples = len(df_features_sorted)
    train_end_idx = int(total_samples * 0.70)
    val_end_idx = int(total_samples * 0.85)
    
    # Divisione dataset
    df_train = df_features_sorted.iloc[:train_end_idx]
    df_val = df_features_sorted.iloc[train_end_idx:val_end_idx]
    df_test = df_features_sorted.iloc[val_end_idx:]
    
    logger.info(
        f"Split Temporale - Totale: {total_samples} campioni | "
        f"Train (70%): {len(df_train)} (fino al {df_train['timestamp'].iloc[-1].strftime('%Y-%m-%d')}) | "
        f"Val (15%): {len(df_val)} (fino al {df_val['timestamp'].iloc[-1].strftime('%Y-%m-%d')}) | "
        f"Test (15%): {len(df_test)} (fino al {df_test['timestamp'].iloc[-1].strftime('%Y-%m-%d')})"
    )
    
    X_train_raw = df_train[feature_cols].values
    y_train = df_train['target'].values
    
    X_val_raw = df_val[feature_cols].values
    y_val = df_val['target'].values
    
    X_test_raw = df_test[feature_cols].values
    y_test = df_test['target'].values
    
    # 4. Feature Scaling (Z-Score)
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0.0] = 1e-8  # Previene la divisione per zero
    
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    
    # 5. Addestramento del modello PyTorch
    input_dim = len(feature_cols)
    model = NeuralNetworkV2(input_dim=input_dim)
    
    logger.info("Avvio del ciclo di addestramento in PyTorch v2...")
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=120,
        batch_size=512,
        early_stopping_rounds=15,
        verbose=True
    )
    
    # 6. Valutazione sul Test Set (Out-of-sample)
    test_probs = model.predict(X_test)
    test_preds = (test_probs > 0.50).astype(int)
    
    test_accuracy = np.mean(test_preds == y_test)
    logger.info(f"=== VALUTAZIONE OUT-OF-SAMPLE (TEST SET V2) ===")
    logger.info(f"Accuracy sul Test Set: {test_accuracy*100:.2f}%")
    
    # Calcolo della baseline (la classe più frequente)
    baseline_acc = max(np.mean(y_test == 1), np.mean(y_test == 0))
    logger.info(f"Baseline Accuracy (Classe più frequente): {baseline_acc*100:.2f}%")
    logger.info(f"Performance rispetto alla baseline: {test_accuracy - baseline_acc:+.2f}%")
    
    # 7. Salvataggio del modello addestrato e dei parametri di scaling
    pesi_dir = Path(__file__).resolve().parent / "pesi"
    pesi_dir.mkdir(exist_ok=True)
    filepath = pesi_dir / save_name
    
    # Salvataggio pesi e parametri
    state = {
        "input_dim": input_dim,
        "feature_cols": feature_cols,
        "scaling_mean": mean.tolist(),
        "scaling_std": std.tolist(),
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict()
    }
    torch.save(state, filepath)
    logger.info(f"Pipeline v2 completata con successo! Modello salvato in: {filepath}")
    logger.info("=======================================================")


if __name__ == "__main__":
    # Avviamo l'addestramento sul ticker AAPL per validare la pipeline velocemente
    run_training_pipeline_v2(tickers=["AAPL"], save_name="neural_model_aapl.pth")
