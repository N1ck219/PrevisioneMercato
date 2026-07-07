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
from models.rete_neurale.v1.model import NeuralNetworkV1

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainPipeline")


def prepare_features_and_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Riceve il DataFrame grezzo unito di prezzi ed indicatori.
    Crea feature scala-invarianti e calcola il target binario direzionale.
    """
    # 1. Copia ed ordinamento cronologico per ogni ticker per evitare sfasamenti temporali
    df = df.copy().sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    
    # 2. Creazione del Target: 1 se il prezzo di domani è superiore alla chiusura di oggi, altrimenti 0
    # Utilizziamo groupby per calcolare lo shift a ritroso sul singolo ticker per prevenire contaminazioni
    df['close_tomorrow'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['close_tomorrow'] > df['close']).astype(int)
    
    # 3. Creazione delle feature scala-invarianti (rapporti relativi ed oscillatori)
    # Questo permette di generalizzare l'addestramento indipendentemente dal valore nominale del prezzo.
    close = df['close']
    
    df['sma_10_ratio'] = df['sma_10'] / close
    df['sma_20_ratio'] = df['sma_20'] / close
    df['sma_50_ratio'] = df['sma_50'] / close
    df['sma_200_ratio'] = df['sma_200'] / close
    df['ema_9_ratio'] = df['ema_9'] / close
    df['ema_21_ratio'] = df['ema_21'] / close
    df['bb_upper_ratio'] = df['bb_upper'] / close
    df['bb_lower_ratio'] = df['bb_lower'] / close
    
    # Normalizzazione degli oscillatori e differenziali
    df['macd_ratio'] = df['macd'] / close
    df['macd_signal_ratio'] = df['macd_signal'] / close
    df['macd_hist_ratio'] = df['macd_hist'] / close
    
    # Volatilità percentuale e volume relativo rispetto alla media
    df['atr_14_ratio'] = df['atr_14'] / close
    df['volume_ratio'] = df['volume'] / df.groupby('ticker')['volume'].transform(lambda x: x.rolling(10).mean())
    
    # La colonna rsi_14 è già scala-invariante (compresa tra 0 e 100), la dividiamo per 100
    df['rsi_14_norm'] = df['rsi_14'] / 100.0
    
    # 4. Definizione delle colonne delle feature
    feature_cols = [
        'sma_10_ratio', 'sma_20_ratio', 'sma_50_ratio', 'sma_200_ratio',
        'ema_9_ratio', 'ema_21_ratio', 'bb_upper_ratio', 'bb_lower_ratio',
        'macd_ratio', 'macd_signal_ratio', 'macd_hist_ratio',
        'atr_14_ratio', 'volume_ratio', 'rsi_14_norm'
    ]
    
    # Rimuoviamo le righe con valori NaN (dovuti alle medie mobili rolling iniziali e allo shift del target)
    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
    
    return df_clean[feature_cols + ['timestamp', 'ticker']], df_clean['target'], feature_cols


def run_training_pipeline(tickers: Optional[List[str]] = None, save_name: str = "neural_model.pth") -> None:
    """
    Esegue l'intero flusso di caricamento, preprocessing, addestramento e salvataggio del modello.
    """
    logger.info("=== INIZIO PIPELINE DI ADDESTRAMENTO MODELLO PYTORCH ===")
    
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
    
    # 2. Preprocessing e ingegneria delle feature
    df_features, y_series, feature_cols = prepare_features_and_targets(df_raw)
    logger.info(f"Preprocessing completato. Campioni puliti utilizzabili: {len(df_features)}.")
    logger.info(f"Feature selezionate per l'addestramento: {feature_cols}")
    
    # 3. Suddivisione temporale cronologica per prevenire lookahead bias
    # Ordiniamo l'intero DataFrame cronologicamente prima dello split
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
    
    # 4. Feature Scaling (Z-Score manuale robusto e autocontenuto)
    # Calcoliamo medie e deviazioni standard esclusivamente sul Train Set per evitare data leakage
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0.0] = 1e-8  # Previene la divisione per zero in caso di feature costanti
    
    # Applicazione dello scaling a tutti i set
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    
    # 5. Addestramento del modello PyTorch
    input_dim = len(feature_cols)
    model = NeuralNetworkV1(input_dim=input_dim)
    
    logger.info("Avvio del ciclo di addestramento in PyTorch...")
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
    logger.info(f"=== VALUTAZIONE OUT-OF-SAMPLE (TEST SET) ===")
    logger.info(f"Accuracy sul Test Set: {test_accuracy*100:.2f}%")
    
    # Calcolo della baseline (la classe più frequente)
    baseline_acc = max(np.mean(y_test == 1), np.mean(y_test == 0))
    logger.info(f"Baseline Accuracy (Classe più frequente): {baseline_acc*100:.2f}%")
    logger.info(f"Performance rispetto alla baseline: {test_accuracy - baseline_acc:+.2f}%")
    
    # 7. Salvataggio del modello addestrato e dei parametri di scaling
    # Salviamo i pesi del modello e i parametri di scaling nello stesso percorso
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
    logger.info(f"Pipeline completata con successo! Modello e parametri salvati in: {filepath}")
    logger.info("=======================================================")


if __name__ == "__main__":
    # Avviamo l'addestramento sul ticker ad alta capitalizzazione AAPL per validare la pipeline velocemente
    run_training_pipeline(tickers=["AAPL"], save_name="neural_model_aapl.pth")
