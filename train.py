import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Silenzia avvisi deprecati o future warnings di PyTorch e altre librerie
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Assicuriamoci che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from database.db_manager import DBManager
from models.rete_neurale.v1.model import NeuralNetworkV1
from models.rete_neurale.v2.model import NeuralNetworkV2
from models.rete_neurale.v3.model import NeuralNetworkV3
from models.rete_neurale.v4.model import NeuralNetworkV4
from models.rete_neurale.v5.model import NeuralNetworkV5
from models.rete_neurale.v6.model import NeuralNetworkV6
from models.rete_neurale.v10.model import NeuralNetworkV10
from models.rete_neurale.v11.model import NeuralNetworkV11
from models.rete_neurale.moe_v1.model import MoEModelV1
from models.gnn.v1.model import SpatioTemporalGNNV1

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UnifiedTrain")


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


def prepare_features_and_targets_v2(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Riceve il DataFrame grezzo unito di prezzi ed indicatori.
    Crea feature avanzate scala-invarianti v2 prendendo ispirazione dai file legacy.
    """
    df = df.copy().sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    df['close_tomorrow'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['close_tomorrow'] > df['close']).astype(int)
    
    df['ret'] = df.groupby('ticker')['close'].pct_change().fillna(0)
    df['vol_ret'] = df.groupby('ticker')['volume'].pct_change().fillna(0)
    
    # Calcolo vettoriale nativo di OBV per ticker (elimina gli avvisi e i ValueError con singolo ticker)
    df['obv_raw'] = (np.sign(df['ret']) * df['volume']).fillna(0)
    df['obv'] = df.groupby('ticker')['obv_raw'].cumsum()
    df['OBV_ret'] = df.groupby('ticker')['obv'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df.drop(columns=['obv_raw', 'obv'], inplace=True)
    
    close = df['close']
    df['RSI_14'] = df['rsi_14'] / 100.0
    df['ATRr_14'] = df['atr_14'] / close
    
    df['Bollinger_%B'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['Bollinger_Width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
    
    df['Dist_SMA200'] = (close - df['sma_200']) / (df['sma_200'] + 1e-9)
    df['Dist_SMA50'] = (close - df['sma_50']) / (df['sma_50'] + 1e-9)
    
    feature_cols = [
        'ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width',
        'ATRr_14', 'Dist_SMA200', 'Dist_SMA50', 'OBV_ret'
    ]
    
    # Sanitizzazione: sostituisci Inf e clipping dei valori estremi per prevenire esplosioni numeriche
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in feature_cols:
        df[col] = df[col].clip(-10.0, 10.0)
    
    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
    return df_clean[feature_cols + ['timestamp', 'ticker']], df_clean['target'], feature_cols


def prepare_features_and_targets_v4(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Riceve il DataFrame grezzo unito di prezzi ed indicatori.
    Crea feature avanzate di borsa scala-invarianti v4 (inclusi ROC, Stochastic, Volume standard deviation, medie veloci).
    """
    df = df.copy().sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    df['close_tomorrow'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['close_tomorrow'] > df['close']).astype(int)
    
    # Feature v2 standard
    df['ret'] = df.groupby('ticker')['close'].pct_change().fillna(0)
    df['vol_ret'] = df.groupby('ticker')['volume'].pct_change().fillna(0)
    
    # OBV
    df['obv_raw'] = (np.sign(df['ret']) * df['volume']).fillna(0)
    df['obv'] = df.groupby('ticker')['obv_raw'].cumsum()
    df['OBV_ret'] = df.groupby('ticker')['obv'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df.drop(columns=['obv_raw', 'obv'], inplace=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    df['RSI_14'] = df['rsi_14'] / 100.0
    df['ATRr_14'] = df['atr_14'] / close
    
    df['Bollinger_%B'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['Bollinger_Width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
    
    df['Dist_SMA200'] = (close - df['sma_200']) / (df['sma_200'] + 1e-9)
    df['Dist_SMA50'] = (close - df['sma_50']) / (df['sma_50'] + 1e-9)
    
    # --- Nuove Feature di Borsa Avanzate (v4) ---
    sma_5 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(5).mean())
    ema_12 = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['SMA_5_ratio'] = sma_5 / close
    df['EMA_12_ratio'] = ema_12 / close
    
    df['ROC_10'] = df.groupby('ticker')['close'].pct_change(10).fillna(0)
    
    low_14 = df.groupby('ticker')['low'].transform(lambda x: x.rolling(14).min())
    high_14 = df.groupby('ticker')['high'].transform(lambda x: x.rolling(14).max())
    df['Stoch_K'] = ((close - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
    
    volume_std_10 = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(10).std()).fillna(0)
    df['Volume_Std_Ratio'] = (volume / (volume_std_10 + 1e-9)).fillna(1.0)
    
    feature_cols = [
        'ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width',
        'ATRr_14', 'Dist_SMA200', 'Dist_SMA50', 'OBV_ret',
        'ROC_10', 'Stoch_K', 'SMA_5_ratio', 'EMA_12_ratio', 'Volume_Std_Ratio'
    ]
    
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in feature_cols:
        df[col] = df[col].clip(-10.0, 10.0)
        
    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
    return df_clean[feature_cols + ['timestamp', 'ticker']], df_clean['target'], feature_cols


def prepare_features_and_targets_v5(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Riceve il DataFrame grezzo unito di prezzi ed indicatori.
    Crea feature avanzate v4 + feature relative al mercato (v5).
    Inoltre restituisce 'close_tomorrow_ret' da usare come pesi nella Profit-Weighted Loss function.
    """
    df = df.copy().sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    df['close_tomorrow'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['close_tomorrow'] > df['close']).astype(int)
    
    # Rendimento effettivo di domani (per la pesatura della Loss)
    df['close_tomorrow_ret'] = ((df['close_tomorrow'] - df['close']) / (df['close'] + 1e-9)).fillna(0.0)
    
    # Feature v2 standard
    df['ret'] = df.groupby('ticker')['close'].pct_change().fillna(0)
    df['vol_ret'] = df.groupby('ticker')['volume'].pct_change().fillna(0)
    
    # OBV
    df['obv_raw'] = (np.sign(df['ret']) * df['volume']).fillna(0)
    df['obv'] = df.groupby('ticker')['obv_raw'].cumsum()
    df['OBV_ret'] = df.groupby('ticker')['obv'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df.drop(columns=['obv_raw', 'obv'], inplace=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    df['RSI_14'] = df['rsi_14'] / 100.0
    df['ATRr_14'] = df['atr_14'] / close
    
    df['Bollinger_%B'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['Bollinger_Width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
    
    df['Dist_SMA200'] = (close - df['sma_200']) / (df['sma_200'] + 1e-9)
    df['Dist_SMA50'] = (close - df['sma_50']) / (df['sma_50'] + 1e-9)
    
    sma_5 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(5).mean())
    ema_12 = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['SMA_5_ratio'] = sma_5 / close
    df['EMA_12_ratio'] = ema_12 / close
    
    df['ROC_10'] = df.groupby('ticker')['close'].pct_change(10).fillna(0)
    
    low_14 = df.groupby('ticker')['low'].transform(lambda x: x.rolling(14).min())
    high_14 = df.groupby('ticker')['high'].transform(lambda x: x.rolling(14).max())
    df['Stoch_K'] = ((close - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
    
    volume_std_10 = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(10).std()).fillna(0)
    df['Volume_Std_Ratio'] = (volume / (volume_std_10 + 1e-9)).fillna(1.0)
    
    # --- Nuove Feature Relative al Mercato (v5) ---
    # Rendimento medio cross-sectionale giornaliero del mercato
    market_daily_ret = df.groupby('timestamp')['ret'].transform('mean')
    df['Market_Relative_Ret'] = df['ret'] - market_daily_ret
    
    # Volume medio cross-sectionale giornaliero del mercato
    market_daily_vol = df.groupby('timestamp')['volume'].transform('mean')
    df['Market_Relative_Volume'] = df['volume'] / (market_daily_vol + 1e-9)
    
    feature_cols = [
        'ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width',
        'ATRr_14', 'Dist_SMA200', 'Dist_SMA50', 'OBV_ret',
        'ROC_10', 'Stoch_K', 'SMA_5_ratio', 'EMA_12_ratio', 'Volume_Std_Ratio',
        'Market_Relative_Ret', 'Market_Relative_Volume'
    ]
    
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in feature_cols:
        df[col] = df[col].clip(-10.0, 10.0)
        
    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
    return df_clean[feature_cols + ['timestamp', 'ticker', 'close_tomorrow_ret']], df_clean['target'], df_clean['close_tomorrow_ret'], feature_cols


def prepare_features_and_targets_v6(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Riceve il DataFrame grezzo unito di prezzi ed indicatori.
    Crea feature avanzate v5 + feature macro di mercato v6:
    - Market_Return: Rendimento cross-sectionale medio del pool.
    - Market_Volatility: Deviazione standard mobile a 20 giorni di Market_Return.
    Restituisce anche 'close_tomorrow_ret' per la loss pesata e le 18 feature_cols finali.
    """
    df = df.copy().sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    df['close_tomorrow'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['close_tomorrow'] > df['close']).astype(int)
    
    # Rendimento effettivo di domani (per la pesatura della Loss)
    df['close_tomorrow_ret'] = ((df['close_tomorrow'] - df['close']) / (df['close'] + 1e-9)).fillna(0.0)
    
    # Feature v2 standard
    df['ret'] = df.groupby('ticker')['close'].pct_change().fillna(0)
    df['vol_ret'] = df.groupby('ticker')['volume'].pct_change().fillna(0)
    
    # OBV
    df['obv_raw'] = (np.sign(df['ret']) * df['volume']).fillna(0)
    df['obv'] = df.groupby('ticker')['obv_raw'].cumsum()
    df['OBV_ret'] = df.groupby('ticker')['obv'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df.drop(columns=['obv_raw', 'obv'], inplace=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    df['RSI_14'] = df['rsi_14'] / 100.0
    df['ATRr_14'] = df['atr_14'] / close
    
    df['Bollinger_%B'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['Bollinger_Width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
    
    df['Dist_SMA200'] = (close - df['sma_200']) / (df['sma_200'] + 1e-9)
    df['Dist_SMA50'] = (close - df['sma_50']) / (df['sma_50'] + 1e-9)
    
    sma_5 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(5).mean())
    ema_12 = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['SMA_5_ratio'] = sma_5 / close
    df['EMA_12_ratio'] = ema_12 / close
    
    df['ROC_10'] = df.groupby('ticker')['close'].pct_change(10).fillna(0)
    
    low_14 = df.groupby('ticker')['low'].transform(lambda x: x.rolling(14).min())
    high_14 = df.groupby('ticker')['high'].transform(lambda x: x.rolling(14).max())
    df['Stoch_K'] = ((close - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
    
    volume_std_10 = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(10).std()).fillna(0)
    df['Volume_Std_Ratio'] = (volume / (volume_std_10 + 1e-9)).fillna(1.0)
    
    # --- Feature Relative al Mercato (v5) ---
    market_daily_ret = df.groupby('timestamp')['ret'].transform('mean')
    df['Market_Relative_Ret'] = df['ret'] - market_daily_ret
    
    market_daily_vol = df.groupby('timestamp')['volume'].transform('mean')
    df['Market_Relative_Volume'] = df['volume'] / (market_daily_vol + 1e-9)
    
    # --- Nuove Feature Macro di Mercato (v6) ---
    df['Market_Return'] = market_daily_ret
    # Volatilità rolling storica a 20 giorni dei rendimenti del mercato (VIX proxy)
    # Raggruppiamo temporaneamente per ticker per calcolare il rolling orizzontale in modo sicuro
    df['Market_Volatility'] = df.groupby('ticker')['Market_Return'].transform(lambda x: x.rolling(20).std()).fillna(0.0)
    
    feature_cols = [
        'ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width',
        'ATRr_14', 'Dist_SMA200', 'Dist_SMA50', 'OBV_ret',
        'ROC_10', 'Stoch_K', 'SMA_5_ratio', 'EMA_12_ratio', 'Volume_Std_Ratio',
        'Market_Relative_Ret', 'Market_Relative_Volume',
        'Market_Return', 'Market_Volatility'
    ]
    
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in feature_cols:
        df[col] = df[col].clip(-10.0, 10.0)
        
    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
    return df_clean[feature_cols + ['timestamp', 'ticker', 'close_tomorrow_ret']], df_clean['target'], df_clean['close_tomorrow_ret'], feature_cols


def create_temporal_sequences_v5(
    df_scaled: pd.DataFrame, 
    feature_cols: List[str], 
    lookback: int,
    train_cutoff_time: pd.Timestamp,
    val_cutoff_time: pd.Timestamp
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce sequenze 3D per LSTM/Transformer v5 a partire dal DataFrame intero scalato, raggruppando per ticker
    e smistando le sequenze in train/val/test in base alla data dell'ultima barra (senza lookahead).
    Ritorna anche i rendimenti di domani per l'addestramento pesato.
    """
    X_train_list, y_train_list, ret_train_list = [], [], []
    X_val_list, y_val_list, ret_val_list = [], [], []
    X_test_list, y_test_list, ret_test_list = [], [], []
    
    for ticker, group in tqdm(df_scaled.groupby('ticker'), desc="Creazione sequenze temporali v5", leave=False):
        group_sorted = group.sort_values('timestamp').reset_index(drop=True)
        if len(group_sorted) < lookback:
            continue
            
        features = group_sorted[feature_cols].values
        targets = group_sorted['target'].values
        ret_tomorrows = group_sorted['close_tomorrow_ret'].values
        timestamps = group_sorted['timestamp'].values
        
        for i in range(lookback - 1, len(group_sorted)):
            seq = features[i - lookback + 1 : i + 1]
            target = targets[i]
            ret_tomorrow = ret_tomorrows[i]
            time_end = pd.Timestamp(timestamps[i])
            
            if time_end <= train_cutoff_time:
                X_train_list.append(seq)
                y_train_list.append(target)
                ret_train_list.append(ret_tomorrow)
            elif time_end <= val_cutoff_time:
                X_val_list.append(seq)
                y_val_list.append(target)
                ret_val_list.append(ret_tomorrow)
            else:
                X_test_list.append(seq)
                y_test_list.append(target)
                ret_test_list.append(ret_tomorrow)
                
    return (
        np.array(X_train_list, dtype=np.float32), np.array(y_train_list), np.array(ret_train_list, dtype=np.float32),
        np.array(X_val_list, dtype=np.float32), np.array(y_val_list), np.array(ret_val_list, dtype=np.float32),
        np.array(X_test_list, dtype=np.float32), np.array(y_test_list), np.array(ret_test_list, dtype=np.float32)
    )


def create_temporal_sequences(
    df_scaled: pd.DataFrame, 
    feature_cols: List[str], 
    lookback: int,
    train_cutoff_time: pd.Timestamp,
    val_cutoff_time: pd.Timestamp
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce sequenze 3D per LSTM a partire dal DataFrame intero scalato, raggruppando per ticker
    e smistando le sequenze in train/val/test in base alla data dell'ultima barra (senza lookahead).
    """
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    # Raggruppa per ticker
    for ticker, group in tqdm(df_scaled.groupby('ticker'), desc="Creazione sequenze temporali", leave=False):
        group_sorted = group.sort_values('timestamp').reset_index(drop=True)
        if len(group_sorted) < lookback:
            continue
            
        features = group_sorted[feature_cols].values
        targets = group_sorted['target'].values
        timestamps = group_sorted['timestamp'].values
        
        for i in range(lookback - 1, len(group_sorted)):
            seq = features[i - lookback + 1 : i + 1]
            target = targets[i]
            time_end = pd.Timestamp(timestamps[i])
            
            if time_end <= train_cutoff_time:
                X_train_list.append(seq)
                y_train_list.append(target)
            elif time_end <= val_cutoff_time:
                X_val_list.append(seq)
                y_val_list.append(target)
            else:
                X_test_list.append(seq)
                y_test_list.append(target)
                
    return (
        np.array(X_train_list), np.array(y_train_list),
        np.array(X_val_list), np.array(y_val_list),
        np.array(X_test_list), np.array(y_test_list)
    )


def prepare_features_and_targets_v11(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
    """
    Riceve il DataFrame grezzo unito di prezzi ed indicatori.
    Crea feature avanzate v6 + nuove feature macro di mercato v11 caricate dal database:
    - VIX close (fear sentiment)
    - TNX close (tassi d'interesse a 10 anni)
    - DXY close (indice del dollaro)
    - SPY distance from SMA 200 (trend primario mercato)
    - QQQ distance from SMA 200
    - SPY daily return
    - VIX daily return
    
    Restituisce:
    - DataFrame pulito delle feature
    - target (1/0)
    - close_tomorrow_ret (per loss pesata)
    - is_crash_regime (1/0 per loss asimmetrica di regime)
    - feature_cols finali (25 feature)
    """
    db = DBManager()
    macro_tickers = ["^VIX", "^TNX", "DX-Y.NYB", "SPY", "QQQ"]
    macro_dfs = {}
    
    for ticker in macro_tickers:
        try:
            q = f"""
                SELECT o.timestamp, o.close, i.sma_200 
                FROM ohlcv o
                LEFT JOIN indicators i 
                  ON o.ticker = i.ticker AND o.timestamp = i.timestamp
                WHERE o.ticker = '{ticker}'
            """
            m_df = db.execute_query(q)
            if not m_df.empty:
                m_df['timestamp'] = pd.to_datetime(m_df['timestamp'])
                m_df = m_df.rename(columns={
                    'close': f'{ticker}_close',
                    'sma_200': f'{ticker}_sma_200'
                })
                macro_dfs[ticker] = m_df
        except Exception as e:
            logger.warning(f"Impossibile caricare dati macro per {ticker}: {e}")
            
    df = df.copy().sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['close_tomorrow'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['close_tomorrow'] > df['close']).astype(int)
    df['close_tomorrow_ret'] = ((df['close_tomorrow'] - df['close']) / (df['close'] + 1e-9)).fillna(0.0)
    
    df['ret'] = df.groupby('ticker')['close'].pct_change().fillna(0)
    df['vol_ret'] = df.groupby('ticker')['volume'].pct_change().fillna(0)
    
    df['obv_raw'] = (np.sign(df['ret']) * df['volume']).fillna(0)
    df['obv'] = df.groupby('ticker')['obv_raw'].cumsum()
    df['OBV_ret'] = df.groupby('ticker')['obv'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df.drop(columns=['obv_raw', 'obv'], inplace=True)
    
    close = df['close']
    df['RSI_14'] = df['rsi_14'] / 100.0
    df['ATRr_14'] = df['atr_14'] / close
    
    df['Bollinger_%B'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['Bollinger_Width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
    
    df['Dist_SMA200'] = (close - df['sma_200']) / (df['sma_200'] + 1e-9)
    df['Dist_SMA50'] = (close - df['sma_50']) / (df['sma_50'] + 1e-9)
    
    sma_5 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(5).mean())
    ema_12 = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['SMA_5_ratio'] = sma_5 / close
    df['EMA_12_ratio'] = ema_12 / close
    
    df['ROC_10'] = df.groupby('ticker')['close'].pct_change(10).fillna(0)
    
    low_14 = df.groupby('ticker')['low'].transform(lambda x: x.rolling(14).min())
    high_14 = df.groupby('ticker')['high'].transform(lambda x: x.rolling(14).max())
    df['Stoch_K'] = ((close - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
    
    volume_std_10 = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(10).std()).fillna(0)
    df['Volume_Std_Ratio'] = (df['volume'] / (volume_std_10 + 1e-9)).fillna(1.0)
    
    market_daily_ret = df.groupby('timestamp')['ret'].transform('mean')
    df['Market_Relative_Ret'] = df['ret'] - market_daily_ret
    
    market_daily_vol = df.groupby('timestamp')['volume'].transform('mean')
    df['Market_Relative_Volume'] = df['volume'] / (market_daily_vol + 1e-9)
    
    df['Market_Return'] = market_daily_ret
    df['Market_Volatility'] = df.groupby('ticker')['Market_Return'].transform(lambda x: x.rolling(20).std()).fillna(0.0)
    
    for ticker, m_df in macro_dfs.items():
        if not m_df.empty:
            m_df = m_df.drop_duplicates(subset=['timestamp'])
            df = pd.merge(df, m_df, on='timestamp', how='left')
            
    for ticker in macro_tickers:
        close_col = f'{ticker}_close'
        if close_col in df.columns:
            df[close_col] = df.groupby('ticker')[close_col].ffill().bfill().fillna(0.0)
        sma_col = f'{ticker}_sma_200'
        if sma_col in df.columns:
            df[sma_col] = df.groupby('ticker')[sma_col].ffill().bfill().fillna(0.0)
            
    df['VIX_close'] = df['^VIX_close'] if '^VIX_close' in df.columns else 15.0
    df['TNX_close'] = df['^TNX_close'] if '^TNX_close' in df.columns else 4.0
    df['DXY_close'] = df['DX-Y.NYB_close'] if 'DX-Y.NYB_close' in df.columns else 100.0
    
    if 'SPY_close' in df.columns and 'SPY_sma_200' in df.columns:
        df['SPY_dist_sma200'] = ((df['SPY_close'] - df['SPY_sma_200']) / (df['SPY_sma_200'] + 1e-9)).fillna(0.0)
    else:
        df['SPY_dist_sma200'] = 0.0
        
    if 'QQQ_close' in df.columns and 'QQQ_sma_200' in df.columns:
        df['QQQ_dist_sma200'] = ((df['QQQ_close'] - df['QQQ_sma_200']) / (df['QQQ_sma_200'] + 1e-9)).fillna(0.0)
    else:
        df['QQQ_dist_sma200'] = 0.0
        
    df['SPY_daily_ret'] = df.groupby('ticker')['SPY_close'].pct_change().fillna(0.0) if 'SPY_close' in df.columns else 0.0
    df['VIX_daily_ret'] = df.groupby('ticker')['^VIX_close'].pct_change().fillna(0.0) if '^VIX_close' in df.columns else 0.0
    
    df['is_crash_regime'] = ((df['VIX_close'] > 25.0) | (df['SPY_dist_sma200'] < -0.05)).astype(int)
    
    feature_cols = [
        'ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width',
        'ATRr_14', 'Dist_SMA200', 'Dist_SMA50', 'OBV_ret',
        'ROC_10', 'Stoch_K', 'SMA_5_ratio', 'EMA_12_ratio', 'Volume_Std_Ratio',
        'Market_Relative_Ret', 'Market_Relative_Volume',
        'Market_Return', 'Market_Volatility',
        'VIX_close', 'TNX_close', 'DXY_close', 'SPY_dist_sma200', 'QQQ_dist_sma200',
        'SPY_daily_ret', 'VIX_daily_ret'
    ]
    
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in feature_cols:
        df[col] = df[col].clip(-10.0, 10.0)
        
    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
    
    return (
        df_clean[feature_cols + ['timestamp', 'ticker', 'close_tomorrow_ret', 'is_crash_regime']], 
        df_clean['target'], 
        df_clean['close_tomorrow_ret'], 
        df_clean['is_crash_regime'], 
        feature_cols
    )


def create_temporal_sequences_v11(
    df_scaled: pd.DataFrame, 
    feature_cols: List[str], 
    lookback: int,
    train_cutoff_time: pd.Timestamp,
    val_cutoff_time: pd.Timestamp
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce sequenze 3D per V11 con lookback, train/val/test split.
    """
    X_train_list, y_train_list, ret_train_list, crash_train_list = [], [], [], []
    X_val_list, y_val_list, ret_val_list, crash_val_list = [], [], [], []
    X_test_list, y_test_list, ret_test_list, crash_test_list = [], [], [], []
    
    for ticker, group in tqdm(df_scaled.groupby('ticker'), desc="Creazione sequenze temporali v11", leave=False):
        group_sorted = group.sort_values('timestamp').reset_index(drop=True)
        if len(group_sorted) < lookback:
            continue
            
        features = group_sorted[feature_cols].values
        targets = group_sorted['target'].values
        ret_tomorrows = group_sorted['close_tomorrow_ret'].values
        crashes = group_sorted['is_crash_regime'].values
        timestamps = group_sorted['timestamp'].values
        
        for i in range(lookback - 1, len(group_sorted)):
            seq = features[i - lookback + 1 : i + 1]
            target = targets[i]
            ret_tomorrow = ret_tomorrows[i]
            crash = crashes[i]
            time_end = pd.Timestamp(timestamps[i])
            
            if time_end <= train_cutoff_time:
                X_train_list.append(seq)
                y_train_list.append(target)
                ret_train_list.append(ret_tomorrow)
                crash_train_list.append(crash)
            elif time_end <= val_cutoff_time:
                X_val_list.append(seq)
                y_val_list.append(target)
                ret_val_list.append(ret_tomorrow)
                crash_val_list.append(crash)
            else:
                X_test_list.append(seq)
                y_test_list.append(target)
                ret_test_list.append(ret_tomorrow)
                crash_test_list.append(crash)
                
    return (
        np.array(X_train_list, dtype=np.float32), np.array(y_train_list), np.array(ret_train_list, dtype=np.float32), np.array(crash_train_list, dtype=np.float32),
        np.array(X_val_list, dtype=np.float32), np.array(y_val_list), np.array(ret_val_list, dtype=np.float32), np.array(crash_val_list, dtype=np.float32),
        np.array(X_test_list, dtype=np.float32), np.array(y_test_list), np.array(ret_test_list, dtype=np.float32), np.array(crash_test_list, dtype=np.float32)
    )


def prepare_features_and_targets_moe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, List[str]]:
    """
    Riceve il DataFrame grezzo unito di prezzi ed indicatori.
    Crea feature avanzate v11 + regime_label per moe_v1:
    - Regime 0 (Bull): VIX <= 18.0 e SPY_dist_sma200 >= 0.0
    - Regime 1 (Bear/Crash): VIX >= 25.0 o SPY_dist_sma200 < -0.05
    - Regime 2 (Lateral): Altrimenti
    """
    df_clean, target, ret, crash, feature_cols = prepare_features_and_targets_v11(df)
    
    spy_dist = df_clean['SPY_dist_sma200']
    vix = df_clean['VIX_close']
    
    regime = np.zeros(len(df_clean), dtype=np.int32)
    regime[:] = 2  # Lateral di default
    regime[(vix <= 18.0) & (spy_dist >= 0.0)] = 0  # Bull
    regime[(vix >= 25.0) | (spy_dist < -0.05)] = 1  # Bear/Crash
    
    df_clean['regime_label'] = regime
    
    return (
        df_clean[feature_cols + ['timestamp', 'ticker', 'close_tomorrow_ret', 'is_crash_regime', 'regime_label']], 
        target, 
        ret, 
        crash, 
        df_clean['regime_label'], 
        feature_cols
    )


def create_temporal_sequences_moe(
    df_scaled: pd.DataFrame, 
    feature_cols: List[str], 
    lookback: int,
    train_cutoff_time: pd.Timestamp,
    val_cutoff_time: pd.Timestamp
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce sequenze 3D per moe_v1 con lookback, train/val/test split.
    Ritorna anche i label dei regimi (0, 1, 2).
    """
    X_train_list, y_train_list, ret_train_list, crash_train_list, reg_train_list = [], [], [], [], []
    X_val_list, y_val_list, ret_val_list, crash_val_list, reg_val_list = [], [], [], [], []
    X_test_list, y_test_list, ret_test_list, crash_test_list, reg_test_list = [], [], [], [], []
    
    for ticker, group in tqdm(df_scaled.groupby('ticker'), desc="Creazione sequenze temporali moe", leave=False):
        group_sorted = group.sort_values('timestamp').reset_index(drop=True)
        if len(group_sorted) < lookback:
            continue
            
        features = group_sorted[feature_cols].values
        targets = group_sorted['target'].values
        ret_tomorrows = group_sorted['close_tomorrow_ret'].values
        crashes = group_sorted['is_crash_regime'].values
        regimes = group_sorted['regime_label'].values
        timestamps = group_sorted['timestamp'].values
        
        for i in range(lookback - 1, len(group_sorted)):
            seq = features[i - lookback + 1 : i + 1]
            target = targets[i]
            ret_tomorrow = ret_tomorrows[i]
            crash = crashes[i]
            regime = regimes[i]
            time_end = pd.Timestamp(timestamps[i])
            
            if time_end <= train_cutoff_time:
                X_train_list.append(seq)
                y_train_list.append(target)
                ret_train_list.append(ret_tomorrow)
                crash_train_list.append(crash)
                reg_train_list.append(regime)
            elif time_end <= val_cutoff_time:
                X_val_list.append(seq)
                y_val_list.append(target)
                ret_val_list.append(ret_tomorrow)
                crash_val_list.append(crash)
                reg_val_list.append(regime)
            else:
                X_test_list.append(seq)
                y_test_list.append(target)
                ret_test_list.append(ret_tomorrow)
                crash_test_list.append(crash)
                reg_test_list.append(regime)
                
    return (
        np.array(X_train_list, dtype=np.float32), np.array(y_train_list), np.array(ret_train_list, dtype=np.float32), np.array(crash_train_list, dtype=np.float32), np.array(reg_train_list, dtype=np.float32),
        np.array(X_val_list, dtype=np.float32), np.array(y_val_list), np.array(ret_val_list, dtype=np.float32), np.array(crash_val_list, dtype=np.float32), np.array(reg_val_list, dtype=np.float32),
        np.array(X_test_list, dtype=np.float32), np.array(y_test_list), np.array(ret_test_list, dtype=np.float32), np.array(crash_test_list, dtype=np.float32), np.array(reg_test_list, dtype=np.float32)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Piattaforma Trading - Script Unificato di Addestramento Modelli ML"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="nn_v1",
        choices=["nn_v1", "nn_v2", "nn_v3", "nn_v4", "nn_v5", "nn_v6", "nn_v10", "nn_v11", "moe_v1", "gnn_v1"],
        help="Il tipo di modello da allenare (default: nn_v1)."
    )
    
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Lunghezza della lookback window per LSTM (v3) (default: 30)."
    )
    
    parser.add_argument(
        "-t", "--tickers",
        type=str,
        help="Lista di ticker separati da virgola su cui allenare (es. AAPL,MSFT,TSLA). Se omesso, allena su TUTTE le azioni nel DB."
    )
    
    parser.add_argument(
        "-s", "--save_name",
        type=str,
        default="neural_model.pth",
        help="Nome con cui salvare i pesi del modello (default: neural_model.pth)."
    )
    
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=120,
        help="Numero massimo di epoche per l'addestramento (default: 120)."
    )
    
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=512,
        help="Dimensione del batch per l'addestramento (default: 512)."
    )

    parser.add_argument(
        "-p", "--patience",
        type=int,
        default=15,
        help="Numero di epoche di pazienza (patience) per l'Early Stopping (default: 15)."
    )
    
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Se attivo, riprende l'addestramento caricando i pesi e l'ottimizzatore dal file esistente."
    )
    
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default="2024-04-03",
        help="Data di cutoff per escludere il periodo di backtest dall'addestramento e validazione (default: 2024-04-03). Se vuota o 'None', usa lo split index standard."
    )


    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Dimensione delle feature interne (d_model) del Transformer per le reti v6/v10 (default: 64)."
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Numero di teste di attenzione (nhead) per le reti Transformer v6/v10 (default: 4)."
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Numero di layers dell'encoder Transformer per le reti v6/v10 (default: 2)."
    )
    
    parser.add_argument(
        "--penalty_factor",
        type=float,
        default=1.5,
        help="Fattore di penalità per falsi LONG durante i crash (solo nn_v11). Default: 1.5 (originariamente 3.0)."
    )
    
    args = parser.parse_args()
    
    if args.model == "gnn_v1" and args.batch_size > 32:
        logger.warning(f"Batch size {args.batch_size} è troppo grande per la GNN Spazio-Temporale (richiederebbe circa 15GB di VRAM). Auto-riduzione a 16 per prevenire CUDA Out of Memory.")
        args.batch_size = 16
        
    logger.info("=== AVVIO UNIFICATO PIPELINE DI ADDESTRAMENTO ===")
    
    db = DBManager()
    
    # 1. Determinazione dei ticker su cui allenare
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        logger.info(f"Configurato addestramento personalizzato su {len(tickers)} ticker specificati: {tickers}")
    else:
        logger.info("Nessun ticker specificato. Recupero di tutti i ticker disponibili a DB...")
        try:
            tickers = db.execute_query("SELECT DISTINCT ticker FROM ohlcv")['ticker'].tolist()
            if not tickers:
                raise ValueError("Nessun ticker trovato nella tabella ohlcv del database.")
            logger.info(f"Rilevati con successo {len(tickers)} ticker unici a database. Addestramento avviato su TUTTO il paniere!")
        except Exception as e:
            logger.error(f"Impossibile estrarre i ticker dal database: {e}")
            sys.exit(1)
            
    # 2. Caricamento dati storici e indicatori tecnici da SQLite
    logger.info("Caricamento dati storici e indicatori tecnici in corso dal DB SQLite...")
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
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
        WHERE o.ticker IN ({placeholders})
    """
    
    try:
        df_raw = db.execute_query(query, tuple(tickers))
        if df_raw.empty:
            logger.error("Nessun dato trovato nel database per i ticker selezionati. Addestramento annullato.")
            sys.exit(1)
        logger.info(f"Caricamento completato con successo: {len(df_raw)} righe totali caricate.")
    except Exception as e:
        logger.error(f"Errore durante l'estrazione dati dal database: {e}")
        sys.exit(1)
        
    # 3. Preprocessing e ingegneria delle feature scala-invarianti
    logger.info(f"Calcolo delle feature scala-invarianti per {args.model}...")
    if args.model == "moe_v1":
        df_features, y_series, tomorrow_returns_series, crash_regimes_series, regime_labels_series, feature_cols = prepare_features_and_targets_moe(df_raw)
    elif args.model == "nn_v11":
        df_features, y_series, tomorrow_returns_series, crash_regimes_series, feature_cols = prepare_features_and_targets_v11(df_raw)
    elif args.model in ["nn_v6", "nn_v10"]:
        df_features, y_series, tomorrow_returns_series, feature_cols = prepare_features_and_targets_v6(df_raw)
    elif args.model == "nn_v5":
        df_features, y_series, tomorrow_returns_series, feature_cols = prepare_features_and_targets_v5(df_raw)
    elif args.model == "nn_v4":
        df_features, y_series, feature_cols = prepare_features_and_targets_v4(df_raw)
    elif args.model in ["nn_v2", "nn_v3"]:
        df_features, y_series, feature_cols = prepare_features_and_targets_v2(df_raw)
    else:
        df_features, y_series, feature_cols = prepare_features_and_targets(df_raw)
    logger.info(f"Preprocessing completato. Campioni puliti utilizzabili: {len(df_features)}.")
    logger.info(f"Feature selezionate per l'addestramento: {feature_cols}")
    
    # 4. Suddivisione temporale cronologica con Purged & Embargoed Cross-Validation (B2)
    df_features['target'] = y_series
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
    df_features_sorted = df_features.sort_values('timestamp').reset_index(drop=True)
    
    lookback_val = args.lookback if args.model in ["nn_v3", "nn_v4", "nn_v5", "nn_v6", "nn_v10", "nn_v11", "moe_v1"] else 30
    
    if args.cutoff_date and args.cutoff_date.lower() != "none":
        cutoff_ts = pd.Timestamp(args.cutoff_date)
        df_pre_cutoff = df_features_sorted[df_features_sorted['timestamp'] < cutoff_ts]
        df_test = df_features_sorted[df_features_sorted['timestamp'] >= cutoff_ts]
        
        if len(df_pre_cutoff) == 0:
            raise ValueError(f"Nessun dato trovato prima della data di cutoff {args.cutoff_date}!")
        if len(df_test) == 0:
            raise ValueError(f"Nessun dato trovato dopo o a partire dalla data di cutoff {args.cutoff_date}!")
            
        n_pre_cutoff = len(df_pre_cutoff)
        train_end_idx = int(n_pre_cutoff * 0.80)
        
        df_train = df_pre_cutoff.iloc[:train_end_idx]
        df_val = df_pre_cutoff.iloc[train_end_idx + lookback_val :]
        
        if len(df_train) == 0 or len(df_val) == 0:
            raise ValueError("I dati pre-cutoff sono troppo pochi per effettuare lo split Train/Validation!")
            
        logger.info(
            f"Split Temporale con Cutoff Rigido ({args.cutoff_date}) -\n"
            f"  Train (80% pre-cutoff): {len(df_train)} campioni (fino al {df_train['timestamp'].iloc[-1].strftime('%Y-%m-%d')}) |\n"
            f"  Val (20% pre-cutoff):   {len(df_val)} campioni (dal {df_val['timestamp'].iloc[0].strftime('%Y-%m-%d')} al {df_val['timestamp'].iloc[-1].strftime('%Y-%m-%d')}) |\n"
            f"  Test (post-cutoff):     {len(df_test)} campioni (dal {df_test['timestamp'].iloc[0].strftime('%Y-%m-%d')} al {df_test['timestamp'].iloc[-1].strftime('%Y-%m-%d')})"
        )
    else:
        total_samples = len(df_features_sorted)
        train_end_idx = int(total_samples * 0.70)
        val_end_idx = int(total_samples * 0.85)
        
        # Applicazione dell'embargo cronologico per eliminare leakage informativi di sequenza
        df_train = df_features_sorted.iloc[:train_end_idx]
        df_val = df_features_sorted.iloc[train_end_idx + lookback_val : val_end_idx]
        df_test = df_features_sorted.iloc[val_end_idx + lookback_val :]
        
        logger.info(
            f"Split Temporale Standard (Purged & Embargoed CV) - Totale: {total_samples} campioni | "
            f"Train (70%): {len(df_train)} (fino al {df_train['timestamp'].iloc[-1].strftime('%Y-%m-%d')}) | "
            f"Val (15%): {len(df_val)} (dal {df_val['timestamp'].iloc[0].strftime('%Y-%m-%d')} al {df_val['timestamp'].iloc[-1].strftime('%Y-%m-%d')}) | "
            f"Test (15%): {len(df_test)} (dal {df_test['timestamp'].iloc[0].strftime('%Y-%m-%d')} al {df_test['timestamp'].iloc[-1].strftime('%Y-%m-%d')})"
        )
    
    X_train_raw = df_train[feature_cols].values
    
    # 5. Feature Scaling (Z-Score manuale robusto e autocontenuto)
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0.0] = 1e-8  # Previene la divisione per zero
    
    # Se il modello è sequenziale (nn_v3, nn_v4, nn_v5, nn_v6, nn_v10, nn_v11, moe_v1, gnn_v1), creiamo le sequenze 3D
    if args.model in ["nn_v3", "nn_v4", "nn_v5", "nn_v6", "nn_v10", "nn_v11", "moe_v1", "gnn_v1"]:
        logger.info(f"Creazione delle sequenze temporali per {args.model.upper()} con lookback = {args.lookback}...")
        train_cutoff_time = pd.Timestamp(df_train['timestamp'].iloc[-1])
        val_cutoff_time = pd.Timestamp(df_val['timestamp'].iloc[-1])
        
        # Applichiamo lo scaling Z-Score all'intero DataFrame per poi spezzettarlo in sequenze 3D coerenti
        df_scaled = df_features_sorted.copy()
        df_scaled[feature_cols] = (df_scaled[feature_cols] - mean) / std
        
        if args.model == "moe_v1":
            X_train, y_train, ret_train, crash_train, reg_train, X_val, y_val, ret_val, crash_val, reg_val, X_test, y_test, ret_test, crash_test, reg_test = create_temporal_sequences_moe(
                df_scaled, feature_cols, args.lookback, train_cutoff_time, val_cutoff_time
            )
        elif args.model == "nn_v11":
            X_train, y_train, ret_train, crash_train, X_val, y_val, ret_val, crash_val, X_test, y_test, ret_test, crash_test = create_temporal_sequences_v11(
                df_scaled, feature_cols, args.lookback, train_cutoff_time, val_cutoff_time
            )
        elif args.model in ["nn_v5", "nn_v6", "nn_v10"]:
            X_train, y_train, ret_train, X_val, y_val, ret_val, X_test, y_test, ret_test = create_temporal_sequences_v5(
                df_scaled, feature_cols, args.lookback, train_cutoff_time, val_cutoff_time
            )
        elif args.model == "gnn_v1":
            # For GNN, we align cross-sectionally. We can reconstruct spatiotemporal tensors (B, N, L, F)
            # Find unique tickers and timestamps
            tickers_list = sorted(df_scaled['ticker'].unique())
            
            # Align sequences on a unified grid per split: Train, Val, Test
            def build_gnn_grid_dataset(df_split):
                if df_split.empty:
                    return np.empty((0, len(tickers_list), args.lookback, len(feature_cols))), np.empty((0, len(tickers_list)))
                
                timestamps = sorted(df_split['timestamp'].unique())
                if len(timestamps) < args.lookback:
                    return np.empty((0, len(tickers_list), args.lookback, len(feature_cols))), np.empty((0, len(tickers_list)))
                
                # Create a multi-index representing the full grid of (timestamp x ticker)
                full_index = pd.MultiIndex.from_product([timestamps, tickers_list], names=['timestamp', 'ticker'])
                
                # Reindex df_split to the full grid, filling missing values
                df_grid = df_split.set_index(['timestamp', 'ticker']).reindex(full_index)
                
                # Reshape features and targets
                T = len(timestamps)
                N = len(tickers_list)
                features_grid = df_grid[feature_cols].fillna(0.0).values.reshape(T, N, len(feature_cols))
                targets_grid = df_grid['target'].fillna(0).astype(int).values.reshape(T, N)
                
                # Filter out snapshots with less than 50% non-NaN data in the original df_split
                ticker_counts = df_split.groupby('timestamp')['ticker'].nunique()
                valid_timestamps = ticker_counts[ticker_counts >= N * 0.5].index
                
                # Filter the grid arrays
                valid_timestamps_set = set(valid_timestamps)
                valid_mask = np.array([ts in valid_timestamps_set for ts in timestamps])
                grid_data = features_grid[valid_mask]
                grid_targets = targets_grid[valid_mask]
                
                if len(grid_data) < args.lookback:
                    return np.empty((0, len(tickers_list), args.lookback, len(feature_cols))), np.empty((0, len(tickers_list)))
                
                X_win = []
                y_win = []
                for i in range(len(grid_data) - args.lookback):
                    X_win.append(grid_data[i : i + args.lookback])
                    y_win.append(grid_targets[i + args.lookback])
                
                # Transpose to (B, N, L, F)
                X_arr = np.array(X_win).transpose(0, 2, 1, 3)
                y_arr = np.array(y_win)
                return X_arr, y_arr
            
            X_train, y_train = build_gnn_grid_dataset(df_scaled[df_scaled['timestamp'] <= train_cutoff_time])
            X_val, y_val = build_gnn_grid_dataset(df_scaled[(df_scaled['timestamp'] > train_cutoff_time) & (df_scaled['timestamp'] <= val_cutoff_time)])
            X_test, y_test = build_gnn_grid_dataset(df_scaled[df_scaled['timestamp'] > val_cutoff_time])
            
            # Compute pearson correlation for Adjacency matrix using training returns (feature_cols[0])
            if len(X_train) > 0 and len(tickers_list) > 1:
                returns_matrix = X_train[:, :, -1, 0].T # (N, B)
                corr = np.corrcoef(returns_matrix)
                corr = np.nan_to_num(corr)
                # Threshold at 0.3 correlation
                adj_matrix = (np.abs(corr) > 0.3).astype(float)
                np.fill_diagonal(adj_matrix, 1.0)
            else:
                adj_matrix = np.eye(len(tickers_list))
            
            # Keep adjacency matrix as an instance attribute of the model wrapper later
            args.gnn_adj = torch.tensor(adj_matrix, dtype=torch.float32)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = create_temporal_sequences(
                df_scaled, feature_cols, args.lookback, train_cutoff_time, val_cutoff_time
            )
        
        logger.info(
            f"Sequenze generate - Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}"
        )
    else:
        # Procedura standard per MLP 2D (nn_v1, nn_v2)
        X_train = (X_train_raw - mean) / std
        X_val = (df_val[feature_cols].values - mean) / std
        X_test = (df_test[feature_cols].values - mean) / std
        y_train = df_train['target'].values
        y_val = df_val['target'].values
        y_test = df_test['target'].values
    
    # 6. Addestramento del modello PyTorch selezionato
    input_dim = len(feature_cols)
    
    if args.model == "moe_v1":
        model = MoEModelV1(
            input_dim=input_dim, 
            lookback=args.lookback,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            penalty_factor=args.penalty_factor
        )
        pesi_dir = config.BASE_DIR / "models" / "rete_neurale" / "moe_v1" / "pesi"
    elif args.model == "nn_v11":
        model = NeuralNetworkV11(
            input_dim=input_dim, 
            lookback=args.lookback,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            penalty_factor=args.penalty_factor
        )
        pesi_dir = config.BASE_DIR / "models" / "rete_neurale" / "v11" / "pesi"
    elif args.model == "nn_v10":
        model = NeuralNetworkV10(
            input_dim=input_dim, 
            lookback=args.lookback,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers
        )
        pesi_dir = config.BASE_DIR / "models" / "rete_neurale" / "v10" / "pesi"
    elif args.model == "nn_v6":
        model = NeuralNetworkV6(
            input_dim=input_dim, 
            lookback=args.lookback,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers
        )
        pesi_dir = config.BASE_DIR / "models" / "rete_neurale" / "v6" / "pesi"
    elif args.model == "nn_v5":
        model = NeuralNetworkV5(input_dim=input_dim, lookback=args.lookback)
        pesi_dir = config.BASE_DIR / "models" / "rete_neurale" / "v5" / "pesi"
    elif args.model == "nn_v4":
        model = NeuralNetworkV4(input_dim=input_dim, lookback=args.lookback)
        pesi_dir = config.BASE_DIR / "models" / "rete_neurale" / "v4" / "pesi"
    elif args.model == "nn_v3":
        model = NeuralNetworkV3(input_dim=input_dim, lookback=args.lookback)
        pesi_dir = config.BASE_DIR / "models" / "rete_neurale" / "v3" / "pesi"
    elif args.model == "nn_v2":
        model = NeuralNetworkV2(input_dim=input_dim)
        pesi_dir = config.BASE_DIR / "models" / "rete_neurale" / "v2" / "pesi"
    elif args.model == "gnn_v1":
        model = SpatioTemporalGNNV1(input_dim=input_dim)
        pesi_dir = config.BASE_DIR / "models" / "gnn" / "v1" / "pesi"
    else:
        model = NeuralNetworkV1(input_dim=input_dim)
        pesi_dir = config.BASE_DIR / "models" / "rete_neurale" / "v1" / "pesi"
        
    filepath = pesi_dir / args.save_name
    if args.resume:
        if filepath.exists():
            logger.info(f"Rilevato flag --resume. Caricamento modello esistente da: {filepath}...")
            model.load(str(filepath))
        else:
            logger.warning(f"Flag --resume specificato, ma file {filepath} non trovato. L'addestramento partirà da zero.")
    
    logger.info(f"Avvio dell'addestramento del modello PyTorch '{args.model}' (Patience: {args.patience})...")
    if args.model == "moe_v1":
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tomorrow_returns_train=ret_train,
            tomorrow_returns_val=ret_val,
            crash_regimes_train=crash_train,
            crash_regimes_val=crash_val,
            regime_labels_train=reg_train,
            regime_labels_val=reg_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_rounds=args.patience,
            verbose=True
        )
    elif args.model == "nn_v11":
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tomorrow_returns_train=ret_train,
            tomorrow_returns_val=ret_val,
            crash_regimes_train=crash_train,
            crash_regimes_val=crash_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_rounds=args.patience,
            verbose=True
        )
    elif args.model in ["nn_v5", "nn_v6", "nn_v10"]:
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tomorrow_returns_train=ret_train,
            tomorrow_returns_val=ret_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_rounds=args.patience,
            verbose=True
        )
    elif args.model == "gnn_v1":
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            adj=args.gnn_adj,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_rounds=args.patience,
            verbose=True
        )
    else:
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_rounds=args.patience,
            verbose=True
        )
    
    # 6.5. Salvataggio della cronologia di addestramento (history) in un file CSV
    try:
        history_df = pd.DataFrame(history)
        history_df.index = history_df.index + 1
        history_df.index.name = "epoch"
        history_csv_path = pesi_dir / f"{Path(args.save_name).stem}_history.csv"
        # Assicuriamoci che la directory esista prima di salvare
        pesi_dir.mkdir(exist_ok=True, parents=True)
        history_df.to_csv(history_csv_path)
        logger.info(f"Cronologia di addestramento (history) salvata con successo in: {history_csv_path}")
    except Exception as e:
        logger.error(f"Errore nel salvataggio della history in CSV: {e}")
        
    # 7. Valutazione sul Test Set (Out-of-sample)
    if args.model == "gnn_v1":
        test_probs = model.predict(X_test, adj=args.gnn_adj)
    else:
        test_probs = model.predict(X_test)
        
    test_preds = (test_probs > 0.50).astype(int)
    test_accuracy = np.mean(test_preds == y_test)
    
    logger.info(f"=== VALUTAZIONE OUT-OF-SAMPLE (TEST SET - {args.model.upper()}) ===")
    logger.info(f"Accuracy sul Test Set: {test_accuracy*100:.2f}%")
    
    baseline_acc = max(np.mean(y_test == 1), np.mean(y_test == 0))
    logger.info(f"Baseline Accuracy (Classe più frequente): {baseline_acc*100:.2f}%")
    logger.info(f"Performance rispetto alla baseline: {test_accuracy - baseline_acc:+.2f}%")
    
    # 8. Salvataggio unificato dei pesi, parametri e iperparametri
    pesi_dir.mkdir(exist_ok=True, parents=True)
    
    state = {
        "input_dim": input_dim,
        "feature_cols": feature_cols,
        "scaling_mean": mean.tolist(),
        "scaling_std": std.tolist(),
        "lr": model.lr,
        "weight_decay": model.weight_decay,
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict()
    }
    if args.model in ["nn_v5", "nn_v6", "nn_v10", "nn_v11", "moe_v1"]:
        state["lookback"] = args.lookback
        state["d_model"] = model.d_model
        state["nhead"] = model.nhead
        state["num_layers"] = model.num_layers
        state["alpha"] = model.alpha
        if args.model == "nn_v11" or args.model == "moe_v1":
            state["penalty_factor"] = model.penalty_factor
        if args.model == "moe_v1":
            state["lambda_gating"] = model.lambda_gating
    elif args.model == "nn_v4":
        state["lookback"] = args.lookback
        state["d_model"] = model.d_model
        state["nhead"] = model.nhead
        state["num_layers"] = model.num_layers
    elif args.model == "nn_v3":
        state["lookback"] = args.lookback
    elif args.model == "gnn_v1":
        state["lookback"] = args.lookback
        state["hidden_dim"] = model.hidden_dim
        state["adj"] = args.gnn_adj.cpu().tolist()
        state["tickers"] = tickers_list
        
    torch.save(state, filepath)
    logger.info(f"Addestramento concluso con successo! Modello e parametri salvati in: {filepath}")
    print(f"\n[SUCCESS] Modello addestrato ({args.model}) salvato in: {filepath.relative_to(config.BASE_DIR)}")
    print(f"          Accuracy sul Test Set: {test_accuracy*100:.2f}%\n")


if __name__ == "__main__":
    main()
