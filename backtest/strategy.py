from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import logging
import config

class BaseStrategy(ABC):
    """
    Classe base astratta (blueprint) per tutte le strategie di trading.
    Ogni nuova strategia deve ereditare da questa classe e implementare generate_signals.
    """
    @abstractmethod
    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,  # Evitiamo import circolari per Portfolio
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analizza i dati storici disponibili fino alla data corrente (inclusa)
        e restituisce un dizionario di segnali per ciascun ticker.
        
        Ritorna:
            Dict[str, Dict[str, Any]]: es. {"AAPL": {"action": "BUY", "weight": 0.1, "stop_loss_pct": 0.02, "take_profit_pct": 0.05}}
            Le azioni possibili sono: "BUY", "SELL", "HOLD"
        """
        pass


class SMAXStrategy(BaseStrategy):
    """
    Strategia di esempio basata sull'incrocio di Medie Mobili Semplici (SMA Crossover).
    Se la SMA a breve termine supera la SMA a lungo termine, genera un segnale BUY.
    Se la SMA a breve termine scende sotto la SMA a lungo termine, genera un segnale SELL.
    """
    def __init__(self, short_window: int = 10, long_window: int = 50) -> None:
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}

        for ticker, df in historical_data.items():
            # Ci servono abbastanza dati per calcolare la media mobile a lungo termine
            if len(df) < self.long_window:
                continue

            # Calcolo delle medie mobili sugli ultimi prezzi disponibili fino a oggi
            close_prices = df['close']
            
            short_sma = close_prices.rolling(window=self.short_window).mean().iloc[-1]
            long_sma = close_prices.rolling(window=self.long_window).mean().iloc[-1]
            
            # Recuperiamo i valori del giorno precedente per verificare l'incrocio
            short_sma_prev = close_prices.rolling(window=self.short_window).mean().iloc[-2]
            long_sma_prev = close_prices.rolling(window=self.long_window).mean().iloc[-2]
            
            # Verifica incrocio rialzista (Golden Cross) -> BUY
            if short_sma_prev <= long_sma_prev and short_sma > long_sma:
                signals[ticker] = {
                    "action": "BUY",
                    "weight": config.BACKTEST_MAX_POSITION_SIZE,
                    "stop_loss_pct": config.BACKTEST_STOP_LOSS,
                    "take_profit_pct": config.BACKTEST_TAKE_PROFIT
                }
            # Verifica incrocio ribassista (Death Cross) -> SELL
            elif short_sma_prev >= long_sma_prev and short_sma < long_sma:
                signals[ticker] = {
                    "action": "SELL"
                }
            else:
                signals[ticker] = {
                    "action": "HOLD"
                }

        return signals


class NeuralNetworkStrategy(BaseStrategy):
    """
    Strategia quantitativa basata sul modello predittivo di Deep Learning in PyTorch.
    Carica i pesi pre-addestrati e i parametri di scaling, calcola in tempo reale
    le feature scala-invarianti, interroga la rete neurale MLP e genera
    segnali BUY/SELL stabili in base ad una soglia probabilistica impostata.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model_aapl.pth", 
        probability_threshold: float = 0.55
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        # Assicuriamoci che la directory radice sia nel path
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v1.model import NeuralNetworkV1
        
        self.probability_threshold = probability_threshold
        
        # Determinazione del percorso del file del modello
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v1" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia: file dei pesi non trovato in: {model_path}. "
                f"Assicurati di aver prima completato con successo l'addestramento lanciando train.py!"
            )
            
        # Carichiamo i metadati e i parametri di scaling Z-Score salvati
        logger = logging.getLogger("NeuralNetworkStrategy")
        logger.info(f"Caricamento del modello predittivo da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        
        # Inizializziamo il modello PyTorch MLP
        self.model = NeuralNetworkV1(input_dim=self.input_dim)
        self.model.load(model_path)
        
        logger.info("Modello e parametri di scaling Z-Score caricati ed attivati con successo.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}

        for ticker, df in historical_data.items():
            # Ci servono abbastanza dati storici per il volume rolling (almeno 10 righe)
            if len(df) < 10:
                continue

            # Estraiamo l'ultima riga dei dati disponibili fino a oggi
            latest_row = df.iloc[-1]
            
            # Verifichiamo la presenza di eventuali valori NaN nelle colonne necessarie
            required_cols = [
                'close', 'volume', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 
                'ema_9', 'ema_21', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 
                'bb_upper', 'bb_lower', 'atr_14'
            ]
            if latest_row[required_cols].isna().any():
                signals[ticker] = {"action": "HOLD"}
                continue
                
            close = latest_row['close']
            
            # Ricreiamo esattamente le stesse feature scala-invarianti calcolate nel train.py
            features = {
                'sma_10_ratio': latest_row['sma_10'] / close,
                'sma_20_ratio': latest_row['sma_20'] / close,
                'sma_50_ratio': latest_row['sma_50'] / close,
                'sma_200_ratio': latest_row['sma_200'] / close,
                'ema_9_ratio': latest_row['ema_9'] / close,
                'ema_21_ratio': latest_row['ema_21'] / close,
                'bb_upper_ratio': latest_row['bb_upper'] / close,
                'bb_lower_ratio': latest_row['bb_lower'] / close,
                'macd_ratio': latest_row['macd'] / close,
                'macd_signal_ratio': latest_row['macd_signal'] / close,
                'macd_hist_ratio': latest_row['macd_hist'] / close,
                'atr_14_ratio': latest_row['atr_14'] / close,
                'volume_ratio': latest_row['volume'] / df['volume'].rolling(10).mean().iloc[-1],
                'rsi_14_norm': latest_row['rsi_14'] / 100.0
            }
            
            # Estraiamo il vettore delle feature ordinato nello stesso esatto modo dell'addestramento
            feature_vector = np.array([features[col] for col in self.feature_cols])
            
            # Applichiamo lo scaling Z-Score usando i parametri di addestramento salvati
            feature_vector_scaled = (feature_vector - self.mean) / self.std
            feature_vector_scaled = feature_vector_scaled.reshape(1, -1)
            
            # Eseguiamo il Forward Pass sul modello PyTorch
            prob = self.model.predict(feature_vector_scaled)[0]
            
            # Generazione Segnali
            # Se la probabilità stimata di rialzo supera la soglia probabilistica di acquisto
            if prob >= self.probability_threshold:
                # Se non abbiamo posizioni aperte, compriamo
                if ticker not in portfolio.positions:
                    signals[ticker] = {
                        "action": "BUY",
                        "weight": config.BACKTEST_MAX_POSITION_SIZE,
                        "stop_loss_pct": config.BACKTEST_STOP_LOSS,
                        "take_profit_pct": config.BACKTEST_TAKE_PROFIT
                    }
                else:
                    signals[ticker] = {"action": "HOLD"}
                    
            # Se la probabilità stimata di rialzo scende sotto la soglia di liquidazione (segnale debole)
            elif prob < (1 - self.probability_threshold + 0.10):
                # Se abbiamo una posizione aperta, liquidiamo
                if ticker in portfolio.positions:
                    signals[ticker] = {"action": "SELL"}
                else:
                    signals[ticker] = {"action": "HOLD"}
            else:
                signals[ticker] = {"action": "HOLD"}

        return signals


class NeuralNetworkV2Strategy(BaseStrategy):
    """
    Strategia quantitativa basata sul modello residuo di Deep Learning v2 in PyTorch.
    Carica i pesi v2 e i parametri di scaling, calcola le feature avanzate scala-invarianti v2
    in tempo reale e genera segnali stabili.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model.pth", 
        probability_threshold: float = 0.55
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v2.model import NeuralNetworkV2
        
        self.probability_threshold = probability_threshold
        
        # Percorso dei pesi v2
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v2" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v2: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV2Strategy")
        logger.info(f"Caricamento del modello residuo v2 da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        
        # Inizializza e carica il modello v2
        self.model = NeuralNetworkV2(input_dim=self.input_dim)
        self.model.load(str(model_path))
        
        logger.info("Modello residuo v2 e parametri di scaling caricati con successo.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}

        for ticker, df in historical_data.items():
            # Ci servono abbastanza dati storici per gli indicatori rolling (almeno 200 righe per SMA200)
            if len(df) < 200:
                continue

            latest_row = df.iloc[-1]
            
            # Verifichiamo la presenza di eventuali valori NaN nelle colonne necessarie
            required_cols = [
                'close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 
                'bb_upper', 'bb_lower', 'bb_middle', 'atr_14'
            ]
            if latest_row[required_cols].isna().any():
                signals[ticker] = {"action": "HOLD"}
                continue
                
            close = latest_row['close']
            
            # Calcoliamo le feature scala-invarianti v2
            ret_series = df['close'].pct_change().fillna(0)
            vol_ret_series = df['volume'].pct_change().fillna(0)
            
            # OBV_ret
            obv_series = (np.sign(ret_series) * df['volume']).fillna(0).cumsum()
            obv_ret = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).iloc[-1]
            
            features = {
                'ret': ret_series.iloc[-1],
                'vol_ret': vol_ret_series.iloc[-1],
                'RSI_14': latest_row['rsi_14'] / 100.0,
                'Bollinger_%B': (close - latest_row['bb_lower']) / (latest_row['bb_upper'] - latest_row['bb_lower'] + 1e-9),
                'Bollinger_Width': (latest_row['bb_upper'] - latest_row['bb_lower']) / (latest_row['bb_middle'] + 1e-9),
                'ATRr_14': latest_row['atr_14'] / close,
                'Dist_SMA200': (close - latest_row['sma_200']) / (latest_row['sma_200'] + 1e-9),
                'Dist_SMA50': (close - latest_row['sma_50']) / (latest_row['sma_50'] + 1e-9),
                'OBV_ret': obv_ret
            }
            
            feature_vector = np.array([features[col] for col in self.feature_cols])
            feature_vector_scaled = (feature_vector - self.mean) / self.std
            feature_vector_scaled = feature_vector_scaled.reshape(1, -1)
            
            prob = self.model.predict(feature_vector_scaled)[0]
            
            # Generazione Segnali
            if prob >= self.probability_threshold:
                if ticker not in portfolio.positions:
                    signals[ticker] = {
                        "action": "BUY",
                        "weight": config.BACKTEST_MAX_POSITION_SIZE,
                        "stop_loss_pct": config.BACKTEST_STOP_LOSS,
                        "take_profit_pct": config.BACKTEST_TAKE_PROFIT
                    }
                else:
                    signals[ticker] = {"action": "HOLD"}
            elif prob < (1 - self.probability_threshold + 0.10):
                if ticker in portfolio.positions:
                    signals[ticker] = {"action": "SELL"}
                else:
                    signals[ticker] = {"action": "HOLD"}
            else:
                signals[ticker] = {"action": "HOLD"}

        return signals


class NeuralNetworkV3Strategy(BaseStrategy):
    """
    Strategia quantitativa basata sul modello sequenziale LSTM + Attention v3 in PyTorch.
    Carica i pesi v3, le feature e i parametri di scaling.
    Mantiene e costruisce finestre temporali storiche (lookback) in tempo reale,
    calcolando gli indicatori scala-invarianti v2 ad ogni passo della sequenza,
    prima di alimentare l'LSTM per la generazione stabili di segnali di trading.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model.pth", 
        probability_threshold: float = 0.58
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v3.model import NeuralNetworkV3
        
        self.probability_threshold = probability_threshold
        
        # Percorso dei pesi v3
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v3" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v3: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV3Strategy")
        logger.info(f"Caricamento del modello LSTM v3 da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        # Inizializza e carica il modello v3
        self.model = NeuralNetworkV3(input_dim=self.input_dim, lookback=self.lookback)
        self.model.load(str(model_path))
        
        logger.info(f"Modello LSTM v3 (lookback = {self.lookback}) e parametri di scaling caricati con successo.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}

        for ticker, df in historical_data.items():
            # Ci servono abbastanza dati storici per riempire la lookback window più il calcolo degli indicatori
            # (almeno lookback + 200 righe per SMA200)
            if len(df) < self.lookback + 200:
                continue

            # Selezioniamo le ultime lookback righe per le predizioni,
            # ma utilizziamo il DataFrame intero per calcolare correttamente le medie mobili e indicatori rolling
            close_series = df['close']
            volume_series = df['volume']
            
            # Calcolo dei rendimenti e OBV su tutta la serie per stabilità
            ret_series = close_series.pct_change().fillna(0)
            vol_ret_series = volume_series.pct_change().fillna(0)
            
            obv_series = (np.sign(ret_series) * volume_series).fillna(0).cumsum()
            obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            
            rsi_series = df['rsi_14'] / 100.0
            atr_series = df['atr_14'] / close_series
            
            bb_b_series = (close_series - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
            bb_w_series = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
            
            dist_200_series = (close_series - df['sma_200']) / (df['sma_200'] + 1e-9)
            dist_50_series = (close_series - df['sma_50']) / (df['sma_50'] + 1e-9)
            
            # Ricostruiamo la sequenza temporale di lunghezza `lookback` terminante all'ultima riga
            seq_features = []
            has_nan = False
            
            for idx in range(-self.lookback, 0):
                # Verifichiamo se ci sono valori NaN nelle colonne necessarie a quell'indice temporale
                check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
                if df.iloc[idx][check_cols].isna().any():
                    has_nan = True
                    break
                    
                feat_dict = {
                    'ret': ret_series.iloc[idx],
                    'vol_ret': vol_ret_series.iloc[idx],
                    'RSI_14': rsi_series.iloc[idx],
                    'Bollinger_%B': bb_b_series.iloc[idx],
                    'Bollinger_Width': bb_w_series.iloc[idx],
                    'ATRr_14': atr_series.iloc[idx],
                    'Dist_SMA200': dist_200_series.iloc[idx],
                    'Dist_SMA50': dist_50_series.iloc[idx],
                    'OBV_ret': obv_ret_series.iloc[idx]
                }
                feature_vector = np.array([feat_dict[col] for col in self.feature_cols])
                # Applichiamo lo scaling Z-Score con i parametri del train
                feature_vector_scaled = (feature_vector - self.mean) / self.std
                seq_features.append(feature_vector_scaled)
                
            if has_nan:
                signals[ticker] = {"action": "HOLD"}
                continue
                
            # Shape finale: (1, lookback, input_dim)
            seq_features_arr = np.array(seq_features, dtype=np.float32).reshape(1, self.lookback, -1)
            
            # Forward pass sull'LSTM
            prob = self.model.predict(seq_features_arr)[0]
            
            # Generazione Segnali con Stop Loss e Take Profit dinamico basato su ATR
            # A. Controllo per posizioni attive (Uscite)
            if ticker in portfolio.positions:
                pos = portfolio.positions[ticker]
                if pos.position_type == "LONG":
                    if prob < (1 - self.probability_threshold + 0.10):
                        signals[ticker] = {"action": "SELL", "probability": float(prob)}
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:  # SHORT
                    if prob > (self.probability_threshold - 0.10):
                        signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
            # B. Controllo per nuovi ingressi (Entrate)
            else:
                latest_row = df.iloc[-1]
                close = latest_row['close']
                atr = latest_row['atr_14']
                
                # Stop loss basato sulla volatilità dell'ATR (4.0 * ATR / Prezzo)
                # Limiti di protezione: tra 1.5% e 8.0%
                stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                take_profit_pct = float(stop_loss_pct * 2.0)
                
                if prob >= self.probability_threshold:
                    signals[ticker] = {
                        "action": "BUY",
                        "probability": float(prob),
                        "stop_loss_pct": stop_loss_pct,
                        "take_profit_pct": take_profit_pct
                    }
                elif prob <= (1.0 - self.probability_threshold):
                    signals[ticker] = {
                        "action": "SELL_SHORT",
                        "probability": float(prob),
                        "stop_loss_pct": stop_loss_pct,
                        "take_profit_pct": take_profit_pct
                    }
                else:
                    signals[ticker] = {"action": "HOLD", "probability": float(prob)}

        return signals


class NeuralNetworkV4Strategy(BaseStrategy):
    """
    Strategia quantitativa basata sul modello sequenziale Transformer v4 in PyTorch.
    Carica i pesi v4, le feature (incluse ROC, Stochastic, Volume standard deviation) e i parametri di scaling.
    Costruisce e mantiene le lookback window, calcola le feature di borsa avanzate
    e genera segnali bidirezionali LONG/SHORT ordinabili per confidenza.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model.pth", 
        probability_threshold: float = 0.525,
        ranking_mode: bool = True,
        top_pct: float = 0.03,
        exit_pct: float = 0.60,
        exit_long_threshold: float = 0.485,
        exit_short_threshold: float = 0.515,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v4.model import NeuralNetworkV4
        
        self.probability_threshold = probability_threshold
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        
        # Percorso dei pesi v4
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v4" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v4: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV4Strategy")
        logger.info(f"Caricamento del modello Transformer v4 da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        # Inizializza e carica il modello v4
        self.model = NeuralNetworkV4(
            input_dim=self.input_dim, 
            lookback=self.lookback,
            d_model=state.get("d_model", 64),
            nhead=state.get("nhead", 4),
            num_layers=state.get("num_layers", 2)
        )
        self.model.load(str(model_path))
        
        logger.info(f"Modello Transformer v4 (lookback = {self.lookback}) caricato con successo.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}
        
        valid_tickers = []
        seq_features_list = []
        ticker_dfs = {}

        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue

            # Pre-slicing per ottimizzazione massiva (calcoliamo solo gli ultimi lookback + 30 giorni)
            slice_len = self.lookback + 30
            close_slice = df['close'].iloc[-slice_len:]
            high_slice = df['high'].iloc[-slice_len:]
            low_slice = df['low'].iloc[-slice_len:]
            volume_slice = df['volume'].iloc[-slice_len:]
            
            ret_series = close_slice.pct_change().fillna(0)
            vol_ret_series = volume_slice.pct_change().fillna(0)
            
            obv_series = (np.sign(ret_series) * volume_slice).fillna(0).cumsum()
            obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            
            rsi_series = df['rsi_14'].iloc[-self.lookback:] / 100.0
            atr_series = df['atr_14'].iloc[-self.lookback:] / close_slice.iloc[-self.lookback:]
            
            bb_upper_s = df['bb_upper'].iloc[-self.lookback:]
            bb_lower_s = df['bb_lower'].iloc[-self.lookback:]
            bb_middle_s = df['bb_middle'].iloc[-self.lookback:]
            close_lookback = close_slice.iloc[-self.lookback:]
            
            bb_b_series = (close_lookback - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-9)
            bb_w_series = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-9)
            
            dist_200_series = (close_lookback - df['sma_200'].iloc[-self.lookback:]) / (df['sma_200'].iloc[-self.lookback:] + 1e-9)
            dist_50_series = (close_lookback - df['sma_50'].iloc[-self.lookback:]) / (df['sma_50'].iloc[-self.lookback:] + 1e-9)
            
            # --- Calcolo Indicatori di Borsa Avanzati su fetta ridotta ---
            sma_5 = close_slice.rolling(5).mean()
            ema_12 = close_slice.ewm(span=12, adjust=False).mean()
            
            roc_10_series = close_slice.pct_change(10).fillna(0)
            
            low_14 = low_slice.rolling(14).min()
            high_14 = high_slice.rolling(14).max()
            stoch_k_series = ((close_slice - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
            
            sma_5_ratio_series = sma_5 / close_slice
            ema_12_ratio_series = ema_12 / close_slice
            
            volume_std_10 = volume_slice.rolling(10).std().fillna(0)
            volume_std_ratio_series = (volume_slice / (volume_std_10 + 1e-9)).fillna(1.0)
            
            # Controlla la presenza di valori NaN negli indicatori fondamentali per l'intera lookback window in un colpo solo
            check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
            if df.iloc[-self.lookback:][check_cols].isna().any().any():
                signals[ticker] = {"action": "HOLD", "probability": 0.5}
                continue
                
            # Estrazione dei valori numpy per la lookback window (elimina le letture riga-per-riga via pandas)
            ret_arr = ret_series.values[-self.lookback:]
            vol_ret_arr = vol_ret_series.values[-self.lookback:]
            rsi_arr = rsi_series.values[-self.lookback:]
            bb_b_arr = bb_b_series.values[-self.lookback:]
            bb_w_arr = bb_w_series.values[-self.lookback:]
            atr_arr = atr_series.values[-self.lookback:]
            dist_200_arr = dist_200_series.values[-self.lookback:]
            dist_50_arr = dist_50_series.values[-self.lookback:]
            obv_ret_arr = obv_ret_series.values[-self.lookback:]
            roc_10_arr = roc_10_series.values[-self.lookback:]
            stoch_k_arr = stoch_k_series.values[-self.lookback:]
            sma_5_ratio_arr = sma_5_ratio_series.values[-self.lookback:]
            ema_12_ratio_arr = ema_12_ratio_series.values[-self.lookback:]
            volume_std_ratio_arr = volume_std_ratio_series.values[-self.lookback:]
            
            feature_arrays = {
                'ret': ret_arr,
                'vol_ret': vol_ret_arr,
                'RSI_14': rsi_arr,
                'Bollinger_%B': bb_b_arr,
                'Bollinger_Width': bb_w_arr,
                'ATRr_14': atr_arr,
                'Dist_SMA200': dist_200_arr,
                'Dist_SMA50': dist_50_arr,
                'OBV_ret': obv_ret_arr,
                'ROC_10': roc_10_arr,
                'Stoch_K': stoch_k_arr,
                'SMA_5_ratio': sma_5_ratio_arr,
                'EMA_12_ratio': ema_12_ratio_arr,
                'Volume_Std_Ratio': volume_std_ratio_arr
            }
            
            # Stack delle feature in un array numpy 2D (lookback, input_dim)
            seq_feature_vectors = np.column_stack([feature_arrays[col] for col in self.feature_cols])
            
            # Scaling Z-Score vettorializzato
            seq_features_scaled = (seq_feature_vectors - self.mean) / self.std
            
            valid_tickers.append(ticker)
            seq_features_list.append(seq_features_scaled)
            ticker_dfs[ticker] = df

        if not valid_tickers:
            return signals

        # Eseguiamo la predizione in batch quotidiano!
        batch_x = np.array(seq_features_list, dtype=np.float32)
        probs = self.model.predict(batch_x)
        
        if self.ranking_mode:
            # --- MODALITÀ RELATIVE STRENGTH RANKING CROSS-SECTIONALE ---
            ticker_probs = {ticker: float(prob) for ticker, prob in zip(valid_tickers, probs)}
            sorted_tickers = sorted(ticker_probs.items(), key=lambda x: x[1], reverse=True)
            
            N = len(valid_tickers)
            # Top K e Bottom K (es. top_pct del pool, default 3%)
            K = max(1, int(N * self.top_pct))
            top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
            bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
            
            # Gruppi allargati per isteresi uscite (es. exit_pct, default 60%)
            K_out = max(1, int(N * self.exit_pct))
            top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
            bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
            
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                # Determinazione del trend macro
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                # Configurazione soglie adattive
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else self.probability_threshold
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else 0.505
                else:
                    thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else self.probability_threshold
                    thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else (1.0 - self.probability_threshold)
                
                # A. Controllo per posizioni attive (Uscite)
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    if pos.position_type == "LONG":
                        # Esce se la probabilità è negativa (< exit_long_threshold), se esce dal top_out_tickers,
                        # O se il trend macro diventa ribassista (in down-trend tagliamo i LONG)
                        if prob < self.exit_long_threshold or ticker not in top_out_tickers or (self.trend_filter and not is_uptrend):
                            signals[ticker] = {"action": "SELL", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:  # SHORT
                        # Esce se la probabilità è positiva (> exit_short_threshold), se esce dal bottom_out_tickers,
                        # O se il trend macro diventa rialzista (in up-trend tagliamo gli SHORT)
                        if prob > self.exit_short_threshold or ticker not in bottom_out_tickers or (self.trend_filter and is_uptrend):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                # B. Controllo per nuovi ingressi (Entrate)
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    take_profit_pct = float(stop_loss_pct * 2.0)
                    
                    # Entra LONG se è nel top K ed ha probabilità favorevole
                    if ticker in top_K_tickers and prob >= thresh_long:
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct
                        }
                    # Entra SHORT se è nel bottom K ed ha probabilità favorevole
                    elif ticker in bottom_K_tickers and prob <= thresh_short:
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
        else:
            # --- MODALITÀ SOGLIA ASSOLUTA ---
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                # Determinazione del trend macro
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                # Configurazione soglie adattive
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else self.probability_threshold
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else 0.505
                else:
                    thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else self.probability_threshold
                    thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else (1.0 - self.probability_threshold)
                
                # A. Controllo per posizioni attive (Uscite)
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    if pos.position_type == "LONG":
                        # Uscita con isteresi o inversione di trend
                        if prob < 0.495 or (self.trend_filter and not is_uptrend):
                            signals[ticker] = {"action": "SELL", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:  # SHORT
                        # Uscita con isteresi o inversione di trend
                        if prob > 0.505 or (self.trend_filter and is_uptrend):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                # B. Controllo per nuovi ingressi (Entrate)
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    take_profit_pct = float(stop_loss_pct * 2.0)
                    
                    if prob >= thresh_long:
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct
                        }
                    elif prob <= thresh_short:
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
 
        return signals


class NeuralNetworkV5Strategy(BaseStrategy):
    """
    Strategia quantitativa basata sul modello ibrido CNN-Transformer v5 in PyTorch.
    Estrae le feature relative al mercato a runtime e genera segnali bidirezionali.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model.pth", 
        probability_threshold: float = 0.525,
        ranking_mode: bool = True,
        top_pct: float = 0.03,
        exit_pct: float = 0.60,
        exit_long_threshold: float = 0.485,
        exit_short_threshold: float = 0.515,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v5.model import NeuralNetworkV5
        
        self.probability_threshold = probability_threshold
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v5" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v5: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV5Strategy")
        logger.info(f"Caricamento del modello CNN-Transformer v5 da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        self.model = NeuralNetworkV5(
            input_dim=self.input_dim, 
            lookback=self.lookback,
            d_model=state.get("d_model", 64),
            nhead=state.get("nhead", 4),
            num_layers=state.get("num_layers", 2),
            alpha=state.get("alpha", 50.0)
        )
        self.model.load(str(model_path))
        
        logger.info(f"Modello CNN-Transformer v5 (lookback = {self.lookback}) caricato con successo.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}
        
        # 1. Calcolo preventivo cross-sectionale del mercato (benchmark medio del pool di oggi)
        slice_len = self.lookback + 30
        ticker_rets = {}
        ticker_vols = {}
        
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
            ticker_rets[ticker] = df['close'].iloc[-slice_len:].pct_change().fillna(0)
            ticker_vols[ticker] = df['volume'].iloc[-slice_len:]
            
        if not ticker_rets:
            return signals
            
        df_rets_all = pd.DataFrame(ticker_rets)
        df_vols_all = pd.DataFrame(ticker_vols)
        
        market_daily_ret = df_rets_all.mean(axis=1)
        market_daily_vol = df_vols_all.mean(axis=1)
        
        valid_tickers = []
        seq_features_list = []
        ticker_dfs = {}

        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue

            close_slice = df['close'].iloc[-slice_len:]
            high_slice = df['high'].iloc[-slice_len:]
            low_slice = df['low'].iloc[-slice_len:]
            volume_slice = df['volume'].iloc[-slice_len:]
            
            ret_series = close_slice.pct_change().fillna(0)
            vol_ret_series = volume_slice.pct_change().fillna(0)
            
            obv_series = (np.sign(ret_series) * volume_slice).fillna(0).cumsum()
            obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            
            rsi_series = df['rsi_14'].iloc[-self.lookback:] / 100.0
            atr_series = df['atr_14'].iloc[-self.lookback:] / close_slice.iloc[-self.lookback:]
            
            bb_upper_s = df['bb_upper'].iloc[-self.lookback:]
            bb_lower_s = df['bb_lower'].iloc[-self.lookback:]
            bb_middle_s = df['bb_middle'].iloc[-self.lookback:]
            close_lookback = close_slice.iloc[-self.lookback:]
            
            bb_b_series = (close_lookback - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-9)
            bb_w_series = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-9)
            
            dist_200_series = (close_lookback - df['sma_200'].iloc[-self.lookback:]) / (df['sma_200'].iloc[-self.lookback:] + 1e-9)
            dist_50_series = (close_lookback - df['sma_50'].iloc[-self.lookback:]) / (df['sma_50'].iloc[-self.lookback:] + 1e-9)
            
            sma_5 = close_slice.rolling(5).mean()
            ema_12 = close_slice.ewm(span=12, adjust=False).mean()
            
            roc_10_series = close_slice.pct_change(10).fillna(0)
            
            low_14 = low_slice.rolling(14).min()
            high_14 = high_slice.rolling(14).max()
            stoch_k_series = ((close_slice - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
            
            sma_5_ratio_series = sma_5 / close_slice
            ema_12_ratio_series = ema_12 / close_slice
            
            volume_std_10 = volume_slice.rolling(10).std().fillna(0)
            volume_std_ratio_series = (volume_slice / (volume_std_10 + 1e-9)).fillna(1.0)
            
            # --- Feature Relative al Mercato (v5) ---
            market_relative_ret_series = ret_series - market_daily_ret
            market_relative_volume_series = volume_slice / (market_daily_vol + 1e-9)
            
            check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
            if df.iloc[-self.lookback:][check_cols].isna().any().any():
                signals[ticker] = {"action": "HOLD", "probability": 0.5}
                continue
                
            ret_arr = ret_series.values[-self.lookback:]
            vol_ret_arr = vol_ret_series.values[-self.lookback:]
            rsi_arr = rsi_series.values[-self.lookback:]
            bb_b_arr = bb_b_series.values[-self.lookback:]
            bb_w_arr = bb_w_series.values[-self.lookback:]
            atr_arr = atr_series.values[-self.lookback:]
            dist_200_arr = dist_200_series.values[-self.lookback:]
            dist_50_arr = dist_50_series.values[-self.lookback:]
            obv_ret_arr = obv_ret_series.values[-self.lookback:]
            roc_10_arr = roc_10_series.values[-self.lookback:]
            stoch_k_arr = stoch_k_series.values[-self.lookback:]
            sma_5_ratio_arr = sma_5_ratio_series.values[-self.lookback:]
            ema_12_ratio_arr = ema_12_ratio_series.values[-self.lookback:]
            volume_std_ratio_arr = volume_std_ratio_series.values[-self.lookback:]
            market_relative_ret_arr = market_relative_ret_series.values[-self.lookback:]
            market_relative_volume_arr = market_relative_volume_series.values[-self.lookback:]
            
            feature_arrays = {
                'ret': ret_arr,
                'vol_ret': vol_ret_arr,
                'RSI_14': rsi_arr,
                'Bollinger_%B': bb_b_arr,
                'Bollinger_Width': bb_w_arr,
                'ATRr_14': atr_arr,
                'Dist_SMA200': dist_200_arr,
                'Dist_SMA50': dist_50_arr,
                'OBV_ret': obv_ret_arr,
                'ROC_10': roc_10_arr,
                'Stoch_K': stoch_k_arr,
                'SMA_5_ratio': sma_5_ratio_arr,
                'EMA_12_ratio': ema_12_ratio_arr,
                'Volume_Std_Ratio': volume_std_ratio_arr,
                'Market_Relative_Ret': market_relative_ret_arr,
                'Market_Relative_Volume': market_relative_volume_arr
            }
            
            seq_feature_vectors = np.column_stack([feature_arrays[col] for col in self.feature_cols])
            seq_features_scaled = (seq_feature_vectors - self.mean) / self.std
            
            valid_tickers.append(ticker)
            seq_features_list.append(seq_features_scaled)
            ticker_dfs[ticker] = df

        if not valid_tickers:
            return signals

        batch_x = np.array(seq_features_list, dtype=np.float32)
        probs = self.model.predict(batch_x)
        
        if self.ranking_mode:
            ticker_probs = {ticker: float(prob) for ticker, prob in zip(valid_tickers, probs)}
            sorted_tickers = sorted(ticker_probs.items(), key=lambda x: x[1], reverse=True)
            
            N = len(valid_tickers)
            K = max(1, int(N * self.top_pct))
            top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
            bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
            
            K_out = max(1, int(N * self.exit_pct))
            top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
            bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
            
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else self.probability_threshold
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else 0.505
                else:
                    thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else self.probability_threshold
                    thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else (1.0 - self.probability_threshold)
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    if pos.position_type == "LONG":
                        if prob < self.exit_long_threshold or ticker not in top_out_tickers or (self.trend_filter and not is_uptrend):
                            signals[ticker] = {"action": "SELL", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:
                        if prob > self.exit_short_threshold or ticker not in bottom_out_tickers or (self.trend_filter and is_uptrend):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    take_profit_pct = float(stop_loss_pct * 2.0)
                    
                    if ticker in top_K_tickers and prob >= thresh_long:
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct
                        }
                    elif ticker in bottom_K_tickers and prob <= thresh_short:
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
        else:
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else self.probability_threshold
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else 0.505
                else:
                    thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else self.probability_threshold
                    thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else (1.0 - self.probability_threshold)
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    if pos.position_type == "LONG":
                        if prob < 0.495 or (self.trend_filter and not is_uptrend):
                            signals[ticker] = {"action": "SELL", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:
                        if prob > 0.505 or (self.trend_filter and is_uptrend):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    take_profit_pct = float(stop_loss_pct * 2.0)
                    
                    if prob >= thresh_long:
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct
                        }
                    elif prob <= thresh_short:
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
 
        return signals


class NeuralNetworkV6Strategy(BaseStrategy):
    """
    Strategia quantitativa basata sul modello ibrido CNN-Transformer v6 in PyTorch.
    Estrae le feature relative e macro a runtime, calcola la volatilità di mercato
    per soglie dinamiche regolate sul regime, e applica il Kelly Sizing dinamico.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model.pth", 
        probability_threshold: float = 0.525,
        ranking_mode: bool = True,
        top_pct: float = 0.03,
        exit_pct: float = 0.60,
        exit_long_threshold: float = 0.485,
        exit_short_threshold: float = 0.515,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v6.model import NeuralNetworkV6
        
        self.probability_threshold = probability_threshold
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v6" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v6: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV6Strategy")
        logger.info(f"Caricamento del modello CNN-Transformer v6 da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        self.model = NeuralNetworkV6(
            input_dim=self.input_dim, 
            lookback=self.lookback,
            d_model=state.get("d_model", 64),
            nhead=state.get("nhead", 4),
            num_layers=state.get("num_layers", 2),
            alpha=state.get("alpha", 50.0)
        )
        self.model.load(str(model_path))
        
        logger.info(f"Modello CNN-Transformer v6 (lookback = {self.lookback}) caricato con successo.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}
        
        # 1. Calcolo preventivo cross-sectionale del mercato (benchmark medio del pool + volatilità rolling a 20 giorni)
        slice_len = self.lookback + 50
        ticker_rets = {}
        ticker_vols = {}
        
        # Calcolo Breadth di Mercato (percentuale di titoli sopra la SMA 200)
        uptrend_counts = 0
        total_valid = 0
        
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
            ticker_rets[ticker] = df['close'].iloc[-slice_len:].pct_change().fillna(0)
            ticker_vols[ticker] = df['volume'].iloc[-slice_len:]
            
            # Breadth check
            if len(df) >= 200:
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                if pd.notna(sma_200):
                    total_valid += 1
                    if close >= sma_200:
                        uptrend_counts += 1
            
        if not ticker_rets:
            return signals
            
        market_breadth = (uptrend_counts / total_valid) if total_valid > 0 else 0.5
            
        df_rets_all = pd.DataFrame(ticker_rets)
        df_vols_all = pd.DataFrame(ticker_vols)
        
        market_daily_ret = df_rets_all.mean(axis=1)
        market_daily_vol = df_vols_all.mean(axis=1)
        
        # Volatilità rolling del mercato a 20 giorni
        market_rolling_vol = market_daily_ret.rolling(20).std().fillna(0.0)
        
        # Volatilità odierna vs Volatilità media storica della finestra (regime-filter)
        vol_avg = market_rolling_vol.mean()
        vol_today = market_rolling_vol.iloc[-1]
        vol_ratio = vol_today / (vol_avg + 1e-9)
        vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
        
        # 2. Regolazione Asimmetrica Adattiva delle Soglie di Confidenza (v6)
        base_threshold = self.probability_threshold
        raw_thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else base_threshold
        raw_thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else (1.0 - base_threshold)
        
        # Più il mercato è volatile (vol_ratio > 1.0), più le soglie si distanziano (maggiore selettività)
        adjusted_thresh_long = 0.50 + (raw_thresh_long - 0.50) * vol_ratio
        adjusted_thresh_short = 0.50 - (0.50 - raw_thresh_short) * vol_ratio
        
        adjusted_thresh_long = float(np.clip(adjusted_thresh_long, 0.515, 0.56))
        adjusted_thresh_short = float(np.clip(adjusted_thresh_short, 0.44, 0.485))
        
        # Disabilita gli SHORT se la market breadth è superiore al 40% (regime rialzista/rialzo forte)
        if market_breadth > 0.40:
            adjusted_thresh_short = 0.0
        
        valid_tickers = []
        seq_features_list = []
        ticker_dfs = {}

        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue

            close_slice = df['close'].iloc[-slice_len:]
            high_slice = df['high'].iloc[-slice_len:]
            low_slice = df['low'].iloc[-slice_len:]
            volume_slice = df['volume'].iloc[-slice_len:]
            
            ret_series = close_slice.pct_change().fillna(0)
            vol_ret_series = volume_slice.pct_change().fillna(0)
            
            obv_series = (np.sign(ret_series) * volume_slice).fillna(0).cumsum()
            obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            
            rsi_series = df['rsi_14'].iloc[-self.lookback:] / 100.0
            atr_series = df['atr_14'].iloc[-self.lookback:] / close_slice.iloc[-self.lookback:]
            
            bb_upper_s = df['bb_upper'].iloc[-self.lookback:]
            bb_lower_s = df['bb_lower'].iloc[-self.lookback:]
            bb_middle_s = df['bb_middle'].iloc[-self.lookback:]
            close_lookback = close_slice.iloc[-self.lookback:]
            
            bb_b_series = (close_lookback - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-9)
            bb_w_series = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-9)
            
            dist_200_series = (close_lookback - df['sma_200'].iloc[-self.lookback:]) / (df['sma_200'].iloc[-self.lookback:] + 1e-9)
            dist_50_series = (close_lookback - df['sma_50'].iloc[-self.lookback:]) / (df['sma_50'].iloc[-self.lookback:] + 1e-9)
            
            sma_5 = close_slice.rolling(5).mean()
            ema_12 = close_slice.ewm(span=12, adjust=False).mean()
            
            roc_10_series = close_slice.pct_change(10).fillna(0)
            
            low_14 = low_slice.rolling(14).min()
            high_14 = high_slice.rolling(14).max()
            stoch_k_series = ((close_slice - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
            
            sma_5_ratio_series = sma_5 / close_slice
            ema_12_ratio_series = ema_12 / close_slice
            
            volume_std_10 = volume_slice.rolling(10).std().fillna(0)
            volume_std_ratio_series = (volume_slice / (volume_std_10 + 1e-9)).fillna(1.0)
            
            # --- Feature Relative al Mercato (v5) ---
            market_relative_ret_series = ret_series - market_daily_ret
            market_relative_volume_series = volume_slice / (market_daily_vol + 1e-9)
            
            # --- Feature Macro di Mercato (v6) ---
            market_return_series = market_daily_ret
            market_volatility_series = market_rolling_vol
            
            check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
            if df.iloc[-self.lookback:][check_cols].isna().any().any():
                signals[ticker] = {"action": "HOLD", "probability": 0.5}
                continue
                
            ret_arr = ret_series.values[-self.lookback:]
            vol_ret_arr = vol_ret_series.values[-self.lookback:]
            rsi_arr = rsi_series.values[-self.lookback:]
            bb_b_arr = bb_b_series.values[-self.lookback:]
            bb_w_arr = bb_w_series.values[-self.lookback:]
            atr_arr = atr_series.values[-self.lookback:]
            dist_200_arr = dist_200_series.values[-self.lookback:]
            dist_50_arr = dist_50_series.values[-self.lookback:]
            obv_ret_arr = obv_ret_series.values[-self.lookback:]
            roc_10_arr = roc_10_series.values[-self.lookback:]
            stoch_k_arr = stoch_k_series.values[-self.lookback:]
            sma_5_ratio_arr = sma_5_ratio_series.values[-self.lookback:]
            ema_12_ratio_arr = ema_12_ratio_series.values[-self.lookback:]
            volume_std_ratio_arr = volume_std_ratio_series.values[-self.lookback:]
            market_relative_ret_arr = market_relative_ret_series.values[-self.lookback:]
            market_relative_volume_arr = market_relative_volume_series.values[-self.lookback:]
            market_return_arr = market_return_series.values[-self.lookback:]
            market_volatility_arr = market_volatility_series.values[-self.lookback:]
            
            feature_arrays = {
                'ret': ret_arr,
                'vol_ret': vol_ret_arr,
                'RSI_14': rsi_arr,
                'Bollinger_%B': bb_b_arr,
                'Bollinger_Width': bb_w_arr,
                'ATRr_14': atr_arr,
                'Dist_SMA200': dist_200_arr,
                'Dist_SMA50': dist_50_arr,
                'OBV_ret': obv_ret_arr,
                'ROC_10': roc_10_arr,
                'Stoch_K': stoch_k_arr,
                'SMA_5_ratio': sma_5_ratio_arr,
                'EMA_12_ratio': ema_12_ratio_arr,
                'Volume_Std_Ratio': volume_std_ratio_arr,
                'Market_Relative_Ret': market_relative_ret_arr,
                'Market_Relative_Volume': market_relative_volume_arr,
                'Market_Return': market_return_arr,
                'Market_Volatility': market_volatility_arr
            }
            
            seq_feature_vectors = np.column_stack([feature_arrays[col] for col in self.feature_cols])
            seq_features_scaled = (seq_feature_vectors - self.mean) / self.std
            
            valid_tickers.append(ticker)
            seq_features_list.append(seq_features_scaled)
            ticker_dfs[ticker] = df
 
        if not valid_tickers:
            return signals

        batch_x = np.array(seq_features_list, dtype=np.float32)
        probs = self.model.predict(batch_x)
        
        if self.ranking_mode:
            ticker_probs = {ticker: float(prob) for ticker, prob in zip(valid_tickers, probs)}
            sorted_tickers = sorted(ticker_probs.items(), key=lambda x: x[1], reverse=True)
            
            N = len(valid_tickers)
            K = max(1, int(N * self.top_pct))
            top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
            bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
            
            K_out = max(1, int(N * self.exit_pct))
            top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
            bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
            
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = adjusted_thresh_long
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = adjusted_thresh_short
                else:
                    thresh_long = adjusted_thresh_long
                    thresh_short = adjusted_thresh_short
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                         # Uscita se crolla probabilità, se inverte trend macro, O se esce dai top ed è tenuto da almeno 3 giorni
                         should_exit = (
                             prob < self.exit_long_threshold or 
                             (self.trend_filter and not is_uptrend) or
                             (ticker not in top_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "SELL", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else: # SHORT
                         # Uscita se sale probabilità, se inverte trend macro, O se esce dai bottom ed è tenuto da almeno 3 giorni
                         should_exit = (
                             prob > self.exit_short_threshold or 
                             (self.trend_filter and is_uptrend) or
                             (ticker not in bottom_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    take_profit_pct = float(stop_loss_pct * 2.0)
                    
                    # Calcolo trailing stop dinamico basato su ATR (2.0 * ATR)
                    trailing_stop_pct = (2.0 * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.05))
                    
                    if ticker in top_K_tickers and prob >= thresh_long:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        conf_mult = 1.0 + (prob - thresh_long) * 5.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif ticker in bottom_K_tickers and prob <= thresh_short:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        conf_mult = 1.0 + (thresh_short - prob) * 5.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
        else:
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = adjusted_thresh_long
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = adjusted_thresh_short
                else:
                    thresh_long = adjusted_thresh_long
                    thresh_short = adjusted_thresh_short
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                        # Uscita se prob < 0.495 o inversione trend macro
                        if prob < 0.495 or (self.trend_filter and not is_uptrend):
                            signals[ticker] = {"action": "SELL", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:
                        # Uscita se prob > 0.505 o inversione trend macro
                        if prob > 0.505 or (self.trend_filter and is_uptrend):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    take_profit_pct = float(stop_loss_pct * 2.0)
                    
                    # Calcolo trailing stop dinamico basato su ATR (2.0 * ATR)
                    trailing_stop_pct = (2.0 * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.05))
                    
                    if prob >= thresh_long:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        conf_mult = 1.0 + (prob - thresh_long) * 5.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif prob <= thresh_short:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        conf_mult = 1.0 + (thresh_short - prob) * 5.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
 
        return signals


class NeuralNetworkV7Strategy(BaseStrategy):
    """
    Strategia quantitativa avanzata V7.
    Incorpora tre raffinamenti quantitativi di alto livello:
    1. Capital Allocation Dinamica (Kelly & Regime Exposure basata su Market Breadth)
    2. Regime Clustering non supervisionato (GMM con NumPy EM/Quantile fallback)
    3. Statistical Arbitrage Overlay su Visa (V) / Mastercard (MA) market-neutral.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model.pth", 
        probability_threshold: float = 0.525,
        ranking_mode: bool = True,
        top_pct: float = 0.03,
        exit_pct: float = 0.60,
        exit_long_threshold: float = 0.485,
        exit_short_threshold: float = 0.515,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v6.model import NeuralNetworkV6
        
        self.probability_threshold = probability_threshold
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        
        # Inizializziamo i parametri di allocazione dinamica che l'engine leggerà
        self.current_max_slots = 5
        self.current_cash_reserve_pct = 0.0
        
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v6" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v7: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV7Strategy")
        logger.info(f"Caricamento del modello CNN-Transformer v6 per la strategia V7 da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        self.model = NeuralNetworkV6(
            input_dim=self.input_dim, 
            lookback=self.lookback,
            d_model=state.get("d_model", 64),
            nhead=state.get("nhead", 4),
            num_layers=state.get("num_layers", 2),
            alpha=state.get("alpha", 50.0)
        )
        self.model.load(str(model_path))
        
        logger.info(f"Modello CNN-Transformer caricato con successo per la strategia V7.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}
        logger = logging.getLogger("NeuralNetworkV7Strategy")
        
        # ---------------------------------------------------------------------
        # MODULO 1: Calcolo Market Breadth ed Esposizione Dinamica (Capital Allocation)
        # ---------------------------------------------------------------------
        slice_len = self.lookback + 50
        ticker_rets = {}
        ticker_vols = {}
        
        uptrend_counts = 0
        total_valid = 0
        
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
            ticker_rets[ticker] = df['close'].iloc[-slice_len:].pct_change().fillna(0)
            ticker_vols[ticker] = df['volume'].iloc[-slice_len:]
            
            if len(df) >= 200:
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                if pd.notna(sma_200):
                    total_valid += 1
                    if close >= sma_200:
                        uptrend_counts += 1
            
        if not ticker_rets:
            return signals
            
        market_breadth = (uptrend_counts / total_valid) if total_valid > 0 else 0.5
        
        # Regole di allocazione di capitale basate sulla salute del mercato (Market Breadth)
        if market_breadth > 0.70:
            # Forte trend rialzista/bassa volatilità sistemica: concentrati su pochi leader aggressivi
            self.current_max_slots = 3
            self.current_cash_reserve_pct = 0.0
        elif market_breadth < 0.30:
            # Trend ribassista forte o incertezza: massima difesa, tieni 50% cash liquido e diversifica su 10 slot max
            self.current_max_slots = 10
            self.current_cash_reserve_pct = 0.50
        else:
            # Regime neutrale/standard
            self.current_max_slots = 5
            self.current_cash_reserve_pct = 0.0
            
        df_rets_all = pd.DataFrame(ticker_rets)
        df_vols_all = pd.DataFrame(ticker_vols)
        
        market_daily_ret = df_rets_all.mean(axis=1)
        market_daily_vol = df_vols_all.mean(axis=1)
        
        # Volatilità rolling a 20 giorni
        market_rolling_vol = market_daily_ret.rolling(20).std().fillna(0.0)
        
        # Volatilità odierna vs volatilità media storica
        vol_avg = market_rolling_vol.mean()
        vol_today = market_rolling_vol.iloc[-1]
        vol_ratio = vol_today / (vol_avg + 1e-9)
        vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
        
        # ---------------------------------------------------------------------
        # MODULO 2: Regime Clustering non supervisionato (GMM / Fallback)
        # ---------------------------------------------------------------------
        regime = 0  # 0: Toro (Trending Up), 1: Orso (Trending Down / High Vol), 2: Laterale (Mean-Reverting)
        
        # Costruiamo il dataset delle ultime 250 giornate per il clustering
        gmm_data_len = min(250, len(market_daily_ret))
        if gmm_data_len >= 50:
            features = np.column_stack([
                market_daily_ret.iloc[-gmm_data_len:].values,
                market_rolling_vol.iloc[-gmm_data_len:].values,
                market_daily_vol.iloc[-gmm_data_len:].values
            ])
            
            try:
                from sklearn.mixture import GaussianMixture
                # Disattiviamo temporaneamente i log di convergenza rumorosi di sklearn
                gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, max_iter=100)
                gmm.fit(features)
                labels = gmm.predict(features)
                latest_label = labels[-1]
                
                # Ordiniamo i cluster per volatilità crescente per garantire consistenza di significato delle etichette
                means = gmm.means_
                vols_per_cluster = means[:, 1]  # La seconda colonna è la volatilità rolling
                sorted_cluster_indices = np.argsort(vols_per_cluster)
                
                low_vol_idx = sorted_cluster_indices[0]
                mid_vol_idx = sorted_cluster_indices[1]
                high_vol_idx = sorted_cluster_indices[2]
                
                if latest_label == high_vol_idx:
                    regime = 1  # Orso (Alta Volatilità)
                elif latest_label == low_vol_idx:
                    # Tra i due cluster a volatilità più bassa, quello con ritorno medio maggiore è Toro
                    ret_low = means[low_vol_idx, 0]
                    ret_mid = means[mid_vol_idx, 0]
                    if ret_low > ret_mid:
                        regime = 0  # Toro
                    else:
                        regime = 2  # Laterale
                else:
                    ret_low = means[low_vol_idx, 0]
                    ret_mid = means[mid_vol_idx, 0]
                    if ret_mid > ret_low:
                        regime = 0  # Toro
                    else:
                        regime = 2  # Laterale
            except Exception:
                # Fallback deterministico a regole basato su volatilità e momentum
                recent_ret_avg = market_daily_ret.iloc[-20:].mean()
                if vol_today > vol_avg * 1.3:
                    regime = 1  # Orso
                elif recent_ret_avg > 0.0005:
                    regime = 0  # Toro
                else:
                    regime = 2  # Laterale
        else:
            # Fallback deterministico in caso di storico insufficiente
            recent_ret_avg = market_daily_ret.iloc[-gmm_data_len:].mean() if gmm_data_len > 0 else 0
            if vol_today > vol_avg * 1.3:
                regime = 1
            elif recent_ret_avg > 0.0005:
                regime = 0
            else:
                regime = 2
                
        # Regolazione asimmetrica delle soglie per la rete neurale
        base_threshold = self.probability_threshold
        raw_thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else base_threshold
        raw_thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else (1.0 - base_threshold)
        
        adjusted_thresh_long = 0.50 + (raw_thresh_long - 0.50) * vol_ratio
        adjusted_thresh_short = 0.50 - (0.50 - raw_thresh_short) * vol_ratio
        
        adjusted_thresh_long = float(np.clip(adjusted_thresh_long, 0.515, 0.56))
        adjusted_thresh_short = float(np.clip(adjusted_thresh_short, 0.44, 0.485))
        
        # Disabilita gli SHORT se la market breadth è superiore al 40%
        if market_breadth > 0.40:
            adjusted_thresh_short = 0.0

        # Ticker dedicati all'Arbitraggio Statistico (esclusi dal ranking standard)
        pair_tickers = {"V", "MA"}
        
        # ---------------------------------------------------------------------
        # MODULO 3: Statistical Arbitrage Overlay su Visa (V) / Mastercard (MA)
        # ---------------------------------------------------------------------
        stat_arb_signals = {}
        if "V" in historical_data and "MA" in historical_data:
            df_v = historical_data["V"]
            df_ma = historical_data["MA"]
            
            if len(df_v) >= 20 and len(df_ma) >= 20:
                close_v = df_v['close'].iloc[-20:]
                close_ma = df_ma['close'].iloc[-20:]
                
                # Calcolo spread basato su log-prezzi
                spread_series = np.log(close_ma) - np.log(close_v)
                latest_spread = spread_series.iloc[-1]
                
                mean_spread = spread_series.mean()
                std_spread = spread_series.std() + 1e-9
                z_score = (latest_spread - mean_spread) / std_spread
                
                has_position_ma = "MA" in portfolio.positions
                has_position_v = "V" in portfolio.positions
                
                if has_position_ma or has_position_v:
                    # Controllo chiusura arbitraggio per convergenza spread (Z-Score vicino a 0)
                    if abs(z_score) < 0.5:
                        if has_position_ma:
                            pos_ma = portfolio.positions["MA"]
                            action_ma = "SELL" if pos_ma.position_type == "LONG" else "BUY_TO_COVER"
                            stat_arb_signals["MA"] = {"action": action_ma, "probability": 0.5}
                        if has_position_v:
                            pos_v = portfolio.positions["V"]
                            action_v = "SELL" if pos_v.position_type == "LONG" else "BUY_TO_COVER"
                            stat_arb_signals["V"] = {"action": action_v, "probability": 0.5}
                    else:
                        # Controllo Stop Loss di arbitraggio per perdita combinata > 10%
                        pnl_comb = 0.0
                        total_cost = 0.0
                        if has_position_ma:
                            pos_ma = portfolio.positions["MA"]
                            pnl_comb += pos_ma.unrealized_pnl
                            total_cost += pos_ma.shares * pos_ma.entry_price
                        if has_position_v:
                            pos_v = portfolio.positions["V"]
                            pnl_comb += pos_v.unrealized_pnl
                            total_cost += pos_v.shares * pos_v.entry_price
                            
                        if total_cost > 0 and (pnl_comb / total_cost) <= -0.10:
                            logger.warning(f"[{current_date.strftime('%Y-%m-%d')}] StatArb Pair V/MA interrotto per Stop Loss combinato (-10% sforato). Chiudo.")
                            if has_position_ma:
                                action_ma = "SELL" if portfolio.positions["MA"].position_type == "LONG" else "BUY_TO_COVER"
                                stat_arb_signals["MA"] = {"action": action_ma, "probability": 0.5}
                            if has_position_v:
                                action_v = "SELL" if portfolio.positions["V"].position_type == "LONG" else "BUY_TO_COVER"
                                stat_arb_signals["V"] = {"action": action_v, "probability": 0.5}
                else:
                    # Ingressi di arbitraggio
                    if z_score > 2.0:
                        # MA è sopravvalutato rispetto a V -> SHORT MA, LONG V
                        stat_arb_signals["MA"] = {
                            "action": "SELL_SHORT",
                            "probability": 0.0,
                            "stop_loss_pct": 0.08,
                            "take_profit_pct": 0.15,
                            "confidence_multiplier": 1.0
                        }
                        stat_arb_signals["V"] = {
                            "action": "BUY",
                            "probability": 1.0,
                            "stop_loss_pct": 0.08,
                            "take_profit_pct": 0.15,
                            "confidence_multiplier": 1.0
                        }
                    elif z_score < -2.0:
                        # MA è sottovalutato rispetto a V -> LONG MA, SHORT V
                        stat_arb_signals["MA"] = {
                            "action": "BUY",
                            "probability": 1.0,
                            "stop_loss_pct": 0.08,
                            "take_profit_pct": 0.15,
                            "confidence_multiplier": 1.0
                        }
                        stat_arb_signals["V"] = {
                            "action": "SELL_SHORT",
                            "probability": 0.0,
                            "stop_loss_pct": 0.08,
                            "take_profit_pct": 0.15,
                            "confidence_multiplier": 1.0
                        }

        # ---------------------------------------------------------------------
        # COMPORTAMENTO ADATTIVO DEI MODULI CORE (CNN-Transformer vs Oscillatori)
        # ---------------------------------------------------------------------
        valid_tickers = []
        seq_features_list = []
        ticker_dfs = {}
        
        # Eseguiamo il comportamento in base al regime
        if regime == 2:
            # --- REGIME LATERALE (Mean-Reverting): usa RSI e Stocastico K ---
            for ticker, df in historical_data.items():
                if len(df) < self.lookback + 20:
                    continue
                # Escludiamo i ticker gestiti esclusivamente dall'arbitraggio statistico
                if ticker in pair_tickers:
                    continue
                    
                latest_row = df.iloc[-1]
                close = latest_row['close']
                atr = latest_row['atr_14']
                
                rsi = latest_row['rsi_14']
                
                # Calcolo stocastico veloce %K su 14 periodi
                low_14 = df['low'].iloc[-14:].min()
                high_14 = df['high'].iloc[-14:].max()
                stoch_k = ((close - low_14) / (high_14 - low_14 + 1e-9)) * 100.0
                
                stop_loss_pct = (3.5 * atr) / (close + 1e-9)
                stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.06))
                take_profit_pct = float(stop_loss_pct * 1.8)
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                        # Uscita se RSI ipercomprato o stocastico saturo
                        if rsi > 68 or stoch_k > 85 or days_held >= 10:
                            signals[ticker] = {"action": "SELL", "probability": 0.4}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": 0.5}
                    else: # SHORT
                        if rsi < 32 or stoch_k < 15 or days_held >= 10:
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": 0.6}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": 0.5}
                else:
                    # Ingressi basati su ipervenduto/ipercomprato estremo
                    if rsi < 28 and stoch_k < 12:
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": 0.75,
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "confidence_multiplier": 0.8
                        }
                    elif rsi > 72 and stoch_k > 88:
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": 0.25,
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "confidence_multiplier": 0.8
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": 0.5}
        else:
            # --- REGIMI TORO/ORSO: usa il modello CNN-Transformer standard ---
            for ticker, df in historical_data.items():
                if len(df) < self.lookback + 220:
                    continue
                # Escludiamo i ticker gestiti dall'arbitraggio statistico
                if ticker in pair_tickers:
                    continue
                    
                close_slice = df['close'].iloc[-slice_len:]
                high_slice = df['high'].iloc[-slice_len:]
                low_slice = df['low'].iloc[-slice_len:]
                volume_slice = df['volume'].iloc[-slice_len:]
                
                ret_series = close_slice.pct_change().fillna(0)
                vol_ret_series = volume_slice.pct_change().fillna(0)
                
                obv_series = (np.sign(ret_series) * volume_slice).fillna(0).cumsum()
                obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
                
                rsi_series = df['rsi_14'].iloc[-self.lookback:] / 100.0
                atr_series = df['atr_14'].iloc[-self.lookback:] / close_slice.iloc[-self.lookback:]
                
                bb_upper_s = df['bb_upper'].iloc[-self.lookback:]
                bb_lower_s = df['bb_lower'].iloc[-self.lookback:]
                bb_middle_s = df['bb_middle'].iloc[-self.lookback:]
                close_lookback = close_slice.iloc[-self.lookback:]
                
                bb_b_series = (close_lookback - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-9)
                bb_w_series = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-9)
                
                dist_200_series = (close_lookback - df['sma_200'].iloc[-self.lookback:]) / (df['sma_200'].iloc[-self.lookback:] + 1e-9)
                dist_50_series = (close_lookback - df['sma_50'].iloc[-self.lookback:]) / (df['sma_50'].iloc[-self.lookback:] + 1e-9)
                
                sma_5 = close_slice.rolling(5).mean()
                ema_12 = close_slice.ewm(span=12, adjust=False).mean()
                
                roc_10_series = close_slice.pct_change(10).fillna(0)
                
                low_14 = low_slice.rolling(14).min()
                high_14 = high_slice.rolling(14).max()
                stoch_k_series = ((close_slice - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
                
                sma_5_ratio_series = sma_5 / close_slice
                ema_12_ratio_series = ema_12 / close_slice
                
                volume_std_10 = volume_slice.rolling(10).std().fillna(0)
                volume_std_ratio_series = (volume_slice / (volume_std_10 + 1e-9)).fillna(1.0)
                
                market_relative_ret_series = ret_series - market_daily_ret
                market_relative_volume_series = volume_slice / (market_daily_vol + 1e-9)
                
                market_return_series = market_daily_ret
                market_volatility_series = market_rolling_vol
                
                check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
                if df.iloc[-self.lookback:][check_cols].isna().any().any():
                    signals[ticker] = {"action": "HOLD", "probability": 0.5}
                    continue
                    
                ret_arr = ret_series.values[-self.lookback:]
                vol_ret_arr = vol_ret_series.values[-self.lookback:]
                rsi_arr = rsi_series.values[-self.lookback:]
                bb_b_arr = bb_b_series.values[-self.lookback:]
                bb_w_arr = bb_w_series.values[-self.lookback:]
                atr_arr = atr_series.values[-self.lookback:]
                dist_200_arr = dist_200_series.values[-self.lookback:]
                dist_50_arr = dist_50_series.values[-self.lookback:]
                obv_ret_arr = obv_ret_series.values[-self.lookback:]
                roc_10_arr = roc_10_series.values[-self.lookback:]
                stoch_k_arr = stoch_k_series.values[-self.lookback:]
                sma_5_ratio_arr = sma_5_ratio_series.values[-self.lookback:]
                ema_12_ratio_arr = ema_12_ratio_series.values[-self.lookback:]
                volume_std_ratio_arr = volume_std_ratio_series.values[-self.lookback:]
                market_relative_ret_arr = market_relative_ret_series.values[-self.lookback:]
                market_relative_volume_arr = market_relative_volume_series.values[-self.lookback:]
                market_return_arr = market_return_series.values[-self.lookback:]
                market_volatility_arr = market_volatility_series.values[-self.lookback:]
                
                feature_arrays = {
                    'ret': ret_arr,
                    'vol_ret': vol_ret_arr,
                    'RSI_14': rsi_arr,
                    'Bollinger_%B': bb_b_arr,
                    'Bollinger_Width': bb_w_arr,
                    'ATRr_14': atr_arr,
                    'Dist_SMA200': dist_200_arr,
                    'Dist_SMA50': dist_50_arr,
                    'OBV_ret': obv_ret_arr,
                    'ROC_10': roc_10_arr,
                    'Stoch_K': stoch_k_arr,
                    'SMA_5_ratio': sma_5_ratio_arr,
                    'EMA_12_ratio': ema_12_ratio_arr,
                    'Volume_Std_Ratio': volume_std_ratio_arr,
                    'Market_Relative_Ret': market_relative_ret_arr,
                    'Market_Relative_Volume': market_relative_volume_arr,
                    'Market_Return': market_return_arr,
                    'Market_Volatility': market_volatility_arr
                }
                
                seq_feature_vectors = np.column_stack([feature_arrays[col] for col in self.feature_cols])
                seq_features_scaled = (seq_feature_vectors - self.mean) / self.std
                
                valid_tickers.append(ticker)
                seq_features_list.append(seq_features_scaled)
                ticker_dfs[ticker] = df

            if valid_tickers:
                batch_x = np.array(seq_features_list, dtype=np.float32)
                probs = self.model.predict(batch_x)
                
                ticker_probs = {ticker: float(prob) for ticker, prob in zip(valid_tickers, probs)}
                sorted_tickers = sorted(ticker_probs.items(), key=lambda x: x[1], reverse=True)
                
                N = len(valid_tickers)
                K = max(1, int(N * self.top_pct))
                top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
                bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
                
                K_out = max(1, int(N * self.exit_pct))
                top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
                bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
                
                for ticker, prob in zip(valid_tickers, probs):
                    df = ticker_dfs[ticker]
                    
                    latest_row = df.iloc[-1]
                    close = latest_row['close']
                    sma_200 = latest_row.get('sma_200', np.nan)
                    
                    is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                    
                    if self.trend_filter:
                        if is_uptrend:
                            thresh_long = adjusted_thresh_long
                            thresh_short = 0.30
                        else:
                            thresh_long = 0.70
                            thresh_short = adjusted_thresh_short
                    else:
                        thresh_long = adjusted_thresh_long
                        thresh_short = adjusted_thresh_short
                    
                    if ticker in portfolio.positions:
                        pos = portfolio.positions[ticker]
                        days_held = (current_date - pos.entry_date).days
                        
                        # Escludiamo le posizioni aperte da arbitraggio (non hanno probability nel portfolio o le gestiamo separatamente)
                        # Nota: pos.shares * pos.entry_price > 0 è sempre vero, ma il Pair Trading ha probability fisse a 0.5.
                        if pos.position_type == "LONG":
                             should_exit = (
                                 prob < self.exit_long_threshold or 
                                 (self.trend_filter and not is_uptrend) or
                                 (ticker not in top_out_tickers and days_held >= 3)
                             )
                             if should_exit:
                                 signals[ticker] = {"action": "SELL", "probability": float(prob)}
                             else:
                                 signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                        else: # SHORT
                             should_exit = (
                                 prob > self.exit_short_threshold or 
                                 (self.trend_filter and is_uptrend) or
                                 (ticker not in bottom_out_tickers and days_held >= 3)
                             )
                             if should_exit:
                                 signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                             else:
                                 signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:
                        atr = latest_row['atr_14']
                        
                        stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                        stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                        take_profit_pct = float(stop_loss_pct * 2.0)
                        
                        trailing_stop_pct = (2.0 * atr) / (close + 1e-9)
                        trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.05))
                        
                        if ticker in top_K_tickers and prob >= thresh_long:
                            # Kelly Sizing Dinamico per Confidenza
                            conf_mult = 1.0 + (prob - thresh_long) * 5.0
                            conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                            
                            signals[ticker] = {
                                "action": "BUY",
                                "probability": float(prob),
                                "stop_loss_pct": stop_loss_pct,
                                "take_profit_pct": take_profit_pct,
                                "trailing_stop_pct": trailing_stop_pct,
                                "confidence_multiplier": conf_mult
                            }
                        elif ticker in bottom_K_tickers and prob <= thresh_short:
                            conf_mult = 1.0 + (thresh_short - prob) * 5.0
                            conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                            
                            signals[ticker] = {
                                "action": "SELL_SHORT",
                                "probability": float(prob),
                                "stop_loss_pct": stop_loss_pct,
                                "take_profit_pct": take_profit_pct,
                                "trailing_stop_pct": trailing_stop_pct,
                                "confidence_multiplier": conf_mult
                            }
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                            
        # Sovrascriviamo o uniamo i segnali di arbitraggio statistico con priorità assoluta per MA e V
        for ticker, sig_info in stat_arb_signals.items():
            signals[ticker] = sig_info
        return signals


class NeuralNetworkV8Strategy(BaseStrategy):
    """
    Strategia quantitativa V8 - Edizione Ottimizzata e Ultra-Stabile.
    Risolve le instabilità della V7 introducendo:
    1. Capital Allocation Dinamica Stabile: max_slots fisso a 5 per evitare stop-loss giganti,
       modulando solo la riserva di cash in base alla Market Breadth.
    2. Regime Adaptivity Smussato (Regime Smoothing): Filtro a maggioranza rolling a 5 giorni
       sulla classificazione di regime macro per prevenire oscillazioni quotidiane spurne,
       stabilizzando il comportamento predittivo.
    3. Statistical Arbitrage Overlay Robusto: SL/TP individuali disabilitati (None) per MA/V,
       con maggiore selettività (z-score ingresso 2.2, uscita 0.3) e stop loss combinato del 10%.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model.pth", 
        probability_threshold: float = 0.525,
        ranking_mode: bool = True,
        top_pct: float = 0.03,
        exit_pct: float = 0.60,
        exit_long_threshold: float = 0.485,
        exit_short_threshold: float = 0.515,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v6.model import NeuralNetworkV6
        
        self.probability_threshold = probability_threshold
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        
        # Inizializziamo i parametri di allocazione dinamica che l'engine leggerà
        self.current_max_slots = 5
        self.current_cash_reserve_pct = 0.0
        
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v6" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v8: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV8Strategy")
        logger.info(f"Caricamento del modello CNN-Transformer v6 per la strategia V8 da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        self.model = NeuralNetworkV6(
            input_dim=self.input_dim, 
            lookback=self.lookback,
            d_model=state.get("d_model", 64),
            nhead=state.get("nhead", 4),
            num_layers=state.get("num_layers", 2),
            alpha=state.get("alpha", 50.0)
        )
        self.model.load(str(model_path))
        
        logger.info(f"Modello CNN-Transformer caricato con successo per la strategia V8.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}
        logger = logging.getLogger("NeuralNetworkV8Strategy")
        
        # ---------------------------------------------------------------------
        # MODULO 1: Calcolo Market Breadth ed Esposizione Dinamica Stabile
        # ---------------------------------------------------------------------
        slice_len = self.lookback + 50
        ticker_rets = {}
        ticker_vols = {}
        
        uptrend_counts = 0
        total_valid = 0
        
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
            ticker_rets[ticker] = df['close'].iloc[-slice_len:].pct_change().fillna(0)
            ticker_vols[ticker] = df['volume'].iloc[-slice_len:]
            
            if len(df) >= 200:
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                if pd.notna(sma_200):
                    total_valid += 1
                    if close >= sma_200:
                        uptrend_counts += 1
            
        if not ticker_rets:
            return signals
            
        market_breadth = (uptrend_counts / total_valid) if total_valid > 0 else 0.5
        
        # Capital Allocation Dinamica Stabile: max_slots fisso a 5, cash reserve a 50% se breadth < 30%
        self.current_max_slots = 5
        if market_breadth < 0.30:
            self.current_cash_reserve_pct = 0.50
        else:
            self.current_cash_reserve_pct = 0.0
            
        df_rets_all = pd.DataFrame(ticker_rets)
        df_vols_all = pd.DataFrame(ticker_vols)
        
        market_daily_ret = df_rets_all.mean(axis=1)
        market_daily_vol = df_vols_all.mean(axis=1)
        
        # Volatilità rolling a 20 giorni
        market_rolling_vol = market_daily_ret.rolling(20).std().fillna(0.0)
        
        # Volatilità odierna vs volatilità media storica
        vol_avg = market_rolling_vol.mean()
        vol_today = market_rolling_vol.iloc[-1]
        vol_ratio = vol_today / (vol_avg + 1e-9)
        vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
        
        # ---------------------------------------------------------------------
        # MODULO 2: Regime Clustering non supervisionato (GMM / Fallback / Regime Smoothing)
        # ---------------------------------------------------------------------
        regime = 0  # 0: Toro (Trending Up), 1: Orso (Trending Down / High Vol), 2: Laterale (Mean-Reverting)
        
        # Costruiamo il dataset delle ultime 250 giornate per il clustering
        gmm_data_len = min(250, len(market_daily_ret))
        if gmm_data_len >= 50:
            features = np.column_stack([
                market_daily_ret.iloc[-gmm_data_len:].values,
                market_rolling_vol.iloc[-gmm_data_len:].values,
                market_daily_vol.iloc[-gmm_data_len:].values
            ])
            
            try:
                from sklearn.mixture import GaussianMixture
                # Disattiviamo temporaneamente i log di convergenza rumorosi di sklearn
                gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, max_iter=100)
                gmm.fit(features)
                labels = gmm.predict(features)
                
                # Ordiniamo i cluster per volatilità crescente per garantire consistenza di significato delle etichette
                means = gmm.means_
                vols_per_cluster = means[:, 1]  # La seconda colonna è la volatilità rolling
                sorted_cluster_indices = np.argsort(vols_per_cluster)
                
                low_vol_idx = sorted_cluster_indices[0]
                mid_vol_idx = sorted_cluster_indices[1]
                high_vol_idx = sorted_cluster_indices[2]
                
                # Regime Smoothing: Filtro a maggioranza rolling a 5 giorni
                regimes = []
                for t in range(-5, 0):
                    label_t = labels[t]
                    if label_t == high_vol_idx:
                        regimes.append(1)
                    elif label_t == low_vol_idx:
                        ret_low = means[low_vol_idx, 0]
                        ret_mid = means[mid_vol_idx, 0]
                        regimes.append(0 if ret_low > ret_mid else 2)
                    else:
                        ret_low = means[low_vol_idx, 0]
                        ret_mid = means[mid_vol_idx, 0]
                        regimes.append(0 if ret_mid > ret_low else 2)
                from collections import Counter
                regime = Counter(regimes).most_common(1)[0][0]
            except Exception:
                # Fallback rolling a 5 giorni su regole basate su volatilità e momentum
                regimes = []
                for t in range(-5, 0):
                    vol_t = market_rolling_vol.iloc[t]
                    vol_avg_t = market_rolling_vol.iloc[:t].mean() if t < -1 else vol_avg
                    recent_ret_avg_t = market_daily_ret.iloc[t-20:t].mean() if t < -1 else market_daily_ret.iloc[-20:].mean()
                    
                    if vol_t > vol_avg_t * 1.3:
                        regimes.append(1)
                    elif recent_ret_avg_t > 0.0005:
                        regimes.append(0)
                    else:
                        regimes.append(2)
                from collections import Counter
                regime = Counter(regimes).most_common(1)[0][0]
        else:
            # Fallback rolling a 5 giorni in caso di storico insufficiente
            regimes = []
            for t in range(max(-5, -gmm_data_len), 0):
                vol_t = market_rolling_vol.iloc[t]
                vol_avg_t = market_rolling_vol.iloc[:t].mean() if t < -1 else vol_avg
                recent_ret_avg_t = market_daily_ret.iloc[:t].mean() if t < -1 else market_daily_ret.iloc[-gmm_data_len:].mean()
                if vol_t > vol_avg_t * 1.3:
                    regimes.append(1)
                elif recent_ret_avg_t > 0.0005:
                    regimes.append(0)
                else:
                    regimes.append(2)
            if regimes:
                from collections import Counter
                regime = Counter(regimes).most_common(1)[0][0]
            else:
                regime = 2
                
        # Regolazione asimmetrica delle soglie per la rete neurale
        base_threshold = self.probability_threshold
        raw_thresh_long = self.probability_threshold_long if self.probability_threshold_long is not None else base_threshold
        raw_thresh_short = self.probability_threshold_short if self.probability_threshold_short is not None else (1.0 - base_threshold)
        
        adjusted_thresh_long = 0.50 + (raw_thresh_long - 0.50) * vol_ratio
        adjusted_thresh_short = 0.50 - (0.50 - raw_thresh_short) * vol_ratio
        
        adjusted_thresh_long = float(np.clip(adjusted_thresh_long, 0.515, 0.56))
        adjusted_thresh_short = float(np.clip(adjusted_thresh_short, 0.44, 0.485))
        
        # Disabilita gli SHORT se la market breadth è superiore al 40%
        if market_breadth > 0.40:
            adjusted_thresh_short = 0.0
            
        # ---------------------------------------------------------------------
        # MODULO 3: Statistical Arbitrage Overlay su Visa (V) / Mastercard (MA)
        # ---------------------------------------------------------------------
        pair_tickers = {"V", "MA"}
        stat_arb_signals = {}
        if "V" in historical_data and "MA" in historical_data:
            df_v = historical_data["V"]
            df_ma = historical_data["MA"]
            
            if len(df_v) >= 20 and len(df_ma) >= 20:
                close_v = df_v['close'].iloc[-20:]
                close_ma = df_ma['close'].iloc[-20:]
                
                spread_series = np.log(close_ma) - np.log(close_v)
                latest_spread = spread_series.iloc[-1]
                
                mean_spread = spread_series.mean()
                std_spread = spread_series.std() + 1e-9
                z_score = (latest_spread - mean_spread) / std_spread
                
                has_position_ma = "MA" in portfolio.positions
                has_position_v = "V" in portfolio.positions
                
                if has_position_ma or has_position_v:
                    # Controllo chiusura arbitraggio per convergenza spread (Z-Score vicino a 0)
                    if abs(z_score) < 0.3:
                        if has_position_ma:
                            pos_ma = portfolio.positions["MA"]
                            action_ma = "SELL" if pos_ma.position_type == "LONG" else "BUY_TO_COVER"
                            stat_arb_signals["MA"] = {"action": action_ma, "probability": 0.5}
                        if has_position_v:
                            pos_v = portfolio.positions["V"]
                            action_v = "SELL" if pos_v.position_type == "LONG" else "BUY_TO_COVER"
                            stat_arb_signals["V"] = {"action": action_v, "probability": 0.5}
                    else:
                        # Controllo Stop Loss di arbitraggio per perdita combinata > 10%
                        pnl_comb = 0.0
                        total_cost = 0.0
                        if has_position_ma:
                            pos_ma = portfolio.positions["MA"]
                            pnl_comb += pos_ma.unrealized_pnl
                            total_cost += pos_ma.shares * pos_ma.entry_price
                        if has_position_v:
                            pos_v = portfolio.positions["V"]
                            pnl_comb += pos_v.unrealized_pnl
                            total_cost += pos_v.shares * pos_v.entry_price
                            
                        if total_cost > 0 and (pnl_comb / total_cost) <= -0.10:
                            logger.warning(f"[{current_date.strftime('%Y-%m-%d')}] StatArb Pair V/MA interrotto per Stop Loss combinato (-10% sforato). Chiudo.")
                            if has_position_ma:
                                action_ma = "SELL" if portfolio.positions["MA"].position_type == "LONG" else "BUY_TO_COVER"
                                stat_arb_signals["MA"] = {"action": action_ma, "probability": 0.5}
                            if has_position_v:
                                action_v = "SELL" if portfolio.positions["V"].position_type == "LONG" else "BUY_TO_COVER"
                                stat_arb_signals["V"] = {"action": action_v, "probability": 0.5}
                else:
                    # Ingressi di arbitraggio
                    if z_score > 2.2:
                        # MA è sopravvalutato rispetto a V -> SHORT MA, LONG V
                        stat_arb_signals["MA"] = {
                            "action": "SELL_SHORT",
                            "probability": 0.0,
                            "stop_loss_pct": None,
                            "take_profit_pct": None,
                            "confidence_multiplier": 1.0
                        }
                        stat_arb_signals["V"] = {
                            "action": "BUY",
                            "probability": 1.0,
                            "stop_loss_pct": None,
                            "take_profit_pct": None,
                            "confidence_multiplier": 1.0
                        }
                    elif z_score < -2.2:
                        # MA è sottovalutato rispetto a V -> LONG MA, SHORT V
                        stat_arb_signals["MA"] = {
                            "action": "BUY",
                            "probability": 1.0,
                            "stop_loss_pct": None,
                            "take_profit_pct": None,
                            "confidence_multiplier": 1.0
                        }
                        stat_arb_signals["V"] = {
                            "action": "SELL_SHORT",
                            "probability": 0.0,
                            "stop_loss_pct": None,
                            "take_profit_pct": None,
                            "confidence_multiplier": 1.0
                        }

        # ---------------------------------------------------------------------
        # COMPORTAMENTO ADATTIVO DEI MODULI CORE (CNN-Transformer vs Oscillatori)
        # ---------------------------------------------------------------------
        valid_tickers = []
        seq_features_list = []
        ticker_dfs = {}
        
        # Eseguiamo il comportamento in base al regime
        if regime == 2:
            # --- REGIME LATERALE (Mean-Reverting): usa RSI e Stocastico K ---
            for ticker, df in historical_data.items():
                if len(df) < self.lookback + 20:
                    continue
                if ticker in pair_tickers:
                    continue
                    
                latest_row = df.iloc[-1]
                close = latest_row['close']
                atr = latest_row['atr_14']
                
                rsi = latest_row['rsi_14']
                
                low_14 = df['low'].iloc[-14:].min()
                high_14 = df['high'].iloc[-14:].max()
                stoch_k = ((close - low_14) / (high_14 - low_14 + 1e-9)) * 100.0
                
                stop_loss_pct = (3.5 * atr) / (close + 1e-9)
                stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.06))
                take_profit_pct = float(stop_loss_pct * 1.8)
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                        if rsi > 68 or stoch_k > 85 or days_held >= 10:
                            signals[ticker] = {"action": "SELL", "probability": 0.4}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": 0.5}
                    else: # SHORT
                        if rsi < 32 or stoch_k < 15 or days_held >= 10:
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": 0.6}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": 0.5}
                else:
                    if rsi < 28 and stoch_k < 12:
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": 0.75,
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "confidence_multiplier": 0.8
                        }
                    elif rsi > 72 and stoch_k > 88:
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": 0.25,
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "confidence_multiplier": 0.8
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": 0.5}
        else:
            # --- REGIMI TORO/ORSO: usa il modello CNN-Transformer standard ---
            for ticker, df in historical_data.items():
                if len(df) < self.lookback + 220:
                    continue
                if ticker in pair_tickers:
                    continue
                    
                close_slice = df['close'].iloc[-slice_len:]
                high_slice = df['high'].iloc[-slice_len:]
                low_slice = df['low'].iloc[-slice_len:]
                volume_slice = df['volume'].iloc[-slice_len:]
                
                ret_series = close_slice.pct_change().fillna(0)
                vol_ret_series = volume_slice.pct_change().fillna(0)
                
                obv_series = (np.sign(ret_series) * volume_slice).fillna(0).cumsum()
                obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
                
                rsi_series = df['rsi_14'].iloc[-self.lookback:] / 100.0
                atr_series = df['atr_14'].iloc[-self.lookback:] / close_slice.iloc[-self.lookback:]
                
                bb_upper_s = df['bb_upper'].iloc[-self.lookback:]
                bb_lower_s = df['bb_lower'].iloc[-self.lookback:]
                bb_middle_s = df['bb_middle'].iloc[-self.lookback:]
                close_lookback = close_slice.iloc[-self.lookback:]
                
                bb_b_series = (close_lookback - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-9)
                bb_w_series = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-9)
                
                dist_200_series = (close_lookback - df['sma_200'].iloc[-self.lookback:]) / (df['sma_200'].iloc[-self.lookback:] + 1e-9)
                dist_50_series = (close_lookback - df['sma_50'].iloc[-self.lookback:]) / (df['sma_50'].iloc[-self.lookback:] + 1e-9)
                
                sma_5 = close_slice.rolling(5).mean()
                ema_12 = close_slice.ewm(span=12, adjust=False).mean()
                
                roc_10_series = close_slice.pct_change(10).fillna(0)
                
                low_14 = low_slice.rolling(14).min()
                high_14 = high_slice.rolling(14).max()
                stoch_k_series = ((close_slice - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
                
                sma_5_ratio_series = sma_5 / close_slice
                ema_12_ratio_series = ema_12 / close_slice
                
                volume_std_10 = volume_slice.rolling(10).std().fillna(0)
                volume_std_ratio_series = (volume_slice / (volume_std_10 + 1e-9)).fillna(1.0)
                
                market_relative_ret_series = ret_series - market_daily_ret
                market_relative_volume_series = volume_slice / (market_daily_vol + 1e-9)
                
                market_return_series = market_daily_ret
                market_volatility_series = market_rolling_vol
                
                check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
                if df.iloc[-self.lookback:][check_cols].isna().any().any():
                    signals[ticker] = {"action": "HOLD", "probability": 0.5}
                    continue
                    
                ret_arr = ret_series.values[-self.lookback:]
                vol_ret_arr = vol_ret_series.values[-self.lookback:]
                rsi_arr = rsi_series.values[-self.lookback:]
                bb_b_arr = bb_b_series.values[-self.lookback:]
                bb_w_arr = bb_w_series.values[-self.lookback:]
                atr_arr = atr_series.values[-self.lookback:]
                dist_200_arr = dist_200_series.values[-self.lookback:]
                dist_50_arr = dist_50_series.values[-self.lookback:]
                obv_ret_arr = obv_ret_series.values[-self.lookback:]
                roc_10_arr = roc_10_series.values[-self.lookback:]
                stoch_k_arr = stoch_k_series.values[-self.lookback:]
                sma_5_ratio_arr = sma_5_ratio_series.values[-self.lookback:]
                ema_12_ratio_arr = ema_12_ratio_series.values[-self.lookback:]
                volume_std_ratio_arr = volume_std_ratio_series.values[-self.lookback:]
                market_relative_ret_arr = market_relative_ret_series.values[-self.lookback:]
                market_relative_volume_arr = market_relative_volume_series.values[-self.lookback:]
                market_return_arr = market_return_series.values[-self.lookback:]
                market_volatility_arr = market_volatility_series.values[-self.lookback:]
                
                feature_arrays = {
                    'ret': ret_arr,
                    'vol_ret': vol_ret_arr,
                    'RSI_14': rsi_arr,
                    'Bollinger_%B': bb_b_arr,
                    'Bollinger_Width': bb_w_arr,
                    'ATRr_14': atr_arr,
                    'Dist_SMA200': dist_200_arr,
                    'Dist_SMA50': dist_50_arr,
                    'OBV_ret': obv_ret_arr,
                    'ROC_10': roc_10_arr,
                    'Stoch_K': stoch_k_arr,
                    'SMA_5_ratio': sma_5_ratio_arr,
                    'EMA_12_ratio': ema_12_ratio_arr,
                    'Volume_Std_Ratio': volume_std_ratio_arr,
                    'Market_Relative_Ret': market_relative_ret_arr,
                    'Market_Relative_Volume': market_relative_volume_arr,
                    'Market_Return': market_return_arr,
                    'Market_Volatility': market_volatility_arr
                }
                
                seq_feature_vectors = np.column_stack([feature_arrays[col] for col in self.feature_cols])
                seq_features_scaled = (seq_feature_vectors - self.mean) / self.std
                
                valid_tickers.append(ticker)
                seq_features_list.append(seq_features_scaled)
                ticker_dfs[ticker] = df

            if valid_tickers:
                batch_x = np.array(seq_features_list, dtype=np.float32)
                probs = self.model.predict(batch_x)
                
                ticker_probs = {ticker: float(prob) for ticker, prob in zip(valid_tickers, probs)}
                sorted_tickers = sorted(ticker_probs.items(), key=lambda x: x[1], reverse=True)
                
                N = len(valid_tickers)
                K = max(1, int(N * self.top_pct))
                top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
                bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
                
                K_out = max(1, int(N * self.exit_pct))
                top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
                bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
                
                for ticker, prob in zip(valid_tickers, probs):
                    df = ticker_dfs[ticker]
                    
                    latest_row = df.iloc[-1]
                    close = latest_row['close']
                    sma_200 = latest_row.get('sma_200', np.nan)
                    
                    is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                    
                    if self.trend_filter:
                        if is_uptrend:
                            thresh_long = adjusted_thresh_long
                            thresh_short = 0.30
                        else:
                            thresh_long = 0.70
                            thresh_short = adjusted_thresh_short
                    else:
                        thresh_long = adjusted_thresh_long
                        thresh_short = adjusted_thresh_short
                    
                    if ticker in portfolio.positions:
                        pos = portfolio.positions[ticker]
                        days_held = (current_date - pos.entry_date).days
                        
                        if pos.position_type == "LONG":
                             should_exit = (
                                 prob < self.exit_long_threshold or 
                                 (self.trend_filter and not is_uptrend) or
                                 (ticker not in top_out_tickers and days_held >= 3)
                             )
                             if should_exit:
                                 signals[ticker] = {"action": "SELL", "probability": float(prob)}
                             else:
                                 signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                        else: # SHORT
                             should_exit = (
                                 prob > self.exit_short_threshold or 
                                 (self.trend_filter and is_uptrend) or
                                 (ticker not in bottom_out_tickers and days_held >= 3)
                             )
                             if should_exit:
                                 signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                             else:
                                 signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:
                        atr = latest_row['atr_14']
                        
                        stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                        stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                        take_profit_pct = float(stop_loss_pct * 2.0)
                        
                        trailing_stop_pct = (2.0 * atr) / (close + 1e-9)
                        trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.05))
                        
                        if ticker in top_K_tickers and prob >= thresh_long:
                            conf_mult = 1.0 + (prob - thresh_long) * 5.0
                            conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                            
                            signals[ticker] = {
                                "action": "BUY",
                                "probability": float(prob),
                                "stop_loss_pct": stop_loss_pct,
                                "take_profit_pct": take_profit_pct,
                                "trailing_stop_pct": trailing_stop_pct,
                                "confidence_multiplier": conf_mult
                            }
                        elif ticker in bottom_K_tickers and prob <= thresh_short:
                            conf_mult = 1.0 + (thresh_short - prob) * 5.0
                            conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                            
                            signals[ticker] = {
                                "action": "SELL_SHORT",
                                "probability": float(prob),
                                "stop_loss_pct": stop_loss_pct,
                                "take_profit_pct": take_profit_pct,
                                "trailing_stop_pct": trailing_stop_pct,
                                "confidence_multiplier": conf_mult
                            }
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                            
        for ticker, sig_info in stat_arb_signals.items():
            signals[ticker] = sig_info
        return signals


class NeuralNetworkV9Strategy(BaseStrategy):
    """
    Strategia quantitativa V9 - L'Edizione Definitiva (Ritorno alla Base V6 Potenziata).
    Riconosce che i moduli di GMM/RSI mean-reverting e il Pair Trading MA/V introcessi in V7/V8
    hanno declassato le performance spegnendo la potenza del Transformer e bloccando il capitale.
    
    Questa versione ripristina la struttura vincente della V6 e la ottimizza con:
    1. AI Always-On: La rete neurale CNN-Transformer guida le decisioni in ogni regime di mercato.
    2. Adattività tramite Selettività (Filtro Breadth Dinamico): Invece di tenere il cash fermo a marcire,
       regoliamo la soglia probabilistica di ingresso in base alla salute del mercato (Market Breadth):
       - Mercato solido (Breadth >= 40%): Soglia standard a 0.525 (massima partecipazione).
       - Mercato volatile/bear (Breadth < 40%): La soglia si alza a 0.540 (massima selettività).
    3. Capitale Centralizzato senza Riserve: Massima potenza allocativa sul Global Pool.
    4. Trailing Stop ATR ed Exit Buffer di 3 giorni ereditati dalla V6 per lasciar correre i profitti.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model.pth", 
        probability_threshold: float = 0.525,
        ranking_mode: bool = True,
        top_pct: float = 0.03,
        exit_pct: float = 0.60,
        exit_long_threshold: float = 0.485,
        exit_short_threshold: float = 0.515,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v6.model import NeuralNetworkV6
        
        self.probability_threshold = probability_threshold
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        
        # Stabilità e centralizzazione
        self.current_max_slots = 5
        self.current_cash_reserve_pct = 0.0
        
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v6" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v9: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV9Strategy")
        logger.info(f"Caricamento del modello CNN-Transformer v6 per la strategia V9 da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        self.model = NeuralNetworkV6(
            input_dim=self.input_dim, 
            lookback=self.lookback,
            d_model=state.get("d_model", 64),
            nhead=state.get("nhead", 4),
            num_layers=state.get("num_layers", 2),
            alpha=state.get("alpha", 50.0)
        )
        self.model.load(str(model_path))
        
        logger.info(f"Modello CNN-Transformer caricato con successo per la strategia V9.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}
        
        # 1. Calcolo Breadth di Mercato (percentuale di titoli sopra la SMA 200)
        slice_len = self.lookback + 50
        ticker_rets = {}
        ticker_vols = {}
        
        uptrend_counts = 0
        total_valid = 0
        
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
            ticker_rets[ticker] = df['close'].iloc[-slice_len:].pct_change().fillna(0)
            ticker_vols[ticker] = df['volume'].iloc[-slice_len:]
            
            if len(df) >= 200:
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                if pd.notna(sma_200):
                    total_valid += 1
                    if close >= sma_200:
                        uptrend_counts += 1
            
        if not ticker_rets:
            return signals
            
        market_breadth = (uptrend_counts / total_valid) if total_valid > 0 else 0.5
        
        df_rets_all = pd.DataFrame(ticker_rets)
        df_vols_all = pd.DataFrame(ticker_vols)
        
        market_daily_ret = df_rets_all.mean(axis=1)
        market_daily_vol = df_vols_all.mean(axis=1)
        
        # Volatilità rolling del mercato a 20 giorni
        market_rolling_vol = market_daily_ret.rolling(20).std().fillna(0.0)
        
        vol_avg = market_rolling_vol.mean()
        vol_today = market_rolling_vol.iloc[-1]
        vol_ratio = vol_today / (vol_avg + 1e-9)
        vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
        
        # 2. Regolazione Adattiva delle Soglie in base alla salute del mercato (Market Breadth)
        base_threshold = self.probability_threshold
        
        # Se il mercato è debole o instabile (Breadth < 40%), aumentiamo la selettività della soglia per proteggere il capitale
        if market_breadth < 0.40:
            # Mercato debole: alza la soglia di ingresso per i LONG a 0.540
            raw_thresh_long = 0.540
            raw_thresh_short = 0.460
        else:
            # Mercato forte: soglia standard a 0.525 per massima partecipazione
            raw_thresh_long = base_threshold
            raw_thresh_short = 1.0 - base_threshold
            
        if self.probability_threshold_long is not None:
            raw_thresh_long = self.probability_threshold_long
        if self.probability_threshold_short is not None:
            raw_thresh_short = self.probability_threshold_short
            
        adjusted_thresh_long = 0.50 + (raw_thresh_long - 0.50) * vol_ratio
        adjusted_thresh_short = 0.50 - (0.50 - raw_thresh_short) * vol_ratio
        
        adjusted_thresh_long = float(np.clip(adjusted_thresh_long, 0.515, 0.56))
        adjusted_thresh_short = float(np.clip(adjusted_thresh_short, 0.44, 0.485))
        
        # Disabilita gli SHORT se la market breadth è superiore al 40% (regime rialzista/rialzo forte)
        if market_breadth > 0.40:
            adjusted_thresh_short = 0.0
            
        valid_tickers = []
        seq_features_list = []
        ticker_dfs = {}

        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue

            close_slice = df['close'].iloc[-slice_len:]
            high_slice = df['high'].iloc[-slice_len:]
            low_slice = df['low'].iloc[-slice_len:]
            volume_slice = df['volume'].iloc[-slice_len:]
            
            ret_series = close_slice.pct_change().fillna(0)
            vol_ret_series = volume_slice.pct_change().fillna(0)
            
            obv_series = (np.sign(ret_series) * volume_slice).fillna(0).cumsum()
            obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            
            rsi_series = df['rsi_14'].iloc[-self.lookback:] / 100.0
            atr_series = df['atr_14'].iloc[-self.lookback:] / close_slice.iloc[-self.lookback:]
            
            bb_upper_s = df['bb_upper'].iloc[-self.lookback:]
            bb_lower_s = df['bb_lower'].iloc[-self.lookback:]
            bb_middle_s = df['bb_middle'].iloc[-self.lookback:]
            close_lookback = close_slice.iloc[-self.lookback:]
            
            bb_b_series = (close_lookback - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-9)
            bb_w_series = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-9)
            
            dist_200_series = (close_lookback - df['sma_200'].iloc[-self.lookback:]) / (df['sma_200'].iloc[-self.lookback:] + 1e-9)
            dist_50_series = (close_lookback - df['sma_50'].iloc[-self.lookback:]) / (df['sma_50'].iloc[-self.lookback:] + 1e-9)
            
            sma_5 = close_slice.rolling(5).mean()
            ema_12 = close_slice.ewm(span=12, adjust=False).mean()
            
            roc_10_series = close_slice.pct_change(10).fillna(0)
            
            low_14 = low_slice.rolling(14).min()
            high_14 = high_slice.rolling(14).max()
            stoch_k_series = ((close_slice - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
            
            sma_5_ratio_series = sma_5 / close_slice
            ema_12_ratio_series = ema_12 / close_slice
            
            volume_std_10 = volume_slice.rolling(10).std().fillna(0)
            volume_std_ratio_series = (volume_slice / (volume_std_10 + 1e-9)).fillna(1.0)
            
            market_relative_ret_series = ret_series - market_daily_ret
            market_relative_volume_series = volume_slice / (market_daily_vol + 1e-9)
            
            market_return_series = market_daily_ret
            market_volatility_series = market_rolling_vol
            
            check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
            if df.iloc[-self.lookback:][check_cols].isna().any().any():
                signals[ticker] = {"action": "HOLD", "probability": 0.5}
                continue
                
            ret_arr = ret_series.values[-self.lookback:]
            vol_ret_arr = vol_ret_series.values[-self.lookback:]
            rsi_arr = rsi_series.values[-self.lookback:]
            bb_b_arr = bb_b_series.values[-self.lookback:]
            bb_w_arr = bb_w_series.values[-self.lookback:]
            atr_arr = atr_series.values[-self.lookback:]
            dist_200_arr = dist_200_series.values[-self.lookback:]
            dist_50_arr = dist_50_series.values[-self.lookback:]
            obv_ret_arr = obv_ret_series.values[-self.lookback:]
            roc_10_arr = roc_10_series.values[-self.lookback:]
            stoch_k_arr = stoch_k_series.values[-self.lookback:]
            sma_5_ratio_arr = sma_5_ratio_series.values[-self.lookback:]
            ema_12_ratio_arr = ema_12_ratio_series.values[-self.lookback:]
            volume_std_ratio_arr = volume_std_ratio_series.values[-self.lookback:]
            market_relative_ret_arr = market_relative_ret_series.values[-self.lookback:]
            market_relative_volume_arr = market_relative_volume_series.values[-self.lookback:]
            market_return_arr = market_return_series.values[-self.lookback:]
            market_volatility_arr = market_volatility_series.values[-self.lookback:]
            
            feature_arrays = {
                'ret': ret_arr,
                'vol_ret': vol_ret_arr,
                'RSI_14': rsi_arr,
                'Bollinger_%B': bb_b_arr,
                'Bollinger_Width': bb_w_arr,
                'ATRr_14': atr_arr,
                'Dist_SMA200': dist_200_arr,
                'Dist_SMA50': dist_50_arr,
                'OBV_ret': obv_ret_arr,
                'ROC_10': roc_10_arr,
                'Stoch_K': stoch_k_arr,
                'SMA_5_ratio': sma_5_ratio_arr,
                'EMA_12_ratio': ema_12_ratio_arr,
                'Volume_Std_Ratio': volume_std_ratio_arr,
                'Market_Relative_Ret': market_relative_ret_arr,
                'Market_Relative_Volume': market_relative_volume_arr,
                'Market_Return': market_return_arr,
                'Market_Volatility': market_volatility_arr
            }
            
            seq_feature_vectors = np.column_stack([feature_arrays[col] for col in self.feature_cols])
            seq_features_scaled = (seq_feature_vectors - self.mean) / self.std
            
            valid_tickers.append(ticker)
            seq_features_list.append(seq_features_scaled)
            ticker_dfs[ticker] = df
 
        if not valid_tickers:
            return signals

        batch_x = np.array(seq_features_list, dtype=np.float32)
        probs = self.model.predict(batch_x)
        
        if self.ranking_mode:
            ticker_probs = {ticker: float(prob) for ticker, prob in zip(valid_tickers, probs)}
            sorted_tickers = sorted(ticker_probs.items(), key=lambda x: x[1], reverse=True)
            
            N = len(valid_tickers)
            K = max(1, int(N * self.top_pct))
            top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
            bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
            
            K_out = max(1, int(N * self.exit_pct))
            top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
            bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
            
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = adjusted_thresh_long
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = adjusted_thresh_short
                else:
                    thresh_long = adjusted_thresh_long
                    thresh_short = adjusted_thresh_short
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                         should_exit = (
                             prob < self.exit_long_threshold or 
                             (self.trend_filter and not is_uptrend) or
                             (ticker not in top_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "SELL", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else: # SHORT
                         should_exit = (
                             prob > self.exit_short_threshold or 
                             (self.trend_filter and is_uptrend) or
                             (ticker not in bottom_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    take_profit_pct = float(stop_loss_pct * 2.0)
                    
                    # Calcolo trailing stop dinamico basato su ATR (2.0 * ATR)
                    trailing_stop_pct = (2.0 * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.05))
                    
                    if ticker in top_K_tickers and prob >= thresh_long:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        conf_mult = 1.0 + (prob - thresh_long) * 5.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif ticker in bottom_K_tickers and prob <= thresh_short:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        conf_mult = 1.0 + (thresh_short - prob) * 5.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
        else:
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = adjusted_thresh_long
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = adjusted_thresh_short
                else:
                    thresh_long = adjusted_thresh_long
                    thresh_short = adjusted_thresh_short
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                        if prob < 0.495 or (self.trend_filter and not is_uptrend):
                            signals[ticker] = {"action": "SELL", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:
                        if prob > 0.505 or (self.trend_filter and is_uptrend):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    take_profit_pct = float(stop_loss_pct * 2.0)
                    
                    trailing_stop_pct = (2.0 * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.05))
                    
                    if prob >= thresh_long:
                        conf_mult = 1.0 + (prob - thresh_long) * 5.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif prob <= thresh_short:
                        conf_mult = 1.0 + (thresh_short - prob) * 5.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
 
        return signals


class NeuralNetworkV10Strategy(BaseStrategy):
    """
    Strategia quantitativa V10 - Temporal Attention Pooling (CNN-Transformer v10).
    Sfrutta la nuova architettura con Attention Pooling e skip connection residua temporale.
    Eredita la logica di gestione del rischio adattiva sul Breadth della V9 ed è estesa
    con moduli per Kelly Sizing, Trailing Stop e allocazione dinamica degli slot.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model_v10.pth", 
        probability_threshold: float = 0.525,
        ranking_mode: bool = True,
        top_pct: float = 0.03,
        exit_pct: float = 0.60,
        exit_long_threshold: float = 0.485,
        exit_short_threshold: float = 0.515,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None,
        stop_loss_atr_mult: float = 5.5,
        take_profit_mult: float = 2.0,
        use_trailing_only: bool = False,
        trailing_stop_atr_mult: float = 3.0,
        dynamic_slots: bool = True,
        base_max_slots: int = 5,
        short_breadth_thresh: float = 0.40
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v10.model import NeuralNetworkV10
        
        self.probability_threshold = probability_threshold
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_mult = take_profit_mult
        self.use_trailing_only = use_trailing_only
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.dynamic_slots = dynamic_slots
        self.base_max_slots = base_max_slots
        self.short_breadth_thresh = short_breadth_thresh
        
        # Stabilità e centralizzazione (inizializzate ai valori base, regolate a runtime se dynamic_slots è True)
        self.current_max_slots = base_max_slots
        self.current_cash_reserve_pct = 0.0
        
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v10" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v10: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV10Strategy")
        logger.info(f"Caricamento del modello CNN-Transformer v10 con Attention Pooling da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        self.model = NeuralNetworkV10(
            input_dim=self.input_dim, 
            lookback=self.lookback,
            d_model=state.get("d_model", 64),
            nhead=state.get("nhead", 4),
            num_layers=state.get("num_layers", 2),
            alpha=state.get("alpha", 50.0)
        )
        self.model.load(str(model_path))
        
        logger.info(f"Modello CNN-Transformer v10 caricato con successo per la strategia V10.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}
        
        # 1. Calcolo Breadth di Mercato (percentuale di titoli sopra la SMA 200)
        slice_len = self.lookback + 50
        ticker_rets = {}
        ticker_vols = {}
        
        uptrend_counts = 0
        total_valid = 0
        
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
            ticker_rets[ticker] = df['close'].iloc[-slice_len:].pct_change().fillna(0)
            ticker_vols[ticker] = df['volume'].iloc[-slice_len:]
            
            if len(df) >= 200:
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                if pd.notna(sma_200):
                    total_valid += 1
                    if close >= sma_200:
                        uptrend_counts += 1
            
        if not ticker_rets:
            return signals
            
        market_breadth = (uptrend_counts / total_valid) if total_valid > 0 else 0.5
        
        # 1.B Filtro Breadth Reattivo (Dynamic Slots & Cash Reserve Allocation)
        if self.dynamic_slots:
            if market_breadth >= 0.65:
                # Mercato Toro forte e stabile: concentrazione su 4 slot, 0% cash reserve (aggressivo)
                self.current_max_slots = 4
                self.current_cash_reserve_pct = 0.0
            elif market_breadth >= 0.40:
                # Mercato moderato/laterale: 5 slot standard, 15% cash reserve
                self.current_max_slots = 5
                self.current_cash_reserve_pct = 0.15
            else:
                # Mercato debole/Orso: allarghiamo a 8 slot per diversificazione, 40% cash reserve per protezione
                self.current_max_slots = 8
                self.current_cash_reserve_pct = 0.40
        else:
            self.current_max_slots = self.base_max_slots
            self.current_cash_reserve_pct = 0.0
        
        df_rets_all = pd.DataFrame(ticker_rets)
        df_vols_all = pd.DataFrame(ticker_vols)
        
        market_daily_ret = df_rets_all.mean(axis=1)
        market_daily_vol = df_vols_all.mean(axis=1)
        
        # Volatilità rolling del mercato a 20 giorni
        market_rolling_vol = market_daily_ret.rolling(20).std().fillna(0.0)
        
        vol_avg = market_rolling_vol.mean()
        vol_today = market_rolling_vol.iloc[-1]
        vol_ratio = vol_today / (vol_avg + 1e-9)
        vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
        
        # 2. Regolazione Adattiva delle Soglie in base alla salute del mercato (Market Breadth)
        base_threshold = self.probability_threshold
        
        if market_breadth < 0.40:
            # Mercato debole: alza la soglia di ingresso per i LONG a 0.540
            raw_thresh_long = 0.540
            raw_thresh_short = 0.460
        else:
            # Mercato forte: soglia standard a 0.525 per massima partecipazione
            raw_thresh_long = base_threshold
            raw_thresh_short = 1.0 - base_threshold
            
        if self.probability_threshold_long is not None:
            raw_thresh_long = self.probability_threshold_long
        if self.probability_threshold_short is not None:
            raw_thresh_short = self.probability_threshold_short
            
        adjusted_thresh_long = 0.50 + (raw_thresh_long - 0.50) * vol_ratio
        adjusted_thresh_short = 0.50 - (0.50 - raw_thresh_short) * vol_ratio
        
        adjusted_thresh_long = float(np.clip(adjusted_thresh_long, 0.515, 0.56))
        adjusted_thresh_short = float(np.clip(adjusted_thresh_short, 0.44, 0.485))
        
        # Disabilita gli SHORT se la market breadth è superiore alla soglia configurata
        if market_breadth > self.short_breadth_thresh:
            adjusted_thresh_short = 0.0
            
        valid_tickers = []
        seq_features_list = []
        ticker_dfs = {}

        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue

            close_slice = df['close'].iloc[-slice_len:]
            high_slice = df['high'].iloc[-slice_len:]
            low_slice = df['low'].iloc[-slice_len:]
            volume_slice = df['volume'].iloc[-slice_len:]
            
            ret_series = close_slice.pct_change().fillna(0)
            vol_ret_series = volume_slice.pct_change().fillna(0)
            
            obv_series = (np.sign(ret_series) * volume_slice).fillna(0).cumsum()
            obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            
            rsi_series = df['rsi_14'].iloc[-self.lookback:] / 100.0
            atr_series = df['atr_14'].iloc[-self.lookback:] / close_slice.iloc[-self.lookback:]
            
            bb_upper_s = df['bb_upper'].iloc[-self.lookback:]
            bb_lower_s = df['bb_lower'].iloc[-self.lookback:]
            bb_middle_s = df['bb_middle'].iloc[-self.lookback:]
            close_lookback = close_slice.iloc[-self.lookback:]
            
            bb_b_series = (close_lookback - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-9)
            bb_w_series = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-9)
            
            dist_200_series = (close_lookback - df['sma_200'].iloc[-self.lookback:]) / (df['sma_200'].iloc[-self.lookback:] + 1e-9)
            dist_50_series = (close_lookback - df['sma_50'].iloc[-self.lookback:]) / (df['sma_50'].iloc[-self.lookback:] + 1e-9)
            
            sma_5 = close_slice.rolling(5).mean()
            ema_12 = close_slice.ewm(span=12, adjust=False).mean()
            
            roc_10_series = close_slice.pct_change(10).fillna(0)
            
            low_14 = low_slice.rolling(14).min()
            high_14 = high_slice.rolling(14).max()
            stoch_k_series = ((close_slice - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
            
            sma_5_ratio_series = sma_5 / close_slice
            ema_12_ratio_series = ema_12 / close_slice
            
            volume_std_10 = volume_slice.rolling(10).std().fillna(0)
            volume_std_ratio_series = (volume_slice / (volume_std_10 + 1e-9)).fillna(1.0)
            
            market_relative_ret_series = ret_series - market_daily_ret
            market_relative_volume_series = volume_slice / (market_daily_vol + 1e-9)
            
            market_return_series = market_daily_ret
            market_volatility_series = market_rolling_vol
            
            check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
            if df.iloc[-self.lookback:][check_cols].isna().any().any():
                signals[ticker] = {"action": "HOLD", "probability": 0.5}
                continue
                
            ret_arr = ret_series.values[-self.lookback:]
            vol_ret_arr = vol_ret_series.values[-self.lookback:]
            rsi_arr = rsi_series.values[-self.lookback:]
            bb_b_arr = bb_b_series.values[-self.lookback:]
            bb_w_arr = bb_w_series.values[-self.lookback:]
            atr_arr = atr_series.values[-self.lookback:]
            dist_200_arr = dist_200_series.values[-self.lookback:]
            dist_50_arr = dist_50_series.values[-self.lookback:]
            obv_ret_arr = obv_ret_series.values[-self.lookback:]
            roc_10_arr = roc_10_series.values[-self.lookback:]
            stoch_k_arr = stoch_k_series.values[-self.lookback:]
            sma_5_ratio_arr = sma_5_ratio_series.values[-self.lookback:]
            ema_12_ratio_arr = ema_12_ratio_series.values[-self.lookback:]
            volume_std_ratio_arr = volume_std_ratio_series.values[-self.lookback:]
            market_relative_ret_arr = market_relative_ret_series.values[-self.lookback:]
            market_relative_volume_arr = market_relative_volume_series.values[-self.lookback:]
            market_return_arr = market_return_series.values[-self.lookback:]
            market_volatility_arr = market_volatility_series.values[-self.lookback:]
            
            feature_arrays = {
                'ret': ret_arr,
                'vol_ret': vol_ret_arr,
                'RSI_14': rsi_arr,
                'Bollinger_%B': bb_b_arr,
                'Bollinger_Width': bb_w_arr,
                'ATRr_14': atr_arr,
                'Dist_SMA200': dist_200_arr,
                'Dist_SMA50': dist_50_arr,
                'OBV_ret': obv_ret_arr,
                'ROC_10': roc_10_arr,
                'Stoch_K': stoch_k_arr,
                'SMA_5_ratio': sma_5_ratio_arr,
                'EMA_12_ratio': ema_12_ratio_arr,
                'Volume_Std_Ratio': volume_std_ratio_arr,
                'Market_Relative_Ret': market_relative_ret_arr,
                'Market_Relative_Volume': market_relative_volume_arr,
                'Market_Return': market_return_arr,
                'Market_Volatility': market_volatility_arr
            }
            
            seq_feature_vectors = np.column_stack([feature_arrays[col] for col in self.feature_cols])
            seq_features_scaled = (seq_feature_vectors - self.mean) / self.std
            
            valid_tickers.append(ticker)
            seq_features_list.append(seq_features_scaled)
            ticker_dfs[ticker] = df
 
        if not valid_tickers:
            return signals

        batch_x = np.array(seq_features_list, dtype=np.float32)
        probs = self.model.predict(batch_x)
        
        if self.ranking_mode:
            ticker_probs = {ticker: float(prob) for ticker, prob in zip(valid_tickers, probs)}
            sorted_tickers = sorted(ticker_probs.items(), key=lambda x: x[1], reverse=True)
            
            N = len(valid_tickers)
            K = max(1, int(N * self.top_pct))
            top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
            bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
            
            K_out = max(1, int(N * self.exit_pct))
            top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
            bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
            
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = adjusted_thresh_long
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = adjusted_thresh_short
                else:
                    thresh_long = adjusted_thresh_long
                    thresh_short = adjusted_thresh_short
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                         should_exit = (
                             prob < self.exit_long_threshold or 
                             (self.trend_filter and not is_uptrend) or
                             (ticker not in top_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "SELL", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else: # SHORT
                         should_exit = (
                             prob > self.exit_short_threshold or 
                             (self.trend_filter and is_uptrend) or
                             (ticker not in bottom_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (self.stop_loss_atr_mult * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    
                    if self.use_trailing_only:
                        take_profit_pct = None
                    else:
                        take_profit_pct = float(stop_loss_pct * self.take_profit_mult)
                    
                    # Calcolo trailing stop dinamico basato su ATR (moltiplicatore ATR configurabile)
                    trailing_stop_pct = (self.trailing_stop_atr_mult * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.08))
                    
                    if ticker in top_K_tickers and prob >= thresh_long:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        # Varia da 0.5x (vicino alla soglia di attivazione) a 1.5x (confidenza predittiva molto alta)
                        conf_mult = 1.0 + (prob - thresh_long) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif ticker in bottom_K_tickers and prob <= thresh_short:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        conf_mult = 1.0 + (thresh_short - prob) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
        else:
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = adjusted_thresh_long
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = adjusted_thresh_short
                else:
                    thresh_long = adjusted_thresh_long
                    thresh_short = adjusted_thresh_short
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                        if prob < 0.495 or (self.trend_filter and not is_uptrend):
                            signals[ticker] = {"action": "SELL", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:
                        if prob > 0.505 or (self.trend_filter and is_uptrend):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (self.stop_loss_atr_mult * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    
                    if self.use_trailing_only:
                        take_profit_pct = None
                    else:
                        take_profit_pct = float(stop_loss_pct * self.take_profit_mult)
                    
                    # Calcolo trailing stop dinamico basato su ATR (moltiplicatore ATR configurabile)
                    trailing_stop_pct = (self.trailing_stop_atr_mult * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.08))
                    
                    if prob >= thresh_long:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        conf_mult = 1.0 + (prob - thresh_long) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif prob <= thresh_short:
                        # --- Kelly Sizing Dinamico per Confidenza ---
                        conf_mult = 1.0 + (thresh_short - prob) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
 
        return signals


class NeuralNetworkV11Strategy(BaseStrategy):
    """
    Strategia quantitativa V11 - Temporal Attention Pooling + Cross-Feature Attention (v11).
    Integra dati macro/sentiment (VIX, TNX, DXY, SPY, QQQ) per una robustezza estrema ai crash.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model_v11.pth", 
        probability_threshold: float = 0.525,
        ranking_mode: bool = True,
        top_pct: float = 0.03,
        exit_pct: float = 0.60,
        exit_long_threshold: float = 0.485,
        exit_short_threshold: float = 0.515,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None,
        stop_loss_atr_mult: float = 5.5,
        take_profit_mult: float = 2.0,
        use_trailing_only: bool = False,
        trailing_stop_atr_mult: float = 3.0,
        dynamic_slots: bool = True,
        base_max_slots: int = 5,
        short_breadth_thresh: float = 0.40
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.v11.model import NeuralNetworkV11
        from database.db_manager import DBManager
        
        self.probability_threshold = probability_threshold
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_mult = take_profit_mult
        self.use_trailing_only = use_trailing_only
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.dynamic_slots = dynamic_slots
        self.base_max_slots = base_max_slots
        self.short_breadth_thresh = short_breadth_thresh
        
        self.current_max_slots = base_max_slots
        self.current_cash_reserve_pct = 0.0
        
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "v11" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia v11: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkV11Strategy")
        logger.info(f"Caricamento del modello CNN-Transformer v11 con Feature Attention da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        self.model = NeuralNetworkV11(
            input_dim=self.input_dim, 
            lookback=self.lookback,
            d_model=state.get("d_model", 64),
            nhead=state.get("nhead", 4),
            num_layers=state.get("num_layers", 2),
            alpha=state.get("alpha", 50.0),
            penalty_factor=state.get("penalty_factor", 3.0)
        )
        self.model.load(str(model_path))
        
        # Precaricamento dati macro/sentiment dal database SQLite
        db = DBManager()
        self.macro_data = {}
        for m_ticker in ["^VIX", "^TNX", "DX-Y.NYB", "SPY", "QQQ"]:
            try:
                q = f"""
                    SELECT o.timestamp, o.close, i.sma_200 
                    FROM ohlcv o
                    LEFT JOIN indicators i 
                      ON o.ticker = i.ticker AND o.timestamp = i.timestamp
                    WHERE o.ticker = '{m_ticker}'
                """
                m_df = db.execute_query(q)
                if not m_df.empty:
                    m_df['timestamp'] = pd.to_datetime(m_df['timestamp'])
                    m_df = m_df.rename(columns={
                        'close': f'{m_ticker}_close',
                        'sma_200': f'{m_ticker}_sma_200'
                    })
                    m_df = m_df.drop_duplicates(subset=['timestamp'])
                    self.macro_data[m_ticker] = m_df
            except Exception as e:
                logger.warning(f"Impossibile precaricare dati macro per {m_ticker} in backtest: {e}")
                
        logger.info(f"Modello CNN-Transformer v11 e dati macro caricati con successo.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        
        signals: Dict[str, Dict[str, Any]] = {}
        
        # 1. Calcolo Breadth di Mercato
        slice_len = self.lookback + 50
        ticker_rets = {}
        ticker_vols = {}
        
        uptrend_counts = 0
        total_valid = 0
        
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
            ticker_rets[ticker] = df['close'].iloc[-slice_len:].pct_change().fillna(0)
            ticker_vols[ticker] = df['volume'].iloc[-slice_len:]
            
            if len(df) >= 200:
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                if pd.notna(sma_200):
                    total_valid += 1
                    if close >= sma_200:
                        uptrend_counts += 1
            
        if not ticker_rets:
            return signals
            
        market_breadth = (uptrend_counts / total_valid) if total_valid > 0 else 0.5
        
        # Filtro Breadth Reattivo
        if self.dynamic_slots:
            if market_breadth >= 0.65:
                self.current_max_slots = 4
                self.current_cash_reserve_pct = 0.0
            elif market_breadth >= 0.40:
                self.current_max_slots = 5
                self.current_cash_reserve_pct = 0.15
            else:
                self.current_max_slots = 8
                self.current_cash_reserve_pct = 0.40
        else:
            self.current_max_slots = self.base_max_slots
            self.current_cash_reserve_pct = 0.0
        
        df_rets_all = pd.DataFrame(ticker_rets)
        df_vols_all = pd.DataFrame(ticker_vols)
        
        market_daily_ret = df_rets_all.mean(axis=1)
        market_daily_vol = df_vols_all.mean(axis=1)
        market_rolling_vol = market_daily_ret.rolling(20).std().fillna(0.0)
        
        vol_avg = market_rolling_vol.mean()
        vol_today = market_rolling_vol.iloc[-1]
        vol_ratio = vol_today / (vol_avg + 1e-9)
        vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
        
        base_threshold = self.probability_threshold
        
        if market_breadth < 0.40:
            raw_thresh_long = 0.540
            raw_thresh_short = 0.460
        else:
            raw_thresh_long = base_threshold
            raw_thresh_short = 1.0 - base_threshold
            
        if self.probability_threshold_long is not None:
            raw_thresh_long = self.probability_threshold_long
        if self.probability_threshold_short is not None:
            raw_thresh_short = self.probability_threshold_short
            
        adjusted_thresh_long = 0.50 + (raw_thresh_long - 0.50) * vol_ratio
        adjusted_thresh_short = 0.50 - (0.50 - raw_thresh_short) * vol_ratio
        
        adjusted_thresh_long = float(np.clip(adjusted_thresh_long, 0.515, 0.56))
        adjusted_thresh_short = float(np.clip(adjusted_thresh_short, 0.44, 0.485))
        
        if market_breadth > self.short_breadth_thresh:
            adjusted_thresh_short = 0.0
            
        valid_tickers = []
        seq_features_list = []
        ticker_dfs = {}
        
        # Allineamento ed Estrazione Feature per ciascun ticker
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
                
            df_ticker = df.copy()
            if 'timestamp' not in df_ticker.columns:
                df_ticker = df_ticker.reset_index()
                if 'index' in df_ticker.columns:
                    df_ticker = df_ticker.rename(columns={'index': 'timestamp'})
                elif 'Date' in df_ticker.columns:
                    df_ticker = df_ticker.rename(columns={'Date': 'timestamp'})
            df_ticker['timestamp'] = pd.to_datetime(df_ticker['timestamp'])
            
            for m_t, m_df in self.macro_data.items():
                if not m_df.empty:
                    df_ticker = pd.merge(df_ticker, m_df, on='timestamp', how='left')
                    
            for m_t in ["^VIX", "^TNX", "DX-Y.NYB", "SPY", "QQQ"]:
                close_col = f'{m_t}_close'
                if close_col in df_ticker.columns:
                    df_ticker[close_col] = df_ticker[close_col].ffill().bfill().fillna(0.0)
                sma_col = f'{m_t}_sma_200'
                if sma_col in df_ticker.columns:
                    df_ticker[sma_col] = df_ticker[sma_col].ffill().bfill().fillna(0.0)
                    
            df_ticker['VIX_close'] = df_ticker['^VIX_close'] if '^VIX_close' in df_ticker.columns else 15.0
            df_ticker['TNX_close'] = df_ticker['^TNX_close'] if '^TNX_close' in df_ticker.columns else 4.0
            df_ticker['DXY_close'] = df_ticker['DX-Y.NYB_close'] if 'DX-Y.NYB_close' in df_ticker.columns else 100.0
            
            if 'SPY_close' in df_ticker.columns and 'SPY_sma_200' in df_ticker.columns:
                df_ticker['SPY_dist_sma200'] = ((df_ticker['SPY_close'] - df_ticker['SPY_sma_200']) / (df_ticker['SPY_sma_200'] + 1e-9)).fillna(0.0)
            else:
                df_ticker['SPY_dist_sma200'] = 0.0
                
            if 'QQQ_close' in df_ticker.columns and 'QQQ_sma_200' in df_ticker.columns:
                df_ticker['QQQ_dist_sma200'] = ((df_ticker['QQQ_close'] - df_ticker['QQQ_sma_200']) / (df_ticker['QQQ_sma_200'] + 1e-9)).fillna(0.0)
            else:
                df_ticker['QQQ_dist_sma200'] = 0.0
                
            df_ticker['SPY_daily_ret'] = df_ticker['SPY_close'].pct_change().fillna(0.0) if 'SPY_close' in df_ticker.columns else 0.0
            df_ticker['VIX_daily_ret'] = df_ticker['^VIX_close'].pct_change().fillna(0.0) if '^VIX_close' in df_ticker.columns else 0.0
            
            close_slice = df_ticker['close'].iloc[-slice_len:]
            high_slice = df_ticker['high'].iloc[-slice_len:]
            low_slice = df_ticker['low'].iloc[-slice_len:]
            volume_slice = df_ticker['volume'].iloc[-slice_len:]
            
            ret_series = close_slice.pct_change().fillna(0)
            vol_ret_series = volume_slice.pct_change().fillna(0)
            
            obv_series = (np.sign(ret_series) * volume_slice).fillna(0).cumsum()
            obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            
            rsi_series = df_ticker['rsi_14'].iloc[-self.lookback:] / 100.0
            atr_series = df_ticker['atr_14'].iloc[-self.lookback:] / close_slice.iloc[-self.lookback:]
            
            bb_upper_s = df_ticker['bb_upper'].iloc[-self.lookback:]
            bb_lower_s = df_ticker['bb_lower'].iloc[-self.lookback:]
            bb_middle_s = df_ticker['bb_middle'].iloc[-self.lookback:]
            close_lookback = close_slice.iloc[-self.lookback:]
            
            bb_b_series = (close_lookback - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-9)
            bb_w_series = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-9)
            
            dist_200_series = (close_lookback - df_ticker['sma_200'].iloc[-self.lookback:]) / (df_ticker['sma_200'].iloc[-self.lookback:] + 1e-9)
            dist_50_series = (close_lookback - df_ticker['sma_50'].iloc[-self.lookback:]) / (df_ticker['sma_50'].iloc[-self.lookback:] + 1e-9)
            
            sma_5 = close_slice.rolling(5).mean()
            ema_12 = close_slice.ewm(span=12, adjust=False).mean()
            
            roc_10_series = close_slice.pct_change(10).fillna(0)
            
            low_14 = low_slice.rolling(14).min()
            high_14 = high_slice.rolling(14).max()
            stoch_k_series = ((close_slice - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
            
            sma_5_ratio_series = sma_5 / close_slice
            ema_12_ratio_series = ema_12 / close_slice
            
            volume_std_10 = volume_slice.rolling(10).std().fillna(0)
            volume_std_ratio_series = (volume_slice / (volume_std_10 + 1e-9)).fillna(1.0)
            
            market_relative_ret_series = ret_series - market_daily_ret
            market_relative_volume_series = volume_slice / (market_daily_vol + 1e-9)
            
            market_return_series = market_daily_ret
            market_volatility_series = market_rolling_vol
            
            check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
            if df_ticker.iloc[-self.lookback:][check_cols].isna().any().any():
                signals[ticker] = {"action": "HOLD", "probability": 0.5}
                continue
                
            ret_arr = ret_series.values[-self.lookback:]
            vol_ret_arr = vol_ret_series.values[-self.lookback:]
            rsi_arr = rsi_series.values[-self.lookback:]
            bb_b_arr = bb_b_series.values[-self.lookback:]
            bb_w_arr = bb_w_series.values[-self.lookback:]
            atr_arr = atr_series.values[-self.lookback:]
            dist_200_arr = dist_200_series.values[-self.lookback:]
            dist_50_arr = dist_50_series.values[-self.lookback:]
            obv_ret_arr = obv_ret_series.values[-self.lookback:]
            roc_10_arr = roc_10_series.values[-self.lookback:]
            stoch_k_arr = stoch_k_series.values[-self.lookback:]
            sma_5_ratio_arr = sma_5_ratio_series.values[-self.lookback:]
            ema_12_ratio_arr = ema_12_ratio_series.values[-self.lookback:]
            volume_std_ratio_arr = volume_std_ratio_series.values[-self.lookback:]
            market_relative_ret_arr = market_relative_ret_series.values[-self.lookback:]
            market_relative_volume_arr = market_relative_volume_series.values[-self.lookback:]
            market_return_arr = market_return_series.values[-self.lookback:]
            market_volatility_arr = market_volatility_series.values[-self.lookback:]
            
            vix_close_arr = df_ticker['VIX_close'].iloc[-self.lookback:].values
            tnx_close_arr = df_ticker['TNX_close'].iloc[-self.lookback:].values
            dxy_close_arr = df_ticker['DXY_close'].iloc[-self.lookback:].values
            spy_dist_arr = df_ticker['SPY_dist_sma200'].iloc[-self.lookback:].values
            qqq_dist_arr = df_ticker['QQQ_dist_sma200'].iloc[-self.lookback:].values
            spy_ret_arr = df_ticker['SPY_daily_ret'].iloc[-self.lookback:].values
            vix_ret_arr = df_ticker['VIX_daily_ret'].iloc[-self.lookback:].values
            
            feature_arrays = {
                'ret': ret_arr,
                'vol_ret': vol_ret_arr,
                'RSI_14': rsi_arr,
                'Bollinger_%B': bb_b_arr,
                'Bollinger_Width': bb_w_arr,
                'ATRr_14': atr_arr,
                'Dist_SMA200': dist_200_arr,
                'Dist_SMA50': dist_50_arr,
                'OBV_ret': obv_ret_arr,
                'ROC_10': roc_10_arr,
                'Stoch_K': stoch_k_arr,
                'SMA_5_ratio': sma_5_ratio_arr,
                'EMA_12_ratio': ema_12_ratio_arr,
                'Volume_Std_Ratio': volume_std_ratio_arr,
                'Market_Relative_Ret': market_relative_ret_arr,
                'Market_Relative_Volume': market_relative_volume_arr,
                'Market_Return': market_return_arr,
                'Market_Volatility': market_volatility_arr,
                'VIX_close': vix_close_arr,
                'TNX_close': tnx_close_arr,
                'DXY_close': dxy_close_arr,
                'SPY_dist_sma200': spy_dist_arr,
                'QQQ_dist_sma200': qqq_dist_arr,
                'SPY_daily_ret': spy_ret_arr,
                'VIX_daily_ret': vix_ret_arr
            }
            
            seq_feature_vectors = np.column_stack([feature_arrays[col] for col in self.feature_cols])
            seq_features_scaled = (seq_feature_vectors - self.mean) / self.std
            
            valid_tickers.append(ticker)
            seq_features_list.append(seq_features_scaled)
            ticker_dfs[ticker] = df_ticker
            
        if not valid_tickers:
            return signals
            
        batch_x = np.array(seq_features_list, dtype=np.float32)
        probs = self.model.predict(batch_x)
        
        if self.ranking_mode:
            ticker_probs = {ticker: float(prob) for ticker, prob in zip(valid_tickers, probs)}
            sorted_tickers = sorted(ticker_probs.items(), key=lambda x: x[1], reverse=True)
            
            N = len(valid_tickers)
            K = max(1, int(N * self.top_pct))
            top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
            bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
            
            K_out = max(1, int(N * self.exit_pct))
            top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
            bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
            
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = adjusted_thresh_long
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = adjusted_thresh_short
                else:
                    thresh_long = adjusted_thresh_long
                    thresh_short = adjusted_thresh_short
                    
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                         should_exit = (
                             prob < self.exit_long_threshold or 
                             (self.trend_filter and not is_uptrend) or
                             (ticker not in top_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "SELL", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else: # SHORT
                         should_exit = (
                             prob > self.exit_short_threshold or 
                             (self.trend_filter and is_uptrend) or
                             (ticker not in bottom_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (self.stop_loss_atr_mult * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    
                    if self.use_trailing_only:
                        take_profit_pct = None
                    else:
                        take_profit_pct = float(stop_loss_pct * self.take_profit_mult)
                        
                    trailing_stop_pct = (self.trailing_stop_atr_mult * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.08))
                    
                    if ticker in top_K_tickers and prob >= thresh_long:
                        conf_mult = 1.0 + (prob - thresh_long) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif ticker in bottom_K_tickers and prob <= thresh_short:
                        conf_mult = 1.0 + (thresh_short - prob) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
        else:
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = adjusted_thresh_long
                        thresh_short = 0.30
                    else:
                        thresh_long = 0.70
                        thresh_short = adjusted_thresh_short
                else:
                    thresh_long = adjusted_thresh_long
                    thresh_short = adjusted_thresh_short
                    
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                        if prob < 0.495 or (self.trend_filter and not is_uptrend):
                            signals[ticker] = {"action": "SELL", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:
                        if prob > 0.505 or (self.trend_filter and is_uptrend):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (self.stop_loss_atr_mult * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    
                    if self.use_trailing_only:
                        take_profit_pct = None
                    else:
                        take_profit_pct = float(stop_loss_pct * self.take_profit_mult)
                        
                    trailing_stop_pct = (self.trailing_stop_atr_mult * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.015, 0.08))
                    
                    if prob >= thresh_long:
                        conf_mult = 1.0 + (prob - thresh_long) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif prob <= thresh_short:
                        conf_mult = 1.0 + (thresh_short - prob) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                        
        return signals


class MoEStrategyV1(BaseStrategy):
    """
    Strategia quantitativa MoE V1 - Regime-Gated Mixture of Experts.
    Utilizza 3 esperti coordinati da un gating network macroeconomico.
    """
    def __init__(
        self, 
        model_filename: str = "neural_model_moe.pth", 
        probability_threshold: float = 0.0,  # Impostato a 0.0 per bypassare il filtro fisso dell'engine di backtest
        ranking_mode: bool = True,
        top_pct: float = 0.03,
        exit_pct: float = 0.60,
        exit_long_threshold: float = 0.485,
        exit_short_threshold: float = 0.515,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None,
        stop_loss_atr_mult: float = 5.5,
        take_profit_mult: float = 2.0,
        use_trailing_only: bool = False,
        trailing_stop_atr_mult: float = 3.0,
        dynamic_slots: bool = True,
        base_max_slots: int = 5,
        short_breadth_thresh: float = 0.40
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.rete_neurale.moe_v1.model import MoEModelV1
        from database.db_manager import DBManager
        
        self.probability_threshold = 0.0  # Forza a 0.0 per bypassare il filtro fisso dell'engine di backtest
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.exit_long_threshold = exit_long_threshold
        self.exit_short_threshold = exit_short_threshold
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_mult = take_profit_mult
        self.use_trailing_only = use_trailing_only
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.dynamic_slots = dynamic_slots
        self.base_max_slots = base_max_slots
        self.short_breadth_thresh = short_breadth_thresh
        
        self.current_max_slots = base_max_slots
        self.current_cash_reserve_pct = 0.0
        
        model_path = Path(__file__).resolve().parent.parent / "models" / "rete_neurale" / "moe_v1" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia MoE: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("MoEStrategyV1")
        logger.info(f"Caricamento del modello MoE moe_v1 da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        
        self.model = MoEModelV1(
            input_dim=self.input_dim, 
            lookback=self.lookback,
            d_model=state.get("d_model", 64),
            nhead=state.get("nhead", 4),
            num_layers=state.get("num_layers", 2),
            alpha=state.get("alpha", 50.0),
            penalty_factor=state.get("penalty_factor", 1.5),
            lambda_gating=state.get("lambda_gating", 0.5)
        )
        self.model.load(str(model_path))
        
        # Precaricamento dati macro/sentiment dal database SQLite
        db = DBManager()
        self.macro_data = {}
        for m_ticker in ["^VIX", "^TNX", "DX-Y.NYB", "SPY", "QQQ"]:
            try:
                q = f"""
                    SELECT o.timestamp, o.close, i.sma_200 
                    FROM ohlcv o
                    LEFT JOIN indicators i 
                      ON o.ticker = i.ticker AND o.timestamp = i.timestamp
                    WHERE o.ticker = '{m_ticker}'
                """
                m_df = db.execute_query(q)
                if not m_df.empty:
                    m_df['timestamp'] = pd.to_datetime(m_df['timestamp'])
                    m_df = m_df.rename(columns={
                        'close': f'{m_ticker}_close',
                        'sma_200': f'{m_ticker}_sma_200'
                    })
                    m_df = m_df.drop_duplicates(subset=['timestamp'])
                    self.macro_data[m_ticker] = m_df
            except Exception as e:
                logger.warning(f"Impossibile precaricare dati macro per {m_ticker} in backtest: {e}")
                
        logger.info(f"Modello MoE moe_v1 e dati macro caricati con successo.")

    def generate_signals(
        self,
        historical_data: dict,
        portfolio: any,
        current_date: any
    ) -> dict:
        import numpy as np
        import pandas as pd
        import logging
        
        signals = {}
        
        # 1. Calcolo Breadth di Mercato
        slice_len = self.lookback + 50
        ticker_rets = {}
        ticker_vols = {}
        
        uptrend_counts = 0
        total_valid = 0
        
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
            ticker_rets[ticker] = df['close'].iloc[-slice_len:].pct_change().fillna(0)
            ticker_vols[ticker] = df['volume'].iloc[-slice_len:]
            
            if len(df) >= 200:
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                if pd.notna(sma_200):
                    total_valid += 1
                    if close >= sma_200:
                        uptrend_counts += 1
            
        if not ticker_rets:
            return signals
            
        market_breadth = (uptrend_counts / total_valid) if total_valid > 0 else 0.5
        
        # Filtro Breadth Reattivo
        if self.dynamic_slots:
            if market_breadth >= 0.65:
                self.current_max_slots = 4
                self.current_cash_reserve_pct = 0.0
            elif market_breadth >= 0.40:
                self.current_max_slots = 5
                self.current_cash_reserve_pct = 0.15
            else:
                self.current_max_slots = 8
                self.current_cash_reserve_pct = 0.40
        else:
            self.current_max_slots = self.base_max_slots
            self.current_cash_reserve_pct = 0.0
        
        df_rets_all = pd.DataFrame(ticker_rets)
        df_vols_all = pd.DataFrame(ticker_vols)
        
        market_daily_ret = df_rets_all.mean(axis=1)
        market_daily_vol = df_vols_all.mean(axis=1)
        market_rolling_vol = market_daily_ret.rolling(20).std().fillna(0.0)
        
        # Le soglie verranno calcolate dinamicamente tramite percentili (Soluzione A)
        # dopo aver ottenuto le predizioni del modello per tutti i ticker.
            
        valid_tickers = []
        seq_features_list = []
        ticker_dfs = {}
        
        for ticker, df in historical_data.items():
            if len(df) < self.lookback + 220:
                continue
                
            df_ticker = df.copy()
            if 'timestamp' not in df_ticker.columns:
                df_ticker = df_ticker.reset_index()
                if 'index' in df_ticker.columns:
                    df_ticker = df_ticker.rename(columns={'index': 'timestamp'})
                elif 'Date' in df_ticker.columns:
                    df_ticker = df_ticker.rename(columns={'Date': 'timestamp'})
            df_ticker['timestamp'] = pd.to_datetime(df_ticker['timestamp'])
            
            for m_t, m_df in self.macro_data.items():
                if not m_df.empty:
                    df_ticker = pd.merge(df_ticker, m_df, on='timestamp', how='left')
                    
            for m_t in ["^VIX", "^TNX", "DX-Y.NYB", "SPY", "QQQ"]:
                close_col = f'{m_t}_close'
                if close_col in df_ticker.columns:
                    df_ticker[close_col] = df_ticker[close_col].ffill().bfill().fillna(0.0)
                sma_col = f'{m_t}_sma_200'
                if sma_col in df_ticker.columns:
                    df_ticker[sma_col] = df_ticker[sma_col].ffill().bfill().fillna(0.0)
                    
            df_ticker['VIX_close'] = df_ticker['^VIX_close'] if '^VIX_close' in df_ticker.columns else 15.0
            df_ticker['TNX_close'] = df_ticker['^TNX_close'] if '^TNX_close' in df_ticker.columns else 4.0
            df_ticker['DXY_close'] = df_ticker['DX-Y.NYB_close'] if 'DX-Y.NYB_close' in df_ticker.columns else 100.0
            
            if 'SPY_close' in df_ticker.columns and 'SPY_sma_200' in df_ticker.columns:
                df_ticker['SPY_dist_sma200'] = ((df_ticker['SPY_close'] - df_ticker['SPY_sma_200']) / (df_ticker['SPY_sma_200'] + 1e-9)).fillna(0.0)
            else:
                df_ticker['SPY_dist_sma200'] = 0.0
                
            if 'QQQ_close' in df_ticker.columns and 'QQQ_sma_200' in df_ticker.columns:
                df_ticker['QQQ_dist_sma200'] = ((df_ticker['QQQ_close'] - df_ticker['QQQ_sma_200']) / (df_ticker['QQQ_sma_200'] + 1e-9)).fillna(0.0)
            else:
                df_ticker['QQQ_dist_sma200'] = 0.0
                
            df_ticker['SPY_daily_ret'] = df_ticker['SPY_close'].pct_change().fillna(0.0) if 'SPY_close' in df_ticker.columns else 0.0
            df_ticker['VIX_daily_ret'] = df_ticker['^VIX_close'].pct_change().fillna(0.0) if '^VIX_close' in df_ticker.columns else 0.0
            
            close_slice = df_ticker['close'].iloc[-slice_len:]
            high_slice = df_ticker['high'].iloc[-slice_len:]
            low_slice = df_ticker['low'].iloc[-slice_len:]
            volume_slice = df_ticker['volume'].iloc[-slice_len:]
            
            ret_series = close_slice.pct_change().fillna(0)
            vol_ret_series = volume_slice.pct_change().fillna(0)
            
            obv_series = (np.sign(ret_series) * volume_slice).fillna(0).cumsum()
            obv_ret_series = obv_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            
            rsi_series = df_ticker['rsi_14'].iloc[-self.lookback:] / 100.0
            atr_series = df_ticker['atr_14'].iloc[-self.lookback:] / close_slice.iloc[-self.lookback:]
            
            bb_upper_s = df_ticker['bb_upper'].iloc[-self.lookback:]
            bb_lower_s = df_ticker['bb_lower'].iloc[-self.lookback:]
            bb_middle_s = df_ticker['bb_middle'].iloc[-self.lookback:]
            close_lookback = close_slice.iloc[-self.lookback:]
            
            bb_b_series = (close_lookback - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-9)
            bb_w_series = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-9)
            
            dist_200_series = (close_lookback - df_ticker['sma_200'].iloc[-self.lookback:]) / (df_ticker['sma_200'].iloc[-self.lookback:] + 1e-9)
            dist_50_series = (close_lookback - df_ticker['sma_50'].iloc[-self.lookback:]) / (df_ticker['sma_50'].iloc[-self.lookback:] + 1e-9)
            
            sma_5 = close_slice.rolling(5).mean()
            ema_12 = close_slice.ewm(span=12, adjust=False).mean()
            
            roc_10_series = close_slice.pct_change(10).fillna(0)
            
            low_14 = low_slice.rolling(14).min()
            high_14 = high_slice.rolling(14).max()
            stoch_k_series = ((close_slice - low_14) / (high_14 - low_14 + 1e-9)).fillna(0.5)
            
            sma_5_ratio_series = sma_5 / close_slice
            ema_12_ratio_series = ema_12 / close_slice
            
            volume_std_10 = volume_slice.rolling(10).std().fillna(0)
            volume_std_ratio_series = (volume_slice / (volume_std_10 + 1e-9)).fillna(1.0)
            
            market_relative_ret_series = ret_series - market_daily_ret
            market_relative_volume_series = volume_slice / (market_daily_vol + 1e-9)
            
            market_return_series = market_daily_ret
            market_volatility_series = market_rolling_vol
            
            check_cols = ['close', 'volume', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'bb_middle', 'atr_14']
            if df_ticker.iloc[-self.lookback:][check_cols].isna().any().any():
                signals[ticker] = {"action": "HOLD", "probability": 0.5}
                continue
                
            ret_arr = ret_series.values[-self.lookback:]
            vol_ret_arr = vol_ret_series.values[-self.lookback:]
            rsi_arr = rsi_series.values[-self.lookback:]
            bb_b_arr = bb_b_series.values[-self.lookback:]
            bb_w_arr = bb_w_series.values[-self.lookback:]
            atr_arr = atr_series.values[-self.lookback:]
            dist_200_arr = dist_200_series.values[-self.lookback:]
            dist_50_arr = dist_50_series.values[-self.lookback:]
            obv_ret_arr = obv_ret_series.values[-self.lookback:]
            roc_10_arr = roc_10_series.values[-self.lookback:]
            stoch_k_arr = stoch_k_series.values[-self.lookback:]
            sma_5_ratio_arr = sma_5_ratio_series.values[-self.lookback:]
            ema_12_ratio_arr = ema_12_ratio_series.values[-self.lookback:]
            volume_std_ratio_arr = volume_std_ratio_series.values[-self.lookback:]
            market_relative_ret_arr = market_relative_ret_series.values[-self.lookback:]
            market_relative_volume_arr = market_relative_volume_series.values[-self.lookback:]
            market_return_arr = market_return_series.values[-self.lookback:]
            market_volatility_arr = market_volatility_series.values[-self.lookback:]
            
            vix_close_arr = df_ticker['VIX_close'].iloc[-self.lookback:].values
            tnx_close_arr = df_ticker['TNX_close'].iloc[-self.lookback:].values
            dxy_close_arr = df_ticker['DXY_close'].iloc[-self.lookback:].values
            spy_dist_arr = df_ticker['SPY_dist_sma200'].iloc[-self.lookback:].values
            qqq_dist_arr = df_ticker['QQQ_dist_sma200'].iloc[-self.lookback:].values
            spy_ret_arr = df_ticker['SPY_daily_ret'].iloc[-self.lookback:].values
            vix_ret_arr = df_ticker['VIX_daily_ret'].iloc[-self.lookback:].values
            
            feature_arrays = {
                'ret': ret_arr,
                'vol_ret': vol_ret_arr,
                'RSI_14': rsi_arr,
                'Bollinger_%B': bb_b_arr,
                'Bollinger_Width': bb_w_arr,
                'ATRr_14': atr_arr,
                'Dist_SMA200': dist_200_arr,
                'Dist_SMA50': dist_50_arr,
                'OBV_ret': obv_ret_arr,
                'ROC_10': roc_10_arr,
                'Stoch_K': stoch_k_arr,
                'SMA_5_ratio': sma_5_ratio_arr,
                'EMA_12_ratio': ema_12_ratio_arr,
                'Volume_Std_Ratio': volume_std_ratio_arr,
                'Market_Relative_Ret': market_relative_ret_arr,
                'Market_Relative_Volume': market_relative_volume_arr,
                'Market_Return': market_return_arr,
                'Market_Volatility': market_volatility_arr,
                'VIX_close': vix_close_arr,
                'TNX_close': tnx_close_arr,
                'DXY_close': dxy_close_arr,
                'SPY_dist_sma200': spy_dist_arr,
                'QQQ_dist_sma200': qqq_dist_arr,
                'SPY_daily_ret': spy_ret_arr,
                'VIX_daily_ret': vix_ret_arr
            }
            
            seq_feature_vectors = np.column_stack([feature_arrays[col] for col in self.feature_cols])
            seq_features_scaled = (seq_feature_vectors - self.mean) / self.std
            
            valid_tickers.append(ticker)
            seq_features_list.append(seq_features_scaled)
            ticker_dfs[ticker] = df_ticker
            
        if not valid_tickers:
            return signals
            
        batch_x = np.array(seq_features_list, dtype=np.float32)
        probs = self.model.predict(batch_x)
        
        # Soluzione A: Dynamic Percentile-Based Thresholds
        # Calcoliamo i percentili sulla distribuzione odierna delle probabilità predette
        # per contrastare il problema della compressione delle probabilità.
        pct_long_val = float(np.percentile(probs, 90))
        pct_short_val = float(np.percentile(probs, 10))
        pct_exit_long_val = float(np.percentile(probs, 40))
        pct_exit_short_val = float(np.percentile(probs, 60))
        
        # Se la breadth di mercato è sopra la soglia, disattiviamo lo shorting
        if market_breadth > self.short_breadth_thresh:
            pct_short_val = 0.0
            
        logger = logging.getLogger("MoEStrategyV1")
        # Logghiamo l'1% delle volte per monitorare le soglie dinamiche senza inondare i log
        if np.random.rand() < 0.01:
            logger.info(
                f"Soglie dinamiche calcolate per {current_date.strftime('%Y-%m-%d')} - "
                f"Long: {pct_long_val:.4f}, Short: {pct_short_val:.4f}, "
                f"Exit Long: {pct_exit_long_val:.4f}, Exit Short: {pct_exit_short_val:.4f}"
            )
        
        if self.ranking_mode:
            ticker_probs = {ticker: float(prob) for ticker, prob in zip(valid_tickers, probs)}
            sorted_tickers = sorted(ticker_probs.items(), key=lambda x: x[1], reverse=True)
            
            N = len(valid_tickers)
            K = max(1, int(N * self.top_pct))
            top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
            bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
            
            K_out = max(1, int(N * self.exit_pct))
            top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
            bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
            
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = pct_long_val
                        thresh_short = 0.0
                    else:
                        thresh_long = 1.0
                        thresh_short = pct_short_val
                else:
                    thresh_long = pct_long_val
                    thresh_short = pct_short_val
                    
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                         should_exit = (
                             prob < pct_exit_long_val or 
                             (self.trend_filter and not is_uptrend) or
                             (ticker not in top_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "SELL", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else: # SHORT
                         should_exit = (
                             prob > pct_exit_short_val or 
                             (self.trend_filter and is_uptrend) or
                             (ticker not in bottom_out_tickers and days_held >= 3)
                         )
                         if should_exit:
                             signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                         else:
                             signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (self.stop_loss_atr_mult * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.03, 0.08))
                    
                    if self.use_trailing_only:
                        take_profit_pct = None
                    else:
                        take_profit_pct = float(stop_loss_pct * self.take_profit_mult)
                        
                    trailing_stop_pct = (self.trailing_stop_atr_mult * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.03, 0.08))
                    
                    if ticker in top_K_tickers and prob >= thresh_long:
                        conf_mult = 1.0 + (prob - thresh_long) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif ticker in bottom_K_tickers and prob <= thresh_short:
                        conf_mult = 1.0 + (thresh_short - prob) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
        else:
            for ticker, prob in zip(valid_tickers, probs):
                df = ticker_dfs[ticker]
                
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = pct_long_val
                        thresh_short = 0.0
                    else:
                        thresh_long = 1.0
                        thresh_short = pct_short_val
                else:
                    thresh_long = pct_long_val
                    thresh_short = pct_short_val
                    
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                        if prob < pct_exit_long_val or (self.trend_filter and not is_uptrend):
                            signals[ticker] = {"action": "SELL", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                    else:
                        if prob > pct_exit_short_val or (self.trend_filter and is_uptrend):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": float(prob)}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                else:
                    atr = latest_row['atr_14']
                    
                    stop_loss_pct = (self.stop_loss_atr_mult * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.03, 0.08))
                    
                    if self.use_trailing_only:
                        take_profit_pct = None
                    else:
                        take_profit_pct = float(stop_loss_pct * self.take_profit_mult)
                        
                    trailing_stop_pct = (self.trailing_stop_atr_mult * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.03, 0.08))
                    
                    if prob >= thresh_long:
                        conf_mult = 1.0 + (prob - thresh_long) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    elif prob <= thresh_short:
                        conf_mult = 1.0 + (thresh_short - prob) * 8.0
                        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))
                        
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": float(prob),
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct,
                            "confidence_multiplier": conf_mult
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": float(prob)}
                        
        return signals


class NeuralNetworkGNNStrategy(BaseStrategy):
    """
    Strategia quantitativa basata sul modello Spatio-Temporal GNN v1.
    Carica i pesi GNN, calcola la griglia delle feature per tutti i ticker e
    genera segnali di trading stabili basati sulla matrice di adiacenza del grafo,
    sulla forza relativa (ranking) e su filtri di trend macro.
    """
    def __init__(
        self, 
        model_filename: str = "gnn_model.pth", 
        probability_threshold: float = 0.55,
        ranking_mode: bool = True,
        top_pct: float = 0.05,
        exit_pct: float = 0.60,
        trend_filter: bool = True,
        probability_threshold_long: Optional[float] = None,
        probability_threshold_short: Optional[float] = None,
        stop_loss_atr_mult: float = 4.0,
        take_profit_mult: float = 2.0,
        use_trailing_only: bool = False,
        trailing_stop_atr_mult: float = 3.0
    ) -> None:
        import sys
        import torch
        from pathlib import Path
        
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from models.gnn.v1.model import SpatioTemporalGNNV1
        from database.db_manager import DBManager
        
        self.probability_threshold = probability_threshold
        self.ranking_mode = ranking_mode
        self.top_pct = top_pct
        self.exit_pct = exit_pct
        self.trend_filter = trend_filter
        self.probability_threshold_long = probability_threshold_long
        self.probability_threshold_short = probability_threshold_short
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_mult = take_profit_mult
        self.use_trailing_only = use_trailing_only
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        
        model_path = Path(__file__).resolve().parent.parent / "models" / "gnn" / "v1" / "pesi" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Impossibile avviare la strategia GNN: file dei pesi non trovato in: {model_path}."
            )
            
        logger = logging.getLogger("NeuralNetworkGNNStrategy")
        logger.info(f"Caricamento del modello GNN da: {model_path}...")
        
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.feature_cols = state["feature_cols"]
        self.mean = np.array(state["scaling_mean"])
        self.std = np.array(state["scaling_std"])
        self.input_dim = state["input_dim"]
        self.lookback = state.get("lookback", 30)
        self.adj_matrix = torch.tensor(state["adj"], dtype=torch.float32)
        
        # Risoluzione della lista ticker
        db = DBManager()
        all_db_tickers = sorted(db.execute_query("SELECT DISTINCT ticker FROM ohlcv")['ticker'].tolist())
        num_nodes = len(state["adj"])
        if "tickers" in state:
            self.tickers_list = state["tickers"]
        else:
            self.tickers_list = all_db_tickers[:num_nodes]
            logger.warning(
                f"Modello sprovvisto di lista ticker salvata. Ripiego sui primi {num_nodes} ticker in ordine alfabetico a DB."
            )
            
        # Carica il modello GNN
        self.model = SpatioTemporalGNNV1(input_dim=self.input_dim, hidden_dim=state.get("hidden_dim", 32))
        self.model.load(str(model_path))
        logger.info("Modello GNN e parametri di scaling caricati correttamente.")

    def generate_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        portfolio: Any,
        current_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        import torch
        
        signals: Dict[str, Dict[str, Any]] = {}
        
        N = len(self.tickers_list)
        L = self.lookback
        F = len(self.feature_cols)
        
        grid_data = np.zeros((N, L, F), dtype=np.float32)
        valid_tickers_mask = np.zeros(N, dtype=bool)
        
        for idx, ticker in enumerate(self.tickers_list):
            if ticker not in historical_data:
                continue
            df = historical_data[ticker]
            if len(df) < L + 200:
                continue
                
            close_series = df['close']
            
            # Calcolo delle feature scala-invarianti standard sullo storico del ticker
            sma_10_r = df['sma_10'] / close_series
            sma_20_r = df['sma_20'] / close_series
            sma_50_r = df['sma_50'] / close_series
            sma_200_r = df['sma_200'] / close_series
            ema_9_r = df['ema_9'] / close_series
            ema_21_r = df['ema_21'] / close_series
            bb_u_r = df['bb_upper'] / close_series
            bb_l_r = df['bb_lower'] / close_series
            macd_r = df['macd'] / close_series
            macd_s_r = df['macd_signal'] / close_series
            macd_h_r = df['macd_hist'] / close_series
            atr_r = df['atr_14'] / close_series
            volume_r = df['volume'] / df['volume'].rolling(10).mean()
            rsi_norm = df['rsi_14'] / 100.0
            
            feat_dict = {
                'sma_10_ratio': sma_10_r,
                'sma_20_ratio': sma_20_r,
                'sma_50_ratio': sma_50_r,
                'sma_200_ratio': sma_200_r,
                'ema_9_ratio': ema_9_r,
                'ema_21_ratio': ema_21_r,
                'bb_upper_ratio': bb_u_r,
                'bb_lower_ratio': bb_l_r,
                'macd_ratio': macd_r,
                'macd_signal_ratio': macd_s_r,
                'macd_hist_ratio': macd_h_r,
                'atr_14_ratio': atr_r,
                'volume_ratio': volume_r,
                'rsi_14_norm': rsi_norm
            }
            
            # Estraiamo gli ultimi L elementi e applichiamo lo scaling
            ticker_features = []
            has_nan = False
            for lookback_idx in range(-L, 0):
                check_cols = ['close', 'volume', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 'rsi_14', 'bb_upper', 'bb_lower', 'atr_14']
                if df.iloc[lookback_idx][check_cols].isna().any():
                    has_nan = True
                    break
                    
                vec = []
                for col in self.feature_cols:
                    val = feat_dict[col].iloc[lookback_idx]
                    vec.append(val)
                
                vec_scaled = (np.array(vec) - self.mean) / self.std
                ticker_features.append(vec_scaled)
                
            if not has_nan and len(ticker_features) == L:
                grid_data[idx] = np.array(ticker_features, dtype=np.float32)
                valid_tickers_mask[idx] = True
                
        # Creiamo il tensore (1, N, L, F)
        X_tensor = torch.as_tensor(grid_data, dtype=torch.float32).unsqueeze(0)
        
        # Predict con il modello GNN
        self.model.model.eval()
        with torch.no_grad():
            probs = self.model.predict(X_tensor, adj=self.adj_matrix)[0] # (N,)
            
        # Raccogliamo le probabilità per i soli ticker attivi e validi simulati
        valid_active_tickers = []
        valid_active_probs = []
        ticker_dfs = {}
        
        for idx, ticker in enumerate(self.tickers_list):
            if not valid_tickers_mask[idx]:
                continue
            valid_active_tickers.append(ticker)
            valid_active_probs.append(float(probs[idx]))
            ticker_dfs[ticker] = historical_data[ticker]
            
        if not valid_active_tickers:
            return signals
            
        # Calcolo delle soglie dinamiche basate sui percentili
        pct_long_val = float(np.percentile(valid_active_probs, 85)) if len(valid_active_probs) > 1 else self.probability_threshold
        pct_short_val = float(np.percentile(valid_active_probs, 15)) if len(valid_active_probs) > 1 else 0.0
        
        if self.probability_threshold_long is not None:
            pct_long_val = self.probability_threshold_long
        if self.probability_threshold_short is not None:
            pct_short_val = self.probability_threshold_short
            
        pct_exit_long_val = float(np.percentile(valid_active_probs, 35)) if len(valid_active_probs) > 1 else (1.0 - self.probability_threshold + 0.10)
        pct_exit_short_val = float(np.percentile(valid_active_probs, 65)) if len(valid_active_probs) > 1 else (self.probability_threshold - 0.10)
        
        if self.ranking_mode:
            ticker_probs_dict = {t: p for t, p in zip(valid_active_tickers, valid_active_probs)}
            sorted_tickers = sorted(ticker_probs_dict.items(), key=lambda x: x[1], reverse=True)
            
            N_active = len(valid_active_tickers)
            K = max(1, int(N_active * self.top_pct))
            top_K_tickers = set([t[0] for t in sorted_tickers[:K]])
            bottom_K_tickers = set([t[0] for t in sorted_tickers[-K:]])
            
            K_out = max(1, int(N_active * self.exit_pct))
            top_out_tickers = set([t[0] for t in sorted_tickers[:K_out]])
            bottom_out_tickers = set([t[0] for t in sorted_tickers[-K_out:]])
            
            for ticker, prob in zip(valid_active_tickers, valid_active_probs):
                df = ticker_dfs[ticker]
                latest_row = df.iloc[-1]
                close = latest_row['close']
                sma_200 = latest_row.get('sma_200', np.nan)
                
                is_uptrend = (close >= sma_200) if pd.notna(sma_200) else True
                
                if self.trend_filter:
                    if is_uptrend:
                        thresh_long = pct_long_val
                        thresh_short = 0.0
                    else:
                        thresh_long = 1.0 # Impossibile da superare
                        thresh_short = pct_short_val
                else:
                    thresh_long = pct_long_val
                    thresh_short = pct_short_val
                    
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    days_held = (current_date - pos.entry_date).days
                    
                    if pos.position_type == "LONG":
                        should_exit = (
                            prob < pct_exit_long_val or 
                            (self.trend_filter and not is_uptrend) or
                            (ticker not in top_out_tickers and days_held >= 3)
                        )
                        if should_exit:
                            signals[ticker] = {"action": "SELL", "probability": prob}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": prob}
                    else: # SHORT
                        should_exit = (
                            prob > pct_exit_short_val or 
                            (self.trend_filter and is_uptrend) or
                            (ticker not in bottom_out_tickers and days_held >= 3)
                        )
                        if should_exit:
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": prob}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": prob}
                else:
                    atr = latest_row['atr_14']
                    stop_loss_pct = (self.stop_loss_atr_mult * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.02, 0.08))
                    
                    if self.use_trailing_only:
                        take_profit_pct = None
                    else:
                        take_profit_pct = float(stop_loss_pct * self.take_profit_mult)
                        
                    trailing_stop_pct = (self.trailing_stop_atr_mult * atr) / (close + 1e-9)
                    trailing_stop_pct = float(np.clip(trailing_stop_pct, 0.02, 0.08))
                    
                    if ticker in top_K_tickers and prob >= thresh_long:
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": 0.99,  # Forza il passaggio del filtro rigido del motore di backtest
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct
                        }
                    elif ticker in bottom_K_tickers and prob <= thresh_short:
                        signals[ticker] = {
                            "action": "SELL_SHORT",
                            "probability": 0.01,  # Forza il passaggio del filtro rigido del motore di backtest per shorting (prob <= 1 - threshold)
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct,
                            "trailing_stop_pct": trailing_stop_pct
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": prob}
        else:
            # Modalità standard senza ranking
            for ticker, prob in zip(valid_active_tickers, valid_active_probs):
                df = ticker_dfs[ticker]
                latest_row = df.iloc[-1]
                close = latest_row['close']
                
                if ticker in portfolio.positions:
                    pos = portfolio.positions[ticker]
                    if pos.position_type == "LONG":
                        if prob < (1 - self.probability_threshold + 0.10):
                            signals[ticker] = {"action": "SELL", "probability": prob}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": prob}
                    else: # SHORT
                        if prob > (self.probability_threshold - 0.10):
                            signals[ticker] = {"action": "BUY_TO_COVER", "probability": prob}
                        else:
                            signals[ticker] = {"action": "HOLD", "probability": prob}
                else:
                    atr = latest_row['atr_14']
                    stop_loss_pct = (4.0 * atr) / (close + 1e-9)
                    stop_loss_pct = float(np.clip(stop_loss_pct, 0.015, 0.08))
                    take_profit_pct = float(stop_loss_pct * 2.0)
                    
                    if prob >= self.probability_threshold:
                        signals[ticker] = {
                            "action": "BUY",
                            "probability": prob,
                            "stop_loss_pct": stop_loss_pct,
                            "take_profit_pct": take_profit_pct
                        }
                    else:
                        signals[ticker] = {"action": "HOLD", "probability": prob}
                        
        return signals


