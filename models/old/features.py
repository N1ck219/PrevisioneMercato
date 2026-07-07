import pandas as pd
import numpy as np
from core.config import MACRO_LABELS_ORDERED

class FeatureEngine:
    @staticmethod
    def add_technical_indicators(df_in):
        df = df_in.copy()
        delta = df['prezzo'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        if 'High' in df.columns and 'Low' in df.columns:
            tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['prezzo'].shift()), abs(df['Low']-df['prezzo'].shift())], axis=1).max(axis=1)
            df['ATRr_14'] = tr.rolling(14).mean()
            
        df['SMA_20'] = df['prezzo'].rolling(20).mean()
        df['STD_20'] = df['prezzo'].rolling(20).std() + 1e-9
        df['Bollinger_%B'] = (df['prezzo'] - (df['SMA_20'] - 2*df['STD_20'])) / (4*df['STD_20'])
        df['Bollinger_Width'] = (4 * df['STD_20']) / (df['SMA_20'] + 1e-9)
        
        df['SMA_200'] = df['prezzo'].rolling(200).mean()
        df['Dist_SMA200'] = (df['prezzo'] - df['SMA_200']) / (df['SMA_200'] + 1e-9)
        
        df['SMA_50'] = df['prezzo'].rolling(50).mean()
        df['Dist_SMA50'] = (df['prezzo'] - df['SMA_50']) / (df['SMA_50'] + 1e-9)
        
        df['ret'] = df['prezzo'].pct_change().fillna(0)
        
        if 'Volume' in df.columns:
            df['vol_ret'] = df['Volume'].pct_change().fillna(0)
            df['OBV_ret'] = (np.sign(df['ret']) * df['Volume']).fillna(0).cumsum().pct_change().fillna(0)
            
        return df

    @staticmethod
    def process_stock_features(df_t, macro_dict=None):
        df = df_t.copy()
        if macro_dict:
            for label, df_m in macro_dict.items(): 
                df = df.merge(df_m, on='Date', how='left')
                
        df = df.ffill().bfill()
        
        if 'Close' in df.columns:
            df.rename(columns={'Date': 'data', 'Close': 'prezzo'}, inplace=True)
            
        df = FeatureEngine.add_technical_indicators(df)
        
        if macro_dict:
            # Crea feature macro per compatibilità con tutti i bot azionari (v4.3, 4.6, 5.6, 6.4)
            mapping_v4 = {'nasdaq_close': 'nasdaq_ret', 'vix_close': 'vix_ret', 'tnx_close': 'tnx_ret', 'soxx_close': 'soxx_ret', 'gld_close': 'gld_ret'}
            for old, new in mapping_v4.items():
                if old in df.columns:
                    ret_calc = df[old].pct_change().fillna(0)
                    df[new] = ret_calc
                    df[f"{old}_ret"] = ret_calc  # Usato da v5.6 e v6.4
                    
        return df.replace([np.inf, -np.inf], np.nan).fillna(0).dropna()
    
    @staticmethod
    def extract_features(df, macro_labels=MACRO_LABELS_ORDERED):
        # Questo serve per la V5.6 e V6.4 che estraggono tensori separati
        tech_cols = ['ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width', 'ATRr_14', 'Dist_SMA200', 'Dist_SMA50', 'OBV_ret']
        
        # Aggiungiamo anche il sentiment globale alle feature macro se presente
        macro_cols = [f"{m}_ret" for m in macro_labels]
        if 'Global_Sentiment_Score' in df.columns:
            macro_cols.extend(['Global_Sentiment_Score', 'Global_Confidence', 'Global_Volatility'])
            
        return df[tech_cols].values, df[macro_cols].values

    @staticmethod
    def process_crypto_features(df_raw):
        """Feature engineering per crypto (senza dati macro, usa SMA_50 invece di SMA_200)."""
        df = df_raw.copy()
        df = FeatureEngine.add_technical_indicators(df)
        return df.replace([np.inf, -np.inf], np.nan).fillna(0).dropna()

    # ── V7.0 Intraday Features ────────────────────────────────

    @staticmethod
    def add_vwap(df):
        """Calcola VWAP con reset corretto ad ogni apertura di sessione (09:30 ET).
        
        Il VWAP viene resettato all'inizio di ogni giornata di trading.
        Aggiunge le colonne:
            - VWAP_calc: il VWAP calcolato (usato se il dato API è assente)
            - VWAP_ratio: (prezzo / VWAP) - 1, indica la posizione relativa al VWAP
        """
        df = df.copy()
        dt = pd.to_datetime(df['Datetime'])
        df['_date'] = dt.dt.date
        
        # Prezzo tipico per ogni barra
        if 'High' in df.columns and 'Low' in df.columns:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
        else:
            typical_price = df['Close']
        
        vol = df['Volume'].fillna(0)
        
        # Reset cumulativo per sessione (ogni giorno)
        cum_pv = (typical_price * vol).groupby(df['_date']).cumsum()
        cum_vol = vol.groupby(df['_date']).cumsum()
        
        vwap = cum_pv / (cum_vol + 1e-9)
        
        # Usa il VWAP da API se disponibile, altrimenti il calcolato
        if 'VWAP' in df.columns and df['VWAP'].sum() > 0:
            df['VWAP_final'] = df['VWAP'].where(df['VWAP'] > 0, vwap)
        else:
            df['VWAP_final'] = vwap
        
        df['VWAP_ratio'] = (df['Close'] / (df['VWAP_final'] + 1e-9)) - 1.0
        df.drop(columns=['_date'], inplace=True)
        
        return df

    @staticmethod
    def add_session_features(df):
        """Aggiunge feature temporali della sessione di trading.
        
        Colonne aggiunte:
            - minutes_since_open: minuti trascorsi dalle 09:30 ET (0-390)
            - session_pct: percentuale della sessione completata (0.0-1.0)
            - is_power_hour: 1 se nell'ultima ora di trading (15:00-16:00)
            - is_opening_range: 1 se nei primi 30 minuti (09:30-10:00)
        """
        df = df.copy()
        dt = pd.to_datetime(df['Datetime'])
        
        # Minuti dall'apertura (09:30 = 570 minuti dalla mezzanotte)
        minutes_from_midnight = dt.dt.hour * 60 + dt.dt.minute
        market_open_minutes = 9 * 60 + 30  # 09:30
        
        df['minutes_since_open'] = (minutes_from_midnight - market_open_minutes).clip(lower=0, upper=390)
        df['session_pct'] = df['minutes_since_open'] / 390.0
        df['is_power_hour'] = ((dt.dt.hour == 15) & (dt.dt.minute >= 0)).astype(int)
        df['is_opening_range'] = (df['minutes_since_open'] <= 30).astype(int)
        
        return df

    @staticmethod
    def process_intraday_features(df_intraday, macro_dict=None):
        """Feature engineering per dati intraday a 1 minuto (V7.0).
        
        Pipeline completa:
        1. Indicatori tecnici standard (RSI, Bollinger, ATR) adattati ai dati 1-min
        2. VWAP con reset di sessione
        3. Feature di sessione (minuti dall'apertura, power hour, ecc.)
        4. Merge con dati macro (risoluzione daily, forward-fill su barre intraday)
        """
        df = df_intraday.copy()
        
        # Rinomina Close in prezzo per compatibilità con add_technical_indicators
        if 'Close' in df.columns and 'prezzo' not in df.columns:
            df['prezzo'] = df['Close']
        
        # 1. Indicatori tecnici
        df = FeatureEngine.add_technical_indicators(df)
        
        # 2. VWAP
        df = FeatureEngine.add_vwap(df)
        
        # 3. Feature di sessione
        df = FeatureEngine.add_session_features(df)
        
        # 4. Merge macro (risoluzione daily → forward-fill su dati intraday)
        if macro_dict:
            dt = pd.to_datetime(df['Datetime'])
            df['_merge_date'] = dt.dt.strftime('%Y-%m-%d')
            
            for label, df_m in macro_dict.items():
                df_m_copy = df_m.copy()
                df_m_copy.rename(columns={'Date': '_merge_date'}, inplace=True)
                df = df.merge(df_m_copy, on='_merge_date', how='left')
            
            df.drop(columns=['_merge_date'], inplace=True)
            df = df.ffill().bfill()
            
            # Calcola rendimenti macro
            for label in macro_dict.keys():
                if label in df.columns:
                    df[f"{label}_ret"] = df[label].pct_change().fillna(0)
        
        return df.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def extract_intraday_features(df, macro_labels=MACRO_LABELS_ORDERED):
        """Estrae feature separate per i due rami del modello Split Brain V7.0.
        
        Returns:
            tech_features: ndarray — Feature tecniche + VWAP + sessione (ramo tecnico)
            macro_features: ndarray — Feature macro + flag sessione (ramo macro)
        """
        tech_cols = [
            'ret', 'vol_ret', 'RSI_14', 'Bollinger_%B', 'Bollinger_Width',
            'ATRr_14', 'Dist_SMA50', 'OBV_ret', 'VWAP_ratio',
            'minutes_since_open', 'session_pct'
        ]
        
        macro_cols = [f"{m}_ret" for m in macro_labels]
        macro_cols.extend(['is_power_hour', 'is_opening_range'])
        
        # Filtra solo colonne che esistono
        tech_cols = [c for c in tech_cols if c in df.columns]
        macro_cols = [c for c in macro_cols if c in df.columns]
        
        return df[tech_cols].values, df[macro_cols].values
