import logging
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import threading

import config

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DBManager")


class DBManager:
    """
    Gestore di database locale basato su SQLite3.
    Gestisce la connessione, l'inizializzazione delle tabelle e le query.
    Fornisce compatibilità thread-safe con blocco mutex ed eleganti busy timeout
    per sventare completamente i problemi di file locking (WAL) tipici di Windows.
    """
    _instance: Optional['DBManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'DBManager':
        """Implementazione Pattern Singleton per connessione condivisa stabile."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DBManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        
        self.db_path = config.DB_PATH
        self.conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        
        # Connessione ed inizializzazione
        self.connect()
        self.initialize_tables()
        
        self._initialized = True

    def connect(self) -> None:
        """Stabilisce la connessione al database SQLite3."""
        with self._lock:
            if self.conn is None:
                try:
                    # Creazione directory se non esistente
                    self.db_path.parent.mkdir(exist_ok=True)
                    
                    # Connessione SQLite robusta con busy timeout di 30 secondi
                    # e check_same_thread=False protetto dal nostro lock mutex.
                    self.conn = sqlite3.connect(
                        str(self.db_path),
                        timeout=30.0,
                        check_same_thread=False
                    )
                    
                    # Abilitiamo la modalità WAL (Write-Ahead Logging) e ottimizzazioni di performance
                    cursor = self.conn.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA synchronous=NORMAL;")
                    cursor.execute("PRAGMA foreign_keys=ON;")
                    
                    logger.info(f"Connesso con successo al DB SQLite (WAL Mode): {self.db_path}")
                except Exception as e:
                    logger.error(f"Errore durante la connessione a SQLite: {e}")
                    raise e

    def get_connection(self) -> sqlite3.Connection:
        """Restituisce la connessione attiva. Se chiusa, la riapre."""
        if self.conn is None:
            self.connect()
        return self.conn

    def initialize_tables(self) -> None:
        """Crea le tabelle necessarie se non esistono già nel database."""
        conn = self.get_connection()
        with self._lock:
            try:
                cursor = conn.cursor()
                
                # Tabella per lo storico dei prezzi OHLCV giornaliero
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ohlcv (
                        ticker TEXT,
                        timestamp TEXT,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        PRIMARY KEY (ticker, timestamp)
                    )
                """)
                
                # Tabella per gli indicatori tecnici relazionata
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS indicators (
                        ticker TEXT,
                        timestamp TEXT,
                        sma_10 REAL,
                        sma_20 REAL,
                        sma_50 REAL,
                        sma_200 REAL,
                        ema_9 REAL,
                        ema_21 REAL,
                        rsi_14 REAL,
                        macd REAL,
                        macd_signal REAL,
                        macd_hist REAL,
                        bb_upper REAL,
                        bb_middle REAL,
                        bb_lower REAL,
                        atr_14 REAL,
                        PRIMARY KEY (ticker, timestamp),
                        FOREIGN KEY (ticker, timestamp) REFERENCES ohlcv (ticker, timestamp) ON DELETE CASCADE
                    )
                """)
                
                # Creazione indici velocizzati
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_time ON ohlcv (ticker, timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_ticker_time ON indicators (ticker, timestamp)")
                
                conn.commit()
                logger.info("Tabelle SQLite inizializzate correttamente.")
            except Exception as e:
                conn.rollback()
                logger.error(f"Errore durante l'inizializzazione delle tabelle SQLite: {e}")
                raise e

    def insert_ohlcv(self, df: pd.DataFrame) -> None:
        """
        Inserisce o aggiorna i dati OHLCV all'interno del DB tramite UPSERT nativo di SQLite.
        """
        if df.empty:
            logger.warning("Tentato inserimento di un DataFrame vuoto.")
            return

        required_cols = {'ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Il DataFrame deve contenere le colonne: {required_cols}")

        conn = self.get_connection()
        with self._lock:
            try:
                temp_df = df.copy()
                # SQLite archivia i timestamp come stringhe ISO
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Batch list per l'inserimento efficiente
                data_list = temp_df[['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO ohlcv (ticker, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, timestamp) 
                    DO UPDATE SET 
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume
                """, data_list)
                conn.commit()
                logger.info(f"Inserite/Aggiornate con successo {len(df)} righe in SQLite (ohlcv).")
            except Exception as e:
                conn.rollback()
                logger.error(f"Errore durante l'inserimento batch dei dati OHLCV in SQLite: {e}")
                raise e

    def insert_indicators(self, df: pd.DataFrame) -> None:
        """
        Inserisce o aggiorna i dati degli indicatori tecnici nel DB SQLite in modalità thread-safe.
        """
        if df.empty:
            return

        required_cols = {
            'ticker', 'timestamp', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_9', 'ema_21', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr_14'
        }
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Il DataFrame degli indicatori deve contenere le colonne: {required_cols}")

        conn = self.get_connection()
        with self._lock:
            try:
                temp_df = df.copy()
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                cols = [
                    'ticker', 'timestamp', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                    'ema_9', 'ema_21', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                    'bb_upper', 'bb_middle', 'bb_lower', 'atr_14'
                ]
                data_list = temp_df[cols].values.tolist()
                
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO indicators (
                        ticker, timestamp, sma_10, sma_20, sma_50, sma_200,
                        ema_9, ema_21, rsi_14, macd, macd_signal, macd_hist,
                        bb_upper, bb_middle, bb_lower, atr_14
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker, timestamp)
                    DO UPDATE SET
                        sma_10 = excluded.sma_10,
                        sma_20 = excluded.sma_20,
                        sma_50 = excluded.sma_50,
                        sma_200 = excluded.sma_200,
                        ema_9 = excluded.ema_9,
                        ema_21 = excluded.ema_21,
                        rsi_14 = excluded.rsi_14,
                        macd = excluded.macd,
                        macd_signal = excluded.macd_signal,
                        macd_hist = excluded.macd_hist,
                        bb_upper = excluded.bb_upper,
                        bb_middle = excluded.bb_middle,
                        bb_lower = excluded.bb_lower,
                        atr_14 = excluded.atr_14
                """, data_list)
                conn.commit()
                logger.info(f"Inseriti/Aggiornati con successo gli indicatori per {len(df)} righe in SQLite.")
            except Exception as e:
                conn.rollback()
                logger.error(f"Errore durante l'inserimento batch degli indicatori in SQLite: {e}")
                raise e

    def get_last_indicator_date(self, ticker: str) -> Optional[pd.Timestamp]:
        """
        Recupera l'ultima data registrata nella tabella indicators per un dato ticker.
        """
        conn = self.get_connection()
        with self._lock:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(timestamp) FROM indicators WHERE ticker = ?", (ticker,))
                res = cursor.fetchone()
                if res and res[0] is not None:
                    return pd.Timestamp(res[0])
                return None
            except Exception as e:
                return None

    def get_last_date(self, ticker: str) -> Optional[pd.Timestamp]:
        """
        Recupera l'ultima data registrata nel database per un dato ticker.
        """
        conn = self.get_connection()
        with self._lock:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(timestamp) FROM ohlcv WHERE ticker = ?", (ticker,))
                res = cursor.fetchone()
                if res and res[0] is not None:
                    return pd.Timestamp(res[0])
                return None
            except Exception as e:
                logger.error(f"Errore nel recupero dell'ultima data per {ticker}: {e}")
                return None

    def get_first_date(self, ticker: str) -> Optional[pd.Timestamp]:
        """
        Recupera la prima data (la più vecchia) registrata nel database per un dato ticker.
        """
        conn = self.get_connection()
        with self._lock:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MIN(timestamp) FROM ohlcv WHERE ticker = ?", (ticker,))
                res = cursor.fetchone()
                if res and res[0] is not None:
                    return pd.Timestamp(res[0])
                return None
            except Exception as e:
                logger.error(f"Errore nel recupero della prima data per {ticker}: {e}")
                return None

    def fetch_ohlcv(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Recupera i dati storici dal database SQLite come DataFrame di Pandas.
        """
        conn = self.get_connection()
        query = "SELECT * FROM ohlcv WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp ASC"
        
        with self._lock:
            try:
                df = pd.read_sql_query(query, conn, params=params)
                if not df.empty and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                logger.error(f"Errore nel recupero dati per {ticker}: {e}")
                return pd.DataFrame()

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Esegue una query di selezione generica e restituisce il risultato come DataFrame.
        """
        conn = self.get_connection()
        with self._lock:
            try:
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)
                if not df.empty and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                logger.error(f"Errore durante l'esecuzione della query SQL: {e}")
                raise e

    def close(self) -> None:
        """Chiude in sicurezza la connessione a SQLite3."""
        with self._lock:
            if self.conn is not None:
                try:
                    self.conn.close()
                    self.conn = None
                    logger.info("Connessione a SQLite chiusa correttamente.")
                except Exception as e:
                    logger.error(f"Errore durante la chiusura di SQLite: {e}")
