import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import config
from database.db_manager import DBManager
from database.indicators import calculate_technical_indicators

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataIngestion")

try:
    import yfinance as yf
except ImportError:
    logger.error(
        "\n" + "="*70 +
        "\n[ERRORE DI SISTEMA] La libreria 'yfinance' non è installata!" +
        "\nPer favore, installala eseguendo il seguente comando nel terminale:" +
        "\n\n    pip install yfinance" +
        "\n" + "="*70 + "\n"
    )
    yf = None


def get_sp500_tickers() -> List[str]:
    """
    Recupera dinamicamente la lista aggiornata di tutti i costituenti dell'S&P 500
    estrapolando la tabella ufficiale da Wikipedia tramite requests e BeautifulSoup,
    eliminando la dipendenza facoltativa da lxml.
    Sostituisce i punti con trattini per allinearli al formato richiesto da Yahoo Finance.
    """
    try:
        logger.info("Recupero dei ticker S&P 500 in corso da Wikipedia (via BeautifulSoup)...")
        import requests
        from bs4 import BeautifulSoup
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        
        if not table:
            raise ValueError("Tabella dei costituenti 'constituents' non trovata nella pagina Wikipedia.")
            
        tickers = []
        for row in table.find_all("tr")[1:]:  # Salta la riga dell'intestazione
            cols = row.find_all("td")
            if cols:
                ticker = cols[0].text.strip()
                # Yahoo Finance usa il trattino '-' per le classi di azioni (es. BRK-B)
                ticker = ticker.replace('.', '-')
                tickers.append(ticker)
                
        tickers = sorted(list(set(tickers)))
        logger.info(f"Rilevati con successo {len(tickers)} ticker costituenti l'S&P 500.")
        return tickers
    except Exception as e:
        logger.error(f"Errore durante il recupero dei ticker S&P 500 da Wikipedia: {e}")
        logger.warning("Uso della lista di ticker di fallback definita nel file config.py.")
        return config.TICKERS


class YahooFinanceDataIngestion:
    """
    Gestisce il download e l'aggiornamento incrementale dei dati storici giornalieri (OHLCV)
    tramite Yahoo Finance. Esegue lo scaricamento in parallelo (ultra-veloce)
    ma salva i dati a DB in modo sequenziale per prevenire il file locking (WAL) su Windows.
    """

    def __init__(self) -> None:
        if yf is None:
            raise ImportError("Impossibile avviare l'ingestione dati: libreria 'yfinance' mancante.")
        self.db = DBManager()

    def run_ingestion(self, tickers: List[str] = config.TICKERS, max_workers: int = 12) -> None:
        """
        Esegue il processo di ingestione incrementale/allineamento per una lista di ticker.
        Combina la velocità del download HTTP parallelo multi-thread con la sicurezza
        di scritture sequenziali sul thread principale per prevenire crash di DuckDB su Windows.
        """
        total_tickers = len(tickers)
        logger.info(f"Avvio scaricamento parallelo (Max Workers: {max_workers}) per {total_tickers} ticker...")
        
        # Data di inizio richiesta in base alla configurazione
        required_start_date = datetime.now() - timedelta(days=config.DATA_DOWNLOAD_ANNI * 365)
        
        completed_count = 0
        failed_count = 0
        skipped_count = 0

        # Mappa per memorizzare i risultati dei download in memoria: Ticker -> (DataFrame, start_date)
        download_results = {}
        log_lock = threading.Lock()

        # FASE 1: Download parallelo tramite ThreadPoolExecutor (solo richieste HTTP, nessuna scrittura a DB)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for ticker in tickers:
                # Verifica dello stato attuale del DB (lettura sicura thread-safe)
                first_date = self.db.get_first_date(ticker)
                last_date = self.db.get_last_date(ticker)
                
                # Caso A: Nessun dato a DB
                if first_date is None or last_date is None:
                    start_date = required_start_date
                # Caso B: Richiesto allineamento storico (config aumentata)
                elif first_date > required_start_date + timedelta(days=7):
                    start_date = required_start_date
                # Caso C: Storico conforme. Incrementale
                else:
                    start_date = last_date + pd.Timedelta(days=1)
                    start_date = start_date.to_pydatetime()
                    
                    today = datetime.now()
                    if start_date.date() >= today.date():
                        # Controlla se anche gli indicatori sono allineati
                        last_indicator_date = self.db.get_last_indicator_date(ticker)
                        if last_indicator_date is not None and last_indicator_date.date() >= last_date.date():
                            skipped_count += 1
                            completed_count += 1
                            continue
                        
                        # Prezzi aggiornati ma indicatori mancanti: non scarichiamo nulla (df = None)
                        download_results[ticker] = (None, start_date)
                        continue

                # Sottomettiamo il download HTTP parallelo
                futures[executor.submit(self._fetch_from_yahoo, ticker, start_date, datetime.now())] = (ticker, start_date)

            # Monitora i completamenti del download parallelo
            for idx, future in enumerate(as_completed(futures)):
                ticker, start_date = futures[future]
                try:
                    df = future.result()
                    download_results[ticker] = (df, start_date)
                    
                    # Log di progresso periodico per il download
                    total_downloaded = len(download_results) + skipped_count
                    if total_downloaded % 50 == 0 or total_downloaded == total_tickers:
                        with log_lock:
                            logger.info(f"Scaricamento: {total_downloaded}/{total_tickers} completati...")
                except Exception as e:
                    with log_lock:
                        failed_count += 1
                        logger.error(f"[{ticker}] Errore nello scaricamento: {e}")

        # FASE 2: Scrittura sequenziale a Database e calcolo indicatori (sul thread principale)
        logger.info("Scaricamento completato. Inizio salvataggio ed elaborazione sequenziale su DuckDB...")
        
        tickers_to_process = sorted(list(download_results.keys()))
        for idx, ticker in enumerate(tickers_to_process):
            df, start_date = download_results[ticker]
            try:
                # Stampa avviso allineamento se necessario
                first_date = self.db.get_first_date(ticker)
                if first_date is not None and first_date > required_start_date + timedelta(days=7):
                    logger.warning(
                        f"[{ticker}] Allineamento storico: DB attuale (da {first_date.strftime('%Y-%m-%d')}) "
                        f"è inferiore a richiesto. Allineamento in corso..."
                    )
                
                # Inserimento prezzi se presenti
                if df is not None and not df.empty:
                    self.db.insert_ohlcv(df)
                    
                # Calcolo ed inserimento indicatori tecnici in batch
                df_full_history = self.db.fetch_ohlcv(ticker)
                if not df_full_history.empty:
                    df_indicators = calculate_technical_indicators(df_full_history)
                    if not df_indicators.empty:
                        self.db.insert_indicators(df_indicators)
                
                completed_count += 1
                
                # Log avanzamento della scrittura
                if (idx + 1) % 50 == 0 or (idx + 1) == len(tickers_to_process):
                    logger.info(f"Salvataggio DB: {idx + 1}/{len(tickers_to_process)} ticker elaborati e salvati.")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"[{ticker}] Errore salvataggio a DB o calcolo indicatori: {e}")

        logger.info(
            f"=== REPORT INGESTIONE FINALE ===\n"
            f"Ticker Processati con successo: {completed_count}/{total_tickers} "
            f"(di cui saltati già aggiornati: {skipped_count})\n"
            f"Ticker Falliti: {failed_count}"
        )

    def _fetch_from_yahoo(self, ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """
        Interroga le API di Yahoo Finance per scaricare i dati storici di un ticker.
        I prezzi sono rettificati automaticamente per split e dividendi (auto_adjust=True).
        """
        start_str = start.strftime("%Y-%m-%d")
        end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")  # End in Yahoo Finance è esclusivo
        
        try:
            yt_ticker = yf.Ticker(ticker)
            df_history = yt_ticker.history(start=start_str, end=end_str, interval="1d", auto_adjust=True)
            
            if df_history.empty:
                return None
                
            # Formattiamo per lo schema DuckDB
            df_history = df_history.reset_index()
            
            df_formatted = pd.DataFrame()
            df_formatted['ticker'] = [ticker] * len(df_history)
            df_formatted['timestamp'] = pd.to_datetime(df_history['Date']).dt.tz_localize(None)
            df_formatted['open'] = df_history['Open'].astype(float)
            df_formatted['high'] = df_history['High'].astype(float)
            df_formatted['low'] = df_history['Low'].astype(float)
            df_formatted['close'] = df_history['Close'].astype(float)
            df_formatted['volume'] = df_history['Volume'].astype(int)
            
            return df_formatted
            
        except Exception as e:
            raise e
