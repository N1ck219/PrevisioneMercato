import sys
import os
import logging
import argparse
from pathlib import Path

# Assicuriamoci che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RunIngestion")


def main():
    parser = argparse.ArgumentParser(
        description="Piattaforma Trading - Script di Scaricamento e Aggiornamento Dati"
    )
    
    parser.add_argument(
        "-t", "--tickers",
        type=str,
        help="Lista di ticker separati da virgola da scaricare (es. AAPL,TSLA). Se omesso, usa config.py."
    )
    
    parser.add_argument(
        "--sp500",
        action="store_true",
        help="Scarica dinamicamente ed inserisce i dati storici per tutti i 500 costituenti dell'S&P 500."
    )
    
    parser.add_argument(
        "--macro",
        action="store_true",
        help="Scarica e aggiorna i dati storici dei ticker macroeconomici e di sentiment (^VIX, ^TNX, DX-Y.NYB, SPY, QQQ)."
    )
    
    parser.add_argument(
        "-y", "--years",
        type=int,
        help=f"Anni di storico da scaricare. Se omesso, usa il default del config (attualmente {config.DATA_DOWNLOAD_ANNI} anni)."
    )
    
    parser.add_argument(
        "-c", "--clear",
        action="store_true",
        help="Cancella il database locale prima di iniziare il download per eseguire un caricamento pulito."
    )
    
    args = parser.parse_args()

    logger.info("=== AVVIO SCARICAMENTO E AGGIORNAMENTO DATI (Yahoo Finance) ===")

    # 1. Gestione cancellazione DB se richiesto (--clear)
    if args.clear:
        db_path = config.DB_PATH
        if db_path.exists():
            try:
                logger.warning(f"Opzione --clear attiva. Rimozione del database esistente in corso: {db_path}")
                os.remove(db_path)
                logger.info("Database cancellato con successo.")
            except Exception as e:
                logger.error(f"Impossibile cancellare il file del database: {e}")
                sys.exit(1)
        else:
            logger.info("Database non esistente, nessuna cancellazione necessaria.")

    # 2. Configurazione parametri dinamici passati da riga di comando
    if args.years is not None:
        logger.info(f"Parametro anni personalizzato da riga di comando: {args.years} anni (config: {config.DATA_DOWNLOAD_ANNI})")
        config.DATA_DOWNLOAD_ANNI = args.years
        
    tickers_to_load = config.TICKERS
    
    if args.macro:
        tickers_to_load = ["^VIX", "^TNX", "DX-Y.NYB", "SPY", "QQQ"]
        logger.info(f"Parametro --macro attivo. Ingestione dei ticker macro: {tickers_to_load}")
    elif args.sp500:
        try:
            from database.data_ingestion import get_sp500_tickers
            tickers_to_load = get_sp500_tickers()
        except Exception as e:
            logger.error(f"Impossibile recuperare i ticker S&P 500: {e}")
            sys.exit(1)
    elif args.tickers is not None:
        # Parsing dei ticker da stringa separata da virgola
        tickers_to_load = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        logger.info(f"Parametro ticker personalizzato da riga di comando: {tickers_to_load}")

    # 3. Avvio dell'ingestione dati
    try:
        from database.data_ingestion import YahooFinanceDataIngestion
    except ImportError:
        logger.error("Errore di importazione. Verifica di aver installato la libreria 'yfinance'.")
        sys.exit(1)
        
    try:
        # Inizializza l'ingestione dati (creerà anche il nuovo database se cancellato)
        ingestor = YahooFinanceDataIngestion()
        
        # Esegue il download/update incrementale per i ticker decisi
        ingestor.run_ingestion(tickers=tickers_to_load)
        
        logger.info("=== PROCESSO DI AGGIORNAMENTO COMPLETATO CON SUCCESSO! ===")
        print("\n[INFO] I tuoi dati nel database sono ora aggiornati e pronti per il trading/backtesting.")
        print("       Puoi lanciare nuovamente questo script con argomenti opzionali:")
        print("       Esempio: python run_ingestion.py --tickers TSLA,NVDA --years 5")
        
    except Exception as e:
        logger.error(f"Errore critico durante l'esecuzione dell'ingestione: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
