import sys
import argparse
from pathlib import Path
import pandas as pd

# Assicuriamoci che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from database.db_manager import DBManager


def main():
    parser = argparse.ArgumentParser(
        description="Piattaforma Trading - Visualizzatore Database Locale SQLite"
    )
    
    parser.add_argument(
        "-t", "--ticker",
        type=str,
        help="Visualizza i dati specifici per un ticker (es. AAPL)."
    )
    
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="Numero massimo di righe da visualizzare (default: 10)."
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Esegue ed esibisce una query SQL arbitraria sul DB."
    )
    
    args = parser.parse_args()
    
    db = DBManager()
    
    print("\n" + "="*60)
    print(" VISUALIZZATORE DATABASE LOCALE (SQLite)")
    print("="*60)
    print(f"File DB: {config.DB_PATH.resolve()}")
    print("-"*60)
    
    # 1. Se l'utente ha passato una query personalizzata
    if args.query:
        print(f"\n[QUERY PERSONALIZZATA] Esecuzione: {args.query}")
        try:
            df = db.execute_query(args.query)
            print(f"Righe restituite: {len(df)}")
            print("-"*60)
            print(df.to_string(index=False))
            print("="*60 + "\n")
            return
        except Exception as e:
            print(f"Errore durante l'esecuzione della query: {e}")
            return

    # 2. Informazioni Generali e Conteggi
    try:
        # Conteggio ohlcv
        cnt_ohlcv = db.execute_query("SELECT COUNT(*) FROM ohlcv").iloc[0, 0]
        # Conteggio ohlcv distinti ticker
        cnt_tickers = db.execute_query("SELECT COUNT(DISTINCT ticker) FROM ohlcv").iloc[0, 0]
        
        # Conteggio indicators (se tabella esiste)
        try:
            cnt_indicators = db.execute_query("SELECT COUNT(*) FROM indicators").iloc[0, 0]
        except Exception:
            cnt_indicators = "Tabella non inizializzata"
            
        print(f"Asset (Ticker) unici nel DB: {cnt_tickers}")
        print(f"Totale righe Prezzi (ohlcv):  {cnt_ohlcv:,}")
        if isinstance(cnt_indicators, int):
            print(f"Totale righe Indicatori:      {cnt_indicators:,}")
        else:
            print(f"Totale righe Indicatori:      {cnt_indicators}")
        print("-"*60)
        
    except Exception as e:
        print(f"Errore nel recupero delle informazioni del DB: {e}")
        return

    # 3. Visualizzazione campioni
    if args.ticker:
        ticker = args.ticker.upper()
        print(f"\n[DETTAGLIO TICKER] Asset: {ticker} (Ultime {args.limit} righe)")
        
        # Query che unisce i prezzi (ohlcv) e gli indicatori calcolati (indicators)
        query = """
            SELECT 
                o.ticker, 
                o.timestamp, 
                o.open, 
                o.high, 
                o.low, 
                o.close, 
                o.volume,
                i.sma_10,
                i.sma_50,
                i.rsi_14,
                i.macd,
                i.macd_signal,
                i.bb_upper,
                i.bb_lower,
                i.atr_14
            FROM ohlcv o
            LEFT JOIN indicators i 
                ON o.ticker = i.ticker 
               AND o.timestamp = i.timestamp
            WHERE o.ticker = ?
            ORDER BY o.timestamp DESC
            LIMIT ?
        """
        try:
            df = db.execute_query(query, (ticker, args.limit))
            if df.empty:
                print(f"Nessun dato trovato per il ticker [{ticker}].")
            else:
                # Arrotondiamo i decimali per una visualizzazione pulita in console
                numeric_cols = df.select_dtypes(include=['float64']).columns
                df[numeric_cols] = df[numeric_cols].round(4)
                print(df.to_string(index=False))
        except Exception as e:
            print(f"Errore nella query del ticker: {e}")
            
    else:
        # Mostriamo i ticker presenti e un campione aggregato
        print("\n[LISTA ASSSET DISPONIBILI] (Ticker e conteggio giorni)")
        query_assets = """
            SELECT 
                ticker, 
                MIN(timestamp) as da, 
                MAX(timestamp) as a, 
                COUNT(*) as record_totali
            FROM ohlcv 
            GROUP BY ticker 
            ORDER BY record_totali DESC, ticker ASC
            LIMIT 15
        """
        df_assets = db.execute_query(query_assets)
        print(df_assets.to_string(index=False))
        
        if cnt_tickers > 15:
            print(f"... ed altri {cnt_tickers - 15} asset.")
            
        print("\n[INFO] Per esaminare un ticker con i suoi indicatori, usa:")
        print(f"       python view_db.py --ticker AAPL --limit 10")
        
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
