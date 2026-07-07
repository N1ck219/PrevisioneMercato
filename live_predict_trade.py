#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
📈 Script Modulare per Previsioni e Operatività in Tempo Reale (Alpaca Paper)
Questo script consente di caricare un modello specifico, effettuare previsioni basate
sull'ultimo stato del database SQLite locale e sincronizzare le posizioni con un account
Alpaca Paper (associato dinamicamente al modello selezionato).
"""

import os
import sys
import argparse
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Assicuriamoci che la directory radice sia nel path di sistema
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import config
from database.db_manager import DBManager
from backtest.engine import Portfolio, Position

# Logger configurato in modo chiaro e professionale
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "live_trading.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("LiveTrading")

# Mappatura delle strategie per nome breve
STRATEGY_IMPORTS = {
    "nn_v1": ("backtest.strategy", "NeuralNetworkStrategy"),
    "nn_v2": ("backtest.strategy", "NeuralNetworkV2Strategy"),
    "nn_v3": ("backtest.strategy", "NeuralNetworkV3Strategy"),
    "nn_v4": ("backtest.strategy", "NeuralNetworkV4Strategy"),
    "nn_v5": ("backtest.strategy", "NeuralNetworkV5Strategy"),
    "nn_v6": ("backtest.strategy", "NeuralNetworkV6Strategy"),
    "nn_v7": ("backtest.strategy", "NeuralNetworkV7Strategy"),
    "nn_v8": ("backtest.strategy", "NeuralNetworkV8Strategy"),
    "nn_v9": ("backtest.strategy", "NeuralNetworkV9Strategy"),
    "nn_v10": ("backtest.strategy", "NeuralNetworkV10Strategy"),
    "nn_v11": ("backtest.strategy", "NeuralNetworkV11Strategy"),
    "moe_v1": ("backtest.strategy", "MoEStrategyV1"),
    "sma": ("backtest.strategy", "SMAXStrategy")
}

DEFAULT_MODELS = {
    "nn_v1": "neural_model.pth",
    "nn_v2": "neural_model.pth",
    "nn_v3": "neural_model.pth",
    "nn_v4": "neural_model.pth",
    "nn_v5": "neural_model.pth",
    "nn_v6": "neural_model.pth",
    "nn_v7": "neural_model.pth",
    "nn_v8": "neural_model.pth",
    "nn_v9": "neural_model.pth",
    "nn_v10": "neural_model_v10.pth",
    "nn_v11": "neural_model_v11.pth",
    "moe_v1": "neural_model_moe.pth",
}


class AlpacaClient:
    """
    Client leggero per interagire con le API REST di Alpaca.
    """
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }

    def get_account(self) -> Dict[str, Any]:
        """Recupera le informazioni sull'account (bilancio, equity, cash)."""
        url = f"{self.base_url}/v2/account"
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Recupera le posizioni aperte correnti."""
        url = f"{self.base_url}/v2/positions"
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def submit_order(
        self, 
        ticker: str, 
        qty: float, 
        side: str, 
        order_type: str = "market", 
        time_in_force: str = "gtc",
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Invia un ordine ad Alpaca (supporta ordini bracket per SL/TP)."""
        url = f"{self.base_url}/v2/orders"
        data = {
            "symbol": ticker,
            "qty": str(qty),
            "side": side.lower(),  # "buy" o "sell"
            "type": order_type,
            "time_in_force": time_in_force
        }
        
        if take_profit_price is not None or stop_loss_price is not None:
            data["order_class"] = "bracket"
            if take_profit_price is not None:
                data["take_profit"] = {"limit_price": f"{take_profit_price:.2f}"}
            if stop_loss_price is not None:
                data["stop_loss"] = {"stop_price": f"{stop_loss_price:.2f}"}
                
        response = requests.post(url, json=data, headers=self.headers, timeout=15)
        if response.status_code != 200 and response.status_code != 201:
            logger.error(f"Errore invio ordine per {ticker}: {response.text}")
        response.raise_for_status()
        return response.json()

    def close_all_positions(self) -> List[Dict[str, Any]]:
        """Liquida tutte le posizioni aperte."""
        url = f"{self.base_url}/v2/positions"
        response = requests.delete(url, headers=self.headers, timeout=15)
        response.raise_for_status()
        return response.json()


def load_alpaca_credentials_for_model(model_name: str, custom_env: Optional[str] = None) -> AlpacaClient:
    """
    Risolve e carica le credenziali Alpaca per il modello selezionato.
    Cerca in ordine:
    1. Un file .env specificato dall'utente (--env_file)
    2. Un file specifico per il modello (es. .env.nn_v6 o .env.nn_v10)
    3. Il file .env predefinito nella root del progetto.
    """
    # Ripristiniamo/Saliamo il setup originale
    original_env = {
        "ALPACA_API_KEY": os.environ.get("ALPACA_API_KEY"),
        "ALPACA_SECRET_KEY": os.environ.get("ALPACA_SECRET_KEY"),
        "ALPACA_PAPER": os.environ.get("ALPACA_PAPER")
    }

    env_to_load = None
    if custom_env:
        env_to_load = custom_env
    else:
        # Cerchiamo .env.<model_name>
        model_env_file = f".env.{model_name}"
        if (BASE_DIR / model_env_file).exists():
            env_to_load = model_env_file
        else:
            env_to_load = ".env"

    logger.info(f"Caricamento dell'ambiente Alpaca da: {env_to_load}...")
    
    # Carica l'ambiente
    config.load_env(env_to_load)

    api_key = config.ALPACA_API_KEY
    secret_key = config.ALPACA_SECRET_KEY
    paper = config.ALPACA_PAPER

    if api_key == "YOUR_API_KEY" or not api_key or not secret_key:
        logger.error(
            f"Errore: Credenziali Alpaca non impostate nel file {env_to_load}. "
            "Assicurati che ALPACA_API_KEY e ALPACA_SECRET_KEY siano valorizzate correttamente."
        )
        sys.exit(1)

    logger.info(f"Credenziali Alpaca caricate. Key ID: ...{api_key[-6:] if len(api_key) > 6 else 'N/D'} (Paper: {paper})")
    
    # Ripristiniamo l'ambiente globale per non inquinare altre parti
    for k, v in original_env.items():
        if v is not None:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]

    return AlpacaClient(api_key, secret_key, paper)


def get_latest_market_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Recupera gli ultimi 300 record per ciascun ticker dal database SQLite locale
    per consentire alle medie mobili e al lookback del modello di calcolare correttamente i segnali.
    """
    db = DBManager()
    historical_data: Dict[str, pd.DataFrame] = {}

    logger.info(f"Recupero ultimi dati di mercato per {len(tickers)} ticker dal database locale...")

    for ticker in tickers:
        query = """
            SELECT 
                o.ticker, o.timestamp, o.open, o.high, o.low, o.close, o.volume,
                i.sma_10, i.sma_20, i.sma_50, i.sma_200, 
                i.ema_9, i.ema_21, i.rsi_14, 
                i.macd, i.macd_signal, i.macd_hist, 
                i.bb_upper, i.bb_middle, i.bb_lower, i.atr_14
            FROM ohlcv o
            LEFT JOIN indicators i 
                ON o.ticker = i.ticker 
               AND o.timestamp = i.timestamp
            WHERE o.ticker = ?
            ORDER BY o.timestamp DESC
            LIMIT 300
        """
        df = db.execute_query(query, (ticker,))
        if df.empty:
            logger.warning(f"Nessun dato trovato nel database per il ticker: {ticker}")
            continue

        # Invertiamo il DataFrame per averlo in ordine cronologico corretto (dal più vecchio al più recente)
        df = df.iloc[::-1].reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        historical_data[ticker] = df

    logger.info(f"Dati storici estratti con successo per {len(historical_data)}/{len(tickers)} ticker.")
    return historical_data


def instantiate_strategy(model_name: str, model_file: Optional[str], probability_threshold: float) -> Any:
    """
    Importa dinamicamente ed istanzia la classe della strategia selezionata.
    """
    if model_name not in STRATEGY_IMPORTS:
        raise ValueError(f"Modello '{model_name}' non supportato. Scelte valide: {list(STRATEGY_IMPORTS.keys())}")

    module_name, class_name = STRATEGY_IMPORTS[model_name]
    
    # Importazione dinamica
    try:
        module = __import__(module_name, fromlist=[class_name])
        strategy_class = getattr(module, class_name)
    except Exception as e:
        logger.error(f"Impossibile importare {class_name} da {module_name}: {e}")
        raise e

    # Determinazione del file dei pesi
    model_to_use = model_file if model_file is not None else DEFAULT_MODELS.get(model_name, "neural_model.pth")
    
    logger.info(f"Inizializzazione della strategia '{class_name}' con il file pesi '{model_to_use}'...")

    # Instanziamo la strategia
    # Alcuni modelli di base (es. sma) non richiedono model_filename
    if model_name == "sma":
        return strategy_class()
    else:
        # Per i modelli v4+, passiamo i parametri ottimali del backtest
        try:
            return strategy_class(
                model_filename=model_to_use,
                probability_threshold=probability_threshold
            )
        except TypeError:
            # Fallback se la strategia non ha i parametri standard
            return strategy_class(model_filename=model_to_use)


def rebuild_portfolio_state(alpaca_client: AlpacaClient) -> Portfolio:
    """
    Interroga l'API di Alpaca per recuperare bilancio ed esposizioni reali
    e ricostruire l'oggetto Portfolio locale compatibile con la logica della strategia.
    """
    account_info = alpaca_client.get_account()
    positions_info = alpaca_client.get_positions()

    equity = float(account_info["equity"])
    cash = float(account_info["cash"])

    logger.info(f"Stato Conto Alpaca - Equity Totale: ${equity:,.2f} | Liquidità Disponibile: ${cash:,.2f}")

    # Costruiamo il portfolio locale
    portfolio = Portfolio(initial_capital=equity)
    portfolio.sub_balances = {"SHARED": cash}

    # Popoliamo con le posizioni reali attive su Alpaca
    for pos in positions_info:
        ticker = pos["symbol"]
        qty = float(pos["qty"])
        avg_entry_price = float(pos["avg_entry_price"])
        current_price = float(pos["current_price"])
        side = pos["side"].upper()  # LONG o SHORT

        # Ricostruiamo la posizione locale
        local_pos = Position(
            ticker=ticker,
            shares=qty,
            entry_price=avg_entry_price,
            entry_date=datetime.now(),  # Fallback data odierna
            position_type="SHORT" if side == "SHORT" else "LONG"
        )
        local_pos.current_price = current_price
        local_pos.unrealized_pnl = float(pos["unrealized_intraday_pl"])
        
        portfolio.positions[ticker] = local_pos
        logger.info(f"Rilevata posizione attiva: {side} {qty} {ticker} a Prezzo Medio ${avg_entry_price:.2f}")

    return portfolio


def execute_trades_on_alpaca(alpaca_client: AlpacaClient, signals: Dict[str, Dict[str, Any]], portfolio: Portfolio, tickers: List[str], max_slots: int = 8) -> List[str]:
    """
    Invia gli ordini necessari ad Alpaca per allineare il portafoglio reale
    con le decisioni generate dalla strategia.
    Ritorna una lista di stringhe con i dettagli delle operazioni effettuate.
    """
    logger.info("\n=== INIZIO FASE DI ESECUZIONE ORDINI ===\n")
    trades_log: List[str] = []
    
    # 1. Gestione delle Vendite / Chiusure Posizioni (eseguite per prime per sbloccare liquidità)
    for ticker in tickers:
        signal = signals.get(ticker, {"action": "HOLD"})
        action = signal["action"]

        if ticker in portfolio.positions:
            pos = portfolio.positions[ticker]
            # Se la strategia dice SELL e abbiamo una posizione LONG aperta, oppure se dice BUY_TO_COVER e siamo SHORT
            if (action == "SELL" and pos.position_type == "LONG") or \
               (action == "BUY_TO_COVER" and pos.position_type == "SHORT"):
                msg = f"🔄 Chiusura posizione su {ticker} ({pos.position_type}): {pos.shares} quote."
                logger.info(f"[ORDINE] {msg}")
                alpaca_client.submit_order(ticker=ticker, qty=pos.shares, side="sell" if pos.position_type == "LONG" else "buy")
                trades_log.append(msg)
                # Aggiorniamo lo stato locale del portafoglio per il conteggio degli slot
                del portfolio.positions[ticker]
            
            # Se abbiamo una posizione opposta a quella voluta
            elif action == "BUY" and pos.position_type == "SHORT":
                msg = f"🔄 Inversione: Chiusura SHORT di {pos.shares} quote su {ticker}."
                logger.info(f"[ORDINE] {msg}")
                alpaca_client.submit_order(ticker=ticker, qty=pos.shares, side="buy")
                trades_log.append(msg)
                del portfolio.positions[ticker]
            elif action == "SELL_SHORT" and pos.position_type == "LONG":
                msg = f"🔄 Inversione: Chiusura LONG di {pos.shares} quote su {ticker}."
                logger.info(f"[ORDINE] {msg}")
                alpaca_client.submit_order(ticker=ticker, qty=pos.shares, side="sell")
                trades_log.append(msg)
                del portfolio.positions[ticker]

    # 2. Gestione degli Acquisti / Apertura Posizioni
    # Recuperiamo l'equity per calcolare le size corrette
    account_info = alpaca_client.get_account()
    available_cash = float(account_info["cash"])
    
    # Calcolo di allocazione standard basata sui pesi del segnale
    for ticker in tickers:
        signal = signals.get(ticker, {"action": "HOLD"})
        action = signal["action"]
        # Calcolo del peso reale basato sulla size massima modificata dal confidence_multiplier del modello
        base_weight = config.BACKTEST_MAX_POSITION_SIZE
        weight = signal.get("weight", base_weight * signal.get("confidence_multiplier", 1.0))

        # Se non abbiamo posizioni attive e il segnale è BUY
        if ticker not in portfolio.positions:
            if action in ["BUY", "SELL_SHORT"]:
                # Verifichiamo il limite massimo di posizioni (slots)
                if len(portfolio.positions) >= max_slots:
                    logger.warning(f"Limite massimo posizioni ({max_slots}) raggiunto. Salto apertura su {ticker}.")
                    continue

                if action == "BUY":
                    # Calcola il valore in USD da investire
                    investment_usd = float(account_info["equity"]) * weight
                    # Se supera la cassa disponibile, riduciamo all'85% della cassa per sicurezza/commissioni
                    if investment_usd > available_cash:
                        investment_usd = available_cash * 0.95

                    if investment_usd > 10.0:  # Soglia minima $10 per trade
                        # Otteniamo l'ultimo prezzo di chiusura per stimare le quote da comprare
                        db = DBManager()
                        last_price_df = db.execute_query("SELECT close FROM ohlcv WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1", (ticker,))
                        if not last_price_df.empty:
                            last_close = float(last_price_df.iloc[0, 0])
                            qty_to_buy = int(investment_usd / last_close)
                            if qty_to_buy > 0:
                                # Calcolo dei prezzi di SL e TP per l'ordine bracket
                                sl_pct = signal.get("stop_loss_pct")
                                tp_pct = signal.get("take_profit_pct")
                                sl_price = round(last_close * (1.0 - sl_pct), 2) if sl_pct else None
                                tp_price = round(last_close * (1.0 + tp_pct), 2) if tp_pct else None
                                
                                msg = f"🟢 BUY {ticker}: {qty_to_buy} quote (Prezzo: ${last_close:.2f}, SL: {sl_price}, TP: {tp_price})."
                                logger.info(f"[ORDINE] {msg}")
                                alpaca_client.submit_order(
                                    ticker=ticker, 
                                    qty=qty_to_buy, 
                                    side="buy",
                                    stop_loss_price=sl_price,
                                    take_profit_price=tp_price
                                )
                                trades_log.append(msg)
                            else:
                                logger.warning(f"Capitale insufficiente per comprare anche 1 quota di {ticker} a ${last_close:.2f}")
            
            elif action == "SELL_SHORT":
                investment_usd = float(account_info["equity"]) * weight
                if investment_usd > available_cash:
                    investment_usd = available_cash * 0.95

                if investment_usd > 10.0:
                    db = DBManager()
                    last_price_df = db.execute_query("SELECT close FROM ohlcv WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1", (ticker,))
                    if not last_price_df.empty:
                        last_close = float(last_price_df.iloc[0, 0])
                        qty_to_short = int(investment_usd / last_close)
                        if qty_to_short > 0:
                            # Calcolo dei prezzi di SL e TP per l'ordine bracket
                            sl_pct = signal.get("stop_loss_pct")
                            tp_pct = signal.get("take_profit_pct")
                            sl_price = round(last_close * (1.0 + sl_pct), 2) if sl_pct else None
                            tp_price = round(last_close * (1.0 - tp_pct), 2) if tp_pct else None
                            
                            msg = f"🔴 SELL SHORT {ticker}: {qty_to_short} quote (Prezzo: ${last_close:.2f}, SL: {sl_price}, TP: {tp_price})."
                            logger.info(f"[ORDINE] {msg}")
                            alpaca_client.submit_order(
                                ticker=ticker, 
                                qty=qty_to_short, 
                                side="sell",
                                stop_loss_price=sl_price,
                                take_profit_price=tp_price
                            )
                            trades_log.append(msg)

    logger.info("\n=== FASE DI ESECUZIONE ORDINI COMPLETATA ===\n")
    return trades_log


def send_telegram_message(message: str) -> None:
    """Invia un messaggio di notifica su Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram non configurato (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID assenti). Salto la notifica.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("Notifica Telegram inviata con successo.")
        else:
            logger.error(f"Errore invio Telegram ({response.status_code}): {response.text}")
    except Exception as e:
        logger.error(f"Eccezione durante l'invio su Telegram: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Piattaforma Trading - Script di Previsione & Trading in Produzione (Alpaca Paper)"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="nn_v6",
        choices=list(STRATEGY_IMPORTS.keys()),
        help="Il modello/strategia da caricare (default: nn_v6)."
    )

    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="Il file specifico dei pesi del modello da caricare (es. neural_model.pth)."
    )

    parser.add_argument(
        "-pt", "--probability_threshold",
        type=float,
        default=0.525,
        help="La soglia probabilistica di attivazione per generare segnali di BUY (default: 0.525)."
    )

    parser.add_argument(
        "-t", "--tickers",
        type=str,
        help="Lista di ticker separati da virgola da analizzare. Se omesso, usa i ticker configurati in config.py."
    )

    parser.add_argument(
        "--env_file",
        type=str,
        default=None,
        help="Specifica manualmente il file .env delle credenziali (es. .env.nn_v6)."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Se attivo, invia realmente gli ordini di allineamento al conto Alpaca Paper. Altrimenti effettua solo la simulazione/predizione (Dry Run)."
    )

    parser.add_argument(
        "--pool_size",
        type=int,
        default=None,
        help="Se specificato, seleziona i primi N ticker più attivi nel database automaticamente."
    )

    parser.add_argument(
        "--max_slots",
        type=int,
        default=8,
        help="Numero massimo di posizioni aperte contemporaneamente (default: 8)."
    )

    parser.add_argument(
        "--alphabetical",
        action="store_true",
        help="Se attivo e --pool_size è impostato, seleziona i ticker in ordine alfabetico invece che per volume medio."
    )

    args = parser.parse_args()

    print("\n" + "="*75)
    print("           📈 LIVE PREDICTION & TRADING MANAGER (ALPACA PAPER) 📈")
    print("="*75)
    logger.info(f"Modello selezionato: {args.model}")

    # 1. Carica le credenziali Alpaca per il modello ed istanzia il client
    alpaca_client = load_alpaca_credentials_for_model(args.model, args.env_file)

    # 2. Definisce la lista dei ticker
    db = DBManager()
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        logger.info(f"Configurata lista personalizzata di {len(tickers)} ticker.")
    elif args.pool_size is not None:
        if args.alphabetical:
            query = "SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker ASC LIMIT ?"
            tickers = db.execute_query(query, (args.pool_size,))['ticker'].tolist()
            logger.info(f"Selezionati automaticamente i primi {args.pool_size} ticker in ordine alfabetico dal DB.")
        else:
            query = "SELECT ticker, AVG(volume) as avg_vol FROM ohlcv GROUP BY ticker ORDER BY avg_vol DESC LIMIT ?"
            tickers = db.execute_query(query, (args.pool_size,))['ticker'].tolist()
            logger.info(f"Selezionati automaticamente i primi {args.pool_size} ticker più attivi dal DB.")
    else:
        tickers = config.TICKERS
        logger.info("Nessuna lista specificata. Utilizzo dei ticker predefiniti in config.py.")
    
    logger.info(f"Asset da monitorare ({len(tickers)}): {tickers}")

    # 3. Recupera gli ultimi dati di mercato con gli indicatori tecnici dal DB locale
    historical_data = get_latest_market_data(tickers)
    
    if not historical_data:
        logger.error("Nessun dato di mercato disponibile nel database locale. Impossibile calcolare i segnali.")
        sys.exit(1)

    # 4. Ricostruisce lo stato del portafoglio reale recuperandolo da Alpaca
    portfolio = rebuild_portfolio_state(alpaca_client)

    # 5. Carica il modello PyTorch ed istanzia la strategia
    try:
        strategy = instantiate_strategy(args.model, args.model_file, args.probability_threshold)
    except Exception as e:
        logger.critical(f"Errore critico nell'istanziazione della strategia: {e}")
        sys.exit(1)

    # 6. Genera i segnali predittivi per la data corrente (l'ultimo record disponibile)
    # Troviamo la data massima tra tutti i ticker caricati
    latest_dates = [df.index[-1] for df in historical_data.values()]
    current_date = max(latest_dates)
    logger.info(f"Generazione segnali per la data di mercato più recente: {current_date.strftime('%Y-%m-%d')}")

    signals = strategy.generate_signals(historical_data, portfolio, current_date)

    # Mostra i segnali calcolati
    print("\n" + "-"*50)
    print(f"   SEGNALI DI TRADING GENERATI (Modello: {args.model})")
    print("-"*50)
    
    any_action = False
    for ticker, sig in signals.items():
        action = sig["action"]
        # La strategia non ritorna "weight" direttamente, ma usa un peso base di config modificato dal confidence_multiplier
        base_weight = config.BACKTEST_MAX_POSITION_SIZE
        weight = sig.get("weight", base_weight * sig.get("confidence_multiplier", 1.0))
        if action != "HOLD":
            any_action = True
            logger.info(f"🟢 [SEGNALE] Ticker: {ticker:<6} | Azione: {action:<10} | Peso Allocazione: {weight*100:.2f}%")
        else:
            logger.debug(f"⚪ [HOLD] Ticker: {ticker:<6} | Azione: HOLD")
            
    if not any_action:
        logger.info("Nessuna azione consigliata per la giornata odierna (Tutti i ticker sono HOLD).")
    print("-"*50 + "\n")

    # 7. Esecuzione reale o simulata (Dry-Run)
    mode_str = "REAL TRADE" if args.execute else "DRY RUN"
    telegram_lines = []
    
    if args.execute:
        logger.info("🚀 Modalità ESECUZIONE ATTIVA! Invio degli ordini di allineamento ad Alpaca...")
        executed_trades = execute_trades_on_alpaca(
            alpaca_client, 
            signals, 
            portfolio, 
            list(historical_data.keys()),
            max_slots=args.max_slots
        )
        if executed_trades:
            telegram_lines = executed_trades
        else:
            telegram_lines = ["Nessuna operazione eseguita (portafoglio già allineato)."]
        logger.info("Operatività di oggi completata con successo.")
    else:
        logger.info("ℹ️ Modalità DRY RUN (Simulazione). Nessun ordine inviato ad Alpaca.")
        logger.info("Usa il flag '--execute' per inviare realmente gli ordini in produzione.")
        
        # In simulazione, creiamo l'elenco delle operazioni raccomandate
        recommended_trades = []
        for ticker, sig in signals.items():
            action = sig["action"]
            if action != "HOLD":
                db = DBManager()
                last_price_df = db.execute_query("SELECT close FROM ohlcv WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1", (ticker,))
                last_close = float(last_price_df.iloc[0, 0]) if not last_price_df.empty else 0.0
                sl_pct = sig.get("stop_loss_pct", 0.0)
                tp_pct = sig.get("take_profit_pct", 0.0)
                sl_price = round(last_close * (1.0 - sl_pct), 2) if action == "BUY" else round(last_close * (1.0 + sl_pct), 2)
                tp_price = round(last_close * (1.0 + tp_pct), 2) if action == "BUY" else round(last_close * (1.0 - tp_pct), 2)
                
                recommended_trades.append(f"🤖 {action} {ticker} @ ${last_close:.2f} (SL: {sl_price}, TP: {tp_price})")
        
        if recommended_trades:
            telegram_lines = recommended_trades
        else:
            telegram_lines = ["Nessun segnale operativo generato (Tutti i ticker sono HOLD)."]

    # Invio del messaggio Telegram
    telegram_msg = f"<b>[{mode_str}] Report Operazioni {args.model.upper()}</b>\n"
    telegram_msg += f"Data: {current_date.strftime('%Y-%m-%d')}\n\n"
    telegram_msg += "\n".join(telegram_lines)
    
    send_telegram_message(telegram_msg)


if __name__ == "__main__":
    main()
