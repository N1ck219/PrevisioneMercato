import logging
import json
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import config
from database.db_manager import DBManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BacktestEngine")


class Position:
    """
    Rappresenta una posizione finanziaria aperta nel portafoglio.
    """
    def __init__(
        self,
        ticker: str,
        shares: float,
        entry_price: float,
        entry_date: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_type: str = "LONG",
        trailing_stop_pct: Optional[float] = None
    ) -> None:
        self.ticker = ticker
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_type = position_type
        self.trailing_stop_pct = trailing_stop_pct
        
        self.current_price = entry_price
        self.highest_price = entry_price if position_type == "LONG" else None
        self.lowest_price = entry_price if position_type == "SHORT" else None
        self.unrealized_pnl = 0.0

    def update_price(self, price: float) -> None:
        """Aggiorna il prezzo corrente e calcola il PnL non realizzato ed eventuale trailing stop."""
        self.current_price = price
        if self.position_type == "LONG":
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.shares
            if self.highest_price is not None:
                self.highest_price = max(self.highest_price, price)
                if self.trailing_stop_pct is not None:
                    new_sl = self.highest_price * (1.0 - self.trailing_stop_pct)
                    if self.stop_loss is None or new_sl > self.stop_loss:
                        self.stop_loss = new_sl
        else:
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.shares
            if self.lowest_price is not None:
                self.lowest_price = min(self.lowest_price, price)
                if self.trailing_stop_pct is not None:
                    new_sl = self.lowest_price * (1.0 + self.trailing_stop_pct)
                    if self.stop_loss is None or new_sl < self.stop_loss:
                        self.stop_loss = new_sl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date.strftime("%Y-%m-%d"),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_type": self.position_type,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl
        }


class Portfolio:
    """
    Gestisce la liquidità (con sub-balances separati per ticker), le posizioni aperte,
    le commissioni di trading e registra lo storico delle transazioni.
    """
    def __init__(
        self, 
        initial_capital: float = config.BACKTEST_CAPITALE_INIZIALE,
        tickers: Optional[List[str]] = None,
        commission_rate: float = getattr(config, "BACKTEST_COMMISSION_RATE", 0.001)
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.positions: Dict[str, Position] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
        # Inizializzazione sub-balances separati per evitare contaminazioni di budget
        self.tickers = tickers if tickers else []
        if self.tickers:
            sub_capital = initial_capital / len(self.tickers)
            self.sub_balances = {t: sub_capital for t in self.tickers}
        else:
            self.sub_balances = {"SHARED": initial_capital}

    @property
    def cash(self) -> float:
        """La liquidità totale del portafoglio (somma di tutti i sub-balances)."""
        return sum(self.sub_balances.values())

    @property
    def total_value(self) -> float:
        """Valore totale del portafoglio (Liquidità + Valore di mercato delle posizioni)."""
        positions_value = sum(pos.shares * pos.current_price for pos in self.positions.values())
        return self.cash + positions_value

    def open_position(
        self,
        ticker: str,
        shares: float,
        price: float,
        date: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_type: str = "LONG",
        trailing_stop_pct: Optional[float] = None
    ) -> bool:
        """Apre una nuova posizione usando il budget specifico del ticker e applicando le commissioni."""
        cost_clean = shares * price
        commission = cost_clean * self.commission_rate
        cost_total = cost_clean + commission
        
        # Recuperiamo il sub-balance del ticker (o SHARED)
        bal_key = ticker if ticker in self.sub_balances else "SHARED"
        available_cash = self.sub_balances[bal_key]
        
        if cost_total > available_cash:
            # Adeguamento quote dinamico: se i soldi sono insufficienti, investiamo tutta la cassa a disposizione (soglia minima $0.10)
            if available_cash > 0.10:
                shares = (available_cash / (price * (1 + self.commission_rate))) * 0.999
                cost_clean = shares * price
                commission = cost_clean * self.commission_rate
                cost_total = cost_clean + commission
                logger.info(
                    f"Budget insufficiente per size piena su {ticker}. "
                    f"Ridotta size a {shares:.6f} azioni per investire tutta la cassa disponibile (${available_cash:.2f})"
                )
            else:
                logger.warning(
                    f"Budget insufficiente per aprire posizione su {ticker}. "
                    f"Richiesto (con comm.): ${cost_total:.2f}, Budget Disponibile: ${available_cash:.2f}"
                )
                return False
        
        self.sub_balances[bal_key] -= cost_total
        
        if ticker in self.positions:
            # Media del prezzo di carico
            existing = self.positions[ticker]
            new_shares = existing.shares + shares
            new_entry_price = ((existing.entry_price * existing.shares) + (price * shares)) / new_shares
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=new_shares,
                entry_price=new_entry_price,
                entry_date=date,
                stop_loss=stop_loss or existing.stop_loss,
                take_profit=take_profit or existing.take_profit,
                position_type=position_type,
                trailing_stop_pct=trailing_stop_pct or existing.trailing_stop_pct
            )
        else:
            self.positions[ticker] = Position(ticker, shares, price, date, stop_loss, take_profit, position_type, trailing_stop_pct)
            
        action_type = "ACQUISTO (SHORT)" if position_type == "SHORT" else "ACQUISTO"
        logger.info(f"[{date.strftime('%Y-%m-%d')}] {action_type}: {shares:.2f} quote di {ticker} a ${price:.2f}. Comm: ${commission:.2f}")
        return True

    def close_position(self, ticker: str, price: float, date: datetime, reason: str = "SIGNAL") -> None:
        """Chiude completamente una posizione registrando la transazione e aggiungendo il ricavato al sub-balance (al netto delle commissioni)."""
        if ticker not in self.positions:
            return
        
        pos = self.positions.pop(ticker)
        revenue_clean = pos.shares * price
        commission = revenue_clean * self.commission_rate
        
        if pos.position_type == "LONG":
            revenue_total = revenue_clean - commission
            pnl = revenue_total - (pos.shares * pos.entry_price)
            pnl_pct = (price - pos.entry_price) / pos.entry_price
        else:
            # Per lo SHORT, buy back: la liquidità restituita è il collaterale entry + il profitto (entry - exit)
            revenue_total = (2.0 * pos.entry_price - price) * pos.shares - commission
            pnl = revenue_total - (pos.shares * pos.entry_price)
            pnl_pct = (pos.entry_price - price) / pos.entry_price
        
        # Accreditiamo al sub-balance specifico
        bal_key = ticker if ticker in self.sub_balances else "SHARED"
        self.sub_balances[bal_key] += revenue_total
        
        transaction = {
            "ticker": ticker,
            "shares": pos.shares,
            "entry_date": pos.entry_date.strftime("%Y-%m-%d"),
            "entry_price": pos.entry_price,
            "exit_date": date.strftime("%Y-%m-%d"),
            "exit_price": price,
            "position_type": pos.position_type,
            "pnl_usd": pnl,
            "pnl_pct": pnl_pct,
            "commission_usd": commission,
            "reason": reason
        }
        self.transactions.append(transaction)
        logger.info(
            f"[{date.strftime('%Y-%m-%d')}] VENDITA ({reason}): {pos.shares:.2f} quote di {ticker} a ${price:.2f}. "
            f"Comm: ${commission:.2f}. PnL Netto: ${pnl:.2f} ({pnl_pct*100:.2f}%)"
        )

    def update_valuations(self, prices: Dict[str, float]) -> None:
        """Aggiorna il valore di mercato corrente per tutte le posizioni aperte."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].update_price(price)

    def record_equity(self, date: datetime) -> None:
        """Registra lo stato giornaliero del portafoglio."""
        self.equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "cash": self.cash,
            "positions_value": sum(pos.shares * pos.current_price for pos in self.positions.values()),
            "total_value": self.total_value
        })


class BacktestEngine:
    """
    Motore di backtesting Event-Driven.
    Simula il mercato giorno per giorno prevenendo lookahead bias.
    """
    def __init__(self, start_date: str, end_date: str, tickers: List[str] = config.TICKERS, max_slots: int = 20) -> None:
        self.db = DBManager()
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.max_slots = max_slots
        
        self.portfolio = Portfolio(tickers=self.tickers)
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.all_trading_dates: List[pd.Timestamp] = []
        
        self._load_data()

    def _load_data(self) -> None:
        """Carica i dati di mercato (prezzi ed indicatori uniti) con un periodo di warm-up padding."""
        logger.info("Caricamento dati storici e indicatori tecnici dal database SQLite con warm-up period...")
        
        # Calcoliamo una data di inizio con padding temporale (circa 365 giorni solari prima)
        # per consentire agli indicatori calcolati a runtime (es. SMA200) di essere subito disponibili all'inizio del backtest.
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        padding_dt = start_dt - timedelta(days=365)
        padding_date_str = padding_dt.strftime("%Y-%m-%d")
        
        all_dates_set = set()
        
        for ticker in self.tickers:
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
                WHERE o.ticker = ? AND o.timestamp >= ? AND o.timestamp <= ?
                ORDER BY o.timestamp ASC
            """
            df = self.db.execute_query(query, (ticker, padding_date_str, self.end_date))
            
            if df.empty:
                logger.warning(f"Nessun dato trovato nel database per {ticker} nel periodo padded {padding_date_str} - {self.end_date}")
                continue
                
            # Assicuriamoci che il timestamp sia un oggetto DateTimeIndex
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').set_index('timestamp')
            
            self.historical_data[ticker] = df
            
            # Popoliamo all_dates_set solo con le date che ricadono nel periodo effettivo di backtest (>= start_date)
            df_backtest = df.loc[df.index >= pd.to_datetime(self.start_date)]
            all_dates_set.update(df_backtest.index.tolist())
            
        self.all_trading_dates = sorted(list(all_dates_set))
        logger.info(f"Dati caricati con successo (con warm-up). Ticker attivi: {len(self.historical_data)}. Giorni effettivi di simulazione: {len(self.all_trading_dates)}")

    def run(self, strategy_class, split_equally: bool = False) -> Dict[str, Any]:
        """
        Esegue il backtest giorno per giorno.
        Accetta una classe Strategy (o istanza) per la logica decisionale.
        """
        strategy_name = strategy_class.__name__ if hasattr(strategy_class, "__name__") else "CustomStrategy"
        if strategy_name == "<lambda>":
            try:
                temp_strat = strategy_class()
                strategy_name = temp_strat.__class__.__name__
            except Exception:
                strategy_name = "NeuralNetworkStrategy"
                
        # Rimuoviamo caratteri non validi per i nomi delle cartelle su Windows/UNIX
        for char in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
            strategy_name = strategy_name.replace(char, "")
            
        logger.info(f"Inizio simulazione di backtest per la strategia '{strategy_name}' (Split Equamente: {split_equally})...")
        
        # Inizializziamo la strategia con i parametri desiderati
        strategy = strategy_class()
        
        # Raccogliamo metadati della strategia per i grafici
        self.strategy_info = {
            "name": strategy_name,
            "threshold": getattr(strategy, "probability_threshold", None),
            "split_equally": split_equally
        }
        
        # Array temporaneo per gestire gli ordini pendenti da eseguire sul prezzo OPEN del giorno successivo
        pending_orders: List[Dict[str, Any]] = []
        
        # Silenziamo temporaneamente i log operativi individuali per mostrare la barra di avanzamento nel terminale
        old_level = logger.level
        logger.setLevel(logging.WARNING)

        total_dates = len(self.all_trading_dates)
        
        # Gestione tqdm o barra di progressione custom
        if HAS_TQDM:
            iterator = tqdm(
                self.all_trading_dates, 
                desc=f"[Backtest {strategy_name}]",
                bar_format="{l_bar}{bar:30}{r_bar}"
            )
        else:
            iterator = self.all_trading_dates
            
        for idx, current_date in enumerate(iterator):
            current_date_pydt = current_date.to_pydatetime()
            
            # Se tqdm non è attivo, stampiamo la nostra barra custom
            if not HAS_TQDM:
                percent = int((idx + 1) / total_dates * 100)
                bar_length = 30
                filled_length = int(bar_length * (idx + 1) // total_dates)
                bar = "#" * filled_length + "-" * (bar_length - filled_length)
                sys.stdout.write(f"\r[Backtest] Avanzamento: |{bar}| {percent}% ({idx + 1}/{total_dates} giorni)")
                sys.stdout.flush()
            
            # --- FASE 1: Esecuzione Ordini Pendenti (sul prezzo OPEN di oggi) ---
            # Le decisioni prese al giorno T vengono eseguite all'OPEN di T+1
            todays_opens: Dict[str, float] = {}
            for ticker in self.tickers:
                if ticker in self.historical_data and current_date in self.historical_data[ticker].index:
                    todays_opens[ticker] = self.historical_data[ticker].at[current_date, 'open']

            # Eseguiamo gli ordini inseriti ieri
            for order in pending_orders:
                ticker = order["ticker"]
                action = order["action"]
                
                if ticker not in todays_opens:
                    continue  # Nessun dato disponibile oggi per questo ticker, salta
                
                execution_price = todays_opens[ticker]
                
                if action == "BUY":
                    allocated_cash = order["cash"]
                    shares = allocated_cash / execution_price
                    sl_pct = order.get("stop_loss_pct")
                    tp_pct = order.get("take_profit_pct")
                    sl_level = execution_price * (1.0 - sl_pct) if sl_pct is not None else None
                    tp_level = execution_price * (1.0 + tp_pct) if tp_pct is not None else None
                    
                    self.portfolio.open_position(
                        ticker=ticker,
                        shares=shares,
                        price=execution_price,
                        date=current_date_pydt,
                        stop_loss=sl_level,
                        take_profit=tp_level,
                        position_type="LONG",
                        trailing_stop_pct=order.get("trailing_stop_pct")
                    )
                elif action == "SELL_SHORT":
                    allocated_cash = order["cash"]
                    shares = allocated_cash / execution_price
                    sl_pct = order.get("stop_loss_pct")
                    tp_pct = order.get("take_profit_pct")
                    sl_level = execution_price * (1.0 + sl_pct) if sl_pct is not None else None
                    tp_level = execution_price * (1.0 - tp_pct) if tp_pct is not None else None
                    
                    self.portfolio.open_position(
                        ticker=ticker,
                        shares=shares,
                        price=execution_price,
                        date=current_date_pydt,
                        stop_loss=sl_level,
                        take_profit=tp_level,
                        position_type="SHORT",
                        trailing_stop_pct=order.get("trailing_stop_pct")
                    )
                elif action in ["SELL", "BUY_TO_COVER"]:
                    self.portfolio.close_position(
                        ticker=ticker,
                        price=execution_price,
                        date=current_date_pydt,
                        reason="SIGNAL"
                    )
            
            # Svuotiamo gli ordini eseguiti
            pending_orders.clear()

            # --- FASE 2: Aggiornamento Portafoglio & Controllo Risk Management (SL/TP) ---
            todays_prices: Dict[str, Dict[str, float]] = {}
            for ticker in self.tickers:
                if ticker in self.historical_data and current_date in self.historical_data[ticker].index:
                    df_t = self.historical_data[ticker]
                    todays_prices[ticker] = {
                        "open": df_t.at[current_date, "open"],
                        "high": df_t.at[current_date, "high"],
                        "low": df_t.at[current_date, "low"],
                        "close": df_t.at[current_date, "close"]
                    }

            # Aggiorniamo le valutazioni correnti delle posizioni aperte al Close di oggi
            closes = {t: prices["close"] for t, prices in todays_prices.items()}
            self.portfolio.update_valuations(closes)

            # Controllo se Stop Loss o Take Profit sono stati toccati durante la giornata corrente
            active_tickers = list(self.portfolio.positions.keys())
            for ticker in active_tickers:
                if ticker not in todays_prices:
                    continue
                
                pos = self.portfolio.positions[ticker]
                high_today = todays_prices[ticker]["high"]
                low_today = todays_prices[ticker]["low"]
                
                # Controllo Risk Management (SL/TP) per LONG vs SHORT
                if pos.position_type == "LONG":
                    # Controllo Stop Loss
                    if pos.stop_loss is not None and low_today <= pos.stop_loss:
                        execution_price = min(pos.stop_loss, todays_prices[ticker]["open"])
                        self.portfolio.close_position(ticker, execution_price, current_date_pydt, reason="STOP_LOSS")
                        continue
                    
                    # Controllo Take Profit
                    if pos.take_profit is not None and high_today >= pos.take_profit:
                        execution_price = max(pos.take_profit, todays_prices[ticker]["open"])
                        self.portfolio.close_position(ticker, execution_price, current_date_pydt, reason="TAKE_PROFIT")
                        continue
                else:
                    # Controllo Stop Loss per SHORT (prezzo sale)
                    if pos.stop_loss is not None and high_today >= pos.stop_loss:
                        execution_price = max(pos.stop_loss, todays_prices[ticker]["open"])
                        self.portfolio.close_position(ticker, execution_price, current_date_pydt, reason="STOP_LOSS")
                        continue
                    
                    # Controllo Take Profit per SHORT (prezzo scende)
                    if pos.take_profit is not None and low_today <= pos.take_profit:
                        execution_price = min(pos.take_profit, todays_prices[ticker]["open"])
                        self.portfolio.close_position(ticker, execution_price, current_date_pydt, reason="TAKE_PROFIT")
                        continue

            # Registriamo l'equity giornaliera di fine giornata
            self.portfolio.record_equity(current_date_pydt)

            # --- FASE 3: Generazione Segnali Strategia (alla fine del giorno T) ---
            context_data: Dict[str, pd.DataFrame] = {}
            for ticker, df in self.historical_data.items():
                pos = df.index.searchsorted(current_date, side='right')
                context_data[ticker] = df.iloc[:pos]

            signals = strategy.generate_signals(context_data, self.portfolio, current_date_pydt)
            
            if idx == len(self.all_trading_dates) - 1:
                break

            # Tracciamento dei bilanci virtuali per calcolare accuratamente la liquidità disponibile a fine giornata
            virtual_balances = self.portfolio.sub_balances.copy()

            # 1. Gestione Uscite Immediate (SELL per LONG, BUY_TO_COVER per SHORT)
            exiting_tickers = set()
            for ticker in list(self.portfolio.positions.keys()):
                pos = self.portfolio.positions[ticker]
                sig_info = signals.get(ticker, {})
                action = sig_info.get("action")
                
                bal_key = ticker if ticker in virtual_balances else "SHARED"
                est_price = todays_prices[ticker]["close"] if ticker in todays_prices else pos.current_price
                
                if pos.position_type == "LONG" and action == "SELL":
                    pending_orders.append({
                        "ticker": ticker,
                        "action": "SELL"
                    })
                    exiting_tickers.add(ticker)
                    
                    # Rilascio virtuale di liquidità derivante dalla vendita
                    revenue_clean = pos.shares * est_price
                    commission = revenue_clean * self.portfolio.commission_rate
                    revenue_total = revenue_clean - commission
                    virtual_balances[bal_key] += revenue_total
                    
                elif pos.position_type == "SHORT" and action == "BUY_TO_COVER":
                    pending_orders.append({
                        "ticker": ticker,
                        "action": "BUY_TO_COVER"
                    })
                    exiting_tickers.add(ticker)
                    
                    # Rilascio virtuale di liquidità derivante dal buy back dello short
                    revenue_clean = (2.0 * pos.entry_price - est_price) * pos.shares
                    commission = pos.shares * est_price * self.portfolio.commission_rate
                    revenue_total = revenue_clean - commission
                    virtual_balances[bal_key] += revenue_total

            # 2. Calcolo Slot Rimasti
            max_slots = getattr(strategy, "current_max_slots", self.max_slots)
            future_active_count = len(self.portfolio.positions) - len(exiting_tickers)
            slots_available = max(0, max_slots - future_active_count)

            # 3. Filtro e Ordinamento Ingressi per Confidenza
            entries = []
            for ticker, sig_info in signals.items():
                if ticker in self.portfolio.positions:
                    continue
                action = sig_info.get("action")
                prob = sig_info.get("probability")
                
                if prob is None:
                    prob = 0.6 if action in ["BUY", "SELL_SHORT"] else 0.5
                
                threshold = getattr(strategy, "probability_threshold", 0.55)
                
                if action == "BUY" and prob >= threshold:
                    confidence = prob - threshold
                    entries.append({
                        "ticker": ticker,
                        "action": "BUY",
                        "confidence": confidence,
                        "sig_info": sig_info
                    })
                elif action == "SELL_SHORT" and prob <= (1.0 - threshold):
                    confidence = (1.0 - threshold) - prob
                    entries.append({
                        "ticker": ticker,
                        "action": "SELL_SHORT",
                        "confidence": confidence,
                        "sig_info": sig_info
                    })

            entries = sorted(entries, key=lambda x: x["confidence"], reverse=True)
            selected_entries = entries[:slots_available]

            # 4. Generazione Ordini di Ingresso con Budget di Slot e Margine di Sicurezza
            for entry in selected_entries:
                ticker = entry["ticker"]
                action = entry["action"]
                sig_info = entry["sig_info"]
                
                # Calcoliamo il budget teorico dello slot applicando un fattore di utilizzo (es. 95%)
                # e un moltiplicatore di confidenza dinamico (Kelly Sizing) se definito nella strategia.
                slot_ratio = 0.95
                conf_mult = sig_info.get("confidence_multiplier", 1.0)
                
                # Supporto a riserva di cash dinamica della strategia
                cash_reserve_pct = getattr(strategy, "current_cash_reserve_pct", 0.0)
                usable_capital = self.portfolio.total_value * (1.0 - cash_reserve_pct)
                budget = (usable_capital / max_slots) * slot_ratio * conf_mult
                
                bal_key = ticker if ticker in virtual_balances else "SHARED"
                # Teniamo un margine del 2% (0.98) sulla liquidità virtuale reale disponibile
                # (decurtata della riserva di cash se presente) per assorbire gap di prezzo overnight ed evitare warning di cassa insufficiente.
                raw_available = virtual_balances[bal_key] - (self.portfolio.total_value * cash_reserve_pct)
                available_cash = max(0.0, raw_available) * 0.98
                budget = min(budget, available_cash)
                
                buy_cash = budget / (1 + self.portfolio.commission_rate)
                if buy_cash < 10.0:
                    continue

                sl_pct = sig_info.get("stop_loss_pct", config.BACKTEST_STOP_LOSS)
                tp_pct = sig_info.get("take_profit_pct", config.BACKTEST_TAKE_PROFIT)
                trailing_stop_pct = sig_info.get("trailing_stop_pct")

                pending_orders.append({
                    "ticker": ticker,
                    "action": action,
                    "cash": buy_cash,
                    "stop_loss_pct": sl_pct,
                    "take_profit_pct": tp_pct,
                    "trailing_stop_pct": trailing_stop_pct
                })
                
                # Detrazione virtuale della liquidità prenotata per l'ingresso
                virtual_balances[bal_key] -= budget

        # Ripristiniamo il livello di logging originario
        logger.setLevel(old_level)
        
        # Se non avevamo tqdm, stampiamo una nuova riga per andare a capo
        if not HAS_TQDM:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # --- FASE 4: Calcolo delle Metriche di Performance Finali ---
        report = self._calculate_performance_metrics()
        
        # Calcolo dell'equity curve del benchmark Buy & Hold
        bh_equity = self._calculate_buy_and_hold_equity()
        
        # Salvataggio report e curve in una sottocartella dedicata alla run
        run_dir = self._save_report(report, strategy_name)
        
        # Generazione del grafico di confronto
        self._generate_plots(report, run_dir, bh_equity)
        
        # Generazione del grafico di performance individuale dei ticker vs S&H
        self._generate_ticker_plots(report, run_dir)
        
        report["run_dir"] = str(run_dir)
        
        return report

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Elabora l'equity curve e i trade chiusi per estrarre le metriche di performance."""
        eq_df = pd.DataFrame(self.portfolio.equity_curve)
        
        if eq_df.empty:
            return {"error": "Nessuna data registrata durante il backtest."}
            
        initial_val = self.portfolio.initial_capital
        final_val = self.portfolio.total_value
        
        # Ritorno cumulativo
        total_return = (final_val - initial_val) / initial_val
        
        # Calcolo dei Drawdown
        eq_df['peak'] = eq_df['total_value'].cummax()
        eq_df['drawdown'] = (eq_df['total_value'] - eq_df['peak']) / eq_df['peak']
        max_drawdown = float(eq_df['drawdown'].min())
        
        # Rendimenti giornalieri per Sharpe Ratio
        eq_df['daily_return'] = eq_df['total_value'].pct_change()
        daily_returns = eq_df['daily_return'].dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            # Sharpe Ratio annualizzato (252 giorni finanziari)
            sharpe_ratio = float(np.sqrt(252) * daily_returns.mean() / daily_returns.std())
        else:
            sharpe_ratio = 0.0
            
        # Analisi dei Trade Chiusi
        closed_trades = self.portfolio.transactions
        total_trades = len(closed_trades)
        
        winning_trades = [t for t in closed_trades if t["pnl_usd"] > 0]
        losing_trades = [t for t in closed_trades if t["pnl_usd"] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(t["pnl_usd"] for t in closed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        
        # Disaggregazione per ticker
        ticker_performance: Dict[str, Dict[str, Any]] = {}
        for ticker in self.tickers:
            t_trades = [t for t in closed_trades if t["ticker"] == ticker]
            t_total = len(t_trades)
            if t_total > 0:
                t_wins = [t for t in t_trades if t["pnl_usd"] > 0]
                t_pnl = sum(t["pnl_usd"] for t in t_trades)
                ticker_performance[ticker] = {
                    "total_trades": t_total,
                    "win_rate": len(t_wins) / t_total,
                    "net_pnl": t_pnl
                }

        report = {
            "period": {
                "start": self.start_date,
                "end": self.end_date
            },
            "capital": {
                "initial": initial_val,
                "final": final_val,
                "net_profit": final_val - initial_val
            },
            "metrics": {
                "total_return_pct": total_return * 100,
                "max_drawdown_pct": max_drawdown * 100,
                "sharpe_ratio": sharpe_ratio,
                "win_rate_pct": win_rate * 100
            },
            "trades": {
                "total": total_trades,
                "wins": len(winning_trades),
                "losses": len(losing_trades),
                "avg_pnl_usd": avg_pnl,
                "log": closed_trades
            },
            "ticker_breakdown": ticker_performance,
            "equity_curve": self.portfolio.equity_curve
        }
        
        return report

    def _save_report(self, report: Dict[str, Any], strategy_name: str) -> Path:
        """Salva il report del backtest in formato JSON e CSV per l'equity curve in una sottocartella dedicata."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determina dinamicamente la sottocartella del modello in base al nome della strategia
        model_subfolder = "moe"
        strat_lower = strategy_name.lower()
        if "moe" in strat_lower:
            model_subfolder = "moe"
        elif "neuralnetwork" in strat_lower or "nn" in strat_lower:
            model_subfolder = "neural_network"
        elif "transformer" in strat_lower:
            model_subfolder = "transformer"
        elif "sma" in strat_lower:
            model_subfolder = "sma"
            
        run_dir = config.BASE_DIR / "risultati_backtest" / model_subfolder / f"run_{strategy_name}_{timestamp_str}"
        run_dir.mkdir(exist_ok=True, parents=True)
        
        # Salva JSON completo
        json_path = run_dir / "backtest_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Report completo di backtest salvato in: {json_path}")
        
        # Salva CSV dell'equity curve per grafici rapidi (solo se non ci sono errori)
        if "error" in report:
            logger.warning(f"Salvataggio CSV dell'equity curve saltato: {report['error']}")
            return run_dir

        eq_df = pd.DataFrame(report["equity_curve"])
        if not eq_df.empty:
            csv_path = run_dir / "equity_curve.csv"
            eq_df.to_csv(csv_path, index=False)
            logger.info(f"Equity curve salvata in CSV: {csv_path}")
            
        return run_dir

    def _calculate_buy_and_hold_equity(self) -> List[float]:
        """
        Calcola l'equity curve di una strategia Buy & Hold di riferimento.
        Divide il capitale iniziale equamente all'inizio del backtest e ne traccia il valore nel tempo.
        """
        initial_capital = self.portfolio.initial_capital
        num_tickers = len(self.historical_data)
        if num_tickers == 0:
            return [initial_capital] * len(self.all_trading_dates)
            
        capital_per_ticker = initial_capital / num_tickers
        shares: Dict[str, float] = {}
        
        # Determiniamo le quote acquistate all'inizio del periodo effettivo di backtest (non del padding)
        first_backtest_date = self.all_trading_dates[0] if self.all_trading_dates else None
        for ticker, df in self.historical_data.items():
            if not df.empty and first_backtest_date is not None:
                # Cerchiamo il prezzo open alla prima data di backtest (o la più vicina successiva)
                df_from_start = df[df.index >= first_backtest_date]
                if not df_from_start.empty:
                    first_price = df_from_start['open'].iloc[0]
                else:
                    first_price = df['open'].iloc[-1]  # fallback all'ultimo prezzo disponibile
                if first_price > 0:
                    shares[ticker] = capital_per_ticker / first_price
                else:
                    shares[ticker] = 0.0
            else:
                shares[ticker] = 0.0
                
        bh_equity = []
        for current_date in self.all_trading_dates:
            value = 0.0
            for ticker, df in self.historical_data.items():
                if current_date in df.index:
                    price = df.loc[current_date, 'close']
                    value += shares[ticker] * price
                else:
                    # Fallback all'ultimo prezzo disponibile se manca
                    prev_df = df[df.index < current_date]
                    if not prev_df.empty:
                        price = prev_df['close'].iloc[-1]
                        value += shares[ticker] * price
                    else:
                        value += capital_per_ticker
            bh_equity.append(value)
            
        return bh_equity

    def _generate_plots(self, report: Dict[str, Any], run_dir: Path, bh_equity: List[float]) -> None:
        """
        Genera un grafico di confronto tra la strategia di trading e il benchmark Buy & Hold.
        Aggiunge un box informativo con le metriche principali salvando l'immagine in PNG.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            eq_curve = [x["total_value"] for x in report["equity_curve"]]
            dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in report["equity_curve"]]
            
            # Usiamo uno stile grafico pulito
            plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
            
            # Calcolo dei rendimenti percentuali cumulativi
            initial_val = report["capital"]["initial"]
            strat_pct = [(v - initial_val) / initial_val * 100 for v in eq_curve]
            bh_pct = [(v - initial_val) / initial_val * 100 for v in bh_equity]
            
            # Disegno delle curve
            ax.plot(dates, strat_pct, label=f"Strategia ({report['metrics']['total_return_pct']:.2f}%)", color="#1f77b4", linewidth=2.5)
            ax.plot(dates, bh_pct, label=f"Buy & Hold ({bh_pct[-1]:.2f}%)", color="#d62728", linewidth=2.0, linestyle="--")
            
            # Formattazione
            threshold_val = self.strategy_info.get("threshold", "N/D")
            split_str = "Budget: Separato" if self.strategy_info.get("split_equally") else "Budget: Condiviso (Global Pool)"
            ax.set_title(f"Confronto Performance: Strategia vs Buy & Hold\n({split_str} | Soglia Confidenza: {threshold_val})", fontsize=13, fontweight="bold", pad=15)
            ax.set_xlabel("Data", fontsize=12, labelpad=10)
            ax.set_ylabel("Ritorno Cumulativo (%)", fontsize=12, labelpad=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor="white", edgecolor="gray")
            
            # Calcolo Drawdown per Buy & Hold
            bh_series = pd.Series(bh_equity)
            bh_peak = bh_series.cummax()
            bh_dd = (bh_series - bh_peak) / bh_peak * 100
            max_bh_dd = bh_dd.min()
            
            # Testo metriche chiave
            metrics_text = (
                f"=== METADATI RUN ===\n"
                f"Strategia:         {self.strategy_info.get('name', 'N/D')}\n"
                f"Config. Budget:    {split_str}\n"
                f"Soglia Confidenza: {threshold_val}\n\n"
                f"=== STRATEGIA ===\n"
                f"Ritorno Netto:     ${report['capital']['net_profit']:,.2f} ({report['metrics']['total_return_pct']:.2f}%)\n"
                f"Max Drawdown:      {report['metrics']['max_drawdown_pct']:.2f}%\n"
                f"Sharpe Ratio:      {report['metrics']['sharpe_ratio']:.2f}\n\n"
                f"=== OPERATIVITA' ===\n"
                f"Operazioni Totali: {report['trades']['total']}\n"
                f"Operazioni Vincenti: {report['trades']['wins']}\n"
                f"Operazioni Perdenti: {report['trades']['losses']}\n"
                f"Win Rate:          {report['metrics']['win_rate_pct']:.2f}%\n"
                f"PnL Medio/Trade:   ${report['trades']['avg_pnl_usd']:,.2f}\n\n"
                f"=== BENCHMARK B&H ===\n"
                f"Ritorno B&H:       {bh_pct[-1]:.2f}%\n"
                f"Max Drawdown B&H:  {max_bh_dd:.2f}%"
            )
            
            props = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9)
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left', bbox=props, family='monospace')
            
            fig.tight_layout()
            
            # Salvataggio
            plot_path = run_dir / "performance_comparison.png"
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info(f"Grafico di confronto performance salvato in: {plot_path}")
            
        except Exception as e:
            logger.error(f"Errore durante la generazione del grafico di confronto: {e}")

    def _generate_ticker_plots(self, report: Dict[str, Any], run_dir: Path) -> None:
        """
        Genera un grafico a barre che mostra per ciascun ticker il profitto/perdita netta
        della strategia in confronto al benchmark Buy & Hold su quel ticker.
        Aggiunge indicazione della configurazione usata nel titolo.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Calcolo dei rendimenti e profitti per ciascun ticker
            tickers_used = []
            strategy_pnls = []
            bh_pnls = []
            
            initial_capital = self.portfolio.initial_capital
            num_tickers = len(self.historical_data)
            if num_tickers == 0:
                return
                
            capital_per_ticker = initial_capital / num_tickers
            
            for ticker in self.tickers:
                if ticker not in self.historical_data or self.historical_data[ticker].empty:
                    continue
                    
                df = self.historical_data[ticker]
                first_price = df['open'].iloc[0]
                last_price = df['close'].iloc[-1]
                
                # Buy & Hold Profit per questo ticker
                bh_profit = capital_per_ticker * ((last_price - first_price) / first_price)
                
                # Strategia Profit per questo ticker (se ha fatto operazioni, altrimenti 0)
                strat_info = report["ticker_breakdown"].get(ticker, {})
                strat_profit = strat_info.get("net_pnl", 0.0)
                
                tickers_used.append(ticker)
                strategy_pnls.append(strat_profit)
                bh_pnls.append(bh_profit)
                
            if not tickers_used:
                return
                
            # Disegniamo il grafico a barre accostate
            plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
            
            x = np.arange(len(tickers_used))
            width = 0.35
            
            rects1 = ax.bar(x - width/2, strategy_pnls, width, label=f"Strategia ({self.strategy_info.get('name', 'Strategia')})", color='#1f77b4')
            rects2 = ax.bar(x + width/2, bh_pnls, width, label='Buy & Hold (Benchmark)', color='#d62728', alpha=0.7)
            
            ax.set_ylabel('Profitto / Perdita Netta ($)', fontsize=12)
            threshold_val = self.strategy_info.get("threshold", "N/D")
            split_str = "Budget: Separato" if self.strategy_info.get("split_equally") else "Budget: Condiviso"
            ax.set_title(f"Confronto Profitto/Perdita per Singolo Asset: {self.strategy_info.get('name', 'Strategia')} vs Buy & Hold\n({split_str} | Soglia: {threshold_val})", fontsize=13, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(tickers_used, rotation=45, fontsize=10)
            ax.legend(loc='best', fontsize=11, frameon=True, facecolor='white', edgecolor='gray')
            
            ax.axhline(0, color='black', linewidth=1.0, linestyle='-')
            ax.grid(True, linestyle=":", alpha=0.6)
            
            fig.tight_layout()
            
            plot_path = run_dir / "ticker_performance_comparison.png"
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info(f"Grafico delle performance dei ticker salvato in: {plot_path}")
            
        except Exception as e:
            logger.error(f"Errore durante la generazione del grafico ticker performance: {e}")
