"""
base_strategy.py — Classe base per tutte le strategie di trading.

Fornisce il template comune (setup DB, macro loading, report, cleanup)
e lascia alle sottoclassi solo la logica specifica di ogni strategia.
"""

import os
import datetime
import warnings
import logging

from abc import ABC, abstractmethod

from core.config import DB_MARKET, MACRO_MAP
from core.data.data_manager import DataManager
from core.utils.notifier import TelegramNotifier

# Silenzia TensorFlow (ereditato da tutte le strategie)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.FATAL)


class BaseStrategy(ABC):
    """
    Template Method Pattern per strategie di trading.
    
    Ogni sottoclasse deve definire:
        - bot_name:       str — Nome del bot per il report Telegram
        - db_trades_path: str — Percorso al DB operativo della strategia
        - model_path:     str — Percorso al file .h5 del modello
        - use_macro:      bool — Se caricare i dati macroeconomici (default: True)
    
    E implementare:
        - setup_trades_db()   — Crea le tabelle nel DB operativo
        - execute()           — Logica principale della strategia
    
    Opzionalmente sovrascrivere:
        - build_report()      — Per personalizzare il report Telegram
    """

    bot_name: str = ""
    db_trades_path: str = ""
    db_market_path: str = DB_MARKET
    model_path: str = ""
    use_macro: bool = True

    def __init__(self):
        self.conn_market = None
        self.conn_trades = None
        self.macro = {}
        self._report_data = {}

    # ── Template Method ───────────────────────────────────

    def run(self):
        """Entry point principale. Gestisce il ciclo di vita completo."""
        print(f"🚀 AVVIO {self.bot_name} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # 1. Setup connessioni
        self.conn_market = DataManager.setup_db(self.db_market_path)
        if self.db_trades_path:
            self.conn_trades = DataManager.setup_db(self.db_trades_path)
            self.setup_trades_db()

        # 2. Carica dati macro (se richiesto)
        if self.use_macro:
            self.load_macro()

        # 3. Verifica modello
        if self.model_path and not os.path.exists(self.model_path):
            print(f"❌ ERRORE: Modello {self.model_path} non trovato!")
            self._cleanup()
            return

        # 4. Esecuzione strategia
        try:
            self.execute()
        except Exception as e:
            print(f"❌ ERRORE CRITICO in {self.bot_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

        # 5. Invio report
        self.send_report()
        print(f"✅ {self.bot_name} Completata.")

    # ── Metodi astratti (da implementare nelle sottoclassi) ──

    @abstractmethod
    def setup_trades_db(self):
        """Crea le tabelle necessarie nel DB operativo."""
        ...

    @abstractmethod
    def execute(self):
        """Logica principale della strategia (model loading, ticker loop, trading)."""
        ...

    # ── Metodi helper (condivisi) ─────────────────────────

    def load_macro(self):
        """Carica i dati macroeconomici dal DB di mercato."""
        self.macro = {
            label: DataManager.get_cached_market_data(m, self.conn_market)[['Date', 'Close']].rename(columns={'Close': label})
            for m, label in MACRO_MAP.items()
        }

    def set_report_data(self, **kwargs):
        """Imposta i dati per il report Telegram.
        
        Args:
            balance_str: Stringa del bilancio (es. "$10,000.00")
            trades_str:  Stringa delle operazioni fatte
            win_rate_str: Stringa del win rate
            extra_str:   Info extra (es. esposizione mercato)
            logs_str:    Log aggiuntivi
        """
        self._report_data.update(kwargs)

    def send_report(self):
        """Costruisce e invia il report Telegram."""
        msg_html = TelegramNotifier.build_report(
            bot_name=self.bot_name,
            **self._report_data
        )
        TelegramNotifier.send_message(msg_html)

    def get_market_data(self, ticker):
        """Shortcut per ottenere dati di mercato dal DB condiviso."""
        return DataManager.get_cached_market_data(ticker, self.conn_market)

    def _cleanup(self):
        """Chiude le connessioni ai database."""
        if self.conn_market:
            try: self.conn_market.close()
            except: pass
        if self.conn_trades:
            try: self.conn_trades.close()
            except: pass
