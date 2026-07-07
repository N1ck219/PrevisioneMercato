import os
from pathlib import Path
from typing import List, Dict, Any

# Percorso base del progetto
BASE_DIR = Path(__file__).resolve().parent

# Configurazione del Database locale 
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "trading_platform.db"

# Variabili credenziali globali (con valori di default temporanei)
ALPACA_API_KEY: str = "YOUR_API_KEY"
ALPACA_SECRET_KEY: str = "YOUR_SECRET_KEY"
ALPACA_PAPER: bool = True

# Supporto per il caricamento dinamico di file .env per gestire più account Alpaca
def load_env(env_filename: str = ".env") -> None:
    """
    Carica le variabili d'ambiente da un file .env specifico.
    Cerca prima la variabile d'ambiente 'ENV_FILE' per caricare file alternativi,
    altrimenti utilizza il file specificato come parametro.
    """
    global ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER, ALPACA_BASE_URL
    
    selected_env = os.getenv("ENV_FILE", env_filename)
    env_path = BASE_DIR / selected_env
    
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Ignora righe vuote o commenti
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val
        
        # Aggiorna le variabili globali del modulo config
        ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", ALPACA_API_KEY)
        ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", ALPACA_SECRET_KEY)
        
        if "ALPACA_PAPER" in os.environ:
            ALPACA_PAPER = os.environ["ALPACA_PAPER"].lower() in ("true", "1", "yes")
            
        print(f"[INFO] Caricato ambiente Alpaca da: {selected_env} (Key ID: ...{ALPACA_API_KEY[-6:] if len(ALPACA_API_KEY) > 6 else 'N/D'})")
    else:
        # Fallback alle variabili d'ambiente di sistema
        ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", ALPACA_API_KEY)
        ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", ALPACA_SECRET_KEY)
        print(f"[WARNING] File ambiente '{selected_env}' non trovato. Fallback a variabili d'ambiente di sistema.")

# Caricamento iniziale dell'ambiente (default .env)
load_env()

# Creazione delle cartelle necessarie se non esistono (utilizzando 'models' come modificato dall'utente)
for folder in ["database", "risultati_backtest", "risultati_reali", "models"]:
    (BASE_DIR / folder).mkdir(exist_ok=True)

# Definizione URL in base all'ambiente caricato
ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"

# Selezione degli asset (ticker) da tracciare e negoziare
# Iniziamo con i principali ETF indici e alcune mega-cap dell'S&P 500
TICKERS: List[str] = [
    "SPY",   # S&P 500 ETF
    "QQQ",   # Nasdaq 100 ETF
    "DIA",   # Dow Jones ETF
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corp.
    "AMZN",  # Amazon.com Inc.
    "NVDA",  # NVIDIA Corp.
    "GOOGL"  # Alphabet Inc.
]

# Parametri di Trading globale e Backtesting
BACKTEST_CAPITALE_INIZIALE: float = 100000.0  # Capitale di partenza in USD
BACKTEST_COMMISSION_RATE: float = 0.0001      # Commissione dello 0.01% per ciascuna operazione (acquisto/vendita)
BACKTEST_STOP_LOSS: float = 0.02              # Stop Loss predefinito (2%)
BACKTEST_TAKE_PROFIT: float = 0.05            # Take Profit predefinito (5%)
BACKTEST_MAX_POSITION_SIZE: float = 0.10      # Dimensione massima della singola posizione (10% del portafoglio)

# Configurazione delle Feature e degli Indicatori Tecnici
INDICATORI_ATTIVI: Dict[str, Dict[str, Any]] = {
    "SMA": {"windows": [10, 20, 50, 200]},     # Simple Moving Averages
    "EMA": {"windows": [9, 21]},               # Exponential Moving Averages
    "RSI": {"window": 14},                     # Relative Strength Index
    "MACD": {"fast": 12, "slow": 26, "signal": 9}, # MACD
    "BBANDS": {"window": 20, "std_dev": 2},    # Bollinger Bands
    "ATR": {"window": 14}                      # Average True Range (misura volatilità)
}

# Parametri per il download storico
DATA_DOWNLOAD_ANNI: int = 15  # Anni di storico da scaricare al primo avvio
