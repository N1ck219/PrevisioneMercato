import os
import sys
import logging
import argparse
import requests
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# Silenzia avvisi deprecati o future warnings di PyTorch e altre librerie
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Assicuriamoci che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from backtest.engine import BacktestEngine, Portfolio
from backtest.strategy import SMAXStrategy, NeuralNetworkStrategy, NeuralNetworkV2Strategy, NeuralNetworkV3Strategy, NeuralNetworkV4Strategy, NeuralNetworkV5Strategy, NeuralNetworkV6Strategy, NeuralNetworkV7Strategy, NeuralNetworkV8Strategy, NeuralNetworkV9Strategy, NeuralNetworkV10Strategy, NeuralNetworkV11Strategy, MoEStrategyV1, NeuralNetworkGNNStrategy
from database.db_manager import DBManager

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UnifiedBacktest")

# Pool predefinito di 20 mega-cap dell'S&P 500 per le simulazioni di base
DEFAULT_20_POOL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", 
    "META", "BRK-B", "LLY", "AVGO", "JPM", 
    "TSLA", "XOM", "UNH", "JNJ", "V", 
    "PG", "HD", "MA", "ABBV", "CVX"
]


def get_alpaca_account_capital() -> float:
    """
    Interroga le API di Alpaca (paper o live in base al config) per recuperare
    il saldo dell'equity reale del conto. In caso di errore o credenziali mancanti,
    ripiega sul capitale di fallback configurato a livello globale.
    """
    logger.info("Verifica del capitale disponibile su Alpaca Account...")
    
    # Controlla se le credenziali sono quelle di default
    if (config.ALPACA_API_KEY == "YOUR_API_KEY" or 
        config.ALPACA_SECRET_KEY == "YOUR_SECRET_KEY" or 
        not config.ALPACA_API_KEY or 
        not config.ALPACA_SECRET_KEY):
        logger.warning("Credenziali Alpaca non configurate in .env. Utilizzo capitale di fallback.")
        return config.BACKTEST_CAPITALE_INIZIALE

    url = f"{config.ALPACA_BASE_URL}/v2/account"
    headers = {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Utilizziamo l'equity totale (valore portafoglio) se disponibile, altrimenti il cash liquido
            equity = float(data.get("equity", data.get("cash", config.BACKTEST_CAPITALE_INIZIALE)))
            logger.info(f"Saldo reale Alpaca rilevato con successo: ${equity:,.2f}")
            return equity
        else:
            logger.warning(
                f"Chiamata alle API Alpaca fallita. Status Code: {response.status_code}. "
                f"Dettaglio: {response.text}. Utilizzo capitale di fallback."
            )
    except Exception as e:
        logger.error(f"Errore di connessione alle API Alpaca: {e}. Utilizzo capitale di fallback.")
        
    return config.BACKTEST_CAPITALE_INIZIALE


def main():
    parser = argparse.ArgumentParser(
        description="Piattaforma Trading - Script Unificato di Esecuzione Backtest"
    )
    
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        default="nn_v1",
        choices=["nn_v1", "nn_v2", "nn_v3", "nn_v4", "nn_v5", "nn_v6", "nn_v7", "nn_v8", "nn_v9", "nn_v10", "nn_v11", "moe_v1", "gnn_v1", "sma"],
        help="La strategia/modello da simulare (default: nn_v1)."
    )
    
    parser.add_argument(
        "-t", "--tickers",
        type=str,
        help="Lista di ticker separati da virgola su cui simulare. Se omesso, usa il pool predefinito di 20 azioni."
    )
    
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="Il file dei pesi del modello PyTorch da caricare (se omesso, usa il default specifico della strategia)."
    )
    
    parser.add_argument(
        "-pt", "--probability_threshold",
        type=float,
        default=0.58,
        help="La soglia di probabilità per generare un segnale di BUY (default: 0.58)."
    )
    
    parser.add_argument(
        "--start_date",
        type=str,
        default="2024-04-03",
        help="Data inizio simulazione nel formato YYYY-MM-DD (default: 2024-04-03, inizio periodo Out-Of-Sample)."
    )
    
    parser.add_argument(
        "--end_date",
        type=str,
        default="2026-05-22",
        help="Data fine simulazione nel formato YYYY-MM-DD (default: 2026-05-22, fine periodo Out-Of-Sample)."
    )
    
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Disattiva la ripartizione equa del budget tra gli asset (usa il peso fisso del config)."
    )
    
    parser.add_argument(
        "--pool_size",
        type=int,
        default=None,
        help="Se specificato, seleziona i primi N ticker più attivi nel database automaticamente."
    )
    
    parser.add_argument(
        "--no_ranking",
        action="store_true",
        help="Disattiva la modalità Relative Strength Ranking per la strategia v4 (usa la logica a soglia assoluta)."
    )
    
    parser.add_argument(
        "--no_trend_filter",
        action="store_true",
        help="Disattiva il filtro di trend macro basato su SMA 200 per la strategia v4."
    )
    
    parser.add_argument(
        "--top_pct",
        type=float,
        default=0.03,
        help="Percentuale di top asset da selezionare nel ranking (default: 0.03 = 3%)."
    )
    
    parser.add_argument(
        "--exit_pct",
        type=float,
        default=0.60,
        help="Percentuale di asset da mantenere nel ranking prima di liquidare (default: 0.60 = 60%)."
    )

    parser.add_argument(
        "--stop_loss_atr_mult",
        type=float,
        default=5.5,
        help="Moltiplicatore ATR per il calcolo dello Stop Loss (default: 5.5)."
    )

    parser.add_argument(
        "--take_profit_mult",
        type=float,
        default=2.0,
        help="Moltiplicatore dello Stop Loss per il calcolo del Take Profit (default: 2.0)."
    )
    
    parser.add_argument(
        "--prob_threshold_long",
        type=float,
        default=None,
        help="Soglia probabilistica personalizzata per posizioni LONG nella strategia v4."
    )
    
    parser.add_argument(
        "--prob_threshold_short",
        type=float,
        default=None,
        help="Soglia probabilistica personalizzata per posizioni SHORT nella strategia v4."
    )
    
    parser.add_argument(
        "--max_slots",
        type=int,
        default=20,
        help="Numero massimo di posizioni aperte contemporaneamente (default: 20)."
    )
    
    parser.add_argument(
        "--use_trailing_only",
        action="store_true",
        help="Se attivo, disabilita il Take Profit fisso e si affida esclusivamente al Trailing Stop basato su ATR per chiudere i trade in profitto."
    )

    parser.add_argument(
        "--trailing_stop_atr_mult",
        type=float,
        default=3.0,
        help="Moltiplicatore ATR per il calcolo della distanza del Trailing Stop dinamico (default: 3.0)."
    )

    parser.add_argument(
        "--no_dynamic_slots",
        action="store_true",
        help="Se attivo, disattiva il Filtro Breadth Reattivo che regola dinamicamente max_slots e la riserva di cash della strategia, forzando l'utilizzo di parametri statici."
    )

    parser.add_argument(
        "--short_breadth_thresh",
        type=float,
        default=0.40,
        help="Soglia di market breadth al di sopra della quale lo shorting viene disabilitato (default: 0.40)."
    )
    
    args = parser.parse_args()
    
    # Se la strategia è nn_v4, nn_v5, nn_v6, nn_v7, nn_v8, nn_v9 o nn_v10 e la soglia è quella di default (0.58), la abbassiamo a 0.525 per supportare la modalità ranking ed evitare compressione di probabilità
    if args.strategy in ["nn_v4", "nn_v5", "nn_v6", "nn_v7", "nn_v8", "nn_v9", "nn_v10", "nn_v11", "moe_v1"] and args.probability_threshold == 0.58:
        args.probability_threshold = 0.525
        logger.info(f"Rilevata strategia {args.strategy}: impostata automaticamente la soglia probabilistica a 0.525 per ottimizzare il Relative Ranking.")
    

    print("\n" + "="*75)
    print("                AVVIO SIMULATORE DI BACKTEST UNIFICATO")
    print("="*75)
    
    db = DBManager()
    
    # 1. Determinazione dei ticker da utilizzare
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        logger.info(f"Configurata simulazione personalizzata su {len(tickers)} ticker specificati: {tickers}")
    elif args.pool_size is not None:
        # Carica automaticamente i top N ticker ordinati per volume medio
        query = "SELECT ticker, AVG(volume) as avg_vol FROM ohlcv GROUP BY ticker ORDER BY avg_vol DESC LIMIT ?"
        tickers = db.execute_query(query, (args.pool_size,))['ticker'].tolist()
        logger.info(f"Selezionati automaticamente i primi {args.pool_size} ticker più attivi dal DB: {tickers}")
    else:
        tickers = DEFAULT_20_POOL
        logger.info(f"Nessun ticker specificato. Caricato il Pool di Base di 20 azioni: {tickers}")
        
    # Verifica disponibilità ticker nel DB SQLite
    db = DBManager()
    available_tickers = []
    for ticker in tickers:
        cnt = db.execute_query("SELECT COUNT(*) FROM ohlcv WHERE ticker = ?", (ticker,)).iloc[0, 0]
        if cnt > 0:
            available_tickers.append(ticker)
        else:
            logger.warning(f"Ticker [{ticker}] omesso: nessun dato storico trovato nel DB SQLite. Esegui run_ingestion.py prima.")
            
    if not available_tickers:
        logger.error("Nessuno dei ticker selezionati è presente a DB. Backtest impossibile. Esco.")
        sys.exit(1)
        
    logger.info(f"Asset validati per il backtest ({len(available_tickers)}/{len(tickers)}): {available_tickers}")
    
    # 2. Recupero del capitale di partenza da Alpaca
    alpaca_capital = get_alpaca_account_capital()
    
    # 3. Configurazione del motore di Backtest
    logger.info(f"Inizializzazione motore per periodo {args.start_date} - {args.end_date}...")
    engine = BacktestEngine(
        start_date=args.start_date, 
        end_date=args.end_date, 
        tickers=available_tickers,
        max_slots=args.max_slots
    )
    
    # 4. Selezione e configurazione della strategia
    split_equally = not args.no_split
    
    # Sovrascriviamo il portafoglio iniziale con il capitale di Alpaca e passiamo i ticker per abilitare i sub-balance solo se split_equally è attivo
    if split_equally:
        engine.portfolio = Portfolio(initial_capital=alpaca_capital, tickers=available_tickers)
    else:
        engine.portfolio = Portfolio(initial_capital=alpaca_capital, tickers=None)
    
    # Risoluzione del file del modello in base alla strategia
    default_models = {
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
        "gnn_v1": "gnn_model.pth",
    }
    model_to_use = args.model_file if args.model_file is not None else default_models.get(args.strategy, "neural_model.pth")
    
    if args.strategy == "moe_v1":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "moe_v1" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_moe.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "moe_v1" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: MoEStrategyV1(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short,
            top_pct=args.top_pct,
            exit_pct=args.exit_pct,
            stop_loss_atr_mult=args.stop_loss_atr_mult,
            take_profit_mult=args.take_profit_mult,
            use_trailing_only=args.use_trailing_only,
            trailing_stop_atr_mult=args.trailing_stop_atr_mult,
            dynamic_slots=not args.no_dynamic_slots,
            base_max_slots=args.max_slots,
            short_breadth_thresh=args.short_breadth_thresh
        )
    elif args.strategy == "gnn_v1":
        model_path = config.BASE_DIR / "models" / "gnn" / "v1" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "gnn_model.pth"
            fallback_path = config.BASE_DIR / "models" / "gnn" / "v1" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkGNNStrategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short,
            top_pct=args.top_pct,
            exit_pct=args.exit_pct,
            stop_loss_atr_mult=args.stop_loss_atr_mult,
            take_profit_mult=args.take_profit_mult,
            use_trailing_only=args.use_trailing_only,
            trailing_stop_atr_mult=args.trailing_stop_atr_mult
        )
    elif args.strategy == "nn_v11":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v11" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_v11.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v11" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV11Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short,
            top_pct=args.top_pct,
            exit_pct=args.exit_pct,
            stop_loss_atr_mult=args.stop_loss_atr_mult,
            take_profit_mult=args.take_profit_mult,
            use_trailing_only=args.use_trailing_only,
            trailing_stop_atr_mult=args.trailing_stop_atr_mult,
            dynamic_slots=not args.no_dynamic_slots,
            base_max_slots=args.max_slots,
            short_breadth_thresh=args.short_breadth_thresh
        )
    elif args.strategy == "nn_v10":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v10" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_v10.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v10" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV10Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short,
            top_pct=args.top_pct,
            exit_pct=args.exit_pct,
            stop_loss_atr_mult=args.stop_loss_atr_mult,
            take_profit_mult=args.take_profit_mult,
            use_trailing_only=args.use_trailing_only,
            trailing_stop_atr_mult=args.trailing_stop_atr_mult,
            dynamic_slots=not args.no_dynamic_slots,
            base_max_slots=args.max_slots,
            short_breadth_thresh=args.short_breadth_thresh
        )
    elif args.strategy == "nn_v9":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v6" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_aapl.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v6" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV9Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short
        )
    elif args.strategy == "nn_v8":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v6" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_aapl.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v6" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV8Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short
        )
    elif args.strategy == "nn_v7":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v6" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_aapl.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v6" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV7Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short
        )
    elif args.strategy == "nn_v6":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v6" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_aapl.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v6" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV6Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short
        )
    elif args.strategy == "nn_v5":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v5" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_aapl.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v5" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV5Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short
        )
    elif args.strategy == "nn_v4":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v4" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_aapl.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v4" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV4Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold,
            ranking_mode=not args.no_ranking,
            trend_filter=not args.no_trend_filter,
            probability_threshold_long=args.prob_threshold_long,
            probability_threshold_short=args.prob_threshold_short
        )
    elif args.strategy == "nn_v3":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v3" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_aapl.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v3" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV3Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold
        )
    elif args.strategy == "nn_v2":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v2" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_aapl.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v2" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkV2Strategy(
            model_filename=model_to_use,
            probability_threshold=args.probability_threshold
        )
    elif args.strategy == "nn_v1":
        model_path = config.BASE_DIR / "models" / "rete_neurale" / "v1" / "pesi" / model_to_use
        if not model_path.exists():
            fallback_model = "neural_model_aapl.pth"
            fallback_path = config.BASE_DIR / "models" / "rete_neurale" / "v1" / "pesi" / fallback_model
            if fallback_path.exists():
                logger.warning(f"File '{model_to_use}' non trovato. Ripiego sul modello di prova addestrato '{fallback_model}'.")
                model_to_use = fallback_model
                
        strategy_class = lambda: NeuralNetworkStrategy(
            model_filename=model_to_use, 
            probability_threshold=args.probability_threshold
        )
    else:
        strategy_class = SMAXStrategy
        
    # 5. Esecuzione del Backtest
    logger.info(f"Avvio simulazione con strategia: {args.strategy.upper()}...")
    report = engine.run(strategy_class, split_equally=split_equally)
    
    # 6. Presentazione dei Risultati
    if "error" in report:
        logger.error(f"Errore durante l'esecuzione: {report['error']}")
        sys.exit(1)
        
    capital = report["capital"]
    metrics = report["metrics"]
    trades = report["trades"]
    
    print("\n" + "="*55)
    print(f"      RISULTATI SIMULAZIONE ({args.strategy.upper()})")
    print("="*55)
    print(f"Periodo Simulato:       dal {args.start_date} al {args.end_date}")
    print(f"Asset Coinvolti:        {len(available_tickers)} azioni")
    print(f"Ripartizione Budget:    {'EQUA (1/N per asset - SPLIT)' if split_equally else 'DINAMICA (Pool condiviso - NO SPLIT)'}")
    if "nn" in args.strategy:
        print(f"Soglia di Confidenza:   {args.probability_threshold:.2f}")
    print("-"*55)
    print(f"Capitale Iniziale:      ${capital['initial']:,.2f} (da saldo Alpaca/Config)")
    print(f"Capitale Finale:        ${capital['final']:,.2f}")
    print(f"Profitto Netto:         ${capital['net_profit']:,.2f}")
    print(f"Ritorno Cumulativo:     {metrics['total_return_pct']:.2f}%")
    print(f"Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
    print(f"Win Rate:               {metrics['win_rate_pct']:.2f}%")
    print("-"*55)
    print(f"Operazioni Eseguite:    {trades['total']}")
    print(f"Operazioni Vincenti:    {trades['wins']}")
    print(f"Operazioni Perdenti:    {trades['losses']}")
    print(f"Profitto Medio/Trade:   ${trades['avg_pnl_usd']:,.2f}")
    print("="*55)
    run_dir_str = report.get("run_dir", "risultati_backtest/")
    print(f"\n[SUCCESS] Risultati completi salvati nella cartella dedicata:")
    print(f"          {run_dir_str}\n")
    print(f"          - Grafico di confronto (PNG): performance_comparison.png")
    print(f"          - Report dettagliato (JSON):  backtest_report.json")
    print(f"          - Equity curve (CSV):         equity_curve.csv\n")


if __name__ == "__main__":
    main()
