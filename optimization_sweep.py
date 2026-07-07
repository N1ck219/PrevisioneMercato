import os
import sys
import argparse
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Assicuriamoci che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from backtest.engine import BacktestEngine, Portfolio
from backtest.strategy import (
    SMAXStrategy, 
    NeuralNetworkStrategy, 
    NeuralNetworkV2Strategy, 
    NeuralNetworkV3Strategy, 
    NeuralNetworkV4Strategy, 
    NeuralNetworkV5Strategy,
    NeuralNetworkV6Strategy
)
from database.db_manager import DBManager
from backtest.strategy import BaseStrategy

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OptimizationSweep")

DEFAULT_20_POOL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", 
    "META", "BRK-B", "LLY", "AVGO", "JPM", 
    "TSLA", "XOM", "UNH", "JNJ", "V", 
    "PG", "HD", "MA", "ABBV", "CVX"
]

def check_model_exists(strategy: str, model_file: str) -> bool:
    """Verifica se il file dei pesi per una determinata strategia esiste."""
    folder_map = {
        "nn_v1": "v1",
        "nn_v2": "v2",
        "nn_v3": "v3",
        "nn_v4": "v4",
        "nn_v5": "v5",
        "nn_v6": "v6",
    }
    if strategy not in folder_map:
        return True # smax non ha un modello fisico
    
    path = config.BASE_DIR / "models" / "rete_neurale" / folder_map[strategy] / "pesi" / model_file
    return path.exists()

def get_alpaca_account_capital() -> float:
    """
    Interroga le API di Alpaca per recuperare il saldo del conto.
    In caso di errore o credenziali mancanti, ripiega sul capitale di fallback.
    """
    if (config.ALPACA_API_KEY == "YOUR_API_KEY" or 
        config.ALPACA_SECRET_KEY == "YOUR_SECRET_KEY" or 
        not config.ALPACA_API_KEY or 
        not config.ALPACA_SECRET_KEY):
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
            return float(data.get("equity", data.get("cash", config.BACKTEST_CAPITALE_INIZIALE)))
    except Exception:
        pass
        
    return config.BACKTEST_CAPITALE_INIZIALE

def run_single_backtest(
    strategy_name: str,
    tickers: List[str],
    max_slots: int,
    threshold: float,
    no_split: bool,
    start_date: str,
    end_date: str,
    model_file: str = "neural_model.pth"
) -> Dict[str, Any]:
    """Esegue una singola simulazione di backtest programmando l'engine ed estraendo le metriche."""
    # Recuperiamo il capitale reale di Alpaca o quello di default da config
    try:
        alpaca_capital = get_alpaca_account_capital()
    except Exception:
        alpaca_capital = 100000.0

    # Inizializziamo l'engine
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        max_slots=max_slots
    )
    
    # Configurazione Portfolio
    if not no_split:
        engine.portfolio = Portfolio(initial_capital=alpaca_capital, tickers=tickers)
    else:
        engine.portfolio = Portfolio(initial_capital=alpaca_capital, tickers=None)
        
    # Mappatura della strategia con la classe corretta
    if strategy_name == "nn_v6":
        strategy_class = lambda: NeuralNetworkV6Strategy(
            model_filename=model_file,
            probability_threshold=threshold,
            ranking_mode=True,
            trend_filter=True
        )
    elif strategy_name == "nn_v5":
        strategy_class = lambda: NeuralNetworkV5Strategy(
            model_filename=model_file,
            probability_threshold=threshold,
            ranking_mode=True,
            trend_filter=True
        )
    elif strategy_name == "nn_v4":
        strategy_class = lambda: NeuralNetworkV4Strategy(
            model_filename=model_file,
            probability_threshold=threshold,
            ranking_mode=True,
            trend_filter=True
        )
    elif strategy_name == "nn_v3":
        strategy_class = lambda: NeuralNetworkV3Strategy(
            model_filename=model_file,
            probability_threshold=threshold
        )
    elif strategy_name == "nn_v2":
        strategy_class = lambda: NeuralNetworkV2Strategy(
            model_filename=model_file,
            probability_threshold=threshold
        )
    elif strategy_name == "nn_v1":
        strategy_class = lambda: NeuralNetworkStrategy(
            model_filename=model_file,
            probability_threshold=threshold
        )
    else:
        strategy_class = SMAXStrategy

    # Esecuzione
    try:
        report = engine.run(strategy_class, split_equally=(not no_split))
        if "error" in report:
            return {"error": report["error"]}
        
        return {
            "strategy": strategy_name,
            "max_slots": max_slots,
            "threshold": threshold,
            "no_split": no_split,
            "net_profit": report["capital"]["net_profit"],
            "return_pct": report["metrics"]["total_return_pct"],
            "max_dd_pct": report["metrics"]["max_drawdown_pct"],
            "sharpe": report["metrics"]["sharpe_ratio"],
            "win_rate": report["metrics"]["win_rate_pct"],
            "total_trades": report["trades"]["total"]
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Sweep di ottimizzazione iperparametri multicombinazione per modelli di trading.")
    parser.add_argument("--pool_size", type=int, default=20, choices=[10, 20, 100], help="Dimensione del pool di asset da testare.")
    parser.add_argument("--start_date", type=str, default="2024-04-03", help="Data di inizio backtest.")
    parser.add_argument("--end_date", type=str, default="2026-05-22", help="Data di fine backtest.")
    parser.add_argument("--model_file", type=str, default="neural_model.pth", help="Nome del file dei pesi del modello.")
    args = parser.parse_args()

    print("=========================================================================")
    print("                AVVIO OTTIMIZZATORE MULTI-MODELLO E PARAMETRI")
    print("=========================================================================")
    
    # 1. Selezione ticker in base alla dimensione del pool desiderata
    db = DBManager()
    if args.pool_size == 10:
        # Selezioniamo 10 ticker storicamente peggiori o predefiniti
        tickers = DEFAULT_20_POOL[:10]
    elif args.pool_size == 20:
        tickers = DEFAULT_20_POOL
    else:
        # Recupera tutti i 100 ticker più attivi a DB
        query = """
            SELECT ticker, COUNT(*) as cnt 
            FROM ohlcv 
            GROUP BY ticker 
            ORDER BY cnt DESC 
            LIMIT 100
        """
        df_tickers = db.execute_query(query)
        tickers = df_tickers['ticker'].tolist()

    # Validazione ticker
    valid_tickers = []
    for t in tickers:
        df = db.execute_query("SELECT COUNT(*) as count FROM ohlcv WHERE ticker = ?", (t,))
        if not df.empty and df.iloc[0]['count'] > 200:
            valid_tickers.append(t)
            
    print(f"Asset validati per lo sweep ({len(valid_tickers)}/{len(tickers)}): {valid_tickers}")
    print(f"Periodo di test: {args.start_date} - {args.end_date}\n")

    # 2. Definizione della griglia di iperparametri da esplorare
    strategies = ["nn_v6", "nn_v5", "nn_v4", "nn_v3", "nn_v2", "nn_v1", "smax"]
    max_slots_options = [3, 5, 8]
    threshold_options = [0.515, 0.525, 0.535]
    budget_split_options = [True]  # no_split = True (budget condiviso per capitalizzazione massima)

    results = []
    run_idx = 1
    total_runs = len(strategies) * len(max_slots_options) * len(threshold_options) * len(budget_split_options)

    # 3. Esecuzione del Grid Search Sweep
    for strat in strategies:
        # Verifica preventiva dell'esistenza dei pesi
        if not check_model_exists(strat, args.model_file):
            print(f"[Warning] Saltata strategia '{strat}': modello '{args.model_file}' non trovato nei pesi.")
            continue
            
        for slots in max_slots_options:
            for thresh in threshold_options:
                for no_split in budget_split_options:
                    # smax non ha una soglia probabilistica di confidenza
                    if strat == "smax" and thresh != threshold_options[0]:
                        continue # Evita di raddoppiare inutilmente i run di smax
                        
                    print(f"[{run_idx}/{total_runs}] Testando: {strat.upper()} | Slots: {slots} | Thresh: {thresh} | Budget: Condiviso...")
                    
                    res = run_single_backtest(
                        strategy_name=strat,
                        tickers=valid_tickers,
                        max_slots=slots,
                        threshold=thresh,
                        no_split=no_split,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        model_file=args.model_file
                    )
                    
                    if "error" not in res:
                        results.append(res)
                        print(f"   => Successo! Ritorno: {res['return_pct']:.2f}% | Sharpe: {res['sharpe']:.2f} | Trade: {res['total_trades']}")
                    else:
                        print(f"   => [ERRORE] Saltato: {res['error']}")
                        
                    run_idx += 1

    if not results:
        print("[ERRORE] Nessun backtest completato con successo. Verifica i file dei modelli o il DB.")
        sys.exit(1)

    # 4. Elaborazione e salvataggio dei risultati in CSV e JSON
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="net_profit", ascending=False).reset_index(drop=True)

    output_dir = config.BASE_DIR / "risultati_backtest" / "ottimizzazione"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    csv_path = output_dir / "sweep_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Sweep completato! Tabella risultati salvata in: {csv_path}")

    # 5. Visualizzazione in formato Markdown a schermo (con ripiego a to_string se tabulate manca)
    print("\n=========================================================================")
    print("                    TOP 15 COMBINAZIONI DI PARAMETRI")
    print("=========================================================================")
    try:
        print(df_res.head(15).to_markdown(index=False))
    except ImportError:
        print(df_res.head(15).to_string(index=False))

    # 6. Generazione del Grafico Estetico Premium
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

    # Mostriamo solo le prime 15 combinazioni per una lettura pulita ed estetica
    plot_df = df_res.head(15).copy()
    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['strategy'].upper()} (Slots: {r['max_slots']}, Thresh: {r['threshold']})", axis=1
    )
    
    # Ordiniamo in modo che la migliore stia in alto nel grafico orizzontale
    plot_df = plot_df.iloc[::-1]

    # Colormap basata sullo Sharpe Ratio per una visualizzazione bidimensionale fantastica
    norm = plt.Normalize(plot_df["sharpe"].min(), plot_df["sharpe"].max())
    colors = plt.cm.coolwarm(norm(plot_df["sharpe"]))

    bars = ax.barh(plot_df["label"], plot_df["return_pct"], color=colors, edgecolor='none', height=0.6)
    
    # Aggiunta di etichette di testo su ogni barra con le informazioni
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        width = bar.get_width()
        text_align = 'left' if width >= 0 else 'right'
        padding = 0.5 if width >= 0 else -0.5
        ax.text(
            width + padding, 
            bar.get_y() + bar.get_height()/2, 
            f"+{width:.2f}% (Sharpe: {row['sharpe']:.2f})", 
            va='center', 
            ha=text_align, 
            fontsize=9, 
            fontweight='bold',
            color='#333333'
        )

    # Barra del colore dello Sharpe
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Sharpe Ratio", fontsize=11, labelpad=10)

    ax.set_title("Sweep Optimization: Classifica Top Combinazioni di Parametri\n(Ordinato per Ritorno Cumulativo %, Colore = Sharpe Ratio)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Ritorno Cumulativo (%)", fontsize=12, labelpad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Rimuovi i bordi grafici inutili per massima pulizia
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    chart_path = output_dir / "optimization_sweep.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()

    print(f"\n[SUCCESS] Grafico comparativo salvato in: {chart_path}")
    print("=========================================================================\n")

if __name__ == "__main__":
    main()
