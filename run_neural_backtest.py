import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Assicuriamoci che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from backtest.engine import BacktestEngine
from backtest.strategy import SMAXStrategy, NeuralNetworkStrategy

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NeuralBacktestRunner")


def main():
    print("\n" + "="*70)
    print("      SIMULATORE DI TRADING - CONFRONTO STRATEGIE QUANTITATIVE")
    print("="*70)
    print("Confronto: NeuralNetworkStrategy (PyTorch MLP) vs SMAXStrategy (SMA Crossover)")
    print("-"*70)

    # Definiamo le date per il backtest sul periodo Out-Of-Sample (Test Set: 2024-04-03 - 2026-05-22)
    # per misurare la reale capacità predittiva del modello su dati non visti durante il training!
    start_date = "2024-04-03"
    end_date = "2026-05-22"
    ticker = "AAPL"
    
    logger.info(f"Avvio Backtest per {ticker} dal {start_date} al {end_date}")

    # --- 1. ESECUZIONE STRATEGIA RETE NEURALE (PyTorch) ---
    logger.info("\n>>> ESECUZIONE STRATEGIA RETE NEURALE (PyTorch MLP) <<<")
    
    # Inizializzazione motore per la Rete Neurale
    engine_nn = BacktestEngine(start_date=start_date, end_date=end_date, tickers=[ticker])
    
    # Eseguiamo il backtest passando la classe e configurando la strategia con il file del modello AAPL
    report_nn = engine_nn.run(
        lambda: NeuralNetworkStrategy(
            model_filename="neural_model_aapl.pth", 
            probability_threshold=0.53
        )
    )

    # --- 2. ESECUZIONE STRATEGIA BASELINE (SMA Crossover) ---
    logger.info("\n>>> ESECUZIONE STRATEGIA BASELINE (SMA Crossover) <<<")
    
    # Inizializzazione motore per la SMA
    engine_sma = BacktestEngine(start_date=start_date, end_date=end_date, tickers=[ticker])
    
    # Eseguiamo il backtest
    report_sma = engine_sma.run(SMAXStrategy)

    # --- 3. CONFRONTO FINALE DELLE PERFORMANCE ---
    print("\n" + "="*70)
    print("                     REPORT COMPARATIVO FINALE")
    print("="*70)
    
    metrics_nn = report_nn["metrics"]
    capital_nn = report_nn["capital"]
    trades_nn = report_nn["trades"]
    
    metrics_sma = report_sma["metrics"]
    capital_sma = report_sma["capital"]
    trades_sma = report_sma["trades"]

    # Formattazione tabellare dei risultati
    print(f"{'Metrica':<30} | {'Rete Neurale (PyTorch)':<22} | {'SMA Crossover (Baseline)':<22}")
    print("-"*78)
    print(f"{'Capitale Iniziale':<30} | ${capital_nn['initial']:<21,.2f} | ${capital_sma['initial']:<21,.2f}")
    print(f"{'Capitale Finale':<30} | ${capital_nn['final']:<21,.2f} | ${capital_sma['final']:<21,.2f}")
    print(f"{'Profitto/Perdita Netta':<30} | ${capital_nn['net_profit']:<21,.2f} | ${capital_sma['net_profit']:<21,.2f}")
    print(f"{'Ritorno Cumulativo':<30} | {metrics_nn['total_return_pct']:<20.2f}% | {metrics_sma['total_return_pct']:<20.2f}%")
    print(f"{'Max Drawdown':<30} | {metrics_nn['max_drawdown_pct']:<20.2f}% | {metrics_sma['max_drawdown_pct']:<20.2f}%")
    print(f"{'Sharpe Ratio':<30} | {metrics_nn['sharpe_ratio']:<21.2f} | {metrics_sma['sharpe_ratio']:<21.2f}")
    print(f"{'Win Rate':<30} | {metrics_nn['win_rate_pct']:<20.2f}% | {metrics_sma['win_rate_pct']:<20.2f}%")
    print(f"{'Operazioni Chiuse':<30} | {trades_nn['total']:<21} | {trades_sma['total']:<21}")
    print(f"{'Trade Vincenti/Perdenti':<30} | {trades_nn['wins']}/{trades_nn['losses']:<17} | {trades_sma['wins']}/{trades_sma['losses']:<17}")
    print(f"{'Profitto Medio per Trade':<30} | ${trades_nn['avg_pnl_usd']:<21,.2f} | ${trades_sma['avg_pnl_usd']:<21,.2f}")
    print("="*70)
    print("\n[SUCCESS] Risultati completi salvati nelle cartelle dedicate:")
    print(f"          - Rete Neurale (PyTorch): {report_nn.get('run_dir')}")
    print(f"          - SMA Baseline (SMAX):    {report_sma.get('run_dir')}")
    print("          Ciascuna cartella contiene il report completo JSON, l'equity curve CSV e il grafico 'performance_comparison.png'.")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
