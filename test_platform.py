import os
import logging
from datetime import datetime, timedelta

from database.data_ingestion import YahooFinanceDataIngestion
from backtest.engine import BacktestEngine
from backtest.strategy import SMAXStrategy

# Configurazione del logger per lo script principale
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestPlatform")


def main():
    logger.info("=== BENVENUTO NELLA PIATTAFORMA DI ALGORITHMIC TRADING ===")
    
    # 1. Inizializzazione ed Ingestione Dati (Logica Incrementale)
    logger.info("\n--- 1. FASE DI INGESTIONE DATI ---")
    ingestor = YahooFinanceDataIngestion()
    
    # Eseguiamo l'ingestione (scaricherà i dati storici tramite Yahoo Finance)
    ingestor.run_ingestion()
    
    # 2. Configurazione del Backtest
    logger.info("\n--- 2. FASE DI BACKTEST EVENT-DRIVEN ---")
    
    # Definiamo le date per il backtest (ad esempio, gli ultimi 2 anni per velocità)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    
    logger.info(f"Periodo di Backtest: dal {start_date} al {end_date}")
    
    # Inizializziamo il motore di backtesting
    engine = BacktestEngine(start_date=start_date, end_date=end_date)
    
    # Eseguiamo il backtest con la strategia Simple Moving Average Crossover (SMAXStrategy)
    report = engine.run(SMAXStrategy)
    
    # 3. Presentazione delle Metriche Principali
    logger.info("\n--- 3. REPORT FINALE DEL BACKTEST ---")
    if "error" in report:
        logger.error(f"Errore durante il backtest: {report['error']}")
        return
        
    capital = report["capital"]
    metrics = report["metrics"]
    trades = report["trades"]
    
    print("\n" + "="*45)
    print(f" RISULTATI SIMULAZIONE (SMAXStrategy)")
    print("="*45)
    print(f"Capitale Iniziale:      ${capital['initial']:,.2f}")
    print(f"Capitale Finale:        ${capital['final']:,.2f}")
    print(f"Profitto Netto:         ${capital['net_profit']:,.2f}")
    print(f"Ritorno Cumulativo:     {metrics['total_return_pct']:.2f}%")
    print(f"Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
    print(f"Win Rate:               {metrics['win_rate_pct']:.2f}%")
    print("-"*45)
    print(f"Operazioni Totali:      {trades['total']}")
    print(f"Operazioni Vincenti:    {trades['wins']}")
    print(f"Operazioni Perdenti:    {trades['losses']}")
    print(f"Profitto Medio/Trade:   ${trades['avg_pnl_usd']:,.2f}")
    print("="*45)
    
    print("\n[INFO] Il log dettagliato delle transazioni e l'Equity Curve sono stati salvati")
    print("       nella cartella 'risultati_backtest/' in formato JSON e CSV.")


if __name__ == "__main__":
    main()
