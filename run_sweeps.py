import os
import sys
import json
import argparse
import logging
import requests
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Silenzia avvisi deprecati o future warnings di PyTorch e altre librerie
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Assicuriamoci che la directory radice sia nel path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from backtest.engine import BacktestEngine, Portfolio
from backtest.strategy import (
    NeuralNetworkV6Strategy, 
    NeuralNetworkV10Strategy, 
    NeuralNetworkV11Strategy
)
from database.db_manager import DBManager

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ParameterSweeps")

def main():
    parser = argparse.ArgumentParser(
        description="Piattaforma Trading - Script di Sweep Parametrici in Serie"
    )
    
    parser.add_argument(
        "-v", "--version",
        type=str,
        required=True,
        choices=["v6", "v10", "v11"],
        help="La versione della strategia da testare (v6, v10, v11)."
    )
    
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="Il file dei pesi specifico da caricare. Se omesso, usa i default ottimali."
    )
    
    parser.add_argument(
        "--pool_size",
        type=int,
        default=100,
        help="Numero di ticker da caricare dal database SQLite (default: 100)."
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Modalità dello sweep. 'quick' prova 6 configurazioni chiave ben distinte; 'full' esegue una griglia più ampia."
    )
    
    parser.add_argument(
        "--start_date",
        type=str,
        default="2024-04-03",
        help="Data inizio backtest YYYY-MM-DD (default: 2024-04-03)."
    )
    
    parser.add_argument(
        "--end_date",
        type=str,
        default="2026-05-22",
        help="Data fine backtest YYYY-MM-DD (default: 2026-05-22)."
    )
    
    args = parser.parse_args()
    
    # 1. Scelta automatica del modello di default in base alla versione
    model_file = args.model_file
    if not model_file:
        if args.version == "v6":
            model_file = "neural_model.pth"
        elif args.version == "v10":
            model_file = "neural_model_v10_deep.pth"
            # Fallback se il modello deep non esiste
            if not (config.BASE_DIR / "models" / "rete_neurale" / "v10" / "pesi" / model_file).exists():
                model_file = "neural_model_v10.pth"
        elif args.version == "v11":
            model_file = "neural_model_v11.pth"
            
    print("\n" + "="*80)
    print(f"       AVVIO SESSIONE DI SWEEP PARAMETRICI - STRATEGIA {args.version.upper()}")
    print(f"       Modello: {model_file} | Pool: {args.pool_size} Asset | Modo: {args.mode.upper()}")
    print("="*80)
    
    # 2. Caricamento dei ticker dal database (ordinati per volume medio per evitare ordinamento alfabetico spuro)
    db = DBManager()
    query = "SELECT ticker, AVG(volume) as avg_vol FROM ohlcv GROUP BY ticker ORDER BY avg_vol DESC LIMIT ?"
    tickers = db.execute_query(query, (args.pool_size,))['ticker'].tolist()
    
    available_tickers = []
    for ticker in tickers:
        cnt = db.execute_query("SELECT COUNT(*) FROM ohlcv WHERE ticker = ?", (ticker,)).iloc[0, 0]
        if cnt > 0:
            available_tickers.append(ticker)
            
    if not available_tickers:
        logger.error("Nessun ticker trovato nel database per lo sweep. Esco.")
        sys.exit(1)
        
    logger.info(f"Asset validati per gli sweep ({len(available_tickers)}/{args.pool_size}): {available_tickers[:10]}... + altri")
    
    # Recupera il saldo iniziale reale da Alpaca o fallback
    def get_alpaca_account_capital() -> float:
        logger.info("Verifica del capitale disponibile su Alpaca Account...")
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

    alpaca_capital = get_alpaca_account_capital()
    
    # 3. Definizione delle combinazioni di parametri
    if args.mode == "quick":
        # 6 Configurazioni chiave logiche e diversificate per risparmiare tempo
        combinations = [
            {
                "name": "1_Conservative_Concentrated",
                "max_slots": 5,
                "stop_loss_atr_mult": 4.0,
                "trailing_stop_atr_mult": 2.0,
                "probability_threshold": 0.525,
                "prob_threshold_long": 0.525,
                "prob_threshold_short": 0.475
            },
            {
                "name": "2_Aggressive_Concentrated",
                "max_slots": 5,
                "stop_loss_atr_mult": 5.0,
                "trailing_stop_atr_mult": 3.0,
                "probability_threshold": 0.525,
                "prob_threshold_long": 0.525,
                "prob_threshold_short": 0.475
            },
            {
                "name": "3_Tight_Concentrated",
                "max_slots": 5,
                "stop_loss_atr_mult": 3.5,
                "trailing_stop_atr_mult": 1.5,
                "probability_threshold": 0.535,
                "prob_threshold_long": 0.535,
                "prob_threshold_short": 0.465
            },
            {
                "name": "4_Conservative_Diversified",
                "max_slots": 8,
                "stop_loss_atr_mult": 4.0,
                "trailing_stop_atr_mult": 2.0,
                "probability_threshold": 0.525,
                "prob_threshold_long": 0.525,
                "prob_threshold_short": 0.475
            },
            {
                "name": "5_Aggressive_Diversified",
                "max_slots": 8,
                "stop_loss_atr_mult": 5.0,
                "trailing_stop_atr_mult": 3.0,
                "probability_threshold": 0.525,
                "prob_threshold_long": 0.525,
                "prob_threshold_short": 0.475
            },
            {
                "name": "6_High_Confidence_Momentum",
                "max_slots": 6,
                "stop_loss_atr_mult": 4.5,
                "trailing_stop_atr_mult": 2.5,
                "probability_threshold": 0.535,
                "prob_threshold_long": 0.535,
                "prob_threshold_short": 0.465
            }
        ]
    else:
        # Griglia di sweep full più ampia (12 combinazioni logiche)
        combinations = []
        c_idx = 1
        for slots in [5, 8]:
            for sl in [3.5, 4.0, 5.0]:
                for ts in [2.0, 3.0]:
                    combinations.append({
                        "name": f"{c_idx}_Slots{slots}_SL{sl}_TS{ts}",
                        "max_slots": slots,
                        "stop_loss_atr_mult": sl,
                        "trailing_stop_atr_mult": ts,
                        "probability_threshold": 0.525,
                        "prob_threshold_long": 0.525,
                        "prob_threshold_short": 0.475
                    })
                    c_idx += 1

    results = []
    all_equity_curves = {}
    all_reports = {}
    
    # 4. Esecuzione dei Backtest in serie
    for idx, params in enumerate(combinations):
        print("\n" + "-"*80)
        print(f" [{idx+1}/{len(combinations)}] ESECUZIONE RUN: {params['name']}")
        print(f" Parametri: Slots={params['max_slots']} | SL_mult={params['stop_loss_atr_mult']} | Trailing={params['trailing_stop_atr_mult']} | Soglia={params['probability_threshold']}")
        print("-"*80)
        
        # Inizializziamo il motore di Backtest
        engine = BacktestEngine(
            start_date=args.start_date,
            end_date=args.end_date,
            tickers=available_tickers,
            max_slots=params["max_slots"]
        )
        
        # Sovrascriviamo il portafoglio iniziale con sub-balance
        engine.portfolio = Portfolio(initial_capital=alpaca_capital, tickers=available_tickers)
        
        # Monkey-patching dei metodi di salvataggio del motore per evitare di sporcare il disco con report individuali e grafici ridondanti
        engine._save_report = lambda report, strategy_name: config.BASE_DIR / "risultati_backtest"
        engine._generate_plots = lambda report, run_dir, bh_equity: None
        engine._generate_ticker_plots = lambda report, run_dir: None
        
        # Selezione della classe di strategia appropriata
        if args.version == "v6":
            strategy_class = lambda: NeuralNetworkV6Strategy(
                model_filename=model_file,
                probability_threshold=params["probability_threshold"],
                probability_threshold_long=params["prob_threshold_long"],
                probability_threshold_short=params["prob_threshold_short"]
            )
        elif args.version == "v10":
            strategy_class = lambda: NeuralNetworkV10Strategy(
                model_filename=model_file,
                probability_threshold=params["probability_threshold"],
                probability_threshold_long=params["prob_threshold_long"],
                probability_threshold_short=params["prob_threshold_short"],
                stop_loss_atr_mult=params["stop_loss_atr_mult"],
                trailing_stop_atr_mult=params["trailing_stop_atr_mult"],
                base_max_slots=params["max_slots"]
            )
        elif args.version == "v11":
            strategy_class = lambda: NeuralNetworkV11Strategy(
                model_filename=model_file,
                probability_threshold=params["probability_threshold"],
                probability_threshold_long=params["prob_threshold_long"],
                probability_threshold_short=params["prob_threshold_short"],
                stop_loss_atr_mult=params["stop_loss_atr_mult"],
                trailing_stop_atr_mult=params["trailing_stop_atr_mult"],
                base_max_slots=params["max_slots"]
            )
            
        # Esecuzione
        report = engine.run(strategy_class, split_equally=False)
        
        # Estrazione metriche essenziali
        if "error" in report:
            logger.error(f"Errore nella run {params['name']}: {report['error']}")
            continue
            
        metrics = report["metrics"]
        trades = report["trades"]
        
        # Salviamo la curva di equity e il report
        if "equity_curve" in report and report["equity_curve"]:
            all_equity_curves[params["name"]] = report["equity_curve"]
            all_reports[params["name"]] = report
            
        res_entry = {
            "name": params["name"],
            "parameters": {
                "max_slots": params["max_slots"],
                "stop_loss_atr_mult": params["stop_loss_atr_mult"],
                "trailing_stop_atr_mult": params["trailing_stop_atr_mult"],
                "probability_threshold": params["probability_threshold"]
            },
            "metrics": {
                "total_return_pct": round(metrics["total_return_pct"], 2),
                "max_drawdown_pct": round(metrics["max_drawdown_pct"], 2),
                "sharpe_ratio": round(metrics["sharpe_ratio"], 2),
                "win_rate_pct": round(metrics["win_rate_pct"], 2),
                "total_trades": trades["total"],
                "final_value": round(report["capital"]["final"], 2)
            }
        }
        results.append(res_entry)
        
        # Stampiamo un recap istantaneo dettagliato
        print(f" -> RISULTATI: Ritorno={res_entry['metrics']['total_return_pct']}% | DD={res_entry['metrics']['max_drawdown_pct']}% | Sharpe={res_entry['metrics']['sharpe_ratio']} | WR={res_entry['metrics']['win_rate_pct']}% | Trade={res_entry['metrics']['total_trades']} | Capitale={res_entry['metrics']['final_value']}")

    # 5. Classificazione e Ordinamento dei Risultati
    # Ordiniamo per Sharpe Ratio decrescente (e ritorno in caso di pareggio)
    results_sorted = sorted(results, key=lambda x: (x["metrics"]["sharpe_ratio"], x["metrics"]["total_return_pct"]), reverse=True)
    
    # 6. Scrittura del Report di Confronto finale
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    confronto_dir = config.BASE_DIR / "risultati_backtest" / "confronto"
    
    # Creiamo la sottocartella specifica per il report di questa sessione
    report_folder_name = f"report_{args.version.lower()}_{timestamp_str}"
    report_run_dir = confronto_dir / report_folder_name
    report_run_dir.mkdir(exist_ok=True, parents=True)
    
    report_filename = f"report_{args.version.upper()}_{timestamp_str}"
    
    # Generazione del grafico di confronto equity curve
    plot_filename = "equity_curves_comparison.png"
    plot_path = report_run_dir / plot_filename
    if all_equity_curves:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
            
            # Troviamo le date e plottiamo ciascuna curva
            for run_name, curve in all_equity_curves.items():
                dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in curve]
                values = [x["total_value"] for x in curve]
                
                # Calcolo rendimento cumulativo in percentuale
                initial_val = curve[0]["total_value"]
                pct_returns = [(v - initial_val) / initial_val * 100 for v in values]
                
                ax.plot(dates, pct_returns, label=f"{run_name}", linewidth=2.0)
                
            ax.set_title(f"Confronto Equity Curves - Strategia {args.version.upper()}\n(Sweep Parametrico su {len(available_tickers)} Asset)", fontsize=13, fontweight="bold", pad=15)
            ax.set_xlabel("Data", fontsize=11, labelpad=8)
            ax.set_ylabel("Rendimento Cumulativo (%)", fontsize=11, labelpad=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white", edgecolor="gray")
            
            fig.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info(f"Grafico delle curve di profitto salvato in: {plot_path}")
        except Exception as e:
            logger.error(f"Errore durante la generazione del grafico cumulativo: {e}")
            plot_path = None

    # Salvataggio CSV riepilogativo delle curve di equity
    if all_equity_curves:
        try:
            csv_data = {}
            for run_name, curve in all_equity_curves.items():
                for x in curve:
                    d = x["date"]
                    v = x["total_value"]
                    if d not in csv_data:
                        csv_data[d] = {}
                    csv_data[d][run_name] = v
            
            dates_sorted = sorted(list(csv_data.keys()))
            csv_rows = []
            for d in dates_sorted:
                row = {"date": d}
                row.update(csv_data[d])
                csv_rows.append(row)
            
            import pandas as pd
            summary_df = pd.DataFrame(csv_rows)
            summary_df.to_csv(report_run_dir / "equity_curves_summary.csv", index=False)
            logger.info(f"CSV riepilogativo curve salvato in: {report_run_dir / 'equity_curves_summary.csv'}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio del CSV delle curve: {e}")

    # Salvataggio in JSON
    json_path = report_run_dir / f"{report_filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": args.version,
            "model_file": model_file,
            "pool_size": args.pool_size,
            "timestamp": timestamp_str,
            "runs": results_sorted
        }, f, indent=4)
        
    # Salvataggio in Markdown per una visualizzazione premium
    md_path = report_run_dir / f"{report_filename}.md"
    
    best_run = results_sorted[0] if results_sorted else None
    
    # Salvataggio dei trade della miglior run
    if best_run and best_run["name"] in all_reports:
        try:
            best_report = all_reports[best_run["name"]]
            best_trades = best_report["trades"]["log"]
            if best_trades:
                import pandas as pd
                best_trades_df = pd.DataFrame(best_trades)
                best_trades_df.to_csv(report_run_dir / "best_run_trades.csv", index=False)
                logger.info(f"I trade della miglior run salvati in: {report_run_dir / 'best_run_trades.csv'}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio dei trade della miglior run: {e}")

    md_content = f"""# Report di Sweep Parametrico - Strategia {args.version.upper()}

Questo report confronta in serie le combinazioni di parametri del modello **{args.version.upper()}** (pesi: `{model_file}`) per identificare la configurazione di risk management ottimale.

* **Data Report**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
* **Asset Universe**: {len(available_tickers)} azioni
* **Periodo Backtest**: dal {args.start_date} al {args.end_date}
* **Cartella Output**: `risultati_backtest/confronto/{report_folder_name}/`

---

## 🏆 Miglior Combinazione Rilevata

### **{best_run['name'] if best_run else 'N/D'}**
* **Ritorno Cumulativo**: {best_run['metrics']['total_return_pct'] if best_run else '0'}%
* **Max Drawdown**: {best_run['metrics']['max_drawdown_pct'] if best_run else '0'}%
* **Sharpe Ratio**: {best_run['metrics']['sharpe_ratio'] if best_run else '0'}
* **Win Rate**: {best_run['metrics']['win_rate_pct'] if best_run else '0'}%
* **Operazioni Eseguite**: {best_run['metrics']['total_trades'] if best_run else '0'}
* **Valore Portafoglio Finale**: ${best_run['metrics']['final_value'] if best_run else '0':,.2f}

---

## 📈 Andamento delle Curve di Profitto (Equity Curves)

Ecco il grafico di confronto dell'andamento cumulativo percentuale dei profitti per ciascuna configurazione testata:

![Confronto Equity Curves]({plot_filename})

---

## 📊 Tabella Comparativa delle Prestazioni

Di seguito le configurazioni testate, ordinate in base al **Sharpe Ratio** (efficienza corretta per il rischio):

| Rango | Nome Run | Slots | SL (ATR) | Trailing (ATR) | Ritorno (%) | Drawdown (%) | Sharpe | Win Rate (%) | Trade | Capitale Finale ($) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for rank, res in enumerate(results_sorted):
        md_content += f"| {rank+1} | **{res['name']}** | {res['parameters']['max_slots']} | {res['parameters']['stop_loss_atr_mult']} | {res['parameters']['trailing_stop_atr_mult']} | **{res['metrics']['total_return_pct']}%** | {res['metrics']['max_drawdown_pct']}% | **{res['metrics']['sharpe_ratio']}** | {res['metrics']['win_rate_pct']}% | {res['metrics']['total_trades']} | ${res['metrics']['final_value']:,.2f} |\n"
        
    md_content += f"""
---
### 📁 File Salvati nella Cartella dei Risultati:
1. **Report in Markdown**: `{report_filename}.md` (questo file)
2. **Report in JSON**: `{report_filename}.json` (dati completi strutturati)
3. **Grafico Equity Curves**: `{plot_filename}` (immagine delle curve cumulative)
4. **CSV Riepilogativo Curve**: `equity_curves_summary.csv` (perfetto per Excel/Python)
5. **CSV Trade Miglior Run**: `best_run_trades.csv` (log completo delle operazioni della miglior configurazione)

*Nota: Il salvataggio dei grafici e dei report individuali per ogni singola run è stato disabilitato per preservare spazio su disco.*"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    # 7. Stampa del Recap Finale a Terminale
    print("\n" + "="*80)
    print("      RECAP GENERALE DETTAGLIATO DEGLI SWEEP PARAMETRICI")
    print("="*80)
    print(f"Salvataggio Report in: D:\\python\\PrevisioneMercato\\risultati_backtest\\confronto\\{report_folder_name}\\{report_filename}.md")
    print("-"*80)
    print(f"MIGLIOR STRATEGIA RILEVATA: {best_run['name'] if best_run else 'N/D'}")
    print(f"  - Ritorno Cumulativo:       {best_run['metrics']['total_return_pct'] if best_run else '0'}%")
    print(f"  - Max Drawdown:             {best_run['metrics']['max_drawdown_pct'] if best_run else '0'}%")
    print(f"  - Sharpe Ratio:             {best_run['metrics']['sharpe_ratio'] if best_run else '0'}")
    print(f"  - Win Rate:                 {best_run['metrics']['win_rate_pct'] if best_run else '0'}%")
    print(f"  - Numero Trade:             {best_run['metrics']['total_trades'] if best_run else '0'}")
    print("-"*80)
    print("\nCLASSIFICA DI PRESTAZIONE GENERALE (Ordinata per Sharpe):")
    for idx, r in enumerate(results_sorted):
        print(f" {idx+1:2d}. [{r['name']}] Return: {r['metrics']['total_return_pct']:>6.2f}% | DD: {r['metrics']['max_drawdown_pct']:>6.2f}% | Sharpe: {r['metrics']['sharpe_ratio']:>5.2f} | WR: {r['metrics']['win_rate_pct']:>5.2f}% | Trade: {r['metrics']['total_trades']:3d} | Capitale: ${r['metrics']['final_value']:,.2f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
