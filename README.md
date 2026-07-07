# 📈 Piattaforma Unificata di Trading Algoritmico & Previsione Mercato

Questa piattaforma unificata combina il recupero dei dati di mercato storici, il pre-elaborazione delle feature finanziarie, l'addestramento di modelli di Deep Learning all'avanguardia (dalle reti MLP e LSTM fino ad architetture multi-scala CNN-Transformer) e un motore di backtesting unificato ad alte prestazioni integrato con le API di Alpaca.

---

## 🛠️ Requisiti e Installazione

Prima di iniziare, assicurati di aver installato tutte le dipendenze richieste:

```bash
pip install -r requirements.txt
```

Assicurati di configurare le tue credenziali Alpaca all'interno del file `.env` posizionato nella cartella radice:
```ini
ALPACA_API_KEY="LA_TUA_API_KEY"
ALPACA_SECRET_KEY="LA_TUA_SECRET_KEY"
ALPACA_PAPER=true
```

---

## 📥 1. Ingestione dei Dati (`run_ingestion.py`)

Prima di addestrare qualsiasi modello o eseguire backtest, è necessario popolare il database SQLite locale con i dati storici e gli indicatori tecnici calcolati.

```bash
python run_ingestion.py [argomenti]
```

### 📋 Argomenti Disponibili

| Argomento | Short | Tipo | Default | Descrizione |
| :--- | :--- | :--- | :--- | :--- |
| `--tickers` | `-t` | `str` | `config.py` | Lista di ticker separati da virgola da scaricare (es. `AAPL,MSFT,TSLA`). |
| `--sp500` | - | `flag` | `False` | Scarica dinamicamente ed inserisce i dati per tutti i 500 costituenti dell'S&P 500. |
| `--years` | `-y` | `int` | `15` | Numero di anni di storico da scaricare (sovrascrive il default di `config.py`). |
| `--clear` | `-c` | `flag` | `False` | Cancella il database locale prima di iniziare il download per una pulizia totale. |

#### Esempi di Utilizzo:
```bash
# Scaricamento base dei ticker predefiniti nel config per 15 anni
python run_ingestion.py

# Download pulito per Apple e Tesla per gli ultimi 5 anni
python run_ingestion.py -t AAPL,TSLA -y 5 --clear

# Ingestione dell'intero paniere S&P 500
python run_ingestion.py --sp500 -y 10
```

---

## 🧠 2. Addestramento dei Modelli ML (`train.py`)

La piattaforma supporta 6 versioni incrementali di reti neurali, da semplici percettroni multistrato (MLP) fino a modelli sequenziali 3D basati su LSTM e Transformer multi-scala con Loss pesata sui profitti futuri.

```bash
python train.py [argomenti]
```

### 📋 Argomenti Disponibili

| Argomento | Short | Tipo | Default | Descrizione |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | `-m` | `str` | `nn_v1` | Versione del modello da addestrare. Scelte: `[nn_v1, nn_v2, nn_v3, nn_v4, nn_v5, nn_v6]`. |
| `--lookback` | - | `int` | `30` | Lunghezza della finestra temporale (lookback) in giorni per i modelli sequenziali (v3+). |
| `--tickers` | `-t` | `str` | Tutti | Lista di ticker separati da virgola su cui allenare (es. `AAPL,MSFT`). Se omesso, allena su **tutti** i ticker presenti nel DB. |
| `--save_name` | `-s` | `str` | `neural_model.pth` | Nome del file in cui salvare i pesi e i parametri del modello. |
| `--epochs` | `-e` | `int` | `120` | Numero massimo di epoche per l'addestramento. |
| `--batch_size` | `-b` | `int` | `512` | Dimensione del batch di addestramento. |
| `--patience` | `-p` | `int` | `15` | Epoche di attesa per l'Early Stopping (se la Val Loss non migliora). |
| `--resume` | `-r` | `flag` | `False` | Se attivo, riprende l'addestramento caricando i pesi e lo stato dell'ottimizzatore esistente. |

### 🔍 Dettaglio Modelli Supportati:
*   **`nn_v1`**: MLP Standard (14 Feature scala-invarianti standard).
*   **`nn_v2`**: MLP Avanzato (9 Feature v2 avanzate incl. Bollinger %B, Dist_SMA200, OBV_ret).
*   **`nn_v3`**: LSTM Sequenziale (Lookback tridimensionale con Feature v2).
*   **`nn_v4`**: CNN-Transformer Sequenziale (Feature v4 aggiuntive: ROC_10, Stoch_K, SMA_5_ratio, EMA_12_ratio, Volume_Std_Ratio).
*   **`nn_v5`**: CNN-Transformer + Feature Relative al Mercato (v5) + Profit-Weighted Loss (Loss pesata sui rendimenti effettivi futuri).
*   **`nn_v6`**: CNN-Transformer Multi-Scale + Feature Macro (Rendimento e Volatilità del mercato cross-sectionale) + Profit-Weighted Loss.

#### Esempi di Utilizzo:
```bash
# Addestramento di base del modello MLP v1 su tutti i ticker a DB
python train.py

# Addestramento del modello avanzato CNN-Transformer v6 con finestra a 45 giorni su AAPL e MSFT
python train.py -m nn_v6 --lookback 45 -t AAPL,MSFT -e 150 --save_name neural_model_v6.pth
```

---

## 📊 3. Simulatore di Backtest Unificato (`backtest.py`)

Il modulo esegue simulazioni storiche accurate (Out-Of-Sample) riproducendo logiche di portafoglio realistiche, commissioni, stop loss, take profit e interrogando in tempo reale il saldo reale del conto Alpaca per determinare il capitale iniziale.

```bash
python backtest.py [argomenti]
```

### 📋 Argomenti Disponibili

| Argomento | Short | Tipo | Default | Descrizione |
| :--- | :--- | :--- | :--- | :--- |
| `--strategy` | `-s` | `str` | `nn_v1` | La strategia o il modello neurale da simulare. Scelte: `[nn_v1, nn_v2, nn_v3, nn_v4, nn_v5, nn_v6, nn_v7, nn_v8, nn_v9, sma]`. |
| `--tickers` | `-t` | `str` | Pool 20 | Ticker separati da virgola per il backtest. Se omesso, usa un pool predefinito di 20 mega-cap. |
| `--model_file`| - | `str` | `neural_model.pth` | Nome del file dei pesi del modello PyTorch da caricare (cercato nella cartella del modello corrispondente). |
| `--probability_threshold` | `-pt` | `float` | `0.58` | Soglia di probabilità per generare un segnale di BUY (impostata automaticamente a `0.525` per i modelli con Ranking v4+). |
| `--start_date`| - | `str` | `2024-04-03` | Data d'inizio simulazione (formato `YYYY-MM-DD`). |
| `--end_date`  | - | `str` | `2026-05-22` | Data di fine simulazione (formato `YYYY-MM-DD`). |
| `--no_split`  | - | `flag` | `False` | Disattiva la ripartizione equa del budget tra gli asset (usa il peso fisso del capitale per trade dinamico). |
| `--pool_size` | - | `int` | `None` | Se impostato, seleziona in automatico i primi N ticker più attivi nel database. |
| `--no_ranking`| - | `flag` | `False` | Disattiva la modalità Relative Strength Ranking per le strategie v4+ (ritorna alla soglia probabilistica assoluta). |
| `--no_trend_filter`| - | `flag` | `False` | Disattiva il filtro di trend macro basato sulla media mobile semplice a 200 periodi (SMA 200) per le strategie v4+. |
| `--prob_threshold_long`| - | `float` | `None` | Soglia probabilistica personalizzata per posizioni LONG (strategie v4+). |
| `--prob_threshold_short`| - | `float` | `None` | Soglia probabilistica personalizzata per posizioni SHORT (strategie v4+). |
| `--max_slots` | - | `int` | `20` | Numero massimo di posizioni che possono essere aperte contemporaneamente. |

### 📈 Strategie di Trading Avanzate:
*   **`nn_v1`** a **`nn_v3`**: Decisioni basate sulla soglia probabilistica assoluta impostata con `-pt`.
*   **`nn_v4`** a **`nn_v9`**: Sfruttano il **Relative Strength Ranking**. Invece di guardare solo se un asset supera una soglia, gli asset vengono ordinati in base alla confidenza della rete neurale per comprare solo la crème de la crème del paniere (con gestione del trend macro via SMA 200).
    *   *Nota*: La strategia `nn_v9` implementa protezioni aggiuntive sul drawdown e una gestione dinamica delle size.

#### Esempi di Utilizzo:
```bash
# Esecuzione del backtest standard con strategia nn_v1
python backtest.py -s nn_v1

# Backtest della strategia CNN-Transformer v9 senza frazionamento fisso del budget, usando budget condiviso dinamico
python backtest.py -s nn_v9 --no_split --start_date 2024-01-01 --max_slots 8
```

#### 📁 Output Generati:
Al termine di ogni esecuzione, i risultati vengono archiviati in una directory dedicata sotto `risultati_backtest/` contenente:
1.  `backtest_report.json`: Report analitico con tutte le metriche chiave (Profitto, Drawdown, Sharpe Ratio, Win Rate, PnL per trade).
2.  `equity_curve.csv`: Cronistoria giornaliera del valore del portafoglio e del cash.
3.  `performance_comparison.png`: Grafico ad alta definizione di confronto tra la performance della strategia e il portafoglio Buy & Hold.
4.  `ticker_performance_comparison.png`: Grafico comparativo delle performance dei singoli ticker durante la simulazione.

---

## 🧪 4. Tuning & Ottimizzazione Parametrica

La piattaforma dispone di due moduli avanzati per trovare la configurazione perfetta del modello e del trading:

### A. Ricerca Ottimale Iperparametri Modello (`hyperparameter_tuning.py`)
Utilizza l'ottimizzazione bayesiana di **Optuna** per trovare i parametri ideali della rete neurale (lookback, dimensione modello, learning rate, dropout).
```bash
python hyperparameter_tuning.py --tickers AAPL,MSFT,NVDA --trials 25
```

### B. Sweep di Ottimizzazione Trading (`optimization_sweep.py`)
Esegue un Grid Search multi-strategia esplorando incroci di iperparametri di trading (strategia, max_slots, soglia di confidenza) su pool variabili di asset e genera una classifica in formato markdown e un grafico comparativo premium salvato in `risultati_backtest/ottimizzazione/`.
```bash
python optimization_sweep.py --pool_size 20 --start_date 2024-04-03
```
*   **Argomenti**:
    *   `--pool_size`: Dimensione del pool (`10`, `20` o `100` ticker più attivi).
    *   `--start_date`: Data inizio sweep.
    *   `--end_date`: Data fine sweep.
    *   `--model_file`: File del modello da valutare.

---

## 📈 5. Sweep Parametrici in Serie di Risk Management (`run_sweeps.py`)

Questo script esegue sessioni di backtesting in serie su combinazioni predefinite di parametri di risk management per identificare il setup ottimale (in termini di Sharpe Ratio, Max Drawdown e Win Rate) per una determinata versione del modello neurale (`v6`, `v10` o `v11`). 

Per evitare di saturare il disco e semplificare l'analisi, lo script disabilita la scrittura dei singoli report del motore di backtest e genera un unico pacchetto informativo strutturato e organizzato all'interno di una cartella dedicata per ciascuna esecuzione.

```bash
python run_sweeps.py [argomenti]
```

### 📋 Argomenti Disponibili

| Argomento | Short | Tipo | Default | Descrizione |
| :--- | :--- | :--- | :--- | :--- |
| `--version` | `-v` | `str` | **Richiesto** | La versione del modello neurale da testare. Scelte: `[v6, v10, v11]`. |
| `--model_file`| - | `str` | `None` | Nome del file dei pesi specifico. Se omesso, carica automaticamente il file ottimale di default per la versione. |
| `--pool_size` | - | `int` | `100` | Numero di asset da estrarre ordinandoli per volume medio decrescente dal database. |
| `--mode` | - | `str` | `quick` | Modalità sweep: `quick` (6 run predefinite) o `full` (griglia a 12 combinazioni). |
| `--start_date`| - | `str` | `2024-04-03`| Data inizio backtest (`YYYY-MM-DD`). |
| `--end_date`  | - | `str` | `2026-05-22`| Data fine backtest (`YYYY-MM-DD`). |

---

### ⚙️ Combinazioni di Parametri Provate

#### 1. Modalità Quick (`--mode quick` - 6 Combinazioni)
Progettata per esplorare configurazioni di trading ben distinte con la massima velocità:
*   **`1_Conservative_Concentrated`**: Slots: `5` | SL ATR Mult: `4.0` | Trailing ATR Mult: `2.0` | Soglia BUY: `0.525`
*   **`2_Aggressive_Concentrated`**: Slots: `5` | SL ATR Mult: `5.0` | Trailing ATR Mult: `3.0` | Soglia BUY: `0.525`
*   **`3_Tight_Concentrated`**: Slots: `5` | SL ATR Mult: `3.5` | Trailing ATR Mult: `1.5` | Soglia BUY: `0.535`
*   **`4_Conservative_Diversified`**: Slots: `8` | SL ATR Mult: `4.0` | Trailing ATR Mult: `2.0` | Soglia BUY: `0.525`
*   **`5_Aggressive_Diversified`**: Slots: `8` | SL ATR Mult: `5.0` | Trailing ATR Mult: `3.0` | Soglia BUY: `0.525`
*   **`6_High_Confidence_Momentum`**: Slots: `6` | SL ATR Mult: `4.5` | Trailing ATR Mult: `2.5` | Soglia BUY: `0.535`

#### 2. Modalità Full (`--mode full` - 12 Combinazioni)
Esegue una ricerca a griglia (Grid Search) combinando i seguenti parametri:
*   **Max Slots**: `[5, 8]`
*   **Stop Loss ATR Multiplier**: `[3.5, 4.0, 5.0]`
*   **Trailing Stop ATR Multiplier**: `[2.0, 3.0]`
*   **Soglia di Probabilità**: `0.525`

---

### 📁 Output Generati
I risultati dello sweep vengono racchiusi in una cartella isolata: `risultati_backtest/confronto/report_[versione]_[data_ora]/` contenente:
1.  `report_[VERSION]_[TIMESTAMP].md`: Report analitico riassuntivo con tabella comparativa delle prestazioni ordinata per Sharpe Ratio e metadati di esecuzione.
2.  `report_[VERSION]_[TIMESTAMP].json`: JSON contenente i dati e le metriche di tutte le run.
3.  `equity_curves_comparison.png`: Grafico di confronto cumulato dell'andamento dei profitti di tutte le run nel corso del tempo (incorporato anche nel file `.md`).
4.  `equity_curves_summary.csv`: Serie storica giornaliera delle curve di equity di tutte le run per analisi esterne.
5.  `best_run_trades.csv`: Il log dettagliato delle singole transazioni effettuate dalla miglior configurazione dello sweep.

---

### 🚀 Esempi di Comandi per Versione

Per testare le singole versioni su un pool decorrelato di 30 asset liquidi in modalità veloce:

#### A. Sweep Parametrico Strategia V6 (Baseline CNN-Transformer)
```bash
python run_sweeps.py -v v6 --pool_size 30 --mode quick
```

#### B. Sweep Parametrico Strategia V10 (Temporal Attention Pooling)
```bash
python run_sweeps.py -v v10 --pool_size 30 --mode quick
```

#### C. Sweep Parametrico Strategia V11 (Cross-Feature & Dati Macro)
```bash
python run_sweeps.py -v v11 --pool_size 30 --mode quick
```

---

## 🔮 Sviluppi Futuri & Miglioramenti in Corso

### 1. Modulo di Capital Allocation Dinamica (Kelly Criterion & Regime Exposure)
Attualmente la dimensione di ogni trade è fissa ($1/\text{max_slots}$ del capitale per slot).
*   **L'affinamento**: Implementare un modulo che regola l'esposizione globale in base alla salute del mercato.
*   In mercati a bassissima volatilità e forte spinta rialzista (Market Breadth $> 70\%$), il modulo si farà aggressivo riducendo `max_slots` a 3 (allocando il $33\%$ a trade sui trend più forti).
*   In mercati volatili o ribassisti (Market Breadth $< 30\%$), il modulo si fa difensivo aumentando `max_slots` a 10 (allocando solo il $10\%$ a trade) o mantenendo il $50\%$ del portafoglio in cash liquido, azzerando il rischio sistemico che invece colpisce in pieno il Buy & Hold.

### 2. Modulo di Regime Clustering con Machine Learning (GMM / Hidden Markov Models)
Invece di usare un semplice filtro a medie mobili o breadth lineare:
*   **L'affinamento**: Un modello di Machine Learning non supervisionato (es. Gaussian Mixture Models) analizza variabili macro (come l'indice VIX, i volumi di borsa, lo spread dei rendimenti obbligazionari e il momento dell'S&P500) per classificare ogni giorno il mercato in 3 regimi: Toro (Bassa Volatilità), Orso (Alta Volatilità), Laterale (Mean-Reverting).
*   La strategia adatta il suo comportamento: nel regime Laterale disattiva i Transformer di momentum e attiva logiche di oscillazione (RSI/Stochastic), mentre nei regimi direzionali attiva la potenza predittiva delle reti CNN-Transformer.

### 3. Modulo di Arbitraggio Statistico (Pair Trading / Market Neutral)
*   **L'affinamento**: Identificare coppie di titoli storicamente co-integrati (es. Coca-Cola KO e Pepsi PEP, oppure Mastercard MA e Visa V).
*   Quando lo spread di prezzo tra i due diverge eccessivamente dalla media storica, la strategia apre contemporaneamente una posizione LONG sul titolo sottovalutato e una SHORT sul titolo sopravvalutato. Questa operazione è intrinsecamente Market-Neutral (insensibile all'andamento del mercato generale) e genera profitti costanti sia che il mercato salga sia che crolli, battendo sistematicamente il Buy & Hold nelle fasi laterali o correttive.