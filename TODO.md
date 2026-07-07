# 📋 TODO: Architetture Neurali Future per la Previsione del Mercato

Questo documento tiene traccia delle proposte di sviluppo per superare il plateau prestazionale riscontrato con le reti sequenziali standard (CNN-Transformer v10/v11).

---

## 🚀 Proposte di Architetture Neurali & Approcci Avanzati

### 1. Spatio-Temporal Graph Neural Networks (GNN)
*   **Idea**: Modellare il mercato azionario come un grafo interconnesso anziché trattare ogni ticker come un'entità isolata. I nodi sono le azioni e gli archi rappresentano relazioni industriali, correlazioni storiche o appartenenza allo stesso settore.
*   **Componenti**:
    *   [ ] Definizione del grafo dinamico (matrice di adiacenza basata su correlazione dei rendimenti e settori).
    *   [ ] Integrazione di layer **Graph Attention (GAT)** o **Graph Convolutional (GCN)**.
    *   [ ] Modulo temporale (CNN-Transformer o LSTM) accoppiato spazialmente.
*   **Potenziale**: Molto Alto (cattura dinamiche cross-asset e rotazione settoriale).
*   **Complessità**: Alta (richiede la gestione di grafi dinamici e il superamento del rischio di data leakage cross-sectional).

### 2. Deep Reinforcement Learning (DRL) per Portfolio Allocation
*   **Idea**: Cambiare paradigma da *Predict-then-Decide* (classificazione/regressione seguita da euristiche di trading) a *Direct Action*, addestrando un agente neurale a massimizzare direttamente metriche finanziarie.
*   **Componenti**:
    *   [ ] Creazione di un ambiente Gym/Gymnasium custom per la simulazione del trading multivariato.
    *   [ ] Sviluppo di reti Actor-Critic (PPO, DDPG) che emettono in output i pesi ottimali del portafoglio (es. % di allocazione per ogni ticker e cash).
    *   [ ] Funzione di reward basata sullo **Sharpe Ratio** differenziale o **Sortino Ratio**.
*   **Potenziale**: Altissimo (ottimizza congiuntamente previsione, size della posizione e costi di transazione).
*   **Complessità**: Altissima (difficile da convergere, richiede tuning rigoroso degli iperparametri e stabilità dell'ambiente).

### 3. Bayesian Neural Networks (BNN) & Stima dell'Incertezza
*   **Idea**: Quantificare l'incertezza predittiva del modello (epistemica ed aleatoria) per bloccare l'operatività nelle giornate in cui il mercato è privo di una chiara direzione o strutturalmente diverso dai dati storici di addestramento.
*   **Componenti**:
    *   [ ] Implementazione di **Monte Carlo Dropout** (MC Dropout) in fase di inferenza sul modello CNN-Transformer esistente.
    *   [ ] Calcolo della varianza delle predizioni attraverso $N$ passaggi stocastici di forward.
    *   [ ] Integrazione nel modulo di trading per filtrare i segnali con varianza superiore a una soglia prestabilita.
*   **Potenziale**: Alto (riduce significativamente i falsi positivi e preserva il capitale).
*   **Complessità**: Media (facile da implementare partendo dall'infrastruttura attuale, richiede solo più risorse computazionali in fase di backtest/predizione).

### 4. Wavelet Neural Networks (WNN) per Denoising
*   **Idea**: Decomporre la serie storica dei prezzi in diverse frequenze per isolare il trend di fondo dal rumore giornaliero prima di alimentare il modello neurale.
*   **Componenti**:
    *   [ ] Pipeline di pre-elaborazione basata sulla **Trasformata Wavelet Discreta (DWT)** o **MODWT**.
    *   [ ] Separazione del segnale in componenti di approssimazione (trend a lungo termine) e dettagli (micro-oscillazioni).
    *   [ ] Addestramento di reti dedicate per frequenze diverse o rimozione delle componenti di rumore a frequenza più alta.
*   **Potenziale**: Medio-Alto (pulisce il segnale in ingresso riducendo l'overfitting).
*   **Complessità**: Media (richiede l'integrazione di librerie esterne come `PyWavelets` e la gestione del look-ahead bias).

### 5. Mixture of Experts (MoE) per Regimi di Mercato
*   **Idea**: Utilizzare reti specializzate in base alla fase macroeconomica attuale (es. Trend Rialzista, Trend Ribassista, Laterale ad alta volatilità).
*   **Componenti**:
    *   [ ] Sviluppo di un modello di **Regime Clustering** (es. Gaussian Mixture Model o Hidden Markov Model) alimentato da dati macro (VIX, Breadth, Spread dei tassi).
    *   [ ] Creazione di una rete di Gating che decide come pesare l'output di più reti neurali ("Experts") specializzate.
    *   [ ] Addestramento mirato di ciascun expert sulle partizioni storiche del rispettivo regime.
*   **Potenziale**: Altissimo (risolve il problema della non-stazionarietà del mercato).
*   **Complessità**: Medio-Alta (richiede un'architettura modulare e una corretta etichettatura o routing dinamico senza ritardi).

### 6. Modelli State-Space Selettivi (Mamba)
*   **Idea**: Sostituire l'encoder Transformer con blocchi Mamba per estendere drasticamente la finestra di lookback (es. a 120-250 giorni di trading) senza soffrire della complessità computazionale quadratica e del rumore accumulato nell'attention.
*   **Componenti**:
    *   [ ] Integrazione di moduli Mamba in PyTorch.
    *   [ ] Sostituzione dei layer Transformer nei modelli `v10`/`v11`.
    *   [ ] Addestramento con finestre di lookback estese.
*   **Potenziale**: Alto (miglior gestione delle dipendenze di lungo periodo).
*   **Complessità**: Alta (richiede compilatori specifici C++/CUDA o implementazioni PyTorch pure compatibili con Windows, installazione di dipendenze complesse).

### 7. Generazione di Scenari via Generative Models (TimeGAN / Diffusion Models)
*   **Idea**: Addestrare modelli generativi a produrre percorsi di prezzo storici sintetici ma plausibili, da utilizzare come data augmentation per aumentare la robustezza delle reti predittive e testare le strategie in scenari mai visti prima.
*   **Componenti**:
    *   [ ] Implementazione di un modello di diffusione temporale condizionato o TimeGAN.
    *   [ ] Generazione di dataset sintetici massivi.
    *   [ ] Pre-addestramento (Transfer Learning) delle reti predittive esistenti su dati sintetici.
*   **Potenziale**: Medio-Alto (riduce il problema dell'overfitting dovuto a campioni storici limitati).
*   **Complessità**: Alta (l'addestramento stabile di modelli generativi su serie storiche finanziarie è notoriamente difficile e computazionalmente intensivo).

---

## 📈 Proposte di Ottimizzazione Dati & Addestramento (Data-Centric AI)

### 8. Split Temporale con Data di Cutoff Rigida
*   **Idea**: Eliminare la sezione di test interna a `train.py` e addestrare/validare il modello esclusivamente su dati precedenti l'inizio del backtest (es. pre-2024). Questo previene qualsiasi perdita di informazione (*validation leakage*) ed aumenta i dati disponibili per l'addestramento vero e proprio.
*   **Componenti**:
    *   [ ] Filtro di data rigido all'ingestione per l'addestramento.
    *   [ ] Split 80/20 Train/Val sui dati rimanenti.
*   **Potenziale**: Alto (previene leakage da model selection e ottimizza l'uso dei dati storici).
*   **Complessità**: Bassa (richiede modifiche minori alla logica di split in `train.py`).

### 9. Label Smoothing (Regolarizzazione dei Target)
*   **Idea**: Trasformare i target binari rigidi (0/1) in valori sfumati (es. 0.1/0.9) per evitare che il modello diventi sovra-confidente su dati dominati da un forte rumore quotidiano.
*   **Componenti**:
    *   [ ] Modifica della loss function in `AsymmetricMoELoss` per calcolare la cross-entropy con label smoothing.
*   **Potenziale**: Medio-Alto (riduce drasticamente l'overfitting sul rumore e stabilizza le probabilità predette).
*   **Complessità**: Bassa (supportato nativamente da molti moduli PyTorch).

### 10. Pesatura delle Classi (Class Weighting)
*   **Idea**: Calcolare le proporzioni storiche delle giornate UP/DOWN e applicare pesi inversamente proporzionali nella loss function per eliminare il bias indotto dai mercati storicamente rialzisti.
*   **Componenti**:
    *   [ ] Calcolo automatico dei pesi a inizio training.
    *   [ ] Passaggio del parametro `pos_weight` nella loss.
*   **Potenziale**: Medio-Alto (riduce la tendenza del modello a predire ciecamente LONG nei mercati rialzisti).
*   **Complessità**: Bassa.

### 11. Z-Score Cross-Sectionale Giornaliero
*   **Idea**: Normalizzare le feature di ciascun ticker confrontandole con la media e la deviazione standard del mercato in quel preciso istante $t$, anziché utilizzare medie storiche globali.
*   **Componenti**:
    *   [ ] Sviluppo del preprocessore cross-sectionale in `train.py` e `strategy.py`.
*   **Potenziale**: Alto (isola la forza relativa del singolo asset eliminando i trend di marea).
*   **Complessità**: Media (richiede un allineamento preciso delle date a runtime).

### 12. Esclusione dei Ticker a Bassa Liquidità
*   **Idea**: Filtrare il dataset di addestramento escludendo titoli a bassa capitalizzazione o con volumi giornalieri esigui per evitare di inquinare il modello con dinamiche di prezzo erratiche e non liquide.
*   **Componenti**:
    *   [ ] Aggiunta di un filtro sul volume medio o prezzo minimo nel caricamento dati di `train.py`.
*   **Potenziale**: Medio (migliora la pulizia del segnale).
*   **Complessità**: Bassa.

### 13. Learning Rate con Warmup e Cosine Annealing
*   **Idea**: Sostituire lo scheduler del learning rate con una fase iniziale di riscaldamento (Warmup) seguita da una discesa a coseno per convergere a configurazioni dei pesi più stabili e robuste.
*   **Componenti**:
    *   [ ] Sostituzione dello scheduler PyTorch in `model.py`.
*   **Potenziale**: Medio-Alto (migliore convergenza e capacità di generalizzazione).
*   **Complessità**: Bassa.

