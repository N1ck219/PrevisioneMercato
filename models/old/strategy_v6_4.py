import os
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from strategies.base_strategy import BaseStrategy
from core.config import MODELS_DIR, DB_TRADES_V64, TARGET_TICKERS_AZIONARIO, ALPACA_API_KEY_6_4, ALPACA_SECRET_KEY_6_4, MACRO_LABELS_ORDERED
from core.models.model_factory import get_model
from core.data.features import FeatureEngine

LOOKBACK_DAYS = 60
MAX_ALLOCATION_PER_ASSET = 0.25
RISCHIO_BERSAGLIO_PCT = 0.015

alpaca = None
if ALPACA_API_KEY_6_4 and ALPACA_SECRET_KEY_6_4:
    try: alpaca = TradingClient(ALPACA_API_KEY_6_4, ALPACA_SECRET_KEY_6_4, paper=True)
    except: pass


class StrategyV64(BaseStrategy):
    bot_name = "V6.4 APEX FUND (Macro Oracle)"
    db_trades_path = DB_TRADES_V64
    model_path = os.path.join(MODELS_DIR, "base_brain_v5_0.h5")

    def setup_trades_db(self):
        self.conn_trades.execute('''CREATE TABLE IF NOT EXISTS state_v64 (Ticker TEXT PRIMARY KEY, entry REAL, atr_entry REAL, highest REAL, lowest REAL, invested REAL, pos INTEGER, half_sold INTEGER, qty INTEGER)''')
        for t in TARGET_TICKERS_AZIONARIO:
            if not self.conn_trades.execute("SELECT count(*) FROM state_v64 WHERE Ticker=?", (t,)).fetchone()[0]:
                self.conn_trades.execute("INSERT INTO state_v64 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (t, 0.0, 0.0, 0.0, 999999.0, 0.0, 0, 0, 0))
        self.conn_trades.commit()

    def _load_state(self, ticker):
        row = self.conn_trades.execute("SELECT entry, atr_entry, highest, lowest, invested, pos, half_sold, qty FROM state_v64 WHERE Ticker=?", (ticker,)).fetchone()
        return {'entry': row[0], 'atr_entry': row[1], 'highest': row[2], 'lowest': row[3], 'invested': row[4], 'pos': row[5], 'half_sold': bool(row[6]), 'qty': row[7]}

    def _save_state(self, ticker, s):
        self.conn_trades.execute("UPDATE state_v64 SET entry=?, atr_entry=?, highest=?, lowest=?, invested=?, pos=?, half_sold=?, qty=? WHERE Ticker=?", (s['entry'], s['atr_entry'], s['highest'], s['lowest'], s['invested'], s['pos'], int(s['half_sold']), s['qty'], ticker))
        self.conn_trades.commit()

    def execute(self):
        df_base = FeatureEngine.process_stock_features(self.get_market_data("AAPL"), self.macro)
        t_feat, m_feat = FeatureEngine.extract_features(df_base, macro_labels=MACRO_LABELS_ORDERED)
        scaler_t = StandardScaler().fit(t_feat)
        scaler_m = StandardScaler().fit(m_feat)

        tf.keras.backend.clear_session()
        model = get_model("6.4", self.model_path, shape_t=(LOOKBACK_DAYS, t_feat.shape[1]), shape_m=(LOOKBACK_DAYS, m_feat.shape[1]))

        try:
            acc = alpaca.get_account()
            portfolio_value = float(acc.equity)
            cash_disponibile = float(acc.buying_power)
        except:
            portfolio_value = 10000.0
            cash_disponibile = 10000.0

        app_state = {t: self._load_state(t) for t in TARGET_TICKERS_AZIONARIO}
        opportunities = []

        for ticker in tqdm(TARGET_TICKERS_AZIONARIO, desc="🎯 Generazione Segnali"):
            df_target = FeatureEngine.process_stock_features(self.get_market_data(ticker), self.macro)
            if len(df_target) < LOOKBACK_DAYS + 1: continue

            prezzo_oggi = df_target['prezzo'].iloc[-1]
            atr_oggi = max(df_target['ATRr_14'].iloc[-1], 1e-9)
            trend_rialzista = df_target['Dist_SMA200'].iloc[-1] > 0

            t_feat_raw, m_feat_raw = FeatureEngine.extract_features(df_target, macro_labels=MACRO_LABELS_ORDERED)
            X_t = np.array([scaler_t.transform(t_feat_raw)[-LOOKBACK_DAYS:]], dtype=np.float32)
            X_m = np.array([scaler_m.transform(m_feat_raw)[-LOOKBACK_DAYS:]], dtype=np.float32)

            delta_p = model.predict([X_t, X_m], verbose=0)[0][0] - 0.50
            state = app_state[ticker]

            if state['pos'] != 0:
                rend_pct = ((prezzo_oggi - state['entry']) / state['entry']) * (1 if state['pos'] == 1 else -1)
                if rend_pct <= -0.05:
                    if alpaca:
                        try: alpaca.close_position(ticker); alpaca.cancel_orders(symbol=ticker)
                        except: pass
                    state['pos'] = 0; state['invested'] = 0.0; state['qty'] = 0

            if state['pos'] == 0 and abs(delta_p) > 0.05:
                if (delta_p > 0 and trend_rialzista) or (delta_p < 0 and not trend_rialzista):
                    opportunities.append({'ticker': ticker, 'delta': delta_p, 'prezzo': prezzo_oggi, 'atr': atr_oggi, 'trend': trend_rialzista})

        operazioni_fatte = []
        opportunities.sort(key=lambda x: abs(x['delta']), reverse=True)

        for opp in opportunities[:3]:
            t = opp['ticker']; prezzo = opp['prezzo']; p = app_state[t]
            distanza_sl_pct = (4.2 * opp['atr']) / prezzo
            investimento = min(portfolio_value * MAX_ALLOCATION_PER_ASSET, (portfolio_value * RISCHIO_BERSAGLIO_PCT) / distanza_sl_pct) * min(1.0, abs(opp['delta']) / 0.35)
            if investimento > cash_disponibile: investimento = cash_disponibile * 0.95
            qty = int(investimento // prezzo)
            if qty <= 0: continue

            try:
                if opp['delta'] > 0 and p['pos'] <= 0:
                    if alpaca:
                        if p['pos'] == -1: alpaca.close_position(t); alpaca.cancel_orders(symbol=t); time.sleep(1)
                        alpaca.submit_order(MarketOrderRequest(symbol=t, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                    cash_disponibile -= investimento
                    p['entry'] = ((p['entry'] * p['invested']) + (prezzo * investimento)) / (p['invested'] + investimento) if p['invested'] > 0 else prezzo
                    p['atr_entry'] = opp['atr']; p['highest'] = max(p['highest'], prezzo); p['half_sold'] = False
                    p['invested'] += investimento; p['pos'] = 1; p['qty'] += qty
                    operazioni_fatte.append(f"🟢 <b>{t}</b>: BUY LONG (${int(investimento)})")

                elif opp['delta'] < 0 and p['pos'] >= 0:
                    if alpaca:
                        if p['pos'] == 1: alpaca.close_position(t); alpaca.cancel_orders(symbol=t); time.sleep(1)
                        alpaca.submit_order(MarketOrderRequest(symbol=t, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                    cash_disponibile -= investimento
                    p['entry'] = ((p['entry'] * p['invested']) + (prezzo * investimento)) / (p['invested'] + investimento) if p['invested'] > 0 else prezzo
                    p['atr_entry'] = opp['atr']; p['lowest'] = min(p['lowest'], prezzo); p['half_sold'] = False
                    p['invested'] += investimento; p['pos'] = -1; p['qty'] += qty
                    operazioni_fatte.append(f"🔴 <b>{t}</b>: SELL SHORT (${int(investimento)})")

                self._save_state(t, p)
            except Exception as e: print(f"❌ Errore Invio Ordine {t}: {e}")

        capitale_investito = sum([s['invested'] for s in app_state.values()])
        esposizione_pct = (capitale_investito / portfolio_value) * 100 if portfolio_value > 0 else 0

        trades_str = "\n".join(operazioni_fatte) if operazioni_fatte else ""
        self.set_report_data(
            balance_str=f"${portfolio_value:,.2f}",
            extra_str=f"Esposizione Mercato: {esposizione_pct:.1f}%",
            trades_str=trades_str
        )


def run():
    StrategyV64().run()

if __name__ == "__main__":
    run()