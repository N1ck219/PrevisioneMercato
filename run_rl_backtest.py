import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Assicura che la directory radice sia nel PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent))

import config
from train_rl import engineer_features
from models.rl.trading_env import TradingEnv
from models.rl.rl_model import RLTradingModel


def calculate_metrics(equity_curve: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
    """
    Calcola le metriche quantitative ufficiali per la valutazione delle performance.
    """
    if len(equity_curve) < 2:
        return {}

    total_return = (equity_curve[-1] / equity_curve[0]) - 1.0
    cagr = ((equity_curve[-1] / equity_curve[0]) ** (252.0 / max(len(equity_curve), 1))) - 1.0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns) + 1e-8
    sharpe = (mean_ret / std_ret) * np.sqrt(252)

    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) + 1e-8 if len(downside_returns) > 0 else 1e-8
    sortino = (mean_ret / downside_std) * np.sqrt(252)

    # Peak-to-Trough Drawdown
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / peaks
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    calmar = (cagr / max_drawdown) if max_drawdown > 1e-4 else 0.0

    trades = returns[returns != 0]
    win_rate = (np.sum(trades > 0) / len(trades)) * 100.0 if len(trades) > 0 else 0.0

    pos_gains = np.sum(returns[returns > 0])
    neg_losses = np.abs(np.sum(returns[returns < 0]))
    profit_factor = (pos_gains / neg_losses) if neg_losses > 1e-8 else np.nan

    return {
        "Total Return (%)": total_return * 100.0,
        "CAGR (%)": cagr * 100.0,
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Calmar Ratio": float(calmar),
        "Max Drawdown (%)": max_drawdown * 100.0,
        "Win Rate (%)": win_rate,
        "Profit Factor": float(profit_factor) if not np.isnan(profit_factor) else 0.0,
    }


def run_env_backtest(df: pd.DataFrame, feature_cols: List[str], model: RLTradingModel = None, strategy: str = "RL") -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Esegue un backtest sequenziale su un dato DataFrame.
    """
    env = TradingEnv(
        df=df,
        feature_cols=feature_cols,
        window_size=30,
        random_start=False,
        episode_length=len(df),
    )

    obs, _ = env.reset()
    equity_curve = [env.equity]
    returns = []
    positions = []

    done = False
    idx = 0

    while not done and idx < len(df) - 32:
        if strategy == "RL" and model is not None:
            norm_obs = model.vec_env.normalize_obs(obs) if model.vec_env is not None else obs
            action, _ = model.model.predict(norm_obs, deterministic=True)
        elif strategy == "BUY_AND_HOLD":
            action = np.array([1.0], dtype=np.float32)
        elif strategy == "SMA_CROSSOVER":
            # SMA 50 / 200 crossover rule
            curr_row = df.iloc[env.current_idx]
            close = curr_row["Close"]
            sma50 = df.iloc[max(0, env.current_idx-50):env.current_idx+1]["Close"].mean()
            sma200 = df.iloc[max(0, env.current_idx-200):env.current_idx+1]["Close"].mean()
            action = np.array([1.0 if sma50 >= sma200 else -1.0], dtype=np.float32)
        else:
            action = np.array([0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        equity_curve.append(info["equity"])
        returns.append(info["net_return"])
        positions.append(info["position"])

        done = terminated or truncated
        idx += 1

    return np.array(equity_curve), np.array(returns), positions


def run_friction_stress_test(df: pd.DataFrame, feature_cols: List[str], model: RLTradingModel):
    """
    Stress testing a livelli crescenti di commissioni e slippage.
    """
    print("\n" + "-" * 70)
    print("      EXECUTION FRICTION ANALYSIS (STRESS TESTING SLIPPAGE/FEE)")
    print("-" * 70)

    fee_rates = [0.0005, 0.0010, 0.0020]
    slippage_rates = [0.0005, 0.0010, 0.0025]

    results = []
    for fee in fee_rates:
        for slip in slippage_rates:
            env = TradingEnv(
                df=df,
                feature_cols=feature_cols,
                window_size=30,
                fee_rate=fee,
                slippage_rate=slip,
                random_start=False,
                episode_length=len(df),
            )
            obs, _ = env.reset()
            eq_list, ret_list = [env.equity], []
            done = False
            while not done:
                norm_obs = model.vec_env.normalize_obs(obs) if model.vec_env is not None else obs
                action, _ = model.model.predict(norm_obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                eq_list.append(info["equity"])
                ret_list.append(info["net_return"])
                done = terminated or truncated

            m = calculate_metrics(np.array(eq_list), np.array(ret_list))
            results.append({
                "Fee (bps)": int(fee * 10000),
                "Slippage (bps)": int(slip * 10000),
                "Sharpe": round(m.get("Sharpe Ratio", 0.0), 2),
                "Max DD (%)": round(m.get("Max Drawdown (%)", 0.0), 2),
                "Return (%)": round(m.get("Total Return (%)", 0.0), 2),
            })

    stress_df = pd.DataFrame(results)
    print(stress_df.to_string(index=False))


def main():
    print("=" * 70)
    print("     VALUTAZIONE OUT-OF-SAMPLE E BACKTEST DEL MODELLO RL")
    print("=" * 70)

    ticker = "SPY"
    model_path = config.BASE_DIR / "models" / "rl_ppo_model"

    # 1. Caricamento dati e suddivisione per il ticker di test (SPY)
    from database.db_manager import DBManager
    db = DBManager()
    query = f"SELECT timestamp as Date, open as Open, high as High, low as Low, close as Close, volume as Volume FROM ohlcv WHERE ticker='{ticker}' ORDER BY timestamp ASC"
    df_raw = db.execute_query(query)
    if df_raw.empty:
        import yfinance as yf
        df_raw = yf.download(ticker, period="10y", interval="1d").reset_index()
        df_raw.rename(columns={"Date": "Date", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}, inplace=True)
    else:
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])

    df_feat, feature_cols = engineer_features(df_raw)
    n = len(df_feat)
    val_end = int(n * 0.85)
    test_df = df_feat.iloc[val_end:].copy().reset_index(drop=True)

    # 2. Caricamento modello RL (se esistente)
    rl_model = RLTradingModel(algorithm="PPO", window_size=30)
    if (Path(str(model_path) + ".zip")).exists():
        rl_model.load(str(model_path))
    else:
        print(f"[WARNING] Modello salvato non trovato in {model_path}.zip. Eseguire prima `python train_rl.py`.")
        print("[INFO] Verrà eseguita la valutazione sui benchmark di riferimento.")
        rl_model = None

    # 3. Esecuzione Backtest Out-of-Sample (Test Set: 15%)
    print(f"\n[INFO] Avvio Backtest Out-of-Sample per {ticker} su {len(test_df)} giorni di borsa...")

    eq_rl, ret_rl, _ = run_env_backtest(test_df, feature_cols, model=rl_model, strategy="RL" if rl_model else "NONE")
    eq_bah, ret_bah, _ = run_env_backtest(test_df, feature_cols, strategy="BUY_AND_HOLD")
    eq_sma, ret_sma, _ = run_env_backtest(test_df, feature_cols, strategy="SMA_CROSSOVER")

    metrics_rl = calculate_metrics(eq_rl, ret_rl) if rl_model else {}
    metrics_bah = calculate_metrics(eq_bah, ret_bah)
    metrics_sma = calculate_metrics(eq_sma, ret_sma)

    # 4. Stampa Report Comparativo
    print("\n" + "=" * 70)
    print("              REPORT COMPARATIVO PRESTAZIONI OUT-OF-SAMPLE")
    print("=" * 70)

    comparison_dict = {
        "Benchmark: Buy & Hold (SPY)": metrics_bah,
        "Benchmark: SMA Crossover (50/200)": metrics_sma,
    }
    if rl_model:
        comparison_dict["Modello RL (PPO Agent)"] = metrics_rl

    comp_df = pd.DataFrame(comparison_dict).T
    print(comp_df.to_string())

    # 5. Friction Sensitivity Stress Testing
    if rl_model:
        run_friction_stress_test(test_df, feature_cols, rl_model)

    print("\n" + "=" * 70)
    print("                    BACKTEST COMPLETATO CON SUCCESSO")
    print("=" * 70)


if __name__ == "__main__":
    main()
