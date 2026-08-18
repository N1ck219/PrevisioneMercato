import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union


class TradingEnv(gym.Env):
    """
    Ambiente Gymnasium di Trading finanziario per Reinforcement Learning.
    Supporta il Cross-Asset Multi-Ticker Training passando una lista di DataFrame.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: Union[pd.DataFrame, List[pd.DataFrame]],
        feature_cols: List[str],
        window_size: int = 30,
        initial_balance: float = 100000.0,
        fee_rate: float = 0.0010,
        slippage_rate: float = 0.0005,
        dsr_eta: float = 0.01,
        max_drawdown_penalty_weight: float = 1.0,
        inactivity_penalty_weight: float = 0.1,
        random_start: bool = True,
        episode_length: int = 252,
    ):
        super().__init__()

        if isinstance(df, list):
            self.dfs = [d.reset_index(drop=True) for d in df if len(d) >= (window_size + 2)]
            if not self.dfs:
                raise ValueError("Nessun DataFrame valido fornito nella lista per TradingEnv.")
            self.df = self.dfs[0]
        else:
            self.df = df.reset_index(drop=True)
            self.dfs = [self.df]

        self.feature_cols = feature_cols
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.dsr_eta = dsr_eta
        self.lambda_mdd = max_drawdown_penalty_weight
        self.lambda_inact = inactivity_penalty_weight
        self.random_start = random_start
        self.episode_length = episode_length

        self.num_features = len(feature_cols)

        # Spazio delle Azioni: Target Position Weight w_{t+1} in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Spazio delle Osservazioni: (Window_Size * Num_Features) + 3 (Portfolio State)
        obs_dim = self.window_size * self.num_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_internal_state()

    def _reset_internal_state(self) -> None:
        self.current_step = 0
        self.start_idx = 0
        self.end_idx = 0
        self.position = 0.0  # w_t in [-1.0, 1.0]
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.holding_period = 0

        # Stime ricorsive per Differential Sharpe Ratio (DSR)
        self.dsr_A = 0.0
        self.dsr_B = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._reset_internal_state()

        # Selezione casuale di uno dei ticker dalla lista di DataFrame disponibili
        if len(self.dfs) > 1:
            rand_idx = int(self.np_random.integers(0, len(self.dfs)))
            self.df = self.dfs[rand_idx]
        else:
            self.df = self.dfs[0]

        min_required_len = self.window_size + 2
        if len(self.df) < min_required_len:
            raise ValueError(f"Dataframe troppo corto: {len(self.df)} riga/e, richiesti almeno {min_required_len}")

        if self.random_start and len(self.df) > (self.window_size + self.episode_length):
            max_start = len(self.df) - self.episode_length - 1
            self.start_idx = int(self.np_random.integers(self.window_size, max_start))
        else:
            self.start_idx = self.window_size

        self.end_idx = min(self.start_idx + self.episode_length, len(self.df) - 2)
        self.current_idx = self.start_idx

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_observation(self) -> np.ndarray:
        # Finestra temporale di feature storiche [current_idx - W : current_idx]
        sub_df = self.df.iloc[self.current_idx - self.window_size : self.current_idx]
        feat_matrix = sub_df[self.feature_cols].values.astype(np.float32)

        flat_feats = feat_matrix.flatten()

        # Stato interno del portafoglio: Posizione attuale, Holding period, Current Max Drawdown
        current_dd = (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)
        portfolio_state = np.array(
            [self.position, self.holding_period / 252.0, current_dd],
            dtype=np.float32,
        )

        return np.concatenate([flat_feats, portfolio_state])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        target_position = float(np.clip(action[0], -1.0, 1.0))
        
        # Filtro Isteresi per prevenire micro-riallocazioni continue
        delta_w = target_position - self.position
        if abs(delta_w) < 0.02:
            delta_w = 0.0
            target_position = self.position

        # Timing: Decisione a Close(t), esecuzione a Open(t+1)
        next_row = self.df.iloc[self.current_idx + 1]

        open_next = float(next_row["Open"])
        close_next = float(next_row["Close"])

        # Log-return dell'asset tra Open(t+1) e Close(t+1)
        if open_next > 0:
            asset_return = (close_next / open_next) - 1.0
        else:
            asset_return = 0.0

        # Calcolo Frizioni (Fee di transazione + Slippage)
        transaction_fee = self.fee_rate * abs(delta_w)
        slippage = self.slippage_rate * abs(delta_w)
        total_friction = transaction_fee + slippage

        # Net Return del portafoglio
        net_return = (target_position * asset_return) - total_friction

        # Aggiornamento Capitale ed Equity Peak
        self.equity *= (1.0 + net_return)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Aggiornamento Holding Period
        if delta_w != 0.0:
            self.holding_period = 0
        else:
            self.holding_period += 1

        self.position = target_position

        # Calcolo della Reward bilanciata
        reward = self._calculate_reward(net_return, asset_return)

        # Condizioni di Terminazione (fine episodio o rovina del capitale)
        self.current_idx += 1
        self.current_step += 1

        terminated = bool(self.current_idx >= self.end_idx or self.equity <= 0.2 * self.initial_balance)
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = self._get_info(net_return=net_return, friction=total_friction)

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, net_return: float, asset_return: float) -> float:
        # 1. Differential Sharpe Ratio (DSR) - Moody & Saffell
        delta_A = net_return - self.dsr_A
        delta_B = (net_return ** 2) - self.dsr_B

        self.dsr_A += self.dsr_eta * delta_A
        self.dsr_B += self.dsr_eta * delta_B

        denom = (self.dsr_B - (self.dsr_A ** 2)) ** 1.5
        if denom > 1e-6:
            dsr_reward = (self.dsr_B * delta_A - 0.5 * self.dsr_A * delta_B) / denom
        else:
            dsr_reward = 0.0

        # 2. Maximum Drawdown Penalty (Peak-to-Trough)
        current_dd = (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)
        mdd_penalty = 0.0
        if current_dd > 0.10:
            mdd_penalty = ((current_dd - 0.10) / 0.90) ** 2

        # 3. Opportunity Cost / Inactivity Penalty (Lazy Agent Prevention)
        inactivity_penalty = 0.0
        if abs(self.position) < 0.05 and abs(asset_return) > 0.01:
            inactivity_penalty = abs(asset_return)

        total_reward = (
            net_return
            + 0.1 * dsr_reward
            - self.lambda_mdd * mdd_penalty
            - self.lambda_inact * inactivity_penalty
        )

        return float(np.clip(total_reward, -10.0, 10.0))

    def _get_info(self, net_return: float = 0.0, friction: float = 0.0) -> Dict[str, Any]:
        current_dd = (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)
        return {
            "equity": float(self.equity),
            "peak_equity": float(self.peak_equity),
            "position": float(self.position),
            "drawdown": float(current_dd),
            "net_return": float(net_return),
            "friction": float(friction),
            "current_idx": int(self.current_idx),
        }
