import os
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union, List
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from models.base_model import BaseModel
from models.rl.trading_env import TradingEnv


class RLTradingModel(BaseModel):
    """
    Wrapper per algoritmi di Reinforcement Learning (PPO / SAC di Stable-Baselines3)
    conforme all'interfaccia BaseModel del progetto.
    """

    def __init__(
        self,
        algorithm: str = "PPO",
        window_size: int = 30,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        ent_coef: float = 0.01,
        total_timesteps: int = 200_000,
        random_seed: int = 42,
        device: str = "auto",
        n_envs: int = 8,
        **model_kwargs: Any
    ):
        super().__init__()
        self.algorithm_name = algorithm.upper()
        self.window_size = window_size
        self.policy = policy
        self.learning_rate = learning_rate
        self.ent_coef = ent_coef
        self.total_timesteps = total_timesteps
        self.random_seed = random_seed
        self.device = device
        self.n_envs = n_envs
        self.model_kwargs = model_kwargs

        self.model = None
        self.vec_env = None
        self.feature_cols: List[str] = []

    def _make_env(self, df: Union[pd.DataFrame, List[pd.DataFrame]], feature_cols: List[str], n_envs: int = 8, random_start: bool = True):
        """Crea l'ambiente vettoriale per Stable-Baselines3 (singolo o parallelo)."""
        def _make_env_fn(rank: int):
            def _env_fn():
                return TradingEnv(
                    df=df,
                    feature_cols=feature_cols,
                    window_size=self.window_size,
                    random_start=random_start,
                )
            return _env_fn

        if n_envs > 1:
            env_fns = [_make_env_fn(i) for i in range(n_envs)]
            return SubprocVecEnv(env_fns)
        else:
            return DummyVecEnv([_make_env_fn(0)])

    def train(
        self,
        X_train: Union[pd.DataFrame, List[pd.DataFrame]],
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Addestra l'agente RL su X_train (singolo DataFrame o Lista di DataFrame multi-ticker).
        """
        if isinstance(X_train, list):
            sample_df = X_train[0]
        elif isinstance(X_train, pd.DataFrame):
            sample_df = X_train
        else:
            raise TypeError("RLTradingModel richiede un pandas DataFrame o una Lista di DataFrames.")

        self.feature_cols = kwargs.get("feature_cols", [c for c in sample_df.columns if c not in ["Open", "High", "Low", "Close", "Volume", "Ticker", "Date"]])
        total_timesteps = kwargs.get("total_timesteps", self.total_timesteps)
        n_envs = kwargs.get("n_envs", self.n_envs)

        raw_env = self._make_env(X_train, self.feature_cols, n_envs=n_envs, random_start=True)
        self.vec_env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

        if self.algorithm_name == "PPO":
            self.model = PPO(
                policy=self.policy,
                env=self.vec_env,
                learning_rate=self.learning_rate,
                ent_coef=self.ent_coef,
                seed=self.random_seed,
                device=self.device,
                verbose=1,
                **self.model_kwargs
            )
        elif self.algorithm_name == "SAC":
            self.model = SAC(
                policy=self.policy,
                env=self.vec_env,
                learning_rate=self.learning_rate,
                ent_coef=self.ent_coef,
                seed=self.random_seed,
                device=self.device,
                verbose=1,
                **self.model_kwargs
            )
        else:
            raise ValueError(f"Algoritmo '{self.algorithm_name}' non supportato. Scegliere 'PPO' o 'SAC'.")

        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)

        return {
            "algorithm": self.algorithm_name,
            "total_timesteps": total_timesteps,
            "num_features": len(self.feature_cols),
        }

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Genera l'allocazione del capitale (target position weight -1.0 a 1.0)
        per ogni punto temporale nel DataFrame X.
        """
        if self.model is None or self.vec_env is None:
            raise RuntimeError("Il modello non è stato addestrato o caricato.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("RLTradingModel richiede un pandas DataFrame per le predizioni.")

        eval_env = TradingEnv(
            df=X,
            feature_cols=self.feature_cols,
            window_size=self.window_size,
            random_start=False,
        )

        obs, _ = eval_env.reset()
        actions = []

        done = False
        while not done:
            norm_obs = self.vec_env.normalize_obs(obs)
            action, _ = self.model.predict(norm_obs, deterministic=True)
            actions.append(float(action[0]))

            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

        return np.array(actions, dtype=np.float32)

    def save(self, filepath: str) -> None:
        """Salva il modello SB3 e lo stato di VecNormalize."""
        if self.model is None:
            raise RuntimeError("Nessun modello da salvare.")

        base_path = Path(filepath)
        model_path = base_path.with_suffix(".zip")
        vec_path = base_path.with_name(base_path.stem + "_vecnorm.pkl")

        self.model.save(str(model_path))

        if self.vec_env is not None:
            with open(vec_path, "wb") as f:
                pickle.dump(
                    {
                        "obs_rms": self.vec_env.obs_rms,
                        "ret_rms": self.vec_env.ret_rms,
                        "feature_cols": self.feature_cols,
                        "window_size": self.window_size,
                        "algorithm_name": self.algorithm_name,
                    },
                    f,
                )
        print(f"[INFO] Modello RL salvato in: {model_path} e {vec_path}")

    def load(self, filepath: str) -> None:
        """Carica il modello SB3 e lo stato di VecNormalize."""
        base_path = Path(filepath)
        model_path = base_path.with_suffix(".zip")
        vec_path = base_path.with_name(base_path.stem + "_vecnorm.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Impossibile trovare il modello salvato: {model_path}")

        if vec_path.exists():
            with open(vec_path, "rb") as f:
                saved_state = pickle.load(f)
                self.feature_cols = saved_state.get("feature_cols", [])
                self.window_size = saved_state.get("window_size", self.window_size)
                self.algorithm_name = saved_state.get("algorithm_name", self.algorithm_name)

        if self.algorithm_name == "PPO":
            self.model = PPO.load(str(model_path))
        elif self.algorithm_name == "SAC":
            self.model = SAC.load(str(model_path))
        else:
            raise ValueError(f"Algoritmo sconosciuto: {self.algorithm_name}")

        print(f"[INFO] Modello RL caricato con successo da: {model_path}")
