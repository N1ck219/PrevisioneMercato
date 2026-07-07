import logging
import os
import math
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NeuralNetworkV4")


class PositionalEncoding(nn.Module):
    """
    Positional Encoding per aggiungere informazioni sull'ordine temporale
    all'input del Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Gestione d_model dispari
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model // 2]
            
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TemporalTransformerModel(nn.Module):
    """
    Architettura neurale basata su Transformer Encoder e Multi-Head Attention
    per serie storiche finanziarie (v4).
    """
    def __init__(
        self, 
        input_dim: int, 
        lookback: int, 
        d_model: int = 64, 
        nhead: int = 4, 
        num_layers: int = 2, 
        dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        self.lookback = lookback
        
        # Proiezione lineare delle feature in ingresso a dimensione d_model
        self.linear_in = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=lookback + 10)
        
        # Layer Encoder del Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Testa decisionale Fully Connected
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)  # Logit grezzo per BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, input_dim)
        x_proj = self.linear_in(x)  # (batch_size, seq_len, d_model)
        x_pe = self.pos_encoder(x_proj)  # (batch_size, seq_len, d_model)
        
        # Transformer pass
        trans_out = self.transformer_encoder(x_pe)  # (batch_size, seq_len, d_model)
        
        # Pooling temporale: prendiamo l'ultimo elemento (lo stato corrente)
        last_step = trans_out[:, -1, :]  # (batch_size, d_model)
        
        logits = self.fc(last_step)  # (batch_size, 1)
        return logits


class NeuralNetworkV4(BaseModel):
    """
    Modello wrapper NeuralNetworkV4 che implementa BaseModel per l'architettura a Transformer.
    Gestisce l'addestramento sequenziale batch-by-batch su GPU/CPU per prevenire CUDA OOM.
    """
    def __init__(
        self, 
        input_dim: int, 
        lookback: int = 30, 
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        lr: float = 0.0005, 
        weight_decay: float = 1e-4
    ) -> None:
        self.input_dim = input_dim
        self.lookback = lookback
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = TemporalTransformerModel(
            input_dim=self.input_dim,
            lookback=self.lookback,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        ).to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        logger.info(f"Modello PyTorch Transformer v4 inizializzato (d_model={d_model}, nhead={nhead}, layers={num_layers}). Device: {self.device}")

    def train(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame], 
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        epochs: int = 120,
        batch_size: int = 512,
        early_stopping_rounds: int = 15,
        verbose: bool = True
    ) -> Dict[str, Any]:
        
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_tr = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(X_tr, -10.0, 10.0, out=X_tr)
        
        X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = X_val.values if isinstance(X_val, pd.DataFrame) else np.array(X_val)
            y_v = y_val.values if isinstance(y_val, pd.Series) else np.array(y_val)
            X_v = np.nan_to_num(X_v, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(X_v, -10.0, 10.0, out=X_v)
            X_v_tensor = torch.tensor(X_v, dtype=torch.float32)
            y_v_tensor = torch.tensor(y_v, dtype=torch.float32).unsqueeze(1)
            
        history = {"train_loss": [], "val_loss": []}
        
        best_val_loss = float("inf")
        best_model_state = None
        no_improvement_epochs = 0
        
        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
            
        epoch_iterator = range(1, epochs + 1)
        if HAS_TQDM and verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Progresso Totale v4", leave=True)
            
        num_samples = len(X_tr_tensor)
        
        for epoch in epoch_iterator:
            self.model.train()
            epoch_loss = 0.0
            
            indices = torch.randperm(num_samples)
            X_tr_shuffled = X_tr_tensor[indices]
            y_tr_shuffled = y_tr_tensor[indices]
            
            batch_indices = list(range(0, num_samples, batch_size))
            if HAS_TQDM and verbose:
                batch_iterator = tqdm(
                    batch_indices, 
                    desc=f"Epoca v4 {epoch:03d}/{epochs}", 
                    leave=True
                )
            else:
                batch_iterator = batch_indices
                
            for i in batch_iterator:
                batch_X = X_tr_shuffled[i:i+batch_size].to(self.device)
                batch_y = y_tr_shuffled[i:i+batch_size].to(self.device)
                
                if len(batch_X) <= 1:
                    continue
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_X)
                
                if HAS_TQDM and verbose:
                    batch_iterator.set_postfix_str(f"Loss: {loss.item():.4f}")
                
            train_loss = epoch_loss / num_samples
            history["train_loss"].append(train_loss)
            
            val_loss = None
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_loss_sum = 0.0
                    num_val_samples = len(X_v_tensor)
                    for j in range(0, num_val_samples, batch_size):
                        val_batch_X = X_v_tensor[j:j+batch_size].to(self.device)
                        val_batch_y = y_v_tensor[j:j+batch_size].to(self.device)
                        val_pred = self.model(val_batch_X)
                        val_loss_sum += self.criterion(val_pred, val_batch_y).item() * len(val_batch_X)
                    val_loss = val_loss_sum / num_val_samples
                    history["val_loss"].append(val_loss)
            
            if verbose:
                if HAS_TQDM:
                    val_str = f" - Val Loss: {val_loss:.6f}" if has_val else ""
                    batch_iterator.set_postfix_str(f"Train Loss: {train_loss:.6f}{val_str}")
                elif epoch % 10 == 0 or epoch == 1 or epoch == epochs or no_improvement_epochs == early_stopping_rounds - 1:
                    val_str = f" - Val Loss: {val_loss:.6f}" if has_val else ""
                    logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}{val_str}")
                
            if has_val:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                    
                if no_improvement_epochs >= early_stopping_rounds:
                    if HAS_TQDM and verbose:
                        tqdm.write(f"Early stopping scattato all'epoca {epoch}. Ripristino dei migliori pesi della validazione.")
                    else:
                        logger.info(f"Early stopping scattato all'epoca {epoch}. Ripristino dei migliori pesi della validazione.")
                    self.model.load_state_dict(best_model_state)
                    break
                    
        return history

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(X_arr, -10.0, 10.0, out=X_arr)
        
        self.model.eval()
        probabilities = []
        batch_size = 1024
        
        with torch.no_grad():
            for i in range(0, len(X_arr), batch_size):
                batch_X = torch.tensor(X_arr[i:i+batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(batch_X)
                probs = torch.sigmoid(logits)
                probabilities.append(probs.cpu().numpy().flatten())
                
        return np.concatenate(probabilities)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = {
            "input_dim": self.input_dim,
            "lookback": self.lookback,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(state, filepath)
        logger.info(f"Modello v4 (Transformer) salvato all'indirizzo: {filepath}")

    def load(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File dei pesi v4 non trovato all'indirizzo: {filepath}")
            
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.input_dim = state.get("input_dim", self.input_dim)
        self.lookback = state.get("lookback", self.lookback)
        self.d_model = state.get("d_model", self.d_model)
        self.nhead = state.get("nhead", self.nhead)
        self.num_layers = state.get("num_layers", self.num_layers)
        self.lr = state.get("lr", self.lr)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        
        self.model = TemporalTransformerModel(
            input_dim=self.input_dim,
            lookback=self.lookback,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        ).to(self.device)
        
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
            
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if "optimizer_state_dict" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Impossibile ripristinare lo stato dell'optimizer v4: {e}.")
        
        logger.info(f"Modello v4 (Transformer) caricato con successo da: {filepath}")
