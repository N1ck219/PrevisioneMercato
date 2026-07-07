import logging
import os
import math
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MoEModelV1")


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
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model // 2]
            
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CrossFeatureAttention(nn.Module):
    """
    Self-Attention applicata lungo l'asse delle feature (canali)
    per catturare le interazioni incrociate tra gli indicatori ad ogni timestep.
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.q = nn.Linear(input_dim, input_dim)
        self.k = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, input_dim)
        self.scale = math.sqrt(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        
        Q = self.q(x)  # (batch_size, seq_len, input_dim)
        K = self.k(x)  # (batch_size, seq_len, input_dim)
        V = self.v(x)  # (batch_size, seq_len, input_dim)
        
        # Calcoliamo i pesi di attenzione lungo l'asse delle feature
        scores = torch.matmul(Q.unsqueeze(-1), K.unsqueeze(-2)) / self.scale  # (batch_size, seq_len, input_dim, input_dim)
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, V.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len, input_dim)
        return x + out


class TemporalAttentionPooling(nn.Module):
    """
    Modulo di Attention Pooling temporale per aggregare l'output del Transformer.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        attn_weights = self.attn(x)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalizza lungo la dimensione seq_len
        pooled = torch.sum(x * attn_weights, dim=1)  # Somma pesata: (batch_size, d_model)
        return pooled


class CNNTransformerExpert(nn.Module):
    """
    Rete neurale esperta basata sull'architettura v11.
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
        self.d_model = d_model
        
        # 1. Modulo Cross-Feature Attention
        self.feature_attention = CrossFeatureAttention(input_dim)
        
        # 2. Blocco Convoluzionale Parallelo Multi-Scale (Inception-like)
        d_sub = d_model // 3
        d_last = d_model - 2 * d_sub
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=d_sub, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(d_sub),
            nn.Dropout(dropout_rate)
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=d_sub, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(d_sub),
            nn.Dropout(dropout_rate)
        )
        self.branch9 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=d_last, kernel_size=9, padding=4),
            nn.GELU(),
            nn.BatchNorm1d(d_last),
            nn.Dropout(dropout_rate)
        )
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=lookback + 10)
        
        # 4. Layer Encoder del Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Modulo di Temporal Attention Pooling
        self.attention_pooling = TemporalAttentionPooling(d_model)
        
        # 6. Testa decisionale Fully Connected
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)  # Logit grezzo
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (batch_size, seq_len, input_dim)
        x_f_attn = self.feature_attention(x)
        x_conv = x_f_attn.transpose(1, 2)
        
        out3 = self.branch3(x_conv)
        out5 = self.branch5(x_conv)
        out9 = self.branch9(x_conv)
        
        x_features = torch.cat([out3, out5, out9], dim=1)  # (batch_size, d_model, seq_len)
        x_transformer_in = x_features.transpose(1, 2)
        
        x_pe = self.pos_encoder(x_transformer_in)
        trans_out = self.transformer_encoder(x_pe)
        
        attn_pooled = self.attention_pooling(trans_out)  # (batch_size, d_model)
        last_step = trans_out[:, -1, :]  # (batch_size, d_model)
        
        pooled = attn_pooled + last_step  # (batch_size, d_model)
        logits = self.fc(pooled)  # (batch_size, 1)
        return logits


class RegimeGatingNetwork(nn.Module):
    """
    Rete neurale di Gating basata su MLP per classificare le macro feature correnti nei 3 regimi.
    """
    def __init__(self, macro_dim: int, num_regimes: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(macro_dim, 32),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_regimes)  # Output logits per i regimi [Bull, Bear, Lateral]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegimeGatedMoEModel(nn.Module):
    """
    Modello principale Regime-Gated Mixture of Experts (moe_v1).
    Fonde le predizioni di 3 esperti basandosi sui pesi emessi dalla rete di Gating.
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
        self.input_dim = input_dim
        self.lookback = lookback
        
        # 3 Esperti specializzati (0: Bull, 1: Bear/Crash, 2: Lateral)
        self.experts = nn.ModuleList([
            CNNTransformerExpert(input_dim, lookback, d_model, nhead, num_layers, dropout_rate)
            for _ in range(3)
        ])
        
        # Gating Network alimentata dalle 9 macro feature di mercato (VIX, SPY, QQQ, ecc.)
        self.gating = RegimeGatingNetwork(macro_dim=9, num_regimes=3, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_dim)
        # Estraiamo le 9 feature macroeconomiche dall'ultimo timestep della sequenza (colonne da 16 a 24)
        x_macro = x[:, -1, 16:]
        
        # Otteniamo i pesi normalizzati (Softmax) per gli esperti dal gating network
        gating_logits = self.gating(x_macro)  # (batch_size, 3)
        gating_weights = torch.softmax(gating_logits, dim=-1)  # (batch_size, 3)
        
        # Calcoliamo la predizione di ciascun esperto
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch_size, 3, 1)
        
        # Somma pesata delle predizioni in base ai pesi del gating
        final_logits = torch.sum(expert_logits * gating_weights.unsqueeze(-1), dim=1)  # (batch_size, 1)
        
        return final_logits, gating_logits


class AsymmetricMoELoss(nn.Module):
    """
    Loss congiunta per il modello MoE:
    1. Loss Predittiva: BCE pesata sul profitto ed asimmetrica per i crash.
    2. Loss Gating: CrossEntropy rispetto alla classificazione deterministica dei regimi.
    """
    def __init__(
        self, 
        alpha: float = 50.0, 
        penalty_factor: float = 1.5, 
        lambda_gating: float = 0.5
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.penalty_factor = penalty_factor
        self.lambda_gating = lambda_gating

    def forward(
        self, 
        logits: torch.Tensor, 
        gating_logits: torch.Tensor,
        targets: torch.Tensor, 
        tomorrow_returns: torch.Tensor,
        is_crash: torch.Tensor,
        true_regimes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Loss Predittiva Asimmetrica
        loss_pred_raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weights = 1.0 + self.alpha * torch.abs(tomorrow_returns)
        probs = torch.sigmoid(logits)
        asymmetric_penalty = 1.0 + (self.penalty_factor - 1.0) * is_crash * (1.0 - targets) * probs
        loss_pred = (loss_pred_raw * weights * asymmetric_penalty).mean()
        
        # 2. Loss di Gating (spinge la rete a imitare i regimi storici noti)
        loss_gating = nn.functional.cross_entropy(gating_logits, true_regimes.squeeze(-1).long())
        
        # 3. Loss Totale
        loss_total = loss_pred + self.lambda_gating * loss_gating
        
        return loss_total, loss_pred, loss_gating


class MoEModelV1(BaseModel):
    """
    Wrapper BaseModel per il modello Regime-Gated MoE (moe_v1).
    """
    def __init__(
        self, 
        input_dim: int, 
        lookback: int = 30, 
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        lr: float = 0.0005, 
        weight_decay: float = 1e-4,
        alpha: float = 50.0,
        penalty_factor: float = 1.5,
        lambda_gating: float = 0.5
    ) -> None:
        self.input_dim = input_dim
        self.lookback = lookback
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.penalty_factor = penalty_factor
        self.lambda_gating = lambda_gating
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = RegimeGatedMoEModel(
            input_dim=self.input_dim,
            lookback=self.lookback,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        ).to(self.device)
        
        self.criterion = AsymmetricMoELoss(
            alpha=self.alpha, 
            penalty_factor=self.penalty_factor, 
            lambda_gating=self.lambda_gating
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        logger.info(f"Modello PyTorch Regime-Gated MoE moe_v1 inizializzato. Device: {self.device}")

    def train(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame], 
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        tomorrow_returns_train: Optional[Union[np.ndarray, pd.Series]] = None,
        tomorrow_returns_val: Optional[Union[np.ndarray, pd.Series]] = None,
        crash_regimes_train: Optional[Union[np.ndarray, pd.Series]] = None,
        crash_regimes_val: Optional[Union[np.ndarray, pd.Series]] = None,
        regime_labels_train: Optional[Union[np.ndarray, pd.Series]] = None,
        regime_labels_val: Optional[Union[np.ndarray, pd.Series]] = None,
        epochs: int = 120,
        batch_size: int = 512,
        early_stopping_rounds: int = 15,
        verbose: bool = True
    ) -> Dict[str, Any]:
        
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_tr = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        
        ret_tr = tomorrow_returns_train.values if isinstance(tomorrow_returns_train, pd.Series) else np.array(tomorrow_returns_train) if tomorrow_returns_train is not None else np.zeros(len(y_tr))
        crash_tr = crash_regimes_train.values if isinstance(crash_regimes_train, pd.Series) else np.array(crash_regimes_train) if crash_regimes_train is not None else np.zeros(len(y_tr))
        reg_tr = regime_labels_train.values if isinstance(regime_labels_train, pd.Series) else np.array(regime_labels_train) if regime_labels_train is not None else np.zeros(len(y_tr))
        
        np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.clip(X_tr, -10.0, 10.0, out=X_tr)
        
        X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        ret_tr_tensor = torch.tensor(ret_tr, dtype=torch.float32).unsqueeze(1)
        crash_tr_tensor = torch.tensor(crash_tr, dtype=torch.float32).unsqueeze(1)
        reg_tr_tensor = torch.tensor(reg_tr, dtype=torch.float32).unsqueeze(1)
        
        train_dataset = torch.utils.data.TensorDataset(X_tr_tensor, y_tr_tensor, ret_tr_tensor, crash_tr_tensor, reg_tr_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = X_val.values if isinstance(X_val, pd.DataFrame) else np.array(X_val)
            y_v = y_val.values if isinstance(y_val, pd.Series) else np.array(y_val)
            
            ret_v = tomorrow_returns_val.values if isinstance(tomorrow_returns_val, pd.Series) else np.array(tomorrow_returns_val) if tomorrow_returns_val is not None else np.zeros(len(y_v))
            crash_v = crash_regimes_val.values if isinstance(crash_regimes_val, pd.Series) else np.array(crash_regimes_val) if crash_regimes_val is not None else np.zeros(len(y_v))
            reg_v = regime_labels_val.values if isinstance(regime_labels_val, pd.Series) else np.array(regime_labels_val) if regime_labels_val is not None else np.zeros(len(y_v))
            
            np.nan_to_num(X_v, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            np.clip(X_v, -10.0, 10.0, out=X_v)
            
            X_v_tensor = torch.tensor(X_v, dtype=torch.float32)
            y_v_tensor = torch.tensor(y_v, dtype=torch.float32).unsqueeze(1)
            ret_v_tensor = torch.tensor(ret_v, dtype=torch.float32).unsqueeze(1)
            crash_v_tensor = torch.tensor(crash_v, dtype=torch.float32).unsqueeze(1)
            reg_v_tensor = torch.tensor(reg_v, dtype=torch.float32).unsqueeze(1)
            
            val_dataset = torch.utils.data.TensorDataset(X_v_tensor, y_v_tensor, ret_v_tensor, crash_v_tensor, reg_v_tensor)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=False)
            
        best_val_loss = float('inf')
        best_epoch = -1
        epochs_no_improve = 0
        best_weights = None
        
        history = {"train_loss": [], "val_loss": [], "train_loss_pred": [], "train_loss_gating": []}
        
        for epoch in range(epochs):
            self.model.train()
            train_loss_accum = 0.0
            train_loss_pred_accum = 0.0
            train_loss_gating_accum = 0.0
            
            if HAS_TQDM and verbose:
                loader_iter = tqdm(
                    train_loader, 
                    desc=f"Epoca {epoch+1:03d}/{epochs:03d}", 
                    bar_format="{l_bar}{bar:20}{r_bar}",
                    leave=False
                )
            else:
                loader_iter = train_loader
                
            for X_batch, y_batch, ret_batch, crash_batch, reg_batch in loader_iter:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                ret_batch = ret_batch.to(self.device)
                crash_batch = crash_batch.to(self.device)
                reg_batch = reg_batch.to(self.device)
                
                self.optimizer.zero_grad()
                logits, gating_logits = self.model(X_batch)
                loss, loss_pred, loss_gating = self.criterion(
                    logits, gating_logits, y_batch, ret_batch, crash_batch, reg_batch
                )
                
                loss.backward()
                self.optimizer.step()
                
                train_loss_accum += loss.item() * X_batch.size(0)
                train_loss_pred_accum += loss_pred.item() * X_batch.size(0)
                train_loss_gating_accum += loss_gating.item() * X_batch.size(0)
                
                if HAS_TQDM and verbose:
                    loader_iter.set_postfix(Loss=f"{loss.item():.4f}", Pred=f"{loss_pred.item():.4f}", Gate=f"{loss_gating.item():.4f}")
                
            train_loss = train_loss_accum / len(X_tr_tensor)
            train_loss_pred = train_loss_pred_accum / len(X_tr_tensor)
            train_loss_gating = train_loss_gating_accum / len(X_tr_tensor)
            
            history["train_loss"].append(train_loss)
            history["train_loss_pred"].append(train_loss_pred)
            history["train_loss_gating"].append(train_loss_gating)
            
            val_loss = 0.0
            val_loss_pred = 0.0
            val_loss_gating = 0.0
            if has_val:
                self.model.eval()
                val_loss_accum = 0.0
                val_loss_pred_accum = 0.0
                val_loss_gating_accum = 0.0
                with torch.no_grad():
                    for X_v_batch, y_v_batch, ret_v_batch, crash_v_batch, reg_v_batch in val_loader:
                        X_v_batch = X_v_batch.to(self.device)
                        y_v_batch = y_v_batch.to(self.device)
                        ret_v_batch = ret_v_batch.to(self.device)
                        crash_v_batch = crash_v_batch.to(self.device)
                        reg_v_batch = reg_v_batch.to(self.device)
                        
                        logits_v, gating_logits_v = self.model(X_v_batch)
                        loss_v, loss_v_pred, loss_v_gating = self.criterion(
                            logits_v, gating_logits_v, y_v_batch, ret_v_batch, crash_v_batch, reg_v_batch
                        )
                        val_loss_accum += loss_v.item() * X_v_batch.size(0)
                        val_loss_pred_accum += loss_v_pred.item() * X_v_batch.size(0)
                        val_loss_gating_accum += loss_v_gating.item() * X_v_batch.size(0)
                        
                val_loss = val_loss_accum / len(X_v_tensor)
                val_loss_pred = val_loss_pred_accum / len(X_v_tensor)
                val_loss_gating = val_loss_gating_accum / len(X_v_tensor)
                history["val_loss"].append(val_loss)
                
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                self.scheduler.step(train_loss)
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
            if verbose:
                val_str = f" - Val Loss: {val_loss:.4f} (Pred: {val_loss_pred:.4f}, Gate: {val_loss_gating:.4f})" if has_val else ""
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoca {epoch+1:03d}/{epochs:03d} - Loss: {train_loss:.4f} (Pred: {train_loss_pred:.4f}, Gate: {train_loss_gating:.4f}){val_str} - LR: {current_lr:.6f}")
                
            if has_val and epochs_no_improve >= early_stopping_rounds:
                logger.info(f"Early stopping attivato all'epoca {epoch+1}. Miglior epoca: {best_epoch+1} con Val Loss: {best_val_loss:.4f}")
                break
                
        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
            
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
                logits, _ = self.model(batch_X)
                probs = torch.sigmoid(logits)
                probabilities.append(probs.cpu().numpy().flatten())
                
        return np.concatenate(probabilities)

    def save(self, filepath: str) -> None:
        """Salva i pesi del modello e le impostazioni in un file .pth"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'lookback': self.lookback,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'alpha': self.alpha,
            'penalty_factor': self.penalty_factor,
            'lambda_gating': self.lambda_gating
        }
        torch.save(save_dict, filepath)
        logger.info(f"Modello MoE salvato con successo in: {filepath}")

    def load(self, filepath: str) -> None:
        """Carica i pesi e l'architettura da file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.input_dim = checkpoint.get('input_dim', self.input_dim)
        self.lookback = checkpoint.get('lookback', self.lookback)
        self.d_model = checkpoint.get('d_model', self.d_model)
        self.nhead = checkpoint.get('nhead', self.nhead)
        self.num_layers = checkpoint.get('num_layers', self.num_layers)
        self.alpha = checkpoint.get('alpha', self.alpha)
        self.penalty_factor = checkpoint.get('penalty_factor', self.penalty_factor)
        self.lambda_gating = checkpoint.get('lambda_gating', self.lambda_gating)
        
        self.model = RegimeGatedMoEModel(
            input_dim=self.input_dim,
            lookback=self.lookback,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Modello MoE caricato con successo da: {filepath}")
