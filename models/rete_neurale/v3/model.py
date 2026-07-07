import logging
import os
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NeuralNetworkV3")


class TemporalAttention(nn.Module):
    """
    Self-Attention temporale (simile al livello Attention di Keras).
    Pesatura automatica dell'importanza di ciascuno step temporale nella lookback window.
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, hidden_dim)
        q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        k = self.key(x)    # (batch_size, seq_len, hidden_dim)
        v = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Prodotto scalare dell'attenzione: (batch_size, seq_len, seq_len)
        scores = torch.bmm(q, k.transpose(1, 2)) / (x.shape[-1] ** 0.5)
        attn_weights = self.softmax(scores)
        
        # Output pesato: (batch_size, seq_len, hidden_dim)
        out = torch.bmm(attn_weights, v)
        return out


class PyTorchLSTMModel(nn.Module):
    """
    Architettura neurale ricorrente profonda con Self-Attention temporale per serie storiche (v3).
    Ispirata ai modelli v4.3 e v4.6 del codice legacy.
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64, dropout_rate: float = 0.3) -> None:
        super().__init__()
        # Strato 1: LSTM CuDNN-compatible (nessun recurrent_dropout, activation='tanh')
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Strato 2: Self-Attention
        self.attention = TemporalAttention(hidden_dim1)
        
        # Strato 3: LSTM per sintesi finale
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Strato 4: Decodifica decisionale e logit
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)  # Emette logit grezzo per stabilità numerica con BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm1_out, _ = self.lstm1(x)  # (batch_size, seq_len, hidden_dim1)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Attenzione
        attn_out = self.attention(lstm1_out)  # (batch_size, seq_len, hidden_dim1)
        
        # Sintesi temporale
        lstm2_out, _ = self.lstm2(attn_out)  # (batch_size, seq_len, hidden_dim2)
        # Prendiamo l'ultimo time step della sequenza sintetizzata
        last_step = lstm2_out[:, -1, :]  # (batch_size, hidden_dim2)
        last_step = self.dropout2(last_step)
        
        # Logits finali
        logits = self.fc(last_step)
        return logits


class NeuralNetworkV3(BaseModel):
    """
    Modello wrapper NeuralNetworkV3 che implementa BaseModel per un addestramento sequenziale.
    Gestisce l'addestramento sequenziale su GPU VRAM, predizioni e caricamento/salvataggio.
    """
    def __init__(self, input_dim: int, lookback: int = 30, lr: float = 0.0005, weight_decay: float = 1e-4) -> None:
        self.input_dim = input_dim
        self.lookback = lookback
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Rilevamento automatico GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        
        self.model = PyTorchLSTMModel(input_dim=self.input_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        logger.info(f"Modello PyTorch LSTM + Attention v3 inizializzato. Device: {self.device}")

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
        """
        Addestra il modello sequenziale tenendo i grandi dataset sequenziali in CPU RAM,
        spostando solo il singolo batch in GPU VRAM durante l'iterazione per evitare CUDA OOM.
        """
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_tr = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        
        # Sanitizzazione preventiva: sostituisce NaN e Inf con 0 e clamp valori estremi
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(X_tr, -10.0, 10.0, out=X_tr)
        
        # Caricamento in CPU RAM (invece di spostare subito giga di dati in GPU VRAM!)
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
        
        # tqdm progress bar
        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
            
        epoch_iterator = range(1, epochs + 1)
        if HAS_TQDM and verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Progresso Totale v3", leave=True)
            
        num_samples = len(X_tr_tensor)
        
        for epoch in epoch_iterator:
            self.model.train()
            epoch_loss = 0.0
            
            # Shuffling su CPU per non caricare la VRAM
            indices = torch.randperm(num_samples)
            X_tr_shuffled = X_tr_tensor[indices]
            y_tr_shuffled = y_tr_tensor[indices]
            
            batch_indices = list(range(0, num_samples, batch_size))
            if HAS_TQDM and verbose:
                batch_iterator = tqdm(
                    batch_indices, 
                    desc=f"Epoca v3 {epoch:03d}/{epochs}", 
                    leave=True
                )
            else:
                batch_iterator = batch_indices
                
            for i in batch_iterator:
                # Spostamento dinamico del singolo batch in GPU (latenza trascurabile, grande risparmio di VRAM!)
                batch_X = X_tr_shuffled[i:i+batch_size].to(self.device)
                batch_y = y_tr_shuffled[i:i+batch_size].to(self.device)
                
                # Se la dimensione del batch è 1, saltiamo per evitare problemi con BatchNorm
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
                    # Per evitare picchi di memoria con dataset di validazione enormi, 
                    # calcoliamo la val loss in batch successivi
                    val_loss_sum = 0.0
                    num_val_samples = len(X_v_tensor)
                    for j in range(0, num_val_samples, batch_size):
                        val_batch_X = X_v_tensor[j:j+batch_size].to(self.device)
                        val_batch_y = y_v_tensor[j:j+batch_size].to(self.device)
                        val_pred = self.model(val_batch_X)
                        val_loss_sum += self.criterion(val_pred, val_batch_y).item() * len(val_batch_X)
                    val_loss = val_loss_sum / num_val_samples
                    history["val_loss"].append(val_loss)
            
            # Logging e Early Stopping
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
        """
        Genera le probabilità stimate per l'input 3D X in lotti per risparmiare memoria.
        Applica la Sigmoid ai logit grezzi del modello per restituire probabilità in [0, 1].
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(X_arr, -10.0, 10.0, out=X_arr)
        
        self.model.eval()
        probabilities = []
        batch_size = 1024  # Batch size di sicurezza per prevenire OOM su dataset giganti
        
        with torch.no_grad():
            for i in range(0, len(X_arr), batch_size):
                batch_X = torch.tensor(X_arr[i:i+batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(batch_X)
                probs = torch.sigmoid(logits)
                probabilities.append(probs.cpu().numpy().flatten())
                
        return np.concatenate(probabilities)

    def save(self, filepath: str) -> None:
        """
        Salva i pesi e i parametri del modello su disco.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        state = {
            "input_dim": self.input_dim,
            "lookback": self.lookback,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(state, filepath)
        logger.info(f"Modello v3 salvato all'indirizzo: {filepath}")

    def load(self, filepath: str) -> None:
        """
        Carica i pesi e la configurazione del modello da disco in modo robusto.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File dei pesi v3 non trovato all'indirizzo: {filepath}")
            
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Ricava parametri configurati
        self.input_dim = state.get("input_dim", self.input_dim)
        self.lookback = state.get("lookback", self.lookback)
        self.lr = state.get("lr", self.lr)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        
        # Ricostruisce il modello v3
        self.model = PyTorchLSTMModel(input_dim=self.input_dim).to(self.device)
        
        # Carica lo stato dei pesi
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
            
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if "optimizer_state_dict" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Impossibile ripristinare lo stato dell'optimizer v3: {e}.")
        
        logger.info(f"Modello v3 caricato con successo da: {filepath}")
