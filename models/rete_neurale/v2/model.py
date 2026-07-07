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
logger = logging.getLogger("NeuralNetworkV2")


class ResidualBlock(nn.Module):
    """
    Blocco residuo per feature tabulari con Batch Normalization, GELU, Dropout e skip connection.
    """
    def __init__(self, dim: int, dropout_rate: float = 0.25) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PyTorchMLPV2(nn.Module):
    """
    Architettura Neurale Residenziale (v2) in PyTorch.
    Ottimizzata con connessioni residue, Batch Normalization, Dropout e attivazioni GELU.
    """
    def __init__(self, input_dim: int, dropout_rate: float = 0.25) -> None:
        super(PyTorchMLPV2, self).__init__()
        
        # Strato iniziale di proiezione
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Blocchi residui e passaggi di transizione dimensionale
        self.res_block1 = ResidualBlock(128, dropout_rate)
        
        self.transition1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.res_block2 = ResidualBlock(64, dropout_rate)
        
        self.transition2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.res_block3 = ResidualBlock(32, dropout_rate)
        
        # Strato finale di classificazione (output = logit grezzo, senza Sigmoid)
        # La Sigmoid è gestita internamente da BCEWithLogitsLoss per stabilità numerica
        self.output_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.transition1(x)
        x = self.res_block2(x)
        x = self.transition2(x)
        x = self.res_block3(x)
        return self.output_layer(x)


class NeuralNetworkV2(BaseModel):
    """
    Modello wrapper NeuralNetworkV2 che implementa BaseModel.
    Gestisce l'addestramento residuo v2 su GPU VRAM, la predizione e il salvataggio dei pesi.
    """
    def __init__(self, input_dim: int, lr: float = 0.001, weight_decay: float = 1e-4) -> None:
        self.input_dim = input_dim
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Rilevamento automatico GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        
        self.model = PyTorchMLPV2(input_dim=self.input_dim).to(self.device)
        # BCEWithLogitsLoss combina Sigmoid + BCE in modo numericamente stabile,
        # essenziale con connessioni residue che possono generare valori fuori [0,1]
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        logger.info(f"Modello PyTorch Residual MLP v2 inizializzato. Device: {self.device}")

    def train(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame], 
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        epochs: int = 150,
        batch_size: int = 512,
        early_stopping_rounds: int = 15,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Addestra il modello PyTorch residuo v2 caricando l'intero dataset direttamente in GPU VRAM
        e usando shuffling a latenza zero.
        """
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_tr = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        
        # Sanitizzazione preventiva: sostituisce NaN e Inf con 0 e clamp valori estremi
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(X_tr, -10.0, 10.0, out=X_tr)
        
        # Spostamento in GPU VRAM per una velocità di addestramento ultra-rapida
        X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = X_val.values if isinstance(X_val, pd.DataFrame) else np.array(X_val)
            y_v = y_val.values if isinstance(y_val, pd.Series) else np.array(y_val)
            X_v = np.nan_to_num(X_v, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(X_v, -10.0, 10.0, out=X_v)
            X_v_tensor = torch.tensor(X_v, dtype=torch.float32).to(self.device)
            y_v_tensor = torch.tensor(y_v, dtype=torch.float32).unsqueeze(1).to(self.device)
            
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
            epoch_iterator = tqdm(epoch_iterator, desc="Progresso Totale v2", leave=True)
            
        num_samples = len(X_tr_tensor)
        
        for epoch in epoch_iterator:
            self.model.train()
            epoch_loss = 0.0
            
            # Shuffling su GPU
            indices = torch.randperm(num_samples, device=self.device)
            X_tr_shuffled = X_tr_tensor[indices]
            y_tr_shuffled = y_tr_tensor[indices]
            
            batch_indices = list(range(0, num_samples, batch_size))
            if HAS_TQDM and verbose:
                batch_iterator = tqdm(
                    batch_indices, 
                    desc=f"Epoca v2 {epoch:03d}/{epochs}", 
                    leave=True
                )
            else:
                batch_iterator = batch_indices
                
            for i in batch_iterator:
                batch_X = X_tr_shuffled[i:i+batch_size]
                batch_y = y_tr_shuffled[i:i+batch_size]
                
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
                    val_predictions = self.model(X_v_tensor)
                    val_loss_item = self.criterion(val_predictions, y_v_tensor)
                    val_loss = val_loss_item.item()
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
        Genera le probabilità stimate per l'input X.
        Applica Sigmoid ai logit grezzi del modello per ottenere probabilità in [0, 1].
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(X_arr, -10.0, 10.0, out=X_arr)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.sigmoid(logits)
            return probabilities.cpu().numpy().flatten()

    def save(self, filepath: str) -> None:
        """
        Salva i pesi e i parametri del modello su disco.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        state = {
            "input_dim": self.input_dim,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(state, filepath)
        logger.info(f"Modello v2 salvato all'indirizzo: {filepath}")

    def load(self, filepath: str) -> None:
        """
        Carica i pesi e la configurazione del modello da disco in modo robusto.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File dei pesi v2 non trovato all'indirizzo: {filepath}")
            
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Determina o ricava input_dim
        if "input_dim" in state:
            self.input_dim = state["input_dim"]
        else:
            s_dict = state.get("model_state_dict", state)
            if "input_layer.0.weight" in s_dict:
                self.input_dim = s_dict["input_layer.0.weight"].shape[1]
                logger.info(f"Dedicuta input_dim = {self.input_dim} dai pesi del primo layer v2.")
            else:
                logger.warning("Impossibile determinare input_dim dal file. Mantengo il valore corrente.")
        
        self.lr = state.get("lr", self.lr)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        
        # Ricostruisce il modello v2
        self.model = PyTorchMLPV2(input_dim=self.input_dim).to(self.device)
        
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
                logger.warning(f"Impossibile ripristinare lo stato dell'optimizer v2: {e}.")
        
        logger.info(f"Modello v2 caricato con successo da: {filepath}")
