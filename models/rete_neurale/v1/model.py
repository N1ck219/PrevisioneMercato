import logging
import os
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NeuralNetworkV1")


class PyTorchMLP(nn.Module):
    """
    Architettura di rete neurale Multi-Layer Perceptron (MLP) in PyTorch.
    Ottimizzata con Batch Normalization, Dropout e attivazioni GELU per serie storiche tabulari.
    """
    def __init__(self, input_dim: int, dropout_rate: float = 0.2) -> None:
        super(PyTorchMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Strato 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # Strato 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # Strato 3
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            
            # Output (Classificazione Binaria -> Sigmoid)
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class NeuralNetworkV1(BaseModel):
    """
    Modello wrapper NeuralNetworkV1 che implementa BaseModel.
    Gestisce l'addestramento, la predizione, il salvataggio su disco e l'early stopping.
    """
    def __init__(self, input_dim: int, lr: float = 0.001, weight_decay: float = 1e-4) -> None:
        self.input_dim = input_dim
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Rilevamento automatico della GPU se disponibile (CUDA o MPS per Mac)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        
        self.model = PyTorchMLP(input_dim=self.input_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        logger.info(f"Modello PyTorch MLP inizializzato con successo. Device: {self.device}")

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
        Addestra il modello PyTorch MLP con gestione del batch e Early Stopping.
        """
        # Conversione dei dati in numpy array se Pandas
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_tr = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        
        # Carichiamo l'intero dataset direttamente sulla GPU (CUDA) all'inizio per velocità estrema (solo ~66MB!)
        X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = X_val.values if isinstance(X_val, pd.DataFrame) else np.array(X_val)
            y_v = y_val.values if isinstance(y_val, pd.Series) else np.array(y_val)
            X_v_tensor = torch.tensor(X_v, dtype=torch.float32).to(self.device)
            y_v_tensor = torch.tensor(y_v, dtype=torch.float32).unsqueeze(1).to(self.device)
            
        history = {"train_loss": [], "val_loss": []}
        
        best_val_loss = float("inf")
        best_model_state = None
        no_improvement_epochs = 0
        
        # Import tqdm se disponibile per mostrare l'avanzamento epoch-by-epoch
        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
            
        epoch_iterator = range(1, epochs + 1)
        if HAS_TQDM and verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Progresso Totale", leave=True)
            
        num_samples = len(X_tr_tensor)
        
        for epoch in epoch_iterator:
            self.model.train()
            epoch_loss = 0.0
            
            # Generiamo indici casuali sulla GPU per uno shuffling velocissimo a latenza zero
            indices = torch.randperm(num_samples, device=self.device)
            X_tr_shuffled = X_tr_tensor[indices]
            y_tr_shuffled = y_tr_tensor[indices]
            
            # Creiamo l'iteratore per i batch
            batch_indices = list(range(0, num_samples, batch_size))
            if HAS_TQDM and verbose:
                # Mostra la barra singola specifica per questa epoca
                batch_iterator = tqdm(
                    batch_indices, 
                    desc=f"Epoca {epoch:03d}/{epochs}", 
                    leave=True
                )
            else:
                batch_iterator = batch_indices
                
            # Iterazione a batch diretta interamente in VRAM
            for i in batch_iterator:
                batch_X = X_tr_shuffled[i:i+batch_size]
                batch_y = y_tr_shuffled[i:i+batch_size]
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_X)
                
                # Aggiorna la perdita in tempo reale sulla barra dei batch
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
            
            # Log avanzamento
            if verbose:
                if HAS_TQDM:
                    val_str = f" - Val Loss: {val_loss:.6f}" if has_val else ""
                    # Aggiorniamo la barra dei batch con le metriche finali dell'epoca prima che si chiuda
                    batch_iterator.set_postfix_str(f"Train Loss: {train_loss:.6f}{val_str}")
                elif epoch % 10 == 0 or epoch == 1 or epoch == epochs or no_improvement_epochs == early_stopping_rounds - 1:
                    val_str = f" - Val Loss: {val_loss:.6f}" if has_val else ""
                    logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}{val_str}")
                
            # Early Stopping check
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
        Effettua il forward pass sul modello e restituisce le probabilità stimate.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            probabilities = self.model(X_tensor)
            return probabilities.cpu().numpy().flatten()

    def save(self, filepath: str) -> None:
        """
        Salva i pesi del modello, configurazione e optimizer su disco.
        """
        # Assicuriamoci che la directory esista
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        state = {
            "input_dim": self.input_dim,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(state, filepath)
        logger.info(f"Modello salvato con successo all'indirizzo: {filepath}")

    def load(self, filepath: str) -> None:
        """
        Carica i pesi e la configurazione del modello da disco in modo robusto.
        Previene KeyError per file di pesi con diversi formati di salvataggio.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File dei pesi non trovato all'indirizzo: {filepath}")
            
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Estraiamo o deduciamo l'input dimension
        if "input_dim" in state:
            self.input_dim = state["input_dim"]
        else:
            # Proviamo ad inferire l'input dimension dal tensore dei pesi del primo layer lineare
            s_dict = state.get("model_state_dict", state)
            if "network.0.weight" in s_dict:
                self.input_dim = s_dict["network.0.weight"].shape[1]
                logger.info(f"Dedicuta input_dim = {self.input_dim} dai pesi del primo layer.")
            else:
                logger.warning("Impossibile determinare input_dim dal file. Mantengo il valore corrente.")
        
        self.lr = state.get("lr", self.lr)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        
        # Ricostruiamo il modello
        self.model = PyTorchMLP(input_dim=self.input_dim).to(self.device)
        
        # Carichiamo lo stato del modello
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
            
        # Carichiamo o inizializziamo l'ottimizzatore
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if "optimizer_state_dict" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Impossibile ripristinare lo stato completo dell'optimizer: {e}. Creato un nuovo optimizer AdamW.")
        
        logger.info(f"Modello caricato con successo da: {filepath}")
