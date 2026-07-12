import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Optional, Union, List

from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SpatioTemporalGNNV1")


class GraphAttentionLayer(nn.Module):
    """
    Custom Graph Attention (GAT) Layer implemented in native PyTorch.
    Computes spatial attention weights across tickers dynamically.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Linear projection weight matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameter vector
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node feature matrix of shape (B, N, F_in) or (N, F_in)
            adj: Adjacency matrix of shape (N, N)
        Returns:
            Projected node features with graph attention aggregation of shape (B, N, F_out) or (N, F_out)
        """
        if h.dim() == 3:
            B, N, _ = h.shape
            # Step 1: Linear projection -> (B, N, F_out)
            Wh = torch.matmul(h, self.W)
            
            # Step 2: Attention Coefficients Computation
            # a: (2 * F_out, 1) -> split into a_1: (F_out, 1), a_2: (F_out, 1)
            a_1 = self.a[:self.out_features, :]
            a_2 = self.a[self.out_features:, :]
            
            # f_1: (B, N, 1), f_2: (B, N, 1)
            f_1 = torch.matmul(Wh, a_1)
            f_2 = torch.matmul(Wh, a_2)
            
            # e: (B, N, N) where e_ij = LeakyReLU(f_1_i + f_2_j)
            e = self.leakyrelu(f_1 + f_2.transpose(1, 2))
            
            # Mask out non-neighbors: if adj_ij = 0, set to -9e15
            zero_vec = -9e15 * torch.ones_like(e)
            # Expand adj to (1, N, N) to broadcast across batch dimension
            attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)
            
            # Softmax normalization over neighbors
            attention = torch.softmax(attention, dim=2)
            attention = self.dropout_layer(attention)
            
            # Step 3: Aggregation -> (B, N, F_out)
            h_prime = torch.matmul(attention, Wh)
            return h_prime
        else:
            # 2D Input: (N, F_in)
            # Step 1: Linear projection
            Wh = torch.mm(h, self.W)
            num_nodes = Wh.size()[0]
            
            # Step 2: Attention Coefficients Computation
            a_1 = self.a[:self.out_features, :]
            a_2 = self.a[self.out_features:, :]
            
            f_1 = torch.matmul(Wh, a_1)
            f_2 = torch.matmul(Wh, a_2)
            
            e = self.leakyrelu(f_1 + f_2.T)
            
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            
            attention = torch.softmax(attention, dim=1)
            attention = self.dropout_layer(attention)
            
            # Step 3: Aggregation
            h_prime = torch.matmul(attention, Wh)
            return h_prime


class SpatioTemporalGNNModel(nn.Module):
    """
    Spatio-Temporal GNN Network combining Graph Attention (spatial) and a temporal GRU layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout_rate: float = 0.2):
        super(SpatioTemporalGNNModel, self).__init__()
        
        # Spatial Graph Attention
        self.gat1 = GraphAttentionLayer(in_features=input_dim, out_features=hidden_dim, dropout=dropout_rate)
        self.gat_norm = nn.LayerNorm(hidden_dim)
        
        # Temporal Component (per node)
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1) # Raw logit output
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features over time. Shape: (batch_size, num_nodes, seq_len, input_dim)
               We assume each batch contains a snapshot of all nodes.
            adj: Adjacency matrix of shape (num_nodes, num_nodes)
        """
        batch_size, num_nodes, seq_len, input_dim = x.size()
        
        # Reshape x from (B, N, L, F_in) to (B * L, N, F_in) to apply GAT in a vectorized manner
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_nodes, input_dim)
        
        # Process all batch elements and time steps at once
        spatial_features_flat = self.gat1(x_reshaped, adj) # (B * L, N, hidden_dim)
        
        # Reshape back to (B, L, N, hidden_dim)
        spatial_features = spatial_features_flat.view(batch_size, seq_len, num_nodes, -1)
        
        # Permute to (B, N, L, hidden_dim) to keep consistency with the original LayerNorm step
        spatial_features = spatial_features.permute(0, 2, 1, 3)
        spatial_features = self.gat_norm(spatial_features)
        
        # Step 2: Temporal Layer (GRU)
        # Reshape to run GRU per node across all batch items:
        # (batch_size * num_nodes, seq_len, hidden_dim)
        spatial_features_flat = spatial_features.view(batch_size * num_nodes, seq_len, -1)
        
        gru_out, _ = self.temporal_gru(spatial_features_flat) # (batch_size * num_nodes, seq_len, hidden_dim)
        
        # Use final time step for classification
        last_temp_features = gru_out[:, -1, :] # (batch_size * num_nodes, hidden_dim)
        
        # Step 3: Decision Head
        logits = self.fc(last_temp_features) # (batch_size * num_nodes, 1)
        
        # Reshape back to (batch_size, num_nodes, 1)
        logits = logits.view(batch_size, num_nodes, 1)
        return torch.sigmoid(logits)


class SpatioTemporalGNNV1(BaseModel):
    """
    SpatioTemporalGNNV1 wrapper implementing BaseModel.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, lr: float = 0.001, weight_decay: float = 1e-4) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpatioTemporalGNNModel(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        logger.info(f"SpatioTemporalGNNV1 initialized on {self.device}")

    def train(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame, torch.Tensor], 
        y_train: Union[np.ndarray, pd.Series, torch.Tensor],
        X_val: Optional[Union[np.ndarray, pd.DataFrame, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None,
        adj: Optional[torch.Tensor] = None,
        epochs: int = 40,
        batch_size: int = 32,
        early_stopping_rounds: int = 10,
        verbose: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Train SpatioTemporalGNNV1 model using aligned spatiotemporal sequence tensors.
        """
        # Ensure Tensors
        X_tr = torch.as_tensor(X_train, dtype=torch.float32)
        y_tr = torch.as_tensor(y_train, dtype=torch.float32)
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = torch.as_tensor(X_val, dtype=torch.float32)
            y_v = torch.as_tensor(y_val, dtype=torch.float32)

        if adj is None:
            # Default identity matrix if no adjacency matrix is provided
            num_nodes = X_tr.shape[1]
            adj = torch.eye(num_nodes, device=self.device)
        else:
            adj = torch.as_tensor(adj, dtype=torch.float32).to(self.device)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_model_state = None
        no_improvement_epochs = 0

        # Try to import tqdm
        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False

        epoch_iterator = range(1, epochs + 1)
        if HAS_TQDM and verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="GNN Training Progress", leave=True)

        for epoch in epoch_iterator:
            self.model.train()
            epoch_loss = 0.0
            
            # Shuffle indices
            indices = torch.randperm(len(X_tr))
            X_tr_shuffled = X_tr[indices]
            y_tr_shuffled = y_tr[indices]
            
            num_batches = int(np.ceil(len(X_tr) / batch_size))
            
            # Inner progress bar per epoch
            batch_iterator = range(num_batches)
            if HAS_TQDM and verbose:
                batch_iterator = tqdm(
                    batch_iterator,
                    desc=f"Epoch {epoch:03d}/{epochs:03d} Batches",
                    leave=False
                )
                
            for b in batch_iterator:
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, len(X_tr))
                
                batch_X = X_tr_shuffled[start_idx:end_idx].to(self.device)
                batch_y = y_tr_shuffled[start_idx:end_idx].to(self.device)
                
                self.optimizer.zero_grad()
                preds = self.model(batch_X, adj)
                loss = self.criterion(preds.squeeze(2), batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * (end_idx - start_idx)
                if HAS_TQDM and verbose:
                    batch_iterator.set_postfix(Loss=f"{loss.item():.4f}")
                
            train_loss = epoch_loss / len(X_tr)
            history["train_loss"].append(train_loss)
            
            val_loss = 0.0
            if has_val:
                self.model.eval()
                val_loss_accum = 0.0
                with torch.no_grad():
                    num_val_batches = int(np.ceil(len(X_v) / batch_size))
                    for b in range(num_val_batches):
                        start_idx = b * batch_size
                        end_idx = min(start_idx + batch_size, len(X_v))
                        
                        batch_X = X_v[start_idx:end_idx].to(self.device)
                        batch_y = y_v[start_idx:end_idx].to(self.device)
                        
                        preds = self.model(batch_X, adj)
                        loss_v = self.criterion(preds.squeeze(2), batch_y)
                        val_loss_accum += loss_v.item() * (end_idx - start_idx)
                        
                val_loss = val_loss_accum / len(X_v)
                history["val_loss"].append(val_loss)
                
                if HAS_TQDM and verbose:
                    epoch_iterator.set_postfix_str(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
                elif verbose:
                    logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f}")
                
                # Check early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                    
                if no_improvement_epochs >= early_stopping_rounds:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best validation weights.")
                    self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
                    break
            else:
                if HAS_TQDM and verbose:
                    epoch_iterator.set_postfix_str(f"Train Loss: {train_loss:.5f}")
                elif verbose:
                    logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.5f}")

        return history

    def predict(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor], adj: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Predict directional probabilities using GNN model.
        """
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        if adj is None:
            num_nodes = X_tensor.shape[1]
            adj = torch.eye(num_nodes, device=self.device)
        else:
            adj = torch.as_tensor(adj, dtype=torch.float32).to(self.device)

        self.model.eval()
        probabilities = []
        batch_size = 16
        
        with torch.no_grad():
            num_batches = int(np.ceil(len(X_tensor) / batch_size))
            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, len(X_tensor))
                
                batch_X = X_tensor[start_idx:end_idx].to(self.device)
                preds = self.model(batch_X, adj)
                probabilities.append(preds.cpu().numpy())
                
        # Concatenate batches: (num_samples, num_nodes, 1) -> (num_samples, num_nodes)
        return np.concatenate(probabilities, axis=0).squeeze(2)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "model_state_dict": self.model.state_dict()
        }
        torch.save(state, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        state = torch.load(filepath, map_location=self.device)
        self.input_dim = state.get("input_dim", self.input_dim)
        self.hidden_dim = state.get("hidden_dim", self.hidden_dim)
        self.model = SpatioTemporalGNNModel(input_dim=self.input_dim, hidden_dim=self.hidden_dim).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info(f"Model loaded from {filepath}")
