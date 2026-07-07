from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Dict, Union, Optional

class BaseModel(ABC):
    """
    Classe base astratta (blueprint) per tutti i modelli predittivi del sistema.
    Qualsiasi modello futuro (reti neurali, alberi di decisione, modelli classici)
    dovrà ereditare da questa classe e implementare i metodi descritti.
    """

    @abstractmethod
    def train(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame], 
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Addestra il modello sulle feature X_train e i target y_train.
        Fornisce opzionalmente dati di validazione per early stopping/tuning.
        
        Ritorna:
            Dict[str, Any]: Statistiche del training (es. loss history, accuracy, ecc.).
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Genera le predizioni per il dataset di input.
        
        Ritorna:
            np.ndarray: Vettore di predizioni (probabilità in caso di classificazione, valori reali per regressione).
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Salva lo stato del modello (pesi e configurazione) su disco.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Carica lo stato del modello (pesi e configurazione) da disco.
        """
        pass
