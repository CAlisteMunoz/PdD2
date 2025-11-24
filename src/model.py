import torch
import torch.nn as nn

class ClimateEmulator(nn.Module):
    def __init__(self, input_size, hidden_layers=[512, 512], dropout_rate=0.2):
        """
        MLP para emulación climática.
        Args:
            input_size: Cantidad total de píxeles (lat * lon).
            hidden_layers: Lista con el tamaño de capas ocultas.
            dropout_rate: Para evitar overfitting.
        """
        super(ClimateEmulator, self).__init__()
        
        layers = []
        in_dim = input_size
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            #layers.append(nn.BatchNorm1d(h_dim)) # Normalización para estabilidad
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        # Capa de salida (reconstruye el mapa completo)
        layers.append(nn.Linear(in_dim, input_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
