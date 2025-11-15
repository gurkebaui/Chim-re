import torch
import torch.nn as nn
from mamba_ssm import Mamba

class Brain(nn.Module):
    def __init__(self, perception_vector_size: int, action_vector_size: int, d_model: int = 256, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.action_vector_size = action_vector_size

        self.input_expansion = nn.Linear(perception_vector_size, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)

        # --- START DER ÄNDERUNG ---
        # Statt einem Output-Kopf haben wir jetzt zwei:
        # 1. mu_head: Sagt den Mittelwert (die wahrscheinlichste Aktion) voraus.
        self.mu_head = nn.Linear(d_model, action_vector_size)
        
        # 2. log_std_head: Sagt den Logarithmus der Standardabweichung voraus.
        #    Dies sorgt für eine positive Standardabweichung und stabilisiert das Training.
        self.log_std_head = nn.Linear(d_model, action_vector_size)
        # --- ENDE DER ÄNDERUNG ---

        self._initialize_weights()

    def _initialize_weights(self):
        # ... (unverändert)
        nn.init.xavier_uniform_(self.input_expansion.weight)
        nn.init.zeros_(self.input_expansion.bias)
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)


    def forward(self, perception_tensor: torch.Tensor):
        x = perception_tensor.unsqueeze(1)
        x = self.input_expansion(x)
        x = self.mamba(x)
        x = x.squeeze(1) # Form anpassen für die Köpfe

        # --- START DER ÄNDERUNG ---
        # Berechne die Parameter der Verteilung
        mu = torch.tanh(self.mu_head(x)) # Mittelwert auf [-1, 1] begrenzen
        log_std = self.log_std_head(x)
        
        # Begrenze log_std, um extrem große/kleine Standardabweichungen zu vermeiden
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        
        return mu, log_std