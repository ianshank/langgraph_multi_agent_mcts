"""
Policy-Value Network for state evaluation.

Primarily acts as a Value Network (Critic) to estimate the expected outcome (win probability)
of a given state (text description/code). Can be extended with a Policy head if needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

class PolicyValueNetwork(nn.Module):
    """
    Estimates the value V(s) of a state s.
    
    Args:
        input_dim: Dimension of the input embedding (from SystemEncoder).
        hidden_dim: Dimension of hidden layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Value Head: Input -> Hidden -> Scalar (tanh for -1 to 1, or sigmoid for 0 to 1)
        # Assuming outcome is success (0 or 1). Sigmoid is appropriate.
        # If outcome is -1/1, Tanh is appropriate.
        # ContinuousLearning.py records outcome 1.0 (success).
        
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() 
        )
        
        # Placeholder for Policy Head (if we were successfully distilling discrete actions)
        # self.policy_head = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict value from state embedding.
        
        Args:
            x: [Batch, Input_Dim] tensor.
            
        Returns:
            value: [Batch, 1] tensor (0.0 to 1.0)
        """
        return self.value_head(x)
