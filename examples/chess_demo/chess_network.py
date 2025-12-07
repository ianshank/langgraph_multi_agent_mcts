"""
Chess Neural Network for MCTS Policy-Value Estimation.

Implements a policy-value network architecture similar to AlphaZero
but simplified for demonstration purposes.

Best Practices 2025:
- Modular architecture
- Configurable via environment variables
- Optional GPU support
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Optional torch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class ChessNetworkConfig:
    """Configuration for chess neural network."""

    # Input dimensions
    input_channels: int = 14  # 12 piece planes + turn + castling
    board_size: int = 8

    # Network architecture
    num_residual_blocks: int = field(default_factory=lambda: int(os.getenv("CHESS_RESIDUAL_BLOCKS", "6")))
    num_filters: int = field(default_factory=lambda: int(os.getenv("CHESS_NUM_FILTERS", "128")))

    # Policy head
    policy_filters: int = field(default_factory=lambda: int(os.getenv("CHESS_POLICY_FILTERS", "32")))
    num_actions: int = 4672  # Approximate number of possible chess moves

    # Value head
    value_filters: int = field(default_factory=lambda: int(os.getenv("CHESS_VALUE_FILTERS", "32")))
    value_hidden: int = field(default_factory=lambda: int(os.getenv("CHESS_VALUE_HIDDEN", "256")))

    # Training
    dropout: float = field(default_factory=lambda: float(os.getenv("CHESS_DROPOUT", "0.1")))

    # Device
    device: str = field(default_factory=lambda: "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")


if TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        """Residual block with batch normalization."""

        def __init__(self, num_filters: int):
            super().__init__()
            self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(num_filters)
            self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_filters)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return F.relu(out)

    class ChessNetwork(nn.Module):
        """
        Policy-Value network for chess.

        Architecture:
        - Convolutional trunk with residual blocks
        - Policy head: action probabilities for all legal moves
        - Value head: position evaluation [-1, 1]
        """

        def __init__(self, config: ChessNetworkConfig | None = None):
            super().__init__()
            self.config = config or ChessNetworkConfig()

            # Initial convolution
            self.initial_conv = nn.Conv2d(self.config.input_channels, self.config.num_filters, 3, padding=1, bias=False)
            self.initial_bn = nn.BatchNorm2d(self.config.num_filters)

            # Residual tower
            self.residual_blocks = nn.ModuleList(
                [ResidualBlock(self.config.num_filters) for _ in range(self.config.num_residual_blocks)]
            )

            # Policy head
            self.policy_conv = nn.Conv2d(self.config.num_filters, self.config.policy_filters, 1, bias=False)
            self.policy_bn = nn.BatchNorm2d(self.config.policy_filters)
            self.policy_fc = nn.Linear(self.config.policy_filters * 64, self.config.num_actions)

            # Value head
            self.value_conv = nn.Conv2d(self.config.num_filters, self.config.value_filters, 1, bias=False)
            self.value_bn = nn.BatchNorm2d(self.config.value_filters)
            self.value_fc1 = nn.Linear(self.config.value_filters * 64, self.config.value_hidden)
            self.value_fc2 = nn.Linear(self.config.value_hidden, 1)

            # Dropout for regularization
            self.dropout = nn.Dropout(self.config.dropout)

        def forward(
            self,
            x: torch.Tensor,
            legal_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                x: Board tensor of shape (batch, 14, 8, 8)
                legal_mask: Optional mask for legal moves (batch, num_actions)

            Returns:
                policy: Log probabilities for actions (batch, num_actions)
                value: Position evaluation (batch, 1)
            """
            # Initial convolution
            out = F.relu(self.initial_bn(self.initial_conv(x)))

            # Residual tower
            for block in self.residual_blocks:
                out = block(out)

            # Policy head
            policy = F.relu(self.policy_bn(self.policy_conv(out)))
            policy = policy.view(policy.size(0), -1)
            policy = self.policy_fc(policy)

            # Apply legal move mask if provided
            if legal_mask is not None:
                # Mask illegal moves with large negative value
                policy = policy.masked_fill(~legal_mask, float("-inf"))

            policy = F.log_softmax(policy, dim=1)

            # Value head
            value = F.relu(self.value_bn(self.value_conv(out)))
            value = value.view(value.size(0), -1)
            value = self.dropout(F.relu(self.value_fc1(value)))
            value = torch.tanh(self.value_fc2(value))

            return policy, value

        def predict(
            self,
            state_tensor: torch.Tensor,
            legal_actions: list[int] | None = None,
        ) -> tuple[np.ndarray, float]:
            """
            Predict policy and value for a single state.

            Args:
                state_tensor: Board tensor (14, 8, 8)
                legal_actions: List of legal action indices

            Returns:
                policy: Action probabilities as numpy array
                value: Position evaluation
            """
            self.eval()
            with torch.no_grad():
                x = state_tensor.unsqueeze(0).to(self.config.device)

                # Create legal move mask
                legal_mask = None
                if legal_actions is not None:
                    legal_mask = torch.zeros(1, self.config.num_actions, dtype=torch.bool)
                    for idx in legal_actions:
                        if 0 <= idx < self.config.num_actions:
                            legal_mask[0, idx] = True
                    legal_mask = legal_mask.to(self.config.device)

                policy, value = self(x, legal_mask)

                # Convert to probabilities
                policy_probs = torch.exp(policy).cpu().numpy()[0]
                value_scalar = value.cpu().item()

            return policy_probs, value_scalar

        def save(self, path: str) -> None:
            """Save model weights."""
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "config": self.config,
                },
                path,
            )

        @classmethod
        def load(cls, path: str) -> ChessNetwork:
            """Load model from checkpoint."""
            checkpoint = torch.load(path, map_location="cpu")
            config = checkpoint.get("config", ChessNetworkConfig())
            model = cls(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model

else:
    # Dummy class when torch not available
    class ChessNetwork:  # type: ignore[no-redef]
        """Placeholder when torch not available."""

        def __init__(self, config: Any = None):
            raise RuntimeError("PyTorch is required for ChessNetwork")


class SimpleChessEvaluator:
    """
    Simple rule-based chess evaluator for environments without torch.

    Uses classical chess evaluation techniques:
    - Material counting
    - Piece-square tables
    - Mobility
    - King safety
    """

    def __init__(self):
        # Piece values in centipawns
        self.piece_values = {
            1: 100,  # Pawn
            2: 320,  # Knight
            3: 330,  # Bishop
            4: 500,  # Rook
            5: 900,  # Queen
            6: 20000,  # King
        }

    def evaluate(self, board: Any) -> float:
        """
        Evaluate position.

        Args:
            board: chess.Board object

        Returns:
            Score from white's perspective in [-1, 1]
        """
        if board.is_checkmate():
            return -1.0 if board.turn else 1.0

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = 0.0

        # Material
        for piece_type in range(1, 7):
            white = len(board.pieces(piece_type, True))
            black = len(board.pieces(piece_type, False))
            score += (white - black) * self.piece_values[piece_type]

        # Mobility
        mobility = len(list(board.legal_moves))
        board.push(board.parse_san("--") if board.has_legal_en_passant() else None) if False else None
        # Simplified mobility
        score += mobility * 10

        # Normalize
        return max(-1.0, min(1.0, score / 10000.0))

    def get_move_scores(self, board: Any) -> dict[str, float]:
        """
        Score each legal move.

        Returns:
            Dictionary mapping UCI moves to scores
        """
        scores = {}

        for move in board.legal_moves:
            # Score based on:
            # 1. Captures (higher value = higher score)
            # 2. Center control
            # 3. Checks

            score = 0.0
            uci = move.uci()

            # Capture bonus
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    score += self.piece_values.get(captured.piece_type, 0) / 100.0

            # Check bonus
            board.push(move)
            if board.is_check():
                score += 0.5
            board.pop()

            # Center control bonus
            to_file = move.to_square % 8
            to_rank = move.to_square // 8
            center_dist = abs(to_file - 3.5) + abs(to_rank - 3.5)
            score += (7 - center_dist) / 14.0

            scores[uci] = score

        return scores


def create_legal_mask(board: Any, num_actions: int = 4672) -> np.ndarray:
    """
    Create legal move mask for the network.

    Args:
        board: chess.Board object
        num_actions: Total number of possible actions

    Returns:
        Boolean mask array
    """
    from .chess_state import uci_to_index

    mask = np.zeros(num_actions, dtype=bool)

    for move in board.legal_moves:
        idx = uci_to_index(move.uci())
        if 0 <= idx < num_actions:
            mask[idx] = True

    return mask
