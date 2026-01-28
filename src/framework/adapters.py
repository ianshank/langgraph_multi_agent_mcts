"""
Neuro-Symbolic Adapter Module.

Bridges the gap between Symbolic (Text) and Neural (Tensor) reasoning.
Translates natural language queries into tensor representations for the HRM,
and translates HRM latent plans back into structured text strategies.

Includes:
- NeuralPlanner: Trainable transformer-based planner for strategy generation
- NeuroSymbolicAdapter: Bridge between text and neural reasoning
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.agents.hrm_agent import HRMAgent, SubProblem
except ImportError:
    # Type stub for development if src.agents is not in path
    class HRMAgent:
        pass

    class SubProblem:
        pass


class DecompositionStrategy(Enum):
    """Available decomposition strategies for neural planning."""

    DIRECT_ANSWER = 0  # Simple, direct response
    DEEP_RESEARCH = 1  # Requires extensive research/RAG
    STEP_BY_STEP = 2  # Multi-step reasoning
    TOOL_USE = 3  # Requires external tools
    DELEGATION = 4  # Delegate to specialized agent

    @classmethod
    def from_index(cls, idx: int) -> DecompositionStrategy:
        """Get strategy from index."""
        for member in cls:
            if member.value == idx:
                return member
        return cls.DIRECT_ANSWER

    @classmethod
    def num_strategies(cls) -> int:
        """Get number of available strategies."""
        return len(cls)


@dataclass
class NeuralPlanStep:
    """Structured step derived from neural decomposition."""

    level: int
    description: str
    strategy: str  # e.g., "Deep Research", "Direct Answer"
    confidence: float
    strategy_logits: list[float] = field(default_factory=list)  # Raw logits for training


@dataclass
class NeuralPlannerOutput:
    """Output from the Neural Planner."""

    steps: list[NeuralPlanStep]  # Sequence of planning steps
    strategy_probs: torch.Tensor  # [batch, max_steps, num_strategies]
    confidence_scores: torch.Tensor  # [batch, max_steps]
    attention_weights: torch.Tensor | None  # For interpretability
    hidden_states: torch.Tensor  # For downstream use


class NeuralPlanner(nn.Module):
    """
    Trainable Neural Planner for generating decomposition strategies.

    Architecture:
    - Transformer encoder for query understanding
    - Strategy classification head for each step
    - Confidence prediction head
    - Autoregressive step generation

    The planner learns to:
    1. Understand the query complexity
    2. Generate a sequence of decomposition steps
    3. Assign appropriate strategies to each step
    4. Predict confidence for each step
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_strategies: int = 5,
        max_steps: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pretrained_encoder: bool = False,
    ):
        """
        Initialize Neural Planner.

        Args:
            hidden_dim: Hidden dimension for transformer
            num_strategies: Number of decomposition strategies
            max_steps: Maximum planning steps
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_pretrained_encoder: Whether to use pretrained encoder
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        self.max_steps = max_steps

        # Query encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Step position embeddings
        self.step_embedding = nn.Embedding(max_steps, hidden_dim)

        # Query pooling (CLS-like token)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Strategy classification head
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_strategies),
        )

        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Step continuation head (should we generate more steps?)
        self.continuation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Description generator (optional, for generating step descriptions)
        self.description_projection = nn.Linear(hidden_dim, hidden_dim)

        # Learnable step difficulty estimator
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        num_steps: int | None = None,
        return_attention: bool = False,
    ) -> NeuralPlannerOutput:
        """
        Generate a planning sequence from query embedding.

        Args:
            query_embedding: [batch, seq_len, hidden_dim] query representation
            num_steps: Number of steps to generate (None = adaptive)
            return_attention: Whether to return attention weights

        Returns:
            NeuralPlannerOutput with planning sequence
        """
        batch_size = query_embedding.shape[0]
        device = query_embedding.device

        # Add query token
        query_token = self.query_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([query_token, query_embedding], dim=1)

        # Encode query
        encoded = self.encoder(encoder_input)
        query_repr = encoded[:, 0, :]  # [batch, hidden_dim]

        # Generate steps autoregressively
        steps = []
        strategy_probs_list = []
        confidence_list = []
        hidden_states_list = []

        max_gen_steps = num_steps or self.max_steps

        for step_idx in range(max_gen_steps):
            # Get step embedding
            step_emb = self.step_embedding(torch.tensor([step_idx], device=device)).expand(batch_size, -1)

            # Combine query representation with step embedding
            step_hidden = query_repr + step_emb
            hidden_states_list.append(step_hidden)

            # Predict strategy
            strategy_logits = self.strategy_head(step_hidden)  # [batch, num_strategies]
            strategy_probs = F.softmax(strategy_logits, dim=-1)
            strategy_probs_list.append(strategy_probs)

            # Predict confidence
            confidence = self.confidence_head(step_hidden).squeeze(-1)  # [batch]
            confidence_list.append(confidence)

            # Check if we should continue (for adaptive generation)
            if num_steps is None:
                continue_prob = self.continuation_head(step_hidden).squeeze(-1)
                if continue_prob.mean() < 0.5 and step_idx > 0:
                    break

            # Get best strategy
            best_strategy_idx = strategy_probs.argmax(dim=-1)  # [batch]
            strategy = DecompositionStrategy.from_index(best_strategy_idx[0].item())

            # Create step
            step = NeuralPlanStep(
                level=step_idx,
                description=f"Step {step_idx + 1}: {strategy.name.replace('_', ' ').title()}",
                strategy=strategy.name.replace("_", " ").title(),
                confidence=confidence[0].item(),
                strategy_logits=strategy_logits[0].tolist(),
            )
            steps.append(step)

        # Stack outputs
        strategy_probs = torch.stack(strategy_probs_list, dim=1)  # [batch, steps, num_strategies]
        confidence_scores = torch.stack(confidence_list, dim=1)  # [batch, steps]
        hidden_states = torch.stack(hidden_states_list, dim=1)  # [batch, steps, hidden_dim]

        return NeuralPlannerOutput(
            steps=steps,
            strategy_probs=strategy_probs,
            confidence_scores=confidence_scores,
            attention_weights=None,  # Could add attention extraction
            hidden_states=hidden_states,
        )

    def compute_loss(
        self,
        planner_output: NeuralPlannerOutput,
        target_strategies: torch.Tensor,
        target_confidences: torch.Tensor | None = None,
        strategy_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute training loss for the neural planner.

        Args:
            planner_output: Output from forward pass
            target_strategies: [batch, num_steps] ground truth strategy indices
            target_confidences: [batch, num_steps] ground truth confidences (optional)
            strategy_weights: [num_strategies] class weights for imbalanced data

        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss components
        """
        device = planner_output.strategy_probs.device

        # Strategy classification loss (cross-entropy)
        strategy_probs = planner_output.strategy_probs  # [batch, steps, num_strategies]
        batch_size, num_steps, num_strategies = strategy_probs.shape

        # Reshape for cross-entropy
        strategy_probs_flat = strategy_probs.view(-1, num_strategies)
        target_flat = target_strategies[:, :num_steps].reshape(-1)

        if strategy_weights is not None:
            strategy_loss = F.cross_entropy(
                strategy_probs_flat,
                target_flat,
                weight=strategy_weights,
            )
        else:
            strategy_loss = F.cross_entropy(strategy_probs_flat, target_flat)

        # Confidence prediction loss (MSE)
        confidence_loss = torch.tensor(0.0, device=device)
        if target_confidences is not None:
            confidence_scores = planner_output.confidence_scores
            target_conf = target_confidences[:, :num_steps]
            confidence_loss = F.mse_loss(confidence_scores, target_conf)

        # Total loss
        total_loss = strategy_loss + 0.5 * confidence_loss

        loss_dict = {
            "total": total_loss.item(),
            "strategy": strategy_loss.item(),
            "confidence": confidence_loss.item() if isinstance(confidence_loss, torch.Tensor) else confidence_loss,
        }

        return total_loss, loss_dict

    def get_strategy_accuracy(
        self,
        planner_output: NeuralPlannerOutput,
        target_strategies: torch.Tensor,
    ) -> float:
        """Compute strategy prediction accuracy."""
        strategy_probs = planner_output.strategy_probs
        num_steps = strategy_probs.shape[1]

        predicted = strategy_probs.argmax(dim=-1)  # [batch, steps]
        target = target_strategies[:, :num_steps]

        accuracy = (predicted == target).float().mean().item()
        return accuracy


class NeuralPlannerLoss(nn.Module):
    """
    Loss function for training the Neural Planner.

    Combines:
    - Strategy classification loss (cross-entropy)
    - Confidence calibration loss (MSE + calibration)
    - Step continuation loss (binary cross-entropy)
    - Diversity regularization (encourage varied strategies)
    """

    def __init__(
        self,
        strategy_weight: float = 1.0,
        confidence_weight: float = 0.5,
        continuation_weight: float = 0.3,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.strategy_weight = strategy_weight
        self.confidence_weight = confidence_weight
        self.continuation_weight = continuation_weight
        self.diversity_weight = diversity_weight

    def forward(
        self,
        planner_output: NeuralPlannerOutput,
        target_strategies: torch.Tensor,
        target_confidences: torch.Tensor | None = None,
        target_num_steps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            planner_output: Output from neural planner
            target_strategies: Ground truth strategy indices
            target_confidences: Ground truth confidence scores
            target_num_steps: Ground truth number of steps (for continuation loss)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components
        """
        device = planner_output.strategy_probs.device

        # Strategy loss
        strategy_loss, _ = self._compute_strategy_loss(
            planner_output.strategy_probs,
            target_strategies,
        )

        # Confidence loss
        confidence_loss = torch.tensor(0.0, device=device)
        if target_confidences is not None:
            confidence_loss = self._compute_confidence_loss(
                planner_output.confidence_scores,
                target_confidences,
            )

        # Diversity regularization
        diversity_loss = self._compute_diversity_loss(planner_output.strategy_probs)

        # Total loss
        total_loss = (
            self.strategy_weight * strategy_loss
            + self.confidence_weight * confidence_loss
            + self.diversity_weight * diversity_loss
        )

        loss_dict = {
            "total": total_loss.item(),
            "strategy": strategy_loss.item(),
            "confidence": confidence_loss.item() if isinstance(confidence_loss, torch.Tensor) else confidence_loss,
            "diversity": diversity_loss.item(),
        }

        return total_loss, loss_dict

    def _compute_strategy_loss(
        self,
        strategy_probs: torch.Tensor,
        target_strategies: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compute strategy classification loss."""
        batch_size, num_steps, num_strategies = strategy_probs.shape

        # Flatten for cross-entropy
        probs_flat = strategy_probs.view(-1, num_strategies)
        target_flat = target_strategies[:, :num_steps].reshape(-1)

        loss = F.cross_entropy(probs_flat, target_flat)

        # Accuracy
        predicted = strategy_probs.argmax(dim=-1)
        accuracy = (predicted == target_strategies[:, :num_steps]).float().mean().item()

        return loss, accuracy

    def _compute_confidence_loss(
        self,
        confidence_scores: torch.Tensor,
        target_confidences: torch.Tensor,
    ) -> torch.Tensor:
        """Compute confidence prediction loss with calibration."""
        num_steps = confidence_scores.shape[1]
        target = target_confidences[:, :num_steps]

        # MSE loss
        mse_loss = F.mse_loss(confidence_scores, target)

        # Calibration loss: predicted confidence should match actual accuracy
        # (simplified version)
        calibration_loss = torch.abs(confidence_scores.mean() - target.mean())

        return mse_loss + 0.1 * calibration_loss

    def _compute_diversity_loss(
        self,
        strategy_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage diverse strategy selection across steps.

        Penalize if all steps use the same strategy.
        """
        # Average strategy distribution across steps
        avg_probs = strategy_probs.mean(dim=1)  # [batch, num_strategies]

        # Entropy of average distribution (higher = more diverse)
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum(dim=-1).mean()

        # We want to maximize entropy, so minimize negative entropy
        # Normalize by max possible entropy
        max_entropy = torch.log(torch.tensor(strategy_probs.shape[-1], dtype=torch.float32))
        diversity_loss = 1.0 - (entropy / max_entropy)

        return diversity_loss


class NeuroSymbolicAdapter:
    """
    Adapter to interface between LangGraph (Text) and Neural Agents (Tensors).

    Supports two modes:
    1. HRM-based: Uses HRMAgent for decomposition (legacy)
    2. Planner-based: Uses NeuralPlanner for strategy generation (trainable)
    """

    def __init__(
        self,
        neural_agent: HRMAgent | None = None,
        neural_planner: NeuralPlanner | None = None,
        embedding_model: Any = None,
        device: str | torch.device = "cpu",
        use_planner: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            neural_agent: Pre-loaded HRMAgent instance.
            neural_planner: Optional NeuralPlanner for trainable planning.
            embedding_model: Model to encode text to embeddings.
            device: Device to run computations on.
            use_planner: If True, prefer NeuralPlanner over HRMAgent.
        """
        self.agent = neural_agent
        self.planner = neural_planner
        self.embedding_model = embedding_model
        self.device = device
        self.use_planner = use_planner and neural_planner is not None

        # Ensure components are on correct device
        if self.agent is not None and hasattr(self.agent, "to"):
            self.agent.to(self.device)
        if self.planner is not None:
            self.planner.to(self.device)

    def _embed_query(self, query: str) -> torch.Tensor:
        """
        Convert text query to tensor embedding.

        Args:
            query: Input text string.

        Returns:
            Tensor of shape [1, seq_len, hidden_dim]
        """
        if self.embedding_model:
            # Assume embedding model returns tensor or numpy
            # implementation specific details would go here
            try:
                embeddings = self.embedding_model.encode(query)
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings)
                return embeddings.unsqueeze(0).to(self.device)
            except Exception:
                # Fallback if encoding fails
                pass

        # Fallback/Mock: Random embedding for architectural demonstration
        # Uses the hidden dimension from agent config if available
        if self.agent is not None and hasattr(self.agent, "config"):
            h_dim = getattr(self.agent.config, "h_dim", 768)
        elif self.planner is not None:
            h_dim = self.planner.hidden_dim
        else:
            h_dim = 768

        # Create a "sequence" based on query length approx
        seq_len = min(len(query.split()) + 2, 32)
        return torch.randn(1, seq_len, h_dim).to(self.device)

    async def process_query(
        self,
        query: str,
        context: str = "",
        return_planner_output: bool = False,
    ) -> dict[str, Any]:
        """
        Process a natural language query through the neural agent or planner.

        Args:
            query: User query string.
            context: Optional context string.
            return_planner_output: If True, include raw planner output for training.

        Returns:
            Dictionary containing the neural plan and metadata.
        """
        full_input = f"{query}\nContext: {context}" if context else query

        # 1. Input Processing (Text -> Tensor)
        input_tensor = self._embed_query(full_input)

        # 2. Choose processing path
        if self.use_planner and self.planner is not None:
            return await self._process_with_planner(input_tensor, query, return_planner_output)
        else:
            return await self._process_with_hrm(input_tensor)

    async def _process_with_planner(
        self,
        input_tensor: torch.Tensor,
        query: str,
        return_planner_output: bool = False,
    ) -> dict[str, Any]:
        """Process query using the Neural Planner."""
        self.planner.eval()

        with torch.no_grad():
            planner_output = self.planner(input_tensor)

        # Convert to neural plan format
        neural_plan = []
        for step in planner_output.steps:
            plan_step = NeuralPlanStep(
                level=step.level,
                description=step.description,
                strategy=step.strategy,
                confidence=step.confidence,
                strategy_logits=step.strategy_logits,
            )
            neural_plan.append(asdict(plan_step))

        result = {
            "neural_plan": neural_plan,
            "ponder_cost": 0.0,  # Planner doesn't use pondering
            "metadata": {
                "source": "neural_planner",
                "num_steps": len(planner_output.steps),
                "tensor_shape": str(tuple(input_tensor.shape)),
                "avg_confidence": planner_output.confidence_scores.mean().item(),
            },
        }

        if return_planner_output:
            result["planner_output"] = planner_output

        return result

    async def _process_with_hrm(self, input_tensor: torch.Tensor) -> dict[str, Any]:
        """Process query using the HRM Agent (legacy path)."""
        if self.agent is None:
            # Return mock response
            return self._get_mock_response(input_tensor)

        self.agent.eval()

        # Handle different agent interfaces or mocks
        try:
            if hasattr(self.agent, "forward"):
                with torch.no_grad():
                    # Request decomposition from HRM
                    output = self.agent.forward(
                        input_tensor,
                        return_decomposition=True,
                        return_ponder_output=True,
                    )
                    subproblems = output.subproblems
                    ponder_cost = output.total_ponder_cost
                    halt_step = output.halt_step
                    convergence = output.convergence_path
                    ponder_output = output.ponder_output
            else:
                raise AttributeError("Agent missing forward method")
        except (AttributeError, NotImplementedError):
            return self._get_mock_response(input_tensor)

        # 3. Output Translation (Tensor -> Text Plan)
        neural_plan = []

        for sp in subproblems:
            # Heuristic mapping from confidence/level to strategy
            # Low confidence -> Needs research/tools
            # High confidence -> Direct execution
            confidence = getattr(sp, "confidence", 0.0)
            if confidence < 0.4:
                strategy = "Deep Research"
            elif confidence < 0.6:
                strategy = "Step By Step"
            elif confidence < 0.8:
                strategy = "Tool Use"
            else:
                strategy = "Direct Answer"

            step = NeuralPlanStep(
                level=getattr(sp, "level", 0),
                description=getattr(sp, "description", "Unknown step"),
                strategy=strategy,
                confidence=confidence,
            )
            neural_plan.append(asdict(step))

        result = {
            "neural_plan": neural_plan,
            "ponder_cost": float(ponder_cost),
            "metadata": {
                "source": "hrm_agent",
                "halt_step": halt_step,
                "convergence_path": convergence,
                "tensor_shape": str(tuple(input_tensor.shape)),
            },
        }

        # Include PonderNet output if available
        if ponder_output is not None:
            result["metadata"]["expected_steps"] = ponder_output.expected_steps
            result["metadata"]["kl_divergence"] = ponder_output.kl_divergence.item()

        return result

    def _get_mock_response(self, input_tensor: torch.Tensor) -> dict[str, Any]:
        """Generate mock response for testing/development."""
        subproblems = [
            type("SubProblem", (), {"level": 0, "description": "Analyze request", "confidence": 0.9}),
            type("SubProblem", (), {"level": 1, "description": "Retrieve information", "confidence": 0.4}),
        ]

        neural_plan = []
        for sp in subproblems:
            strategy = "Deep Research" if getattr(sp, "confidence", 0.0) < 0.6 else "Direct Answer"
            step = NeuralPlanStep(
                level=getattr(sp, "level", 0),
                description=getattr(sp, "description", "Unknown step"),
                strategy=strategy,
                confidence=getattr(sp, "confidence", 0.0),
            )
            neural_plan.append(asdict(step))

        return {
            "neural_plan": neural_plan,
            "ponder_cost": 0.5,
            "metadata": {
                "source": "mock",
                "halt_step": 2,
                "convergence_path": [0.1, 0.8],
                "tensor_shape": str(tuple(input_tensor.shape)),
            },
        }

    def train_mode(self) -> None:
        """Set adapter components to training mode."""
        if self.agent is not None:
            self.agent.train()
        if self.planner is not None:
            self.planner.train()

    def eval_mode(self) -> None:
        """Set adapter components to evaluation mode."""
        if self.agent is not None:
            self.agent.eval()
        if self.planner is not None:
            self.planner.eval()


def create_neural_planner(
    hidden_dim: int = 512,
    num_strategies: int = 5,
    max_steps: int = 8,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    device: str = "cpu",
) -> NeuralPlanner:
    """
    Factory function to create and initialize Neural Planner.

    Args:
        hidden_dim: Hidden dimension for transformer
        num_strategies: Number of decomposition strategies
        max_steps: Maximum planning steps
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        device: Device to place model on

    Returns:
        Initialized NeuralPlanner
    """
    planner = NeuralPlanner(
        hidden_dim=hidden_dim,
        num_strategies=num_strategies,
        max_steps=max_steps,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)

    planner.apply(init_weights)
    planner.to(device)

    return planner
