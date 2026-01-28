"""
Hierarchical Reasoning Model (HRM) Agent.

Implements the HRM architecture with:
- H-Module: High-level planning and decomposition
- L-Module: Low-level execution and refinement
- Adaptive Computation Time (ACT) for dynamic depth
- PonderNet-style trainable halting with KL divergence loss
- Halting mechanism based on confidence thresholds

Based on: "Hierarchical Reasoning for Compositional Generalization"
         "PonderNet: Learning to Ponder" (Banino et al., 2021)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.system_config import HRMConfig


@dataclass
class SubProblem:
    """Represents a decomposed subproblem in the hierarchy."""

    level: int  # Hierarchy level (0 = root, higher = more abstract)
    description: str  # Natural language description
    state: torch.Tensor  # Latent state representation
    parent_id: int | None = None  # Parent subproblem ID
    confidence: float = 0.0  # Confidence in this decomposition


@dataclass
class PonderNetOutput:
    """Output from PonderNet-style processing with per-step information."""

    halt_probs: list[torch.Tensor]  # Halting probability at each step
    step_outputs: list[torch.Tensor]  # Intermediate outputs at each step
    halting_distribution: torch.Tensor  # p(halt at step n) for all steps
    expected_steps: float  # Expected number of pondering steps
    kl_divergence: torch.Tensor  # KL divergence from geometric prior


@dataclass
class HRMOutput:
    """Output from HRM processing."""

    final_state: torch.Tensor  # Final processed state
    subproblems: list[SubProblem]  # Hierarchical decomposition
    halt_step: int  # Step at which halting occurred
    total_ponder_cost: float  # Total computation cost (for training)
    convergence_path: list[float]  # Confidence at each step
    ponder_output: PonderNetOutput | None = None  # PonderNet details for training


class PonderNet(nn.Module):
    """
    PonderNet-style Adaptive Computation with trainable halting.

    Implements the PonderNet algorithm from Banino et al. (2021):
    - Learns when to stop "thinking" via a halting policy
    - Uses geometric prior for regularization (KL divergence)
    - Supports reconstruction loss at each pondering step

    The key insight is that halting probability follows a geometric distribution,
    and we regularize towards this prior to encourage efficient computation.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_ponder_steps: int = 16,
        lambda_p: float = 0.01,
        geometric_prior_p: float = 0.5,
        epsilon: float = 1e-6,
    ):
        """
        Initialize PonderNet.

        Args:
            hidden_dim: Dimension of hidden states
            max_ponder_steps: Maximum number of pondering steps
            lambda_p: Weight for KL divergence regularization
            geometric_prior_p: Parameter p for geometric prior (higher = fewer steps)
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_ponder_steps = max_ponder_steps
        self.lambda_p = lambda_p
        self.geometric_prior_p = geometric_prior_p
        self.epsilon = epsilon

        # Halting unit: predicts probability of halting at each step
        self.halt_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Step embedding for position-aware halting
        self.step_embedding = nn.Embedding(max_ponder_steps, hidden_dim)

        # Learnable temperature for halting distribution
        self.temperature = nn.Parameter(torch.ones(1))

    def compute_geometric_prior(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """
        Compute geometric prior distribution over halting steps.

        p(halt at step n) = p * (1-p)^(n-1) for n = 1, 2, ...

        Args:
            num_steps: Number of steps to compute prior for
            device: Device to place tensor on

        Returns:
            prior: [num_steps] geometric prior probabilities
        """
        p = self.geometric_prior_p
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        prior = p * ((1 - p) ** steps)
        # Normalize to sum to 1 (truncated geometric)
        prior = prior / prior.sum()
        return prior

    def forward(
        self,
        hidden_states: torch.Tensor,
        step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute halting probability for current step.

        Args:
            hidden_states: [batch, seq, hidden_dim] current hidden states
            step: Current pondering step index

        Returns:
            halt_prob: [batch, seq] probability of halting at this step
            halt_logits: [batch, seq] raw logits before sigmoid
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Add step embedding
        step_emb = self.step_embedding(torch.tensor([step], device=device))
        step_emb = step_emb.expand(batch_size, seq_len, -1)

        # Combine hidden state with step embedding
        combined = hidden_states + step_emb

        # Compute halting logits
        halt_logits = self.halt_fc(combined).squeeze(-1)  # [batch, seq]

        # Apply temperature scaling
        halt_logits = halt_logits / (self.temperature + self.epsilon)

        # Sigmoid to get probability
        halt_prob = torch.sigmoid(halt_logits)

        return halt_prob, halt_logits

    def compute_halting_distribution(
        self,
        halt_probs: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the halting distribution p(halt at step n) from per-step halt probs.

        p(halt at step n) = lambda_n * prod_{i<n}(1 - lambda_i)

        where lambda_n is the halting probability at step n.

        Args:
            halt_probs: List of [batch, seq] tensors, one per step

        Returns:
            halting_dist: [batch, seq, num_steps] halting distribution
        """
        if not halt_probs:
            return torch.zeros(1)

        num_steps = len(halt_probs)
        batch_size, seq_len = halt_probs[0].shape
        device = halt_probs[0].device

        # Stack halt probs: [num_steps, batch, seq]
        stacked = torch.stack(halt_probs, dim=0)

        # Compute cumulative product of (1 - lambda_i) for i < n
        # survival_prob[n] = prod_{i<n}(1 - lambda_i)
        # Use list to avoid in-place modification issues
        survival_list = [torch.ones(batch_size, seq_len, device=device)]

        for n in range(1, num_steps):
            # Compute survival probability for step n (no in-place ops)
            prev_survival = survival_list[n - 1]
            curr_survival = prev_survival * (1 - stacked[n - 1] + self.epsilon)
            survival_list.append(curr_survival)

        # Stack survival probabilities: [num_steps, batch, seq]
        survival_probs = torch.stack(survival_list, dim=0)

        # p(halt at step n) = lambda_n * survival_prob[n]
        halting_dist = stacked * survival_probs

        # Normalize (in case of numerical issues)
        halting_dist = halting_dist / (halting_dist.sum(dim=0, keepdim=True) + self.epsilon)

        # Reshape to [batch, seq, num_steps]
        halting_dist = halting_dist.permute(1, 2, 0)

        return halting_dist

    def compute_kl_divergence(
        self,
        halting_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between halting distribution and geometric prior.

        KL(q || p) = sum_n q(n) * log(q(n) / p(n))

        Args:
            halting_dist: [batch, seq, num_steps] learned halting distribution

        Returns:
            kl_div: Scalar KL divergence (averaged over batch and sequence)
        """
        num_steps = halting_dist.shape[-1]
        device = halting_dist.device

        # Get geometric prior
        prior = self.compute_geometric_prior(num_steps, device)

        # Clamp for numerical stability
        halting_dist = halting_dist.clamp(min=self.epsilon)
        prior = prior.clamp(min=self.epsilon)

        # KL divergence
        kl_div = halting_dist * (torch.log(halting_dist) - torch.log(prior))

        # Sum over steps, average over batch and sequence
        kl_div = kl_div.sum(dim=-1).mean()

        return kl_div

    def compute_expected_steps(self, halting_dist: torch.Tensor) -> float:
        """
        Compute expected number of pondering steps.

        E[N] = sum_n n * p(halt at step n)

        Args:
            halting_dist: [batch, seq, num_steps] halting distribution

        Returns:
            expected_steps: Expected number of steps (scalar)
        """
        num_steps = halting_dist.shape[-1]
        device = halting_dist.device

        # Step indices (1-indexed for interpretation)
        steps = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)

        # E[N] = sum_n n * p(n)
        expected = (halting_dist * steps).sum(dim=-1).mean()

        return expected.item()

    def get_ponder_loss(self, ponder_output: PonderNetOutput) -> torch.Tensor:
        """
        Compute the PonderNet regularization loss.

        This is lambda_p * KL(q || p) where:
        - q is the learned halting distribution
        - p is the geometric prior

        Args:
            ponder_output: Output from pondering process

        Returns:
            ponder_loss: Weighted KL divergence loss
        """
        return self.lambda_p * ponder_output.kl_divergence


class AdaptiveComputationTime(nn.Module):
    """
    Adaptive Computation Time (ACT) mechanism for dynamic depth.

    Allows the model to "ponder" longer on difficult problems by
    dynamically adjusting the number of processing steps.

    This is a simplified version. For full PonderNet training,
    use the PonderNet class instead.
    """

    def __init__(self, hidden_dim: int, epsilon: float = 0.01):
        super().__init__()
        self.epsilon = epsilon

        # Halting unit: predicts probability of halting
        self.halt_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Compute halting probabilities.

        Args:
            hidden_states: [batch, seq, hidden_dim]

        Returns:
            halt_probs: [batch, seq] probability of halting
            ponder_cost: Scalar cost for training
        """
        # Compute halting probabilities
        halt_logits = self.halt_fc(hidden_states)  # [batch, seq, 1]
        halt_probs = halt_logits.squeeze(-1)  # [batch, seq]

        # Ponder cost is the expected number of steps
        ponder_cost = halt_probs.sum(dim=-1).mean()

        return halt_probs, ponder_cost


class HModule(nn.Module):
    """
    H-Module: High-level planning and abstract reasoning.

    Responsible for:
    - Decomposing problems into subproblems
    - Abstract planning and strategy
    - Coordinating L-module executions
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        # Multi-head self-attention for relational reasoning
        self.attention = nn.MultiheadAttention(
            embed_dim=config.h_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.h_dim, config.h_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.h_dim * 4, config.h_dim),
            nn.Dropout(config.dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.h_dim)
        self.norm2 = nn.LayerNorm(config.h_dim)

        # Decomposition head: outputs subproblem structure
        self.decompose_head = nn.Sequential(
            nn.Linear(config.h_dim, config.h_dim),
            nn.ReLU(),
            nn.Linear(config.h_dim, config.h_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through high-level reasoning.

        Args:
            x: [batch, seq, h_dim] input tensor

        Returns:
            [batch, seq, h_dim] processed tensor
        """
        # Self-attention for relational reasoning
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward processing
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

    def decompose(self, x: torch.Tensor) -> torch.Tensor:
        """Generate subproblem representations."""
        return self.decompose_head(x)


class LModule(nn.Module):
    """
    L-Module: Low-level execution and concrete operations.

    Responsible for:
    - Executing concrete operations
    - Processing individual subproblems
    - Generating intermediate results
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        # Projection from H-module to L-module dimension
        self.h_to_l = nn.Linear(config.h_dim, config.l_dim)

        # GRU for sequential processing
        self.gru = nn.GRU(
            input_size=config.l_dim,
            hidden_size=config.l_dim,
            num_layers=config.num_l_layers,
            dropout=config.dropout if config.num_l_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.l_dim, config.l_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.l_dim * 2, config.l_dim),
        )

        # Back-projection to H-module dimension
        self.l_to_h = nn.Linear(config.l_dim, config.h_dim)

    def forward(self, x: torch.Tensor, h_context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute low-level processing.

        Args:
            x: [batch, seq, h_dim] input from H-module
            h_context: Optional hidden state

        Returns:
            output: [batch, seq, l_dim] processed output
            l_to_h: [batch, seq, h_dim] back-projection to H-module
        """
        # Project to L-module dimension
        x_l = self.h_to_l(x)

        # Sequential processing
        gru_out, _ = self.gru(x_l, h_context)

        # Output processing
        output = self.output_proj(gru_out)

        # Back-project to H-module dimension for feedback
        feedback = self.l_to_h(output)

        return output, feedback


class HRMAgent(nn.Module):
    """
    Complete Hierarchical Reasoning Model agent.

    Combines H-module and L-module with ACT/PonderNet for adaptive computation.
    Supports both legacy ACT mode and full PonderNet training mode.
    """

    def __init__(
        self,
        config: HRMConfig,
        device: str = "cpu",
        use_ponder_net: bool = True,
        ponder_lambda: float = 0.01,
        geometric_prior_p: float = 0.5,
    ):
        """
        Initialize HRM Agent.

        Args:
            config: HRM configuration
            device: Device to place model on
            use_ponder_net: If True, use PonderNet for training. If False, use legacy ACT.
            ponder_lambda: Weight for PonderNet KL divergence loss
            geometric_prior_p: Parameter for geometric prior (higher = fewer steps)
        """
        super().__init__()
        self.config = config
        self.device = device
        self.use_ponder_net = use_ponder_net

        # Input embedding
        self.input_proj = nn.Linear(config.h_dim, config.h_dim)

        # Core modules
        self.h_module = nn.ModuleList([HModule(config) for _ in range(config.num_h_layers)])

        self.l_module = LModule(config)

        # Adaptive computation: PonderNet or legacy ACT
        if use_ponder_net:
            self.ponder_net = PonderNet(
                hidden_dim=config.h_dim,
                max_ponder_steps=config.max_ponder_steps,
                lambda_p=ponder_lambda,
                geometric_prior_p=geometric_prior_p,
                epsilon=config.ponder_epsilon,
            )
            self.act = None
        else:
            self.act = AdaptiveComputationTime(config.h_dim, config.ponder_epsilon)
            self.ponder_net = None

        # State integration
        self.integrate = nn.Sequential(
            nn.Linear(config.h_dim * 2, config.h_dim),
            nn.LayerNorm(config.h_dim),
            nn.GELU(),
        )

        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        max_steps: int | None = None,
        return_decomposition: bool = False,
        return_ponder_output: bool = False,
    ) -> HRMOutput:
        """
        Process input through hierarchical reasoning.

        Args:
            x: [batch, seq, h_dim] input tensor
            max_steps: Maximum outer loop steps (defaults to config)
            return_decomposition: Whether to return subproblem decomposition
            return_ponder_output: Whether to return full PonderNet output for training

        Returns:
            HRMOutput containing final state and optional decomposition
        """
        batch_size, seq_len, _ = x.shape
        max_steps = max_steps or self.config.max_outer_steps

        # Initial projection
        h_state = self.input_proj(x)

        # Tracking
        subproblems = []
        convergence_path = []
        total_ponder_cost = 0.0

        # PonderNet tracking
        halt_probs_list = []
        step_outputs_list = []

        # Outer loop: iterative refinement
        for step in range(max_steps):
            # H-module: high-level planning
            for h_layer in self.h_module:
                h_state = h_layer(h_state)

            # Store intermediate output for PonderNet reconstruction loss
            if return_ponder_output:
                step_outputs_list.append(h_state.clone())

            # Check halting condition
            if self.use_ponder_net and self.ponder_net is not None:
                halt_probs, halt_logits = self.ponder_net(h_state, step)
                halt_probs_list.append(halt_probs)
                ponder_cost = halt_probs.mean().item()
            else:
                halt_probs, ponder_cost = self.act(h_state)

            total_ponder_cost += ponder_cost

            # Average halting probability across sequence
            avg_halt_prob = halt_probs.mean().item()
            convergence_path.append(avg_halt_prob)

            # Generate subproblem decomposition if requested
            if return_decomposition:
                subproblem_repr = self.h_module[0].decompose(h_state)
                # Create subproblem entries (simplified)
                for i in range(min(3, seq_len)):  # Top 3 subproblems
                    subproblems.append(
                        SubProblem(
                            level=step,
                            description=f"Subproblem at step {step}, position {i}",
                            state=subproblem_repr[:, i, :].detach(),
                            confidence=halt_probs[:, i].mean().item(),
                        )
                    )

            # Halt if confident enough
            if avg_halt_prob >= self.config.halt_threshold:
                break

            # L-module: low-level execution
            l_output, l_feedback = self.l_module(h_state)

            # Integrate L-module feedback
            h_state = self.integrate(torch.cat([h_state, l_feedback], dim=-1))

        # Compute PonderNet output for training
        ponder_output = None
        if return_ponder_output and self.use_ponder_net and halt_probs_list:
            halting_dist = self.ponder_net.compute_halting_distribution(halt_probs_list)
            kl_div = self.ponder_net.compute_kl_divergence(halting_dist)
            expected_steps = self.ponder_net.compute_expected_steps(halting_dist)

            ponder_output = PonderNetOutput(
                halt_probs=halt_probs_list,
                step_outputs=step_outputs_list,
                halting_distribution=halting_dist,
                expected_steps=expected_steps,
                kl_divergence=kl_div,
            )

        return HRMOutput(
            final_state=h_state,
            subproblems=subproblems,
            halt_step=step + 1,
            total_ponder_cost=total_ponder_cost,
            convergence_path=convergence_path,
            ponder_output=ponder_output,
        )

    async def decompose_problem(self, query: str, state: torch.Tensor) -> list[SubProblem]:
        """
        Decompose a problem into hierarchical subproblems.

        Args:
            query: Natural language problem description
            state: Initial state representation

        Returns:
            List of subproblems in hierarchical order
        """
        # Ensure state is batched
        if state.dim() == 2:
            state = state.unsqueeze(0)  # [1, seq, dim]

        # Forward pass with decomposition
        output = self.forward(state, return_decomposition=True)

        # Add query context to subproblems
        for i, sp in enumerate(output.subproblems):
            sp.description = f"{query} -> Level {sp.level} Subproblem {i}"

        return output.subproblems

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Training utilities
class HRMLoss(nn.Module):
    """
    Combined loss for HRM training.

    Includes:
    - Task loss (e.g., cross-entropy for classification)
    - Ponder cost regularization (encourages efficiency)
    - Consistency loss (encourages stable convergence)
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        ponder_weight: float = 0.01,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.task_weight = task_weight
        self.ponder_weight = ponder_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        hrm_output: HRMOutput,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_loss_fn: nn.Module,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            hrm_output: Output from HRM forward pass
            predictions: Model predictions
            targets: Ground truth targets
            task_loss_fn: Loss function for the task

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Task loss
        task_loss = task_loss_fn(predictions, targets)

        # Ponder cost (encourages efficiency)
        ponder_loss = hrm_output.total_ponder_cost

        # Consistency loss (encourages monotonic convergence)
        if len(hrm_output.convergence_path) > 1:
            conv_tensor = torch.tensor(hrm_output.convergence_path)
            # Penalize non-monotonic increases
            diffs = conv_tensor[1:] - conv_tensor[:-1]
            consistency_loss = F.relu(-diffs).mean()  # Penalize decreases
        else:
            consistency_loss = torch.tensor(0.0)

        # Combine losses
        total_loss = (
            self.task_weight * task_loss + self.ponder_weight * ponder_loss + self.consistency_weight * consistency_loss
        )

        loss_dict = {
            "total": total_loss.item(),
            "task": task_loss.item(),
            "ponder": ponder_loss,
            "consistency": consistency_loss.item(),
            "halt_step": hrm_output.halt_step,
        }

        return total_loss, loss_dict


class PonderNetLoss(nn.Module):
    """
    PonderNet-style loss for training adaptive computation.

    Implements the loss from "PonderNet: Learning to Ponder" (Banino et al., 2021):

    L = L_rec + lambda_p * KL(q || p)

    Where:
    - L_rec is the reconstruction/task loss (weighted sum over pondering steps)
    - KL(q || p) is the KL divergence from geometric prior
    - lambda_p is the regularization weight

    The reconstruction loss is computed as:
    L_rec = sum_n p(halt at n) * L_task(output_n, target)

    This encourages the model to produce correct outputs at each step,
    weighted by the probability of halting at that step.
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        kl_weight: float = 0.01,
        consistency_weight: float = 0.1,
        reconstruction_weight: float = 1.0,
    ):
        """
        Initialize PonderNet loss.

        Args:
            task_weight: Weight for final task loss
            kl_weight: Weight for KL divergence regularization (lambda_p)
            consistency_weight: Weight for convergence consistency loss
            reconstruction_weight: Weight for per-step reconstruction loss
        """
        super().__init__()
        self.task_weight = task_weight
        self.kl_weight = kl_weight
        self.consistency_weight = consistency_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(
        self,
        hrm_output: HRMOutput,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_loss_fn: nn.Module,
        output_head: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute PonderNet loss.

        Args:
            hrm_output: Output from HRM forward pass (must include ponder_output)
            predictions: Final model predictions
            targets: Ground truth targets
            task_loss_fn: Loss function for the task
            output_head: Optional head to apply to intermediate outputs for reconstruction loss

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        device = predictions.device

        # Task loss on final output
        task_loss = task_loss_fn(predictions, targets)

        # Initialize additional losses
        kl_loss = torch.tensor(0.0, device=device)
        reconstruction_loss = torch.tensor(0.0, device=device)
        expected_steps = 0.0

        # PonderNet-specific losses
        if hrm_output.ponder_output is not None:
            ponder_out = hrm_output.ponder_output

            # KL divergence from geometric prior
            kl_loss = ponder_out.kl_divergence
            expected_steps = ponder_out.expected_steps

            # Reconstruction loss: weighted sum over steps
            if output_head is not None and ponder_out.step_outputs:
                halting_dist = ponder_out.halting_distribution  # [batch, seq, num_steps]

                for step_idx, step_output in enumerate(ponder_out.step_outputs):
                    # Get predictions for this step
                    step_pred = output_head(step_output)

                    # Compute loss for this step
                    step_loss = task_loss_fn(step_pred, targets)

                    # Weight by halting probability at this step
                    # Average over batch and sequence
                    if step_idx < halting_dist.shape[-1]:
                        step_weight = halting_dist[:, :, step_idx].mean()
                        reconstruction_loss = reconstruction_loss + step_weight * step_loss

        # Consistency loss (encourages monotonic convergence)
        if len(hrm_output.convergence_path) > 1:
            conv_tensor = torch.tensor(hrm_output.convergence_path, device=device)
            diffs = conv_tensor[1:] - conv_tensor[:-1]
            consistency_loss = F.relu(-diffs).mean()
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        # Combine losses
        total_loss = (
            self.task_weight * task_loss
            + self.kl_weight * kl_loss
            + self.consistency_weight * consistency_loss
            + self.reconstruction_weight * reconstruction_loss
        )

        loss_dict = {
            "total": total_loss.item(),
            "task": task_loss.item(),
            "kl_divergence": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "reconstruction": reconstruction_loss.item()
            if isinstance(reconstruction_loss, torch.Tensor)
            else reconstruction_loss,
            "consistency": consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            "halt_step": hrm_output.halt_step,
            "expected_steps": expected_steps,
            "ponder_cost": hrm_output.total_ponder_cost,
        }

        return total_loss, loss_dict


class CurriculumPonderScheduler:
    """
    Curriculum learning scheduler for PonderNet training.

    Implements a curriculum that:
    1. Starts with fixed pondering steps (no adaptive halting)
    2. Gradually allows more adaptive halting
    3. Increases task difficulty over time

    This helps the model learn stable representations before
    learning when to halt.
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        curriculum_epochs: int = 20,
        initial_lambda_p: float = 0.0,
        final_lambda_p: float = 0.01,
        initial_max_steps: int = 3,
        final_max_steps: int = 16,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            warmup_epochs: Epochs with fixed steps (no halting learning)
            curriculum_epochs: Epochs to transition from fixed to adaptive
            initial_lambda_p: Starting KL weight (0 = no halting pressure)
            final_lambda_p: Final KL weight
            initial_max_steps: Starting max pondering steps
            final_max_steps: Final max pondering steps
        """
        self.warmup_epochs = warmup_epochs
        self.curriculum_epochs = curriculum_epochs
        self.initial_lambda_p = initial_lambda_p
        self.final_lambda_p = final_lambda_p
        self.initial_max_steps = initial_max_steps
        self.final_max_steps = final_max_steps

        self.current_epoch = 0

    def step(self) -> None:
        """Advance to next epoch."""
        self.current_epoch += 1

    def get_lambda_p(self) -> float:
        """Get current KL weight based on curriculum."""
        if self.current_epoch < self.warmup_epochs:
            return self.initial_lambda_p

        progress = min((self.current_epoch - self.warmup_epochs) / self.curriculum_epochs, 1.0)

        return self.initial_lambda_p + progress * (self.final_lambda_p - self.initial_lambda_p)

    def get_max_steps(self) -> int:
        """Get current max pondering steps based on curriculum."""
        if self.current_epoch < self.warmup_epochs:
            return self.initial_max_steps

        progress = min((self.current_epoch - self.warmup_epochs) / self.curriculum_epochs, 1.0)

        return int(self.initial_max_steps + progress * (self.final_max_steps - self.initial_max_steps))

    def get_halt_threshold(self, base_threshold: float = 0.95) -> float:
        """
        Get current halt threshold based on curriculum.

        During warmup, use very high threshold (effectively no early halting).
        Gradually lower to allow adaptive halting.
        """
        if self.current_epoch < self.warmup_epochs:
            return 1.0  # Never halt early during warmup

        progress = min((self.current_epoch - self.warmup_epochs) / self.curriculum_epochs, 1.0)

        # Transition from 1.0 to base_threshold
        return 1.0 - progress * (1.0 - base_threshold)

    def update_ponder_net(self, ponder_net: PonderNet) -> None:
        """Update PonderNet parameters based on curriculum."""
        ponder_net.lambda_p = self.get_lambda_p()

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "curriculum_epochs": self.curriculum_epochs,
            "initial_lambda_p": self.initial_lambda_p,
            "final_lambda_p": self.final_lambda_p,
            "initial_max_steps": self.initial_max_steps,
            "final_max_steps": self.final_max_steps,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.current_epoch = state_dict["current_epoch"]
        self.warmup_epochs = state_dict["warmup_epochs"]
        self.curriculum_epochs = state_dict["curriculum_epochs"]
        self.initial_lambda_p = state_dict["initial_lambda_p"]
        self.final_lambda_p = state_dict["final_lambda_p"]
        self.initial_max_steps = state_dict["initial_max_steps"]
        self.final_max_steps = state_dict["final_max_steps"]


def create_hrm_agent(
    config: HRMConfig,
    device: str = "cpu",
    use_ponder_net: bool = True,
    ponder_lambda: float = 0.01,
    geometric_prior_p: float = 0.5,
) -> HRMAgent:
    """
    Factory function to create and initialize HRM agent.

    Args:
        config: HRM configuration
        device: Device to place model on
        use_ponder_net: If True, use PonderNet for adaptive computation
        ponder_lambda: Weight for PonderNet KL divergence loss
        geometric_prior_p: Parameter for geometric prior (higher = fewer steps)

    Returns:
        Initialized HRMAgent
    """
    agent = HRMAgent(
        config,
        device,
        use_ponder_net=use_ponder_net,
        ponder_lambda=ponder_lambda,
        geometric_prior_p=geometric_prior_p,
    )

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)

    agent.apply(init_weights)

    return agent
