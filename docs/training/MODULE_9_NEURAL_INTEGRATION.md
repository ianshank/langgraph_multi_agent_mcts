# Module 9: Neural Network Integration

## Overview

This module covers the integration of neural networks with the LangGraph Multi-Agent MCTS Framework, enabling hybrid architectures that combine the reasoning capabilities of LLMs with the efficiency of trained neural networks.

**Duration**: 6-8 hours
**Difficulty**: Advanced
**Prerequisites**: Modules 1-7, Python, PyTorch basics, understanding of neural network training

## Learning Objectives

By the end of this module, you will be able to:

1. Design and implement policy networks for agent action selection
2. Build value networks for position evaluation in MCTS
3. Create hybrid LLM-neural architectures for optimal cost-performance tradeoffs
4. Set up training infrastructure with data collection from MCTS rollouts
5. Deploy neural models to production with monitoring and A/B testing
6. Analyze cost savings and performance tradeoffs between pure LLM and hybrid approaches

## Module Structure

- **Section 1**: Policy Network Architecture (90 min)
- **Section 2**: Value Network Architecture (90 min)
- **Section 3**: Hybrid LLM-Neural Architectures (120 min)
- **Section 4**: Training Infrastructure (90 min)
- **Section 5**: Production Deployment (90 min)
- **Labs**: Hands-on Implementation (120 min)
- **Assessment**: Final Project and Quiz (60 min)

---

## Section 1: Policy Network Architecture

### 1.1 Introduction to Policy Networks

**Policy networks** learn to select actions directly from states, bypassing expensive LLM calls for routine decisions. In the MCTS context, policy networks guide tree exploration by predicting promising actions.

**Key Concepts**:
- **Input**: State representations (board positions, conversation context, etc.)
- **Output**: Action probabilities or Q-values
- **Training**: Supervised learning from expert trajectories or self-play

**Architecture Design Principles**:

1. **Input Encoding**: Convert raw states to vector representations
   - For chess: Board tensors (8x8x12 for piece positions)
   - For dialogue: Embedding vectors from conversation history
   - For code: AST features or token embeddings

2. **Network Depth**: Balance expressiveness with inference speed
   - Shallow (2-4 layers): Fast inference, good for simple domains
   - Deep (8-16 layers): Better generalization, higher latency

3. **Output Layer**: Match action space structure
   - Discrete actions: Softmax over action space
   - Continuous actions: Gaussian policy with mean/std output

### 1.2 Policy Network Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    Policy network for action selection in MCTS.

    Maps state representations to action probabilities,
    enabling fast action selection without LLM calls.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256, 128],
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build feature extraction layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Policy head: outputs action logits
        self.policy_head = nn.Linear(prev_dim, action_dim)

        # Value head (optional): estimate state value
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.

        Args:
            state: [batch, state_dim] state tensor

        Returns:
            policy_logits: [batch, action_dim] unnormalized action scores
            state_value: [batch, 1] estimated value of state
        """
        features = self.feature_extractor(state)
        policy_logits = self.policy_head(features)
        state_value = self.value_head(features)

        return policy_logits, state_value

    def select_action(
        self,
        state: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> tuple[int, float]:
        """
        Select action using the policy network.

        Args:
            state: [state_dim] state tensor
            temperature: Exploration parameter (lower = more greedy)
            top_k: If set, sample from top-k actions only

        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            policy_logits, _ = self.forward(state)
            policy_logits = policy_logits / temperature

            # Apply top-k filtering if requested
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(policy_logits, top_k, dim=-1)
                # Create mask for top-k
                mask = torch.full_like(policy_logits, float('-inf'))
                mask.scatter_(-1, top_k_indices, top_k_logits)
                policy_logits = mask

            # Sample from policy
            probs = F.softmax(policy_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            log_prob = F.log_softmax(policy_logits, dim=-1)[0, action].item()

            return action, log_prob

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over actions."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            policy_logits, _ = self.forward(state)
            return F.softmax(policy_logits, dim=-1)
```

### 1.3 Integration with HRM/TRM Agents

The policy network integrates with existing agents by:

1. **Action Priors**: Provide initial action distribution to MCTS
2. **Fast Rollouts**: Replace LLM calls during simulation
3. **Hybrid Selection**: Use policy network for common cases, LLM for novel situations

**Example Integration**:

```python
class PolicyGuidedMCTS:
    """MCTS with policy network guidance."""

    def __init__(self, policy_net: PolicyNetwork, llm_client, threshold: float = 0.8):
        self.policy_net = policy_net
        self.llm_client = llm_client
        self.confidence_threshold = threshold

    async def select_action(self, state):
        # Get policy network prediction
        action_probs = self.policy_net.get_action_probs(state)
        max_prob = action_probs.max().item()

        # Use policy network if confident, otherwise query LLM
        if max_prob >= self.confidence_threshold:
            action, log_prob = self.policy_net.select_action(state)
            return action, {"source": "policy_net", "confidence": max_prob}
        else:
            # Fall back to LLM for uncertain cases
            action = await self.llm_client.get_action(state)
            return action, {"source": "llm", "confidence": None}
```

### 1.4 Training Data Collection

Policy networks require training data from MCTS rollouts:

**Data Format**:
```python
@dataclass
class PolicyTrainingExample:
    state: torch.Tensor          # State representation
    action: int                  # Action taken
    mcts_policy: torch.Tensor    # MCTS visit counts (target distribution)
    outcome: float               # Final game outcome or reward
    metadata: dict               # Additional context
```

**Collection Strategy**:
1. Run MCTS with LLM to generate high-quality trajectories
2. Record state-action pairs with MCTS visit counts
3. Store final outcomes for reward signal
4. Balance dataset across different state types

### Lab 1.1: Implement and Test Policy Network

**Objective**: Build a policy network for a simple environment (e.g., Tic-Tac-Toe)

**Tasks**:
1. Define state encoding for Tic-Tac-Toe (9-dimensional one-hot)
2. Implement PolicyNetwork with appropriate dimensions
3. Generate synthetic training data from random games
4. Train the policy network using cross-entropy loss
5. Evaluate accuracy on held-out test set

**Expected Outcome**: Policy network achieving >80% accuracy on legal move prediction

**Time**: 30 minutes

---

## Section 2: Value Network Architecture

### 2.1 Introduction to Value Networks

**Value networks** estimate the expected outcome from a given state, enabling efficient position evaluation in MCTS without full tree search.

**Key Concepts**:
- **Input**: State representation
- **Output**: Scalar value (expected reward or win probability)
- **Training**: Regression on actual game outcomes or TD learning

**Use Cases in MCTS**:
1. **Node Evaluation**: Quickly assess leaf nodes without rollouts
2. **Pruning**: Skip unpromising branches
3. **UCB Balancing**: Improve exploration-exploitation tradeoff

### 2.2 Value Network Architecture

```python
class ValueNetwork(nn.Module):
    """
    Value network for position evaluation in MCTS.

    Estimates expected reward from current state,
    enabling faster tree search with fewer simulations.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
        output_activation: str = "tanh"  # "tanh" for [-1, 1], "sigmoid" for [0, 1]
    ):
        super().__init__()
        self.state_dim = state_dim
        self.output_activation = output_activation

        # Build feature extraction layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Value head: single scalar output
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.

        Args:
            state: [batch, state_dim] state tensor

        Returns:
            value: [batch, 1] estimated value
        """
        features = self.feature_extractor(state)
        value = self.value_head(features)

        # Apply output activation
        if self.output_activation == "tanh":
            value = torch.tanh(value)
        elif self.output_activation == "sigmoid":
            value = torch.sigmoid(value)

        return value

    def evaluate(self, state: torch.Tensor) -> float:
        """
        Evaluate a single state.

        Args:
            state: [state_dim] state tensor

        Returns:
            value: Scalar value estimate
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            value = self.forward(state)
            return value.item()
```

### 2.3 Self-Play Data Generation

Value networks are typically trained on self-play data:

**Self-Play Loop**:
1. Agent plays against itself using current policy
2. Record all states and final outcomes
3. Assign values based on outcome (1 for win, 0 for loss, 0.5 for draw)
4. Train value network to predict outcomes
5. Iterate: improved value network → better MCTS → better training data

**Implementation**:

```python
class SelfPlayDataGenerator:
    """Generate training data through self-play."""

    def __init__(self, policy_net: PolicyNetwork, value_net: ValueNetwork):
        self.policy_net = policy_net
        self.value_net = value_net
        self.replay_buffer = []

    async def generate_game(self, max_moves: int = 100) -> list[dict]:
        """
        Play one game and collect training examples.

        Returns:
            examples: List of (state, value) pairs
        """
        examples = []
        state = self.reset_game()

        for move_num in range(max_moves):
            # Record current state
            examples.append({
                "state": state.clone(),
                "move_num": move_num
            })

            # Select action using policy network
            action, _ = self.policy_net.select_action(state)

            # Apply action
            state, done, winner = self.apply_action(state, action)

            if done:
                # Assign final outcome to all states in game
                outcome = self.get_outcome(winner)
                for example in examples:
                    example["value"] = outcome
                break

        return examples

    async def generate_batch(self, num_games: int) -> list[dict]:
        """Generate multiple games in parallel."""
        import asyncio
        games = await asyncio.gather(*[
            self.generate_game() for _ in range(num_games)
        ])

        # Flatten list of games into list of examples
        all_examples = []
        for game in games:
            all_examples.extend(game)

        return all_examples
```

### 2.4 Loss Functions and Training

**Loss Functions**:

1. **Mean Squared Error (MSE)**: Standard regression loss
   ```python
   loss = F.mse_loss(predicted_values, target_values)
   ```

2. **Temporal Difference (TD) Learning**: Learn from bootstrapped estimates
   ```python
   # TD(0): one-step lookahead
   td_target = reward + gamma * value_net(next_state)
   loss = F.mse_loss(value_net(state), td_target)
   ```

3. **Combined Loss with Regularization**:
   ```python
   class ValueLoss(nn.Module):
       def __init__(self, mse_weight=1.0, l2_weight=0.001):
           super().__init__()
           self.mse_weight = mse_weight
           self.l2_weight = l2_weight

       def forward(self, predictions, targets, model):
           # Value prediction loss
           mse_loss = F.mse_loss(predictions, targets)

           # L2 regularization
           l2_reg = sum(p.pow(2).sum() for p in model.parameters())

           total_loss = self.mse_weight * mse_loss + self.l2_weight * l2_reg

           return total_loss, {
               "mse": mse_loss.item(),
               "l2": l2_reg.item()
           }
   ```

### Lab 2.1: Train Value Network on Self-Play Data

**Objective**: Build and train a value network for position evaluation

**Tasks**:
1. Implement SelfPlayDataGenerator for Tic-Tac-Toe
2. Generate 1000 self-play games
3. Train ValueNetwork using MSE loss
4. Evaluate prediction accuracy on held-out games
5. Compare MCTS performance with/without value network

**Expected Outcome**: Value network reducing MCTS simulations by 50% while maintaining accuracy

**Time**: 45 minutes

---

## Section 3: Hybrid LLM-Neural Architectures

### 3.1 Motivation for Hybrid Approaches

**Pure LLM Approach**:
- ✅ Excellent reasoning and generalization
- ✅ Handles novel situations well
- ❌ Expensive (GPT-4: $0.03/1K tokens)
- ❌ Slow (100-500ms latency)

**Pure Neural Approach**:
- ✅ Fast inference (<10ms)
- ✅ Cheap (fraction of cent per 1M inferences)
- ❌ Limited generalization to new scenarios
- ❌ Requires large training datasets

**Hybrid Approach**:
- ✅ Use LLM for high-level reasoning and novel cases
- ✅ Use neural nets for routine decisions
- ✅ Achieve 80-90% cost savings while maintaining quality
- ✅ Adaptive: fall back to LLM when uncertain

### 3.2 Hybrid Agent Architecture

```python
from typing import Literal
from dataclasses import dataclass

@dataclass
class HybridConfig:
    """Configuration for hybrid LLM-neural agent."""

    # Model selection thresholds
    policy_confidence_threshold: float = 0.8
    value_confidence_threshold: float = 0.7

    # Mode selection
    mode: Literal["auto", "neural_only", "llm_only"] = "auto"

    # Cost tracking
    track_costs: bool = True
    neural_cost_per_call: float = 0.000001  # $1e-6
    llm_cost_per_1k_tokens: float = 0.03    # GPT-4 pricing

    # Performance monitoring
    log_decisions: bool = True
    langsmith_project: str | None = None


class HybridAgent(nn.Module):
    """
    Hybrid agent combining LLM reasoning with neural network efficiency.

    Uses neural networks for routine decisions and LLM for complex
    reasoning, achieving optimal cost-performance tradeoff.
    """

    def __init__(
        self,
        policy_net: PolicyNetwork,
        value_net: ValueNetwork,
        llm_client,
        config: HybridConfig
    ):
        super().__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.llm_client = llm_client
        self.config = config

        # Statistics tracking
        self.stats = {
            "neural_calls": 0,
            "llm_calls": 0,
            "total_neural_cost": 0.0,
            "total_llm_cost": 0.0,
            "neural_failures": 0,  # Cases where neural was tried but failed
        }

    async def select_action(
        self,
        state: torch.Tensor,
        context: dict | None = None
    ) -> tuple[int, dict]:
        """
        Select action using hybrid approach.

        Args:
            state: Current state representation
            context: Optional context (conversation history, metadata, etc.)

        Returns:
            action: Selected action
            metadata: Decision metadata (source, confidence, cost, etc.)
        """
        if self.config.mode == "llm_only":
            return await self._llm_select_action(state, context)

        if self.config.mode == "neural_only":
            return self._neural_select_action(state)

        # Auto mode: try neural first, fall back to LLM if uncertain
        action, metadata = self._neural_select_action(state)

        if metadata["confidence"] < self.config.policy_confidence_threshold:
            # Neural network is uncertain, query LLM
            action, llm_metadata = await self._llm_select_action(state, context)
            metadata.update(llm_metadata)
            metadata["decision"] = "llm_fallback"

        return action, metadata

    def _neural_select_action(self, state: torch.Tensor) -> tuple[int, dict]:
        """Select action using policy network."""
        action, log_prob = self.policy_net.select_action(state)
        confidence = torch.exp(torch.tensor(log_prob)).item()

        # Update statistics
        self.stats["neural_calls"] += 1
        self.stats["total_neural_cost"] += self.config.neural_cost_per_call

        metadata = {
            "source": "policy_network",
            "confidence": confidence,
            "cost": self.config.neural_cost_per_call,
            "latency_ms": 5.0  # Typical neural inference latency
        }

        return action, metadata

    async def _llm_select_action(
        self,
        state: torch.Tensor,
        context: dict | None = None
    ) -> tuple[int, dict]:
        """Select action using LLM."""
        # Convert state to natural language prompt
        prompt = self._state_to_prompt(state, context)

        # Query LLM
        import time
        start_time = time.time()
        response = await self.llm_client.generate(prompt)
        latency_ms = (time.time() - start_time) * 1000

        # Parse action from response
        action = self._parse_action(response)

        # Estimate cost (rough approximation)
        num_tokens = len(prompt.split()) + len(response.split())
        cost = (num_tokens / 1000) * self.config.llm_cost_per_1k_tokens

        # Update statistics
        self.stats["llm_calls"] += 1
        self.stats["total_llm_cost"] += cost

        metadata = {
            "source": "llm",
            "confidence": None,  # LLMs don't provide calibrated confidence
            "cost": cost,
            "latency_ms": latency_ms,
            "response": response
        }

        return action, metadata

    async def evaluate_position(
        self,
        state: torch.Tensor,
        use_llm_if_uncertain: bool = True
    ) -> tuple[float, dict]:
        """
        Evaluate position using hybrid approach.

        Returns:
            value: Position evaluation
            metadata: Source and confidence information
        """
        # Get neural network evaluation
        value = self.value_net.evaluate(state)

        # Estimate confidence based on value magnitude
        # (Values near 0.5 are uncertain)
        confidence = abs(value - 0.5) * 2  # Map [0, 1] → [1, 0, 1]

        metadata = {
            "source": "value_network",
            "confidence": confidence,
            "cost": self.config.neural_cost_per_call
        }

        # Fall back to LLM if uncertain and enabled
        if use_llm_if_uncertain and confidence < self.config.value_confidence_threshold:
            llm_value, llm_meta = await self._llm_evaluate(state)

            # Blend neural and LLM estimates
            blended_value = 0.3 * value + 0.7 * llm_value
            metadata["llm_value"] = llm_value
            metadata["neural_value"] = value
            metadata["decision"] = "blended"

            return blended_value, metadata

        return value, metadata

    def get_cost_savings(self) -> dict:
        """
        Calculate cost savings from hybrid approach.

        Compares actual costs to hypothetical pure LLM approach.
        """
        total_calls = self.stats["neural_calls"] + self.stats["llm_calls"]
        actual_cost = self.stats["total_neural_cost"] + self.stats["total_llm_cost"]

        # Estimate cost if all calls were LLM
        avg_llm_cost = (
            self.stats["total_llm_cost"] / self.stats["llm_calls"]
            if self.stats["llm_calls"] > 0
            else 0.05  # Default estimate
        )
        hypothetical_llm_cost = total_calls * avg_llm_cost

        savings = hypothetical_llm_cost - actual_cost
        savings_pct = (savings / hypothetical_llm_cost * 100) if hypothetical_llm_cost > 0 else 0

        return {
            "actual_cost": actual_cost,
            "hypothetical_llm_cost": hypothetical_llm_cost,
            "savings": savings,
            "savings_percentage": savings_pct,
            "neural_percentage": self.stats["neural_calls"] / total_calls * 100 if total_calls > 0 else 0
        }
```

### 3.3 Cost-Performance Tradeoffs

**Tuning the Hybrid Agent**:

| Threshold | Neural % | Cost Savings | Quality |
|-----------|----------|--------------|---------|
| 0.5 | 95% | 92% | ⭐⭐⭐ (Good) |
| 0.7 | 85% | 80% | ⭐⭐⭐⭐ (Very Good) |
| 0.9 | 60% | 55% | ⭐⭐⭐⭐⭐ (Excellent) |

**Optimization Strategies**:

1. **Progressive Fallback**: Try fast methods first
   ```
   Neural Policy → Neural Value → LLM Reasoning → Full MCTS
   ```

2. **Contextual Switching**: Use domain knowledge
   - Simple states → Neural only
   - Novel states → LLM only
   - Ambiguous states → Hybrid

3. **Batch Processing**: Amortize LLM costs
   - Queue uncertain cases
   - Batch LLM queries
   - Process neural cases in parallel

### Lab 3.1: Build and Benchmark Hybrid Agent

**Objective**: Create a hybrid agent and measure cost-performance tradeoffs

**Tasks**:
1. Implement HybridAgent with trained policy and value networks
2. Run 100 test episodes with different confidence thresholds
3. Measure: cost, latency, win rate, neural usage %
4. Plot Pareto frontier of cost vs. performance
5. Identify optimal threshold for your use case

**Expected Outcome**: Achieve 70-80% cost savings with <5% performance degradation

**Time**: 45 minutes

---

## Section 4: Training Infrastructure

### 4.1 Data Collection Pipelines

**Pipeline Architecture**:

```
MCTS Rollouts → Experience Buffer → Data Processing → Training Dataset
      ↓              ↓                    ↓                ↓
  LLM Calls    State-Action Pairs    Augmentation    Train/Val Split
```

**Implementation**:

```python
from collections import deque
import pickle
from pathlib import Path

class ExperienceBuffer:
    """
    Circular buffer for storing training experiences.

    Implements efficient storage and sampling for neural network training.
    """

    def __init__(
        self,
        max_size: int = 100000,
        save_dir: str | None = None
    ):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def add(self, experience: dict):
        """Add single experience to buffer."""
        self.buffer.append(experience)

    def add_batch(self, experiences: list[dict]):
        """Add multiple experiences."""
        self.buffer.extend(experiences)

    def sample(self, batch_size: int) -> list[dict]:
        """Sample random batch of experiences."""
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def get_all(self) -> list[dict]:
        """Get all experiences as list."""
        return list(self.buffer)

    def save(self, filename: str):
        """Save buffer to disk."""
        if self.save_dir is None:
            raise ValueError("save_dir not specified")

        filepath = self.save_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, filename: str):
        """Load buffer from disk."""
        if self.save_dir is None:
            raise ValueError("save_dir not specified")

        filepath = self.save_dir / filename
        with open(filepath, "rb") as f:
            experiences = pickle.load(f)
            self.buffer.extend(experiences)

    def __len__(self) -> int:
        return len(self.buffer)


class DataCollector:
    """
    Collect training data from MCTS rollouts.

    Orchestrates game playing, experience collection, and dataset creation.
    """

    def __init__(
        self,
        mcts_engine,
        llm_client,
        buffer: ExperienceBuffer,
        config: dict
    ):
        self.mcts = mcts_engine
        self.llm = llm_client
        self.buffer = buffer
        self.config = config

        self.games_played = 0
        self.total_moves = 0

    async def collect_game(self) -> list[dict]:
        """
        Play one game and collect experiences.

        Returns:
            experiences: List of (state, action, value) tuples
        """
        experiences = []
        state = self.mcts.reset()

        while not self.mcts.is_terminal(state):
            # Run MCTS to get action distribution
            mcts_policy = await self.mcts.search(state, num_simulations=100)

            # Select action
            action = self.mcts.select_action(mcts_policy)

            # Get value estimate from MCTS
            value = self.mcts.get_value(state)

            # Record experience
            experiences.append({
                "state": state.clone(),
                "action": action,
                "mcts_policy": mcts_policy,
                "value": value,
                "game_id": self.games_played,
                "move_num": len(experiences)
            })

            # Apply action
            state = self.mcts.apply_action(state, action)

        # Get final outcome
        outcome = self.mcts.get_outcome(state)

        # Update values with actual outcome
        for exp in experiences:
            exp["outcome"] = outcome

        self.games_played += 1
        self.total_moves += len(experiences)

        return experiences

    async def collect_batch(
        self,
        num_games: int,
        save_every: int = 100
    ) -> int:
        """
        Collect multiple games in parallel.

        Args:
            num_games: Number of games to collect
            save_every: Save buffer every N games

        Returns:
            total_experiences: Number of experiences collected
        """
        import asyncio
        from tqdm import tqdm

        total_experiences = 0

        for i in tqdm(range(0, num_games, save_every)):
            batch_size = min(save_every, num_games - i)

            # Collect games in parallel
            game_batches = await asyncio.gather(*[
                self.collect_game() for _ in range(batch_size)
            ])

            # Add to buffer
            for game in game_batches:
                self.buffer.add_batch(game)
                total_experiences += len(game)

            # Save checkpoint
            self.buffer.save(f"checkpoint_{self.games_played}.pkl")

        return total_experiences
```

### 4.2 Distributed Training Setup

**Multi-GPU Training with PyTorch DDP**:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_distributed(rank: int, world_size: int, config: dict):
    """
    Distributed training worker.

    Args:
        rank: Process rank (GPU ID)
        world_size: Total number of processes
        config: Training configuration
    """
    setup_distributed(rank, world_size)

    # Create model and move to GPU
    model = PolicyNetwork(**config["model"])
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Create data loader with distributed sampler
    from torch.utils.data import DataLoader, DistributedSampler

    dataset = PolicyDataset(config["data_path"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler)

    # Training loop
    for epoch in range(config["num_epochs"]):
        sampler.set_epoch(epoch)

        for batch in dataloader:
            states, actions, targets = batch
            states = states.to(rank)
            actions = actions.to(rank)
            targets = targets.to(rank)

            # Forward pass
            policy_logits, _ = model(states)
            loss = F.cross_entropy(policy_logits, actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()
```

### 4.3 Model Versioning and Deployment

**MLflow Integration**:

```python
import mlflow
import mlflow.pytorch

class ModelTracker:
    """Track model training and deployment with MLflow."""

    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name: str):
        """Start new MLflow run."""
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        """Log hyperparameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int):
        """Log training metrics."""
        mlflow.log_metrics(metrics, step=step)

    def save_model(self, model: nn.Module, model_name: str):
        """Save model with versioning."""
        mlflow.pytorch.log_model(model, model_name)

    def load_model(self, run_id: str, model_name: str) -> nn.Module:
        """Load model by run ID."""
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.pytorch.load_model(model_uri)

    def end_run(self):
        """End current run."""
        mlflow.end_run()
```

### Lab 4.1: Set Up Training Pipeline

**Objective**: Build end-to-end training pipeline with data collection and versioning

**Tasks**:
1. Implement DataCollector for your domain
2. Collect 1000 training games into ExperienceBuffer
3. Set up distributed training with 2 GPUs (or simulate)
4. Train policy and value networks for 10 epochs
5. Track training with MLflow
6. Save model checkpoints

**Expected Outcome**: Trained models with tracked experiments and version history

**Time**: 60 minutes

---

## Section 5: Production Deployment

### 5.1 Model Serving with FastAPI

**Production Inference Server**:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from contextlib import asynccontextmanager

# Global model registry
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    # Load policy network
    models["policy"] = torch.load("models/policy_net.pt")
    models["policy"].eval()

    # Load value network
    models["value"] = torch.load("models/value_net.pt")
    models["value"].eval()

    yield

    # Cleanup
    models.clear()

app = FastAPI(lifespan=lifespan)

class PredictionRequest(BaseModel):
    state: list[float]
    temperature: float = 1.0

class PredictionResponse(BaseModel):
    action: int
    confidence: float
    value: float
    latency_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Get action and value predictions.

    Args:
        request: State and parameters

    Returns:
        Predictions with metadata
    """
    import time
    start_time = time.time()

    try:
        # Convert to tensor
        state = torch.tensor(request.state, dtype=torch.float32)

        # Get predictions
        with torch.no_grad():
            policy_logits, _ = models["policy"](state.unsqueeze(0))
            value = models["value"](state.unsqueeze(0))

            # Apply temperature
            policy_probs = F.softmax(policy_logits / request.temperature, dim=-1)
            action = torch.argmax(policy_probs, dim=-1).item()
            confidence = policy_probs[0, action].item()

        latency_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            action=action,
            confidence=confidence,
            value=value.item(),
            latency_ms=latency_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }
```

### 5.2 A/B Testing Neural vs LLM Agents

**A/B Test Framework**:

```python
import random
from enum import Enum

class VariantType(Enum):
    CONTROL = "llm_only"
    TREATMENT_A = "neural_only"
    TREATMENT_B = "hybrid_low_threshold"
    TREATMENT_C = "hybrid_high_threshold"

class ABTestFramework:
    """
    A/B testing framework for comparing agent variants.

    Tracks performance metrics for each variant and enables
    data-driven decision making.
    """

    def __init__(self, variants: dict[VariantType, HybridAgent]):
        self.variants = variants
        self.metrics = {v: [] for v in variants}
        self.assignments = {}  # user_id -> variant

    def assign_variant(self, user_id: str) -> VariantType:
        """
        Assign user to variant (sticky assignment).

        Args:
            user_id: Unique user identifier

        Returns:
            variant: Assigned variant
        """
        if user_id in self.assignments:
            return self.assignments[user_id]

        # Random assignment with equal probability
        variant = random.choice(list(self.variants.keys()))
        self.assignments[user_id] = variant

        return variant

    async def run_trial(
        self,
        user_id: str,
        state: torch.Tensor,
        context: dict
    ) -> tuple[int, dict]:
        """
        Run single trial with assigned variant.

        Args:
            user_id: User identifier
            state: Current state
            context: Context information

        Returns:
            action: Selected action
            metadata: Trial metadata including variant
        """
        variant = self.assign_variant(user_id)
        agent = self.variants[variant]

        # Run agent
        import time
        start_time = time.time()
        action, agent_metadata = await agent.select_action(state, context)
        latency = time.time() - start_time

        # Record metrics
        trial_data = {
            "variant": variant.value,
            "latency": latency,
            "cost": agent_metadata.get("cost", 0),
            "source": agent_metadata.get("source"),
            **agent_metadata
        }
        self.metrics[variant].append(trial_data)

        return action, trial_data

    def get_statistics(self) -> dict:
        """
        Compute summary statistics for each variant.

        Returns:
            stats: Nested dict of variant → metric → value
        """
        import numpy as np

        stats = {}

        for variant, trials in self.metrics.items():
            if not trials:
                continue

            latencies = [t["latency"] for t in trials]
            costs = [t["cost"] for t in trials]

            stats[variant.value] = {
                "num_trials": len(trials),
                "avg_latency_ms": np.mean(latencies) * 1000,
                "p95_latency_ms": np.percentile(latencies, 95) * 1000,
                "total_cost": sum(costs),
                "avg_cost": np.mean(costs),
                "neural_percentage": sum(1 for t in trials if t["source"] == "policy_network") / len(trials) * 100
            }

        return stats

    def run_statistical_test(self, metric: str = "latency") -> dict:
        """
        Run statistical significance test between variants.

        Args:
            metric: Metric to compare

        Returns:
            results: Test results with p-values
        """
        from scipy import stats as scipy_stats

        control_data = [t[metric] for t in self.metrics[VariantType.CONTROL]]

        results = {}

        for variant, trials in self.metrics.items():
            if variant == VariantType.CONTROL or not trials:
                continue

            treatment_data = [t[metric] for t in trials]

            # Two-sample t-test
            t_stat, p_value = scipy_stats.ttest_ind(control_data, treatment_data)

            results[variant.value] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }

        return results
```

### 5.3 Performance Monitoring

**Prometheus Metrics**:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
neural_predictions = Counter(
    'neural_predictions_total',
    'Total neural network predictions',
    ['model_type']
)

llm_calls = Counter(
    'llm_calls_total',
    'Total LLM API calls',
    ['fallback_reason']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency',
    ['model_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

model_confidence = Histogram(
    'model_confidence',
    'Model confidence scores',
    ['model_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

cost_tracker = Counter(
    'inference_cost_dollars',
    'Total inference cost in USD',
    ['source']
)

class MonitoredHybridAgent(HybridAgent):
    """HybridAgent with Prometheus monitoring."""

    async def select_action(self, state, context=None):
        import time

        # Get action from parent class
        start_time = time.time()
        action, metadata = await super().select_action(state, context)
        latency = time.time() - start_time

        # Record metrics
        source = metadata["source"]

        if source == "policy_network":
            neural_predictions.labels(model_type="policy").inc()
        else:
            llm_calls.labels(fallback_reason="low_confidence").inc()

        prediction_latency.labels(model_type=source).observe(latency)

        if metadata.get("confidence"):
            model_confidence.labels(model_type=source).observe(metadata["confidence"])

        cost_tracker.labels(source=source).inc(metadata["cost"])

        return action, metadata
```

**LangSmith Integration**:

```python
from langsmith import Client
from langsmith.run_helpers import traceable

class LangSmithIntegration:
    """Integrate neural network calls with LangSmith tracing."""

    def __init__(self, project_name: str):
        self.client = Client()
        self.project_name = project_name

    @traceable(name="neural_policy_prediction")
    async def traced_policy_prediction(
        self,
        policy_net: PolicyNetwork,
        state: torch.Tensor
    ) -> tuple[int, float]:
        """Policy prediction with LangSmith tracing."""
        action, log_prob = policy_net.select_action(state)
        confidence = torch.exp(torch.tensor(log_prob)).item()

        # Log additional metadata
        self.client.create_run(
            name="policy_prediction",
            run_type="chain",
            inputs={"state_shape": state.shape},
            outputs={"action": action, "confidence": confidence},
            project_name=self.project_name
        )

        return action, log_prob

    @traceable(name="neural_value_prediction")
    async def traced_value_prediction(
        self,
        value_net: ValueNetwork,
        state: torch.Tensor
    ) -> float:
        """Value prediction with LangSmith tracing."""
        value = value_net.evaluate(state)

        self.client.create_run(
            name="value_prediction",
            run_type="chain",
            inputs={"state_shape": state.shape},
            outputs={"value": value},
            project_name=self.project_name
        )

        return value
```

### Lab 5.1: Deploy and Monitor Hybrid Agent

**Objective**: Deploy hybrid agent to production with full monitoring

**Tasks**:
1. Package policy and value networks with FastAPI server
2. Deploy server locally (or to cloud)
3. Set up Prometheus metrics collection
4. Configure LangSmith tracing for neural predictions
5. Run A/B test with 3 variants: pure LLM, pure neural, hybrid
6. Collect metrics and determine winning variant

**Expected Outcome**: Production-ready deployment with comprehensive monitoring and A/B test results

**Time**: 60 minutes

---

## Labs: Hands-On Implementation

### Lab 1: Complete Policy Network Pipeline

**Duration**: 45 minutes

**Objective**: Implement, train, and evaluate a policy network

**Steps**:
1. Define a simple environment (Tic-Tac-Toe or CartPole)
2. Implement state encoding
3. Create PolicyNetwork with appropriate architecture
4. Generate training data (1000 examples)
5. Train using cross-entropy loss
6. Evaluate on test set
7. Visualize learned policy

**Deliverables**:
- Trained policy network checkpoint
- Training curves (loss, accuracy)
- Test set evaluation metrics

### Lab 2: Value Network Self-Play

**Duration**: 60 minutes

**Objective**: Train value network through self-play

**Steps**:
1. Implement game environment with reset/step
2. Create SelfPlayDataGenerator
3. Generate 500 self-play games
4. Train ValueNetwork on outcomes
5. Compare value predictions to actual outcomes
6. Integrate with MCTS and measure speedup

**Deliverables**:
- Self-play dataset
- Trained value network
- MCTS performance comparison

### Lab 3: Hybrid Agent Implementation

**Duration**: 60 minutes

**Objective**: Build and benchmark hybrid agent

**Steps**:
1. Load trained policy and value networks
2. Implement HybridAgent with LLM fallback
3. Test with different confidence thresholds
4. Measure cost, latency, and accuracy
5. Create cost-performance Pareto frontier
6. Select optimal configuration

**Deliverables**:
- HybridAgent implementation
- Performance benchmarks
- Cost analysis report

### Lab 4: Production Deployment

**Duration**: 45 minutes

**Objective**: Deploy hybrid agent with monitoring

**Steps**:
1. Create FastAPI inference server
2. Add Prometheus metrics
3. Set up health checks
4. Load test with 100 concurrent requests
5. Monitor latency and throughput
6. Configure autoscaling (optional)

**Deliverables**:
- Production inference server
- Prometheus dashboard
- Load test results

---

## Assessment

### Quiz (30 minutes)

**Part A: Conceptual Understanding (50 points)**

1. What are the key advantages of hybrid LLM-neural architectures over pure approaches? (10 points)

2. Explain the difference between policy networks and value networks in MCTS. (10 points)

3. How does self-play data generation work, and why is it effective for training value networks? (10 points)

4. What factors determine when to use neural networks vs. LLM in a hybrid agent? (10 points)

5. Describe three monitoring metrics essential for production deployment of neural agents. (10 points)

**Part B: Code Analysis (25 points)**

Given the following code snippet, identify and fix three issues:

```python
class BrokenPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)

    def select_action(self, state):
        logits = self.forward(state)
        return torch.argmax(logits).item()
```

**Part C: Practical Application (25 points)**

Design a hybrid agent for a customer service chatbot that:
- Uses neural networks for FAQ matching
- Falls back to LLM for complex queries
- Maintains <100ms p95 latency
- Achieves 80% cost reduction

Describe your architecture, thresholds, and monitoring approach.

### Practical Project (60 minutes)

**Objective**: Build a complete neural-enhanced MCTS system

**Requirements**:
1. Implement policy and value networks for Connect-4 or similar game
2. Collect training data through self-play (minimum 500 games)
3. Train both networks to convergence
4. Create hybrid agent with configurable thresholds
5. Deploy with FastAPI server and Prometheus monitoring
6. Run A/B test comparing pure LLM vs. hybrid
7. Document results and recommendations

**Grading Rubric** (100 points):
- Code quality and documentation: 20 points
- Network architecture design: 15 points
- Training pipeline implementation: 15 points
- Hybrid agent logic: 15 points
- Monitoring and metrics: 15 points
- A/B test methodology: 10 points
- Results analysis and insights: 10 points

**Pass Criteria**: 70+ points

---

## Success Metrics

Upon completing Module 9, you should achieve:

### Technical Skills
- ✅ Implement policy and value networks in PyTorch
- ✅ Set up self-play training pipelines
- ✅ Build hybrid LLM-neural agents
- ✅ Deploy models with FastAPI
- ✅ Configure Prometheus monitoring
- ✅ Run A/B tests with statistical analysis

### Performance Targets
- ✅ 70-90% cost reduction with hybrid approach
- ✅ <10ms neural inference latency
- ✅ >80% accuracy on policy prediction
- ✅ <0.1 MSE on value prediction
- ✅ <5% quality degradation vs. pure LLM

### Production Readiness
- ✅ Deployable inference server
- ✅ Comprehensive monitoring
- ✅ Model versioning with MLflow
- ✅ A/B testing framework
- ✅ Cost tracking and optimization

---

## Additional Resources

### Papers
- "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (Silver et al., 2017)
- "Language Models are Few-Shot Learners" (Brown et al., 2020)
- "Efficient Training of Language Models to Fill in the Middle" (Bavarian et al., 2022)

### Libraries
- PyTorch: https://pytorch.org/docs/stable/index.html
- FastAPI: https://fastapi.tiangolo.com/
- MLflow: https://mlflow.org/docs/latest/index.html
- Prometheus: https://prometheus.io/docs/introduction/overview/

### Example Projects
- AlphaZero (DeepMind): https://github.com/suragnair/alpha-zero-general
- MuZero (DeepMind): https://github.com/werner-duvaud/muzero-general
- LangChain Agents: https://python.langchain.com/docs/modules/agents/

### Community
- LangGraph Discord: Join for discussions on multi-agent systems
- PyTorch Forums: https://discuss.pytorch.org/
- r/MachineLearning: Reddit community for ML practitioners

---

## Troubleshooting

### Common Issues

**Issue 1: Policy network overfitting**
- **Symptoms**: High training accuracy, low test accuracy
- **Solutions**:
  - Increase dropout rate (try 0.2-0.3)
  - Add L2 regularization
  - Collect more diverse training data
  - Use data augmentation

**Issue 2: Value network giving constant predictions**
- **Symptoms**: All values near 0.5, high MSE
- **Solutions**:
  - Check data distribution (should have wins/losses/draws)
  - Reduce learning rate
  - Verify loss function implementation
  - Increase network capacity

**Issue 3: Hybrid agent always using LLM**
- **Symptoms**: No cost savings, high latency
- **Solutions**:
  - Lower confidence threshold
  - Check if neural models are loaded correctly
  - Verify state encoding matches training
  - Add logging to debug decision logic

**Issue 4: FastAPI server timing out**
- **Symptoms**: 504 errors, slow responses
- **Solutions**:
  - Move model to GPU
  - Batch predictions
  - Use async inference
  - Enable model quantization

**Issue 5: Prometheus metrics not appearing**
- **Symptoms**: Empty dashboards
- **Solutions**:
  - Check `/metrics` endpoint directly
  - Verify Prometheus scrape config
  - Ensure metrics are incremented in code
  - Check firewall rules

---

## Next Steps

After completing Module 9, you are ready for:

1. **Advanced Topics**:
   - Continuous learning and model updates
   - Multi-task learning across domains
   - Transfer learning from large language models
   - Federated learning for privacy

2. **Integration Projects**:
   - Integrate neural components with existing MCTS framework
   - Build domain-specific hybrid agents
   - Deploy at scale with Kubernetes
   - Implement continuous training pipelines

3. **Research Directions**:
   - Explore uncertainty quantification
   - Investigate active learning strategies
   - Experiment with model distillation
   - Study emergent capabilities

4. **Certification Path**:
   - Complete all 9 modules
   - Pass final comprehensive exam
   - Build capstone project
   - Receive framework certification

---

## Appendix A: Mathematical Foundations

### Policy Gradient Theorem

The policy gradient for a stochastic policy π_θ is:

```
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) Q^π(s,a)]
```

Where:
- J(θ) is the expected return
- π_θ(a|s) is the policy
- Q^π(s,a) is the action-value function

### Value Function Approximation

The value network approximates V^π(s):

```
V_θ(s) ≈ V^π(s) = E_π[R_t | s_t = s]
```

Training objective (MSE):

```
L(θ) = E[(V_θ(s) - R_t)^2]
```

### UCB in MCTS with Value Network

Modified UCB score with value network:

```
UCB(s, a) = Q(s, a) + c * P(s, a) * sqrt(N(s)) / (1 + N(s, a)) + V_θ(s')
```

Where:
- Q(s, a) is the action-value
- P(s, a) is the policy prior
- V_θ(s') is the value network estimate
- c is exploration constant

---

## Appendix B: Code Templates

### Complete Training Script Template

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

class TrainingPipeline:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["num_epochs"]
        )

        # Initialize wandb
        wandb.init(project=config["project_name"], config=config)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            states, targets = batch
            states = states.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(states)
            loss = self.compute_loss(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                states, targets = batch
                states = states.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(states)
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')

        for epoch in range(self.config["num_epochs"]):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Log metrics
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch": epoch
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        wandb.finish()
```

---

## Conclusion

Module 9 provides comprehensive coverage of neural network integration with LangGraph Multi-Agent MCTS Framework. By combining the reasoning power of LLMs with the efficiency of neural networks, you can build production-ready systems that achieve optimal cost-performance tradeoffs.

**Key Takeaways**:
1. Policy networks enable fast action selection
2. Value networks accelerate MCTS with position evaluation
3. Hybrid architectures achieve 70-90% cost savings
4. Production deployment requires monitoring and A/B testing
5. Continuous improvement through data collection and retraining

**Next Steps**:
- Complete all lab exercises
- Pass the assessment with 70+ points
- Deploy your hybrid agent to production
- Share learnings with the community

Good luck, and happy building!
