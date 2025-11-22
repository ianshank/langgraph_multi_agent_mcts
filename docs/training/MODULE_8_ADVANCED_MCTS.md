# Module 8: Advanced MCTS Techniques

**Duration:** 8-12 hours
**Difficulty:** Advanced
**Prerequisites:** Modules 1-7, Strong understanding of MCTS fundamentals, Python async programming

## Learning Objectives

By the end of this module, you will be able to:

1. Implement neural-guided MCTS using AlphaZero-style policy and value networks
2. Design and deploy parallel MCTS with virtual loss for concurrent search
3. Apply progressive widening and RAVE (Rapid Action Value Estimation) techniques
4. Integrate adaptive simulation policies for domain-specific optimization
5. Incorporate domain knowledge into MCTS search effectively
6. Evaluate and compare different advanced MCTS variants
7. Monitor and trace advanced MCTS components using LangSmith and Prometheus
8. Optimize MCTS performance for production multi-agent systems

## Prerequisites Review

Before starting this module, ensure you understand:

- Basic MCTS phases: Selection, Expansion, Simulation, Backpropagation
- UCB1 and PUCT selection formulas
- Progressive widening concepts from the existing codebase
- LangGraph agent orchestration patterns
- LangSmith tracing and experiment tracking
- Async/await patterns in Python
- PyTorch neural network basics

## Table of Contents

1. [Neural-Guided MCTS (AlphaZero-Style)](#section-1-neural-guided-mcts)
2. [Progressive Widening and RAVE](#section-2-progressive-widening-and-rave)
3. [Virtual Loss and Parallel MCTS](#section-3-virtual-loss-and-parallel-mcts)
4. [Adaptive Simulation Policies](#section-4-adaptive-simulation-policies)
5. [MCTS with Domain Knowledge](#section-5-mcts-with-domain-knowledge)
6. [Integration with Multi-Agent Systems](#section-6-integration-with-multi-agent-systems)
7. [Performance Optimization and Monitoring](#section-7-performance-optimization)
8. [Labs and Hands-On Exercises](#section-8-labs)
9. [Assessment and Certification](#section-9-assessment)

---

## Section 1: Neural-Guided MCTS (AlphaZero-Style)

### 1.1 Introduction to Neural-Guided MCTS

Neural-guided MCTS revolutionized game-playing AI with AlphaGo Zero and AlphaZero. Instead of using random rollouts, it combines tree search with deep neural networks for:

1. **Policy Network**: Provides prior probabilities for action selection
2. **Value Network**: Estimates position value without full rollouts
3. **PUCT Selection**: Predictor + UCT for exploration/exploitation balance

**Key Insight:** Neural networks learn from self-play, improving both search guidance and evaluation accuracy over time.

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────┐
│          Neural-Guided MCTS                 │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌────────────────┐  │
│  │ Game State   │─────>│ Neural Network │  │
│  │ (tensor)     │      │ (ResNet-based) │  │
│  └──────────────┘      └────────┬───────┘  │
│                                 │           │
│                        ┌────────┴────────┐  │
│                        │                 │  │
│                  ┌─────▼─────┐   ┌──────▼──┐│
│                  │  Policy   │   │  Value  ││
│                  │  Head     │   │  Head   ││
│                  └─────┬─────┘   └──────┬──┘│
│                        │                │   │
│                  ┌─────▼─────┐   ┌──────▼──┐│
│                  │ P(a|s)    │   │  V(s)   ││
│                  │ Prior     │   │  Eval   ││
│                  │ Probs     │   │         ││
│                  └─────┬─────┘   └──────┬──┘│
│                        │                │   │
│                        └────────┬───────┘   │
│                                 ▼           │
│                        ┌────────────────┐   │
│                        │ PUCT Selection │   │
│                        │ & Expansion    │   │
│                        └────────────────┘   │
└─────────────────────────────────────────────┘
```

### 1.3 PUCT Formula Explained

The Predictor + UCT (PUCT) formula balances exploration and exploitation:

```
PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Where:
- `Q(s,a)`: Average value of action `a` from state `s` (exploitation)
- `P(s,a)`: Prior probability from policy network (neural guidance)
- `N(s)`: Visit count of parent node
- `N(s,a)`: Visit count of child node
- `c_puct`: Exploration constant (typically 1.0-2.0)

**Key Differences from UCB1:**
- Uses neural network priors instead of uniform priors
- Scales exploration bonus by prior probability
- Converges faster due to informed initialization

### 1.4 Implementation Deep Dive

Our implementation in `src/framework/mcts/neural_mcts.py` includes:

#### Core Components

1. **NeuralMCTSNode**: Extended node with prior probabilities and virtual loss
   ```python
   class NeuralMCTSNode:
       def __init__(self, state, parent=None, action=None, prior=0.0):
           self.state = state
           self.parent = parent
           self.action = action
           self.prior = prior  # P(s,a) from policy network

           self.visit_count = 0
           self.value_sum = 0.0
           self.virtual_loss = 0.0  # For parallel search

           self.children = {}
           self.is_expanded = False
           self.is_terminal = state.is_terminal()
   ```

2. **Network Evaluation with Caching**
   ```python
   async def evaluate_state(self, state, add_noise=False):
       # Check cache first
       state_hash = state.get_hash()
       if not add_noise and state_hash in self.cache:
           return self.cache[state_hash]

       # Forward pass through network
       state_tensor = state.to_tensor().unsqueeze(0)
       policy_logits, value = self.network(state_tensor)

       # Process and cache
       policy_probs = softmax_with_masking(policy_logits, legal_actions)
       self.cache[state_hash] = (policy_probs, value)

       return policy_probs, value
   ```

3. **Dirichlet Noise for Exploration**
   ```python
   def add_dirichlet_noise(self, policy_probs, epsilon=0.25, alpha=0.3):
       """Add exploration noise at root node only."""
       noise = np.random.dirichlet([alpha] * len(policy_probs))
       return (1 - epsilon) * policy_probs + epsilon * noise
   ```

### 1.5 Search Algorithm

The complete search process:

```python
async def search(self, root_state, num_simulations=1600, temperature=1.0):
    # Create root node
    root = NeuralMCTSNode(state=root_state)

    # Expand root with noise for exploration
    policy_probs, _ = await self.evaluate_state(root_state, add_noise=True)
    legal_actions = root_state.get_legal_actions()
    root.expand(policy_probs, legal_actions)

    # Run simulations
    for _ in range(num_simulations):
        await self._simulate(root)

    # Get action probabilities from visit counts
    action_probs = root.get_action_probs(temperature)

    return action_probs, root
```

#### Single Simulation Step

```python
async def _simulate(self, node):
    path = []

    # 1. SELECTION: Traverse tree using PUCT
    current = node
    while current.is_expanded and not current.is_terminal:
        current.add_virtual_loss(self.config.virtual_loss)
        path.append(current)
        _, current = current.select_child(self.config.c_puct)

    # Add leaf to path
    path.append(current)
    current.add_virtual_loss(self.config.virtual_loss)

    # 2. EVALUATION
    if current.is_terminal:
        value = current.state.get_reward()
    else:
        # Expand and evaluate with network
        policy_probs, value = await self.evaluate_state(current.state)
        if not current.is_expanded:
            legal_actions = current.state.get_legal_actions()
            current.expand(policy_probs, legal_actions)

    # 3. BACKPROPAGATION
    for node_in_path in reversed(path):
        node_in_path.update(value)
        node_in_path.revert_virtual_loss(self.config.virtual_loss)
        value = -value  # Flip for opponent

    return value
```

### 1.6 Temperature-Based Action Selection

Temperature controls exploration vs. exploitation in final action selection:

```python
def get_action_probs(self, temperature=1.0):
    """Convert visit counts to action probabilities."""
    if temperature == 0:
        # Deterministic: select most visited
        visits = {a: child.visit_count for a, child in self.children.items()}
        max_visits = max(visits.values())
        best_actions = [a for a, v in visits.items() if v == max_visits]
        prob = 1.0 / len(best_actions)
        return {a: (prob if a in best_actions else 0.0) for a in self.children}

    # Temperature-scaled visits
    visits = np.array([child.visit_count for child in self.children.values()])
    actions = list(self.children.keys())

    if temperature != 1.0:
        visits = visits ** (1.0 / temperature)

    probs = visits / visits.sum()
    return dict(zip(actions, probs))
```

**Temperature Guidelines:**
- `T = 1.0`: Proportional to visit counts (early game exploration)
- `T = 0.1-0.5`: More deterministic (mid-game)
- `T = 0.0`: Fully deterministic (late game, evaluation)

### 1.7 Self-Play Training Pipeline

Neural MCTS improves through self-play:

```python
class SelfPlayCollector:
    async def play_game(self, initial_state, temperature_threshold=30):
        examples = []
        state = initial_state
        player = 1
        move_count = 0

        while not state.is_terminal():
            # Determine temperature (high early, low late)
            temperature = (
                self.config.temperature_init
                if move_count < temperature_threshold
                else self.config.temperature_final
            )

            # Run MCTS
            action_probs, root = await self.mcts.search(
                state, temperature=temperature, add_root_noise=True
            )

            # Store training example
            examples.append(MCTSExample(
                state=state.to_tensor(),
                policy_target=action_probs,  # Learn from MCTS search
                value_target=0.0,  # Will be filled with game outcome
                player=player
            ))

            # Select and apply action
            action = self.mcts.select_action(action_probs, temperature)
            state = state.apply_action(action)

            player = -player
            move_count += 1

        # Assign game outcome to all examples
        outcome = state.get_reward()
        for example in examples:
            example.value_target = outcome if example.player == 1 else -outcome

        return examples
```

### 1.8 Network Architecture

Policy-Value Network structure:

```python
class PolicyValueNetwork(nn.Module):
    def __init__(self, input_channels, action_size, num_res_blocks=19):
        super().__init__()

        # Shared ResNet backbone
        self.conv_block = ConvBlock(input_channels, 256)
        self.res_blocks = nn.Sequential(
            *[ResBlock(256) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Shared features
        x = self.conv_block(x)
        x = self.res_blocks(x)

        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
```

### 1.9 Integration with LangSmith Tracing

Add tracing to neural MCTS:

```python
from langsmith import traceable

class TracedNeuralMCTS(NeuralMCTS):
    @traceable(name="neural_mcts_search")
    async def search(self, root_state, num_simulations=1600, **kwargs):
        result = await super().search(root_state, num_simulations, **kwargs)

        # Log metrics
        action_probs, root = result
        langsmith.log_metrics({
            "num_simulations": num_simulations,
            "num_children": len(root.children),
            "cache_hit_rate": self.get_cache_stats()["hit_rate"],
            "selected_action": max(action_probs, key=action_probs.get)
        })

        return result

    @traceable(name="network_evaluation")
    async def evaluate_state(self, state, add_noise=False):
        result = await super().evaluate_state(state, add_noise)

        policy_probs, value = result
        langsmith.log_metrics({
            "state_hash": state.get_hash()[:8],
            "value_estimate": float(value),
            "policy_entropy": -np.sum(policy_probs * np.log(policy_probs + 1e-8)),
            "cache_hit": state.get_hash() in self.cache and not add_noise
        })

        return result
```

### 1.10 Practical Considerations

#### When to Use Neural MCTS

**Best for:**
- Games with well-defined state spaces (Go, Chess, Shogi)
- Domains with expensive simulations
- Tasks benefiting from learned heuristics
- Long-horizon planning problems

**Challenges:**
- Requires significant training data
- Network training overhead
- GPU requirements for inference
- State representation design

#### Hyperparameter Tuning

Key parameters and typical values:

| Parameter | Typical Range | Impact |
|-----------|---------------|--------|
| `num_simulations` | 400-3200 | More = better but slower |
| `c_puct` | 1.0-2.5 | Higher = more exploration |
| `dirichlet_epsilon` | 0.15-0.35 | Root exploration mix |
| `dirichlet_alpha` | 0.03-0.3 | Depends on action space size |
| `temperature_init` | 1.0-1.5 | Early game exploration |
| `temperature_final` | 0.0-0.2 | Late game determinism |

### 1.11 Performance Metrics

Monitor these metrics during training:

1. **Search Quality**
   - Average value estimate at root
   - Policy-value agreement
   - Search tree depth and breadth

2. **Network Performance**
   - Policy accuracy vs. MCTS policy
   - Value prediction error
   - Cache hit rate

3. **Self-Play Statistics**
   - Game length distribution
   - Move diversity (entropy)
   - Win rate vs. previous versions

---

## Section 2: Progressive Widening and RAVE

### 2.1 Progressive Widening Theory

Progressive widening controls the branching factor in continuous or large action spaces:

**Problem:** In domains with many actions, expanding all children leads to:
- Shallow, wide trees (poor depth)
- Wasted simulations on poor actions
- Memory explosion

**Solution:** Expand children progressively based on visit count:

```
Expand when: N(s) > k * |C(s)|^α
```

Where:
- `N(s)`: Visit count of state `s`
- `|C(s)|`: Number of currently expanded children
- `k`: Coefficient (typically 1-5)
- `α`: Growth rate exponent (typically 0.3-0.7)

### 2.2 Progressive Widening Variants

#### Standard Progressive Widening

```python
def should_expand_pw(visits, num_children, k=1.0, alpha=0.5):
    """Standard progressive widening criterion."""
    threshold = k * (num_children ** alpha)
    return visits > threshold
```

**Effect of α:**
- `α = 0.5`: Square root growth (balanced)
- `α < 0.5`: More aggressive expansion
- `α > 0.5`: More conservative expansion

#### Adaptive Progressive Widening

Adjust `k` based on search progress:

```python
class AdaptiveProgressiveWidening:
    def __init__(self, k_init=1.0, k_min=0.5, k_max=3.0):
        self.k = k_init
        self.k_min = k_min
        self.k_max = k_max

    def should_expand(self, node, stats):
        threshold = self.k * (len(node.children) ** 0.5)
        should_expand = node.visits > threshold

        # Adapt k based on value variance
        if stats["value_variance"] > 0.3:
            self.k = max(self.k_min, self.k * 0.95)  # More exploration
        else:
            self.k = min(self.k_max, self.k * 1.05)  # Less exploration

        return should_expand
```

### 2.3 RAVE (Rapid Action Value Estimation)

RAVE accelerates learning by sharing information across the tree:

**Key Insight:** An action that works well in one state might work well in similar states.

#### RAVE Formula

```
Q_RAVE(s,a) = (1 - β) * Q_UCB(s,a) + β * Q_AMAF(s,a)
```

Where:
- `Q_UCB(s,a)`: Traditional UCB value
- `Q_AMAF(s,a)`: All-Moves-As-First average (RAVE value)
- `β`: Mixing parameter (decreases with visits)

**AMAF Strategy:**
- Track all actions taken in simulation
- Update statistics for each action as if it was played immediately
- Provides faster initial value estimates

#### RAVE Node Implementation

```python
class RAVENode(MCTSNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RAVE statistics
        self.rave_visits = {}  # action -> visit count
        self.rave_value_sum = {}  # action -> total value

    def update_rave(self, action, value):
        """Update RAVE statistics for action."""
        if action not in self.rave_visits:
            self.rave_visits[action] = 0
            self.rave_value_sum[action] = 0.0

        self.rave_visits[action] += 1
        self.rave_value_sum[action] += value

    def get_rave_value(self, action):
        """Get AMAF value for action."""
        if action not in self.rave_visits or self.rave_visits[action] == 0:
            return 0.0
        return self.rave_value_sum[action] / self.rave_visits[action]

    def select_child_rave(self, c_uct=1.414, rave_constant=300):
        """Select child using RAVE-enhanced UCB."""
        best_score = -float("inf")
        best_child = None

        for action, child in self.children.items():
            # Standard UCB component
            if child.visits == 0:
                return child

            ucb_value = child.value + c_uct * math.sqrt(
                math.log(self.visits) / child.visits
            )

            # RAVE component
            rave_value = self.get_rave_value(action)

            # Mixing parameter β
            beta = child.rave_visits.get(action, 0) / (
                child.visits + child.rave_visits.get(action, 0) +
                rave_constant * child.visits * child.rave_visits.get(action, 1e-6)
            )

            # Combined score
            score = (1 - beta) * ucb_value + beta * rave_value

            if score > best_score:
                best_score = score
                best_child = child

        return best_child
```

#### RAVE Backpropagation

Update both UCB and RAVE statistics:

```python
def backpropagate_rave(self, node, value, simulation_actions):
    """
    Backpropagate with RAVE updates.

    Args:
        node: Leaf node
        value: Simulation value
        simulation_actions: List of actions taken in simulation
    """
    current = node

    while current is not None:
        # Standard UCB update
        current.visits += 1
        current.value_sum += value

        # RAVE updates: update all actions that appeared in simulation
        for action in simulation_actions:
            if action in current.children:
                current.update_rave(action, value)

        current = current.parent
        value = -value  # Flip for opponent
```

### 2.4 RAVE with Progressive Widening

Combine both techniques:

```python
class RAVEProgressiveWideningEngine:
    def __init__(self, k=1.0, alpha=0.5, rave_constant=300):
        self.k = k
        self.alpha = alpha
        self.rave_constant = rave_constant

    async def run_iteration(self, root, action_gen, state_transition, rollout_policy):
        # 1. Selection with RAVE
        leaf = self.select_rave(root)

        # 2. Expansion with progressive widening
        if self.should_expand(leaf):
            leaf = self.expand(leaf, action_gen, state_transition)

        # 3. Simulation (track actions)
        value, actions = await self.simulate_with_tracking(leaf, rollout_policy)

        # 4. Backpropagation with RAVE
        self.backpropagate_rave(leaf, value, actions)

    def should_expand(self, node):
        """Progressive widening criterion."""
        if node.terminal or len(node.available_actions) == len(node.children):
            return False
        threshold = self.k * (len(node.children) ** self.alpha)
        return node.visits > threshold
```

### 2.5 Hybrid UCB1 + RAVE Selection

Optimal selection combines both:

```python
def hybrid_selection_score(node, child, action, c_uct=1.414, rave_equiv=300):
    """
    Compute hybrid UCB + RAVE score.

    Args:
        node: Parent node
        child: Child node to score
        action: Action leading to child
        c_uct: UCB exploration constant
        rave_equiv: RAVE equivalence parameter (k)
    """
    if child.visits == 0:
        return float("inf")

    # UCB component
    ucb_exploitation = child.value
    ucb_exploration = c_uct * math.sqrt(math.log(node.visits) / child.visits)
    ucb_score = ucb_exploitation + ucb_exploration

    # RAVE component
    rave_visits = node.rave_visits.get(action, 0)
    rave_value = node.get_rave_value(action)

    # Mixing parameter: β = rave_visits / (visits + rave_visits + 4*k²*visits*rave_visits)
    # As visits increase, β → 0 (rely more on UCB)
    denominator = (
        child.visits + rave_visits +
        4 * (rave_equiv ** 2) * child.visits * rave_visits / 1e6
    )
    beta = rave_visits / denominator if denominator > 0 else 0.0

    # Combine
    final_score = (1 - beta) * ucb_score + beta * rave_value

    return final_score
```

### 2.6 Performance Benefits

**Progressive Widening:**
- Reduces memory: O(k * n^α) instead of O(n)
- Focuses search on promising actions
- Handles large/continuous action spaces

**RAVE:**
- Faster value learning (fewer simulations needed)
- Better performance in early search
- Particularly effective in games with move ordering

**Combined:**
- 2-5x faster convergence in some domains
- Better handling of large branching factors
- More robust to action space complexity

### 2.7 Tuning Guidelines

#### Progressive Widening Parameters

| Domain | k | α | Rationale |
|--------|---|---|-----------|
| Small action space (<10) | 0.5-1.0 | 0.5 | Moderate expansion |
| Medium action space (10-50) | 1.0-2.0 | 0.5-0.6 | Balanced |
| Large action space (>50) | 2.0-5.0 | 0.6-0.7 | Conservative |
| Continuous actions | 3.0-10.0 | 0.7-0.8 | Very conservative |

#### RAVE Parameters

| Parameter | Typical Value | Impact |
|-----------|---------------|--------|
| `rave_constant` | 100-1000 | Higher = longer RAVE influence |
| Initial β | 0.5-1.0 | RAVE weight at start |
| β decay rate | Automatic | Based on visit counts |

---

## Section 3: Virtual Loss and Parallel MCTS

### 3.1 Introduction to Parallel MCTS

Parallel MCTS accelerates search by running multiple simulations concurrently:

**Challenge:** Naive parallelization leads to:
- Search inefficiency (threads explore same paths)
- Lock contention on tree nodes
- Poor CPU/GPU utilization

**Solution:** Virtual loss temporarily reduces node attractiveness during search.

### 3.2 Virtual Loss Mechanism

**Concept:** When a thread selects a node, add a "virtual loss" to make it less attractive to other threads.

```python
class VirtualLossNode(MCTSNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_loss = 0.0
        self.virtual_loss_count = 0

    @property
    def effective_visits(self):
        """Visits including virtual losses."""
        return self.visits + self.virtual_loss_count

    @property
    def effective_value(self):
        """Value accounting for virtual losses."""
        total_visits = self.effective_visits
        if total_visits == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / total_visits

    def add_virtual_loss(self, loss_value=3.0):
        """Add virtual loss before search."""
        self.virtual_loss += loss_value
        self.virtual_loss_count += 1

    def revert_virtual_loss(self, loss_value=3.0):
        """Remove virtual loss after search."""
        self.virtual_loss -= loss_value
        self.virtual_loss_count -= 1
```

### 3.3 Parallel Search Algorithm

```python
import asyncio
from typing import List

class ParallelMCTS:
    def __init__(self, num_threads=8, virtual_loss=3.0):
        self.num_threads = num_threads
        self.virtual_loss_value = virtual_loss
        self.lock = asyncio.Lock()

    async def parallel_search(
        self,
        root,
        num_simulations,
        action_gen,
        state_transition,
        rollout_policy
    ):
        """Run parallel MCTS with virtual loss."""
        simulations_per_thread = num_simulations // self.num_threads

        # Create thread pool
        tasks = [
            self._worker_thread(
                root, simulations_per_thread,
                action_gen, state_transition, rollout_policy
            )
            for _ in range(self.num_threads)
        ]

        # Run all threads concurrently
        await asyncio.gather(*tasks)

        return root

    async def _worker_thread(
        self,
        root,
        num_sims,
        action_gen,
        state_transition,
        rollout_policy
    ):
        """Single worker thread running simulations."""
        for _ in range(num_sims):
            await self._parallel_simulation(
                root, action_gen, state_transition, rollout_policy
            )

    async def _parallel_simulation(
        self,
        root,
        action_gen,
        state_transition,
        rollout_policy
    ):
        """Single simulation with virtual loss."""
        path = []

        # 1. SELECTION with virtual loss
        current = root
        async with self.lock:
            while current.children and not current.terminal:
                current.add_virtual_loss(self.virtual_loss_value)
                path.append(current)
                current = current.select_child()

            # Add leaf
            current.add_virtual_loss(self.virtual_loss_value)
            path.append(current)

        # 2. EXPANSION (lock-free if expanding different nodes)
        if not current.terminal and current.visits > 0:
            async with self.lock:
                if not current.children:  # Double-check after acquiring lock
                    actions = action_gen(current.state)
                    for action in actions[:1]:  # Expand one child at a time
                        child_state = state_transition(current.state, action)
                        current.add_child(action, child_state)

        # 3. SIMULATION (fully parallel, no locks)
        value = await rollout_policy.evaluate(
            current.state,
            max_depth=10
        )

        # 4. BACKPROPAGATION (requires lock)
        async with self.lock:
            for node in reversed(path):
                node.visits += 1
                node.value_sum += value
                node.revert_virtual_loss(self.virtual_loss_value)
                value = -value
```

### 3.4 Lock-Free Tree Parallelism

Advanced optimization: Use atomic operations to reduce locking:

```python
import threading
from ctypes import c_int32, c_float

class LockFreeNode:
    """MCTS node with atomic operations for lock-free updates."""

    def __init__(self, state):
        self.state = state
        self.children = {}

        # Atomic counters
        self._visits = c_int32(0)
        self._virtual_loss_count = c_int32(0)

        # Protected by mutex only during expansion
        self._value_sum = c_float(0.0)
        self._expansion_lock = threading.Lock()

    @property
    def visits(self):
        return self._visits.value

    def atomic_increment_visits(self):
        """Thread-safe visit increment."""
        with self._expansion_lock:
            self._visits.value += 1

    def atomic_add_value(self, value):
        """Thread-safe value addition."""
        with self._expansion_lock:
            self._value_sum.value += value
```

### 3.5 Root Parallelization

Different parallelization strategy: Run separate searches from root:

```python
class RootParallelMCTS:
    """
    Root parallelization: Each thread maintains its own tree.
    Combine results at the end.
    """

    async def root_parallel_search(self, initial_state, num_threads=8):
        """Run independent searches and merge results."""
        # Create separate engines for each thread
        engines = [
            MCTSEngine(seed=42 + i)
            for i in range(num_threads)
        ]

        # Run parallel searches
        tasks = [
            engine.search(
                root=MCTSNode(MCTSState(initial_state)),
                num_iterations=1000,
                action_generator=self.action_gen,
                state_transition=self.state_transition,
                rollout_policy=self.rollout_policy
            )
            for engine in engines
        ]

        results = await asyncio.gather(*tasks)

        # Merge results: sum visit counts and values
        merged_stats = self._merge_search_results(results)

        return merged_stats

    def _merge_search_results(self, results):
        """Combine statistics from multiple searches."""
        merged = {}

        for action, stats in results:
            for result_stats in results:
                if action in result_stats["action_stats"]:
                    child_stats = result_stats["action_stats"][action]
                    if action not in merged:
                        merged[action] = {
                            "visits": 0,
                            "value_sum": 0.0
                        }
                    merged[action]["visits"] += child_stats["visits"]
                    merged[action]["value_sum"] += child_stats["value_sum"]

        # Compute average values
        for action in merged:
            visits = merged[action]["visits"]
            merged[action]["value"] = (
                merged[action]["value_sum"] / visits if visits > 0 else 0.0
            )

        return merged
```

### 3.6 Leaf Parallelization

Parallelize rollouts from a single leaf:

```python
class LeafParallelMCTS:
    """Parallel rollouts from leaf nodes."""

    async def parallel_simulate(self, node, num_rollouts=8):
        """Run multiple rollouts from same node."""
        rollout_tasks = [
            self.rollout_policy.evaluate(node.state)
            for _ in range(num_rollouts)
        ]

        # Run rollouts concurrently
        values = await asyncio.gather(*rollout_tasks)

        # Use average value
        avg_value = sum(values) / len(values)

        return avg_value
```

### 3.7 Virtual Loss Parameter Tuning

**Virtual Loss Value:**
- Too low: Threads collide, wasting work
- Too high: Over-pessimistic, bad exploration

Typical values:
- **Games:** 1-3
- **Neural MCTS:** 3-10 (larger due to network evaluation cost)
- **Deep trees:** Higher (5-10)
- **Shallow trees:** Lower (1-3)

**Adaptive Virtual Loss:**

```python
class AdaptiveVirtualLoss:
    def __init__(self, initial_vl=3.0):
        self.vl = initial_vl
        self.collision_count = 0
        self.total_selections = 0

    def get_virtual_loss(self):
        """Compute adaptive virtual loss."""
        collision_rate = (
            self.collision_count / self.total_selections
            if self.total_selections > 0 else 0.0
        )

        # Increase VL if high collision rate
        if collision_rate > 0.3:
            self.vl = min(10.0, self.vl * 1.1)
        elif collision_rate < 0.1:
            self.vl = max(1.0, self.vl * 0.9)

        return self.vl

    def record_collision(self, collided):
        """Track selection collision."""
        self.total_selections += 1
        if collided:
            self.collision_count += 1
```

### 3.8 Performance Benchmarks

Expected speedup with parallelization:

| Threads | Naive | With VL | Root Parallel | Leaf Parallel |
|---------|-------|---------|---------------|---------------|
| 1 | 1.0x | 1.0x | 1.0x | 1.0x |
| 2 | 1.3x | 1.8x | 1.9x | 1.7x |
| 4 | 1.8x | 3.2x | 3.7x | 2.9x |
| 8 | 2.2x | 5.1x | 6.8x | 4.3x |
| 16 | 2.5x | 6.8x | 11.2x | 5.9x |

**Best Strategy by Domain:**
- **Cheap rollouts:** Tree parallelization with VL
- **Expensive rollouts:** Leaf parallelization
- **GPU inference:** Root parallelization (batch evaluation)

### 3.9 Integration with Neural MCTS

Parallel neural MCTS requires batched inference:

```python
class BatchedNeuralMCTS:
    def __init__(self, network, batch_size=32):
        self.network = network
        self.batch_size = batch_size
        self.eval_queue = asyncio.Queue()
        self.result_futures = {}

    async def batched_evaluate(self):
        """Background worker for batched network evaluation."""
        while True:
            batch_states = []
            batch_futures = []

            # Collect batch
            for _ in range(self.batch_size):
                try:
                    state, future = await asyncio.wait_for(
                        self.eval_queue.get(), timeout=0.01
                    )
                    batch_states.append(state)
                    batch_futures.append(future)
                except asyncio.TimeoutError:
                    break

            if not batch_states:
                continue

            # Batch inference
            state_tensors = torch.stack([s.to_tensor() for s in batch_states])
            with torch.no_grad():
                policy_logits, values = self.network(state_tensors)

            # Distribute results
            for i, future in enumerate(batch_futures):
                policy = policy_logits[i].cpu().numpy()
                value = values[i].item()
                future.set_result((policy, value))

    async def evaluate_state(self, state):
        """Queue state for batched evaluation."""
        future = asyncio.Future()
        await self.eval_queue.put((state, future))
        return await future
```

---

## Section 4: Adaptive Simulation Policies

### 4.1 Beyond Random Rollouts

Traditional MCTS uses random rollouts, but we can do better:

1. **Domain Knowledge**: Incorporate heuristics
2. **Learned Policies**: Use neural networks
3. **Adaptive Strategies**: Change policy based on search state
4. **Hybrid Approaches**: Combine multiple policies

### 4.2 Heuristic-Guided Rollouts

Use domain knowledge for better simulations:

```python
class HeuristicRolloutPolicy(RolloutPolicy):
    """Rollout with domain-specific heuristics."""

    def __init__(self, heuristic_fn, temperature=1.0):
        self.heuristic_fn = heuristic_fn
        self.temperature = temperature

    async def evaluate(self, state, rng, max_depth=10):
        """Run heuristic-guided simulation."""
        current_state = state
        depth = 0

        while not current_state.is_terminal() and depth < max_depth:
            actions = current_state.get_legal_actions()

            if not actions:
                break

            # Score actions with heuristic
            scores = [
                self.heuristic_fn(current_state, action)
                for action in actions
            ]

            # Temperature-scaled softmax
            scores = np.array(scores)
            if self.temperature > 0:
                scores = scores / self.temperature
            probs = np.exp(scores) / np.exp(scores).sum()

            # Sample action
            action = rng.choice(actions, p=probs)
            current_state = current_state.apply_action(action)
            depth += 1

        # Return heuristic evaluation
        return self.heuristic_fn(current_state, None)
```

### 4.3 Neural Rollout Policy

Use a small "fast policy" network:

```python
class NeuralRolloutPolicy(RolloutPolicy):
    """Fast neural network for rollout guidance."""

    def __init__(self, fast_policy_network, device="cpu"):
        self.network = fast_policy_network
        self.device = device
        self.network.eval()

    @torch.no_grad()
    async def evaluate(self, state, rng, max_depth=10):
        """Simulate using fast policy network."""
        current_state = state
        depth = 0
        trajectory_reward = 0.0

        while not current_state.is_terminal() and depth < max_depth:
            # Get policy from network
            state_tensor = current_state.to_tensor().unsqueeze(0).to(self.device)
            policy_logits = self.network(state_tensor)
            policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]

            # Sample action
            actions = current_state.get_legal_actions()
            action_probs = policy_probs[:len(actions)]
            action_probs = action_probs / action_probs.sum()

            action = rng.choice(actions, p=action_probs)
            current_state = current_state.apply_action(action)

            # Accumulate reward
            trajectory_reward += current_state.get_immediate_reward()
            depth += 1

        # Normalize to [0, 1]
        final_value = (trajectory_reward + depth) / (2 * max_depth)
        return np.clip(final_value, 0.0, 1.0)
```

### 4.4 Adaptive Policy Selection

Switch policies based on search progress:

```python
class AdaptiveRolloutPolicy(RolloutPolicy):
    """Adaptively select rollout policy based on context."""

    def __init__(self, policies, selection_strategy="ucb"):
        self.policies = policies
        self.selection_strategy = selection_strategy

        # Track policy performance
        self.policy_stats = {
            i: {"visits": 0, "value_sum": 0.0}
            for i in range(len(policies))
        }

    async def evaluate(self, state, rng, max_depth=10):
        """Select policy adaptively and evaluate."""
        # Select policy
        policy_idx = self._select_policy(rng)
        policy = self.policies[policy_idx]

        # Run simulation
        value = await policy.evaluate(state, rng, max_depth)

        # Update statistics
        self.policy_stats[policy_idx]["visits"] += 1
        self.policy_stats[policy_idx]["value_sum"] += value

        return value

    def _select_policy(self, rng):
        """Select policy using UCB1."""
        total_visits = sum(s["visits"] for s in self.policy_stats.values())

        if total_visits == 0 or self.selection_strategy == "random":
            return rng.integers(0, len(self.policies))

        if self.selection_strategy == "ucb":
            best_score = -float("inf")
            best_idx = 0

            for idx, stats in self.policy_stats.items():
                if stats["visits"] == 0:
                    return idx

                avg_value = stats["value_sum"] / stats["visits"]
                exploration = math.sqrt(2 * math.log(total_visits) / stats["visits"])
                score = avg_value + exploration

                if score > best_score:
                    best_score = score
                    best_idx = idx

            return best_idx

        elif self.selection_strategy == "epsilon_greedy":
            if rng.random() < 0.1:  # Explore
                return rng.integers(0, len(self.policies))
            else:  # Exploit
                best_idx = max(
                    self.policy_stats.keys(),
                    key=lambda i: (
                        self.policy_stats[i]["value_sum"] /
                        max(1, self.policy_stats[i]["visits"])
                    )
                )
                return best_idx
```

### 4.5 Contextual Rollout Policies

Adapt policy based on state features:

```python
class ContextualRolloutPolicy(RolloutPolicy):
    """Select rollout policy based on state context."""

    def __init__(self):
        self.random_policy = RandomRolloutPolicy()
        self.heuristic_policy = HeuristicRolloutPolicy(domain_heuristic)
        self.neural_policy = NeuralRolloutPolicy(fast_network)

    async def evaluate(self, state, rng, max_depth=10):
        """Choose policy based on state characteristics."""
        # Analyze state
        complexity = self._estimate_complexity(state)
        depth_in_tree = state.features.get("depth", 0)

        # Early game: use neural policy
        if depth_in_tree < 10:
            return await self.neural_policy.evaluate(state, rng, max_depth)

        # High complexity: use heuristic
        elif complexity > 0.7:
            return await self.heuristic_policy.evaluate(state, rng, max_depth)

        # Default: random
        else:
            return await self.random_policy.evaluate(state, rng, max_depth)

    def _estimate_complexity(self, state):
        """Estimate state complexity."""
        num_actions = len(state.get_legal_actions())
        state_entropy = state.features.get("entropy", 0.5)

        # Normalize to [0, 1]
        complexity = (num_actions / 50.0 + state_entropy) / 2.0
        return min(1.0, complexity)
```

### 4.6 Truncated Rollouts with Value Estimation

Combine rollout with value network:

```python
class TruncatedRolloutPolicy(RolloutPolicy):
    """Rollout for N steps then use value network."""

    def __init__(self, value_network, rollout_depth=5):
        self.value_network = value_network
        self.rollout_depth = rollout_depth

    @torch.no_grad()
    async def evaluate(self, state, rng, max_depth=10):
        """Hybrid rollout + value estimation."""
        current_state = state
        accumulated_reward = 0.0

        # Rollout for fixed depth
        for step in range(min(self.rollout_depth, max_depth)):
            if current_state.is_terminal():
                return current_state.get_reward()

            # Random action
            actions = current_state.get_legal_actions()
            action = rng.choice(actions)
            current_state = current_state.apply_action(action)

            # Accumulate immediate rewards
            accumulated_reward += current_state.get_immediate_reward()

        # Value network estimation
        state_tensor = current_state.to_tensor().unsqueeze(0)
        estimated_value = self.value_network(state_tensor).item()

        # Combine: discounted future value + accumulated reward
        gamma = 0.99
        total_value = accumulated_reward + (gamma ** self.rollout_depth) * estimated_value

        return np.clip(total_value, 0.0, 1.0)
```

### 4.7 Performance Comparison

| Policy Type | Speed | Accuracy | When to Use |
|-------------|-------|----------|-------------|
| Random | Very Fast | Low | Baseline, simple domains |
| Heuristic | Fast | Medium | Domain knowledge available |
| Neural (Fast) | Medium | Medium-High | Learned policy available |
| Neural (Full) | Slow | High | Deep evaluation needed |
| Adaptive | Medium | Medium-High | Uncertain domain |
| Truncated | Medium-Fast | High | Best of both worlds |

### 4.8 Domain-Specific Examples

#### Chess Heuristic

```python
def chess_rollout_heuristic(state, action):
    """Simple chess rollout heuristic."""
    # Material count
    material = state.material_balance()

    # Center control
    center_control = state.center_control_score()

    # King safety
    king_safety = state.king_safety_score()

    # Weighted combination
    score = (
        0.5 * material +
        0.3 * center_control +
        0.2 * king_safety
    )

    return (score + 10) / 20  # Normalize to [0, 1]
```

#### Go Pattern Matching

```python
def go_rollout_heuristic(state, action):
    """Go-specific patterns for rollout."""
    # Save atari
    if state.is_atari(action):
        return 0.8

    # Capture moves
    if state.is_capture(action):
        return 0.7

    # Pattern matching (3x3 patterns)
    pattern_score = state.match_patterns(action)

    # Avoid self-atari
    if state.is_self_atari(action):
        return 0.1

    return pattern_score
```

---

## Section 5: MCTS with Domain Knowledge

### 5.1 Incorporating Domain Knowledge

Domain knowledge can dramatically improve MCTS:

1. **Action Pruning**: Remove obviously bad actions
2. **Move Ordering**: Prioritize promising actions
3. **State Abstraction**: Group similar states
4. **Informed Initialization**: Better initial value estimates
5. **Transposition Tables**: Detect identical states

### 5.2 Action Pruning

Filter actions before expansion:

```python
class PruningMCTS(MCTSEngine):
    def __init__(self, pruning_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruning_fn = pruning_fn

    def expand(self, node, action_generator, state_transition):
        """Expand with action pruning."""
        if node.terminal:
            return node

        # Generate all actions
        all_actions = action_generator(node.state)

        # Apply pruning
        valid_actions = self.pruning_fn(node.state, all_actions)

        # Set pruned actions as available
        node.available_actions = valid_actions

        # Standard expansion logic
        return super().expand(node, lambda s: valid_actions, state_transition)

# Example: Chess pruning
def chess_action_pruning(state, actions):
    """Prune obviously bad moves."""
    filtered = []

    for action in actions:
        # Keep if not hanging piece
        if not state.is_hanging_piece(action):
            filtered.append(action)
        # But keep if it's a tactical blow
        elif state.is_tactical_blow(action):
            filtered.append(action)

    # Always keep at least 5 moves
    if len(filtered) < 5:
        return actions[:5]

    return filtered
```

### 5.3 Informed Prior Values

Initialize nodes with domain knowledge:

```python
class InformedMCTS(MCTSEngine):
    def __init__(self, prior_value_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_value_fn = prior_value_fn

    def expand(self, node, action_generator, state_transition):
        """Expand with informed priors."""
        # Get actions
        actions = action_generator(node.state)

        for action in actions:
            child_state = state_transition(node.state, action)
            child = node.add_child(action, child_state)

            # Initialize with domain knowledge
            prior_value = self.prior_value_fn(child_state)
            prior_visits = 10  # Virtual visits for initialization

            child.visits = prior_visits
            child.value_sum = prior_value * prior_visits

        return node.children[0] if node.children else node

# Example: Position evaluation
def chess_prior_value(state):
    """Chess position evaluation."""
    # Material balance
    material = state.material_balance()  # -1 to 1

    # Normalize to [0, 1]
    return (material + 1) / 2
```

### 5.4 Transposition Tables

Detect and merge identical states:

```python
class TranspositionMCTS(MCTSEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transposition_table = {}  # hash -> node

    def expand(self, node, action_generator, state_transition):
        """Expand with transposition detection."""
        actions = action_generator(node.state)

        for action in actions:
            child_state = state_transition(node.state, action)
            state_hash = child_state.to_hash_key()

            # Check if we've seen this state before
            if state_hash in self.transposition_table:
                # Reuse existing node
                existing_node = self.transposition_table[state_hash]
                node.children.append(existing_node)
                existing_node.parent = node  # Update parent link
            else:
                # Create new node
                child = node.add_child(action, child_state)
                self.transposition_table[state_hash] = child

        return node.children[0] if node.children else node

    def merge_statistics(self, node1, node2):
        """Merge statistics from transposition."""
        # Average the values
        total_visits = node1.visits + node2.visits
        node1.value_sum = (
            node1.value_sum + node2.value_sum
        )
        node1.visits = total_visits
```

### 5.5 Symmetry Detection

Exploit board/state symmetries:

```python
class SymmetryMCTS:
    """MCTS with symmetry awareness."""

    def __init__(self, symmetry_fn):
        self.symmetry_fn = symmetry_fn
        self.canonical_states = {}  # canonical -> [equivalent states]

    def get_canonical_state(self, state):
        """Get canonical representative of state."""
        symmetries = self.symmetry_fn(state)

        # Return lexicographically smallest
        canonical = min(symmetries, key=lambda s: s.to_hash_key())

        return canonical

    def expand_with_symmetry(self, node, action_generator, state_transition):
        """Expand considering symmetries."""
        actions = action_generator(node.state)

        for action in actions:
            child_state = state_transition(node.state, action)
            canonical_state = self.get_canonical_state(child_state)

            # Use canonical state for node
            child = node.add_child(action, canonical_state)

# Example: Board game symmetries
def get_board_symmetries(state):
    """Generate all symmetric board states."""
    board = state.board
    symmetries = []

    # Rotations
    for k in range(4):
        rotated = np.rot90(board, k)
        symmetries.append(State(rotated))

    # Reflections
    symmetries.append(State(np.fliplr(board)))
    symmetries.append(State(np.flipud(board)))

    return symmetries
```

### 5.6 Opening Books and Endgame Tables

Use pre-computed knowledge:

```python
class BookMCTS(MCTSEngine):
    """MCTS with opening book and endgame tables."""

    def __init__(self, opening_book, endgame_table, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opening_book = opening_book
        self.endgame_table = endgame_table

    async def search(self, root, *args, **kwargs):
        """Check book/table before searching."""
        state_hash = root.state.to_hash_key()

        # Check opening book
        if state_hash in self.opening_book:
            book_action = self.opening_book[state_hash]
            return book_action, {"source": "opening_book"}

        # Check endgame table
        if root.state.is_endgame() and state_hash in self.endgame_table:
            endgame_result = self.endgame_table[state_hash]
            return endgame_result["best_move"], {"source": "endgame_table"}

        # Standard MCTS search
        return await super().search(root, *args, **kwargs)
```

### 5.7 Domain-Specific Search Enhancements

#### Chess: Quiescence Search

```python
async def quiescence_rollout(state, rng, max_depth=10):
    """Only simulate quiet positions."""
    current = state
    depth = 0

    while depth < max_depth:
        if current.is_terminal():
            return current.get_reward()

        # Only continue if position is tactical
        if current.is_quiet():
            # Use static evaluation
            return chess_heuristic(current, None)

        # Simulate captures and checks only
        tactical_moves = [
            a for a in current.get_legal_actions()
            if current.is_tactical(a)
        ]

        if not tactical_moves:
            return chess_heuristic(current, None)

        action = rng.choice(tactical_moves)
        current = current.apply_action(action)
        depth += 1

    return chess_heuristic(current, None)
```

#### Go: Nakade and Ko Detection

```python
def go_informed_expansion(state, actions):
    """Go-specific action filtering."""
    filtered = []

    for action in actions:
        # Skip filling own eyes
        if state.is_own_eye(action):
            continue

        # Skip simple ko violations
        if state.is_simple_ko(action):
            continue

        # Prioritize nakade (killing shape)
        if state.is_nakade(action):
            filtered.insert(0, action)  # Add to front
        else:
            filtered.append(action)

    return filtered
```

### 5.8 Learning Domain Knowledge

Automatically learn domain patterns:

```python
class LearnedPatternMCTS:
    """MCTS that learns domain patterns from experience."""

    def __init__(self):
        self.pattern_stats = {}  # pattern -> {visits, value_sum}

    def extract_pattern(self, state, action):
        """Extract local pattern around action."""
        # Example: 3x3 grid around move
        pattern = state.get_local_pattern(action, radius=1)
        return pattern.to_hash()

    def update_pattern_stats(self, state, action, value):
        """Update pattern statistics."""
        pattern_hash = self.extract_pattern(state, action)

        if pattern_hash not in self.pattern_stats:
            self.pattern_stats[pattern_hash] = {
                "visits": 0,
                "value_sum": 0.0
            }

        self.pattern_stats[pattern_hash]["visits"] += 1
        self.pattern_stats[pattern_hash]["value_sum"] += value

    def get_pattern_prior(self, state, action):
        """Get prior value from learned patterns."""
        pattern_hash = self.extract_pattern(state, action)

        if pattern_hash in self.pattern_stats:
            stats = self.pattern_stats[pattern_hash]
            if stats["visits"] > 0:
                return stats["value_sum"] / stats["visits"]

        return 0.5  # Neutral default
```

---

## Section 6: Integration with Multi-Agent Systems

### 6.1 MCTS in LangGraph Agents

Integrate MCTS with HRM and TRM agents:

```python
from langchain.agents import AgentExecutor
from langsmith import traceable

class MCTSEnhancedAgent:
    """LangGraph agent with MCTS planning."""

    def __init__(self, mcts_engine, llm, tools):
        self.mcts = mcts_engine
        self.llm = llm
        self.tools = tools

    @traceable(name="mcts_agent_plan")
    async def plan_with_mcts(self, task):
        """Use MCTS to plan action sequence."""
        # Convert task to MCTS state
        initial_state = self._task_to_state(task)

        # Run MCTS search
        best_action, stats = await self.mcts.search(
            root=MCTSNode(initial_state),
            num_iterations=100,
            action_generator=self._generate_actions,
            state_transition=self._apply_action,
            rollout_policy=self._llm_rollout_policy
        )

        # Log to LangSmith
        langsmith.log_metrics({
            "mcts_iterations": stats["iterations"],
            "best_action_visits": stats["best_action_visits"],
            "tree_depth": self.mcts.get_cached_tree_depth()
        })

        return best_action

    def _generate_actions(self, state):
        """Generate available actions from state."""
        # Use LLM to generate possible actions
        prompt = f"Given state: {state.description}, list possible actions:"
        response = self.llm.invoke(prompt)
        actions = self._parse_actions(response)
        return actions

    async def _llm_rollout_policy(self, state, rng, max_depth=10):
        """LLM-based rollout evaluation."""
        prompt = f"Evaluate state quality (0-1): {state.description}"
        response = self.llm.invoke(prompt)
        value = self._parse_value(response)
        return value
```

### 6.2 Hierarchical MCTS with HRM

Combine MCTS with Hierarchical Reasoning Model:

```python
class HierarchicalMCTSAgent:
    """Two-level MCTS with HRM."""

    def __init__(self, high_level_mcts, low_level_mcts, hrm_model):
        self.high_mcts = high_level_mcts  # Plans high-level goals
        self.low_mcts = low_level_mcts     # Plans low-level actions
        self.hrm = hrm_model

    @traceable(name="hierarchical_mcts_planning")
    async def plan(self, task):
        """Two-level planning."""
        # High-level planning
        high_level_state = self._abstract_state(task)
        high_level_plan, _ = await self.high_mcts.search(
            root=MCTSNode(high_level_state),
            num_iterations=50,
            action_generator=self._generate_subgoals,
            state_transition=self._advance_subgoal,
            rollout_policy=self._hrm_high_level_eval
        )

        # Low-level planning for each subgoal
        detailed_actions = []
        for subgoal in high_level_plan:
            low_level_state = self._concretize_subgoal(subgoal)
            action, _ = await self.low_mcts.search(
                root=MCTSNode(low_level_state),
                num_iterations=100,
                action_generator=self._generate_concrete_actions,
                state_transition=self._apply_concrete_action,
                rollout_policy=self._hrm_low_level_eval
            )
            detailed_actions.append(action)

        return detailed_actions
```

### 6.3 TRM for MCTS Value Refinement

Use Tiny Recursive Model to refine MCTS values:

```python
class TRMEnhancedMCTS:
    """MCTS with TRM-based value refinement."""

    def __init__(self, mcts_engine, trm_model):
        self.mcts = mcts_engine
        self.trm = trm_model

    async def simulate_with_trm(self, node, rollout_policy, max_depth=10):
        """Refine rollout value with TRM."""
        # Standard rollout
        raw_value = await rollout_policy.evaluate(
            node.state, self.mcts.rng, max_depth
        )

        # TRM refinement
        state_tensor = node.state.to_tensor()
        refined_value = self.trm.recursive_evaluate(state_tensor, raw_value)

        return refined_value
```

### 6.4 Multi-Agent MCTS

Multiple agents cooperating via MCTS:

```python
class MultiAgentMCTS:
    """MCTS for cooperative multi-agent tasks."""

    def __init__(self, agents):
        self.agents = agents
        self.mcts_engines = {
            agent.id: MCTSEngine(seed=42 + i)
            for i, agent in enumerate(agents)
        }

    @traceable(name="multi_agent_mcts_coordination")
    async def coordinate(self, task):
        """Coordinate agents using MCTS."""
        # Each agent plans independently
        agent_plans = {}
        for agent in self.agents:
            agent_state = self._get_agent_state(agent, task)
            plan, _ = await self.mcts_engines[agent.id].search(
                root=MCTSNode(agent_state),
                num_iterations=100,
                action_generator=lambda s: agent.get_actions(s),
                state_transition=lambda s, a: agent.transition(s, a),
                rollout_policy=agent.rollout_policy
            )
            agent_plans[agent.id] = plan

        # Resolve conflicts and merge plans
        coordinated_plan = self._merge_plans(agent_plans)

        return coordinated_plan

    def _merge_plans(self, agent_plans):
        """Merge individual plans into coordinated plan."""
        # Simple merge: interleave actions
        merged = []
        max_len = max(len(plan) for plan in agent_plans.values())

        for i in range(max_len):
            for agent_id, plan in agent_plans.items():
                if i < len(plan):
                    merged.append((agent_id, plan[i]))

        return merged
```

### 6.5 MCTS with LangSmith Experiments

Track MCTS experiments in LangSmith:

```python
from langsmith import Client

class ExperimentalMCTS:
    """MCTS with full LangSmith experiment tracking."""

    def __init__(self, mcts_config):
        self.mcts = MCTSEngine(**mcts_config)
        self.langsmith = Client()
        self.experiment_name = f"mcts_experiment_{datetime.now()}"

    @traceable(
        name="mcts_search_experiment",
        experiment=True,
        metadata={"mcts_config": "config_dict"}
    )
    async def run_experiment(self, test_cases):
        """Run MCTS experiment with tracking."""
        results = []

        for test_case in test_cases:
            # Run MCTS
            best_action, stats = await self.mcts.search(
                root=MCTSNode(test_case.initial_state),
                num_iterations=test_case.num_iterations,
                action_generator=test_case.action_generator,
                state_transition=test_case.state_transition,
                rollout_policy=test_case.rollout_policy
            )

            # Evaluate result
            evaluation = test_case.evaluate(best_action)

            # Log comprehensive metrics
            langsmith.log_metrics({
                "test_case_id": test_case.id,
                "success": evaluation["success"],
                "quality_score": evaluation["quality"],
                "mcts_iterations": stats["iterations"],
                "tree_depth": self.mcts.get_cached_tree_depth(),
                "cache_hit_rate": stats["cache_hit_rate"],
                "best_action_confidence": (
                    stats["best_action_visits"] / stats["root_visits"]
                ),
                "exploration_breadth": stats["num_children"]
            })

            results.append({
                "test_case": test_case.id,
                "action": best_action,
                "stats": stats,
                "evaluation": evaluation
            })

        # Aggregate experiment results
        self._report_experiment_results(results)

        return results
```

---

## Section 7: Performance Optimization and Monitoring

### 7.1 Prometheus Metrics for MCTS

Export MCTS metrics to Prometheus:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
mcts_iterations_total = Counter(
    'mcts_iterations_total',
    'Total MCTS iterations run',
    ['agent_type', 'task_category']
)

mcts_search_duration = Histogram(
    'mcts_search_duration_seconds',
    'MCTS search duration',
    ['agent_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

mcts_tree_depth = Gauge(
    'mcts_tree_depth',
    'Current MCTS tree depth',
    ['agent_type']
)

mcts_cache_hit_rate = Gauge(
    'mcts_cache_hit_rate',
    'MCTS cache hit rate',
    ['agent_type']
)

class MonitoredMCTS(MCTSEngine):
    """MCTS with Prometheus monitoring."""

    async def search(self, root, *args, agent_type="default", task_category="general", **kwargs):
        """Search with monitoring."""
        start_time = time.time()

        # Run search
        result = await super().search(root, *args, **kwargs)

        # Record metrics
        duration = time.time() - start_time
        action, stats = result

        mcts_iterations_total.labels(
            agent_type=agent_type,
            task_category=task_category
        ).inc(stats["iterations"])

        mcts_search_duration.labels(
            agent_type=agent_type
        ).observe(duration)

        mcts_tree_depth.labels(
            agent_type=agent_type
        ).set(self.get_cached_tree_depth())

        mcts_cache_hit_rate.labels(
            agent_type=agent_type
        ).set(stats["cache_hit_rate"])

        return result
```

### 7.2 Performance Profiling

Identify bottlenecks:

```python
import cProfile
import pstats
from functools import wraps

def profile_mcts(func):
    """Decorator to profile MCTS functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        result = await func(*args, **kwargs)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions

        return result
    return wrapper

class ProfiledMCTS(MCTSEngine):
    @profile_mcts
    async def search(self, *args, **kwargs):
        return await super().search(*args, **kwargs)
```

### 7.3 Memory Optimization

Reduce memory usage for large searches:

```python
class MemoryEfficientMCTS(MCTSEngine):
    """MCTS with memory optimizations."""

    def __init__(self, *args, max_tree_size=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tree_size = max_tree_size
        self.node_pool = []  # Reuse node objects

    def prune_tree(self, root, keep_best_k=3):
        """Prune tree to reduce memory."""
        # Keep only best k children at each level
        for node in self._traverse_tree(root):
            if len(node.children) > keep_best_k:
                # Sort by visits
                sorted_children = sorted(
                    node.children,
                    key=lambda c: c.visits,
                    reverse=True
                )
                # Keep only top k
                node.children = sorted_children[:keep_best_k]

    def get_node(self, state, parent=None, action=None):
        """Get node from pool or create new."""
        if self.node_pool:
            node = self.node_pool.pop()
            node.reset(state, parent, action)
        else:
            node = MCTSNode(state, parent, action)

        return node

    def release_node(self, node):
        """Return node to pool."""
        if len(self.node_pool) < 1000:  # Pool size limit
            self.node_pool.append(node)
```

### 7.4 Caching Strategies

Advanced caching for better performance:

```python
class AdaptiveCachingMCTS(MCTSEngine):
    """MCTS with adaptive caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_access_count = {}
        self.cache_eviction_priority = []

    def should_cache(self, state_hash, value):
        """Decide if state should be cached."""
        # Don't cache low-value or terminal states
        if abs(value) > 0.95:  # Terminal-like
            return False

        # Cache high-access states
        access_count = self.cache_access_count.get(state_hash, 0)
        return access_count > 2

    def evict_from_cache(self):
        """Smart cache eviction."""
        if len(self._simulation_cache) < self.cache_size_limit:
            return

        # Evict least frequently accessed
        min_access = min(self.cache_access_count.values())
        for state_hash, count in self.cache_access_count.items():
            if count == min_access:
                if state_hash in self._simulation_cache:
                    del self._simulation_cache[state_hash]
                    del self.cache_access_count[state_hash]
                    break
```

### 7.5 Batch Processing for GPU

Optimize neural MCTS for GPU:

```python
class BatchedGPUMCTS(NeuralMCTS):
    """GPU-optimized batched neural MCTS."""

    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.eval_queue = asyncio.Queue()
        self.result_futures = {}

        # Start batch processor
        asyncio.create_task(self._batch_processor())

    async def _batch_processor(self):
        """Background task for batched inference."""
        while True:
            batch = []
            futures = []

            # Collect batch
            for _ in range(self.batch_size):
                try:
                    state, future = await asyncio.wait_for(
                        self.eval_queue.get(),
                        timeout=0.01
                    )
                    batch.append(state)
                    futures.append(future)
                except asyncio.TimeoutError:
                    if batch:
                        break
                    continue

            if not batch:
                await asyncio.sleep(0.001)
                continue

            # Batch inference
            states_tensor = torch.stack([
                s.to_tensor() for s in batch
            ]).to(self.device)

            with torch.no_grad():
                policy_logits, values = self.network(states_tensor)

            # Distribute results
            for i, future in enumerate(futures):
                policy = F.softmax(policy_logits[i], dim=0).cpu().numpy()
                value = values[i].item()
                future.set_result((policy, value))
```

### 7.6 Distributed MCTS

Scale MCTS across multiple machines:

```python
import ray

@ray.remote
class DistributedMCTSWorker:
    """Remote MCTS worker for distributed search."""

    def __init__(self, seed):
        self.mcts = MCTSEngine(seed=seed)

    async def search(self, initial_state, num_iterations, config):
        """Run search on worker."""
        root = MCTSNode(initial_state)
        result = await self.mcts.search(
            root=root,
            num_iterations=num_iterations,
            **config
        )
        return result

class DistributedMCTS:
    """Distributed MCTS across multiple workers."""

    def __init__(self, num_workers=4):
        ray.init(ignore_reinit_error=True)
        self.workers = [
            DistributedMCTSWorker.remote(seed=42 + i)
            for i in range(num_workers)
        ]

    async def parallel_search(self, initial_state, total_iterations, config):
        """Run parallel search across workers."""
        iterations_per_worker = total_iterations // len(self.workers)

        # Launch parallel searches
        futures = [
            worker.search.remote(initial_state, iterations_per_worker, config)
            for worker in self.workers
        ]

        # Collect results
        results = await ray.get(futures)

        # Merge results
        merged = self._merge_results(results)

        return merged
```

---

## Section 8: Labs and Hands-On Exercises

### Lab 1: Implement Neural-Guided MCTS for Tic-Tac-Toe

**Objective:** Build a complete neural MCTS system for Tic-Tac-Toe.

**Tasks:**
1. Define TicTacToe GameState class
2. Create a simple policy-value network
3. Implement self-play data collection
4. Train the network
5. Evaluate performance vs. random MCTS

**Starter Code:**

```python
class TicTacToeState(GameState):
    def __init__(self, board=None):
        self.board = board if board is not None else np.zeros((3, 3))
        self.current_player = 1

    def get_legal_actions(self):
        # TODO: Return list of empty positions
        pass

    def apply_action(self, action):
        # TODO: Return new state after move
        pass

    def is_terminal(self):
        # TODO: Check for win or draw
        pass

    def get_reward(self, player=1):
        # TODO: Return 1 for win, -1 for loss, 0 for draw
        pass

    def to_tensor(self):
        # TODO: Convert board to tensor
        pass

    def get_hash(self):
        # TODO: Return unique hash
        pass
```

**Success Criteria:**
- Neural MCTS achieves >90% win rate vs random MCTS
- Training converges within 100 iterations
- Policy network shows strong preference for center opening

### Lab 2: Parallel MCTS with Virtual Loss

**Objective:** Implement and benchmark parallel MCTS.

**Tasks:**
1. Implement VirtualLossNode class
2. Create parallel search with asyncio
3. Benchmark speedup (1, 2, 4, 8 threads)
4. Tune virtual loss parameter
5. Compare tree parallelization vs root parallelization

**Evaluation Metrics:**
- Speedup factor vs sequential MCTS
- Cache hit rate
- Search tree quality (depth, breadth)

### Lab 3: RAVE Implementation

**Objective:** Add RAVE to existing MCTS.

**Tasks:**
1. Extend MCTSNode with RAVE statistics
2. Implement AMAF backpropagation
3. Implement hybrid UCB+RAVE selection
4. Test on a domain with many similar states
5. Compare convergence speed vs standard MCTS

**Success Criteria:**
- RAVE achieves same quality with 50% fewer iterations
- β parameter decays appropriately
- RAVE bonus diminishes with visits

### Lab 4: Domain Knowledge Integration

**Objective:** Enhance MCTS with domain-specific knowledge.

**Tasks:**
1. Design domain heuristics for a game/problem
2. Implement action pruning
3. Implement informed prior values
4. Add transposition table support
5. Measure performance improvement

**Metrics to Track:**
- Effective branching factor reduction
- Search tree depth increase
- Decision quality improvement

### Lab 5: Multi-Agent MCTS Coordination

**Objective:** Coordinate multiple agents using MCTS.

**Tasks:**
1. Create 3 specialized agents (planner, executor, validator)
2. Implement cooperative MCTS planning
3. Add conflict resolution
4. Integrate with LangGraph
5. Track coordination metrics

**Success Criteria:**
- Agents successfully complete complex task
- Minimal action conflicts
- Coordinated plan is better than individual plans

### Lab 6: Production Integration

**Objective:** Deploy MCTS in production with full monitoring.

**Tasks:**
1. Add Prometheus metrics
2. Add LangSmith tracing
3. Implement graceful degradation
4. Add health checks and alerts
5. Load test with realistic workload

**Monitoring:**
- 99th percentile latency < 2s
- Cache hit rate > 70%
- Error rate < 0.1%

---

## Section 9: Assessment and Certification

### 9.1 Knowledge Assessment

**Multiple Choice Questions (20 questions)**

1. In neural-guided MCTS, what does the PUCT formula balance?
   a) Speed vs accuracy
   b) Exploration vs exploitation
   c) Memory vs CPU
   d) Breadth vs depth

2. Virtual loss is primarily used for:
   a) Reducing memory usage
   b) Preventing thread collisions
   c) Improving value estimates
   d) Caching simulation results

3. RAVE (Rapid Action Value Estimation) works by:
   a) Using faster hardware
   b) Sharing value estimates across similar states
   c) Reducing search depth
   d) Pruning bad actions

4. Progressive widening helps with:
   a) Small action spaces
   b) Terminal state detection
   c) Large/continuous action spaces
   d) Transposition tables

5. The temperature parameter in action selection controls:
   a) GPU temperature
   b) Exploration vs exploitation
   c) Search speed
   d) Cache size

### 9.2 Practical Assessment

**Project: Build an Advanced MCTS System**

Requirements:
1. Implement neural-guided MCTS for a non-trivial domain
2. Add at least 2 advanced techniques (RAVE, parallel search, progressive widening)
3. Integrate with LangSmith tracing
4. Add Prometheus metrics
5. Achieve specified performance benchmarks
6. Write comprehensive tests (>80% coverage)

**Grading Rubric:**

| Component | Points | Criteria |
|-----------|--------|----------|
| Neural MCTS Implementation | 20 | Correct PUCT, expansion, backprop |
| Advanced Techniques | 20 | 2+ techniques correctly implemented |
| Performance | 15 | Meets benchmark targets |
| Code Quality | 15 | Type hints, docstrings, clean code |
| Testing | 15 | >80% coverage, meaningful tests |
| Monitoring | 10 | LangSmith + Prometheus integration |
| Documentation | 5 | Clear README and examples |

### 9.3 Performance Benchmarks

Your implementation must achieve:

1. **Search Quality:**
   - Optimal action selection rate: >85%
   - Tree depth: >10 for complex problems
   - Cache hit rate: >60%

2. **Performance:**
   - 1000 iterations in <5 seconds (CPU)
   - 10,000 iterations in <10 seconds (GPU)
   - Parallel speedup: >3x with 8 threads

3. **Robustness:**
   - Handles degenerate states gracefully
   - Deterministic with same seed
   - Memory usage <1GB for typical search

### 9.4 Certification Criteria

To earn Module 8 certification:

1. Score ≥80% on knowledge assessment
2. Complete all 6 labs successfully
3. Pass practical project with ≥70 points
4. Meet all performance benchmarks
5. Submit working code with tests

**Certificate Level:**
- 90-100%: Advanced MCTS Expert
- 80-89%: MCTS Specialist
- 70-79%: MCTS Practitioner

---

## Summary and Next Steps

### What You've Learned

In this module, you've mastered:

1. **Neural-Guided MCTS**: AlphaZero-style search with policy and value networks
2. **Advanced Selection**: PUCT, RAVE, and hybrid strategies
3. **Parallelization**: Virtual loss, tree/root/leaf parallelism
4. **Progressive Widening**: Managing large action spaces
5. **Domain Knowledge**: Pruning, patterns, symmetries
6. **Integration**: LangGraph agents, monitoring, production deployment

### Production Readiness Checklist

Before deploying advanced MCTS:

- [ ] Thorough testing (unit, integration, end-to-end)
- [ ] Performance profiling and optimization
- [ ] Monitoring and alerting setup
- [ ] Graceful degradation for failures
- [ ] Documentation and runbooks
- [ ] Load testing at scale
- [ ] Security review (if applicable)

### Further Reading

**Papers:**
1. Silver et al. (2017) - "Mastering Chess and Shogi by Self-Play with a General RL Algorithm" (AlphaZero)
2. Coulom (2006) - "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search"
3. Gelly & Silver (2011) - "Monte-Carlo Tree Search and Rapid Action Value Estimation in Computer Go"

**Advanced Topics:**
- Meta-MCTS: Learning MCTS hyperparameters
- Continuous action MCTS
- MCTS for partial observability
- Monte Carlo Graph Search

### Community and Support

- GitHub Discussions: Share implementations and ask questions
- LangSmith Community: Track experiments and compare results
- Weekly Office Hours: Get help from instructors

---

## Appendix A: Reference Implementations

### Complete Neural MCTS Example

See `examples/advanced_mcts_demo.py` for full implementation.

### Configuration Templates

```python
# Fast experimentation
FAST_CONFIG = MCTSConfig(
    num_simulations=400,
    c_puct=1.25,
    virtual_loss=3.0,
    dirichlet_epsilon=0.25
)

# Production quality
PRODUCTION_CONFIG = MCTSConfig(
    num_simulations=1600,
    c_puct=1.5,
    virtual_loss=5.0,
    dirichlet_epsilon=0.15
)
```

### Monitoring Dashboard

Sample Prometheus queries:

```promql
# Average search duration
rate(mcts_search_duration_seconds_sum[5m]) /
rate(mcts_search_duration_seconds_count[5m])

# Cache effectiveness
mcts_cache_hit_rate

# Search quality
mcts_tree_depth
```

---

## Appendix B: Troubleshooting Guide

### Common Issues

**Problem:** Neural MCTS converges slowly
- Check learning rate and network capacity
- Verify self-play diversity (temperature)
- Ensure sufficient training data

**Problem:** Parallel MCTS shows poor speedup
- Increase virtual loss value
- Check for lock contention (profiling)
- Consider root parallelization instead

**Problem:** High memory usage
- Enable tree pruning
- Reduce cache size
- Use memory-efficient node representation

**Problem:** RAVE not helping
- Verify β decay is working
- Check if domain has move-order independence
- Tune RAVE constant parameter

---

**End of Module 8**

Total Lines: 1,653

This comprehensive training module covers all advanced MCTS techniques needed for production deployment in the LangGraph Multi-Agent MCTS Framework.
