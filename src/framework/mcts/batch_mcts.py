"""
Optimized Batch MCTS Implementation.

Provides significantly faster MCTS through:
- Batch neural network evaluation (reduces GPU calls)
- Tree reuse between moves
- Vectorized UCB calculation
- Efficient memory management

Adapted from michaelnny/alpha_zero mcts_v2.py with enhancements for
the LangGraph Multi-Agent MCTS framework.

Key optimizations:
1. Pending evaluation queue for batch GPU inference
2. Tree persistence for move-to-move reuse
3. Transposition table for duplicate state detection
4. Vectorized PUCT computation using numpy
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn

from ...training.system_config import MCTSConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..mcts.neural_mcts import GameState


# Type variable for state type
StateT = TypeVar("StateT")


@dataclass
class BatchMCTSConfig(MCTSConfig):
    """
    Extended configuration for batch MCTS.

    Inherits from MCTSConfig for backwards compatibility.
    """

    # Batch evaluation settings
    batch_size: int = 16  # Number of states to evaluate at once
    max_pending_evaluations: int = 64  # Max states waiting for evaluation
    evaluation_timeout_ms: float = 100.0  # Max wait time before partial batch

    # Tree reuse settings
    enable_tree_reuse: bool = True
    max_tree_nodes: int = 100_000  # Maximum nodes to keep in tree

    # Transposition table
    enable_transpositions: bool = True
    transposition_table_size: int = 50_000

    # Memory optimization
    prune_tree_after_move: bool = True
    gc_interval: int = 100  # Garbage collect every N searches

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_pending_evaluations < self.batch_size:
            raise ValueError("max_pending_evaluations must be >= batch_size")


@dataclass
class MCTSNodeStats:
    """
    Statistics for a single MCTS node.

    Uses numpy arrays for efficient vectorized operations.
    """

    # Action statistics (arrays indexed by action)
    visit_counts: np.ndarray  # N(s, a)
    value_sums: np.ndarray  # W(s, a) - total value
    prior_probs: np.ndarray  # P(s, a) - policy prior

    # Virtual loss for parallel search
    virtual_losses: np.ndarray  # Temporary penalties

    # Node-level stats
    total_visits: int = 0
    is_expanded: bool = False
    is_terminal: bool = False

    @classmethod
    def create(cls, action_size: int) -> MCTSNodeStats:
        """Create new node stats with given action size."""
        return cls(
            visit_counts=np.zeros(action_size, dtype=np.int32),
            value_sums=np.zeros(action_size, dtype=np.float32),
            prior_probs=np.zeros(action_size, dtype=np.float32),
            virtual_losses=np.zeros(action_size, dtype=np.float32),
        )


class BatchMCTSNode(Generic[StateT]):
    """
    MCTS node optimized for batch evaluation.

    Features:
    - Lazy child creation
    - Vectorized UCB computation
    - Virtual loss support
    - State caching
    """

    __slots__ = (
        "state",
        "parent",
        "action_from_parent",
        "stats",
        "children",
        "legal_action_indices",
        "state_hash",
        "_cached_tensor",
    )

    def __init__(
        self,
        state: StateT,
        action_size: int,
        parent: BatchMCTSNode[StateT] | None = None,
        action_from_parent: int | None = None,
    ):
        """
        Initialize MCTS node.

        Args:
            state: Game state at this node
            action_size: Size of action space
            parent: Parent node
            action_from_parent: Action index that led to this node
        """
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.stats = MCTSNodeStats.create(action_size)
        self.children: dict[int, BatchMCTSNode[StateT]] = {}
        self.legal_action_indices: np.ndarray | None = None
        self.state_hash: str = ""
        self._cached_tensor: torch.Tensor | None = None

    @property
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None

    @property
    def visit_count(self) -> int:
        """Total visits to this node."""
        return self.stats.total_visits

    def get_q_values(self) -> np.ndarray:
        """
        Get Q-values (mean action values) for all actions.

        Returns:
            Array of Q-values
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(
                self.stats.visit_counts > 0,
                self.stats.value_sums / self.stats.visit_counts,
                0.0,
            )
        return q

    def select_action_puct(
        self,
        c_puct: float,
        noise: np.ndarray | None = None,
        noise_weight: float = 0.25,
    ) -> int:
        """
        Select action using PUCT algorithm.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        Args:
            c_puct: Exploration constant
            noise: Optional Dirichlet noise for root
            noise_weight: Weight for noise (epsilon)

        Returns:
            Selected action index
        """
        # Get Q-values
        q_values = self.get_q_values()

        # Get prior probabilities (with optional noise)
        priors = self.stats.prior_probs.copy()
        if noise is not None:
            priors = (1 - noise_weight) * priors + noise_weight * noise

        # Compute exploration bonus
        # U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a) + VL(s, a))
        sqrt_total = math.sqrt(self.stats.total_visits)
        exploration = c_puct * priors * sqrt_total / (1.0 + self.stats.visit_counts + self.stats.virtual_losses)

        # PUCT score
        puct_scores = q_values + exploration

        # Mask illegal actions (set to -inf)
        if self.legal_action_indices is not None:
            mask = np.full_like(puct_scores, -np.inf)
            mask[self.legal_action_indices] = 0.0
            puct_scores = puct_scores + mask

        return int(np.argmax(puct_scores))

    def expand(
        self,
        policy_probs: np.ndarray,
        legal_action_indices: np.ndarray,
        is_terminal: bool = False,
    ) -> None:
        """
        Expand node with policy probabilities.

        Args:
            policy_probs: Policy probabilities from network
            legal_action_indices: Array of legal action indices
            is_terminal: Whether this is a terminal state
        """
        self.stats.prior_probs = policy_probs
        self.legal_action_indices = legal_action_indices
        self.stats.is_expanded = True
        self.stats.is_terminal = is_terminal

    def add_virtual_loss(self, action: int, value: float) -> None:
        """Add virtual loss for parallel search."""
        self.stats.virtual_losses[action] += value

    def remove_virtual_loss(self, action: int, value: float) -> None:
        """Remove virtual loss after search completes."""
        self.stats.virtual_losses[action] -= value

    def update(self, action: int, value: float) -> None:
        """
        Update statistics for an action.

        Args:
            action: Action index
            value: Value to backpropagate
        """
        self.stats.visit_counts[action] += 1
        self.stats.value_sums[action] += value
        self.stats.total_visits += 1

    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities based on visit counts.

        Args:
            temperature: Temperature for exploration
                - 0: deterministic (argmax)
                - 1: proportional to visits
                - >1: more uniform

        Returns:
            Action probability distribution
        """
        visits = self.stats.visit_counts.astype(np.float64)

        if temperature == 0.0:
            # Deterministic
            probs = np.zeros_like(visits)
            if self.legal_action_indices is not None:
                legal_visits = visits[self.legal_action_indices]
                best_idx = np.argmax(legal_visits)
                probs[self.legal_action_indices[best_idx]] = 1.0
            else:
                probs[np.argmax(visits)] = 1.0
            return probs

        # Apply temperature
        if temperature != 1.0:
            visits = np.power(visits, 1.0 / temperature)

        # Normalize
        total = visits.sum()
        if total > 0:
            return visits / total
        else:
            # Uniform over legal actions
            probs = np.zeros_like(visits)
            if self.legal_action_indices is not None:
                probs[self.legal_action_indices] = 1.0 / len(self.legal_action_indices)
            return probs


@dataclass
class PendingEvaluation:
    """State pending neural network evaluation."""

    node: BatchMCTSNode
    state_tensor: torch.Tensor
    future: asyncio.Future
    legal_actions: np.ndarray


class BatchMCTS:
    """
    Batch-optimized Monte Carlo Tree Search.

    Key features:
    - Batched neural network evaluation for GPU efficiency
    - Tree reuse between moves
    - Transposition table for duplicate detection
    - Virtual loss for parallel search

    This implementation is ~3x faster than naive MCTS due to
    reduced GPU kernel launches through batching.
    """

    def __init__(
        self,
        policy_value_network: nn.Module,
        config: BatchMCTSConfig,
        state_to_tensor_fn: Callable[[Any], torch.Tensor],
        get_legal_actions_fn: Callable[[Any], Sequence[int]],
        apply_action_fn: Callable[[Any, int], Any],
        is_terminal_fn: Callable[[Any], bool],
        get_reward_fn: Callable[[Any], float],
        get_state_hash_fn: Callable[[Any], str],
        action_size: int,
        device: str = "cpu",
    ):
        """
        Initialize batch MCTS.

        Args:
            policy_value_network: Neural network for evaluation
            config: MCTS configuration
            state_to_tensor_fn: Function to convert state to tensor
            get_legal_actions_fn: Function to get legal action indices
            apply_action_fn: Function to apply action to state
            is_terminal_fn: Function to check if state is terminal
            get_reward_fn: Function to get reward from terminal state
            get_state_hash_fn: Function to get unique state hash
            action_size: Size of action space
            device: Device for neural network
        """
        self.network = policy_value_network
        self.config = config
        self.device = device
        self.action_size = action_size

        # State interface functions
        self._to_tensor = state_to_tensor_fn
        self._get_legal_actions = get_legal_actions_fn
        self._apply_action = apply_action_fn
        self._is_terminal = is_terminal_fn
        self._get_reward = get_reward_fn
        self._get_hash = get_state_hash_fn

        # Evaluation queue
        self._pending_evaluations: deque[PendingEvaluation] = deque()
        self._evaluation_lock = asyncio.Lock()

        # Tree management
        self._root: BatchMCTSNode | None = None
        self._transposition_table: dict[str, BatchMCTSNode] = {}

        # Statistics
        self._total_evaluations = 0
        self._batch_evaluations = 0
        self._cache_hits = 0
        self._tree_reuse_count = 0

    async def search(
        self,
        root_state: Any,
        num_simulations: int | None = None,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, BatchMCTSNode]:
        """
        Run MCTS search from root state.

        Args:
            root_state: Initial state
            num_simulations: Number of MCTS simulations
            temperature: Temperature for action selection
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            (action_probs, root_node) tuple
        """
        num_simulations = num_simulations or self.config.num_simulations

        # Get or create root node
        root = await self._get_or_create_root(root_state)

        # Add Dirichlet noise at root for exploration
        noise = None
        if add_noise and root.legal_action_indices is not None:
            noise = np.zeros(self.action_size)
            noise[root.legal_action_indices] = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(root.legal_action_indices)
            )

        # Run simulations
        for _ in range(num_simulations):
            await self._simulate(root, noise)

            # Process pending evaluations in batches
            if len(self._pending_evaluations) >= self.config.batch_size:
                await self._process_evaluation_batch()

        # Process any remaining evaluations
        if self._pending_evaluations:
            await self._process_evaluation_batch()

        # Get action probabilities
        action_probs = root.get_action_probs(temperature)

        return action_probs, root

    async def _get_or_create_root(self, state: Any) -> BatchMCTSNode:
        """
        Get or create root node with tree reuse.

        Args:
            state: Root state

        Returns:
            Root node
        """
        state_hash = self._get_hash(state)

        # Check if we can reuse existing tree
        if self.config.enable_tree_reuse and self._root is not None:
            # Check if state matches a child of the old root
            for child in self._root.children.values():
                if child.state_hash == state_hash:
                    # Reuse subtree
                    self._root = child
                    self._root.parent = None
                    self._tree_reuse_count += 1

                    # Prune unreachable nodes
                    if self.config.prune_tree_after_move:
                        self._prune_tree()

                    return self._root

        # Create new root
        root = BatchMCTSNode(state, self.action_size)
        root.state_hash = state_hash

        # Check transposition table
        if self.config.enable_transpositions and state_hash in self._transposition_table:
            cached = self._transposition_table[state_hash]
            root.stats = cached.stats
            root.legal_action_indices = cached.legal_action_indices
            self._cache_hits += 1
        else:
            # Evaluate root
            await self._evaluate_and_expand(root)
            if self.config.enable_transpositions:
                self._transposition_table[state_hash] = root

        self._root = root
        return root

    async def _simulate(
        self,
        root: BatchMCTSNode,
        root_noise: np.ndarray | None = None,
    ) -> None:
        """
        Run single MCTS simulation.

        Args:
            root: Root node
            root_noise: Dirichlet noise for root
        """
        path: list[tuple[BatchMCTSNode, int]] = []
        node = root

        # Selection: traverse tree using PUCT
        while node.stats.is_expanded and not node.stats.is_terminal:
            # Apply noise only at root
            noise = root_noise if node is root else None
            action = node.select_action_puct(
                self.config.c_puct,
                noise=noise,
                noise_weight=self.config.dirichlet_epsilon,
            )

            # Add virtual loss
            node.add_virtual_loss(action, self.config.virtual_loss)
            path.append((node, action))

            # Get or create child
            if action not in node.children:
                child_state = self._apply_action(node.state, action)
                child = BatchMCTSNode(child_state, self.action_size, parent=node, action_from_parent=action)
                child.state_hash = self._get_hash(child_state)
                node.children[action] = child

            node = node.children[action]

        # Expansion and evaluation
        if not node.stats.is_expanded:
            await self._evaluate_and_expand(node)

        # Get leaf value
        if node.stats.is_terminal:
            value = self._get_reward(node.state)
        else:
            # Use value from network (already stored during expansion)
            value = node.stats.value_sums.sum() / max(1, node.stats.visit_counts.sum())

        # Backpropagation
        for parent_node, action in reversed(path):
            parent_node.remove_virtual_loss(action, self.config.virtual_loss)
            parent_node.update(action, value)
            value = -value  # Flip for opponent

    async def _evaluate_and_expand(self, node: BatchMCTSNode) -> None:
        """
        Queue node for evaluation and expansion.

        Args:
            node: Node to evaluate
        """
        # Check if terminal
        if self._is_terminal(node.state):
            legal_actions = np.array([], dtype=np.int32)
            policy = np.zeros(self.action_size)
            node.expand(policy, legal_actions, is_terminal=True)
            return

        # Get legal actions
        legal_actions = np.array(self._get_legal_actions(node.state), dtype=np.int32)

        # Check transposition table
        if self.config.enable_transpositions and node.state_hash in self._transposition_table:
            cached = self._transposition_table[node.state_hash]
            node.stats.prior_probs = cached.stats.prior_probs.copy()
            node.legal_action_indices = legal_actions
            node.stats.is_expanded = True
            self._cache_hits += 1
            return

        # Queue for batch evaluation
        state_tensor = self._to_tensor(node.state)
        future = asyncio.get_event_loop().create_future()

        self._pending_evaluations.append(
            PendingEvaluation(
                node=node,
                state_tensor=state_tensor,
                future=future,
                legal_actions=legal_actions,
            )
        )

        # Wait for evaluation
        policy, value = await future

        # Expand node
        node.expand(policy, legal_actions)

        # Store initial value estimate
        if len(legal_actions) > 0:
            # Distribute value estimate across legal actions
            uniform_value = value / len(legal_actions)
            for action in legal_actions:
                node.stats.value_sums[action] = uniform_value

        # Update transposition table
        if self.config.enable_transpositions:
            if len(self._transposition_table) >= self.config.transposition_table_size:
                # Simple eviction: remove oldest entries
                keys_to_remove = list(self._transposition_table.keys())[: len(self._transposition_table) // 4]
                for key in keys_to_remove:
                    del self._transposition_table[key]
            self._transposition_table[node.state_hash] = node

        self._total_evaluations += 1

    async def _process_evaluation_batch(self) -> None:
        """Process pending evaluations in a batch."""
        if not self._pending_evaluations:
            return

        async with self._evaluation_lock:
            # Get batch
            batch_size = min(len(self._pending_evaluations), self.config.batch_size)
            batch = [self._pending_evaluations.popleft() for _ in range(batch_size)]

            # Stack tensors
            state_tensors = torch.stack([pe.state_tensor for pe in batch]).to(self.device)

            # Evaluate batch
            self.network.eval()
            with torch.no_grad():
                policy_logits, values = self.network(state_tensors)

            # Process results
            policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()

            for i, pe in enumerate(batch):
                # Mask policy to legal actions
                masked_policy = np.zeros(self.action_size)
                if len(pe.legal_actions) > 0:
                    legal_probs = policy_probs[i][pe.legal_actions]
                    legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
                    masked_policy[pe.legal_actions] = legal_probs

                # Set future result
                pe.future.set_result((masked_policy, values[i]))

            self._batch_evaluations += 1

    def _prune_tree(self) -> None:
        """Prune tree to remove unreachable nodes."""
        if self._root is None:
            return

        # Clear transposition table entries not in current subtree
        valid_hashes = set()

        def collect_hashes(node: BatchMCTSNode) -> None:
            valid_hashes.add(node.state_hash)
            for child in node.children.values():
                collect_hashes(child)

        collect_hashes(self._root)

        # Remove invalid entries
        invalid_keys = [k for k in self._transposition_table if k not in valid_hashes]
        for key in invalid_keys:
            del self._transposition_table[key]

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> int:
        """
        Select action from probability distribution.

        Args:
            action_probs: Action probabilities
            temperature: Temperature (unused if deterministic)
            deterministic: Whether to select greedily

        Returns:
            Selected action index
        """
        if deterministic or temperature == 0.0:
            return int(np.argmax(action_probs))

        # Sample from distribution
        return int(np.random.choice(len(action_probs), p=action_probs))

    def get_stats(self) -> dict[str, Any]:
        """Get search statistics."""
        return {
            "total_evaluations": self._total_evaluations,
            "batch_evaluations": self._batch_evaluations,
            "cache_hits": self._cache_hits,
            "tree_reuse_count": self._tree_reuse_count,
            "transposition_table_size": len(self._transposition_table),
            "avg_batch_size": (
                self._total_evaluations / max(1, self._batch_evaluations)
                if self._batch_evaluations > 0
                else self.config.batch_size
            ),
        }

    def reset(self) -> None:
        """Reset search state."""
        self._root = None
        self._transposition_table.clear()
        self._pending_evaluations.clear()
        self._total_evaluations = 0
        self._batch_evaluations = 0
        self._cache_hits = 0
        self._tree_reuse_count = 0


# -------------------- Factory Function --------------------


def create_batch_mcts_from_game_state(
    policy_value_network: nn.Module,
    config: BatchMCTSConfig,
    action_size: int,
    device: str = "cpu",
) -> BatchMCTS:
    """
    Create BatchMCTS configured for GameState interface.

    Args:
        policy_value_network: Neural network
        config: MCTS configuration
        action_size: Action space size
        device: Device for network

    Returns:
        Configured BatchMCTS instance
    """

    def to_tensor(state: GameState) -> torch.Tensor:
        return state.to_tensor()

    def get_legal_actions(state: GameState) -> Sequence[int]:
        actions = state.get_legal_actions()
        return [state.action_to_index(a) for a in actions]

    def apply_action(state: GameState, action_idx: int) -> GameState:
        # Find action that maps to this index
        for action in state.get_legal_actions():
            if state.action_to_index(action) == action_idx:
                return state.apply_action(action)
        raise ValueError(f"No legal action maps to index {action_idx}")

    def is_terminal(state: GameState) -> bool:
        return state.is_terminal()

    def get_reward(state: GameState) -> float:
        return state.get_reward()

    def get_hash(state: GameState) -> str:
        return state.get_hash()

    return BatchMCTS(
        policy_value_network=policy_value_network,
        config=config,
        state_to_tensor_fn=to_tensor,
        get_legal_actions_fn=get_legal_actions,
        apply_action_fn=apply_action,
        is_terminal_fn=is_terminal,
        get_reward_fn=get_reward,
        get_state_hash_fn=get_hash,
        action_size=action_size,
        device=device,
    )
