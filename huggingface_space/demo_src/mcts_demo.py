"""
Educational MCTS demonstration using the production framework.

This demo uses the real MCTSEngine from src.framework.mcts.core to provide
an authentic learning experience while remaining accessible for demonstrations.
"""

from __future__ import annotations

from typing import Any

from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.policies import RolloutPolicy, SelectionPolicy


class DemoRolloutPolicy(RolloutPolicy):
    """
    Educational rollout policy for demo purposes.

    Evaluates states based on:
    - Depth of exploration (deeper = more thorough)
    - Action quality (domain-specific heuristics)
    - Exploration randomness
    """

    def __init__(self, category: str, action_templates: dict[str, list[str]]):
        """
        Initialize demo rollout policy.

        Args:
            category: Query category for heuristic evaluation
            action_templates: Available action templates for scoring
        """
        self.category = category
        self.action_templates = action_templates

        # Define key terms that indicate quality actions per category
        self.quality_indicators = {
            "architecture": ["scalability", "consistency", "requirements"],
            "optimization": ["profile", "caching", "parallel"],
            "database": ["patterns", "relationships", "scaling"],
            "distributed": ["circuit", "retry", "bulkhead"],
            "default": ["decompose", "constraints", "trade-offs"],
        }

    async def evaluate(
        self,
        state: MCTSState,
        rng,
        max_depth: int = 10,
    ) -> float:
        """
        Evaluate a state through heuristic analysis.

        This combines:
        - Depth bonus: rewards thorough exploration
        - Action quality: rewards domain-appropriate actions
        - Noise: adds exploration randomness

        Args:
            state: State to evaluate
            rng: Random number generator
            max_depth: Maximum depth (unused in heuristic)

        Returns:
            Estimated value in [0, 1] range
        """
        # Base value
        base_value = 0.5

        # Depth bonus: deeper exploration = more value (up to 0.3)
        depth = state.features.get("depth", 0)
        depth_bonus = min(depth * 0.1, 0.3)

        # Action quality bonus
        action_bonus = 0.0
        last_action = state.features.get("last_action", "")

        if last_action:
            # Check if action contains quality indicators for this category
            indicators = self.quality_indicators.get(self.category, self.quality_indicators["default"])
            for term in indicators:
                if term in last_action.lower():
                    action_bonus = 0.15
                    break

        # Add exploration noise
        noise = rng.uniform(-0.1, 0.1)

        # Combine components
        value = base_value + depth_bonus + action_bonus + noise

        # Clamp to [0, 1]
        return max(0.0, min(1.0, value))


class MCTSDemo:
    """
    Educational MCTS demonstration using the production framework.

    This class wraps the production MCTSEngine to provide:
    - Simple, educational interface for demos
    - Category-based action selection
    - Tree visualization for learning
    - Deterministic behavior with seeds

    Unlike the old mock implementation, this uses the real MCTS algorithm
    with all its features: UCB1 selection, progressive widening, caching, etc.
    """

    def __init__(self, max_depth: int = 5):
        """
        Initialize MCTS demo.

        Args:
            max_depth: Maximum tree depth for exploration
        """
        self.max_depth = max_depth

        # Action templates for different query types
        # These provide domain-specific reasoning paths
        self.action_templates = {
            "architecture": [
                "Consider microservices for scalability",
                "Evaluate monolith for simplicity",
                "Analyze team capabilities",
                "Assess deployment requirements",
                "Review data consistency needs",
            ],
            "optimization": [
                "Profile application hotspots",
                "Implement caching layer",
                "Use parallel processing",
                "Optimize database queries",
                "Reduce memory allocations",
            ],
            "database": [
                "Analyze query patterns",
                "Consider data relationships",
                "Evaluate consistency requirements",
                "Plan for horizontal scaling",
                "Assess read/write ratios",
            ],
            "distributed": [
                "Implement circuit breakers",
                "Add retry mechanisms",
                "Use message queues",
                "Apply bulkhead pattern",
                "Design for eventual consistency",
            ],
            "default": [
                "Decompose the problem",
                "Identify constraints",
                "Evaluate trade-offs",
                "Consider alternatives",
                "Validate assumptions",
            ],
        }

    def _categorize_query(self, query: str) -> str:
        """
        Categorize query to select appropriate action templates.

        Args:
            query: User's input query

        Returns:
            Category name for action selection
        """
        query_lower = query.lower()
        if "architecture" in query_lower or "microservice" in query_lower:
            return "architecture"
        elif "optim" in query_lower or "performance" in query_lower:
            return "optimization"
        elif "database" in query_lower or "sql" in query_lower:
            return "database"
        elif "distribut" in query_lower or "fault" in query_lower:
            return "distributed"
        return "default"

    def _create_action_generator(self, category: str):
        """
        Create action generator function for this query category.

        Args:
            category: Query category

        Returns:
            Function that generates actions for a given state
        """

        def action_generator(state: MCTSState) -> list[str]:
            """Generate available actions from current state."""
            # Get category-specific actions
            actions = self.action_templates.get(category, self.action_templates["default"])

            # Filter out already-used actions (track via state features)
            used_actions = state.features.get("used_actions", set())
            available = [a for a in actions if a not in used_actions]

            # If all actions used, allow re-exploring top 2
            if not available:
                return actions[:2]

            return available

        return action_generator

    def _create_state_transition(self, category: str):
        """
        Create state transition function for this query category.

        Args:
            category: Query category

        Returns:
            Function that computes next state from current state + action
        """

        def state_transition(state: MCTSState, action: str) -> MCTSState:
            """Compute next state by applying action."""
            # Track action history
            action_history = list(state.features.get("action_history", []))
            action_history.append(action)

            # Track used actions
            used_actions = set(state.features.get("used_actions", set()))
            used_actions.add(action)

            # Increment depth
            depth = state.features.get("depth", 0) + 1

            # Create new state ID from action history
            state_id = " -> ".join(action_history)

            # Build new state
            new_state = MCTSState(
                state_id=state_id,
                features={
                    "action_history": action_history,
                    "used_actions": used_actions,
                    "depth": depth,
                    "last_action": action,
                    "category": category,
                },
            )

            return new_state

        return state_transition

    def _generate_tree_visualization(self, root: MCTSNode, max_nodes: int = 20) -> str:
        """
        Generate ASCII visualization of the MCTS tree.

        This provides educational insight into the search process.

        Args:
            root: Root node of the tree
            max_nodes: Maximum nodes to display

        Returns:
            ASCII art representation of the tree
        """
        max_nodes = max(1, max_nodes)
        lines = []
        lines.append("MCTS Tree Visualization")
        lines.append("=" * 50)

        nodes_rendered = 0

        def format_node(node: MCTSNode, prefix: str = "", is_last: bool = True) -> list[str]:
            nonlocal nodes_rendered
            result = []

            # Node representation
            connector = "└── " if is_last else "├── "

            if nodes_rendered >= max_nodes:
                result.append(f"{prefix}{connector}... (truncated)")
                return result

            nodes_rendered += 1

            # Display action or state
            node_str = f"{node.state.state_id[:30]}..."
            if node.action:
                node_str = f"{node.action[:25]}..."

            stats = f"[V:{node.visits}, Q:{node.value:.3f}]"

            result.append(f"{prefix}{connector}{node_str} {stats}")

            # Recursively add children
            new_prefix = prefix + ("    " if is_last else "│   ")

            # Limit children shown
            children_to_show = node.children[:3]
            for i, child in enumerate(children_to_show):
                is_child_last = i == len(children_to_show) - 1
                result.extend(format_node(child, new_prefix, is_child_last))

            if len(node.children) > 3:
                result.append(f"{new_prefix}    ... and {len(node.children) - 3} more")

            return result

        # Start with root
        lines.append(f"Root: {root.state.state_id[:40]}... [V:{root.visits}, Q:{root.value:.3f}]")
        nodes_rendered += 1

        for i, child in enumerate(root.children[:5]):
            is_last = i == len(root.children[:5]) - 1
            lines.extend(format_node(child, "", is_last))

        if len(root.children) > 5:
            lines.append(f"... and {len(root.children) - 5} more branches")

        return "\n".join(lines)

    async def search(
        self,
        query: str,
        iterations: int = 25,
        exploration_weight: float = 1.414,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Run MCTS search on the query using the production framework.

        This method demonstrates the full MCTS algorithm:
        1. Selection: UCB1-based tree traversal
        2. Expansion: Progressive widening of nodes
        3. Simulation: Heuristic evaluation (rollout)
        4. Backpropagation: Value updates up the tree

        Args:
            query: The input query to analyze
            iterations: Number of MCTS iterations (more = better but slower)
            exploration_weight: UCB1 exploration constant (higher = more exploration)
            seed: Random seed for deterministic results

        Returns:
            Dictionary with:
            - best_action: Recommended next step
            - best_value: Confidence in recommendation
            - statistics: Search metrics and performance data
            - tree_visualization: ASCII art of search tree
        """
        # Determine query category
        category = self._categorize_query(query)

        # Initialize MCTS engine with production features
        engine = MCTSEngine(
            seed=seed if seed is not None else 42,
            exploration_weight=exploration_weight,
            progressive_widening_k=1.0,  # Moderate expansion
            progressive_widening_alpha=0.5,
            max_parallel_rollouts=4,
            cache_size_limit=10000,
        )

        # Create root state
        root_state = MCTSState(
            state_id=f"Query: {query[:50]}",
            features={
                "query": query,
                "category": category,
                "action_history": [],
                "used_actions": set(),
                "depth": 0,
                "last_action": "",
            },
        )

        # Create root node
        root = MCTSNode(state=root_state, rng=engine.rng)

        # Create domain-specific functions
        action_generator = self._create_action_generator(category)
        state_transition = self._create_state_transition(category)
        rollout_policy = DemoRolloutPolicy(category, self.action_templates)

        # Run MCTS search with production engine
        best_action, stats = await engine.search(
            root=root,
            num_iterations=iterations,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
            max_rollout_depth=self.max_depth,
            selection_policy=SelectionPolicy.MAX_VISITS,  # Most robust
        )

        # Extract best child info
        best_child = None
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)

        # Compile results for demo interface
        result = {
            "best_action": best_action or "No action found",
            "best_value": round(best_child.value, 4) if best_child else 0.0,
            "root_visits": root.visits,
            "total_nodes": engine.get_cached_node_count(),
            "max_depth_reached": engine.get_cached_tree_depth(),
            "iterations_completed": iterations,
            "exploration_weight": exploration_weight,
            "seed": seed,
            "category": category,
            # Top actions sorted by visits
            "top_actions": [
                {
                    "action": child.action,
                    "visits": child.visits,
                    "value": round(child.value, 4),
                    "ucb1": round(child.visits / root.visits if root.visits > 0 else 0.0, 4),  # Simplified UCB display
                }
                for child in sorted(root.children, key=lambda c: -c.visits)[:5]
            ],
            # Framework statistics
            "framework_stats": {
                "cache_hits": stats.get("cache_hits", 0),
                "cache_misses": stats.get("cache_misses", 0),
                "cache_hit_rate": round(stats.get("cache_hit_rate", 0.0), 4),
                "total_simulations": stats.get("total_simulations", 0),
            },
            # Educational visualization
            "tree_visualization": self._generate_tree_visualization(root),
        }

        return result
