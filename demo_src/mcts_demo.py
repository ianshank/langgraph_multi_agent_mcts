"""
Integrated MCTS Demo.

Wraps the production MCTS engine for demonstration purposes.
"""

from typing import Any

from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.policies import RandomRolloutPolicy


class MCTSDemo:
    """
    MCTS Demo wrapper using the actual production engine.
    """

    def __init__(self, max_depth: int = 5):
        """
        Initialize MCTS demo.

        Args:
            max_depth: Maximum tree depth
        """
        self.max_depth = max_depth
        self.engine = MCTSEngine(
            seed=42,
            exploration_weight=1.414,
            progressive_widening_k=1.0,
            progressive_widening_alpha=0.5,
        )

        # Action templates for different query types (kept for demo flavor)
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
        """Categorize query for action selection."""
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

    async def search(
        self, query: str, iterations: int = 25, exploration_weight: float = 1.414, seed: int | None = None
    ) -> dict[str, Any]:
        """
        Run MCTS search on the query using the production engine.

        Args:
            query: The input query
            iterations: Number of MCTS iterations
            exploration_weight: UCB1 exploration parameter
            seed: Random seed for reproducibility

        Returns:
            Dictionary with search results
        """
        if seed is not None:
            self.engine.reset_seed(seed)

        self.engine.exploration_weight = exploration_weight

        # Setup state and actions
        category = self._categorize_query(query)
        possible_actions = self.action_templates.get(category, self.action_templates["default"])

        def action_generator(state: MCTSState) -> list[str]:
            """Generate actions based on state."""
            # Simple logic: avoid repeating actions in path
            history = state.features.get("history", [])
            available = [a for a in possible_actions if a not in history]
            if not available or len(history) >= self.max_depth:
                return []
            return available

        def state_transition(state: MCTSState, action: str) -> MCTSState:
            """Transition to new state."""
            history = state.features.get("history", []) + [action]
            new_id = f"State_{len(history)}"
            return MCTSState(state_id=new_id, features={"history": history, "query": query})

        # Create root
        root = MCTSNode(state=MCTSState(state_id="root", features={"history": [], "query": query}), rng=self.engine.rng)

        # Run search
        best_action, stats = await self.engine.search(
            root=root,
            num_iterations=iterations,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=RandomRolloutPolicy(),
            max_rollout_depth=self.max_depth,
        )

        # augment stats with visualization
        stats["tree_visualization"] = self._generate_tree_visualization(root)

        # Add top actions for compatibility
        if root.children:
            top_children = sorted(root.children, key=lambda c: -c.visits)[:5]
            stats["top_actions"] = [
                {
                    "action": child.action,
                    "visits": child.visits,
                    "value": round(child.value, 4),
                    "ucb1": round(
                        child.select_child(exploration_weight).value if child.children else 0.0, 4
                    ),  # approximate
                }
                for child in top_children
            ]
        else:
            stats["top_actions"] = []

        return stats

    def _generate_tree_visualization(self, root: MCTSNode, max_nodes: int = 20) -> str:
        """Generate ASCII visualization of the tree."""
        # Simplified visualizer compatible with MCTSNode
        max_nodes = max(1, max_nodes)
        lines = []
        lines.append("MCTS Tree Visualization (Production Engine)")
        lines.append("=" * 50)

        nodes_rendered = 0

        def format_node(node: MCTSNode, prefix: str = "", is_last: bool = True) -> list[str]:
            nonlocal nodes_rendered
            result = []
            connector = "└── " if is_last else "├── "

            if nodes_rendered >= max_nodes:
                result.append(f"{prefix}{connector}... (truncated)")
                return result

            nodes_rendered += 1

            # Construct node string
            node_str = "Root" if not node.action else f"{node.action[:30]}..."
            stats = f"[V:{node.visits}, Q:{node.value:.3f}]"

            result.append(f"{prefix}{connector}{node_str} {stats}")

            # Children
            new_prefix = prefix + ("    " if is_last else "│   ")
            children_to_show = node.children[:3]
            for i, child in enumerate(children_to_show):
                is_child_last = i == len(children_to_show) - 1
                result.extend(format_node(child, new_prefix, is_child_last))

            if len(node.children) > 3:
                result.append(f"{new_prefix}    ... and {len(node.children) - 3} more")

            return result

        lines.extend(format_node(root))
        return "\n".join(lines)
