"""
Simplified MCTS implementation for Hugging Face Spaces demo.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""

    state: str
    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    action: str = ""
    depth: int = 0

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb1_score(self, exploration_weight: float = 1.414) -> float:
        """Calculate UCB1 score for node selection."""
        if self.visits == 0:
            return float("inf")

        if self.parent is None or self.parent.visits == 0:
            return self.value

        exploitation = self.value
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_weight: float = 1.414) -> "MCTSNode":
        """Select best child based on UCB1."""
        if not self.children:
            raise ValueError("No children to select from")

        return max(self.children, key=lambda c: c.ucb1_score(exploration_weight))

    def add_child(self, state: str, action: str) -> "MCTSNode":
        """Add a child node."""
        child = MCTSNode(state=state, parent=self, action=action, depth=self.depth + 1)
        self.children.append(child)
        return child


class MCTSDemo:
    """Simplified MCTS for demonstration purposes."""

    def __init__(self, max_depth: int = 5):
        """Initialize MCTS demo.

        Args:
            max_depth: Maximum tree depth
        """
        self.max_depth = max_depth

        # Action templates for different query types
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

    def _get_possible_actions(self, state: str, category: str) -> list[str]:
        """Get possible actions from current state."""
        # Use category-specific actions
        actions = self.action_templates.get(category, self.action_templates["default"])

        # Filter out actions already taken (simplified state tracking)
        used_actions = set(state.split(" -> ")[1:]) if " -> " in state else set()
        available = [a for a in actions if a not in used_actions]

        return available if available else actions[:2]

    def _simulate(self, node: MCTSNode, category: str) -> float:
        """Simulate random playout from node and return value."""
        # Simple heuristic: deeper exploration with good actions = higher value
        base_value = 0.5

        # Bonus for depth (exploring more options)
        depth_bonus = min(node.depth * 0.1, 0.3)

        # Bonus for action quality (simplified)
        action_bonus = 0.0
        if node.action:
            # Some actions are "better" for certain categories
            key_terms = {
                "architecture": ["scalability", "consistency", "requirements"],
                "optimization": ["profile", "caching", "parallel"],
                "database": ["patterns", "relationships", "scaling"],
                "distributed": ["circuit", "retry", "bulkhead"],
            }
            for term in key_terms.get(category, []):
                if term in node.action.lower():
                    action_bonus = 0.15
                    break

        # Add randomness for exploration
        noise = random.uniform(-0.1, 0.1)

        value = base_value + depth_bonus + action_bonus + noise
        return max(0.0, min(1.0, value))

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent

    def _generate_tree_visualization(self, root: MCTSNode, max_nodes: int = 20) -> str:
        """Generate ASCII visualization of the tree."""
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

            node_str = f"{node.state[:30]}..."
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
        lines.append(f"Root: {root.state[:40]}... [V:{root.visits}, Q:{root.value:.3f}]")
        nodes_rendered += 1

        for i, child in enumerate(root.children[:5]):
            is_last = i == len(root.children[:5]) - 1
            lines.extend(format_node(child, "", is_last))

        if len(root.children) > 5:
            lines.append(f"... and {len(root.children) - 5} more branches")

        return "\n".join(lines)

    def search(
        self, query: str, iterations: int = 25, exploration_weight: float = 1.414, seed: int | None = None
    ) -> dict[str, Any]:
        """Run MCTS search on the query.

        Args:
            query: The input query
            iterations: Number of MCTS iterations
            exploration_weight: UCB1 exploration parameter
            seed: Random seed for reproducibility

        Returns:
            Dictionary with search results
        """
        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)

        # Initialize root
        root = MCTSNode(state=f"Query: {query[:50]}")
        category = self._categorize_query(query)

        # Track statistics
        total_nodes = 1
        max_depth_reached = 0

        # MCTS iterations
        for _ in range(iterations):
            # Selection - traverse tree using UCB1
            node = root
            path = [node]

            while node.children and node.depth < self.max_depth:
                node = node.best_child(exploration_weight)
                path.append(node)

            # Expansion - add new child if not at max depth
            if node.depth < self.max_depth and node.visits > 0:
                actions = self._get_possible_actions(node.state, category)
                if actions:
                    action = random.choice(actions)
                    new_state = f"{node.state} -> {action}"
                    node = node.add_child(new_state, action)
                    path.append(node)
                    total_nodes += 1

            # Simulation - evaluate the node
            value = self._simulate(node, category)

            # Backpropagation - update statistics
            self._backpropagate(node, value)

            # Track max depth
            max_depth_reached = max(max_depth_reached, node.depth)

        # Find best action from root
        best_child = None
        best_action = "No action found"
        best_value = 0.0

        if root.children:
            # Select by visit count (more robust than value)
            best_child = max(root.children, key=lambda c: c.visits)
            best_action = best_child.action
            best_value = best_child.value

        # Compile results
        result = {
            "best_action": best_action,
            "best_value": round(best_value, 4),
            "root_visits": root.visits,
            "total_nodes": total_nodes,
            "max_depth_reached": max_depth_reached,
            "iterations_completed": iterations,
            "exploration_weight": exploration_weight,
            "seed": seed,
            "top_actions": [
                {
                    "action": child.action,
                    "visits": child.visits,
                    "value": round(child.value, 4),
                    "ucb1": round(child.ucb1_score(exploration_weight), 4),
                }
                for child in sorted(root.children, key=lambda c: -c.visits)[:5]
            ],
            "tree_visualization": self._generate_tree_visualization(root),
        }

        # Reset random seed
        if seed is not None:
            random.seed()

        return result
