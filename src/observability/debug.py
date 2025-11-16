"""
Debug utilities for multi-agent MCTS framework.

Provides:
- MCTS tree visualization (text-based)
- Step-by-step MCTS execution logging when LOG_LEVEL=DEBUG
- UCB score logging at each selection
- State diff tracking between iterations
- Export tree to DOT format for graphviz
"""

import logging
import os
from typing import Any

from .logging import get_logger


class MCTSDebugger:
    """
    Comprehensive debugger for MCTS operations.

    Provides detailed step-by-step logging, tree visualization,
    and state tracking for MCTS execution.
    """

    def __init__(self, session_id: str = "default", enabled: bool | None = None):
        """
        Initialize MCTS debugger.

        Args:
            session_id: Unique identifier for debug session
            enabled: Enable debugging (defaults to LOG_LEVEL=DEBUG)
        """
        self.session_id = session_id
        self.logger = get_logger("observability.debug")

        # Auto-enable if LOG_LEVEL is DEBUG
        if enabled is None:
            log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
            self.enabled = log_level == "DEBUG"
        else:
            self.enabled = enabled

        # State tracking
        self._iteration_count = 0
        self._state_history: list[dict[str, Any]] = []
        self._selection_history: list[dict[str, Any]] = []
        self._ucb_history: list[dict[str, float]] = []

    def log_iteration_start(self, iteration: int) -> None:
        """Log the start of an MCTS iteration."""
        if not self.enabled:
            return

        self._iteration_count = iteration
        self.logger.debug(
            f"=== MCTS Iteration {iteration} START ===",
            extra={
                "debug_event": "iteration_start",
                "mcts_iteration": iteration,
                "session_id": self.session_id,
            },
        )

    def log_selection(
        self,
        node_id: str,
        ucb_score: float,
        visits: int,
        value: float,
        depth: int,
        children_count: int,
        is_selected: bool = False,
    ) -> None:
        """Log UCB score and selection decision for a node."""
        if not self.enabled:
            return

        selection_data = {
            "node_id": node_id,
            "ucb_score": round(ucb_score, 6),
            "visits": visits,
            "value": round(value, 6),
            "avg_value": round(value / max(visits, 1), 6),
            "depth": depth,
            "children_count": children_count,
            "is_selected": is_selected,
        }

        self._selection_history.append(selection_data)

        log_msg = f"Selection: node={node_id} UCB={ucb_score:.4f} " f"visits={visits} value={value:.4f} depth={depth}"

        if is_selected:
            log_msg += " [SELECTED]"

        self.logger.debug(
            log_msg,
            extra={
                "debug_event": "mcts_selection",
                "selection": selection_data,
                "session_id": self.session_id,
                "mcts_iteration": self._iteration_count,
            },
        )

    def log_ucb_comparison(
        self,
        parent_id: str,
        children_ucb: dict[str, float],
        selected_child: str,
    ) -> None:
        """Log UCB score comparison for all children of a node."""
        if not self.enabled:
            return

        self._ucb_history.append(children_ucb)

        ucb_summary = ", ".join(
            [
                f"{cid}={score:.4f}{'*' if cid == selected_child else ''}"
                for cid, score in sorted(children_ucb.items(), key=lambda x: x[1], reverse=True)
            ]
        )

        self.logger.debug(
            f"UCB Comparison at {parent_id}: {ucb_summary}",
            extra={
                "debug_event": "ucb_comparison",
                "parent_id": parent_id,
                "children_ucb": {k: round(v, 6) for k, v in children_ucb.items()},
                "selected_child": selected_child,
                "session_id": self.session_id,
                "mcts_iteration": self._iteration_count,
            },
        )

    def log_expansion(
        self,
        parent_id: str,
        action: str,
        new_node_id: str,
        available_actions: list[str],
    ) -> None:
        """Log node expansion details."""
        if not self.enabled:
            return

        self.logger.debug(
            f"Expansion: parent={parent_id} action={action} new_node={new_node_id} "
            f"available={len(available_actions)} actions",
            extra={
                "debug_event": "mcts_expansion",
                "parent_id": parent_id,
                "action": action,
                "new_node_id": new_node_id,
                "available_actions": available_actions,
                "session_id": self.session_id,
                "mcts_iteration": self._iteration_count,
            },
        )

    def log_simulation(
        self,
        node_id: str,
        simulation_result: float,
        simulation_details: dict[str, Any] | None = None,
    ) -> None:
        """Log simulation/rollout results."""
        if not self.enabled:
            return

        self.logger.debug(
            f"Simulation: node={node_id} result={simulation_result:.4f}",
            extra={
                "debug_event": "mcts_simulation",
                "node_id": node_id,
                "simulation_result": round(simulation_result, 6),
                "simulation_details": simulation_details or {},
                "session_id": self.session_id,
                "mcts_iteration": self._iteration_count,
            },
        )

    def log_backpropagation(
        self,
        path: list[str],
        value: float,
        updates: list[dict[str, Any]],
    ) -> None:
        """Log backpropagation path and value updates."""
        if not self.enabled:
            return

        self.logger.debug(
            f"Backprop: path={' -> '.join(path)} value={value:.4f}",
            extra={
                "debug_event": "mcts_backprop",
                "path": path,
                "value": round(value, 6),
                "updates": updates,
                "session_id": self.session_id,
                "mcts_iteration": self._iteration_count,
            },
        )

    def log_iteration_end(
        self,
        iteration: int,
        best_action: str,
        best_ucb: float,
        tree_size: int,
    ) -> None:
        """Log the end of an MCTS iteration."""
        if not self.enabled:
            return

        self.logger.debug(
            f"=== MCTS Iteration {iteration} END === "
            f"best_action={best_action} UCB={best_ucb:.4f} tree_size={tree_size}",
            extra={
                "debug_event": "iteration_end",
                "mcts_iteration": iteration,
                "best_action": best_action,
                "best_ucb": round(best_ucb, 6),
                "tree_size": tree_size,
                "session_id": self.session_id,
            },
        )

    def log_state_diff(
        self,
        old_state: dict[str, Any],
        new_state: dict[str, Any],
        description: str = "State change",
    ) -> None:
        """Log differences between two states."""
        if not self.enabled:
            return

        diff = self._compute_state_diff(old_state, new_state)

        if diff:
            self._state_history.append(
                {
                    "iteration": self._iteration_count,
                    "description": description,
                    "diff": diff,
                }
            )

            self.logger.debug(
                f"State diff: {description}",
                extra={
                    "debug_event": "state_diff",
                    "description": description,
                    "diff": diff,
                    "session_id": self.session_id,
                    "mcts_iteration": self._iteration_count,
                },
            )

    def _compute_state_diff(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
        prefix: str = "",
    ) -> dict[str, Any]:
        """Compute differences between two dictionaries."""
        diff = {}

        all_keys = set(old.keys()) | set(new.keys())

        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key

            if key not in old:
                diff[full_key] = {"added": new[key]}
            elif key not in new:
                diff[full_key] = {"removed": old[key]}
            elif old[key] != new[key]:
                if isinstance(old[key], dict) and isinstance(new[key], dict):
                    nested_diff = self._compute_state_diff(old[key], new[key], full_key)
                    diff.update(nested_diff)
                else:
                    diff[full_key] = {"old": old[key], "new": new[key]}

        return diff

    def get_debug_summary(self) -> dict[str, Any]:
        """Get summary of debug information collected."""
        return {
            "session_id": self.session_id,
            "total_iterations": self._iteration_count,
            "selection_history_count": len(self._selection_history),
            "state_changes_count": len(self._state_history),
            "ucb_comparisons_count": len(self._ucb_history),
        }


def visualize_mcts_tree(
    root_node: Any,
    max_depth: int = 10,
    max_children: int = 5,
    show_ucb: bool = True,
    indent: str = "  ",
) -> str:
    """
    Generate text-based visualization of MCTS tree.

    Args:
        root_node: Root MCTSNode
        max_depth: Maximum depth to visualize
        max_children: Maximum children to show per node
        show_ucb: Show UCB scores
        indent: Indentation string

    Returns:
        Text representation of the tree
    """
    lines = ["MCTS Tree Visualization", "=" * 40]

    def render_node(node: Any, depth: int = 0, prefix: str = "") -> None:
        if depth > max_depth:
            lines.append(f"{prefix}{indent}... (max depth reached)")
            return

        # Node info
        visits = getattr(node, "visits", 0)
        value = getattr(node, "value", 0.0)
        action = getattr(node, "action", "root")
        state_id = getattr(node, "state_id", "unknown")

        avg_value = value / max(visits, 1)

        node_info = f"[{state_id}] action={action} visits={visits} value={value:.3f} avg={avg_value:.3f}"

        if show_ucb and hasattr(node, "ucb1") and visits > 0:
            try:
                ucb = node.ucb1()
                if ucb != float("inf"):
                    node_info += f" UCB={ucb:.3f}"
            except Exception:
                pass

        lines.append(f"{prefix}{node_info}")

        # Children
        children = getattr(node, "children", [])
        if children:
            # Sort by visits (most visited first)
            sorted_children = sorted(children, key=lambda c: getattr(c, "visits", 0), reverse=True)
            display_children = sorted_children[:max_children]

            for i, child in enumerate(display_children):
                is_last = i == len(display_children) - 1
                child_prefix = prefix + indent + ("└── " if is_last else "├── ")
                next_prefix = prefix + indent + ("    " if is_last else "│   ")

                lines.append(f"{child_prefix[:-4]}")
                render_node(child, depth + 1, next_prefix)

            if len(children) > max_children:
                lines.append(f"{prefix}{indent}... and {len(children) - max_children} more children")

    render_node(root_node)
    lines.append("=" * 40)

    return "\n".join(lines)


def export_tree_to_dot(
    root_node: Any,
    filename: str = "mcts_tree.dot",
    max_depth: int = 10,
    include_ucb: bool = True,
) -> str:
    """
    Export MCTS tree to DOT format for graphviz visualization.

    Args:
        root_node: Root MCTSNode
        filename: Output filename (optional)
        max_depth: Maximum depth to export
        include_ucb: Include UCB scores in labels

    Returns:
        DOT format string
    """
    lines = [
        "digraph MCTSTree {",
        '    graph [rankdir=TB, label="MCTS Tree", fontsize=16];',
        "    node [shape=box, style=filled, fillcolor=lightblue];",
        "    edge [fontsize=10];",
        "",
    ]

    node_counter = [0]  # Use list for mutable counter in closure

    def add_node(node: Any, depth: int = 0, parent_dot_id: str | None = None) -> None:
        if depth > max_depth:
            return

        # Generate unique DOT ID
        dot_id = f"node_{node_counter[0]}"
        node_counter[0] += 1

        # Node attributes
        visits = getattr(node, "visits", 0)
        value = getattr(node, "value", 0.0)
        action = getattr(node, "action", "root")
        state_id = getattr(node, "state_id", "unknown")

        avg_value = value / max(visits, 1)

        # Build label
        label_parts = [
            f"ID: {state_id}",
            f"Action: {action}",
            f"Visits: {visits}",
            f"Value: {value:.3f}",
            f"Avg: {avg_value:.3f}",
        ]

        if include_ucb and hasattr(node, "ucb1") and visits > 0:
            try:
                ucb = node.ucb1()
                if ucb != float("inf"):
                    label_parts.append(f"UCB: {ucb:.3f}")
            except Exception:
                pass

        label = "\\n".join(label_parts)

        # Color based on value
        if avg_value >= 0.7:
            color = "lightgreen"
        elif avg_value >= 0.4:
            color = "lightyellow"
        else:
            color = "lightcoral"

        lines.append(f'    {dot_id} [label="{label}", fillcolor={color}];')

        # Edge from parent
        if parent_dot_id:
            lines.append(f'    {parent_dot_id} -> {dot_id} [label="{action}"];')

        # Process children
        children = getattr(node, "children", [])
        for child in children:
            add_node(child, depth + 1, dot_id)

    add_node(root_node)
    lines.append("}")

    dot_content = "\n".join(lines)

    # Write to file if filename provided
    if filename:
        with open(filename, "w") as f:
            f.write(dot_content)

    return dot_content


def print_debug_banner(message: str, char: str = "=", width: int = 60) -> None:
    """Print a debug banner message."""
    logger = get_logger("observability.debug")
    border = char * width
    logger.debug(border)
    logger.debug(f"{message:^{width}}")
    logger.debug(border)


def log_agent_state_snapshot(
    agent_name: str,
    state: dict[str, Any],
    include_keys: list[str] | None = None,
) -> None:
    """
    Log a snapshot of agent state for debugging.

    Args:
        agent_name: Name of the agent
        state: Current state dictionary
        include_keys: Specific keys to include (None = all)
    """
    logger = get_logger("observability.debug")

    filtered_state = {k: state.get(k) for k in include_keys if k in state} if include_keys else state

    logger.debug(
        f"Agent {agent_name} state snapshot",
        extra={
            "debug_event": "agent_state_snapshot",
            "agent_name": agent_name,
            "state": filtered_state,
        },
    )


def enable_verbose_debugging() -> None:
    """Enable verbose debugging by setting LOG_LEVEL to DEBUG."""
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Reconfigure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    for handler in root_logger.handlers:
        handler.setLevel(logging.DEBUG)

    logger = get_logger("observability.debug")
    logger.info("Verbose debugging ENABLED")


def disable_verbose_debugging() -> None:
    """Disable verbose debugging by setting LOG_LEVEL to INFO."""
    os.environ["LOG_LEVEL"] = "INFO"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers:
        handler.setLevel(logging.INFO)

    logger = get_logger("observability.debug")
    logger.info("Verbose debugging DISABLED")
