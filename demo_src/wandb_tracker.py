"""
Weights & Biases integration for experiment tracking.
"""

import os
from datetime import datetime
from typing import Any

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandBTracker:
    """Weights & Biases experiment tracker for multi-agent MCTS demo."""

    def __init__(self, project_name: str = "langgraph-mcts-demo", entity: str | None = None, enabled: bool = True):
        """Initialize W&B tracker.

        Args:
            project_name: W&B project name
            entity: W&B entity (username or team)
            enabled: Whether tracking is enabled
        """
        self.project_name = project_name
        self.entity = entity
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        self.run_id = None

    def is_available(self) -> bool:
        """Check if W&B is available."""
        return WANDB_AVAILABLE

    def init_run(
        self, run_name: str | None = None, config: dict[str, Any] | None = None, tags: list[str] | None = None
    ) -> bool:
        """Initialize a new W&B run.

        Args:
            run_name: Optional name for the run
            config: Configuration dictionary to log
            tags: Tags for the run

        Returns:
            True if run initialized successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Generate run name if not provided
            if run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"mcts_query_{timestamp}"

            # Default tags
            if tags is None:
                tags = ["demo", "multi-agent", "mcts"]

            # Initialize run
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=config or {},
                tags=tags,
                reinit=True,
            )

            self.run_id = self.run.id
            return True

        except Exception as e:
            print(f"W&B init error: {e}")
            self.enabled = False
            return False

    def log_query_config(self, config: dict[str, Any]):
        """Log query configuration.

        Args:
            config: Configuration dictionary with agent settings, MCTS params, etc.
        """
        if not self.enabled or not self.run:
            return

        try:
            wandb.config.update(config)
        except Exception as e:
            print(f"W&B config log error: {e}")

    def log_agent_result(
        self,
        agent_name: str,
        response: str,
        confidence: float,
        execution_time_ms: float,
        reasoning_steps: list[str] | None = None,
    ):
        """Log individual agent results.

        Args:
            agent_name: Name of the agent (HRM, TRM, MCTS)
            response: Agent's response text
            confidence: Confidence score (0-1)
            execution_time_ms: Execution time in milliseconds
            reasoning_steps: Optional list of reasoning steps
        """
        if not self.enabled or not self.run:
            return

        try:
            metrics = {
                f"{agent_name}/confidence": confidence,
                f"{agent_name}/execution_time_ms": execution_time_ms,
                f"{agent_name}/response_length": len(response),
            }

            if reasoning_steps:
                metrics[f"{agent_name}/num_reasoning_steps"] = len(reasoning_steps)

            wandb.log(metrics)

            # Log response as text
            wandb.log({f"{agent_name}/response": wandb.Html(f"<pre>{response}</pre>")})

        except Exception as e:
            print(f"W&B agent result log error: {e}")

    def log_mcts_result(self, mcts_result: dict[str, Any]):
        """Log MCTS-specific metrics.

        Args:
            mcts_result: Dictionary containing MCTS search results
        """
        if not self.enabled or not self.run:
            return

        try:
            # Extract key metrics
            metrics = {
                "mcts/best_value": mcts_result.get("best_value", 0),
                "mcts/root_visits": mcts_result.get("root_visits", 0),
                "mcts/total_nodes": mcts_result.get("total_nodes", 0),
                "mcts/max_depth": mcts_result.get("max_depth_reached", 0),
                "mcts/iterations": mcts_result.get("iterations_completed", 0),
                "mcts/exploration_weight": mcts_result.get("exploration_weight", 1.414),
            }

            wandb.log(metrics)

            # Log top actions as table
            if "top_actions" in mcts_result:
                top_actions_data = []
                for action in mcts_result["top_actions"]:
                    top_actions_data.append(
                        [
                            action.get("action", ""),
                            action.get("visits", 0),
                            action.get("value", 0),
                            action.get("ucb1", 0),
                        ]
                    )

                if top_actions_data:
                    table = wandb.Table(data=top_actions_data, columns=["Action", "Visits", "Value", "UCB1"])
                    wandb.log({"mcts/top_actions_table": table})

            # Log tree visualization as text artifact
            if "tree_visualization" in mcts_result:
                wandb.log({"mcts/tree_visualization": wandb.Html(f"<pre>{mcts_result['tree_visualization']}</pre>")})

        except Exception as e:
            print(f"W&B MCTS result log error: {e}")

    def log_consensus(self, consensus_score: float, agents_used: list[str], final_response: str):
        """Log consensus metrics.

        Args:
            consensus_score: Agreement score between agents (0-1)
            agents_used: List of agent names that were used
            final_response: Final synthesized response
        """
        if not self.enabled or not self.run:
            return

        try:
            wandb.log(
                {
                    "consensus/score": consensus_score,
                    "consensus/num_agents": len(agents_used),
                    "consensus/agents": ", ".join(agents_used),
                    "consensus/final_response_length": len(final_response),
                }
            )

            # Categorize consensus level
            if consensus_score > 0.7:
                consensus_level = "high"
            elif consensus_score > 0.4:
                consensus_level = "medium"
            else:
                consensus_level = "low"

            wandb.log({"consensus/level": consensus_level})

        except Exception as e:
            print(f"W&B consensus log error: {e}")

    def log_performance(self, total_time_ms: float):
        """Log overall performance metrics.

        Args:
            total_time_ms: Total execution time in milliseconds
        """
        if not self.enabled or not self.run:
            return

        try:
            wandb.log({"performance/total_time_ms": total_time_ms, "performance/total_time_s": total_time_ms / 1000})
        except Exception as e:
            print(f"W&B performance log error: {e}")

    def log_full_result(self, result: dict[str, Any]):
        """Log the complete result as an artifact.

        Args:
            result: Full framework result dictionary
        """
        if not self.enabled or not self.run:
            return

        try:
            # Create artifact
            artifact = wandb.Artifact(name=f"query_result_{self.run_id}", type="result")

            # Add result as JSON
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(result, f, indent=2, default=str)
                temp_path = f.name

            artifact.add_file(temp_path, name="result.json")
            wandb.log_artifact(artifact)

            # Clean up temp file
            os.unlink(temp_path)

        except Exception as e:
            print(f"W&B full result log error: {e}")

    def log_query_summary(
        self, query: str, use_hrm: bool, use_trm: bool, use_mcts: bool, consensus_score: float, total_time_ms: float
    ):
        """Log a summary row for the query.

        Args:
            query: The input query
            use_hrm: Whether HRM was enabled
            use_trm: Whether TRM was enabled
            use_mcts: Whether MCTS was enabled
            consensus_score: Final consensus score
            total_time_ms: Total execution time
        """
        if not self.enabled or not self.run:
            return

        try:
            # Create summary table entry
            summary_data = [
                [
                    query[:100] + "..." if len(query) > 100 else query,
                    "✓" if use_hrm else "✗",
                    "✓" if use_trm else "✗",
                    "✓" if use_mcts else "✗",
                    f"{consensus_score:.1%}",
                    f"{total_time_ms:.2f}",
                ]
            ]

            table = wandb.Table(data=summary_data, columns=["Query", "HRM", "TRM", "MCTS", "Consensus", "Time (ms)"])

            wandb.log({"query_summary": table})

        except Exception as e:
            print(f"W&B summary log error: {e}")

    def finish_run(self):
        """Finish the current W&B run."""
        if not self.enabled or not self.run:
            return

        try:
            wandb.finish()
            self.run = None
            self.run_id = None
        except Exception as e:
            print(f"W&B finish error: {e}")

    def get_run_url(self) -> str | None:
        """Get the URL for the current run.

        Returns:
            URL string or None if no active run
        """
        if not self.enabled or not self.run:
            return None

        try:
            return self.run.get_url()
        except Exception:
            return None


# Global tracker instance
_global_tracker: WandBTracker | None = None


def get_tracker(
    project_name: str = "langgraph-mcts-demo", entity: str | None = None, enabled: bool = True
) -> WandBTracker:
    """Get or create the global W&B tracker.

    Args:
        project_name: W&B project name
        entity: W&B entity
        enabled: Whether tracking is enabled

    Returns:
        WandBTracker instance
    """
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = WandBTracker(project_name=project_name, entity=entity, enabled=enabled)

    return _global_tracker


def is_wandb_available() -> bool:
    """Check if W&B is available."""
    return WANDB_AVAILABLE
