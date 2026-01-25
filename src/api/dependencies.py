"""
API Dependencies.
"""

from typing import Annotated

from fastapi import Depends

from src.framework.mcts.llm_guided.integration import UnifiedSearchOrchestrator
from src.training.model_registry import ModelRegistry
from src.training.system_config import SystemConfig

# Global Singleton (for MVP simplicity)
_orchestrator: UnifiedSearchOrchestrator | None = None
_registry: ModelRegistry | None = None


def get_system_config() -> SystemConfig:
    return SystemConfig()


from pathlib import Path


def get_registry(config: Annotated[SystemConfig, Depends(get_system_config)]) -> ModelRegistry:
    global _registry
    if _registry is None:
        registry_path = Path(config.data_dir) / "registry"
        # Use SQL Registry
        _registry = ModelRegistry(registry_dir=registry_path, db_url="sqlite:///./registry.db")
    return _registry


class MockLLMClient:
    async def complete(self, prompt: str) -> str:
        return '{"subproblems": ["mock subproblem"], "refined_code": "print(hello)"}'


def get_orchestrator(
    config: Annotated[SystemConfig, Depends(get_system_config)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> UnifiedSearchOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        # Initialize Orchestrator with "Best" models from registry
        # Mock LLM Client for now if direct import fails or requires key
        llm_client = MockLLMClient()

        _orchestrator = UnifiedSearchOrchestrator(
            llm_client=llm_client,
            # Add adapters loaded from registry here...
        )
    return _orchestrator
