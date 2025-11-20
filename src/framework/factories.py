"""
Factory patterns for creating framework components.

This module provides factory classes for creating various components
in a consistent, testable, and modular way. Factories enable:
- Dependency injection
- Component configuration management
- Easy mocking for testing
- Loose coupling between components

Best Practices 2025:
- Use Protocol classes for type safety
- Support async initialization
- Enable configuration from multiple sources
- Provide sensible defaults
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from src.adapters.llm.base import LLMClient
from src.config.settings import Settings, get_settings


class ComponentFactory(Protocol):
    """Protocol for component factories."""

    def create(self, **kwargs: Any) -> Any:
        """Create a component instance with the given configuration."""
        ...


class LLMClientFactory:
    """
    Factory for creating LLM client instances.

    Supports multiple providers with unified configuration.
    Enables easy testing via dependency injection.

    Example:
        >>> factory = LLMClientFactory()
        >>> client = factory.create(provider="openai", model="gpt-4")
        >>> # Or use settings-based creation
        >>> client = factory.create_from_settings()
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize factory with optional settings.

        Args:
            settings: Settings instance (uses defaults if not provided)
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)

    def create(
        self,
        provider: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> LLMClient:
        """
        Create an LLM client with specified configuration.

        Args:
            provider: LLM provider name (openai, anthropic, lmstudio)
            model: Model identifier
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured LLM client instance

        Raises:
            ValueError: If provider is not supported
        """
        from src.adapters.llm import create_client

        provider = provider or self.settings.LLM_PROVIDER
        model = model or self._get_default_model(provider)
        timeout = timeout if timeout is not None else 60.0
        max_retries = max_retries if max_retries is not None else 3

        self.logger.info(f"Creating LLM client: provider={provider}, model={model}")

        return create_client(
            provider=provider,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    def create_from_settings(self) -> LLMClient:
        """
        Create LLM client using current settings.

        Returns:
            Configured LLM client based on environment settings
        """
        return self.create()

    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider."""
        defaults = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-sonnet-20240229",
            "lmstudio": "local-model",
        }
        return defaults.get(provider, "gpt-4-turbo-preview")


class AgentFactory:
    """
    Factory for creating agent instances.

    Manages agent lifecycle and dependencies.

    Example:
        >>> factory = AgentFactory(llm_client=client, logger=logger)
        >>> hrm_agent = factory.create_hrm_agent(max_depth=5)
        >>> trm_agent = factory.create_trm_agent(max_iterations=3)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        logger: logging.Logger | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize agent factory.

        Args:
            llm_client: LLM client for agent communication
            logger: Optional logger instance
            settings: Optional settings instance
        """
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.settings = settings or get_settings()

    def create_hrm_agent(self, **config: Any) -> Any:
        """
        Create a Hierarchical Reasoning Model (HRM) agent.

        Args:
            **config: Agent-specific configuration

        Returns:
            Configured HRM agent instance
        """
        # This would import and create the actual HRM agent
        # For now, this is a placeholder for the pattern
        self.logger.info("Creating HRM agent with config: %s", config)
        # from src.agents.hrm import HRMAgent
        # return HRMAgent(llm_client=self.llm_client, **config)
        raise NotImplementedError("HRM agent creation to be implemented")

    def create_trm_agent(self, **config: Any) -> Any:
        """
        Create a Task Refinement Model (TRM) agent.

        Args:
            **config: Agent-specific configuration

        Returns:
            Configured TRM agent instance
        """
        self.logger.info("Creating TRM agent with config: %s", config)
        # from src.agents.trm import TRMAgent
        # return TRMAgent(llm_client=self.llm_client, **config)
        raise NotImplementedError("TRM agent creation to be implemented")


class MCTSEngineFactory:
    """
    Factory for creating MCTS engine instances.

    Supports different MCTS implementations and configurations.

    Example:
        >>> factory = MCTSEngineFactory()
        >>> engine = factory.create(
        ...     seed=42,
        ...     exploration_weight=1.414,
        ...     config_preset="balanced"
        ... )
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize MCTS engine factory.

        Args:
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)

    def create(
        self,
        seed: int | None = None,
        exploration_weight: float | None = None,
        config_preset: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create an MCTS engine instance.

        Args:
            seed: Random seed for determinism
            exploration_weight: UCB1 exploration constant
            config_preset: Configuration preset name (fast, balanced, thorough)
            **kwargs: Additional engine configuration

        Returns:
            Configured MCTS engine instance
        """
        from src.framework.mcts.core import MCTSEngine

        seed = seed if seed is not None else self.settings.SEED
        exploration_weight = exploration_weight if exploration_weight is not None else self.settings.MCTS_C

        if config_preset:
            config = self._get_preset_config(config_preset)
            kwargs.update(config)

        self.logger.info(
            f"Creating MCTS engine: seed={seed}, "
            f"exploration_weight={exploration_weight}, "
            f"preset={config_preset}"
        )

        return MCTSEngine(
            seed=seed,
            exploration_weight=exploration_weight,
            **kwargs,
        )

    def _get_preset_config(self, preset: str) -> dict[str, Any]:
        """Get configuration for a preset."""
        from src.framework.mcts.config import BALANCED_CONFIG, FAST_CONFIG, THOROUGH_CONFIG

        presets = {
            "fast": FAST_CONFIG.__dict__,
            "balanced": BALANCED_CONFIG.__dict__,
            "thorough": THOROUGH_CONFIG.__dict__,
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Valid presets: {list(presets.keys())}")

        return presets[preset]


class FrameworkFactory:
    """
    Master factory for creating complete framework instances.

    Coordinates all component factories to build the entire system.

    Example:
        >>> factory = FrameworkFactory()
        >>> framework = factory.create_framework(
        ...     llm_provider="openai",
        ...     mcts_enabled=True,
        ...     mcts_seed=42
        ... )
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize framework factory.

        Args:
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        self.llm_factory = LLMClientFactory(settings=self.settings)
        self.mcts_factory = MCTSEngineFactory(settings=self.settings)

    def create_framework(
        self,
        llm_provider: str | None = None,
        mcts_enabled: bool | None = None,
        mcts_seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a complete framework instance.

        Args:
            llm_provider: LLM provider to use
            mcts_enabled: Whether to enable MCTS
            mcts_seed: Random seed for MCTS
            **kwargs: Additional framework configuration

        Returns:
            Configured framework instance
        """
        mcts_enabled = mcts_enabled if mcts_enabled is not None else self.settings.MCTS_ENABLED

        # Create LLM client
        llm_client = self.llm_factory.create(provider=llm_provider)

        # Create agent factory
        agent_factory = AgentFactory(llm_client=llm_client, settings=self.settings)

        # Create MCTS engine if enabled
        mcts_engine = None
        if mcts_enabled:
            mcts_engine = self.mcts_factory.create(seed=mcts_seed)

        self.logger.info(
            f"Creating framework: provider={llm_provider}, "
            f"mcts_enabled={mcts_enabled}, "
            f"seed={mcts_seed}"
        )

        # This would create the actual framework
        # For now, returning a dict with components
        return {
            "llm_client": llm_client,
            "agent_factory": agent_factory,
            "mcts_engine": mcts_engine,
            "settings": self.settings,
        }


# Convenience function for quick framework creation
def create_framework(**kwargs: Any) -> Any:
    """
    Convenience function to create a framework instance.

    Args:
        **kwargs: Framework configuration arguments

    Returns:
        Configured framework instance

    Example:
        >>> framework = create_framework(llm_provider="openai", mcts_enabled=True)
    """
    factory = FrameworkFactory()
    return factory.create_framework(**kwargs)
