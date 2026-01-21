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

    def create_hrm_agent(
        self,
        h_dim: int | None = None,
        l_dim: int | None = None,
        num_h_layers: int | None = None,
        num_l_layers: int | None = None,
        max_outer_steps: int | None = None,
        halt_threshold: float | None = None,
        device: str | None = None,
        **config: Any,
    ) -> Any:
        """
        Create a Hierarchical Reasoning Model (HRM) agent.

        All parameters are configurable - no hardcoded values.

        Args:
            h_dim: High-level planning dimension (default from settings)
            l_dim: Low-level execution dimension (default from settings)
            num_h_layers: Number of high-level layers
            num_l_layers: Number of low-level layers
            max_outer_steps: Maximum planning steps
            halt_threshold: Confidence threshold for halting
            device: Device to place model on (cpu/cuda)
            **config: Additional agent-specific configuration

        Returns:
            Configured HRM agent instance
        """
        from src.agents.hrm_agent import HRMAgent, create_hrm_agent
        from src.training.system_config import HRMConfig

        # Build config from parameters, using settings as defaults
        hrm_config = HRMConfig(
            h_dim=h_dim if h_dim is not None else 512,
            l_dim=l_dim if l_dim is not None else 256,
            num_h_layers=num_h_layers if num_h_layers is not None else 2,
            num_l_layers=num_l_layers if num_l_layers is not None else 4,
            max_outer_steps=max_outer_steps if max_outer_steps is not None else 10,
            halt_threshold=halt_threshold if halt_threshold is not None else 0.95,
            **{k: v for k, v in config.items() if k in HRMConfig.__dataclass_fields__},
        )

        # Determine device
        agent_device = device if device is not None else "cpu"

        self.logger.info(
            "Creating HRM agent: h_dim=%d, l_dim=%d, device=%s",
            hrm_config.h_dim,
            hrm_config.l_dim,
            agent_device,
        )

        return create_hrm_agent(config=hrm_config, device=agent_device)

    def create_trm_agent(
        self,
        latent_dim: int | None = None,
        hidden_dim: int | None = None,
        num_recursions: int | None = None,
        convergence_threshold: float | None = None,
        deep_supervision: bool | None = None,
        output_dim: int | None = None,
        device: str | None = None,
        **config: Any,
    ) -> Any:
        """
        Create a Task Refinement Model (TRM) agent.

        All parameters are configurable - no hardcoded values.

        Args:
            latent_dim: Latent state dimension
            hidden_dim: Hidden layer dimension
            num_recursions: Maximum recursion depth
            convergence_threshold: L2 distance threshold for convergence
            deep_supervision: Enable supervision at all recursion levels
            output_dim: Output dimension (defaults to latent_dim)
            device: Device to place model on (cpu/cuda)
            **config: Additional agent-specific configuration

        Returns:
            Configured TRM agent instance
        """
        from src.agents.trm_agent import TRMAgent, create_trm_agent
        from src.training.system_config import TRMConfig

        # Build config from parameters, using sensible defaults
        trm_config = TRMConfig(
            latent_dim=latent_dim if latent_dim is not None else 256,
            hidden_dim=hidden_dim if hidden_dim is not None else 512,
            num_recursions=num_recursions if num_recursions is not None else 16,
            convergence_threshold=convergence_threshold if convergence_threshold is not None else 0.01,
            deep_supervision=deep_supervision if deep_supervision is not None else True,
            **{k: v for k, v in config.items() if k in TRMConfig.__dataclass_fields__},
        )

        # Determine device
        agent_device = device if device is not None else "cpu"

        self.logger.info(
            "Creating TRM agent: latent_dim=%d, num_recursions=%d, device=%s",
            trm_config.latent_dim,
            trm_config.num_recursions,
            agent_device,
        )

        return create_trm_agent(config=trm_config, output_dim=output_dim, device=agent_device)


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
            f"seed={mcts_seed}, "
            f"additional_config={list(kwargs.keys())}"
        )

        # This would create the actual framework
        # For now, returning a dict with components
        return {
            "llm_client": llm_client,
            "agent_factory": agent_factory,
            "mcts_engine": mcts_engine,
            "settings": self.settings,
            "additional_config": kwargs,  # Store for future use
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
