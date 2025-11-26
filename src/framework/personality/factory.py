"""
Factory for creating personality-enhanced agents.

Provides:
- Factory pattern for agent creation
- Builder pattern for fluent configuration
- Backward-compatible agent wrapping
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from .config import PersonalityConfig
from .modules import (
    AspirationModule,
    CuriosityModule,
    EthicalReasoningModule,
    LoyaltyModule,
    TransparencyModule,
)
from .profiles import PersonalityProfile
from .protocols import PersonalityModuleProtocol

if TYPE_CHECKING:
    from src.framework.agents.base import AsyncAgentBase

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="AsyncAgentBase")


class PersonalityFactory:
    """Factory for creating personality-enhanced agents.

    Provides centralized creation of personality modules
    and agent wrapping with dependency injection support.

    Example:
        >>> agent = PersonalityFactory.create_personality_agent(
        ...     base_agent=hrm_agent,
        ...     personality=PersonalityProfile(loyalty=0.9),
        ... )
    """

    # Registry of custom module classes
    _module_registry: dict[str, type[PersonalityModuleProtocol]] = {}

    @classmethod
    def register_module(
        cls,
        trait_name: str,
        module_class: type[PersonalityModuleProtocol],
    ) -> None:
        """Register a custom personality module.

        Args:
            trait_name: Name of the trait this module handles
            module_class: Module class implementing PersonalityModuleProtocol
        """
        cls._module_registry[trait_name] = module_class
        logger.info("Registered custom module for trait: %s", trait_name)

    @classmethod
    def create_modules(
        cls,
        personality: PersonalityProfile,
        config: PersonalityConfig | None = None,
    ) -> dict[str, PersonalityModuleProtocol]:
        """Create all personality modules for a profile.

        Args:
            personality: Personality profile
            config: Optional personality configuration

        Returns:
            Dictionary of trait name to module instance
        """
        config = config or PersonalityConfig()

        modules: dict[str, PersonalityModuleProtocol] = {
            "loyalty": LoyaltyModule(
                personality=personality,
                config=config.loyalty_config,
            ),
            "curiosity": CuriosityModule(
                personality=personality,
                config=config.curiosity_config,
            ),
            "aspiration": AspirationModule(
                personality=personality,
                config=config.aspiration_config,
            ),
            "ethical": EthicalReasoningModule(
                personality=personality,
                config=config.ethical_config,
            ),
            "transparency": TransparencyModule(
                personality=personality,
                config=config.transparency_config,
            ),
        }

        # Add any registered custom modules
        for trait_name, module_class in cls._module_registry.items():
            if trait_name not in modules:
                trait_value = getattr(personality, trait_name, 0.5)
                modules[trait_name] = module_class(trait_value=trait_value)

        return modules

    @classmethod
    def create_personality_agent(
        cls,
        base_agent: T,
        personality: PersonalityProfile | dict[str, float],
        config: PersonalityConfig | None = None,
        custom_modules: dict[str, PersonalityModuleProtocol] | None = None,
    ) -> T:
        """Create a personality-enhanced agent.

        Wraps an existing agent with personality modules using
        the decorator pattern.

        Args:
            base_agent: Agent to enhance with personality
            personality: Personality profile or dict of traits
            config: Optional personality configuration
            custom_modules: Optional custom module implementations

        Returns:
            PersonalityDrivenAgent wrapping the base agent
        """
        # Convert dict to PersonalityProfile
        if isinstance(personality, dict):
            personality = PersonalityProfile(**personality)

        # Create modules
        modules = cls.create_modules(personality, config)

        # Override with custom modules
        if custom_modules:
            modules.update(custom_modules)

        # Import here to avoid circular dependency
        from .agent import PersonalityDrivenAgent

        return PersonalityDrivenAgent(  # type: ignore
            base_agent=base_agent,
            personality=personality,
            modules=modules,
            config=config,
        )

    @classmethod
    def wrap_if_enabled(
        cls,
        agent: T,
        config: PersonalityConfig | None = None,
    ) -> T:
        """Conditionally wrap agent with personality.

        If config is None or disabled, returns agent unchanged.
        Provides backward compatibility.

        Args:
            agent: Agent to potentially wrap
            config: Optional personality configuration

        Returns:
            Original agent or PersonalityDrivenAgent
        """
        if config is None or not config.enabled:
            return agent  # No wrapping

        # Determine profile based on agent type
        agent_name = getattr(agent, "name", str(type(agent).__name__))
        profile = config.get_profile_for_agent(agent_name)

        return cls.create_personality_agent(
            base_agent=agent,
            personality=profile,
            config=config,
        )


class PersonalityAgentBuilder:
    """Builder for fluent personality agent configuration.

    Example:
        >>> agent = (
        ...     PersonalityAgentBuilder(hrm_agent)
        ...     .with_loyalty(0.9)
        ...     .with_curiosity(0.8)
        ...     .with_transparency(0.95)
        ...     .build()
        ... )
    """

    def __init__(self, base_agent: AsyncAgentBase) -> None:
        """Initialize builder with base agent.

        Args:
            base_agent: Agent to enhance
        """
        self._base_agent = base_agent
        self._traits: dict[str, float] = {}
        self._config: PersonalityConfig | None = None
        self._custom_modules: dict[str, PersonalityModuleProtocol] = {}

    def with_loyalty(self, value: float) -> PersonalityAgentBuilder:
        """Set loyalty trait.

        Args:
            value: Trait value [0.0, 1.0]

        Returns:
            Self for chaining
        """
        self._traits["loyalty"] = value
        return self

    def with_curiosity(self, value: float) -> PersonalityAgentBuilder:
        """Set curiosity trait.

        Args:
            value: Trait value [0.0, 1.0]

        Returns:
            Self for chaining
        """
        self._traits["curiosity"] = value
        return self

    def with_aspiration(self, value: float) -> PersonalityAgentBuilder:
        """Set aspiration trait.

        Args:
            value: Trait value [0.0, 1.0]

        Returns:
            Self for chaining
        """
        self._traits["aspiration"] = value
        return self

    def with_ethical_weight(self, value: float) -> PersonalityAgentBuilder:
        """Set ethical weight trait.

        Args:
            value: Trait value [0.0, 1.0]

        Returns:
            Self for chaining
        """
        self._traits["ethical_weight"] = value
        return self

    def with_transparency(self, value: float) -> PersonalityAgentBuilder:
        """Set transparency trait.

        Args:
            value: Trait value [0.0, 1.0]

        Returns:
            Self for chaining
        """
        self._traits["transparency"] = value
        return self

    def with_profile(
        self,
        profile: PersonalityProfile,
    ) -> PersonalityAgentBuilder:
        """Set complete personality profile.

        Args:
            profile: Personality profile

        Returns:
            Self for chaining
        """
        self._traits = {
            "loyalty": profile.loyalty,
            "curiosity": profile.curiosity,
            "aspiration": profile.aspiration,
            "ethical_weight": profile.ethical_weight,
            "transparency": profile.transparency,
        }
        return self

    def with_config(
        self,
        config: PersonalityConfig,
    ) -> PersonalityAgentBuilder:
        """Set personality configuration.

        Args:
            config: Personality configuration

        Returns:
            Self for chaining
        """
        self._config = config
        return self

    def with_custom_module(
        self,
        trait_name: str,
        module: PersonalityModuleProtocol,
    ) -> PersonalityAgentBuilder:
        """Add custom personality module.

        Args:
            trait_name: Name for the module
            module: Module instance

        Returns:
            Self for chaining
        """
        self._custom_modules[trait_name] = module
        return self

    def build(self) -> AsyncAgentBase:
        """Build the personality-enhanced agent.

        Returns:
            PersonalityDrivenAgent wrapping the base agent
        """
        # Create profile from accumulated traits
        profile = PersonalityProfile(**self._traits) if self._traits else PersonalityProfile()

        return PersonalityFactory.create_personality_agent(
            base_agent=self._base_agent,
            personality=profile,
            config=self._config,
            custom_modules=self._custom_modules if self._custom_modules else None,
        )
