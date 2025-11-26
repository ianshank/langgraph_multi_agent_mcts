"""
Google AI Studio Client

This module provides a client for interacting with Google AI Studio / Gemini API
for development and prototyping purposes.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator

import logging

logger = logging.getLogger(__name__)


class GeminiModel(str, Enum):
    """Available Gemini models."""

    GEMINI_2_0_FLASH = "gemini-2.0-flash-001"
    GEMINI_2_0_PRO = "gemini-2.0-pro-001"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"


@dataclass
class AIStudioConfig:
    """
    Configuration for Google AI Studio client.

    For Google AI Studio (development):
        - Set GOOGLE_API_KEY or GEMINI_API_KEY

    For Vertex AI (production):
        - Set GOOGLE_CLOUD_PROJECT
        - Set GOOGLE_GENAI_USE_VERTEXAI=true
        - Use Application Default Credentials
    """

    api_key: str | None = None
    model: GeminiModel = GeminiModel.GEMINI_2_0_FLASH
    temperature: float = 0.7
    max_output_tokens: int = 4096
    use_vertex_ai: bool = False
    project_id: str | None = None
    location: str = "us-central1"

    @classmethod
    def from_env(cls) -> AIStudioConfig:
        """Create configuration from environment variables."""
        use_vertex_ai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

        return cls(
            api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            model=GeminiModel(
                os.getenv("GEMINI_MODEL", GeminiModel.GEMINI_2_0_FLASH.value)
            ),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
            use_vertex_ai=use_vertex_ai,
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.use_vertex_ai and not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY is required for AI Studio. "
                "Set GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI with ADC."
            )
        if self.use_vertex_ai and not self.project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT is required when using Vertex AI."
            )


class AIStudioClient:
    """
    Client for Google AI Studio / Gemini API.

    This client provides a unified interface for:
    - Google AI Studio (development with API key)
    - Vertex AI (production with Google Cloud credentials)

    Example:
        ```python
        # Using API key (development)
        config = AIStudioConfig(api_key="your-api-key")
        client = AIStudioClient(config)

        # Using Vertex AI (production)
        config = AIStudioConfig(
            use_vertex_ai=True,
            project_id="your-project"
        )
        client = AIStudioClient(config)

        # Generate content
        response = await client.generate("What is MCTS?")
        ```
    """

    def __init__(self, config: AIStudioConfig | None = None):
        """
        Initialize the AI Studio client.

        Args:
            config: Client configuration. If None, loads from environment.
        """
        self.config = config or AIStudioConfig.from_env()
        self.config.validate()
        self._client = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the underlying client."""
        if self._initialized:
            return

        try:
            from google import genai

            if self.config.use_vertex_ai:
                # Use Vertex AI backend
                self._client = genai.Client(
                    vertexai=True,
                    project=self.config.project_id,
                    location=self.config.location,
                )
                logger.info(
                    f"Initialized Vertex AI client for project: {self.config.project_id}"
                )
            else:
                # Use AI Studio with API key
                self._client = genai.Client(api_key=self.config.api_key)
                logger.info("Initialized AI Studio client with API key")

            self._initialized = True

        except ImportError:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            )

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt
            system_instruction: Optional system instruction
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Build generation config
            generation_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_output_tokens": kwargs.get(
                    "max_output_tokens", self.config.max_output_tokens
                ),
            }

            # Build the request
            contents = prompt

            # Generate response
            response = self._client.models.generate_content(
                model=self.config.model.value,
                contents=contents,
                config=generation_config,
            )

            return response.text

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the model.

        Args:
            prompt: The input prompt
            system_instruction: Optional system instruction
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated
        """
        if not self._initialized:
            await self.initialize()

        try:
            generation_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_output_tokens": kwargs.get(
                    "max_output_tokens", self.config.max_output_tokens
                ),
            }

            # Stream response
            response = self._client.models.generate_content_stream(
                model=self.config.model.value,
                contents=prompt,
                config=generation_config,
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    async def chat(
        self,
        messages: list[dict[str, str]],
        system_instruction: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Conduct a multi-turn conversation.

        Args:
            messages: List of messages with 'role' and 'content' keys
            system_instruction: Optional system instruction
            **kwargs: Additional generation parameters

        Returns:
            The generated response
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Build contents from messages
            contents = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

            generation_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_output_tokens": kwargs.get(
                    "max_output_tokens", self.config.max_output_tokens
                ),
            }

            response = self._client.models.generate_content(
                model=self.config.model.value,
                contents=contents,
                config=generation_config,
            )

            return response.text

        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model": self.config.model.value,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "backend": "vertex_ai" if self.config.use_vertex_ai else "ai_studio",
            "project": self.config.project_id if self.config.use_vertex_ai else None,
            "location": self.config.location if self.config.use_vertex_ai else None,
        }


# Convenience function for quick usage
async def generate_with_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash-001",
    temperature: float = 0.7,
) -> str:
    """
    Quick helper to generate a response with Gemini.

    Args:
        prompt: The input prompt
        model: Model to use
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    config = AIStudioConfig.from_env()
    config.model = GeminiModel(model)
    config.temperature = temperature

    client = AIStudioClient(config)
    return await client.generate(prompt)
