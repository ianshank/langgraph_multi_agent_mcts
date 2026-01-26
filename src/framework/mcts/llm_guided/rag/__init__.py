"""
RAG (Retrieval-Augmented Generation) Module for LLM-Guided MCTS.

Provides:
- RAGContextProvider: Retrieves relevant code examples and documentation
- Enhanced prompts with retrieved context
"""

from .context import (
    RAGContext,
    RAGContextProvider,
    RAGContextProviderConfig,
    create_rag_provider,
)
from .prompts import (
    RAGPromptBuilder,
    build_generator_prompt_with_rag,
    build_reflector_prompt_with_rag,
)

__all__ = [
    # Context provider
    "RAGContextProvider",
    "RAGContextProviderConfig",
    "RAGContext",
    "create_rag_provider",
    # Prompts
    "RAGPromptBuilder",
    "build_generator_prompt_with_rag",
    "build_reflector_prompt_with_rag",
]
