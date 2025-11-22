"""
Web Search Agent for Multi-Agent MCTS Framework.

Provides web search capability with:
- Multiple search provider support (Tavily, SerpAPI, DuckDuckGo)
- RAG integration for search result processing
- Intelligent result synthesis
- Vector storage for search history
- Comprehensive logging and debugging
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.adapters.web_search import (
    SearchResult,
    WebSearchClient,
    WebSearchError,
    WebSearchResponse,
    create_web_search_client,
)
from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class WebSearchAgentConfig:
    """Configuration for Web Search Agent."""

    max_results: int = 5
    include_raw_content: bool = True
    enable_synthesis: bool = True  # Use LLM to synthesize results
    enable_vector_storage: bool = True  # Store results in vector DB
    search_timeout_seconds: float = 10.0
    max_concurrent_searches: int = 3  # For multi-query scenarios


@dataclass
class WebSearchAgentOutput:
    """Output from Web Search Agent processing."""

    query: str
    search_response: WebSearchResponse
    synthesized_answer: str | None = None
    sources: list[dict] | None = None
    metadata: dict | None = None
    execution_time_ms: float = 0.0
    cached: bool = False


class WebSearchAgent:
    """
    Web Search Agent for augmenting multi-agent reasoning with web data.

    Integrates with:
    - Multiple search providers (Tavily, SerpAPI, DuckDuckGo)
    - RAG pipeline for result processing
    - Vector stores for search history
    - LLM for result synthesis
    """

    def __init__(
        self,
        model_adapter: Any | None = None,
        logger: logging.Logger | None = None,
        vector_store: Any | None = None,
        settings: Settings | None = None,
        config: WebSearchAgentConfig | None = None,
        search_client: WebSearchClient | None = None,
    ):
        """
        Initialize Web Search Agent.

        Args:
            model_adapter: LLM adapter for result synthesis
            logger: Logger instance
            vector_store: Vector store for search result storage
            settings: Application settings
            config: Agent configuration
            search_client: Optional pre-configured search client
        """
        self.model_adapter = model_adapter
        self.logger = logger or logging.getLogger(__name__)
        self.vector_store = vector_store
        self.settings = settings or get_settings()
        self.config = config or WebSearchAgentConfig()

        # Create search client if not provided
        if search_client is None:
            try:
                self.search_client = create_web_search_client(settings=self.settings)
            except WebSearchError as e:
                self.logger.warning(f"Failed to create search client: {e}")
                self.search_client = None
        else:
            self.search_client = search_client

        # Statistics
        self._search_count = 0
        self._total_results_retrieved = 0
        self._total_execution_time_ms = 0.0

        self.logger.info(
            f"WebSearchAgent initialized (provider={self.settings.WEB_SEARCH_PROVIDER}, "
            f"max_results={self.config.max_results})"
        )

    async def process(
        self,
        query: str,
        *,
        rag_context: str | None = None,
        max_results: int | None = None,
        include_raw_content: bool | None = None,
        search_type: str = "general",
        **kwargs: Any,
    ) -> dict:
        """
        Process a query with web search.

        Args:
            query: Search query or question
            rag_context: Additional context from RAG (optional)
            max_results: Override default max results
            include_raw_content: Override default content inclusion
            search_type: Type of search ("general", "news", etc.)
            **kwargs: Additional search parameters

        Returns:
            Dictionary with response, metadata, and sources
        """
        import time

        start_time = time.perf_counter()

        self.logger.info(f"WebSearchAgent processing query: {query[:100]}...")

        # Check if search client is available
        if self.search_client is None:
            self.logger.error("Search client not available")
            return {
                "response": "Web search is not configured. Please set up a search provider.",
                "metadata": {"error": "search_client_unavailable"},
                "sources": [],
            }

        try:
            # Determine parameters
            max_results = max_results or self.config.max_results
            include_raw_content = include_raw_content if include_raw_content is not None else self.config.include_raw_content

            # Perform search
            search_response = await self.search_client.search(
                query=query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                search_type=search_type,
                **kwargs,
            )

            self._search_count += 1
            self._total_results_retrieved += len(search_response.results)

            # Store results in vector DB if enabled
            if self.config.enable_vector_storage and self.vector_store:
                await self._store_results_in_vector_db(search_response)

            # Synthesize results if enabled
            synthesized_answer = None
            if self.config.enable_synthesis and self.model_adapter:
                synthesized_answer = await self._synthesize_results(
                    query=query,
                    search_response=search_response,
                    rag_context=rag_context,
                )
            else:
                # Fallback: simple concatenation
                synthesized_answer = self._simple_synthesis(search_response)

            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._total_execution_time_ms += execution_time_ms

            # Build output
            output = WebSearchAgentOutput(
                query=query,
                search_response=search_response,
                synthesized_answer=synthesized_answer,
                sources=[
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "relevance_score": r.relevance_score,
                    }
                    for r in search_response.results
                ],
                metadata={
                    "provider": search_response.provider,
                    "total_results": search_response.total_results,
                    "search_time_ms": search_response.search_time_ms,
                    "execution_time_ms": execution_time_ms,
                    "cached": search_response.cached,
                    "search_type": search_type,
                    "synthesis_enabled": self.config.enable_synthesis,
                    "vector_storage_enabled": self.config.enable_vector_storage,
                },
                execution_time_ms=execution_time_ms,
                cached=search_response.cached,
            )

            self.logger.info(
                f"WebSearchAgent completed: {len(search_response.results)} results "
                f"in {execution_time_ms:.0f}ms (cached={search_response.cached})"
            )

            return {
                "response": synthesized_answer,
                "metadata": output.metadata,
                "sources": output.sources,
            }

        except WebSearchError as e:
            self.logger.error(f"Web search failed: {e}")
            return {
                "response": f"Web search encountered an error: {str(e)}",
                "metadata": {"error": str(e), "error_type": type(e).__name__},
                "sources": [],
            }
        except Exception as e:
            self.logger.exception(f"Unexpected error in WebSearchAgent: {e}")
            return {
                "response": f"An unexpected error occurred during web search: {str(e)}",
                "metadata": {"error": str(e), "error_type": type(e).__name__},
                "sources": [],
            }

    async def _synthesize_results(
        self,
        query: str,
        search_response: WebSearchResponse,
        rag_context: str | None = None,
    ) -> str:
        """
        Synthesize search results using LLM.

        Args:
            query: Original search query
            search_response: Search results
            rag_context: Additional RAG context

        Returns:
            Synthesized answer
        """
        if not self.model_adapter:
            return self._simple_synthesis(search_response)

        try:
            # Build synthesis prompt
            prompt = self._build_synthesis_prompt(query, search_response, rag_context)

            # Generate synthesis
            self.logger.debug(f"Synthesizing {len(search_response.results)} search results")

            response = await self.model_adapter.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for factual synthesis
                max_tokens=1000,
            )

            return response.text.strip()

        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            return self._simple_synthesis(search_response)

    def _build_synthesis_prompt(
        self,
        query: str,
        search_response: WebSearchResponse,
        rag_context: str | None = None,
    ) -> str:
        """Build prompt for LLM synthesis."""
        prompt = f"""Based on the following web search results, provide a comprehensive answer to the query.

Query: {query}

Web Search Results ({search_response.provider}):
"""

        for i, result in enumerate(search_response.results, 1):
            prompt += f"""
{i}. {result.title}
   URL: {result.url}
   Content: {result.snippet}
   Relevance: {result.relevance_score:.2f}
"""

        if rag_context:
            prompt += f"\n\nAdditional Context:\n{rag_context}\n"

        prompt += """
Provide a comprehensive, accurate answer based on these search results. Include:
1. A direct answer to the query
2. Key insights from the sources
3. Any important caveats or limitations

Be concise but thorough. Cite sources by number when making specific claims.

Answer:"""

        return prompt

    def _simple_synthesis(self, search_response: WebSearchResponse) -> str:
        """
        Simple synthesis without LLM (fallback).

        Concatenates top results with formatting.
        """
        if not search_response.results:
            return "No results found for your query."

        synthesis = f"Found {len(search_response.results)} relevant results:\n\n"

        for i, result in enumerate(search_response.results[:5], 1):
            synthesis += f"{i}. **{result.title}**\n"
            synthesis += f"   {result.snippet}\n"
            synthesis += f"   Source: {result.url}\n\n"

        return synthesis.strip()

    async def _store_results_in_vector_db(self, search_response: WebSearchResponse) -> None:
        """
        Store search results in vector database for RAG.

        Args:
            search_response: Search results to store
        """
        if not self.vector_store:
            return

        try:
            # Prepare documents for storage
            documents = []
            for result in search_response.results:
                # Use raw content if available, otherwise use snippet
                content = result.raw_content or result.snippet

                # Create document with metadata
                doc_id = f"search_{result.get_content_hash()}"
                documents.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": {
                        "source": "web_search",
                        "provider": search_response.provider,
                        "query": search_response.query,
                        "title": result.title,
                        "url": result.url,
                        "relevance_score": result.relevance_score,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                })

            # Store in vector DB (implementation depends on vector store type)
            self.logger.debug(f"Storing {len(documents)} search results in vector DB")

            # Example for ChromaDB/FAISS-like interface
            if hasattr(self.vector_store, "add_documents"):
                await self.vector_store.add_documents(documents)
            elif hasattr(self.vector_store, "add_texts"):
                texts = [doc["content"] for doc in documents]
                metadatas = [doc["metadata"] for doc in documents]
                await self.vector_store.add_texts(texts, metadatas=metadatas)

        except Exception as e:
            self.logger.error(f"Failed to store results in vector DB: {e}")

    async def multi_query_search(
        self,
        queries: list[str],
        **kwargs: Any,
    ) -> dict:
        """
        Perform multiple searches concurrently.

        Args:
            queries: List of search queries
            **kwargs: Search parameters

        Returns:
            Combined results from all queries
        """
        self.logger.info(f"Performing multi-query search: {len(queries)} queries")

        # Limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_searches)

        async def search_with_limit(query: str) -> dict:
            async with semaphore:
                return await self.process(query, **kwargs)

        # Execute searches concurrently
        results = await asyncio.gather(
            *[search_with_limit(q) for q in queries],
            return_exceptions=True,
        )

        # Combine results
        all_sources = []
        all_responses = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Query {i} failed: {result}")
                continue

            all_responses.append(result["response"])
            all_sources.extend(result.get("sources", []))

        # Deduplicate sources by URL
        unique_sources = {src["url"]: src for src in all_sources}.values()

        return {
            "response": "\n\n".join(all_responses),
            "metadata": {
                "query_count": len(queries),
                "successful_queries": len([r for r in results if not isinstance(r, Exception)]),
                "total_sources": len(unique_sources),
            },
            "sources": list(unique_sources),
        }

    @property
    def stats(self) -> dict:
        """Get agent statistics."""
        base_stats = {
            "search_count": self._search_count,
            "total_results_retrieved": self._total_results_retrieved,
            "total_execution_time_ms": self._total_execution_time_ms,
            "avg_execution_time_ms": (
                self._total_execution_time_ms / self._search_count if self._search_count > 0 else 0.0
            ),
            "avg_results_per_search": (
                self._total_results_retrieved / self._search_count if self._search_count > 0 else 0.0
            ),
        }

        # Include search client stats if available
        if self.search_client and hasattr(self.search_client, "stats"):
            base_stats["search_client"] = self.search_client.stats

        return base_stats

    async def close(self) -> None:
        """Clean up resources."""
        if self.search_client:
            await self.search_client.close()


# Factory function for backward compatibility
def create_web_search_agent(
    model_adapter: Any,
    logger: logging.Logger,
    vector_store: Any | None = None,
    settings: Settings | None = None,
    **config_kwargs: Any,
) -> WebSearchAgent:
    """
    Factory function to create Web Search Agent.

    Args:
        model_adapter: LLM adapter
        logger: Logger instance
        vector_store: Vector store for RAG
        settings: Application settings
        **config_kwargs: Configuration overrides

    Returns:
        Initialized WebSearchAgent
    """
    config = WebSearchAgentConfig(**config_kwargs)
    return WebSearchAgent(
        model_adapter=model_adapter,
        logger=logger,
        vector_store=vector_store,
        settings=settings,
        config=config,
    )
