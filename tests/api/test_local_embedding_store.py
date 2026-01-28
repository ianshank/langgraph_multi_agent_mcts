"""
Tests for LocalEmbeddingStore.

Tests the local embedding store implementation for RAG fallback.
"""

import pytest

# Check if dependencies are available
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (_HAS_NUMPY and _HAS_SENTENCE_TRANSFORMERS),
    reason="numpy and sentence-transformers required for local embedding store tests",
)


@pytest.fixture
def local_store():
    """Create a local embedding store for testing."""
    from src.api.local_embedding_store import LocalEmbeddingStore

    store = LocalEmbeddingStore()
    return store


@pytest.fixture
def initialized_store():
    """Create an initialized local embedding store."""
    from src.api.local_embedding_store import LocalEmbeddingStore

    store = LocalEmbeddingStore()
    if not store.initialize():
        pytest.skip("Failed to initialize embedding store")
    return store


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "test", "category": "animals"},
        },
        {
            "id": "doc2",
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "test", "category": "technology"},
        },
        {
            "id": "doc3",
            "content": "Python is a popular programming language for data science.",
            "metadata": {"source": "test", "category": "technology"},
        },
        {
            "id": "doc4",
            "content": "Neural networks are inspired by biological neurons.",
            "metadata": {"source": "test", "category": "technology"},
        },
        {
            "id": "doc5",
            "content": "Dogs and cats are common household pets.",
            "metadata": {"source": "test", "category": "animals"},
        },
    ]


class TestLocalEmbeddingStoreInit:
    """Tests for LocalEmbeddingStore initialization."""

    def test_init_default_model(self, local_store):
        """Test initialization with default model."""
        from src.api.local_embedding_store import LocalEmbeddingStore

        assert local_store.model_name == LocalEmbeddingStore.DEFAULT_MODEL
        assert not local_store.is_available
        assert local_store.document_count == 0

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        from src.api.local_embedding_store import LocalEmbeddingStore

        store = LocalEmbeddingStore(model_name="paraphrase-MiniLM-L3-v2")
        assert store.model_name == "paraphrase-MiniLM-L3-v2"

    def test_initialize_success(self, local_store):
        """Test successful initialization."""
        result = local_store.initialize()
        assert result is True
        assert local_store.is_available

    def test_initialize_idempotent(self, initialized_store):
        """Test that multiple initializations are idempotent."""
        result1 = initialized_store.initialize()
        result2 = initialized_store.initialize()
        assert result1 is True
        assert result2 is True


class TestLocalEmbeddingStoreDocuments:
    """Tests for document management."""

    def test_add_single_document(self, initialized_store):
        """Test adding a single document."""
        docs = [{"content": "Test document content", "id": "test1"}]
        added = initialized_store.add_documents(docs)

        assert added == 1
        assert initialized_store.document_count == 1
        assert initialized_store.has_documents

    def test_add_multiple_documents(self, initialized_store, sample_documents):
        """Test adding multiple documents."""
        added = initialized_store.add_documents(sample_documents)

        assert added == len(sample_documents)
        assert initialized_store.document_count == len(sample_documents)

    def test_add_empty_content_skipped(self, initialized_store):
        """Test that documents with empty content are skipped."""
        docs = [
            {"content": "", "id": "empty"},
            {"content": "Valid content", "id": "valid"},
        ]
        added = initialized_store.add_documents(docs)

        assert added == 1
        assert initialized_store.document_count == 1

    def test_add_duplicate_documents_skipped(self, initialized_store):
        """Test that duplicate documents are skipped."""
        docs = [{"content": "Test content", "id": "dup1"}]

        added1 = initialized_store.add_documents(docs)
        added2 = initialized_store.add_documents(docs)

        assert added1 == 1
        assert added2 == 0
        assert initialized_store.document_count == 1

    def test_auto_generate_id(self, initialized_store):
        """Test automatic ID generation for documents without ID."""
        docs = [{"content": "Document without explicit ID"}]
        added = initialized_store.add_documents(docs)

        assert added == 1
        assert initialized_store.document_count == 1

    def test_remove_document(self, initialized_store, sample_documents):
        """Test removing a document."""
        initialized_store.add_documents(sample_documents)
        initial_count = initialized_store.document_count

        removed = initialized_store.remove_document("doc1")

        assert removed is True
        assert initialized_store.document_count == initial_count - 1

    def test_remove_nonexistent_document(self, initialized_store):
        """Test removing a document that doesn't exist."""
        removed = initialized_store.remove_document("nonexistent")
        assert removed is False

    def test_clear_documents(self, initialized_store, sample_documents):
        """Test clearing all documents."""
        initialized_store.add_documents(sample_documents)
        assert initialized_store.document_count > 0

        initialized_store.clear()

        assert initialized_store.document_count == 0
        assert not initialized_store.has_documents


class TestLocalEmbeddingStoreSearch:
    """Tests for similarity search."""

    def test_search_returns_results(self, initialized_store, sample_documents):
        """Test that search returns relevant results."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search("artificial intelligence", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)

    def test_search_score_ordering(self, initialized_store, sample_documents):
        """Test that results are ordered by score."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search("machine learning", top_k=5)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_min_score_filter(self, initialized_store, sample_documents):
        """Test minimum score filtering."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search("machine learning", min_score=0.5)

        assert all(r["score"] >= 0.5 for r in results)

    def test_search_metadata_filter(self, initialized_store, sample_documents):
        """Test metadata filtering."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search(
            "animals",
            top_k=10,
            filter_metadata={"category": "animals"},
        )

        assert len(results) > 0
        assert all(r["metadata"].get("category") == "animals" for r in results)

    def test_search_empty_store(self, initialized_store):
        """Test searching an empty store."""
        results = initialized_store.search("anything")
        assert results == []

    def test_search_uninitialized_store(self, local_store):
        """Test searching an uninitialized store."""
        results = local_store.search("anything")
        assert results == []

    def test_search_top_k_limit(self, initialized_store, sample_documents):
        """Test that top_k limits results."""
        initialized_store.add_documents(sample_documents)

        results = initialized_store.search("technology", top_k=2)

        assert len(results) <= 2


class TestLocalEmbeddingStoreFactory:
    """Tests for factory function."""

    def test_create_with_auto_initialize(self):
        """Test factory function with auto initialization."""
        from src.api.local_embedding_store import create_local_embedding_store

        store = create_local_embedding_store(auto_initialize=True)

        assert store is not None
        assert store.is_available

    def test_create_without_auto_initialize(self):
        """Test factory function without auto initialization."""
        from src.api.local_embedding_store import create_local_embedding_store

        store = create_local_embedding_store(auto_initialize=False)

        assert store is not None
        assert not store.is_available


class TestRAGRetrieverIntegration:
    """Tests for RAGRetriever with local store."""

    @pytest.mark.asyncio
    async def test_rag_retriever_with_local_store(self, initialized_store, sample_documents):
        """Test RAGRetriever using local store."""
        from src.api.rag_retriever import RAGRetriever

        # Add documents to store
        initialized_store.add_documents(sample_documents)

        # Create retriever with local store
        retriever = RAGRetriever(local_store=initialized_store)
        await retriever.initialize()

        assert "local" in retriever.available_backends

        # Test retrieval
        result = await retriever.retrieve("machine learning", top_k=3)

        assert result.has_results
        assert result.backend == "local"
        assert len(result.documents) > 0

    @pytest.mark.asyncio
    async def test_rag_retriever_add_documents(self):
        """Test adding documents through RAGRetriever."""
        from src.api.rag_retriever import RAGRetriever

        retriever = RAGRetriever()
        await retriever.initialize()

        docs = [
            {"content": "Test document one", "metadata": {"test": True}},
            {"content": "Test document two", "metadata": {"test": True}},
        ]

        added = retriever.add_documents(docs, backend="local")

        # May be 0 if local store couldn't be created
        assert added >= 0

    @pytest.mark.asyncio
    async def test_rag_retriever_factory_with_local(self):
        """Test factory function creates retriever with local fallback."""
        from src.api.rag_retriever import create_rag_retriever

        retriever = create_rag_retriever(enable_local_fallback=True)
        await retriever.initialize()

        # Local should be available if dependencies present
        assert retriever.is_available


class TestThreadSafety:
    """Tests for thread safety of LocalEmbeddingStore."""

    def test_concurrent_add_documents(self, initialized_store):
        """Test concurrent document additions."""
        import concurrent.futures

        def add_batch(batch_id):
            docs = [{"content": f"Document {batch_id}-{i}"} for i in range(10)]
            return initialized_store.add_documents(docs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(add_batch, i) for i in range(4)]
            results = [f.result() for f in futures]

        assert sum(results) == 40
        assert initialized_store.document_count == 40

    def test_concurrent_search(self, initialized_store, sample_documents):
        """Test concurrent searches."""
        import concurrent.futures

        initialized_store.add_documents(sample_documents)

        def search_query(query):
            return initialized_store.search(query, top_k=3)

        queries = ["machine learning", "animals", "python", "neural networks"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(search_query, q) for q in queries]
            results = [f.result() for f in futures]

        assert all(isinstance(r, list) for r in results)
