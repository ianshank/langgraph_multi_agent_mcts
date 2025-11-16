"""
RAG (Retrieval-Augmented Generation) Builder Module

Handles vector index construction, document processing, and retrieval optimization
for the PRIMUS-Seed cybersecurity document corpus.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import yaml

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from training.data_pipeline import DocumentChunk, PRIMUSProcessor

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""
    doc_id: str
    chunk_id: int
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class IndexStats:
    """Statistics about the vector index."""
    total_documents: int
    total_chunks: int
    index_size_mb: float
    avg_chunk_length: float
    domains: Dict[str, int]


class ChunkingStrategy:
    """Smart document chunking strategies."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize chunking strategy.

        Args:
            config: RAG configuration
        """
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.strategy = config.get("chunk_strategy", "semantic")

        logger.info(
            f"ChunkingStrategy: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, strategy={self.strategy}"
        )

    def chunk_document(self, text: str, doc_id: str = "") -> List[DocumentChunk]:
        """
        Chunk a document using the configured strategy.

        Args:
            text: Document text
            doc_id: Document identifier

        Returns:
            List of document chunks
        """
        if self.strategy == "semantic":
            return self._semantic_chunking(text, doc_id)
        elif self.strategy == "recursive":
            return self._recursive_chunking(text, doc_id)
        else:  # fixed
            return self._fixed_chunking(text, doc_id)

    def _fixed_chunking(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={"strategy": "fixed", "start": start, "end": end}
                ))
                chunk_id += 1

            start = end - self.chunk_overlap
            if start >= len(text) - 1:
                break

        return chunks

    def _semantic_chunking(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """Semantic chunking based on sentence boundaries."""
        chunks = []
        sentences = self._split_into_sentences(text)
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={"strategy": "semantic", "num_sentences": len(current_chunk)}
                ))
                chunk_id += 1

                # Keep overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_len

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(DocumentChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=chunk_text,
                metadata={"strategy": "semantic", "num_sentences": len(current_chunk)}
            ))

        return chunks

    def _recursive_chunking(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """Recursive chunking with multiple separators."""
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(text, separators, doc_id, 0, [])

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        doc_id: str,
        chunk_id: int,
        results: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Recursively split text with separators."""
        if len(text) <= self.chunk_size:
            results.append(DocumentChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=text.strip(),
                metadata={"strategy": "recursive"}
            ))
            return results

        if not separators:
            # Force split at chunk_size
            results.append(DocumentChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=text[:self.chunk_size].strip(),
                metadata={"strategy": "recursive", "forced": True}
            ))
            return self._recursive_split(
                text[self.chunk_size - self.chunk_overlap:],
                ["\n\n", "\n", ". ", " "],
                doc_id,
                chunk_id + 1,
                results
            )

        separator = separators[0]
        parts = text.split(separator)

        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                current_chunk += part + separator
            else:
                if current_chunk:
                    results.append(DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=len(results),
                        text=current_chunk.strip(),
                        metadata={"strategy": "recursive", "separator": separator}
                    ))
                # Try next separator
                if len(part) > self.chunk_size:
                    self._recursive_split(part, separators[1:], doc_id, len(results), results)
                else:
                    current_chunk = part + separator

        if current_chunk.strip():
            results.append(DocumentChunk(
                doc_id=doc_id,
                chunk_id=len(results),
                text=current_chunk.strip(),
                metadata={"strategy": "recursive", "separator": separator}
            ))

        return results

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class VectorIndexBuilder:
    """Build and manage vector indices for RAG."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vector index builder.

        Args:
            config: RAG configuration
        """
        self.config = config
        self.embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = config.get("embedding_dim", 384)
        self.index_type = config.get("index_type", "faiss")
        self.index_path = Path(config.get("index_path", "./cache/rag_index"))
        self.num_neighbors = config.get("num_neighbors", 10)

        self.index_path.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            logger.warning("SentenceTransformers not available, using random embeddings")
            self.embedding_model = None

        # Initialize index
        self.index = None
        self.chunk_store = []  # Store chunk metadata
        self.bm25_index = None

        logger.info(f"VectorIndexBuilder initialized with {self.embedding_model_name}")

    def build_index(self, documents: Iterator[DocumentChunk]) -> IndexStats:
        """
        Build vector index from document chunks.

        Args:
            documents: Iterator of document chunks

        Returns:
            Index statistics
        """
        logger.info("Building vector index...")

        # Initialize FAISS index
        if HAS_FAISS:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            logger.warning("FAISS not available, using numpy-based index")
            self.embeddings_array = []

        all_texts = []
        chunk_lengths = []
        domain_counts = {}

        for chunk in documents:
            # Generate embedding
            embedding = self._embed_text(chunk.text)

            # Add to index
            if HAS_FAISS:
                self.index.add(embedding.reshape(1, -1).astype(np.float32))
            else:
                self.embeddings_array.append(embedding)

            # Store metadata
            self.chunk_store.append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            })

            # Statistics
            all_texts.append(chunk.text)
            chunk_lengths.append(len(chunk.text))

            domain = chunk.metadata.get("category", "general")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            if len(self.chunk_store) % 1000 == 0:
                logger.info(f"Indexed {len(self.chunk_store)} chunks")

        # Build BM25 index for hybrid search
        if HAS_BM25 and self.config.get("hybrid_search", {}).get("enabled", False):
            logger.info("Building BM25 index for hybrid search")
            tokenized = [text.lower().split() for text in all_texts]
            self.bm25_index = BM25Okapi(tokenized)

        # Calculate stats
        index_size = 0
        if HAS_FAISS:
            # Save and measure size
            temp_path = self.index_path / "temp_index.bin"
            faiss.write_index(self.index, str(temp_path))
            index_size = temp_path.stat().st_size / (1024 * 1024)  # MB
            temp_path.unlink()
        else:
            self.embeddings_array = np.array(self.embeddings_array)
            index_size = self.embeddings_array.nbytes / (1024 * 1024)

        stats = IndexStats(
            total_documents=len(set(c["doc_id"] for c in self.chunk_store)),
            total_chunks=len(self.chunk_store),
            index_size_mb=index_size,
            avg_chunk_length=np.mean(chunk_lengths) if chunk_lengths else 0,
            domains=domain_counts
        )

        logger.info(f"Index built: {stats.total_chunks} chunks, {stats.index_size_mb:.2f} MB")
        return stats

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.embedding_model:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        else:
            # Random embedding for testing
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def search(self, query: str, k: int = None) -> List[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query: Query text
            k: Number of results (default: num_neighbors)

        Returns:
            List of search results
        """
        if k is None:
            k = self.num_neighbors

        query_embedding = self._embed_text(query)

        # Dense search
        if HAS_FAISS and self.index:
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                k
            )
            dense_scores = 1.0 / (1.0 + distances[0])  # Convert distance to similarity
            dense_indices = indices[0]
        else:
            # Numpy fallback
            similarities = np.dot(self.embeddings_array, query_embedding)
            dense_indices = np.argsort(similarities)[-k:][::-1]
            dense_scores = similarities[dense_indices]

        # Hybrid search (if enabled)
        if self.bm25_index and self.config.get("hybrid_search", {}).get("enabled", False):
            bm25_weight = self.config["hybrid_search"]["bm25_weight"]
            dense_weight = self.config["hybrid_search"]["dense_weight"]

            # BM25 scores
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)

            # Normalize and combine
            bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-6)

            combined_scores = {}
            for i, idx in enumerate(dense_indices):
                combined_scores[idx] = dense_weight * dense_scores[i] + bm25_weight * bm25_scores[idx]

            # Re-rank
            sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
            results = []
            for idx in sorted_indices[:k]:
                chunk = self.chunk_store[idx]
                results.append(SearchResult(
                    doc_id=chunk["doc_id"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=combined_scores[idx],
                    metadata=chunk["metadata"]
                ))
        else:
            # Dense-only results
            results = []
            for i, idx in enumerate(dense_indices):
                if idx < len(self.chunk_store):
                    chunk = self.chunk_store[idx]
                    results.append(SearchResult(
                        doc_id=chunk["doc_id"],
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        score=float(dense_scores[i]),
                        metadata=chunk["metadata"]
                    ))

        return results

    def save_index(self, path: Optional[Path] = None) -> None:
        """Save index to disk."""
        if path is None:
            path = self.index_path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if HAS_FAISS and self.index:
            faiss.write_index(self.index, str(path / "faiss.index"))
        elif hasattr(self, "embeddings_array"):
            np.save(path / "embeddings.npy", self.embeddings_array)

        # Save metadata
        with open(path / "chunks.json", 'w') as f:
            json.dump(self.chunk_store, f)

        # Save BM25 corpus for rebuild
        if self.bm25_index:
            corpus = [chunk["text"].lower().split() for chunk in self.chunk_store]
            with open(path / "bm25_corpus.json", 'w') as f:
                json.dump(corpus, f)

        logger.info(f"Index saved to {path}")

    def load_index(self, path: Optional[Path] = None) -> None:
        """Load index from disk."""
        if path is None:
            path = self.index_path

        path = Path(path)

        # Load FAISS index
        if HAS_FAISS and (path / "faiss.index").exists():
            self.index = faiss.read_index(str(path / "faiss.index"))
        elif (path / "embeddings.npy").exists():
            self.embeddings_array = np.load(path / "embeddings.npy")

        # Load metadata
        with open(path / "chunks.json", 'r') as f:
            self.chunk_store = json.load(f)

        # Rebuild BM25 if corpus exists
        if HAS_BM25 and (path / "bm25_corpus.json").exists():
            with open(path / "bm25_corpus.json", 'r') as f:
                corpus = json.load(f)
            self.bm25_index = BM25Okapi(corpus)

        logger.info(f"Index loaded from {path}, {len(self.chunk_store)} chunks")

    def add_documents(self, documents: Iterator[DocumentChunk]) -> int:
        """
        Add new documents to existing index (incremental update).

        Args:
            documents: New document chunks

        Returns:
            Number of documents added
        """
        added = 0
        new_texts = []

        for chunk in documents:
            embedding = self._embed_text(chunk.text)

            if HAS_FAISS and self.index:
                self.index.add(embedding.reshape(1, -1).astype(np.float32))
            elif hasattr(self, "embeddings_array"):
                self.embeddings_array = np.vstack([self.embeddings_array, embedding])

            self.chunk_store.append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            })

            new_texts.append(chunk.text)
            added += 1

        # Update BM25 index
        if self.bm25_index and new_texts:
            # Rebuild BM25 with all texts
            all_texts = [c["text"] for c in self.chunk_store]
            tokenized = [text.lower().split() for text in all_texts]
            self.bm25_index = BM25Okapi(tokenized)

        logger.info(f"Added {added} new documents to index")
        return added


class RetrievalOptimizer:
    """Optimize retrieval quality and performance."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize retrieval optimizer.

        Args:
            config: RAG configuration
        """
        self.config = config

    def evaluate_retrieval(
        self,
        index: VectorIndexBuilder,
        queries: List[str],
        ground_truth: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality.

        Args:
            index: Vector index
            queries: Test queries
            ground_truth: Expected document IDs for each query

        Returns:
            Dictionary of metrics
        """
        precisions = []
        recalls = []
        mrrs = []

        for query, expected_docs in zip(queries, ground_truth):
            results = index.search(query)
            retrieved_docs = [r.doc_id for r in results]

            # Precision@K
            relevant = len(set(retrieved_docs) & set(expected_docs))
            precision = relevant / len(retrieved_docs) if retrieved_docs else 0
            precisions.append(precision)

            # Recall
            recall = relevant / len(expected_docs) if expected_docs else 0
            recalls.append(recall)

            # MRR (Mean Reciprocal Rank)
            mrr = 0
            for i, doc in enumerate(retrieved_docs):
                if doc in expected_docs:
                    mrr = 1.0 / (i + 1)
                    break
            mrrs.append(mrr)

        metrics = {
            "precision_at_k": np.mean(precisions),
            "recall": np.mean(recalls),
            "mrr": np.mean(mrrs),
            "f1_score": 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls) + 1e-6)
        }

        logger.info(f"Retrieval metrics: {metrics}")
        return metrics

    def optimize_chunk_size(
        self,
        documents: List[str],
        test_queries: List[str],
        chunk_sizes: List[int] = [256, 512, 768, 1024]
    ) -> int:
        """
        Find optimal chunk size through experimentation.

        Args:
            documents: Sample documents
            test_queries: Test queries
            chunk_sizes: Chunk sizes to test

        Returns:
            Optimal chunk size
        """
        best_size = 512
        best_score = 0

        for chunk_size in chunk_sizes:
            config = self.config.copy()
            config["chunk_size"] = chunk_size

            chunker = ChunkingStrategy(config)

            # Create test index
            chunks = []
            for i, doc in enumerate(documents[:100]):  # Sample
                chunks.extend(chunker.chunk_document(doc, f"doc_{i}"))

            # Build index
            builder = VectorIndexBuilder(config)
            builder.build_index(iter(chunks))

            # Evaluate
            # Simplified evaluation without ground truth
            avg_results = 0
            for query in test_queries[:10]:
                results = builder.search(query)
                avg_results += len(results)

            score = avg_results / len(test_queries[:10])

            if score > best_score:
                best_score = score
                best_size = chunk_size

            logger.info(f"Chunk size {chunk_size}: score = {score}")

        logger.info(f"Optimal chunk size: {best_size}")
        return best_size

    def query_expansion(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms.

        Args:
            query: Original query

        Returns:
            List of expanded queries
        """
        expansions = [query]

        # Simple expansion using common cybersecurity terms
        cybersecurity_synonyms = {
            "attack": ["threat", "exploit", "breach"],
            "vulnerability": ["weakness", "flaw", "CVE"],
            "malware": ["virus", "trojan", "ransomware"],
            "defense": ["protection", "mitigation", "security"],
        }

        query_lower = query.lower()
        for term, synonyms in cybersecurity_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded = query_lower.replace(term, synonym)
                    expansions.append(expanded)

        return expansions[:5]  # Limit expansions


class RAGIndexManager:
    """Manage multiple domain-specific indices."""

    def __init__(self, config_path: str = "training/config.yaml"):
        """
        Initialize RAG index manager.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.rag_config = self.config["rag"]
        self.indices = {}
        self.chunking_strategy = ChunkingStrategy(self.rag_config)
        self.optimizer = RetrievalOptimizer(self.rag_config)

        logger.info("RAGIndexManager initialized")

    def build_domain_indices(self, processor: PRIMUSProcessor) -> Dict[str, IndexStats]:
        """
        Build separate indices for each domain.

        Args:
            processor: PRIMUS document processor

        Returns:
            Statistics for each domain index
        """
        stats = {}

        for domain_config in self.rag_config.get("domain_indices", []):
            domain_name = domain_config["name"]
            categories = domain_config["categories"]

            logger.info(f"Building index for domain: {domain_name}")

            # Filter documents by category
            def filter_documents():
                for chunk in processor.stream_documents():
                    if categories == ["all"] or chunk.metadata.get("category") in categories:
                        yield chunk

            # Build index
            index_config = self.rag_config.copy()
            index_config["index_path"] = f"./cache/rag_index/{domain_name}"

            builder = VectorIndexBuilder(index_config)
            domain_stats = builder.build_index(filter_documents())

            self.indices[domain_name] = builder
            stats[domain_name] = domain_stats

        return stats

    def search_all_domains(self, query: str, k: int = 10) -> Dict[str, List[SearchResult]]:
        """
        Search across all domain indices.

        Args:
            query: Search query
            k: Number of results per domain

        Returns:
            Results from each domain
        """
        results = {}

        for domain_name, index in self.indices.items():
            domain_results = index.search(query, k=k)
            results[domain_name] = domain_results

        return results

    def route_query(self, query: str) -> str:
        """
        Route query to most appropriate domain index.

        Args:
            query: User query

        Returns:
            Domain name
        """
        query_lower = query.lower()

        # Simple keyword-based routing
        if any(term in query_lower for term in ["mitre", "att&ck", "technique", "tactic"]):
            return "cybersecurity"
        elif any(term in query_lower for term in ["tactical", "operation", "mission"]):
            return "tactical"
        else:
            return "general"

    def save_all_indices(self) -> None:
        """Save all indices to disk."""
        for domain_name, index in self.indices.items():
            index.save_index()
            logger.info(f"Saved {domain_name} index")

    def load_all_indices(self) -> None:
        """Load all indices from disk."""
        for domain_config in self.rag_config.get("domain_indices", []):
            domain_name = domain_config["name"]

            index_config = self.rag_config.copy()
            index_config["index_path"] = f"./cache/rag_index/{domain_name}"

            builder = VectorIndexBuilder(index_config)

            try:
                builder.load_index()
                self.indices[domain_name] = builder
                logger.info(f"Loaded {domain_name} index")
            except Exception as e:
                logger.warning(f"Could not load {domain_name} index: {e}")


if __name__ == "__main__":
    # Test the RAG builder
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing RAG Builder Module")

    # Load config
    config_path = "training/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    rag_config = config["rag"]

    # Test chunking
    chunker = ChunkingStrategy(rag_config)
    test_text = """
    This is a test document about cybersecurity.
    It contains multiple paragraphs discussing various security topics.

    Paragraph 2: The MITRE ATT&CK framework provides a comprehensive matrix of attack techniques.
    Organizations can use this framework to understand adversary behaviors.

    Paragraph 3: Vulnerability management is crucial for maintaining security posture.
    Regular patching and assessment help reduce attack surface.
    """

    chunks = chunker.chunk_document(test_text, "test_doc_1")
    logger.info(f"Created {len(chunks)} chunks from test document")

    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i}: {len(chunk.text)} chars, metadata: {chunk.metadata}")

    # Test vector index
    builder = VectorIndexBuilder(rag_config)

    # Create sample chunks
    sample_chunks = [
        DocumentChunk("doc1", 0, "Cybersecurity threat analysis and defense strategies", {"category": "cyber"}),
        DocumentChunk("doc1", 1, "MITRE ATT&CK framework for understanding adversaries", {"category": "mitre"}),
        DocumentChunk("doc2", 0, "Vulnerability assessment and penetration testing", {"category": "cyber"}),
        DocumentChunk("doc2", 1, "Incident response and forensic investigation", {"category": "tactical"}),
    ]

    stats = builder.build_index(iter(sample_chunks))
    logger.info(f"Index stats: {stats}")

    # Test search
    results = builder.search("cybersecurity defense")
    logger.info(f"Search results for 'cybersecurity defense':")
    for result in results:
        logger.info(f"  - {result.doc_id}/{result.chunk_id}: {result.score:.4f}")

    # Test optimization
    optimizer = RetrievalOptimizer(rag_config)
    expanded = optimizer.query_expansion("malware attack defense")
    logger.info(f"Query expansions: {expanded}")

    logger.info("RAG Builder Module test complete")
