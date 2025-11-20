# arXiv Research Corpus Builder

A comprehensive system for ingesting AI/ML papers from arXiv.org and integrating them with the RAG pipeline for multi-agent MCTS training.

## Overview

The Research Corpus Builder fetches academic papers from arXiv, processes them into structured document chunks, and optionally indexes them in Pinecone for retrieval-augmented generation (RAG).

### Key Features

- **Smart Paper Fetching**: Query by categories, keywords, or date ranges
- **Efficient Processing**: Stream-based processing for memory efficiency
- **Section-Aware Chunking**: Intelligent text splitting with overlap
- **Deduplication**: Automatic deduplication based on arXiv IDs
- **Caching & Resumability**: Local caching for interrupted builds
- **Rate Limiting**: Respects arXiv API guidelines
- **Error Handling**: Robust error recovery and retry logic
- **Pinecone Integration**: Seamless integration with vector storage
- **Rich Metadata**: Extracts authors, categories, citations, keywords

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `arxiv>=2.1.0` - arXiv API client
- `pinecone-client>=3.0.0` - Vector database
- `sentence-transformers>=2.2.0` - Embeddings
- `rank-bm25>=0.2.2` - Hybrid search

### 2. Set Up Pinecone (Optional)

If you want to index papers in Pinecone:

```bash
export PINECONE_API_KEY='your-pinecone-api-key'
```

Get your API key from: https://www.pinecone.io/

## Configuration

### Option 1: Using config.yaml

The corpus builder integrates with `training/config.yaml`. Add or modify the `research_corpus` section:

```yaml
research_corpus:
  # Paper fetching
  categories:
    - "cs.AI"      # Artificial Intelligence
    - "cs.LG"      # Machine Learning
    - "cs.CL"      # NLP
    - "cs.NE"      # Neural Computing

  keywords:
    - "MCTS"
    - "AlphaZero"
    - "reinforcement learning"
    - "multi-agent"
    - "chain-of-thought"
    - "RLHF"
    - "DPO"

  # Date range
  date_start: "2020-01-01"
  date_end: "2025-12-31"

  # Limits
  max_results: 1000
  max_per_keyword: 100
  max_per_category: 500

  # Processing
  chunk_size: 512
  chunk_overlap: 50
  cache_dir: "./cache/research_corpus"

  # Pinecone integration
  pinecone_namespace: "arxiv_research"
  batch_size: 100
```

### Option 2: Programmatic Configuration

```python
from training.research_corpus_builder import ResearchCorpusBuilder

config = {
    "categories": ["cs.AI", "cs.LG"],
    "keywords": ["MCTS", "reinforcement learning"],
    "date_start": "2023-01-01",
    "date_end": "2024-12-31",
    "max_results": 500,
    "cache_dir": "./cache/research_corpus",
    "chunk_size": 512,
    "chunk_overlap": 50,
}

builder = ResearchCorpusBuilder(config=config)
```

## Usage

### Quick Start

#### 1. Build Corpus (No Indexing)

Fetch and process papers without indexing:

```bash
python training/examples/build_arxiv_corpus.py --mode keywords --max-papers 100
```

#### 2. Build and Index in Pinecone

```bash
export PINECONE_API_KEY='your-api-key'
python training/examples/build_arxiv_corpus.py \
    --mode keywords \
    --max-papers 200 \
    --index \
    --test-search
```

#### 3. Use Configuration File

```bash
python training/examples/build_arxiv_corpus.py \
    --config training/config.yaml \
    --index \
    --save-index
```

### Command-Line Options

```
Options:
  --mode {keywords,categories,all}
                        Fetching mode (default: keywords)
  --index               Index papers in Pinecone
  --config PATH         Path to YAML config file
  --categories CATS     Comma-separated categories (e.g., 'cs.AI,cs.LG')
  --keywords WORDS      Comma-separated keywords
  --date-start DATE     Start date (YYYY-MM-DD)
  --date-end DATE       End date (YYYY-MM-DD)
  --max-papers N        Maximum papers to process
  --cache-dir PATH      Cache directory
  --no-cache            Disable cache
  --clear-cache         Clear cache before starting
  --batch-size N        Batch size for indexing (default: 100)
  --save-index          Save index metadata
  --test-search         Run test searches after indexing
  --export-metadata PATH Export paper metadata to JSON
  --verbose, -v         Enable verbose logging
```

### Programmatic Usage

#### Basic Corpus Building

```python
from training.research_corpus_builder import ResearchCorpusBuilder

# Initialize builder
builder = ResearchCorpusBuilder(config_path="training/config.yaml")

# Build corpus and iterate through chunks
for chunk in builder.build_corpus(mode="keywords", max_papers=100):
    print(f"Doc: {chunk.doc_id}, Section: {chunk.metadata['section']}")
    print(f"Text: {chunk.text[:100]}...")

# Get statistics
stats = builder.get_statistics()
print(f"Papers processed: {stats.total_papers_processed}")
print(f"Chunks created: {stats.total_chunks_created}")
```

#### Integration with RAG Pipeline

```python
from training.research_corpus_builder import ResearchCorpusBuilder, integrate_with_rag_pipeline
from training.rag_builder import VectorIndexBuilder
import yaml

# Load configuration
with open("training/config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize builders
corpus_builder = ResearchCorpusBuilder(config=config["research_corpus"])
rag_builder = VectorIndexBuilder(config["rag"])

# Build and index
stats = integrate_with_rag_pipeline(
    corpus_builder=corpus_builder,
    rag_index_builder=rag_builder,
    batch_size=100
)

print(f"Indexed {stats['corpus']['papers_processed']} papers")
print(f"Created {stats['index']['total_chunks']} vectors")
```

#### Search Indexed Papers

```python
from training.rag_builder import VectorIndexBuilder
import yaml

# Load RAG configuration
with open("training/config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize index builder
rag_config = config["rag"]
rag_config["namespace"] = "arxiv_research"  # Use arXiv namespace
index = VectorIndexBuilder(rag_config)

# Search
results = index.search("Monte Carlo Tree Search in AlphaZero", k=5)

for i, result in enumerate(results, 1):
    print(f"\nResult {i} (score: {result.score:.4f}):")
    print(f"  Title: {result.metadata['title']}")
    print(f"  Authors: {', '.join(result.metadata['authors'])}")
    print(f"  arXiv ID: {result.metadata['arxiv_id']}")
    print(f"  Categories: {', '.join(result.metadata['categories'])}")
    print(f"  Published: {result.metadata['published_date']}")
    print(f"  Text: {result.text[:200]}...")
```

## Fetching Modes

### 1. Keywords Mode (Recommended)

Fetches papers matching specific keywords across all categories:

```bash
python build_arxiv_corpus.py --mode keywords --max-papers 200
```

- Searches for papers containing configured keywords
- Automatically deduplicates across keywords
- Best for targeted research topics

### 2. Categories Mode

Fetches recent papers from specific arXiv categories:

```bash
python build_arxiv_corpus.py --mode categories --max-papers 500
```

- Fetches papers from each category
- Good for broad coverage
- Higher paper count

### 3. All Mode

Fetches all papers matching category filters:

```bash
python build_arxiv_corpus.py --mode all --max-papers 1000
```

- Most comprehensive
- No keyword filtering
- Useful for complete category coverage

## arXiv Categories

Common AI/ML categories:

- **cs.AI** - Artificial Intelligence
- **cs.LG** - Machine Learning
- **cs.CL** - Computation and Language (NLP)
- **cs.NE** - Neural and Evolutionary Computing
- **cs.CV** - Computer Vision
- **cs.RO** - Robotics
- **stat.ML** - Machine Learning (Statistics)

## Document Structure

Each paper is processed into multiple chunks:

### Chunk Types

1. **Title Chunk** (chunk_id=0)
   - Paper title, authors, categories, publication date
   - Section: "title"
   - Type: "metadata"

2. **Abstract Chunks** (chunk_id=1+)
   - Paper abstract (may span multiple chunks)
   - Section: "abstract"
   - Type: "abstract"

3. **Comment Chunk** (optional)
   - Paper comments/notes from arXiv
   - Section: "comments"
   - Type: "metadata"

### Metadata Fields

Every chunk includes rich metadata:

```python
{
    "source": "arxiv",
    "arxiv_id": "2301.12345",
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2", ...],
    "primary_category": "cs.AI",
    "categories": ["cs.AI", "cs.LG"],
    "published_date": "2023-01-15T10:30:00",
    "updated_date": "2023-01-20T14:45:00",
    "pdf_url": "https://arxiv.org/pdf/2301.12345",
    "keywords": ["MCTS", "reinforcement learning"],
    "doi": "10.1234/example",
    "journal_ref": "Conference 2023",
    "section": "abstract",
    "chunk_type": "abstract"
}
```

## Caching & Resumability

The builder automatically caches processed papers for resumability:

### Cache Structure

```
cache/research_corpus/
├── papers_metadata.json      # Paper metadata cache
├── processed_ids.txt         # List of processed arXiv IDs
└── processing_stats.json     # Build statistics
```

### Resume Interrupted Build

Simply rerun the same command - already processed papers will be skipped:

```bash
# First run (interrupted)
python build_arxiv_corpus.py --mode keywords --max-papers 1000

# Resume from where it left off
python build_arxiv_corpus.py --mode keywords --max-papers 1000
```

### Clear Cache

```bash
# Clear cache and rebuild from scratch
python build_arxiv_corpus.py --mode keywords --max-papers 100 --clear-cache
```

## Rate Limiting

The builder respects arXiv API guidelines:

- **Default delay**: 3 seconds between requests
- **Retries**: 3 automatic retry attempts on failure
- **Configurable**: Adjust `rate_limit_delay` in config

From arXiv API docs:
> Please limit requests to no more than 1 request every 3 seconds

## Performance Tips

### 1. Optimize Fetching Strategy

- **Keywords mode**: Best for targeted topics (faster, fewer papers)
- **Categories mode**: Good balance of speed and coverage
- **All mode**: Most comprehensive but slowest

### 2. Batch Size

Adjust batch size for Pinecone indexing based on available memory:

```bash
--batch-size 50   # For limited memory
--batch-size 100  # Default, good balance
--batch-size 200  # For high-memory systems
```

### 3. Parallel Processing

The builder uses streaming for memory efficiency, but you can run multiple instances with different configurations:

```bash
# Terminal 1: Process cs.AI papers
python build_arxiv_corpus.py --categories "cs.AI" --max-papers 500

# Terminal 2: Process cs.LG papers
python build_arxiv_corpus.py --categories "cs.LG" --max-papers 500
```

### 4. Date Range Optimization

Narrow date ranges fetch faster:

```bash
# Last year only
python build_arxiv_corpus.py \
    --date-start "2024-01-01" \
    --date-end "2024-12-31" \
    --max-papers 500
```

## Example Workflows

### Workflow 1: Build Research Dataset

```bash
# Build a focused dataset on MCTS and RL
python training/examples/build_arxiv_corpus.py \
    --mode keywords \
    --keywords "MCTS,Monte Carlo Tree Search,AlphaZero,MuZero" \
    --date-start "2020-01-01" \
    --max-papers 500 \
    --export-metadata research_papers.json \
    --verbose
```

### Workflow 2: Full RAG Pipeline Integration

```bash
# Set up Pinecone
export PINECONE_API_KEY='your-api-key'

# Build corpus and index in one step
python training/examples/build_arxiv_corpus.py \
    --config training/config.yaml \
    --index \
    --save-index \
    --test-search \
    --verbose
```

### Workflow 3: Incremental Updates

```bash
# Initial build (e.g., 2020-2023)
python build_arxiv_corpus.py \
    --date-start "2020-01-01" \
    --date-end "2023-12-31" \
    --max-papers 1000 \
    --index

# Later: Add recent papers (2024)
python build_arxiv_corpus.py \
    --date-start "2024-01-01" \
    --date-end "2024-12-31" \
    --max-papers 500 \
    --index
```

### Workflow 4: Export for Analysis

```bash
# Build corpus and export metadata for analysis
python build_arxiv_corpus.py \
    --mode categories \
    --categories "cs.AI,cs.LG,cs.CL" \
    --max-papers 1000 \
    --export-metadata analysis/papers_metadata.json

# Then analyze with pandas, etc.
```

## Troubleshooting

### Issue: "arxiv library not found"

```bash
pip install arxiv>=2.1.0
```

### Issue: "Rate limit exceeded"

Increase the delay in config:

```yaml
research_corpus:
  rate_limit_delay: 5.0  # Increase from 3.0 to 5.0 seconds
```

### Issue: "PINECONE_API_KEY not set"

```bash
export PINECONE_API_KEY='your-api-key'
```

Or add to your `.bashrc`/`.zshrc`:

```bash
echo 'export PINECONE_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### Issue: "Memory error during indexing"

Reduce batch size:

```bash
python build_arxiv_corpus.py --index --batch-size 50
```

### Issue: "Connection timeout"

The arXiv API may be slow or unavailable. The builder will automatically retry. You can also increase retry attempts:

```yaml
research_corpus:
  retry_attempts: 5  # Increase from 3 to 5
```

## Best Practices

### 1. Start Small

Test with a small number of papers first:

```bash
python build_arxiv_corpus.py --mode keywords --max-papers 10 --verbose
```

### 2. Use Keywords Mode

For targeted research, keywords mode is most efficient:

```bash
python build_arxiv_corpus.py --mode keywords --max-papers 200
```

### 3. Monitor Progress

Use `--verbose` flag and check logs:

```bash
python build_arxiv_corpus.py --mode keywords --max-papers 500 --verbose
```

Logs are saved to `arxiv_corpus_build.log`.

### 4. Regular Backups

Backup your cache directory periodically:

```bash
tar -czf research_corpus_backup_$(date +%Y%m%d).tar.gz cache/research_corpus/
```

### 5. Namespace Organization

Use different Pinecone namespaces for different paper collections:

```yaml
research_corpus:
  pinecone_namespace: "arxiv_mcts"      # MCTS papers
  # pinecone_namespace: "arxiv_llm"     # LLM papers
  # pinecone_namespace: "arxiv_rl"      # RL papers
```

## Integration with Training Pipeline

The research corpus integrates seamlessly with the training pipeline:

```python
from training.data_pipeline import DataOrchestrator
from training.rag_builder import VectorIndexBuilder
from training.research_corpus_builder import ResearchCorpusBuilder

# Build research corpus
corpus_builder = ResearchCorpusBuilder(config_path="training/config.yaml")
research_chunks = corpus_builder.build_corpus(mode="keywords", max_papers=500)

# Add to RAG index
orchestrator = DataOrchestrator("training/config.yaml")
rag_documents = orchestrator.get_rag_documents()

# Combine research corpus with existing documents
all_documents = itertools.chain(rag_documents, research_chunks)

# Build unified index
rag_config = config["rag"]
index = VectorIndexBuilder(rag_config)
index.build_index(all_documents)
```

## License & Attribution

This tool uses the [arXiv API](https://arxiv.org/help/api/) to fetch paper metadata.

**Important**: When using arXiv papers:
- Cite papers appropriately
- Respect arXiv's [terms of use](https://arxiv.org/help/api/tou)
- Follow the [arXiv API guidelines](https://arxiv.org/help/api/user-manual)

## Support & Contributing

For issues, questions, or contributions:
1. Check the logs: `arxiv_corpus_build.log`
2. Enable verbose mode: `--verbose`
3. Review the configuration: `training/config.yaml`

## Additional Resources

- [arXiv API Documentation](https://arxiv.org/help/api/user-manual)
- [arXiv Category Taxonomy](https://arxiv.org/category_taxonomy)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [LangGraph Multi-Agent MCTS](https://github.com/ianshank/langgraph_multi_agent_mcts)
