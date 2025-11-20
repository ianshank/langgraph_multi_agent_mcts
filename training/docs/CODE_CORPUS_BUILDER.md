# Code Corpus Builder - Searchable Code Knowledge Base

## Overview

The Code Corpus Builder is a sophisticated system for ingesting, parsing, and indexing code repositories to create a searchable knowledge base. It extracts functions, classes, and patterns from target repositories and integrates them with the RAG (Retrieval-Augmented Generation) system.

## Features

### 1. Multi-Repository Support
- Clones or fetches code from GitHub repositories
- Configurable priority-based processing
- License compliance checking
- GitHub API integration for metadata

### 2. Advanced Code Parsing
- AST-based Python code analysis
- Extracts:
  - Functions and methods
  - Classes with inheritance
  - Docstrings and documentation
  - Type hints and annotations
  - Import dependencies
  - Paper references (arXiv, DOI)

### 3. Usage Pattern Extraction
- Finds usage examples from `examples/` directories
- Links functions to their test files
- Extracts inline code examples from docstrings
- Context-aware example extraction

### 4. Quality Metrics
- Comprehensive quality scoring:
  - Docstring presence (30%)
  - Usage examples (20%)
  - Type hints (15%)
  - Complexity (15%)
  - Test coverage (10%)
  - Code length (10%)
- Configurable quality threshold filtering
- Deduplication based on content hash

### 5. RAG Integration
- Converts code chunks to searchable documents
- Pinecone vector database integration
- Hybrid search (dense + BM25)
- Namespace-based organization
- Metadata-rich indexing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Code Corpus Builder                        │
└─────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │ Fetcher │      │ Parser  │      │ Filter  │
    │         │      │         │      │         │
    │ • Clone │      │ • AST   │      │ • Quality│
    │ • GitHub│      │ • Extract│      │ • Dedup │
    │ • License│     │ • Examples│     │         │
    └─────────┘      └─────────┘      └─────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                      ┌────▼────┐
                      │  Code   │
                      │ Chunks  │
                      └────┬────┘
                           │
                 ┌─────────┴─────────┐
                 │                   │
            ┌────▼────┐        ┌────▼────┐
            │Document │        │Pinecone │
            │ Chunks  │───────▶│  RAG    │
            └─────────┘        └─────────┘
```

## Target Repositories

### High Priority
1. **deepmind/mctx** - JAX-based MCTS implementation
2. **langchain-ai/langgraph** - Multi-agent state machines
3. **openai/gym** - RL environments
4. **karpathy/nanoGPT** - Minimal GPT implementation

### Medium Priority
5. **facebookresearch/ReAgent** - Production RL systems
6. **google-deepmind/alphatensor** - AlphaTensor research
7. **microsoft/DeepSpeed** - Deep learning optimization

### Low Priority
8. **huggingface/transformers** - NLP models (sampled)

## Usage

### Basic Usage

```python
from training.code_corpus_builder import CodeCorpusBuilder

# Initialize builder
builder = CodeCorpusBuilder("training/config.yaml")

# Build corpus from repositories
chunks = builder.build_corpus(max_repos=4)

# Save to disk
builder.save_corpus()

# Get statistics
stats = builder.get_corpus_statistics()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Avg quality: {stats['avg_quality_score']:.2f}")
```

### RAG Integration

```python
from training.code_corpus_builder import CodeCorpusBuilder
from training.rag_builder import VectorIndexBuilder

# Build code corpus
builder = CodeCorpusBuilder("training/config.yaml")
chunks = builder.build_corpus(max_repos=4)

# Create RAG index
rag_config = {...}  # Load from config
index_builder = VectorIndexBuilder(rag_config)

# Index code chunks
document_chunks = list(builder.stream_document_chunks())
index_stats = index_builder.build_index(iter(document_chunks))

# Search for code
results = index_builder.search("How to implement UCB1 in Python?", k=5)

for result in results:
    print(f"Repo: {result.metadata['repo_name']}")
    print(f"Function: {result.metadata['function_name']}")
    print(f"Score: {result.score:.3f}")
```

### Command Line Usage

```bash
# Build corpus from all high-priority repositories
python training/code_corpus_builder.py --max-repos 4

# Load existing corpus and search
python training/code_corpus_builder.py --load --search "MCTS tree search"

# Build and specify output directory
python training/code_corpus_builder.py --output ./my_corpus --max-repos 2

# View statistics
python training/code_corpus_builder.py --load
```

### Integration Example

```bash
# Run full integration example
python training/examples/code_corpus_integration.py --mode build --max-repos 2

# Search existing corpus
python training/examples/code_corpus_integration.py --mode search

# Load and inspect
python training/examples/code_corpus_integration.py --mode load
```

## Configuration

Add to `training/config.yaml`:

```yaml
code_corpus:
  # Repository settings
  cache_dir: "./cache/code_repos"
  output_dir: "./cache/code_corpus"
  use_github_api: true
  github_token: null  # Set via GITHUB_TOKEN env var
  shallow_clone: true

  # Parsing settings
  min_function_lines: 3
  max_function_lines: 200
  extract_tests: true
  extract_examples: true
  find_tests: true

  # Quality filtering
  min_quality_score: 0.5

  # Pinecone integration
  pinecone:
    namespace: "code-corpus"
    index_name: "multi-agent-mcts-rag"
```

## Environment Variables

```bash
# GitHub API token (optional, but recommended for metadata)
export GITHUB_TOKEN="your_github_token"

# Pinecone API key (required for RAG integration)
export PINECONE_API_KEY="your_pinecone_key"
```

## Code Chunk Structure

Each extracted code chunk contains:

```python
@dataclass
class CodeChunk:
    repo_name: str              # Repository name
    file_path: str              # Relative file path
    function_name: str          # Function/class name
    code: str                   # Source code
    docstring: str              # Documentation
    imports: List[str]          # Import dependencies
    usage_examples: List[str]   # Usage examples
    related_papers: List[str]   # Paper references
    metadata: dict              # Additional metadata
    start_line: int             # Start line number
    end_line: int               # End line number
    complexity_score: float     # Complexity metric
    dependencies: List[str]     # Code dependencies
    test_files: List[str]       # Related test files
```

## Search Queries

The system supports natural language queries:

### Example Queries

```python
# MCTS-related queries
"How to implement UCB1 in Python?"
"MCTS with neural network evaluation"
"Monte Carlo Tree Search backpropagation"

# LangGraph queries
"LangGraph state machine example"
"Multi-agent coordination with LangGraph"
"State transition handling"

# Reinforcement Learning queries
"Deep Q-Network implementation"
"Policy gradient methods"
"RL environment wrapper"

# Training queries
"Distributed training with DeepSpeed"
"LoRA fine-tuning example"
"Mixed precision training"
```

## Quality Metrics

### Quality Score Components

1. **Docstring (30%)** - Presence of comprehensive documentation
2. **Examples (20%)** - Availability of usage examples
3. **Type Hints (15%)** - Type annotations for better clarity
4. **Complexity (15%)** - Appropriate complexity level
5. **Tests (10%)** - Linked test coverage
6. **Length (10%)** - Reasonable code length

### Quality Filtering

```python
# Only index high-quality code
builder.quality_filter.min_quality_score = 0.7  # Strict
builder.quality_filter.min_quality_score = 0.5  # Balanced
builder.quality_filter.min_quality_score = 0.3  # Permissive
```

## Output Files

```
./cache/code_corpus/
├── code_chunks.json          # All extracted code chunks
├── repo_metadata.json        # Repository metadata
└── corpus_statistics.json    # Statistics and metrics

./cache/rag_index/code-corpus/
├── chunks.json              # Document chunks for RAG
├── bm25_corpus.json         # BM25 index data
└── index_config.json        # Index configuration
```

## Performance

### Processing Speed
- **Small repo** (nanoGPT): ~30 seconds
- **Medium repo** (langgraph): ~2 minutes
- **Large repo** (transformers): ~10 minutes (sampled)

### Index Size
- **4 repositories**: ~500-1000 chunks
- **Pinecone vectors**: ~50-100MB
- **Local cache**: ~200-500MB

### Resource Usage
- **CPU**: Light (mostly I/O bound)
- **Memory**: 500MB-2GB depending on repo size
- **Disk**: 1-5GB for cached repositories

## Advanced Features

### Custom Repository Processing

```python
from training.code_corpus_builder import CodeCorpusBuilder

builder = CodeCorpusBuilder("training/config.yaml")

# Define custom repositories
custom_repos = [
    {
        "name": "myorg/myrepo",
        "description": "Custom repo",
        "priority": "high",
        "topics": ["custom", "domain"],
    }
]

# Process custom repositories
chunks = builder.build_corpus(repositories=custom_repos)
```

### Custom Quality Filtering

```python
from training.code_corpus_builder import CodeQualityFilter

# Create custom filter
filter_obj = CodeQualityFilter(config)

# Calculate scores
for chunk in chunks:
    score = filter_obj.calculate_quality_score(chunk)
    if score > 0.8:
        print(f"High quality: {chunk.function_name}")
```

### Example Extraction

```python
from training.code_corpus_builder import ExampleExtractor

extractor = ExampleExtractor(config)

# Extract examples from repository
examples_map = extractor.extract_examples_from_repo(repo_path, chunks)

# Find test files
for chunk in chunks:
    test_files = extractor.find_test_files(repo_path, chunk)
    chunk.test_files = test_files
```

## Testing

```bash
# Run all tests
pytest training/tests/test_code_corpus_builder.py -v

# Run specific test
pytest training/tests/test_code_corpus_builder.py::test_parse_function -v

# Run with coverage
pytest training/tests/test_code_corpus_builder.py --cov=training.code_corpus_builder
```

## Troubleshooting

### Issue: Git Clone Fails
```bash
# Solution: Use GitHub token for private repos
export GITHUB_TOKEN="your_token"

# Or disable git clone and use API only
# In config.yaml:
use_github_api: true
```

### Issue: Pinecone API Errors
```bash
# Solution: Verify API key
export PINECONE_API_KEY="your_key"

# Check Pinecone dashboard for quota limits
```

### Issue: Low Quality Chunks
```python
# Solution: Adjust quality threshold
# In config.yaml:
min_quality_score: 0.3  # Lower threshold
```

### Issue: Too Many Chunks
```python
# Solution: Limit repositories or increase quality filter
max_repos: 2  # Process fewer repos
min_quality_score: 0.7  # Higher threshold
```

## License Compliance

The system automatically checks licenses:

```python
from training.code_corpus_builder import RepositoryFetcher

fetcher = RepositoryFetcher(config)
license_info = fetcher.check_license_compliance(repo_path)

if license_info["compliant"]:
    print(f"Safe to use: {license_info['type']}")
else:
    print(f"Review required: {license_info['type']}")
```

**Supported Licenses:**
- MIT ✅
- Apache-2.0 ✅
- BSD ✅
- GPL ⚠️ (review required)

## Best Practices

1. **Start Small**: Process 2-3 repositories first
2. **Use GitHub Token**: Avoid rate limits
3. **Set Quality Threshold**: Balance quantity vs quality
4. **Enable Caching**: Reuse cloned repositories
5. **Monitor Resources**: Watch disk space
6. **Regular Updates**: Re-clone periodically for updates
7. **Namespace Organization**: Use separate namespaces for different domains

## Future Enhancements

- [ ] Multi-language support (JavaScript, Go, Rust)
- [ ] Incremental updates (delta processing)
- [ ] Semantic code search with CodeBERT
- [ ] Function call graph analysis
- [ ] Automatic API documentation generation
- [ ] Code similarity detection
- [ ] License classification ML model
- [ ] GitHub Stars/Activity weighting

## Contributing

To add new repositories:

1. Update `REPOSITORIES` in `code_corpus_builder.py`
2. Set priority and topics
3. Run `build_corpus()` to process

To extend parsing:

1. Add new extractors to `PythonCodeParser`
2. Update `CodeChunk` metadata
3. Add tests in `test_code_corpus_builder.py`

## References

- [AST Module Documentation](https://docs.python.org/3/library/ast.html)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [GitHub API](https://docs.github.com/en/rest)
- [Code Search Best Practices](https://github.blog/2023-02-06-the-technology-behind-githubs-new-code-search/)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review test files for examples
3. Check logs in `./logs/code_corpus_builder.log`
4. Review configuration in `training/config.yaml`
