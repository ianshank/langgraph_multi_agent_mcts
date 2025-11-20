# Research Corpus Builder - Quick Start Guide

Build a searchable database of AI/ML research papers from arXiv in minutes.

## Installation (1 minute)

```bash
# Install dependencies
pip install arxiv>=2.1.0 pinecone-client>=3.0.0 sentence-transformers>=2.2.0

# Set up Pinecone (optional, for indexing)
export PINECONE_API_KEY='your-api-key'
```

## Basic Usage (3 commands)

### 1. Fetch Papers (No Indexing)

```bash
python training/examples/build_arxiv_corpus.py \
    --mode keywords \
    --max-papers 100
```

**Output**:
- Papers cached in `./cache/research_corpus/`
- Processing stats displayed
- Ready for indexing

### 2. Fetch and Index

```bash
python training/examples/build_arxiv_corpus.py \
    --mode keywords \
    --max-papers 200 \
    --index \
    --test-search
```

**Output**:
- Papers fetched and processed
- Vectors indexed in Pinecone
- Test searches executed
- Index ready for queries

### 3. Use Config File

```bash
python training/examples/build_arxiv_corpus.py \
    --config training/config.yaml \
    --index
```

**Output**: Full pipeline using your configuration

## Configuration (2 minutes)

Edit `training/config.yaml`:

```yaml
research_corpus:
  # What to fetch
  categories: ["cs.AI", "cs.LG", "cs.CL"]
  keywords: ["MCTS", "AlphaZero", "reinforcement learning"]

  # Date range
  date_start: "2020-01-01"
  date_end: "2025-12-31"

  # How many
  max_results: 1000
  max_per_keyword: 100

  # Where to store
  cache_dir: "./cache/research_corpus"
  pinecone_namespace: "arxiv_research"
```

## Programmatic Usage (Python)

### Simple Example

```python
from training.research_corpus_builder import ResearchCorpusBuilder

# Build corpus
builder = ResearchCorpusBuilder(config_path="training/config.yaml")

# Iterate through papers
for chunk in builder.build_corpus(mode="keywords", max_papers=50):
    print(f"Paper: {chunk.metadata['title']}")
    print(f"Authors: {chunk.metadata['authors']}")
    print(f"Abstract: {chunk.text[:200]}...")
    print()

# Get stats
stats = builder.get_statistics()
print(f"Processed {stats.total_papers_processed} papers")
print(f"Created {stats.total_chunks_created} chunks")
```

### Full Integration with RAG

```python
from training.research_corpus_builder import ResearchCorpusBuilder, integrate_with_rag_pipeline
from training.rag_builder import VectorIndexBuilder
import yaml

# Load config
with open("training/config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize
corpus_builder = ResearchCorpusBuilder(config=config["research_corpus"])
rag_builder = VectorIndexBuilder(config["rag"])

# Build and index
stats = integrate_with_rag_pipeline(
    corpus_builder=corpus_builder,
    rag_index_builder=rag_builder,
    batch_size=100
)

print(f"Indexed {stats['corpus']['papers_processed']} papers")
```

### Search Indexed Papers

```python
from training.rag_builder import VectorIndexBuilder
import yaml

# Load config
with open("training/config.yaml") as f:
    config = yaml.safe_load(f)

# Connect to index
rag_config = config["rag"]
rag_config["namespace"] = "arxiv_research"
index = VectorIndexBuilder(rag_config)

# Search
results = index.search("Monte Carlo Tree Search in AlphaZero", k=5)

for result in results:
    print(f"\nScore: {result.score:.4f}")
    print(f"Title: {result.metadata['title']}")
    print(f"Authors: {', '.join(result.metadata['authors'])}")
    print(f"arXiv: {result.metadata['arxiv_id']}")
```

## Common Tasks

### Task: Get Papers on Specific Topic

```bash
python training/examples/build_arxiv_corpus.py \
    --keywords "MCTS,Monte Carlo Tree Search" \
    --max-papers 100 \
    --export-metadata mcts_papers.json
```

### Task: Recent Papers Only

```bash
python training/examples/build_arxiv_corpus.py \
    --date-start "2024-01-01" \
    --date-end "2024-12-31" \
    --max-papers 500
```

### Task: Specific Categories

```bash
python training/examples/build_arxiv_corpus.py \
    --categories "cs.AI,cs.LG" \
    --mode categories \
    --max-papers 500
```

### Task: Resume Interrupted Build

Just rerun the same command - cached papers are skipped:

```bash
# First run (interrupted)
python training/examples/build_arxiv_corpus.py --max-papers 1000

# Resume (skips already processed papers)
python training/examples/build_arxiv_corpus.py --max-papers 1000
```

### Task: Clear Cache and Rebuild

```bash
python training/examples/build_arxiv_corpus.py \
    --max-papers 100 \
    --clear-cache
```

## What Gets Created

### Directory Structure

```
cache/research_corpus/
├── papers_metadata.json      # All paper metadata
├── processed_ids.txt         # Processed arXiv IDs
└── processing_stats.json     # Build statistics
```

### Paper Metadata

Each paper includes:
- **Basic**: Title, authors, abstract, arXiv ID
- **Categories**: Primary category + all categories
- **Dates**: Published date, last updated
- **Links**: PDF URL, DOI, journal reference
- **Content**: Extracted keywords, comments

### Document Chunks

Each paper becomes multiple chunks:
1. **Title chunk**: Metadata (title, authors, categories)
2. **Abstract chunks**: Paper abstract (may span multiple)
3. **Comment chunk**: arXiv comments (if available)

All chunks include full metadata for filtering and search.

## Common Issues

### "arxiv library not found"
```bash
pip install arxiv>=2.1.0
```

### "Rate limit exceeded"
Increase delay in config:
```yaml
research_corpus:
  rate_limit_delay: 5.0  # seconds
```

### "PINECONE_API_KEY not set"
```bash
export PINECONE_API_KEY='your-api-key'
```

### "Memory error"
Reduce batch size:
```bash
--batch-size 50
```

## Performance Tips

| Papers | Time (approx) | Best Mode |
|--------|---------------|-----------|
| 50     | 3-5 min       | keywords  |
| 200    | 10-15 min     | keywords  |
| 500    | 25-35 min     | categories|
| 1000   | 50-70 min     | categories|

**Note**: Times depend on arXiv API speed and your network.

## Next Steps

1. **Read full docs**: `training/examples/ARXIV_CORPUS_README.md`
2. **Run tests**: `pytest training/tests/test_research_corpus_builder.py`
3. **Customize config**: Edit `training/config.yaml`
4. **Integrate with training**: Use in your training pipeline

## Example Workflows

### Research Dataset Creation

```bash
# 1. Fetch papers on your topic
python training/examples/build_arxiv_corpus.py \
    --keywords "MCTS,reinforcement learning,tree search" \
    --date-start "2020-01-01" \
    --max-papers 500

# 2. Export metadata for analysis
python training/examples/build_arxiv_corpus.py \
    --export-metadata research_data.json

# 3. Index for RAG
python training/examples/build_arxiv_corpus.py \
    --index \
    --save-index
```

### RAG-Powered Q&A System

```python
from training.rag_builder import VectorIndexBuilder
import yaml

# 1. Build corpus (one-time)
# python build_arxiv_corpus.py --config training/config.yaml --index

# 2. Query system
with open("training/config.yaml") as f:
    config = yaml.safe_load(f)

index = VectorIndexBuilder(config["rag"])

# Ask questions
question = "How does MCTS work in AlphaZero?"
results = index.search(question, k=3)

# Generate answer from results
context = "\n\n".join([r.text for r in results])
print(f"Context from papers:\n{context}")
```

### Incremental Updates

```bash
# Initial: 2020-2023 papers
python build_arxiv_corpus.py \
    --date-start "2020-01-01" \
    --date-end "2023-12-31" \
    --max-papers 1000 \
    --index

# Update: Add 2024 papers
python build_arxiv_corpus.py \
    --date-start "2024-01-01" \
    --date-end "2024-12-31" \
    --max-papers 500 \
    --index
```

## File Reference

| File | Purpose |
|------|---------|
| `training/research_corpus_builder.py` | Main module |
| `training/examples/build_arxiv_corpus.py` | CLI tool |
| `training/examples/ARXIV_CORPUS_README.md` | Full documentation |
| `training/tests/test_research_corpus_builder.py` | Unit tests |
| `training/config.yaml` | Configuration |

## Support

- **Logs**: Check `arxiv_corpus_build.log`
- **Verbose mode**: Add `--verbose` flag
- **Test mode**: Start with `--max-papers 10`

---

**Ready to build your research corpus?**

```bash
python training/examples/build_arxiv_corpus.py --mode keywords --max-papers 100 --verbose
```
