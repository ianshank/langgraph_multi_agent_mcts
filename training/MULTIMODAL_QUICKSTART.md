# Multi-Modal Knowledge Base - Quick Start Guide

Get started with the multi-modal knowledge base in 5 minutes!

## Prerequisites

1. **Python 3.10+**
2. **API Keys** (at least one):
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."  # For Claude 3.5 Sonnet
   export OPENAI_API_KEY="sk-..."          # For GPT-4V
   export PINECONE_API_KEY="..."           # For vector storage
   ```

## Installation (5 minutes)

### Step 1: Install Dependencies

```bash
# Navigate to training directory
cd training

# Install main requirements
pip install -r requirements.txt

# Install multimodal requirements
pip install -r requirements_multimodal.txt

# Install PDF processing (choose one)
pip install PyMuPDF  # Recommended
# OR
sudo apt-get install poppler-utils && pip install pdf2image  # Alternative
```

### Step 2: Verify Installation

```bash
python -c "
import fitz  # PyMuPDF
from transformers import CLIPModel
from PIL import Image
print('✓ All dependencies installed!')
"
```

## Quick Examples

### Example 1: Extract Images from PDF (2 minutes)

Create `test_extract.py`:

```python
from training.multimodal_knowledge_base import ImageProcessor

# Configure processor
processor = ImageProcessor({
    "min_size": [100, 100],
    "images_dir": "./my_images",
})

# Extract images from a paper
images = processor.extract_images_from_pdf("paper.pdf")

print(f"✓ Extracted {len(images)} images")
for img in images[:3]:
    path = processor.save_image(img)
    print(f"  - {img.width}x{img.height} saved to {path}")
```

Run it:
```bash
python test_extract.py
```

### Example 2: Describe Images with AI (3 minutes)

Create `test_describe.py`:

```python
import asyncio
from training.multimodal_knowledge_base import (
    ImageProcessor,
    VisionModelAdapter
)

async def describe_images():
    # Extract images
    processor = ImageProcessor({"images_dir": "./my_images"})
    images = processor.extract_images_from_pdf("paper.pdf")

    # Setup vision model
    adapter = VisionModelAdapter({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 512,
    })

    # Describe first image
    if images:
        desc = await adapter.generate_image_description(images[0])
        print(f"✓ Description:\n{desc}")

        # Classify type
        img_type = adapter.classify_image_type(desc)
        print(f"✓ Type: {img_type.value}")

asyncio.run(describe_images())
```

Run it:
```bash
python test_describe.py
```

### Example 3: Extract Code from Text (1 minute)

Create `test_code.py`:

```python
from training.multimodal_knowledge_base import CodeBlockExtractor

text = """
Here's MCTS in Python:

```python
def mcts(root, num_sims=1000):
    for _ in range(num_sims):
        node = select(root)
        reward = simulate(node)
        backpropagate(node, reward)
    return best_action(root)
```
"""

extractor = CodeBlockExtractor({"min_code_lines": 3})
codes = extractor.extract_code_blocks(text, "doc_1")

print(f"✓ Extracted {len(codes)} code blocks")
for code in codes:
    print(f"  - Language: {code.language}")
    print(f"  - Lines: {len(code.code.splitlines())}")
```

Run it:
```bash
python test_code.py
```

### Example 4: Full Multi-Modal RAG (5 minutes)

Create `test_rag.py`:

```python
import asyncio
from training.multimodal_knowledge_base import MultiModalRAG
from pathlib import Path

async def full_pipeline():
    # Initialize RAG
    rag = MultiModalRAG(config_path="config.yaml")

    # Process a paper
    pdf_path = "paper.pdf"
    if Path(pdf_path).exists():
        print("Processing paper...")
        stats = await rag.process_document(pdf_path, {
            "title": "My Paper Title",
            "abstract": "Abstract text...",
        })
        print(f"✓ Stats: {stats}")

    # Query across all modalities
    query = "How does the algorithm work?"
    print(f"\nQuerying: {query}")

    results = await rag.retrieve(query, k=3)

    # Show results
    for modality, items in results.items():
        print(f"\n{modality.upper()}: {len(items)} results")
        for i, item in enumerate(items[:2], 1):
            print(f"  {i}. Score: {item.score:.4f}")

    # Generate answer
    print("\nGenerating answer...")
    response = await rag.generate_response(query, results)
    print(f"✓ Answer:\n{response[:300]}...")

asyncio.run(full_pipeline())
```

Run it:
```bash
python test_rag.py
```

## Common Workflows

### Workflow 1: Process ArXiv Papers

```python
import asyncio
from training.research_corpus_builder import ResearchCorpusBuilder
from training.multimodal_knowledge_base import MultiModalRAG

async def process_arxiv():
    # Fetch papers from arXiv
    builder = ResearchCorpusBuilder(config_path="config.yaml")
    papers = list(builder.fetch_papers_by_keywords(max_per_keyword=5))

    # Initialize multimodal RAG
    rag = MultiModalRAG(config_path="config.yaml")

    # Process each paper
    for paper in papers:
        print(f"Processing: {paper.title}")
        # You would download PDF here
        # stats = await rag.process_document(pdf_path, paper.__dict__)
        # print(f"  Images: {stats['images_extracted']}")

asyncio.run(process_arxiv())
```

### Workflow 2: Search for Diagrams

```python
from training.multimodal_knowledge_base import VisualIndexBuilder, ImageType

# Build index
builder = VisualIndexBuilder({
    "index_name": "my-rag",
    "namespace": "images",
})

# Search for specific diagram types
queries = [
    ("MCTS tree", ImageType.TREE_VISUALIZATION),
    ("neural network", ImageType.NEURAL_NETWORK),
    ("performance plot", ImageType.PLOT_CHART),
]

for query, img_type in queries:
    results = builder.search_by_text(query, k=5, filter_type=img_type)
    print(f"{query}: {len(results)} results")
```

### Workflow 3: Build Code Index

```python
from training.multimodal_knowledge_base import (
    CodeBlockExtractor,
    MultiModalRAG
)
from training.rag_builder import VectorIndexBuilder
from training.data_pipeline import DocumentChunk

# Extract code from multiple documents
extractor = CodeBlockExtractor({"min_code_lines": 3})

all_code = []
for doc_path in ["paper1.txt", "paper2.txt"]:
    with open(doc_path) as f:
        text = f.read()
    codes = extractor.extract_code_blocks(text, doc_path)
    all_code.extend(codes)

print(f"Extracted {len(all_code)} code blocks")

# Convert to DocumentChunks and index
chunks = [
    DocumentChunk(
        doc_id=code.code_id,
        chunk_id=0,
        text=f"Language: {code.language}\n\n{code.code}",
        metadata={"type": "code", "language": code.language}
    )
    for code in all_code
]

# Build index (reuse RAG config)
index = VectorIndexBuilder({
    "index_name": "my-rag",
    "namespace": "code",
    # ... other config
})

stats = index.build_index(iter(chunks))
print(f"Indexed {stats.total_chunks} code chunks")
```

## Configuration Quick Reference

Edit `training/config.yaml`:

```yaml
multimodal:
  # Vision model
  vision_model:
    model: "claude-3-5-sonnet-20241022"  # Fast & good quality
    # model: "gpt-4-vision-preview"      # Alternative

  # Image settings
  image_processor:
    min_size: [100, 100]     # Skip tiny images
    max_size: [4096, 4096]   # Resize large images

  # CLIP model
  embeddings:
    clip_model: "openai/clip-vit-large-patch14"  # Best quality
    # clip_model: "openai/clip-vit-base-patch32"  # Faster

  # Pinecone namespaces
  text_namespace: "multimodal_text"
  code_namespace: "multimodal_code"
  image_namespace: "images"
```

## Debugging Tips

### Check if dependencies are installed
```bash
python -c "
try:
    import fitz; print('✓ PyMuPDF')
except: print('✗ PyMuPDF - run: pip install PyMuPDF')

try:
    from transformers import CLIPModel; print('✓ CLIP')
except: print('✗ CLIP - run: pip install transformers')

try:
    from pinecone import Pinecone; print('✓ Pinecone')
except: print('✗ Pinecone - run: pip install pinecone')
"
```

### Check API keys
```bash
echo "Anthropic: ${ANTHROPIC_API_KEY:0:10}..."
echo "OpenAI: ${OPENAI_API_KEY:0:10}..."
echo "Pinecone: ${PINECONE_API_KEY:0:10}..."
```

### Test image extraction
```bash
python -c "
from training.multimodal_knowledge_base import ImageProcessor
p = ImageProcessor({'images_dir': './test'})
print(f'Backend: {p.backend}')
# Should print: Backend: pymupdf or Backend: pdf2image
"
```

### Test vision model connection
```bash
python -c "
import asyncio
from training.multimodal_knowledge_base import VisionModelAdapter

async def test():
    adapter = VisionModelAdapter({'model': 'claude-3-5-sonnet-20241022'})
    print(f'Provider: {adapter.provider}')
    print(f'Client: {adapter._client}')

asyncio.run(test())
"
```

## Next Steps

1. ✅ **Read full documentation**: `training/MULTIMODAL_README.md`
2. ✅ **Run examples**: `python training/examples/multimodal_example.py`
3. ✅ **Integrate with your pipeline**: See integration examples
4. ✅ **Customize**: Extend classes for your specific needs

## Common Issues

| Issue | Solution |
|-------|----------|
| "PyMuPDF not found" | `pip install PyMuPDF` |
| "ANTHROPIC_API_KEY not set" | `export ANTHROPIC_API_KEY="sk-ant-..."` |
| "CLIP model download fails" | Check internet; try smaller model |
| "Out of memory" | Reduce `max_size` in config |
| "No images extracted" | Check PDF has embedded images |

## Performance Benchmarks

On a typical research paper (20 pages, 10 images):

- **Image extraction**: ~2-5 seconds (PyMuPDF) or ~10-20 seconds (pdf2image)
- **Image description**: ~3-5 seconds per image (Claude 3.5 Sonnet)
- **Code extraction**: <1 second
- **Embedding generation**: ~0.1 seconds per item (CLIP)
- **Total processing**: ~1-2 minutes per paper

## Getting Help

- 📖 Full docs: `training/MULTIMODAL_README.md`
- 💡 Examples: `training/examples/multimodal_example.py`
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

---

**Ready to go!** 🚀

Start with Example 1 and work your way through the examples. Each one builds on the previous.
