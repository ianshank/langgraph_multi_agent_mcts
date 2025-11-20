# Multi-Modal Knowledge Base

A comprehensive multi-modal knowledge base system for the LangGraph Multi-Agent MCTS training pipeline that handles text, images, diagrams, and code from research papers and documentation.

## Features

### 1. **Image & Diagram Processing**
- Extract images from PDFs using PyMuPDF or pdf2image
- Automatic figure caption extraction
- Image classification (architecture diagrams, flowcharts, plots, trees, etc.)
- Size filtering and automatic resizing

### 2. **Vision Model Integration**
- Support for GPT-4V and Claude 3.5 Sonnet
- Automatic image description generation
- Context-aware descriptions using paper metadata
- Image type classification based on content

### 3. **Cross-Modal Embeddings**
- CLIP for unified image-text embedding space
- Text-to-image search
- Image-to-image similarity search
- Cross-modal retrieval

### 4. **Code Block Extraction**
- Markdown code blocks (```language)
- LaTeX listings (\begin{lstlisting})
- Algorithm environments
- Automatic language detection
- Context extraction around code
- Pseudocode support

### 5. **Multi-Modal RAG**
- Unified retrieval across text, images, and code
- Separate Pinecone namespaces for each modality
- Hybrid search combining semantic and keyword matching
- Context-aware response generation

## Installation

### Base Requirements

```bash
# Install main dependencies
pip install -r training/requirements.txt

# Install multimodal dependencies
pip install -r training/requirements_multimodal.txt
```

### System Dependencies

#### For PyMuPDF (Recommended - Faster)
```bash
pip install PyMuPDF
```

#### For pdf2image (Alternative)
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils
pip install pdf2image

# macOS
brew install poppler
pip install pdf2image
```

#### For OCR (Optional)
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr
pip install pytesseract

# macOS
brew install tesseract
pip install pytesseract
```

### Environment Variables

```bash
# Required for vision models
export ANTHROPIC_API_KEY="your-anthropic-key"
# OR
export OPENAI_API_KEY="your-openai-key"

# Required for Pinecone vector storage
export PINECONE_API_KEY="your-pinecone-key"
```

## Quick Start

### 1. Basic Image Extraction

```python
from training.multimodal_knowledge_base import ImageProcessor

config = {
    "min_size": [100, 100],
    "max_size": [4096, 4096],
    "formats": ["png", "jpg"],
    "images_dir": "./cache/images",
}

processor = ImageProcessor(config)
images = processor.extract_images_from_pdf("paper.pdf")

print(f"Extracted {len(images)} images")
for img in images:
    print(f"  {img.width}x{img.height}, caption: {img.caption}")
```

### 2. Generate Image Descriptions

```python
import asyncio
from training.multimodal_knowledge_base import VisionModelAdapter

config = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "temperature": 0.3,
}

adapter = VisionModelAdapter(config)

async def describe_image(image):
    context = "Paper about MCTS and reinforcement learning"
    description = await adapter.generate_image_description(image, context)
    return description

# Use asyncio to run
description = asyncio.run(describe_image(images[0]))
print(description)
```

### 3. Extract Code Blocks

```python
from training.multimodal_knowledge_base import CodeBlockExtractor

config = {
    "min_code_lines": 3,
    "context_window": 200,
}

extractor = CodeBlockExtractor(config)

text = """
Here's an MCTS implementation:

```python
def mcts_search(root, num_simulations):
    for _ in range(num_simulations):
        node = select(root)
        reward = simulate(node)
        backpropagate(node, reward)
    return best_action(root)
```
"""

code_blocks = extractor.extract_code_blocks(text, "doc_id")
for code in code_blocks:
    print(f"Language: {code.language}")
    print(f"Code:\n{code.code}")
```

### 4. Cross-Modal Search

```python
from training.multimodal_knowledge_base import MultiModalEmbedder

config = {
    "clip_model": "openai/clip-vit-large-patch14",
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "use_clip": True,
}

embedder = MultiModalEmbedder(config)

# Embed image
image_embedding = embedder.embed_image(image)

# Embed text (using CLIP for compatibility)
text_embedding = embedder.embed_text("MCTS tree diagram", use_clip=True)

# Calculate similarity
similarity = embedder.cross_modal_similarity(image, "MCTS tree diagram")
print(f"Similarity: {similarity:.4f}")
```

### 5. Full Multi-Modal RAG

```python
import asyncio
from training.multimodal_knowledge_base import MultiModalRAG

async def main():
    # Initialize RAG system
    rag = MultiModalRAG(config_path="training/config.yaml")

    # Process a document
    doc_metadata = {
        "title": "AlphaZero: Mastering Chess and Shogi",
        "abstract": "We introduce AlphaZero...",
        "text": "Full paper text...",
    }

    stats = await rag.process_document("alphazero.pdf", doc_metadata)
    print(f"Processed: {stats}")

    # Query across all modalities
    query = "How does AlphaZero's neural network architecture work?"
    results = await rag.retrieve(query, k=5, modalities=["text", "image", "code"])

    # Generate response
    response = await rag.generate_response(query, results, max_tokens=1024)
    print(f"\nResponse:\n{response}")

asyncio.run(main())
```

## Configuration

Edit `training/config.yaml` to configure the multi-modal system:

```yaml
multimodal:
  # Vision model settings
  vision_model:
    model: "claude-3-5-sonnet-20241022"  # or "gpt-4-vision-preview"
    provider: "anthropic"
    max_tokens: 1024
    temperature: 0.3

  # Image processing
  image_processor:
    min_size: [100, 100]
    max_size: [4096, 4096]
    formats: ["png", "jpg", "jpeg"]
    images_dir: "./cache/images"

  # CLIP embeddings
  embeddings:
    clip_model: "openai/clip-vit-large-patch14"
    text_model: "sentence-transformers/all-MiniLM-L6-v2"
    use_clip: true

  # Code extraction
  code_extractor:
    min_code_lines: 3
    context_window: 200

  # Pinecone namespaces
  text_namespace: "multimodal_text"
  code_namespace: "multimodal_code"
  image_namespace: "images"
```

## Use Cases

### 1. "Show me MCTS tree diagram"

```python
results = await rag.retrieve(
    query="MCTS tree diagram",
    modalities=["image"],
    k=10
)

# Returns images classified as tree visualizations
for result in results["image"]:
    print(f"Score: {result.score:.4f}")
    print(f"Type: {result.content.image_type}")
    print(f"Description: {result.content.description}")
```

### 2. "Architecture of AlphaZero"

```python
results = await rag.retrieve(
    query="AlphaZero architecture",
    modalities=["image", "text"],
    k=10
)

# Returns:
# - Architecture diagrams
# - Text descriptions
# - Related figures
```

### 3. "UCB1 implementation"

```python
results = await rag.retrieve(
    query="UCB1 implementation",
    modalities=["code", "text"],
    k=10
)

# Returns:
# - Code implementations
# - Explanation text
# - Related algorithms
```

### 4. "Performance comparison chart"

```python
results = await rag.retrieve(
    query="performance comparison",
    modalities=["image"],
    k=10
)

# Filter for plot/chart type
plots = [
    r for r in results["image"]
    if r.content.image_type == ImageType.PLOT_CHART
]
```

## Integration with Research Corpus Builder

Combine multi-modal processing with the existing research corpus builder:

```python
from training.research_corpus_builder import ResearchCorpusBuilder
from training.multimodal_knowledge_base import MultiModalRAG
import asyncio

async def process_arxiv_papers():
    # Build research corpus
    corpus_builder = ResearchCorpusBuilder(config_path="training/config.yaml")

    # Initialize multimodal RAG
    multimodal_rag = MultiModalRAG(config_path="training/config.yaml")

    # Process papers
    for paper in corpus_builder.fetch_papers_by_keywords(max_per_keyword=10):
        print(f"Processing: {paper.title}")

        # Download PDF if available
        pdf_path = download_pdf(paper.pdf_url)  # Your download function

        # Extract images and code
        doc_metadata = {
            "title": paper.title,
            "abstract": paper.abstract,
            "arxiv_id": paper.arxiv_id,
        }

        stats = await multimodal_rag.process_document(pdf_path, doc_metadata)
        print(f"  Images: {stats['images_extracted']}")
        print(f"  Code blocks: {stats['code_blocks_extracted']}")

asyncio.run(process_arxiv_papers())
```

## Image Types Supported

The system automatically classifies images into these categories:

- **ARCHITECTURE_DIAGRAM**: System architectures, pipelines, frameworks
- **FLOWCHART**: Process flows, decision trees
- **PLOT_CHART**: Performance plots, accuracy graphs, loss curves
- **TABLE**: Comparison tables, matrices
- **TREE_VISUALIZATION**: MCTS trees, game trees, search trees
- **NEURAL_NETWORK**: Neural network diagrams, layer visualizations
- **ALGORITHM**: Algorithm visualizations, pseudocode diagrams
- **EQUATION**: Mathematical formulas, equations
- **OTHER**: Unclassified images

## Advanced Features

### Custom Image Classification

```python
from training.multimodal_knowledge_base import ImageType

def custom_classifier(description: str, caption: str) -> ImageType:
    text = (description + " " + (caption or "")).lower()

    if "my_custom_pattern" in text:
        return ImageType.CUSTOM

    # Fall back to default
    return adapter.classify_image_type(description, caption)

# Use in processing
for image in images:
    image.image_type = custom_classifier(image.description, image.caption)
```

### Batch Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def batch_process_papers(pdf_paths: list[str]):
    rag = MultiModalRAG(config_path="training/config.yaml")

    async def process_one(pdf_path):
        return await rag.process_document(pdf_path)

    # Process up to 5 papers concurrently
    tasks = [process_one(path) for path in pdf_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results

# Usage
results = asyncio.run(batch_process_papers(pdf_list))
```

### Custom Code Language Detection

```python
from training.multimodal_knowledge_base import CodeBlockExtractor

class CustomCodeExtractor(CodeBlockExtractor):
    def _detect_language(self, code: str) -> str:
        # Add custom language detection
        if "algorithm" in code.lower() and "input:" in code.lower():
            return "pseudocode"

        # Fall back to parent implementation
        return super()._detect_language(code)

extractor = CustomCodeExtractor(config)
```

## Performance Tips

### 1. **Use PyMuPDF for faster extraction**
```bash
pip install PyMuPDF
```

### 2. **Cache embeddings**
```yaml
multimodal:
  processing:
    cache_embeddings: true
    cache_descriptions: true
```

### 3. **Batch image descriptions**
```python
# Process multiple images in parallel
async def batch_describe(images, adapter):
    tasks = [
        adapter.generate_image_description(img, context)
        for img in images
    ]
    descriptions = await asyncio.gather(*tasks)
    return descriptions
```

### 4. **Use appropriate CLIP model**
- `openai/clip-vit-base-patch32`: Fast, 512-dim (Good for most use cases)
- `openai/clip-vit-large-patch14`: Better quality, 768-dim (Recommended)
- Download size: ~350MB (base) to ~1.7GB (large)

### 5. **Optimize Pinecone queries**
```python
# Use metadata filters for faster search
results = visual_index.search_by_text(
    query="neural network",
    k=10,
    filter_type=ImageType.NEURAL_NETWORK  # Filter by type
)
```

## Troubleshooting

### Issue: "PyMuPDF not found"
**Solution**: Install PyMuPDF or pdf2image
```bash
pip install PyMuPDF
# OR
sudo apt-get install poppler-utils && pip install pdf2image
```

### Issue: "CLIP model download fails"
**Solution**:
1. Check internet connection
2. Try smaller model: `openai/clip-vit-base-patch32`
3. Manual download from HuggingFace

### Issue: "Vision model API error"
**Solution**:
1. Check API key is set: `echo $ANTHROPIC_API_KEY`
2. Verify model name in config
3. Check rate limits

### Issue: "Pinecone connection error"
**Solution**:
1. Verify API key: `echo $PINECONE_API_KEY`
2. Check index name exists
3. Verify region/cloud settings

### Issue: "Out of memory with large PDFs"
**Solution**:
```python
# Process in batches
config["image_processor"]["max_size"] = [2048, 2048]  # Smaller max size
config["processing"]["skip_large_images"] = True
```

## Examples

See `training/examples/multimodal_example.py` for complete working examples:

```bash
python training/examples/multimodal_example.py
```

## Architecture

```
MultiModalRAG
├── ImageProcessor (PDF → Images)
│   ├── PyMuPDF backend
│   └── pdf2image backend
├── VisionModelAdapter (Images → Descriptions)
│   ├── Claude 3.5 Sonnet
│   └── GPT-4V
├── MultiModalEmbedder (Text/Images → Vectors)
│   ├── CLIP (cross-modal)
│   └── SentenceTransformers (text-only)
├── CodeBlockExtractor (Text → Code)
│   ├── Markdown parser
│   ├── LaTeX parser
│   └── Language detector
├── VisualIndexBuilder (Images → Pinecone)
│   └── Namespace: "images"
├── Text Index (Text → Pinecone)
│   └── Namespace: "multimodal_text"
└── Code Index (Code → Pinecone)
    └── Namespace: "multimodal_code"
```

## Future Enhancements

- [ ] OCR for text extraction from images
- [ ] Video frame extraction and indexing
- [ ] Audio transcription and embedding
- [ ] 3D model visualization support
- [ ] Interactive diagram parsing
- [ ] Table extraction and structuring
- [ ] Equation recognition and LaTeX conversion
- [ ] Citation graph visualization
- [ ] Multi-language support
- [ ] Real-time collaborative editing

## References

- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Claude 3.5 Sonnet**: [Anthropic Documentation](https://docs.anthropic.com/claude/docs)
- **GPT-4V**: [OpenAI GPT-4 Vision](https://openai.com/research/gpt-4v-system-card)
- **PyMuPDF**: [Documentation](https://pymupdf.readthedocs.io/)
- **Pinecone**: [Documentation](https://docs.pinecone.io/)

## License

This module is part of the LangGraph Multi-Agent MCTS project and follows the same license.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review examples in `training/examples/multimodal_example.py`
3. Check configuration in `training/config.yaml`
4. Open an issue on GitHub
