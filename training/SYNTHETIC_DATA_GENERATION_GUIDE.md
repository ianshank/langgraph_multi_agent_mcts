# Synthetic Knowledge Generation Guide

Complete guide for generating high-quality training data at scale using LLMs.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Quality Control](#quality-control)
7. [Cost Management](#cost-management)
8. [Integration](#integration)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## Overview

The Synthetic Knowledge Generator creates high-quality Q&A pairs for training multi-agent MCTS systems using LLMs. It supports:

- **Scale**: Generate 1,000-10,000+ Q&A pairs
- **Quality**: Automated validation and scoring (0.0-1.0)
- **Diversity**: 9 categories, 80+ templates, dynamic filling
- **Efficiency**: Async/parallel generation, rate limiting, checkpointing
- **Cost Control**: Token tracking, cost estimation, budget limits
- **Integration**: LangSmith upload, existing dataset merging

### Categories Covered

1. **MCTS Algorithms** - UCB1, PUCT, tree search fundamentals
2. **Exploration/Exploitation** - Multi-armed bandits, regret bounds
3. **AlphaZero & Neural MCTS** - Self-play, policy/value networks
4. **LangGraph Workflows** - State machines, conditional branching
5. **Multi-Agent Coordination** - Agent communication, consensus
6. **Code Implementation** - Python examples, debugging scenarios
7. **System Design** - Architecture, scalability, fault tolerance
8. **Advanced MCTS** - RAVE, progressive widening, virtual loss
9. **Practical Applications** - Production deployment, monitoring

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (choose one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
# Or use local LM Studio (no key needed)
```

### Generate 100 Q&A Pairs

```bash
# Using OpenAI GPT-3.5 (fast and cheap)
python scripts/generate_synthetic_training_data.py \
    --num-samples 100 \
    --model gpt-3.5-turbo \
    --batch-size 10 \
    --min-quality 0.6

# Output:
# ✓ Generated 100 Q&A pairs
# ✓ Saved to training/synthetic_data/
# ✓ Estimated cost: $0.50
```

### Generate 1,000 High-Quality Pairs

```bash
# Using GPT-4 for better quality
python scripts/generate_synthetic_training_data.py \
    --num-samples 1000 \
    --provider openai \
    --model gpt-4-turbo-preview \
    --batch-size 20 \
    --min-quality 0.7 \
    --upload-langsmith

# Estimated cost: ~$20-30
```

### Generate 10,000 Pairs (Production)

```bash
# Large-scale generation with checkpointing
python scripts/generate_synthetic_training_data.py \
    --num-samples 10000 \
    --provider openai \
    --model gpt-4-turbo-preview \
    --batch-size 50 \
    --min-quality 0.6 \
    --max-cost 100.0 \
    --resume \
    --upload-langsmith \
    --log-file training/logs/generation.log

# Features:
# - Automatic checkpointing every 50 pairs
# - Resume if interrupted
# - Stop at $100 budget
# - Upload to LangSmith
# - Full logging
```

## Architecture

### Components

```
SyntheticKnowledgeGenerator
├── LLM Client (OpenAI/Anthropic/LMStudio)
├── Question Templates (80+ templates)
├── Domain Vocabularies (200+ terms)
├── Quality Validator
├── Checkpoint Manager
└── Statistics Tracker
```

### Generation Pipeline

```
1. Template Selection
   ↓
2. Template Filling (with domain vocabulary)
   ↓
3. LLM Generation (question → answer)
   ↓
4. Context Extraction
   ↓
5. Quality Validation
   ↓
6. Quality Scoring
   ↓
7. Deduplication Check
   ↓
8. Save to Dataset
```

### Data Flow

```python
Template: "Explain {algorithm} step by step"
    ↓
Filled: "Explain UCB1 step by step"
    ↓
LLM generates detailed answer (200-2000 tokens)
    ↓
Extract 2-4 contexts from answer
    ↓
Validate (length, format, content)
    ↓
Score quality (0.0-1.0)
    ↓
QAPair(question, answer, contexts, metadata, quality_score)
```

## Configuration

### Basic Configuration (YAML)

```yaml
# training/synthetic_generator_config.yaml

llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  rate_limit_per_minute: 60

generation:
  target_samples: 10000
  batch_size: 20
  min_quality_score: 0.6
  temperature: 0.7
  max_tokens: 2000

  categories:
    - mcts_algorithms
    - exploration_exploitation
    - alphazero_neural
    - langgraph_workflows
    - multi_agent_coordination

  difficulty_distribution:
    easy: 0.3
    medium: 0.5
    hard: 0.2

cost:
  max_total_cost: 100.0
  cost_alert_threshold: 50.0

output:
  directory: "training/synthetic_data"
  formats:
    - langsmith
    - raw
```

### Using Configuration File

```bash
python scripts/generate_synthetic_training_data.py \
    --config training/synthetic_generator_config.yaml \
    --num-samples 5000
```

## Usage Examples

### Example 1: Category-Specific Generation

```bash
# Generate only MCTS and AlphaZero questions
python scripts/generate_synthetic_training_data.py \
    --num-samples 500 \
    --categories mcts_algorithms alphazero_neural advanced_mcts \
    --min-quality 0.7
```

### Example 2: Using Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python scripts/generate_synthetic_training_data.py \
    --provider anthropic \
    --model claude-3-sonnet-20240229 \
    --num-samples 1000 \
    --batch-size 15 \
    --rate-limit 50
```

### Example 3: Local LLM (LM Studio)

```bash
# Start LM Studio server first
# Then:

python scripts/generate_synthetic_training_data.py \
    --provider lmstudio \
    --base-url http://localhost:1234/v1 \
    --num-samples 100 \
    --batch-size 5
```

### Example 4: Python API Usage

```python
import asyncio
from src.adapters.llm import create_client
from training.synthetic_knowledge_generator import SyntheticKnowledgeGenerator

async def generate():
    # Create client
    client = create_client(
        provider="openai",
        model="gpt-4-turbo-preview",
        rate_limit_per_minute=60
    )

    # Create generator
    generator = SyntheticKnowledgeGenerator(
        llm_client=client,
        output_dir="my_output",
        config={
            "min_question_length": 20,
            "min_answer_length": 100
        }
    )

    # Generate
    pairs = await generator.generate_batch(
        num_samples=100,
        categories=["mcts_algorithms", "langgraph_workflows"],
        batch_size=10
    )

    # Filter and save
    high_quality = generator.filter_by_quality(pairs, min_score=0.7)
    generator.save_dataset(high_quality, "output.json", format="langsmith")

    # Get stats
    stats = generator.get_statistics()
    print(f"Generated {stats['valid_pairs']} pairs")
    print(f"Cost: ${stats['total_cost']:.2f}")

asyncio.run(generate())
```

## Quality Control

### Quality Scoring (0.0 - 1.0)

The generator automatically scores each Q&A pair:

```python
Quality Score Breakdown:
├── Answer Length (0.2)
│   ├── >= 500 chars: +0.20
│   ├── >= 300 chars: +0.15
│   └── >= 200 chars: +0.10
├── Examples & Code (0.2)
│   ├── Code blocks: +0.15
│   └── Examples: +0.05
├── Structure (0.2)
│   ├── Numbered lists: +0.07
│   ├── Bullet points: +0.07
│   └── Headings: +0.07
├── Technical Depth (0.2)
│   └── Technical terms: +0.05 each
├── Context Quality (0.2)
│   ├── 3+ contexts: +0.15
│   ├── 2 contexts: +0.10
│   └── 1 context: +0.05
└── Reasoning Paths (0.1 bonus)
    └── Multiple paths: +0.10
```

### Validation Rules

All Q&A pairs must pass:

1. **Length Requirements**
   - Question: >= 20 characters
   - Answer: >= 100 characters

2. **Format Requirements**
   - Question ends with '?'
   - No placeholder text ({}, [TODO])

3. **Content Requirements**
   - At least 1 context provided
   - No answer that just repeats question

4. **Quality Requirements**
   - Quality score >= min_quality threshold

### Filtering Examples

```bash
# Keep only high-quality pairs (>= 0.7)
--min-quality 0.7

# Medium quality acceptable (>= 0.5)
--min-quality 0.5

# Accept all valid pairs (>= 0.0)
--min-quality 0.0
```

## Cost Management

### Cost Estimation

The generator tracks costs in real-time:

```python
Estimated Costs (per 1K tokens):
├── GPT-4-Turbo: $0.03
├── GPT-3.5-Turbo: $0.002
├── Claude-3-Opus: $0.015
├── Claude-3-Sonnet: $0.003
└── Claude-3-Haiku: $0.00025
```

### Expected Costs

```
100 Q&A pairs:
├── GPT-3.5-Turbo: $0.50 - $1.00
├── GPT-4-Turbo: $5.00 - $10.00
└── Claude-3-Sonnet: $2.00 - $4.00

1,000 Q&A pairs:
├── GPT-3.5-Turbo: $5 - $10
├── GPT-4-Turbo: $50 - $100
└── Claude-3-Sonnet: $20 - $40

10,000 Q&A pairs:
├── GPT-3.5-Turbo: $50 - $100
├── GPT-4-Turbo: $500 - $1,000
└── Claude-3-Sonnet: $200 - $400
```

### Budget Control

```bash
# Stop at $100
--max-cost 100.0

# Get alert at $50
python scripts/generate_synthetic_training_data.py \
    --num-samples 10000 \
    --max-cost 100.0 \
    # Will log warning at $50
```

### Cost Optimization Tips

1. **Use GPT-3.5 for bulk generation** (10x cheaper than GPT-4)
2. **Use GPT-4 only for high-quality subset** (--min-quality 0.8)
3. **Generate in batches** (easier to monitor costs)
4. **Use local LLM for testing** (zero cost)

## Integration

### LangSmith Upload

```bash
export LANGSMITH_API_KEY="lsv2_..."

python scripts/generate_synthetic_training_data.py \
    --num-samples 1000 \
    --upload-langsmith \
    --langsmith-dataset "my-training-data" \
    --langsmith-project "multi-agent-mcts"
```

### Merge with Existing Dataset

```bash
# Merge with existing RAG eval dataset
python scripts/generate_synthetic_training_data.py \
    --num-samples 500 \
    --merge-existing
```

### Data Pipeline Integration

```python
from training.data_pipeline import DataOrchestrator
from training.synthetic_knowledge_generator import SyntheticKnowledgeGenerator

# Load existing pipeline
orchestrator = DataOrchestrator("training/config.yaml")

# Generate synthetic data
generator = SyntheticKnowledgeGenerator(...)
synthetic_pairs = await generator.generate_batch(1000)

# Combine datasets
combined = orchestrator.dabstep_loader._splits["train"] + synthetic_pairs
```

## Advanced Features

### Multiple Reasoning Paths

Generate diverse reasoning approaches:

```python
pairs = await generator.generate_batch(
    num_samples=100,
    categories=["mcts_algorithms"]
)

# For high-quality pairs, generate reasoning paths
for pair in pairs:
    if pair.quality_score >= 0.7:
        pair.reasoning_paths = await generator._generate_reasoning_paths(
            pair.question,
            num_paths=3
        )
```

### Custom Templates

Add your own templates:

```python
from training.synthetic_knowledge_generator import QUESTION_TEMPLATES

# Add custom category
QUESTION_TEMPLATES["custom_category"] = [
    "How do you implement {feature} in production?",
    "What are the challenges of {problem}?",
    # ... more templates
]

# Add vocabulary
DOMAIN_VOCABULARIES["feature"] = [
    "distributed MCTS",
    "neural network caching",
    "async tree search"
]
```

### Deduplication

The generator automatically prevents duplicates:

```python
# Uses MD5 hash of normalized question
# Skips if hash already exists

Normalization:
- Convert to lowercase
- Remove extra whitespace
- Compare hash
```

### Checkpointing & Resume

```bash
# First run (interrupted at 500/1000)
python scripts/generate_synthetic_training_data.py \
    --num-samples 1000

# Resume from checkpoint
python scripts/generate_synthetic_training_data.py \
    --num-samples 1000 \
    --resume

# Will load checkpoint and continue from 500
```

### Parallel Generation

```bash
# Increase batch size for faster generation
--batch-size 50

# But respect rate limits
--rate-limit 60  # max 60 req/min
```

## Troubleshooting

### Rate Limit Errors

```
Error: Rate limit exceeded
```

**Solution:**
```bash
# Reduce batch size
--batch-size 5

# Lower rate limit
--rate-limit 30

# The generator will automatically retry
```

### Low Quality Scores

```
Warning: Average quality score: 0.35
```

**Solution:**
```bash
# Use better model
--model gpt-4-turbo-preview

# Lower quality threshold
--min-quality 0.4

# Focus on specific categories
--categories mcts_algorithms advanced_mcts
```

### Cost Exceeded

```
Warning: Cost exceeded maximum: $105 > $100
```

**Solution:**
```bash
# Generation stopped automatically at budget limit
# Review generated data:
ls training/synthetic_data/

# Continue with remaining budget:
--max-cost 50.0
```

### Checkpoint Not Found

```
Warning: Checkpoint not found
```

**Solution:**
- First run creates checkpoint
- Subsequent runs with `--resume` load it
- Check `training/synthetic_data/checkpoint.json` exists

### API Key Errors

```
Error: OPENAI_API_KEY not set
```

**Solution:**
```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Performance Metrics

### Generation Speed

```
Model Performance (on i7 CPU, 60 req/min limit):
├── GPT-3.5-Turbo: 30-40 pairs/minute
├── GPT-4-Turbo: 20-30 pairs/minute
└── Claude-3-Sonnet: 25-35 pairs/minute

Typical Generation Times:
├── 100 pairs: 3-5 minutes
├── 1,000 pairs: 30-50 minutes
└── 10,000 pairs: 5-8 hours
```

### Quality Distribution

```
Expected Quality Distribution:
├── Easy questions: 0.4-0.6
├── Medium questions: 0.5-0.7
└── Hard questions: 0.6-0.8

With GPT-4:
├── Easy: 0.5-0.7
├── Medium: 0.6-0.8
└── Hard: 0.7-0.9
```

## Best Practices

### 1. Start Small

```bash
# Test with 10-100 samples first
python scripts/generate_synthetic_training_data.py \
    --num-samples 10 \
    --model gpt-3.5-turbo
```

### 2. Review Quality

```bash
# Check generated samples
cat training/synthetic_data/*.json | jq '.[0]'

# Adjust quality threshold based on results
```

### 3. Scale Gradually

```bash
# 100 → 1,000 → 10,000
# Check quality at each step
```

### 4. Use Checkpointing

```bash
# Always enable for large runs
--resume
```

### 5. Monitor Costs

```bash
# Set budget limits
--max-cost 100.0

# Use cheaper models for bulk
--model gpt-3.5-turbo
```

### 6. Diversify Categories

```bash
# Don't generate all from one category
--categories mcts_algorithms exploration_exploitation \
    langgraph_workflows multi_agent_coordination
```

## Next Steps

1. **Generate Initial Dataset**
   ```bash
   python scripts/generate_synthetic_training_data.py \
       --num-samples 1000 \
       --min-quality 0.6
   ```

2. **Review Quality**
   ```bash
   # Check examples
   cat training/synthetic_data/*.json | jq '.[0:5]'
   ```

3. **Upload to LangSmith**
   ```bash
   # Enable tracing and evaluation
   --upload-langsmith
   ```

4. **Integrate with Training**
   ```python
   from training.data_pipeline import DataOrchestrator
   # Use synthetic data in training pipeline
   ```

5. **Evaluate Performance**
   ```bash
   # Run experiments with synthetic data
   python scripts/run_experiments.py
   ```

## Support

- **Documentation**: `docs/training/`
- **Examples**: `examples/synthetic_data_generation_example.py`
- **Tests**: `tests/integration/test_synthetic_knowledge_generator.py`
- **Issues**: GitHub Issues

## License

See LICENSE file in project root.
