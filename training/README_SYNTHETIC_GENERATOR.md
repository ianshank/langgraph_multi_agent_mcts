# Synthetic Knowledge Generator - Quick Reference

Generate high-quality Q&A training data at scale using LLMs.

## Installation

```bash
# All dependencies are in main requirements.txt
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Quick Start

### 1. Generate 100 Samples (Test)

```bash
python scripts/generate_synthetic_training_data.py \
    --num-samples 100 \
    --min-quality 0.5
```

**Cost**: ~$0.50 (GPT-3.5) or ~$5 (GPT-4)
**Time**: ~3-5 minutes

### 2. Generate 1,000 Samples (Standard)

```bash
python scripts/generate_synthetic_training_data.py \
    --num-samples 1000 \
    --model gpt-4-turbo-preview \
    --batch-size 20 \
    --min-quality 0.6 \
    --upload-langsmith
```

**Cost**: ~$50 (GPT-4)
**Time**: ~30-50 minutes

### 3. Generate 10,000 Samples (Production)

```bash
python scripts/generate_synthetic_training_data.py \
    --num-samples 10000 \
    --model gpt-4-turbo-preview \
    --batch-size 50 \
    --min-quality 0.6 \
    --max-cost 100.0 \
    --resume \
    --upload-langsmith \
    --log-file training/logs/generation.log
```

**Cost**: ~$500 (GPT-4)
**Time**: ~5-8 hours

## Common Commands

### Using Different Providers

```bash
# OpenAI GPT-4
--provider openai --model gpt-4-turbo-preview

# OpenAI GPT-3.5 (cheaper)
--provider openai --model gpt-3.5-turbo

# Anthropic Claude
--provider anthropic --model claude-3-sonnet-20240229

# Local LM Studio
--provider lmstudio --base-url http://localhost:1234/v1
```

### Category Selection

```bash
# Specific categories
--categories mcts_algorithms alphazero_neural langgraph_workflows

# All categories (default)
# No --categories flag
```

### Quality Control

```bash
# High quality only
--min-quality 0.7

# Accept medium quality
--min-quality 0.5

# All valid pairs
--min-quality 0.0
```

### Cost Management

```bash
# Set budget limit
--max-cost 100.0

# Monitor costs
# Check: training/synthetic_data/generation_stats.json
```

## File Locations

### Generated Files

```
training/synthetic_data/
├── synthetic_qa_langsmith_20241120_143022.json  # LangSmith format
├── synthetic_qa_raw_20241120_143022.json        # Raw format
├── generation_stats.json                         # Statistics
└── checkpoint.json                               # Resume point
```

### Source Files

```
training/
├── synthetic_knowledge_generator.py      # Main generator
├── synthetic_generator_config.yaml       # Configuration
├── SYNTHETIC_DATA_GENERATION_GUIDE.md    # Full guide
└── README_SYNTHETIC_GENERATOR.md         # This file

scripts/
├── generate_synthetic_training_data.py   # CLI script
└── extend_rag_eval_dataset.py           # RAG extension

examples/
└── synthetic_data_generation_example.py  # Usage examples

tests/integration/
└── test_synthetic_knowledge_generator.py # Tests
```

## Output Format

### LangSmith Format

```json
{
  "inputs": {
    "question": "How does UCB1 balance exploration and exploitation?",
    "contexts": [
      "UCB1 uses a formula with two terms...",
      "The exploration term is sqrt(ln(N_parent)/N)..."
    ]
  },
  "outputs": {
    "ground_truth": "UCB1 balances exploration and exploitation through..."
  },
  "metadata": {
    "category": "exploration_exploitation",
    "difficulty": "medium",
    "quality_score": 0.75,
    "model": "gpt-4-turbo-preview"
  }
}
```

## Categories

1. **mcts_algorithms** - UCB1, PUCT, tree search
2. **exploration_exploitation** - Multi-armed bandits
3. **alphazero_neural** - Neural MCTS, self-play
4. **langgraph_workflows** - State machines, workflows
5. **multi_agent_coordination** - Agent communication
6. **code_implementation** - Python implementations
7. **system_design** - Architecture, scaling
8. **advanced_mcts** - RAVE, progressive widening
9. **practical_applications** - Production deployment

## Troubleshooting

### Rate Limits

```bash
# Reduce batch size
--batch-size 5

# Lower rate limit
--rate-limit 30
```

### Low Quality

```bash
# Use better model
--model gpt-4-turbo-preview

# Lower threshold
--min-quality 0.5
```

### Resume Generation

```bash
# Automatically resumes from checkpoint
--resume
```

### Check Progress

```bash
# View stats
cat training/synthetic_data/generation_stats.json | jq

# View samples
cat training/synthetic_data/*.json | jq '.[0:3]'
```

## Python API

```python
import asyncio
from src.adapters.llm import create_client
from training.synthetic_knowledge_generator import SyntheticKnowledgeGenerator

async def generate():
    client = create_client("openai", model="gpt-4-turbo-preview")

    generator = SyntheticKnowledgeGenerator(
        llm_client=client,
        output_dir="my_data"
    )

    pairs = await generator.generate_batch(
        num_samples=100,
        batch_size=10
    )

    generator.save_dataset(pairs, "output.json", format="langsmith")

    return generator.get_statistics()

stats = asyncio.run(generate())
print(f"Cost: ${stats['total_cost']:.2f}")
```

## Integration

### Upload to LangSmith

```bash
export LANGSMITH_API_KEY="lsv2_..."

python scripts/generate_synthetic_training_data.py \
    --num-samples 1000 \
    --upload-langsmith
```

### Extend RAG Dataset

```bash
python scripts/extend_rag_eval_dataset.py \
    --num-samples 500 \
    --upload-langsmith
```

### Use in Training

```python
from training.data_pipeline import DataOrchestrator

orchestrator = DataOrchestrator("training/config.yaml")
# Load synthetic data and combine with pipeline
```

## Testing

```bash
# Run integration tests
pytest tests/integration/test_synthetic_knowledge_generator.py -v

# Run examples
python examples/synthetic_data_generation_example.py
```

## Performance

### Generation Speed

- **GPT-3.5**: 30-40 pairs/min
- **GPT-4**: 20-30 pairs/min
- **Claude-3-Sonnet**: 25-35 pairs/min

### Costs (per 1,000 pairs)

- **GPT-3.5**: $5-10
- **GPT-4**: $50-100
- **Claude-3-Sonnet**: $20-40

## Best Practices

1. **Start small** - Test with 10-100 samples
2. **Review quality** - Check generated samples
3. **Use checkpointing** - Always `--resume` for large runs
4. **Set budgets** - Use `--max-cost` to control spending
5. **Monitor stats** - Check `generation_stats.json`
6. **Upload to LangSmith** - Enable evaluation tracking

## Support

- **Full Guide**: [SYNTHETIC_DATA_GENERATION_GUIDE.md](SYNTHETIC_DATA_GENERATION_GUIDE.md)
- **Examples**: `examples/synthetic_data_generation_example.py`
- **Tests**: `tests/integration/test_synthetic_knowledge_generator.py`
- **Config**: `training/synthetic_generator_config.yaml`

## Next Steps

1. Generate test batch: `--num-samples 10`
2. Review quality in output files
3. Scale to production: `--num-samples 10000`
4. Upload to LangSmith: `--upload-langsmith`
5. Integrate with training pipeline

---

**Quick Example**:

```bash
# Generate 100 samples, upload to LangSmith
python scripts/generate_synthetic_training_data.py \
    --num-samples 100 \
    --min-quality 0.6 \
    --upload-langsmith

# Check output
cat training/synthetic_data/*.json | jq '.[0]'
```
