# Multi-Agent MCTS Training Pipeline

A comprehensive training pipeline for multi-agent systems including Hierarchical Reasoning Model (HRM), Task Refinement Model (TRM), Monte Carlo Tree Search (MCTS), and neural meta-controllers.

## Overview

This training pipeline provides:
- **Data Pipeline**: Load and preprocess DABStep, PRIMUS-Seed, and PRIMUS-Instruct datasets
- **Agent Training**: LoRA-based fine-tuning for HRM, TRM, and MCTS neural components
- **RAG Builder**: Vector index construction for cybersecurity document retrieval
- **Meta-Controller**: Neural router and ensemble aggregator training
- **Evaluation Suite**: Comprehensive benchmarking and production validation
- **Continual Learning**: Online learning with drift detection and A/B testing
- **Monitoring**: Real-time training observability and alerting

## Installation

```bash
# Install dependencies
pip install -r training/requirements.txt

# Optional: Install GPU support
pip install faiss-gpu  # Instead of faiss-cpu
```

## Quick Start

### 1. Run Full Training Pipeline

```bash
python -m training.cli train --config training/config.yaml
```

### 2. Run Specific Training Phase

```bash
# Base pre-training
python -m training.cli train --config training/config.yaml --phase base_pretraining

# MCTS self-play
python -m training.cli train --config training/config.yaml --phase mcts_self_play

# Meta-controller training
python -m training.cli train --config training/config.yaml --phase meta_controller_training
```

### 3. Evaluate Trained Models

```bash
python -m training.cli evaluate --model models/hrm_checkpoint.pt --config training/config.yaml --validate-production
```

### 4. Build RAG Index

```bash
python -m training.cli build-rag --output cache/rag_index --config training/config.yaml
```

### 5. Train Meta-Controller

```bash
python -m training.cli meta-controller --config training/config.yaml --epochs 10 --generate-traces --num-traces 10000
```

## Configuration

The main configuration file (`training/config.yaml`) controls all aspects of training:

```yaml
data:
  dabstep:
    path: "adyen/DABstep"
    train_split: 0.8
    val_split: 0.1
    test_split: 0.1

training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 10
  gradient_accumulation_steps: 4
  fp16: true

agents:
  hrm:
    model_name: "microsoft/deberta-v3-base"
    max_decomposition_depth: 5
    lora_rank: 16
  trm:
    model_name: "microsoft/deberta-v3-base"
    max_refinement_iterations: 3
  mcts:
    simulations: 200
    exploration_constant: 1.414

rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  hybrid_search:
    enabled: true
    bm25_weight: 0.3

evaluation:
  success_criteria:
    hrm_accuracy: 0.85
    trm_avg_iterations: 3.0
    rag_precision_at_10: 0.90
    router_accuracy: 0.80
```

## Directory Structure

```
training/
├── __init__.py              # Package exports
├── cli.py                   # Command-line interface
├── config.yaml              # Main configuration
├── requirements.txt         # Dependencies
├── README.md               # This file
│
├── data_pipeline.py        # Dataset loading and preprocessing
├── agent_trainer.py        # Agent training framework
├── rag_builder.py          # RAG index construction
├── meta_controller.py      # Neural router/aggregator training
├── evaluation.py           # Benchmarking and validation
├── orchestrator.py         # Training pipeline coordination
├── continual_learning.py   # Online learning capabilities
├── monitoring.py           # Observability and alerting
├── integrate.py            # Production integration utilities
│
├── tests/                  # Unit and integration tests
│   ├── test_data_pipeline.py
│   ├── test_agent_trainer.py
│   └── ...
│
├── models/                 # Saved model checkpoints
│   ├── checkpoints/
│   └── production/
│
├── notebooks/              # Analysis and experimentation
└── reports/                # Evaluation reports and dashboards
```

## Training Phases

The pipeline runs in sequential phases:

1. **Base Pre-training** (1-2 days)
   - Train HRM on hierarchical decomposition
   - Train TRM on task refinement
   - Build RAG indices from PRIMUS-Seed

2. **Instruction Fine-tuning** (4-8 hours)
   - Fine-tune with PRIMUS-Instruct samples
   - Lower learning rate for refinement

3. **MCTS Self-Play** (2-3 days)
   - Generate self-play experience
   - Train value and policy networks
   - Reinforcement learning loop

4. **Meta-Controller Training** (1 day)
   - Collect execution traces
   - Train neural router
   - Train ensemble aggregator

5. **Evaluation and Validation** (4 hours)
   - Comprehensive benchmarking
   - Production readiness checks
   - Generate reports

## Performance Targets

- **HRM Accuracy**: >85% on DABStep decomposition
- **TRM Iterations**: <3 average refinement steps
- **MCTS Win Rate**: >75% prediction accuracy
- **RAG Precision@10**: >90% retrieval precision
- **Router Accuracy**: >80% agent selection accuracy

## Monitoring

### Training Metrics

- Loss curves per agent
- Gradient norms
- Learning rate schedules
- Resource usage (CPU, GPU, memory)

### Alerting

Automatic alerts for:
- Loss spikes (>2x baseline)
- Gradient explosions (>100 norm)
- OOM warnings (>90% GPU memory)
- Accuracy degradation

### Dashboard

Generate HTML dashboard:
```bash
python -m training.cli monitor --static --config training/config.yaml
```

## Experiment Tracking

Supports multiple platforms:
- **Weights & Biases**: `wandb`
- **MLflow**: `mlflow`
- **Local JSON**: Fallback option

Configure in `config.yaml`:
```yaml
monitoring:
  experiment_tracking:
    platform: "wandb"
    project_name: "multi-agent-mcts-training"
```

## Testing

Run the test suite:

```bash
# All tests
pytest training/tests/ -v

# Specific module
pytest training/tests/test_data_pipeline.py -v

# With coverage
pytest training/tests/ --cov=training --cov-report=html
```

## Continual Learning

After deployment, enable continual learning:

1. **Feedback Collection**: Collect user feedback on predictions
2. **Drift Detection**: Monitor data distribution shifts
3. **Incremental Training**: Update models without catastrophic forgetting
4. **A/B Testing**: Validate improvements before full rollout

## Production Integration

Export trained models:
```bash
python -m training.cli integrate --models-dir training/models/checkpoints --export models/production --production-config
```

Hot-swap models:
```python
from training.integrate import HotSwapper

swapper = HotSwapper()
swapper.prepare_swap("hrm", new_model)
swapper.execute_swap("hrm")
# If issues, rollback
swapper.rollback_swap("hrm")
```

## Advanced Usage

### Custom Training Loop

```python
from training.agent_trainer import HRMTrainer
from training.data_pipeline import DataOrchestrator

# Load data
orchestrator = DataOrchestrator("training/config.yaml")
orchestrator.prepare_data()
dataloader = orchestrator.get_hrm_dataloader("train")

# Train
trainer = HRMTrainer(config)
for epoch in range(10):
    loss = trainer.train_epoch(dataloader)
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

### Custom Evaluation Metrics

```python
from training.evaluation import DABStepBenchmark

benchmark = DABStepBenchmark(config)
report = benchmark.evaluate_model(model, test_data)

# Custom analysis
for difficulty, metrics in report.per_difficulty_metrics.items():
    print(f"{difficulty}: Accuracy = {metrics['accuracy']:.2%}")
```

### RAG Query Testing

```python
from training.rag_builder import VectorIndexBuilder

builder = VectorIndexBuilder(config)
builder.load_index()

results = builder.search("MITRE ATT&CK defense strategies", k=5)
for result in results:
    print(f"Score: {result.score:.4f} - {result.text[:100]}...")
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Enable gradient accumulation
- Use `fp16` training
- Increase gradient accumulation steps

### Slow Training

- Enable `fp16` mixed precision
- Increase number of workers
- Use SSD for data caching
- Consider distributed training

### Poor Convergence

- Adjust learning rate (try 1e-5 to 5e-5)
- Check data quality
- Enable curriculum learning
- Increase warmup ratio

### Dataset Loading Issues

- Check internet connection for HuggingFace
- Verify cache directory permissions
- Monitor disk space for large datasets
- Use streaming mode for PRIMUS-Seed

## Contributing

1. Add tests for new features
2. Follow existing code patterns
3. Update configuration schema
4. Document new modules

## License

See main project LICENSE file.

## References

- [DABStep Dataset](https://huggingface.co/datasets/adyen/DABstep)
- [PRIMUS-Seed](https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed)
- [PRIMUS-Instruct](https://huggingface.co/datasets/trendmicro-ailab/Primus-Instruct)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
