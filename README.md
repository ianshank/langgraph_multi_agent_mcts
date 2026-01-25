# LangGraph Multi-Agent MCTS Framework

**Production-Ready DeepMind-Style AI System with Neural MCTS and Hierarchical Reasoning**

![Architecture](docs/img/architecture_overview.png)

This framework implements a state-of-the-art multi-agent system combining hierarchical reasoning (HRM), iterative refinement (TRM), and Monte Carlo Tree Search (MCTS) guided by neural networks. It features a complete training pipeline, synthetic data generation, and RAG integration.

## 🚀 Key Features

### 🧠 Core Architecture

- **HRM (Hierarchical Reasoning Module)**: DeBERTa-based agent for complex problem decomposition.
- **TRM (Task Refinement Module)**: Iterative agent for refining and optimizing solutions.
- **Neural MCTS**: AlphaZero-style tree search guided by policy/value networks.
- **Meta-Controller**: Neural router (GRU/BERT) that dynamically assigns tasks to the best agent.
- **Chess Domain Expansion**: Pure AlphaZero approach for strategic games (Chess).
- **Continuous Learning Loop**: Iterative self-play and model distillation pipeline.

### 🛠️ Training Pipeline

- **End-to-End Orchestration**: Automated multi-stage training (Pre-training → Fine-tuning → Self-Play).
- **Synthetic Data Generation**: LLM-powered generator for creating high-quality training datasets.
- **Research Corpus Builder**: Automated fetching and indexing of arXiv papers for RAG.
- **Model Registry**: SQLModel-based registry for versioning and metadata management.
- **Docker Support**: Fully containerized training and inference environments.

### 📊 Observability & RAG

- **RAG Integration**: Pinecone vector database for retrieving domain knowledge.
- **Experiment Tracking**: Full integration with Weights & Biases.
- **Production Monitoring**: Prometheus/Grafana metrics for latency, memory, and model performance.
- **Semantic Caching**: Efficient retrieval of previously computed results.

## 📦 Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized workflow)
- NVIDIA GPU (recommended for training)

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ianshank/langgraph_multi_agent_mcts.git
   cd langgraph_multi_agent_mcts
   ```

2. **Set up environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys (OpenAI, Pinecone, W&B)
   ```

3. **Run with Docker (Recommended):**

   ```bash
   # Run the demo pipeline (builds image, generates data, trains models)
   bash scripts/run_docker_training.sh
   ```

## 🏗️ Training Workflow

The framework supports a comprehensive training lifecycle:

1. **Data Generation**:

    ```bash
    # Generate synthetic Q&A pairs
    python -m scripts.generate_synthetic_training_data --num-samples 1000
    ```

2. **Corpus Building**:

    ```bash
    # Fetch and index arXiv papers
    python -m training.examples.build_arxiv_corpus --mode keywords --max-papers 200
    ```

3. **Production Training**:

    ```bash
    # Run full training pipeline
    bash scripts/run_production_training.sh
    ```

## 🧪 Testing

Run the comprehensive test suite to verify system integrity:

```bash
# Run all tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run specific deployed model tests
pytest tests/integration/test_deployed_models.py
```

## 📚 Documentation

- **[Architecture Guide](docs/C4_ARCHITECTURE.md)**: Detailed C4 diagrams of system components.
- **[Training Guide](docs/LOCAL_TRAINING_GUIDE.md)**: How to train models locally or in the cloud.
- **[Synthetic Data](training/SYNTHETIC_DATA_GENERATION_GUIDE.md)**: Guide to generating training data.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and development process.

## 📜 License

MIT License - see the [LICENSE](LICENSE) file for details.
