---
title: LangGraph Multi-Agent MCTS Demo
emoji: 🌳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - multi-agent
  - mcts
  - reasoning
  - langgraph
  - ai-agents
  - wandb
  - experiment-tracking
short_description: Multi-agent reasoning framework with Monte Carlo Tree Search
---

# LangGraph Multi-Agent MCTS Framework

**Production Demo with Trained Neural Models** - Experience real trained meta-controllers for intelligent agent routing

## What This Demo Shows

This interactive demo showcases trained neural meta-controllers that dynamically route queries to specialized agents:

### 🤖 Trained Meta-Controllers

1. **RNN Meta-Controller**
   - GRU-based recurrent neural network
   - Learns sequential patterns in agent performance
   - Fast inference (~2ms latency)
   - Trained on 1000+ synthetic routing examples

2. **BERT Meta-Controller with LoRA**
   - Transformer-based text understanding
   - Parameter-efficient fine-tuning with LoRA adapters
   - Context-aware routing decisions
   - Better generalization to unseen query patterns

### 🧠 Three Specialized Agents

1. **HRM (Hierarchical Reasoning Module)**
   - Best for: Complex decomposition, multi-level problems
   - Technique: Hierarchical planning with adaptive computation

2. **TRM (Tree Reasoning Module)**
   - Best for: Iterative refinement, comparison tasks
   - Technique: Recursive refinement with convergence detection

3. **MCTS (Monte Carlo Tree Search)**
   - Best for: Optimization, strategic planning
   - Technique: UCB1 exploration with value backpropagation

### 📊 Key Features

- **Real Trained Models**: Production-ready neural meta-controllers
- **Intelligent Routing**: Models learn optimal agent selection patterns
- **Routing Visualization**: See confidence scores and probability distributions
- **Feature Engineering**: Demonstrates query → features → routing pipeline
- **Performance Metrics**: Track execution time and routing accuracy

## How to Use

1. **Enter a Query**: Type your question or select an example
2. **Select Controller**: Choose RNN (fast) or BERT (context-aware)
3. **Process Query**: Click "🚀 Process Query"
4. **Review Results**:
   - See which agent the controller selected
   - View routing confidence and probabilities
   - Examine features used for decision-making
   - Check agent execution details

## Weights & Biases Integration

Track your experiments with **Weights & Biases** for:
- 📈 **Metrics Dashboard**: Visualize consensus scores, execution times, agent performance
- 🔄 **Run Comparison**: Compare different configurations side-by-side
- 📊 **Experiment History**: Track all your queries and results
- 🌳 **MCTS Visualization**: Log tree exploration patterns

### Setting Up W&B

1. **Get API Key**: Sign up at [wandb.ai](https://wandb.ai) and get your API key
2. **Configure Space Secret** (if deploying your own):
   - Go to Space Settings → Repository secrets
   - Add: `WANDB_API_KEY` = your API key
3. **Enable in UI**:
   - Expand "Weights & Biases Tracking" accordion
   - Check "Enable W&B Tracking"
   - Set project name (optional)
   - Set run name (optional, auto-generated if empty)
4. **View Results**: After processing, click the W&B run URL to see your dashboard

### Logged Metrics

- **Per Agent**: Confidence, execution time, response length, reasoning steps
- **MCTS**: Best value, visits, tree depth, top actions with UCB1 scores
- **Consensus**: Score, level (high/medium/low), number of agents
- **Performance**: Total processing time
- **Artifacts**: Full JSON results, tree visualizations

## Example Queries

- "What are the key factors to consider when choosing between microservices and monolithic architecture?"
- "How can we optimize a Python application that processes 10GB of log files daily?"
- "Should we use SQL or NoSQL database for a social media application with 1M users?"
- "How to design a fault-tolerant message queue system?"

## Technical Details

### Architecture

```
Query Input
    │
    ├─→ HRM Agent (Hierarchical Decomposition)
    │      ├─ Component Analysis
    │      └─ Structured Synthesis
    │
    ├─→ TRM Agent (Iterative Refinement)
    │      ├─ Initial Response
    │      ├─ Clarity Enhancement
    │      └─ Validation Check
    │
    └─→ MCTS Engine (Strategic Search)
           ├─ Selection (UCB1)
           ├─ Expansion
           ├─ Simulation
           └─ Backpropagation
                    │
                    ▼
           Consensus Scoring
                    │
                    ▼
           Final Synthesized Response
```

### MCTS Algorithm

The Monte Carlo Tree Search implementation uses:

- **UCB1 Selection**: `Q(s,a) + C * sqrt(ln(N(s)) / N(s,a))`
- **Progressive Widening**: Controls branching factor
- **Domain-Aware Actions**: Contextual decision options
- **Value Backpropagation**: Updates entire path statistics

### Consensus Calculation

```
consensus = average_confidence * agreement_factor
agreement_factor = max(0, 1 - std_deviation * 2)
```

High consensus (>70%) indicates agents agree on approach.
Low consensus (<40%) suggests uncertainty or conflicting strategies.

## Demo Scope

This demonstration focuses on **meta-controller training and routing**:

- ✅ **Real Trained Models**: Production RNN and BERT controllers
- ✅ **Actual Model Loading**: PyTorch and HuggingFace Transformers
- ✅ **Feature Engineering**: Query analysis → feature vectors
- ✅ **Routing Visualization**: See controller decision-making
- ⚠️ **Simplified Agents**: Agent responses are mocked for demo purposes
- ⚠️ **No Live LLM Calls**: Agents don't call actual LLMs (to reduce latency/cost)

## Full Production Framework

The complete repository includes all production features:

- ✅ **Neural Meta-Controllers**: RNN and BERT with LoRA (deployed here!)
- ✅ **Agent Implementations**: Full HRM, TRM, and MCTS with PyTorch
- ✅ **Training Pipeline**: Data generation, training, evaluation
- ✅ **LLM Integration**: OpenAI, Anthropic, LM Studio support
- ✅ **RAG Systems**: ChromaDB, FAISS, Pinecone vector stores
- ✅ **Observability**: OpenTelemetry tracing, Prometheus metrics
- ✅ **Storage**: S3 artifact storage, experiment tracking
- ✅ **CI/CD**: Automated testing, security scanning, deployment

**GitHub Repository**: [ianshank/langgraph_multi_agent_mcts](https://github.com/ianshank/langgraph_multi_agent_mcts)

## Technical Stack

- **Python**: 3.11+
- **UI**: Gradio 4.x
- **ML Frameworks**: PyTorch 2.1+, Transformers, PEFT (LoRA)
- **Models**: GRU-based RNN, BERT-mini with LoRA adapters
- **Architecture**: Neural meta-controller + multi-agent system
- **Experiment Tracking**: Weights & Biases (optional)
- **Numerical**: NumPy

## Research Applications

This framework demonstrates concepts applicable to:

- Complex decision-making systems
- AI-assisted software architecture decisions
- Multi-perspective problem analysis
- Strategic planning with uncertainty

## Citation

If you use this framework in research, please cite:

```bibtex
@software{langgraph_mcts_2024,
  title={LangGraph Multi-Agent MCTS Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/ianshank/langgraph_multi_agent_mcts}
}
```

## License

MIT License - See repository for details.

---

**Built with** LangGraph, Gradio, and Python | **Demo Version**: 1.0.0
