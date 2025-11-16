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

**Proof-of-Concept Demo** - Experience multi-agent AI reasoning with Monte Carlo Tree Search (MCTS)

## What This Demo Shows

This interactive demo showcases a sophisticated multi-agent framework that combines:

### 🧠 Three Specialized Agents

1. **HRM (Hierarchical Reasoning Module)**
   - Decomposes complex queries into hierarchical components
   - Analyzes relationships between different levels of abstraction
   - Synthesizes insights from structured decomposition

2. **TRM (Tree Reasoning Module)**
   - Iteratively refines responses through multiple passes
   - Improves confidence with each refinement iteration
   - Employs different strategies: clarity, depth, validation

3. **MCTS (Monte Carlo Tree Search)**
   - Strategically explores solution space
   - Balances exploration vs exploitation
   - Visualizes decision tree with UCB1 scores

### 📊 Key Features

- **Consensus Scoring**: Measures agreement between agents
- **Configurable Parameters**: Adjust MCTS exploration weight and iterations
- **Deterministic Results**: Use seeds for reproducible searches
- **Performance Metrics**: Track execution time and token usage
- **Tree Visualization**: See the MCTS exploration in ASCII format
- **Weights & Biases Integration**: Track experiments, visualize metrics, compare runs

## How to Use

1. **Enter a Query**: Type your reasoning question or select an example
2. **Configure Agents**: Enable/disable HRM, TRM, and MCTS
3. **Adjust MCTS Settings**:
   - **Iterations**: More = better search (25-100 recommended)
   - **Exploration Weight**: Higher = more diverse search (default: 1.414)
   - **Seed**: Set for reproducible results
4. **Enable W&B Tracking** (optional): Expand "Weights & Biases Tracking" section
5. **Process**: Click to see multi-agent reasoning in action
6. **Analyze Results**: Review individual agent outputs and consensus

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

## Limitations (POC Demo)

This is a **proof-of-concept** demonstration with intentional simplifications:

- ❌ **Mock LLM Responses**: Uses rule-based responses, not production LLMs
- ❌ **No RAG/Vector Store**: Simplified context handling
- ❌ **Limited Domain Knowledge**: Pattern-matched responses only
- ❌ **Simplified MCTS**: Not connected to actual problem solving
- ❌ **No State Persistence**: Results not stored between sessions

## Full Production Framework

The complete framework includes:

- ✅ OpenAI, Anthropic, LM Studio LLM integration
- ✅ RAG with ChromaDB, FAISS, Pinecone vector stores
- ✅ Neural Meta-Controller for dynamic agent routing
- ✅ OpenTelemetry distributed tracing
- ✅ Prometheus metrics and structured logging
- ✅ S3 artifact storage
- ✅ Comprehensive input validation and security
- ✅ CI/CD pipeline with security scanning

**GitHub Repository**: [ianshank/langgraph_multi_agent_mcts](https://github.com/ianshank/langgraph_multi_agent_mcts)

## Technical Stack

- **Python**: 3.11+
- **UI**: Gradio 4.x
- **Algorithm**: Monte Carlo Tree Search with UCB1
- **Architecture**: Multi-agent orchestration pattern
- **Experiment Tracking**: Weights & Biases
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
