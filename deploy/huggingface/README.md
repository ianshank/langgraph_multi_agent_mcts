---
title: LangGraph Multi-Agent MCTS
emoji: "🧠"
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "4.0.0"
app_port: 7860
python_version: "3.11"
pinned: false
fullWidth: true
short_description: DeepMind-style AI system with Neural MCTS and hierarchical reasoning
suggested_hardware: cpu-basic
---

# LangGraph Multi-Agent MCTS Framework

**Production-Ready DeepMind-Style AI System with Neural MCTS and Hierarchical Reasoning**

## Features

- **HRM (Hierarchical Reasoning Module)**: DeBERTa-based agent for complex problem decomposition
- **TRM (Task Refinement Module)**: Iterative agent for refining and optimizing solutions
- **Neural MCTS**: AlphaZero-style tree search guided by policy/value networks
- **Meta-Controller**: Neural router (GRU/BERT) that dynamically assigns tasks to the best agent

## How It Works

The framework uses trained neural meta-controllers to intelligently route queries:

1. **RNN Meta-Controller**: GRU-based sequential pattern recognition
2. **BERT with LoRA**: Transformer-based text understanding for routing

Queries are routed to the optimal agent based on their characteristics:
- **HRM**: Complex decomposition problems
- **TRM**: Iterative refinement tasks
- **MCTS**: Strategic optimization problems

## Configuration

This Space requires the following secrets to be configured:

| Secret | Required | Description |
|--------|----------|-------------|
| `OPENAI_API_KEY` | Optional | OpenAI API key for LLM capabilities |
| `ANTHROPIC_API_KEY` | Optional | Anthropic API key for Claude models |
| `PINECONE_API_KEY` | Optional | Pinecone API key for RAG features |
| `WANDB_API_KEY` | Optional | Weights & Biases for experiment tracking |

## Usage

1. Enter your query in the text box
2. Select a meta-controller (RNN or BERT)
3. Click "Process Query" to see the routing decision and response

## Repository

[GitHub - langgraph_multi_agent_mcts](https://github.com/ianshank/langgraph_multi_agent_mcts)

## License

MIT License
