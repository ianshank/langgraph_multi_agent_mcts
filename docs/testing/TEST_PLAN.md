# Comprehensive E2E Testing Plan

## Overview

This document outlines the comprehensive testing strategy for the Multi-Agent MCTS Training Pipeline, including E2E validations, integration testing, and user journey flows.

## Test Infrastructure

### Directory Structure
```
tests/
├── e2e/                          # End-to-end user journey tests
│   ├── test_complete_query_flow.py
│   ├── test_mcts_simulation_flow.py
│   └── ...
├── api/                          # API server tests
│   ├── test_rest_endpoints.py
│   └── ...
├── training/                     # Training pipeline tests
│   ├── test_dabstep_integration.py
│   ├── test_experiment_tracking.py
│   └── ...
├── mocks/                        # Mock external services
│   └── mock_external_services.py
├── fixtures/                     # Test data and scenarios
│   ├── tactical_scenarios.py
│   └── ...
└── unit/                         # Existing unit tests
```

### Test Markers
- `@pytest.mark.e2e` - End-to-end user journey tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.training` - Training pipeline tests
- `@pytest.mark.dataset` - Dataset integration tests
- `@pytest.mark.performance` - Performance and load tests
- `@pytest.mark.security` - Security validation tests

## API Keys Required

| Service | Environment Variable | Purpose | Required |
|---------|---------------------|---------|----------|
| OpenAI | `OPENAI_API_KEY` | Primary LLM provider | Yes |
| Anthropic | `ANTHROPIC_API_KEY` | Multi-provider testing | Optional |
| Braintrust | `BRAINTRUST_API_KEY` | Experiment tracking | Optional |
| W&B | `WANDB_API_KEY` | Training visualization | Optional |
| Pinecone | `PINECONE_API_KEY` | Vector storage | Optional |

## Datasets for Training/Testing

### Primary Datasets (FREE, Open-Source)

1. **DABStep** (CC-BY-4.0)
   - Source: Hugging Face (`adyen/DABstep`)
   - Size: 450+ multi-step reasoning tasks
   - Use: HRM/TRM agent training
   - Cost: $0

2. **PRIMUS-Seed** (ODC-BY)
   - Source: Trend Micro AI Lab
   - Size: 674,848 cybersecurity documents
   - Use: RAG knowledge base, domain grounding
   - Cost: $0

3. **PRIMUS-Instruct** (ODC-BY)
   - Source: Trend Micro AI Lab
   - Size: 835 instruction-tuning samples
   - Use: Instruction fine-tuning
   - Cost: $0

**Total Dataset Cost: $0** (All open-source with attribution required)

## E2E User Journey Tests

### Journey 1: Complete Query Processing
- **File**: `tests/e2e/test_complete_query_flow.py`
- **Flow**: Query → Validation → Multi-agent processing → Response
- **Expected**: 80-90% accuracy, <2s latency (simple queries)

### Journey 2: MCTS Tactical Simulation
- **File**: `tests/e2e/test_mcts_simulation_flow.py`
- **Flow**: MCTS initialization → UCB1 selection → Simulation → Backpropagation
- **Expected**: 200 iterations in <30s, deterministic with same seed

### Journey 3: Multi-Provider LLM Switching
- **Flow**: OpenAI → Anthropic → LMStudio
- **Expected**: Consistent interface, provider-specific optimizations

### Journey 4: Neural Meta-Controller Routing
- **Flow**: Feature extraction → RNN/BERT prediction → Agent selection
- **Expected**: 70-80% optimal routing accuracy

### Journey 5: Dataset-Driven Training Pipeline
- **File**: `tests/training/test_dabstep_integration.py`
- **Flow**: Load data → Preprocess → Augment → Train → Evaluate
- **Expected**: <$100 compute cost, 1-2 weeks to trained model

### Journey 6: Experiment Tracking
- **File**: `tests/training/test_experiment_tracking.py`
- **Flow**: Initialize experiment → Log metrics → Track artifacts → Compare
- **Expected**: Full auditability, reproducible results

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Simple query p95 | <2 seconds | End-to-end latency |
| MCTS 100 iterations | <5 seconds | Simulation time |
| MCTS 200 iterations | <30 seconds | Full tree search |
| Memory usage | <2GB | Under sustained load |
| GPU memory | 4-8GB | Fine-tuning + inference |
| Accuracy | 80-90% | Complex tactical tasks |
| Hallucination reduction | 40-60% | vs. base models |
| Consensus agreement | 75-85% | Inter-agent confidence |

## Running Tests

### Local Development
```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/e2e/ -m e2e -v
pytest tests/api/ -m api -v
pytest tests/training/ -m training -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance tests
pytest tests/performance/ -m performance --timeout=600
```

### CI/CD Pipeline
The GitHub Actions workflow (`/.github/workflows/ci.yml`) includes:
- **E2E Tests**: 30-minute timeout, all user journeys
- **API Tests**: Endpoint validation, authentication, rate limiting
- **Training Pipeline Tests**: Dataset loading, experiment tracking
- **Performance Tests**: Runs on main branch push only

## Expected Outcomes (Q1 2026)

### System Performance
- 80-90% accuracy on multi-step tactical reasoning
- 40-60% fewer hallucinations vs. base models
- 75-85% inter-agent consensus
- 10-30 second latency for complex MCTS simulations

### Business Impact
- Analysis time reduction: 4-8 hours → 10-30 minutes
- Analyst productivity: 2-3x increase
- Training cost: <$100-500 (one-time)
- Per-query cost: $0.10-1.00
- ROI break-even: 50-100 decisions (2-4 weeks)

### Success Criteria
- ✅ 70-80% system recommendations adopted
- ✅ 50-70% reduction in decision time
- ✅ >8/10 analyst satisfaction score
- ✅ 85%+ accuracy on retrospective cases
- ✅ <10% false alarm rate
- ✅ 3-5 blindspot discoveries per week

## Contributing

When adding new tests:
1. Use appropriate pytest markers
2. Follow existing test patterns
3. Add fixtures to `tests/fixtures/`
4. Update this documentation
5. Ensure CI/CD pipeline passes
