# Continual Learning System

Production-ready feedback loop for continuous improvement from real-world usage.

## Overview

The Continual Learning module enables your multi-agent MCTS system to learn and improve continuously from production data. It implements a complete feedback loop:

```
Production → Logging → Analysis → Selection → Retraining → Testing → Deployment
     ↑                                                                      ↓
     └──────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. ProductionInteractionLogger

Logs all production interactions with privacy preservation and efficient storage.

**Features:**
- SQLite database for queryable storage
- Compressed JSON backups
- Automatic PII sanitization
- LangSmith integration for tracing
- Async/non-blocking logging
- Batch writes for efficiency

**Usage:**
```python
from training.continual_learning import ProductionInteractionLogger, ProductionInteraction

# Initialize logger
config = {
    "enabled": True,
    "storage": "./cache/production_logs",
    "use_sqlite": True,
    "sanitize_pii": True,
}
logger = ProductionInteractionLogger(config)

# Log interaction
interaction = ProductionInteraction(
    interaction_id="unique_id",
    timestamp=time.time(),
    session_id="session_123",
    user_query="What is MCTS?",
    agent_selected="HRM",
    agent_confidence=0.85,
    response="MCTS is...",
    user_feedback_score=4.5,
    latency_ms=1200.0,
)

await logger.log_interaction(interaction)

# Query interactions
recent_failures = logger.query_interactions(
    max_feedback_score=3.0,
    limit=100
)
```

### 2. DataQualityValidator

Validates and sanitizes data for quality and privacy compliance.

**Features:**
- PII detection (email, phone, SSN, credit cards, API keys, IP addresses)
- Automatic PII redaction
- Length and coherence validation
- Custom blocked patterns
- Feedback consistency checks

**Supported PII Types:**
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- IP addresses
- API keys and tokens

**Usage:**
```python
from training.continual_learning import DataQualityValidator

validator = DataQualityValidator(config)

# Validate interaction
is_valid, issues = validator.validate_interaction(interaction)

# Sanitize PII
sanitized = validator.sanitize_interaction(interaction)
```

### 3. FailurePatternAnalyzer

Identifies systematic failure modes from production data.

**Detected Patterns:**
- Low-rated responses (< 3 stars)
- Poor retrieval quality
- Agent routing mistakes (low confidence)
- Hallucinations
- Slow responses (> 5s)
- Clustered similar failures (DBSCAN)

**Usage:**
```python
from training.continual_learning import FailurePatternAnalyzer

analyzer = FailurePatternAnalyzer({
    "min_cluster_size": 5,
    "similarity_threshold": 0.7
})

# Analyze failures
patterns = analyzer.analyze_failures(interactions)

for pattern in patterns:
    print(f"{pattern.pattern_type}: {pattern.description}")
    print(f"Suggested fix: {pattern.suggested_fix}")
```

### 4. ActiveLearningSelector

Selects most valuable examples for human annotation.

**Selection Strategies:**
- **Uncertainty sampling**: Select samples with low model confidence
- **Diversity sampling**: Cover feature space with diverse samples
- **Failure prioritization**: Focus on errors and failures
- **Hybrid**: Combine uncertainty and diversity

**Usage:**
```python
from training.continual_learning import ActiveLearningSelector

selector = ActiveLearningSelector({
    "selection_strategy": "hybrid",
    "diversity_weight": 0.3
})

# Select top candidates for annotation
candidates = selector.select_for_annotation(
    interactions,
    budget=50
)

for candidate in candidates:
    print(f"{candidate.interaction_id}: {candidate.selection_reason}")
```

### 5. IncrementalRetrainingPipeline

Orchestrates end-to-end model retraining with production data.

**Pipeline Steps:**
1. **Data Preparation**: Filter and split high-quality samples
2. **Meta-Controller Update**: Retrain routing model with new decisions
3. **RAG Index Update**: Add new documents from corrections
4. **Benchmark Evaluation**: Validate improvements
5. **A/B Testing**: Compare new vs old model

**Usage:**
```python
from training.continual_learning import IncrementalRetrainingPipeline

pipeline = IncrementalRetrainingPipeline({
    "schedule": "weekly",
    "min_new_samples": 100,
    "validation_split": 0.2,
    "enable_ab_test": True
})

# Check if should retrain
if pipeline.should_retrain(last_retrain_time, num_new_samples):
    result = await pipeline.retrain(
        new_data=interactions,
        old_model_path="models/v1",
        output_path="models/v2"
    )
```

### 6. DriftDetector

Detects distribution shifts in production data.

**Detection Methods:**
- Kolmogorov-Smirnov test
- Population Stability Index (PSI)

**Usage:**
```python
from training.continual_learning import DriftDetector

detector = DriftDetector({
    "window_size": 1000,
    "threshold": 0.1,
    "detection_method": "kolmogorov_smirnov"
})

# Set reference distribution
detector.set_reference_distribution(training_data)

# Monitor production data
for sample in production_stream:
    drift_report = detector.add_sample(sample)
    if drift_report:
        print(f"Drift detected: {drift_report.severity}")
```

### 7. ABTestFramework

A/B test new models against production baseline.

**Features:**
- Hash-based consistent assignment
- Statistical significance testing
- Configurable traffic split
- Automatic analysis when sufficient samples

**Usage:**
```python
from training.continual_learning import ABTestFramework

framework = ABTestFramework({
    "traffic_split": 0.1,  # 10% to new model
    "min_samples": 1000,
    "confidence_level": 0.95
})

# Create test
test_id = framework.create_test(
    "model_v2_test",
    model_a=old_model,
    model_b=new_model,
    metric_fn=lambda inp, out: calculate_success_metric(inp, out)
)

# Route traffic
group = framework.assign_group(test_id, request_id)
model = new_model if group == "B" else old_model

# Record result
framework.record_result(test_id, group, input_data, output, metric)

# Get recommendation
result = framework.end_test(test_id)
print(result["recommendation"])  # "Deploy B" or "Keep A"
```

## Configuration

Add to `training/config.yaml`:

```yaml
continual_learning:
  # Production interaction logging
  logging:
    enabled: true
    storage: "./cache/production_logs"
    use_sqlite: true
    use_compression: true
    buffer_size: 1000
    sanitize_pii: true
    use_langsmith: false
    langsmith_project: "production-feedback"

  # Failure pattern analysis
  failure_analysis:
    enabled: true
    min_cluster_size: 5
    similarity_threshold: 0.7
    analyze_frequency: "daily"

  # Active learning
  active_learning:
    enabled: true
    budget_per_cycle: 50
    selection_strategy: "hybrid"  # uncertainty, diversity, hybrid, failure

  # Incremental retraining
  retraining:
    schedule: "weekly"  # daily, weekly, monthly
    min_new_samples: 100
    validation_split: 0.2
    enable_ab_test: true

  # Data drift detection
  drift_detection:
    enabled: true
    window_size: 1000
    threshold: 0.1
    detection_method: "kolmogorov_smirnov"

  # A/B testing
  ab_testing:
    enabled: false
    traffic_split: 0.1
    min_samples: 1000
    confidence_level: 0.95

  # Privacy and security
  privacy:
    anonymize_user_ids: true
    hash_session_ids: true
    redact_sensitive_fields: true
    retention_days: 90
```

## Complete Workflow Example

```python
import asyncio
from training.continual_learning import *

async def continual_learning_loop():
    """Complete continual learning workflow."""

    # 1. Initialize components
    logger = ProductionInteractionLogger(config["logging"])
    analyzer = FailurePatternAnalyzer(config["failure_analysis"])
    selector = ActiveLearningSelector(config["active_learning"])
    pipeline = IncrementalRetrainingPipeline(config["retraining"])

    # 2. Log production interactions
    for interaction in production_stream:
        await logger.log_interaction(interaction)

    # 3. Analyze failures (daily)
    interactions = logger.query_interactions(limit=10000)
    patterns = analyzer.analyze_failures(interactions)

    print(f"Identified {len(patterns)} failure patterns")
    for pattern in patterns:
        print(f"- {pattern.pattern_type}: {pattern.suggested_fix}")

    # 4. Select samples for annotation (weekly)
    candidates = selector.select_for_annotation(
        interactions,
        budget=50
    )

    # Send to annotation platform
    annotated_samples = await send_for_annotation(candidates)

    # 5. Retrain models (when enough data)
    if pipeline.should_retrain(last_retrain_time, len(annotated_samples)):
        result = await pipeline.retrain(
            new_data=annotated_samples,
            old_model_path="models/production",
            output_path="models/candidate"
        )

        if result["status"] == "completed":
            print("Retraining successful!")

            # 6. A/B test new model
            ab_framework = ABTestFramework(config["ab_testing"])
            test_id = ab_framework.create_test(
                "retrain_test",
                model_a="production",
                model_b="candidate",
                metric_fn=success_metric
            )

            # Monitor test...
            # Deploy winner if significant improvement

# Run loop
asyncio.run(continual_learning_loop())
```

## Metrics Tracked

The system automatically tracks:

- **User Satisfaction**: Star ratings, thumbs up/down
- **Retrieval Quality**: Precision, recall, nDCG
- **Agent Performance**: Selection accuracy, confidence calibration
- **Latency**: Response time distribution, p95, p99
- **Failure Rates**: By category (retrieval, routing, hallucination, etc.)
- **Drift Metrics**: Distribution shifts over time

## Privacy & Security

### PII Removal
- Automatic detection of sensitive data
- Regex-based pattern matching
- Configurable redaction rules

### Data Retention
- Configurable retention periods
- Automatic data expiration
- Secure deletion

### Access Control
- Anonymized user IDs
- Hashed session identifiers
- Role-based access (via external system)

## Integration Points

### LangSmith
```python
# Enable LangSmith tracing
config["logging"]["use_langsmith"] = True
config["logging"]["langsmith_project"] = "production-feedback"

# Interactions automatically logged to LangSmith
```

### Existing Training Infrastructure
- `training/agent_trainer.py`: Used for model retraining
- `training/meta_controller.py`: Updated with new routing decisions
- `training/rag_builder.py`: RAG index updates
- `training/benchmark_suite.py`: Evaluation metrics

## Storage

### SQLite Database Schema
```sql
CREATE TABLE interactions (
    interaction_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    user_query TEXT NOT NULL,
    agent_selected TEXT,
    agent_confidence REAL,
    response TEXT NOT NULL,
    user_feedback_score REAL,
    thumbs_up_down TEXT,
    latency_ms REAL,
    tokens_used INTEGER,
    cost REAL,
    hallucination_detected INTEGER,
    retrieval_failed INTEGER,
    error_occurred INTEGER,
    retrieval_quality REAL,
    metadata TEXT
);
```

### Compressed JSON Backups
- Automatic gzip compression
- Daily batch exports
- Easy archival to cold storage

## Running Tests

```bash
# Run all tests
pytest training/tests/test_continual_learning.py -v

# Run specific test class
pytest training/tests/test_continual_learning.py::TestProductionInteractionLogger -v

# Run with coverage
pytest training/tests/test_continual_learning.py --cov=training.continual_learning
```

## Demo

Run the complete demo:

```bash
python training/examples/continual_learning_demo.py
```

This demonstrates:
1. Production interaction logging
2. Failure pattern identification
3. Active learning selection
4. Incremental retraining
5. Drift detection
6. A/B testing

## Performance Considerations

### Logging
- Batch writes reduce I/O overhead
- Async logging doesn't block production
- SQLite provides fast queries
- Compression reduces storage by ~70%

### Analysis
- Failure analysis: O(n) where n = interactions
- Clustering: O(n²) for DBSCAN (run offline)
- Active learning: O(n log n) for sorting

### Storage
- SQLite: 1M interactions ≈ 500MB
- Compressed JSON: 1M interactions ≈ 150MB
- Automatic archival after 30 days

## Best Practices

1. **Start Small**: Begin with 10% sampling rate
2. **Monitor Quality**: Track validation failure rates
3. **Gradual Rollout**: Use A/B testing for all changes
4. **Regular Analysis**: Run failure analysis daily
5. **Privacy First**: Always sanitize PII
6. **Automate**: Schedule retraining workflows
7. **Validate**: Benchmark before deployment
8. **Document**: Track what was learned from each cycle

## Troubleshooting

### High Validation Failure Rate
- Check PII patterns are too strict
- Verify query/response length limits
- Review blocked patterns

### No Patterns Detected
- Increase min_cluster_size
- Lower similarity_threshold
- Ensure sufficient failure samples

### Slow Logging
- Increase buffer_size
- Disable compression for testing
- Check database indices

### Drift False Positives
- Increase detection threshold
- Use larger window_size
- Try different detection method (PSI vs KS)

## Roadmap

- [ ] Multi-modal feedback (images, code)
- [ ] Automated annotation with LLMs
- [ ] Real-time dashboard
- [ ] Distributed logging
- [ ] Advanced drift detection (DDM, ADWIN)
- [ ] Causal analysis of failures
- [ ] Automated remediation suggestions

## License

Part of the LangGraph Multi-Agent MCTS project.

## Support

For issues or questions:
1. Check existing tests for examples
2. Review demo script
3. Consult configuration documentation
4. Open an issue on GitHub
