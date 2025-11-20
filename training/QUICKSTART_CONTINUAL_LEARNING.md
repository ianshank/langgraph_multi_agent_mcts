# Continual Learning - Quick Start Guide

Get started with the production feedback loop in 5 minutes!

## 1. Run the Demo (1 minute)

```bash
cd /home/user/langgraph_multi_agent_mcts
python training/examples/continual_learning_demo.py
```

This demonstrates all 6 components working together with synthetic data.

## 2. Integrate into Your Agent (2 minutes)

### Step 1: Initialize Logger

```python
import asyncio
from training.continual_learning import ProductionInteractionLogger, ProductionInteraction
import yaml

# Load config
with open('training/config.yaml') as f:
    config = yaml.safe_load(f)['continual_learning']

# Initialize logger
logger = ProductionInteractionLogger(config['logging'])
```

### Step 2: Log Interactions

```python
async def handle_user_query(query: str):
    """Your production agent handler."""

    # Generate response (your existing code)
    start_time = time.time()
    agent_selected, confidence = route_to_agent(query)
    retrieved_chunks = rag_retrieve(query)
    response = agent_generate(query, retrieved_chunks)
    latency = (time.time() - start_time) * 1000

    # Log interaction
    interaction = ProductionInteraction(
        interaction_id=str(uuid.uuid4()),
        timestamp=time.time(),
        session_id=get_session_id(),
        user_query=query,
        agent_selected=agent_selected,
        agent_confidence=confidence,
        response=response,
        latency_ms=latency,
        retrieval_quality=calculate_retrieval_score(retrieved_chunks),
    )

    # Async logging (non-blocking)
    asyncio.create_task(logger.log_interaction(interaction))

    return response
```

### Step 3: Collect User Feedback

```python
async def record_feedback(interaction_id: str, score: float, thumbs: str):
    """Record user feedback after response."""

    # Update the logged interaction
    interactions = logger.query_interactions(limit=1)
    # ... update with feedback

    # Re-log with feedback
    await logger.log_interaction(updated_interaction)
```

## 3. Analyze and Improve (2 minutes)

### Daily Failure Analysis

```python
from training.continual_learning import FailurePatternAnalyzer

async def daily_analysis():
    """Run daily to identify issues."""

    # Get recent interactions
    yesterday = time.time() - 86400
    interactions = logger.query_interactions(
        start_time=yesterday,
        limit=10000
    )

    # Analyze failures
    analyzer = FailurePatternAnalyzer(config['failure_analysis'])
    patterns = analyzer.analyze_failures(interactions)

    # Alert on high-severity issues
    for pattern in patterns:
        if pattern.severity > 0.8:
            send_alert(f"Critical: {pattern.description}")
            send_alert(f"Fix: {pattern.suggested_fix}")
```

### Weekly Retraining

```python
from training.continual_learning import (
    ActiveLearningSelector,
    IncrementalRetrainingPipeline
)

async def weekly_retrain():
    """Retrain models with production data."""

    # Get recent interactions
    last_week = time.time() - 604800
    interactions = logger.query_interactions(
        start_time=last_week,
        min_feedback_score=3.0,  # Quality filter
        limit=10000
    )

    # Select valuable samples
    selector = ActiveLearningSelector(config['active_learning'])
    candidates = selector.select_for_annotation(
        interactions,
        budget=50
    )

    # (Optional) Send for human annotation
    annotated = await get_annotations(candidates)

    # Retrain
    pipeline = IncrementalRetrainingPipeline(config['retraining'])
    result = await pipeline.retrain(
        new_data=annotated or interactions,
        old_model_path="models/production",
        output_path="models/candidate"
    )

    if result['status'] == 'completed':
        print("Retraining successful!")
        # A/B test is automatically setup if enabled
```

## 4. Monitor Metrics

### Query Statistics

```python
# Get overall stats
stats = logger.get_statistics()
print(f"Total interactions: {stats['total_in_db']}")
print(f"Avg satisfaction: {stats['avg_feedback_score']:.2f}/5.0")
print(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")

# Find low-rated interactions
low_rated = logger.query_interactions(
    max_feedback_score=3.0,
    limit=100
)
print(f"Low-rated: {len(low_rated)} interactions")

# Find slow responses
# (Query directly from SQLite for advanced filtering)
import sqlite3
conn = sqlite3.connect("./cache/production_logs/interactions.db")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM interactions WHERE latency_ms > 5000")
slow_count = cursor.fetchone()[0]
print(f"Slow responses (>5s): {slow_count}")
```

## 5. Complete Workflow Script

Create `scripts/continual_learning_workflow.py`:

```python
#!/usr/bin/env python
"""
Production continual learning workflow.
Run daily via cron: 0 2 * * * python scripts/continual_learning_workflow.py
"""

import asyncio
import time
from datetime import datetime
import yaml
from training.continual_learning import *

async def main():
    # Load config
    with open('training/config.yaml') as f:
        config = yaml.safe_load(f)['continual_learning']

    print(f"[{datetime.now()}] Starting continual learning workflow")

    # 1. Initialize components
    logger = ProductionInteractionLogger(config['logging'])
    analyzer = FailurePatternAnalyzer(config['failure_analysis'])
    selector = ActiveLearningSelector(config['active_learning'])
    pipeline = IncrementalRetrainingPipeline(config['retraining'])

    # 2. Get recent data
    yesterday = time.time() - 86400
    interactions = logger.query_interactions(
        start_time=yesterday,
        limit=10000
    )
    print(f"Loaded {len(interactions)} interactions from last 24h")

    # 3. Analyze failures
    patterns = analyzer.analyze_failures(interactions)
    print(f"Identified {len(patterns)} failure patterns")

    high_severity = [p for p in patterns if p.severity > 0.8]
    if high_severity:
        print(f"⚠️  {len(high_severity)} high-severity patterns:")
        for pattern in high_severity:
            print(f"   - {pattern.pattern_type}: {pattern.description}")
            print(f"     Fix: {pattern.suggested_fix}")

    # 4. Select annotation candidates (weekly)
    if datetime.now().weekday() == 0:  # Monday
        candidates = selector.select_for_annotation(
            interactions,
            budget=50
        )
        print(f"Selected {len(candidates)} candidates for annotation")

        # Export for annotation
        with open('data/annotation_queue.json', 'w') as f:
            json.dump([c.__dict__ for c in candidates], f, indent=2)

    # 5. Check for retraining (weekly)
    last_retrain_time = get_last_retrain_time()
    num_new_samples = get_new_sample_count(last_retrain_time)

    if pipeline.should_retrain(last_retrain_time, num_new_samples):
        print("Triggering retraining...")
        result = await pipeline.retrain(
            new_data=interactions,
            old_model_path="models/production",
            output_path="models/candidate"
        )
        print(f"Retraining: {result['status']}")

    print(f"[{datetime.now()}] Workflow complete")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Tips

### For Development
```yaml
continual_learning:
  logging:
    buffer_size: 10  # Small buffer for fast testing
    sanitize_pii: false  # Disable for testing

  retraining:
    min_new_samples: 5  # Low threshold for testing
```

### For Production
```yaml
continual_learning:
  logging:
    buffer_size: 1000  # Larger buffer for efficiency
    sanitize_pii: true  # Always enable
    use_langsmith: true  # Enable tracing

  retraining:
    min_new_samples: 100  # Reasonable threshold
    enable_ab_test: true  # Always A/B test
```

## Common Patterns

### Pattern 1: Real-time Alerting
```python
async def check_for_alerts():
    """Run every 5 minutes."""
    recent = logger.query_interactions(
        start_time=time.time() - 300,  # Last 5 min
        limit=100
    )

    # Alert on spike in failures
    failures = [i for i in recent if i['user_feedback_score'] < 2]
    if len(failures) > 10:
        send_alert(f"Failure spike: {len(failures)} in last 5min")
```

### Pattern 2: Personalized Feedback
```python
async def analyze_user_sessions():
    """Identify struggling users."""
    conn = sqlite3.connect(logger.db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT session_id, AVG(user_feedback_score) as avg_score
        FROM interactions
        WHERE timestamp > ?
        GROUP BY session_id
        HAVING avg_score < 3.0
    """, [time.time() - 86400])

    for session_id, avg_score in cursor.fetchall():
        # Reach out to struggling users
        send_feedback_request(session_id)
```

### Pattern 3: Cost Monitoring
```python
async def monitor_costs():
    """Track API costs."""
    stats = logger.query_interactions(limit=10000)
    total_cost = sum(s.get('cost', 0) for s in stats)
    avg_latency = np.mean([s['latency_ms'] for s in stats])

    print(f"Last 10K interactions:")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Cost per interaction: ${total_cost/len(stats):.4f}")
    print(f"  Avg latency: {avg_latency:.0f}ms")
```

## Troubleshooting

### Issue: Logs not appearing in database
```python
# Force flush
await logger._flush_buffer()

# Check buffer
print(f"Buffer size: {len(logger.interaction_buffer)}")

# Verify database
import sqlite3
conn = sqlite3.connect(logger.db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM interactions")
print(f"DB rows: {cursor.fetchone()[0]}")
```

### Issue: PII not being removed
```python
# Test PII detection
from training.continual_learning import DataQualityValidator

validator = DataQualityValidator(config['logging'])
text = "Contact me at test@example.com"
pii_found = validator._detect_pii(text)
print(f"PII detected: {pii_found}")

sanitized = validator._remove_pii(text)
print(f"Sanitized: {sanitized}")
```

### Issue: Retraining not triggering
```python
pipeline = IncrementalRetrainingPipeline(config['retraining'])

print(f"Schedule: {pipeline.schedule}")
print(f"Min samples: {pipeline.min_new_samples}")

# Check conditions
from datetime import datetime, timedelta
last_retrain = datetime.now() - timedelta(days=8)
should_retrain = pipeline.should_retrain(last_retrain, 150)
print(f"Should retrain: {should_retrain}")
```

## Next Steps

1. ✅ Run the demo to see it in action
2. ✅ Add logging to your production code
3. ✅ Set up daily failure analysis
4. ✅ Configure weekly retraining
5. 📊 Build a dashboard (optional)
6. 🔔 Set up alerting (optional)
7. 📈 Monitor metrics over time

## Resources

- Full documentation: `training/CONTINUAL_LEARNING.md`
- Demo script: `training/examples/continual_learning_demo.py`
- Tests: `training/tests/test_continual_learning.py`
- Configuration: `training/config.yaml`

## Support

Questions? Check:
1. Demo output for examples
2. Test cases for usage patterns
3. Documentation for detailed guides
4. Config comments for options

Happy learning! 🚀
