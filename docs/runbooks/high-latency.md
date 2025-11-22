# Incident Runbook: High Latency

## Alert Details
- **Alert Name**: HighLatencyP95 / HighLatencyP99
- **Severity**: Warning (P95), Critical (P99)
- **Threshold**: P95 > 30s (warning), P99 > 60s (critical)
- **Prometheus Query**: `histogram_quantile(0.95, rate(mcts_request_duration_seconds_bucket[5m])) > 30`

## Symptoms
- Slow API response times
- Increased request queue depth
- Potential timeout errors
- User complaints about performance

## Impact
- **Users**: Degraded user experience, slow responses
- **System**: Resource saturation, potential cascading delays
- **Business**: Reduced throughput, potential SLA violations

## Investigation Steps

### 1. Confirm High Latency (2 minutes)
```bash
# Check current P95/P99 latency
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.95, rate(mcts_request_duration_seconds_bucket[5m]))'
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.99, rate(mcts_request_duration_seconds_bucket[5m]))'

# View latency by endpoint
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.95, rate(mcts_request_duration_seconds_bucket[5m])) by (endpoint)'
```

### 2. Identify Bottleneck Layer (5 minutes)
```bash
# Check agent-level latency
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.95, rate(mcts_agent_request_latency_seconds_bucket[5m])) by (agent_type)'

# Check MCTS iteration latency
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.95, rate(mcts_iteration_latency_seconds_bucket[5m]))'

# Check LLM request latency
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.95, rate(mcts_llm_request_latency_seconds_bucket[5m])) by (provider)'

# Check RAG retrieval latency
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.95, rate(mcts_rag_retrieval_latency_seconds_bucket[5m]))'
```

### 3. Check System Resources (3 minutes)
```bash
# Check CPU usage
docker stats langgraph-mcts --no-stream

# Check memory pressure
curl http://prometheus:9090/api/v1/query?query='process_resident_memory_bytes/(1024*1024*1024)'

# Check active operations
curl http://prometheus:9090/api/v1/query?query='mcts_active_operations'

# Check request queue depth
curl http://prometheus:9090/api/v1/query?query='mcts_request_queue_depth'
```

### 4. Review Distributed Traces (5 minutes)
```bash
# Open Jaeger UI
# Navigate to: http://localhost:16686
# Search for slow traces (duration > 30s)
# Analyze span timeline to identify bottleneck

# Check for external service latency
# Look for spans with tags: component=llm, component=rag
```

## Common Root Causes

### 1. LLM API Slowness
**Symptoms**: High `mcts_llm_request_latency_seconds`, spans show LLM calls taking >20s

**Resolution**:
```bash
# Check LLM provider performance
curl -w "%{time_total}\n" http://localhost:1234/v1/completions -d '{...}'

# Reduce MCTS iterations to decrease LLM calls
# Temporarily reduce from 100 to 50 iterations
docker exec langgraph-mcts sh -c 'echo "MCTS_DEFAULT_ITERATIONS=50" >> .env'
docker restart langgraph-mcts

# Consider switching to faster model
# Edit config to use smaller/faster model variant
```

### 2. RAG Vector Store Latency
**Symptoms**: High `mcts_rag_retrieval_latency_seconds`

**Resolution**:
```bash
# Check Pinecone performance
python -c "
from src.storage.pinecone_store import PineconeStore
import time
store = PineconeStore()
start = time.time()
results = store.query('test query', top_k=10)
print(f'Query time: {time.time() - start:.2f}s')
"

# Reduce number of retrieved documents temporarily
# Edit .env: RAG_TOP_K=5  # Reduce from 10

# Enable result caching
# Edit config to enable Redis caching for RAG results
```

### 3. MCTS Simulation Depth
**Symptoms**: High `mcts_iteration_latency_seconds`, deep simulation trees

**Resolution**:
```bash
# Check current simulation depth
curl http://prometheus:9090/api/v1/query?query='mcts_simulation_depth'

# Reduce max depth temporarily
# Edit MCTS config:
# MAX_SIMULATION_DEPTH=10  # Reduce from 20

# Reduce exploration constant (UCT)
# Lower exploration = faster convergence
# UCT_EXPLORATION_CONSTANT=1.0  # Reduce from 1.414
```

### 4. Resource Contention
**Symptoms**: High CPU/memory, many active operations

**Resolution**:
```bash
# Check resource limits
docker inspect langgraph-mcts | grep -A 10 "Resources"

# Increase container resources
# Edit docker-compose.yml:
#   resources:
#     limits:
#       cpus: '8'        # Increase from 4
#       memory: 16G      # Increase from 8G

# Scale horizontally
docker-compose up -d --scale mcts-framework=3

# Enable load balancing
# Add nginx/traefik load balancer
```

### 5. Database Connection Pool Exhaustion
**Symptoms**: Many operations waiting for connections

**Resolution**:
```bash
# Check connection pool metrics (if available)
curl http://prometheus:9090/api/v1/query?query='db_connection_pool_active'

# Increase pool size
# Edit database config:
# DB_POOL_SIZE=20  # Increase from 10
# DB_MAX_OVERFLOW=10

# Restart application
docker restart langgraph-mcts
```

### 6. Inefficient Query Patterns
**Symptoms**: Specific endpoints consistently slow

**Resolution**:
```bash
# Identify slow endpoints
docker logs langgraph-mcts --tail 500 | grep -E "duration=[0-9]{5,}"

# Enable query profiling
# Add timing logs to specific operations

# Optimize N+1 queries
# Review database query patterns in slow endpoints

# Add caching layer
# Enable Redis caching for frequently accessed data
```

## Resolution Steps

### Immediate Actions (5-10 minutes)
1. **Identify bottleneck**: Use investigation steps to find slow component
2. **Apply quick mitigation**: Reduce iterations, scale resources, or disable non-critical features
3. **Monitor impact**: Watch latency metrics for improvement

### Short-term Fixes (15-30 minutes)
1. **Scale resources**: Increase CPU/memory or add instances
2. **Tune parameters**: Adjust MCTS iterations, RAG top_k, timeouts
3. **Enable caching**: Add Redis caching for expensive operations
4. **Load balancing**: Distribute traffic across multiple instances

### Long-term Optimization (1-4 hours)
1. **Code optimization**: Profile and optimize slow code paths
2. **Architecture changes**: Introduce async processing, worker queues
3. **Infrastructure upgrades**: Better hardware, CDN, edge caching
4. **Algorithm tuning**: Optimize MCTS policy, better heuristics

## Escalation Path
1. **Level 1** (0-15 min): On-call engineer applies quick mitigations
2. **Level 2** (15-30 min): Senior engineer reviews traces, applies advanced fixes
3. **Level 3** (30-60 min): Engineering lead + SRE team for infrastructure changes
4. **Level 4** (60+ min): Product/business stakeholders informed if SLA risk

## Performance Tuning Quick Reference

### MCTS Parameters
```python
# Fast mode (low latency, lower quality)
MCTS_DEFAULT_ITERATIONS=50
MAX_SIMULATION_DEPTH=5
UCT_EXPLORATION_CONSTANT=1.0

# Balanced mode (default)
MCTS_DEFAULT_ITERATIONS=100
MAX_SIMULATION_DEPTH=10
UCT_EXPLORATION_CONSTANT=1.414

# Quality mode (high latency, better results)
MCTS_DEFAULT_ITERATIONS=200
MAX_SIMULATION_DEPTH=20
UCT_EXPLORATION_CONSTANT=2.0
```

### RAG Parameters
```python
# Fast retrieval
RAG_TOP_K=5
RAG_RERANK_ENABLED=false

# Balanced
RAG_TOP_K=10
RAG_RERANK_ENABLED=true
RAG_RERANK_TOP_N=5

# Comprehensive
RAG_TOP_K=20
RAG_RERANK_ENABLED=true
RAG_RERANK_TOP_N=10
```

## Prevention
1. **Performance budgets**: Set latency budgets per component
2. **Load testing**: Regular performance testing at scale
3. **Caching strategy**: Implement multi-layer caching
4. **Auto-scaling**: Configure horizontal pod autoscaling
5. **Rate limiting**: Protect against traffic spikes

## Post-Incident
1. **Document findings**: Record what caused latency spike
2. **Optimize hot paths**: Fix identified performance bottlenecks
3. **Add monitoring**: Create alerts for component-level latency
4. **Capacity planning**: Review if infrastructure needs upgrades

## Related Runbooks
- [High Error Rate](./high-error-rate.md)
- [Service Down](./service-down.md)
- [Resource Exhaustion](./resource-exhaustion.md)

## References
- [Performance Profiling Guide](../guides/performance-profiling.md)
- [Grafana Performance Dashboard](http://grafana:3000/d/mcts-performance)
- [Jaeger Traces](http://localhost:16686)
- [Architecture Documentation](../langgraph_mcts_architecture.md)
