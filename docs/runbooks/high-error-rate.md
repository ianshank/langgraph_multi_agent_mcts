# Incident Runbook: High Error Rate

## Alert Details
- **Alert Name**: HighErrorRate
- **Severity**: Critical
- **Threshold**: Error rate > 5% for 5 minutes
- **Prometheus Query**: `rate(mcts_requests_total{status="error"}[5m]) / rate(mcts_requests_total[5m]) > 0.05`

## Symptoms
- Increased number of failed API requests
- Error responses returned to clients
- Possible degraded service quality
- Alert triggered in Alertmanager

## Impact
- **Users**: May experience failed requests or degraded functionality
- **System**: Increased load on retry mechanisms, potential cascading failures
- **Business**: Reduced service availability, potential SLA violations

## Investigation Steps

### 1. Verify the Alert (2 minutes)
```bash
# Check current error rate
curl http://prometheus:9090/api/v1/query?query='rate(mcts_requests_total{status="error"}[5m])/rate(mcts_requests_total[5m])'

# View recent errors in Grafana
# Navigate to: Dashboards > MCTS Framework > System Overview > Error Rate panel
```

### 2. Identify Error Types (5 minutes)
```bash
# Check error distribution by type
curl http://prometheus:9090/api/v1/query?query='topk(10, rate(mcts_errors_total[5m]))'

# Check application logs
docker logs langgraph-mcts --tail 100 | grep ERROR

# Check specific error patterns
docker logs langgraph-mcts --tail 500 | grep -E "(ValidationError|AuthenticationError|TimeoutError)"
```

### 3. Check Upstream Dependencies (3 minutes)
```bash
# Check LLM provider errors
curl http://prometheus:9090/api/v1/query?query='rate(mcts_llm_request_errors_total[5m])'

# Check RAG/vector store connectivity
curl http://localhost:8000/health

# Verify external service health
# - LMStudio: curl http://localhost:1234/v1/models
# - Pinecone: Check dashboard
# - LangSmith: Check status page
```

### 4. Review Recent Changes (2 minutes)
```bash
# Check recent deployments
git log -10 --oneline

# Check container restart history
docker ps -a | grep langgraph-mcts

# Review recent configuration changes
git diff HEAD~5 config/
```

## Common Root Causes

### 1. LLM Provider Issues
**Symptoms**: High rate of `mcts_llm_request_errors_total`

**Resolution**:
```bash
# Check LLM provider availability
curl http://localhost:1234/v1/models  # LMStudio
# or check OpenAI/Anthropic status pages

# Restart LLM service if needed
docker restart lmstudio  # if using containerized LLM

# Enable circuit breaker to fail fast
# Edit config and set LLM_TIMEOUT to lower value temporarily
```

### 2. Rate Limiting Exceeded
**Symptoms**: High `mcts_rate_limit_exceeded_total` concurrent with errors

**Resolution**:
```bash
# Check rate limit metrics
curl http://prometheus:9090/api/v1/query?query='rate(mcts_rate_limit_exceeded_total[5m])'

# Temporarily increase rate limits
# Edit .env or config/settings.py
RATE_LIMIT_PER_MINUTE=120  # Increase from 60

# Restart service
docker restart langgraph-mcts
```

### 3. Input Validation Failures
**Symptoms**: High rate of validation errors in logs

**Resolution**:
```bash
# Check validation error patterns
docker logs langgraph-mcts --tail 200 | grep ValidationError

# Review recent API schema changes
git diff HEAD~5 src/models/validation.py

# If invalid: Rollback deployment
git revert <commit-hash>
docker-compose up -d --build
```

### 4. Database/Vector Store Connection Issues
**Symptoms**: Connection timeout errors, RAG failures

**Resolution**:
```bash
# Check Pinecone connectivity
python -c "from src.storage.pinecone_store import PineconeStore; store = PineconeStore(); print('Connected')"

# Verify environment variables
docker exec langgraph-mcts env | grep -E "(PINECONE|LANGSMITH|OPENAI)"

# Restart connections
docker restart langgraph-mcts
```

### 5. Memory/Resource Exhaustion
**Symptoms**: OOM errors, slow responses leading to timeouts

**Resolution**:
```bash
# Check container resource usage
docker stats langgraph-mcts --no-stream

# Check memory metrics
curl http://prometheus:9090/api/v1/query?query='process_resident_memory_bytes/(1024*1024*1024)'

# If memory high: Restart container
docker restart langgraph-mcts

# Scale horizontally if persistent
docker-compose up -d --scale mcts-framework=3
```

## Resolution Steps

### Immediate Actions (5-10 minutes)
1. **Assess severity**: Check if error rate is still above threshold
2. **Identify root cause**: Use investigation steps above
3. **Apply quick fix**: Based on root cause identified
4. **Monitor recovery**: Watch error rate metrics for 5 minutes

### Short-term Mitigation (15-30 minutes)
1. **Scale resources**: If resource-related, increase container limits
2. **Enable circuit breakers**: Fail fast on downstream failures
3. **Adjust rate limits**: If client-related, tune rate limiting
4. **Deploy hotfix**: If code-related, deploy emergency patch

### Long-term Resolution (1-4 hours)
1. **Root cause analysis**: Document findings in incident report
2. **Implement permanent fix**: Code changes, infrastructure updates
3. **Add monitoring**: Create new alerts to catch earlier
4. **Update documentation**: Add learnings to this runbook

## Escalation Path
1. **Level 1** (0-15 min): On-call engineer investigates and attempts resolution
2. **Level 2** (15-30 min): Escalate to senior engineer if unresolved
3. **Level 3** (30+ min): Engage engineering lead and product team
4. **Level 4** (Critical): Notify stakeholders, declare major incident

## Communication Templates

### Internal (Slack/Teams)
```
🚨 INCIDENT: High Error Rate Detected
- Status: Investigating
- Error Rate: X%
- Impact: [Brief description]
- ETA: [Time to resolution]
- Owner: [Your name]
```

### External (Status Page)
```
We are currently experiencing elevated error rates.
Our team is actively investigating and working on a resolution.
We will provide updates every 15 minutes.
```

## Prevention
1. **Implement gradual rollouts**: Use canary deployments for changes
2. **Add comprehensive testing**: Increase test coverage for edge cases
3. **Improve monitoring**: Add predictive alerts before thresholds breach
4. **Conduct chaos testing**: Regularly test failure scenarios
5. **Review SLOs**: Ensure error budget allows for planned maintenance

## Post-Incident
1. **Create incident report**: Use template in `docs/incidents/template.md`
2. **Schedule post-mortem**: Within 48 hours of resolution
3. **Implement action items**: Track in issue tracker
4. **Update runbook**: Add new learnings from incident

## Related Runbooks
- [High Latency](./high-latency.md)
- [Service Down](./service-down.md)
- [LLM Provider Failures](./llm-provider-failures.md)

## References
- [Prometheus Alerts](../monitoring/alerts.yml)
- [Grafana Dashboards](http://grafana:3000/d/mcts-overview)
- [Application Logs](http://localhost:8000/logs)
- [Architecture Documentation](../langgraph_mcts_architecture.md)
