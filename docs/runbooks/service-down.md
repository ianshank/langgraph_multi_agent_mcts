# Incident Runbook: Service Down

## Alert Details
- **Alert Name**: ServiceDown
- **Severity**: Critical
- **Threshold**: Service unreachable for > 1 minute
- **Prometheus Query**: `up{job="mcts-framework"} == 0`

## Symptoms
- Health check endpoint returns non-200 status or times out
- Container not running or crashed
- No response from API endpoints
- Prometheus shows service as down

## Impact
- **Users**: Complete service unavailability, all requests failing
- **System**: Total outage, no traffic being processed
- **Business**: Critical SLA violation, revenue impact

## Investigation Steps

### 1. Verify Service Status (1 minute)
```bash
# Check container status
docker ps -a | grep langgraph-mcts

# Check health endpoint
curl -v http://localhost:8000/health

# Check if service is listening on port
netstat -an | grep 8000
# or
ss -tulpn | grep 8000
```

### 2. Check Container Logs (2 minutes)
```bash
# View recent logs
docker logs langgraph-mcts --tail 100

# Look for errors
docker logs langgraph-mcts --tail 500 | grep -E "(ERROR|CRITICAL|Fatal|Exception)"

# Check for OOM kills
docker inspect langgraph-mcts | grep OOMKilled

# Check exit code
docker inspect langgraph-mcts --format='{{.State.ExitCode}}'
```

### 3. Check System Resources (2 minutes)
```bash
# Check host resources
free -h
df -h
top -bn1 | head -20

# Check Docker daemon
systemctl status docker
docker info

# Check for port conflicts
sudo lsof -i :8000
```

### 4. Review Recent Changes (1 minute)
```bash
# Check recent deployments
git log -5 --oneline

# Check when container was started
docker inspect langgraph-mcts --format='{{.State.StartedAt}}'

# Check container restart count
docker inspect langgraph-mcts --format='{{.RestartCount}}'
```

## Common Root Causes

### 1. Container Crash/Exit
**Symptoms**: Container status is "Exited", exit code != 0

**Resolution**:
```bash
# Check exit code and reason
docker inspect langgraph-mcts --format='{{.State.ExitCode}} - {{.State.Error}}'

# View crash logs
docker logs langgraph-mcts --tail 200

# Restart container
docker restart langgraph-mcts

# If restart fails, check docker-compose
docker-compose up -d langgraph-mcts

# Monitor startup
docker logs -f langgraph-mcts
```

**Common Exit Codes**:
- Exit 0: Normal exit (check why it stopped)
- Exit 1: Application error (check logs for exception)
- Exit 137: OOM killed (increase memory limits)
- Exit 139: Segmentation fault (check dependencies)

### 2. Out of Memory (OOM)
**Symptoms**: OOMKilled=true, exit code 137

**Resolution**:
```bash
# Verify OOM kill
docker inspect langgraph-mcts | grep -A 5 OOMKilled

# Check current memory limits
docker stats langgraph-mcts --no-stream

# Increase memory limit
# Edit docker-compose.yml:
#   resources:
#     limits:
#       memory: 16G  # Increase from 8G

# Restart with new limits
docker-compose up -d

# Monitor memory usage
watch -n 1 docker stats langgraph-mcts
```

### 3. Failed Health Checks
**Symptoms**: Container running but failing health checks

**Resolution**:
```bash
# Check container health status
docker inspect langgraph-mcts --format='{{.State.Health.Status}}'

# View health check logs
docker inspect langgraph-mcts --format='{{range .State.Health.Log}}{{.Output}}{{end}}'

# Test health endpoint manually
docker exec langgraph-mcts curl -f http://localhost:8000/health

# Check if app is ready but health check is misconfigured
docker exec langgraph-mcts ps aux | grep python

# Temporarily disable health check to diagnose
# Edit docker-compose.yml, comment out healthcheck, restart
```

### 4. Missing Environment Variables
**Symptoms**: Startup errors about missing configuration

**Resolution**:
```bash
# Check environment variables
docker exec langgraph-mcts env | grep -E "(API_KEY|PROVIDER|DATABASE)"

# Verify .env file
cat .env

# Check if secrets are mounted
docker exec langgraph-mcts ls -la /run/secrets/ || echo "No secrets"

# Re-deploy with correct environment
docker-compose down
docker-compose up -d

# Verify startup
docker logs -f langgraph-mcts
```

### 5. Port Binding Conflict
**Symptoms**: Error about port already in use

**Resolution**:
```bash
# Check what's using port 8000
sudo lsof -i :8000
# or
sudo netstat -tulpn | grep 8000

# Kill conflicting process
sudo kill -9 <PID>

# Or change port in docker-compose.yml
# ports:
#   - "8001:8000"  # Map to different host port

# Restart service
docker-compose up -d
```

### 6. Dependency Service Failure
**Symptoms**: App starts but crashes due to missing dependencies

**Resolution**:
```bash
# Check all services
docker-compose ps

# Check dependency connectivity
docker exec langgraph-mcts ping prometheus -c 3
docker exec langgraph-mcts nc -zv otel-collector 4317

# Restart dependent services
docker-compose restart prometheus otel-collector

# Restart app after dependencies are up
docker-compose restart mcts-framework
```

### 7. Disk Space Exhaustion
**Symptoms**: Cannot write logs, database errors

**Resolution**:
```bash
# Check disk usage
df -h

# Check Docker disk usage
docker system df

# Clean up if needed
docker system prune -a --volumes
# WARNING: This removes unused containers, images, volumes

# Clean up old logs
find /var/log -type f -name "*.log" -mtime +7 -delete

# Restart service
docker-compose up -d
```

## Resolution Steps

### Immediate Recovery (2-5 minutes)
1. **Restart container**:
   ```bash
   docker restart langgraph-mcts
   # or
   docker-compose restart mcts-framework
   ```

2. **Verify recovery**:
   ```bash
   # Check health
   curl http://localhost:8000/health

   # Check logs for errors
   docker logs langgraph-mcts --tail 50
   ```

3. **If restart fails, redeploy**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Rollback Procedure (5-10 minutes)
If recent deployment caused the issue:

```bash
# Identify last working version
git log --oneline -10

# Rollback to previous version
git checkout <previous-commit-hash>

# Rebuild and deploy
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Verify
curl http://localhost:8000/health

# If working, create hotfix branch
git checkout -b hotfix/rollback-broken-deployment
git push origin hotfix/rollback-broken-deployment
```

### Full Recovery (10-20 minutes)
If simple restart doesn't work:

```bash
# 1. Stop everything
docker-compose down

# 2. Check for orphaned processes
docker ps -a | grep langgraph

# 3. Remove orphaned containers
docker rm -f $(docker ps -aq -f name=langgraph-mcts)

# 4. Pull latest images (if using registry)
docker-compose pull

# 5. Rebuild from scratch
docker-compose build --no-cache

# 6. Start with fresh state
docker-compose up -d

# 7. Monitor startup closely
docker-compose logs -f mcts-framework

# 8. Verify all services healthy
docker-compose ps
```

## Emergency Procedures

### Complete Cluster Failure
```bash
# 1. Take down entire stack
docker-compose down -v  # WARNING: Removes volumes

# 2. Check Docker daemon
sudo systemctl restart docker

# 3. Redeploy from clean state
docker-compose up -d

# 4. Restore data from backups (if needed)
# Follow backup restoration procedure
```

### Split-Brain / Multiple Instances Running
```bash
# 1. Find all running instances
docker ps -a | grep langgraph-mcts

# 2. Stop all instances
docker stop $(docker ps -q -f name=langgraph-mcts)

# 3. Remove all instances
docker rm $(docker ps -aq -f name=langgraph-mcts)

# 4. Start single instance
docker-compose up -d mcts-framework

# 5. Verify only one instance running
docker ps | grep langgraph-mcts | wc -l  # Should be 1
```

## Escalation Path
1. **Level 1** (0-5 min): On-call engineer attempts immediate restart
2. **Level 2** (5-10 min): Senior engineer investigates root cause, attempts rollback
3. **Level 3** (10-20 min): Engineering lead + SRE team for infrastructure issues
4. **Level 4** (20+ min): Declare P0 incident, engage all stakeholders

## Communication Templates

### Initial Alert (0-2 minutes)
```
🔴 P0 INCIDENT: MCTS Framework Service Down
Status: Investigating
Impact: Complete service outage
Actions: Restarting service
ETA: 5 minutes
Owner: [Your name]
```

### During Investigation (every 5 minutes)
```
🔴 P0 UPDATE: MCTS Framework Service Down
Status: [Investigating/Identified/Fixing]
Root Cause: [Brief description if known]
Progress: [What's been tried]
Next Steps: [What will be tried next]
ETA: [Updated estimate]
```

### Resolution
```
✅ RESOLVED: MCTS Framework Service Restored
Duration: [Total downtime]
Root Cause: [Brief summary]
Resolution: [What fixed it]
Follow-up: Post-mortem scheduled for [date/time]
```

## Prevention
1. **High availability**: Deploy multiple instances with load balancing
2. **Health monitoring**: Comprehensive health checks including dependencies
3. **Auto-restart**: Configure restart policies for automatic recovery
4. **Resource monitoring**: Alert on high memory/CPU before OOM
5. **Graceful degradation**: Service should handle dependency failures
6. **Chaos engineering**: Regular failure testing to validate resilience

## Post-Incident
1. **Incident report**: Complete detailed post-mortem within 24 hours
2. **Root cause analysis**: Deep dive into why service went down
3. **Action items**: Track improvements to prevent recurrence
4. **Update runbook**: Add new learnings to this document
5. **Review SLAs**: Assess if SLA was violated, notify stakeholders

## Related Runbooks
- [High Error Rate](./high-error-rate.md)
- [High Latency](./high-latency.md)
- [Database Connection Issues](./database-connection-issues.md)
- [Resource Exhaustion](./resource-exhaustion.md)

## References
- [Docker Compose Configuration](../../docker-compose.yml)
- [Deployment Guide](../deployment/README.md)
- [Health Check Configuration](../../Dockerfile)
- [Monitoring Dashboard](http://grafana:3000/d/mcts-overview)
