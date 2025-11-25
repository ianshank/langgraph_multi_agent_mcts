# Deployment Fix Toolkit

**Comprehensive multi-layered fix for Hugging Face Space deployment issues**

## 🚀 Quick Start

### One-Command Fix (Fastest)

```bash
python deploy.py --strategy minimal --auto --push
```

This gives you a working demo in 2-5 minutes with zero ML dependencies.

### Full-Featured Fix (Recommended)

```bash
python deploy.py --strategy conservative --auto --push
```

This gives you the full trained models with stable dependencies in 5-10 minutes.

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **QUICK_FIX.md** | Fast fixes and decision tree |
| **DEPLOYMENT_FIX_GUIDE.md** | Comprehensive guide with backpropagation workflow |
| **DEPLOYMENT_FIX_SUMMARY.md** | Complete overview of all files and strategies |
| **This file** | Getting started guide |

## 🛠️ Tools

### Core Scripts

| Script | Purpose | Example Usage |
|--------|---------|---------------|
| `deploy.py` | Master orchestrator | `python deploy.py --auto --push` |
| `verify_deployment.py` | Verify dependencies | `python verify_deployment.py --verbose` |
| `fix_dependencies.py` | Resolve conflicts | `python fix_dependencies.py --list` |
| `force_cache_bust.py` | Force rebuild | `python force_cache_bust.py --commit --push` |
| `app_minimal_fallback.py` | Zero-dep fallback | `cp app_minimal_fallback.py app.py` |

### Quick Command Reference

```bash
# Verify current state
python verify_deployment.py

# List all strategies
python fix_dependencies.py --list

# Get recommendation
python fix_dependencies.py --recommend

# Apply a strategy
python fix_dependencies.py --strategy minimal --apply

# Force cache-bust
python force_cache_bust.py --commit --push

# Full deployment (interactive)
python deploy.py

# Full deployment (automatic)
python deploy.py --strategy conservative --auto --push
```

## 🎯 Resolution Strategies

| Strategy | Best For | Risk | Time |
|----------|----------|------|------|
| **minimal** | Immediate fix needed | Low | 2 min |
| **conservative** | Production stability | Low | 5 min |
| **modern** | Latest features | Medium | 10 min |
| **no-lora** | PEFT conflicts | Low | 5 min |
| **cpu-only** | Smaller footprint | Medium | 10 min |

## 🔍 Problem Diagnosis

Run verification to identify issues:

```bash
python verify_deployment.py --verbose
```

This checks:
- ✅ Python version compatibility
- ✅ All dependencies installed
- ✅ Import compatibility
- ✅ Known conflicts (PEFT/transformers, sentence-transformers)
- ✅ transformers.modeling_layers availability

## 🏗️ Architecture

The toolkit uses **backpropagation principles**:

```
1. Identify surface errors (import failures)
2. Trace back through dependency chain
3. Find root cause (version conflicts)
4. Fix at each layer
5. Verify complete stack
```

### Error → Fix Mapping

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `transformers.modeling_layers` not found | PEFT 0.10 + transformers 4.46 | Upgrade PEFT or downgrade transformers |
| sentence-transformers conflicts | Incompatible with transformers 4.46 | Use conservative strategy |
| Cached old version | Space caching | Force cache-bust |
| Build failures | Dependency conflicts | Apply resolution strategy |

## 📋 Workflows

### Workflow 1: Fastest Fix

```bash
cd huggingface_space
python deploy.py --strategy minimal --auto --push
```

**Result:** Working demo (basic features only)

### Workflow 2: Production Deployment

```bash
cd huggingface_space

# Step 1: Verify
python verify_deployment.py --verbose

# Step 2: Fix dependencies
python fix_dependencies.py --strategy conservative --apply

# Step 3: Cache-bust and deploy
python force_cache_bust.py --commit --push
```

**Result:** Full-featured demo with trained models

### Workflow 3: Custom Strategy

```bash
cd huggingface_space

# Step 1: List strategies
python fix_dependencies.py --list

# Step 2: Test strategy (dry run)
python fix_dependencies.py --strategy modern --test

# Step 3: Apply if satisfied
python fix_dependencies.py --strategy modern --apply

# Step 4: Deploy
python force_cache_bust.py --commit --push
```

**Result:** Custom configuration

### Workflow 4: Interactive Deployment

```bash
cd huggingface_space
python deploy.py
```

Follow prompts to select strategy and deploy.

## 🔧 Troubleshooting

### Space won't rebuild?

```bash
python force_cache_bust.py --commit --push
```

### Dependencies still conflict?

```bash
python fix_dependencies.py --recommend
python fix_dependencies.py --strategy <recommended> --apply
```

### Everything fails?

```bash
python deploy.py --strategy minimal --auto --push
```

This uses zero ML dependencies and always works.

### Need to verify locally?

```bash
python verify_deployment.py --verbose
```

## ✅ Success Criteria

Your deployment is successful when:

1. ✅ Space builds without errors
2. ✅ No import errors on startup
3. ✅ Query submission works
4. ✅ Agent routing functions
5. ✅ Responses generated correctly

## 📖 More Information

- **QUICK_FIX.md** - Fast reference and decision tree
- **DEPLOYMENT_FIX_GUIDE.md** - Detailed guide with backpropagation workflow
- **DEPLOYMENT_FIX_SUMMARY.md** - Complete file overview

## 🆘 Support

If issues persist:
1. Use minimal strategy for guaranteed working demo
2. Check Space build logs for specific errors
3. Review detailed guides
4. Test locally with Python 3.10+
5. Open GitHub issue with error details

## 🎓 Learning

This toolkit demonstrates:
- **Backpropagation-style debugging** - trace errors to root cause
- **Multi-layered fixes** - fix at each dependency level
- **Cache-busting techniques** - force platform rebuilds
- **Graceful degradation** - fallback to simpler versions
- **Strategy pattern** - multiple resolution approaches

---

**Remember:** When in doubt, use `python deploy.py --strategy minimal --auto --push` for a guaranteed working demo!

**Space URL:** https://huggingface.co/spaces/ianshank/langgraph-mcts-demo
