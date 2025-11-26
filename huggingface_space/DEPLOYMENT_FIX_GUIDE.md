# Comprehensive Deployment Fix Guide for Hugging Face Spaces

**Space:** ianshank/langgraph-mcts-demo
**Last Updated:** 2025-11-25
**Status:** Multi-layered fix implementation

## Executive Summary

This guide provides a comprehensive, multi-layered approach to fixing the Hugging Face Space deployment using **backpropagation principles**: identify errors at the surface, trace back through dependencies, and fix at each level.

## Current Problems

### 1. Dependency Conflicts
- **PEFT 0.10.0** incompatible with **transformers 4.46.0**
  - Error: `ModuleNotFoundError: No module named 'transformers.modeling_layers'`
  - Root cause: PEFT 0.10.0 expects `modeling_layers` module removed in transformers 4.46+

- **sentence-transformers** incompatible with **transformers 4.46.0**
  - Indirect dependency conflicts
  - Causes build failures

### 2. Caching Issues
- Hugging Face Spaces using cached old version of app.py
- requirements.txt changes not triggering rebuild
- Old dependencies cached even after updates

### 3. Build Strategy Issues
- No verification before deployment
- No fallback for dependency failures
- No systematic conflict resolution

## Multi-Layered Fix Strategy

### Layer 1: Immediate Fixes (Quick Resolution)

#### Option A: Use Minimal Fallback App (RECOMMENDED for immediate fix)

This uses zero ML dependencies and works immediately.

```bash
# In your local huggingface_space directory
cd huggingface_space

# Copy minimal fallback as main app
cp app_minimal_fallback.py app.py

# Use minimal requirements
python fix_dependencies.py --strategy minimal --apply

# Force cache bust
python force_cache_bust.py --commit --push
```

**Pros:**
- Works immediately
- Zero ML dependencies
- No conflicts possible

**Cons:**
- No trained models
- Heuristic-based routing only

#### Option B: Conservative Versions Strategy

Use older stable versions for maximum compatibility.

```bash
# Apply conservative strategy
python fix_dependencies.py --strategy conservative --apply

# Force cache bust
python force_cache_bust.py --commit --push
```

**Requirements:**
```
transformers>=4.40.0,<4.46.0
peft==0.10.0
sentence-transformers>=2.2.0,<3.0.0
```

### Layer 2: Dependency Resolution (Fix Conflicts)

#### Step 1: Verify Current State

```bash
# Run verification
python verify_deployment.py --verbose

# Analyze conflicts
python fix_dependencies.py --recommend
```

#### Step 2: Choose Resolution Strategy

```bash
# List all strategies
python fix_dependencies.py --list

# Test a strategy (dry run)
python fix_dependencies.py --strategy modern --test

# Apply strategy
python fix_dependencies.py --strategy modern --apply
```

**Available Strategies:**

| Strategy | Description | Risk | Use When |
|----------|-------------|------|----------|
| **minimal** | Zero ML dependencies | Low | Need immediate working demo |
| **conservative** | Older stable versions | Low | Maximum compatibility |
| **modern** | Latest versions | Medium | Want latest features |
| **no-lora** | Disable PEFT/LoRA | Low | LoRA conflicts |
| **cpu-only** | CPU-only PyTorch | Medium | Reduce size |

#### Step 3: Update App Configuration

If using **no-lora** strategy, update app.py:

```python
# In app.py, line ~226
self.bert_controller = BERTMetaController(
    name="BERTController",
    seed=42,
    device=self.device,
    use_lora=False  # Change to False
)
```

If using **modern** strategy, disable sentence-transformers:

```python
# In app.py or feature_extractor.py
# Comment out sentence-transformers imports
# Use heuristic-based extraction instead
```

### Layer 3: Cache Busting (Force Rebuild)

#### Aggressive Cache-Busting Techniques

```bash
# Full cache bust with commit and push
python force_cache_bust.py --commit --push
```

This script:
1. Adds unique timestamp to requirements.txt
2. Updates version markers in app.py
3. Creates deployment marker file
4. Generates unique commit message
5. Forces git push

**Manual cache-busting (if script fails):**

```bash
# Add timestamp to requirements.txt
echo "# Force rebuild: $(date)" >> requirements.txt

# Update app.py version
# Change APP_VERSION manually

# Commit with unique message
git add .
git commit -m "fix: force rebuild $(date +%s)"
git push
```

### Layer 4: Verification & Testing

#### Pre-Deployment Verification

```bash
# Full verification suite
python verify_deployment.py --verbose

# Expected output:
# ✅ Python version check
# ✅ All critical imports
# ✅ No compatibility issues
# ✅ transformers.modeling_layers available
```

#### Post-Deployment Monitoring

1. **Check Build Logs** on Hugging Face Spaces
   - Look for dependency installation errors
   - Verify all packages installed
   - Check for import errors

2. **Test Space Functionality**
   - Submit test query
   - Verify agent routing
   - Check response generation

3. **Monitor Performance**
   - Build time
   - Runtime errors
   - Response times

## Backpropagation Workflow

This follows backpropagation principles to trace and fix errors:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Surface Error                                  │
│ "ModuleNotFoundError: transformers.modeling_layers"     │
└─────────────────────────────────────────────────────────┘
                         ↓ Trace back
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Import Chain                                   │
│ PEFT → transformers.modeling_layers (missing in 4.46+)  │
└─────────────────────────────────────────────────────────┘
                         ↓ Trace back
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Version Mismatch                               │
│ PEFT 0.10.0 requires transformers <4.46                 │
│ App specifies transformers >=4.46.0                     │
└─────────────────────────────────────────────────────────┘
                         ↓ Trace back
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Root Cause                                     │
│ requirements.txt has incompatible version constraints   │
└─────────────────────────────────────────────────────────┘
                         ↓ Fix at each level
┌─────────────────────────────────────────────────────────┐
│ Fix Layer 1: Update requirements.txt                    │
│ Option A: peft>=0.13.0 + transformers>=4.46.0           │
│ Option B: peft==0.10.0 + transformers<4.46.0            │
│ Option C: Remove PEFT, use base BERT                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Fix Layer 2: Update app.py if needed                    │
│ Disable LoRA if using Option C                          │
│ Add graceful fallbacks for missing modules              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Fix Layer 3: Force cache refresh                        │
│ Update timestamps, version markers, deployment IDs      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Fix Layer 4: Verify and deploy                          │
│ Run verification, test locally, deploy to HF Spaces     │
└─────────────────────────────────────────────────────────┘
```

## Recommended Fix Sequence

### For Immediate Working Demo (Fastest)

```bash
# 1. Use minimal fallback
cp app_minimal_fallback.py app.py
python fix_dependencies.py --strategy minimal --apply

# 2. Force rebuild
python force_cache_bust.py --commit --push

# 3. Monitor Space rebuild
# Visit: https://huggingface.co/spaces/ianshank/langgraph-mcts-demo
```

**Time to fix:** ~5 minutes
**Reliability:** Very high
**Features:** Basic demo only

### For Full Features with Stability (Recommended)

```bash
# 1. Use conservative versions
python fix_dependencies.py --strategy conservative --apply

# 2. Verify locally (optional but recommended)
python verify_deployment.py --verbose

# 3. Force rebuild
python force_cache_bust.py --commit --push

# 4. Monitor deployment
```

**Time to fix:** ~10 minutes
**Reliability:** High
**Features:** Full trained models

### For Latest Features (Higher Risk)

```bash
# 1. Use modern versions
python fix_dependencies.py --strategy modern --apply

# 2. Update app.py to disable sentence-transformers if needed
# (comment out imports in feature_extractor.py)

# 3. Verify
python verify_deployment.py --verbose

# 4. Deploy
python force_cache_bust.py --commit --push
```

**Time to fix:** ~15-20 minutes
**Reliability:** Medium
**Features:** Latest versions

## Troubleshooting

### Problem: Cache still using old version

**Solution:**
```bash
# Nuclear option - change filename
mv app.py app_old.py
cp app_minimal_fallback.py app.py

# Force push with unique message
git add .
git commit -m "fix: complete app replacement $(date +%s)"
git push --force
```

### Problem: Dependencies still conflict

**Solution:**
```bash
# Verify what's actually needed
python verify_deployment.py --verbose

# Try minimal strategy
python fix_dependencies.py --strategy minimal --apply

# Or manually create requirements.txt with only:
cat > requirements.txt << EOF
gradio>=4.0.0,<5.0.0
numpy>=1.24.0,<2.0.0
EOF
```

### Problem: Build succeeds but runtime errors

**Solution:**
```bash
# Check app.py imports
# Add try-except for all ML imports
# Use app_minimal_fallback.py which has graceful fallbacks
```

### Problem: Space won't rebuild

**Solution:**
```bash
# 1. Make a trivial change to trigger rebuild
echo "# Rebuild trigger: $(date)" >> README.md
git add README.md
git commit -m "chore: trigger rebuild"
git push

# 2. Check Space settings
# - Verify SDK is "gradio"
# - Verify Python version (3.10+)
# - Check for build errors in logs
```

## Scripts Reference

### verify_deployment.py
Comprehensive verification of all dependencies.

```bash
# Basic verification
python verify_deployment.py

# Verbose output
python verify_deployment.py --verbose

# Custom requirements file
python verify_deployment.py --requirements custom_requirements.txt
```

### fix_dependencies.py
Multi-strategy dependency resolver.

```bash
# List strategies
python fix_dependencies.py --list

# Recommend best strategy
python fix_dependencies.py --recommend

# Test strategy (dry run)
python fix_dependencies.py --strategy modern --test

# Apply strategy
python fix_dependencies.py --strategy modern --apply
```

### force_cache_bust.py
Aggressive cache-busting for HF Spaces.

```bash
# Update files only
python force_cache_bust.py

# Update and commit
python force_cache_bust.py --commit

# Update, commit, and push
python force_cache_bust.py --commit --push
```

### app_minimal_fallback.py
Zero-dependency fallback app.

```bash
# Use as main app
cp app_minimal_fallback.py app.py

# Or run standalone for testing
python app_minimal_fallback.py
```

## Success Criteria

Your deployment is successful when:

1. ✅ Build completes without errors
2. ✅ Space launches without import errors
3. ✅ Query submission works
4. ✅ Agent routing functions
5. ✅ Responses generated correctly

## Additional Resources

- **HF Spaces Docs:** https://huggingface.co/docs/hub/spaces
- **Gradio Docs:** https://gradio.app/docs/
- **Transformers Compatibility:** https://github.com/huggingface/transformers/releases
- **PEFT Compatibility:** https://github.com/huggingface/peft#compatibility

## Support

If issues persist after trying all fixes:

1. Check Space build logs for specific errors
2. Review HF Spaces status page
3. Test locally with same Python version (3.10+)
4. Consider using minimal fallback app temporarily
5. Open issue on GitHub repository with error details

---

**Remember:** The backpropagation approach means fixing from the root cause up, not just patching surface errors!
