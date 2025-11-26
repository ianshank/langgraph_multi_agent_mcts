# Quick Fix Reference

**Problem:** Hugging Face Space deployment failing
**Space:** ianshank/langgraph-mcts-demo

## Fastest Fix (2 minutes)

Use minimal fallback app with zero dependencies:

```bash
cd huggingface_space

# One-command fix
python deploy.py --strategy minimal --auto --push
```

This will:
1. Switch to minimal fallback app
2. Use zero ML dependencies
3. Force cache rebuild
4. Push to Space

**Result:** Working demo in ~2-5 minutes

## Standard Fix (5 minutes)

Use stable versions with full features:

```bash
cd huggingface_space

# Interactive deployment
python deploy.py
# Select option 2 (conservative) when prompted
```

**Result:** Full-featured demo with trained models

## Manual Quick Fixes

### Fix 1: Just Dependencies

```bash
# Apply conservative strategy
python fix_dependencies.py --strategy conservative --apply

# Force rebuild
python force_cache_bust.py --commit --push
```

### Fix 2: Nuclear Option

If nothing else works:

```bash
# Use minimal app
cp app_minimal_fallback.py app.py

# Minimal requirements
cat > requirements.txt << EOF
gradio>=4.0.0,<5.0.0
numpy>=1.24.0,<2.0.0
EOF

# Force push
git add .
git commit -m "fix: minimal fallback $(date +%s)"
git push --force
```

## Verification

Check if deployment will work:

```bash
python verify_deployment.py
```

## Troubleshooting

### Space still cached?

```bash
# Aggressive cache bust
python force_cache_bust.py --commit --push
```

### Dependency conflicts?

```bash
# List all strategies
python fix_dependencies.py --list

# Get recommendation
python fix_dependencies.py --recommend
```

### Still failing?

```bash
# Use minimal fallback
python deploy.py --strategy minimal --auto --push
```

## Scripts Quick Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy.py` | Master deployment | `python deploy.py --auto --push` |
| `verify_deployment.py` | Check dependencies | `python verify_deployment.py` |
| `fix_dependencies.py` | Resolve conflicts | `python fix_dependencies.py --strategy minimal --apply` |
| `force_cache_bust.py` | Force rebuild | `python force_cache_bust.py --commit --push` |
| `app_minimal_fallback.py` | Zero-dependency app | `cp app_minimal_fallback.py app.py` |

## Decision Tree

```
Is Space failing?
│
├─ Yes → Need immediate fix?
│   │
│   ├─ Yes → Use minimal fallback
│   │         python deploy.py --strategy minimal --auto --push
│   │
│   └─ No → Want full features?
│       │
│       ├─ Yes → Use conservative strategy
│       │         python deploy.py --strategy conservative --auto --push
│       │
│       └─ No → Use modern strategy
│                 python deploy.py --strategy modern --auto --push
│
└─ No → Just verify
         python verify_deployment.py
```

## Common Scenarios

### Scenario 1: PEFT/Transformers Conflict

**Error:** `ModuleNotFoundError: transformers.modeling_layers`

**Fix:**
```bash
python fix_dependencies.py --strategy modern --apply
python force_cache_bust.py --commit --push
```

### Scenario 2: sentence-transformers Conflict

**Error:** Incompatible with transformers 4.46.0

**Fix:**
```bash
python fix_dependencies.py --strategy conservative --apply
python force_cache_bust.py --commit --push
```

### Scenario 3: Everything Fails

**Fix:**
```bash
python deploy.py --strategy minimal --auto --push
```

---

**TL;DR:** Run `python deploy.py --strategy minimal --auto --push` for guaranteed working demo.
