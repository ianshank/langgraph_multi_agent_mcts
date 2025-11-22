# Local Training Guide - 16GB GPU Demo Mode

This guide explains how to run the complete Multi-Agent MCTS training pipeline in **demo mode** on a local machine with a 16GB GPU. The demo provides end-to-end verification of all training components with reduced dataset sizes and optimized models.

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Environment Setup](#environment-setup)
- [Running the Demo](#running-the-demo)
- [Understanding Demo Mode](#understanding-demo-mode)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)
- [Scaling to Production](#scaling-to-production)
- [FAQ](#faq)

---

## Overview

The **Local Demo Mode** is a streamlined version of the full training pipeline designed to:

✅ Verify all training components work correctly
✅ Validate external service integrations (Pinecone, W&B, GitHub)
✅ Complete training in **30-45 minutes** instead of days
✅ Run on a single 16GB GPU without distributed training
✅ Provide a feedback loop for development and debugging

### What Gets Verified

| Component | Verification |
|-----------|-------------|
| **HRM (Hierarchical Reasoning Model)** | Training with reduced depth and LoRA |
| **TRM (Task Refinement Model)** | Training with fewer iterations |
| **MCTS (Monte Carlo Tree Search)** | Self-play with reduced simulations |
| **RAG (Retrieval-Augmented Generation)** | Indexing and retrieval with Pinecone |
| **Meta-Controller** | Router training on synthetic traces |
| **Monitoring** | W&B experiment tracking integration |

---

## Hardware Requirements

### Minimum Requirements

- **GPU**: NVIDIA GPU with 16GB VRAM (e.g., RTX 4080, A4000, V100)
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 32GB system memory
- **Storage**: 50GB free space (SSD recommended)

### Tested Configurations

| GPU | VRAM | Status | Notes |
|-----|------|--------|-------|
| RTX 4090 | 24GB | ✅ Excellent | Completes in ~25 minutes |
| RTX 4080 | 16GB | ✅ Good | Completes in ~35 minutes |
| RTX 3090 | 24GB | ✅ Excellent | Completes in ~30 minutes |
| A100 | 40GB | ✅ Excellent | Overkill for demo, but works |
| V100 | 16GB | ✅ Good | Completes in ~40 minutes |
| RTX 3080 | 10GB | ⚠️ Marginal | May encounter OOM errors |

### GPU Check

To verify your GPU meets requirements:

```bash
# Check GPU memory
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check CUDA version
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## Software Requirements

### Required

- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (12.x recommended)
- **PyTorch**: 2.1+ with CUDA support
- **Git**: For cloning repository (optional)

### Python Packages

All required packages are in [`requirements.txt`](../requirements.txt):

```bash
pip install -r requirements.txt
```

Key packages:
- `torch>=2.1.0` - Deep learning framework
- `transformers` - Hugging Face models
- `pinecone-client>=3.0.0` - Vector database
- `wandb>=0.16.0` - Experiment tracking
- `sentence-transformers` - Embeddings
- `pydantic>=2.0.0` - Configuration validation
- `rich>=13.0.0` - Terminal UI

---

## Environment Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/langgraph_multi_agent_mcts.git
cd langgraph_multi_agent_mcts
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Step 4: Set Environment Variables

Create a `.env` file in the project root:

```bash
# Required for demo mode
PINECONE_API_KEY=your-pinecone-api-key
WANDB_API_KEY=your-wandb-api-key
GITHUB_TOKEN=your-github-token

# Optional
OPENAI_API_KEY=your-openai-api-key
NEO4J_PASSWORD=your-neo4j-password

# Configuration
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=.
```

#### How to Get API Keys

**Pinecone:**
1. Sign up at [https://www.pinecone.io/](https://www.pinecone.io/)
2. Navigate to API Keys in dashboard
3. Create a new API key
4. Copy to `.env` file

**Weights & Biases:**
1. Sign up at [https://wandb.ai/](https://wandb.ai/)
2. Go to Settings → API Keys
3. Copy your API key
4. Run `wandb login` or add to `.env`

**GitHub Token:**
1. Go to GitHub Settings → Developer Settings → Personal Access Tokens
2. Generate new token (classic)
3. Select scopes: `repo`, `read:org`
4. Copy token to `.env`

### Step 5: Verify Environment

```bash
# Run verification script
python scripts/verify_external_services.py --config training/config_local_demo.yaml

# Expected output:
# ✓ Pinecone - Connected successfully (X indexes)
# ✓ W&B - Authenticated as username
# ✓ GitHub - Authenticated as username
```

---

## Running the Demo

### Quick Start (Recommended)

**Windows:**
```powershell
.\scripts\run_local_demo.ps1
```

**Linux/Mac:**
```bash
python -m training.cli train --demo
```

### Manual Execution

```bash
# Step 1: Verify services
python scripts/verify_external_services.py --config training/config_local_demo.yaml

# Step 2: Run demo training
python -m training.cli train --demo

# Step 3: View results
python -m training.cli monitor --static --config training/config_local_demo.yaml
```

### Advanced Options

```bash
# Skip service verification (not recommended)
python -m training.cli train --demo --skip-verification

# Enable verbose logging
python -m training.cli train --demo --log-level DEBUG

# Run specific phase only
python -m training.cli train --demo --phase mcts_self_play

# Custom output location
python -m training.cli train --demo --output results/demo_results.json
```

---

## Understanding Demo Mode

### Configuration Differences

| Parameter | Full Training | Demo Mode |
|-----------|--------------|-----------|
| **DABStep Samples** | 450+ | 100 |
| **PRIMUS Documents** | 674k | 500 |
| **Model Size** | deberta-v3-base (768d) | deberta-v3-small (512d) |
| **Batch Size** | 32 | 8 |
| **Epochs** | 10 | 3 |
| **LoRA Rank** | 16 | 8 |
| **MCTS Simulations** | 200 | 50 |
| **Decomposition Depth** | 5 | 3 |
| **Duration** | ~152 hours | ~45 minutes |

### Training Phases

The demo runs all 5 training phases:

```
Phase 1: Base Pre-training         (10 minutes)
    ↓
Phase 2: Instruction Fine-tuning   (8 minutes)
    ↓
Phase 3: MCTS Self-play            (15 minutes)
    ↓
Phase 4: Meta-controller Training  (7 minutes)
    ↓
Phase 5: Evaluation                (5 minutes)
```

### Memory Optimization

Demo mode includes several optimizations for 16GB VRAM:

1. **Gradient Checkpointing**: Trades compute for memory
2. **Mixed Precision (FP16)**: Reduces memory by ~50%
3. **Smaller Batch Size**: 8 instead of 32
4. **Reduced LoRA Rank**: 8 instead of 16
5. **Cache Clearing**: Empties CUDA cache every 10 steps

### Expected Metrics

Demo mode uses relaxed success criteria:

| Metric | Production Target | Demo Target |
|--------|------------------|-------------|
| HRM Accuracy | 85% | 70% |
| TRM Avg Iterations | 3.0 | 2.5 |
| MCTS Win Rate | 75% | 60% |
| RAG Precision@10 | 90% | 75% |
| Router Accuracy | 80% | 65% |

---

## Interpreting Results

### Console Output

During training, you'll see:

```
================================================================================
DEMO MODE: Local 16GB GPU Verification Training
================================================================================

Step 1/3: Verifying external services...
--------------------------------------------------------------------------------
✓ All critical services verified successfully

Step 2/3: Checking GPU availability...
--------------------------------------------------------------------------------
✓ GPU detected: NVIDIA GeForce RTX 4080
✓ GPU memory: 16.0 GB

Step 3/3: Starting demo training pipeline...
--------------------------------------------------------------------------------
[Phase 1/5] Base Pre-training
  Epoch 1/3: 100%|██████████| loss=2.341, acc=0.651
  ...
```

### Artifacts Generated

After completion, check these locations:

```
checkpoints/demo/
├── hrm_checkpoint_epoch_3.pt       # HRM model weights
├── trm_checkpoint_epoch_3.pt       # TRM model weights
├── mcts_checkpoint.pt              # MCTS networks
├── meta_controller_checkpoint.pt   # Router weights
└── pipeline_state.json             # Resumption state

logs/demo/
├── training_YYYYMMDD_HHMMSS.log   # Training logs
├── evaluation_results.json         # Benchmark results
└── metrics.json                    # Collected metrics

reports/
└── demo_benchmark.json             # DABStep benchmark report
```

### Weights & Biases Dashboard

1. Navigate to [https://wandb.ai/](https://wandb.ai/)
2. Find project: `multi-agent-mcts-demo`
3. View your run (auto-named with timestamp)

Key metrics to monitor:
- **Loss curves**: Should decrease steadily
- **GPU memory**: Should stay below 15GB
- **Training speed**: Steps per second
- **Validation metrics**: Accuracy, F1, etc.

### Evaluation Report

Check `reports/demo_benchmark.json`:

```json
{
  "hrm_accuracy": 0.72,
  "trm_avg_iterations": 2.3,
  "mcts_win_rate": 0.63,
  "rag_precision_at_10": 0.78,
  "router_accuracy": 0.67,
  "total_time_minutes": 42.3,
  "gpu_memory_peak_gb": 14.8
}
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size in `config_local_demo.yaml`: `batch_size: 4`
- Enable aggressive caching: `empty_cache_freq: 5`
- Close other GPU applications
- Use mixed precision: `fp16: true` (already enabled)

#### 2. Service Verification Failed

**Symptoms:**
```
✗ Pinecone - Invalid API key
```

**Solutions:**
- Check `.env` file exists and has correct values
- Verify API key is active (not expired)
- Test API key manually:
  ```bash
  curl https://api.pinecone.io/indexes -H "Api-Key: YOUR_KEY"
  ```
- Use `--skip-verification` as temporary workaround

#### 3. CUDA Not Available

**Symptoms:**
```
CUDA is not available!
```

**Solutions:**
- Reinstall PyTorch with CUDA:
  ```bash
  pip uninstall torch
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
- Verify CUDA installation: `nvidia-smi`
- Check `CUDA_VISIBLE_DEVICES` env var

#### 4. Dataset Download Slow

**Symptoms:**
- Phase 1 takes very long to start
- "Downloading" message stuck

**Solutions:**
- Use cached datasets if available
- Check internet connection
- Increase timeout in `data_pipeline.py`
- Download manually:
  ```bash
  python -c "from datasets import load_dataset; load_dataset('adyen/DABstep')"
  ```

#### 5. Permission Errors (Windows)

**Symptoms:**
```
PermissionError: [WinError 5] Access is denied
```

**Solutions:**
- Run PowerShell as Administrator
- Check file permissions in project directory
- Disable antivirus temporarily
- Use different checkpoint directory:
  ```yaml
  checkpointing:
    output_dir: "C:/Temp/demo_checkpoints"
  ```

### Performance Issues

#### Training Too Slow

- **Expected**: ~45 minutes for full demo
- **If slower**:
  - Check GPU utilization: `nvidia-smi dmon`
  - Verify CUDA vs CPU: `torch.cuda.is_available()`
  - Reduce num_workers in config
  - Enable pin_memory: `pin_memory: true`

#### High CPU Usage

- Reduce data loading workers:
  ```yaml
  memory_optimization:
    num_workers: 1
  ```

### Logging and Debugging

Enable verbose logging:

```bash
# Python logging
python -m training.cli train --demo --log-level DEBUG

# PowerShell verbose
.\scripts\run_local_demo.ps1 -Verbose

# Save logs to file
python -m training.cli train --demo --log-file debug.log
```

---

## Scaling to Production

### From Demo to Full Training

Once demo completes successfully, scale up:

1. **Use production config:**
   ```bash
   python -m training.cli train --config training/config.yaml
   ```

2. **Adjust hardware:**
   - Multi-GPU: Set `distributed.enabled: true`
   - Cloud: Use A100 80GB or H100 instances
   - Cluster: Configure SLURM or Kubernetes

3. **Update configuration:**
   ```yaml
   data:
     dabstep:
       max_samples: null  # Use all data
     primus_seed:
       max_documents: null  # Use all 674k docs

   training:
     batch_size: 32  # Increase batch size
     epochs: 10      # Full training epochs

   agents:
     hrm:
       model_name: "microsoft/deberta-v3-base"  # Larger model
       lora_rank: 16
   ```

4. **Enable advanced features:**
   ```yaml
   training:
     curriculum:
       enabled: true

   rag:
     hybrid_search:
       enabled: true

     embeddings:
       ensemble:
         enabled: true
   ```

### Cost Estimation

| Setup | Duration | Cost (AWS p3.2xlarge) | Cost (AWS p4d.24xlarge) |
|-------|----------|------------------------|-------------------------|
| Demo (16GB) | ~45 min | ~$0.50 | ~$5 |
| Full Training (Single GPU) | ~152 hours | ~$460 | ~$4,560 |
| Distributed (4x GPUs) | ~40 hours | ~$480 | ~$4,800 |

### Monitoring Production

For production training:

1. **Set up alerts:**
   ```yaml
   monitoring:
     alerts:
       loss_spike_threshold: 2.0
       oom_warning_threshold: 0.9
   ```

2. **Enable profiling:**
   ```yaml
   monitoring:
     profiling:
       enabled: true
       cpu_profiling: true
       memory_profiling: true
   ```

3. **Configure checkpointing:**
   ```yaml
   training:
     checkpointing:
       save_strategy: "steps"
       save_steps: 1000
       save_total_limit: 10
   ```

---

## FAQ

### Q: Can I run demo mode without a GPU?

**A:** No, demo mode requires CUDA-enabled GPU. The training pipeline uses mixed precision and gradient checkpointing which depend on GPU acceleration.

### Q: How much does the demo cost to run?

**A:** If you already have the hardware, the only costs are:
- **Pinecone**: Free tier supports demo workload
- **W&B**: Free tier includes 100GB storage
- **GitHub**: No cost for token generation

Total: **$0** if using free tiers.

### Q: Can I pause and resume the demo?

**A:** Yes! The pipeline saves checkpoints every epoch. Resume with:
```bash
python -m training.cli train --demo --resume checkpoints/demo/pipeline_state.json
```

### Q: What if I don't have all API keys?

**A:** You can skip optional services (OpenAI, Neo4j) but **must have**:
- Pinecone (for RAG)
- W&B (for monitoring)
- GitHub (for data access)

### Q: Can I run on CPU only?

**A:** Not recommended. CPU training would take 50-100x longer (~40 hours instead of 40 minutes).

### Q: How do I interpret the benchmark scores?

**A:** Demo scores are expected to be lower than production due to reduced training. Focus on:
- ✅ All phases complete without errors
- ✅ Loss decreases over time
- ✅ Validation accuracy > 0.6
- ✅ No OOM errors

### Q: What if my GPU has only 12GB?

**A:** Try these modifications in `config_local_demo.yaml`:
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4

agents:
  hrm:
    model_name: "prajjwal1/bert-tiny"  # Even smaller model
```

### Q: Can I run multiple demos in parallel?

**A:** Not recommended on a single 16GB GPU. Each demo uses ~14-15GB VRAM. Use separate machines or run sequentially.

### Q: How do I clean up after demo?

**A:** Use the cleanup script:
```powershell
.\scripts\run_local_demo.ps1 -CleanArtifacts
```

Or manually:
```bash
rm -rf checkpoints/demo
rm -rf logs/demo
rm -rf cache/dabstep
rm -rf cache/primus_*
```

---

## Additional Resources

- **Main Documentation**: [README.md](../README.md)
- **Architecture**: [C4_ARCHITECTURE.md](C4_ARCHITECTURE.md)
- **Training Configuration**: [config_local_demo.yaml](../training/config_local_demo.yaml)
- **API Reference**: [API.md](API.md)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

### Support

- **Issues**: [GitHub Issues](https://github.com/your-org/langgraph_multi_agent_mcts/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/langgraph_multi_agent_mcts/discussions)
- **Discord**: [Community Discord](https://discord.gg/your-server)

---

## Changelog

### Version 1.0.0 (2025-01-20)

- Initial release of demo mode
- Support for 16GB GPU
- Automated service verification
- PowerShell and Python runners
- Comprehensive documentation

---

**Happy Training! 🚀**

For questions or issues, please open an issue on GitHub or reach out on Discord.
