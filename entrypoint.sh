#!/bin/bash
set -e

echo "=========================================="
echo "Multi-Agent MCTS Training Pipeline"
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

echo "=========================================="
echo "Starting training pipeline..."
echo "=========================================="

exec "$@"
