#!/usr/bin/env python3
"""Health check script for Docker container"""
import sys

try:
    import torch

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    # Check GPU count
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("ERROR: No GPUs detected", file=sys.stderr)
        sys.exit(1)

    # Success
    print(f"OK: {gpu_count} GPU(s) available")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {gpu_name}")

    sys.exit(0)

except Exception as e:
    print(f"ERROR: Health check failed: {e}", file=sys.stderr)
    sys.exit(1)
