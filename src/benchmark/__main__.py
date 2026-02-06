"""Allow running benchmark as: python -m src.benchmark"""

import sys

from src.benchmark.cli import main

try:
    main()
except KeyboardInterrupt:
    print("\nBenchmark interrupted.", file=sys.stderr)
    sys.exit(130)
