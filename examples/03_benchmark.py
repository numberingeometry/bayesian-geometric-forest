"""Compatibility entry point for the declared benchmark protocol.

python examples/03_benchmark.py --profile quick
python examples/03_benchmark.py --profile standard --only clustering
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_benchmarks.py"), run_name="__main__")
