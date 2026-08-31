"""Root wrapper. Prefer: python3 experiments/benchmark.py"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "experiments" / "benchmark.py"), run_name="__main__")
