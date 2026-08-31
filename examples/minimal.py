"""Minimal library usage — train a small fabric and print a summary."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synapseos import Simulator, anomaly_detection_task, save_network

sim = Simulator(n_cells=16, seed=3)
sim.run_evolution(anomaly_detection_task, generations=8, verbose=True)
print(sim.net.summary())
save_network(sim.net, "example_fabric.json")
print("saved example_fabric.json")
