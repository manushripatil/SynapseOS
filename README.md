# SynapseOS
**A Self-Organising Neuromorphic Computing Architecture**  
**Core Engine v1**

> Computational cells dynamically evolve their roles and connectivity  
> to create task-specific processing structures.

## One-sentence description
SynapseOS is a self-organising neuromorphic computing architecture where computational cells dynamically evolve their roles and connectivity to create task-specific processing structures.

## Project Structure

```
synapseos/
├── cell.py          # Adaptive computational cell + Connection
├── network.py       # Adaptive Cell Fabric
├── evolution.py     # Evolution Engine + task functions
├── simulator.py     # High-level experiment orchestration
├── visualiser.py    # ASCII + matplotlib visualisations
├── persistence.py   # Save / load evolved networks
├── runner.py        # Multi-seed statistical experiments
├── experiments.py   # Single-run experiments
└── README.md
```

## Quick Start

```bash
cd synapseos

# Single-run experiments
python3 experiments.py          # all three
python3 experiments.py --exp 2  # damage recovery only

# Multi-seed statistics (recommended)
python3 runner.py --seeds 6
python3 runner.py --exp 1 --seeds 8
```

## Core Engine v1 Features

- Adaptive cells with soft role vectors
- Dynamic connections (Hebbian + structural plasticity)
- Fitness-driven evolution with elite protection
- Numerical stability (clipped activations)
- **Save / load** evolved fabrics (JSON)
- **Multi-seed** experiment runner with mean ± std
- Damage recovery + environment-shift experiments

## Save / Load Example

```python
from network import Network
from persistence import save_network, load_network
from evolution import anomaly_detection_task
from simulator import Simulator

sim = Simulator(n_cells=24, seed=7)
sim.run_evolution(anomaly_detection_task, generations=15, verbose=False)

save_network(sim.net, "my_fabric.json")
net2 = load_network("my_fabric.json")
print(net2.summary())
```

## Dependencies

```bash
pip3 install numpy matplotlib
```

## Roadmap (high level)

- **Phase 0** – Core Engine v1 (this release)
- **Phase 1** – Live visual simulator (web UI)
- **Phase 2** – True self-organisation (cell birth/death)
- **Phase 3** – Stronger biological mechanisms
- **Phase 4** – Killer damage-recovery demo

## Phase 1 — Visual Simulator

```bash
cd synapseos
python3 visual/app.py
```

Then open **http://127.0.0.1:5050** in your browser.

Controls:
- **Start / Pause** — live evolution
- **Step** — one generation
- **Damage** — destroy 3 random cells and watch recovery
- **Task** — switch between Anomaly Detection and Pattern Recognition
- **Speed** — evolution rate

You will see cells coloured by dominant role, links thickening/thinning, and the accuracy sparkline updating live.

## Phase 2 — Self-Organising Population

Cells can now **appear and disappear** on their own:

- **Birth** — when fitness is good and capacity may help, a new cell is created with a role bias toward under-represented functions, then lightly wired in.
- **Death** — chronically idle or almost disconnected cells are removed (population stays above a minimum).

The fabric is no longer a fixed-size grid. It grows and shrinks as the task demands.

Watch it in the visual simulator: cell count changes live, and the message bar shows `+born/-died`.

## Phase 3 — Biological Mechanisms

1. **Synaptic plasticity** — Hebbian / co-activity updates; unused links decay; expensive low-utility links pruned under energy pressure.
2. **Cell maturation** — cells age; plasticity falls as they specialise; mature cells glow brighter in the visualiser.
3. **Energy economy** — global budget per generation; cell activity + synapse maintenance cost energy; inefficient cells are more likely to die; birth only when energy allows.

Watch **Maturity** climb in the UI as the fabric specialises.

## Phase 4 — Killer Demo (Damage Recovery)

Dramatic demonstration of self-repair:

```bash
python3 demo_recovery.py
python3 demo_recovery.py --cells 80 --damage 0.40
```

Story:
1. Train fabric to high accuracy
2. Destroy ~35% of cells
3. Accuracy collapses
4. Fabric reorganises (boosted plasticity + birth/death)
5. Compare against a static network that cannot recover

In the visual simulator, use the **☠ Killer Demo** button to destroy 35% of cells and auto-start recovery.
