# SynapseOS

**A self-organising neuromorphic computing architecture**

Computational cells dynamically evolve their roles and connectivity to create task-specific processing structures.

> SynapseOS is a research-style software prototype: identical cells change roles, rewire, grow/shrink, mature under an energy budget, and can reorganise after damage.

---

## One-sentence description

SynapseOS is a self-organising neuromorphic computing architecture where computational cells dynamically evolve their roles and connectivity to create task-specific processing structures.

---


## Use as a library

```bash
cd synapseos
pip3 install -e ".[all]"
```

```python
from synapseos import Simulator, anomaly_detection_task, save_network

sim = Simulator(n_cells=24, seed=7)
sim.run_evolution(anomaly_detection_task, generations=15, verbose=False)
print(sim.net.summary())
save_network(sim.net, "my_fabric.json")
```

Core lives in `synapseos.core`. Demos and the web UI stay outside the engine:

- `experiments/` — runner, suite, killer demo, **benchmark**
- `visual/` — live web simulator

```bash
python3 experiments/benchmark.py --seeds 6 --cells 24
python3 experiments/demo_recovery.py
python3 examples/minimal.py
```

## Live demo

After deploy (e.g. Render free tier):

- Open your service URL (first load may take 30–60s while the free instance wakes)
- **Start** — live evolution
- **Damage** / **Killer Demo** — destroy cells and watch recovery
- Cells are coloured by dominant role; links thicken/thin as strength changes

Local visual UI:

```bash
pip3 install -r requirements.txt
python3 visual/app.py
# → http://127.0.0.1:5050
```

---

## What it does

| Capability | Behaviour |
|------------|-----------|
| **Roles** | Soft mix over processing / memory / routing / sensing / decision |
| **Connections** | Hebbian strengthen, decay when unused, structural add/prune |
| **Population** | Cells can be born or die (self-sizing fabric) |
| **Maturation** | Plasticity falls as cells specialise |
| **Energy** | Global budget; costly low-utility structure is discouraged |
| **Damage recovery** | Delete cells + links → re-evolve under boosted plasticity |
| **Task shift** | Same fabric can adapt when the evaluation task changes |

This is a **simulation prototype**, not a hardware chip or a large trained neural net.

---

## Project structure

```text
synapseos/
├── cell.py            # Cell + Connection (roles, maturity, Hebbian links)
├── network.py         # Adaptive cell fabric (activate, birth/death)
├── evolution.py       # Fitness, plasticity, population control, tasks
├── simulator.py       # Experiment orchestration + damage injection
├── persistence.py     # Save / load evolved networks (JSON)
├── visualiser.py      # ASCII + matplotlib helpers
├── experiments.py     # Single-run experiment suite
├── runner.py          # Multi-seed statistics (mean ± std)
├── demo_recovery.py   # Phase 4 killer damage-recovery demo
├── requirements.txt
├── Dockerfile / Procfile
└── visual/            # Live web simulator
    ├── app.py
    ├── wsgi.py
    ├── static/
    └── templates/
```

---

## Quick start

```bash
cd synapseos
pip3 install -r requirements.txt

# Killer demo (train → damage → recover)
python3 demo_recovery.py
python3 demo_recovery.py --cells 80 --damage 0.40

# Multi-seed stats (recommended for claims)
python3 runner.py --seeds 6
python3 runner.py --exp 1 --seeds 8

# Classic single-run suite
python3 experiments.py

# Live UI
python3 visual/app.py
```

---

## Coordination protocol (short)

1. **Activate** the fabric with a task input for a few message-passing steps.
2. Cells update **activity**; roles drift slowly toward useful specialisations.
3. **Connections** update from co-activity (Hebbian); weak/unused links decay and may be pruned.
4. **Fitness** ≈ accuracy − energy − complexity guides how aggressive plasticity is.
5. **Population**: idle/inefficient cells may die; under capacity + energy headroom, new cells may be born and lightly wired in.
6. **Maturity** reduces plasticity so specialised cells lock in more than juveniles.

### What "damage" does

Selected cells are **permanently removed**, along with all incident connections. Accuracy usually drops immediately. Recovery is the same evolutionary loop with **higher plasticity** (and optional birth/death): remaining cells re-specialise and rewire. A static network with no adaptation cannot reorganise.

---

## Save / load

```python
from persistence import save_network, load_network
from simulator import Simulator
from evolution import anomaly_detection_task

sim = Simulator(n_cells=24, seed=7)
sim.run_evolution(anomaly_detection_task, generations=15, verbose=False)
save_network(sim.net, "my_fabric.json")
net2 = load_network("my_fabric.json")
print(net2.summary())
```

---

## Dependencies

```bash
pip3 install -r requirements.txt
# core: numpy, matplotlib, flask
# deploy: gunicorn (see Dockerfile / Procfile)
```

---

## Roadmap status

| Phase | Content | Status |
|-------|---------|--------|
| **0** | Core engine (cells, evolution, save/load, multi-seed) | Done |
| **1** | Live visual simulator (web UI) | Done |
| **2** | Self-organising population (birth/death) | Done |
| **3** | Maturation, synaptic plasticity, energy economy | Done |
| **4** | Killer damage-recovery demo | Done |
| **5** | Packaging / docs / public presentation | In progress |

---

## Honest scope

SynapseOS demonstrates **architecture-level adaptation** in a controllable software fabric. Task accuracy numbers depend on the simplified evaluation tasks and random seeds; use `runner.py` for multi-seed means. It is intended as a student / research prototype for discussion of self-organisation, plasticity, and recovery — not a production ML system.
