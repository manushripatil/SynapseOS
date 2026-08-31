"""
SynapseOS — controlled multi-run benchmark
==========================================
Compares static vs adapted fabric, then damage vs recovery,
across several seeds. Writes CSV + prints a table.

    python3 experiments/benchmark.py
    python3 experiments/benchmark.py --seeds 8 --cells 24
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from synapseos.core import Simulator, Network, anomaly_detection_task


def eval_acc(net, n_samples: int = 32, repeats: int = 2) -> float:
    scores = []
    for _ in range(repeats):
        acc, _ = anomaly_detection_task(net, n_samples=n_samples)
        scores.append(acc)
    return float(np.mean(scores))


def run_one_seed(seed: int, n_cells: int, train_gens: int, recover_gens: int, damage_frac: float) -> dict:
    rng = np.random.default_rng(seed)

    # Static: no evolution
    static_net = Network(n_cells=n_cells, seed=seed)
    static_acc = eval_acc(static_net)

    # Adaptive train
    sim = Simulator(n_cells=n_cells, seed=seed)
    for g in range(train_gens):
        sim.evo.step(
            anomaly_detection_task,
            do_structural_mutation=(g % 5 == 0),
            fast=True,
        )
    trained_acc = eval_acc(sim.net)
    if sim.evo.history:
        trained_acc = max(trained_acc, max(h.get("accuracy", 0) for h in sim.evo.history))

    # Damage
    ids = list(sim.net.cells.keys())
    n_kill = max(1, int(len(ids) * damage_frac))
    victims = list(rng.choice(ids, size=min(n_kill, len(ids)), replace=False))
    for v in victims:
        sim.net.damage(v)
    damaged_acc = eval_acc(sim.net)

    # Recovery
    for g in range(recover_gens):
        sim.evo.step(
            anomaly_detection_task,
            do_structural_mutation=(g % 3 == 0),
            recovery_mode=True,
            fast=True,
        )
    recovered_acc = eval_acc(sim.net)
    if sim.evo.history:
        recovered_acc = max(recovered_acc, max(h.get("accuracy", 0) for h in sim.evo.history[-recover_gens:]))

    return {
        "seed": seed,
        "n_cells": n_cells,
        "n_killed": n_kill,
        "static": round(static_acc, 4),
        "trained": round(trained_acc, 4),
        "damaged": round(damaged_acc, 4),
        "recovered": round(recovered_acc, 4),
        "delta_adapt": round(trained_acc - static_acc, 4),
        "recover_ratio": round(recovered_acc / trained_acc, 4) if trained_acc > 1e-6 else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="SynapseOS controlled benchmark")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--cells", type=int, default=24)
    p.add_argument("--train", type=int, default=16)
    p.add_argument("--recover", type=int, default=16)
    p.add_argument("--damage", type=float, default=0.30)
    p.add_argument("--out", type=str, default="benchmark_results.csv")
    args = p.parse_args()

    print("SYNAPSEOS BENCHMARK")
    print(f"  seeds={args.seeds} cells={args.cells} train={args.train} "
          f"recover={args.recover} damage={args.damage:.0%}")
    rows = []
    t0 = time.time()
    for i in range(args.seeds):
        row = run_one_seed(i, args.cells, args.train, args.recover, args.damage)
        rows.append(row)
        print(f"  seed {i}: static={row['static']:.3f}  trained={row['trained']:.3f}  "
              f"dmg={row['damaged']:.3f}  rec={row['recovered']:.3f}  "
              f"Δadapt={row['delta_adapt']:+.3f}  rec/pre={row['recover_ratio']:.2f}")

    keys = list(rows[0].keys())
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys if k != "seed"}
    print("\n  MEAN across seeds")
    print(f"    static     {means['static']:.3f}")
    print(f"    trained    {means['trained']:.3f}   Δadapt={means['delta_adapt']:+.3f}")
    print(f"    damaged    {means['damaged']:.3f}")
    print(f"    recovered  {means['recovered']:.3f}   ratio={means['recover_ratio']:.2f}")
    print(f"  wall {time.time()-t0:.1f}s")

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out.resolve()}")


if __name__ == "__main__":
    main()
