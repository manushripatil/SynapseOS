"""
SynapseOS - Multi-seed Experiment Runner
========================================
Runs experiments across many random seeds and reports mean ± std.
This is the statistical backbone of Core Engine v1.
"""

from __future__ import annotations
from typing import Dict, List, Callable, Optional, Tuple
import numpy as np
import time
import os
import json

from simulator import Simulator
from evolution import anomaly_detection_task, pattern_recognition_task
from persistence import save_network


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def run_experiment_1_multi(
    n_seeds: int = 8,
    n_cells: int = 24,
    generations: int = 12,
    verbose: bool = True,
) -> Dict:
    """
    Experiment 1 across many seeds.
    Returns aggregated statistics.
    """
    static_accs = []
    best_accs = []
    final_accs = []
    improvements = []

    if verbose:
        print("\n" + "═" * 64)
        print(f"  EXPERIMENT 1 – Multi-seed ({n_seeds} runs)")
        print("═" * 64)

    for i, seed in enumerate(range(n_seeds)):
        sim_static = Simulator(n_cells=n_cells, seed=seed)
        acc_s, _ = anomaly_detection_task(sim_static.net, n_samples=40)
        static_accs.append(acc_s)

        sim = Simulator(n_cells=n_cells, seed=seed)
        history = sim.run_evolution(
            anomaly_detection_task,
            generations=generations,
            structural_every=5,
            verbose=False,
            fast=True,
        )
        best = max(h["accuracy"] for h in history)
        final = history[-1]["accuracy"]
        best_accs.append(best)
        final_accs.append(final)
        improvements.append(best - acc_s)

        if verbose:
            print(f"  Seed {seed:2d} | static={acc_s:.3f}  best={best:.3f}  "
                  f"final={final:.3f}  Δ={best - acc_s:+.3f}")

    s_mean, s_std = _mean_std(static_accs)
    b_mean, b_std = _mean_std(best_accs)
    f_mean, f_std = _mean_std(final_accs)
    i_mean, i_std = _mean_std(improvements)

    result = {
        "n_seeds": n_seeds,
        "static_acc_mean": s_mean,
        "static_acc_std": s_std,
        "best_acc_mean": b_mean,
        "best_acc_std": b_std,
        "final_acc_mean": f_mean,
        "final_acc_std": f_std,
        "improvement_mean": i_mean,
        "improvement_std": i_std,
        "raw": {
            "static": static_accs,
            "best": best_accs,
            "final": final_accs,
        },
    }

    if verbose:
        print("\n── Aggregated ─────────────────────────────────────")
        print(f"  Static accuracy     : {s_mean:.3f} ± {s_std:.3f}")
        print(f"  SynapseOS best      : {b_mean:.3f} ± {b_std:.3f}")
        print(f"  SynapseOS final     : {f_mean:.3f} ± {f_std:.3f}")
        print(f"  Mean improvement    : {i_mean:+.3f} ± {i_std:.3f}")

    return result


def run_experiment_2_multi(
    n_seeds: int = 6,
    n_cells: int = 32,
    n_damage: int = 4,
    verbose: bool = True,
) -> Dict:
    """Experiment 2 (Damage Recovery) across seeds."""
    pre_accs, imm_accs, post_accs, recovered_flags = [], [], [], []

    if verbose:
        print("\n" + "═" * 64)
        print(f"  EXPERIMENT 2 – Damage Recovery Multi-seed ({n_seeds} runs)")
        print("═" * 64)

    for seed in range(n_seeds):
        sim = Simulator(n_cells=n_cells, seed=seed + 100)
        result = sim.run_damage_recovery_experiment(
            task_fn=anomaly_detection_task,
            pre_generations=14,
            post_generations=18,
            n_damage=n_damage,
        )
        # Suppress the long prints inside by not using verbose there;
        # the method itself prints, so we accept some noise or we can silence later.
        pre_accs.append(result["pre_accuracy"])
        imm_accs.append(result["immediate_accuracy"])
        post_accs.append(result["post_accuracy"])
        recovered_flags.append(1.0 if result["recovered"] else 0.0)

        if verbose:
            print(f"  Seed {seed:2d} | pre={result['pre_accuracy']:.3f}  "
                  f"imm={result['immediate_accuracy']:.3f}  "
                  f"post={result['post_accuracy']:.3f}  "
                  f"recovered={result['recovered']}")

    p_mean, p_std = _mean_std(pre_accs)
    i_mean, i_std = _mean_std(imm_accs)
    o_mean, o_std = _mean_std(post_accs)
    rec_rate = float(np.mean(recovered_flags))

    result = {
        "n_seeds": n_seeds,
        "pre_acc_mean": p_mean,
        "pre_acc_std": p_std,
        "immediate_acc_mean": i_mean,
        "immediate_acc_std": i_std,
        "post_acc_mean": o_mean,
        "post_acc_std": o_std,
        "recovery_rate": rec_rate,
    }

    if verbose:
        print("\n── Aggregated ─────────────────────────────────────")
        print(f"  Pre-damage accuracy : {p_mean:.3f} ± {p_std:.3f}")
        print(f"  Immediate after     : {i_mean:.3f} ± {i_std:.3f}")
        print(f"  After recovery      : {o_mean:.3f} ± {o_std:.3f}")
        print(f"  Recovery rate       : {rec_rate:.0%} of runs")

    return result


def run_experiment_3_multi(
    n_seeds: int = 6,
    n_cells: int = 28,
    verbose: bool = True,
) -> Dict:
    """Experiment 3 (Environment Shift) across seeds."""
    acc_a_list, acc_b_list, adapted_flags = [], [], []

    if verbose:
        print("\n" + "═" * 64)
        print(f"  EXPERIMENT 3 – Environment Shift Multi-seed ({n_seeds} runs)")
        print("═" * 64)

    for seed in range(n_seeds):
        sim = Simulator(n_cells=n_cells, seed=seed + 200)
        result = sim.run_environment_shift(
            task_a=anomaly_detection_task,
            task_b=pattern_recognition_task,
            gens_a=14,
            gens_b=18,
        )
        acc_a_list.append(result["env_a_final_acc"])
        acc_b_list.append(result["env_b_final_acc"])
        adapted_flags.append(1.0 if result["adapted"] else 0.0)

        if verbose:
            print(f"  Seed {seed:2d} | EnvA={result['env_a_final_acc']:.3f}  "
                  f"EnvB={result['env_b_final_acc']:.3f}  "
                  f"adapted={result['adapted']}")

    a_mean, a_std = _mean_std(acc_a_list)
    b_mean, b_std = _mean_std(acc_b_list)
    adapt_rate = float(np.mean(adapted_flags))

    result = {
        "n_seeds": n_seeds,
        "env_a_mean": a_mean,
        "env_a_std": a_std,
        "env_b_mean": b_mean,
        "env_b_std": b_std,
        "adaptation_rate": adapt_rate,
    }

    if verbose:
        print("\n── Aggregated ─────────────────────────────────────")
        print(f"  Env A final accuracy : {a_mean:.3f} ± {a_std:.3f}")
        print(f"  Env B final accuracy : {b_mean:.3f} ± {b_std:.3f}")
        print(f"  Adaptation rate      : {adapt_rate:.0%} of runs")

    return result


def run_all_multi(n_seeds: int = 6, save_dir: str = "saved") -> Dict:
    """Full multi-seed suite + optional save of one good network."""
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()

    print("\n" + "█" * 64)
    print("  SYNAPSEOS CORE ENGINE v1 – MULTI-SEED SUITE")
    print("█" * 64)

    r1 = run_experiment_1_multi(n_seeds=n_seeds)
    r2 = run_experiment_2_multi(n_seeds=max(4, n_seeds - 2))
    r3 = run_experiment_3_multi(n_seeds=max(4, n_seeds - 2))

    # Save one representative network from a fresh short run
    sim = Simulator(n_cells=28, seed=42)
    sim.run_evolution(anomaly_detection_task, generations=12, verbose=False)
    path = save_network(sim.net, os.path.join(save_dir, "example_fabric.json"))
    print(f"\n  Example network saved → {path}")

    elapsed = time.time() - start
    print(f"\n  Total wall time: {elapsed:.1f}s")

    summary = {
        "experiment_1": r1,
        "experiment_2": r2,
        "experiment_3": r3,
        "elapsed_seconds": elapsed,
    }

    # Write summary JSON
    summary_path = os.path.join(save_dir, "multi_seed_summary.json")
    # Convert to JSON-serializable
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(summary_path, "w") as f:
        json.dump(_clean(summary), f, indent=2)
    print(f"  Summary written → {summary_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SynapseOS multi-seed runner")
    parser.add_argument("--seeds", type=int, default=6, help="Number of random seeds")
    parser.add_argument("--exp", type=int, choices=[0, 1, 2, 3], default=0,
                        help="0=all, 1/2/3=individual")
    args = parser.parse_args()

    if args.exp == 1:
        run_experiment_1_multi(n_seeds=args.seeds)
    elif args.exp == 2:
        run_experiment_2_multi(n_seeds=args.seeds)
    elif args.exp == 3:
        run_experiment_3_multi(n_seeds=args.seeds)
    else:
        run_all_multi(n_seeds=args.seeds)
