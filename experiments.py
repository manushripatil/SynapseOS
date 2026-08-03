"""
SynapseOS - Experiments
=======================
Runnable scripts for the three core experiments defined in the project vision.
"""

from __future__ import annotations
import os
import sys
import numpy as np

# Ensure local imports work when run as script
sys.path.insert(0, os.path.dirname(__file__))

from simulator import Simulator
from evolution import anomaly_detection_task, pattern_recognition_task
from visualiser import (
    print_network_ascii,
    plot_fitness_history,
    plot_role_evolution,
    plot_connectivity_graph,
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def experiment_1_fixed_vs_adaptive(n_cells: int = 30, generations: int = 25):
    """
    Experiment 1 – Fixed Neural Network vs SynapseOS
    (Here we compare a randomly initialised static network
     against an evolved SynapseOS fabric on the same task.)
    """
    print("\n" + "█" * 64)
    print("  EXPERIMENT 1 – Fixed vs Adaptive Architecture")
    print("█" * 64)

    # --- Static baseline (no evolution) ---
    print("\n[Baseline] Static random network (no adaptation)")
    static = Simulator(n_cells=n_cells, seed=42)
    # Evaluate once
    acc_static, eng_static = anomaly_detection_task(static.net, n_samples=40)
    print(f"  Accuracy : {acc_static:.3f}")
    print(f"  Energy   : {eng_static:.3f}")
    print_network_ascii(static.net)

    # --- Adaptive SynapseOS ---
    print("\n[SynapseOS] Evolving adaptive fabric")
    adaptive = Simulator(n_cells=n_cells, seed=42)
    history = adaptive.run_evolution(
        anomaly_detection_task,
        generations=generations,
        structural_every=5,
        verbose=True,
    )
    final = history[-1]
    best_acc = max(h["accuracy"] for h in history)
    best_fit = max(h["fitness"] for h in history)
    print(f"\n  Best Accuracy  : {best_acc:.3f}")
    print(f"  Final Accuracy : {final['accuracy']:.3f}")
    print(f"  Final Energy   : {final['energy']:.3f}")
    print(f"  Best Fitness   : {best_fit:.3f}")
    print_network_ascii(adaptive.net)

    # Visuals
    plot_fitness_history(
        history,
        title="Experiment 1 – SynapseOS Evolution (Anomaly Detection)",
        save_path=os.path.join(OUTPUT_DIR, "exp1_fitness.png"),
    )
    plot_role_evolution(
        history,
        save_path=os.path.join(OUTPUT_DIR, "exp1_roles.png"),
    )
    plot_connectivity_graph(
        adaptive.net,
        title="Experiment 1 – Final Adaptive Fabric",
        save_path=os.path.join(OUTPUT_DIR, "exp1_graph.png"),
    )

    print("\n── Comparison ──────────────────────────────────────")
    print(f"  Static accuracy        : {acc_static:.3f}")
    print(f"  SynapseOS best accuracy: {best_acc:.3f}")
    print(f"  Improvement            : {best_acc - acc_static:+.3f}")
    return {
        "static_acc": acc_static,
        "adaptive_acc": best_acc,
        "history": history,
    }


def experiment_2_damage_recovery(n_cells: int = 36, n_damage: int = 4):
    """Experiment 2 – Damage Recovery."""
    print("\n" + "█" * 64)
    print("  EXPERIMENT 2 – Damage Recovery")
    print("█" * 64)

    sim = Simulator(n_cells=n_cells, seed=123)
    result = sim.run_damage_recovery_experiment(
        task_fn=anomaly_detection_task,
        pre_generations=18,
        post_generations=22,
        n_damage=n_damage,
    )

    print("\n── Recovery Summary ────────────────────────────────")
    print(f"  Pre-damage accuracy     : {result['pre_accuracy']:.3f}")
    print(f"  Immediate after damage  : {result['immediate_accuracy']:.3f}")
    print(f"  After recovery evolution: {result['post_accuracy']:.3f}")
    print(f"  Recovered (≥90% of pre) : {result['recovered']}")
    print(f"  Destroyed cells         : {result['victims']}")

    # Plot the combined history
    plot_fitness_history(
        sim.log,
        title="Experiment 2 – Damage & Recovery",
        save_path=os.path.join(OUTPUT_DIR, "exp2_recovery.png"),
    )
    plot_connectivity_graph(
        sim.net,
        title="Experiment 2 – Fabric after Recovery",
        save_path=os.path.join(OUTPUT_DIR, "exp2_graph.png"),
    )
    return result


def experiment_3_environment_shift(n_cells: int = 30):
    """Experiment 3 – Changing Environment."""
    print("\n" + "█" * 64)
    print("  EXPERIMENT 3 – Environment Shift")
    print("█" * 64)

    sim = Simulator(n_cells=n_cells, seed=99)
    result = sim.run_environment_shift(
        task_a=anomaly_detection_task,
        task_b=pattern_recognition_task,
        gens_a=18,
        gens_b=25,
    )

    print("\n── Adaptation Summary ──────────────────────────────")
    print(f"  Env A (Anomaly) final accuracy : {result['env_a_final_acc']:.3f}")
    print(f"  Env B (Pattern) final accuracy : {result['env_b_final_acc']:.3f}")
    print(f"  Successfully adapted           : {result['adapted']}")

    # Combined history for plotting
    combined = result["history_a"] + [
        {**h, "generation": h["generation"] + len(result["history_a"])}
        for h in result["history_b"]
    ]
    plot_fitness_history(
        combined,
        title="Experiment 3 – Environment Shift (A → B)",
        save_path=os.path.join(OUTPUT_DIR, "exp3_shift.png"),
    )
    plot_role_evolution(
        combined,
        save_path=os.path.join(OUTPUT_DIR, "exp3_roles.png"),
    )
    plot_connectivity_graph(
        sim.net,
        title="Experiment 3 – Fabric after Environment B",
        save_path=os.path.join(OUTPUT_DIR, "exp3_graph.png"),
    )
    return result


def run_all():
    """Execute the full experimental suite."""
    print("\n" + "═" * 64)
    print("  SYNAPSEOS – FULL EXPERIMENTAL SUITE")
    print("═" * 64)

    r1 = experiment_1_fixed_vs_adaptive(n_cells=28, generations=22)
    r2 = experiment_2_damage_recovery(n_cells=32, n_damage=4)
    r3 = experiment_3_environment_shift(n_cells=28)

    print("\n" + "═" * 64)
    print("  ALL EXPERIMENTS COMPLETE")
    print("═" * 64)
    print(f"  Exp1 – Adaptive advantage : {r1['adaptive_acc'] - r1['static_acc']:+.3f}")
    print(f"  Exp2 – Recovered          : {r2['recovered']}")
    print(f"  Exp3 – Adapted to new env : {r3['adapted']}")
    print(f"\n  Plots saved to: {os.path.abspath(OUTPUT_DIR)}")
    return r1, r2, r3


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SynapseOS Experiments")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 0], default=0,
                        help="Which experiment to run (0 = all)")
    args = parser.parse_args()

    if args.exp == 1:
        experiment_1_fixed_vs_adaptive()
    elif args.exp == 2:
        experiment_2_damage_recovery()
    elif args.exp == 3:
        experiment_3_environment_shift()
    else:
        run_all()
