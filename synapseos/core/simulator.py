"""
SynapseOS - Simulator
=====================
High-level orchestration of experiments, damage injection,
environment changes and logging.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
import time

from .network import Network
from .evolution import EvolutionEngine, anomaly_detection_task, pattern_recognition_task


class Simulator:
    def __init__(self, n_cells: int = 36, seed: int = 42):
        self.net = Network(n_cells=n_cells, seed=seed)
        self.evo = EvolutionEngine(self.net)
        self.log: List[Dict] = []
        self.seed = seed

    def run_evolution(
        self,
        task_fn: Callable,
        generations: int = 30,
        structural_every: int = 8,
        verbose: bool = True,
        recovery_mode: bool = False,
        fast: bool = False,
    ) -> List[Dict]:
        """Run a multi-generation evolutionary loop."""
        results = []
        start = time.time()

        for g in range(generations):
            do_mut = (g > 0) and (g % structural_every == 0)
            step_result = self.evo.step(
                task_fn,
                do_structural_mutation=do_mut,
                recovery_mode=recovery_mode,
                fast=fast,
            )

            entry = {
                "generation": g,
                "fitness": step_result["fitness"],
                "accuracy": self.evo.history[-1]["accuracy"],
                "energy": self.evo.history[-1]["energy"],
                "n_links": self.evo.history[-1]["n_links"],
                "roles": self.evo.history[-1]["roles"],
            }
            results.append(entry)
            self.log.append(entry)

            if verbose and (g % 5 == 0 or g == generations - 1):
                print(f"[Gen {g:3d}] fitness={entry['fitness']:.3f}  "
                      f"acc={entry['accuracy']:.2f}  "
                      f"energy={entry['energy']:.2f}  "
                      f"links={entry['n_links']}")

        elapsed = time.time() - start
        if verbose:
            print(f"\nFinished {generations} generations in {elapsed:.1f}s")
            print(self.net.summary())
        return results

    def inject_damage(self, n_remove: int = 1, verbose: bool = True) -> List[str]:
        """Remove random cells (simulates hardware failure)."""
        ids = list(self.net.cells.keys())
        if n_remove >= len(ids):
            n_remove = max(0, len(ids) // 4)
        victims = list(np.random.choice(ids, size=n_remove, replace=False))
        for vid in victims:
            self.net.damage(vid)
            if verbose:
                print(f"  ✗ Cell {vid} destroyed")
        return victims

    def run_damage_recovery_experiment(
        self,
        task_fn: Callable = anomaly_detection_task,
        pre_generations: int = 15,
        post_generations: int = 20,
        n_damage: int = 3,
    ) -> Dict:
        """
        Experiment 2: Train → Damage → Recover.
        Returns before / after metrics.
        """
        print("=" * 60)
        print("EXPERIMENT 2 – Damage Recovery")
        print("=" * 60)

        print("\n[Phase A] Pre-damage evolution")
        pre = self.run_evolution(task_fn, generations=pre_generations, verbose=True)
        pre_acc = max(h["accuracy"] for h in pre)   # best during pre-training
        pre_fit = max(h["fitness"] for h in pre)

        print(f"\n[Phase B] Injecting damage ({n_damage} cells)...")
        victims = self.inject_damage(n_damage)

        # Immediate post-damage evaluation (no evolution yet)
        acc_after, eng_after = task_fn(self.net)
        print(f"  Immediate accuracy after damage: {acc_after:.2f}")

        print("\n[Phase C] Recovery evolution (boosted plasticity)")
        post = self.run_evolution(
            task_fn,
            generations=post_generations,
            structural_every=4,
            verbose=True,
            recovery_mode=True,
        )
        post_acc = max(h["accuracy"] for h in post)
        post_fit = max(h["fitness"] for h in post)

        return {
            "pre_accuracy": pre_acc,
            "pre_fitness": pre_fit,
            "immediate_accuracy": acc_after,
            "post_accuracy": post_acc,
            "post_fitness": post_fit,
            "victims": victims,
            "recovered": post_acc >= pre_acc * 0.85,  # slightly more lenient
        }

    def run_environment_shift(
        self,
        task_a: Callable = anomaly_detection_task,
        task_b: Callable = pattern_recognition_task,
        gens_a: int = 20,
        gens_b: int = 25,
    ) -> Dict:
        """
        Experiment 3: Train on Environment A, then switch to B.
        Can the fabric reorganise?
        """
        print("=" * 60)
        print("EXPERIMENT 3 – Changing Environment")
        print("=" * 60)

        print("\n[Environment A] Anomaly Detection")
        res_a = self.run_evolution(task_a, generations=gens_a, verbose=True)
        acc_a = res_a[-1]["accuracy"]

        print("\n[Environment B] Pattern Recognition (shift)")
        res_b = self.run_evolution(task_b, generations=gens_b, verbose=True)
        acc_b = res_b[-1]["accuracy"]

        return {
            "env_a_final_acc": acc_a,
            "env_b_final_acc": acc_b,
            "adapted": acc_b > 0.55,  # better than chance
            "history_a": res_a,
            "history_b": res_b,
        }


# Convenience runner
def quick_demo(n_cells: int = 25, generations: int = 20):
    sim = Simulator(n_cells=n_cells, seed=7)
    print(sim.net.summary())
    print("\nRunning anomaly detection evolution...")
    results = sim.run_evolution(anomaly_detection_task, generations=generations)
    return sim, results
