"""
SynapseOS - Evolution Engine
============================
Drives structural and role adaptation of the Adaptive Cell Fabric.

Fitness =
    Accuracy
  - λ_energy  * EnergyCost
  - λ_complex * ComplexityPenalty
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from copy import deepcopy

from .network import Network
from .cell import Cell


class EvolutionEngine:
    def __init__(
        self,
        network: Network,
        energy_weight: float = 0.08,
        complexity_weight: float = 0.04,
        mutation_rate: float = 0.10,
        elite_fraction: float = 0.25,
    ):
        self.net = network
        self.energy_weight = energy_weight
        self.complexity_weight = complexity_weight
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction

        self.history: List[Dict] = []
        self.best_fitness: float = -np.inf
        self.best_snapshot: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        task_fn: Callable[[Network], Tuple[float, float]],
        n_evals: int = 3,
    ) -> float:
        """
        task_fn(network) → (accuracy, energy)
        Runs the task several times and averages to reduce noise.
        Returns scalar fitness.
        """
        accs, engs = [], []
        for _ in range(max(1, n_evals)):
            a, e = task_fn(self.net)
            accs.append(a)
            engs.append(e)
        accuracy = float(np.mean(accs))
        energy = float(np.mean(engs))

        n_links = self.net.connection_count()
        complexity = n_links / max(1, len(self.net.cells) * 3)

        fitness = accuracy - self.energy_weight * energy - self.complexity_weight * complexity

        record = {
            "generation": self.net.generation,
            "fitness": fitness,
            "accuracy": accuracy,
            "energy": energy,
            "n_links": n_links,
            "avg_strength": self.net.average_strength(),
            "roles": self.net.role_distribution(),
        }
        self.history.append(record)

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_snapshot = {
                "roles": {cid: dict(c.role) for cid, c in self.net.cells.items()},
                "connections": [
                    (c.source_id, c.target_id, c.strength) for c in self.net.connections
                ],
            }

        return fitness

    # ------------------------------------------------------------------
    # Structural evolution operators
    # ------------------------------------------------------------------

    def evolve_roles(self, reward: float = 0.0) -> None:
        """Each cell adapts its role vector based on recent performance."""
        for cell in self.net.cells.values():
            # Stronger cells (higher activity) get positive local reward
            local_reward = reward * (0.5 + 0.5 * cell.activity)
            cell.adapt_roles(local_reward)

    def evolve_connections(self) -> Dict[str, int]:
        """
        Phase 3 synaptic plasticity:
        - Hebbian reinforce from co-activity
        - Decay unused links
        - Prune weak / expensive low-utility links
        - Occasional exploratory links
        """
        stats = {"reinforced": 0, "decayed": 0, "pruned": 0, "created": 0}

        # 1. Hebbian updates from recent activity
        for conn in self.net.connections:
            src = self.net.cells.get(conn.source_id)
            tgt = self.net.cells.get(conn.target_id)
            if src is None or tgt is None:
                continue
            pre = src.activity
            post = tgt.activity
            before = conn.strength
            conn.hebbian_update(pre, post)
            if conn.strength > before:
                stats["reinforced"] += 1
            else:
                conn.decay(0.02)
                stats["decayed"] += 1

        # 2. Prune weak links
        stats["pruned"] = self.net.prune_weak_connections(threshold=0.09)

        # 3. Energy-aware: drop expensive, rarely used links under budget pressure
        if self.net.energy_spent > self.net.energy_budget * 0.9:
            costly = sorted(
                self.net.connections,
                key=lambda c: c.maintenance_cost() / (c.usage_frequency + 0.05),
                reverse=True,
            )
            for conn in costly[: max(1, len(costly) // 15)]:
                self.net.remove_connection(conn)
                stats["pruned"] += 1

        # 4. Create exploratory connections
        n_cells = len(self.net.cells)
        n_new = max(0, int(n_cells * self.mutation_rate * 0.15))
        ids = list(self.net.cells.keys())
        for _ in range(n_new):
            if len(ids) < 2:
                break
            src, tgt = np.random.choice(ids, size=2, replace=False)
            if self.net.add_connection(src, tgt, strength=np.random.uniform(0.28, 0.48)):
                stats["created"] += 1

        return stats

    def structural_mutation(self) -> None:
        """Larger random rewiring (used occasionally)."""
        ids = list(self.net.cells.keys())
        if len(ids) < 3:
            return
        # Randomly drop a few links
        if self.net.connections and np.random.rand() < 0.4:
            n_drop = max(1, len(self.net.connections) // 12)
            for conn in np.random.choice(self.net.connections, size=min(n_drop, len(self.net.connections)), replace=False):
                self.net.remove_connection(conn)

        # Add a few random long-range links
        for _ in range(max(1, len(ids) // 10)):
            src, tgt = np.random.choice(ids, size=2, replace=False)
            self.net.add_connection(src, tgt, strength=np.random.uniform(0.3, 0.6))


    # ------------------------------------------------------------------
    # Phase 2 — Autonomous cell lifecycle
    # ------------------------------------------------------------------

    def evolve_population(self, reward: float = 0.0) -> dict:
        """
        Phase 3 population control with utility + energy economy.
        """
        stats = {"born": 0, "died": 0}
        n = len(self.net.cells)
        min_cells, max_cells = 12, 64

        # Maturation tick for every cell
        for cell in self.net.cells.values():
            cell.mature_step()
            cell.update_utility(reward)
            cell.lifetime_energy += cell.energy_used

        # Per-generation energy = recent cell usage + synapse maintenance
        cell_energy = sum(min(c.energy_used, 1.2) for c in self.net.cells.values())
        syn_energy = sum(c.maintenance_cost() for c in self.net.connections)
        self.net.energy_spent = cell_energy + syn_energy

        # --- Death: low utility, high cost, or idle ---
        if n > min_cells:
            # Rank by utility / (energy + 0.1) — kill inefficient cells
            ranked = sorted(
                self.net.cells.values(),
                key=lambda c: c.utility / (c.energy_used + 0.15),
            )
            kill_budget = 2 if (reward < 0.2 or self.net.energy_spent > self.net.energy_budget) else 1
            for cell in ranked[:kill_budget]:
                if len(self.net.cells) <= min_cells:
                    break
                if cell.utility < 0.12 and cell.age > 8:
                    if self.net.death_cell(cell.id):
                        stats["died"] += 1

        # --- Birth: when reward is good and energy allows ---
        energy_ok = self.net.energy_spent < self.net.energy_budget * 0.85
        if n < max_cells and reward > 0.22 and energy_ok:
            roles = self.net.role_distribution()
            weakest = min(roles, key=roles.get)
            preferred = weakest
            if reward > 0.5 and roles.get("processing", 0) < 0.22:
                preferred = "processing"
            elif reward > 0.5 and roles.get("decision", 0) < 0.18:
                preferred = "decision"

            if np.random.rand() < 0.30:
                cid = self.net.birth_cell(preferred_role=preferred, n_links=3)
                if cid:
                    stats["born"] += 1

        if n < max_cells and energy_ok and 0.1 < reward < 0.35 and np.random.rand() < 0.10:
            cid = self.net.birth_cell(n_links=2)
            if cid:
                stats["born"] += 1

        return stats

    # ------------------------------------------------------------------
    # Main evolutionary step
    # ------------------------------------------------------------------

    def step(
        self,
        task_fn: Callable[[Network], Tuple[float, float]],
        do_structural_mutation: bool = False,
        recovery_mode: bool = False,
        fast: bool = False,
    ) -> Dict:
        """One full generation of evaluation + adaptation."""
        # fast=True → single evaluation (for multi-seed runs)
        if fast:
            n_evals = 1
        else:
            n_evals = 3 if (self.best_fitness > 0.4) else 2
        fitness = self.evaluate(task_fn, n_evals=n_evals)

        # Reward signal
        if self.best_fitness > -np.inf:
            relative = fitness - self.best_fitness
            reward = np.clip(0.35 * fitness + 0.65 * relative, -0.5, 1.0)
        else:
            reward = np.clip(fitness, -0.4, 1.0)

        # When already good, reduce plasticity to protect the solution
        plasticity_scale = 0.45 if fitness > 0.55 else 0.75
        if recovery_mode:
            plasticity_scale = 1.3

        if recovery_mode:
            old_plast = {}
            for cell in self.net.cells.values():
                old_plast[cell.id] = cell.plasticity
                cell.plasticity = min(0.20, cell.plasticity * 2.5)
            self.evolve_roles(reward * plasticity_scale)
            for cell in self.net.cells.values():
                cell.plasticity = old_plast.get(cell.id, 0.06)
        else:
            self.evolve_roles(reward * plasticity_scale)

        # Connection plasticity (always)
        conn_stats = self.evolve_connections()

        # Phase 2: autonomous birth / death
        pop_stats = self.evolve_population(reward=float(reward))

        # Structural mutation only when exploring or recovering
        if recovery_mode or (do_structural_mutation and fitness < self.best_fitness + 0.05):
            self.structural_mutation()

        # Stronger elite protection when performance drops
        if fitness < self.best_fitness - 0.12 and self.best_snapshot is not None:
            for cid, roles in self.best_snapshot["roles"].items():
                if cid in self.net.cells and np.random.rand() < 0.55:
                    self.net.cells[cid].set_role_vector(roles)

        self.net.generation += 1

        return {
            "fitness": fitness,
            "conn_stats": conn_stats,
            "pop_stats": pop_stats,
            "summary": self.net.summary(),
        }

    def restore_best(self) -> None:
        """Rollback network to the best observed configuration (optional)."""
        if self.best_snapshot is None:
            return
        # Restore roles
        for cid, roles in self.best_snapshot["roles"].items():
            if cid in self.net.cells:
                self.net.cells[cid].set_role_vector(roles)
        # Rebuild connections from scratch
        for c in list(self.net.connections):
            self.net.remove_connection(c)
        for src, tgt, strength in self.best_snapshot["connections"]:
            if src in self.net.cells and tgt in self.net.cells:
                self.net.add_connection(src, tgt, strength)


# ------------------------------------------------------------------
# Simple task functions used by experiments
# ------------------------------------------------------------------

def anomaly_detection_task(net: Network, n_samples: int = 24) -> Tuple[float, float]:
    """
    Harder anomaly detection.
    Abnormal patterns contain a *relative* signature (high variance +
    elevated last dimension) that benefits from processing + decision roles
    rather than pure magnitude pass-through.
    """
    correct = 0
    total_energy = 0.0

    for _ in range(n_samples):
        is_abnormal = np.random.rand() < 0.42
        if is_abnormal:
            # Relative signature: high variance + rising tail
            base = np.random.uniform(0.25, 0.65, size=4)
            base[2] += np.random.uniform(0.25, 0.55)
            base[3] += np.random.uniform(0.30, 0.60)
            vec = base + np.random.normal(0, 0.07, size=4)
        else:
            # Smooth, lower variance, flatter
            vec = np.random.uniform(0.20, 0.55, size=4) + np.random.normal(0, 0.04, size=4)

        vec = np.clip(vec, 0.0, 1.7).astype(np.float32)

        acts = net.activate(vec, max_steps=2)
        out = net.get_output_vector(acts, n_dims=4)

        # Decision uses contrast features (encourages processing + decision cells)
        contrast = float(out[3] - out[0] + 0.6 * (out[2] - out[1]))
        magnitude = float(np.mean(np.abs(out)))
        score = 0.55 * contrast + 0.45 * magnitude
        prediction = 1 if score > 0.18 else 0
        label = 1 if is_abnormal else 0
        if prediction == label:
            correct += 1
        total_energy += net.total_energy

    accuracy = correct / n_samples
    avg_energy = total_energy / n_samples
    return accuracy, avg_energy


def pattern_recognition_task(net: Network, n_samples: int = 20) -> Tuple[float, float]:
    """
    Distinguish two simple patterns (A vs B).
    Pattern A: rising values
    Pattern B: alternating high-low
    """
    correct = 0
    total_energy = 0.0

    for _ in range(n_samples):
        label = np.random.randint(0, 2)
        if label == 0:  # rising
            vec = np.linspace(0.15, 0.95, 4).astype(np.float32) + np.random.normal(0, 0.04, 4)
        else:           # alternating
            vec = np.array([0.9, 0.15, 0.85, 0.2], dtype=np.float32) + np.random.normal(0, 0.04, 4)

        vec = np.clip(vec, 0, 1.6).astype(np.float32)
        acts = net.activate(vec, max_steps=2)
        out = net.get_output_vector(acts)

        # Decision based on slope-like feature in the output
        slope = float(out[-1] - out[0])
        prediction = 0 if slope > 0.05 else 1
        if prediction == label:
            correct += 1
        total_energy += net.total_energy

    accuracy = correct / n_samples
    avg_energy = total_energy / n_samples
    return accuracy, avg_energy
