"""
SynapseOS - Network / Adaptive Cell Fabric
==========================================
Manages a population of identical cells and the adaptive connections
between them. Provides topology queries, message passing and structural
mutation helpers used by the Evolution Engine.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from collections import defaultdict

from .cell import Cell, Connection


class Network:
    """
    The Adaptive Cell Fabric.

    Cells are stored in a dictionary for O(1) lookup.
    Connections are stored both as objects and as adjacency maps
    for efficient traversal.
    """

    def __init__(self, n_cells: int = 36, seed: Optional[int] = None, empty: bool = False):
        if seed is not None:
            np.random.seed(seed)

        self.cells: Dict[str, Cell] = {}
        self.connections: List[Connection] = []
        self.adj_out: Dict[str, List[Connection]] = defaultdict(list)
        self.adj_in: Dict[str, List[Connection]] = defaultdict(list)
        self.generation = 0
        self.total_energy = 0.0
        # Phase 3 — global energy budget
        self.energy_budget = 25.0          # available energy per generation
        self.energy_spent = 0.0

        if empty or n_cells <= 0:
            return

        # Create cells
        for i in range(n_cells):
            cid = f"C{i:03d}"
            cell = Cell(cid)
            # Slight role diversity at birth
            noise = np.random.dirichlet(np.ones(5) * 2.0)
            roles = {k: float(v) for k, v in zip(Cell.ROLE_KEYS, noise)}
            cell.set_role_vector(roles)
            self.cells[cid] = cell

        self._initial_wiring(density=0.18)

    # ------------------------------------------------------------------
    # Topology construction & maintenance
    # ------------------------------------------------------------------

    def _initial_wiring(self, density: float = 0.15) -> None:
        """Create a sparse random directed graph."""
        ids = list(self.cells.keys())
        n = len(ids)
        max_possible = n * (n - 1)
        target = int(max_possible * density)

        created = 0
        attempts = 0
        while created < target and attempts < target * 4:
            src, tgt = np.random.choice(ids, size=2, replace=False)
            if not self._has_connection(src, tgt):
                self.add_connection(src, tgt, strength=np.random.uniform(0.3, 0.7))
                created += 1
            attempts += 1

    def _has_connection(self, src: str, tgt: str) -> bool:
        return any(c.target_id == tgt for c in self.adj_out[src])

    def add_connection(self, src: str, tgt: str, strength: float = 0.5) -> Optional[Connection]:
        if src not in self.cells or tgt not in self.cells or src == tgt:
            return None
        if self._has_connection(src, tgt):
            return None
        conn = Connection(src, tgt, strength=float(np.clip(strength, 0.05, 1.0)))
        self.connections.append(conn)
        self.adj_out[src].append(conn)
        self.adj_in[tgt].append(conn)
        self.cells[src].neighbours[tgt] = conn.strength
        return conn

    def remove_connection(self, conn: Connection) -> None:
        if conn in self.connections:
            self.connections.remove(conn)
        if conn in self.adj_out[conn.source_id]:
            self.adj_out[conn.source_id].remove(conn)
        if conn in self.adj_in[conn.target_id]:
            self.adj_in[conn.target_id].remove(conn)
        self.cells[conn.source_id].neighbours.pop(conn.target_id, None)

    def prune_weak_connections(self, threshold: float = 0.12) -> int:
        """Remove connections that have become too weak."""
        to_remove = [c for c in self.connections if c.strength < threshold]
        for c in to_remove:
            self.remove_connection(c)
        return len(to_remove)

    # ------------------------------------------------------------------
    # Message passing / activation
    # ------------------------------------------------------------------

    def activate(self, input_data: np.ndarray, max_steps: int = 4) -> Dict[str, np.ndarray]:
        """
        Propagate an external input through the fabric for a few steps.
        Returns final activation of every cell.
        Messages and summed inputs are clipped for numerical stability.
        """
        safe_input = np.clip(input_data.astype(np.float32), -5.0, 5.0)

        # Inject input into cells that have significant sensing role
        activations: Dict[str, np.ndarray] = {}
        for cid, cell in self.cells.items():
            if cell.role["sensing"] > 0.22:
                activations[cid] = cell.process(safe_input)
            else:
                activations[cid] = np.zeros_like(safe_input)

        # Multi-step diffusion along connections
        for step in range(max_steps):
            new_act: Dict[str, np.ndarray] = {
                cid: np.zeros_like(safe_input) for cid in self.cells
            }

            for conn in self.connections:
                src_act = activations.get(conn.source_id)
                if src_act is None:
                    continue
                # Weighted + clipped message
                msg = np.clip(src_act * conn.strength, -3.0, 3.0)
                new_act[conn.target_id] = np.clip(
                    new_act[conn.target_id] + msg, -6.0, 6.0
                )
                conn.usage_frequency = min(1.0, conn.usage_frequency + 0.04)
                # Energy cost of communication
                if conn.source_id in self.cells:
                    self.cells[conn.source_id].energy_used += conn.energy_cost * 0.25

            # Each cell processes the summed incoming messages
            for cid, cell in self.cells.items():
                if np.any(new_act[cid]):
                    activations[cid] = cell.process(new_act[cid])

        # Aggregate energy (clipped)
        self.total_energy = float(sum(
            min(c.energy_used, 1.5) for c in self.cells.values()
        ))
        return activations

    def get_output_vector(self, activations: Dict[str, np.ndarray], n_dims: int = 4) -> np.ndarray:
        """
        Read-out: average the activity of decision-heavy and processing-heavy cells.
        """
        collected = []
        weights = []
        for cid, act in activations.items():
            cell = self.cells[cid]
            w = cell.role["decision"] * 1.5 + cell.role["processing"]
            if w > 0.15:
                collected.append(act[:n_dims])
                weights.append(w)
        if not collected:
            return np.zeros(n_dims, dtype=np.float32)
        weights = np.array(weights)
        weights /= weights.sum()
        return np.average(collected, axis=0, weights=weights).astype(np.float32)

    # ------------------------------------------------------------------
    # Structural queries & statistics
    # ------------------------------------------------------------------

    def connection_count(self) -> int:
        return len(self.connections)

    def average_strength(self) -> float:
        if not self.connections:
            return 0.0
        return float(np.mean([c.strength for c in self.connections]))

    def role_distribution(self) -> Dict[str, float]:
        """Average role mix across the whole fabric."""
        totals = {k: 0.0 for k in Cell.ROLE_KEYS}
        for cell in self.cells.values():
            for k, v in cell.role.items():
                totals[k] += v
        n = len(self.cells)
        return {k: v / n for k, v in totals.items()}

    def damage(self, cell_id: str) -> bool:
        """Simulate permanent failure of a cell (used in Experiment 2)."""
        if cell_id not in self.cells:
            return False
        # Remove all connections involving this cell
        to_remove = [c for c in self.connections
                     if c.source_id == cell_id or c.target_id == cell_id]
        for c in to_remove:
            self.remove_connection(c)
        del self.cells[cell_id]
        return True

    # ------------------------------------------------------------------
    # Phase 2 — Cell lifecycle (birth / death)
    # ------------------------------------------------------------------

    def _next_cell_id(self) -> str:
        existing = [int(cid[1:]) for cid in self.cells if cid.startswith("C") and cid[1:].isdigit()]
        n = (max(existing) + 1) if existing else 0
        return f"C{n:03d}"

    def birth_cell(self, preferred_role: str | None = None, n_links: int = 3) -> str | None:
        """
        Create a new cell and lightly wire it into the fabric.
        preferred_role biases the initial role vector.
        """
        cid = self._next_cell_id()
        cell = Cell(cid)
        if preferred_role and preferred_role in Cell.ROLE_KEYS:
            roles = {k: 0.08 for k in Cell.ROLE_KEYS}
            roles[preferred_role] = 0.68
            cell.set_role_vector(roles)
        else:
            noise = np.random.dirichlet(np.ones(5) * 1.5)
            cell.set_role_vector({k: float(v) for k, v in zip(Cell.ROLE_KEYS, noise)})
        self.cells[cid] = cell

        # Connect to a few existing cells
        ids = [i for i in self.cells if i != cid]
        if ids:
            targets = list(np.random.choice(ids, size=min(n_links, len(ids)), replace=False))
            for t in targets:
                self.add_connection(cid, t, strength=np.random.uniform(0.3, 0.55))
                if np.random.rand() < 0.5:
                    self.add_connection(t, cid, strength=np.random.uniform(0.25, 0.5))
        return cid

    def death_cell(self, cell_id: str) -> bool:
        """Remove a cell and all its connections (natural death)."""
        return self.damage(cell_id)

    def underperforming_cells(self, activity_threshold: float = 0.08, max_candidates: int = 5):
        """Return cell ids with very low recent activity (candidates for death)."""
        ranked = sorted(self.cells.values(), key=lambda c: c.activity)
        return [c.id for c in ranked if c.activity < activity_threshold][:max_candidates]

    def summary(self) -> str:
        roles = self.role_distribution()
        role_str = " | ".join(f"{k[:4]}:{v:.0%}" for k, v in roles.items())
        return (f"Network gen={self.generation} | cells={len(self.cells)} | "
                f"links={self.connection_count()} | avg_str={self.average_strength():.2f}\n"
                f"  Roles → {role_str}")


# Quick self-test
if __name__ == "__main__":
    net = Network(n_cells=16, seed=42)
    print(net.summary())
    x = np.array([0.2, 0.3, 0.2, 0.1], dtype=np.float32)
    acts = net.activate(x)
    out = net.get_output_vector(acts)
    print("Output vector:", out)
    print("Total energy:", net.total_energy)
