"""
SynapseOS - Cell Module
=======================
Each Cell is an identical adaptive computational unit.
It can dynamically change its role mix (Processing / Memory / Routing / ...)
and maintain local state + connections.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import uuid


@dataclass
class Connection:
    """Adaptive directed link between two cells (Phase 3 synaptic plasticity)."""
    source_id: str
    target_id: str
    strength: float = 0.5          # 0.0 – 1.0
    energy_cost: float = 0.1       # base cost of maintaining / using this link
    usage_frequency: float = 0.0   # how often it has been used recently
    age: int = 0                   # generations since creation
    last_pre: float = 0.0          # for simple STDP-like trace
    last_post: float = 0.0

    def maintenance_cost(self) -> float:
        """Energy to keep this synapse alive (stronger = slightly costlier)."""
        return self.energy_cost * (0.4 + 0.6 * self.strength)

    def decay(self, rate: float = 0.03) -> None:
        """Natural decay of unused connections."""
        unused = 1.0 - self.usage_frequency
        self.strength = max(0.0, self.strength - rate * unused)
        self.usage_frequency *= 0.88
        self.age += 1

    def reinforce(self, amount: float = 0.09) -> None:
        """Strengthen a useful connection (Hebbian)."""
        self.strength = min(1.0, self.strength + amount)
        self.usage_frequency = min(1.0, self.usage_frequency + 0.18)

    def hebbian_update(self, pre_activity: float, post_activity: float) -> None:
        """
        Simple Hebbian / STDP-style update:
        correlated pre+post → strengthen; pre without post → weaken slightly.
        """
        corr = pre_activity * post_activity
        if corr > 0.05:
            self.reinforce(0.04 + 0.08 * corr)
        elif pre_activity > 0.15 and post_activity < 0.05:
            self.decay(0.02)
        self.last_pre = pre_activity
        self.last_post = post_activity


class Cell:
    """
    A single adaptive computational cell.

    Role vector (always sums ≈ 1.0):
        processing : compute / transform
        memory     : store / recall
        routing    : forward / relay
        sensing    : interface with external input (optional)
        decision   : high-level choice / gating
    """

    ROLE_KEYS = ["processing", "memory", "routing", "sensing", "decision"]

    def __init__(self, cell_id: Optional[str] = None):
        self.id: str = cell_id or str(uuid.uuid4())[:8]

        # Soft role distribution (sums to 1.0)
        self.role: Dict[str, float] = {
            "processing": 0.4,
            "memory": 0.2,
            "routing": 0.2,
            "sensing": 0.1,
            "decision": 0.1,
        }
        self._normalize_roles()

        # Local working memory (simple key-value + short vector buffer)
        self.local_memory: Dict[str, float] = {}
        self.vector_buffer: np.ndarray = np.zeros(8, dtype=np.float32)

        # Activity & energy tracking
        self.activity: float = 0.0          # recent activation level
        self.energy_used: float = 0.0       # cumulative energy this generation
        self.last_output: Optional[np.ndarray] = None

        # Phase 3 — lifecycle & energy economy
        self.age: int = 0                   # generations since birth
        self.maturity: float = 0.0          # 0 = juvenile, 1 = fully specialised
        self.lifetime_energy: float = 0.0   # total energy ever spent
        self.utility: float = 0.0           # recent usefulness score

        # Adaptation parameters
        self.base_plasticity: float = 0.10  # plasticity at birth
        self.plasticity: float = 0.10
        self.fitness_contribution: float = 0.0

        # Incoming / outgoing connection references are managed by Network
        self.neighbours: Dict[str, float] = {}  # neighbour_id → strength

    def _normalize_roles(self) -> None:
        # Clamp to non-negative first
        for k in self.role:
            self.role[k] = max(0.0, self.role[k])
        total = sum(self.role.values())
        if total > 1e-8:
            for k in self.role:
                self.role[k] /= total
        else:
            # Fallback uniform
            for k in self.role:
                self.role[k] = 1.0 / len(self.role)

    def set_role_vector(self, new_roles: Dict[str, float]) -> None:
        """Directly set a new role distribution (will be normalised)."""
        for k in self.ROLE_KEYS:
            self.role[k] = max(0.0, new_roles.get(k, 0.0))
        self._normalize_roles()

    def adapt_roles(self, reward_signal: float, preferred: Optional[Dict[str, float]] = None) -> None:
        """
        Slowly shift role distribution toward more useful configurations.
        reward_signal > 0 → reinforce current dominant roles
        reward_signal < 0 → explore / diversify
        """
        if preferred is None:
            dominant = max(self.role, key=self.role.get)
            delta = self.plasticity * abs(reward_signal)
            if reward_signal >= 0:
                # Reinforce dominant
                for k in self.role:
                    if k == dominant:
                        self.role[k] += delta
                    else:
                        self.role[k] -= delta / (len(self.role) - 1)
            else:
                # Explore: push away from dominant a little
                for k in self.role:
                    if k == dominant:
                        self.role[k] -= delta * 0.6
                    else:
                        self.role[k] += delta * 0.6 / (len(self.role) - 1)
        else:
            # Move toward a preferred distribution
            for k in self.role:
                target = preferred.get(k, 0.0)
                self.role[k] += self.plasticity * (target - self.role[k])

        self._normalize_roles()

    def process(self, input_vec: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Core computational step.
        Behaviour is a soft mixture of the cell's current roles.
        All signals are clipped for numerical stability.
        """
        self.activity = 0.0
        self.energy_used = 0.0
        # Hard clip incoming signal to prevent explosion after topology changes
        x = np.clip(input_vec.astype(np.float32), -5.0, 5.0)
        output = np.zeros_like(x, dtype=np.float32)

        # --- Processing role ---
        if self.role["processing"] > 0.05:
            processed = np.tanh(x * (0.7 + 0.5 * self.role["processing"]))
            output += self.role["processing"] * processed
            self.energy_used += 0.12 * self.role["processing"]
            self.activity += self.role["processing"]

        # --- Memory role ---
        if self.role["memory"] > 0.05:
            buf_len = len(self.vector_buffer)
            src = x[:buf_len] if len(x) >= buf_len else np.pad(x, (0, buf_len - len(x)))
            self.vector_buffer = np.clip(0.7 * self.vector_buffer + 0.3 * src, -3.0, 3.0)
            if len(self.vector_buffer) >= len(x):
                recalled = self.vector_buffer[:len(x)]
            else:
                recalled = np.pad(self.vector_buffer, (0, len(x) - len(self.vector_buffer)))
            output += self.role["memory"] * recalled
            self.energy_used += 0.05 * self.role["memory"]
            self.activity += 0.6 * self.role["memory"]

        # --- Routing role ---
        if self.role["routing"] > 0.05:
            output += self.role["routing"] * (x * 0.92)
            self.energy_used += 0.03 * self.role["routing"]
            self.activity += 0.4 * self.role["routing"]

        # --- Sensing role ---
        if self.role["sensing"] > 0.05:
            # Gentle amplification only
            output += self.role["sensing"] * (x * 1.05)
            self.energy_used += 0.08 * self.role["sensing"]
            self.activity += self.role["sensing"]

        # --- Decision role ---
        if self.role["decision"] > 0.05:
            mean_x = float(np.mean(x))
            gate = 1.0 / (1.0 + np.exp(-np.clip(mean_x, -8.0, 8.0) * 2.5))
            output = output * (0.55 + 0.45 * gate * self.role["decision"] * 1.6)
            self.energy_used += 0.15 * self.role["decision"]
            self.activity += self.role["decision"]

        # Soft temperature / noise for exploration
        if temperature > 0:
            noise = np.random.normal(0, 0.015 * temperature, size=output.shape).astype(np.float32)
            output += noise

        # Final hard clip – prevents overflow propagation through the fabric
        output = np.clip(output, -4.0, 4.0)
        self.activity = float(np.clip(self.activity, 0.0, 2.0))
        self.energy_used = float(np.clip(self.energy_used, 0.0, 2.0))
        self.last_output = output
        return output

    def store(self, key: str, value: float) -> None:
        """Simple local memory write (used by memory-dominant cells)."""
        self.local_memory[key] = value
        self.energy_used += 0.02

    def recall(self, key: str, default: float = 0.0) -> float:
        return self.local_memory.get(key, default)

    def mature_step(self) -> None:
        """
        Phase 3: age the cell and reduce plasticity as it specialises.
        Juvenile cells explore; mature cells lock in their dominant role.
        """
        self.age += 1
        # Maturity grows with age and how peaked the role vector is
        dominant = max(self.role.values()) if self.role else 0.0
        target_maturity = min(1.0, (self.age / 40.0) * 0.5 + dominant * 0.5)
        self.maturity = 0.9 * self.maturity + 0.1 * target_maturity
        # Plasticity falls as maturity rises
        self.plasticity = self.base_plasticity * (1.0 - 0.75 * self.maturity)

    def update_utility(self, global_reward: float) -> None:
        """Track how useful this cell has been recently."""
        self.utility = 0.8 * self.utility + 0.2 * (self.activity * (0.5 + 0.5 * global_reward))

    def get_role_summary(self) -> str:
        parts = [f"{k[:4]}:{v:.0%}" for k, v in sorted(self.role.items(), key=lambda x: -x[1])]
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"Cell({self.id} | {self.get_role_summary()} | act={self.activity:.2f})"


# Quick self-test
if __name__ == "__main__":
    c = Cell("test01")
    print(c)
    x = np.array([0.2, 0.5, -0.1, 0.8], dtype=np.float32)
    y = c.process(x)
    print("Input :", x)
    print("Output:", y)
    print("Energy:", c.energy_used)
