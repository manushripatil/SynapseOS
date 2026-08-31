"""
SynapseOS - Persistence
=======================
Save and load evolved Adaptive Cell Fabrics.
Supports JSON (human-readable) and a compact dict format.
"""

from __future__ import annotations
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

from .cell import Cell, Connection
from .network import Network


def network_to_dict(net: Network) -> Dict[str, Any]:
    """Serialize a Network into a plain Python dict."""
    cells_data = {}
    for cid, cell in net.cells.items():
        cells_data[cid] = {
            "role": dict(cell.role),
            "plasticity": cell.plasticity,
            "vector_buffer": cell.vector_buffer.tolist(),
            "local_memory": dict(cell.local_memory),
        }

    connections_data = [
        {
            "source_id": c.source_id,
            "target_id": c.target_id,
            "strength": c.strength,
            "energy_cost": c.energy_cost,
            "usage_frequency": c.usage_frequency,
            "age": c.age,
        }
        for c in net.connections
    ]

    return {
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "generation": net.generation,
        "n_cells": len(net.cells),
        "cells": cells_data,
        "connections": connections_data,
    }


def dict_to_network(data: Dict[str, Any]) -> Network:
    """Rebuild a Network from a serialized dict."""
    net = Network(n_cells=0, empty=True)
    net.generation = data.get("generation", 0)

    # Rebuild cells
    for cid, cdata in data["cells"].items():
        cell = Cell(cid)
        cell.set_role_vector(cdata["role"])
        cell.plasticity = cdata.get("plasticity", 0.06)
        import numpy as np
        buf = cdata.get("vector_buffer", [0.0] * 8)
        cell.vector_buffer = np.array(buf, dtype=np.float32)
        cell.local_memory = dict(cdata.get("local_memory", {}))
        net.cells[cid] = cell

    # Rebuild connections
    for cdata in data["connections"]:
        src = cdata["source_id"]
        tgt = cdata["target_id"]
        if src in net.cells and tgt in net.cells:
            conn = net.add_connection(src, tgt, strength=cdata["strength"])
            if conn is not None:
                conn.energy_cost = cdata.get("energy_cost", 0.1)
                conn.usage_frequency = cdata.get("usage_frequency", 0.0)
                conn.age = cdata.get("age", 0)

    return net


def save_network(net: Network, path: str) -> str:
    """Save network to a JSON file. Returns the path written."""
    data = network_to_dict(net)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_network(path: str) -> Network:
    """Load a network from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict_to_network(data)


def save_best(net: Network, evo, directory: str = "saved") -> Optional[str]:
    """Convenience: save the current best snapshot if one exists."""
    if evo.best_snapshot is None:
        return None
    # Temporarily restore best into a fresh network for clean save
    # (we save the live network + metadata instead)
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"synapseos_best_gen{net.generation}_{stamp}.json")
    return save_network(net, path)
