"""
SynapseOS — self-organising neuromorphic fabric.

Usage:
    from synapseos import Simulator, anomaly_detection_task
    from synapseos.core import Network, EvolutionEngine
"""

from .core import (
    Cell,
    Connection,
    Network,
    EvolutionEngine,
    Simulator,
    quick_demo,
    anomaly_detection_task,
    pattern_recognition_task,
    save_network,
    load_network,
    network_to_dict,
)

__version__ = "1.1.0"
__all__ = [
    "Cell",
    "Connection",
    "Network",
    "EvolutionEngine",
    "Simulator",
    "quick_demo",
    "anomaly_detection_task",
    "pattern_recognition_task",
    "save_network",
    "load_network",
    "network_to_dict",
]
