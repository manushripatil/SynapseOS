"""
SynapseOS – Self-Organising Neuromorphic Computing Architecture
Core Engine v1
"""

from .cell import Cell, Connection
from .network import Network
from .evolution import EvolutionEngine, anomaly_detection_task, pattern_recognition_task
from .simulator import Simulator, quick_demo
from .persistence import save_network, load_network, network_to_dict

__version__ = "1.0.0"
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
