"""SynapseOS core engine — cells, fabric, evolution, persistence."""

from .cell import Cell, Connection
from .network import Network
from .evolution import EvolutionEngine, anomaly_detection_task, pattern_recognition_task
from .simulator import Simulator, quick_demo
from .persistence import save_network, load_network, network_to_dict

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
