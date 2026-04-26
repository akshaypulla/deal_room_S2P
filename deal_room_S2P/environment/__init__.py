"""
Environment module for DealRoom v3 - OpenEnv wrapper, lookahead simulator.
"""

from .lookahead import LookaheadSimulator, SimulationResult
from .dealroom_v3 import DealRoomV3S2P

DealRoomEnvironment = DealRoomV3S2P

__all__ = [
    "LookaheadSimulator",
    "SimulationResult",
    "DealRoomV3S2P",
    "DealRoomEnvironment",
]
