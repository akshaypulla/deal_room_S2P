"""
DealRoom v3 - Research-grade OpenEnv RL environment for enterprise B2B negotiation.
"""

try:
    from deal_room_S2P.environment.dealroom_v3 import DealRoomV3S2P
except ImportError:
    from environment.dealroom_v3 import DealRoomV3S2P

DealRoomEnvironment = DealRoomV3S2P

__all__ = ["DealRoomV3S2P", "DealRoomEnvironment"]
