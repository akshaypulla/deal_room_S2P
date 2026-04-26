"""
DealRoomEnvironment — V3 wrapper.

This module re-exports DealRoomV3S2P as DealRoomEnvironment for backward compatibility
with existing server code that expects the DealRoomEnvironment name.
All business logic is in deal_room/environment/dealroom_v3.py.
"""

from deal_room_S2P.environment.dealroom_v3 import DealRoomV3S2P

DealRoomEnvironment = DealRoomV3S2P

__all__ = ["DealRoomEnvironment", "DealRoomV3S2P"]