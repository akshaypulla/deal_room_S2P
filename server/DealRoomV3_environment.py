"""
DealRoomEnvironment — V3 wrapper.

This module re-exports DealRoomV3 as DealRoomEnvironment for backward compatibility
with existing server code that expects the DealRoomEnvironment name.
All business logic is in deal_room/environment/dealroom_v3.py.
"""

from deal_room.environment.dealroom_v3 import DealRoomV3

DealRoomEnvironment = DealRoomV3

__all__ = ["DealRoomEnvironment", "DealRoomV3"]