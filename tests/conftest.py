from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DealRoomAction
from deal_room_S2P.environment.dealroom_v3 import DealRoomV3S2P


@pytest.fixture
def env() -> DealRoomV3S2P:
    return DealRoomV3S2P()


@pytest.fixture
def aligned_env(env: DealRoomV3S2P) -> DealRoomV3S2P:
    env.reset(seed=42, task_id="aligned")
    return env


@pytest.fixture
def simple_action() -> DealRoomAction:
    return DealRoomAction(
        action_type="direct_message",
        target="all",
        message="I want to understand the real blocker so I can tailor the proposal responsibly.",
    )