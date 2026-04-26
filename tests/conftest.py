from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DealRoomAction
from deal_room.environment.dealroom_v3 import DealRoomV3


@pytest.fixture
def env() -> DealRoomV3:
    return DealRoomV3()


@pytest.fixture
def aligned_env(env: DealRoomV3) -> DealRoomV3:
    env.reset(seed=42, task_id="aligned")
    return env


@pytest.fixture
def simple_action() -> DealRoomAction:
    return DealRoomAction(
        action_type="direct_message",
        target="all",
        message="I want to understand the real blocker so I can tailor the proposal responsibly.",
    )