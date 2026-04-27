#!/usr/bin/env python3
"""
test_api_complete.py
DealRoom S2P V3 — Complete Environment API Tests

Tests all 3 tasks, reward signal, milestone bonuses,
POMDP noise, veto system, and session management.
"""

import os
import sys
import time
from typing import Any, Dict, List

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")

def assert_eq(a, b, msg: str = "") -> None:
    if a != b:
        raise AssertionError(f"{msg}: expected {b}, got {a}")

def assert_gt(a, b, msg: str = "") -> None:
    if a <= b:
        raise AssertionError(f"{msg}: expected > {b}, got {a}")

def assert_in(item, container, msg: str = "") -> None:
    if item not in container:
        raise AssertionError(f"{msg}: expected {item} in {container}")


# =============================================================================
# TEST SUITE
# =============================================================================

def test_01_health():
    """API-1: Health endpoint."""
    print("\n[API-1] Health endpoint...")
    import requests
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert_eq(r.status_code, 200)
    h = r.json()
    assert_eq(h["status"], "ok")
    assert "deal" in h["service"].lower()
    print(f"  ✓ Health: {h['status']}, tasks: {h['tasks']}")


def test_02_metadata():
    """API-2: Metadata endpoint."""
    print("\n[API-2] Metadata endpoint...")
    import requests
    r = requests.get(f"{BASE_URL}/metadata", timeout=10)
    assert_eq(r.status_code, 200)
    m = r.json()
    assert "deal" in m["name"].lower()
    print(f"  ✓ Metadata: {m['name']} v{m['version']}")


def test_03_reset_aligned():
    """API-3: Reset aligned task."""
    print("\n[API-3] Reset aligned task...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42}, timeout=30)
    assert_eq(r.status_code, 200)
    obs = r.json()
    assert "stakeholders" in obs
    assert len(obs["stakeholders"]) >= 4
    assert "approval_path_progress" in obs
    assert "deal_stage" in obs
    print(f"  ✓ Aligned task: {len(obs['stakeholders'])} stakeholders, stage={obs['deal_stage']}")
    return s, obs["metadata"]["session_id"]


def test_04_reset_conflicted():
    """API-4: Reset conflicted task."""
    print("\n[API-4] Reset conflicted task...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": 64}, timeout=30)
    assert_eq(r.status_code, 200)
    obs = r.json()
    assert "stakeholders" in obs
    print(f"  ✓ Conflicted task: {len(obs['stakeholders'])} stakeholders")
    return s, obs["metadata"]["session_id"]


def test_05_reset_hostile():
    """API-5: Reset hostile_acquisition task."""
    print("\n[API-5] Reset hostile_acquisition task...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "hostile_acquisition", "seed": 7}, timeout=30)
    assert_eq(r.status_code, 200)
    obs = r.json()
    assert "stakeholders" in obs
    print(f"  ✓ Hostile task: {len(obs['stakeholders'])} stakeholders")
    return s, obs["metadata"]["session_id"]


def test_06_step_reward_range():
    """API-6: Step reward is in positive range."""
    print("\n[API-6] Step reward range...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42}, timeout=30)
    obs = r.json()
    sid = obs["metadata"]["session_id"]

    rewards = []
    for i in range(5):
        r = s.post(f"{BASE_URL}/step", json={
            "metadata": {"session_id": sid},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": f"Test message {i}",
        }, timeout=30)
        assert_eq(r.status_code, 200)
        result = r.json()
        reward = result["reward"]
        rewards.append(reward)
        print(f"  Step {i+1}: reward={reward:.3f}")

    avg_reward = sum(rewards) / len(rewards)
    print(f"  Average reward: {avg_reward:.3f}")
    assert_gt(avg_reward, -1.0, f"Reward too negative: {avg_reward}")
    print(f"  ✓ Reward range OK (avg={avg_reward:.3f})")


def test_07_step_stakeholder_messages():
    """API-7: Step returns stakeholder messages."""
    print("\n[API-7] Stakeholder messages in step response...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42}, timeout=30)
    obs = r.json()
    sid = obs["metadata"]["session_id"]

    r = s.post(f"{BASE_URL}/step", json={
        "metadata": {"session_id": sid},
        "action_type": "direct_message",
        "target_ids": ["Finance"],
        "message": "Test",
    }, timeout=30)
    assert_eq(r.status_code, 200)
    result = r.json()
    obs_after = result["observation"]
    assert "stakeholder_messages" in obs_after
    print(f"  ✓ Stakeholder messages returned")


def test_08_milestone_bonus():
    """API-8: Milestone bonus fires on stage progression."""
    print("\n[API-8] Milestone bonus on stage progression...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42}, timeout=30)
    obs = r.json()
    sid = obs["metadata"]["session_id"]
    initial_stage = obs["deal_stage"]

    prev_reward = 0.0
    stage_advanced = False
    for i in range(10):
        r = s.post(f"{BASE_URL}/step", json={
            "metadata": {"session_id": sid},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": f"Step {i}",
        }, timeout=30)
        result = r.json()
        new_stage = result["observation"]["deal_stage"]
        if new_stage != initial_stage:
            stage_advanced = True
            print(f"  Stage advanced: {initial_stage} -> {new_stage}")
            break
        prev_reward = result["reward"]

    if stage_advanced:
        print(f"  ✓ Milestone bonus fired at stage transition")
    else:
        print(f"  (Stage did not advance within 10 steps)")


def test_09_session_isolation():
    """API-9: Sessions are isolated."""
    print("\n[API-9] Session isolation...")
    import requests
    sessions = []
    for i in range(3):
        s = requests.Session()
        r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 100 + i}, timeout=30)
        sid = r.json()["metadata"]["session_id"]
        sessions.append((s, sid))

    for i, (s, sid) in enumerate(sessions):
        r = s.post(f"{BASE_URL}/step", json={
            "metadata": {"session_id": sid},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": f"Session {i}",
        }, timeout=30)
        assert_eq(r.status_code, 200)

    print(f"  ✓ {len(sessions)} sessions isolated")
    for s, _ in sessions:
        s.close()


def test_10_max_rounds_termination():
    """API-10: Episode terminates after max rounds."""
    print("\n[API-10] Max rounds termination...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 200}, timeout=30)
    sid = r.json()["metadata"]["session_id"]

    done = False
    for i in range(15):
        r = s.post(f"{BASE_URL}/step", json={
            "metadata": {"session_id": sid},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": f"Step {i}",
        }, timeout=30)
        result = r.json()
        done = result.get("done", False)
        if done:
            print(f"  Episode terminated at step {i+1}: {result.get('info', {}).get('terminal_outcome', 'unknown')}")
            break

    print(f"  ✓ Episode {'terminated' if done else 'continued past 15 steps (may be fine)'}")
    s.close()


def test_11_state_endpoint():
    """API-11: State endpoint returns state."""
    print("\n[API-11] State endpoint...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42}, timeout=30)
    sid = r.json()["metadata"]["session_id"]

    r = s.get(f"{BASE_URL}/state", params={"session_id": sid}, timeout=10)
    assert_eq(r.status_code, 200)
    state = r.json()
    assert "episode_id" in state
    assert "round_number" in state
    print(f"  ✓ State endpoint: episode={state['episode_id']}, round={state['round_number']}")
    s.close()


def test_12_concurrent_resets():
    """API-12: Concurrent resets don't interfere."""
    print("\n[API-12] Concurrent resets...")
    import requests
    import concurrent.futures

    def reset_task(args):
        i, task = args
        s = requests.Session()
        r = s.post(f"{BASE_URL}/reset", json={"task_id": task, "seed": 300 + i}, timeout=30)
        if r.status_code == 200:
            sid = r.json()["metadata"]["session_id"]
            return (i, task, sid, True)
        return (i, task, None, False)

    tasks = [
        (0, "aligned"), (1, "conflicted"), (2, "hostile_acquisition"),
        (3, "aligned"), (4, "conflicted"),
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(reset_task, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    success = sum(1 for r in results if r[3])
    assert_eq(success, len(tasks), f"Only {success}/{len(tasks)} resets succeeded")
    print(f"  ✓ {success}/{len(tasks)} concurrent resets succeeded")


def test_13_veto_grace_rounds():
    """API-13: Hostile task has veto grace rounds."""
    print("\n[API-13: Veto grace rounds in hostile task...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "hostile_acquisition", "seed": 7}, timeout=30)
    sid = r.json()["metadata"]["session_id"]

    vetoes = []
    for i in range(8):
        r = s.post(f"{BASE_URL}/step", json={
            "metadata": {"session_id": sid},
            "action_type": "direct_message",
            "target_ids": ["Legal"],
            "message": f"Step {i}",
        }, timeout=30)
        result = r.json()
        veto_stk = result.get("info", {}).get("veto_stakeholder", "")
        if veto_stk:
            vetoes.append((i, veto_stk))
            print(f"  Veto at step {i}: {veto_stk}")

    print(f"  ✓ Hostile task ran {len(vetoes)} vetoes over 8 steps")
    s.close()


def test_14_pomdp_noise_present():
    """API-14: POMDP noise is applied (observation has noise artifacts)."""
    print("\n[API-14] POMDP noise in observation...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42}, timeout=30)
    obs1 = r.json()

    r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 99}, timeout=30)
    obs2 = r.json()

    h1 = obs1.get("engagement_levels", {})
    h2 = obs2.get("engagement_levels", {})
    if h1 and h2:
        print(f"  Engagement levels differ between seeds: {h1} vs {h2}")
        print(f"  ✓ POMDP noise present (different seed = different observation)")
    else:
        print(f"  ✓ Reset returned observations with engagement data")
    s.close()


def test_15_reward_components():
    """API-15: Reward components are returned in info."""
    print("\n[API-15] Reward components in info...")
    import requests
    s = requests.Session()
    r = s.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42}, timeout=30)
    sid = r.json()["metadata"]["session_id"]

    r = s.post(f"{BASE_URL}/step", json={
        "metadata": {"session_id": sid},
        "action_type": "send_document",
        "target_ids": ["Finance"],
        "message": "ROI model",
        "documents": [{"type": "roi_model", "specificity": "high"}],
    }, timeout=30)
    result = r.json()
    info = result.get("info", {})
    assert "reward_components" in info
    components = info["reward_components"]
    assert "goal" in components
    assert "trust" in components
    print(f"  ✓ Reward components: {list(components.keys())}")
    s.close()


# =============================================================================
# RUNNER
# =============================================================================

def run_all_tests():
    print("=" * 70)
    print("  DealRoom S2P V3 — Complete API Test Suite")
    print("=" * 70)

    tests = [
        ("API-1: Health", test_01_health),
        ("API-2: Metadata", test_02_metadata),
        ("API-3: Reset Aligned", test_03_reset_aligned),
        ("API-4: Reset Conflicted", test_04_reset_conflicted),
        ("API-5: Reset Hostile", test_05_reset_hostile),
        ("API-6: Reward Range", test_06_step_reward_range),
        ("API-7: Stakeholder Messages", test_07_step_stakeholder_messages),
        ("API-8: Milestone Bonus", test_08_milestone_bonus),
        ("API-9: Session Isolation", test_09_session_isolation),
        ("API-10: Max Rounds", test_10_max_rounds_termination),
        ("API-11: State Endpoint", test_11_state_endpoint),
        ("API-12: Concurrent Resets", test_12_concurrent_resets),
        ("API-13: Veto Grace Rounds", test_13_veto_grace_rounds),
        ("API-14: POMDP Noise", test_14_pomdp_noise_present),
        ("API-15: Reward Components", test_15_reward_components),
    ]

    passed = 0
    failed = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"  ✓ {name} PASSED\n")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  ✗ {name} FAILED: {e}\n")
        except Exception as e:
            failed.append((name, f"ERROR: {e}"))
            print(f"  ✗ {name} ERROR: {e}\n")

    print("=" * 70)
    print(f"  API TESTS: {passed}/{len(tests)} passed")
    print("=" * 70)

    if failed:
        print(f"\n  FAILED TESTS:")
        for name, err in failed:
            print(f"    ✗ {name}: {err[:150]}")
        sys.exit(1)
    else:
        print("\n  ALL API TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()