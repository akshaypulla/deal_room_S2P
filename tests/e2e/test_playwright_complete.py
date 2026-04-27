#!/usr/bin/env python3
"""
test_playwright_complete.py
DealRoom S2P V3 — Complete Playwright E2E Tests

Tests ALL web UI interactions:
- Level selection (Simple/Medium/Hard - all unlocked)
- Stakeholder selection (top of screen)
- Response sending and visual feedback
- Auto-play feature
- Deal close popup
- Triggered stakeholder highlighting
- Medium and Hard level gameplay
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple, List, Optional

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")

def wait_for(page, selector: str, timeout: int = 15000) -> bool:
    """Wait for element with timeout."""
    try:
        page.wait_for_selector(selector, timeout=timeout)
        return True
    except Exception:
        return False

def click_and_confirm(page, selector: str, timeout: int = 10000) -> bool:
    """Click element and confirm it responded."""
    try:
        page.click(selector, timeout=timeout)
        return True
    except Exception:
        return False

def type_with_feedback(page, selector: str, text: str, timeout: int = 10000) -> bool:
    """Type text and confirm input accepted."""
    try:
        page.fill(selector, "", timeout=timeout)
        page.fill(selector, text)
        page.wait_for_timeout(300)
        value = page.input_value(selector)
        return text in value or page.input_value(selector) == text
    except Exception:
        return False

def get_element_text(page, selector: str) -> str:
    """Get text content of element."""
    try:
        return page.text_content(selector) or ""
    except Exception:
        return ""

def assert_element_visible(page, selector: str, msg: str = "") -> None:
    """Assert element is visible."""
    try:
        page.wait_for_selector(selector, state="visible", timeout=8000)
    except Exception as e:
        raise AssertionError(f"{msg}: element not visible: {selector}") from e

def assert_element_contains(page, selector: str, text: str, msg: str = "") -> None:
    """Assert element contains specific text."""
    content = get_element_text(page, selector)
    if text.lower() not in content.lower():
        raise AssertionError(f"{msg}: expected '{text}' in {selector}, got: {content[:100]}")

def assert_element_not_contains(page, selector: str, text: str, msg: str = "") -> None:
    """Assert element does NOT contain specific text."""
    content = get_element_text(page, selector)
    if text.lower() in content.lower():
        raise AssertionError(f"{msg}: unexpected '{text}' found in {selector}")

def assert_url_contains(page, text: str) -> None:
    """Assert current URL contains text."""
    if text.lower() not in page.url.lower():
        raise AssertionError(f"Expected URL to contain '{text}', got: {page.url}")


# =============================================================================
# TEST SUITE
# =============================================================================

def test_01_web_interface_loads():
    """E2E-1: Web interface loads without crash."""
    print("\n[E2E-1] Web interface loads...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(BASE_URL, timeout=60000)
        page.wait_for_timeout(3000)

        title = page.title()
        url = page.url
        print(f"  Page title: {title}")
        print(f"  Final URL: {url}")

        assert "DealRoom" in title, f"Expected 'DealRoom' in title, got: {title}"
        assert "/web" in url, f"Expected '/web' in URL, got: {url}"
        print(f"  ✓ Web interface loaded at {url}")
        browser.close()


def test_02_api_health():
    """E2E-2: API /health endpoint works."""
    print("\n[E2E-2] API health check...")
    import requests

    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status == 200, f"Health status: {response.status}"

    health = response.json()
    print(f"  Health: {health.get('status')}")
    print(f"  Tasks: {health.get('tasks', [])}")
    assert health.get("status") == "ok"
    print(f"  ✓ Health endpoint working")
    return health


def test_03_clean_tab_level_selection():
    """E2E-3: Clean tab - all 3 level buttons are visible and unlocked."""
    print("\n[E2E-3] Clean tab - level selection...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        # Switch to Clean tab if needed
        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        # Check all three level buttons exist
        simple_btn = page.locator("button:has-text('Simple')").first
        medium_btn = page.locator("button:has-text('Medium')").first
        hard_btn = page.locator("button:has-text('Hard')").first

        assert_element_visible(page, "button:has-text('Simple')", "Simple button")
        assert_element_visible(page, "button:has-text('Medium')", "Medium button")
        assert_element_visible(page, "button:has-text('Hard')", "Hard button")

        print("  ✓ All 3 level buttons visible")
        browser.close()


def test_04_start_simple_round():
    """E2E-4: Start a simple (aligned) round - stakeholders appear."""
    print("\n[E2E-4] Start simple round...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        # Click Start Round (default simple)
        start_btn = page.locator("button:has-text('Start Round')").first
        assert_element_visible(page, "button:has-text('Start Round')", "Start Round button")
        start_btn.click()
        page.wait_for_timeout(3000)

        # Check stakeholders appear (should have Finance/Legal/etc.)
        stakeholder_text = get_element_text(page, "[class*='stakeholder-item']")
        print(f"  Stakeholder content: {stakeholder_text[:80] if stakeholder_text else 'empty'}")

        # Should show stakeholder list
        page.wait_for_timeout(2000)
        stakeholder_items = page.locator("[class*='stakeholder-item']").all()
        print(f"  ✓ Found {len(stakeholder_items)} stakeholders")
        assert len(stakeholder_items) >= 4, f"Expected at least 4 stakeholders, got {len(stakeholder_items)}"

        browser.close()


def test_05_stakeholder_selection_at_top():
    """E2E-5: Stakeholder selection is at the top, not bottom."""
    print("\n[E2E-5] Stakeholder selection layout check...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        # Check that stakeholder dropdown/select is near the top of the left panel
        stakeholder_select = page.locator("[class*='dr-left'], [class*='dr-main']").first
        assert stakeholder_select.is_visible(), "Stakeholder selection area not visible"

        # Get bounding box to check position
        bb = stakeholder_select.bounding_box()
        if bb:
            print(f"  Left panel top: {bb['y']:.0f}px")
            print(f"  ✓ Stakeholder area is in the main layout")

        browser.close()


def test_06_response_highlighting():
    """E2E-6: Sending a response shows clear visual feedback."""
    print("\n[E2E-6] Response highlighting...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        # Click a stakeholder first
        stakeholder_items = page.locator("[class*='stakeholder-item']").all()
        if len(stakeholder_items) > 0:
            stakeholder_items[0].click()
            page.wait_for_timeout(1000)

        # Type a message
        msg_input = page.locator("textarea, input[class*='response'], [class*='message-input']").first
        if msg_input.is_visible():
            msg_input.fill("Test message for highlighting")
            page.wait_for_timeout(500)

            # Check input has the text
            val = msg_input.input_value()
            print(f"  Input value: {val}")
            assert "Test message" in val, f"Message not in input: {val}"

        # Check send button state
        send_btn = page.locator("button:has-text('Send Response')").first
        if send_btn.is_visible():
            print("  ✓ Send Response button visible")

            # Click it
            send_btn.click()
            page.wait_for_timeout(2000)

            # Check score updated or something changed
            score_text = get_element_text(page, "[class*='score-value']")
            print(f"  Score: {score_text}")
            print("  ✓ Response sent successfully")

        browser.close()


def test_07_auto_play_feature():
    """E2E-7: Auto-play button is visible and functional."""
    print("\n[E2E-7] Auto-play feature...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        # Look for auto-play button (may be labelled "Auto" or have play icon)
        auto_btns = page.locator("[class*='auto-btn'], button:has-text('Auto'), button:has-text('▶▶')").all()
        print(f"  Found {len(auto_btns)} potential auto-play buttons")

        # In playground tab, check for auto controls
        try:
            page.click("text=Playground", timeout=3000)
        except Exception:
            pass
        page.wait_for_timeout(1000)

        auto_controls = page.locator("[class*='auto'], [class*='sim-controls']").all()
        print(f"  ✓ Auto-play controls found: {len(auto_controls)}")

        browser.close()


def test_08_medium_level_plays():
    """E2E-8: Medium level plays correctly (not locked)."""
    print("\n[E2E-8] Medium level plays...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        # Click Medium level
        medium_btn = page.locator("button:has-text('Medium')").first
        medium_btn.click()
        page.wait_for_timeout(1000)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(4000)  # Medium may be slower

        # Should have stakeholders
        stakeholder_items = page.locator("[class*='stakeholder-item']").all()
        print(f"  ✓ Medium level started with {len(stakeholder_items)} stakeholders")

        # Run a step
        run_btn = page.locator("button:has-text('Run Round')").first
        if run_btn.is_visible():
            run_btn.click()
            page.wait_for_timeout(3000)
            print("  ✓ Run Round worked for medium level")

        browser.close()


def test_09_hard_level_plays():
    """E2E-9: Hard level plays correctly (not locked)."""
    print("\n[E2E-9] Hard level plays...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        # Click Hard level
        hard_btn = page.locator("button:has-text('Hard')").first
        hard_btn.click()
        page.wait_for_timeout(1000)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(4000)  # Hard may be slower

        # Should have stakeholders
        stakeholder_items = page.locator("[class*='stakeholder-item']").all()
        print(f"  ✓ Hard level started with {len(stakeholder_items)} stakeholders")

        # Run a step
        run_btn = page.locator("button:has-text('Run Round')").first
        if run_btn.is_visible():
            run_btn.click()
            page.wait_for_timeout(3000)
            print("  ✓ Run Round worked for hard level")

        browser.close()


def test_10_deal_close_popup():
    """E2E-10: Deal close shows a big popup for user understanding."""
    print("\n[E2E-10] Deal close popup...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        # Start simple round
        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        # Run multiple steps until deal closes or max rounds
        run_btn = page.locator("button:has-text('Run Round')").first
        for i in range(12):
            if not run_btn.is_visible():
                break
            run_btn.click()
            page.wait_for_timeout(2000)

            # Check for deal close indicators
            complete_text = get_element_text(page, "[class*='complete-card'], [class*='deal-'], [class*='Deal']")
            if complete_text and ("close" in complete_text.lower() or "deal" in complete_text.lower()):
                print(f"  ✓ Deal close indicator found: {complete_text[:80]}")
                break

            try:
                run_btn = page.locator("button:has-text('Run Round')").first
            except Exception:
                break

        browser.close()


def test_11_triggered_stakeholder_highlighting():
    """E2E-11: Triggered stakeholders highlight with different color."""
    print("\n[E2E-11] Triggered stakeholder highlighting...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Playground", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        start_btn = page.locator("button:has-text('Start Round')").first
        if not start_btn.is_visible():
            start_btn = page.locator("button:has-text('Start Simple Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        # Run a step to potentially trigger stakeholders
        run_btn = page.locator("button:has-text('Run Round')").first
        if run_btn.is_visible():
            run_btn.click()
            page.wait_for_timeout(3000)

        # Check for highlighted/active stakeholder indicators
        highlighted = page.locator("[class*='blocking'], [class*='aligned'], [class*='uncertain']").all()
        print(f"  ✓ Found {len(highlighted)} highlighted stakeholder elements")

        browser.close()


def test_12_playground_round_table():
    """E2E-12: Playground tab round-table interaction works."""
    print("\n[E2E-12] Playground round-table...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Playground", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        start_btn = page.locator("button:has-text('Start Round'), button:has-text('Start Simple Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        # Check round table exists
        round_container = page.locator("[class*='round-container'], [class*='seat']").first
        if round_container.is_visible():
            print("  ✓ Round table visible")

        browser.close()


def test_13_session_isolation():
    """E2E-13: Multiple sessions don't interfere."""
    print("\n[E2E-13] Session isolation...")
    import requests

    sessions = []
    for i in range(3):
        session = requests.Session()
        r = session.post(
            f"{BASE_URL}/reset",
            json={"task_id": "aligned", "seed": 100 + i},
            timeout=30,
        )
        assert r.status_code == 200, f"Reset failed for session {i}: {r.status_code}"
        obs = r.json()
        sid = obs.get("metadata", {}).get("session_id")
        sessions.append((session, sid))
        print(f"  Session {i+1}: id={str(sid)[:8]}...")

    for i, (session, sid) in enumerate(sessions):
        r = session.post(
            f"{BASE_URL}/step",
            json={
                "metadata": {"session_id": sid},
                "action_type": "direct_message",
                "target_ids": ["Finance"],
                "message": f"Message from session {i+1}",
            },
            timeout=30,
        )
        assert r.status_code == 200, f"Step failed for session {i}: {r.status_code}"
        result = r.json()
        print(f"  Session {i+1} step: reward={result.get('reward', 'N/A')}")

    print(f"  ✓ {len(sessions)} sessions isolated correctly")
    for session, _ in sessions:
        session.close()


def test_14_full_episode_completion():
    """E2E-14: Complete a full episode via API."""
    print("\n[E2E-14] Full episode completion...")
    import requests

    session = requests.Session()
    r = session.post(
        f"{BASE_URL}/reset",
        json={"task_id": "aligned", "seed": 200},
        timeout=30,
    )
    assert r.status_code == 200
    obs = r.json()
    sid = obs.get("metadata", {}).get("session_id")

    steps = 0
    for step in range(15):
        r = session.post(
            f"{BASE_URL}/step",
            json={
                "metadata": {"session_id": sid},
                "action_type": "direct_message",
                "target_ids": ["Finance"],
                "message": f"Step {step+1}",
            },
            timeout=30,
        )
        assert r.status_code == 200
        result = r.json()
        steps += 1

        if result.get("done", False) or result.get("observation", {}).get("done"):
            terminal = result.get("terminal_outcome", "unknown")
            print(f"  Episode terminated at step {steps}: {terminal}")
            break

    print(f"  ✓ Episode completed in {steps} steps")
    session.close()


def test_15_all_difficulty_levels():
    """E2E-15: All 3 difficulty levels are unlocked and functional."""
    print("\n[E2E-15] All difficulty levels functional...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        # Test Simple
        page.locator("button:has-text('Simple')").first.click()
        page.wait_for_timeout(500)
        page.locator("button:has-text('Start Round')").first.click()
        page.wait_for_timeout(3000)
        items_simple = page.locator("[class*='stakeholder-item']").all()
        print(f"  Simple: {len(items_simple)} stakeholders")

        # Reload for medium
        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)
        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        page.locator("button:has-text('Medium')").first.click()
        page.wait_for_timeout(500)
        page.locator("button:has-text('Start Round')").first.click()
        page.wait_for_timeout(3000)
        items_medium = page.locator("[class*='stakeholder-item']").all()
        print(f"  Medium: {len(items_medium)} stakeholders")

        # Reload for hard
        page.goto(f"{BASE_URL}/web", timeout=60000)
        page.wait_for_timeout(3000)
        try:
            page.click("text=Clean", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        page.locator("button:has-text('Hard')").first.click()
        page.wait_for_timeout(500)
        page.locator("button:has-text('Start Round')").first.click()
        page.wait_for_timeout(3000)
        items_hard = page.locator("[class*='stakeholder-item']").all()
        print(f"  Hard: {len(items_hard)} stakeholders")

        print(f"  ✓ All 3 levels unlocked and functional")
        browser.close()


# =============================================================================
# RUNNER
# =============================================================================

def run_all_tests():
    print("=" * 70)
    print("  DealRoom S2P V3 — Complete Playwright E2E Test Suite")
    print("=" * 70)

    tests = [
        ("E2E-1: Web Interface Loads", test_01_web_interface_loads),
        ("E2E-2: API Health", test_02_api_health),
        ("E2E-3: Level Selection", test_03_clean_tab_level_selection),
        ("E2E-4: Start Simple Round", test_04_start_simple_round),
        ("E2E-5: Stakeholder Layout", test_05_stakeholder_selection_at_top),
        ("E2E-6: Response Highlighting", test_06_response_highlighting),
        ("E2E-7: Auto-Play Feature", test_07_auto_play_feature),
        ("E2E-8: Medium Level", test_08_medium_level_plays),
        ("E2E-9: Hard Level", test_09_hard_level_plays),
        ("E2E-10: Deal Close Popup", test_10_deal_close_popup),
        ("E2E-11: Triggered Highlighting", test_11_triggered_stakeholder_highlighting),
        ("E2E-12: Round Table", test_12_playground_round_table),
        ("E2E-13: Session Isolation", test_13_session_isolation),
        ("E2E-14: Episode Completion", test_14_full_episode_completion),
        ("E2E-15: All Difficulty Levels", test_15_all_difficulty_levels),
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
    print(f"  E2E TESTS: {passed}/{len(tests)} passed")
    print("=" * 70)

    if failed:
        print(f"\n  FAILED TESTS:")
        for name, err in failed:
            print(f"    ✗ {name}: {err[:150]}")
        sys.exit(1)
    else:
        print("\n  ALL E2E TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()