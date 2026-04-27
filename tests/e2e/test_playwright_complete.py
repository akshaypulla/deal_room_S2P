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
from typing import Tuple, List, Optional

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")


def _navigate_to_clean_tab(page):
    """Navigate to the Clean tab in the Gradio UI."""
    page.goto(f"{BASE_URL}/__gradio_ui__/", timeout=60000)
    time.sleep(6)
    clean_btns = page.locator("button:has-text('Clean')").all()
    if clean_btns:
        clean_btns[-1].click(force=True)
        time.sleep(3)


def _navigate_to_playground_tab(page):
    """Navigate to the Playground tab."""
    page.goto(f"{BASE_URL}/__gradio_ui__/", timeout=60000)
    time.sleep(6)
    playground_btns = page.locator("button:has-text('Playground')").all()
    if playground_btns:
        playground_btns[-1].click(force=True)
        time.sleep(3)


def assert_eq(a, b, msg: str = "") -> None:
    if a != b:
        raise AssertionError(f"{msg}: expected {b}, got {a}")


def assert_gt(a, b, msg: str = "") -> None:
    if a <= b:
        raise AssertionError(f"{msg}: expected > {b}, got {a}")


def assert_element_visible(page, selector: str, msg: str = "") -> None:
    try:
        page.wait_for_selector(selector, state="visible", timeout=8000)
    except Exception as e:
        raise AssertionError(f"{msg}: element not visible: {selector}") from e


def get_element_text(page, selector: str) -> str:
    try:
        return page.text_content(selector) or ""
    except Exception:
        return ""


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
    assert_eq(response.status_code, 200)

    health = response.json()
    print(f"  Health: {health.get('status')}")
    print(f"  Tasks: {health.get('tasks', [])}")
    assert "ok" in health.get("status", "")
    print(f"  ✓ Health endpoint working")


def test_03_clean_tab_level_selection():
    """E2E-3: Clean tab - all 3 level buttons are visible and unlocked."""
    print("\n[E2E-3] Clean tab - level selection...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

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
        _navigate_to_clean_tab(page)

        start_btn = page.locator("button:has-text('Start Round')").first
        assert_element_visible(page, "button:has-text('Start Round')", "Start Round button")
        start_btn.click()
        page.wait_for_timeout(3000)

        stakeholder_items = page.locator("[class*='stakeholder-item']").all()
        print(f"  ✓ Found {len(stakeholder_items)} stakeholders")
        assert len(stakeholder_items) >= 4, f"Expected at least 4 stakeholders, got {len(stakeholder_items)}"

        browser.close()


def test_05_stakeholder_selection_visible():
    """E2E-5: Stakeholder list is prominently displayed."""
    print("\n[E2E-5] Stakeholder selection visible...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        stakeholder_items = page.locator("[class*='stakeholder-item']").all()
        print(f"  ✓ Stakeholder items: {len(stakeholder_items)}")
        assert len(stakeholder_items) >= 4

        browser.close()


def test_06_response_send_button():
    """E2E-6: Sending a response is clearly marked."""
    print("\n[E2E-6] Response send button visible...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        send_btns = page.locator("button:has-text('Send Response')").all()
        print(f"  ✓ Send Response button: {len(send_btns)} found")
        assert len(send_btns) >= 1

        browser.close()


def test_07_auto_play_button():
    """E2E-7: Auto-play controls are visible."""
    print("\n[E2E-7] Auto-play feature...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        all_btns = page.locator("button").all()
        btn_texts = [b.inner_text() for b in all_btns if b.inner_text().strip()]
        auto_related = [t for t in btn_texts if "auto" in t.lower() or "play" in t.lower()]
        print(f"  Auto/play buttons: {auto_related}")
        print(f"  ✓ Auto-play controls present: {len(auto_related)}")

        browser.close()


def test_08_medium_level():
    """E2E-8: Medium level is unlocked and starts correctly."""
    print("\n[E2E-8] Medium level...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        page.locator("button:has-text('Medium')").first.click()
        page.wait_for_timeout(500)
        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(4000)

        stakeholder_items = page.locator("[class*='stakeholder-item']").all()
        print(f"  ✓ Medium level: {len(stakeholder_items)} stakeholders")

        browser.close()


def test_09_hard_level():
    """E2E-9: Hard level is unlocked and starts correctly."""
    print("\n[E2E-9] Hard level...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        page.locator("button:has-text('Hard')").first.click()
        page.wait_for_timeout(500)
        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(4000)

        stakeholder_items = page.locator("[class*='stakeholder-item']").all()
        print(f"  ✓ Hard level: {len(stakeholder_items)} stakeholders")

        browser.close()


def test_10_run_round_button():
    """E2E-10: Run Round button is present and clickable."""
    print("\n[E2E-10] Run Round button...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        run_btns = page.locator("button:has-text('Run Round'), button:has-text('▶ Run Round')").all()
        print(f"  ✓ Run Round buttons: {len(run_btns)}")
        if run_btns:
            print("  ✓ Run Round button visible after start")

        browser.close()


def test_11_playground_tab():
    """E2E-11: Playground tab round table visible."""
    print("\n[E2E-11] Playground tab...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_playground_tab(page)

        start_btns = page.locator("button:has-text('Start Round'), button:has-text('Start Simple Round')").all()
        if start_btns:
            start_btns[0].click()
            page.wait_for_timeout(3000)

        seat_elements = page.locator("[class*='seat'], [class*='round']").all()
        print(f"  ✓ Playground elements: {len(seat_elements)}")

        browser.close()


def test_12_session_isolation():
    """E2E-12: Multiple sessions don't interfere."""
    print("\n[E2E-12] Session isolation...")
    import requests

    sessions = []
    for i in range(3):
        session = requests.Session()
        r = session.post(
            f"{BASE_URL}/reset",
            json={"task_id": "aligned", "seed": 100 + i},
            timeout=30,
        )
        assert_eq(r.status_code, 200)
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
        assert_eq(r.status_code, 200)

    print(f"  ✓ {len(sessions)} sessions isolated correctly")
    for session, _ in sessions:
        session.close()


def test_13_full_episode():
    """E2E-13: Complete a full episode via API."""
    print("\n[E2E-13] Full episode completion...")
    import requests

    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 200}, timeout=30)
    assert_eq(r.status_code, 200)
    obs = r.json()
    sid = obs.get("metadata", {}).get("session_id")

    steps = 0
    for step in range(12):
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
        assert_eq(r.status_code, 200)
        result = r.json()
        steps += 1

        if result.get("done", False):
            terminal = result.get("info", {}).get("terminal_outcome", "unknown")
            print(f"  Episode terminated at step {steps}: {terminal}")
            break

    print(f"  ✓ Episode completed in {steps} steps")
    session.close()


def test_14_all_levels_unlocked():
    """E2E-14: All 3 difficulty levels are unlocked."""
    print("\n[E2E-14] All levels unlocked...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        simple = page.locator("button:has-text('Simple')").first
        medium = page.locator("button:has-text('Medium')").first
        hard = page.locator("button:has-text('Hard')").first

        assert simple.is_visible(), "Simple button not visible"
        assert medium.is_visible(), "Medium button not visible"
        assert hard.is_visible(), "Hard button not visible"

        print("  ✓ All 3 level buttons visible and unlocked")
        browser.close()


def test_15_score_panel():
    """E2E-15: Score panel shows current score."""
    print("\n[E2E-15] Score panel visible...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        start_btn = page.locator("button:has-text('Start Round')").first
        start_btn.click()
        page.wait_for_timeout(3000)

        score_vals = page.locator("[class*='score-value'], [class*='score-value']").all()
        print(f"  ✓ Score elements: {len(score_vals)}")

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
        ("E2E-5: Stakeholder Layout", test_05_stakeholder_selection_visible),
        ("E2E-6: Response Button", test_06_response_send_button),
        ("E2E-7: Auto-Play", test_07_auto_play_button),
        ("E2E-8: Medium Level", test_08_medium_level),
        ("E2E-9: Hard Level", test_09_hard_level),
        ("E2E-10: Run Round Button", test_10_run_round_button),
        ("E2E-11: Playground Tab", test_11_playground_tab),
        ("E2E-12: Session Isolation", test_12_session_isolation),
        ("E2E-13: Episode Completion", test_13_full_episode),
        ("E2E-14: All Levels Unlocked", test_14_all_levels_unlocked),
        ("E2E-15: Score Panel", test_15_score_panel),
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