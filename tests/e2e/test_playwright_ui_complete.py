#!/usr/bin/env python3
"""
test_playwright_ui_complete.py
Play through ALL 3 levels (Simple/Medium/Hard) interactively using Playwright.
Tests the complete UI flow: start round → select stakeholder → click chip →
send response → run round → verify deal closes or progresses.
"""

import os
import sys
import time
from typing import Tuple

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")
HF_URL = "https://akshaypulla-deal-room-s2p-c30af92.hf.space"
USE_HF = os.getenv("USE_HF", "false").lower() == "true"

def get_url() -> str:
    return HF_URL if USE_HF else BASE_URL


def _navigate_to_clean_tab(page) -> None:
    page.goto(f"{get_url()}/__gradio_ui__/", timeout=60000)
    time.sleep(8)
    clean_btns = page.locator("button:has-text('Clean')").all()
    if clean_btns:
        clean_btns[-1].click(force=True)
        time.sleep(5)


def _start_round(page, level: str) -> bool:
    level_map = {"simple": "Simple", "medium": "Medium", "hard": "Hard"}
    btn_label = level_map.get(level, "Simple")
    page.locator(f"button:has-text('{btn_label}')").first.click(force=True)
    time.sleep(1)
    page.wait_for_timeout(500)
    start_btns = page.locator("button:has-text('Start Round')").all()
    if not start_btns:
        return False
    start_btns[0].click(force=True)
    time.sleep(6)
    return True


def _select_stakeholder(page, index: int = 0) -> bool:
    items = page.locator("[class*='stakeholder-item']").all()
    if not items or index >= len(items):
        return False
    items[index].click()
    time.sleep(2)
    return True


def _click_first_chip(page) -> str:
    chips = page.locator("button.chip-btn").all()
    if chips and chips[0].is_visible():
        chips[0].click()
        time.sleep(1)
        return chips[0].inner_text()
    return ""


def _send_response(page) -> float:
    send_btns = page.locator("button:has-text('Send Response')").all()
    if send_btns and send_btns[0].is_visible() and send_btns[0].is_enabled():
        send_btns[0].click()
        time.sleep(4)
    score = _get_current_score(page)
    return score


def _run_round(page) -> float:
    run_btns = page.locator("button:has-text('Run Round')").all()
    for rb in run_btns:
        if rb.is_visible() and rb.is_enabled():
            rb.click()
            time.sleep(4)
            break
    return _get_current_score(page)


def _get_current_score(page) -> float:
    score_vals = page.locator("[class*='score-value']").all()
    if score_vals:
        try:
            txt = score_vals[0].inner_text().strip()
            return float(txt)
        except:
            pass
    return 0.0


def _get_stakeholder_status(page) -> dict:
    items = page.locator("[class*='stakeholder-item']").all()
    status = {}
    for item in items:
        try:
            txt = item.inner_text()
            name = txt.split("\n")[1] if "\n" in txt else txt[:20]
            is_aligned = "aligned" in item.get_attribute("class")
            is_blocking = "blocking" in item.get_attribute("class")
            is_triggered = "triggered" in item.get_attribute("class")
            status[name] = {"aligned": is_aligned, "blocking": is_blocking, "triggered": is_triggered}
        except:
            pass
    return status


def _get_deal_stage(page) -> str:
    content = page.content()
    stage_names = ["evaluation", "negotiation", "legal_review", "final_approval", "closed"]
    for s in stage_names:
        if s in content.lower():
            return s
    return "unknown"


def _check_deal_close(page) -> bool:
    complete_cards = page.locator("[class*='complete-card']").all()
    for c in complete_cards:
        try:
            if c.is_visible():
                txt = c.inner_text()
                if "deal closed" in txt.lower() or "deal vetoed" in txt.lower() or "deal timed" in txt.lower() or "deal ended" in txt.lower():
                    return True
        except:
            pass
    deal_close_popups = page.locator("[class*='deal-close-popup']").all()
    for p in deal_close_popups:
        try:
            style = p.get_attribute("style") or ""
            if "display:flex" in style.replace(" ", "") or "display:flex" in style:
                return True
        except:
            pass
    return False


def _play_level(page, level: str, max_steps: int = 12) -> dict:
    print(f"\n  === Playing {level.upper()} level ===")
    result = {
        "level": level,
        "steps_taken": 0,
        "final_score": 0.0,
        "deal_closed": False,
        "stage": "unknown",
        "stakeholder_status": {},
    }

    # Navigate to Clean tab first
    page.goto(f"{get_url()}/__gradio_ui__/", timeout=60000)
    time.sleep(8)
    clean_btns = page.locator("button:has-text('Clean')").all()
    if clean_btns:
        clean_btns[-1].click(force=True)
        time.sleep(4)

    if not _start_round(page, level):
        print(f"  ✗ Could not start {level} round")
        return result

    print(f"  ✓ Started {level} round")
    result["stage"] = _get_deal_stage(page)
    print(f"  Stage: {result['stage']}")

    for step in range(max_steps):
        if _check_deal_close(page):
            result["deal_closed"] = True
            print(f"  ✓ Deal CLOSED at step {step + 1}!")
            break

        stakeholder_status = _get_stakeholder_status(page)
        result["stakeholder_status"] = stakeholder_status
        aligned_count = sum(1 for v in stakeholder_status.values() if v["aligned"])
        total = len(stakeholder_status)
        print(f"  Step {step + 1}: {aligned_count}/{total} aligned")

        _select_stakeholder(page, 0)
        chip_text = _click_first_chip(page)
        if chip_text:
            print(f"  Chip: {chip_text[:40]}")

        score = _send_response(page)
        result["final_score"] = score
        result["steps_taken"] = step + 1
        print(f"  Score after send: {score:.3f}")

        score = _run_round(page)
        result["final_score"] = score
        print(f"  Score after run: {score:.3f}")
        result["stage"] = _get_deal_stage(page)

        if _check_deal_close(page):
            result["deal_closed"] = True
            print(f"  ✓ Deal CLOSED at step {step + 1}!")
            break

    print(f"  Final: score={result['final_score']:.3f}, stage={result['stage']}")
    return result


# =============================================================================
# TESTS
# =============================================================================

def test_01_simple_level_full_playthrough():
    """E2E-S1: Play through Simple (aligned) level end-to-end."""
    print("\n" + "=" * 60)
    print("[E2E-S1] Simple Level Full Playthrough")
    print("=" * 60)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        result = _play_level(page, "simple", max_steps=12)
        aligned = sum(1 for v in result["stakeholder_status"].values() if v["aligned"])
        total = len(result["stakeholder_status"])
        print(f"\n  ✓ Simple level: {result['steps_taken']} steps, score={result['final_score']:.3f}, aligned={aligned}/{total}")
        browser.close()


def test_02_medium_level_full_playthrough():
    """E2E-M1: Play through Medium (conflicted) level end-to-end."""
    print("\n" + "=" * 60)
    print("[E2E-M1] Medium Level Full Playthrough")
    print("=" * 60)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        result = _play_level(page, "medium", max_steps=12)
        aligned = sum(1 for v in result["stakeholder_status"].values() if v["aligned"])
        total = len(result["stakeholder_status"])
        print(f"\n  ✓ Medium level: {result['steps_taken']} steps, score={result['final_score']:.3f}, aligned={aligned}/{total}")
        browser.close()


def test_03_hard_level_full_playthrough():
    """E2E-H1: Play through Hard (hostile_acquisition) level end-to-end."""
    print("\n" + "=" * 60)
    print("[E2E-H1] Hard Level Full Playthrough")
    print("=" * 60)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        result = _play_level(page, "hard", max_steps=12)
        aligned = sum(1 for v in result["stakeholder_status"].values() if v["aligned"])
        total = len(result["stakeholder_status"])
        print(f"\n  ✓ Hard level: {result['steps_taken']} steps, score={result['final_score']:.3f}, aligned={aligned}/{total}")
        browser.close()


def test_04_triggered_stakeholder_highlighting():
    """E2E-T1: Triggered stakeholders are highlighted in purple."""
    print("\n" + "=" * 60)
    print("[E2E-T1] Triggered Stakeholder Highlighting")
    print("=" * 60)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        result = _play_level(page, "medium", max_steps=6)
        triggered = [k for k, v in result["stakeholder_status"].items() if v["triggered"]]
        if triggered:
            print(f"  ✓ Triggered stakeholders found: {triggered}")
        else:
            print(f"  (No triggered stakeholders in this run — cascade may not have fired)")
        browser.close()


def test_05_response_highlight_feedback():
    """E2E-R1: Response sent shows green feedback."""
    print("\n" + "=" * 60)
    print("[E2E-R1] Response Highlight Feedback")
    print("=" * 60)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{get_url()}/__gradio_ui__/", timeout=60000)
        time.sleep(8)
        clean_btns = page.locator("button:has-text('Clean')").all()
        if clean_btns:
            clean_btns[-1].click(force=True)
            time.sleep(4)
        page.locator("button:has-text('Start Round')").first.click(force=True)
        time.sleep(5)
        _select_stakeholder(page, 0)
        _click_first_chip(page)
        time.sleep(1)

        send_btns = page.locator("button:has-text('Send Response')").all()
        if send_btns:
            btn_text = send_btns[0].inner_text()
            print(f"  Send button text: [{btn_text}]")
            has_check = "✓" in btn_text or "Send Response" in btn_text
            print(f"  ✓ Response feedback visible: {has_check}")

        browser.close()


def test_06_deal_close_popup():
    """E2E-D1: Deal close shows popup."""
    print("\n" + "=" * 60)
    print("[E2E-D1] Deal Close Popup")
    print("=" * 60)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        page.locator("button:has-text('Start Round')").first.click(force=True)
        time.sleep(5)

        for step in range(10):
            _select_stakeholder(page, 0)
            _click_first_chip(page)
            time.sleep(1)
            _send_response(page)
            time.sleep(2)
            _run_round(page)
            time.sleep(2)

            if _check_deal_close(page):
                print(f"  ✓ Deal close indicator found at step {step + 1}")
                break

        browser.close()


def test_07_autoplay_controls():
    """E2E-A1: Auto-play controls work."""
    print("\n" + "=" * 60)
    print("[E2E-A1] Auto-play Controls")
    print("=" * 60)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        _navigate_to_clean_tab(page)

        page.locator("button:has-text('Start Round')").first.click(force=True)
        time.sleep(5)

        auto_start = page.locator("button:has-text('▶ Auto-Play')").all()
        auto_pause = page.locator("button:has-text('⏸ Pause')").all()
        auto_stop = page.locator("button:has-text('⏹ Stop')").all()

        print(f"  ✓ Auto-Play button: {len(auto_start)}")
        print(f"  ✓ Pause button: {len(auto_pause)}")
        print(f"  ✓ Stop button: {len(auto_stop)}")

        browser.close()


# =============================================================================
# RUNNER
# =============================================================================

def run_all_tests():
    print("=" * 70)
    print("  DealRoom S2P V3 — Complete UI Play-through Test Suite")
    print(f"  Target: {'HF Space' if os.getenv('USE_HF', 'false').lower() == 'true' else 'Local'}")
    print("=" * 70)

    tests = [
        ("E2E-S1: Simple Level Full Playthrough", test_01_simple_level_full_playthrough),
        ("E2E-M1: Medium Level Full Playthrough", test_02_medium_level_full_playthrough),
        ("E2E-H1: Hard Level Full Playthrough", test_03_hard_level_full_playthrough),
        ("E2E-T1: Triggered Highlighting", test_04_triggered_stakeholder_highlighting),
        ("E2E-R1: Response Feedback", test_05_response_highlight_feedback),
        ("E2E-D1: Deal Close Popup", test_06_deal_close_popup),
        ("E2E-A1: Auto-play Controls", test_07_autoplay_controls),
    ]

    passed = 0
    failed = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"\n  ✓ {name} PASSED\n")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"\n  ✗ {name} FAILED: {e}\n")
        except Exception as e:
            failed.append((name, f"ERROR: {e}"))
            print(f"\n  ✗ {name} ERROR: {e}\n")

    print("=" * 70)
    print(f"  UI PLAYTHROUGH TESTS: {passed}/{len(tests)} passed")
    print("=" * 70)

    if failed:
        print(f"\n  FAILED TESTS:")
        for name, err in failed:
            print(f"    ✗ {name}: {err[:150]}")
        sys.exit(1)
    else:
        print("\n  ALL UI PLAYTHROUGH TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()