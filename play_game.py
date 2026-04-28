"""
Playwright script to play through DealRoom S2P game across all levels.
Records all rewards and fixes UI bugs encountered.
"""

import json
import time
import subprocess

BASE_URL = "http://localhost:7860"
RESULTS_FILE = "game_results.md"

def log_result(level: str, step: int, action: str, reward: float, done: bool, stage: str):
    """Log result to console."""
    status = "DONE" if done else "IN_PROGRESS"
    print(f"[{level}] Step {step}: {action} | Reward: {reward:.4f} | Stage: {stage} | {status}")

def api_reset(task_id: str, seed: int = 42) -> dict:
    """Call reset API using curl."""
    cmd = [
        "curl", "-s", "-X", "POST",
        f"{BASE_URL}/reset",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"task_id": task_id, "seed": seed})
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def api_step(session_id: str, action: dict) -> dict:
    """Call step API using curl with cookie."""
    cmd = [
        "curl", "-s", "-X", "POST",
        f"{BASE_URL}/step",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(action),
        "-b", f"dealroom_session_id={session_id}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def api_state(session_id: str) -> dict:
    """Get current internal state."""
    cmd = ["curl", "-s", f"{BASE_URL}/state", "-b", f"dealroom_session_id={session_id}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def submit_proposal(session_id: str, stakeholders: list) -> dict:
    """Submit final proposal to close the deal."""
    action = {
        "action_type": "submit_proposal",
        "target": "all",
        "target_ids": stakeholders,
        "message": "Final proposal for approval with all compliance attestations.",
        "metadata": {"session_id": session_id},
        "submit_proposal": {
            "pricing_table": {
                "base_price": 180000,
                "discount_tiers": [],
                "payment_terms": "net_30",
                "optional_addons": []
            },
            "sla_commitments": {
                "uptime_guarantee": 0.99,
                "response_time_hours": 4,
                "resolution_time_hours": 24,
                "support_level": "named_support_lead",
                "penalties": None
            },
            "attached_documents": ["dpa", "security_cert"],
            "message": "Final proposal for approval.",
            "compliance_attestations": ["GDPR", "SOC2", "ISO27001"]
        }
    }
    return api_step(session_id, action)

def play_level_api(level: str, max_steps: int = 25) -> list:
    """Play through a level using documented walkthrough sequence."""
    task_id = level

    # Reset to get session
    reset_data = api_reset(task_id, seed=42)
    session_id = reset_data.get("metadata", {}).get("session_id")
    if not session_id:
        print(f"Reset failed for {level}")
        return []

    print(f"Started {level} with session {session_id}")

    results = []
    step = 0

    # Walkthrough sequence
    actions_sequence = [
        ("group_proposal", {"action_type": "group_proposal", "target": "all", "target_ids": ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"], "message": "I believe we have alignment.", "proposed_terms": {"price": 150000, "timeline_weeks": 12}, "metadata": {"session_id": session_id}}),
        ("send_document(dpa)_to_Legal", {"action_type": "send_document", "target": "Legal", "target_ids": ["Legal"], "message": "DPA", "documents": [{"type": "dpa", "specificity": "high"}], "metadata": {"session_id": session_id}}),
        ("send_document(security_cert)_to_TechLead", {"action_type": "send_document", "target": "TechLead", "target_ids": ["TechLead"], "message": "Security cert", "documents": [{"type": "security_cert", "specificity": "high"}], "metadata": {"session_id": session_id}}),
        ("send_document(roi_model)_to_Finance", {"action_type": "send_document", "target": "Finance", "target_ids": ["Finance"], "message": "ROI", "documents": [{"type": "roi_model", "specificity": "high"}], "metadata": {"session_id": session_id}}),
        ("send_document(implementation_timeline)_to_Operations", {"action_type": "send_document", "target": "Operations", "target_ids": ["Operations"], "message": "Timeline", "documents": [{"type": "implementation_timeline", "specificity": "high"}], "metadata": {"session_id": session_id}}),
        ("send_document(vendor_packet)_to_Procurement", {"action_type": "send_document", "target": "Procurement", "target_ids": ["Procurement"], "message": "Vendor packet", "documents": [{"type": "vendor_packet", "specificity": "high"}], "metadata": {"session_id": session_id}}),
        ("group_proposal_final", {"action_type": "group_proposal", "target": "all", "target_ids": ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"], "message": "Ready for final approval.", "proposed_terms": {"price": 180000, "timeline_weeks": 14}, "metadata": {"session_id": session_id}}),
    ]

    # Execute predefined sequence
    for action_name, action in actions_sequence:
        step += 1
        result = api_step(session_id, action)
        obs = result.get("observation", {})
        current_stage = obs.get("deal_stage", "?") if obs else "?"
        reward = result.get("reward", 0)
        done = result.get("done", False)
        terminal = result.get("info", {}).get("terminal_outcome", "")

        results.append({
            "level": level, "step": step, "action": action_name,
            "reward": reward, "done": done,
            "stage": current_stage, "terminal_outcome": terminal
        })
        log_result(level, step, action_name, reward, done, current_stage)

        if done:
            print(f"  -> Episode ended: {terminal}")
            return results

        # Check state after this step
        # If we're in legal_review, we need to immediately submit proposal in next step
        # because stage may advance to final_approval on the next action
        if current_stage == "legal_review":
            state = api_state(session_id)
            offer_state = state.get("offer_state", {})
            # If we have dpa and security but not proposal_submitted, submit now
            if offer_state.get("has_dpa") and offer_state.get("has_security_cert") and not offer_state.get("proposal_submitted"):
                step += 1
                result = submit_proposal(session_id, ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"])
                obs = result.get("observation", {})
                current_stage = obs.get("deal_stage", "?") if obs else "?"
                reward = result.get("reward", 0)
                done = result.get("done", False)
                terminal = result.get("info", {}).get("terminal_outcome", "")

                results.append({
                    "level": level, "step": step, "action": "submit_proposal",
                    "reward": reward, "done": done,
                    "stage": current_stage, "terminal_outcome": terminal
                })
                log_result(level, step, "submit_proposal", reward, done, current_stage)

                if done:
                    print(f"  -> Episode ended: {terminal}")
                    return results

    # Continue with auto-actions until done or max steps
    for extra_step in range(step + 1, max_steps + 1):
        # Check current state
        state = api_state(session_id)
        deal_stage = state.get("deal_stage", "unknown")
        offer_state = state.get("offer_state", {})
        proposal_submitted = offer_state.get("proposal_submitted", False)

        # If in final_approval stage and proposal not submitted, submit it
        if deal_stage == "final_approval" and not proposal_submitted:
            action_name = "submit_proposal"
            result = submit_proposal(session_id, ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"])
        else:
            # Otherwise send direct message to try to close
            action_name = "direct_message_to_close"
            action = {
                "action_type": "direct_message",
                "target": "ExecSponsor",
                "target_ids": ["ExecSponsor"],
                "message": "Please confirm final approval so we can close the deal.",
                "metadata": {"session_id": session_id}
            }
            result = api_step(session_id, action)

        obs = result.get("observation", {})
        current_stage = obs.get("deal_stage", "?") if obs else "?"
        reward = result.get("reward", 0)
        done = result.get("done", False)
        terminal = result.get("info", {}).get("terminal_outcome", "")

        results.append({
            "level": level, "step": extra_step, "action": action_name,
            "reward": reward, "done": done,
            "stage": current_stage, "terminal_outcome": terminal
        })
        log_result(level, extra_step, action_name, reward, done, current_stage)

        if done:
            print(f"  -> Episode ended: {terminal}")
            break

    return results

def write_results_table(all_results: list):
    """Write results to markdown table."""
    with open(RESULTS_FILE, "w") as f:
        f.write("# DealRoom S2P - Game Play Results\n\n")
        f.write(f"## Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for lvl in ["aligned", "conflicted", "hostile_acquisition"]:
            level_results = [r for r in all_results if r.get("level") == lvl]
            if not level_results:
                continue

            f.write(f"## {lvl.upper()} Level\n\n")
            f.write("| Step | Action | Reward | Done | Stage | Terminal Outcome |\n")
            f.write("|------|--------|--------|------|-------|------------------|\n")

            for r in level_results:
                f.write(f"| {r.get('step', '?')} | {r.get('action', '?')} | {r.get('reward', 0):.4f} | {r.get('done', False)} | {r.get('stage', '?')} | {r.get('terminal_outcome', '')} |\n")

            final_result = level_results[-1] if level_results else {}
            f.write(f"\n**Final Score**: {final_result.get('reward', 0):.4f}\n")
            f.write(f"**Terminal Outcome**: {final_result.get('terminal_outcome', 'unknown')}\n\n")

        f.write("\n## Summary\n\n")
        f.write("| Level | Final Reward | Outcome |\n")
        f.write("|-------|--------------|--------|\n")
        for lvl in ["aligned", "conflicted", "hostile_acquisition"]:
            level_results = [r for r in all_results if r.get("level") == lvl]
            if level_results:
                final = level_results[-1]
                f.write(f"| {lvl} | {final.get('reward', 0):.4f} | {final.get('terminal_outcome', 'unknown')} |\n")

def main():
    all_results = []

    print("=" * 60)
    print("DealRoom S2P - Playing All Levels (Fixed)")
    print("=" * 60)

    levels = ["aligned", "conflicted", "hostile_acquisition"]

    for level in levels:
        print(f"\n{'='*40}")
        print(f"Playing {level.upper()} level")
        print(f"{'='*40}")

        results = play_level_api(level, max_steps=25)
        all_results.extend(results)

        time.sleep(1)

    write_results_table(all_results)
    print(f"\nResults written to {RESULTS_FILE}")

    return all_results

if __name__ == "__main__":
    main()