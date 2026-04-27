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

def play_level_api(level: str, max_steps: int = 20) -> list:
    """Play through a level using documented walkthrough sequence."""
    task_id = level

    # Reset to get session
    reset_data = api_reset(task_id, seed=42)
    session_id = reset_data.get("metadata", {}).get("session_id")
    if not session_id:
        print(f"Reset failed for {level}")
        return []

    print(f"Started {level} with session {session_id}")

    # Define the walkthrough sequence based on walkthrough_data.py
    # For aligned: Finance -> Legal -> Operations -> Procurement -> group_proposal
    # For conflicted: similar but with hostile dynamics
    # For hostile_acquisition: different starting conditions

    results = []
    step = 0

    # Step 1: Initial group proposal to start the negotiation
    action = {
        "action_type": "group_proposal",
        "target": "all",
        "target_ids": ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"],
        "message": "I believe we have alignment to proceed with the negotiation.",
        "proposed_terms": {"price": 150000, "timeline_weeks": 12},
        "metadata": {"session_id": session_id}
    }
    result = api_step(session_id, action)
    results.append({
        "level": level, "step": 1, "action": "group_proposal",
        "reward": result.get("reward", 0), "done": result.get("done", False),
        "stage": result.get("observation", {}).get("deal_stage", "?"),
        "terminal_outcome": result.get("info", {}).get("terminal_outcome", "")
    })
    log_result(level, 1, "group_proposal", result.get("reward", 0), result.get("done", False), result.get("observation", {}).get("deal_stage", "?"))
    if result.get("done"):
        print(f"  -> Episode ended: {result.get('info', {}).get('terminal_outcome')}")
        return results

    # Step 2: Send DPA to Legal
    action = {
        "action_type": "send_document",
        "target": "Legal",
        "target_ids": ["Legal"],
        "message": "Here is the DPA with GDPR-aligned privacy commitments and review-ready clauses.",
        "documents": [{"type": "dpa", "specificity": "high"}],
        "metadata": {"session_id": session_id}
    }
    result = api_step(session_id, action)
    results.append({
        "level": level, "step": 2, "action": "send_document(dpa)_to_Legal",
        "reward": result.get("reward", 0), "done": result.get("done", False),
        "stage": result.get("observation", {}).get("deal_stage", "?"),
        "terminal_outcome": result.get("info", {}).get("terminal_outcome", "")
    })
    log_result(level, 2, "send_document(dpa)_to_Legal", result.get("reward", 0), result.get("done", False), result.get("observation", {}).get("deal_stage", "?"))
    if result.get("done"):
        print(f"  -> Episode ended: {result.get('info', {}).get('terminal_outcome')}")
        return results

    # Step 3: Send security cert to TechLead
    action = {
        "action_type": "send_document",
        "target": "TechLead",
        "target_ids": ["TechLead"],
        "message": "Here is our security certification and compliance documentation.",
        "documents": [{"type": "security_cert", "specificity": "high"}],
        "metadata": {"session_id": session_id}
    }
    result = api_step(session_id, action)
    results.append({
        "level": level, "step": 3, "action": "send_document(security_cert)_to_TechLead",
        "reward": result.get("reward", 0), "done": result.get("done", False),
        "stage": result.get("observation", {}).get("deal_stage", "?"),
        "terminal_outcome": result.get("info", {}).get("terminal_outcome", "")
    })
    log_result(level, 3, "send_document(security_cert)_to_TechLead", result.get("reward", 0), result.get("done", False), result.get("observation", {}).get("deal_stage", "?"))
    if result.get("done"):
        print(f"  -> Episode ended: {result.get('info', {}).get('terminal_outcome')}")
        return results

    # Step 4: Send ROI to Finance
    action = {
        "action_type": "send_document",
        "target": "Finance",
        "target_ids": ["Finance"],
        "message": "Here is our ROI analysis with explicit payback assumptions and downside cases.",
        "documents": [{"type": "roi_model", "specificity": "high"}],
        "metadata": {"session_id": session_id}
    }
    result = api_step(session_id, action)
    results.append({
        "level": level, "step": 4, "action": "send_document(roi_model)_to_Finance",
        "reward": result.get("reward", 0), "done": result.get("done", False),
        "stage": result.get("observation", {}).get("deal_stage", "?"),
        "terminal_outcome": result.get("info", {}).get("terminal_outcome", "")
    })
    log_result(level, 4, "send_document(roi_model)_to_Finance", result.get("reward", 0), result.get("done", False), result.get("observation", {}).get("deal_stage", "?"))
    if result.get("done"):
        print(f"  -> Episode ended: {result.get('info', {}).get('terminal_outcome')}")
        return results

    # Step 5: Send implementation timeline to Operations
    action = {
        "action_type": "send_document",
        "target": "Operations",
        "target_ids": ["Operations"],
        "message": "Here is our implementation timeline with milestones, owners, and delivery guardrails.",
        "documents": [{"type": "implementation_timeline", "specificity": "high"}],
        "metadata": {"session_id": session_id}
    }
    result = api_step(session_id, action)
    results.append({
        "level": level, "step": 5, "action": "send_document(implementation_timeline)_to_Operations",
        "reward": result.get("reward", 0), "done": result.get("done", False),
        "stage": result.get("observation", {}).get("deal_stage", "?"),
        "terminal_outcome": result.get("info", {}).get("terminal_outcome", "")
    })
    log_result(level, 5, "send_document(implementation_timeline)_to_Operations", result.get("reward", 0), result.get("done", False), result.get("observation", {}).get("deal_stage", "?"))
    if result.get("done"):
        print(f"  -> Episode ended: {result.get('info', {}).get('terminal_outcome')}")
        return results

    # Step 6: Send vendor packet to Procurement
    action = {
        "action_type": "send_document",
        "target": "Procurement",
        "target_ids": ["Procurement"],
        "message": "Here is our supplier onboarding packet including process, insurance, and vendor details.",
        "documents": [{"type": "vendor_packet", "specificity": "high"}],
        "metadata": {"session_id": session_id}
    }
    result = api_step(session_id, action)
    results.append({
        "level": level, "step": 6, "action": "send_document(vendor_packet)_to_Procurement",
        "reward": result.get("reward", 0), "done": result.get("done", False),
        "stage": result.get("observation", {}).get("deal_stage", "?"),
        "terminal_outcome": result.get("info", {}).get("terminal_outcome", "")
    })
    log_result(level, 6, "send_document(vendor_packet)_to_Procurement", result.get("reward", 0), result.get("done", False), result.get("observation", {}).get("deal_stage", "?"))
    if result.get("done"):
        print(f"  -> Episode ended: {result.get('info', {}).get('terminal_outcome')}")
        return results

    # Step 7: Final group proposal
    action = {
        "action_type": "group_proposal",
        "target": "all",
        "target_ids": ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"],
        "message": "We have addressed all requirements. Ready for final approval on concrete, reviewable terms.",
        "proposed_terms": {
            "price": 180000,
            "timeline_weeks": 14,
            "security_commitments": ["gdpr", "audit rights"],
            "support_level": "named_support_lead",
            "liability_cap": "mutual_cap",
        },
        "metadata": {"session_id": session_id}
    }
    result = api_step(session_id, action)
    results.append({
        "level": level, "step": 7, "action": "group_proposal_final",
        "reward": result.get("reward", 0), "done": result.get("done", False),
        "stage": result.get("observation", {}).get("deal_stage", "?"),
        "terminal_outcome": result.get("info", {}).get("terminal_outcome", "")
    })
    log_result(level, 7, "group_proposal_final", result.get("reward", 0), result.get("done", False), result.get("observation", {}).get("deal_stage", "?"))
    if result.get("done"):
        print(f"  -> Episode ended: {result.get('info', {}).get('terminal_outcome')}")
        return results

    # Continue with auto actions until done or max steps
    for extra_step in range(8, max_steps + 1):
        # Check current state
        state = api_state(session_id)
        deal_stage = state.get("deal_stage", "unknown")
        offer_state = state.get("offer_state", {})
        proposal_submitted = offer_state.get("proposal_submitted", False)

        # If in final_approval stage and proposal not submitted, submit it
        if deal_stage == "final_approval" and not proposal_submitted:
            result = submit_proposal(session_id, ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"])
        else:
            # Otherwise send direct message
            action = {
                "action_type": "direct_message",
                "target": "ExecSponsor",
                "target_ids": ["ExecSponsor"],
                "message": "Let me know if there's anything else you need before we proceed.",
                "metadata": {"session_id": session_id}
            }
            result = api_step(session_id, action)

        results.append({
            "level": level, "step": extra_step, "action": "auto_action",
            "reward": result.get("reward", 0), "done": result.get("done", False),
            "stage": result.get("observation", {}).get("deal_stage", "?"),
            "terminal_outcome": result.get("info", {}).get("terminal_outcome", "")
        })
        log_result(level, extra_step, "auto_action", result.get("reward", 0), result.get("done", False), result.get("observation", {}).get("deal_stage", "?"))

        if result.get("done"):
            print(f"  -> Episode ended: {result.get('info', {}).get('terminal_outcome')}")
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
    print("DealRoom S2P - Playing All Levels (Documented Walkthrough)")
    print("=" * 60)

    levels = ["aligned", "conflicted", "hostile_acquisition"]

    for level in levels:
        print(f"\n{'='*40}")
        print(f"Playing {level.upper()} level")
        print(f"{'='*40}")

        results = play_level_api(level, max_steps=20)
        all_results.extend(results)

        time.sleep(1)

    write_results_table(all_results)
    print(f"\nResults written to {RESULTS_FILE}")

    return all_results

if __name__ == "__main__":
    main()