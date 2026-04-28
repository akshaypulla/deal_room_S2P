#!/usr/bin/env python3
"""
Play DealRoom S2P levels - complete version.
"""

import requests
from typing import Dict, List, Any

BASE_URL = "http://localhost:7860"

def reset(task_id: str, seed: int = 42) -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": seed})
    resp.raise_for_status()
    return resp.json()

def step_fn(action: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    action["metadata"] = {"session_id": session_id}
    resp = requests.post(f"{BASE_URL}/step", json=action)
    resp.raise_for_status()
    return resp.json()

def get_state(session_id: str) -> Dict[str, Any]:
    resp = requests.get(f"{BASE_URL}/state", params={"session_id": session_id})
    resp.raise_for_status()
    return resp.json()

def play_level(task_id: str, level_name: str, seed: int = 42) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Playing {level_name} ({task_id}) seed={seed}")
    print(f"{'='*60}")

    obs = reset(task_id, seed)
    session_id = obs.get("metadata", {}).get("session_id")
    if not session_id:
        session_id = obs.get("session_id", "")

    trajectory = []
    total_reward = 0.0

    print(f"Initial stage: {obs.get('deal_stage')}")

    base_strategy = [
        {"action_type": "send_document", "target": "Finance", "target_ids": ["Finance"], "message": "ROI model.", "documents": [{"type": "roi_model", "specificity": "high"}]},
        {"action_type": "send_document", "target": "Legal", "target_ids": ["Legal"], "message": "DPA document.", "documents": [{"type": "dpa", "specificity": "high"}]},
        {"action_type": "send_document", "target": "TechLead", "target_ids": ["TechLead"], "message": "Security cert.", "documents": [{"type": "security_cert", "specificity": "high"}]},
        {"action_type": "send_document", "target": "Procurement", "target_ids": ["Procurement"], "message": "Vendor packet.", "documents": [{"type": "vendor_packet", "specificity": "high"}]},
        {"action_type": "send_document", "target": "Operations", "target_ids": ["Operations"], "message": "Implementation timeline.", "documents": [{"type": "implementation_timeline", "specificity": "high"}]},
    ]

    for i, planned in enumerate(base_strategy):
        action = dict(planned)
        action["metadata"] = {"session_id": session_id}

        print(f"\nStep {i+1}: {action['action_type']} -> {action.get('target', 'all')}")

        result = step_fn(action, session_id)
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {})

        total_reward += reward

        trajectory.append({
            "step": i + 1,
            "action_type": action["action_type"],
            "target": action.get("target", "all"),
            "reward": reward,
            "total_reward": total_reward,
            "stage": obs.get("deal_stage"),
            "blockers": obs.get("active_blockers", []),
            "done": done,
            "terminal_outcome": info.get("terminal_outcome", "")
        })

        print(f"Reward: {reward:.4f} (Total: {total_reward:.4f})")
        print(f"Stage: {obs.get('deal_stage')}")
        print(f"Blockers: {obs.get('active_blockers', [])}")

        if done:
            print(f"\nGAME OVER: {info.get('terminal_outcome', 'unknown')}")
            print(f"Final Score: {total_reward:.4f}")
            return {
                "task_id": task_id,
                "level_name": level_name,
                "seed": seed,
                "total_reward": total_reward,
                "rounds_played": len(trajectory),
                "terminal_outcome": info.get("terminal_outcome", "unknown"),
                "trajectory": trajectory
            }

    print("\nBase strategy complete. Continuing to close deal...")

    prev_stage = obs.get("deal_stage")
    for attempt in range(10):
        state = get_state(session_id)
        offer_state = state.get("offer_state", {})
        deal_stage = obs.get("deal_stage")
        active_blockers = obs.get("active_blockers", [])

        if deal_stage != prev_stage:
            print(f"\n*** Stage changed to: {deal_stage} ***")
            prev_stage = deal_stage

        if deal_stage == "final_approval":
            print("In final_approval - SUBMITTING GROUP PROPOSAL")
            action = {
                "action_type": "group_proposal",
                "target": "all",
                "target_ids": ["Finance", "Legal", "TechLead", "Procurement", "Operations", "ExecSponsor"],
                "message": "I believe we have enough alignment to move to final approval on concrete, reviewable terms.",
                "proposed_terms": {
                    "price": 180000,
                    "timeline_weeks": 14,
                    "security_commitments": ["gdpr", "audit rights"],
                    "support_level": "named_support_lead",
                    "liability_cap": "mutual_cap",
                },
                "metadata": {"session_id": session_id}
            }
        elif deal_stage == "legal_review":
            print("In legal_review stage")
            if not offer_state.get("has_dpa"):
                action = {
                    "action_type": "send_document",
                    "target": "Legal",
                    "target_ids": ["Legal"],
                    "message": "DPA for legal review.",
                    "documents": [{"type": "dpa", "specificity": "high"}],
                    "metadata": {"session_id": session_id}
                }
            elif not offer_state.get("has_security_cert"):
                action = {
                    "action_type": "send_document",
                    "target": "TechLead",
                    "target_ids": ["TechLead"],
                    "message": "Security certification.",
                    "documents": [{"type": "security_cert", "specificity": "high"}],
                    "metadata": {"session_id": session_id}
                }
            else:
                action = {
                    "action_type": "direct_message",
                    "target": "Legal",
                    "target_ids": ["Legal"],
                    "message": "Ready for final approval.",
                    "metadata": {"session_id": session_id}
                }
        elif deal_stage == "negotiation":
            print("In negotiation stage")
            if active_blockers:
                action = {
                    "action_type": "direct_message",
                    "target": active_blockers[0],
                    "target_ids": [active_blockers[0]],
                    "message": "Let me address your concerns.",
                    "metadata": {"session_id": session_id}
                }
            else:
                action = {
                    "action_type": "direct_message",
                    "target": "all",
                    "target_ids": [],
                    "message": "All major points are aligned.",
                    "metadata": {"session_id": session_id}
                }
        else:
            print(f"In {deal_stage} stage")
            action = {
                "action_type": "direct_message",
                "target": "all",
                "target_ids": [],
                "message": "Please let me know if you need anything.",
                "metadata": {"session_id": session_id}
            }

        print(f"\nAdaptive Step {len(trajectory) + 1}: {action['action_type']} -> {action.get('target', 'all')}")

        result = step_fn(action, session_id)
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {})

        total_reward += reward

        trajectory.append({
            "step": len(trajectory) + 1,
            "action_type": action["action_type"],
            "target": action.get("target", "all"),
            "reward": reward,
            "total_reward": total_reward,
            "stage": obs.get("deal_stage"),
            "blockers": obs.get("active_blockers", []),
            "done": done,
            "terminal_outcome": info.get("terminal_outcome", "")
        })

        print(f"Reward: {reward:.4f} (Total: {total_reward:.4f})")
        print(f"Stage: {obs.get('deal_stage')}")
        print(f"Blockers: {obs.get('active_blockers', [])}")

        if done:
            print(f"\nGAME OVER: {info.get('terminal_outcome', 'unknown')}")
            print(f"Final Score: {total_reward:.4f}")
            break

    return {
        "task_id": task_id,
        "level_name": level_name,
        "seed": seed,
        "total_reward": total_reward,
        "rounds_played": len(trajectory),
        "terminal_outcome": trajectory[-1].get("terminal_outcome", "unknown") if trajectory else "unknown",
        "trajectory": trajectory
    }

def main():
    results = []

    results.append(play_level("aligned", "Easy", seed=42))
    results.append(play_level("conflicted", "Medium", seed=64))
    results.append(play_level("hostile_acquisition", "Hard", seed=42))

    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print(f"| Level | Rounds | Final Reward | Terminal Outcome |")
    print(f"|-------|--------|--------------|----------------|")
    for r in results:
        print(f"| {r['level_name']} | {r['rounds_played']} | {r['total_reward']:.4f} | {r['terminal_outcome']} |")

    with open("/Users/akshaypulla/Documents/deal_room_S2P/game_results.md", "w") as f:
        f.write("# DealRoom S2P Game Results\n\n")
        f.write("## Summary Table\n\n")
        f.write(f"| Level | Rounds | Final Reward | Terminal Outcome |\n")
        f.write(f"|-------|--------|--------------|----------------|\n")
        for r in results:
            f.write(f"| {r['level_name']} | {r['rounds_played']} | {r['total_reward']:.4f} | {r['terminal_outcome']} |\n")

        f.write("\n## Trajectories\n\n")
        for r in results:
            f.write(f"### {r['level_name']} ({r['task_id']}) - Seed {r['seed']}\n\n")
            f.write("| Step | Action | Target | Reward | Cumulative | Stage | Blocker(s) |\n")
            f.write("|------|--------|-------|--------|------------|-------|------------|\n")
            for t in r["trajectory"]:
                blockers = ", ".join(t.get("blockers", [])) if t.get("blockers") else "-"
                f.write(f"| {t['step']} | {t['action_type']} | {t['target']} | {t['reward']:.4f} | {t['total_reward']:.4f} | {t['stage']} | {blockers} |\n")
            f.write("\n")

        f.write("\n## Notes\n\n")
        f.write("- **soft_veto_by_Legal**: Legal's CVaR risk preference triggers a veto when engagement drops. This is expected for conflicted/hostile tasks.\n")
        f.write("- **hard_veto::missing_final_proposal**: Required a group_proposal action in final_approval stage.\n")

    print(f"\nResults written to game_results.md")

if __name__ == "__main__":
    main()