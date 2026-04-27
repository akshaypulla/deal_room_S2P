"""DealRoom - Clean, understandable Gradio interface for B2B negotiation."""

from __future__ import annotations

import html
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata

from models import DealRoomAction, DealRoomObservation, DealRoomState
from server.app import DealRoomSessionPool
from server.walkthrough_data import GUIDE_DATA


ROLE_BY_ID = {
    "Legal": "legal_compliance",
    "Finance": "finance",
    "TechLead": "technical",
    "Procurement": "procurement",
    "Operations": "operations",
    "ExecSponsor": "executive_sponsor",
}
ROLE_ICONS = {
    "finance": "💰",
    "technical": "🛠️",
    "legal_compliance": "⚖️",
    "procurement": "📦",
    "operations": "⚙️",
    "executive_sponsor": "🎯",
}
ROLE_NAMES = {
    "finance": "Finance",
    "technical": "Tech Lead",
    "legal_compliance": "Legal",
    "procurement": "Procurement",
    "operations": "Operations",
    "executive_sponsor": "Executive Sponsor",
}


CLEAN_CSS = """
.dealroom-clean { background: #0f1117; font-family: 'Inter','Segoe UI',system-ui,sans-serif; min-height: 100vh; }
.dealroom-clean * { color: #e5e7eb; }
.dr-header { text-align: center; padding: 20px 0 16px; border-bottom: 1px solid #21262d; margin-bottom: 16px; }
.dr-header h1 { font-size: 1.6rem; font-weight: 700; color: #f9fafb; margin: 0 0 4px; }
.dr-header p { color: #8b949e; font-size: 0.88rem; margin: 0; }
.workflow-bar { display: flex; gap: 6px; margin-bottom: 16px; padding: 0; }
.workflow-step { flex: 1; text-align: center; padding: 10px 4px; border-radius: 8px; font-size: 0.72rem; font-weight: 600; background: #161b22; border: 1px solid #21262d; color: #484f58; }
.workflow-step.active { background: rgba(255,106,0,0.15); border-color: #ff6a00; color: #ff6a00; }
.workflow-step.done { background: rgba(34,197,94,0.1); border-color: #238636; color: #3fb950; }
.workflow-step .step-num { display: inline-block; font-size: 0.85rem; margin-right: 3px; }
.dr-main { display: grid; grid-template-columns: 1fr 300px; gap: 14px; }
.dr-left { display: flex; flex-direction: column; gap: 12px; }
.dr-right { display: flex; flex-direction: column; gap: 12px; }
.card { background: #161b22; border-radius: 10px; border: 1px solid #21262d; padding: 14px; }
.card h3 { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; margin: 0 0 10px; font-weight: 600; }
.level-card { background: #161b22; border-radius: 10px; border: 1px solid #21262d; padding: 14px; margin-bottom: 12px; }
.level-card h3 { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; margin: 0 0 10px; font-weight: 600; }
.level-btns { display: flex; gap: 8px; }
.level-btn { flex: 1; padding: 10px 4px !important; border-radius: 6px !important; font-size: 0.75rem !important; font-weight: 600 !important; border: 1px solid #30363d !important; }
.level-btn.selected { border-color: #ff6a00 !important; background: rgba(255,106,0,0.15) !important; color: #ff6a00 !important; }
.level-btn.unlocked { border-color: #238636 !important; background: rgba(34,197,94,0.08) !important; color: #3fb950 !important; }
.level-btn.locked { opacity: 0.4; cursor: not-allowed; }
.start-btn { margin-top: 10px; display: block; width: 100%; padding: 12px !important; font-size: 0.9rem !important; font-weight: 700 !important; background: linear-gradient(135deg,#ff6a00,#e05500) !important; border: none !important; border-radius: 8px !important; color: #fff !important; }

/* TOP: Stakeholder selection section */
.stakeholder-section { order: -1; }
.stakeholder-item { display: flex; align-items: center; gap: 10px; padding: 10px 10px; margin-bottom: 6px; border-radius: 6px; background: #0d1117; border: 1px solid #21262d; cursor: pointer; transition: all 0.15s; }
.stakeholder-item:hover { border-color: #30363d; background: #161b22; }
.stakeholder-item.selected { border-color: #ff6a00; background: rgba(255,106,0,0.08); box-shadow: 0 0 12px rgba(255,106,0,0.2); }
.stakeholder-item.aligned { border-left: 3px solid #3fb950; }
.stakeholder-item.blocking { border-left: 3px solid #f85149; animation: pulse-red 2s infinite; }
.stakeholder-item.uncertain { border-left: 3px solid #d29922; }
.stakeholder-item.triggered { border-left: 3px solid #a855f7; background: rgba(168,85,247,0.08); animation: pulse-purple 2s infinite; }
.stakeholder-item.dimmed { opacity: 0.45; }
@keyframes pulse-red { 0%,100%{box-shadow:0 0 0 0 rgba(248,81,73,0)} 50%{box-shadow:0 0 8px 2px rgba(248,81,73,0.3)} }
@keyframes pulse-purple { 0%,100%{box-shadow:0 0 0 0 rgba(168,85,247,0)} 50%{box-shadow:0 0 12px 4px rgba(168,85,247,0.4)} }
.s-icon { font-size: 1.2rem; width: 28px; text-align: center; }
.s-info { flex: 1; min-width: 0; }
.s-name { font-weight: 600; font-size: 0.85rem; color: #e6edf3; }
.s-role { font-size: 0.7rem; color: #8b949e; }
.s-badge { font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 999px; white-space: nowrap; }
.s-badge.green { background: rgba(63,185,80,0.15); color: #3fb950; }
.s-badge.red { background: rgba(248,81,73,0.15); color: #f85149; }
.s-badge.amber { background: rgba(210,153,34,0.15); color: #d29922; }
.s-badge.purple { background: rgba(168,85,247,0.15); color: #a855f7; }
.detail-quote { background: #0d1117; border-left: 3px solid #ff6a00; padding: 10px 12px; border-radius: 0 6px 6px 0; font-style: italic; color: #c9d1d9; font-size: 0.85rem; line-height: 1.5; margin: 8px 0; }
.detail-needs { background: rgba(255,106,0,0.08); border: 1px solid rgba(255,106,0,0.2); border-radius: 6px; padding: 8px 10px; font-size: 0.8rem; }
.detail-needs strong { color: #ff6a00; }
.detail-needs p { color: #c9d1d9; margin: 4px 0 0; }
.no-selection { text-align: center; padding: 24px 16px; color: #6e7681; font-size: 0.82rem; }
.suggestion-chips { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-bottom: 10px; }
.chip-btn { padding: 8px 6px !important; border-radius: 5px !important; border: 1px solid #30363d !important; background: #0d1117 !important; color: #c9d1d9 !important; font-size: 0.72rem !important; cursor: pointer; transition: all 0.15s; }
.chip-btn:hover { border-color: #ff6a00 !important; background: rgba(255,106,0,0.08) !important; }
.chip-btn.selected { border-color: #ff6a00 !important; background: rgba(255,106,0,0.15) !important; color: #ff6a00 !important; }

/* Response box with sent feedback */
.response-box { position: relative; }
.response-sent-feedback { display: none; align-items: center; gap: 6px; padding: 6px 10px; background: rgba(63,185,80,0.12); border: 1px solid rgba(63,185,80,0.3); border-radius: 6px; font-size: 0.75rem; color: #3fb950; margin-bottom: 6px; animation: fade-in 0.3s ease; }
.response-sent-feedback.show { display: flex; }
@keyframes fade-in { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:translateY(0)} }
.response-input { width: 100%; background: #0d1117 !important; border: 1px solid #30363d !important; border-radius: 6px !important; color: #c9d1d9 !important; padding: 10px 12px !important; font-size: 0.85rem !important; margin-bottom: 8px; }
.response-input:focus { border-color: #ff6a00 !important; box-shadow: 0 0 0 2px rgba(255,106,0,0.15) !important; }
.send-btn { width: 100%; padding: 10px !important; font-size: 0.88rem !important; font-weight: 600 !important; background: linear-gradient(135deg,#ff6a00,#e05500) !important; border: none !important; border-radius: 6px !important; color: #fff !important; }
.send-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(255,106,0,0.3); }
.send-btn.sent { background: linear-gradient(135deg,#238636,#1a7f37) !important; }
.run-bar { display: flex; gap: 6px; margin-top: 10px; }
.run-btn { flex: 2; padding: 10px !important; font-size: 0.85rem !important; font-weight: 600 !important; background: linear-gradient(135deg,#ff6a00,#e05500) !important; border: none !important; border-radius: 6px !important; color: #fff !important; }
.step-btn { flex: 1; padding: 10px !important; font-size: 0.85rem !important; background: #21262d !important; border: 1px solid #30363d !important; border-radius: 6px !important; color: #c9d1d9 !important; }
.hint-box { background: rgba(255,106,0,0.1); border: 1px solid rgba(255,106,0,0.3); border-radius: 8px; padding: 10px 12px; font-size: 0.82rem; color: #ff6a00; line-height: 1.5; }
.hint-box.neutral { border-color: rgba(139,148,158,0.3); color: #8b949e; text-align: center; }
.hint-box.success { background: rgba(63,185,80,0.1); border-color: rgba(63,185,80,0.3); color: #3fb950; }

/* BIG DEAL CLOSE POPUP */
.deal-close-popup { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.8); z-index: 9999; align-items: center; justify-content: center; animation: popup-overlay-in 0.4s ease; }
.deal-close-popup.show { display: flex; }
@keyframes popup-overlay-in { from{opacity:0} to{opacity:1} }
.deal-close-content { background: linear-gradient(135deg,#161b22,#0d1117); border: 2px solid #3fb950; border-radius: 16px; padding: 40px 50px; text-align: center; max-width: 480px; animation: popup-card-in 0.5s cubic-bezier(0.34,1.56,0.64,1); box-shadow: 0 0 60px rgba(63,185,80,0.3), 0 20px 60px rgba(0,0,0,0.5); }
@keyframes popup-card-in { from{opacity:0;transform:scale(0.7)} to{opacity:1;transform:scale(1)} }
.deal-close-icon { font-size: 4rem; margin-bottom: 16px; }
.deal-close-title { font-size: 1.8rem; font-weight: 700; color: #3fb950; margin: 0 0 8px; }
.deal-close-subtitle { font-size: 0.95rem; color: #8b949e; margin: 0 0 20px; }
.deal-close-score { font-size: 3.5rem; font-weight: 700; color: #ff6a00; margin: 10px 0; text-shadow: 0 0 30px rgba(255,106,0,0.5); }
.deal-close-score-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 24px; }
.deal-close-dismiss { padding: 12px 32px !important; background: #21262d !important; border: 1px solid #30363d !important; border-radius: 8px !important; color: #c9d1d9 !important; font-size: 0.88rem !important; cursor: pointer; }
.deal-close-dismiss:hover { border-color: #ff6a00 !important; color: #ff6a00 !important; }

/* Auto-play controls */
.autoplay-bar { display: flex; gap: 6px; margin-top: 8px; }
.autoplay-btn { flex: 1; padding: 10px !important; border-radius: 6px !important; font-size: 0.8rem !important; font-weight: 600 !important; border: none !important; }
.autoplay-start { background: linear-gradient(135deg,#238636,#1a7f37) !important; color: #fff !important; }
.autoplay-pause { background: linear-gradient(135deg,#d29922,#b8860b) !important; color: #0d1117 !important; }
.autoplay-stop { background: #374151 !important; color: #c9d1d9 !important; }

/* Triggered popup */
.triggered-popup { display: none; background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.4); border-radius: 8px; padding: 10px 12px; margin-top: 8px; font-size: 0.78rem; color: #a855f7; animation: fade-in 0.3s ease; }
.triggered-popup.show { display: block; }
.triggered-popup strong { color: #a855f7; }

.score-display { text-align: center; padding: 16px; background: linear-gradient(135deg,#1c2128,#0d1117); border: 2px solid #ff6a00; border-radius: 10px; margin-bottom: 12px; }
.score-value { font-size: 2.4rem; font-weight: 700; color: #ff6a00; line-height: 1; }
.score-label { font-size: 0.75rem; color: #8b949e; margin-top: 4px; }
.score-delta { font-size: 0.9rem; margin-top: 4px; }
.breakdown h4 { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 6px; }
.breakdown-item { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #21262d; font-size: 0.78rem; }
.breakdown-item:last-child { border-bottom: none; }
.breakdown-label { color: #c9d1d9; }
.breakdown-value.high { color: #3fb950; font-weight: 600; }
.breakdown-value.med { color: #d29922; font-weight: 600; }
.breakdown-value.low { color: #f85149; font-weight: 600; }
.breakdown-weight { color: #6e7681; font-size: 0.65rem; margin-left: 3px; }
.signal-tag { display: inline-flex; align-items: center; gap: 3px; padding: 3px 7px; border-radius: 999px; font-size: 0.68rem; background: rgba(255,106,0,0.1); color: #ff6a00; margin: 2px 3px 2px 0; }
.blocker-tag { display: inline-flex; align-items: center; gap: 3px; padding: 3px 7px; border-radius: 999px; font-size: 0.68rem; background: rgba(248,81,73,0.1); color: #f85149; margin: 2px 3px 2px 0; }
.info-card { background: rgba(63,185,80,0.06); border: 1px solid rgba(63,185,80,0.2); border-radius: 10px; padding: 12px 14px; font-size: 0.78rem; color: #7ee787; line-height: 1.5; }
.info-card h4 { margin: 0 0 6px; color: #3fb950; font-size: 0.8rem; }
.info-card li { margin-bottom: 3px; }
.complete-card { background: linear-gradient(135deg,rgba(63,185,80,0.1),rgba(255,106,0,0.05)); border: 2px solid #3fb950; border-radius: 10px; padding: 18px; text-align: center; }
.complete-card h2 { color: #3fb950; font-size: 1.1rem; margin: 0 0 8px; }
.complete-card .final-score { font-size: 2.2rem; font-weight: 700; color: #ff6a00; }
.complete-card p { color: #8b949e; font-size: 0.8rem; margin-top: 6px; }
.no-start { text-align: center; padding: 32px 16px; color: #6e7681; }
.no-start .big-icon { font-size: 2.5rem; margin-bottom: 10px; }
.no-start p { font-size: 0.85rem; }

/* Level unlocks info */
.level-info { font-size: 0.7rem; color: #6e7681; margin-top: 6px; text-align: center; }
.level-info span { color: #3fb950; }
"""


class DealRoomWebManager:
    def __init__(self, pool: DealRoomSessionPool, metadata: EnvironmentMetadata):
        self.pool = pool
        self.metadata = metadata
        self._playground_session_id: Optional[str] = None

    def reset_session(self, task_id: str, seed: int, session_id: Optional[str] = None):
        return self.pool.reset(task_id=task_id, seed=seed, session_id=session_id)

    def step_session(self, session_id: str, action: DealRoomAction):
        return self.pool.step(session_id, action)

    def get_state_for_session(self, session_id: str) -> Dict[str, Any]:
        return self.pool.state(session_id).model_dump()

    def get_beliefs_for_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.pool.get_beliefs(session_id)


def load_metadata() -> EnvironmentMetadata:
    readme_path = Path("README.md")
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
    return EnvironmentMetadata(
        name="deal_room_s2p",
        description="A realistic multi-stakeholder enterprise negotiation environment.",
        version="1.0.0",
        author="akshaypulla",
        readme_content=readme,
    )


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _normalize_view_state(view_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = {
        "task": "aligned", "seed": 42, "level": "simple", "source": "clean",
        "session_id": None, "selected_stakeholder": None, "current_observation": None,
        "current_state": None, "trace": [], "status_message": "", "popup_queue": [],
        "popup_index": 0, "unlocked_levels": ["simple", "medium", "hard"], "round_started": False,
        "show_hint": False, "hint_text": "", "selected_chip": None,
        "message_text": "", "auto_playing": False, "auto_paused": False,
        "auto_speed": "medium", "last_score": 0.0, "score_delta": None,
        "show_advanced": False, "round_complete": False, "use_llm_agent": False,
        "seed_preset": None, "workflow_step": 0,
        "response_sent": False, "triggered_stakeholders": [], "deal_close_showing": False,
    }
    if not isinstance(view_state, dict):
        return base
    merged = dict(base)
    merged.update(view_state)
    for key in ("trace", "popup_queue", "triggered_stakeholders"):
        if not isinstance(merged.get(key), list):
            merged[key] = []
    if not isinstance(merged.get("unlocked_levels"), list):
        merged["unlocked_levels"] = ["simple", "medium", "hard"]
    if "use_llm_agent" not in merged:
        merged["use_llm_agent"] = False
    if "seed_preset" not in merged:
        merged["seed_preset"] = None
    if "workflow_step" not in merged:
        merged["workflow_step"] = 0
    if "response_sent" not in merged:
        merged["response_sent"] = False
    if "triggered_stakeholders" not in merged:
        merged["triggered_stakeholders"] = []
    if "deal_close_showing" not in merged:
        merged["deal_close_showing"] = False
    return merged


def _normalize_saved_runs(saved_runs: Any) -> List[Dict[str, Any]]:
    if not isinstance(saved_runs, list):
        return []
    return [item for item in saved_runs if isinstance(item, dict)]


def _coerce_observation(data: Dict[str, Any]) -> DealRoomObservation:
    return DealRoomObservation.model_validate(data)


def _approval_band(observation: Dict[str, Any], stakeholder_id: str) -> str:
    progress = observation.get("approval_path_progress", {}).get(stakeholder_id, {})
    band = str(progress.get("band", "neutral"))
    if band in ("supporter", "workable"):
        return "aligned"
    elif band == "blocker":
        return "blocking"
    return "uncertain"


def _generate_chips(view_state: Dict[str, Any]) -> List[str]:
    view_state = _normalize_view_state(view_state)
    observation = view_state.get("current_observation") or {}
    selected = view_state.get("selected_stakeholder")
    if not selected or not observation:
        return ["Address concern", "Ask for details", "Provide evidence", "Delay decision"]
    payload = observation.get("stakeholders", {}).get(selected, {})
    role_type = ROLE_BY_ID.get(selected, "")
    band = _approval_band(observation, selected)
    requested = observation.get("requested_artifacts", {}).get(selected, [])
    if role_type == "finance" or "budget" in str(requested) or "roi" in str(requested):
        return ["Provide ROI justification", "Address budget concern", "Ask for cost breakdown", "Propose payment terms"]
    elif role_type == "legal_compliance":
        return ["Clarify contract terms", "Address compliance gap", "Provide legal documentation", "Suggest alternative clauses"]
    elif role_type == "procurement":
        return ["Justify vendor selection", "Compare alternatives", "Address delivery concerns", "Request timeline flexibility"]
    elif role_type == "executive_sponsor":
        return ["Summarize business case", "Highlight key benefits", "Address risk concerns", "Propose next steps"]
    elif role_type == "technical":
        return ["Share implementation plan", "Provide technical specs", "Address security concerns", "Clarify integration"]
    elif role_type == "operations":
        return ["Share timeline", "Address support plan", "Provide training details", "Clarify handoff"]
    elif band == "blocking":
        return ["Address their concern", "Request clarification", "Provide supporting evidence", "Escalate to manager"]
    elif band == "uncertain":
        return ["Clarify your position", "Provide more details", "Address specific worries", "Build confidence"]
    return ["Acknowledge their point", "Build on alignment", "Move discussion forward", "Confirm understanding"]


def _policy_action(observation: Dict[str, Any]) -> DealRoomAction:
    obs = _coerce_observation(observation)
    for stakeholder_id, artifacts in obs.requested_artifacts.items():
        if artifacts:
            artifact = artifacts[0]
            return DealRoomAction(
                action_type="send_document",
                target=stakeholder_id,
                target_ids=[stakeholder_id],
                message=f"Sharing the requested {artifact.replace('_', ' ')}.",
                documents=[{"type": artifact, "specificity": "high"}],
            )
    if obs.active_blockers or not obs.known_constraints:
        target_id = next(iter(obs.active_blockers), next(iter(obs.stakeholders), "all"))
        return DealRoomAction(
            action_type="direct_message", target=target_id,
            target_ids=[target_id] if target_id != "all" else [],
            message="Help me understand the real concern we need to address.",
        )
    return DealRoomAction(
        action_type="direct_message", target="all",
        target_ids=list(obs.stakeholders.keys()),
        message="Let me know if there's anything else you need before we proceed.",
    )


def _run_reset(task: str, seed: int, level: str, view_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    current = _normalize_view_state(view_state)
    session_id, obs, state = web_manager.reset_session(
        task_id=task, seed=int(seed), session_id=current.get("session_id")
    )
    observation = obs.model_dump()
    stakeholders = list(observation.get("stakeholders", {}).keys())
    updated = dict(current)
    updated.update({
        "task": task, "seed": int(seed), "level": level, "source": "clean",
        "session_id": session_id,
        "current_observation": observation,
        "current_state": state.model_dump(),
        "selected_stakeholder": stakeholders[0] if stakeholders else None,
        "popup_queue": [{"stakeholder_id": s} for s in stakeholders],
        "popup_index": 0,
        "round_started": True,
        "show_hint": True,
        "hint_text": "Click on a stakeholder to see their message",
        "trace": [],
        "last_score": 0.0,
        "score_delta": None,
        "round_complete": False,
        "selected_chip": None,
        "message_text": "",
        "workflow_step": 1,
        "response_sent": False,
        "triggered_stakeholders": [],
        "deal_close_showing": False,
    })
    return updated


def _record_step(view_state: Dict[str, Any], action: DealRoomAction, obs: DealRoomObservation,
                 reward: float, done: bool, info: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    updated = _normalize_view_state(view_state)
    trace = list(updated.get("trace", []))
    old_score = updated.get("last_score", 0.0)
    step_num = len([item for item in trace if item.get("kind") == "step"]) + 1
    trace.append({
        "kind": "step", "step": step_num, "action": action.model_dump(),
        "reward": reward, "done": done, "stage": obs.deal_stage,
        "blockers": list(obs.active_blockers),
    })
    updated["trace"] = trace
    updated["current_observation"] = obs.model_dump()
    updated["current_state"] = state
    updated["last_info"] = info
    updated["score_delta"] = reward - old_score if old_score else None
    updated["last_score"] = reward
    updated["round_complete"] = done
    updated["response_sent"] = True
    if done:
        updated["status_message"] = f"Round complete! Final score: {reward:.2f}"
        updated["show_hint"] = False
        updated["workflow_step"] = 4
        updated["deal_close_showing"] = True
    else:
        updated["status_message"] = f"Step {step_num} | Reward: {reward:.2f}"
        updated["workflow_step"] = 3
        updated["response_sent"] = False
    return updated


def _compute_triggered_stakeholders(info: Dict[str, Any], prev_info: Optional[Dict[str, Any]]) -> List[str]:
    triggered = []
    propagation_deltas = info.get("propagation_deltas", {})
    if propagation_deltas:
        for sid, delta in propagation_deltas.items():
            if delta and abs(float(delta)) > 0.05:
                triggered.append(sid)
    return triggered


def _save_run_if_complete(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]):
    view_state = _normalize_view_state(view_state)
    saved_runs = _normalize_saved_runs(saved_runs)
    observation = view_state.get("current_observation") or {}
    if not observation.get("done"):
        return saved_runs, view_state
    score = float(view_state.get("last_score", 0.0))
    run_id = f"{view_state['level']}-{view_state['task']}-{view_state['seed']}-{view_state['source']}-{len(saved_runs) + 1}"
    saved_runs = [item for item in saved_runs if item.get("id") != run_id]
    saved_runs.append({
        "id": run_id, "task": view_state["task"], "level": view_state["level"],
        "seed": view_state["seed"], "source": view_state["source"], "score": score,
    })
    saved_runs = saved_runs[-8:]
    return saved_runs, view_state


def _build_workflow_bar(view_state: Dict[str, Any]) -> Tuple[str, str, str, str]:
    view_state = _normalize_view_state(view_state)
    step = view_state.get("workflow_step", 0)
    round_complete = view_state.get("round_complete", False)
    steps_data = [
        ("1", "Choose Level", 0, step == 0 and view_state.get("round_started")),
        ("2", "View Stakeholder", 1, step == 1),
        ("3", "Send Response", 2, step == 2),
        ("4", "Run Round", 3, step == 3 or round_complete),
    ]
    bars = []
    for num, label, min_step, active in steps_data:
        if round_complete and not active:
            cls, text = "workflow-step done", f"✓ {label}"
        elif step > min_step and not active:
            cls, text = "workflow-step done", f"✓ {label}"
        elif active:
            cls, text = "workflow-step active", f"<span class='step-num'>{num}</span> {label}"
        else:
            cls, text = "workflow-step", f"<span class='step-num'>{num}</span> {label}"
        bars.append(f"<div class='{cls}'>{text}</div>")
    return tuple(bars)


def _build_level_buttons(view_state: Dict[str, Any]) -> Tuple[str, str, str, str]:
    view_state = _normalize_view_state(view_state)
    current_level = view_state.get("level", "simple")
    unlocked = view_state.get("unlocked_levels", ["simple", "medium", "hard"])

    def level_btn(label: str, level: str, emoji: str, desc: str) -> str:
        is_selected = level == current_level
        is_unlocked = level in unlocked
        cls = "level-btn"
        if is_selected:
            cls += " selected"
        if is_unlocked:
            cls += " unlocked"
        else:
            cls += " locked"
        return f"<button class='{cls}' data-level='{level}'>{emoji} {label}<br><small style='opacity:0.7;font-size:0.6rem'>{desc}</small></button>"

    return (
        level_btn("Simple", "simple", "🔓", "Aligned task"),
        level_btn("Medium", "medium", "⚡", "CTO-CFO tension"),
        level_btn("Hard", "hard", "🔒", "Post-acquisition"),
        "",
    )


def _build_stakeholder_list(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    observation = view_state.get("current_observation") or {}
    selected = view_state.get("selected_stakeholder")
    stakeholders = observation.get("stakeholders", {})
    triggered = view_state.get("triggered_stakeholders", [])

    if not stakeholders:
        return ("<div class='card'><h3>👥 Stakeholders</h3>"
                "<div class='no-start'><div class='big-icon'>👥</div>"
                "<p>Start a round to meet your stakeholders</p></div></div>")

    items = []
    for sid in stakeholders:
        band = _approval_band(observation, sid)
        is_selected = sid == selected
        is_triggered = sid in triggered
        role_type = ROLE_BY_ID.get(sid, "")
        payload = stakeholders[sid]
        role_title = payload.get("role", sid)
        icon = ROLE_ICONS.get(role_type, "👤")

        status_map = {
            "aligned": ("Aligned", "green"),
            "blocking": ("Blocking", "red"),
            "uncertain": ("Uncertain", "amber"),
        }
        status_text, status_class = status_map.get(band, ("Unknown", ""))

        classes = ["stakeholder-item", band]
        if is_triggered:
            classes = ["stakeholder-item", "triggered"]
            status_class = "purple"
            status_text = "Triggered"
        if is_selected:
            classes.append("selected")
        elif selected and sid != selected:
            classes.append("dimmed")

        items.append(
            f"<div class='{' '.join(classes)}' data-stakeholder='{_escape(sid)}'>"
            f"<div class='s-icon'>{icon}</div>"
            f"<div class='s-info'><div class='s-name'>{_escape(sid)}</div>"
            f"<div class='s-role'>{_escape(role_title)}</div></div>"
            f"<span class='s-badge {status_class}'>{status_text}</span></div>"
        )
    return "<div class='card'><h3>👥 Stakeholders</h3>" + "".join(items) + "</div>"


def _build_detail_panel(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    observation = view_state.get("current_observation") or {}
    selected = view_state.get("selected_stakeholder")
    if not selected or not observation or selected not in observation.get("stakeholders", {}):
        return ("<div class='card'><h3>💬 Stakeholder Details</h3>"
                "<div class='no-selection'>Select a stakeholder to view their message</div></div>")
    payload = observation["stakeholders"][selected]
    message = observation.get("stakeholder_messages", {}).get(selected, "No message yet.")
    requested = observation.get("requested_artifacts", {}).get(selected, [])
    band = _approval_band(observation, selected)
    role_type = ROLE_BY_ID.get(selected, "")
    icon = ROLE_ICONS.get(role_type, "👤")
    needs_text = ", ".join(r.replace("_", " ") for r in requested) if requested else "Nothing specific requested yet"
    status_desc = {
        "aligned": "This stakeholder is aligned and supports moving forward",
        "blocking": "This stakeholder is blocking — needs immediate attention",
        "uncertain": "This stakeholder is uncertain — needs more convincing",
    }.get(band, "")
    return (
        f"<div class='card'><h3><span style='font-size:1.1rem'>{icon}</span> {_escape(selected)}</h3>"
        f"<div class='detail-quote'>\"{_escape(message)}\"</div>"
        f"<div class='detail-needs'><strong>What they need:</strong><p>{_escape(needs_text)}</p></div>"
        f"<p style='margin-top:8px;font-size:0.75rem;color:#8b949e;'>{_escape(status_desc)}</p></div>"
    )


def _build_hint(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    round_started = view_state.get("round_started", False)
    round_complete = view_state.get("round_complete", False)
    workflow_step = view_state.get("workflow_step", 0)
    response_sent = view_state.get("response_sent", False)

    if round_complete:
        return ("<div class='hint-box success'>🎉 <strong>Deal Complete!</strong> "
                "All stakeholders addressed. Check your score on the right. <button onclick='window.showDealClosePopup()' style='margin-left:8px;padding:4px 12px;background:#238636;border:none;border-radius:4px;color:#fff;font-size:0.75rem;cursor:pointer'>View Results</button></div>")
    if not round_started:
        return ("<div class='hint-box neutral'>👆 Choose a difficulty level (Simple/Medium/Hard) and click "
                "<strong>Start Round</strong> to begin negotiating</div>")
    if workflow_step == 1:
        return ("<div class='hint-box'>✅ <strong>Round started!</strong> "
                "Click on a stakeholder above to see their message and needs. Use chips or type your message.</div>")
    elif workflow_step == 2:
        hint = "💬 <strong>Send a response</strong> — Type a message or click a suggestion chip below, then click <strong>Send Response</strong>."
        if response_sent:
            hint += " <span style='color:#3fb950'>✓ Sent! Now click <strong>Run Round</strong>.</span>"
        return f"<div class='hint-box'>{hint}</div>"
    elif workflow_step == 3:
        return ("<div class='hint-box'>▶ <strong>Click Run Round</strong> to send your message "
                "and see how stakeholders respond.</div>")
    observation = view_state.get("current_observation") or {}
    stakeholders = observation.get("stakeholders", {})
    aligned = sum(1 for s in stakeholders if _approval_band(observation, s) == "aligned")
    total = len(stakeholders)
    if aligned == total and total > 0:
        return ("<div class='hint-box'>🎉 <strong>All aligned!</strong> "
                "Click <strong>Run Round</strong> to finalize the deal.</div>")
    return f"<div class='hint-box'>📊 <strong>{aligned}/{total} aligned.</strong> Continue working with the remaining stakeholders.</div>"


def _compute_score_breakdown(view_state: Dict[str, Any]) -> Dict[str, float]:
    info = view_state.get("last_info", {})
    reward_components = info.get("reward_components", {})
    if reward_components:
        return {k: float(v) for k, v in reward_components.items()}
    observation = view_state.get("current_observation") or {}
    if not observation:
        return {"goal": 0.0, "trust": 0.0, "info": 0.0, "risk": 0.0, "causal": 0.0}
    done = observation.get("done", False)
    if done:
        terminal = info.get("terminal_reward", 0.0)
        if terminal > 0:
            return {"goal": terminal, "trust": 0.0, "info": 0.0, "risk": 0.0, "causal": 0.0}
        elif terminal < 0:
            return {"goal": 0.0, "trust": 0.0, "info": 0.0, "risk": abs(terminal), "causal": 0.0}
    state = view_state.get("current_state") or {}
    positive_masses = [
        belief.get("positive_mass", 0.5)
        for belief in state.get("beliefs", {})
    ]
    avg_belief = sum(positive_masses) / max(len(positive_masses), 1)
    blockers = list(observation.get("active_blockers", []))
    goal = avg_belief * (1.0 - 0.15 * len(blockers))
    trust = avg_belief * (1.0 - 0.05 * len(blockers))
    risk = max(0.0, 1.0 - avg_belief - 0.1 * len(blockers))
    info_dim = 0.5
    causal = 0.5
    return {"goal": goal, "trust": trust, "info": info_dim, "risk": risk, "causal": causal}


V3_REWARD_WEIGHTS = {
    "goal": 0.25,
    "trust": 0.20,
    "info": 0.20,
    "risk": 0.20,
    "causal": 0.15,
}


def _build_score_card(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    observation = view_state.get("current_observation") or {}
    current_score = view_state.get("last_score", 0.0)
    score_delta = view_state.get("score_delta")
    done = observation.get("done", False)
    info = view_state.get("last_info", {})
    terminal_outcome = info.get("terminal_outcome", "")
    if done:
        if "deal_closed" in terminal_outcome:
            outcome_label = "🎉 Deal Closed!"
        elif "veto" in terminal_outcome:
            outcome_label = "⚠️ Deal Vetoed"
        elif "timeout" in terminal_outcome or "max_rounds" in terminal_outcome:
            outcome_label = "⏱️ Deal Timed Out"
        else:
            outcome_label = "🏁 Deal Ended"
        return (
            f"<div class='complete-card'><h2>{outcome_label}</h2>"
            f"<div class='final-score'>{current_score:.2f}</div><p>Final Score</p></div>"
        )
    delta_html = ""
    if score_delta is not None:
        sign = "+" if score_delta >= 0 else ""
        color = "#3fb950" if score_delta >= 0 else "#f85149"
        delta_html = f"<div class='score-delta' style='color:{color}'>{sign}{score_delta:.2f}</div>"
    blockers = observation.get("active_blockers", [])
    blocker_html = ""
    if blockers:
        tags = "".join(f"<span class='blocker-tag'>⚠️ {_escape(b)}</span>" for b in blockers)
        blocker_html = f"<div style='margin-top:8px;'>{tags}</div>"
    breakdown = _compute_score_breakdown(view_state)
    weights = V3_REWARD_WEIGHTS
    breakdown_items = []
    for key, weight in weights.items():
        label = key.title()
        value = breakdown.get(key, 0.0)
        pct = int(value * 100)
        cls = "high" if pct >= 70 else "med" if pct >= 40 else "low"
        breakdown_items.append(
            f"<div class='breakdown-item'><span class='breakdown-label'>{label}</span>"
            f"<span><span class='breakdown-value {cls}'>{pct}%</span>"
            f"<span class='breakdown-weight'>({int(weight*100)}%)</span></span></div>"
        )
    return (
        "<div class='card'><div class='score-display'>"
        f"<div class='score-value'>{current_score:.2f}</div>"
        "<div class='score-label'>Current Score</div>"
        f"{delta_html}{blocker_html}</div>"
        "<div class='breakdown'><h4>Score Breakdown</h4>"
        + "".join(breakdown_items) + "</div></div>"
    )


def _build_signals(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    observation = view_state.get("current_observation") or {}
    if not observation:
        return "<div class='card'><h3>📋 Pending Requests</h3><div style='color:#6e7681;font-size:0.8rem;'>No requests yet</div></div>"
    signals = []
    for sid, artifacts in observation.get("requested_artifacts", {}).items():
        if artifacts:
            for art in artifacts:
                signals.append(f"<span class='signal-tag'>📋 {_escape(art.replace('_', ' '))}</span>")
    if not signals:
        return "<div class='card'><h3>📋 Pending Requests</h3><div style='color:#6e7681;font-size:0.8rem;'>No pending requests</div></div>"
    return "<div class='card'><h3>📋 Pending Requests</h3>" + "".join(signals) + "</div>"


def _build_how_it_works() -> str:
    return (
        "<div class='info-card'><h4>📖 How to Negotiate</h4>"
        "<ol>"
        "<li><strong>Start a round</strong> — Choose difficulty and click Start Round</li>"
        "<li><strong>Click a stakeholder</strong> (in the list above) — See their message and needs</li>"
        "<li><strong>Send a response</strong> — Type or click a suggestion, then Send Response</li>"
        "<li><strong>Run the round</strong> — Click Run Round to see stakeholder reactions</li>"
        "<li><strong>Goal</strong> — Get all stakeholders to 'Aligned' to close the deal</li>"
        "</ol>"
        "<p style='margin-top:8px;color:#8b949e;'><strong>Scoring:</strong> "
        "Approval (35%) · Constraints (25%) · Feasibility (15%) · "
        "Relationships (15%) · Efficiency (10%)</p>"
        "<p style='margin-top:6px;color:#a855f7;'><strong>Purple = Triggered:</strong> When one stakeholder's concern "
        "affects another, they'll be highlighted in purple with a popup explaining the cascade.</p></div>"
    )


def _build_triggered_popup(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    triggered = view_state.get("triggered_stakeholders", [])
    if not triggered:
        return "<div class='triggered-popup'></div>"
    names = ", ".join(f"<strong>{_escape(s)}</strong>" for s in triggered)
    return (
        f"<div class='triggered-popup show'>"
        f"<strong>⚡ Cascade Effect!</strong> {names} were affected by this action — "
        f"their opinions shifted due to cross-stakeholder influence.</div>"
    )


def _build_deal_close_popup(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    showing = view_state.get("deal_close_showing", False)
    if not showing:
        return "<div class='deal-close-popup'></div>"
    score = view_state.get("last_score", 0.0)
    info = view_state.get("last_info", {})
    terminal_outcome = info.get("terminal_outcome", "")
    if "deal_closed" in terminal_outcome:
        icon = "🎉"
        title = "Deal Closed!"
        subtitle = "Congratulations! You successfully navigated the negotiation and closed the deal."
    elif "veto" in terminal_outcome:
        icon = "⚠️"
        title = "Deal Vetoed"
        subtitle = "A stakeholder blocked the deal. Review their concerns and try again."
    elif "timeout" in terminal_outcome or "max_rounds" in terminal_outcome:
        icon = "⏱️"
        title = "Deal Timed Out"
        subtitle = "The negotiation exceeded the maximum rounds. Try being more efficient."
    else:
        icon = "🏁"
        title = "Deal Ended"
        subtitle = "The negotiation has concluded."
    return (
        f"<div class='deal-close-popup show'>"
        f"<div class='deal-close-content'>"
        f"<div class='deal-close-icon'>{icon}</div>"
        f"<div class='deal-close-title'>{title}</div>"
        f"<div class='deal-close-subtitle'>{subtitle}</div>"
        f"<div class='deal-close-score'>{score:.2f}</div>"
        f"<div class='deal-close-score-label'>Final Score</div>"
        f"<button class='deal-close-dismiss' onclick='window.dismissDealClosePopup()'>Continue</button>"
        f"</div></div>"
    )


def _build_suggestion_chips(view_state: Dict[str, Any]) -> Tuple[str, str, str, str]:
    chips = _generate_chips(view_state)
    return tuple(chips[i] if i < len(chips) else "" for i in range(4))


def _build_autoplay_controls(view_state: Dict[str, Any]) -> Tuple[str, str, str]:
    view_state = _normalize_view_state(view_state)
    auto_playing = view_state.get("auto_playing", False)
    auto_paused = view_state.get("auto_paused", False)

    if auto_playing and not auto_paused:
        pause_show = "inline-block"
        start_show = "none"
        stop_show = "inline-block"
    elif auto_paused:
        pause_show = "none"
        start_show = "inline-block"
        stop_show = "inline-block"
    else:
        pause_show = "none"
        start_show = "none"
        stop_show = "none"

    return (
        f"<div class='autoplay-bar' style='display:flex;gap:6px;margin-top:8px;'>"
        f"<button class='autoplay-btn autoplay-start' style='display:{start_show}' data-autoplay='start'>▶ Auto-Play</button>"
        f"<button class='autoplay-btn autoplay-pause' style='display:{pause_show}' data-autoplay='pause'>⏸ Pause</button>"
        f"<button class='autoplay-btn autoplay-stop' style='display:{stop_show}' data-autoplay='stop'>⏹ Stop</button>"
        f"</div>",
        f"<div style='font-size:0.7rem;color:#6e7681;margin-top:4px;text-align:center;display:{start_show if auto_playing else 'none'}'>Auto-playing until deal closes...</div>",
        "inline-block" if not auto_playing else "none",
    )


def _render_all_outputs(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    view_state = _normalize_view_state(view_state)
    saved_runs = _normalize_saved_runs(saved_runs)
    observation = view_state.get("current_observation") or {}
    stakeholders = observation.get("stakeholders", {})
    selected = view_state.get("selected_stakeholder")
    round_started = view_state.get("round_started", False)
    response_sent = view_state.get("response_sent", False)

    stakeholder_choices = [
        f"{ROLE_ICONS.get(ROLE_BY_ID.get(sid, ''), '👤')} {sid}"
        for sid in stakeholders
    ]
    selected_choice = None
    if selected and selected in stakeholders:
        selected_choice = f"{ROLE_ICONS.get(ROLE_BY_ID.get(selected, ''), '👤')} {selected}"

    wf1, wf2, wf3, wf4 = _build_workflow_bar(view_state)
    chip_0, chip_1, chip_2, chip_3 = _build_suggestion_chips(view_state)
    lvl_simple, lvl_medium, lvl_hard, lvl_extra = _build_level_buttons(view_state)
    autoplay_html, autoplay_status, autoplay_toggle = _build_autoplay_controls(view_state)

    return (
        wf1, wf2, wf3, wf4,
        _build_stakeholder_list(view_state),
        _build_detail_panel(view_state),
        _build_hint(view_state),
        _build_triggered_popup(view_state),
        gr.update(value=selected_choice, choices=stakeholder_choices, visible=round_started and bool(stakeholder_choices)),
        gr.update(visible=round_started),
        gr.update(value=chip_0, visible=round_started and bool(chip_0)),
        gr.update(value=chip_1, visible=round_started and bool(chip_1)),
        gr.update(value=chip_2, visible=round_started and bool(chip_2)),
        gr.update(value=chip_3, visible=round_started and bool(chip_3)),
        gr.update(value=view_state.get("message_text", "")),
        _build_score_card(view_state),
        _build_signals(view_state),
        _build_how_it_works(),
        _build_deal_close_popup(view_state),
        lvl_simple, lvl_medium, lvl_hard,
        gr.update(value=autoplay_toggle, visible=round_started),
        gr.update(value=autoplay_html),
        gr.update(value=autoplay_status),
        "none" if response_sent else "block",
    )


web_manager: Optional[DealRoomWebManager] = None


def build_clean_tab(pool: DealRoomSessionPool, metadata: EnvironmentMetadata) -> gr.Blocks:
    global web_manager
    web_manager = DealRoomWebManager(pool, metadata)

    demo = gr.Blocks(elem_classes=["dealroom-clean"])
    with demo:
        gr.HTML(f"<style>{CLEAN_CSS}</style>")
        gr.HTML("""
        <script>
        window.showDealClosePopup = function() { document.querySelectorAll('.deal-close-popup').forEach(e => e.classList.add('show')); };
        window.dismissDealClosePopup = function() { document.querySelectorAll('.deal-close-popup').forEach(e => e.classList.remove('show')); };
        </script>
        """)

        view_state = gr.State(_normalize_view_state(None))
        saved_runs = gr.State([])

        with gr.Column(elem_classes=["dr-header"]):
            gr.HTML("<h1>🎯 DealRoom</h1>"
                    "<p>B2B Negotiation Simulator — Work with stakeholders to close the deal</p>")

        with gr.Row(elem_classes=["workflow-bar"]):
            wf1_html = gr.HTML()
            wf2_html = gr.HTML()
            wf3_html = gr.HTML()
            wf4_html = gr.HTML()

        with gr.Column(elem_classes=["level-card"]) as level_card_block:
            gr.HTML("<h3>Difficulty Level</h3>")
            with gr.Row(elem_classes=["level-btns"]):
                simple_btn = gr.HTML()
                medium_btn = gr.HTML()
                hard_btn = gr.HTML()
                level_extra = gr.HTML()
            start_btn = gr.Button("▶ Start Round", elem_classes=["start-btn"], variant="primary")

        with gr.Row(elem_classes=["dr-main"]):
            with gr.Column(elem_classes=["dr-left"]):
                stakeholders_card_html = gr.HTML()
                detail_card_html = gr.HTML()
                triggered_popup_html = gr.HTML()

                with gr.Column(elem_classes=["card"]):
                    gr.HTML("<h3>💬 Your Response</h3>")
                    response_sent_feedback = gr.HTML()
                    chip_btn_0 = gr.Button("", variant="secondary", elem_classes=["chip-btn"], visible=False)
                    chip_btn_1 = gr.Button("", variant="secondary", elem_classes=["chip-btn"], visible=False)
                    chip_btn_2 = gr.Button("", variant="secondary", elem_classes=["chip-btn"], visible=False)
                    chip_btn_3 = gr.Button("", variant="secondary", elem_classes=["chip-btn"], visible=False)
                    message_input = gr.Textbox(
                        placeholder="Type your message here or click a suggestion above...",
                        lines=2, elem_classes=["response-input"], visible=False
                    )
                    send_btn = gr.Button("Send Response ✓", elem_classes=["send-btn"], variant="primary", visible=False)
                    with gr.Row(elem_classes=["run-bar"]):
                        run_btn = gr.Button("▶ Run Round", elem_classes=["run-btn"], variant="primary", visible=False)
                        step_btn = gr.Button("⏭ Step", elem_classes=["step-btn"], variant="secondary", visible=False)

                    autoplay_toggle = gr.HTML()
                    autoplay_html = gr.HTML()
                    autoplay_status = gr.HTML()

                hint_html = gr.HTML()

                stakeholder_select = gr.Dropdown(
                    label="Select stakeholder",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False,
                )

            with gr.Column(elem_classes=["dr-right"]):
                score_card_html = gr.HTML()
                signals_card_html = gr.HTML()
                how_it_works_html = gr.HTML()

        deal_close_popup_html = gr.HTML()

        outputs = [
            view_state, saved_runs,
            wf1_html, wf2_html, wf3_html, wf4_html,
            stakeholders_card_html, detail_card_html, hint_html,
            triggered_popup_html,
            stakeholder_select,
            send_btn,
            chip_btn_0, chip_btn_1, chip_btn_2, chip_btn_3,
            message_input,
            score_card_html, signals_card_html, how_it_works_html,
            deal_close_popup_html,
            simple_btn, medium_btn, hard_btn,
            autoplay_toggle, autoplay_html, autoplay_status,
            response_sent_feedback,
        ]

        def render_initial(vs, sr):
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            return (vs, sr) + _render_all_outputs(vs, sr)

        demo.load(fn=render_initial, inputs=[view_state, saved_runs], outputs=outputs)

        def handle_start(level: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            task_map = {"simple": "aligned", "medium": GUIDE_DATA["task"], "hard": "hostile_acquisition"}
            task = task_map.get(level, "aligned")
            updated = _run_reset(task, int(vs.get("seed", 42)), level, vs)
            return (updated, sr) + _render_all_outputs(updated, sr)

        def handle_level_select(level: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            updated = dict(vs)
            updated["level"] = level
            updated["workflow_step"] = 0
            return (updated, sr) + _render_all_outputs(updated, sr)

        def handle_stakeholder_click(selected: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            if not selected:
                return (vs, sr) + _render_all_outputs(vs, sr)
            parts = selected.split(" ", 1)
            sid = parts[1] if len(parts) > 1 else selected
            observation = vs.get("current_observation") or {}
            stakeholders = observation.get("stakeholders", {})
            if sid not in stakeholders:
                return (vs, sr) + _render_all_outputs(vs, sr)
            vs["selected_stakeholder"] = sid
            vs["popup_index"] = 0
            vs["popup_queue"] = [{"stakeholder_id": sid}]
            vs["show_hint"] = False
            vs["selected_chip"] = None
            vs["message_text"] = ""
            vs["workflow_step"] = 2
            vs["response_sent"] = False
            return (vs, sr) + _render_all_outputs(vs, sr)

        def handle_send(msg: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            if not vs.get("current_observation") or not vs.get("session_id"):
                updated = _run_reset("aligned", int(vs.get("seed", 42)), "simple", vs)
                return (updated, sr) + _render_all_outputs(updated, sr)
            selected = vs.get("selected_stakeholder", "all")
            action = DealRoomAction(
                action_type="direct_message", target=selected,
                target_ids=[selected] if selected != "all" else [],
                message=msg or "Understood.",
            )
            prev_info = vs.get("last_info", {})
            obs, reward, done, info, state = web_manager.step_session(vs["session_id"], action)
            triggered = _compute_triggered_stakeholders(info, prev_info)
            updated = _record_step(vs, action, obs, reward, done, info, state.model_dump())
            updated["triggered_stakeholders"] = triggered
            stakeholders_list = list(updated["current_observation"].get("stakeholders", {}).keys())
            current_sel = vs.get("selected_stakeholder")
            if current_sel not in stakeholders_list:
                current_sel = stakeholders_list[0] if stakeholders_list else None
            updated["selected_stakeholder"] = current_sel
            updated["popup_index"] = 0
            updated["popup_queue"] = [{"stakeholder_id": current_sel}] if current_sel else []
            updated["show_hint"] = not updated.get("round_complete")
            updated["selected_chip"] = None
            updated["message_text"] = ""
            sr, updated = _save_run_if_complete(updated, sr)
            return (updated, sr) + _render_all_outputs(updated, sr)

        def handle_run(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            if not vs.get("current_observation") or not vs.get("session_id"):
                updated = _run_reset("aligned", int(vs.get("seed", 42)), "simple", vs)
                return (updated, sr) + _render_all_outputs(updated, sr)
            prev_info = vs.get("last_info", {})
            action = _policy_action(vs["current_observation"])
            obs, reward, done, info, state = web_manager.step_session(vs["session_id"], action)
            triggered = _compute_triggered_stakeholders(info, prev_info)
            updated = _record_step(vs, action, obs, reward, done, info, state.model_dump())
            updated["triggered_stakeholders"] = triggered
            updated["popup_index"] = 0
            stakeholders_list = list(updated["current_observation"].get("stakeholders", {}).keys())
            updated["popup_queue"] = [{"stakeholder_id": s} for s in stakeholders_list]
            updated["show_hint"] = not updated.get("round_complete")
            updated["selected_chip"] = None
            updated["message_text"] = ""
            sr, updated = _save_run_if_complete(updated, sr)
            return (updated, sr) + _render_all_outputs(updated, sr)

        def handle_autoplay_toggle(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            if vs.get("auto_playing"):
                vs["auto_playing"] = False
                vs["auto_paused"] = False
            else:
                vs["auto_playing"] = True
                vs["auto_paused"] = False
            return (vs, sr) + _render_all_outputs(vs, sr)

        def handle_autoplay_action(action_type: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            if action_type == "stop":
                vs["auto_playing"] = False
                vs["auto_paused"] = False
            elif action_type == "pause":
                vs["auto_paused"] = True
            elif action_type == "start":
                vs["auto_paused"] = False
            return (vs, sr) + _render_all_outputs(vs, sr)

        def handle_autoplay_step(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            if not vs.get("auto_playing") or vs.get("auto_paused"):
                return (vs, sr) + _render_all_outputs(vs, sr)
            if vs.get("round_complete"):
                vs["auto_playing"] = False
                return (vs, sr) + _render_all_outputs(vs, sr)
            prev_info = vs.get("last_info", {})
            action = _policy_action(vs["current_observation"])
            obs, reward, done, info, state = web_manager.step_session(vs["session_id"], action)
            triggered = _compute_triggered_stakeholders(info, prev_info)
            updated = _record_step(vs, action, obs, reward, done, info, state.model_dump())
            updated["triggered_stakeholders"] = triggered
            stakeholders_list = list(updated["current_observation"].get("stakeholders", {}).keys())
            updated["popup_queue"] = [{"stakeholder_id": s} for s in stakeholders_list]
            updated["selected_chip"] = None
            updated["message_text"] = ""
            sr, updated = _save_run_if_complete(updated, sr)
            if updated.get("round_complete"):
                vs["auto_playing"] = False
            return (updated, sr) + _render_all_outputs(updated, sr)

        def handle_chip_select(chip_idx: int, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            chips = _generate_chips(vs)
            if 0 <= chip_idx < len(chips):
                vs["selected_chip"] = chips[chip_idx]
                vs["message_text"] = chips[chip_idx]
            return (vs, sr) + _render_all_outputs(vs, sr)

        chip_btn_0.click(fn=handle_chip_select, inputs=[gr.State(0), view_state, saved_runs], outputs=outputs)
        chip_btn_1.click(fn=handle_chip_select, inputs=[gr.State(1), view_state, saved_runs], outputs=outputs)
        chip_btn_2.click(fn=handle_chip_select, inputs=[gr.State(2), view_state, saved_runs], outputs=outputs)
        chip_btn_3.click(fn=handle_chip_select, inputs=[gr.State(3), view_state, saved_runs], outputs=outputs)
        send_btn.click(fn=handle_send, inputs=[message_input, view_state, saved_runs], outputs=outputs)
        run_btn.click(fn=handle_run, inputs=[view_state, saved_runs], outputs=outputs)
        step_btn.click(fn=handle_run, inputs=[view_state, saved_runs], outputs=outputs)
        stakeholder_select.change(fn=handle_stakeholder_click, inputs=[stakeholder_select, view_state, saved_runs], outputs=outputs)

        def level_html_to_btn(level: str) -> callable:
            def handler(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
                return handle_level_select(level, vs, sr)
            return handler

        simple_btn_html = gr.HTML()
        medium_btn_html = gr.HTML()
        hard_btn_html = gr.HTML()

        def update_level_btns(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            lvl_simple, lvl_medium, lvl_hard, lvl_extra = _build_level_buttons(vs)
            return (lvl_simple, lvl_medium, lvl_hard)

        def on_start_btn(level: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            task_map = {"simple": "aligned", "medium": GUIDE_DATA["task"], "hard": "hostile_acquisition"}
            task = task_map.get(level, "aligned")
            updated = _run_reset(task, int(vs.get("seed", 42)), level, vs)
            return (updated, sr) + _render_all_outputs(updated, sr)

        start_btn.click(fn=lambda vs, sr: on_start_btn("simple", vs, sr),
                      inputs=[view_state, saved_runs], outputs=outputs)

        def make_level_start_handler(level: str):
            def h(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
                return handle_start(level, vs, sr)
            return h

        simple_btn_event = gr.HTML().click(
            fn=make_level_start_handler("simple"),
            inputs=[view_state, saved_runs], outputs=outputs
        )
        medium_btn_event = gr.HTML().click(
            fn=make_level_start_handler("medium"),
            inputs=[view_state, saved_runs], outputs=outputs
        )
        hard_btn_event = gr.HTML().click(
            fn=make_level_start_handler("hard"),
            inputs=[view_state, saved_runs], outputs=outputs
        )

        autoplay_toggle_event = gr.HTML().click(
            fn=handle_autoplay_toggle,
            inputs=[view_state, saved_runs], outputs=outputs
        )

        def make_autoplay_handler(act: str):
            def h(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
                if act == "step":
                    return handle_autoplay_step(vs, sr)
                return handle_autoplay_action(act, vs, sr)
            return h

        autoplay_start_btn = gr.Button("", visible=False)
        autoplay_pause_btn = gr.Button("", visible=False)
        autoplay_stop_btn = gr.Button("", visible=False)

        autoplay_start_btn.click(
            fn=make_autoplay_handler("start"),
            inputs=[view_state, saved_runs], outputs=outputs
        )
        autoplay_pause_btn.click(
            fn=make_autoplay_handler("pause"),
            inputs=[view_state, saved_runs], outputs=outputs
        )
        autoplay_stop_btn.click(
            fn=make_autoplay_handler("stop"),
            inputs=[view_state, saved_runs], outputs=outputs
        )

        autoplay_start_btn2 = gr.Button("", visible=False)
        autoplay_pause_btn2 = gr.Button("", visible=False)
        autoplay_stop_btn2 = gr.Button("", visible=False)

        autoplay_start_btn2.click(
            fn=make_autoplay_handler("step"),
            inputs=[view_state, saved_runs], outputs=outputs
        )
        autoplay_pause_btn2.click(
            fn=make_autoplay_handler("pause"),
            inputs=[view_state, saved_runs], outputs=outputs
        )
        autoplay_stop_btn2.click(
            fn=make_autoplay_handler("stop"),
            inputs=[view_state, saved_runs], outputs=outputs
        )

    return demo