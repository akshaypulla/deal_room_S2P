"""DealRoom - Clean, simple Gradio interface for B2B negotiation."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata

from models import DealRoomAction, DealRoomObservation
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


CLEAN_CSS = """
.dealroom-clean { background: #0f1117; font-family: 'Inter','Segoe UI',system-ui,sans-serif; min-height: 100vh; }
.dealroom-clean * { color: #e5e7eb; }
.dr-header { text-align: center; padding: 20px 0 16px; border-bottom: 1px solid #21262d; margin-bottom: 16px; }
.dr-header h1 { font-size: 1.6rem; font-weight: 700; color: #f9fafb; margin: 0 0 4px; }
.dr-header p { color: #8b949e; font-size: 0.88rem; margin: 0; }

.level-row { display: flex; gap: 10px; margin-bottom: 16px; align-items: center; }
.level-btn { flex: 1; padding: 14px 8px !important; border-radius: 10px !important; font-size: 0.85rem !important; font-weight: 700 !important; border: 2px solid #30363d !important; background: #161b22 !important; color: #8b949e !important; cursor: pointer; transition: all 0.2s; }
.level-btn.selected { border-color: #ff6a00 !important; background: rgba(255,106,0,0.15) !important; color: #ff6a00 !important; box-shadow: 0 0 20px rgba(255,106,0,0.3); }
.level-btn:hover { border-color: #ff6a00 !important; color: #ff6a00 !important; }
.level-btn.aligned-task { border-left: 4px solid #3fb950; }
.level-btn.conflicted-task { border-left: 4px solid #d29922; }
.level-btn.hostile-task { border-left: 4px solid #f85149; }
.start-btn { padding: 14px 24px !important; font-size: 1rem !important; font-weight: 700 !important; background: linear-gradient(135deg,#ff6a00,#e05500) !important; border: none !important; border-radius: 10px !important; color: #fff !important; box-shadow: 0 4px 20px rgba(255,106,0,0.4); }
.start-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(255,106,0,0.5); }

.main-grid { display: grid; grid-template-columns: 1fr 320px; gap: 16px; }
.left-col { display: flex; flex-direction: column; gap: 14px; }
.right-col { display: flex; flex-direction: column; gap: 14px; }

.stakeholders-card { background: #161b22; border-radius: 12px; border: 1px solid #21262d; padding: 14px; }
.stakeholders-card h3 { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; margin: 0 0 10px; font-weight: 600; }
.stakeholders-list { display: flex; flex-direction: column; gap: 6px; }
.stakeholder-item { display: flex; align-items: center; gap: 10px; padding: 12px; border-radius: 8px; background: #0d1117; border: 2px solid #21262d; cursor: pointer; transition: all 0.2s; }
.stakeholder-item:hover { border-color: #30363d; background: #161b22; }
.stakeholder-item.selected { border-color: #ff6a00; background: rgba(255,106,0,0.1); box-shadow: 0 0 16px rgba(255,106,0,0.25); }
.stakeholder-item.aligned { border-left: 4px solid #3fb950; }
.stakeholder-item.blocking { border-left: 4px solid #f85149; animation: pulse-blocking 2s infinite; }
.stakeholder-item.uncertain { border-left: 4px solid #d29922; }
.stakeholder-item.triggered { border-left: 4px solid #a855f7; background: rgba(168,85,247,0.08); animation: pulse-triggered 2s infinite; }
.stakeholder-item.dimmed { opacity: 0.5; pointer-events: none; }
@keyframes pulse-blocking { 0%,100%{box-shadow:0 0 0 0 rgba(248,81,73,0)} 50%{box-shadow:0 0 12px 3px rgba(248,81,73,0.35)} }
@keyframes pulse-triggered { 0%,100%{box-shadow:0 0 0 0 rgba(168,85,247,0)} 50%{box-shadow:0 0 16px 4px rgba(168,85,247,0.45)} }
.s-icon { font-size: 1.4rem; width: 32px; text-align: center; }
.s-name { font-weight: 600; font-size: 0.9rem; color: #e6edf3; }
.s-role { font-size: 0.72rem; color: #6e7681; }
.s-status { font-size: 0.65rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
.s-status.green { background: rgba(63,185,80,0.15); color: #3fb950; }
.s-status.red { background: rgba(248,81,73,0.15); color: #f85149; }
.s-status.amber { background: rgba(210,153,34,0.15); color: #d29922; }
.s-status.purple { background: rgba(168,85,247,0.15); color: #a855f7; }

.stakeholder-select-wrap { margin-bottom: 12px; }
.stakeholder-select-wrap label { font-size: 0.7rem !important; color: #8b949e !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600 !important; margin-bottom: 6px !important; display: block; }
.stakeholder-select-wrap select { width: 100%; background: #0d1117 !important; border: 2px solid #30363d !important; border-radius: 8px !important; color: #c9d1d9 !important; padding: 10px 12px !important; font-size: 0.9rem !important; cursor: pointer; }
.stakeholder-select-wrap select:focus { border-color: #ff6a00 !important; outline: none; box-shadow: 0 0 0 3px rgba(255,106,0,0.15) !important; }

.detail-card { background: #161b22; border-radius: 12px; border: 1px solid #21262d; padding: 14px; cursor: pointer; transition: all 0.2s; }
.detail-card:hover { border-color: #30363d; }
.detail-card h3 { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; margin: 0 0 10px; font-weight: 600; }
.detail-quote { background: #0d1117; border-left: 3px solid #ff6a00; padding: 10px 12px; border-radius: 0 6px 6px 0; font-style: italic; color: #c9d1d9; font-size: 0.85rem; line-height: 1.5; margin-bottom: 10px; cursor: pointer; transition: background 0.2s; }
.detail-quote:hover { background: #161b22; }
.detail-needs { background: rgba(255,106,0,0.08); border: 1px solid rgba(255,106,0,0.2); border-radius: 6px; padding: 8px 10px; font-size: 0.8rem; }
.detail-needs strong { color: #ff6a00; }
.detail-needs p { color: #8b949e; margin: 4px 0 0; }
.detail-hint { font-size: 0.72rem; color: #6e7681; margin-top: 8px; font-style: italic; }

.message-card { background: #161b22; border-radius: 12px; border: 1px solid #21262d; padding: 14px; }
.message-card h3 { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; margin: 0 0 10px; font-weight: 600; }
.chips-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.chip-btn { padding: 8px 12px !important; border-radius: 6px !important; border: 1px solid #30363d !important; background: #0d1117 !important; color: #c9d1d9 !important; font-size: 0.78rem !important; cursor: pointer; transition: all 0.15s; }
.chip-btn:hover { border-color: #ff6a00 !important; background: rgba(255,106,0,0.1) !important; color: #ff6a00 !important; }
.chip-btn.selected { border-color: #ff6a00 !important; background: rgba(255,106,0,0.2) !important; color: #ff6a00 !important; }
.message-input { width: 100%; background: #0d1117 !important; border: 2px solid #30363d !important; border-radius: 8px !important; color: #c9d1d9 !important; padding: 12px !important; font-size: 0.9rem !important; margin-bottom: 10px; form: message-form; }
.message-input:focus { border-color: #ff6a00 !important; box-shadow: 0 0 0 3px rgba(255,106,0,0.15) !important; }
#message-form { display: contents; }
.send-btn { width: 100%; padding: 14px !important; font-size: 1rem !important; font-weight: 700 !important; background: linear-gradient(135deg,#ff6a00,#e05500) !important; border: none !important; border-radius: 8px !important; color: #fff !important; box-shadow: 0 4px 16px rgba(255,106,0,0.3); cursor: pointer; transition: all 0.2s; }
.send-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(255,106,0,0.4); }
.send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

.run-bar { display: flex; gap: 8px; margin-top: 10px; }
.run-btn { flex: 2; padding: 12px !important; font-size: 0.9rem !important; font-weight: 600 !important; background: linear-gradient(135deg,#238636,#1a7f37) !important; border: none !important; border-radius: 8px !important; color: #fff !important; }
.run-btn:disabled { opacity: 0.5; background: #21262d !important; }
.step-btn { flex: 1; padding: 12px !important; font-size: 0.9rem !important; background: #21262d !important; border: 1px solid #30363d !important; border-radius: 8px !important; color: #c9d1d9 !important; }

.autoplay-bar { display: flex; gap: 8px; margin-top: 10px; }
.autoplay-start { flex: 1; padding: 12px !important; font-size: 0.9rem !important; font-weight: 600 !important; background: linear-gradient(135deg,#238636,#1a7f37) !important; border: none !important; border-radius: 8px !important; color: #fff !important; }
.autoplay-pause { flex: 1; padding: 12px !important; font-size: 0.9rem !important; font-weight: 600 !important; background: linear-gradient(135deg,#d29922,#b8860b) !important; border: none !important; border-radius: 8px !important; color: #0d1117 !important; }
.autoplay-stop { flex: 1; padding: 12px !important; font-size: 0.9rem !important; background: #374151 !important; border: 1px solid #30363d !important; border-radius: 8px !important; color: #c9d1d9 !important; }

.score-card { background: linear-gradient(135deg,#1c2128,#0d1117); border: 2px solid #ff6a00; border-radius: 12px; padding: 16px; text-align: center; }
.score-value { font-size: 3rem; font-weight: 700; color: #ff6a00; line-height: 1; text-shadow: 0 0 30px rgba(255,106,0,0.4); }
.score-label { font-size: 0.75rem; color: #6e7681; margin-top: 4px; }
.score-delta { font-size: 1.1rem; margin-top: 6px; font-weight: 600; }

.info-card { background: rgba(63,185,80,0.06); border: 1px solid rgba(63,185,80,0.2); border-radius: 10px; padding: 12px; font-size: 0.8rem; color: #7ee787; line-height: 1.5; }
.info-card h4 { margin: 0 0 8px; color: #3fb950; font-size: 0.85rem; }
.info-card li { margin-bottom: 4px; }

.triggered-popup { background: rgba(168,85,247,0.1); border: 2px solid rgba(168,85,247,0.4); border-radius: 10px; padding: 12px 14px; margin-top: 10px; font-size: 0.82rem; color: #a855f7; animation: fade-in 0.3s ease; }
.triggered-popup strong { color: #a855f7; }
@keyframes fade-in { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:translateY(0)} }

.hint-box { background: rgba(255,106,0,0.1); border: 1px solid rgba(255,106,0,0.3); border-radius: 8px; padding: 10px 12px; font-size: 0.82rem; color: #ff6a00; line-height: 1.5; margin-top: 10px; }
.hint-box.success { background: rgba(63,185,80,0.1); border-color: rgba(63,185,80,0.3); color: #3fb950; }
.hint-box.error { background: rgba(248,81,73,0.1); border-color: rgba(248,81,73,0.3); color: #f85149; }

/* BIG DEAL CLOSE POPUP */
.deal-close-popup { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 9999; align-items: center; justify-content: center; }
.deal-close-popup.show { display: flex !important; }
.deal-close-content { background: linear-gradient(135deg,#161b22,#0d1117); border: 3px solid #3fb950; border-radius: 20px; padding: 48px 60px; text-align: center; max-width: 500px; animation: popup-in 0.5s cubic-bezier(0.34,1.56,0.64,1); box-shadow: 0 0 80px rgba(63,185,80,0.4), 0 25px 60px rgba(0,0,0,0.6); }
@keyframes popup-in { from{opacity:0;transform:scale(0.6)} to{opacity:1;transform:scale(1)} }
.deal-close-icon { font-size: 5rem; margin-bottom: 16px; }
.deal-close-title { font-size: 2rem; font-weight: 700; color: #3fb950; margin: 0 0 10px; }
.deal-close-subtitle { font-size: 0.95rem; color: #8b949e; margin: 0 0 24px; line-height: 1.5; }
.deal-close-score { font-size: 4rem; font-weight: 700; color: #ff6a00; margin: 10px 0; text-shadow: 0 0 40px rgba(255,106,0,0.5); }
.deal-close-score-label { font-size: 0.8rem; color: #6e7681; margin-bottom: 28px; }
.deal-close-dismiss { padding: 14px 36px !important; background: #21262d !important; border: 1px solid #30363d !important; border-radius: 10px !important; color: #c9d1d9 !important; font-size: 0.95rem !important; cursor: pointer; }
.deal-close-dismiss:hover { border-color: #ff6a00 !important; color: #ff6a00 !important; }

.complete-card { background: linear-gradient(135deg,rgba(63,185,80,0.1),rgba(255,106,0,0.05)); border: 2px solid #3fb950; border-radius: 12px; padding: 20px; text-align: center; }
.complete-card h2 { color: #3fb950; font-size: 1.2rem; margin: 0 0 10px; }
.complete-card .final-score { font-size: 2.5rem; font-weight: 700; color: #ff6a00; }
.complete-card p { color: #8b949e; font-size: 0.8rem; margin-top: 6px; }

.blocker-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.blocker-tag { display: inline-flex; align-items: center; gap: 3px; padding: 3px 8px; border-radius: 999px; font-size: 0.68rem; background: rgba(248,81,73,0.15); color: #f85149; }

/* Hidden elements for internal use */
.dr-hidden { display: none !important; }
.stakeholder-dropdown-wrap { display: none !important; }

/* Workflow step indicator */
.workflow-steps { display: flex; justify-content: center; gap: 8px; margin-bottom: 16px; }
.workflow-step { display: flex; align-items: center; gap: 6px; padding: 8px 14px; border-radius: 20px; background: #161b22; border: 1px solid #21262d; font-size: 0.75rem; color: #6e7681; }
.workflow-step.active { background: rgba(255,106,0,0.15); border-color: #ff6a00; color: #ff6a00; }
.workflow-step.completed { background: rgba(63,185,80,0.1); border-color: #3fb950; color: #3fb950; }
.workflow-step-num { width: 20px; height: 20px; border-radius: 50%; background: #21262d; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.7rem; }
.workflow-step.active .workflow-step-num { background: #ff6a00; color: #fff; }
.workflow-step.completed .workflow-step-num { background: #3fb950; color: #fff; }

/* Workflow step bar - always visible */
.workflow-step-bar { display: flex; justify-content: center; gap: 12px; margin-bottom: 20px; padding: 0 16px; }
.workflow-step-bar .step { display: flex; align-items: center; gap: 8px; padding: 10px 18px; border-radius: 24px; background: #161b22; border: 2px solid #21262d; font-size: 0.85rem; font-weight: 600; color: #6e7681; transition: all 0.3s; }
.workflow-step-bar .step.active { background: rgba(255,106,0,0.15); border-color: #ff6a00; color: #ff6a00; box-shadow: 0 0 16px rgba(255,106,0,0.25); }
.workflow-step-bar .step.completed { background: rgba(63,185,80,0.15); border-color: #3fb950; color: #3fb950; }
.workflow-step-bar .step-num { width: 24px; height: 24px; border-radius: 50%; background: #21262d; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.75rem; }
.workflow-step-bar .step.active .step-num { background: #ff6a00; color: #fff; }
.workflow-step-bar .step.completed .step-num { background: #3fb950; color: #fff; }
.workflow-step-bar .step-arrow { color: #30363d; font-size: 1.2rem; }
"""


class DealRoomWebManager:
    def __init__(self, pool: DealRoomSessionPool, metadata: EnvironmentMetadata):
        self.pool = pool
        self.metadata = metadata

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
        "task": "aligned", "seed": 42, "level": "simple",
        "session_id": None, "selected_stakeholder": None, "current_observation": None,
        "current_state": None, "trace": [], "status_message": "",
        "popup_queue": [], "popup_index": 0,
        "round_started": False, "show_hint": False, "hint_text": "",
        "selected_chip": None, "message_text": "", "auto_playing": False, "auto_paused": False,
        "last_score": 0.0, "score_delta": None,
        "round_complete": False, "use_llm_agent": False,
        "workflow_step": 0, "response_sent": False, "triggered_stakeholders": [],
        "deal_close_showing": False, "last_info": {},
        "proposal_submitted": False,
    }
    if not isinstance(view_state, dict):
        return base
    merged = dict(base)
    merged.update(view_state)
    for key in ("trace", "popup_queue", "triggered_stakeholders"):
        if not isinstance(merged.get(key), list):
            merged[key] = []
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
        return ["Acknowledge concern", "Request details", "Provide information", "Move forward"]
    payload = observation.get("stakeholders", {}).get(selected, {})
    role_type = ROLE_BY_ID.get(selected, "")
    band = _approval_band(observation, selected)
    requested = observation.get("requested_artifacts", {}).get(selected, [])
    if role_type == "finance" or "budget" in str(requested) or "roi" in str(requested):
        return ["Provide ROI justification", "Address budget concern", "Ask for breakdown", "Propose terms"]
    elif role_type == "legal_compliance":
        return ["Clarify contract terms", "Address compliance gap", "Provide documentation", "Suggest alternative"]
    elif role_type == "procurement":
        return ["Justify vendor selection", "Compare alternatives", "Address delivery", "Request flexibility"]
    elif role_type == "executive_sponsor":
        return ["Summarize business case", "Highlight key benefits", "Address risks", "Propose next steps"]
    elif role_type == "technical":
        return ["Share implementation plan", "Provide technical specs", "Address security", "Clarify integration"]
    elif role_type == "operations":
        return ["Share timeline", "Address support plan", "Provide training details", "Clarify handoff"]
    elif band == "blocking":
        return ["Address their concern", "Request clarification", "Provide evidence", "Escalate"]
    elif band == "uncertain":
        return ["Clarify position", "Provide more details", "Address worries", "Build confidence"]
    return ["Acknowledge point", "Build on alignment", "Move discussion forward", "Confirm understanding"]


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


def _run_reset(task: str, seed: int, level: str, view_state: Dict[str, Any]) -> Dict[str, Any]:
    current = _normalize_view_state(view_state)
    session_id, obs, state = web_manager.reset_session(
        task_id=task, seed=int(seed), session_id=current.get("session_id")
    )
    observation = obs.model_dump()
    stakeholders = list(observation.get("stakeholders", {}).keys())
    updated = dict(current)
    updated.update({
        "task": task, "seed": int(seed), "level": level,
        "session_id": session_id,
        "current_observation": observation,
        "current_state": state.model_dump(),
        "selected_stakeholder": stakeholders[0] if stakeholders else None,
        "popup_queue": [{"stakeholder_id": s} for s in stakeholders],
        "popup_index": 0,
        "round_started": True,
        "show_hint": True,
        "hint_text": "Click on a stakeholder, then type or click a suggestion, then click Send Response",
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
        "proposal_submitted": False,
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

    prev_info = view_state.get("last_info", {})
    propagation_deltas = info.get("propagation_deltas", {})
    triggered = []
    if propagation_deltas:
        for sid, delta in propagation_deltas.items():
            if delta and abs(float(delta)) > 0.05:
                triggered.append(sid)
    updated["triggered_stakeholders"] = triggered

    if done:
        updated["status_message"] = f"Round complete! Final score: {reward:.2f}"
        updated["show_hint"] = False
        updated["deal_close_showing"] = True
    else:
        updated["status_message"] = f"Step {step_num} | Reward: {reward:.2f}"
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


def _build_stakeholder_list(view_state: Dict[str, Any], stakeholder_ids: List[str]) -> str:
    view_state = _normalize_view_state(view_state)
    observation = view_state.get("current_observation") or {}
    selected = view_state.get("selected_stakeholder")
    stakeholders = observation.get("stakeholders", {})
    triggered = view_state.get("triggered_stakeholders", [])

    if not stakeholders:
        return ""

    items = []
    for sid in stakeholder_ids:
        band = _approval_band(observation, sid)
        is_selected = sid == selected
        is_triggered = sid in triggered
        role_type = ROLE_BY_ID.get(sid, "")
        payload = stakeholders.get(sid, {})
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
            f"<div class='s-name'>{_escape(sid)}</div>"
            f"<div class='s-role'>{_escape(payload.get('role', sid))}</div>"
            f"<span class='s-status {status_class}'>{status_text}</span></div>"
        )
    return (
        "<div class='stakeholders-card'>"
        "<h3>👥 Stakeholders</h3>"
        "<div class='stakeholders-list'>" + "".join(items) + "</div>"
        "</div>"
    )


def _build_detail_panel(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    observation = view_state.get("current_observation") or {}
    selected = view_state.get("selected_stakeholder")
    if not selected or not observation or selected not in observation.get("stakeholders", {}):
        return "<div class='detail-card'><h3>📋 Details</h3><p style='color:#6e7681;font-size:0.85rem'>Select a stakeholder to see their message</p></div>"
    message = observation.get("stakeholder_messages", {}).get(selected, "No message yet.")
    requested = observation.get("requested_artifacts", {}).get(selected, [])
    role_type = ROLE_BY_ID.get(selected, "")
    icon = ROLE_ICONS.get(role_type, "👤")
    needs_text = ", ".join(r.replace("_", " ") for r in requested) if requested else "Nothing specific requested yet"
    return (
        f"<div class='detail-card' data-selected-stakeholder='{_escape(selected)}'>"
        f"<h3>{icon} {_escape(selected)} Details</h3>"
        f"<div class='detail-quote' title='Click to select this stakeholder'>\"{_escape(message)}\"</div>"
        f"<div class='detail-needs'><strong>What they need:</strong><p>{_escape(needs_text)}</p></div>"
        f"<p class='detail-hint'>💡 Click the message above or card on the left to select this stakeholder</p>"
        f"</div>"
    )


def _build_message_panel(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    observation = view_state.get("current_observation") or {}
    selected = view_state.get("selected_stakeholder")
    chips = _generate_chips(view_state)
    msg_txt = view_state.get("message_text", "")

    chip_btns = ""
    for i, chip in enumerate(chips[:4]):
        if chip:
            selected_cls = " selected" if view_state.get("selected_chip") == chip else ""
            chip_btns += f"<button class='chip-btn{selected_cls}' data-chip='{_escape(chip)}'>{_escape(chip)}</button>"

    return (
        "<div class='message-card'>"
        f"<h3>💬 {'Your Response' if selected else 'Select a stakeholder first'}</h3>"
        f"<div class='chips-row' data-chip-wrapper='true'>{chip_btns}</div>"
        "</div>"
    )


def _build_score_panel(view_state: Dict[str, Any]) -> str:
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
        blocker_html = f"<div class='blocker-tags'>{tags}</div>"

    return (
        "<div class='score-card'>"
        f"<div class='score-value'>{current_score:.2f}</div>"
        "<div class='score-label'>Current Score</div>"
        f"{delta_html}{blocker_html}</div>"
    )


def _build_triggered_popup(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    triggered = view_state.get("triggered_stakeholders", [])
    if not triggered:
        return ""
    names = ", ".join(f"<strong>{_escape(s)}</strong>" for s in triggered)
    return (
        f"<div class='triggered-popup'>"
        f"<strong>⚡ Cascade Effect!</strong> {names} were affected by this action — "
        f"their opinions shifted due to cross-stakeholder influence.</div>"
    )


def _build_workflow_steps(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    response_sent = view_state.get("response_sent", False)
    workflow_step = view_state.get("workflow_step", 0)
    round_started = view_state.get("round_started", False)

    step_info = [
        ("1", "Select\nStakeholder"),
        ("2", "Send\nResponse"),
        ("3", "Run\nRound"),
    ]

    items = []
    for i, (num, label) in enumerate(step_info):
        if not round_started:
            cls = ""
        elif i < workflow_step or (i == 1 and response_sent):
            cls = "completed"
        elif i == workflow_step and not response_sent:
            cls = "active"
        else:
            cls = ""

        arrow = "<span class='step-arrow'>→</span>" if i < len(step_info) - 1 else ""
        items.append(
            f"<div class='step {cls}'>"
            f"<span class='step-num'>{num}</span>"
            f"<span>{label.replace(chr(10), '<br>')}</span>"
            f"</div>{arrow}"
        )

    return "<div class='workflow-step-bar'>" + "".join(items) + "</div>"


def _build_hint(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    round_started = view_state.get("round_started", False)
    round_complete = view_state.get("round_complete", False)
    response_sent = view_state.get("response_sent", False)
    selected = view_state.get("selected_stakeholder")

    if round_complete:
        return "<div class='hint-box success'>🎉 Deal complete! Check your final score on the right panel.</div>"
    if not round_started:
        return "<div class='hint-box'>👆 Choose a difficulty level (Easy/Medium/Hard) and click <strong>Start Round</strong> to begin negotiating.</div>"
    if not selected:
        return "<div class='hint-box'>📝 <strong>Step 1:</strong> Click on a <strong>stakeholder card</strong> on the left to select them.</div>"
    if not response_sent:
        return "<div class='hint-box'>📝 <strong>Step 2:</strong> Type a message or click a <strong>suggestion chip</strong>, then click <strong>Send Response</strong>.</div>"
    return "<div class='hint-box'>📝 <strong>Step 3:</strong> Click <strong>Run Round</strong> to send your message and see stakeholder reactions.</div>"


def _build_how_it_works() -> str:
    return (
        "<div class='info-card'>"
        "<h4>📖 How to Play</h4>"
        "<ol style='margin:0;padding-left:16px'>"
        "<li><strong>Start a round</strong> — Choose level and click Start Round</li>"
        "<li><strong>Select a stakeholder</strong> — Click on their card above</li>"
        "<li><strong>Send a response</strong> — Type or click a suggestion, then Send Response</li>"
        "<li><strong>Run the round</strong> — Click Run Round to see reactions</li>"
        "<li><strong>Goal</strong> — Get all stakeholders to 'Aligned' to close the deal</li>"
        "</ol>"
        "<p style='margin-top:8px;color:#a855f7;'><strong>🟣 Purple = Triggered:</strong> When one stakeholder affects another, they'll pulse purple. Click to see what changed.</p>"
        "</div>"
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
        subtitle = "Congratulations! You successfully navigated the negotiation and closed the deal with all stakeholders aligned."
    elif "veto" in terminal_outcome:
        icon = "⚠️"
        title = "Deal Vetoed"
        subtitle = "A stakeholder blocked the deal. Review their concerns (shown in red blockers) and try again."
    elif "timeout" in terminal_outcome or "max_rounds" in terminal_outcome:
        icon = "⏱️"
        title = "Deal Timed Out"
        subtitle = "The negotiation exceeded the maximum rounds. Try being more efficient with your messages."
    else:
        icon = "🏁"
        title = "Deal Ended"
        subtitle = "The negotiation has concluded. Review your score and try again."
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


def _build_level_buttons(view_state: Dict[str, Any]) -> Tuple[str, str, str]:
    view_state = _normalize_view_state(view_state)
    current_level = view_state.get("level", "simple")
    levels = [
        ("simple", "Easy", "Aligned interests", "aligned-task"),
        ("medium", "Medium", "Conflicted committee", "conflicted-task"),
        ("hard", "Hard", "Hostile acquisition", "hostile-task"),
    ]
    btns = []
    for level, label, desc, color_cls in levels:
        is_selected = level == current_level
        cls = f"level-btn {color_cls}" + (" selected" if is_selected else "")
        btns.append(f"<button class='{cls}' data-level='{level}'>{label}<br><small style='opacity:0.7;font-size:0.65rem'>{desc}</small></button>")
    return (btns[0], btns[1], btns[2])


web_manager: Optional[DealRoomWebManager] = None


def build_clean_tab(pool: DealRoomSessionPool, metadata: EnvironmentMetadata) -> gr.Blocks:
    global web_manager
    web_manager = DealRoomWebManager(pool, metadata)

    demo = gr.Blocks(elem_classes=["dealroom-clean"])
    with demo:
        gr.HTML(f"<style>{CLEAN_CSS}</style>")

        view_state = gr.State(_normalize_view_state(None))
        saved_runs = gr.State([])

        with gr.Column(elem_classes=["dr-header"]):
            gr.HTML("<h1>🎯 DealRoom</h1>"
                    "<p>B2B Negotiation Simulator — Get all stakeholders aligned to close the deal</p>")

        workflow_steps_html = gr.HTML()

        with gr.Row(elem_classes=["level-row"]):
            simple_btn = gr.HTML()
            medium_btn = gr.HTML()
            hard_btn = gr.HTML()
            start_btn = gr.Button("▶ Start Round", elem_classes=["start-btn"], variant="primary")

        with gr.Row(elem_classes=["main-grid"]):
            with gr.Column(elem_classes=["left-col"]):
                stakeholder_dropdown = gr.Dropdown(
                    choices=[], label="👈 Select a stakeholder to focus on",
                    elem_classes=["stakeholder-select-wrap"]
                )
                stakeholder_hidden_input = gr.Textbox(
                    elem_id="dr-stakeholder-input",
                    elem_classes=["dr-hidden"],
                    visible=False,
                )
                stakeholders_card_html = gr.HTML()
                detail_card_html = gr.HTML()
                detail_click_handler = gr.HTML(elem_classes=["dr-hidden"], visible=True)
                message_card_html = gr.HTML()
                chip_wrapper_html = gr.HTML()
                message_input = gr.Textbox(
                    lines=2, placeholder="Type your message...",
                    elem_classes=["message-input"],
                    elem_id="dr-message-input",
                )
                send_btn = gr.Button("Send Response", elem_classes=["send-btn"], variant="primary")
                with gr.Row(elem_classes=["run-bar"]):
                    run_btn = gr.Button("▶ Run Round", elem_classes=["run-btn"], variant="primary")
                    step_btn = gr.Button("⏭ Step", elem_classes=["step-btn"], variant="secondary")
                hint_html = gr.HTML()
                triggered_popup_html = gr.HTML()

            with gr.Column(elem_classes=["right-col"]):
                score_card_html = gr.HTML()
                how_it_works_html = gr.HTML()

        deal_close_popup_html = gr.HTML()

        outputs = [
            view_state, saved_runs,
            stakeholders_card_html, detail_card_html, message_card_html,
            send_btn, run_btn, step_btn,
            hint_html, triggered_popup_html,
            score_card_html, how_it_works_html,
            deal_close_popup_html,
            simple_btn, medium_btn, hard_btn,
            stakeholder_dropdown,
            message_input,
            chip_wrapper_html,
            detail_click_handler,
            workflow_steps_html,
            stakeholder_hidden_input,
        ]

        def render_initial(vs, sr):
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            lvl_simple, lvl_medium, lvl_hard = _build_level_buttons(vs)
            return (vs, sr, "", "", _build_message_panel(vs), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    _build_hint(vs), "", _build_score_panel(vs), _build_how_it_works(), _build_deal_close_popup(vs),
                    lvl_simple, lvl_medium, lvl_hard,
                    gr.update(visible=False), gr.update(visible=False), "", "", "", "", "")

        demo.load(fn=render_initial, inputs=[view_state, saved_runs], outputs=outputs, js="""
        function() {
        window.showDealClosePopup = function() { document.querySelectorAll('.deal-close-popup').forEach(function(e) { e.classList.add('show'); }); };
        window.dismissDealClosePopup = function() { document.querySelectorAll('.deal-close-popup').forEach(function(e) { e.classList.remove('show'); }); };
        window._drHandleChipClick = function(chipBtn) {
          document.querySelectorAll('.chip-btn').forEach(function(b) {
            b.style.background = '';
            b.style.borderColor = '';
          });
          chipBtn.style.background = 'rgba(255,106,0,0.3)';
          chipBtn.style.borderColor = '#ff6a00';
          var msgInput = document.querySelector('#dr-message-input textarea') || document.querySelector('#dr-message-input input');
          if (msgInput) {
            var chipValue = chipBtn.getAttribute('data-chip') || chipBtn.textContent.trim();
            Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value') || 
            Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
            msgInput.value = chipValue;
            msgInput.dispatchEvent(new Event('input', { bubbles: true }));
            msgInput.dispatchEvent(new Event('change', { bubbles: true }));
          }
        };
        window._drHandleStakeholderClick = function(stakeholderItem) {
          var stakeholderId = stakeholderItem.getAttribute('data-stakeholder');
          if (stakeholderId) {
            window._drCurrentSelectedStakeholder = stakeholderId;
            var hiddenInput = document.querySelector('#dr-stakeholder-input input') || document.querySelector('#dr-stakeholder-input textarea');
            if (!hiddenInput) hiddenInput = document.querySelector('#dr-stakeholder-input');
            if (hiddenInput) {
              hiddenInput.value = stakeholderId;
              hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
              hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
            }
          }
        };
        window._drSetupHandlers = function() {
          document.addEventListener('click', function(e) {
            var chipBtn = e.target.closest('.chip-btn');
            if (chipBtn) {
              e.stopPropagation();
              window._drHandleChipClick(chipBtn);
            }
            var stakeholderItem = e.target.closest('.stakeholder-item');
            if (stakeholderItem) {
              e.stopPropagation();
              window._drHandleStakeholderClick(stakeholderItem);
            }
          });
        };
        window._drCurrentSelectedStakeholder = null;
        window._drStakeholderCallback = null;
        if (document.readyState === 'loading') {
          document.addEventListener('DOMContentLoaded', window._drSetupHandlers);
        } else {
          window._drSetupHandlers();
        }
        }
        """)

        def handle_start(level: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            task_map = {"simple": "aligned", "medium": GUIDE_DATA["task"], "hard": "hostile_acquisition"}
            task = task_map.get(level, "aligned")
            updated = _run_reset(task, int(vs.get("seed", 42)), level, vs)
            lvl_simple, lvl_medium, lvl_hard = _build_level_buttons(updated)
            stakeholder_ids = list(updated["current_observation"].get("stakeholders", {}).keys())
            first_stakeholder = stakeholder_ids[0] if stakeholder_ids else ""
            return (updated, sr,
                    _build_stakeholder_list(updated, stakeholder_ids), _build_detail_panel(updated), _build_message_panel(updated),
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    _build_hint(updated), _build_triggered_popup(updated),
                    _build_score_panel(updated), _build_how_it_works(), _build_deal_close_popup(updated),
                    lvl_simple, lvl_medium, lvl_hard,
                    gr.update(choices=stakeholder_ids, value=first_stakeholder, visible=True),
                    gr.update(visible=True, value=""),
                    "", "", _build_workflow_steps(updated), "")

        def handle_stakeholder_click(stakeholder_id: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            stakeholder_ids = list((vs.get("current_observation") or {}).get("stakeholders", {}).keys())
            if not stakeholder_id or stakeholder_id not in stakeholder_ids:
                return (vs, sr) + tuple([gr.update()] * (len(outputs) - 2))
            updated = dict(vs)
            updated["selected_stakeholder"] = stakeholder_id
            updated["selected_chip"] = None
            updated["message_text"] = ""
            updated["response_sent"] = False
            return (updated, sr,
                    _build_stakeholder_list(updated, stakeholder_ids), _build_detail_panel(updated), _build_message_panel(updated),
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    _build_hint(updated), _build_triggered_popup(updated),
                    _build_score_panel(updated), _build_how_it_works(), _build_deal_close_popup(updated),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(choices=stakeholder_ids, value=updated.get("selected_stakeholder", "")),
                    gr.update(visible=True, value=""),
                    "", "", _build_workflow_steps(updated), "")

        def handle_send(msg: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            if not vs.get("current_observation") or not vs.get("session_id"):
                updated = _run_reset("aligned", int(vs.get("seed", 42)), "simple", vs)
            else:
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
                stakeholder_ids = list(updated["current_observation"].get("stakeholders", {}).keys())
                current_sel = vs.get("selected_stakeholder")
                if current_sel not in stakeholder_ids:
                    current_sel = stakeholder_ids[0] if stakeholder_ids else None
                updated["selected_stakeholder"] = current_sel
                updated["show_hint"] = not updated.get("round_complete")
                updated["selected_chip"] = None
                updated["message_text"] = ""
                if updated.get("round_complete"):
                    updated["deal_close_showing"] = True

            lvl_simple, lvl_medium, lvl_hard = _build_level_buttons(updated)
            stakeholder_ids = list(updated["current_observation"].get("stakeholders", {}).keys())
            return (updated, sr,
                    _build_stakeholder_list(updated, stakeholder_ids), _build_detail_panel(updated), _build_message_panel(updated),
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),
                    _build_hint(updated), _build_triggered_popup(updated),
                    _build_score_panel(updated), _build_how_it_works(), _build_deal_close_popup(updated),
                    lvl_simple, lvl_medium, lvl_hard,
                    gr.update(choices=stakeholder_ids, value=updated.get("selected_stakeholder", "")),
                    gr.update(value=""),
                    "", "", _build_workflow_steps(updated), "")

        def handle_run(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            if not vs.get("current_observation") or not vs.get("session_id"):
                updated = _run_reset("aligned", int(vs.get("seed", 42)), "simple", vs)
            else:
                prev_info = vs.get("last_info", {})
                obs_current = _coerce_observation(vs["current_observation"])
                deal_stage = obs_current.deal_stage
                active_blockers = obs_current.active_blockers or []
                real_blockers = [b for b in active_blockers if b != "missing_final_proposal"]
                proposal_submitted = vs.get("proposal_submitted", False)

                if deal_stage == "final_approval" and not real_blockers and not proposal_submitted:
                    action = DealRoomAction(
                        action_type="group_proposal",
                        target="all",
                        target_ids=list(obs_current.stakeholders.keys()),
                        message="I believe we have enough alignment to move to final approval on concrete, reviewable terms.",
                        proposed_terms={
                            "price": 180000,
                            "timeline_weeks": 14,
                            "security_commitments": ["gdpr", "audit rights"],
                            "support_level": "named_support_lead",
                            "liability_cap": "mutual_cap",
                        },
                    )
                    updated_run = dict(vs)
                    updated_run["proposal_submitted"] = True
                else:
                    action = _policy_action(vs["current_observation"])
                    updated_run = dict(vs)

                obs, reward, done, info, state = web_manager.step_session(vs["session_id"], action)
                triggered = _compute_triggered_stakeholders(info, prev_info)
                updated = _record_step(updated_run, action, obs, reward, done, info, state.model_dump())
                updated["triggered_stakeholders"] = triggered
                updated["popup_index"] = 0
                stakeholder_ids = list(updated["current_observation"].get("stakeholders", {}).keys())
                updated["popup_queue"] = [{"stakeholder_id": s} for s in stakeholder_ids]
                updated["show_hint"] = not updated.get("round_complete")
                updated["selected_chip"] = None
                updated["message_text"] = ""
                if updated.get("round_complete"):
                    updated["deal_close_showing"] = True

            lvl_simple, lvl_medium, lvl_hard = _build_level_buttons(updated)
            stakeholder_ids = list(updated["current_observation"].get("stakeholders", {}).keys())
            return (updated, sr,
                    _build_stakeholder_list(updated, stakeholder_ids), _build_detail_panel(updated), _build_message_panel(updated),
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),
                    _build_hint(updated), _build_triggered_popup(updated),
                    _build_score_panel(updated), _build_how_it_works(), _build_deal_close_popup(updated),
                    lvl_simple, lvl_medium, lvl_hard,
                    gr.update(choices=stakeholder_ids, value=updated.get("selected_stakeholder", "")),
                    gr.update(),
                    "", "", _build_workflow_steps(updated), "")

        def handle_autoplay(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            if vs.get("round_complete"):
                vs["auto_playing"] = False
                return (vs, sr) + tuple([gr.update()] * (len(outputs) - 2))
            prev_info = vs.get("last_info", {})
            obs_current = _coerce_observation(vs["current_observation"])
            deal_stage = obs_current.deal_stage
            active_blockers = obs_current.active_blockers or []
            real_blockers = [b for b in active_blockers if b != "missing_final_proposal"]
            proposal_submitted = vs.get("proposal_submitted", False)

            if deal_stage == "final_approval" and not real_blockers and not proposal_submitted:
                action = DealRoomAction(
                    action_type="group_proposal",
                    target="all",
                    target_ids=list(obs_current.stakeholders.keys()),
                    message="I believe we have enough alignment to move to final approval on concrete, reviewable terms.",
                    proposed_terms={
                        "price": 180000,
                        "timeline_weeks": 14,
                        "security_commitments": ["gdpr", "audit rights"],
                        "support_level": "named_support_lead",
                        "liability_cap": "mutual_cap",
                    },
                )
                updated_run = dict(vs)
                updated_run["proposal_submitted"] = True
            else:
                action = _policy_action(vs["current_observation"])
                updated_run = dict(vs)

            obs, reward, done, info, state = web_manager.step_session(vs["session_id"], action)
            triggered = _compute_triggered_stakeholders(info, prev_info)
            updated = _record_step(updated_run, action, obs, reward, done, info, state.model_dump())
            updated["triggered_stakeholders"] = triggered
            stakeholder_ids = list(updated["current_observation"].get("stakeholders", {}).keys())
            updated["popup_queue"] = [{"stakeholder_id": s} for s in stakeholder_ids]
            if updated.get("round_complete"):
                updated["deal_close_showing"] = True
            lvl_simple, lvl_medium, lvl_hard = _build_level_buttons(updated)
            return (updated, sr,
                    _build_stakeholder_list(updated, stakeholder_ids), _build_detail_panel(updated), _build_message_panel(updated),
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),
                    gr.update(choices=stakeholder_ids, value=updated.get("selected_stakeholder", "")),
                    gr.update(),
                    "", "", _build_workflow_steps(vs), "")

        def handle_level_select(level: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            updated = dict(vs)
            updated["level"] = level
            lvl_simple, lvl_medium, lvl_hard = _build_level_buttons(updated)
            stakeholder_ids = list(updated["current_observation"].get("stakeholders", {}).keys())
            return (updated, sr,
                    _build_stakeholder_list(updated, stakeholder_ids), _build_detail_panel(updated), _build_message_panel(updated),
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),
                    _build_hint(updated), _build_triggered_popup(updated),
                    _build_score_panel(updated), _build_how_it_works(), _build_deal_close_popup(updated),
                    lvl_simple, lvl_medium, lvl_hard,
                    gr.update(choices=stakeholder_ids, value=updated.get("selected_stakeholder", "")),
                    gr.update(value=""),
                    "", "", _build_workflow_steps(updated), "")

        def handle_chip_select(chip_text: str, vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            updated = dict(vs)
            updated["selected_chip"] = chip_text
            updated["message_text"] = chip_text
            stakeholder_ids = list((vs.get("current_observation") or {}).get("stakeholders", {}).keys())
            return (updated, sr,
                    _build_stakeholder_list(updated, stakeholder_ids) if stakeholder_ids else "", _build_detail_panel(updated), _build_message_panel(updated),
                    gr.update(), gr.update(), gr.update(),
                    _build_hint(updated), _build_triggered_popup(updated),
                    _build_score_panel(updated), _build_how_it_works(), _build_deal_close_popup(updated),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(choices=stakeholder_ids, value=updated.get("selected_stakeholder", "")),
                    gr.update(value=chip_text),
                    "", "", _build_workflow_steps(updated), "")

        def handle_detail_reselect(vs: Dict[str, Any], sr: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            stakeholder_ids = list((vs.get("current_observation") or {}).get("stakeholders", {}).keys())
            return (vs, sr,
                    _build_stakeholder_list(vs, stakeholder_ids) if stakeholder_ids else "", _build_detail_panel(vs), _build_message_panel(vs),
                    gr.update(), gr.update(), gr.update(),
                    _build_hint(vs), _build_triggered_popup(vs),
                    _build_score_panel(vs), _build_how_it_works(), _build_deal_close_popup(vs),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(choices=stakeholder_ids, value=vs.get("selected_stakeholder", "")),
                    gr.update(value=vs.get("message_text", "")),
                    "", "", _build_workflow_steps(vs), "")

        start_btn.click(fn=handle_start, inputs=[gr.State("simple"), view_state, saved_runs], outputs=outputs)

        stakeholder_dropdown.change(
            fn=handle_stakeholder_click,
            inputs=[stakeholder_dropdown, view_state, saved_runs],
            outputs=outputs
        )

        stakeholder_hidden_input.change(
            fn=handle_stakeholder_click,
            inputs=[stakeholder_hidden_input, view_state, saved_runs],
            outputs=outputs
        )

        simple_btn_evt = gr.HTML().click(fn=handle_level_select, inputs=[gr.State("simple"), view_state, saved_runs], outputs=outputs)
        medium_btn_evt = gr.HTML().click(fn=handle_level_select, inputs=[gr.State("medium"), view_state, saved_runs], outputs=outputs)
        hard_btn_evt = gr.HTML().click(fn=handle_level_select, inputs=[gr.State("hard"), view_state, saved_runs], outputs=outputs)

        send_btn.click(fn=handle_send, inputs=[message_input, view_state, saved_runs], outputs=outputs)
        run_btn.click(fn=handle_run, inputs=[view_state, saved_runs], outputs=outputs)
        step_btn.click(fn=handle_run, inputs=[view_state, saved_runs], outputs=outputs)
        detail_click_handler.click(
            fn=handle_detail_reselect,
            inputs=[view_state, saved_runs],
            outputs=outputs
        )

        stakeholders_card_html.click(
            fn=handle_stakeholder_click,
            inputs=[stakeholders_card_html, view_state, saved_runs],
            outputs=outputs
        )

    return demo
