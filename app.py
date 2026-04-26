"""
DealRoom Gradio App for HuggingFace Spaces - Minimal Version

This is the entry point for the HuggingFace Space. It creates a standalone
Gradio interface using only essential dependencies.
"""

import os
import sys
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import html
from pathlib import Path

STANDARD_STAKEHOLDERS = ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"]


@dataclass
class EnvironmentMetadata:
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    readme_content: Optional[str] = None
    tasks: List[Dict[str, Any]] = field(default_factory=list)


SESSION_COOKIE_NAME = "dealroom_session_id"


@dataclass
class SessionEntry:
    env: Any
    last_access: float


class DealRoomSessionPool:
    """Keeps one environment instance per browser/client session."""

    def __init__(self, max_sessions: int = 128, ttl_seconds: int = 60 * 60 * 6):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    def reset(
        self,
        task_id: str,
        seed: Optional[int],
        session_id: Optional[str] = None,
    ) -> Tuple[str, Any, Any]:
        _base = os.environ.get("DEALROOM_ENV_PATH", ".")
        if _base not in sys.path:
            sys.path.insert(0, _base)

        try:
            from deal_room_S2P.environment.dealroom_v3 import DealRoomV3S2P
        except ImportError:
            try:
                from environment.dealroom_v3 import DealRoomV3S2P
            except ImportError:
                from dealroom_v3 import DealRoomV3S2P

        with self._lock:
            self._prune_locked()
            resolved_session_id = session_id or self._new_session_id()
            entry = self._sessions.get(resolved_session_id)
            if entry is None:
                entry = SessionEntry(env=DealRoomV3S2P(), last_access=time.time())
                self._sessions[resolved_session_id] = entry
            obs = entry.env.reset(seed=seed, task_id=task_id)
            entry.last_access = time.time()
            state = entry.env._state
            if len(self._sessions) > self.max_sessions:
                self._prune_oldest_locked()
            return resolved_session_id, obs, state

    def step(
        self,
        session_id: str,
        action: Any,
    ) -> Tuple[Any, float, bool, dict, Any]:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                raise KeyError(session_id)
            obs, reward, done, info = entry.env.step(action)
            entry.last_access = time.time()
            return obs, reward, done, info, entry.env._state

    def state(self, session_id: str) -> Any:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                raise KeyError(session_id)
            entry.last_access = time.time()
            return entry.env._state

    def get_beliefs(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            entry.last_access = time.time()
            return entry.env._beliefs

    def has_session(self, session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        with self._lock:
            return session_id in self._sessions

    @staticmethod
    def _new_session_id() -> str:
        return uuid.uuid4().hex[:16]

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, entry in self._sessions.items()
            if now - entry.last_access > self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _prune_oldest_locked(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.items(), key=lambda item: item[1].last_access)[0]
        self._sessions.pop(oldest, None)


_role_by_id = {
    "Legal": "legal_compliance",
    "Finance": "finance",
    "TechLead": "technical",
    "Procurement": "procurement",
    "Operations": "operations",
    "ExecSponsor": "executive_sponsor",
}
_role_icons = {
    "finance": "💰",
    "technical": "🛠️",
    "legal_compliance": "⚖️",
    "procurement": "📦",
    "operations": "⚙️",
    "executive_sponsor": "🎯",
}

_clean_css = """
.dealroom-clean { background: #0f1117; font-family: 'Inter','Segoe UI',system-ui,sans-serif; min-height: 100vh; }
.dealroom-clean * { color: #e5e7eb; }
.dr-header { text-align: center; padding: 20px 0 16px; border-bottom: 1px solid #21262d; margin-bottom: 16px; }
.dr-header h1 { font-size: 1.6rem; font-weight: 700; color: #f9fafb; margin: 0 0 4px; }
.dr-header p { color: #8b949e; font-size: 0.88rem; margin: 0; }
.level-card { background: #161b22; border-radius: 10px; border: 1px solid #21262d; padding: 14px; margin-bottom: 12px; }
.level-card h3 { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; margin: 0 0 10px; font-weight: 600; }
.level-btns { display: flex; gap: 8px; }
.level-btn { flex: 1; padding: 10px 4px !important; border-radius: 6px !important; font-size: 0.75rem !important; font-weight: 600 !important; border: 1px solid #30363d !important; }
.level-btn.selected { border-color: #ff6a00 !important; background: rgba(255,106,0,0.15) !important; color: #ff6a00 !important; }
.start-btn { margin-top: 10px; display: block; width: 100%; padding: 12px !important; font-size: 0.9rem !important; font-weight: 700 !important; background: linear-gradient(135deg,#ff6a00,#e05500) !important; border: none !important; border-radius: 8px !important; color: #fff !important; }
.stakeholder-item { display: flex; align-items: center; gap: 10px; padding: 10px 10px; margin-bottom: 6px; border-radius: 6px; background: #0d1117; border: 1px solid #21262d; cursor: pointer; transition: all 0.15s; }
.stakeholder-item.selected { border-color: #ff6a00; background: rgba(255,106,0,0.08); }
.s-icon { font-size: 1.2rem; width: 28px; text-align: center; }
.s-info { flex: 1; min-width: 0; }
.s-name { font-weight: 600; font-size: 0.85rem; color: #e6edf3; }
.s-role { font-size: 0.7rem; color: #8b949e; }
.s-badge { font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 999px; white-space: nowrap; }
.s-badge.green { background: rgba(63,185,80,0.15); color: #3fb950; }
.s-badge.red { background: rgba(248,81,73,0.15); color: #f85149; }
.s-badge.amber { background: rgba(210,153,34,0.15); color: #d29922; }
.card { background: #161b22; border-radius: 10px; border: 1px solid #21262d; padding: 14px; }
.card h3 { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px; margin: 0 0 10px; font-weight: 600; }
.no-start { text-align: center; padding: 32px 16px; color: #6e7681; }
.no-start .big-icon { font-size: 2.5rem; margin-bottom: 10px; }
.hint-box { background: rgba(255,106,0,0.1); border: 1px solid rgba(255,106,0,0.3); border-radius: 8px; padding: 10px 12px; font-size: 0.82rem; color: #ff6a00; line-height: 1.5; }
.hint-box.neutral { border-color: rgba(139,148,158,0.3); color: #8b949e; text-align: center; }
.send-btn { width: 100%; padding: 10px !important; font-size: 0.88rem !important; font-weight: 600 !important; background: linear-gradient(135deg,#ff6a00,#e05500) !important; border: none !important; border-radius: 6px !important; color: #fff !important; }
.response-input { width: 100%; background: #0d1117 !important; border: 1px solid #30363d !important; border-radius: 6px !important; color: #c9d1d9 !important; padding: 10px 12px !important; font-size: 0.85rem !important; }
"""


def _escape(value: Any) -> str:
    return html.escape(str(value))


def load_metadata() -> EnvironmentMetadata:
    readme_path = Path("README.md")
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
    return EnvironmentMetadata(
        name="deal_room_s2p",
        description="A realistic multi-stakeholder enterprise negotiation environment.",
        version="1.0.0",
        author="akshaypulla",
        readme_content=readme,
        tasks=[
            {"id": "aligned", "description": "Low-friction cooperative scenario"},
            {"id": "conflicted", "description": "CTO-CFO tension with coalition dynamics"},
            {"id": "hostile_acquisition", "description": "Post-acquisition with high CVaR risk"},
        ],
    )


def _normalize_view_state(view_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = {
        "task": "aligned", "seed": 42, "level": "simple",
        "session_id": None, "selected_stakeholder": None, "current_observation": None,
        "current_state": None, "trace": [], "status_message": "",
        "round_started": False, "show_hint": False, "hint_text": "",
        "message_text": "", "last_score": 0.0, "score_delta": None,
        "round_complete": False,
    }
    if not isinstance(view_state, dict):
        return base
    merged = dict(base)
    merged.update(view_state)
    for key in ("trace",):
        if not isinstance(merged.get(key), list):
            merged[key] = []
    return merged


def _build_stakeholder_list(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    observation = view_state.get("current_observation") or {}
    stakeholders = observation.get("stakeholders", {})
    if not stakeholders:
        return ("<div class='card'><h3>👥 Stakeholders</h3>"
                "<div class='no-start'><div class='big-icon'>👥</div>"
                "<p>Start a round to meet your stakeholders</p></div></div>")
    items = []
    for sid in stakeholders:
        role_type = _role_by_id.get(sid, "")
        icon = _role_icons.get(role_type, "👤")
        items.append(
            f"<div class='stakeholder-item' data-stakeholder='{_escape(sid)}'>"
            f"<div class='s-icon'>{icon}</div>"
            f"<div class='s-info'><div class='s-name'>{_escape(sid)}</div></div></div>"
        )
    return "<div class='card'><h3>👥 Stakeholders</h3>" + "".join(items) + "</div>"


def _build_hint(view_state: Dict[str, Any]) -> str:
    view_state = _normalize_view_state(view_state)
    round_started = view_state.get("round_started", False)
    if not round_started:
        return ("<div class='hint-box neutral'>👆 Choose a difficulty level and click "
                "<strong>Start Round</strong> to begin negotiating</div>")
    return "<div class='hint-box'>✅ <strong>Round started!</strong> Interact with stakeholders.</div>"


def _build_how_it_works() -> str:
    return (
        "<div class='card'><h4>📖 How to Negotiate</h4>"
        "<ol>"
        "<li><strong>Start a round</strong> — Choose difficulty and click Start Round</li>"
        "<li><strong>View stakeholder concerns</strong> — Click on a stakeholder</li>"
        "<li><strong>Send a response</strong> — Type a message and send it</li>"
        "<li><strong>Goal</strong> — Get all stakeholders aligned to close the deal</li>"
        "</ol></div>"
    )


class WebManager:
    def __init__(self, pool: DealRoomSessionPool, metadata: EnvironmentMetadata):
        self.pool = pool
        self.metadata = metadata

    def reset_session(self, task_id: str, seed: int, session_id: Optional[str] = None):
        return self.pool.reset(task_id=task_id, seed=seed, session_id=session_id)

    def step_session(self, session_id: str, action: Any):
        return self.pool.step(session_id, action)


def _run_reset(task: str, seed: int, level: str, view_state: Optional[Dict[str, Any]], web_mgr: WebManager) -> Dict[str, Any]:
    current = _normalize_view_state(view_state)
    session_id, obs, state = web_mgr.reset_session(
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
        "round_started": True,
        "show_hint": True,
        "hint_text": "Click on a stakeholder to see their message",
        "trace": [],
        "last_score": 0.0,
        "score_delta": None,
        "round_complete": False,
        "message_text": "",
    })
    return updated


def create_dealroom_gradio_app(pool: DealRoomSessionPool) -> gr.Blocks:
    metadata = load_metadata()
    web_mgr = WebManager(pool, metadata)

    demo = gr.Blocks(elem_classes=["dealroom-clean"])
    with demo:
        gr.HTML(f"<style>{_clean_css}</style>")

        view_state = gr.State(_normalize_view_state(None))

        with gr.Column(elem_classes=["dr-header"]):
            gr.HTML("<h1>🎯 DealRoom</h1>"
                    "<p>B2B Negotiation Simulator</p>")

        with gr.Column(elem_classes=["level-card"]):
            gr.HTML("<h3>Difficulty Level</h3>")
            with gr.Row(elem_classes=["level-btns"]):
                simple_btn = gr.Button("🔓 Simple", elem_classes=["level-btn"], variant="secondary")
                medium_btn = gr.Button("⚡ Medium", elem_classes=["level-btn"], variant="secondary")
                hard_btn = gr.Button("🔒 Hard", elem_classes=["level-btn"], variant="secondary")
            start_btn = gr.Button("▶ Start Round", elem_classes=["start-btn"], variant="primary")

        with gr.Row():
            with gr.Column():
                stakeholders_card_html = gr.HTML()
                hint_html = gr.HTML()
                with gr.Column(elem_classes=["card"]):
                    gr.HTML("<h3>💬 Your Response</h3>")
                    message_input = gr.Textbox(
                        placeholder="Type your message here...",
                        lines=2, elem_classes=["response-input"]
                    )
                    send_btn = gr.Button("Send Response", elem_classes=["send-btn"], variant="primary")

            with gr.Column():
                how_it_works_html = gr.HTML()

        outputs = [
            view_state,
            stakeholders_card_html, hint_html,
            message_input, how_it_works_html,
        ]

        def render_initial(vs):
            vs = _normalize_view_state(vs)
            return (vs, _build_stakeholder_list(vs), _build_hint(vs),
                    gr.update(value=""), _build_how_it_works())

        demo.load(fn=render_initial, inputs=[view_state], outputs=outputs)

        def handle_start(level: str, vs: Dict[str, Any]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            task_map = {"simple": "aligned", "medium": "conflicted", "hard": "hostile_acquisition"}
            task = task_map.get(level, "aligned")
            updated = _run_reset(task, int(vs.get("seed", 42)), level, vs, web_mgr)
            return (updated, _build_stakeholder_list(updated), _build_hint(updated),
                    gr.update(value=""), _build_how_it_works())

        def handle_send(msg: str, vs: Dict[str, Any]) -> Tuple[Any, ...]:
            vs = _normalize_view_state(vs)
            if not vs.get("current_observation") or not vs.get("session_id"):
                updated = _run_reset("aligned", int(vs.get("seed", 42)), "simple", vs, web_mgr)
                return (updated, _build_stakeholder_list(updated), _build_hint(updated),
                        gr.update(value=""), _build_how_it_works())
            return (vs, _build_stakeholder_list(vs), _build_hint(vs),
                    gr.update(value=""), _build_how_it_works())

        simple_btn.click(fn=handle_start, inputs=[gr.State("simple"), view_state], outputs=outputs)
        medium_btn.click(fn=handle_start, inputs=[gr.State("medium"), view_state], outputs=outputs)
        hard_btn.click(fn=handle_start, inputs=[gr.State("hard"), view_state], outputs=outputs)
        start_btn.click(fn=handle_start, inputs=[gr.State("simple"), view_state], outputs=outputs)
        send_btn.click(fn=handle_send, inputs=[message_input, view_state], outputs=outputs)

    return demo


_pool = DealRoomSessionPool()
_app = create_dealroom_gradio_app(_pool)

if __name__ == "__main__":
    _app.launch()