"""
DealRoom V2.5 Pydantic models.

The environment stays OpenEnv-compatible while exposing a richer dynamic roster,
hidden constraints, and partial observability.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class PricingTable(BaseModel):
    base_price: int = 0
    discount_tiers: List[Dict[str, Any]] = Field(default_factory=list)
    payment_terms: str = "net_30"
    optional_addons: List[str] = Field(default_factory=list)


class SLACommitment(BaseModel):
    uptime_guarantee: float = 0.99
    response_time_hours: int = 4
    resolution_time_hours: int = 24
    support_level: str = "named_support_lead"
    penalties: Optional[Dict[str, Any]] = None


class SubmitProposalAction(BaseModel):
    pricing_table: PricingTable = Field(default_factory=PricingTable)
    sla_commitments: SLACommitment = Field(default_factory=SLACommitment)
    attached_documents: List[str] = Field(default_factory=list)
    message: str = ""
    compliance_attestations: List[str] = Field(default_factory=list)


class RedlineClauseAction(BaseModel):
    clause_id: str = ""
    proposed_text: str = ""
    rationale: str = ""
    impact_summary: Optional[str] = None


class AcknowledgeStageAction(BaseModel):
    acknowledged_stage: str = ""
    confirmation_message: str = ""


class DealRoomAction(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    action_type: str = "direct_message"
    target: str = "all"
    target_ids: List[str] = Field(default_factory=list)
    message: str = ""
    documents: List[Dict[str, str]] = Field(default_factory=list)
    proposed_terms: Optional[Dict[str, Any]] = None
    channel: str = "formal"
    mode: str = "async_email"
    lookahead: Optional["LookaheadRequest"] = None
    submit_proposal: Optional[SubmitProposalAction] = None
    redline_clause: Optional[RedlineClauseAction] = None
    acknowledge_stage: Optional[AcknowledgeStageAction] = None

    @field_validator("message")
    @classmethod
    def truncate_message(cls, value: str) -> str:
        return value[:1200]

    @field_validator("target_ids")
    @classmethod
    def normalize_target_ids(cls, value: List[str]) -> List[str]:
        normalized = [item.strip() for item in value if item and item.strip()]
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def sync_targets(self) -> "DealRoomAction":
        if self.target_ids and self.target == "all":
            self.target = ",".join(self.target_ids)
        return self

    def get_parsed_proposal(self) -> Optional[SubmitProposalAction]:
        return self.submit_proposal

    def get_parsed_redline(self) -> Optional[RedlineClauseAction]:
        return self.redline_clause

    def get_parsed_acknowledge(self) -> Optional[AcknowledgeStageAction]:
        return self.acknowledge_stage


class DealRoomObservation(BaseModel):
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    round_number: int = 0
    max_rounds: int = 10
    stakeholders: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    stakeholder_messages: Dict[str, str] = Field(default_factory=dict)
    engagement_level: Dict[str, float] = Field(default_factory=dict)
    weak_signals: Dict[str, List[str]] = Field(default_factory=dict)
    known_constraints: List[Dict[str, Any]] = Field(default_factory=list)
    requested_artifacts: Dict[str, List[str]] = Field(default_factory=dict)
    approval_path_progress: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    deal_momentum: str = "stalling"
    deal_stage: str = "evaluation"
    competitor_events: List[str] = Field(default_factory=list)
    veto_precursors: Dict[str, str] = Field(default_factory=dict)
    scenario_hint: Optional[str] = None
    active_blockers: List[str] = Field(default_factory=list)
    days_to_deadline: int = 30
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
    engagement_level_delta: Optional[float] = None
    engagement_history: List[Dict[str, float]] = Field(default_factory=list)
    cross_stakeholder_echoes: List[Dict[str, str]] = Field(default_factory=list)
    committee_vote: Optional[Dict[str, str]] = None
    exec_sponsor_activated: bool = False
    silent_period_duration: int = 0


class DealRoomReward(BaseModel):
    value: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class DealRoomState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    round_number: int = 0
    max_rounds: int = 10

    stakeholders: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    stakeholder_private: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    hidden_constraints: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    relationship_edges: List[Dict[str, Any]] = Field(default_factory=list)
    commitment_ledger: List[Dict[str, Any]] = Field(default_factory=list)
    deferred_effects: List[Dict[str, Any]] = Field(default_factory=list)
    offer_state: Dict[str, Any] = Field(default_factory=dict)
    feasibility_state: Dict[str, Any] = Field(default_factory=dict)

    active_blockers: List[str] = Field(default_factory=list)
    deal_stage: str = "evaluation"
    deal_momentum: str = "stalling"
    stage_regressions: int = 0
    rounds_since_last_contact: Dict[str, int] = Field(default_factory=dict)
    approval_caps: Dict[str, float] = Field(default_factory=dict)
    semantic_threshold_jitter: Dict[str, float] = Field(default_factory=dict)
    weak_signal_history: Dict[str, List[str]] = Field(default_factory=dict)
    requested_artifacts: Dict[str, List[str]] = Field(default_factory=dict)
    discovered_constraints: List[str] = Field(default_factory=list)
    milestone_flags: Dict[str, bool] = Field(default_factory=dict)
    external_events: List[str] = Field(default_factory=list)

    validation_failures: int = 0
    malformed_actions: int = 0
    last_action_error: Optional[str] = None

    deal_closed: bool = False
    deal_failed: bool = False
    failure_reason: str = ""
    final_terms: Optional[Dict[str, Any]] = None
    terminal_outcome: str = ""
    veto_stakeholder: Optional[str] = None

    @field_validator("stakeholder_private")
    @classmethod
    def validate_tracks(
        cls, value: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        required = {"trust", "approval", "perceived_fit", "private_resistance"}
        for stakeholder_id, payload in value.items():
            missing = required - set(payload.keys())
            if missing:
                raise ValueError(f"{stakeholder_id} missing tracks: {sorted(missing)}")
        return value

    def __call__(self) -> "DealRoomState":
        return self


class LookaheadRequest(BaseModel):
    action_draft: "DealRoomAction"
    n_hypotheses: int = 2
    depth: int = 2


class SimulationResult(BaseModel):
    predicted_responses: Dict[str, str] = Field(default_factory=dict)
    predicted_belief_deltas: Dict[str, float] = Field(default_factory=dict)
    cvar_impact: Dict[str, float] = Field(default_factory=dict)
    graph_information_gain: float = 0.0
    cost: float = 0.07
