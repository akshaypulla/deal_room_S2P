"""
Committee deliberation engine for DealRoom v3 - belief propagation with optional MiniMax summary.

Decision 5 Architecture:
- Four sub-agents: BusinessOwner, Legal, FinanceLead, ExecutiveSponsor (latent node)
- Asynchronous deliberation with staggered responses
- ExecutiveSponsor is dormant by default, activated only via escalation, veto cast, or stage gate
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .causal_graph import BeliefDistribution, propagate_beliefs
from deal_room_S2P.environment.llm_client import generate_deliberation_summary


DELIBERATION_STEPS = {
    "aligned": 3,
    "conflicted": 3,
    "hostile_acquisition": 4,
}

COMMITTEE_SUB_AGENTS = ["BusinessOwner", "Legal", "FinanceLead"]
EXEC_SPONSOR_LATENT = "ExecutiveSponsor"
STAGE_GATE_TRIGGERS = {"escalation", "veto_cast", "stage_advance"}
SILENT_PERIOD_BASE = {"aligned": 2, "conflicted": 3, "hostile_acquisition": 4}


EXEC_SPONSOR_LATENT = "ExecutiveSponsor"
STAGE_GATE_TRIGGERS = {"escalation", "veto_cast", "stage_advance"}
SILENT_PERIOD_BASE = {"aligned": 2, "conflicted": 3, "hostile_acquisition": 4}


@dataclass
class StaggeredResponse:
    stakeholder_id: str
    response_delay: int
    belief_delta: float
    influence_weight: float
    vote: Optional[str] = None
    veto_cast: bool = False


@dataclass
class CommitteeMember:
    stakeholder_id: str
    influence_weights: Dict[str, float] = field(default_factory=dict)
    veto_power: bool = False
    is_active: bool = True
    activation_trigger: Optional[str] = None
    pending_response: Optional[StaggeredResponse] = None


@dataclass
class DeliberationResult:
    updated_beliefs: Dict[str, BeliefDistribution]
    summary_dialogue: Optional[str]
    propagation_deltas: Dict[str, float]
    committee_vote: Optional[Dict[str, str]] = None
    exec_sponsor_activated: bool = False
    silent_period_duration: int = 0


def _minimax_call(
    prompt: str, max_tokens: int = 220, temperature: float = 0.8, timeout: float = 5.0
) -> str:
    return generate_deliberation_summary(
        prompt=prompt, context="deliberation", timeout=timeout
    )


class CommitteeDeliberationEngine:
    def __init__(
        self,
        graph,
        n_deliberation_steps: int = 3,
        issue_influence_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.graph = graph
        self.n_steps = n_deliberation_steps
        self.issue_influence_weights = issue_influence_weights or {}
        self._exec_sponsor_active = False
        self._exec_sponsor_trigger: Optional[str] = None

    def run(
        self,
        vendor_action,
        beliefs_before_action: Dict[str, BeliefDistribution],
        beliefs_after_vendor_action: Dict[str, BeliefDistribution],
        render_summary: bool = True,
    ) -> DeliberationResult:
        self._exec_sponsor_active = False
        self._exec_sponsor_trigger = None

        updated_beliefs = propagate_beliefs(
            graph=self.graph,
            beliefs_before_action=beliefs_before_action,
            beliefs_after_action=beliefs_after_vendor_action,
            n_steps=self.n_steps,
        )

        committee_vote = self._compute_committee_vote(
            beliefs_before=beliefs_before_action,
            beliefs_after=updated_beliefs,
            vendor_action=vendor_action,
        )

        self._check_exec_sponsor_activation(
            committee_vote=committee_vote,
            vendor_action=vendor_action,
            beliefs_after=updated_beliefs,
        )

        silent_duration = self._compute_reactive_silent_period(
            beliefs_before=beliefs_before_action,
            beliefs_after=updated_beliefs,
            committee_vote=committee_vote,
        )

        summary = None
        if render_summary and vendor_action.target_ids:
            summary = self._generate_summary(
                beliefs_before=beliefs_before_action,
                beliefs_after=updated_beliefs,
                targeted_stakeholder=vendor_action.target_ids[0],
            )

        return DeliberationResult(
            updated_beliefs=updated_beliefs,
            summary_dialogue=summary,
            propagation_deltas={
                sid: (
                    updated_beliefs[sid].positive_mass()
                    - beliefs_before_action[sid].positive_mass()
                )
                for sid in updated_beliefs
            },
            committee_vote=committee_vote,
            exec_sponsor_activated=self._exec_sponsor_active,
            silent_period_duration=silent_duration,
        )

    def _compute_committee_vote(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
        vendor_action,
    ) -> Dict[str, str]:
        vote = {}
        action_issue = getattr(vendor_action, "action_type", "general")

        for sid in COMMITTEE_SUB_AGENTS:
            b_before = beliefs_before.get(sid)
            b_after = beliefs_after.get(sid)
            if b_before is None or b_after is None:
                continue

            issue_weights = self.issue_influence_weights.get(action_issue, {})
            influence = issue_weights.get(sid, self.graph.get_weight(sid, sid) if self.graph else 0.5)

            pm_delta = b_after.positive_mass() - b_before.positive_mass()
            auth_weight = self.graph.authority_weights.get(sid, 0.5) if self.graph else 0.5

            weighted_signal = pm_delta * influence * auth_weight

            if weighted_signal >= 0.15:
                vote[sid] = "approve"
            elif weighted_signal <= -0.15:
                vote[sid] = "block"
            else:
                vote[sid] = "abstain"

        return vote

    def _check_exec_sponsor_activation(
        self,
        committee_vote: Dict[str, str],
        vendor_action,
        beliefs_after: Dict[str, BeliefDistribution],
    ) -> None:
        blocks = [sid for sid, v in committee_vote.items() if v == "block"]
        if len(blocks) >= 2:
            self._exec_sponsor_active = True
            self._exec_sponsor_trigger = "veto_cast"
            return

        if getattr(vendor_action, "action_type", "") == "exec_escalation":
            self._exec_sponsor_active = True
            self._exec_sponsor_trigger = "escalation"
            return

        negative_votes = sum(1 for v in committee_vote.values() if v == "block")
        total_votes = len([v for v in committee_vote.values() if v != "abstain"])
        if total_votes > 0 and negative_votes / total_votes > 0.5:
            self._exec_sponsor_active = True
            self._exec_sponsor_trigger = "stage_advance"

    def _compute_reactive_silent_period(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
        committee_vote: Dict[str, str],
    ) -> int:
        task_type = getattr(self.graph, "scenario_type", "aligned")
        base_silent = SILENT_PERIOD_BASE.get(task_type, 2)

        conflict_count = sum(1 for v in committee_vote.values() if v == "block")
        total_votes = len([v for v in committee_vote.values() if v != "abstain"])

        if total_votes == 0:
            return base_silent

        conflict_ratio = conflict_count / total_votes if total_votes > 0 else 0

        belief_deltas = []
        for sid in beliefs_after:
            b_before = beliefs_before.get(sid)
            b_after = beliefs_after.get(sid)
            if b_before and b_after:
                delta = abs(b_after.positive_mass() - b_before.positive_mass())
                belief_deltas.append(delta)

        avg_delta = sum(belief_deltas) / len(belief_deltas) if belief_deltas else 0.0

        conflict_penalty = int(conflict_ratio * 3)
        delta_penalty = int(avg_delta * 4) if avg_delta < 0.05 else 0

        silent_duration = base_silent + conflict_penalty + delta_penalty

        max_silent = {"aligned": 6, "conflicted": 8, "hostile_acquisition": 10}.get(task_type, 6)
        return min(silent_duration, max_silent)

    def _generate_summary(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
        targeted_stakeholder: str,
    ) -> str:
        target_belief_before = beliefs_before.get(targeted_stakeholder)
        target_belief_after = beliefs_after.get(targeted_stakeholder)
        if target_belief_before is None or target_belief_after is None:
            return ""

        pm_before = target_belief_before.positive_mass()
        pm_after = target_belief_after.positive_mass()
        pm_delta = pm_after - pm_before

        confidence_before = getattr(target_belief_before, "confidence", 0.5)
        confidence_after = getattr(target_belief_after, "confidence", 0.5)
        conf_delta = confidence_after - confidence_before

        other_deltas = {}
        for sid, b_after in beliefs_after.items():
            if sid == targeted_stakeholder:
                continue
            b_before = beliefs_before.get(sid)
            if b_before is not None:
                delta = b_after.positive_mass() - b_before.positive_mass()
                if abs(delta) > 0.01:
                    other_deltas[sid] = delta

        prompt = (
            f"Deliberation summary for deal room committee discussion:\n\n"
            f"Targeted stakeholder: {targeted_stakeholder}\n"
            f"Belief shift for targeted: positive mass {pm_before:.2f} -> {pm_after:.2f} (delta {pm_delta:+.2f})\n"
            f"Confidence shift: {confidence_before:.2f} -> {confidence_after:.2f} (delta {conf_delta:+.2f})\n"
        )
        if other_deltas:
            prompt += "Other stakeholder deltas:\n"
            for sid, delta in sorted(other_deltas.items()):
                prompt += f"  {sid}: {delta:+.2f}\n"
        prompt += (
            f"\nSummarize in 2-4 sentences how the committee's understanding evolved. "
            f"Focus on the targeted stakeholder's changed perception and downstream effects."
        )
        try:
            return _minimax_call(prompt, max_tokens=220, temperature=0.8, timeout=5.0)
        except Exception:
            return ""
