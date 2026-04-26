"""
Curriculum module for DealRoom v3 - adaptive curriculum generator.

Decision 8: Adaptive progression with weighted utility gates.
- Agent must maintain S >= θ_comp over rolling window of M=10 turns per stage
- S = Σ(a_i·u_i)/Σ(a_i) — same weighted committee utility used for stage advancement
- θ_comp set slightly below θ_stall
- Supplier NPCs introduced at stage 4
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from deal_room.environment.constants import STAGE_GATE_THETA_COMP, STAGE_GATE_WINDOW_M


@dataclass
class FailureAnalysis:
    failure_modes: Dict[str, float] = field(default_factory=dict)
    worst_graph_configs: List = field(default_factory=list)
    worst_cvar_configs: List = field(default_factory=list)
    agent_capability_estimate: float = 0.0
    weighted_utility_history: List[float] = field(default_factory=list)
    stage_progression: Dict[str, int] = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    analysis_batch_size: int = 10
    easy_ratio: float = 0.20
    frontier_ratio: float = 0.60
    hard_ratio: float = 0.20
    max_graph_variation: float = 0.3
    stage_gate_window_m: int = STAGE_GATE_WINDOW_M
    stage_gate_theta_comp: float = STAGE_GATE_THETA_COMP
    supplier_npc_stage: int = 4


FAILURE_MODE_DESCRIPTIONS = {
    "F1": "CVaR veto despite positive expected outcome",
    "F2": "Trust collapse mid-episode",
    "F3": "Failed graph inference",
    "F4": "Timeout without coalition formation",
    "F5": "Single-dimension reward hacking",
    "F6": "Authority shift blindness",
    "F7": "Stage gate regression — insufficient weighted utility",
}


class AdaptiveCurriculumGenerator:
    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()
        self._scenario_pool: List[Dict] = []
        self._difficulty_distribution = [
            self.config.easy_ratio,
            self.config.frontier_ratio,
            self.config.hard_ratio,
        ]
        self._rng = np.random.default_rng(42)
        self._initialize_scenario_pool()
        self._current_stage = "evaluation"
        self._turns_in_stage = 0
        self._weighted_utilities_buffer: List[float] = []
        self._agent_capability = 0.3
        self._episodes_seen = 0
        self._veto_streak = 0
        self._consecutive_successes = 0

    def _initialize_scenario_pool(self):
        base_configs = [
            {"task_id": "aligned", "difficulty": "easy"},
            {"task_id": "conflicted", "difficulty": "frontier"},
            {"task_id": "hostile_acquisition", "difficulty": "hard"},
        ]
        for _ in range(5):
            for config in base_configs:
                variant = dict(config)
                variant["seed"] = int(self._rng.integers(0, 2**31))
                self._scenario_pool.append(variant)

    def check_stage_gate(self, weighted_utility: float) -> Tuple[bool, str]:
        self._weighted_utilities_buffer.append(weighted_utility)
        if len(self._weighted_utilities_buffer) > self.config.stage_gate_window_m:
            self._weighted_utilities_buffer.pop(0)

        window_avg = (
            sum(self._weighted_utilities_buffer) / len(self._weighted_utilities_buffer)
            if self._weighted_utilities_buffer
            else 0.0
        )

        if window_avg >= self.config.stage_gate_theta_comp:
            return True, "advance"
        return False, "stall"

    def should_enable_supplier_npcs(self, current_stage: str) -> bool:
        stage_order = ["evaluation", "negotiation", "legal_review", "final_approval"]
        try:
            stage_idx = stage_order.index(current_stage)
            return stage_idx >= self.config.supplier_npc_stage - 1
        except ValueError:
            return False

    def analyze_failures(self, trajectories: List) -> FailureAnalysis:
        failure_counts: Dict[str, int] = {}
        weighted_utility_history: List[float] = []
        stage_progression: Dict[str, int] = {"evaluation": 0, "negotiation": 0, "legal_review": 0}
        veto_count = 0
        success_count = 0

        for traj in trajectories:
            detected = self._detect_failures(traj)
            for failure_id in detected:
                failure_counts[failure_id] = failure_counts.get(failure_id, 0) + 1

            if hasattr(traj, "weighted_utilities"):
                weighted_utility_history.extend(traj.weighted_utilities)

            if hasattr(traj, "stages_visited"):
                for stage in traj.stages_visited:
                    if stage in stage_progression:
                        stage_progression[stage] += 1

            terminal = getattr(traj, "terminal_outcome", "")
            if "veto" in terminal or "hard_veto" in terminal:
                veto_count += 1
            elif "deal_closed" in terminal or terminal == "":
                success_count += 1

        total = len(trajectories)
        failure_modes = (
            {k: v / total for k, v in failure_counts.items()} if total > 0 else {}
        )

        self._episodes_seen += total
        self._veto_streak = veto_count if veto_count == total else max(0, self._veto_streak - 1)
        self._consecutive_successes = success_count if success_count == total else max(0, self._consecutive_successes - 1)

        recent_rewards = []
        for t in trajectories[-5:]:
            if hasattr(t, "rewards"):
                step_rewards = t.rewards[-1] if t.rewards else [0.0]
                weighted = sum(
                    r * w for r, w in zip(step_rewards, [0.25, 0.20, 0.20, 0.20, 0.15])
                )
                recent_rewards.append(weighted)
        capability = float(np.mean(recent_rewards)) if recent_rewards else 0.5
        self._agent_capability = 0.9 * self._agent_capability + 0.1 * capability if self._episodes_seen > 5 else capability

        return FailureAnalysis(
            failure_modes=failure_modes,
            worst_graph_configs=[],
            worst_cvar_configs=[],
            agent_capability_estimate=self._agent_capability,
            weighted_utility_history=weighted_utility_history[-50:],
            stage_progression=stage_progression,
        )

    def _detect_failures(self, traj) -> Dict[str, int]:
        failures: Dict[str, int] = {}

        if hasattr(traj, "terminal_outcome"):
            if "veto" in traj.terminal_outcome:
                failures["F1"] = 1

        if hasattr(traj, "rewards") and len(traj.rewards) >= 7:
            trust_rewards = [r[1] if len(r) > 1 else 0.0 for r in traj.rewards[6:10]]
            if len(trust_rewards) >= 2:
                max_drop = max(trust_rewards) - min(trust_rewards)
                if max_drop > 0.20:
                    failures["F2"] = 1

        if hasattr(traj, "rewards"):
            causal_rewards = [r[4] if len(r) > 4 else 0.0 for r in traj.rewards]
            if all(0.15 <= c <= 0.30 for c in causal_rewards):
                failures["F3"] = 1

        if hasattr(traj, "terminal_outcome") and "stage_regression" in traj.terminal_outcome:
            failures["F7"] = 1

        return failures

    def select_next_scenario(
        self, failure_analysis: Optional[FailureAnalysis] = None
    ) -> Dict:
        if failure_analysis is None or failure_analysis.agent_capability_estimate < 0.3:
            return self._rng.choice(self._scenario_pool)

        capability = failure_analysis.agent_capability_estimate

        if capability < 0.5:
            difficulty = "easy"
        elif capability < 0.75:
            difficulty = "frontier"
        else:
            difficulty = "hard"

        candidates = [s for s in self._scenario_pool if s["difficulty"] == difficulty]
        if not candidates:
            candidates = self._scenario_pool

        selected = self._rng.choice(candidates)
        selected = dict(selected)

        if failure_analysis.stage_progression:
            max_stage_reached = max(
                failure_analysis.stage_progression.keys(),
                key=lambda k: failure_analysis.stage_progression[k],
                default="evaluation"
            )
            if max_stage_reached in ["legal_review", "final_approval"]:
                selected["enable_supplier_npcs"] = True

        return selected

    def generate_adaptive_scenario(
        self, failure_analysis: Optional[FailureAnalysis] = None
    ) -> Dict:
        if self._episodes_seen < 10:
            scenario = self._rng.choice([s for s in self._scenario_pool if s["difficulty"] == "easy"])
            scenario = dict(scenario)
            scenario["seed"] = int(self._rng.integers(0, 2**31))
            return scenario

        scenario = self.select_next_scenario(failure_analysis)

        if self._veto_streak >= 3:
            scenario["difficulty"] = "easy"
            scenario["reduce_cvar_tension"] = True

        if failure_analysis and failure_analysis.failure_modes:
            for failure_id, freq in failure_analysis.failure_modes.items():
                if failure_id == "F1" and freq > 0.3:
                    scenario["difficulty"] = "easy"
                    scenario["reduce_cvar_tension"] = True

        if failure_analysis and failure_analysis.weighted_utility_history:
            recent_avg = (
                sum(failure_analysis.weighted_utility_history[-10:])
                / len(failure_analysis.weighted_utility_history[-10:])
            )
            if recent_avg < STAGE_GATE_THETA_COMP:
                scenario["difficulty"] = "easy"
                scenario["focus_stage"] = "evaluation"

        if self._consecutive_successes >= 5 and self._agent_capability > 0.6:
            self._consecutive_successes = 0

        return scenario


def create_curriculum_generator(
    config: Optional[CurriculumConfig] = None,
) -> AdaptiveCurriculumGenerator:
    return AdaptiveCurriculumGenerator(config=config)
