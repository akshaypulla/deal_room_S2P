"""
Pareto efficiency checker for DealRoom v3 - terminal reward determination.

Decision 7 Terminal Reward Table:
- Deal closed: +1.0
- Hard veto (policy breach): -1.0
- Soft veto (CVaR): -0.8
- Stage regression / impasse: -0.75
- Timeout (max rounds): -0.5
"""

from typing import Dict, List, Tuple

from deal_room.environment.constants import TERMINAL_REWARDS, TERMINAL_REWARDS_V2


def check_pareto_optimality(
    all_utilities: Dict[str, float],
    cvar_losses: Dict[str, float],
    thresholds: Dict[str, float],
) -> bool:
    for stakeholder_id, utility in all_utilities.items():
        cvar = cvar_losses.get(stakeholder_id, 0.0)
        threshold = thresholds.get(stakeholder_id, 0.4)
        if cvar > threshold:
            return False

    pareto_frontier = []
    for sid, util in all_utilities.items():
        dominated = False
        for other_sid, other_util in all_utilities.items():
            if other_sid == sid:
                continue
            if other_util >= util:
                cvar_sid = cvar_losses.get(sid, 0.0)
                cvar_other = cvar_losses.get(other_sid, 0.0)
                if cvar_other <= cvar_sid:
                    dominated = True
                    break
        if not dominated:
            pareto_frontier.append(sid)

    return len(pareto_frontier) > 0


def compute_terminal_reward(
    deal_closed: bool,
    veto_triggered: bool,
    veto_stakeholder: str,
    max_rounds_reached: bool,
    stage_regressions: int,
    all_utilities: Dict[str, float],
    cvar_losses: Dict[str, float],
    thresholds: Dict[str, float],
    is_hard_veto: bool = False,
) -> Tuple[float, str]:
    if deal_closed:
        return TERMINAL_REWARDS_V2["deal_closed"], "deal_closed"

    if veto_triggered:
        if is_hard_veto:
            return TERMINAL_REWARDS_V2["hard_veto"], f"hard_veto_by_{veto_stakeholder}"
        return TERMINAL_REWARDS_V2["soft_veto"], f"soft_veto_by_{veto_stakeholder}"

    if max_rounds_reached:
        is_pareto = check_pareto_optimality(all_utilities, cvar_losses, thresholds)
        if is_pareto:
            return 0.0, "max_rounds_pareto"
        return TERMINAL_REWARDS_V2["timeout"], "max_rounds_no_deal"

    if stage_regressions > 0:
        penalty = TERMINAL_REWARDS_V2["stage_regression"] * min(stage_regressions, 3)
        return penalty, f"stage_regression_{stage_regressions}"

    return TERMINAL_REWARDS_V2["stage_regression"], "impasse"


def get_pareto_frontier_stakeholders(
    all_utilities: Dict[str, float], cvar_losses: Dict[str, float]
) -> List[str]:
    frontier = []
    for sid, util in all_utilities.items():
        dominated = False
        for other_sid, other_util in all_utilities.items():
            if other_sid == sid:
                continue
            if other_util >= util:
                cvar_sid = cvar_losses.get(sid, 0.0)
                cvar_other = cvar_losses.get(other_sid, 0.0)
                if cvar_other <= cvar_sid:
                    dominated = True
                    break
        if not dominated:
            frontier.append(sid)
    return frontier
