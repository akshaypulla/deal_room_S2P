"""
Minimal correct GRPO reward function for DealRoom S2P V3.

Key invariants:
  1. Same base seed for ALL completions within a batch → meaningful advantage
  2. Slight jitter per batch (seed + 0-99) → prevents overfitting to specific seeds
  3. Hard -1.0 penalty on parse failure → parser learns correct format (raised later)
  4. No silent fallback → model learns format, not "safe action" shortcuts
  5. 2-step rollout (0.3 discount) → credit stays anchored to model's action
"""

import random
from deal_room_S2P.environment.dealroom_v3 import DealRoomV3S2P
from deal_room_S2P.environment.prompts import parse_action_text

PARSE_PENALTY = -0.5       # penalty for invalid output format
DIVERSITY_BONUS = 0.05     # reward for using a new action type in this batch
DIVERSITY_PENALTY = -0.03  # penalty for repeating the same action type
MAX_ROLLOUT_STEPS = 2      # 2 steps: model action + 1 heuristic (not 3 — keeps credit clean)
ROLLOUT_WEIGHT = 0.3       # 0.3 discount makes reward reflect model's action more than environment

# Seed pool: different seed per batch (NOT per completion)
SEED_POOL = list(range(42, 202))   # 160 seeds for batch diversity
TASK_POOL = (
    ['aligned'] * 15 +
    ['conflicted'] * 30 +
    ['hostile_acquisition'] * 30
)

# Diverse heuristic follow-ups for multi-step rollout
HEURISTIC_FOLLOWUPS = [
    'send_document Legal dpa | Our DPA covers all GDPR Article 28 obligations.',
    'send_document Finance roi_model | 3-year NPV shows $2.1M return on investment.',
    'send_document TechLead security_cert | ISO 27001 certification attached.',
    'direct_message Procurement | We accept standard vendor qualification terms.',
    'concession Finance | price=175000 We can reduce to $175k annually.',
    'direct_message Operations | Our implementation plan minimises disruption.',
    'group_proposal | We are ready to proceed with finalised terms for all stakeholders.',
    'send_document Legal security_cert | Our ISO 27001 scope covers customer-facing systems.',
    'direct_message Legal | Acknowledging your concerns about liability exposure.',
    'send_document Legal dpa | DPA with enhanced liability caps and review-ready clauses.',
]


class MinimalDealRoomReward:
    __name__ = 'MinimalDealRoomReward'

    def __init__(self, n_rollout_steps: int = MAX_ROLLOUT_STEPS):
        self._batch_ctr = 0
        self.n_rollout_steps = n_rollout_steps
        self._parse_fails = 0
        self._last_fail_text = None
        self._batch_action_types = []  # track action types per batch for diversity

    def __call__(self, prompts, completions, **kwargs):
        # ONE base seed + ONE task for ALL completions in this batch
        idx = self._batch_ctr % len(SEED_POOL)
        base_seed = SEED_POOL[idx]
        # slight jitter for generalization (same base per batch, not per completion)
        seed = base_seed + random.randint(0, 99)
        task = TASK_POOL[idx]
        self._batch_ctr += 1  # ← advances ONCE per batch, never inside loop

        # Track action types this batch for diversity scoring
        self._batch_action_types = []

        rewards = []
        for completion in completions:
            r = self._score(completion.strip(), seed, task)
            rewards.append(r)

        return rewards

    def _diversity_adjust(self, action):
        """Apply diversity bonus/penalty based on action type usage in this batch."""
        at = action.action_type if action else None
        if at is None:
            return 0.0

        if at not in self._batch_action_types:
            self._batch_action_types.append(at)
            return DIVERSITY_BONUS  # reward new action type
        else:
            return DIVERSITY_PENALTY  # penalty for repeating

    def _score(self, completion: str, seed: int, task: str) -> float:
        env = DealRoomV3S2P()
        try:
            env.reset(seed=seed, task_id=task)
        finally:
            env.close()  # always clean up

        # ── Parse ─────────────────────────────────────────────────────────────
        action = self._parse_action(completion)
        if action is None:
            self._parse_fails += 1
            self._last_fail_text = completion[:100]
            return PARSE_PENALTY  # hard penalty — model MUST learn format

        # ── Diversity adjustment ─────────────────────────────────────────────
        cumulative = self._diversity_adjust(action)
        weight = 1.0
        rng = random.Random(seed + 99999)

        for step_i in range(self.n_rollout_steps):
            act_text = completion if step_i == 0 else rng.choice(HEURISTIC_FOLLOWUPS)

            act = self._parse_action(act_text)
            if act is None:
                act = self._parse_action(rng.choice(HEURISTIC_FOLLOWUPS))

            _, step_r, done, _ = env.step(act)
            cumulative += weight * step_r
            weight *= ROLLOUT_WEIGHT

            if done:
                cumulative += weight * self._terminal_bonus(env)
                break

        return float(cumulative)

    def _parse_action(self, text: str):
        """Returns DealRoomAction or None on parse failure (no silent fallback)."""
        text = self._clean_output(text)
        try:
            return parse_action_text(text)
        except Exception:
            return None  # ← NO fallback, NO safe default

    def _clean_output(self, text: str) -> str:
        """Strip EOS tokens and find the first valid action line."""
        text = text.replace("</s>", "").replace("EOS", "").replace("<|endoftext|>", "").replace("###", "")
        lines = text.strip().split("\n")
        valid_starts = ("send_document", "direct_message", "concession",
                        "group_proposal", "exec_escalation", "submit_proposal")
        for line in lines:
            line = line.strip()
            if line.startswith(valid_starts):
                return line
        return lines[0].strip() if lines else ""

    def _terminal_bonus(self, env) -> float:
        outcome = env._state.terminal_outcome if env._state else ''
        if 'deal_closed' in outcome:
            return 1.0
        if 'hard_veto' in outcome:
            return -1.0
        if 'soft_veto' in outcome:
            return -0.8
        if 'stage_regression' in outcome:
            return -0.75
        return -0.5  # timeout

    def debug_summary(self):
        return {
            'parse_fails': self._parse_fails,
            'last_fail_text': self._last_fail_text,
        }


reward_fn = MinimalDealRoomReward(n_rollout_steps=MAX_ROLLOUT_STEPS)