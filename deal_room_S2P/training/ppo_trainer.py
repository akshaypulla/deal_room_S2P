"""
Minimal PPO implementation for DealRoom v3.

Implements:
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function critic
- Per-step advantage computation for proper credit assignment
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from deal_room_S2P.curriculum.adaptive_generator import AdaptiveCurriculumGenerator
from deal_room_S2P.environment.constants import REWARD_WEIGHTS, STEP_PENALTY, TERMINAL_REWARDS_V2
from deal_room_S2P.environment.dealroom_v3 import DealRoomV3S2P
from models import DealRoomAction, DealRoomObservation, LookaheadRequest


REWARD_DIMENSIONS = ("goal", "trust", "info", "risk", "causal")


def weighted_reward(reward_vector: Sequence[float]) -> float:
    return float(
        sum(
            REWARD_WEIGHTS[dimension] * value
            for dimension, value in zip(REWARD_DIMENSIONS, reward_vector)
        )
    )


@dataclass
class PPOTrainingMetrics:
    goal_reward: float = 0.0
    trust_reward: float = 0.0
    info_reward: float = 0.0
    risk_reward: float = 0.0
    causal_reward: float = 0.0
    total_reward: float = 0.0
    weighted_reward: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    episodes_completed: int = 0
    terminal_outcomes: Dict[str, int] = field(default_factory=dict)
    lookahead_usage_rate: float = 0.0
    prediction_accuracy: float = 0.0


@dataclass
class Trajectory:
    task_id: str
    terminal_outcome: str = ""
    observations: List[DealRoomObservation] = field(default_factory=list)
    actions: List[DealRoomAction] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    reward_vectors: List[List[float]] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    lookahead_used: List[bool] = field(default_factory=list)
    prediction_accuracies: List[float] = field(default_factory=list)
    seed: Optional[int] = None


class SimpleValueCritic:
    def __init__(self, observation_dim: int = 128, hidden_dim: int = 64):
        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.optimizer = AdamW(self.net.parameters(), lr=3e-4)

    def forward(self, obs_features: torch.Tensor) -> torch.Tensor:
        return self.net(obs_features).squeeze(-1)

    def compute_value_loss(self, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(values, returns)

    def update(self, values: torch.Tensor, returns: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.compute_value_loss(values, returns)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class SimplePPOTrainer:
    def __init__(
        self,
        policy_fn: Callable[[DealRoomObservation], DealRoomAction],
        log_prob_fn: Callable[[str], float],
        observation_encoder: Callable[[DealRoomObservation], np.ndarray],
        learning_rate: float = 2e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        max_grad_norm: float = 0.5,
        value_epochs: int = 4,
       ppo_epochs: int = 4,
        seed: int = 42,
    ):
        self.policy_fn = policy_fn
        self.log_prob_fn = log_prob_fn
        self.observation_encoder = observation_encoder
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.value_epochs = value_epochs
        self.ppo_epochs = ppo_epochs
        self.rng = np.random.default_rng(seed)

        self.value_critic = SimpleValueCritic()
        self._total_steps = 0

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        advantages = []
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages.insert(0, last_gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def collect_trajectory(
        self,
        env: DealRoomV3S2P,
        max_steps: int = 10,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        use_lookahead_prob: float = 0.0,
    ) -> Trajectory:
        obs = env.reset(seed=seed, task_id=task_id or "aligned")
        traj = Trajectory(task_id=task_id or "aligned", seed=seed or 42)

        for step in range(max_steps):
            traj.observations.append(obs)

            action = self.policy_fn(obs)
            if use_lookahead_prob > 0 and self.rng.random() < use_lookahead_prob:
                action.lookahead = LookaheadRequest(
                    action_draft=action.model_copy(deep=True),
                    n_hypotheses=2,
                    depth=2,
                )
            traj.actions.append(action)
            traj.lookahead_used.append(action.lookahead is not None)

            obs, reward, done, info = env.step(action)
            traj.rewards.append(reward)
            traj.prediction_accuracies.append(float(info.get("prediction_accuracy", 0.0)))

            reward_components = info.get("reward_components") or {}
            reward_vector = [
                float(reward_components.get(dimension, reward))
                for dimension in REWARD_DIMENSIONS
            ]
            traj.reward_vectors.append(reward_vector)

            obs_feat = self.observation_encoder(obs)
            value = float(self.value_critic.forward(
                torch.tensor(obs_feat, dtype=torch.float32)
            ).item())
            traj.values.append(value)

            action_text = f"{action.action_type} {action.target} {action.message[:50]}"
            log_prob = self.log_prob_fn(action_text)
            traj.log_probs.append(log_prob)

            if done:
                break

        traj.terminal_outcome = info.get("terminal_outcome", "max_rounds") if done else ""

        advantages, returns = self.compute_gae(
            traj.rewards,
            traj.values,
            [done] * len(traj.rewards),
        )
        traj.advantages = advantages
        traj.returns = returns

        return traj

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
        approx_kl = (old_log_probs - log_probs).mean().item()

        return policy_loss, approx_kl, clip_fraction

    def update(
        self,
        trajectories: List[Trajectory],
    ) -> PPOTrainingMetrics:
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []

        for traj in trajectories:
            for obs, action, old_log_prob, adv, ret in zip(
                traj.observations,
                traj.actions,
                traj.log_probs,
                traj.advantages,
                traj.returns,
            ):
                obs_feat = self.observation_encoder(obs)
                all_obs.append(obs_feat)
                action_text = f"{action.action_type} {action.target} {action.message[:50]}"
                all_actions.append(action_text)
                all_old_log_probs.append(old_log_prob)
                all_advantages.append(adv)
                all_returns.append(ret)

        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
        old_log_probs_tensor = torch.tensor(all_old_log_probs, dtype=torch.float32)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        value_loss = 0.0
        for _ in range(self.value_epochs):
            values = self.value_critic.forward(obs_tensor)
            value_loss += self.value_critic.update(values, returns_tensor)
        value_loss /= max(self.value_epochs, 1)

        policy_losses = []
        entropy_sum = 0.0
        clip_fractions = []
        approx_kls = []

        for _ in range(self.ppo_epochs):
            for i in range(len(all_actions)):
                action_text = all_actions[i]
                new_log_prob = self.log_prob_fn(action_text)
                action_idx = i % len(all_actions)

                log_probs = old_log_probs_tensor.clone()
                log_probs[action_idx] = new_log_prob

                policy_loss, approx_kl, clip_frac = self.compute_policy_loss(
                    log_probs.unsqueeze(0),
                    old_log_probs_tensor.unsqueeze(0),
                    advantages_tensor.unsqueeze(0),
                )
                policy_losses.append(policy_loss.item())
                clip_fractions.append(clip_frac)
                approx_kls.append(approx_kl)

        total_policy_loss = sum(policy_losses) / max(len(policy_losses), 1)

        per_dimension: Dict[str, List[float]] = {d: [] for d in REWARD_DIMENSIONS}
        total_rewards = []
        weighted_rewards = []
        lookahead_count = 0
        prediction_values = []
        terminal_outcomes: Dict[str, int] = {}

        for traj in trajectories:
            total_rewards.append(sum(traj.rewards))
            weighted_rewards.append(sum(weighted_reward(rv) for rv in traj.reward_vectors))
            terminal_outcomes[traj.terminal_outcome] = terminal_outcomes.get(traj.terminal_outcome, 0) + 1

            for rv in traj.reward_vectors:
                for dim, val in zip(REWARD_DIMENSIONS, rv):
                    per_dimension[dim].append(float(val))

            lookahead_count += sum(traj.lookahead_used)
            prediction_values.extend(traj.prediction_accuracies)

        self._total_steps += sum(len(t.observations) for t in trajectories)

        return PPOTrainingMetrics(
            goal_reward=float(np.mean(per_dimension["goal"])) if per_dimension["goal"] else 0.0,
            trust_reward=float(np.mean(per_dimension["trust"])) if per_dimension["trust"] else 0.0,
            info_reward=float(np.mean(per_dimension["info"])) if per_dimension["info"] else 0.0,
            risk_reward=float(np.mean(per_dimension["risk"])) if per_dimension["risk"] else 0.0,
            causal_reward=float(np.mean(per_dimension["causal"])) if per_dimension["causal"] else 0.0,
            total_reward=float(np.mean(total_rewards)) if total_rewards else 0.0,
            weighted_reward=float(np.mean(weighted_rewards)) if weighted_rewards else 0.0,
            policy_loss=total_policy_loss,
            value_loss=value_loss,
            entropy=entropy_sum / max(len(trajectories), 1),
            approx_kl=sum(approx_kls) / max(len(approx_kls), 1),
            clip_fraction=sum(clip_fractions) / max(len(clip_fractions), 1),
            episodes_completed=len(trajectories),
            terminal_outcomes=terminal_outcomes,
            lookahead_usage_rate=lookahead_count / max(self._total_steps, 1),
            prediction_accuracy=float(np.mean(prediction_values)) if prediction_values else 0.0,
        )

    def compute_training_metrics(
        self,
        trajectories: List[Trajectory],
    ) -> PPOTrainingMetrics:
        per_dimension: Dict[str, List[float]] = {d: [] for d in REWARD_DIMENSIONS}
        total_rewards = []
        weighted_rewards = []
        terminal_outcomes: Dict[str, int] = {}
        lookahead_count = 0
        prediction_values = []

        for traj in trajectories:
            total_rewards.append(sum(traj.rewards))
            weighted_rewards.append(sum(weighted_reward(rv) for rv in traj.reward_vectors))
            terminal_outcomes[traj.terminal_outcome] = terminal_outcomes.get(traj.terminal_outcome, 0) + 1

            for rv in traj.reward_vectors:
                for dim, val in zip(REWARD_DIMENSIONS, rv):
                    per_dimension[dim].append(float(val))

            lookahead_count += sum(traj.lookahead_used)
            prediction_values.extend(traj.prediction_accuracies)

        return PPOTrainingMetrics(
            goal_reward=float(np.mean(per_dimension["goal"])) if per_dimension["goal"] else 0.0,
            trust_reward=float(np.mean(per_dimension["trust"])) if per_dimension["trust"] else 0.0,
            info_reward=float(np.mean(per_dimension["info"])) if per_dimension["info"] else 0.0,
            risk_reward=float(np.mean(per_dimension["risk"])) if per_dimension["risk"] else 0.0,
            causal_reward=float(np.mean(per_dimension["causal"])) if per_dimension["causal"] else 0.0,
            total_reward=float(np.mean(total_rewards)) if total_rewards else 0.0,
            weighted_reward=float(np.mean(weighted_rewards)) if weighted_rewards else 0.0,
            episodes_completed=len(trajectories),
            terminal_outcomes=terminal_outcomes,
            lookahead_usage_rate=lookahead_count / max(sum(len(t.observations) for t in trajectories), 1),
            prediction_accuracy=float(np.mean(prediction_values)) if prediction_values else 0.0,
        )


class HeuristicPolicyAdapter:
    def __init__(self):
        self.enable_lookahead = False
        self.prefer_concessions = False

    def __call__(self, obs: DealRoomObservation) -> DealRoomAction:
        if obs.veto_precursors:
            target = next(iter(obs.veto_precursors))
            if target.lower() == "legal":
                return DealRoomAction(
                    action_type="send_document",
                    target=target,
                    target_ids=[target],
                    message="Here is the DPA with stronger liability safeguards and review-ready clauses.",
                    documents=[{"type": "dpa", "name": "DPA"}],
                    proposed_terms={"liability_cap": 1500000},
                )
            elif target.lower() == "finance":
                return DealRoomAction(
                    action_type="send_document",
                    target=target,
                    target_ids=[target],
                    message="Here is the ROI model with downside cases and approval-friendly assumptions.",
                    documents=[{"type": "roi_model", "name": "ROI Model"}],
                )
            else:
                return DealRoomAction(
                    action_type="direct_message",
                    target=target,
                    target_ids=[target],
                    message="I want to address your tail-risk concerns directly and concretely.",
                )
        else:
            target = self._pick_target(obs)
            return self._build_target_action(target)

    def _pick_target(self, obs: DealRoomObservation) -> str:
        if obs.active_blockers:
            return obs.active_blockers[0]
        if obs.weak_signals:
            for stakeholder_id, signals in obs.weak_signals.items():
                if "declining_engagement" in signals or "low_engagement" in signals:
                    return stakeholder_id
        if obs.engagement_level:
            return min(
                obs.engagement_level,
                key=lambda sid: obs.engagement_level[sid],
            )
        return "Legal"

    def _build_target_action(self, target: str) -> DealRoomAction:
        target_lower = target.lower()
        if target_lower == "legal":
            return DealRoomAction(
                action_type="send_document",
                target=target,
                target_ids=[target],
                message="Here are the compliance and liability safeguards for review.",
                documents=[{"type": "dpa", "name": "DPA"}],
            )
        if target_lower == "finance":
            return DealRoomAction(
                action_type="send_document",
                target=target,
                target_ids=[target],
                message="Here is the ROI model with downside cases and approval-friendly assumptions.",
                documents=[{"type": "roi_model", "name": "ROI Model"}],
            )
        if target_lower in {"techlead", "operations"}:
            return DealRoomAction(
                action_type="send_document",
                target=target,
                target_ids=[target],
                message="Here is the implementation plan with delivery safeguards and named ownership.",
                documents=[{"type": "implementation_timeline", "name": "Implementation Plan"}],
            )
        if self.prefer_concessions:
            return DealRoomAction(
                action_type="concession",
                target=target,
                target_ids=[target],
                message="I can make the commercial and risk posture safer for your team.",
                proposed_terms={"liability_cap": 1500000},
            )
        return DealRoomAction(
            action_type="direct_message",
            target=target,
            target_ids=[target],
            message="Help me understand the remaining constraint so I can de-risk it responsibly.",
        )


def observation_to_features(obs: DealRoomObservation) -> np.ndarray:
    features = []
    features.append(obs.round_number / 10.0)
    features.append(1.0 if obs.deal_momentum == "progressing" else 0.0)
    features.append(1.0 if obs.deal_momentum == "stalling" else 0.0)
    features.append(1.0 if obs.deal_momentum == "fragile" else 0.0)
    features.append(1.0 if obs.deal_momentum == "critical" else 0.0)

    stage_map = {"evaluation": 0.0, "negotiation": 0.25, "legal_review": 0.5, "final_approval": 0.75, "closed": 1.0}
    features.append(stage_map.get(obs.deal_stage, 0.0))

    for sid in ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"]:
        eng = obs.engagement_level.get(sid, 0.5)
        features.append(eng)

    features.append(len(obs.veto_precursors) / 6.0)
    features.append(len(obs.active_blockers) / 5.0)
    features.append(obs.days_to_deadline / 30.0)

    for sid in ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"]:
        signals = obs.weak_signals.get(sid, [])
        features.append(1.0 if "declining_engagement" in signals else 0.0)
        features.append(1.0 if "low_engagement" in signals else 0.0)

    features.append(1.0 if obs.committee_vote else 0.0)
    features.append(1.0 if obs.exec_sponsor_activated else 0.0)
    features.append(obs.silent_period_duration / 5.0)

    while len(features) < 128:
        features.append(0.0)
    return np.array(features[:128], dtype=np.float32)


def action_log_prob(tokenizer, model, text: str, device: str = "cuda") -> float:
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        tokens = inputs["input_ids"]
        token_log_probs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.mean().item())