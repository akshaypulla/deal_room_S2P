# DealRoom S2P V2 Architecture Documentation

## Overview

**Purpose**: DealRoom S2P V2 is an LLM training environment designed to train Large Language Models to negotiate B2B software deals with a committee of stakeholders. The environment simulates enterprise procurement dynamics where an AI vendor must navigate a multi-stakeholder approval process by providing appropriate documentation, making concessions, and building trust across different personas (Legal, Finance, TechLead, Procurement, Operations, ExecSponsor).

**Target User**: LLM training pipelines using GRPO (Group Relative Policy Optimization) or similar RL algorithms.

---

## 1. Environment Core

### 1.1 DealRoomV3 (Main Environment Class)

**What**: The central environment class that manages episodes, processes actions, and returns observations.

**Why**: Serves as the main interface between the LLM agent and the negotiation simulation. It orchestrates all other components (belief tracking, deliberation, reward computation) and maintains the episode state.

**How**:

```python
class DealRoomV3:
    def reset(self, seed: Optional[int] = None, task_id: str = "aligned") -> DealRoomObservation
    def step(self, action: DealRoomAction) -> Tuple[DealRoomObservation, float, bool, Dict[str, Any]]
```

**Key Implementation Details**:

- **Task Scenarios**: Three scenarios with increasing difficulty
  - `aligned`: Cooperative stakeholders, easier negotiations
  - `conflicted`: Mixed interests, moderate tension
  - `hostile_acquisition`: Adversarial setting, high risk of veto

- **Stakeholders** (6 committee members):
  ```python
  STANDARD_STAKEHOLDERS = ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"]
  STANDARD_HIERARCHY = {"Legal": 3, "Finance": 3, "TechLead": 2, "Procurement": 2, "Operations": 2, "ExecSponsor": 5}
  ```

- **Episode Flow**:
  1. `reset()` initializes a new episode with a sampled causal graph, initial beliefs, and noisy engagement levels
  2. `step()` processes vendor actions through belief updates, deliberation, and reward computation
  3. Episode terminates on veto, max rounds, or successful deal closure

- **Deal Stages** (progression gates):
  ```python
  ["evaluation", "negotiation", "legal_review", "final_approval", "closed"]
  ```
  Advancement requires weighted utility threshold (θ_pass=0.65); regression occurs below stall threshold (θ_stall=0.40)

---

### 1.2 DealRoomState (State Model)

**What**: Pydantic model maintaining comprehensive episode state including stakeholder private state, offer terms, constraints, and terminal outcomes.

**Why**: Provides structured state management with validation, ensuring all stakeholder properties are properly tracked throughout the negotiation.

**How**:
```python
class DealRoomState(BaseModel):
    episode_id: str
    step_count: int
    task_id: str
    stakeholders: Dict[str, Dict[str, Any]]  # Public stakeholder info
    stakeholder_private: Dict[str, Dict[str, Any]]  # Hidden trust, approval, resistance
    offer_state: Dict[str, Any]  # Current deal terms
    hidden_constraints: Dict[str, Dict[str, Any]]  # Unknown requirements to discover
    active_blockers: List[str]  # Current veto blockers
    deal_stage: str
    stage_regressions: int
    terminal_outcome: str
```

---

### 1.3 DealRoomObservation (Observation Model)

**What**: The structured observation returned to the LLM agent after each step.

**Why**: Provides partial observability of the negotiation state while hiding internal research-grade information (causal graph, beliefs, CVaR parameters).

**How**:
```python
class DealRoomObservation(BaseModel):
    round_number: int
    max_rounds: int
    stakeholders: Dict[str, Dict[str, Any]]
    stakeholder_messages: Dict[str, str]  # Committee responses
    engagement_level: Dict[str, float]  # Noisy engagement signals
    weak_signals: Dict[str, List[str]]  # Hints about hidden concerns
    veto_precursors: Dict[str, str]  # Warning signals before veto
    deal_momentum: str  # "stalling", "progressing", "critical"
    deal_stage: str
    active_blockers: List[str]
    cross_stakeholder_echoes: List[Dict[str, str]]  # Belief propagation signals
    done: bool
    info: Dict[str, Any]  # Detailed diagnostics
```

**Hidden Fields** (NOT exposed to agent):
- `G`, `causal_graph`, `graph`, `true_beliefs`, `belief_distributions`, `belief_state`, `B_i`, `V_i`
- `tau`, `tau_i`, `risk_thresholds`, `cvar_thresholds`, `edge_weights`, `w_ij`
- `deliberation_transcript`, `deliberation_log`, `internal_dialogue`
- `u_i`, `u_ij`

---

### 1.4 DealRoomAction (Action Model)

**What**: The structured action the agent can take.

**Why**: Provides a rich action space including messages, document sending, concessions, proposals, and lookahead requests.

**How**:
```python
class DealRoomAction(BaseModel):
    action_type: str  # "direct_message", "send_document", "concession", "group_proposal", "exec_escalation"
    target: str  # Target stakeholder name
    target_ids: List[str]  # Normalized target list
    message: str  # Communication content (truncated to 1200 chars)
    documents: List[Dict[str, str]]  # [{"name": "DPA", "content": "..."}]
    proposed_terms: Optional[Dict[str, Any]]  # Term modifications
    lookahead: Optional[LookaheadRequest]  # Forward simulation request
```

**Action Types**:
1. `direct_message`: Send a text message to target stakeholder
2. `send_document`: Attach documents (DPA, security_cert, roi_model, implementation_timeline)
3. `concession`: Offer better terms (price reduction, liability cap increase)
4. `group_proposal`: Present terms to entire committee
5. `exec_escalation`: Request executive intervention
6. `backchannel`: Private communication channel
7. `submit_proposal`: Formal proposal submission with pricing, SLA, attestations
8. `redline_clause`: Propose specific contract clause changes
9. `acknowledge_stage`: Confirm stage advancement

---

## 2. Causal Graph System

### 2.1 CausalGraph (Graph Structure)

**What**: Represents the influence relationships between stakeholders as a directed weighted graph.

**Why**: Models how actions toward one stakeholder affect others through social influence pathways. This is the "hidden state" that agents must learn to infer through observation.

**How**:
```python
@dataclass
class CausalGraph:
    nodes: List[str]  # Stakeholder IDs
    edges: Dict[Tuple[str, str], float]  # (source, dest) -> influence weight
    authority_weights: Dict[str, float]  # Normalized authority per stakeholder
    scenario_type: str
    seed: int
```

**Graph Sampling** (`sample_graph` function):
- Authority nodes (level ≥ 4) always have outgoing edges
- Intra-cluster edges (Finance+Procurement, Legal+Compliance, TechLead+Operations) boosted
- Cross-cluster edges penalized
- Authority edges (ExecSponsor→all) have 65-85% probability based on scenario

**Scenario Parameters**:
```python
SCENARIO_PARAMS = {
    "aligned": {"base_edge_prob": 0.30, "intra_boost": 0.40, "cross_penalty": 0.20, "authority_edge_prob": 0.85},
    "conflicted": {"base_edge_prob": 0.45, "intra_boost": 0.50, "cross_penalty": 0.15, "authority_edge_prob": 0.80},
    "hostile_acquisition": {"base_edge_prob": 0.60, "intra_boost": 0.25, "cross_penalty": 0.05, "authority_edge_prob": 0.65},
}
```

---

### 2.2 BeliefDistribution

**What**: Probability distribution over vendor characterizations (competent, incompetent, trustworthy, deceptive, aligned, misaligned).

**Why**: Tracks each stakeholder's mental model of the vendor, which evolves based on actions and propagates through the causal graph.

**How**:
```python
@dataclass
class BeliefDistribution:
    distribution: Dict[str, float]  # P(competent), P(incompetent), etc.
    stakeholder_role: str
    confidence: float  # 1 - (entropy / log(6))
    history: List[Tuple]  # Record of updates

    def positive_mass(self) -> float:
        return sum(P(competent), P(trustworthy), P(aligned))

    def negative_mass(self) -> float:
        return sum(P(incompetent), P(deceptive), P(misaligned))
```

---

### 2.3 Belief Propagation (`propagate_beliefs`)

**What**: Multi-step iterative process where belief changes diffuse through the causal graph.

**Why**: Models real committee dynamics where one member's opinion influences others.

**How**:
```python
def propagate_beliefs(graph, beliefs_before_action, beliefs_after_action, n_steps=3):
    current_beliefs = {sid: b.copy() for sid, b in beliefs_after_action.items()}

    for step in range(n_steps):
        next_beliefs = {sid: b.copy() for sid, b in current_beliefs.items()}
        for dest_id in graph.nodes:
            influencers = graph.get_influencers(dest_id)
            total_delta = sum(weight * (source.positive_mass() - beliefs_before_action[source].positive_mass())
                              for source_id, weight in influencers.items())
            # Apply scaled delta with damping
            scaled_delta = total_delta * (abs(total_delta) / (abs(total_delta) + 1/10))
            if abs(scaled_delta) > epsilon_floor:
                next_beliefs[dest_id] = _apply_belief_delta(current_beliefs[dest_id], scaled_delta, damping=0.95**step)
        current_beliefs = next_beliefs
    return current_beliefs
```

**Key Properties**:
- Damping prevents runaway amplification (0.95^step)
- Sigmoid-like scaling (k_gate=10.0) prevents infinitesimal changes
- Epsilon floor (0.003) ensures minimum meaningful signal

---

### 2.4 Betweenness Centrality (`get_betweenness_centrality`)

**What**: Measures how central a stakeholder is in the influence network.

**Why**: Determines the "causal" reward dimension - targeting central nodes has broader committee impact.

**How**:
```python
def get_betweenness_centrality(graph: CausalGraph, stakeholder: str) -> float:
    # Count paths (source -> stakeholder -> dest) / total possible paths
    betweenness = 0.0
    for source in graph.nodes:
        for dest in graph.nodes:
            if source != dest != stakeholder and shortest_path_exists(graph, source, stakeholder) and shortest_path_exists(graph, stakeholder, dest):
                betweenness += 1.0
    return betweenness / ((n-1) * (n-2))
```

---

## 3. Stakeholder Models

### 3.1 StakeholderRiskProfile (Archetypes)

**What**: Risk preference parameters for each stakeholder persona.

**Why**: Different stakeholders have different risk tolerances (tau thresholds) and care about different outcome domains (compliance, cost, implementation).

**How**:
```python
@dataclass
class StakeholderRiskProfile:
    stakeholder_id: str
    role: str
    alpha: float  # CVaR quantile (0.70-0.95, higher = more risk-averse)
    tau: float  # Veto threshold (0.10-0.40, lower = easier to veto)
    lambda_risk: float  # Risk weight in utility (0.25-0.70)
    veto_power: bool  # Can unilaterally block deal
    utility_weights: Dict[str, float]  # What they optimize for
    uncertainty_domains: List[str]  # Their concern areas
```

**Archetype Profiles**:
| Stakeholder | Alpha | Tau | Veto Power | Primary Concerns |
|-------------|-------|-----|-------------|------------------|
| Legal | 0.95 | 0.10 | Yes | Compliance, liability, data protection |
| Finance | 0.90 | 0.15 | Yes | ROI, payment terms, cost predictability |
| TechLead | 0.80 | 0.25 | No | Implementation feasibility, integration |
| Procurement | 0.85 | 0.20 | No | Contract compliance, price competitiveness |
| Operations | 0.80 | 0.30 | No | Operational continuity, timeline |
| ExecSponsor | 0.70 | 0.40 | Yes | Strategic alignment, organizational consensus |

---

### 3.2 CVaR Preference Model

**What**: Conditional Value at Risk computation for stakeholder utility under uncertainty.

**Why**: Enables "tail-risk" veto - deals can be blocked even with positive expected utility if bad outcomes are too severe.

**How**:
```python
def compute_cvar(outcomes: np.ndarray, alpha: float) -> float:
    sorted_outcomes = np.sort(outcomes)
    cutoff_index = int(len(sorted_outcomes) * (1 - alpha))
    tail_losses = 1.0 - sorted_outcomes[:cutoff_index + 1]
    cvar = np.mean(tail_losses)  # Average of worst (1-alpha)% outcomes
    return float(cvar)
```

**Outcome Distribution** (`compute_outcome_distribution`):
- Base success probability: 0.70-0.92 depending on deal terms and domain
- Compliance domain: base 0.80, boosted to 0.92 with DPA+security_cert
- Failed outcomes weighted by severity (0.0-0.6 range)

**Veto Trigger**:
```python
def check_veto_trigger(cvar_loss: float, stakeholder_profile: StakeholderRiskProfile) -> bool:
    return cvar_loss > stakeholder_profile.tau
```

**Core Research Claim**: CVaR veto fires even when Expected Utility > 0 (tail-risk awareness, not loss-chasing).

---

### 3.3 Bayesian Belief Updates (`bayesian_update`)

**What**: Updates stakeholder beliefs based on vendor actions using likelihood tables.

**Why**: Provides principled belief updating based on action profiles - sending DPA to Legal has different implications than aggressive escalation.

**How**:
```python
def bayesian_update(belief, action_type, documents, stakeholder_role, is_targeted):
    likelihoods = _get_likelihood(action_type, documents, stakeholder_role)
    damping = 1.0 if is_targeted else 0.7  # Targeted actions have full effect

    new_dist = {}
    for vendor_type, prior_prob in belief.distribution.items():
        likelihood = likelihoods.get(vendor_type, 0.5)
        dampened_likelihood = 1.0 + damping * (likelihood - 1.0)
        new_dist[vendor_type] = prior_prob * dampened_likelihood

    # Normalize and compute confidence
    total = sum(new_dist.values())
    new_dist = {k: max(0.01, v / total) for k, v in new_dist.items()}
    entropy = -sum(p * log(p, 2) for p in new_dist.values())
    confidence = 1.0 - (entropy / LOG2_6)
```

**Likelihood Tables** (key entries):
```python
ACTION_LIKELIHOODS = {
    "send_document(DPA)_proactive": {"competent": 0.85, "trustworthy": 0.80, "aligned": 0.80, ...},
    "send_document(roi_model)_to_finance": {"competent": 0.75, "trustworthy": 0.60, "aligned": 0.70, ...},
    "direct_message_role_specific": {"competent": 0.70, "trustworthy": 0.65, "aligned": 0.70, ...},
    "default": {vendor_type: 0.50 for all types}
}
```

---

## 4. Deliberation Engine

### 4.1 CommitteeDeliberationEngine

**What**: Simulates asynchronous committee deliberation after each vendor action.

**Why**: Models real committee dynamics where members discuss, influence each other, and potentially escalate to executives.

**How**:
```python
class CommitteeDeliberationEngine:
    def run(self, vendor_action, beliefs_before_action, beliefs_after_vendor_action, render_summary):
        # 1. Propagate beliefs through graph
        updated_beliefs = propagate_beliefs(graph, beliefs_before_action, beliefs_after_vendor_action, n_steps=3)

        # 2. Compute committee vote
        committee_vote = self._compute_committee_vote(beliefs_before, updated_beliefs, vendor_action)

        # 3. Check ExecSponsor activation
        self._check_exec_sponsor_activation(committee_vote, vendor_action, updated_beliefs)

        # 4. Compute silent period
        silent_duration = self._compute_reactive_silent_period(beliefs_before, updated_beliefs, committee_vote)

        # 5. Generate LLM summary (optional)
        if render_summary:
            summary = self._generate_summary(beliefs_before, updated_beliefs, targeted_stakeholder)
```

**Committee Vote**:
- `approve`: weighted_signal ≥ 0.15
- `block`: weighted_signal ≤ -0.15
- `abstain`: otherwise

Where weighted_signal = positive_mass_delta × influence_weight × authority_weight

**ExecSponsor Activation Triggers**:
1. ≥2 committee members vote "block"
2. Vendor uses `exec_escalation` action
3. >50% of non-abstainers vote negative

**Silent Period**: Duration based on scenario (aligned: 2, conflicted: 3, hostile: 4) plus conflict penalty and belief delta penalty. During silent period, non-targeted stakeholders don't respond.

---

## 5. Lookahead Simulator

### 5.1 LookaheadSimulator

**What**: Forward-simulation mechanism allowing the agent to preview action outcomes.

**Why**: Enables strategic planning - the agent can test hypotheses about what would happen if it takes an action, at a fixed cost.

**How**:
```python
class LookaheadSimulator:
    def simulate(self, action_draft, current_beliefs, n_hypotheses=2, depth=2):
        # Generate hypotheses: optimistic and pessimistic belief states
        hypotheses = self._generate_hypotheses(target_id, current_beliefs)

        # Simulate each hypothesis
        for hypothesis in hypotheses:
            sim_result = self._simulate_one_hypothesis(action_draft, target_id, hypothesis, depth)
            simulation_results.append(sim_result)

        # Return worst-case (minimax)
        return min(simulation_results, key=lambda x: x.predicted_goal_delta)
```

**Hypothesis Generation**:
```python
def _generate_hypotheses(self, target_stakeholder, current_beliefs):
    optimistic_dist = current.distribution
    optimistic_dist["competent"] += 0.15
    optimistic_dist["trustworthy"] += 0.10
    optimistic_dist["aligned"] += 0.10

    pessimistic_dist = current.distribution
    pessimistic_dist["incompetent"] += 0.15
    pessimistic_dist["deceptive"] += 0.10
    pessimistic_dist["misaligned"] += 0.10
```

**Lookahead Cost**: Exactly 0.07 subtracted from goal reward when lookahead is used.

---

## 6. Reward System

### 6.1 UtteranceScorer

**What**: Computes five-dimensional reward signal based on world-state deltas.

**Why**: Provides rich learning signal across multiple dimensions of negotiation quality, not just terminal outcome.

**How** - Five Dimensions:

```python
class UtteranceScorer:
    def score(self, action, state_before, state_after, true_graph, lookahead_used):
        goal = self._score_goal(beliefs_before, beliefs_after, blockers_before, blockers_after, ...)
        trust = self._score_trust(beliefs_before, beliefs_after, targeted_ids)
        info = self._score_info(beliefs_before, beliefs_after)
        risk = self._score_risk(beliefs_before, beliefs_after, risk_profiles, deal_terms)
        causal = self._score_causal(true_graph, targeted_ids)

        if lookahead_used:
            goal = max(0.0, goal - LOOKAHEAD_COST)  # 0.07 penalty

        return UtteranceScore(goal=goal, trust=trust, info=info, risk=risk, causal=causal)
```

**Dimension Details**:

1. **r^goal**: Measures approval improvement weighted by authority
   - Δapproval × authority_weight averaged across stakeholders
   - +blocker resolution bonus (0.15 per resolved)
   - -blocker creation penalty (0.10 per new)
   - CVaR headroom improvement for risk-averse stakeholders

2. **r^trust**: Measures trust change for targeted stakeholder
   - 0.6 × positive_mass_delta + 0.4 × trustworthy_mass_delta

3. **r^info**: Measures entropy reduction (information gain)
   - H(B_before) - H(B_after) normalized by log2(6)

4. **r^risk**: Measures CVaR improvement for risk-averse stakeholders
   - (CVaR_before - CVaR_after) / CVaR_before

5. **r^causal**: Measures betweenness centrality of targeted stakeholder
   - betweenness(target) / max_possible_betweenness

**Reward Weights**:
```python
REWARD_WEIGHTS = {
    "goal": 0.25,
    "trust": 0.20,
    "info": 0.20,
    "risk": 0.20,
    "causal": 0.15,
}
```

**Final Scalar Reward**: Weighted sum of five dimensions, bounded [0, 1].

---

### 6.2 Terminal Rewards

**What**: Episode-end rewards based on deal outcome.

**Why**: Provides strong terminal signal for deal success/failure.

**How**:
```python
TERMINAL_REWARDS_V2 = {
    "deal_closed": 1.0,
    "hard_veto": -1.0,    # Policy breach (missing required docs at stage gate)
    "soft_veto": -0.8,     # CVaR-based veto
    "stage_regression": -0.75,
    "timeout": -0.5,       # Max rounds without deal
}
```

**Pareto Check**: On timeout, if deal is Pareto-optimal (no stakeholder can be improved without worsening another), reward = 0.0 instead of -0.5.

---

## 7. Curriculum System

### 7.1 AdaptiveCurriculumGenerator

**What**: Adaptive scenario selection based on agent performance history.

**Why**: Optimizes learning by presenting appropriately difficult scenarios - not too easy (no learning) and not too hard (frustration).

**How**:
```python
class AdaptiveCurriculumGenerator:
    def generate_adaptive_scenario(self, failure_analysis):
        # First 10 episodes: always easy
        if self._episodes_seen < 10:
            return easy_scenario

        # Select based on capability estimate
        if capability < 0.5: difficulty = "easy"
        elif capability < 0.75: difficulty = "frontier"
        else: difficulty = "hard"

        # If veto streak ≥3: reduce difficulty
        if self._veto_streak >= 3:
            scenario["difficulty"] = "easy"
            scenario["reduce_cvar_tension"] = True

        # If F1 (CVaR veto) > 30%: reduce difficulty
        if failure_analysis.failure_modes.get("F1", 0) > 0.3:
            scenario["difficulty"] = "easy"
            scenario["reduce_cvar_tension"] = True

        return scenario
```

**Failure Modes Detected**:
- F1: CVaR veto despite positive expected outcome
- F2: Trust collapse mid-episode
- F3: Failed graph inference (causal rewards stuck in narrow range)
- F4: Timeout without coalition formation
- F5: Single-dimension reward hacking
- F6: Authority shift blindness
- F7: Stage gate regression

**Stage Gate Check**:
```python
def check_stage_gate(self, weighted_utility):
    self._weighted_utilities_buffer.append(weighted_utility)
    # Rolling window of M=10
    window_avg = mean(buffer[-10:])
    return window_avg >= THETA_COMP (0.35)  # Slightly below stall threshold
```

---

## 8. Training Infrastructure

### 8.1 GRPOTrainer

**What**: Group Relative Policy Optimization trainer for policy improvement.

**Why**: Provides the training loop infrastructure for improving the LLM policy using the multi-dimensional reward signal.

**How**:
```python
class GRPOTrainer:
    def run_self_play_episode(self, env, policy_adapter, max_steps=10, task_id=None, seed=None):
        # Run episode with current policy
        trajectory = EpisodeTrajectory(task_id=task_id, seed=seed)
        for _ in range(max_steps):
            action = adapter.act(observation, rng)
            observation, reward, done, info = env.step(action)
            trajectory.observations.append(observation)
            trajectory.actions.append(action)
            trajectory.rewards.append(reward_vector)
        return trajectory

    def compute_group_relative_advantage(self, episode_rewards, group_rewards):
        # GRPO advantage: (individual_reward - group_mean) / group_std
        aggregated = [weighted_reward(rewards) for rewards in episode_rewards]
        mean = np.mean(group_aggregated)
        std = np.std(group_aggregated) + 1e-8
        return [(reward - mean) / std for reward in aggregated]
```

**Policy Adapters**:
1. `RandomPolicyAdapter`: Random action selection baseline
2. `HeuristicPolicyAdapter`: Rule-based policy targeting blockers and low engagement
3. `ModelPolicyAdapter`: Wraps arbitrary policy function (e.g., LLM)

**Training Loop**:
```python
def run_training_loop(self, n_episodes=50, episodes_per_batch=4, max_steps=10):
    for batch_index in range(n_episodes):
        batch_trajectories = []
        for _ in range(episodes_per_batch):
            trajectory = self.run_self_play_episode(env, policy_adapter, max_steps)
            batch_trajectories.append(trajectory)

        # Analyze failures for curriculum
        failure_analysis = curriculum_generator.analyze_failures(batch_trajectories)

        # Update policy
        policy_adapter.update_from_batch(batch_trajectories)

        # Compute and log metrics
        metrics = self.compute_training_metrics(batch_trajectories)
```

**Metrics Tracked**:
- Per-dimension rewards (goal, trust, info, risk, causal)
- Lookahead usage rate
- Prediction accuracy
- Task mix distribution
- Terminal outcomes distribution

---

### 8.2 DealRoomTextEnv (TRL-Compatible Wrapper)

**What**: Text-in/text-out wrapper for TRL GRPOTrainer compatibility.

**Why**: Allows integration with HuggingFace TRL library for LLM training.

**How**:
```python
class DealRoomTextEnv:
    def reset(self) -> str:
        # Returns initial situation prompt
        return build_situation_prompt(self.current_obs)

    def step(self, action_text: str) -> Tuple[str, float, bool, Dict]:
        # Parse text action to DealRoomAction
        action = parse_action_text(action_text)
        # Execute in environment
        self.current_obs, reward, done, info = self.env.step(action)
        # Return next prompt
        return build_situation_prompt(self.current_obs), float(reward), bool(done), info
```

**Prompt Building**:
```python
def build_situation_prompt(obs: DealRoomObservation) -> str:
    # Builds human-readable situation summary
    # Includes: round number, deal stage, active blockers, engagement levels,
    # weak signals, stakeholder messages, available actions
```

**Action Parsing**:
```python
# Supported patterns:
"send_document Finance roi_model Here is our ROI model."
"direct_message Legal I want to address your concerns.###"
"concession Finance liability_cap=1500000###"
"group_proposal I propose we finalize the terms.###"
"exec_escalation Requesting executive meeting.###"
```

---

## 9. Observation Mechanism

### 9.1 Engagement Noise

**What**: Stochastic perturbation added to engagement observations.

**Why**: Models real uncertainty in reading committee sentiment - you can't perfectly observe internal states.

**How**:
```python
# Configuration
engagement_noise_sigma: float = 0.03  # ~3% noise
echo_recall_probability: float = 0.70  # 70% chance other stakeholders notice

# Noise application
true_engagement = compute_engagement_level(belief)
noisy_engagement = np.clip(true_engagement + N(0, sigma), 0.0, 1.0)

# Engagement history window (size 5)
engagement_history[sid].append(noisy_engagement)
engagement_history[sid].pop(0)  # Slide window
```

**Key Property**: Noise cannot be cancelled because it's i.i.d. Gaussian - averaging multiple observations doesn't systematically reduce it.

---

### 9.2 Weak Signals

**What**: Hints about stakeholder concerns that the agent can infer.

**Why**: Provides indirect information about hidden constraints without explicit revelation.

**How**:
```python
def _generate_weak_signals(self) -> Dict[str, List[str]]:
    for sid in stakeholders:
        signals = []
        eng_level = noisy_engagement[sid]
        delta = eng_level - historical_average

        if eng_level > 0.7: signals.append("high_engagement")
        elif eng_level < 0.3: signals.append("low_engagement")

        if delta > 0.1: signals.append("improving_engagement")
        elif delta < -0.1: signals.append("declining_engagement")

        if belief.confidence < 0.4: signals.append("high_uncertainty")

        weak_signals[sid] = signals if signals else ["neutral"]
```

---

### 9.3 Cross-Stakeholder Echoes

**What**: Records of which stakeholders noticed the vendor's action to others.

**Why**: Observable evidence of belief propagation through the causal graph.

**How**:
```python
def _generate_cross_stakeholder_echoes(self, action):
    echoes = []
    targeted = action.target_ids[0]
    for sid in stakeholders:
        if sid == targeted: continue
        if random.random() < 0.70:  # echo_recall_probability
            echoes.append({"from": targeted, "to": sid, "content": "cross_reference"})
    return echoes
```

---

## 10. Deal Stage System

### 10.1 Stage Progression Gates

**What**: Milestone-based deal progression with approval thresholds.

**Why**: Structures the negotiation into distinct phases with different requirements.

**How**:
```python
# Stage advancement requirements:
# evaluation → negotiation: weighted_utility >= 0.65 OR contacted mandatory stakeholders
# negotiation → legal_review: weighted_utility >= 0.65 AND known constraints
# legal_review → final_approval: mandatory_workable AND requested_clear
# final_approval → closed: can_close(action)

# Stage regression (if blockers active in legal_review/final_approval):
# Regress to previous stage, increment stage_regressions counter
```

**Hard Veto Conditions** (immediate termination):
- Legal review without DPA → "missing_dpa"
- Legal review without security_cert → "missing_security_cert"
- Final approval without proposal_submitted → "missing_final_proposal"

---

## 11. LLM Integration

### 11.1 LLM Client (`llm_client.py`)

**What**: GPT-4o-mini integration for stakeholder response generation and deliberation summaries.

**Why**: Provides realistic committee dialogue when enabled, at the cost of API latency and reliability.

**How**:
```python
def llm_call_text(prompt, call_type, temperature, context, allow_skip, policy, timeout):
    # Retry with exponential backoff for auto-recoverable errors
    # Interactive pause for auth/rate-limit issues
    # Return None (skip) if API unavailable and allow_skip=True
```

**Call Types**:
- `stakeholder_response`: Generate stakeholder reply (200 tokens max)
- `deliberation_summary`: Summarize committee deliberation (220 tokens max)

**Error Handling**:
- Auto-retry: network timeouts, 5xx errors, empty responses
- Interactive: auth failures, quota exceeded (requires human intervention)
- Skip fallback: returns None, caller uses template responses

---

### 11.2 Prompt Templates

**What**: Template-based stakeholder responses when LLM is disabled.

**Why**: Ensures environment works without API access, enabling deterministic testing.

**How**:
```python
TEMPLATE_RESPONSES = {
    "Legal": {
        "supportive": "Thank you for the document. The liability safeguards look reasonable...",
        "neutral": "I've reviewed the materials. There are some compliance considerations...",
        "skeptical": "I have significant concerns about the risk allocation...",
        "hostile": "This proposal raises serious compliance concerns..."
    },
    # ... similar for Finance, TechLead, Procurement, Operations, ExecSponsor
}

def get_template_response(stakeholder_name, stance):
    return TEMPLATE_RESPONSES[stakeholder_name].get(stance, TEMPLATE_RESPONSES[stakeholder_name]["neutral"])
```

---

## 12. Environment Constants

```python
# Reward Weights
REWARD_WEIGHTS = {"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}

# Terminal Rewards
TERMINAL_REWARDS_V2 = {
    "deal_closed": 1.0,
    "hard_veto": -1.0,
    "soft_veto": -0.8,
    "stage_regression": -0.75,
    "timeout": -0.5,
}

# Stage Gates
STAGE_GATE_THETA_PASS = 0.65      # Advance if weighted utility >= this
STAGE_GATE_THETA_STALL = 0.40       # Regress if weighted utility < this
STAGE_GATE_THETA_COMP = 0.35       # Curriculum progression gate
STAGE_GATE_WINDOW_M = 10           # Rolling window size for stage gate

# Other
DEFAULT_MAX_ROUNDS = 10
STEP_PENALTY = -0.01               # Per-step cost to encourage efficiency
VETO_WARNING_THRESHOLD_RATIO = 0.70  # Precursor fires at 70% of tau
LOOKAHEAD_COST = 0.07             # Exact cost for lookahead action
```

---

## 13. Key Design Decisions

### 13.1 Why Partial Observability?

The causal graph G is intentionally hidden from the agent. Agents must infer committee structure through observation of cross-stakeholder echoes and weak signals. This:

- Prevents shortcut learning where agent memorizes graph topology
- Requires genuine causal reasoning
- Makes the environment more realistic (real committees don't publish their influence network)

### 13.2 Why CVaR Veto?

Standard RL environments use expected utility maximization. CVaR (Conditional Value at Risk) specifically protects against bad tail outcomes, which is critical for:

- Enterprise procurement (regulatory compliance, legal liability)
- Multi-stakeholder settings where minority objections can block deals
- Realistic committee dynamics where risk-averse members have veto power

### 13.3 Why Five Reward Dimensions?

A single scalar reward provides insufficient learning signal for complex negotiations. Five dimensions allow:

- Disentangling what the agent does well vs poorly
- Curriculum learning (focus on specific weaknesses)
- Better interpretability of agent behavior

### 13.4 Why Two Veto Types?

- **Soft veto (CVaR)**: Rational risk management - deal killed by risk-averse member despite good expected value
- **Hard veto (policy breach)**: Punishes clearly suboptimal behavior (not bringing DPA to legal review)

This separation ensures the agent learns both risk awareness and procedural compliance.

---

## 14. Data Flow Summary

```
Agent Action
    ↓
Action Validation & Normalization
    ↓
Bayesian Belief Update (per stakeholder)
    ↓
Committee Deliberation (belief propagation)
    ↓
Risk Evaluation (CVaR computation per stakeholder)
    ↓
Reward Scoring (5 dimensions)
    ↓
Veto Check (soft + hard)
    ↓
Stage Advancement/Regression
    ↓
Observation Construction (partial observability filter)
    ↓
Return (observation, reward, done, info)
```

---

## 15. Dependencies

```
numpy>=1.21
pydantic>=2.0
openai>=1.0 (optional, for LLM stakeholder responses)
requests>=2.28 (for containerized API tests)
pytest>=7.0 (for testing)
torch>=2.0 (optional, for checkpointing)
```

---

*This architecture document describes DealRoom S2P V2, an LLM training environment for B2B software negotiation. The environment is designed to train Large Language Models, not neural network agents, using GRPO or similar reinforcement learning algorithms.*
