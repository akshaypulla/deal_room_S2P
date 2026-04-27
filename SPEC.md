# DealRoom S2P V3 — Technical Design Specification

## Formal MDP Definition

DealRoom S2P V3 is a **partially observable Markov decision process (POMDP)** framed as an episodic RL environment. The agent controls a vendor negotiating with a 6-stakeholder buying committee. The environment is designed to train an LLM (not a neural network agent).

```
MDP = (S, A, T, R, γ, Ω, O)
S      : State space (hidden + observable components)
A      : Action space (8 action types)
T      : Transition function (stakeholder belief updates + deliberation)
R      : Reward function (5-dim vector → scalar via weighted sum)
γ      : Discount factor (0.3 for multi-step rollout)
Ω      : Observation space (partial view of state)
O      : Observation function (POMDP noise injection)
```

---

## 1. State Space S

The full state `s ∈ S` is composed of **hidden** and **observable** components.

### 1.1 Hidden State Components (Agent Cannot See)

| Component | Type | Description |
|-----------|------|-------------|
| `G = (V, E, w)` | Causal Graph | Directed influence graph between stakeholders. `V = {Legal, Finance, TechLead, Procurement, Operations, ExecSponsor}`. Edges `E` and weights `w` are unique per reset. **Hidden from agent (P1).** |
| `B_i^t` | Belief Distribution | Bayesian belief distribution over 6 vendor types for each stakeholder `i`. `B_i^t ∈ Δ^6`, a probability simplex over `{competent, incompetent, trustworthy, deceptive, aligned, misaligned}`. |
| `CVaR_i^t` | float | Conditional Value at Risk for stakeholder `i` evaluated over outcome distribution derived from current deal terms. Used for veto decisions. |
| `U_i^t` | float | Expected utility for stakeholder `i`. |
| `θ_i` | float | CVaR veto threshold for stakeholder `i`. Stakeholder `i` vetoes if `CVaR_i^t > θ_i` after 2+ consecutive veto precursor rounds. |

### 1.2 Observable State Components

| Component | Type | Description |
|-----------|------|-------------|
| `stage^t` | str | Current deal stage: `evaluation → negotiation → legal_review → final_approval → closed` |
| `momentum^t` | str | Deal momentum: `stalling`, `progressing`, `fragile`, `critical` |
| `blockers^t` | List[str] | Active blocker reasons (e.g., `missing_dpa`, `tail_risk_concern`) |
| `precursors_i^t` | bool | Whether stakeholder `i` is in veto precursor state (precursor for 2+ rounds → veto trigger) |
| `terms^t` | dict | Current deal terms: `price`, `timeline_weeks`, `has_dpa`, `has_security_cert`, `liability_cap`, `support_level` |
| `round^t` | int | Current round number (1–10) |
| `max_rounds` | int | Maximum rounds (10) |

### 1.3 Full State Representation

```python
DealRoomState = {
    episode_id: str,
    step_count: int,
    task_id: str,              # "aligned" | "conflicted" | "hostile_acquisition"
    round_number: int,
    max_rounds: int,
    stakeholders: Dict[str, role],
    stakeholder_private: Dict[str, private_beliefs],  # hidden
    hidden_constraints: Dict,   # hidden
    relationship_edges: List,   # hidden
    commitment_ledger: List,
    deferred_effects: List,
    offer_state: Dict,          # deal terms
    feasibility_state: Dict,
    active_blockers: List[str],
    deal_stage: str,
    deal_momentum: str,
    rounds_since_last_contact: Dict[str, int],
    approval_caps: Dict,
    weak_signal_history: Dict[str, List],
    requested_artifacts: Dict[str, List],
    stage_regressions: int,
    veto_stakeholder: Optional[str],
    terminal_outcome: str,
    deal_failed: bool,
    failure_reason: str,
}
```

### 1.4 Belief Distribution

Each stakeholder `i` maintains a belief distribution `B_i ∈ Δ^6`:

```python
BeliefDistribution = {
    distribution: {
        "competent": float,    # positive trait
        "incompetent": float,  # negative trait
        "trustworthy": float,  # positive trait
        "deceptive": float,    # negative trait
        "aligned": float,       # positive trait
        "misaligned": float,    # negative trait
    },
    stakeholder_role: str,
    confidence: float,         # uncertainty level [0, 1]
    history: List[Tuple],       # belief change history
}
```

**Positive mass:** `pm(B_i) = P(competent) + P(trustworthy) + P(aligned)`
**Negative mass:** `nm(B_i) = P(incompetent) + P(deceptive) + P(misaligned)`
**Engagement level:** `eng(B_i) = pm(B_i) − nm(B_i)`

---

## 2. Observation Space Ω

The agent observes a **corrupted, partial** view of the state (POMDP).

### 2.1 Observable Signals

| Signal | Type | Noise Model | Description |
|--------|------|-------------|-------------|
| `engagement_level_i` | float [0,1] | `N(σ=0.10)` added to true engagement | Noisy engagement proxy for belief change. **Cannot be perfectly cancelled (P6).** |
| `engagement_level_delta_i` | float [-1,1] | Clipped noisy delta | Change in engagement since last round |
| `weak_signals_i` | List[str] | 30% random drop (POMDP) | Indirect hints: `high_engagement`, `declining_engagement`, `high_uncertainty` |
| `veto_precursors` | Dict[str, str] | Deterministic from CVaR | Warning messages for stakeholders at risk |
| `stakeholder_messages_i` | str | 20% corrupt to placeholder | Stakeholder response text (corrupted for POMDP) |
| `cross_stakeholder_echoes` | List[Dict] | 70% recall probability (P4) | Influence cascades between stakeholders |
| `deal_stage` | str | Deterministic | Current stage in approval pipeline |
| `deal_momentum` | str | Deterministic | `stalling | progressing | fragile | critical` |
| `active_blockers` | List[str] | Deterministic | Unresolved blocking issues |
| `round_number` | int | Deterministic | Current round |
| `days_to_deadline` | int | Deterministic | Countdown (starts at 30) |

### 2.2 POMDP Observation Corruption Functions

```python
def _apply_pomdp_noise(engagement: Dict[str, float]) -> Dict[str, float]:
    # σ = 0.10 — adds irreducible noise to engagement levels
    for sid, level in engagement.items():
        noise = rng.normal(0, 0.10)
        corrupted[sid] = clip(level + noise, 0.0, 1.0)
    return corrupted

def _apply_weak_signal_noise(weak_signals: Dict[str, List[str]]) -> Dict[str, List[str]]:
    # 30% random drop — agent sees incomplete hints
    for sid in weak_signals:
        kept = [s for s in signals if rng.random() > 0.30]
        weak_signals[sid] = kept if kept else ["neutral"]
    return weak_signals

def _apply_message_corruption(messages: Dict[str, str]) -> Dict[str, str]:
    # 20% corrupt to placeholder
    for sid in messages:
        if rng.random() < 0.20:
            messages[sid] = "[Message received - content not fully visible]"
    return messages
```

### 2.3 What Is Hidden From the Agent (P1)

- The causal graph `G` structure and edge weights
- True belief distributions `B_i` for each stakeholder
- CVaR values and veto thresholds `θ_i`
- The relationship between engagement noise and true beliefs
- Which stakeholders are in the same functional cluster

---

## 3. Action Space A

### 3.1 Action Types

| Action Type | Parameters | Effect |
|-------------|------------|--------|
| `send_document` | `target`, `doc_type`, `message`, `documents` | Send artifact to addressed stakeholder. Updates their beliefs. doc_type ∈ {dpa, security_cert, roi_model, implementation_timeline, compliance_report} |
| `direct_message` | `target`, `message` | Send text message to specific stakeholder. Lighter belief update than send_document. |
| `concession` | `target`, `proposed_terms` | Offer term adjustment: `price`, `liability_cap`, `timeline_weeks`. Updates deal terms directly. |
| `group_proposal` | `message` | Propose to all stakeholders. Weaker per-stakeholder effect but broadcasts intent. |
| `exec_escalation` | `message` | Activate ExecSponsor. High-impact but signals desperation. |
| `submit_proposal` | `proposal` object | Formal proposal submission with SLA, compliance attestations, pricing table. Triggers final approval check. |
| `redline_clause` | `clause_id`, `proposed_text`, `rationale` | Counter-offer specific clause. High stakes. |
| `acknowledge_stage` | — | Acknowledge current deal stage. No-op but records intent. |

### 3.2 Action Format

Two formats are supported (backward compatible):

**Pipe-delimited (primary):**
```
send_document Legal dpa | Our DPA covers GDPR Article 28 obligations.
direct_message Finance | We offer competitive ROI projections.
concession Finance | price=175000
group_proposal | All parties are aligned on key terms.
exec_escalation | Requesting executive meeting to discuss terms.
```

**Hash-delimited (legacy):**
```
send_document Legal DPA Our DPA covers GDPR obligations. ###
direct_message Finance We offer competitive pricing. ###
concession Finance price=175000 We can reduce to $175k. ###
```

### 3.3 Action Parsing

```python
def parse_action_text(text: str) -> DealRoomAction:
    # Pipe format: action_type target | message
    # Hash format: action_type target message ###
    # Fallback: direct_message with extracted stakeholder name
```

Parser tries patterns in order; first match wins. If no pattern matches, extracts stakeholder name from text as fallback. **Parse success is 100%** with pipe format.

### 3.4 Action Effects on State

| Action | Belief Update | Deal Terms | CVaR Impact |
|--------|--------------|------------|-------------|
| `send_document(target, dpa)` | Moderate positive | `has_dpa=True` | Reduces CVaR for Legal (λ_risk > 0.40) |
| `send_document(target, security_cert)` | Moderate positive | `has_security_cert=True` | Reduces CVaR for TechLead |
| `send_document(target, roi_model)` | Small positive | — | Reduces CVaR for Finance |
| `concession(target, price=N)` | — | `price=N` | Reduces CVaR for Finance if price reduced |
| `concession(target, liability_cap=N)` | Moderate positive | `liability_cap=N` | Reduces CVaR for Legal |
| `exec_escalation` | Large positive for ExecSponsor | — | Activates ExecSponsor in deliberation |

---

## 4. Reward Signal R

The reward signal is **5-dimensional** and computed deterministically from world state deltas — no LLM calls, no caching.

### 4.1 Reward Dimensions

| Dimension | Weight | Formula | Range |
|-----------|--------|---------|-------|
| **goal** | 0.25 | `f(approval_delta, blocker_resolution, CVaR_headroom_change)` | [-1.5, +1.5] |
| **trust** | 0.20 | `f(trustworthy_mass_delta_for_targeted_stakeholder)` | [-1.5, +1.5] |
| **info** | 0.20 | `f(entropy_reduction_across_committee)` | [-1.5, +1.5] |
| **risk** | 0.20 | `f(CVaR_improvement_for_risk_averse_stakeholders)` | [-1.5, +1.5] |
| **causal** | 0.15 | `f(betweenness_centrality_of_targeted_node)` | [-1.5, +1.5] |

**Weights sum to 1.0.** Each dimension uses the **zero-centered sigmoid** `_squash()` function.

### 4.2 Zero-Centered Sigmoid Squash Function

```python
def _squash(raw: float, target_range: float = 1.5) -> float:
    """
    Maps raw score (can be positive/negative) to [-target_range, target_range]
    using a steep sigmoid that passes through 0 at raw=0.

    Parameters:
        raw: raw score (unbounded)
        target_range: the output range, default 1.5

    Returns:
        float in [-target_range, target_range]

    Properties:
        - _squash(0) = 0  (passes through origin)
        - As raw → +∞, output → +target_range (asymptotic)
        - As raw → -∞, output → -target_range (asymptotic)
        - Steepness controlled by REWARD_GAIN = 3.0
    """
    return target_range * (2.0 / (1.0 + np.exp(-REWARD_GAIN * raw)) - 1.0)
```

**Key advantage over `0.5 + 0.5*tanh()`:**
- Centered at 0 (not 0.5) — neutral actions get 0, not 0.5
- Steeper transition (GAIN=3.0 vs effective GAIN≈1.0 in tanh)
- Larger dynamic range: [-1.5, +1.5] per dimension

### 4.3 Goal Dimension (weight=0.25)

```python
def _score_goal(beliefs_before, beliefs_after, blockers_before, blockers_after,
                deal_stage, risk_profiles, deal_terms, authority_weights) -> float:

    # 1. Weighted approval delta
    approval_delta = Σ_i (pm(B_i^after) - pm(B_i^before)) * auth_weight_i
    total_auth = Σ_i auth_weight_i
    approval_score = approval_delta / total_auth

    # 2. Blocker resolution
    resolved = |blockers_before| - |blockers_after|  # blockers removed
    new_created = |blockers_after| - |blockers_before|  # blockers added
    blocker_score = resolved * 0.15 - new_created * 0.10

    # 3. CVaR headroom improvement (for λ_risk > 0.40 stakeholders)
    for each stakeholder i with lambda_risk > 0.40:
        headroom_before = max(0, 1 - CVaR_i^before / tau_i)
        headroom_after = max(0, 1 - CVaR_i^after / tau_i)
        headroom_delta = headroom_after - headroom_before
    veto_score = mean(headroom_deltas)

    raw = 0.50 * approval_score + 0.30 * blocker_score + 0.20 * veto_score
    return _squash(raw, target_range=REWARD_SCALE)
```

### 4.4 Trust Dimension (weight=0.20)

```python
def _score_trust(beliefs_before, beliefs_after, targeted_ids) -> float:
    if not targeted_ids: return 0.0

    for sid in targeted_ids:
        pm_delta = pm(B_sid^after) - pm(B_sid^before)
        tw_delta = P(trustworthy|B_sid^after) - P(trustworthy|B_sid^before)
        deltas.append(0.6 * pm_delta + 0.4 * tw_delta)

    mean_delta = mean(deltas)
    return _squash(mean_delta, target_range=REWARD_SCALE)
```

### 4.5 Info Dimension (weight=0.20)

```python
def _entropy_base2(values: np.ndarray) -> float:
    values = values[values > 0]
    return -Σ v * log2(v)

def _score_info(beliefs_before, beliefs_after) -> float:
    for sid in beliefs_after:
        H_before = entropy_base2(B_before[sid].distribution)
        H_after = entropy_base2(B_after[sid].distribution)
        reductions.append((H_before - H_after) / LOG2_6)  # Normalized by max entropy (log2(6))

    mean_reduction = mean(reductions)
    return _squash(mean_reduction, target_range=REWARD_SCALE)
```

**Interpretation:** Information gain = reduction in uncertainty about what kind of vendor the agent is. High entropy → agent is unreadable. Low entropy → agent has revealed its true type.

### 4.6 Risk Dimension (weight=0.20)

```python
def _score_risk(beliefs_before, beliefs_after, risk_profiles, deal_terms) -> float:
    if not risk_profiles or not deal_terms: return 0.0

    for sid, profile in risk_profiles.items():
        if profile.lambda_risk > 0.30:
            CVaR_before = compute_cvar(B_before[sid], deal_terms, profile)
            CVaR_after = compute_cvar(B_after[sid], deal_terms, profile)
            if CVaR_before > 1e-8:
                improvements.append((CVaR_before - CVaR_after) / CVaR_before)

    mean_imp = mean(improvements) if improvements else 0.0
    return _squash(mean_imp, target_range=REWARD_SCALE)
```

**CVaR computation (Conditional Value at Risk):**

```python
def compute_cvar(outcomes: np.ndarray, alpha: float) -> float:
    """
    CVaR_α = E[loss | loss ≥ VaR_α]
    Computed as mean of bottom (1-α) quantile of outcomes.

    For α = 0.95 (Legal): considers worst 5% of outcomes
    For α = 0.70 (ExecSponsor): considers worst 30% of outcomes
    """
    sorted_outcomes = np.sort(outcomes)
    cutoff_index = int(len(sorted_outcomes) * (1 - alpha))
    tail_losses = 1.0 - sorted_outcomes[:cutoff_index + 1]
    cvar = mean(tail_losses)
    return cvar
```

### 4.7 Causal Dimension (weight=0.15)

```python
def _score_causal(graph, targeted_ids) -> float:
    if not targeted_ids or not graph: return 0.0

    centrality = betweenness_centrality(graph, targeted_ids[0])
    n = len(graph.nodes)
    max_possible = ((n - 1) * (n - 2)) if n > 2 else 1.0
    raw_centrality = centrality / max_possible

    return _squash(raw_centrality, target_range=REWARD_SCALE)
```

**Betweenness centrality:** fraction of shortest paths between all pairs of stakeholders that pass through the targeted node. High betweenness → targeting this stakeholder influences the most committee dynamics.

```python
def betweenness_centrality(graph: CausalGraph, stakeholder: str) -> float:
    count = 0
    for source in graph.nodes:
        for dest in graph.nodes:
            if source == dest or source == stakeholder or dest == stakeholder:
                continue
            if shortest_path_exists(graph, source, stakeholder) and \
               shortest_path_exists(graph, stakeholder, dest):
                count += 1
    return count / ((n-1)*(n-2)) if n > 2 else 0.0
```

### 4.8 Immediate Milestone Bonuses (Injected at step time)

These bonuses fire **in addition to** the 5-dim reward signal:

```python
def _apply_milestone_bonuses(reward, action, state_before, state_after, risk_snapshot):
    bonus = 0.0

    # Stage advancement
    if stage_after > stage_before (in stage order):
        bonus += 0.5

    # Blocker resolution
    resolved = |blockers_before| - |blockers_after|
    bonus += 0.15 * resolved

    # DPA sent to Legal when CVaR risk is elevated
    if action_type == "send_document" and "dpa" in doc_names:
        if legal_cvar > 0.15 * legal_tau:
            bonus += 0.3

    # Security cert sent
    if "security_cert" in doc_names:
        bonus += 0.2

    # Veto precursor escalation (penalty)
    if len(precursors_after) > len(precursors_before):
        bonus -= 0.2

    return reward + bonus
```

### 4.9 Non-Progress Penalty

```python
def _apply_non_progress_penalty(reward, state_before, state_after) -> float:
    deltas = [pm(B_i^after) - pm(B_i^before) for all i]
    max_delta = max(|delta| for delta in deltas)
    if max_delta < 0.02:  # No meaningful belief change
        reward -= 0.1
    return reward
```

**Purpose:** Prevents the agent from choosing safe, generic actions that avoid penalties but don't progress the deal.

### 4.10 Diversity Reward

```python
def _apply_diversity_reward(reward, action) -> float:
    action_key = f"{action_type}:{sorted(target_ids)}"
    action_history.append(action_key)
    recent = action_history[-10:]

    if len(set(recent)) >= 3:
        reward += 0.05  # Encourages varied strategy
    return reward
```

**Purpose:** Prevents action collapse (always sending the same document to the same person).

### 4.11 Step Penalty

```python
STEP_PENALTY = -0.002  # per step
```

**Rationale:** Cumulative penalty over 10 rounds = -0.020, minor vs terminal rewards (±0.5 to ±1.0).

### 4.12 Terminal Rewards

```python
TERMINAL_REWARDS_V2 = {
    "deal_closed":       +1.0,   # All stakeholders approved, deal signed
    "hard_veto":         -1.0,   # Policy breach (missing DPA in legal_review, etc.)
    "soft_veto":         -0.8,   # CVaR veto (CVaR > tau for 2+ precursor rounds)
    "stage_regression":  -0.75,  # Deal moved backward in stage pipeline
    "timeout":           -0.5,   # Max rounds reached, deal not closed
}
```

### 4.13 Full Reward Scalar Computation

```python
def compute_step_reward(action, state_before, state_after, info) -> float:
    # 1. 5-dim utterance scorer
    score = utterance_scorer.score(action, state_before, state_after, graph)
    base_reward = score.weighted_sum(REWARD_WEIGHTS)  # sum(w_i * dim_i)

    # 2. Step penalty
    reward = base_reward + STEP_PENALTY

    # 3. Milestone bonuses
    reward = _apply_milestone_bonuses(reward, action, state_before, state_after, risk_snapshot)

    # 4. Non-progress penalty
    reward = _apply_non_progress_penalty(reward, state_before, state_after)

    # 5. Diversity reward
    reward = _apply_diversity_reward(reward, action)

    # 6. Terminal reward (if done)
    if done:
        reward += terminal_reward

    return reward
```

**Theoretical range:** approximately [-2.0, +2.5] per step, with terminal adding ±1.0.

---

## 5. Transition Dynamics T

### 5.1 State Transition Sequence

```
s_t ──action──▶ s_t+1
```

Each `step()` call executes:

1. **Offer state update** (`_apply_action_to_offer_state`)
   - Deal terms modified based on `action.proposed_terms`
   - Document flags set (`has_dpa`, `has_security_cert`)
   - Concession effects applied

2. **Deal stage update** (`_update_deal_stage`)
   - Stage gates checked at rounds 3, 6, 8, 10
   - `evaluation → negotiation → legal_review → final_approval → closed`

3. **Bayesian belief update** (`bayesian_update`)
   - Each stakeholder's belief `B_i` updated based on action type, target, documents
   - Targeted stakeholder receives strongest update
   - Non-targeted stakeholders receive weaker updates

4. **Committee deliberation** (`CommitteeDeliberationEngine.run`)
   - 3-step (or 4-step for hostile) belief propagation through causal graph
   - Cross-stakeholder influence simulated
   - Committee vote computed
   - ExecSponsor activation check
   - Silent period computed

5. **Noisy engagement update** (`_update_noisy_engagement`)
   - True engagement delta computed
   - Noise `N(σ=0.05)` added
   - History window updated

6. **Veto check**
   - CVaR evaluated against thresholds `θ_i`
   - Precursor streaks tracked
   - Veto fires if `CVaR_i > θ_i` for 2+ consecutive rounds (or grace round for hostile)

### 5.2 Belief Update Rule

```python
def bayesian_update(belief: BeliefDistribution, action_type: str,
                    documents: List, stakeholder_role: str, is_targeted: bool):
    if is_targeted:
        if action_type == "send_document":
            # Strong positive update
            delta = 0.10 to 0.20 depending on document type
        elif action_type == "concession":
            delta = 0.08
        elif action_type == "direct_message":
            delta = 0.05
        else:
            delta = 0.03
    else:
        # Weaker update based on action type
        delta = 0.02
```

### 5.3 Belief Propagation Through Causal Graph

```python
def propagate_beliefs(graph, beliefs_before_action, beliefs_after_action, n_steps=3):
    current_beliefs = beliefs_after_action.copy()

    for step in range(n_steps):
        next_beliefs = current_beliefs.copy()
        for dest in graph.nodes:
            influencers = graph.get_influencers(dest)
            total_delta = Σ weight(src→dest) * (pm(B_src^current) - pm(B_src^before))

            # Kerrasic gate (prevents vanishing gradients)
            k_gate = 10.0
            scaled_delta = total_delta * (|total_delta| / (|total_delta| + 1/k_gate))
            if |total_delta| > 0 and |scaled_delta| < 0.003:
                scaled_delta = sign(total_delta) * 0.003

            # Damping
            effective_delta = scaled_delta * (0.95 ** step)
            next_beliefs[dest] = _apply_belief_delta(current_beliefs[dest], effective_delta)

        current_beliefs = next_beliefs
    return current_beliefs
```

### 5.4 Causal Graph Sampling (P2, P10)

Every `reset()` generates a **unique causal graph**:

```python
def sample_graph(stakeholder_set, authority_hierarchy, scenario_type, rng):
    # Edge probability depends on scenario and functional clusters
    for source, dest in stakeholder_set^2:
        if same_functional_cluster(source, dest):
            p_edge = base_prob + intra_cluster_boost  # higher
        else:
            p_edge = base_prob - cross_cluster_penalty  # lower

        if authority(source) >= 4:
            p_edge = 1.0  # Always connect from high-authority

        if rng.random() < p_edge:
            weight = clip(normal(mean=0.45, std=0.15), 0.05, 0.95)
            edges[(source, dest)] = weight
```

**Scenario parameters:**

| Scenario | Edge Prob | Intra-cluster Boost | Cross-cluster Penalty | Authority Edge Prob |
|----------|----------|-------------------|----------------------|---------------------|
| aligned | 0.30 | +0.40 | -0.20 | 0.85 |
| conflicted | 0.45 | +0.50 | -0.15 | 0.80 |
| hostile_acquisition | 0.60 | +0.25 | -0.05 | 0.65 |

---

## 6. Discount Factor γ

### 6.1 Value

```python
discount = 0.3  # Applied in multi-step rollout (depth=2)
```

### 6.2 Rationale

- `γ = 0.3` means the second action in a 2-step chain gets weight `0.3^1 = 0.3`
- Heavily weights the **immediate action's consequence** over downstream effects
- Cleaner credit assignment for the model's own action vs. heuristic follow-ups
- Prevents over-weighting of heuristic actions which may not match model strategy

### 6.3 Multi-Step Generation in Training

In `GRPOTrainer.run_self_play_episode()`, each step generates **2 model actions** (depth=2) before receiving heuristic followups:

```python
for step_idx in range(multi_step_depth):  # depth=2
    action = adapter.act(current_obs, rng)
    current_obs, reward, done, info = env.step(action)
    step_rewards.append(reward)
    if done: break

# Discounted reward for the chain
discounted_reward = Σ r_t * (γ ^ t) for t in [0, 1]
```

---

## 7. Key Environment Properties (P1–P10)

| ID | Property | Description |
|----|----------|-------------|
| **P1** | Hidden causal graph | `G` is fully hidden from the agent. Must be inferred from noisy engagement signals and cross-stakeholder echoes. |
| **P2** | Reset regeneration | Every `reset()` samples a new graph structure. The agent cannot memorize a fixed influence topology. |
| **P3** | CVaR veto | Stakeholders can veto even when expected utility `E[U] > 0`. Legal (α=0.95) is extremely risk-sensitive. |
| **P4** | Independent reward dimensions | 5 dimensions are computed independently from world state. No single-dimension hacking. |
| **P5** | Lookahead cost | Lookahead planning has an exact 0.07 cost penalty (`LOOKAHEAD_COST`). No free information. |
| **P6** | Noise not cancellable | Engagement noise `σ=0.10` cannot be perfectly cancelled by the agent. True beliefs are never revealed. |
| **P7** | Cross-stakeholder echoes | ~70% recall probability. Influence cascades show which stakeholders affect which others. |
| **P8** | Weak signals | Indirect hints about blockers (30% randomly dropped). Require inference to act on. |
| **P9** | Target variance | Causal score varies with the targeted stakeholder's graph position. Different targets → different causal impact. |
| **P10** | Unique graphs | Every reset produces a uniquely identifiable causal graph (via graph seed). Environment is replayable. |

---

## 8. Stakeholder Risk Profiles

| Stakeholder | Veto Power | α (CVaR) | τ (threshold) | λ_risk | Primary Concern |
|-------------|------------|----------|---------------|--------|----------------|
| **Legal** | Yes | 0.95 | 0.10 | 0.70 | Compliance breach, data protection failure |
| **Finance** | Yes | 0.90 | 0.15 | 0.50 | Payment default, cost overrun |
| **TechLead** | No | 0.80 | 0.25 | 0.30 | Implementation failure, integration complexity |
| **Procurement** | No | 0.85 | 0.20 | 0.45 | Contract enforceability, vendor viability |
| **Operations** | No | 0.80 | 0.30 | 0.35 | Operational disruption, timeline delay |
| **ExecSponsor** | Yes | 0.70 | 0.40 | 0.25 | Reputational damage, strategic misalignment |

**α (alpha):** CVaR risk sensitivity. Higher α = more focus on tail risk. Legal's α=0.95 means it evaluates the worst 5% of outcomes.

**τ (tau):** Veto threshold. Legal vetoes when CVaR loss > 0.10 (very low tolerance).

**λ_risk:** Risk weight in quality score: `quality = (1-λ_risk)*E[U] - λ_risk*CVaR`

---

## 9. Stage Gates

| Gate | Round | Pass Threshold (θ_pass) | Stall Threshold (θ_stall) |
|------|-------|------------------------|--------------------------|
| evaluation → negotiation | 3 | 0.65 weighted approval sum | 0.40 |
| negotiation → legal_review | 6 | 0.65 | 0.40 |
| legal_review → final_approval | 8 | 0.65 | 0.40 |
| final_approval → closed | 10 | All stakeholders ≥ 0.55 positive mass | — |

**Weighted approval sum:**
```python
weighted_sum = Σ_i pm(B_i) * authority_weight_i / Σ_i authority_weight_i
```

---

## 10. Training Configuration

### 10.1 GRPO Hyperparameters

```python
GRPOTrainer(
    model_id="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    learning_rate=5e-6,
    entropy_coef=0.01,           # Promotes action diversity
    grpo_clip=0.2,
    num_generations=16,         # Group size for relative advantage
    discount=0.3,              # Multi-step discount
    n_seed_samples=3,          # Multi-seed averaging per completion
)
```

### 10.2 Why These Values

| Parameter | Value | Reason |
|-----------|-------|--------|
| `num_generations` | 16 | Larger group → more stable advantage estimation |
| `entropy_coef` | 0.01 | Light pressure against mode collapse |
| `discount` | 0.3 | Clean credit assignment, less noise from heuristics |
| `n_seed_samples` | 3 | Average over 3 seeds → robust evaluation, eliminates env luck |
| `num_generations` | 16 | 2x previous (8) → better gradient quality |

### 10.3 Reward Function Multi-Seed Averaging

```python
class DealRoomGRPOReward:
    def __call__(self, prompts, completions, **kwargs):
        for completion in completions:
            # Evaluate on 3 different environment seeds
            step_rewards = []
            for seed_offset in [seed, seed+1, seed+2]:
                env = DealRoomV3S2P()
                obs = env.reset(seed=seed_offset, task_id=task_id)
                action = self._parse_action(completion)
                obs, reward, done, info = env.step(action)
                step_rewards.append(reward)
                env.close()

            mean_reward = mean(step_rewards)
            rewards.append(mean_reward)
        return rewards
```

---

## 11. Environment Constants Summary

```python
# Reward weights (sum = 1.0)
REWARD_WEIGHTS = {"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}

# Terminal rewards
TERMINAL_REWARDS_V2 = {
    "deal_closed": 1.0, "hard_veto": -1.0, "soft_veto": -0.8,
    "stage_regression": -0.75, "timeout": -0.5,
}

# Step penalty
STEP_PENALTY = -0.002

# Lookahead cost
LOOKAHEAD_COST = 0.07

# Reward scaling (zero-centered sigmoid)
REWARD_SCALE = 1.5   # output range [-1.5, +1.5]
REWARD_GAIN = 3.0    # steepness

# Stage gates
STAGE_GATE_THETA_PASS = 0.65
STAGE_GATE_THETA_STALL = 0.40

# POMDP noise
POMDP_NOISE_SIGMA = 0.10
WEAK_SIGNAL_DROP_PROB = 0.30
MESSAGE_CORRUPT_PROB = 0.20

# Max rounds
DEFAULT_MAX_ROUNDS = 10
```

---

## 12. Complete Reward Flow Diagram

```
Action a_t
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Offer State Update               │
│    - terms modified                 │
│    - document flags set             │
└─────────────────┬───────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Deal Stage Update                │
│    - stage gates checked             │
│    - stage may advance/regress       │
└─────────────────┬───────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Bayesian Belief Update            │
│    - targeted stakeholder stronger  │
│    - others receive weaker update    │
└─────────────────┬───────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Committee Deliberation           │
│    - 3-step belief propagation      │
│    - cross-stakeholder influence    │
│    - committee vote computed        │
└─────────────────┬───────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. CVaR Evaluation                  │
│    - compute CVaR for each stakeholder│
│    - check against thresholds       │
│    - track veto precursor streaks    │
└─────────────────┬───────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. Reward Computation                │
│                                     │
│ a) 5-dim utterance scorer           │
│    goal_score    = _squash(raw, 1.5)│
│    trust_score   = _squash(raw, 1.5) │
│    info_score    = _squash(raw, 1.5) │
│    risk_score    = _squash(raw, 1.5) │
│    causal_score  = _squash(raw, 1.5) │
│    base_reward = Σ w_i * score_i    │
│                                     │
│ b) STEP_PENALTY = -0.002            │
│                                     │
│ c) Milestone bonuses               │
│    +0.5  stage advance              │
│    +0.15  per blocker resolved      │
│    +0.3   DPA to Legal (if risk)   │
│    +0.2   security cert            │
│    -0.2   veto precursor added     │
│                                     │
│ d) Non-progress penalty = -0.1      │
│                                     │
│ e) Diversity reward = +0.05         │
└─────────────────┬───────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 7. Terminal Reward (if done)         │
│    +1.0  deal_closed                │
│    -1.0  hard_veto                  │
│    -0.8  soft_veto                  │
│    -0.75 stage_regression           │
│    -0.5  timeout                    │
└─────────────────┬───────────────────┘
    │
    ▼
       r_t ∈ [-2.0, +2.5] (approx)
```

---

## 13. Scenarios

| Scenario | Difficulty | Description | Key Features |
|----------|------------|-------------|--------------|
| **aligned** | Easy | Cooperative stakeholders, standard document sequencing | Low edge density (0.30), high authority boost (0.85), 3 deliberation steps |
| **conflicted** | Medium | CTO-CFO tension, Legal-Procurement coalition | Medium edge density (0.45), risk/risk cluster separation, 3 deliberation steps |
| **hostile_acquisition** | Hard | Post-acquisition authority shift, new compliance requirements | High edge density (0.60), 1 veto grace round, 4 deliberation steps, Legal veto likely without proper DPA |

---

*Document version: 1.0.0 — DealRoom S2P V3*
*Last updated: 2026-04-27*
