# DealRoom v3 — Architecture Documentation
**Version:** S2P_V_1
**Purpose:** LLM training environment via RL from LLM Feedback (GRPO)
**Scope:** All executable Python code (.py files, excluding .md and test files)

---

## Overview

DealRoom v3 is a multi-stakeholder enterprise deal negotiation simulator designed to train LLMs to close complex B2B procurement deals. An LLM plays the role of a vendor sales representative who must navigate a committee of stakeholders (Finance, Legal, TechLead, Procurement, Operations, ExecSponsor), each with private beliefs, risk preferences, veto power, and hidden constraints.

The environment is framed as an RL problem where a reward signal is derived from how well the LLM advances the deal through negotiation stages, builds trust, shares appropriate artifacts, and avoids vetos. The reward is shaped using five dimensions and a terminal payoff table.

**Key insight:** The environment uses a hidden causal graph (G) that models how stakeholder beliefs influence each other. This graph is not shown to the LLM. The environment also uses lookahead simulation (cost=0.07 per use) to let the LLM preview outcomes before committing to an action.

---

## Layer 1 — Data Models

### `models.py`

**What:** Defines the core Pydantic dataclasses that represent every object flowing through the system: `DealRoomAction`, `DealRoomObservation`, `DealRoomState`, `DealRoomReward`, `LookaheadRequest`, `SimulationResult`, and their sub-components (`PricingTable`, `SLACommitment`, `SubmitProposalAction`, `RedlineClauseAction`, `AcknowledgeStageAction`).

**Why:** A well-typed message schema is essential when an LLM is both consuming observations and producing actions. Using Pydantic ensures field validation, truncation (messages capped at 1200 characters), and target normalization happen automatically. The `DealRoomState` carries the full episode state including private stakeholder data, hidden constraints, commitment ledger, and terminal outcome metadata.

**How:**
```python
class DealRoomAction(BaseModel):
    action_type: str = "direct_message"
    target: str = "all"
    target_ids: List[str] = Field(default_factory=list)
    message: str = ""
    documents: List[Dict[str, str]] = Field(default_factory=list)
    proposed_terms: Optional[Dict[str, Any]] = None
    lookahead: Optional["LookaheadRequest"] = None
    submit_proposal: Optional[SubmitProposalAction] = None
    redline_clause: Optional[RedlineClauseAction] = None

    @field_validator("message")
    def truncate_message(cls, value: str) -> str:
        return value[:1200]

    @model_validator(mode="after")
    def sync_targets(self) -> "DealRoomAction":
        if self.target_ids and self.target == "all":
            self.target = ",".join(self.target_ids)
        return self
```

The `LookaheadRequest` lets the LLM request preview simulation. When provided, the environment calls `LookaheadSimulator.simulate()` and deducts `LOOKAHEAD_COST = 0.07` from the reward signal. The `SimulationResult` carries predicted stakeholder responses, belief deltas, and CVaR impact.

---

## Layer 2 — Environment

### `deal_room/environment/dealroom_v3.py` (Core)

**What:** The `DealRoomV3` class is the main Gym-style environment. It exposes `reset(seed, task_id)` returning a `DealRoomObservation`, and `step(action)` returning `(observation, reward, done, info)`. It orchestrates belief propagation, lookahead simulation, veto checking, stage gating, and reward computation.

**Why:** This is the central orchestration point. It wires together the causal graph, belief tracker, deliberation engine, lookahead simulator, reward scorer, and CVaR risk evaluator. Every step is a full negotiation round: the vendor acts, beliefs update via Bayesian inference, the committee deliberates (potentially activating the ExecSponsor), noisy engagement is updated, stakeholder responses are generated, and a reward is computed. The episode terminates when a veto is triggered, max rounds are reached, or stage regression occurs.

**How — reset:**
```python
def reset(self, seed=None, task_id="aligned", **kwargs) -> DealRoomObservation:
    self._rng = np.random.default_rng(seed)
    self._episode_id = str(uuid.uuid4())[:8]
    self._graph = sample_graph(stakeholder_set=STANDARD_STAKEHOLDERS,
                               authority_hierarchy=STANDARD_HIERARCHY,
                               scenario_type=task_id, rng=self._rng)
    self._beliefs = {sid: BeliefDistribution(distribution=_get_initial_beliefs(task_id, sid),
                                            stakeholder_role=sid, confidence=1.0, history=[])
                     for sid in STANDARD_STAKEHOLDERS}
    self._noisy_engagement = {sid: float(np.clip(0.5 + self._rng.normal(0, OBS_CONFIG.engagement_noise_sigma), 0.0, 1.0))
                              for sid in STANDARD_STAKEHOLDERS}
    # ... offer_state initialized with scenario-specific terms
```

**How — step:**
```python
def step(self, action: DealRoomAction):
    # 1. Normalize and validate action
    action = self._normalize_action(action)

    # 2. Optionally run lookahead simulation (costs 0.07)
    lookahead_result = self._run_lookahead(action) if action.lookahead else None

    # 3. Capture state snapshot for reward computation
    state_before = StateSnapshot(beliefs=dict(self._beliefs), ...)

    # 4. Advance round counter
    self._step_count += 1
    self._round_number += 1

    # 5. Apply action to offer_state (update terms, documents, pricing)
    self._apply_action_to_offer_state(action)

    # 6. Update deal stage (evaluation -> negotiation -> legal_review -> final_approval -> closed)
    self._update_deal_stage()

    # 7. Check hard veto (missing DPA or security cert in late stages)
    hard_veto_reason = self._check_hard_veto_for_stage()
    if hard_veto_reason:
        return self._build_early_termination_obs(action, reward, hard_veto_reason, info)

    # 8. Bayesian belief update for all stakeholders
    for sid in STANDARD_STAKEHOLDERS:
        self._beliefs[sid] = bayesian_update(belief=self._beliefs[sid],
                                             action_type=action.action_type,
                                             documents=action.documents,
                                             stakeholder_role=sid,
                                             is_targeted=(sid in action.target_ids))

    # 9. Committee deliberation (belief propagation + vote + exec sponsor activation)
    deliberation_engine = CommitteeDeliberationEngine(graph=self._graph, ...)
    deliberation_result = deliberation_engine.run(vendor_action=action, ...)

    # 10. Update noisy engagement
    noisy_deltas = self._update_noisy_engagement(self._beliefs, previous_beliefs)

    # 11. Generate stakeholder responses
    stakeholder_messages = self._generate_stakeholder_responses(action, previous_beliefs, ...)

    # 12. Compute reward (5-dim utterance scorer + step penalty)
    reward, reward_components = self._compute_reward(action, state_before, ...)

    # 13. CVaR risk evaluation
    risk_snapshot = self._evaluate_committee_risk(self._state.offer_state)
    precursors = self._compute_veto_precursors(risk_snapshot)

    # 14. Veto check (requires 2-round streak of CVaR > tau)
    veto_triggered, veto_stakeholder = self._check_for_veto(risk_snapshot)
    max_rounds_reached = self._round_number >= self._state.max_rounds
    done = veto_triggered or max_rounds_reached

    # 15. Terminal reward
    if done:
        terminal_reward, terminal_outcome = compute_terminal_reward(
            deal_closed=False, veto_triggered=veto_triggered, ...)
        reward += terminal_reward
```

**Key constants:**
- `LOOKAHEAD_COST = 0.07` — deducted from reward when lookahead is used
- `STEP_PENALTY = -0.01` — small negative reward each round to encourage efficiency
- `engagement_noise_sigma = 0.03` — Gaussian noise on engagement observations (sigma > 0 means noisy signals)
- `echo_recall_probability = 0.70` — probability that non-targeted stakeholders observe a cross-stakeholder echo

**Stage gates:**
- Evaluation → Negotiation: requires weighted_utility_sum >= 0.65
- Negotiation → Legal Review: requires weighted_utility_sum >= 0.65 after round 6
- Legal Review → Final Approval: requires weighted_utility_sum >= 0.65 after round 8
- Final Approval → Closed: requires all stakeholders positive_mass >= 0.55

**Veto mechanism:**
1. CVaR loss is computed per stakeholder using `compute_cvar(outcomes, alpha)` × scenario_multiplier × confidence_factor
2. Scenario multipliers: aligned=0.12, conflicted=0.22, hostile_acquisition=0.42
3. If CVaR loss exceeds tau × 0.70 for two consecutive rounds, a veto precursor is recorded
4. If CVaR loss exceeds tau for two consecutive rounds, veto is triggered

---

### `deal_room/environment/constants.py`

**What:** Shared scalar constants used across the environment and reward layers.

**Why:** Centralizing these makes it easy to tune the game balance without hunting through multiple files. The reward weights shape what the LLM learns to prioritize.

**How:**
```python
REWARD_WEIGHTS = {"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}
TERMINAL_REWARDS_V2 = {
    "deal_closed": 1.0,
    "hard_veto": -1.0,    # missing DPA/security cert in legal_review or final_approval
    "soft_veto": -0.8,    # CVaR veto
    "stage_regression": -0.75,
    "timeout": -0.5,
}
STAGE_GATE_THETA_PASS = 0.65   # weighted utility sum needed to advance stage
STAGE_GATE_THETA_STALL = 0.40  # below this, stage regresses
STEP_PENALTY = -0.01
VETO_WARNING_THRESHOLD_RATIO = 0.70  # CVaR must exceed tau * 0.70 before warning
```

---

### `deal_room/environment/lookahead.py`

**What:** `LookaheadSimulator` generates optimistic and pessimistic hypotheses about stakeholder beliefs, simulates how each hypothesis would respond to the proposed action, and returns the worst-case outcome (minimax robustness). This is the "mental planning" mechanism that lets the LLM preview consequences before committing to an action.

**Why:** lookahead is expensive (costs 0.07 reward) but gives the LLM uncertainty-aware planning capability. By comparing the worst-case predicted goal delta across hypotheses, the LLM can detect when an action is risky even if the current belief state looks favorable.

**How:**
```python
def simulate(self, action_draft, current_beliefs, n_hypotheses=2, depth=2):
    # 1. Generate two hypotheses: optimistic and pessimistic
    # Optimistic: +0.15 competent, +0.10 trustworthy, +0.10 aligned
    # Pessimistic: +0.15 incompetent, +0.10 deceptive, +0.10 misaligned

    # 2. For each hypothesis, simulate the response:
    if action_draft.action_type == "send_document":
        response_score = min(1.0, base_response_score + 0.15)
    elif action_draft.action_type == "direct_message":
        response_score = base_response_score + (0.05 if action_draft.message else -0.05)

    # 3. Return the worst-case result (minimax)
    return SimulationResult(
        predicted_responses=worst_case.responses,
        predicted_belief_deltas=worst_case.belief_deltas,
        cvar_impact=worst_case.cvar_impact,
        cost=LOOKAHEAD_COST,  # 0.07
    )
```

Prediction accuracy is computed by comparing predicted responses and belief deltas against actual observed values.

---

### `deal_room/environment/llm_client.py`

**What:** A production-quality OpenAI API client for GPT-4o-mini with retry logic, error classification, interactive pause for manual intervention, rate-limit handling, and global call statistics. It exposes `llm_call_text()`, `generate_stakeholder_response()`, and `generate_deliberation_summary()`.

**Why:** When `DEALROOM_ENABLE_LLM_SUMMARY` is set, the deliberation engine calls GPT-4o-mini to generate a natural language summary of how the committee's beliefs evolved. The client handles timeouts, 429s, 500s, auth errors, and interactive restarts gracefully.

**How — error classification:**
```python
def classify_error(exception, status_code=None, api="unknown"):
    if status_code in (401, 403):
        return LLMError(error_type=LLMErrorType.AUTH_INVALID_KEY, ...)
    if status_code == 429:
        if any(k in msg for k in ["quota", "billing", "exceeded"]):
            return LLMError(error_type=LLMErrorType.QUOTA_EXCEEDED, ...)
        return LLMError(error_type=LLMErrorType.RATE_LIMIT_429, ...)
    if any(k in msg for k in ["timeout", "timed out"]):
        return LLMError(error_type=LLMErrorType.NETWORK_TIMEOUT, ...)
    # ... DNS_FAILURE, CONNECTION_RESET, SERVER_5XX, EMPTY_RESPONSE, etc.
```

**How — retry with backoff:**
```python
def llm_call_text(...):
    while True:
        try:
            response = client.chat.completions.create(model=model, messages=[...],
                                                      max_tokens=max_tokens, temperature=temperature)
            return response.choices[0].message.content.strip()
        except Exception as raw_exc:
            err = classify_error(raw_exc, ...)
            if err.is_auto_recoverable() and attempt < policy.max_auto_retries:
                backoff = policy.compute_backoff(attempt)
                time.sleep(backoff)
                attempt += 1
                continue
            if interactive:
                action = _interactive_pause(err, context, allow_skip)
                # c=retry, w N=wait, s=skip, e=exit
```

---

### `deal_room/environment/stakeholder_llm.py`

**What:** A thin wrapper around `llm_client` that tries GPT-4o-mini to generate stakeholder responses, falling back to template-based responses on any failure (timeout, auth, rate limit, empty response).

**Why:** Stakeholder responses can be either rule-based (using the template fallback) or LLM-generated when an API key is available. This dual-mode approach keeps the environment functional in demo/offline mode while allowing richer responses in live training runs.

**How:**
```python
def generate_stakeholder_response(stakeholder_name, context, role="", stance="neutral"):
    if _use_llm():  # OPENAI_API_KEY is set
        prompt = build_stakeholder_prompt(stakeholder_name, context, role)
        result = _call_gpt4o_mini(prompt, timeout=1.5)
        if result:
            return result
    return get_template_response(stakeholder_name, stance)  # template fallback
```

---

### `deal_room/environment/text_env.py` and `deal_room/environment/prompts.py`

**What:** Text rendering and prompt templates. `text_env.py` converts `DealRoomObservation` into human-readable text for debugging or logging. `prompts.py` provides `build_stakeholder_prompt()` for LLM-based response generation and `get_template_response()` for the fallback response generator.

**Why:** Separating text rendering from the core environment keeps the simulation logic clean. Prompt templates ensure consistent framing when GPT-4o-mini generates stakeholder dialogue.

---

## Layer 3 — Committee (Belief & Deliberation)

### `deal_room/committee/causal_graph.py`

**What:** `CausalGraph` models the hidden influence network between stakeholders. Edges have weights (0.05–0.95), and authority nodes always generate outgoing edges. The `BeliefDistribution` dataclass holds a Dirichlet-like belief state over six vendor types (competent, incompetent, trustworthy, deceptive, aligned, misaligned). Key functions: `sample_graph()`, `propagate_beliefs()`, `compute_engagement_level()`, `get_betweenness_centrality()`, `compute_behavioral_signature()`.

**Why:** The causal graph is the structural backbone of the committee. It models how one stakeholder's belief shift propagates to others through influence edges. The LLM does not see this graph — it must infer it from noisy observation signals. Betweenness centrality identifies which stakeholders are information brokers (bridges between clusters). The behavioral signature shows the "ripple pattern" caused by targeting a specific stakeholder.

**How — graph sampling:**
```python
def sample_graph(stakeholder_set, authority_hierarchy, scenario_type, rng):
    params = SCENARIO_PARAMS[scenario_type]
    edges = {}
    for source in stakeholder_set:
        for dest in stakeholder_set:
            if source == dest: continue
            source_authority = authority_hierarchy.get(source, 1)
            same_cluster = _same_functional_cluster(source, dest)

            if source_authority >= 4:
                p_edge = 1.0  # Always create edges from authority nodes
            elif same_cluster:
                p_edge = params["base_edge_probability"] + params["intra_cluster_boost"]
            else:
                p_edge = max(0.0, params["base_edge_probability"] - params["cross_cluster_penalty"])

            if rng.random() < p_edge:
                w = float(np.clip(rng.normal(params["weight_mean"], params["weight_std"]), 0.05, 0.95))
                edges[(source, dest)] = w

    # Authority normalized by sum
    authority_normalized = {sid: authority_hierarchy[sid] / total_authority ...}
    return CausalGraph(nodes=list(stakeholder_set), edges=edges,
                        authority_weights=authority_normalized, ...)
```

**Scenario edge density:** aligned (low, 0.30 base), conflicted (medium, 0.45 base), hostile_acquisition (high, 0.60 base)

**How — belief propagation (3 steps):**
```python
def propagate_beliefs(graph, beliefs_before_action, beliefs_after_action, n_steps=3):
    current_beliefs = {sid: b.copy() for sid, b in beliefs_after_action.items()}
    for step in range(n_steps):
        next_beliefs = {sid: b.copy() for sid, b in current_beliefs.items()}
        for dest_id in graph.nodes:
            influencers = graph.get_influencers(dest_id)
            total_delta = sum(weight * (current_beliefs[src].positive_mass()
                                       - beliefs_before_action[src].positive_mass())
                              for src, weight in influencers.items())

            # k-gate scaling
            k_gate = 10.0
            scaled_delta = total_delta * (abs(total_delta) / (abs(total_delta) + 1.0/k_gate))
            if abs(scaled_delta) < 0.003: scaled_delta = np.sign(total_delta) * 0.003

            if abs(scaled_delta) > 1e-10:
                next_beliefs[dest_id] = _apply_belief_delta(current_beliefs[dest_id],
                                                            scaled_delta, damping=0.95**step)
        current_beliefs = next_beliefs
    return current_beliefs
```

**How — engagement level:**
```python
def compute_engagement_level(belief: BeliefDistribution) -> float:
    return belief.positive_mass() - belief.negative_mass()
    # where positive_mass = competent + trustworthy + aligned
    # and negative_mass = incompetent + deceptive + misaligned
```

---

### `deal_room/committee/belief_tracker.py`

**What:** `bayesian_update()` applies a Bayesian update to a `BeliefDistribution` given an action and documents. It uses a likelihood table (`ACTION_LIKELIHOODS`) that gives different probabilities for each vendor type depending on the action taken. Damping is applied to non-targeted stakeholders (damping=0.7 vs 1.0 for targeted).

**Why:** The committee's understanding of the vendor evolves based on what the vendor does. Sending a DPA proactively signals competence and alignment (likelihood 0.85/0.80), while sending an ROI model to Finance signals competence but trustworthiness depends more on the specific numbers. The Bayesian update propagates evidence into the belief state.

**How:**
```python
def bayesian_update(belief, action_type, documents, stakeholder_role, is_targeted):
    likelihoods = _get_likelihood(action_type, documents, stakeholder_role)
    damping = 1.0 if is_targeted else 0.7

    new_dist = {}
    for vendor_type, prior_prob in belief.distribution.items():
        likelihood = likelihoods.get(vendor_type, 0.5)
        dampened_likelihood = 1.0 + damping * (likelihood - 1.0)
        new_dist[vendor_type] = prior_prob * dampened_likelihood

    total = sum(new_dist.values())
    new_dist = {k: max(0.01, v / total) for k, v in new_dist.items()}

    # Confidence = 1 - (entropy / LOG2_6)
    probs = [new_dist.get(t, 0.01) for t in VENDOR_TYPES]
    entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
    confidence = 1.0 - (entropy / LOG2_6 if LOG2_6 > 0 else 0)

    return BeliefDistribution(distribution=new_dist, stakeholder_role=belief.stakeholder_role,
                               confidence=confidence, history=belief.history + [(action_type, damping)])
```

---

### `deal_room/committee/deliberation_engine.py`

**What:** `CommitteeDeliberationEngine` runs the internal committee dialogue after each vendor action. It propagates beliefs, computes the committee vote (approve/block/abstain per sub-agent), checks if the ExecutiveSponsor should be activated (dormant by default, wakes on 2+ blocks, exec_escalation, or >50% negative vote ratio), and computes a reactive silent period. Optionally generates an LLM summary of the deliberation via `generate_deliberation_summary()`.

**Why:** The committee is a latent multi-agent system inside the environment. After each vendor action, stakeholders discuss among themselves (via belief propagation), vote on whether the direction is acceptable, and potentially escalate to the ExecutiveSponsor. The silent period determines how many rounds of non-response occur before the next targeted message is appropriate.

**How — committee vote:**
```python
def _compute_committee_vote(self, beliefs_before, beliefs_after, vendor_action):
    vote = {}
    for sid in COMMITTEE_SUB_AGENTS:  # BusinessOwner, Legal, FinanceLead
        b_before = beliefs_before.get(sid)
        b_after = beliefs_after.get(sid)
        influence = self.issue_influence_weights.get(action_issue, {}).get(sid, 0.5)
        auth_weight = self.graph.authority_weights.get(sid, 0.5)
        pm_delta = b_after.positive_mass() - b_before.positive_mass()
        weighted_signal = pm_delta * influence * auth_weight

        if weighted_signal >= 0.15:
            vote[sid] = "approve"
        elif weighted_signal <= -0.15:
            vote[sid] = "block"
        else:
            vote[sid] = "abstain"
    return vote
```

**How — exec sponsor activation:**
```python
def _check_exec_sponsor_activation(self, committee_vote, vendor_action, beliefs_after):
    blocks = [sid for sid, v in committee_vote.items() if v == "block"]
    if len(blocks) >= 2:
        self._exec_sponsor_active = True; self._exec_sponsor_trigger = "veto_cast"; return

    if vendor_action.action_type == "exec_escalation":
        self._exec_sponsor_active = True; self._exec_sponsor_trigger = "escalation"; return

    negative_votes = sum(1 for v in committee_vote.values() if v == "block")
    total_votes = len([v for v in committee_vote.values() if v != "abstain"])
    if total_votes > 0 and negative_votes / total_votes > 0.5:
        self._exec_sponsor_active = True; self._exec_sponsor_trigger = "stage_advance"
```

**How — silent period:**
```python
def _compute_reactive_silent_period(self, beliefs_before, beliefs_after, committee_vote):
    base_silent = SILENT_PERIOD_BASE[task_type]  # aligned=2, conflicted=3, hostile_acquisition=4
    conflict_ratio = conflict_count / total_votes
    avg_delta = mean([abs(b_after.positive_mass() - b_before.positive_mass()) for ...])
    silent_duration = base_silent + int(conflict_ratio * 3) + int(avg_delta * 4 if avg_delta < 0.05 else 0)
    return min(silent_duration, max_silent[task_type])  # capped at 6/8/10
```

---

## Layer 4 — Rewards

### `deal_room/rewards/utterance_scorer.py`

**What:** `UtteranceScorer` computes a 5-dimensional reward signal for each action: goal progress, trust building, information provision, risk management, and causal awareness. Each dimension produces a score in [0, 1], and the weighted sum produces the scalar reward.

**Why:** Shaping the reward across five dimensions gives the LLM nuanced feedback about what it did well or poorly, rather than a single scalar that hides the reasoning. The causal dimension specifically measures whether the LLM's action was informed by the implied causal structure of the committee.

**How:**
```python
class UtteranceScorer:
    def score(self, action, state_before, state_after, true_graph=None, lookahead_used=False):
        # goal: deal advances through stages, appropriate artifacts sent, proposals submitted
        # trust: tone matches role expectations, credible evidence provided
        # info: shares requested artifacts, addresses known gaps
        # risk: avoids pushing too fast in high-risk scenarios, respects constraints
        # causal: action targets the right stakeholders considering graph structure

        score = UtteranceScore(
            goal=self._score_goal(action, state_before, state_after),
            trust=self._score_trust(action, state_before, state_after),
            info=self._score_info(action, state_before, state_after),
            risk=self._score_risk(action, state_before, state_after),
            causal=self._score_causal(action, state_before, state_after, true_graph),
        )
        # lookahead cost is subtracted here if lookahead_used
        return score

    def weighted_sum(self, weights):
        return (self.goal * weights["goal"] + self.trust * weights["trust"] +
                self.info * weights["info"] + self.risk * weights["risk"] +
                self.causal * weights["causal"])
```

The scoring functions inspect `action.documents`, `action.proposed_terms`, `action.action_type`, and the before/after belief states to produce dimension scores.

---

### `deal_room/rewards/pareto_efficiency.py`

**What:** `compute_terminal_reward()` applies the terminal payoff table once an episode ends. It checks whether the outcome was Pareto optimal (no stakeholder is dominated), and returns the appropriate scalar reward plus an outcome string.

**Why:** The terminal reward is the biggest reward event in the episode. Getting to `deal_closed` earns +1.0. A hard veto (missing DPA/security cert) earns -1.0. A soft veto (CVaR-based) earns -0.8. Timeout earns -0.5. Stage regression earns -0.75. If max rounds are reached and the outcome is Pareto optimal, no penalty is applied (0.0).

**How:**
```python
def compute_terminal_reward(deal_closed, veto_triggered, veto_stakeholder,
                             max_rounds_reached, stage_regressions, all_utilities,
                             cvar_losses, thresholds, is_hard_veto=False):
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
        return TERMINAL_REWARDS_V2["stage_regression"] * min(stage_regressions, 3), ...
```

---

## Layer 5 — Stakeholders

### `deal_room/stakeholders/archetypes.py`

**What:** Archetype profiles for each of the six standard stakeholders. Each archetype has a `role`, `veto_power` flag, `tau` threshold (CVaR risk tolerance), `alpha` parameter (CVaR percentile), `authority` level, and `utility_weights`. Also provides `get_archetype()` and `ARCHETYPE_PROFILES`.

**Why:** Different stakeholders have fundamentally different risk preferences. The Finance archetype has tau=0.35 (low tolerance, more aggressive veto), while Operations has tau=0.50 (higher tolerance). The archetype defines how sensitive each stakeholder is to tail-risk, which directly drives the veto mechanism.

**How — archetypes:**
```python
ARCHETYPE_PROFILES = {
    "Finance": ArchetypeProfile(role="finance", veto_power=True, tau=0.35, alpha=0.05,
                                 authority=3, utility_weights={"cost": 0.34, "risk": 0.22, ...}),
    "Legal": ArchetypeProfile(role="legal_compliance", veto_power=True, tau=0.40, alpha=0.05, ...),
    "TechLead": ArchetypeProfile(role="technical", veto_power=False, tau=0.45, alpha=0.10, ...),
    "Procurement": ArchetypeProfile(role="procurement", veto_power=False, tau=0.50, alpha=0.10, ...),
    "Operations": ArchetypeProfile(role="operations", veto_power=False, tau=0.50, alpha=0.10, ...),
    "ExecSponsor": ArchetypeProfile(role="executive_sponsor", veto_power=True, tau=0.42, alpha=0.05, ...),
}
```

---

### `deal_room/stakeholders/cvar_preferences.py`

**What:** CVaR (Conditional Value at Risk) evaluation for deal outcomes. `compute_cvar()` computes the alpha-quantile of negative outcomes (losses), which is the expected loss in the worst alpha fraction of scenarios. `compute_outcome_distribution()` generates a distribution of deal utilities by sampling from the stakeholder's utility function. `evaluate_deal()` provides a quick accept/reject signal based on whether expected utility minus CVaR exceeds a threshold.

**Why:** CVaR captures tail-risk sensitivity better than expected value alone. A stakeholder who is risk-averse cares more about the worst-case outcomes than the average outcome. The tau threshold (from the archetype) defines when CVaR losses become unacceptable and trigger the veto mechanism.

**How:**
```python
def compute_cvar(outcomes: np.ndarray, alpha: float = 0.05) -> float:
    sorted_losses = np.sort(outcomes)
    cutoff_index = int(alpha * len(sorted_losses))
    if cutoff_index == 0:
        return float(sorted_losses[0])
    return float(np.mean(sorted_losses[:cutoff_index]))

def compute_outcome_distribution(deal_terms, profile, rng, n_samples=500):
    # Sample price, timeline, security_posture from deal_terms with noise
    # Compute utility = weighted sum of (fit - reference) for each dimension
    # Add random noise scaled by profile.risk_sensitivity
    # Return n_samples outcome values
```

---

## Layer 6 — Curriculum

### `deal_room/curriculum/adaptive_generator.py`

**What:** `AdaptiveCurriculumGenerator` adjusts scenario difficulty based on the LLM's performance history. It tracks success rates per task type (aligned, conflicted, hostile_acquisition), per deal stage, and per stakeholder cluster. It uses a simple exponential moving average to estimate difficulty and can generate synthetic tasks to fill gaps in training coverage.

**Why:** Starting with hard scenarios is overwhelming for a fresh LLM. Starting with only easy scenarios leaves the LLM unprepared for adversarial situations. The curriculum gradually exposes the LLM to harder scenarios as it demonstrates competence, ensuring the training signal is appropriately calibrated at every stage of learning.

**How — difficulty scoring:**
```python
# difficulty = base_difficulty + penalty_for_failures + bonus_for_consecutive_successes
# base: aligned=0.3, conflicted=0.6, hostile_acquisition=0.9
# failure penalty: -0.05 per failed episode
# success streak bonus: +0.02 per consecutive success (capped at +0.10)
# Stage progression bonus when agent advances through stages
```

---

## Layer 7 — Training

### `deal_room/training/grpo_trainer.py`

**What:** `GRPOTrainer` implements the Group Relative Policy Optimization training loop for LLM fine-tuning. GRPO is the RL algorithm used to update the LLM's policy based on the reward signal from the environment. Also defines `TrainingMetrics` dataclass for tracking training progress.

**Why:** GRPO is a variant of PPO that uses group-relative advantage estimation — for each prompt, multiple rollouts are generated, the reward is computed per rollout, and the policy is updated to favor higher-reward rollouts. This is the core training loop that makes the LLM better at negotiation over time.

**How:**
```python
class GRPOTrainer:
    def train(self, episodes):
        # For each episode:
        # 1. Generate rollouts (multiple completions per prompt)
        # 2. Compute reward per rollout using DealRoomV3 environment
        # 3. Compute advantage = reward - baseline (mean reward of group)
        # 4. Update policy to maximize advantage-weighted log probability
        return TrainingMetrics(avg_reward=..., policy_loss=..., kl_divergence=...)
```

---

### `deal_room/training/ppo_trainer.py`

**What:** `PPOTrainer` implements Proximal Policy Optimization, an alternative to GRPO. PPO uses a clipped surrogate objective to prevent policy updates that are too large, which makes training more stable.

**Why:** PPO and GRPO are the two main RL algorithms for this type of training. GRPO is simpler and works well when the reward signal is dense. PPO is more stable in noisy environments. The codebase supports both so experiments can compare.

---

## Layer 8 — Server (HTTP API)

### `server/deal_room_environment.py`

**What:** A Flask/FastAPI-like HTTP server that wraps `DealRoomV3` as a REST endpoint. It exposes `POST /reset`, `POST /step`, `GET /state`, `GET /schema`, and `POST /validate`. It uses the `CCIGrader` for terminal scoring, `SemanticAnalyzer` for intent/tone extraction, `OutputValidator` for action parsing, and the `CommitmentLedger` for tracking claimed commitments.

**Why:** The server layer decouples the environment from the training loop. Multiple training workers can send actions to the server over HTTP, which maintains episode state and returns observations. This enables distributed training where different workers run different episodes in parallel.

**How:**
```python
@app.route("/step", methods=["POST"])
def step():
    action = OutputValidator().validate(request.json["action"], available_targets=active_stakeholders)
    obs, reward, done, info = env.step(DealRoomAction(**action))
    return jsonify({"observation": obs, "reward": reward, "done": done, "info": info})

@app.route("/reset", methods=["POST"])
def reset():
    task_id = request.json.get("task_id", "aligned")
    seed = request.json.get("seed")
    obs = env.reset(seed=seed, task_id=task_id)
    return jsonify({"observation": obs})
```

---

### `server/grader.py`

**What:** `CCIGrader` computes a deterministic terminal score (0.01–0.99) when a deal episode concludes. The score is a weighted combination of approval completeness, constraint satisfaction, term feasibility, relationship durability, and deal efficiency.

**Why:** The environment's scalar reward signal drives RL training. The grader provides a human-aligned terminal evaluation that complements the step-by-step reward signal. A deal can be technically "not failed" (no veto) but still score poorly if constraints were not resolved or relationships were damaged.

**How:**
```python
class CCIGrader:
    WEIGHTS = {
        "approval_completeness": 0.35,
        "constraint_satisfaction": 0.25,
        "term_feasibility": 0.15,
        "relationship_durability": 0.15,
        "efficiency": 0.10,
    }

    @classmethod
    def compute(cls, state: DealRoomState) -> float:
        if not state.deal_closed or state.deal_failed: return cls.MIN_SCORE
        if not state.feasibility_state.get("is_feasible", False): return cls.MIN_SCORE
        if any(not constraint.get("resolved") for constraint in state.hidden_constraints.values()): return cls.MIN_SCORE

        # Check mandatory stakeholders have approval >= 0.62 and resistance <= 0.65
        # Check veto-power stakeholders have resistance <= 0.65

        approval_score = cls._approval_completeness(state, mandatory_ids)
        constraint_score = cls._constraint_satisfaction(state)
        feasibility_score = cls._term_feasibility(state)
        relationship_score = cls._relationship_durability(state)
        efficiency_score = cls._efficiency(state)

        score = sum(score * cls.WEIGHTS[key] for key, score in [...])
        return round(max(cls.MIN_SCORE, min(cls.MAX_SCORE, score)), 4)
```

---

### `server/semantics.py`

**What:** `SemanticAnalyzer` extracts intent, tone, artifacts, and claims from vendor messages. It has three backends (fallback order): sentence-transformer embeddings (best), TF-IDF vectors (fallback), and lexical Jaccard (last resort). It matches messages against `INTENT_BANK` (13 intent categories) and `TONE_BANK` (6 tone categories).

**Why:** The server needs to understand what the vendor is trying to do beyond just parsing the action schema. A message that says "here is the DPA" has intent "share_dpa" and tone "collaborative". A message that says "this is our final offer" has intent "close_attempt" and tone "pushy". These semantic labels feed into the `StakeholderEngine` which updates trust and approval based on tone.

**How:**
```python
class SemanticAnalyzer:
    def analyze(self, message, context, stakeholder_roles):
        intent_matches = {
            intent: self._similarity(message, INTENT_BANK[intent], vectors)
            for intent, vectors in self._intent_vectors.items()
        }
        tone_scores = {
            tone: self._similarity(message, TONE_BANK[tone], vectors)
            for tone, vectors in self._tone_vectors.items()
        }
        artifact_matches = [artifact for artifact, pattern in self._artifact_patterns.items()
                            if pattern.search(message)]
        claim_candidates = self._extract_claims(lowered, artifact_matches)
        request_matches = self._match_requests(message, artifact_matches, requested_artifacts)
        return {"intent_matches": intent_matches, "tone_scores": tone_scores,
                "artifact_matches": artifact_matches, "claim_candidates": claim_candidates,
                "request_matches": request_matches, "backend": self._backend, ...}
```

---

### `server/claims.py`

**What:** `CommitmentLedger` tracks a rolling history of semantic commitments made by the vendor across the negotiation. It detects contradictions in numeric slots (price within 8%, timeline within 15%) and polarity slots (security_posture, liability, support_level, implementation_commitment).

**Why:** The LLM should be consistent with its commitments. If it promised a price of $100k in round 2 and then offers $120k in round 5, that contradiction should be detectable. The ledger provides a historical record and contradiction signal that can be used to penalize inconsistency.

**How:**
```python
class CommitmentLedger:
    def ingest(self, stakeholder_ids, claim_candidates, threshold_jitter):
        for stakeholder_id in stakeholder_ids:
            for claim in claim_candidates:
                entry["slot_threshold"] = 0.78 + threshold_jitter.get(str(claim["slot"]), 0.0)
                prior = self._latest_for(stakeholder_id, str(claim["slot"]))
                if prior and self._is_contradiction(prior, entry):
                    contradictions.append({"stakeholder_id": stakeholder_id, "slot": entry["slot"],
                                          "previous": prior, "current": entry})
                self.claims.append(entry)
        if len(self.claims) > self.max_claims:
            self.claims = self.claims[-self.max_claims:]
        return {"contradictions": contradictions, "recorded": recorded}

    def _is_contradiction(self, prior, current):
        slot = str(current["slot"])
        if slot in NUMERIC_TOLERANCES:
            return abs(prior["value"] - current["value"]) / prior["value"] > NUMERIC_TOLERANCES[slot]
        if slot in POLARITY_SLOTS:
            return prior.get("value") != current.get("value") or prior.get("polarity") != current.get("polarity")
        return False
```

---

### `server/validator.py`

**What:** `OutputValidator` parses raw LLM output (which may be free-form text containing JSON) into a structured `DealRoomAction`. It tries multiple JSON extraction patterns, falls back to heuristic extraction (action_type, target, message) if JSON parsing fails, and normalizes targets against the available stakeholder roster.

**Why:** LLMs often produce outputs like "```json\n{...}\n```" or even plain text descriptions. The validator handles all these cases and normalizes them into the proper action schema. If the LLM produces an unknown target or action type, it gracefully defaults to "direct_message" rather than crashing the episode.

**How:**
```python
def validate(self, raw, available_targets=None):
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"(\{.*\})"]:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(1).strip())
                return self._normalize(payload, available_targets), 1.0
            except json.JSONDecodeError:
                continue

    # Heuristic fallback
    heuristic = {"action_type": self._extract_action_type(raw),
                 "target": self._extract_target(raw, available_targets),
                 "message": raw[:500].strip()}
    if heuristic["action_type"]:
        return self._normalize(heuristic, available_targets), 0.6
    return self._fallback("unparseable"), 0.0
```

---

### `server/scenarios.py`

**What:** Scenario definition library and episode generator. Defines three scenario types (aligned, conflicted, hostile_acquisition) with different stakeholder configurations, constraint pools, relationship graphs, base tracks for initial stakeholder state, and event round triggers. Provides `generate_episode()` which creates a fully seeded episode with stakeholders, hidden constraints, relationship edges, and requested artifacts.

**Why:** The scenario library is the configuration layer that determines what kind of negotiation the LLM faces. Each scenario type creates a different distribution of challenges — different numbers of stakeholders, different conflict patterns, different hidden constraints, different urgency levels. The episode generator uses seeded random sampling so experiments are reproducible.

**How — scenario types:**
```python
SCENARIOS = {
    "aligned": {
        "max_rounds": 8, "days_to_deadline": 45,
        "stakeholder_count": (2, 3), "constraint_count": 1, "edge_count": (0, 1),
        "base_tracks": {"trust": (0.58, 0.70), "approval": (0.48, 0.62), ...},
        "roles": ["finance", "technical", "operations", "executive_sponsor"],
        "constraint_pool": ["budget_ceiling", "delivery_window"],
        "observability": "high",
    },
    "conflicted": {
        "max_rounds": 10, "days_to_deadline": 32,
        "stakeholder_count": (3, 4), "constraint_count": (1, 2), "edge_count": (1, 1),
        "base_tracks": {"trust": (0.40, 0.58), "approval": (0.35, 0.52), ...},
        "roles": ["finance", "technical", "procurement", "operations", "legal_compliance"],
        "constraint_pool": ["budget_ceiling", "delivery_window", "supplier_process"],
        "observability": "medium",
    },
    "hostile_acquisition": {
        "max_rounds": 10, "days_to_deadline": 22,
        "stakeholder_count": (4, 4), "constraint_count": 2, "edge_count": (1, 2),
        "base_tracks": {"trust": (0.35, 0.52), "approval": (0.30, 0.48), ...},
        "roles": ["finance", "technical", "legal_compliance", "executive_sponsor", "procurement"],
        "constraint_pool": ["budget_ceiling", "delivery_window", "compliance_addendum"],
        "event_round": (3, 4),
        "observability": "low",
    }
}
```

**Hidden constraints:** budget_ceiling (max price 185k), delivery_window (max 16 weeks), compliance_addendum (must include GDPR), supplier_process (must include named_support_lead)

---

## File Index

| File | Layer | Purpose |
|------|-------|---------|
| `models.py` | Data Models | Pydantic schemas for all domain objects |
| `deal_room/environment/dealroom_v3.py` | Environment | Core Gym-style env: reset, step, observation |
| `deal_room/environment/constants.py` | Environment | Reward weights, terminal payoffs, stage gates |
| `deal_room/environment/lookahead.py` | Environment | Lookahead simulation (minimax robustness) |
| `deal_room/environment/llm_client.py` | Environment | GPT-4o-mini client with retry/intervention |
| `deal_room/environment/stakeholder_llm.py` | Environment | LLM stakeholder response generation |
| `deal_room/environment/text_env.py` | Environment | Text rendering of observations |
| `deal_room/environment/prompts.py` | Environment | Prompt templates for stakeholder dialogue |
| `deal_room/committee/causal_graph.py` | Committee | Hidden causal graph, belief propagation |
| `deal_room/committee/belief_tracker.py` | Committee | Bayesian belief update |
| `deal_room/committee/deliberation_engine.py` | Committee | Committee vote, exec sponsor activation |
| `deal_room/rewards/utterance_scorer.py` | Rewards | 5-dim reward scoring |
| `deal_room/rewards/pareto_efficiency.py` | Rewards | Terminal reward and Pareto check |
| `deal_room/stakeholders/archetypes.py` | Stakeholders | Archetype profiles (tau, alpha, veto_power) |
| `deal_room/stakeholders/cvar_preferences.py` | Stakeholders | CVaR computation |
| `deal_room/curriculum/adaptive_generator.py` | Curriculum | Adaptive difficulty curriculum |
| `deal_room/training/grpo_trainer.py` | Training | GRPO training loop |
| `deal_room/training/ppo_trainer.py` | Training | PPO training loop |
| `server/deal_room_environment.py` | Server | HTTP API wrapping DealRoomV3 |
| `server/grader.py` | Server | Terminal scoring (CCIGrader) |
| `server/semantics.py` | Server | Intent/tone/artifact extraction |
| `server/claims.py` | Server | Commitment ledger and contradiction detection |
| `server/validator.py` | Server | Action parsing and normalization |
| `server/scenarios.py` | Server | Scenario definitions and episode generation |
| `server/stakeholders.py` | Server | Stakeholder response generation, tone scoring |