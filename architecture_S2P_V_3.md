# DealRoom S2P V3 Architecture Documentation

**Purpose**: This document describes the complete architecture of the DealRoom S2P (Source-to-Pay) Version 3 environment, which is designed to train and improve a Large Language Model (LLM) acting as a vendor negotiating B2B enterprise software deals with a multi-stakeholder buying committee. This is NOT a Neural Network agent training system—it is an LLM fine-tuning environment using GRPO/PPO algorithms to improve a language model's negotiation capabilities.

---

## 1. Overview

The DealRoom S2P V3 is a reinforcement learning environment where an LLM agent plays the role of a vendor sales executive navigating complex multi-stakeholder procurement negotiations. The environment models a realistic enterprise software buying committee with distinct personalities, risk tolerances, hidden constraints, and influence relationships. The training objective is to improve the LLM's ability to close deals by understanding stakeholder concerns, providing appropriate documentation, making strategic concessions, and avoiding veto triggers.

---

## 2. Core Environment: DealRoomV3

**File**: `deal_room_S2P/environment/dealroom_v3.py`

### What
`DealRoomV3` is the core OpenAI Gym-compatible RL environment that manages the complete negotiation simulation. It handles state transitions, committee deliberation, belief propagation, CVaR-based risk evaluation, and reward computation.

### Why
This is the central simulation engine that drives the entire training loop. It must accurately model real enterprise procurement dynamics where multiple stakeholders with conflicting priorities must reach consensus before a deal can close.

### How

**Key State Components**:
- `_state`: Full `DealRoomState` including episode tracking, stakeholder data (public/private), hidden constraints, relationship graph edges, commitment ledger, offer state, blocker list, stage/momentum tracking
- `_beliefs`: Dict mapping stakeholder_id to `BeliefDistribution` representing what each stakeholder believes about the vendor
- `_causal_graph`: `CausalGraph` representing committee influence network (hidden from agent)
- `_rng`: NumPy random generator for reproducibility

**Stakeholders** (STANDARD_STAKEHOLDERS): Legal, Finance, TechLead, Procurement, Operations, ExecSponsor

**Authority Hierarchy** (STANDARD_HIERARCHY): Legal/Finance=3, TechLead/Procurement/Operations=2, ExecSponsor=5

**Deal Stages**: evaluation → negotiation → legal_review → final_approval → closed

**Step Flow**:
```
1. _normalize_action(action) - Canonicalize target names
2. _apply_action_to_offer_state(action) - Update offer state
3. _update_deal_stage() - Check stage gates
4. _run_lookahead(action) - Optional lookahead simulation
5. _generate_stakeholder_responses() - Generate responses via LLM or templates
6. _update_noisy_engagement() - Apply observation noise
7. _evaluate_committee_risk() - Compute CVaR per stakeholder
8. _check_for_veto() - Determine if veto triggered
9. deliberation_engine.run() - Run committee deliberation
10. utterance_scorer.score() - Compute 5-dim reward
11. _build_observation() - Construct observation for agent
```

**Observation Signals** (5 observable):
1. `round_number` - Current negotiation round
2. `stakeholders` - Dict with engagement levels, messages, weak signals per stakeholder
3. `engagement_history` - Window of 5 engagement values per stakeholder
4. `cross_stakeholder_echoes` - List of {from, to, content} showing influence cascades
5. `veto_precursors` - Warning signals before veto fires

**Stage Gates**:
- `_check_gate_advance()`: Requires weighted_utility_sum >= 0.65 (STAGE_GATE_THETA_PASS)
- `_check_gate_regress()`: Triggers if weighted_utility_sum < 0.40 (STAGE_GATE_THETA_STALL)

**Lookahead**: If action contains `lookahead_request`, runs `LookaheadSimulator.simulate()` with n_hypotheses and depth parameters, returns minimax-robust prediction without advancing state.

---

## 3. Text Environment Wrapper

**File**: `deal_room_S2P/environment/text_env.py`

### What
`DealRoomTextEnv` is a TRL (Transformers Reinforcement Library) compatible text-in/text-out wrapper that bridges DealRoomV3 to work with GRPOTrainer's reward_function interface.

### Why
GRPO training in TRL expects text inputs/outputs. This wrapper converts DealRoomV3's structured observations and actions to text prompts and parses text actions back to structured DealRoomAction objects.

### How

**`reset()`**: Returns `build_situation_prompt(obs)` - natural language description of current negotiation state

**`step(action_text)`**:
1. Parse via `parse_action_text(action_text)` from prompts.py
2. Call `env.step(parsed_action)`
3. Return `build_situation_prompt(new_obs)` or terminal prompt

**`execute(prompt, completion)`**: TRL interface - runs one action, returns scalar reward

**`build_reward_function(task_id, seed, use_llm_stakeholders)`**: Factory returning TRL-compatible `reward_function(prompts, completions)`

---

## 4. Prompt System

**File**: `deal_room_S2P/environment/prompts.py`

### What
Handles LLM interaction - converting observations to text prompts and parsing LLM action outputs back into DealRoomAction objects.

### Why
The agent sees natural language, not structured data. This module provides the translation layer between environment state and LLM communication.

### How

**`build_situation_prompt(obs)`**: Converts DealRoomObservation to readable text including:
- Current round, deal stage, momentum
- Active blockers and days to deadline
- Veto warnings
- Committee member engagement levels and last messages
- Weak signals (hints about hidden concerns)
- Requested documents
- Available action syntax examples

**`parse_action_text(text)`**: Uses regex to extract structured actions:
- `send_document <target> <doc_type> [message]` - Sends a document
- `direct_message <target> <message> ###` - Sends a direct message
- `concession <target> <term>=<value> ###` - Offers a concession
- `group_proposal <message> ###` - Makes a group proposal
- `exec_escalation <message> ###` - Escalates to executive sponsor

**`get_template_response(stakeholder_name, stance)`**: Returns deterministic template responses per stakeholder across 4 stances (supportive, neutral, skeptical, hostile).

---

## 5. LLM Client

**File**: `deal_room_S2P/environment/llm_client.py`

### What
Thin wrapper around OpenAI GPT-4o-mini API with comprehensive error handling, retry logic, and interactive debugging.

### Why
Stakeholder responses and deliberation summaries require LLM calls. The client must handle network errors, rate limits, and provide fallback behavior.

### How

**`llm_call_text(prompt, call_type, temperature, context, allow_skip, policy, timeout)`**:
- Validates API key presence
- Creates OpenAI client per call
- Handles error categories: NETWORK_TIMEOUT, RATE_LIMIT_429, AUTH_INVALID_KEY, SERVER_500, EMPTY_RESPONSE, CONTENT_FILTER
- Retry policy: exponential backoff (max 3 retries, base 2s, max 60s)
- Returns `None` on skip/failure

**Interactive Debugging**: When `DEALROOM_LLM_INTERACTIVE=1`, pauses on failures with options: continue, wait N seconds, retry, skip, exit.

---

## 6. Stakeholder LLM

**File**: `deal_room_S2P/environment/stakeholder_llm.py`

### What
Generates stakeholder responses using GPT-4o-mini with template-based fallback.

### Why
Each stakeholder has distinct personality and concerns. The LLM generates contextually appropriate responses, falling back to templates when API unavailable.

### How

**`generate_stakeholder_response(stakeholder_name, context, role, stance)`**:
- If API key available: builds prompt via `build_stakeholder_prompt()` and calls GPT-4o-mini
- On failure: returns `get_template_response(stakeholder_name, stance)`

---

## 7. Lookahead Simulator

**File**: `deal_room_S2P/environment/lookahead.py`

### What
Implements lookahead simulation for minimax robustness - predicts stakeholder mental states and outcomes under uncertainty.

### Why
The agent can request lookahead to simulate "what if" scenarios before committing to an action. This enables strategic planning beyond immediate reward.

### How

**`LookaheadSimulator.simulate(action_draft, current_beliefs, n_hypotheses=2, depth=2)`**:
1. Generates 2 hypotheses (optimistic/pessimistic mental states)
2. Simulates each hypothesis
3. Returns worst-case result (minimizing predicted_goal_delta)

**BeliefDistribution**: Represents belief state with distribution over vendor types (competent, incompetent, trustworthy, deceptive, aligned, misaligned)

**Hypothesis Generation**:
- Optimistic: bumps competent/trustworthy/aligned probabilities
- Pessimistic: bumps incompetent/deceptive/misaligned probabilities

**Simulation**: Maps action type to response score and belief deltas

---

## 8. Causal Graph

**File**: `deal_room_S2P/committee/causal_graph.py`

### What
Causal graph-based belief propagation system modeling how stakeholder opinions influence each other through a committee network.

### Why
Real committees have influence hierarchies. When a vendor convinces one stakeholder, that belief should cascade to influenced parties. The causal graph models this hidden influence structure.

### How

**CausalGraph Dataclass**:
- `nodes`: List of stakeholder IDs
- `edges`: Dict mapping (source, dest) → influence weight
- `authority_weights`: Normalized authority scores per stakeholder

**`propagate_beliefs(initial_beliefs, graph, steps=3)`**:
1. Start with initial belief distributions per stakeholder
2. For each step, iteratively pass messages along edges
3. Use sigmoid gating (k_gate=10.0) to control update magnitude
4. Apply damping to prevent runaway beliefs
5. Return final propagated beliefs

**BeliefDistribution**: Models belief state with:
- Distribution over 6 vendor types: competent, incompetent, trustworthy, deceptive, aligned, misaligned
- `positive_mass()`: Sum of competent + trustworthy + aligned
- `negative_mass()`: Sum of incompetent + deceptive + misaligned

**`sample_graph(scenario_type, seed)`**: Generates random graph based on scenario:
- Authority nodes (authority >= 4) always create edges
- Intra-cluster edges boosted, cross-cluster edges penalized
- Edge weights from truncated normal distribution

**Graph Clusters**:
- cost: Finance, Procurement
- risk: Legal, Compliance
- implementation: TechLead, Operations
- authority: ExecSponsor

---

## 9. Belief Tracker

**File**: `deal_room_S2P/committee/belief_tracker.py`

### What
Bayesian belief updating based on observed vendor actions. Links specific actions to likelihoods of vendor types.

### Why
When a vendor performs an action (e.g., sends DPA document), stakeholders update their beliefs about the vendor based on whether that action is consistent with being competent/trustworthy.

### How

**ACTION_LIKELIHOODS**: Dict mapping action types to vendor-type-specific likelihoods
- `send_document(DPA)_proactive`: competent=0.85, trustworthy=0.80, aligned=0.80
- `send_document(security_cert)_proactive`: competent=0.80, trustworthy=0.75, aligned=0.75
- `direct_message_role_specific`: competent=0.70, trustworthy=0.65, aligned=0.70
- `default`: all 0.50

**`bayesian_update(prior_distribution, action, target_stakeholder, is_targeted)`**:
1. Look up likelihoods for action type
2. Apply damping (1.0 for targeted, 0.7 for non-targeted)
3. Compute posterior: `prior * dampened_likelihood`
4. Renormalize to valid probability distribution
5. Compute entropy-based confidence score

**Confidence**: `1.0 - (entropy / log2(6))` - higher when belief is more concentrated

---

## 10. Deliberation Engine

**File**: `deal_room_S2P/committee/deliberation_engine.py`

### What
Orchestrates committee deliberation dynamics - simulates how the stakeholder committee reacts to vendor actions, computes votes, and determines Executive Sponsor activation.

### Why
The committee doesn't instantly react. Deliberation takes time and produces committee-level decisions (approvals, blocks, abstentions) and determines if executive escalation is needed.

### How

**`run(beliefs, causal_graph, targeted_stakeholder, scenario_type)`**:
1. Call `propagate_beliefs()` to cascade belief changes
2. Compute committee vote via `_compute_committee_vote()`
3. Check Executive Sponsor activation via `_check_exec_sponsor_activation()`
4. Compute reactive silent period via `_compute_reactive_silent_period()`
5. Optionally generate LLM summary via `_generate_summary()`

**Committee Vote**:
- For each sub-agent: weighted_signal = positive_mass_delta * influence * authority_weight
- Thresholds: >= 0.15 = approve, <= -0.15 = block, else abstain

**ExecSponsor Activation Triggers**:
- 2+ members block
- `exec_escalation` action type
- >50% of non-abstain votes are blocks

**Silent Period**: Base period extended by conflict penalty and delta penalty when belief shift is low

---

## 11. CVaR Preferences

**File**: `deal_room_S2P/stakeholders/cvar_preferences.py`

### What
CVaR (Conditional Value at Risk) preference modeling for stakeholders. Models risk-averse utility functions where stakeholders care about tail losses, not just expected value.

### Why
Real stakeholders are risk-averse. A deal with 80% success rate but 20% catastrophic failure may be rejected by Legal even if expected value is positive. CVaR captures this by measuring expected loss in the worst-tail outcomes.

### How

**`compute_outcome_distribution(stakeholder_profile, deal_terms, n_samples=1000)`**: Monte Carlo simulation:
- Base success probability varies by domain
- Accounts for deal terms (DPA, security cert, price, timeline, liability cap)
- Returns array of outcomes in [0, 1]

**`compute_cvar(outcomes, alpha)`**: CVaR at alpha quantile:
1. Sort outcomes ascending
2. Cutoff at (1 - alpha) quantile
3. Return average of (1 - outcome) in tail

**`check_veto_trigger(cvar_loss, tau)`**: Returns True if cvar_loss > tau threshold

**StakeholderRiskProfile**:
- `alpha`: CVaR quantile threshold (tail sensitivity)
- `tau`: Veto threshold
- `lambda_risk`: Risk aversion weight
- `veto_power`: Boolean for veto authority

---

## 12. Stakeholder Archetypes

**File**: `deal_room_S2P/stakeholders/archetypes.py`

### What
Pre-configured stakeholder risk profiles (archetypes) for all committee members.

### Why
Each committee member has distinct risk tolerance, concerns, and veto power. These archetypes provide the personality configuration.

### How

**Six Archetypes**:

| Stakeholder | alpha | tau | lambda_risk | Veto | Top Uncertainty |
|-------------|-------|-----|-------------|------|-----------------|
| Legal | 0.95 | 0.10 | 0.70 | Yes | compliance_breach |
| Finance | 0.90 | 0.15 | 0.50 | Yes | payment_default |
| TechLead | 0.80 | 0.25 | 0.30 | No | implementation_failure |
| Procurement | 0.85 | 0.20 | 0.45 | No | contract_enforceability |
| Operations | 0.80 | 0.30 | 0.35 | No | operational_disruption |
| ExecSponsor | 0.70 | 0.40 | 0.25 | Yes | reputational_damage |

---

## 13. Utterance Scorer

**File**: `deal_room_S2P/rewards/utterance_scorer.py`

### What
Computes multi-dimensional utterance scores for RL reward signal. Scores vendor utterances across 5 dimensions without LLM calls.

### Why
The reward signal must capture multiple aspects of good negotiation: goal progress, trust building, information provision, risk management, and causal reasoning.

### How

**Five Dimensions**:
1. **goal (0.25)**: Weighted approval delta + blocker resolution + veto headroom
2. **trust (0.20)**: Positive mass delta for targeted stakeholders
3. **info (0.20)**: Entropy reduction across all stakeholders
4. **risk (0.20)**: CVaR improvement for risk-averse stakeholders
5. **causal (0.15)**: Betweenness centrality of targeted node

**`score(action, state_before, state_after, true_graph, lookahead_used)`**:
- Computes each dimension
- Applies tanh scaling to bound values in [0, 1]
- If lookahead_used: goal dimension reduced by LOOKAHEAD_COST (0.07)

**LOOKAHEAD_COST = 0.07**: Exact penalty for lookahead action

---

## 14. Pareto Efficiency

**File**: `deal_room_S2P/rewards/pareto_efficiency.py`

### What
Determines terminal rewards when deals reach endpoint states.

### Why
Not all terminal outcomes are equal. A deal closed is +1.0, but a veto has different penalties based on severity.

### How

**Terminal Rewards**:
- deal_closed: +1.0
- hard_veto: -1.0 (policy breach)
- soft_veto: -0.8 (CVaR veto)
- stage_regression: -0.75
- timeout: -0.5
- impasse: -0.75

**`check_pareto_optimality()`**: Determines if any stakeholder is non-dominated

**`compute_terminal_reward(terminal_outcome, cvar_losses, threshold)`**: Decision tree for final reward

---

## 15. Adaptive Curriculum Generator

**File**: `deal_room_S2P/curriculum/adaptive_generator.py`

### What
Adaptive curriculum learning that generates increasingly difficult scenarios based on agent performance and failure analysis.

### Why
Training should start easy (aligned scenarios) and progressively introduce harder scenarios (conflicted, hostile_acquisition) as the agent improves. This prevents both boredom (too easy) and frustration (too hard).

### How

**Failure Modes (F1-F7)**:
- F1: CVaR veto despite positive expected outcome
- F2: Trust collapse mid-episode
- F3: Failed graph inference
- F4: Timeout without coalition formation
- F5: Single-dimension reward hacking
- F6: Authority shift blindness
- F7: Stage gate regression

**CurriculumConfig**:
- easy_ratio: 0.20, frontier_ratio: 0.60, hard_ratio: 0.20
- stage_gate_window M=10, threshold theta_comp=0.65

**`analyze_failures(trajectories)`**: Detects failure modes, updates capability estimate using EMA

**`select_next_scenario(failure_analysis)`**: Picks difficulty based on capability (<0.5=easy, <0.75=frontier, else=hard)

---

## 16. GRPO Trainer

**File**: `deal_room_S2P/training/grpo_trainer.py`

### What
GRPO (Group Relative Policy Optimization) training harness with self-play episodes, multiple policy adapters, curriculum integration, and benchmarking.

### Why
GRPO is simpler than PPO (no value critic, no GAE) and uses group-relative advantages. This trains the LLM policy through trial and error.

### How

**PolicyAdapter Protocol**: Interface requiring `act()`, `update_from_batch()`, `state_dict()`, `load_state_dict()`

**Implementations**:
- `RandomPolicyAdapter`: Random action selection
- `HeuristicPolicyAdapter`: Hand-coded negotiation strategy
- `ModelPolicyAdapter`: Wraps arbitrary callable policy

**`run_self_play_episode()`**: Runs single episode with adaptive scenario generation

**`compute_group_relative_advantage(episode_rewards, group_rewards)`**: Normalizes episode rewards against group mean/std

**`run_training_loop()`**: Main training loop running batches of episodes, analyzing failures, updating policy

**`benchmark_policies()`**: Compares multiple policies across scenarios

---

## 17. PPO Trainer

**File**: `deal_room_S2P/training/ppo_trainer.py`

### What
Minimal PPO (Proximal Policy Optimization) implementation with GAE advantage estimation, clipped surrogate objective, and value function critic.

### Why
PPO provides more stable policy updates through clipped objectives, suitable for fine-grained policy tuning.

### How

**`compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95)`**: Generalized Advantage Estimation

**`compute_policy_loss(log_probs, old_log_probs, advantages, epsilon=0.2)`**: Clipped surrogate objective

**`SimpleValueCritic`**: 3-layer MLP (128 → 64 → 64 → 1) computing value estimates

**`collect_trajectory()`**: Collects episode by running policy in environment, computing GAE at end

**`update(trajectories)`**: Normalizes advantages, trains value critic, runs PPO updates

---

## 18. Models

**File**: `models.py`

### What
Pydantic data models for all DealRoom entities - actions, observations, states, and simulation types.

### Why
Type-safe data structures ensure contract between components and enable validation.

### How

**DealRoomAction**: action_type, target_ids, message, documents, proposed_terms, lookahead_request

**DealRoomObservation**: round_number, stakeholders, engagement_levels, weak_signals, cross_stakeholder_echoes, approval_path, deal_momentum, deal_stage, veto_precursors, active_blockers, days_to_deadline, committee_votes, engagement_history, metadata

**DealRoomState**: Full environment state with episode tracking, stakeholder data (public/private), hidden_constraints, relationship graph, commitment ledger, offer_state, blocker_list, stage_momentum, terminal_outcome

**LookaheadRequest**: Wraps action draft with n_hypotheses and depth

**SimulationResult**: Predicted responses, belief deltas, CVaR impact, graph information gain

---

## 19. Constants

**File**: `deal_room_S2P/environment/constants.py`

### What
Centralized configuration constants shared across the environment.

### How

**REWARD_WEIGHTS**: goal=0.25, trust=0.20, info=0.20, risk=0.20, causal=0.15

**TERMINAL_REWARDS_V2**: deal_closed=1.0, hard_veto=-1.0, soft_veto=-0.8, stage_regression=-0.75, timeout=-0.5

**DEFAULT_MAX_ROUNDS**: 10

**STAGE_GATE_THETA_PASS**: 0.65 (advance threshold)

**STAGE_GATE_THETA_STALL**: 0.40 (stall threshold)

**STEP_PENALTY**: -0.01

---

## 20. FastAPI Server

**File**: `server/app.py`

### What
Thin HTTP wrapper providing REST API for the DealRoom environment.

### Why
Enables remote training, web interface, and multi-client access.

### How

**DealRoomSessionPool**: Thread-safe session management with TTL and max sessions

**Endpoints**:
- `GET /health`: Service health
- `GET /metadata`: Service metadata
- `POST /reset`: Initialize session, returns observation
- `POST /step`: Execute action, returns (obs, reward, done, info)
- `GET /state`: Get current state

**Gradio UI**: Mounted at `/__gradio_ui__/` for web interface

---

## 21. Training Notebooks

**Files**: `*.ipynb`

### What
Jupyter notebooks demonstrating GRPO/PPO training workflows.

### How

**Key Notebooks**:

1. **dealroom_v3_grpo_training.ipynb**: GRPO training on DealRoomV3 with QLoRA-finetuned Qwen2.5-3B
2. **grpo_training_S2P.ipynb**: kube-sre-gym pattern with 5-dim reward decomposition
3. **notebooks/06_deal_room_training_v3_7_colab.ipynb**: PPO with adaptive curriculum and lookahead
4. **notebooks/05_llm_training.ipynb**: TRL GRPOTrainer integration

**Common Pattern**:
1. Load Qwen2.5-3B-Instruct with QLoRA adapters
2. Build reward function that parses LLM output and executes in environment
3. Run GRPO/PPO training for 50 steps
4. Evaluate on held-out scenarios

---

## 22. System Integration

```
LLM Agent (Qwen2.5-3B-Instruct + QLoRA)
     │
     ▼
DealRoomTextEnv (TRL wrapper)
     │
     ▼
DealRoomV3 (Core Environment)
     │
     ├──► prompts.py (observation→text, text→action)
     │
     ├──► stakeholder_llm.py → llm_client.py (GPT-4o-mini for responses)
     │
     ├──► lookahead.py (minimax simulation)
     │
     ├──► causal_graph.py + belief_tracker.py (belief propagation)
     │
     ├──► deliberation_engine.py (committee dynamics)
     │
     ├──► cvar_preferences.py + archetypes.py (risk evaluation)
     │
     ├──► utterance_scorer.py (5-dim reward)
     │
     └──► pareto_efficiency.py (terminal reward)
     │
     ▼
GRPO/PPO Trainer
     │
     ▼
Updated LLM Policy
```

---

## 23. Environment Configuration

**Scenarios**:
- `aligned`: Cooperative stakeholders, easy to close
- `conflicted`: Mixed interests, moderate difficulty
- `hostile_acquisition`: Adversarial committee, hardest scenarios

**Research Properties** (P1-P12):
- P1: G (causal graph) hidden from agent
- P2: Reset regenerates different G
- P3: CVaR veto despite EU > 0
- P4: 5 reward dimensions are independent
- P5: Lookahead cost exactly 0.07
- P6: Observation noise not cancellable
- P7: Cross-stakeholder echoes present
- P8: Weak signals present
- P9: Causal score varies with target position
- P10: Every reset produces unique G
- P11: All scenarios complete without crash
- P12: Training infrastructure imports work

---

## 24. Data Flow Summary

**Episode Flow**:
1. Reset → sample causal graph → initialize beliefs → return observation
2. Agent sees natural language situation → generates action text
3. Parse action text → apply to environment → update beliefs → deliberation
4. Score reward (5-dim) → return to agent
5. Repeat until terminal (deal_closed, veto, timeout)
6. Apply terminal reward

**Training Flow**:
1. Generate batch of episodes with current policy
2. Compute group-relative advantages
3. Update policy via GRPO/PPO
4. Analyze failures → adapt curriculum
5. Repeat

---

## 25. Key Design Decisions

1. **LLM, Not Neural Agent**: The environment is designed for LLM fine-tuning (Qwen, Llama), not neural network RL agents. Actions are text, observations are text.

2. **CVaR Over Expected Value**: Stakeholders use CVaR (Conditional Value at Risk) for veto decisions, modeling real risk-averse procurement officers.

3. **Hidden Causal Graph**: The committee influence structure (G) is hidden, requiring the LLM to infer it through observation.

4. **Noisy Observations**: Engagement levels contain noise that cannot be perfectly cancelled, preventing overfitting.

5. **5-Dimensional Reward**: Single scalar rewards are insufficient for complex negotiations; 5 dimensions capture goal, trust, info, risk, causal aspects.

6. **Lookahead with Cost**: The agent can request lookahead simulation at a fixed cost (0.07), enabling strategic planning without breaking the RL loop.

7. **Adaptive Curriculum**: Training starts easy and progressively harder based on detected failure modes.
