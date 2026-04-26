# DealRoom v3 — Testing Documentation
**Version:** S2P_V_1
**Purpose:** Document all test files for the DealRoom v3 codebase
**Scope:** All `.py` test files in `tests/unit/` and `tests/v3/`
**Separation:** Unit tests (standalone, no server required) vs. v3 integration tests (require HTTP server)

---

## Overview

The test suite has two distinct layers:

1. **Unit tests** (`tests/unit/`) — Run directly without any server. Each test file imports the modules it tests directly. These test pure functions, dataclasses, and class logic in isolation.

2. **v3 integration tests** (`tests/v3/`) — Run against the live HTTP server at `http://127.0.0.1:7860`. These test end-to-end behavior including API endpoints, action validation, reward computation, causal inference, veto logic, and episode isolation. They require the server to be started first.

---

## Unit Tests

All unit tests live in `tests/unit/` and use `pytest`.

### `tests/unit/test_deliberation_engine.py`

**What:** Tests the two-layer deliberation engine: Layer 1 (pure belief propagation) and Layer 2 (optional MiniMax summary generation). Tests committee vote computation, exec sponsor activation, and silent period calculation.

**Why:** The deliberation engine is a core architectural component. It determines how stakeholder beliefs propagate through the causal graph after each action, when the ExecutiveSponsor is activated, and how long the silent period lasts before the next targeted action is appropriate. These behaviors must be deterministic and correct.

**How:**
```python
# Test belief propagation follows graph structure
# A->B->C chain, B should change more than C when A gets positive delta
result = propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps=3)
b_change = abs(result["B"].positive_mass() - beliefs_before["B"].positive_mass())
c_change = abs(result["C"].positive_mass() - beliefs_before["C"].positive_mass())
assert b_change > c_change  # B is closer to A in graph

# Test exec sponsor activation when 2+ blocks
blocks = ["Legal", "Finance"]
if len(blocks) >= 2:
    self._exec_sponsor_active = True  # triggers veto_cast

# Test silent period = base + conflict_penalty + delta_penalty
silent_duration = base_silent + int(conflict_ratio * 3) + int(avg_delta * 4)
```

**Key checks:**
- `DELIBERATION_STEPS` mapping: aligned=3, conflicted=3, hostile_acquisition=4
- `propagation_deltas` records delta for each stakeholder
- No self-edges in causal graph
- Damping prevents runaway propagation (10 steps with 0.95^step damping)
- Layer 2 returns string or empty string on failure (no crash)

---

### `tests/unit/test_utterance_scorer.py`

**What:** Tests the 5-dimension `UtteranceScorer`: goal, trust, info, risk, causal. Tests that `LOOKAHEAD_COST = 0.07` is correctly subtracted from the goal dimension when lookahead is used. Tests that all scores stay in [0, 1] range, that causal scoring is deterministic, and that `weighted_sum()` produces correct scalar reward.

**Why:** The reward signal is the primary training signal for the LLM. If the scorer produces out-of-range values, the RL training will be unstable. If the lookahead cost is not correctly applied, the LLM will not learn to use lookahead efficiently.

**How:**
```python
# Test lookahead cost subtraction
score_without = scorer.score(action, state_before, state_after, graph, lookahead_used=False)
score_with = scorer.score(action, state_before, state_after, graph, lookahead_used=True)
assert abs(score_without.goal - score_with.goal - LOOKAHEAD_COST) < 1e-6

# Test all dimensions in [0, 1] for 20 random inputs
for i in range(20):
    score = scorer.score(action, state_before, state_after, graph, False)
    assert 0.0 <= score.goal <= 1.0
    assert 0.0 <= score.trust <= 1.0
    ...

# Test causal score deterministic
score1 = scorer.score(action1, state, state, graph, False)
score2 = scorer.score(action2, state, state, graph, False)
assert score1.causal == score2.causal  # same graph + same target

# Test weighted_sum
score = UtteranceScore(goal=0.8, trust=0.6, info=0.7, risk=0.5, causal=0.4)
result = score.weighted_sum(REWARD_WEIGHTS)  # REWARD_WEIGHTS = {goal:0.25, trust:0.20, ...}
expected = 0.25*0.8 + 0.20*0.6 + 0.20*0.7 + 0.20*0.5 + 0.15*0.4
```

**Key checks:**
- `LOOKAHEAD_COST == 0.07` (matches `lookahead.py` and `utterance_scorer.py`)
- Goal difference exactly equals LOOKAHEAD_COST when lookahead toggled
- All five dimensions bounded in [0, 1]
- Causal score deterministic: same graph+target → same causal score
- `weighted_sum` produces correct weighted average per REWARD_WEIGHTS

---

### `tests/unit/test_curriculum.py`

**What:** Tests the `AdaptiveCurriculumGenerator` which adjusts scenario difficulty based on the LLM's performance history. Tests failure mode detection (F1: CVaR veto, F2: stage regression, F3: causal awareness stagnation, F4: trust erosion, F5: info asymmetry, F6: efficiency collapse), difficulty scoring, and the full analyze-and-generate cycle.

**Why:** The curriculum is what allows the LLM to start with easy scenarios and gradually tackle harder ones. If failure detection is wrong, the LLM will get stuck or overwhelmed. If scenario generation is broken, training will not progress properly.

**How:**
```python
# Test failure mode detection
traj = MockTrajectory(terminal_outcome="veto", rewards=[[0.6,0.6,0.6,0.2,0.5]]*10)
failures = generator._detect_failures(traj)  # Low r^risk + veto → F1

# Test curriculum config ratios
config = CurriculumConfig()
assert config.easy_ratio == 0.20
assert config.frontier_ratio == 0.60
assert config.hard_ratio == 0.20  # Must sum to 1.0

# Test analyze + generate cycle
analysis = generator.analyze_failures(trajectories)
scenario = generator.generate_adaptive_scenario(analysis)
assert scenario["task_id"] in ["aligned", "conflicted", "hostile_acquisition"]
```

**Key checks:**
- All 7 failure mode descriptions defined (F1-F7)
- Scenario pool has easy/frontier/hard difficulties
- `create_curriculum_generator()` returns `AdaptiveCurriculumGenerator` instance
- `analyze_failures([])` handles empty trajectory list
- Capability-based selection works for both low (0.2) and high (0.9) estimates

---

### `tests/unit/test_cvar_preferences.py`

**What:** Tests CVaR computation and the veto trigger mechanism. The central test validates the "silent veto" claim: Legal veto fires even when expected utility is positive, because CVaR (tail-risk) exceeds tau. Also tests risk profile ordering (Legal > Procurement > Finance > Operations > TechLead > ExecSponsor), observable signals, and archetype parameter locking.

**Why:** CVaR-based veto is the primary deal-termination mechanism. If CVaR doesn't fire correctly, the environment cannot properly simulate stakeholder risk sensitivity. The silent veto is a key behavioral property — it means the LLM can't just optimize for expected value, it must also manage tail-risk.

**How:**
```python
# The central "silent veto" test
legal_profile = get_archetype("Legal")
deal_terms = {"price": 100000, "timeline_weeks": 12, "has_dpa": False, "has_security_cert": False}
expected_utility, cvar_loss = evaluate_deal(deal_terms, legal_profile, rng, n_samples=1000)
veto_triggered = check_veto_trigger(cvar_loss, legal_profile)

assert expected_utility > 0  # E[u] > 0
assert veto_triggered  # but CVaR > tau → veto fires

# This is the "silent veto": utility looks positive but tail-risk is unacceptable

# Test CVaR monotonicity: adding documentation must reduce CVaR
cvar_no_docs > cvar_dpa >= cvar_with_both_docs

# Test risk profile ordering for same deal terms
cvar_values = {}
for archetype_name in ["Legal", "Procurement", "Finance", "Operations", "TechLead", "ExecSponsor"]:
    profile = get_archetype(archetype_name)
    _, cvar = evaluate_deal(deal_terms, profile, rng, n_samples=500)
    cvar_values[archetype_name] = cvar
# Legal CVaR should be highest, ExecSponsor lowest
```

**Key checks:**
- CVaR veto fires despite positive expected utility (the "silent veto" property)
- Full documentation (DPA + security cert) reduces CVaR by > 0.05 vs no docs
- CVaR decreases monotonically as documentation is added
- CVaR ordering: Legal > Procurement > Finance > Operations > TechLead > ExecSponsor
- Archetype parameter locking: Legal alpha=0.95, tau=0.10, lambda_risk=0.70
- `compute_cvar` on uniform [0,1] distribution: alpha=0.50 < alpha=0.95 (lower alpha = higher CVaR)
- Empty outcomes array returns 0.0

---

### `tests/unit/test_causal_graph.py`

**What:** Tests the causal graph structure, belief distribution operations, propagation logic, betweenness centrality, and graph identifiability theorem. The identifiability test validates that targeting different stakeholders in the same graph produces measurably different behavioral signatures (L1 distance > 0.02).

**Why:** The causal graph is the hidden backbone of the environment. The LLM must be able to infer it from observation signals. If two different targets produce indistinguishable signatures, the LLM cannot learn to distinguish their effects. The identifiability theorem guarantees that the environment provides enough signal for the LLM to learn the graph structure.

**How:**
```python
# Test identifiability: targeting A vs B in same graph → different signatures
for target in STANDARD_STAKEHOLDERS:
    sig = compute_behavioral_signature(graph, target, belief_delta=0.30, n_steps=3)
    signatures[target] = sig

# L1 distance between any two signatures must be > 0.02
for i, s1 in enumerate(STANDARD_STAKEHOLDERS):
    for j, s2 in enumerate(STANDARD_STAKEHOLDERS):
        if i >= j: continue
        l1_dist = sum(abs(signatures[s1].get(n,0) - signatures[s2].get(n,0)) for n in all_nodes)
        assert l1_dist > 0.02  # signatures must be distinguishable

# Test propagation direction: A->B, no A->C
# B should receive positive influence from A, C should not
graph = CausalGraph(nodes=["A","B","C"], edges={("A","B"):0.7, ("A","C"):0.0}, ...)
beliefs_after["A"] = apply_positive_delta(beliefs_before["A"], 0.4)
result = propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps=3)
assert result["B"].positive_mass() > beliefs_before["B"].positive_mass() + 0.05

# Test damping prevents runaway in dense graph (5 steps, 0.95^step damping)
for sid in ["A","B","C","D"]:
    pm = result[sid].positive_mass()
    assert 0.0 < pm < 1.0  # must stay in valid range
```

**Key checks:**
- No self-edges in any sampled graph
- All edge weights in [0.05, 0.95]
- Authority weights sum to 1.0
- `positive_mass()` = competent + trustworthy + aligned
- `negative_mass()` = incompetent + deceptive + misaligned
- `copy()` creates independent deep copy
- `create_neutral_beliefs` creates uniform distribution over 6 vendor types
- Propagation damping prevents runaway: 5 steps in dense graph → all beliefs in (0, 1)
- Betweenness centrality: hub node >= leaf nodes
- Isolated node has zero centrality
- Behavioral signatures pairwise distinguishable (L1 > 0.02)
- Aligned scenario produces sparser graphs than conflicted

---

### `tests/unit/test_grader.py`

**What:** Tests the `CCIGrader` terminal scoring. Validates that infeasible deals score MIN_SCORE (0.01), feasible deals score in (0, 1), unresolved constraints trigger MIN_SCORE, and relationship damage (permanent marks) reduces the score.

**Why:** The grader provides a human-aligned terminal evaluation. It should be strict enough to penalize bad deals but generous enough to reward good ones. The score range (0.01–0.99) ensures the RL algorithm can distinguish quality differences.

**How:**
```python
def make_state(feasible=True):
    return DealRoomState(
        deal_closed=True,
        stakeholder_private={
            "finance": {"trust": 0.8, "approval": 0.8, "perceived_fit": 0.8,
                        "private_resistance": 0.3, "mandatory": True, "veto_power": True, "permanent_marks": []},
            "technical": {"trust": 0.75, "approval": 0.78, "perceived_fit": 0.76,
                          "private_resistance": 0.3, "mandatory": True, "veto_power": False, "permanent_marks": []},
        },
        hidden_constraints={"budget_ceiling": {"resolved": feasible}},
        feasibility_state={"is_feasible": feasible, "violations": [] if feasible else ["price"]},
        round_number=4, max_rounds=10,
    )

assert CCIGrader.compute(make_state(feasible=False)) == CCIGrader.MIN_SCORE
assert 0.0 < CCIGrader.compute(make_state(feasible=True)) < 1.0

# Unresolved constraint
state = make_state(feasible=True)
state.hidden_constraints["budget_ceiling"]["resolved"] = False
assert CCIGrader.compute(state) == CCIGrader.MIN_SCORE

# Relationship damage
damaged.stakeholder_private["finance"]["permanent_marks"] = ["premature_close", "semantic_contradiction"]
assert CCIGrader.compute(damaged) < CCIGrader.compute(clean)
```

**Key checks:**
- Infeasible close → MIN_SCORE (0.01)
- Feasible close → score in (0, 1)
- Unresolved constraint → MIN_SCORE
- Score strictly inside open interval (0.0, 1.0)
- Relationship damage penalizes score (permanent marks reduce trust/approval durability)

---

## v3 Integration Tests

All v3 tests live in `tests/v3/` and require a running server at `http://127.0.0.1:7860`. The server must be started before running these tests.

### `tests/v3/conftest.py`

**What:** Shared pytest fixtures for v3 integration tests. Provides `base_url` (defaults to `http://127.0.0.1:7860`), `reset_env()` fixture that calls `POST /reset` with task_id and seed, and `validate_action()` helper.

**Why:** Avoids duplicating the server URL and reset logic across all v3 test files. Centralizes the HTTP interaction pattern.

---

### `tests/v3/test_01_schema_validation.py`

**What:** Tests the action schema validation API endpoint. Validates that required fields are present, hidden fields are not leaked, field types are correct, and the action schema is properly enforced.

**Why:** The LLM produces actions as JSON. If the schema validation is too strict, valid actions will be rejected. If it's too loose, malformed actions will be accepted and cause errors downstream. 11 validation checks cover the schema contract.

**Key checks:**
- Required fields present (action_type, target_ids, message)
- Hidden fields not leaked (graph structure, private beliefs, CVaR thresholds)
- Field types correct (action_type is string, target_ids is list)
- Action schema enforced (valid action types, valid targets)

---

### `tests/v3/test_02_reward_integrity.py`

**What:** Tests the reward computation API endpoint. Validates that reward format is correct, lookahead cost (0.07) is correctly deducted, reward computation is deterministic for identical inputs, and reward variance is within acceptable bounds across runs.

**Why:** The reward signal drives RL training. If reward computation is non-deterministic, training will be unstable. If lookahead cost is not applied, the LLM will overuse lookahead. 10 checks cover reward integrity.

**Key checks:**
- Reward is a float in reasonable range
- Lookahead cost deducted from goal dimension
- Determinism: same action, same state → same reward (within floating point tolerance)
- Variance across runs below threshold (seed-based determinism)

---

### `tests/v3/test_03_causal_inference.py`

**What:** Tests the causal inference API. Validates that belief propagation follows the causal graph structure, cross-stakeholder echoes are generated (70% probability), engagement noise (sigma > 0) is applied, and the graph structure affects belief updates.

**Why:** Causal inference is what separates DealRoom v3 from simpler negotiation simulators. The LLM must be able to infer the hidden graph structure from observation signals. If propagation is broken, the environment loses its causal structure. 8 checks cover propagation, echoes, noise, and graph structure.

**Key checks:**
- Positive delta to targeted stakeholder propagates to neighbors in graph
- Non-targeted stakeholders generate cross-echoes at 70% probability
- Engagement noise sigma > 0 (observation is not deterministic)
- Different graph configurations produce different belief update patterns

---

### `tests/v3/test_04_cvar_veto.py`

**What:** Tests the CVaR veto mechanism end-to-end. Validates that the veto precursor system works (CVaR > tau * 0.70 triggers warning), that CVaR veto fires when conditions are met, and that terminal outcomes correctly reflect veto events.

**Why:** CVaR veto is the primary deal-termination mechanism. If it doesn't fire correctly, the environment cannot properly terminate episodes. 8 checks cover the full veto lifecycle.

**Key checks:**
- Veto precursor appears when CVaR > tau * 0.70 for 2 consecutive rounds
- CVaR veto fires when CVaR > tau for 2 consecutive rounds
- Terminal reward correctly computed for veto outcomes
- Hard veto (missing DPA/security cert) vs soft veto (CVaR) distinction

---

### `tests/v3/test_05_episode_isolation.py`

**What:** Tests that episodes are properly isolated from each other. Validates seed-based isolation, proper reset behavior, and that round_number resets correctly.

**Why:** In distributed training, multiple workers run episodes in parallel. If episodes share state, training will be corrupted. 8 checks cover episode independence.

**Key checks:**
- Same seed produces same initial state
- Different seeds produce different initial states
- Reset properly clears all episode state
- Round number resets to 0 on reset

---

### `tests/v3/test_06_probabilistic_signals.py`

**What:** Tests probabilistic signal generation. Validates that weak signals are generated (high_engagement, low_engagement, improving_engagement, declining_engagement, high_uncertainty), that cross-stakeholder echoes fire at the expected rate (70%), and that engagement noise affects observations.

**Why:** The LLM operates in a noisy, partially observable environment. Weak signals and echoes are the primary channels through which the LLM infers the hidden causal graph structure. 7 checks cover signal generation and firing rates.

**Key checks:**
- Weak signals generated for all stakeholder states
- Echo firing rate within statistical bounds of 70%
- Engagement noise sigma > 0 confirmed
- Signal generation is stochastic but bounded

---

### `tests/v3/test_07_causal_graph.py`

**What:** Tests causal graph API endpoints. Validates graph sampling, edge weight ranges, authority normalization, belief propagation API, and the behavioral signature API for graph identifiability.

**Why:** The causal graph is the hidden structure that the LLM must learn to navigate. Testing the graph API ensures the environment correctly generates and exposes graph-related data in a way the LLM can use for inference. 9 checks cover graph construction and propagation.

**Key checks:**
- Graph sampling produces valid structure (no self-edges, weights in [0.05, 0.95])
- Authority weights sum to 1.0
- Belief propagation API works correctly
- Behavioral signatures are pairwise distinguishable (L1 > threshold)

---

### `tests/v3/test_08_cvar_preferences.py`

**What:** Tests CVaR preference API endpoints. Validates that different stakeholder archetypes produce different CVaR values for the same deal terms, that documentation reduces CVaR, and that the observable signals API returns correct risk signals.

**Why:** This is the server-side version of the unit tests for `cvar_preferences.py`. It tests the API layer rather than the Python module directly. 6 checks cover archetype ordering and documentation effects.

**Key checks:**
- CVaR ordering across archetypes (Legal highest, ExecSponsor lowest)
- DPA reduces CVaR for Legal stakeholder
- Security cert reduces CVaR for Legal stakeholder
- Observable signals API returns compliance_concern and risk_tolerance

---

### `tests/v3/test_09_full_episode_e2e.py`

**What:** Tests complete end-to-end episodes from reset through terminal state. Validates that full episodes run correctly, that strategy comparison works (comparing two different action sequences), and that terminal reward is correctly computed for all terminal outcomes.

**Why:** This is the most important integration test — it validates that the entire system works together from start to finish. 8 checks cover complete episode runs and terminal payoff computation.

**Key checks:**
- Full episode from reset to terminal state completes without errors
- Different action strategies produce different terminal outcomes
- Terminal reward for deal_closed = +1.0
- Terminal reward for veto = -1.0 (hard) or -0.8 (soft)
- Terminal reward for max_rounds = -0.5

---

### `tests/v3/test_10_training_infrastructure.py`

**What:** Tests the training infrastructure integration. Validates that the GRPO trainer can be imported, the PPO trainer can be imported, the colab notebook runs without errors, and the training metrics API works.

**Why:** Training infrastructure is what turns the environment into a training system. If the trainers can't be imported or the notebook doesn't run, the system can't be used for RL training. 6 checks cover imports and basic training API.

**Key checks:**
- `GRPOTrainer` and `PPOTrainer` import successfully
- `TrainingMetrics` dataclass works correctly
- Colab notebook is runnable (or at least importable)
- Training metrics API returns valid data

---

### `tests/v3/test_11_research_properties.py`

**What:** Tests the 12 research properties (P1–P12) that define the scientific claims of DealRoom v3. These include the silent veto property, graph identifiability, causal awareness reward, CVaR monotonicity with documentation, episode isolation, engagement noise positivity, echo firing rate, stage gate behavior, lookahead cost accuracy, and Pareto optimality detection.

**Why:** These are the core scientific claims that distinguish DealRoom v3 from a simple negotiation game. Each property must be verifiable by tests. If any property fails, the research claim is undermined. 12 checks (P1–P12) cover the full research agenda.

**Key research properties:**
- P1: Silent veto — CVaR veto fires despite positive expected utility
- P2: Graph identifiability — behavioral signatures are pairwise distinguishable
- P3: Causal awareness reward captures graph-informed actions
- P4: CVaR monotonically decreases with documentation
- P5: Episode isolation — same seed = same episode
- P6: Engagement noise sigma > 0 (noisy observations)
- P7: Cross-stakeholder echo firing rate ≈ 70%
- P8: Stage gates advance/regress based on weighted_utility_sum
- P9: Lookahead cost exactly 0.07
- P10: Pareto front detection works
- P11: Committee vote computation is correct
- P12: Exec sponsor activates on 2+ blocks or escalation

---

## Test File Index

### Unit Tests (`tests/unit/`)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_deliberation_engine.py` | DeliberationResult structure, DELIBERATION_STEPS mapping, belief propagation, propagation_deltas recording, no-summary on no-targets, Layer 1 pure propagation, Layer 2 string-or-empty, single-step deliberation, many-steps damping, propagation follows graph structure, no-edges no-propagation | Committee deliberation, exec sponsor activation, silent period |
| `test_utterance_scorer.py` | LOOKAHEAD_COST value, goal subtraction equals cost, all dims in [0,1] for 20 random inputs, causal deterministic, UtteranceScore defaults, weighted_sum accuracy, to_dict | 5-dim reward scoring |
| `test_curriculum.py` | FailureAnalysis defaults, CurriculumConfig ratios, generator init, select_next_scenario, generate_adaptive_scenario, analyze_failures empty, failure detection F1/F3, failure mode descriptions, scenario pool difficulties, capability-based selection, scenario structure, full analyze+generate cycle, create_curriculum_generator | Adaptive difficulty curriculum |
| `test_cvar_preferences.py` | Silent veto (E[u]>0 but veto fires), full docs reduce CVaR, monotonic decrease with docs, CVaR on uniform distribution, empty outcomes, risk profile ordering, lambda_risk impact, veto fires above tau, veto regardless of expected utility, outcome distribution domain, observable signals for Legal, risk_tolerance ordering, all archetypes defined, archetype values locked, archetype utility weights | CVaR computation, veto trigger, archetype profiles |
| `test_causal_graph.py` | sample_graph returns valid, no self-edges, weight range [0.05, 0.95], authority invariant (ExecSponsor outgoing edges), authority weights sum to 1, positive_mass, negative_mass, copy independence, create_neutral_beliefs, propagation direction, damping prevents runaway, apply_belief_delta positive/negative/with-damping, betweenness centrality hub>leaf, isolated node zero centrality, behavioral signature distinct, graph identifiability statistical, engagement level range, engagement level neutral is zero, aligned sparser than conflicted | Causal graph structure, belief propagation, graph identifiability |
| `test_grader.py` | Zero for infeasible, positive for feasible, zero when constraint unresolved, score strictly inside (0,1), penalizes relationship damage | Terminal scoring |

### v3 Integration Tests (`tests/v3/`)

| File | Checks | Server required |
|------|--------|-----------------|
| `test_01_schema_validation.py` | 11 schema checks | Yes |
| `test_02_reward_integrity.py` | 10 reward checks | Yes |
| `test_03_causal_inference.py` | 8 causal inference checks | Yes |
| `test_04_cvar_veto.py` | 8 veto lifecycle checks | Yes |
| `test_05_episode_isolation.py` | 8 episode isolation checks | Yes |
| `test_06_probabilistic_signals.py` | 7 probabilistic signal checks | Yes |
| `test_07_causal_graph.py` | 9 graph API checks | Yes |
| `test_08_cvar_preferences.py` | 6 CVaR API checks | Yes |
| `test_09_full_episode_e2e.py` | 8 end-to-end episode checks | Yes |
| `test_10_training_infrastructure.py` | 6 training infra checks | Yes |
| `test_11_research_properties.py` | 12 research property checks (P1-P12) | Yes |

---

## Running the Tests

### Unit tests (no server required):
```bash
cd /Users/akshaypulla/Documents/deal_room_S2P
pytest tests/unit/ -v
```

### v3 integration tests (server required):
```bash
# Terminal 1: Start the server
export OPENAI_API_KEY=your_key_here  # optional, for LLM summaries
python -m server.deal_room_environment  # or whatever the startup command is

# Terminal 2: Run integration tests
cd /Users/akshaypulla/Documents/deal_room_S2P
pytest tests/v3/ -v
```

### All tests:
```bash
pytest tests/ -v
```