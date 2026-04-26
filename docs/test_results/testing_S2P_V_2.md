# DealRoom S2P V2 Testing Documentation

## Overview

**Purpose**: This document describes the testing strategy, test cases, and validation approach for DealRoom S2P V2 environment. The tests are designed to ensure the environment correctly implements the research-grade RL desiderata for training LLMs.

**Testing Philosophy**: Tests focus on verifying correct behavior rather than implementation details. All tests validate what the system does and why it does it, ensuring the environment is suitable for LLM training.

---

## 1. Test Structure

### 1.1 Test File Organization

```
tests/
├── v3/                          # Integration/container tests
│   ├── test_00_environment_setup.py
│   ├── test_01_schema_validation.py
│   ├── test_02_reward_integrity.py
│   ├── test_03_causal_inference.py
│   ├── test_04_cvar_veto.py
│   ├── test_05_episode_isolation.py
│   ├── test_06_probabilistic_signals.py
│   ├── test_07_causal_graph.py
│   ├── test_08_cvar_preferences.py
│   ├── test_09_full_episode_e2e.py
│   ├── test_10_training_infrastructure.py
│   ├── test_11_research_properties.py
│   └── test_P0_comprehensive.py
├── unit/                        # Unit tests
│   ├── test_belief_tracker.py
│   ├── test_curriculum.py
│   ├── test_utterance_scorer.py
│   └── ...
├── integration/
│   ├── test_api_sessions.py
│   ├── test_web_ui.py
│   └── test_end_to_end.py
└── conftest.py                  # Shared fixtures
```

---

## 2. Test Categories

### 2.1 Section 0: Environment Setup (`test_00_environment_setup.py`)

**What**: Validates the testing environment and runtime dependencies.

**Why**: Ensures all prerequisites are available before running functional tests.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_0_1_python_deps` | Checks `requests`, `numpy`, `python-dotenv` installed | Required dependencies must be present |
| `test_0_2_api_keys_configured` | Reports API key status (optional) | Graceful degradation when API unavailable |
| `test_0_3_docker_container_running` | Verifies Docker container running | Tests require containerized API |
| `test_0_4_server_endpoints_responsive` | Tests `/health`, `/metadata`, `/reset`, `/step` endpoints | API must respond correctly |
| `test_0_5_llm_client_imports` | Imports `llm_client` module | Ensures LLM client available |

---

### 2.2 Section 1: Schema Validation (`test_01_schema_validation.py`)

**What**: Validates observation and action schema correctness.

**Why**: The LLM receives observations in a specific format; schema violations would break training.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_1_1_all_required_fields_present` | Checks 18 required fields in observation | Missing fields break LLM parsing |
| `test_1_2_no_hidden_fields_exposed` | Verifies hidden fields (G, tau, etc.) not in observation | Research integrity - agent must not see internal state |
| `test_1_3_field_types_correct` | Validates each field has correct type (int, dict, list, etc.) | Type errors cause training failures |
| `test_1_4_engagement_history_window_size` | Checks history has ≥5 entries | Sliding window must be maintained |
| `test_1_5_engagement_level_delta_single_float` | Ensures delta is numeric, not dict | Must be single value for learning |
| `test_1_6_cross_stakeholder_echoes_is_list` | Validates echo structure | Correct format for propagation signals |
| `test_1_7_stakeholder_messages_populated_after_step` | Verifies messages appear after action | Committee responses must be generated |
| `test_1_8_action_schema_accepts_lookahead` | Tests lookahead action accepted | Forward simulation must work |
| `test_1_9_approval_path_progress_structure` | Validates band structure | Approval tracking must be correct |
| `test_1_10_deal_stage_valid_transitions` | Checks round increment and stage values | Progression system must work |
| `test_1_11_documents_field_format` | Validates document attachment format | Document sending must be structured |

**Hidden Fields List** (must NOT be exposed):
```python
HIDDEN_FIELDS = {
    "G", "causal_graph", "graph", "true_beliefs",
    "belief_distributions", "belief_state", "B_i", "V_i",
    "tau", "tau_i", "risk_thresholds", "cvar_thresholds",
    "edge_weights", "w_ij",
    "deliberation_transcript", "deliberation_log", "internal_dialogue",
    "u_i", "u_ij"
}
```

---

### 2.3 Section 2: Reward Integrity (`test_02_reward_integrity.py`)

**What**: Validates reward computation correctness and unhackability.

**Why**: Rewards drive LLM learning; corrupted rewards would train broken policies.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_2_1_reward_is_single_float` | Reward is numeric, not dict, bounded [0,1] | Single scalar required for GRPO |
| `test_2_2_lookahead_cost_is_exactly_007` | Lookahead penalty is exactly 0.07 | Cost must be precise, not approximate |
| `test_2_3_reward_in_range_after_valid_actions` | All 5 dimensions stay [0,1] | Out-of-range indicates scoring bug |
| `test_2_4_deterministic_reward_with_seed` | Same seed → same reward | Reproducibility required |
| `test_2_5_repeat_same_action_does_not_escalate_reward` | Repeating action doesn't inflate reward | Prevents reward hacking |
| `test_2_6_different_targets_different_causal_scores` | Different targets produce different causal scores | Causal dimension must be discriminative |
| `test_2_7_informative_action_outperforms_empty` | Substantive action ≥ empty message score | Prevents no-op exploitation |
| `test_2_8_reward_non_trivial_variance` | Reward variance > 0.05 across actions | Learning requires signal variance |
| `test_2_9_good_documentation_higher_than_poor` | Full docs ≥ minimal docs score | Quality must be rewarded |
| `test_2_10_lookahead_improves_prediction_accuracy` | Lookahead accuracy > 60% | Lookahead must provide useful signal |

**Key Invariants**:
- Lookahead cost = exactly 0.07 (not 0.06-0.08)
- Reward bounded [0, 1] always
- Deterministic with same seed
- No reward inflation from repetition

---

### 2.4 Section 3: Causal Inference (`test_03_causal_inference.py`)

**What**: Validates belief propagation and causal signal mechanisms.

**Why**: Causal reasoning is a core research claim; broken propagation would make environment unsuitable.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_3_1_targeted_stakeholder_engagement_changes` | Delta is numeric after targeted action | Engagement must change on action |
| `test_3_2_cross_stakeholder_echoes_detected` | Echoes appear ≥65% of episodes | Belief propagation must be active |
| `test_3_3_engagement_history_window_slides` | History size constant across steps | Window must slide, not grow |
| `test_3_4_engagement_noise_not_cancellable` | ≥2 non-zero deltas in 5 steps | Noise σ > 0 confirmed |
| `test_3_5_different_targets_different_patterns` | Finance vs Legal produce diff patterns | Targeting matters |
| `test_3_6_weak_signals_for_non_targeted` | Signals appear ≥20% of episodes | Weak signal mechanism active |
| `test_3_7_echo_content_structure` | Echoes have correct dict structure | Proper format for learning |
| `test_3_8_causal_signal_responds_to_graph_structure` | Hub (ExecSponsor) > leaf (Procurement) impact | Graph position matters |

**Propagation Requirement**: ≥65% echo recall rate (70% design target with tolerance)

---

### 2.5 Section 4: CVaR Veto (`test_04_cvar_veto.py`)

**What**: Validates CVaR-based veto mechanism.

**Why**: Core research claim: CVaR veto fires even with positive expected utility.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_4_1_veto_precursor_before_veto` | Precursor appears before veto fires | 70% tau warning mechanism works |
| `test_4_2_aligned_no_early_veto` | Aligned scenario survives first step | No premature veto in easy scenarios |
| `test_4_3_veto_terminates_episode` | Veto → done=True | Terminal condition must work |
| `test_4_3_veto_deterministic` | Seed 42 + Legal escalation → veto by step 5 | Deterministic veto behavior |
| `test_4_4_timeout_terminates` | Max rounds → episode ends | Timeout must work |
| `test_4_5_scenario_difficulty_differentiation` | hostile reaches pressure earlier than aligned | Difficulty scaling correct |
| `test_4_6_veto_precursor_is_stakeholder_specific` | Precursors tied to specific stakeholders | Not global warning |
| `test_4_7_cveto_not_just_eu` | Veto fires in hostile despite EU>0 | Core research claim verified |

**Veto Trigger Logic**:
```python
# Precursor fires at 70% of tau
if cvar_loss > tau * 0.70:
    precursor = "rising concern"

# Veto fires after 2 consecutive precursors AND cvar_loss > tau
if cvar_loss > tau and precursor_streak >= 2:
    veto_triggered = True
```

---

### 2.6 Section 5: Episode Isolation (`test_05_episode_isolation.py`)

**What**: Validates episode state management and determinism.

**Why**: Training requires independent episodes; state leakage would corrupt learning.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_5_1_different_seeds_different_initial_state` | Different seeds → different engagement levels | G is regenerated per episode |
| `test_5_2_round_number_resets_to_zero` | Reset sets round=0 | Fresh episode state |
| `test_5_3_done_false_after_reset` | done=False immediately after reset | Episode not prematurely terminated |
| `test_5_4_engagement_history_initialized` | History has ≥5 entries | Window pre-populated |
| `test_5_5_round_number_increments_correctly` | 0→1→2→... after steps | Counter must increment |
| `test_5_6_all_three_scenarios_work` | aligned, conflicted, hostile all run | All scenario types functional |
| `test_5_7_same_session_same_state_across_steps` | State consistent within episode | No random state corruption |
| `test_5_8_reset_clears_all_session_state` | Reset clears round, done, history | Full state isolation |

---

### 2.7 Section 6: Probabilistic Signals (`test_06_probabilistic_signals.py`)

**What**: Validates probabilistic observation mechanisms (runs in container).

**Why**: Noise and stochastic signals are part of partial observability requirements.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_6_1_weak_signals_field_exists` | `weak_signals` in observation | Field must exist |
| `test_6_2_cross_stakeholder_echoes_exists` | `cross_stakeholder_echoes` in observation | Field must exist |
| `test_6_3_echo_structure` | Echoes are list of dicts with from/to | Correct format |
| `test_6_4_weak_signals_populated_after_action` | Signals appear after step | Mechanism activates |
| `test_6_5_echo_firing_rate_nonzero` | Echoes ≥65% of episodes | Design target verified |
| `test_6_6_weak_signal_threshold_respected` | Signal values are string tags, not numeric | Format as designed |
| `test_6_7_echo_recall_probability_configured` | echo_recall_probability ≈ 0.70 | Configuration correct |

---

### 2.8 Section 7: Causal Graph Unit Tests (`test_07_causal_graph.py`)

**What**: Validates graph structure and propagation algorithms (runs in container).

**Why**: Causal graph is foundation for belief propagation and reward computation.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_7_1_propagation_direction` | A→B signal flows, C unchanged | Edges determine propagation |
| `test_7_2_signal_carries_through_edge` | B changes when A changes | Connected nodes share influence |
| `test_7_3_damping_prevents_runaway` | All beliefs stay [0,1] after dense propagation | Damping prevents overflow |
| `test_7_4_beliefs_normalized_after_propagation` | Sum of beliefs = 1.0 | Invariant maintained |
| `test_7_5_no_self_loops` | No edges (A→A) in any scenario | Self-loops would cause instability |
| `test_7_6_exec_sponsor_outgoing_authority` | ExecSponsor has ≥2 outgoing edges | Authority invariant preserved |
| `test_7_7_hub_centrality_beats_leaf` | Hub betweenness > leaf | Centrality calculation correct |
| `test_7_8_graph_identifiability` | All 20 sampled graphs unique | Every G is distinguishable |
| `test_7_9_hub_node_has_higher_centrality_impact` | Hub impact ratio > 1.3× leaf | Targeting hub has more effect |

**Graph Identifiability**: 20 graphs must produce 190 unique pairs (all distinguishable)

---

### 2.9 Section 8: CVaR Preferences Unit Tests (`test_08_cvar_preferences.py`)

**What**: Validates CVaR computation and stakeholder risk profiles (runs in container).

**Why**: CVaR is the core research innovation; mathematical correctness essential.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_8_1_core_claim` | eu>0 yet cvar>tau → veto condition met | Core research claim mathematically verified |
| `test_8_2_good_docs_lower_cvar_than_poor` | DPA+cert reduces CVaR vs poor docs | Documentation reward justified |
| `test_8_3_cvar_formula_correct` | High outcomes → low CVaR, low outcomes → high CVaR | Formula correct |
| `test_8_4_tau_ordering` | Legal < Finance < ExecSponsor tau | Risk tolerance ordering correct |
| `test_8_5_aggressive_timeline_higher_cvar` | 4-week vs 16-week → higher CVaR | Timeline risk reflected |
| `test_8_6_cvar_subadditivity_sanity` | CVaR ≤ max outcome | Coherence property satisfied |

**Core Claim Verification**:
```python
# Given terms with no DPA, no security_cert:
terms = {"has_dpa": False, "has_security_cert": False, "liability_cap": 0.2}
eu, cvar_loss = evaluate_deal(terms, legal_profile, rng)
# Assert: eu > 0 AND cvar_loss > tau
# This proves CVaR can veto despite positive EU
```

---

### 2.10 Section 9: Full Episode E2E (`test_09_full_episode_e2e.py`)

**What**: End-to-end episode validation across all scenarios.

**Why**: Integration testing ensures components work together.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_9_1_aligned_completes` | Aligned + neutral → episode ends | Basic functionality works |
| `test_9_2_conflicted_completes` | Conflicted + comprehensive → episode ends | More complex scenario works |
| `test_9_3_hostile_aggressive_produces_veto_or_timeout` | Hostile + aggressive → veto or timeout | Adversarial handled |
| `test_9_4_reward_collected_across_episode` | Rewards accumulated throughout | Learning signal available |
| `test_9_5_reward_variance_non_trivial` | Spread > 0.05 across episode | Signal discriminative |
| `test_9_6_strategy_comparison_possible` | Comprehensive > aggressive by >0.15 | Strategy matters |
| `test_9_7_terminal_outcome_meaningful` | done=True fires at natural endpoint | Termination correct |
| `test_9_8_multidocument_action_works` | Multi-doc action returns reward | Document handling works |

**Strategies Tested**:
- `neutral`: Basic direct_messages
- `comprehensive`: Targeted document sending to each stakeholder
- `aggressive`: Repeated exec_escalation

---

### 2.11 Section 10: Training Infrastructure (`test_10_training_infrastructure.py`)

**What**: Validates GRPO training pipeline components (runs in container).

**Why**: Training infrastructure must work correctly for LLM improvement.

**How**:

| Test | What It Does | Why |
|------|-------------|-----|
| `test_10_1_grpo_trainer_imports` | `GRPOTrainer`, `TrainingMetrics` importable | Classes available |
| `test_10_2_training_metrics_fields` | All 6 reward fields present | Metrics tracked |
| `test_10_3_curriculum_generator_imports` | `AdaptiveCurriculumGenerator` importable | Curriculum available |
| `test_10_4_colab_notebook_exists` | Notebook ≥5 cells with valid structure | Training guide available |
| `test_10_5_training_loop_smoke_test` | `GRPOTrainer` class properly defined | Basic interface works |
| `test_10_6_checkpoint_save_load_smoke` | PyTorch availability check | Checkpointing optional |

---

### 2.12 Section 11: Research Properties (`test_11_research_properties.py`)

**What**: Validates all 12 research desiderata.

**Why**: Comprehensive validation of research claims in single test suite.

**How**:

| Property | What It Validates | How |
|---------|-------------------|-----|
| P1 | G is hidden from agent | Hidden fields not in observation |
| P2 | Episode reset regenerates G | Different seeds → different states |
| P3 | CVaR veto despite positive EU | eu>0 AND cvar>tau verified |
| P4 | Five reward dims discriminative | Variance > threshold |
| P5 | Lookahead cost exactly 0.07 | Diff measured vs theoretical |
| P6 | Engagement noise not cancellable | Non-zero deltas confirmed |
| P7 | Cross-stakeholder echoes present | Field exists and populated |
| P8 | Weak signals present | Field exists |
| P9 | r^causal varies with target | Different targets → different scores |
| P10 | Every reset different G | 5 resets → ≥2 unique states |
| P11 | Full episode no crash | All scenarios complete |
| P12 | Training loop imports | GRPOTrainer + curriculum import |

---

### 2.13 P0 Comprehensive Tests (`test_P0_comprehensive.py`)

**What**: Critical issues validation from design review.

**Why**: Tests the 4 P0 concerns identified during architecture review.

**How**:

#### P0-1: Training Improvement Validation

| Test | What It Does | Why |
|------|-------------|-----|
| `test_P0_1a_baseline_vs_trained` | Random vs trained policy comparison | Improvement >0.10 required |
| `test_P0_1b_multi_episode_improvement` | Later batches ≥ earlier batches | Curriculum helps |
| `test_P0_1c_dimension_wise_improvement` | ≥2 dimensions improve with training | Specific improvements |
| `test_P0_1d_policy_persistence` | Save/load checkpoint preserves state | Training state maintained |

#### P0-2: CVaR Mathematical Correctness

| Test | What It Does | Why |
|------|-------------|-----|
| `test_P0_2a_cvar_deterministic_calculation` | Manual vs computed CVaR match | Formula implemented correctly |
| `test_P0_2b_cvar_veto_with_positive_eu` | Veto fires despite EU>0 | Core claim validated |
| `test_P0_2c_cvar_per_stakeholder` | Legal CVaR ≠ Finance CVaR | Profiles differentiated |

#### P0-3: Lookahead Usefulness Validation

| Test | What It Does | Why |
|------|-------------|-----|
| `test_P0_3a_lookahead_improves_decisions` | With lookahead ≥ without | Benefit exceeds cost |
| `test_P0_3b_lookahead_prediction_accuracy` | Accuracy > 55% | Predictions useful |
| `test_P0_3c_lookahead_cost_exactly_007` | Cost = 0.07 exactly | Precision verified |

#### P0-4: Full Debug Trace

| Test | What It Does | Why |
|------|-------------|-----|
| `test_P0_4a_belief_state_trace` | Step-by-step reward signals | Transparency |
| `test_P0_4b_cvar_breakdown_per_stakeholder` | EU, CVaR, tau per stakeholder | Interpretability |
| `test_P0_4c_action_effect_trace` | Stage/momentum changes logged | Behavior understanding |

---

## 3. Unit Tests

### 3.1 Belief Tracker Tests (`test_belief_tracker.py`)

| Test | What It Does | Why |
|------|-------------|-----|
| `test_likelihood_values_in_range` | P(action\|type) ∈ [0.10, 0.95] | Valid probabilities |
| `test_get_likelihood_exact_match` | Document name matching works | Flexible lookup |
| `test_distribution_normalizes` | Belief sums to 1.0 after update | Invariant maintained |
| `test_bayesian_update_concentrates_belief` | 10 competent actions → positive_mass > 0.70 | Learning works |
| `test_targeted_vs_nontargeted_strength` | Targeted delta > non-targeted | Targeting matters |
| `test_positive_mass_increases_on_competent_action` | DPA → positive_mass increases | Action effects correct |
| `test_damping_factor_applied` | Non-targeted 0.7× targeted | Damping formula correct |
| `test_engagement_level_bounds` | [-1, 1] range | Proper normalization |

---

### 3.2 Curriculum Tests (`test_curriculum.py`)

| Test | What It Does | Why |
|------|-------------|-----|
| `test_config_defaults` | easy+frontier+hard = 1.0 | Valid distribution |
| `test_generator_initialization` | Pool initialized | Setup works |
| `test_select_next_scenario` | Returns valid dict | Selection works |
| `test_analyze_failures_empty_trajectories` | Handles empty input | Edge case |
| `test_failure_detection_f1` | F1 (CVaR veto) detected | Detection works |
| `test_scenario_pool_has_all_difficulties` | Pool has easy, frontier, hard | Coverage |
| `test_capability_based_selection` | Difficulty respects capability | Adaptation works |

---

### 3.3 Utterance Scorer Tests (`test_utterance_scorer.py`)

| Test | What It Does | Why |
|------|-------------|-----|
| `test_lookahead_cost_value` | LOOKAHEAD_COST = 0.07 | Constant correct |
| `test_lookahead_cost_subtracted_from_goal` | With lookahead → goal reduced by 0.07 | Cost applied |
| `test_all_dimensions_in_range` | All 5 dims ∈ [0, 1] for random inputs | Bounded |
| `test_causal_score_deterministic` | Same graph/target → identical score | Deterministic |
| `test_utterance_score_defaults` | Default values correct | Initialization |
| `test_weighted_sum` | Scalar = weighted sum of dims | Combination correct |

---

## 4. Test Execution

### 4.1 Container Tests (Run Inside Docker)

```bash
docker exec dealroom-v3-test python -m pytest tests/v3/test_06_probabilistic_signals.py
docker exec dealroom-v3-test python -m pytest tests/v3/test_07_causal_graph.py
docker exec dealroom-v3-test python -m pytest tests/v3/test_08_cvar_preferences.py
docker exec dealroom-v3-test python -m pytest tests/v3/test_10_training_infrastructure.py
```

### 4.2 API Tests (Require Running Container)

```bash
DEALROOM_BASE_URL=http://127.0.0.1:7860 python -m pytest tests/v3/test_00_environment_setup.py
python -m pytest tests/v3/test_01_schema_validation.py
python -m pytest tests/v3/test_02_reward_integrity.py
```

### 4.3 Unit Tests (No Container Required)

```bash
python -m pytest tests/unit/ -v
```

---

## 5. Test Design Principles

### 5.1 What vs Why

Each test clearly states:
- **What**: The specific behavior being tested
- **Why**: The research or engineering reason this behavior matters

Example:
- **What**: `test_2_5_repeat_same_action_does_not_escalate_reward`
- **Why**: "Prevents reward hacking - agent must take different actions to accumulate reward"

### 5.2 Fail-Fast with Tolerance

Stochastic elements (noise, randomness) are handled by:
- Fixed seeds for deterministic checks
- Tolerance ranges for stochastic validation (e.g., ≥65% not exactly 70%)
- Multiple trials averaged

### 5.3 Research Claims Separated

Critical claims are validated independently:
- CVaR veto with EU>0 (test_8_1_core_claim, test_P0_2b)
- Causal propagation (Section 3, 7)
- Hidden state (test_1_2, P1)

### 5.4 No Implementation Leakage

Tests validate observable behavior, not internal implementation:
- Don't test private methods directly (test public interface)
- Don't assert on specific variable names
- Focus on outputs and side effects

---

## 6. Interpreting Test Results

### 6.1 Success Criteria

- **Section 0**: All 5 tests pass (environment ready)
- **Section 1**: All 11 tests pass (schema valid)
- **Section 2**: All 10 tests pass (reward integrity)
- **Sections 3-5**: All tests pass (causal, veto, isolation)
- **Sections 6-8**: All container tests pass (probabilistic, graph, CVaR)
- **Section 9**: All 8 tests pass (E2E)
- **Section 10**: All 6 tests pass (training infrastructure)
- **Section 11**: All 12 properties confirmed
- **P0**: All critical issues validated

### 6.2 Failure Modes

| Failure | Likely Cause | Fix |
|---------|-------------|-----|
| test_0_3 Docker fails | Container not running | Start container |
| test_1_2 hidden fields exposed | Bug in observation filter | Fix observation construction |
| test_2_4 non-deterministic | Random not seeded properly | Check seed handling |
| test_3_2 low echo rate | Propagation bug | Check propagate_beliefs |
| test_4_1 no precursor | CVaR precursor logic | Check veto warning threshold |
| test_7_8 low identifiability | Graph sampling flawed | Check sample_graph |

---

*This testing documentation describes the test suite for DealRoom S2P V2, designed to validate an LLM training environment for B2B software negotiation.*
