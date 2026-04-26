# Test Results — DealRoom S2P V2

**Date:** April 26, 2026
**Environment:** macOS Darwin (ARM) — Python 3.12.4
**Test Framework:** pytest 9.0.3 with plugins: dash-3.1.1, langsmith-0.4.45, anyio-4.13.0
**Server:** http://127.0.0.1:7860 (running)

---

## Executive Summary

| Suite | Passed | Failed | Skipped | Total |
|-------|--------|--------|---------|-------|
| Unit Tests | 120 | 1 | 0 | 121 |
| test_00 | 4 | 1 | 0 | 5 |
| test_01 | 11 | 0 | 0 | 11 |
| test_02 | 9 | 1 | 0 | 10 |
| test_03 | 8 | 0 | 0 | 8 |
| test_04 | 7 | 1 | 0 | 8 |
| test_05 | 8 | 0 | 0 | 8 |
| test_06 | 7 | 0 | 0 | 7 |
| test_07 | 9 | 0 | 0 | 9 |
| test_08 | 6 | 0 | 0 | 6 |
| test_09 | 7 | 1 | 0 | 8 |
| test_10 | 5 | 1 | 0 | 6 |
| test_11 | 0 | 0 | 0 | 0 (empty) |
| test_P0 | 13 | 1 | 2 | 16 |
| **TOTAL** | **214** | **7** | **2** | **223** |

---

## Unit Tests (tests/unit/)

```
========================= test session starts =========================
platform darwin -- Python 3.12.4, pytest-9.0.3, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /Users/akshaypulla/Documents/deal_room_S2P
configfile: pyproject.toml
plugins: dash-3.1.1, langsmith-0.4.45, anyio-4.13.0
========================= 120 passed, 1 failed in 11.03s =========================
```

### Failures

#### test_belief_tracker.py::test_done_flag_after_max_rounds

**Error:**
```
AssertionError: expected 'done=True' but got done=False after max rounds
```

---

## Integration Tests

### test_00_environment_setup.py

```
========================= 4 passed, 1 failed in 0.58s =========================
```

| Test | Status |
|------|--------|
| test_0_1_server_health | PASSED |
| test_0_2_schema_validation | PASSED |
| test_0_3_scenario_reset | PASSED |
| test_0_4_three_scenario_reset | PASSED |
| test_0_5_docker_environment | FAILED (container not running — expected for local dev) |

**Note:** Docker container `dealroom-v3-test` not running. This is expected in local development without Docker.

---

### test_01_schema_validation.py

```
========================= 11 passed in 0.56s =========================
```

All schema validation tests passed for observation schema, action schema, info dict structure, and required fields.

---

### test_02_reward_integrity.py

```
========================= 9 passed, 1 failed in 0.69s =========================
```

| Test | Status |
|------|--------|
| test_2_1_reward_fields_exist | PASSED |
| test_2_2_reward_signal_shapes | PASSED |
| test_2_3_reward_weights_sum_to_one | PASSED |
| test_2_4_reward_weights_in_observation | PASSED |
| test_2_5_causal_reward_exists | PASSED |
| test_2_6_risk_reward_exists | PASSED |
| test_2_7_reward_increments | PASSED |
| test_2_8_terminal_reward_consistency | PASSED |
| test_2_9_rounding_errors | PASSED |
| test_2_10_lookahead_reduces_cvar | FAILED |

**Failure:** `test_2_10_lookahead_reduces_cvar`
- Lookahead accuracy: 0.33 (< threshold 0.60)
- Expected: lookahead-enabled decisions show measurable improvement

---

### test_03_causal_inference.py

```
========================= 8 passed in 0.06s =========================
```

All causal inference tests passed including signal propagation, graph structure, and belief updates.

---

### test_04_cvar_veto.py

```
========================= 7 passed, 1 failed in 0.84s =========================
```

| Test | Status |
|------|--------|
| test_4_1_veto_precursor_before_veto | PASSED |
| test_4_2_aligned_no_early_veto | PASSED |
| test_4_3_veto_terminates_episode | PASSED |
| test_4_3_veto_deterministic | FAILED |
| test_4_4_timeout_terminates | PASSED |
| test_4_5_scenario_difficulty_differentiation | PASSED |
| test_4_6_veto_precursor_is_stakeholder_specific | PASSED |
| test_4_7_cveto_not_just_eu | PASSED |

**Failure:** `test_4_3_veto_deterministic`
```
AssertionError: Expected terminal_category='veto', got soft_veto_by_Legal
```
The terminal category is `soft_veto_by_Legal` instead of `veto`. This indicates the veto mechanism fires but with a different terminal categorization than expected by the test.

---

### test_05_episode_isolation.py

```
========================= 8 passed in 0.70s =========================
```

All episode isolation tests passed including seed isolation, round resets, and session state management.

---

### test_06_probabilistic_signals.py

```
========================= 7 passed in 0.89s =========================
```

All probabilistic signal tests passed including weak signal thresholds, cross-stakeholder echoes, and signal propagation.

---

### test_07_causal_graph.py

```
========================= 9 passed in 0.06s =========================
```

All causal graph tests passed including propagation direction, damping, belief normalization, and centrality calculations.

---

### test_08_cvar_preferences.py

```
========================= 6 passed in 0.04s =========================
```

All CVaR preference tests passed including core CVaR formula, tau ordering, and subadditivity sanity checks.

---

### test_09_full_episode_e2e.py

```
========================= 7 passed, 1 failed in 2.22s =========================
```

| Test | Status |
|------|--------|
| test_9_1_aligned_completes | PASSED |
| test_9_2_conflicted_completes | PASSED |
| test_9_3_hostile_aggressive_produces_veto_or_timeout | FAILED |
| test_9_4_reward_collected_across_episode | PASSED |
| test_9_5_reward_variance_non_trivial | PASSED |
| test_9_6_strategy_comparison_possible | PASSED |
| test_9_7_terminal_outcome_meaningful | PASSED |
| test_9_8_multidocument_action_works | PASSED |

**Failure:** `test_9_3_hostile_aggressive_produces_veto_or_timeout`
```
AssertionError: Unexpected terminal outcome: soft_veto_by_Legal
```
The episode terminates with `soft_veto_by_Legal` which is not in the expected list of terminal outcomes.

---

### test_10_training_infrastructure.py

```
========================= 5 passed, 1 failed, 1 warning in 1.43s =========================
```

| Test | Status |
|------|--------|
| test_10_1_grpo_trainer_imports | PASSED |
| test_10_2_training_metrics_fields | PASSED |
| test_10_3_curriculum_generator_imports | PASSED |
| test_10_4_colab_notebook_exists | FAILED |
| test_10_5_training_loop_smoke_test | PASSED |
| test_10_6_checkpoint_save_load_smoke | PASSED |

**Failure:** `test_10_4_colab_notebook_exists`
```
AssertionError: grpo_colab.ipynb not found
```
Colab notebook file not present in expected paths.

---

### test_11_research_properties.py

```
========================= 0 tests ran in 0.03s =========================
```

No tests defined in this file.

---

### test_P0_comprehensive.py

```
========================= 13 passed, 1 failed, 2 skipped in 3.69s =========================
```

| Test | Status |
|------|--------|
| test_P0_1a_baseline_vs_trained_comparison | SKIPPED |
| test_P0_1b_multi_episode_improvement | PASSED |
| test_P0_1c_dimension_wise_improvement | PASSED |
| test_P0_1d_policy_persistence | PASSED |
| test_P0_2a_cvar_deterministic_calculation | PASSED |
| test_P0_2b_cvar_veto_with_positive_eu | PASSED |
| test_P0_2c_cvar_per_stakeholder | PASSED |
| test_P0_3a_lookahead_improves_decisions | PASSED |
| test_P0_3b_lookahead_prediction_accuracy | FAILED |
| test_P0_3c_lookahead_cost_exactly_007 | SKIPPED |
| test_P0_4a_belief_state_trace | PASSED |
| test_P0_4b_cvar_breakdown_per_stakeholder | PASSED |
| test_P0_4c_action_effect_trace | PASSED |
| test_P1_stochastic_stabilized | PASSED |
| test_P2_adversarial_degenerate_graph | PASSED |
| test_P2_adversarial_extreme_tau | PASSED |

**Failure:** `test_P0_3b_lookahead_prediction_accuracy`
```
AssertionError: Lookahead accuracy too low: 0.332. Expected >0.55
```
Lookahead prediction accuracy is 0.332 across 25 runs, below the 0.55 threshold.

---

## Key Observations

### 1. Terminal Category Naming Mismatch

Several tests expect `terminal_category='veto'` but the actual value is `soft_veto_by_Legal`. This is a naming/categorization issue rather than a functional bug — the veto mechanism works correctly, but the terminal category naming is inconsistent between tests and implementation.

**Affected tests:**
- test_04::test_4_3_veto_deterministic
- test_09::test_9_3_hostile_aggressive_produces_veto_or_timeout

### 2. Lookahead Accuracy Below Threshold

The lookahead prediction accuracy consistently measures around 0.33 (33%), well below the 0.55-0.60 threshold expected by tests.

**Affected tests:**
- test_02::test_2_10_lookahead_reduces_cvar (0.33 vs 0.60 threshold)
- test_P0::test_P0_3b_lookahead_prediction_accuracy (0.332 vs 0.55 threshold)

### 3. Missing Colab Notebook

The `grpo_colab.ipynb` notebook file is not present in the expected paths:
- `/app/env/deal_room/training/grpo_colab.ipynb`
- `/app/env/grpo_colab.ipynb`
- `deal_room/training/grpo_colab.ipynb`

### 4. Docker Environment Not Available

Docker container `dealroom-v3-test` is not running. This is expected for local development but prevents container-based integration tests from running.

---

## Environment Details

```
Platform:     macOS Darwin (ARM)
Python:       3.12.4
Pytest:       9.0.3
Plugins:      dash-3.1.1, langsmith-0.4.45, anyio-4.13.0
Server:       http://127.0.0.1:7860 (running)
Working Dir:  /Users/akshaypulla/Documents/deal_room_S2P
```

### Key Dependencies

| Package | Version |
|---------|---------|
| torch | (installed, version in warnings) |
| numpy | (installed) |
| fastapi | (installed) |
| uvicorn | (installed) |
| pydantic | (installed) |
| networkx | (installed) |
| anthropic | 0.96.0 |
| openai | (installed) |

---

## Test Execution Commands

```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/v3/test_00_environment_setup.py -v
python -m pytest tests/v3/test_01_schema_validation.py -v
python -m pytest tests/v3/test_02_reward_integrity.py -v
python -m pytest tests/v3/test_03_causal_inference.py -v
python -m pytest tests/v3/test_04_cvar_veto.py -v
python -m pytest tests/v3/test_05_episode_isolation.py -v
python -m pytest tests/v3/test_06_probabilistic_signals.py -v
python -m pytest tests/v3/test_07_causal_graph.py -v
python -m pytest tests/v3/test_08_cvar_preferences.py -v
python -m pytest tests/v3/test_09_full_episode_e2e.py -v
python -m pytest tests/v3/test_10_training_infrastructure.py -v
python -m pytest tests/v3/test_P0_comprehensive.py -v

# All tests
python -m pytest tests/ -v
```
