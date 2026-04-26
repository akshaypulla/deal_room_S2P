# Testing Documentation - DealRoom S2P V3

This document describes the test suite structure, what each test does, why it exists, and how it works.

---

## Test Directory Structure

```
tests/
├── conftest.py                    # Shared pytest fixtures
├── unit/                           # Unit tests (121 tests)
├── integration/                     # Integration tests (20 tests)
├── v3/                             # V3 research property tests (containers required)
├── e2e/                            # End-to-end workflow tests
├── performance/                    # Performance benchmarks
└── container_test_api.py          # Docker container API tests
```

---

## Unit Tests (`tests/unit/`)

### `test_belief_tracker.py` — Bayesian Belief Updates

**What**: Tests the belief tracking system that updates stakeholder beliefs about the vendor based on observed actions.

**Why**: The environment relies on Bayesian inference to model how stakeholders update their beliefs. If belief updates are incorrect, the entire committee dynamics break down.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_likelihood_values_in_range` | All P(action\|type) in [0.10, 0.95] | Valid probability bounds | Iterates ACTION_LIKELIHOODS dict, asserts each value in range |
| `test_get_likelihood_exact_match` | Exact action match returns correct likelihoods | Priority matching | Calls `_get_likelihood()` with known action "send_document(DPA)_proactive" |
| `test_get_likelihood_document_matching` | Document name used for matching | Fallback behavior | Passes docs list with "DPA", checks "competent" key in result |
| `test_get_likelihood_default` | Unknown action returns default likelihoods | Graceful degradation | Uses unknown action, checks all types get 0.50 |
| `test_distribution_normalizes` | Post-update distribution sums to 1.0 | Conservation property | Performs bayesian_update, sums distribution values |
| `test_bayesian_update_concentrates_belief` | 10 consistent competent actions → positive_mass > 0.70 | Convergence behavior | Iterates 10 updates with DPA action, checks concentration |
| `test_targeted_vs_nontargeted_strength` | Non-targeted delta < targeted × 0.8 | Damping effect | Compares deltas for same action with different targeting |
| `test_positive_mass_increases_on_competent_action` | DPA increases positive_mass | Positive action effect | Checks positive_mass increased after DPA action |
| `test_negative_mass_increases_on_deceptive_action` | Deceptive action increases negative_mass | Negative action effect | Iterates deceptive action, checks negative_mass increase |
| `test_damping_factor_applied` | Non-targeted uses ~0.7 damping | Damping verification | Compares delta ratio, expects 0.4 < ratio < 0.9 |
| `test_confidence_increases_with_informatie_action` | Confidence decreases after informative action | Entropy reduction | Checks confidence decreased (more certainty = lower entropy) |
| `test_history_records_updates` | History records each update | Audit trail | Checks history length increased, last entry correct |
| `test_engagement_level_matches_positive_minus_negative` | Engagement = positive_mass - negative_mass | Engagement formula | Direct computation comparison |
| `test_engagement_level_bounds` | Engagement level in [-1, 1] | Bounded metric | Random distribution test, 100 trials |
| `test_neutral_beliefs_uniform` | Neutral beliefs uniform over 6 vendor types | Initial belief state | Checks each vendor type has 1/6 probability |
| `test_neutral_beliefs_confidence` | Neutral beliefs start with confidence >= 0.9 | High initial uncertainty | Direct assertion on confidence value |

---

### `test_causal_graph.py` — Causal Graph and Belief Propagation

**What**: Tests the causal graph structure that models stakeholder influence relationships and the belief propagation algorithm.

**Why**: The causal graph is the hidden backbone of committee dynamics. Stakeholder influence must flow correctly through edges, and every reset must produce a unique graph for variety.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_sample_graph_returns_valid_graph` | Graph construction produces valid CausalGraph | Basic construction | Samples graph, checks type and field types |
| `test_sample_graph_no_self_edges` | No self-edges in any scenario | Invariant: no self-influence | Samples 3 scenarios, asserts no self-loop |
| `test_weight_range` | All edge weights in [0.05, 0.95] | Valid weight bounds | Samples 10 graphs, asserts weight range |
| `test_authority_invariant` | ExecSponsor has >= 1 outgoing edge in all scenarios | Authority hub requirement | Samples across scenarios/seeds |
| `test_authority_weights_sum_to_one` | Authority weights normalized to 1.0 | Normalization invariant | Sums authority_weights, asserts 1.0 |
| `test_positive_mass_calculation` | positive_mass = competent + trustworthy + aligned | Mass computation | Direct formula test |
| `test_negative_mass_calculation` | negative_mass = incompetent + deceptive + misaligned | Mass computation | Direct formula test |
| `test_belief_copy_independent` | copy() creates independent deep copy | Memory isolation | Modifies copy, checks original unchanged |
| `test_create_neutral_beliefs` | Neutral = uniform over 6 vendor types | Initial belief state | Checks each type probability |
| `test_propagation_direction` | Delta propagates positively along edges | Propagation correctness | A→B edge, applies delta to A, checks B increased |
| `test_damping_prevents_runaway` | Dense graph beliefs stay in (0, 1) after 5 steps | Damping effectiveness | Fully connected graph, 5 steps, bounds check |
| `test_apply_belief_delta_positive` | Positive delta shifts mass from negative to positive types | Mass shift behavior | Applies +0.3 delta, checks mass shift direction |
| `test_apply_belief_delta_negative` | Negative delta shifts mass from positive to negative types | Mass shift direction | Applies -0.3 delta, checks reverse shift |
| `test_apply_belief_delta_with_damping` | Higher damping = smaller effective delta | Damping formula | Compares damped vs undamped delta magnitude |
| `test_centrality_hub_higher_than_leaves` | Hub node has higher betweenness than leaves | Centrality computation | Star graph, hub vs leaves |
| `test_isolated_node_has_zero_centrality` | Isolated node has zero betweenness centrality | Edge case: no paths through node | Graph with isolated C, checks centrality=0 |
| `test_behavioral_signature_distinct` | Different targets produce different signatures | Graph identifiability | Samples graph, computes signatures for all targets, checks pairwise L1 distance |
| `test_graph_identifiability_statistical` | 20 sampled graphs produce pairwise distinguishable signatures | Statistical identifiability | Samples 5 graphs, checks signature distinctness |
| `test_engagement_level_range` | Engagement level in [-1, 1] | Bounded metric | Direct range check |
| `test_engagement_level_neutral_is_zero` | Neutral belief has engagement near 0 | Neutral baseline | Uniform distribution test |
| `test_aligned_sparser_than_conflicted` | Aligned graphs sparser than conflicted | Scenario complexity difference | Samples two graphs, compares edge counts |

---

### `test_deliberation_engine.py` — Committee Deliberation

**What**: Tests the deliberation engine that orchestrates committee dynamics, including belief propagation, voting, and Executive Sponsor activation.

**Why**: The deliberation engine is critical for modeling how the committee collectively responds to vendor actions. It must correctly aggregate individual stakeholder beliefs into committee decisions.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_deliberation_result_structure` | DeliberationResult has all required fields | Validates return structure | Checks updated_beliefs, summary_dialogue, propagation_deltas |
| `test_deliberation_steps_per_scenario` | DELIBERATION_STEPS mapping correct (aligned=3, conflicted=3, hostile=4) | Scenario-specific deliberation depth | Direct assertions on dict values |
| `test_deliberation_updates_beliefs` | Beliefs updated through propagation | Core deliberation behavior | Runs deliberation, checks all 3 stakeholders have updated beliefs |
| `test_deliberation_propagation_deltas_recorded` | propagation_deltas records change per stakeholder | Debug/tracing for belief changes | Checks delta dict contains both A and B |
| `test_deliberation_no_summary_when_no_targets` | Empty summary when action has no targets | Edge case handling | Verifies summary is None or empty |
| `test_deliberation_pure_python_layer1` | Layer 1 (propagate_beliefs) runs without LLM | Ensures belief propagation is deterministic | Directly calls propagate_beliefs(), checks B's mass changed |
| `test_layer2_returns_string_or_empty` | Layer 2 returns string or empty on failure | LLM call failure handling | Calls deliberation with render_summary=True |
| `test_single_step_deliberation` | 1-step deliberation produces valid results | Minimum deliberation depth | Runs with n_deliberation_steps=1 |
| `test_many_steps_damping` | 10 steps with damping prevents runaway | Ensures damping works with many steps | Runs 10 steps, checks all beliefs stay in (0,1) |
| `test_propagation_follows_graph_structure` | Belief changes follow edge structure (A→B→C chain) | Graph-structured belief propagation | Applies delta to A, checks B > C change magnitude |
| `test_no_edges_no_propagation` | No edges = no propagation to non-targeted | Isolated node behavior | Checks B unchanged when no edge to A |

---

### `test_curriculum.py` — Adaptive Curriculum Generator

**What**: Tests the adaptive curriculum system that adjusts training difficulty based on detected failure modes.

**Why**: Curriculum learning ensures the agent progresses from easy to hard scenarios. The system must detect 7 distinct failure modes and adjust difficulty accordingly.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_failure_analysis_defaults` | FailureAnalysis has correct default values | Validates failure analysis dataclass | Checks failure_modes dict, worst_graph_configs list, capability bounds |
| `test_config_defaults` | easy_ratio=0.20, frontier_ratio=0.60, hard_ratio=0.20 | Validates difficulty distribution ratios | Direct assertions on config fields |
| `test_generator_initialization` | Generator initializes with scenario pool | Ensures curriculum has scenarios to select from | Checks _scenario_pool non-empty, RNG initialized |
| `test_select_next_scenario` | Returns valid scenario dict with task_id | Validates scenario selection | Asserts dict with valid task_id |
| `test_generate_adaptive_scenario` | Generates scenario dict | Tests adaptive scenario generation | Direct dict structure assertion |
| `test_analyze_failures_empty_trajectories` | Handles empty trajectory list | Edge case for failure analysis | Asserts FailureAnalysis returned with empty failure_modes |
| `test_failure_detection_f1` | F1 (CVaR veto) detected from trajectory | Validates F1 failure mode detection | Creates trajectory with veto outcome, checks failures |
| `test_failure_detection_f3` | F3 (causal no improvement) detected | Validates F3 failure mode detection | Creates trajectory with low causal rewards |
| `test_failure_mode_descriptions_exist` | All 7 failure mode descriptions defined | Ensures failure modes are documented | Checks FAILURE_MODE_DESCRIPTIONS dict |
| `test_scenario_pool_has_all_difficulties` | Pool contains easy, frontier, hard scenarios | Ensures curriculum diversity | Collects unique difficulties, asserts all present |
| `test_capability_based_selection` | Selection respects capability estimate | Tests adaptive difficulty selection | Checks scenario selection with different capability levels |
| `test_generate_scenario_with_seed` | Reproducible scenario generation | Ensures seeded reproducibility | Generates two scenarios, asserts dict types |
| `test_scenario_structure` | Required fields present in generated scenario | Schema validation | Checks task_id and difficulty fields |
| `test_full_cycle_analyze_and_generate` | Full analyze → generate cycle | Integration test for curriculum pipeline | Analyzes mock trajectories, generates scenario |
| `test_create_curriculum_generator` | Factory function returns generator | Validates factory pattern | Direct type assertion |
| `test_create_with_custom_config` | Custom config accepted | Tests configurability | Checks config values after creation |

---

### `test_cvar_preferences.py` — CVaR Risk Preferences

**What**: Tests the CVaR (Conditional Value at Risk) preference modeling for stakeholders, including veto triggers.

**Why**: CVaR is the core risk metric used by stakeholders to make veto decisions. Legal (alpha=0.95) is highly risk-averse and can veto even profitable deals if tail risk is high. This validates research property P3.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_cvar_veto_fires_despite_positive_expected_utility` | **P3 Core Claim**: Veto fires when CVaR > tau even with EU > 0 | Research hypothesis validation | Evaluates deal without DPA for Legal, asserts EU > 0 AND veto_triggered |
| `test_cvar_does_not_fire_with_full_documentation` | Full docs reduce CVaR significantly | Documentation value | Compares CVaR with/without DPA+security cert |
| `test_cvar_decreases_monotonically_with_documentation` | Adding docs never increases CVaR | Monotonicity guarantee | Compares DPA-only vs no docs |
| `test_cvar_computation_on_uniform_distribution` | CVaR on uniform [0,1] returns expected value | CVaR formula correctness | Checks 0.50 < CVaR_0.95 < 1.0, CVaR_0.50 < CVaR_0.95 |
| `test_cvar_handles_empty_outcomes` | Empty array returns 0.0 | Edge case handling | Direct assertion |
| `test_risk_profile_ordering` | CVaR ordering: Legal > Procurement > Finance > Ops > TechLead > Exec | Risk sensitivity ordering | Computes CVaR for all archetypes, checks relative ordering |
| `test_lambda_risk_weight` | High lambda (Legal) more influenced by CVaR than low lambda (TechLead) | Risk weighting behavior | Compares quality scores between archetypes |
| `test_veto_fires_above_tau` | Veto fires when CVaR > tau | Veto threshold logic | Direct threshold tests |
| `test_veto_regardless_of_expected_utility` | Veto based on CVaR alone | Silent veto behavior | Checks high_cvar > tau triggers veto |
| `test_outcome_distribution_sums_to_one` | Mean outcome in [0, 1] | Valid outcome range | Samples outcomes, checks mean bounds |
| `test_outcome_distribution_respects_domain` | Different uncertainty domains produce different distributions | Domain-specific risk | Compares compliance vs cost profile distributions |
| `test_observable_signals_for_legal` | Legal compliance concern signal present | Observable signal generation | Checks "compliance_concern" in signals |
| `test_observable_signals_risk_tolerance` | Risk tolerance inversely related to lambda_risk | Signal consistency | Compares Legal vs TechLead signals |
| `test_all_archetypes_defined` | All 6 archetypes present | Archetype completeness | Direct lookup for each archetype |
| `test_archetype_values_locked` | Archetype parameter values match locked spec | Parameter lock verification | Checks alpha, tau, lambda_risk values |
| `test_archetype_utility_weights` | Archetypes have appropriate utility weights | Domain-appropriate weights | Checks domain-specific utility weights |

---

### `test_utterance_scorer.py` — 5-Dimensional Reward Scoring

**What**: Tests the utterance scorer that computes multi-dimensional reward signals (goal, trust, info, risk, causal).

**Why**: The reward signal is the primary learning signal for the LLM. Each dimension must be bounded [0,1], deterministic, and correctly computed.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_lookahead_cost_value` | LOOKAHEAD_COST == 0.07 constant | Validates exact lookahead penalty value | Direct assertion |
| `test_lookahead_cost_subtracted_from_goal` | Goal dimension reduced by exactly LOOKAHEAD_COST when lookahead used | Ensures lookahead cost is precisely applied | Creates two identical states, scores with/without lookahead, compares goal diff |
| `test_all_dimensions_in_range` | All 5 dimensions (goal, trust, info, risk, causal) stay in [0.0, 1.0] | Bounds checking for all reward components | Runs 20 random trials, checks each dimension |
| `test_causal_score_deterministic` | Same graph + same target = identical r^causal | Ensures causal scoring is deterministic, not random | Scores two identical actions, asserts equality |
| `test_utterance_score_defaults` | Default UtteranceScore values are all 0.0 | Validates dataclass defaults | Direct field assertions |
| `test_weighted_sum` | weighted_sum(REWARD_WEIGHTS) returns correct scalar | Tests weighted combination of 5 dimensions | Computes expected weighted sum, compares |
| `test_to_dict` | to_dict() returns correct dict | Validates serialization | Direct dict key/value assertions |

---

### `test_observation_mechanism.py` — Environment Observation System

**What**: Tests the observation mechanism including the 5 observable signals, noise properties, and hidden field exclusion.

**Why**: The agent must receive structured observations without access to hidden state (causal graph, true beliefs). The observation system must correctly implement noise, echoes, weak signals, and veto precursors.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_g_never_in_observation` | No 'graph' or 'causal_graph' attribute in observation | **P1: G hidden from agent** | Checks hasattr for graph/causal_graph/G |
| `test_g_not_in_string_fields` | No G info embedded in string fields | Deep hidden field check | Scans string fields for edge weight patterns |
| `test_engagement_history_length` | engagement_history has exactly 5 values per stakeholder | Engagement window size | Resets, steps, checks history structure |
| `test_engagement_not_cancellable` | Agent cannot recover true delta from engagement deltas | **P6: Noise not cancellable** | Checks engagement_delta is noisy/uncomputable |
| `test_reset_clears_all_state` | Reset reinitializes all accumulators | State reset behavior | Two episodes, checks different engagement levels |
| `test_episode_seed_reproducibility` | Same seed = same initial engagement | Reproducibility guarantee | Two envs, same seed, asserts identical obs |
| `test_weak_signal_structure` | weak_signals is dict of string lists | Signal structure | Direct type checks |
| `test_weak_signals_after_action` | weak_signals populated after actions | Signal generation | Sends document, checks non-empty |
| `test_cross_echoes_structure` | Echoes is list of {from, to, content} dicts | Echo structure | Direct type and key checks |
| `test_echo_recall_rate` | ~70% echo recall rate (1.5-4.5 avg) | **P7: Echo mechanism active** | 100 episodes, counts echoes per action |
| `test_veto_precursors_structure` | veto_precursors dict mapping to strings | Precursor warning structure | Direct type checks |
| `test_veto_precursors_dont_expose_tau` | No tau values in veto precursors | Hidden parameter invariant | 20 episodes, scans for "tau" in warnings |
| `test_all_five_signals_present` | All 5 signals present | Complete observation | Checks all signal field names |
| `test_observation_schema_complete` | All required fields present | Schema completeness | Checks 18 required fields |
| `test_stakeholder_messages_after_targeted_action` | Messages populated after targeted action | Message generation | Direct message content check |
| `test_engagement_delta_noise_present` | engagement_level_delta has noise | Noise presence | Checks delta is not None |
| `test_round_number_increments` | round_number increments after each step | Round tracking | Steps 3 times, checks 0→1→2→3 |
| `test_done_flag_after_max_rounds` | done=True after max_rounds | Terminal condition | 10 steps, checks done on step 10 |

---

### `test_validator.py` — Action Validation

**What**: Tests the output validator that parses and validates agent actions.

**Why**: Actions must be normalized and validated before being applied to the environment.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_validator_normalizes_dynamic_targets` | "target": "finance" normalized to target_ids: ["finance"] | Dynamic target normalization | Validates JSON with string target, checks normalization |
| `test_validator_soft_rejects_unknown_target` | Unknown target marked as malformed_action=True | Graceful handling of invalid targets | Uses unknown target, checks error fields set |
| `test_validator_filters_proposed_terms` | Unknown fields filtered from proposed_terms | Security/validation of term proposals | Sends with junk field, checks only "price" retained |

---

### `test_models.py` — Pydantic Model Validation

**What**: Tests core Pydantic model validation and serialization.

**Why**: Models are the contract between components. Validation ensures type safety.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_action_target_ids_are_deduplicated` | Duplicate targets removed | Deduplication invariant | Creates action with duplicates, checks deduplication |
| `test_state_is_callable_for_state_and_state_method_compat` | state() returns state | Callable compatibility | Direct call assertion |
| `test_observation_supports_dynamic_fields` | Dynamic fields accepted in observation | Extensibility | Creates observation with dynamic fields |
| `test_reward_model_tracks_value_done_and_info` | Reward tracks value, done, info dict | Reward interface | Direct field assertions |

---

### `test_grader.py` — CCI Grading

**What**: Tests the CCIGrader for deal closure scoring.

**Why**: The grader provides an independent evaluation of deal quality.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_grader_returns_zero_for_infeasible_close` | Score = MIN_SCORE when infeasible | Infeasibility penalty | Creates state with feasibility_state.is_feasible=False |
| `test_grader_returns_positive_for_feasible_close` | Feasible close returns score in (0, 1) | Reward for successful deal | Creates feasible state, asserts 0 < score < 1 |
| `test_grader_returns_zero_when_constraint_unresolved` | MIN_SCORE when budget_ceiling unresolved | Constraint resolution requirement | Sets hidden_constraints["budget_ceiling"]["resolved"] = False |
| `test_grader_score_is_strictly_inside_open_interval` | Both feasible and infeasible scores are strictly inside (0,1) | Bounded score requirement | Checks both scores |
| `test_grader_penalizes_relationship_damage` | Permanent marks reduce score | Relationship damage penalty | Compares clean vs damaged state scores |

---

### `test_claims.py` — Commitment Ledger

**What**: Tests the commitment ledger for tracking negotiation claims.

**Why**: Claims must be tracked to detect contradictions.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_commitment_ledger_flags_numeric_contradiction` | Contradiction detected for conflicting numeric claims | Detecting inconsistent commitments | Ingests two claims with conflicting price values |
| `test_commitment_ledger_trims_history` | History trimmed to max_claims | Memory management | Creates ledger with max_claims=3, ingests 5, checks len=3 |

---

### `test_semantics.py` — Semantic Analysis

**What**: Tests the semantic analyzer for extracting claims and artifacts.

**Why**: Semantic analysis extracts structured information from natural language.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_semantic_analyzer_extracts_claims_and_artifacts` | Extracts price, timeline_weeks slots and roi_model artifact | Claim extraction from text | Analyzes text with ROI model, checks slots and artifacts |
| `test_semantic_analyzer_returns_known_backend` | Returns known backend (embedding/tfidf/lexical) | Backend identification | Analyzes text, checks backend value |

---

## Integration Tests (`tests/integration/`)

### `test_api_sessions.py` — Session Management

**What**: Tests FastAPI session management and isolation.

**Why**: Multiple clients must maintain separate sessions without interference.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_state_requires_active_session` | /state returns 400 without reset | Session requirement | Calls /state without session |
| `test_sessions_are_isolated_by_client_cookie` | Different clients maintain separate sessions | Session isolation | Two clients reset to different scenarios, verifies isolation |

---

### `test_web_ui.py` — Web Interface Routing

**What**: Tests web UI routing and health endpoints.

**Why**: The web interface must be accessible and properly routed.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_root_redirects_to_web` | / redirects to /web | Root routing | Checks 302 redirect location |
| `test_web_page_exposes_wrapper_without_redirect` | /web serves page with iframe and gradio | Web interface serving | Checks 200, body contains iframe/gradio/dealroom |
| `test_web_slash_page_redirects_to_web` | /web/ redirects to /web | Trailing slash handling | Checks redirect |
| `test_ui_blocked_direct_access` | /ui/ redirects to /web | Direct UI access blocked | Checks redirect to /web |
| `test_health_endpoint_still_works` | /health returns 200 | Health monitoring | Checks response |

---

### `test_end_to_end.py` — Full Episode Tests

**What**: Tests complete episodes from reset to terminal state.

**Why**: End-to-end tests verify the complete RL loop works without crashes.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_full_episode_runs_without_crash` | 5-step episode completes without error | Basic e2e functionality | Reset + 5 steps, checks types |
| `test_reward_vector_all_five_dimensions` | Reward accessible and >= 0 | Reward structure | Checks reward is numeric and non-negative |
| `test_environment_reset_produces_valid_observation` | reset() produces valid observation | Reset correctness | Checks initial observation fields |
| `test_veto_flag_possible` | Veto can trigger during episode | Veto mechanism | Runs 10 steps in hostile scenario |
| `test_lookahead_action_does_not_advance_state` | Lookahead actions don't advance round | Lookahead behavior | Sets lookahead, checks round doesn't advance |
| `test_lookahead_reduces_goal_score` | Lookahead reduces goal by 0.07 | Lookahead cost | Compares with/without lookahead |
| `test_aligned_scenario_runs` | Aligned scenario completes | Scenario completeness | 10 steps in aligned |
| `test_conflicted_scenario_runs` | Conflicted scenario completes | Scenario completeness | 10 steps in conflicted |
| `test_hostile_acquisition_scenario_runs` | Hostile acquisition completes | Scenario completeness | 10 steps in hostile |
| `test_send_document_action` | send_document action works | Action type coverage | Sends DPA document |
| `test_multiple_targets` | Multi-target action works | Group action handling | Sends to Legal + Finance |
| `test_reward_non_negative` | Reward is numeric | Type integrity | 5 steps, checks reward type |
| `test_info_dict_contains_debug_info` | Info dict populated | Debug information | Checks info is dict |

---

## V3 Tests (`tests/v3/`) — Container-Required

These tests require the Docker container running with the FastAPI server.

### `test_00_environment_setup.py` — Environment Prerequisites

**What**: Validates environment prerequisites (Python deps, API keys, Docker, server endpoints).

**Why**: Ensures all dependencies and runtime requirements are met before running other tests.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_0_1_python_deps` | requests, numpy, dotenv available | Dependency check | Imports each dep |
| `test_0_2_api_keys_configured` | Optional API keys reported | API key discovery | Checks env vars |
| `test_0_3_docker_container_running` | Docker container running | Runtime requirement | docker ps check |
| `test_0_4_server_endpoints_responsive` | /health, /metadata, /reset, /step respond | Endpoint availability | HTTP requests to each endpoint |
| `test_0_5_llm_client_imports` | llm_client module importable | LLM client availability | Imports from deal_room.environment.llm_client |

---

### `test_01_schema_validation.py` — Observation Schema

**What**: Validates observation schema completeness and hidden field exclusion.

**Why**: Ensures the agent receives correct information without access to hidden state (P1).

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_1_1_all_required_fields_present` | All 18 required fields present | Schema completeness | POST /reset, checks field list |
| `test_1_2_no_hidden_fields_exposed` | Hidden fields (G, tau, B_i, etc.) not exposed | **P1: G hidden** | Recursive dict search for hidden field names |
| `test_1_3_field_types_correct` | Field types match specification | Type safety | Step action, checks types |
| `test_1_4_engagement_history_window_size` | engagement_history >= 5 entries | Window size requirement | Checks history length |
| `test_1_5_engagement_level_delta_single_float` | delta is single float, not dict | **P5: Lookahead cost precision** | Checks delta is numeric |
| `test_1_6_cross_stakeholder_echoes_is_list` | Echoes is list of dicts with from field | Echo structure | Checks type and required keys |
| `test_1_7_stakeholder_messages_populated_after_step` | Messages populated after step | Message generation | Sends doc, checks messages dict |
| `test_1_8_action_schema_accepts_lookahead` | Lookahead action schema accepted | Lookahead feature | Sends action with lookahead nested |
| `test_1_9_approval_path_progress_structure` | Progress has 'band' field with valid values | Approval tracking | Checks band values |
| `test_1_10_deal_stage_valid_transitions` | round_number increments, stage in valid values | Round tracking | Steps, checks round and stage |
| `test_1_11_documents_field_format` | Documents formatted as [{name, content}, ...] | Document handling | Sends multi-doc action |

---

### `test_02_reward_integrity.py` — Reward Integrity

**What**: Reward integrity and "unhackability" tests validating reward bounds, lookahead cost, determinism.

**Why**: The reward system must be deterministic, bounded, and resistant to gaming.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_2_1_reward_is_single_float` | Reward is numeric in [0, 1], not dict | **P4: Single scalar reward** | Checks reward type and range |
| `test_2_2_lookahead_cost_is_exactly_007` | Goal cost exactly 0.07 (within 0.015) | **P5: Exact lookahead cost** | Compares goal dimension with/without lookahead |
| `test_2_3_reward_in_range_after_valid_actions` | All reward components in [0, 1] | Dimension bounds | Multiple actions, checks component ranges |
| `test_2_4_deterministic_reward_with_seed` | Same seed/action → same reward | **Determinism requirement** | 3 trials with same seed, checks spread |
| `test_2_5_repeat_same_action_does_not_escalate_reward` | Repeating same action doesn't inflate reward | Reward stability | 3 repeats, checks trend <= 0.01 |
| `test_2_6_different_targets_different_causal_scores` | Different targets produce different causal scores | **P9: Causal varies with target** | 5 seeds × 3 targets, checks score variance |
| `test_2_7_informative_action_outperforms_empty` | Substantive action >= empty - 0.1 | Action quality reward | Compares empty vs substantive action |
| `test_2_8_reward_non_trivial_variance` | Reward spread > 0.05 across actions | Learning signal variance | Multiple action types, checks spread |
| `test_2_9_good_documentation_higher_than_poor` | Good docs reward >= poor docs | **Documentation value** | Compares poor vs good doc actions |
| `test_2_10_lookahead_improves_prediction_accuracy` | Lookahead accuracy > 60% | Lookahead effectiveness | 20 runs, checks mean accuracy |

---

### `test_03_causal_inference.py` — Causal Inference Signals

**What**: Validates causal inference signals including engagement changes, cross-stakeholder echoes, noise properties.

**Why**: The agent must be able to infer causal structure from observations (P7, P8, P9).

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_3_1_targeted_stakeholder_engagement_changes` | Engagement delta is numeric | Signal generation | Targets Finance, checks delta type |
| `test_3_2_cross_stakeholder_echoes_detected` | Echoes in >= 65% episodes | **P7: Echo mechanism** | 12 episodes, counts echo rate |
| `test_3_3_engagement_history_window_slides` | History window stable across steps | Window maintenance | 3 steps, checks history length constant |
| `test_3_4_engagement_noise_not_cancellable` | >= 2/5 non-zero deltas (noise active) | **P6: Noise not cancellable** | 5 steps, counts non-zero deltas |
| `test_3_5_different_targets_different_patterns` | Finance vs Legal produce different deltas | **P9: Target discrimination** | 4 each, compares mean delta distance |
| `test_3_6_weak_signals_for_non_targeted` | Weak signals in >= 20% episodes | **P8: Weak signals** | 12 episodes, checks signal rate |
| `test_3_7_echo_content_structure` | Echoes have correct {from, to, content} structure | Echo schema | Checks echo structure |
| `test_3_8_causal_signal_responds_to_graph_structure` | Hub (ExecSponsor) > leaf (Procurement) impact | **P9: Graph structure response** | 8 each, compares impact metric |

---

### `test_04_cvar_veto.py` — CVaR Veto Mechanism

**What**: Validates CVaR veto mechanism and scenario difficulty differentiation.

**Why**: CVaR-based veto is a core research property (P3). The environment must correctly identify veto precursors and trigger vetos appropriately.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_4_1_veto_precursor_before_veto` | Veto precursor seen before veto fires | Early warning requirement | Aggressive actions in hostile, checks precursor before veto |
| `test_4_2_aligned_no_early_veto` | Aligned scenario doesn't veto on first step | Scenario correctness | Single step in aligned, checks not done |
| `test_4_3_veto_terminates_episode` | Hostile + aggressive produces veto terminal | Veto termination | 20 aggressive steps in hostile |
| `test_4_3_veto_deterministic` | Legal veto fires by step 5 with seed=42 | Deterministic veto | Fixed seed, aggressive to Legal, checks terminal_category |
| `test_4_4_timeout_terminates` | Episode terminates after max_rounds | Timeout termination | max_rounds + 2 steps |
| `test_4_5_scenario_difficulty_differentiation` | Hostile reaches veto pressure earlier than aligned | **Scenario difficulty ordering** | Measures first pressure round for each |
| `test_4_6_veto_precursor_is_stakeholder_specific` | Precursors tied to specific stakeholders | Precursor specificity | 15 aggressive steps, collects precursor stakeholders |
| `test_4_7_cveto_not_just_eu` | CVaR veto fires even when EU > 0 | **P3: CVaR veto despite EU > 0** | 18 aggressive steps in hostile, checks veto fired |

---

### `test_05_episode_isolation.py` — Episode Isolation

**What**: Episode isolation and determinism tests.

**Why**: Different seeds must produce different states, and reset must fully clear state.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_5_1_different_seeds_different_initial_state` | Different seeds → different engagement levels | **P2: Reset regenerates G** | Two resets, compares engagement level diff |
| `test_5_2_round_number_resets_to_zero` | Reset sets round_number=0 | State reset | 3 steps, reset, checks round=0 |
| `test_5_3_done_false_after_reset` | done=False immediately after reset | Initial terminal state | 3 scenarios, checks done=False |
| `test_5_4_engagement_history_initialized` | History initialized with >= 5 entries | Window initialization | Checks history length |
| `test_5_5_round_number_increments_correctly` | 0 → 1 → 2 after steps | Round increment | 5 steps, checks round counter |
| `test_5_6_all_three_scenarios_work` | aligned, conflicted, hostile_acquisition all work | Scenario coverage | One step in each scenario |
| `test_5_7_same_session_same_state_across_steps` | Session state consistent within episode | Session persistence | 2 steps, checks round increment |
| `test_5_8_reset_clears_all_session_state` | Reset clears round_number, done, history | Full state reset | 3 steps, reset, checks all reset |

---

### `test_06_probabilistic_signals.py` — Probabilistic Signal Mechanisms

**What**: Container-side tests for probabilistic signal mechanisms (weak signals, cross-stakeholder echoes).

**Why**: Validates P7 (echoes) and P8 (weak signals) are present and correctly formatted.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_6_1_weak_signals_field_exists` | weak_signals field exists in observation | **P8: Weak signals present** | Directly instantiates env, checks hasattr |
| `test_6_2_cross_stakeholder_echoes_exists` | cross_stakeholder_echoes field exists | **P7: Echoes present** | Checks hasattr |
| `test_6_3_echo_structure` | Echoes list of dicts with from field | Echo schema | Calls _generate_cross_stakeholder_echoes() |
| `test_6_4_weak_signals_populated_after_action` | weak_signals dict populated after step | Signal population | Sends DPA doc, checks dict non-empty |
| `test_6_5_echo_firing_rate_nonzero` | Echoes in >= 65% episodes | **P7: Echo recall ~70%** | 40 episodes, counts firing rate |
| `test_6_6_weak_signal_threshold_respected` | Signal values are list[str] tags | Format validation | Checks value types |
| `test_6_7_echo_recall_probability_configured` | echo_recall_probability in [0.65, 0.75] | Design target tolerance | Checks OBS_CONFIG.echo_recall_probability |

---

### `test_07_causal_graph.py` — Container-Side Causal Graph

**What**: Container-side unit tests for causal graph core properties.

**Why**: Validates P9 (causal varies with target) and P10 (every G unique) within the container environment.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_7_1_propagation_direction` | Signal flows A→B along edge | Propagation correctness | A→B edge, applies delta to A |
| `test_7_2_signal_carries_through_edge` | B's belief changes when A changes | Edge transmission | Applies delta to A, checks B changed |
| `test_7_3_damping_prevents_runaway` | Dense graph beliefs stay in (0,1) after 5 steps | Damping verification | Fully connected, 5 steps |
| `test_7_4_beliefs_normalized_after_propagation` | All beliefs sum to 1.0 after propagation | Normalization invariant | Standard 5-graph propagation |
| `test_7_5_no_self_loops` | No self-loop edges | Invariant check | All 3 scenarios checked |
| `test_7_6_exec_sponsor_outgoing_authority` | ExecSponsor has >= 2 outgoing edges | Authority hub invariant | 10 scenarios × 10 seeds |
| `test_7_7_hub_centrality_beats_leaf` | Hub betweenness > leaf | Centrality property | Star graph centrality |
| `test_7_8_graph_identifiability` | 20 sampled graphs pairwise unique | **P10: Every G unique** | Samples 20 graphs, computes signatures, all pairs unique |
| `test_7_9_hub_node_has_higher_centrality_impact` | Hub impact >= 30% higher than leaf | **P9: Hub impact** | 10 seeds, compares hub/leaf signature impact |

---

### `test_08_cvar_preferences.py` — Container-Side CVaR

**What**: Container-side CVaR unit tests validating core research claims.

**Why**: Validates P3 (CVaR veto despite EU > 0) and CVaR formula correctness.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_8_1_core_claim` | EU>0 AND CVaR>tau → veto fires | **P3: CVaR veto despite EU>0** | Evaluates deal without DPA for Legal |
| `test_8_2_good_docs_lower_cvar_than_poor` | Full docs CVaR < poor docs | **Documentation value** | Compares CVaR with/without docs |
| `test_8_3_cvar_formula_correct` | High outcomes → low CVaR | CVaR formula correctness | Tests on [1.0×5, 0.5×5] |
| `test_8_4_tau_ordering` | Legal.tau < Finance.tau < ExecSponsor.tau | Risk sensitivity ordering | Direct value comparison |
| `test_8_5_aggressive_timeline_higher_cvar` | 4-week CVaR > 16-week CVaR for TechLead | Timeline risk effect | Compares CVaR for short vs long timeline |
| `test_8_6_cvar_subadditivity_sanity` | CVaR <= max outcome (coherence) | Risk measure coherence | Tests on [0.3, 0.4, 0.5, 0.6, 0.9] |

---

### `test_09_full_episode_e2e.py` — Full Episode Validation

**What**: Full episode end-to-end validation across scenarios and strategies.

**Why**: Ensures complete episodes work without crashes and produce meaningful terminal outcomes.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_9_1_aligned_completes` | Aligned + neutral completes | Scenario completeness | 20 steps |
| `test_9_2_conflicted_completes` | Conflicted + comprehensive completes | Scenario completeness | 20 steps |
| `test_9_3_hostile_aggressive_produces_veto_or_timeout` | Hostile + aggressive: valid terminal (veto/timeout/error) | Terminal validation | 15 aggressive steps |
| `test_9_4_reward_collected_across_episode` | Reward entries collected | Reward tracking | 15 steps, counts rewards |
| `test_9_5_reward_variance_non_trivial` | Reward spread > 0.05 | Learning signal variance | 12 steps, checks spread |
| `test_9_6_strategy_comparison_possible` | Comprehensive > aggressive + 0.15 | Strategy differentiation | Compares 3 seeds each |
| `test_9_7_terminal_outcome_meaningful` | Non-empty terminal outcomes across scenarios | Terminal meaningfulness | 9 (3×3) combinations |
| `test_9_8_multidocument_action_works` | Multi-doc action returns reward | Document handling | Sends DPA+cert+liability |

---

### `test_10_training_integration.py` — Training Effectiveness

**What**: Validates that training actually improves policy performance over random baseline.

**Why**: Ensures the training system produces real improvement, not just noise.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_training_actually_improves` | Trained policy beats random by >0.15 | **Training effectiveness** | Trains 4 episodes, evaluates both policies, compares weighted_reward |

---

### `test_P0_comprehensive.py` — P0 Critical Issues

**What**: P0 critical issues validation from grill-me session.

**Why**: Validates training improvement, CVaR math, lookahead usefulness, and debug traces.

**How**:
| Test | What It Tests | Category |
|------|--------------|----------|
| `test_P0_1a_baseline_vs_trained_comparison` | Trained > random by >0.10 | P0-1: Training |
| `test_P0_1b_multi_episode_improvement` | Later batches >= earlier | P0-1: Training |
| `test_P0_1c_dimension_wise_improvement` | >=2 dimensions improve with training | P0-1: Training |
| `test_P0_1d_policy_persistence` | Save/load preserves trained state | P0-1: Training |
| `test_P0_2a_cvar_deterministic_calculation` | CVaR formula: worst 40% mean | P0-2: CVaR math |
| `test_P0_2b_cvar_veto_with_positive_eu` | CVaR veto in hostile scenarios | P0-2: CVaR math |
| `test_P0_2c_cvar_per_stakeholder` | Different profiles → different CVaR | P0-2: CVaR math |
| `test_P0_3a_lookahead_improves_decisions` | Lookahead >= -0.05 vs no-lookahead | P0-3: Lookahead |
| `test_P0_3b_lookahead_prediction_accuracy` | Accuracy > 55% | P0-3: Lookahead |
| `test_P0_3c_lookahead_cost_exactly_007` | (SKIPPED - test design error) | P0-3: Lookahead |
| `test_P0_4a_belief_state_trace` | 5 steps traced with rewards | P0-4: Debug |
| `test_P0_4b_cvar_breakdown_per_stakeholder` | CVaR per stakeholder computed | P0-4: Debug |
| `test_P0_4c_action_effect_trace` | Stage/momentum changes tracked | P0-4: Debug |
| `test_P1_stochastic_stabilized` | >=2/5 vetoes in hostile with fixed seeds | P1: Stochastic |
| `test_P2_adversarial_degenerate_graph` | No-edge graph episode completes | P2: Adversarial |
| `test_P2_adversarial_extreme_tau` | Extreme tau handled without crash | P2: Adversarial |

---

### `test_assertion_hygiene.py` — AST Analysis

**What**: Guard against print-only critical tests using AST analysis.

**Why**: Ensures all critical tests have actual assertions, not just console output.

**How**:
| Test | What It Tests | Why | How |
|------|--------------|-----|-----|
| `test_critical_v3_tests_have_assertions_or_explicit_failures` | No test function lacks assert/raise/pytest.fail | Test quality assurance | Parses AST for critical v3 test files |

---

## End-to-End Tests (`tests/e2e/`)

### `test_workflows.py`

**What**: End-to-end workflow tests using inference.run_task().

**Why**: Validates complete workflows from inference script.

### `test_playwright_e2e.py`

**What**: Playwright E2E tests for web interface using real browser automation.

**Why**: Ensures web interface works in real browser.

---

## Summary Statistics

| Test Category | Count | Purpose |
|--------------|-------|---------|
| Unit Tests | 121 | Component-level validation |
| Integration Tests | 20 | API and web interface |
| V3 Tests (container) | ~100+ | Research property validation |
| E2E Tests | ~10 | Full workflow validation |

**Total**: ~250 tests covering the complete DealRoom S2P V3 system.