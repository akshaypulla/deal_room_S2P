# DealRoom

An OpenEnv-compatible reinforcement learning environment for enterprise B2B software negotiation. The agent acts as a vendor-side negotiator navigating a multi-stakeholder buying committee to close a deal.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation and Problem Context](#motivation-and-problem-context)
3. [System Architecture](#system-architecture)
4. [Environment Design (RL Formulation)](#environment-design-rl-formulation)
5. [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
6. [Implementation Details](#implementation-details)
7. [Installation and Setup](#installation-and-setup)
8. [Usage](#usage)
9. [Example Outputs](#example-outputs)
10. [Project Structure](#project-structure)
11. [Extensibility](#extensibility)
12. [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

DealRoom simulates the negotiation dynamics of an enterprise software deal. The agent (vendor) must navigate a committee of stakeholders—Legal, Finance, TechLead, Procurement, Operations, and ExecSponsor—each with distinct priorities, veto authority, and influence relationships.

The environment implements a sequential decision-making problem where:

- The state is partially observable (hidden constraints, private stakeholder beliefs)
- Actions have delayed and indirect effects (committee deliberation, belief propagation)
- Multiple stakeholders must be individually addressed before a deal can close
- Irreversible actions (trust damage) create path dependencies

Key capabilities:
- **Causal graph modeling**: Stakeholder influence relationships are modeled as a weighted graph
- **Bayesian belief tracking**: Each stakeholder maintains beliefs about the vendor (competent/incompetent, trustworthy/deceptive, aligned/misaligned)
- **Committee deliberation**: Multi-step belief propagation through the influence graph
- **CVaR risk modeling**: Risk-averse stakeholders (Legal, Finance) have CVaR preferences over deal outcomes
- **Adaptive curriculum**: Training scenarios adjust difficulty based on failure patterns

---

## Motivation and Problem Context

Enterprise software procurement is a multi-stakeholder negotiation process, not a single purchase decision. A deal typically requires:

1. **Technical validation** (TechLead, Operations)
2. **Financial approval** (Finance, CFO)
3. **Legal review** (Legal, Compliance)
4. **Procurement oversight** (Procurement)
5. **Executive sign-off** (ExecSponsor, CEO)

Each stakeholder has veto power at different stages, and hidden constraints (budget ceilings, compliance requirements, delivery windows) can surface late and collapse a near-closed deal.

**Why this is challenging for RL:**

| Property | Implication |
|----------|-------------|
| Partial observability | Hidden constraints must be discovered through targeted actions |
| Long-horizon | 10-round episodes require credit assignment across many steps |
| Multi-agent | 6 stakeholders with competing utilities must all reach alignment |
| Irreversible damage | Trust violations cannot be undone within an episode |
| Sequential lobbying | Wrong stakeholder order can create blocking coalitions |
| Delayed feedback | Actions in early rounds affect outcomes many steps later |

The problem is not solvable by local pattern matching—understanding committee dynamics and strategic sequencing is essential.

---

## System Architecture

### High-Level Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AGENT / CLIENT                                   │
│                    (LLM-based policy or heuristic adapter)                    │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │ DealRoomAction
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI SERVER (server/app.py)                        │
│                                                                               │
│   ┌─────────┐   ┌──────────────┐   ┌─────────────────────────────────────┐  │
│   │ /reset  │   │   /step      │   │         /__gradio_ui__/             │  │
│   │ /state  │   │              │   │     (Gradio Web Interface)          │  │
│   │ /health │   │              │   │                                     │  │
│   └─────────┘   └──────────────┘   └─────────────────────────────────────┘  │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │ session_pool.py
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    DEALROOM SESSION POOL                                      │
│              (manages per-session environment instances)                      │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     DEALROOMV3 (deal_room/environment/dealroom_v3.py)         │
│                                                                               │
│   ┌────────────┐   ┌─────────────────┐   ┌────────────────────────────────┐  │
│   │ reset()   │   │    step()       │   │        _build_observation()    │  │
│   │           │──▶│                 │──▶│                                │  │
│   └────────────┘   └─────────────────┘   └────────────────────────────────┘  │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                    CORE ENVIRONMENT COMPONENTS                       │    │
│   │                                                                       │    │
│   │  ┌───────────────┐   ┌────────────────┐   ┌────────────────────┐   │    │
│   │  │  CausalGraph  │   │ BeliefTracker  │   │ DeliberationEngine │   │    │
│   │  │ (causal_graph)│   │(belief_tracker)│   │ (deliberation_engine│   │    │
│   │  └───────────────┘   └────────────────┘   └────────────────────┘   │    │
│   │         │                  │                       │                 │    │
│   │         ▼                  ▼                       ▼                 │    │
│   │  ┌──────────────────────────────────────────────────────────────┐   │    │
│   │  │              UtteranceScorer (rewards/utterance_scorer.py) │   │    │
│   │  │   goal | trust | information | risk | causal → weighted sum │   │    │
│   │  └──────────────────────────────────────────────────────────────┘   │    │
│   │         │                                                        │    │
│   │         ▼                                                        │    │
│   │  ┌──────────────────────────────────────────────────────────────┐   │    │
│   │  │                 DealRoomObservation                          │   │    │
│   │  │   stakeholders, messages, engagement, veto_precursors, etc.  │   │    │
│   │  └──────────────────────────────────────────────────────────────┘   │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Agent-Environment Interaction Flow

```
Agent                          DealRoomV3 Environment
  │                                      │
  │  reset(task_id="conflicted")        │
  ├─────────────────────────────────────▶│
  │                                      │  1. Sample causal graph
  │                                      │  2. Initialize beliefs per stakeholder
  │                                      │  3. Initialize noisy engagement levels
  │◀─────────────────────────────────────│
  │        DealRoomObservation           │
  │                                      │
  │  step(DealRoomAction)                │
  ├─────────────────────────────────────▶│
  │                                      │  1. Normalize action
  │                                      │  2. Apply to offer state
  │                                      │  3. Update deal stage
  │                                      │  4. Bayesian belief update per stakeholder
  │                                      │  5. Committee deliberation (3 steps)
  │                                      │  6. Belief propagation through graph
  │                                      │  7. Generate stakeholder responses
  │                                      │  8. Score utterance (5-dim reward)
  │                                      │  9. Check veto conditions
  │                                      │  10. Build observation
  │◀─────────────────────────────────────│
  │        (observation, reward, done, info)
  │
  ▼
(repeat until done=True or max_rounds reached)
```

### Committee Deliberation Subsystem

```
Vendor Action
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│              CommitteeDeliberationEngine                         │
│                                                                  │
│   ┌─────────────────┐    ┌─────────────────┐                    │
│   │ Propagate Beliefs│    │ Compute Vote    │                    │
│   │ (3 deliberation  │───▶│ (per stakeholder│                    │
│   │  steps)          │    │  approve/block) │                    │
│   └─────────────────┘    └─────────────────┘                    │
│          │                        │                              │
│          ▼                        ▼                              │
│   ┌─────────────────┐    ┌─────────────────┐                    │
│   │ ExecSponsor     │    │ Check Silent    │                    │
│   │ Activation      │    │ Period          │                    │
│   │ (latent node)    │    │                 │                    │
│   └─────────────────┘    └─────────────────┘                    │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────┐                                          │
│   │ DeliberationResult│                                         │
│   │ • updated_beliefs│                                          │
│   │ • committee_vote │                                         │
│   │ • summary_dialogue│                                         │
│   │ • exec_sponsor_activated                                 │
│   └─────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
Beliefs updated → Stakeholder responses generated
```

---

## Environment Design (RL Formulation)

### State Space

**Internal State** (`DealRoomState`):
- `stakeholders`: Public role information per stakeholder
- `stakeholder_private`: Private tracks per stakeholder (trust, approval, perceived_fit, private_resistance)
- `hidden_constraints`: Constraints not yet discovered
- `offer_state`: Current proposed commercial terms
- `deal_stage`: One of {evaluation, negotiation, legal_review, final_approval, closed}
- `active_blockers`: Stakeholders currently blocking progress

**Observation** (`DealRoomObservation`):
- `round_number`, `max_rounds`: Episode progress
- `stakeholders`: Active roster with roles
- `stakeholder_messages`: Visible messages from stakeholders
- `engagement_level`: Noisy proxy for stakeholder movement (σ=0.03)
- `weak_signals`: Indirect hints about hidden blockers
- `known_constraints`: Discovered constraints
- `deal_momentum`: {progressing, stalling, critical}
- `veto_precursors`: Warning signals before veto
- `committee_vote`: Current committee vote status
- `exec_sponsor_activated`: Boolean flag for exec involvement

**Belief State** (per stakeholder, not directly observable):
- Distribution over 6 vendor types: {competent, incompetent, trustworthy, deceptive, aligned, misaligned}
- Confidence score tracking certainty

### Action Space

**Action Types**:
| Action | Description |
|--------|-------------|
| `direct_message` | Send targeted message to a stakeholder |
| `send_document` | Share artifact (DPA, ROI, security cert, timeline) |
| `backchannel` | Informal check-in or coordination move |
| `group_proposal` | Propose terms to multiple/all stakeholders |
| `concession` | Offer ground on terms or process |
| `walkaway_signal` | Signal risk of disengagement |
| `reframe_value_prop` | Reposition value proposition |
| `exec_escalation` | Push toward executive attention |

**Action Fields**:
```python
class DealRoomAction(BaseModel):
    action_type: str           # Action family
    target: str               # Target stakeholder or "all"
    target_ids: List[str]     # Explicit recipient IDs
    message: str              # Communication content (max 1200 chars)
    documents: List[dict]     # Attached artifacts
    proposed_terms: dict      # Structured commercial terms
    channel: str              # Communication channel
    mode: str                 # Communication style
    lookahead: LookaheadRequest  # Optional lookahead simulation
```

### Transition Dynamics

```
State_t → Action → State_{t+1}

Components:
1. Apply action to offer_state
2. Update deal_stage if stage gate threshold passed
3. Bayesian belief update per stakeholder
   - P(belief | action, role, targeting)
   - Targeted actions have stronger effect
   - Document sharing has informational effect
4. Multi-step belief propagation through causal graph
   - 3 deliberation steps (4 for hostile_acquisition)
   - Each step: beliefs diffuse along edges weighted by authority
5. Noisy engagement update
   - True belief change + Gaussian noise (σ=0.03)
   - Echo recall: previous engagement affects current observation
6. Stakeholder response generation
   - Based on updated beliefs and archetype preferences
7. Veto check
   - Hard veto: resistance > threshold for stage
   - Soft veto: CVaR risk exceeds threshold
```

### Reward Function

**5-Dimensional Utterance Score** (deterministic, no LLM calls):

| Dimension | Weight | Computation |
|-----------|--------|-------------|
| `goal` | 0.25 | Δbelief_positive_mass + Δresolved_blockers + CVaR_headroom_change |
| `trust` | 0.20 | Δtrustworthy_mass for targeted stakeholder |
| `info` | 0.20 | H(B_before) - H(B_after) (entropy reduction) |
| `risk` | 0.20 | -CVaR_delta across risk-averse stakeholders |
| `causal` | 0.15 | Betweenness centrality of targeted node in causal graph |

**Terminal Rewards**:
| Outcome | Reward |
|---------|--------|
| Deal closed | +1.0 |
| Hard veto | -1.0 |
| Soft veto | -0.8 |
| Stage regression | -0.75 |
| Timeout (max rounds) | -0.5 |

**Step penalty**: -0.01 per step to encourage efficiency

### Episode Definition

- **Reset**: `env.reset(task_id, seed)` → initial observation
- **Step**: `env.step(action)` → (observation, reward, done, info)
- **Done conditions**:
  - Deal closed successfully
  - Veto triggered (hard or soft)
  - Max rounds reached (10)
  - Stage regression beyond threshold
- **Max episode length**: 10 rounds

### Deal Stages

```
evaluation ──▶ negotiation ──▶ legal_review ──▶ final_approval ──▶ closed
    │              │               │                │
    └──────────────┴───────────────┴────────────────┘
                    (can regress)
```

Stage gates use theta thresholds:
- `θ_pass = 0.65` for advancement
- `θ_stall = 0.40` for regression warning

---

## Design Decisions and Trade-offs

### Reward Shaping Strategy

**Why 5 dimensions instead of single scalar:**
- Single reward obscures which behavioral dimension an agent is exploiting
- Multi-dimensional reward provides more gradient signal per step
- Prevents reward hacking on one dimension while ignoring others

**Why deterministic scoring (no LLM):**
- LLM-based scoring is non-deterministic across runs
- Environment must be reproducible for RL training
- Fast scoring enables many environment steps per second

### Causal Graph Modeling

**Why model stakeholder influence explicitly:**
- Real procurement involves committee dynamics—stakeholders discuss before voting
- Beliefs propagate through informal networks, not just direct vendor contact
- ExecSponsor activation is latent until triggered by escalation or veto

**Trade-off: Graph sampling vs. fixed structure**
- Randomly sampled graphs per episode add variety but reduce comparability
- Fixed graph would be more reproducible but less realistic

### Hidden Constraints

**Why not reveal constraints directly:**
- Real negotiation involves discovery—probing for budget ceilings, compliance gaps
- Information asymmetry is central to the strategic challenge
- Partial observability forces efficient information gathering

**Limitation**: Constraint discovery relies on correct action types (send_document with specific artifact types)

### Lookahead Mechanism

**Purpose**: Allow the agent to simulate "what if" before committing to an action

**Trade-offs**:
- Adds computational cost (0.07 reward penalty per lookahead)
- Requires `LookaheadSimulator` that mirrors the environment
- Enables strategic planning at decision points

### No Opponent Modeling

**Deliberation is not opponent modeling**—it models how beliefs propagate through the committee after the agent acts, not how stakeholders form initial responses.

**Limitation**: The environment does not predict stakeholder responses ahead of time (only lookahead does)

---

## Implementation Details

### Key Classes

| Class | Module | Purpose |
|------|--------|---------|
| `DealRoomV3` | `deal_room/environment/dealroom_v3.py` | Main RL environment implementing reset/step |
| `DealRoomTextEnv` | `deal_room/environment/text_env.py` | TRL-compatible text wrapper for GRPO training |
| `CausalGraph` | `deal_room/committee/causal_graph.py` | Stakeholder influence graph with authority weights |
| `BeliefDistribution` | `deal_room/committee/causal_graph.py` | Belief state over 6 vendor types with confidence |
| `CommitteeDeliberationEngine` | `deal_room/committee/deliberation_engine.py` | Multi-step belief propagation and committee voting |
| `bayesian_update()` | `deal_room/committee/belief_tracker.py` | Bayesian belief update given an action |
| `UtteranceScorer` | `deal_room/rewards/utterance_scorer.py` | 5-dimension deterministic reward scoring |
| `StakeholderRiskProfile` | `deal_room/stakeholders/cvar_preferences.py` | CVaR risk profile per stakeholder archetype |
| `ARCHETYPE_PROFILES` | `deal_room/stakeholders/archetypes.py` | 6 stakeholder archetypes with risk parameters |
| `GRPOTrainer` | `deal_room/training/grpo_trainer.py` | GRPO-style self-play training harness |
| `AdaptiveCurriculumGenerator` | `deal_room/curriculum/adaptive_generator.py` | Scenario difficulty based on failure analysis |
| `LookaheadSimulator` | `deal_room/environment/lookahead.py` | "What-if" simulation before committing action |

### Pydantic Models

**`models.py`** defines the core interface types:

```python
# Action sent by agent
DealRoomAction: action_type, target, target_ids, message, documents,
                proposed_terms, channel, mode, lookahead

# Observation returned by environment
DealRoomObservation: reward, round_number, stakeholders,
                    stakeholder_messages, engagement_level, weak_signals,
                    known_constraints, deal_momentum, deal_stage,
                    veto_precursors, active_blockers, done, info

# Internal state (not exposed to agent directly)
DealRoomState: episode_id, stakeholders, stakeholder_private,
               hidden_constraints, offer_state, deal_stage,
               active_blockers, deal_momentum, terminal_outcome

# Lookahead simulation
LookaheadRequest: action_draft, n_hypotheses, depth
SimulationResult: predicted_responses, predicted_belief_deltas, cvar_impact
```

### Configuration

**Three benchmark tasks**:

| Task | Difficulty | Description |
|------|------------|-------------|
| `aligned` | Easy | Low-friction deal, correct document sequencing |
| `conflicted` | Medium | CTO-CFO tension, Legal-Procurement coalition |
| `hostile_acquisition` | Hard | Post-acquisition authority shift, compressed timeline |

**Stakeholder archetypes** (6 standard):
```python
ARCHETYPE_PROFILES = {
    "Legal": {"veto_power": True, "alpha": 0.95, "tau": 0.10},  # Risk-averse
    "Finance": {"veto_power": True, "alpha": 0.90, "tau": 0.15},
    "TechLead": {"alpha": 0.80, "tau": 0.25},
    "Procurement": {"alpha": 0.85, "tau": 0.20},
    "Operations": {"alpha": 0.80, "tau": 0.30},
    "ExecSponsor": {"veto_power": True, "alpha": 0.70, "tau": 0.40},
}
```

### Performance Considerations

- **Deterministic**: Same seed → same episode (reproducible training)
- **No LLM calls in hot path**: Reward scoring is algebraic
- **Lookahead is optional**: Costs 0.07 reward but enables planning
- **Session pooling**: Server maintains environment instances per session with TTL

---

## Installation and Setup

### Prerequisites

- Python 3.10+
- pip or uv package manager

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/your-org/deal_room_s2p.git
cd deal_room_s2p

# Install dependencies
pip install -r requirements.txt

# Or with uv
uv sync
```

### Option 2: Docker

```bash
# Build the image
docker build -t dealroom-web:latest .

# Run the container
docker run -d -p 7860:7860 dealroom-web:latest

# Access the web interface
# http://localhost:7860/web
```

### Verify Installation

```bash
# Run the server
uvicorn server.app:app --reload --port 7860

# In another terminal, check health
curl http://localhost:7860/health

# Should return:
# {"status":"ok","service":"deal-room","tasks":["aligned","conflicted","hostile_acquisition"]}
```

---

## Usage

### Web Interface

Open `http://localhost:7860/web` in a browser. The Gradio interface provides:

- **Playground tab**: Low-level state inspection
- **Clean tab**: Streamlined negotiation UI with 4-step workflow

### Python API

```python
from models import DealRoomAction, DealRoomObservation
from deal_room.environment.dealroom_v3 import DealRoomV3

# Initialize environment
env = DealRoomV3()
obs = env.reset(seed=42, task_id="aligned")

print(f"Stage: {obs.deal_stage}")  # "evaluation"
print(f"Stakeholders: {list(obs.stakeholders.keys())}")
# ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"]

# Take a step
action = DealRoomAction(
    action_type="send_document",
    target="Finance",
    target_ids=["Finance"],
    message="Here is the ROI model with explicit payback assumptions.",
    documents=[{"type": "roi_model", "name": "ROI"}],
)
obs, reward, done, info = env.step(action)

print(f"Reward: {reward:.4f}")
print(f"Done: {done}")
```

### REST API

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "aligned", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "direct_message",
    "target": "Finance",
    "message": "Help me understand the budget ceiling."
  }'

# Get full state
curl http://localhost:7860/state
```

### Training with GRPO

```python
from deal_room.training.grpo_trainer import GRPOTrainer, RandomPolicyAdapter, HeuristicPolicyAdapter
from deal_room.environment.dealroom_v3 import DealRoomV3

env = DealRoomV3()
policy = HeuristicPolicyAdapter()  # or RandomPolicyAdapter(), or custom ModelPolicyAdapter

trainer = GRPOTrainer(
    environment=env,
    policy=policy,
    n_episodes=100,
    task_mix={"aligned": 0.5, "conflicted": 0.3, "hostile_acquisition": 0.2},
)

results = trainer.train()
```

### Running Baseline Inference

```bash
python inference.py
```

---

## Example Outputs

### Reset Output

```python
obs = env.reset(seed=42, task_id="aligned")

# Observation fields
obs.round_number        # 0
obs.max_rounds          # 10
obs.deal_stage          # "evaluation"
obs.deal_momentum       # "stalling"
obs.stakeholders        # {"Legal": {"role": "General Counsel / Legal"}, ...}
obs.engagement_level    # {"Legal": 0.52, "Finance": 0.48, ...}
obs.weak_signals        # {} (no signals initially)
obs.known_constraints   # [] (hidden until discovered)
obs.veto_precursors     # {} (no warnings initially)
```

### Step with Reward

```python
action = DealRoomAction(
    action_type="send_document",
    target="Finance",
    target_ids=["Finance"],
    message="Here is the ROI model.",
    documents=[{"type": "roi_model", "name": "ROI"}],
)
obs, reward, done, info = env.step(action)

# Reward breakdown (via info dict)
info["reward_components"]  # {"goal": 0.03, "trust": 0.02, "info": 0.04, "risk": 0.01, "causal": 0.01}
info["terminal_outcome"]   # "" (not terminal yet)

# Observation updates
obs.stakeholder_messages["Finance"]  # May contain Finance's response
obs.engagement_level["Finance"]      # May increase if action was well-received
obs.deal_momentum                    # May shift toward "progressing"
```

### Terminal Episode

```python
# After max rounds or veto
obs.done                           # True
obs.info["terminal_outcome"]       # "deal_closed", "veto", "max_rounds", etc.

# Final score from reward
reward                             # e.g., 0.86 for aligned task
```

### Console Output During Training

```
Episode 1/100 | Task: aligned | Rounds: 5/10 | Reward: 0.44 | Outcome: running
Episode 2/100 | Task: conflicted | Rounds: 10/10 | Reward: -0.50 | Outcome: timeout
Episode 3/100 | Task: hostile_acquisition | Rounds: 3/10 | Reward: -1.00 | Outcome: veto
...
Training complete | Avg Reward: 0.31 | Deal Rate: 45%
```

---

## Project Structure

```
deal_room_s2p/
├── deal_room/                          # Main Python package
│   ├── environment/
│   │   ├── dealroom_v3.py               # Main RL environment (DealRoomV3 class)
│   │   ├── text_env.py                  # TRL-compatible text wrapper
│   │   ├── constants.py                 # Reward weights, stage gates, penalties
│   │   ├── prompts.py                   # LLM prompt templates
│   │   ├── lookahead.py                 # LookaheadSimulator for "what-if" planning
│   │   ├── llm_client.py                # LLM client for deliberation summaries
│   │   └── stakeholder_llm.py          # LLM-based stakeholder responses
│   ├── committee/
│   │   ├── causal_graph.py              # CausalGraph, BeliefDistribution, propagate_beliefs
│   │   ├── deliberation_engine.py        # CommitteeDeliberationEngine
│   │   └── belief_tracker.py            # bayesian_update()
│   ├── stakeholders/
│   │   ├── archetypes.py                 # ARCHETYPE_PROFILES for 6 stakeholders
│   │   └── cvar_preferences.py          # StakeholderRiskProfile, compute_cvar()
│   ├── rewards/
│   │   ├── utterance_scorer.py          # 5-dim UtteranceScorer
│   │   └── pareto_efficiency.py        # Terminal reward computation
│   ├── training/
│   │   ├── grpo_trainer.py              # GRPOTrainer with policy adapters
│   │   ├── ppo_trainer.py              # SimplePPOTrainer with GAE
│   │   └── run_benchmark.py            # Benchmarking script
│   └── curriculum/
│       └── adaptive_generator.py       # AdaptiveCurriculumGenerator
├── server/
│   ├── app.py                           # FastAPI server (reset, step, state endpoints)
│   ├── session_pool.py                  # DealRoomSessionPool
│   ├── gradio_clean.py                  # Clean Gradio web interface
│   ├── gradio_custom.py                 # Custom Gradio interface
│   ├── validator.py                     # OutputValidator
│   ├── grader.py                        # CCIGrader
│   ├── claims.py                        # ClaimsTracker
│   ├── semantics.py                     # Semantic analyzer
│   ├── scenarios.py                    # Task scenario configurations
│   └── stakeholders.py                 # StakeholderEngine
├── models.py                            # Pydantic models (DealRoomAction, Observation, State)
├── pyproject.toml                       # Project configuration
├── Dockerfile                           # Multi-stage Docker build
├── inference.py                         # Baseline inference script
├── calibrate.py                        # Calibration script
└── episode_timer.py                    # Episode timing utilities
```

### Directory Responsibilities

| Directory | Responsibility |
|-----------|----------------|
| `deal_room/environment/` | Core RL loop, state management, observation construction |
| `deal_room/committee/` | Committee dynamics—causal graph, belief propagation, deliberation |
| `deal_room/stakeholders/` | Stakeholder archetypes, risk profiles, CVaR preferences |
| `deal_room/rewards/` | Reward computation—utterance scoring, terminal rewards |
| `deal_room/training/` | Training harnesses—GRPO, PPO, policy adapters |
| `deal_room/curriculum/` | Adaptive difficulty based on failure analysis |
| `server/` | HTTP wrapper, session management, web interface |

---

## Extensibility

### Modifying the Environment

**Add a new stakeholder archetype:**
```python
# In deal_room/stakeholders/archetypes.py
NEW_PROFILE = {
    "name": "Security",
    "role": "CISO / Security",
    "alpha": 0.85,        # Risk sensitivity
    "tau": 0.20,          # Risk tolerance
    "veto_power": True,
    "authority": 3,
    "concerns": ["data_security", "compliance", "penetration_testing"],
}
ARCHETYPE_PROFILES["Security"] = NEW_PROFILE
```

**Adjust reward weights:**
```python
# In deal_room/environment/constants.py
REWARD_WEIGHTS = {
    "goal": 0.30,   # Increase goal weight
    "trust": 0.15,  # Decrease trust weight
    "info": 0.20,
    "risk": 0.20,
    "causal": 0.15,
}
```

**Modify causal graph sampling:**
```python
# In deal_room/committee/causal_graph.py
SCENARIO_PARAMS = {
    "aligned": {
        "base_edge_probability": 0.35,  # Increase edge density
        "authority_edge_prob": 0.90,    # Stronger authority influence
        # ...
    },
}
```

### Plugging in New Agents

**Implement the PolicyAdapter protocol:**
```python
from deal_room.training.grpo_trainer import PolicyAdapter

class MyPolicyAdapter:
    name = "my_policy"

    def act(self, observation, rng):
        # Your policy logic here
        return DealRoomAction(...)

    def update_from_batch(self, trajectories):
        # Training update logic
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
```

**Use with GRPOTrainer:**
```python
trainer = GRPOTrainer(
    environment=env,
    policy=MyPolicyAdapter(),
    n_episodes=100,
)
```

### Experimenting with Reward Functions

**Create custom utterance scorer:**
```python
from deal_room.rewards.utterance_scorer import UtteranceScorer, UtteranceScore

class MyScorer(UtteranceScorer):
    def score(self, action, state_before, state_after, true_graph, lookahead_used=False):
        # Custom scoring logic
        return UtteranceScore(goal=0.1, trust=0.1, info=0.1, risk=0.1, causal=0.1)
```

**Replace in environment:**
```python
env = DealRoomV3()
env._utterance_scorer = MyScorer()
```

---

## Limitations and Future Work

### Current Limitations

| Limitation | Description |
|------------|-------------|
| **No opponent modeling** | Environment predicts belief propagation but not stakeholder initial responses |
| **Fixed stakeholder set** | 6 stakeholders is standard; adding/removing requires code changes |
| **Single deal type** | Software negotiation only; other deal types not modeled |
| **No competitor modeling** | Other vendors not simulated |
| **Lookahead complexity** | LookaheadSimulator must mirror full environment logic |
| **No negotiation history** | Full conversation transcript not maintained between rounds |
| **Reward hacking possible** | Agent may exploit dimensions without achieving true deal closure |
| **No explainability** | No mechanism to explain why a stakeholder blocked |

### Known Technical Debt

- `dealroom_v3.py` is 1267 lines—should be split into smaller modules
- Some documentation refers to v2.5 architecture (outdated)
- Test coverage incomplete in some subsystems

### Future Work

| Direction | Description |
|-----------|-------------|
| **Multi-turn negotiation history** | Maintain full conversation transcript for context-aware responses |
| **Learned stakeholder responses** | Replace rule-based response generation with fine-tuned LLM |
| **Dynamic stakeholder addition** | Support variable committee composition per episode |
| **Explainability layer** | Add Why-Blocked explanations for veto events |
| **Curriculum progression** | Automatic difficulty adjustment based on success rate |
| **Multi-deal scenarios** | Agent managing multiple concurrent deals |
| **Cost-benefit analysis** | Explicit modeling of deal economics beyond CVaR |

---

## License

BSD 3-Clause License. See LICENSE file for details.

---

## References

- OpenEnv Framework: https://github.com/meta-pytorch/OpenEnv
- GRPO Training: Refer to `deal_room/training/grpo_trainer.py`
- CCIGrader Specification: See `server/grader.py`
- Causal Committee Dynamics: See `deal_room/committee/`