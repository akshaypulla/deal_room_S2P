# Root-Level Utility Files

This file documents the purpose of utility scripts and files in the project root.

## Core Project Files (Required)

| File | Purpose |
|------|---------|
| `README.md` | Project overview and documentation (stays at root) |
| `models.py` | Pydantic models for DealRoom action/observation/state |
| `pyproject.toml` | Project configuration and dependencies |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Multi-stage Docker build |
| `LICENSE` | BSD 3-Clause License |
| `__init__.py` | Package initialization (exports DealRoomV3) |
| `client.py` | OpenEnv client implementation for API access |

## Utility Scripts (Root Level)

| File | Purpose |
|------|---------|
| `episode_timer.py` | Episode timing utilities |
| `reproduce.sh` | Reproducible setup script |

## Debug Scripts (Development Only)

The following files are **development utilities** stored in `debug/`:

| File | Purpose |
|------|---------|
| `debug/debug_reward.py` | Debug reward computation across episodes |
| `debug/debug_reward_correct.py` | Corrected reward debug |
| `debug/debug_reward_deep.py` | Deep reward debugging |
| `debug/debug_reward_deep2.py` | Deep reward debugging (v2) |
| `debug/verify_fixes.py` | Verify recent fixes |
| `debug/final_verify.py` | Final verification script |

These files are used during development and may be removed once the system is stable.

## Experiment Scripts

The following files are **research experiments** stored in `experiments/`:

| File | Purpose |
|------|---------|
| `experiments/inference.py` | Baseline inference script with heuristic policy |
| `experiments/demo_inference.py` | Standalone LLM inference demo |
| `experiments/calibrate.py` | Environment parameter calibration |
| `experiments/learnability_test.py` | Learnability testing |
| `experiments/learning_clean.py` | Learning progression testing |
| `experiments/learning_progression_test.py` | Learning progression tests |

These files contain experimental code and research implementations.

## Legacy/Obsolete Files

The following files have been **superseded by the current codebase**:

| File | Reason |
|------|--------|
| `server/deal_room_environment.py` | Legacy environment - superseded by `deal_room/environment/dealroom_v3.py` |
| `server/gradio_custom.py` | Legacy Gradio interface - superseded by `server/gradio_clean.py` |

These legacy files are retained for reference but are not actively used in the current system.

## Notes

- The core RL logic lives in `deal_room/` package
- The server logic lives in `server/` package
- The primary interface is via `server/app.py` (FastAPI) or Gradio (`server/gradio_clean.py`)