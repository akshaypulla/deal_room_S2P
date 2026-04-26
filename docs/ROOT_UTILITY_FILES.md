# Root-Level Utility Files

This file documents the purpose of utility scripts and debug files in the project root.

## Core Project Files (Required)

| File | Purpose |
|------|---------|
| `README.md` | Project overview and documentation |
| `models.py` | Pydantic models for DealRoom action/observation/state |
| `pyproject.toml` | Project configuration and dependencies |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Multi-stage Docker build |
| `LICENSE` | BSD 3-Clause License |

## Utility Scripts

| File | Purpose |
|------|---------|
| `inference.py` | Baseline inference script - runs the environment with a simple policy |
| `calibrate.py` | Calibration script for environment parameters |
| `episode_timer.py` | Episode timing utilities |
| `client.py` | Client implementation for API access |

## Debug and Development Scripts

The following files are **development utilities** and may be removed once the system is stable:

| File | Purpose | Status |
|------|---------|--------|
| `debug_reward.py` | Debug reward computation | Development |
| `debug_reward_correct.py` | Corrected reward debug | Development |
| `debug_reward_deep.py` | Deep reward debugging | Development |
| `debug_reward_deep2.py` | Deep reward debugging (v2) | Development |
| `final_verify.py` | Final verification script | Development |
| `verify_fixes.py` | Verify recent fixes | Development |
| `demo_inference.py` | Demo inference script | Development |
| `learnability_test.py` | Learnability testing | Development |
| `learning_clean.py` | Learning progression testing | Development |
| `learning_progression_test.py` | Learning progression tests | Development |

## Notebook Files

| File | Purpose |
|------|---------|
| `dealroom_grpo_training.ipynb` | GRPO training notebook |

## Scripts

| File | Purpose |
|------|---------|
| `reproduce.sh` | Reproducible setup script |

## Deprecated/Obsolete Files

The following files should be removed as they are **superseded by the current codebase**:

| File | Reason |
|------|--------|
| `server/deal_room_environment.py` | Legacy environment - superseded by `deal_room/environment/dealroom_v3.py` |
| `server/gradio_custom.py` | Legacy Gradio interface - superseded by `server/gradio_clean.py` |

These legacy files are retained for reference but are not actively used in the current system.

## Notes

- The core RL logic lives in `deal_room/` package
- The server logic lives in `server/` package
- The primary interface is via `server/app.py` (FastAPI) or Gradio (`server/gradio_clean.py`)
- Debug scripts were used during development and may be removed in production