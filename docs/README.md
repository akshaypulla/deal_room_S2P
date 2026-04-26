# DealRoom Documentation

This directory contains all documentation for the DealRoom project.

## Directory Structure

```
docs/
├── README.md                    # This file - documentation overview
├── blog_post.md                 # Hugging Face blog post (narrative-driven)
├── architecture.md              # Current environment architecture (v2.5 reference)
├── architecture_enterprise.md   # Enterprise architecture notes
├── testing.md                   # General testing documentation
├── testing_enterprise.md        # Enterprise testing documentation
├── usage.md                     # Practical usage guide
├── openenv.yaml                 # OpenEnv specification copy
│
├── test_results/                # Test execution reports and results
│   ├── TEST_RESULTS.md         # General test results
│   ├── TEST_EXECUTION_REPORT.md # Test execution summary
│   ├── test_results_S2P_V_1.md  # Results for S2P V1
│   ├── test_results_v3.6.md    # Results for v3.6
│   ├── test_results_v3.7.md     # Results for v3.7
│   ├── testing_S2P_V_1.md      # Testing notes for S2P V1
│   ├── testing_v3.2.md         # Testing notes for v3.2
│   ├── testing_v3.3.md         # Testing notes for v3.3
│   ├── testing_v3.4.md         # Testing notes for v3.4
│   ├── testing_v3.7.md         # Testing notes for v3.7
│   └── tests_v3.6.md           # v3.6 test documentation
│
├── versions/                    # Architecture documentation by version
│   ├── ARCHITECTURE.md          # Original architecture spec
│   ├── ARCHITECTURE_IMPLEMENTED.md # Implemented architecture
│   ├── ARCHITECTURE_V3_IMPLEMENTED.md
│   ├── ARCHITECTURE_V3_UPDATED.md
│   ├── architecture_S2P_V_1.md
│   ├── architecture_v3.2.md
│   ├── architecture_v3.3.md
│   ├── architecture_v3.4.md
│   ├── architecture_v3.6.md
│   ├── architecture_v3.7.md
│   ├── TESTING_IMPLEMENTED.md
│   ├── TESTING_V3_IMPLEMENTED.md
│   └── TESTING_V3_UPDATED.md
│
├── legacy/                      # Older documentation (superseded)
│   ├── TESTING.md              # Superseded by docs/testing.md
│   └── CONTRIBUTING.md         # General contribution guide
│
└── (spec files)
    ├── CAUSAL_COMMITTEE_DYNAMICS.md
    ├── CVAR_BAYESIAN_PREFERENCES.md
    ├── DELIBERATION_AND_CURRICULUM.md
    ├── Observation_Mechanism.md
    ├── PRD_DEALROOM_V3.md
    ├── REWARD_HACKING_IMPOSSIBILITY.md
    ├── UTTERANCE_REWARD_SPEC.md
    └── deal_room_v3_plan.md
```

## File Categories

### Current Reference Documentation

| File | Description |
|------|-------------|
| `architecture.md` | Current environment architecture (v2.5). Start here for understanding the system. |
| `testing.md` | General testing approach and coverage |
| `usage.md` | Practical guide for running the environment |
| `architecture_enterprise.md` | Enterprise-specific architecture notes |

### Narrative Content

| File | Description |
|------|-------------|
| `blog_post.md` | Hugging Face blog post - narrative-driven explanation for general audience |
| `CAUSAL_COMMITTEE_DYNAMICS.md` | Deep dive into committee belief propagation |
| `CVAR_BAYESIAN_PREFERENCES.md` | CVaR modeling and Bayesian belief tracking |
| `DELIBERATION_AND_CURRICULUM.md` | Deliberation engine and adaptive curriculum design |
| `Observation_Mechanism.md` | How observations are constructed and noise is applied |
| `REWARD_HACKING_IMPOSSIBILITY.md` | Why the reward design prevents reward hacking |
| `UTTERANCE_REWARD_SPEC.md` | Detailed reward specification |
| `PRD_DEALROOM_V3.md` | Product requirements document for v3 |
| `deal_room_v3_plan.md` | Implementation plan for v3 |

### Test Results (by version)

| File | Description |
|------|-------------|
| `test_results/S2P_V_1*` | Test results for S2P V1 release |
| `test_results/v3.6*` | Test results for v3.6 release |
| `test_results/v3.7*` | Test results for v3.7 release |

### Version-Specific Architecture Docs

The `versions/` directory contains architecture documentation from specific versions. These are kept for historical reference but may be outdated. The most recent architecture documentation should be used for understanding the current system.

### Legacy Documentation

| File | Why Legacy |
|------|------------|
| `legacy/TESTING.md` | Superseded by `docs/testing.md` which has more current information |
| `legacy/CONTRIBUTING.md` | General contribution guide - kept for reference |

## Quick Reference

**For understanding the environment:** Start with `docs/architecture.md`
**For running the code:** See `docs/usage.md`
**For testing:** See `docs/testing.md`
**For a narrative overview:** Read `docs/blog_post.md`