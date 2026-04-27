---
title: deal-room-S2P
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# Closing the Deal: Teaching an Agent to Navigate Enterprise Negotiation

_What if every decision could backfire—and the stakes only become clear three moves later?_

---

## The Room Where Deals Go to Die

It's Wednesday afternoon. You've been negotiating a six-figure enterprise software deal for two weeks. The CFO seemed aligned last week. The CTO gave strong signals. Legal sent over comments on Monday but hasn't responded to your replies.

Now Procurement is suddenly asking for references from clients you've never heard of. Finance is circling back on terms you thought were settled. And in 72 hours, the executive sponsor—who's been mostly hands-off—has scheduled a call.

You don't know why things shifted. You don't know who talked to whom. You just know the deal that felt close a week ago now feels like it could unravel at any moment.

---

## What Would You Do?

So you make a call: Do you push for the close, or do you try to understand what's changed?

Here's the trap: you won't find out if you were right until long after you've committed. And by then, the landscape may have shifted again. Enterprise negotiation is a game of **sequential decisions under deep uncertainty**—where feedback is delayed, causality is murky, and the people at the table are talking to each other with or without you.

---

## Why This Problem Defies Simple Solutions

Every executive, account manager, and procurement lead has sat in some version of that room. The challenge isn't just persuasion—it's:

- **Sequential decision-making**: Every move changes the landscape for the next move
- **Partial observability**: You don't know what others are thinking, only what they say
- **Delayed feedback**: The consequences of your decisions surface days or weeks later
- **Multi-party dynamics**: It's not you versus one person—it's you versus a committee with internal politics, competing interests, and veto power

These properties make this class of problem genuinely hard for AI systems. Standard approaches fall apart because they can't handle the interconnected consequences of actions across time.

---

## Meet DealRoom

That's exactly why we built **DealRoom S2P V3**—a simulated negotiation environment where an AI agent (specifically an LLM) plays the role of a vendor navigating a complex enterprise deal.

Instead of one buyer, there are **six stakeholders**: Legal, Finance, TechLead, Procurement, Operations, and an ExecSponsor. Each has different priorities. Some have veto power. They influence each other through a hidden causal graph. They update their views based on what others say. And they don't always tell you what they're actually worried about.

The agent's goal isn't just to land a signature—it's to move all six stakeholders into alignment without triggering a veto, running out of rounds, or creating irreversible damage to trust.

Think of it as chess where the pieces talk back, change their minds, and sometimes coalesce against you without warning.

---

## What the Agent Actually Sees

On each turn, the agent receives a **corrupted, partial observation** (POMDP):

- **Stakeholder messages**: What each person has said—or chosen not to say (20% corrupted to placeholder)
- **Engagement levels**: Noisy signals (σ=0.10) about whether each stakeholder is moving toward or away from agreement
- **Weak signals**: Hints about hidden concerns that haven't been explicitly stated (30% randomly dropped)
- **Veto warnings**: Early indicators that someone might block the deal
- **Cross-stakeholder echoes**: Influence cascades showing how information flows through the committee (~70% recall)
- **Deal stage**: Where in the approval process the deal currently sits (evaluation → negotiation → legal_review → final_approval → closed)

The agent doesn't see the full internal state. It has to infer the hidden causal graph, true beliefs, and CVaR thresholds from imperfect signals—just like in real negotiation.

---

## What the Agent Can Do

The agent chooses from **8 action types** using a pipe-delimited format:

```
send_document Legal dpa | Our DPA covers GDPR Article 28 obligations.
direct_message Finance | We offer competitive ROI projections.
concession Finance | price=175000
group_proposal | All parties are aligned on key terms.
exec_escalation | Requesting executive meeting.
```

| Action | Effect |
|--------|--------|
| `send_document <target> <doc_type>` | Send artifact (DPA, security cert, ROI model, timeline). Strong belief update. |
| `direct_message <target>` | Send text message to specific stakeholder. Lighter update. |
| `concession <target> | <term>=<value>` | Offer term adjustment. Updates deal terms directly. |
| `group_proposal | <message>` | Propose to full committee. Broadcasts intent. |
| `exec_escalation | <message>` | Activate ExecSponsor. High-impact but signals urgency. |
| `submit_proposal` | Formal proposal with SLA, pricing, compliance attestations. Triggers final approval. |
| `redline_clause` | Counter-offer specific clause. High stakes. |
| `acknowledge_stage` | Acknowledge current stage. No-op but records intent. |

Sending the right document to the right person at the right time can unlock progress. Sending the wrong one—or nothing at all—can stall a deal that was moving.

---

## How Progress Gets Measured

Each action produces a **5-dimensional reward signal** using a **zero-centered sigmoid** (range approximately [-1.5, +1.5] per dimension):

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Goal** | 0.25 | Deal progress: weighted approval delta + blocker resolution + CVaR headroom |
| **Trust** | 0.20 | Trustworthy mass delta for targeted stakeholder |
| **Info** | 0.20 | Entropy reduction across committee beliefs (uncertainty → clarity) |
| **Risk** | 0.20 | CVaR improvement for risk-averse stakeholders |
| **Causal** | 0.15 | Betweenness centrality of targeted node in hidden influence graph |

**Immediate milestone bonuses** reward sharp signals:
- Stage advance: **+0.5**
- Blocker resolved: **+0.15** per blocker
- DPA to Legal (when CVaR risk elevated): **+0.3**
- Security cert sent: **+0.2**
- Veto precursor escalation: **-0.2**

**Penalties prevent inaction:**
- Non-progress (no belief delta > 0.02): **-0.1**
- Diversity reward (≥3 unique actions in last 10): **+0.05**

**Terminal rewards** fire at episode end:
- Deal closed: **+1.0**
- Hard veto (policy breach): **-1.0**
- Soft veto (CVaR veto): **-0.8**
- Stage regression: **-0.75**
- Timeout: **-0.5**

The agent isn't optimizing a single number—it's seeing a vector of signals that together tell a richer story about whether a move was wise or harmful.

---

## What Makes This Different

Standard RL environments tend to be either too simple (one opponent, immediate feedback, clear success criteria) or too simulated (perfect information, stable opponents, no hidden state).

DealRoom S2P V3 breaks from convention in key ways:

**1. The causal graph is hidden (P1).** Stakeholders influence each other through an unseen influence structure. If Legal raises a concern, Finance might start asking questions—even if your message was perfectly reasonable. The agent must infer this structure from noisy signals.

**2. Beliefs are tracked as Bayesian distributions.** Each stakeholder maintains a probability distribution over 6 vendor types: competent/incompetent, trustworthy/deceptive, aligned/misaligned. These beliefs evolve via Bayesian updates and propagate through the causal graph.

**3. CVaR-based veto (P3).** Legal (α=0.95) vetoes based on tail risk, not expected value. A deal with positive expected utility can still be vetoed if the worst 5% of outcomes are bad enough.

**4. Training uses multi-step generation.** The agent generates 2 actions per decision point before receiving heuristic follow-ups, learning strategy rather than just first moves.

**5. True POMDP observability.** Engagement noise (σ=0.10) cannot be perfectly cancelled. Weak signals are 30% randomly dropped. The agent never sees true beliefs.

---

## A Negotiation in Progress

Let's trace through a real interaction sequence:

**Round 1**: Agent sends ROI analysis to Finance.
- Finance engagement increases slightly
- Reward: +0.15 (info), +0.05 (goal)

**Round 2**: Agent sends implementation timeline to TechLead.
- TechLead responds with technical questions
- Legal's engagement drops—maybe they expected to be looped in earlier?
- Reward: +0.08 (causal), -0.12 (risk from incomplete engagement)

**Round 3**: Agent sends DPA to Legal.
- Legal engagement stabilizes
- Finance engagement rises
- Committee vote shifts from "mixed" to "leaning positive"
- **Milestone bonus: +0.3** (DPA to Legal when CVaR risk elevated)
- Reward: +0.22 (trust + info), +0.15 (goal)

**Round 4**: Agent proposes terms to all stakeholders.
- Procurement raises concern about reference customers
- Two stakeholders enter "veto precursor" state
- **Milestone penalty: -0.2** (veto precursor added)
- Reward: -0.15 (risk). Message: "Deal stalling."

**Round 5**: Agent addresses Procurement's concern with specific customer references.
- Procurement engagement improves
- Veto precursors clear
- Deal momentum shifts to "progressing"
- **Milestone bonus: +0.5** (stage advance)
- Reward: +0.30 (goal + trust). Stage advances.

The agent doesn't just optimize the next move—it plans across the full arc of the negotiation, anticipating how one stakeholder's concerns might cascade to others.

---

## The Technical Layer

Under the hood, DealRoom S2P V3 is framed as a **POMDP** with partial observability:

- **Hidden state**: Causal graph G, true belief distributions B_i, CVaR values, veto thresholds
- **Observable state**: Engagement (noisy), weak signals (partial), messages (corrupted), deal stage, blockers
- **Actions**: 8 types with pipe-delimited format
- **Reward**: 5-dim vector → scalar via weighted sum, then milestone/penalty adjustments
- **Discount factor**: γ=0.3 for multi-step credit assignment

The environment uses:

- **Bayesian belief updates**: Stakeholder beliefs evolve based on actions (`belief_tracker.py`)
- **CVaR risk modeling**: Legal (α=0.95), Finance (α=0.90) evaluate proposals using Conditional Value at Risk
- **Committee deliberation**: 3-step belief propagation through the hidden influence graph
- **Lookahead simulation**: Optional minimax-robust planning with a 0.07 exact cost penalty
- **Zero-centered sigmoid reward**: `_squash(raw) = 1.5 * (2/(1+e^(-3*raw)) - 1)` — centered at 0, no baseline bias
- **POMDP noise injection**: σ=0.10 engagement noise, 30% weak signal drop, 20% message corruption
- **Multi-seed reward evaluation**: 3-seed averaging per completion eliminates environment luck
- **Entropy regularization**: entropy_coef=0.01 prevents premature policy convergence

Training is via **GRPO** (Group Relative Policy Optimization) with num_generations=16 for stable advantage estimation.

---

## What We've Learned

Running experiments in DealRoom S2P V3 has surfaced insights that aren't obvious from theory:

**Document sequencing matters more than message content.** In the early rounds, which artifact you send matters more than what you say. Sending the DPA to Legal before the ROI to Finance creates a different committee dynamic than the reverse order—even if the messages are identical.

**Trust is easy to lose and hard to regain.** A single infeasible promise creates a permanent mark on the trust track. The agent can still close the deal, but it will have to work significantly harder in subsequent rounds to overcome the damage.

**Veto warnings are not vetoes.** The system flags "veto precursors" before they become actual blocks. A well-timed concession or evidence package can clear the warning. But if the agent ignores the signal for 2 rounds, the precursor becomes a hard veto—and recovery becomes much harder.

**The "aligned" scenario is deceptively difficult.** You'd expect the low-friction scenario to be simple. But even there, getting all six stakeholders to approval thresholds requires careful sequencing. The baseline performance isn't a ceiling—it's a genuine challenge.

**Zero-centered reward matters.** The shift from `0.5 + 0.5*tanh()` to `_squash()` eliminated the 0.5 baseline bias. Neutral actions now score 0, not 0.5. The gradient is ~3× steeper, and good/bad actions are now instantly distinguishable.

---

## Why This Matters

Enterprise negotiation isn't a niche problem. It touches sales, procurement, partnership management, and executive alignment at every company that sells complex B2B software or services.

And it's a domain where AI can genuinely help—if it can reason about multi-party dynamics, not just individual preferences. The ability to model how one stakeholder's concerns might ripple to others, to plan across multiple rounds, to understand when to push and when to wait—these are exactly the capabilities that make AI useful beyond simple Q&A.

For AI researchers, DealRoom S2P V3 offers a reproducible benchmark where the challenge is strategic reasoning, partial observability, and credit assignment—not just getting to a known optimal policy.

---

## Try It Yourself

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g2B0AEz2M8lyLQzlU3QKcgdLyNyengsy?usp=sharing)

**The Colab notebook gives you:**
- A view of all six stakeholders with their roles and current engagement status
- The ability to select any stakeholder and see their message history and what they need
- Context-aware action suggestions or free-form message composition
- Real-time tracking of your score across the five reward dimensions
- Visibility into deal stage progression and veto warnings
- GRPO training with multi-seed evaluation and entropy regularization

**What to observe:**
- How does engagement change after you send a document versus a direct message?
- What happens to Finance's engagement when Legal raises a compliance concern?
- Can you identify a "weak signal" before it becomes a veto precursor?
- How does the committee's overall mood shift after a group proposal versus individual outreach?
- How does the zero-centered sigmoid reward make good and bad actions immediately distinguishable?

The interface is designed to make committee dynamics visible. Use it to build intuition about negotiation strategy, then explore how different action sequences affect outcomes.

---

## The Deal That Didn't Close

Here's the thing about enterprise negotiation: sometimes you do everything right and the deal still falls apart. A competitor emerges. Budget gets frozen. The executive sponsor moves to another company.

DealRoom S2P V3 doesn't promise that perfect play always wins. It promises that you'll understand why a deal succeeded or failed—and that's actually more useful.

Because in the room where deals go to die, the winners aren't the ones who never lost. They're the ones who learned to read the signals earlier.

---

_DealRoom S2P V3 is an OpenEnv-compatible RL environment for enterprise B2B negotiation. The environment is designed to train an LLM (not a neural network agent). Explore the code, try the Colab notebook, and dig into the research on [GitHub](https://github.com/akshaypulla/deal_room_S2P)._
