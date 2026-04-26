---
title: deal-room-S2P
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
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

That's exactly why we built **DealRoom**—a simulated negotiation environment where an AI agent plays the role of a vendor navigating a complex enterprise deal.

Instead of one buyer, there are **six stakeholders**: Legal, Finance, TechLead, Procurement, Operations, and an ExecSponsor. Each has different priorities. Some have veto power. They influence each other. They update their views based on what others say. And they don't always tell you what they're actually worried about.

The agent's goal isn't just to land a signature—it's to move all six stakeholders into alignment without triggering a veto, running out of rounds, or creating irreversible damage to trust.

Think of it as chess where the pieces talk back, change their minds, and sometimes coalesce against you without warning.

---

## What the Agent Actually Sees

On each turn, the agent receives an observation containing:

- **Stakeholder messages**: What each person has said—or chosen not to say
- **Engagement levels**: Noisy signals about whether each stakeholder is moving toward or away from agreement
- **Weak signals**: Hints about hidden concerns that haven't been explicitly stated
- **Veto warnings**: Early indicators that someone might block the deal
- **Deal stage**: Where in the approval process the deal currently sits (evaluation → negotiation → legal_review → final_approval → closed)

The agent doesn't see the full internal state. It has to infer from partial, imperfect signals—just like in real negotiation.

---

## What the Agent Can Do

The agent chooses from these action types:

| Action               | Effect                                                                 |
| -------------------- | ---------------------------------------------------------------------- |
| `direct_message`     | Target a specific stakeholder with a message                           |
| `send_document`      | Share evidence—ROI analysis, compliance docs, implementation timelines |
| `group_proposal`     | Propose terms to the full committee                                    |
| `concession`         | Offer ground on pricing, timeline, or contract terms                   |
| `backchannel`        | Send a softer, lower-stakes communication                              |
| `exec_escalation`    | Push for executive-level attention                                     |
| `walkaway_signal`    | Signal willingness to exit if terms aren't met                         |
| `reframe_value_prop` | Shift how the value proposition is presented                           |

Documents matter enormously. Sending the right document to the right person at the right time can unlock progress. Sending the wrong one—or nothing at all—can stall a deal that was moving.

---

## How Progress Gets Measured

Each action produces a **5-dimensional reward signal**:

- **Goal progress** (+0.30 weight): Did the action move stakeholders closer to alignment?
- **Trust** (+0.15 weight): Did the action build or erode confidence in the vendor?
- **Information gain** (+0.20 weight): Did the action reduce uncertainty about what stakeholders need?
- **Risk** (+0.20 weight): Did it increase or decrease exposure to CVaR losses for risk-averse stakeholders?
- **Causal impact** (+0.15 weight): Did the action influence committee dynamics, not just individual opinions?

The agent isn't optimizing a single number—it's seeing a vector of signals that together tell a richer story about whether a move was wise or harmful.

Episodes run for up to **10 rounds**. The deal can close successfully, trigger a veto, or time out. The final reward reflects the full arc of the negotiation, not just the last move.

---

## What Makes This Different

Standard RL environments tend to be either too simple (one opponent, immediate feedback, clear success criteria) or too simulated (perfect information, stable opponents, no hidden state).

DealRoom breaks from convention in three key ways:

**1. Committee dynamics aren't static.** Stakeholders influence each other through a hidden causal graph. If Legal raises a concern, Finance might start asking questions—even if your message was perfectly reasonable. The influence structure shifts how information and sentiment flow through the committee.

**2. Beliefs are tracked over time.** Each stakeholder maintains a Bayesian belief distribution over what kind of vendor they're dealing with: competent or incompetent, trustworthy or deceptive, aligned with their interests or not. These beliefs evolve based on your actions and what others say. You can't just apologize your way out of a trust deficit.

**3. Veto is not the same as failure.** Sometimes a veto is avoidable with better sequencing. Sometimes it's a signal that trust was damaged three rounds earlier. The agent has to reason about causality, not just correlation.

---

## A Negotiation in Progress

Let's trace through a real interaction sequence:

**Round 1**: Agent sends ROI analysis to Finance.

- Finance engagement increases slightly
- Reward: +0.03 (info gain), +0.01 (goal progress)

**Round 2**: Agent sends implementation timeline to TechLead.

- TechLead responds with technical questions
- Legal's engagement drops—maybe they expected to be looped in earlier?
- Reward: +0.02 (causal influence), -0.01 (risk from incomplete engagement)

**Round 3**: Agent sends DPA (Data Processing Agreement) to Legal and copies Finance.

- Legal engagement stabilizes
- Finance engagement rises
- Committee vote shifts from "mixed" to "leaning positive"
- Reward: +0.04 (trust + info), +0.02 (goal)

**Round 4**: Agent proposes terms to all stakeholders.

- Procurement raises concern about reference customers
- Two stakeholders enter "veto precursor" state
- Reward: -0.02 (risk). Message: "Deal stalling."

**Round 5**: Agent addresses Procurement's concern with specific customer references.

- Procurement engagement improves
- Veto precursors clear
- Deal momentum shifts to "progressing"
- Reward: +0.05 (goal + trust). Stage advances.

The agent doesn't just optimize the next move—it plans across the full arc of the negotiation, anticipating how one stakeholder's concerns might cascade to others.

---

## The Technical Layer

Under the hood, DealRoom is framed as a **Markov Decision Process (MDP)** with partial observability:

- **State**: The full DealRoomState includes each stakeholder's belief distribution, engagement levels, the hidden causal graph, deal stage, and history
- **Observation**: The agent sees only the observable signals (messages, engagement, weak signals, deal stage)
- **Actions**: 8 action types with parameters (target stakeholder, message content, document type)
- **Reward**: 5-dimensional vector shaped per round, with a terminal reward signal from the CCIGrader

The environment uses:

- **Bayesian belief updates**: Stakeholder beliefs evolve based on actions using `belief_tracker.py`
- **CVaR risk modeling**: Legal and Finance stakeholders evaluate proposals using Conditional Value at Risk, not just expected value
- **Committee deliberation**: 3-step (or 4-step for hostile scenarios) belief propagation through the influence graph
- **Lookahead simulation**: Optional minimax-robust planning with a 0.07 exact cost penalty

The environment is built on **OpenEnv** (from Meta/PyTorch), making it compatible with standard RL tooling like Hugging Face TRL.

---

## What We've Learned

Running experiments in DealRoom has surfaced insights that aren't obvious from theory:

**Document sequencing matters more than message content.** In the early rounds, _which_ artifact you send matters more than _what you say_. Sending the DPA to Legal before the ROI to Finance creates a different committee dynamic than the reverse order—even if the messages are identical.

**Trust is easy to lose and hard to regain.** A single infeasible promise creates a permanent mark on the trust track. The agent can still close the deal, but it will have to work significantly harder in subsequent rounds to overcome the damage.

**Veto warnings are not vetoes.** The system flags "veto precursors" before they become actual blocks. A well-timed concession or evidence package can clear the warning. But if the agent ignores the signal for 2-3 rounds, the precursor becomes a hard veto—and recovery becomes much harder.

**The "aligned" scenario is deceptively difficult.** You'd expect the low-friction scenario to be simple. But even there, getting all six stakeholders to approval thresholds requires careful sequencing. The baseline performance isn't a ceiling—it's a genuine challenge.

---

## Why This Matters

Enterprise negotiation isn't a niche problem. It touches sales, procurement, partnership management, and executive alignment at every company that sells complex B2B software or services.

And it's a domain where AI can genuinely help—if it can reason about multi-party dynamics, not just individual preferences. The ability to model how one stakeholder's concerns might ripple to others, to plan across multiple rounds, to understand when to push and when to wait—these are exactly the capabilities that make AI useful beyond simple Q&A.

For AI researchers, DealRoom offers a reproducible benchmark where the challenge is strategic reasoning, partial observability, and credit assignment—not just getting to a known optimal policy.

---

## Try It Yourself

You can interact with DealRoom directly on our Hugging Face Space:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g2B0AEz2M8lyLQzlU3QKcgdLyNyengsy?usp=sharing)

**The interface gives you:**

- A view of all six stakeholders with their roles and current engagement status
- The ability to select any stakeholder and see their message history and what they need
- Context-aware action suggestions or free-form message composition
- Real-time tracking of your score across the five reward dimensions
- Visibility into deal stage progression and veto warnings

**What to observe:**

- How does engagement change after you send a document versus a direct message?
- What happens to Finance's engagement when Legal raises a compliance concern?
- Can you identify a "weak signal" before it becomes a veto precursor?
- How does the committee's overall mood shift after a group proposal versus individual outreach?

The interface is designed to make committee dynamics visible. Use it to build intuition about negotiation strategy, then explore how different action sequences affect outcomes.

---

## The Deal That Didn't Close

Here's the thing about enterprise negotiation: sometimes you do everything right and the deal still falls apart. A competitor emerges. Budget gets frozen. The executive sponsor moves to another company.

DealRoom doesn't promise that perfect play always wins. It promises that you'll understand _why_ a deal succeeded or failed—and that's actually more useful.

Because in the room where deals go to die, the winners aren't the ones who never lost. They're the ones who learned to read the signals earlier.

---

_DealRoom is an OpenEnv-compatible reinforcement learning environment for enterprise B2B negotiation. Explore the code, try the interface, and dig into the research on [GitHub](https://github.com/akshaypulla/deal_room_S2P)._
