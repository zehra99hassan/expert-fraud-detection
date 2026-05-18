# Expert Fraud Detection using Reinforcement Learning and NLP

> Identify candidates who overstate their experience using AI assistance.

A 3-layer detection system combining static profile signals, historical profile delta analysis, and LLM-pattern detection — with a Q-learning agent that improves its decisions over multiple episodes.

---

## The Problem

Traditional fraud detection verifies identity. This system solves a harder problem: **verifying depth of knowledge**.

A "ChatGPT expert" is someone who:
- Uses AI to write polished screening answers they couldn't write themselves
- Inflates their LinkedIn profile with bulk-imported skills overnight
- Claims years of experience they never had

They pass identity checks. They fail experience checks — if you know what to look for.

---

## Core Insight

- **Fake experts** = polished but shallow. No failures mentioned. No specific tools. Perfect prose.
- **Real experts** = specific and sometimes imperfect. They name exact tools, mention what went wrong, and write unevenly.

---

## Architecture: 3 Detection Layers

### Layer 1 — Static Signal Scoring (`main.py`)

Scores the current profile snapshot 0–100.

Signal weight rationale (most important first):

- **Career jumps** — HIGHEST weight — You can polish an answer overnight. You cannot rewrite 3 years of job history convincingly.
- **Skills added recently** — HIGH weight — Real engineers add ~1-2 skills per year. 20+ skills in a week = bulk import before applying.
- **Buzzword density** — MEDIUM weight — Useful but gameable by a coached candidate.
- **Answer specificity** — MEDIUM weight — Real experts give imperfect, specific answers. Fakes give vague, universal ones.
- **Concrete examples** — MEDIUM weight — Fakes describe outcomes ("delivered results"). Experts describe the process ("I traced it with redis-cli and found...").
- **Profile age** — LOW weight — Noisy. Legitimate reasons exist for new profiles.
- **Unknown endorsements** — LOW weight — Easy to game, weak signal alone.

### Layer 2 — Profile Delta Analysis (`profile_delta.py`)

Compares profile snapshots across time. **This is the hardest signal to fake.**

Key metrics:
- Skills added per month (natural rate: ~0.2/month)
- Experience inflation vs real time elapsed (claiming 5 new years in 6 months = impossible)
- Certification binge rate
- Title changes alongside other inflation

### Layer 3 — LLM Answer Detection (`llm_detector.py`)

Detects whether a screening answer was likely written by ChatGPT.

Signals:
- Buzzword and power verb density
- Absence of failure or trade-off language
- No first-person imperfection ("I struggled", "we were wrong")
- No specific tools named
- Suspiciously uniform sentence length variance

### Combined Score

```
Combined = (Static × 0.40) + (Delta × 0.35) + (LLM × 0.25)
```

Delta gets high weight because it's the hardest to fabricate.

---

## RL Agent (`agent.py`)

A Q-learning agent that learns which action (approve / flag / reject) to take per score range.

- **State**: fraud score bucketed into 10 bins (0-9, 10-19, ... 90-99)
- **Actions**: approve, flag, reject
- **Exploration**: ε-greedy (starts random, converges to learned policy)
- **Update rule**: Bellman single-step — `Q(s,a) ← Q(s,a) + α(r - Q(s,a))`

### Reward Function

```
Reject  + Fraud      →  +2.0   caught a fake
Approve + Legit      →  +2.0   real expert passed through
Flag    + Ambiguous  →  +1.0   uncertain case handled carefully
Flag    + Fraud      →  +1.0   cautious but not confident
Approve + Ambiguous  →  +0.5   risky but not harmful
Flag    + Legit      →   0.0   wasted review time, no harm
Reject  + Ambiguous  →  -1.0   too aggressive
Approve + Fraud      →  -3.0   WORST: fraud slipped through
Reject  + Legit      →  -3.0   WORST: real expert wrongly rejected
```

The asymmetric -3 penalty makes the system cautious over aggressive.

---

## Run It

No dependencies — pure Python 3.

```bash
# Single episode (agent explores randomly)
python main.py

# Train over 20 episodes (agent learns and converges)
python main.py --episodes 20

# Manual mode — you make each call, system scores you
python main.py --mode manual
```

---

## Results

After 20 training episodes:

```
Episodes run:      20
Total evaluations: 80
Correct decisions: 68 / 80
Accuracy:          85%
Total RL reward:   +114.0

Learned Q-table:
  Score Range    Approve Q    Flag Q   Reject Q   Best Action
  10–19            +1.997     +0.070     -0.830   approve
  20–29            +0.304     +0.997     -0.461   flag
  80–89            -0.830     +0.559     +1.996   reject
  90–99            -0.830     +0.559     +1.996   reject
```

The agent learned the correct policy purely from reward feedback — no hardcoded thresholds.

---

## Bonus: Live Video Interview Detection

To detect fraud in real-time video interviews, three signal layers matter:

**Visual — Eye movement**
Frequent off-screen glances in a consistent direction suggest reading from a secondary monitor. The key signal is gaze shifts on unexpected follow-up questions — prepared fraudsters have smooth eye movement on scripted answers but break pattern when surprised.

**Audio — Response latency**
Real experts answer fast on easy questions and slow on hard ones — latency follows difficulty. Fraudsters show flat, uniform latency (always slow = reading) or suspiciously short latency on complex questions (pre-generated answer ready).

**Screen — Activity signals** *(if screen share enabled)*
Alt-tab events, clipboard spikes, or browser focus-loss during questions are detectable via screen share metadata. These are strong signals because they require an action, not just a passive behavior.

**RL training for live detection:**
Each interview frame becomes a state. Actions remain approve / flag / reject. Ground truth comes from post-hire performance reviews, creating a delayed reward signal. The model trains across many interviews and learns to weight signal co-occurrence — one signal alone is noise, two or more simultaneously is fraud.

**Rule of thumb:** flag for human review if any 2 of 3 signals are active simultaneously. Never auto-reject — false positives on nervous-but-legitimate candidates are costly and legally risky.

---

## Project Structure

```
fake-expert-detector/
├── main.py           ← entry point, static scoring, combined score, simulation runner
├── profile_delta.py  ← historical profile change analysis
├── llm_detector.py   ← LLM-generated answer detection
├── agent.py          ← Q-learning RL agent
└── README.md
```

---

## Author

Built as a take-home assessment submission for an ML Engineer role.
Focus: human-led signal prioritization over ML complexity.
