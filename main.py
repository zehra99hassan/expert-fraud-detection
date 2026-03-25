"""
main.py
────────
Fake Expert Detector — Full RL Simulation
==========================================
Combines three detection layers:
 
  Layer 1: Static Signal Scoring    (profile snapshot)
  Layer 2: Profile Delta Analysis   (how the profile changed over time)
  Layer 3: LLM Answer Detection     (was this answer written by ChatGPT?)
 
The combined score feeds into a Q-learning RL agent that learns
which action (approve / flag / reject) to take per score range.
 
Run:
    python main.py              # auto mode, 1 episode
    python main.py --episodes 20  # train agent over 20 episodes
    python main.py --mode manual  # you make the decisions
"""
 
import argparse
import time
import random
 
from profile_delta import compute_delta, score_delta, explain_delta, build_sample_deltas
from llm_detector import analyze_answer, explain_llm_signals
from agent import QLearningAgent
 
 
# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────
 
C = {
    "green":  "\033[92m", "red":    "\033[91m", "yellow": "\033[93m",
    "cyan":   "\033[96m", "bold":   "\033[1m",  "dim":    "\033[2m",
    "reset":  "\033[0m",
}
def c(text, color): return f"{C[color]}{text}{C['reset']}"
def bar(score, width=28):
    filled = int((score / 100) * width)
    col = "red" if score >= 75 else "yellow" if score >= 40 else "green"
    return c("█" * filled, col) + c("░" * (width - filled), "dim")
 
 
# ─────────────────────────────────────────────
# CANDIDATE DATA
# ─────────────────────────────────────────────
 
CANDIDATES = [
    {
        "name": "Amir Hassan",
        "role": "Senior ML Engineer",
        "ground_truth": "fraud",
        "static_signals": {
            "career_jumps": 4, "skills_added_recently": 23,
            "buzzword_density": 0.9, "answer_specificity": 0.1,
            "has_concrete_examples": False, "profile_age_months": 3,
            "endorsements_from_unknown": 12,
        },
        "answer": (
            "I leveraged cutting-edge monitoring pipelines and proactively synergized "
            "with the team to deliver a seamless, SOTA remediation strategy. "
            "Our hyper-scalable solution was AGI-ready and future-proof across all verticals."
        ),
    },
    {
        "name": "Priya Mehta",
        "role": "Backend Engineer",
        "ground_truth": "legit",
        "static_signals": {
            "career_jumps": 1, "skills_added_recently": 1,
            "buzzword_density": 0.1, "answer_specificity": 0.9,
            "has_concrete_examples": True, "profile_age_months": 72,
            "endorsements_from_unknown": 1,
        },
        "answer": (
            "Our Redis cache hit 95% memory at 2x traffic during a flash sale. "
            "I traced it with redis-cli INFO and found a hot key — one product ID "
            "getting 80% of reads. We moved it to a dedicated shard. "
            "Took about 40 minutes and cut p99 latency by 60%. "
            "In hindsight we should have set up key-level monitoring earlier."
        ),
    },
    {
        "name": "Sara Khan",
        "role": "Data Scientist",
        "ground_truth": "ambiguous",
        "static_signals": {
            "career_jumps": 2, "skills_added_recently": 5,
            "buzzword_density": 0.5, "answer_specificity": 0.5,
            "has_concrete_examples": True, "profile_age_months": 36,
            "endorsements_from_unknown": 4,
        },
        "answer": (
            "I usually try SMOTE or class weights depending on the dataset. "
            "I once used class_weight='balanced' in sklearn on a churn model — "
            "precision improved but recall was tricky to tune further. "
            "It was challenging to explain the trade-off to stakeholders."
        ),
    },
    {
        "name": "Reza Tehrani",
        "role": "Cloud Architect",
        "ground_truth": "fraud",
        "static_signals": {
            "career_jumps": 5, "skills_added_recently": 40,
            "buzzword_density": 0.95, "answer_specificity": 0.05,
            "has_concrete_examples": False, "profile_age_months": 2,
            "endorsements_from_unknown": 20,
        },
        "answer": (
            "I architected a hyper-scalable, future-proof, multi-cloud solution "
            "that leveraged best practices across all verticals and stakeholder ecosystems. "
            "End-to-end synergy was achieved across all cloud-native touchpoints "
            "enabling seamless digital transformation at enterprise scale."
        ),
    },
]
 
 
# ─────────────────────────────────────────────
# STATIC SIGNAL SCORING
# ─────────────────────────────────────────────
 
def compute_static_score(sig: dict) -> float:
    """
    Score 0–100 from static profile signals.
 
    Weight rationale:
      Career jumps + sudden skills → highest weight because these are
      hardest to fake retroactively. You can polish an answer overnight
      but you can't rewrite 3 years of job history convincingly.
 
      Buzzwords + answer vagueness → medium weight. Useful signal but
      a coached candidate can partially fake these.
 
      Profile age + ghost endorsements → low weight. Noisy signals —
      legitimate reasons exist (career change, new LinkedIn account).
    """
    s = sig
    score = 0.0
    score += min(s["career_jumps"] * 8, 25)           # max 25
    score += min(s["skills_added_recently"] * 2, 20)  # max 20
    score += s["buzzword_density"] * 20                # max 20
    vagueness = (1.0 - s["answer_specificity"]) * 15
    if not s["has_concrete_examples"]:
        vagueness += 5
    score += vagueness                                 # max 20
    if s["profile_age_months"] < 6:
        score += 10
    elif s["profile_age_months"] < 12:
        score += 5                                     # max 10
    score += min(s["endorsements_from_unknown"] * 0.5, 5)  # max 5
    return round(min(score, 100), 1)
 
 
# ─────────────────────────────────────────────
# COMBINED SCORE
# ─────────────────────────────────────────────
 
def combined_score(static: float, delta: float, llm: float) -> float:
    """
    Weighted combination of all three detection layers.
 
    Weights:
      Static signals  40% — profile snapshot (reliable but gameable)
      Profile delta   35% — change over time (hardest to fake)
      LLM detection   25% — answer linguistics (useful but noisy)
    """
    return round((static * 0.40) + (delta * 0.35) + (llm * 0.25), 1)
 
 
# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────
 
def print_candidate(cand, static_s, delta_s, delta_sig, llm_sig, combined, action, reward, correct, episode_mode=False):
    gt = cand["ground_truth"]
    print(f"\n{'─'*62}")
    print(c(f"  {cand['name']}", "bold") + f"  —  {cand['role']}")
    print(f"{'─'*62}")
 
    # Three layer scores
    print(f"\n  {'DETECTION LAYERS':}")
    print(f"  {'Layer 1 · Static signals:':<32} {static_s:>5}/100  [{bar(static_s)}]")
    print(f"  {'Layer 2 · Profile delta:':<32} {delta_s:>5}/100  [{bar(delta_s)}]")
    print(f"  {'Layer 3 · LLM answer detection:':<32} {llm_sig.llm_score:>5}/100  [{bar(llm_sig.llm_score)}]")
    print(f"  {'─'*58}")
    print(f"  {'Combined fraud score:':<32} {combined:>5}/100  [{bar(combined)}]")
 
    # Delta explanation
    if delta_sig:
        expl = explain_delta(delta_sig, delta_s)
        print(f"\n  Profile delta: {c(expl, 'yellow') if delta_s > 40 else expl}")
 
    # LLM signals
    llm_notes = explain_llm_signals(llm_sig)
    print(f"\n  Answer analysis:")
    for note in llm_notes:
        col = "yellow" if llm_sig.llm_score > 40 else "dim"
        print(f"    · {c(note, col) if llm_sig.llm_score > 40 else note}")
 
    # Decision
    ac = "red" if action == "reject" else "yellow" if action == "flag" else "green"
    rc = "green" if reward > 0 else "red" if reward < 0 else "yellow"
    ok = c("✓ correct", "green") if correct else c("✗ wrong", "red")
    print(f"\n  Decision:     {c(action.upper(), ac)}")
    print(f"  Ground truth: {gt.upper()}")
    print(f"  Reward:       {c(f'{reward:+.1f}', rc)}   {ok}")
 
 
def print_summary(agent: QLearningAgent, total: int, episodes: int):
    correct = sum(1 for h in agent.history if
                  (h["action"] == "reject" and h["ground_truth"] == "fraud") or
                  (h["action"] == "approve" and h["ground_truth"] == "legit") or
                  (h["action"] == "flag" and h["ground_truth"] == "ambiguous"))
    accuracy = correct / len(agent.history) * 100 if agent.history else 0
 
    print(f"\n{'═'*62}")
    print(c("  SIMULATION COMPLETE", "bold"))
    print(f"{'═'*62}")
    print(f"  Episodes run:      {episodes}")
    print(f"  Total evaluations: {len(agent.history)}")
    print(f"  Correct decisions: {correct} / {len(agent.history)}")
    print(f"  Accuracy:          {accuracy:.0f}%")
    rc = "green" if agent.total_reward > 0 else "red"
    print(f"  Total RL reward:   {c(f'{agent.total_reward:+.1f}', rc)}")
 
    print(f"\n  Learned Q-table (what the agent now prefers per score range):")
    agent.print_q_table()
    print(f"{'═'*62}\n")
 
 
# ─────────────────────────────────────────────
# SIMULATION MODES
# ─────────────────────────────────────────────
 
def run_episode(agent: QLearningAgent, deltas_map: dict, verbose: bool = True):
    """Run one full episode across all candidates."""
    for cand in CANDIDATES:
        static_s = compute_static_score(cand["static_signals"])
        llm_sig = analyze_answer(cand["answer"])
 
        delta_sig, delta_s = None, 0.0
        if cand["name"] in deltas_map:
            old, new, months = deltas_map[cand["name"]]
            delta_sig = compute_delta(old, new, months)
            delta_s = score_delta(delta_sig)
 
        combined = combined_score(static_s, delta_s, llm_sig.llm_score)
 
        action = agent.select_action(combined)
        reward = agent.update(combined, action, cand["ground_truth"])
        correct = reward > 0
 
        if verbose:
            print_candidate(cand, static_s, delta_s, delta_sig, llm_sig,
                            combined, action, reward, correct)
            time.sleep(0.2)
 
 
def run_manual(agent: QLearningAgent, deltas_map: dict):
    print(c("\n  MODE: Manual — you make the calls\n", "cyan"))
    for cand in CANDIDATES:
        static_s = compute_static_score(cand["static_signals"])
        llm_sig = analyze_answer(cand["answer"])
 
        delta_sig, delta_s = None, 0.0
        if cand["name"] in deltas_map:
            old, new, months = deltas_map[cand["name"]]
            delta_sig = compute_delta(old, new, months)
            delta_s = score_delta(delta_sig)
 
        combined = combined_score(static_s, delta_s, llm_sig.llm_score)
 
        print(f"\n{'─'*62}")
        print(c(f"  {cand['name']}", "bold") + f"  —  {cand['role']}")
        print(f"  Combined fraud score: {combined}/100  [{bar(combined)}]")
        print(f"  Answer (excerpt): {cand['answer'][:100]}...")
        print(f"\n  [a] approve   [f] flag   [r] reject")
 
        while True:
            choice = input("  Your decision: ").strip().lower()
            if choice in ("a", "approve"):   action = "approve"; break
            elif choice in ("f", "flag"):    action = "flag";    break
            elif choice in ("r", "reject"):  action = "reject";  break
            else: print("  Enter a, f, or r.")
 
        reward = agent.update(combined, action, cand["ground_truth"])
        correct = reward > 0
        print_candidate(cand, static_s, delta_s, delta_sig, llm_sig,
                        combined, action, reward, correct)
 
 
# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="Fake Expert Detector — RL Simulation")
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of training episodes in auto mode (default: 1)")
    args = parser.parse_args()
 
    print(c("\n  ╔══════════════════════════════════════════╗", "bold"))
    print(c("  ║   FAKE EXPERT DETECTOR  v2.0              ║", "bold"))
    print(c("  ║   3-Layer RL Detection System             ║", "bold"))
    print(c("  ╚══════════════════════════════════════════╝", "bold"))
    print(f"\n  Layers: Static Signals · Profile Delta · LLM Detection")
    print(f"  Agent:  Q-Learning (ε-greedy, α=0.3, γ=0.1)\n")
 
    # Build delta lookup map
    raw_deltas = build_sample_deltas()
    deltas_map = {name: (old, new, months) for name, old, new, months in raw_deltas}
 
    agent = QLearningAgent()
 
    if args.mode == "manual":
        run_manual(agent, deltas_map)
        print_summary(agent, len(CANDIDATES), 1)
 
    else:
        episodes = args.episodes
        if episodes == 1:
            print(c("  MODE: Auto — single episode\n", "cyan"))
            run_episode(agent, deltas_map, verbose=True)
        else:
            print(c(f"  MODE: Auto — training over {episodes} episodes\n", "cyan"))
            print("  (Only final episode shown in detail)\n")
            for ep in range(1, episodes + 1):
                verbose = (ep == episodes)
                if not verbose and ep % 5 == 0:
                    print(f"  Episode {ep:>3}/{episodes}  |  "
                          f"ε={agent.epsilon:.3f}  |  "
                          f"reward={agent.total_reward:+.1f}")
                run_episode(agent, deltas_map, verbose=verbose)
 
        print_summary(agent, len(CANDIDATES), episodes)
 
 
if __name__ == "__main__":
    main()