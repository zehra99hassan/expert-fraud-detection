"""
agent.py
─────────
A Q-learning agent that learns to classify candidates as fraud/legit
based on discretized state (fraud score buckets).

Why Q-learning here instead of a fixed policy?
  The task asks for an RL framework. A fixed threshold policy
  (if score > 75: reject) is rule-based, not RL. A Q-learning agent
  actually updates its action preferences based on reward feedback —
  it can discover that the threshold should be 70, not 75, purely
  from experience.

State space:
  We bucket the combined fraud score (0–100) into 10 discrete bins.
  Bin 0 = score 0–9, Bin 1 = score 10–19, ..., Bin 9 = score 90–100.

Action space:
  0 = approve
  1 = flag
  2 = reject

Q-table:
  10 states × 3 actions = 30 values.
  Updated via: Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s') - Q(s,a)]
  (Standard Bellman update, single-step since episodes are independent.)
"""

import random
from dataclasses import dataclass, field


ACTIONS = {0: "approve", 1: "flag", 2: "reject"}
ACTION_LABELS = list(ACTIONS.values())

REWARD_TABLE = {
    ("reject",  "fraud"):     +2.0,
    ("approve", "legit"):     +2.0,
    ("flag",    "ambiguous"): +1.0,
    ("flag",    "fraud"):     +1.0,
    ("approve", "ambiguous"): +0.5,
    ("flag",    "legit"):      0.0,
    ("reject",  "ambiguous"): -1.0,
    ("approve", "fraud"):     -3.0,
    ("reject",  "legit"):     -3.0,
}


@dataclass
class QLearningAgent:
    """
    Q-learning agent for expert fraud detection.

    Hyperparameters:
      alpha (learning rate): how fast Q-values update per experience
      gamma (discount):      future reward weight (low here — single-step)
      epsilon:               exploration rate (random action probability)
      epsilon_decay:         how fast epsilon shrinks as agent gains experience
      epsilon_min:           floor for epsilon (always explore a little)
    """
    alpha: float = 0.3
    gamma: float = 0.1
    epsilon: float = 1.0
    epsilon_decay: float = 0.92
    epsilon_min: float = 0.05
    num_states: int = 10
    num_actions: int = 3
    q_table: list = field(default_factory=list)
    episode: int = 0
    total_reward: float = 0.0
    history: list = field(default_factory=list)

    def __post_init__(self):
        # Initialize Q-table with small optimistic values
        # Slight positive bias → agent prefers action over doing nothing
        self.q_table = [[0.1] * self.num_actions for _ in range(self.num_states)]

    def _state(self, fraud_score: float) -> int:
        """Map a 0–100 fraud score to a discrete state bin (0–9)."""
        return min(int(fraud_score // 10), self.num_states - 1)

    def select_action(self, fraud_score: float) -> str:
        """
        Epsilon-greedy action selection.
        - With probability epsilon: explore (random action)
        - Otherwise: exploit (pick action with highest Q-value)
        """
        state = self._state(fraud_score)
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.num_actions - 1)
        else:
            action_idx = self.q_table[state].index(max(self.q_table[state]))
        return ACTIONS[action_idx]

    def update(self, fraud_score: float, action: str, ground_truth: str):
        """
        Update Q-table after observing reward for (state, action) pair.
        Uses single-step Bellman update (no next state — episodes are i.i.d.)
        """
        state = self._state(fraud_score)
        action_idx = ACTION_LABELS.index(action)
        reward = REWARD_TABLE.get((action, ground_truth), 0.0)

        old_q = self.q_table[state][action_idx]
        # Single-step: no next state, so target = reward only
        self.q_table[state][action_idx] = old_q + self.alpha * (reward - old_q)

        self.total_reward += reward
        self.episode += 1
        self.history.append({
            "episode": self.episode,
            "fraud_score": fraud_score,
            "state_bin": state,
            "action": action,
            "ground_truth": ground_truth,
            "reward": reward,
            "epsilon": round(self.epsilon, 3),
        })

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return reward

    def best_action_for_state(self, state: int) -> str:
        return ACTIONS[self.q_table[state].index(max(self.q_table[state]))]

    def print_q_table(self):
        """Print the learned Q-table in a readable format."""
        header = f"  {'Score Range':<16} {'Approve Q':>10} {'Flag Q':>10} {'Reject Q':>10}  {'Best Action'}"
        print(header)
        print("  " + "─" * 60)
        for i, row in enumerate(self.q_table):
            low, high = i * 10, i * 10 + 9
            best = self.best_action_for_state(i)
            print(f"  {low:>3}–{high:<10}     {row[0]:>8.3f}   {row[1]:>8.3f}   {row[2]:>8.3f}  {best}")