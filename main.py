import random

# This simulation evaluates candidates based on profile inconsistency,
# LLM-like response patterns, and behavioral consistency.
# The agent classifies candidates as Fraud (1) or Genuine (0).

# -----------------------------
# Mock Candidate Data
# -----------------------------
# label: 1 = fraud, 0 = genuine
data = [
    {"profile_jump": 0.9, "llm_score": 0.8, "consistency": 0.2, "label": 1},
    {"profile_jump": 0.2, "llm_score": 0.1, "consistency": 0.9, "label": 0},
    {"profile_jump": 0.7, "llm_score": 0.6, "consistency": 0.3, "label": 1},
    {"profile_jump": 0.1, "llm_score": 0.2, "consistency": 0.8, "label": 0},
    {"profile_jump": 0.8, "llm_score": 0.7, "consistency": 0.4, "label": 1},
]

# -----------------------------
# Environment
# -----------------------------
class ExpertEnv:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def reset(self):
        self.index = 0
        return self.data[self.index]

    def step(self, action):
        current = self.data[self.index]

        # reward logic
        if action == current["label"]:
            reward = 1
        else:
            reward = -1

        self.index += 1
        done = self.index >= len(self.data)

        next_state = None if done else self.data[self.index]

        return next_state, reward, done

# -----------------------------
# Agent
# -----------------------------
class Agent:
    def decide(self, state):
        """
        Simple scoring logic:
        - High profile jump = suspicious
        - High llm score = suspicious
        - High consistency = genuine
        """

        score = (
            state["profile_jump"] +
            state["llm_score"] -
            state["consistency"]
        )

        # threshold decision
        return 1 if score > 1 else 0


# -----------------------------
# Simulation
# -----------------------------
env = ExpertEnv(data)
agent = Agent()

state = env.reset()
total_reward = 0

print("Starting Evaluation...\n")

while state is not None:
    action = agent.decide(state)

    print(f"Candidate: {state}")
    print(f"Predicted: {'Fraud' if action == 1 else 'Genuine'}")

    state, reward, done = env.step(action)
    total_reward += reward

    print(f"Reward: {reward}\n")

print("Final Total Reward:", total_reward)