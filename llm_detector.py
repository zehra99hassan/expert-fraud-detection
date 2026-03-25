"""
llm_detector.py
────────────────
Detects whether a screening answer was likely written by an LLM (ChatGPT, etc.)
rather than a real human expert recalling genuine experience.

Key insight from the task brief:
  "The fraud we are looking for is a ChatGPT expert."

Real experts write imperfectly. They hedge, digress, mention failures,
use domain-specific jargon naturally, and give uneven sentence lengths.
LLM-generated answers are suspiciously polished — uniform structure,
no failures mentioned, vague outcomes, heavy use of power verbs.

We score answers on 6 linguistic signals:
  1. Buzzword / power verb density
  2. Absence of failure or trade-off language
  3. Sentence length uniformity (LLMs are very consistent)
  4. Outcome vagueness (no real numbers or tool names)
  5. Absence of first-person imperfection ("I struggled", "we were wrong")
  6. Perfect structure (bullet-point thinking embedded in prose)
"""

import re
import math
from dataclasses import dataclass


# ─────────────────────────────────────────────
# SIGNAL WORD LISTS
# ─────────────────────────────────────────────

POWER_VERBS = {
    "leveraged", "synergized", "orchestrated", "spearheaded", "pioneered",
    "architected", "implemented", "optimized", "streamlined", "transformed",
    "delivered", "drove", "enabled", "ensured", "facilitated", "achieved",
    "executed", "established", "developed", "created", "built", "designed",
    "managed", "led", "oversaw", "coordinated", "aligned", "accelerated",
}

BUZZWORDS = {
    "cutting-edge", "state-of-the-art", "best-in-class", "world-class",
    "innovative", "robust", "scalable", "seamless", "holistic", "synergy",
    "paradigm", "ecosystem", "end-to-end", "stakeholder", "value-add",
    "best practices", "thought leader", "deep dive", "bandwidth", "agile",
    "proactive", "dynamic", "strategic", "leverage", "utilize", "impactful",
    "sota", "agi-ready", "hyper-scale", "future-proof", "cloud-native",
    "mission-critical", "game-changing", "next-generation", "disruptive",
}

FAILURE_WORDS = {
    "failed", "wrong", "mistake", "struggled", "difficult", "challenging",
    "took longer", "didn't work", "had to revert", "broke", "issue",
    "problem", "bug", "error", "bottleneck", "tradeoff", "trade-off",
    "downside", "limitation", "wasn't", "couldn't", "we were wrong",
    "underestimated", "unexpected", "surprised", "learned the hard way",
}

IMPERFECTION_PHRASES = {
    "i struggled", "we struggled", "it took us", "i was wrong", "we were wrong",
    "i didn't know", "i had to google", "i made a mistake", "it didn't work",
    "i learned", "that was a mistake", "looking back", "in hindsight",
    "i wish we had", "we underestimated",
}

SPECIFIC_TOOLS = {
    "postgres", "redis", "kafka", "kubernetes", "docker", "terraform",
    "pytorch", "tensorflow", "sklearn", "fastapi", "django", "react",
    "spark", "airflow", "dbt", "snowflake", "bigquery", "elasticsearch",
    "celery", "rabbitmq", "nginx", "grafana", "prometheus", "github actions",
    "pytest", "pandas", "numpy", "langchain", "pinecone", "qdrant",
}


# ─────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class LLMSignals:
    buzzword_count: int
    power_verb_count: int
    has_failure_language: bool
    has_imperfection: bool
    has_specific_tools: bool
    sentence_length_variance: float   # low = suspiciously uniform
    word_count: int
    llm_score: float                  # 0–100, higher = more likely LLM


# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────

def analyze_answer(answer: str) -> LLMSignals:
    """
    Analyze a screening answer for LLM-generation signals.
    Returns an LLMSignals object with a 0–100 score.
    """
    text_lower = answer.lower()
    words = text_lower.split()
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 10]

    # 1. Buzzword + power verb density
    buzzword_count = sum(1 for b in BUZZWORDS if b in text_lower)
    power_verb_count = sum(1 for w in words if w.rstrip('.,;:') in POWER_VERBS)

    # 2. Failure / trade-off language
    has_failure = any(f in text_lower for f in FAILURE_WORDS)

    # 3. First-person imperfection
    has_imperfection = any(p in text_lower for p in IMPERFECTION_PHRASES)

    # 4. Specific tool mentions (real experts name real tools)
    has_specific_tools = any(t in text_lower for t in SPECIFIC_TOOLS)

    # 5. Sentence length variance
    # LLMs produce very uniform sentence lengths — low variance is suspicious
    if len(sentences) >= 2:
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0

    # ── Score computation ──────────────────────────────────────────────
    score = 0.0

    # Buzzword density (max 25 pts)
    score += min(buzzword_count * 5, 25)

    # Power verb density (max 20 pts)
    score += min(power_verb_count * 4, 20)

    # No failure language = suspicious (max 20 pts)
    if not has_failure:
        score += 20

    # No imperfection = suspicious (max 15 pts)
    if not has_imperfection:
        score += 15

    # No specific tools mentioned (max 15 pts)
    if not has_specific_tools:
        score += 15

    # Low sentence variance = LLM-uniform prose (max 5 pts)
    if std_dev < 3.0 and len(sentences) >= 2:
        score += 5

    return LLMSignals(
        buzzword_count=buzzword_count,
        power_verb_count=power_verb_count,
        has_failure_language=has_failure,
        has_imperfection=has_imperfection,
        has_specific_tools=has_specific_tools,
        sentence_length_variance=round(std_dev, 2),
        word_count=len(words),
        llm_score=round(min(score, 100), 1),
    )


def explain_llm_signals(sig: LLMSignals) -> list:
    """Return a list of plain-English explanations for what was detected."""
    notes = []
    if sig.buzzword_count >= 3:
        notes.append(f"{sig.buzzword_count} buzzwords detected (LLM pattern)")
    if sig.power_verb_count >= 3:
        notes.append(f"{sig.power_verb_count} power verbs (polished, non-human cadence)")
    if not sig.has_failure_language:
        notes.append("No failure or trade-off language (real experts mention what went wrong)")
    if not sig.has_imperfection:
        notes.append("No first-person imperfection (real experts say 'I struggled' or 'we were wrong')")
    if not sig.has_specific_tools:
        notes.append("No specific tools named (real engineers mention exact tech they used)")
    if sig.sentence_length_variance < 3.0:
        notes.append(f"Low sentence variance ({sig.sentence_length_variance}) — suspiciously uniform prose")
    if not notes:
        notes.append("Answer reads naturally — no strong LLM patterns detected")
    return notes