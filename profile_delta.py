"""
profile_delta.py
─────────────────
Analyzes changes in a candidate's profile over time.

This is the key signal the task specifically asked for:
"historical profile deltas" — not just what a profile looks like NOW,
but HOW it changed. Natural career growth looks very different from
overnight inflation.

Key insight:
  A real expert accumulates skills over years, changes jobs with
  logical progression, and builds endorsements from real colleagues.
  A ChatGPT expert inflates everything at once — right before applying.
"""

from dataclasses import dataclass


@dataclass
class ProfileSnapshot:
    """A point-in-time snapshot of a candidate's LinkedIn profile."""
    timestamp_label: str       # e.g. "6 months ago", "today"
    num_skills: int
    num_endorsements: int
    num_connections: int
    job_title: str
    years_experience_claimed: int
    certifications: int


@dataclass
class DeltaSignals:
    """
    Derived signals from comparing two profile snapshots.
    These are harder to fake than a static profile — you can't
    rewrite history convincingly across time.
    """
    skills_added: int           # raw count added between snapshots
    endorsements_added: int
    connections_added: int
    experience_jump_years: int  # claimed experience delta (suspicious if > real time elapsed)
    certs_added: int
    title_changed: bool
    months_elapsed: int         # real time between snapshots


def compute_delta(old: ProfileSnapshot, new: ProfileSnapshot, months_elapsed: int) -> DeltaSignals:
    """Compare two snapshots and extract delta signals."""
    return DeltaSignals(
        skills_added=max(new.num_skills - old.num_skills, 0),
        endorsements_added=max(new.num_endorsements - old.num_endorsements, 0),
        connections_added=max(new.num_connections - old.num_connections, 0),
        experience_jump_years=max(new.years_experience_claimed - old.years_experience_claimed, 0),
        certs_added=max(new.certifications - old.certifications, 0),
        title_changed=(old.job_title != new.job_title),
        months_elapsed=months_elapsed,
    )


def score_delta(delta: DeltaSignals) -> float:
    """
    Score how suspicious a profile's changes are. Returns 0–100.

    Design rationale:
    - Skills/month is the strongest signal: real engineers add ~1-2 skills/year
      A spike of 15+ skills in one month = bulk import before applying
    - Experience inflation is a hard fraud signal: you cannot gain 3 years
      of experience in 6 months of real time
    - Certification spikes matter less (people do binge-study)
    - Connection spikes alone are weak (networking surges happen)
    """
    score = 0.0
    months = max(delta.months_elapsed, 1)  # avoid divide-by-zero

    # Skills added per month (max 35 pts)
    skills_per_month = delta.skills_added / months
    if skills_per_month > 5:
        score += 35
    elif skills_per_month > 2:
        score += 20
    elif skills_per_month > 1:
        score += 10

    # Experience inflation — impossible growth (max 30 pts)
    # If someone claims 2+ more years of experience than real time elapsed, that's fraud
    impossible_exp = max(delta.experience_jump_years - (months / 12), 0)
    score += min(impossible_exp * 15, 30)

    # Certification binge (max 15 pts)
    certs_per_month = delta.certs_added / months
    if certs_per_month > 2:
        score += 15
    elif certs_per_month > 1:
        score += 8

    # Endorsement spike without time to earn them (max 10 pts)
    endorsements_per_month = delta.endorsements_added / months
    if endorsements_per_month > 10:
        score += 10
    elif endorsements_per_month > 5:
        score += 5

    # Sudden title upgrade alongside everything else (max 10 pts)
    if delta.title_changed and score > 20:
        score += 10

    return round(min(score, 100), 1)


def explain_delta(delta: DeltaSignals, score: float) -> str:
    """Generate a human-readable explanation of why the delta is suspicious."""
    months = max(delta.months_elapsed, 1)
    lines = []

    skills_per_month = delta.skills_added / months
    if skills_per_month > 2:
        lines.append(
            f"Added {delta.skills_added} skills in {months} months "
            f"({skills_per_month:.1f}/month — natural rate is ~0.2/month)"
        )

    impossible_exp = max(delta.experience_jump_years - (months / 12), 0)
    if impossible_exp > 0:
        lines.append(
            f"Claims {delta.experience_jump_years} more years of experience "
            f"but only {months/12:.1f} real years elapsed — "
            f"{impossible_exp:.1f} years unaccounted for"
        )

    if delta.certs_added > 2:
        lines.append(f"Added {delta.certs_added} certifications in {months} months")

    if delta.title_changed:
        lines.append("Job title changed alongside rapid profile inflation")

    if not lines:
        lines.append("Profile growth looks natural and consistent with real time elapsed")

    return " | ".join(lines)


# ─────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────

def build_sample_deltas():
    """
    Returns (candidate_name, old_snapshot, new_snapshot, months_elapsed)
    for each simulated candidate.
    """
    return [
        (
            "Amir Hassan",
            ProfileSnapshot("6 months ago", num_skills=12, num_endorsements=8,
                            num_connections=150, job_title="Junior Developer",
                            years_experience_claimed=1, certifications=0),
            ProfileSnapshot("today", num_skills=47, num_endorsements=31,
                            num_connections=820, job_title="Senior ML Engineer",
                            years_experience_claimed=6, certifications=5),
            6,  # months elapsed
        ),
        (
            "Priya Mehta",
            ProfileSnapshot("6 months ago", num_skills=18, num_endorsements=24,
                            num_connections=310, job_title="Backend Engineer",
                            years_experience_claimed=4, certifications=2),
            ProfileSnapshot("today", num_skills=20, num_endorsements=27,
                            num_connections=340, job_title="Backend Engineer",
                            years_experience_claimed=5, certifications=2),
            6,
        ),
        (
            "Sara Khan",
            ProfileSnapshot("6 months ago", num_skills=14, num_endorsements=11,
                            num_connections=200, job_title="Data Analyst",
                            years_experience_claimed=2, certifications=1),
            ProfileSnapshot("today", num_skills=22, num_endorsements=18,
                            num_connections=280, job_title="Data Scientist",
                            years_experience_claimed=3, certifications=3),
            6,
        ),
        (
            "Reza Tehrani",
            ProfileSnapshot("3 months ago", num_skills=5, num_endorsements=2,
                            num_connections=90, job_title="IT Support",
                            years_experience_claimed=0, certifications=0),
            ProfileSnapshot("today", num_skills=52, num_endorsements=41,
                            num_connections=910, job_title="Cloud Architect",
                            years_experience_claimed=8, certifications=9),
            3,
        ),
    ]