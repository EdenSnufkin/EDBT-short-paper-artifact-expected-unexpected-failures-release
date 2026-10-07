# webapp/settings.py
"""Central configuration of the study web application.

Every knob of the study lives here; nothing else in the package hardcodes these values. The values below are the
ones used in the study. Deployment-specific values (question file, database, Prolific codes) are read from
environment variables so that no secret has to be committed.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# -------------------------
# Question bank
# -------------------------
# JSON file with the questions (format: see data/README.md). The repository ships a small demo bank; the
# questions of the study are not distributed, point this variable to your own bank to run your own study.
QUESTIONS_PATH = Path(os.environ.get("STUDY_QUESTIONS_PATH", HERE / "data" / "demo_questions.json"))
TOPIC = "Maths"  # the study tracks one skill per session; every question of the bank is assigned to it

# -------------------------
# Recommendation method
# -------------------------
# With RANDOMIZE_RECOMMENDATION_METHOD each new participant gets the choice that currently has the fewest
# sessions (assignment.pick_balanced). RECOMMENDATION_METHOD is the fallback when randomisation is off.
RANDOMIZE_RECOMMENDATION_METHOD = True
RECOMMENDATION_METHOD_CHOICES = ["hillclimbing", "mab"]
RECOMMENDATION_METHOD = "hillclimbing"  # "hillclimbing" | "mab"

HILLCLIMBING_ALGO = "MOO"                 # "MOO" | "MOEG" | "MOAG" | "MOAE"
MAB_ALGO = "thompson"                     # "thompson" | "egreedy" | "softmax" | "ucb" | "random"
MAB_ARMS = ("MOO", "MOAE", "MOAG", "MOEG")
HC_TIMES = 20                             # random restarts per batch recommendation

# -------------------------
# Session structure
# -------------------------
BATCH_SIZE = 3                            # questions per batch
N_BATCHES = 15                            # maximum number of batches
MASTERY_STOP_THRESHOLD = 0.9              # stop early once mastery reaches this value
INIT_MASTERY = 0.2
NCC_WINDOW = 2                            # consecutive successes at a difficulty needed to raise mastery
PRETEST_N = 5                             # pretest questions, spread over the difficulty range

# -------------------------
# Failure scores (paper, Sec. 3)
# -------------------------
DELTA = 0.20                              # tolerance around the mastery estimate
LAMBDA = 0.20                             # decay applied on correct answers

# -------------------------
# NASA-TLX-style survey scheduling
# -------------------------
# "end_only" | "fixed_count" | "every_failure" | "threshold" | "fixed_count_or_threshold"
# As for the recommendation method, RANDOMIZE_SURVEY_MODE assigns the least-filled choice to each new participant.
RANDOMIZE_SURVEY_MODE = True
SURVEY_MODE_CHOICES = ["fixed_count", "fixed_count_or_threshold"]
SURVEY_MODE = "fixed_count"
SURVEY_FIXED_COUNT = 3                    # surveys spread evenly over the planned batches
SURVEY_FAILURE_THRESHOLD = 0.5            # AUFS value that triggers a survey (tau)

# -------------------------
# Prolific (optional)
# -------------------------
# Set both variables to enable the redirect to Prolific at the end of the study; without them the application shows a
# plain "thank you" page. The codes of the study are deliberately not part of this repository.
PROLIFIC_COMPLETION_CODE = os.environ.get("PROLIFIC_COMPLETION_CODE", "")
PROLIFIC_NO_BONUS_CODE = os.environ.get("PROLIFIC_NO_BONUS_CODE", "")
PROLIFIC_COMPLETION_URL_BASE = "https://app.prolific.com/submissions/complete?cc="

# Without a PROLIFIC_PID in the URL the app refuses to start, unless this is set (local demo / testing).
ALLOW_ANONYMOUS_DEMO = os.environ.get("STUDY_ALLOW_DEMO", "") == "1"

RANDOM_SEED = None                        # None: every session draws a fresh seed (stored in the database)


def completion_url(code: str) -> str:
    return f"{PROLIFIC_COMPLETION_URL_BASE}{code}" if code else ""
