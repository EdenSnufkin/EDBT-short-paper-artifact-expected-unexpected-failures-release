# Adaptive question recommender

A clean, dependency-light (NumPy only) implementation of the **recommender used in the user study** of the paper: a multi-objective *hill-climbing* batch recommender (AdUp) and a *multi-armed bandit* over its variants, on top of a simple learner model and the expected / unexpected failure scores of the paper.

In the study, every participant was randomly assigned one of the two recommendation strategies and answered a pretest followed by up to 15 batches of 3 questions. This package contains the recommendation logic as it ran, without the research scaffolding around it (simulation harnesses, fitted response models, plotting).

```bash
cd recommender
pip install -e ".[test]"
python examples/simulate_session.py --method hillclimbing --ability 0.6
python examples/simulate_session.py --method mab --policy thompson --ability 0.4 --seed 3
pytest                                   # 27 tests, a few seconds
```

## How it works

### Learner model (`learner.py`)
Mastery is an *estimate* in [0, 1] per topic, starting at 0.2. It rises to a question's difficulty when the learner answers `ncc_window = 2` consecutive questions **of that exact difficulty** correctly. A failure erases the evidence collected at the failed difficulty and at every harder difficulty. Mastery is never lowered. The learner also keeps the sets of cleared and failed questions.

### Question bank (`question_bank.py`, `data/question_bank.csv`)
179 questions with an estimated difficulty in [0, 1] (21 levels, steps of 0.05). The recommender only needs `question_index`, `item_id`, `topic` and `difficulty`. The question texts and answer options are not part of this repository. The CSV matches the questions logged in the released study data (all 3,985 logged answers).

### Objectives (`objectives.py`)
Each candidate batch is scored on three objectives, all to be **maximised**:

| Objective | Meaning |
|---|---|
| `aptitude` | mean of `difficulty - mastery`: progression potential |
| `expected_performance` | minus the mean distance to the difficulties of questions already **cleared**: stay close to what the learner can do |
| `negative_gap` | minus the mean distance to the difficulties of questions **failed**: work on known weaknesses |

A batch *dominates* another when it is not worse on any objective and better on at least one (Pareto dominance). An objective that is undefined (no past success or failure yet) neither helps nor hurts.

### Hill climbing (`hill_climbing.py`)
1. draw `restarts = 20` random batches of unseen questions;
2. improve each by moving one question at a time to the next lower or higher difficulty level, accepting a move that is Pareto-optimal among the neighbours and not dominated by the current batch, until the current batch dominates all of them (at most 500 moves);
3. keep the Pareto front of the improved batches and return one of them.

| Variant | Pareto objectives | Final choice |
|---|---|---|
| `MOO` (the paper's HC) | aptitude, expected_performance | best `negative_gap` |
| `MOEG` | expected_performance, negative_gap | random |
| `MOAG` | aptitude, negative_gap | random |
| `MOAE` | aptitude, expected_performance | random |

### Bandit (`bandit.py`)
The four variants are the arms. A policy picks an arm before each batch and is rewarded with the **mastery progression** of the batch (`mastery_after - mastery_before`; for Thompson sampling, 1 if mastery increased and 0 otherwise). The study used `thompson`; `egreedy` (ε = 0.1), `softmax` (temperature 0.05), `ucb` and `random` are available.

### Session (`session.py`)
`LearningSession` runs the loop of the study: pretest of 5 questions spread over the difficulty range, then batches of 3 until 15 batches are done or mastery reaches 0.9. It never decides how answers are obtained: you pass whether each answer was correct, so it serves a web application as well as a simulated learner (`simulation.py`, a logistic learner for demos; **not** the study's response model).

```python
from adaptive_recommender import QuestionBank, LearningSession

bank = QuestionBank.from_csv("data/question_bank.csv")
session = LearningSession(bank, method="mab", policy="thompson", seed=0)

pretest = session.pretest_questions()
session.submit_pretest(pretest, [answer_is_correct(q) for q in pretest])    # your own scoring
while not session.is_done:
    batch = session.next_batch()                                             # 3 questions to show
    record = session.submit_batch([answer_is_correct(q) for q in batch])    # mastery, AUFS, AEFS after the batch
```

### Failure scores (`failure_scores.py`)
UFS, EFS, AUFS, AEFS and TOFT exactly as in Sec. 3 of the paper (`delta = 0.20`, `lambda = 0.20`, `tau = 0.50`). The platform logged AUFS and used it to trigger an extra survey when it reached 0.5; AEFS uses the same accumulation. `LearningSession` updates them answer by answer, with the mastery *before* the answer.

## Validation against the study

1. **Same seeds, same recommendations.** The research implementation and this package were run side by side with identical random seeds, all four hill-climbing variants and all five bandit policies, with a simulated learner: 108 of 108 simulated sessions (12 seeds × 4 hill-climbing variants and 5 bandit policies, 1,515 batches) were identical, batch by batch. This comparison needs the original research code, so it is not part of the test suite.
2. **Replay of the study logs.** `tests/test_replay_study_logs.py` feeds every answer logged during the study (128 sessions, 3,985 answers) through the learner model and failure scores and compares the result with the mastery and AUFS the platform logged after each answer: 126 sessions match exactly; the two others are the known anomalies below.
3. **Paper example.** `tests/test_components.py` reproduces the running example of Figure 1.

### Known anomalies in the study logs
Two logged sessions cannot be replayed exactly. In session 64 of `study_120_users.sqlite` the logged state returns to its initial values (mastery 0.2, AUFS 0) at the start of the second batch, while the answers before that had raised both; this looks like the platform's engine state being re-initialised mid-session. In session 51 the mastery logged before batch 10 is 0.05 above the mastery logged after batch 9. Both are kept in the released data; the paper's analysis of mastery trajectories uses the logged values.

## Differences with the research code
* One clean API (dataclasses, explicit random generators) instead of the research code base's dictionaries and global random state; a session is reproducible from one seed.
* The IRT / BKT response models, the simulation harness and the web application are not included: the study used real participants.
* Same behaviour otherwise: for identical seeds the recommended batches are identical (see above).
