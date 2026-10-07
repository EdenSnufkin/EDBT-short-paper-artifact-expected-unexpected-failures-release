import math
import random
import warnings

import numpy as np
import pytest

from adaptive_recommender import (BanditRecommender, Candidate, FailureScores, HillClimbingRecommender, Learner, LearningSession,
                                  Question, QuestionBank, SimulatedLearner, aptitude, dominates, efs, expected_performance,
                                  negative_gap, non_dominated, run_session, toft, ufs)


# ---------------------------------------------------------------------- question bank
def test_bank_content(bank):
    assert len(bank) == 179 and bank.topics == ["Maths"]
    assert min(q.difficulty for q in bank.questions) == 0.0 and max(q.difficulty for q in bank.questions) == 1.0


def test_random_batch_is_distinct_and_respects_exclusions(bank):
    rng = random.Random(1)
    excluded = list(range(0, 150))
    batch = bank.random_batch(3, "Maths", exclude=excluded, rng=rng)
    assert len({q.index for q in batch}) == 3 and all(q.index not in excluded for q in batch)


def test_neighbour_moves_to_the_adjacent_difficulty_level(bank):
    rng = random.Random(0)
    q = next(q for q in bank.questions if q.difficulty == 0.5)
    lower, higher = bank.neighbour(q, -1, rng=rng), bank.neighbour(q, +1, rng=rng)
    assert lower.difficulty == 0.45 and higher.difficulty == 0.55


def test_neighbour_skips_excluded_levels_and_returns_the_question_at_the_boundary(bank):
    rng = random.Random(0)
    q = next(q for q in bank.questions if q.difficulty == 0.5)
    level_45 = {c.index for c in bank.questions if c.difficulty == 0.45}
    assert bank.neighbour(q, -1, exclude=level_45, rng=rng).difficulty == 0.4
    top = next(q for q in bank.questions if q.difficulty == 1.0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert bank.neighbour(top, +1, rng=rng) is top


def test_initial_questions_cover_the_difficulty_range(bank):
    questions = bank.initial_questions(5, "Maths", rng=random.Random(0))
    assert [q.difficulty for q in questions] == sorted(q.difficulty for q in questions)
    assert questions[0].difficulty <= 0.05 and questions[-1].difficulty >= 0.95 and len({q.index for q in questions}) == 5


# ---------------------------------------------------------------------- learner model
def question(index, difficulty):
    return Question(index, index + 1, "Maths", difficulty)


def test_mastery_rises_after_two_consecutive_successes_at_the_same_difficulty():
    learner = Learner.new("l", ["Maths"], 0.2)
    learner.update_mastery(question(0, 0.5), True)
    assert learner.mastery["Maths"] == 0.2
    learner.update_mastery(question(1, 0.5), True)
    assert learner.mastery["Maths"] == 0.5


def test_a_failure_erases_the_evidence_at_that_difficulty_and_above():
    learner = Learner.new("l", ["Maths"], 0.2)
    learner.update_mastery(question(0, 0.6), True)
    learner.update_mastery(question(1, 0.4), False)        # harder evidence (0.6) is erased
    learner.update_mastery(question(2, 0.6), True)
    assert learner.mastery["Maths"] == 0.2


def test_mastery_is_never_lowered():
    learner = Learner.new("l", ["Maths"], 0.2)
    for i in range(2):
        learner.update_mastery(question(i, 0.5), True)
    for i in range(2, 6):
        learner.update_mastery(question(i, 0.1), False)
    assert learner.mastery["Maths"] == 0.5


# ---------------------------------------------------------------------- failure scores (running example of the paper, Fig. 1)
def test_failure_scores_reproduce_the_running_example_of_the_paper():
    scores, aufs, aefs = FailureScores(delta=0.20, lam=0.20), [], []
    for difficulty, correct in [(0.45, False), (0.80, False), (0.95, False), (0.70, True)]:
        scores.update(0.70, difficulty, correct)
        aufs.append(round(scores.aufs, 2)); aefs.append(round(scores.aefs, 2))
    assert aufs == [0.45, 0.55, 0.55, 0.35] and aefs == [0.0, 0.10, 0.35, 0.15]
    assert toft(aufs, tau=0.50) == 0.5


def test_single_answer_scores():
    assert ufs(0.7, 0.45, False) == pytest.approx(0.45) and efs(0.7, 0.45, False) == 0
    assert ufs(0.7, 0.80, False) == pytest.approx(0.10) and efs(0.7, 0.80, False) == pytest.approx(0.10)    # within delta: both
    assert ufs(0.7, 0.95, False) == 0 and efs(0.7, 0.95, False) == pytest.approx(0.25)
    assert ufs(0.7, 0.45, True) == 0 and efs(0.7, 0.95, True) == 0


# ---------------------------------------------------------------------- objectives and dominance
def test_objectives_on_a_known_learner():
    learner = Learner.new("l", ["Maths"], 0.4)
    assert math.isnan(negative_gap([question(0, 0.5)], learner)) and math.isnan(expected_performance([question(0, 0.5)], learner))
    learner.record(question(10, 0.3), True)
    learner.record(question(11, 0.7), False)
    batch = [question(0, 0.5), question(1, 0.9)]
    assert aptitude(batch, learner) == pytest.approx(((0.5 - 0.4) + (0.9 - 0.4)) / 2)
    assert expected_performance(batch, learner) == pytest.approx(-((0.2 + 0.6) / 2))
    assert negative_gap(batch, learner) == pytest.approx(-((0.2 + 0.2) / 2))


def candidate(**scores):
    return Candidate([], scores)


def test_dominance_and_pareto_front():
    a, b, c = candidate(x=2, y=2), candidate(x=1, y=1), candidate(x=3, y=0)
    assert dominates(a, b, ["x", "y"]) and not dominates(b, a, ["x", "y"]) and not dominates(a, c, ["x", "y"])
    assert non_dominated([a, b, c], ["x", "y"]) == [a, c]
    assert not dominates(a, candidate(x=2, y=2), ["x", "y"])           # ties do not dominate


def test_an_undefined_objective_neither_helps_nor_hurts():
    nan = math.nan
    assert dominates(candidate(x=2, y=nan), candidate(x=1, y=5), ["x", "y"])
    assert not dominates(candidate(x=nan, y=nan), candidate(x=1, y=1), ["x", "y"])


# ---------------------------------------------------------------------- recommenders
@pytest.mark.parametrize("variant", ["MOO", "MOEG", "MOAG", "MOAE"])
def test_hill_climbing_returns_a_valid_batch(bank, variant):
    learner = Learner.new("l", bank.topics, 0.2)
    for q in bank.questions[:10]:
        learner.record(q, q.index % 2 == 0)
    rec = HillClimbingRecommender(bank, learner, batch_size=3, restarts=5, variant=variant, rng=random.Random(0))
    batch = rec.recommend("Maths")
    assert len(batch) == 3 and len({q.index for q in batch}) == 3
    assert all(q.index not in learner.cleared for q in batch)


def test_hill_climbing_is_reproducible_and_rejects_unknown_variants(bank):
    def run(seed):
        learner = Learner.new("l", bank.topics, 0.2)
        return [q.index for q in HillClimbingRecommender(bank, learner, restarts=5, rng=random.Random(seed)).recommend("Maths")]
    assert run(3) == run(3)
    with pytest.raises(ValueError):
        HillClimbingRecommender(bank, Learner.new("l", bank.topics), variant="nope")


@pytest.mark.parametrize("policy", ["thompson", "egreedy", "softmax", "ucb", "random"])
def test_bandit_selects_arms_and_records_progression(bank, policy):
    learner = Learner.new("l", bank.topics, 0.2)
    bandit = BanditRecommender(bank, learner, restarts=3, policy=policy, rng=random.Random(0), np_rng=np.random.RandomState(0))
    batch, arm = bandit.recommend("Maths")
    assert len(batch) == 3 and 0 <= arm < 4 and bandit.arms[arm].times_used == 1
    bandit.update(arm, 0.2, 0.5)
    assert bandit.arms[arm].total_progression == pytest.approx(0.3)


def test_thompson_sampling_learns_to_prefer_the_rewarded_arm():
    from adaptive_recommender.bandit import ThompsonPolicy
    policy = ThompsonPolicy(4, random.Random(0), np.random.RandomState(0))
    for _ in range(60):
        policy.update(2, 1.0)
        for other in (0, 1, 3):
            policy.update(other, 0.0)
    assert sum(policy.select() == 2 for _ in range(100)) > 90


# ---------------------------------------------------------------------- session
def test_a_full_simulated_session(bank):
    session = LearningSession(bank, method="hillclimbing", restarts=3, seed=0)
    run_session(session, SimulatedLearner(ability=0.7, seed=1))
    assert session.is_done and 1 <= len(session.batches) <= 15
    assert all(len(b.questions) == 3 for b in session.batches)
    assert session.mastery >= 0.9 or len(session.batches) == 15
    assert all(a.aufs >= 0 and a.aefs >= 0 for a in session.answers)


def test_session_protocol_errors(bank):
    session = LearningSession(bank, restarts=2, seed=0)
    with pytest.raises(RuntimeError):
        session.submit_batch([True, True, True])             # no pending batch
    session.next_batch()
    with pytest.raises(RuntimeError):
        session.next_batch()                                 # the pending batch must be answered first
    with pytest.raises(ValueError):
        session.submit_batch([True])                         # wrong number of answers
