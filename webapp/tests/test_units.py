import random

import pytest

import settings
from assignment import pick_balanced
from db import SessionLocal, StudySession, User, init_db
from engine import StudyEngine
from latex_rendering import split_latex_segments
from questions import ContentQuestion, answer_display_text, is_correct_answer, load_questions
from survey import SurveyTriggerState, needs_fallback_final_survey, should_trigger_after_batch


def test_demo_bank_loads_with_difficulty_levels():
    bank, content = load_questions(settings.QUESTIONS_PATH, settings.TOPIC)
    assert len(bank) == len(content) == 84 and bank.topics == ["Maths"]
    assert sorted({q.difficulty for q in bank.questions}) == [round(0.05 * i, 2) for i in range(21)]


def test_study_style_option_strings_are_parsed(tmp_path):
    path = tmp_path / "q.json"
    path.write_text('[{"id": "7", "question": "q", "difficulty": "0.3", "options": "[\'a\', \'b\']", "correct_answer": "1"}]')
    bank, content = load_questions(path, "Maths")
    assert content[7].options == ["a", "b"] and content[7].correct_answer_index == 1 and bank.questions[0].difficulty == 0.3


def test_scoring_multiple_choice_and_free_text():
    mcq = ContentQuestion(1, "q", ["x", "y"], 1, None)
    assert is_correct_answer(mcq, "1") and not is_correct_answer(mcq, "0") and not is_correct_answer(mcq, None)
    assert answer_display_text(mcq, "1") == "B. y"
    free = ContentQuestion(2, "q", None, None, "3.0")
    assert is_correct_answer(free, "3") and is_correct_answer(free, " 3.0 ") and not is_correct_answer(free, "4")
    assert is_correct_answer(ContentQuestion(3, "q", None, None, "Yes"), "yes")


def test_survey_schedule_fixed_count():
    state = SurveyTriggerState()
    assert state.fixed_count_batch_targets(15) == [4, 9, 14]                       # surveys after batches 5, 10 and 15
    hits = [b for b in range(15) if should_trigger_after_batch(state, b, 15, False, 0.0, "fixed_count")]
    assert hits == [4, 9, 14]


def test_survey_threshold_mode_and_fallback():
    state = SurveyTriggerState()
    assert should_trigger_after_batch(state, 1, 15, True, 0.5, "fixed_count_or_threshold")      # threshold reached
    assert not should_trigger_after_batch(state, 1, 15, True, 0.49, "fixed_count_or_threshold")
    assert should_trigger_after_batch(state, 4, 15, False, 0.0, "fixed_count_or_threshold")     # scheduled
    assert not should_trigger_after_batch(state, 0, 15, True, 9.0, "end_only")
    assert should_trigger_after_batch(state, 0, 15, True, 0.0, "every_failure") and not should_trigger_after_batch(state, 0, 15, False, 0.0, "every_failure")
    assert needs_fallback_final_survey(state, "fixed_count") and not needs_fallback_final_survey(state, "end_only")
    state.administrations_done = 1
    assert not needs_fallback_final_survey(state, "fixed_count")
    with pytest.raises(ValueError):
        should_trigger_after_batch(state, 0, 15, False, 0.0, "nope")


def test_balanced_assignment_fills_the_least_used_condition(tmp_path):
    init_db()
    db = SessionLocal()
    try:
        user = User(username="assign-test")
        db.add(user); db.commit()
        choices = ["hillclimbing", "mab"]
        assigned = []
        for _ in range(10):
            method = pick_balanced(db, StudySession.recommendation_method, choices)
            assigned.append(method)
            db.add(StudySession(
                user_id=user.id, dataset="d", recommendation_method=method, batch_size=3, n_batches=15, mastery_stop_threshold=0.9,
                pretest_n=5, delta=0.2, lambda_decay=0.2, survey_mode="fixed_count", status="created"))
            db.commit()
        assert abs(assigned.count("mab") - assigned.count("hillclimbing")) <= 1
    finally:
        db.query(StudySession).delete(); db.query(User).delete(); db.commit(); db.close()


def test_latex_is_split_into_markdown_and_blocks():
    segments = split_latex_segments("Let $a \\in \\mathbb{R}$ and $$\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}$$")
    assert [s.kind for s in segments] == ["markdown", "latex"] and "$a \\in \\mathbb{R}$" in segments[0].content


@pytest.mark.parametrize("method", ["hillclimbing", "mab"])
def test_engine_runs_a_full_session(method):
    bank, _ = load_questions(settings.QUESTIONS_PATH, settings.TOPIC)
    engine = StudyEngine(bank, "p", seed=1, recommendation_method=method)
    rng = random.Random(0)
    pretest = engine.pretest_questions()
    assert len(pretest) == settings.PRETEST_N
    engine.submit_pretest(pretest, [q.difficulty < 0.5 for q in pretest])
    n = 0
    while not engine.is_learning_done():
        batch, algo, arm = engine.recommend_batch()
        assert len(batch) == settings.BATCH_SIZE and (arm is None) == (method == "hillclimbing")
        outcome = engine.submit_batch([rng.random() < 1 - q.difficulty * 0.7 for q in batch])
        assert len(outcome.answers) == 3 and (outcome.reward is None) == (method == "hillclimbing")
        n += 1
    assert n <= settings.N_BATCHES
