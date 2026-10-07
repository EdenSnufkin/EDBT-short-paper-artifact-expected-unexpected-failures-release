"""Drive the Streamlit application headlessly (streamlit.testing) through a whole session and check what it logs."""
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

import settings
from db import DB_PATH
from questions import load_questions

APP = str(settings.HERE / "app.py")
_, CONTENT = load_questions(settings.QUESTIONS_PATH, settings.TOPIC)


def click(at, label):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r}; buttons: {[b.label for b in at.button]}"
    matches[0].click().run()


def answer_page(at, prefix, n, accuracy_rule):
    """Pick an option for each question shown (keys ``<prefix>_<i>_pick_<j>``); accuracy_rule(i, content) -> bool."""
    questions = at.session_state.pretest_questions if prefix == "pre" else at.session_state.learning_current_batch
    for i, q in enumerate(questions):
        content = CONTENT[q.item_id]
        wanted = content.correct_answer_index if accuracy_rule(i, content) else (content.correct_answer_index + 1) % len(content.options)
        at.button(key=f"{prefix}_{i}_pick_{wanted}").click().run()


def start(pid):
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["PROLIFIC_PID"] = pid
    at.run()
    assert not at.exception
    return at


def rows(sql, *args):
    con = sqlite3.connect(DB_PATH)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def test_entry_without_participant_id_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_ANONYMOUS_DEMO", False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert [e.value for e in at.error] == ["Invalid study entry"]


def test_a_participant_can_end_the_session_early():
    at = start("quitter")
    click(at, "Next")
    at.checkbox[0].check().run()
    click(at, "Continue to preliminary test")
    click(at, "End session")
    click(at, "Yes, end session")
    assert not at.exception and any("Session ended" in m.value for m in at.markdown)
    assert rows("SELECT status, abandoned_phase FROM study_sessions s JOIN users u ON u.id = s.user_id WHERE u.username = ?", "quitter") == [("abandoned", "pretest_intro")]
    again = start("quitter")                       # a second visit shows the final page, it does not start a new session
    assert any("Session ended" in m.value for m in again.markdown)


@pytest.mark.parametrize("always_fail", [False, True])
def test_full_session_is_logged(always_fail):
    pid = f"full-{always_fail}"
    at = start(pid)
    click(at, "Next")
    click_continue = [b for b in at.button if b.label == "Continue to preliminary test"]
    assert not click_continue                      # the consent box has to be ticked first
    at.checkbox[0].check().run()
    click(at, "Continue to preliminary test")
    click(at, "Start preliminary test")

    answer_page(at, "pre", settings.PRETEST_N, lambda i, c: not always_fail and i % 2 == 0)
    click(at, "Submit preliminary test")
    click(at, "Continue to question batches")
    click(at, "Start question batches")

    surveys = batches = 0
    for _ in range(settings.N_BATCHES * 3):
        phase = at.session_state.phase
        if phase == "learning":
            answer_page(at, f"learn_{at.session_state.learning_step}", settings.BATCH_SIZE, lambda i, c: not always_fail and i != 1)
            click(at, "Submit batch")
            batches += 1
        elif phase == "learning_review":
            click(at, "Continue")
        elif phase == "survey":
            for item in ("temporal_demand", "mental_demand", "frustration", "perceived_performance", "effort"):
                at.button(key=f"survey_{surveys}_{item}_btn_{3 + surveys % 3}").click().run()
            click(at, "Submit")
            surveys += 1
        elif phase == "learning_done":
            click(at, "Continue")
        elif phase == "completed":
            break
        assert not at.exception, at.exception
    assert at.session_state.phase == "completed"

    (sid, method, mode, status, final_mastery, final_aufs, final_aefs), = rows(
        "SELECT s.id, recommendation_method, survey_mode, status, final_mastery, final_aufs, final_aefs "
        "FROM study_sessions s JOIN users u ON u.id = s.user_id WHERE u.username = ?", pid)
    assert status == "completed" and method in settings.RECOMMENDATION_METHOD_CHOICES and mode in settings.SURVEY_MODE_CHOICES
    attempts = rows("SELECT phase, batch_index, slot_index, is_correct, mastery_after, aufs_after, aefs_after FROM attempts WHERE session_id = ? "
                    "ORDER BY id", sid)
    assert len([a for a in attempts if a[0] == "pretest"]) == settings.PRETEST_N
    assert len([a for a in attempts if a[0] == "learning"]) == batches * settings.BATCH_SIZE
    assert attempts[-1][4] == pytest.approx(final_mastery) and attempts[-1][5] == pytest.approx(final_aufs) and attempts[-1][6] == pytest.approx(final_aefs)
    steps = rows("SELECT batch_index, algo_used, mab_arm_index, mab_reward FROM batch_steps WHERE session_id = ? ORDER BY batch_index", sid)
    assert [s[0] for s in steps] == list(range(batches))
    assert all((s[2] is None) == (method == "hillclimbing") for s in steps)
    assert len(rows("SELECT id FROM survey_responses WHERE session_id = ?", sid)) == surveys >= 1
    if always_fail:
        assert batches == settings.N_BATCHES and final_aufs >= 0
