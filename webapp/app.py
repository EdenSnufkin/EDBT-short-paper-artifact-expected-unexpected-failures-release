# webapp/app.py
"""Streamlit application of the user study: consent, pretest, adaptive question batches, NASA-TLX-style surveys.

    cd webapp && STUDY_ALLOW_DEMO=1 streamlit run app.py

The page flow is a state machine stored in ``st.session_state.phase``; every database write is guarded against the
double execution that Streamlit reruns can cause.
"""
import datetime as dt
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import streamlit as st
from sqlalchemy.exc import IntegrityError

import questions as question_files
import settings
import ui_components as ui
from assignment import pick_balanced
from db import Attempt, BatchStep, SessionLocal, StudySession, SurveyResponse, User, init_db, now_utc, to_json
from engine import StudyEngine
from questions import is_correct_answer
from survey import TLX_ITEMS, SurveyTriggerState, needs_fallback_final_survey, should_trigger_after_batch

st.set_page_config(page_title="Adaptive Question Study", layout="centered")


@st.cache_resource(show_spinner=False)
def batch_prefetch_executor() -> ThreadPoolExecutor:
    """Shared worker pool; each submitted job only touches its own participant's engine."""
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="batch-prefetch")


@st.cache_resource(show_spinner=False)
def load_question_bank():
    """The question bank is read once per server process and shared (read-only) by all participants."""
    return question_files.load_questions(settings.QUESTIONS_PATH, settings.TOPIC)


# -------------------------
# Session state
# -------------------------

PER_RUN_DEFAULTS = {
    "pretest_questions": None,
    "pretest_t0": {},

    "learning_step": 0,
    "learning_current_batch": None,
    "learning_current_algo": None,
    "learning_current_arm": None,
    "learning_t0": {},
    "learning_mastery_before": None,
    "learning_aufs_before": None,
    "learning_aefs_before": None,
    "learning_prefetch_future": None,
    "learning_prefetch_mastery_before": None,
    "learning_prefetch_aufs_before": None,
    "learning_prefetch_aefs_before": None,

    "learning_review_batch": None,
    "learning_review_answers": None,
    "learning_review_batch_index": None,
    "after_review_phase": None,

    "survey_previous_answers": None,
    "survey_trigger_reason": None,
    "survey_batch_index_at_trigger": None,
    "after_survey_phase": None,

    "engine": None,
}


def fresh_run_state() -> dict:
    return {**{k: (dict(v) if isinstance(v, dict) else v) for k, v in PER_RUN_DEFAULTS.items()},
            "survey_trigger_state": SurveyTriggerState()}


def require_state_defaults():
    defaults = {
        "phase": "landing",
        "user_id": None,
        "username": None,
        "study_session_id": None,
        "completion_code": None,
        "session_terminated": False,
        "show_exit_confirm": False,
        **fresh_run_state(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_run_state():
    """Reset per-run state when a brand new StudySession starts."""
    future = st.session_state.get("learning_prefetch_future")
    if future is not None:
        future.cancel()
    for key, value in fresh_run_state().items():
        st.session_state[key] = value


def current_page_key() -> str:
    """A stable key for each user-visible page, including repeated phases (used to scroll to the top on a new page)."""
    if st.session_state.get("session_terminated", False):
        return "session_ended"
    if st.session_state.get("show_exit_confirm", False):
        return "exit_confirmation"
    phase = st.session_state.phase
    if phase == "learning":
        return f"learning:{st.session_state.learning_step}"
    if phase == "learning_review":
        return f"learning_review:{st.session_state.learning_review_batch_index}"
    if phase == "survey":
        return f"survey:{st.session_state.survey_trigger_state.administrations_done}"
    return phase


def start_learning_batch_prefetch(engine: StudyEngine) -> None:
    """Start recommending the next batch in a worker thread while the participant reads the previous feedback."""
    if st.session_state.learning_prefetch_future is not None:
        return
    st.session_state.learning_prefetch_mastery_before = engine.mastery
    st.session_state.learning_prefetch_aufs_before = engine.aufs
    st.session_state.learning_prefetch_aefs_before = engine.aefs
    st.session_state.learning_prefetch_future = batch_prefetch_executor().submit(engine.recommend_batch)


def prepare_learning_batch(engine: StudyEngine) -> None:
    """Populate the current batch from the prefetched result, or recommend synchronously."""
    if st.session_state.learning_current_batch is not None:
        return
    future: Future | None = st.session_state.learning_prefetch_future
    if future is None:
        batch, algo_used, arm_index = engine.recommend_batch()
        mastery_before, aufs_before, aefs_before = engine.mastery, engine.aufs, engine.aefs
    else:
        batch, algo_used, arm_index = future.result()
        mastery_before = st.session_state.learning_prefetch_mastery_before
        aufs_before = st.session_state.learning_prefetch_aufs_before
        aefs_before = st.session_state.learning_prefetch_aefs_before

    st.session_state.learning_current_batch = batch
    st.session_state.learning_current_algo = algo_used
    st.session_state.learning_current_arm = arm_index
    st.session_state.learning_mastery_before = mastery_before
    st.session_state.learning_aufs_before = aufs_before
    st.session_state.learning_aefs_before = aefs_before
    st.session_state.learning_t0 = {}
    st.session_state.learning_prefetch_future = None
    st.session_state.learning_prefetch_mastery_before = None
    st.session_state.learning_prefetch_aufs_before = None
    st.session_state.learning_prefetch_aefs_before = None


# -------------------------
# Database helpers
# -------------------------

def get_or_create_user(db, participant_id: str) -> User:
    user = db.query(User).filter(User.username == participant_id).one_or_none()
    if user is None:
        user = User(username=participant_id)
        db.add(user)
        db.commit()
    return user


def get_active_session(db, user_id: int):
    return (
        db.query(StudySession)
        .filter(StudySession.user_id == user_id, StudySession.status.notin_(["completed", "abandoned"]))
        .order_by(StudySession.started_at.desc())
        .first()
    )


def survey_mode_param(mode: str) -> float | None:
    if mode in ("fixed_count", "fixed_count_or_threshold"):
        # For the mixed mode the fixed-count schedule is the primary one and is what is recorded here;
        # the threshold is a constant of settings.py.
        return float(settings.SURVEY_FIXED_COUNT)
    if mode == "threshold":
        return float(settings.SURVEY_FAILURE_THRESHOLD)
    return None


def choose_recommendation_method(db) -> str:
    if not settings.RANDOMIZE_RECOMMENDATION_METHOD:
        return settings.RECOMMENDATION_METHOD
    return pick_balanced(db, StudySession.recommendation_method, settings.RECOMMENDATION_METHOD_CHOICES)


def choose_survey_mode(db) -> str:
    if not settings.RANDOMIZE_SURVEY_MODE:
        return settings.SURVEY_MODE
    return pick_balanced(db, StudySession.survey_mode, settings.SURVEY_MODE_CHOICES)


def start_new_study_session(db, user_id: int) -> StudySession:
    recommendation_method = choose_recommendation_method(db)
    survey_mode = choose_survey_mode(db)
    sess = StudySession(
        user_id=user_id,
        dataset=settings.QUESTIONS_PATH.name,
        recommendation_method=recommendation_method,
        hillclimbing_algo=settings.HILLCLIMBING_ALGO if recommendation_method == "hillclimbing" else None,
        mab_algo=settings.MAB_ALGO if recommendation_method == "mab" else None,
        batch_size=settings.BATCH_SIZE,
        n_batches=settings.N_BATCHES,
        mastery_stop_threshold=settings.MASTERY_STOP_THRESHOLD,
        pretest_n=settings.PRETEST_N,
        seed=StudyEngine.new_seed(),
        delta=settings.DELTA,
        lambda_decay=settings.LAMBDA,
        survey_mode=survey_mode,
        survey_mode_param=survey_mode_param(survey_mode),
        status="created",
    )
    db.add(sess)
    db.commit()

    st.session_state.study_session_id = sess.id
    st.session_state.phase = "consent"
    reset_run_state()
    return sess


def mark_session_status(db, session_id: int, status: str, abandoned_phase: str | None = None):
    sess = db.query(StudySession).filter(StudySession.id == session_id).one()
    sess.status = status
    if status == "abandoned":
        sess.ended_at = sess.abandoned_at = now_utc()
        sess.abandoned_phase = abandoned_phase
    if status == "completed":
        sess.ended_at = now_utc()
    db.commit()


def load_engine(participant_id: str, sess: StudySession | None):
    # No StudySession row yet (consent page): nothing to build the engine from, and no such page needs it.
    if sess is None:
        return
    bank, content = load_question_bank()
    st.session_state.bank = bank
    st.session_state.content_by_item_id = content
    if st.session_state.engine is None:
        st.session_state.engine = StudyEngine(
            bank, participant_id, seed=sess.seed, recommendation_method=sess.recommendation_method,
            hillclimbing_algo=sess.hillclimbing_algo, mab_algo=sess.mab_algo,
        )


# -------------------------
# Logging helpers (the dedup guards protect against Streamlit rerun double writes)
# -------------------------

def record_attempt(db, sess, phase, batch_index, slot_index, question, content, chosen_answer, answer, t0):
    exists = db.query(Attempt).filter(
        Attempt.session_id == sess.id, Attempt.phase == phase,
        Attempt.batch_index == batch_index, Attempt.slot_index == slot_index,
    ).first()
    if exists is not None:
        return
    correct_answer = str(content.correct_answer_index) if content.options is not None else content.correct_answer_text
    db.add(Attempt(
        session_id=sess.id, phase=phase, batch_index=batch_index, slot_index=slot_index,
        question_index=question.index, item_id=question.item_id, topic=question.topic, difficulty=float(question.difficulty),
        chosen_answer=chosen_answer, correct_answer=correct_answer, is_correct=bool(answer.correct),
        mastery_after=float(answer.mastery_after), aufs_after=float(answer.aufs), aefs_after=float(answer.aefs),
        ufs=float(answer.ufs), efs=float(answer.efs),
        started_at=dt.datetime.fromtimestamp(t0, tz=dt.timezone.utc), answered_at=now_utc(),
        response_time_ms=ui.ms_since(t0),
    ))


def record_batch_step(db, sess, batch_index, algo_used, arm_index, reward, mastery_before, mastery_after,
                      aufs_before, aufs_after, aefs_before, aefs_after, question_ids):
    exists = db.query(BatchStep).filter(BatchStep.session_id == sess.id, BatchStep.batch_index == batch_index).first()
    if exists is not None:
        return
    db.add(BatchStep(
        session_id=sess.id, batch_index=batch_index, algo_used=algo_used, mab_arm_index=arm_index, mab_reward=reward,
        mastery_before=mastery_before, mastery_after=mastery_after, aufs_before=aufs_before, aufs_after=aufs_after,
        aefs_before=aefs_before, aefs_after=aefs_after, question_ids_json=to_json(question_ids), ended_at=now_utc(),
    ))


def commit_or_rollback(db):
    try:
        db.commit()
    except IntegrityError:
        db.rollback()


# -------------------------
# End-session controls
# -------------------------

def end_session_footer(db, sess):
    if st.session_state.get("show_exit_confirm", False):
        return
    left, right = st.columns([8, 2])
    left.markdown(":red[Reminder:] You may end the session at any time by clicking **End session**.")
    if right.button("End session", key="end_session"):
        st.session_state.show_exit_confirm = True
        st.rerun()


def render_exit_confirmation(db, sess):
    st.warning(
        "Are you sure you want to end the session? Your progress so far has been recorded, "
        "but you will not be able to resume."
    )
    col1, col2 = st.columns(2)
    if col1.button("Cancel"):
        st.session_state.show_exit_confirm = False
        st.rerun()
    if col2.button("Yes, end session"):
        st.session_state.show_exit_confirm = False
        mark_session_status(db, sess.id, "abandoned", abandoned_phase=st.session_state.phase)
        st.session_state.session_terminated = True
        st.rerun()


def participant_id_from_url() -> str | None:
    pid = st.query_params.get("PROLIFIC_PID")
    if pid:
        return pid
    if settings.ALLOW_ANONYMOUS_DEMO:
        if "demo_participant_id" not in st.session_state:
            st.session_state.demo_participant_id = f"demo-{uuid.uuid4().hex[:8]}"
        return st.session_state.demo_participant_id
    return None


# -------------------------
# Pages
# -------------------------

def page_consent(db, user, sess):
    consented = ui.consent_block()
    if consented and st.button("Continue to preliminary test"):
        if st.session_state.study_session_id is None:
            sess = start_new_study_session(db, user.id)
        mark_session_status(db, sess.id, "pretest")
        st.session_state.phase = "pretest_intro"
        st.rerun()
    elif not consented:
        st.info("Please provide consent to proceed.")
    if sess is not None:
        end_session_footer(db, sess)


def render_questions(batch, key_prefix, t0_by_slot, content_by_item_id):
    chosen_answers = []
    for i, q in enumerate(batch):
        content = content_by_item_id[q.item_id]
        st.markdown(f"**Q{i + 1}**")
        if i not in t0_by_slot:
            t0_by_slot[i] = time.time()
        chosen_answers.append(ui.render_question(content, key_prefix=f"{key_prefix}_{i}"))
        st.markdown("---")
    return chosen_answers


def page_pretest(db, sess, engine, content_by_item_id):
    st.subheader(f"Preliminary test ({settings.PRETEST_N} questions)")
    if st.session_state.pretest_questions is None:
        st.session_state.pretest_questions = engine.pretest_questions()
    batch = st.session_state.pretest_questions
    chosen_answers = render_questions(batch, "pre", st.session_state.pretest_t0, content_by_item_id)

    if st.button("Submit preliminary test"):
        if not all(a is not None for a in chosen_answers):
            st.error("Please answer every question before submitting.")
            return
        correct = [is_correct_answer(content_by_item_id[q.item_id], a) for q, a in zip(batch, chosen_answers)]
        answers = engine.submit_pretest(batch, correct)
        for i, (q, a, chosen) in enumerate(zip(batch, answers, chosen_answers)):
            record_attempt(db, sess, "pretest", None, i, q, content_by_item_id[q.item_id], chosen, a,
                           st.session_state.pretest_t0[i])
        sess.pretest_correct = sum(correct)
        sess.pretest_mastery_init = engine.mastery
        commit_or_rollback(db)
        mark_session_status(db, sess.id, "pretest_done")
        start_learning_batch_prefetch(engine)
        st.session_state.phase = "pretest_result"
        st.rerun()
    end_session_footer(db, sess)


def page_learning(db, sess, engine, content_by_item_id):
    batch_index = st.session_state.learning_step
    st.subheader(f"Question batch {batch_index + 1}/{settings.N_BATCHES}")

    if st.session_state.learning_current_batch is None:
        future = st.session_state.learning_prefetch_future
        if future is not None and not future.done():
            with st.spinner("Preparing your next question batch..."):
                prepare_learning_batch(engine)
        else:
            prepare_learning_batch(engine)

    batch = st.session_state.learning_current_batch
    chosen_answers = render_questions(batch, f"learn_{batch_index}", st.session_state.learning_t0, content_by_item_id)

    if st.button("Submit batch"):
        if not all(a is not None for a in chosen_answers):
            st.error("Please answer every question in this batch.")
            return

        correct = [is_correct_answer(content_by_item_id[q.item_id], a) for q, a in zip(batch, chosen_answers)]
        outcome = engine.submit_batch(correct)
        for i, (q, a, chosen) in enumerate(zip(batch, outcome.answers, chosen_answers)):
            record_attempt(db, sess, "learning", batch_index, i, q, content_by_item_id[q.item_id], chosen, a,
                           st.session_state.learning_t0[i])
        record_batch_step(
            db, sess, batch_index, st.session_state.learning_current_algo, st.session_state.learning_current_arm,
            outcome.reward, st.session_state.learning_mastery_before, engine.mastery,
            st.session_state.learning_aufs_before, engine.aufs, st.session_state.learning_aefs_before, engine.aefs,
            [q.item_id for q in batch],
        )
        commit_or_rollback(db)

        trigger_state: SurveyTriggerState = st.session_state.survey_trigger_state
        done = engine.is_learning_done()
        any_incorrect = not all(correct)

        triggered = False
        if sess.survey_mode != "end_only":
            triggered = should_trigger_after_batch(
                trigger_state, batch_index, settings.N_BATCHES, any_incorrect, engine.aufs, mode=sess.survey_mode)
        if done and sess.survey_mode == "end_only":
            triggered = True
        if done and not triggered and needs_fallback_final_survey(trigger_state, mode=sess.survey_mode):
            triggered = True

        st.session_state.learning_step = batch_index + 1
        st.session_state.learning_review_batch = batch
        st.session_state.learning_review_answers = list(chosen_answers)
        st.session_state.learning_review_batch_index = batch_index
        st.session_state.learning_current_batch = None
        st.session_state.learning_current_algo = None
        st.session_state.learning_current_arm = None
        st.session_state.learning_t0 = {}

        if triggered:
            st.session_state.survey_trigger_reason = sess.survey_mode
            st.session_state.survey_batch_index_at_trigger = batch_index
            st.session_state.after_survey_phase = "learning_done" if done else "learning"
            st.session_state.after_review_phase = "survey"
        else:
            st.session_state.after_review_phase = "learning_done" if done else "learning"

        if not done:
            start_learning_batch_prefetch(engine)
        st.session_state.phase = "learning_review"
        st.rerun()
    end_session_footer(db, sess)


def page_review(db, sess, content_by_item_id):
    batch = st.session_state.learning_review_batch
    chosen_answers = st.session_state.learning_review_answers
    batch_index = st.session_state.learning_review_batch_index

    if batch is None or chosen_answers is None or batch_index is None:
        st.session_state.phase = st.session_state.after_review_phase or "learning"
        st.rerun()

    st.subheader(f"Review batch {batch_index + 1}/{settings.N_BATCHES}")
    st.caption("Correct answers are shown in green. If your answer was incorrect, it is shown in red.")
    for i, (question, chosen) in enumerate(zip(batch, chosen_answers)):
        st.markdown(f"**Q{i + 1}**")
        ui.render_answer_review(content_by_item_id[question.item_id], chosen, key_prefix=f"review_{batch_index}_{i}")
        st.markdown("---")

    if st.button("Continue"):
        next_phase = st.session_state.after_review_phase or "learning"
        st.session_state.learning_review_batch = None
        st.session_state.learning_review_answers = None
        st.session_state.learning_review_batch_index = None
        st.session_state.after_review_phase = None
        if next_phase == "survey":
            mark_session_status(db, sess.id, "survey")
        st.session_state.phase = next_phase
        st.rerun()
    end_session_footer(db, sess)


def page_survey(db, sess):
    trigger_state: SurveyTriggerState = st.session_state.survey_trigger_state
    admin_index = trigger_state.administrations_done
    previous = st.session_state.survey_previous_answers

    ui.survey_intro()
    answers = {}
    for field_name, _label, prompt in TLX_ITEMS:
        prev_value = previous[field_name] if previous is not None else None
        answers[field_name] = ui.likert7(prompt, key=f"survey_{admin_index}_{field_name}", previous=prev_value)
    free_text = st.text_area("Any comments? (optional)", key=f"survey_{admin_index}_text")

    if st.button("Submit", disabled=not all(v is not None for v in answers.values())):
        opinion_changed = None if previous is None else any(answers[f] != previous[f] for f in answers)
        exists = db.query(SurveyResponse).filter(
            SurveyResponse.session_id == sess.id, SurveyResponse.administration_index == admin_index).first()
        if exists is None:
            db.add(SurveyResponse(
                session_id=sess.id, administration_index=admin_index,
                trigger_reason=st.session_state.survey_trigger_reason or sess.survey_mode,
                batch_index_at_trigger=st.session_state.survey_batch_index_at_trigger,
                opinion_changed=opinion_changed, free_text=free_text or None,
                **{name: int(value) for name, value in answers.items()},
            ))
            commit_or_rollback(db)
        trigger_state.administrations_done += 1
        st.session_state.survey_previous_answers = answers
        st.session_state.phase = st.session_state.after_survey_phase or "learning_done"
        st.session_state.after_survey_phase = None
        st.rerun()
    end_session_footer(db, sess)


def page_learning_done(db, sess, engine):
    if ui.learning_done_block():
        attempts = db.query(Attempt).filter(Attempt.session_id == sess.id, Attempt.phase == "learning").all()
        sess.final_mastery = engine.mastery
        sess.final_aufs, sess.final_aefs = engine.aufs, engine.aefs
        sess.final_accuracy = sum(a.is_correct for a in attempts) / len(attempts) if attempts else None
        mark_session_status(db, sess.id, "completed")
        st.session_state.completion_code = settings.PROLIFIC_COMPLETION_CODE
        st.session_state.phase = "completed"
        st.rerun()


# -------------------------
# Main
# -------------------------

STATUS_TO_PHASE = {
    "created": "consent", "pretest": "pretest", "pretest_done": "pretest_result",
    "learning": "learning", "survey": "survey", "completed": "completed",
}


def show_final_page(finished_status: str):
    if finished_status == "completed":
        ui.header("completed")
        ui.study_complete_block(settings.PROLIFIC_COMPLETION_CODE, settings.completion_url(settings.PROLIFIC_COMPLETION_CODE))
    else:
        ui.header("session_ended")
        ui.study_ended_block(settings.PROLIFIC_NO_BONUS_CODE, settings.completion_url(settings.PROLIFIC_NO_BONUS_CODE))


def main():
    participant_id = participant_id_from_url()
    if not participant_id:
        ui.prolific_entry_error_page()
        return

    init_db()
    require_state_defaults()

    if st.session_state.get("session_terminated", False):
        show_final_page("abandoned")
        return

    db = SessionLocal()
    try:
        user = get_or_create_user(db, participant_id)
        st.session_state.user_id, st.session_state.username = user.id, participant_id

        active = get_active_session(db, user.id)
        finished_before = (
            db.query(StudySession)
            .filter(StudySession.user_id == user.id, StudySession.status.in_(["abandoned", "completed"]))
            .order_by(StudySession.started_at.desc()).first()
        )
        if active is None and finished_before is not None:       # one session per participant
            show_final_page(finished_before.status)
            return

        sess = active
        if sess is not None:
            st.session_state.study_session_id = sess.id
            if st.session_state.phase == "landing":              # page reload: resume where the participant was
                st.session_state.phase = STATUS_TO_PHASE.get(sess.status, "consent")

        ui.header(current_page_key())

        if st.session_state.get("show_exit_confirm", False):
            if sess is None:
                st.error("Session not found. Please refresh the page.")
                return
            render_exit_confirmation(db, sess)
            return

        if st.session_state.phase == "landing":
            if ui.landing_overview():
                st.session_state.phase = "consent"
                st.rerun()
            return

        load_engine(participant_id, sess)
        engine: StudyEngine = st.session_state.engine
        content_by_item_id = st.session_state.get("content_by_item_id")

        if sess is None and st.session_state.study_session_id is not None:
            sess = db.query(StudySession).filter(StudySession.id == st.session_state.study_session_id).one_or_none()
            if sess is None:
                st.session_state.study_session_id = None

        phase = st.session_state.phase
        if phase not in ("landing", "consent") and sess is None:
            st.session_state.phase = "landing"
            st.rerun()

        if phase == "consent":
            page_consent(db, user, sess)
        elif phase == "pretest_intro":
            if ui.pretest_intro(settings.PRETEST_N):
                st.session_state.phase = "pretest"
                st.rerun()
            end_session_footer(db, sess)
        elif phase == "pretest":
            page_pretest(db, sess, engine, content_by_item_id)
        elif phase == "pretest_result":
            if ui.pretest_results_block(sess.pretest_correct or 0, settings.PRETEST_N):
                st.session_state.phase = "learning_intro"
                st.rerun()
            end_session_footer(db, sess)
        elif phase == "learning_intro":
            if ui.learning_intro(settings.N_BATCHES, settings.BATCH_SIZE):
                mark_session_status(db, sess.id, "learning")
                st.session_state.phase = "learning"
                st.rerun()
            end_session_footer(db, sess)
        elif phase == "learning":
            page_learning(db, sess, engine, content_by_item_id)
        elif phase == "learning_review":
            page_review(db, sess, content_by_item_id)
        elif phase == "survey":
            page_survey(db, sess)
        elif phase == "learning_done":
            page_learning_done(db, sess, engine)
        elif phase == "completed":
            ui.study_complete_block(settings.PROLIFIC_COMPLETION_CODE, settings.completion_url(settings.PROLIFIC_COMPLETION_CODE))
    finally:
        db.close()


if __name__ == "__main__":
    main()
