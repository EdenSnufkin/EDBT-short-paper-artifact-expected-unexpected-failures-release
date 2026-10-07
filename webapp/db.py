# webapp/db.py
import os
import json
import datetime as dt
from typing import Any

from sqlalchemy import (
    create_engine, event, text,
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DB_PATH = os.environ.get("STUDY_DB_PATH", os.path.join(os.path.dirname(__file__), "study.sqlite"))
DB_URL = os.environ.get("STUDY_DB_URL", f"sqlite:///{DB_PATH}")

Base = declarative_base()
engine = create_engine(DB_URL, future=True)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)

    username = Column(String(64), nullable=False, unique=True)  # participant identifier (Prolific PID in the study)

    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    sessions = relationship("StudySession", back_populates="user")


class StudySession(Base):
    __tablename__ = "study_sessions"
    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Settings snapshot (so results stay interpretable even if settings.py changes later)
    dataset = Column(String(32), nullable=False)
    recommendation_method = Column(String(32), nullable=False)  # "hillclimbing" | "mab"
    hillclimbing_algo = Column(String(16), nullable=True)
    mab_algo = Column(String(16), nullable=True)
    batch_size = Column(Integer, nullable=False)
    n_batches = Column(Integer, nullable=False)
    mastery_stop_threshold = Column(Float, nullable=False)
    pretest_n = Column(Integer, nullable=False)

    seed = Column(Integer, nullable=True)  # seed of the recommender random generators (reproduces the session)
    delta = Column(Float, nullable=False)  # tolerance of the failure scores
    lambda_decay = Column(Float, nullable=False)  # decay of the failure scores

    survey_mode = Column(String(32), nullable=False)
    survey_mode_param = Column(Float, nullable=True)

    status = Column(String(32), nullable=False, default="created")
    # created|consent|pretest|pretest_done|learning|survey|completed|abandoned
    started_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    pretest_correct = Column(Integer, nullable=True)
    pretest_mastery_init = Column(Float, nullable=True)

    final_mastery = Column(Float, nullable=True)
    final_aufs = Column(Float, nullable=True)
    final_aefs = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)

    abandoned_phase = Column(String(32), nullable=True)
    abandoned_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="sessions")
    attempts = relationship("Attempt", back_populates="session")
    batch_steps = relationship("BatchStep", back_populates="session")
    survey_responses = relationship("SurveyResponse", back_populates="session")


class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False)

    phase = Column(String(16), nullable=False)  # "pretest" | "learning"
    batch_index = Column(Integer, nullable=True)  # null for pretest
    slot_index = Column(Integer, nullable=False)  # 0-based position within the pretest/batch;
    # NOT the same as question_index: slot_index (not question identity) is what is unique.

    question_index = Column(Integer, nullable=False)  # index in the question bank
    item_id = Column(Integer, nullable=False)  # source dataset "id"
    topic = Column(String(128), nullable=False)
    difficulty = Column(Float, nullable=False)

    chosen_answer = Column(String(512), nullable=True)
    correct_answer = Column(String(512), nullable=True)
    is_correct = Column(Boolean, nullable=False)

    mastery_after = Column(Float, nullable=True)
    aufs_after = Column(Float, nullable=True)   # accumulated unexpected-failure score after this answer
    aefs_after = Column(Float, nullable=True)   # accumulated expected-failure score after this answer
    ufs = Column(Float, nullable=True)          # unexpected-failure score of this answer alone
    efs = Column(Float, nullable=True)          # expected-failure score of this answer alone

    started_at = Column(DateTime(timezone=True), nullable=False)
    answered_at = Column(DateTime(timezone=True), nullable=False)
    response_time_ms = Column(Integer, nullable=False)

    session = relationship("StudySession", back_populates="attempts")
    __table_args__ = (
        UniqueConstraint("session_id", "phase", "batch_index", "slot_index",
                          name="uq_attempt_per_session_phase_batch_slot"),
    )


class BatchStep(Base):
    __tablename__ = "batch_steps"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False)
    batch_index = Column(Integer, nullable=False)

    algo_used = Column(String(32), nullable=False)  # HillClimbing variant, or MAB arm name
    mab_arm_index = Column(Integer, nullable=True)
    mab_reward = Column(Float, nullable=True)

    mastery_before = Column(Float, nullable=True)
    mastery_after = Column(Float, nullable=True)
    aufs_before = Column(Float, nullable=True)
    aufs_after = Column(Float, nullable=True)
    aefs_before = Column(Float, nullable=True)
    aefs_after = Column(Float, nullable=True)

    question_ids_json = Column(Text, nullable=False)

    started_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    session = relationship("StudySession", back_populates="batch_steps")
    __table_args__ = (UniqueConstraint("session_id", "batch_index", name="uq_batch_per_session"),)


class SurveyResponse(Base):
    __tablename__ = "survey_responses"
    id = Column(Integer, primary_key=True)

    session_id = Column(Integer, ForeignKey("study_sessions.id"), nullable=False)
    administration_index = Column(Integer, nullable=False)  # 0-based, per session
    trigger_reason = Column(String(32), nullable=False)  # end_only|fixed_count|every_failure|threshold
    batch_index_at_trigger = Column(Integer, nullable=True)

    # NASA-TLX-style subscales, 1..7 (Low -> High)
    temporal_demand = Column(Integer, nullable=False)
    mental_demand = Column(Integer, nullable=False)
    frustration = Column(Integer, nullable=False)
    perceived_performance = Column(Integer, nullable=False)
    effort = Column(Integer, nullable=False)

    opinion_changed = Column(Boolean, nullable=True)  # null for the first administration
    free_text = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    session = relationship("StudySession", back_populates="survey_responses")
    __table_args__ = (
        UniqueConstraint("session_id", "administration_index", name="uq_survey_per_session_admin"),
    )


def _ensure_column(engine, table: str, col: str, coltype_sql: str):
    with engine.begin() as conn:
        cols = [r[1] for r in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()]
        if col not in cols:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {coltype_sql}"))


def _ensure_index(engine, index_name: str, create_sql: str):
    with engine.begin() as conn:
        conn.execute(text(create_sql))


def init_db():
    Base.metadata.create_all(engine)

    _ensure_index(engine, "ix_study_sessions_user_id",
                  "CREATE INDEX IF NOT EXISTS ix_study_sessions_user_id ON study_sessions(user_id)")
    _ensure_index(engine, "ix_study_sessions_status",
                  "CREATE INDEX IF NOT EXISTS ix_study_sessions_status ON study_sessions(status)")
    _ensure_index(engine, "ix_attempts_session_phase",
                  "CREATE INDEX IF NOT EXISTS ix_attempts_session_phase ON attempts(session_id, phase)")
    _ensure_index(engine, "ix_batch_steps_session",
                  "CREATE INDEX IF NOT EXISTS ix_batch_steps_session ON batch_steps(session_id)")
    _ensure_index(engine, "ix_survey_responses_session",
                  "CREATE INDEX IF NOT EXISTS ix_survey_responses_session ON survey_responses(session_id)")


# -----------------------
# JSON helpers
# -----------------------

def _to_jsonable(obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()

    if isinstance(obj, (dt.datetime, dt.date)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_jsonable(v) for v in sorted(obj)]

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    return str(obj)


def to_json(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False)
