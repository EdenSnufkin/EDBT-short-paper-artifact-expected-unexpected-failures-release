"""Replay the answers logged during the user study through the clean learner model and failure scores.

The released study data contain, for every answer, the mastery estimate and the accumulated unexpected-failure
score (AUFS) that the platform computed at that moment. Feeding the same answers to this package must give the
same values: it checks the learner model (consecutive-correct-answers rule) and AUFS (Eqs. 1 and 3 of the paper)
against what actually happened during the study.
"""
import sqlite3

import pytest

from adaptive_recommender import LearningSession
from conftest import REPO

DATABASES = ["study_45_users.sqlite", "study_120_users.sqlite"]
TOLERANCE = 1e-9


def logged_sessions():
    for name in DATABASES:
        con = sqlite3.connect(REPO / "data" / name)
        sessions = [r[0] for r in con.execute("SELECT id FROM study_sessions ORDER BY id")]
        for sid in sessions:
            rows = con.execute(
                "SELECT phase, batch_index, slot_index, question_index, is_correct, mastery_after, unexpected_failure_after "
                "FROM attempts WHERE session_id = ? ORDER BY CASE phase WHEN 'pretest' THEN 0 ELSE 1 END, batch_index, slot_index", (sid,)).fetchall()
            if rows:
                yield name, sid, rows
        con.close()


def replay(bank, rows):
    """Return the largest deviation of mastery and AUFS from the logged values."""
    session = LearningSession(bank, topic="Maths")
    worst_mastery = worst_aufs = 0.0
    for phase, batch_index, _slot, question_index, is_correct, mastery_after, aufs_after in rows:
        record = session.record_answer(bank.questions[question_index], bool(is_correct), phase, batch_index)
        worst_mastery = max(worst_mastery, abs(record.mastery_after - mastery_after))
        worst_aufs = max(worst_aufs, abs(record.aufs - aufs_after))
    return worst_mastery, worst_aufs


def test_replay_reproduces_logged_mastery_and_aufs(bank):
    deviating = []
    n_sessions = 0
    for name, sid, rows in logged_sessions():
        n_sessions += 1
        d_mastery, d_aufs = replay(bank, rows)
        if max(d_mastery, d_aufs) > TOLERANCE:
            deviating.append((name, sid, round(d_mastery, 3), round(d_aufs, 3)))
    assert n_sessions > 100
    # Two logged sessions cannot be replayed (see README, "Known anomalies in the study logs"): in study_120 session 51 the
    # logged mastery before batch 10 differs from the mastery after batch 9, and in session 64 the platform state jumps back
    # to its initial values after the first batch (the engine state was re-initialised). All other sessions match exactly.
    known = {("study_120_users.sqlite", 51), ("study_120_users.sqlite", 64)}
    assert {(name, sid) for name, sid, *_ in deviating} <= known, f"unexpected deviations: {deviating}"
    assert len(deviating) <= len(known)
