# webapp/assignment.py
"""Balanced (minimization) random assignment of study conditions across participants.

Plain per-participant coin-flipping can leave sizable imbalances at the sample
sizes typical of a Prolific study. Instead, each new participant is assigned to
whichever choice currently has the fewest StudySession rows recorded for a
given column, so group sizes stay roughly equal as participants accrue.
"""
import random

from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from db import StudySession


def pick_balanced(db: DBSession, column, choices: list[str]) -> str:
    """Return the choice among `choices` with the fewest existing StudySession rows.

    Counts every StudySession ever created for that column value (any status),
    so a choice that has accrued many quick abandons isn't kept artificially
    "under-filled" and over-assigned relative to a choice whose participants
    tend to finish. Ties -- including the all-zero starting case -- are broken
    uniformly at random.
    """
    counts = {choice: 0 for choice in choices}
    rows = (
        db.query(column, func.count(StudySession.id))
        .filter(column.in_(choices))
        .group_by(column)
        .all()
    )
    for value, count in rows:
        counts[value] = count

    fewest = min(counts.values())
    least_filled = [choice for choice, count in counts.items() if count == fewest]
    return random.choice(least_filled)
