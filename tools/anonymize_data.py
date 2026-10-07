"""Create the anonymised study databases that are released with the paper.

The original study databases are private (they come from a Prolific study and are linked to participant
accounts). This script documents exactly what was done to produce the files in ``data/``. It needs the
private databases, so it cannot be re-run from this repository alone; it is included for transparency.

What is removed or changed
--------------------------
* ``users`` (and every table other than the four analysed ones) is dropped: it is the only table that links a
  session to a participant account.
* ``study_sessions.user_id`` is replaced by a random pseudonymous integer (one per participant, consistent
  across both databases); the mapping is never written to disk.
* ``survey_responses.free_text`` (optional comments) is set to NULL.
* Every timestamp is rewritten relative to the start of its own session (session start = 2000-01-01 00:00:00),
  so durations and the order of events are kept but calendar dates and times of day are not.
* Everything else (answers, difficulties, mastery and failure-score traces, NASA-TLX scores, response times,
  algorithm and survey-mode assignments) is unchanged.

Usage
-----
    python anonymize_data.py ORIGINAL_45.sqlite ORIGINAL_120.sqlite OUTPUT_DIR
"""
import datetime as dt
import random
import shutil
import sqlite3
import sys
from pathlib import Path

KEEP_TABLES = {"study_sessions", "attempts", "batch_steps", "survey_responses"}
REF = dt.datetime(2000, 1, 1)
SEED = 20271
# (table, timestamp columns, column holding the session id)
TIME_COLUMNS = [
    ("attempts", ["started_at", "answered_at"], "session_id"),
    ("batch_steps", ["started_at", "ended_at"], "session_id"),
    ("survey_responses", ["created_at"], "session_id"),
]
SESSION_TIME_COLUMNS = ["started_at", "ended_at", "abandoned_at"]


def parse(ts):
    return dt.datetime.fromisoformat(ts) if ts else None


def fmt(t):
    return t.isoformat(sep=" ") if t is not None else None


def anonymise(sources, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # one pseudonym per original user id, consistent across the databases and not tied to the original order
    users = set()
    for src in sources:
        con = sqlite3.connect(src)
        users |= {r[0] for r in con.execute("SELECT DISTINCT user_id FROM study_sessions")}
        con.close()
    users = sorted(users)
    shuffled = users[:]
    random.Random(SEED).shuffle(shuffled)
    pseudo = {u: i + 1 for i, u in enumerate(shuffled)}

    outputs = []
    for src, name in zip(sources, ["study_45_users.sqlite", "study_120_users.sqlite"]):
        dst = out_dir / name
        shutil.copyfile(src, dst)
        con = sqlite3.connect(dst)
        con.execute("PRAGMA foreign_keys=OFF")
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            if t not in KEEP_TABLES and not t.startswith("sqlite_"):
                con.execute(f"DROP TABLE {t}")
        # pseudonymous participant ids
        for row_id, uid in con.execute("SELECT id, user_id FROM study_sessions").fetchall():
            con.execute("UPDATE study_sessions SET user_id = ? WHERE id = ?", (pseudo[uid], row_id))
        # comments
        con.execute("UPDATE survey_responses SET free_text = NULL")
        # timestamps relative to the start of the session
        t0 = {sid: parse(s) for sid, s in con.execute("SELECT id, started_at FROM study_sessions")}
        for sid, start in t0.items():
            cols = ", ".join(f"{c} = ?" for c in SESSION_TIME_COLUMNS)
            vals = [fmt(REF + (parse(v) - start)) if v else None
                    for v in con.execute(f"SELECT {', '.join(SESSION_TIME_COLUMNS)} FROM study_sessions WHERE id = ?", (sid,)).fetchone()]
            con.execute(f"UPDATE study_sessions SET {cols} WHERE id = ?", (*vals, sid))
        for table, cols, key in TIME_COLUMNS:
            for row in con.execute(f"SELECT id, {key}, {', '.join(cols)} FROM {table}").fetchall():
                rid, sid, ts = row[0], row[1], row[2:]
                new = [fmt(REF + (parse(v) - t0[sid])) if v else None for v in ts]
                con.execute(f"UPDATE {table} SET {', '.join(c + ' = ?' for c in cols)} WHERE id = ?", (*new, rid))
        con.commit()
        con.execute("VACUUM")
        con.close()
        outputs.append(dst)
        print("wrote", dst)
    return outputs


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    anonymise([sys.argv[1], sys.argv[2]], sys.argv[3])
