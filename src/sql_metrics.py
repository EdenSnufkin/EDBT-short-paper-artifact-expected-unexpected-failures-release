# webapp/sql_metrics.py
"""Analyze webapp/study.sqlite: per-participant ("trajectory") metrics, their
aggregates, and comparison against this repo's simulated trajectories.

Real StudySession rows are converted into the exact step-dict shape metrics.py
already consumes, so metrics.py's own extract_metrics/aggregate_metrics/
unexpectedFailure_table and every plot_* function work unmodified on real
participant data -- there is only one implementation of each metric, shared
between simulated and real trajectories.

Usage:
    python sql_metrics.py
    python sql_metrics.py --group-by recommendation_method dataset
    python sql_metrics.py --models-dir ../Experiments_IRT_HillClimbing/MathE_FixedMastery \
                           --output-dir metrics_output
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import metrics as sim_metrics
from db import DB_PATH

from sqlalchemy import create_engine

WEBAPP_DIR = Path(__file__).resolve().parent

SESSION_STATUS_COMPLETED = "completed"

# Columns available to group real sessions by (see build_real_models / --group-by).
GROUPABLE_SESSION_COLUMNS = [
    "recommendation_method", "hillclimbing_algo", "mab_algo",
    "dataset", "frustration_type", "survey_mode",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tables(db_path: str | Path = DB_PATH) -> dict[str, pd.DataFrame]:
    """Load the four study tables into DataFrames, keyed by table name."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"No database at {db_path}. Run the webapp at least once, or pass --db-path."
        )
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        return {
            "sessions": pd.read_sql_query("SELECT * FROM study_sessions", engine),
            "attempts": pd.read_sql_query("SELECT * FROM attempts", engine),
            "batch_steps": pd.read_sql_query("SELECT * FROM batch_steps", engine),
            "surveys": pd.read_sql_query("SELECT * FROM survey_responses", engine),
        }
    finally:
        engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# Real session -> metrics.py trajectory shim
# ─────────────────────────────────────────────────────────────────────────────

def session_to_trajectory(
    session: pd.Series, attempts: pd.DataFrame, batch_steps: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert one StudySession's rows into a metrics.py-shaped trajectory
    (a list of step dicts, one per completed learning batch).

    Mirrors the step_info dicts built in examples/run_simulation.py /
    run_MAB.py, so metrics.py's extract_metrics() reads it unmodified.
    """
    session_id = session["id"]
    topic = attempts["topic"].iloc[0] if not attempts.empty else "unknown"

    learning_attempts = (
        attempts[(attempts["session_id"] == session_id) & (attempts["phase"] == "learning")]
        .sort_values(["batch_index", "slot_index"])
    )
    steps = (
        batch_steps[batch_steps["session_id"] == session_id]
        .sort_values("batch_index")
    )

    # Full per-question mastery_history, matching Student.mastery_history: one
    # {skill: value} dict per answered question, in answer order.
    mastery_history = [
        {topic: float(row.mastery_after)}
        for row in learning_attempts.itertuples()
        if row.mastery_after is not None
    ]

    trajectory: list[dict[str, Any]] = []
    is_mab = pd.notna(session.get("mab_algo"))

    for step_row in steps.itertuples():
        batch_attempts = learning_attempts[learning_attempts["batch_index"] == step_row.batch_index]

        step_info: dict[str, Any] = {
            "step": int(step_row.batch_index),
            "state": {
                "student_id": str(session["user_id"]),
                "mastery_after": float(step_row.mastery_after) if step_row.mastery_after is not None else np.nan,
                "unexpectedFailure": (
                    float(step_row.unexpected_failure_after)
                    if step_row.unexpected_failure_after is not None else np.nan
                ),
                "failed_questions": {},
                "cleared_questions": {},
            },
            "action": {
                "selected_questions": batch_attempts["question_index"].tolist(),
                "questions_difficulty": batch_attempts["difficulty"].tolist(),
                "question_results": {
                    int(r.question_index): bool(r.is_correct) for r in batch_attempts.itertuples()
                },
            },
            # Human response time spent on the batch (seconds) -- NOT the same
            # quantity as simulated "latency" (recommender compute time), but
            # kept under the same key for structural compatibility; metrics.py
            # does not currently read this field for either source.
            "latency": float(batch_attempts["response_time_ms"].sum()) / 1000.0,
        }
        if is_mab and pd.notna(step_row.algo_used):
            step_info["action"]["arm_used"] = step_row.algo_used

        trajectory.append(step_info)

    if trajectory:
        trajectory[-1]["state"]["mastery_history"] = mastery_history

    return trajectory


def build_real_models(
    sessions: pd.DataFrame, attempts: pd.DataFrame, batch_steps: pd.DataFrame,
    group_by: list[str] | None = None, completed_only: bool = True, label_prefix: str = "real",
) -> dict[str, list[list[dict[str, Any]]]]:
    """Group real sessions into named "models" of trajectories, metrics.py-shaped.

    group_by: StudySession columns to group by (see GROUPABLE_SESSION_COLUMNS).
    None or [] puts every qualifying session into one group, f"{label_prefix}".
    """
    df = sessions[sessions["status"] == SESSION_STATUS_COMPLETED] if completed_only else sessions
    df = df.copy()

    if not group_by:
        df["_group"] = label_prefix
    else:
        df["_group"] = df[group_by].astype(str).agg("_".join, axis=1)
        df["_group"] = label_prefix + "_" + df["_group"]

    models: dict[str, list[list[dict[str, Any]]]] = {}
    for group_name, group_df in df.groupby("_group"):
        trajectories = [
            session_to_trajectory(row, attempts, batch_steps) for _, row in group_df.iterrows()
        ]
        trajectories = [t for t in trajectories if t]  # drop sessions with zero completed batches
        if trajectories:
            models[group_name] = trajectories

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory (per-session) summary -- the human-only, SQL-native view
# ─────────────────────────────────────────────────────────────────────────────

def session_summary_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per session: config, outcomes, accuracy, response times, survey count."""
    sessions, attempts, batch_steps, surveys = (
        tables["sessions"], tables["attempts"], tables["batch_steps"], tables["surveys"]
    )
    rows = []
    for _, sess in sessions.iterrows():
        sid = sess["id"]
        pretest = attempts[(attempts["session_id"] == sid) & (attempts["phase"] == "pretest")]
        learning = attempts[(attempts["session_id"] == sid) & (attempts["phase"] == "learning")]
        steps = batch_steps[batch_steps["session_id"] == sid]
        sess_surveys = surveys[surveys["session_id"] == sid]

        rows.append({
            "session_id": sid,
            "user_id": sess["user_id"],
            "status": sess["status"],
            "abandoned_phase": sess["abandoned_phase"],
            "dataset": sess["dataset"],
            "recommendation_method": sess["recommendation_method"],
            "algo": sess["hillclimbing_algo"] or sess["mab_algo"],
            "frustration_type": sess["frustration_type"],
            "batches_completed": len(steps),
            "batches_planned": sess["n_batches"],
            "pretest_accuracy": pretest["is_correct"].mean() if len(pretest) else np.nan,
            "learning_accuracy": learning["is_correct"].mean() if len(learning) else np.nan,
            "initial_mastery": sess["pretest_mastery_init"],
            "final_mastery": sess["final_mastery"],
            "skill_gained": (
                sess["final_mastery"] - sess["pretest_mastery_init"]
                if pd.notna(sess["final_mastery"]) and pd.notna(sess["pretest_mastery_init"]) else np.nan
            ),
            "reached_mastery_threshold": (
                pd.notna(sess["final_mastery"]) and sess["final_mastery"] >= sess["mastery_stop_threshold"]
            ),
            "final_unexpected_failure": sess["final_unexpected_failure"],
            "mean_response_time_ms": learning["response_time_ms"].mean() if len(learning) else np.nan,
            "median_response_time_ms": learning["response_time_ms"].median() if len(learning) else np.nan,
            "n_survey_administrations": len(sess_surveys),
            "started_at": sess["started_at"],
            "ended_at": sess["ended_at"],
        })
    return pd.DataFrame(rows)


def aggregate_summary(summary: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    """Mean/median/std/min/max/count of every numeric metric in session_summary_table,
    optionally grouped by one or more columns (e.g. ["recommendation_method"])."""
    numeric_cols = summary.select_dtypes(include="number").columns.tolist()
    if not group_by:
        agg = summary[numeric_cols].agg(["mean", "median", "std", "min", "max", "count"])
        return agg.T
    return summary.groupby(group_by)[numeric_cols].agg(["mean", "median", "std", "min", "max", "count"])


def funnel_summary(sessions: pd.DataFrame) -> pd.DataFrame:
    """Where participants are / dropped off: counts by status, and by abandoned_phase."""
    by_status = sessions["status"].value_counts().rename("count").to_frame()
    by_abandon_phase = (
        sessions[sessions["status"] == "abandoned"]["abandoned_phase"]
        .value_counts().rename("count").to_frame()
    )
    return pd.concat({"by_status": by_status, "by_abandoned_phase": by_abandon_phase})


def response_time_summary(attempts: pd.DataFrame) -> pd.DataFrame:
    """Response-time stats overall and split by phase / correctness -- no
    simulated equivalent, since simulated answers have no human think-time."""
    def stats(df: pd.DataFrame) -> dict:
        rt = df["response_time_ms"]
        return {
            "n": len(rt), "mean_ms": rt.mean(), "median_ms": rt.median(),
            "std_ms": rt.std(), "min_ms": rt.min(), "max_ms": rt.max(),
        }

    rows = {"all": stats(attempts)}
    for phase in attempts["phase"].unique():
        rows[f"phase={phase}"] = stats(attempts[attempts["phase"] == phase])
    for is_correct, label in [(True, "correct"), (False, "incorrect")]:
        rows[f"answer={label}"] = stats(attempts[attempts["is_correct"] == is_correct])
    return pd.DataFrame(rows).T


def survey_summary(surveys: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """NASA-TLX subscale means overall / by trigger reason / by administration
    index, plus how often participants said their opinion had changed."""
    if surveys.empty:
        return {"overall": pd.DataFrame(), "by_trigger_reason": pd.DataFrame(),
                "by_administration_index": pd.DataFrame(), "opinion_changed_rate": pd.DataFrame()}

    subscales = ["temporal_demand", "mental_demand", "frustration", "perceived_performance", "effort"]
    overall = surveys[subscales].agg(["mean", "std", "count"]).T
    by_trigger = surveys.groupby("trigger_reason")[subscales].mean()
    by_admin = surveys.groupby("administration_index")[subscales].mean()

    changed = surveys["opinion_changed"].dropna()
    opinion_changed_rate = pd.DataFrame({
        "n_repeat_administrations": [len(changed)],
        "changed_rate": [changed.mean() if len(changed) else np.nan],
    })

    return {
        "overall": overall,
        "by_trigger_reason": by_trigger,
        "by_administration_index": by_admin,
        "opinion_changed_rate": opinion_changed_rate,
    }


def mab_arm_usage_summary(sessions: pd.DataFrame, batch_steps: pd.DataFrame) -> pd.DataFrame:
    """Per real-MAB session: fraction of batches spent on each HillClimbing arm
    and the mean observed reward -- SQL-native complement to
    metrics.plot_mab_arm_usage_statistics (which the merged-model comparison
    path also produces, once real sessions are shimmed into trajectories)."""
    mab_sessions = sessions[sessions["recommendation_method"] == "mab"]
    if mab_sessions.empty:
        return pd.DataFrame()

    rows = []
    for _, sess in mab_sessions.iterrows():
        steps = batch_steps[batch_steps["session_id"] == sess["id"]]
        if steps.empty:
            continue
        usage = steps["algo_used"].value_counts(normalize=True)
        row = {"session_id": sess["id"], "mab_algo": sess["mab_algo"],
               "mean_reward": steps["mab_reward"].mean()}
        row.update(usage.to_dict())
        rows.append(row)
    return pd.DataFrame(rows).set_index("session_id")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison against simulated trajectories (reuses metrics.py wholesale)
# ─────────────────────────────────────────────────────────────────────────────

def compare_to_simulated(
    real_models: dict[str, list[list[dict[str, Any]]]],
    models_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, dict]:
    """Merge real (SQL-derived) and simulated (models_dir) trajectories into
    one dict and run metrics.py's own stats + every plot_* function on it, so
    real participants are compared to simulated policies with the identical
    metric definitions and plot styling metrics.py already uses.
    """
    simulated_models = sim_metrics.load_all_models(Path(models_dir))
    merged = {**simulated_models, **real_models}
    if not merged:
        raise ValueError("No trajectories to compare (empty models_dir and no real trajectories).")

    print(f"Comparing {len(simulated_models)} simulated group(s) "
          f"and {len(real_models)} real group(s): {list(merged)}")

    stats = sim_metrics.compute_model_stats(merged)
    styles = sim_metrics._model_styles(list(stats))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_metrics.plot_skill_gain(stats, styles, output_dir)
    sim_metrics.plot_skill_progression(stats, styles, output_dir)
    sim_metrics.plot_mastery_and_iterations(stats, styles, output_dir)
    sim_metrics.plot_unexpectedFailure(stats, styles, output_dir)
    sim_metrics.plot_time_over_failure_threshold(stats, styles, output_dir)
    if any("arm_used" in step.get("action", {}) for trajs in merged.values() for traj in trajs for step in traj):
        sim_metrics.plot_mab_arm_usage_statistics(merged, save_dir=output_dir)

    uf_table = sim_metrics.unexpectedFailure_table(merged)
    uf_table.to_csv(output_dir / "unexpectedFailure_table.csv")
    print("\nunexpectedFailure summary (real + simulated):")
    print(uf_table)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze webapp/study.sqlite and optionally compare to simulated trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python sql_metrics.py --group-by recommendation_method \\\n"
            "      --models-dir ../Experiments_IRT_HillClimbing/MathE_FixedMastery \\\n"
            "      --output-dir metrics_output"
        ),
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH,
                         help=f"Path to study.sqlite (default: {DB_PATH}).")
    parser.add_argument("--output-dir", type=Path, default=WEBAPP_DIR / "metrics_output",
                         help="Directory to write CSV tables and (if --models-dir given) plots into.")
    parser.add_argument("--group-by", nargs="*", default=["recommendation_method"],
                         choices=GROUPABLE_SESSION_COLUMNS,
                         help="StudySession columns to group real trajectories by for comparison "
                              "(default: recommendation_method). Pass with no values to lump all "
                              "real sessions into one group.")
    parser.add_argument("--include-incomplete", action="store_true",
                         help="Include abandoned/in-progress sessions in the trajectory comparison "
                              "(default: completed sessions only; the SQL-only tables always include all).")
    parser.add_argument("--models-dir", type=Path, default=None,
                         help="Directory of simulated model subdirectories (as consumed by metrics.py) "
                              "to compare real participants against, e.g. "
                              "Experiments_IRT_HillClimbing/MathE_FixedMastery.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_tables(args.db_path)
    sessions, attempts, batch_steps, surveys = (
        tables["sessions"], tables["attempts"], tables["batch_steps"], tables["surveys"]
    )

    if sessions.empty:
        print(f"No sessions found in {args.db_path}.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(sessions)} session(s), {len(attempts)} attempt(s), "
          f"{len(batch_steps)} batch step(s), {len(surveys)} survey response(s).\n")

    # ── per-trajectory + aggregate (the "each trajectory, and the average") ──
    summary = session_summary_table(tables)
    summary.to_csv(args.output_dir / "session_summary.csv", index=False)
    print("=== Per-session summary ===")
    print(summary.to_string(index=False))

    print("\n=== Aggregate summary (all sessions) ===")
    print(aggregate_summary(summary))

    if args.group_by:
        print(f"\n=== Aggregate summary (by {', '.join(args.group_by)}) ===")
        print(aggregate_summary(summary, group_by=args.group_by))

    # ── funnel / response time / survey (human-only views) ──
    print("\n=== Completion funnel ===")
    print(funnel_summary(sessions))

    if not attempts.empty:
        print("\n=== Response time summary ===")
        print(response_time_summary(attempts))

    if not surveys.empty:
        print("\n=== Survey (NASA-TLX-style) summary ===")
        for name, df in survey_summary(surveys).items():
            print(f"-- {name} --")
            print(df)

    mab_usage = mab_arm_usage_summary(sessions, batch_steps)
    if not mab_usage.empty:
        print("\n=== MAB arm usage (per real session) ===")
        print(mab_usage)
        mab_usage.to_csv(args.output_dir / "mab_arm_usage.csv")

    # ── comparison against simulated trajectories ──
    if args.models_dir is not None:
        real_models = build_real_models(
            sessions, attempts, batch_steps,
            group_by=args.group_by, completed_only=not args.include_incomplete,
        )
        if not real_models:
            print("\nNo qualifying real trajectories to compare (try --include-incomplete).")
        else:
            compare_to_simulated(real_models, args.models_dir, args.output_dir)


if __name__ == "__main__":
    main()
