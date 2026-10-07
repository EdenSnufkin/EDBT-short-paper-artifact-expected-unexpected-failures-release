import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

MASTERY_THRESHOLD = 0.9
unexpectedFailure_THRESHOLD = 0.5
SAVE_FIGURES = True
MAB_USED = True

# Matches a trailing "_YYYY-MM-DD_HH-MM-SS" timestamp appended to a model dir name.
_DATE_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_date(name: str) -> str:
    """Strip a trailing run timestamp (e.g. '_2026-08-13_13-11-48') from a model name."""
    return _DATE_SUFFIX_RE.sub("", name)


# Okabe-Ito colorblind-safe palette (plus black / grey for the 9th and 10th entries).
_MODEL_COLORS = ["#0072B2", "#E69F00", "#D55E00", "#56B4E9", "#7f7f7f",
                 "#009E73", "#CC79A7", "#F0E442", "#000000", "#999999"]
_MODEL_HATCHES = ["//", "\\\\", "..", "--", "xx", "++", "oo", "OO", "**", "||"]
_MODEL_LINESTYLES = ["--", "-.", ":", "-", (0, (3, 1, 1, 1)),
                      (0, (5, 1)), (0, (1, 1))]


def _model_styles(names: list[str]) -> dict[str, dict]:
    """Assign each model a consistent (color, hatch, linestyle) triple.

    Used across every plot in the script so a given model always looks the
    same, whether it's a bar, a box, or a line.
    """
    return {
        name: {
            "color": _MODEL_COLORS[i % len(_MODEL_COLORS)],
            "hatch": _MODEL_HATCHES[i % len(_MODEL_HATCHES)],
            "linestyle": _MODEL_LINESTYLES[i % len(_MODEL_LINESTYLES)],
        }
        for i, name in enumerate(names)
    }


def _style_legend_handles(styles: dict[str, dict], names: list[str]) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(facecolor="white", edgecolor=styles[name]["color"],
                        hatch=styles[name]["hatch"], linewidth=1.5, label=name)
        for name in names
    ]


def _bottom_legend(fig, handles, ncol: int):
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, -0.05), frameon=False)


def _pad_sequences(seqs: list[list[float]], pad_with_last: bool = True) -> np.ndarray:
    max_len = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        pad = s[-1] if (s and pad_with_last) else 0.0
        out.append(s + [pad] * (max_len - len(s)))
    return np.array(out)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_trajectories(model_dir: Path) -> list[list[dict]]:
    """Return a list of trajectories from a model directory.

    Each trajectory is a list of step dicts with keys:
        step, state (mastery_after, mastery_history, unexpectedFailure, …), action, latency
    """
    traj_file = _find_trajectory_file(model_dir)
    with open(traj_file) as f:
        data = json.load(f)

    if not data:
        return []

    # Top-level list of trajectories (each a list of steps).
    if isinstance(data[0], list):
        return data
    # Top-level flat list of steps (single trajectory).
    return [data]


def _find_trajectory_file(model_dir: Path) -> Path:
    """Find one dataset/topic trajectory file, with legacy-name support."""
    trajectory_files = sorted(model_dir.glob("*_trajectories.json"))
    if len(trajectory_files) == 1:
        return trajectory_files[0]
    if len(trajectory_files) > 1:
        raise ValueError(
            f"Multiple trajectory files found in {model_dir}: {trajectory_files}"
        )

    legacy_file = model_dir / "trajectories.json"
    if legacy_file.exists():
        return legacy_file
    raise FileNotFoundError(f"No trajectory JSON found in {model_dir}.")


def load_all_models(models_dir: Path) -> dict[str, list[list[dict]]]:
    """Return {model_name: [trajectory, …]} for every model subdirectory."""
    models: dict[str, list[list[dict]]] = {}
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            models[_strip_date(subdir.name)] = load_trajectories(subdir)
        except FileNotFoundError:
            continue
    return models


# ── Per-trajectory metric extraction ─────────────────────────────────────────

def _mastery_progression(traj: list[dict]) -> list[float]:
    """Per-question mastery from the cumulative mastery_history in the last step.

    mastery_history is appended after every question (not every batch), so this
    gives a smoother picture than sampling per step.
    """
    history = traj[-1]["state"].get("mastery_history", [])
    if not history:
        # Fall back to per-step mastery_after if history is missing.
        return [float(np.mean(s["state"]["mastery_after"])) for s in traj]
    return [float(np.mean(list(m.values()))) for m in history]


def _unexpectedFailure_progression(traj: list[dict]) -> list[float]:
    """Cumulative unexpectedFailure after each step (last value in each batch list)."""
    prog = []
    for step in traj:
        unexpectedFailure = step["state"].get("unexpectedFailure", np.nan)
        if isinstance(unexpectedFailure, list):
            prog.append(float(unexpectedFailure[-1]) if unexpectedFailure else np.nan)
        else:
            prog.append(float(unexpectedFailure))
    return prog


def extract_metrics(traj: list[dict]) -> dict:
    mastery_prog = _mastery_progression(traj)
    unexpectedFailure_prog = _unexpectedFailure_progression(traj)

    initial_mastery = mastery_prog[0] if mastery_prog else np.nan
    final_mastery = mastery_prog[-1] if mastery_prog else np.nan

    return {
        "skillgained": final_mastery - initial_mastery,
        "skillprogression": mastery_prog,
        "masteryrate": float(final_mastery >= MASTERY_THRESHOLD),
        "iterations": len(traj),
        "unexpectedFailure": unexpectedFailure_prog,
        "timeoverthreshold": sum(1 for f in unexpectedFailure_prog if f > unexpectedFailure_THRESHOLD),
    }


# ── Aggregation across trajectories ──────────────────────────────────────────

def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _extreme_err(mean: float, values: list[float]) -> tuple[float, float]:
    """(low, high) error-bar lengths spanning from the min to the max value."""
    if not values:
        return 0.0, 0.0
    return mean - min(values), max(values) - mean


def aggregate_metrics(per_traj: list[dict]) -> dict:
    """Average metrics across all trajectories of one model."""
    skill_progs = _pad_sequences([m["skillprogression"] for m in per_traj])
    unexpectedFailure_progs = _pad_sequences([m["unexpectedFailure"] for m in per_traj], pad_with_last=True)

    skillgained_vals    = [m["skillgained"]       for m in per_traj]
    masteryrate_vals    = [m["masteryrate"]        for m in per_traj]
    iterations_vals     = [m["iterations"]         for m in per_traj]
    tot_vals            = [m["timeoverthreshold"]  for m in per_traj]

    return {
        "skillgained":              float(np.mean(skillgained_vals)),
        "skillgained_vals":         skillgained_vals,
        "skillprogression":         np.mean(skill_progs, axis=0).tolist(),
        "masteryrate":              float(np.mean(masteryrate_vals)),
        "masteryrate_std":          _std(masteryrate_vals),
        "masteryrate_vals":         masteryrate_vals,
        "iterations":               float(np.mean(iterations_vals)),
        "iterations_vals":          iterations_vals,
        "unexpectedFailure":              np.mean(unexpectedFailure_progs, axis=0).tolist(),
        "timeoverthreshold":        float(np.mean(tot_vals)),
        "timeoverthreshold_vals":   tot_vals,
    }


def compute_model_stats(models: dict[str, list[list[dict]]]) -> dict[str, dict]:
    return {
        name: aggregate_metrics([extract_metrics(traj) for traj in trajectories])
        for name, trajectories in models.items()
    }


# ── Summary tables ────────────────────────────────────────────────────────────

def unexpectedFailure_table(models: dict[str, list[list[dict]]]) -> pd.DataFrame:
    """Per-algorithm summary statistics of unexpectedFailure (UF).

    Each trajectory contributes its own mean UF across steps; the table then
    reports mean/std/min/max/median of that per-trajectory scalar across all
    of a model's trajectories, plus the trajectory count.
    """
    rows = {}
    for name, trajs in models.items():
        traj_means = [float(np.mean(_unexpectedFailure_progression(traj))) for traj in trajs]
        rows[name] = {
            "mean":   float(np.mean(traj_means)),
            "std":    _std(traj_means),
            "min":    float(min(traj_means)),
            "max":    float(max(traj_means)),
            "median": float(np.median(traj_means)),
            "n":      len(traj_means),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _boxplot(ax, model_names, data_per_model, styles: dict[str, dict], xlabel, title, legend: bool = True):
    bp = ax.boxplot(
        data_per_model,
        patch_artist=True,
        orientation="horizontal",
        medianprops={"color": "black", "linewidth": 2},
        showfliers=False,
    )
    for patch, name in zip(bp["boxes"], model_names):
        patch.set_facecolor("white")
        patch.set_edgecolor(styles[name]["color"])
        patch.set_hatch(styles[name]["hatch"])
        patch.set_linewidth(1.5)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    if legend:
        _bottom_legend(ax.figure, _style_legend_handles(styles, model_names), len(model_names))


def plot_skill_gain(stats: dict[str, dict], styles: dict[str, dict], save_dir: Path):
    models = list(stats)
    data = [stats[m]["skillgained_vals"] for m in models]
    fig, ax = plt.subplots(figsize=(10, 6))
    _boxplot(ax, models, data, styles, "Skill Gain", "Skill Gain")
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(save_dir / "skill_gain.png", bbox_inches="tight")


def plot_skill_progression(stats: dict[str, dict], styles: dict[str, dict], save_dir: Path):
    models = list(stats)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in models:
        s = styles[name]
        ax.plot(stats[name]["skillprogression"], label=name,
                color=s["color"], linestyle=s["linestyle"], linewidth=1.8)
    ax.axhline(MASTERY_THRESHOLD, color="black", linestyle="dotted", label="mastery threshold")
    ax.set_title("Skill Progression")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Skill progression")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.05), frameon=False)
    if SAVE_FIGURES:
        fig.savefig(save_dir / "skill_progression.png", bbox_inches="tight")


def plot_skill_gain_per_step(stats: dict[str, dict], styles: dict[str, dict], save_dir: Path):
    """Average per-step skill gain (Δ mastery between consecutive questions), one line per model."""
    models = list(stats)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in models:
        s = styles[name]
        gains = np.diff(stats[name]["skillprogression"]).tolist()
        ax.plot(gains, label=name, color=s["color"], linestyle=s["linestyle"], linewidth=1.8)
    ax.axhline(0, color="black", linestyle="dotted", linewidth=1)
    ax.set_title("Average Skill Gain per Step")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Δ Skill progression")
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.05), frameon=False)
    if SAVE_FIGURES:
        fig.savefig(save_dir / "skill_gain_per_step.png", bbox_inches="tight")


def plot_mastery_and_iterations(stats: dict[str, dict], styles: dict[str, dict], save_dir: Path):
    """Side-by-side (I) mastery rate and (II) iterations bars, one per model."""
    models = list(stats)
    x = np.arange(len(models))
    bar_kw = dict(width=0.6, facecolor="white", linewidth=1.5,
                  capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"})

    fig, (ax_mastery, ax_iters) = plt.subplots(1, 2, figsize=(10, 5))

    mastery_vals = [stats[m]["masteryrate"] * 100 for m in models]
    for xi, name, v in zip(x, models, mastery_vals):
        s = styles[name]
        ax_mastery.bar(xi, v, edgecolor=s["color"], hatch=s["hatch"], **bar_kw)
    ax_mastery.set_xticks([])
    ax_mastery.set_ylabel("Mastery (%)")
    ax_mastery.set_ylim(0, 110)
    ax_mastery.set_title("I", fontweight="bold")
    ax_mastery.grid(axis="y", linestyle=":", alpha=0.4)

    iter_vals = [stats[m]["iterations"] for m in models]
    for xi, name, v in zip(x, models, iter_vals):
        s = styles[name]
        low, high = _extreme_err(v, stats[name]["iterations_vals"])
        ax_iters.bar(xi, v, yerr=[[low], [high]], edgecolor=s["color"], hatch=s["hatch"], **bar_kw)
    ax_iters.set_xticks([])
    ax_iters.set_ylabel("Iterations")
    ax_iters.set_title("II", fontweight="bold")
    ax_iters.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    _bottom_legend(fig, _style_legend_handles(styles, models), len(models))
    if SAVE_FIGURES:
        fig.savefig(save_dir / "mastery_and_iterations.png", bbox_inches="tight")


def plot_unexpectedFailure(stats: dict[str, dict], styles: dict[str, dict], save_dir: Path):
    models = list(stats)
    fig, ax = plt.subplots(figsize=(10, 6))
    all_vals = []
    for name in models:
        s = styles[name]
        unexpectedFailure = stats[name]["unexpectedFailure"]
        all_vals.extend(unexpectedFailure)
        ax.plot(unexpectedFailure, label=name, color=s["color"],
                linestyle=s["linestyle"], linewidth=1.8)
    ax.set_title("unexpectedFailure Progression")
    ax.set_xlabel("Step")
    ax.set_ylabel("unexpectedFailure")
    if all_vals:
        ax.set_ylim(min(all_vals) * 1.1 if min(all_vals) < 0 else None,
                    max(all_vals) * 1.1 if max(all_vals) > 0 else None)
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.05), frameon=False)
    if SAVE_FIGURES:
        fig.savefig(save_dir / "unexpectedFailure.png", bbox_inches="tight")


def plot_time_over_failure_threshold(stats: dict[str, dict], styles: dict[str, dict], save_dir: Path):
    models = list(stats)
    data = [stats[m]["timeoverthreshold_vals"] for m in models]
    fig, ax = plt.subplots(figsize=(10, 6))
    _boxplot(ax, models, data, styles, "Steps above threshold", f"Time over Failure Threshold = {unexpectedFailure_THRESHOLD}")
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(save_dir / "time_over_failure_threshold.png", bbox_inches="tight")


def _time_over_threshold_vals(models: dict[str, list[list[dict]]], threshold: float) -> dict[str, list[int]]:
    """Per-trajectory count of steps with unexpectedFailure > threshold, for each model."""
    return {
        name: [
            sum(1 for f in _unexpectedFailure_progression(traj) if f > threshold)
            for traj in trajs
        ]
        for name, trajs in models.items()
    }


def plot_time_over_failure_threshold_range(
    models: dict[str, list[list[dict]]],
    styles: dict[str, dict],
    thresholds: list[float],
    save_dir: Path,
    save: bool = True,
) -> None:
    """One figure with a Time-over-Failure-Threshold boxplot panel per value in `thresholds`.

    Recomputes the metric straight from the raw trajectories (rather than
    `stats`, which only carries the single threshold baked into extract_metrics),
    so it works for any threshold sweep. Pass save=False to only display the
    figure without writing a PNG.
    """
    model_names = list(models)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, threshold in zip(axes, thresholds):
        vals_by_model = _time_over_threshold_vals(models, threshold)
        data = [vals_by_model[m] for m in model_names]
        _boxplot(ax, model_names, data, styles, "Steps above threshold",
                 f"Threshold = {threshold:g}", legend=False)

    fig.tight_layout()
    _bottom_legend(fig, _style_legend_handles(styles, model_names), len(model_names))
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_dir / "time_over_failure_threshold_range.png", bbox_inches="tight")


# ── MAB Gantt chart ───────────────────────────────────────────────────────────

def _arm_runs(traj: list[dict], arm_name: str) -> list[tuple[int, int]]:
    """Return (start_step, width) pairs for consecutive runs of arm_name."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    run_end: int | None = None
    for step_info in traj:
        s = step_info["step"]
        used = step_info.get("action", {}).get("arm_used")
        if used == arm_name:
            if run_start is None:
                run_start = s
            run_end = s
        else:
            if run_start is not None:
                runs.append((run_start, run_end - run_start + 1))  # type: ignore[operator]
                run_start = run_end = None
    if run_start is not None:
        runs.append((run_start, run_end - run_start + 1))  # type: ignore[operator]
    return runs

def plot_mab_arm_usage(
    models: dict[str, list[list[dict]]],
    save_dir: Path,
    episode: int = 0,
) -> None:
    """Gantt-style chart of MAB arm selection per step.

    Y-axis: one row per (model, arm) pair, grouped by model.
    X-axis: step index.
    Each rectangle spans the consecutive steps where that arm was active.
    """
    # ── filter models that have arm_used in their trajectories ────────────────
    mab_models: dict[str, list[dict]] = {}
    for name, trajs in models.items():
        if episode >= len(trajs):
            print(f"  {name}: episode {episode} not available (only {len(trajs)} trajectories), skipping")
            continue
        traj = trajs[episode]
        if any("arm_used" in step.get("action", {}) for step in traj):
            mab_models[name] = traj

    if not mab_models:
        print("plot_mab_arm_usage: no MAB trajectories found (missing action.arm_used)")
        return

    # ── collect all arm names (order of first appearance) ─────────────────────
    seen: dict[str, None] = {}
    for traj in mab_models.values():
        for step in traj:
            arm = step.get("action", {}).get("arm_used")
            if arm:
                seen[arm] = None
    all_arms: list[str] = list(seen)
    arm_styles = _model_styles(all_arms)

    # ── build y-axis layout ───────────────────────────────────────────────────
    # One row per arm per model; a small gap between model groups.
    ROW_H = 0.8
    GAP = 0.5
    y_pos: dict[tuple[str, str], float] = {}
    y_ticks: list[float] = []
    y_labels: list[str] = []
    model_separators: list[float] = []

    y = 0.0
    for model_name in mab_models:
        for arm in all_arms:
            y_pos[(model_name, arm)] = y
            y_ticks.append(y)
            y_labels.append(f"{model_name}  |  {arm}")
            y += 1.0
        model_separators.append(y - 0.5)
        y += GAP

    total_height = max(4, int(y * 0.5))
    fig, ax = plt.subplots(figsize=(14, total_height))

    # ── draw rectangles ───────────────────────────────────────────────────────
    max_step = 0
    for model_name, traj in mab_models.items():
        if traj:
            max_step = max(max_step, traj[-1]["step"])
        for arm in all_arms:
            runs = _arm_runs(traj, arm)
            if not runs:
                continue
            row_y = y_pos[(model_name, arm)]
            ax.broken_barh(
                runs,
                (row_y - ROW_H / 2, ROW_H),
                facecolors="white",
                edgecolors=arm_styles[arm]["color"],
                hatch=arm_styles[arm]["hatch"],
                linewidth=1.2,
            )

    # ── model-group separator lines ───────────────────────────────────────────
    for sep_y in model_separators[:-1]:
        ax.axhline(sep_y, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlim(0, max_step + 1)
    ax.set_xlabel("Step")
    ax.set_title(f"MAB Arm Usage — episode {episode}")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.invert_yaxis()

    ax.legend(handles=_style_legend_handles(arm_styles, all_arms), loc="upper right", fontsize=9)

    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(save_dir / f"mab_arm_usage_ep{episode}.png")

def plot_mab_arm_usage_statistics(
    models: dict[str, list[list[dict]]],
    save_dir: Path,
) -> None:
    """Stacked horizontal bar chart of average arm-usage share per MAB model.

    For each model, counts how many steps each arm was chosen across all
    episodes, then divides by the total to get a fraction. The bars are
    stacked so the full bar always reaches 1.0.
    """
    mab_models: dict[str, list[list[dict]]] = {}
    for name, trajs in models.items():
        if trajs and any("arm_used" in step.get("action", {}) for step in trajs[0]):
            mab_models[name] = trajs

    if not mab_models:
        print("plot_mab_arm_usage_statistics: no MAB trajectories found")
        return

    # ── collect all arm names ─────────────────────────────────────────────────
    seen: dict[str, None] = {}
    for trajs in mab_models.values():
        for traj in trajs:
            for step in traj:
                arm = step.get("action", {}).get("arm_used")
                if arm:
                    seen[arm] = None
    all_arms: list[str] = list(seen)
    arm_styles = _model_styles(all_arms)

    # ── compute per-episode fractions, then mean ± std per (model, arm) ─────────
    model_names = list(mab_models)

    # ep_fracs[model][arm] = list of per-episode fractions
    ep_fracs: dict[str, dict[str, list[float]]] = {
        m: {a: [] for a in all_arms} for m in model_names
    }
    for model_name, trajs in mab_models.items():
        for traj in trajs:
            counts: dict[str, int] = {a: 0 for a in all_arms}
            for step in traj:
                arm = step.get("action", {}).get("arm_used")
                if arm and arm in counts:
                    counts[arm] += 1
            total = sum(counts.values())
            for arm in all_arms:
                ep_fracs[model_name][arm].append(counts[arm] / total if total else 0.0)

    fractions: dict[str, dict[str, float]] = {
        m: {a: float(np.mean(ep_fracs[m][a])) for a in all_arms} for m in model_names
    }
    stds: dict[str, dict[str, float]] = {
        m: {a: _std(ep_fracs[m][a]) for a in all_arms} for m in model_names
    }

    # ── stacked horizontal bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(3, len(model_names) * 0.8 + 1)))

    y_positions = np.arange(len(model_names))
    bar_height = 0.5

    for model_idx, model_name in enumerate(model_names):
        left = 0.0
        for arm in all_arms:
            frac = fractions[model_name][arm]
            std  = stds[model_name][arm]
            if frac == 0.0:
                left += frac
                continue
            ax.barh(
                y_positions[model_idx],
                frac,
                left=left,
                height=bar_height,
                facecolor="white",
                edgecolor=arm_styles[arm]["color"],
                hatch=arm_styles[arm]["hatch"],
                linewidth=1.2,
            )
            if frac > 0.04:
                ax.text(
                    left + frac / 2,
                    y_positions[model_idx],
                    f"{frac:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )
            # error bar centred on the segment midpoint
            ax.errorbar(
                left + frac / 2,
                y_positions[model_idx],
                xerr=std,
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=3,
            )
            left += frac

    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_names)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Fraction of steps")
    ax.set_title("MAB Arm Usage — average across all episodes")
    ax.axvline(1.0, color="gray", linewidth=0.5)

    ax.legend(handles=_style_legend_handles(arm_styles, all_arms), loc="lower right", fontsize=9)

    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(save_dir / "mab_arm_usage_statistics.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot HillClimbing metrics.")
    parser.add_argument("--models-dir", type=Path, required=True,
                         help="Directory containing one subdirectory per model.")
    parser.add_argument("--output-dir", type=Path, required=True,
                         help="Directory to save the generated plots into.")
    parser.add_argument("--threshold-range", type=float, nargs=3, metavar=("START", "STOP", "STEP"),
                         help="Sweep the failure threshold from START to STOP (inclusive) in STEP "
                              "increments, producing one Time-over-Failure-Threshold plot per value.")
    parser.add_argument("--no-save-threshold-range", action="store_true",
                         help="Show the threshold-range plots without saving them to --output-dir.")
    parser.add_argument("--only-threshold-range", action="store_true",
                         help="Skip every other plot and only produce the threshold-range sweep "
                              "(requires --threshold-range).")
    args = parser.parse_args()
    if args.only_threshold_range and not args.threshold_range:
        parser.error("--only-threshold-range requires --threshold-range")
    return args


def main():
    args = parse_args()
    models_dir = args.models_dir
    output_dir = args.output_dir

    models = load_all_models(models_dir)
    if not models:
        print(f"No models found in {models_dir}")
        return

    print(f"Loaded {len(models)} model(s): {list(models)}")
    for name, trajs in models.items():
        print(f"  {name}: {len(trajs)} trajectory/trajectories")

    stats = compute_model_stats(models)
    styles = _model_styles(list(stats))

    if SAVE_FIGURES:
        os.makedirs(output_dir, exist_ok=True)

    if not args.only_threshold_range:
        plot_skill_gain(stats, styles, output_dir)
        plot_skill_progression(stats, styles, output_dir)
        plot_skill_gain_per_step(stats, styles, output_dir)
        plot_mastery_and_iterations(stats, styles, output_dir)
        plot_unexpectedFailure(stats, styles, output_dir)
        plot_time_over_failure_threshold(stats, styles, output_dir)
        if MAB_USED:
            plot_mab_arm_usage(models, save_dir=output_dir, episode=0)
            plot_mab_arm_usage_statistics(models, save_dir=output_dir)

        uf_table = unexpectedFailure_table(models)
        print("\nunexpectedFailure summary:")
        print(uf_table)
        if SAVE_FIGURES:
            uf_table.to_csv(output_dir / "unexpectedFailure_table.csv")

    if args.threshold_range:
        start, stop, step = args.threshold_range
        thresholds = np.arange(start, stop + step / 2, step).tolist()
        plot_time_over_failure_threshold_range(
            models, styles, thresholds, output_dir, save=not args.no_save_threshold_range
        )

    plt.show()


if __name__ == "__main__":
    main()
