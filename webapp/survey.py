# webapp/survey.py
"""NASA-TLX-style survey item definitions and SURVEY_MODE trigger scheduling."""
from dataclasses import dataclass, field

import settings

# (field_name, label, prompt)
TLX_ITEMS = [
    ("temporal_demand", "Temporal Demand",
     "How hurried or rushed did you feel during the session?"),
    ("mental_demand", "Mental Demand",
     "How mentally demanding was it to answer the questions?"),
    ("frustration", "Frustration",
     "How insecure, discouraged, irritated, stressed, or annoyed did you feel?"),
    ("perceived_performance", "Perceived Performance",
     "How successful do you feel you were at answering the questions?"),
    ("effort", "Effort",
     "How hard did you have to work to reach your level of performance?"),
]

SCALE_MIN_LABEL = "Low"
SCALE_MAX_LABEL = "High"
SCALE_POINTS = 7


@dataclass
class SurveyTriggerState:
    """Per-session scheduling state for SURVEY_MODE. Lives in st.session_state."""

    administrations_done: int = 0
    # The threshold is never raised after a survey, as in the study: in the threshold modes a survey is triggered after
    # every batch for which the cumulative score stays at or above it (see README, "Behaviour worth knowing").
    next_threshold: float = field(default_factory=lambda: settings.SURVEY_FAILURE_THRESHOLD)

    def fixed_count_batch_targets(self, n_batches: int) -> list[int]:
        """0-based batch indices after which a survey should fire, spread
        evenly across the planned batches with the last target on the final batch."""
        count = max(1, settings.SURVEY_FIXED_COUNT)
        return sorted({
            min(n_batches - 1, max(0, round(i * n_batches / count) - 1))
            for i in range(1, count + 1)
        })


def should_trigger_after_batch(
    state: SurveyTriggerState,
    batch_index: int,
    n_batches: int,
    any_incorrect: bool,
    cumulative_unexpected_failure: float,
    mode: str,
) -> bool:
    """Decide whether a survey should be shown right after this learning batch.
    mode == "end_only" never triggers here; that case is handled once, after
    the learning loop ends. `mode` is the session's own assigned survey_mode
    (see assignment.pick_balanced), not necessarily settings.SURVEY_MODE."""
    if mode == "end_only":
        return False

    if mode == "fixed_count":
        return batch_index in state.fixed_count_batch_targets(n_batches)

    if mode == "every_failure":
        return any_incorrect

    if mode == "threshold":
        return cumulative_unexpected_failure >= state.next_threshold

    if mode == "fixed_count_or_threshold":
        fixed_hit = batch_index in state.fixed_count_batch_targets(n_batches)
        threshold_hit = cumulative_unexpected_failure >= state.next_threshold
        return fixed_hit or threshold_hit

    raise ValueError(f"Unknown SURVEY_MODE: {mode!r}")


def needs_fallback_final_survey(state: SurveyTriggerState, mode: str) -> bool:
    """Guarantee at least one survey response per session for non-end_only modes."""
    return mode != "end_only" and state.administrations_done == 0
