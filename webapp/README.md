# Study web application

The Streamlit application that ran the user study: consent, a 5-question pretest, up to 15 batches of 3 adaptive questions, NASA-TLX-style surveys, and logging of everything to SQLite. Recommendations come from the [`recommender`](../recommender) package of this repository, so the application is a thin shell around the algorithms described in the paper.

```bash
pip install -r requirements.txt -e ../recommender
STUDY_ALLOW_DEMO=1 streamlit run app.py          # try it on the demo question bank
pip install pytest && pytest                      # 13 tests, ~1 minute (includes a headless end-to-end session)
```

Without `STUDY_ALLOW_DEMO=1` the application only starts for URLs carrying a `?PROLIFIC_PID=<id>` parameter, as in the study (`?PROLIFIC_PID=anything` also works locally).

## Participant flow

`landing → consent → pretest → pretest result → batches → (review → survey?) × n → thank-you / Prolific redirect`

* **Assignment.** Each new participant gets the recommendation method (`hillclimbing` or `mab`) and the survey mode (`fixed_count` or `fixed_count_or_threshold`) that currently have the fewest sessions (`assignment.py`), ties broken at random.
* **Pretest.** 5 questions spread over the difficulty range; the answers feed the learner model like any other answer.
* **Batches.** After each batch the participant sees the correct answers. The next batch is recommended in a background thread while they read the feedback. The loop ends after 15 batches or when mastery reaches 0.9.
* **Surveys.** Five 7-point items (temporal demand, mental demand, frustration, perceived performance, effort). `fixed_count`: after batches 5, 10 and 15. `fixed_count_or_threshold`: the same, plus an extra survey after any batch that ends with AUFS ≥ 0.5. At least one survey is always shown.
* **End session.** Available on every page; the participant is sent to the "no bonus" completion page and cannot resume.

## Code map

| File | Role |
|---|---|
| `app.py` | pages and the state machine; database writes with Streamlit-rerun guards |
| `engine.py` | per-participant wrapper over `adaptive_recommender.LearningSession` |
| `settings.py` | every study parameter (batch size, thresholds, δ, λ, survey modes…) |
| `questions.py` | question file → `QuestionBank` + display content; answer scoring |
| `survey.py` | survey items and trigger rules |
| `assignment.py` | balanced random assignment |
| `db.py` | SQLAlchemy schema (`users`, `study_sessions`, `attempts`, `batch_steps`, `survey_responses`) |
| `ui_components.py`, `latex_rendering.py` | page texts (including the consent text used in the study), question rendering with LaTeX |
| `data/` | demo question bank and its generator, question file format |
| `tests/` | unit tests and a headless end-to-end session (`streamlit.testing`) |

## Configuration

Edit `settings.py` for the study design. Deployment values come from environment variables:

| Variable | Meaning |
|---|---|
| `STUDY_QUESTIONS_PATH` | question file (default: the demo bank) |
| `STUDY_DB_PATH` / `STUDY_DB_URL` | SQLite file (default `webapp/study.sqlite`) or any SQLAlchemy URL |
| `PROLIFIC_COMPLETION_CODE`, `PROLIFIC_NO_BONUS_CODE` | codes shown/redirected to at the end; without them a plain thank-you page is shown. The codes of the study are **not** in this repository |
| `STUDY_ALLOW_DEMO` | `1` lets the app start without `PROLIFIC_PID` |

## Differences with the application that ran the study

* The recommendation core is the clean `recommender` package (identical recommendations for identical seeds, see its README) instead of the research code base; the web layer was cleaned and reorganised, behaviour is unchanged.
* The database stores AUFS **and** AEFS for every answer and batch (`aufs_after`, `aefs_after`, `ufs`, `efs`, `aufs_before`, …), plus the session `seed`, `delta` and `lambda_decay`. The study database had a single "unexpected failure" column per table (`unexpected_failure_after`, `unexpected_failure_instant`, …), and no expected-failure columns; the released data keep that original layout.
* The Prolific completion codes, the unused completion-code pool and the question files were removed; the demo bank is synthetic.
* Every session draws a fresh random seed, stored in `study_sessions.seed`.

## Behaviour worth knowing

These are the behaviours of the application as it ran, kept on purpose:

* **Repeated surveys in the threshold mode.** `SurveyTriggerState.next_threshold` is never raised, so once AUFS reaches 0.5 a survey is shown after *every* following batch while it stays at or above that value (AUFS only drops when answers are correct).
* **A reload loses the learner state.** The learner model and recommender live in the Streamlit session (browser tab). After a page reload or a server restart the participant resumes at the page their database status indicates, but with a fresh learner model (mastery 0.2, no history) and a new recommender, while the database keeps the earlier answers.
* **One session per participant.** A participant identifier that already has a completed or abandoned session only sees the final page.
* **Consent text.** `ui_components.consent_block` is the text shown to participants. Replace it with the text approved by your own ethics board before running a study.
