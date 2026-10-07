# Expected and unexpected failures in adaptive learning: code, data and figures

Companion material of the short paper **"On the Importance of Modeling Failure in Educational Recommender Systems"** (EDBT 2027, short paper).

The paper distinguishes failures that occur *above* a learner's estimated mastery (**expected**) from failures *below* it (**unexpected**), introduces measures of their severity and accumulation (UFS, EFS, AUFS, AEFS, TOFT), and studies them in 97 crowdsourced adaptive mathematics-learning sessions: their relation to mastery trajectories (RQ1), to self-reported frustration (RQ2), and whether unexpectedness explains frustration beyond the number of failures (RQ3).

This repository contains everything needed to **reproduce every figure and every number of the paper** from the study data.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+ ; developed and tested with 3.13.5
pip install -r requirements.txt

jupyter lab notebooks/paper_results.ipynb             # run all cells (about 30 seconds)
# or, without a browser:
python tools/run_notebook.py
```

The last cell of the notebook compares each quoted number with the paper and ends with

```
61 of 61 groups of quoted numbers match the paper.
```

The figures are written to `figures/` (300 dpi PNG and vector PDF), drawn at their final print size (3.33 in for one column, 7 in for `figure*`).

## Repository content

```
notebooks/paper_results.ipynb   the whole analysis, in the order of the paper (outputs included)
figures/                        the 8 figures of the paper (PNG + PDF), as produced by the notebook
data/                           anonymised study data (two SQLite databases), see below
src/                            the analysis modules the notebook imports
recommender/                    clean, tested implementation of the recommender used in the study (see recommender/README.md)
webapp/                         the Streamlit application of the study, on top of the recommender (see webapp/README.md)
tools/run_notebook.py           executes the notebook headlessly
tools/anonymize_data.py         how the released data were derived from the private study databases
requirements.txt                tested package versions
```

### The notebook

| Section | Paper |
|---|---|
| 1. Data | Sec. 4.1: 144 recorded sessions, 97 completed, participants, conditions, 293 surveys |
| 2. Failure scores | Sec. 3: UFS, EFS, AUFS, AEFS, TOFT (Eqs. 1-5) |
| 3. Figure 1 | running example, with the hyper-parameters of the experiment |
| 4. RQ1 | Sec. 4.2, Figures 2-6: outcomes, failure patterns, last batch, mastery profiles |
| 5. RQ2 | Sec. 4.3, Figure 7: NASA-TLX and frustration |
| 6. RQ3 | Sec. 4.4, Figure 8: unexpectedness beyond failure counts |
| 7. Check | every quoted number against the paper |

Hyper-parameters (Sec. 3 and 4.1): tolerance $\delta = 0.20$, decay $\lambda = 0.20$, strain threshold $\tau = 0.50$, batches of $k = 3$ questions, at most 15 batches, strict expected-failure score. All random procedures (permutation tests, bootstraps, jitter) use fixed seeds, so the results are deterministic.

### Source modules (`src/`)

Copied unchanged from the research code base that ran the study:

| File | Role |
|---|---|
| `sql_metrics.py`, `db.py` | read the study tables and turn a session into a trajectory |
| `metrics.py`, `benchmark.py` | trajectory metrics shared with the simulations of the research code base |
| `estimate_unexpectedFailure.py`, `unexpectedFailure_models.py` | recomputation of the unexpected-failure score from the answers |
| `estimate_expectedFailure.py`, `expectedFailure_models.py` | same for the expected-failure score |

The notebook uses them to reconstruct the mastery estimate from the answers and to read the gap between each question's difficulty and the mastery estimate; AUFS, AEFS and all analyses are then computed in the notebook with the paper's definitions, and AUFS is asserted to equal the library implementation.

## Data

`data/study_45_users.sqlite` and `data/study_120_users.sqlite` come from two runs of the same web application (participants recruited on Prolific). Four tables are provided in each:

| Table | One row per | Main columns |
|---|---|---|
| `study_sessions` | session | `id`, `user_id` (pseudonym), `recommendation_method` (`hillclimbing`/`mab`), `survey_mode` (`fixed_count`/`fixed_count_or_threshold`), `batch_size`, `n_batches`, `status` (`completed`, `abandoned`, `pretest`, `learning`), `pretest_correct`, `pretest_n`, `pretest_mastery_init`, `final_mastery`, `lambda_frustration`, `beta_frustration`, relative timestamps |
| `attempts` | answered question | `session_id`, `phase` (`pretest`/`learning`), `batch_index`, `slot_index`, `question_index`, `item_id`, `topic`, `difficulty`, `chosen_answer`, `correct_answer`, `is_correct`, `mastery_after`, `response_time_ms` |
| `batch_steps` | completed learning batch | `session_id`, `batch_index`, `algo_used`, `mastery_before`, `mastery_after`, `unexpected_failure_before`, `unexpected_failure_after`, `question_ids_json` |
| `survey_responses` | NASA-TLX administration | `session_id`, `administration_index`, `trigger_reason`, `batch_index_at_trigger`, `temporal_demand`, `mental_demand`, `frustration`, `perceived_performance`, `effort` (all 1-7), `opinion_changed` |

The `unexpected_failure_*` columns hold the value logged by the platform while the study ran; the paper's AUFS and AEFS are recomputed from the answers in the notebook.

### Anonymisation

The private databases are linked to participant accounts, so the released copies were derived with `tools/anonymize_data.py`:

* the table that links sessions to participant accounts (`users`) is not included;
* `user_id` is replaced by a random pseudonymous integer (consistent across the two databases);
* the free-text comments of the surveys are removed (`free_text` is empty);
* all timestamps are rewritten relative to the start of their session (start = 2000-01-01 00:00:00), which keeps durations and the order of events but removes calendar dates and times of day;
* everything else is unchanged: answers, difficulties, mastery and failure traces, NASA-TLX scores, response times, algorithm and survey-mode assignments.

The notebook gives the same results on the anonymised data as on the original data (all figures are pixel-identical and all quoted numbers match).

## Citation

```
TODO: add the BibTeX entry of the paper once the reference is final.
```

## License

TODO: choose a license for the code and one for the data before publishing the repository.
