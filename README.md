# Employee Churn Risk MVP

An individual-level tool to detect employee attrition risk early — **how likely** someone is to leave, and **how soon** — built as a proof-of-concept ahead of integrating the client's real (currently unavailable) questionnaire data.

This MVP uses the **IBM HR Analytics Attrition dataset** as a structural stand-in. The pipeline is architected so swapping in real data later requires no redesign — only reconfiguration.

---

## Project status

🟡 **MVP / demo stage.** Runs entirely on synthetic data (the public IBM dataset, plus fabricated AI-tooling questionnaire responses used only to demonstrate the pipeline's ability to absorb new data sources). Nothing here is a real-world finding yet. Real data collection is the recommended next step once stakeholders have reviewed this MVP.

---

## What this MVP actually answers

Two distinct questions, each backed by a different kind of model:

| Question | Answer type | Where it's built |
|---|---|---|
| **How likely** is this person to leave? | A probability (0–100%), tiered Low/Medium/High | `03_model_improve.ipynb`, `04_feature_selection.ipynb` |
| **How soon** might they leave? | An estimated time horizon (e.g. "~2.3 years"), from survival analysis | `07_time_to_event_modeling.ipynb` |

Together these drive a triage decision — a "high risk, near-term" employee gets a different response than a "high risk, long horizon" one. Neither number alone tells you what to do; combined, they do.

---

## How to read this repo, in order

The `notebooks/` folder is a sequence, not a pile — each one either sets up, extends, or exercises what came before. Run in this order the first time; after that, most can be re-run independently.

| # | Notebook | What it does |
|---|---|---|
| 00 | `00_setup_and_push.ipynb` | One-time: scaffolds the repo and pushes the initial pipeline code to GitHub |
| 01 | `01_first_run.ipynb` | Original standalone demo (superseded by 02+ below; kept for reference) |
| 02 | `02_dev_and_verify.ipynb` | Clones fresh from GitHub, runs the full weekly pipeline end-to-end, inspects raw input, checks training diagnostics (over/underfitting) |
| 03 | `03_model_improve.ipynb` | Compares 7 tree-based models (RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost) via Optuna hyperparameter search, 3-seed fairness evaluation, and a skill-weighted ensemble. Saves the actual best model as a reusable artifact. |
| 04 | `04_feature_selection.ipynb` | Checks whether all 39 input features are actually useful — finds redundant/noisy ones, trims to an optimal subset (fewer features, *better* test performance) |
| 05 | `05_ai_questionnaire_schema_demo.ipynb` | Demonstrates the schema-evolution pipeline (handling a questionnaire revision) using a **synthetic** AI-tooling questionnaire, kept in a quarantined data store, structurally separate from the churn model |
| 06 | `06_ai_questionnaire_experiment.ipynb` | A controlled experiment: does adding those synthetic fields change model performance? (Answer: no — as expected, since they carry no real signal by design) |
| 07 | `07_time_to_event_modeling.ipynb` | The "how soon" model — Kaplan-Meier retention curves and a Random Survival Forest for individual time-to-departure estimates |

**Dependencies between notebooks** (each one checks for what it needs and fails with a clear message if a prerequisite hasn't run):
```
00 ──▶ 02 ──▶ 03 ──┬──▶ 06 (needs 03's model + 05's data)
              │
              04    05 ──▶ 06
              │
              └──▶ 07 (needs 03's model, optional)
```

---

## Key findings so far (all on synthetic/public data — not yet real conclusions)

- **Model comparison** (03): a real 7-model, Optuna-tuned, 3-seed-fair comparison identifies a single best model (or an ensemble, whichever actually wins) — not an assumed default.
- **Overfitting is real and was measured, not assumed** (02, 03): every model shows a meaningful train/test performance gap on this small dataset; this is reported honestly rather than hidden behind a single accuracy number.
- **Fewer, better features beat more features** (04): trimming from 39 to ~18 well-chosen features *improved* test performance while cutting the overfitting gap nearly in half.
- **The synthetic AI-questionnaire data adds nothing — correctly** (05, 06): a rigorous check confirmed the fabricated fields carry no real predictive signal, exactly as expected since they were generated without reference to the outcome. This is a validation of the pipeline's honesty, not a disappointing result.
- **A genuine "when" model exists** (07): Random Survival Forest gives individual time-to-departure estimates, evaluated with the survival-analysis equivalent of AUC (concordance index), with duration-correlated features explicitly excluded to avoid a leakage trap.

---

## What's deliberately *not* done yet, and why

- **No real employee data.** By design — this MVP demonstrates the mechanics; stakeholder review comes before real data collection, not after.
- **The turnover-intent question was excluded from the synthetic questionnaire on purpose** — see `docs/decisions/0003-turnover-intent-question.md`. It's a genuinely valuable signal, but only with real answers and a proper time lag from the outcome; fabricating it would be circular, not informative.
- **The AI-questionnaire experiment's data is quarantined**, never merged into the actual churn model's training data — see `05`'s and `06`'s own summaries for the reasoning.

---

## Data sources & synthetic data methodology

This section documents exactly where every piece of data in this MVP came from, and — where synthetic — precisely how it was generated. Full transparency here matters: nothing below should be mistaken for real employee data.

### a) Data sources used

| Source | Type | Role in this MVP |
|---|---|---|
| **IBM HR Analytics Attrition dataset** | Public, fictional (created by IBM data scientists, not real employee records) | The core dataset — 1,470 employees, 35 columns. Used as-is; not itself further modified except for the weekly-snapshot noise described below. |
| **Weekly snapshot simulation** | Synthetic (light, documented perturbation of the IBM data) | Stands in for a live weekly HR data feed, since the IBM dataset itself is a single static file |
| **AI-tooling questionnaire responses** | Fully synthetic, fabricated for this MVP | Demonstrates schema-evolution handling; deliberately quarantined from the churn model — see notebooks `05`/`06` |
| **SMOTE-resampled training data** | Synthetic (algorithmically generated, standard technique) | Used only during model training to correct class imbalance, never presented as real observations |

### c) IBM dataset — source and reference

**Kaggle listing:** `https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset`
**Description (from the dataset's own listing):** a fictional dataset created by IBM data scientists to explore factors behind employee attrition.
**Technical mirror used in this repo's code** (`src/data_loader.py`'s `DATA_URL`, a public GitHub-hosted CSV of the same dataset, avoiding a Kaggle-API dependency): `https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv`

### b) & d) How synthetic data was generated, and the math behind it

**1. Weekly snapshot noise** (`data_loader.py`, `simulate_weekly_snapshot()`)
Since the IBM dataset is a single static file with no real time dimension, a small amount of week-to-week variation is simulated on three ordinal satisfaction fields (`JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`):
- **Distribution**: discrete uniform noise over `{-1, 0, +1}`, added to the field's existing value, then clipped back to the valid 1–4 range.
- **Purpose**: gives the weekly pipeline something to actually process each run — not a claim about real behavioral change.

**2. SMOTE (Synthetic Minority Oversampling Technique)** — used throughout model training (notebooks `02`, `03`, `04`, `06`, `07`)
Addresses the ~16% class imbalance (few actual "left" cases) by generating new synthetic minority-class examples, rather than just duplicating existing ones:
- **Method**: for each real minority-class point, one of its *k*-nearest minority-class neighbors is picked, and a new synthetic point is placed along the straight line between them: `x_new = x_i + λ(x_neighbor - x_i)`, where `λ` is drawn from a **Uniform(0, 1)** distribution.
- Applied **only to the training split**, never to validation/test data — so evaluation always reflects the real class distribution.

**3. AI-tooling questionnaire responses** (`05_ai_questionnaire_schema_demo.ipynb`) — the most involved synthetic generation in this project
For each of the 10 questions, every employee's answer is a **weighted random draw from a categorical (multinomial) distribution**, computed in three steps:
1. **Base rate**: a population-level starting probability per answer option. Where real external data existed, this was calibrated to it — most notably AI usage frequency, anchored to Gallup's Q4 2025 workplace AI-use study (≈49% never/rarely, ≈39% frequently, ≈12% daily). The other 9 questions use defensible but *not* externally-benchmarked base rates — this distinction is intentional and stated plainly, not implied to be equally rigorous.
2. **Conditional shift**: the base probabilities are adjusted using only **existing, real, non-target employee attributes** — Age, JobLevel, Department, JobRole (e.g., younger employees and R&D-aligned roles shifted toward higher AI usage, consistent with Gallup's reported generational and role-based adoption gaps). **`Attrition` is never referenced anywhere in this generation logic** — this is what allows a later experiment (notebook `06`) to confirm the synthetic fields carry no real predictive signal.
3. **Random draw**: `numpy.random.default_rng(seed=42).choice(options, p=probabilities)` — a seeded, reproducible weighted draw, not a deterministic rule.

Full generation code and per-field probability tables are in `05_ai_questionnaire_schema_demo.ipynb`, Section 4.

---

## Repository structure

```
employee-churn-mvp/
├── README.md                          # this file
├── requirements.txt
├── .github/workflows/                 # weekly pipeline automation
├── docs/
│   ├── data_dictionary.md             # every raw input field explained
│   └── decisions/                     # why specific design choices were made
├── schema_registry/                   # versioned questionnaire schemas + mapping rules
├── src/                                # production pipeline modules (see below)
├── scripts/run_weekly_pipeline.py     # the weekly orchestrator
├── notebooks/                         # 00-07, see table above
├── models/registry/                   # saved model artifacts, hyperparameters, comparison results
├── data/                              # raw, canonical (schema-mapped), and reference data
└── outputs/                           # weekly scores, reports, DQ checks, review flags, experiments
```

### `src/` modules, briefly

| Module | Role |
|---|---|
| `data_loader.py` | Pulls the raw data, simulates weekly snapshots |
| `data_quality.py` | The DQ radar — 6-dimension check, gates the pipeline before scoring |
| `feature_engineering.py` | Builds model-ready features from raw fields |
| `train_model.py`, `predict.py`, `explain.py` | Baseline training, weekly scoring, per-employee SHAP explanations |
| `schema_registry.py`, `schema_mapper.py`, `canonical_store.py`, `source_router.py` | Schema-evolution handling — lets the questionnaire change structure without breaking the pipeline or losing history |
| `model_evaluation.py`, `ensemble.py` | Champion/challenger model comparison and weighting |
| `review_trigger.py` | Flags when a human should review the pipeline (objective causes vs. ranked, ambiguous ones) |
| `report_generator.py` | Turns weekly scores into a readable per-employee report |

---

## How to run this

Everything runs in **Google Colab**, not locally — clone, install, execute, push, all from within each notebook. See any notebook's Section 0 for the one-time setup (a GitHub token stored as a Colab Secret).

---

## Path to production (once stakeholders are aligned)

1. Replace `data_loader.py`'s source with the client's real HRIS/questionnaire feed
2. Field a real version of the revised AI-tooling questionnaire from `05` (question wording already improved during this MVP)
3. Add a real, time-lagged turnover-intent question (see the decision doc) rather than continuing to exclude it
4. Re-run `03`/`04`'s model comparison and feature selection on real data — the synthetic-data results above should not be assumed to hold
5. Re-validate the survival model's concordance index on real durations
6. Pilot with a subset of consenting teams before full rollout

---

## Disclaimer

This is a demonstration MVP built on public, fictional data (IBM HR Analytics Attrition dataset) plus deliberately-fabricated synthetic questionnaire data used only to test pipeline mechanics. Nothing here should be used for actual HR decision-making in its current form.
