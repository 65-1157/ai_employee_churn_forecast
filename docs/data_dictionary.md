# Data Dictionary — Raw Input (IBM HR Attrition dataset)

Field definitions for the raw source used in this MVP (`data_loader.load_raw_data()`).
This is a public, fictional dataset used as a structural stand-in for the client's
real questionnaire data — column meanings below are the dataset's own documented
definitions, not derived from any client data.

Verified against the actual data: unique-value counts and constant-field flags below
were checked directly against `data/raw/ibm_hr_attrition.csv`, not assumed.

## Self-explanatory fields (no lookup needed)

| Column | Type | Notes |
|---|---|---|
| `Age` | int | Employee age in years |
| `Attrition` | Yes/No | **Target variable** — whether the employee left |
| `Department` | text | Sales / Research & Development / Human Resources |
| `DistanceFromHome` | int | Distance from home to workplace |
| `EducationField` | text | Field of study |
| `EmployeeNumber` | int | Unique employee identifier (1,470 unique values — one per row) |
| `Gender` | text | |
| `JobRole` | text | 9 distinct roles |
| `MaritalStatus` | text | Single / Married / Divorced |
| `MonthlyIncome` | int | Monthly salary |
| `NumCompaniesWorked` | int | Number of employers prior to this one |
| `OverTime` | Yes/No | Whether the employee regularly works overtime |
| `PercentSalaryHike` | int | % salary increase at the last review |
| `TotalWorkingYears` | int | Total years of professional experience (any employer) |
| `TrainingTimesLastYear` | int | Number of trainings attended in the last year |
| `YearsAtCompany` | int | Tenure at the current company |
| `YearsInCurrentRole` | int | Years in the current role specifically |
| `YearsSinceLastPromotion` | int | Years since the last promotion |
| `YearsWithCurrManager` | int | Years reporting to the current manager |

## Fields that need a lookup — ordinal scales, not raw counts

These look like plain integers but are actually **coded categories**. Treating
them as continuous numbers (e.g. averaging them) is technically possible but
loses the intended meaning — worth knowing before building features on top of them.

| Column | Scale | Meaning |
|---|---|---|
| `Education` | 1–5 | 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor |
| `EnvironmentSatisfaction` | 1–4 | 1=Low, 2=Medium, 3=High, 4=Very High |
| `JobInvolvement` | 1–4 | 1=Low, 2=Medium, 3=High, 4=Very High |
| `JobSatisfaction` | 1–4 | 1=Low, 2=Medium, 3=High, 4=Very High |
| `RelationshipSatisfaction` | 1–4 | 1=Low, 2=Medium, 3=High, 4=Very High |
| `WorkLifeBalance` | 1–4 | 1=Bad, 2=Good, 3=Better, 4=Best |
| `PerformanceRating` | 1–4 (only 3–4 appear in this data) | 1=Low, 2=Good, 3=Excellent, 4=Outstanding — **verified: this dataset only contains values 3 and 4**, so the scale's lower half is unrepresented here |
| `JobLevel` | 1–5 | Seniority level, 1=junior … 5=most senior (not separately documented by IBM beyond ordinal ranking) |
| `StockOptionLevel` | 0–3 | Tier of stock option grant, 0=none |

## Fields that are genuinely ambiguous, even with the dataset's own documentation

| Column | Issue |
|---|---|
| `DailyRate`, `HourlyRate`, `MonthlyRate` | Three separate "rate" fields that are **not mutually consistent** (e.g. `MonthlyRate` does not equal `MonthlyIncome`, and `DailyRate × ~30` doesn't reconcile with `MonthlyIncome` either). Widely noted about this dataset: these appear to be independently randomized synthetic values, not derived from a real consistent pay structure. Treat with caution in any feature that assumes they represent real, reconcilable compensation figures. |

## Constant / zero-information fields — verified, not assumed

Checked directly against the data: these columns have exactly **one** unique value
across all 1,470 rows, so they carry no predictive signal and should not be
expected to appear with meaningful importance in the model.

| Column | Constant value | Why it's in the dataset anyway |
|---|---|---|
| `EmployeeCount` | always `1` | Artifact of how IBM generated this synthetic dataset |
| `Over18` | always `Y` | Same — every record represents an adult employee by construction |
| `StandardHours` | always `80` | Same — standard biweekly hours, not a real variable in this data |

These three are worth explicitly excluding from feature engineering (they already
are — see `feature_engineering.py`'s `NUMERIC`/`CATEGORICAL` lists, which don't
reference them) and are flagged here so nobody spends time investigating why they
show zero importance in a model — it's expected, not a bug.

## Important caveat

This dictionary describes the **public IBM demo dataset**, not the client's real
questionnaire. When real data replaces this source (see `schema_registry/` for
how that transition is handled), this file should be superseded by a dictionary
describing the client's actual fields — the scale meanings, constant-field list,
and rate-field caveats above almost certainly won't carry over unchanged.
