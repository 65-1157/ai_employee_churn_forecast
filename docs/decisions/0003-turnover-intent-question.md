# 0003 — The turnover-intent question ("have you thought about seeking another job?")

## Status
Deliberately excluded from `schema_v3` (the AI-tooling questionnaire) and from
all synthetic data generation. This is a recommendation for **real, future
data collection**, not something to fabricate now.

## Context
The original 11-question AI-tooling draft included: *"Have you ever thought
to seek for another job, private or government? a) Yes b) No"*.

## Why it's excluded from the current schema and synthetic generation
Turnover intent is not a predictor of attrition in the way the other
questions are — it is, functionally, the target variable restated as a
survey question. Two concrete problems:

1. **Circularity in synthetic data.** Any synthetic answer we generated for
   this question would have to be built from some rule or correlation we
   define ourselves. If that rule is later "discovered" by a model as
   predictive of attrition, it isn't a finding — it's us reporting back a
   pattern we constructed. This would misrepresent the MVP's demonstrated
   accuracy to a client.
2. **Leakage in real data, if collected without care.** Even with real
   responses, if this question is fielded at the same time attrition is
   later measured, it's temporally entangled with the outcome rather than a
   leading indicator of it.

## Why it shouldn't just be thrown away permanently
Turnover-intent questions are one of the few direct ways to catch
**disengagement that hasn't yet shown up in any structural or behavioral
field** — exactly the "ghost quitting" scenario this project exists to
address. A model built only on tenure, overtime, and satisfaction scores can
miss someone who has already mentally checked out. This is a real signal,
not a flaw in the idea — the flaw is only in using it naively.

## Recommendation for real deployment
When real questionnaire data becomes available:
- Field a turnover-intent question as a **first-class, forward-looking
  predictor**, not excluded on principle.
- Collect it with an explicit **time lag** from the attrition outcome (e.g.,
  surveyed this quarter, attrition measured the following quarter) — this
  is what turns it from leakage into a legitimate early-warning signal.
- Treat it as one of the highest-value fields to validate first, given how
  directly it targets the disengagement pattern this MVP was commissioned
  to catch.
- Model it explicitly as a **leading indicator with a defined lag**, e.g. a
  new canonical field `turnover_intent_at_t_minus_1` distinct from the
  `attrition` outcome at time `t`, so the temporal structure is explicit in
  the schema rather than implicit and easy to violate.
