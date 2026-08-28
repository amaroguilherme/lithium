Classify a question for an evidence-synthesis system working on **treatment options for
bipolar I disorder with comorbid generalized anxiety disorder**.

The classification decides routing, and a wrong label is expensive in a specific way: a
question sent to the automatic research loop that literature *cannot* answer burns three
rounds of searching and then produces a fabricated answer with real-looking citations.

| kind | meaning | |---|---| | `FACTUAL` | Answerable by finding the right studies | |
`SYNTHESIS` | Answerable, but requires weighing several studies against each other | |
`PREFERENCE` | A trade-off between goals; evidence cannot settle it | | `CONTEXT` |
Depends on specifics of this case that no paper contains | | `METHODOLOGICAL` | About
how to conduct the search itself |

## The test

Ask yourself: **could a sufficiently thorough literature search settle this?**

- Yes, by locating studies → `FACTUAL` - Yes, but only by reconciling several →
`SYNTHESIS` - No, because it asks which outcome *should* matter more → `PREFERENCE` -
No, because it asks about this patient, this history, this constraint → `CONTEXT` - No,
because it asks how to search → `METHODOLOGICAL`

Empirical-sounding phrasing does not make a question `FACTUAL`. "Is it better to accept
residual anxiety than to risk a manic episode?" contains no answerable proposition — it
is a values judgment wearing clinical vocabulary. `PREFERENCE`.

Conversely, "how often does antidepressant monotherapy precipitate mania in bipolar I?"
is `FACTUAL` even though the answer informs a values judgment.

`targets` — the intervention or topic the question is about, lowercase. `""` if none.

`reason` — one sentence for the classification.

## Question

«question»
