Distil durable lessons from the research this system has just done.

These lessons are saved **without asking the user** — they are the system learning about
its own work, and pausing for permission on each would be friction with no gain. That
autonomy comes with a hard limit, below.

## Two categories, two different bars

**Process lessons** — about *how to do the research*. No citation needed, because they
assert nothing about biology.

**Substantive patterns** (`kind: "pattern"`) — a generalisation about the domain itself.
Allowed, and often the most valuable thing you can record. But it **requires
`claim_ids`**: the `[k]` indices of the verified claims the pattern is derived from,
taken from the claims block below. If any index you give was not shown to you, the whole
lesson is thrown away — not just that index.

That gate is the whole reason patterns are permitted. A lesson saved here carries no
PMID of its own and is injected into future prompts — without provenance, the system
would be teaching itself its own guesses, and no gate downstream would catch it. With
`claim_ids`, the pattern is traceable to material that already passed two verification
gates.

| record this | as | why | |---|---|---| | "Queries pairing a mechanism term with an
outcome term return nothing; upstream and applied literature use different vocabulary" |
`search_lesson` | about searching | | "The dextromethorphan hypothesis was refuted for
inferring an acute effect from chronic-dosing data" | `dead_end` | about the reasoning
that failed | | "Three refuted hypotheses shared the same failure: acutely raising
glutamatergic tone" | `pattern` + `claim_ids` | substantive, derived from cited material
| | "Mechanism X does not produce outcome Y" | **nothing** | a bare assertion with no
claims behind it |

If you want to record a pattern but cannot name the verified claims it comes from,
record nothing. Recording nothing is correct, not a failure.

## What is worth recording

`dead_end` — a hypothesis or mechanism that was refuted, and *the specific reason*. The
reason is the whole value: "refuted" alone tells a future generator nothing, while
"refuted because it inferred an acute effect from chronic-dosing data" stops it from
making the same inferential move.

`search_lesson` — something about how to search this domain. A query shape that returns
nothing, a vocabulary mismatch between upstream and applied literature, an index term
that works better than the free-text equivalent.

`source_lesson` — something about a source's behaviour. Unstructured abstracts, missing
publication types, records without abstracts.

## What is not worth recording

- restatements of what the corpus already holds as claims - one-off events with no
generalisation ("this fetch timed out") - anything you could not act on next time -
observations already in the lessons list below

The bar: **would knowing this change how the next round of research is run?** If not,
skip it. These lessons are injected into future prompts, so a padded list degrades every
later decision — and unlike the user-facing memories, nobody is confirming these, so
restraint here is the only filter.

Recording nothing is a valid and common answer.

## Fields

`text` — one self-contained sentence, understandable without this context.

`provenance_note` — which event you are generalising from: a `[k]` from the list below,
or a PMID. This is what makes the lesson auditable later; the system replaces the `[k]`
with a durable label before saving.

`claim_ids` — required for `kind: "pattern"`, empty for process lessons. Give the `[k]`
indices from the claims block, nothing else.

## Lessons already recorded — do not repeat

«lessons»

## How to refer to what you were shown

Every citable item below is numbered `[k]`. **`[k]` is the only kind of number you may
write.** It is a position in this list, not a database id — the same object gets a
different `[k]` next time, so the system rewrites your `[k]` into a durable label when
it saves the lesson. Two consequences:

- `claim_ids` takes `[k]` indices, and only ones from the claims block. An index that
names a hypothesis or a search is a wrong reference, not a near miss: the lesson is
discarded and the mistake is logged. - Do not put `[k]` inside `text`. The lesson has to
read correctly on its own.

PMIDs in the sterile-sources block have no `[k]`. Those sources hold no verified claim,
so there is nothing there to cite. Refer to them by PMID in `provenance_note`.

## Recent research activity

«activity»
