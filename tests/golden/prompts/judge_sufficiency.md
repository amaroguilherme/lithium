You are deciding whether the retrieved evidence is enough to answer a research question
about **treatment options for bipolar I disorder with comorbid generalized anxiety
disorder**.

You are NOT writing the answer. You never see a draft — on purpose. A model judging text
it just produced says "sufficient" almost every time; judging the evidence itself is the
only version of this gate that carries information.

## The question

«question»

## The evidence retrieved for it

«evidence»

## What each field means

`sufficient` — is this evidence enough to write a four-to-six sentence research note
that a psychiatrist will read **with the citations in front of them**? That is the whole
question. It is *not* whether the topic is settled, and it is not whether you would
prescribe from it.

`addresses_question_directly` — does this evidence answer *this* question, or an
adjacent one? Evidence about pure GAD does not answer a question about GAD comorbid with
bipolar I: the population is the whole point of this project. Be strict here; it is the
field that separates a real answer from a plausible one.

`n_independent_sources` — how many distinct PMIDs actually bear on the question. Two
statements extracted from the same paper are one source.

`sources_agree` — do they point the same way? Disagreement does not block an answer; it
lowers its confidence and must be reported.

`missing` — name the specific thing that would close the gap, so the next search round
has somewhere to go. "More evidence" is useless; a named, missing comparison in the
target population is actionable.

## What is enough, and what is merely unproven

Refuse for **absence**, never for weakness.

Absence is: the evidence is about a different intervention, or a different outcome, or a
different population; there is only one source; the question asks for a number the
evidence does not contain.

Weakness is not refusal. "No head-to-head trial", "small sample", "no meta-analysis",
"relative efficacy unknown", "generalisability unclear" — every one of those belongs in
`missing`, and every one of them is compatible with `sufficient: true`. They describe
the literature, and the note is allowed to say so.

This matters here more than it would elsewhere, because the scarcity is **structural**:
GAD trials exclude bipolar patients, and bipolar trials treat anxiety as a secondary
outcome. No amount of extra searching produces the head-to-head that does not exist. A
bar that requires it refuses everything forever, and the reader gets nothing instead of
getting an honest note with its limits named.

`blocked_reason` — fill this ONLY when more searching cannot help. A question that needs
the user's values, or the patient's history, or that rests on an irreconcilable conflict
in the literature, will not be resolved by another round: say so and it goes to a human.
Leaving this empty when searching is hopeless burns the whole round budget.
