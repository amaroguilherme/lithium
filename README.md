# lithium

An autonomous research agent that reads, forms hypotheses, argues against them, and asks
you when it gets stuck.

It is also, and mainly, a proof of concept for a harder question: **how does a model
accumulate knowledge over months without accumulating delusion?** Retrieval fades, memory
drifts, and a system that learns from its own output eventually teaches itself its own
guesses. Every mechanism here exists to make that failure structurally impossible rather
than merely discouraged.

---

## The two ideas

**1. Knowledge is one brain; the focus is a lens on it.**

A claim's *strength* is universal: study design, sample, effect. Its *relevance* is not:
it is a relation between what was studied and what you are asking about. Most systems
collapse these into one number and lose the ability to say "strong result, wrong
population."

Here they stay separate. Weight is `grade × directness × confidence`, multiplicative, with
a single origin. `directness` is not a column on the claim; it is an **edge**
that names the target it was judged against. Change focus and the same claim carries a
different judgement, while the old one stays intact and reconstructible.

The practical consequence: a cohort study in exactly your population outweighs a
meta-analysis in an adjacent one. That is the judgement a careful human makes, and it is
what makes the system useful where direct evidence barely exists.

**2. Everything the model believes must be traceable to something it did not invent.**

Two gates stand between text and belief. The first requires a verbatim quote present in the
source. The second asks, in isolation, whether that quote actually supports the claim,
never shown the draft it is judging. Both fail closed.

This is enforced in more places than feels necessary, because the failure is silent. A
pattern the system derives about its own work must cite the specific records it was shown.
Something read on the open web can *point at* a paper, but its prose can never become
evidence. The identifier crosses, the text does not. What you tell it in conversation
annotates evidence and never reorders it.

---

## What it actually does

Runs a daemon that harvests literature, extracts claims behind both gates, generates its
own research questions, and tries to answer them — escalating to you only when it cannot.
In parallel it runs a speculative track: proposes mechanistic hypotheses, criticises them
adversarially, turns the survivors into targeted searches, and re-grounds the chains as the
corpus grows.

It reflects on its own work and writes durable lessons: dead ends, sterile query shapes,
patterns across claims. It goes out to the open web, reads what it finds, and
reports back; nothing it read becomes evidence until you say so.

Two consent regimes, deliberately different: what it learns about **its own work** it
records on its own; anything about **you**, or anything read from the open web, waits for
your confirmation.

---

## Running it

Needs `llama.cpp` and roughly 9 GB of local weights.

```bash
uv venv && uv pip install -e ".[dev]"
brew install llama.cpp                    # macOS; Windows: CUDA build from ggml-org
lithium init                              # database, evidence weights, seed focus
lithium serve                             # llama-servers + workers; Ctrl-C stops
```

```bash
lithium status                    # counts, queue, focus, recon budget
lithium ask "..."                 # inject a question at top priority
lithium questions                 # what needs you, and what the machine is chasing
lithium answer 42 "..."           # answer and free the queue slot
lithium chat                      # conversation; it sees the corpus and asks before remembering

lithium focus                     # list; --use / --new / --show / --relens
lithium sources                   # the source registry; --approve / --spec / --revoke
lithium discoveries               # what the web scout is waiting on; --approve / --reject
lithium mode off                  # stop autonomous research; on-demand still works
```

`lithium --help` lists the rest.

---

## Design notes worth knowing before you touch it

**Prompts fail before the POST, not after.** A prompt that will not fit the window raises
locally. Otherwise llama-server returns 400, the client correctly does not retry a 4xx, and
the task dies in the dead-letter queue, which is how the speculative track once went silent
as the corpus grew.

**Structured output is grammar-constrained** from Pydantic schemas, so the model cannot emit
a value outside an enum rather than being asked politely not to. This also means anything in
a schema docstring reaches the model: a channel no golden-render test can see.

**Research mode is one switch, and memory ignores it.** Turning autonomous research off
still serves what you ask for; learning from conversation is conversation, not research.

**A focus scaffold refuses to load until reviewed.** `focus --new` copies the reference
profile, which is deliberate — a blank profile teaches nothing — but filling in only the
target would produce prompts that announce one subject and reason about another.

**Every lock carries the mutation that must kill it.** Two defect classes recur in this
codebase and the tests are shaped against them: expectations derived from the source they
police, and wiring no test exercises. The second happened six times.

---
