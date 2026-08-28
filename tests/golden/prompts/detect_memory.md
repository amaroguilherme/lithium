Decide whether the user's last message contains something **durable** worth keeping.

You are not summarising the conversation. You are looking for a single fact about the
*user or the case* that will still matter in three months.

## Remember

- `preference` — how they want the work done, or what they value. "Prefiro entender o
mecanismo antes de ver a evidência." - `constraint` — a hard limit. "Não considerar nada
que exija monitoramento sérico." - `context` — a fact about the case the literature
cannot supply. "Já tentou lamotrigina e teve rash." - `fact` — anything else durable and
specific about them or the project's scope.

## Do not remember

- questions, however interesting - speculation, hypotheses, thinking out loud - anything
the system already knows from the literature - passing reactions ("interessante", "faz
sentido") - restatements of something already in the memory list below - transient state
— what they are doing today, what they just read

The bar is: **would recalling this in three months change how you work?** If you're
unsure, answer `false`. A missed memory is recoverable — they can always tell you again.
A memory list padded with noise is not: it degrades every future retrieval, and it makes
the confirmation prompt annoying enough that the user stops reading it.

Being conservative here is what keeps the feature usable.

## Fields

`text` — third person, self-contained, understandable without the surrounding
conversation. "O usuário evita fármacos que exijam monitoramento sérico" — not "ele
disse que não quer isso".

`kind` — one of `preference`, `context`, `constraint`, `fact`.

## Already remembered — do not propose again

«existing»

## The user said

«message»
