"""Screen de segurança determinístico. Ver `screen.py` e `rules.py`.

**Nenhum vocabulário é reexportado.** `CONCEPTS` e `RULES` eram bindings de import time
e MEDI o que isso custava: `rules.RULES = ()` deixava `screen.RULES` com as 8 regras, e
trocar de foco sem reiniciar mantinha o screen do foco ANTIGO ativo e disparando. O
vocabulário agora vem de `ruleset_from_profile(perfil)` e viaja por parâmetro.
"""

from lithium.safety.rules import (
    ABSENT,
    Concept,
    Rule,
    RuleSet,
    concepts_implicated_by,
    concepts_in,
    ruleset_from_profile,
)
from lithium.safety.screen import NO_RULESET, Alert, Segment, render, screen

__all__ = [
    "ABSENT", "Alert", "Concept", "NO_RULESET", "Rule", "RuleSet", "Segment",
    "concepts_implicated_by", "concepts_in", "render", "ruleset_from_profile",
    "screen",
]
