"""Perfil de foco em disco — o vocabulário de um foco, versionável.

Fronteira, e ela é disjunta de propósito: o **TOML é dono do VOCABULÁRIO** (prosa,
estratégias, taxonomia, safety) e o **banco é dono da IDENTIDADE** (`focuses.id`,
`focuses.scale_id`) e do **HISTÓRICO DE JULGAMENTO** (`claim_directness`,
`scale_levels.weight`). Na maior parte dos campos não existe divergência possível
porque os conjuntos não se sobrepõem.

A ÚNICA sobreposição é `target`, e ali o BANCO VENCE: toda aresta em `claim_directness`
foi julgada contra o target que estava no banco no momento da escrita. O TOML só
escreve o target na CRIAÇÃO (`lithium focus new`). Editar depois não reescreve nada —
muda o `judgment_hash` e faz `lithium focus show` gritar.
"""

from lithium.focus.profile import (
    PROMPTS_WITH_PROFILE_BLOCKS,
    Directness4,
    FocusProfile,
    ProfileError,
    StandingRisk,
    load_profile,
    profile_dir,
)
from lithium.focus.resolve import (
    NoActiveFocus,
    active_profile,
    load_profile_cached,
    profile_for_slug,
)

__all__ = [
    "PROMPTS_WITH_PROFILE_BLOCKS",
    "Directness4",
    "FocusProfile",
    "NoActiveFocus",
    "ProfileError",
    "StandingRisk",
    "active_profile",
    "load_profile",
    "load_profile_cached",
    "profile_dir",
    "profile_for_slug",
]
