"""Resolver o perfil do foco — UMA vez, no topo de cada operação.

Vive num módulo próprio, e não em `pipeline/`, porque `pipeline/__init__.py` importa
oito submódulos: qualquer coisa que leia o banco a partir dali viraria
`pipeline -> db -> pipeline` em import time. Aqui a dependência é só
`db -> focus -> types`, que é acíclica.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from lithium.focus.profile import FocusProfile, load_profile, profile_dir


class NoActiveFocus(RuntimeError):
    """Nenhum foco ativo, e a operação julgaria contra um alvo desconhecido.

    Fail-closed na LEITURA, fail-LOUD na ESCRITA — a assimetria é deliberada. MEDIDO o
    que acontece sem ela na extração: `_persist` grava a claim (autocommit,
    `isolation_level=None`), o INSERT em `claim_directness` levanta `NOT NULL constraint
    failed: focus_id`, e a claim fica COMMITADA com `scale_id` NULL e sem aresta **para
    sempre** — restaurar o foco não a recupera. Cada retry deixa mais uma órfã.
    """


@lru_cache(maxsize=8)
def _cached(directory: str, mtime_key: tuple) -> FocusProfile:
    return load_profile(Path(directory))


def _stamp(directory: Path) -> tuple:
    """Assinatura de mtime dos 4 arquivos, para o cache soltar quando o usuário edita.

    Um perfil é um arquivo que o usuário edita à mão enquanto o daemon roda; um cache
    puro por caminho o congelaria até o reinício, e o usuário concluiria que a edição
    não fez nada. `focus.toml` primeiro porque é quem decide se existe `safety.toml`.
    """
    out = []
    for name in ("focus.toml", "strategies.toml", "taxonomy.toml", "safety.toml"):
        path = directory / name
        out.append(path.stat().st_mtime_ns if path.is_file() else 0)
    return tuple(out)


def load_profile_cached(directory: Path) -> FocusProfile:
    directory = Path(directory)
    return _cached(str(directory), _stamp(directory) if directory.is_dir() else ())


def profile_for_slug(focuses_dir: Path, slug: str) -> FocusProfile:
    return load_profile_cached(profile_dir(Path(focuses_dir), slug))


def active_profile(store, focuses_dir: Path) -> tuple:
    """`(row_do_foco_ativo, perfil)`, resolvidos JUNTOS e UMA vez.

    Devolver os dois juntos não é conveniência: a linha do banco carrega a IDENTIDADE
    (`id`, `scale_id`, `target` autoritativo) e o TOML carrega o VOCABULÁRIO. Resolver
    cada metade num ponto diferente da mesma operação é exatamente o defeito que faz um
    julgamento ser feito contra o alvo de um foco e gravado como aresta de outro.
    """
    focus = store.active_focus()
    if focus is None:
        raise NoActiveFocus(
            "nenhum foco ativo: a operação julgaria contra um alvo desconhecido. "
            "Escolha um com `lithium focus --use <slug>`."
        )
    return focus, profile_for_slug(focuses_dir, focus["slug"])
