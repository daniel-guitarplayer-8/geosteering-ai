# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_jax_cache_setup.py                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Backend JAX — setup do cache de compilação XLA             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-10                                                 ║
# ║  Status      : Produção (guarda da revisão adversarial)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Guarda do fallback do cache de compilação XLA (``_setup_xla_environment``).

Recuperado/endurecido no item 4 da triagem + correção da revisão adversarial:
um ``JAX_COMPILATION_CACHE_DIR`` inválido/não-gravável NÃO pode vazar para o
``jax.config.update`` — deve cair para ``~/.cache`` (ou ``$TMPDIR``), ou, se
nada for gravável, ter a env var REMOVIDA (nunca um caminho ruim).

Cada teste roda em SUBPROCESSO para isolar o estado global de ``os.environ`` e
o import único de ``jax`` (o setup roda no import do subpacote ``_jax``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

_PROBE = (
    "import geosteering_ai.simulation._jax as _j, os, sys; "
    "sys.stdout.write(repr(os.environ.get('JAX_COMPILATION_CACHE_DIR')))"
)


def _resolve_cache_dir(env_override: str | None) -> str | None:
    """Importa o subpacote _jax em subprocesso e devolve o cache dir resolvido."""
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    if env_override is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = env_override
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    assert proc.returncode == 0, f"import _jax falhou: {proc.stderr[:300]}"
    out = proc.stdout.strip()
    return None if out == "None" else out.strip("'\"")


def test_invalid_override_does_not_leak():
    """Override inválido NÃO vaza p/ o JAX — cai p/ um dir gravável ou é removido."""
    bad = "/nonexistent/cannot/possibly/write/here"
    resolved = _resolve_cache_dir(bad)
    assert resolved != bad, "caminho de override inválido VAZOU (regressão crítica)"
    # Resolveu p/ um dir gravável real, OU removeu a env var (None) — nunca o caminho ruim.
    assert resolved is None or os.path.isdir(
        resolved
    ), f"cache dir resolvido {resolved!r} não é um diretório válido"


def test_normal_import_sets_writable_cache():
    """Import normal (sem override) configura um cache dir gravável."""
    resolved = _resolve_cache_dir(None)
    assert resolved is not None, "import normal deveria configurar um cache dir"
    assert os.path.isdir(resolved), f"cache dir {resolved!r} não foi criado"
