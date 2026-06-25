# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_jax_vram_chunk.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Turn 7 — (c) ordem do env de pré-alocação de VRAM no subprocesso +        ║
# ║           (d) jax_chunk_size_models FIXO no path JAX do SM                 ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-24                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    (c) Garante que o pool JAX persistente fixa XLA_PYTHON_CLIENT_PREALLOCATE║
# ║        no FILHO (via initializer, antes de qualquer import) e o INVARIANTE  ║
# ║        de que NENHUM ``import jax`` de topo existe fora de simulation/_jax/  ║
# ║        — base da corretude da ordem do env. O initializer é jax-free (TLS). ║
# ║    (d) Garante que o SM passa jax_chunk_size_models=64 a simulate_batch nos  ║
# ║        2 call-sites. A INVARIÂNCIA física do chunk (bit-paridade <1e-13) já  ║
# ║        é coberta por test_simulation_jax_o4_batched_bucketed.py             ║
# ║        ::test_o4_chunk_size_models_invariante — aqui só o cabeamento do SM.  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Turn 7 (c)+(d) — ordem do env de VRAM no subprocesso + chunk JAX fixo do SM."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[1]


# ════════════════════════════════════════════════════════════════════════════
# (c) — initializer do pool persistente: ordem do env de VRAM, jax-free, invariante
# ════════════════════════════════════════════════════════════════════════════
def test_jax_pool_initializer_is_jax_free_and_sets_preallocate(monkeypatch):
    """O initializer só mexe em os.environ (setdefault PREALLOCATE=false) e NÃO importa jax."""
    import sys

    # Vive em sim_request (módulo Qt-FREE), NÃO em base.py — p/ o filho do spawn
    # resolvê-lo sem importar base.py (que puxa Qt). Ver _acquire_jax_pool.
    from geosteering_ai.gui.services.sim_request import _jax_pool_initializer

    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    # garante que medimos um import NOVO de jax durante a chamada
    had_jax = "jax" in sys.modules
    _jax_pool_initializer()
    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"
    if not had_jax:
        assert "jax" not in sys.modules, "initializer NÃO pode importar jax (TLS-safe)"


def test_jax_pool_initializer_respects_user_override(monkeypatch):
    """setdefault: se o usuário já exportou PREALLOCATE, o initializer NÃO sobrescreve."""
    from geosteering_ai.gui.services.sim_request import _jax_pool_initializer

    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    _jax_pool_initializer()
    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "true"


def test_initializer_lives_in_qt_free_module():
    """Revisão Turn 7 (op c): o initializer DEVE viver num módulo Qt-FREE (sim_request),
    p/ o filho do spawn NÃO importar base.py (que puxa Qt) ao resolver a referência."""
    from geosteering_ai.gui.services.sim_request import _jax_pool_initializer

    assert _jax_pool_initializer.__module__ == "geosteering_ai.gui.services.sim_request"


def test_resolving_initializer_does_not_pull_qt():
    """Subprocesso LIMPO: importar sim_request (onde vive o initializer) NÃO puxa Qt
    nem base.py → o filho do spawn fica Qt-free (revisão Turn 7 op c). Espelha o padrão
    de test_sm_stochastic::test_stochastic_geology_importable_without_qt."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from geosteering_ai.gui.services.sim_request import _jax_pool_initializer  # noqa\n"
        "qt = [m for m in sys.modules if m.split('.')[0] in ('PyQt6', 'PySide6')]\n"
        "assert not qt, f'sim_request puxou Qt: {qt}'\n"
        "assert 'geosteering_ai.gui.services.base' not in sys.modules, 'puxou base.py'\n"
        "print('QT_FREE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(_REPO)
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "QT_FREE_OK" in proc.stdout


def test_persistent_pool_child_has_preallocate_false(monkeypatch):
    """Integração: o FILHO do pool persistente roda o initializer → PREALLOCATE=false.

    Spawn real (CPU-safe; ``os.getenv`` é picklável e NÃO importa jax — só lê o env do
    filho). delenv garante que o filho HERDA o env limpo e o initializer é quem fixa o
    valor. Libera o pool ao fim p/ não vazar o subprocesso.
    """
    from geosteering_ai.gui.services.base import _acquire_jax_pool, release_jax_pool

    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    release_jax_pool()  # garante pool NOVO (spawn herda o env limpo)
    try:
        pool = _acquire_jax_pool()
        # os.getenv é função de módulo (picklável p/ spawn); roda no FILHO.
        val = pool.submit(os.getenv, "XLA_PYTHON_CLIENT_PREALLOCATE").result(timeout=60)
        assert val == "false", f"esperava 'false' no filho, obtido {val!r}"
    finally:
        release_jax_pool()


def test_no_toplevel_jax_import_outside_jax_pkg():
    """INVARIANTE (base da ordem do env): nenhum ``import jax`` de COLUNA-0 existe fora
    de geosteering_ai/simulation/_jax/. Se quebrar, o setdefault de _setup_xla_environment
    pode rodar TARDE (XLA já capturou PREALLOCATE=true). Os imports jax legítimos são
    LAZY (indentados, dentro de funções)."""
    pkg = _REPO / "geosteering_ai"
    allowed = pkg / "simulation" / "_jax"
    pat = re.compile(r"^(?:import|from)\s+jax(?:\b|\.)")
    offenders: list[str] = []
    for py in pkg.rglob("*.py"):
        if allowed in py.parents:
            continue
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            # coluna-0 = sem indentação (imports de topo de módulo)
            if line[:1] not in (" ", "\t") and pat.match(line):
                offenders.append(f"{py.relative_to(_REPO)}:{n}: {line}")
    assert not offenders, (
        "import jax de topo fora de _jax/ quebra a ordem do env:\n"
        + "\n".join(offenders)
    )


# ════════════════════════════════════════════════════════════════════════════
# (d) — jax_chunk_size_models FIXO no SimRequest + cabeado nos 2 call-sites
# ════════════════════════════════════════════════════════════════════════════
def test_simrequest_jax_chunk_default_is_fixed_64():
    """SimRequest carrega o chunk FIXO (64), alinhado ao _SM_CANON_N_MODELS do warmup."""
    from geosteering_ai.gui.services.sim_request import (
        _SM_JAX_CHUNK_SIZE_MODELS,
        SimRequest,
    )

    assert _SM_JAX_CHUNK_SIZE_MODELS == 64
    assert SimRequest().jax_chunk_size_models == 64


def _fake_simulate_batch(captured: dict):
    def _fn(rho_h, rho_v, esp, positions_z, **kw):  # noqa: ANN001
        captured.update(kw)
        n_models = int(np.asarray(rho_h).shape[0])
        n_pos = int(np.asarray(positions_z).shape[0])
        n_f = len(kw["frequencies_hz"])
        n_tr = len(kw["tr_spacings_m"])
        n_ang = len(kw["dip_degs"])
        h6 = np.zeros((n_models, n_tr, n_ang, n_pos, n_f, 9), dtype=np.complex128)
        return h6, {"backend": "numba", "n_geometry_groups": 1, "elapsed_s": 0.0}

    return _fn


def test_sm_fixed_path_passes_jax_chunk(monkeypatch):
    """Ramo geologia FIXA: _run_simulation passa jax_chunk_size_models=64 a simulate_batch."""
    import geosteering_ai.simulation as sim
    from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

    captured: dict[str, Any] = {}
    monkeypatch.setattr(sim, "simulate_batch", _fake_simulate_batch(captured))
    _run_simulation(
        SimRequest(
            geology_mode="fixed", n_models=2, backend="numba", frequencies_hz=(20000.0,)
        )
    )
    assert captured.get("jax_chunk_size_models") == 64


def test_sm_grouped_path_passes_jax_chunk(monkeypatch):
    """Ramo ESTOCÁSTICO (agrupado): cada simulate_batch por grupo recebe o chunk=64."""
    import geosteering_ai.simulation as sim
    from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

    captured: dict[str, Any] = {}
    monkeypatch.setattr(sim, "simulate_batch", _fake_simulate_batch(captured))
    _run_simulation(
        SimRequest(
            geology_mode="stochastic",
            n_layers_fixed=5,
            n_models=4,
            rng_seed=1,
            backend="numba",
            frequencies_hz=(20000.0,),
        )
    )
    assert captured.get("jax_chunk_size_models") == 64
