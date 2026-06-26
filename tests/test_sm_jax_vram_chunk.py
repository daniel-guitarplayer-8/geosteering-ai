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
# (d) — chunk ADAPTATIVO por grupo (Turn 8): cap VRAM; grupo pequeno → None (sem fatia)
# ════════════════════════════════════════════════════════════════════════════
def test_simrequest_jax_chunk_cap_default_256():
    """O cap default é 256 (VRAM-safe; acima do tamanho típico de grupo ragged)."""
    from geosteering_ai.gui.services.sim_request import _SM_JAX_CHUNK_CAP, SimRequest

    assert _SM_JAX_CHUNK_CAP == 256
    assert SimRequest().jax_chunk_size_models == 256


def test_resolve_group_chunk_adaptive():
    """grupo ≤ cap → None (roda inteiro, 1 forma XLA); > cap → cap; cap=None → None."""
    from geosteering_ai.gui.services.sim_request import _resolve_group_chunk

    assert _resolve_group_chunk(70, 256) is None  # ragged típico → sem fatia
    assert _resolve_group_chunk(256, 256) is None  # fronteira == cap → inteiro
    assert _resolve_group_chunk(2000, 256) == 256  # grupo fixo grande → fatiado
    assert _resolve_group_chunk(2000, None) is None  # cap None → nunca fatia
    assert _resolve_group_chunk(1, 256) is None


def _fake_simulate_batch(captured: dict):
    def _fn(rho_h, rho_v, esp, positions_z, **kw):  # noqa: ANN001
        captured.setdefault("chunks", []).append(kw.get("jax_chunk_size_models"))
        captured.update(kw)
        n_models = int(np.asarray(rho_h).shape[0])
        n_pos = int(np.asarray(positions_z).shape[0])
        n_f = len(kw["frequencies_hz"])
        n_tr = len(kw["tr_spacings_m"])
        n_ang = len(kw["dip_degs"])
        h6 = np.zeros((n_models, n_tr, n_ang, n_pos, n_f, 9), dtype=np.complex128)
        return h6, {"backend": "numba", "n_geometry_groups": 1, "elapsed_s": 0.0}

    return _fn


def test_sm_small_group_passes_none(monkeypatch):
    """Grupo pequeno (≤ cap): _run_simulation passa chunk=None (roda inteiro, sem fatia-cauda)."""
    import geosteering_ai.simulation as sim
    from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

    captured: dict[str, Any] = {}
    monkeypatch.setattr(sim, "simulate_batch", _fake_simulate_batch(captured))
    _run_simulation(
        SimRequest(
            geology_mode="fixed", n_models=2, backend="numba", frequencies_hz=(20000.0,)
        )
    )
    assert captured.get("jax_chunk_size_models") is None  # 2 ≤ 256 → None


def test_sm_large_group_passes_cap(monkeypatch):
    """Grupo grande (> cap): passa o cap (256) p/ limitar a VRAM."""
    import geosteering_ai.simulation as sim
    from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

    captured: dict[str, Any] = {}
    monkeypatch.setattr(sim, "simulate_batch", _fake_simulate_batch(captured))
    _run_simulation(
        SimRequest(
            geology_mode="fixed",
            n_models=600,
            backend="numba",
            frequencies_hz=(20000.0,),
        )
    )
    assert captured.get("jax_chunk_size_models") == 256  # 600 > 256 → cap


def test_sm_grouped_path_chunk_per_group(monkeypatch):
    """Ramo ESTOCÁSTICO: cada grupo (n_layers) recebe o chunk resolvido p/ o SEU tamanho."""
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
    # 1 grupo de 4 modelos ≤ 256 → None (sem fatia-cauda)
    assert captured.get("chunks") == [None]


# ════════════════════════════════════════════════════════════════════════════
# #1 (Turn 8 parte 2) — GEOMETRIA DETERMINÍSTICA no colapso (cache XLA acerta no 2º run)
# Causa-raiz da regressão JAX-lento: rng_seed=None → esp aleatório por run → o cache de
# disco XLA ERRA → recompila ~216 programas (349 s). Fix: templates de geometria fixados
# por uma semente ESTÁVEL da config (esp idêntico entre runs), ρ/λ seguem aleatórios.
# ════════════════════════════════════════════════════════════════════════════
def test_stable_geometry_seed_deterministic_and_config_sensitive():
    """Mesma config → MESMA semente; campos de geometria distintos → semente distinta."""
    from geosteering_ai.gui.services.sim_request import (
        SimRequest,
        _genconfig_from_request,
        _stable_geometry_seed,
    )

    def seed(**kw: Any) -> int:
        base = dict(tj=120.0, min_thickness=0.5, n_layers_fixed=15)
        return _stable_geometry_seed(
            _genconfig_from_request(SimRequest(**{**base, **kw}))
        )

    assert seed() == seed()  # determinístico (mesma config)
    assert seed(min_thickness=0.5) != seed(min_thickness=1.5)  # sensível ao piso
    assert seed(tj=120.0) != seed(tj=100.0)  # sensível ao total_depth
    assert seed(n_layers_fixed=15) != seed(n_layers_fixed=20)  # sensível a n_layers


def test_stable_geometry_seed_stable_across_processes():
    """A semente usa zlib.crc32 (estável entre processos), NUNCA hash() (salgado por
    PYTHONHASHSEED) — senão o cache XLA de disco erraria a cada novo processo do worker.
    """
    import subprocess
    import sys

    code = (
        "from geosteering_ai.gui.services.sim_request import ("
        "SimRequest, _genconfig_from_request, _stable_geometry_seed)\n"
        "r = SimRequest(tj=120.0, min_thickness=0.5, n_layers_fixed=15, "
        "n_layers_min=3, n_layers_max=32)\n"
        "print(_stable_geometry_seed(_genconfig_from_request(r)))\n"
    )

    def seed_with_hashseed(hashseed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            env=env,
        )
        assert proc.returncode == 0, proc.stderr[-1500:]
        return proc.stdout.strip()

    # PYTHONHASHSEED diferente NÃO pode mudar a semente (prova: crc32, não hash()).
    assert seed_with_hashseed("0") == seed_with_hashseed("123456789")


def test_collapse_deterministic_geometry_stable_rho_random():
    """Colapso com gen_cfg+stable_seed: esp IDÊNTICO entre runs (rng_seed=None nos dois),
    porém ρ/λ DIFEREM → cache XLA acerta sem sacrificar a diversidade Monte-Carlo."""
    from geosteering_ai.gui.services.sim_request import (
        SimRequest,
        _collapse_geometry_to_templates,
        _genconfig_from_request,
        _generate_stochastic_models,
        _stable_geometry_seed,
    )

    def run() -> tuple:
        r = SimRequest(
            geology_mode="stochastic",
            n_layers_fixed=15,
            n_models=120,
            rng_seed=None,  # ρ/λ aleatório a cada run
            backend="jax",
            frequencies_hz=(20000.0,),
        )
        cfg = _genconfig_from_request(r)
        seed = _stable_geometry_seed(cfg)
        models = _generate_stochastic_models(r)
        models = _collapse_geometry_to_templates(
            models, r.n_geometries, gen_cfg=cfg, stable_seed=seed
        )
        esp = [tuple(m["thicknesses"]) for m in models]
        rho = [tuple(m["rho_h"]) for m in models]
        return esp, rho

    esp_a, rho_a = run()
    esp_b, rho_b = run()
    assert esp_a == esp_b  # geometria estável entre runs → cache XLA acerta
    assert (
        rho_a != rho_b
    )  # ρ continua sorteado (rng_seed=None) → diversidade preservada
    assert len(set(esp_a)) <= 4  # ≤ K templates (cap=_TEMPLATE_GEOMETRIES_CAP)


def test_collapse_ragged_deterministic_per_group():
    """RAGGED (n_layers variável): os templates por (n_layers) são idênticos entre runs."""
    from geosteering_ai.gui.services.sim_request import (
        SimRequest,
        _collapse_geometry_to_templates,
        _genconfig_from_request,
        _generate_stochastic_models,
        _stable_geometry_seed,
    )

    def templates() -> set:
        r = SimRequest(
            geology_mode="stochastic",
            n_layers_fixed=None,
            n_layers_min=3,
            n_layers_max=10,
            n_models=300,
            rng_seed=None,
            backend="jax",
            frequencies_hz=(20000.0,),
        )
        cfg = _genconfig_from_request(r)
        seed = _stable_geometry_seed(cfg)
        models = _generate_stochastic_models(r)
        models = _collapse_geometry_to_templates(
            models, r.n_geometries, gen_cfg=cfg, stable_seed=seed
        )
        return {(int(m["n_layers"]), tuple(m["thicknesses"])) for m in models}

    assert templates() == templates()  # mesmo conjunto (n_layers, esp) entre runs


def test_collapse_backcompat_legacy_without_cfg():
    """Sem gen_cfg/stable_seed → comportamento LEGADO (primeiras K espessuras da
    população, round-robin). Garante callers diretos/testes existentes intocados."""
    from geosteering_ai.gui.services.sim_request import _collapse_geometry_to_templates

    # 6 modelos n_layers=5 (ncamint=3 espessuras), espessuras distintas por modelo.
    models = [
        {
            "n_layers": 5,
            "thicknesses": [float(i), float(i) + 1.0, float(i) + 2.0],
            "rho_h": [1.0] * 5,
        }
        for i in range(6)
    ]
    out = _collapse_geometry_to_templates([dict(m) for m in models], None)
    # k = min(4, 6//2) = 3 → templates = espessuras dos modelos 0,1,2 (round-robin).
    assert out[0]["thicknesses"] == [0.0, 1.0, 2.0]  # template 0
    assert out[3]["thicknesses"] == [0.0, 1.0, 2.0]  # local 3 % 3 == 0 → template 0
    assert out[4]["thicknesses"] == [1.0, 2.0, 3.0]  # local 4 % 3 == 1 → template 1


# ════════════════════════════════════════════════════════════════════════════
# #2 (Turn 8 parte 2) — CHUNK BALANCEADO no JAX core (1 forma XLA por bucket, sem cauda)
# [256, 244] = 2 formas = 2 compilações/bucket; [250, 250] = 1 forma = 1 compilação.
# Corta ~metade do custo a frio do grupo FIXO grande. VRAM ≤ cap. Paridade <1e-13 já é
# coberta por test_simulation_jax_o4_batched_bucketed::test_o4_chunk_size_models_invariante.
# ════════════════════════════════════════════════════════════════════════════
def test_balanced_chunk_slices_single_shape_common():
    """Casos de produção (subgrupos pares): UMA forma de fatia → 1 compilação XLA/bucket."""
    from geosteering_ai.simulation._jax.multi_forward import _balanced_chunk_slices

    for n, cap in [(500, 256), (2000, 256), (600, 256), (1000, 256)]:
        sl = _balanced_chunk_slices(n, cap)
        sizes = [s.stop - s.start for s in sl]
        assert len(set(sizes)) == 1, (n, cap, sizes)  # 1 forma XLA
        assert sum(sizes) == n  # cobre todos os modelos
        assert sl[0].start == 0 and sl[-1].stop == n  # cobre [0, n)
        assert all(  # fatias contíguas (sem buraco/sobreposição)
            sl[i].stop == sl[i + 1].start for i in range(len(sl) - 1)
        )
        assert max(sizes) <= cap  # VRAM ≤ cap


def test_balanced_chunk_slices_whole_when_fits():
    """chunk=None ou n ≤ cap → UMA fatia inteira (sem fatiamento)."""
    from geosteering_ai.simulation._jax.multi_forward import _balanced_chunk_slices

    assert _balanced_chunk_slices(70, 256) == [slice(0, 70)]
    assert _balanced_chunk_slices(256, 256) == [slice(0, 256)]  # fronteira == cap
    assert _balanced_chunk_slices(1000, None) == [slice(0, 1000)]  # None → inteiro


def test_balanced_chunk_slices_vram_bound_uneven():
    """Subgrupo ÍMPAR: no máx. 2 formas (diferem por ≤1), soma exata, max ≤ cap."""
    from geosteering_ai.simulation._jax.multi_forward import _balanced_chunk_slices

    sl = _balanced_chunk_slices(257, 256)
    sizes = [s.stop - s.start for s in sl]
    assert sum(sizes) == 257  # cobertura exata
    assert max(sizes) <= 256  # VRAM-safe (nunca pior que o cap)
    assert len(set(sizes)) <= 2  # fatias balanceadas (≤ 2 formas, diferem por ≤1)
    assert max(sizes) - min(sizes) <= 1
