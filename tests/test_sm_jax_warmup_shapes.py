# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_jax_warmup_shapes.py                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Warmup JAX GPU config-aware (shape-matching) — specs determinísticos      ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garante que ``build_warmup_specs`` reproduz EXATAMENTE a forma que a 1ª ║
# ║    sim real do SM dispara (esp templates determinísticos == colapso de     ║
# ║    produção; positions_z == compute_n_pos; n_models == chunk balanceado) — ║
# ║    a base do cache-hit. Tudo jax-FREE (roda no CI CPU sem jax). A prova     ║
# ║    empírica de cache-hit (warm → 0 novos na sim) é validada no A6000.       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do warmup JAX config-aware (shape-matching) — jax-free, determinístico."""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _req(**over):
    from geosteering_ai.gui.services.sim_request import SimRequest

    base = dict(
        frequencies_hz=(20000.0,),
        tr_spacings_m=(1.0,),
        dip_degs=(0.0,),
        h1=10.0,
        tj=120.0,
        n_models=1000,
        geology_mode="stochastic",
        n_layers_fixed=15,
        n_layers_min=3,
        n_layers_max=32,
        min_thickness=0.5,
        rho_h_min=1.0,
        rho_h_max=1800.0,
        anisotropic=True,
        lambda_min=1.0,
        lambda_max=2**0.5,
        generator="sobol",
        rng_seed=None,
        backend="jax",
        geometry_diversity="auto",
        n_geometries=None,
        hankel_filter="werthmuller_201pt",
    )
    return SimRequest(**{**base, **over})


# ════════════════════════════════════════════════════════════════════════════
# _balanced_chunk_dim (réplica jax-free) == _balanced_chunk_slices (kernel)
# ════════════════════════════════════════════════════════════════════════════
def test_balanced_chunk_dim_matches_slices():
    """A réplica jax-free ``_balanced_chunk_dim`` casa o tamanho da 1ª fatia real
    de ``_balanced_chunk_slices`` — guard contra drift do duplicado TLS-safe."""
    from geosteering_ai.gui.services.sim_request import _balanced_chunk_dim
    from geosteering_ai.simulation._jax.multi_forward import _balanced_chunk_slices

    for n in (1, 2, 17, 69, 70, 250, 256, 257, 500, 600, 1000, 2000):
        for cap in (None, 32, 64, 256, 512):
            sl = _balanced_chunk_slices(n, cap)
            assert _balanced_chunk_dim(n, cap) == (sl[0].stop - sl[0].start), (n, cap)


def test_warmup_spec_gpu_threshold_matches_dispatch():
    """O limiar local espelha ``dispatch._N_MODELS_GPU_THRESHOLD`` (não aquecer grupos
    que a produção roda no Numba)."""
    from geosteering_ai.gui.services.jax_warmup_spec import _GPU_THRESHOLD
    from geosteering_ai.simulation.dispatch import _N_MODELS_GPU_THRESHOLD

    assert _GPU_THRESHOLD == _N_MODELS_GPU_THRESHOLD


# ════════════════════════════════════════════════════════════════════════════
# build_warmup_specs — SHAPE-MATCHING com a produção (a base do cache-hit)
# ════════════════════════════════════════════════════════════════════════════
def test_specs_match_production_geometry_fixed():
    """FIXO: esp dos specs == templates determinísticos do colapso de PRODUÇÃO;
    positions_z == _compute_positions_z; n_models == chunk balanceado; dtype c128."""
    from geosteering_ai.gui.services.jax_warmup_spec import build_warmup_specs
    from geosteering_ai.gui.services.sim_request import (
        _collapse_geometry_to_templates,
        _compute_positions_z,
        _genconfig_from_request,
        _generate_stochastic_models,
        _stable_geometry_seed,
    )

    req = _req()
    specs = build_warmup_specs(req)
    assert specs, "config jax fixa deve gerar specs"

    # esp dos specs == templates determinísticos que a produção colapsaria
    cfg = _genconfig_from_request(req)
    seed = _stable_geometry_seed(cfg)
    models = _generate_stochastic_models(req)
    models = _collapse_geometry_to_templates(
        models, req.n_geometries, gen_cfg=cfg, stable_seed=seed
    )
    prod = sorted({tuple(np.round(m["thicknesses"], 9)) for m in models})
    got = sorted({tuple(np.round(s["esp_template"], 9)) for s in specs})
    assert got == prod  # geometria bit-idêntica à produção

    pz = _compute_positions_z(req)
    for s in specs:
        assert np.array_equal(s["positions_z"], pz)  # mesmo grid
        assert s["n_layers"] == 15
        assert s["n_models"] == 250  # grupo 1000 / K4 = 250 ≤ cap → dim-líder 250
        assert s["complex_dtype"] == "complex128"
        assert len(s["esp_template"]) == 13  # n_layers - 2


def test_specs_empty_for_numba_backend():
    """backend=numba (não compila XLA) → nenhum spec."""
    from geosteering_ai.gui.services.jax_warmup_spec import build_warmup_specs

    assert build_warmup_specs(_req(backend="numba")) == []


def test_specs_ragged_modal_coverage():
    """RAGGED: cobertura MODAL (≤ max_n_layers valores de n_layers); grupos < 32 (que a
    produção roda no Numba) são filtrados."""
    from geosteering_ai.gui.services.jax_warmup_spec import build_warmup_specs

    # 2000 modelos / faixa 3..11 (8 valores) ⇒ ~250/grupo ≥ 32 → batelável.
    req = _req(n_layers_fixed=None, n_layers_min=3, n_layers_max=11, n_models=2000)
    specs = build_warmup_specs(req, max_n_layers=2)
    n_layers_set = {s["n_layers"] for s in specs}
    assert 1 <= len(n_layers_set) <= 2  # modal
    assert all(3 <= nl < 11 for nl in n_layers_set)

    # n_models pequeno → grupos ragged < 32 → produção usa Numba → 0 specs.
    assert build_warmup_specs(_req(n_layers_fixed=None, n_models=10)) == []


def test_specs_manual_mode_single_geometry():
    """MANUAL: 1 spec com a geometria manual (replicada, 1 grupo)."""
    from geosteering_ai.gui.services.jax_warmup_spec import build_warmup_specs

    req = _req(
        geology_mode="manual",
        manual_n_layers=5,
        manual_rho_h=(1.0, 10.0, 100.0, 50.0, 5.0),
        manual_rho_v=(1.0, 10.0, 100.0, 50.0, 5.0),
        manual_thicknesses=(3.0, 4.0, 5.0),
        n_models=100,
    )
    specs = build_warmup_specs(req)
    assert len(specs) == 1
    assert specs[0]["n_layers"] == 5
    assert [round(x, 6) for x in specs[0]["esp_template"]] == [3.0, 4.0, 5.0]


# ════════════════════════════════════════════════════════════════════════════
# TLS-safety — o construtor de specs roda na GUI e NÃO pode importar jax
# ════════════════════════════════════════════════════════════════════════════
def test_build_warmup_specs_does_not_import_simulator_jax():
    """``build_warmup_specs`` roda no processo da GUI → NÃO pode importar o ``_jax`` do
    SIMULADOR (que dispara init CUDA/XLA no processo da GUI e conflita com o worker spawn).

    Nota: o MÓDULO ``jax`` pode estar presente via o probe de backend do Keras 3 (benigno,
    não inicializa o simulador) — por isso o invariante testável é NÃO puxar
    ``geosteering_ai.simulation._jax``, não a ausência de ``jax``."""
    import subprocess

    code = (
        "import sys\n"
        "from geosteering_ai.gui.services.jax_warmup_spec import build_warmup_specs\n"
        "from geosteering_ai.gui.services.sim_request import SimRequest\n"
        "build_warmup_specs(SimRequest(backend='jax', n_layers_fixed=15, n_models=1000))\n"
        "bad = [m for m in sys.modules if m.startswith('geosteering_ai.simulation._jax')]\n"
        "assert not bad, f'build_warmup_specs puxou o _jax do simulador (viola TLS): {bad}'\n"
        "print('TLS_OK')\n"
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=repo
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "TLS_OK" in proc.stdout


# ════════════════════════════════════════════════════════════════════════════
# warmup_jax_shapes — skip gracioso sem jax + esp_template no warmup_jax_simulator
# ════════════════════════════════════════════════════════════════════════════
def test_warmup_jax_shapes_graceful_empty_and_no_jax():
    """``warmup_jax_shapes([])`` é seguro com/sem jax (no-op); sem jax → skip gracioso."""
    from geosteering_ai.simulation._jax import HAS_JAX
    from geosteering_ai.simulation._jax.warmup import warmup_jax_shapes

    r = warmup_jax_shapes(
        []
    )  # lista vazia: nunca compila → seguro em qualquer ambiente
    assert r["specs_total"] == 0
    assert r["programs_warmed"] == 0
    if not HAS_JAX:
        assert r["skipped"] is True and r["reason"] == "jax_absent"


def test_warmup_simulator_esp_template_length_guard():
    """``esp_template`` com comprimento != n_layers-2 levanta ValueError (shape inválido)."""
    from geosteering_ai.simulation._jax import HAS_JAX
    from geosteering_ai.simulation._jax.warmup import warmup_jax_simulator

    if not HAS_JAX:
        import pytest

        pytest.skip("warmup_jax_simulator exige jax p/ exercitar o guard pós-skip")
    import pytest

    with pytest.raises(ValueError, match="esp_template"):
        warmup_jax_simulator(
            n_layers=5, esp_template=[1.0, 2.0], n_models=1
        )  # precisa 3
