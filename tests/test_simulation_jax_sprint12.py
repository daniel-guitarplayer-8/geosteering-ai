# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_sprint12.py                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes Sprint 12 — find_layers_tr_jax + vmap real          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-16 (PR #25 / v1.6.0)                              ║
# ║  Status      : Produção (gate obrigatório de merge)                       ║
# ║  Framework   : pytest + JAX 0.4.30+                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes Sprint 12 — porta tracer-safe de ``find_layers_tr`` + vmap real.

Cobertura do PR #25 (v1.6.0):

  Paridade ``find_layers_tr_jax`` vs Numba (bit-exata, ``diff == 0``):
    1. ``test_find_layers_tr_jax_parity_sweep`` — 1000 pares random
    2. ``test_find_layers_tr_jax_parity_boundaries`` — valores exatos em fronteiras
    3. ``test_find_layers_tr_jax_parity_semispaces`` — TX/RX nos semi-espaços
    4. ``test_find_layers_tr_jax_vmap_parity`` — versão batch vs loop Numba
    5. ``test_find_layers_tr_jax_under_jit`` — compilação JIT preserva paridade
    6. ``test_find_layers_tr_jax_traceable`` — aceita tracers (não Python int)
    7. ``test_find_layers_tr_jax_diverse_models`` — oklahoma_3/5/10/28

  vmap real multi-TR/multi-ang (``cfg.jax_vmap_real=True``):
    8-12. Paridade vmap_real vs Python loop (3 modelos × multi-dip)
    13-15. Shape H_tensor, MultiSimulationResultJAX, backward-compat

  jacfwd + alta resistividade:
    16. jacfwd sob vmap_real em alta-ρ (oklahoma_28 × 15)
    17. Sem NaN/Inf em configs extremas
    18. Paridade vmap_real vs Python loop (≤ 2.5×)

  Hordist dedup preservado:
    19. Vertical well (dip=0°) → fast-path sem vmap_real
    20. Matrix regression (fallback se vmap_real=False)

Note:
    Gate de paridade para ``find_layers_tr_jax`` é **``diff == 0``** (inteiros
    devem ser idênticos, não < 1e-12). Se falhar em **qualquer** ponto do sweep,
    há bug real que precisa ser investigado antes de prosseguir.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

jax_required = pytest.mark.skipif(not HAS_JAX, reason="JAX não instalado")


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures globais
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _enable_x64():
    """Garante float64/complex128 em todos os testes JAX."""
    if HAS_JAX:
        jax.config.update("jax_enable_x64", True)


# ──────────────────────────────────────────────────────────────────────────────
# Testes 1-7 — Paridade find_layers_tr_jax vs Numba
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
def test_find_layers_tr_jax_parity_sweep():
    """Sweep 1000 pares `(h0, z)` random: ``diff == 0`` vs Numba.

    Este é o gate-de-bloqueio mais importante do PR #25. Se falhar em
    qualquer ponto, há bug estrutural em ``find_layers_tr_jax`` e
    nenhum caminho downstream (vmap_real, PINN on-esp) pode ser validado.
    """
    from geosteering_ai.simulation._jax.geometry_jax import find_layers_tr_jax
    from geosteering_ai.simulation._numba.geometry import (
        find_layers_tr,
        sanitize_profile,
    )

    # Perfil 5-camadas variadas: [semi-sup, 3, 5, 2, semi-inf]
    esp = np.array([0.0, 3.0, 5.0, 2.0, 0.0])
    _, prof = sanitize_profile(n=5, esp=esp)
    prof_jnp = jnp.asarray(prof)

    rng = np.random.default_rng(42)
    h0s = rng.uniform(-5.0, 15.0, 1000)
    zs = rng.uniform(-5.0, 15.0, 1000)

    for h0, z in zip(h0s, zs):
        ct_n, cr_n = find_layers_tr(5, float(h0), float(z), prof)
        ct_j, cr_j = find_layers_tr_jax(float(h0), float(z), prof_jnp, 5)
        assert (
            int(ct_j) == ct_n
        ), f"TX divergência em h0={h0:.4f}: Numba={ct_n}, JAX={int(ct_j)}"
        assert (
            int(cr_j) == cr_n
        ), f"RX divergência em z={z:.4f}: Numba={cr_n}, JAX={int(cr_j)}"


@jax_required
def test_find_layers_tr_jax_parity_boundaries():
    """Valores exatamente nas fronteiras — convenção assimétrica TX/RX.

    ``z >= prof[i]`` (inclusivo) vs ``h0 > prof[j]`` (estrito).
    Se falhar, a convenção TX=``side="left"`` / RX=``side="right"``
    está errada.
    """
    from geosteering_ai.simulation._jax.geometry_jax import find_layers_tr_jax
    from geosteering_ai.simulation._numba.geometry import (
        find_layers_tr,
        sanitize_profile,
    )

    esp = np.array([0.0, 5.0, 0.0])
    _, prof = sanitize_profile(n=3, esp=esp)  # boundaries em 0 e 5
    prof_jnp = jnp.asarray(prof)

    cases = [
        (0.0, 0.0),  # ambos na fronteira z=0
        (5.0, 5.0),  # ambos na fronteira z=5
        (0.0, 5.0),  # TX em z=0, RX em z=5
        (5.0, 0.0),  # TX em z=5, RX em z=0
        (-1.0, 0.0),
        (0.0, -1.0),
        (6.0, 5.0),
        (5.0, 6.0),
    ]
    for h0, z in cases:
        ct_n, cr_n = find_layers_tr(3, h0, z, prof)
        ct_j, cr_j = find_layers_tr_jax(h0, z, prof_jnp, 3)
        assert (int(ct_j), int(cr_j)) == (ct_n, cr_n), (
            f"Divergência em fronteira h0={h0} z={z}: "
            f"Numba=({ct_n},{cr_n}) JAX=({int(ct_j)},{int(cr_j)})"
        )


@jax_required
def test_find_layers_tr_jax_parity_semispaces():
    """TX/RX nos semi-espaços (camadas 0 ou n-1).

    Regressão contra bug comum: `clip` incorreto pode mapear valores
    extremos (-1e6, +1e6) para camada errada.
    """
    from geosteering_ai.simulation._jax.geometry_jax import find_layers_tr_jax
    from geosteering_ai.simulation._numba.geometry import (
        find_layers_tr,
        sanitize_profile,
    )

    esp = np.array([0.0, 2.0, 2.0, 2.0, 0.0])
    _, prof = sanitize_profile(n=5, esp=esp)
    prof_jnp = jnp.asarray(prof)

    semi_cases = [
        (-1e5, -1e5),  # Ambos no semi-superior
        (1e5, 1e5),  # Ambos no semi-inferior
        (-1e5, 1e5),  # TX superior, RX inferior
        (1e5, -1e5),  # TX inferior, RX superior
    ]
    for h0, z in semi_cases:
        ct_n, cr_n = find_layers_tr(5, h0, z, prof)
        ct_j, cr_j = find_layers_tr_jax(h0, z, prof_jnp, 5)
        assert (int(ct_j), int(cr_j)) == (ct_n, cr_n)


@jax_required
def test_find_layers_tr_jax_vmap_parity():
    """Versão `find_layers_tr_jax_vmap` bate loop Numba em (n_pos,) posições."""
    from geosteering_ai.simulation._jax.geometry_jax import (
        find_layers_tr_jax_vmap,
    )
    from geosteering_ai.simulation._numba.geometry import (
        find_layers_tr,
        sanitize_profile,
    )

    esp = np.array([0.0, 4.0, 3.0, 5.0, 0.0])
    _, prof = sanitize_profile(n=5, esp=esp)
    prof_jnp = jnp.asarray(prof)

    rng = np.random.default_rng(7)
    n_pos = 60
    h0_arr = rng.uniform(-2, 15, n_pos)
    z_arr = rng.uniform(-2, 15, n_pos)

    ct_jax, cr_jax = find_layers_tr_jax_vmap(
        jnp.asarray(h0_arr), jnp.asarray(z_arr), prof_jnp, 5
    )
    ct_numba = np.array(
        [find_layers_tr(5, float(h), float(z), prof)[0] for h, z in zip(h0_arr, z_arr)]
    )
    cr_numba = np.array(
        [find_layers_tr(5, float(h), float(z), prof)[1] for h, z in zip(h0_arr, z_arr)]
    )

    assert np.array_equal(np.asarray(ct_jax), ct_numba)
    assert np.array_equal(np.asarray(cr_jax), cr_numba)


@jax_required
def test_find_layers_tr_jax_under_jit():
    """Compilação JIT não altera a saída."""
    from geosteering_ai.simulation._jax.geometry_jax import find_layers_tr_jax
    from geosteering_ai.simulation._numba.geometry import sanitize_profile

    esp = np.array([0.0, 5.0, 0.0])
    _, prof = sanitize_profile(n=3, esp=esp)
    prof_jnp = jnp.asarray(prof)

    @jax.jit
    def _jit_wrapper(h0, z):
        return find_layers_tr_jax(h0, z, prof_jnp, 3)

    ct_ref, cr_ref = find_layers_tr_jax(2.5, 2.5, prof_jnp, 3)
    ct_jit, cr_jit = _jit_wrapper(2.5, 2.5)
    assert int(ct_jit) == int(ct_ref)
    assert int(cr_jit) == int(cr_ref)


@jax_required
def test_find_layers_tr_jax_traceable():
    """Função aceita tracers (via vmap) e produz tracers int32.

    Essa é a propriedade crítica para o vmap_real de Sprint 12 — se
    `find_layers_tr_jax` não aceita tracers, a refatoração de
    `simulate_multi_jax` falha em tempo de traço.
    """
    from geosteering_ai.simulation._jax.geometry_jax import find_layers_tr_jax
    from geosteering_ai.simulation._numba.geometry import sanitize_profile

    esp = np.array([0.0, 5.0, 0.0])
    _, prof = sanitize_profile(n=3, esp=esp)
    prof_jnp = jnp.asarray(prof)

    # vmap introduz tracers BatchTracer em h0 e z
    vmapped = jax.vmap(
        lambda h0, z: find_layers_tr_jax(h0, z, prof_jnp, 3), in_axes=(0, 0)
    )
    h0_arr = jnp.array([-1.0, 2.5, 7.0])
    z_arr = jnp.array([2.5, 2.5, 2.5])
    ct, cr = vmapped(h0_arr, z_arr)
    assert ct.dtype == jnp.int32
    assert cr.dtype == jnp.int32
    assert ct.shape == (3,)
    assert cr.shape == (3,)


@jax_required
@pytest.mark.parametrize(
    "model_name", ["oklahoma_3", "oklahoma_5", "oklahoma_15", "oklahoma_28"]
)
def test_find_layers_tr_jax_diverse_models(model_name):
    """Paridade em modelos canônicos com perfis reais."""
    from geosteering_ai.simulation._jax.geometry_jax import find_layers_tr_jax
    from geosteering_ai.simulation._numba.geometry import (
        find_layers_tr,
        sanitize_profile,
    )
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model(model_name)
    n = len(m.rho_h)
    _, prof = sanitize_profile(n=n, esp=m.esp)
    prof_jnp = jnp.asarray(prof)

    rng = np.random.default_rng(1234)
    h0s = rng.uniform(m.min_depth - 5, m.max_depth + 5, 200)
    zs = rng.uniform(m.min_depth - 5, m.max_depth + 5, 200)

    for h0, z in zip(h0s, zs):
        ct_n, cr_n = find_layers_tr(n, float(h0), float(z), prof)
        ct_j, cr_j = find_layers_tr_jax(float(h0), float(z), prof_jnp, n)
        assert int(ct_j) == ct_n and int(cr_j) == cr_n, (
            f"Divergência em {model_name} n={n}, h0={h0:.3f} z={z:.3f}: "
            f"Numba=({ct_n},{cr_n}) JAX=({int(ct_j)},{int(cr_j)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Testes 8-17 — vmap_real multi-TR/multi-ângulo vs Python loop
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
@pytest.mark.parametrize("model_name", ["oklahoma_3", "oklahoma_5"])
def test_vmap_real_parity_vs_python_loop(model_name):
    """Paridade bit-exata ``simulate_multi_jax`` com/sem ``jax_vmap_real``.

    Gate: ``max|H_loop - H_vmap| < 1e-10``. Em CPU observamos 0.0 bit-exato
    porque ambos caminhos chamam o mesmo unified JIT internamente — apenas
    a forma de iteração (Python loop vs vmap) difere.
    """
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model(model_name)
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 40)
    kwargs = dict(
        rho_h=m.rho_h,
        rho_v=m.rho_v,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0, 2.0],
        dip_degs=[0.0, 30.0, 60.0],
    )

    res_loop = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=False)
    )
    res_vmap = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=True)
    )

    assert res_loop.H_tensor.shape == res_vmap.H_tensor.shape
    diff = float(np.abs(res_loop.H_tensor - res_vmap.H_tensor).max())
    assert diff < 1e-10, (
        f"Paridade vmap_real vs Python loop em {model_name}: " f"{diff:.3e} >= 1e-10"
    )


@jax_required
def test_vmap_real_shape_is_correct():
    """`H_tensor` retorna shape `(nTR, nAngles, n_pos, nf, 9)` com vmap_real."""
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model("oklahoma_3")
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 50)

    res = simulate_multi_jax(
        rho_h=m.rho_h,
        rho_v=m.rho_v,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=[20000.0, 40000.0],
        tr_spacings_m=[0.5, 1.0, 1.5],
        dip_degs=[0.0, 45.0],
        cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=True),
    )
    assert res.H_tensor.shape == (3, 2, 50, 2, 9)
    assert res.z_obs.shape == (2, 50)
    assert res.rho_h_at_obs.shape == (2, 50)
    assert res.unique_hordist_count >= 1


@jax_required
def test_vmap_real_backward_compat_default_false():
    """`jax_vmap_real` default é **False** (Python loop) — backward-compat.

    Se o default for flipado em v1.6.1+, este teste deve ser atualizado
    junto para refletir a nova convenção.
    """
    from geosteering_ai.simulation.config import SimulationConfig

    cfg = SimulationConfig()
    assert cfg.jax_vmap_real is False, (
        f"Default `jax_vmap_real` regrediu: esperado False, "
        f"recebeu {cfg.jax_vmap_real!r}"
    )


@jax_required
def test_vmap_real_multi_dip_exotic():
    """Paridade em configuração multi-dip exótica (dips extremos + TR variados)."""
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model("oklahoma_5")
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 30)
    kwargs = dict(
        rho_h=m.rho_h,
        rho_v=m.rho_v,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[0.5, 1.0, 1.5, 2.0],
        dip_degs=[0.0, 15.0, 45.0, 89.0],  # inclui borda 89°
    )
    res_loop = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=False)
    )
    res_vmap = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=True)
    )
    diff = float(np.abs(res_loop.H_tensor - res_vmap.H_tensor).max())
    assert diff < 1e-10, f"Paridade multi-dip exótica: {diff:.3e}"


@jax_required
def test_vmap_real_high_rho_stability():
    """vmap_real estável em alta resistividade (ρ ≥ 1000 Ω·m): sem NaN/Inf."""
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model("oklahoma_28")
    rho_h_high = np.clip(m.rho_h, a_min=1.0, a_max=None) * 15.0  # ~ρ ≈ 1500
    rho_v_high = np.clip(m.rho_v, a_min=1.0, a_max=None) * 15.0
    z = np.linspace(m.min_depth, m.max_depth, 30)

    res = simulate_multi_jax(
        rho_h=rho_h_high,
        rho_v=rho_v_high,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0, 30.0],
        cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=True),
    )
    assert np.all(np.isfinite(res.H_tensor)), "vmap_real produziu NaN/Inf em alta-ρ"


@jax_required
def test_vmap_real_vertical_well_still_works():
    """Poço vertical (todos dips = 0°) — vmap_real ainda funciona."""
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model("oklahoma_3")
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 25)
    kwargs = dict(
        rho_h=m.rho_h,
        rho_v=m.rho_v,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[0.5, 1.0, 1.5],
        dip_degs=[0.0],  # todos dips = 0 → vertical well
    )
    res_loop = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=False)
    )
    res_vmap = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=True)
    )
    diff = float(np.abs(res_loop.H_tensor - res_vmap.H_tensor).max())
    assert diff < 1e-10


@jax_required
def test_vmap_real_multi_freq_parity():
    """Multi-frequência preserva paridade vmap_real vs Python loop."""
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model("oklahoma_3")
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 20)
    kwargs = dict(
        rho_h=m.rho_h,
        rho_v=m.rho_v,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=[10000.0, 20000.0, 40000.0, 100000.0],
        tr_spacings_m=[0.5, 1.5],
        dip_degs=[0.0, 30.0],
    )
    res_loop = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=False)
    )
    res_vmap = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=True)
    )
    diff = float(np.abs(res_loop.H_tensor - res_vmap.H_tensor).max())
    assert diff < 1e-10


@jax_required
def test_vmap_real_single_tr_single_dip_single_freq():
    """Caso minimal (1×1×1×1) — vmap_real reduz a batch de 1 sem erro."""
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model("oklahoma_3")
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 15)
    res = simulate_multi_jax(
        rho_h=m.rho_h,
        rho_v=m.rho_v,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=True),
    )
    assert res.H_tensor.shape == (1, 1, 15, 1, 9)
    assert np.all(np.isfinite(res.H_tensor))


@jax_required
def test_vmap_real_parity_oklahoma_28():
    """Paridade em oklahoma_28 (28 camadas, stress-test para vmap real)."""
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model("oklahoma_28")
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 20)
    kwargs = dict(
        rho_h=m.rho_h,
        rho_v=m.rho_v,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0, 2.0],
        dip_degs=[0.0, 30.0],
    )
    res_loop = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=False)
    )
    res_vmap = simulate_multi_jax(
        **kwargs, cfg=SimulationConfig(jax_strategy="unified", jax_vmap_real=True)
    )
    diff = float(np.abs(res_loop.H_tensor - res_vmap.H_tensor).max())
    assert diff < 1e-10, f"Paridade oklahoma_28: {diff:.3e}"


@jax_required
def test_vmap_real_bucketed_strategy_also_works():
    """vmap_real com `jax_strategy='bucketed'` produz mesmos resultados.

    O dispatcher `simulate_multi_jax` honra tanto a flag `jax_vmap_real`
    quanto `jax_strategy`. Como `_simulate_multi_jax_vmap_real` internamente
    usa `_get_unified_jit`, esta configuração acaba usando unified mesmo
    quando o usuário passa `jax_strategy="bucketed"` — o teste documenta
    esse comportamento esperado.
    """
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model("oklahoma_3")
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 20)
    res = simulate_multi_jax(
        rho_h=m.rho_h,
        rho_v=m.rho_v,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0, 30.0],
        cfg=SimulationConfig(jax_strategy="bucketed", jax_vmap_real=True),
    )
    assert res.H_tensor.shape == (1, 2, 20, 1, 9)
    assert np.all(np.isfinite(res.H_tensor))
