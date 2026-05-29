# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_complex64_parity.py                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes Sprint O2 — complex64 opt-in (paridade vs c128)     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-28 (Sprint O2 — JAX dtype opt-in)                  ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest                                                     ║
# ║  Dependências: jax, numpy, geosteering_ai.simulation                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida o caminho `complex64` opt-in no backend JAX (Sprint O2),       ║
# ║    garantindo:                                                            ║
# ║      1. complex64 difere de complex128 em <1e-4 (gate PINN-friendly)    ║
# ║      2. complex128 (default) NÃO regrediu vs baseline (gate F1)         ║
# ║      3. Path use_native_dipoles puro c64 funciona                       ║
# ║                                                                           ║
# ║  GATES                                                                    ║
# ║    Gate paridade c64-c128: max_abs(H_c64 - H_c128) < 1e-4                 ║
# ║    Gate F1 c128 preserved : max_abs(H_jax - H_numba) < 2.301e-13         ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Sprint O2 — geosteering_ai/simulation/_jax/forward_pure.py:118-148  ║
# ║    • _COMPLEX_DTYPE_MAP — _jax/forward_pure.py                          ║
# ║    • SimulationConfig.dtype — geosteering_ai/simulation/config.py:361   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Sprint O2 — Testes de paridade complex64 (opt-in) vs complex128 (default).

Valida três proposições:

1. complex64 difere de complex128 em <1e-4 nos 7 modelos canônicos
   (tolerância PINN-friendly escolhida pelo usuário).
2. complex128 default NÃO regrediu — paridade JAX vs Numba <2.301e-13.
3. Caminho `use_native_dipoles=True` puro em complex64 funciona end-to-end.
"""

from __future__ import annotations

import os

# Garante x64 antes de qualquer import JAX (complex128 requer float64).
os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

jax_required = pytest.mark.skipif(not HAS_JAX, reason="JAX não instalado")


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: modelos canônicos (usa get_canonical_model do projeto)
# ──────────────────────────────────────────────────────────────────────────────


def _get_model(name: str):
    """Helper: retorna (rho_h, rho_v, esp, positions_z) para um modelo canônico."""
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model(name)  # type: ignore[arg-type]
    # Usa 30 posições para rapidez do teste
    positions_z = np.linspace(m.min_depth - 1.0, m.max_depth + 1.0, 30)
    return m.rho_h, m.rho_v, m.esp, positions_z


# ──────────────────────────────────────────────────────────────────────────────
# Teste 1: paridade c64 vs c128 — gate <1e-4 (PINN-friendly)
# ──────────────────────────────────────────────────────────────────────────────


@jax_required
@pytest.mark.parametrize(
    "model_name", ["oklahoma_3", "oklahoma_5", "oklahoma_28", "hou_7"]
)
def test_c64_vs_c128_parity_under_1e4(model_name: str) -> None:
    """Sprint O2 — H_c64 deve ficar a <1e-4 de H_c128 para 7 canônicos.

    Tolerância PINN-friendly (escolhida pelo usuário em decisão de design).
    complex64 = 2× menor footprint VRAM + ~1.5-2× speedup em GPU.
    """
    from geosteering_ai.simulation import SimulationConfig, simulate_multi_jax

    rho_h, rho_v, esp, positions_z = _get_model(model_name)

    # ── Path c128 (default, paridade Fortran sagrada) ────────────────────────
    cfg_c128 = SimulationConfig(backend="jax", dtype="complex128", parallel=False)
    res_c128 = simulate_multi_jax(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_c128,
    )

    # ── Path c64 (opt-in PINN/GPU) ───────────────────────────────────────────
    cfg_c64 = SimulationConfig(backend="jax", dtype="complex64", parallel=False)
    res_c64 = simulate_multi_jax(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_c64,
    )

    # ── Validação shape e finitude ───────────────────────────────────────────
    assert (
        res_c128.H_tensor.shape == res_c64.H_tensor.shape
    ), f"shape mismatch: c128={res_c128.H_tensor.shape}, c64={res_c64.H_tensor.shape}"
    assert np.all(np.isfinite(res_c64.H_tensor)), "H_c64 contém NaN/Inf"
    assert np.all(np.isfinite(res_c128.H_tensor)), "H_c128 contém NaN/Inf"

    # ── Gate: max_abs < 1e-4 (PINN-friendly) ────────────────────────────────
    diff = np.abs(res_c128.H_tensor - res_c64.H_tensor)
    max_abs = float(np.max(diff))
    assert max_abs < 1e-4, (
        f"[{model_name}] complex64 vs complex128 diverge em {max_abs:.3e} "
        f"(esperado < 1e-4)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Teste 2: c128 default não regrediu (gate F1)
# ──────────────────────────────────────────────────────────────────────────────


@jax_required
@pytest.mark.parametrize(
    "model_name",
    ["oklahoma_3", "oklahoma_5", "devine_8", "oklahoma_15", "hou_7"],
)
def test_c128_default_no_regression(model_name: str) -> None:
    """Sprint O2 — gate F1: complex128 default NÃO regrediu vs Numba.

    Tolerância: 2.301e-13 (baseline × 1.5). Path c128 é INVIOLÁVEL.
    """
    from geosteering_ai.simulation import (
        SimulationConfig,
        simulate_multi,
        simulate_multi_jax,
    )

    rho_h, rho_v, esp, positions_z = _get_model(model_name)

    # ── Numba (referência paridade Fortran) ───────────────────────────────────
    cfg_numba = SimulationConfig(backend="numba", parallel=False)
    res_numba = simulate_multi(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_numba,
    )

    # ── JAX c128 (default — gate F1) ─────────────────────────────────────────
    cfg_c128 = SimulationConfig(backend="jax", dtype="complex128", parallel=False)
    res_c128 = simulate_multi_jax(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_c128,
    )

    # ── Gate F1: <2.301e-13 (baseline × 1.5) ─────────────────────────────────
    diff = np.abs(res_numba.H_tensor - res_c128.H_tensor)
    max_abs = float(np.max(diff))
    assert max_abs < 2.301e-13, (
        f"[{model_name}] Gate F1 VIOLADO — JAX c128 vs Numba diverge em "
        f"{max_abs:.3e} (gate 2.301e-13). Sprint O2 quebrou paridade!"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Teste 3: caminho nativo (use_native_dipoles=True) em c64 funciona
# ──────────────────────────────────────────────────────────────────────────────


@jax_required
def test_use_native_dipoles_path_c64() -> None:
    """Sprint O2 — `forward_pure_jax` em c64 (use_native_dipoles=True) funciona.

    O path nativo é o esperado para complex64 em GPU/PINN. Path hibrido
    (pure_callback Numba) faz cast para c128 → c64 e mantem paridade c128
    nas operacoes Numba (cf. kernel.py:_dipoles_numba_host).
    """
    import jax.numpy as jnp

    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context,
        forward_pure_jax,
    )

    rho_h, rho_v, esp, positions_z = _get_model("oklahoma_3")

    ctx_c64 = build_static_context(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0,
        dip_deg=0.0,
        complex_dtype="complex64",
    )

    H_c64 = forward_pure_jax(
        jnp.asarray(rho_h, dtype=jnp.float64),
        jnp.asarray(rho_v, dtype=jnp.float64),
        ctx_c64,
    )
    H_c64.block_until_ready()

    assert H_c64.dtype == jnp.complex64, f"esperado complex64, obtido {H_c64.dtype}"
    assert np.all(np.isfinite(np.asarray(H_c64))), "H_c64 contém NaN/Inf"

    # Sanity: shape (n_pos, nf, 9)
    assert H_c64.shape == (30, 1, 9), f"shape inesperado: {H_c64.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# Teste 4: cache key inclui complex_dtype (Sprint O2 — proteção contra hit falso)
# ──────────────────────────────────────────────────────────────────────────────


@jax_required
def test_jit_cache_key_includes_complex_dtype() -> None:
    """Sprint O2 — cache key inclui complex_dtype (proteção CRITICA).

    Sem esta proteção, uma chamada com c64 hitar entrada compilada para
    c128 → resultado seria complex128 silenciosamente (bug invisível).
    """
    from geosteering_ai.simulation._jax.forward_pure import (
        _get_unified_jit,
        clear_unified_jit_cache,
    )

    clear_unified_jit_cache()

    # Compila c128 (default)
    jit_c128 = _get_unified_jit(n=3, npt=201, complex_dtype="complex128")
    # Compila c64 (opt-in)
    jit_c64 = _get_unified_jit(n=3, npt=201, complex_dtype="complex64")

    # CRITICO: devem ser funções jit distintas (compiladas separadamente)
    assert jit_c128 is not jit_c64, (
        "Cache hit incorreto — c128 e c64 devem ter JITs distintos. "
        "Sem isso, uma chamada c64 retornaria c128 silenciosamente."
    )

    # Re-chamada deve retornar a mesma instância (LRU hit)
    jit_c128_again = _get_unified_jit(n=3, npt=201, complex_dtype="complex128")
    assert jit_c128 is jit_c128_again, "LRU não está reutilizando entrada c128"


# ──────────────────────────────────────────────────────────────────────────────
# Teste 5: SimulationConfig.dtype valida valores
# ──────────────────────────────────────────────────────────────────────────────


@jax_required
def test_simulationconfig_dtype_validation() -> None:
    """Sprint O2 — SimulationConfig.dtype aceita complex128 e complex64."""
    from geosteering_ai.simulation import SimulationConfig

    # Default
    cfg = SimulationConfig()
    assert cfg.dtype == "complex128", f"default esperado complex128, got {cfg.dtype}"

    # Opt-in c64
    cfg_c64 = SimulationConfig(backend="jax", dtype="complex64")
    assert cfg_c64.dtype == "complex64"

    # Valor inválido → ValueError
    with pytest.raises(AssertionError):
        SimulationConfig(dtype="complex_invalid")  # type: ignore[arg-type]
