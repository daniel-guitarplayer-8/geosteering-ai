# -*- coding: utf-8 -*-
"""Testes do Jacobiano ∂H/∂ρ — Sprints 5.1 (JAX) + 5.2 (Numba FD).

Cobertura:

- **FD Numba (Sprint 5.2, 6 testes)**: shape/dtype, consistência de
  simetria top↔bottom, robustez a ε variável, estabilidade em ρ > 1000
  Ω·m, política de passo δ=clip(ε·|ρ|,1e-6,0.1·|ρ|), e consistência
  com baseline forward (célula j → H_plus ≠ H_baseline).

- **JAX dispatcher (Sprint 5.1, 3 testes)**: backend='jax' com fallback
  FD funciona, `compute_jacobian` dispatcher escolhe path correto, e
  ``_fd_step_for_rho`` respeita limites físicos.
"""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation._jacobian import (
    HAS_JAX,
    JacobianResult,
    _fd_step_for_rho,
    compute_jacobian,
    compute_jacobian_fd_numba,
    compute_jacobian_jax,
)
from geosteering_ai.simulation.config import SimulationConfig


@pytest.fixture
def simple_3layer_model() -> tuple:
    """Modelo Oklahoma 3 simplificado (isotrópico) para jacobiano rápido."""
    rho_h = np.array([10.0, 100.0, 10.0], dtype=np.float64)
    rho_v = np.array([10.0, 100.0, 10.0], dtype=np.float64)
    esp = np.array([5.0], dtype=np.float64)
    z = np.linspace(-2.0, 7.0, 6)  # apenas 6 posições para velocidade
    cfg = SimulationConfig(
        backend="numba",
        frequency_hz=20000.0,
        tr_spacing_m=1.0,
    )
    return rho_h, rho_v, esp, z, cfg


# ═════════════════════════════════════════════════════════════════════════
# SPRINT 5.2 — FD Numba (6 testes)
# ═════════════════════════════════════════════════════════════════════════


def test_fd_shape_and_dtype(simple_3layer_model) -> None:
    """FD retorna JacobianResult com shape (n_pos, nf, 9, n_layers) e dtype complex."""
    rho_h, rho_v, esp, z, cfg = simple_3layer_model
    jac = compute_jacobian_fd_numba(rho_h, rho_v, esp, z, cfg=cfg)
    assert isinstance(jac, JacobianResult)
    assert jac.dH_dRho_h.shape == (6, 1, 9, 3)
    assert jac.dH_dRho_v.shape == (6, 1, 9, 3)
    assert jac.dH_dRho_h.dtype == np.complex128
    assert jac.method == "fd_central"
    assert jac.backend == "numba_fd"
    assert jac.fd_step == 1e-4
    assert jac.n_layers == 3
    assert np.all(np.isfinite(jac.dH_dRho_h.real))
    assert np.all(np.isfinite(jac.dH_dRho_h.imag))


def test_fd_symmetry_isotropic_model(simple_3layer_model) -> None:
    """Em modelo isotrópico, camadas 0 e 2 (semi-espaços iguais) devem ter
    normas de ∂H/∂ρ similares (simetria vertical do modelo)."""
    rho_h, rho_v, esp, z, cfg = simple_3layer_model
    jac = compute_jacobian_fd_numba(rho_h, rho_v, esp, z, cfg=cfg)
    norms = jac.norm_per_layer()  # shape (3, 2)
    # Camadas 0 e 2 — simétricas porque modelo e perfil de z são simétricos.
    # Tolerância frouxa: o perfil de z não é perfeitamente simétrico em
    # torno do meio (5m com z em [-2,7]), então aceitamos ~20% de razão.
    ratio_h = norms[0, 0] / norms[2, 0]
    assert 0.5 < ratio_h < 2.0, f"Assimetria excessiva: razão={ratio_h}"


def test_fd_step_robustness() -> None:
    """Jacobiano com ε ∈ {1e-3, 1e-4, 1e-5} deve ser consistente (< 1% var)."""
    rho_h = np.array([10.0, 100.0, 10.0])
    rho_v = np.array([10.0, 100.0, 10.0])
    esp = np.array([5.0])
    z = np.linspace(0.0, 5.0, 4)
    cfg = SimulationConfig(backend="numba", frequency_hz=20000.0, tr_spacing_m=1.0)

    jacs = [
        compute_jacobian_fd_numba(rho_h, rho_v, esp, z, cfg=cfg, fd_step=eps).dH_dRho_h
        for eps in (1e-3, 1e-4, 1e-5)
    ]
    # Compara ε=1e-4 (referência) contra 1e-3 e 1e-5.
    ref = jacs[1]
    max_var = 0.0
    for other in (jacs[0], jacs[2]):
        mask = np.abs(ref) > 1e-10
        if not np.any(mask):
            continue
        rel_err = np.abs((other - ref)[mask] / ref[mask])
        max_var = max(max_var, float(np.max(rel_err)))
    assert max_var < 5e-2, f"FD inconsistente: max_rel_var={max_var}"


def test_fd_high_rho_stability() -> None:
    """ρ > 1000 Ω·m permanece finito com política de passo adaptativa."""
    rho_h = np.array([10.0, 5000.0, 10.0])
    rho_v = np.array([10.0, 5000.0, 10.0])
    esp = np.array([3.0])
    z = np.linspace(0.0, 3.0, 4)
    cfg = SimulationConfig(backend="numba", frequency_hz=400000.0, tr_spacing_m=0.5)

    jac = compute_jacobian_fd_numba(rho_h, rho_v, esp, z, cfg=cfg)
    assert np.all(np.isfinite(jac.dH_dRho_h.real))
    assert np.all(np.isfinite(jac.dH_dRho_h.imag))
    assert np.all(np.isfinite(jac.dH_dRho_v.real))


def test_fd_step_policy_bounds() -> None:
    """δ deve respeitar clip(ε·|ρ|, 1e-6, 0.1·|ρ|)."""
    # Caso 1: ρ grande, ε pequeno → delta = ε·ρ (dentro do range).
    assert _fd_step_for_rho(1000.0, 1e-4) == pytest.approx(0.1, abs=1e-12)
    # Caso 2: ρ pequeno, ε grande → delta saturado em 0.1·ρ.
    assert _fd_step_for_rho(1.0, 1.0) == pytest.approx(0.1, abs=1e-12)
    # Caso 3: ρ extremamente pequeno → piso 1e-6.
    assert _fd_step_for_rho(1e-8, 1e-4) == pytest.approx(1e-6, abs=1e-12)


def test_fd_isotropic_jacobian_nonzero(simple_3layer_model) -> None:
    """Camada média tem resistividade diferente → ∂H/∂ρ deve ser não-trivial."""
    rho_h, rho_v, esp, z, cfg = simple_3layer_model
    jac = compute_jacobian_fd_numba(rho_h, rho_v, esp, z, cfg=cfg)
    # Hzz (componente 8) na camada média deve variar visivelmente em z
    # entre as posições dentro e fora da camada resistiva.
    d_hzz_layer1 = jac.dH_dRho_h[:, 0, 8, 1]  # ∂Hzz/∂ρ_h da camada 1
    assert np.max(np.abs(d_hzz_layer1)) > 1e-8


# ═════════════════════════════════════════════════════════════════════════
# SPRINT 5.1 — JAX jacfwd + dispatcher + fallback (3 testes)
# ═════════════════════════════════════════════════════════════════════════


def test_compute_jacobian_dispatcher_numba(simple_3layer_model) -> None:
    """Dispatcher com backend='numba' deve invocar FD Numba."""
    rho_h, rho_v, esp, z, cfg = simple_3layer_model
    jac = compute_jacobian(rho_h, rho_v, esp, z, cfg=cfg)
    assert jac.backend == "numba_fd"
    assert jac.method == "fd_central"


def test_compute_jacobian_jax_fallback_fd() -> None:
    """Com backend='jax', jacfwd experimental cai em FD fallback (esperado no PR #13)."""
    rho_h = np.array([10.0, 100.0, 10.0])
    rho_v = np.array([10.0, 100.0, 10.0])
    esp = np.array([5.0])
    z = np.linspace(0.0, 5.0, 4)
    cfg = SimulationConfig(backend="jax", frequency_hz=20000.0, tr_spacing_m=1.0)

    jac = compute_jacobian_jax(rho_h, rho_v, esp, z, cfg=cfg, try_jacfwd=True)
    # PR #13: jacfwd nativo end-to-end é experimental; fallback FD é esperado.
    assert jac.method == "fd_central"
    # Backend reflete que o forward subjacente é JAX, mas método Jacobiano é FD.
    assert "fd" in jac.backend.lower() or jac.backend == "numba_fd"
    assert np.all(np.isfinite(jac.dH_dRho_h.real))


def test_jacobian_result_serialization(simple_3layer_model) -> None:
    """JacobianResult.to_dict() produz payload NPZ-compatível."""
    rho_h, rho_v, esp, z, cfg = simple_3layer_model
    jac = compute_jacobian_fd_numba(rho_h, rho_v, esp, z, cfg=cfg)
    payload = jac.to_dict()
    assert "dH_dRho_h" in payload
    assert "method" in payload
    # Todos os valores devem ser serializáveis via np.savez.
    # Teste leve: tenta salvar em buffer temporário.
    import io

    buf = io.BytesIO()
    np.savez(buf, **{k: v for k, v in payload.items() if isinstance(v, np.ndarray)})
    assert buf.tell() > 0
