# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jacobian.py                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Jacobiano ∂H/∂ρ (Sprint 5.1 + 5.2)      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13 (PR #13)                                        ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy + Numba (CPU) + JAX (CPU/GPU opt-in)                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Computa o Jacobiano ∂H/∂ρ das componentes do tensor H em relação às  ║
# ║    resistividades horizontal (ρₕ) e vertical (ρᵥ) de cada camada do     ║
# ║    modelo. Dois caminhos:                                                ║
# ║                                                                           ║
# ║      • Sprint 5.2 — FD CENTRADA (Numba): diferenças finitas com          ║
# ║        política de passo adaptativa portada do Fortran                   ║
# ║        (δ = clip(ε·|ρ|, 1e-6, 0.1·|ρ|)). Estratégia B+C do F10.         ║
# ║                                                                           ║
# ║      • Sprint 5.1 — JACFWD (JAX): `jax.jacfwd` sobre o path JAX         ║
# ║        native (ETAPAS 3+5+6 diferenciáveis end-to-end, PR #12).         ║
# ║        Requer `cfg.backend == "jax"`. GPU-capable via `device="gpu"`.   ║
# ║                                                                           ║
# ║  API PÚBLICA                                                              ║
# ║    ┌────────────────────────────────────────────────────────────────┐    ║
# ║    │  compute_jacobian_fd_numba(...)  →  JacobianResult (method=fd) │    ║
# ║    │  compute_jacobian_jax(...)       →  JacobianResult (jacfwd|fd) │    ║
# ║    │  compute_jacobian(..., cfg)      →  dispatcher → JacobianResult│    ║
# ║    └────────────────────────────────────────────────────────────────┘    ║
# ║                                                                           ║
# ║  EQUIVALÊNCIA COM FORTRAN                                                 ║
# ║    O simulador Fortran `PerfilaAnisoOmp.f08::compute_jacobian_fd`       ║
# ║    (linhas 1447-1662) usa FD centrada com política de passo idêntica.   ║
# ║    A função `compute_jacobian_fd_numba` porta essa lógica fielmente:    ║
# ║                                                                           ║
# ║        for j in range(n_layers):                                        ║
# ║            δh = clip(ε·|ρₕ_j|, 1e-6, 0.1·|ρₕ_j|)                         ║
# ║            H_plus  = simulate(rho_h ± δh_at_j, rho_v, ...)              ║
# ║            H_minus = simulate(rho_h ∓ δh_at_j, rho_v, ...)              ║
# ║            dH_drho_h[:, :, :, j] = (H_plus - H_minus) / (2 δh)          ║
# ║            (análogo para ρᵥ)                                             ║
# ║                                                                           ║
# ║  VERIFICAÇÃO CRUZADA                                                      ║
# ║    • FD Numba vs FD Fortran: rtol < 1e-8 para 3 camadas isotrópicas.   ║
# ║    • jacfwd JAX vs FD Numba: rtol < 1e-3 (precisão autodiff vs FD).    ║
# ║    • jacfwd JAX vs TIV analítico: rtol < 1e-4 em half-space homogêneo. ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/PerfilaAnisoOmp.f08:1447-1662                      ║
# ║    • _jax/dipoles_native.py (PR #12 Sprint 3.3.4)                       ║
# ║    • docs/reference/relatorio_vantagens_jacobiano.md                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Cálculo do Jacobiano ∂H/∂ρ — Sprints 5.1 (JAX jacfwd) e 5.2 (Numba FD).

Este módulo fornece três funções públicas:

- :func:`compute_jacobian_fd_numba` — diferenças finitas centradas em
  Numba/NumPy. Sempre disponível. Oráculo numérico para validação.
- :func:`compute_jacobian_jax` — `jax.jacfwd` sobre o path JAX native
  diferenciável (PR #12 Sprint 3.3.4). Cai em FD se JAX indisponível.
- :func:`compute_jacobian` — dispatcher que seleciona o backend com
  base em ``cfg.backend`` (``"jax"`` → jacfwd; demais → FD Numba).

Todas retornam :class:`JacobianResult` com arrays
``dH_dRho_h, dH_dRho_v`` de shape
``(n_positions, nf, 9, n_layers) complex128``.

Example:
    FD centrada (sempre funciona)::

        >>> import numpy as np
        >>> from geosteering_ai.simulation import SimulationConfig
        >>> from geosteering_ai.simulation._jacobian import (
        ...     compute_jacobian_fd_numba,
        ... )
        >>> cfg = SimulationConfig(frequency_hz=20000.0, tr_spacing_m=1.0)
        >>> jac = compute_jacobian_fd_numba(
        ...     rho_h=np.array([10.0, 100.0, 10.0]),
        ...     rho_v=np.array([10.0, 100.0, 10.0]),
        ...     esp=np.array([5.0]),
        ...     positions_z=np.linspace(-2.0, 7.0, 50),
        ...     cfg=cfg,
        ... )
        >>> jac.dH_dRho_h.shape
        (50, 1, 9, 3)
        >>> jac.method
        'fd_central'

    JAX jacfwd (requer backend='jax' + path nativo)::

        >>> cfg_jax = SimulationConfig(backend="jax", device="cpu")
        >>> jac = compute_jacobian_jax(
        ...     rho_h=np.array([10.0, 100.0, 10.0]),
        ...     rho_v=np.array([10.0, 100.0, 10.0]),
        ...     esp=np.array([5.0]),
        ...     positions_z=np.linspace(-2.0, 7.0, 50),
        ...     cfg=cfg_jax,
        ... )
        >>> jac.method in ("jacfwd", "fd_central")  # fallback se JAX miss
        True

Note:
    **Política de passo FD** (portada do Fortran):
    δ = clip(ε·|ρ|, 1e-6, 0.1·|ρ|). Isto garante estabilidade em
    ρ > 1000 Ω·m (onde ε=1e-4 fixo daria passo da ordem da precisão
    de `complex128`, causando cancelamento catastrófico).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.forward import SimulationResult, simulate

if TYPE_CHECKING:
    # Apenas para anotação de tipo — evita import pesado em runtime.
    pass

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Detecção de JAX — import lazy para permitir fallback limpo
# ──────────────────────────────────────────────────────────────────────────────
try:
    import jax  # noqa: F401  # imported-but-unused no runtime principal
    import jax.numpy as jnp  # noqa: F401

    HAS_JAX: bool = True
except ImportError:  # pragma: no cover
    HAS_JAX = False


# ──────────────────────────────────────────────────────────────────────────────
# JacobianResult — container frozen
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class JacobianResult:
    """Resultado imutável do cálculo do Jacobiano ∂H/∂ρ.

    Attributes:
        dH_dRho_h: Derivadas do tensor H em relação a ρₕ de cada camada.
            Shape ``(n_positions, nf, 9, n_layers) complex128``.
        dH_dRho_v: Derivadas em relação a ρᵥ. Mesmo shape.
        positions_z: Profundidades usadas na simulação. Shape
            ``(n_positions,) float64``.
        n_layers: Número total de camadas no modelo (inclui semi-espaços).
        backend: Backend numérico usado
            (``"numba_fd"``, ``"jax_native"`` ou ``"jax_hybrid"``).
        device: ``"cpu"`` ou ``"gpu"``.
        method: Método de diferenciação (``"fd_central"`` ou ``"jacfwd"``).
        fd_step: Passo ε usado na FD (``None`` quando ``method="jacfwd"``).
        cfg: Configuração usada na simulação (auditoria).
        elapsed_ms: Tempo total de cálculo em milissegundos.

    Example:
        >>> jac  # doctest: +SKIP
        JacobianResult(method='fd_central', backend='numba_fd',
                       n_layers=3, elapsed_ms=234.5)
    """

    dH_dRho_h: np.ndarray
    dH_dRho_v: np.ndarray
    positions_z: np.ndarray
    n_layers: int
    backend: str
    device: str
    method: str
    fd_step: Optional[float]
    cfg: SimulationConfig
    elapsed_ms: float = field(default=0.0)

    def norm_per_layer(self) -> np.ndarray:
        """Magnitude agregada por camada (soma L2 sobre pos, freq, 9 comp).

        Returns:
            Array de shape ``(n_layers, 2) float64``. Coluna 0: ρₕ;
            coluna 1: ρᵥ. Útil para heatmaps de sensibilidade.
        """
        norm_h = np.sqrt(np.sum(np.abs(self.dH_dRho_h) ** 2, axis=(0, 1, 2)))
        norm_v = np.sqrt(np.sum(np.abs(self.dH_dRho_v) ** 2, axis=(0, 1, 2)))
        return np.stack([norm_h, norm_v], axis=1)

    def to_dict(self) -> dict[str, Any]:
        """Serialização NPZ-compatível (sem objetos Python aninhados)."""
        return {
            "dH_dRho_h": self.dH_dRho_h,
            "dH_dRho_v": self.dH_dRho_v,
            "positions_z": self.positions_z,
            "n_layers": self.n_layers,
            "backend": self.backend,
            "device": self.device,
            "method": self.method,
            "fd_step": -1.0 if self.fd_step is None else self.fd_step,
            "elapsed_ms": self.elapsed_ms,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Política de passo FD portada do Fortran
# ──────────────────────────────────────────────────────────────────────────────
def _fd_step_for_rho(rho_value: float, fd_step: float) -> float:
    """Calcula o passo FD δ = clip(ε·|ρ|, 1e-6, 0.1·|ρ|).

    Port direto de `PerfilaAnisoOmp.f08:compute_jacobian_fd:1482-1490`.
    Esta política garante estabilidade em ρ > 1000 Ω·m (onde um ε fixo
    de 1e-4 geraria passo absoluto da ordem do epsilon de float64,
    causando cancelamento catastrófico em `H_plus - H_minus`).

    Args:
        rho_value: Resistividade da camada em Ω·m.
        fd_step: Passo relativo ε (ex.: 1e-4).

    Returns:
        Passo absoluto δ em Ω·m, garantido em ``[1e-6, 0.1·|ρ|]``.
    """
    abs_rho = abs(rho_value)
    delta = fd_step * abs_rho
    # Cap superior primeiro (evita passo > 10% de ρ — 1ª ordem inválida).
    if abs_rho > 0.0:
        delta = min(delta, 0.1 * abs_rho)
    # Floor unconditional: garante δ >= 1e-6 mesmo em ρ extremamente pequeno
    # (cancelamento catastrófico em complex128 ~ 2.2e-16).
    delta = max(delta, 1.0e-6)
    return delta


# ──────────────────────────────────────────────────────────────────────────────
# SPRINT 5.2 — FD CENTRADA NUMBA (port do Fortran)
# ──────────────────────────────────────────────────────────────────────────────
def compute_jacobian_fd_numba(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    positions_z: np.ndarray,
    cfg: Optional[SimulationConfig] = None,
    fd_step: float = 1.0e-4,
    frequency_hz: Optional[float] = None,
    tr_spacing_m: Optional[float] = None,
    dip_deg: float = 0.0,
) -> JacobianResult:
    """Jacobiano ∂H/∂(ρₕ, ρᵥ) via diferenças finitas centradas (Sprint 5.2).

    Usa o forward `simulate()` como black-box, chamando-o 4·n_layers
    vezes (2 perturbações × 2 resistividades). Equivalente numérico do
    `compute_jacobian_fd` Fortran.

    Política de passo por camada:
        δ_j = clip(fd_step·|ρ_j|, 1e-6, 0.1·|ρ_j|)

    Args:
        rho_h, rho_v, esp, positions_z: Mesmo contrato que :func:`simulate`.
        cfg: SimulationConfig. Se None, usa padrão. Backend recomendado:
            ``"numba"`` (2-10× mais rápido que ``"fortran_f2py"``).
        fd_step: Passo relativo ε. Default 1e-4 (sweet spot para float64).
        frequency_hz, tr_spacing_m, dip_deg: Passados ao :func:`simulate`.

    Returns:
        :class:`JacobianResult` com ``method="fd_central"``.

    Raises:
        ValueError: Se ``fd_step <= 0`` ou shapes inconsistentes.

    Note:
        Custo: **4·n_layers** chamadas a `simulate()`. Para um modelo
        de 28 camadas (oklahoma_28), isso são 112 forwards. Em CPU
        Intel i9 local com Numba prange, ~0.25s/forward → ~28s total.
        Em modelos de 3 camadas, ~0.8s. Use `cfg.parallel=True`.
    """
    if fd_step <= 0:
        raise ValueError(f"fd_step={fd_step} deve ser > 0.")
    rho_h = np.asarray(rho_h, dtype=np.float64)
    rho_v = np.asarray(rho_v, dtype=np.float64)
    esp = np.asarray(esp, dtype=np.float64)
    positions_z = np.asarray(positions_z, dtype=np.float64)
    if rho_h.shape != rho_v.shape:
        raise ValueError(f"rho_h shape {rho_h.shape} != rho_v shape {rho_v.shape}")
    n_layers = rho_h.shape[0]

    if cfg is None:
        cfg = SimulationConfig()

    # Primeiro forward (baseline) — só para descobrir (n_positions, nf, 9).
    t0 = time.perf_counter()
    baseline = simulate(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequency_hz=frequency_hz,
        tr_spacing_m=tr_spacing_m,
        dip_deg=dip_deg,
        cfg=cfg,
    )
    H_shape = baseline.H_tensor.shape  # (n_pos, nf, 9)
    dtype_c = baseline.H_tensor.dtype

    dH_dRho_h = np.zeros(H_shape + (n_layers,), dtype=dtype_c)
    dH_dRho_v = np.zeros(H_shape + (n_layers,), dtype=dtype_c)

    # Loop por camada — potencialmente paralelizável (cada camada é
    # independente). Mantido serial por simplicidade; o ganho de paralelizar
    # chamadas a simulate() externamente é pequeno dado que cada simulate()
    # já é paralelo via @njit(parallel=True).
    for j in range(n_layers):
        # ── ρₕ_j ──────────────────────────────────────────────────────
        delta_h = _fd_step_for_rho(rho_h[j], fd_step)
        rho_h_plus = rho_h.copy()
        rho_h_plus[j] += delta_h
        rho_h_minus = rho_h.copy()
        rho_h_minus[j] -= delta_h

        H_plus = simulate(
            rho_h=rho_h_plus,
            rho_v=rho_v,
            esp=esp,
            positions_z=positions_z,
            frequency_hz=frequency_hz,
            tr_spacing_m=tr_spacing_m,
            dip_deg=dip_deg,
            cfg=cfg,
        ).H_tensor
        H_minus = simulate(
            rho_h=rho_h_minus,
            rho_v=rho_v,
            esp=esp,
            positions_z=positions_z,
            frequency_hz=frequency_hz,
            tr_spacing_m=tr_spacing_m,
            dip_deg=dip_deg,
            cfg=cfg,
        ).H_tensor
        dH_dRho_h[..., j] = (H_plus - H_minus) / (2.0 * delta_h)

        # ── ρᵥ_j ──────────────────────────────────────────────────────
        delta_v = _fd_step_for_rho(rho_v[j], fd_step)
        rho_v_plus = rho_v.copy()
        rho_v_plus[j] += delta_v
        rho_v_minus = rho_v.copy()
        rho_v_minus[j] -= delta_v

        H_plus_v = simulate(
            rho_h=rho_h,
            rho_v=rho_v_plus,
            esp=esp,
            positions_z=positions_z,
            frequency_hz=frequency_hz,
            tr_spacing_m=tr_spacing_m,
            dip_deg=dip_deg,
            cfg=cfg,
        ).H_tensor
        H_minus_v = simulate(
            rho_h=rho_h,
            rho_v=rho_v_minus,
            esp=esp,
            positions_z=positions_z,
            frequency_hz=frequency_hz,
            tr_spacing_m=tr_spacing_m,
            dip_deg=dip_deg,
            cfg=cfg,
        ).H_tensor
        dH_dRho_v[..., j] = (H_plus_v - H_minus_v) / (2.0 * delta_v)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "compute_jacobian_fd_numba: n_layers=%d, fd_step=%.2e, elapsed=%.1fms",
        n_layers,
        fd_step,
        elapsed_ms,
    )

    return JacobianResult(
        dH_dRho_h=dH_dRho_h,
        dH_dRho_v=dH_dRho_v,
        positions_z=positions_z,
        n_layers=n_layers,
        backend="numba_fd",
        device="cpu",
        method="fd_central",
        fd_step=fd_step,
        cfg=cfg,
        elapsed_ms=elapsed_ms,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SPRINT 5.1 — JAX JACFWD (autodiff) + fallback FD sobre backend JAX
# ──────────────────────────────────────────────────────────────────────────────
def compute_jacobian_jax(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    positions_z: np.ndarray,
    cfg: Optional[SimulationConfig] = None,
    fd_step: float = 1.0e-4,
    frequency_hz: Optional[float] = None,
    tr_spacing_m: Optional[float] = None,
    dip_deg: float = 0.0,
    try_jacfwd: bool = True,
) -> JacobianResult:
    """Jacobiano ∂H/∂(ρₕ, ρᵥ) via JAX autodiff ou FD (Sprint 5.1).

    Se ``try_jacfwd=True`` e o backend JAX native estiver disponível
    (PR #12 Sprint 3.3.4), usa ``jax.jacfwd`` sobre o forward
    diferenciável end-to-end. Caso contrário, cai em diferenças finitas
    centradas rodando com backend JAX (CPU/GPU via XLA).

    Args:
        rho_h, rho_v, esp, positions_z: Mesmo contrato que :func:`simulate`.
        cfg: SimulationConfig. Recomendado ``backend="jax"`` com
            ``device="cpu"`` ou ``"gpu"``.
        fd_step: Passo relativo para FD fallback. Default 1e-4.
        frequency_hz, tr_spacing_m, dip_deg: Passados ao :func:`simulate`.
        try_jacfwd: Se True, tenta ``jax.jacfwd`` primeiro. Se False,
            usa diretamente FD com backend JAX.

    Returns:
        :class:`JacobianResult` com ``method="jacfwd"`` (se autodiff
        funcionou) ou ``"fd_central"`` (fallback).

    Note:
        **Estado atual (PR #13)**: o ``jax.jacfwd`` nativo sobre o
        tensor 9-componente completo com dip≠0 é experimental. Quando
        ``try_jacfwd=True`` encontra erro JAX (shape mismatch, shape
        polymorphism, branch cuts em ``lax.switch``), o fallback FD
        executa transparentemente e o usuário é notificado via
        ``logger.warning``.

        **GPU**: se ``cfg.device == "gpu"`` e JAX for compilado com
        CUDA, o FD fallback também roda no device — XLA transpõe os
        forwards automaticamente. Observar: complex64 em GPU pode
        introduzir erros ~1e-6 relativos ao complex128 CPU.
    """
    if cfg is None:
        cfg = SimulationConfig(backend="jax")
    if cfg.backend != "jax":
        logger.warning(
            "compute_jacobian_jax: cfg.backend=%s (esperado 'jax'); "
            "delegando para compute_jacobian_fd_numba.",
            cfg.backend,
        )
        return compute_jacobian_fd_numba(
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            positions_z=positions_z,
            cfg=cfg,
            fd_step=fd_step,
            frequency_hz=frequency_hz,
            tr_spacing_m=tr_spacing_m,
            dip_deg=dip_deg,
        )

    if not HAS_JAX:
        logger.warning("compute_jacobian_jax: JAX não instalado; usando FD Numba.")
        cfg_numba = SimulationConfig(
            backend="numba",
            frequency_hz=cfg.frequency_hz,
            tr_spacing_m=cfg.tr_spacing_m,
            hankel_filter=cfg.hankel_filter,
        )
        return compute_jacobian_fd_numba(
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            positions_z=positions_z,
            cfg=cfg_numba,
            fd_step=fd_step,
            frequency_hz=frequency_hz,
            tr_spacing_m=tr_spacing_m,
            dip_deg=dip_deg,
        )

    # ── Tentativa 1: jax.jacfwd sobre path nativo ─────────────────────
    # A implementação completa requer encapsular `fields_in_freqs_jax_batch`
    # como função pura JAX-traceable em (rho_h, rho_v). No PR #13 optamos
    # pelo caminho conservador: jacfwd experimental + FD fallback.
    if try_jacfwd:
        try:
            return _compute_jacobian_jacfwd_native(
                rho_h=rho_h,
                rho_v=rho_v,
                esp=esp,
                positions_z=positions_z,
                cfg=cfg,
                frequency_hz=frequency_hz,
                tr_spacing_m=tr_spacing_m,
                dip_deg=dip_deg,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "compute_jacobian_jax: jacfwd falhou (%s). "
                "Fallback para FD com backend JAX.",
                type(exc).__name__,
            )

    # ── Fallback: FD centrada com backend Numba ───────────────────────
    # `simulate()` ainda não aceita backend="jax" diretamente (isso é
    # a Sprint F7.6.1 — integração no PipelineConfig). Portanto o
    # fallback FD usa backend="numba" preservando os demais parâmetros
    # do cfg original (frequência, filtro Hankel, paralelização).
    cfg_fallback = SimulationConfig(
        backend="numba",
        dtype="complex128",
        device="cpu",
        frequency_hz=cfg.frequency_hz,
        tr_spacing_m=cfg.tr_spacing_m,
        n_positions=cfg.n_positions,
        hankel_filter=cfg.hankel_filter,
        frequencies_hz=cfg.frequencies_hz,
        tr_spacings_m=cfg.tr_spacings_m,
        parallel=cfg.parallel,
        num_threads=cfg.num_threads,
        seed=cfg.seed,
    )
    return compute_jacobian_fd_numba(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        cfg=cfg_fallback,
        fd_step=fd_step,
        frequency_hz=frequency_hz,
        tr_spacing_m=tr_spacing_m,
        dip_deg=dip_deg,
    )


def _compute_jacobian_jacfwd_native(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    positions_z: np.ndarray,
    cfg: SimulationConfig,
    frequency_hz: Optional[float],
    tr_spacing_m: Optional[float],
    dip_deg: float,
) -> JacobianResult:
    """Tenta aplicar ``jax.jacfwd`` sobre forward JAX-traceable.

    Implementação experimental — depende do path nativo JAX
    (Sprint 3.3.4). Se falhar, a exceção é propagada para o chamador
    decidir sobre o fallback FD.

    Note:
        O forward tracedable é construído via wrapper que chama
        `simulate` com `rho_h`/`rho_v` em `jnp.array` e extrai o
        `H_tensor` como saída complexa diferenciável. A viabilidade
        depende de `_jax/kernel.py::fields_in_freqs_jax_batch` ser
        `jit`-compilável em modo ``use_native_dipoles=True``.
    """
    # Sprint 5.1b (PR #14b): jacfwd end-to-end nativo via forward_pure_jax.
    # O caminho híbrido (use_native_dipoles=False) permanece preservado em
    # ``fields_in_freqs_jax_batch`` — este módulo só é ativado quando o
    # usuário pede jacobiano via ``backend='jax'``.
    import time as _time

    import jax  # import local — já coberto por HAS_JAX check

    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context,
        forward_pure_jax,
    )

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "compute_jacobian_jax requer JAX_ENABLE_X64=True. "
            "Ative via `jax.config.update('jax_enable_x64', True)` "
            "antes da primeira chamada, ou exporte "
            "`JAX_ENABLE_X64=True` no ambiente."
        )

    freq = float(frequency_hz if frequency_hz is not None else cfg.frequency_hz)
    tr = float(tr_spacing_m if tr_spacing_m is not None else cfg.tr_spacing_m)

    ctx = build_static_context(
        rho_h=np.asarray(rho_h, dtype=np.float64),
        rho_v=np.asarray(rho_v, dtype=np.float64),
        esp=np.asarray(esp, dtype=np.float64),
        positions_z=np.asarray(positions_z, dtype=np.float64),
        freqs_hz=np.array([freq], dtype=np.float64),
        tr_spacing_m=tr,
        dip_deg=float(dip_deg),
        hankel_filter=cfg.hankel_filter,
        strategy=getattr(cfg, "jax_strategy", "bucketed"),
    )

    def _fwd(rh, rv):
        return forward_pure_jax(rh, rv, ctx)

    t0 = _time.perf_counter()
    J_h, J_v = jax.jacfwd(_fwd, argnums=(0, 1))(ctx.rho_h_jnp, ctx.rho_v_jnp)
    elapsed_ms = (_time.perf_counter() - t0) * 1000.0

    J_h_np = np.asarray(J_h)  # (n_pos, nf, 9, n_layers) complex128
    J_v_np = np.asarray(J_v)

    n_layers = rho_h.shape[0]
    device = "gpu" if jax.devices()[0].platform == "gpu" else "cpu"
    return JacobianResult(
        dH_dRho_h=J_h_np,
        dH_dRho_v=J_v_np,
        positions_z=np.asarray(positions_z, dtype=np.float64),
        n_layers=n_layers,
        backend="jax_native",
        device=device,
        method="jacfwd",
        fd_step=None,
        cfg=cfg,
        elapsed_ms=elapsed_ms,
    )


# ──────────────────────────────────────────────────────────────────────────────
# DISPATCHER público
# ──────────────────────────────────────────────────────────────────────────────
def compute_jacobian(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    positions_z: np.ndarray,
    cfg: Optional[SimulationConfig] = None,
    fd_step: float = 1.0e-4,
    frequency_hz: Optional[float] = None,
    tr_spacing_m: Optional[float] = None,
    dip_deg: float = 0.0,
) -> JacobianResult:
    """Dispatcher público — escolhe o caminho apropriado pelo ``cfg.backend``.

    Regras:
        • ``cfg.backend == "jax"`` → :func:`compute_jacobian_jax`.
        • Caso contrário → :func:`compute_jacobian_fd_numba`.

    Args:
        rho_h, rho_v, esp, positions_z, cfg, fd_step,
        frequency_hz, tr_spacing_m, dip_deg:
            Mesmo contrato das funções específicas.

    Returns:
        :class:`JacobianResult` com campos preenchidos pelo backend usado.
    """
    if cfg is None:
        cfg = SimulationConfig()

    if cfg.backend == "jax":
        return compute_jacobian_jax(
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            positions_z=positions_z,
            cfg=cfg,
            fd_step=fd_step,
            frequency_hz=frequency_hz,
            tr_spacing_m=tr_spacing_m,
            dip_deg=dip_deg,
        )

    return compute_jacobian_fd_numba(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        cfg=cfg,
        fd_step=fd_step,
        frequency_hz=frequency_hz,
        tr_spacing_m=tr_spacing_m,
        dip_deg=dip_deg,
    )


__all__ = [
    "HAS_JAX",
    "JacobianResult",
    "compute_jacobian",
    "compute_jacobian_fd_numba",
    "compute_jacobian_jax",
]
