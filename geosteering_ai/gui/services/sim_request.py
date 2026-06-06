# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/services/sim_request.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimRequest + construção de batch + chamada simulate_batch  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — services (spec 0011a)                                ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação (walking skeleton)                     ║
# ║  Framework   : stdlib + numpy PURO — NÃO importa Qt (Princípio X)          ║
# ║  Dependências: numpy; geosteering_ai.simulation (lazy)                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    A parte PURA (sem Qt) da camada de simulação: o ``SimRequest`` (que o   ║
# ║    ViewModel PURO monta) e ``_run_simulation`` (que o ``Worker`` roda).    ║
# ║    Separado de ``simulation_service.py`` (que importa Qt via BaseService)  ║
# ║    para que o ViewModel possa importar ``SimRequest`` SEM puxar Qt.        ║
# ║                                                                           ║
# ║  FIDELIDADE (inviolável)                                                  ║
# ║    ``_run_simulation`` só chama ``simulate_batch`` — não copia kernel nem  ║
# ║    altera ordem de ops (paridade Fortran <1e-12 preservada). Geometria     ║
# ║    FIXA + ``backend="numba"`` no skeleton → JAX não inicializa (TLS-safe). ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimRequest · compute_n_pos                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimRequest`` + construção de batch + ``_run_simulation`` — parte PURA (sem Qt)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

__all__ = ["SimRequest", "compute_n_pos"]


def compute_n_pos(tj: float, p_med: float, dip0_deg: float) -> int:
    """Nº de pontos de medição pela CONVENÇÃO FORTRAN (fonte única da fórmula).

    ``n_pos = max(1, ceil(tj / (p_med · cos(dip0))))`` — o ``cos(dip0)`` projeta o
    passo no eixo vertical (guard ``1e-6`` evita ÷0 em 90°). Usado por
    :func:`_compute_positions_z` e pela property ``n_pos`` do ViewModel.

    Args:
        tj: janela de investigação (m).
        p_med: passo entre medidas (m) — DEVE ser > 0 (é o denominador).
        dip0_deg: 1º ângulo de mergulho (graus).

    Returns:
        int ``≥ 1`` — nº de posições de medição.

    Raises:
        ValueError: se ``p_med <= 0`` (divisão por zero). Hardening de API
            pública — os chamadores internos (ViewModel/View) já guardam, mas a
            função é exportada e não confia no chamador.
    """
    if p_med <= 0.0:
        raise ValueError(f"p_med deve ser > 0 (got {p_med}).")
    cos_d = max(1e-6, math.cos(math.radians(abs(dip0_deg))))
    return max(1, int(math.ceil(tj / (p_med * cos_d))))


@dataclass(frozen=True)
class SimRequest:
    """Requisição de simulação (Fatia 2 — params completos da ParametersPage).

    Attributes:
        frequencies_hz: frequências em Hz (range do simulador [10, 2e6]; default 20 kHz).
        tr_spacings_m: espaçamentos transmissor-receptor em m (range [0.1, 50]; default 1.0).
        dip_degs: ângulos de mergulho em graus (range [0, 90]; default 0°).
        h1: altura do 1º ponto-médio acima da interface (m) — convenção Fortran.
        tj: janela de investigação (m) — extensão da varredura de profundidade.
        p_med: passo entre medidas (m).
        n_models: nº de modelos no batch (pequeno no skeleton; default 2).
        backend: ``"numba"`` no skeleton (evita JAX/TLS); auto/jax = fatia 5.

    Note:
        ``h1``/``tj``/``p_med`` (+ ``dip_degs[0]``) determinam ``positions_z`` pela
        convenção Fortran (ver :func:`_compute_positions_z`).
    """

    frequencies_hz: Tuple[float, ...] = (20000.0,)
    tr_spacings_m: Tuple[float, ...] = (1.0,)
    dip_degs: Tuple[float, ...] = (0.0,)
    h1: float = 1.0
    tj: float = 10.0
    p_med: float = 1.0
    n_models: int = 2
    backend: str = "numba"


# Modelo TIV de referência (3 camadas): ρₕ por camada + λ²=2 (anisotropia branda,
# λ=√2≈1.414 ∈ [1,5]) + 1 espessura interna de 8 m. Valores dentro da errata física
# (ρ ∈ [0.01, 1e6]). Geometria FIXA no skeleton — geração estocástica é fatia futura.
_BASE_RHO_H = np.array([1.0, 10.0, 100.0], dtype=np.float64)  # (n_layers=3,)
_LAMBDA_SQ = 2.0
_INNER_THICKNESS_M = 8.0


def _compute_positions_z(request: SimRequest) -> np.ndarray:
    """Posições de medição ``positions_z`` pela CONVENÇÃO FORTRAN (Fatia 2).

    Replica EXATAMENTE o cálculo do monólito (``simulation_manager.py:~8221``):
    o nº de pontos é ``ceil(tj / (p_med · cos(dip0)))`` e ``z_obs`` vai de ``-h1``
    (acima da 1ª interface) a ``tj - h1`` (abaixo), com as interfaces em ``z=0``.
    O ``cos(dip0)`` projeta o passo no eixo vertical (guard ``1e-6`` evita ÷0 em 90°).

    Args:
        request: a requisição (usa ``h1``, ``tj``, ``p_med`` e ``dip_degs[0]``).

    Returns:
        ``positions_z`` shape ``(n_pos,)`` float64 — ``linspace(-h1, tj-h1, n_pos)``.
    """
    # n_pos via :func:`compute_n_pos` (FONTE ÚNICA da fórmula — não reinlinar).
    n_pos = compute_n_pos(request.tj, request.p_med, request.dip_degs[0])
    return np.linspace(-request.h1, request.tj - request.h1, n_pos, dtype=np.float64)


def _build_batch(
    request: SimRequest,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constrói um batch TIV pequeno e VÁLIDO para ``simulate_batch``.

    Geologia FIXA (3-camadas TIV; ρₕ varia ×(1+0.1·i) por modelo para não degenerar;
    esp = n_layers−2 internas) — geração estocástica é a Fatia 3. ``positions_z`` vem
    da convenção Fortran (:func:`_compute_positions_z`, a partir de h1/tj/p_med/dip).

    Args:
        request: a requisição (usa ``n_models`` + h1/tj/p_med/dip via positions_z).

    Returns:
        ``(rho_h, rho_v, esp, positions_z)`` — shapes ``(n,3)``, ``(n,3)``,
        ``(n,1)``, ``(n_pos,)`` float64.
    """
    n = max(1, int(request.n_models))
    rho_h = np.stack([_BASE_RHO_H * (1.0 + 0.1 * i) for i in range(n)])  # (n, 3)
    rho_v = rho_h * _LAMBDA_SQ  # ρᵥ = ρₕ·λ²  (λ=√2 → anisotropia TIV branda)
    esp = np.full((n, 1), _INNER_THICKNESS_M, dtype=np.float64)  # n_layers−2 = 1
    positions_z = _compute_positions_z(request)
    return rho_h, rho_v, esp, positions_z


def _run_simulation(request: SimRequest) -> Dict[str, Any]:
    """Roda ``simulate_batch`` (na worker thread) e empacota o resultado.

    PURO/picklable (módulo-nível) — chamável direto em teste de fidelidade SEM Qt.
    Importa ``simulate_batch`` lazy (não pesa o import deste módulo).

    Args:
        request: a requisição validada.

    Returns:
        dict com ``H6`` (n_models, nTR, nAng, n_pos, nf, 9) complexo, ``positions_z``,
        ``info`` (dispatcher) e ``backend`` efetivo.

    Note:
        A 1ª chamada dispara o JIT warmup do Numba (~1-30 s); as seguintes são
        rápidas (cache ``.nbc``). DÍVIDA (Fatia 5): o Numba ``prange`` (parallel,
        nogil) usa pool OMP próprio — rodar dentro de uma ``QThread`` funciona
        (pools distintos), mas a arquitetura manda ``ProcessPoolExecutor`` para
        CPU-bound em produção (isola o pool Numba, melhor p/ multi-sim concorrente).
    """
    from geosteering_ai.simulation import simulate_batch  # lazy (Numba pesado)

    rho_h, rho_v, esp, positions_z = _build_batch(request)
    h6, info = simulate_batch(
        rho_h,
        rho_v,
        esp,
        positions_z,
        frequencies_hz=list(request.frequencies_hz),
        tr_spacings_m=list(request.tr_spacings_m),
        dip_degs=list(request.dip_degs),
        backend=request.backend,
    )
    return {
        "H6": h6,
        "positions_z": positions_z,
        "info": info,
        "backend": request.backend,
    }
