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
# ║    SimRequest                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimRequest`` + construção de batch + ``_run_simulation`` — parte PURA (sem Qt)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

__all__ = ["SimRequest"]


@dataclass(frozen=True)
class SimRequest:
    """Requisição de simulação (FATIA 1 — campos mínimos; cresce nas fatias 2+).

    Attributes:
        frequencies_hz: frequências em Hz (range físico [100, 1e6]; default 20 kHz).
        tr_spacings_m: espaçamentos transmissor-receptor em m (default 1.0).
        dip_degs: ângulos de mergulho em graus (default 0°).
        n_models: nº de modelos no batch (pequeno no skeleton; default 2).
        backend: ``"numba"`` no skeleton (evita JAX/TLS); auto/jax = fatia 5.
    """

    frequencies_hz: Tuple[float, ...] = (20000.0,)
    tr_spacings_m: Tuple[float, ...] = (1.0,)
    dip_degs: Tuple[float, ...] = (0.0,)
    n_models: int = 2
    backend: str = "numba"


# Modelo TIV de referência (3 camadas): ρₕ por camada + λ²=2 (anisotropia branda,
# λ=√2≈1.414 ∈ [1,5]) + 1 espessura interna de 8 m. Valores dentro da errata física
# (ρ ∈ [0.01, 1e6]). Geometria FIXA no skeleton — geração estocástica é fatia futura.
_BASE_RHO_H = np.array([1.0, 10.0, 100.0], dtype=np.float64)  # (n_layers=3,)
_LAMBDA_SQ = 2.0
_INNER_THICKNESS_M = 8.0
_N_POSITIONS = 50


def _build_batch(
    request: SimRequest,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constrói um batch TIV pequeno e VÁLIDO para ``simulate_batch`` (skeleton).

    Cada modelo varia ρₕ levemente (×(1+0.1·i)) para o batch não ser degenerado.
    Espelha a convenção de ``synthetic_generator`` (esp = n_layers−2 internas;
    ``positions_z`` cobre o modelo + margem).

    Args:
        request: a requisição (usa ``n_models``).

    Returns:
        ``(rho_h, rho_v, esp, positions_z)`` — shapes ``(n,3)``, ``(n,3)``,
        ``(n,1)``, ``(50,)`` float64.
    """
    n = max(1, int(request.n_models))
    rho_h = np.stack([_BASE_RHO_H * (1.0 + 0.1 * i) for i in range(n)])  # (n, 3)
    rho_v = rho_h * _LAMBDA_SQ  # ρᵥ = ρₕ·λ²  (λ=√2 → anisotropia TIV branda)
    esp = np.full((n, 1), _INNER_THICKNESS_M, dtype=np.float64)  # n_layers−2 = 1
    total_thick = float(esp.sum(axis=1).max())
    positions_z = np.linspace(-1.0, total_thick + 1.0, _N_POSITIONS)
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
