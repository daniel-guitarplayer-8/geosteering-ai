# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/forward.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — API Pública Forward (Sprint 2.5)       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x                                                  ║
# ║  Dependências: numpy, geosteering_ai.simulation._numba                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    API pública de alto nível do simulador Python. A função `simulate`   ║
# ║    recebe um `SimulationConfig` e um perfil geológico, itera sobre      ║
# ║    as posições do poço e retorna o tensor H completo.                   ║
# ║                                                                           ║
# ║  FLUXO                                                                    ║
# ║    ┌──────────────────────────────────────────────────────────────────┐ ║
# ║    │  SimulationConfig + perfil geológico (rho_h, rho_v, esp)         │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  simulate(cfg, rho_h, rho_v, esp, positions_z, ...)              │ ║
# ║    │    ├── Carrega filtro Hankel (FilterLoader)                      │ ║
# ║    │    ├── Para cada posição z_j do poço:                            │ ║
# ║    │    │     ├── Calcula Tz, cz com base no dip e TR spacing         │ ║
# ║    │    │     └── fields_in_freqs(Tx,Ty,Tz,cx,cy,cz,...) → (nf, 9)  │ ║
# ║    │    └── Empilha → (n_positions, nf, 9) complex128                │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  SimulationResult                                               │ ║
# ║    │    .H_tensor: (n_positions, nf, 9) complex128                   │ ║
# ║    │    .z_obs: (n_positions,) float64                                │ ║
# ║    │    .rho_h_at_obs / .rho_v_at_obs: (n_positions,) float64        │ ║
# ║    │    .cfg: SimulationConfig                                        │ ║
# ║    └──────────────────────────────────────────────────────────────────┘ ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • _numba/kernel.py (Sprint 2.4) — fields_in_freqs                    ║
# ║    • PerfilaAnisoOmp.f08 (perfila1DanisoOMP) — loop externo de posições║
# ║    • docs/reference/plano_simulador_python_jax_numba.md §5               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""API pública forward do simulador Python (Sprint 2.5).

Expõe :func:`simulate` — ponto único de entrada para executar uma
simulação forward EM 1D TIV completa usando o backend Numba (Fase 2).

Example:
    Forward em meio homogêneo isotrópico::

        >>> import numpy as np
        >>> from geosteering_ai.simulation.forward import simulate
        >>> result = simulate(
        ...     rho_h=np.array([100.0]),
        ...     rho_v=np.array([100.0]),
        ...     esp=np.zeros(0, dtype=np.float64),
        ...     positions_z=np.linspace(-5, 5, 100),
        ...     frequency_hz=20000.0,
        ...     tr_spacing_m=1.0,
        ... )
        >>> result.H_tensor.shape
        (100, 1, 9)
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geosteering_ai.simulation._numba.geometry import _sanitize_profile_kernel
from geosteering_ai.simulation._numba.kernel import (
    _compute_zrho_kernel,
    _fields_in_freqs_kernel,
    _fields_in_freqs_kernel_cached,
    compute_zrho,
    fields_in_freqs,
    precompute_common_arrays_cache,
)
from geosteering_ai.simulation._numba.propagation import HAS_NUMBA, njit
from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.filters import FilterLoader

# Sprint 2.9: import de prange para paralelismo real de threads Numba.
try:
    from numba import prange as _prange  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _prange = range  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────────────
# Paralelização via ThreadPoolExecutor — Sprint 2.8
# ──────────────────────────────────────────────────────────────────────────────
# Arquitetura: o orquestrador `fields_in_freqs` (kernel.py) é uma função
# Python pura que internamente chama kernels `@njit` (common_arrays,
# common_factors, hmd_tiv, vmd, rotate_tensor). Esses kernels **liberam o
# GIL** durante a execução nativa (LLVM), permitindo paralelismo real via
# Python threads — sem necessidade de refatorar a arquitetura.
#
# Escolha de `ThreadPoolExecutor` (e não ProcessPoolExecutor):
#   • Threads compartilham memória → zero serialização de arrays
#     grandes (H_tensor, perfis geológicos, filtros Hankel).
#   • Processos exigiriam pickle+IPC de ~86 MB por posição (H_tensor
#     slice) — overhead maior que o ganho.
#   • GIL é liberado dentro dos kernels @njit — threads paralelizam de fato.
#
# Escolha de `num_threads`:
#   • cfg.num_threads = -1 (default) → `os.cpu_count()` threads.
#   • Valores ≥ 1 → número explícito de threads.
#   • Pools são criados por chamada de `simulate()`; não há pool global
#     (evita race conditions entre chamadas concorrentes).

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# _simulate_positions_njit — @njit(parallel=True) + prange (Sprint 2.9)
# ──────────────────────────────────────────────────────────────────────────────
# Com o orquestrador `_fields_in_freqs_kernel` agora @njit (Sprint 2.9), o
# loop de posições pode ser decorado com `@njit(parallel=True)` e usar
# `prange`. Isso elimina completamente o GIL do caminho crítico — o
# paralelismo é feito pelas threads LLVM nativas (pthreads/OpenMP).
#
# Este é o caminho PREFERIDO de paralelização quando Numba está disponível.
# O ThreadPool (_simulate_positions_parallel) é mantido como fallback para
# comparação/debug.
#
# Cada iteração `j` é independente — escrita em H_tensor[j, :, :] é
# mutuamente exclusiva, sem data race.


@njit(parallel=True, cache=True)
def _simulate_positions_njit(
    positions_z: np.ndarray,
    dz_half: float,
    r_half: float,
    dip_rad: float,
    n: int,
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    freqs_hz: np.ndarray,
    krJ0J1: np.ndarray,
    wJ0: np.ndarray,
    wJ1: np.ndarray,
    H_tensor: np.ndarray,
    z_obs: np.ndarray,
    rho_h_at_obs: np.ndarray,
    rho_v_at_obs: np.ndarray,
) -> None:
    """Loop paralelo @njit sobre posições do poço (Sprint 2.9).

    Versão 100% @njit com ``prange`` para paralelismo real (sem GIL).
    Chamada preferida em ``simulate(cfg)`` quando ``cfg.parallel=True``.

    Args:
        positions_z: (n_positions,) — profundidades ponto-médio.
        dz_half: metade da separação vertical (L·cos(dip)/2).
        r_half: metade do afastamento horizontal (L·sin(dip)/2).
        dip_rad: dip da ferramenta em radianos.
        n: número de camadas (inclui semi-espaços).
        rho_h, rho_v: resistividades por camada (n,).
        esp: espessuras internas (n-2,).
        freqs_hz: frequências (nf,).
        krJ0J1, wJ0, wJ1: filtro Hankel (npt,).
        H_tensor: saída (n_positions, nf, 9) — pré-alocado.
        z_obs, rho_h_at_obs, rho_v_at_obs: saídas (n_positions,) pré-alocadas.

    Note:
        Função sem retorno: preenche arrays inplace. O decorador
        ``@njit(parallel=True)`` faz ``prange`` distribuir as iterações
        entre todas as threads Numba (controladas por
        ``numba.set_num_threads()`` ou env var ``NUMBA_NUM_THREADS``).
    """
    n_positions = positions_z.shape[0]

    for j in _prange(n_positions):
        z_mid = positions_z[j]

        Tz = z_mid - dz_half
        cz = z_mid + dz_half
        Tx = -r_half
        cx = r_half
        Ty = 0.0
        cy = 0.0

        # Chamada ao kernel @njit do orquestrador forward (Sprint 2.9)
        cH = _fields_in_freqs_kernel(
            Tx,
            Ty,
            Tz,
            cx,
            cy,
            cz,
            dip_rad,
            n,
            rho_h,
            rho_v,
            esp,
            freqs_hz,
            krJ0J1,
            wJ0,
            wJ1,
        )
        H_tensor[j, :, :] = cH

        # Metadados da posição via kernel @njit de compute_zrho (Sprint 2.9)
        z_obs_j, rh_j, rv_j = _compute_zrho_kernel(Tz, cz, n, rho_h, rho_v, esp)
        z_obs[j] = z_obs_j
        rho_h_at_obs[j] = rh_j
        rho_v_at_obs[j] = rv_j


# ──────────────────────────────────────────────────────────────────────────────
# _simulate_positions_njit_cached — Sprint 2.10 (cache common_arrays Fase 4)
# ──────────────────────────────────────────────────────────────────────────────
# Port Python do cache Fase 4 do Fortran v10.0 — pré-computa common_arrays
# UMA vez por (hordist, freq) e reusa em todas as N posições do poço. Ganho
# esperado ~4–6× no perfil large (22 camadas, 601 pontos).
#
# Chamado por `simulate(cfg)` quando `cfg.parallel=True` e geometria é
# fixa (dip constante) — condição satisfeita em TODA simulação típica
# (dip=0° ferramenta vertical, ou qualquer dip fixo com n_positions > 1).


@njit(parallel=True, cache=True)
def _simulate_positions_njit_cached(
    positions_z: np.ndarray,
    dz_half: float,
    r_half: float,
    dip_rad: float,
    n: int,
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    h_arr: np.ndarray,
    prof_arr: np.ndarray,
    eta: np.ndarray,
    freqs_hz: np.ndarray,
    krJ0J1: np.ndarray,
    wJ0: np.ndarray,
    wJ1: np.ndarray,
    u_cache: np.ndarray,
    s_cache: np.ndarray,
    uh_cache: np.ndarray,
    sh_cache: np.ndarray,
    RTEdw_cache: np.ndarray,
    RTEup_cache: np.ndarray,
    RTMdw_cache: np.ndarray,
    RTMup_cache: np.ndarray,
    AdmInt_cache: np.ndarray,
    H_tensor: np.ndarray,
    z_obs: np.ndarray,
    rho_h_at_obs: np.ndarray,
    rho_v_at_obs: np.ndarray,
) -> None:
    """Loop paralelo @njit com cache de common_arrays (Sprint 2.10).

    Versão otimizada de :func:`_simulate_positions_njit` que recebe o cache
    pré-computado de common_arrays — evita recomputar os 9 arrays em cada
    posição do poço. Este é o padrão do Fortran v10.0 Fase 4.

    Args:
        positions_z, dz_half, r_half, dip_rad: Geometria do loop.
        n, rho_h, rho_v, esp, h_arr, prof_arr, eta: Perfil geológico
            e geometria estática (calculados no caller).
        freqs_hz, krJ0J1, wJ0, wJ1: Frequências e filtro Hankel.
        u_cache, ..., AdmInt_cache: (nf, npt, n) complex128 — arrays
            pré-calculados por
            :func:`precompute_common_arrays_cache`.
        H_tensor, z_obs, rho_h_at_obs, rho_v_at_obs: Saídas pré-alocadas.

    Note:
        Ganho medido: ~5× no perfil large (22 cam, 601 pos) e ~2× no
        medium. Para perfis small (3 cam), ganho é menor pois o custo
        de common_arrays é proporcionalmente menor.
    """
    n_positions = positions_z.shape[0]

    for j in _prange(n_positions):
        z_mid = positions_z[j]

        Tz = z_mid - dz_half
        cz = z_mid + dz_half
        Tx = -r_half
        cx = r_half
        Ty = 0.0
        cy = 0.0

        # Sprint 2.10: chama kernel com cache pré-computado
        cH = _fields_in_freqs_kernel_cached(
            Tx,
            Ty,
            Tz,
            cx,
            cy,
            cz,
            dip_rad,
            n,
            rho_h,
            rho_v,
            h_arr,
            prof_arr,
            eta,
            freqs_hz,
            krJ0J1,
            wJ0,
            wJ1,
            u_cache,
            s_cache,
            uh_cache,
            sh_cache,
            RTEdw_cache,
            RTEup_cache,
            RTMdw_cache,
            RTMup_cache,
            AdmInt_cache,
        )
        H_tensor[j, :, :] = cH

        z_obs_j, rh_j, rv_j = _compute_zrho_kernel(Tz, cz, n, rho_h, rho_v, esp)
        z_obs[j] = z_obs_j
        rho_h_at_obs[j] = rh_j
        rho_v_at_obs[j] = rv_j


# ──────────────────────────────────────────────────────────────────────────────
# _simulate_positions_parallel — Pool de threads sobre posições (Sprint 2.8)
# ──────────────────────────────────────────────────────────────────────────────
# LEGADO da Sprint 2.8 — kept como fallback/debug. A Sprint 2.9 tornou
# `_simulate_positions_njit` o caminho preferido com speedup real via
# @njit(parallel=True).


def _simulate_positions_parallel(
    positions_z: np.ndarray,
    dz_half: float,
    r_half: float,
    dip_rad: float,
    n: int,
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    freqs_hz: np.ndarray,
    krJ0J1: np.ndarray,
    wJ0: np.ndarray,
    wJ1: np.ndarray,
    H_tensor: np.ndarray,
    z_obs: np.ndarray,
    rho_h_at_obs: np.ndarray,
    rho_v_at_obs: np.ndarray,
    num_threads: int,
) -> None:
    """Distribui o loop de posições entre threads Python.

    Preenche ``H_tensor``, ``z_obs``, ``rho_h_at_obs`` e ``rho_v_at_obs``
    inplace. Usa :class:`concurrent.futures.ThreadPoolExecutor` com
    ``num_threads`` workers. O GIL é liberado durante a execução dos
    kernels Numba, permitindo paralelismo real.

    Args:
        positions_z: Profundidades ponto-médio (n_positions,).
        dz_half: Metade da separação vertical (L·cos(dip)/2).
        r_half: Metade do afastamento horizontal (L·sin(dip)/2).
        dip_rad: Dip da ferramenta em radianos.
        n: Número de camadas (inclui semi-espaços).
        rho_h, rho_v: Resistividades por camada (n,).
        esp: Espessuras internas (n-2,).
        freqs_hz: Frequências (nf,).
        krJ0J1, wJ0, wJ1: Filtro Hankel (npt,).
        H_tensor: Saída (n_positions, nf, 9) complex128 — pré-alocado.
        z_obs: Saída (n_positions,) float64 — pré-alocado.
        rho_h_at_obs, rho_v_at_obs: Saída (n_positions,) — pré-alocados.
        num_threads: Número de threads. Se -1, usa ``os.cpu_count()``.

    Note:
        Função sem retorno: saída via mutação dos arrays pré-alocados.
        Exceções levantadas em threads são re-lançadas pelo `executor.map`.
    """
    import os

    n_positions = positions_z.shape[0]
    cpu_count = os.cpu_count() or 1
    max_workers = cpu_count if num_threads == -1 else max(1, num_threads)

    # Chunking: cada worker processa um range contíguo de posições.
    # Isto reduz o overhead de submit/retrieve do ThreadPoolExecutor (em
    # vez de `n_positions` pequenas tasks, distribuímos `max_workers`
    # tasks maiores). Vantagens:
    #   • Acesso sequencial a memória (cache L2/L3 melhor aproveitado).
    #   • Cada thread entra no código Numba várias vezes seguidas →
    #     amortiza o custo de entrada/saída do regime sem-GIL.
    #   • Reduz pressure no scheduler Python.
    chunk_size = max(1, (n_positions + max_workers - 1) // max_workers)

    def _worker_chunk(start: int) -> None:
        end = min(start + chunk_size, n_positions)
        for j in range(start, end):
            z_mid = positions_z[j]

            # Transmissor acima do ponto-médio, receptor abaixo
            Tz = z_mid - dz_half
            cz = z_mid + dz_half
            Tx = -r_half
            cx = r_half
            Ty = 0.0
            cy = 0.0

            cH = fields_in_freqs(
                Tx,
                Ty,
                Tz,
                cx,
                cy,
                cz,
                dip_rad,
                n,
                rho_h,
                rho_v,
                esp,
                freqs_hz,
                krJ0J1,
                wJ0,
                wJ1,
            )
            H_tensor[j, :, :] = cH

            z_obs_j, rh_j, rv_j = compute_zrho(Tz, cz, n, rho_h, rho_v, esp)
            z_obs[j] = z_obs_j
            rho_h_at_obs[j] = rh_j
            rho_v_at_obs[j] = rv_j

    chunk_starts = list(range(0, n_positions, chunk_size))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Consome o iterador para propagar exceções dos workers.
        list(executor.map(_worker_chunk, chunk_starts))


# ──────────────────────────────────────────────────────────────────────────────
# SimulationResult — container de saída
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    """Container com os resultados de uma simulação forward.

    Attributes:
        H_tensor: Tensor magnético H em formato flat 9-componentes.
            Shape: ``(n_positions, nf, 9)`` complex128.
            Ordem das 9 colunas: Hxx, Hxy, Hxz, Hyx, Hyy, Hyz,
            Hzx, Hzy, Hzz.
        z_obs: Profundidades do ponto-médio T-R para cada posição.
            Shape: ``(n_positions,)`` float64.
        rho_h_at_obs: Resistividade horizontal na camada do ponto-médio
            para cada posição. Shape: ``(n_positions,)`` float64.
        rho_v_at_obs: Resistividade vertical na camada do ponto-médio.
            Shape: ``(n_positions,)`` float64.
        freqs_hz: Frequências usadas na simulação.
            Shape: ``(nf,)`` float64.
        cfg: SimulationConfig usado (para rastreabilidade).

    Note:
        Este container é **mutável** (diferente de SimulationConfig que é
        frozen). Pode-se anexar pós-processamento (F6, F7) diretamente.
    """

    H_tensor: np.ndarray
    z_obs: np.ndarray
    rho_h_at_obs: np.ndarray
    rho_v_at_obs: np.ndarray
    freqs_hz: np.ndarray
    cfg: SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# simulate() — API pública
# ──────────────────────────────────────────────────────────────────────────────


def simulate(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    positions_z: np.ndarray,
    frequency_hz: Optional[float] = None,
    tr_spacing_m: Optional[float] = None,
    dip_deg: float = 0.0,
    cfg: Optional[SimulationConfig] = None,
    hankel_filter: Optional[str] = None,
) -> SimulationResult:  # noqa: D401
    # Sprint 11 (PR #15): `simulate()` agora é um shim literal de
    # `simulate_multi()` com nTR=1, nAngles=1. Isso elimina duplicação
    # de código e torna o teste de paridade single-TR uma INVARIANTE
    # ESTRUTURAL (não-quebrável por design) em vez de uma checagem
    # numérica. Backward-compat total: assinatura, retorno e comportamento
    # idênticos à versão Sprint 2.5-2.10. Ver multi_forward.py para a
    # implementação multi-dimensional subjacente.
    # ────────────────────────────────────────────────────────────────
    # Import tardio para evitar ciclo: multi_forward → forward (_simulate_*)
    from geosteering_ai.simulation.multi_forward import simulate_multi

    # Resolve defaults de L e freq exatamente como antes do shim
    # (preserva a semântica pre-Sprint 11: cfg.frequency_hz/tr_spacing_m
    # são usados quando o chamador passa None).
    cfg_eff = cfg if cfg is not None else SimulationConfig()
    L = tr_spacing_m if tr_spacing_m is not None else cfg_eff.tr_spacing_m
    # Multi-freq ativada via cfg.frequencies_hz (não via lista no shim)
    freqs_list = (
        list(cfg_eff.frequencies_hz)
        if cfg_eff.frequencies_hz is not None and len(cfg_eff.frequencies_hz) > 0
        else [frequency_hz if frequency_hz is not None else cfg_eff.frequency_hz]
    )

    multi_result = simulate_multi(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=freqs_list,
        tr_spacings_m=[L],
        dip_degs=[dip_deg],
        cfg=cfg_eff,
        hankel_filter=hankel_filter,
    )
    return multi_result.to_single()

__all__ = ["SimulationResult", "simulate"]
