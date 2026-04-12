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

from geosteering_ai.simulation._numba.kernel import compute_zrho, fields_in_freqs
from geosteering_ai.simulation._numba.propagation import HAS_NUMBA
from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.filters import FilterLoader

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
# _simulate_positions_parallel — Pool de threads sobre posições (Sprint 2.8)
# ──────────────────────────────────────────────────────────────────────────────
# Por que `ThreadPoolExecutor` e não `@njit(parallel=True)`?
#   • `fields_in_freqs` (kernel.py) é uma função Python pura que orquestra
#     chamadas a kernels @njit. Não pode ser chamada de dentro de uma
#     função @njit(parallel=True) sem refatoração profunda.
#   • Solução pragmática: usar threads no nível Python. Durante a execução
#     dos kernels @njit internos (common_arrays, common_factors, hmd_tiv,
#     vmd, rotate_tensor), o GIL é LIBERADO — as threads executam em
#     paralelo de verdade, explorando todos os cores disponíveis.
#   • Vantagem sobre ProcessPoolExecutor: threads compartilham memória,
#     zero serialização de arrays grandes (perfil, filtros, H_tensor).
#
# Cada iteração `j` é independente (sem dependências entre posições),
# escrita em `H_tensor[j, :, :]` é mutuamente exclusiva — sem data race.


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
) -> SimulationResult:
    """Executa uma simulação forward EM 1D TIV completa.

    Ponto único de entrada para o simulador Python. Itera sobre as
    posições do poço `positions_z`, calcula o tensor H em cada posição
    usando `fields_in_freqs` (Sprint 2.4) e retorna um
    :class:`SimulationResult` com todos os resultados.

    Args:
        rho_h: Resistividades horizontais por camada em Ω·m.
            Shape ``(n_layers,)`` float64. Inclui os 2 semi-espaços.
        rho_v: Resistividades verticais por camada em Ω·m.
            Shape ``(n_layers,)`` float64. Deve ter mesmo shape que `rho_h`.
        esp: Espessuras das camadas internas em metros.
            Shape ``(n_layers - 2,)`` float64. Pode ser vazio (shape
            ``(0,)``) para modelos full-space (n=1) ou 2 semi-espaços.
        positions_z: Profundidades ao longo do poço em metros.
            Shape ``(n_positions,)`` float64. Cada entrada é a posição
            vertical do ponto-médio do arranjo T-R (para dip=0°, o TX
            fica em z - L/2 e o RX em z + L/2, onde L é o espaçamento).
        frequency_hz: Frequência de operação em Hz. Se None, usa
            ``cfg.frequency_hz`` (default 20000). Se fornecido,
            sobrescreve ``cfg.frequency_hz``.
        tr_spacing_m: Espaçamento T-R em metros. Se None, usa
            ``cfg.tr_spacing_m`` (default 1.0).
        dip_deg: Dip da ferramenta em graus. 0° = vertical (ferramenta
            alinhada com o eixo z); 90° = horizontal. Default: 0°.
        cfg: SimulationConfig para parâmetros adicionais (backend,
            filtro, threads). Se None, usa `SimulationConfig()` default.
        hankel_filter: Override do filtro Hankel (ex.:
            ``"anderson_801pt"`` para máxima precisão). Se None, usa
            ``cfg.hankel_filter``.

    Returns:
        :class:`SimulationResult` com `H_tensor` shape
        ``(n_positions, nf, 9)`` complex128 e metadados.

    Raises:
        ValueError: Se shapes forem inconsistentes.
        NotImplementedError: Se ``cfg.backend`` não for ``"numba"``
            ou ``"fortran_f2py"`` (backends JAX ainda não implementados).

    Note:
        **Geometria TX/RX**: para um ponto-médio `z_mid` e espaçamento
        `L`, o TX fica em `z_mid - L/2` e o RX em `z_mid + L/2` ao
        longo do eixo vertical. Para dip ≠ 0°, o afastamento horizontal
        é `r = L · sin(dip)` e a separação vertical é `Δz = L · cos(dip)`.

        **Multi-frequência**: se `cfg.frequencies_hz` estiver definido,
        todas as frequências são simuladas em cada posição. Caso contrário,
        usa apenas `frequency_hz` (nf=1).

        **Multi-TR**: se `cfg.tr_spacings_m` estiver definido, cada
        espaçamento TR gera um resultado separado. Não implementado nesta
        Sprint; o resultado é para o TR primário apenas.

    Example:
        Forward simples com 3 camadas::

            >>> import numpy as np
            >>> from geosteering_ai.simulation.forward import simulate
            >>> result = simulate(
            ...     rho_h=np.array([1.0, 100.0, 1.0]),
            ...     rho_v=np.array([1.0, 100.0, 1.0]),
            ...     esp=np.array([5.0]),
            ...     positions_z=np.linspace(-2, 7, 50),
            ...     frequency_hz=20000.0,
            ...     tr_spacing_m=1.0,
            ...     dip_deg=0.0,
            ... )
            >>> result.H_tensor.shape
            (50, 1, 9)
    """
    # ── Config ────────────────────────────────────────────────────
    if cfg is None:
        cfg = SimulationConfig()

    freq = frequency_hz if frequency_hz is not None else cfg.frequency_hz
    L = tr_spacing_m if tr_spacing_m is not None else cfg.tr_spacing_m
    filter_name = hankel_filter if hankel_filter is not None else cfg.hankel_filter

    # Verificar backend (por enquanto só Numba implementado)
    if cfg.backend not in ("numba", "fortran_f2py"):
        raise NotImplementedError(
            f"Backend {cfg.backend!r} ainda não implementado. "
            f"Use 'numba' ou 'fortran_f2py' (que usa Numba fallback)."
        )

    # ── Validação de inputs ───────────────────────────────────────
    rho_h = np.ascontiguousarray(rho_h, dtype=np.float64)
    rho_v = np.ascontiguousarray(rho_v, dtype=np.float64)
    esp = np.ascontiguousarray(esp, dtype=np.float64)
    positions_z = np.ascontiguousarray(positions_z, dtype=np.float64)

    if rho_h.ndim != 1 or rho_v.ndim != 1:
        raise ValueError(f"rho_h/rho_v devem ser 1D: {rho_h.shape}/{rho_v.shape}")
    if rho_h.shape != rho_v.shape:
        raise ValueError(f"rho_h.shape={rho_h.shape} != rho_v.shape={rho_v.shape}")

    n = rho_h.shape[0]
    n_positions = positions_z.shape[0]

    # ── Frequências ───────────────────────────────────────────────
    if cfg.frequencies_hz is not None and len(cfg.frequencies_hz) > 0:
        freqs_hz = np.array(cfg.frequencies_hz, dtype=np.float64)
    else:
        freqs_hz = np.array([freq], dtype=np.float64)
    nf = freqs_hz.shape[0]

    # ── Filtro Hankel ─────────────────────────────────────────────
    filt = FilterLoader().load(filter_name)
    krJ0J1 = filt.abscissas
    wJ0 = filt.weights_j0
    wJ1 = filt.weights_j1

    # ── Geometria (dip → afastamento horizontal e vertical) ───────
    dip_rad = np.deg2rad(dip_deg)
    half_L = L / 2.0
    # Para dip=0° (vertical): TX em z - L/2, RX em z + L/2, r=0 (axial)
    # Para dip>0°: separação vertical reduzida, afastamento horizontal surge
    dz_half = half_L * math.cos(dip_rad)  # metade da separação vertical
    r_half = half_L * math.sin(dip_rad)  # metade do afastamento horizontal

    # ── Pre-alocação ──────────────────────────────────────────────
    H_tensor = np.empty((n_positions, nf, 9), dtype=np.complex128)
    z_obs = np.empty(n_positions, dtype=np.float64)
    rho_h_at_obs = np.empty(n_positions, dtype=np.float64)
    rho_v_at_obs = np.empty(n_positions, dtype=np.float64)

    # ── Loop sobre posições (Sprint 2.8 — paralelo opcional) ──────
    # Se `cfg.parallel=True` e Numba disponível, delega a
    # `_simulate_positions_parallel` (njit + prange). Caso contrário,
    # executa serial no Python puro (debug-friendly, portável).
    logger.debug(
        "simulate: n_positions=%d, nf=%d, n_layers=%d, dip=%.1f°, L=%.2f m, parallel=%s",
        n_positions,
        nf,
        n,
        dip_deg,
        L,
        cfg.parallel,
    )

    if cfg.parallel and HAS_NUMBA and n_positions > 1:
        # Caminho paralelo: ThreadPoolExecutor distribui as n_positions
        # entre workers. Os kernels Numba internos liberam o GIL, dando
        # paralelismo real. Arrays de saída preenchidos inplace.
        _simulate_positions_parallel(
            positions_z,
            dz_half,
            r_half,
            dip_rad,
            n,
            rho_h,
            rho_v,
            esp,
            freqs_hz,
            krJ0J1,
            wJ0,
            wJ1,
            H_tensor,
            z_obs,
            rho_h_at_obs,
            rho_v_at_obs,
            cfg.num_threads,
        )
    else:
        # Caminho serial: executa no Python puro (para debug ou quando
        # Numba não está instalado / parallel foi desabilitado por config).
        for j in range(n_positions):
            z_mid = positions_z[j]

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

    logger.debug(
        "simulate: concluída — H_tensor shape=%s, all finite=%s",
        H_tensor.shape,
        np.all(np.isfinite(H_tensor)),
    )

    return SimulationResult(
        H_tensor=H_tensor,
        z_obs=z_obs,
        rho_h_at_obs=rho_h_at_obs,
        rho_v_at_obs=rho_v_at_obs,
        freqs_hz=freqs_hz,
        cfg=cfg,
    )


__all__ = ["SimulationResult", "simulate"]
