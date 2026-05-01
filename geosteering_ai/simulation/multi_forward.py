# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/multi_forward.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — API Multi-TR/Multi-Ângulo (Sprint 11)   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-14 (PR #15)                                        ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x + Numba 0.60+                                    ║
# ║  Dependências: numpy, numba (opcional via forward.py)                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Expõe :func:`simulate_multi` — API de alto nível do simulador Python   ║
# ║    Numba JIT com suporte NATIVO a multi-TR, multi-ângulo, multi-freq,     ║
# ║    F6 (compensação midpoint CDR) e F7 (antenas inclinadas).               ║
# ║                                                                           ║
# ║    Matematicamente equivalente ao Fortran `perfila1DanisoOMP` v10.0:      ║
# ║      • Loop externo nTR (compatível com `do itr=1,nTR`)                   ║
# ║      • Loop médio ntheta (compatível com `do k=1,ntheta`)                 ║
# ║      • Loop interno n_pos (compatível com `do j=1,nmed(k)`)               ║
# ║      • Dedup de common_arrays por hordist (Fortran cache Fase 4)          ║
# ║                                                                           ║
# ║  FLUXO                                                                    ║
# ║    ┌──────────────────────────────────────────────────────────────────┐ ║
# ║    │  simulate_multi(rho_h, rho_v, esp, positions_z,                  │ ║
# ║    │                 tr_spacings_m=[0.5, 1.0, 1.5],                   │ ║
# ║    │                 dip_degs=[0, 30, 60],                            │ ║
# ║    │                 frequencies_hz=[20000, 200000],                  │ ║
# ║    │                 use_compensation=True, comp_pairs=((0, 2),))     │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  1. Deduplica caches por hordist = L·|sin(dip)|                 │ ║
# ║    │     Para dip=0° (poço vertical): 1 cache compartilhado          │ ║
# ║    │     Para multi-ângulo: 1 cache por hordist distinto             │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  2. Loop aninhado (iTR, iAng):                                  │ ║
# ║    │     reusa cache[hordist]; chama _simulate_positions_njit_cached │ ║
# ║    │     que paraleliza via prange sobre n_pos                       │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  3. Pós-processamento opt-in:                                   │ ║
# ║    │     F6 → apply_compensation(H_tensor, comp_pairs)               │ ║
# ║    │     F7 → apply_tilted_antennas(H_tensor, tilted_configs)        │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  MultiSimulationResult                                          │ ║
# ║    │    .H_tensor: (nTR, nAngles, n_pos, nf, 9) complex128           │ ║
# ║    │    .z_obs: (nAngles, n_pos) float64                             │ ║
# ║    │    .rho_h_at_obs / .rho_v_at_obs: (nAngles, n_pos)              │ ║
# ║    │    .H_comp / .phase_diff_deg / .atten_db (se F6 ativo)          │ ║
# ║    │    .H_tilted (se F7 ativo)                                      │ ║
# ║    └──────────────────────────────────────────────────────────────────┘ ║
# ║                                                                           ║
# ║  DEDUPLICAÇÃO DE CACHE                                                    ║
# ║    `precompute_common_arrays_cache` retorna 9 arrays dependentes APENAS  ║
# ║    de (hordist, freqs, eta). Ângulos com mesmo |sin(dip)| e mesmo L       ║
# ║    compartilham cache. Isto cobre:                                       ║
# ║      • Poço vertical (dip=0°, todos TR): hordist=0 uniforme, 1 cache    ║
# ║      • Multi-dip em TR fixo: K caches para K ângulos distintos          ║
# ║    Resultado: custo ≈ O(unique_hordist × n_pos) em vez de                ║
# ║      O(nTR × nAngles × n_pos).                                          ║
# ║                                                                           ║
# ║  PARALELISMO                                                              ║
# ║    Loop externo (iTR, iAng): SERIAL — apenas indexa cache e chama       ║
# ║    kernel. Overhead Python irrelevante.                                  ║
# ║    Loop interno (n_pos): PARALELO via `prange` dentro de                 ║
# ║    `_simulate_positions_njit_cached` — satura os N cores da CPU.        ║
# ║                                                                           ║
# ║  API BACKWARD-COMPAT                                                      ║
# ║    `simulate()` (forward.py) torna-se um shim de 5 linhas chamando      ║
# ║    `simulate_multi()` com `tr_spacings_m=[tr_spacing_m]` e              ║
# ║    `dip_degs=[dip_deg]`, então desembrulha H_tensor[0, 0, ...]           ║
# ║    via `MultiSimulationResult.to_single()`. Zero breaking change.       ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/PerfilaAnisoOmp.f08 (perfila1DanisoOMP, linhas 604-801)║
# ║    • docs/reference/sprint_11_multi_tr_angle_numba.md (plano completo)  ║
# ║    • postprocess/compensation.py (F6)                                   ║
# ║    • postprocess/tilted.py (F7)                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""API multi-TR + multi-ângulo do simulador Python (Sprint 11).

Expõe :func:`simulate_multi` — ponto único de entrada para uma simulação
forward EM 1D TIV **fielmente equivalente** ao Fortran `tatu.x` v10.0, com
loops nativos sobre nTR × nAngles × nfreq × n_pos, deduplicação de cache
por hordist, e wiring F6/F7 opt-in.

Example:
    Multi-TR com compensação CDR::

        >>> import numpy as np
        >>> from geosteering_ai.simulation import simulate_multi
        >>> result = simulate_multi(
        ...     rho_h=np.array([1.0, 100.0, 1.0]),
        ...     rho_v=np.array([1.0, 100.0, 1.0]),
        ...     esp=np.array([5.0]),
        ...     positions_z=np.linspace(-2, 7, 100),
        ...     tr_spacings_m=[0.5, 1.0, 1.5],
        ...     dip_degs=[0.0],
        ...     frequencies_hz=[20000.0],
        ...     use_compensation=True,
        ...     comp_pairs=((0, 2),),
        ... )
        >>> result.H_tensor.shape       # (nTR, nAngles, n_pos, nf, 9)
        (3, 1, 100, 1, 9)
        >>> result.H_comp.shape          # (n_pairs, nAngles, n_pos, nf, 9)
        (1, 1, 100, 1, 9)

Note:
    Para paridade byte-exata com Fortran `.dat`, use
    :func:`io.binary_dat_multi.export_multi_tr_dat` após `simulate_multi`.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Re-usa helpers e kernels da implementação single (forward.py Sprint 2.10)
from geosteering_ai.simulation._numba.geometry import _sanitize_profile_kernel
from geosteering_ai.simulation._numba.kernel import (
    _fields_in_freqs_kernel,
    _fields_in_freqs_kernel_cached,
    compute_zrho,
    fields_in_freqs,
    precompute_common_arrays_cache,
)
from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.filters import FilterLoader

# F6/F7 pós-processadores (já implementados em Sprint 2.2)
from geosteering_ai.simulation.postprocess.compensation import apply_compensation
from geosteering_ai.simulation.postprocess.tilted import apply_tilted_antennas

# Numba detection (import defensivo — idem forward.py)
try:
    import numba  # type: ignore[import-not-found]  # noqa: F401

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de dedup
# ──────────────────────────────────────────────────────────────────────────────
# Tolerância numérica para chaves de `_CacheDict`. `hordist` tem magnitude
# típica em [0, 10] m; arredondamento em 12 casas decimais corresponde a
# ~1 ppt (parte por trilhão), muito abaixo do menor `hordist` fisicamente
# distinguível no contexto de perfilagem LWD.
_HORDIST_KEY_DECIMALS: int = 12

# ──────────────────────────────────────────────────────────────────────────────
# Sprint 13.2 (v2.13) — Cache cross-call para precompute_common_arrays_cache
# ──────────────────────────────────────────────────────────────────────────────
# Cache global persistente entre chamadas de `simulate_multi` (opt-in via
# `cache_persistent=True`). Acelera cenários PINN/inversão onde a mesma
# geometria é simulada 50× ou mais com perturbações pequenas — primeira
# chamada paga setup, próximas reutilizam.
#
# Estrutura da chave:
#   (round(hordist, 12), freqs_signature, n_layers, eta_hash, h_hash)
#
# - hordist arredondado a 12 casas decimais (~1 ppt — abaixo do ruído físico)
# - freqs_signature: tuple ordenada de frequências arredondadas a 1 Hz
# - n_layers: número de camadas (invariante de modelo)
# - eta_hash, h_hash: tuple de valores arredondados (12 casas) — garante
#   colisão zero entre modelos com mesma geometria mas perfis distintos
#
# Thread-safety: `_CACHE_LOCK` protege acesso concorrente quando chamadas
# múltiplas de `simulate_multi` rodam em ThreadPoolExecutor (notebooks,
# treino offline, UI responsiva).
#
# Memória: cache cresce indefinidamente com chamadas de geometrias distintas.
# Default `cache_persistent=False` preserva comportamento v2.12 (cache local
# por chamada). Usuário deve chamar `release_numba_cache()` para liberar.
_GLOBAL_HORDIST_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, ...]] = {}
_CACHE_LOCK = threading.Lock()


def _build_cache_key(
    hordist: float,
    freqs_hz: np.ndarray,
    n: int,
    h_arr: np.ndarray,
    eta: np.ndarray,
) -> Tuple[Any, ...]:
    """Constrói chave hashable para `_GLOBAL_HORDIST_CACHE` (Sprint 13.2 v2.13).

    A chave codifica TODAS as entradas que afetam o resultado de
    `precompute_common_arrays_cache`:

    - ``hordist``: distância horizontal arredondada a 12 casas decimais
    - ``freqs_signature``: tuple ordenada de frequências (Hz, int rounded)
    - ``n``: número de camadas
    - ``eta_hash``: bytes da view contígua de ``eta`` (n×2 complex128)
    - ``h_hash``: bytes da view contígua de ``h_arr`` (n float64)

    Retornar bytes (em vez de tuple float) garante detecção bit-exata de
    qualquer variação numérica. A chave é hashable e picklable.

    Args:
        hordist: Distância horizontal (m).
        freqs_hz: (nf,) — frequências em Hz.
        n: Número de camadas.
        h_arr: (n,) — espessuras (m).
        eta: (n, 2) — condutividades complexas.

    Returns:
        Tupla hashable única para os argumentos dados.
    """
    hordist_key = round(float(hordist), _HORDIST_KEY_DECIMALS)
    freqs_signature = tuple(int(round(float(f))) for f in freqs_hz)
    eta_bytes = np.ascontiguousarray(eta, dtype=np.complex128).tobytes()
    h_bytes = np.ascontiguousarray(h_arr, dtype=np.float64).tobytes()
    return (hordist_key, freqs_signature, int(n), eta_bytes, h_bytes)


def release_numba_cache() -> int:
    """Libera o cache global de common_arrays (Sprint 13.2 v2.13).

    Chamar quando o usuário terminar uma sessão de simulações com
    ``cache_persistent=True``. Recomendado em ``closeEvent`` de UI
    (espelha ``release_pool()`` da v2.12).

    Thread-safe via ``_CACHE_LOCK``.

    Returns:
        Número de entradas liberadas (útil para diagnóstico / logging).

    Example:
        >>> from geosteering_ai.simulation import release_numba_cache
        >>> n_freed = release_numba_cache()
        >>> print(f"Liberadas {n_freed} entradas de cache.")
    """
    with _CACHE_LOCK:
        n_freed = len(_GLOBAL_HORDIST_CACHE)
        _GLOBAL_HORDIST_CACHE.clear()
    if n_freed > 0:
        logger.debug(
            "release_numba_cache: %d entradas liberadas do _GLOBAL_HORDIST_CACHE",
            n_freed,
        )
    return n_freed


def get_numba_cache_size() -> int:
    """Retorna número atual de entradas em ``_GLOBAL_HORDIST_CACHE`` (Sprint 13.2).

    Útil para diagnóstico, logging, e tests. Thread-safe.

    Returns:
        Número de entradas vivas no cache global.
    """
    with _CACHE_LOCK:
        return len(_GLOBAL_HORDIST_CACHE)


# ──────────────────────────────────────────────────────────────────────────────
# MultiSimulationResult — container de saída multi-dimensional
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MultiSimulationResult:
    """Container com os resultados de uma simulação multi-TR/multi-ângulo.

    Esta é a saída pública de :func:`simulate_multi`. Expõe o tensor H em
    layout de 5 dimensões compatível com o Fortran ``cH1(ntheta, nmmax,
    nf, 9)`` com a dimensão adicional ``nTR`` no início (paridade com
    ``cH_all_tr(nTR, ntheta, nmmax, nf, 9)`` da Feature 1 multi-TR do
    Fortran v10.0).

    Attributes:
        H_tensor: Tensor magnético H em formato 5-D.
            Shape: ``(nTR, nAngles, n_pos, nf, 9)`` complex128.
            Ordem das 9 componentes flat:
            ``[Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz]``.
        z_obs: Profundidades do ponto-médio T-R por (ângulo, posição).
            Shape: ``(nAngles, n_pos)`` float64. Depende de `dip` pois
            a posição vertical do ponto-médio é invariante em TR mas
            varia com ângulo.
        rho_h_at_obs: Resistividade horizontal na camada do ponto-médio.
            Shape: ``(nAngles, n_pos)`` float64.
        rho_v_at_obs: Resistividade vertical na camada do ponto-médio.
            Shape: ``(nAngles, n_pos)`` float64.
        freqs_hz: Frequências da simulação. Shape: ``(nf,)`` float64.
        tr_spacings_m: Espaçamentos T-R. Shape: ``(nTR,)`` float64.
        dip_degs: Ângulos dip. Shape: ``(nAngles,)`` float64.
        cfg: SimulationConfig usado (para rastreabilidade).
        H_comp: (Opcional, se F6 ativo) tensor compensado CDR.
            Shape: ``(n_pairs, nAngles, n_pos, nf, 9)`` complex128.
        phase_diff_deg: (Opcional, se F6) diferença de fase em graus.
            Shape idêntica a H_comp.
        atten_db: (Opcional, se F6) atenuação em dB.
            Shape idêntica a H_comp.
        H_tilted: (Opcional, se F7 ativo) resposta de antenas inclinadas.
            Shape: ``(n_tilted, nTR, nAngles, n_pos, nf)`` complex128.
        unique_hordist_count: (Metadata) número de caches distintos
            criados pela dedup. Útil para benchmark e diagnóstico.

    Example:
        Extração de um corte por TR::

            >>> result = simulate_multi(..., tr_spacings_m=[0.5, 1.0, 1.5])
            >>> H_tr1 = result.H_tensor[1]      # shape (nAngles, n_pos, nf, 9)
            >>> # Equivalente a simulate(tr_spacing_m=1.0)

    Note:
        Container **mutável**, idêntico ao padrão de :class:`SimulationResult`.
        Campos opcionais têm default ``None`` — pós-processamento não-solicitado
        não aloca memória.

        **Ordem de axes**: adota convenção Fortran (`cH_all_tr` indices):
        ``iTR`` é o primeiro eixo, seguido de ``iAng``, `n_pos` e `nf`. Isso
        mapeia diretamente na assinatura de `apply_compensation` que espera
        `(n_tr, ntheta, nmeds, nf, 9)`.
    """

    H_tensor: np.ndarray
    z_obs: np.ndarray
    rho_h_at_obs: np.ndarray
    rho_v_at_obs: np.ndarray
    freqs_hz: np.ndarray
    tr_spacings_m: np.ndarray
    dip_degs: np.ndarray
    cfg: SimulationConfig
    # F6 (opt-in via use_compensation=True)
    H_comp: Optional[np.ndarray] = None
    phase_diff_deg: Optional[np.ndarray] = None
    atten_db: Optional[np.ndarray] = None
    # F7 (opt-in via use_tilted=True)
    H_tilted: Optional[np.ndarray] = None
    # Metadados de diagnóstico (dedup)
    unique_hordist_count: int = 0

    def to_single(self):
        """Desembrulha `(nTR=1, nAngles=1)` em :class:`SimulationResult`.

        Retorna o container single-TR/single-angle usado pela API legada
        :func:`simulate`. Útil para o shim de backward-compat e para
        casos onde o chamador só precisa de 1 corte.

        Returns:
            :class:`SimulationResult` com ``H_tensor.shape == (n_pos, nf, 9)``.

        Raises:
            ValueError: Se `nTR != 1` ou `nAngles != 1`.
        """
        # Import tardio para evitar ciclo de imports forward.py ↔ multi_forward.py
        from geosteering_ai.simulation.forward import SimulationResult

        nTR, nAngles = self.H_tensor.shape[:2]
        if nTR != 1 or nAngles != 1:
            raise ValueError(
                f"to_single() requer nTR=1 e nAngles=1 "
                f"(got nTR={nTR}, nAngles={nAngles})"
            )
        # Corte (0, 0) → shape (n_pos, nf, 9), dados compartilhados (view).
        return SimulationResult(
            H_tensor=self.H_tensor[0, 0],
            z_obs=self.z_obs[0],
            rho_h_at_obs=self.rho_h_at_obs[0],
            rho_v_at_obs=self.rho_v_at_obs[0],
            freqs_hz=self.freqs_hz,
            cfg=self.cfg,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Deduplicação de cache por hordist
# ──────────────────────────────────────────────────────────────────────────────
def _build_unique_hordist_caches(
    tr_spacings_m: np.ndarray,
    dip_rads: np.ndarray,
    n: int,
    npt: int,
    krJ0J1: np.ndarray,
    freqs_hz: np.ndarray,
    h_arr: np.ndarray,
    eta: np.ndarray,
    *,
    cache_persistent: bool = False,
) -> Dict[float, Tuple[np.ndarray, ...]]:
    """Pré-computa caches de common_arrays deduplicados por `hordist`.

    A função ``precompute_common_arrays_cache`` depende APENAS de
    ``(hordist, freqs, eta, h)`` no caminho Sprint 2.10. Como `freqs`,
    `eta`, `h` são constantes durante uma simulação, basta indexar o
    cache por `hordist`.

    Args:
        tr_spacings_m: (nTR,) float64 — espaçamentos T-R em metros.
        dip_rads: (nAngles,) float64 — ângulos em radianos.
        n, npt, krJ0J1, freqs_hz, h_arr, eta: Parâmetros passados
            inalterados a ``precompute_common_arrays_cache``.

    Returns:
        Dicionário ``{hordist_key: cache_tuple}`` onde `hordist_key` é
        `round(hordist, 12)` e `cache_tuple` é o retorno de 9 arrays
        de ``precompute_common_arrays_cache``.

    Note:
        Para poço vertical (`dip=0°` em todas as linhas de `dip_rads`):
        ``sin(0)=0 → hordist=0`` para todos os TR → **1 único cache**.

        Para multi-ângulo: ângulos com ``|sin(θ_i)| == |sin(θ_j)|`` e
        mesmo L compartilham cache. Exemplo: θ=30° e θ=150° geram mesma
        `hordist` → cache único.

        **Colisão física (intencional)**: L=2, θ=30° e L=1, θ=90°
        produzem o mesmo ``hordist=1.0``. Elas SIM compartilham
        `common_arrays` pois a física da propagação depende apenas de
        `hordist`. As diferenças (``dz_half``, ``dip_rad``) entram
        APENAS na geometria TX/RX e na rotação final, que são aplicadas
        no kernel ``_fields_in_freqs_kernel_cached`` via parâmetros
        distintos — o cache é corretamente reutilizado.
    """
    caches: Dict[float, Tuple[np.ndarray, ...]] = {}
    # Sprint 13.2 (v2.13): contadores para diagnóstico do cache cross-call
    n_global_hits = 0
    n_global_misses = 0

    for L in tr_spacings_m:
        for dip_rad in dip_rads:
            # hordist = 2·r_half = 2·(L/2)·|sin(dip)| = L·|sin(dip)|
            hordist = float(L) * abs(math.sin(float(dip_rad)))
            key = round(hordist, _HORDIST_KEY_DECIMALS)
            if key in caches:
                # Já populado nesta chamada (dedup local clássico)
                continue

            # Sprint 13.2: tenta cache global persistente quando opt-in
            if cache_persistent:
                global_key = _build_cache_key(hordist, freqs_hz, n, h_arr, eta)
                with _CACHE_LOCK:
                    cached = _GLOBAL_HORDIST_CACHE.get(global_key)
                if cached is not None:
                    caches[key] = cached
                    n_global_hits += 1
                    continue
                n_global_misses += 1

            cache_tuple = precompute_common_arrays_cache(
                n,
                npt,
                hordist,
                krJ0J1,
                freqs_hz,
                h_arr,
                eta,
            )
            caches[key] = cache_tuple

            # Sprint 13.2: persiste no cache global se opt-in
            if cache_persistent:
                with _CACHE_LOCK:
                    _GLOBAL_HORDIST_CACHE[global_key] = cache_tuple

    if cache_persistent and (n_global_hits or n_global_misses):
        logger.debug(
            "simulate_multi cache_persistent: %d global-hits, %d global-misses, "
            "cache global agora com %d entradas",
            n_global_hits,
            n_global_misses,
            get_numba_cache_size(),
        )

    return caches


# ──────────────────────────────────────────────────────────────────────────────
# Validação de entradas
# ──────────────────────────────────────────────────────────────────────────────
def _validate_multi_inputs(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    tr_spacings_m: Sequence[float],
    dip_degs: Sequence[float],
    use_compensation: bool,
    comp_pairs: Optional[Tuple[Tuple[int, int], ...]],
    use_tilted: bool,
    tilted_configs: Optional[Tuple[Tuple[float, float], ...]],
) -> None:
    """Valida entradas de ``simulate_multi`` (fail-fast)."""
    if rho_h.ndim != 1 or rho_v.ndim != 1:
        raise ValueError(f"rho_h/rho_v devem ser 1-D (got {rho_h.shape}/{rho_v.shape})")
    if rho_h.shape != rho_v.shape:
        raise ValueError(f"rho_h.shape={rho_h.shape} != rho_v.shape={rho_v.shape}")
    n = rho_h.shape[0]
    if n >= 2 and esp.shape[0] != n - 2:
        raise ValueError(f"esp.shape[0]={esp.shape[0]} != n-2={n-2}")
    if len(tr_spacings_m) == 0:
        raise ValueError("tr_spacings_m vazio — requer ao menos 1 espaçamento.")
    if len(dip_degs) == 0:
        raise ValueError("dip_degs vazio — requer ao menos 1 ângulo.")
    # Range físico (mesma validação de SimulationConfig)
    for i, L in enumerate(tr_spacings_m):
        if not (0.1 <= float(L) <= 50.0):
            raise ValueError(f"tr_spacings_m[{i}]={L} m fora do range [0.1, 50.0] m.")
    for i, th in enumerate(dip_degs):
        if not (0.0 <= float(th) <= 90.0):
            raise ValueError(
                f"dip_degs[{i}]={th}° fora do range [0, 90]° "
                f"(paridade Fortran linha 254)."
            )
    # F6
    if use_compensation:
        if len(tr_spacings_m) < 2:
            raise ValueError(
                f"use_compensation=True requer len(tr_spacings_m)>=2 "
                f"(got {len(tr_spacings_m)}). F6 é inerentemente multi-TR."
            )
        if comp_pairs is None or len(comp_pairs) == 0:
            raise ValueError("use_compensation=True requer comp_pairs não-vazio.")
        nTR = len(tr_spacings_m)
        for i, pair in enumerate(comp_pairs):
            if len(pair) != 2:
                raise ValueError(f"comp_pairs[{i}]={pair} deve ter 2 elementos.")
            near, far = pair
            if not (0 <= near < nTR and 0 <= far < nTR):
                raise ValueError(
                    f"comp_pairs[{i}]=({near},{far}) fora do range [0, {nTR})."
                )
            if near == far:
                raise ValueError(
                    f"comp_pairs[{i}]=({near},{far}): near==far é degenerado."
                )
    # F7
    if use_tilted:
        if tilted_configs is None or len(tilted_configs) == 0:
            raise ValueError("use_tilted=True requer tilted_configs não-vazio.")
        for i, cfg_t in enumerate(tilted_configs):
            if len(cfg_t) != 2:
                raise ValueError(
                    f"tilted_configs[{i}]={cfg_t} deve ter 2 elementos (beta, phi)."
                )


# ──────────────────────────────────────────────────────────────────────────────
# simulate_multi — API pública multi-dimensional
# ──────────────────────────────────────────────────────────────────────────────
def simulate_multi(
    rho_h: Optional[np.ndarray] = None,
    rho_v: Optional[np.ndarray] = None,
    esp: Optional[np.ndarray] = None,
    positions_z: Optional[np.ndarray] = None,
    *,
    frequencies_hz: Optional[Sequence[float]] = None,
    tr_spacings_m: Optional[Sequence[float]] = None,
    dip_degs: Optional[Sequence[float]] = None,
    cfg: Optional[SimulationConfig] = None,
    hankel_filter: Optional[str] = None,
    use_compensation: bool = False,
    comp_pairs: Optional[Tuple[Tuple[int, int], ...]] = None,
    use_tilted: bool = False,
    tilted_configs: Optional[Tuple[Tuple[float, float], ...]] = None,
    # ── Sprint 12.1 (v2.12) — Workers Nativos ───────────────────────────
    models: Optional[List[Dict[str, Any]]] = None,
    n_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
    # ── Sprint 13.2 (v2.13) — Cache cross-call ─────────────────────────
    cache_persistent: bool = False,
) -> Union["MultiSimulationResult", "MultiSimulationResultBatch"]:
    """Simulação forward EM 1D TIV com multi-TR, multi-ângulo, multi-freq.

    API multi-dimensional **fielmente equivalente** ao Fortran v10.0:
    combina todas as dimensões de parâmetros da ferramenta (nTR × ntheta
    × nf) em uma única chamada, com deduplicação automática de cache
    por hordist e wiring F6/F7 opcional.

    **Sprint 12.1 (v2.12)** — adiciona suporte nativo a batch
    multi-modelo via 3 kwargs opcionais:

      • ``models``: lista de dicts ``{rho_h, rho_v, esp}`` — quando
        fornecido, ativa o caminho batch (`_workers.run_batch`).
      • ``n_workers``: número de workers do `ProcessPoolExecutor`
        (default = 1, single-process). Requer ``models`` definido.
      • ``threads_per_worker``: threads Numba por worker. ``None`` ativa
        anti-oversubscription auto: ``cpu // n_workers``.

    Quando ``models is None`` (default), o comportamento permanece o
    de v2.11: 1 modelo single-pass. Backward-compat total.

    Args:
        rho_h: (n,) Ω·m — resistividades horizontais por camada (inclui
            os 2 semi-espaços).
        rho_v: (n,) Ω·m — resistividades verticais (mesmo shape de rho_h).
        esp: (n-2,) m — espessuras das camadas internas. Vazio `(0,)`
            para `n==2` ou `n==1` (semi-espaços apenas).
        positions_z: (n_pos,) m — profundidades ponto-médio T-R.
        frequencies_hz: Lista de frequências em Hz. Default:
            `cfg.frequencies_hz` ou `[cfg.frequency_hz]`.
        tr_spacings_m: Lista de espaçamentos T-R em metros. Default:
            `cfg.tr_spacings_m` ou `[cfg.tr_spacing_m]`.
        dip_degs: Lista de ângulos dip em graus. Default: `[0.0]`.
        cfg: SimulationConfig para parâmetros de backend/filtro/threads.
            Se None, usa `SimulationConfig()` default.
        hankel_filter: Override do filtro (`"werthmuller_201pt"`,
            `"kong_61pt"`, `"anderson_801pt"`).
        use_compensation: Ativa F6 (compensação midpoint CDR).
            Requer `len(tr_spacings_m) >= 2`.
        comp_pairs: Tupla de pares `(near_idx, far_idx)` 0-based.
            Obrigatório se `use_compensation=True`.
        use_tilted: Ativa F7 (antenas inclinadas).
        tilted_configs: Tupla de pares `(beta_deg, phi_deg)`.
            Obrigatório se `use_tilted=True`.

    Returns:
        :class:`MultiSimulationResult` com tensor H 5-D e pós-processamento
        opcional. Ver docstring do dataclass.

    Raises:
        ValueError: Se entradas violarem shapes, ranges físicos, ou
            pré-requisitos F6/F7.

    Note:
        **Paridade Fortran**: o layout de loops (TR → ângulo → posição)
        segue exatamente `perfila1DanisoOMP` (PerfilaAnisoOmp.f08:604-801).
        A dedup de cache por `hordist` reproduz o cache Fase 4 do Fortran
        (`u_cache(npt, n, nf, ntheta)`) com economia adicional via
        hash por `hordist` em vez de por `(itr, k)`.

        **Performance**: custo total ≈
        ``O(unique_hordist × precompute_cache) + O(nTR × nAngles × n_pos)``.
        Para `dip_degs=[0.0]` (poço vertical): ``unique_hordist=1``.

    Example:
        Multi-TR (nTR=3) com F6::

            >>> import numpy as np
            >>> from geosteering_ai.simulation import simulate_multi
            >>> result = simulate_multi(
            ...     rho_h=np.array([1.0, 100.0, 1.0]),
            ...     rho_v=np.array([1.0, 100.0, 1.0]),
            ...     esp=np.array([5.0]),
            ...     positions_z=np.linspace(-2, 7, 100),
            ...     tr_spacings_m=[0.5, 1.0, 1.5],
            ...     frequencies_hz=[20000.0],
            ...     use_compensation=True,
            ...     comp_pairs=((0, 2),),
            ... )
            >>> result.H_tensor.shape
            (3, 1, 100, 1, 9)
            >>> result.H_comp is not None
            True

        Sprint 12.1 (v2.12) — Batch multi-modelo com 4 workers::

            >>> models = [
            ...     {"rho_h": np.array([1.0, 10.0+i, 1.0]),
            ...      "rho_v": np.array([1.0, 10.0+i, 1.0]),
            ...      "esp": np.array([5.0])} for i in range(8)
            ... ]
            >>> result = simulate_multi(
            ...     models=models,
            ...     positions_z=np.linspace(-2, 7, 100),
            ...     n_workers=4, threads_per_worker=2,
            ... )
            >>> result.H_stack.shape
            (8, 1, 1, 100, 1, 9)
            >>> result.mode
            'D'
    """
    # ── Sprint 12.1 (v2.12) — Dispatcher: batch multi-modelo ──────
    # Quando `models` é fornecido, delega para `_workers.run_batch`
    # que orquestra o ProcessPoolExecutor (Modos C/D) ou execução
    # sequencial in-process (Modos A/B). Backward-compat total:
    # `models is None` mantém o caminho single-modelo abaixo.
    if models is not None:
        if rho_h is not None or rho_v is not None or esp is not None:
            raise ValueError(
                "simulate_multi(models=[...]) é mutuamente exclusivo com "
                "rho_h/rho_v/esp. Forneça apenas um dos modos: "
                "(a) single: rho_h=..., rho_v=..., esp=...; "
                "(b) batch: models=[...]."
            )
        if positions_z is None:
            raise ValueError("simulate_multi(models=[...]) requer positions_z explícito.")
        # Import lazy para evitar ciclo na inicialização do módulo.
        from geosteering_ai.simulation._workers import run_batch

        # Resolve cfg para extrair backend/hankel_filter (passados ao pool).
        _cfg = cfg if cfg is not None else SimulationConfig()
        _filter = hankel_filter if hankel_filter is not None else _cfg.hankel_filter
        # Resolve n_workers/threads (kwargs explícitos vencem cfg).
        _n_workers = (
            n_workers
            if n_workers is not None
            else (_cfg.n_workers if _cfg.n_workers is not None else 1)
        )
        _threads = (
            threads_per_worker
            if threads_per_worker is not None
            else _cfg.threads_per_worker
        )
        # Kwargs comuns repassados a `simulate_multi` dentro de cada worker.
        # Code-review fix P0 #2 (pickling): coerção `list(...)` para
        # frequencies_hz/tr_spacings_m/dip_degs. `Sequence[float]` aceita
        # generators/maps que não são picklables — quando o ProcessPool
        # serializa `sim_kwargs`, falharia silenciosamente em workers C/D.
        # `list()` converte iteráveis arbitrários em estrutura picklável.
        sim_kwargs: Dict[str, Any] = {
            "positions_z": np.asarray(positions_z, dtype=np.float64),
            "frequencies_hz": (
                list(frequencies_hz) if frequencies_hz is not None else None
            ),
            "tr_spacings_m": (list(tr_spacings_m) if tr_spacings_m is not None else None),
            "dip_degs": list(dip_degs) if dip_degs is not None else None,
            "cfg": _cfg,
            "hankel_filter": hankel_filter,
            "use_compensation": use_compensation,
            "comp_pairs": comp_pairs,
            "use_tilted": use_tilted,
            "tilted_configs": tilted_configs,
        }
        return run_batch(
            models=models,
            sim_kwargs=sim_kwargs,
            n_workers=int(_n_workers),
            threads_per_worker=_threads,
            backend=_cfg.backend,
            hankel_filter=str(_filter),
        )

    # ── Caminho single-modelo (v2.11 e anteriores) ────────────────
    # `rho_h`, `rho_v`, `esp`, `positions_z` são obrigatórios aqui.
    if rho_h is None or rho_v is None or esp is None or positions_z is None:
        raise ValueError(
            "simulate_multi single-modelo requer rho_h, rho_v, esp e "
            "positions_z não-None. Para batch, use models=[...]."
        )

    # ── Config + defaults ──────────────────────────────────────────
    if cfg is None:
        cfg = SimulationConfig()

    # Frequências: lista ou default scalar
    if frequencies_hz is not None:
        freqs_hz = np.asarray(list(frequencies_hz), dtype=np.float64)
    elif cfg.frequencies_hz is not None and len(cfg.frequencies_hz) > 0:
        freqs_hz = np.asarray(list(cfg.frequencies_hz), dtype=np.float64)
    else:
        freqs_hz = np.asarray([cfg.frequency_hz], dtype=np.float64)

    # TR spacings: lista ou default scalar
    if tr_spacings_m is not None:
        tr_arr = np.asarray(list(tr_spacings_m), dtype=np.float64)
    elif cfg.tr_spacings_m is not None and len(cfg.tr_spacings_m) > 0:
        tr_arr = np.asarray(list(cfg.tr_spacings_m), dtype=np.float64)
    else:
        tr_arr = np.asarray([cfg.tr_spacing_m], dtype=np.float64)

    # Ângulos: lista ou default [0.0]
    if dip_degs is not None:
        dip_arr = np.asarray(list(dip_degs), dtype=np.float64)
    else:
        dip_arr = np.asarray([0.0], dtype=np.float64)

    # Filtro
    filter_name = hankel_filter if hankel_filter is not None else cfg.hankel_filter

    # ── Normalização dos arrays geológicos ─────────────────────────
    rho_h = np.ascontiguousarray(rho_h, dtype=np.float64)
    rho_v = np.ascontiguousarray(rho_v, dtype=np.float64)
    esp = np.ascontiguousarray(esp, dtype=np.float64)
    positions_z = np.ascontiguousarray(positions_z, dtype=np.float64)

    # ── Validação ─────────────────────────────────────────────────
    _validate_multi_inputs(
        rho_h,
        rho_v,
        esp,
        tr_arr.tolist(),
        dip_arr.tolist(),
        use_compensation,
        comp_pairs,
        use_tilted,
        tilted_configs,
    )

    # Backend check (Numba caminho preferido; serial fallback para n_pos<=1)
    if cfg.backend not in ("numba", "fortran_f2py"):
        raise NotImplementedError(
            f"Backend {cfg.backend!r} não suportado em simulate_multi. "
            f"Use 'numba' ou 'fortran_f2py' (que usa Numba fallback)."
        )

    # ── Setup de geometria estática (comum a todas as simulações) ──
    n = rho_h.shape[0]
    n_pos = positions_z.shape[0]
    nf = freqs_hz.shape[0]
    nTR = tr_arr.shape[0]
    nAngles = dip_arr.shape[0]

    logger.debug(
        "simulate_multi: n=%d n_pos=%d nf=%d nTR=%d nAngles=%d F6=%s F7=%s",
        n,
        n_pos,
        nf,
        nTR,
        nAngles,
        use_compensation,
        use_tilted,
    )

    # Perfil sanitizado (idêntico a forward.py)
    if n == 1:
        h_arr_static = np.zeros(1, dtype=np.float64)
        prof_arr_static = np.array([-1.0e300, 1.0e300], dtype=np.float64)
    else:
        h_arr_static, prof_arr_static = _sanitize_profile_kernel(n, esp)

    # eta estático (condutividades)
    eta_static = np.empty((n, 2), dtype=np.float64)
    for i_lay in range(n):
        eta_static[i_lay, 0] = 1.0 / rho_h[i_lay]
        eta_static[i_lay, 1] = 1.0 / rho_v[i_lay]

    # Filtro Hankel
    filt = FilterLoader().load(filter_name)
    krJ0J1 = filt.abscissas
    wJ0 = filt.weights_j0
    wJ1 = filt.weights_j1
    npt = krJ0J1.shape[0]

    # ── Ângulos em radianos (pre-conversão, reutilizada) ──────────
    dip_rads = np.deg2rad(dip_arr)

    # ── Deduplicação de caches por hordist ─────────────────────────
    # Chama a versão Sprint 2.10 + 13.2: um cache por `hordist` único.
    # Sprint 13.2 (v2.13): se `cache_persistent=True`, reutiliza entradas
    # do cache global (`_GLOBAL_HORDIST_CACHE`) entre chamadas para
    # acelerar PINN/inversão (50× chamadas com mesma geometria).
    caches = _build_unique_hordist_caches(
        tr_arr,
        dip_rads,
        n,
        npt,
        krJ0J1,
        freqs_hz,
        h_arr_static,
        eta_static,
        cache_persistent=cache_persistent,
    )
    logger.debug(
        "simulate_multi: deduplicação produziu %d cache(s) distinto(s) "
        "para %d combinações (TR × ângulo)",
        len(caches),
        nTR * nAngles,
    )

    # ── Pré-alocação dos tensores de saída ─────────────────────────
    H_tensor = np.empty((nTR, nAngles, n_pos, nf, 9), dtype=np.complex128)
    z_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    rho_h_at_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    rho_v_at_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    # Flag para evitar recomputar z_obs quando vários TR compartilham o mesmo ângulo
    z_obs_filled = np.zeros(nAngles, dtype=bool)

    # ── Loop principal: iTR × iAng (serial-outer, paralelo-inner) ──
    # Import tardio para evitar import circular via forward.py no teste.
    # v2.15 P1: bloco único (eliminado import duplicado de _simulate_combined_prange).
    # _simulate_combined_prange é o dispatcher Sprint 13.3 (prange flat TR×ang).
    from geosteering_ai.simulation.forward import (
        _simulate_combined_prange,
        _simulate_positions_njit_cached,
        _simulate_positions_parallel,
    )

    # Sprint 2.10: numero de threads (respeita cfg.num_threads)
    if cfg.num_threads > 0 and HAS_NUMBA:
        import numba as _numba

        _numba.set_num_threads(cfg.num_threads)

    # ── Sprint 13.3: Materialização pré-dispatch para prange(nTR*nAngles*n_pos) flat ──
    # Estrutura para despacho adaptativo: usar nova prange quando n_combos >= 2
    n_combos = nTR * nAngles

    if n_combos >= 2 and cfg.parallel and HAS_NUMBA and n_pos > 1:
        # ── Passo 1: Geometria flat ────────────────────────────────────
        dz_halfs = np.empty(n_combos, dtype=np.float64)
        r_halfs = np.empty(n_combos, dtype=np.float64)
        dip_rads_flat = np.empty(n_combos, dtype=np.float64)
        for i_tr in range(nTR):
            L = float(tr_arr[i_tr])
            for i_ang in range(nAngles):
                k = i_tr * nAngles + i_ang
                dip_rad = float(dip_rads[i_ang])
                dz_halfs[k] = 0.5 * L * math.cos(dip_rad)
                r_halfs[k] = 0.5 * L * math.sin(dip_rad)
                dip_rads_flat[k] = dip_rad

        # ── Passo 2: Deduplicação (cache_indices + key_to_idx) ─────────
        unique_keys = list(
            dict.fromkeys(
                round(2.0 * r_halfs[k], _HORDIST_KEY_DECIMALS) for k in range(n_combos)
            )
        )
        key_to_idx = {key: idx for idx, key in enumerate(unique_keys)}
        cache_indices = np.array(
            [
                key_to_idx[round(2.0 * r_halfs[k], _HORDIST_KEY_DECIMALS)]
                for k in range(n_combos)
            ],
            dtype=np.int64,
        )

        # ── Passo 3: Stack de caches únicos [n_unique, nf, npt, n] ─────
        unique_tuples = [caches[key] for key in unique_keys]
        u_unique = np.stack([t[0] for t in unique_tuples], axis=0)
        s_unique = np.stack([t[1] for t in unique_tuples], axis=0)
        uh_unique = np.stack([t[2] for t in unique_tuples], axis=0)
        sh_unique = np.stack([t[3] for t in unique_tuples], axis=0)
        RTEdw_unique = np.stack([t[4] for t in unique_tuples], axis=0)
        RTEup_unique = np.stack([t[5] for t in unique_tuples], axis=0)
        RTMdw_unique = np.stack([t[6] for t in unique_tuples], axis=0)
        RTMup_unique = np.stack([t[7] for t in unique_tuples], axis=0)
        AdmInt_unique = np.stack([t[8] for t in unique_tuples], axis=0)

        # ── Passo 4: Despacho adaptativo Sprint 13.3 — prange flat ──────
        _simulate_combined_prange(
            dz_halfs,
            r_halfs,
            dip_rads_flat,
            cache_indices,
            nAngles,
            u_unique,
            s_unique,
            uh_unique,
            sh_unique,
            RTEdw_unique,
            RTEup_unique,
            RTMdw_unique,
            RTMup_unique,
            AdmInt_unique,
            positions_z,
            n,
            rho_h,
            rho_v,
            h_arr_static,
            prof_arr_static,
            eta_static,
            freqs_hz,
            krJ0J1,
            wJ0,
            wJ1,
            H_tensor,
        )

        # ── Passo 5: z_obs em Python (não hot-path) ───────────────────
        # v2.15 P1 (code review): preserva semântica v2.14 (z_obs amostrado
        # apenas pelo primeiro TR para cada ângulo) eliminando o loop
        # interior O(nTR) seguido de ``break`` imediato. O resultado
        # bit-exato é o mesmo, com complexidade O(nAngles × n_pos) em vez
        # de O(nTR × nAngles × n_pos).
        L0 = float(tr_arr[0])  # primeiro TR — semântica v2.14 preservada
        for i_ang in range(nAngles):
            dip_rad = float(dip_rads[i_ang])
            dz_half = 0.5 * L0 * math.cos(dip_rad)
            for j in range(n_pos):
                z_mid = positions_z[j]
                Tz = z_mid + dz_half
                cz = z_mid - dz_half
                z_obs_j, rh_j, rv_j = compute_zrho(Tz, cz, n, rho_h, rho_v, esp)
                z_obs[i_ang, j] = z_obs_j
                rho_h_at_obs[i_ang, j] = rh_j
                rho_v_at_obs[i_ang, j] = rv_j

    else:
        # ── Fallback: dispatcher serial v2.13 (single-combo ou serial) ──
        for i_tr in range(nTR):
            L = float(tr_arr[i_tr])
            for i_ang in range(nAngles):
                dip_rad = float(dip_rads[i_ang])
                dz_half = 0.5 * L * math.cos(dip_rad)
                r_half = 0.5 * L * math.sin(dip_rad)
                hordist = 2.0 * r_half
                key = round(hordist, _HORDIST_KEY_DECIMALS)
                cache_tuple = caches[key]

                # Output slice para este par (iTR, iAng)
                H_slice = H_tensor[i_tr, i_ang]  # (n_pos, nf, 9)
                z_slice = z_obs[i_ang]
                rh_slice = rho_h_at_obs[i_ang]
                rv_slice = rho_v_at_obs[i_ang]

                # Para (i_ang) já preenchido, use um buffer temporário descartável
                if z_obs_filled[i_ang]:
                    z_tmp = np.empty(n_pos, dtype=np.float64)
                    rh_tmp = np.empty(n_pos, dtype=np.float64)
                    rv_tmp = np.empty(n_pos, dtype=np.float64)
                else:
                    z_tmp = z_slice
                    rh_tmp = rh_slice
                    rv_tmp = rv_slice
                    z_obs_filled[i_ang] = True

                if cfg.parallel and HAS_NUMBA and n_pos > 1:
                    # Caminho paralelo preferido (Sprint 2.10: prange + cache)
                    _simulate_positions_njit_cached(
                        positions_z,
                        dz_half,
                        r_half,
                        dip_rad,
                        n,
                        rho_h,
                        rho_v,
                        esp,
                        h_arr_static,
                        prof_arr_static,
                        eta_static,
                        freqs_hz,
                        krJ0J1,
                        wJ0,
                        wJ1,
                        *cache_tuple,
                        H_slice,
                        z_tmp,
                        rh_tmp,
                        rv_tmp,
                    )
                else:
                    # Caminho serial (debug ou Numba ausente)
                    for j in range(n_pos):
                        z_mid = positions_z[j]
                        # Convenção Fortran (PerfilaAnisoOmp.f08:677-679):
                        # Transmissor ABAIXO do ponto-médio (+z), Receptor ACIMA (−z).
                        # Transmissor em +x, Receptor em −x.
                        Tz = z_mid + dz_half
                        cz = z_mid - dz_half
                        Tx = r_half
                        cx = -r_half
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
                        H_slice[j, :, :] = cH
                        z_obs_j, rh_j, rv_j = compute_zrho(Tz, cz, n, rho_h, rho_v, esp)
                        z_tmp[j] = z_obs_j
                        rh_tmp[j] = rh_j
                        rv_tmp[j] = rv_j

    # ── Construção do resultado base ───────────────────────────────
    result = MultiSimulationResult(
        H_tensor=H_tensor,
        z_obs=z_obs,
        rho_h_at_obs=rho_h_at_obs,
        rho_v_at_obs=rho_v_at_obs,
        freqs_hz=freqs_hz,
        tr_spacings_m=tr_arr,
        dip_degs=dip_arr,
        cfg=cfg,
        unique_hordist_count=len(caches),
    )

    # ── F6: Compensação Midpoint CDR (opt-in) ──────────────────────
    if use_compensation:
        # apply_compensation espera (nTR, ntheta, nmeds, nf, 9) → layout exato de H_tensor
        H_comp, phase_diff_deg, atten_db = apply_compensation(
            H_tensors_per_tr=H_tensor,
            comp_pairs=comp_pairs,
        )
        result.H_comp = H_comp
        result.phase_diff_deg = phase_diff_deg
        result.atten_db = atten_db

    # ── F7: Antenas Inclinadas (opt-in) ────────────────────────────
    if use_tilted:
        # apply_tilted_antennas aceita shape (..., 9) → projeta para (n_tilted, *prefix)
        # Aplicamos sobre H_tensor (nTR, nAngles, n_pos, nf, 9) →
        # saída (n_tilted, nTR, nAngles, n_pos, nf).
        result.H_tilted = apply_tilted_antennas(
            H_tensor=H_tensor,
            tilted_configs=tilted_configs,
        )

    return result


__all__ = ["MultiSimulationResult", "simulate_multi"]
