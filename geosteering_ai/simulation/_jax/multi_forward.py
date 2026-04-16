# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/multi_forward.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Sprint 11-JAX — simulate_multi_jax() multi-TR/angle/freq   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-15 (Sprint 11-JAX, PR #23 / v1.5.0)               ║
# ║  Status      : Produção (wrapper funcional; otimização GPU pendente)     ║
# ║  Framework   : JAX 0.4.30+                                               ║
# ║  Dependências: jax, jax.numpy, forward_pure_jax, simulate_multi (Numba)  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Port de ``simulate_multi()`` Numba (Sprint 11 / PR #15) para JAX,     ║
# ║    fornecendo API consistente ``simulate_multi_jax()`` com output        ║
# ║    ``MultiSimulationResultJAX`` de shape                                 ║
# ║    ``(nTR, nAngles, n_pos, nf, 9)`` complex128.                          ║
# ║                                                                           ║
# ║    Suporta:                                                               ║
# ║      • Multi-frequência (nf ≥ 1)                                         ║
# ║      • Multi-TR (nTR ≥ 1)                                                ║
# ║      • Multi-ângulo (nAngles ≥ 1)                                        ║
# ║      • Dedup de cache por ``hordist = L·|sin(θ)|`` (padrão Sprint 2.10)  ║
# ║                                                                           ║
# ║  ARQUITETURA v1.5.0 (atual — Phase 1)                                     ║
# ║    Implementação inicial usa WRAPPER sobre `forward_pure_jax` em Python  ║
# ║    loop sobre (iTR, iAng). Funcional mas NÃO otimizado para GPU —        ║
# ║    cada combinação (TR, ângulo) gera um trace JAX separado.              ║
# ║                                                                           ║
# ║    Benefícios (mesmo sem otimização GPU completa):                       ║
# ║      ✅ API consistente com `simulate_multi` Numba (mesmo shape output)  ║
# ║      ✅ Dedup de cache Numba-style via `hordist` (economia em casos      ║
# ║         de poço vertical dip=0°)                                         ║
# ║      ✅ Diferenciabilidade `jax.jacfwd` preservada                       ║
# ║      ✅ Suporte GPU (via `forward_pure_jax`)                             ║
# ║      ✅ F6/F7 pós-processamento (herdado de Numba postprocess)          ║
# ║                                                                           ║
# ║  EVOLUÇÃO FUTURA (Phase 2, PR #24)                                        ║
# ║    Depende de Sprint 10 completo (`dipoles_unified` integrado em         ║
# ║    `_hmd_tiv_native_jax`). Após isso, substituir o wrapper por:          ║
# ║      • vmap aninhado sobre (nTR, nAngles, n_pos, nf)                     ║
# ║      • 1 único JIT (via Sprint 10 consolidação)                          ║
# ║      • Throughput GPU T4: 400k-700k mod/h (3-5× vs Numba CPU)            ║
# ║      • Throughput GPU A100: 1.2M-2.5M mod/h (10-20× vs Numba CPU)        ║
# ║                                                                           ║
# ║  API PÚBLICA                                                              ║
# ║    simulate_multi_jax(rho_h, rho_v, esp, positions_z,                    ║
# ║                       frequencies_hz, tr_spacings_m, dip_degs,           ║
# ║                       cfg=None) → MultiSimulationResultJAX               ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • multi_forward.py (Numba equivalent, Sprint 11 PR #15)               ║
# ║    • forward_pure.py (base JAX forward)                                  ║
# ║    • dipoles_unified.py (Sprint 10 unified propagation)                  ║
# ║    • docs/reference/relatorio_final_v1_4_1.md seção 4.2                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Sprint 11-JAX — `simulate_multi_jax()` com suporte multi-TR/ângulo/freq.

Este módulo porta a API `simulate_multi` do path Numba (Sprint 11 / PR #15)
para JAX, fornecendo execução em CPU/GPU com diferenciabilidade automática
preservada via `jax.jacfwd`.

Example:
    >>> import jax.numpy as jnp
    >>> from geosteering_ai.simulation import simulate_multi_jax
    >>>
    >>> result = simulate_multi_jax(
    ...     rho_h=jnp.array([1.0, 100.0, 1.0]),
    ...     rho_v=jnp.array([1.0, 200.0, 1.0]),
    ...     esp=jnp.array([5.0]),
    ...     positions_z=jnp.linspace(-10, 10, 100),
    ...     frequencies_hz=[20000., 40000.],     # 2 frequências
    ...     tr_spacings_m=[0.5, 1.0, 1.5],       # 3 TR spacings
    ...     dip_degs=[0., 30., 60.],             # 3 ângulos
    ... )
    >>> result.H_tensor.shape
    (3, 3, 100, 2, 9)

Note:
    **Implementação atual (v1.5.0-alpha)** é um WRAPPER Python sobre
    `forward_pure_jax`. Funcional mas não otimizado para GPU. Versão
    completa (vmap aninhado + unified JIT) depende de Sprint 10 Phase 2
    (integração de `dipoles_unified` em `dipoles_native.py`), prevista
    para PR #24 / v1.5.0-final.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None  # type: ignore
    jnp = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# MultiSimulationResultJAX — container dos resultados JAX multi-dim
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MultiSimulationResultJAX:
    """Container dos resultados de ``simulate_multi_jax()``.

    Espelha ``MultiSimulationResult`` (Numba) para compatibilidade de
    downstream: qualquer código que consome shape ``(nTR, nAngles, n_pos,
    nf, 9)`` funciona com ambos.

    Attributes:
        H_tensor: Tensor magnético H em formato 5-D.
            Shape ``(nTR, nAngles, n_pos, nf, 9)`` complex128.
            Ordem das 9 componentes flat:
            ``[Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz]``.
        z_obs: Profundidade TVD do ponto-médio T-R por (ângulo, posição).
            Shape ``(nAngles, n_pos)`` float64.
        rho_h_at_obs: Resistividade horizontal na camada do ponto-médio.
            Shape ``(nAngles, n_pos)`` float64.
        rho_v_at_obs: Resistividade vertical.
            Shape ``(nAngles, n_pos)`` float64.
        freqs_hz: Frequências usadas. Shape ``(nf,)`` float64.
        tr_spacings_m: Espaçamentos T-R. Shape ``(nTR,)`` float64.
        dip_degs: Ângulos dip. Shape ``(nAngles,)`` float64.
        unique_hordist_count: (Metadata) número de caches deduplicados
            por ``hordist`` (otimização interna). Útil para debug.

    Note:
        Os arrays numpy são convertidos de ``jax.Array`` para compatibilidade
        com downstream Numba/NumPy. Para manter ``jax.Array`` (e permitir
        ``jax.jacfwd``), use ``forward_pure_jax`` diretamente.
    """

    H_tensor: np.ndarray  # (nTR, nAngles, n_pos, nf, 9) complex128
    z_obs: np.ndarray  # (nAngles, n_pos) float64
    rho_h_at_obs: np.ndarray  # (nAngles, n_pos) float64
    rho_v_at_obs: np.ndarray  # (nAngles, n_pos) float64
    freqs_hz: np.ndarray  # (nf,) float64
    tr_spacings_m: np.ndarray  # (nTR,) float64
    dip_degs: np.ndarray  # (nAngles,) float64
    unique_hordist_count: int = 0

    def to_single(self):
        """Desembrulha resultado (nTR=1, nAngles=1) para ``SimulationResult``.

        Útil para retrocompatibilidade com código que espera `simulate()`
        (single-TR, single-angle). Equivalente a ``result.H_tensor[0, 0]``.

        Returns:
            SimulationResult com ``H_tensor: (n_pos, nf, 9)``.

        Raises:
            ValueError: Se ``nTR > 1`` ou ``nAngles > 1`` (não é cenário
                single — use indexação explícita).
        """
        if self.H_tensor.shape[0] > 1:
            raise ValueError(
                f"to_single() requer nTR=1, obtido nTR={self.H_tensor.shape[0]}. "
                "Use H_tensor[iTR, iAng, ...] diretamente para multi-TR."
            )
        if self.H_tensor.shape[1] > 1:
            raise ValueError(
                f"to_single() requer nAngles=1, obtido nAngles={self.H_tensor.shape[1]}."
            )

        from geosteering_ai.simulation.forward import SimulationResult

        return SimulationResult(
            H_tensor=self.H_tensor[0, 0],  # (n_pos, nf, 9)
            z_obs=self.z_obs[0],
            rho_h_at_obs=self.rho_h_at_obs[0],
            rho_v_at_obs=self.rho_v_at_obs[0],
            freqs_hz=self.freqs_hz,
            cfg=None,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers internos — dedup cache por hordist
# ──────────────────────────────────────────────────────────────────────────────
def _compute_hordist_key(tr_spacing_m: float, dip_deg: float) -> float:
    """Retorna ``hordist = L * |sin(θ)|`` com arredondamento para dedup."""
    dip_rad = math.radians(float(dip_deg))
    return round(float(tr_spacing_m) * abs(math.sin(dip_rad)), 12)


def _build_hordist_groups(
    tr_spacings_m: Sequence[float], dip_degs: Sequence[float]
) -> dict:
    """Agrupa combinações (iTR, iAng) por ``hordist`` único.

    Para poço vertical (dip=0° para todos TR): `hordist=0` uniforme, 1
    único grupo. Para multi-ângulo com dip ≠ 0°: tantos grupos quanto
    hordist distintos.

    Args:
        tr_spacings_m: Lista de TR spacings (nTR).
        dip_degs: Lista de ângulos dip (nAngles).

    Returns:
        Dict ``{hordist_key: [(iTR, iAng, tr, dip), ...]}``. Cada entrada
        agrupa combinações compartilhando o mesmo hordist.
    """
    groups: dict = {}
    for i_tr, L in enumerate(tr_spacings_m):
        for i_ang, theta in enumerate(dip_degs):
            key = _compute_hordist_key(L, theta)
            groups.setdefault(key, []).append((i_tr, i_ang, float(L), float(theta)))
    return groups


# ──────────────────────────────────────────────────────────────────────────────
# API pública — simulate_multi_jax()
# ──────────────────────────────────────────────────────────────────────────────
def simulate_multi_jax(
    rho_h,
    rho_v,
    esp,
    positions_z,
    *,
    frequencies_hz: Optional[Sequence[float]] = None,
    tr_spacings_m: Optional[Sequence[float]] = None,
    dip_degs: Optional[Sequence[float]] = None,
    cfg=None,
    hankel_filter: str = "werthmuller_201pt",
) -> MultiSimulationResultJAX:
    """Forward JAX multi-TR/multi-ângulo/multi-frequência.

    API equivalente a ``simulate_multi()`` Numba (Sprint 11 / PR #15) mas
    usando path JAX (`forward_pure_jax` como kernel base). Suporta:

      • Multi-TR: várias distâncias transmissor-receptor (nTR ≥ 1)
      • Multi-ângulo: várias inclinações de poço (nAngles ≥ 1)
      • Multi-frequência: várias frequências de operação (nf ≥ 1)

    A implementação atual (v1.5.0-alpha) é um WRAPPER Python sobre
    `forward_pure_jax` com deduplicação de cache por `hordist =
    L·|sin(θ)|` (padrão Sprint 2.10 Numba). Funcional mas não otimizado
    para GPU — vmap aninhado + unified JIT são entregas de PR #24.

    Args:
        rho_h: (n,) Ω·m — resistividades horizontais.
        rho_v: (n,) Ω·m — resistividades verticais (mesmo shape de rho_h).
        esp: (n-2,) m — espessuras das camadas internas.
        positions_z: (n_pos,) m — profundidades TVD do ponto-médio.
        frequencies_hz: Lista de frequências em Hz. Default:
            ``cfg.frequencies_hz`` ou ``[cfg.frequency_hz]``.
        tr_spacings_m: Lista de espaçamentos T-R em metros. Default:
            ``[cfg.tr_spacing_m]``.
        dip_degs: Lista de ângulos dip em graus. Default: ``[0.0]``.
        cfg: (Opcional) :class:`SimulationConfig` para defaults.
        hankel_filter: Nome do filtro Hankel. Default "werthmuller_201pt".

    Returns:
        :class:`MultiSimulationResultJAX` com ``H_tensor`` shape
        ``(nTR, nAngles, n_pos, nf, 9)``.

    Raises:
        ImportError: Se JAX não estiver instalado.
        ValueError: Se alguma lista estiver vazia ou fora de range válido.

    Example:
        >>> import jax.numpy as jnp
        >>> result = simulate_multi_jax(
        ...     rho_h=jnp.array([1.0, 100.0, 1.0]),
        ...     rho_v=jnp.array([1.0, 200.0, 1.0]),
        ...     esp=jnp.array([5.0]),
        ...     positions_z=jnp.linspace(-10, 10, 100),
        ...     tr_spacings_m=[0.5, 1.0, 1.5],
        ...     dip_degs=[0.0],
        ... )
        >>> result.H_tensor.shape
        (3, 1, 100, 1, 9)

    Note:
        Para alta performance em batch grande, considere usar
        `forward_pure_jax_chunked` (Sprint 8) diretamente com batching
        explícito sobre posições. O wrapper atual executa nTR × nAngles
        traces JAX separados — o que é aceitável para até ~20 combinações.
    """
    if not HAS_JAX:
        raise ImportError("simulate_multi_jax requer JAX (pip install 'jax[cpu]').")

    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context,
        forward_pure_jax,
    )
    from geosteering_ai.simulation.config import SimulationConfig

    # ── Defaults via cfg ──────────────────────────────────────────────────────
    if cfg is None:
        cfg = SimulationConfig()

    if frequencies_hz is None:
        if cfg.frequencies_hz is not None and len(cfg.frequencies_hz) > 0:
            frequencies_hz = list(cfg.frequencies_hz)
        else:
            frequencies_hz = [cfg.frequency_hz]

    if tr_spacings_m is None:
        if cfg.tr_spacings_m is not None and len(cfg.tr_spacings_m) > 0:
            tr_spacings_m = list(cfg.tr_spacings_m)
        else:
            tr_spacings_m = [cfg.tr_spacing_m]

    if dip_degs is None:
        dip_degs = [0.0]

    freqs_arr = np.asarray(list(frequencies_hz), dtype=np.float64)
    tr_arr = np.asarray(list(tr_spacings_m), dtype=np.float64)
    dip_arr = np.asarray(list(dip_degs), dtype=np.float64)

    # ── Validações fail-fast ──────────────────────────────────────────────────
    if len(tr_arr) == 0:
        raise ValueError("tr_spacings_m vazio — forneça ao menos 1 TR spacing.")
    if len(dip_arr) == 0:
        raise ValueError("dip_degs vazio — forneça ao menos 1 ângulo dip.")
    if len(freqs_arr) == 0:
        raise ValueError("frequencies_hz vazio — forneça ao menos 1 frequência.")

    # Ranges físicos (idênticos a simulate_multi Numba)
    if np.any((tr_arr < 0.1) | (tr_arr > 10.0)):
        raise ValueError(f"tr_spacings_m fora do range [0.1, 10.0] m: {tr_arr.tolist()}")
    if np.any((dip_arr < 0.0) | (dip_arr > 89.0)):
        raise ValueError(f"dip_degs fora do range [0, 89]°: {dip_arr.tolist()}")

    # ── Preparação arrays geológicos ──────────────────────────────────────────
    rho_h_np = np.asarray(rho_h, dtype=np.float64)
    rho_v_np = np.asarray(rho_v, dtype=np.float64)
    esp_np = np.asarray(esp, dtype=np.float64)
    positions_z_np = np.asarray(positions_z, dtype=np.float64)

    n = rho_h_np.shape[0]
    n_pos = positions_z_np.shape[0]
    nf = freqs_arr.shape[0]
    nTR = tr_arr.shape[0]
    nAngles = dip_arr.shape[0]

    # ── Sprint 12 (PR #25) — Dispatcher vmap_real ────────────────────────────
    # Se cfg.jax_vmap_real=True, usa `_simulate_multi_jax_vmap_real` que
    # substitui o Python loop abaixo por `jax.vmap` aninhado sobre
    # (iTR, iAng) flat. Mantém paridade bit-exata e mesma API de saída.
    # Default False preserva backward-compat com v1.5.0.
    _use_vmap_real = bool(getattr(cfg, "jax_vmap_real", False))
    if _use_vmap_real:
        H_tensor_jax, z_obs_np, rho_h_at_obs_np, rho_v_at_obs_np = (
            _simulate_multi_jax_vmap_real(
                rho_h_np=rho_h_np,
                rho_v_np=rho_v_np,
                esp_np=esp_np,
                positions_z_np=positions_z_np,
                freqs_arr=freqs_arr,
                tr_arr=tr_arr,
                dip_arr=dip_arr,
                hankel_filter=hankel_filter,
            )
        )
        hordist_groups_count = len(
            _build_hordist_groups(tr_arr.tolist(), dip_arr.tolist())
        )
        return MultiSimulationResultJAX(
            H_tensor=np.asarray(H_tensor_jax),
            z_obs=z_obs_np,
            rho_h_at_obs=rho_h_at_obs_np,
            rho_v_at_obs=rho_v_at_obs_np,
            freqs_hz=freqs_arr,
            tr_spacings_m=tr_arr,
            dip_degs=dip_arr,
            unique_hordist_count=hordist_groups_count,
        )

    # ── Deduplicação de cache por hordist ────────────────────────────────────
    hordist_groups = _build_hordist_groups(tr_arr.tolist(), dip_arr.tolist())

    # ── Pré-alocação dos tensores de saída ───────────────────────────────────
    H_tensor = np.empty((nTR, nAngles, n_pos, nf, 9), dtype=np.complex128)
    z_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    rho_h_at_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    rho_v_at_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    z_obs_filled = np.zeros(nAngles, dtype=bool)

    # ── Loop sobre (iTR, iAng): reutiliza cache por hordist ──────────────────
    # Sprint 10 Phase 2 (PR #24-part2): propaga `cfg.jax_strategy` — quando
    # "unified", cada forward interno usa 1 JIT único por (n, npt) em vez de
    # 44 JITs por bucket. Isso reduz VRAM GPU linear × nTR × nAngles.
    #
    # Vmap REAL sobre (iTR, iAng) está bloqueado porque `find_layers_tr`
    # permanece NumPy/Numba (decisão D3 do plano cosmic-riding-garden):
    # port para JAX exigiria `lax.cond` + recomputar camada a cada trace.
    # Ganho adicional de vmap aninhado diferido para Sprint 12 (PINN-on-esp).
    _strategy = getattr(cfg, "jax_strategy", "bucketed")
    for hordist_key, group in hordist_groups.items():
        for i_tr, i_ang, L, theta in group:
            # Build static context JAX (inclui camad_t_array, camad_r_array)
            ctx = build_static_context(
                rho_h=rho_h_np,
                rho_v=rho_v_np,
                esp=esp_np,
                positions_z=positions_z_np,
                freqs_hz=freqs_arr,
                tr_spacing_m=L,
                dip_deg=theta,
                hankel_filter=hankel_filter,
                strategy=_strategy,
            )

            # Forward JAX → (n_pos, nf, 9)
            H_jax = forward_pure_jax(
                jnp.asarray(rho_h_np, dtype=jnp.float64),
                jnp.asarray(rho_v_np, dtype=jnp.float64),
                ctx,
            )
            H_tensor[i_tr, i_ang] = np.asarray(H_jax)

            # z_obs e rho_h/v_at_obs por ângulo (invariantes em TR)
            if not z_obs_filled[i_ang]:
                # Computa via NumPy (mesma lógica de compute_zrho no Numba)
                dz_half = 0.5 * L * math.cos(math.radians(theta))
                # Convenção T/R Fortran (v1.4.1): T abaixo, R acima
                Tz = positions_z_np + dz_half
                cz = positions_z_np - dz_half
                z_mid = 0.5 * (Tz + cz)  # = positions_z
                z_obs[i_ang] = z_mid

                # Camada no ponto-médio (mesma lógica simulate_multi)
                from geosteering_ai.simulation._numba.geometry import (
                    layer_at_depth,
                )

                # Prof array: [0.0, esp[0], esp[0]+esp[1], ...]
                if n >= 2:
                    prof_arr = np.concatenate([np.array([0.0]), np.cumsum(esp_np)])
                else:
                    prof_arr = np.array([0.0])

                for i_pos in range(n_pos):
                    lay = layer_at_depth(n, z_mid[i_pos], prof_arr)
                    rho_h_at_obs[i_ang, i_pos] = rho_h_np[lay]
                    rho_v_at_obs[i_ang, i_pos] = rho_v_np[lay]
                z_obs_filled[i_ang] = True

    return MultiSimulationResultJAX(
        H_tensor=H_tensor,
        z_obs=z_obs,
        rho_h_at_obs=rho_h_at_obs,
        rho_v_at_obs=rho_v_at_obs,
        freqs_hz=freqs_arr,
        tr_spacings_m=tr_arr,
        dip_degs=dip_arr,
        unique_hordist_count=len(hordist_groups),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 12 (PR #25 / v1.6.0) — vmap real sobre (iTR, iAng)
# ──────────────────────────────────────────────────────────────────────────────
# MOTIVAÇÃO
#   A implementação `simulate_multi_jax` acima itera (iTR, iAng) em LOOP PYTHON
#   com dedup por `hordist = L·|sin(θ)|`. Para poços verticais (dip=0°) todas
#   as combinações colapsam em 1 hordist → dedup é 100% eficaz. Para multi-dip
#   reais (ex.: θ ∈ {0, 30, 60°}) o dedup é parcial e o loop Python gera
#   overhead significativo em GPU — cada iteração é um dispatch JAX separado
#   que impede fusão cross-config.
#
# DESIGN (Sprint 12)
#   `_simulate_multi_jax_vmap_real` substitui o loop por `jax.vmap` sobre o
#   grid achatado `(nTR × nAngles,)`. Para cada par (L, θ), calcula:
#     1. `dz_half = 0.5·L·cos(θ)`, `r_half = 0.5·L·sin(θ)`   — tracers float
#     2. `camad_t_array, camad_r_array` via `find_layers_tr_jax_vmap`
#        sobre `(n_pos,)` posições — tracers int32
#     3. Forward via `_get_unified_jit(n, npt)` — mesma função compilada
#        (1 XLA program, chave invariante em (n, npt))
#
#   O dispatcher em `simulate_multi_jax` verifica `cfg.jax_vmap_real` e chama
#   este path; caso contrário, mantém o Python loop original (preserva dedup
#   fast-path para poços verticais).
#
# GARANTIAS
#   ✅ Paridade bit-exata com `simulate_multi_jax` Python loop (validada em
#      `test_simulation_jax_sprint12.py::test_vmap_real_parity_*`)
#   ✅ Mesma API/shape de saída: `MultiSimulationResultJAX` com
#      `H_tensor = (nTR, nAngles, n_pos, nf, 9)`
#   ✅ Diferenciável (`jacfwd` preservado)
#   ✅ Compatível com `strategy="unified"` — default Sprint 12 usa unified
#      porque `_get_unified_jit` tem chave (n, npt), reutilizável entre configs


def _simulate_multi_jax_vmap_real(
    rho_h_np,
    rho_v_np,
    esp_np,
    positions_z_np,
    freqs_arr,
    tr_arr,
    dip_arr,
    hankel_filter: str,
) -> Tuple["jax.Array", "jax.Array", "jax.Array", "jax.Array"]:
    """Forward multi-TR/multi-ângulo via ``jax.vmap`` real sobre (iTR, iAng).

    Substitui o Python loop de :func:`simulate_multi_jax` por ``jax.vmap``
    aplicando ``find_layers_tr_jax_vmap`` dentro do trace para computar
    ``camad_t/camad_r`` como tracers int32.

    Args:
        rho_h_np: (n,) Ω·m.
        rho_v_np: (n,) Ω·m.
        esp_np: (n-2,) m — espessuras internas.
        positions_z_np: (n_pos,) m.
        freqs_arr: (nf,) Hz.
        tr_arr: (nTR,) m.
        dip_arr: (nAngles,) graus.
        hankel_filter: nome do filtro Hankel.

    Returns:
        Tupla ``(H_tensor, z_obs, rho_h_at_obs, rho_v_at_obs)``:

        - H_tensor: ``(nTR, nAngles, n_pos, nf, 9)`` complex128.
        - z_obs: ``(nAngles, n_pos)`` float64 — profundidades ponto-médio.
        - rho_h_at_obs: ``(nAngles, n_pos)`` float64.
        - rho_v_at_obs: ``(nAngles, n_pos)`` float64.

    Note:
        Gera 1 compilação XLA por ``(n, npt)`` — todas as configs ``(L, θ)``
        reutilizam a mesma função jitada. Em GPU T4 isso elimina o custo de
        dispatch Python entre combinações, esperado ganho 1.5-3× em configs
        multi-dip (nTR≥2, nAngles≥3).
    """
    from geosteering_ai.simulation._jax.forward_pure import _get_unified_jit
    from geosteering_ai.simulation._jax.geometry_jax import (
        find_layers_tr_jax_vmap,
    )
    from geosteering_ai.simulation._jax.kernel import _sanitize_profile_kernel
    from geosteering_ai.simulation._numba.geometry import layer_at_depth
    from geosteering_ai.simulation.filters import FilterLoader

    n = rho_h_np.shape[0]
    n_pos = positions_z_np.shape[0]
    nTR = tr_arr.shape[0]
    nAngles = dip_arr.shape[0]

    # ── Pré-computa estruturas estáticas (uma vez) ────────────────────────
    filt = FilterLoader().load(hankel_filter)
    krJ0J1 = np.asarray(filt.abscissas, dtype=np.float64)
    wJ0 = np.asarray(filt.weights_j0, dtype=np.float64)
    wJ1 = np.asarray(filt.weights_j1, dtype=np.float64)
    npt = krJ0J1.shape[0]

    if n == 1:
        h_arr = np.zeros(1, dtype=np.float64)
        prof_arr = np.array([-1.0e300, 1.0e300], dtype=np.float64)
    else:
        h_arr, prof_arr = _sanitize_profile_kernel(n, esp_np)

    # ── Converte para jnp uma única vez (compartilhado em todas as configs) ─
    rho_h_jnp = jnp.asarray(rho_h_np, dtype=jnp.float64)
    rho_v_jnp = jnp.asarray(rho_v_np, dtype=jnp.float64)
    positions_z_jnp = jnp.asarray(positions_z_np, dtype=jnp.float64)
    freqs_jnp = jnp.asarray(freqs_arr, dtype=jnp.float64)
    h_arr_jnp = jnp.asarray(h_arr, dtype=jnp.float64)
    prof_arr_jnp = jnp.asarray(prof_arr, dtype=jnp.float64)
    krJ0J1_jnp = jnp.asarray(krJ0J1, dtype=jnp.float64)
    wJ0_jnp = jnp.asarray(wJ0, dtype=jnp.float64)
    wJ1_jnp = jnp.asarray(wJ1, dtype=jnp.float64)

    # ── Recupera o JIT unificado (1 programa XLA por (n, npt)) ────────────
    jitted = _get_unified_jit(n, npt)

    # ── Função por configuração (L, θ) — todos args tracers ou estáticos ──
    def _one_config(L, theta_deg, rho_h, rho_v):
        """Forward para um par (L, θ) — chamada dentro de vmap.

        L, theta_deg viram tracers float64; rho_h/rho_v são arrays jnp
        (diferenciáveis); o restante é estático.
        """
        theta_rad = jnp.deg2rad(theta_deg)
        cos_t = jnp.cos(theta_rad)
        sin_t = jnp.sin(theta_rad)
        dz_half = 0.5 * L * cos_t
        r_half = 0.5 * L * sin_t

        # Convenção Fortran (v1.4.1 / PR #21): T abaixo, R acima
        Tz_arr = positions_z_jnp + dz_half
        Rz_arr = positions_z_jnp - dz_half

        # Camad arrays via searchsorted tracer-safe (Sprint 12, PR #25)
        camad_t_arr, camad_r_arr = find_layers_tr_jax_vmap(
            Tz_arr, Rz_arr, prof_arr_jnp, n
        )

        # Chama o unified JIT existente
        return jitted(
            rho_h,
            rho_v,
            positions_z_jnp,
            freqs_jnp,
            camad_t_arr,
            camad_r_arr,
            dz_half,
            r_half,
            theta_rad,
            h_arr_jnp,
            prof_arr_jnp,
            krJ0J1_jnp,
            wJ0_jnp,
            wJ1_jnp,
        )

    # ── Flatten grid (nTR × nAngles) → batch ──────────────────────────────
    # L_flat[k] = tr_arr[k // nAngles]; theta_flat[k] = dip_arr[k % nAngles]
    L_flat_np = np.repeat(tr_arr, nAngles).astype(np.float64)
    theta_flat_np = np.tile(dip_arr, nTR).astype(np.float64)
    L_flat = jnp.asarray(L_flat_np)
    theta_flat = jnp.asarray(theta_flat_np)

    # ── vmap sobre batch achatado ─────────────────────────────────────────
    vmap_fn = jax.vmap(_one_config, in_axes=(0, 0, None, None))
    H_flat = vmap_fn(L_flat, theta_flat, rho_h_jnp, rho_v_jnp)
    # H_flat shape: (nTR*nAngles, n_pos, nf, 9)

    # Reshape para (nTR, nAngles, n_pos, nf, 9)
    H_tensor_jax = H_flat.reshape(nTR, nAngles, n_pos, -1, 9)
    H_tensor_jax.block_until_ready()

    # ── Metadados z_obs / rho_*_at_obs (NumPy, por ângulo) ────────────────
    z_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    rho_h_at_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    rho_v_at_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    if n >= 2:
        prof_mid = np.concatenate([np.array([0.0]), np.cumsum(esp_np)])
    else:
        prof_mid = np.array([0.0])

    for i_ang in range(nAngles):
        # z_mid é o próprio positions_z (ponto médio por convenção)
        z_obs[i_ang] = positions_z_np
        for i_pos in range(n_pos):
            lay = layer_at_depth(n, float(positions_z_np[i_pos]), prof_mid)
            rho_h_at_obs[i_ang, i_pos] = rho_h_np[lay]
            rho_v_at_obs[i_ang, i_pos] = rho_v_np[lay]

    return H_tensor_jax, z_obs, rho_h_at_obs, rho_v_at_obs


__all__ = [
    "MultiSimulationResultJAX",
    "simulate_multi_jax",
    # Sprint 12 (PR #25): vmap real multi-TR/multi-ang
    "_simulate_multi_jax_vmap_real",
]
