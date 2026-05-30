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

import logging
import math
from dataclasses import dataclass
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

# Logger do módulo (D9 — nunca print). Usado para avisar fallback de estratégia
# batched (ex.: geometria heterogênea → bucketing inaplicável → unified).
logger = logging.getLogger(__name__)


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
    # Sprint v2.45: paridade de campo com o MultiSimulationResult Numba (F5
    # tilted-coil). O backend JAX não computa H_tilted (None); o campo existe
    # para compat com os writers .dat (export_info_out lê result.H_tilted).
    H_tilted: Optional[np.ndarray] = None

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
# Sprint A1.5 — MultiSimulationResultBatchedJAX (eixo n_models)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MultiSimulationResultBatchedJAX:
    """Container de resultados de :func:`simulate_multi_jax_batched`.

    Versão batched de :class:`MultiSimulationResultJAX` — adiciona dimensão
    líder ``n_models`` ao ``H_tensor`` e a ``rho_*_at_obs``. ``z_obs``,
    ``freqs_hz``, ``tr_spacings_m`` e ``dip_degs`` são **compartilhados**
    pelos modelos (invariantes ao perfil de resistividade).

    Restrição: todos os modelos do batch compartilham o mesmo ``n``
    (número de camadas). Para ``n`` heterogêneo, agrupar por ``n`` e
    chamar separadamente — ver docstring de
    :func:`simulate_multi_jax_batched`.

    Attributes:
        H_tensor: Tensor magnético H em formato 6-D batched.
            Shape ``(n_models, nTR, nAngles, n_pos, nf, 9)`` complex128.
            Ordem das 9 componentes flat:
            ``[Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz]``.
        z_obs: Profundidade TVD ponto-médio. Shape ``(nAngles, n_pos)``
            float64. **Compartilhado** entre modelos (depende só de
            ``positions_z`` e ``dip_degs``).
        rho_h_at_obs: Resistividade horizontal na camada do ponto-médio.
            Shape ``(n_models, nAngles, n_pos)`` float64 — depende do
            perfil ``rho_h`` por modelo.
        rho_v_at_obs: Resistividade vertical na camada do ponto-médio.
            Shape ``(n_models, nAngles, n_pos)`` float64.
        freqs_hz: Frequências usadas. Shape ``(nf,)`` float64.
        tr_spacings_m: Espaçamentos T-R. Shape ``(nTR,)`` float64.
        dip_degs: Ângulos dip. Shape ``(nAngles,)`` float64.
        n_models: Número de modelos no batch (``H_tensor.shape[0]``).

    Note:
        Para extrair um modelo individual: ``H_tensor[i_model, ...]`` retorna
        shape ``(nTR, nAngles, n_pos, nf, 9)`` — equivalente ao
        ``H_tensor`` de :class:`MultiSimulationResultJAX`.

        A API batched elimina o overhead Python serial do loop sobre modelos
        (5-20 ms/modelo de `build_static_context` + `np.asarray` sync).
        Ver `docs/sprints/v2.41.md` para análise da causa-raiz.
    """

    H_tensor: np.ndarray  # (n_models, nTR, nAngles, n_pos, nf, 9) complex128
    z_obs: np.ndarray  # (nAngles, n_pos) float64 — compartilhado
    rho_h_at_obs: np.ndarray  # (n_models, nAngles, n_pos) float64
    rho_v_at_obs: np.ndarray  # (n_models, nAngles, n_pos) float64
    freqs_hz: np.ndarray  # (nf,) float64
    tr_spacings_m: np.ndarray  # (nTR,) float64
    dip_degs: np.ndarray  # (nAngles,) float64
    n_models: int = 0
    # Sprint v2.45: compat F5 tilted-coil (None no backend JAX) — usado pelos
    # writers .dat (export_info_out lê result.H_tilted).
    H_tilted: Optional[np.ndarray] = None

    def get_model(self, i_model: int) -> MultiSimulationResultJAX:
        """Extrai resultado de um modelo individual como :class:`MultiSimulationResultJAX`.

        Útil para validar paridade contra ``simulate_multi_jax`` chamada
        em loop Python. Não é cópia — usa views NumPy.

        Args:
            i_model: Índice do modelo (0 ≤ i_model < n_models).

        Returns:
            :class:`MultiSimulationResultJAX` com ``H_tensor`` shape
            ``(nTR, nAngles, n_pos, nf, 9)``.

        Raises:
            IndexError: Se ``i_model`` fora do range.
        """
        if not (0 <= i_model < self.n_models):
            raise IndexError(f"i_model={i_model} fora do range [0, {self.n_models})")
        return MultiSimulationResultJAX(
            H_tensor=self.H_tensor[i_model],
            z_obs=self.z_obs,
            rho_h_at_obs=self.rho_h_at_obs[i_model],
            rho_v_at_obs=self.rho_v_at_obs[i_model],
            freqs_hz=self.freqs_hz,
            tr_spacings_m=self.tr_spacings_m,
            dip_degs=self.dip_degs,
            unique_hordist_count=0,  # dedup hordist não aplicável em batched
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
# Sprint A1.5 — Helper interno _sanitize_profile_batch (batch de modelos)
# ──────────────────────────────────────────────────────────────────────────────
def _sanitize_profile_batch(
    n: int, esp_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pré-computa ``h_arr`` e ``prof_arr`` para um batch de modelos.

    Wrapper Python sobre :func:`_sanitize_profile_kernel` (Numba @njit) que
    empilha resultados de ``n_models`` modelos em arrays 2D, prontos para
    consumo por ``jax.vmap(in_axes=(..., 0, 0))``. Mantido em Python (não
    portado para JAX) porque a operação é cheap (<1 ms por modelo); o
    benefício de portar não justifica o custo de manutenção.

    Args:
        n: Número de camadas (compartilhado por todos os modelos do batch).
        esp_batch: Espessuras internas. Shape ``(n_models, n-2)`` float64.
            Para ``n=1`` ou ``n=2``, deve ter shape ``(n_models, 0)``.

    Returns:
        Tupla ``(h_arr_batch, prof_arr_batch)``:

        - ``h_arr_batch``: Shape ``(n_models, n)`` float64.
        - ``prof_arr_batch``: Shape ``(n_models, n+1)`` float64 — boundaries
          com sentinelas ``-1e300`` no topo e ``+1e300`` no fundo.

    Note:
        Edge case ``n=1`` (semi-espaço único): retorna shape determinístico
        ``h_arr_batch=zeros((n_models, 1))`` e
        ``prof_arr_batch=tile([-1e300, 1e300], (n_models, 1))`` — alinhado
        com ``_simulate_multi_jax_vmap_real:557-561``.

    Example:
        >>> n_models, n = 5, 3
        >>> esp_batch = np.random.uniform(2, 10, size=(n_models, n - 2))
        >>> h_arr_batch, prof_arr_batch = _sanitize_profile_batch(n, esp_batch)
        >>> h_arr_batch.shape
        (5, 3)
        >>> prof_arr_batch.shape
        (5, 4)
    """
    from geosteering_ai.simulation._numba.geometry import _sanitize_profile_kernel

    n_models = esp_batch.shape[0]

    # ── Edge case n=1: shapes fixos para todos os modelos ──────────────
    if n == 1:
        h_arr_batch = np.zeros((n_models, 1), dtype=np.float64)
        prof_arr_batch = np.tile(
            np.array([-1.0e300, 1.0e300], dtype=np.float64), (n_models, 1)
        )
        return h_arr_batch, prof_arr_batch

    # ── Loop Python — sanitize é cheap (<1 ms × n_models = <50 ms total) ─
    h_list: list = []
    prof_list: list = []
    for i in range(n_models):
        h, prof = _sanitize_profile_kernel(n, np.ascontiguousarray(esp_batch[i]))
        h_list.append(h)
        prof_list.append(prof)

    return np.stack(h_list, axis=0), np.stack(prof_list, axis=0)


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
        build_static_context_cached,
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
        raise ValueError(
            f"tr_spacings_m fora do range [0.1, 10.0] m: {tr_arr.tolist()}"
        )
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
    # Sprint O2 (v2.43): cfg.dtype propagado tambem para vmap_real path.
    _cfg_dtype = getattr(cfg, "dtype", "complex128")
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
                complex_dtype=_cfg_dtype,
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
    # Sprint 12 E4: chunk_size=None → monolítico (v1.5.0/v1.6.0); inteiro →
    # lax.scan sobre blocos de posições (fix regressão 600pos×3f).
    _chunk_size = getattr(cfg, "jax_position_chunk_size", None)
    # Sprint O2 (v2.43): complex_dtype propagado de cfg.dtype.
    # Default "complex128" preserva paridade Fortran <1e-13 inviolavel.
    _complex_dtype = getattr(cfg, "dtype", "complex128")

    # ── Pré-alocação dos tensores de saída ───────────────────────────────────
    # Sprint O3 (v2.43): H_tensor agora respeita `cfg.dtype` — antes era
    # hard-coded `np.complex128`, anulando o ganho de memória de c64.
    _np_complex_dtype: type = (
        np.complex64 if _complex_dtype == "complex64" else np.complex128
    )
    H_tensor: np.ndarray = np.empty(
        (nTR, nAngles, n_pos, nf, 9), dtype=_np_complex_dtype
    )
    z_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    rho_h_at_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    rho_v_at_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    z_obs_filled = np.zeros(nAngles, dtype=bool)
    for hordist_key, group in hordist_groups.items():
        for i_tr, i_ang, L, theta in group:
            # Build static context JAX (inclui camad_t_array, camad_r_array).
            # Sprint O3p1 (v2.43): usa `build_static_context_cached` para
            # amortizar setup em batches de N modelos com mesma geometria
            # (rho varia, esp/positions/freqs/TR/dip constantes — caso PINN).
            # LRU 32 entries default — controle via `set_ctx_cache_maxsize`.
            ctx = build_static_context_cached(
                rho_h=rho_h_np,
                rho_v=rho_v_np,
                esp=esp_np,
                positions_z=positions_z_np,
                freqs_hz=freqs_arr,
                tr_spacing_m=L,
                dip_deg=theta,
                hankel_filter=hankel_filter,
                strategy=_strategy,
                chunk_size=_chunk_size,
                complex_dtype=_complex_dtype,
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
    complex_dtype: str = "complex128",
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

    # ── Recupera o JIT unificado (1 programa XLA por (n, npt, dtype)) ─────
    # Sprint O2 (v2.43): complex_dtype propagado para chave do cache.
    jitted = _get_unified_jit(n, npt, complex_dtype)

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


# ──────────────────────────────────────────────────────────────────────────────
# Sprint A1.5 (v2.42) — API pública batched sobre eixo n_models
# ──────────────────────────────────────────────────────────────────────────────
# MOTIVAÇÃO
#   Sprint A1 (validação Colab T4) revelou que JAX GPU é 2-5× MAIS LENTO que
#   Numba CPU em hot-cache, mesmo após eliminar o cold-start. Causa-raiz
#   arquitetural: a assinatura de simulate_multi_jax aceita apenas UM modelo
#   por chamada. O benchmark notebook itera "for model in models" em Python:
#     • build_static_context() recriado por modelo: 5-20 ms overhead
#     • np.asarray(H_jax) força GPU→CPU sync por modelo (kill async pipeline)
#     • _BUCKET_JIT_CACHE keyed (ct, cr, n, npt) → ~50 compilações XLA em Run 1
#   Para n_pos ≤ 600 o compute JAX GPU é <1 ms, mas overhead Python é 5-20 ms
#   por modelo. A GPU fica ociosa > 90% do tempo.
#
# DESIGN
#   simulate_multi_jax_batched(rho_h_batch, ...) aceita arrays 2D (n_models, n)
#   e aplica jax.vmap sobre o eixo n_models. Reutiliza _UNIFIED_JIT_CACHE
#   (keyed apenas (n, npt) — invariante a valores de modelo) → 1 compilação
#   XLA para o batch inteiro. Um único block_until_ready() ao final.
#
# RESTRIÇÃO ARQUITETURAL
#   Todos os modelos do batch DEVEM compartilhar o mesmo n (n_camadas).
#   prof_arr tem shape (n+1,) — stack via np.stack requer n homogêneo.
#   Para n heterogêneo, agrupar por n e chamar separadamente (loop Python
#   sobre poucos grupos, não sobre n_models).
#
# GARANTIAS
#   ✅ Paridade vs loop serial: mesma _get_unified_jit(n, npt) é chamada
#      → diferença esperada bit-exata (verificada em T1-T3 dos testes)
#   ✅ Diferenciabilidade jacfwd/grad preservada (vmap distribui sobre grad)
#   ✅ Backward-compat: simulate_multi_jax legada inalterada


def simulate_multi_jax_batched(
    rho_h_batch,
    rho_v_batch,
    esp_batch,
    positions_z,
    *,
    frequencies_hz: Optional[Sequence[float]] = None,
    tr_spacings_m: Optional[Sequence[float]] = None,
    dip_degs: Optional[Sequence[float]] = None,
    cfg=None,
    hankel_filter: str = "werthmuller_201pt",
) -> MultiSimulationResultBatchedJAX:
    """Forward JAX batched sobre eixo ``n_models`` via :func:`jax.vmap`.

    Versão batched de :func:`simulate_multi_jax` — aceita arrays 2D
    ``(n_models, n)`` e processa todos os modelos em um único trace JAX,
    eliminando o loop Python serial que ocorre quando o usuário itera
    ``for model in models: simulate_multi_jax(...)``.

    Reutiliza ``_UNIFIED_JIT_CACHE`` (Sprint 10 Phase 2, keyed ``(n, npt)``)
    — 1 compilação XLA para o batch inteiro, invariante a valores de
    modelo. Elimina:

    - ``build_static_context()`` recriado por modelo (5-20 ms × n_models)
    - ``np.asarray()`` GPU→CPU sync por modelo (kill async pipeline)
    - Bucket cache explosion (~50 compilações XLA em Run 1)

    Args:
        rho_h_batch: Resistividades horizontais em batch.
            Shape ``(n_models, n)`` float64, Ω·m. Todos os modelos
            compartilham o mesmo ``n`` (n_camadas).
        rho_v_batch: Resistividades verticais em batch.
            Shape ``(n_models, n)`` float64, Ω·m.
        esp_batch: Espessuras das camadas internas em batch.
            Shape ``(n_models, n-2)`` float64, m. Para ``n ≤ 2`` use shape
            ``(n_models, 0)``.
        positions_z: Profundidades TVD do ponto-médio T-R, **compartilhadas**.
            Shape ``(n_pos,)`` float64, m.
        frequencies_hz: Lista de frequências em Hz. Default:
            ``cfg.frequencies_hz`` ou ``[cfg.frequency_hz]``.
        tr_spacings_m: Lista de espaçamentos T-R em metros. Default:
            ``[cfg.tr_spacing_m]``.
        dip_degs: Lista de ângulos dip em graus. Default: ``[0.0]``.
            Range válido: ``[0, 89]°``.
        cfg: (Opcional) :class:`SimulationConfig` para defaults.
        hankel_filter: Nome do filtro Hankel. Default ``"werthmuller_201pt"``.

    Returns:
        :class:`MultiSimulationResultBatchedJAX` com:

        - ``H_tensor``: shape ``(n_models, nTR, nAngles, n_pos, nf, 9)``
          complex128.
        - ``z_obs``: shape ``(nAngles, n_pos)`` — compartilhado entre modelos.
        - ``rho_h_at_obs``, ``rho_v_at_obs``: shape ``(n_models, nAngles, n_pos)``.

    Raises:
        ImportError: Se JAX não estiver instalado.
        ValueError: Se shapes inconsistentes entre batches, ``n``
            heterogêneo, listas vazias ou ranges físicos inválidos.

    Example:
        >>> import numpy as np
        >>> n_models, n, n_pos = 50, 3, 600
        >>> rng = np.random.default_rng(42)
        >>> rho_h_batch = rng.uniform(1.0, 100.0, size=(n_models, n))
        >>> rho_v_batch = rho_h_batch.copy()  # isotrópico
        >>> esp_batch = rng.uniform(2.0, 10.0, size=(n_models, n - 2))
        >>> positions_z = np.linspace(-10, 10, n_pos)
        >>> result = simulate_multi_jax_batched(
        ...     rho_h_batch, rho_v_batch, esp_batch, positions_z,
        ...     frequencies_hz=[20e3], tr_spacings_m=[1.0], dip_degs=[0.0],
        ... )
        >>> result.H_tensor.shape
        (50, 1, 1, 600, 1, 9)

    Note:
        Para paridade vs ``simulate_multi_jax`` (loop serial), use o método
        ``result.get_model(i)`` que retorna :class:`MultiSimulationResultJAX`
        equivalente ao i-ésimo modelo. Diferença esperada <1e-12 (chamam
        a mesma ``_get_unified_jit(n, npt)``).
    """
    if not HAS_JAX:
        raise ImportError(
            "simulate_multi_jax_batched requer JAX (pip install 'jax[cpu]')."
        )

    from geosteering_ai.simulation._numba.geometry import layer_at_depth
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.filters import FilterLoader

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

    # ── Validações fail-fast (mesmas de simulate_multi_jax + batched-specific) ─
    if len(tr_arr) == 0:
        raise ValueError("tr_spacings_m vazio — forneça ao menos 1 TR spacing.")
    if len(dip_arr) == 0:
        raise ValueError("dip_degs vazio — forneça ao menos 1 ângulo dip.")
    if len(freqs_arr) == 0:
        raise ValueError("frequencies_hz vazio — forneça ao menos 1 frequência.")

    if np.any((tr_arr < 0.1) | (tr_arr > 10.0)):
        raise ValueError(
            f"tr_spacings_m fora do range [0.1, 10.0] m: {tr_arr.tolist()}"
        )
    if np.any((dip_arr < 0.0) | (dip_arr > 89.0)):
        raise ValueError(f"dip_degs fora do range [0, 89]°: {dip_arr.tolist()}")

    # ── Validações batched-specific ───────────────────────────────────────────
    rho_h_batch_np = np.ascontiguousarray(np.asarray(rho_h_batch, dtype=np.float64))
    rho_v_batch_np = np.ascontiguousarray(np.asarray(rho_v_batch, dtype=np.float64))
    esp_batch_np = np.ascontiguousarray(np.asarray(esp_batch, dtype=np.float64))
    positions_z_np = np.asarray(positions_z, dtype=np.float64)

    # GAP-C1 (Sprint A1.5 review): guard positions_z vazio/inválido
    if positions_z_np.ndim != 1:
        raise ValueError(
            f"positions_z deve ser 1D (n_pos,); obtido shape {positions_z_np.shape}"
        )
    if positions_z_np.shape[0] == 0:
        raise ValueError(
            "positions_z vazio — forneça ao menos 1 posição TVD para simular."
        )

    if rho_h_batch_np.ndim != 2:
        raise ValueError(
            f"rho_h_batch deve ser 2D (n_models, n_camadas); obtido shape "
            f"{rho_h_batch_np.shape}"
        )
    if rho_v_batch_np.shape != rho_h_batch_np.shape:
        raise ValueError(
            f"rho_v_batch shape {rho_v_batch_np.shape} != rho_h_batch "
            f"{rho_h_batch_np.shape}"
        )

    n_models, n = rho_h_batch_np.shape
    if n_models < 1:
        raise ValueError(f"n_models deve ser >= 1; obtido {n_models}")

    # n_camadas homogêneo é restrição arquitetural (prof_arr shape varia com n)
    expected_esp_cols = max(n - 2, 0)
    if esp_batch_np.ndim != 2 or esp_batch_np.shape[0] != n_models:
        raise ValueError(
            f"esp_batch shape inconsistente: esperado "
            f"({n_models}, {expected_esp_cols}); obtido {esp_batch_np.shape}. "
            f"Para n_camadas heterogêneo, agrupar por n e chamar separadamente."
        )
    if esp_batch_np.shape[1] != expected_esp_cols:
        raise ValueError(
            f"esp_batch.shape[1]={esp_batch_np.shape[1]} != n-2={expected_esp_cols} "
            f"(n_camadas={n}). Para n=1 ou n=2, esp_batch deve ter shape "
            f"({n_models}, 0)."
        )

    n_pos = positions_z_np.shape[0]
    nTR = tr_arr.shape[0]
    nAngles = dip_arr.shape[0]

    # ── Pré-computa arrays estáticos (1× por batch — não por modelo) ──────────
    filt = FilterLoader().load(hankel_filter)
    krJ0J1 = np.asarray(filt.abscissas, dtype=np.float64)
    wJ0 = np.asarray(filt.weights_j0, dtype=np.float64)
    wJ1 = np.asarray(filt.weights_j1, dtype=np.float64)
    npt = krJ0J1.shape[0]

    # Sanitize profile por modelo (Python loop, cheap <50 ms total)
    h_arr_batch_np, prof_arr_batch_np = _sanitize_profile_batch(n, esp_batch_np)

    # ── Converte para jnp (1× — compartilhado em todo o batch) ────────────────
    positions_z_jnp = jnp.asarray(positions_z_np, dtype=jnp.float64)
    freqs_jnp = jnp.asarray(freqs_arr, dtype=jnp.float64)
    krJ0J1_jnp = jnp.asarray(krJ0J1, dtype=jnp.float64)
    wJ0_jnp = jnp.asarray(wJ0, dtype=jnp.float64)
    wJ1_jnp = jnp.asarray(wJ1, dtype=jnp.float64)
    rho_h_batch_jnp = jnp.asarray(rho_h_batch_np, dtype=jnp.float64)
    rho_v_batch_jnp = jnp.asarray(rho_v_batch_np, dtype=jnp.float64)
    h_arr_batch_jnp = jnp.asarray(h_arr_batch_np, dtype=jnp.float64)
    prof_arr_batch_jnp = jnp.asarray(prof_arr_batch_np, dtype=jnp.float64)

    # ── Dispatcher de montagem do H_tensor por estratégia (Sprint O4) ────────
    # complex_dtype propagado de cfg.dtype (Sprint O2; default complex128).
    _complex_dtype_batched = getattr(cfg, "dtype", "complex128")
    _strategy = getattr(cfg, "jax_strategy", "bucketed")

    # Pré-condição do path BUCKETED: geometria COMPARTILHADA entre modelos
    # (esp bit-idêntico). Só então camad_t/camad_r são iguais p/ todos e o
    # bucketing único é fisicamente correto. Igualdade EXATA (atol=0): camad
    # depende de comparações de fronteira bit-exatas — tolerância geraria
    # buckets divergentes silenciosos. n=1/n=2 → esp shape (n_models, 0) →
    # np.allclose vacuamente True (geometria trivialmente compartilhada).
    geom_homogeneous = bool(
        np.allclose(esp_batch_np, esp_batch_np[0:1], rtol=0.0, atol=0.0)
    )

    nf = freqs_arr.shape[0]

    # Warning de fallback emitido 1× (não por chunk).
    if _strategy == "bucketed" and not geom_homogeneous:
        logger.warning(
            "simulate_multi_jax_batched: geometria HETEROGÊNEA detectada "
            "(esp varia entre modelos do batch) — bucketing compartilhado "
            "inaplicável; usando fallback 'unified' (correto, mais lento). "
            "Para o ganho bucketed, agrupe modelos por geometria idêntica."
        )

    # ── Chunk do eixo de modelos (Sprint O4) — fix OOM Cenário H ─────────────
    # jax_chunk_size_models=K processa o batch em fatias de K modelos (loop
    # Python, mesma compilação XLA reusada, 1 sync por fatia). Reduz o pico de
    # VRAM em n_models/K. None = batch monolítico (vmap sobre todos de uma vez).
    _chunk_models = getattr(cfg, "jax_chunk_size_models", None)

    def _dispatch_slice(sl: slice, sub_n_models: int):
        return _build_H_tensor_batched_dispatch(
            strategy=_strategy,
            complex_dtype=_complex_dtype_batched,
            geom_homogeneous=geom_homogeneous,
            n=n,
            npt=npt,
            n_models=sub_n_models,
            n_pos=n_pos,
            nf=nf,
            nTR=nTR,
            nAngles=nAngles,
            tr_arr=tr_arr,
            dip_arr=dip_arr,
            positions_z_np=positions_z_np,
            positions_z_jnp=positions_z_jnp,
            freqs_arr=freqs_arr,
            freqs_jnp=freqs_jnp,
            krJ0J1_jnp=krJ0J1_jnp,
            wJ0_jnp=wJ0_jnp,
            wJ1_jnp=wJ1_jnp,
            hankel_filter=hankel_filter,
            rho_h_batch_jnp=rho_h_batch_jnp[sl],
            rho_v_batch_jnp=rho_v_batch_jnp[sl],
            h_arr_batch_jnp=h_arr_batch_jnp[sl],
            prof_arr_batch_jnp=prof_arr_batch_jnp[sl],
            rho_h_batch_np=rho_h_batch_np[sl],
            rho_v_batch_np=rho_v_batch_np[sl],
            esp_batch_np=esp_batch_np[sl],
        )

    if _chunk_models is None or n_models <= _chunk_models:
        H_tensor_np = _dispatch_slice(slice(0, n_models), n_models)
    else:
        # Loop sobre ⌈n_models/K⌉ fatias; concatena no eixo de modelos (CPU).
        _parts: list = []
        for _start in range(0, n_models, _chunk_models):
            _end = min(_start + _chunk_models, n_models)
            _parts.append(_dispatch_slice(slice(_start, _end), _end - _start))
        H_tensor_np = np.concatenate(_parts, axis=0)

    # ── Metadados z_obs / rho_*_at_obs (NumPy, compartilhado + por modelo) ────
    # z_obs depende apenas de positions_z (ponto-médio é o próprio z_mid).
    z_obs = np.empty((nAngles, n_pos), dtype=np.float64)
    for i_ang in range(nAngles):
        z_obs[i_ang] = positions_z_np

    # rho_*_at_obs depende do perfil de cada modelo
    rho_h_at_obs = np.empty((n_models, nAngles, n_pos), dtype=np.float64)
    rho_v_at_obs = np.empty((n_models, nAngles, n_pos), dtype=np.float64)
    for i_model in range(n_models):
        rho_h_m = rho_h_batch_np[i_model]
        rho_v_m = rho_v_batch_np[i_model]
        if n >= 2:
            esp_m = esp_batch_np[i_model]
            prof_mid = np.concatenate([np.array([0.0]), np.cumsum(esp_m)])
        else:
            prof_mid = np.array([0.0])
        for i_ang in range(nAngles):
            for i_pos in range(n_pos):
                lay = layer_at_depth(n, float(positions_z_np[i_pos]), prof_mid)
                rho_h_at_obs[i_model, i_ang, i_pos] = rho_h_m[lay]
                rho_v_at_obs[i_model, i_ang, i_pos] = rho_v_m[lay]

    result = MultiSimulationResultBatchedJAX(
        H_tensor=H_tensor_np,
        z_obs=z_obs,
        rho_h_at_obs=rho_h_at_obs,
        rho_v_at_obs=rho_v_at_obs,
        freqs_hz=freqs_arr,
        tr_spacings_m=tr_arr,
        dip_degs=dip_arr,
        n_models=n_models,
    )

    # ── Sprint v2.45 — geração de dataset .dat por modelo (opt-in) ───────────
    # Quando cfg.export_per_model=True, exporta 1 conjunto de .dat por modelo
    # do batch via o writer Fortran-compatível (reusa export_multi_tr_dat).
    # MultiSimulationResultJAX expõe H_tilted=None → compat com export_info_out.
    if getattr(cfg, "export_per_model", False):
        _export_batched_models_dat(result, cfg)

    return result


def _export_batched_models_dat(result, cfg) -> list:
    """Exporta 1 conjunto de arquivos .dat por modelo de um batch JAX.

    Itera ``result.get_model(i)`` (view NumPy, sem cópia) e chama
    :func:`export_multi_tr_dat` por modelo, gerando
    ``{output_filename}_model{i:06d}[_TR{j}].dat`` em ``cfg.output_dir``.

    Args:
        result: :class:`MultiSimulationResultBatchedJAX`.
        cfg: :class:`SimulationConfig` com ``output_dir``/``output_filename``.

    Returns:
        Lista achatada de ``Path`` dos arquivos .dat criados (todos modelos).

    Note:
        Reusa o writer 22-col validado (paridade byte-exata vs tatu.x). O
        ``MultiSimulationResultJAX`` tem ``H_tilted=None`` (compat
        ``export_info_out``). Caso de uso: dataset sintético para treino.
    """
    from geosteering_ai.simulation.io.binary_dat_multi import export_multi_tr_dat

    base = getattr(cfg, "output_filename", "simulation")
    out_dir = getattr(cfg, "output_dir", ".")
    paths: list = []
    for i_model in range(result.n_models):
        single = result.get_model(i_model)
        paths.extend(export_multi_tr_dat(single, f"{base}_model{i_model:06d}", out_dir))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Sprint O4 (v2.44) — Helpers de montagem do H_tensor batched por estratégia
# ──────────────────────────────────────────────────────────────────────────────
# MOTIVAÇÃO
#   `simulate_multi_jax_batched` historicamente HARDCODAVA `_get_unified_jit`
#   (lax.fori_loop com camad_t/camad_r dinâmicos — ~6.9× mais lento que
#   bucketed). A montagem do H_tensor foi fatorada em helpers por estratégia
#   para permitir, sem regressão, o path BUCKETED quando a geometria é
#   compartilhada entre os modelos do batch (caso PINN/on-the-fly dominante).
#
#   Este helper (`_build_H_tensor_batched_unified`) é o EXTRACT bit-exato do
#   código original (vmap-modelos ∘ vmap-configs ∘ kernel unified). O sibling
#   bucketed (`_build_H_tensor_batched_bucketed`) é adicionado no commit
#   seguinte. O dispatcher em `simulate_multi_jax_batched` escolhe entre eles.


def _build_H_tensor_batched_unified(
    *,
    n: int,
    npt: int,
    n_models: int,
    n_pos: int,
    nf: int,
    nTR: int,
    nAngles: int,
    complex_dtype: str,
    tr_arr,
    dip_arr,
    positions_z_jnp,
    freqs_jnp,
    krJ0J1_jnp,
    wJ0_jnp,
    wJ1_jnp,
    rho_h_batch_jnp,
    rho_v_batch_jnp,
    h_arr_batch_jnp,
    prof_arr_batch_jnp,
):
    """Monta o ``H_tensor`` batched via o kernel UNIFIED (1 JIT por (n,npt,dtype)).

    Extract bit-exato do código original de :func:`simulate_multi_jax_batched`
    (pré-Sprint O4). ``camad_t/camad_r`` são computados como **tracers** dentro
    do ``vmap`` sobre modelos (via :func:`find_layers_tr_jax_vmap`), o que exige
    o kernel unified (``lax.fori_loop`` com bounds dinâmicos). Funciona para
    geometria heterogênea entre modelos (cada linha de ``prof_arr_batch`` pode
    diferir) — por isso é o **fallback** quando a geometria NÃO é compartilhada.

    Estrutura: ``vmap_modelos(vmap_configs(jitted_unified))`` — vmap externo
    sobre o eixo ``n_models``, vmap interno sobre as ``nTR×nAngles`` configs.

    Args:
        n: Número de camadas (homogêneo no batch).
        npt: Número de pontos do filtro Hankel.
        n_models: Tamanho do batch (eixo 0 de ``rho_*_batch_jnp``).
        n_pos: Número de posições TVD compartilhadas.
        nf: Número de frequências.
        nTR: Número de espaçamentos T-R.
        nAngles: Número de ângulos dip.
        complex_dtype: ``"complex128"`` (default) ou ``"complex64"`` — chave
            do cache JIT (Sprint O2) que garante o dtype correto do kernel.
        tr_arr: ``(nTR,)`` espaçamentos T-R (m), NumPy float64.
        dip_arr: ``(nAngles,)`` ângulos dip (graus), NumPy float64.
        positions_z_jnp: ``(n_pos,)`` profundidades TVD compartilhadas (jnp).
        freqs_jnp: ``(nf,)`` frequências (Hz) (jnp).
        krJ0J1_jnp, wJ0_jnp, wJ1_jnp: arrays do filtro Hankel (jnp).
        rho_h_batch_jnp, rho_v_batch_jnp: ``(n_models, n)`` resistividades (jnp).
        h_arr_batch_jnp: ``(n_models, n)`` espessuras sanitizadas (jnp).
        prof_arr_batch_jnp: ``(n_models, n+1)`` fronteiras por modelo (jnp).

    Returns:
        ``np.ndarray`` complex de shape ``(n_models, nTR, nAngles, n_pos, nf, 9)``
        — host array (1 único sync GPU→CPU ao final).

    Note:
        Diferenciável (``jacfwd``) — ``eta`` é reconstruído traceable no kernel.
        Paridade vs loop serial ``simulate_multi_jax`` (unified): <1e-13.
    """
    from geosteering_ai.simulation._jax.forward_pure import _get_unified_jit
    from geosteering_ai.simulation._jax.geometry_jax import find_layers_tr_jax_vmap

    # 1 programa XLA por (n, npt, complex_dtype) — reutilizado p/ todos modelos.
    jitted = _get_unified_jit(n, npt, complex_dtype)

    # L_flat[k] = tr_arr[k // nAngles]; theta_flat[k] = dip_arr[k % nAngles].
    L_flat = jnp.asarray(np.repeat(tr_arr, nAngles).astype(np.float64))
    theta_flat = jnp.asarray(np.tile(dip_arr, nTR).astype(np.float64))

    def _one_model(rho_h, rho_v, h_arr, prof_arr):
        """Forward para 1 modelo cobrindo (nTR × nAngles) configs via vmap."""

        def _one_config(L, theta_deg):
            theta_rad = jnp.deg2rad(theta_deg)
            cos_t = jnp.cos(theta_rad)
            sin_t = jnp.sin(theta_rad)
            dz_half = 0.5 * L * cos_t
            r_half = 0.5 * L * sin_t

            # Convenção Fortran (v1.4.1): T abaixo, R acima.
            Tz_arr = positions_z_jnp + dz_half
            Rz_arr = positions_z_jnp - dz_half

            # find_layers vmapped sobre posições; prof_arr é tracer do outer vmap.
            camad_t_arr, camad_r_arr = find_layers_tr_jax_vmap(
                Tz_arr, Rz_arr, prof_arr, n
            )

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
                h_arr,
                prof_arr,
                krJ0J1_jnp,
                wJ0_jnp,
                wJ1_jnp,
            )

        vmap_configs = jax.vmap(_one_config, in_axes=(0, 0))
        return vmap_configs(L_flat, theta_flat)  # (nTR*nAngles, n_pos, nf, 9)

    # Vmap externo sobre modelos.
    batched_fn = jax.vmap(_one_model, in_axes=(0, 0, 0, 0))
    H_flat_batched = batched_fn(
        rho_h_batch_jnp, rho_v_batch_jnp, h_arr_batch_jnp, prof_arr_batch_jnp
    )
    # (n_models, nTR*nAngles, n_pos, nf, 9) → reshape + ÚNICO sync GPU→CPU.
    H_tensor_jax = H_flat_batched.reshape(n_models, nTR, nAngles, n_pos, nf, 9)
    H_tensor_jax.block_until_ready()  # 1× ao final (não n_models×)
    return np.asarray(H_tensor_jax)


def _forward_config_buckets_over_models(
    *,
    ctx,
    rho_h_batch_jnp,
    rho_v_batch_jnp,
    n: int,
    npt: int,
    n_models: int,
    n_pos: int,
    nf: int,
    complex_dtype: str,
):
    """Forward BUCKETED de 1 config (L, θ) sobre TODO o eixo de modelos via vmap.

    Núcleo do Sprint O4. Reusa EXATAMENTE o mesmo kernel ``_get_bucket_jit`` do
    path serial (:func:`_forward_pure_jax_bucketed_impl`), apenas envolvido por
    um ``jax.vmap`` que mapeia o eixo 0 (modelos) em ``rho_h_batch``/``rho_v_batch``
    e faz broadcast (``in_axes=None``) de TODOS os arrays de geometria/filtro do
    ``ctx`` — válido porque a geometria é COMPARTILHADA entre os modelos (mesmo
    ``esp``/``positions_z``/``L``/``θ`` → mesmos ``camad_t/camad_r`` concretos).

    O agrupamento por bucket ``(camad_t, camad_r)`` é feito em NumPy CONCRETO
    (fora do trace), via ``np.unique`` sobre ``ctx.camad_t_array/camad_r_array``
    — idêntico a :func:`_forward_pure_jax_bucketed_impl`. Cada bucket fecha
    ``ct/cr`` estáticos no closure (XLA especializa → sem ``lax.fori_loop``
    dinâmico do unified).

    Args:
        ctx: :class:`ForwardPureContext` CONCRETO desta config (camad_t/camad_r
            numpy, ``dz_half``/``r_half``/``dip_rad`` floats via ``np.cos`` —
            bit-idênticos ao serial).
        rho_h_batch_jnp, rho_v_batch_jnp: ``(n_models, n)`` — únicos eixos vmapped.
        n, npt, n_models, n_pos, nf: dimensões estáticas.
        complex_dtype: chave do cache JIT (Sprint O2) + dtype do H_config.

    Returns:
        ``jax.Array`` ``(n_models, n_pos, nf, 9)`` na ORDEM ORIGINAL de posições
        (scatter via ``.at[:, indices].set`` preserva a ordem). NÃO sincronizado
        (o sync único acontece no caller :func:`_build_H_tensor_batched_bucketed`).

    Note:
        Diferenciável: ``rho_h/rho_v`` entram como tracers do vmap E como
        argumentos diferenciáveis do kernel (``eta`` reconstruído via
        ``jnp.stack(1/rho)``). Paridade vs serial bucketed: bit-exata (<1e-13).
    """
    from geosteering_ai.simulation._jax.forward_pure import (
        _COMPLEX_DTYPE_MAP,
        _get_bucket_jit,
    )

    # Agrupa posições por (camad_t, camad_r) ÚNICO — NumPy concreto, fora do
    # trace (mesma lógica de _forward_pure_jax_bucketed_impl:937-940).
    ct_arr = np.asarray(ctx.camad_t_array, dtype=np.int32)
    cr_arr = np.asarray(ctx.camad_r_array, dtype=np.int32)
    key_arr = ct_arr.astype(np.int64) * 10_000 + cr_arr.astype(np.int64)
    unique_keys, inverse = np.unique(key_arr, return_inverse=True)

    H_config = jnp.zeros(
        (n_models, n_pos, nf, 9), dtype=_COMPLEX_DTYPE_MAP[complex_dtype]
    )
    # _forward_bucket assinatura: (rho_h, rho_v, z_bucket, freqs, dz_half,
    # r_half, dip_rad, h_arr, prof_arr, krJ0J1, wJ0, wJ1). Mapeia eixo 0
    # (modelos) APENAS em rho_h/rho_v; resto broadcast (geometria compartilhada).
    _in_axes = (0, 0, None, None, None, None, None, None, None, None, None, None)
    for bucket_idx in range(len(unique_keys)):
        indices = np.nonzero(inverse == bucket_idx)[0]
        ct = int(ct_arr[indices[0]])
        cr = int(cr_arr[indices[0]])
        z_bucket = ctx.positions_z_jnp[indices]  # (n_pos_bucket,)

        jitted = _get_bucket_jit(ct, cr, n, npt, complex_dtype)
        bucket_over_models = jax.vmap(jitted, in_axes=_in_axes)
        H_bucket = bucket_over_models(
            rho_h_batch_jnp,
            rho_v_batch_jnp,
            z_bucket,
            ctx.freqs_hz_jnp,
            ctx.dz_half,
            ctx.r_half,
            ctx.dip_rad,
            ctx.h_arr_jnp,
            ctx.prof_arr_jnp,
            ctx.krJ0J1,
            ctx.wJ0,
            ctx.wJ1,
        )  # (n_models, n_pos_bucket, nf, 9)

        # Scatter no eixo de POSIÇÃO (eixo 1) — preserva ordem original.
        H_config = H_config.at[:, jnp.asarray(indices)].set(H_bucket)

    return H_config


def _build_H_tensor_batched_bucketed(
    *,
    n: int,
    npt: int,
    n_models: int,
    n_pos: int,
    nf: int,
    nTR: int,
    nAngles: int,
    complex_dtype: str,
    rho_h_ref_np,
    rho_v_ref_np,
    esp_ref_np,
    positions_z_np,
    freqs_arr,
    tr_arr,
    dip_arr,
    hankel_filter: str,
    rho_h_batch_jnp,
    rho_v_batch_jnp,
):
    """Monta o ``H_tensor`` batched via kernel BUCKETED (geometria compartilhada).

    Sprint O4 (v2.44). Path ótimo para o regime PINN/on-the-fly: todos os
    modelos do batch compartilham a MESMA geometria (``esp``/``positions_z``),
    variando apenas ``rho_h``/``rho_v``. Nesse regime ``camad_t/camad_r`` são
    idênticos entre modelos e podem ser computados UMA vez por config (L, θ)
    em NumPy concreto (via :func:`build_static_context_cached`), permitindo
    ``np.unique``/bucketing fora do trace e ``jax.vmap`` dos kernels de bucket
    sobre o eixo de modelos.

    Estrutura (loop Python sobre nTR×nAngles configs; dedup por hordist como
    :func:`simulate_multi_jax`):

      for (i_tr, i_ang, L, θ):
        ctx = build_static_context_cached(..., L, θ, strategy="bucketed")  # cache hit p/ geom fixa
        H_config[i_tr][i_ang] = _forward_config_buckets_over_models(ctx, rho_batch)  # (n_models,n_pos,nf,9)
      H_tensor = stack/moveaxis → (n_models, nTR, nAngles, n_pos, nf, 9)
      ÚNICO block_until_ready + np.asarray no fim.

    Args:
        n, npt, n_models, n_pos, nf, nTR, nAngles: dimensões estáticas.
        complex_dtype: ``"complex128"`` (default) ou ``"complex64"``.
        rho_h_ref_np, rho_v_ref_np, esp_ref_np: perfil do modelo 0 (referência
            de GEOMETRIA — ``esp`` é idêntico em todos por pré-condição checada
            pelo dispatcher; ``rho`` do ref NÃO é usado no kernel, só geometria).
        positions_z_np, freqs_arr, tr_arr, dip_arr: arrays compartilhados (NumPy).
        hankel_filter: nome do filtro Hankel.
        rho_h_batch_jnp, rho_v_batch_jnp: ``(n_models, n)`` — eixos vmapped.

    Returns:
        ``np.ndarray`` ``(n_models, nTR, nAngles, n_pos, nf, 9)`` — host array
        (1 único sync GPU→CPU ao final do batch inteiro).

    Note:
        PRÉ-CONDIÇÃO (garantida pelo dispatcher): geometria compartilhada entre
        modelos (``esp`` bit-idêntico). Geometria heterogênea → usar
        :func:`_build_H_tensor_batched_unified`. Paridade vs serial bucketed e
        vs batched-unified: <1e-13 (mesmo kernel ``_get_bucket_jit``).
    """
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context_cached,
    )

    # Storage por config (preenchido na ordem original i_tr, i_ang).
    configs_jnp = [[None] * nAngles for _ in range(nTR)]

    # Loop sobre configs com dedup por hordist (max cache hit em ctx — geom fixa).
    hordist_groups = _build_hordist_groups(tr_arr.tolist(), dip_arr.tolist())
    for _hordist_key, group in hordist_groups.items():
        for i_tr, i_ang, L, theta in group:
            # ctx CONCRETO desta config — camad_t/camad_r + dz/r via np.cos
            # (bit-idêntico ao serial). Cache LRU amortiza geometria fixa.
            ctx = build_static_context_cached(
                rho_h=rho_h_ref_np,
                rho_v=rho_v_ref_np,
                esp=esp_ref_np,
                positions_z=positions_z_np,
                freqs_hz=freqs_arr,
                tr_spacing_m=L,
                dip_deg=theta,
                hankel_filter=hankel_filter,
                strategy="bucketed",
                chunk_size=None,
                complex_dtype=complex_dtype,
            )
            configs_jnp[i_tr][i_ang] = _forward_config_buckets_over_models(
                ctx=ctx,
                rho_h_batch_jnp=rho_h_batch_jnp,
                rho_v_batch_jnp=rho_v_batch_jnp,
                n=n,
                npt=npt,
                n_models=n_models,
                n_pos=n_pos,
                nf=nf,
                complex_dtype=complex_dtype,
            )

    # Empilha (nTR, nAngles, n_models, n_pos, nf, 9) e move eixo de modelos p/
    # frente → (n_models, nTR, nAngles, n_pos, nf, 9). ÚNICO sync GPU→CPU.
    H_stack = jnp.stack(
        [
            jnp.stack([configs_jnp[i_tr][i_ang] for i_ang in range(nAngles)], axis=0)
            for i_tr in range(nTR)
        ],
        axis=0,
    )  # (nTR, nAngles, n_models, n_pos, nf, 9)
    H_tensor_jax = jnp.moveaxis(H_stack, 2, 0)
    H_tensor_jax.block_until_ready()  # 1× ao final (sync único por batch)
    return np.asarray(H_tensor_jax)


def _build_H_tensor_batched_dispatch(
    *,
    strategy: str,
    complex_dtype: str,
    geom_homogeneous: bool,
    n: int,
    npt: int,
    n_models: int,
    n_pos: int,
    nf: int,
    nTR: int,
    nAngles: int,
    tr_arr,
    dip_arr,
    positions_z_np,
    positions_z_jnp,
    freqs_arr,
    freqs_jnp,
    krJ0J1_jnp,
    wJ0_jnp,
    wJ1_jnp,
    hankel_filter: str,
    rho_h_batch_jnp,
    rho_v_batch_jnp,
    h_arr_batch_jnp,
    prof_arr_batch_jnp,
    rho_h_batch_np,
    rho_v_batch_np,
    esp_batch_np,
):
    """Escolhe bucketed (geom compartilhada) vs unified e monta o H_tensor.

    Núcleo do dispatcher de :func:`simulate_multi_jax_batched`, fatorado para
    ser chamável por (sub)batch — permite o loop de ``jax_chunk_size_models``
    (commit O4 chunking) reusar a MESMA lógica de escolha de estratégia por
    fatia de modelos. NÃO emite o warning de fallback (responsabilidade do
    caller, 1× por chamada — não por chunk).

    Args:
        strategy: ``cfg.jax_strategy`` (``"bucketed"``|``"unified"``).
        geom_homogeneous: pré-computado pelo caller sobre o batch COMPLETO
            (subset de homogêneo é homogêneo; conservador e correto).
        n_models: tamanho deste (sub)batch (pode ser < n_models total no chunk).
        (demais args: ver :func:`_build_H_tensor_batched_bucketed` /
        :func:`_build_H_tensor_batched_unified`).

    Returns:
        ``np.ndarray`` ``(n_models, nTR, nAngles, n_pos, nf, 9)`` deste (sub)batch.
    """
    if strategy == "bucketed" and geom_homogeneous:
        return _build_H_tensor_batched_bucketed(
            n=n,
            npt=npt,
            n_models=n_models,
            n_pos=n_pos,
            nf=nf,
            nTR=nTR,
            nAngles=nAngles,
            complex_dtype=complex_dtype,
            rho_h_ref_np=rho_h_batch_np[0],
            rho_v_ref_np=rho_v_batch_np[0],
            esp_ref_np=esp_batch_np[0],
            positions_z_np=positions_z_np,
            freqs_arr=freqs_arr,
            tr_arr=tr_arr,
            dip_arr=dip_arr,
            hankel_filter=hankel_filter,
            rho_h_batch_jnp=rho_h_batch_jnp,
            rho_v_batch_jnp=rho_v_batch_jnp,
        )
    return _build_H_tensor_batched_unified(
        n=n,
        npt=npt,
        n_models=n_models,
        n_pos=n_pos,
        nf=nf,
        nTR=nTR,
        nAngles=nAngles,
        complex_dtype=complex_dtype,
        tr_arr=tr_arr,
        dip_arr=dip_arr,
        positions_z_jnp=positions_z_jnp,
        freqs_jnp=freqs_jnp,
        krJ0J1_jnp=krJ0J1_jnp,
        wJ0_jnp=wJ0_jnp,
        wJ1_jnp=wJ1_jnp,
        rho_h_batch_jnp=rho_h_batch_jnp,
        rho_v_batch_jnp=rho_v_batch_jnp,
        h_arr_batch_jnp=h_arr_batch_jnp,
        prof_arr_batch_jnp=prof_arr_batch_jnp,
    )


__all__ = [
    "MultiSimulationResultJAX",
    "simulate_multi_jax",
    # Sprint 12 (PR #25): vmap real multi-TR/multi-ang
    "_simulate_multi_jax_vmap_real",
    # Sprint A1.5 (v2.42): batched API sobre eixo n_models
    "MultiSimulationResultBatchedJAX",
    "simulate_multi_jax_batched",
]
