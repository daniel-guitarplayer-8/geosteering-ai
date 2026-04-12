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

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geosteering_ai.simulation._numba.kernel import compute_zrho, fields_in_freqs
from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.filters import FilterLoader

logger = logging.getLogger(__name__)

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

    # ── Loop sobre posições ───────────────────────────────────────
    # Este loop será paralelizável via prange na Sprint 2.7 (benchmark).
    # Por agora, serial — correto e portável.
    logger.debug(
        "simulate: n_positions=%d, nf=%d, n_layers=%d, dip=%.1f°, L=%.2f m",
        n_positions,
        nf,
        n,
        dip_deg,
        L,
    )

    for j in range(n_positions):
        z_mid = positions_z[j]

        # Transmissor acima do ponto-médio, receptor abaixo
        Tz = z_mid - dz_half
        cz = z_mid + dz_half
        # Afastamento horizontal (simétrico em x)
        Tx = -r_half
        cx = r_half
        Ty = 0.0
        cy = 0.0

        # Forward para todas as frequências nesta posição
        cH = fields_in_freqs(
            Tx=Tx,
            Ty=Ty,
            Tz=Tz,
            cx=cx,
            cy=cy,
            cz=cz,
            dip_rad=dip_rad,
            n=n,
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            freqs_hz=freqs_hz,
            krJ0J1=krJ0J1,
            wJ0=wJ0,
            wJ1=wJ1,
        )
        H_tensor[j, :, :] = cH

        # Metadados da posição
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
