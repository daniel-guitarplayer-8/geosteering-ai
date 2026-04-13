# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/visualization/plot_tensor.py                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Plotagem de Tensor H (Sprint 2.8)      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : matplotlib                                                 ║
# ║  Dependências: matplotlib, numpy                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Funções de plotagem de amostras geológicas e tensor H no padrão       ║
# ║    GridSpec(3,7) adotado por `buildValidamodels.py`. Consomem            ║
# ║    :class:`SimulationResult` diretamente, evitando manipulação ad-hoc   ║
# ║    de arrays 22-col binários do formato Fortran.                         ║
# ║                                                                           ║
# ║  LAYOUT DE FIGURA                                                         ║
# ║    Coluna 0 (width=1.8):  perfil ρₕ/ρᵥ (semilogx) + interfaces          ║
# ║    Colunas 1-6 (width=1): 3×2 painéis Re/Im dos 9 componentes H_ij      ║
# ║                                                                           ║
# ║      ┌─────────┬──────┬──────┬──────┬──────┬──────┬──────┐             ║
# ║      │         │ Hxx  │      │ Hxy  │      │ Hxz  │      │             ║
# ║      │  ρh,ρv  │ Re   │ Im   │ Re   │ Im   │ Re   │ Im   │             ║
# ║      │         ├──────┼──────┼──────┼──────┼──────┼──────┤             ║
# ║      │  log    │ Hyx  │      │ Hyy  │      │ Hyz  │      │             ║
# ║      │  scale  │ Re   │ Im   │ Re   │ Im   │ Re   │ Im   │             ║
# ║      │         ├──────┼──────┼──────┼──────┼──────┼──────┤             ║
# ║      │  depth  │ Hzx  │      │ Hzy  │      │ Hzz  │      │             ║
# ║      │  down   │ Re   │ Im   │ Re   │ Im   │ Re   │ Im   │             ║
# ║      └─────────┴──────┴──────┴──────┴──────┴──────┴──────┘             ║
# ║                                                                           ║
# ║  CONVENÇÕES GEOLÓGICAS                                                    ║
# ║    • invert_yaxis() — profundidade cresce para baixo                     ║
# ║    • semilogx(rho) — resistividade spans 1e-1 a 1e6 Ω·m                 ║
# ║    • ρₕ: steelblue sólido; ρᵥ: darkorange tracejada                     ║
# ║    • Interfaces: axhline(interf, color='black', ls='--', alpha=0.7)     ║
# ║    • Re: paleta azul (darkblue, navy, steelblue, royalblue, cornflower) ║
# ║    • Im: paleta vermelha (darkred, firebrick, indianred, salmon, tomato)║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/buildValidamodels.py:571-628 — layout original     ║
# ║    • geosteering_ai/simulation/forward.py — SimulationResult             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Plotagem de amostras geológicas e tensor H do simulador Python.

Adaptado de `Fortran_Gerador/buildValidamodels.py` para consumir
:class:`SimulationResult` diretamente. Layout GridSpec(3,7) preserva o
padrão visual usado nos relatórios de validação do simulador Fortran.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

# Import lazy de matplotlib (opcional em ambientes headless minimal)
try:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:  # pragma: no cover — matplotlib é dep opcional
    _HAS_MPL = False
    plt = None  # type: ignore[assignment]
    gridspec = None  # type: ignore[assignment]
    Figure = object  # type: ignore[misc,assignment]

from geosteering_ai.simulation.forward import SimulationResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes visuais
# ──────────────────────────────────────────────────────────────────────────────
# Paletas idênticas a `buildValidamodels.py:540-545`. Cores são aplicadas
# cíclicamente quando há múltiplas frequências/ângulos sobrepostos no
# mesmo painel.
_PALETTE_RE: tuple[str, ...] = (
    "darkblue",
    "navy",
    "steelblue",
    "royalblue",
    "cornflowerblue",
)
_PALETTE_IM: tuple[str, ...] = (
    "darkred",
    "firebrick",
    "indianred",
    "salmon",
    "tomato",
)

# Nomes canônicos dos 9 componentes do tensor H 3×3 em formato flat
# (linha=dipolo, coluna=campo observado). Ordem compatível com o
# SimulationResult.H_tensor (axis=-1 tem 9 entradas).
_TENSOR_NAMES: tuple[tuple[str, str, str], ...] = (
    ("Hxx", "Hxy", "Hxz"),
    ("Hyx", "Hyy", "Hyz"),
    ("Hzx", "Hzy", "Hzz"),
)


# ──────────────────────────────────────────────────────────────────────────────
# plot_resistivity_profile — painel standalone do perfil ρₕ/ρᵥ
# ──────────────────────────────────────────────────────────────────────────────


def plot_resistivity_profile(
    result: SimulationResult,
    ax=None,
    *,
    show_interfaces: bool = True,
    title: Optional[str] = None,
):
    """Plota o perfil de resistividade ρₕ/ρᵥ em função da profundidade.

    Args:
        result: Resultado da simulação contendo `rho_h_at_obs`,
            `rho_v_at_obs` e `z_obs`.
        ax: Eixo matplotlib opcional. Se None, cria figura nova.
        show_interfaces: Se True, marca interfaces de camadas com
            linhas pretas tracejadas (detectadas por mudança de ρₕ).
        title: Título do painel. Default: ``"ρₕ e ρᵥ (perfil geológico)"``.

    Returns:
        Tupla ``(fig, ax)``. ``fig`` é None se ``ax`` foi fornecido.

    Example:
        >>> from geosteering_ai.simulation.visualization import (
        ...     plot_resistivity_profile
        ... )
        >>> fig, ax = plot_resistivity_profile(result)
        >>> fig.savefig("profile.png")
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib é necessário para plot_resistivity_profile. "
            "Instale via `pip install matplotlib`."
        )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 8))
        created_fig = True
    else:
        fig = None

    rho_h = result.rho_h_at_obs
    rho_v = result.rho_v_at_obs
    z_obs = result.z_obs

    ax.semilogx(rho_h, z_obs, color="steelblue", label=r"$\rho_h$", linewidth=1.5)
    ax.semilogx(
        rho_v,
        z_obs,
        color="darkorange",
        linestyle="--",
        label=r"$\rho_v$",
        linewidth=1.5,
    )

    if show_interfaces:
        # Detecta interfaces onde ρₕ muda abruptamente (mudança entre
        # amostras consecutivas). Marca com axhline tracejada preta.
        interfaces = _detect_interfaces(rho_h, z_obs)
        for z_interf in interfaces:
            ax.axhline(
                y=z_interf, color="black", linestyle="--", linewidth=1.0, alpha=0.7
            )

    ax.invert_yaxis()  # Profundidade cresce para baixo (convenção geofísica)
    ax.set_xlabel(r"Resistividade ($\Omega{\cdot}$m)")
    ax.set_ylabel("Profundidade (m)")
    ax.set_title(title or r"$\rho_h$ e $\rho_v$ (perfil geológico)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    return (fig if created_fig else None), ax


# ──────────────────────────────────────────────────────────────────────────────
# plot_tensor_profile — figura completa GridSpec(3,7)
# ──────────────────────────────────────────────────────────────────────────────


def plot_tensor_profile(
    result: SimulationResult,
    *,
    freq_idx: int = 0,
    title: Optional[str] = None,
    model_name: Optional[str] = None,
    dip_deg: Optional[float] = None,
    figsize: tuple[float, float] = (28.0, 12.0),
    show_interfaces: bool = True,
) -> Figure:
    """Plota tensor H completo + perfil ρ no layout GridSpec(3, 7).

    Replica exatamente o padrão de figuras de validação do simulador
    Fortran (`buildValidamodels.py:571-628`), adaptado para
    :class:`SimulationResult`.

    Layout:
      - Coluna 0 (width=1.8): perfil ρₕ/ρᵥ (semilogx, invert y-axis)
      - Colunas 1-6 (width=1): 3×2 painéis Re/Im dos 9 componentes

    Args:
        result: Resultado da simulação. ``result.H_tensor`` deve ter
            shape ``(n_positions, nf, 9)`` complex128.
        freq_idx: Índice da frequência a plotar (quando ``nf > 1``).
            Default: 0. Valores fora do range levantam ValueError.
        title: Título da figura. Se None, auto-gera com nome do modelo,
            frequência, TR, dip e filtro Hankel.
        model_name: Nome legível do modelo (ex.: ``"Oklahoma 3"``) para
            incluir no título. Se None, usa apenas ``"Tensor H"``.
        dip_deg: Dip da ferramenta em graus (θ) para exibir no título.
            Se None, omite.
        figsize: Tamanho da figura em polegadas. Default: (28, 12).
        show_interfaces: Se True, marca interfaces em todos os painéis.

    Returns:
        Objeto :class:`matplotlib.figure.Figure`. O caller é responsável
        por ``fig.savefig(...)`` ou ``plt.show()`` — esta função não
        exibe nem salva.

    Raises:
        ImportError: Se matplotlib não estiver instalado.
        ValueError: Se ``freq_idx`` fora do range ``[0, nf)``.

    Example:
        >>> from geosteering_ai.simulation.visualization import (
        ...     plot_tensor_profile
        ... )
        >>> fig = plot_tensor_profile(result, title="Modelo Oklahoma 3")
        >>> fig.savefig("oklahoma3.png", dpi=300, bbox_inches="tight")
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib é necessário para plot_tensor_profile. "
            "Instale via `pip install matplotlib`."
        )

    H = result.H_tensor  # shape: (n_positions, nf, 9)
    if H.ndim != 3 or H.shape[-1] != 9:
        raise ValueError(
            f"H_tensor deve ter shape (n_positions, nf, 9); obtido {H.shape}"
        )

    n_positions, nf, _ = H.shape
    if not (0 <= freq_idx < nf):
        raise ValueError(f"freq_idx={freq_idx} fora do range [0, {nf}).")

    z_obs = result.z_obs
    freq = float(result.freqs_hz[freq_idx])
    cfg = result.cfg

    # ── Título auto-gerado com metadados ricos ────────────────────
    # Detecta número de camadas reais pela mudança de ρₕ (não pelo
    # n_positions como era antes — bug da Sprint 2.8).
    if title is None:
        interfaces = _detect_interfaces(result.rho_h_at_obs, z_obs)
        n_layers_detected = len(interfaces) + 1  # N interfaces → N+1 camadas

        parts = []
        # Cabeçalho: nome do modelo (se fornecido) ou "Tensor H"
        header = f"Tensor H — {model_name}" if model_name else "Tensor H"
        parts.append(f"{header} ({n_layers_detected} cam.)")

        # Frequência em unidades adequadas
        if freq < 1e3:
            parts.append(f"f = {freq:.1f} Hz")
        elif freq < 1e6:
            parts.append(f"f = {freq / 1e3:.3g} kHz")
        else:
            parts.append(f"f = {freq / 1e6:.3g} MHz")

        # Espaçamento TR em m
        parts.append(f"TR = {cfg.tr_spacing_m:.2f} m")

        # Dip (theta) se fornecido
        if dip_deg is not None:
            parts.append(rf"$\theta$ = {dip_deg:.1f}°")

        # Filtro Hankel
        parts.append(f"filtro = {cfg.hankel_filter}")

        # Número de posições
        parts.append(f"N = {result.H_tensor.shape[0]} pts")

        title = " | ".join(parts)

    # ── GridSpec(3, 7): ρ panel (col 0) + tensor 3×6 (cols 1-6) ──
    fig = plt.figure(figsize=figsize)
    _gs = gridspec.GridSpec(
        3,
        7,
        figure=fig,
        width_ratios=[1.8, 1, 1, 1, 1, 1, 1],
        hspace=0.35,
        wspace=0.30,
    )
    fig.suptitle(title, fontsize=15, y=0.99)

    # ── Coluna 0: perfil de resistividade ─────────────────────────
    ax_rho = fig.add_subplot(_gs[:, 0])
    plot_resistivity_profile(
        result,
        ax=ax_rho,
        show_interfaces=show_interfaces,
        title=r"$\rho_h$ e $\rho_v$",
    )

    # ── Interfaces para overlay nos painéis H_ij ──────────────────
    interfaces = _detect_interfaces(result.rho_h_at_obs, z_obs) if show_interfaces else ()

    # ── Colunas 1-6: tensor 3×3 × (Re, Im) ────────────────────────
    # Cada linha (ti) representa o dipolo (0=x, 1=y, 2=z); cada par
    # de colunas (2*tj, 2*tj+1) representa Re, Im do componente j do
    # campo observado.
    flat_idx = 0
    for ti in range(3):
        for tj in range(3):
            name = _TENSOR_NAMES[ti][tj]
            H_component = H[:, freq_idx, flat_idx]  # (n_positions,) complex

            # Painel Re
            ax_re = fig.add_subplot(_gs[ti, 1 + 2 * tj], sharey=ax_rho)
            ax_re.plot(
                np.real(H_component),
                z_obs,
                color=_PALETTE_RE[freq_idx % len(_PALETTE_RE)],
                linewidth=1.5,
            )
            ax_re.set_title(rf"$\mathrm{{Re}}({name})$", fontsize=11)
            _style_tensor_panel(
                ax_re,
                interfaces,
                is_bottom=(ti == 2),
                xlabel_text="Amplitude",
            )

            # Painel Im
            ax_im = fig.add_subplot(_gs[ti, 1 + 2 * tj + 1], sharey=ax_rho)
            ax_im.plot(
                np.imag(H_component),
                z_obs,
                color=_PALETTE_IM[freq_idx % len(_PALETTE_IM)],
                linewidth=1.5,
            )
            ax_im.set_title(rf"$\mathrm{{Im}}({name})$", fontsize=11)
            _style_tensor_panel(
                ax_im,
                interfaces,
                is_bottom=(ti == 2),
                xlabel_text="Amplitude",
            )

            flat_idx += 1

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers privados
# ──────────────────────────────────────────────────────────────────────────────


def _detect_interfaces(
    rho_h: np.ndarray, z_obs: np.ndarray, *, tol: float = 1e-6
) -> tuple[float, ...]:
    """Detecta interfaces de camadas a partir do perfil de ρₕ amostrado.

    Uma interface é marcada no ponto médio entre duas amostras consecutivas
    cuja resistividade difere além de ``tol`` (relativo). O método é
    heurístico mas suficiente para perfis tipicamente bem amostrados.

    Args:
        rho_h: Array (n,) de resistividade horizontal nas posições.
        z_obs: Array (n,) de profundidades correspondentes.
        tol: Tolerância relativa para detectar mudança de camada.

    Returns:
        Tupla de profundidades (m) onde interfaces foram detectadas.
    """
    if rho_h.size < 2:
        return ()

    # Mudança relativa consecutiva (evita divisão por zero com fator min)
    denom = np.maximum(np.abs(rho_h[:-1]), 1e-30)
    rel_change = np.abs(rho_h[1:] - rho_h[:-1]) / denom
    jumps = np.where(rel_change > tol)[0]

    # Profundidade da interface = média entre as duas amostras
    interfaces = tuple(float((z_obs[i] + z_obs[i + 1]) / 2.0) for i in jumps)
    return interfaces


def _style_tensor_panel(
    ax,
    interfaces: Sequence[float],
    *,
    is_bottom: bool,
    xlabel_text: str,
) -> None:
    """Aplica estilo padrão a um painel de tensor H.

    Ajusta tick_params, grid, xlabel (apenas na linha inferior) e
    desenha interfaces de camadas como axhline tracejadas.
    """
    ax.tick_params(labelleft=False)  # y-ticks herdados de ax_rho via sharey
    for z_interf in interfaces:
        ax.axhline(y=z_interf, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.grid(True, linestyle=":", alpha=0.5)
    if is_bottom:
        ax.set_xlabel(xlabel_text, fontsize=10)


__all__ = [
    "plot_tensor_profile",
    "plot_resistivity_profile",
]
