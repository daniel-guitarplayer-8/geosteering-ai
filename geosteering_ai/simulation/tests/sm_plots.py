# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_plots.py                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Plots EM e Resistividade              ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Atualizado  : 2026-04-18 (PlotStyle + tensor 3×6 + modos físicos)       ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : matplotlib + Qt backend (FigureCanvasQTAgg)                ║
# ║  Dependências: matplotlib ≥ 3.5, numpy                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Encapsula o canvas matplotlib + helpers de plotagem para os campos    ║
# ║    eletromagnéticos (Hxx..Hzz, 9 componentes), perfis de resistividade   ║
# ║    (ρh, ρv) e geosinais derivados (USD/UAD/UHR/UHA). Comparação lado-a-  ║
# ║    lado entre Numba e Fortran quando ambos estão disponíveis.             ║
# ║                                                                           ║
# ║  MODOS DE PLOT IMPLEMENTADOS                                              ║
# ║    ┌────────────────────────────┬────────────────────────────────────┐   ║
# ║    │  Tensor H — 3×6 Re/Im      │  plot_tensor_full (referência)     │   ║
# ║    │  Magnitude + Fase          │  plot_em_profile (mag/phase)       │   ║
# ║    │  Re/Im por componente      │  plot_em_profile (kind="reim")     │   ║
# ║    │  Amplitude dB (log)        │  plot_em_profile (kind="db")       │   ║
# ║    │  Anisotropia λ²=ρᵥ/ρₕ      │  plot_anisotropy                  │   ║
# ║    │  Benchmark overlay         │  plot_benchmark_compare            │   ║
# ║    │  Geosinais USD/UAD/UHR/UHA │  plot_geosignals                   │   ║
# ║    │  Perfil ρₕ/ρᵥ + camadas    │  plot_resistivity_profile          │   ║
# ║    └────────────────────────────┴────────────────────────────────────┘   ║
# ║                                                                           ║
# ║  API PÚBLICA                                                              ║
# ║    • PlotStyle dataclass — parametrização global de estilo (cores, dpi,  ║
# ║       font size, line width, grid, precisão de eixos)                    ║
# ║    • EMCanvas      — QWidget com FigureCanvasQTAgg encapsulado            ║
# ║    • apply_style(style) — aplica PlotStyle ao matplotlib rcParams        ║
# ║    • plot_tensor_full(canvas, ...)      — grid 3×6 Re/Im (referência)   ║
# ║    • plot_em_profile(canvas, ..., kind) — mag/phase, reim, db           ║
# ║    • plot_resistivity_profile(canvas, ...)                                ║
# ║    • plot_benchmark_compare(canvas, ...) — Numba vs Fortran sobreposto   ║
# ║    • plot_geosignals(canvas, ...)        — USD/UAD/UHR/UHA grid         ║
# ║    • plot_anisotropy(canvas, ...)        — λ = √(ρᵥ/ρₕ) perfil          ║
# ║    • COMPONENT_NAMES                      — nomes canônicos Hxx..Hzz    ║
# ║    • PLOT_KINDS                           — lista de modos físicos      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Plotagem de campos EM e perfis geológicos com canvas Qt-embedded.

O módulo fornece ``EMCanvas`` — um ``QWidget`` com figura matplotlib
redimensionável — e funções utilitárias para os tipos de plot mais
comuns no Simulation Manager (perfil, campo EM, comparação benchmark,
grid tensor 3×6 Re/Im como referência visual do paper).

Exemplo de uso::

    >>> style = PlotStyle(dpi=150, line_width=1.8, font_size=11)
    >>> apply_style(style)
    >>> plot_tensor_full(canvas, H_tensor, z_obs, freqs, trs, dips,
    ...                  title="Oklahoma 15 cam · TR1 · θ=0° · f=6 kHz",
    ...                  style=style)

Note:
    Todos os labels/valores formatados usam ponto (``.``) como separador
    decimal (jamais vírgula), em linha com a convenção científica e os
    testes automatizados da suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .sm_qt_compat import QtWidgets

# Import matplotlib com backend Qt — fallback leve se matplotlib ausente
try:
    import matplotlib

    matplotlib.use("QtAgg", force=False)
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False
    FigureCanvasQTAgg = object  # type: ignore
    Figure = object  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
# Constantes — nomes canônicos do tensor H
# ──────────────────────────────────────────────────────────────────────────
COMPONENT_NAMES: List[str] = [
    "Hxx",
    "Hxy",
    "Hxz",
    "Hyx",
    "Hyy",
    "Hyz",
    "Hzx",
    "Hzy",
    "Hzz",
]

# Modos físicos suportados pelo plot_em_profile
PLOT_KINDS: List[str] = [
    "Magnitude + Fase",
    "Re + Im",
    "Magnitude (dB)",
    "Só Re",
    "Só Im",
    "Só Magnitude",
    "Só Fase",
]


# ──────────────────────────────────────────────────────────────────────────
# PlotStyle — parametrização global
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class PlotStyle:
    """Parâmetros de estilo para todas as figuras do Simulation Manager.

    Attributes:
        dpi: Resolução em pontos por polegada (padrão 100). Valores 120–200
            são recomendados para apresentações; 300+ para publicação.
        font_family: Família tipográfica (``"DejaVu Sans"`` por padrão no
            matplotlib, ``"Helvetica"`` em macOS, ``"Arial"`` em Windows).
        font_size: Tamanho base em pontos — títulos/labels escalam a partir
            deste valor.
        line_width: Espessura padrão das linhas de plot (mag, fase, Re, Im).
        grid: Habilita grid de fundo.
        grid_alpha: Transparência do grid (0.0-1.0).
        color_real: Cor das curvas de parte real do tensor H.
        color_imag: Cor das curvas de parte imaginária.
        color_mag: Cor das curvas de magnitude.
        color_phase: Cor das curvas de fase.
        color_rho_h: Cor de ρₕ nos perfis de resistividade.
        color_rho_v: Cor de ρᵥ nos perfis (tracejado).
        color_numba: Cor para curvas Numba em comparação benchmark.
        color_fortran: Cor para marcadores Fortran em comparação.
        color_layer_boundary: Cor das linhas horizontais de interfaces.
        axis_precision: Nº de casas decimais nos ticks dos eixos.
        show_layer_boundaries: Se ``True``, desenha interfaces pontilhadas.
        palette: Nome de paleta matplotlib para curvas multi-freq/multi-TR
            (``"tab10"``, ``"viridis"``, ``"cividis"``, ``"Set1"``...).
        background: Cor de fundo do painel de plot.
        tight_layout: Se ``True``, usa ``constrained_layout=True`` nos axes.
    """

    dpi: int = 100
    font_family: str = "DejaVu Sans"
    font_size: int = 10
    line_width: float = 1.6
    grid: bool = True
    grid_alpha: float = 0.35
    color_real: str = "#1f4ea8"
    color_imag: str = "#a3272f"
    color_mag: str = "#1f4ea8"
    color_phase: str = "#a3272f"
    color_rho_h: str = "#1f77b4"
    color_rho_v: str = "#ff7f0e"
    color_numba: str = "#1f77b4"
    color_fortran: str = "#d62728"
    color_layer_boundary: str = "#7f7f7f"
    axis_precision: int = 4
    show_layer_boundaries: bool = True
    palette: str = "tab10"
    background: str = "#ffffff"
    tight_layout: bool = True

    # ── Customizações adicionais (2026-04-18) ────────────────────────────
    # use_latex: ativa rcParams['text.usetex']. Exige LaTeX instalado
    #   (pdflatex, dvipng). Default False — usuário habilita em Preferências.
    # use_mathtext: sempre renderiza $…$ com fonte Computer Modern do
    #   matplotlib nativo (sem dependência externa). Default True.
    # legend_location / title_location: posicionamento consistente em todos
    #   os plots (legendas e títulos).
    # minor_ticks: ativa eixos menores para leitura mais fina.
    # spine_width: espessura das bordas do axes.
    # line_style: estilo de linha padrão ("-", "--", "-.", ":").
    use_latex: bool = False
    use_mathtext: bool = True
    legend_location: str = "best"
    title_location: str = "center"
    minor_ticks: bool = False
    spine_width: float = 1.0
    line_style: str = "-"
    marker_size: float = 4.0
    marker_style: str = "o"

    # ── v2.6: tema dark/light/auto para integração com UI VSCode-style ────
    # theme="dark" sobrescreve background e cores de eixos para integrar
    # visualmente com o tema dark da app. Cores de curva também são
    # adaptadas via _theme_color_overrides() abaixo.
    # theme="auto" segue a app (default) — equivalente a "dark" quando a
    # MainWindow usa o stylesheet escuro.
    theme: str = "auto"  # "light" | "dark" | "auto"


def apply_style(style: PlotStyle) -> None:
    """Aplica um ``PlotStyle`` globalmente aos rcParams do matplotlib.

    Args:
        style: Instância de ``PlotStyle`` contendo as preferências.

    Note:
        Esta função modifica o estado global do matplotlib. Todas as
        figuras criadas após esta chamada herdam o novo estilo.
    """
    if not _HAS_MPL:
        return
    # v2.6 U1+P1: tema dark/light. "auto" segue app (default = dark hoje).
    theme = getattr(style, "theme", "auto")
    is_dark = theme == "dark" or theme == "auto"
    if is_dark:
        bg = "#1e1e1e"
        fg = "#d4d4d4"
        grid_color = "#3c3c3c"
    else:
        bg = style.background
        fg = "#000000"
        grid_color = "#cccccc"
    rc = {
        "figure.dpi": style.dpi,
        "savefig.dpi": max(style.dpi, 150),
        "font.family": style.font_family,
        "font.size": style.font_size,
        "axes.labelsize": style.font_size,
        "axes.titlesize": style.font_size + 1,
        "xtick.labelsize": max(6, style.font_size - 2),
        "ytick.labelsize": max(6, style.font_size - 2),
        "legend.fontsize": max(6, style.font_size - 2),
        "lines.linewidth": style.line_width,
        "lines.linestyle": style.line_style,
        "lines.markersize": style.marker_size,
        "axes.grid": style.grid,
        "grid.alpha": style.grid_alpha,
        "axes.facecolor": bg,
        "figure.facecolor": bg,
        "axes.linewidth": style.spine_width,
        "axes.titlelocation": style.title_location,
        "legend.loc": style.legend_location,
        "axes.formatter.use_locale": False,
        "axes.unicode_minus": True,
        "xtick.minor.visible": style.minor_ticks,
        "ytick.minor.visible": style.minor_ticks,
        # v2.6: cores de eixos/texto/grid adaptadas ao tema
        "axes.edgecolor": fg,
        "axes.labelcolor": fg,
        "axes.titlecolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "text.color": fg,
        "grid.color": grid_color,
        "savefig.facecolor": bg,
        "savefig.edgecolor": bg,
    }
    # LaTeX externo (requer TeX no PATH). Usuário assume responsabilidade
    # de instalar o ambiente. Se o render falhar no runtime, matplotlib
    # emite warnings e o plot ainda funciona com mathtext nativo.
    if style.use_latex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = (
            r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}"
        )
    else:
        rc["text.usetex"] = False
        # Mathtext nativo (renderização de $…$ com Computer Modern). Ativa
        # mesmo quando use_latex=False para suportar legendas com
        # expressões como r"$\rho_h$", r"$\lambda$".
        rc["mathtext.default"] = "regular" if not style.use_mathtext else "it"
        rc["mathtext.fontset"] = "dejavusans"
    matplotlib.rcParams.update(rc)


# ──────────────────────────────────────────────────────────────────────────
# Canvas Qt-embedded
# ──────────────────────────────────────────────────────────────────────────


class EMCanvas(QtWidgets.QWidget):
    """Canvas matplotlib embutido em ``QWidget`` para uso em abas PyQt.

    Attributes:
        figure: ``matplotlib.figure.Figure`` — acesso direto para customização.
        canvas: ``FigureCanvasQTAgg`` — widget Qt renderizador.
        style: ``PlotStyle`` — cache local (pode ser substituído em tempo
            de execução; a próxima chamada de plot honra o novo estilo).
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        figsize: Tuple[float, float] = (10, 6),
        style: Optional[PlotStyle] = None,
    ) -> None:
        super().__init__(parent)
        self.style = style or PlotStyle()
        if not _HAS_MPL:
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(
                QtWidgets.QLabel(
                    "matplotlib não instalado — plots desabilitados.\n"
                    "Instale com: pip install matplotlib"
                )
            )
            self.figure = None
            self.canvas = None
            return

        self.figure = Figure(figsize=figsize, constrained_layout=True, dpi=self.style.dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear(self) -> None:
        """Limpa a figura atual."""
        if self.figure is None:
            return
        self.figure.clear()
        self.canvas.draw_idle()

    def draw(self) -> None:
        """Dispara redesenho imediato."""
        if self.canvas is not None:
            self.canvas.draw_idle()

    def set_style(self, style: PlotStyle) -> None:
        """Atualiza o estilo e o dpi da figura sem recriar o canvas."""
        self.style = style
        if self.figure is not None:
            try:
                self.figure.set_dpi(style.dpi)
                self.figure.set_facecolor(style.background)
            except Exception:
                pass

    def save(self, path: str, dpi: Optional[int] = None) -> None:
        """Salva o conteúdo atual em disco (PNG/PDF/SVG via extensão)."""
        if self.figure is not None:
            self.figure.savefig(
                path,
                dpi=dpi or max(self.style.dpi, 150),
                bbox_inches="tight",
                facecolor=self.style.background,
            )


# ──────────────────────────────────────────────────────────────────────────
# Utilidades internas
# ──────────────────────────────────────────────────────────────────────────


def _boundaries_from_thicknesses(
    thicknesses: Sequence[float], top: float = 0.0
) -> np.ndarray:
    """Gera vetor de topos das interfaces a partir de espessuras das
    camadas internas (n-2 valores para n camadas, conforme convenção
    tatu.x / TIV).
    """
    thick = np.asarray(thicknesses, dtype=np.float64)
    if thick.size == 0:
        return np.array([top], dtype=np.float64)
    bounds = [top]
    for t in thick:
        bounds.append(bounds[-1] + float(t))
    return np.asarray(bounds, dtype=np.float64)


def _draw_layer_boundaries(
    ax: Any, thicknesses: Sequence[float], style: PlotStyle, top: float = 0.0
) -> None:
    """Desenha linhas horizontais finas nas interfaces das camadas.

    Aceita listas, tuplas e ``np.ndarray`` — evita ``bool(array)`` ambíguo.
    """
    if not style.show_layer_boundaries:
        return
    thick_arr = (
        np.asarray(thicknesses, dtype=np.float64)
        if thicknesses is not None
        else np.array([])
    )
    if thick_arr.size == 0:
        return
    bounds = _boundaries_from_thicknesses(thick_arr.tolist(), top=top)
    for b in bounds:
        ax.axhline(
            y=float(b),
            color=style.color_layer_boundary,
            lw=0.6,
            ls="--",
            alpha=0.55,
            zorder=0,
        )


def _axis_formatter(precision: int) -> Any:
    """Formatter matplotlib com `precision` casas decimais e ponto."""
    from matplotlib.ticker import FormatStrFormatter

    return FormatStrFormatter(f"%.{max(0, int(precision))}f")


def _palette_colors(n: int, palette: str) -> List[str]:
    """Retorna ``n`` cores distintas a partir do nome de paleta."""
    if not _HAS_MPL or n <= 0:
        return ["#1f77b4"]
    try:
        cmap = matplotlib.colormaps.get_cmap(palette)
    except Exception:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    if hasattr(cmap, "N") and cmap.N <= 20:
        return [matplotlib.colors.to_hex(cmap(i % cmap.N)) for i in range(n)]
    return [matplotlib.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def _safe_log10(x: np.ndarray) -> np.ndarray:
    """log10 protegido contra zeros (clip em 1e-30)."""
    return np.log10(np.clip(x, 1e-30, None))


# ──────────────────────────────────────────────────────────────────────────
# Plot — perfil de resistividade
# ──────────────────────────────────────────────────────────────────────────


def _rho_per_z(
    z_obs: np.ndarray,
    rho_per_layer: Sequence[float],
    thicknesses: Sequence[float],
) -> np.ndarray:
    """Retorna ρ[z] replicado por camada em cada posição ``z_obs``.

    Reproduz a convenção do simulador Fortran (``PerfilaAnisoOmp.f08``)
    gravada no ``.dat`` 22-col: para cada z_obs, identifica a camada
    correspondente e retorna a resistividade dessa camada. Plot direto
    via ``semilogx(rho_per_z, z_obs)`` produz o perfil step-function
    **idêntico** ao de ``buildValidamodels.py`` (linha 588-592).

    Convenção de interfaces (alinhada a SM v2.3 ``positions_z``):
      - z=0 na primeira interface; camada 0 (topo) abaixo de z=0.
      - Interfaces internas em z=esp[0], esp[0]+esp[1], …, Σesp.
      - Camada índice i ocupa [cumsum[i-1], cumsum[i]] (com cumsum[-1]=0).

    Args:
        z_obs: Array 1D de posições de medição (m).
        rho_per_layer: Lista de N resistividades por camada (inclui semi-espaços).
        thicknesses: Espessuras das N-2 camadas internas, ou N valores
            (primeira e última ignoradas — semi-espaços).

    Returns:
        Array 1D ``float64`` com mesma forma de ``z_obs``, ρ[camada(z)].
    """
    z = np.asarray(z_obs, dtype=np.float64).ravel()
    rho = np.asarray(rho_per_layer, dtype=np.float64)
    thick = np.asarray(thicknesses, dtype=np.float64)
    n = rho.size
    # Normaliza thicknesses: precisa de N-2 espessuras internas.
    if thick.size == n - 2:
        internals = thick
    elif thick.size == n:
        # Caso 1: primeira e última s\u00e3o semi-espa\u00e7os (h=0 ou ∞) \u2014 descarta.
        internals = thick[1:-1]
    elif thick.size == n - 1:
        # Borda: N camadas, N-1 espessuras internas (topo tem h=0).
        internals = thick[1:]
    else:
        internals = thick[: max(n - 2, 0)]
    # Interfaces cumulativas: [0, esp[0], esp[0]+esp[1], ...]
    boundaries = np.concatenate(([0.0], np.cumsum(internals)))  # tamanho n-1
    # Para cada z, searchsorted retorna o \u00edndice da camada
    # side='right' \u2192 z no boundary (ex.: z=0.0) cai na camada de cima
    idx = np.searchsorted(boundaries, z, side="right")
    idx = np.clip(idx, 0, n - 1)
    return rho[idx]


def plot_resistivity_profile(
    canvas: EMCanvas,
    rho_h: Sequence[float],
    rho_v: Sequence[float],
    thicknesses: Sequence[float],
    title: str = "Perfil de Resistividade",
    style: Optional[PlotStyle] = None,
    z_obs: Optional[np.ndarray] = None,
    scale_mode: str = "rho_log10",
) -> None:
    """Desenha ρₕ/ρᵥ vs profundidade replicando a convenção de ``buildValidamodels.py``.

    Quando ``z_obs`` é fornecido, usa o helper :func:`_rho_per_z` para
    construir os arrays ρ por posição de medição (idêntico ao que o Fortran
    grava no ``.dat`` 22-col) e chama ``ax.semilogx(rho_per_z, z_obs)`` —
    formato bit-exato do plot do buildValidamodels.py (linha 588-592).

    Fallback (quando ``z_obs`` é None): usa step-function manual construída
    a partir de thicknesses (convenção legada pre-v2.3).

    Args:
        canvas: EMCanvas destino.
        rho_h, rho_v: Listas de N resistividades por camada (Ω·m).
        thicknesses: Espessuras das N-2 camadas internas (m).
        title: Título do plot.
        style: Override de PlotStyle.
        z_obs: Vetor 1D/2D de posições de medição (m). Default None usa
            margem legada de ±5 m do Σesp.
    """
    if canvas.figure is None:
        return
    style = style or canvas.style
    apply_style(style)
    canvas.figure.clear()
    ax = canvas.figure.add_subplot(1, 1, 1)

    rho_h_arr = np.asarray(rho_h, dtype=np.float64)
    rho_v_arr = np.asarray(rho_v, dtype=np.float64)
    thick_arr = np.asarray(thicknesses, dtype=np.float64)

    # Janela de investigação efetiva (z_obs da simulação), quando disponível.
    if z_obs is not None:
        z_flat = np.asarray(z_obs, dtype=np.float64).ravel()
        z_win_min = float(np.min(z_flat)) if z_flat.size > 0 else None
        z_win_max = float(np.max(z_flat)) if z_flat.size > 0 else None
    else:
        z_flat = None
        z_win_min = z_win_max = None

    n = len(rho_h_arr)
    have_valid_layers = n >= 2 and thick_arr.size in (n - 2, n - 1, n)

    if z_flat is not None and z_flat.size > 0 and have_valid_layers:
        # Caminho bit-exato a buildValidamodels.py: plot direto com \u03c1 por z_obs.
        rho_h_z = _rho_per_z(z_flat, rho_h_arr, thick_arr)
        rho_v_z = _rho_per_z(z_flat, rho_v_arr, thick_arr)
        # Ordena por z para evitar cruzamentos visuais se z_obs n\u00e3o monot\u00f4nico
        order = np.argsort(z_flat)
        z_sorted = z_flat[order]
        # v2.4c: escala linear vs log10 conforme scale_mode (default rho_log10)
        plot_fn = ax.plot if scale_mode == "rho_linear" else ax.semilogx
        plot_fn(
            rho_h_z[order],
            z_sorted,
            color=style.color_rho_h,
            linewidth=style.line_width + 0.6,
            label=r"$\rho_h$",
        )
        plot_fn(
            rho_v_z[order],
            z_sorted,
            color=style.color_rho_v,
            linewidth=style.line_width + 0.6,
            linestyle="--",
            label=r"$\rho_v$",
        )
        # Interfaces reais como linhas horizontais tracejadas (mesma conven\u00e7\u00e3o
        # de buildValidamodels.py L594: ax_rho.axhline(y=interf, ls='--'))
        if thick_arr.size >= 1:
            # Normaliza thicknesses para N-2 internas
            if thick_arr.size == n - 2:
                internals = thick_arr
            elif thick_arr.size == n:
                internals = thick_arr[1:-1]
            elif thick_arr.size == n - 1:
                internals = thick_arr[1:]
            else:
                internals = thick_arr[: max(n - 2, 0)]
            boundaries = np.concatenate(([0.0], np.cumsum(internals)))
            for interf in boundaries:
                ax.axhline(
                    y=float(interf),
                    color="#000000",
                    linestyle="--",
                    lw=1.0,
                    alpha=0.6,
                )
    else:
        # Fallback legado: step-function manual (pre-v2.3).
        if n >= 3 and len(thick_arr) == n - 2:
            top = 0.0
            boundaries = [top]
            for t in thick_arr:
                boundaries.append(boundaries[-1] + float(t))
            top_edge = boundaries[0] - 5.0
            bot_edge = boundaries[-1] + 5.0
            depths = [top_edge] + boundaries + [bot_edge]
        else:
            depths = np.linspace(0.0, 1.0, n + 1).tolist()
        y_edges = np.array(depths, dtype=np.float64)
        centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        _draw_layer_boundaries(ax, thick_arr, style)
        ax.step(
            rho_h_arr,
            centers,
            where="mid",
            color=style.color_rho_h,
            label=r"$\rho_h$",
            lw=style.line_width + 0.3,
        )
        ax.step(
            rho_v_arr,
            centers,
            where="mid",
            color=style.color_rho_v,
            label=r"$\rho_v$",
            lw=style.line_width + 0.3,
            ls="--",
        )
        # v2.4c: respeita scale_mode no fallback legado
        if scale_mode != "rho_linear":
            ax.set_xscale("log")

    ax.set_xlabel(r"Resistividade ($\Omega\cdot m$)")
    ax.set_ylabel("Profundidade (m)")
    if z_win_min is not None and z_win_max is not None:
        ax.set_ylim(z_win_max, z_win_min)  # invertido (depth ↓)
    else:
        ax.invert_yaxis()
    ax.grid(True, which="both", ls=":", alpha=style.grid_alpha)
    ax.legend(loc="lower right")
    ax.set_title(title)
    canvas.draw()


# ──────────────────────────────────────────────────────────────────────────
# Plot — Tensor H completo 3×6 Re/Im (grid de referência)
# ──────────────────────────────────────────────────────────────────────────


def _tensor_style(base: PlotStyle) -> PlotStyle:
    """Clona ``base`` com ajustes de fonte/linha dedicados ao Tensor H (E15).

    O plot 3×6 (+ρ) é denso; fontes maiores evitam overflow dos títulos
    ``Re(Hxx)..Im(Hzz)`` e melhoram legibilidade em telas ≥ 1440p. Não
    altera o ``PlotStyle`` base (outros plots continuam iguais).
    """
    from dataclasses import replace as _dc_replace

    return _dc_replace(
        base,
        font_size=max(base.font_size, 12),
        line_width=max(base.line_width, 1.8),
    )


def plot_tensor_full(
    canvas: EMCanvas,
    H_tensor: np.ndarray,
    z_obs: Optional[np.ndarray],
    freqs: Sequence[float],
    trs: Sequence[float],
    dips: Sequence[float],
    title: str = "Tensor H — componentes Re/Im",
    thicknesses: Optional[Sequence[float]] = None,
    rho_h: Optional[Sequence[float]] = None,
    rho_v: Optional[Sequence[float]] = None,
    style: Optional[PlotStyle] = None,
    include_resistivity: bool = True,
    tr_mask: Optional[Sequence[bool]] = None,
    ang_mask: Optional[Sequence[bool]] = None,
    freq_mask: Optional[Sequence[bool]] = None,
    combos: Optional[Sequence[Tuple[int, int, int]]] = None,
    scale_mode: str = "re_im",
    layout: str = "default",
) -> None:
    """Plota o tensor H em grid 3×6 com Re/Im pareados por componente.

    Layout inspirado na figura de referência do pacote (imagem anexa
    pelo usuário): coluna esquerda opcional com perfil de resistividade,
    demais colunas agrupadas em pares (Re, Im) para cada componente
    ``Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz``.

    Args:
        canvas: ``EMCanvas`` destino.
        H_tensor: Tensor complexo ``(nTR, nAng, n_pos, nf, 9)``.
        z_obs: Profundidades ``(nAng, n_pos)`` ou ``(n_pos,)``.
        freqs: Frequências (Hz).
        trs: Espaçamentos TR (m).
        dips: Ângulos de dip (graus).
        title: Título principal.
        thicknesses: Espessuras das camadas internas (para linhas guia).
        rho_h, rho_v: Vetores de resistividade (para coluna auxiliar).
        style: Override opcional do ``PlotStyle`` (default = canvas.style).
        include_resistivity: Se ``True`` (e ``rho_h`` fornecido), inclui
            coluna adicional com o perfil de resistividade à esquerda.
    """
    if canvas.figure is None:
        return
    # v2.4/E15: fontes maiores por padrão no Tensor H (3×6 é denso)
    style = _tensor_style(style or canvas.style)
    apply_style(style)
    canvas.figure.clear()

    H = np.asarray(H_tensor)
    if H.ndim != 5:
        ax = canvas.figure.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            f"H_tensor shape inesperado: {H.shape}",
            ha="center",
            transform=ax.transAxes,
        )
        canvas.draw()
        return

    nTR, nAng, n_pos, nf, _ = H.shape
    # v2.4/E12: resolve m\u00e1scaras TR/ang/freq. Cada combina\u00e7\u00e3o selecionada
    # vira uma curva sobreposta (cor distinta) nos 18 subplots Re/Im do tensor.
    # v2.4c: combos (lista de tuplas) tem prioridade; fallback p/ masks legadas.
    if combos is not None:
        _combo_list_v24c: List[Tuple[int, int, int]] = [
            (int(itr), int(iang), int(ifq))
            for (itr, iang, ifq) in combos
            if 0 <= int(itr) < nTR and 0 <= int(iang) < nAng and 0 <= int(ifq) < nf
        ]
        _used_trs = {c[0] for c in _combo_list_v24c}
        _used_angs = {c[1] for c in _combo_list_v24c}
        _used_freqs = {c[2] for c in _combo_list_v24c}
        tr_mask = [i in _used_trs for i in range(nTR)]
        ang_mask = [i in _used_angs for i in range(nAng)]
        freq_mask = [i in _used_freqs for i in range(nf)]
    tr_sel = [
        i for i in range(nTR) if tr_mask is None or (i < len(tr_mask) and tr_mask[i])
    ]
    ang_sel = [
        i for i in range(nAng) if ang_mask is None or (i < len(ang_mask) and ang_mask[i])
    ]
    freq_sel = [
        i for i in range(nf) if freq_mask is None or (i < len(freq_mask) and freq_mask[i])
    ]
    if not tr_sel or not ang_sel or not freq_sel:
        ax = canvas.figure.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            "Selecione ao menos 1 combina\u00e7\u00e3o TR/\u00c2ngulo/Frequ\u00eancia.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=style.font_size + 1,
        )
        canvas.draw()
        return
    # v2.5: layout pode forçar inclusão/exclusão do perfil ρ
    #   "tensor_3x6"      → grade 3×6 sem ρ (clássico v2.4)
    #   "tensor_3x7_rho"  → grade 3×7 com ρ_h/ρ_v à esquerda
    #   "default"         → respeita include_resistivity explícito
    if layout == "tensor_3x6":
        include_resistivity = False
    elif layout == "tensor_3x7_rho":
        include_resistivity = True
    have_rho = include_resistivity and rho_h is not None and len(rho_h) > 0

    # Grid: 3 linhas × (1 col. ρ + 6 pares Re/Im = 6 cols × 2 componentes)
    # Implementação: se have_rho, usamos GridSpec com coluna esquerda mais
    # larga. Caso contrário, simplificamos para 3×6 com 2 subplots por célula.
    n_comp_rows = 3  # Hxx/Hxy/Hxz, Hyx/Hyy/Hyz, Hzx/Hzy/Hzz
    pair_cols = 3  # 3 componentes por linha × 2 (Re+Im) = 6 colunas
    total_cols = 2 * pair_cols + (1 if have_rho else 0)

    from matplotlib.gridspec import GridSpec

    width_ratios = (
        [1.4] + [1.0] * (2 * pair_cols) if have_rho else [1.0] * (2 * pair_cols)
    )
    gs = GridSpec(
        n_comp_rows,
        total_cols,
        figure=canvas.figure,
        width_ratios=width_ratios,
        wspace=0.32,
        hspace=0.38,
    )

    # Cores
    c_real = style.color_real
    c_imag = style.color_imag

    # ── Determina range da janela de investigação a partir de z_obs ──────
    # Convenção buildValidamodels.py: z_obs ∈ [−h1, tj−h1], interfaces em
    # [0, Σesp]. O eixo Y do perfil ρ deve cobrir a MESMA faixa do tensor H
    # (sharey-equivalente). Se z_obs não estiver disponível, cai para uma
    # margem de 5 m em torno das camadas (comportamento legado).
    if z_obs is not None:
        z_flat = np.asarray(z_obs, dtype=np.float64).ravel()
        z_win_min = float(np.min(z_flat)) if z_flat.size > 0 else None
        z_win_max = float(np.max(z_flat)) if z_flat.size > 0 else None
    else:
        z_win_min = z_win_max = None

    # Coluna opcional ρ — convenção bit-exata a buildValidamodels.py (L582-600):
    # usa ρ replicada em cada z_obs (_rho_per_z) + axhline nas interfaces.
    if have_rho:
        ax_rho = canvas.figure.add_subplot(gs[:, 0])
        thick_arr = (
            np.asarray(thicknesses, dtype=np.float64)
            if thicknesses is not None
            else np.array([], dtype=np.float64)
        )
        rh = np.asarray(rho_h, dtype=np.float64)
        rv = np.asarray(rho_v, dtype=np.float64) if rho_v is not None else rh.copy()

        # z_flat \u2014 posi\u00e7\u00f5es de medi\u00e7\u00e3o efetivas (1D)
        z_flat_rho: Optional[np.ndarray] = None
        if z_obs is not None:
            z_flat_rho = np.asarray(z_obs, dtype=np.float64).ravel()
        have_valid_layers = rh.size >= 2 and thick_arr.size in (
            rh.size - 2,
            rh.size - 1,
            rh.size,
        )

        if z_flat_rho is not None and z_flat_rho.size > 0 and have_valid_layers:
            # Caminho buildValidamodels.py: \u03c1[z_obs] \u2192 semilogx direto
            rho_h_z = _rho_per_z(z_flat_rho, rh, thick_arr)
            rho_v_z = _rho_per_z(z_flat_rho, rv, thick_arr)
            order = np.argsort(z_flat_rho)
            z_sorted = z_flat_rho[order]
            ax_rho.semilogx(
                rho_h_z[order],
                z_sorted,
                color=style.color_rho_h,
                linewidth=style.line_width + 0.6,
                label=r"$\rho_h$",
            )
            ax_rho.semilogx(
                rho_v_z[order],
                z_sorted,
                color=style.color_rho_v,
                linewidth=style.line_width + 0.6,
                linestyle="--",
                label=r"$\rho_v$",
            )
            # Interfaces horizontais (idem buildValidamodels.py L594)
            if thick_arr.size == rh.size - 2:
                internals = thick_arr
            elif thick_arr.size == rh.size:
                internals = thick_arr[1:-1]
            elif thick_arr.size == rh.size - 1:
                internals = thick_arr[1:]
            else:
                internals = thick_arr[: max(rh.size - 2, 0)]
            boundaries = np.concatenate(([0.0], np.cumsum(internals)))
            for interf in boundaries:
                ax_rho.axhline(
                    y=float(interf),
                    color="#000000",
                    linestyle="--",
                    lw=1.0,
                    alpha=0.6,
                )
        else:
            # Fallback legado \u2014 sem z_obs, usa step-function manual
            if rh.size >= 3 and thick_arr.size == rh.size - 2:
                bounds = [0.0]
                for t in thick_arr:
                    bounds.append(bounds[-1] + float(t))
                top_edge = bounds[0] - 5.0
                bot_edge = bounds[-1] + 5.0
                depths = [top_edge] + bounds + [bot_edge]
            else:
                depths = np.linspace(0.0, 1.0, rh.size + 1).tolist()
            y_edges = np.asarray(depths, dtype=np.float64)
            centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            _draw_layer_boundaries(ax_rho, thick_arr, style)
            ax_rho.step(
                rh,
                centers,
                where="mid",
                color=style.color_rho_h,
                label=r"$\rho_h$",
                lw=style.line_width + 0.2,
            )
            ax_rho.step(
                rv,
                centers,
                where="mid",
                color=style.color_rho_v,
                label=r"$\rho_v$",
                lw=style.line_width + 0.2,
                ls="--",
            )
            ax_rho.set_xscale("log")

        ax_rho.set_xlabel(r"Resistividade ($\Omega\cdot m$)")
        ax_rho.set_ylabel("Profundidade (m)")
        ax_rho.set_title(r"$\rho_h$ e $\rho_v$ (modelo verdadeiro)")
        # Ajusta limites Y à janela efetiva z_obs quando disponível — mesma
        # convenção de buildValidamodels.py.
        if z_win_min is not None and z_win_max is not None:
            ax_rho.set_ylim(z_win_max, z_win_min)  # invertido (depth ↓)
        else:
            ax_rho.invert_yaxis()
        ax_rho.grid(True, which="both", ls=":", alpha=style.grid_alpha)
        ax_rho.legend(loc="lower right", fontsize=max(6, style.font_size - 2))

    # v2.4/E12: itera sobre todas as combinações TR/ângulo/freq marcadas.
    # Cada combinação é uma curva sobreposta (cor distinta da paleta).
    thick_arr = (
        np.asarray(thicknesses, dtype=np.float64)
        if thicknesses is not None
        else np.array([], dtype=np.float64)
    )
    n_curves = len(tr_sel) * len(ang_sel) * len(freq_sel)
    colors = _palette_colors(max(n_curves, 1), style.palette)

    comp_order = [
        ["Hxx", "Hxy", "Hxz"],
        ["Hyx", "Hyy", "Hyz"],
        ["Hzx", "Hzy", "Hzz"],
    ]
    # Cria axes uma única vez; curvas são adicionadas dentro do loop.
    # v2.6 P1: sharey vincula todos os 18 subplots ao primeiro — zoom em
    # qualquer um propaga para todos. Y-axis = profundidade compartilhada.
    axes_re: List[Any] = []
    axes_im: List[Any] = []
    first_ax: Optional[Any] = None
    for row in range(n_comp_rows):
        for col_pair in range(pair_cols):
            comp_name = comp_order[row][col_pair]
            base_col = (1 if have_rho else 0) + 2 * col_pair
            kw = {"sharey": first_ax} if first_ax is not None else {}
            ax_re = canvas.figure.add_subplot(gs[row, base_col], **kw)
            ax_im = canvas.figure.add_subplot(gs[row, base_col + 1], **kw)
            if first_ax is None:
                first_ax = ax_re
            _draw_layer_boundaries(ax_re, thick_arr, style)
            _draw_layer_boundaries(ax_im, thick_arr, style)
            ax_re.set_title(f"Re$({{{comp_name}}})$")
            ax_im.set_title(f"Im$({{{comp_name}}})$")
            ax_re.invert_yaxis()
            ax_im.invert_yaxis()
            ax_re.grid(True, ls=":", alpha=style.grid_alpha)
            ax_im.grid(True, ls=":", alpha=style.grid_alpha)
            ax_re.xaxis.set_major_formatter(_axis_formatter(style.axis_precision))
            ax_im.xaxis.set_major_formatter(_axis_formatter(style.axis_precision))
            if row == n_comp_rows - 1:
                ax_re.set_xlabel("Amplitude")
                ax_im.set_xlabel("Amplitude")
            axes_re.append(ax_re)
            axes_im.append(ax_im)

    # Usa cor Re/Im padrão quando apenas 1 combo (backwards-compat visual),
    # senão usa paleta para distinguir curvas.
    single_combo = n_curves == 1
    curve_idx = 0
    label_first_only = True
    for itr in tr_sel:
        for iang in ang_sel:
            for ifq in freq_sel:
                if z_obs is None:
                    z = np.arange(n_pos, dtype=np.float64)
                elif z_obs.ndim == 2:
                    z = z_obs[iang] if z_obs.shape[0] > iang else z_obs[0]
                else:
                    z = np.asarray(z_obs, dtype=np.float64)
                lbl = (
                    f"TR={trs[itr]:.2f} θ={dips[iang]:g}° "
                    f"f={freqs[ifq] / 1e3:.1f} kHz"
                )
                color_re = c_real if single_combo else colors[curve_idx % len(colors)]
                color_im = c_imag if single_combo else colors[curve_idx % len(colors)]
                for ax_re, ax_im, (row_i, col_pair_i) in zip(
                    axes_re,
                    axes_im,
                    [(r, c) for r in range(n_comp_rows) for c in range(pair_cols)],
                ):
                    comp_name = comp_order[row_i][col_pair_i]
                    idx = COMPONENT_NAMES.index(comp_name)
                    H_sel = H[itr, iang, :, ifq, idx]
                    ax_re.plot(
                        H_sel.real,
                        z,
                        color=color_re,
                        lw=style.line_width,
                        label=lbl if label_first_only else None,
                    )
                    ax_im.plot(
                        H_sel.imag,
                        z,
                        color=color_im,
                        lw=style.line_width,
                        label=lbl if label_first_only else None,
                    )
                curve_idx += 1
                label_first_only = False  # legenda só na 1ª volta (itera axes)

    # Legenda compacta — apenas se houver múltiplas curvas e couber
    if n_curves > 1 and n_curves <= 8 and axes_re:
        axes_re[0].legend(loc="best", fontsize=max(6, style.font_size - 4))

    # Título global: inclui info de TR, dip, f
    try:
        ft = freqs[ifq] / 1e3 if ifq < len(freqs) else 0.0
        lbl = (
            f"{title}  |  TR={trs[itr]:.2f} m  |  "
            f"θ={dips[iang]:g}°  |  f={ft:.1f} kHz"
        )
    except Exception:
        lbl = title
    # v2.4c: scale_mode no tensor 3x6 é fixo em Re/Im (propósito: panorama);
    # se o usuário escolheu outra escala, sugere em parênteses usar a aba
    # "Componentes EM" para obter magnitude/fase/dB individual.
    if scale_mode and scale_mode != "re_im":
        lbl = lbl + f"   (dica: use 'Componentes EM' para visualizar em {scale_mode})"
    canvas.figure.suptitle(lbl, fontsize=style.font_size + 2)
    canvas.draw()


# ──────────────────────────────────────────────────────────────────────────
# Plot — EM profile (mag/phase, Re/Im, dB)
# ──────────────────────────────────────────────────────────────────────────


def plot_em_profile(
    canvas: EMCanvas,
    H_tensor: np.ndarray,
    z_obs: Optional[np.ndarray],
    freqs: Sequence[float],
    trs: Sequence[float],
    dips: Sequence[float],
    components: Sequence[str] = ("Hxx", "Hzz"),
    title: str = "Campos EM",
    kind: str = "Magnitude + Fase",
    style: Optional[PlotStyle] = None,
    thicknesses: Optional[Sequence[float]] = None,
    tr_mask: Optional[Sequence[bool]] = None,
    ang_mask: Optional[Sequence[bool]] = None,
    freq_mask: Optional[Sequence[bool]] = None,
    combos: Optional[Sequence[Tuple[int, int, int]]] = None,
    scale_mode: str = "re_im",
    layout: str = "default",
) -> None:
    """Plota componentes EM em função da profundidade em múltiplos modos.

    Args:
        kind: Um dos valores em :data:`PLOT_KINDS`. Determina o layout
            (2-col: Magnitude + Fase, ou Re + Im, ou 1-col: só Magnitude / Fase / Re / Im / dB).
        layout: v2.5 — preset de layout escolhido no PlotComposerDialog.
            Quando ``"em_vertical_2col"`` força ``kind="Magnitude + Fase"``;
            ``"em_vertical_1col"`` força ``kind="Magnitude (linear)"``.
            ``"default"`` respeita o ``kind`` explícito.
    """
    if canvas.figure is None:
        return
    # v2.5: layout sobrescreve `kind` para garantir N×2 (Mag+Fase) ou N×1 (Mag).
    if layout == "em_vertical_2col":
        kind = "Magnitude + Fase"
    elif layout == "em_vertical_1col":
        kind = "Magnitude (linear)"
    style = style or canvas.style
    apply_style(style)
    canvas.figure.clear()

    H = np.asarray(H_tensor)
    if H.ndim != 5:
        ax = canvas.figure.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            f"H_tensor shape inesperado: {H.shape}",
            ha="center",
            transform=ax.transAxes,
        )
        canvas.draw()
        return

    nTR, nAng, n_pos, nf, _ = H.shape
    # v2.4c: resolve combinações (lista de tuplas tem prioridade; senão masks).
    if combos is not None:
        combo_iter_em: List[Tuple[int, int, int]] = [
            (int(itr), int(iang), int(ifq))
            for (itr, iang, ifq) in combos
            if 0 <= int(itr) < nTR and 0 <= int(iang) < nAng and 0 <= int(ifq) < nf
        ]
    else:
        _tr_sel = [
            i for i in range(nTR) if tr_mask is None or (i < len(tr_mask) and tr_mask[i])
        ]
        _ang_sel = [
            i
            for i in range(nAng)
            if ang_mask is None or (i < len(ang_mask) and ang_mask[i])
        ]
        _freq_sel = [
            i
            for i in range(nf)
            if freq_mask is None or (i < len(freq_mask) and freq_mask[i])
        ]
        combo_iter_em = [
            (itr, iang, ifq) for itr in _tr_sel for iang in _ang_sel for ifq in _freq_sel
        ]
    # v2.4d: scale_mode só sobrescreve `kind` para modos 1-col. Para modos
    # 2-col ("Magnitude + Fase" e "Re + Im") o `kind` vem do combo_kind_mode
    # e determina o LAYOUT; scale_mode apenas modula COMO cada coluna é
    # calculada (log10/dB/rad) — preserva o comportamento v2.4/v2.4b no qual
    # "Magnitude + Fase" plota ambos lado-a-lado na mesma imagem.
    if kind not in ("Magnitude + Fase", "Re + Im"):
        _scale_to_kind = {
            "re_im": "Re + Im",
            "mag_lin": "Só Magnitude",
            "mag_log10": "Só Magnitude",
            "mag_db": "Magnitude (dB)",
            "phase_deg": "Só Fase",
            "phase_rad": "Só Fase",
        }
        if scale_mode in _scale_to_kind:
            kind = _scale_to_kind[scale_mode]
    n_comp = max(1, len(components))
    two_cols = kind in ("Magnitude + Fase", "Re + Im")
    ncols = 2 if two_cols else 1
    axes = canvas.figure.subplots(n_comp, ncols, squeeze=False)

    comp_idx = {name: i for i, name in enumerate(COMPONENT_NAMES)}
    n_curves = max(1, len(combo_iter_em))
    colors = _palette_colors(n_curves, style.palette)
    thick_arr = (
        np.asarray(thicknesses, dtype=np.float64)
        if thicknesses is not None
        else np.array([], dtype=np.float64)
    )

    for row, cname in enumerate(components):
        idx = comp_idx.get(cname, 0)
        ax_a = axes[row][0]
        ax_b = axes[row][1] if two_cols else None
        if thick_arr.size:
            _draw_layer_boundaries(ax_a, thick_arr, style)
            if ax_b is not None:
                _draw_layer_boundaries(ax_b, thick_arr, style)
        curve = 0
        # v2.4c: itera apenas sobre combinações selecionadas (combos unificado)
        for itr, iang, ifq in combo_iter_em:
            label = (
                f"TR={trs[itr]:.2f} m · θ={dips[iang]:g}° · "
                f"f={freqs[ifq] / 1e3:.1f} kHz"
            )
            if z_obs is not None and np.asarray(z_obs).ndim == 2:
                z = np.asarray(z_obs)[iang]
            elif z_obs is not None and np.asarray(z_obs).ndim == 1:
                z = np.asarray(z_obs)
            else:
                z = np.arange(n_pos, dtype=np.float64)
            H_sel = H[itr, iang, :, ifq, idx]
            mag = np.abs(H_sel)
            phase = np.degrees(np.angle(H_sel))
            color = colors[curve % len(colors)]
            curve += 1
            if kind == "Magnitude + Fase":
                # v2.4d: scale_mode modula escala da magnitude (eixo A)
                # e da fase (eixo B) — mantém ambos visíveis lado a lado.
                if scale_mode == "mag_log10":
                    mag_plot = _safe_log10(mag)
                elif scale_mode == "mag_db":
                    mag_plot = 20.0 * _safe_log10(mag)
                else:
                    mag_plot = mag  # "mag_lin" ou default
                phase_plot = np.radians(phase) if scale_mode == "phase_rad" else phase
                ax_a.plot(mag_plot, z, label=label, lw=style.line_width, color=color)
                ax_b.plot(phase_plot, z, label=label, lw=style.line_width, color=color)
            elif kind == "Re + Im":
                ax_a.plot(H_sel.real, z, label=label, lw=style.line_width, color=color)
                ax_b.plot(H_sel.imag, z, label=label, lw=style.line_width, color=color)
            elif kind == "Magnitude (dB)":
                db = 20.0 * _safe_log10(mag)
                ax_a.plot(db, z, label=label, lw=style.line_width, color=color)
            elif kind == "Só Re":
                ax_a.plot(H_sel.real, z, label=label, lw=style.line_width, color=color)
            elif kind == "Só Im":
                ax_a.plot(H_sel.imag, z, label=label, lw=style.line_width, color=color)
            elif kind == "Só Magnitude":
                # v2.4c: scale_mode="mag_log10" troca magnitude por log10(|H|)
                mag_plot = _safe_log10(mag) if scale_mode == "mag_log10" else mag
                ax_a.plot(mag_plot, z, label=label, lw=style.line_width, color=color)
            elif kind == "Só Fase":
                ax_a.plot(phase, z, label=label, lw=style.line_width, color=color)

        # Labels e formatação
        if kind == "Magnitude + Fase":
            # v2.4d: reflete scale_mode nos xlabels e no xscale
            if scale_mode == "mag_log10":
                ax_a.set_xlabel(f"log10 |{cname}|")
            elif scale_mode == "mag_db":
                ax_a.set_xlabel(f"|{cname}| (dB)")
            elif scale_mode == "mag_lin":
                ax_a.set_xlabel(f"|{cname}| (A/m)")
            else:
                ax_a.set_xlabel(f"|{cname}| (A/m)")
                ax_a.set_xscale("log")
            ax_b.set_xlabel(
                f"Fase {cname} (rad)"
                if scale_mode == "phase_rad"
                else f"Fase {cname} (°)"
            )
        elif kind == "Re + Im":
            ax_a.set_xlabel(f"Re({cname})")
            ax_b.set_xlabel(f"Im({cname})")
        elif kind == "Magnitude (dB)":
            ax_a.set_xlabel(f"|{cname}| (dB)")
        elif kind == "Só Re":
            ax_a.set_xlabel(f"Re({cname})")
        elif kind == "Só Im":
            ax_a.set_xlabel(f"Im({cname})")
        elif kind == "Só Magnitude":
            if scale_mode == "mag_log10":
                ax_a.set_xlabel(f"log10 |{cname}|")
            else:
                ax_a.set_xlabel(f"|{cname}| (A/m)")
                ax_a.set_xscale("log")
        elif kind == "Só Fase":
            ax_a.set_xlabel(f"Fase {cname} (°)")
        for ax in (ax_a, ax_b):
            if ax is None:
                continue
            ax.set_ylabel("Profundidade (m)")
            ax.invert_yaxis()
            ax.grid(True, ls=":", alpha=style.grid_alpha)
            ax.xaxis.set_major_formatter(_axis_formatter(style.axis_precision))
        if row == 0 and n_curves <= 12:
            ax_a.legend(loc="best", fontsize=max(6, style.font_size - 3))
    canvas.figure.suptitle(title, fontsize=style.font_size + 2)
    canvas.draw()


# ──────────────────────────────────────────────────────────────────────────
# Plot — benchmark compare (Numba vs Fortran)
# ──────────────────────────────────────────────────────────────────────────


def plot_benchmark_compare(
    canvas: EMCanvas,
    H_numba: Optional[np.ndarray],
    H_fortran: Optional[np.ndarray],
    z_obs: Optional[np.ndarray],
    freqs: Sequence[float],
    trs: Sequence[float],
    dips: Sequence[float],
    components: Sequence[str] = ("Hxx", "Hzz"),
    title: str = "Benchmark — Numba vs Fortran",
    style: Optional[PlotStyle] = None,
    thicknesses: Optional[Sequence[float]] = None,
) -> None:
    """Plota sobreposição Numba (linha) vs Fortran (pontos) para comparação."""
    if canvas.figure is None:
        return
    style = style or canvas.style
    apply_style(style)
    canvas.figure.clear()
    H_list: List[Tuple[np.ndarray, str, str, float, Optional[str], str]] = []
    if H_numba is not None:
        H_list.append((H_numba, "Numba", "-", style.line_width, None, style.color_numba))
    if H_fortran is not None:
        H_list.append((H_fortran, "Fortran", "", 0.0, "o", style.color_fortran))
    if not H_list:
        ax = canvas.figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "Sem dados para comparar.", ha="center", transform=ax.transAxes)
        canvas.draw()
        return

    ref = H_list[0][0]
    if ref.ndim != 5:
        ax = canvas.figure.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            f"Shape H inesperado: {ref.shape}",
            ha="center",
            transform=ax.transAxes,
        )
        canvas.draw()
        return
    nTR, nAng, n_pos, nf, _ = ref.shape
    comp_idx = {name: i for i, name in enumerate(COMPONENT_NAMES)}
    axes = canvas.figure.subplots(len(components), 1, squeeze=False)
    thick_arr = (
        np.asarray(thicknesses, dtype=np.float64)
        if thicknesses is not None
        else np.array([], dtype=np.float64)
    )

    for row, cname in enumerate(components):
        idx = comp_idx.get(cname, 0)
        ax = axes[row][0]
        if thick_arr.size:
            _draw_layer_boundaries(ax, thick_arr, style)
        for H, lbl, ls, lw, mk, color in H_list:
            for itr in range(min(nTR, H.shape[0])):
                for iang in range(min(nAng, H.shape[1])):
                    for ifq in range(min(nf, H.shape[3])):
                        if z_obs is not None and np.asarray(z_obs).ndim == 2:
                            z = np.asarray(z_obs)[iang]
                        elif z_obs is not None and np.asarray(z_obs).ndim == 1:
                            z = np.asarray(z_obs)
                        else:
                            z = np.arange(H.shape[2], dtype=np.float64)
                        mag = np.abs(H[itr, iang, :, ifq, idx])
                        suffix = (
                            f" TR={trs[itr]:.2f} θ={dips[iang]:g}° "
                            f"f={freqs[ifq] / 1e3:.1f}k"
                        )
                        if mk:
                            ax.plot(
                                mag,
                                z,
                                marker=mk,
                                ms=3,
                                ls="none",
                                color=color,
                                label=f"{lbl}{suffix}",
                            )
                        else:
                            ax.plot(
                                mag,
                                z,
                                ls=ls,
                                lw=lw,
                                color=color,
                                label=f"{lbl}{suffix}",
                            )
        ax.set_xlabel(f"|{cname}| (A/m)")
        ax.set_ylabel("Profundidade (m)")
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.grid(True, ls=":", alpha=style.grid_alpha)
        ax.xaxis.set_major_formatter(_axis_formatter(style.axis_precision))
        if row == 0:
            ax.legend(loc="best", fontsize=max(6, style.font_size - 3), ncol=2)
    canvas.figure.suptitle(title, fontsize=style.font_size + 2)
    canvas.draw()


# ──────────────────────────────────────────────────────────────────────────
# Plot — geosinais USD/UAD/UHR/UHA
# ──────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────
# Geosinais v2.4 — 5 sinais complexos (amplitude + fase) com filtros
# ──────────────────────────────────────────────────────────────────────────

# Ordem canônica dos geosinais (usada pela máscara na ResultsPage).
GEOSIGNAL_NAMES = ("USD", "UAD", "UHR", "UHA", "U3DF")


def _compute_geosignal(name: str, H_slice: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    """Computa um dos 5 geosinais canônicos como quantidade COMPLEXA.

    Cada sinal é retornado como array complex128; chamador extrai
    ``np.abs`` (com log10) e ``np.angle`` (em graus) para amplitude/fase.

    Definições (convenção de indexação: Hxx=0, Hxy=1, Hxz=2, Hyx=3,
    Hyy=4, Hyz=5, Hzx=6, Hzy=7, Hzz=8):

    | Sinal | Expressão complexa              | Física                |
    |:-----:|:--------------------------------|:----------------------|
    | USD   | Hxx / Hyy                       | Razão planar (TIV)    |
    | UAD   | Hxx − Hyy                       | Diferença planar      |
    | UHR   | Hxz / Hzz                       | Razão axial/planar    |
    | UHA   | Hxz − Hzz                       | Diferença axial       |
    | U3DF  | (Hxy − Hyx) / (Hxy + Hyx + ε)   | Fator 3D assimétrico  |

    Compatibilidade v2.3: ``USD_amp`` = ``log10|USD|`` ≡ antigo USD;
    ``USD_phase`` = ``∠(USD)`` em graus ≡ antigo UAD. Idem UHR/UHA.
    """
    Hxx = H_slice[..., 0]
    Hxy = H_slice[..., 1]
    Hxz = H_slice[..., 2]
    Hyx = H_slice[..., 3]
    Hyy = H_slice[..., 4]
    Hzz = H_slice[..., 8]
    if name == "USD":
        return Hxx / np.where(np.abs(Hyy) < eps, eps, Hyy)
    if name == "UAD":
        return Hxx - Hyy
    if name == "UHR":
        return Hxz / np.where(np.abs(Hzz) < eps, eps, Hzz)
    if name == "UHA":
        return Hxz - Hzz
    if name == "U3DF":
        num = Hxy - Hyx
        den = Hxy + Hyx + eps
        return num / den
    raise ValueError(f"Geosinal desconhecido: {name!r}")


def plot_geosignals(
    canvas: EMCanvas,
    H_tensor: np.ndarray,
    z_obs: Optional[np.ndarray],
    freqs: Sequence[float],
    trs: Sequence[float],
    dips: Sequence[float],
    title: str = "Geosinais — Amplitude + Fase",
    style: Optional[PlotStyle] = None,
    thicknesses: Optional[Sequence[float]] = None,
    geosignal_mask: Optional[Sequence[bool]] = None,
    tr_mask: Optional[Sequence[bool]] = None,
    ang_mask: Optional[Sequence[bool]] = None,
    freq_mask: Optional[Sequence[bool]] = None,
    combos: Optional[Sequence[Tuple[int, int, int]]] = None,
    scale_mode: str = "geo_log10_deg",
    layout: str = "default",
    include_resistivity: bool = False,
    rho_h: Optional[Sequence[float]] = None,
    rho_v: Optional[Sequence[float]] = None,
) -> None:
    """Plota até 5 geosinais derivados do tensor H em grade (N×2) amp/fase.

    Args:
        canvas: ``EMCanvas`` onde desenhar.
        H_tensor: Tensor (nTR, nAng, n_pos, nf, 9) complexo.
        z_obs: Profundidades (m). ``None`` usa índices 0..n_pos-1.
        freqs, trs, dips: Listas de configurações (para rótulos/legenda).
        title: Título da figura.
        style: ``PlotStyle`` para fontes/cores.
        thicknesses: Espessuras de camadas para desenhar interfaces.
        geosignal_mask: Máscara booleana de 5 elementos ordem
            [USD, UAD, UHR, UHA, U3DF]. Default ``None`` = todos marcados.
        tr_mask / ang_mask / freq_mask: Máscaras booleanas para filtrar
            quais combinações plotar. ``None`` = todas.

    Layout: até 5 linhas × 2 colunas (amplitude esquerda, fase direita).
    Sinais desmarcados são omitidos (grid shrink para N selecionados).
    """
    if canvas.figure is None:
        return
    style = style or canvas.style
    apply_style(style)
    canvas.figure.clear()
    H = np.asarray(H_tensor)
    if H.ndim != 5:
        ax = canvas.figure.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            f"Shape H inesperado: {H.shape}",
            ha="center",
            transform=ax.transAxes,
        )
        canvas.draw()
        return
    nTR, nAng, n_pos, nf, _ = H.shape
    thick_arr = (
        np.asarray(thicknesses, dtype=np.float64)
        if thicknesses is not None
        else np.array([], dtype=np.float64)
    )

    # Resolve máscaras (default: todos True)
    geo_mask = list(
        geosignal_mask if geosignal_mask is not None else [True] * len(GEOSIGNAL_NAMES)
    )
    geo_mask = geo_mask + [True] * (len(GEOSIGNAL_NAMES) - len(geo_mask))
    selected_geo = [g for g, on in zip(GEOSIGNAL_NAMES, geo_mask) if on]
    if not selected_geo:
        ax = canvas.figure.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            "Nenhum geosinal selecionado.\nMarque ao menos um em 'Filtro — Geosinais'.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=style.font_size + 1,
        )
        canvas.draw()
        return

    # v2.4c: combos (lista de tuplas) tem prioridade; fallback p/ masks legadas.
    if combos is not None:
        _combo_list_v24c: List[Tuple[int, int, int]] = [
            (int(itr), int(iang), int(ifq))
            for (itr, iang, ifq) in combos
            if 0 <= int(itr) < nTR and 0 <= int(iang) < nAng and 0 <= int(ifq) < nf
        ]
        _used_trs = {c[0] for c in _combo_list_v24c}
        _used_angs = {c[1] for c in _combo_list_v24c}
        _used_freqs = {c[2] for c in _combo_list_v24c}
        tr_mask = [i in _used_trs for i in range(nTR)]
        ang_mask = [i in _used_angs for i in range(nAng)]
        freq_mask = [i in _used_freqs for i in range(nf)]
    tr_sel = [
        i for i in range(nTR) if tr_mask is None or (i < len(tr_mask) and tr_mask[i])
    ]
    ang_sel = [
        i for i in range(nAng) if ang_mask is None or (i < len(ang_mask) and ang_mask[i])
    ]
    freq_sel = [
        i for i in range(nf) if freq_mask is None or (i < len(freq_mask) and freq_mask[i])
    ]
    if not tr_sel or not ang_sel or not freq_sel:
        ax = canvas.figure.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            "Selecione ao menos 1 combinação TR/Ângulo/Frequência.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=style.font_size + 1,
        )
        canvas.draw()
        return

    n_rows = len(selected_geo)
    # v2.5: layout `geo_nx2_rho` ou flag `include_resistivity=True` adiciona
    # coluna ρ_h/ρ_v à esquerda (similar ao layout 3×7 do tensor). Reutiliza
    # `_rho_per_z()` para amostragem em z_obs (bit-exato a buildValidamodels.py).
    rho_arr_h = (
        np.asarray(rho_h, dtype=np.float64)
        if rho_h is not None and len(rho_h) > 0
        else None
    )
    rho_arr_v = (
        np.asarray(rho_v, dtype=np.float64)
        if rho_v is not None and len(rho_v) > 0
        else (rho_arr_h.copy() if rho_arr_h is not None else None)
    )
    use_rho_column = (
        layout == "geo_nx2_rho" or include_resistivity
    ) and rho_arr_h is not None
    if use_rho_column:
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(
            n_rows,
            3,
            figure=canvas.figure,
            width_ratios=[1.4, 1.0, 1.0],
            wspace=0.32,
            hspace=0.38,
        )
        ax_rho = canvas.figure.add_subplot(gs[:, 0])
        axes = np.empty((n_rows, 2), dtype=object)
        # v2.6 P1: sharey vincula geosinais N×2 ao primeiro — zoom Y propaga.
        first_geo_ax: Optional[Any] = None
        for r in range(n_rows):
            kw = {"sharey": first_geo_ax} if first_geo_ax is not None else {}
            axes[r][0] = canvas.figure.add_subplot(gs[r, 1], **kw)
            axes[r][1] = canvas.figure.add_subplot(gs[r, 2], **kw)
            if first_geo_ax is None:
                first_geo_ax = axes[r][0]
        # Desenha o perfil ρ (log10 com axhlines nas interfaces).
        # Reaproveita o mesmo helper canônico v2.4b usado por plot_tensor_full.
        z_flat_rho = (
            np.asarray(z_obs, dtype=np.float64).ravel() if z_obs is not None else None
        )
        if z_flat_rho is not None and z_flat_rho.size > 0:
            order = np.argsort(z_flat_rho)
            z_sorted = z_flat_rho[order]
            rho_h_z = _rho_per_z(z_flat_rho, rho_arr_h, thick_arr)[order]
            rho_v_z = _rho_per_z(z_flat_rho, rho_arr_v, thick_arr)[order]
            ax_rho.semilogx(
                rho_h_z,
                z_sorted,
                color=style.color_rho_h,
                linewidth=style.line_width + 0.6,
                label=r"$\rho_h$",
            )
            ax_rho.semilogx(
                rho_v_z,
                z_sorted,
                color=style.color_rho_v,
                linewidth=style.line_width + 0.6,
                linestyle="--",
                label=r"$\rho_v$",
            )
            # Interfaces internas — normaliza thick_arr da mesma forma que
            # plot_tensor_full (L810-817 do mesmo arquivo). Sem normalização,
            # se `thicknesses` incluir valores sentinela das semi-camadas
            # (size == n ou n−1 em vez de n−2), as axhlines aparecem em
            # profundidades erradas. Garante paridade com a coluna ρ do
            # plot_tensor_full e com a referência buildValidamodels.py.
            if thick_arr.size and rho_arr_h.size >= 2:
                if thick_arr.size == rho_arr_h.size - 2:
                    internals = thick_arr
                elif thick_arr.size == rho_arr_h.size:
                    internals = thick_arr[1:-1]
                elif thick_arr.size == rho_arr_h.size - 1:
                    internals = thick_arr[1:]
                else:
                    internals = thick_arr[: max(rho_arr_h.size - 2, 0)]
                boundaries = np.concatenate(([0.0], np.cumsum(internals)))
                for interf in boundaries:
                    ax_rho.axhline(
                        y=float(interf),
                        color="#000000",
                        linestyle="--",
                        lw=1.0,
                        alpha=0.6,
                    )
        ax_rho.invert_yaxis()
        ax_rho.set_xlabel(r"Resistividade ($\Omega \cdot$m)")
        ax_rho.set_ylabel("Profundidade TVD (m)")
        ax_rho.set_title(r"$\rho_h$ e $\rho_v$ (modelo verdadeiro)")
        ax_rho.grid(True, ls=":", alpha=style.grid_alpha)
        ax_rho.legend(loc="best", fontsize=max(6, style.font_size - 2))
    else:
        axes = canvas.figure.subplots(n_rows, 2, squeeze=False)
    n_curves = len(tr_sel) * len(ang_sel) * len(freq_sel)
    colors = _palette_colors(n_curves, style.palette)

    for r, gname in enumerate(selected_geo):
        ax_amp = axes[r][0]
        ax_pha = axes[r][1]
        curve = 0
        for itr in tr_sel:
            for iang in ang_sel:
                for ifq in freq_sel:
                    if z_obs is not None and np.asarray(z_obs).ndim == 2:
                        z = np.asarray(z_obs)[iang]
                    elif z_obs is not None and np.asarray(z_obs).ndim == 1:
                        z = np.asarray(z_obs)
                    else:
                        z = np.arange(n_pos, dtype=np.float64)
                    H_slice = H[itr, iang, :, ifq, :]
                    signal = _compute_geosignal(gname, H_slice)
                    # v2.4c: amplitude/fase dependem de scale_mode
                    raw_amp = np.abs(signal) + 1e-30
                    if scale_mode == "geo_lin_deg":
                        amp = raw_amp
                        phase = np.degrees(np.angle(signal))
                    elif scale_mode == "geo_db_rad":
                        amp = 20.0 * _safe_log10(raw_amp)
                        phase = np.angle(signal)  # radianos
                    else:  # geo_log10_deg (default)
                        amp = _safe_log10(raw_amp)
                        phase = np.degrees(np.angle(signal))
                    lbl = (
                        f"TR={trs[itr]:.2f} θ={dips[iang]:g}° "
                        f"f={freqs[ifq] / 1e3:.1f} kHz"
                    )
                    color = colors[curve % len(colors)]
                    curve += 1
                    ax_amp.plot(amp, z, label=lbl, lw=style.line_width, color=color)
                    ax_pha.plot(phase, z, label=lbl, lw=style.line_width, color=color)

        # v2.4c: rótulos refletem a escala selecionada
        if scale_mode == "geo_lin_deg":
            ax_amp.set_xlabel(f"{gname} — amplitude |·|")
            ax_pha.set_xlabel(f"{gname} — fase ∠· (°)")
        elif scale_mode == "geo_db_rad":
            ax_amp.set_xlabel(f"{gname} — amplitude (dB)")
            ax_pha.set_xlabel(f"{gname} — fase ∠· (rad)")
        else:
            ax_amp.set_xlabel(f"{gname} — amplitude log₁₀|·|")
            ax_pha.set_xlabel(f"{gname} — fase ∠· (°)")
        for ax in (ax_amp, ax_pha):
            ax.set_ylabel("Profundidade (m)")
            ax.invert_yaxis()
            ax.grid(True, ls=":", alpha=style.grid_alpha)
            ax.xaxis.set_major_formatter(_axis_formatter(style.axis_precision))
            if thick_arr.size:
                _draw_layer_boundaries(ax, thick_arr, style)

    if n_curves <= 12:
        axes[0][0].legend(loc="best", fontsize=max(6, style.font_size - 3))
    canvas.figure.suptitle(title, fontsize=style.font_size + 2)
    canvas.draw()


# ──────────────────────────────────────────────────────────────────────────
# Plot — anisotropia λ = √(ρᵥ/ρₕ)
# ──────────────────────────────────────────────────────────────────────────


def plot_anisotropy(
    canvas: EMCanvas,
    rho_h: Sequence[float],
    rho_v: Sequence[float],
    thicknesses: Sequence[float],
    title: str = "Fator de Anisotropia λ = √(ρᵥ/ρₕ)",
    style: Optional[PlotStyle] = None,
) -> None:
    """Plota o perfil de anisotropia λ em função da profundidade."""
    if canvas.figure is None:
        return
    style = style or canvas.style
    apply_style(style)
    canvas.figure.clear()
    ax = canvas.figure.add_subplot(1, 1, 1)

    rh = np.asarray(rho_h, dtype=np.float64)
    rv = np.asarray(rho_v, dtype=np.float64)
    thick_arr = (
        np.asarray(thicknesses, dtype=np.float64)
        if thicknesses is not None
        else np.array([], dtype=np.float64)
    )
    n = rh.size
    if rv.size != n:
        rv = np.resize(rv, n)
    lam = np.sqrt(np.clip(rv / np.clip(rh, 1e-12, None), 1.0, None))

    if n >= 3 and thick_arr.size == n - 2:
        bounds = [0.0]
        for t in thick_arr:
            bounds.append(bounds[-1] + float(t))
        depths = [bounds[0] - 5.0] + bounds + [bounds[-1] + 5.0]
    else:
        depths = np.linspace(0.0, 1.0, n + 1).tolist()
    y_edges = np.asarray(depths, dtype=np.float64)
    centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    _draw_layer_boundaries(ax, thick_arr, style)
    ax.step(lam, centers, where="mid", color=style.color_rho_v, lw=style.line_width + 0.3)
    ax.set_xlabel(r"$\lambda = \sqrt{\rho_v/\rho_h}$")
    ax.set_ylabel("Profundidade (m)")
    ax.invert_yaxis()
    ax.grid(True, ls=":", alpha=style.grid_alpha)
    ax.set_title(title)
    canvas.draw()


__all__ = [
    "COMPONENT_NAMES",
    "EMCanvas",
    "PLOT_KINDS",
    "PlotStyle",
    "apply_style",
    "plot_anisotropy",
    "plot_benchmark_compare",
    "plot_em_profile",
    "plot_geosignals",
    "plot_resistivity_profile",
    "plot_tensor_full",
]
