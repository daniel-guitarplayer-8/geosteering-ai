# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/plot_backends/base.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — Abstração Multi-Backend de Plotagem                  ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Status      : Produção (v2.6)                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Define a interface abstrata `PlotCanvas` que cada backend de           ║
# ║    plotagem (Matplotlib, PyQtGraph, Plotly, Vispy) implementa. Permite    ║
# ║    que o código de plotagem (sm_plots.py) chame operações de alto nível   ║
# ║    sem conhecer o backend ativo, viabilizando troca em runtime.           ║
# ║                                                                           ║
# ║  PADRÃO ARQUITETURAL                                                      ║
# ║    Strategy + Factory híbrido:                                            ║
# ║      • Strategy: ABC `PlotCanvas` com métodos comuns                      ║
# ║      • Factory: `make_canvas(backend, ...)` despacha para o concrete      ║
# ║      • Lazy import: cada backend importa suas deps somente quando o       ║
# ║        usuário escolhe aquele backend (matplotlib é exceção — sempre OK)  ║
# ║                                                                           ║
# ║  INVARIANTES                                                              ║
# ║    • Não toca em backends de SIMULAÇÃO (JAX/Numba/Fortran).               ║
# ║    • Cada subclasse `PlotCanvas` é um QWidget Qt-compatible.              ║
# ║    • `add_subplot_grid(rows, cols, sharey=True)` é a API canônica de     ║
# ║      criação de subplots — substitui `figure.add_subplot()` direto.      ║
# ║    • `SubplotHandle` é Protocol opaco — cada backend escolhe seu tipo.   ║
# ║    • Backward-compat: `sm_plots.py` continua aceitando `EMCanvas` legado ║
# ║      via branch isinstance(canvas, PlotCanvas) — preserva 1464+ testes.  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Interface abstrata multi-backend para plotagem do Simulation Manager."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Protocol, Sequence, Tuple

import numpy as np

__all__ = [
    "PlotBackend",
    "AxisConfig",
    "SubplotHandle",
    "PlotCanvas",
    "make_canvas",
    "available_backends",
]


class PlotBackend(str, Enum):
    """Backends de plotagem suportados pelo Simulation Manager v2.6.

    Note:
        - MATPLOTLIB: padrão, sempre disponível, qualidade publicação.
        - PYQTGRAPH: Qt-native, GPU opcional, hover/crosshair triviais.
        - PLOTLY: HTML via QWebEngineView, hover rich nativo.
        - VISPY: GL puro, experimental, sem axes nativos.
    """

    MATPLOTLIB = "matplotlib"
    PYQTGRAPH = "pyqtgraph"
    PLOTLY = "plotly"
    VISPY = "vispy"


@dataclass(frozen=True)
class AxisConfig:
    """Configuração declarativa de um eixo (subplot).

    Attributes:
        title: Título do subplot (acima do plot).
        xlabel: Label do eixo X (horizontal).
        ylabel: Label do eixo Y (vertical, geralmente "Profundidade (m)").
        invert_y: Inverte eixo Y (depth cresce para baixo). Default True.
        grid: Mostra grid. Default True.
        log_x: Eixo X em escala log10. Default False.
        log_y: Eixo Y em escala log10. Default False.

    Example:
        >>> cfg = AxisConfig(title="Re(Hxx)", xlabel="A/m",
        ...                  ylabel="z (m)", invert_y=True)
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = "Profundidade (m)"
    invert_y: bool = True
    grid: bool = True
    log_x: bool = False
    log_y: bool = False


class SubplotHandle(Protocol):
    """Handle opaco para um subplot — cada backend retorna seu tipo nativo.

    Para matplotlib é um ``matplotlib.axes.Axes``; para PyQtGraph é um
    ``pyqtgraph.PlotItem``; para Plotly é uma tupla ``(row, col)``;
    para Vispy é um ``vispy.scene.ViewBox``.
    """

    ...


def _build_step_polyline(
    x: np.ndarray, y: np.ndarray, where: str = "post"
) -> Tuple[np.ndarray, np.ndarray]:
    """Constrói a polilinha em degraus de ``(x, y)`` (quantidade x vs profundidade y).

    Os perfis ρ/λ (Fatia 6d) têm a QUANTIDADE no eixo X piecewise-constant ao longo
    da PROFUNDIDADE no eixo Y (monotônica) — saltos nas interfaces de camada. Como Y
    é a variável independente (não X), ``matplotlib.step``/``stepMode`` (que degrau-eiam
    Y como função de X) têm orientação ERRADA aqui; por isso montamos a polilinha
    explicitamente e a desenhamos como linha comum (backend-agnóstico).

    Args:
        x: valores da quantidade (n,) — ρ, λ, … (eixo horizontal).
        y: profundidades (n,) monotônicas (eixo vertical).
        where: ``"post"`` (default) mantém ``x[i]`` de ``y[i]`` até ``y[i+1]`` (salto
            DEPOIS do ponto); ``"pre"`` mantém ``x[i]`` de ``y[i-1]`` até ``y[i]``.

    Returns:
        ``(xs, ys)`` — arrays ``(2n−1,)`` da polilinha em degraus (ou ``(≤1,)`` se ``n≤1``).
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    n = int(min(xv.size, yv.size))
    if n <= 1:
        return xv[:n], yv[:n]
    xv, yv = xv[:n], yv[:n]
    xs = np.empty(2 * n - 1, dtype=float)
    ys = np.empty(2 * n - 1, dtype=float)
    xs[0::2] = xv
    ys[0::2] = yv
    if where == "pre":
        # Salto ANTES do ponto: já assume x[i+1] em y[i].
        xs[1::2] = xv[1:]
        ys[1::2] = yv[:-1]
    else:  # "post": mantém o valor atual até a próxima profundidade.
        xs[1::2] = xv[:-1]
        ys[1::2] = yv[1:]
    return xs, ys


class PlotCanvas(ABC):
    """Interface comum aos 4 backends de plotagem.

    Princípio: a função ``plot_tensor_full()`` em ``sm_plots.py`` não
    importa qual backend está rodando — ela chama ``add_subplot_grid``,
    ``plot_line``, ``add_hline``, ``set_axis_config``, ``draw``.
    Cada concrete subclass implementa essas operações em seu modelo nativo.

    Note:
        Todas as subclasses DEVEM ser QWidget-compatible (retornar QWidget
        em ``widget()``) para integração com o layout Qt da ResultsPage.
    """

    @abstractmethod
    def widget(self) -> Any:
        """Retorna o QWidget nativo a ser inserido no layout Qt."""

    @abstractmethod
    def clear(self) -> None:
        """Remove todos os subplots e elementos do canvas."""

    @abstractmethod
    def draw(self) -> None:
        """Renderiza o canvas (após adicionar/modificar plots)."""

    @abstractmethod
    def save(self, path: str, dpi: int = 150) -> None:
        """Salva o canvas em PNG/PDF/SVG (formato detectado pela extensão)."""

    @abstractmethod
    def add_subplot_grid(
        self,
        rows: int,
        cols: int,
        sharey: bool = True,
        width_ratios: Optional[Sequence[float]] = None,
        height_ratios: Optional[Sequence[float]] = None,
    ) -> List[List[SubplotHandle]]:
        """Cria grid de subplots; retorna matriz [r][c] de handles.

        Args:
            rows: Número de linhas.
            cols: Número de colunas.
            sharey: Compartilhar eixo Y entre todos os subplots (depth comum).
            width_ratios: Razões opcionais de largura por coluna.
            height_ratios: Razões opcionais de altura por linha.

        Returns:
            Matriz 2D de SubplotHandle (cada handle é opaco; backend-específico).
        """

    @abstractmethod
    def plot_line(
        self,
        ax: SubplotHandle,
        x: np.ndarray,
        y: np.ndarray,
        *,
        label: str = "",
        color: Optional[str] = None,
        linewidth: float = 1.5,
        linestyle: str = "-",
    ) -> None:
        """Adiciona uma curva a um subplot."""

    @abstractmethod
    def add_hline(
        self,
        ax: SubplotHandle,
        y: float,
        *,
        color: str = "#7f7f7f",
        linestyle: str = "--",
        alpha: float = 0.4,
        linewidth: float = 0.8,
    ) -> None:
        """Adiciona linha horizontal (e.g., interface de camadas)."""

    @abstractmethod
    def set_axis_config(self, ax: SubplotHandle, cfg: AxisConfig) -> None:
        """Aplica configuração declarativa a um eixo."""

    @abstractmethod
    def set_dark_mode(self, dark: bool) -> None:
        """Aplica tema dark/light ao canvas inteiro.

        Para matplotlib: ajusta facecolor, edgecolor, tick colors.
        Para PyQtGraph: ``setConfigOption('background', ...)``.
        Para Plotly: usa template ``plotly_dark`` ou ``plotly_white``.
        Para Vispy: ajusta bgcolor do SceneCanvas.
        """

    # ── Métodos opcionais (default no-op) ─────────────────────────────────

    def plot_step(
        self,
        ax: SubplotHandle,
        x: np.ndarray,
        y: np.ndarray,
        *,
        where: str = "post",
        label: str = "",
        color: Optional[str] = None,
        linewidth: float = 1.5,
        linestyle: str = "-",
    ) -> None:
        """Plota uma curva STEP (perfis ρ/λ) — quantidade x constante por camada.

        Implementação BACKEND-AGNÓSTICA (concreta na ABC): monta a polilinha em
        degraus via :func:`_build_step_polyline` e delega a :meth:`plot_line` — assim
        os 4 backends ganham step correto sem código nativo (``ax.step``/``stepMode``
        degrau-eiam Y(X), orientação errada quando Y=profundidade é independente).

        Args:
            ax: handle do subplot.
            x: quantidade (ρ, λ, …) por posição (eixo horizontal).
            y: profundidades monotônicas (eixo vertical).
            where: ``"post"`` (default) | ``"pre"`` (ver :func:`_build_step_polyline`).
            label/color/linewidth/linestyle: repassados a :meth:`plot_line`.
        """
        xs, ys = _build_step_polyline(x, y, where)
        self.plot_line(
            ax,
            xs,
            ys,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )

    def plot_image(
        self,
        ax: SubplotHandle,
        data: np.ndarray,
        *,
        extent: Optional[Tuple[float, float, float, float]] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Any:
        """Plota uma imagem 2-D (heatmap de ensemble — n_models × n_pos).

        Default: ``NotImplementedError`` (plotly/vispy herdam erro claro). Backends
        reais (matplotlib ``imshow`` / PyQtGraph ``ImageItem``) sobrescrevem. A
        galeria captura o erro e cai para matplotlib (``_build_canvas``).

        Args:
            ax: handle do subplot.
            data: matriz 2-D ``(rows, cols)`` (row-major; ex.: modelo × posição).
            extent: ``(left, right, bottom, top)`` em coordenadas de dados (opcional).
            cmap: nome do colormap (estilo matplotlib).
            vmin/vmax: limites de cor (opcionais; auto se ``None``).

        Returns:
            Handle da imagem (mappable/ImageItem) para :meth:`set_colorbar`.

        Raises:
            NotImplementedError: no backend que não suporta imagens.
        """
        raise NotImplementedError(
            f"plot_image não suportado no backend {type(self).__name__} "
            "(use matplotlib ou pyqtgraph)."
        )

    def set_colorbar(self, ax: SubplotHandle, image: Any, *, label: str = "") -> None:
        """Adiciona uma colorbar à imagem de :meth:`plot_image`. Default: no-op."""
        return None

    def set_legend(self, ax: SubplotHandle, visible: bool = True) -> None:
        """Mostra/oculta legenda em um subplot. Default: no-op."""
        return None

    def reset_zoom(self) -> None:
        """Reseta zoom de todos os subplots ao range automático.

        Default: no-op. Implementação opcional por backend.
        """
        return None


def available_backends() -> List[PlotBackend]:
    """Retorna lista de backends disponíveis no ambiente atual.

    Tenta importar cada um lazy; backends sem deps instaladas não aparecem.

    Returns:
        Lista de PlotBackend instaláveis. Sempre inclui MATPLOTLIB.
    """
    available: List[PlotBackend] = [PlotBackend.MATPLOTLIB]
    try:
        import pyqtgraph  # noqa: F401

        available.append(PlotBackend.PYQTGRAPH)
    except ImportError:
        pass
    try:
        import plotly  # noqa: F401

        available.append(PlotBackend.PLOTLY)
    except ImportError:
        pass
    try:
        import vispy  # noqa: F401

        available.append(PlotBackend.VISPY)
    except ImportError:
        pass
    return available


def make_canvas(
    backend: PlotBackend,
    parent: Any = None,
    figsize: Tuple[float, float] = (14, 9),
    style: Any = None,
) -> PlotCanvas:
    """Factory de canvas — instancia o backend solicitado com import lazy.

    Args:
        backend: Backend a instanciar.
        parent: Widget pai Qt.
        figsize: Tamanho da figura em polegadas (largura, altura).
        style: Instância opcional de PlotStyle de sm_plots.py.

    Returns:
        Instância de PlotCanvas concreta.

    Raises:
        ImportError: Se o backend solicitado não tiver deps instaladas.
        ValueError: Se backend não reconhecido.
    """
    if backend == PlotBackend.MATPLOTLIB:
        from .mpl_canvas import MatplotlibCanvas

        return MatplotlibCanvas(parent=parent, figsize=figsize, style=style)
    if backend == PlotBackend.PYQTGRAPH:
        from .pyqtgraph_canvas import PyQtGraphCanvas

        return PyQtGraphCanvas(parent=parent, figsize=figsize, style=style)
    if backend == PlotBackend.PLOTLY:
        from .plotly_canvas import PlotlyCanvas

        return PlotlyCanvas(parent=parent, figsize=figsize, style=style)
    if backend == PlotBackend.VISPY:
        from .vispy_canvas import VispyCanvas

        return VispyCanvas(parent=parent, figsize=figsize, style=style)
    raise ValueError(f"Backend desconhecido: {backend}")
