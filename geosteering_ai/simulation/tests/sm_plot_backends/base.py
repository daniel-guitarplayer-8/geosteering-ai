# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_plot_backends/base.py                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Abstração Multi-Backend de Plotagem   ║
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

        # QWebEngineView opcional — se não estiver, Plotly só exporta HTML
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
