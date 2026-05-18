# Plano de Pesquisa: Abstração Multi-Backend de Plotagem para Qt

**Contexto**: Aplicação PyQt6 com `sm_plots.py` (~1700 LOC, matplotlib + `FigureCanvasQTAgg`).
Adicionar 3 backends alternativos sem quebrar o caminho matplotlib.

**API atual** (`sm_plots.py:255-322` + 6 funções `plot_*(canvas, ...)`):
- `EMCanvas(QWidget)` expõe `figure`, `canvas`, `clear()`, `draw()`, `save(path)`, `set_style()`
- Funções consumidoras chamam `canvas.figure.add_subplot(...)` diretamente

---

## 1. Padrão de Abstração — Strategy + Factory híbrido

**Recomendação**: Combinar **Strategy** (interface `PlotCanvas` abstrata) + **Factory** (`make_canvas(backend)`)
+ camada utilitária **Adapter** para suavizar diferenças entre subplot-grid de cada backend.

Os 4 backends têm modelos mentais incompatíveis demais para um único Adapter:

| Backend | Subplot model | Refresh model | Coords |
|:--------|:--------------|:--------------|:-------|
| matplotlib | `figure.add_subplot(gs[r,c])` | `canvas.draw_idle()` | data-coords nativo |
| pyqtgraph | `glw.addPlot(row=r, col=c)` | imediato (paint event) | data-coords (Y invertido para depth) |
| plotly | `make_subplots(rows=3,cols=6)` → HTML | `setHtml(new_html)` | JSON layout |
| vispy | `SceneCanvas.central_widget.add_grid()` | `canvas.update()` | GL viewport, sem axes nativos |

A solução pragmática: a **interface comum cobre apenas o ciclo de vida** (clear/draw/save) e operações de **alto nível** (`add_subplot_grid`, `plot_line`, `add_hline`, `set_axis_config`). As funções `plot_*` chamam essas operações em vez de tocar matplotlib diretamente. Cada backend implementa o protocolo a seu modo.

### Skeleton — `sm_plot_backends/base.py`

```python
# sm_plot_backends/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, Tuple
import numpy as np
from PyQt6 import QtWidgets


class PlotBackend(str, Enum):
    MATPLOTLIB = "matplotlib"   # default — baseline
    PYQTGRAPH  = "pyqtgraph"    # GPU-accelerated, interativo
    PLOTLY     = "plotly"       # HTML, hover rico
    VISPY      = "vispy"        # GL puro, 100k+ pontos


@dataclass
class AxisConfig:
    """Configuração de um eixo (depth no Y, geralmente invertido)."""
    title: str = ""
    xlabel: str = ""
    ylabel: str = "Profundidade (m)"
    invert_y: bool = True          # depth cresce para baixo
    grid: bool = True
    log_x: bool = False


class SubplotHandle(Protocol):
    """Handle opaco — cada backend retorna seu tipo nativo aqui."""
    ...


class PlotCanvas(ABC):
    """Interface comum aos 4 backends. Substitui EMCanvas onde possível.

    Princípio: a função plot_tensor_full() não importa qual backend está
    rodando — ela chama add_subplot, plot_line, set_axis_config, draw.
    """

    @abstractmethod
    def widget(self) -> QtWidgets.QWidget: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def draw(self) -> None: ...

    @abstractmethod
    def save(self, path: str, dpi: int = 150) -> None: ...

    @abstractmethod
    def add_subplot_grid(self, rows: int, cols: int) -> Sequence[Sequence[SubplotHandle]]:
        """Retorna matriz [r][c] de handles. Cada backend escolhe o tipo."""

    @abstractmethod
    def plot_line(self, ax: SubplotHandle, x: np.ndarray, y: np.ndarray,
                  *, label: str = "", color: Optional[str] = None,
                  linewidth: float = 1.5, linestyle: str = "-") -> None: ...

    @abstractmethod
    def add_hline(self, ax: SubplotHandle, y: float, *, color: str = "k",
                  linestyle: str = "--", alpha: float = 0.4) -> None: ...

    @abstractmethod
    def set_axis_config(self, ax: SubplotHandle, cfg: AxisConfig) -> None: ...


def make_canvas(backend: PlotBackend, parent=None, **kwargs) -> PlotCanvas:
    """Factory — single point of construction."""
    if backend == PlotBackend.MATPLOTLIB:
        from .mpl_canvas import MatplotlibCanvas
        return MatplotlibCanvas(parent=parent, **kwargs)
    if backend == PlotBackend.PYQTGRAPH:
        from .pyqtgraph_canvas import PyQtGraphCanvas
        return PyQtGraphCanvas(parent=parent, **kwargs)
    if backend == PlotBackend.PLOTLY:
        from .plotly_canvas import PlotlyCanvas
        return PlotlyCanvas(parent=parent, **kwargs)
    if backend == PlotBackend.VISPY:
        from .vispy_canvas import VispyCanvas
        return VispyCanvas(parent=parent, **kwargs)
    raise ValueError(f"backend desconhecido: {backend}")
```

**Por que não Adapter puro**: tentar mapear `matplotlib.Axes` → `pyqtgraph.PlotItem` 1:1 é uma armadilha — `ax.set_xlim` vs `pi.setXRange`, `ax.imshow` vs `pi.addItem(ImageItem)`, `ax.text` vs `TextItem`. Adapter funciona quando 80% da API é coincidente; aqui é <30%.

**Por que não Strategy puro sem Factory**: Strategy isolado força quem instancia a importar todos os 4 backends. O Factory diferenciado por enum permite **import lazy** — pyqtgraph/plotly/vispy só são importados se selecionados.

---

## 2. Backend PyQtGraph — quase 1:1 com matplotlib

PyQtGraph é o backend de **menor atrito** depois do matplotlib. Tem `GraphicsLayoutWidget` (grid de subplots), eixos com labels, exporters embutidos. Y-invert nativo via `pi.invertY(True)`.

### Skeleton — `sm_plot_backends/pyqtgraph_canvas.py`

```python
# sm_plot_backends/pyqtgraph_canvas.py
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter
from PyQt6 import QtWidgets, QtCore
import numpy as np
from .base import PlotCanvas, AxisConfig


class PyQtGraphCanvas(PlotCanvas):
    """Canvas pyqtgraph — drop-in para EMCanvas via .figure shim."""

    def __init__(self, parent=None, figsize=(10, 6), style=None):
        self._w = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(self._w)
        layout.setContentsMargins(0, 0, 0, 0)
        # Estilo cinza claro ~ matplotlib whitegrid
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOption("antialias", True)
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)
        self._plots: list[list[pg.PlotItem]] = []
        # Crosshair compartilhado (depth comum em 18 subplots)
        self._crosshair_lines: list[pg.InfiniteLine] = []
        self._proxies: list[pg.SignalProxy] = []

    def widget(self): return self._w
    def clear(self):
        self.glw.clear()
        self._plots.clear()
        self._crosshair_lines.clear()
        self._proxies.clear()
    def draw(self): pass  # paint event imediato

    def save(self, path: str, dpi: int = 150) -> None:
        if path.lower().endswith(".svg"):
            SVGExporter(self.glw.scene()).export(path)
        else:
            exp = ImageExporter(self.glw.scene())
            exp.parameters()["width"] = int(10 * dpi)  # ~1500 px @ dpi=150
            exp.export(path)

    def add_subplot_grid(self, rows: int, cols: int):
        """Replica gs = canvas.figure.add_gridspec(3, 6) para o tensor 3×6."""
        self.glw.clear()
        self._plots = [[None]*cols for _ in range(rows)]  # type: ignore
        for r in range(rows):
            for c in range(cols):
                pi = self.glw.addPlot(row=r, col=c)
                pi.invertY(True)             # depth ↓
                pi.showGrid(x=True, y=True, alpha=0.3)
                pi.setMenuEnabled(False)
                self._plots[r][c] = pi
                # Sincroniza Y entre TODOS os subplots da coluna
                if r > 0:
                    pi.setYLink(self._plots[0][c])
        return self._plots

    def plot_line(self, ax, x, y, *, label="", color=None, linewidth=1.5, linestyle="-"):
        pen = pg.mkPen(color=color or "tab:blue", width=linewidth,
                       style=_qstyle(linestyle))
        ax.plot(x, y, pen=pen, name=label or None)

    def add_hline(self, ax, y, *, color="k", linestyle="--", alpha=0.4):
        line = pg.InfiniteLine(pos=y, angle=0,
                               pen=pg.mkPen(color=color, style=_qstyle(linestyle),
                                            width=1))
        ax.addItem(line)

    def set_axis_config(self, ax, cfg: AxisConfig):
        ax.setTitle(cfg.title)
        ax.setLabel("bottom", cfg.xlabel)
        ax.setLabel("left", cfg.ylabel)
        ax.invertY(cfg.invert_y)
        if cfg.log_x:
            ax.setLogMode(x=True, y=False)
        ax.showGrid(x=cfg.grid, y=cfg.grid, alpha=0.3)


def _qstyle(s: str):
    return {"-": QtCore.Qt.PenStyle.SolidLine,
            "--": QtCore.Qt.PenStyle.DashLine,
            ":": QtCore.Qt.PenStyle.DotLine,
            "-.": QtCore.Qt.PenStyle.DashDotLine}.get(s, QtCore.Qt.PenStyle.SolidLine)
```

**Pontos importantes**:
- `setYLink` em loop replica `sharey` do matplotlib — pan/zoom sincronizado em todos os 18 subplots de profundidade.
- `ImageExporter` exige `pyqtgraph >= 0.13` para suporte estável a `width=` parametrizável.
- Retorne sempre `pi.invertY(True)` antes de plotar — caso contrário a primeira linha define o range invertido.

---

## 3. Backend Plotly — `QWebEngineView` + HTML

Plotly tem o **melhor hover** dos 4, mas o pior performance/memória (Chromium embedded). Para 18 subplots com 600 pontos cada (10.800 pontos) ainda é viável; >100k pontos começa a travar.

### Skeleton — `sm_plot_backends/plotly_canvas.py`

```python
# sm_plot_backends/plotly_canvas.py
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6 import QtWidgets, QtCore
from .base import PlotCanvas, AxisConfig


class PlotlyCanvas(PlotCanvas):
    """Canvas Plotly — renderiza via Chromium embarcado (QWebEngineView).

    Estratégia de update: bufferiza traces no self._fig durante o ciclo
    plot_*, e dispara um único setHtml() em draw() (evita re-render
    do Chromium a cada plot_line).
    """

    def __init__(self, parent=None, figsize=(10, 6), style=None):
        self._w = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(self._w)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = QWebEngineView()
        # Necessário para PyInstaller / sandboxes restritos
        self.view.settings().setAttribute(
            self.view.settings().WebAttribute.LocalContentCanAccessRemoteUrls, True)
        layout.addWidget(self.view)
        self._fig: go.Figure | None = None
        self._rows = 1
        self._cols = 1

    def widget(self): return self._w

    def clear(self):
        self._fig = None
        self.view.setHtml("<html><body style='background:#fff'/></html>")

    def draw(self):
        if self._fig is None: return
        # include_plotlyjs="cdn" reduz HTML de ~3MB → ~5KB; offline use "inline"
        html = self._fig.to_html(include_plotlyjs="cdn", full_html=True,
                                 config={"displaylogo": False, "responsive": True})
        self.view.setHtml(html, QtCore.QUrl("about:blank"))

    def save(self, path: str, dpi: int = 150):
        if path.endswith(".html"):
            self._fig.write_html(path)
        else:
            # Requer kaleido instalado
            self._fig.write_image(path, scale=dpi/72)

    def add_subplot_grid(self, rows: int, cols: int):
        self._rows, self._cols = rows, cols
        self._fig = make_subplots(
            rows=rows, cols=cols,
            shared_yaxes=True,                  # depth comum
            horizontal_spacing=0.02,
            vertical_spacing=0.04,
        )
        # Inverte Y em TODOS os subplots (depth ↓)
        for r in range(1, rows+1):
            for c in range(1, cols+1):
                self._fig.update_yaxes(autorange="reversed", row=r, col=c)
        # Handles = tuplas (r, c) — Plotly não tem objeto Axes
        return [[(r+1, c+1) for c in range(cols)] for r in range(rows)]

    def plot_line(self, ax, x, y, *, label="", color=None, linewidth=1.5, linestyle="-"):
        r, c = ax
        self._fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                name=label, line=dict(width=linewidth, color=color,
                                      dash=_pdash(linestyle)),
                hovertemplate=("z=%{y:.2f} m<br>"
                               "val=%{x:.4e}<br>"
                               "<extra>" + label + "</extra>"),
            ),
            row=r, col=c)

    def add_hline(self, ax, y, *, color="k", linestyle="--", alpha=0.4):
        r, c = ax
        self._fig.add_hline(y=y, line_color=color, line_dash=_pdash(linestyle),
                            opacity=alpha, row=r, col=c)

    def set_axis_config(self, ax, cfg: AxisConfig):
        r, c = ax
        self._fig.update_xaxes(title_text=cfg.xlabel, type="log" if cfg.log_x else "linear",
                               row=r, col=c, showgrid=cfg.grid)
        self._fig.update_yaxes(title_text=cfg.ylabel,
                               autorange="reversed" if cfg.invert_y else True,
                               row=r, col=c, showgrid=cfg.grid)
        if cfg.title:
            self._fig.layout.annotations += (dict(
                text=cfg.title, xref="paper", yref="paper", showarrow=False), )


def _pdash(s): return {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}.get(s, "solid")
```

**Performance — 18 subplots**:
- `setHtml` com 18 subplots × 600 pontos: ~120ms inicial em Chromium, depois interações são GPU-aceleradas.
- **NÃO** chame `setHtml` por trace — bufferize tudo em `_fig`, depois chame `draw()` uma vez.
- Para >50k pontos por subplot, troque `mode="lines"` por `Scattergl` (WebGL).
- `include_plotlyjs="cdn"` requer rede; em ambiente offline use `"inline"` (HTML ~3 MB).

**hovertemplate** já dá tooltip nativo com depth + valor + nome do trace — sem código adicional.

---

## 4. Backend Vispy — vale a pena?

**Resposta curta: NÃO, para este caso.**

**Por quê**:
- Vispy não tem *axes nativos* — você precisa montar `AxisVisual` manualmente para cada um dos 18 subplots, posicionar tick labels, montar grid lines. Estima-se +400 LOC só para reproduzir o que o pyqtgraph dá grátis.
- O ganho de Vispy aparece em **>1M pontos por linha**. Você tem ~600 pontos × 18 subplots = 10.800 — pyqtgraph já desenha isso a 60 fps.
- A integração Qt do Vispy (`vispy.app.use_app("pyqt6")`) pode entrar em conflito com `QtWebEngineView` (do Plotly) — bug histórico, eventos OpenGL competem.
- O usuário-alvo é geofísico, espera **labels de eixos sempre visíveis e formatados**. Vispy força cada label a ser código.

**Quando reconsiderar**: se em algum momento você plotar tensor 3D (Picasso volumétrico, ~10⁶ voxels) ou animação de inversão em tempo real com >100k pontos/frame.

**Skeleton mínimo** (caso queira deixar plumbed para o futuro):

```python
# sm_plot_backends/vispy_canvas.py
from vispy import scene
from vispy.scene.visuals import Line
from PyQt6 import QtWidgets
import numpy as np
from .base import PlotCanvas, AxisConfig


class VispyCanvas(PlotCanvas):
    """Vispy — opt-in apenas para >1M pontos. Axes manuais requeridos."""

    def __init__(self, parent=None, figsize=(10, 6), style=None):
        self._w = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(self._w)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="white",
                                        parent=self._w)
        layout.addWidget(self.canvas.native)
        self.grid = self.canvas.central_widget.add_grid()
        self._views: list[list[scene.ViewBox]] = []

    def widget(self): return self._w
    def clear(self):
        self.grid.parent = None
        self.grid = self.canvas.central_widget.add_grid()
        self._views.clear()
    def draw(self): self.canvas.update()
    def save(self, path, dpi=150):
        img = self.canvas.render(alpha=False)
        from PIL import Image; Image.fromarray(img).save(path)

    def add_subplot_grid(self, rows, cols):
        self._views = [[None]*cols for _ in range(rows)]  # type: ignore
        for r in range(rows):
            for c in range(cols):
                vb = self.grid.add_view(row=r, col=c, camera="panzoom")
                vb.camera.flip = (False, True, False)  # invert Y
                self._views[r][c] = vb
        return self._views

    def plot_line(self, ax, x, y, *, label="", color=None, linewidth=1.5, linestyle="-"):
        pts = np.column_stack([x, y]).astype(np.float32)
        line = Line(pts, color=color or (0.2, 0.4, 0.8, 1.0), width=linewidth,
                    parent=ax.scene)
        ax.camera.set_range()  # autoscale

    def add_hline(self, ax, y, **kw):  # pragma: no cover - opcional
        ...
    def set_axis_config(self, ax, cfg):  # pragma: no cover - opcional
        # AxisVisual manual + Text labels — não-trivial, ~80 LOC
        ...
```

**Recomendação**: deixe `PlotBackend.VISPY` no enum mas marque a classe como experimental (`raise NotImplementedError("Vispy backend é experimental — sem axes")` no `set_axis_config`). Implemente só quando demanda real surgir.

---

## 5. Toggle de Backend em Runtime

### Estratégia de troca

**Importante**: NÃO tente trocar canvas in-place. PyQtGraph e QWebEngineView têm threads de render distintos — destruir um e criar outro é mais seguro que reconfigurar.

```python
# Em simulation_manager.py — diff conceitual
class SimulationManager(QtWidgets.QMainWindow):
    def _build_plot_panel(self):
        self.plot_container = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self._current_canvas: PlotCanvas | None = None

        # Combo box no toolbar
        self.backend_combo = QtWidgets.QComboBox()
        for b in PlotBackend:
            self.backend_combo.addItem(b.value, b)
        # Restaura escolha persistida
        settings = QtCore.QSettings("GeosteeringAI", "SimulationManager")
        saved = settings.value("plot_backend", PlotBackend.MATPLOTLIB.value)
        idx = self.backend_combo.findText(saved)
        self.backend_combo.setCurrentIndex(max(0, idx))
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)

        self._switch_backend(PlotBackend(self.backend_combo.currentData()))

    def _on_backend_changed(self, _idx: int):
        backend = PlotBackend(self.backend_combo.currentData())
        QtCore.QSettings("GeosteeringAI", "SimulationManager").setValue(
            "plot_backend", backend.value)
        self._switch_backend(backend)
        # Replota com dados atuais
        if self._last_plot_call is not None:
            fn, args, kwargs = self._last_plot_call
            fn(self._current_canvas, *args, **kwargs)

    def _switch_backend(self, backend: PlotBackend):
        # 1. Remove canvas atual do layout
        if self._current_canvas is not None:
            self.plot_layout.removeWidget(self._current_canvas.widget())
            self._current_canvas.widget().deleteLater()
        # 2. Cria novo canvas
        self._current_canvas = make_canvas(backend, parent=self.plot_container)
        self.plot_layout.addWidget(self._current_canvas.widget())
```

**Persistência**: `QSettings("GeosteeringAI", "SimulationManager")` salva no INI nativo do OS (macOS plist, Linux ~/.config, Windows registry). Padrão Qt; sem dependências.

### Organização de arquivos — recomendação

**NÃO** crie `sm_plots_pyqtgraph.py`, `sm_plots_plotly.py`. Isso duplica 1700 LOC × 4. Em vez disso:

```
geosteering_ai/simulation/tests/
├── sm_plots.py                    # funções plot_*() agnósticas (refatoradas)
├── sm_plot_backends/
│   ├── __init__.py                # exporta make_canvas, PlotBackend
│   ├── base.py                    # PlotCanvas ABC, AxisConfig, factory
│   ├── mpl_canvas.py              # MatplotlibCanvas (substitui EMCanvas)
│   ├── pyqtgraph_canvas.py        # PyQtGraphCanvas
│   ├── plotly_canvas.py           # PlotlyCanvas
│   └── vispy_canvas.py            # VispyCanvas (stub)
└── sm_plot_cache.py               # cache existente (sem mudanças)
```

A refatoração de `sm_plots.py` substitui `canvas.figure.add_subplot(gs[r,c])` por `canvas.add_subplot_grid(rows, cols)[r][c]`. As ~30 chamadas viram um único helper. Mantém compatibilidade: se canvas é `MatplotlibCanvas`, `add_subplot_grid` retorna `Axes` matplotlib; código legado que acessar `ax.set_xlabel` continua funcionando porque é o objeto nativo (Adapter parcial — vazamento controlado).

---

## 6. Hover / Inspeção de Dados

### matplotlib (offline + crosshair)

```python
# sm_plot_backends/mpl_canvas.py — método auxiliar
def _attach_hover(self, ax, lines: list, tooltip_fn):
    """tooltip_fn(x, y, line_idx) -> str"""
    annot = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="lemonchiffon", alpha=0.9),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def on_motion(event):
        if event.inaxes is not ax: return
        for i, ln in enumerate(lines):
            cont, info = ln.contains(event)
            if cont:
                idx = info["ind"][0]
                xs, ys = ln.get_data()
                annot.xy = (xs[idx], ys[idx])
                annot.set_text(tooltip_fn(xs[idx], ys[idx], i))
                annot.set_visible(True)
                self.canvas.draw_idle()
                return
        if annot.get_visible():
            annot.set_visible(False)
            self.canvas.draw_idle()

    self.canvas.mpl_connect("motion_notify_event", on_motion)
```

### PyQtGraph (SignalProxy)

```python
def _attach_hover_pg(self, ax: pg.PlotItem, curves: list, tooltip_fn):
    label = pg.TextItem("", anchor=(0, 1), color="k",
                        fill=pg.mkBrush(255, 255, 200, 220))
    ax.addItem(label, ignoreBounds=True)
    def on_move(evt):
        pos = evt[0]
        if not ax.sceneBoundingRect().contains(pos): return
        mp = ax.vb.mapSceneToView(pos)
        # busca ponto mais próximo na curva mais próxima
        best = None
        for i, curve in enumerate(curves):
            xd, yd = curve.getData()
            d = (xd - mp.x())**2 + (yd - mp.y())**2
            j = int(d.argmin())
            if best is None or d[j] < best[0]:
                best = (d[j], i, j, xd[j], yd[j])
        if best:
            _, i, j, xv, yv = best
            label.setText(tooltip_fn(xv, yv, i))
            label.setPos(xv, yv)
    proxy = pg.SignalProxy(ax.scene().sigMouseMoved, rateLimit=60, slot=on_move)
    self._proxies.append(proxy)  # IMPORTANTE: manter referência viva
```

**Bug clássico**: se `proxy` for variável local, garbage collector destrói antes do primeiro evento. Sempre persista em `self._proxies`.

### Plotly (zero código)

`hovertemplate` no `plot_line` já gera tooltip rico. Para mostrar layer info dinamicamente, embute camada como `customdata`:

```python
go.Scatter(
    x=x, y=y, customdata=layer_idx_per_point,
    hovertemplate=("z=%{y:.2f} m<br>"
                   "|H|=%{x:.4e}<br>"
                   "Camada %{customdata}<br>"
                   "<extra></extra>"))
```

### Tooltip-fn comum (geofísica)

```python
def em_tooltip(x, y, line_idx, *, layer_z_tops, layer_rho_h):
    """Tooltip rico para perfil EM."""
    layer = int(np.searchsorted(layer_z_tops, y, side="right") - 1)
    rho = layer_rho_h[max(0, min(layer, len(layer_rho_h)-1))]
    return (f"z = {y:.2f} m\n"
            f"valor = {x:.4e}\n"
            f"camada {layer} (ρh={rho:.1f} Ω·m)\n"
            f"linha {COMPONENT_NAMES[line_idx]}")
```

---

## 7. Crosshair Sincronizado em 18 Subplots

Caso de uso: arrastar mouse em qualquer um dos 18 subplots → linha horizontal aparece em **todos** na mesma profundidade.

### matplotlib (blitting — performance crítica)

```python
class CrossHairSync:
    """Crosshair Y-sync com blitting em N subplots."""

    def __init__(self, canvas, axes_flat: list):
        self.canvas = canvas              # FigureCanvasQTAgg
        self.axes = axes_flat
        self.lines = [ax.axhline(y=0, color="r", lw=0.8, alpha=0.6,
                                 visible=False, animated=True)
                      for ax in axes_flat]
        self.bg = None
        canvas.mpl_connect("draw_event", self._on_draw)
        canvas.mpl_connect("motion_notify_event", self._on_move)

    def _on_draw(self, _evt):
        # Captura background SEM as linhas — chave do blitting
        self.bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        for ln in self.lines:
            ln.axes.draw_artist(ln)

    def _on_move(self, event):
        if event.inaxes not in self.axes or self.bg is None:
            return
        y = event.ydata
        self.canvas.restore_region(self.bg)
        for ln in self.lines:
            ln.set_ydata([y, y])
            ln.set_visible(True)
            ln.axes.draw_artist(ln)
        self.canvas.blit(self.canvas.figure.bbox)
```

Sem blitting, cada movimento do mouse força re-render de 18 axes (~80ms = 12 fps); com blitting fica em ~3ms (>200 fps perceptíveis).

### PyQtGraph (sigPositionChanged)

```python
def install_crosshair_sync(canvas: PyQtGraphCanvas):
    """Em pyqtgraph, basta uma InfiniteLine + sinal por subplot."""
    plots_flat = [p for row in canvas._plots for p in row]
    # Linha mestre — fonte da verdade
    master = pg.InfiniteLine(angle=0, pen=pg.mkPen("r", width=1, style=Qt.DashLine),
                             movable=True)
    plots_flat[0].addItem(master, ignoreBounds=True)
    slaves = []
    for ax in plots_flat[1:]:
        s = pg.InfiniteLine(angle=0, pen=pg.mkPen("r", width=1, style=Qt.DashLine),
                            movable=False)
        ax.addItem(s, ignoreBounds=True)
        slaves.append(s)
    def on_change():
        y = master.value()
        for s in slaves:
            s.setPos(y)
    master.sigPositionChanged.connect(on_change)
    # Mouse-driven (não só drag): proxy global
    def on_mouse(evt):
        pos = evt[0]
        for ax in plots_flat:
            if ax.sceneBoundingRect().contains(pos):
                y = ax.vb.mapSceneToView(pos).y()
                master.setPos(y)
                break
    proxy = pg.SignalProxy(canvas.glw.scene().sigMouseMoved, rateLimit=60,
                           slot=on_mouse)
    canvas._proxies.append(proxy)
```

### Plotly (limitado)

Plotly tem `spikemode="across"` em `update_xaxes`/`update_yaxes` que dá crosshair por subplot, mas **não sincroniza Y entre subplots por padrão**. Para sincronia real, precisaria JS injetado via `<script>` no HTML. Aceitável: ligar `shared_yaxes=True` no `make_subplots` (zoom Y já é compartilhado) e deixar o spike só por subplot — no caso geofísico, é suficiente.

---

## Resumo de Decisões e Esforço

| Item | Decisão | Esforço estimado |
|:-----|:--------|:-----------------|
| Padrão | Strategy + Factory híbrido | — |
| Refatorar `sm_plots.py` | Substituir `canvas.figure.add_subplot` → `canvas.add_subplot_grid()` | ~4h (30 sites) |
| Backend matplotlib | Wrapper `MatplotlibCanvas` em torno do `EMCanvas` atual | ~2h |
| Backend pyqtgraph | Implementar `PyQtGraphCanvas` completo | ~6h |
| Backend plotly | Implementar `PlotlyCanvas` (QWebEngineView) | ~4h |
| Backend vispy | Stub apenas — `raise NotImplementedError` | ~30min |
| Toggle UI + QSettings | Combo box no toolbar + persistência | ~1h |
| Hover (mpl + pyqtgraph) | Helpers `_attach_hover_*` + tooltip_fn comum | ~3h |
| Crosshair sync | Blitting (mpl) + SignalProxy (pyqtgraph) | ~4h |
| **Total** | | **~24h** |

## Arquivos relevantes (paths absolutos)

- `/Users/daniel/Geosteering_AI/geosteering_ai/simulation/tests/sm_plots.py` — 1691 LOC, 6 funções plot_*, classe EMCanvas (linhas 255-322)
- `/Users/daniel/Geosteering_AI/geosteering_ai/simulation/tests/simulation_manager.py` — 7646 LOC, integra EMCanvas em abas
- `/Users/daniel/Geosteering_AI/geosteering_ai/simulation/tests/sm_plot_cache.py` — cache (sem mudanças)
- `/Users/daniel/Geosteering_AI/geosteering_ai/simulation/tests/sm_qt_compat.py` — compat PyQt5/6 já existente

## Riscos / Pontos de atenção

1. **PyQt6 + QWebEngine**: `PyQt6-WebEngine` é pacote separado (`pip install PyQt6-WebEngine`). Adicionar em `pyproject.toml` extras: `[project.optional-dependencies] plotly = ["plotly>=5", "PyQt6-WebEngine>=6.5"]`.
2. **constrained_layout** do matplotlib não tem equivalente em pyqtgraph — labels longos podem cortar. Usar `glw.ci.layout.setContentsMargins(20, 10, 10, 30)`.
3. **Y-invert**: 3 dos 4 backends têm jeitos diferentes (`ax.invert_yaxis()`, `pi.invertY(True)`, `update_yaxes(autorange='reversed')`, `camera.flip[1]=True`). Centralizar em `set_axis_config`.
4. **Vazamento controlado de tipo**: `SubplotHandle` é `Protocol` mas na prática é `Axes`/`PlotItem`/tuple/`ViewBox`. Funções `plot_*` que precisam de feature exclusiva (ex: `ax.imshow` para Picasso) terão `if isinstance(canvas, MatplotlibCanvas): ...` — aceitável, e justifica manter a maior parte da lógica na ABC.
5. **Testes**: `test_plots.py` deve rodar com `MATPLOTLIB` (offscreen Agg) — backends pyqtgraph/plotly/vispy só smoke-test em CI com `xvfb` e marcador `pytest.mark.gui`.
