# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/results_viewmodel.py            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ResultsViewModel — estado da galeria de resultados (PURO)  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0011d)         ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — galeria do ensemble (Fatia 4)                    ║
# ║  Framework   : Python + numpy PURO — NÃO importa Qt (Princípio X)          ║
# ║  Dependências: gui.viewmodels (BaseViewModel/VMSignal), gui.persistence    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Guarda o ÚLTIMO resultado (H6) + o estado dos seletores da galeria      ║
# ║    (componente Hxx..Hzz, plot-kind Re/Im/|H|/fase, índices TR/dip/freq,    ║
# ║    paginação) e DERIVA as curvas por modelo (transform do plot-kind sobre  ║
# ║    ``H6[m, tr, dip, :, freq, comp]``). NÃO simula — só re-exibe o H6 já     ║
# ║    computado. Curvas são memorizadas num LRU (gui.persistence.plot_cache). ║
# ║                                                                           ║
# ║  PUREZA (Princípio X)                                                     ║
# ║    SÓ numpy + BaseViewModel/VMSignal + LRUPlotCache (todos Qt-free).        ║
# ║    A ResultsView (Qt) liga-se a ``changed``/``results_changed`` e plota.   ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ResultsViewModel · COMPONENT_NAMES · PLOT_KINDS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ResultsViewModel`` — estado da galeria do ensemble (PURO; curvas + cache LRU)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache
from geosteering_ai.gui.plot_backends.base import PlotBackend  # str-Enum PURO (sem Qt)
from geosteering_ai.gui.services.derived import (  # PUROS (numpy; sem Qt)
    GEOSIGNALS,
    compute_geosignal,
    lambda_profile,
    rho_profile,
)
from geosteering_ai.gui.viewmodels.base import BaseViewModel
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = [
    "ResultsViewModel",
    "COMPONENT_NAMES",
    "CHANNELS",
    "PLOT_KINDS",
    "PLOT_MODES",
]

# Ordem canônica das 9 componentes no eixo −1 do H6 (multi_forward.py:395):
# [Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz] — index 0=Hxx … 8=Hzz.
COMPONENT_NAMES: Tuple[str, ...] = (
    "Hxx",
    "Hxy",
    "Hxz",
    "Hyx",
    "Hyy",
    "Hyz",
    "Hzx",
    "Hzy",
    "Hzz",
)

# Canais plotáveis: 9 componentes cruas (Hxx..Hzz) + 5 geosinais derivados
# (USD/UAD/UHR/UHA/U3DF — byte-fiéis, ver gui/services/derived.py). Os índices
# 0..8 são componentes; 9..13 são geosinais (Fatia 6d).
CHANNELS: Tuple[str, ...] = COMPONENT_NAMES + GEOSIGNALS

# Modos de plot (transform aplicado à componente complexa selecionada).
PLOT_KINDS: Tuple[str, ...] = ("re", "im", "mag", "phase")

# Modos da galeria (o QUE plotar): curvas do canal, perfil ρ(z), anisotropia λ(z),
# ou heatmap do ensemble (imagem n_models × n_pos do canal/transform atuais).
PLOT_MODES: Tuple[str, ...] = ("curve", "rho", "lambda", "heatmap")


def _kind_transform(kind: str, h: np.ndarray) -> np.ndarray:
    """Aplica o plot-kind à curva complexa ``h`` → curva real ``(n_pos,)``."""
    if kind == "im":
        return np.asarray(np.imag(h), dtype=np.float64)
    if kind == "mag":
        return np.asarray(np.abs(h), dtype=np.float64)
    if kind == "phase":
        return np.asarray(np.degrees(np.angle(h)), dtype=np.float64)
    return np.asarray(np.real(h), dtype=np.float64)  # "re" (default)


_DEFAULT_PAGE_SIZE = 12
_DEFAULT_CACHE_MAXLEN = 256


class ResultsViewModel(BaseViewModel):
    """ViewModel da galeria de resultados (PURO; seletores + curvas + cache LRU).

    Attributes:
        results_changed: ``VMSignal`` — emitido ao receber um NOVO resultado
            (a View re-monta a grade). (herda ``changed: VMSignal`` p/ seletores.)

    Example:
        >>> rvm = ResultsViewModel()
        >>> rvm.results_changed.connect(view._render)
        >>> rvm.changed.connect(view._render)          # troca de seletor
        >>> rvm.set_result({"H6": h6, "positions_z": z})
        >>> rvm.plot_kind = "mag"; rvm.component_index = 8   # |Hzz|
        >>> [rvm.curve_for(m) for m in rvm.page_models()]

    Note:
        Os seletores são CLAMPADOS aos ``dims`` em :meth:`set_result` e nos
        setters — nunca apontam para um índice fora do H6 atual.
    """

    _STATE_FIELDS = (
        "_channel_index",
        "_plot_kind",
        "_plot_mode",
        "_tr_index",
        "_dip_index",
        "_freq_index",
        "_page",
        "_focus_model",
    )

    def __init__(
        self,
        page_size: int = _DEFAULT_PAGE_SIZE,
        cache: Optional[LRUPlotCache] = None,
        plot_backend: PlotBackend = PlotBackend.PYQTGRAPH,
    ) -> None:
        """Inicializa sem resultado (galeria vazia).

        Args:
            page_size: nº de modelos por página da galeria (default 12).
            cache: LRU de curvas (default ``LRUPlotCache(maxlen=256)``) — bounded
                por contagem; injetável p/ teste.
            plot_backend: backend de plot inicial (default ``PYQTGRAPH`` — interativo/
                dinâmico, Fatia 6d). A View recria o canvas ao trocar e cai p/
                MATPLOTLIB (``_build_canvas``) se o PyQtGraph faltar no ambiente.
        """
        super().__init__()
        self._result: Optional[Dict[str, Any]] = None
        self._h6: Optional[np.ndarray] = None
        self._positions_z: Optional[np.ndarray] = None
        self._channel_index: int = 0  # 0..8 componentes; 9..13 geosinais (CHANNELS)
        self._plot_kind: str = "re"
        self._plot_mode: str = "curve"  # curve | rho | lambda | heatmap (PLOT_MODES)
        self._tr_index: int = 0
        self._dip_index: int = 0
        self._freq_index: int = 0
        self._page: int = 0
        self._focus_model: int = 0  # modelo em foco (animation bar — single-model)
        self._plot_backend: PlotBackend = plot_backend
        self._page_size: int = max(1, int(page_size))
        # ``is not None`` (NÃO ``cache or ...``): um LRUPlotCache VAZIO é falsy
        # (``__len__`` == 0) e seria descartado pelo ``or``.
        self._cache: LRUPlotCache = (
            cache
            if cache is not None
            else LRUPlotCache(maxlen=_DEFAULT_CACHE_MAXLEN, max_bytes=64_000_000.0)
        )
        self.results_changed: VMSignal = VMSignal()

    # ── Recebe o resultado ───────────────────────────────────────────────────
    def set_result(self, result: Optional[Dict[str, Any]]) -> None:
        """Recebe um NOVO resultado (dict com ``H6``/``positions_z``); clampa seletores.

        Limpa o cache de curvas e re-clampa todos os índices aos novos ``dims``
        (mantém o seletor onde possível). Emite ``results_changed`` (a View re-monta).

        Args:
            result: dict de :func:`_run_simulation` (``H6``, ``positions_z``, …) ou
                ``None`` (galeria vazia).
        """
        self._result = result
        h6 = result.get("H6") if isinstance(result, dict) else None
        # Aceita só um H6 6-D com EXATAMENTE 9 componentes (Hxx..Hzz no eixo −1) —
        # rejeita H6 malformado na ENTRADA (não delega o erro a curve_for).
        self._h6 = (
            h6
            if (h6 is not None and getattr(h6, "ndim", 0) == 6 and h6.shape[5] == 9)
            else None
        )
        self._positions_z = (
            result.get("positions_z") if isinstance(result, dict) else None
        )
        self._cache.clear()
        # Re-clampa seletores aos novos dims (preserva onde couber).
        dims = self.dims
        if dims is not None:
            n_models, n_tr, n_ang, _n_pos, n_f = dims
            self._channel_index = int(np.clip(self._channel_index, 0, len(CHANNELS) - 1))
            self._tr_index = int(np.clip(self._tr_index, 0, n_tr - 1))
            self._dip_index = int(np.clip(self._dip_index, 0, n_ang - 1))
            self._freq_index = int(np.clip(self._freq_index, 0, n_f - 1))
            self._page = int(np.clip(self._page, 0, max(0, self.n_pages - 1)))
            self._focus_model = int(np.clip(self._focus_model, 0, n_models - 1))
        self.results_changed.emit(result)

    # ── Properties derivadas (read-only) ─────────────────────────────────────
    @property
    def has_result(self) -> bool:
        """``True`` se há um H6 6-D válido carregado."""
        return self._h6 is not None

    @property
    def dims(self) -> Optional[Tuple[int, int, int, int, int]]:
        """``(n_models, nTR, nAng, n_pos, nf)`` ou ``None`` (sem resultado)."""
        if self._h6 is None:
            return None
        s = self._h6.shape
        return (int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4]))

    @property
    def n_models(self) -> int:
        """Nº de modelos no ensemble (0 se sem resultado)."""
        return 0 if self._h6 is None else int(self._h6.shape[0])

    @property
    def n_pages(self) -> int:
        """Nº de páginas da galeria (``ceil(n_models/page_size)``)."""
        if self.n_models == 0:
            return 0
        return -(-self.n_models // self._page_size)  # ceil

    @property
    def page_size(self) -> int:
        """Nº de modelos por página (read-only)."""
        return self._page_size

    @property
    def depth(self) -> Optional[np.ndarray]:
        """``positions_z`` ``(n_pos,)`` ou ``None``."""
        return self._positions_z

    @property
    def channel_name(self) -> str:
        """Nome do canal selecionado (Hxx..Hzz OU geosinal USD..U3DF)."""
        return CHANNELS[int(np.clip(self._channel_index, 0, len(CHANNELS) - 1))]

    @property
    def is_geosignal(self) -> bool:
        """``True`` se o canal atual é um geosinal derivado (índice ≥ 9)."""
        return self._channel_index >= len(COMPONENT_NAMES)

    @property
    def component_name(self) -> str:
        """(Compat) Nome de componente (Hxx..Hzz) — geosinal cai no clip a 0..8."""
        return COMPONENT_NAMES[int(np.clip(self._channel_index, 0, 8))]

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        """O último dict de resultado recebido (ou ``None``)."""
        return self._result

    def page_models(self) -> List[int]:
        """Índices de modelo da página atual (parcial na última)."""
        if self.n_models == 0:
            return []
        start = self._page * self._page_size
        return list(range(start, min(start + self._page_size, self.n_models)))

    def curve_for(self, model_index: int) -> np.ndarray:
        """Curva ``(n_pos,)`` do modelo no canal/plot-kind atuais (componente OU geosinal).

        Para um canal componente (0..8): ``H6[m, tr, dip, :, freq, ch]``. Para um
        geosinal (9..13): extrai as 9 componentes ``H6[m, tr, dip, :, freq, :]`` e
        aplica :func:`compute_geosignal` (byte-fiel). Em ambos os casos aplica
        ``transform`` = Re/Im/|·|/fase(°) e memoriza num LRU.

        Args:
            model_index: índice do modelo (0 ≤ m < n_models).

        Returns:
            ``np.ndarray`` float64 shape ``(n_pos,)``.

        Raises:
            IndexError: se não há resultado ou ``model_index`` fora de range.
        """
        if self._h6 is None:
            raise IndexError("sem resultado — chame set_result() primeiro.")
        if not (0 <= model_index < self.n_models):
            raise IndexError(f"model_index {model_index} fora de [0, {self.n_models}).")
        key = (
            f"{model_index}|c{self._channel_index}|{self._plot_kind}"
            f"|{self._tr_index}|{self._dip_index}|{self._freq_index}"
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached["curve"]  # type: ignore[no-any-return]
        if self._channel_index < len(COMPONENT_NAMES):
            # Componente crua (índice 0..8 no eixo −1 do H6).
            h = self._h6[
                model_index,
                self._tr_index,
                self._dip_index,
                :,
                self._freq_index,
                self._channel_index,
            ]
        else:
            # Geosinal: precisa das 9 componentes (eixo −1 inteiro) → derived.
            h9 = self._h6[
                model_index, self._tr_index, self._dip_index, :, self._freq_index, :
            ]
            h = compute_geosignal(self.channel_name, h9)
        curve = _kind_transform(self._plot_kind, h)
        self._cache.put(key, {"curve": curve})
        return curve

    # ── Perfis ρ/λ + heatmap de ensemble (Fatia 6d) ──────────────────────────
    def geology_for(self, model_index: int) -> Optional[Dict[str, Any]]:
        """Geologia do modelo (``{rho_h, rho_v, thicknesses}``) ou ``None`` (ausente)."""
        if not isinstance(self._result, dict):
            return None
        geo = self._result.get("geology")
        if not isinstance(geo, list) or not (0 <= model_index < len(geo)):
            return None
        entry = geo[model_index]
        return entry if isinstance(entry, dict) else None

    def rho_curves_for(
        self, model_index: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """``(depth, ρₕ(z), ρᵥ(z))`` step do modelo, ou ``None`` se faltam dados.

        Usa a geologia exposta por ``_run_simulation`` (Fatia 6d) + ``positions_z``;
        os perfis são step-functions via :func:`rho_profile` (interfaces ``cumsum``).
        """
        geo = self.geology_for(model_index)
        if geo is None or self._positions_z is None:
            return None
        z = np.asarray(self._positions_z, dtype=float)
        rho_h = rho_profile(z, geo["rho_h"], geo["thicknesses"])
        rho_v = rho_profile(z, geo["rho_v"], geo["thicknesses"])
        return z, rho_h, rho_v

    def lambda_curve_for(
        self, model_index: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(depth, λ(z))`` step (anisotropia TIV ``√(ρᵥ/ρₕ)≥1``) ou ``None`` (sem dados)."""
        geo = self.geology_for(model_index)
        if geo is None or self._positions_z is None:
            return None
        z = np.asarray(self._positions_z, dtype=float)
        lam = lambda_profile(z, geo["rho_h"], geo["rho_v"], geo["thicknesses"])
        return z, lam

    def ensemble_image(
        self,
    ) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float], str]]:
        """Heatmap do ensemble: imagem ``(n_models × n_pos)`` do canal/transform atuais.

        Empilha :meth:`curve_for` de TODOS os modelos (linha = modelo, coluna =
        posição/profundidade) — imagem multidimensional interativa (showcase 6d).

        Returns:
            ``(img, extent, label)`` — ``img`` ``(n_models, n_pos)``; ``extent`` =
            ``(z_min, z_max, n_models−0.5, −0.5)`` (origin=upper: modelo 0 no topo);
            ``label`` = canal+transform. ``None`` se sem resultado/profundidade.
        """
        if self._h6 is None or self._positions_z is None or self.n_models == 0:
            return None
        rows = [self.curve_for(m) for m in range(self.n_models)]
        img = np.vstack(rows)
        z = np.asarray(self._positions_z, dtype=float)
        z_min = float(z.min()) if z.size else 0.0
        z_max = float(z.max()) if z.size else 1.0
        extent = (z_min, z_max, float(self.n_models) - 0.5, -0.5)
        kind_label = {"re": "Re", "im": "Im", "mag": "|·|", "phase": "fase°"}.get(
            self._plot_kind, self._plot_kind
        )
        label = f"{kind_label} {self.channel_name}"
        return img, extent, label

    # ── Properties de seletor (setters clampam + emitem ``changed``) ─────────
    @property
    def channel_index(self) -> int:
        """Índice do canal (0..8 componentes Hxx..Hzz; 9..13 geosinais USD..U3DF)."""
        return self._channel_index

    @channel_index.setter
    def channel_index(self, value: int) -> None:
        self._set("_channel_index", int(np.clip(int(value), 0, len(CHANNELS) - 1)))

    @property
    def component_index(self) -> int:
        """(Compat) Índice de componente (0=Hxx … 8=Hzz) — alias de ``channel_index``."""
        return self._channel_index

    @component_index.setter
    def component_index(self, value: int) -> None:
        # Compat: componentes vivem em 0..8 (clip preserva o contrato legado).
        self._set("_channel_index", int(np.clip(int(value), 0, 8)))

    @property
    def plot_kind(self) -> str:
        """Modo de plot: re | im | mag | phase."""
        return self._plot_kind

    @plot_kind.setter
    def plot_kind(self, value: str) -> None:
        self._set("_plot_kind", value if value in PLOT_KINDS else "re")

    @property
    def plot_mode(self) -> str:
        """Modo da galeria: curve | rho | lambda | heatmap (o QUE plotar)."""
        return self._plot_mode

    @plot_mode.setter
    def plot_mode(self, value: str) -> None:
        self._set("_plot_mode", value if value in PLOT_MODES else "curve")

    @property
    def tr_index(self) -> int:
        """Índice do espaçamento TR a exibir."""
        return self._tr_index

    @tr_index.setter
    def tr_index(self, value: int) -> None:
        hi = (self.dims[1] - 1) if self.dims else 0
        self._set("_tr_index", int(np.clip(int(value), 0, hi)))

    @property
    def dip_index(self) -> int:
        """Índice do dip a exibir."""
        return self._dip_index

    @dip_index.setter
    def dip_index(self, value: int) -> None:
        hi = (self.dims[2] - 1) if self.dims else 0
        self._set("_dip_index", int(np.clip(int(value), 0, hi)))

    @property
    def freq_index(self) -> int:
        """Índice da frequência a exibir."""
        return self._freq_index

    @freq_index.setter
    def freq_index(self, value: int) -> None:
        hi = (self.dims[4] - 1) if self.dims else 0
        self._set("_freq_index", int(np.clip(int(value), 0, hi)))

    @property
    def page(self) -> int:
        """Página atual da galeria (0-based)."""
        return self._page

    @page.setter
    def page(self, value: int) -> None:
        hi = max(0, self.n_pages - 1)
        self._set("_page", int(np.clip(int(value), 0, hi)))

    @property
    def focus_model(self) -> int:
        """Modelo em foco (animation bar — varre o ensemble em single-model playback)."""
        return self._focus_model

    @focus_model.setter
    def focus_model(self, value: int) -> None:
        hi = max(0, self.n_models - 1)
        self._set("_focus_model", int(np.clip(int(value), 0, hi)))

    @property
    def plot_backend(self) -> PlotBackend:
        """Backend de plot da galeria (matplotlib/pyqtgraph/…). A View recria o canvas."""
        return self._plot_backend

    @plot_backend.setter
    def plot_backend(self, value: PlotBackend | str) -> None:
        # ``PlotBackend(value)`` aceita o enum OU sua str (".value") — útil ao
        # restaurar de um ``.session`` (onde vira string JSON).
        self._set("_plot_backend", PlotBackend(value))
