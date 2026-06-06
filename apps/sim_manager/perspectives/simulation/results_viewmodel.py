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
from geosteering_ai.gui.viewmodels.base import BaseViewModel
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["ResultsViewModel", "COMPONENT_NAMES", "PLOT_KINDS"]

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

# Modos de plot (transform aplicado à componente complexa selecionada).
PLOT_KINDS: Tuple[str, ...] = ("re", "im", "mag", "phase")


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
        "_component_index",
        "_plot_kind",
        "_tr_index",
        "_dip_index",
        "_freq_index",
        "_page",
    )

    def __init__(
        self,
        page_size: int = _DEFAULT_PAGE_SIZE,
        cache: Optional[LRUPlotCache] = None,
        plot_backend: PlotBackend = PlotBackend.MATPLOTLIB,
    ) -> None:
        """Inicializa sem resultado (galeria vazia).

        Args:
            page_size: nº de modelos por página da galeria (default 12).
            cache: LRU de curvas (default ``LRUPlotCache(maxlen=256)``) — bounded
                por contagem; injetável p/ teste.
            plot_backend: backend de plot inicial (default ``MATPLOTLIB``); a View
                recria o canvas ao trocar.
        """
        super().__init__()
        self._result: Optional[Dict[str, Any]] = None
        self._h6: Optional[np.ndarray] = None
        self._positions_z: Optional[np.ndarray] = None
        self._component_index: int = 0
        self._plot_kind: str = "re"
        self._tr_index: int = 0
        self._dip_index: int = 0
        self._freq_index: int = 0
        self._page: int = 0
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
            _n_models, n_tr, n_ang, _n_pos, n_f = dims
            self._component_index = int(np.clip(self._component_index, 0, 8))
            self._tr_index = int(np.clip(self._tr_index, 0, n_tr - 1))
            self._dip_index = int(np.clip(self._dip_index, 0, n_ang - 1))
            self._freq_index = int(np.clip(self._freq_index, 0, n_f - 1))
            self._page = int(np.clip(self._page, 0, max(0, self.n_pages - 1)))
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
    def component_name(self) -> str:
        """Nome da componente selecionada (Hxx..Hzz)."""
        return COMPONENT_NAMES[int(np.clip(self._component_index, 0, 8))]

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
        """Curva ``(n_pos,)`` do modelo na config/componente/plot-kind atuais.

        ``transform(H6[m, tr_index, dip_index, :, freq_index, component_index])``
        com ``transform`` = Re/Im/|·|/fase(°). Memorizada num LRU (mesma chave =
        mesmo array, vindo do cache).

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
            f"{model_index}|{self._component_index}|{self._plot_kind}"
            f"|{self._tr_index}|{self._dip_index}|{self._freq_index}"
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached["curve"]  # type: ignore[no-any-return]
        h = self._h6[
            model_index,
            self._tr_index,
            self._dip_index,
            :,
            self._freq_index,
            self._component_index,
        ]
        curve = _kind_transform(self._plot_kind, h)
        self._cache.put(key, {"curve": curve})
        return curve

    # ── Properties de seletor (setters clampam + emitem ``changed``) ─────────
    @property
    def component_index(self) -> int:
        """Índice da componente (0=Hxx … 8=Hzz)."""
        return self._component_index

    @component_index.setter
    def component_index(self, value: int) -> None:
        self._set("_component_index", int(np.clip(int(value), 0, 8)))

    @property
    def plot_kind(self) -> str:
        """Modo de plot: re | im | mag | phase."""
        return self._plot_kind

    @plot_kind.setter
    def plot_kind(self, value: str) -> None:
        self._set("_plot_kind", value if value in PLOT_KINDS else "re")

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
    def plot_backend(self) -> PlotBackend:
        """Backend de plot da galeria (matplotlib/pyqtgraph/…). A View recria o canvas."""
        return self._plot_backend

    @plot_backend.setter
    def plot_backend(self, value: PlotBackend | str) -> None:
        # ``PlotBackend(value)`` aceita o enum OU sua str (".value") — útil ao
        # restaurar de um ``.session`` (onde vira string JSON).
        self._set("_plot_backend", PlotBackend(value))
