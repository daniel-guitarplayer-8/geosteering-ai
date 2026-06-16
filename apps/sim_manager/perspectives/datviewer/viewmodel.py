# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/datviewer/viewmodel.py                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : DatViewerViewModel — estado PURO do visualizador .dat      ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app — perspectiva Visualizador .dat (Fatia 6h)           ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: gui.viewmodels.base (BaseViewModel), .service              ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    ViewModel PURO (Princípio X — NÃO importa Qt) do visualizador .dat:     ║
# ║    orquestra a leitura (via DatViewerService injetado) e expõe o último     ║
# ║    resultado + um resumo textual. Sinais: ``loaded`` (sucesso) e            ║
# ║    ``load_error`` (falha). SOMENTE LEITURA — nunca recalcula física.        ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    DatViewerViewModel                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``DatViewerViewModel`` — estado PURO do visualizador ``.dat`` (Fatia 6h)."""

from __future__ import annotations

from typing import Any, Optional

from apps.sim_manager.perspectives.datviewer.service import DatLoadResult
from geosteering_ai.gui.viewmodels.base import BaseViewModel
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["DatViewerViewModel"]


class DatViewerViewModel(BaseViewModel):
    """ViewModel PURO do visualizador ``.dat`` (carrega + expõe + resume).

    Attributes:
        loaded: ``VMSignal`` emit(result: DatLoadResult) em leitura bem-sucedida.
        load_error: ``VMSignal`` emit(msg: str) em falha de leitura.

    Example:
        >>> vm = DatViewerViewModel(service=DatViewerService())
        >>> vm.open("/tmp/sm_output.dat")     # doctest: +SKIP
        >>> vm.result.n_cols                   # doctest: +SKIP
        22

    Note:
        NÃO importa Qt. A View liga ``loaded``/``load_error`` a slots Qt. O serviço
        nunca levanta (degradação graciosa) → ``open`` sempre emite exatamente um
        dos dois sinais.
    """

    _STATE_FIELDS = ("_dat_path",)

    def __init__(self, service: Any) -> None:
        """Inicializa com o serviço de leitura injetado.

        Args:
            service: objeto com ``load(path) -> DatLoadResult`` (duck-typed;
                tipicamente :class:`DatViewerService`).
        """
        super().__init__()
        self._service = service
        self._dat_path: str = ""
        self._result: Optional[DatLoadResult] = None

        self.loaded: VMSignal = VMSignal()
        self.load_error: VMSignal = VMSignal()

    @property
    def dat_path(self) -> str:
        """Caminho do último ``.dat`` para o qual ``open`` foi chamado."""
        return self._dat_path

    @property
    def result(self) -> Optional[DatLoadResult]:
        """Último :class:`DatLoadResult` (``None`` antes do 1º ``open``)."""
        return self._result

    def open(self, path: str) -> None:
        """Lê ``path`` via serviço e emite ``loaded`` (sucesso) ou ``load_error``.

        Args:
            path: caminho do ``.dat`` a carregar.
        """
        result = self._service.load(path)
        self._result = result
        self._set("_dat_path", result.dat_path)
        if result.error:
            self.load_error.emit(result.error)
        else:
            self.loaded.emit(result)

    def summary(self) -> str:
        """Resumo textual de uma linha do resultado atual (para o header da View)."""
        res = self._result
        if res is None:
            return "Nenhum arquivo carregado."
        if res.error:
            return f"Erro: {res.error}"
        out = "com .out" if res.out_metadata else "sem .out"
        return (
            f"{res.dat_path} · {res.n_rows} linhas × {res.n_cols} col · "
            f"{res.fmt} · {res.size_mb:.1f} MB · {out}"
        )
