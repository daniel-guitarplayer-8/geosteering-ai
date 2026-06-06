# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/viewmodel.py                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimulationViewModel — estado/lógica da perspectiva (PURO)  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0011a)         ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — walking skeleton                                ║
# ║  Framework   : Python PURO — NÃO importa Qt (Princípio X; testável s/ qt)  ║
# ║  Dependências: gui.viewmodels (BaseViewModel/VMSignal), gui.services.sim_request║
# ║  Padrão      : ViewModel (MVVM) — Service INJETADO (duck-typed)            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Guarda o estado de UI da Simulação (frequência, dip, nº modelos,        ║
# ║    status), valida os parâmetros e dispara a simulação delegando a um      ║
# ║    SERVICE injetado. Emite ``changed`` (mudança de estado) e               ║
# ║    ``result_ready`` (resultado pronto) — a View se liga a esses sinais.   ║
# ║                                                                           ║
# ║  PUREZA (Princípio X)                                                     ║
# ║    NÃO importa Qt nem o ``SimulationService`` (Qt) — recebe um ``service`` ║
# ║    por INJEÇÃO (qualquer objeto com ``.run(req)`` + VMSignals ``finished`` ║
# ║    e ``error``). Em teste, injeta-se um stub puro. ``SimRequest`` vem de   ║
# ║    ``gui.services.sim_request`` (módulo PURO, sem Qt).                     ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimulationViewModel                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimulationViewModel`` — estado + validação + run (PURO, service injetado)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geosteering_ai.gui.services.sim_request import SimRequest
from geosteering_ai.gui.viewmodels.base import BaseViewModel
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["SimulationViewModel"]

# Errata física — validação fail-fast no VM (espelha o que o CORE aceita):
#   FREQ [100, 1e6] Hz  → api/schemas.py / PipelineConfig.
#   DIP  [0, 90]°       → simulation/multi_forward.py:622 (Numba, paridade Fortran;
#                         dip>90° é REJEITADO pelo simulador). JAX limita a 89°.
# Validar [0, 90] no VM evita aceitar um dip que o simulate_batch recusaria.
_FREQ_MIN_HZ, _FREQ_MAX_HZ = 100.0, 1.0e6
_DIP_MIN_DEG, _DIP_MAX_DEG = 0.0, 90.0


class SimulationViewModel(BaseViewModel):
    """ViewModel da perspectiva Simulação (PURO; ``service`` injetado).

    Attributes:
        result_ready: ``VMSignal`` — emitido com o dict de resultado ao concluir.
        (herda ``changed: VMSignal`` de :class:`BaseViewModel`).

    Example:
        >>> vm = SimulationViewModel(service=SimulationService())
        >>> vm.changed.connect(view._on_changed)
        >>> vm.result_ready.connect(view._on_result)
        >>> vm.frequency_hz = 20000.0
        >>> vm.run()    # valida → service.run(SimRequest); resultado via result_ready

    Note:
        ``status`` ∈ {``"idle"``, ``"running"``, ``"done"``, ``"error"``}. Em erro
        de validação, ``run`` não chama o service (status → ``"error"``,
        ``last_result["errors"]``).
    """

    _STATE_FIELDS = ("_frequency_hz", "_dip_deg", "_n_models")

    def __init__(self, service: Any) -> None:
        """Inicializa com um ``service`` injetado (duck-typed).

        Args:
            service: objeto com ``run(request)`` e VMSignals ``finished``/``error``
                (ex.: :class:`SimulationService` real, ou um stub em teste).
        """
        super().__init__()
        self._service = service
        self._frequency_hz: float = 20000.0
        self._dip_deg: float = 0.0
        self._n_models: int = 2
        self._status: str = "idle"
        self._last_result: Optional[Dict[str, Any]] = None
        self.result_ready: VMSignal = VMSignal()
        # Liga-se aos VMSignals do service (pub/sub puro — sem Qt aqui).
        service.finished.connect(self._on_sim_finished)
        service.error.connect(self._on_sim_error)

    # ── Properties de estado (setters emitem ``changed`` via _set) ───────────
    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @frequency_hz.setter
    def frequency_hz(self, value: float) -> None:
        self._set("_frequency_hz", float(value))

    @property
    def dip_deg(self) -> float:
        return self._dip_deg

    @dip_deg.setter
    def dip_deg(self, value: float) -> None:
        self._set("_dip_deg", float(value))

    @property
    def n_models(self) -> int:
        return self._n_models

    @n_models.setter
    def n_models(self, value: int) -> None:
        self._set("_n_models", int(value))

    @property
    def status(self) -> str:
        """Estado atual: idle | running | done | error (read-only)."""
        return self._status

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        """Último resultado (dict) ou ``None`` (read-only)."""
        return self._last_result

    # ── Validação + execução ─────────────────────────────────────────────────
    def validate(self) -> List[str]:
        """Valida os parâmetros (errata física). Lista vazia = OK.

        Returns:
            Lista de mensagens de erro (PT-BR); vazia se todos os params válidos.
        """
        errors: List[str] = []
        if not (_FREQ_MIN_HZ <= self._frequency_hz <= _FREQ_MAX_HZ):
            errors.append(
                f"Frequência {self._frequency_hz:g} Hz fora de "
                f"[{_FREQ_MIN_HZ:g}, {_FREQ_MAX_HZ:g}] Hz."
            )
        if not (_DIP_MIN_DEG <= self._dip_deg <= _DIP_MAX_DEG):
            errors.append(
                f"Dip {self._dip_deg:g}° fora de [{_DIP_MIN_DEG:g}, {_DIP_MAX_DEG:g}]°."
            )
        if self._n_models < 1:
            errors.append(f"nº de modelos ({self._n_models}) deve ser ≥ 1.")
        return errors

    def run(self) -> None:
        """Valida e dispara a simulação (não-bloqueante) via o service injetado.

        Em erro de validação NÃO chama o service: status → ``"error"`` e
        ``last_result = {"errors": [...]}``.
        """
        errors = self.validate()
        if errors:
            self._last_result = {"errors": errors}
            self._set("_status", "error")
            return
        request = SimRequest(
            frequencies_hz=(self._frequency_hz,),
            dip_degs=(self._dip_deg,),
            n_models=self._n_models,
            backend="numba",
        )
        self._set("_status", "running")
        self._service.run(request)

    # ── Callbacks do service (rodam na MAIN thread) ──────────────────────────
    def _on_sim_finished(self, result: Dict[str, Any]) -> None:
        self._last_result = result
        self._set("_status", "done")
        self.result_ready.emit(result)

    def _on_sim_error(self, message: str) -> None:
        self._last_result = {"error": message}
        self._set("_status", "error")
