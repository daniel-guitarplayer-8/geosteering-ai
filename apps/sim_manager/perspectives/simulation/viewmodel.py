# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/viewmodel.py                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimulationViewModel — estado/lógica da perspectiva (PURO)  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0011b)         ║
# ║  Versão      : v0.2                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — params completos (Fatia 2)                       ║
# ║  Framework   : Python PURO — NÃO importa Qt (Princípio X; testável s/ qt)  ║
# ║  Dependências: gui.viewmodels (BaseViewModel/VMSignal), gui.services.sim_request║
# ║  Padrão      : ViewModel (MVVM) — Service INJETADO (duck-typed)            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Guarda o estado de UI da Simulação MULTI-CONFIG (listas de              ║
# ║    frequências/dips/espaçamentos-TR + geometria h1/tj/p_med + nº modelos   ║
# ║    + status), valida os parâmetros POR ELEMENTO, deriva ``n_pos`` pela      ║
# ║    convenção Fortran e dispara a simulação delegando a um SERVICE injetado. ║
# ║    Emite ``changed`` (mudança de estado) e ``result_ready`` (resultado      ║
# ║    pronto) — a View se liga a esses sinais.                                ║
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

from typing import Any, Dict, List, Optional, Sequence, Tuple

from geosteering_ai.gui.services.sim_request import SimRequest, compute_n_pos
from geosteering_ai.gui.viewmodels.base import BaseViewModel
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["SimulationViewModel"]

# Errata física — validação fail-fast no VM. ESPELHA O SIMULADOR (não a errata
# do pipeline DL): a rota deste app é simulate_batch, cujo gate efetivo é
# ``_validate_multi_inputs`` + ``SimulationConfig`` — NÃO o ``PipelineConfig`` do
# DL (que é mais estreito: [100,1e6] Hz / [0.1,10] m). Mirror correto:
#   FREQ [10, 2e6] Hz   → simulation/config.py:158 (_FREQUENCY_HZ_RANGE).
#   DIP  [0, 90]°       → simulation/multi_forward.py:620 (paridade Fortran;
#                         dip>90° é REJEITADO). JAX limita a 89°.
#   TR   [0.1, 50] m    → simulation/multi_forward.py:617 (_validate_multi_inputs;
#                         máx 50 m cobre deep-reading PeriScope ~20 m, config.py:134).
# Validar POR ELEMENTO no VM evita enviar ao simulate_batch um valor que ele
# recusaria — e NÃO rejeitar um valor que ele aceitaria (assimetria nos 2 sentidos).
_FREQ_MIN_HZ, _FREQ_MAX_HZ = 10.0, 2.0e6
_DIP_MIN_DEG, _DIP_MAX_DEG = 0.0, 90.0
_TR_MIN_M, _TR_MAX_M = 0.1, 50.0

# Teto de robustez (NÃO é fidelidade — é guardrail de recurso): em geometria
# degenerada (dip→90° ⇒ cos→0, ou p_med ínfimo vs tj) o n_pos derivado explode
# (dip=90°, tj/p_med=10 ⇒ n_pos≈1e7 ⇒ tensor H6 de dezenas de GB ⇒ OOM). O
# simulate_batch/monólito NÃO barram isso; o VM o faz com erro claro ANTES de
# despachar. ``_compute_positions_z`` permanece byte-fiel à fórmula Fortran — o
# teto vive aqui (camada de aplicação), não no kernel/fórmula.
_N_POS_MAX = 100_000


class SimulationViewModel(BaseViewModel):
    """ViewModel da perspectiva Simulação (PURO; ``service`` injetado).

    Attributes:
        result_ready: ``VMSignal`` — emitido com o dict de resultado ao concluir.
        (herda ``changed: VMSignal`` de :class:`BaseViewModel`).

    Example:
        >>> vm = SimulationViewModel(service=SimulationService())
        >>> vm.changed.connect(view._on_changed)
        >>> vm.result_ready.connect(view._on_result)
        >>> vm.frequencies = (20000.0, 40000.0)   # multi-config
        >>> vm.dips = (0.0, 30.0)
        >>> vm.n_pos                               # derivado (read-only) p/ exibir
        >>> vm.run()    # valida → service.run(SimRequest); resultado via result_ready

    Note:
        ``status`` ∈ {``"idle"``, ``"running"``, ``"done"``, ``"error"``}. Em erro
        de validação, ``run`` não chama o service (status → ``"error"``,
        ``last_result["errors"]``).
    """

    _STATE_FIELDS = (
        "_frequencies",
        "_dips",
        "_tr_spacings",
        "_h1",
        "_tj",
        "_p_med",
        "_n_models",
    )

    def __init__(self, service: Any) -> None:
        """Inicializa com um ``service`` injetado (duck-typed).

        Args:
            service: objeto com ``run(request)`` e VMSignals ``finished``/``error``
                (ex.: :class:`SimulationService` real, ou um stub em teste).
        """
        super().__init__()
        self._service = service
        # Listas multi-config (defaults dentro da errata; 1 valor cada).
        self._frequencies: Tuple[float, ...] = (20000.0,)
        self._dips: Tuple[float, ...] = (0.0,)
        self._tr_spacings: Tuple[float, ...] = (1.0,)
        # Geometria Fortran (positions_z = linspace(-h1, tj-h1, n_pos)).
        self._h1: float = 1.0
        self._tj: float = 10.0
        self._p_med: float = 1.0
        self._n_models: int = 2
        self._status: str = "idle"
        self._last_result: Optional[Dict[str, Any]] = None
        self.result_ready: VMSignal = VMSignal()
        # Liga-se aos VMSignals do service (pub/sub puro — sem Qt aqui).
        service.finished.connect(self._on_sim_finished)
        service.error.connect(self._on_sim_error)

    # ── Properties multi-valor (setters emitem ``changed`` via _set) ─────────
    @property
    def frequencies(self) -> Tuple[float, ...]:
        """Frequências (Hz) — lista multi-config."""
        return self._frequencies

    @frequencies.setter
    def frequencies(self, value: Sequence[float]) -> None:
        self._set("_frequencies", tuple(float(v) for v in value))

    @property
    def dips(self) -> Tuple[float, ...]:
        """Ângulos de mergulho (graus) — lista multi-config."""
        return self._dips

    @dips.setter
    def dips(self, value: Sequence[float]) -> None:
        self._set("_dips", tuple(float(v) for v in value))

    @property
    def tr_spacings(self) -> Tuple[float, ...]:
        """Espaçamentos transmissor-receptor (m) — lista multi-config."""
        return self._tr_spacings

    @tr_spacings.setter
    def tr_spacings(self, value: Sequence[float]) -> None:
        self._set("_tr_spacings", tuple(float(v) for v in value))

    @property
    def h1(self) -> float:
        """Altura do 1º ponto-médio acima da interface (m) — convenção Fortran."""
        return self._h1

    @h1.setter
    def h1(self, value: float) -> None:
        self._set("_h1", float(value))

    @property
    def tj(self) -> float:
        """Janela de investigação (m) — extensão da varredura de profundidade."""
        return self._tj

    @tj.setter
    def tj(self, value: float) -> None:
        self._set("_tj", float(value))

    @property
    def p_med(self) -> float:
        """Passo entre medidas (m)."""
        return self._p_med

    @p_med.setter
    def p_med(self, value: float) -> None:
        self._set("_p_med", float(value))

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

    @property
    def n_pos(self) -> int:
        """Nº de posições de medição DERIVADO (read-only, p/ exibir na View).

        Espelha a convenção Fortran (:func:`compute_n_pos`, fonte única). Retorna
        ``0`` se ainda não há dip ou ``p_med`` inválido (≤0) — apenas para exibição;
        a validação real é em :meth:`validate`.
        """
        if not self._dips or self._p_med <= 0.0 or self._tj <= 0.0:
            return 0
        return compute_n_pos(self._tj, self._p_med, self._dips[0])

    # ── Validação + execução ─────────────────────────────────────────────────
    def validate(self) -> List[str]:
        """Valida os parâmetros (errata do SIMULADOR, POR ELEMENTO). Lista vazia = OK.

        Checa: listas não-vazias; cada freq∈[10,2e6] Hz, cada dip∈[0,90]°,
        cada TR∈[0.1,50] m; ``h1``/``tj``/``p_med`` > 0; ``n_models`` ≥ 1; e o
        ``n_pos`` derivado ≤ ``_N_POS_MAX`` (rejeita geometria degenerada que
        causaria OOM — ver constante).

        Returns:
            Lista de mensagens de erro (PT-BR); vazia se todos os params válidos.
        """
        errors: List[str] = []
        # ── Listas não-vazias (uma lista vazia geraria batch sem eixo) ───────
        if not self._frequencies:
            errors.append("Informe ao menos uma frequência.")
        if not self._dips:
            errors.append("Informe ao menos um dip.")
        if not self._tr_spacings:
            errors.append("Informe ao menos um espaçamento TR.")
        # ── Errata POR ELEMENTO ──────────────────────────────────────────────
        for f in self._frequencies:
            if not (_FREQ_MIN_HZ <= f <= _FREQ_MAX_HZ):
                errors.append(
                    f"Frequência {f:g} Hz fora de [{_FREQ_MIN_HZ:g}, {_FREQ_MAX_HZ:g}] Hz."
                )
        for d in self._dips:
            if not (_DIP_MIN_DEG <= d <= _DIP_MAX_DEG):
                errors.append(
                    f"Dip {d:g}° fora de [{_DIP_MIN_DEG:g}, {_DIP_MAX_DEG:g}]°."
                )
        for tr in self._tr_spacings:
            if not (_TR_MIN_M <= tr <= _TR_MAX_M):
                errors.append(
                    f"Espaçamento TR {tr:g} m fora de [{_TR_MIN_M:g}, {_TR_MAX_M:g}] m."
                )
        # ── Geometria Fortran (> 0; positions_z exige p_med>0 e tj>0) ────────
        if self._h1 <= 0.0:
            errors.append(f"h1 ({self._h1:g} m) deve ser > 0.")
        if self._tj <= 0.0:
            errors.append(f"tj ({self._tj:g} m) deve ser > 0.")
        if self._p_med <= 0.0:
            errors.append(f"p_med ({self._p_med:g} m) deve ser > 0.")
        if self._n_models < 1:
            errors.append(f"nº de modelos ({self._n_models}) deve ser ≥ 1.")
        # ── Guardrail de recurso: n_pos derivado não pode explodir (OOM) ─────
        # self.n_pos já guarda dips-vazio/p_med≤0/tj≤0 (retorna 0) → só dispara
        # o teto quando a geometria é válida mas degenerada (dip≈90°).
        n_pos = self.n_pos
        if n_pos > _N_POS_MAX:
            errors.append(
                f"Geometria degenerada: n_pos derivado ({n_pos}) > {_N_POS_MAX} "
                f"(dip≈90° e/ou p_med ínfimo vs tj). Reduza o dip ou aumente p_med."
            )
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
            frequencies_hz=self._frequencies,
            tr_spacings_m=self._tr_spacings,
            dip_degs=self._dips,
            h1=self._h1,
            tj=self._tj,
            p_med=self._p_med,
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
