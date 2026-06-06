# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/viewmodel.py                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimulationViewModel — estado/lógica da perspectiva (PURO)  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0011c)         ║
# ║  Versão      : v0.3                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — geração estocástica (Fatia 3)                    ║
# ║  Framework   : Python PURO — NÃO importa Qt (Princípio X; testável s/ qt)  ║
# ║  Dependências: gui.viewmodels, gui.services.{sim_request,stochastic_geology}║
# ║  Padrão      : ViewModel (MVVM) — Service INJETADO (duck-typed)            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Guarda o estado de UI da Simulação MULTI-CONFIG (listas de              ║
# ║    frequências/dips/espaçamentos-TR + geometria h1/tj/p_med + nº modelos   ║
# ║    + params de GEOLOGIA estocástica + status), valida os parâmetros POR     ║
# ║    ELEMENTO (+ errata da geologia), deriva ``n_pos`` pela convenção Fortran ║
# ║    e dispara a simulação delegando a um SERVICE injetado. Emite ``changed`` ║
# ║    (mudança de estado) e ``result_ready`` (resultado) — a View se liga.    ║
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

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from apps.sim_manager.perspectives.simulation.results_viewmodel import ResultsViewModel
from geosteering_ai.gui.services.sim_request import SimRequest, compute_n_pos
from geosteering_ai.gui.services.stochastic_geology import GENERATORS_AVAILABLE
from geosteering_ai.gui.viewmodels.base import BaseViewModel
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["SimulationViewModel"]

# Geologia estocástica (Fatia 3) — validação espelha ``GenConfig.validate`` do
# gerador puro (gui.services.stochastic_geology), evitando enviar uma config que
# o gerador recusaria. ``_RHO_DISTRIBUTIONS``/``GENERATORS_AVAILABLE`` casam as
# opções do combo da View. ``_LAMBDA_MIN`` = 1.0 (TIV física: ρᵥ ≥ ρₕ).
_RHO_DISTRIBUTIONS = ("loguni", "uniform")
_LAMBDA_MIN_PHYS = 1.0
_N_LAYERS_MIN_PHYS = 3  # inclui 2 semi-espaços

# Backends de simulação (spec 0012). "numba" roda in-thread; "jax"/"auto" rodam
# num subprocesso (TLS-safe — JAX numa QThread crasha). Default "numba" (rápido,
# sem spawn; sem regressão). O usuário opta por jax/auto p/ GPU.
_BACKENDS = ("numba", "jax", "auto")

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
        # ── Fatia 3 — geologia estocástica ──
        "_geology_mode",
        "_n_layers_fixed",
        "_n_layers_min",
        "_n_layers_max",
        "_rho_h_min",
        "_rho_h_max",
        "_rho_h_distribution",
        "_anisotropic",
        "_lambda_min",
        "_lambda_max",
        "_min_thickness",
        "_generator",
        "_rng_seed",
        "_backend",
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
        # ── Fatia 3 — geologia estocástica (defaults p/ um ensemble TIV brando) ──
        # Default "stochastic": é o propósito do app (modelos diversos). O modo
        # "fixed" (geologia determinística da Fatia 2) fica disponível p/ smoke.
        self._geology_mode: str = "stochastic"
        self._n_layers_fixed: Optional[int] = 5  # None ⇒ amostra [min, max)
        self._n_layers_min: int = 3
        self._n_layers_max: int = 11
        self._rho_h_min: float = 1.0
        self._rho_h_max: float = 1000.0
        self._rho_h_distribution: str = "loguni"
        self._anisotropic: bool = True
        self._lambda_min: float = 1.0
        self._lambda_max: float = 2.0**0.5  # √2 — anisotropia TIV branda
        self._min_thickness: float = 1.0
        self._generator: str = "sobol"
        # ALEATÓRIO por default (None): paridade com o monólito (v2.19) e KB-018
        # — cada execução gera um ensemble TIV distinto. Reprodutibilidade é
        # opt-in (o usuário desmarca "Semente aleatória" e fixa um inteiro).
        self._rng_seed: Optional[int] = None
        # Backend de simulação (spec 0012). Default "numba" (in-thread, rápido, sem
        # spawn); "jax"/"auto" rodam em subprocesso (GPU, TLS-safe).
        self._backend: str = "numba"
        self._status: str = "idle"
        self._last_result: Optional[Dict[str, Any]] = None
        self.result_ready: VMSignal = VMSignal()
        # ── Fatia 6a — feedback de execução (progresso/timing/log) ───────────
        self._progress_done: int = 0
        self._progress_total: int = 0
        self._sim_start_time: float = 0.0
        self._sim_elapsed_s: float = 0.0
        self._sim_throughput: float = 0.0  # modelos/s
        # Log textual (string por linha) — a perspectiva liga à secondary sidebar.
        self.log_entry: VMSignal = VMSignal()
        # Sub-VM PURO da galeria de resultados (Fatia 4) — alimentado ao concluir.
        self.results: ResultsViewModel = ResultsViewModel()
        # Liga-se aos VMSignals do service (pub/sub puro — sem Qt aqui).
        service.finished.connect(self._on_sim_finished)
        service.error.connect(self._on_sim_error)
        # Progresso é opcional no contrato do service (stubs de teste podem não tê-lo).
        if hasattr(service, "progress"):
            service.progress.connect(self._on_progress)

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

    # ── Properties de geologia estocástica (Fatia 3) ─────────────────────────
    @property
    def geology_mode(self) -> str:
        """``"fixed"`` (3-camadas determinístico) ou ``"stochastic"`` (N modelos TIV)."""
        return self._geology_mode

    @geology_mode.setter
    def geology_mode(self, value: str) -> None:
        self._set("_geology_mode", str(value))

    @property
    def n_layers_fixed(self) -> Optional[int]:
        """Nº de camadas fixo (≥3) ou ``None`` (amostra [min, max))."""
        return self._n_layers_fixed

    @n_layers_fixed.setter
    def n_layers_fixed(self, value: Optional[int]) -> None:
        self._set("_n_layers_fixed", None if value is None else int(value))

    @property
    def n_layers_min(self) -> int:
        """Limite inferior de camadas amostradas (inclusive)."""
        return self._n_layers_min

    @n_layers_min.setter
    def n_layers_min(self, value: int) -> None:
        self._set("_n_layers_min", int(value))

    @property
    def n_layers_max(self) -> int:
        """Limite superior de camadas amostradas (exclusive)."""
        return self._n_layers_max

    @n_layers_max.setter
    def n_layers_max(self, value: int) -> None:
        self._set("_n_layers_max", int(value))

    @property
    def rho_h_min(self) -> float:
        """ρₕ mínimo (Ω·m) da geração estocástica."""
        return self._rho_h_min

    @rho_h_min.setter
    def rho_h_min(self, value: float) -> None:
        self._set("_rho_h_min", float(value))

    @property
    def rho_h_max(self) -> float:
        """ρₕ máximo (Ω·m) da geração estocástica."""
        return self._rho_h_max

    @rho_h_max.setter
    def rho_h_max(self, value: float) -> None:
        self._set("_rho_h_max", float(value))

    @property
    def rho_h_distribution(self) -> str:
        """``"loguni"`` (log-uniforme) ou ``"uniform"``."""
        return self._rho_h_distribution

    @rho_h_distribution.setter
    def rho_h_distribution(self, value: str) -> None:
        self._set("_rho_h_distribution", str(value))

    @property
    def anisotropic(self) -> bool:
        """Se ``True``, ρᵥ=λ²·ρₕ (λ∈[min,max]); senão ρᵥ=ρₕ (isotrópico)."""
        return self._anisotropic

    @anisotropic.setter
    def anisotropic(self, value: bool) -> None:
        self._set("_anisotropic", bool(value))

    @property
    def lambda_min(self) -> float:
        """Fator de anisotropia λ mínimo (≥1)."""
        return self._lambda_min

    @lambda_min.setter
    def lambda_min(self, value: float) -> None:
        self._set("_lambda_min", float(value))

    @property
    def lambda_max(self) -> float:
        """Fator de anisotropia λ máximo."""
        return self._lambda_max

    @lambda_max.setter
    def lambda_max(self, value: float) -> None:
        self._set("_lambda_max", float(value))

    @property
    def min_thickness(self) -> float:
        """Piso de espessura por camada (m)."""
        return self._min_thickness

    @min_thickness.setter
    def min_thickness(self, value: float) -> None:
        self._set("_min_thickness", float(value))

    @property
    def generator(self) -> str:
        """Gerador estocástico (∈ ``GENERATORS_AVAILABLE``)."""
        return self._generator

    @generator.setter
    def generator(self, value: str) -> None:
        self._set("_generator", str(value))

    @property
    def rng_seed(self) -> Optional[int]:
        """Semente reprodutível (int) ou ``None`` (aleatória a cada run)."""
        return self._rng_seed

    @rng_seed.setter
    def rng_seed(self, value: Optional[int]) -> None:
        self._set("_rng_seed", None if value is None else int(value))

    @property
    def backend(self) -> str:
        """Backend de simulação: ``numba`` (in-thread) | ``jax`` | ``auto`` (subprocesso)."""
        return self._backend

    @backend.setter
    def backend(self, value: str) -> None:
        self._set("_backend", str(value))

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
        if self._backend not in _BACKENDS:
            errors.append(
                f"Backend {self._backend!r} inválido (use {list(_BACKENDS)})."
            )
        # ── Guardrail de recurso: n_pos derivado não pode explodir (OOM) ─────
        # self.n_pos já guarda dips-vazio/p_med≤0/tj≤0 (retorna 0) → só dispara
        # o teto quando a geometria é válida mas degenerada (dip≈90°).
        n_pos = self.n_pos
        if n_pos > _N_POS_MAX:
            errors.append(
                f"Geometria degenerada: n_pos derivado ({n_pos}) > {_N_POS_MAX} "
                f"(dip≈90° e/ou p_med ínfimo vs tj). Reduza o dip ou aumente p_med."
            )
        # ── Geologia estocástica (só quando o modo a usa) ────────────────────
        if self._geology_mode == "stochastic":
            errors.extend(self._validate_geology())
        return errors

    def _validate_geology(self) -> List[str]:
        """Valida os params de geologia estocástica (espelha ``GenConfig.validate``).

        Evita despachar uma config que o gerador puro recusaria (fail-fast no VM).

        Returns:
            Lista de mensagens de erro (PT-BR); vazia se OK.
        """
        errors: List[str] = []
        if not (self._rho_h_min > 0.0 and self._rho_h_max > self._rho_h_min):
            errors.append(
                f"ρₕ inválido: precisa 0 < ρ_min ({self._rho_h_min:g}) < "
                f"ρ_max ({self._rho_h_max:g}) Ω·m."
            )
        if self._rho_h_distribution not in _RHO_DISTRIBUTIONS:
            errors.append(
                f"Distribuição de ρₕ {self._rho_h_distribution!r} inválida "
                f"(use {list(_RHO_DISTRIBUTIONS)})."
            )
        if self._anisotropic:
            if self._lambda_min < _LAMBDA_MIN_PHYS:
                errors.append(
                    f"λ mínimo ({self._lambda_min:g}) deve ser ≥ {_LAMBDA_MIN_PHYS:g} "
                    f"(TIV física: ρᵥ ≥ ρₕ)."
                )
            if self._lambda_max < self._lambda_min:
                errors.append(
                    f"λ máximo ({self._lambda_max:g}) deve ser ≥ λ mínimo "
                    f"({self._lambda_min:g})."
                )
        if self._generator not in GENERATORS_AVAILABLE:
            errors.append(
                f"Gerador {self._generator!r} desconhecido (use {GENERATORS_AVAILABLE})."
            )
        if self._min_thickness <= 0.0:
            errors.append(f"min_thickness ({self._min_thickness:g} m) deve ser > 0.")
        # n_layers: fixo (≥3) OU range válido [min, max) com min ≥ 3.
        if self._n_layers_fixed is not None:
            if self._n_layers_fixed < _N_LAYERS_MIN_PHYS:
                errors.append(
                    f"n_layers fixo ({self._n_layers_fixed}) deve ser ≥ "
                    f"{_N_LAYERS_MIN_PHYS} (inclui 2 semi-espaços)."
                )
        else:
            if self._n_layers_min < _N_LAYERS_MIN_PHYS:
                errors.append(
                    f"n_layers mínimo ({self._n_layers_min}) deve ser ≥ "
                    f"{_N_LAYERS_MIN_PHYS}."
                )
            if self._n_layers_max <= self._n_layers_min:
                errors.append(
                    f"n_layers máximo ({self._n_layers_max}) deve ser > mínimo "
                    f"({self._n_layers_min})."
                )
        return errors

    def run(self) -> None:
        """Valida e dispara a simulação (não-bloqueante) via o service injetado.

        Em erro de validação NÃO chama o service: status → ``"error"`` e
        ``last_result = {"errors": [...]}``.

        Guard de re-entrância: se já há uma simulação em voo (``status ==
        "running"``), retorna sem despachar — o invariante "1 simulação por vez"
        vive AQUI (camada VM), não num efeito colateral de ``setEnabled`` na View
        (que depende do filtro de eventos do Qt e não cobre chamadas programáticas).
        """
        if self._status == "running":
            return
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
            backend=self._backend,
            geology_mode=self._geology_mode,
            n_layers_min=self._n_layers_min,
            n_layers_max=self._n_layers_max,
            n_layers_fixed=self._n_layers_fixed,
            rho_h_min=self._rho_h_min,
            rho_h_max=self._rho_h_max,
            rho_h_distribution=self._rho_h_distribution,
            anisotropic=self._anisotropic,
            lambda_min=self._lambda_min,
            lambda_max=self._lambda_max,
            min_thickness=self._min_thickness,
            generator=self._generator,
            rng_seed=self._rng_seed,
        )
        # Reseta o feedback de execução (Fatia 6a) ANTES de iniciar.
        self._progress_done = 0
        self._progress_total = max(1, int(self._n_models))
        self._sim_start_time = time.time()
        self._sim_elapsed_s = 0.0
        self._sim_throughput = 0.0
        self.log_entry.emit(
            f"▶ Iniciando: {self._n_models} modelo(s) · backend={self._backend} · "
            f"geologia={self._geology_mode}"
        )
        self._set("_status", "running")
        self._service.run(request)

    # ── Controle de execução (Fatia 6a) ─────────────────────────────────────
    def request_cancel(self) -> None:
        """Solicita cancelamento cooperativo (só enquanto ``running``)."""
        if self._status != "running":
            return
        if hasattr(self._service, "request_cancel"):
            self._service.request_cancel()
        self.log_entry.emit("⏹ Cancelamento solicitado…")

    def request_pause(self) -> None:
        """Pausa cooperativa (numba). No-op se não houver suporte no service."""
        if self._status != "running":
            return
        if hasattr(self._service, "request_pause"):
            self._service.request_pause()
        self.log_entry.emit("⏸ Pausado.")

    def request_resume(self) -> None:
        """Retoma de uma pausa cooperativa."""
        if self._status != "running":
            return
        if hasattr(self._service, "request_resume"):
            self._service.request_resume()
        self.log_entry.emit("▶ Retomado.")

    @property
    def progress_done(self) -> int:
        """Nº de modelos concluídos na simulação em voo (Fatia 6a)."""
        return self._progress_done

    @property
    def progress_total(self) -> int:
        """Nº total de modelos da simulação em voo (≥1)."""
        return self._progress_total

    @property
    def status_display(self) -> Dict[str, str]:
        """Snapshot ATÔMICO p/ a status bar (estado/elapsed/throughput) — PURO.

        Calculado num único método (evita race entre properties lidas em momentos
        diferentes pela View). Strings prontas para exibição.
        """
        labels = {
            "idle": "● ocioso",
            "running": "◉ executando",
            "done": "✓ concluído",
            "error": "✕ erro",
            "cancelled": "⏹ cancelado",
        }
        elapsed = "—"
        throughput = "—"
        if self._status in ("running", "done"):
            elapsed = f"{self._sim_elapsed_s:.1f} s"
            if self._sim_throughput > 0.0:
                throughput = f"{self._sim_throughput:.0f} mod/s"
        return {
            "state": labels.get(self._status, self._status),
            "elapsed": elapsed,
            "throughput": throughput,
        }

    # ── Callbacks do service (rodam na MAIN thread) ──────────────────────────
    def _on_progress(self, done: int, total: int) -> None:
        """Atualiza progresso + timing/throughput (Fatia 6a). Roda na MAIN thread."""
        self._progress_done = int(done)
        self._progress_total = max(1, int(total))
        self._sim_elapsed_s = max(0.0, time.time() - self._sim_start_time)
        if self._sim_elapsed_s > 0.0:
            self._sim_throughput = self._progress_done / self._sim_elapsed_s
        # Sinaliza a View (chave "_progress" — não usa _set p/ não exigir membership).
        self.changed.emit("_progress", self._progress_done)

    def _on_sim_finished(self, result: Dict[str, Any]) -> None:
        # Resultado de cancelamento (Fatia 6a): NÃO alimenta a galeria nem result_ready.
        if result.get("cancelled"):
            self._last_result = result
            self.log_entry.emit("⏹ Simulação cancelada (resultado parcial descartado).")
            self._set("_status", "cancelled")
            return
        self._sim_elapsed_s = max(0.0, time.time() - self._sim_start_time)
        n = max(1, int(self._n_models))
        if self._sim_elapsed_s > 0.0:
            self._sim_throughput = n / self._sim_elapsed_s
        self._progress_done = self._progress_total  # garante 100% no fim
        self._last_result = result
        self.results.set_result(result)  # alimenta a galeria (Fatia 4)
        self.log_entry.emit(
            f"✓ Concluído: {n} modelo(s) em {self._sim_elapsed_s:.1f} s "
            f"({self._sim_throughput:.0f} mod/s)"
        )
        self._set("_status", "done")
        self.result_ready.emit(result)

    def _on_sim_error(self, message: str) -> None:
        self._last_result = {"error": message}
        self.log_entry.emit(f"✕ Erro: {message}")
        self._set("_status", "error")

    # ── Persistência .session (params; resultado reproduzível pela seed) ─────
    def to_session_dict(self) -> Dict[str, Any]:
        """Serializa os PARAMS atuais (JSON-serializável) p/ ``.session``.

        NÃO inclui o H6 (grande/complexo) — a simulação é reproduzível pela seed.

        Returns:
            dict com listas/escalares (tuplas viram listas).
        """
        return {
            "frequencies": list(self._frequencies),
            "dips": list(self._dips),
            "tr_spacings": list(self._tr_spacings),
            "h1": self._h1,
            "tj": self._tj,
            "p_med": self._p_med,
            "n_models": self._n_models,
            "geology_mode": self._geology_mode,
            "n_layers_fixed": self._n_layers_fixed,
            "n_layers_min": self._n_layers_min,
            "n_layers_max": self._n_layers_max,
            "rho_h_min": self._rho_h_min,
            "rho_h_max": self._rho_h_max,
            "rho_h_distribution": self._rho_h_distribution,
            "anisotropic": self._anisotropic,
            "lambda_min": self._lambda_min,
            "lambda_max": self._lambda_max,
            "min_thickness": self._min_thickness,
            "generator": self._generator,
            "rng_seed": self._rng_seed,
            "backend": self._backend,
            # Preferência da galeria (str do enum) — reproduzível.
            "plot_backend": self.results.plot_backend.value,
        }

    def load_session_dict(self, data: Dict[str, Any]) -> None:
        """Repopula os params a partir de um dict de ``.session`` (via setters).

        Usa os setters (coerção + ``changed`` → a View sincroniza). Chaves
        ausentes são ignoradas (forward/backward-compat).

        Args:
            data: dict de :meth:`to_session_dict` (ou um ``.session`` carregado).
        """
        if "frequencies" in data:
            self.frequencies = data["frequencies"]
        if "dips" in data:
            self.dips = data["dips"]
        if "tr_spacings" in data:
            self.tr_spacings = data["tr_spacings"]
        for key in (
            "h1",
            "tj",
            "p_med",
            "n_models",
            "geology_mode",
            "n_layers_fixed",
            "n_layers_min",
            "n_layers_max",
            "rho_h_min",
            "rho_h_max",
            "rho_h_distribution",
            "anisotropic",
            "lambda_min",
            "lambda_max",
            "min_thickness",
            "generator",
            "rng_seed",
            "backend",
        ):
            if key in data:
                setattr(self, key, data[key])
        # ── Saneia o backend vindo do .session ────────────────────────────────
        # Um .session corrompido/editado à mão pode trazer um backend inválido
        # (ex.: "cuda"). O combo da View só tem {numba,jax,auto} → setCurrentText
        # ignoraria silenciosamente (combo dessincronizado do VM) e o erro só
        # apareceria no run(). Aqui caímos p/ "numba" no load (via setter → emite
        # changed → o combo sincroniza), em vez de carregar um estado inválido.
        if self._backend not in _BACKENDS:
            self.backend = "numba"
        # Preferência da galeria (sub-VM results) — aceita str do enum.
        if "plot_backend" in data:
            self.results.plot_backend = data["plot_backend"]
