# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/config.py                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Configuração (SimulationConfig)        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 1.2)                                   ║
# ║  Status      : Produção                                                  ║
# ║  Framework   : stdlib (dataclasses) + numpy + yaml (opcional)            ║
# ║  Dependências: PyYAML (lazy import para .to_yaml/.from_yaml)             ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Dataclass imutável que centraliza TODOS os parâmetros de execução    ║
# ║    do simulador Python (backend, dtype, filtro Hankel, frequência,      ║
# ║    geometria TR, features F5/F6/F7/F10). Validação de errata na        ║
# ║    `__post_init__` segue rigorosamente o padrão do `PipelineConfig`    ║
# ║    principal, com mensagens contextualizadas e ranges físicos          ║
# ║    justificados.                                                         ║
# ║                                                                           ║
# ║  DESIGN                                                                   ║
# ║    • `@dataclass(frozen=True)` — imutável por design. Qualquer          ║
# ║      mutação requer `dataclasses.replace(cfg, field=novo_valor)`,      ║
# ║      o que força validação renovada via `__post_init__`.               ║
# ║    • Validação via `assert` com mensagens explicitando valor recebido  ║
# ║      e range esperado (padrão PipelineConfig).                         ║
# ║    • Exclusividade mútua entre backends e dispositivos. Ex.: backend   ║
# ║      fortran_f2py é exclusivamente CPU; backend jax+gpu exige device   ║
# ║      'gpu' explícito.                                                    ║
# ║    • Presets via `@classmethod` (default, high_precision,              ║
# ║      production_gpu, realtime_cpu) como atalhos para combinações        ║
# ║      típicas de uso.                                                    ║
# ║    • YAML roundtrip via `to_yaml/from_yaml` (lazy import pyyaml).       ║
# ║                                                                           ║
# ║  ARQUITETURA                                                              ║
# ║    ┌────────────────────────────────────────────────────────────────┐  ║
# ║    │  YAML file                                                     │  ║
# ║    │    │                                                           │  ║
# ║    │    │  SimulationConfig.from_yaml(path)                         │  ║
# ║    │    ▼                                                           │  ║
# ║    │  SimulationConfig instance (frozen, validated)                │  ║
# ║    │    │                                                           │  ║
# ║    │    │  simulate(cfg)  [Fase 2+, pendente]                      │  ║
# ║    │    ▼                                                           │  ║
# ║    │  H_tensor (shape=(n_positions, 9) complex)                    │  ║
# ║    └────────────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  VALIDAÇÃO DE ERRATA (MIRROR DO PipelineConfig)                          ║
# ║    • frequency_hz ∈ [100.0, 1.0e6]    (LWD: 2 kHz–400 kHz comum)       ║
# ║    • tr_spacing_m ∈ [0.1, 10.0]        (geometria física realista)     ║
# ║    • n_positions ∈ [10, 100_000]       (mínimo útil a máximo prático) ║
# ║    • backend ∈ {fortran_f2py, numba, jax}                              ║
# ║    • dtype ∈ {complex128, complex64}                                   ║
# ║    • device ∈ {cpu, gpu}                                                ║
# ║    • hankel_filter ∈ catálogo de FilterLoader                          ║
# ║    • backend fortran_f2py ⟹ device cpu (Fortran não roda GPU)         ║
# ║    • frequencies_hz e tr_spacings_m se definidos DEVEM ter ≥ 1 item   ║
# ║                                                                           ║
# ║  COMPATIBILIDADE                                                          ║
# ║    Este config é **independente** de `PipelineConfig`. O simulador     ║
# ║    Python pode ser invocado diretamente para geração de datasets,      ║
# ║    debugging e testes unitários sem depender do pipeline de treino.   ║
# ║    A integração com `PipelineConfig` (Fase 6) será feita via           ║
# ║    adapter que converte `PipelineConfig → SimulationConfig`.          ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • geosteering_ai/config.py (PipelineConfig — padrão de referência)  ║
# ║    • docs/reference/plano_simulador_python_jax_numba.md (seção 5)      ║
# ║    • Fortran model.in v10.0 (estrutura equivalente)                    ║
# ║    • .claude/commands/geosteering-simulator-python.md (sub skill)     ║
# ║                                                                           ║
# ║  NOTAS DE IMPLEMENTAÇÃO                                                  ║
# ║    1. frozen=True + mutação via dataclasses.replace (re-valida).       ║
# ║    2. PyYAML é lazy-imported (pipeline mínimo não precisa de yaml).    ║
# ║    3. Optional[T] em vez de T | None (consistência com PipelineConfig).║
# ║    4. Mensagens de erro em PT-BR com acentuação correta.              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Configuração do simulador Python otimizado (Sprint 1.2).

Este módulo expõe a classe :class:`SimulationConfig`, um dataclass imutável
que agrega todos os parâmetros numéricos e de backend necessários para
rodar uma simulação forward EM 1D TIV no simulador Python. A instância
validada é consumida pelos backends Numba (Fase 2) e JAX (Fase 3).

Example:
    Uso direto com defaults::

        >>> from geosteering_ai.simulation import SimulationConfig
        >>> cfg = SimulationConfig()
        >>> cfg.frequency_hz
        20000.0
        >>> cfg.hankel_filter
        'werthmuller_201pt'

    Uso com preset de produção GPU::

        >>> cfg = SimulationConfig.production_gpu()
        >>> cfg.backend, cfg.device, cfg.dtype
        ('jax', 'gpu', 'complex64')

    Roundtrip YAML (requer PyYAML)::

        >>> cfg = SimulationConfig.default()
        >>> cfg.to_yaml("/tmp/sim_config.yaml")
        >>> cfg2 = SimulationConfig.from_yaml("/tmp/sim_config.yaml")
        >>> cfg == cfg2
        True

Note:
    Sprint 1.2 entrega apenas o dataclass e sua validação. A API
    `simulate(cfg)` (forward) será adicionada na Fase 2 (backend Numba)
    em `geosteering_ai/simulation/forward.py`.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE VALIDAÇÃO (errata expandida pós-revisão Sprint 2.1)
# ──────────────────────────────────────────────────────────────────────────────
# Ranges de validação para os parâmetros físicos. Os limites refletem a
# capacidade real do simulador Fortran (validada em revisão de 2026-04-11)
# após investigação dos comentários em `fifthBuildTIVModels.py` que
# indicam uso de dTR = [8.19, 20.43] m (ferramentas tipo ARC/PeriScope).
#
# Mudanças em relação à Sprint 1.2 (limites antigos):
#   • frequency_hz max: 1e6 → 2e6 (paridade com ARC/PeriScope 2 MHz)
#   • frequency_hz min: 100 → 10 (permite pesquisa em MT/CSAMT baixa freq)
#   • tr_spacing_m max: 10 → 50 (cobre deep-reading PeriScope 20 m + margem)
#   • tr_spacing_m min: 0.1 → 0.01 (ferramentas curtas/experimentais)
#   • novos: _RESISTIVITY_OHM_M_RANGE para ρh/ρv (alta resistividade)
#
# Justificativa física:
#   • Limite 2 MHz: aproximação quasi-estática (zeta = i·ω·μ, usada no
#     Fortran utils.f08 linhas 636, 971, 1066) permanece válida
#     (|ωε/σ| < 1%) até ~2 MHz mesmo em ρ = 10 000 Ω·m. Acima disso,
#     corrente de deslocamento torna-se não desprezível — fora do escopo.
#   • Limite 50 m: Werthmüller 201pt tem abscissas kr ∈ [8.7e-4, 93.7]
#     adequado para r ≤ ~30 m. Para r > 30 m, Anderson 801pt cobre até
#     ~1000 m. Setamos 50 m como limite conservador.
#   • Limite 10 Hz: quasi-estático permanece válido em freqs baixíssimas
#     (CSAMT, MT controlado). Abaixo disso, air-wave domina e o modelo
#     1D TIV não captura geometria 2D/3D.
#   • Resistividade 0.1–1e6 Ω·m: cobre argilas (~1 Ω·m), arenitos (10–100),
#     carbonatos (100–10 000), sal (10 000–100 000), crosta seca (1e5–1e6).
#
# Referências:
#   • Anderson et al. (2008) — "Multiple Array Logging While Drilling..."
#     SPWLA, ARC6 com arranjos de 8–28 pés (2.4–8.5 m).
#   • Omeragic et al. (2009) — "Deep Directional Electromagnetic Measurements"
#     SPWLA, PeriScope HD com TR até 20 m para DOI de 15 m.
#   • Moran & Gianzero (1979) — Geophysics 44, quasi-estática TIV.
_FREQUENCY_HZ_RANGE: Final[tuple[float, float]] = (10.0, 2.0e6)
_TR_SPACING_M_RANGE: Final[tuple[float, float]] = (0.01, 50.0)
_N_POSITIONS_RANGE: Final[tuple[int, int]] = (10, 100_000)
# Resistividade horizontal e vertical (Ω·m). Range amplo cobre toda a
# variedade de litologias encontradas em LWD, desde folhelhos salinos
# (~1 Ω·m) até rochas salinas/ígneas (~10⁶ Ω·m).
_RESISTIVITY_OHM_M_RANGE: Final[tuple[float, float]] = (0.1, 1.0e6)

# Conjuntos de valores válidos para campos enum-like. Mantidos como
# `frozenset` para lookup O(1) em `__post_init__` e imutabilidade.
_VALID_BACKENDS: Final[frozenset[str]] = frozenset(
    {
        "fortran_f2py",  # chama tatu.x via f2py (status quo, Fases 0-6)
        "numba",  # backend CPU nativo (Fase 2)
        "jax",  # backend JAX CPU/GPU/TPU (Fase 3)
    }
)

_VALID_DTYPES: Final[frozenset[str]] = frozenset(
    {
        "complex128",  # paridade bit-exact com Fortran real(dp), default
        "complex64",  # ~2× throughput GPU, uso em produção apenas
    }
)

_VALID_DEVICES: Final[frozenset[str]] = frozenset(
    {
        "cpu",  # default; backend fortran_f2py exige cpu
        "gpu",  # requer backend jax
    }
)

# Filtros Hankel conhecidos. Esta lista é espelhada em
# `geosteering_ai/simulation/filters/loader.py::_FILTER_CATALOG` — se um
# filtro novo for adicionado, atualize ambos os lugares E o preset default.
_VALID_HANKEL_FILTERS: Final[frozenset[str]] = frozenset(
    {
        "werthmuller_201pt",  # ★ default, filter_type=0 Fortran
        "kong_61pt",  # filter_type=1, 3.3× mais rápido
        "anderson_801pt",  # filter_type=2, máxima precisão
    }
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DATACLASS SimulationConfig
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SimulationConfig:
    """Configuração imutável do simulador Python.

    Dataclass que agrega os parâmetros numéricos e de backend do simulador
    Python otimizado. A validação de errata ocorre em ``__post_init__``
    e falha cedo com mensagens contextualizadas se qualquer campo estiver
    fora do range físico ou se houver conflito de configuração (ex.:
    backend Fortran com device GPU).

    Para mutar um campo, use ``dataclasses.replace(cfg, field=novo)`` —
    isso reconstrói a instância, re-validando todos os campos no processo.

    Attributes:
        frequency_hz: Frequência de operação em Hertz (float).
            Range: [100.0, 1.0e6]. Default: 20000.0 (20 kHz).
        tr_spacing_m: Espaçamento transmissor-receptor em metros.
            Range: [0.1, 10.0]. Default: 1.0.
        n_positions: Número de posições ao longo da trajetória do poço.
            Range: [10, 100_000]. Default: 600.
        backend: Backend de execução. Opções: ``"fortran_f2py"``
            (status quo), ``"numba"`` (Fase 2), ``"jax"`` (Fase 3).
            Default: ``"fortran_f2py"``.
        dtype: Precisão de ponto flutuante. Opções: ``"complex128"``
            (paridade Fortran) ou ``"complex64"`` (2× GPU throughput).
            Default: ``"complex128"``.
        device: Dispositivo de execução. Opções: ``"cpu"`` ou ``"gpu"``.
            Default: ``"cpu"``. Incompatível com ``backend="fortran_f2py"``
            se ``device="gpu"``.
        hankel_filter: Filtro Hankel digital. Opções: filtros no
            catálogo de :class:`FilterLoader`. Default:
            ``"werthmuller_201pt"``.
        frequencies_hz: Lista de frequências adicionais (F5 multi-frequência).
            Se None, usa apenas ``frequency_hz``. Cada valor na lista
            deve estar no mesmo range de ``frequency_hz``.
        tr_spacings_m: Lista de espaçamentos TR adicionais (multi-TR).
            Se None, usa apenas ``tr_spacing_m``. Cada valor no mesmo
            range de ``tr_spacing_m``.
        compute_jacobian: Se True, calcula ∂H/∂ρ via método apropriado
            ao backend (finite differences no Numba, jax.jacfwd no JAX).
            Default: False.
        num_threads: Número de threads para backends paralelos. ``-1``
            auto-detecta (usa todos os cores disponíveis). Default: -1.
        seed: Seed para inicialização reprodutível (futuro uso em
            benchmarks estocásticos). Default: 42.

    Example:
        Configuração padrão (paridade Fortran baseline)::

            >>> cfg = SimulationConfig()
            >>> cfg.frequency_hz, cfg.tr_spacing_m, cfg.n_positions
            (20000.0, 1.0, 600)
            >>> cfg.backend, cfg.dtype, cfg.hankel_filter
            ('fortran_f2py', 'complex128', 'werthmuller_201pt')

        Configuração com multi-frequência + multi-TR::

            >>> cfg = SimulationConfig(
            ...     frequency_hz=20000.0,
            ...     frequencies_hz=[20000.0, 100000.0, 400000.0],
            ...     tr_spacings_m=[0.5, 1.0, 1.5],
            ... )

    Raises:
        AssertionError: Se qualquer campo estiver fora do range válido
            ou se houver conflito mútuo (ex.: backend fortran_f2py com
            device gpu). A mensagem inclui o valor recebido, o range
            esperado e o motivo físico/arquitetural da restrição.

    Note:
        Este dataclass é independente do `PipelineConfig` principal. A
        integração acontecerá na Fase 6 via adapter. Os campos aqui
        refletem **apenas** o que é necessário para executar uma
        simulação; parâmetros de treino (loss, noise, arquitetura,
        curriculum) permanecem no `PipelineConfig`.
    """

    # ── Grupo 1: Parâmetros Físicos (frequência e geometria) ────────
    # Os três campos abaixo cobrem o mínimo para uma simulação
    # single-frequency single-TR, que é o caso de uso de 95% dos
    # modelos no dataset atual. Campos opcionais para multi-frequência
    # e multi-TR estão mais abaixo (Grupo 4).
    frequency_hz: float = 20000.0
    tr_spacing_m: float = 1.0
    n_positions: int = 600

    # ── Grupo 2: Backend de Execução ─────────────────────────────────
    # `backend` seleciona qual implementação usar; `dtype` seleciona
    # precisão numérica; `device` seleciona CPU vs GPU. O default
    # mantém o status quo (fortran_f2py via tatu.x) até que as Fases
    # 2-3 estejam prontas e validadas.
    backend: str = "fortran_f2py"
    dtype: str = "complex128"
    device: str = "cpu"

    # ── Grupo 3: Filtro Hankel ───────────────────────────────────────
    # Filtro Hankel digital usado pelas quadraturas ∫ f(kr)·Jν(kr·r) dkr.
    # Default: werthmuller_201pt (paridade filter_type=0 do Fortran).
    hankel_filter: str = "werthmuller_201pt"

    # ── Grupo 4: Extensões opcionais (F5 multi-f, multi-TR) ──────────
    # Se None, o simulador usa apenas `frequency_hz` e `tr_spacing_m`
    # (modo single-point). Se listas, aplica multi-frequência/multi-TR
    # no estilo das features F5 do Fortran v10.0.
    frequencies_hz: Optional[List[float]] = None
    tr_spacings_m: Optional[List[float]] = None

    # ── Grupo 5: Jacobiano ∂H/∂ρ (F10) ───────────────────────────────
    # Se True, o simulador calcula gradientes H em relação a ρh e ρv
    # de cada camada. No backend JAX usa `jax.jacfwd` (autodiff); no
    # Numba usa diferenças finitas centradas.
    compute_jacobian: bool = False

    # ── Grupo 6: Performance tuning ──────────────────────────────────
    # -1 no `num_threads` significa auto-detect (usa todos os cores
    # disponíveis via `multiprocessing.cpu_count()`). Valores > 0
    # forçam número específico. 0 é inválido.
    num_threads: int = -1
    seed: int = 42

    # ─────────────────────────────────────────────────────────────────
    # VALIDAÇÃO (errata imutável, inspired by PipelineConfig)
    # ─────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Valida invariantes de errata e conflitos mútuos entre campos.

        Raises:
            AssertionError: Com mensagem contextualizada (valor recebido
                + range esperado + motivo físico/arquitetural).

        Note:
            A ordem de validação é: primeiro ranges numéricos por campo,
            depois enums de string, depois conflitos mútuos (ex.: backend
            × device). Isso facilita debugging — a primeira assertion
            que falha é a mais informativa.
        """
        # ── Ranges numéricos ──────────────────────────────────────────
        fmin, fmax = _FREQUENCY_HZ_RANGE
        assert fmin <= self.frequency_hz <= fmax, (
            f"frequency_hz={self.frequency_hz} Hz fora do range válido "
            f"[{fmin}, {fmax}]. Ferramentas LWD comerciais modernas (ARC6, "
            f"PeriScope, EcoScope, AziTrak) operam em 400 kHz e 2 MHz como "
            f"pares duais. O range expandido 10 Hz–2 MHz cobre desde CSAMT "
            f"controlado (baixa freq) até dual-frequency LWD (2 MHz). "
            f"Acima de ~2 MHz, a aproximação quasi-estática |ωε/σ|≪1 "
            f"começa a falhar em rochas de alta resistividade (ρ > 1e4 Ω·m)."
        )

        smin, smax = _TR_SPACING_M_RANGE
        assert smin <= self.tr_spacing_m <= smax, (
            f"tr_spacing_m={self.tr_spacing_m} m fora do range válido "
            f"[{smin}, {smax}]. O range expandido 0.01–50 m cobre "
            f'ferramentas curtas (ARC6 arranjos 22"–40" ~0.56–1.02 m), '
            f"intermediárias (8.19 m, ARC ultra-longo) e deep-reading "
            f"(PeriScope HD 20.43 m para DOI de 15 m em geosteering). "
            f"Fora deste range, a precisão do filtro Hankel Werthmüller "
            f"201pt (abscissas kr∈[8.7e-4, 93.7]) pode degradar — use "
            f"Anderson 801pt para r > 30 m."
        )

        nmin, nmax = _N_POSITIONS_RANGE
        assert nmin <= self.n_positions <= nmax, (
            f"n_positions={self.n_positions} fora do range válido "
            f"[{nmin}, {nmax}]. Mínimo 10 posições para qualquer "
            f"exercício didático; máximo {nmax} antes de saturar "
            f"memória CPU em batches do pipeline."
        )

        # ── Enums de string ──────────────────────────────────────────
        assert self.backend in _VALID_BACKENDS, (
            f"backend={self.backend!r} inválido. Opções: "
            f"{sorted(_VALID_BACKENDS)}. O default 'fortran_f2py' "
            f"permanecerá até a Fase 6 do plano; 'numba' e 'jax' serão "
            f"ativados conforme Fases 2 e 3, respectivamente."
        )

        assert self.dtype in _VALID_DTYPES, (
            f"dtype={self.dtype!r} inválido. Opções: "
            f"{sorted(_VALID_DTYPES)}. complex128 é default (bit-exato "
            f"com real(dp) Fortran); complex64 dá ~2× throughput GPU "
            f"mas reduz precisão — uso em produção apenas."
        )

        assert self.device in _VALID_DEVICES, (
            f"device={self.device!r} inválido. Opções: " f"{sorted(_VALID_DEVICES)}."
        )

        assert self.hankel_filter in _VALID_HANKEL_FILTERS, (
            f"hankel_filter={self.hankel_filter!r} inválido. Opções: "
            f"{sorted(_VALID_HANKEL_FILTERS)}. werthmuller_201pt é o "
            f"default (paridade filter_type=0 do Fortran)."
        )

        # ── Conflitos mútuos entre backend e device ──────────────────
        # O backend fortran_f2py chama o binário tatu.x via f2py, que
        # roda exclusivamente em CPU. Tentar usá-lo com GPU é um erro
        # de configuração — o usuário provavelmente quis usar 'jax'.
        assert not (self.backend == "fortran_f2py" and self.device == "gpu"), (
            f"backend='fortran_f2py' é incompatível com device='gpu'. "
            f"O binário tatu.x roda apenas em CPU. Para GPU, use "
            f"backend='jax' (disponível na Fase 3)."
        )

        # ── Exclusividade entre backend e dtype em GPU ───────────────
        # O backend Numba não suporta GPU no roadmap atual. Se o
        # usuário pediu Numba + GPU, provavelmente quis JAX.
        assert not (self.backend == "numba" and self.device == "gpu"), (
            f"backend='numba' roda apenas em CPU no roadmap atual. "
            f"Para GPU, use backend='jax'."
        )

        # ── Listas opcionais (multi-f, multi-TR) ─────────────────────
        # Se definidas, devem conter ≥ 1 elemento no range físico. Uma
        # lista vazia (len=0) é sempre erro de configuração.
        if self.frequencies_hz is not None:
            assert len(self.frequencies_hz) >= 1, (
                f"frequencies_hz=[] inválido: se definido, deve conter "
                f"≥ 1 frequência."
            )
            for i, f in enumerate(self.frequencies_hz):
                assert fmin <= f <= fmax, (
                    f"frequencies_hz[{i}]={f} Hz fora do range " f"[{fmin}, {fmax}]."
                )

        if self.tr_spacings_m is not None:
            assert len(self.tr_spacings_m) >= 1, (
                f"tr_spacings_m=[] inválido: se definido, deve conter "
                f"≥ 1 espaçamento."
            )
            for i, s in enumerate(self.tr_spacings_m):
                assert smin <= s <= smax, (
                    f"tr_spacings_m[{i}]={s} m fora do range " f"[{smin}, {smax}]."
                )

        # ── Performance tuning ───────────────────────────────────────
        assert self.num_threads == -1 or self.num_threads >= 1, (
            f"num_threads={self.num_threads} inválido. Use -1 para "
            f"auto-detectar (usa todos os cores) ou um inteiro >= 1."
        )

        # ── Jacobiano: só faz sentido em backends de produção ────────
        # Apesar do backend fortran_f2py ter F10 nativo, o campo
        # `compute_jacobian` do SimulationConfig refere-se à API
        # unificada Python. Para Fortran, o valor é ignorado pelo
        # adapter (que sempre passa compute_jacobian=False via model.in).
        # Logging informativo — não é erro.
        if self.compute_jacobian and self.backend == "fortran_f2py":
            logger.debug(
                "compute_jacobian=True com backend=fortran_f2py — "
                "habilitado via model.in na Fase 6 (adapter pendente)"
            )

    # ─────────────────────────────────────────────────────────────────
    # PRESETS (@classmethod)
    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def default(cls) -> "SimulationConfig":
        """Preset padrão — paridade total com baseline Fortran.

        Returns:
            SimulationConfig com frequency_hz=20000, tr_spacing_m=1.0,
            n_positions=600, backend=fortran_f2py, dtype=complex128,
            device=cpu, hankel_filter=werthmuller_201pt.

        Note:
            Este é o mesmo resultado de `SimulationConfig()` — o
            preset existe para explicitar a intenção no código chamador
            (auto-documentação). Corresponde ao conjunto de valores
            físicos da "Errata Imutável" do projeto.
        """
        return cls()

    @classmethod
    def high_precision(cls) -> "SimulationConfig":
        """Preset de máxima precisão numérica.

        Usa filtro Anderson 801pt (precisão ~10⁻⁸) e complex128.
        Indicado para estudos de referência, validação científica e
        geração de ground-truth para testes de regressão.

        Returns:
            SimulationConfig com hankel_filter=anderson_801pt,
            dtype=complex128, backend=fortran_f2py (default até Fase 6).
        """
        return cls(
            hankel_filter="anderson_801pt",
            dtype="complex128",
        )

    @classmethod
    def production_gpu(cls) -> "SimulationConfig":
        """Preset de produção em GPU (máximo throughput).

        Usa backend JAX com precisão simples (complex64) e filtro
        Kong 61pt (~3.3× mais rápido que Werthmüller 201pt) para
        maximizar o throughput em GPU T4/A100. A perda de precisão
        é aceitável para treinamento DL (dataset generation).

        Returns:
            SimulationConfig com backend=jax, device=gpu, dtype=complex64,
            hankel_filter=kong_61pt.

        Note:
            Requer que a Fase 3 (backend JAX) esteja completa. Chamar
            `simulate(cfg)` com este preset antes da Fase 3 resultará
            em NotImplementedError no dispatcher de backend.
        """
        return cls(
            backend="jax",
            device="gpu",
            dtype="complex64",
            hankel_filter="kong_61pt",
        )

    @classmethod
    def realtime_cpu(cls) -> "SimulationConfig":
        """Preset de inferência em tempo real em CPU.

        Usa backend Numba com filtro Kong 61pt para minimizar latência
        por modelo. Indicado para cenários causais (geosteering realtime)
        onde cada novo ponto de medida dispara uma nova simulação.

        Returns:
            SimulationConfig com backend=numba, device=cpu,
            dtype=complex128, hankel_filter=kong_61pt.

        Note:
            Requer que a Fase 2 (backend Numba) esteja completa.
        """
        return cls(
            backend="numba",
            device="cpu",
            dtype="complex128",
            hankel_filter="kong_61pt",
        )

    # ─────────────────────────────────────────────────────────────────
    # SERIALIZAÇÃO (YAML roundtrip)
    # ─────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a configuração em um dict Python.

        Returns:
            Dict com todos os campos do dataclass. Listas são
            copiadas (shallow) para evitar compartilhamento de
            referências com chamadores.

        Note:
            Usa `dataclasses.asdict` que faz deep-copy automático de
            listas e dicts aninhados. O resultado é JSON-serializável.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Reconstrói uma configuração a partir de um dict.

        Args:
            data: Dict produzido por `to_dict` (ou equivalente).
                Pode conter chaves adicionais que serão ignoradas
                com warning; chaves faltantes usarão defaults do
                dataclass.

        Returns:
            Nova instância validada de SimulationConfig.

        Raises:
            AssertionError: Se a instância construída falhar na
                validação de `__post_init__`.
        """
        # Filtra apenas campos conhecidos; ignora extras com log.
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}

        extra = set(data.keys()) - known_fields
        if extra:
            logger.warning(
                "from_dict ignorando chaves desconhecidas: %s",
                sorted(extra),
            )

        return cls(**filtered)

    def to_yaml(self, path: Path | str) -> None:
        """Serializa a configuração para um arquivo YAML.

        Args:
            path: Caminho do arquivo de saída.

        Raises:
            ImportError: Se PyYAML não estiver instalado.

        Note:
            Lazy import de `yaml` — o pipeline mínimo do simulador
            não precisa de PyYAML. Apenas chamadas explícitas a
            `to_yaml`/`from_yaml` ativam o import.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML é necessário para SimulationConfig.to_yaml. "
                "Instale com: pip install pyyaml"
            ) from exc

        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        logger.debug("SimulationConfig gravado em %s", path)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SimulationConfig":
        """Reconstrói uma configuração a partir de um arquivo YAML.

        Args:
            path: Caminho do arquivo YAML de entrada.

        Returns:
            Nova instância validada.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ImportError: Se PyYAML não estiver instalado.
            AssertionError: Se a configuração carregada for inválida.

        Note:
            Usa `yaml.safe_load` (sem execução de código arbitrário).
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML é necessário para SimulationConfig.from_yaml. "
                "Instale com: pip install pyyaml"
            ) from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração YAML não encontrado: {path}"
            )

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"YAML em {path} não contém um mapeamento (dict) no "
                f"topo. Conteúdo lido: {type(data).__name__}."
            )

        logger.debug("SimulationConfig lido de %s", path)
        return cls.from_dict(data)


__all__ = ["SimulationConfig"]
