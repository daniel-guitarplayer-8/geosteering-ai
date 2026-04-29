# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_model_gen.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Gerador Estocástico de Modelos TIV    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy + SciPy (qmc.Sobol/Halton)                           ║
# ║  Dependências: numpy ≥ 1.24, scipy ≥ 1.10                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Sintetiza perfis geológicos TIV (ρh, ρv, esp) a partir de múltiplos   ║
# ║    geradores de números aleatórios e quasi-aleatórios. Port leve e auto- ║
# ║    contido de `Fortran_Gerador/fifthBuildTIVModels.py`, sem dependências  ║
# ║    externas do pacote raiz.                                               ║
# ║                                                                           ║
# ║  GERADORES SUPORTADOS                                                     ║
# ║    ┌────────────────────┬─────────────────────────────────────────────┐  ║
# ║    │  Sobol (★ padrão)  │  QMC — cobertura espacial densa (scipy.qmc) │  ║
# ║    │  Halton            │  QMC — baixa discrepância base prima        │  ║
# ║    │  Niederreiter      │  QMC — fallback para Halton (scipy gap)     │  ║
# ║    │  Mersenne Twister  │  PRNG — padrão Python/NumPy RNG             │  ║
# ║    │  Uniforme          │  PRNG — alias `uniform` via Generator.random │  ║
# ║    │  Normal            │  PRNG — ρ log-normal com μ,σ parametrizáveis │  ║
# ║    │  Box-Muller        │  PRNG — transform analítico U→N (sanity)    │  ║
# ║    └────────────────────┴─────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  API PÚBLICA                                                              ║
# ║    • GenConfig dataclass — parametrização completa                       ║
# ║    • generate_models(cfg, n_models, rng_seed) → list[dict]               ║
# ║    • MODEL_KEYS — constante de chaves canônicas do dict de modelo        ║
# ║                                                                           ║
# ║  FORMATO DE SAÍDA (MODEL DICT)                                            ║
# ║    {                                                                       ║
# ║      "n_layers": int,                                                     ║
# ║      "rho_h":    list[float] (n_layers,),                                 ║
# ║      "rho_v":    list[float] (n_layers,),                                 ║
# ║      "lambda":   list[float] (n_layers,),                                 ║
# ║      "thicknesses": list[float] (n_layers - 2,)                           ║
# ║    }                                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Gerador estocástico de perfis TIV para Simulation Manager.

Port auto-contido de ``fifthBuildTIVModels.py`` que oferece geração
paramétrica de perfis TIV para o Simulation Manager, com suporte a sete
famílias de geradores de números aleatórios e quasi-aleatórios.

Example:
    Geração básica (Sobol, 100 modelos, 20 camadas, alta-ρ)::

        >>> cfg = GenConfig(
        ...     n_layers_fixed=20, rho_h_min=1000.0, rho_h_max=10000.0,
        ...     anisotropic=True, lambda_min=1.0, lambda_max=1.7,
        ...     generator="sobol",
        ... )
        >>> models = generate_models(cfg, n_models=100, rng_seed=42)
        >>> len(models), models[0]["n_layers"]
        (100, 20)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# SciPy QMC é opcional — se não disponível, degradamos para np.random.
try:
    from scipy.stats import qmc as _qmc

    _HAS_SCIPY_QMC = True
except Exception:  # pragma: no cover
    _HAS_SCIPY_QMC = False


# ──────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────
MODEL_KEYS = ("n_layers", "rho_h", "rho_v", "lambda", "thicknesses")

GENERATORS_AVAILABLE = [
    "sobol",
    "halton",
    "niederreiter",
    "mersenne_twister",
    "uniform",
    "normal",
    "box_muller",
]

# ──────────────────────────────────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class GenConfig:
    """Parâmetros de geração estocástica de perfis TIV.

    Attributes:
        total_depth: Profundidade total da janela geológica em metros (``tji``).
            Default ``100.0`` m.
        n_layers_min: Limite inferior de camadas sorteadas (inclusive).
        n_layers_max: Limite superior de camadas sorteadas (exclusive).
        n_layers_fixed: Se ≥ 3, força todos os modelos a terem exatamente
            esse número de camadas. Default ``None`` (amostragem uniforme).
        rho_h_min: Limite inferior de ρₕ em Ω·m.
        rho_h_max: Limite superior de ρₕ em Ω·m.
        rho_h_distribution: ``"loguni"`` (log-uniforme) ou ``"uniform"``.
        anisotropic: Se ``True``, aplica fator anisotrópico λ. Se ``False``,
            força ``rho_v = rho_h`` (isotrópico).
        lambda_min, lambda_max: Range do fator de anisotropia λ (ρᵥ = λ²·ρₕ).
        min_thickness: Espessura mínima por camada em metros.
        generator: Nome do gerador (ver ``GENERATORS_AVAILABLE``).
        normal_mu_log, normal_sigma_log: Parâmetros log-normal quando
            ``generator="normal"``.
    """

    total_depth: float = 100.0
    n_layers_min: int = 3
    n_layers_max: int = 31
    n_layers_fixed: Optional[int] = None
    rho_h_min: float = 1.0
    rho_h_max: float = 1800.0
    rho_h_distribution: str = "loguni"
    anisotropic: bool = True
    lambda_min: float = 1.0
    lambda_max: float = math.sqrt(2.0)
    min_thickness: float = 1.0
    generator: str = "sobol"
    normal_mu_log: float = 2.0
    normal_sigma_log: float = 1.0

    def validate(self) -> None:
        """Valida ranges e consistência dos parâmetros."""
        if self.n_layers_fixed is not None and self.n_layers_fixed < 3:
            raise ValueError("n_layers_fixed deve ser ≥ 3 (inclui 2 semi-espaços).")
        if self.rho_h_min <= 0 or self.rho_h_max <= self.rho_h_min:
            raise ValueError("Intervalo de ρh inválido (precisa ρ_min>0 e ρ_max>ρ_min).")
        if self.lambda_min < 1.0:
            raise ValueError("λ mínimo deve ser ≥ 1.0 (ρᵥ ≥ ρₕ em TIV física).")
        if self.generator not in GENERATORS_AVAILABLE:
            raise ValueError(
                f"Gerador {self.generator!r} desconhecido. "
                f"Opções: {GENERATORS_AVAILABLE}"
            )
        if self.rho_h_distribution not in ("loguni", "uniform"):
            raise ValueError("rho_h_distribution deve ser 'loguni' ou 'uniform'.")


# ──────────────────────────────────────────────────────────────────────────
# Transformações amostrais
# ──────────────────────────────────────────────────────────────────────────


def _log_transform(samples: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mapeia U[0,1] → log-uniforme [lo, hi]."""
    return np.exp(np.log(lo) + samples * (np.log(hi) - np.log(lo)))


def _uniform_transform(samples: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mapeia U[0,1] → uniforme [lo, hi]."""
    return lo + samples * (hi - lo)


# ──────────────────────────────────────────────────────────────────────────
# Fontes de amostragem (QMC e PRNG)
# ──────────────────────────────────────────────────────────────────────────


def _samples_sobol(rng: np.random.Generator, n_dim: int, seed: int) -> np.ndarray:
    """Amostra Sobol (scramble=True) de 1 ponto com n_dim dimensões."""
    if _HAS_SCIPY_QMC:
        # seed garante reprodutibilidade entre workers
        sampler = _qmc.Sobol(d=n_dim, scramble=True, seed=seed)
        return sampler.random(1).flatten()
    return rng.random(n_dim)  # fallback MT19937


def _samples_halton(rng: np.random.Generator, n_dim: int, seed: int) -> np.ndarray:
    """Amostra Halton com bases primas (scipy ≥ 1.7)."""
    if _HAS_SCIPY_QMC:
        sampler = _qmc.Halton(d=n_dim, scramble=True, seed=seed)
        return sampler.random(1).flatten()
    return rng.random(n_dim)


def _samples_niederreiter(rng: np.random.Generator, n_dim: int, seed: int) -> np.ndarray:
    """Aproximação Niederreiter via Halton (scipy não traz Niederreiter nativo)."""
    # SciPy < 1.12 não tem classe Niederreiter; Halton é o substituto mais próximo.
    if _HAS_SCIPY_QMC:
        sampler = _qmc.Halton(d=n_dim, scramble=True, seed=seed + 17)
        return sampler.random(1).flatten()
    return rng.random(n_dim)


def _samples_mersenne(rng: np.random.Generator, n_dim: int, _seed: int) -> np.ndarray:
    """Mersenne Twister PRNG (default NumPy)."""
    return rng.random(n_dim)


def _samples_uniform(rng: np.random.Generator, n_dim: int, _seed: int) -> np.ndarray:
    """Amostragem uniforme PRNG (alias de Mersenne)."""
    return rng.uniform(0.0, 1.0, n_dim)


def _samples_normal(
    rng: np.random.Generator, n_dim: int, _seed: int, mu: float, sigma: float
) -> np.ndarray:
    """Normal padronizada, mapeada para [0,1] via CDF (Φ).

    Usa ``math.erf`` da stdlib para a CDF, evitando dependência de
    ``scipy.special`` (que não é obrigatória para rodar o Simulation
    Manager — scipy só é necessário para geradores QMC Sobol/Halton).
    """
    raw = rng.normal(loc=mu, scale=sigma, size=n_dim)
    z = (raw - mu) / max(sigma, 1e-12)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return np.clip(cdf, 1e-6, 1.0 - 1e-6)


def _samples_box_muller(rng: np.random.Generator, n_dim: int, _seed: int) -> np.ndarray:
    """Box-Muller transform: U1,U2 → N(0,1) → clipa em U[0,1] via CDF."""
    u1 = rng.random(n_dim)
    u2 = rng.random(n_dim)
    z = np.sqrt(-2.0 * np.log(np.clip(u1, 1e-12, 1.0))) * np.cos(2.0 * np.pi * u2)
    # Mapeia via CDF empírica Φ(z)
    u = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return np.clip(u, 1e-6, 1.0 - 1e-6)


def _sample_vector(
    cfg: GenConfig, rng: np.random.Generator, n_dim: int, seed_hint: int
) -> np.ndarray:
    """Seleciona o gerador conforme ``cfg.generator`` e retorna vetor U[0,1]^n_dim."""
    gen = cfg.generator
    if gen == "sobol":
        return _samples_sobol(rng, n_dim, seed_hint)
    if gen == "halton":
        return _samples_halton(rng, n_dim, seed_hint)
    if gen == "niederreiter":
        return _samples_niederreiter(rng, n_dim, seed_hint)
    if gen == "mersenne_twister":
        return _samples_mersenne(rng, n_dim, seed_hint)
    if gen == "uniform":
        return _samples_uniform(rng, n_dim, seed_hint)
    if gen == "normal":
        return _samples_normal(
            rng, n_dim, seed_hint, cfg.normal_mu_log, cfg.normal_sigma_log
        )
    if gen == "box_muller":
        return _samples_box_muller(rng, n_dim, seed_hint)
    raise ValueError(f"Gerador {gen!r} não suportado.")


# ──────────────────────────────────────────────────────────────────────────
# Geração de espessuras (stick-breaking + piso mínimo)
# ──────────────────────────────────────────────────────────────────────────


def _generate_thicknesses(
    internal_layers: int,
    sobol_portion: np.ndarray,
    total_depth: float,
    min_thickness: float,
) -> np.ndarray:
    """Stick-breaking com piso mínimo, idêntico a fifthBuildTIVModels.

    Usa ordenação da amostra quasi-aleatória para particionar o intervalo
    [0, total_depth] em ``internal_layers`` segmentos, corrigindo
    iterativamente os que ficam abaixo do piso ``min_thickness``.
    """
    if internal_layers <= 0:
        return np.array([], dtype=np.float64)
    samples = np.sort(np.asarray(sobol_portion, dtype=np.float64).flatten())
    points = np.concatenate(([0.0], samples, [1.0]))
    thicknesses = np.diff(points) * total_depth

    for _ in range(10):
        idx_small = np.where(thicknesses < min_thickness)[0]
        if len(idx_small) == 0:
            break
        excess = (min_thickness - thicknesses[idx_small]).sum()
        thicknesses[idx_small] = min_thickness
        idx_rest = np.where(thicknesses >= min_thickness)[0]
        if len(idx_rest) > 0:
            thicknesses[idx_rest] -= excess / len(idx_rest)
        else:
            thicknesses += (total_depth - thicknesses.sum()) / internal_layers

    # Normaliza soma para total_depth (garante consistência numérica)
    total = thicknesses.sum()
    if total > 0:
        thicknesses = thicknesses / total * total_depth
    return thicknesses


# ──────────────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────────────


def _generate_one_model(
    cfg: GenConfig,
    rng: np.random.Generator,
    n_layer_choices: np.ndarray,
    index: int,
    rng_seed: int,
) -> dict:
    """Constrói um único modelo TIV — núcleo extraído para reuso (v2.11).

    Esta função encapsula a lógica de uma iteração do loop ``generate_models``,
    permitindo que a versão síncrona (:func:`generate_models`) e a versão
    assíncrona (:class:`ModelGenerationThread`) compartilhem exatamente o
    mesmo caminho de código — garantindo bit-paridade entre ambas.

    Args:
        cfg: Configuração de geração (já validada).
        rng: Gerador NumPy mestre (reutilizado entre iterações para
            reprodutibilidade idêntica ao loop original).
        n_layer_choices: Array pré-computado de opções de ``n_layers``.
        index: Índice do modelo na sequência (0..n_models-1) — usado
            apenas para derivar ``seed_hint`` consistente.
        rng_seed: Semente original — combinada com ``index`` para
            ``seed_hint``.

    Returns:
        Dict no formato ``MODEL_KEYS``.
    """
    if cfg.n_layers_fixed is None:
        ncam = int(rng.choice(n_layer_choices))
    else:
        ncam = int(cfg.n_layers_fixed)

    ncamint = ncam - 2  # camadas internas (finitas)
    n_thick_dims = max(0, ncamint - 1)
    # Dimensões totais: ρh (ncam) + λ (ncam) + espessuras (n_thick_dims)
    n_dim = ncam + ncam + n_thick_dims
    seed_hint = int((rng_seed + 1) * 1_000_003 + index)

    sample = _sample_vector(cfg, rng, n_dim, seed_hint)

    # Fatiamento semântico do vetor amostral
    rho_h_portion = sample[0:ncam]
    lambda_portion = sample[ncam : 2 * ncam]
    thick_portion = sample[2 * ncam : 2 * ncam + n_thick_dims]

    # Transformação ρₕ
    if cfg.rho_h_distribution == "uniform":
        rho_h = _uniform_transform(rho_h_portion, cfg.rho_h_min, cfg.rho_h_max)
    else:
        rho_h = _log_transform(rho_h_portion, cfg.rho_h_min, cfg.rho_h_max)

    # Transformação λ + ρᵥ
    if cfg.anisotropic:
        lambdas = _uniform_transform(lambda_portion, cfg.lambda_min, cfg.lambda_max)
    else:
        lambdas = np.ones(ncam, dtype=np.float64)
    rho_v = (lambdas**2) * rho_h

    thicknesses = _generate_thicknesses(
        ncamint, thick_portion, cfg.total_depth, cfg.min_thickness
    )

    return {
        "n_layers": ncam,
        "rho_h": [float(v) for v in rho_h],
        "rho_v": [float(v) for v in rho_v],
        "lambda": [float(v) for v in lambdas],
        "thicknesses": [float(v) for v in thicknesses],
    }


def _build_n_layer_choices(cfg: GenConfig) -> np.ndarray:
    """Pré-computa o array de opções de ``n_layers`` (uniforme entre min e max)."""
    if cfg.n_layers_fixed is None:
        return np.arange(cfg.n_layers_min, cfg.n_layers_max)
    return np.array([cfg.n_layers_fixed], dtype=np.int64)


def generate_models(cfg: GenConfig, n_models: int, rng_seed: int = 42) -> List[dict]:
    """Gera lista de perfis geológicos estocásticos (versão síncrona).

    Esta API mantém compatibilidade total com chamadores antigos. Para
    geração não-bloqueante na GUI (Simulation Manager v2.11+), use
    :class:`ModelGenerationThread`.

    Args:
        cfg: Configuração paramétrica de geração.
        n_models: Número total de perfis a gerar.
        rng_seed: Seed reprodutível para o PRNG mestre.

    Returns:
        Lista de dicionários no formato ``MODEL_KEYS``.

    Note:
        Para N grande (>5k) na main thread Qt, esta função BLOQUEIA o
        event loop. Use :class:`ModelGenerationThread` na GUI.
    """
    cfg.validate()
    rng = np.random.default_rng(rng_seed)
    n_layer_choices = _build_n_layer_choices(cfg)
    return [
        _generate_one_model(cfg, rng, n_layer_choices, i, rng_seed)
        for i in range(n_models)
    ]


# ──────────────────────────────────────────────────────────────────────────
# Versão assíncrona — QThread (v2.11)
# ──────────────────────────────────────────────────────────────────────────
#
# Por que uma QThread? Para N grande (30k+ modelos) o loop de geração
# bloqueia a main thread por 3-30s ANTES de qualquer simulação iniciar
# (gargalo identificado no profiling baseline v2.11). Movendo o loop
# para um QThread separado, a main thread permanece responsiva e o
# usuário vê uma barra de progresso "Gerando modelos: 12.000 / 30.000".
#
# Por que chunk_size=500? Equilíbrio entre overhead de signal emit (cada
# emit custa ~50µs) e granularidade de UI (atualizar a cada 500 modelos
# dá ~60 atualizações por sessão de 30k → 1Hz, suave para o usuário).
#
# Por que cancelamento cooperativo? Worker thread Python não pode ser
# preemptado com segurança. Verificamos ``_cancelled`` apenas entre
# chunks para evitar deixar estado inconsistente.

# Default chunk size — calibrado para 60 fps perceptível do usuário.
# A 500 modelos/chunk × 0.5ms/modelo = 250ms por chunk → 4 chunks/segundo.
DEFAULT_GEN_CHUNK_SIZE: int = 500

# Lazy import of Qt — sm_qt_compat deve existir no mesmo pacote.
try:
    from .sm_qt_compat import QThread, Signal  # type: ignore

    _HAS_QT = True
except Exception:  # pragma: no cover
    _HAS_QT = False
    # Stubs para permitir import deste módulo em contexto headless (testes
    # de unidade sem Qt). ``ModelGenerationThread`` checa ``_HAS_QT``.
    QThread = object  # type: ignore[misc, assignment]
    Signal = lambda *_a, **_kw: None  # type: ignore[misc, assignment]  # noqa: E731


class ModelGenerationThread(QThread):  # type: ignore[misc, valid-type]
    """Geração assíncrona de modelos TIV em ``QThread`` separada.

    Resolve o **Gargalo #1** identificado no profiling baseline v2.11:
    o loop síncrono de :func:`generate_models` bloqueava a main thread
    Qt por 3-30s para N=30k modelos. Esta thread executa o mesmo loop
    em background, emitindo progresso em chunks.

    O loop interno usa exatamente :func:`_generate_one_model` (mesmo
    núcleo que :func:`generate_models`), garantindo que ambas as APIs
    produzam **modelos idênticos bit-a-bit** para a mesma seed.

    Attributes:
        progress: Sinal Qt ``Signal(int, int)`` — emitido a cada chunk.
            Argumentos: ``(n_done, n_total)``.
        finished_models: Sinal Qt ``Signal(list)`` — emitido ao concluir
            com a lista completa de modelos.
        error: Sinal Qt ``Signal(str)`` — emitido em caso de exceção
            ou cancelamento. Argumento: mensagem.
        cancelled: Sinal Qt ``Signal()`` — emitido após cancelamento
            cooperativo (post `request_cancel`).

    Example:
        Uso típico em ``MainWindow._start_simulation``::

            >>> self._gen_thread = ModelGenerationThread(
            ...     gen_cfg, n_models=30000, rng_seed=42
            ... )
            >>> self._gen_thread.progress.connect(self._on_gen_progress)
            >>> self._gen_thread.finished_models.connect(self._on_models_ready)
            >>> self._gen_thread.error.connect(self._on_gen_error)
            >>> self._gen_thread.start()

    Note:
        O cancelamento é **cooperativo** — verifica ``_cancelled`` apenas
        entre chunks. O atraso de cancelamento é, no pior caso, o tempo
        de um chunk (~250ms para 500 modelos × 0.5ms/modelo).
    """

    # Sinais Qt — emitidos da worker thread e processados na main thread
    # via sistema de queued connections automático do Qt (signal-safe).
    progress = Signal(int, int)
    finished_models = Signal(list)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        cfg: GenConfig,
        n_models: int,
        rng_seed: int = 42,
        chunk_size: int = DEFAULT_GEN_CHUNK_SIZE,
        parent: Optional[object] = None,
    ) -> None:
        """Inicializa a thread sem startá-la (chame ``start()``).

        Args:
            cfg: Configuração de geração — copiada por referência;
                imutabilidade é responsabilidade do chamador.
            n_models: Total de modelos a gerar.
            rng_seed: Semente reprodutível.
            chunk_size: Modelos por chunk antes de emitir ``progress``.
                Default 500 (equilíbrio overhead × granularidade).
            parent: Qt parent opcional (default ``None``).
        """
        if not _HAS_QT:
            raise RuntimeError(
                "ModelGenerationThread requer um binding Qt6 (PyQt6/PySide6)."
            )
        super().__init__(parent)
        self._cfg = cfg
        self._n_models = max(1, int(n_models))
        self._rng_seed = int(rng_seed)
        self._chunk_size = max(1, int(chunk_size))
        # Flag de cancelamento cooperativo — set pelo `request_cancel()`,
        # checado entre chunks. Acesso atômico em CPython (GIL).
        self._cancelled: bool = False

    def request_cancel(self) -> None:
        """Solicita cancelamento cooperativo (verificado entre chunks).

        O método retorna imediatamente; a thread pode levar até
        ``chunk_size × tempo_por_modelo`` ms para responder. Após o
        cancelamento efetivo, o sinal ``cancelled`` é emitido.
        """
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """``True`` se ``request_cancel`` foi chamado (mesmo que ainda rodando)."""
        return self._cancelled

    def run(self) -> None:  # noqa: D401 — Qt convention
        """Loop principal — executa em thread separada via ``QThread.start()``.

        Não chame diretamente. Use ``start()`` para o Qt iniciar a thread.
        """
        try:
            # Validação no início — qualquer erro aqui é exception, não silent.
            self._cfg.validate()
            rng = np.random.default_rng(self._rng_seed)
            n_layer_choices = _build_n_layer_choices(self._cfg)

            models: List[dict] = []
            chunk_size = self._chunk_size
            n_total = self._n_models

            # Loop em chunks: emite progress entre eles para main thread.
            # A check de _cancelled fica entre chunks (cooperativa).
            for chunk_start in range(0, n_total, chunk_size):
                if self._cancelled:
                    # Cancelamento solicitado — emite sinal e sai limpo.
                    self.cancelled.emit()
                    return
                chunk_end = min(chunk_start + chunk_size, n_total)
                # Loop interno do chunk — sem Qt overhead, máxima velocidade.
                for i in range(chunk_start, chunk_end):
                    models.append(
                        _generate_one_model(
                            self._cfg, rng, n_layer_choices, i, self._rng_seed
                        )
                    )
                # Emite progresso para a UI atualizar barra/label.
                self.progress.emit(len(models), n_total)
            # Conclusão normal — emite lista completa.
            self.finished_models.emit(models)
        except Exception as exc:  # noqa: BLE001 — top-level Qt thread guard
            # Qualquer exceção vai para o slot de erro — main thread decide
            # como apresentar (QMessageBox, log, etc.).
            self.error.emit(str(exc))


__all__ = [
    "DEFAULT_GEN_CHUNK_SIZE",
    "GENERATORS_AVAILABLE",
    "GenConfig",
    "MODEL_KEYS",
    "ModelGenerationThread",
    "generate_models",
]
