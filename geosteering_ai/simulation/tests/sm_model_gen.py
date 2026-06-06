# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_model_gen.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — casca Qt do gerador estocástico TIV   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18  ·  Reescrita 0011c: 2026-06-06                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy + Qt6 (via gui.qt_compat)                            ║
# ║  Dependências: gui.services.stochastic_geology (core PURO), gui.qt_compat ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    A casca Qt (assíncrona) do gerador estocástico de perfis TIV. O CORE    ║
# ║    PURO (``GenConfig``, ``generate_models``, 7 samplers, stick-breaking,    ║
# ║    seed) foi EXTRAÍDO para ``geosteering_ai.gui.services.stochastic_       ║
# ║    geology`` (spec 0011c, extração Strangler — DRY): este módulo agora      ║
# ║    RE-IMPORTA esse core e mantém apenas ``ModelGenerationThread`` (QThread). ║
# ║    Compatibilidade total: todos os símbolos antes expostos aqui continuam   ║
# ║    importáveis via este módulo (re-export em ``__all__``).                  ║
# ║                                                                           ║
# ║  API PÚBLICA (re-exportada do core + Qt)                                  ║
# ║    • GenConfig · generate_models · MODEL_KEYS · GENERATORS_AVAILABLE       ║
# ║    • _resolve_rng_seed  (core)                                            ║
# ║    • DEFAULT_GEN_CHUNK_SIZE · ModelGenerationThread  (casca Qt)            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Casca Qt do gerador estocástico TIV — core puro em ``gui.services.stochastic_geology``.

A geração paramétrica de perfis TIV (7 geradores aleatórios/quasi-aleatórios) vive
agora no core PURO ``geosteering_ai.gui.services.stochastic_geology`` (extração 0011c),
compartilhado por este módulo (monólito) e pelo app MVVM. Aqui fica só a versão
assíncrona :class:`ModelGenerationThread` (QThread, para N grande não bloquear a GUI).

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

import logging
from typing import List, Optional

import numpy as np

# ── Core PURO (extraído em 0011c) — re-importado p/ retrocompat dos consumidores ──
from geosteering_ai.gui.services.stochastic_geology import (
    GENERATORS_AVAILABLE,
    MODEL_KEYS,
    GenConfig,
    _build_n_layer_choices,
    _generate_one_model,
    _resolve_rng_seed,
    generate_models,
)

logger = logging.getLogger(__name__)


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

# Lazy import of Qt — via gui.qt_compat (spec 0011 Fase 0).
try:
    from geosteering_ai.gui.qt_compat import QThread, Signal  # type: ignore

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
    núcleo que :func:`generate_models`, agora em ``stochastic_geology``),
    garantindo que ambas as APIs produzam **modelos idênticos bit-a-bit**
    para a mesma seed.

    Attributes:
        progress: Sinal Qt ``Signal(int, int)`` — emitido a cada chunk.
            Argumentos: ``(n_done, n_total)``.
        finished_models: Sinal Qt ``Signal(list)`` — emitido ao concluir
            com a lista completa de modelos.
        error: Sinal Qt ``Signal(str)`` — emitido em caso de exceção
            ou cancelamento. Argumento: mensagem.
        cancelled: Sinal Qt ``Signal()`` — emitido após cancelamento
            cooperativo (post `request_cancel`).
        seed_used: Sinal Qt ``Signal(int)`` (v2.19) — emitido no início
            do ``run()`` com a semente PRNG efetivamente usada.
            Permite à GUI logar a semente para reprodutibilidade,
            mesmo quando o usuário escolhe modo aleatório.

    Example:
        Uso típico em ``MainWindow._start_simulation``::

            >>> # Modo aleatório (v2.19 default) — modelos distintos a cada exec
            >>> self._gen_thread = ModelGenerationThread(
            ...     gen_cfg, n_models=30000, rng_seed=None
            ... )
            >>> self._gen_thread.progress.connect(self._on_gen_progress)
            >>> self._gen_thread.finished_models.connect(self._on_models_ready)
            >>> self._gen_thread.error.connect(self._on_gen_error)
            >>> self._gen_thread.seed_used.connect(self._log_seed)
            >>> self._gen_thread.start()

            >>> # Modo reprodutível — semente fixa
            >>> ModelGenerationThread(gen_cfg, n_models=30000, rng_seed=42)

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
    # v2.19: emite a semente realmente usada (resolvida em run()) para
    # logging/reprodutibilidade — útil quando rng_seed=None (aleatória).
    seed_used = Signal(int)

    def __init__(
        self,
        cfg: GenConfig,
        n_models: int,
        rng_seed: Optional[int] = None,
        chunk_size: int = DEFAULT_GEN_CHUNK_SIZE,
        parent: Optional[object] = None,
    ) -> None:
        """Inicializa a thread sem startá-la (chame ``start()``).

        Args:
            cfg: Configuração de geração — copiada por referência;
                imutabilidade é responsabilidade do chamador.
            n_models: Total de modelos a gerar.
            rng_seed: ``None`` (default v2.19) gera semente aleatória
                a cada execução. Passe inteiro fixo para reprodutibilidade.
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
        # v2.19: preserva None (resolve em run()) para que cada start()
        # gere modelos distintos quando o usuário escolhe modo aleatório.
        self._rng_seed: Optional[int] = None if rng_seed is None else int(rng_seed)
        self._actual_seed: Optional[int] = None  # resolvido em run()
        self._chunk_size = max(1, int(chunk_size))
        # Flag de cancelamento cooperativo — set pelo `request_cancel()`,
        # checado entre chunks. Acesso atômico em CPython (GIL).
        self._cancelled: bool = False

    @property
    def actual_seed(self) -> Optional[int]:
        """Semente PRNG realmente usada em ``run()`` (None antes do start)."""
        return self._actual_seed

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
            # v2.19: resolve a semente uma única vez no início da run().
            # Quando self._rng_seed é None, cada start() gera uma semente
            # aleatória distinta — comportamento físico esperado para
            # ensembles estatísticos diversos.
            self._actual_seed = _resolve_rng_seed(self._rng_seed)
            seed_mode = "aleatória" if self._rng_seed is None else "fixa"
            logger.info(
                "ModelGenerationThread: seed=%d (%s), n_models=%d",
                self._actual_seed,
                seed_mode,
                self._n_models,
            )
            # Emite a semente para a GUI logar antes do progresso começar.
            self.seed_used.emit(int(self._actual_seed))
            rng = np.random.default_rng(self._actual_seed)
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
                            self._cfg, rng, n_layer_choices, i, self._actual_seed
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
    "_resolve_rng_seed",
    "generate_models",
]
