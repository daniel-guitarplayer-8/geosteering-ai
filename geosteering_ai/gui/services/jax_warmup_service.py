# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/services/jax_warmup_service.py                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : JaxWarmupService — aquece o worker JAX PERSISTENTE (async)  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — services (boot warmup do SM)                          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : Qt6 via BaseService (orquestra QThread → VMSignal)          ║
# ║  Dependências: gui.services.base (pool persistente)                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Pré-aquece o worker JAX GPU PERSISTENTE no boot do SM (opt-in), tirando ║
# ║    o cold-start XLA da 1ª simulação do usuário. Submete a forma CANÔNICA   ║
# ║    do SM ao MESMO subprocesso persistente de ``base.py`` (não um efêmero), ║
# ║    em background numa ``QThread`` — a 1ª sim real reusa o worker quente.    ║
# ║                                                                           ║
# ║  TLS-SAFETY (inviolável)                                                  ║
# ║    O processo da GUI NUNCA importa JAX: o guard usa ``find_spec`` (resolve ║
# ║    sem importar) e o warmup roda no FILHO (``_warmup_in_pool`` picklável).  ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    JaxWarmupService                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``JaxWarmupService`` — pré-aquece o worker JAX persistente do SM no boot (opt-in)."""

from __future__ import annotations

from typing import Any

from geosteering_ai.gui.services.base import BaseService

__all__ = ["JaxWarmupService"]


def _warmup_in_pool() -> dict:
    """Roda NO SUBPROCESSO FILHO: importa JAX e pré-compila a forma CANÔNICA do SM.

    Picklável (módulo-nível) — submetido ao pool persistente de ``base.py``. Importa
    ``warmup_jax_simulator`` AQUI (no filho), nunca no processo da GUI (TLS-safe). Se o
    JAX estiver ausente, ``warmup_jax_simulator`` retorna ``{"skipped": True}`` gracioso.

    A forma canônica espelha a config do report (20 kHz · dip 0° · TR 1 m · 600 posições ·
    20 camadas · werthmuller_201pt · complex128 · bucketed). NÃO é física — só governa
    quais shapes XLA são pré-compilados. ``n_models=1`` aquece o caminho/CUDA + a forma de
    1 grupo; geometrias estocásticas não-canônicas ainda compilam na 1ª sim (ver
    ``docs/reference/sm_jax_persistent_worker.md``, caveat de cobertura).

    Returns:
        dict de diagnóstico do warmup (buckets aquecidos, persisted, elapsed_s, …).
    """
    from geosteering_ai.simulation._jax.warmup import warmup_jax_simulator

    # Kwargs EXPLÍCITOS (não `**dict`) p/ o mypy casar os tipos por-parâmetro.
    return warmup_jax_simulator(
        n_layers=20,
        n_positions=600,
        hankel_filter="werthmuller_201pt",
        complex_dtype="complex128",
        dip_degs=(0.0,),
        tr_spacings_m=(1.0,),
        freqs_hz=(20000.0,),
        n_models=1,
        jax_strategy="bucketed",
        verbose=False,
    )


def _submit_warmup() -> Any:
    """Roda NA QThread (processo da GUI): submete o warmup ao pool PERSISTENTE.

    Usa ``_acquire_jax_pool`` (de ``base.py``) — o MESMO subprocesso que as simulações
    jax/auto usarão depois, garantindo que o warmup deixe o worker REUTILIZÁVEL quente
    (não um filho efêmero). Bloqueia no ``future.result()`` (na QThread, não na UI).
    Import LOCAL de ``_acquire_jax_pool`` (não importa JAX no processo da GUI).

    Returns:
        O dict de diagnóstico devolvido por :func:`_warmup_in_pool` (do filho). O retorno
        é ``Any`` porque ``Future.result()`` é opaco ao type-checker — o consumidor (o
        VMSignal ``finished``) o trata como dict de diagnóstico.
    """
    from geosteering_ai.gui.services.base import _acquire_jax_pool

    return _acquire_jax_pool().submit(_warmup_in_pool).result()


class JaxWarmupService(BaseService):
    """Service que aquece o worker JAX persistente em background (boot do SM).

    Reusa o orquestrador de ``BaseService`` (``_run_async`` → QThread →
    ``finished``/``error`` VMSignal na main thread). Não bloqueia a UI.

    Note:
        :meth:`warmup` é no-op se já houver um warmup em voo (``is_busy``) ou se o JAX
        não estiver instalado (guard via ``importlib.util.find_spec`` — resolve SEM
        importar jax, preservando o TLS do processo da GUI).
    """

    def warmup(self) -> None:
        """Dispara o pré-aquecimento (não-bloqueante). Guarda re-entrância + ausência de JAX."""
        if self.is_busy():
            return
        import importlib.util

        # find_spec resolve o módulo SEM importá-lo → o processo da GUI NÃO puxa JAX
        # (TLS-safe). Sem JAX → sinaliza "skipped" e não cria worker.
        if importlib.util.find_spec("jax") is None:
            self.finished.emit({"skipped": True, "reason": "jax_absent"})
            return
        self._run_async(_submit_warmup)
