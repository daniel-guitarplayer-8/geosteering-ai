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

from typing import Any, Dict, List

from geosteering_ai.gui.services.base import BaseService

__all__ = ["JaxWarmupService"]

# n_models canônico do boot — ALINHA com o CLI ``_SM_CANON_N_MODELS=64`` (cli/warmup.py)
# p/ compartilhar o cache de disco. Era 1 (dim-líder errada → não casava com o SM real,
# que usa ~250 por subgrupo). Ver docs/reference/sm_jax_persistent_worker.md.
_BOOT_CANON_N_MODELS: int = 64


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
        n_models=_BOOT_CANON_N_MODELS,
        jax_strategy="bucketed",
        verbose=False,
    )


def _warmup_config_in_pool(specs: List[Dict[str, Any]]) -> dict:
    """Roda NO SUBPROCESSO FILHO: aquece os SHAPES EXATOS da config (warmup_jax_shapes).

    Picklável (módulo-nível). Importa ``warmup_jax_shapes`` AQUI (no filho), nunca no
    processo da GUI (TLS-safe). Os ``specs`` (de ``build_warmup_specs``, dicts com numpy
    arrays) são pickláveis. Pré-compila os mesmos programas XLA que a 1ª sim real dispara
    → cache-hit. Se o JAX estiver ausente, retorna ``{"skipped": True}`` gracioso.

    Args:
        specs: lista de specs de warmup shape-matching (de ``build_warmup_specs``).

    Returns:
        dict de diagnóstico de :func:`warmup_jax_shapes`.
    """
    from geosteering_ai.simulation._jax.warmup import warmup_jax_shapes

    return warmup_jax_shapes(specs, verbose=False)


def _submit_warmup_config(specs: List[Dict[str, Any]]) -> Any:
    """Roda NA QThread (processo da GUI): submete o warmup config-aware ao pool PERSISTENTE.

    Usa o MESMO subprocesso que as simulações jax/auto usarão (``_acquire_jax_pool``),
    deixando o worker quente p/ a 1ª sim real. Bloqueia no ``future.result()`` (na QThread,
    NÃO na UI). Import LOCAL de ``_acquire_jax_pool`` (não importa JAX no processo da GUI).

    Args:
        specs: lista de specs de warmup shape-matching.

    Returns:
        O dict de diagnóstico devolvido por :func:`_warmup_config_in_pool` (do filho).
    """
    from geosteering_ai.gui.services.base import _acquire_jax_pool

    return _acquire_jax_pool().submit(_warmup_config_in_pool, specs).result()


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

    def warmup_config(self, specs: List[Dict[str, Any]]) -> None:
        """Aquece os SHAPES EXATOS da config corrente (não-bloqueante, on-select).

        Diferente de :meth:`warmup` (forma canônica de boot), aqui os ``specs``
        (de ``build_warmup_specs``) descrevem a geometria DETERMINÍSTICA da config que o
        usuário vai rodar → a 1ª sim real fica cache-hit (190 s → ~12 s). Roda no MESMO
        worker persistente, em background. No-op se já houver warmup em voo (``is_busy``),
        se ``specs`` estiver vazia, ou se o JAX não estiver instalado (guard ``find_spec``
        — resolve SEM importar jax, preservando o TLS do processo da GUI).

        Args:
            specs: lista de specs shape-matching (de ``build_warmup_specs``).
        """
        if self.is_busy():
            return
        if not specs:
            self.finished.emit({"skipped": True, "reason": "no_specs"})
            return
        import importlib.util

        if importlib.util.find_spec("jax") is None:
            self.finished.emit({"skipped": True, "reason": "jax_absent"})
            return
        # Captura ``specs`` num thunk no-arg (``_run_async`` espera um callable sem args).
        self._run_async(lambda: _submit_warmup_config(specs))
