# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/boot_warmup.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : schedule_boot_warmup — aquece o JAX no boot do SM (opt-in)  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — boot warmup (op 3)                          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Agenda (opt-in) o pré-aquecimento do worker JAX persistente logo após o ║
# ║    boot do SM, em background, com feedback no status bar. Tira o cold-start ║
# ║    XLA da 1ª simulação do usuário. No-op se desligado OU se o JAX estiver   ║
# ║    ausente (CI CPU / dev sem jax) — nunca bloqueia nem derruba o boot.       ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    schedule_boot_warmup                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``schedule_boot_warmup`` — pré-aquece o worker JAX no boot do SM (opt-in, async)."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["schedule_boot_warmup"]


def schedule_boot_warmup(ctx: Any, window: Any, *, enabled: bool) -> Optional[Any]:
    """Agenda o warmup do JAX no boot (opt-in, async). Retorna o service ou ``None``.

    No-op (retorna ``None``) se ``enabled`` for falso OU se o JAX não estiver instalado
    (``find_spec`` resolve SEM importar jax → o processo da GUI permanece TLS-safe). Caso
    contrário, instancia o :class:`JaxWarmupService`, fia mensagens transitórias no
    status bar ("Aquecendo JAX GPU…" → limpa ao concluir, SILENCIOSO em erro — uma falha
    de warmup nunca alarma o usuário) e dispara o aquecimento em background.

    Args:
        ctx: o ``AppContext`` (o service é estacionado em ``ctx.extras`` p/ não ser
            coletado pelo GC — senão a QThread seria abortada no meio).
        window: a ``QMainWindow`` (usa ``window.statusBar().showMessage`` p/ feedback).
        enabled: preferência ``jax_boot_warmup`` (default ``False`` — opt-in).

    Returns:
        O :class:`JaxWarmupService` em execução, ou ``None`` se pulado.
    """
    if not enabled:
        return None
    # Guard TLS: find_spec resolve sem importar jax (a GUI nunca puxa JAX).
    if importlib.util.find_spec("jax") is None:
        logger.info("boot warmup: JAX ausente — pulado.")
        return None
    # Guard anti-double-schedule: se já há um warmup parkado E em voo, REUSA-o em vez de
    # sobrescrever ctx.extras — sobrescrever dropava a única ref forte ao service anterior
    # → GC da QThread em voo (QThread destroyed-while-running). Hoje há 1 call site; este
    # guard blinda contra um futuro "re-aquecer".
    existing = ctx.extras.get("jax_warmup_service")
    if existing is not None and getattr(existing, "is_busy", lambda: False)():
        return existing

    from geosteering_ai.gui.services.jax_warmup_service import JaxWarmupService

    svc = JaxWarmupService()

    def _status(msg: str) -> None:
        # Best-effort: a status bar pode não existir em shells mínimas / teardown.
        try:
            window.statusBar().showMessage(msg)
        except Exception:  # noqa: BLE001 — feedback é cosmético; nunca derruba
            pass

    def _done(_result: Any) -> None:
        _status("")  # limpa a mensagem transitória (concluído — silencioso)

    def _err(_msg: str) -> None:
        # Warmup é otimização: falha NÃO alarma (só log debug + limpa o status).
        logger.debug("boot warmup falhou (ignorado): %s", _msg)
        _status("")

    svc.finished.connect(_done)
    svc.error.connect(_err)
    _status("Aquecendo JAX GPU (uma vez)…")
    svc.warmup()
    # Estaciona o service para mantê-lo vivo (senão o GC abortaria a QThread).
    ctx.extras["jax_warmup_service"] = svc
    return svc
