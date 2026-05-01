# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_snapshot_persist.py                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Persistência Assíncrona de Experimento║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-29 (v2.11)                                         ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : QThread + json.dumps (stdlib)                              ║
# ║  Dependências: sm_qt_compat                                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Move a serialização JSON + escrita em disco do ``ExperimentState``    ║
# ║    para uma ``QThread`` separada, evitando o bloqueio da main thread    ║
# ║    quando o usuário invoca "Salvar Experimento" em uma sessão com       ║
# ║    histórico grande (50+ snapshots, parâmetros densos).                ║
# ║                                                                           ║
# ║  POR QUE QTHREAD                                                          ║
# ║    `json.dumps` puro é Python (loop interno O(N) sobre estruturas      ║
# ║    aninhadas) e roda detido pelo GIL. ``f.write`` é I/O bloqueante.    ║
# ║    Para experimentos de 100+ snapshots × parâmetros completos, o tempo  ║
# ║    pode chegar a 200-500ms — perceptível como freeze.                  ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    thread = SnapshotPersistThread(experiment, target_path)               ║
# ║    thread.finished_ok.connect(self._on_save_ok)                          ║
# ║    thread.error.connect(self._on_save_error)                             ║
# ║    thread.start()                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Persistência assíncrona de :class:`ExperimentState` em ``QThread``.

Provê :class:`SnapshotPersistThread` que serializa o experimento em
JSON e grava em disco em background, eliminando o gap na main thread
durante a ação "Salvar Experimento" (ou auto-save).

Diferente de :class:`ModelGenerationThread`, esta classe **não recebe
``ExperimentState``** diretamente — recebe o JSON *já serializado*
ou um *snapshot dict* (cópia do estado). Isto evita race conditions
caso o usuário edite o experimento enquanto a thread está salvando.

Example:
    Save action assíncrono::

        >>> # 1. Capturar snapshot do estado atual (rápido, main thread)
        >>> json_str = self._experiment.to_json()
        >>> # 2. Delegar I/O para QThread
        >>> self._save_thread = SnapshotPersistThread(
        ...     json_text=json_str, target_path=path
        ... )
        >>> self._save_thread.finished_ok.connect(self._on_save_ok)
        >>> self._save_thread.error.connect(self._on_save_error)
        >>> self._save_thread.start()
"""
from __future__ import annotations

import os
from typing import Optional

from .sm_qt_compat import QThread, Signal


class SnapshotPersistThread(QThread):
    """Grava JSON serializado em disco em ``QThread`` separada.

    Recebe o texto JSON pré-serializado (não o ``ExperimentState``)
    para evitar acessos concorrentes ao estado mutável durante a
    escrita. A serialização propriamente dita (``to_json()``) é
    rápida o suficiente para rodar na main thread; a I/O é o que
    pesa em sistemas com disco lento (Colab, NFS).

    Attributes:
        finished_ok: Sinal Qt ``Signal(str)`` — emitido em sucesso.
            Argumento: caminho final do arquivo gravado.
        error: Sinal Qt ``Signal(str)`` — emitido em falha (permission,
            disk full, etc.). Argumento: mensagem de erro.

    Example:
        Captura snapshot e delega I/O::

            >>> json_str = experiment.to_json()  # main thread, rápido
            >>> thread = SnapshotPersistThread(json_str, "/tmp/exp.json")
            >>> thread.finished_ok.connect(lambda p: print(f"Saved to {p}"))
            >>> thread.error.connect(lambda e: print(f"FAIL: {e}"))
            >>> thread.start()

    Note:
        A thread não toca em ``ExperimentState``. Se o usuário modificar
        o experimento durante a escrita, as mudanças não são persistidas
        no save em curso — mas a próxima Save irá capturá-las.
    """

    # Sinais Qt — finalização do trabalho (sucesso ou erro).
    # Argumentos: path final ou mensagem de erro respectivamente.
    finished_ok = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        json_text: str,
        target_path: str,
        parent: Optional[object] = None,
    ) -> None:
        """Inicializa a thread sem iniciar a escrita.

        Args:
            json_text: Texto JSON serializado (resultado de
                ``ExperimentState.to_json()``).
            target_path: Caminho absoluto do arquivo destino. Diretório
                pai será criado automaticamente se não existir.
            parent: Qt parent opcional.
        """
        super().__init__(parent)
        # Snapshot imutável de json + path — capturados no construtor.
        # Mudanças posteriores em ExperimentState não afetam esta save.
        self._json_text = json_text
        self._target_path = target_path

    def run(self) -> None:  # noqa: D401
        """Escreve JSON em disco. Não chame diretamente — use ``start()``.

        Cria diretório pai se não existir. Em sucesso emite
        ``finished_ok(path)``; em qualquer falha emite ``error(msg)``.
        """
        try:
            # Garantir diretório pai (idempotente).
            parent_dir = os.path.dirname(self._target_path) or "."
            os.makedirs(parent_dir, exist_ok=True)
            # Escrita atômica seria ideal (write-temp-then-rename), mas
            # para simplicidade v2.11 fazemos write direto. O caso comum
            # é escrita em ~/sm_experiments/ — sem concorrência.
            with open(self._target_path, "w", encoding="utf-8") as f:
                f.write(self._json_text)
            self.finished_ok.emit(self._target_path)
        except Exception as exc:  # noqa: BLE001 — top-level Qt thread guard
            self.error.emit(str(exc))


__all__ = ["SnapshotPersistThread"]
