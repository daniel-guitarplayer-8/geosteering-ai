# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/persistence/atomic.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Escrita atômica de arquivos (crash-safe)                   ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — persistência (spec 0007)                            ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : stdlib PURO (os, tempfile) — NÃO importa Qt (Princípio X)   ║
# ║  Dependências: os, tempfile                                               ║
# ║  Padrão      : write-temp → fsync → os.replace (transação POSIX)          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Grava texto em disco de forma ATÔMICA: ou o arquivo ANTIGO permanece   ║
# ║    intacto, ou o NOVO aparece completo — NUNCA um arquivo truncado/meio   ║
# ║    escrito (ex.: crash, disco cheio, kill no meio da escrita). Endurece    ║
# ║    o ``open(path,"w").write()`` direto do Simulation Manager (spec 0007).  ║
# ║                                                                           ║
# ║  COMO (garantia de atomicidade)                                          ║
# ║    ┌────────────────────────────────────────────────────────────────┐   ║
# ║    │  1. tmp = mkstemp(dir = MESMO diretório do destino)             │   ║
# ║    │     (mesmo filesystem → os.replace é rename intra-FS, sem EXDEV)│   ║
# ║    │  2. write(tmp) → flush → fsync(tmp)   (durabilidade do conteúdo)│   ║
# ║    │  3. os.replace(tmp, destino)          (TROCA atômica — POSIX)   │   ║
# ║    │  falha em qualquer passo → unlink(tmp); destino antigo intacto  │   ║
# ║    └────────────────────────────────────────────────────────────────┘   ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    atomic_write_text                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Escrita atômica de texto em disco (``write-temp`` → ``fsync`` → ``os.replace``).

Puro-Python (``os``/``tempfile``), sem Qt — base reutilizável por ``SessionDocument``
(``.session``), ``SnapshotPersistThread`` e, futuramente, ``.gsproj`` (spec 0018).
"""

from __future__ import annotations

import os
import stat
import tempfile

__all__ = ["atomic_write_text"]


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Grava ``text`` em ``path`` de forma ATÔMICA (crash-safe).

    Escreve primeiro num arquivo temporário no MESMO diretório do destino, força
    a durabilidade (``flush`` + ``fsync``) e então faz ``os.replace`` — uma troca
    atômica no POSIX. Se algo falhar, o destino ANTIGO permanece intacto e o
    temporário é removido.

    Args:
        path: caminho do arquivo destino. O diretório pai é criado se não existir.
        text: conteúdo textual a gravar.
        encoding: codificação do texto (default ``"utf-8"``).

    Raises:
        OSError: erros de I/O (permissão, disco cheio, …) — propagados após
            limpar o temporário; o destino antigo NÃO é corrompido.

    Note:
        O temporário é criado no MESMO diretório do destino de propósito:
        ``os.replace`` só é atômico DENTRO do mesmo filesystem (evita
        ``OSError: [Errno 18] EXDEV`` ao cruzar dispositivos, ex.: ``/tmp`` em
        tmpfs vs. ``$HOME`` em disco).

    Note:
        Permissões: se o destino já EXISTE, seu modo (ex.: ``0644``) é
        PRESERVADO — sem isto, o ``0600`` do ``mkstemp`` regrediria um arquivo
        legível por grupo/outros para privado. Arquivos NOVOS ficam ``0600``
        (privado ao usuário), apropriado para estado de sessão.
    """
    target = os.fspath(path)
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)

    # Preserva o modo do destino existente (mkstemp cria 0600 → sem isto, um
    # arquivo 0644 viraria 0600 após o replace; regressão de permissão).
    try:
        orig_mode: int | None = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        orig_mode = None  # arquivo novo → mantém 0600 do mkstemp (privado)

    # Temporário no MESMO diretório → garante rename intra-FS no os.replace.
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())  # durabilidade — alguns FS não suportam
            except OSError:
                pass  # fsync best-effort; a atomicidade vem do os.replace abaixo
        if orig_mode is not None:
            try:
                os.chmod(tmp_path, orig_mode)  # restaura perms ANTES do replace
            except OSError:
                pass  # best-effort; pior caso o arquivo fica 0600 (mais restrito)
        os.replace(tmp_path, target)  # troca ATÔMICA (POSIX rename)
    except BaseException:
        # Falha em qualquer passo → remove o temporário; destino antigo intacto.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
