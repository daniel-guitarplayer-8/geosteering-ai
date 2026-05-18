# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MÓDULO: api/cli.py                                                        ║
# ║  Bloco: 11 — API REST (NOVO em v2.39)                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning      ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: argparse + uvicorn                                             ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Docker CPU             ║
# ║  Pacote: geosteering_ai.api                                                ║
# ║                                                                            ║
# ║  Propósito:                                                                ║
# ║    • Entry point `geosteering-api` (definido em pyproject.toml)            ║
# ║    • Wrapper sobre uvicorn.run() com argparse                              ║
# ║    • Permite `geosteering-api --host 0.0.0.0 --port 8000` sem invocar      ║
# ║      uvicorn diretamente                                                   ║
# ║                                                                            ║
# ║  Dependências: argparse (stdlib), uvicorn (extra [api])                    ║
# ║  Exports: main, build_parser                                               ║
# ║  Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1        ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v2.39 (2026-05-18) — Implementação inicial (Sprint I2.7, commit 5/9)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""CLI `geosteering-api` — Wrapper sobre uvicorn.run().

Espelha o padrão de `geosteering-cli` (`geosteering_ai/cli/_main.py`):
argparse com subcomando único + lazy imports.

Exemplo:

.. code-block:: bash

    geosteering-api --host 0.0.0.0 --port 8000
    geosteering-api --reload                          # dev hot-reload
    geosteering-api --workers 4 --log-level warning   # produção multi-worker
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos para `geosteering-api`.

    Returns:
        argparse.ArgumentParser configurado.

    Note:
        Defaults alinhados com Dockerfile.cpu (host 0.0.0.0, port 8000).
    """
    parser = argparse.ArgumentParser(
        prog="geosteering-api",
        description=(
            "API REST para inferência de resistividade via Deep Learning. "
            "Wrapper sobre uvicorn para a app geosteering_ai.api.app:app."
        ),
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Endereço de bind (default: 0.0.0.0 — todas as interfaces).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Porta TCP (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Ativa hot-reload (apenas para desenvolvimento).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Número de workers uvicorn (default: 1). "
            "Ignorado se --reload estiver ativo."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Nível de log do uvicorn (default: info).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point do `geosteering-api`.

    Args:
        argv: Lista de argumentos (default: sys.argv[1:]). Útil para testes.

    Returns:
        Código de saída (0 para sucesso).

    Note:
        Lazy import de uvicorn — permite que o pacote `geosteering_ai.api.cli`
        seja importado mesmo sem as deps de [api] instaladas (apenas para
        introspecção). A execução real falha com ImportError claro se
        uvicorn não estiver disponível.
    """
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # ── Lazy import — uvicorn só é necessário ao executar ─────────
    try:
        import uvicorn
    except ImportError as exc:
        msg = (
            "uvicorn não está instalado. Instale com:\n"
            '    pip install -e ".[api]"\n'
            f"Erro: {exc}"
        )
        # Print para stderr aceitavel aqui: nao temos logging configurado
        # antes do uvicorn iniciar, e e a unica forma de sinalizar erro
        # de instalacao para o usuario na linha de comando.
        sys.stderr.write(msg + "\n")
        return 1

    logger.info(
        "Iniciando geosteering-api em %s:%d (workers=%d, reload=%s)",
        args.host,
        args.port,
        args.workers,
        args.reload,
    )

    uvicorn_kwargs: dict = {
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
    }
    if args.reload:
        uvicorn_kwargs["reload"] = True
    else:
        uvicorn_kwargs["workers"] = max(1, args.workers)

    uvicorn.run("geosteering_ai.api.app:app", **uvicorn_kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
