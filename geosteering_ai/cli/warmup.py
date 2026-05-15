# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/warmup.py                                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Entry point standalone — warmup síncrono do cache JIT/LLVM ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.32 — geosteering-warmup)                ║
# ║  Versão      : v2.32                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-13                                                 ║
# ║  Última Mod. : 2026-05-13                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Licença     : MIT (idem projeto)                                         ║
# ║  Framework   : argparse (Python stdlib)                                   ║
# ║  Dependências: geosteering_ai.cli.main (lazy — só após env setup)         ║
# ║  Padrão      : entry-point pip-installable (`geosteering-warmup`)         ║
# ║  Testes      : tests/test_cli_warmup_entry_point.py (4 testes)            ║
# ║  Revisão     : CodeRabbit 2 iterações — 0 findings críticos/major          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Comando standalone que executa o warmup do cache JIT/LLVM de forma     ║
# ║    SÍNCRONA e BLOQUEANTE — o processo retorna ao shell só após o          ║
# ║    aquecimento completar. Casos de uso:                                   ║
# ║                                                                           ║
# ║      • CI: pre-step antes de `geosteering-cli benchmark` para isolar     ║
# ║        cold-start do tempo de execução medido.                            ║
# ║      • Notebooks: executar antes do primeiro `simulate_multi` para que    ║
# ║        a primeira chamada não pague o custo de compilação JIT.            ║
# ║      • Debug: timing visível por fase com `--verbose`.                    ║
# ║                                                                           ║
# ║    Diferença vs background thread em `geosteering-cli`:                   ║
# ║      • `geosteering-warmup` = SÍNCRONO, bloqueia até terminar, exit code  ║
# ║         indica sucesso (0) ou falha parcial (1).                          ║
# ║      • Background thread em `cli/main.py` = fire-and-forget daemon,       ║
# ║         falhas silenciosas, não bloqueia o usuário.                       ║
# ║                                                                           ║
# ║    Reusa `_warmup_numba_tier2_sync` extraído em `cli/main.py` (Fase 2     ║
# ║    do Sprint v2.32).                                                      ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    build_parser: ArgumentParser com flags --verbose e --version          ║
# ║    main: ponto de entrada (recebe argv ou sys.argv[1:])                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Entry point standalone para warmup síncrono do cache JIT/LLVM (Sprint v2.32).

Comando dedicado que aquece o cache de compilação Numba + LLVM Tier 2 e
retorna ao shell. Existe um único subcomando (warmup padrão) com duas flags
de modulação: ``--verbose`` (timing por fase) e ``--version`` (exibe versão).

Exemplos de uso::

    $ geosteering-warmup
    Warming up Geosteering AI v2.32...
    OK (12.4s)

    $ geosteering-warmup --verbose
    Warming up Geosteering AI v2.32...
      [warmup] filter loaded (0.31s)
      [warmup] JAX callback path warm (12.18s)
    OK (12.2s)

    $ geosteering-warmup --version
    Geosteering AI Simulation Manager v2.32

    # Cenário CI: warmup isolado, depois benchmark com cache quente
    $ geosteering-warmup --verbose
    $ geosteering-cli benchmark --scenario E --n 200

Padrão de design:
    - argparse (alinhado a ``cli/main.py``)
    - env vars setadas em escopo de módulo ANTES de qualquer import JAX
      (idempotente com ``cli/main.py``)
    - lazy import da função de warmup (após env setup)
    - exit code:
        * 0 = warmup completou
        * 1 = warmup falhou (exceção propagada do core síncrono)
        * 2 = erro de argumento (argparse)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time

# Símbolos públicos exportados (D8)
__all__ = ["build_parser", "main"]


# ──────────────────────────────────────────────────────────────────────────────
# Sprint v2.32 — Env vars (espelho de cli/main.py)
# ──────────────────────────────────────────────────────────────────────────────
# Estas duas variáveis DEVEM ser setadas antes de qualquer `import jax` ou
# `import numba`. Quando o entry point é invocado via console_script do pip,
# Python só executa este módulo após resolver `geosteering_ai.cli.warmup:main`
# — sem chance de importar `cli/main.py` antes. Por isso duplicamos o setup
# (idempotente: `setdefault` + `if not in environ`).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

if "NUMBA_CACHE_DIR" not in os.environ:
    _default_numba_cache_dir = os.path.join(
        tempfile.gettempdir(), "geosteering_numba_cache"
    )
    try:
        # Permissões 0o700 — apenas o dono lê/escreve/executa (consistente
        # com `cli/main.py`, CodeRabbit v2.31 major finding).
        os.makedirs(_default_numba_cache_dir, mode=0o700, exist_ok=True)
        os.chmod(_default_numba_cache_dir, 0o700)
        os.environ["NUMBA_CACHE_DIR"] = _default_numba_cache_dir
    except OSError:
        # Falha rara (permissões em /tmp) — Numba cai no default $CWD/__pycache__
        pass


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser argparse do entry point ``geosteering-warmup``.

    Args:
        Nenhum.

    Returns:
        ``argparse.ArgumentParser`` com duas flags mutuamente compatíveis:
        ``--verbose`` (timing por fase) e ``--version`` (exibir versão e sair).

    Raises:
        Nenhuma exceção é lançada diretamente. ``argparse`` pode levantar
        ``SystemExit`` em chamadas subsequentes a ``parse_args()`` quando
        argumentos inválidos forem fornecidos (não nesta função).

    Example:
        >>> parser = build_parser()
        >>> args = parser.parse_args(["--version"])
        >>> args.version
        True

    Note:
        Entry point relacionado: ``geosteering-warmup`` (registrado em
        ``pyproject.toml`` como ``geosteering_ai.cli.warmup:main``).
        Reusa ``_warmup_numba_tier2_sync`` do Simulation Manager (módulo
        ``geosteering_ai.cli.main``).
    """
    parser = argparse.ArgumentParser(
        prog="geosteering-warmup",
        description=(
            "Aquece sincronicamente o cache JIT/LLVM do simulador Geosteering "
            "AI. Útil em CI/notebooks para isolar o cold-start do tempo de "
            "simulação medido. Sprint v2.32."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Mostra timing por fase do warmup (filter load, JAX callback)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Exibe versão do Simulation Manager e sai",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI ``geosteering-warmup``.

    Args:
        argv: Lista de argumentos. Se ``None``, usa ``sys.argv[1:]``.

    Returns:
        Exit code:

        - ``0``: warmup completou com sucesso (ou ``--version`` retornado).
        - ``1``: warmup falhou (exceção propagada do core síncrono); a
          mensagem da exceção é logada via ``logger.error`` em stderr.
        - ``2``: argumento inválido (raised por ``argparse``).

    Raises:
        SystemExit: lançada por ``argparse.parse_args()`` com código ``2``
            quando argumentos inválidos são fornecidos. Não é capturada
            aqui — o caller padrão (entry point) propaga o exit code.

    Note:
        Logging segue o mesmo padrão de ``cli/main.py``: ``basicConfig`` no
        nível INFO + ``propagate=False`` no logger ``jax`` para evitar
        duplicação de mensagens (hotfix v2.31 commit ``68b93c7``).

        Mensagens de progresso e versão usam ``print`` direto em stdout
        (exceção D9 documentada para CLI: stdout é parte do contrato
        observável do comando, não logging interno).

    Example:
        >>> # Em produção, chamada via entry point pip:
        >>> # $ geosteering-warmup --version
        >>> # >>> Geosteering AI Simulation Manager v2.32
    """
    # Logging consistente com cli/main.py
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("jax").propagate = False
    logger = logging.getLogger(__name__)

    args = build_parser().parse_args(argv)

    # Import lazy — só após env vars estarem setadas
    from geosteering_ai.cli._main import SIMULATION_MANAGER_VERSION

    if args.version:
        # Contrato CLI: stdout limpo (D9 exception documentada)
        print(f"Geosteering AI Simulation Manager {SIMULATION_MANAGER_VERSION}")
        return 0

    print(f"Warming up Geosteering AI {SIMULATION_MANAGER_VERSION}...")
    t0 = time.perf_counter()

    from geosteering_ai.cli._main import _warmup_numba_tier2_sync

    try:
        _warmup_numba_tier2_sync(verbose=args.verbose)
    except Exception as e:
        # Diferente do background thread (silencioso), aqui propagamos a
        # falha ao usuário via logger.error (D9) + exit code 1. CI detecta
        # tanto pelo exit code quanto pela mensagem em stderr.
        logger.error("Warmup parcial (%s): %s", e.__class__.__name__, e)
        return 1

    dt = time.perf_counter() - t0
    print(f"OK ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
