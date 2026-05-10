# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/main.py                                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Dispatcher principal da CLI (argparse top-level)           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.24 — I2.6)                              ║
# ║  Versão      : v2.24                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-10                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : argparse (Python stdlib)                                   ║
# ║  Dependências: geosteering_ai.cli.{simulate,benchmark} (lazy)             ║
# ║  Padrão      : subparsers — cada subcomando importa seu handler em        ║
# ║                runtime (ver `main()` linhas finais)                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Define o parser argparse top-level com 3 subparsers (`simulate`,       ║
# ║    `benchmark`, `version`). O dispatcher central (`main()`) seleciona     ║
# ║    o handler apropriado, importando-o preguiçosamente para que o          ║
# ║    overhead de `--help` permaneça <5s mesmo no primeiro uso.              ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SIMULATION_MANAGER_VERSION: string da versão (sincronizada manual)    ║
# ║    build_parser: constrói o ArgumentParser com todos os subparsers       ║
# ║    main: ponto de entrada (recebe argv ou sys.argv[1:])                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Dispatcher principal da CLI Geosteering AI (Sprint v2.24 — I2.6).

Define o argparse top-level com subparsers ``simulate``, ``benchmark`` e
``version``. Cada subcomando importa preguiçosamente seu handler para
evitar overhead em ``geosteering-cli --help``.
"""

from __future__ import annotations

import argparse
import logging
import sys

# Versão exibida pelo subcomando `version` — sincronizada manualmente
# com CLAUDE.md linha 16 ao final de cada sprint.
SIMULATION_MANAGER_VERSION = "v2.24"


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser argparse top-level com todos os subparsers.

    Args:
        (sem argumentos)

    Returns:
        ``argparse.ArgumentParser`` configurado com 3 subparsers
        (``simulate``, ``benchmark``, ``version``). Cada subparser
        define os argumentos específicos do seu subcomando.

    Raises:
        Não levanta exceções diretas — argparse internamente pode
        emitir ``SystemExit`` em chamadas de `--help`.

    Note:
        O parser é construído lazy: importa apenas argparse no topo.
        Os handlers (``simulate.py``, ``benchmark.py``) são importados
        dentro de ``main()`` apenas quando o subcomando é selecionado.

    Example:
        >>> parser = build_parser()
        >>> args = parser.parse_args(["version"])
        >>> args.command
        'version'
    """
    parser = argparse.ArgumentParser(
        prog="geosteering-cli",
        description=(
            "CLI MVP do Geosteering AI v2.0 — Simulation Manager "
            f"{SIMULATION_MANAGER_VERSION}. "
            "Permite executar simulações forward EM 1D TIV e benchmarks "
            "sem escrever código Python."
        ),
    )
    sub = parser.add_subparsers(
        dest="command",
        title="comandos",
        metavar="{simulate,benchmark,version}",
    )

    # ── simulate ───────────────────────────────────────────────────
    p_sim = sub.add_parser(
        "simulate",
        help="Gera modelos sintéticos via simulate_multi",
    )
    p_sim.add_argument(
        "--models",
        type=int,
        default=10,
        help="número de modelos sintéticos (default: 10)",
    )
    p_sim.add_argument(
        "--n-pos",
        type=int,
        default=600,
        help="posições por modelo (default: 600 — Inv0Dip 0°)",
    )
    p_sim.add_argument(
        "--workers",
        type=int,
        default=None,
        help="workers paralelos (default: auto-detect via Sprint v2.23 A.2)",
    )
    p_sim.add_argument(
        "--threads",
        type=int,
        default=None,
        help="threads Numba por worker (default: auto-detect)",
    )
    p_sim.add_argument(
        "--seed",
        type=int,
        default=42,
        help="semente do gerador aleatório (default: 42)",
    )
    p_sim.add_argument(
        "--out",
        type=str,
        default=None,
        help="diretório de saída (default: não grava arquivos)",
    )
    p_sim.add_argument(
        "--quiet",
        action="store_true",
        help="suprime saída informativa (mantém erros)",
    )

    # ── benchmark ──────────────────────────────────────────────────
    p_bench = sub.add_parser(
        "benchmark",
        help="Executa cenário de benchmark e reporta throughput",
    )
    p_bench.add_argument(
        "--scenario",
        choices=["A", "B", "C", "D", "E", "F"],
        default="A",
        help="cenário canônico (A=padrão, E=Inv0Dip)",
    )
    p_bench.add_argument(
        "--n",
        type=int,
        default=100,
        help="número de modelos do benchmark (default: 100)",
    )
    p_bench.add_argument(
        "--workers",
        type=int,
        default=None,
        help="workers paralelos (default: auto-detect)",
    )
    p_bench.add_argument(
        "--threads",
        type=int,
        default=None,
        help="threads por worker (default: auto-detect)",
    )

    # ── version ────────────────────────────────────────────────────
    sub.add_parser(
        "version",
        help="Exibe versão do Simulation Manager",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI Geosteering AI.

    Configura logging centralizado e despacha para o handler do
    subcomando selecionado via ``argparse``.

    Args:
        argv: Lista de argumentos de linha de comando. Se ``None``,
            usa ``sys.argv[1:]`` automaticamente. Útil para passar
            argumentos em testes.

    Returns:
        Exit code:

        - 0: sucesso
        - 1: erro do handler (modelo inválido, OSError, etc.)
        - 2: argumento inválido ou subcomando não fornecido

    Raises:
        Não propaga exceções — todos os erros do handler retornam
        exit code 1 com mensagem em ``logger.error``.

    Note:
        Imports dos handlers (``simulate.py``, ``benchmark.py``) são
        feitos dentro deste corpo para que ``--help`` permaneça
        rápido (<5s) mesmo no primeiro uso (numba é caro de importar).

    Example:
        >>> import sys
        >>> # Em produção, chamada via entry point pip:
        >>> # $ geosteering-cli version
        >>> # >>> Geosteering AI Simulation Manager v2.24
    """
    # Logging centralizado (W4 do code-review): config no main, não nos
    # handlers — evita reconfigurar logger raiz quando handlers forem
    # importados como biblioteca por código externo.
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        return 2

    if args.command == "version":
        # CLI: stdout limpo é parte do contrato (D9 exception documentada)
        print(f"Geosteering AI Simulation Manager {SIMULATION_MANAGER_VERSION}")
        return 0

    if args.command == "simulate":
        from geosteering_ai.cli.simulate import handle_simulate

        return handle_simulate(args)

    if args.command == "benchmark":
        from geosteering_ai.cli.benchmark import handle_benchmark

        return handle_benchmark(args)

    parser.error(f"comando desconhecido: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
