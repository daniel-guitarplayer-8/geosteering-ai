# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/main.py                                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Dispatcher principal da CLI (argparse top-level)           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.31 — warmup bg thread)                  ║
# ║  Versão      : v2.31                                                      ║
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
import os
import sys
import tempfile
import threading

# ──────────────────────────────────────────────────────────────────────────────
# Sprint v2.31 — Variáveis de ambiente: NUMBA_CACHE_DIR + JAX_PLATFORMS
# ──────────────────────────────────────────────────────────────────────────────
# JAX_PLATFORMS=cpu: impede que o JAX 0.4+ sonde backends ausentes (ROCM, TPU,
# CUDA) durante a inicialização. Sem este env var, cada processo Python emite
# mensagens INFO "Unable to initialize backend 'rocm'" e "Unable to initialize
# backend 'tpu'" no stderr — ruído para usuários CLI sem GPU ROCM/TPU.
# Deve ser setado ANTES de qualquer `import jax` (inclusive via importação
# transitiva — e.g. `_jax/kernel.py`). `setdefault` preserva override manual
# do usuário via `export JAX_PLATFORMS=cuda` ou `JAX_PLATFORMS=metal`.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# NUMBA_CACHE_DIR em tmpfs: `cache=True` em @njit armazena LLVM bitcode (.nbc),
# não código de máquina. Cada novo processo Python ainda recompila o bitcode →
# assembly nativo no backend LLVM (~111 s para 16 .nbc no projeto). Apontar
# NUMBA_CACHE_DIR para /tmp (tmpfs no macOS/Linux) mantém os .nbc em memória
# após a primeira leitura, reduzindo I/O em invocações subsequentes. Impacto:
# −10-30 s em SSD; mais em HDD. Pode ser sobrescrito via `export NUMBA_CACHE_DIR=`.
if "NUMBA_CACHE_DIR" not in os.environ:
    _default_numba_cache_dir = os.path.join(
        tempfile.gettempdir(), "geosteering_numba_cache"
    )
    try:
        # Permissões 0o700: somente o dono lê/escreve/executa — evita que
        # outros usuários em sistemas multi-tenant injetem .nbc maliciosos
        # (CodeRabbit major finding v2.31). chmod explícito após makedirs
        # blinda contra `umask` ou diretório pré-existente com permissões
        # frouxas (compat: mantém o cache se já existe e ajusta o modo).
        os.makedirs(_default_numba_cache_dir, mode=0o700, exist_ok=True)
        os.chmod(_default_numba_cache_dir, 0o700)
        os.environ["NUMBA_CACHE_DIR"] = _default_numba_cache_dir
    except OSError:
        # Falha rara (permissões em /tmp) — Numba cai no default $CWD/__pycache__
        pass

# Versão exibida pelo subcomando `version` — sincronizada manualmente
# com CLAUDE.md linha 16 ao final de cada sprint.
SIMULATION_MANAGER_VERSION = "v2.31"


# ──────────────────────────────────────────────────────────────────────────────
# Sprint v2.31 Part 2 — Background warmup thread para LLVM Tier 2 offline
# ──────────────────────────────────────────────────────────────────────────────
def _warmup_numba_tier2_background() -> None:
    """Aquece Tier 2 LLVM offline via JAX callback path.

    Dispara simulação sintética rápida (3 pontos de profundidade, 1 frequência,
    isotropia) para forçar compilação JIT de `hmd_tiv`/`vmd` e acionamento do
    otimizador Tier 2 do LLVM backend em background. Roda em thread daemon
    para não bloquear CLI.

    Note:
        Sprint v2.31 Part 2 — Mitigação para oscilação Run 2 observada em
        5 runs empíricos. Tier 2 (PGO) roda background durante Run 2,
        contendendo com computação user + macOS thermal throttle. Pre-aquecendo
        Tier 2 offline permite que optimize sem contenda.

        Impacto esperado: Run 1 mantém ~65-75s (Tier 1 inevitable), mas Run 2+
        beneficiam de Tier 2 pré-otimizado → consistência desde início.
    """
    try:
        import numpy as np

        from geosteering_ai.simulation._jax.kernel import fields_in_freqs_jax_batch
        from geosteering_ai.simulation.filters import FilterLoader

        filt = FilterLoader().load("werthmuller_201pt")
        rho_h = np.array([10.0] * 5)
        rho_v = np.array([10.0] * 5)
        esp = np.array([5.0] * 3)
        positions_z = np.linspace(-1.0, 6.0, 3)
        freqs_hz = np.array([20000.0])

        # Dispara JAX callback path (aquece Numba + Tier 2 LLVM)
        fields_in_freqs_jax_batch(
            positions_z=positions_z,
            dz_half=0.5,
            r_half=0.0,
            dip_rad=0.0,
            n=5,
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            freqs_hz=freqs_hz,
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
            use_native_dipoles=False,
        )
    except Exception:
        # Silent fail — warmup é best-effort, não crítico para CLI
        pass


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
        "--frequencies",
        type=str,
        default=None,
        metavar="HZ",
        help=(
            "frequências EM em Hz separadas por vírgula "
            "(ex: 2000,20000,100000). Default: 20000"
        ),
    )
    p_sim.add_argument(
        "--dips",
        type=str,
        default=None,
        metavar="DEG",
        help=(
            "ângulos de inclinação em graus separados por vírgula "
            "(ex: 0,15,30,45). Default: 0"
        ),
    )
    p_sim.add_argument(
        "--tr-spacings",
        type=str,
        default=None,
        metavar="M",
        help=(
            "espaçamentos transmissor-receptor em metros separados por vírgula "
            "(ex: 0.5,1.0,1.5,2.0). Default: 1.0"
        ),
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
        choices=["A", "B", "C", "D", "E", "F", "G"],
        default="A",
        help=(
            "cenário canônico (A=padrão, E=Inv0Dip, "
            "G=máxima combinatória 4freq×4TR×4dips)"
        ),
    )
    p_bench.add_argument(
        "--frequencies",
        type=str,
        default=None,
        metavar="HZ",
        help=(
            "sobrescreve frequências do cenário — Hz separados por vírgula "
            "(ex: 2000,20000). Default: usa freqs do cenário"
        ),
    )
    p_bench.add_argument(
        "--dips",
        type=str,
        default=None,
        metavar="DEG",
        help=(
            "sobrescreve ângulos de dip do cenário — graus separados por vírgula "
            "(ex: 0,15,30). Default: usa dips do cenário"
        ),
    )
    p_bench.add_argument(
        "--tr-spacings",
        type=str,
        default=None,
        metavar="M",
        help=(
            "sobrescreve espaçamentos T-R do cenário — metros separados por vírgula "
            "(ex: 0.5,1.0,1.5). Default: usa TRs do cenário"
        ),
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

    # Evita duplicação de mensagens do logger 'jax': o JAX instala um
    # StreamHandler próprio; com propagate=True (padrão), cada mensagem seria
    # emitida pelo handler do 'jax' E pelo handler raiz do basicConfig — 2×.
    # Desabilitar propagate corta o segundo caminho sem suprimir o primeiro.
    logging.getLogger("jax").propagate = False

    # Sprint v2.31 Part 2 — Inicializar background warmup thread (daemon)
    # para pré-aquecer LLVM Tier 2 offline, reduzindo contenda com user work
    threading.Thread(target=_warmup_numba_tier2_background, daemon=True).start()

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
