# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/main.py                                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Dispatcher principal da CLI (argparse top-level)           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.32 — entry point geosteering-warmup)    ║
# ║  Versão      : v2.32                                                      ║
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
import time

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

# Sprint v2.55 — guard TLS-safe: pinar threads de BLAS/OpenMP ANTES de qualquer
# import de numba/jax. OMP/OPENBLAS=1 impede que o pool de threads da BLAS consuma
# o surplus de TLS estático — o que faria o init do threading-layer (libgomp) do
# Numba estourar com `_dl_allocate_tls_init` quando o CUDA do JAX já alocou seu TLS
# (cenário do usuário: `JAX_PLATFORMS=cuda ... --backend jax`). NUMBA_NUM_THREADS é
# pinado (≥1) p/ fixar o pool prange. `setdefault` preserva override manual.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count() or 1))

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
SIMULATION_MANAGER_VERSION = "v2.37"


# ──────────────────────────────────────────────────────────────────────────────
# Sprint v2.32 — Warmup core síncrono (reutilizado por background thread +
# entry point `geosteering-warmup`)
# ──────────────────────────────────────────────────────────────────────────────
def _warmup_numba_tier2_sync(verbose: bool = False) -> None:
    """Aquece Tier 2 LLVM via JAX callback path (síncrono).

    Dispara simulação sintética rápida (3 pontos de profundidade, 1 frequência,
    isotropia) para forçar compilação JIT de `hmd_tiv`/`vmd` e acionamento do
    otimizador Tier 2 do LLVM backend. Função puramente síncrona: bloqueia
    até completar e propaga exceções ao chamador.

    Args:
        verbose: Se ``True``, imprime timing por fase (load filtro, JAX
            callback, total). Útil para o entry point ``geosteering-warmup``.

    Raises:
        Exception: propaga qualquer erro de importação ou simulação. O
            chamador decide se trata silenciosamente (background) ou expõe
            ao usuário (entry point standalone).

    Note:
        Sprint v2.31 Part 2 / Sprint v2.32 — Mitigação para oscilação Run 2
        observada em 5 runs empíricos. Tier 2 (PGO) roda background durante
        Run 2, contendendo com computação user + macOS thermal throttle.
        Pre-aquecendo Tier 2 offline permite otimização sem contenda.

        Impacto esperado: Run 1 mantém ~65-75s (Tier 1 inevitable), mas Run 2+
        beneficiam de Tier 2 pré-otimizado → consistência desde início.
    """
    import numpy as np

    t0 = time.perf_counter()

    from geosteering_ai.simulation._jax.kernel import fields_in_freqs_jax_batch
    from geosteering_ai.simulation.filters import FilterLoader

    filt = FilterLoader().load("werthmuller_201pt")
    if verbose:
        print(f"  [warmup] filter loaded ({time.perf_counter() - t0:.2f}s)")

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
    if verbose:
        print(f"  [warmup] JAX callback path warm ({time.perf_counter() - t0:.2f}s)")


def _warmup_numba_tier2_background() -> None:
    """Wrapper silencioso de ``_warmup_numba_tier2_sync`` para daemon thread.

    Captura toda exceção: warmup é best-effort, não crítico para a CLI.
    Falhas (e.g. JAX ausente) não devem propagar para o usuário que apenas
    invocou ``geosteering-cli simulate``.

    Note:
        Para warmup com erros visíveis ao usuário, use o entry point
        ``geosteering-warmup`` (Sprint v2.32), que chama
        ``_warmup_numba_tier2_sync`` diretamente e propaga exceções.
    """
    try:
        _warmup_numba_tier2_sync(verbose=False)
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
    # ── Backend + geometria + timing (spec 0003 + v2.53-v2.56) ─────
    p_sim.add_argument(
        "--backend",
        choices=["numba", "jax", "auto"],
        default=None,
        help=(
            "motor de simulação: numba (CPU) | jax (GPU/CPU) | auto (decide "
            "pela árvore do dispatcher). Default implícito: numba (com aviso "
            "de deprecação — mudará p/ auto). Ver spec 0003."
        ),
    )
    p_sim.add_argument(
        "--geometry",
        choices=["per-model", "templates", "quantized"],
        default="per-model",
        help=(
            "amostragem da geometria (esp): per-model (default, único por modelo) "
            "| templates (poucas geometrias distintas → agrupável p/ JAX) | "
            "quantized (esp arredondada → agrupável parcial)"
        ),
    )
    p_sim.add_argument(
        "--warmup",
        action="store_true",
        help="aquece o backend (JIT/XLA) antes da medição (best-effort)",
    )
    p_sim.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="nº de rodadas cronometradas (reporta o melhor tempo). Default: 1",
    )
    p_sim.add_argument(
        "--dtype",
        choices=["complex64", "complex128"],
        default="complex128",
        help="dtype complexo do path JAX (default: complex128)",
    )
    p_sim.add_argument(
        "--jax-strategy",
        choices=["bucketed", "unified", "vmap_real"],
        default="bucketed",
        help="estratégia de paralelismo JAX (default: bucketed)",
    )
    p_sim.add_argument(
        "--jax-chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="fragmenta o eixo de modelos do vmap JAX (anti-OOM). None = auto",
    )
    p_sim.add_argument(
        "--format",
        choices=["npz", "dat", "none"],
        default="npz",
        help=(
            "formato de saída quando --out é dado: npz (tensor H) | dat (22-col "
            "Fortran-compat + .out) | none (não grava). Default: npz"
        ),
    )
    p_sim.add_argument(
        "--json",
        action="store_true",
        help="emite o resultado (backend efetivo, timing) em JSON no stdout",
    )
    p_sim.add_argument(
        "--compare-backends",
        action="store_true",
        help="roda numba E jax lado-a-lado: throughput, speedup e paridade max|Δ|",
    )

    # ── benchmark ──────────────────────────────────────────────────
    p_bench = sub.add_parser(
        "benchmark",
        help="Executa cenário de benchmark e reporta throughput",
    )
    p_bench.add_argument(
        "--scenario",
        choices=["A", "B", "C", "D", "E", "F", "G", "H"],
        default="A",
        help=(
            "cenário canônico (A=padrão, E=Inv0Dip, "
            "G=máxima combinatória 4freq×4TR×4dips, "
            "H=estresse multi-core 8freq×8TR×8dips)"
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
    # ── Backend + saída (spec 0003 + v2.53) ────────────────────────
    p_bench.add_argument(
        "--backend",
        choices=["numba", "jax", "auto"],
        default=None,
        help=(
            "motor: numba (CPU) | jax (GPU/CPU) | auto (árvore do dispatcher). "
            "Default implícito: numba (com aviso de deprecação). Ver spec 0003."
        ),
    )
    p_bench.add_argument(
        "--compare-backends",
        action="store_true",
        help="roda numba E jax lado-a-lado: throughput, speedup e paridade max|Δ|",
    )
    p_bench.add_argument(
        "--json",
        action="store_true",
        help="emite o resultado (backend efetivo, throughput) em JSON no stdout",
    )
    p_bench.add_argument(
        "--quiet",
        action="store_true",
        help="suprime saída informativa (mantém erros e o JSON se --json)",
    )
    # ── Geometria + dtype + JAX + repeat (paridade c/ simulate) ────
    # Recuperado do archive WIP (item 3 da triagem): fecha o gap simulate↔benchmark.
    # SEM o rename de versão (a constante SIMULATION_MANAGER_VERSION permanece v2.37).
    p_bench.add_argument(
        "--geometry",
        choices=["per-model", "templates", "quantized"],
        default="per-model",
        help=(
            "amostragem da geometria (esp): per-model (default, único por modelo "
            "→ JAX degenera) | templates (poucas geometrias → agrupável, JAX satura) "
            "| quantized (esp arredondada → agrupável parcial)"
        ),
    )
    p_bench.add_argument(
        "--n-geometries",
        type=int,
        default=None,
        metavar="K",
        help="(só --geometry templates) nº de geometrias distintas. None = auto",
    )
    p_bench.add_argument(
        "--dtype",
        choices=["complex64", "complex128"],
        default="complex128",
        help="dtype complexo do path JAX (default: complex128)",
    )
    p_bench.add_argument(
        "--jax-strategy",
        choices=["bucketed", "unified", "vmap_real"],
        default="bucketed",
        help="estratégia de paralelismo JAX (default: bucketed)",
    )
    p_bench.add_argument(
        "--jax-chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="fragmenta o eixo de modelos do vmap JAX (anti-OOM). None = auto",
    )
    p_bench.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="nº de rodadas cronometradas (reporta a melhor). Default: 1",
    )
    p_bench.add_argument(
        "--list-scenarios",
        action="store_true",
        help="lista os cenários canônicos (A..H) com suas dimensões e sai",
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

    parser = build_parser()
    args = parser.parse_args(argv)

    # Sprint v2.38 — Background warmup APENAS quando não há subcomando compute-heavy.
    # `simulate` e `benchmark` spawn workers via ProcessPoolExecutor que executam
    # seu PRÓPRIO warmup determinístico em `_simulate_worker_init` (sm_workers).
    # Disparar o warmup no PAI nesses casos só contende com spawn de workers para
    # CPU/imports, adicionando 0.5-1s de jitter sem benefício (cada processo
    # worker JIT-compila ou recarrega do .nbc cache independentemente). A
    # mitigação original (Sprint v2.31 Part 2) supunha múltiplos runs no MESMO
    # processo Python — não é o caso da CLI (cada invocação = novo processo).
    if args.command not in ("simulate", "benchmark"):
        threading.Thread(target=_warmup_numba_tier2_background, daemon=True).start()

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
