# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/main.py                                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Dispatcher principal da CLI (argparse top-level)           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.56 — wall-clock JAX + --jax-chunk-size)  ║
# ║  Versão      : v2.56                                                      ║
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
# ║    GEOSTEERING_CLI_VERSION: string da versão (sincronizada manual)       ║
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

# ──────────────────────────────────────────────────────────────────────────────
# Sprint v2.55 — Teto de threads ANTES de qualquer import pesado (mitigação TLS)
# ──────────────────────────────────────────────────────────────────────────────
# Quando o backend JAX inicializa o CUDA no MESMO processo (cuBLAS/cuDNN/NCCL via
# dlopen consomem o *surplus de TLS estático* fixo do glibc) e, em seguida, o Numba
# `@njit(parallel=True)/prange` cria seu pool libgomp com `NUMBA_NUM_THREADS` threads
# (default = os.cpu_count() lógico = 64 no Threadripper), cada thread aloca um slot
# de TLS estático via `_dl_allocate_tls_init` → ESTOURA → crash
# `Inconsistency detected by ld.so: ... Assertion 'listp != NULL' failed`.
# Limitar NUMBA_NUM_THREADS (a cores FÍSICOS — sem oversubscription de HT) reduz o
# nº de allocs de TLS para caber no surplus. `OMP/OPENBLAS=1` cortam runtimes
# concorrentes que também consomem TLS no re-import. `setdefault` preserva override
# do usuário. NÃO regride o throughput do POOL: `_workers._acquire_pool` sobrescreve
# `NUMBA_NUM_THREADS` por worker antes do spawn. Espelha
# `scripts/diagnose_numba_warmup.py` (mitigação validada por bissecção: 4 threads OK,
# 64 CRASHA). Ref: relatório v2.55.
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# NUMBA_CACHE_DIR: `cache=True` em @njit armazena LLVM bitcode (.nbc), não código
# de máquina. Cada novo processo Python ainda recompila o bitcode → assembly nativo
# no backend LLVM (~111 s para 16 .nbc no projeto). Sprint v2.52 (warmup Numba): o
# default vira ESTÁVEL `~/.cache/geosteering/numba_cache` (sobrevive REBOOT) em vez
# de tmpfs (efêmero — apagado no reboot → 1º run pós-reboot re-pagava a compilação
# COMPLETA ~300-600 s; com dir estável o .nbc persiste → pós-reboot ~111 s). Fallback
# gracioso p/ `$TMPDIR/geosteering_numba_cache` se o home não for gravável (CI/
# containers). Pode ser sobrescrito via `export NUMBA_CACHE_DIR=`.
if "NUMBA_CACHE_DIR" not in os.environ:
    _stable_numba_cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "geosteering", "numba_cache"
    )
    _fallback_numba_cache_dir = os.path.join(
        tempfile.gettempdir(), "geosteering_numba_cache"
    )
    for _candidate_numba_cache_dir in (
        _stable_numba_cache_dir,
        _fallback_numba_cache_dir,
    ):
        try:
            # Permissões 0o700: somente o dono lê/escreve/executa — evita que
            # outros usuários em sistemas multi-tenant injetem .nbc maliciosos
            # (CodeRabbit major finding v2.31). chmod explícito após makedirs
            # blinda contra `umask` ou diretório pré-existente com permissões
            # frouxas (compat: mantém o cache se já existe e ajusta o modo).
            os.makedirs(_candidate_numba_cache_dir, mode=0o700, exist_ok=True)
            os.chmod(_candidate_numba_cache_dir, 0o700)
            os.environ["NUMBA_CACHE_DIR"] = _candidate_numba_cache_dir
            break
        except OSError:
            # Candidato não-gravável — tenta o próximo; se ambos falham, Numba
            # cai no default $CWD/__pycache__ (raro).
            continue

# Versão exibida pelo subcomando `version` — sincronizada manualmente
# com CLAUDE.md linha 16 ao final de cada sprint.
#
# NOTA DE PRODUTO (2026-06-05): a CLI ("Geosteering AI CLI") e o Simulation
# Manager são produtos SEPARADOS. Esta constante reporta a versão da CLI; o
# nome do produto exibido ao usuário é "Geosteering AI CLI" (nunca mais
# "Simulation Manager"). A unificação do esquema de versionamento entre os
# produtos é decisão pendente (QP1 do relatório de portfólio).
GEOSTEERING_CLI_VERSION = "v2.56"


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
    """Wrapper silencioso de warmup Numba para daemon thread.

    Sprint v2.52: aponta para o warmup Numba de cobertura completa
    (``warmup_numba_simulator``, JAX-independente, aquece os kernels prange de
    produção) em vez do legado ``_warmup_numba_tier2_sync`` (callback JAX, 3-pos).
    Usa ``n_positions`` pequeno (64) para um daemon barato. Captura toda exceção:
    warmup é best-effort, não crítico para a CLI — falhas (e.g. backend ausente)
    não devem propagar para o usuário que apenas invocou ``geosteering-cli simulate``.

    Note:
        Para warmup com erros visíveis ao usuário, use o entry point
        ``geosteering-warmup`` (Sprint v2.32+), que chama o warmup diretamente
        e propaga exceções.
    """
    try:
        from geosteering_ai.simulation._numba.warmup import warmup_numba_simulator

        warmup_numba_simulator(n_layers=5, n_positions=64, verbose=False)
    except Exception:
        # Silent fail — warmup é best-effort, não crítico para CLI
        pass


def _add_common_backend_args(p: argparse.ArgumentParser) -> None:
    """Adiciona os argumentos de backend/observabilidade comuns (v2.53).

    Compartilhados entre ``simulate`` e ``benchmark`` para manter a UX
    consistente (DRY): seleção de backend, dtype/estratégia JAX, warmup,
    saída JSON, repetições e comparação de backends.

    Args:
        p: Subparser (``simulate`` ou ``benchmark``) a receber os argumentos.

    Returns:
        None. Efeito colateral: registra os argumentos em ``p``.
    """
    p.add_argument(
        "--backend",
        choices=["numba", "jax", "auto"],
        # Sentinela: None = usuário NÃO escolheu → default implícito "numba" +
        # DeprecationWarning (spec 0003). Escolha explícita silencia o aviso.
        default=None,
        help=(
            "backend de simulação: numba (CPU) | jax (GPU) | auto (escolhe via a "
            "árvore medida do dispatcher — GPU+n≥32+geometria agrupável → jax, "
            "senão numba). Default atual: numba (com aviso de deprecação); o "
            "default mudará para 'auto' em v2.57.0."
        ),
    )
    p.add_argument(
        "--dtype",
        choices=["complex128", "complex64"],
        default="complex128",
        help="dtype complexo do caminho JAX (default: complex128 — paridade)",
    )
    p.add_argument(
        "--jax-strategy",
        choices=["bucketed", "unified"],
        default="bucketed",
        help="estratégia do kernel JAX (default: bucketed — seguro/anti-OOM)",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help="aquece o backend (JIT/XLA) antes da medição cronometrada",
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emite os resultados como JSON no stdout (além/no lugar da tabela)",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="N execuções hot + mediana do throughput (default: 1)",
    )
    p.add_argument(
        "--compare-backends",
        action="store_true",
        help="roda numba E jax lado-a-lado (throughput, speedup, paridade max|Δ|)",
    )
    # ── Geometria dos modelos sintéticos (batchabilidade no JAX) — v2.54 ──
    # O JAX-grouped só satura a GPU quando muitos modelos COMPARTILHAM geometria
    # (esp). 'per-model' (default) = esp único por modelo → no JAX cai p/ Numba
    # (não-agrupável). 'templates'/'quantized' criam compartilhamento → JAX vence.
    p.add_argument(
        "--geometry",
        choices=["per-model", "templates", "quantized"],
        default="per-model",
        help=(
            "amostragem de geometria (esp): per-model (padrão, esp único/modelo) "
            "| templates (K geometrias replicadas → JAX agrupável/rápido) "
            "| quantized (esp arredondado → agrupável parcial)"
        ),
    )
    p.add_argument(
        "--n-geometries",
        type=int,
        default=None,
        metavar="K",
        help=(
            "(só --geometry templates) nº de geometrias distintas K "
            "(default: max(1, n_models//32))"
        ),
    )
    p.add_argument(
        "--quantize-step",
        type=float,
        default=None,
        metavar="M",
        help="(só --geometry quantized) passo de quantização de esp em metros",
    )
    p.add_argument(
        "--jax-chunk-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "(só JAX) fragmenta o eixo de modelos do vmap em chunks de N "
            "(anti-OOM em high-config G/H). Default: auto (64 em high-config)"
        ),
    )


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
            "Geosteering AI CLI "
            f"{GEOSTEERING_CLI_VERSION}. "
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
        "--format",
        dest="out_format",
        choices=["npz", "dat", "none"],
        default="npz",
        help=(
            "formato de gravação quando --out é fornecido: npz (default), "
            "dat (.dat/.out 22-col Fortran-compat) ou none (só tabela)"
        ),
    )
    p_sim.add_argument(
        "--quiet",
        action="store_true",
        help="suprime saída informativa e a tabela de resultados (mantém erros)",
    )
    _add_common_backend_args(p_sim)

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
    p_bench.add_argument(
        "--list-scenarios",
        action="store_true",
        help="lista os cenários disponíveis (A–H) e sai",
    )
    p_bench.add_argument(
        "--quiet",
        action="store_true",
        help="suprime a tabela de resultados (mantém a linha grep-able mod/h)",
    )
    _add_common_backend_args(p_bench)

    # ── version ────────────────────────────────────────────────────
    sub.add_parser(
        "version",
        help="Exibe a versão da Geosteering AI CLI",
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
        >>> # >>> Geosteering AI CLI v2.56
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
        print(f"Geosteering AI CLI {GEOSTEERING_CLI_VERSION}")
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
