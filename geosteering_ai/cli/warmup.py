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
# ║  Dependências: geosteering_ai.cli._main (lazy — só após env setup)         ║
# ║  Padrão      : entry-point pip-installable (`geosteering-warmup`)         ║
# ║  Testes      : tests/test_cli_warmup_entry_point.py (entry-point + flags) ║
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
# ║      • Background thread em `cli/_main.py` = fire-and-forget daemon,       ║
# ║         falhas silenciosas, não bloqueia o usuário.                       ║
# ║                                                                           ║
# ║    Reusa `_warmup_numba_tier2_sync` extraído em `cli/_main.py` (Fase 2     ║
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
    Geosteering AI CLI v2.56

    # Cenário CI: warmup isolado, depois benchmark com cache quente
    $ geosteering-warmup --verbose
    $ geosteering-cli benchmark --scenario E --n 200

Padrão de design:
    - argparse (alinhado a ``cli/_main.py``)
    - env vars setadas em escopo de módulo ANTES de qualquer import JAX
      (idempotente com ``cli/_main.py``)
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
# Sprint v2.32 — Env vars (espelho de cli/_main.py)
# ──────────────────────────────────────────────────────────────────────────────
# Estas duas variáveis DEVEM ser setadas antes de qualquer `import jax` ou
# `import numba`. Quando o entry point é invocado via console_script do pip,
# Python só executa este módulo após resolver `geosteering_ai.cli.warmup:main`
# — sem chance de importar `cli/_main.py` antes. Por isso duplicamos o setup
# (idempotente: `setdefault` + `if not in environ`).
#
# Sprint v2.51: capturamos o valor ORIGINAL de JAX_PLATFORMS ANTES do setdefault
# para distinguir "usuário forçou CPU" de "default CPU". O warmup --jax/--gpu
# precisa liberar a GPU (remover o force-CPP default) — mas só se o usuário NÃO
# setou explicitamente (respeita override).
_USER_JAX_PLATFORMS = os.environ.get("JAX_PLATFORMS")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Sprint v2.52: NUMBA_CACHE_DIR default ESTÁVEL ~/.cache (sobrevive reboot →
# persistência cross-reboot do .nbc) com fallback $TMPDIR. Idêntico a cli/_main.py.
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
            # Permissões 0o700 — apenas o dono lê/escreve/executa (consistente
            # com `cli/_main.py`, CodeRabbit v2.31 major finding).
            os.makedirs(_candidate_numba_cache_dir, mode=0o700, exist_ok=True)
            os.chmod(_candidate_numba_cache_dir, 0o700)
            os.environ["NUMBA_CACHE_DIR"] = _candidate_numba_cache_dir
            break
        except OSError:
            continue


def _gpu_available() -> bool:
    """Detecta GPU NVIDIA via ``nvidia-smi`` SEM importar jax.

    Mesmo padrão de ``_jax/__init__.py::_setup_xla_environment`` — subprocess
    com timeout curto, falha graciosa (sem GPU / nvidia-smi ausente → False).

    Returns:
        ``True`` se ``nvidia-smi`` lista ao menos uma GPU; ``False`` caso contrário.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.SubprocessError,
        OSError,
    ):
        return False


def _resolve_jax_warmup(args, user_jax_platforms, env) -> tuple[bool, bool]:
    """Decide ``(want_jax, lift_cpu_pin)`` a partir dos args + estado de ambiente.

    Helper PURO e testável (não importa jax, não muta ``env``) que encapsula a
    lógica de maior risco da CLI: decidir se aquece o JAX-nativo e se deve
    LIBERAR o ``JAX_PLATFORMS=cpu`` default (para o jax auto-detectar a GPU).

    Args:
        args: ``argparse.Namespace`` com ``jax`` (bool) e ``auto`` (bool).
        user_jax_platforms: valor de ``JAX_PLATFORMS`` ANTES do ``setdefault``
            de módulo (``None`` se o usuário não setou). Distingue "usuário
            forçou CPU" de "default CPU".
        env: mapeamento de ambiente (``os.environ``) — só LIDO, nunca mutado.

    Returns:
        ``(want_jax, lift_cpu_pin)``:
          - ``want_jax``: ``--jax``/``--gpu`` OU (``--auto`` e GPU presente);
          - ``lift_cpu_pin``: ``True`` se deve ``del env["JAX_PLATFORMS"]`` —
            só quando ``want_jax`` E o usuário NÃO setou (``user_jax_platforms is
            None``) E o valor atual é o default ``"cpu"``.

    Note:
        Testado em ``tests/test_cli_warmup_entry_point.py`` (tabela-verdade +
        monkeypatch de ``_gpu_available``). Garante que NÃO há o ordering-bug
        que silenciaria o warmup em CPU.
    """
    want_jax = bool(args.jax)
    if getattr(args, "auto", False) and not want_jax:
        want_jax = _gpu_available()
    lift_cpu_pin = (
        want_jax and user_jax_platforms is None and env.get("JAX_PLATFORMS") == "cpu"
    )
    return want_jax, lift_cpu_pin


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
        Reusa ``_warmup_numba_tier2_sync`` do módulo
        ``geosteering_ai.cli._main``.
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
        help="Mostra timing por fase do warmup (Numba CPU + JAX GPU opcional)",
    )
    # ── Sprint v2.52 — warmup Numba CPU de cobertura completa (DEFAULT) ───────
    # JAX-independente: aquece os kernels prange de produção via simulate_multi.
    parser.add_argument(
        "--numba",
        dest="numba",
        action="store_true",
        default=True,
        help="Aquece o caminho Numba CPU de produção (default ligado).",
    )
    parser.add_argument(
        "--no-numba",
        dest="numba",
        action="store_false",
        help="Pula o warmup Numba CPU.",
    )
    parser.add_argument(
        "--numba-n-pos",
        type=int,
        default=200,
        help="(--numba) Posições do grid a aquecer. Default 200.",
    )
    parser.add_argument(
        "--numba-n-layers",
        type=int,
        default=5,
        help="(--numba) Camadas do modelo de warmup. Default 5.",
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help="(--numba) Nº de threads (cfg.num_threads). Default auto.",
    )
    # ── Sprint v2.51 — warmup JAX-nativo (kernel bucketed de produção GPU) ────
    parser.add_argument(
        "--jax",
        "--gpu",
        dest="jax",
        action="store_true",
        help=(
            "Também aquece o kernel JAX bucketed nativo de produção "
            "(use_native_dipoles=True) — pré-compila o cache JIT + popula o "
            "cache XLA de disco. Use em GPU antes do workload real."
        ),
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detecta GPU e habilita --jax automaticamente quando presente",
    )
    parser.add_argument(
        "--jax-n-layers",
        type=int,
        default=3,
        help="(--jax) Camadas do modelo de warmup. Default 3.",
    )
    parser.add_argument(
        "--jax-n-pos",
        type=int,
        default=600,
        help="(--jax) Posições do grid a aquecer (casar com produção). Default 600.",
    )
    parser.add_argument(
        "--jax-n-models",
        type=int,
        default=1,
        help="(--jax) Modelos no batch de warmup (casar com produção). Default 1.",
    )
    parser.add_argument(
        "--jax-dtype",
        choices=("complex128", "complex64"),
        default="complex128",
        help="(--jax) Precisão complexa a aquecer. Default complex128 (paridade).",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Exibe a versão da Geosteering AI CLI e sai",
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
        Logging segue o mesmo padrão de ``cli/_main.py``: ``basicConfig`` no
        nível INFO + ``propagate=False`` no logger ``jax`` para evitar
        duplicação de mensagens (hotfix v2.31 commit ``68b93c7``).

        Mensagens de progresso e versão usam ``print`` direto em stdout
        (exceção D9 documentada para CLI: stdout é parte do contrato
        observável do comando, não logging interno).

    Example:
        >>> # Em produção, chamada via entry point pip:
        >>> # $ geosteering-warmup --version
        >>> # >>> Geosteering AI CLI v2.56
    """
    # Logging consistente com cli/_main.py
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("jax").propagate = False
    logger = logging.getLogger(__name__)

    args = build_parser().parse_args(argv)

    # Import lazy — só após env vars estarem setadas
    from geosteering_ai.cli._main import GEOSTEERING_CLI_VERSION

    if args.version:
        # Contrato CLI: stdout limpo (D9 exception documentada)
        print(f"Geosteering AI CLI {GEOSTEERING_CLI_VERSION}")
        return 0

    print(f"Warming up Geosteering AI {GEOSTEERING_CLI_VERSION}...")
    t0 = time.perf_counter()

    # ── Sprint v2.51 — decide GPU warmup ANTES do 1º import jax ──────────────
    # O warmup --jax (abaixo) importa jax. Se o usuário pediu --jax/--gpu (ou
    # --auto + GPU presente), LIBERA a GPU removendo o force-CPU default (mas só
    # se o usuário não setou JAX_PLATFORMS explicitamente). O warmup Numba (v2.52)
    # é JAX-INDEPENDENTE — não importa jax.
    want_jax, lift_cpu_pin = _resolve_jax_warmup(args, _USER_JAX_PLATFORMS, os.environ)
    if lift_cpu_pin:
        del os.environ["JAX_PLATFORMS"]  # jax auto-detecta GPU/CUDA

    # ── Sprint v2.52 — warmup Numba CPU de cobertura completa (DEFAULT) ───────
    # Substitui o `_warmup_numba_tier2_sync` (callback JAX, 3-pos) que NÃO aquecia
    # os kernels prange de produção E requeria JAX. Este roda o caminho REAL
    # `simulate_multi(backend="numba")` → aquece flat/cached/precompute + inlinados.
    if args.numba:
        try:
            from geosteering_ai.simulation._numba.warmup import warmup_numba_simulator

            info = warmup_numba_simulator(
                n_layers=args.numba_n_layers,
                n_positions=args.numba_n_pos,
                threads=args.numba_threads,
                verbose=args.verbose,
            )
            if info.get("skipped"):
                print(f"[warmup-numba] pulado ({info.get('reason')})")
            else:
                print(
                    f"[warmup-numba] {len(info['functions_warmed'])} kernels em "
                    f"{info['elapsed_s']:.1f}s (threads={info['threads']})"
                )
        except (ModuleNotFoundError, ImportError) as e:
            # Backend ausente NÃO é falha — warmup é observabilidade, nunca gateia
            # o build (Sprint O4 v2.44). CI sem o extra `sim` segue.
            logger.warning(
                "Warmup Numba pulado — backend ausente (%s): %s. "
                "Instale o extra 'sim' (numba) para aquecer.",
                e.__class__.__name__,
                e,
            )
        except Exception as e:  # noqa: BLE001 — warmup nunca deve gatear o build
            logger.error("Warmup Numba parcial (%s): %s", e.__class__.__name__, e)
            return 1

    # ── Sprint v2.51 — warmup JAX-nativo (kernel bucketed de produção GPU) ────
    # Fecha o gap: o warmup Numba acima usa use_native_dipoles=False (path Numba).
    # Este pré-compila o kernel JAX nativo + popula o cache XLA de disco.
    if want_jax:
        try:
            from geosteering_ai.simulation._jax.warmup import warmup_jax_simulator

            info = warmup_jax_simulator(
                n_layers=args.jax_n_layers,
                n_positions=args.jax_n_pos,
                n_models=args.jax_n_models,
                complex_dtype=args.jax_dtype,
                verbose=args.verbose,
            )
            if info.get("skipped"):
                print(f"[warmup-jax] pulado ({info.get('reason')})")
            else:
                print(
                    f"[warmup-jax] {info['buckets_warmed']} buckets em "
                    f"{info['elapsed_s']:.1f}s "
                    f"(total_xla={info['total_xla_programs']}, "
                    f"persisted={info['persisted']})"
                )
        except (ModuleNotFoundError, ImportError) as e:
            logger.warning("Warmup JAX pulado — backend ausente: %s", e)
        except Exception as e:  # noqa: BLE001 — warmup nunca deve gatear o build
            logger.error("Warmup JAX parcial (%s): %s", e.__class__.__name__, e)
            return 1

    dt = time.perf_counter() - t0
    print(f"OK ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
