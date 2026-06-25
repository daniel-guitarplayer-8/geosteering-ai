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
retorna ao shell. Flags: ``--verbose`` (timing por fase), ``--version`` (exibe
versão) e o grupo JAX ``--jax``/``--jax-only``/``--jax-auto`` (op 4 — aquece a
forma canônica do SM no cache XLA; default = só Numba). O banner reporta quais
backends serão aquecidos (``numba=…, jax=…``).

Exemplos de uso::

    $ geosteering-warmup
    Warming up Geosteering AI v2.XX (numba=True, jax=False)...
    OK (12.4s)

    $ geosteering-warmup --verbose
    Warming up Geosteering AI v2.XX (numba=True, jax=False)...
      [warmup] filter loaded (0.31s)
    OK (12.2s)

    # GPU: cuda,cpu (cuda compute + cpu p/ o jax.pure_callback; só "cuda" quebra)
    $ JAX_PLATFORMS=cuda,cpu geosteering-warmup --jax-auto
    Warming up Geosteering AI v2.XX (numba=True, jax=True)...
    OK (105.1s)

    $ geosteering-warmup --version
    Geosteering AI Simulation Manager v2.XX

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


# ──────────────────────────────────────────────────────────────────────────────
# Forma CANÔNICA do SM p/ o warmup JAX (NÃO é física — só governa shapes XLA)
# ──────────────────────────────────────────────────────────────────────────────
# Espelha a config do report do SM (20 kHz · dip 0° · TR 1 m · 600 posições · 20
# camadas · werthmuller_201pt · complex128 · bucketed). n_models=64 casa o chunk
# JAX-auto do `geosteering-cli` (dim líder do HLO vmapado) → o cache de disco aquecido
# aqui é compartilhado com aquele caminho. NÃO cobre TODA forma do SM (geometrias
# estocásticas/n_models distintos compilam na 1ª sim) — ver o caveat no --help e em
# docs/reference/sm_jax_persistent_worker.md.
_SM_CANON_N_POS: int = 600
_SM_CANON_N_LAYERS: int = 20
_SM_CANON_N_MODELS: int = 64
_SM_CANON_DTYPE: str = "complex128"
_SM_CANON_STRATEGY: str = "bucketed"
_SM_CANON_FILTER: str = "werthmuller_201pt"
_SM_CANON_DIPS: tuple[float, ...] = (0.0,)
_SM_CANON_TRS: tuple[float, ...] = (1.0,)
_SM_CANON_FREQS: tuple[float, ...] = (20000.0,)


def _gpu_visible() -> bool:
    """``True`` se o JAX enxerga uma GPU (gate do ``--jax-auto``).

    Import LAZY de ``dispatch._jax_gpu_available`` — em ambiente sem o extra ``sim``
    (jax ausente) o ``ImportError`` é tratado como "sem GPU" (no-op limpo no CI CPU).
    """
    try:
        from geosteering_ai.simulation.dispatch import _jax_gpu_available

        return bool(_jax_gpu_available())
    except Exception:  # noqa: BLE001 — jax ausente / erro de init → trate como sem GPU
        return False


def _warmup_jax_canonical_sm(verbose: bool) -> dict:
    """Aquece o caminho JAX bucketed na forma CANÔNICA do SM (popula o cache de disco).

    Reusa :func:`geosteering_ai.simulation._jax.warmup.warmup_jax_simulator` (sem editar
    o kernel). Aquece o que ``JAX_PLATFORMS`` resolver (``export JAX_PLATFORMS=cuda`` p/
    a A6000; default ``cpu`` no CI/dev). Import LAZY (só após as env vars).

    Args:
        verbose: repassa o timing/diagnóstico do warmup.

    Returns:
        dict de diagnóstico de :func:`warmup_jax_simulator`.
    """
    from geosteering_ai.simulation._jax.warmup import warmup_jax_simulator

    return warmup_jax_simulator(
        n_layers=_SM_CANON_N_LAYERS,
        n_positions=_SM_CANON_N_POS,
        hankel_filter=_SM_CANON_FILTER,
        complex_dtype=_SM_CANON_DTYPE,
        dip_degs=_SM_CANON_DIPS,
        tr_spacings_m=_SM_CANON_TRS,
        freqs_hz=_SM_CANON_FREQS,
        n_models=_SM_CANON_N_MODELS,
        jax_strategy=_SM_CANON_STRATEGY,
        verbose=verbose,
    )


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
    # ── Warmup JAX (op 4) — aditivo; default = só Numba (comportamento legado) ──
    # Mutuamente exclusivo. Aquece o que JAX_PLATFORMS resolver — para a GPU use
    # `export JAX_PLATFORMS=cuda,cpu` (cuda p/ o compute + cpu p/ o jax.pure_callback
    # do kernel; SÓ "cuda" QUEBRA o callback → "failed to find a local CPU device").
    # Default cpu no CI/dev. O cache XLA aquecido é compartilhado com o SM (worker
    # persistente) e o geosteering-cli. CAVEAT: warmup homogêneo prima 1 das K
    # geometrias-template; geometrias estocásticas/n_models distintos do SM ainda
    # compilam na 1ª sim.
    jax_group = parser.add_mutually_exclusive_group()
    jax_group.add_argument(
        "--jax",
        action="store_true",
        help="Aquece TAMBÉM o JAX (forma canônica do SM), além do Numba.",
    )
    jax_group.add_argument(
        "--jax-only",
        action="store_true",
        help="Aquece SÓ o JAX (pula o Numba).",
    )
    jax_group.add_argument(
        "--jax-auto",
        action="store_true",
        help=(
            "Aquece o JAX só se houver GPU JAX visível (no-op limpo no CI CPU). "
            "Recomendado p/ CI. Para a GPU: export JAX_PLATFORMS=cuda,cpu (NÃO só 'cuda')."
        ),
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

    # Seleção de backends a aquecer (op 4 — aditivo; default = só Numba, legado).
    #   --jax-only  → só JAX (pula Numba)
    #   --jax       → Numba + JAX
    #   --jax-auto  → Numba + (JAX só se houver GPU JAX visível; no-op limpo no CI CPU)
    do_numba = not args.jax_only
    do_jax = args.jax or args.jax_only or (args.jax_auto and _gpu_visible())

    print(
        f"Warming up Geosteering AI {SIMULATION_MANAGER_VERSION} "
        f"(numba={do_numba}, jax={do_jax})..."
    )
    t0 = time.perf_counter()

    from geosteering_ai.cli._main import _warmup_numba_tier2_sync

    try:
        if do_numba:
            _warmup_numba_tier2_sync(verbose=args.verbose)
        if do_jax:
            info = _warmup_jax_canonical_sm(verbose=args.verbose)
            if info.get("skipped"):
                # JAX presente como flag mas ausente em runtime (HAS_JAX False) →
                # observabilidade, não gating (mesmo contrato v2.44 do Numba).
                logger.warning(
                    "Warmup JAX pulado (%s) — instale jax e/ou use JAX_PLATFORMS=cuda.",
                    info.get("reason"),
                )
        elif args.jax_auto:
            # --jax-auto sem GPU visível: no-op explícito (recomendado p/ CI CPU).
            print("JAX warmup pulado — sem GPU JAX visível (--jax-auto).")
    except (ModuleNotFoundError, ImportError) as e:
        # Sprint O4 (v2.44): backend ausente (jax/numba não instalados) NÃO é
        # falha de warmup — é ambiente sem os backends a aquecer. Warmup é
        # observabilidade (priming de cache JIT/LLVM), nunca deve gatear o
        # build. Loga aviso e retorna 0 (no-op). CI sem o extra `sim` segue.
        logger.warning(
            "Warmup pulado — backend ausente (%s): %s. "
            "Instale o extra 'sim' (jax+numba) para aquecer.",
            e.__class__.__name__,
            e,
        )
        return 0
    except Exception as e:
        # Falha REAL de warmup (não dep ausente): propaga via logger.error
        # (D9) + exit code 1. CI detecta tanto pelo exit code quanto stderr.
        logger.error("Warmup parcial (%s): %s", e.__class__.__name__, e)
        return 1

    dt = time.perf_counter() - t0
    print(f"OK ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
