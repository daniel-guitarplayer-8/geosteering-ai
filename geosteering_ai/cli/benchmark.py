# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/benchmark.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Subcomando `benchmark` da CLI                              ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.35 — Cenário H estresse multi-core)     ║
# ║  Versão      : v2.35                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-10                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : argparse + simulate_multi (geosteering_ai.simulation)      ║
# ║  Dependências: numpy, geosteering_ai.simulation (lazy import)             ║
# ║  Padrão      : Hexagonal — adapter externo do simulate_multi              ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementa o handler do subcomando ``benchmark``: executa cenários    ║
# ║    canônicos (A/B/C/D/E/F) com warmup explícito + run cronometrado e     ║
# ║    reporta throughput em modelos/hora. Cenários derivam dos benchmarks   ║
# ║    históricos do projeto (``benchmarks/bench_v214_numba.py``).            ║
# ║                                                                           ║
# ║  CENÁRIOS                                                                 ║
# ║    ┌────┬─────────────────────────────────┬──────┬────┬─────┬──────┐     ║
# ║    │ Id │  Característica                 │ npos │ nf │ nTR │ nAng │     ║
# ║    ├────┼─────────────────────────────────┼──────┼────┼─────┼──────┤     ║
# ║    │ A  │  Single-pos, 1 freq, 1 TR       │   1  │  1 │  1  │  1   │     ║
# ║    │ B  │  Multi-pos, 1 freq              │ 100  │  1 │  1  │  1   │     ║
# ║    │ C  │  Multi-pos, multi-freq          │ 100  │  4 │  1  │  1   │     ║
# ║    │ D  │  Single-pos, multi-TR           │   1  │  1 │  4  │  1   │     ║
# ║    │ E  │  Inv0Dip 0° (default)           │ 600  │  1 │  1  │  1   │     ║
# ║    │ F  │  Multi-TR + multi-freq          │ 100  │  4 │  4  │  1   │     ║
# ║    │ G  │  Máxima combinatória (v2.30)    │ 100  │  4 │  4  │  4   │     ║
# ║    │ H  │  Estresse multi-core (v2.35)    │ 100  │  8 │  8  │  8   │     ║
# ║    └────┴─────────────────────────────────┴──────┴────┴─────┴──────┘     ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SCENARIOS: dict com configuração de cada cenário                       ║
# ║    handle_benchmark: handler do subcomando                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Subcomando ``benchmark`` da CLI Geosteering AI (Sprint v2.24 — I2.6).

Executa cenários de benchmark canônicos do projeto (A/B/C/D/E/F/G/H) e
reporta throughput em modelos/hora. Cenários derivam dos benchmarks
históricos em ``benchmarks/bench_v214_numba.py`` e ``bench_v212_workers.py``.

Cenários disponíveis:

| Cenário | Característica                  | n_pos | nf | nTR | nAng |
|:-------:|:--------------------------------|:-----:|:--:|:---:|:----:|
| A       | Single-pos, 1 freq, 1 TR        | 1     | 1  | 1   | 1    |
| B       | Multi-pos, 1 freq               | 100   | 1  | 1   | 1    |
| C       | Multi-pos, multi-freq           | 100   | 4  | 1   | 1    |
| D       | Single-pos, multi-TR            | 1     | 1  | 4   | 1    |
| E       | Inv0Dip 0° (default)            | 600   | 1  | 1   | 1    |
| F       | Multi-TR + multi-freq           | 100   | 4  | 4   | 1    |
| G       | Máxima combinatória (v2.30)     | 100   | 4  | 4   | 4    |
| H       | Estresse multi-core (v2.35)     | 100   | 8  | 8   | 8    |

Use ``--frequencies``, ``--dips``, ``--tr-spacings`` para sobrescrever
os parâmetros de qualquer cenário diretamente na linha de comando.

Cenário H — observações de desempenho (Sprint v2.35):
    8×8×8 = 512 combos por posição × 100 posições × N modelos. Stress-test
    para CPUs multi-core (recomenda-se ``--workers 4 --threads 2`` ou
    superior). Em laptops M-series 8C/16T, ``--n 2`` completa em <120s
    com cache JIT quente. Para CI, usar ``--n 2 --workers 2 --threads 2``.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def _parse_float_list(text: str | None, default: list[float]) -> list[float]:
    """Interpreta string CSV de floats; retorna ``default`` se None ou inválido.

    Aceita vírgula ou ponto-e-vírgula como separador. Espaços são ignorados.
    Mantida localmente para preservar lazy imports independentes entre
    os subcomandos ``simulate`` e ``benchmark``.

    Args:
        text: string no formato ``"1.0,2.0,3.0"`` ou ``None``.
        default: lista retornada quando ``text`` é None, vazio ou inválido.

    Returns:
        Lista de floats parseados, ou ``default`` em caso de erro.

    Example:
        >>> _parse_float_list("0, 15, 30", [0.0])
        [0.0, 15.0, 30.0]
    """
    if not text or not text.strip():
        return default
    try:
        parsed = [
            float(v.strip()) for v in text.replace(";", ",").split(",") if v.strip()
        ]
        return parsed if parsed else default
    except ValueError:
        logger.warning(
            "Valor inválido em lista de floats: %r — usando default %s", text, default
        )
        return default


# Definição dos cenários canônicos (n_pos, frequencies_hz, tr_spacings_m, dip_degs)
SCENARIOS = {
    "A": {"n_pos": 1, "freqs": (20000.0,), "trs": (1.0,), "dips": (0.0,)},
    "B": {"n_pos": 100, "freqs": (20000.0,), "trs": (1.0,), "dips": (0.0,)},
    "C": {
        "n_pos": 100,
        "freqs": (2000.0, 20000.0, 100000.0, 400000.0),
        "trs": (1.0,),
        "dips": (0.0,),
    },
    "D": {"n_pos": 1, "freqs": (20000.0,), "trs": (0.5, 1.0, 1.5, 2.0), "dips": (0.0,)},
    "E": {"n_pos": 600, "freqs": (20000.0,), "trs": (1.0,), "dips": (0.0,)},
    "F": {
        "n_pos": 100,
        "freqs": (2000.0, 20000.0, 100000.0, 400000.0),
        "trs": (0.5, 1.0, 1.5, 2.0),
        "dips": (0.0,),
    },
    "G": {
        "n_pos": 100,
        "freqs": (2000.0, 20000.0, 100000.0, 400000.0),
        "trs": (0.5, 1.0, 1.5, 2.0),
        "dips": (0.0, 15.0, 30.0, 45.0),
    },
    # Cenário H (Sprint v2.35) — estresse multi-core: 8×8×8 = 512 combos.
    # Frequências em escala log para cobrir bandas LWD reais (1 kHz–200 kHz).
    # TRs ampliados de 0.25–2.5 m. Dips em 0°–90° (12.5° step) dentro do
    # range válido [0, 90]° (paridade Fortran — multi_forward.py linha 254).
    # Fix v2.36 D2: 105° → 90° (105° disparava ValueError no worker).
    "H": {
        "n_pos": 100,
        "freqs": (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5),
        "trs": (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5),
        "dips": (0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 90.0),
    },
}


def _build_models(n_models: int, seed: int = 42) -> list[dict]:
    """Gera modelos 5-camada determinísticos para benchmark.

    Args:
        n_models: número de modelos a gerar (≥ 1).
        seed: semente do gerador (default 42, reprodutibilidade).

    Returns:
        Lista de ``n_models`` dicts ``{rho_h, rho_v, esp}``, todos
        em ``np.float64``. ``rho_h`` ∈ [1, 100] Ω·m, ``rho_v / rho_h``
        ∈ [1, 3], espessuras ∈ [2, 10] m.

    Raises:
        Não levanta — ``np.random.default_rng`` aceita ``n_models=0``
        retornando lista vazia.

    Note:
        Idêntico ao gerador de ``simulate.py`` — mantido aqui por
        independência de imports (CLI subcomandos não devem se cruzar
        para preservar lazy import do `simulate.py` quando apenas
        benchmark é invocado).

    Example:
        >>> models = _build_models(2)
        >>> len(models)
        2
    """
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(n_models):
        rho_h = rng.uniform(1.0, 100.0, size=5).astype(np.float64)
        rho_v = rho_h * rng.uniform(1.0, 3.0, size=5).astype(np.float64)
        esp = rng.uniform(2.0, 10.0, size=3).astype(np.float64)
        models.append({"rho_h": rho_h, "rho_v": rho_v, "esp": esp})
    return models


def handle_benchmark(args: argparse.Namespace) -> int:
    """Handler do subcomando ``benchmark``.

    Executa o cenário selecionado em 2 fases:

    1. **Warmup**: 1 modelo descartado para amortizar JIT compile.
       Usa o mesmo ``positions_z``, ``frequencies_hz`` e ``tr_spacings_m``
       do run cronometrado para que todas as especializações JIT sejam
       pré-aquecidas (W2 do code-review).
    2. **Run cronometrado**: ``args.n`` modelos com ``time.perf_counter``.

    Args:
        args: ``argparse.Namespace`` com campos:

            - ``scenario`` (str): identificador do cenário (A..G)
            - ``n`` (int): número de modelos do benchmark
            - ``workers`` (int | None): workers paralelos
            - ``threads`` (int | None): threads Numba por worker
            - ``frequencies`` (str | None): CSV Hz para sobrescrever cenário
            - ``dips`` (str | None): CSV graus para sobrescrever cenário
            - ``tr_spacings`` (str | None): CSV metros para sobrescrever cenário

    Returns:
        Exit code:

        - 0: sucesso, throughput reportado em ``logger.info`` e stdout
        - 1: erro tratado (ValueError, RuntimeError, OSError, ou
          tempo zero medido)

    Raises:
        Não propaga exceções — falhas viram exit code 1.

    Note:
        Logging é configurado em ``main.py`` (centralizado). Imports
        pesados (``simulate_multi``) são lazy.

        Saída final usa ``print()`` (não logger) — esta é uma exceção
        documentada ao padrão D9 do projeto: stdout limpo é parte do
        contrato de uma CLI bench (permite pipe/grep no shell).

    Example:
        Em produção via CLI::

            $ geosteering-cli benchmark --scenario A --n 100
            Cenário A — 1,180,000 mod/h

            $ geosteering-cli benchmark --scenario G --n 20
            Cenário G — 45,000 mod/h

            $ geosteering-cli benchmark --scenario E --n 50 --dips 0,15,30
            Cenário E — 52,000 mod/h
    """
    # Lazy imports — evita carregar numba em `--help`
    from geosteering_ai.simulation import simulate_multi
    from geosteering_ai.simulation.config import SimulationConfig

    sc = SCENARIOS[args.scenario]

    # Parâmetros base do cenário — overrideable via flags CLI (Sprint v2.30)
    frequencies_hz: list[float] = list(sc["freqs"])  # type: ignore
    tr_spacings_m: list[float] = list(sc["trs"])  # type: ignore
    dip_degs: list[float] = list(sc["dips"])  # type: ignore

    # Aplicar overrides fornecidos na linha de comando
    if getattr(args, "frequencies", None):
        frequencies_hz = _parse_float_list(args.frequencies, frequencies_hz)
    if getattr(args, "dips", None):
        dip_degs = _parse_float_list(args.dips, dip_degs)
    if getattr(args, "tr_spacings", None):
        tr_spacings_m = _parse_float_list(args.tr_spacings, tr_spacings_m)

    n_pos: int = sc["n_pos"]  # type: ignore
    positions_z = np.linspace(-5.0, 5.0, n_pos).astype(np.float64)
    models = _build_models(args.n)

    cfg = SimulationConfig(
        n_workers=args.workers,
        threads_per_worker=args.threads,
    )

    logger.info(
        "Cenário %s: n=%d modelos, n_pos=%d, %d freq, %d TR, %d dips — %dw × %dt",
        args.scenario,
        args.n,
        sc["n_pos"],
        len(frequencies_hz),
        len(tr_spacings_m),
        len(dip_degs),
        cfg.n_workers or 1,
        cfg.threads_per_worker or 1,
    )

    # Warmup com shape COMPLETO (W2 code-review): pré-aquece todas as
    # especializações JIT do cenário incluindo multi-freq, multi-TR e multi-dip.
    _ = simulate_multi(
        positions_z=positions_z,
        models=models[:1],
        cfg=cfg,
        frequencies_hz=frequencies_hz,
        tr_spacings_m=tr_spacings_m,
        dip_degs=dip_degs,
    )

    # Run cronometrado
    t0 = time.perf_counter()
    try:
        _ = simulate_multi(
            positions_z=positions_z,
            models=models,
            cfg=cfg,
            frequencies_hz=frequencies_hz,
            tr_spacings_m=tr_spacings_m,
            dip_degs=dip_degs,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Erro durante benchmark: %s", exc, exc_info=True)
        return 1
    dt = time.perf_counter() - t0

    if dt <= 0:
        logger.error("Benchmark concluído em tempo zero — algo falhou")
        return 1

    rate = (args.n / dt) * 3600.0
    logger.info(
        "Cenário %s — %d modelos em %.2fs → %.0f mod/h",
        args.scenario,
        args.n,
        dt,
        rate,
    )
    # CLI: stdout limpo (D9 exception documentada — ver Note do docstring)
    print(f"Cenário {args.scenario} — {rate:,.0f} mod/h")
    return 0
