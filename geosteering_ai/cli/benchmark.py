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
    # TRs ampliados de 0.25–2.5 m. Dips em 0°–87.5° (12.5° step regular).
    # Fix v2.36 D2: 105° → 90° (105° disparava ValueError no worker).
    # Fix v2.44: 90° → 87.5° — o path JAX valida dip em [0, 89]° (90° é o
    #   poço horizontal degenerado, cos=0); 87.5° mantém o passo 12.5° regular
    #   e torna o Cenário H executável tanto no Numba quanto no JAX GPU.
    "H": {
        "n_pos": 100,
        "freqs": (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5),
        "trs": (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5),
        "dips": (0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5),
    },
}


def _build_models(
    n_models: int,
    seed: int = 42,
    geometry: str = "per-model",
    n_geometries: int | None = None,
) -> list[dict]:
    """Gera modelos 5-camada determinísticos para benchmark.

    Args:
        n_models: número de modelos a gerar (≥ 1).
        seed: semente do gerador (default 42, reprodutibilidade).
        geometry: ``"per-model"`` (default, stream legado) | ``"templates"`` |
            ``"quantized"`` — controla a batchabilidade da ``esp`` no JAX (espelha
            :func:`geosteering_ai.cli._exec.sample_geometry`).
        n_geometries: (só ``templates``) K geometrias distintas; None = auto.

    Returns:
        Lista de ``n_models`` dicts ``{rho_h, rho_v, esp}``, todos
        em ``np.float64``. ``rho_h`` ∈ [1, 100] Ω·m, ``rho_v / rho_h``
        ∈ [1, 3], espessuras ∈ [2, 10] m.

    Raises:
        Não levanta — ``np.random.default_rng`` aceita ``n_models=0``
        retornando lista vazia.

    Note:
        Idêntico ao gerador de ``simulate._build_random_models`` — mantido aqui
        por independência de imports (subcomandos não devem se cruzar p/ preservar
        o lazy import). ``per-model`` preserva o stream rng legado (bit-idêntico).

    Example:
        >>> models = _build_models(2)
        >>> len(models)
        2
    """
    rng = np.random.default_rng(seed)
    if geometry == "per-model":
        # Stream LEGADO (bit-idêntico): rho_h, rho_v=rho_h·ratio, esp — POR modelo.
        models = []
        for _ in range(n_models):
            rho_h = rng.uniform(1.0, 100.0, size=5).astype(np.float64)
            rho_v = rho_h * rng.uniform(1.0, 3.0, size=5).astype(np.float64)
            esp = rng.uniform(2.0, 10.0, size=3).astype(np.float64)
            models.append({"rho_h": rho_h, "rho_v": rho_v, "esp": esp})
        return models

    # templates/quantized — ``esp`` em bloco (compartilhada); ``rho`` por modelo.
    from geosteering_ai.cli._exec import sample_geometry

    esp_batch = sample_geometry(
        rng, n_models, 3, mode=geometry, n_geometries=n_geometries
    )
    models = []
    for i in range(n_models):
        rho_h = rng.uniform(1.0, 100.0, size=5).astype(np.float64)
        rho_v = rho_h * rng.uniform(1.0, 3.0, size=5).astype(np.float64)
        models.append({"rho_h": rho_h, "rho_v": rho_v, "esp": esp_batch[i]})
    return models


def _list_scenarios() -> int:
    """Lista os cenários canônicos (A..H) com suas dimensões em stdout.

    Saída tabular (id · n_pos · nf · nTR · nAng · combos/pos), útil para
    descobrir os cenários disponíveis sem abrir o código. Pura (sem imports
    pesados) — retorna 0. ``print`` é a exceção documentada D9 (stdout limpo é
    contrato da CLI bench).

    Returns:
        ``0`` (sempre — operação puramente informativa).
    """
    print("Cenários de benchmark canônicos:")
    print(
        f"  {'Id':<3} {'n_pos':>6} {'nf':>3} {'nTR':>4} {'nAng':>5} {'combos/pos':>11}"
    )
    print(f"  {'─' * 3} {'─' * 6} {'─' * 3} {'─' * 4} {'─' * 5} {'─' * 11}")
    for sid, sc in SCENARIOS.items():
        nf = len(sc["freqs"])  # type: ignore[arg-type]
        ntr = len(sc["trs"])  # type: ignore[arg-type]
        nang = len(sc["dips"])  # type: ignore[arg-type]
        combos = nf * ntr * nang
        print(f"  {sid:<3} {sc['n_pos']:>6} {nf:>3} {ntr:>4} {nang:>5} {combos:>11}")
    return 0


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
    # ── --list-scenarios: lista e sai (puro, antes de qualquer import pesado) ──
    if bool(getattr(args, "list_scenarios", False)):
        return _list_scenarios()

    # ── Motor compartilhado (cli._exec) — wiring p/ AMBOS os backends ─────────
    # Imports leves (numpy puro); a simulação pesada é lazy DENTRO de run_once.
    from geosteering_ai.cli._exec import (
        finitude_stats,
        resolve_backend_preflight,
        resolve_jax_chunk_size,
        resolve_requested_backend,
        run_compare_backends,
        run_once,
    )

    t_handler = time.perf_counter()  # p/ tempo total (handler) na tabela ASCII
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
    n_models = int(args.n)
    quiet = bool(getattr(args, "quiet", False))
    as_json = bool(getattr(args, "json", False))
    workers = getattr(args, "workers", None)
    threads = getattr(args, "threads", None)
    # ── Geometria + dtype + JAX (paridade c/ simulate — item 3 da triagem) ────
    geometry = getattr(args, "geometry", "per-model")
    n_geometries = getattr(args, "n_geometries", None)
    dtype = getattr(args, "dtype", "complex128")
    jax_strategy = getattr(args, "jax_strategy", "bucketed")
    repeat = max(1, int(getattr(args, "repeat", 1) or 1))
    models = _build_models(n_models, geometry=geometry, n_geometries=n_geometries)

    # ── --compare-backends: numba × jax lado-a-lado (DRY com simulate) ────────
    if bool(getattr(args, "compare_backends", False)):
        return run_compare_backends(
            models=models,
            positions_z=positions_z,
            frequencies_hz=frequencies_hz,
            dip_degs=dip_degs,
            tr_spacings_m=tr_spacings_m,
            n_pos=n_pos,
            workers=workers,
            threads=threads,
            dtype=dtype,
            jax_strategy=jax_strategy,
            warmup=True,
            as_json=as_json,
            quiet=quiet,
            title=f"benchmark --scenario {args.scenario}",
        )

    # ── Backend (pré-voo TLS-safe — spec 0003) ────────────────────────────────
    requested = resolve_requested_backend(args)
    backend, device, _reason = resolve_backend_preflight(requested, models, quiet=quiet)
    n_configs = len(frequencies_hz) * len(tr_spacings_m) * len(dip_degs)
    jax_chunk = resolve_jax_chunk_size(
        backend, n_configs, explicit=getattr(args, "jax_chunk_size", None)
    )
    if not quiet:
        logger.info(
            "Cenário %s: backend=%s (device=%s) n=%d, n_pos=%d, %d freq, %d TR, %d dips",
            args.scenario,
            backend,
            device,
            n_models,
            n_pos,
            len(frequencies_hz),
            len(tr_spacings_m),
            len(dip_degs),
        )

    # Warmup com shape COMPLETO (W2 code-review): run_once(models[:1]) — no NUMBA
    # é byte-idêntico ao legado simulate_multi(models[:1]) (pré-aquece todas as
    # especializações JIT do cenário); no JAX aquece o XLA. Resultado descartado.
    t_warm = time.perf_counter()
    try:
        run_once(
            backend,
            models[:1],
            positions_z,
            frequencies_hz=frequencies_hz,
            dip_degs=dip_degs,
            tr_spacings_m=tr_spacings_m,
            workers=workers,
            threads=threads,
            dtype=dtype,
            jax_strategy=jax_strategy,
            jax_chunk_size_models=jax_chunk,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Erro durante warmup do benchmark: %s", exc, exc_info=True)
        return 1
    warmup_s = time.perf_counter() - t_warm

    # Run CRONOMETRADO — ``--repeat`` rodadas, reporta a MELHOR (menor elapsed →
    # maior throughput), espelhando o simulate. run_once mede internamente; o
    # caminho numba (pool) permanece inalterado.
    h6 = None
    elapsed = float("inf")
    n_groups: int | None = None
    effective = backend
    try:
        for _ in range(repeat):
            h6_i, elapsed_i, n_groups_i, effective_i, _run_reason = run_once(
                backend,
                models,
                positions_z,
                frequencies_hz=frequencies_hz,
                dip_degs=dip_degs,
                tr_spacings_m=tr_spacings_m,
                workers=workers,
                threads=threads,
                dtype=dtype,
                jax_strategy=jax_strategy,
                jax_chunk_size_models=jax_chunk,
            )
            if elapsed_i < elapsed:  # mantém a melhor rodada
                h6, elapsed, n_groups, effective = (
                    h6_i,
                    elapsed_i,
                    n_groups_i,
                    effective_i,
                )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Erro durante benchmark: %s", exc, exc_info=True)
        return 1

    if elapsed <= 0:
        logger.error("Benchmark concluído em tempo zero — algo falhou")
        return 1

    rate = (n_models / elapsed) * 3600.0
    total_s = time.perf_counter() - t_handler
    fin = finitude_stats(h6) if h6 is not None else {}

    if as_json:
        import json

        print(
            json.dumps(
                {
                    "command": "benchmark",
                    "scenario": args.scenario,
                    "backend": effective,
                    "device": device,
                    "n_models": n_models,
                    "n_pos": n_pos,
                    "elapsed_s": elapsed,
                    "warmup_s": warmup_s,
                    "total_s": total_s,
                    "throughput_mod_h": rate,
                    "n_freq": len(frequencies_hz),
                    "n_tr": len(tr_spacings_m),
                    "n_dips": len(dip_degs),
                    "geometry": geometry,
                    "n_geometry_groups": n_groups,
                    "dtype": dtype,
                    "jax_strategy": jax_strategy,
                    "repeat": repeat,
                    "finitude": fin or None,
                }
            )
        )
    elif not quiet:
        # ── Tabela ASCII de resultados (devolve a apresentação do CLI rico) ───
        from geosteering_ai.cli._hwinfo import collect_hardware_info
        from geosteering_ai.cli._table import build_result_rows, render_kv_table

        hw = collect_hardware_info(want_gpu=(effective == "jax"))
        stats = {
            "backend": effective,
            "device": device,
            "throughput_mod_h": rate,
            "elapsed_s": elapsed,
            "warmup_s": warmup_s if warmup_s > 0 else None,
            "total_s": total_s,
            "n_models": n_models,
            "n_pos": n_pos,
            "n_freqs": len(frequencies_hz),
            "n_dips": len(dip_degs),
            "n_trs": len(tr_spacings_m),
            "dtype": dtype,
            "jax_strategy": jax_strategy,
            "n_geometry_groups": n_groups,
            "workers": workers,
            "threads": threads,
            "nan_count": fin.get("nan_count"),
            "inf_count": fin.get("inf_count"),
            "all_finite": fin.get("all_finite"),
        }
        title = f"BENCHMARK {args.scenario} — RESULTADO"
        print(render_kv_table(title, build_result_rows(stats, hw)))
    return 0
