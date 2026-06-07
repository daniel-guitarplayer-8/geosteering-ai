# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/benchmark.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Subcomando `benchmark` da CLI                              ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.56 — wall-clock JAX + transparência)     ║
# ║  Versão      : v2.56                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-10                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : argparse + simulate_multi / simulate_batch (via _exec)      ║
# ║  Dependências: numpy, geosteering_ai.cli._{exec,backend,table,hwinfo}      ║
# ║  Padrão      : Hexagonal — adapter externo do simulador                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementa o handler do subcomando ``benchmark``: executa cenários    ║
# ║    canônicos (A–H) no backend escolhido (numba/jax) com warmup + run     ║
# ║    cronometrado (--repeat → mediana), reporta throughput em modelos/hora ║
# ║    (linha grep-able preservada) + tabela ASCII / JSON. Cenários derivam  ║
# ║    dos benchmarks históricos (``benchmarks/bench_v214_numba.py``).        ║
# ║    Recursos v2.53: --backend · --list-scenarios · --json · --repeat ·    ║
# ║    --compare-backends · --warmup · --quiet.                              ║
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
import json
import logging
import statistics
from time import perf_counter
from typing import List

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
    *,
    geometry: str = "per-model",
    n_geometries: int | None = None,
    quantize_step: float | None = None,
) -> list[dict]:
    """Gera modelos 5-camada determinísticos para benchmark.

    Args:
        n_models: número de modelos a gerar (≥ 1).
        seed: semente do gerador (default 42, reprodutibilidade).
        geometry: modo de amostragem de ``esp`` (batchabilidade no JAX):
            ``"per-model"`` (DEFAULT, esp único por modelo), ``"templates"``
            (K geometrias replicadas → agrupável, JAX satura) ou
            ``"quantized"`` (esp arredondado → parcial).
        n_geometries: (só ``templates``) K geometrias distintas.
        quantize_step: (só ``quantized``) passo de quantização (m).

    Returns:
        Lista de ``n_models`` dicts ``{rho_h, rho_v, esp}``, todos
        em ``np.float64``. ``rho_h`` ∈ [1, 100] Ω·m, ``rho_v / rho_h``
        ∈ [1, 3], espessuras ∈ [2, 10] m.

    Raises:
        ValueError: ``geometry`` inválido ou ``quantize_step <= 0``.

    Note:
        ``per-model`` preserva o stream rng legado (esp no loop);
        ``templates``/``quantized`` pré-computam ``esp`` via
        :func:`geosteering_ai.cli._exec.sample_geometry` (mesma lógica de
        produção). O Numba é indiferente ao modo — só o JAX-grouped se
        beneficia do compartilhamento de geometria.

    Example:
        >>> models = _build_models(2)
        >>> len(models)
        2
    """
    rng = np.random.default_rng(seed)
    esp_all = None
    if geometry != "per-model":
        from geosteering_ai.cli._exec import sample_geometry

        esp_all = sample_geometry(
            rng,
            n_models,
            3,
            mode=geometry,
            n_geometries=n_geometries,
            quantize_step=quantize_step,
        )
    models = []
    for i in range(n_models):
        rho_h = rng.uniform(1.0, 100.0, size=5).astype(np.float64)
        rho_v = rho_h * rng.uniform(1.0, 3.0, size=5).astype(np.float64)
        esp = (
            esp_all[i]
            if esp_all is not None
            else rng.uniform(2.0, 10.0, size=3).astype(np.float64)
        )
        models.append({"rho_h": rho_h, "rho_v": rho_v, "esp": esp})
    return models


def _list_scenarios() -> int:
    """Lista os cenários A–H disponíveis e sai (stdout — exceção D9).

    Returns:
        Exit code 0 — listagem informativa, sem simular.
    """
    print("Cenários disponíveis (geosteering-cli benchmark --scenario X):")
    for key, sc in SCENARIOS.items():
        # SCENARIOS tem valores heterogêneos (int + tuplas) → mypy infere
        # ``object``; extrai-se em locais tipados (mesmo padrão # type: ignore
        # do handle_benchmark legado).
        n_pos = int(sc["n_pos"])  # type: ignore[call-overload]
        nf = len(sc["freqs"])  # type: ignore[arg-type]
        ntr = len(sc["trs"])  # type: ignore[arg-type]
        nd = len(sc["dips"])  # type: ignore[arg-type]
        print(
            f"  {key}: n_pos={n_pos:>4d} | "
            f"{nf} freq × {ntr} TR × {nd} dips = {nf * ntr * nd} combos/pos"
        )
    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """Handler do subcomando ``benchmark`` (Sprint v2.53).

    Executa o cenário selecionado no backend escolhido (numba/jax) em 3 fases:
    (1) warmup JIT com shape completo (1 modelo, descartado); (2) ``--repeat``
    rodadas hot cronometradas → mediana do throughput; (3) tabela ASCII +
    linha grep-able ``Cenário X — Y mod/h`` (contrato preservado).

    Args:
        args: ``argparse.Namespace`` do subparser ``benchmark`` (scenario, n,
            workers, threads, frequencies, dips, tr_spacings, list_scenarios,
            quiet, backend, dtype, jax_strategy, warmup, as_json, repeat,
            compare_backends).

    Returns:
        Exit code: 0 (sucesso) | 1 (erro tratado ou tempo zero medido).

    Raises:
        Não propaga — falhas viram exit code 1.

    Note:
        Saída final usa ``print()`` (exceção D9 documentada — stdout limpo é
        contrato de CLI bench, permite pipe/grep). O caminho Numba preserva
        ``simulate_multi(models=...)`` (pool de workers) sem regressão; o JAX
        usa o dispatcher parity-tested ``simulate_batch`` (via ``_exec``).

    Example:
        $ geosteering-cli benchmark --scenario A --n 100
        Cenário A — 1,180,000 mod/h

        $ geosteering-cli benchmark --scenario E --n 50 --backend jax --json
        $ geosteering-cli benchmark --list-scenarios
    """
    if getattr(args, "list_scenarios", False):
        return _list_scenarios()

    # v2.56 — t_total cobre TUDO do handler (lazy imports pesados numba/jax +
    # build + warmup + rodadas cronometradas), explicando o gap vs o `time real`.
    t_total_start = perf_counter()

    # Lazy imports — evita carregar numba/jax em `--help`/`--list-scenarios`.
    from geosteering_ai.cli._exec import (
        finitude_stats,
        resolve_backend_preflight,
        resolve_jax_chunk_size,
        resolve_requested_backend,
        run_once,
        warmup_backend,
    )
    from geosteering_ai.cli._hwinfo import collect_hardware_info

    sc = SCENARIOS[args.scenario]

    # Parâmetros base do cenário — overrideable via flags CLI (Sprint v2.30).
    frequencies_hz: list[float] = list(sc["freqs"])  # type: ignore
    tr_spacings_m: list[float] = list(sc["trs"])  # type: ignore
    dip_degs: list[float] = list(sc["dips"])  # type: ignore
    if getattr(args, "frequencies", None):
        frequencies_hz = _parse_float_list(args.frequencies, frequencies_hz)
    if getattr(args, "dips", None):
        dip_degs = _parse_float_list(args.dips, dip_degs)
    if getattr(args, "tr_spacings", None):
        tr_spacings_m = _parse_float_list(args.tr_spacings, tr_spacings_m)

    n_pos: int = sc["n_pos"]  # type: ignore
    positions_z = np.linspace(-5.0, 5.0, n_pos).astype(np.float64)
    models = _build_models(
        args.n,
        geometry=getattr(args, "geometry", "per-model"),
        n_geometries=getattr(args, "n_geometries", None),
        quantize_step=getattr(args, "quantize_step", None),
    )

    # ── --compare-backends: caminho dedicado (numba vs jax) ──────────────
    if getattr(args, "compare_backends", False):
        from geosteering_ai.cli._exec import run_compare_backends

        return run_compare_backends(
            models=models,
            positions_z=positions_z,
            frequencies_hz=frequencies_hz,
            dip_degs=dip_degs,
            tr_spacings_m=tr_spacings_m,
            n_pos=n_pos,
            workers=args.workers,
            threads=args.threads,
            dtype=args.dtype,
            jax_strategy=args.jax_strategy,
            warmup=args.warmup,
            as_json=args.as_json,
            quiet=args.quiet,
            title=f"COMPARAÇÃO DE BACKENDS — benchmark {args.scenario}",
        )

    # Normaliza o solicitado (None→numba+DeprecationWarning; spec 0003); então o
    # PRÉ-VOO de geometria (v2.55) conta grupos com NumPy puro ANTES de tocar JAX
    # — se jax/auto+não-agrupável, roda Numba SEM inicializar o CUDA (evita crash TLS).
    requested_backend = resolve_requested_backend(args)
    backend, device, preflight_reason = resolve_backend_preflight(
        requested_backend, models, quiet=args.quiet
    )

    logger.info(
        "Cenário %s: n=%d modelos, n_pos=%d, %d freq, %d TR, %d dips — backend=%s",
        args.scenario,
        args.n,
        n_pos,
        len(frequencies_hz),
        len(tr_spacings_m),
        len(dip_degs),
        backend,
    )

    # Auto-chunk anti-OOM (v2.56, D): com poucos grupos GRANDES, o vmap (n_models,
    # …, nf, 9) cresce; em high-config (G/H, nf·nTR·nAng ≥ 9) fragmenta-se o eixo
    # de modelos para não estourar VRAM. Cenário E (nf=1) → sem chunk (vmap cheio).
    n_configs = len(frequencies_hz) * len(tr_spacings_m) * len(dip_degs)
    jax_chunk = resolve_jax_chunk_size(
        backend, n_configs, explicit=getattr(args, "jax_chunk_size", None)
    )

    # ── Warmup (cronometrado — v2.56 C, transparência) ───────────────────
    t_warmup_start = perf_counter()
    # v2.56 (B): `--warmup`→`warmup_backend` no JAX compila o shape ERRADO
    # (n_models=1) → peso morto (~3s, zero benefício, medido). SKIP no jax; o
    # JIT-warmup abaixo (workload completo) é o warmup EFETIVO. Numba mantém.
    if args.warmup and backend != "jax":
        warmup_backend(
            backend,
            n_pos=n_pos,
            dtype=args.dtype,
            jax_strategy=args.jax_strategy,
            dip_degs=dip_degs,
            tr_spacings_m=tr_spacings_m,
            freqs_hz=frequencies_hz,
        )

    # ── Warmup JIT (descartado) — pré-aquece as especializações do cenário ──
    # v2.55: para JAX, aquecer com o WORKLOAD COMPLETO (`models`), não `models[:1]`:
    #   (1) `models[:1]` (1 modelo = 1 grupo) SEMPRE degeneraria p/ Numba → rodaria
    #       Numba-após-CUDA (crash TLS) E compilaria ZERO programas XLA (warmup inútil);
    #   (2) o batch completo (agrupável, garantido pelo pré-voo) compila o trace XLA
    #       vmap+scatter dos group-sizes REAIS → o timed-run roda QUENTE (~2× Numba).
    # Para Numba, `models[:1]` basta (JIT do shape; o pool reusa as especializações).
    _warmup_models = models if backend == "jax" else models[:1]
    try:
        run_once(
            backend,
            _warmup_models,
            positions_z,
            frequencies_hz=frequencies_hz,
            dip_degs=dip_degs,
            tr_spacings_m=tr_spacings_m,
            workers=args.workers,
            threads=args.threads,
            dtype=args.dtype,
            jax_strategy=args.jax_strategy,
            jax_chunk_size_models=jax_chunk,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Erro durante warmup do benchmark: %s", exc, exc_info=True)
        return 1
    warmup_s = perf_counter() - t_warmup_start

    # ── Rodadas cronometradas (--repeat hot → mediana) ───────────────────
    repeat = max(1, int(args.repeat))
    throughputs: List[float] = []
    elapseds: List[float] = []
    last_H6: np.ndarray | None = None
    last_groups: int | None = None
    effective_backend = backend
    # Motivo inicial = pré-voo (jax→numba por geometria não-agrupável).
    fallback_reason: str | None = preflight_reason
    try:
        for _i in range(repeat):
            H6, elapsed, groups, eff, reason = run_once(
                backend,
                models,
                positions_z,
                frequencies_hz=frequencies_hz,
                dip_degs=dip_degs,
                tr_spacings_m=tr_spacings_m,
                workers=args.workers,
                threads=args.threads,
                dtype=args.dtype,
                jax_strategy=args.jax_strategy,
                jax_chunk_size_models=jax_chunk,
            )
            if elapsed <= 0:
                logger.error("Benchmark concluído em tempo zero — algo falhou")
                return 1
            throughputs.append((args.n / elapsed) * 3600.0)
            elapseds.append(elapsed)
            last_H6, last_groups = H6, groups
            effective_backend = eff
            # Fallback do dispatcher sobrescreve; senão preserva o motivo do pré-voo.
            if eff != backend:
                fallback_reason = reason
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Erro durante benchmark: %s", exc, exc_info=True)
        return 1

    thr_median = statistics.median(throughputs)
    thr_std = statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    # v2.56: elapsed_s = MEDIANA (coerente com thr_median; antes era a última rodada).
    elapsed_median = statistics.median(elapseds)
    total_s = perf_counter() - t_total_start
    # device EFETIVO: jax→numba (geometria não-agrupável) roda em CPU.
    effective_device = "gpu" if effective_backend == "jax" else "cpu"

    logger.info(
        "Cenário %s — %d modelos em %.2fs → %.0f mod/h (backend=%s)",
        args.scenario,
        args.n,
        elapsed_median,
        thr_median,
        effective_backend,
    )
    logger.info(
        "Tempos: warmup=%.2fs · hot(mediana)=%.3fs · total(handler)=%.2fs",
        warmup_s,
        elapsed_median,
        total_s,
    )

    # ── Saída JSON (apta a | jq) OU linha grep-able + tabela ─────────────
    fin = finitude_stats(last_H6) if last_H6 is not None else {}
    stats = {
        "scenario": args.scenario,
        "backend": effective_backend,
        "device": effective_device,
        "throughput_mod_h": thr_median,
        "throughput_std": thr_std,
        "elapsed_s": elapsed_median,
        "warmup_s": warmup_s,
        "total_s": total_s,
        "repeat": repeat,
        "n_models": args.n,
        "n_pos": n_pos,
        "n_freqs": len(frequencies_hz),
        "n_dips": len(dip_degs),
        "n_trs": len(tr_spacings_m),
        "workers": args.workers,
        "threads": args.threads,
        "dtype": args.dtype,
        "jax_strategy": args.jax_strategy,
        "jax_chunk_size": jax_chunk,
        "n_geometry_groups": last_groups,
        "reason": fallback_reason,
        "nan_count": fin.get("nan_count"),
        "inf_count": fin.get("inf_count"),
        "all_finite": fin.get("all_finite"),
    }

    if args.as_json:
        hw = collect_hardware_info(want_gpu=(effective_backend == "jax"))
        print(
            json.dumps(
                {"result": stats, "hardware": hw}, ensure_ascii=False, default=str
            )
        )
        return 0

    # CLI: stdout limpo (D9 exception documentada — contrato grep-able).
    print(f"Cenário {args.scenario} — {thr_median:,.0f} mod/h")
    if not args.quiet:
        from geosteering_ai.cli._table import build_result_rows, render_kv_table

        hw = collect_hardware_info(want_gpu=(effective_backend == "jax"))
        rows = build_result_rows(stats, hw)
        print(render_kv_table(f"BENCHMARK {args.scenario} — geosteering-cli", rows))
    return 0
