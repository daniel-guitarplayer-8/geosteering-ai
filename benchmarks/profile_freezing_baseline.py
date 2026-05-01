#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark de profiling — Geosteering AI Simulation Manager v2.11.

Mede o "freezing" da GUI ao simular N modelos para diferentes valores de N.
Usa :class:`MainThreadHeartbeat` para capturar gaps no event loop Qt e
:class:`PhaseTimer` para medir cada fase do ciclo de simulação.

Uso típico
~~~~~~~~~~

Executar baseline (capturar comportamento PRÉ-fix v2.11)::

    $ python benchmarks/profile_freezing_baseline.py --mode baseline \\
        --n 100 1000 10000 30000 \\
        --backend numba \\
        --out docs/reports/v2.11_baseline_profiling.md

Executar pós-fix (esperando max_gap_ms < 50ms)::

    $ python benchmarks/profile_freezing_baseline.py --mode post_fix \\
        --n 100 1000 10000 30000 \\
        --backend numba \\
        --out docs/reports/v2.11_post_fix_profiling.md

Critérios de aprovação v2.11 (verificados automaticamente)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``max_gap_ms < 50ms`` em qualquer cenário N
- ``sum_gap_ms < 200ms`` total
- ``total_gaps < 5``

Modo offscreen
~~~~~~~~~~~~~~

Em CI ou ambientes sem display::

    $ QT_QPA_PLATFORM=offscreen python benchmarks/profile_freezing_baseline.py \\
        --mode baseline --n 100 --backend numba --offscreen

Note:
    Este script é uma FERRAMENTA DE PROFILING — ele requer que a GUI
    Simulation Manager possa ser inicializada (PyQt6 ou PySide6
    instalados). Em CI offscreen, o ``QApplication`` é criado mas a
    janela não é visível.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# Adicionar pacote ao path se rodando do diretório raiz
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse argumentos da linha de comando."""
    p = argparse.ArgumentParser(
        description="Profiling do freezing GUI no Simulation Manager v2.11."
    )
    p.add_argument(
        "--mode",
        choices=["baseline", "post_fix"],
        default="baseline",
        help="Modo de execução (afeta nome do relatório).",
    )
    p.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[100, 1000, 10000, 30000],
        help="Valores de N (modelos) a testar. Default: 100 1k 10k 30k.",
    )
    p.add_argument(
        "--backend",
        choices=["numba", "fortran"],
        default="numba",
        help="Backend a benchmarkar.",
    )
    p.add_argument(
        "--threshold-ms",
        type=float,
        default=50.0,
        help="Threshold em ms para considerar gap. Default 50.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Arquivo de saída (.md). Default: derivado do modo.",
    )
    p.add_argument(
        "--offscreen",
        action="store_true",
        help="Forçar QT_QPA_PLATFORM=offscreen (CI).",
    )
    return p.parse_args()


def run_one_scenario(
    n_models: int,
    backend: str,
    threshold_ms: float,
    mode: str = "baseline",
) -> Dict[str, Any]:
    """Executa UM cenário (N x backend) e captura métricas.

    Args:
        n_models: Quantidade de modelos a gerar/simular.
        backend: 'numba' ou 'fortran'.
        threshold_ms: Threshold do heartbeat em ms.
        mode: 'baseline' usa generate_models síncrono (= bloqueio main
            thread) para reproduzir o gargalo. 'post_fix' usa
            ModelGenerationThread (= QThread) para validar o fix.

    Returns:
        Dict com keys 'n_models', 'backend', 'mode', 'phase_times',
        'heartbeat', 'wall_clock_total'.
    """
    # Imports locais — só ocorrem se o script é executado (não no parse_args).
    from geosteering_ai.simulation.tests.sm_heartbeat import MainThreadHeartbeat
    from geosteering_ai.simulation.tests.sm_phase_timer import PhaseTimer

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] "
          f"== Cenário: N={n_models} backend={backend} mode={mode} ==")

    # Criar QApplication (idempotente — reutiliza se já existir).
    from geosteering_ai.simulation.tests.sm_qt_compat import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Setup instrumentação.
    heartbeat = MainThreadHeartbeat(threshold_ms=threshold_ms)
    timer = PhaseTimer()
    heartbeat.start()
    t_wall_start = time.perf_counter()

    from geosteering_ai.simulation.tests.sm_model_gen import GenConfig
    cfg = GenConfig()  # defaults — Sobol, log-uniform, anisotropic

    if mode == "baseline":
        # Modo baseline: geração SÍNCRONA na main thread — reproduz
        # o gargalo original. Heartbeat detecta gap proporcional a N.
        from geosteering_ai.simulation.tests.sm_model_gen import generate_models
        timer.begin("generation")
        models = generate_models(cfg, n_models=n_models, rng_seed=42)
        timer.end("generation")
    else:
        # Modo post_fix: geração ASSÍNCRONA via ModelGenerationThread.
        # Main thread permanece responsiva — heartbeat NÃO deve detectar
        # gaps grandes durante a geração.
        from geosteering_ai.simulation.tests.sm_model_gen import (
            ModelGenerationThread,
        )
        models_holder: List[List[dict]] = []
        loop = QtCore.QEventLoop()
        gen_thread = ModelGenerationThread(cfg, n_models=n_models, rng_seed=42)
        gen_thread.finished_models.connect(
            lambda ms: (models_holder.append(ms), loop.quit())
        )
        gen_thread.error.connect(lambda msg: (print(f"  [ERR] {msg}"), loop.quit()))
        timer.begin("generation")
        gen_thread.start()
        # event loop processa app events enquanto thread roda — heartbeat ticks.
        loop.exec()
        timer.end("generation")
        gen_thread.wait(5000)
        models = models_holder[0] if models_holder else []

    print(f"  [generation] {len(models)} modelos em "
          f"{timer.get_summary().get('generation', 0.0):.2f}s")

    # Processar eventos pendentes para o heartbeat capturar ticks finais.
    deadline = time.perf_counter() + 0.3
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.01)

    heartbeat.stop()
    t_wall_end = time.perf_counter()
    report = heartbeat.report()

    print(f"  [heartbeat] {report.format_summary()}")
    print(f"  [criteria] passes_v211={report.passes_v211_criteria()}")

    return {
        "n_models": n_models,
        "backend": backend,
        "mode": mode,
        "phase_times": timer.get_summary(),
        "heartbeat": report.to_dict(),
        "wall_clock_total": t_wall_end - t_wall_start,
    }


def write_markdown_report(
    scenarios: List[Dict[str, Any]],
    mode: str,
    out_path: Path,
) -> None:
    """Gera relatório Markdown a partir dos cenários executados."""
    lines: List[str] = []
    title = "Baseline" if mode == "baseline" else "Pós-Fix"
    lines.append(f"# Relatório de Profiling — Simulation Manager v2.11 ({title})")
    lines.append("")
    lines.append(f"> Gerado em: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"> Modo: `{mode}`")
    lines.append(f"> Cenários: {len(scenarios)}")
    lines.append("")
    lines.append("## Tabela de Resultados")
    lines.append("")
    lines.append("| N | Backend | Geração (s) | Gaps>50ms | Max Gap (ms) | "
                 "Sum Gap (ms) | Passa v2.11? |")
    lines.append("|---|---------|------------:|----------:|-------------:|"
                 "-------------:|:------------:|")
    for sc in scenarios:
        gen = sc["phase_times"].get("generation", 0.0)
        hb = sc["heartbeat"]
        passa = "✅" if hb.get("passes_v211", False) else "❌"
        lines.append(
            f"| {sc['n_models']} | {sc['backend']} | "
            f"{gen:.2f} | {hb['total_gaps']} | "
            f"{hb['max_gap_ms']:.1f} | {hb['sum_gap_ms']:.1f} | {passa} |"
        )
    lines.append("")
    lines.append("## Critérios de Aprovação v2.11")
    lines.append("")
    lines.append("- `max_gap_ms < 50ms` em todos os cenários")
    lines.append("- `sum_gap_ms < 200ms` total")
    lines.append("- `total_gaps < 5`")
    lines.append("")
    lines.append("## JSON Bruto")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(scenarios, indent=2, default=str))
    lines.append("```")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Relatório salvo em: {out_path}")


def main() -> int:
    """Entry point."""
    args = parse_args()

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Default out path
    if args.out is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        suffix = "baseline_profiling" if args.mode == "baseline" else "post_fix_profiling"
        args.out = f"docs/reports/v2.11_{suffix}_{date_str}.md"

    out_path = Path(args.out)

    scenarios: List[Dict[str, Any]] = []
    for n in args.n:
        try:
            sc = run_one_scenario(
                n_models=n,
                backend=args.backend,
                threshold_ms=args.threshold_ms,
                mode=args.mode,
            )
            scenarios.append(sc)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERRO] Cenário N={n} falhou: {exc}", file=sys.stderr)
            continue

    if not scenarios:
        print("[ERRO] Nenhum cenário foi executado com sucesso.", file=sys.stderr)
        return 1

    write_markdown_report(scenarios, args.mode, out_path)
    # Exit code 0 se TODOS os cenários passam; 1 se algum falha.
    all_pass = all(
        sc["heartbeat"].get("passes_v211", False) for sc in scenarios
    )
    if not all_pass and args.mode == "post_fix":
        print("[FAIL] Algum cenário NÃO passa critérios v2.11.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
