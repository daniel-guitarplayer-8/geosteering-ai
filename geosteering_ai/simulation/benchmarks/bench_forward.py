# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/benchmarks/bench_forward.py                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Benchmark Forward (Sprint 2.7)          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x + time (stdlib)                                  ║
# ║  Dependências: numpy, geosteering_ai.simulation.forward                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Benchmark do forward completo do simulador Python, medindo            ║
# ║    throughput em modelos/hora para comparação contra a meta de           ║
# ║    ≥ 40 000 modelos/hora em CPU.                                         ║
# ║                                                                           ║
# ║  METODOLOGIA                                                              ║
# ║    1. Define um modelo Sobol típico (22 camadas TIV, 601 posições,      ║
# ║       filtro Werthmüller 201pt, 1 frequência).                          ║
# ║    2. Aquece: roda 3 iterações descartadas (JIT warm-up se Numba).       ║
# ║    3. Mede: roda N iterações cronometradas e calcula média/std.          ║
# ║    4. Calcula throughput: modelos/hora = 3600 / tempo_médio_segundos.    ║
# ║    5. Compara contra baselines:                                          ║
# ║         • Fortran OpenMP: 58 856 mod/h (registrado)                     ║
# ║         • Meta Python Numba: ≥ 40 000 mod/h (68% do Fortran)            ║
# ║    6. Imprime relatório markdown e opcionalmente grava CSV.              ║
# ║                                                                           ║
# ║  MODELOS DE TESTE                                                         ║
# ║    • small  : 3 camadas isotrópicas, 100 posições (smoke-test rápido)   ║
# ║    • medium : 7 camadas TIV, 300 posições (caso intermediário)          ║
# ║    • large  : 22 camadas TIV, 601 posições (paridade Fortran Sobol)     ║
# ║                                                                           ║
# ║  USO                                                                       ║
# ║    python -m geosteering_ai.simulation.benchmarks.bench_forward            ║
# ║    python -m geosteering_ai.simulation.benchmarks.bench_forward --size large --n-iter 50║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • docs/reference/plano_simulador_python_jax_numba.md §11               ║
# ║    • Fortran benchmark: 58 856 mod/h em 8 cores (MEMORY.md)             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Benchmark do forward completo do simulador Python (Sprint 2.7).

Mede o throughput em modelos/hora para diferentes tamanhos de modelo
e compara contra as metas de performance do projeto.

Example:
    Via linha de comando::

        $ python -m geosteering_ai.simulation.benchmarks.bench_forward

    Via API Python::

        >>> from geosteering_ai.simulation.benchmarks.bench_forward import (
        ...     run_benchmark, BenchmarkResult,
        ... )
        >>> result = run_benchmark(size="small", n_iter=10)
        >>> print(f"{result.throughput_models_per_hour:.0f} mod/h")
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from geosteering_ai.simulation.forward import simulate

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes: baselines de referência
# ──────────────────────────────────────────────────────────────────────────────
FORTRAN_BASELINE_MOD_PER_HOUR = 58_856  # Fortran OpenMP 8-cores (registrado)
PYTHON_TARGET_MOD_PER_HOUR = 40_000  # Meta do plano (≥ 68% do Fortran)


# ──────────────────────────────────────────────────────────────────────────────
# Perfis de modelo para benchmark
# ──────────────────────────────────────────────────────────────────────────────
def _make_model_small():
    """Modelo small: 3 camadas isotrópicas, 100 posições."""
    return {
        "name": "small_3layer_iso",
        "rho_h": np.array([1.0, 100.0, 1.0]),
        "rho_v": np.array([1.0, 100.0, 1.0]),
        "esp": np.array([5.0]),
        "positions_z": np.linspace(-2.0, 7.0, 100),
        "frequency_hz": 20000.0,
        "tr_spacing_m": 1.0,
        "dip_deg": 0.0,
        "n_layers": 3,
        "n_positions": 100,
    }


def _make_model_medium():
    """Modelo medium: 7 camadas TIV, 300 posições, dip=30°."""
    return {
        "name": "medium_7layer_tiv",
        "rho_h": np.array([1.0, 80.0, 1.0, 10.0, 1.0, 0.3, 1.0]),
        "rho_v": np.array([10.0, 80.0, 10.0, 10.0, 10.0, 0.3, 10.0]),
        "esp": np.array([1.52, 2.35, 2.10, 1.88, 0.92]),
        "positions_z": np.linspace(-5.0, 15.0, 300),
        "frequency_hz": 20000.0,
        "tr_spacing_m": 1.0,
        "dip_deg": 30.0,
        "n_layers": 7,
        "n_positions": 300,
    }


def _make_model_large():
    """Modelo large: 22 camadas TIV, 601 posições (paridade Sobol Fortran)."""
    np.random.seed(42)
    n = 22
    rho_h = np.exp(np.random.uniform(np.log(0.3), np.log(500.0), n))
    rho_v = rho_h * np.random.uniform(1.0, 5.0, n)  # anisotropia TIV
    esp = np.random.uniform(0.5, 5.0, n - 2)
    return {
        "name": "large_22layer_tiv_sobol",
        "rho_h": rho_h,
        "rho_v": rho_v,
        "esp": esp,
        "positions_z": np.linspace(-10.0, 80.0, 601),
        "frequency_hz": 20000.0,
        "tr_spacing_m": 1.0,
        "dip_deg": 0.0,
        "n_layers": n,
        "n_positions": 601,
    }


_MODEL_FACTORIES = {
    "small": _make_model_small,
    "medium": _make_model_medium,
    "large": _make_model_large,
}


# ──────────────────────────────────────────────────────────────────────────────
# BenchmarkResult — container de saída
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    """Container com os resultados de um benchmark forward.

    Attributes:
        model_name: Nome do modelo de benchmark.
        n_layers: Número de camadas do modelo.
        n_positions: Número de posições simuladas.
        n_iter: Número de iterações cronometradas.
        n_warmup: Número de iterações de aquecimento.
        times_seconds: Lista com tempo de cada iteração (s).
        mean_time_s: Tempo médio por modelo (s).
        std_time_s: Desvio padrão do tempo (s).
        throughput_models_per_hour: Throughput em modelos/hora.
        fortran_baseline: Baseline Fortran em mod/h.
        python_target: Meta Python em mod/h.
        pass_target: Se o throughput atingiu a meta.
        platform_info: Informações da plataforma (CPU, OS, Python).
        filter_name: Filtro Hankel usado.
        has_numba: Se Numba estava disponível durante o benchmark.
    """

    model_name: str
    n_layers: int
    n_positions: int
    n_iter: int
    n_warmup: int
    times_seconds: list = field(default_factory=list)
    mean_time_s: float = 0.0
    std_time_s: float = 0.0
    throughput_models_per_hour: float = 0.0
    fortran_baseline: float = FORTRAN_BASELINE_MOD_PER_HOUR
    python_target: float = PYTHON_TARGET_MOD_PER_HOUR
    pass_target: bool = False
    platform_info: str = ""
    filter_name: str = "werthmuller_201pt"
    has_numba: bool = False

    def to_markdown(self) -> str:
        """Gera relatório markdown do benchmark."""
        status = "APROVADO" if self.pass_target else "REPROVADO"
        pct_fortran = (
            100.0 * self.throughput_models_per_hour / self.fortran_baseline
            if self.fortran_baseline > 0
            else 0.0
        )
        return f"""## Benchmark Forward — {self.model_name}

| Métrica                    | Valor                              |
|:---------------------------|:-----------------------------------|
| Modelo                     | {self.model_name}                  |
| Camadas                    | {self.n_layers}                    |
| Posições                   | {self.n_positions}                 |
| Filtro Hankel              | {self.filter_name}                 |
| Numba disponível           | {'Sim' if self.has_numba else 'Não (NumPy puro)'}  |
| Iterações (warmup/medição) | {self.n_warmup} / {self.n_iter}    |
| Tempo médio por modelo     | {self.mean_time_s:.4f} s ± {self.std_time_s:.4f} s |
| **Throughput**             | **{self.throughput_models_per_hour:,.0f} mod/h**   |
| Meta Python Numba          | {self.python_target:,.0f} mod/h    |
| Baseline Fortran OpenMP    | {self.fortran_baseline:,.0f} mod/h |
| % do Fortran               | {pct_fortran:.1f}%                 |
| **Gate**                   | **{status}** {'✅' if self.pass_target else '❌'} |
| Plataforma                 | {self.platform_info}               |
"""


# ──────────────────────────────────────────────────────────────────────────────
# run_benchmark — função principal
# ──────────────────────────────────────────────────────────────────────────────
def run_benchmark(
    size: str = "small",
    n_iter: int = 10,
    n_warmup: int = 3,
    filter_name: str = "werthmuller_201pt",
    output_csv: Optional[str] = None,
) -> BenchmarkResult:
    """Executa o benchmark forward e retorna os resultados.

    Args:
        size: Tamanho do modelo: ``"small"`` (3 camadas, 100 pos),
            ``"medium"`` (7 camadas, 300 pos), ``"large"`` (22 camadas,
            601 pos).
        n_iter: Número de iterações cronometradas.
        n_warmup: Número de iterações de aquecimento (descartadas).
        filter_name: Filtro Hankel a usar.
        output_csv: Se fornecido, grava tempos em CSV neste caminho.

    Returns:
        :class:`BenchmarkResult` com todos os dados coletados.

    Example:
        >>> result = run_benchmark(size="small", n_iter=5)
        >>> result.throughput_models_per_hour > 0
        True
    """
    if size not in _MODEL_FACTORIES:
        raise ValueError(f"size={size!r} inválido. Opções: {sorted(_MODEL_FACTORIES)}")

    model = _MODEL_FACTORIES[size]()

    # ── Informações da plataforma ─────────────────────────────────
    from geosteering_ai.simulation._numba.propagation import HAS_NUMBA

    cpu_count = os.cpu_count() or 1
    platform_info = (
        f"{platform.system()} {platform.machine()} | "
        f"Python {sys.version.split()[0]} | "
        f"CPU cores: {cpu_count} | "
        f"Numba: {'sim' if HAS_NUMBA else 'não'}"
    )

    logger.info(
        "Benchmark %s: n_layers=%d, n_positions=%d, n_iter=%d+%d warmup",
        model["name"],
        model["n_layers"],
        model["n_positions"],
        n_warmup,
        n_iter,
    )

    # ── Aquecimento (JIT warm-up) ─────────────────────────────────
    for _ in range(n_warmup):
        simulate(
            rho_h=model["rho_h"],
            rho_v=model["rho_v"],
            esp=model["esp"],
            positions_z=model["positions_z"],
            frequency_hz=model["frequency_hz"],
            tr_spacing_m=model["tr_spacing_m"],
            dip_deg=model["dip_deg"],
            hankel_filter=filter_name,
        )

    # ── Medição ───────────────────────────────────────────────────
    times = []
    for i in range(n_iter):
        t0 = time.perf_counter()
        result = simulate(
            rho_h=model["rho_h"],
            rho_v=model["rho_v"],
            esp=model["esp"],
            positions_z=model["positions_z"],
            frequency_hz=model["frequency_hz"],
            tr_spacing_m=model["tr_spacing_m"],
            dip_deg=model["dip_deg"],
            hankel_filter=filter_name,
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        # Validação: sem NaN/Inf
        assert np.all(
            np.isfinite(result.H_tensor)
        ), f"Iteração {i}: NaN/Inf detectado no H_tensor"

    # ── Cálculos ──────────────────────────────────────────────────
    times_arr = np.array(times)
    mean_time = float(times_arr.mean())
    std_time = float(times_arr.std())
    throughput = 3600.0 / mean_time if mean_time > 0 else 0.0
    pass_target = throughput >= PYTHON_TARGET_MOD_PER_HOUR

    bench_result = BenchmarkResult(
        model_name=model["name"],
        n_layers=model["n_layers"],
        n_positions=model["n_positions"],
        n_iter=n_iter,
        n_warmup=n_warmup,
        times_seconds=times,
        mean_time_s=mean_time,
        std_time_s=std_time,
        throughput_models_per_hour=throughput,
        pass_target=pass_target,
        platform_info=platform_info,
        filter_name=filter_name,
        has_numba=HAS_NUMBA,
    )

    # ── Gravação CSV (opcional) ───────────────────────────────────
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "time_seconds"])
            for i, t in enumerate(times):
                writer.writerow([i + 1, f"{t:.6f}"])
        logger.info("Tempos gravados em %s", csv_path)

    return bench_result


# ──────────────────────────────────────────────────────────────────────────────
# run_all_sizes — roda os 3 tamanhos
# ──────────────────────────────────────────────────────────────────────────────
def run_all_sizes(
    n_iter: int = 10,
    filter_name: str = "werthmuller_201pt",
) -> dict[str, BenchmarkResult]:
    """Roda benchmarks para os 3 tamanhos (small, medium, large).

    Returns:
        Dict com chaves 'small', 'medium', 'large' e
        :class:`BenchmarkResult` como valores.
    """
    results = {}
    for size in ["small", "medium", "large"]:
        results[size] = run_benchmark(size=size, n_iter=n_iter, filter_name=filter_name)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Entry point para benchmark via CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark do forward do simulador Python EM 1D TIV."
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "all"],
        default="medium",
        help="Tamanho do modelo (default: medium).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Número de iterações cronometradas (default: 10).",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="Número de iterações de aquecimento (default: 3).",
    )
    parser.add_argument(
        "--filter",
        default="werthmuller_201pt",
        help="Filtro Hankel (default: werthmuller_201pt).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Caminho para gravar tempos em CSV.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.size == "all":
        results = run_all_sizes(n_iter=args.n_iter, filter_name=args.filter)
        for name, result in results.items():
            print(result.to_markdown())
    else:
        result = run_benchmark(
            size=args.size,
            n_iter=args.n_iter,
            n_warmup=args.n_warmup,
            filter_name=args.filter,
            output_csv=args.csv,
        )
        print(result.to_markdown())


if __name__ == "__main__":
    main()


__all__ = [
    "BenchmarkResult",
    "run_benchmark",
    "run_all_sizes",
    "FORTRAN_BASELINE_MOD_PER_HOUR",
    "PYTHON_TARGET_MOD_PER_HOUR",
]
