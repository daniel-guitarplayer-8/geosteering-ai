#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark local — Simulador Python Numba JIT vs Fortran tatu.x (OpenMP).

Compara desempenho CPU-paralelo máximo entre:

  • Python Numba JIT otimizado (`simulate_multi`, parallel=True, prange)
  • Fortran tatu.x v10.0 (`-fopenmp`, subprocess)

Executa duas configurações canônicas em 6 modelos geológicos canônicos
(Oklahoma 3/5/28, Devine, Hou, Viking Graben), reporta throughput
(modelos/hora), tempo por modelo, speedup e gera plots por modelo. Ao
final roda experimento com 30.000 perfis aleatórios de 20 camadas com
alta resistividade (>1000 Ω·m), aplicando as mesmas métricas.

╔══════════════════════════════════════════════════════════════════════╗
║ Configurações avaliadas                                              ║
╠══════════════════════════════════════════════════════════════════════╣
║ Config A : 1 freq (20 kHz), 1 TR (1 m), 1 dip (0°),  600 posições   ║
║ Config B : 2 freqs (20, 40 kHz), 1 TR (1 m), 2 dips (0°, 30°), 600  ║
╚══════════════════════════════════════════════════════════════════════╝

Uso:

    python benchmarks/bench_numba_vs_fortran_local.py
    python benchmarks/bench_numba_vs_fortran_local.py --fortran-n 1000
    python benchmarks/bench_numba_vs_fortran_local.py --skip-fortran

Saídas em ``benchmarks/results/`` (ou --output-dir):

    • bench_numba_vs_fortran_summary.md    — relatório Markdown final
    • bench_numba_vs_fortran_summary.csv   — tabela completa em CSV
    • plot_<modelo>_config_a.png           — plots Config A por modelo
    • plot_<modelo>_config_b.png           — plots Config B por modelo
    • plot_30k_summary.png                 — (opcional) distribuição 30k

Notas de metodologia:

    • Ambos os simuladores usam paralelização CPU máxima (cpu_count()).
    • Numba: warmup JIT antes do timing; média de N iterações (--n-iter).
    • Fortran: subprocess único por configuração; timing wall-clock.
    • Para o experimento 30k, Fortran roda em subconjunto configurável
      (--fortran-n, default 300) por causa do overhead de spawn de
      processo (~50-100 ms por chamada de tatu.x). Throughput é
      extrapolado e reportado claramente.
    • Para Config B, dip=30° produz internamente 693 posições no Fortran
      (n_pos = ceil(tj/(p_med·cos(θ)))). Python fixa positions_z = 600
      pontos para ambos os ângulos (conforme especificação do usuário).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Configuração de paralelismo máximo — DEVE ocorrer antes de import numba
# ═══════════════════════════════════════════════════════════════════════
_NUM_CPUS = os.cpu_count() or 1
os.environ.setdefault("NUMBA_NUM_THREADS", str(_NUM_CPUS))
os.environ.setdefault("OMP_NUM_THREADS", str(_NUM_CPUS))
os.environ.setdefault("MKL_NUM_THREADS", str(_NUM_CPUS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_NUM_CPUS))

# Imports que dependem dos env vars acima
from geosteering_ai.simulation import (  # noqa: E402
    MultiSimulationResult,
    SimulationConfig,
    simulate_multi,
)
from geosteering_ai.simulation.io.binary_dat_multi import (  # noqa: E402
    export_multi_tr_dat,
    read_multi_tr_dat,
)
from geosteering_ai.simulation.validation.canonical_models import (  # noqa: E402
    get_canonical_model,
)

# ═══════════════════════════════════════════════════════════════════════
# Constantes estruturais do benchmark
# ═══════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parent.parent
FORTRAN_EXEC = REPO_ROOT / "Fortran_Gerador" / "tatu.x"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results"

# Modelos canônicos testados (Oklahoma 3/5/28 + Devine + Hou + Viking Graben)
CANONICAL_MODELS: Tuple[str, ...] = (
    "oklahoma_3",
    "oklahoma_5",
    "oklahoma_28",
    "devine_8",
    "hou_7",
    "viking_graben_10",
)

# Grade de medição (h1 = profundidade inicial do midpoint; tj = janela de
# investigação; p_med = passo). Para dip=0°, isso produz exatamente 600
# posições. Para dip=30°, n_pos = ceil(120/(0.2·cos30°)) = 693 no Fortran.
H1_DEPTH = 29.5418
TJ_WINDOW = 120.0
P_MED = 0.2
N_POS_TARGET = 600

# Configuração A: 1 freq, 1 TR, 1 dip → throughput baseline
CONFIG_A = {
    "label": "A",
    "freqs_hz": [20_000.0],
    "tr_list_m": [1.0],
    "dip_list_deg": [0.0],
    "n_pos": N_POS_TARGET,
}

# Configuração B: 2 freqs, 1 TR, 2 dips → carga 4× vs A
CONFIG_B = {
    "label": "B",
    "freqs_hz": [20_000.0, 40_000.0],
    "tr_list_m": [1.0],
    "dip_list_deg": [0.0, 30.0],
    "n_pos": N_POS_TARGET,
}

# Índices das 9 componentes no tensor H (flatten row-major C×C):
#   0:Hxx  1:Hxy  2:Hxz  3:Hyx  4:Hyy  5:Hyz  6:Hzx  7:Hzy  8:Hzz
IDX_HXX, IDX_HYY, IDX_HZZ = 0, 4, 8
IDX_HXZ = 2

# Cosmético: labels amigáveis para plots/tabelas
MODEL_DISPLAY_NAME = {
    "oklahoma_3": "Oklahoma 3 camadas (TIV)",
    "oklahoma_5": "Oklahoma 5 camadas (TIV)",
    "oklahoma_28": "Oklahoma 28 camadas (TIV forte)",
    "devine_8": "Devine 8 camadas (isotrópico)",
    "hou_7": "Hou 7 camadas (TIV)",
    "viking_graben_10": "Viking Graben 10 camadas (TIV)",
}


# ═══════════════════════════════════════════════════════════════════════
# Tipo para registro de resultado por configuração
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class BenchRecord:
    """Registro de benchmark para uma dupla (modelo, configuração).

    Armazena tempos absolutos e throughput de Numba e Fortran, além do
    speedup Numba/Fortran e um campo opcional de paridade numérica
    (max_abs_err) quando o resultado .dat é comparável.
    """

    model: str
    config: str  # "A" ou "B"
    n_pos: int
    nf: int
    n_tr: int
    n_ang: int
    n_models_per_call: int  # combinações (TR × Ang × freq) por 1 chamada
    numba_ms: Optional[float] = None
    fortran_ms: Optional[float] = None
    numba_throughput: Optional[float] = None  # mod/h
    fortran_throughput: Optional[float] = None  # mod/h
    numba_ms_per_submodel: Optional[float] = None
    fortran_ms_per_submodel: Optional[float] = None
    speedup: Optional[float] = None  # Fortran / Numba
    max_abs_err: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════
# Runner Fortran — escreve model.in, executa tatu.x, parseia .dat
# ═══════════════════════════════════════════════════════════════════════
def _write_model_in(
    workdir: Path,
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    freqs_hz: Sequence[float],
    dip_list: Sequence[float],
    tr_list: Sequence[float],
    *,
    h1: float = H1_DEPTH,
    tj: float = TJ_WINDOW,
    p_med: float = P_MED,
    filename: str = "bench",
) -> None:
    """Escreve o arquivo ``model.in`` consumido pelo binário tatu.x.

    Args:
        workdir: Diretório de trabalho (conterá model.in e tatu.x).
        rho_h/rho_v: Resistividades horizontais/verticais por camada.
        esp: Espessuras das camadas internas (n_layers-2).
        freqs_hz: Lista de frequências em Hz. nf > 1 aciona F5.
        dip_list: Ângulos (graus). 1 ou mais.
        tr_list: Espaçamentos T-R (m). 1 ou mais.
        h1, tj, p_med: Geometria de medição (profundidade inicial,
            janela, passo do midpoint).
        filename: Base para arquivos de saída do Fortran.
    """
    nf = len(freqs_hz)
    freq_lines = "\n".join(f"{float(f):.6e}" for f in freqs_hz)
    dip_lines = "\n".join(f"{float(d):.6f}" for d in dip_list)
    tr_lines = "\n".join(f"{float(L):.6f}" for L in tr_list)
    rho_lines = "\n".join(f"{rh:.10e} {rv:.10e}" for rh, rv in zip(rho_h, rho_v))
    esp_lines = "\n".join(f"{e:.10e}" for e in esp)

    content = (
        f"{nf}\n{freq_lines}\n"
        f"{len(dip_list)}\n{dip_lines}\n"
        f"{h1}\n{tj}\n{p_med}\n"
        f"{len(tr_list)}\n{tr_lines}\n"
        f"{filename}\n"
        f"{len(rho_h)}\n{rho_lines}\n"
    )
    if esp_lines:
        content += esp_lines + "\n"
    content += "1 1\n"
    # F5 (use_arbitrary_freq) obrigatório quando nf > 1 com freqs
    # arbitrárias. tatu.x ignora flag se ausente (nf=1).
    if nf > 1:
        content += "1\n"  # use_arbitrary_freq = 1

    (workdir / "model.in").write_text(content)


def run_fortran(
    workdir: Path,
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    config: dict,
    *,
    filename: str = "bench",
) -> float:
    """Executa tatu.x em ``workdir`` e retorna tempo wall-clock (s).

    Pré-condição: ``tatu.x`` já copiado para ``workdir``. O binário é
    invocado via subprocess com OMP_NUM_THREADS=cpu_count() (herdado do
    ambiente pai).
    """
    _write_model_in(
        workdir,
        rho_h, rho_v, esp,
        freqs_hz=config["freqs_hz"],
        dip_list=config["dip_list_deg"],
        tr_list=config["tr_list_m"],
        filename=filename,
    )
    t0 = time.perf_counter()
    r = subprocess.run(
        ["./tatu.x"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(
            f"tatu.x falhou (rc={r.returncode}):\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    return elapsed


# ═══════════════════════════════════════════════════════════════════════
# Runner Numba — chama simulate_multi com paralelismo máximo
# ═══════════════════════════════════════════════════════════════════════
def _build_positions_z(dip_deg: float, n_pos: int = N_POS_TARGET) -> np.ndarray:
    """Gera vetor positions_z alinhado com a grade do Fortran p/ dip=dip_deg.

    Usa h1=29.5418, p_med=0.2 e pz=p_med·cos(dip) — igual ao Fortran.
    Retorna vetor de profundidades crescentes no sistema Z↑ (valores
    negativos indicam midpoints acima do nível zero da formação).
    """
    pz = P_MED * float(np.cos(np.deg2rad(dip_deg)))
    return -H1_DEPTH + np.arange(n_pos) * pz


def run_numba(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    config: dict,
    *,
    n_iter: int = 3,
    cfg: Optional[SimulationConfig] = None,
    do_warmup: bool = True,
) -> Tuple[float, MultiSimulationResult]:
    """Executa ``simulate_multi`` com warmup + n_iter chamadas cronometradas.

    Retorna (tempo médio por chamada em segundos, resultado da última
    chamada). A config ``SimulationConfig`` default já habilita
    parallel=True e num_threads=-1 (auto = cpu_count()).
    """
    if cfg is None:
        cfg = SimulationConfig(
            frequency_hz=float(config["freqs_hz"][0]),
            frequencies_hz=list(map(float, config["freqs_hz"])),
            backend="numba",
            parallel=True,
            num_threads=-1,
            hankel_filter="werthmuller_201pt",
        )

    positions_z = _build_positions_z(
        dip_deg=float(config["dip_list_deg"][0]),
        n_pos=int(config["n_pos"]),
    )

    kwargs = dict(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=list(map(float, config["freqs_hz"])),
        tr_spacings_m=list(map(float, config["tr_list_m"])),
        dip_degs=list(map(float, config["dip_list_deg"])),
        cfg=cfg,
    )

    # Warmup — força compilação JIT (primeira chamada ~5-10 s)
    if do_warmup:
        _ = simulate_multi(**kwargs)

    t0 = time.perf_counter()
    result: MultiSimulationResult = simulate_multi(**kwargs)
    for _ in range(n_iter - 1):
        result = simulate_multi(**kwargs)
    elapsed = (time.perf_counter() - t0) / n_iter
    return elapsed, result


# ═══════════════════════════════════════════════════════════════════════
# Comparação de paridade numérica (.dat Fortran vs .dat Python)
# ═══════════════════════════════════════════════════════════════════════
def compare_dats(
    fort_dir: Path, py_dir: Path, n_tr: int, n_records: int,
) -> float:
    """Retorna max_abs_err entre .dat Fortran e Python (9 comp).

    Esperamos que ambos tenham o mesmo número de registros (n_records =
    nAng × nf × n_pos). Para Config B com dip>0 o Fortran usa n_pos
    variável por ângulo — nesse caso retornamos NaN (paridade não é
    comparável record-a-record sem remapear).
    """
    max_err = 0.0
    for i_tr in range(1, n_tr + 1):
        suffix = f"_TR{i_tr}" if n_tr > 1 else ""
        fort_dat = fort_dir / f"bench{suffix}.dat"
        py_dat = py_dir / f"bench{suffix}.dat"
        if not fort_dat.exists() or not py_dat.exists():
            return float("nan")
        try:
            fort_data = read_multi_tr_dat(fort_dat, n_records)
            py_data = read_multi_tr_dat(py_dat, n_records)
        except Exception:
            return float("nan")
        for field_name in fort_data.dtype.names:
            if field_name == "i":
                continue
            diff = np.nanmax(np.abs(fort_data[field_name] - py_data[field_name]))
            if np.isfinite(diff) and diff > max_err:
                max_err = float(diff)
    return max_err


# ═══════════════════════════════════════════════════════════════════════
# Benchmark por configuração canônica (6 modelos × 2 configs)
# ═══════════════════════════════════════════════════════════════════════
def benchmark_one_config(
    model_name: str,
    config: dict,
    *,
    n_iter: int,
    skip_fortran: bool,
) -> Tuple[BenchRecord, Optional[MultiSimulationResult]]:
    """Roda Numba + Fortran (se disponível) para (modelo, config)."""
    m = get_canonical_model(model_name)
    rho_h = np.asarray(m.rho_h, dtype=np.float64)
    rho_v = np.asarray(m.rho_v, dtype=np.float64)
    esp = np.asarray(m.esp, dtype=np.float64)

    n_tr = len(config["tr_list_m"])
    n_ang = len(config["dip_list_deg"])
    nf = len(config["freqs_hz"])
    n_pos = int(config["n_pos"])
    # "Submodelo" = 1 combinação (TR × Ang × freq) — unidade de dado física
    n_submodels = n_tr * n_ang * nf

    record = BenchRecord(
        model=model_name,
        config=config["label"],
        n_pos=n_pos,
        nf=nf,
        n_tr=n_tr,
        n_ang=n_ang,
        n_models_per_call=n_submodels,
    )

    # ── Numba ─────────────────────────────────────────────────────────
    t_py, py_res = run_numba(
        rho_h, rho_v, esp, config, n_iter=n_iter, do_warmup=True,
    )
    record.numba_ms = t_py * 1000.0
    record.numba_throughput = 3600.0 / t_py  # 1 chamada = 1 "modelo" geológico
    record.numba_ms_per_submodel = (t_py / n_submodels) * 1000.0

    # ── Fortran ───────────────────────────────────────────────────────
    if skip_fortran or not FORTRAN_EXEC.exists():
        return record, py_res

    with tempfile.TemporaryDirectory() as tmp:
        fort_dir = Path(tmp) / "fort"
        fort_dir.mkdir()
        shutil.copy(FORTRAN_EXEC, fort_dir / "tatu.x")
        try:
            t_fort = run_fortran(fort_dir, rho_h, rho_v, esp, config)
            record.fortran_ms = t_fort * 1000.0
            record.fortran_throughput = 3600.0 / t_fort
            record.fortran_ms_per_submodel = (t_fort / n_submodels) * 1000.0
            record.speedup = t_fort / t_py if t_py > 0 else float("nan")
        except Exception as e:
            print(f"    ⚠️  Fortran falhou: {e}")
            return record, py_res

        # Paridade numérica apenas para Config A (dip=0° uniforme)
        if (config["label"] == "A") and (t_fort > 0):
            py_dir = Path(tmp) / "py"
            py_dir.mkdir()
            try:
                export_multi_tr_dat(py_res, "bench", py_dir)
                n_records = n_ang * nf * n_pos
                record.max_abs_err = compare_dats(
                    fort_dir, py_dir, n_tr=n_tr, n_records=n_records,
                )
            except Exception as e:
                print(f"    ⚠️  Paridade falhou: {e}")

    return record, py_res


def run_canonical_benchmarks(
    *, n_iter: int, skip_fortran: bool,
) -> Tuple[List[BenchRecord], Dict[Tuple[str, str], MultiSimulationResult]]:
    """Varre 6 modelos × 2 configs; coleta métricas e resultados."""
    records: List[BenchRecord] = []
    results: Dict[Tuple[str, str], MultiSimulationResult] = {}

    for model_name in CANONICAL_MODELS:
        for cfg in (CONFIG_A, CONFIG_B):
            label = f"{model_name} — Config {cfg['label']}"
            print(f"  ▶ {label}")
            rec, res = benchmark_one_config(
                model_name, cfg, n_iter=n_iter, skip_fortran=skip_fortran,
            )
            records.append(rec)
            if res is not None:
                results[(model_name, cfg["label"])] = res
            _print_record_line(rec)

    return records, results


def _print_record_line(r: BenchRecord) -> None:
    """Imprime uma linha compacta por benchmark."""
    nb = f"{r.numba_ms:7.1f} ms" if r.numba_ms else "   —   "
    ft = f"{r.fortran_ms:7.1f} ms" if r.fortran_ms else "   —   "
    sp = f"{r.speedup:5.2f}×" if r.speedup else "  — "
    thr_nb = f"{r.numba_throughput:7.0f}" if r.numba_throughput else "    —"
    thr_ft = f"{r.fortran_throughput:7.0f}" if r.fortran_throughput else "    —"
    err = f"{r.max_abs_err:.2e}" if r.max_abs_err else "—"
    print(
        f"    Numba: {nb} | Fortran: {ft} | Speedup: {sp} | "
        f"mod/h Numba: {thr_nb} | Fortran: {thr_ft} | err: {err}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Experimento 30k — perfis aleatórios 20 camadas, alta resistividade
# ═══════════════════════════════════════════════════════════════════════
def generate_random_profile(
    rng: np.random.Generator, n_layers: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera perfil aleatório com alta resistividade (ρh > 1000 Ω·m).

    Distribuição log-uniforme ρh ∈ [1000, 10000] Ω·m em todas as
    camadas; TIV com fator de anisotropia ρv/ρh ∈ [1.0, 3.0];
    espessuras esp ∈ [0.5, 5.0] m. Retorna (rho_h, rho_v, esp).
    """
    # ρh log-uniforme ∈ [1000, 10000] Ω·m — alta resistividade dominante
    log_rho = rng.uniform(np.log10(1000.0), np.log10(10000.0), size=n_layers)
    rho_h = 10.0 ** log_rho

    # Anisotropia TIV λ ∈ [1, 3] (ρv ≥ ρh)
    aniso = rng.uniform(1.0, 3.0, size=n_layers)
    rho_v = rho_h * aniso

    # Espessuras internas (n_layers-2)
    esp = rng.uniform(0.5, 5.0, size=n_layers - 2)
    return rho_h, rho_v, esp


def run_30k_experiment(
    *,
    n_models: int = 30_000,
    fortran_n: int = 300,
    skip_fortran: bool = False,
    n_layers: int = 20,
    seed: int = 42,
) -> Dict[str, object]:
    """Executa o experimento 30k: Numba full + Fortran subset.

    Args:
        n_models: Total de perfis aleatórios (Numba completo).
        fortran_n: Subset Fortran (extrapolação para n_models).
        skip_fortran: Se True, roda apenas Numba.
        n_layers: 20 camadas conforme especificação.
        seed: Reprodutibilidade.

    Returns:
        Dicionário com métricas consolidadas.
    """
    print(f"\n  Gerando {n_models:,} perfis aleatórios (20 camadas, ρh>1000)")
    rng = np.random.default_rng(seed)
    profiles = [generate_random_profile(rng, n_layers) for _ in range(n_models)]
    print(f"  ✓ ρh min/max global: "
          f"[{min(p[0].min() for p in profiles):.1f}, "
          f"{max(p[0].max() for p in profiles):.1f}] Ω·m")

    cfg_np = SimulationConfig(
        frequency_hz=float(CONFIG_A["freqs_hz"][0]),
        frequencies_hz=list(map(float, CONFIG_A["freqs_hz"])),
        backend="numba",
        parallel=True,
        num_threads=-1,
        hankel_filter="werthmuller_201pt",
    )
    positions_z = _build_positions_z(
        dip_deg=float(CONFIG_A["dip_list_deg"][0]),
        n_pos=int(CONFIG_A["n_pos"]),
    )

    # ── Warmup JIT Numba (1 chamada descartada) ─────────────────────
    rho_h0, rho_v0, esp0 = profiles[0]
    _ = simulate_multi(
        rho_h=rho_h0, rho_v=rho_v0, esp=esp0, positions_z=positions_z,
        frequencies_hz=list(map(float, CONFIG_A["freqs_hz"])),
        tr_spacings_m=list(map(float, CONFIG_A["tr_list_m"])),
        dip_degs=list(map(float, CONFIG_A["dip_list_deg"])),
        cfg=cfg_np,
    )

    # ── Numba: 30k chamadas cronometradas ─────────────────────────
    print(f"  ▶ Numba ({n_models:,} modelos)...")
    t0 = time.perf_counter()
    for rho_h, rho_v, esp in profiles:
        _ = simulate_multi(
            rho_h=rho_h, rho_v=rho_v, esp=esp, positions_z=positions_z,
            frequencies_hz=list(map(float, CONFIG_A["freqs_hz"])),
            tr_spacings_m=list(map(float, CONFIG_A["tr_list_m"])),
            dip_degs=list(map(float, CONFIG_A["dip_list_deg"])),
            cfg=cfg_np,
        )
    numba_total = time.perf_counter() - t0
    numba_ms_per_model = (numba_total / n_models) * 1000.0
    numba_throughput = n_models / (numba_total / 3600.0)
    print(f"    ✓ Total: {numba_total:.1f} s | "
          f"{numba_ms_per_model:.2f} ms/modelo | "
          f"{numba_throughput:,.0f} mod/h")

    out = {
        "n_models_total": n_models,
        "numba_total_s": numba_total,
        "numba_ms_per_model": numba_ms_per_model,
        "numba_throughput_per_hour": numba_throughput,
        "fortran_subset_n": 0,
        "fortran_total_s": None,
        "fortran_ms_per_model": None,
        "fortran_throughput_per_hour": None,
        "fortran_total_extrapolated_s": None,
        "speedup": None,
    }

    if skip_fortran or not FORTRAN_EXEC.exists():
        return out

    # ── Fortran: subset cronometrado (com extrapolação) ──────────
    n_subset = min(fortran_n, n_models)
    print(f"  ▶ Fortran subset ({n_subset} modelos, extrapolação p/ {n_models:,})...")
    with tempfile.TemporaryDirectory() as tmp:
        fort_dir = Path(tmp) / "fort"
        fort_dir.mkdir()
        shutil.copy(FORTRAN_EXEC, fort_dir / "tatu.x")
        t0 = time.perf_counter()
        for rho_h, rho_v, esp in profiles[:n_subset]:
            _write_model_in(
                fort_dir, rho_h, rho_v, esp,
                freqs_hz=CONFIG_A["freqs_hz"],
                dip_list=CONFIG_A["dip_list_deg"],
                tr_list=CONFIG_A["tr_list_m"],
            )
            r = subprocess.run(
                ["./tatu.x"], cwd=fort_dir, capture_output=True,
                text=True, timeout=300,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"tatu.x falhou no modelo subset: {r.stderr}"
                )
        fortran_subset_total = time.perf_counter() - t0

    fortran_ms_per_model = (fortran_subset_total / n_subset) * 1000.0
    fortran_throughput = n_subset / (fortran_subset_total / 3600.0)
    fortran_extrapolated_total = fortran_ms_per_model * n_models / 1000.0
    speedup = fortran_ms_per_model / numba_ms_per_model

    out.update({
        "fortran_subset_n": n_subset,
        "fortran_total_s": fortran_subset_total,
        "fortran_ms_per_model": fortran_ms_per_model,
        "fortran_throughput_per_hour": fortran_throughput,
        "fortran_total_extrapolated_s": fortran_extrapolated_total,
        "speedup": speedup,
    })
    print(f"    ✓ Subset {n_subset}: {fortran_subset_total:.1f} s | "
          f"{fortran_ms_per_model:.2f} ms/modelo | "
          f"{fortran_throughput:,.0f} mod/h")
    print(f"    ✓ Extrapolação p/ 30k: {fortran_extrapolated_total/60:.1f} min | "
          f"Speedup Numba/Fortran: {speedup:.2f}×")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Plots por modelo testado (6 modelos × 2 configs)
# ═══════════════════════════════════════════════════════════════════════
def plot_model_results(
    model_name: str,
    result_a: MultiSimulationResult,
    result_b: MultiSimulationResult,
    output_dir: Path,
) -> None:
    """Gera dois PNGs por modelo: Config A (3 painéis) e Config B (4 painéis).

    Config A: perfil ρh/ρv + |Hxx|, |Hzz| vs profundidade
    Config B: perfil + |Hxx| (4 curvas) + |Hzz| (4 curvas) + razão Hxx/Hzz
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib não disponível — pulando plots")
        return

    m = get_canonical_model(model_name)
    display = MODEL_DISPLAY_NAME.get(model_name, model_name)

    # ── Plot Config A ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"{display} — Config A (1 freq, 1 TR, dip=0°, {N_POS_TARGET} posições)",
        fontsize=13,
    )

    # (a) Perfil de resistividade — interfaces acumuladas, topo↓
    ax = axes[0]
    interfaces = np.concatenate([[0.0], np.cumsum(m.esp)])
    # Plota ρh e ρv como stair (constante por camada). Para as duas
    # camadas de meio-espaço, usamos extensão simbólica ±5 m.
    depths = [interfaces[0] - 5.0]
    for i, z_interface in enumerate(interfaces):
        depths.append(z_interface)
    depths.append(interfaces[-1] + 5.0)
    depths = np.asarray(depths)
    # Usa plot step-post para perfil fiel
    for r_arr, lab, ls in [(m.rho_h, "ρ_h", "-"), (m.rho_v, "ρ_v", "--")]:
        x = np.repeat(r_arr, 2)
        y = np.repeat(np.concatenate([[depths[0]], interfaces, [depths[-1]]]), 2)[1:-1]
        ax.plot(x, y, ls, label=lab, linewidth=1.6)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Resistividade (Ω·m)")
    ax.set_ylabel("Profundidade (m)")
    ax.set_title("Perfil de resistividade")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Magnitude Hxx e Hzz vs z_obs (meio z = midpoint)
    ax = axes[1]
    H = result_a.H_tensor  # (1, 1, n_pos, 1, 9)
    z_obs = result_a.z_obs[0]  # (n_pos,) — dip=0
    Hxx = H[0, 0, :, 0, IDX_HXX]
    Hzz = H[0, 0, :, 0, IDX_HZZ]
    ax.semilogy(z_obs, np.abs(Hxx), label="|Hxx|", linewidth=1.2)
    ax.semilogy(z_obs, np.abs(Hzz), label="|Hzz|", linewidth=1.2, linestyle="--")
    ax.set_xlabel("z_obs (m)")
    ax.set_ylabel("|H| (A/m)")
    ax.set_title("Amplitude magnética")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # (c) Re/Im Hxx
    ax = axes[2]
    ax.plot(z_obs, np.real(Hxx), label="Re(Hxx)", linewidth=1.2)
    ax.plot(z_obs, np.imag(Hxx), label="Im(Hxx)", linewidth=1.2, linestyle="--")
    ax.set_xlabel("z_obs (m)")
    ax.set_ylabel("Hxx (A/m)")
    ax.set_title("Parte real/imaginária de Hxx")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_a = output_dir / f"plot_{model_name}_config_a.png"
    plt.savefig(out_a, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Plot Config B ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"{display} — Config B (2 freqs [20, 40 kHz], 1 TR, dips [0°, 30°])",
        fontsize=13,
    )

    # (a) Perfil (mesmo da Config A)
    ax = axes[0]
    for r_arr, lab, ls in [(m.rho_h, "ρ_h", "-"), (m.rho_v, "ρ_v", "--")]:
        x = np.repeat(r_arr, 2)
        y = np.repeat(np.concatenate([[depths[0]], interfaces, [depths[-1]]]), 2)[1:-1]
        ax.plot(x, y, ls, label=lab, linewidth=1.6)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Resistividade (Ω·m)")
    ax.set_ylabel("Profundidade (m)")
    ax.set_title("Perfil de resistividade")
    ax.legend()
    ax.grid(alpha=0.3)

    H = result_b.H_tensor  # (1, 2, n_pos, 2, 9)
    freqs = result_b.freqs_hz
    dips = result_b.dip_degs
    z_obs_all = result_b.z_obs  # (2, n_pos)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # (b) |Hxx| 4 curvas (2 freq × 2 dip)
    ax = axes[1]
    k = 0
    for i_ang in range(2):
        for i_f in range(2):
            z = z_obs_all[i_ang]
            Hxx = H[0, i_ang, :, i_f, IDX_HXX]
            ax.semilogy(
                z, np.abs(Hxx),
                label=f"dip={dips[i_ang]:.0f}°, f={freqs[i_f]/1e3:.0f} kHz",
                color=colors[k], linewidth=1.1,
            )
            k += 1
    ax.set_xlabel("z_obs (m)")
    ax.set_ylabel("|Hxx| (A/m)")
    ax.set_title("Amplitude Hxx")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (c) |Hzz| 4 curvas
    ax = axes[2]
    k = 0
    for i_ang in range(2):
        for i_f in range(2):
            z = z_obs_all[i_ang]
            Hzz = H[0, i_ang, :, i_f, IDX_HZZ]
            ax.semilogy(
                z, np.abs(Hzz),
                label=f"dip={dips[i_ang]:.0f}°, f={freqs[i_f]/1e3:.0f} kHz",
                color=colors[k], linewidth=1.1,
            )
            k += 1
    ax.set_xlabel("z_obs (m)")
    ax.set_ylabel("|Hzz| (A/m)")
    ax.set_title("Amplitude Hzz")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (d) Razão |Hxx/Hzz| — sinal clássico de anisotropia
    ax = axes[3]
    k = 0
    for i_ang in range(2):
        for i_f in range(2):
            z = z_obs_all[i_ang]
            Hxx = H[0, i_ang, :, i_f, IDX_HXX]
            Hzz = H[0, i_ang, :, i_f, IDX_HZZ]
            ratio = np.abs(Hxx) / np.abs(Hzz)
            ax.semilogy(
                z, ratio,
                label=f"dip={dips[i_ang]:.0f}°, f={freqs[i_f]/1e3:.0f} kHz",
                color=colors[k], linewidth=1.1,
            )
            k += 1
    ax.set_xlabel("z_obs (m)")
    ax.set_ylabel("|Hxx/Hzz|")
    ax.set_title("Razão |Hxx/Hzz| (marcador de anisotropia)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out_b = output_dir / f"plot_{model_name}_config_b.png"
    plt.savefig(out_b, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"    ✓ Plots: {out_a.name}, {out_b.name}")


# ═══════════════════════════════════════════════════════════════════════
# Relatórios (CSV + Markdown)
# ═══════════════════════════════════════════════════════════════════════
def write_csv(records: List[BenchRecord], path: Path) -> None:
    """Grava tabela CSV completa (1 linha por modelo × config)."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def write_markdown(
    records: List[BenchRecord],
    exp_30k: Optional[Dict[str, object]],
    path: Path,
    *,
    fortran_available: bool,
    n_iter: int,
) -> None:
    """Gera relatório Markdown consolidado com as 3 tabelas do benchmark."""
    from datetime import datetime

    def fmt(v: Optional[float], specs: str = ".1f") -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"{v:{specs}}"

    lines = [
        "# Benchmark Local — Python Numba JIT vs Fortran tatu.x (OpenMP)",
        "",
        f"**Data**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Host**: CPU local ({_NUM_CPUS} threads, "
        f"NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS')}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})  ",
        f"**Numba warmup**: 1 chamada (JIT) descartada; timing = média de "
        f"N={n_iter} chamadas  ",
        f"**Fortran**: `tatu.x` v10.0 via subprocess; OpenMP herdado do ambiente  ",
        "",
        "## Metodologia",
        "",
        "- **Modelo = 1 chamada**: `mod/h` conta chamadas completas do "
        "simulador (1 `simulate_multi()` ou 1 execução de `tatu.x`), "
        "independente de número de sub-configurações (TR × ângulo × freq) "
        "produzidas por chamada.",
        "- **Submodelo**: 1 combinação TR × ângulo × frequência. "
        "`ms/submodelo` reporta custo computacional por resposta EM.",
        "- **Speedup = Fortran / Numba**: razão de tempos absolutos por "
        "chamada. Valores >1× indicam Numba mais rápido.",
        "- **Paridade (`max_abs_err`)**: comparação record-a-record dos "
        "arquivos `.dat` exportados. Apenas reportada para Config A "
        "(dip=0°, n_pos idêntico entre simuladores).",
        "",
        "## Configurações",
        "",
        "| Config | Freqs (kHz) | TRs (m) | Dips (°) | n_pos | Submodelos/chamada |",
        "|:------:|:-----------:|:-------:|:--------:|:-----:|:------------------:|",
        f"| **A** | 20 | 1.0 | 0 | {N_POS_TARGET} | 1 |",
        f"| **B** | 20, 40 | 1.0 | 0, 30 | {N_POS_TARGET} | 4 |",
        "",
        "## Tabela 1 — Config A (baseline: 1 freq × 1 TR × 1 dip × 600 pos)",
        "",
        "| Modelo | n_pos | Numba (ms) | Fortran (ms) | Numba mod/h | "
        "Fortran mod/h | Speedup | max_abs_err |",
        "|:-------|------:|-----------:|-------------:|------------:|"
        "--------------:|:-------:|:-----------:|",
    ]

    for r in [x for x in records if x.config == "A"]:
        lines.append(
            f"| {MODEL_DISPLAY_NAME.get(r.model, r.model)} "
            f"| {r.n_pos} "
            f"| {fmt(r.numba_ms)} "
            f"| {fmt(r.fortran_ms)} "
            f"| {fmt(r.numba_throughput, ',.0f')} "
            f"| {fmt(r.fortran_throughput, ',.0f')} "
            f"| {fmt(r.speedup, '.2f') + ('×' if r.speedup else '')} "
            f"| {fmt(r.max_abs_err, '.2e')} |"
        )

    lines += [
        "",
        "## Tabela 2 — Config B (carga 4×: 2 freqs × 1 TR × 2 dips × 600 pos)",
        "",
        "| Modelo | Numba (ms) | Fortran (ms) | ms/submodelo Numba | "
        "ms/submodelo Fortran | Numba mod/h | Fortran mod/h | Speedup |",
        "|:-------|-----------:|-------------:|-------------------:|"
        "---------------------:|------------:|--------------:|:-------:|",
    ]
    for r in [x for x in records if x.config == "B"]:
        lines.append(
            f"| {MODEL_DISPLAY_NAME.get(r.model, r.model)} "
            f"| {fmt(r.numba_ms)} "
            f"| {fmt(r.fortran_ms)} "
            f"| {fmt(r.numba_ms_per_submodel, '.2f')} "
            f"| {fmt(r.fortran_ms_per_submodel, '.2f')} "
            f"| {fmt(r.numba_throughput, ',.0f')} "
            f"| {fmt(r.fortran_throughput, ',.0f')} "
            f"| {fmt(r.speedup, '.2f') + ('×' if r.speedup else '')} |"
        )

    # ── Tabela 3: experimento 30k ────────────────────────────────
    if exp_30k:
        lines += [
            "",
            "## Tabela 3 — Experimento 30.000 perfis aleatórios (20 camadas, ρh>1000 Ω·m)",
            "",
            "Configuração: 1 freq (20 kHz), 1 TR (1 m), 1 dip (0°), "
            f"{N_POS_TARGET} posições; perfis log-uniformes ρh∈[1000,10000] "
            "Ω·m, TIV λ∈[1,3], esp∈[0.5, 5] m.",
            "",
        ]
        total = exp_30k["n_models_total"]
        n_ft = exp_30k.get("fortran_subset_n", 0)
        extr_note = (
            f"  (subset {n_ft}, extrapolado para {total:,})"
            if n_ft and n_ft != total else ""
        )
        lines += [
            f"| Métrica | Numba | Fortran{extr_note} |",
            "|:--------|------:|--------:|",
            f"| Total de modelos | {total:,} | "
            f"{n_ft:,} | ",
            f"| Tempo total (s) | "
            f"{fmt(exp_30k['numba_total_s'], '.1f')} | "
            f"{fmt(exp_30k.get('fortran_total_s'), '.1f')} |",
            f"| Tempo por modelo (ms) | "
            f"{fmt(exp_30k['numba_ms_per_model'], '.2f')} | "
            f"{fmt(exp_30k.get('fortran_ms_per_model'), '.2f')} |",
            f"| Throughput (mod/h) | "
            f"{fmt(exp_30k['numba_throughput_per_hour'], ',.0f')} | "
            f"{fmt(exp_30k.get('fortran_throughput_per_hour'), ',.0f')} |",
        ]
        if exp_30k.get("fortran_total_extrapolated_s"):
            lines.append(
                f"| Total extrapolado 30k (s) | — | "
                f"{fmt(exp_30k['fortran_total_extrapolated_s'], '.1f')} |"
            )
        if exp_30k.get("speedup") is not None:
            lines.append(
                f"| **Speedup Fortran/Numba** | — | "
                f"**{exp_30k['speedup']:.2f}×** |"
            )

    # ── Estatísticas agregadas ───────────────────────────────────
    lines += ["", "## Estatísticas agregadas", ""]
    thrs_nb_a = [r.numba_throughput for r in records
                 if r.config == "A" and r.numba_throughput]
    thrs_nb_b = [r.numba_throughput for r in records
                 if r.config == "B" and r.numba_throughput]
    speedups_a = [r.speedup for r in records if r.config == "A" and r.speedup]
    speedups_b = [r.speedup for r in records if r.config == "B" and r.speedup]

    if thrs_nb_a:
        lines.append(
            f"- **Config A** — Numba mod/h médio: **{np.mean(thrs_nb_a):,.0f}** "
            f"(min: {min(thrs_nb_a):,.0f}, max: {max(thrs_nb_a):,.0f})"
        )
    if thrs_nb_b:
        lines.append(
            f"- **Config B** — Numba mod/h médio: **{np.mean(thrs_nb_b):,.0f}** "
            f"(min: {min(thrs_nb_b):,.0f}, max: {max(thrs_nb_b):,.0f})"
        )
    if speedups_a:
        lines.append(
            f"- **Speedup médio Config A** (Fortran/Numba): "
            f"**{np.mean(speedups_a):.2f}×** "
            f"(min: {min(speedups_a):.2f}×, max: {max(speedups_a):.2f}×)"
        )
    if speedups_b:
        lines.append(
            f"- **Speedup médio Config B** (Fortran/Numba): "
            f"**{np.mean(speedups_b):.2f}×** "
            f"(min: {min(speedups_b):.2f}×, max: {max(speedups_b):.2f}×)"
        )
    if not fortran_available:
        lines.append(
            "- ⚠️  **Fortran não executado** (tatu.x ausente ou --skip-fortran). "
            "Tabelas não contêm comparação direta."
        )

    path.write_text("\n".join(lines) + "\n")
    print(f"\n  ✓ Relatório Markdown: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Benchmark local Numba JIT vs Fortran tatu.x — "
                     "Config A, Config B + experimento 30k."),
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Diretório de saída (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--n-iter", type=int, default=3,
        help="Número de iterações cronometradas por config canônica "
        "(default: 3, após warmup).",
    )
    p.add_argument(
        "--fortran-n", type=int, default=300,
        help="Subset Fortran no experimento 30k (default: 300). "
        "Use 30000 para o experimento completo (muito lento — "
        "~30-45 min devido a overhead de subprocess).",
    )
    p.add_argument(
        "--n30k", type=int, default=30_000,
        help="Número de modelos no experimento 30k (default: 30000).",
    )
    p.add_argument(
        "--skip-fortran", action="store_true",
        help="Pula toda execução Fortran (apenas Numba).",
    )
    p.add_argument(
        "--skip-plots", action="store_true",
        help="Pula geração dos plots por modelo.",
    )
    p.add_argument(
        "--skip-30k", action="store_true",
        help="Pula o experimento 30k.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed para os perfis aleatórios do experimento 30k (default: 42).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fortran_available = FORTRAN_EXEC.exists() and not args.skip_fortran

    print("═" * 72)
    print("  Benchmark local — Python Numba JIT vs Fortran tatu.x (OpenMP)")
    print("═" * 72)
    print(f"  Output:    {args.output_dir}")
    print(f"  CPUs:      {_NUM_CPUS} (NUMBA_NUM_THREADS="
          f"{os.environ['NUMBA_NUM_THREADS']}, "
          f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']})")
    print(f"  Fortran:   "
          f"{'ativo (' + str(FORTRAN_EXEC) + ')' if fortran_available else 'DESATIVADO'}")
    print(f"  Modelos:   {', '.join(CANONICAL_MODELS)}")
    print(f"  n_iter:    {args.n_iter}")
    print(f"  30k:       "
          f"{'ativo' if not args.skip_30k else 'pulado'} "
          f"(fortran_n={args.fortran_n})")
    print()

    # ── 1) Benchmarks canônicos (6 modelos × 2 configs) ─────────
    print("━" * 72)
    print("  Fase 1/3: Benchmarks canônicos (6 modelos × 2 configs)")
    print("━" * 72)
    records, results = run_canonical_benchmarks(
        n_iter=args.n_iter,
        skip_fortran=not fortran_available,
    )

    # ── 2) Plots por modelo testado ─────────────────────────────
    if not args.skip_plots:
        print()
        print("━" * 72)
        print("  Fase 2/3: Plots por modelo testado")
        print("━" * 72)
        for model_name in CANONICAL_MODELS:
            res_a = results.get((model_name, "A"))
            res_b = results.get((model_name, "B"))
            if res_a is None or res_b is None:
                print(f"  ⚠️  Resultado ausente para {model_name} — pulando plots")
                continue
            plot_model_results(model_name, res_a, res_b, args.output_dir)

    # ── 3) Experimento 30k ─────────────────────────────────────
    exp_30k: Optional[Dict[str, object]] = None
    if not args.skip_30k:
        print()
        print("━" * 72)
        print(f"  Fase 3/3: Experimento {args.n30k:,} modelos aleatórios")
        print("━" * 72)
        exp_30k = run_30k_experiment(
            n_models=args.n30k,
            fortran_n=args.fortran_n,
            skip_fortran=not fortran_available,
            seed=args.seed,
        )

    # ── 4) Saídas CSV + Markdown ──────────────────────────────
    print()
    print("━" * 72)
    print("  Salvando relatórios")
    print("━" * 72)
    csv_path = args.output_dir / "bench_numba_vs_fortran_summary.csv"
    write_csv(records, csv_path)
    print(f"  ✓ CSV: {csv_path}")
    md_path = args.output_dir / "bench_numba_vs_fortran_summary.md"
    write_markdown(
        records, exp_30k, md_path,
        fortran_available=fortran_available,
        n_iter=args.n_iter,
    )

    print()
    print("═" * 72)
    print("  Concluído. Resultados em:", args.output_dir)
    print("═" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
