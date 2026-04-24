# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_benchmark.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Suite de Benchmark                    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: numpy, geosteering_ai.simulation                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Orquestra a comparação de throughput entre o simulador Python Numba   ║
# ║    JIT otimizado e o simulador Fortran tatu.x, reportando métricas por   ║
# ║    modelo canônico (Oklahoma 3/5/28, Devine 8, Hou 7, Viking Graben 10). ║
# ║                                                                           ║
# ║  EXPERIMENTOS                                                             ║
# ║    ┌─────────┬────────────────────────────────────────────────────────┐  ║
# ║    │ Config A │ 1 freq (20 kHz) · 1 TR (1 m) · 1 dip (0°) · 600 pos   │  ║
# ║    │ Config B │ 2 freqs (20/40) · 1 TR (1 m) · 2 dips (0/30°) · 600 p │  ║
# ║    │ 30k Gen  │ 30k perfis 20 camadas, ρh∈[500,2000], log-uniforme    │  ║
# ║    └─────────┴────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  MÉTRICAS POR CÉLULA                                                      ║
# ║    • throughput_mod_h          — modelos por hora                        ║
# ║    • total_time_s              — tempo total wall-clock                  ║
# ║    • time_per_model_ms         — ms por modelo                           ║
# ║    • speedup_fortran_over_numba — razão Fortran / Numba (>1: Numba rápido) ║
# ║    • max_abs_err                — paridade |H_numba − H_fortran|_∞       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Suite de benchmark para Simulation Manager.

Fornece ``BenchmarkSuite`` (chamável diretamente em CLI ou via QThread) e
``BenchRecord`` (dataclass de saída por célula), além de ``BenchmarkThread``
para execução não-bloqueante na GUI.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sm_model_gen import GenConfig, generate_models
from .sm_qt_compat import QObject, QThread, Signal
from .sm_workers import SimRequest, run_fortran_chunk, run_numba_chunk

# ──────────────────────────────────────────────────────────────────────────
# Estruturas de dados
# ──────────────────────────────────────────────────────────────────────────


CANONICAL_BENCHMARK_MODELS = [
    "oklahoma_3",
    "oklahoma_5",
    "oklahoma_28",
    "devine_8",
    "hou_7",
    "viking_graben_10",
]


# ──────────────────────────────────────────────────────────────────────────
# Config C — parâmetros UNIFORMES (uma única configuração para todos os
# modelos canônicos selecionados), extraídos dos scripts de validação
# (buildValidamodels.py — frequências/TRs comentadas como alternativa).
# ──────────────────────────────────────────────────────────────────────────
CONFIG_C_PARAMS: Dict[str, Any] = {
    "freqs_hz": [2000.0, 6000.0],
    "tr_list_m": [8.19, 20.43],
    "dip_list_deg": [0.0],
    "h1": 10.0,
    "p_med": 0.200,
    "description": (
        "Config C fixa — f=[2 kHz, 6 kHz], TR=[8.19, 20.43] m, dip=0°, "
        "h1=10.000 m, p_med=0.200 m. Aplicada uniformemente a todos os "
        "modelos canônicos selecionados em 'Modelos canônicos (A/B/C)'."
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# Config D — parâmetros canônicos per-model customizáveis
# ──────────────────────────────────────────────────────────────────────────
# Extraídos de:
#   • benchmarks/bench_numba_vs_fortran_local.py — geometria comum fixa
#   • Fortran_Gerador/buildValidamodels.py — geometria derivada por modelo
#   • geosteering_ai/simulation/validation/canonical_models.py — fonte de verdade
#
# "freqs_hz" inclui ambas convenções: [20 kHz] padrão bench, alternativa
# comentada [2 kHz, 6 kHz] do buildValidamodels. Aqui escolhemos o padrão
# principal usado nas validações atuais; o usuário pode customizar via UI.
CANONICAL_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "oklahoma_3": {
        "freqs_hz": [20000.0],
        "tr_list_m": [1.0],
        "dip_list_deg": [0.0],
        "h1": 10.0,
        "p_med": 0.2,
        "description": "3 camadas TIV simples (Tech. Report 32/2011).",
    },
    "oklahoma_5": {
        "freqs_hz": [20000.0],
        "tr_list_m": [1.0],
        "dip_list_deg": [0.0],
        "h1": 10.0,
        "p_med": 0.2,
        "description": "5 camadas TIV gradual (Tech. Report 32/2011).",
    },
    "oklahoma_28": {
        "freqs_hz": [2000.0, 6000.0],
        "tr_list_m": [8.19, 20.43],
        "dip_list_deg": [0.0],
        "h1": 10.0,
        "p_med": 0.2,
        "description": "28 camadas TIV forte, anisotropia ρᵥ=2·ρₕ (Tech. Report 32/2011 p.58).",
    },
    "devine_8": {
        "freqs_hz": [20000.0],
        "tr_list_m": [1.0],
        "dip_list_deg": [0.0],
        "h1": 10.0,
        "p_med": 0.2,
        "description": "8 camadas isotrópico (Tech. Report 32/2011).",
    },
    "hou_7": {
        "freqs_hz": [20000.0],
        "tr_list_m": [1.0],
        "dip_list_deg": [0.0],
        "h1": 10.0,
        "p_med": 0.2,
        "description": "7 camadas TIV (Hou, Mallan & Verdin 2006, Geophysics 71(5) F101).",
    },
    "viking_graben_10": {
        "freqs_hz": [20000.0],
        "tr_list_m": [1.0],
        "dip_list_deg": [0.0],
        "h1": 29.5418,
        "p_med": 0.2,
        "description": "10 camadas reservatório Mar do Norte (Eidesmo et al. 2002, adapt.).",
    },
}


def default_canonical_config(model_name: str) -> Dict[str, Any]:
    """Retorna configuração canônica padrão (cópia profunda) para o modelo.

    Args:
        model_name: Nome canônico do modelo.

    Returns:
        Dict com chaves ``freqs_hz``, ``tr_list_m``, ``dip_list_deg``,
        ``h1``, ``p_med``, ``description`` — pronto para ser editado pelo
        usuário sem mutar o dicionário global.

    Note:
        Se ``model_name`` não estiver no catálogo, retorna um preset
        genérico (20 kHz, TR=1 m, dip=0°) derivado do benchmark Config A.
    """
    default = CANONICAL_MODEL_CONFIGS.get(
        model_name,
        {
            "freqs_hz": [20000.0],
            "tr_list_m": [1.0],
            "dip_list_deg": [0.0],
            "h1": 10.0,
            "p_med": 0.2,
            "description": "(parâmetros padrão — modelo não catalogado)",
        },
    )
    # Retorno é cópia para preservar imutabilidade do catálogo.
    return {
        "freqs_hz": list(default["freqs_hz"]),
        "tr_list_m": list(default["tr_list_m"]),
        "dip_list_deg": list(default["dip_list_deg"]),
        "h1": float(default["h1"]),
        "p_med": float(default["p_med"]),
        "description": str(default.get("description", "")),
    }


def compute_model_tj(model_name: str) -> float:
    """Calcula ``tj`` adequado ao modelo (port de buildValidamodels.py).

    Retorna ``sum(esp) + 20.0`` — janela de investigação que acomoda
    todas as camadas internas + 20 m de folga (10 m acima da primeira
    interface e 10 m abaixo da última).
    """
    try:
        from geosteering_ai.simulation.validation import get_canonical_model

        m = get_canonical_model(model_name)
        return float(np.sum(m.esp) + 20.0)
    except Exception:
        return 120.0  # fallback equivalente ao bench Config A/B


@dataclass
class BenchRecord:
    """Registro de uma célula (modelo × configuração) de benchmark.

    Attributes:
        model_name: Nome do modelo canônico.
        config: ``"A"`` ou ``"B"`` ou ``"30k"``.
        n_models: Número de modelos simulados.
        n_pos: Pontos de medição por modelo.
        nf, nTR, n_dip: Dimensões da simulação.
        numba_total_s, fortran_total_s: Tempos totais em segundos.
        numba_ms_per_model, fortran_ms_per_model: ms/modelo.
        numba_mod_h, fortran_mod_h: Throughput (modelos/hora).
        speedup: ``fortran_total_s / numba_total_s`` (>1 ⇒ Numba mais rápido).
        max_abs_err: Paridade numérica entre os dois backends (0 se desabilitado).
    """

    model_name: str
    config: str
    n_models: int
    n_pos: int
    nf: int
    nTR: int
    n_dip: int
    numba_total_s: float = 0.0
    fortran_total_s: float = 0.0
    numba_ms_per_model: float = 0.0
    fortran_ms_per_model: float = 0.0
    numba_mod_h: float = 0.0
    fortran_mod_h: float = 0.0
    speedup: float = 0.0
    max_abs_err: float = 0.0


@dataclass
class BenchmarkSettings:
    """Configuração do benchmark suite.

    Attributes:
        run_config_a, run_config_b, run_30k: Seletores dos experimentos
            fixos (A/B/30k). Config A/B aplicam a mesma geometria (freq/TR/dip)
            a todos os modelos canônicos selecionados em ``models``.
        run_config_c: Habilita Config C — parâmetros UNIFORMES (uma única
            configuração aplicada a todos os modelos canônicos selecionados
            em ``models``). Usa ``CONFIG_C_PARAMS`` (freq=[2k,6k] Hz,
            TR=[8.19, 20.43] m, dip=0°, h1=10 m, p_med=0.2 m).
        run_config_d: Habilita Config D — parâmetros canônicos per-model
            customizáveis. Cada modelo marcado em ``config_d_models`` é
            rodado com seu próprio override (``config_d_overrides[model]``)
            ou com o padrão canônico (``default_canonical_config(model)``)
            se não houver override.
        config_d_models: Lista de nomes canônicos habilitados em Config D.
        config_d_overrides: Mapeamento ``model_name → override_dict``
            contendo chaves ``freqs_hz``, ``tr_list_m``, ``dip_list_deg``,
            ``h1``, ``p_med`` (mesmo schema de ``default_canonical_config``).
            Chaves ausentes caem no default canônico.
        models: Lista de nomes canônicos para Config A/B/C. Default = todos 6.
        n_iter: Iterações de timing (média) para Config A/B/C/D.
        fortran_subset: Quantidade de modelos rodados pelo Fortran no
            experimento 30k (extrapolação linear para o total).
        n_workers_numba, n_threads_numba: Paralelismo Numba.
        n_workers_fortran, n_threads_fortran: Paralelismo Fortran.
        fortran_binary: Caminho para tatu.x.
        run_numba, run_fortran: Habilita cada backend.
        hankel_filter: Filtro Hankel (aplicado a todas as configs).
        h1, tj, p_med: Geometria de medição (A/B/30k). Em Config C,
            cada modelo tem seu próprio ``h1``/``p_med`` e ``tj`` derivado.
        output_dir: Diretório para CSV/MD.
        n30k_layers, n30k_rho_h_min, n30k_rho_h_max: Geração 30k.
    """

    run_config_a: bool = True
    run_config_b: bool = True
    run_config_c: bool = False
    run_config_d: bool = False
    run_30k: bool = False
    models: List[str] = field(default_factory=lambda: list(CANONICAL_BENCHMARK_MODELS))
    config_d_models: List[str] = field(default_factory=list)
    config_d_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_iter: int = 3
    fortran_subset: int = 300
    n30k_total: int = 30000
    n_workers_numba: int = 4
    n_threads_numba: int = 4
    n_workers_fortran: int = 8
    n_threads_fortran: int = 2
    fortran_binary: str = ""
    run_numba: bool = True
    run_fortran: bool = True
    hankel_filter: str = "werthmuller_201pt"
    h1: float = 10.0
    tj: float = 120.0
    p_med: float = 0.2
    output_dir: str = ""
    n30k_layers: int = 20
    n30k_rho_h_min: float = 500.0
    n30k_rho_h_max: float = 2000.0
    seed: int = 42

    def canonical_params_for(self, model_name: str) -> Dict[str, Any]:
        """Resolve parâmetros efetivos de Config D para um modelo.

        Ordem de precedência: override do usuário → padrão canônico.
        """
        base = default_canonical_config(model_name)
        override = self.config_d_overrides.get(model_name, {})
        for k in ("freqs_hz", "tr_list_m", "dip_list_deg"):
            if k in override and override[k]:
                base[k] = list(override[k])
        for k in ("h1", "p_med"):
            if k in override:
                base[k] = float(override[k])
        return base


# ──────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ──────────────────────────────────────────────────────────────────────────


def _positions_z_for_config(tj: float, p_med: float, dip_deg: float = 0.0) -> np.ndarray:
    """Grade de posições (ponto-médio T-R) com dip arbitrário.

    Mantém 600 pontos quando ``tj=120`` e ``p_med=0.2``, alinhando com os
    benchmarks anteriores do batch_runner.
    """
    import math

    cos_d = max(1e-6, math.cos(math.radians(abs(dip_deg))))
    n_pos = max(1, int(math.ceil(tj / (p_med * cos_d))))
    return np.linspace(0.0, tj, n_pos, dtype=np.float64)


def _get_canonical_dict(model_name: str) -> dict:
    """Converte um CanonicalModel em dict compatível com os workers."""
    from geosteering_ai.simulation.validation import get_canonical_model

    m = get_canonical_model(model_name)
    return {
        "n_layers": int(m.n_layers),
        "rho_h": m.rho_h.tolist(),
        "rho_v": m.rho_v.tolist(),
        "thicknesses": m.esp.tolist(),
        "title": m.title,
    }


# ──────────────────────────────────────────────────────────────────────────
# Execução de uma célula (modelo × config)
# ──────────────────────────────────────────────────────────────────────────


def _single_cell(
    model_dict: dict,
    model_name: str,
    config_label: str,
    freqs: List[float],
    trs: List[float],
    dips: List[float],
    positions_z: np.ndarray,
    settings: BenchmarkSettings,
    log_fn: Callable[[str], None],
) -> Tuple[BenchRecord, Optional[np.ndarray], Optional[np.ndarray]]:
    """Executa uma célula e retorna (registro, H_numba, H_fortran).

    ``H_*`` retornados são apenas do primeiro modelo da repetição (para plots).
    """
    n_iter = max(1, int(settings.n_iter))
    chunk = [model_dict] * n_iter

    rec = BenchRecord(
        model_name=model_name,
        config=config_label,
        n_models=n_iter,
        n_pos=int(positions_z.shape[0]),
        nf=len(freqs),
        nTR=len(trs),
        n_dip=len(dips),
    )
    H_numba: Optional[np.ndarray] = None
    H_fortran: Optional[np.ndarray] = None

    # ── Numba ────────────────────────────────────────────────────────────
    if settings.run_numba:
        log_fn(f"  Numba  [{model_name}/{config_label}] n_iter={n_iter}...")
        _, H_stack, _z_obs, elapsed = run_numba_chunk(
            worker_id=0,
            chunk=chunk,
            positions_z=positions_z,
            freqs=freqs,
            trs=trs,
            dips=dips,
            hankel_filter=settings.hankel_filter,
            n_threads=settings.n_threads_numba * settings.n_workers_numba,
        )
        rec.numba_total_s = float(elapsed)
        rec.numba_ms_per_model = 1000.0 * elapsed / n_iter
        rec.numba_mod_h = n_iter / max(elapsed, 1e-6) * 3600.0
        if H_stack.size:
            H_numba = H_stack[0]

    # ── Fortran ──────────────────────────────────────────────────────────
    if settings.run_fortran and settings.fortran_binary:
        log_fn(f"  Fortran[{model_name}/{config_label}] n_iter={n_iter}...")
        orig_dir = os.getcwd()
        with _chdir_tmp(settings.output_dir) as workdir:
            try:
                _, _files, elapsed_f = run_fortran_chunk(
                    worker_id=0,
                    chunk=chunk,
                    freqs=freqs,
                    trs=trs,
                    dips=dips,
                    h1=settings.h1,
                    tj=settings.tj,
                    p_med=settings.p_med,
                    fortran_binary=settings.fortran_binary,
                    filter_type=_filter_type_index(settings.hankel_filter),
                    n_threads=settings.n_threads_fortran * settings.n_workers_fortran,
                    model_start=1,
                    model_end=n_iter,
                )
                rec.fortran_total_s = float(elapsed_f)
                rec.fortran_ms_per_model = 1000.0 * elapsed_f / n_iter
                rec.fortran_mod_h = n_iter / max(elapsed_f, 1e-6) * 3600.0
            finally:
                os.chdir(orig_dir)

    # Speedup
    if rec.numba_total_s > 0 and rec.fortran_total_s > 0:
        rec.speedup = rec.fortran_total_s / rec.numba_total_s

    return rec, H_numba, H_fortran


def _filter_type_index(name: str) -> int:
    return {"werthmuller_201pt": 0, "kong_61pt": 1, "anderson_801pt": 2}.get(name, 0)


class _chdir_tmp:
    """Context manager para mudar para ``output_dir`` e garantir retorno.

    Usado para que os artefatos ``.dat/.out`` do Fortran caiam na pasta
    correta sem depender do cwd do processo Python host.
    """

    def __init__(self, output_dir: str) -> None:
        self._dir = output_dir or os.getcwd()
        self._prev = os.getcwd()

    def __enter__(self) -> str:
        os.makedirs(self._dir, exist_ok=True)
        os.chdir(self._dir)
        return self._dir

    def __exit__(self, *exc: Any) -> None:
        os.chdir(self._prev)


# ──────────────────────────────────────────────────────────────────────────
# Orquestrador principal
# ──────────────────────────────────────────────────────────────────────────


class BenchmarkSuite:
    """Executor síncrono da suite de benchmark (chamável fora da GUI)."""

    def __init__(self, settings: BenchmarkSettings) -> None:
        self.settings = settings
        self.records: List[BenchRecord] = []
        self.plot_data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def run(
        self,
        log_fn: Callable[[str], None] = print,
        progress_fn: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[BenchRecord]:
        """Executa todos os experimentos conforme settings."""
        s = self.settings
        cells: List[Tuple[str, str, List[float], List[float], List[float]]] = []

        # Cells comuns (A/B): geometria fixa; tj/p_med globais de settings.
        if s.run_config_a:
            for m in s.models:
                cells.append((m, "A", [20000.0], [1.0], [0.0], s.tj, s.p_med))
        if s.run_config_b:
            for m in s.models:
                cells.append(
                    (m, "B", [20000.0, 40000.0], [1.0], [0.0, 30.0], s.tj, s.p_med)
                )
        # Cells Config C: parâmetros UNIFORMES (mesmos para todos os modelos),
        # tj per-model via compute_model_tj() para respeitar espessuras.
        if s.run_config_c:
            c = CONFIG_C_PARAMS
            for m in s.models:
                tj_m = compute_model_tj(m)
                cells.append(
                    (
                        m,
                        "C",
                        list(c["freqs_hz"]),
                        list(c["tr_list_m"]),
                        list(c["dip_list_deg"]),
                        tj_m,
                        float(c["p_med"]),
                    )
                )
        # Cells Config D: parâmetros per-model (catálogo + overrides do usuário).
        if s.run_config_d:
            for m in s.config_d_models:
                params = s.canonical_params_for(m)
                tj_m = compute_model_tj(m)
                cells.append(
                    (
                        m,
                        "D",
                        list(params["freqs_hz"]),
                        list(params["tr_list_m"]),
                        list(params["dip_list_deg"]),
                        tj_m,
                        float(params["p_med"]),
                    )
                )

        total = len(cells) + (1 if s.run_30k else 0)
        idx = 0
        for model_name, config_label, freqs, trs, dips, tj_c, p_med_c in cells:
            idx += 1
            if progress_fn:
                progress_fn(idx, total, f"{model_name}/{config_label}")
            log_fn(f"[{idx}/{total}] {model_name} · Config {config_label}")
            model_dict = _get_canonical_dict(model_name)
            positions_z = _positions_z_for_config(tj_c, p_med_c, dip_deg=0.0)
            rec, Hn, Hf = _single_cell(
                model_dict,
                model_name,
                config_label,
                freqs,
                trs,
                dips,
                positions_z,
                s,
                log_fn,
            )
            self.records.append(rec)
            self.plot_data[(model_name, config_label)] = {
                "model": model_dict,
                "positions_z": positions_z,
                "H_numba": Hn,
                "H_fortran": Hf,
                "freqs": freqs,
                "trs": trs,
                "dips": dips,
            }

        if s.run_30k:
            idx += 1
            if progress_fn:
                progress_fn(idx, total, "30k experiment")
            self._run_30k_experiment(log_fn)

        return self.records

    def _run_30k_experiment(self, log_fn: Callable[[str], None]) -> None:
        """Experimento: 30k modelos aleatórios × Config A (subset Fortran)."""
        s = self.settings
        log_fn(f"[30k] Gerando {s.n30k_total} modelos ({s.n30k_layers} camadas)...")
        cfg_gen = GenConfig(
            total_depth=s.tj,
            n_layers_fixed=s.n30k_layers,
            rho_h_min=s.n30k_rho_h_min,
            rho_h_max=s.n30k_rho_h_max,
            rho_h_distribution="loguni",
            anisotropic=True,
            lambda_min=1.0,
            lambda_max=1.7,
            min_thickness=0.5,
            generator="sobol",
        )
        models = generate_models(cfg_gen, n_models=s.n30k_total, rng_seed=s.seed)
        positions_z = _positions_z_for_config(s.tj, s.p_med, dip_deg=0.0)

        rec = BenchRecord(
            model_name="random_20layer",
            config="30k",
            n_models=s.n30k_total,
            n_pos=int(positions_z.shape[0]),
            nf=1,
            nTR=1,
            n_dip=1,
        )

        # Numba full 30k
        if s.run_numba:
            log_fn(f"  Numba full 30k (workers={s.n_workers_numba})...")
            from .sm_workers import SimRequest, SimulationThread  # evita ciclo

            t0 = time.perf_counter()
            total_elapsed = 0.0
            # Executa em paralelo usando n_workers_numba sem QThread (modo síncrono)
            elapsed = _run_numba_parallel_sync(
                models=models,
                positions_z=positions_z,
                freqs=[20000.0],
                trs=[1.0],
                dips=[0.0],
                hankel_filter=s.hankel_filter,
                n_workers=s.n_workers_numba,
                n_threads=s.n_threads_numba,
            )
            rec.numba_total_s = elapsed
            rec.numba_ms_per_model = 1000.0 * elapsed / s.n30k_total
            rec.numba_mod_h = s.n30k_total / max(elapsed, 1e-6) * 3600.0
            log_fn(f"    Numba 30k: {elapsed:.1f}s ({rec.numba_mod_h:.0f} mod/h)")

        # Fortran subset (extrapolação linear)
        if s.run_fortran and s.fortran_binary:
            subset = max(1, min(int(s.fortran_subset), s.n30k_total))
            log_fn(f"  Fortran subset {subset} (extrapolado para 30k)...")
            with _chdir_tmp(s.output_dir):
                _, _files, elapsed_subset = run_fortran_chunk(
                    worker_id=0,
                    chunk=models[:subset],
                    freqs=[20000.0],
                    trs=[1.0],
                    dips=[0.0],
                    h1=s.h1,
                    tj=s.tj,
                    p_med=s.p_med,
                    fortran_binary=s.fortran_binary,
                    filter_type=_filter_type_index(s.hankel_filter),
                    n_threads=s.n_threads_fortran * s.n_workers_fortran,
                    model_start=1,
                    model_end=subset,
                )
            ms_per_model = 1000.0 * elapsed_subset / subset
            elapsed_extrap = elapsed_subset / subset * s.n30k_total
            rec.fortran_total_s = elapsed_extrap
            rec.fortran_ms_per_model = ms_per_model
            rec.fortran_mod_h = s.n30k_total / max(elapsed_extrap, 1e-6) * 3600.0
            log_fn(
                f"    Fortran subset {subset} em {elapsed_subset:.1f}s → "
                f"extrapolado 30k: {elapsed_extrap:.1f}s ({rec.fortran_mod_h:.0f} mod/h)"
            )

        if rec.numba_total_s > 0 and rec.fortran_total_s > 0:
            rec.speedup = rec.fortran_total_s / rec.numba_total_s
        self.records.append(rec)

    # ── Exportação ───────────────────────────────────────────────────────

    def export_csv(self, path: str) -> None:
        """Exporta registros em CSV."""
        if not self.records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = list(asdict(self.records[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))

    def export_markdown(self, path: str) -> None:
        """Exporta relatório compacto em Markdown (5 tabelas: A/B/C/D/30k)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        recs_a = [r for r in self.records if r.config == "A"]
        recs_b = [r for r in self.records if r.config == "B"]
        recs_c = [r for r in self.records if r.config == "C"]
        recs_d = [r for r in self.records if r.config == "D"]
        recs_30k = [r for r in self.records if r.config == "30k"]

        def _table(records: List[BenchRecord], title: str) -> str:
            if not records:
                return ""
            lines = [f"## {title}\n"]
            lines.append(
                "| Modelo | Numba ms/mod | Fortran ms/mod | Numba mod/h "
                "| Fortran mod/h | Speedup |\n"
            )
            lines.append(
                "|:-------|------------:|--------------:|-----------:|--------------:|:-------:|\n"
            )
            for r in records:
                lines.append(
                    f"| {r.model_name} | {r.numba_ms_per_model:.2f} | "
                    f"{r.fortran_ms_per_model:.2f} | {r.numba_mod_h:,.0f} | "
                    f"{r.fortran_mod_h:,.0f} | {r.speedup:.2f}× |\n"
                )
            lines.append("\n")
            return "".join(lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Simulation Manager — Numba JIT vs Fortran tatu.x\n\n")
            f.write(_table(recs_a, "Config A — 1 freq · 1 TR · 1 dip · 600 pos"))
            f.write(_table(recs_b, "Config B — 2 freq · 1 TR · 2 dips · 600 pos"))
            f.write(
                _table(
                    recs_c,
                    "Config C — uniforme · f=[2k,6k]Hz · TR=[8.19,20.43]m · dip=0° · h1=10m · p_med=0.200m",
                )
            )
            f.write(
                _table(
                    recs_d,
                    "Config D — parâmetros canônicos per-model customizáveis (janela dedicada)",
                )
            )
            f.write(_table(recs_30k, "Experimento 30k — 20 camadas alta-ρ"))


def _run_numba_parallel_sync(
    models: List[dict],
    positions_z: np.ndarray,
    freqs: List[float],
    trs: List[float],
    dips: List[float],
    hankel_filter: str,
    n_workers: int,
    n_threads: int,
) -> float:
    """Executa Numba em paralelo síncrono (sem QThread) e retorna tempo total."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = max(1, int(n_workers))
    n_total = len(models)
    chunk_size = n_total // n_workers
    rem = n_total % n_workers
    batches: List[Tuple[int, List[dict]]] = []
    start = 0
    for w in range(n_workers):
        extra = 1 if w < rem else 0
        end = start + chunk_size + extra
        batches.append((w, models[start:end]))
        start = end

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                run_numba_chunk,
                wid,
                chunk,
                positions_z,
                freqs,
                trs,
                dips,
                hankel_filter,
                n_threads,
            )
            for wid, chunk in batches
        ]
        for _ in as_completed(futures):
            pass
    return time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────
# QThread wrapper
# ──────────────────────────────────────────────────────────────────────────


class BenchmarkThread(QThread):
    """Executa ``BenchmarkSuite`` em thread Qt e emite sinais ricos."""

    progress = Signal(int, int, str)
    log = Signal(str)
    finished_all = Signal(list, dict)  # (records, plot_data)
    error = Signal(str)

    def __init__(
        self, settings: BenchmarkSettings, parent: Optional[QObject] = None
    ) -> None:
        super().__init__(parent)
        self._settings = settings

    def run(self) -> None:
        try:
            suite = BenchmarkSuite(self._settings)
            suite.run(
                log_fn=lambda s: self.log.emit(s),
                progress_fn=lambda i, n, label: self.progress.emit(i, n, label),
            )
            # Export automático (se output_dir definido)
            if self._settings.output_dir:
                os.makedirs(self._settings.output_dir, exist_ok=True)
                csv_path = os.path.join(
                    self._settings.output_dir, "benchmark_summary.csv"
                )
                md_path = os.path.join(self._settings.output_dir, "benchmark_summary.md")
                suite.export_csv(csv_path)
                suite.export_markdown(md_path)
                self.log.emit(f"Relatório salvo em {md_path}")
                self.log.emit(f"CSV salvo em {csv_path}")
            self.finished_all.emit(suite.records, suite.plot_data)
        except Exception as exc:  # pragma: no cover
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()[:2000]}")


__all__ = [
    "BenchmarkSettings",
    "BenchmarkSuite",
    "BenchmarkThread",
    "BenchRecord",
    "CANONICAL_BENCHMARK_MODELS",
    "CANONICAL_MODEL_CONFIGS",
    "CONFIG_C_PARAMS",
    "compute_model_tj",
    "default_canonical_config",
]
