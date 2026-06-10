"""Diagnóstico empírico do warmup do simulador Numba JIT CPU (v2.52).

Mede 3 eixos para o relatório de redução de warmup:
  (i)   Cache .nbc CROSS-PROCESS: 2 subprocessos frescos; run1 (.nbc frio, full
        compile) vs run2 (.nbc morno, bitcode→nativo) → tempo do 1º simulate_multi.
  (ii)  PROVA DO GAP DE COBERTURA: o warmup legado (callback JAX, 3-pos) deixa os
        kernels prange de produção FRIOS (`_simulate_combined_prange_flat.signatures==0`)
        → o 1º simulate_multi paga o compile; o warmup NOVO os pré-compila (>=1).
  (iii) Tempo de warmup + kernels aquecidos.

Uso:
    python scripts/diagnose_numba_warmup.py            # roda os 3 eixos
    python scripts/diagnose_numba_warmup.py --worker   # (interno) subprocesso (i)

O eixo (i) limpa o NUMBA_CACHE_DIR (cache .nbc) APENAS no run1 para medir o
cold-start. Usa modelo tiny p/ o full-compile do run1 ser tolerável.
"""

from __future__ import annotations

import os

# Limita threads ANTES de qualquer import de numba/BLAS — evita exaurir o TLS
# estático (_dl_allocate_tls_init) ao re-importar o stack em subprocessos
# (fork-após-threads). setdefault preserva override do usuário.
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_N_LAYERS = 3
_N_POS = 12  # >= 10 (mínimo do SimulationConfig)


def _p(msg: str) -> None:
    """print com flush (stdout block-buffered quando redirecionado a arquivo)."""
    print(msg, flush=True)


def _tiny_sim_numba() -> float:
    """Roda um simulate_multi(backend="numba") tiny multi-combo; retorna o tempo (s)."""
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.multi_forward import simulate_multi

    cfg = SimulationConfig(
        backend="numba",
        dtype="complex128",
        hankel_filter="werthmuller_201pt",
        n_positions=_N_POS,
    )
    rho = np.full(_N_LAYERS, 10.0)
    esp = np.full(max(_N_LAYERS - 2, 0), 5.0)
    pos = np.linspace(-1.0, 6.0, _N_POS)
    t0 = time.perf_counter()
    simulate_multi(
        rho,
        rho.copy(),
        esp,
        pos,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0, 2.0],
        dip_degs=[0.0],
        cfg=cfg,
    )
    return time.perf_counter() - t0


def _worker() -> int:
    """Subprocesso do eixo (i): imprime o tempo da 1ª chamada."""
    _p(f"FIRSTCALL={_tiny_sim_numba():.3f}")
    return 0


def _axis_i_cross_process() -> None:
    """(i) Cache .nbc cross-process: run1 (frio) vs run2 (morno)."""
    _p("\n## (i) Cache .nbc cross-process (Numba)")
    import geosteering_ai.cli._main  # noqa: F401 — side-effect: aplica NUMBA_CACHE_DIR

    cache_dir = os.environ.get("NUMBA_CACHE_DIR", "")
    _p(f"#   NUMBA_CACHE_DIR={cache_dir}")
    if not cache_dir:
        _p("#   SKIP (cache dir não setado)")
        return

    # Limpa o cache → run1 é COLD (full compile + persiste).
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, mode=0o700, exist_ok=True)

    # Limita threads no subprocesso — o re-import do stack (TF+JAX+Numba) num
    # processo filho pode exaurir o TLS estático (_dl_allocate_tls_init) com 64
    # threads. Reduzir mitiga o erro de fork-após-threads.
    env = dict(os.environ)
    env.update(
        {
            "NUMBA_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "JAX_PLATFORMS": "cpu",
        }
    )

    def _run() -> float:
        out = subprocess.run(
            [sys.executable, __file__, "--worker"],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        for line in out.stdout.splitlines():
            if line.startswith("FIRSTCALL="):
                return float(line.split("=", 1)[1])
        raise RuntimeError(
            f"worker sem FIRSTCALL: {out.stdout[-200:]} {out.stderr[-200:]}"
        )

    try:
        t_cold = _run()  # .nbc frio (full compile)
        t_warm = _run()  # .nbc morno (bitcode→nativo)
    except (RuntimeError, subprocess.SubprocessError) as exc:
        _p(f"#   SKIP — limite ambiental (TLS/fork-após-threads): {exc}")
        _p(
            "#   (a persistência .nbc cross-reboot é ESTRUTURAL via dir estável ~/.cache)"
        )
        return
    speedup = t_cold / t_warm if t_warm > 0 else float("nan")
    _p(f"#   run1 (.nbc FRIO)  : {t_cold:7.2f}s  (full compile + persiste)")
    _p(f"#   run2 (.nbc MORNO) : {t_warm:7.2f}s  (bitcode→nativo)")
    _p(
        f"#   speedup cross-process: {speedup:5.2f}x  ({'EFETIVO' if speedup > 1.3 else 'fraco'})"
    )


def _axis_ii_coverage_gap() -> None:
    """(ii) Prova do gap: warmup legado (JAX-callback) deixa os prange kernels frios."""
    _p("\n## (ii) Gap de cobertura: legado (JAX-callback) vs novo (full)")
    from geosteering_ai.simulation.forward import (
        _simulate_combined_prange_flat,
        _simulate_positions_njit_cached,
    )

    def flat_cached():
        return (
            len(_simulate_combined_prange_flat.signatures),
            len(_simulate_positions_njit_cached.signatures),
        )

    # Warmup LEGADO (callback JAX, 3-pos) — aquece hmd_tiv/vmd, NÃO os prange.
    try:
        from geosteering_ai.cli._main import _warmup_numba_tier2_sync

        _warmup_numba_tier2_sync(verbose=False)
        f, c = flat_cached()
        _p(
            f"#   após warmup LEGADO (JAX-callback): flat={f} cached={c}  "
            f"→ prange kernels {'FRIOS (gap!)' if f == 0 else 'aquecidos'}"
        )
    except Exception as exc:  # noqa: BLE001
        _p(f"#   warmup legado pulado ({exc.__class__.__name__}) — requer JAX")

    # Warmup NOVO (full, JAX-independente) — aquece os prange kernels.
    from geosteering_ai.simulation._numba.warmup import warmup_numba_simulator

    warmup_numba_simulator(n_layers=_N_LAYERS, n_positions=_N_POS)
    f, c = flat_cached()
    _p(
        f"#   após warmup NOVO (full): flat={f} cached={c}  "
        f"→ prange kernels {'AQUECIDOS' if f >= 1 and c >= 1 else 'incompletos'}"
    )


def _axis_iii_warmup_time() -> None:
    """(iii) Tempo de warmup + kernels aquecidos."""
    _p("\n## (iii) Warmup (tempo + kernels)")
    from geosteering_ai.simulation._numba.warmup import warmup_numba_simulator

    info = warmup_numba_simulator(n_layers=5, n_positions=200, verbose=False)
    _p(
        f"#   warmup: {len(info['functions_warmed'])} kernels em "
        f"{info['elapsed_s']:.2f}s (threads={info['threads']}, "
        f"cache={info['cache_dir']})"
    )
    _p(f"#   functions_warmed: {info['functions_warmed']}")


def main() -> int:
    if "--worker" in sys.argv:
        return _worker()
    _p("# === Diagnóstico de warmup Numba CPU (v2.52) ===")
    # Axis (i) PRIMEIRO: spawna subprocessos a partir de um parent LIMPO (antes de
    # carregar o threading-layer do Numba) — evita o erro TLS (_dl_allocate_tls_init)
    # de fork-após-threads. Axes (ii)/(iii) carregam Numba no parent → por último.
    _axis_i_cross_process()
    _axis_ii_coverage_gap()
    _axis_iii_warmup_time()
    _p("\n# === fim ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
