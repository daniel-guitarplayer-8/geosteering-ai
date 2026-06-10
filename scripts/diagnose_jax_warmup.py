"""Diagnóstico empírico do warmup do simulador JAX GPU (v2.51).

Mede 3 eixos para o relatório de redução de warmup:
  (i)   Cache persistente CROSS-PROCESS: 2 subprocessos frescos rodam um sim
        tiny; run1 (disco frio) vs run2 (disco morno) → tempo de 1ª-chamada.
  (ii)  #buckets / #shapes: single-geometria vs multi-geometria (esp aleatório).
  (iii) Tempo de warmup + redução do cold-start (real-call pós-warmup ≈ 0 novos).

Uso:
    python scripts/diagnose_jax_warmup.py            # roda os 3 eixos
    python scripts/diagnose_jax_warmup.py --worker   # (interno) subprocesso (i)

Read-only quanto à fonte do projeto. O eixo (i) limpa o cache de disco JAX
(diretório persistente) APENAS no run1 para medir o cold-start.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import numpy as np


def _tiny_sim() -> float:
    """Roda um simulate_multi_jax_batched tiny single-geom; retorna o tempo (s)."""
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation._jax.multi_forward import (
        simulate_multi_jax_batched,
    )

    cfg = SimulationConfig(
        backend="jax",
        dtype="complex128",
        jax_strategy="bucketed",
        hankel_filter="werthmuller_201pt",
        n_positions=100,
    )
    rho = np.full((1, 3), 10.0)
    esp = np.full((1, 1), 5.0)
    pos = np.linspace(-1.0, 6.0, 100)
    t0 = time.perf_counter()
    res = simulate_multi_jax_batched(
        rho,
        rho.copy(),
        esp,
        pos,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg,
        hankel_filter="werthmuller_201pt",
    )
    _ = res.H_tensor.shape
    return time.perf_counter() - t0


def _worker() -> int:
    """Subprocesso do eixo (i): imprime o tempo da 1ª chamada."""
    dt = _tiny_sim()
    print(f"FIRSTCALL={dt:.3f}")
    return 0


def _axis_i_cross_process() -> None:
    """(i) Cache persistente cross-process: run1 (frio) vs run2 (morno)."""
    print("\n## (i) Cache persistente XLA cross-process")
    import geosteering_ai.simulation._jax as jmod  # aplica _setup_xla_environment

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    print(f"#   cache_dir={cache_dir} | HAS_JAX={jmod.HAS_JAX}")
    if not jmod.HAS_JAX or not cache_dir:
        print("#   SKIP (jax ausente ou cache dir não setado)")
        return

    # Limpa o cache de disco → run1 é COLD (compila do zero + persiste).
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)  # permite GPU no subprocesso

    def _run() -> float:
        out = subprocess.run(
            [sys.executable, __file__, "--worker"],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        for line in out.stdout.splitlines():
            if line.startswith("FIRSTCALL="):
                return float(line.split("=", 1)[1])
        raise RuntimeError(
            f"worker sem FIRSTCALL: {out.stdout[-300:]} {out.stderr[-300:]}"
        )

    t_cold = _run()  # disco frio (acabou de limpar)
    t_warm = _run()  # disco morno (run1 persistiu o HLO)
    speedup = t_cold / t_warm if t_warm > 0 else float("nan")
    print(f"#   run1 (disco FRIO)  : {t_cold:7.2f}s  (compila + persiste)")
    print(f"#   run2 (disco MORNO) : {t_warm:7.2f}s  (reusa HLO de disco)")
    print(
        f"#   speedup cross-process: {speedup:5.2f}x  (cache persistente {'EFETIVO' if speedup > 1.3 else 'fraco'})"
    )


def _axis_ii_bucket_cardinality() -> None:
    """(ii) #buckets/#shapes: single-geom vs multi-geom (esp aleatório)."""
    print("\n## (ii) Cardinalidade de buckets/shapes (single vs multi-geometria)")
    from geosteering_ai.simulation._jax.forward_pure import (
        clear_jit_cache,
        clear_unified_jit_cache,
        get_jit_cache_info,
    )
    from geosteering_ai.simulation._jax.multi_forward import (
        simulate_multi_jax_batched,
    )
    from geosteering_ai.simulation.config import SimulationConfig

    cfg = SimulationConfig(
        backend="jax",
        dtype="complex128",
        jax_strategy="bucketed",
        hankel_filter="werthmuller_201pt",
        n_positions=200,
    )
    pos = np.linspace(-1.0, 11.0, 200)
    rng = np.random.default_rng(0)

    def _run_geoms(esp_list) -> int:
        import jax

        clear_jit_cache()
        clear_unified_jit_cache()
        jax.clear_caches()
        for esp_row in esp_list:
            rho = 10.0 ** rng.uniform(0, 3, (1, 5))
            esp = np.asarray(esp_row, dtype=np.float64).reshape(1, 3)
            r = simulate_multi_jax_batched(
                rho,
                rho.copy(),
                esp,
                pos,
                frequencies_hz=[20000.0],
                tr_spacings_m=[1.0],
                dip_degs=[0.0],
                cfg=cfg,
                hankel_filter="werthmuller_201pt",
            )
            _ = r.H_tensor.shape
        return int(get_jit_cache_info()["bucketed_size"])

    single = _run_geoms([[3.0, 3.0, 3.0]])  # 1 geometria fixa
    multi = _run_geoms(rng.uniform(1.0, 5.0, (8, 3)).tolist())  # 8 geom aleatórias
    print(f"#   single-geometria (1 esp)       → bucketed cache = {single}")
    print(f"#   multi-geometria  (8 esp rand)  → bucketed cache = {multi}")
    print(
        f"#   shape-explosion: {multi}/{single} = {multi / max(single, 1):.1f}× mais buckets/shapes"
    )


def _axis_iii_warmup_reduction() -> None:
    """(iii) Tempo de warmup + redução do cold-start (real-call pós-warmup)."""
    print("\n## (iii) Warmup + redução do cold-start")
    import jax

    from geosteering_ai.simulation._jax.forward_pure import (
        clear_jit_cache,
        clear_unified_jit_cache,
        get_jit_cache_info,
    )
    from geosteering_ai.simulation._jax.warmup import warmup_jax_simulator

    clear_jit_cache()
    clear_unified_jit_cache()
    jax.clear_caches()
    info = warmup_jax_simulator(n_layers=3, n_positions=100, n_models=1)
    n_warm = get_jit_cache_info()["bucketed_size"]
    dt_real = _tiny_sim()  # real-call MESMA config → deve dar cache-hit (≈0 compile)
    n_after = get_jit_cache_info()["bucketed_size"]
    print(
        f"#   warmup: {info['buckets_warmed']} buckets em {info['elapsed_s']:.2f}s "
        f"(persisted={info['persisted']})"
    )
    print(
        f"#   real-call pós-warmup: {dt_real:.3f}s | novos buckets = {n_after - n_warm} "
        f"(0 = cache-hit total)"
    )


def main() -> int:
    if "--worker" in sys.argv:
        return _worker()
    print("# === Diagnóstico de warmup JAX GPU (v2.51) ===")
    _axis_i_cross_process()
    _axis_ii_bucket_cardinality()
    _axis_iii_warmup_reduction()
    print("\n# === fim ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
