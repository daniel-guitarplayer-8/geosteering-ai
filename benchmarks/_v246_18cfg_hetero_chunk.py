"""Follow-up — throughput HETEROGÊNEO (unified fallback) dos 18 configs c/ chunk pequeno.

O caso heterogêneo (esp varia por modelo) cai no kernel 'unified', que a 18 cfg/600 pos
estoura VRAM com chunk=32. Aqui reduz o chunk do eixo de modelos p/ caber em 48 GB e
medir o throughput real. Compara com homogêneo (mesmo n_models) e Numba 16w×4t.
"""

import sys
import time
import numpy as np

sys.path.insert(0, "benchmarks")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from bench_numba_vs_jax_gpu import _get_default_model  # noqa: E402
from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax_batched  # noqa: E402
from geosteering_ai.simulation import SimulationConfig  # noqa: E402

N_POS = 600
N_ITERS = 3
DATASET = 30_000
freqs = [20000.0, 50000.0, 100000.0]
dips = [0.0, 30.0, 60.0]
trs = [0.5, 1.0]

m = _get_default_model()
n = m.n_layers
positions_z = np.linspace(m.min_depth - 1.0, m.max_depth + 1.0, N_POS)
print(f"# {len(freqs)}f×{len(dips)}dip×{len(trs)}TR={len(freqs)*len(dips)*len(trs)} "
      f"configs | n_pos={N_POS} | c128 | HETEROGÊNEO (unified) chunk pequeno", flush=True)

rng = np.random.default_rng(0)


def time_jax(esp_batch, rho_h, rho_v, chunk):
    cfg = SimulationConfig(backend="jax", jax_strategy="bucketed", dtype="complex128",
                           jax_chunk_size_models=chunk)
    a = (rho_h, rho_v, esp_batch, positions_z)
    kw = dict(frequencies_hz=freqs, tr_spacings_m=trs, dip_degs=dips, cfg=cfg)
    simulate_multi_jax_batched(*a, **kw)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        simulate_multi_jax_batched(*a, **kw)
    return (time.perf_counter() - t0) / N_ITERS


print("n_models,chunk,hetero_modh,hetero_30k_min,homo_modh,homo/hetero", flush=True)
for nm, chunk in [(16, 8), (32, 8), (32, 4), (64, 8)]:
    rh = rng.uniform(1.0, 100.0, (nm, n)); rv = rh.copy()
    esp_homo = np.tile(np.asarray(m.esp), (nm, 1))
    esp_het = np.asarray(m.esp)[None, :] * rng.uniform(0.8, 1.2, (nm, np.asarray(m.esp).shape[0]))
    try:
        dt_x = time_jax(esp_het, rh, rv, chunk)
        mh_x = nm / dt_x * 3600.0
    except Exception as e:
        print(f"{nm},{chunk},FALHA,{str(e)[:55]}", flush=True); continue
    try:
        dt_h = time_jax(esp_homo, rh, rv, chunk)
        mh_h = nm / dt_h * 3600.0
    except Exception:
        mh_h = float("nan")
    print(f"{nm},{chunk},{mh_x:.3e},{DATASET/mh_x*60:.1f},{mh_h:.3e},{mh_h/mh_x:.1f}",
          flush=True)
