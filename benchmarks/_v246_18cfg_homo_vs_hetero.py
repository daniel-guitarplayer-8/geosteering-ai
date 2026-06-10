"""Medição (cenário 30k) — throughput EXATO dos 18 configs: HOMOGÊNEO vs HETEROGÊNEO.

18 configs = 3 freq × 3 dip × 2 TR, 600 pos, c128. Mede:
  (1) JAX batched-bucketed, geometria HOMOGÊNEA (esp idêntico, varia rho) — caminho rápido.
  (2) JAX batched, geometria HETEROGÊNEA (esp perturbado por modelo) — cai no fallback
      'unified' (~6.9× mais lento) — caso de dataset com geologia variável.
  (3) Numba 16w×4t (referência, independe de geometria).
Reporta mod/h por n_models ∈ {32,64,128} e o wall-time extrapolado p/ 30k modelos.
NÃO faz parte da suíte.
"""

import sys
import time
import numpy as np

sys.path.insert(0, "benchmarks")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from bench_numba_vs_jax_gpu import _get_default_model, _bench_numba  # noqa: E402
from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax_batched  # noqa: E402
from geosteering_ai.simulation import SimulationConfig  # noqa: E402

N_POS = 600
N_ITERS = 5
DATASET = 30_000
freqs = [20000.0, 50000.0, 100000.0]
dips = [0.0, 30.0, 60.0]
trs = [0.5, 1.0]
N_MODELS = [32, 64, 128]
CHUNK = 32

m = _get_default_model()
n = m.n_layers
positions_z = np.linspace(m.min_depth - 1.0, m.max_depth + 1.0, N_POS)
nf, nTR, nAng = len(freqs), len(trs), len(dips)
print(f"# modelo={m.name} n={n} | n_pos={N_POS} | {nf}f×{nAng}dip×{nTR}TR="
      f"{nf*nTR*nAng} configs | c128 | iters={N_ITERS} | dataset={DATASET}", flush=True)

rng = np.random.default_rng(0)


def _rho_batch(nm):
    rh = rng.uniform(1.0, 100.0, (nm, n))
    return rh, rh.copy()


def _esp_homo(nm):
    return np.tile(np.asarray(m.esp), (nm, 1))


def _esp_hetero(nm):
    # Perturba espessuras por modelo (fator [0.8,1.2]) → camad distintos →
    # geometria heterogênea → fallback 'unified'. Mantém positivo/físico.
    base = np.asarray(m.esp)
    fac = rng.uniform(0.8, 1.2, (nm, base.shape[0]))
    return base[None, :] * fac


def time_jax(esp_batch, rho_h, rho_v, chunk):
    cfg = SimulationConfig(
        backend="jax", jax_strategy="bucketed", dtype="complex128",
        jax_chunk_size_models=chunk,
    )
    args = (rho_h, rho_v, esp_batch, positions_z)
    kw = dict(frequencies_hz=freqs, tr_spacings_m=trs, dip_degs=dips, cfg=cfg)
    simulate_multi_jax_batched(*args, **kw)  # warmup
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        simulate_multi_jax_batched(*args, **kw)
    return (time.perf_counter() - t0) / N_ITERS


def wall_30k(mh):
    """Wall-time (min) p/ DATASET modelos dado throughput mod/h."""
    return DATASET / mh / 60.0 * 60.0  # h→min: DATASET/mh horas ×60


# ── Numba 16w×4t (referência) ────────────────────────────────────────────────
t_n = _bench_numba(
    rho_h=m.rho_h, rho_v=m.rho_v, esp=m.esp, positions_z=positions_z,
    frequencies_hz=freqs, tr_spacings_m=trs, dip_degs=dips,
    n_iters=N_ITERS, n_workers=16, threads_per_worker=4,
)
mh_numba = N_ITERS / t_n * 3600.0
print(f"# NUMBA 16w×4t = {mh_numba:.3e} mod/h | 30k ≈ {DATASET/mh_numba*60:.1f} min "
      f"(independe de geometria)\n", flush=True)

print("n_models,homo_modh,homo_30k_min,homo/numba,hetero_modh,hetero_30k_min,"
      "hetero/numba,homo/hetero", flush=True)
for nm in N_MODELS:
    rh, rv = _rho_batch(nm)
    # Homogêneo (bucketed)
    try:
        dt_h = time_jax(_esp_homo(nm), rh, rv, CHUNK)
        mh_h = nm / dt_h * 3600.0
    except Exception as e:
        mh_h = float("nan")
        print(f"  # homo n={nm} FALHA: {str(e)[:70]}", flush=True)
    # Heterogêneo (→ unified fallback)
    try:
        dt_x = time_jax(_esp_hetero(nm), rh, rv, CHUNK)
        mh_x = nm / dt_x * 3600.0
    except Exception as e:
        mh_x = float("nan")
        print(f"  # hetero n={nm} FALHA: {str(e)[:70]}", flush=True)
    print(f"{nm},{mh_h:.3e},{DATASET/mh_h*60:.1f},{mh_h/mh_numba:.2f},"
          f"{mh_x:.3e},{DATASET/mh_x*60:.1f},{mh_x/mh_numba:.2f},{mh_h/mh_x:.1f}",
          flush=True)
