"""Sweep LIMPO (investigação high-config) — crossover n_models do batched-bucketed atual.

Resolve a questão decisiva: a QUE n_models o `simulate_multi_jax_batched` (bucketed,
caminho de PRODUÇÃO) supera o Numba 16w×4t no regime alvo (27 configs, 600 pos)?

Reusa os benchers VALIDADOS (`_bench_jax_batched`/`_bench_numba` — mesmos do relatório
v2.45 → números comparáveis). Memória limitada via chunk_size_models. Cada n_models é
medido de forma independente (OOM em um n não perde os demais). Mediana de N_ITERS.
"""

import sys
import numpy as np

sys.path.insert(0, "benchmarks")
from bench_numba_vs_jax_gpu import (  # noqa: E402
    _get_default_model,
    _bench_numba,
    _bench_jax_batched,
)

N_POS = 600
N_ITERS = 5
freqs = [20000.0, 50000.0, 100000.0]
trs = [0.5, 1.0, 2.0]
dips = [0.0, 30.0, 60.0]
N_MODELS = [16, 32, 64, 128, 256, 512]

m = _get_default_model()
positions_z = np.linspace(m.min_depth - 1.0, m.max_depth + 1.0, N_POS)
nf, nTR, nAng = len(freqs), len(trs), len(dips)
print(f"# modelo={m.name} n={m.n_layers} | n_pos={N_POS} | "
      f"{nf}f×{nTR}TR×{nAng}dip={nf*nTR*nAng} configs | c128 | iters={N_ITERS}",
      flush=True)

common = dict(
    rho_h=m.rho_h, rho_v=m.rho_v, esp=m.esp, positions_z=positions_z,
    frequencies_hz=freqs, tr_spacings_m=trs, dip_degs=dips, n_iters=N_ITERS,
)

# ── Numba 16w×4t (referência fixa, independe de n_models) ────────────────────
t_n = _bench_numba(n_workers=16, threads_per_worker=4, **common)
mh_numba = N_ITERS / t_n * 3600.0
print(f"# numba 16w×4t = {mh_numba:.3e} mod/h (referência fixa)\n", flush=True)

print("n_models,chunk,jax_batched_modh,jax/numba", flush=True)
for nm in N_MODELS:
    # Chunk do eixo de modelos p/ limitar VRAM (None até 32; depois 32).
    chunk = None if nm <= 32 else 32
    try:
        t = _bench_jax_batched(
            dtype="complex128", strategy="bucketed",
            n_models=nm, chunk_size_models=chunk, **common,
        )
        mh = N_ITERS * nm / t * 3600.0
        flag = "  <-- supera Numba" if mh > mh_numba else ""
        print(f"{nm},{chunk},{mh:.3e},{mh / mh_numba:.2f}{flag}", flush=True)
    except Exception as e:
        print(f"{nm},{chunk},FALHA,{str(e)[:70]}", flush=True)
