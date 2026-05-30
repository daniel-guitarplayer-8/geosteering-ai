"""Driver one-off (Sprint v2.45, item 6) — Cenário H a 600 n_pos.

O cenário H (512 configs = 8 freq × 8 TR × 8 dip) a 600 n_pos é impraticável
pelo bench completo porque o caminho **JAX single-loop** (`_bench_jax`) percorre
512 configs × N_iters dispatches GPU sequenciais em Python single-thread (>30 min,
GPU ociosa). Este driver mede APENAS os dois caminhos de produção relevantes ao
item 6 — **Numba 16w×4t (single)** vs **JAX batched-bucketed (vmap n_models)** —
pulando o single-loop legacy. Complementa `v2.45_600pos_AG_16w4t.csv` (cenários A-G).

NÃO faz parte da suíte. Reusa as funções validadas do bench.
"""

import sys
import numpy as np

sys.path.insert(0, "benchmarks")
from bench_numba_vs_jax_gpu import (  # noqa: E402
    _bench_numba,
    _bench_jax_batched,
    _get_default_model,
    _build_scenario_params,
)

N_POS = 600
N_ITERS = 2
N_MODELS = 16
CHUNK = 8

m = _get_default_model()
p = _build_scenario_params("H", N_POS)
configs = len(p["frequencies_hz"]) * len(p["tr_spacings_m"]) * len(p["dip_degs"])
positions_z = np.linspace(m.min_depth - 1.0, m.max_depth + 1.0, N_POS)
common = dict(
    rho_h=m.rho_h, rho_v=m.rho_v, esp=m.esp,
    positions_z=positions_z,
    frequencies_hz=p["frequencies_hz"],
    tr_spacings_m=p["tr_spacings_m"],
    dip_degs=p["dip_degs"],
    n_iters=N_ITERS,
)
print(
    f"# H @ {N_POS} pos | configs={configs} | modelo={m.name} n_layers={m.n_layers} "
    f"| Numba 16w×4t single vs JAX batched-bucketed n_models={N_MODELS} chunk={CHUNK}",
    flush=True,
)

rows = ["scenario,backend,regime,strategy,dtype,n_pos,n_iters,wall_s,mod_per_h"]

# Numba 16w×4t (single)
t_n = _bench_numba(n_workers=16, threads_per_worker=4, **common)
mh_n = N_ITERS / t_n * 3600.0
print(f"  [numba 16w×4t] wall={t_n:.3f}s | mod/h={mh_n:.3e}", flush=True)
rows.append(f"H,numba,single,16w4t,complex128,{N_POS},{N_ITERS},{t_n:.6f},{mh_n:.6e}")

# JAX batched-bucketed (vmap n_models, chunked)
t_b = _bench_jax_batched(
    dtype="complex128", strategy="bucketed",
    n_models=N_MODELS, chunk_size_models=CHUNK, **common,
)
mh_b = N_ITERS * N_MODELS / t_b * 3600.0
print(f"  [jax/batched/bucketed_k{CHUNK}_x{N_MODELS}] wall={t_b:.3f}s | mod/h={mh_b:.3e}", flush=True)
rows.append(
    f"H,jax_batched,batched,bucketed_k{CHUNK}_x{N_MODELS},complex128,"
    f"{N_POS},{N_ITERS},{t_b:.6f},{mh_b:.6e}"
)

out = "benchmarks/results/v2.45_600pos_H_16w4t.csv"
with open(out, "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"# CSV salvo: {out}", flush=True)
