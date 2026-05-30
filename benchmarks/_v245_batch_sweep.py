"""Driver one-off (Sprint v2.45, item 1) — crossover por batch-size.

Para cada cenário canônico A-H, varre n_models ∈ {8,16,32,64,128,600} e mede
JAX batched-bucketed vs Numba 16w×4t (single). Determina o crossover: menor
n_models onde JAX batched > Numba. Cenários pesados (G/H) limitados a ≤64
modelos (inviável a 128/600). Saída: CSV + tabela.

NÃO faz parte da suíte — script de relatório. Reusa funções do bench validado.
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
from geosteering_ai.cli.benchmark import SCENARIOS  # noqa: E402

BATCH = [8, 16, 32, 64, 128, 600]
HEAVY = {"G", "H"}  # cap em 64 (128/600 inviável a 64/512 configs)
ITERS = 2

m = _get_default_model()
print(f"# modelo={m.name} n_layers={m.n_layers} | Numba 16w×4t single | iters={ITERS}", flush=True)
hdr = "scenario,configs,n_pos,numba_16w4t_single_modh," + ",".join(
    f"jaxbatch_n{n}_modh" for n in BATCH
) + ",crossover_n_models"
print(hdr, flush=True)

rows = [hdr]
for scen in SCENARIOS:
    p = _build_scenario_params(scen, int(SCENARIOS[scen]["n_pos"]))
    npos = p["n_positions"]
    configs = len(p["frequencies_hz"]) * len(p["tr_spacings_m"]) * len(p["dip_degs"])
    common = dict(
        rho_h=m.rho_h, rho_v=m.rho_v, esp=m.esp,
        positions_z=np.linspace(m.min_depth - 1.0, m.max_depth + 1.0, npos),
        frequencies_hz=p["frequencies_hz"],
        tr_spacings_m=p["tr_spacings_m"],
        dip_degs=p["dip_degs"],
        n_iters=ITERS,
    )
    try:
        mh_n = ITERS / _bench_numba(n_workers=16, threads_per_worker=4, **common) * 3600.0
    except Exception as e:
        print(f"  # {scen} numba FALHA: {str(e)[:80]}", flush=True)
        mh_n = float("nan")

    batches = BATCH if scen not in HEAVY else [8, 16, 32, 64]
    jax_vals = {}
    cross = None
    for nm in batches:
        # Chunk do eixo de modelos APENAS sob pressão real de memória (proxy:
        # configs × n_models grande). Cenários leves (poucos configs) rodam
        # monolíticos mesmo a 600 modelos → curva de escala monotônica (sem o
        # artefato de chunk=8 fixo que estrangulava throughput a nm>64).
        chunk = 8 if (configs * nm >= 2048) else None
        try:
            t = _bench_jax_batched(
                dtype="complex128", strategy="bucketed",
                n_models=nm, chunk_size_models=chunk, **common,
            )
            mh = ITERS * nm / t * 3600.0
            jax_vals[nm] = mh
            if cross is None and mh > mh_n:
                cross = nm
        except Exception as e:
            print(f"  # {scen} jax n={nm} FALHA: {str(e)[:80]}", flush=True)
            jax_vals[nm] = float("nan")

    cells = [scen, str(configs), str(npos), f"{mh_n:.0f}"]
    cells += [f"{jax_vals.get(n, float('nan')):.0f}" for n in BATCH]
    cells.append(str(cross) if cross else ">max")
    line = ",".join(cells)
    print(line, flush=True)
    rows.append(line)

out = "benchmarks/results/v2.45_batch_sweep.csv"
with open(out, "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"# CSV salvo: {out}", flush=True)
