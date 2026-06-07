"""Sprint C — Fase A: geração do dataset (rho → H_EM) p/ treino do SurrogateNet.

PRODUTIZADO (v2.49): usa a API on-the-fly
:func:`geosteering_ai.data.surrogate_data.generate_surrogate_dataset` — que roteia
pelo dispatcher Sprint B (`simulate_batch` JAX GPU ⇄ Numba), monta os pares
(x_rho, y_em) VETORIZADO (sem o laço Python O(N×L) do `.dat`) e pula a construção
do `.dat` (`build_dat_22col=False`). Substitui o `_layer_at` local + a montagem
22-col manual desta Fase A. Salva em .npz (Fase B treina em processo separado —
evita contenção JAX/TF na GPU).

Uso: python benchmarks/_sprintc_phase_a_gen.py [N_MODELS] [N_POS] [N_GEOMETRIES]
"""

import sys
import time

import numpy as np

N_MODELS = int(sys.argv[1]) if len(sys.argv) > 1 else 30_000
N_POS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
N_GEO = int(sys.argv[3]) if len(sys.argv) > 3 else 256
N_LAYERS = 5
OUT = "benchmarks/results/sprintc_surrogate_dataset.npz"


def main() -> None:
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.surrogate_data import generate_surrogate_dataset

    cfg = PipelineConfig(
        simulator_backend="auto", surrogate_output_components=["XX", "ZZ"]
    )

    # ── Geração on-the-fly produtizada (templates → bucketed grouped rápido) ──
    # geometry_mode="templates" + n_geometries=N_GEO replica K geometrias distintas
    # → group_by_geometry forma K grupos homogêneos → kernel bucketed (1.5–1.9× Numba).
    # build_dat_22col=False (interno) pula o .dat; reassembly vetorizada (v2.49).
    print(
        f"# Fase A (produtizada) — {N_MODELS} modelos × {N_POS} pos, "
        f"{N_GEO} geometrias (templates), backend=auto...",
        flush=True,
    )
    t0 = time.perf_counter()
    ds = generate_surrogate_dataset(
        cfg,
        n_models=N_MODELS,
        n_positions=N_POS,
        n_layers=N_LAYERS,
        rho_h_range=(1.0, 1000.0),
        rho_v_range=(1.0, 1000.0),
        thickness_range=(1.0, 10.0),
        strategy="log_uniform",
        seed=2026,
        frequencies_hz=[20000.0],
        geometry_mode="templates",
        n_geometries=N_GEO,
        jax_chunk_size_models=128,
    )
    gen_s = time.perf_counter() - t0
    print(
        f"# gerado em {gen_s:.1f}s | throughput={N_MODELS / gen_s * 3600:.3e} mod/h",
        flush=True,
    )
    print(
        f"# SurrogateDataset: x_rho={ds.x_rho.shape} y_em={ds.y_em.shape} "
        f"(K={len(cfg.surrogate_output_components)}: {cfg.surrogate_output_components})",
        flush=True,
    )

    np.savez_compressed(
        OUT,
        x_rho=ds.x_rho.astype(np.float32),
        y_em=ds.y_em.astype(np.float32),
        components=np.array(cfg.surrogate_output_components),
        gen_seconds=gen_s,
        backend="auto",
    )
    print(f"# salvo: {OUT}  (x_rho {ds.x_rho.nbytes / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
