"""Sprint C — Fase A: geração do dataset (rho → H_EM) p/ treino do SurrogateNet.

Usa o DISPATCHER produtizado (Sprint B) `simulate_batch(backend="auto")` diretamente
(roteia JAX GPU p/ n≥32 + geometria agrupável) + construção VETORIZADA do array 22-col
(evita o loop Python O(N×L) do `dat_22col` do gerador, inviável a 30k×600). Extrai os
pares (x_rho, y_em) via `extract_surrogate_pairs` e salva em .npz (Fase B treina em
processo separado — evita contenção JAX/TF na GPU).

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


def _layer_at(positions_z: np.ndarray, esp_all: np.ndarray, n_layers: int) -> np.ndarray:
    """Índice de camada por (modelo, posição) — VETORIZADO (== _find_layer_for_z)."""
    n, n_esp = esp_all.shape
    if n_esp == 0:
        return np.zeros((n, positions_z.shape[0]), dtype=np.int64)
    acc = np.cumsum(esp_all, axis=1)  # (n, n_esp) — fronteiras de camada por modelo
    z = positions_z[None, None, :]  # (1, 1, L)
    count = (acc[:, :, None] <= z).sum(axis=1)  # (n, L) — fronteiras <= z
    layer = count + 1  # z>=0 → camada 1.. (top half-space = 0)
    layer = np.where(positions_z[None, :] < 0, 0, layer)
    return np.clip(layer, 0, n_layers - 1).astype(np.int64)


def main() -> None:
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.surrogate_data import extract_surrogate_pairs
    from geosteering_ai.simulation import (
        SimulationConfig,
        simulate_batch,
        simulate_multi_jax_batched,
    )

    cfg = PipelineConfig(
        simulator_backend="auto", surrogate_output_components=["XX", "ZZ"]
    )
    rng = np.random.default_rng(2026)

    # ── Amostragem: rho diverso (log-uniform) × geometria por TEMPLATES ───────
    templates = rng.uniform(1.0, 10.0, (N_GEO, N_LAYERS - 2))
    esp = templates[np.arange(N_MODELS) % N_GEO].copy()  # ordenado por geometria
    rho_h = 10.0 ** rng.uniform(0.0, 3.0, (N_MODELS, N_LAYERS))
    rho_v = 10.0 ** rng.uniform(0.0, 3.0, (N_MODELS, N_LAYERS))
    total_thick = float(esp.sum(axis=1).max())
    positions_z = np.linspace(-1.0, total_thick + 1.0, N_POS)

    # Sanity: o dispatcher Sprint B roteia "auto" → jax (validado em N=64).
    _, dinfo = simulate_batch(
        rho_h[:64], rho_v[:64], esp[:64], positions_z[:8],
        frequencies_hz=[20000.0], backend="auto",
    )
    print(f"# dispatcher check (auto, n=64): → {dinfo['backend']}", flush=True)

    # ── Fase A: geração PER-GEOMETRIA (homogênea por bloco) + concat ─────────
    # Evita o np.stack O(n_models) do grouped (gargalo a 30k). Cada bloco é
    # geometria homogênea → simulate_multi_jax_batched bucketed direto.
    print(
        f"# Fase A — {N_MODELS} modelos × {N_POS} pos em {N_GEO} blocos homogêneos...",
        flush=True,
    )
    jax_cfg = SimulationConfig(
        backend="jax", jax_strategy="bucketed", dtype="complex128",
        jax_chunk_size_models=128,
    )
    t0 = time.perf_counter()
    H_blocks = []
    per_geo = N_MODELS // N_GEO
    for k in range(N_GEO):
        s = k * per_geo
        e = N_MODELS if k == N_GEO - 1 else (k + 1) * per_geo
        esp_k = np.tile(templates[k], (e - s, 1))
        res = simulate_multi_jax_batched(
            rho_h[s:e], rho_v[s:e], esp_k, positions_z,
            frequencies_hz=[20000.0], tr_spacings_m=[1.0], dip_degs=[0.0], cfg=jax_cfg,
        )
        H_blocks.append(np.asarray(res.H_tensor)[:, 0, 0, :, 0, :])  # (n_k, L, 9)
    H = np.concatenate(H_blocks, axis=0)  # (N, L, 9) — O(N_GEO) concat
    gen_s = time.perf_counter() - t0
    print(
        f"# gerado em {gen_s:.1f}s | throughput={N_MODELS / gen_s * 3600:.3e} mod/h",
        flush=True,
    )

    # ── Monta (N, L, 22) VETORIZADO ──────────────────────────────────────────
    layer = _layer_at(positions_z, esp, N_LAYERS)  # (N, L)
    rho_h_obs = np.take_along_axis(rho_h, layer, axis=1)  # (N, L)
    rho_v_obs = np.take_along_axis(rho_v, layer, axis=1)
    data = np.zeros((N_MODELS, N_POS, 22), dtype=np.float64)
    data[:, :, 1] = positions_z[None, :]
    data[:, :, 2] = rho_h_obs
    data[:, :, 3] = rho_v_obs
    for c in range(9):
        data[:, :, 4 + 2 * c] = H[:, :, c].real
        data[:, :, 5 + 2 * c] = H[:, :, c].imag

    # ── Extrai pares (x_rho, y_em) — apply_decoup=True (H raw) ────────────────
    ds = extract_surrogate_pairs(data, cfg, apply_decoup=True)
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
        backend=dinfo["backend"],
    )
    print(f"# salvo: {OUT}  (x_rho {ds.x_rho.nbytes / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
