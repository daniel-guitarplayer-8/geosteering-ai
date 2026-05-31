"""Sprint C — Fase B: treino do SurrogateNet (rho → H_EM) com TF + XLA.

Carrega o dataset gerado na Fase A (.npz), constrói o SurrogateNet (TCN) via
`build_surrogate`, treina com `jit_compile=True` (XLA) + `tf.data` (prefetch) e
reporta a curva de perda (valida o fim-a-fim). Processo SEPARADO da Fase A (JAX)
para evitar contenção de VRAM JAX/TF.

NOTA FÍSICA: o SurrogateNet aprende o forward map DETERMINÍSTICO (ρ→H); portanto
treina em pares LIMPOS. O "ruído on-the-fly" do projeto é do pipeline de INVERSÃO
(H→ρ), não do surrogate forward. XLA + prefetch são aplicados (treino TF acelerado).

Uso: python benchmarks/_sprintc_phase_b_train.py [EPOCHS] [BATCH]
"""

import sys
import time

import numpy as np

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 64
NPZ = "benchmarks/results/sprintc_surrogate_dataset.npz"


def main() -> None:
    import tensorflow as tf

    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.models.surrogate import build_surrogate

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"# TF GPUs: {len(gpus)} | device: {[g.name for g in gpus]}", flush=True)

    # ── Carrega dataset (Fase A) ─────────────────────────────────────────────
    d = np.load(NPZ, allow_pickle=True)
    x_rho = d["x_rho"].astype(np.float32)  # (N, L, 2)
    y_em = d["y_em"].astype(np.float32)  # (N, L, 2K)
    comps = list(d["components"])
    n_models = x_rho.shape[0]
    print(
        f"# dataset: x_rho={x_rho.shape} y_em={y_em.shape} comps={comps} "
        f"(gen {float(d['gen_seconds']):.1f}s, backend={d['backend']})",
        flush=True,
    )

    # ── Split POR MODELO (P1) — 90/10 ────────────────────────────────────────
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_models)
    n_val = max(1, n_models // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    # Normaliza o alvo H (escala por std do treino → estabiliza a perda).
    y_std = float(y_em[tr_idx].std()) or 1.0
    y_tr, y_va = y_em[tr_idx] / y_std, y_em[val_idx] / y_std
    x_tr, x_va = x_rho[tr_idx], x_rho[val_idx]
    print(f"# split: train={len(tr_idx)} val={len(val_idx)} | y_std={y_std:.3e}", flush=True)

    cfg = PipelineConfig(surrogate_output_components=comps)
    model = build_surrogate(cfg)
    print(f"# SurrogateNet params: {model.count_params():,}", flush=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"],
        jit_compile=True,  # XLA
    )

    def _ds(x, y, training):
        d = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            d = d.shuffle(min(len(x), 8192))
        return d.batch(BATCH).prefetch(tf.data.AUTOTUNE)

    train_ds = _ds(x_tr, y_tr, True)
    val_ds = _ds(x_va, y_va, False)

    # ── Treino (XLA) ─────────────────────────────────────────────────────────
    print(f"# treinando {EPOCHS} épocas, batch={BATCH} (XLA jit_compile)...", flush=True)
    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    train_s = time.perf_counter() - t0

    tl = hist.history["loss"]
    vl = hist.history["val_loss"]
    print("\n# ── CURVA DE PERDA (MSE, alvo normalizado) ──", flush=True)
    for e in range(EPOCHS):
        print(f"#   época {e + 1}: train={tl[e]:.4e}  val={vl[e]:.4e}", flush=True)
    improved = tl[-1] < tl[0] and vl[-1] < vl[0]
    print(
        f"\n# RESUMO: train {tl[0]:.3e}→{tl[-1]:.3e} ({tl[-1] / tl[0]:.2f}×), "
        f"val {vl[0]:.3e}→{vl[-1]:.3e} ({vl[-1] / vl[0]:.2f}×) | "
        f"treino {train_s:.0f}s ({train_s / EPOCHS:.0f}s/época)",
        flush=True,
    )
    print(f"# PERDA DECRESCENTE (fim-a-fim OK): {improved}", flush=True)


if __name__ == "__main__":
    main()
