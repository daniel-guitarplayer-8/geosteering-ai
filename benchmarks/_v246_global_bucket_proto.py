"""Protótipo (investigação high-config) — GLOBAL CROSS-CONFIG BUCKETING.

Hipótese: o batched-bucketed atual (O4) percorre os configs (TR×dip) em LOOP
PYTHON, e dentro de cada config faz 1 vmap-sobre-modelos por bucket (ct,cr). Para
27 configs (3 TR × 3 dip × 3 freq) isso são ~27×B lançamentos GPU pequenos
(n_models=16 → ocupação baixa). O kernel `_single_position_jax` fecha (ct,cr)
ESTÁTICOS mas aceita dz_half/r_half/dip_rad/freq como entradas DINÂMICAS.

Estratégia: agrupar GLOBALMENTE todos os pares (config, posição) por (ct,cr), e
para cada bucket global fazer UM vmap sobre (modelos × entradas × freq), passando
a geometria por-entrada (z,dz,r,dip) como eixos vmapped. Isso:
  - preserva o kernel bucketed rápido ((ct,cr) estáticos no closure);
  - colapsa ~27×B lançamentos em ~B_global lançamentos MAIORES (melhor ocupação);
  - é bit-idêntico (mesmo kernel, mesma física — só reorganiza o batching).

Mede paridade vs simulate_multi_jax_batched (atual) e throughput vs Numba 16w×4t.
NÃO faz parte da suíte — script de investigação.
"""

import sys
import time
import numpy as np

sys.path.insert(0, "benchmarks")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from geosteering_ai.simulation._jax.kernel import _single_position_jax  # noqa: E402
from geosteering_ai.simulation._jax.forward_pure import (  # noqa: E402
    build_static_context_cached,
    _COMPLEX_DTYPE_MAP,
)
from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax_batched  # noqa: E402
from geosteering_ai.simulation import SimulationConfig  # noqa: E402
from bench_numba_vs_jax_gpu import _get_default_model, _bench_numba  # noqa: E402


# ── Kernel de bucket GLOBAL: geometria (dz,r,dip) por-entrada ────────────────
_GLOBAL_JIT = {}


def _get_global_bucket_jit(ct, cr, n, npt, cdtype):
    key = (int(ct), int(cr), int(n), int(npt), cdtype)
    if key in _GLOBAL_JIT:
        return _GLOBAL_JIT[key]
    cdtype_jnp = _COMPLEX_DTYPE_MAP[cdtype]

    def _fwd(rho_h, rho_v, z_arr, dz_arr, r_arr, dip_arr, freqs,
             h_arr, prof_arr, krJ0J1, wJ0, wJ1):
        eta = jnp.stack([1.0 / rho_h, 1.0 / rho_v], axis=-1)

        def _one_entry_one_freq(z_mid, dz_half, r_half, dip_rad, freq):
            Tz = z_mid + dz_half
            cz = z_mid - dz_half
            return _single_position_jax(
                r_half, 0.0, Tz, -r_half, 0.0, cz, dip_rad,
                n, npt, ct, cr, rho_h, rho_v, h_arr, prof_arr, eta, freq,
                krJ0J1, wJ0, wJ1, use_native_dipoles=True, complex_dtype=cdtype_jnp,
            )

        # vmap sobre freq, depois sobre entradas (z,dz,r,dip por-entrada).
        vmap_freq = jax.vmap(_one_entry_one_freq, in_axes=(None, None, None, None, 0))
        vmap_entries = jax.vmap(vmap_freq, in_axes=(0, 0, 0, 0, None))
        return vmap_entries(z_arr, dz_arr, r_arr, dip_arr, freqs)  # (n_entry, nf, 9)

    jitted = jax.jit(_fwd)
    _GLOBAL_JIT[key] = jitted
    return jitted


def global_bucket_forward(rho_h_batch, rho_v_batch, ctxs, n, npt, n_pos, nf,
                          n_configs, cdtype):
    """Forward global-bucket. ctxs: lista de ForwardPureContext por config.

    Retorna (n_models, n_configs, n_pos, nf, 9).
    """
    n_models = rho_h_batch.shape[0]
    cdtype_jnp = _COMPLEX_DTYPE_MAP[cdtype]

    # ── Entradas globais (config, pos) com (ct,cr) e geometria por-entrada ────
    ct_all = np.concatenate([np.asarray(c.camad_t_array, np.int64) for c in ctxs])
    cr_all = np.concatenate([np.asarray(c.camad_r_array, np.int64) for c in ctxs])
    cfg_idx = np.repeat(np.arange(n_configs), n_pos)
    pos_idx = np.tile(np.arange(n_pos), n_configs)
    z_all = np.concatenate([np.asarray(c.positions_z_jnp) for c in ctxs])
    dz_all = np.array([c.dz_half for c in ctxs]).repeat(n_pos)
    r_all = np.array([c.r_half for c in ctxs]).repeat(n_pos)
    dip_all = np.array([c.dip_rad for c in ctxs]).repeat(n_pos)

    # Arrays compartilhados (mesma geometria de camadas/filtro/freqs).
    c0 = ctxs[0]
    freqs = c0.freqs_hz_jnp
    h_arr, prof_arr = c0.h_arr_jnp, c0.prof_arr_jnp
    krJ0J1, wJ0, wJ1 = c0.krJ0J1, c0.wJ0, c0.wJ1

    key_arr = ct_all * 10_000 + cr_all
    unique_keys, inverse = np.unique(key_arr, return_inverse=True)

    H = jnp.zeros((n_models, n_configs, n_pos, nf, 9), dtype=cdtype_jnp)
    in_axes = (0, 0, None, None, None, None, None, None, None, None, None, None)
    for b in range(len(unique_keys)):
        idx = np.nonzero(inverse == b)[0]
        ct = int(ct_all[idx[0]])
        cr = int(cr_all[idx[0]])
        z_arr = jnp.asarray(z_all[idx])
        dz_arr = jnp.asarray(dz_all[idx])
        r_arr = jnp.asarray(r_all[idx])
        dip_arr = jnp.asarray(dip_all[idx])

        jitted = _get_global_bucket_jit(ct, cr, n, npt, cdtype)
        over_models = jax.vmap(jitted, in_axes=in_axes)
        H_bucket = over_models(
            rho_h_batch, rho_v_batch, z_arr, dz_arr, r_arr, dip_arr,
            freqs, h_arr, prof_arr, krJ0J1, wJ0, wJ1,
        )  # (n_models, n_entry, nf, 9)
        H = H.at[:, jnp.asarray(cfg_idx[idx]), jnp.asarray(pos_idx[idx])].set(H_bucket)
    H.block_until_ready()
    return H


def _build_ctxs(model, positions_z, freqs, trs, dips, cdtype):
    ctxs = []
    cfg_order = []  # (i_tr, i_ang) na ordem (tr-outer, dip-inner) p/ casar tensor
    for i_tr, L in enumerate(trs):
        for i_ang, th in enumerate(dips):
            ctx = build_static_context_cached(
                rho_h=model.rho_h, rho_v=model.rho_v, esp=model.esp,
                positions_z=positions_z, freqs_hz=freqs,
                tr_spacing_m=float(L), dip_deg=float(th),
                hankel_filter="werthmuller_201pt", strategy="bucketed",
                chunk_size=None, complex_dtype=cdtype,
            )
            ctxs.append(ctx)
            cfg_order.append((i_tr, i_ang))
    return ctxs, cfg_order


def main():
    m = _get_default_model()
    N_POS, N_MODELS, ITERS = 600, 16, 3
    freqs = [20000.0, 50000.0, 100000.0]
    trs = [0.5, 1.0, 2.0]
    dips = [0.0, 30.0, 60.0]
    nf, nTR, nAng = len(freqs), len(trs), len(dips)
    n_configs = nTR * nAng
    cdtype = "complex128"
    positions_z = np.linspace(m.min_depth - 1.0, m.max_depth + 1.0, N_POS)
    n = m.n_layers
    print(f"# modelo={m.name} n={n} | n_pos={N_POS} | {nf}f×{nTR}TR×{nAng}dip="
          f"{nf*n_configs} configs | n_models={N_MODELS} | {cdtype}", flush=True)

    ctxs, cfg_order = _build_ctxs(m, positions_z, freqs, trs, dips, cdtype)
    npt = ctxs[0].krJ0J1.shape[0]
    rng = np.random.default_rng(0)

    # ── QUICK_WIN: cardinalidade de partições + buckets (NumPy, read-only) ────
    # K = nº de tabelas (ct_array, cr_array) DISTINTAS entre as 9 (TR,dip) configs.
    # Teto de ganho da fusão-por-partição: dispatches caem de Σ_config(buckets) p/
    # n_buckets_global. freq não conta (já interno ao bucket).
    part_keys = set()
    per_config_buckets = []
    all_ct, all_cr = [], []
    for c in ctxs:
        ct = np.asarray(c.camad_t_array, np.int64)
        cr = np.asarray(c.camad_r_array, np.int64)
        part_keys.add((ct.tobytes(), cr.tobytes()))
        per_config_buckets.append(len(np.unique(ct * 10_000 + cr)))
        all_ct.append(ct)
        all_cr.append(cr)
    ct_cat = np.concatenate(all_ct)
    cr_cat = np.concatenate(all_cr)
    n_global_buckets = len(np.unique(ct_cat * 10_000 + cr_cat))
    dispatches_current = sum(per_config_buckets)
    print(f"# K (partições distintas das {n_configs} configs TR×dip) = {len(part_keys)}")
    print(f"# dispatches ATUAL (Σ buckets/config) = {dispatches_current}  "
          f"(buckets/config: {per_config_buckets})")
    print(f"# dispatches PROTO (buckets globais)  = {n_global_buckets}  "
          f"→ teto de redução {dispatches_current / max(1, n_global_buckets):.1f}×", flush=True)

    cfg = SimulationConfig(backend="jax", jax_strategy="bucketed", dtype=cdtype)
    esp_full = np.tile(m.esp, (1, 1))

    def make_runners(nm):
        rho_h_b = jnp.asarray(rng.uniform(1.0, 100.0, (nm, n)))
        rho_v_b = rho_h_b
        rho_h_np = np.asarray(rho_h_b)
        rho_v_np = np.asarray(rho_v_b)
        esp_b = np.tile(m.esp, (nm, 1))

        def run_current():
            res = simulate_multi_jax_batched(
                rho_h_np, rho_v_np, esp_b, positions_z,
                frequencies_hz=freqs, tr_spacings_m=trs, dip_degs=dips, cfg=cfg,
            )
            return res.H_tensor

        def run_proto():
            return global_bucket_forward(
                rho_h_b, rho_v_b, ctxs, n, npt, N_POS, nf, n_configs, cdtype)

        return run_current, run_proto

    # ── Paridade (1× em n_models=16) ─────────────────────────────────────────
    rc16, rp16 = make_runners(16)
    Hc = rc16()
    Hp = np.asarray(rp16())
    diff = np.abs(Hc - Hp.reshape(16, nTR, nAng, N_POS, nf, 9))
    print(f"# PARIDADE proto vs atual (n=16): max|Δ|={diff.max():.3e}  "
          f"mean|Δ|={diff.mean():.3e}  (alvo <1e-13)", flush=True)

    # Numba 16w×4t (independe de n_models — mod/h normalizado por modelo)
    t_n = _bench_numba(
        rho_h=m.rho_h, rho_v=m.rho_v, esp=m.esp, positions_z=positions_z,
        frequencies_hz=freqs, tr_spacings_m=trs, dip_degs=dips,
        n_iters=ITERS, n_workers=16, threads_per_worker=4,
    )
    mh_numba = ITERS / t_n * 3600.0
    print(f"# numba 16w×4t: {mh_numba:.3e} mod/h (referência fixa)\n", flush=True)

    def bench(fn, nm):
        fn()  # warmup
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        dt = (time.perf_counter() - t0) / ITERS
        return nm / dt * 3600.0

    # ── Sweep n_models: atual × proto × numba ────────────────────────────────
    print("n_models,numba_modh,atual_modh,proto_modh,proto/atual,atual/numba,proto/numba")
    for nm in [16, 32, 64, 128, 256]:
        try:
            rc, rp = make_runners(nm)
            mh_cur = bench(rc, nm)
            mh_pro = bench(rp, nm)
            print(f"{nm},{mh_numba:.3e},{mh_cur:.3e},{mh_pro:.3e},"
                  f"{mh_pro/mh_cur:.2f},{mh_cur/mh_numba:.2f},{mh_pro/mh_numba:.2f}",
                  flush=True)
        except Exception as e:
            print(f"{nm},FALHA,{str(e)[:60]}", flush=True)


if __name__ == "__main__":
    main()
