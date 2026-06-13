# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_grouped_reassembly.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : JAX batched grouped — reassembly vetorizado (scatter)      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-10                                                 ║
# ║  Status      : Produção (gate de merge — paridade + pico de memória)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do reassembly VETORIZADO de ``simulate_multi_jax_batched_grouped``.

Recuperado do WIP ``archive/wip-stash-gui0013`` (dívida do Sprint C). A função
passou a pré-alocar o tensor de saída UMA vez e espalhar cada grupo por
indexação avançada ``H_tensor[sel] = Hg`` (scatter), no lugar do padrão antigo
``[None]*n_models`` + ``np.stack``. O ganho é PICO DE MEMÓRIA (~−49% a 30k×600,
evita OOM em runs multi-config); a bit-exatidão é preservada (mesmos índices,
mesmos valores — só muda o padrão de memória).

Cobre:
  (1) **Paridade de ordem** (gate JAX): geometria HETEROGÊNEA com grupos
      INTERCALADOS → cada grupo é reposicionado na ORDEM GLOBAL original,
      bit-idêntico a uma chamada independente do mesmo subconjunto.
  (2) **Pico de memória** (microcosmo NumPy, sem JAX): o padrão prealloc+scatter
      tem pico de RAM medido (``tracemalloc``) sensivelmente abaixo do padrão
      lista+``np.stack`` — validando o mecanismo de forma determinística.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

jax_only = pytest.mark.skipif(not HAS_JAX, reason="JAX não instalado")


# ─────────────────────────────────────────────────────────────────────────────
# (1) Paridade de ordem — grupos intercalados reposicionados corretamente
# ─────────────────────────────────────────────────────────────────────────────
@jax_only
def test_grouped_reassembly_preserves_interleaved_model_order():
    """Scatter ``H_tensor[sel] = Hg`` reposiciona cada grupo na ordem GLOBAL.

    Constrói um batch com DUAS geometrias intercaladas (índices pares → geom A,
    ímpares → geom B). Roda o caminho agrupado e, independentemente, roda cada
    subconjunto SOZINHO por :func:`simulate_multi_jax_batched`. O slice global
    do resultado agrupado DEVE ser bit-idêntico ao subconjunto independente —
    prova que o scatter usou os índices GLOBAIS (``sel``), não locais.

    Esta é a regressão exata que o reassembly antigo→novo poderia introduzir:
    se o scatter usasse índices locais, os grupos sairiam fora de ordem.
    """
    from geosteering_ai.simulation import (
        SimulationConfig,
        simulate_multi_jax_batched,
        simulate_multi_jax_batched_grouped,
    )

    rng = np.random.default_rng(7)
    n_models = 8
    # ── Resistividades distintas por modelo; geometria (esp) em DOIS padrões ──
    rh = rng.uniform(1.0, 100.0, (n_models, 3))
    rv = rh.copy()
    esp = np.empty((n_models, 1), dtype=np.float64)
    esp[0::2] = 5.0  # geom A nos índices pares  → grupo {0,2,4,6}
    esp[1::2] = 9.0  # geom B nos índices ímpares → grupo {1,3,5,7}
    pz = np.linspace(-2.0, 10.0, 16)
    kw = dict(frequencies_hz=[2e4], tr_spacings_m=[1.0], dip_degs=[0.0])
    cfg = SimulationConfig(backend="jax", jax_strategy="bucketed", dtype="complex128")

    H_grp, info = simulate_multi_jax_batched_grouped(rh, rv, esp, pz, cfg=cfg, **kw)
    H_grp = np.asarray(H_grp)
    assert H_grp.shape[0] == n_models
    assert info["n_groups"] == 2, f"esperava 2 grupos, obteve {info['n_groups']}"

    # ── Referência independente: cada subconjunto rodado SOZINHO ──────────────
    for sel in (np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])):
        ref = simulate_multi_jax_batched(rh[sel], rv[sel], esp[sel], pz, cfg=cfg, **kw)
        ref_H = np.asarray(ref.H_tensor)
        # Bit-idêntico: mesmo kernel, mesmos inputs → scatter só reposicionou.
        assert np.array_equal(H_grp[sel], ref_H), (
            f"slice global {sel.tolist()} divergiu da referência independente "
            "→ scatter usou índice errado (regressão de ordem)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# (2) Pico de memória — prealloc+scatter < lista+np.stack (microcosmo NumPy)
# ─────────────────────────────────────────────────────────────────────────────
def _make_group(sel: np.ndarray, slice_shape: tuple[int, ...]) -> np.ndarray:
    """Array de grupo determinístico ``(len(sel), *slice_shape)`` — preenchido
    pelo índice GLOBAL de cada modelo (permite validar a ordem do reassembly)."""
    g = np.empty((len(sel), *slice_shape), dtype=np.complex128)
    for j, gi in enumerate(sel):
        g[j] = float(gi)  # valor = índice global → ordem verificável
    return g


def _old_pattern(index_groups, n_models, slice_shape):
    """Padrão ANTIGO: views em ``per[]`` pinam todos os Hg vivos até ``np.stack``
    → pico ≈ 2× a saída (todos os grupos + tensor final empilhado)."""
    per: list = [None] * n_models
    for sel in index_groups:
        Hg = _make_group(sel, slice_shape)  # fresh alloc por grupo
        for local_j, global_i in enumerate(sel):
            per[int(global_i)] = Hg[local_j]  # view → pina Hg
    return np.stack(per, axis=0)


def _new_pattern(index_groups, n_models, slice_shape):
    """Padrão NOVO: prealloc 1× + scatter; cada Hg é COPIADO e liberado na
    iteração seguinte → pico ≈ 1× a saída + 1 grupo."""
    H = None
    for sel in index_groups:
        Hg = _make_group(sel, slice_shape)
        if H is None:
            H = np.empty((n_models, *slice_shape), dtype=Hg.dtype)
        H[sel] = Hg  # scatter (copia) — Hg liberado depois
    return H


def _peak_mb(fn, *args) -> float:
    """Pico de RAM (MB) de ``fn(*args)`` via tracemalloc. ``tracemalloc`` rastreia
    buffers NumPy neste ambiente (verificado)."""
    tracemalloc.start()
    out = fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Mantém ``out`` vivo até a medição (evita GC precoce mascarar o pico).
    assert out is not None
    return peak / 1e6


def test_scatter_reassembly_peak_below_stack():
    """Mecanismo de memória: prealloc+scatter tem pico < lista+``np.stack``.

    Microcosmo NumPy puro (sem JAX) que replica fielmente os dois padrões de
    reassembly: o antigo retém views de TODOS os K grupos até o stack final
    (pico ≈ 2× saída); o novo copia e libera cada grupo (pico ≈ 1× saída +
    1 grupo). Com K grupos equilibrados, o novo deve ficar bem abaixo do antigo.
    Validação determinística do ganho de −49% medido em produção (30k×600).
    """
    n_models = 64
    K = 8
    slice_shape = (1, 1, 600, 2, 9)  # ~ (nTR, nAng, n_pos, nf, 9) reduzido
    # K grupos equilibrados, índices INTERCALADOS (ordem não-trivial).
    index_groups = [np.arange(k, n_models, K) for k in range(K)]

    # ── Correção primeiro: ambos os padrões produzem a MESMA saída ───────────
    out_old = _old_pattern(index_groups, n_models, slice_shape)
    out_new = _new_pattern(index_groups, n_models, slice_shape)
    assert np.array_equal(out_old, out_new), "padrões divergiram (microcosmo inválido)"
    # Cada modelo i preenchido com seu índice global i → ordem correta.
    for i in range(n_models):
        assert out_new[i].flat[0] == float(i)

    # ── Pico de memória: novo sensivelmente abaixo do antigo ─────────────────
    peak_old = _peak_mb(_old_pattern, index_groups, n_models, slice_shape)
    peak_new = _peak_mb(_new_pattern, index_groups, n_models, slice_shape)
    output_mb = (n_models * np.prod(slice_shape) * 16) / 1e6  # complex128 = 16 B

    # Margem generosa (determinístico, mas evita fragilidade): o novo deve ficar
    # abaixo de 75% do antigo. Teórico: novo ≈ (1+1/K)× saída; antigo ≈ 2× saída.
    assert peak_new < peak_old * 0.75, (
        f"pico novo={peak_new:.1f}MB não ficou <75% do antigo={peak_old:.1f}MB "
        f"(saída≈{output_mb:.1f}MB) — o scatter não reduziu o pico como esperado."
    )
    # Sanidade: o novo não deve exceder ~1.5× a saída (não retém K grupos).
    assert peak_new < output_mb * 1.5, (
        f"pico novo={peak_new:.1f}MB excedeu 1.5×saída={output_mb:.1f}MB "
        "— prealloc+scatter pode estar retendo grupos."
    )
