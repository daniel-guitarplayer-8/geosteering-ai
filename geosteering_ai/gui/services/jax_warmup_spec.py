# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/services/jax_warmup_spec.py                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : build_warmup_specs — specs de warmup SHAPE-MATCHING do SM   ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — services (warmup config-aware do JAX GPU)            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : stdlib + numpy PURO — NÃO importa Qt NEM jax (TLS-safe)      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Deriva, de um ``SimRequest``, a lista de "specs" de geometria EXATA que ║
# ║    a 1ª simulação real do SM vai compilar no JAX GPU — para que o warmup    ║
# ║    (``warmup_jax_shapes`` no worker persistente) pré-compile os MESMOS      ║
# ║    programas XLA → 1ª sim = cache-hit (190 s → ~12 s). Reproduz a FORMA da  ║
# ║    produção: ``positions_z`` (compute_n_pos), os K templates DETERMINÍSTICOS║
# ║    (do fix #1) e a dim-líder do vmap (chunk balanceado do fix #2).          ║
# ║                                                                           ║
# ║  TLS-SAFETY (inviolável)                                                  ║
# ║    NÃO importa ``jax`` nem nada de ``simulation/_jax/`` (importar qualquer  ║
# ║    coisa de ``_jax/`` puxa ``jax`` no processo da GUI). Só usa helpers      ║
# ║    jax-free de ``sim_request`` + a réplica ``_balanced_chunk_dim``.         ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    build_warmup_specs                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``build_warmup_specs`` — deriva os specs de warmup shape-matching do ``SimRequest``."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from geosteering_ai.gui.services.sim_request import SimRequest

logger = logging.getLogger(__name__)

__all__ = ["build_warmup_specs"]

# Espelha ``dispatch._N_MODELS_GPU_THRESHOLD`` (grupos com < isso → Numba, NÃO compilam
# XLA): não aquecemos grupos que a produção roda no Numba. Paridade garantida por
# ``test_warmup_spec_gpu_threshold_matches_dispatch``.
_GPU_THRESHOLD: int = 32

# O SM sempre simula em complex128 (default de ``simulate_batch``; ``SimRequest`` não tem
# campo de dtype) → a chave do cache XLA embute o dtype, então o warmup DEVE casar.
_SM_COMPLEX_DTYPE: str = "complex128"

# Q2 (decisão do usuário): cobertura MODAL — aquece os 1-2 ``n_layers`` representativos do
# ensemble ragged (o resto compila na 1ª sim e persiste no cache de disco p/ o 2º run).
_MODAL_N_LAYERS_DEFAULT: int = 2


def build_warmup_specs(
    request: SimRequest, *, max_n_layers: int = _MODAL_N_LAYERS_DEFAULT
) -> List[Dict[str, Any]]:
    """Constrói os specs de warmup SHAPE-MATCHING a partir do ``SimRequest`` (jax-free).

    Cada spec descreve UMA geometria que a 1ª sim real vai compilar: o ``esp_template``
    (1 dos K templates determinísticos — bit-idêntico ao colapso de produção), o
    ``n_models`` (dim-líder do vmap = chunk balanceado do subgrupo) e os parâmetros de
    forma (positions_z, freq/TR/dip, dtype, filtro). Reproduz a FORMA, não a física — ρ/λ
    são irrelevantes p/ a chave do cache (só esp/positions/TR/n_models/dtype/npt importam).

    Cobertura por modo (``request.geology_mode``):
      ┌──────────────┬──────────────────────────────────────────────────────────┐
      │ "stochastic" │ K templates determinísticos × ``n_layers`` MODAL (fixo→1;  │
      │              │ ragged→1-2 do centro da faixa). Casa o colapso do fix #1.   │
      │ "manual"     │ 1 spec: a geometria manual replicada (1 grupo).            │
      │ "fixed"      │ 1 spec: a geologia fixa 3-camadas (esp idêntico).          │
      └──────────────┴──────────────────────────────────────────────────────────┘

    Args:
        request: a requisição do SM (geologia + n_models + backend + filtro).
        max_n_layers: nº máx. de ``n_layers`` a aquecer no ragged (Q2 modal; default 2).

    Returns:
        Lista de dicts-spec (picklável) p/ ``warmup_jax_shapes``. VAZIA se o backend não é
        jax/auto (Numba não compila XLA) ou se nenhum grupo atinge o limiar de GPU.

    Note:
        SÓ governa quais shapes XLA são pré-compilados — zero física. Reusa
        ``_deterministic_templates``/``_stable_geometry_seed``/``_compute_positions_z``/
        ``_resolve_group_chunk``/``_balanced_chunk_dim`` de ``sim_request`` (jax-free).
    """
    # Numba não compila XLA → warmup JAX só faz sentido p/ jax/auto.
    if request.backend not in ("jax", "jax_gpu", "auto"):
        return []

    from geosteering_ai.gui.services.sim_request import (
        _TEMPLATE_GEOMETRIES_CAP,
        _balanced_chunk_dim,
        _build_batch,
        _compute_positions_z,
        _deterministic_templates,
        _genconfig_from_request,
        _manual_models,
        _resolve_group_chunk,
        _stable_geometry_seed,
    )

    positions_z = _compute_positions_z(request)
    cap = request.jax_chunk_size_models
    common: Dict[str, Any] = {
        "positions_z": positions_z,
        "freqs_hz": tuple(float(f) for f in request.frequencies_hz),
        "tr_spacings_m": tuple(float(t) for t in request.tr_spacings_m),
        "dip_degs": tuple(float(d) for d in request.dip_degs),
        "complex_dtype": _SM_COMPLEX_DTYPE,
        "hankel_filter": request.hankel_filter,
        "jax_strategy": "bucketed",
    }

    specs: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(esp_template: Any, n_layers: int, group_size: int, n_geoms: int) -> None:
        """Acrescenta 1 spec se o grupo for batelável no JAX (≥ limiar GPU). Dedup por forma."""
        if group_size < _GPU_THRESHOLD:
            return  # produção rodaria Numba p/ esse grupo → nada a compilar no JAX
        # subgrupo = grupo / nº de geometrias distintas (K); dim-líder = chunk balanceado.
        subgroup = max(1, group_size // max(1, n_geoms))
        group_chunk = _resolve_group_chunk(group_size, cap)
        n_models = _balanced_chunk_dim(subgroup, group_chunk)
        esp = [float(v) for v in np.asarray(esp_template, dtype=np.float64).reshape(-1)]
        key = (tuple(np.round(esp, 9)), int(n_layers), int(n_models))
        if key in seen:
            return  # forma já coberta (evita compilar 2× o mesmo programa)
        seen.add(key)
        specs.append(
            {
                **common,
                "esp_template": esp,
                "n_layers": int(n_layers),
                "n_models": int(n_models),
            }
        )

    mode = request.geology_mode
    if mode == "manual":
        m = _manual_models(request)[0]
        _add(m["thicknesses"], int(m["n_layers"]), max(1, int(request.n_models)), 1)
    elif mode == "fixed":
        rho_h, _rv, esp_fixed, _pz = _build_batch(request)
        _add(esp_fixed[0], int(rho_h.shape[1]), int(rho_h.shape[0]), 1)
    else:  # "stochastic"
        gen_cfg = _genconfig_from_request(request)
        stable_seed = _stable_geometry_seed(gen_cfg)
        n_total = max(1, int(request.n_models))
        if request.n_layers_fixed is not None:
            n_layers_vals = [int(request.n_layers_fixed)]
            n_distinct_layers = 1
        else:
            lo, hi = int(request.n_layers_min), int(request.n_layers_max)
            all_vals = list(range(lo, hi))
            if not all_vals:
                return specs
            n_distinct_layers = len(all_vals)
            # MODAL: 1-2 n_layers do centro da faixa (Q2).
            mid = len(all_vals) // 2
            picks = [all_vals[mid]]
            if max_n_layers >= 2 and len(all_vals) > 1:
                picks.append(all_vals[min(mid + 1, len(all_vals) - 1)])
            n_layers_vals = sorted(set(picks))
        k_req = (
            _TEMPLATE_GEOMETRIES_CAP
            if request.n_geometries is None
            else int(request.n_geometries)
        )
        for n_layers in n_layers_vals:
            group_size = (
                n_total
                if request.n_layers_fixed is not None
                else max(1, n_total // n_distinct_layers)
            )
            if group_size <= 1:
                continue
            k = max(1, min(k_req, group_size // 2))
            templates = _deterministic_templates(gen_cfg, n_layers, k, stable_seed)
            for esp in templates:
                _add(esp, n_layers, group_size, k)

    return specs
