# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/__init__.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend JAX (Sprint 3.1)               ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Em construção (Sprint 3.1 — fundação CPU)                 ║
# ║  Framework   : JAX 0.4.30+                                                ║
# ║  Dependências: jax, jaxlib (opcional — dual-mode)                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Backend JAX (CPU/GPU/TPU) do simulador Python. Na Sprint 3.1          ║
# ║    implementa os módulos fundamentais (hankel, rotation) que não têm     ║
# ║    recursões complexas. A paridade numérica com o backend Numba é        ║
# ║    validada em tolerância < 1e-12 (float64) via                          ║
# ║    `validation/compare_backends.py`.                                      ║
# ║                                                                           ║
# ║  DUAL-MODE                                                                ║
# ║    Exporta `HAS_JAX: Final[bool]`. Quando JAX não está instalado,       ║
# ║    `HAS_JAX=False` e as funções ``integrate_j0/j1`` levantam           ║
# ║    :class:`ImportError` ao serem chamadas — diferente do backend         ║
# ║    Numba (que tem fallback no-op), JAX é inerentemente substituto        ║
# ║    do Numba (não complementar). Se o usuário pediu JAX, deve instalar.   ║
# ║                                                                           ║
# ║  float64 OBRIGATÓRIO                                                      ║
# ║    Chamamos `jax.config.update("jax_enable_x64", True)` no import do     ║
# ║    submódulo para garantir paridade com Numba. JAX default é float32    ║
# ║    para compatibilidade com TPU, mas o simulador EM exige complex128.   ║
# ║                                                                           ║
# ║  MÓDULOS (Sprint 3.1)                                                    ║
# ║    • hankel.py    — integrate_j0/j1 via jnp.einsum + @jax.jit           ║
# ║    • rotation.py  — build_rotation_matrix + rotate_tensor diferenciáveis║
# ║                                                                           ║
# ║  MÓDULOS FUTUROS (Sprints 3.2-3.4)                                      ║
# ║    • propagation.py — common_arrays + common_factors com jax.lax.scan   ║
# ║    • dipoles.py     — hmd_tiv + vmd com @jit(static_argnames=...)       ║
# ║    • kernel.py      — fields_in_freqs_jax (orquestrador)                 ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • docs/reference/plano_simulador_python_jax_numba.md §4-6            ║
# ║    • .claude/commands/geosteering-simulator-python.md (Seção 3.1)       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Backend JAX do simulador Python — fundação CPU (Sprint 3.1).

Este subpacote implementa as funções fundamentais do simulador forward
EM 1D TIV usando JAX como backend numérico. JAX proporciona:

  - **Auto-diferenciação** via ``jax.grad`` / ``jax.jacfwd`` — essencial
    para treino de PINNs com gradientes ∂H/∂ρ.
  - **Aceleração XLA** em CPU/GPU/TPU sem reescrita de código.
  - **Vetorização automática** via ``jax.vmap`` — substitui loops de
    posições/ângulos por operações matriciais contíguas (sem GIL).

Example:
    Carregamento do filtro Hankel e cálculo de uma integral J₀::

        >>> import jax.numpy as jnp
        >>> from geosteering_ai.simulation.filters import FilterLoader
        >>> from geosteering_ai.simulation._jax import integrate_j0
        >>> filt = FilterLoader().load("werthmuller_201pt")
        >>> # Valores da função a integrar em cada kr
        >>> values = jnp.ones(filt.npt, dtype=jnp.complex128)
        >>> result = integrate_j0(values, jnp.asarray(filt.weights_j0))

Note:
    Paridade com Numba é de < 1e-12 em float64 para todos os módulos
    da Sprint 3.1 (hankel, rotation). Diferenças residuais vêm do
    reordenamento XLA das operações de ponto flutuante.
"""
from typing import Final

# ──────────────────────────────────────────────────────────────────────────────
# Detecção dual-mode
# ──────────────────────────────────────────────────────────────────────────────
try:
    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # noqa: F401

    # Habilita float64 globalmente — CRÍTICO para paridade com Numba.
    # JAX default é float32 para TPU; o simulador EM requer complex128.
    jax.config.update("jax_enable_x64", True)

    HAS_JAX: Final[bool] = True
except ImportError:
    HAS_JAX: Final[bool] = False  # type: ignore[no-redef]


# ──────────────────────────────────────────────────────────────────────────────
# Re-exports públicos
# ──────────────────────────────────────────────────────────────────────────────
if HAS_JAX:
    from geosteering_ai.simulation._jax.dipoles_native import (
        IMPLEMENTATION_STATUS,
        _hmd_tiv_native_jax_unified,
        _vmd_native_jax_unified,
        decoupling_factors_jax,
        native_dipoles_full_jax_unified,
    )
    from geosteering_ai.simulation._jax.dipoles_unified import (
        _hmd_tiv_propagation_unified,
        _vmd_propagation_unified,
    )
    from geosteering_ai.simulation._jax.forward_pure import (
        clear_unified_jit_cache,
        count_compiled_xla_programs,
    )
    from geosteering_ai.simulation._jax.geometry_jax import (
        find_layers_tr_jax,
        find_layers_tr_jax_vmap,
    )
    from geosteering_ai.simulation._jax.hankel import (
        integrate_j0,
        integrate_j0_j1,
        integrate_j1,
    )
    from geosteering_ai.simulation._jax.rotation import (
        build_rotation_matrix,
        rotate_tensor,
    )

    __all__ = [
        "HAS_JAX",
        "integrate_j0",
        "integrate_j1",
        "integrate_j0_j1",
        "build_rotation_matrix",
        "rotate_tensor",
        "decoupling_factors_jax",
        "IMPLEMENTATION_STATUS",
        # Sprint 10 (PR #23 Phase 1 + PR #24-part1 Phase 2)
        "_hmd_tiv_propagation_unified",
        "_vmd_propagation_unified",
        # Sprint 10 Phase 2 final (PR #24-part2): wrappers unified cabeados
        "_hmd_tiv_native_jax_unified",
        "_vmd_native_jax_unified",
        "native_dipoles_full_jax_unified",
        "count_compiled_xla_programs",
        "clear_unified_jit_cache",
        # Sprint 12 (PR #25): geometria tracer-safe para vmap real multi-TR/multi-ang
        "find_layers_tr_jax",
        "find_layers_tr_jax_vmap",
    ]
else:
    __all__ = ["HAS_JAX"]
