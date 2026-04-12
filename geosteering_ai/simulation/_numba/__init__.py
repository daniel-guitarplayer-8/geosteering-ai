# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_numba/__init__.py                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend Numba CPU                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 2.1)                                   ║
# ║  Status      : Em construção (apenas propagation entregue)                ║
# ║  Framework   : NumPy 2.x + Numba 0.60+ (dual-mode)                        ║
# ║  Dependências: numpy (obrigatório), numba (opcional, speedup JIT)        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Backend Numba do simulador Python otimizado. Implementa os kernels    ║
# ║    hot path do forward EM 1D TIV equivalentes ao simulador Fortran       ║
# ║    PerfilaAnisoOmp.f08, com decoradores @njit para compilação LLVM       ║
# ║    paralelizável via prange.                                              ║
# ║                                                                           ║
# ║  DUAL-MODE (Numba opcional)                                               ║
# ║    O subpacote foi desenhado para funcionar **com ou sem Numba**. Se     ║
# ║    Numba estiver disponível, o decorador `@njit` compila as funções      ║
# ║    via LLVM (speedup 10–100×). Se não estiver, um no-op decorator é     ║
# ║    usado e as funções rodam em NumPy puro (correção preservada,        ║
# ║    performance reduzida). Isso permite:                                 ║
# ║      • Testes unitários em qualquer ambiente (CI mínimo)                ║
# ║      • Debugging com breakpoints nativos Python                         ║
# ║      • Fallback gracioso em ambientes com conflitos de dependência      ║
# ║                                                                           ║
# ║  MÓDULOS (Sprints 2.1–2.4)                                               ║
# ║    ┌────────────────────────┬────────────┬──────────────────────────┐   ║
# ║    │  Módulo                 │  Sprint    │  Status                  │   ║
# ║    ├────────────────────────┼────────────┼──────────────────────────┤   ║
# ║    │  propagation.py         │  2.1 ★     │  ✅ Entregue             │   ║
# ║    │  dipoles.py             │  2.2 ★     │  ✅ Entregue             │   ║
# ║    │  hankel.py              │  2.3       │  ⬜ Pendente             │   ║
# ║    │  rotation.py            │  2.3       │  ⬜ Pendente             │   ║
# ║    │  geometry.py            │  2.3       │  ⬜ Pendente             │   ║
# ║    │  kernel.py              │  2.4       │  ⬜ Pendente             │   ║
# ║    └────────────────────────┴────────────┴──────────────────────────┘   ║
# ║                                                                           ║
# ║  API (interna)                                                            ║
# ║    Este subpacote é **interno** ao simulador Python. Consumidores       ║
# ║    externos devem usar `geosteering_ai.simulation.simulate(cfg)` (a    ║
# ║    ser adicionado na Sprint 2.5), não importar diretamente daqui.      ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/utils.f08 (commonarraysMD, commonfactorsMD)       ║
# ║    • Fortran_Gerador/magneticdipoles.f08 (hmd_TIV, vmd)                ║
# ║    • docs/reference/plano_simulador_python_jax_numba.md               ║
# ║    • .claude/commands/geosteering-simulator-python.md                  ║
# ║    • Lam, S.K. et al. (2015). Numba: a LLVM-based Python JIT compiler. ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Backend Numba CPU do simulador Python.

Este subpacote **interno** implementa os kernels computacionais do forward
EM 1D TIV otimizados com `@njit` do Numba. A API pública é exposta via
`geosteering_ai.simulation.simulate()` (Sprint 2.5 em diante). Importar
diretamente daqui é permitido apenas em testes e benchmarks.

Example:
    Uso em teste unitário (Sprint 2.1)::

        >>> from geosteering_ai.simulation._numba.propagation import (
        ...     common_arrays, common_factors,
        ... )
        >>> import numpy as np
        >>> n, npt = 1, 201
        >>> h = np.array([0.0], dtype=np.float64)
        >>> eta = np.array([[1.0, 1.0]], dtype=np.float64)
        >>> krJ0J1 = np.linspace(0.001, 50.0, npt)
        >>> zeta = 1j * 2 * np.pi * 20000.0 * 4e-7 * np.pi
        >>> outs = common_arrays(n, npt, 1.0, krJ0J1, zeta, h, eta)
        >>> len(outs)
        9

Note:
    O decorador `@njit` é importado de forma dual-mode — funciona como
    no-op se Numba não estiver instalado. Ver docstring de `propagation.py`
    para detalhes.
"""
from __future__ import annotations

from geosteering_ai.simulation._numba.dipoles import hmd_tiv, vmd
from geosteering_ai.simulation._numba.geometry import (
    find_layers_tr,
    layer_at_depth,
    sanitize_profile,
)
from geosteering_ai.simulation._numba.hankel import (
    integrate_j0,
    integrate_j0_j1,
    integrate_j1,
    prepare_kr,
)
from geosteering_ai.simulation._numba.kernel import compute_zrho, fields_in_freqs
from geosteering_ai.simulation._numba.propagation import (
    HAS_NUMBA,
    common_arrays,
    common_factors,
)
from geosteering_ai.simulation._numba.rotation import build_rotation_matrix, rotate_tensor

__all__ = [
    "HAS_NUMBA",
    # Sprint 2.1 — propagation
    "common_arrays",
    "common_factors",
    # Sprint 2.2 — dipoles
    "hmd_tiv",
    "vmd",
    # Sprint 2.3 — geometry
    "sanitize_profile",
    "find_layers_tr",
    "layer_at_depth",
    # Sprint 2.3 — rotation
    "build_rotation_matrix",
    "rotate_tensor",
    # Sprint 2.3 — hankel helpers
    "prepare_kr",
    "integrate_j0",
    "integrate_j1",
    "integrate_j0_j1",
    # Sprint 2.4 — kernel orchestrator
    "fields_in_freqs",
    "compute_zrho",
]
