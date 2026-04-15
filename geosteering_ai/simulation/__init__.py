# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/__init__.py                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python Otimizado (backends JAX + Numba)          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 1.1)                                    ║
# ║  Status      : Em construção (Fase 1 — Foundations)                      ║
# ║  Framework   : NumPy 2.x + Numba 0.60+ + JAX 0.4.30+                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Subpacote que contém o simulador Python otimizado de propagação       ║
# ║    eletromagnética 1D em meios TIV (Transversely Isotropic Vertical),    ║
# ║    equivalente matemático do simulador Fortran PerfilaAnisoOmp (tatu.x). ║
# ║                                                                           ║
# ║    Objetivo de performance:                                               ║
# ║      • Numba CPU:  ≥ 40.000 modelos/hora (paridade Fortran)              ║
# ║      • JAX CPU:    ≥ 40.000 modelos/hora (via JIT)                      ║
# ║      • JAX GPU T4: ≥ 200.000 modelos/hora                               ║
# ║      • JAX GPU A100:≥ 500.000 modelos/hora                              ║
# ║                                                                           ║
# ║  ARQUITETURA                                                              ║
# ║    ┌─────────────────────────────────────────────────────────────────┐   ║
# ║    │  geosteering_ai/simulation/                                     │   ║
# ║    │  ├── __init__.py          ← este arquivo (fachada pública)     │   ║
# ║    │  ├── config.py            ← SimulationConfig dataclass          │   ║
# ║    │  ├── forward.py           ← API principal simulate() [PENDENTE] │   ║
# ║    │  ├── _jax/                ← backend JAX      [PENDENTE Fase 3] │   ║
# ║    │  ├── _numba/              ← backend Numba    [PENDENTE Fase 2] │   ║
# ║    │  ├── filters/             ← pesos Hankel .npz + loader (★)     │   ║
# ║    │  ├── geometry.py          ← rotação, dip     [PENDENTE Fase 2] │   ║
# ║    │  ├── postprocess.py       ← FV, GS, tensor   [PENDENTE Fase 2] │   ║
# ║    │  ├── validation/          ← testes contra Fortran/empymod     │   ║
# ║    │  └── benchmarks/          ← bench_forward, bench_scaling       │   ║
# ║    └─────────────────────────────────────────────────────────────────┘   ║
# ║    (★ = entregue na Sprint 1.1)                                           ║
# ║                                                                           ║
# ║  API PÚBLICA (alvo, após Fases 1-3)                                      ║
# ║    from geosteering_ai.simulation import (                               ║
# ║        SimulationConfig,    # configuração + validação de errata         ║
# ║        simulate,            # forward dispatch (numba | jax)             ║
# ║        FilterLoader,        # acesso aos pesos Hankel (disponível)      ║
# ║    )                                                                      ║
# ║                                                                           ║
# ║  ESTADO ATUAL                                                             ║
# ║    Sprints concluídas:                                                   ║
# ║      1.1 ✅ Filtros Hankel (.npz) + FilterLoader                         ║
# ║      1.2 ✅ SimulationConfig + errata + 9 grupos                         ║
# ║      1.3 ✅ Half-space analítico (5 funções NumPy puras)                ║
# ║      2.1 ✅ Backend Numba — commonarraysMD + commonfactorsMD            ║
# ║      2.2 ✅ Backend Numba — hmd_tiv + vmd + I/O + F6/F7 (opt-in)       ║
# ║                                                                           ║
# ║    Próximas Sprints:                                                     ║
# ║      2.3  Hankel quadrature + rotation RtHR + geometry                  ║
# ║      2.4  Kernel orchestrator (fatia por posição, multi-TR loop)       ║
# ║      2.5  API pública simulate(cfg) + dispatcher backend                ║
# ║      2.6  Validação numérica vs half_space.py (gate < 1e-10)           ║
# ║      2.7  Benchmark ≥ 40 000 mod/h (gate final Fase 2)                 ║
# ║      3.x  Backend JAX (vmap + jit + pmap)                               ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • docs/reference/plano_simulador_python_jax_numba.md                 ║
# ║    • docs/reference/documentacao_simulador_fortran.md                   ║
# ║    • .claude/commands/geosteering-simulator-python.md  (sub skill)     ║
# ║    • .claude/commands/geosteering-simulator-fortran.md (sub skill)     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Subpacote do simulador Python otimizado do Geosteering AI v2.0.

Este subpacote implementa (ou implementará, ao longo das Fases 0–7 do plano)
um simulador de propagação eletromagnética 1D em meios TIV completamente em
Python, com dois backends intercambiáveis:

  • **Numba** (CPU, `@njit parallel=True`) — paridade de performance com
    Fortran/OpenMP, usado como backend padrão para validação e treino CPU.
  • **JAX** (CPU/GPU/TPU, `@jit` + `vmap` + `pmap`) — backend de alta
    performance para treino em GPU (T4/A100) e geração de datasets massivos.

O simulador é matematicamente equivalente ao `PerfilaAnisoOmp.f08` (v10.0)
e produz, bit-a-bit (float64) ou com tolerância < 1e-10 (float32), os mesmos
campos de tensor magnético H de 9 componentes (Hxx, Hxy, ..., Hzz) para
qualquer geometria de poço (dip, azimute, afastamento), frequência
(100 Hz – 1 MHz) e perfil de resistividade TIV (ρh, ρv).

Estado atual (2026-04-11):
    Apenas a **Sprint 1.1** está entregue:
      • Pesos Hankel extraídos em `.npz` (Kong 61, Werthmüller 201,
        Anderson 801).
      • `FilterLoader` para leitura dos pesos em tempo de execução.

    Os módulos `forward`, `_jax/`, `_numba/`, `geometry.py` e `postprocess.py`
    **ainda não existem** — serão criados nas Sprints 1.2 a 3.x conforme
    plano em `docs/reference/plano_simulador_python_jax_numba.md`.

Example:
    Uso corrente (Sprint 1.1):

        >>> from geosteering_ai.simulation.filters import FilterLoader
        >>> loader = FilterLoader()
        >>> filt = loader.load("werthmuller_201pt")
        >>> filt.abscissas.shape
        (201,)
        >>> filt.weights_j0.shape
        (201,)

    Uso alvo (Fases 2-3, ainda não disponível):

        >>> from geosteering_ai.simulation import SimulationConfig, simulate
        >>> cfg = SimulationConfig(
        ...     frequency_hz=20000.0,
        ...     rho_h=[100.0, 1.0, 100.0],
        ...     rho_v=[100.0, 2.0, 100.0],
        ...     thicknesses=[0.0, 5.0, 0.0],
        ...     positions_m=np.linspace(-10.0, 10.0, 600),
        ...     backend="numba",
        ... )
        >>> H_tensor = simulate(cfg)  # shape: (600, 9) complex128

Note:
    Enquanto o simulador Python não estiver completo (Fase 6), o backend
    padrão do `PipelineConfig` permanece `fortran_f2py`. A migração para
    `backend='jax'` será avaliada na Fase 3 apenas se a paridade numérica
    e de performance for atingida.
"""
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# EXPORTS PÚBLICOS (Sprints 1.1, 1.2, 1.3)
# ──────────────────────────────────────────────────────────────────────────────
# Estado de exposição pública:
#   Sprint 1.1  →  FilterLoader, HankelFilter           (entregues)
#   Sprint 1.2  →  SimulationConfig                      (entregue)
#   Sprint 1.3  →  validation.half_space (submódulo)    (entregue)
#   Fase 2+     →  simulate, _numba, _jax, forward      (pendente)
from geosteering_ai.simulation._jacobian import (
    JacobianResult,
    compute_jacobian,
    compute_jacobian_fd_numba,
    compute_jacobian_jax,
)
from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.filters import FilterLoader, HankelFilter
from geosteering_ai.simulation.forward import SimulationResult, simulate
from geosteering_ai.simulation.multi_forward import (
    MultiSimulationResult,
    simulate_multi,
)

# Sprint 11-JAX (v1.5.0-alpha): simulate_multi_jax — multi-TR/angle/freq em JAX.
# Import lazy: se JAX não estiver instalado, os símbolos não são exportados
# (mantém compatibilidade com ambientes Numba-only).
try:
    from geosteering_ai.simulation._jax.multi_forward import (
        MultiSimulationResultJAX,
        simulate_multi_jax,
    )

    _HAS_JAX_MULTI = True
except ImportError:
    _HAS_JAX_MULTI = False

__all__ = [
    "FilterLoader",
    "HankelFilter",
    "JacobianResult",
    "MultiSimulationResult",
    "SimulationConfig",
    "SimulationResult",
    "compute_jacobian",
    "compute_jacobian_fd_numba",
    "compute_jacobian_jax",
    "simulate",
    "simulate_multi",
]
if _HAS_JAX_MULTI:
    __all__ += ["MultiSimulationResultJAX", "simulate_multi_jax"]

# Versão do subpacote. Sobe conforme Sprints concluídas.
#   0.1.x → Sprint 1.1 (filtros Hankel extraídos)
#   0.2.x → Sprint 1.2 (SimulationConfig + validação)
#   0.3.x → Sprint 1.3 (testes de referência half-space)
#   0.4.x → Sprint 2.1 (backend Numba propagation)
#   0.5.x → Sprint 2.2 (Numba dipoles + I/O + F6/F7)
#   0.6.x → Sprints 2.3 + 2.4 (geometry + rotation + hankel + kernel)
#   0.7.x → Sprints 2.5 + 2.6 (forward API + validação analítica)
#   0.8.x → Sprint 2.7 (benchmark ≥ 40k mod/h — gate final Fase 2)
#   1.0.x → Sprints 2.8-2.10 (prange + cache) + 3.1-3.3 (JAX hybrid)
#   1.1.x → Sprint 3.3.4 (ETAPAS 3+6 JAX native end-to-end, PR #12)
#   1.2.0 → Sprint 5.1+5.2 (Jacobiano FD Numba + JAX jacfwd) + 4.x TIV analítico (PR #13)
#   1.4.0 → Sprint 11 (PR #15: multi-TR + multi-ângulo Numba nativo + F6/F7 wiring)
#   1.4.1 → PR #21 (fix convenção T/R Numba+JAX + teste regressão dip≠0°)
#   1.5.0-alpha → PR #23 (Sprint 10 Phase 1 fori_loop + Sprint 11-JAX wrapper)
#   1.5.0-beta1 → PR #24-part1 (Sprint 10 Phase 2: _vmd_propagation_unified
#                 + SimulationConfig.jax_strategy flag — opt-in "unified")
#   1.5.0 → Sprint 10 Phase 2 final (wiring forward_pure + vmap multi) [PENDENTE]
#   2.0.0 → Fase 6 concluída (backend dispatch em PipelineConfig)
__version__ = "1.5.0b1"  # PR #24-part1: VMD unified + jax_strategy flag
