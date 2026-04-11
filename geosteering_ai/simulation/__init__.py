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
# ║    Sprint 1.1 (concluída): extração dos pesos Hankel (.npz) +           ║
# ║                            FilterLoader para Kong 61, Werthmüller 201,  ║
# ║                            Anderson 801.                                 ║
# ║    Próximas Sprints:                                                     ║
# ║      1.2  SimulationConfig dataclass + errata validation                ║
# ║      1.3  Testes de referência (half-space analítico)                   ║
# ║      2.x  Backend Numba (commonarraysMD, commonfactorsMD, hmd/vmd)     ║
# ║      3.x  Backend JAX  (vmap + jit + pmap)                             ║
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
# EXPORTS PÚBLICOS (Sprint 1.1)
# ──────────────────────────────────────────────────────────────────────────────
# Até a Fase 2, apenas o subsistema de filtros está disponível. Outras
# exportações (SimulationConfig, simulate, ...) serão adicionadas quando
# os módulos correspondentes forem implementados.
from geosteering_ai.simulation.filters import FilterLoader, HankelFilter

__all__ = [
    "FilterLoader",
    "HankelFilter",
]

# Versão do subpacote. Sobe conforme Sprints concluídas.
#   0.1.x → Sprint 1.1 (filtros Hankel extraídos)
#   0.2.x → Sprint 1.2 (SimulationConfig + validação)
#   0.3.x → Sprint 1.3 (testes de referência half-space)
#   1.0.0 → Fase 6 concluída (simulador 100% funcional)
__version__ = "0.1.0"
