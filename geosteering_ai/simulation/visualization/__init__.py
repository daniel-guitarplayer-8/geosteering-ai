# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/visualization/__init__.py                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Visualização (Sprint 2.8+)             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : matplotlib                                                 ║
# ║  Dependências: matplotlib, numpy                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Utilitários de plotagem para amostras geológicas e tensor H            ║
# ║    produzidos pelo simulador Python. Replica o padrão GridSpec(3,7)      ║
# ║    de `Fortran_Gerador/buildValidamodels.py:571-628`, adaptado para      ║
# ║    consumir objetos :class:`SimulationResult` diretamente.                ║
# ║                                                                           ║
# ║  MÓDULOS                                                                  ║
# ║    • plot_tensor.py     — plot_tensor_profile(result) → GridSpec(3,7)    ║
# ║    • plot_benchmark.py  — plot_benchmark_comparison(results)             ║
# ║                                                                           ║
# ║  CONVENÇÕES GEOLÓGICAS (herdadas de buildValidamodels)                   ║
# ║    • Eixo y invertido (profundidade cresce para baixo)                   ║
# ║    • Resistividade em escala log (semilogx)                              ║
# ║    • ρₕ linha sólida steelblue, ρᵥ linha tracejada darkorange            ║
# ║    • Interfaces como axhline tracejadas pretas                            ║
# ║    • Paleta Re: tons de azul (darkblue → cornflowerblue)                 ║
# ║    • Paleta Im: tons de vermelho (darkred → tomato)                       ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/buildValidamodels.py — layout original              ║
# ║    • geosteering_ai/simulation/forward.py — SimulationResult              ║
# ║    • .claude/commands/geosteering-simulator-python.md (Seção 2.8)       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Utilitários de visualização para o simulador Python.

Este subpacote fornece funções de alto nível para plotar resultados de
simulações EM 1D TIV em padrões visuais idênticos aos utilizados pelo
simulador Fortran de referência (`buildValidamodels.py`).

Example:
    Plot de tensor H completo com perfil de resistividade::

        >>> import numpy as np
        >>> from geosteering_ai.simulation import simulate, SimulationConfig
        >>> from geosteering_ai.simulation.visualization import plot_tensor_profile
        >>> cfg = SimulationConfig()
        >>> result = simulate(
        ...     rho_h=np.array([1.0, 100.0, 1.0]),
        ...     rho_v=np.array([1.0, 200.0, 1.0]),
        ...     esp=np.array([5.0]),
        ...     positions_z=np.linspace(-2, 7, 100),
        ...     cfg=cfg,
        ... )
        >>> fig = plot_tensor_profile(result, title="Modelo 3-camadas TIV")
        >>> fig.savefig("tensor_profile.png", dpi=300)

Note:
    As funções deste módulo **não** chamam ``plt.show()`` — retornam o
    objeto :class:`matplotlib.figure.Figure` para que o caller decida
    entre exibir, salvar ou integrar em subplots.
"""
from geosteering_ai.simulation.visualization.plot_benchmark import (
    plot_benchmark_comparison,
)
from geosteering_ai.simulation.visualization.plot_benchmark_advanced import (
    measure_component_times,
    plot_component_times,
    plot_error_heatmap,
    plot_filter_convergence,
    plot_speedup_curve,
)
from geosteering_ai.simulation.visualization.plot_canonical import (
    plot_all_canonical_models,
    plot_canonical_model,
)
from geosteering_ai.simulation.visualization.plot_geophysical import (
    plot_apparent_resistivity_curves,
    plot_geosignal_response_vs_dip,
    plot_nyquist,
    plot_polar_directivity,
    plot_pseudosection,
    plot_tornado,
)
from geosteering_ai.simulation.visualization.plot_ml import (
    plot_augmentation_preview,
    plot_pinn_loss_decomposition,
    plot_uncertainty_bands,
)
from geosteering_ai.simulation.visualization.plot_physics import (
    plot_anisotropy_ratio_sensitivity,
    plot_attenuation_phase,
    plot_feature_views,
    plot_geosignals,
    plot_sensitivity_kernel,
    plot_skin_depth_heatmap,
)
from geosteering_ai.simulation.visualization.plot_tensor import (
    plot_resistivity_profile,
    plot_tensor_profile,
)

__all__ = [
    # Tensor principal + perfil (Sprint 2.8)
    "plot_tensor_profile",
    "plot_resistivity_profile",
    # Benchmark básico (Sprint 2.8)
    "plot_benchmark_comparison",
    # Modelos canônicos (Sprint 2.9)
    "plot_canonical_model",
    "plot_all_canonical_models",
    # Físicas complementares (Sprint 2.10+)
    "plot_skin_depth_heatmap",
    "plot_attenuation_phase",
    "plot_feature_views",
    "plot_geosignals",
    "plot_sensitivity_kernel",
    # Benchmark avançado (Sprint 2.10+)
    "plot_speedup_curve",
    "plot_filter_convergence",
    "plot_error_heatmap",
    "plot_component_times",
    "measure_component_times",
    # Geofísicas avançadas (Sprint 2.10+)
    "plot_pseudosection",
    "plot_polar_directivity",
    "plot_nyquist",
    "plot_tornado",
    # ML/DL integration (Sprint 2.10+)
    "plot_augmentation_preview",
    "plot_uncertainty_bands",
    # Sprint 3.3.2 — 4 plots LWD/PINN industriais
    "plot_apparent_resistivity_curves",
    "plot_geosignal_response_vs_dip",
    "plot_anisotropy_ratio_sensitivity",
    "plot_pinn_loss_decomposition",
]
