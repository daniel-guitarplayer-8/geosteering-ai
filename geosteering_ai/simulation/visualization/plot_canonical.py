# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/visualization/plot_canonical.py               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Plot de modelos canônicos (Sprint 2.9+)║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : matplotlib + SimulationConfig                              ║
# ║  Dependências: matplotlib, numpy, geosteering_ai.simulation               ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Wrappers de alto nível que combinam `get_canonical_model` +           ║
# ║    `simulate()` + `plot_tensor_profile()` em uma única chamada. Útil   ║
# ║    para geração rápida de figuras de validação reprodutíveis.           ║
# ║                                                                           ║
# ║  FUNÇÕES                                                                  ║
# ║    • plot_canonical_model(name, ...) → Figure                           ║
# ║      Plota um único modelo (Oklahoma 3, Devine 8, etc.)                  ║
# ║                                                                           ║
# ║    • plot_all_canonical_models(output_dir, ...) → List[Path]            ║
# ║      Plota todos os 7 modelos e salva em ``output_dir``                 ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/buildValidamodels.py:571-637 — fluxo original       ║
# ║    • validation/canonical_models.py — catálogo de modelos               ║
# ║    • visualization/plot_tensor.py — layout GridSpec(3,7)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Plot de modelos canônicos do catálogo de validação.

Combina :func:`geosteering_ai.simulation.validation.get_canonical_model`
com :func:`geosteering_ai.simulation.simulate` e
:func:`plot_tensor_profile` em uma API de alto nível.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.forward import simulate
from geosteering_ai.simulation.validation.canonical_models import (
    CanonicalModel,
    get_all_canonical_models,
    get_canonical_model,
)
from geosteering_ai.simulation.visualization.plot_tensor import plot_tensor_profile

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers privados
# ──────────────────────────────────────────────────────────────────────────────


def _default_positions_for_model(
    model: CanonicalModel, *, n_positions: int, margin_m: float
) -> np.ndarray:
    """Constrói vetor de profundidades cobrindo todo o modelo + margem.

    Para modelos com semi-espaços, espalha as posições desde
    ``min_depth - margin_m`` até ``max_depth + margin_m``.
    """
    z_start = model.min_depth - margin_m
    z_end = model.max_depth + margin_m
    return np.linspace(z_start, z_end, n_positions, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# plot_canonical_model — pipeline modelo + simulate + plot
# ──────────────────────────────────────────────────────────────────────────────


def plot_canonical_model(
    name: str,
    *,
    frequency_hz: float = 20_000.0,
    tr_spacing_m: float = 1.0,
    dip_deg: float = 0.0,
    hankel_filter: str = "werthmuller_201pt",
    n_positions: int = 300,
    margin_m: float = 5.0,
    figsize: tuple[float, float] = (28.0, 12.0),
    parallel: bool = True,
) -> "Figure":
    """Plota um modelo canônico: carrega + simula + plota em 1 chamada.

    Args:
        name: Nome do modelo canônico. Ver
            :func:`list_canonical_models` para opções.
        frequency_hz: Frequência de operação em Hz. Default: 20 kHz.
        tr_spacing_m: Espaçamento transmissor-receptor em metros. Default: 1.0.
        dip_deg: Dip da ferramenta em graus. Default: 0° (vertical).
        hankel_filter: Filtro Hankel (``werthmuller_201pt``, ``kong_61pt``,
            ``anderson_801pt``). Default: ``werthmuller_201pt``.
        n_positions: Número de posições ao longo do poço. Default: 300.
        margin_m: Margem (m) acima e abaixo do modelo. Default: 5.0.
        figsize: Tamanho da figura em polegadas.
        parallel: Se True, usa paralelismo @njit (Sprint 2.9).

    Returns:
        :class:`matplotlib.figure.Figure` com layout GridSpec(3, 7)
        (painel ρ + 9 painéis Re/Im do tensor H).

    Example:
        Plot do modelo Oklahoma 3 em 20 kHz, TR 1 m::

            >>> from geosteering_ai.simulation.visualization import (
            ...     plot_canonical_model
            ... )
            >>> fig = plot_canonical_model("oklahoma_3")
            >>> fig.savefig("oklahoma_3.png", dpi=200, bbox_inches="tight")
    """
    model = get_canonical_model(name)  # type: ignore[arg-type]

    cfg = SimulationConfig(
        frequency_hz=frequency_hz,
        tr_spacing_m=tr_spacing_m,
        backend="numba",
        hankel_filter=hankel_filter,
        parallel=parallel,
    )

    positions_z = _default_positions_for_model(
        model, n_positions=n_positions, margin_m=margin_m
    )

    result = simulate(
        rho_h=model.rho_h,
        rho_v=model.rho_v,
        esp=model.esp,
        positions_z=positions_z,
        dip_deg=dip_deg,
        cfg=cfg,
    )

    fig = plot_tensor_profile(
        result,
        model_name=model.title,
        dip_deg=dip_deg,
        figsize=figsize,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_all_canonical_models — batch de todos os modelos
# ──────────────────────────────────────────────────────────────────────────────


def plot_all_canonical_models(
    output_dir: str | Path,
    *,
    frequency_hz: float = 20_000.0,
    tr_spacing_m: float = 1.0,
    dip_deg: float = 0.0,
    hankel_filter: str = "werthmuller_201pt",
    n_positions: int = 300,
    margin_m: float = 5.0,
    dpi: int = 200,
    close_figs: bool = True,
    parallel: bool = True,
) -> list[Path]:
    """Plota todos os 7 modelos canônicos e salva como PNG.

    Args:
        output_dir: Diretório de saída (criado se não existir).
        frequency_hz: Frequência (Hz). Default: 20 kHz.
        tr_spacing_m: Espaçamento TR (m). Default: 1.0.
        dip_deg: Dip (graus). Default: 0.
        hankel_filter: Filtro Hankel. Default: werthmuller_201pt.
        n_positions: N posições por modelo. Default: 300.
        margin_m: Margem axial (m). Default: 5.0.
        dpi: Resolução do PNG. Default: 200.
        close_figs: Se True, fecha cada figura após salvar (libera memória).
        parallel: Ativa paralelismo Numba (@njit + prange) — Sprint 2.9.

    Returns:
        Lista de Path com os arquivos PNG gerados (um por modelo).

    Example:
        Geração de todos os modelos em /tmp/validation::

            >>> from geosteering_ai.simulation.visualization import (
            ...     plot_all_canonical_models
            ... )
            >>> paths = plot_all_canonical_models("/tmp/validation")
            >>> len(paths)
            7
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib é necessário. Instale via `pip install matplotlib`."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for model in get_all_canonical_models():
        freq_tag = f"f{int(frequency_hz / 1e3)}kHz"
        tr_tag = f"TR{tr_spacing_m:.2f}m".replace(".", "p")
        dip_tag = f"dip{int(dip_deg)}"
        filename = f"canonical_{model.name}_{freq_tag}_{tr_tag}_{dip_tag}.png"
        output_path = output_dir / filename

        logger.info(
            "Plotando %s (%d camadas, ρh ∈ [%.3g, %.3g] Ω·m)",
            model.title,
            model.n_layers,
            model.rho_h_min,
            model.rho_h_max,
        )

        fig = plot_canonical_model(
            model.name,
            frequency_hz=frequency_hz,
            tr_spacing_m=tr_spacing_m,
            dip_deg=dip_deg,
            hankel_filter=hankel_filter,
            n_positions=n_positions,
            margin_m=margin_m,
            parallel=parallel,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)

        if close_figs:
            import matplotlib.pyplot as plt

            plt.close(fig)

    logger.info(
        "Salvos %d plots de modelos canônicos em %s", len(saved_paths), output_dir
    )
    return saved_paths


__all__ = ["plot_canonical_model", "plot_all_canonical_models"]
