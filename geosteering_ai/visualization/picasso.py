# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/picasso.py                                          ║
# ║  Bloco: 9 — Visualization                                                ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Mapa Picasso DOD (Depth of Detection) como funcao do contraste       ║
# ║    • Contour plot 2D: DOD(Rt1, Rt2) — profundidade de deteccao           ║
# ║    • Visualiza resolucao da ferramenta LWD por cenario de resistividade   ║
# ║    • Suporte a contour labels, diagonal R1=R2, salvamento em disco       ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), numpy, matplotlib (lazy)      ║
# ║  Exports: ~1 (plot_picasso_dod)                                           ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9.2                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C26)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Mapa Picasso DOD — Depth of Detection por contraste de resistividade.

Gera contour plot 2D mostrando a profundidade de deteccao (DOD) como funcao
do contraste entre resistividades Rt1 (camada adjacente) e Rt2 (camada alvo).
Ferramenta essencial para avaliacao da resolucao vertical da inversao LWD.

A diagonal R1=R2 e plotada como referencia — nela, DOD = 0 (sem contraste).
Contour labels indicam os valores de DOD em metros.

Example:
    >>> from geosteering_ai.visualization import plot_picasso_dod
    >>> import numpy as np
    >>> dod = np.random.rand(50, 50) * 5.0
    >>> rt1 = np.logspace(-1, 3, 50)
    >>> rt2 = np.logspace(-1, 3, 50)
    >>> plot_picasso_dod(dod, rt1, rt2, title="DOD 20kHz", show=False)

Note:
    Matplotlib importado de forma lazy (suporta ambientes headless).
    Referenciado em:
        - evaluation/metrics.py: calculo de DOD
        - notebooks: analise Picasso interativa
    Ref: docs/ARCHITECTURE_v2.md secao 9.2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Funcao principal ---
    "plot_picasso_dod",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de visualizacao
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_DPI = 300
_FIGSIZE = (8, 7)
_CONTOUR_LEVELS = 12  # numero de niveis de contorno
_CMAP = "viridis"     # colormap padrao para DOD


# ──────────────────────────────────────────────────────────────────────
# D2: Funcao principal — mapa Picasso DOD
# ──────────────────────────────────────────────────────────────────────
def plot_picasso_dod(
    dod_map: np.ndarray,
    rt1_range: np.ndarray,
    rt2_range: np.ndarray,
    *,
    config: Optional[PipelineConfig] = None,
    title: str = "Picasso DOD",
    save_path: Optional[str] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> None:
    """Plota mapa Picasso DOD (Depth of Detection) por contraste.

    Contour plot 2D: DOD como funcao de (Rt1, Rt2), onde Rt1 e a
    resistividade da camada adjacente e Rt2 a da camada alvo. A diagonal
    R1=R2 e plotada como referencia (sem contraste = DOD zero).

    Args:
        dod_map: Array (M, N) com valores de DOD em metros.
            M corresponde a len(rt1_range), N a len(rt2_range).
        rt1_range: Array (M,) com resistividades da camada 1 (ohm.m).
            Tipicamente em escala logaritmica (logspace).
        rt2_range: Array (N,) com resistividades da camada 2 (ohm.m).
            Tipicamente em escala logaritmica (logspace).
        config: PipelineConfig opcional para metadados adicionais.
            Se fornecido, frequencia e spacing sao incluidos no titulo.
        title: Titulo do plot (default "Picasso DOD").
        save_path: Caminho completo para salvar a figura (None = nao salva).
            Diretorio pai criado automaticamente se nao existir.
        dpi: Resolucao em DPI para salvamento (default 300).
        show: Se True, chama plt.show() ao final (default True).

    Raises:
        ValueError: Se dod_map.shape != (len(rt1_range), len(rt2_range)).
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import numpy as np
        >>> dod = np.random.rand(50, 50) * 5.0
        >>> rt1 = np.logspace(-1, 3, 50)
        >>> rt2 = np.logspace(-1, 3, 50)
        >>> plot_picasso_dod(dod, rt1, rt2, show=False)

    Note:
        A diagonal R1=R2 indica ausencia de contraste (DOD teorico = 0).
        Ref: docs/ARCHITECTURE_v2.md secao 9.2.
    """
    # --- Validacao de shapes ---
    expected_shape = (len(rt1_range), len(rt2_range))
    if dod_map.shape != expected_shape:
        msg = (
            f"Shape mismatch: dod_map={dod_map.shape}, "
            f"esperado ({len(rt1_range)}, {len(rt2_range)})."
        )
        raise ValueError(msg)

    # --- Import lazy do matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    logger.info(
        "Gerando Picasso DOD: shape=%s, Rt1=[%.2f, %.2f], Rt2=[%.2f, %.2f]",
        dod_map.shape,
        rt1_range[0], rt1_range[-1],
        rt2_range[0], rt2_range[-1],
    )

    # --- Criar meshgrid para contour ---
    rt2_grid, rt1_grid = np.meshgrid(rt2_range, rt1_range)

    # --- Criar figura ---
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    # --- Contour filled + linhas ---
    cf = ax.contourf(
        rt2_grid, rt1_grid, dod_map,
        levels=_CONTOUR_LEVELS,
        cmap=_CMAP,
    )
    cs = ax.contour(
        rt2_grid, rt1_grid, dod_map,
        levels=_CONTOUR_LEVELS,
        colors="white",
        linewidths=0.5,
        alpha=0.6,
    )
    # Contour labels com fundo branco para legibilidade
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    # --- Colorbar ---
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("DOD [m]", fontsize=10)

    # --- Diagonal R1=R2 (referencia: sem contraste) ---
    diag_min = max(rt1_range[0], rt2_range[0])
    diag_max = min(rt1_range[-1], rt2_range[-1])
    ax.plot(
        [diag_min, diag_max], [diag_min, diag_max],
        color="red", linewidth=1.5, linestyle="--",
        label=r"$R_{t1} = R_{t2}$",
    )

    # --- Escala logaritmica ---
    ax.set_xscale("log")
    ax.set_yscale("log")

    # --- Labels e titulo ---
    ax.set_xlabel(r"$R_{t2}$ [$\Omega\cdot$m]", fontsize=11)
    ax.set_ylabel(r"$R_{t1}$ [$\Omega\cdot$m]", fontsize=11)

    # Titulo com metadados opcionais da config
    full_title = title
    if config is not None:
        full_title = (
            f"{title}\n"
            f"f = {config.frequency_hz:.0f} Hz, "
            f"L = {config.spacing_meters:.2f} m"
        )
    ax.set_title(full_title, fontsize=12, fontweight="bold")

    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_obj, dpi=dpi, bbox_inches="tight")
        logger.info("Picasso DOD salvo em: %s", save_path_obj)

    # --- Exibir se solicitado ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_picasso_dod concluido")
