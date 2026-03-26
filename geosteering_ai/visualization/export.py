# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/export.py                                          ║
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
# ║    • Exportacao batch de figuras matplotlib em multiplos formatos         ║
# ║    • Suporte a PNG, PDF, SVG (extensivel a outros formatos)              ║
# ║    • Salvamento individual com criacao automatica de diretorios           ║
# ║    • Integracao com pipeline de avaliacao (save_all_figures)              ║
# ║                                                                            ║
# ║  Dependencias: matplotlib (lazy), pathlib                                ║
# ║  Exports: ~2 (export_all_figures, save_figure)                           ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9.7                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C63 batch export)          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Exportacao de figuras — batch export e salvamento individual.

Funcoes utilitarias para exportar figuras matplotlib geradas pelo pipeline
de avaliacao e diagnostico. Suporta multiplos formatos (PNG, PDF, SVG) e
criacao automatica de diretorios de saida.

Fluxo tipico:
    1. Gerar figuras com plot_training_history, plot_error_heatmap, etc.
    2. Coletar figuras em dicionario {nome: fig}
    3. Chamar export_all_figures(figures, output_dir) para batch export

Example:
    >>> import matplotlib.pyplot as plt
    >>> fig1, _ = plt.subplots(); fig2, _ = plt.subplots()
    >>> figures = {"training_curves": fig1, "error_heatmap": fig2}
    >>> from geosteering_ai.visualization.export import export_all_figures
    >>> export_all_figures(figures, "/tmp/plots", formats=("png", "pdf"))

Note:
    Matplotlib importado de forma lazy (suporta ambientes headless).
    Referenciado em:
        - evaluation/metrics.py: exportar relatorio visual completo
        - training/loop.py: salvar diagnostico pos-treinamento
    Ref: docs/ARCHITECTURE_v2.md secao 9.7.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Funcoes de exportacao ---
    "export_all_figures",
    "save_figure",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de exportacao
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_DPI = 300
_VALID_FORMATS = frozenset({"png", "pdf", "svg", "eps", "tiff", "jpg", "jpeg"})


# ──────────────────────────────────────────────────────────────────────
# D2: Funcao utilitaria — salvamento individual de figura
# ──────────────────────────────────────────────────────────────────────
def save_figure(
    fig: object,
    path: str,
    *,
    dpi: int = _DEFAULT_DPI,
    close: bool = True,
) -> None:
    """Salva uma unica figura matplotlib em disco.

    Cria diretorios pais automaticamente se nao existirem. Opcionalmente
    fecha a figura apos salvar para liberar memoria (recomendado em
    pipelines batch que geram muitas figuras).

    Args:
        fig: Objeto matplotlib.figure.Figure a salvar.
            Deve ter metodo ``savefig``.
        path: Caminho completo para o arquivo de saida.
            Extensao determina formato (e.g., "plot.png", "plot.pdf").
            Diretorios pais sao criados automaticamente.
        dpi: Resolucao em DPI (default 300).
            Valores tipicos: 150 (draft), 300 (publicacao), 600 (print).
        close: Se True, fecha a figura apos salvar (default True).
            Recomendado para pipelines batch para evitar memory leak.

    Raises:
        ValueError: Se extensao do arquivo nao for um formato valido.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure(fig, "/tmp/test_plot.png", dpi=150)

    Note:
        Para exportacao batch de multiplas figuras, use ``export_all_figures``.
        Ref: docs/ARCHITECTURE_v2.md secao 9.7.
    """
    # --- Import lazy do matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    # --- Validar formato ---
    path_obj = Path(path)
    suffix = path_obj.suffix.lstrip(".").lower()
    if suffix and suffix not in _VALID_FORMATS:
        msg = (
            f"Formato '{suffix}' nao suportado. "
            f"Formatos validos: {sorted(_VALID_FORMATS)}"
        )
        raise ValueError(msg)

    # --- Criar diretorios pais ---
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # --- Salvar ---
    fig.savefig(str(path_obj), dpi=dpi, bbox_inches="tight")
    logger.info("Figura salva em: %s (dpi=%d)", path_obj, dpi)

    # --- Fechar se solicitado ---
    if close:
        plt.close(fig)
        logger.debug("Figura fechada apos salvar: %s", path_obj)


# ──────────────────────────────────────────────────────────────────────
# D2: Funcao principal — batch export de multiplas figuras
# ──────────────────────────────────────────────────────────────────────
def export_all_figures(
    figures: Dict[str, object],
    output_dir: str,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = _DEFAULT_DPI,
) -> int:
    """Exporta batch de figuras matplotlib em multiplos formatos.

    Para cada entrada {nome: figura} no dicionario, salva o arquivo
    ``output_dir/nome.{fmt}`` para cada formato solicitado. Cria o
    diretorio de saida se nao existir. Fecha cada figura apos salvar
    para liberar memoria.

    Fluxo:
    ┌─────────────────────────────────────────────────────────────────┐
    │  figures dict ──→ for name, fig ──→ for fmt ──→ save_figure()  │
    │                                                                 │
    │  output_dir/                                                    │
    │    ├── training_curves.png                                      │
    │    ├── training_curves.pdf                                      │
    │    ├── error_heatmap.png                                        │
    │    └── error_heatmap.pdf                                        │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        figures: Dicionario {nome: matplotlib.figure.Figure}.
            Chaves sao usadas como nomes de arquivo (sem extensao).
            Valores sao objetos Figure do matplotlib.
        output_dir: Diretorio de saida para todas as figuras.
            Criado automaticamente se nao existir.
        formats: Tupla de formatos de saida (default ("png",)).
            Valores validos: "png", "pdf", "svg", "eps", "tiff", "jpg".
        dpi: Resolucao em DPI para todas as figuras (default 300).

    Returns:
        Numero total de arquivos salvos com sucesso.

    Raises:
        ValueError: Se figures estiver vazio.
        ValueError: Se algum formato nao for valido.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig1, _ = plt.subplots(); fig2, _ = plt.subplots()
        >>> figs = {"curves": fig1, "errors": fig2}
        >>> n_saved = export_all_figures(figs, "/tmp/plots", formats=("png", "pdf"))
        >>> n_saved
        4

    Note:
        Figuras sao fechadas apos salvar (close=True por padrao).
        Ref: docs/ARCHITECTURE_v2.md secao 9.7.
    """
    # --- Validacao de entrada ---
    if not figures:
        msg = "Dicionario figures esta vazio. Fornecer pelo menos uma figura."
        raise ValueError(msg)

    # --- Validar formatos ---
    invalid_formats = [f for f in formats if f.lower() not in _VALID_FORMATS]
    if invalid_formats:
        msg = (
            f"Formatos invalidos: {invalid_formats}. "
            f"Formatos validos: {sorted(_VALID_FORMATS)}"
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

    # --- Criar diretorio de saida ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "export_all_figures: %d figuras x %d formatos → %s",
        len(figures), len(formats), output_path,
    )

    # --- Exportar cada figura em cada formato ---
    n_saved = 0
    for name, fig in figures.items():
        for fmt in formats:
            fmt_lower = fmt.lower()
            file_path = output_path / f"{name}.{fmt_lower}"
            try:
                # Nao fechar ate todos os formatos dessa figura serem salvos
                fig.savefig(str(file_path), dpi=dpi, bbox_inches="tight")
                n_saved += 1
                logger.info("Exportado: %s", file_path)
            except Exception:
                logger.exception("Erro ao exportar %s", file_path)

        # D7: Fechar figura apos todos os formatos salvos
        plt.close(fig)

    logger.info(
        "export_all_figures concluido: %d/%d arquivos salvos em %s",
        n_saved, len(figures) * len(formats), output_path,
    )

    return n_saved
