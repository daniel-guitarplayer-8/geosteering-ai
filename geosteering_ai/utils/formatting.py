# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: utils/formatting.py                                               ║
# ║  Bloco: 5 — Utilitarios                                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Formatacao de numeros (separador de milhar, K/M/B, bytes)            ║
# ║    • Banners e headers para logging (print_header, print_section)         ║
# ║    • Coloracao semantica de valores de FLAGS                              ║
# ║    • Bloco formatado de FLAGS com alinhamento dot-fill                    ║
# ║                                                                            ║
# ║  Dependencias: logging (stdlib), utils/logger.py (C)                      ║
# ║  Exports: ~7 simbolos — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9 (utils/)                           ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from geosteering_ai.utils.logger import C

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FORMATACAO DE NUMEROS
# ════════════════════════════════════════════════════════════════════════════
# Tres modos de formatacao numerica: completa (com separador de milhar),
# compacta (K/M/B), e bytes (B/KB/MB/GB). Todas retornam strings
# prontas para logging — nenhuma faz print direto (D9).
# ──────────────────────────────────────────────────────────────────────────


def format_number(n: Union[int, float]) -> str:
    """Formata numero com separador de milhar.

    Numeros inteiros recebem separador de milhar. Floats sao
    formatados com 2 casas decimais e separador de milhar.

    Args:
        n: Numero a ser formatado.

    Returns:
        str: Numero formatado. Ex: "1,234,567" ou "1,234.50".

    Example:
        >>> from geosteering_ai.utils.formatting import format_number
        >>> format_number(1234567)
        '1,234,567'
        >>> format_number(1234.5)
        '1,234.50'

    Note:
        Referenciado em: training/loop.py (contagem de parametros).
    """
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:,.2f}"


def format_compact(n: Union[int, float]) -> str:
    """Formata numero em notacao compacta K/M/B.

    Ideal para contagem de parametros de modelos e volumes de dados.
    Valores abaixo de 1000 sao retornados como inteiro.

    Args:
        n: Numero a ser compactado.

    Returns:
        str: Numero compactado. Ex: "1.5K", "2.5M", "1.2B".

    Example:
        >>> from geosteering_ai.utils.formatting import format_compact
        >>> format_compact(1500)
        '1.5K'
        >>> format_compact(2500000)
        '2.5M'

    Note:
        Referenciado em: models/registry.py (resumo de arquiteturas).
    """
    abs_n = abs(n)
    sign = "-" if n < 0 else ""
    if abs_n >= 1_000_000_000:
        # ── Bilhoes: modelos gigantes (improvavel neste pipeline) ─────
        return f"{sign}{abs_n / 1_000_000_000:.1f}B"
    elif abs_n >= 1_000_000:
        # ── Milhoes: parametros tipicos (ResNet-18: ~11M) ─────────────
        return f"{sign}{abs_n / 1_000_000:.1f}M"
    elif abs_n >= 1_000:
        # ── Milhares: parametros de camadas individuais ───────────────
        return f"{sign}{abs_n / 1_000:.1f}K"
    else:
        # ── Unidades: valores pequenos (features, channels) ───────────
        return f"{sign}{abs_n:.0f}"


def format_bytes(n_bytes: Union[int, float]) -> str:
    """Formata quantidade de bytes em unidade legivel.

    Usa escala binaria (1024). Ideal para tamanho de modelos,
    datasets, e uso de memoria.

    Args:
        n_bytes: Quantidade de bytes.

    Returns:
        str: Bytes formatados. Ex: "1.50 KB", "256.00 MB", "1.20 GB".

    Example:
        >>> from geosteering_ai.utils.formatting import format_bytes
        >>> format_bytes(1536)
        '1.50 KB'
        >>> format_bytes(268435456)
        '256.00 MB'

    Note:
        Referenciado em: utils/system.py (memory_usage, gpu_memory_info).
    """
    abs_b = abs(n_bytes)
    sign = "-" if n_bytes < 0 else ""
    if abs_b >= 1024 ** 3:
        # ── Gigabytes: GPU VRAM, datasets grandes ─────────────────────
        return f"{sign}{abs_b / (1024 ** 3):.2f} GB"
    elif abs_b >= 1024 ** 2:
        # ── Megabytes: modelos .keras, scalers ────────────────────────
        return f"{sign}{abs_b / (1024 ** 2):.2f} MB"
    elif abs_b >= 1024:
        # ── Kilobytes: configs, manifests ─────────────────────────────
        return f"{sign}{abs_b / 1024:.2f} KB"
    else:
        # ── Bytes: arquivos triviais ──────────────────────────────────
        return f"{sign}{abs_b:.0f} B"


# ════════════════════════════════════════════════════════════════════════════
# SECAO: BANNERS E HEADERS
# ════════════════════════════════════════════════════════════════════════════
# Funcoes para criar banners visuais formatados via logging.
# Usadas em pontos de entrada de secoes do pipeline (data loading,
# treinamento, avaliacao) para demarcar fases no output.
# Toda saida via logger (NUNCA print — D9).
# ──────────────────────────────────────────────────────────────────────────


def log_header(
    title: str,
    width: int = 70,
    log: Optional[logging.Logger] = None,
) -> None:
    """Loga banner centralizado com bordas Unicode.

    Cria banner visual usando caracteres ``=`` para demarcar inicio
    de secoes do pipeline. Toda saida via logger.info().

    Args:
        title: Titulo do banner.
        width: Largura total em caracteres. Default: 70.
        log: Logger opcional. Se None, usa logger deste modulo.

    Example:
        >>> from geosteering_ai.utils.formatting import log_header
        >>> log_header("TREINAMENTO")

    Note:
        Referenciado em: training/loop.py, evaluation/metrics.py.
        Substitui print_header() do legado C0 (agora via logging).
    """
    _log = log or logger
    border = "=" * width
    _log.info(border)
    _log.info("%s", title.center(width))
    _log.info(border)


def log_section(
    title: str,
    items: Optional[Dict[str, str]] = None,
    col_width: int = 35,
    log: Optional[logging.Logger] = None,
) -> None:
    """Loga secao com titulo e pares chave-valor alinhados.

    Exibe titulo seguido de pares chave-valor com alinhamento
    dot-fill para facil leitura. Usado para blocos de configuracao,
    parametros de treinamento, e resumos.

    Args:
        title: Titulo da secao.
        items: Dicionario de pares chave-valor a exibir.
            Se None, exibe apenas o titulo.
        col_width: Largura da coluna de chaves (dot-fill). Default: 35.
        log: Logger opcional.

    Example:
        >>> from geosteering_ai.utils.formatting import log_section
        >>> log_section("Modelo", {"Tipo": "ResNet_18", "Params": "11.2M"})

    Note:
        Referenciado em: training/loop.py, evaluation/comparison.py.
    """
    _log = log or logger
    _log.info("── %s ──", title)
    if items:
        for key, val in items.items():
            dots = "." * max(1, col_width - len(key))
            _log.info("  %s %s %s", key, dots, val)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FLAG DISPLAY
# ════════════════════════════════════════════════════════════════════════════
# Funcoes para colorir e exibir valores de FLAGS do pipeline.
# bool True = verde, False = amarelo, None = amarelo, outros = branco.
# Usado pelo modulo de relatorio de FLAGS (equivalente a C42A legado).
# ──────────────────────────────────────────────────────────────────────────


def colorize_flag_value(value: Any) -> str:
    """Colore valor de FLAG semanticamente.

    Mapeia tipo/valor para cor ANSI: True=verde, False=amarelo,
    None=amarelo, outros=branco. Util para relatorios de FLAGS
    no terminal.

    Args:
        value: Valor da FLAG (bool, str, int, float, None, etc.).

    Returns:
        str: String com valor colorido e RESET ao final.

    Example:
        >>> from geosteering_ai.utils.formatting import colorize_flag_value
        >>> colorize_flag_value(True)  # verde
        >>> colorize_flag_value(False)  # amarelo

    Note:
        Referenciado em: relatorio de FLAGS (equivalente C42A legado).
    """
    if isinstance(value, bool):
        if value:
            # ── True: verde brilhante (FLAG ativa) ────────────────────
            return f"{C.BRIGHT_GREEN}{value}{C.RESET}"
        else:
            # ── False: amarelo (FLAG inativa) ─────────────────────────
            return f"{C.YELLOW}{value}{C.RESET}"
    elif value is None:
        # ── None: amarelo (valor ausente/default) ─────────────────────
        return f"{C.YELLOW}None{C.RESET}"
    else:
        # ── Outros (str, int, float, list): branco brilhante ─────────
        return f"{C.BRIGHT_WHITE}{value}{C.RESET}"


def log_flag_block(
    flags: Dict[str, Any],
    col_width: int = 35,
    log: Optional[logging.Logger] = None,
) -> None:
    """Loga bloco de FLAGS com alinhamento dot-fill e cores.

    Exibe cada FLAG com nome alinhado, dot-fill, e valor colorido
    semanticamente. Formato:
        ``  FLAG_NAME ............... True``

    Args:
        flags: Dicionario {nome_flag: valor}.
        col_width: Largura da coluna de nomes. Default: 35.
        log: Logger opcional.

    Example:
        >>> from geosteering_ai.utils.formatting import log_flag_block
        >>> log_flag_block({"USE_NOISE": True, "LEARNING_RATE": 1e-4})

    Note:
        Referenciado em: relatorio de FLAGS (equivalente C42A legado).
    """
    _log = log or logger
    for name, value in flags.items():
        dots = "." * max(1, col_width - len(name))
        colored = colorize_flag_value(value)
        _log.info("  %s %s %s", name, dots, colored)


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Formatacao numerica ───────────────────────────────────────────
    "format_number",
    "format_compact",
    "format_bytes",
    # ── Banners e headers ─────────────────────────────────────────────
    "log_header",
    "log_section",
    # ── FLAG display ──────────────────────────────────────────────────
    "colorize_flag_value",
    "log_flag_block",
]
