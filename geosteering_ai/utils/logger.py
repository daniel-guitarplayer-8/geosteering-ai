# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: utils/logger.py                                                   ║
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
# ║    • Logging estruturado com cores ANSI para terminal (D9 obrigatorio)    ║
# ║    • ColoredFormatter com mapeamento semantico por nivel                  ║
# ║    • setup_logger() para configurar o logger raiz do pacote               ║
# ║    • get_logger() como convenience para modulos individuais               ║
# ║                                                                            ║
# ║  Dependencias: logging (stdlib), sys (stdlib)                              ║
# ║  Exports: ~4 simbolos — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9 (utils/)                           ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import logging
import sys


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES — CODIGOS ANSI SEMANTICOS
# ════════════════════════════════════════════════════════════════════════════
# Mapeamento de niveis de logging para cores ANSI.
# Permite diferenciar visualmente severidade no terminal.
# Compativel com terminais que suportam ANSI (VSCode, Colab, iTerm2).
# Terminais sem suporte ignoram os codigos silenciosamente.
# ──────────────────────────────────────────────────────────────────────────


class C:
    """Namespace de codigos ANSI para cores semanticas no terminal.

    Utilizado por ColoredFormatter para colorir saida de logging.
    Todos os atributos sao strings contendo sequencias de escape ANSI.

    Attributes:
        BOLD (str): Negrito.
        DIM (str): Texto esmaecido.
        RESET (str): Reseta todas as formatacoes.
        RED (str): Vermelho — erros criticos.
        GREEN (str): Verde — sucesso, validacoes aprovadas.
        YELLOW (str): Amarelo — avisos.
        BLUE (str): Azul — informacao geral.
        CYAN (str): Ciano — debug.
        BRIGHT_GREEN (str): Verde brilhante — flags True.
        BRIGHT_WHITE (str): Branco brilhante — valores genericos.
        BRIGHT_RED (str): Vermelho brilhante — erros fatais.
        BG_RED (str): Fundo vermelho — banners criticos.

    Example:
        >>> from geosteering_ai.utils.logger import C
        >>> msg = f"{C.BOLD}{C.GREEN}OK{C.RESET}"

    Note:
        Referenciado em: utils/formatting.py (print_header, print_section).
        Ref: Padrao D9 (logging estruturado, NUNCA print).
    """

    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"

    BRIGHT_GREEN = "\033[92m"
    BRIGHT_WHITE = "\033[97m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


# ════════════════════════════════════════════════════════════════════════════
# SECAO: COLORED FORMATTER
# ════════════════════════════════════════════════════════════════════════════
# Formatter customizado que adiciona cores ANSI por nivel de logging.
# Mapeamento semantico: DEBUG=ciano, INFO=azul, WARNING=amarelo,
# ERROR=vermelho, CRITICAL=vermelho brilhante com fundo.
# Ref: Padrao D9 — logging estruturado com diferenciacao visual.
# ──────────────────────────────────────────────────────────────────────────


# Mapeamento nivel → cor ANSI.
# DEBUG: ciano (detalhe tecnico), INFO: azul (operacao normal),
# WARNING: amarelo (atencao), ERROR: vermelho (falha recuperavel),
# CRITICAL: vermelho brilhante fundo (falha irrecuperavel).
_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: C.CYAN,
    logging.INFO: C.BLUE,
    logging.WARNING: C.YELLOW,
    logging.ERROR: C.RED,
    logging.CRITICAL: f"{C.BG_RED}{C.BRIGHT_WHITE}",
}


class ColoredFormatter(logging.Formatter):
    """Formatter de logging com cores ANSI semanticas por nivel.

    Cada mensagem eh prefixada com cor correspondente ao nivel de
    severidade, facilitando triagem visual no terminal. O nome do
    logger aparece em cinza (DIM) para contexto sem poluir visualmente.

    O formato padrao eh:
        ``%(levelname)-8s %(name)s — %(message)s``

    Onde levelname recebe cor e name aparece esmaecido.

    Attributes:
        COLORS (dict[int, str]): Mapeamento nivel → codigo ANSI.

    Example:
        >>> import logging
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(ColoredFormatter())
        >>> logger = logging.getLogger("test")
        >>> logger.addHandler(handler)

    Note:
        Referenciado em: utils/logger.py (setup_logger).
        Compativel com Google Colab, VSCode Terminal, iTerm2.
        Terminais sem ANSI ignoram os codigos silenciosamente.
    """

    COLORS = _LEVEL_COLORS

    def format(self, record: logging.LogRecord) -> str:
        """Formata registro de log com cores ANSI.

        Args:
            record: Registro de logging a ser formatado.

        Returns:
            str: Mensagem formatada com cores ANSI.
        """
        color = self.COLORS.get(record.levelno, "")
        # ── Levelname colorido + nome do logger em DIM ────────────────
        # NOTA: NAO mutar record.msg/record.args — o LogRecord eh
        # compartilhado entre handlers. Mutacao corrompe FileHandlers.
        levelname = f"{color}{record.levelname:<8s}{C.RESET}"
        name = f"{C.DIM}{record.name}{C.RESET}"
        message = record.getMessage()
        return f"{levelname} {name} — {message}"


# ════════════════════════════════════════════════════════════════════════════
# SECAO: SETUP E FACTORY
# ════════════════════════════════════════════════════════════════════════════
# Funcoes para configurar o logger raiz do pacote e obter loggers
# por modulo. setup_logger() deve ser chamado UMA VEZ no inicio
# da execucao (notebook ou script). get_logger() eh um atalho
# para logging.getLogger() com namespace do pacote.
# ──────────────────────────────────────────────────────────────────────────

# Nome raiz do pacote — todos os loggers do geosteering_ai herdam dele.
_ROOT_LOGGER_NAME = "geosteering_ai"


def setup_logger(
    name: str = _ROOT_LOGGER_NAME,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configura o logger raiz do pacote com ColoredFormatter.

    Deve ser chamado UMA VEZ no inicio da execucao (tipicamente no
    notebook Colab ou no script principal). Configura um StreamHandler
    com ColoredFormatter para saida colorida no terminal.

    Previne duplicacao de handlers em chamadas repetidas (re-entrant
    safe para ambientes Jupyter onde celulas podem ser re-executadas).

    Args:
        name: Nome do logger. Default: "geosteering_ai" (raiz do pacote).
            Todos os loggers filhos (e.g., "geosteering_ai.data.loading")
            herdam a configuracao automaticamente.
        level: Nivel minimo de logging. Default: logging.INFO.
            Valores comuns: logging.DEBUG (10), logging.INFO (20),
            logging.WARNING (30).

    Returns:
        logging.Logger: Logger configurado com ColoredFormatter.

    Raises:
        TypeError: Se level nao eh um inteiro valido de logging.

    Example:
        >>> from geosteering_ai.utils.logger import setup_logger
        >>> logger = setup_logger(level=logging.DEBUG)
        >>> logger.info("Pipeline inicializado")

    Note:
        Referenciado em:
            - Notebooks Colab (celula de setup)
            - scripts/ (entrypoints CLI)
            - tests/conftest.py (configuracao de teste)
        Ref: Padrao D9 (logging estruturado, NUNCA print).
        Handler duplicacao prevenida: verifica handlers existentes
        antes de adicionar novo StreamHandler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ── Prevenir duplicacao de handlers ────────────────────────────────
    # Em Jupyter/Colab, celulas podem ser re-executadas. Sem este guard,
    # cada re-execucao adicionaria um handler extra, duplicando output.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)

    # ── Nao propagar para root logger ──────────────────────────────────
    # Evita duplicacao com logging.basicConfig() ou handlers externos.
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Obtem logger com namespace do pacote.

    Convenience wrapper que prefixa o nome do modulo com o namespace
    raiz do pacote. Todos os modulos de geosteering_ai/ DEVEM usar
    esta funcao ou ``logging.getLogger(__name__)``.

    Args:
        name: Nome do modulo. Tipicamente ``__name__`` (e.g.,
            "geosteering_ai.data.loading"). Se nao contem o prefixo
            do pacote, eh adicionado automaticamente.

    Returns:
        logging.Logger: Logger com nome qualificado.

    Example:
        >>> from geosteering_ai.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Modulo carregado")

    Note:
        Referenciado em: todos os modulos de geosteering_ai/.
        Equivalente funcional a logging.getLogger(__name__) quando
        chamado dentro do pacote — oferecido como conveniencia.
    """
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Cores ANSI ────────────────────────────────────────────────────
    "C",
    # ── Formatter ─────────────────────────────────────────────────────
    "ColoredFormatter",
    # ── Setup e factory ───────────────────────────────────────────────
    "setup_logger",
    "get_logger",
]
