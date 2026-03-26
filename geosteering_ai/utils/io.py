# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: utils/io.py                                                       ║
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
# ║    • NumpyEncoder para serializar numpy types para JSON                   ║
# ║    • safe_json_dump() para salvar dicionarios com tratamento de erros     ║
# ║    • safe_json_load() para carregar JSON com tratamento de erros          ║
# ║                                                                            ║
# ║  Dependencias: json (stdlib), logging (stdlib), pathlib (stdlib)           ║
# ║  Exports: ~3 simbolos — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9 (utils/)                           ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: NUMPY JSON ENCODER
# ════════════════════════════════════════════════════════════════════════════
# Encoder customizado para json.dumps() que converte tipos numpy
# (ndarray, integer, floating, bool_) para tipos Python nativos.
# Necessario porque json.dumps() nao serializa numpy types nativamente.
# Usado por safe_json_dump() para salvar FLAGS, manifests, e metricas.
# ──────────────────────────────────────────────────────────────────────────


class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON que converte tipos numpy para tipos Python nativos.

    Estende json.JSONEncoder para lidar com numpy arrays, inteiros,
    floats, e booleans que nao sao serializaveis pelo encoder padrao.
    Necessario para salvar FLAGS, manifestos, e metricas que contem
    valores numpy.

    Conversoes realizadas:
        - np.ndarray → list (via .tolist())
        - np.integer → int
        - np.floating → float
        - np.bool_ → bool

    Attributes:
        Herda de json.JSONEncoder — sem atributos adicionais.

    Example:
        >>> import json, numpy as np
        >>> from geosteering_ai.utils.io import NumpyEncoder
        >>> data = {"array": np.array([1, 2, 3]), "val": np.float32(0.5)}
        >>> json.dumps(data, cls=NumpyEncoder)
        '{"array": [1, 2, 3], "val": 0.5}'

    Note:
        Referenciado em: utils/io.py (safe_json_dump).
        Importacao de numpy eh local para evitar dependencia hard
        em contextos onde apenas json puro eh necessario.
    """

    def default(self, obj: Any) -> Any:
        """Converte tipos numpy para tipos Python serializaveis.

        Args:
            obj: Objeto a serializar. Se nao eh tipo numpy,
                delega para o encoder padrao.

        Returns:
            Any: Objeto Python nativo serializavel por JSON.

        Raises:
            TypeError: Se obj nao eh serializavel (via super().default).
        """
        try:
            import numpy as np  # type: ignore[import-untyped]

            if isinstance(obj, np.ndarray):
                # ── ndarray → list Python nativa ──────────────────────
                return obj.tolist()
            elif isinstance(obj, np.integer):
                # ── np.int32, np.int64 → int ──────────────────────────
                return int(obj)
            elif isinstance(obj, np.floating):
                # ── np.float32, np.float64 → float ────────────────────
                return float(obj)
            elif isinstance(obj, np.bool_):
                # ── np.bool_ → bool ───────────────────────────────────
                return bool(obj)
        except ImportError:
            pass

        return super().default(obj)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: JSON I/O SEGURO
# ════════════════════════════════════════════════════════════════════════════
# Funcoes para salvar e carregar JSON com tratamento de erros,
# encoding UTF-8, e suporte a tipos numpy. Usadas para persistir
# FLAGS, manifestos, metricas de treinamento, e configuracoes.
# ──────────────────────────────────────────────────────────────────────────


def safe_json_dump(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    log: Optional[logging.Logger] = None,
) -> bool:
    """Salva dicionario como JSON com tratamento de erros.

    Usa NumpyEncoder para converter tipos numpy automaticamente.
    Cria diretorios pai se nao existem. Encoding UTF-8 com
    indent=2 para legibilidade.

    Args:
        data: Dicionario a ser salvo.
        filepath: Caminho do arquivo JSON de destino.
        log: Logger opcional.

    Returns:
        bool: True se salvo com sucesso, False em caso de erro.

    Example:
        >>> from geosteering_ai.utils.io import safe_json_dump
        >>> metrics = {"r2": 0.98, "rmse": 0.12}
        >>> safe_json_dump(metrics, "results/metrics.json")
        True

    Note:
        Referenciado em:
            - training/loop.py (salvar FLAGS e metricas)
            - inference/export.py (salvar manifesto de modelo)
        Encoding: UTF-8 para suportar acentos em descricoes PT-BR.
    """
    _log = log or logger
    filepath = Path(filepath)

    try:
        # ── Criar diretorio pai se necessario ─────────────────────────
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

        _log.debug("JSON salvo: %s", filepath)
        return True

    except (OSError, TypeError, ValueError) as e:
        _log.error("Falha ao salvar JSON %s: %s", filepath, e)
        return False


def safe_json_load(
    filepath: Union[str, Path],
    log: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """Carrega arquivo JSON com tratamento de erros.

    Retorna None se o arquivo nao existe ou contem JSON invalido,
    logando o erro sem levantar excecao.

    Args:
        filepath: Caminho do arquivo JSON a carregar.
        log: Logger opcional.

    Returns:
        dict[str, Any] | None: Dicionario carregado, ou None se falhou.

    Example:
        >>> from geosteering_ai.utils.io import safe_json_load
        >>> config = safe_json_load("results/metrics.json")
        >>> if config is not None:
        ...     logger.info("r2: %s", config["r2"])

    Note:
        Referenciado em: inference/pipeline.py (carregar manifesto).
    """
    _log = log or logger
    filepath = Path(filepath)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        _log.debug("JSON carregado: %s", filepath)
        return data

    except FileNotFoundError:
        _log.warning("Arquivo nao encontrado: %s", filepath)
        return None
    except (json.JSONDecodeError, ValueError) as e:
        _log.error("JSON invalido em %s: %s", filepath, e)
        return None


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Encoder ───────────────────────────────────────────────────────
    "NumpyEncoder",
    # ── JSON I/O ──────────────────────────────────────────────────────
    "safe_json_dump",
    "safe_json_load",
]
