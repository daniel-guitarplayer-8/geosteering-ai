# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: utils/system.py                                                   ║
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
# ║    • Deteccao de ambiente (Colab, Kaggle, Jupyter, local)                 ║
# ║    • Deteccao de GPU e hardware de borda                                  ║
# ║    • Gerenciamento de diretorios (safe_mkdir, ensure_dirs)                ║
# ║    • Monitoramento de memoria (RAM e GPU)                                 ║
# ║    • Limpeza de memoria (gc + Keras backend)                              ║
# ║    • Seeding reprodutivel (random, numpy, tensorflow)                     ║
# ║                                                                            ║
# ║  Dependencias: os, sys, gc, platform, logging (stdlib)                    ║
# ║  Exports: ~10 simbolos — ver __all__                                      ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9 (utils/)                           ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import gc
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: DETECCAO DE AMBIENTE
# ════════════════════════════════════════════════════════════════════════════
# Funcoes para detectar o ambiente de execucao do pipeline.
# Importante porque o comportamento muda entre Colab (GPU, paths /content),
# Kaggle (GPU, paths /kaggle), Jupyter local, e script CLI.
# A deteccao influencia: paths de dados, disponibilidade de GPU,
# modo de display (HTML vs texto), e estrategia de cache.
# ──────────────────────────────────────────────────────────────────────────


def is_colab() -> bool:
    """Detecta se esta executando no Google Colab.

    Verifica presenca do modulo ``google.colab`` em sys.modules.
    Colab pre-importa este modulo automaticamente no startup.

    Returns:
        bool: True se executando no Google Colab.

    Example:
        >>> from geosteering_ai.utils.system import is_colab
        >>> if is_colab():
        ...     data_dir = "/content/drive/MyDrive/data"

    Note:
        Referenciado em: data/loading.py (paths), training/loop.py (GPU).
    """
    return "google.colab" in sys.modules


def is_kaggle() -> bool:
    """Detecta se esta executando no Kaggle.

    Verifica variavel de ambiente ``KAGGLE_KERNEL_RUN_TYPE``.

    Returns:
        bool: True se executando no Kaggle.
    """
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def is_jupyter() -> bool:
    """Detecta se esta executando em Jupyter Notebook/Lab.

    Verifica tipo do shell IPython. Retorna False em scripts CLI
    e no Google Colab (que tem deteccao propria).

    Returns:
        bool: True se executando em Jupyter (exceto Colab/Kaggle).
    """
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def has_gpu() -> bool:
    """Detecta se GPU (CUDA/ROCm) esta disponivel via TensorFlow.

    Usa ``tf.config.list_physical_devices("GPU")``. Retorna False
    se TensorFlow nao esta instalado ou nenhuma GPU eh encontrada.

    Returns:
        bool: True se ao menos 1 GPU esta disponivel.

    Example:
        >>> from geosteering_ai.utils.system import has_gpu
        >>> if has_gpu():
        ...     strategy = tf.distribute.MirroredStrategy()

    Note:
        Referenciado em: training/loop.py, notebooks Colab.
        TF importado localmente para evitar dependencia em testes CPU-only.
    """
    try:
        import tensorflow as tf  # type: ignore[import-untyped]
        return len(tf.config.list_physical_devices("GPU")) > 0
    except (ImportError, RuntimeError):
        return False


def detect_environment() -> str:
    """Detecta e retorna nome do ambiente de execucao.

    Returns:
        str: Um de "Google Colab", "Kaggle", "Jupyter Notebook",
        "Local/Script".

    Example:
        >>> from geosteering_ai.utils.system import detect_environment
        >>> env = detect_environment()
        >>> # "Google Colab" | "Kaggle" | "Jupyter Notebook" | "Local/Script"

    Note:
        Referenciado em: relatorio de FLAGS (equivalente C42A).
    """
    if is_colab():
        return "Google Colab"
    elif is_kaggle():
        return "Kaggle"
    elif is_jupyter():
        return "Jupyter Notebook"
    else:
        return "Local/Script"


def get_environment_info() -> Dict[str, Any]:
    """Coleta informacoes completas do ambiente de execucao.

    Retorna dicionario com ambiente detectado, GPU, versao Python,
    plataforma, e diretorio de trabalho.

    Returns:
        dict[str, Any]: Dicionario com chaves:
            - environment (str): Nome do ambiente
            - has_gpu (bool): GPU disponivel
            - python_version (str): Versao do Python
            - platform (str): Sistema operacional
            - architecture (str): Arquitetura do processador
            - cwd (str): Diretorio de trabalho atual

    Example:
        >>> from geosteering_ai.utils.system import get_environment_info
        >>> info = get_environment_info()
        >>> info["environment"]
        'Local/Script'

    Note:
        Referenciado em: notebooks Colab (celula de setup), relatorio FLAGS.
    """
    return {
        "environment": detect_environment(),
        "has_gpu": has_gpu(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cwd": os.getcwd(),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECAO: GERENCIAMENTO DE DIRETORIOS
# ════════════════════════════════════════════════════════════════════════════
# Criacao segura de diretorios com logging. Usado para criar
# EXPERIMENT_DIR e subdiretorios (checkpoints, logs, plots).
# Toda saida via logger (NUNCA print — D9).
# ──────────────────────────────────────────────────────────────────────────


def safe_mkdir(
    path: Union[str, Path],
    log: Optional[logging.Logger] = None,
) -> bool:
    """Cria diretorio de forma segura com logging.

    Wrapper sobre ``os.makedirs(exist_ok=True)`` com tratamento de
    erros e logging do resultado.

    Args:
        path: Caminho do diretorio a criar.
        log: Logger opcional.

    Returns:
        bool: True se diretorio foi criado ou ja existe.

    Example:
        >>> from geosteering_ai.utils.system import safe_mkdir
        >>> safe_mkdir("/content/experiments/S22")
        True

    Note:
        Referenciado em: training/loop.py (EXPERIMENT_DIR).
    """
    _log = log or logger
    try:
        os.makedirs(path, exist_ok=True)
        _log.debug("Diretorio OK: %s", path)
        return True
    except OSError as e:
        _log.error("Falha ao criar diretorio %s: %s", path, e)
        return False


def ensure_dirs(
    paths: List[Union[str, Path]],
    log: Optional[logging.Logger] = None,
) -> Dict[str, bool]:
    """Cria multiplos diretorios de forma segura.

    Batch version de safe_mkdir(). Retorna dicionario com status
    de cada diretorio.

    Args:
        paths: Lista de caminhos de diretorios.
        log: Logger opcional.

    Returns:
        dict[str, bool]: Mapeamento {caminho: sucesso}.

    Example:
        >>> from geosteering_ai.utils.system import ensure_dirs
        >>> results = ensure_dirs(["logs/", "checkpoints/", "plots/"])

    Note:
        Referenciado em: training/loop.py (subdiretorios EXPERIMENT_DIR).
    """
    return {str(p): safe_mkdir(p, log=log) for p in paths}


# ════════════════════════════════════════════════════════════════════════════
# SECAO: MONITORAMENTO DE MEMORIA
# ════════════════════════════════════════════════════════════════════════════
# Funcoes para monitorar uso de RAM e GPU VRAM.
# Estrategia multi-fallback: psutil → /proc/status → None.
# GPU via TensorFlow (tf.config.experimental.get_memory_info).
# ──────────────────────────────────────────────────────────────────────────


def memory_usage() -> Dict[str, Any]:
    """Retorna uso de memoria RAM do processo.

    Estrategia de deteccao:
    1. psutil (preferido, cross-platform)
    2. /proc/self/status (fallback Linux/Colab)
    3. Indisponivel (available=False)

    Returns:
        dict[str, Any]: Dicionario com chaves:
            - available (bool): Se medicao foi possivel
            - rss_mb (float): Resident Set Size em MB (0.0 se indisponivel)
            - method (str): Metodo usado ("psutil", "/proc", "unavailable")

    Example:
        >>> from geosteering_ai.utils.system import memory_usage
        >>> info = memory_usage()
        >>> info["rss_mb"]
        256.5

    Note:
        Referenciado em: training/callbacks.py (MemoryMonitor).
    """
    # ── Estrategia 1: psutil (preferido) ──────────────────────────────
    try:
        import psutil  # type: ignore[import-untyped]
        proc = psutil.Process(os.getpid())
        rss_bytes = proc.memory_info().rss
        return {
            "available": True,
            "rss_mb": rss_bytes / (1024 ** 2),
            "method": "psutil",
        }
    except ImportError:
        pass

    # ── Estrategia 2: /proc/self/status (Linux/Colab) ────────────────
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    return {
                        "available": True,
                        "rss_mb": rss_kb / 1024,
                        "method": "/proc",
                    }
    except (FileNotFoundError, PermissionError):
        pass

    # ── Fallback: indisponivel ────────────────────────────────────────
    return {"available": False, "rss_mb": 0.0, "method": "unavailable"}


def gpu_memory_info() -> Dict[str, Any]:
    """Retorna informacoes de memoria GPU via TensorFlow.

    Usa ``tf.config.experimental.get_memory_info()`` para obter
    uso de VRAM de cada GPU disponivel.

    Returns:
        dict[str, Any]: Dicionario com chaves:
            - available (bool): Se GPU esta disponivel
            - device_count (int): Numero de GPUs
            - devices (list[dict]): Info por GPU (name, current_mb, peak_mb)

    Example:
        >>> from geosteering_ai.utils.system import gpu_memory_info
        >>> info = gpu_memory_info()
        >>> if info["available"]:
        ...     logger.info("VRAM: %.1f MB", info["devices"][0]["current_mb"])

    Note:
        Referenciado em: training/callbacks.py (MemoryMonitor).
        TF importado localmente para evitar dependencia circular.
    """
    try:
        import tensorflow as tf  # type: ignore[import-untyped]
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return {"available": False, "device_count": 0, "devices": []}

        devices = []
        for gpu in gpus:
            try:
                # gpu.name eh "/physical_device:GPU:0", mas get_memory_info
                # espera "GPU:0". Remover prefixo "/physical_device:".
                device_id = gpu.name.replace("/physical_device:", "")
                mem = tf.config.experimental.get_memory_info(device_id)
                devices.append({
                    "name": gpu.name,
                    "current_mb": mem.get("current", 0) / (1024 ** 2),
                    "peak_mb": mem.get("peak", 0) / (1024 ** 2),
                })
            except (RuntimeError, ValueError):
                devices.append({
                    "name": gpu.name,
                    "current_mb": 0.0,
                    "peak_mb": 0.0,
                })

        return {
            "available": True,
            "device_count": len(gpus),
            "devices": devices,
        }
    except (ImportError, RuntimeError):
        return {"available": False, "device_count": 0, "devices": []}


def clear_memory() -> None:
    """Limpa memoria (garbage collector + Keras backend).

    Executa coleta de lixo e limpa sessao Keras para liberar
    memoria de modelos descartados. Util entre trials de HPO
    e entre stages do N-Stage training.

    Example:
        >>> from geosteering_ai.utils.system import clear_memory
        >>> clear_memory()

    Note:
        Referenciado em: training/optuna_hpo.py, training/nstage.py.
        TF importado localmente para evitar dependencia circular.
    """
    gc.collect()
    try:
        import tensorflow as tf  # type: ignore[import-untyped]
        tf.keras.backend.clear_session()
    except ImportError:
        pass
    gc.collect()


# ════════════════════════════════════════════════════════════════════════════
# SECAO: SEEDING REPRODUTIVEL
# ════════════════════════════════════════════════════════════════════════════
# Configuracao de seeds para reproducibilidade total do pipeline.
# Seta seeds em: os.environ, random, numpy, tensorflow.
# Opcionalmente habilita determinismo de operacoes TF.
# ──────────────────────────────────────────────────────────────────────────

# Seed padrao do pipeline. Valor 42 por convencao.
GLOBAL_SEED: int = 42


def set_all_seeds(
    seed: int = GLOBAL_SEED,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Configura seeds para reproducibilidade total.

    Seta seeds em todos os geradores de numeros aleatorios:
    os.environ["PYTHONHASHSEED"], random, numpy, tensorflow.
    Opcionalmente habilita determinismo de operacoes TF
    (pode reduzir performance em GPU).

    Args:
        seed: Valor do seed. Default: 42.
        deterministic: Se True, habilita op determinism no TF.
            Pode reduzir performance em GPU mas garante resultados
            identicos entre execucoes. Default: True.

    Returns:
        dict[str, Any]: Metadata com status de cada seed:
            - seed (int): Valor usado
            - deterministic (bool): Se determinismo foi habilitado
            - python_ok (bool): PYTHONHASHSEED setado
            - numpy_ok (bool): numpy seed setado
            - tf_ok (bool): tensorflow seed setado

    Example:
        >>> from geosteering_ai.utils.system import set_all_seeds
        >>> info = set_all_seeds(seed=42, deterministic=True)
        >>> assert info["numpy_ok"]

    Note:
        Referenciado em: notebooks Colab (celula 0), training/loop.py.
        Ref: Reproducibilidade YAML + tag + seed = resultado identico.
    """
    result: Dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "python_ok": False,
        "numpy_ok": False,
        "tf_ok": False,
    }

    # ── Python hash seed ──────────────────────────────────────────────
    # NOTA: PYTHONHASHSEED so tem efeito se definido ANTES de iniciar
    # o interpretador. Em runtime, apenas random.seed() eh efetivo.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    result["python_ok"] = True

    # ── NumPy seed ────────────────────────────────────────────────────
    try:
        import numpy as np  # type: ignore[import-untyped]
        np.random.seed(seed)
        result["numpy_ok"] = True
    except ImportError:
        pass

    # ── TensorFlow seed ───────────────────────────────────────────────
    try:
        import tensorflow as tf  # type: ignore[import-untyped]
        tf.random.set_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
        result["tf_ok"] = True
    except (ImportError, RuntimeError):
        pass

    logger.info(
        "Seeds configurados: seed=%d, deterministic=%s, "
        "python=%s, numpy=%s, tf=%s",
        seed, deterministic,
        result["python_ok"], result["numpy_ok"], result["tf_ok"],
    )

    return result


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Constantes ────────────────────────────────────────────────────
    "GLOBAL_SEED",
    # ── Deteccao de ambiente ──────────────────────────────────────────
    "is_colab",
    "is_kaggle",
    "is_jupyter",
    "has_gpu",
    "detect_environment",
    "get_environment_info",
    # ── Diretorios ────────────────────────────────────────────────────
    "safe_mkdir",
    "ensure_dirs",
    # ── Memoria ───────────────────────────────────────────────────────
    "memory_usage",
    "gpu_memory_info",
    "clear_memory",
    # ── Seeding ───────────────────────────────────────────────────────
    "set_all_seeds",
]
