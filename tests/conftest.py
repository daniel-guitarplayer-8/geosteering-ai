# -*- coding: utf-8 -*-
"""Root conftest — setup global executado antes de qualquer plugin pytest.

Funções:

1. Alinhar ``QT_API`` (lido por pytest-qt) com o binding que
   ``sm_qt_compat`` carrega (PyQt6 preferido, PySide6 fallback). Necessário
   porque pytest-qt detecta o binding em ``pytest_configure`` (early), antes
   de qualquer ``pytest_plugins`` declarado em arquivos de teste.

   Sem este conftest, pytest-qt pode escolher PySide6 enquanto a aplicação
   real (Simulation Manager) usa PyQt6 — provocando ``TypeError`` em
   ``qtbot.addWidget`` por cross-binding isinstance falso.

   Sprint v2.33 — Suite pytest-qt para GUI.

2. Configurar ``NUMBA_CACHE_DIR`` em tmpfs ANTES de qualquer import Numba
   (v2.36 D3, finalizado v2.38). Reduz I/O do carregamento de ``.nbc``
   entre testes paralelos sem poluir ``geosteering_ai/__pycache__/_numba/``.
   Respeita override do usuário (``NUMBA_CACHE_DIR`` já setado).

   Antes deste setup, ``cli/_main.py`` definia a env var em tempo de
   import — mas só quando o CLI era importado. Testes que importam
   diretamente ``simulation/_numba/*`` não acionavam o setup e geravam
   ``.nbc`` em locais inconsistentes entre runs.
"""

from __future__ import annotations

import os
import tempfile

# ── NUMBA_CACHE_DIR em tmpfs (v2.36 D3) ──────────────────────────────────
# DEVE executar antes de qualquer ``import numba`` ou
# ``from geosteering_ai.simulation._numba import ...``. Idempotente:
# preserva override do usuário (CI, debug).
if "NUMBA_CACHE_DIR" not in os.environ:
    _cache_dir = os.path.join(tempfile.gettempdir(), "geosteering_numba_cache_test")
    try:
        os.makedirs(_cache_dir, mode=0o700, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = _cache_dir
    except OSError:
        # Filesystem read-only ou permissão negada: deixa Numba
        # usar o default (__pycache__/_numba/). Não-fatal.
        pass

# Setar QT_API ANTES de qualquer import Qt ou pytest-qt. Idempotente: respeita
# override do usuário via env var.
if "QT_API" not in os.environ:
    try:
        import PyQt6.QtCore  # noqa: F401

        os.environ["QT_API"] = "pyqt6"
    except ImportError:
        try:
            import PySide6.QtCore  # noqa: F401

            os.environ["QT_API"] = "pyside6"
        except ImportError:
            # Nenhum binding Qt6 — testes GUI serão skipped no módulo
            pass

# Mesmo setup para Qt platform headless (CI Linux). Espelha a lógica de
# ``tests/conftest_qt.py`` mas executado mais cedo no ciclo de carga.
if "DISPLAY" not in os.environ and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"


# ── Skip automático de testes @pytest.mark.gpu sem GPU física (Sprint v2.40 D9) ──
# Detecta GPU via TF list_physical_devices ou JAX devices, lazily (sem import
# eager pesado). Aplica skip ao tempo de coleta — testes não-GPU continuam
# rodando normalmente.
def _detect_gpu_available() -> bool:
    """Retorna True se TF ou JAX detectam GPU física.

    Lazy: imports só acontecem na 1ª chamada. Falhas de import são tratadas
    como ausência de GPU (skip seguro em ambientes CPU-only).
    """
    try:
        import tensorflow as tf  # type: ignore[import-untyped]

        if tf.config.list_physical_devices("GPU"):
            return True
    except Exception:
        pass
    try:
        import jax  # type: ignore[import-untyped]

        # Hardening v2.40 (review code #7): excluir JAX-Metal (macOS) que
        # se identifica como "gpu" mas não é CUDA Tensor Cores — testes mp16
        # GPU assumem CUDA. CLAUDE.md: macOS=Colab, Linux=local p/ GPU real.
        for d in jax.devices():
            if d.platform == "gpu" and "metal" not in str(d).lower():
                return True
    except (ImportError, AttributeError, RuntimeError):
        # Hardening v2.40 (review security #4): exceções específicas
        # em vez de broad Exception (mascarava SystemError/MemoryError).
        pass
    return False


def pytest_collection_modifyitems(config, items):
    """Adiciona skip a testes ``@pytest.mark.gpu`` quando GPU não está disponível.

    Hook nativo pytest. Reusa marker ``gpu`` definido em
    ``pyproject.toml::tool.pytest.ini_options``. Aplica antes da execução
    para que ``pytest -m gpu`` em CPU mostre todos como SKIPPED (não
    collected vazio — mantém visibilidade do escopo).
    """
    import pytest

    if _detect_gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="GPU física não disponível (Sprint v2.40)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
