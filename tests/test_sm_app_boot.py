# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_app_boot.py                                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Ordem de boot do SM MVVM (anti-segfault TF↔Qt-xcb)                       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app — entry-point (apps/sim_manager/app.py)             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-16                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Gate de regressão da ORDEM DE BOOT: ``apps.sim_manager.app`` NÃO pode    ║
# ║    importar TensorFlow/``geosteering_ai`` no nível de módulo. TF importado  ║
# ║    ANTES de instanciar o ``QApplication`` faz o plugin ``xcb`` do Qt dar    ║
# ║    SEGFAULT (conflito de libs C TF↔Qt) — o app cria o QApplication primeiro ║
# ║    e só então importa o resto. Este teste trava esse invariante.           ║
# ║                                                                           ║
# ║    NOTA: o segfault em si só ocorre no plugin REAL (xcb) com display; a     ║
# ║    suíte roda ``offscreen`` (que o mascara), então o gate verifica a CAUSA  ║
# ║    (import de TF no nível de módulo) num interpretador LIMPO (subprocesso). ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Gate de regressão da ordem de boot do SM MVVM (TF-após-QApplication)."""

from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


def test_app_import_does_not_pull_tensorflow():
    """``import apps.sim_manager.app`` NÃO carrega TF/``geosteering_ai`` (interpretador limpo).

    Roda num SUBPROCESSO (sys.modules limpo): dentro do processo do pytest, TF/
    ``geosteering_ai`` já estariam carregados por outros testes, mascarando a regressão.
    Se alguém mover os imports pesados de volta ao topo de ``app.py``, TF carrega no
    import do módulo (ANTES do QApplication) → segfault xcb em produção; este teste falha.
    """
    code = (
        "import sys; import apps.sim_manager.app as a; "
        "assert 'tensorflow' not in sys.modules, "
        "'TensorFlow carregado no import de app (regressão de ordem de boot — segfault xcb)'; "
        "assert 'geosteering_ai' not in sys.modules, "
        "'geosteering_ai carregado no import de app (dispararia geosteering_ai/__init__ → TF)'; "
        "assert hasattr(a, 'main') and hasattr(a, '_create_qapp'); "
        "print('BOOT_IMPORT_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"subprocesso falhou (rc={result.returncode}).\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "BOOT_IMPORT_OK" in result.stdout


@pytest.mark.gui
def test_create_qapp_returns_application(qapp):
    """``_create_qapp`` devolve um ``QApplication`` (binding direto, sem qt_compat/TF)."""
    from apps.sim_manager.app import _create_qapp
    from geosteering_ai.gui.qt_compat import QtWidgets

    app = _create_qapp([])
    assert app is not None
    assert isinstance(app, QtWidgets.QApplication)
    # idempotente: devolve a instância existente, não cria outra
    assert _create_qapp([]) is app
