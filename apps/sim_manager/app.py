# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/app.py                                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : app — entry-point do Simulation Manager MVVM               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — bootstrap (spec 0011a)                     ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — walking skeleton                                ║
# ║  Dependências: gui.qt_compat, gui.shell, .main_window, perspectives        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Sobe o ``QApplication``, cria a ``SM_MainWindow`` e registra a          ║
# ║    perspectiva Simulação. Entry-point PARALELO ao monólito (não o          ║
# ║    substitui) — prova a pilha MVVM end-to-end.                            ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    python -m apps.sim_manager.app                                         ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    main                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Entry-point do Simulation Manager MVVM — ``python -m apps.sim_manager.app`` (0011a)."""

from __future__ import annotations

import sys
from typing import List, Optional

from apps.sim_manager.main_window import SM_MainWindow
from apps.sim_manager.perspectives.simulation.perspective import SimulationPerspective
from geosteering_ai.gui.qt_compat import (
    QT_AVAILABLE,
    QT_IMPORT_ERROR,
    QtWidgets,
    enforce_c_locale,
)
from geosteering_ai.gui.shell.context import AppContext

__all__ = ["main"]

_APP_NAME = "Geosteering AI — Simulation Manager"


def main(argv: Optional[List[str]] = None) -> int:
    """Sobe o app MVVM do Simulation Manager.

    Args:
        argv: argumentos (default ``sys.argv``).

    Returns:
        Código de saída do ``QApplication.exec()`` (0 = ok), ou 1 se não há
        binding Qt6 instalado.
    """
    if not QT_AVAILABLE:
        sys.stderr.write((QT_IMPORT_ERROR or "Qt6 indisponível.") + "\n")
        return 1

    args = argv if argv is not None else sys.argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(args)
    # Locale C: ponto decimal nos QDoubleSpinBox (paridade c/ o monólito; evita
    # que um locale pt-BR mostre/parseie "1,5" e divirja do CSV Python float()).
    enforce_c_locale(app)

    ctx = AppContext(app_name=_APP_NAME)
    window = SM_MainWindow(ctx)
    window.add_perspective(SimulationPerspective())
    window.resize(960, 720)
    window.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover — entry-point manual
    raise SystemExit(main())
