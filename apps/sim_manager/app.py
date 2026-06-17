# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/app.py                                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : app — entry-point do Simulation Manager MVVM               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — bootstrap (spec 0011a)                     ║
# ║  Versão      : v0.2                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — walking skeleton                                ║
# ║  Dependências: gui.qt_compat, gui.shell, .main_window, perspectives        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Sobe o ``QApplication``, cria a ``SM_MainWindow`` e registra as          ║
# ║    perspectivas. Entry-point PARALELO ao monólito (não o substitui).       ║
# ║                                                                           ║
# ║  ORDEM DE BOOT CRÍTICA (NÃO REORDENAR — causa segfault)                   ║
# ║    O ``QApplication`` é criado ANTES de qualquer import de ``geosteering_ai``║
# ║    porque o pacote (``geosteering_ai/__init__`` → noise → training →        ║
# ║    nstage) carrega TensorFlow no import, e importar TF ANTES de instanciar  ║
# ║    o QApplication faz o plugin ``xcb`` do Qt dar SEGFAULT (conflito de       ║
# ║    bibliotecas C TF↔Qt). Por isso o binding Qt é importado DIRETO aqui      ║
# ║    (PyQt6→PySide6) — sem passar por ``gui.qt_compat`` (que dispararia o      ║
# ║    ``__init__`` → TF) — e os demais imports vivem DENTRO de ``main`` após o  ║
# ║    app existir. Ver ``tests/test_sm_app_boot.py``.                          ║
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
from typing import Any, List, Optional

__all__ = ["main"]

_APP_NAME = "Geosteering AI — Simulation Manager"

# Scaffolds das próximas perspectivas (rail Antigravity) — itens cinza "em breve"
# que tornam a organização de TODOS os recursos do SM monólito visível já agora.
# Cada um será habilitado pela sua fatia (paridade total = Fatias 6c-6i).
_SCAFFOLD_PERSPECTIVES = (
    ("results", "Resultados", "📊", 1, "Fatia 6d"),
    ("benchmark", "Benchmark", "⚡", 2, "Fatia 6g"),
    ("analysis", "Análise", "🔬", 3, "Fatia 6f"),
)


def _create_qapp(argv: List[str]) -> Optional[Any]:
    """Cria (ou recupera) o ``QApplication`` ANTES de qualquer import de TF.

    Importa o binding Qt6 DIRETO (PyQt6 → PySide6), deliberadamente SEM passar por
    ``geosteering_ai.gui.qt_compat`` — importar qualquer módulo de ``geosteering_ai``
    dispara ``geosteering_ai/__init__`` que carrega TensorFlow, e TF importado ANTES
    do ``QApplication`` faz o plugin ``xcb`` do Qt dar segfault. Criar o app primeiro
    (com o binding cru) e só então importar o resto (já com TF seguro) elimina o crash.

    Args:
        argv: argumentos da aplicação (tipicamente ``sys.argv``).

    Returns:
        A instância de ``QApplication`` (nova ou já existente), ou ``None`` se nenhum
        binding Qt6 estiver instalado.
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            return None
    return QApplication.instance() or QApplication(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Sobe o app MVVM do Simulation Manager.

    Args:
        argv: argumentos (default ``sys.argv``).

    Returns:
        Código de saída do ``QApplication.exec()`` (0 = ok), ou 1 se não há
        binding Qt6 instalado.
    """
    args = argv if argv is not None else sys.argv

    # ── 1) QApplication PRIMEIRO (antes de qualquer import que puxe TF) ───────
    # Ver "ORDEM DE BOOT CRÍTICA" no cabeçalho: TF-antes-de-QApplication = segfault.
    app = _create_qapp(args)
    if app is None:
        # Sem binding Qt6 — usa qt_compat só para a mensagem de erro coerente
        # (aqui não há QApplication a proteger, então importar TF é inofensivo).
        from geosteering_ai.gui.qt_compat import QT_IMPORT_ERROR

        sys.stderr.write((QT_IMPORT_ERROR or "Qt6 indisponível.") + "\n")
        return 1

    # ── 2) Demais imports (puxam TF via geosteering_ai/__init__, mas o app já
    #       existe → ordem TF-após-QApplication é segura) ──────────────────────
    from apps.sim_manager.main_window import SM_MainWindow
    from apps.sim_manager.perspectives.datviewer.perspective import DatViewerPerspective
    from apps.sim_manager.perspectives.preferences.perspective import (
        PreferencesPerspective,
    )
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.qt_compat import enforce_c_locale
    from geosteering_ai.gui.shell.context import AppContext
    from geosteering_ai.gui.shell.placeholder import PlaceholderPerspective
    from geosteering_ai.gui.theme import apply_theme

    # Locale C: ponto decimal nos QDoubleSpinBox (paridade c/ o monólito; evita
    # que um locale pt-BR mostre/parseie "1,5" e divirja do CSV Python float()).
    enforce_c_locale(app)
    # Tema profissional (dark, paleta Google Antigravity) — QSS global.
    apply_theme(app)

    ctx = AppContext(app_name=_APP_NAME)
    window = SM_MainWindow(ctx)
    window.add_perspective(SimulationPerspective())
    # Fatia 6e — Preferências (perspectiva real: tema, paths, backend, cache LRU).
    window.add_perspective(PreferencesPerspective())
    # Fatia 6h — Visualizador .dat/.out (somente leitura: tabela 22-col + metadados).
    window.add_perspective(DatViewerPerspective())
    # Scaffolds "em breve" na activity rail (organização visível de todos os recursos).
    for pid, title, glyph, order, roadmap in _SCAFFOLD_PERSPECTIVES:
        window.add_perspective(
            PlaceholderPerspective(
                id=pid, title=title, icon_glyph=glyph, order=order, roadmap=roadmap
            )
        )
    window.resize(1180, 800)  # rail + perspectiva + secondary sidebar
    window.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover — entry-point manual
    raise SystemExit(main())
