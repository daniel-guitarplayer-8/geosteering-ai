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


def main(argv: Optional[List[str]] = None, *, show_startup: bool = False) -> int:
    """Sobe o app MVVM do Simulation Manager.

    Args:
        argv: argumentos (default ``sys.argv``).
        show_startup: se ``True``, exibe o diálogo de projeto (abrir/criar) ANTES da
            janela (PR-3 #7a). Default ``False`` — só o entry-point manual (``__main__``)
            ativa; testes/headless NÃO bloqueiam num diálogo modal.

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

    # ── Teardown defense-in-depth: libera o worker JAX persistente no quit ────
    # O backend jax/auto roda num subprocesso PERSISTENTE (CUDA quente entre runs —
    # gui.services.base). ``aboutToQuit`` é o sinal canônico de encerramento do Qt
    # (dispara mesmo quando o fechamento não passa pelo closeEvent da janela). Import
    # LOCAL: ``release_jax_pool`` só mexe no handle do ProcessPool — NÃO importa JAX na
    # GUI (TLS-safe). Complementa o closeEvent da SM_MainWindow + o atexit de base.
    def _release_jax_pool_safe() -> None:
        try:
            from geosteering_ai.gui.services.base import release_jax_pool

            release_jax_pool()
        except Exception:  # noqa: BLE001 — teardown best-effort; quit nunca falha
            pass

    app.aboutToQuit.connect(_release_jax_pool_safe)

    # ── 2) Demais imports (puxam TF via geosteering_ai/__init__, mas o app já
    #       existe → ordem TF-após-QApplication é segura) ──────────────────────
    from apps.sim_manager.main_window import SM_MainWindow
    from apps.sim_manager.perspectives.datviewer.perspective import DatViewerPerspective
    from apps.sim_manager.perspectives.preferences.perspective import (
        PreferencesPerspective,
    )
    from apps.sim_manager.perspectives.results.perspective import ResultsPerspective
    from apps.sim_manager.perspectives.simulation.experiments_service import (
        ExperimentsService,
    )
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from apps.sim_manager.startup_dialog import ProjectStartupDialog
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

    # ── PR-3 (#7a) — diálogo de projeto no startup (abrir/criar) ─────────────
    # Só no entry-point manual (show_startup=True); testes/headless não bloqueiam.
    # Popula ctx.extras ANTES de registrar a Simulação (que adota o projeto).
    if show_startup:
        service = ExperimentsService()
        dialog = ProjectStartupDialog(service)
        if not dialog.exec():
            return 0  # usuário cancelou/saiu — encerra limpo, sem janela
        ctx.extras["experiments_service"] = service  # reusado pela Simulação
        ctx.extras["project"] = dialog.result_state
        if dialog.result_state is not None:
            ctx.extras["project_dir"] = dialog.result_state.output_dir

    window = SM_MainWindow(ctx)
    # Simulação PRIMEIRO (order=0): builda no boot e publica ctx.extras["results_vm"]
    # ANTES de Resultados (order=1) ser ativada (1 VM, 2 Views — PR-2 #1).
    window.add_perspective(SimulationPerspective())
    # Fatia 6i / PR-2 — Resultados (galeria do ensemble; reusa o ResultsViewModel).
    window.add_perspective(ResultsPerspective())
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
    # ── Warmup do JAX GPU (boot canônico opt-in + config-aware on-select) ─────
    # Cria UM JaxWarmupService COMPARTILHADO em ctx.extras (o boot canônico E o warmup
    # config-aware da perspectiva de simulação reusam o MESMO worker persistente). Só
    # instancia se o jax estiver instalado (find_spec NÃO importa jax na GUI). O boot
    # canônico (pref jax_boot_warmup, default False) é disparado aqui; o config-aware
    # (pref jax_auto_warmup, default True) é disparado pela SimulatorView ao selecionar
    # jax/auto. Best-effort: uma falha aqui nunca impede o app de subir.
    try:
        import importlib.util

        from apps.sim_manager.boot_warmup import schedule_boot_warmup
        from apps.sim_manager.perspectives.preferences.service import PreferencesService

        if (
            importlib.util.find_spec("jax") is not None
            and "jax_warmup_service" not in ctx.extras
        ):
            from geosteering_ai.gui.services.jax_warmup_service import JaxWarmupService

            ctx.extras["jax_warmup_service"] = JaxWarmupService()
        _warmup_on = bool(PreferencesService().load().get("jax_boot_warmup", False))
        schedule_boot_warmup(ctx, window, enabled=_warmup_on)
    except Exception:  # noqa: BLE001 — warmup é otimização; boot nunca falha por isto
        pass
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover — entry-point manual
    raise SystemExit(main(show_startup=True))
