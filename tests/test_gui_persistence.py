# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_gui_persistence.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0007-gui-persistence                                       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — persistência (atômico · SessionDocument · shim)      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre os ACs da spec 0007: escrita ATÔMICA (incl. crash-resistance),   ║
# ║    SessionDocument (round-trip · forward-compat · sem pickle), extração   ║
# ║    Strangler-Fig (shim identidade) + hardening do SnapshotPersistThread,  ║
# ║    e a fronteira de import (atomic/session puros, sem Qt).                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0007 — persistência gui/ (atômico · SessionDocument · shim)."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Qt headless — só o teste do SnapshotPersistThread (QThread) precisa; os demais
# são PUROS (sem Qt). Setado no import (antes da coleta).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ════════════════════════════════════════════════════════════════════════════
# RF-1 — escrita atômica (pura)
# ════════════════════════════════════════════════════════════════════════════
def test_atomic_write_creates_file_and_parent(tmp_path):
    """AC-1.1 — atomic_write_text grava o conteúdo e cria o diretório pai."""
    from geosteering_ai.gui.persistence.atomic import atomic_write_text

    target = tmp_path / "sub" / "dir" / "x.session"
    atomic_write_text(str(target), "conteúdo atômico")
    assert target.read_text(encoding="utf-8") == "conteúdo atômico"


def test_atomic_write_overwrites_existing(tmp_path):
    """AC-1.1 — sobrescreve um arquivo existente (caso comum de re-save)."""
    from geosteering_ai.gui.persistence.atomic import atomic_write_text

    target = tmp_path / "x.session"
    target.write_text("ANTIGO", encoding="utf-8")
    atomic_write_text(str(target), "NOVO")
    assert target.read_text(encoding="utf-8") == "NOVO"


def test_atomic_write_crash_preserves_old_file(tmp_path, monkeypatch):
    """AC-1.2 — falha a meio (os.replace) → arquivo ANTIGO intacto, sem .tmp resíduo."""
    import geosteering_ai.gui.persistence.atomic as atomic_mod

    target = tmp_path / "x.session"
    target.write_text("ANTIGO", encoding="utf-8")

    def boom(src, dst):
        raise OSError("crash simulado no os.replace")

    monkeypatch.setattr(atomic_mod.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_mod.atomic_write_text(str(target), "NOVO-NUNCA-GRAVADO")

    # invariante de atomicidade: o destino antigo permanece intacto…
    assert target.read_text(encoding="utf-8") == "ANTIGO"
    # …e nenhum temporário .tmp-* sobrou no diretório
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]
    assert leftovers == [], f"temporários residuais: {leftovers}"


def test_atomic_write_preserves_existing_permissions(tmp_path):
    """AC-1.4 — sobrescrever um arquivo PRESERVA seu modo (mkstemp 0600 não regride 0644)."""
    import stat

    from geosteering_ai.gui.persistence.atomic import atomic_write_text

    target = tmp_path / "x.session"
    target.write_text("ANTIGO", encoding="utf-8")
    os.chmod(target, 0o644)
    atomic_write_text(str(target), "NOVO")
    mode = stat.S_IMODE(os.stat(target).st_mode)
    assert mode == 0o644, f"modo regrediu para {oct(mode)} (esperado 0o644)"


def test_atomic_write_fdopen_failure_no_fd_leak(tmp_path, monkeypatch):
    """code-review: se os.fdopen falhar, o fd do mkstemp É FECHADO (sem leak) + sem .tmp."""
    import geosteering_ai.gui.persistence.atomic as atomic_mod

    captured: dict[str, int] = {}
    real_mkstemp = atomic_mod.tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        captured["fd"] = fd
        return fd, path

    def boom(*args, **kwargs):  # fdopen NÃO assume o fd (como uma falha real)
        raise OSError("falha simulada no fdopen")

    monkeypatch.setattr(atomic_mod.tempfile, "mkstemp", spy_mkstemp)
    monkeypatch.setattr(atomic_mod.os, "fdopen", boom)
    with pytest.raises(OSError):
        atomic_mod.atomic_write_text(str(tmp_path / "x.session"), "NOVO")

    # o fix fechou o fd → fechá-lo de novo deve falhar (sem o fix, o fd vazaria
    # e este os.close teria sucesso, falhando o teste).
    with pytest.raises(OSError):
        os.close(captured["fd"])
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]
    assert leftovers == [], f"temporários residuais: {leftovers}"


# ════════════════════════════════════════════════════════════════════════════
# RF-2 — SessionDocument (puro, sem pickle)
# ════════════════════════════════════════════════════════════════════════════
def test_session_document_json_roundtrip():
    """AC-2.1 — to_json/from_json preservam schema_version + data (round-trip)."""
    from geosteering_ai.gui.persistence.session import SessionDocument

    doc = SessionDocument(data={"perspective": "simulation", "param_ç": 3.14})
    back = SessionDocument.from_json(doc.to_json())
    assert back.schema_version == 1
    assert back.data == {"perspective": "simulation", "param_ç": 3.14}


def test_session_document_forward_compat_preserves_unknown_keys():
    """AC-2.2 — chave top-level desconhecida (versão futura) é PRESERVADA, sem erro."""
    from geosteering_ai.gui.persistence.session import SessionDocument

    raw = '{"schema_version": 2, "data": {"a": 1}, "future_top": 99}'
    doc = SessionDocument.from_json(raw)
    assert doc.schema_version == 2
    assert doc.data == {"a": 1}
    # a chave futura sobrevive ao round-trip (re-emitida em to_json)
    assert "future_top" in doc.to_json()


def test_session_document_save_load_atomic(tmp_path):
    """AC-2.3 — save/load round-trip (via escrita atômica)."""
    from geosteering_ai.gui.persistence.session import SessionDocument

    path = tmp_path / "app.session"
    SessionDocument(data={"backend": "matplotlib"}).save(path)
    loaded = SessionDocument.load(path)
    assert loaded.data == {"backend": "matplotlib"}


def test_session_document_invalid_json_raises():
    """from_json de um JSON não-objeto (ex.: lista) levanta ValueError claro."""
    from geosteering_ai.gui.persistence.session import SessionDocument

    with pytest.raises(ValueError):
        SessionDocument.from_json("[1, 2, 3]")


def test_session_module_has_no_pickle():
    """AC-2.4 — o módulo session NÃO usa pickle (segurança: pickle = RCE)."""
    import geosteering_ai.gui.persistence.session as session_mod

    src = inspect.getsource(session_mod)
    # checa USO real (import/chamada), não a palavra na documentação
    assert "import pickle" not in src
    assert "pickle.load" not in src
    assert "pickle.dump" not in src
    assert "import json" in src  # serialização é JSON


def test_session_document_from_json_non_dict_data_raises():
    """from_json com 'data' não-dict (ex.: "data": 5) levanta ValueError claro."""
    from geosteering_ai.gui.persistence.session import SessionDocument

    with pytest.raises(ValueError):
        SessionDocument.from_json('{"schema_version": 1, "data": 5}')


def test_session_document_from_json_non_int_schema_version_raises():
    """code-review: from_json com schema_version não-int (ex.: "2") levanta ValueError."""
    from geosteering_ai.gui.persistence.session import SessionDocument

    with pytest.raises(ValueError):
        SessionDocument.from_json('{"schema_version": "2", "data": {}}')


def test_session_document_extra_cannot_override_envelope():
    """code-review: chave reservada em `extra` NÃO sobrescreve o envelope no to_json."""
    import json

    from geosteering_ai.gui.persistence.session import SessionDocument

    doc = SessionDocument(schema_version=1, data={"real": 1})
    doc.extra = {"schema_version": 999, "data": {"hijack": True}}  # malicioso/acidental
    payload = json.loads(doc.to_json())
    assert payload["schema_version"] == 1  # envelope canônico vence
    assert payload["data"] == {"real": 1}


def test_session_document_post_init_rejects_non_dict():
    """__post_init__ rejeita data não-dict na CONSTRUÇÃO (fail-fast, não só no save)."""
    from geosteering_ai.gui.persistence.session import SessionDocument

    with pytest.raises(TypeError):
        SessionDocument(data=42)  # type: ignore[arg-type]


def test_session_document_non_serializable_data_raises(tmp_path):
    """data com valor não-JSON (ex.: set) → TypeError no to_json/save (fail-fast)."""
    from geosteering_ai.gui.persistence.session import SessionDocument

    doc = SessionDocument(data={"bad": {1, 2, 3}})  # set não é JSON-serializável
    with pytest.raises(TypeError):
        doc.save(tmp_path / "x.session")


# ════════════════════════════════════════════════════════════════════════════
# RF-3/RF-4 — extração + shim + hardening
# ════════════════════════════════════════════════════════════════════════════
def test_package_qt_free_api():
    """AC-3.1 — a API Qt-free importa do pacote gui.persistence."""
    from geosteering_ai.gui.persistence import (  # noqa: F401
        LRUPlotCache,
        SessionDocument,
        atomic_write_text,
        default_max_bytes,
    )


def test_shim_plot_cache_identity():
    """AC-3.2 — sm_plot_cache (legado) re-exporta os MESMOS objetos de gui.persistence."""
    from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache as GuiLRU
    from geosteering_ai.gui.persistence.plot_cache import default_max_bytes as GuiDMB
    from geosteering_ai.simulation.tests.sm_plot_cache import LRUPlotCache as ShimLRU
    from geosteering_ai.simulation.tests.sm_plot_cache import (
        default_max_bytes as ShimDMB,
    )

    assert ShimLRU is GuiLRU
    assert ShimDMB is GuiDMB


def test_shim_snapshot_persist_identity():
    """AC-3.2 — sm_snapshot_persist (legado) re-exporta o MESMO SnapshotPersistThread."""
    from geosteering_ai.gui.persistence.snapshot import SnapshotPersistThread as GuiSPT
    from geosteering_ai.simulation.tests.sm_snapshot_persist import (
        SnapshotPersistThread as ShimSPT,
    )

    assert ShimSPT is GuiSPT


def test_snapshot_run_uses_atomic_write():
    """AC-3.3 — guard de hardening: run() usa atomic_write_text (não open().write())."""
    from geosteering_ai.gui.persistence import snapshot

    src = inspect.getsource(snapshot.SnapshotPersistThread.run)
    assert "atomic_write_text(" in src
    # não há mais escrita direta: nenhuma chamada `open(... "w" ...).write()`
    # (a menção em comentário é ignorada checando o padrão de CHAMADA real)
    assert "open(self._target_path" not in src


# ════════════════════════════════════════════════════════════════════════════
# RF-4 — hardening exercitado de verdade (Qt, offscreen)
# ════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def offscreen_app():
    """QApplication headless (offscreen) — sem necessidade de DISPLAY/xvfb."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from geosteering_ai.gui.qt_compat import QT_AVAILABLE, QtWidgets

    if not QT_AVAILABLE:
        pytest.skip("nenhum binding Qt6 no ambiente")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.gui
def test_snapshot_thread_writes_atomically(offscreen_app, tmp_path):
    """AC-3.3 (real) — SnapshotPersistThread.run() grava o JSON de forma atômica."""
    from geosteering_ai.gui.persistence.snapshot import SnapshotPersistThread

    target = tmp_path / "deep" / "exp.json"
    thread = SnapshotPersistThread('{"a": 1}', str(target))
    thread.run()  # síncrono (não .start()) — executa a escrita atômica
    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    assert [f for f in os.listdir(target.parent) if f.startswith(".tmp-")] == []


# ════════════════════════════════════════════════════════════════════════════
# RF-5 — fronteira de import (atomic/session puros, sem Qt)
# ════════════════════════════════════════════════════════════════════════════
def test_pure_persistence_modules_do_not_import_qt():
    """AC-5.1 — importar atomic/session/pacote NÃO importa PyQt6/PySide6."""
    code = (
        "import sys\n"
        "import geosteering_ai.gui.persistence  # Qt-free (não puxa snapshot)\n"
        "import geosteering_ai.gui.persistence.atomic\n"
        "import geosteering_ai.gui.persistence.session\n"
        "bad = [m for m in ('PyQt6', 'PySide6') if m in sys.modules]\n"
        "assert not bad, f'persistência pura puxou Qt: {bad}'\n"
        "print('PURE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "PURE_OK" in proc.stdout


def test_core_does_not_import_gui_persistence():
    """AC-5.2 — o core (simulation/…) não importa gui.persistence (fronteira)."""
    code = (
        "import sys\n"
        "import geosteering_ai.simulation.dispatch  # noqa\n"
        "import geosteering_ai.simulation.config  # noqa\n"
        "bad = 'geosteering_ai.gui.persistence' in sys.modules\n"
        "assert not bad, 'core puxou gui.persistence'\n"
        "print('CORE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "CORE_OK" in proc.stdout
