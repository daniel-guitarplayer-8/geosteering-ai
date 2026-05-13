# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_numba_specializations.py                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Guard tests — Numba JIT specializations (Sprint v2.31)     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-12                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Previne regressões da Sprint v2.31:                                    ║
# ║                                                                           ║
# ║    1. ``hmd_tiv`` e ``vmd`` devem ter UMA ÚNICA especialização Numba     ║
# ║       após disparar o caminho JAX (``pure_callback`` → host Numba).     ║
# ║       Antes da v2.31, o caminho JAX passava arrays readonly via         ║
# ║       ``np.asarray``, fazendo Numba gerar duas especializações ``.nbc``║
# ║       (~2 MB extras em disco e +20-40 s no cold-start).                ║
# ║                                                                           ║
# ║    2. ``geosteering_ai.cli.main`` deve definir ``NUMBA_CACHE_DIR`` para   ║
# ║       tmpfs (``/tmp``) quando o usuário não tiver setado, e preservar    ║
# ║       overrides manuais via ``os.environ``.                                ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • docs/reports/warmup_analysis_jit_2026-05-12.md §1.4, §5.1.2, §6.4   ║
# ║    • docs/reports/v2.31_warmup_optimization_2026-05-12.md                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Guard tests para Sprint v2.31 — especializações Numba únicas + cache dir."""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Registrado pela fixture jax_path_triggered para filtrar .nbc de sessões
# anteriores ao fix _to_writeable (v2.31) que persistem no NUMBA_CACHE_DIR tmpfs
_jax_trigger_time: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: dispara o caminho JAX (que estava criando 2ª especialização)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def jax_path_triggered():
    """Dispara ``fields_in_freqs_jax_batch`` para popular Numba signatures.

    Sem esta fixture, ``hmd_tiv.signatures`` e ``vmd.signatures`` podem
    estar vazios (o caminho Numba puro inline-compila e não popula
    signatures separadas). O JAX callback é o único caminho de produção
    que chama os dipolos Numba como ponto de entrada Python.
    """
    global _jax_trigger_time
    pytest.importorskip("jax", reason="JAX necessário para testar callback path")

    from geosteering_ai.simulation._jax.kernel import fields_in_freqs_jax_batch
    from geosteering_ai.simulation.filters import FilterLoader

    filt = FilterLoader().load("werthmuller_201pt")

    rho_h = np.array([1.0, 100.0, 1.0])
    rho_v = np.array([1.0, 200.0, 1.0])
    esp = np.array([5.0])
    positions_z = np.linspace(-1.0, 6.0, 3)
    freqs_hz = np.array([20000.0])

    # Registrar timestamp ANTES da compilação para filtrar .nbc de sessões antigas
    _jax_trigger_time = time.time()

    fields_in_freqs_jax_batch(
        positions_z=positions_z,
        dz_half=0.5,
        r_half=0.0,
        dip_rad=0.0,
        n=3,
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        freqs_hz=freqs_hz,
        krJ0J1=filt.abscissas,
        wJ0=filt.weights_j0,
        wJ1=filt.weights_j1,
        use_native_dipoles=False,
    )
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Testes de especialização única — Sprint v2.31
# ──────────────────────────────────────────────────────────────────────────────
class TestSingleSpecialization:
    """Garante que o caminho JAX gera APENAS 1 especialização Numba (Sprint v2.31).

    A correção em ``_jax/kernel.py::_dipoles_numba_host`` introduziu
    ``_to_writeable`` para normalizar a mutabilidade dos arrays passados
    via ``jax.pure_callback`` a ``hmd_tiv``/``vmd``. Antes da correção,
    duas especializações JIT eram compiladas (mutable=True via Numba puro +
    mutable=False via JAX callback), gerando dois arquivos ``.nbc`` em
    disco (``.1.nbc`` + ``.2.nbc``) e inflando o cold-start em 20-40 s.

    Após a correção, apenas 1 especialização compila — comprovado tanto
    em uso direto (CLI) quanto em pytest. O flag interno do Numba
    (mutable=True vs mutable=False) pode variar com o ambiente, mas o
    COUNT permanece em 1.
    """

    def test_hmd_tiv_single_specialization(self, jax_path_triggered) -> None:
        """Após caminho JAX, hmd_tiv tem exatamente 1 especialização."""
        from geosteering_ai.simulation._numba.dipoles import hmd_tiv

        assert jax_path_triggered  # explicit usage do fixture (lint-friendly)
        n_sigs = len(hmd_tiv.signatures)
        assert n_sigs == 1, (
            f"hmd_tiv tem {n_sigs} especializações (esperado 1). "
            "Sprint v2.31: _to_writeable em _jax/kernel.py::_dipoles_numba_host "
            "deveria garantir mutabilidade uniforme. Se este teste falhou, "
            "verifique se algum caller Python puro passa arrays com flag de "
            "mutabilidade divergente do caminho JAX."
        )

    def test_vmd_single_specialization(self, jax_path_triggered) -> None:
        """Após caminho JAX, vmd tem exatamente 1 especialização."""
        from geosteering_ai.simulation._numba.dipoles import vmd

        assert jax_path_triggered
        n_sigs = len(vmd.signatures)
        assert n_sigs == 1, (
            f"vmd tem {n_sigs} especializações (esperado 1). "
            "Sprint v2.31: _to_writeable em _jax/kernel.py::_dipoles_numba_host "
            "deveria garantir mutabilidade uniforme."
        )

    def test_single_nbc_file_per_function(self, jax_path_triggered) -> None:
        """Nenhuma 2ª especialização ``.nbc`` gerada nesta sessão (anti-regressão).

        Antes do fix ``_to_writeable`` (v2.31): JAX callback criava 2ª
        especialização (mutable=False) → Numba gravava ``.2.nbc`` além do
        ``.1.nbc`` já existente.
        Após o fix: apenas 1 especialização em memória; ``.2.nbc`` não é
        gerado nesta sessão.

        Verificação por timestamp (``_jax_trigger_time``) ignora ``.2.nbc``
        legados de sessões anteriores ao fix no NUMBA_CACHE_DIR tmpfs
        persistente. Numba não recria ``.nbc`` existentes, então só
        arquivos NEW (regressão) teriam timestamp >= trigger.
        """
        from geosteering_ai.simulation._numba.dipoles import hmd_tiv, vmd

        assert jax_path_triggered
        cache_dir = Path(hmd_tiv._cache._cache_path)

        # Verificar que nenhum .2.nbc foi CRIADO nesta sessão
        # (.2.nbc = 2ª especialização = sintoma da regressão pré-fix)
        new_hmd_v2 = [
            f
            for f in cache_dir.glob("dipoles.hmd_tiv-*.2.nbc")
            if f.stat().st_mtime >= _jax_trigger_time
        ]
        new_vmd_v2 = [
            f
            for f in cache_dir.glob("dipoles.vmd-*.2.nbc")
            if f.stat().st_mtime >= _jax_trigger_time
        ]
        assert len(new_hmd_v2) == 0, (
            f"hmd_tiv gerou 2ª especialização nesta sessão: {new_hmd_v2} — "
            "Regressão v2.31? Verificar _to_writeable em _jax/kernel.py"
        )
        assert len(new_vmd_v2) == 0, (
            f"vmd gerou 2ª especialização nesta sessão: {new_vmd_v2} — "
            "Regressão v2.31? Verificar _to_writeable em _jax/kernel.py"
        )
        # Confirmação em memória (diagnóstico complementar)
        n_hmd = len(hmd_tiv.signatures)
        n_vmd = len(vmd.signatures)
        assert n_hmd == 1, f"hmd_tiv tem {n_hmd} especializações (esperado 1)"
        assert n_vmd == 1, f"vmd tem {n_vmd} especializações (esperado 1)"


# ──────────────────────────────────────────────────────────────────────────────
# Testes NUMBA_CACHE_DIR — Sprint v2.31
# ──────────────────────────────────────────────────────────────────────────────
class TestNumbaCacheDir:
    """Garante que cli/main.py configure NUMBA_CACHE_DIR em tmpfs."""

    def _reimport_cli_main(self) -> None:
        """Força reimport de geosteering_ai.cli.main para reexecutar setup."""
        mod_name = "geosteering_ai.cli.main"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        importlib.import_module(mod_name)

    def test_numba_cache_dir_set_after_cli_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importar cli.main sem NUMBA_CACHE_DIR pré-setado define o default tmpfs."""
        monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
        self._reimport_cli_main()

        cache_dir = os.environ.get("NUMBA_CACHE_DIR", "")
        assert cache_dir, "NUMBA_CACHE_DIR não foi setado por cli/main.py"
        assert cache_dir.endswith("geosteering_numba_cache"), (
            f"NUMBA_CACHE_DIR esperado terminar em 'geosteering_numba_cache', "
            f"recebido: {cache_dir!r}"
        )
        assert os.path.isdir(
            cache_dir
        ), f"Diretório NUMBA_CACHE_DIR={cache_dir!r} não foi criado em disco"

    def test_numba_cache_dir_user_override_preserved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Valor pré-existente de NUMBA_CACHE_DIR deve ser preservado."""
        user_dir = str(tmp_path / "user_custom_numba_cache")
        os.makedirs(user_dir, exist_ok=True)
        monkeypatch.setenv("NUMBA_CACHE_DIR", user_dir)
        self._reimport_cli_main()

        assert os.environ["NUMBA_CACHE_DIR"] == user_dir, (
            "cli/main.py sobrescreveu NUMBA_CACHE_DIR do usuário "
            f"(esperado {user_dir!r}, atual {os.environ.get('NUMBA_CACHE_DIR')!r})"
        )
