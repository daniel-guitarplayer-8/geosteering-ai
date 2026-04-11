# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_filters.py                                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto    : Geosteering AI v2.0                                         ║
# ║  Subsistema : Simulador Python — Validação de Filtros Hankel              ║
# ║  Autor      : Daniel Leal                                                 ║
# ║  Criação    : 2026-04-11 (Sprint 1.1)                                    ║
# ║  Framework  : pytest                                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garantir que os artefatos .npz dos filtros Hankel estão:              ║
# ║      1. Presentes e não corrompidos.                                     ║
# ║      2. Bit-idênticos aos valores literais em filtersv2.f08 (spot-check).║
# ║      3. Sincronizados com o hash SHA-256 do arquivo Fortran.            ║
# ║      4. Consumíveis via FilterLoader (alias, canônico, numérico).       ║
# ║      5. Imutáveis (arrays read-only, dataclass frozen).                  ║
# ║                                                                           ║
# ║  ESTRATÉGIA DE VALIDAÇÃO                                                  ║
# ║    Comparamos valores específicos (primeiro, meio, último) extraídos    ║
# ║    contra os valores HARDCODED no arquivo Fortran. Isso garante que:    ║
# ║      • O parser regex funcionou corretamente.                           ║
# ║      • A conversão D→E não introduziu erros de arredondamento.         ║
# ║      • A ordem de acumulação dos blocos Kong (Dialeto A) está correta. ║
# ║                                                                           ║
# ║    Os valores esperados abaixo foram copiados MANUALMENTE do Fortran    ║
# ║    (sem uso de parser) para servir como "ground truth" independente.    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de integração para o subsistema de filtros Hankel.

Estes testes são rápidos (< 1s) e devem rodar em CPU sem dependências
externas além de NumPy + pytest + o próprio pacote `geosteering_ai`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geosteering_ai.simulation.filters import FilterLoader, HankelFilter
from geosteering_ai.simulation.filters.loader import _FILTER_CATALOG

# ──────────────────────────────────────────────────────────────────────────────
# GROUND-TRUTH: valores copiados manualmente de Fortran_Gerador/filtersv2.f08
# ──────────────────────────────────────────────────────────────────────────────
# Estes valores são literais Fortran com notação "D+XX" convertida para
# float Python. Qualquer divergência em bit-level entre Python e estes
# valores indica regressão no parser ou corrupção dos .npz.
#
# Kong 61pt, linhas 13-77 de filtersv2.f08:
KONG_ABSC_0 = 2.3517745856009100e-02  # absc( 1 )
KONG_ABSC_30 = 1.0000000000000000e00  # absc(31) — valor central, kr=1.0
KONG_ABSC_60 = 4.2521082000062783e01  # absc(61) — último
KONG_WJ0_0 = 1.4463210615326699e02  # wJ0( 1 )
KONG_WJ0_60 = 6.7792635718095777e-06  # wJ0(61) — último
KONG_WJ1_0 = 4.6440396425864918e01  # wJ1( 1 )
KONG_WJ1_60 = 1.8788896009128770e-05  # wJ1(61) — último

# Werthmüller 201pt, linhas 1399-1553 de filtersv2.f08:
WER_ABSC_0 = 8.653980893285999343e-04  # primeiro
WER_ABSC_200 = 9.368804274491759543e01  # último
WER_WJ0_0 = 2.940900904253498815e00  # primeiro
WER_WJ0_200 = -2.645962918790746080e-08  # último
WER_WJ1_0 = -2.594301879688918743e-03  # primeiro
WER_WJ1_200 = 3.780807126974975489e-08  # último

# Anderson 801pt, linhas 3569-... de filtersv2.f08:
AND_ABSC_0 = 8.9170998013276122e-14  # primeiro


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def loader() -> FilterLoader:
    """Retorna uma instância de `FilterLoader` com cache limpo.

    Returns:
        FilterLoader configurado para o diretório default.
    """
    loader = FilterLoader()
    loader.clear_cache()
    return loader


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE PRESENÇA E METADATA
# ──────────────────────────────────────────────────────────────────────────────
class TestFilterArtifactsExist:
    """Garante que os 3 artefatos .npz estão presentes e listáveis."""

    def test_three_filters_available(self, loader: FilterLoader) -> None:
        """Todos os 3 filtros do catálogo devem estar instalados."""
        available = loader.available()
        assert set(available) == {
            "werthmuller_201pt",
            "kong_61pt",
            "anderson_801pt",
        }

    def test_filter_catalog_has_three_entries(self) -> None:
        """O catálogo interno expõe exatamente 3 filtros."""
        assert len(_FILTER_CATALOG) == 3


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE BIT-EXACTNESS (spot-check vs Fortran literal)
# ──────────────────────────────────────────────────────────────────────────────
class TestKong61BitExactness:
    """Compara valores específicos do Kong 61pt com o Fortran literal."""

    @pytest.fixture
    def filt(self, loader: FilterLoader) -> HankelFilter:
        return loader.load("kong_61pt")

    def test_npt(self, filt: HankelFilter) -> None:
        assert filt.npt == 61
        assert filt.abscissas.shape == (61,)
        assert filt.weights_j0.shape == (61,)
        assert filt.weights_j1.shape == (61,)

    def test_fortran_filter_type(self, filt: HankelFilter) -> None:
        """Kong corresponde a filter_type=1 no Fortran."""
        assert filt.fortran_filter_type == 1

    def test_absc_first(self, filt: HankelFilter) -> None:
        """absc(1) deve bater bit-a-bit com o valor Fortran."""
        assert filt.abscissas[0] == KONG_ABSC_0

    def test_absc_middle(self, filt: HankelFilter) -> None:
        """absc(31) deve ser 1.0 (ponto central, kr=1)."""
        assert filt.abscissas[30] == KONG_ABSC_30

    def test_absc_last(self, filt: HankelFilter) -> None:
        """absc(61) deve bater bit-a-bit."""
        assert filt.abscissas[-1] == KONG_ABSC_60

    def test_wj0_first(self, filt: HankelFilter) -> None:
        assert filt.weights_j0[0] == KONG_WJ0_0

    def test_wj0_last(self, filt: HankelFilter) -> None:
        assert filt.weights_j0[-1] == KONG_WJ0_60

    def test_wj1_first(self, filt: HankelFilter) -> None:
        assert filt.weights_j1[0] == KONG_WJ1_0

    def test_wj1_last(self, filt: HankelFilter) -> None:
        assert filt.weights_j1[-1] == KONG_WJ1_60


class TestWerthmuller201BitExactness:
    """Compara valores específicos do Werthmüller 201pt com o Fortran."""

    @pytest.fixture
    def filt(self, loader: FilterLoader) -> HankelFilter:
        return loader.load("werthmuller_201pt")

    def test_npt(self, filt: HankelFilter) -> None:
        assert filt.npt == 201
        assert filt.abscissas.shape == (201,)

    def test_fortran_filter_type(self, filt: HankelFilter) -> None:
        """Werthmüller corresponde a filter_type=0 (default)."""
        assert filt.fortran_filter_type == 0

    def test_absc_first(self, filt: HankelFilter) -> None:
        assert filt.abscissas[0] == WER_ABSC_0

    def test_absc_last(self, filt: HankelFilter) -> None:
        assert filt.abscissas[-1] == WER_ABSC_200

    def test_wj0_first(self, filt: HankelFilter) -> None:
        assert filt.weights_j0[0] == WER_WJ0_0

    def test_wj0_last(self, filt: HankelFilter) -> None:
        assert filt.weights_j0[-1] == WER_WJ0_200

    def test_wj1_first(self, filt: HankelFilter) -> None:
        assert filt.weights_j1[0] == WER_WJ1_0

    def test_wj1_last(self, filt: HankelFilter) -> None:
        assert filt.weights_j1[-1] == WER_WJ1_200


class TestAnderson801BitExactness:
    """Compara valores específicos do Anderson 801pt com o Fortran."""

    @pytest.fixture
    def filt(self, loader: FilterLoader) -> HankelFilter:
        return loader.load("anderson_801pt")

    def test_npt(self, filt: HankelFilter) -> None:
        assert filt.npt == 801

    def test_fortran_filter_type(self, filt: HankelFilter) -> None:
        assert filt.fortran_filter_type == 2

    def test_absc_first(self, filt: HankelFilter) -> None:
        assert filt.abscissas[0] == AND_ABSC_0

    def test_absc_strictly_increasing(self, filt: HankelFilter) -> None:
        """Validação semântica: abscissas devem ser crescentes."""
        assert np.all(np.diff(filt.abscissas) > 0)

    def test_absc_strictly_positive(self, filt: HankelFilter) -> None:
        assert np.all(filt.abscissas > 0)


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE API (aliases, cache, imutabilidade)
# ──────────────────────────────────────────────────────────────────────────────
class TestFilterLoaderAPI:
    """Testa a API pública do FilterLoader."""

    def test_load_by_canonical_name(self, loader: FilterLoader) -> None:
        filt = loader.load("werthmuller_201pt")
        assert filt.name == "werthmuller_201pt"

    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("werthmuller", "werthmuller_201pt"),
            ("wer", "werthmuller_201pt"),
            ("kong", "kong_61pt"),
            ("anderson", "anderson_801pt"),
            ("and", "anderson_801pt"),
            ("0", "werthmuller_201pt"),  # filter_type numérico
            ("1", "kong_61pt"),
            ("2", "anderson_801pt"),
        ],
    )
    def test_load_by_alias(
        self,
        loader: FilterLoader,
        alias: str,
        canonical: str,
    ) -> None:
        filt = loader.load(alias)
        assert filt.name == canonical

    def test_load_unknown_raises(self, loader: FilterLoader) -> None:
        with pytest.raises(KeyError, match="desconhecido"):
            loader.load("filtro_inexistente_xyz")

    def test_cache_returns_same_instance(self, loader: FilterLoader) -> None:
        """Cache classe-level: chamadas subsequentes retornam o mesmo obj."""
        a = loader.load("werthmuller_201pt")
        b = loader.load("werthmuller_201pt")
        assert a is b

    def test_cache_cross_instance(self, loader: FilterLoader) -> None:
        """Duas instâncias de FilterLoader compartilham cache classe-level."""
        a = loader.load("kong_61pt")
        b = FilterLoader().load("kong_61pt")
        assert a is b

    def test_clear_cache(self, loader: FilterLoader) -> None:
        """clear_cache() força re-leitura do disco."""
        a = loader.load("kong_61pt")
        loader.clear_cache()
        b = loader.load("kong_61pt")
        # Após clear, nova instância é carregada (objetos distintos).
        assert a is not b
        # Mas os valores numéricos são idênticos (bit-a-bit).
        assert np.array_equal(a.abscissas, b.abscissas)
        assert np.array_equal(a.weights_j0, b.weights_j0)
        assert np.array_equal(a.weights_j1, b.weights_j1)


class TestImmutability:
    """Verifica que os arrays retornados são read-only."""

    @pytest.fixture
    def filt(self, loader: FilterLoader) -> HankelFilter:
        return loader.load("kong_61pt")

    def test_abscissas_read_only(self, filt: HankelFilter) -> None:
        with pytest.raises(ValueError):
            filt.abscissas[0] = 999.0

    def test_wj0_read_only(self, filt: HankelFilter) -> None:
        with pytest.raises(ValueError):
            filt.weights_j0[0] = 999.0

    def test_wj1_read_only(self, filt: HankelFilter) -> None:
        with pytest.raises(ValueError):
            filt.weights_j1[0] = 999.0

    def test_dataclass_frozen(self, filt: HankelFilter) -> None:
        """HankelFilter é frozen — não é possível atribuir novos campos."""
        with pytest.raises(Exception):  # FrozenInstanceError (dataclasses)
            filt.name = "outro_nome"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE SINCRONIA COM FORTRAN
# ──────────────────────────────────────────────────────────────────────────────
class TestSourceSynchronization:
    """Garante que os .npz estão sincronizados com o filtersv2.f08."""

    def _compute_fortran_hash(self) -> str:
        """Computa SHA-256 do filtersv2.f08 para comparação."""
        import hashlib

        root = Path(__file__).resolve().parent.parent
        src = root / "Fortran_Gerador" / "filtersv2.f08"
        h = hashlib.sha256()
        with src.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @pytest.mark.parametrize(
        "filter_name",
        ["werthmuller_201pt", "kong_61pt", "anderson_801pt"],
    )
    def test_source_hash_synchronized(
        self,
        loader: FilterLoader,
        filter_name: str,
    ) -> None:
        """O SHA-256 registrado no .npz deve bater com o Fortran atual.

        Se este teste falhar, significa que `filtersv2.f08` foi alterado
        sem re-rodar `scripts/extract_hankel_weights.py`. Execute o script
        para regenerar os .npz e re-rode os testes.
        """
        filt = loader.load(filter_name)
        current_hash = self._compute_fortran_hash()
        assert filt.source_sha256 == current_hash, (
            f"{filter_name}.npz está dessincronizado com filtersv2.f08. "
            f"Execute: python scripts/extract_hankel_weights.py"
        )

    def test_verify_mode_returns_zero(self) -> None:
        """O comando --verify do extrator deve retornar código 0."""
        import subprocess
        import sys

        root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "scripts/extract_hankel_weights.py", "--verify"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"extract_hankel_weights.py --verify falhou:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
