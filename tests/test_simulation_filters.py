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

# Anderson 801pt, linhas 3569-4173 de filtersv2.f08:
# Valores em múltiplos pontos do array para detectar regressões no parser
# em arrays longos (801 valores em bloco único multilinha — Dialeto B).
AND_ABSC_0 = 8.9170998013276122e-14  # absc[0]  (primeiro)
AND_ABSC_400 = 2.0989539161478380e04  # absc[400] (meio, elemento 401 Fortran)
AND_ABSC_800 = 4.9406282763106685e21  # absc[800] (último, elemento 801 Fortran)
AND_WJ0_0 = 0.21035620538389819885e-28  # weights_j0[0]   (primeiro)
AND_WJ0_400 = 0.42878525031129105819e-05  # weights_j0[400] (meio)
AND_WJ0_800 = 0.90953607290146280299e-10  # weights_j0[800] (último)
AND_WJ1_0 = -0.23779001100582381051e-28  # weights_j1[0]   (primeiro)
AND_WJ1_400 = 0.56033096589579911757e-06  # weights_j1[400] (meio)
AND_WJ1_800 = 0.72149205056137611593e-27  # weights_j1[800] (último)


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────
# O cache de `FilterLoader` é classe-level (compartilhado entre instâncias e
# entre testes). Testes que dependem do estado inicial do cache (vazio ou
# populado) devem explicitamente chamar `clear_cache()` no setup. Por isso,
# usamos escopo `function` (cada teste recebe sua própria instância) e o
# `autouse=True` na fixture que limpa o cache, garantindo ordem independente
# e compatibilidade com `pytest-randomly`.
@pytest.fixture
def loader() -> FilterLoader:
    """Retorna uma instância limpa de `FilterLoader` por teste.

    Returns:
        FilterLoader configurado para o diretório default, com cache
        classe-level limpo. Cada função-teste recebe uma instância nova.
    """
    return FilterLoader()


@pytest.fixture(autouse=True)
def _clean_filter_cache() -> None:
    """Limpa o cache classe-level antes de cada teste (autouse).

    Note:
        Declarado como `autouse=True` para rodar antes de todos os testes
        do módulo sem precisar ser referenciado explicitamente. Isso torna
        a suíte independente de ordem de execução (`pytest-randomly` safe).
    """
    FilterLoader().clear_cache()


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE PRESENÇA E METADATA
# ──────────────────────────────────────────────────────────────────────────────
# Conjunto mínimo de filtros conhecidos que SEMPRE devem estar presentes no
# catálogo. É definido como `>= FILTROS_CONHECIDOS` (subconjunto) para
# permitir que filtros futuros (ex.: Key 401pt, 2001pt Werthmüller) sejam
# adicionados sem quebrar o CI — um teste novo é adicionado ao invés de
# editar a contagem hardcoded.
_FILTROS_CONHECIDOS: frozenset[str] = frozenset(
    {
        "werthmuller_201pt",
        "kong_61pt",
        "anderson_801pt",
    }
)


class TestFilterArtifactsExist:
    """Garante que os filtros conhecidos estão presentes e listáveis."""

    def test_known_filters_available(self, loader: FilterLoader) -> None:
        """Os 3 filtros conhecidos devem estar instalados e listáveis.

        Note:
            Usa `issubset` em vez de igualdade: se um 4° filtro for
            adicionado ao catálogo no futuro, este teste continua
            passando (a suíte cresce com testes adicionais, não com
            edição de contagem hardcoded).
        """
        available = set(loader.available())
        assert _FILTROS_CONHECIDOS.issubset(
            available
        ), f"Filtros faltantes: {_FILTROS_CONHECIDOS - available}"

    def test_catalog_contains_known_filters(self) -> None:
        """O catálogo interno contém (ao menos) os filtros conhecidos.

        Note:
            Substitui o antigo `assert len(_FILTER_CATALOG) == 3`, que
            acoplaria o CI ao número exato de filtros e forçaria edição
            do teste ao adicionar novos. A verificação por subconjunto
            é forward-compatible.
        """
        assert _FILTROS_CONHECIDOS.issubset(set(_FILTER_CATALOG.keys())), (
            f"Catálogo não contém filtros conhecidos: "
            f"{_FILTROS_CONHECIDOS - set(_FILTER_CATALOG.keys())}"
        )


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
    """Compara valores específicos do Anderson 801pt com o Fortran.

    Note:
        Cobertura de spot-check expandida (vs Sprint 1.1 inicial) para
        incluir primeiro, meio e último em abscissas, wJ0 e wJ1 — total
        de 9 valores de referência. Isso detecta regressões no parser
        em acumulação de arrays longos (801 valores em bloco único
        multilinha do Dialeto B).
    """

    @pytest.fixture
    def filt(self, loader: FilterLoader) -> HankelFilter:
        return loader.load("anderson_801pt")

    def test_npt(self, filt: HankelFilter) -> None:
        assert filt.npt == 801
        assert filt.abscissas.shape == (801,)
        assert filt.weights_j0.shape == (801,)
        assert filt.weights_j1.shape == (801,)

    def test_fortran_filter_type(self, filt: HankelFilter) -> None:
        assert filt.fortran_filter_type == 2

    # ── Abscissas: primeiro, meio (índice 400), último (índice 800) ──
    def test_absc_first(self, filt: HankelFilter) -> None:
        assert filt.abscissas[0] == AND_ABSC_0

    def test_absc_middle(self, filt: HankelFilter) -> None:
        """absc[400] detecta regressões no meio do array (1º terço)."""
        assert filt.abscissas[400] == AND_ABSC_400

    def test_absc_last(self, filt: HankelFilter) -> None:
        """absc[800] detecta se o parser completou todos os 801 valores."""
        assert filt.abscissas[-1] == AND_ABSC_800

    # ── weights_j0: primeiro, meio, último ──
    def test_wj0_first(self, filt: HankelFilter) -> None:
        assert filt.weights_j0[0] == AND_WJ0_0

    def test_wj0_middle(self, filt: HankelFilter) -> None:
        assert filt.weights_j0[400] == AND_WJ0_400

    def test_wj0_last(self, filt: HankelFilter) -> None:
        assert filt.weights_j0[-1] == AND_WJ0_800

    # ── weights_j1: primeiro, meio, último ──
    def test_wj1_first(self, filt: HankelFilter) -> None:
        assert filt.weights_j1[0] == AND_WJ1_0

    def test_wj1_middle(self, filt: HankelFilter) -> None:
        assert filt.weights_j1[400] == AND_WJ1_400

    def test_wj1_last(self, filt: HankelFilter) -> None:
        assert filt.weights_j1[-1] == AND_WJ1_800

    # ── Validações semânticas ──
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
