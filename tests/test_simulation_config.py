# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_config.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto    : Geosteering AI v2.0                                         ║
# ║  Subsistema : Simulador Python — SimulationConfig (Sprint 1.2)           ║
# ║  Autor      : Daniel Leal                                                 ║
# ║  Criação    : 2026-04-11                                                 ║
# ║  Framework  : pytest                                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garantir que o dataclass SimulationConfig:                            ║
# ║      1. Constrói com defaults válidos (errata imutável).                ║
# ║      2. Valida ranges numéricos (frequência, spacing, posições).        ║
# ║      3. Valida enums (backend, dtype, device, hankel_filter).           ║
# ║      4. Rejeita conflitos mútuos (fortran_f2py+gpu, numba+gpu, ...).    ║
# ║      5. Suporta presets (default, high_precision, production_gpu, ...).║
# ║      6. Faz roundtrip dict ↔ YAML preservando identidade.              ║
# ║      7. É imutável (frozen dataclass).                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de `SimulationConfig` — Sprint 1.2."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from geosteering_ai.simulation import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
class TestDefaults:
    """Garante que o preset default corresponde à errata imutável."""

    def test_instantiation_with_defaults(self) -> None:
        """Construtor sem argumentos não deve levantar nada."""
        cfg = SimulationConfig()
        assert cfg is not None

    def test_default_frequency(self) -> None:
        """Errata v2.0: frequência default 20000.0 Hz."""
        assert SimulationConfig().frequency_hz == 20000.0

    def test_default_tr_spacing(self) -> None:
        """Errata v2.0: espaçamento TR default 1.0 m."""
        assert SimulationConfig().tr_spacing_m == 1.0

    def test_default_n_positions(self) -> None:
        """Errata v2.0: 600 posições (inv0Dip 0 graus)."""
        assert SimulationConfig().n_positions == 600

    def test_default_backend_is_fortran(self) -> None:
        """Decisão #5: backend default permanece fortran_f2py até Fase 6."""
        assert SimulationConfig().backend == "fortran_f2py"

    def test_default_dtype_is_complex128(self) -> None:
        """Decisão #2: precisão default complex128."""
        assert SimulationConfig().dtype == "complex128"

    def test_default_device_is_cpu(self) -> None:
        assert SimulationConfig().device == "cpu"

    def test_default_hankel_filter_is_werthmuller(self) -> None:
        """Decisão #3: filtro default werthmuller_201pt."""
        assert SimulationConfig().hankel_filter == "werthmuller_201pt"

    def test_default_multi_fields_are_none(self) -> None:
        """Extensões opcionais (multi-f, multi-TR) são None por default."""
        cfg = SimulationConfig()
        assert cfg.frequencies_hz is None
        assert cfg.tr_spacings_m is None

    def test_default_compute_jacobian_false(self) -> None:
        assert SimulationConfig().compute_jacobian is False

    def test_default_num_threads_auto(self) -> None:
        """num_threads = -1 → auto-detectar cores disponíveis."""
        assert SimulationConfig().num_threads == -1


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE VALIDAÇÃO DE RANGES NUMÉRICOS
# ──────────────────────────────────────────────────────────────────────────────
class TestRangeValidation:
    """Falha ao construir com valores fora dos ranges físicos."""

    def test_frequency_too_low_fails(self) -> None:
        with pytest.raises(AssertionError, match="frequency_hz"):
            SimulationConfig(frequency_hz=50.0)  # < 100 Hz

    def test_frequency_too_high_fails(self) -> None:
        with pytest.raises(AssertionError, match="frequency_hz"):
            SimulationConfig(frequency_hz=2.0e6)  # > 1 MHz

    def test_frequency_at_lower_boundary_ok(self) -> None:
        cfg = SimulationConfig(frequency_hz=100.0)
        assert cfg.frequency_hz == 100.0

    def test_frequency_at_upper_boundary_ok(self) -> None:
        cfg = SimulationConfig(frequency_hz=1.0e6)
        assert cfg.frequency_hz == 1.0e6

    def test_tr_spacing_too_low_fails(self) -> None:
        with pytest.raises(AssertionError, match="tr_spacing_m"):
            SimulationConfig(tr_spacing_m=0.05)  # < 0.1 m

    def test_tr_spacing_too_high_fails(self) -> None:
        with pytest.raises(AssertionError, match="tr_spacing_m"):
            SimulationConfig(tr_spacing_m=20.0)  # > 10 m

    def test_n_positions_too_low_fails(self) -> None:
        with pytest.raises(AssertionError, match="n_positions"):
            SimulationConfig(n_positions=5)  # < 10

    def test_n_positions_too_high_fails(self) -> None:
        with pytest.raises(AssertionError, match="n_positions"):
            SimulationConfig(n_positions=200_000)  # > 100k


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE VALIDAÇÃO DE ENUMS (backend, dtype, device, hankel_filter)
# ──────────────────────────────────────────────────────────────────────────────
class TestEnumValidation:
    """Falha ao construir com strings fora dos conjuntos permitidos."""

    def test_invalid_backend(self) -> None:
        with pytest.raises(AssertionError, match="backend"):
            SimulationConfig(backend="pytorch")

    def test_invalid_dtype(self) -> None:
        with pytest.raises(AssertionError, match="dtype"):
            SimulationConfig(dtype="float32")

    def test_invalid_device(self) -> None:
        with pytest.raises(AssertionError, match="device"):
            SimulationConfig(device="tpu")  # tpu não está no conjunto

    def test_invalid_hankel_filter(self) -> None:
        with pytest.raises(AssertionError, match="hankel_filter"):
            SimulationConfig(hankel_filter="key_401pt")

    @pytest.mark.parametrize("backend", ["fortran_f2py", "numba", "jax"])
    def test_valid_backends(self, backend: str) -> None:
        cfg = SimulationConfig(
            backend=backend,
            device="cpu" if backend != "jax" else "cpu",
        )
        assert cfg.backend == backend

    @pytest.mark.parametrize("dtype", ["complex128", "complex64"])
    def test_valid_dtypes(self, dtype: str) -> None:
        assert SimulationConfig(dtype=dtype).dtype == dtype

    @pytest.mark.parametrize(
        "filter_name",
        ["werthmuller_201pt", "kong_61pt", "anderson_801pt"],
    )
    def test_valid_hankel_filters(self, filter_name: str) -> None:
        assert SimulationConfig(hankel_filter=filter_name).hankel_filter == filter_name


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE CONFLITOS MÚTUOS ENTRE CAMPOS
# ──────────────────────────────────────────────────────────────────────────────
class TestMutualExclusivity:
    """Garante rejeição de combinações incompatíveis."""

    def test_fortran_plus_gpu_fails(self) -> None:
        """Fortran tatu.x roda apenas em CPU."""
        with pytest.raises(AssertionError, match="fortran_f2py.*gpu"):
            SimulationConfig(backend="fortran_f2py", device="gpu")

    def test_numba_plus_gpu_fails(self) -> None:
        """Numba no roadmap atual não suporta GPU."""
        with pytest.raises(AssertionError, match="numba.*CPU"):
            SimulationConfig(backend="numba", device="gpu")

    def test_jax_plus_gpu_ok(self) -> None:
        """JAX é o único backend compatível com GPU."""
        cfg = SimulationConfig(backend="jax", device="gpu")
        assert cfg.backend == "jax"
        assert cfg.device == "gpu"

    def test_jax_plus_cpu_ok(self) -> None:
        """JAX também pode rodar em CPU (via JIT)."""
        cfg = SimulationConfig(backend="jax", device="cpu")
        assert cfg.backend == "jax"
        assert cfg.device == "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE LISTAS OPCIONAIS (multi-frequência, multi-TR)
# ──────────────────────────────────────────────────────────────────────────────
class TestOptionalLists:
    """Valida lists opcionais de frequências e spacings."""

    def test_empty_frequencies_fails(self) -> None:
        with pytest.raises(AssertionError, match="frequencies_hz"):
            SimulationConfig(frequencies_hz=[])

    def test_empty_tr_spacings_fails(self) -> None:
        with pytest.raises(AssertionError, match="tr_spacings_m"):
            SimulationConfig(tr_spacings_m=[])

    def test_out_of_range_frequency_in_list_fails(self) -> None:
        with pytest.raises(AssertionError, match="frequencies_hz"):
            SimulationConfig(frequencies_hz=[20000.0, 2.0e6])  # 2e6 > 1e6

    def test_out_of_range_tr_spacing_in_list_fails(self) -> None:
        with pytest.raises(AssertionError, match="tr_spacings_m"):
            SimulationConfig(tr_spacings_m=[1.0, 20.0])  # 20 > 10

    def test_valid_frequencies_list(self) -> None:
        cfg = SimulationConfig(frequencies_hz=[20000.0, 100000.0, 400000.0])
        assert cfg.frequencies_hz == [20000.0, 100000.0, 400000.0]

    def test_valid_tr_spacings_list(self) -> None:
        cfg = SimulationConfig(tr_spacings_m=[0.5, 1.0, 1.5])
        assert cfg.tr_spacings_m == [0.5, 1.0, 1.5]


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE NUM_THREADS
# ──────────────────────────────────────────────────────────────────────────────
class TestNumThreads:
    """Valida campo num_threads."""

    def test_auto_detect_ok(self) -> None:
        """num_threads=-1 → auto-detect (válido)."""
        assert SimulationConfig(num_threads=-1).num_threads == -1

    def test_explicit_positive_ok(self) -> None:
        assert SimulationConfig(num_threads=8).num_threads == 8

    def test_zero_threads_fails(self) -> None:
        with pytest.raises(AssertionError, match="num_threads"):
            SimulationConfig(num_threads=0)

    def test_negative_other_than_minus_one_fails(self) -> None:
        with pytest.raises(AssertionError, match="num_threads"):
            SimulationConfig(num_threads=-5)


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE PRESETS
# ──────────────────────────────────────────────────────────────────────────────
class TestPresets:
    """Testa os 4 presets @classmethod."""

    def test_default_equals_empty_constructor(self) -> None:
        """Preset default é idêntico a SimulationConfig()."""
        assert SimulationConfig.default() == SimulationConfig()

    def test_high_precision_uses_anderson(self) -> None:
        cfg = SimulationConfig.high_precision()
        assert cfg.hankel_filter == "anderson_801pt"
        assert cfg.dtype == "complex128"

    def test_production_gpu_config(self) -> None:
        cfg = SimulationConfig.production_gpu()
        assert cfg.backend == "jax"
        assert cfg.device == "gpu"
        assert cfg.dtype == "complex64"
        assert cfg.hankel_filter == "kong_61pt"

    def test_realtime_cpu_config(self) -> None:
        cfg = SimulationConfig.realtime_cpu()
        assert cfg.backend == "numba"
        assert cfg.device == "cpu"
        assert cfg.dtype == "complex128"
        assert cfg.hankel_filter == "kong_61pt"

    def test_all_presets_pass_validation(self) -> None:
        """Todos os presets produzem instâncias válidas."""
        # A própria construção chama __post_init__; se falhar, levanta.
        SimulationConfig.default()
        SimulationConfig.high_precision()
        SimulationConfig.production_gpu()
        SimulationConfig.realtime_cpu()


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE IMUTABILIDADE (frozen dataclass)
# ──────────────────────────────────────────────────────────────────────────────
class TestImmutability:
    """Garante que SimulationConfig é frozen e mutação usa replace."""

    def test_cannot_mutate_field(self) -> None:
        cfg = SimulationConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.frequency_hz = 100.0  # type: ignore[misc]

    def test_replace_creates_new_instance(self) -> None:
        """dataclasses.replace reconstrói com re-validação."""
        cfg = SimulationConfig()
        cfg2 = dataclasses.replace(cfg, frequency_hz=100000.0)
        assert cfg is not cfg2
        assert cfg.frequency_hz == 20000.0  # original inalterado
        assert cfg2.frequency_hz == 100000.0

    def test_replace_revalidates(self) -> None:
        """dataclasses.replace dispara __post_init__ novamente."""
        cfg = SimulationConfig()
        with pytest.raises(AssertionError, match="frequency_hz"):
            dataclasses.replace(cfg, frequency_hz=50.0)  # fora do range


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE SERIALIZAÇÃO (dict + YAML roundtrip)
# ──────────────────────────────────────────────────────────────────────────────
class TestSerialization:
    """Testa to_dict / from_dict / to_yaml / from_yaml."""

    def test_to_dict_returns_dict(self) -> None:
        cfg = SimulationConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["frequency_hz"] == 20000.0
        assert d["backend"] == "fortran_f2py"

    def test_dict_roundtrip(self) -> None:
        """cfg → to_dict → from_dict == cfg."""
        cfg_original = SimulationConfig(
            frequency_hz=400000.0,
            tr_spacing_m=1.5,
            backend="jax",
            device="gpu",
            dtype="complex64",
            hankel_filter="kong_61pt",
            frequencies_hz=[20000.0, 400000.0],
        )
        d = cfg_original.to_dict()
        cfg_restored = SimulationConfig.from_dict(d)
        assert cfg_original == cfg_restored

    def test_from_dict_ignores_extra_keys(self, caplog) -> None:
        """Chaves desconhecidas são ignoradas com warning."""
        d = {
            "frequency_hz": 100000.0,
            "unknown_field": "ignored",
            "another_extra": 42,
        }
        import logging

        with caplog.at_level(logging.WARNING):
            cfg = SimulationConfig.from_dict(d)
        assert cfg.frequency_hz == 100000.0
        # Log deve avisar sobre chaves ignoradas.
        assert any("unknown_field" in rec.message for rec in caplog.records)

    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        """cfg → to_yaml → from_yaml == cfg."""
        pytest.importorskip("yaml")
        cfg_original = SimulationConfig.high_precision()
        yaml_path = tmp_path / "test_sim_config.yaml"
        cfg_original.to_yaml(yaml_path)
        assert yaml_path.exists()

        cfg_restored = SimulationConfig.from_yaml(yaml_path)
        assert cfg_original == cfg_restored

    def test_from_yaml_missing_file(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        with pytest.raises(FileNotFoundError):
            SimulationConfig.from_yaml(tmp_path / "nao_existe.yaml")

    def test_from_yaml_malformed_top_level(self, tmp_path: Path) -> None:
        """YAML que não contém um dict no topo deve falhar."""
        pytest.importorskip("yaml")
        path = tmp_path / "bad.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")  # lista, não dict
        with pytest.raises(ValueError, match="mapeamento"):
            SimulationConfig.from_yaml(path)


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE IGUALDADE E HASH
# ──────────────────────────────────────────────────────────────────────────────
class TestEquality:
    """Garante que dataclass comparisons funcionam corretamente."""

    def test_equal_configs_are_equal(self) -> None:
        assert SimulationConfig() == SimulationConfig()

    def test_different_configs_are_not_equal(self) -> None:
        a = SimulationConfig(frequency_hz=20000.0)
        b = SimulationConfig(frequency_hz=100000.0)
        assert a != b

    def test_frozen_dataclass_is_hashable(self) -> None:
        """frozen=True torna o dataclass hashable (lists opcionais None)."""
        cfg = SimulationConfig()
        # Deve ser usável como chave de dict
        cache: dict[SimulationConfig, str] = {cfg: "resultado"}
        assert cache[cfg] == "resultado"
