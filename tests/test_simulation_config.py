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
    """Falha ao construir com valores fora dos ranges físicos expandidos.

    Note:
        Ranges atualizados na Sprint 2.1 pós-revisão:
        - frequency_hz: [10, 2e6] — cobre CSAMT baixa freq até 2 MHz (ARC/PeriScope).
        - tr_spacing_m: [0.01, 50] — cobre ferramentas curtas (0.5 m) até
          deep-reading PeriScope HD (20.43 m) com margem.
    """

    def test_frequency_too_low_fails(self) -> None:
        with pytest.raises(AssertionError, match="frequency_hz"):
            SimulationConfig(frequency_hz=5.0)  # < 10 Hz (novo mínimo)

    def test_frequency_too_high_fails(self) -> None:
        with pytest.raises(AssertionError, match="frequency_hz"):
            SimulationConfig(frequency_hz=5.0e6)  # > 2 MHz (novo máximo)

    def test_frequency_at_lower_boundary_ok(self) -> None:
        cfg = SimulationConfig(frequency_hz=10.0)  # novo mínimo
        assert cfg.frequency_hz == 10.0

    def test_frequency_at_upper_boundary_ok(self) -> None:
        cfg = SimulationConfig(frequency_hz=2.0e6)  # novo máximo (ARC/PeriScope 2 MHz)
        assert cfg.frequency_hz == 2.0e6

    def test_tr_spacing_too_low_fails(self) -> None:
        with pytest.raises(AssertionError, match="tr_spacing_m"):
            SimulationConfig(tr_spacing_m=0.005)  # < 0.01 m (novo mínimo)

    def test_tr_spacing_too_high_fails(self) -> None:
        with pytest.raises(AssertionError, match="tr_spacing_m"):
            SimulationConfig(tr_spacing_m=100.0)  # > 50 m (novo máximo)

    def test_tr_spacing_periscope_hd_deep_reading(self) -> None:
        """20.43 m (PeriScope HD deep-reading) agora é válido."""
        cfg = SimulationConfig(tr_spacing_m=20.43)
        assert cfg.tr_spacing_m == 20.43

    def test_tr_spacing_arc_ultra_long_819(self) -> None:
        """8.19 m (ARC ultra-longo) agora é válido."""
        cfg = SimulationConfig(tr_spacing_m=8.19)
        assert cfg.tr_spacing_m == 8.19

    def test_frequency_lwd_2mhz_dual(self) -> None:
        """2 MHz (par dual LWD moderno) agora é válido."""
        cfg = SimulationConfig(frequency_hz=2.0e6)
        assert cfg.frequency_hz == 2.0e6

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
            SimulationConfig(frequencies_hz=[20000.0, 5.0e6])  # 5e6 > 2e6

    def test_out_of_range_tr_spacing_in_list_fails(self) -> None:
        with pytest.raises(AssertionError, match="tr_spacings_m"):
            SimulationConfig(tr_spacings_m=[1.0, 100.0])  # 100 > 50

    def test_valid_frequencies_list(self) -> None:
        cfg = SimulationConfig(frequencies_hz=[20000.0, 100000.0, 400000.0])
        assert cfg.frequencies_hz == [20000.0, 100000.0, 400000.0]

    def test_valid_tr_spacings_list(self) -> None:
        cfg = SimulationConfig(tr_spacings_m=[0.5, 1.0, 1.5])
        assert cfg.tr_spacings_m == [0.5, 1.0, 1.5]

    def test_arc_periscope_multi_tr_valid(self) -> None:
        """Configuração ARC + PeriScope HD: [8.19, 20.43] m agora válida."""
        cfg = SimulationConfig(tr_spacings_m=[8.19, 20.43])
        assert cfg.tr_spacings_m == [8.19, 20.43]

    def test_lwd_dual_frequency_400k_2mhz_valid(self) -> None:
        """Par dual 400 kHz + 2 MHz (ARC6/PeriScope) agora válido."""
        cfg = SimulationConfig(frequencies_hz=[400000.0, 2.0e6])
        assert cfg.frequencies_hz == [400000.0, 2.0e6]


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
            dataclasses.replace(cfg, frequency_hz=5.0)  # < 10 Hz (novo min)


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


# ──────────────────────────────────────────────────────────────────────────────
# TESTES SPRINT 2.2 — GRUPO 7 (I/O Exportadores)
# ──────────────────────────────────────────────────────────────────────────────
class TestGroup7Exporters:
    """Valida os campos de exportação Fortran-compatível (Sprint 2.2)."""

    def test_defaults_all_off(self) -> None:
        """Por default, nenhum exportador está ativo."""
        cfg = SimulationConfig()
        assert cfg.export_model_in is False
        assert cfg.export_binary_dat is False
        assert cfg.output_dir == "."
        assert cfg.output_filename == "simulation"

    def test_export_model_in_with_valid_dir(self) -> None:
        """export_model_in=True + output_dir válido passa."""
        cfg = SimulationConfig(
            export_model_in=True,
            output_dir="/tmp/sim",
            output_filename="test",
        )
        assert cfg.export_model_in is True

    def test_export_with_empty_output_dir_fails(self) -> None:
        """output_dir vazio com export ativo falha."""
        with pytest.raises(AssertionError, match="output_dir"):
            SimulationConfig(export_model_in=True, output_dir="")

    def test_export_with_empty_filename_fails(self) -> None:
        """output_filename vazio com export ativo falha."""
        with pytest.raises(AssertionError, match="output_filename"):
            SimulationConfig(export_model_in=True, output_filename="")

    def test_export_with_invalid_filename_chars_fails(self) -> None:
        """output_filename com caracteres especiais inválidos falha."""
        with pytest.raises(AssertionError, match="caracteres inválidos"):
            SimulationConfig(export_binary_dat=True, output_filename="bad<name>")


# ──────────────────────────────────────────────────────────────────────────────
# TESTES SPRINT 2.2 — GRUPO 8 (F6 Compensação Midpoint)
# ──────────────────────────────────────────────────────────────────────────────
class TestGroup8Compensation:
    """Valida os campos de compensação F6 (Sprint 2.2)."""

    def test_default_off(self) -> None:
        """Por default, use_compensation=False, comp_pairs=None."""
        cfg = SimulationConfig()
        assert cfg.use_compensation is False
        assert cfg.comp_pairs is None

    def test_f6_requires_multi_tr(self) -> None:
        """F6 ativo com nTR < 2 deve falhar."""
        with pytest.raises(AssertionError, match=r"inerentemente multi-TR"):
            SimulationConfig(
                use_compensation=True,
                comp_pairs=((0, 1),),
                tr_spacings_m=[1.0],  # apenas 1 TR
            )

    def test_f6_requires_comp_pairs(self) -> None:
        """F6=True sem comp_pairs deve falhar."""
        with pytest.raises(AssertionError, match="comp_pairs não-vazio"):
            SimulationConfig(
                use_compensation=True,
                tr_spacings_m=[0.5, 1.0, 2.0],
            )

    def test_f6_valid_multi_tr_with_pair(self) -> None:
        """F6 com 3 TRs e 1 par válido passa."""
        cfg = SimulationConfig(
            use_compensation=True,
            tr_spacings_m=[0.5, 1.0, 2.0],
            comp_pairs=((0, 2),),
        )
        assert cfg.use_compensation is True
        assert len(cfg.comp_pairs) == 1

    def test_f6_pair_near_equals_far_fails(self) -> None:
        """comp_pairs com near == far (degenerado) deve falhar."""
        with pytest.raises(AssertionError, match="devem ser diferentes"):
            SimulationConfig(
                use_compensation=True,
                tr_spacings_m=[0.5, 1.0, 2.0],
                comp_pairs=((1, 1),),
            )

    def test_f6_pair_index_out_of_range_fails(self) -> None:
        """comp_pairs com índice >= nTR deve falhar."""
        with pytest.raises(AssertionError, match="fora do"):
            SimulationConfig(
                use_compensation=True,
                tr_spacings_m=[0.5, 1.0],
                comp_pairs=((0, 5),),
            )

    def test_f6_multiple_pairs(self) -> None:
        """F6 com múltiplos pares independentes funciona."""
        cfg = SimulationConfig(
            use_compensation=True,
            tr_spacings_m=[0.5, 1.0, 2.0, 4.0],
            comp_pairs=((0, 1), (2, 3)),
        )
        assert len(cfg.comp_pairs) == 2


# ──────────────────────────────────────────────────────────────────────────────
# TESTES SPRINT 2.2 — GRUPO 9 (F7 Antenas Inclinadas)
# ──────────────────────────────────────────────────────────────────────────────
class TestGroup9TiltedAntennas:
    """Valida os campos de antenas inclinadas F7 (Sprint 2.2)."""

    def test_default_off(self) -> None:
        """Por default, use_tilted_antennas=False."""
        cfg = SimulationConfig()
        assert cfg.use_tilted_antennas is False
        assert cfg.tilted_configs is None

    def test_f7_requires_tilted_configs(self) -> None:
        """F7=True sem tilted_configs deve falhar."""
        with pytest.raises(AssertionError, match="tilted_configs não-vazio"):
            SimulationConfig(use_tilted_antennas=True)

    def test_f7_valid_config_passes(self) -> None:
        """F7 com 1 config (45°, 0°) válida passa."""
        cfg = SimulationConfig(
            use_tilted_antennas=True,
            tilted_configs=((45.0, 0.0),),
        )
        assert cfg.use_tilted_antennas is True
        assert len(cfg.tilted_configs) == 1

    def test_f7_beta_out_of_range_fails(self) -> None:
        """β fora do range [0, 90] falha."""
        with pytest.raises(AssertionError, match="beta"):
            SimulationConfig(
                use_tilted_antennas=True,
                tilted_configs=((95.0, 0.0),),
            )

    def test_f7_phi_out_of_range_fails(self) -> None:
        """φ fora do range [0, 360) falha."""
        with pytest.raises(AssertionError, match="phi"):
            SimulationConfig(
                use_tilted_antennas=True,
                tilted_configs=((45.0, 360.0),),
            )

    def test_f7_multiple_configs_independent(self) -> None:
        """F7 com múltiplas configurações distintas funciona."""
        cfg = SimulationConfig(
            use_tilted_antennas=True,
            tilted_configs=((0.0, 0.0), (45.0, 90.0), (90.0, 180.0)),
        )
        assert len(cfg.tilted_configs) == 3


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE INTEGRAÇÃO — COMBINAÇÕES DOS NOVOS GRUPOS
# ──────────────────────────────────────────────────────────────────────────────
class TestSprint22Integration:
    """Testes de integração dos 3 novos grupos (I/O + F6 + F7)."""

    def test_all_new_groups_disabled_is_default(self) -> None:
        """Nenhuma flag Sprint 2.2 ativa no config default."""
        cfg = SimulationConfig()
        assert cfg.export_model_in is False
        assert cfg.export_binary_dat is False
        assert cfg.use_compensation is False
        assert cfg.use_tilted_antennas is False

    def test_all_new_groups_enabled_together(self) -> None:
        """Todos os grupos Sprint 2.2 ativos juntos é válido."""
        cfg = SimulationConfig(
            tr_spacings_m=[0.5, 1.0, 2.0],
            export_model_in=True,
            export_binary_dat=True,
            output_dir="/tmp/sim",
            output_filename="smoke",
            use_compensation=True,
            comp_pairs=((0, 2),),
            use_tilted_antennas=True,
            tilted_configs=((45.0, 0.0), (45.0, 90.0)),
        )
        assert cfg.use_compensation is True
        assert cfg.use_tilted_antennas is True
        assert cfg.export_model_in is True
