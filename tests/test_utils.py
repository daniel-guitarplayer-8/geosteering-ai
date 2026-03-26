"""Testes para geosteering_ai.utils — Bloco 5.

Cobre os 6 submodulos de utils/:
    - logger: ColoredFormatter, setup_logger, get_logger
    - timer: format_time, elapsed_since, timer_decorator, ProgressTracker
    - validation: ValidationTracker, validate_shape
    - formatting: format_number, format_compact, format_bytes, log_header
    - system: is_colab, has_gpu, detect_environment, safe_mkdir, set_all_seeds
    - io: NumpyEncoder, safe_json_dump, safe_json_load
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# LOGGER
# ═══════════════════════════════════════════════════════════════════════════


class TestC:
    """Testa namespace de cores ANSI."""

    def test_ansi_codes_are_strings(self):
        from geosteering_ai.utils.logger import C
        assert isinstance(C.BOLD, str)
        assert isinstance(C.RESET, str)
        assert isinstance(C.RED, str)
        assert isinstance(C.GREEN, str)

    def test_reset_contains_escape(self):
        from geosteering_ai.utils.logger import C
        assert "\033[" in C.RESET


class TestColoredFormatter:
    """Testa ColoredFormatter."""

    def test_format_adds_color(self):
        from geosteering_ai.utils.logger import ColoredFormatter
        fmt = ColoredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        result = fmt.format(record)
        # Deve conter a mensagem original e codigos ANSI
        assert "hello" in result
        assert "\033[" in result

    def test_different_levels_different_colors(self):
        from geosteering_ai.utils.logger import ColoredFormatter
        fmt = ColoredFormatter()

        info_record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="",
            lineno=0, msg="info", args=(), exc_info=None,
        )
        warn_record = logging.LogRecord(
            name="t", level=logging.WARNING, pathname="",
            lineno=0, msg="warn", args=(), exc_info=None,
        )
        r1 = fmt.format(info_record)
        r2 = fmt.format(warn_record)
        # Devem ser diferentes (cores distintas)
        assert r1 != r2


class TestSetupLogger:
    """Testa setup_logger e get_logger."""

    def test_setup_logger_returns_logger(self):
        from geosteering_ai.utils.logger import setup_logger
        lg = setup_logger("test_setup", level=logging.DEBUG)
        assert isinstance(lg, logging.Logger)
        assert lg.level == logging.DEBUG

    def test_setup_logger_no_duplicate_handlers(self):
        from geosteering_ai.utils.logger import setup_logger
        name = "test_no_dup"
        lg1 = setup_logger(name)
        n_handlers = len(lg1.handlers)
        lg2 = setup_logger(name)
        assert len(lg2.handlers) == n_handlers

    def test_get_logger_prefixes_name(self):
        from geosteering_ai.utils.logger import get_logger
        lg = get_logger("mymodule")
        assert lg.name == "geosteering_ai.mymodule"

    def test_get_logger_no_double_prefix(self):
        from geosteering_ai.utils.logger import get_logger
        lg = get_logger("geosteering_ai.data.loading")
        assert lg.name == "geosteering_ai.data.loading"


# ═══════════════════════════════════════════════════════════════════════════
# TIMER
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatTime:
    """Testa format_time com diferentes escalas."""

    def test_milliseconds(self):
        from geosteering_ai.utils.timer import format_time
        result = format_time(0.045)
        assert "ms" in result
        assert "45.00" in result

    def test_seconds(self):
        from geosteering_ai.utils.timer import format_time
        result = format_time(5.5)
        assert "s" in result
        assert "5.50" in result

    def test_minutes(self):
        from geosteering_ai.utils.timer import format_time
        result = format_time(125.0)
        assert "min" in result

    def test_hours(self):
        from geosteering_ai.utils.timer import format_time
        result = format_time(7200.0)
        assert "h" in result


class TestElapsedSince:
    """Testa elapsed_since."""

    def test_returns_labeled_string(self):
        from geosteering_ai.utils.timer import elapsed_since
        t0 = time.perf_counter()
        result = elapsed_since(t0, "Test")
        assert result.startswith("Test: ")


class TestTimerDecorator:
    """Testa timer_decorator."""

    def test_preserves_return_value(self):
        from geosteering_ai.utils.timer import timer_decorator

        @timer_decorator
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self):
        from geosteering_ai.utils.timer import timer_decorator

        @timer_decorator
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


class TestProgressTracker:
    """Testa ProgressTracker."""

    def test_tracks_progress(self):
        from geosteering_ai.utils.timer import ProgressTracker
        tracker = ProgressTracker(total=10, description="Test")
        for _ in range(10):
            tracker.update()
        assert tracker.current == 10

    def test_finish_does_not_raise(self):
        from geosteering_ai.utils.timer import ProgressTracker
        tracker = ProgressTracker(total=5, description="Test")
        for _ in range(5):
            tracker.update()
        tracker.finish()  # Nao deve levantar excecao

    def test_explicit_step(self):
        from geosteering_ai.utils.timer import ProgressTracker
        tracker = ProgressTracker(total=100, description="Test")
        tracker.update(step=50)
        assert tracker.current == 50

    def test_zero_total_raises(self):
        from geosteering_ai.utils.timer import ProgressTracker
        with pytest.raises(ValueError, match="total deve ser > 0"):
            ProgressTracker(total=0)

    def test_negative_total_raises(self):
        from geosteering_ai.utils.timer import ProgressTracker
        with pytest.raises(ValueError, match="total deve ser > 0"):
            ProgressTracker(total=-5)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestValidationTracker:
    """Testa ValidationTracker."""

    def test_all_pass(self):
        from geosteering_ai.utils.validation import ValidationTracker
        vt = ValidationTracker("test")
        vt.check(True, "Check 1")
        vt.check(True, "Check 2")
        assert vt.finalize() is True
        assert vt.passed == 2
        assert vt.failed == 0

    def test_with_failure(self):
        from geosteering_ai.utils.validation import ValidationTracker
        vt = ValidationTracker("test")
        vt.check(True, "OK")
        vt.check(False, "FAIL")
        assert vt.finalize() is False
        assert vt.passed == 1
        assert vt.failed == 1

    def test_raise_on_error(self):
        from geosteering_ai.utils.validation import ValidationTracker
        vt = ValidationTracker("test")
        vt.check(False, "Bad check")
        with pytest.raises(RuntimeError):
            vt.finalize(raise_on_error=True)

    def test_check_returns_condition(self):
        from geosteering_ai.utils.validation import ValidationTracker
        vt = ValidationTracker("test")
        assert vt.check(True, "ok") is True
        assert vt.check(False, "fail") is False

    def test_sequential_indices(self):
        from geosteering_ai.utils.validation import ValidationTracker
        vt = ValidationTracker("test")
        vt.check(True, "A")
        vt.check(True, "B")
        vt.check(True, "C")
        assert [c["index"] for c in vt.checks] == [1, 2, 3]


class TestValidateShape:
    """Testa validate_shape."""

    def test_exact_match(self):
        from geosteering_ai.utils.validation import validate_shape
        data = np.zeros((32, 600, 5))
        assert validate_shape(data, (32, 600, 5), "test") is True

    def test_wildcard_batch(self):
        from geosteering_ai.utils.validation import validate_shape
        data = np.zeros((64, 600, 5))
        assert validate_shape(data, (None, 600, 5), "test") is True

    def test_wrong_ndim_raises(self):
        from geosteering_ai.utils.validation import validate_shape
        data = np.zeros((32, 600))
        with pytest.raises(ValueError, match="ndim=2"):
            validate_shape(data, (None, 600, 5), "test")

    def test_wrong_dim_raises(self):
        from geosteering_ai.utils.validation import validate_shape
        data = np.zeros((32, 601, 5))
        with pytest.raises(ValueError, match="dim\\[1\\]=601"):
            validate_shape(data, (None, 600, 5), "test")

    def test_dynamic_none_actual_accepted(self):
        """Dimensoes dinamicas TF (None no actual) devem ser aceitas."""
        from geosteering_ai.utils.validation import validate_shape

        class FakeTensor:
            shape = (None, 600, 5)  # simula TF TensorShape com batch dinamico

        assert validate_shape(FakeTensor(), (None, 600, 5), "tf") is True
        assert validate_shape(FakeTensor(), (32, 600, 5), "tf") is True


# ═══════════════════════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatNumber:
    """Testa format_number."""

    def test_integer(self):
        from geosteering_ai.utils.formatting import format_number
        assert format_number(1234567) == "1,234,567"

    def test_float(self):
        from geosteering_ai.utils.formatting import format_number
        assert format_number(1234.5) == "1,234.50"


class TestFormatCompact:
    """Testa format_compact."""

    def test_thousands(self):
        from geosteering_ai.utils.formatting import format_compact
        assert format_compact(1500) == "1.5K"

    def test_millions(self):
        from geosteering_ai.utils.formatting import format_compact
        assert format_compact(2500000) == "2.5M"

    def test_billions(self):
        from geosteering_ai.utils.formatting import format_compact
        assert format_compact(1_200_000_000) == "1.2B"

    def test_small_number(self):
        from geosteering_ai.utils.formatting import format_compact
        assert format_compact(42) == "42"

    def test_negative(self):
        from geosteering_ai.utils.formatting import format_compact
        assert format_compact(-1500) == "-1.5K"


class TestFormatBytes:
    """Testa format_bytes."""

    def test_bytes(self):
        from geosteering_ai.utils.formatting import format_bytes
        assert format_bytes(512) == "512 B"

    def test_kilobytes(self):
        from geosteering_ai.utils.formatting import format_bytes
        result = format_bytes(1536)
        assert "KB" in result
        assert "1.50" in result

    def test_megabytes(self):
        from geosteering_ai.utils.formatting import format_bytes
        result = format_bytes(268435456)
        assert "MB" in result

    def test_gigabytes(self):
        from geosteering_ai.utils.formatting import format_bytes
        result = format_bytes(2 * 1024 ** 3)
        assert "GB" in result
        assert "2.00" in result


class TestLogHeader:
    """Testa log_header (nao leva excecao)."""

    def test_does_not_raise(self):
        from geosteering_ai.utils.formatting import log_header
        log_header("TEST HEADER")


class TestLogSection:
    """Testa log_section."""

    def test_with_items(self):
        from geosteering_ai.utils.formatting import log_section
        log_section("Config", {"model": "ResNet_18", "lr": "1e-4"})

    def test_without_items(self):
        from geosteering_ai.utils.formatting import log_section
        log_section("Empty Section")


class TestColorizeFlag:
    """Testa colorize_flag_value."""

    def test_true_is_green(self):
        from geosteering_ai.utils.formatting import colorize_flag_value
        from geosteering_ai.utils.logger import C
        result = colorize_flag_value(True)
        assert C.BRIGHT_GREEN in result

    def test_false_is_yellow(self):
        from geosteering_ai.utils.formatting import colorize_flag_value
        from geosteering_ai.utils.logger import C
        result = colorize_flag_value(False)
        assert C.YELLOW in result

    def test_none_is_yellow(self):
        from geosteering_ai.utils.formatting import colorize_flag_value
        from geosteering_ai.utils.logger import C
        result = colorize_flag_value(None)
        assert C.YELLOW in result

    def test_string_is_white(self):
        from geosteering_ai.utils.formatting import colorize_flag_value
        from geosteering_ai.utils.logger import C
        result = colorize_flag_value("ResNet_18")
        assert C.BRIGHT_WHITE in result


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvironmentDetection:
    """Testa deteccao de ambiente."""

    def test_is_colab_returns_bool(self):
        from geosteering_ai.utils.system import is_colab
        assert isinstance(is_colab(), bool)

    def test_is_kaggle_returns_bool(self):
        from geosteering_ai.utils.system import is_kaggle
        assert isinstance(is_kaggle(), bool)

    def test_is_jupyter_returns_bool(self):
        from geosteering_ai.utils.system import is_jupyter
        assert isinstance(is_jupyter(), bool)

    def test_has_gpu_returns_bool(self):
        from geosteering_ai.utils.system import has_gpu
        assert isinstance(has_gpu(), bool)

    def test_detect_environment_returns_string(self):
        from geosteering_ai.utils.system import detect_environment
        env = detect_environment()
        assert env in {"Google Colab", "Kaggle", "Jupyter Notebook", "Local/Script"}

    def test_get_environment_info_keys(self):
        from geosteering_ai.utils.system import get_environment_info
        info = get_environment_info()
        assert "environment" in info
        assert "has_gpu" in info
        assert "python_version" in info
        assert "platform" in info


class TestSafeMkdir:
    """Testa safe_mkdir e ensure_dirs."""

    def test_creates_directory(self):
        from geosteering_ai.utils.system import safe_mkdir
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "subdir")
            assert safe_mkdir(target) is True
            assert os.path.isdir(target)

    def test_existing_directory_ok(self):
        from geosteering_ai.utils.system import safe_mkdir
        with tempfile.TemporaryDirectory() as tmpdir:
            assert safe_mkdir(tmpdir) is True

    def test_ensure_dirs_batch(self):
        from geosteering_ai.utils.system import ensure_dirs
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = [
                os.path.join(tmpdir, "a"),
                os.path.join(tmpdir, "b"),
                os.path.join(tmpdir, "c"),
            ]
            results = ensure_dirs(dirs)
            assert all(results.values())
            assert all(os.path.isdir(d) for d in dirs)


class TestMemory:
    """Testa memory_usage e clear_memory."""

    def test_memory_usage_returns_dict(self):
        from geosteering_ai.utils.system import memory_usage
        info = memory_usage()
        assert "available" in info
        assert "rss_mb" in info
        assert "method" in info

    def test_clear_memory_no_error(self):
        from geosteering_ai.utils.system import clear_memory
        clear_memory()  # Nao deve levantar excecao


class TestSetAllSeeds:
    """Testa set_all_seeds."""

    def test_seeds_set_successfully(self):
        from geosteering_ai.utils.system import set_all_seeds
        info = set_all_seeds(seed=123, deterministic=False)
        assert info["seed"] == 123
        assert info["python_ok"] is True
        assert info["numpy_ok"] is True

    def test_reproducible_numpy(self):
        from geosteering_ai.utils.system import set_all_seeds
        set_all_seeds(seed=42, deterministic=False)
        a = np.random.rand(5)
        set_all_seeds(seed=42, deterministic=False)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


class TestNumpyEncoder:
    """Testa NumpyEncoder."""

    def test_ndarray(self):
        from geosteering_ai.utils.io import NumpyEncoder
        data = {"arr": np.array([1, 2, 3])}
        result = json.loads(json.dumps(data, cls=NumpyEncoder))
        assert result["arr"] == [1, 2, 3]

    def test_numpy_int(self):
        from geosteering_ai.utils.io import NumpyEncoder
        data = {"val": np.int64(42)}
        result = json.loads(json.dumps(data, cls=NumpyEncoder))
        assert result["val"] == 42

    def test_numpy_float(self):
        from geosteering_ai.utils.io import NumpyEncoder
        data = {"val": np.float32(0.5)}
        result = json.loads(json.dumps(data, cls=NumpyEncoder))
        assert abs(result["val"] - 0.5) < 1e-6

    def test_numpy_bool(self):
        from geosteering_ai.utils.io import NumpyEncoder
        data = {"flag": np.bool_(True)}
        result = json.loads(json.dumps(data, cls=NumpyEncoder))
        assert result["flag"] is True


class TestSafeJsonDump:
    """Testa safe_json_dump e safe_json_load."""

    def test_roundtrip(self):
        from geosteering_ai.utils.io import safe_json_dump, safe_json_load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            data = {"model": "ResNet_18", "r2": 0.98, "epochs": 100}
            assert safe_json_dump(data, filepath) is True
            loaded = safe_json_load(filepath)
            assert loaded == data

    def test_numpy_roundtrip(self):
        from geosteering_ai.utils.io import safe_json_dump, safe_json_load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "np.json")
            data = {
                "array": np.array([1.0, 2.0, 3.0]),
                "integer": np.int64(42),
                "flag": np.bool_(False),
            }
            assert safe_json_dump(data, filepath) is True
            loaded = safe_json_load(filepath)
            assert loaded["array"] == [1.0, 2.0, 3.0]
            assert loaded["integer"] == 42
            assert loaded["flag"] is False

    def test_creates_parent_dirs(self):
        from geosteering_ai.utils.io import safe_json_dump
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "a", "b", "c", "test.json")
            assert safe_json_dump({"key": "val"}, filepath) is True
            assert os.path.isfile(filepath)

    def test_load_nonexistent_returns_none(self):
        from geosteering_ai.utils.io import safe_json_load
        result = safe_json_load("/nonexistent/path/file.json")
        assert result is None

    def test_load_invalid_json_returns_none(self):
        from geosteering_ai.utils.io import safe_json_load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()
            result = safe_json_load(f.name)
            assert result is None
            os.unlink(f.name)


# ═══════════════════════════════════════════════════════════════════════════
# INIT RE-EXPORTS
# ═══════════════════════════════════════════════════════════════════════════


class TestInitReExports:
    """Verifica que utils/__init__.py re-exporta todos os simbolos."""

    def test_all_exports_accessible(self):
        import geosteering_ai.utils as utils
        # Amostra dos simbolos mais importantes
        assert hasattr(utils, "setup_logger")
        assert hasattr(utils, "get_logger")
        assert hasattr(utils, "C")
        assert hasattr(utils, "format_time")
        assert hasattr(utils, "timer_decorator")
        assert hasattr(utils, "ProgressTracker")
        assert hasattr(utils, "ValidationTracker")
        assert hasattr(utils, "validate_shape")
        assert hasattr(utils, "format_number")
        assert hasattr(utils, "format_compact")
        assert hasattr(utils, "format_bytes")
        assert hasattr(utils, "log_header")
        assert hasattr(utils, "is_colab")
        assert hasattr(utils, "has_gpu")
        assert hasattr(utils, "safe_mkdir")
        assert hasattr(utils, "set_all_seeds")
        assert hasattr(utils, "NumpyEncoder")
        assert hasattr(utils, "safe_json_dump")
        assert hasattr(utils, "safe_json_load")

    def test_all_list_matches_exports(self):
        import geosteering_ai.utils as utils
        for name in utils.__all__:
            assert hasattr(utils, name), f"Missing export: {name}"
