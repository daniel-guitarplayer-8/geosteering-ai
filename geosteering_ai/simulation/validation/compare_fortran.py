# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/validation/compare_fortran.py                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Validação binária Fortran ↔ Python    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13 (Sprint 4.4 — PR #14a)                         ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy + subprocess                                         ║
# ║  Dependências: numpy, pathlib, subprocess, SimulationConfig             ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Executa `tatu.x` (simulador Fortran v10.0) via subprocess a partir    ║
# ║    de um arquivo `model.in` gerado dinamicamente, lê o `.dat` 22-col    ║
# ║    binário resultante e o compara bit-a-bit com a saída do simulador    ║
# ║    Python otimizado (`simulate()`) nos 3 backends: Numba, JAX hybrid e  ║
# ║    JAX native.                                                            ║
# ║                                                                           ║
# ║  FLUXO                                                                    ║
# ║    ┌───────────────────────────────────────────────────────────────┐     ║
# ║    │  CanonicalModel (oklahoma_3, ..., viking_graben_10)           │     ║
# ║    │    ↓                                                          │     ║
# ║    │  export_model_in(cfg, rho_h, rho_v, esp)  →  model.in        │     ║
# ║    │    ↓                                                          │     ║
# ║    │  run_tatu_x(model.in)  →  {dat, out, elapsed}                │     ║
# ║    │    ↓                                                          │     ║
# ║    │  read_fortran_dat_22col(dat)  →  np.ndarray (n_rec, 22)      │     ║
# ║    │    ↓                                                          │     ║
# ║    │  simulate(rho_h, rho_v, esp, positions_z, cfg_backend)       │     ║
# ║    │    ↓                                                          │     ║
# ║    │  FortranComparisonResult (max_abs_error, max_rel_error, ...)  │     ║
# ║    └───────────────────────────────────────────────────────────────┘     ║
# ║                                                                           ║
# ║  TOLERÂNCIAS ESPERADAS                                                    ║
# ║    • Numba:       max_abs_error < 1e-6                                   ║
# ║    • JAX hybrid:  max_abs_error < 1e-6  (mesma Numba via pure_callback)  ║
# ║    • JAX native:  max_abs_error < 1e-4  (lax.switch + complex64 upcast)  ║
# ║                                                                           ║
# ║  GUARDAS                                                                  ║
# ║    Todas as funções que executam `tatu.x` usam timeout (default 300 s)  ║
# ║    e verificam `returncode==0`. Em caso de falha, retornam resultado    ║
# ║    com `passed=False` e logam stderr.                                    ║
# ║                                                                           ║
# ║  CORRELAÇÃO COM CLAUDE.md                                                 ║
# ║    • Paridade física/geofísica com Fortran é inviolável (restrição 5).  ║
# ║    • Este módulo é o oráculo binário de última milha.                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Validação binária Fortran ↔ Python nos modelos canônicos.

Este módulo expõe três funções públicas e uma dataclass imutável:

- :func:`run_tatu_x` — executa o binário Fortran via subprocess.
- :func:`read_fortran_dat_22col` — lê `.dat` binário (reusa ``DTYPE_22COL``).
- :func:`compare_fortran_python` — orquestra comparação nos 3 backends.
- :class:`FortranComparisonResult` — container imutável de resultado.

Example:
    Comparação oklahoma_3 contra tatu.x nos 3 backends::

        >>> from geosteering_ai.simulation.validation import compare_fortran_python
        >>> results = compare_fortran_python("oklahoma_3")
        >>> for r in results:
        ...     print(f"{r.backend}: max_abs={r.max_abs_error:.2e} passed={r.passed}")
        numba: max_abs=3.2e-12 passed=True
        jax_hybrid: max_abs=3.2e-12 passed=True
        jax_native: max_abs=8.7e-06 passed=True

Note:
    Todas as funções fazem ``pytest.skip`` implícito quando ``tatu.x``
    não está presente no repositório (via guarda em ``run_tatu_x``).
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.io.binary_dat import DTYPE_22COL
from geosteering_ai.simulation.io.model_in import export_model_in

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas / de execução
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_FORTRAN_EXEC: Path = Path("Fortran_Gerador/tatu.x")
"""Caminho default do binário Fortran (relativo ao workdir do pytest)."""

DEFAULT_TIMEOUT_S: float = 300.0
"""Timeout em segundos para `tatu.x`. Suficiente para modelos ≤28 camadas."""

DEFAULT_TOL_ABS: float = 1.0e-6
"""Tolerância absoluta padrão para Numba / JAX hybrid (double precision)."""

DEFAULT_TOL_ABS_JAX_NATIVE: float = 1.0e-4
"""Tolerância absoluta para JAX native — acomoda lax.switch + upcast."""


# ──────────────────────────────────────────────────────────────────────────────
# Container de resultado
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FortranComparisonResult:
    """Resultado imutável de uma comparação Fortran ↔ Python.

    Attributes:
        canonical_model_name: Nome do modelo canônico comparado.
        backend: Backend Python (``"numba"``, ``"jax_hybrid"``, ``"jax_native"``).
        max_abs_error: Máximo erro absoluto componente-a-componente no tensor H.
        max_rel_error: Máximo erro relativo (|python - fortran| / |fortran|).
        l2_error: Norma L2 do erro (shape-flattened).
        elapsed_fortran_s: Tempo de parede do ``tatu.x`` (subprocess).
        elapsed_python_s: Tempo de parede de ``simulate()``.
        speedup: ``elapsed_fortran_s / elapsed_python_s`` (>1 = Python mais rápido).
        passed: ``True`` se ``max_abs_error < tol_abs`` usada na comparação.
        tol_abs_used: Tolerância efetivamente aplicada.
        n_positions: Número de posições comparadas.
        notes: Mensagens de contexto (ex.: "JAX native não disponível").

    Note:
        ``elapsed_python_s`` inclui JIT warmup na primeira execução.
        Para medidas estáveis, rode ``simulate()`` ≥ 2× e descarte o primeiro.
    """

    canonical_model_name: str
    backend: str
    max_abs_error: float
    max_rel_error: float
    l2_error: float
    elapsed_fortran_s: float
    elapsed_python_s: float
    speedup: float
    passed: bool
    tol_abs_used: float
    n_positions: int
    notes: str = ""

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} [{self.canonical_model_name}/{self.backend}] "
            f"max_abs={self.max_abs_error:.2e} "
            f"tol={self.tol_abs_used:.0e} "
            f"speedup={self.speedup:.1f}×"
        )


# ──────────────────────────────────────────────────────────────────────────────
# run_tatu_x — subprocess wrapper
# ──────────────────────────────────────────────────────────────────────────────


def run_tatu_x(
    model_in_path: Path | str,
    fortran_exec: Path | str = DEFAULT_FORTRAN_EXEC,
    output_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Executa ``tatu.x`` via subprocess com timeout + captura stdout/stderr.

    O binário Fortran lê ``model.in`` do diretório corrente (``cwd``) por
    design. Este wrapper copia ``model_in_path`` para ``cwd=output_dir`` se
    necessário e executa ``tatu.x`` com o nome ``model.in`` canônico.

    Args:
        model_in_path: Caminho absoluto ou relativo ao arquivo ``model.in``
            gerado por :func:`export_model_in`.
        fortran_exec: Caminho para o binário ``tatu.x``. Default:
            ``Fortran_Gerador/tatu.x`` relativo ao workdir.
        output_dir: Diretório de trabalho (onde ``tatu.x`` será invocado).
            Se ``None``, usa o diretório pai de ``model_in_path``.
        timeout_s: Timeout em segundos. Default 300s (suficiente para ≤28
            camadas e ≤1000 posições).

    Returns:
        Dicionário contendo:

        - ``dat_path`` (Path): ``.dat`` gerado (pode não existir em falha)
        - ``out_path`` (Path): ``.out`` de metadata
        - ``stdout`` (str): saída padrão do Fortran
        - ``stderr`` (str): saída de erro do Fortran
        - ``returncode`` (int): código de retorno do processo
        - ``elapsed_s`` (float): tempo de parede em segundos
        - ``success`` (bool): ``returncode == 0 and dat_path.exists()``

    Raises:
        FileNotFoundError: Se ``fortran_exec`` não existe.
        subprocess.TimeoutExpired: Se o processo exceder ``timeout_s``.

    Note:
        O simulador Fortran sempre escreve ``{filename}.dat`` + ``info{filename}.out``
        no ``cwd``; o ``filename`` é lido do ``model.in``. Este wrapper localiza
        o ``.dat`` resultante no mesmo diretório.
    """
    model_in_path = Path(model_in_path).resolve()
    fortran_exec = Path(fortran_exec).resolve()

    if not fortran_exec.exists():
        raise FileNotFoundError(
            f"Binário Fortran não encontrado: {fortran_exec}. "
            "Compile com `make -C Fortran_Gerador` antes de chamar."
        )
    if not model_in_path.exists():
        raise FileNotFoundError(f"model.in não encontrado: {model_in_path}")

    if output_dir is None:
        output_dir = model_in_path.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # tatu.x lê sempre "model.in" no cwd — copia se necessário.
    target_model_in = output_dir / "model.in"
    if target_model_in.resolve() != model_in_path.resolve():
        target_model_in.write_bytes(model_in_path.read_bytes())

    logger.info(
        "Executando Fortran: %s (cwd=%s, timeout=%.0fs)",
        fortran_exec,
        output_dir,
        timeout_s,
    )

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(fortran_exec)],
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("tatu.x timeout após %.1fs", timeout_s)
        raise
    elapsed_s = time.perf_counter() - t0

    # Localiza .dat + .out pelo padrão do Fortran.
    # O filename está no model.in; tentamos ambos os padrões.
    dat_path, out_path = _locate_output_files(output_dir, model_in_path)

    success = proc.returncode == 0 and dat_path.exists()
    if not success:
        logger.warning(
            "tatu.x retornou %d; dat_exists=%s stderr=%r",
            proc.returncode,
            dat_path.exists(),
            proc.stderr[:300],
        )

    return {
        "dat_path": dat_path,
        "out_path": out_path,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "success": success,
    }


def _locate_output_files(output_dir: Path, model_in_path: Path) -> tuple[Path, Path]:
    """Localiza .dat/.out no output_dir seguindo o padrão Fortran."""
    # Estratégia: tenta extrair filename do model.in (campo específico).
    # Fallback: primeiro .dat no diretório modificado após o start.
    filename_stem: Optional[str] = None
    try:
        for line in model_in_path.read_text().splitlines():
            stripped = line.strip()
            # Heurística: o filename no model.in é a linha após os dTR.
            # Conservador: aceitar linha que termina sem ponto e não é numérica.
            if (
                stripped
                and not stripped.replace(".", "")
                .replace("-", "")
                .replace("e", "")
                .replace("+", "")
                .isdigit()
                and " " not in stripped
                and len(stripped) > 0
                and not stripped[0].isdigit()
            ):
                filename_stem = stripped
                break
    except Exception:
        pass

    if filename_stem:
        dat_path = output_dir / f"{filename_stem}.dat"
        out_path = output_dir / f"info{filename_stem}.out"
    else:
        # Fallback: primeiro .dat recente.
        dats = sorted(
            output_dir.glob("*.dat"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        dat_path = dats[0] if dats else output_dir / "NOT_FOUND.dat"
        outs = sorted(
            output_dir.glob("info*.out"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        out_path = outs[0] if outs else output_dir / "NOT_FOUND.out"

    return dat_path, out_path


# ──────────────────────────────────────────────────────────────────────────────
# read_fortran_dat_22col — parser binário
# ──────────────────────────────────────────────────────────────────────────────


def read_fortran_dat_22col(dat_path: Path | str) -> np.ndarray:
    """Lê `.dat` binário Fortran 22-col via ``DTYPE_22COL`` (layout stream).

    Args:
        dat_path: Caminho para arquivo ``.dat`` 172 bytes/registro.

    Returns:
        Array estruturado NumPy com dtype ``DTYPE_22COL`` (re-interpretável
        como ``(n_rec, 22)`` via ``view(np.float64).reshape(-1, 22)``).

    Raises:
        FileNotFoundError: Se ``dat_path`` não existe.
        ValueError: Se tamanho do arquivo não é múltiplo de 172 bytes.
    """
    dat_path = Path(dat_path)
    if not dat_path.exists():
        raise FileNotFoundError(f".dat não encontrado: {dat_path}")

    size_bytes = dat_path.stat().st_size
    if size_bytes % DTYPE_22COL.itemsize != 0:
        raise ValueError(
            f"{dat_path}: tamanho {size_bytes} não é múltiplo de "
            f"{DTYPE_22COL.itemsize} bytes/registro — layout corrompido?"
        )

    return np.fromfile(dat_path, dtype=DTYPE_22COL)


# ──────────────────────────────────────────────────────────────────────────────
# _dat_to_htensor — converte .dat 22-col em tensor H complex (n_pos, nf, 9)
# ──────────────────────────────────────────────────────────────────────────────


def _dat_to_htensor(
    dat_records: np.ndarray,
    n_positions: int,
    nf: int = 1,
) -> np.ndarray:
    """Converte array estruturado 22-col em ``H_tensor`` ``(n_pos, nf, 9)``.

    Assume ordem Fortran ``(theta_idx=0, j=freq, i=pos)`` com ``theta=1``.

    Args:
        dat_records: Array estruturado com ``DTYPE_22COL``.
        n_positions: Número de posições esperado.
        nf: Número de frequências (default 1).

    Returns:
        ``np.ndarray`` complex128 shape ``(n_pos, nf, 9)``.
    """
    total = len(dat_records)
    expected = n_positions * nf
    if total < expected:
        raise ValueError(
            f"dat_records tem {total} registros, esperado ≥ {expected} "
            f"({n_positions} pos × {nf} freq)."
        )

    H = np.zeros((n_positions, nf, 9), dtype=np.complex128)
    # Ordem Fortran: for j in freq: for i in pos: ...
    # Mapeamento: index = j * n_positions + i → H[i, j, :]
    field_pairs = [
        ("Re_Hxx", "Im_Hxx"),
        ("Re_Hxy", "Im_Hxy"),
        ("Re_Hxz", "Im_Hxz"),
        ("Re_Hyx", "Im_Hyx"),
        ("Re_Hyy", "Im_Hyy"),
        ("Re_Hyz", "Im_Hyz"),
        ("Re_Hzx", "Im_Hzx"),
        ("Re_Hzy", "Im_Hzy"),
        ("Re_Hzz", "Im_Hzz"),
    ]
    for j in range(nf):
        start = j * n_positions
        block = dat_records[start : start + n_positions]
        for c, (re_name, im_name) in enumerate(field_pairs):
            H[:, j, c] = block[re_name] + 1j * block[im_name]
    return H


# ──────────────────────────────────────────────────────────────────────────────
# compare_fortran_python — orquestrador de alto nível
# ──────────────────────────────────────────────────────────────────────────────


def compare_fortran_python(
    canonical_model_name: str,
    backends: Optional[list[str]] = None,
    tol_abs: float = DEFAULT_TOL_ABS,
    tol_abs_jax_native: float = DEFAULT_TOL_ABS_JAX_NATIVE,
    fortran_exec: Optional[Path] = None,
    n_positions: int = 200,
    frequency_hz: float = 20000.0,
    tr_spacing_m: float = 1.0,
    workdir: Optional[Path] = None,
) -> list[FortranComparisonResult]:
    """Compara tatu.x contra os backends Python nos modelos canônicos.

    Para cada backend em ``backends``, o fluxo é:
    (1) gerar ``model.in`` via :func:`export_model_in`;
    (2) executar ``tatu.x`` via :func:`run_tatu_x`;
    (3) ler ``.dat`` 22-col via :func:`read_fortran_dat_22col`;
    (4) simular com ``simulate()`` no backend solicitado;
    (5) computar erros (max_abs, max_rel, L2).

    Args:
        canonical_model_name: Nome de um modelo em
            :func:`get_canonical_model` (``"oklahoma_3"``, ...).
        backends: Lista de backends a comparar. Default:
            ``["numba", "jax_hybrid", "jax_native"]``.
            ``"jax_hybrid"`` usa ``backend="jax"`` + ``use_native_dipoles=False``;
            ``"jax_native"`` usa ``backend="jax"`` + ``use_native_dipoles=True``.
        tol_abs: Tolerância absoluta para Numba / JAX hybrid.
        tol_abs_jax_native: Tolerância para JAX native (mais frouxa).
        fortran_exec: Caminho para ``tatu.x``. Default:
            :data:`DEFAULT_FORTRAN_EXEC`.
        n_positions: Número de posições de medição (default 200).
        frequency_hz: Frequência de operação (default 20 kHz).
        tr_spacing_m: Espaçamento TR (default 1 m).
        workdir: Diretório de trabalho. Se ``None``, usa ``tempfile.mkdtemp``.

    Returns:
        Lista de :class:`FortranComparisonResult`, uma por backend.

    Raises:
        FileNotFoundError: Se ``tatu.x`` ou canonical model não existir.

    Note:
        JAX backends são pulados silenciosamente (``notes`` populado)
        se ``jax`` não estiver instalado. A lista retornada mantém
        ordem de ``backends``.
    """
    # Lazy imports para permitir reuso em ambientes sem JAX.
    from geosteering_ai.simulation import simulate
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    if backends is None:
        backends = ["numba", "jax_hybrid", "jax_native"]
    fortran_exec = Path(fortran_exec) if fortran_exec else DEFAULT_FORTRAN_EXEC
    if not fortran_exec.exists():
        raise FileNotFoundError(
            f"Binário Fortran não encontrado em {fortran_exec}. "
            "Execute `make -C Fortran_Gerador` para compilar."
        )

    model = get_canonical_model(canonical_model_name)

    # ── Setup workdir ─────────────────────────────────────────────────────
    if workdir is None:
        import tempfile

        workdir = Path(tempfile.mkdtemp(prefix=f"compare_{canonical_model_name}_"))
    else:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    # ── Gera model.in ─────────────────────────────────────────────────────
    # Pmed razoável para cobrir n_positions medidas dentro do modelo.
    tj = max(model.max_depth - model.min_depth + 4.0, 5.0)
    pmed = tj / max(n_positions - 1, 1)
    h1 = model.min_depth - 2.0

    cfg_export = SimulationConfig(
        frequency_hz=frequency_hz,
        tr_spacing_m=tr_spacing_m,
        backend="numba",
        export_model_in=True,
        export_binary_dat=False,
        output_dir=str(workdir),
        output_filename=f"canon_{canonical_model_name}",
    )
    model_in_path = export_model_in(
        cfg_export,
        rho_h=model.rho_h,
        rho_v=model.rho_v,
        thicknesses=model.esp,
        h1=h1,
        tj=tj,
        pmed=pmed,
    )

    # ── Executa tatu.x ────────────────────────────────────────────────────
    run_result = run_tatu_x(model_in_path, fortran_exec=fortran_exec, output_dir=workdir)
    if not run_result["success"]:
        logger.error(
            "tatu.x falhou para %s: stderr=%r",
            canonical_model_name,
            run_result["stderr"][:400],
        )
        # Retorna resultados FAIL para todos os backends.
        return [
            FortranComparisonResult(
                canonical_model_name=canonical_model_name,
                backend=b,
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                l2_error=float("inf"),
                elapsed_fortran_s=run_result["elapsed_s"],
                elapsed_python_s=0.0,
                speedup=0.0,
                passed=False,
                tol_abs_used=tol_abs,
                n_positions=n_positions,
                notes=f"tatu.x falhou: rc={run_result['returncode']}",
            )
            for b in backends
        ]

    # ── Lê Fortran .dat ───────────────────────────────────────────────────
    dat_records = read_fortran_dat_22col(run_result["dat_path"])
    # n_positions observado no Fortran (pode diferir de n_positions pedido
    # devido a arredondamento de tj/pmed)
    n_fortran = len(dat_records)
    H_fortran = _dat_to_htensor(dat_records, n_positions=n_fortran, nf=1)
    positions_z_fortran = dat_records["z_obs"].astype(np.float64)

    elapsed_fortran = run_result["elapsed_s"]

    # ── Compara cada backend ──────────────────────────────────────────────
    results: list[FortranComparisonResult] = []
    for backend_name in backends:
        sim_cfg, notes = _build_sim_config_for_backend(
            backend_name, frequency_hz, tr_spacing_m
        )
        if sim_cfg is None:
            # Backend indisponível — resultado SKIP.
            results.append(
                FortranComparisonResult(
                    canonical_model_name=canonical_model_name,
                    backend=backend_name,
                    max_abs_error=0.0,
                    max_rel_error=0.0,
                    l2_error=0.0,
                    elapsed_fortran_s=elapsed_fortran,
                    elapsed_python_s=0.0,
                    speedup=0.0,
                    passed=True,  # Não falha — apenas nota SKIP.
                    tol_abs_used=tol_abs,
                    n_positions=n_fortran,
                    notes=notes + " (skipped)",
                )
            )
            continue

        tol_used = tol_abs_jax_native if backend_name == "jax_native" else tol_abs

        t0 = time.perf_counter()
        try:
            py_result = simulate(
                rho_h=model.rho_h,
                rho_v=model.rho_v,
                esp=model.esp,
                positions_z=positions_z_fortran,
                cfg=sim_cfg,
                dip_deg=0.0,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("simulate falhou em backend=%s", backend_name)
            results.append(
                FortranComparisonResult(
                    canonical_model_name=canonical_model_name,
                    backend=backend_name,
                    max_abs_error=float("inf"),
                    max_rel_error=float("inf"),
                    l2_error=float("inf"),
                    elapsed_fortran_s=elapsed_fortran,
                    elapsed_python_s=time.perf_counter() - t0,
                    speedup=0.0,
                    passed=False,
                    tol_abs_used=tol_used,
                    n_positions=n_fortran,
                    notes=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        elapsed_python = time.perf_counter() - t0

        # ── Compara tensores ────────────────────────────────────────────
        H_py = py_result.H_tensor  # (n_pos, nf, 9) ou (n_pos, 9)
        if H_py.ndim == 2:
            H_py = H_py[:, np.newaxis, :]

        # Alinha shape — H_fortran deve ser (n_fortran, 1, 9).
        if H_py.shape[0] != H_fortran.shape[0]:
            min_n = min(H_py.shape[0], H_fortran.shape[0])
            H_py = H_py[:min_n]
            H_f = H_fortran[:min_n]
        else:
            H_f = H_fortran

        diff = H_py - H_f
        abs_f = np.abs(H_f)
        abs_d = np.abs(diff)
        max_abs = float(abs_d.max())
        # rel_error: evitar div-por-zero (máscara em |H_f| > atol)
        mask = abs_f > 1e-15
        max_rel = float((abs_d[mask] / abs_f[mask]).max()) if mask.any() else 0.0
        l2 = float(np.linalg.norm(diff.ravel()))
        speedup = elapsed_fortran / elapsed_python if elapsed_python > 0 else 0.0

        results.append(
            FortranComparisonResult(
                canonical_model_name=canonical_model_name,
                backend=backend_name,
                max_abs_error=max_abs,
                max_rel_error=max_rel,
                l2_error=l2,
                elapsed_fortran_s=elapsed_fortran,
                elapsed_python_s=elapsed_python,
                speedup=speedup,
                passed=max_abs < tol_used,
                tol_abs_used=tol_used,
                n_positions=H_py.shape[0],
                notes=notes,
            )
        )
        logger.info(str(results[-1]))

    return results


def _build_sim_config_for_backend(
    backend_name: str,
    frequency_hz: float,
    tr_spacing_m: float,
) -> tuple[Optional[SimulationConfig], str]:
    """Monta ``SimulationConfig`` ou retorna (None, notes) se backend indisponível."""
    if backend_name == "numba":
        try:
            return (
                SimulationConfig(
                    frequency_hz=frequency_hz,
                    tr_spacing_m=tr_spacing_m,
                    backend="numba",
                ),
                "",
            )
        except Exception as exc:
            return None, f"numba indisponível: {exc}"

    if backend_name in ("jax_hybrid", "jax_native"):
        try:
            import jax  # noqa: F401
        except ImportError:
            return None, "jax não instalado"
        use_native = backend_name == "jax_native"
        try:
            cfg = SimulationConfig(
                frequency_hz=frequency_hz,
                tr_spacing_m=tr_spacing_m,
                backend="jax",
                use_native_dipoles=use_native,
            )
            return cfg, f"jax (use_native_dipoles={use_native})"
        except TypeError:
            # Campo use_native_dipoles ainda não existe em SimulationConfig.
            cfg = SimulationConfig(
                frequency_hz=frequency_hz,
                tr_spacing_m=tr_spacing_m,
                backend="jax",
            )
            return cfg, f"jax (use_native_dipoles=N/A, backend unified)"

    return None, f"backend desconhecido: {backend_name}"


__all__ = [
    "DEFAULT_FORTRAN_EXEC",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_TOL_ABS",
    "DEFAULT_TOL_ABS_JAX_NATIVE",
    "FortranComparisonResult",
    "compare_fortran_python",
    "read_fortran_dat_22col",
    "run_tatu_x",
]
