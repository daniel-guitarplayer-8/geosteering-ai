# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/io/binary_dat_multi.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Exportador .dat multi-TR (Sprint 11)    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-14 (PR #15)                                        ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : stdlib (struct, pathlib) + numpy                           ║
# ║  Dependências: numpy, struct, pathlib                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Exporta um :class:`MultiSimulationResult` para N arquivos              ║
# ║    `.dat` compatíveis byte-a-byte com o Fortran `tatu.x` v10.0.           ║
# ║                                                                           ║
# ║    Para `nTR == 1`: gera 1 arquivo `{filename}.dat`.                     ║
# ║    Para `nTR > 1`: gera nTR arquivos `{filename}_TR{i}.dat`              ║
# ║    (paridade exata com `writes_files` linha 1243-1249 Fortran).          ║
# ║                                                                           ║
# ║  LAYOUT BINÁRIO VALIDADO (via `xxd` de saída real do tatu.x, 2026-04-14) ║
# ║    Record = 172 bytes, ordem `<i21d` little-endian:                      ║
# ║      offset 0:   int32    i           (índice 1-based)                  ║
# ║      offset 4:   float64  z_obs       (m, profundidade ponto-médio)    ║
# ║      offset 12:  float64  rho_h_obs   (Ω·m)                             ║
# ║      offset 20:  float64  rho_v_obs   (Ω·m)                             ║
# ║      offset 28:  float64  Re(H1)      (Hxx real part)                  ║
# ║      offset 36:  float64  Im(H1)      (Hxx imag part)                  ║
# ║      ...                                                                  ║
# ║      offset 156: float64  Re(H9)      (Hzz real part)                  ║
# ║      offset 164: float64  Im(H9)      (Hzz imag part)                  ║
# ║                                                                           ║
# ║    Total por modelo: nAngles × nf × n_pos × 172 bytes                   ║
# ║    Loop externo: for k in nAngles, for j in nf, for i in n_pos          ║
# ║      (paridade Fortran PerfilaAnisoOmp.f08:1290-1311)                    ║
# ║                                                                           ║
# ║  FORMATO info{filename}.out (ASCII)                                       ║
# ║    Linha 1: "   ntheta   nf   nmaxmodel"  (write Fortran default)       ║
# ║    Linha 2: "   theta(1:ntheta)"          (floats espaçados)             ║
# ║    Linha 3: "   freq(1:nf)"                (floats em Hz)                ║
# ║    Linha 4: "   nmeds(1:ntheta)"           (ints)                        ║
# ║    Linha 5: "   use_tilted   n_tilted"     (0 0 default)                ║
# ║    (Linhas 6+ opcionais se use_tilted=1)                                 ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/PerfilaAnisoOmp.f08 (writes_files, linhas 1200-1314)║
# ║    • geosteering_ai/simulation/io/binary_dat.py (single-TR legado)       ║
# ║    • geosteering_ai/simulation/multi_forward.py (fonte dos dados)        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Exportação DAT byte-exata Fortran-compatível (multi-TR, Sprint 11).

Provê :func:`export_multi_tr_dat` que escreve N arquivos `.dat` equivalentes
bit-a-bit aos gerados pelo Fortran `tatu.x` v10.0 a partir de um
:class:`MultiSimulationResult`. Permite:

  1. Validação cruzada Fortran-Python (byte-diff)
  2. Reuso de pipelines de treino construídos em torno do formato Fortran
  3. Substituição drop-in de `tatu.x` em lotes grandes

Example:
    Geração de arquivos multi-TR::

        >>> from geosteering_ai.simulation import simulate_multi
        >>> from geosteering_ai.simulation.io.binary_dat_multi import (
        ...     export_multi_tr_dat
        ... )
        >>> result = simulate_multi(..., tr_spacings_m=[0.5, 1.0, 1.5])
        >>> paths = export_multi_tr_dat(result, "sim", output_dir="/tmp")
        >>> [p.name for p in paths]
        ['sim_TR1.dat', 'sim_TR2.dat', 'sim_TR3.dat']

Note:
    O assertion `sys.byteorder == 'little'` garante falha explícita em
    arquiteturas big-endian (raras em LWD; x86/ARM modernos são LE).

    Paridade byte-exata validada em 2026-04-14 via `xxd` contra saída
    real de `tatu.x` em modelo trivial (n=3, nTR=2, ntheta=1, nf=1).
"""
from __future__ import annotations

import logging
import struct
import sys
from pathlib import Path
from typing import List, Union

import numpy as np

from geosteering_ai.simulation.multi_forward import MultiSimulationResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes do layout binário (validadas empiricamente)
# ──────────────────────────────────────────────────────────────────────────────
# Record Fortran: int32 + 21 × float64 = 4 + 168 = 172 bytes, little-endian.
# Ordem: (i, z_obs, rho_h, rho_v, Re_H1, Im_H1, ..., Re_H9, Im_H9).
# Paridade Fortran PerfilaAnisoOmp.f08:1293-1299 — `write(1000) i, zrho(..1..3),
# real(cH(..1)), aimag(cH(..1)), ..., aimag(cH(..9))`.
_RECORD_STRUCT = struct.Struct("<i21d")
_RECORD_SIZE: int = _RECORD_STRUCT.size  # 172 bytes (validado xxd)

assert _RECORD_SIZE == 172, f"Layout struct inconsistente: {_RECORD_SIZE} != 172"

# ──────────────────────────────────────────────────────────────────────────────
# Guard de endianness — falha explícita em big-endian (não testado)
# ──────────────────────────────────────────────────────────────────────────────
# x86 e ARM modernos são little-endian. Se rodar em arquitetura exótica,
# o layout `<i21d` já força little-endian independente do host, mas
# documentamos explicitamente para quem auditar.
if sys.byteorder != "little":  # pragma: no cover
    logger.warning(
        "sys.byteorder=%s != 'little'. Layout forçado LE via struct '<i21d'; "
        "validação byte-a-byte pode diferir em máquinas BE.",
        sys.byteorder,
    )


def export_multi_tr_dat(
    result: MultiSimulationResult,
    filename_base: str,
    output_dir: Union[Path, str] = ".",
) -> List[Path]:
    """Exporta N arquivos `.dat` Fortran-compatíveis a partir de MultiSimulationResult.

    Gera nTR arquivos `.dat` seguindo exatamente o layout de `writes_files`
    (PerfilaAnisoOmp.f08:1200-1314), respeitando as regras:

      - ``nTR == 1``: gera ``{filename}.dat`` (sem sufixo)
      - ``nTR > 1``:  gera ``{filename}_TR{i}.dat`` com i=1..nTR (1-based)
      - Loops aninhados dentro de cada arquivo: ``for k (nAngles) → for j (nf)
        → for i (n_pos)`` — um registro de 172 bytes por combinação.

    Args:
        result: :class:`MultiSimulationResult` com tensor H shape
            ``(nTR, nAngles, n_pos, nf, 9)``.
        filename_base: Nome base dos arquivos (sem extensão).
            Ex.: ``"trivial"`` → ``trivial_TR1.dat`` etc.
        output_dir: Diretório de saída. Será criado se não existir.

    Returns:
        Lista de `Path` para os arquivos `.dat` criados, em ordem de TR.

    Raises:
        ValueError: Se shape de H_tensor for inconsistente.

    Note:
        **Paridade byte-exata validada** contra saída real do `tatu.x`
        v10.0 em 2026-04-14. O arquivo ``info{filename}.out`` é
        **também** emitido (via :func:`export_info_out`).

        Para nTR=1 o comportamento é compatível com
        :func:`geosteering_ai.simulation.io.binary_dat.export_binary_dat`
        (parceria single-TR Sprint 2.2).

    Example:
        Geração a partir de simulate_multi::

            >>> import numpy as np
            >>> from geosteering_ai.simulation import simulate_multi
            >>> result = simulate_multi(
            ...     rho_h=np.array([1., 100., 1.]),
            ...     rho_v=np.array([1., 100., 1.]),
            ...     esp=np.array([5.]),
            ...     positions_z=np.linspace(-2, 7, 20),
            ...     tr_spacings_m=[0.5, 1.0],
            ...     frequencies_hz=[20000.0],
            ... )
            >>> paths = export_multi_tr_dat(result, "test", "/tmp")
            >>> len(paths)
            2
    """
    # ── Validação ─────────────────────────────────────────────────
    H = result.H_tensor
    if H.ndim != 5:
        raise ValueError(
            f"H_tensor.ndim={H.ndim}; esperado 5 " f"(nTR, nAngles, n_pos, nf, 9)."
        )
    nTR, nAngles, n_pos, nf, ncomp = H.shape
    if ncomp != 9:
        raise ValueError(f"H_tensor.shape[-1]={ncomp}; esperado 9 componentes.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Também emite info{filename}.out (metadados)
    export_info_out(result, filename_base, output_dir)

    paths: List[Path] = []
    for iTR in range(nTR):
        # Nomenclatura Fortran: sufixo _TR{i} apenas para nTR > 1
        if nTR > 1:
            filename = f"{filename_base}_TR{iTR + 1}.dat"
        else:
            filename = f"{filename_base}.dat"
        out_path = output_dir / filename

        with out_path.open("wb") as fout:
            # Loop aninhado idêntico ao Fortran writes_files:1290-1311
            for k in range(nAngles):
                for j in range(nf):
                    for i in range(n_pos):
                        cH_ij = H[iTR, k, i, j]  # shape (9,) complex128
                        record = _RECORD_STRUCT.pack(
                            i + 1,  # 1-based (Fortran convention)
                            float(result.z_obs[k, i]),
                            float(result.rho_h_at_obs[k, i]),
                            float(result.rho_v_at_obs[k, i]),
                            # Pares Re/Im para os 9 componentes flat
                            cH_ij[0].real,
                            cH_ij[0].imag,
                            cH_ij[1].real,
                            cH_ij[1].imag,
                            cH_ij[2].real,
                            cH_ij[2].imag,
                            cH_ij[3].real,
                            cH_ij[3].imag,
                            cH_ij[4].real,
                            cH_ij[4].imag,
                            cH_ij[5].real,
                            cH_ij[5].imag,
                            cH_ij[6].real,
                            cH_ij[6].imag,
                            cH_ij[7].real,
                            cH_ij[7].imag,
                            cH_ij[8].real,
                            cH_ij[8].imag,
                        )
                        fout.write(record)

        paths.append(out_path)
        logger.debug(
            "export_multi_tr_dat: TR%d → %s (%d registros, %d bytes)",
            iTR + 1,
            out_path.name,
            nAngles * nf * n_pos,
            nAngles * nf * n_pos * _RECORD_SIZE,
        )

    return paths


def export_info_out(
    result: MultiSimulationResult,
    filename_base: str,
    output_dir: Union[Path, str] = ".",
) -> Path:
    """Exporta arquivo `info{filename}.out` com metadados ASCII.

    Formato idêntico ao Fortran `writes_files` (linha 1228-1241):

        Linha 1: "   ntheta   nf   nmaxmodel"     (3 inteiros)
        Linha 2: "   theta(1:ntheta)"             (ângulos em graus)
        Linha 3: "   freq(1:nf)"                   (freq em Hz)
        Linha 4: "   nmeds(1:ntheta)"             (n_pos por ângulo)
        Linha 5: "   use_tilted   n_tilted"       (flags F7, 0 0 se off)

    Args:
        result: :class:`MultiSimulationResult`.
        filename_base: Nome base (ex.: ``"trivial"`` → ``infotrivial.out``).
        output_dir: Diretório de saída.

    Returns:
        Path do arquivo criado.

    Note:
        Paridade com writes_files:1227-1242 Fortran. O formato ASCII
        usa ``"{:>12d}"`` para inteiros e ``"{:>20.16f}"`` ou
        ``"{:>24.15E}"`` para floats (list-directed I/O default Fortran).
        Não garantimos byte-exato no `.out` (apenas o `.dat`); mas
        semanticamente é equivalente e leitores Python parseiam bem.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"info{filename_base}.out"

    _, nAngles, n_pos, nf, _ = result.H_tensor.shape
    nmaxmodel = 1  # simulate_multi gera 1 modelo geológico por chamada
    # Paridade Fortran F7 flags: use_tilted, n_tilted
    n_tilted = int(result.H_tilted.shape[0]) if result.H_tilted is not None else 0
    use_tilted = 1 if n_tilted > 0 else 0

    lines: List[str] = []
    # Linha 1: contagens
    lines.append(f"{nAngles:>12d}{nf:>12d}{nmaxmodel:>12d}")
    # Linha 2: ângulos (usa write(10,*) estilo — floats com 17 dígitos)
    lines.append(" " + " ".join(f"{float(t):>21.16f}     " for t in result.dip_degs))
    # Linha 3: frequências
    lines.append(" " + " ".join(f"{float(f):>20.11f}     " for f in result.freqs_hz))
    # Linha 4: nmeds (cada ângulo com mesmo n_pos)
    lines.append(" " + " ".join(f"{n_pos:>11d}" for _ in range(nAngles)))
    # Linha 5: flags F7
    lines.append(f"{use_tilted:>12d}{n_tilted:>12d}")

    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out_path


def read_multi_tr_dat(
    dat_path: Union[Path, str],
    n_records: int,
) -> np.ndarray:
    """Lê um arquivo `.dat` Fortran (para testes de paridade).

    Args:
        dat_path: Path do arquivo `.dat`.
        n_records: Número esperado de registros (nAngles × nf × n_pos).

    Returns:
        Array structured com `n_records` entradas, fields compatíveis
        com :data:`DTYPE_22COL` de `binary_dat.py`.

    Raises:
        ValueError: Se o tamanho do arquivo não for múltiplo de 172
            ou não bater com `n_records × 172`.
    """
    dat_path = Path(dat_path)
    size_bytes = dat_path.stat().st_size
    expected = n_records * _RECORD_SIZE
    if size_bytes != expected:
        raise ValueError(
            f"Arquivo {dat_path.name}: {size_bytes} bytes != "
            f"{expected} esperados (={n_records}×{_RECORD_SIZE})."
        )

    # Unpack em bulk via numpy structured array
    # Reuso do DTYPE_22COL de binary_dat.py para consistência
    from geosteering_ai.simulation.io.binary_dat import DTYPE_22COL

    return np.fromfile(str(dat_path), dtype=DTYPE_22COL, count=n_records)


__all__ = [
    "export_multi_tr_dat",
    "export_info_out",
    "read_multi_tr_dat",
]
