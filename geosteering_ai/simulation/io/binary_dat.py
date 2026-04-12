# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/io/binary_dat.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Exportador .dat 22-col binário        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 2.2)                                   ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x                                                  ║
# ║  Dependências: numpy                                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Exporta resultados forward do simulador Python em formato binário    ║
# ║    stream idêntico ao gerado por `tatu.x` (PerfilaAnisoOmp.f08:1282).  ║
# ║    O arquivo `.dat` gerado é byte-equivalente ao Fortran, permitindo:  ║
# ║      • Reuso pelo pipeline de treinamento (DataPipeline)               ║
# ║      • Validação cruzada com Fortran (diff binário)                    ║
# ║      • Audit trail de amostras geradas                                  ║
# ║                                                                           ║
# ║  LAYOUT DO .DAT 22-COLUNAS                                                ║
# ║    Registro por medida: 4 bytes (int32) + 21 × 8 bytes (float64)       ║
# ║                          = 172 bytes por registro                       ║
# ║                                                                           ║
# ║    Colunas (ordem nativa):                                              ║
# ║      0    : i          — índice da medida (1-based, int32)              ║
# ║      1    : z_obs      — profundidade do ponto-médio T-R (m)            ║
# ║      2    : rho_h      — resistividade horizontal na camada R (Ω·m)     ║
# ║      3    : rho_v      — resistividade vertical na camada R (Ω·m)       ║
# ║      4-5  : Re(Hxx), Im(Hxx)  (tensor linha 1 col 1)                   ║
# ║      6-7  : Re(Hxy), Im(Hxy)                                            ║
# ║      8-9  : Re(Hxz), Im(Hxz)                                            ║
# ║      10-11: Re(Hyx), Im(Hyx)                                            ║
# ║      12-13: Re(Hyy), Im(Hyy)                                            ║
# ║      14-15: Re(Hyz), Im(Hyz)                                            ║
# ║      16-17: Re(Hzx), Im(Hzx)                                            ║
# ║      18-19: Re(Hzy), Im(Hzy)                                            ║
# ║      20-21: Re(Hzz), Im(Hzz)                                            ║
# ║                                                                           ║
# ║  ORDEM DE ITERAÇÃO (Fortran, preservada):                                 ║
# ║      for k in range(ntheta):                                             ║
# ║          for j in range(nf):                                             ║
# ║              for i in range(nmeds[k]):                                   ║
# ║                  write_record(...)                                       ║
# ║                                                                           ║
# ║  OPT-IN                                                                   ║
# ║    Ativado por `cfg.export_binary_dat=True`. Chamar com flag desativada ║
# ║    levanta `RuntimeError` (proteção fail-fast).                         ║
# ║                                                                           ║
# ║  CORRELAÇÃO COM CLAUDE.md                                                 ║
# ║    • INPUT_FEATURES = [1, 4, 5, 20, 21] —                              ║
# ║      z_obs (1), Re(Hxx) (4), Im(Hxx) (5), Re(Hzz) (20), Im(Hzz) (21) ║
# ║    • OUTPUT_TARGETS = [2, 3] — rho_h (2), rho_v (3)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Exportador binário stream 22-col Fortran-compatível.

Este módulo implementa:

- :func:`export_binary_dat` — escreve `.dat` 22-col binário stream.
- :func:`export_out_metadata` — escreve `info{filename}.out` metadata.
- :data:`DTYPE_22COL` — dtype NumPy para leitura/escrita compatível.

Example:
    Geração + leitura round-trip::

        >>> import numpy as np
        >>> from geosteering_ai.simulation import SimulationConfig
        >>> from geosteering_ai.simulation.io.binary_dat import (
        ...     export_binary_dat, DTYPE_22COL,
        ... )
        >>> cfg = SimulationConfig(
        ...     export_binary_dat=True, output_dir="/tmp/sim"
        ... )
        >>> n_meds = 10
        >>> H_tensor = np.zeros((1, n_meds, 1, 9), dtype=np.complex128)
        >>> z_obs = np.linspace(0, 9, n_meds)
        >>> rho_h = np.ones(n_meds) * 100.0
        >>> rho_v = np.ones(n_meds) * 100.0
        >>> path = export_binary_dat(cfg, H_tensor, z_obs, rho_h, rho_v)
        >>> data = np.fromfile(path, dtype=DTYPE_22COL)
        >>> data.shape
        (10,)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from geosteering_ai.simulation.config import SimulationConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# DTYPE 22-COLUNAS
# ──────────────────────────────────────────────────────────────────────────────
# Layout byte-a-byte idêntico ao Fortran stream output (PerfilaAnisoOmp.f08:1301).
# Total: 172 bytes por registro (4 + 21×8).
# A ordem dos campos importa — NumPy preserva ordem de declaração em dtype
# estruturado, e é exatamente a ordem que o Fortran escreve.
DTYPE_22COL = np.dtype(
    [
        ("i", np.int32),
        ("z_obs", np.float64),
        ("rho_h", np.float64),
        ("rho_v", np.float64),
        ("Re_Hxx", np.float64),
        ("Im_Hxx", np.float64),
        ("Re_Hxy", np.float64),
        ("Im_Hxy", np.float64),
        ("Re_Hxz", np.float64),
        ("Im_Hxz", np.float64),
        ("Re_Hyx", np.float64),
        ("Im_Hyx", np.float64),
        ("Re_Hyy", np.float64),
        ("Im_Hyy", np.float64),
        ("Re_Hyz", np.float64),
        ("Im_Hyz", np.float64),
        ("Re_Hzx", np.float64),
        ("Im_Hzx", np.float64),
        ("Re_Hzy", np.float64),
        ("Im_Hzy", np.float64),
        ("Re_Hzz", np.float64),
        ("Im_Hzz", np.float64),
    ]
)

# Sanidade: tamanho em bytes deve ser 172
assert DTYPE_22COL.itemsize == 172, (
    f"DTYPE_22COL.itemsize={DTYPE_22COL.itemsize} != 172 — "
    "layout incompatível com Fortran stream I/O"
)


def export_binary_dat(
    cfg: SimulationConfig,
    H_tensor: np.ndarray,
    z_obs: np.ndarray,
    rho_h_profile: np.ndarray,
    rho_v_profile: np.ndarray,
    output_path: Union[Path, str, None] = None,
    *,
    append: bool = False,
) -> Path:
    """Escreve `.dat` 22-col binário stream Fortran-compatível.

    Gera um arquivo binário nativo little-endian (paridade Fortran
    `form='unformatted', access='stream'`) com 172 bytes por registro.

    Args:
        cfg: Instância validada de `SimulationConfig`. Deve ter
            `export_binary_dat=True`, senão `RuntimeError` é levantado.
        H_tensor: Array complex128 shape `(ntheta, nmeds, nf, 9)` com
            as 9 componentes do tensor H (ordem Hxx, Hxy, Hxz, Hyx,
            Hyy, Hyz, Hzx, Hzy, Hzz). Pode-se usar shape `(nmeds, 9)`
            para caso single-theta single-freq (automaticamente
            adaptado).
        z_obs: Array float64 shape `(nmeds,)` com profundidades dos
            pontos-médios T-R em metros.
        rho_h_profile: Array float64 shape `(nmeds,)` com resistividade
            horizontal na camada do receptor em cada medida (Ω·m).
        rho_v_profile: Array float64 shape `(nmeds,)` com resistividade
            vertical na camada do receptor em cada medida (Ω·m).
        output_path: Caminho do arquivo de saída. Se `None`, usa
            `{cfg.output_dir}/{cfg.output_filename}.dat`.
        append: Se True, abre em modo `'ab'` (multi-model batch,
            equivalente a `status='old', position='append'` do
            Fortran). Se False (default), trunca e escreve novo.

    Returns:
        `pathlib.Path` absoluto do arquivo criado.

    Raises:
        RuntimeError: Se `cfg.export_binary_dat=False`.
        ValueError: Se shapes não forem compatíveis entre H_tensor,
            z_obs, rho_h_profile, rho_v_profile.

    Note:
        **Ordem de iteração** (preservada do Fortran):
            for k in range(ntheta):
                for j in range(nf):
                    for i in range(nmeds):
                        ...

        Total de registros: `ntheta × nf × nmeds`.

        O argumento `append=True` é usado no Fortran para modelos
        em batch (modelm > 1): o primeiro escreve com trunc, os
        subsequentes apendam.
    """
    # ── Fail-fast ─────────────────────────────────────────────────
    if not cfg.export_binary_dat:
        raise RuntimeError(
            "export_binary_dat requer cfg.export_binary_dat=True. "
            "Use `dataclasses.replace(cfg, export_binary_dat=True)` "
            "para ativar opt-in."
        )

    # ── Normaliza shape do H_tensor ───────────────────────────────
    H_tensor = np.asarray(H_tensor, dtype=np.complex128)
    z_obs = np.asarray(z_obs, dtype=np.float64)
    rho_h_profile = np.asarray(rho_h_profile, dtype=np.float64)
    rho_v_profile = np.asarray(rho_v_profile, dtype=np.float64)

    # Adapta shape 2D → 4D se necessário: (nmeds, 9) → (1, nmeds, 1, 9)
    if H_tensor.ndim == 2:
        H_tensor = H_tensor[np.newaxis, :, np.newaxis, :]
    elif H_tensor.ndim == 3:
        # (ntheta, nmeds, 9) → (ntheta, nmeds, 1, 9)
        H_tensor = H_tensor[:, :, np.newaxis, :]
    elif H_tensor.ndim != 4:
        raise ValueError(f"H_tensor.ndim={H_tensor.ndim} inválido. Esperado 2, 3 ou 4.")

    ntheta, nmeds, nf, ncomp = H_tensor.shape
    if ncomp != 9:
        raise ValueError(f"H_tensor última dim={ncomp}, esperado 9 (Hxx..Hzz).")
    if z_obs.shape != (nmeds,):
        raise ValueError(f"z_obs.shape={z_obs.shape}, esperado ({nmeds},).")
    if rho_h_profile.shape != (nmeds,):
        raise ValueError(
            f"rho_h_profile.shape={rho_h_profile.shape}, esperado ({nmeds},)."
        )
    if rho_v_profile.shape != (nmeds,):
        raise ValueError(
            f"rho_v_profile.shape={rho_v_profile.shape}, esperado ({nmeds},)."
        )

    # ── Resolver output path ──────────────────────────────────────
    if output_path is None:
        output_path = Path(cfg.output_dir) / f"{cfg.output_filename}.dat"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Constrói array estruturado 22-col ─────────────────────────
    # Total de registros = ntheta × nf × nmeds
    n_records = ntheta * nf * nmeds
    records = np.zeros(n_records, dtype=DTYPE_22COL)

    # Ordem Fortran: k → j → i (innermost = medida)
    idx = 0
    for k in range(ntheta):
        for j in range(nf):
            for i in range(nmeds):
                records[idx]["i"] = i + 1  # 1-based Fortran
                records[idx]["z_obs"] = z_obs[i]
                records[idx]["rho_h"] = rho_h_profile[i]
                records[idx]["rho_v"] = rho_v_profile[i]
                records[idx]["Re_Hxx"] = H_tensor[k, i, j, 0].real
                records[idx]["Im_Hxx"] = H_tensor[k, i, j, 0].imag
                records[idx]["Re_Hxy"] = H_tensor[k, i, j, 1].real
                records[idx]["Im_Hxy"] = H_tensor[k, i, j, 1].imag
                records[idx]["Re_Hxz"] = H_tensor[k, i, j, 2].real
                records[idx]["Im_Hxz"] = H_tensor[k, i, j, 2].imag
                records[idx]["Re_Hyx"] = H_tensor[k, i, j, 3].real
                records[idx]["Im_Hyx"] = H_tensor[k, i, j, 3].imag
                records[idx]["Re_Hyy"] = H_tensor[k, i, j, 4].real
                records[idx]["Im_Hyy"] = H_tensor[k, i, j, 4].imag
                records[idx]["Re_Hyz"] = H_tensor[k, i, j, 5].real
                records[idx]["Im_Hyz"] = H_tensor[k, i, j, 5].imag
                records[idx]["Re_Hzx"] = H_tensor[k, i, j, 6].real
                records[idx]["Im_Hzx"] = H_tensor[k, i, j, 6].imag
                records[idx]["Re_Hzy"] = H_tensor[k, i, j, 7].real
                records[idx]["Im_Hzy"] = H_tensor[k, i, j, 7].imag
                records[idx]["Re_Hzz"] = H_tensor[k, i, j, 8].real
                records[idx]["Im_Hzz"] = H_tensor[k, i, j, 8].imag
                idx += 1

    # ── Escrita em modo trunc ou append ───────────────────────────
    mode = "ab" if append else "wb"
    with open(output_path, mode) as f:
        records.tofile(f)

    logger.debug(
        ".dat 22-col gravado em %s (%d registros, %d bytes)",
        output_path,
        n_records,
        n_records * DTYPE_22COL.itemsize,
    )
    return output_path.resolve()


def export_out_metadata(
    cfg: SimulationConfig,
    n_models: int,
    angulos_deg: np.ndarray,
    freqs_hz: np.ndarray,
    nmeds_per_theta: np.ndarray,
    output_path: Union[Path, str, None] = None,
) -> Path:
    """Escreve `info{filename}.out` metadata texto Fortran-compatível.

    Gera um arquivo de texto ASCII descrevendo a estrutura do `.dat`
    (número de ângulos, frequências, medidas por ângulo) para que
    leitores externos possam parsear corretamente.

    Args:
        cfg: SimulationConfig com `export_binary_dat=True`.
        n_models: Número total de modelos no lote (`nmaxmodel` Fortran).
        angulos_deg: Array `(ntheta,)` float64 com ângulos em graus.
        freqs_hz: Array `(nf,)` float64 com frequências em Hz.
        nmeds_per_theta: Array `(ntheta,)` int com número de medidas
            por ângulo de inclinação.
        output_path: Caminho do arquivo. Se `None`, usa
            `{output_dir}/info{output_filename}.out`.

    Returns:
        `pathlib.Path` absoluto do arquivo criado.

    Raises:
        RuntimeError: Se `cfg.export_binary_dat=False`.

    Example:
        Metadata mínimo (1 theta, 1 freq, 600 medidas)::

            >>> cfg = SimulationConfig(
            ...     export_binary_dat=True, output_dir="/tmp/sim"
            ... )
            >>> path = export_out_metadata(
            ...     cfg, n_models=1,
            ...     angulos_deg=np.array([0.0]),
            ...     freqs_hz=np.array([20000.0]),
            ...     nmeds_per_theta=np.array([600]),
            ... )
    """
    if not cfg.export_binary_dat:
        raise RuntimeError("export_out_metadata requer cfg.export_binary_dat=True.")

    angulos_deg = np.asarray(angulos_deg, dtype=np.float64)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    nmeds_per_theta = np.asarray(nmeds_per_theta, dtype=np.int64)

    if output_path is None:
        output_path = Path(cfg.output_dir) / f"info{cfg.output_filename}.out"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ntheta = len(angulos_deg)
    nf = len(freqs_hz)

    # Layout paridade fifthBuildTIVModels.py linhas 1310-1314
    lines: list[str] = []
    lines.append(f" {ntheta} {nf} {n_models}")
    lines.append(" " + " ".join(f"{float(a):.1f}" for a in angulos_deg))
    lines.append(" " + " ".join(f"{float(f):.1f}" for f in freqs_hz))
    lines.append(" " + " ".join(str(int(n)) for n in nmeds_per_theta))

    # Se F7 ativado, adiciona linhas extras
    if cfg.use_tilted_antennas and cfg.tilted_configs is not None:
        n_tilted = len(cfg.tilted_configs)
        lines.append(f" 1 {n_tilted}")
        betas = " ".join(f"{t[0]:.4f}" for t in cfg.tilted_configs)
        phis = " ".join(f"{t[1]:.4f}" for t in cfg.tilted_configs)
        lines.append(f" {betas}")
        lines.append(f" {phis}")
    else:
        lines.append(" 0 0")

    content = "\n".join(lines) + "\n"
    output_path.write_text(content, encoding="utf-8")
    logger.debug("info .out gravado em %s", output_path)

    return output_path.resolve()


__all__ = [
    "DTYPE_22COL",
    "export_binary_dat",
    "export_out_metadata",
]
