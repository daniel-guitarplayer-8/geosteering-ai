#!/usr/bin/env python3
r"""Valida\u00e7\u00e3o num\u00e9rica entre sa\u00eddas bin\u00e1rias de dois benchmarks.

Compara dois arquivos `.dat` (formato unformatted/stream) gerados por
`tatu.x` em fases diferentes (ex: baseline vs phase1) para garantir que
otimiza\u00e7\u00f5es de c\u00f3digo n\u00e3o introduziram regress\u00e3o num\u00e9rica al\u00e9m da
toler\u00e2ncia especificada (default: `atol=1e-10`, `rtol=1e-12`).

Formato esperado do registro (22 colunas, conforme `writes_files`):
    [idx(i4), z_obs(f8), rho_h(f8), rho_v(f8),
     Re(H1)(f8), Im(H1)(f8), ..., Re(H9)(f8), Im(H9)(f8)]

Uso:
    python3 bench/validate_numeric.py baseline.dat phase1.dat [--atol 1e-10]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


# record layout: i4 + 21 f8 = 4 + 168 = 172 bytes (stream access, no padding)
RECORD_FMT = "<i" + "d" * 21
RECORD_SIZE = struct.calcsize(RECORD_FMT)
assert RECORD_SIZE == 172, f"record size mismatch: {RECORD_SIZE}"


def load_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega um arquivo .dat bin\u00e1rio como (indices_int32, values_float64[N, 21])."""
    raw = path.read_bytes()
    if len(raw) % RECORD_SIZE != 0:
        raise ValueError(
            f"{path}: tamanho {len(raw)} n\u00e3o \u00e9 m\u00faltiplo de {RECORD_SIZE} bytes/registro"
        )
    n = len(raw) // RECORD_SIZE
    dtype = np.dtype([("i", "<i4"), ("v", "<f8", 21)])
    arr = np.frombuffer(raw, dtype=dtype, count=n)
    return arr["i"].copy(), arr["v"].copy()


def compare(a_path: Path, b_path: Path, atol: float, rtol: float) -> int:
    print(f"[+] Carregando {a_path.name} ...")
    ai, av = load_dat(a_path)
    print(f"    \u2192 {len(ai)} registros, shape values={av.shape}")

    print(f"[+] Carregando {b_path.name} ...")
    bi, bv = load_dat(b_path)
    print(f"    \u2192 {len(bi)} registros, shape values={bv.shape}")

    if ai.shape != bi.shape:
        print(
            f"[\u2717] FALHA: n\u00famero de registros diferente "
            f"({len(ai)} vs {len(bi)})"
        )
        return 2

    if not np.array_equal(ai, bi):
        print("[\u2717] FALHA: coluna de \u00edndices divergiu")
        return 3

    diff = av - bv
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    rms = float(np.sqrt((diff ** 2).mean()))

    # relativo protegido contra divis\u00e3o por zero
    denom = np.maximum(np.abs(av), np.abs(bv))
    safe = denom > 0
    rel = np.zeros_like(abs_diff)
    rel[safe] = abs_diff[safe] / denom[safe]
    max_rel = float(rel.max())

    # por coluna
    per_col_max = abs_diff.max(axis=0)

    print()
    print("## Compara\u00e7\u00e3o num\u00e9rica")
    print(f"    max |\u0394|            = {max_abs:.4e}")
    print(f"    RMS(\u0394)             = {rms:.4e}")
    print(f"    max rel             = {max_rel:.4e}")
    print(f"    toler\u00e2ncia atol    = {atol:.1e}")
    print(f"    toler\u00e2ncia rtol    = {rtol:.1e}")
    print()
    print("## M\u00e1ximo |\u0394| por coluna (21 cols: z_obs, rho_h, rho_v, Re/Im(H1..H9))")
    col_names = ["z_obs", "rho_h", "rho_v"] + [
        f"{part}(H{k})" for k in range(1, 10) for part in ("Re", "Im")
    ]
    for name, v in zip(col_names, per_col_max):
        marker = "  OK" if v < atol else " !!!"
        print(f"    {name:>9s}: {v:.4e}{marker}")

    ok = np.allclose(av, bv, atol=atol, rtol=rtol)
    print()
    if ok:
        print(f"[\u2713] PASS — diferen\u00e7a dentro da toler\u00e2ncia")
        return 0
    n_bad = int((abs_diff > atol).any(axis=1).sum())
    print(f"[\u2717] FAIL — {n_bad} registros com |\u0394| > atol")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("other", type=Path)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-12)
    args = ap.parse_args()
    return compare(args.baseline, args.other, args.atol, args.rtol)


if __name__ == "__main__":
    sys.exit(main())
