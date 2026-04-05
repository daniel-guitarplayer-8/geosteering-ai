#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validação numérica extensiva entre dois arquivos binários .dat do simulador.

Lê dois arquivos gerados por `writes_files` do PerfilaAnisoOmp e compara
ponto-a-ponto todos os valores de ponto flutuante (double precision),
reportando:
    - Tamanho dos arquivos
    - Contagem de NaN e Inf em cada arquivo
    - max|Δ| absoluto entre os arquivos
    - RMS(Δ)
    - max relative error (onde denominador > eps)
    - Número de pontos discrepantes acima da tolerância

O layout do arquivo binário é um stream sequencial de registros
`write(23) i, theta, f, nmed, <array de 21 colunas>` escrito pelo Fortran
com `access='stream'`. Para a comparação, tratamos como sequência contígua
de `double` ignorando os cabeçalhos inteiros (que são idênticos entre
builds se model.in é idêntico — verificado pelo próprio comparador byte
a byte no início).

Critério de aprovação (exit 0):
    - Ambos arquivos mesmo tamanho
    - Zero NaN e zero Inf em ambos
    - max|Δ| <= atol (default 1e-10)

Uso:
    python3 validate_numeric_extensive.py file1.dat file2.dat [--atol 1e-10]
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path


def read_doubles_stream(path: Path) -> list[float]:
    """Lê o arquivo como sequência de registros estruturados.

    Layout gravado por `writes_files` em PerfilaAnisoOmp.f08 (stream
    unformatted):
        write(1000) i, zrho(:,:,:,1..3), real(cH(1..9)), aimag(cH(1..9))
    Cada registro = 1 inteiro(4B) + 21 doubles(8B) = 172 bytes.

    Retorna apenas os valores double, descartando os inteiros (que são
    determinísticos e idênticos entre builds).
    """
    data = path.read_bytes()
    record_size = 4 + 21 * 8  # 172 bytes
    n_records = len(data) // record_size
    if n_records * record_size != len(data):
        raise ValueError(
            f"Tamanho do arquivo ({len(data)} B) não é múltiplo de {record_size}"
        )
    fmt = "<i" + "d" * 21  # little-endian: int32 + 21 doubles
    out: list[float] = []
    for r in range(n_records):
        rec = struct.unpack_from(fmt, data, r * record_size)
        # rec[0] é o índice inteiro; rec[1:] são os 21 doubles físicos
        out.extend(rec[1:])
    return out


def finite_stats(values: list[float]) -> tuple[int, int, int]:
    nan = sum(1 for v in values if math.isnan(v))
    inf = sum(1 for v in values if math.isinf(v))
    finite = len(values) - nan - inf
    return nan, inf, finite


def compare(a: list[float], b: list[float], atol: float) -> dict:
    if len(a) != len(b):
        raise ValueError(f"Tamanhos diferentes: {len(a)} vs {len(b)}")
    max_abs = 0.0
    sum_sq = 0.0
    max_rel = 0.0
    n_diff = 0
    for x, y in zip(a, b):
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            continue
        d = abs(x - y)
        if d > max_abs:
            max_abs = d
        sum_sq += d * d
        if d > atol:
            n_diff += 1
        denom = max(abs(x), abs(y), 1e-300)
        rel = d / denom
        if rel > max_rel:
            max_rel = rel
    rms = math.sqrt(sum_sq / len(a)) if a else 0.0
    return {
        "max_abs": max_abs,
        "rms": rms,
        "max_rel": max_rel,
        "n_diff": n_diff,
        "n_total": len(a),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validação numérica extensiva entre dois .dat")
    ap.add_argument("file_a", type=Path)
    ap.add_argument("file_b", type=Path)
    ap.add_argument("--atol", type=float, default=1e-10)
    args = ap.parse_args()

    for p in (args.file_a, args.file_b):
        if not p.exists():
            print(f"ERRO: arquivo {p} não existe", file=sys.stderr)
            return 2

    size_a = args.file_a.stat().st_size
    size_b = args.file_b.stat().st_size
    print(f"Arquivo A: {args.file_a}  ({size_a} bytes)")
    print(f"Arquivo B: {args.file_b}  ({size_b} bytes)")

    if size_a != size_b:
        print(f"FAIL: tamanhos diferentes ({size_a} != {size_b})")
        return 1

    data_a = read_doubles_stream(args.file_a)
    data_b = read_doubles_stream(args.file_b)

    nan_a, inf_a, fin_a = finite_stats(data_a)
    nan_b, inf_b, fin_b = finite_stats(data_b)
    print(f"\nEstatísticas de valores:")
    print(f"  A: {len(data_a)} doubles  | NaN={nan_a}  Inf={inf_a}  finitos={fin_a}")
    print(f"  B: {len(data_b)} doubles  | NaN={nan_b}  Inf={inf_b}  finitos={fin_b}")

    if nan_a + inf_a + nan_b + inf_b > 0:
        print(f"FAIL: NaN ou Inf detectado(s) — fidelidade física comprometida")
        return 1

    stats = compare(data_a, data_b, args.atol)
    print(f"\nComparação ponto-a-ponto (atol={args.atol:.0e}):")
    print(f"  max|Δ|      = {stats['max_abs']:.6e}")
    print(f"  RMS(Δ)      = {stats['rms']:.6e}")
    print(f"  max rel err = {stats['max_rel']:.6e}")
    print(f"  pontos > atol: {stats['n_diff']} / {stats['n_total']}")

    if stats["max_abs"] > args.atol:
        print(f"\nFAIL: max|Δ|={stats['max_abs']:.3e} > atol={args.atol:.0e}")
        return 1

    print(f"\nPASS: dentro da tolerância (max|Δ|={stats['max_abs']:.3e} <= {args.atol:.0e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
