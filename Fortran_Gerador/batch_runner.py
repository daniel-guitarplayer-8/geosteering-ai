#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch Runner — Execução paralela de múltiplos modelos via fifthBuildTIVModels.py.

Feature 5 do roadmap: processamento paralelo de modelos independentes usando
multiprocessing para maximizar throughput de geração de datasets.

Uso:
    python3 batch_runner.py --models 3000 --workers 4 --omp-threads 2

Cada worker:
  1. Cria um diretório temporário
  2. Copia tatu.x e model.in para o diretório
  3. Executa tatu.x com OMP_NUM_THREADS limitado
  4. Move os .dat de volta ao diretório principal

O número total de threads = workers × omp_threads deve ser ≤ núcleos físicos.

Ref: docs/reference/analise_evolucao_simulador_fortran_python.md §5
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_model_batch(args):
    """Executa um lote de modelos sequenciais em diretório temporário."""
    worker_id, model_start, model_end, tatu_path, model_in_path, omp_threads = args
    env = {**os.environ, 'OMP_NUM_THREADS': str(omp_threads)}

    with tempfile.TemporaryDirectory(prefix=f'tatu_w{worker_id}_') as tmpdir:
        # Copiar executável
        tatu_dst = os.path.join(tmpdir, 'tatu.x')
        shutil.copy2(tatu_path, tatu_dst)
        os.chmod(tatu_dst, 0o755)

        # Copiar model.in base
        model_dst = os.path.join(tmpdir, 'model.in')
        shutil.copy2(model_in_path, model_dst)

        # Executar sequencialmente os modelos deste lote
        for i in range(model_start, model_end + 1):
            # Atualizar a última linha do model.in com (modelm, nmaxmodel)
            with open(model_dst, 'r') as f:
                lines = f.readlines()
            # Última linha: "modelm nmaxmodel"
            lines[-1] = f'{i} {model_end}         !modelo atual e o número máximo de modelos\n'
            with open(model_dst, 'w') as f:
                f.writelines(lines)

            subprocess.run([tatu_dst], cwd=tmpdir, env=env,
                           capture_output=True, check=True)

        # Mover .dat e .out gerados de volta
        results = []
        for f in Path(tmpdir).glob('*.dat'):
            results.append(str(f))
        for f in Path(tmpdir).glob('*.out'):
            results.append(str(f))

        return worker_id, model_start, model_end, results


def main():
    parser = argparse.ArgumentParser(description='Batch runner para simulador Fortran PerfilaAnisoOmp')
    parser.add_argument('--models', type=int, default=1000, help='Número total de modelos')
    parser.add_argument('--workers', type=int, default=4, help='Número de workers paralelos')
    parser.add_argument('--omp-threads', type=int, default=2, help='Threads OpenMP por worker')
    parser.add_argument('--tatu', type=str, default='./tatu.x', help='Caminho para tatu.x')
    parser.add_argument('--model-in', type=str, default='./model.in', help='Caminho para model.in base')
    args = parser.parse_args()

    total = args.models
    n_workers = args.workers
    omp_t = args.omp_threads

    print(f'[Batch] {total} modelos, {n_workers} workers × {omp_t} OMP threads = {n_workers * omp_t} threads total')

    # Distribuir modelos entre workers
    chunk = total // n_workers
    remainder = total % n_workers
    batches = []
    start = 1
    for w in range(n_workers):
        end = start + chunk - 1 + (1 if w < remainder else 0)
        batches.append((w, start, end, os.path.abspath(args.tatu), os.path.abspath(args.model_in), omp_t))
        start = end + 1

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_model_batch, b): b for b in batches}
        for future in as_completed(futures):
            wid, ms, me, results = future.result()
            print(f'  Worker {wid}: modelos {ms}-{me} concluídos ({len(results)} arquivos)')

    elapsed = time.perf_counter() - t0
    throughput = total / elapsed * 3600
    print(f'[Batch] {total} modelos em {elapsed:.1f}s = {throughput:.0f} modelos/h')


if __name__ == '__main__':
    main()
