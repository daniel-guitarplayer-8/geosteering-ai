#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orquestrador Sequencial Paralelo e Gerador Estocástico para Geração Massiva de Datasets Geofísicos.

Módulo: batch_runner.py
Versão: 3.0 (Metamorfose Dinâmica Clean Way)

Descrição Arquitetural:
-----------------------
Este módulo atua como o Controlador Mestre (Orquestrador) do simulador de perfis eletromagnéticos 
1D utilizado no projeto Geosteering AI. Ele implementa o paradigma "Central Master-Plan, Parallel Execution",
substituindo a antiga operação linear e presa a I/O baseada em modelos estáticos.

Fluxo de Trabalho Primário:
1. Geração Centralizada (Master CPU): Importa o motor matemático nativo `fifthBuildTIVModels` e 
   compila, unicamente em memória RAM, dezenas de milhares de dicionários contendo distribuições 
   Quasi-Monte Carlo (Sobol) para as variáveis de engenharia de poço (resistividades anisotrópicas 
   e espessuras estratigráficas).
2. Distribuição Computacional (Chunks): Fatiamento equitativo dos perfis matemáticos por processos.
3. Execução Isolada (Workers Sandboxes): Cada worker multiprocessado recria o arquivo de entrada (`model.in`)
   de forma dinâmica, reescrevendo estocasticamente a estrutura geológica antes de cada acionamento 
   do Fortran. Instancia o binário compilado (`tatu.x`) isoladamente, impedindo Data Race e colisões.
4. Concatenação Binária e Inspeção: Realiza um merge atômico dos resíduos gerados localmente e inspeciona 
   o binário unificado à procura de de NaNs/Infs (falências numéricas) no espaço complexo eletromagnético,
   empregando Mapeamento de Memória (Memory Mapping) para eficiência algorítmica `O(1)` na RAM.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Injeta o diretório do próprio script no PATH, garantindo que módulos adjacentes
# (como fifthBuildTIVModels.py) sejam localizados independentemente do diretório de
# invocação. Usar __file__ em vez de os.getcwd() evita falha quando o script é
# chamado de outro diretório (ex.: python Fortran_Gerador/batch_runner.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ==============================================================================
# SUB-ROTINAS DO ORQUESTRADOR CENTRAL
# ==============================================================================

def fabricar_universos_aleatorios(ntmodels):
    """
    Sintetiza de forma determinística-aleatória (distribuição estocástica) o conjunto completo 
    de modelos geológicos matemáticos aplicados à propagação EM.

    Utilização:
        A matriz base do universo simulado repousa sobre a divisão fracionária rigorosa projetada 
        para garantir que a Deep Learning que ingira os dados reconheça variabilidades críticas. O motor
        interno apoia-se num hipercubo de amostras Sobol, garantindo cobertura espacial densa para o problema inverso.

    Args:
        ntmodels (int): A volumetria alvo do dataset informada pelo usuário via ArgumentParser.

    Returns:
        list[dict]: Uma lista linear de dicionários estruturados, onde cada dicionário carrega a assinatura de uma formação:
                    - 'n_layers': int, o número de camdas estáticas.
                    - 'rho_h': list de float, Condutividade transversal (Horizontal).
                    - 'rho_v': list de float, Condutividade vertical da fratura orientada.
                    - 'thicknesses': list de float, as espessuras sequenciais relativas à profundidade total (tji).

    Raises:
        ImportError: Capturada caso o gerador de matrizes matemáticas original seja corrompido ou falte no ecossistema.
    """
    try:
         import Fortran_Gerador.fifthBuildTIVModels as fbt
    except ImportError:
        try:
           import fifthBuildTIVModels as fbt
        except ImportError as e:
           print("[FATAL ERROR] Falha Crítica: O motor raiz de Entropia Estocástica ('fifthBuildTIVModels.py') não é alcançável.")
           raise e

    print(f"\n[MATRIZ ESTOCÁSTICA] Injetando a Fatura de Geologias e Sorteios ({ntmodels} perfis via Sobol)...")
    t0 = time.perf_counter()

    # O pipeline constrói lotes fracionados usando arquiteturas patológicas desenhadas 
    # por geofísicos para forçar invariância nos tensores EM das NNs. As frações são fixas:
    m1 = fbt.baseline_empirical_2(nmodels = int(ntmodels * 0.15))
    m2 = fbt.baseline_ncamuniform_2(nmodels = int(ntmodels * 0.10))
    m3 = fbt.baseline_thick_thicknesses_2(nmodels = int(ntmodels * 0.10))
    m4 = fbt.unfriendly_empirical_2(nmodels = int(ntmodels * 0.125))
    m5 = fbt.unfriendly_noisy_2(nmodels = int(ntmodels * 0.175))
    m6 = fbt.generate_pathological_models_2(nmodels = int(ntmodels * 0.1))
    
    m7 = fbt.generate_pathological_models_2(
        nmodels = int(ntmodels * 0.125), rho_h_min = 0.1, rho_h_max = 1500, 
        lambda_min = 1, lambda_max = 1.001, min_thickness_internal = 10, p_camfina = 0.3)
        
    m8 = fbt.baseline_empirical_2(
        nmodels = int(ntmodels * 0.125), rho_h_min = 0.1, rho_h_max = 1500, 
        rhohdistr = 'loguni', lambda_min = 1, lambda_max = 1.0001, min_thickness = 0.5)

    # Concatenação atômica no escopo Mestre de toda a cadeia produzida.
    models_list = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8
    
    # Reparação Algoritmica: Devido ao arredondamento da função de truncamento `int()` operando 
    # em floats (frações do total demandado), é matematicamente previsível a falta de alguns modelos
    # nas decimais somadas. O delta algébrico é saneado adicionando profiles na categoria baseline puro.
    faltante = ntmodels - len(models_list)
    if faltante > 0:
        models_list += fbt.baseline_empirical_2(nmodels = faltante)

    print(f"  -> {len(models_list)} Universos computados de ponta a ponta na Memória Primária Mestre. (T= {time.perf_counter() - t0:.2f}s)")
    return models_list


# ==============================================================================
# F10 — Estratégia B: Paralelização de Perturbações via Workers (ProcessPoolExecutor)
# ==============================================================================

def expand_models_with_perturbations(models_list, delta_rel=1e-4):
    """
    Expande cada modelo geológico em (1 + 4n) sub-modelos para cálculo do Jacobiano
    ∂H/∂ρ via diferenças finitas centradas (F10 — Estratégia B).

    Cada modelo original gera:
      - 1 sub-modelo nominal (is_nominal=True)
      - 4n sub-modelos perturbados: n camadas × 2 componentes (h, v) × 2 sinais (±δ)

    Cada sub-modelo carrega metadados `jacobian_meta` para permitir reagrupamento
    posterior por `parent_id` na função `compute_jacobian_from_perturbations`.

    Fórmula da FD centrada aplicada no pós-processamento:
        J_ijk = (H_i(ρ_k + δ) − H_i(ρ_k − δ)) / (2 δ)
        δ = max(delta_rel × |ρ_k|, 1e-6)

    Args:
        models_list (list[dict]): Lista de modelos geológicos originais.
        delta_rel (float): Perturbação relativa (default 1e-4 = 0,01%).

    Returns:
        list[dict]: Lista estendida com campo 'jacobian_meta' em cada dict.
            - nominal: {'parent_id': i, 'is_nominal': True}
            - perturbado: {'parent_id': i, 'is_nominal': False,
                           'layer': k, 'component': 'h'|'v', 'sign': +1|-1,
                           'delta': δ_usado}

    Ref: docs/reference/relatorio_vantagens_jacobiano.md §9.2 (Estratégia B)
    """
    expanded = []
    for i, model in enumerate(models_list):
        n = model['n_layers']
        # Modelo nominal (cópia com metadado adicional)
        nominal = {
            'n_layers': n,
            'rho_h': list(model['rho_h']),
            'rho_v': list(model['rho_v']),
            'thicknesses': list(model['thicknesses']),
            'jacobian_meta': {'parent_id': i, 'is_nominal': True},
        }
        expanded.append(nominal)

        # Perturbações ±δ em cada (camada, componente)
        for k in range(n):
            for comp, key in [('h', 'rho_h'), ('v', 'rho_v')]:
                rho_ref = float(model[key][k])
                delta = max(delta_rel * abs(rho_ref), 1e-6)
                for sign in (+1, -1):
                    perturbed = {
                        'n_layers': n,
                        'rho_h': list(model['rho_h']),
                        'rho_v': list(model['rho_v']),
                        'thicknesses': list(model['thicknesses']),
                        'jacobian_meta': {
                            'parent_id': i,
                            'is_nominal': False,
                            'layer': k,
                            'component': comp,
                            'sign': sign,
                            'delta': delta,
                        },
                    }
                    perturbed[key][k] = rho_ref + sign * delta
                    expanded.append(perturbed)

    return expanded


def compute_jacobian_from_perturbations(dat_path, expanded_models,
                                          n_original_models, out_jac_path,
                                          n_cols=22, n_freqs=2, n_meds=600,
                                          n_theta=1, n_tr=1):
    """
    Reagrupa sub-modelos por `parent_id` e calcula o Jacobiano ∂H/∂ρ via
    diferenças finitas centradas, a partir dos resultados do .dat mesclado.

    Lê o arquivo binário .dat (formato 22-col, 172 bytes/registro) usando
    numpy memmap (O(1) RAM), extrai o tensor completo de cada sub-modelo e
    computa:
        J[parent, nTR, θ, j, f, c, layer, h/v] = (H_plus − H_minus) / (2 δ)

    Resultado salvo em formato NumPy .npz com chaves:
      - 'dH_dRho_h': complex128 (N_models, nTR, ntheta, nmmax, nf, 9, max_n)
      - 'dH_dRho_v': complex128 (mesma shape)
      - 'n_layers_per_model': int array (N_models,)
      - 'deltas': float array (N_models, max_n, 2)
      - 'parent_ids': int array (N_models,)  -- identifica origem

    Args:
        dat_path (str): Caminho do .dat mesclado gerado pelo batch.
        expanded_models (list[dict]): Lista expandida de sub-modelos (com 'jacobian_meta').
        n_original_models (int): Número de modelos originais (antes da expansão).
        out_jac_path (str): Caminho de saída para o .jac.npz.
        n_cols (int): Colunas do registro binário (22 padrão).
        n_freqs (int): nf no model.in.
        n_meds (int): nmmax (medidas por ângulo).
        n_theta (int): ntheta.
        n_tr (int): nTR.

    Returns:
        None: Salva em disco em `out_jac_path`.

    Ref: docs/reference/relatorio_vantagens_jacobiano.md §9.2
    """
    # dtype do registro 22-col: 1 int32 + 21 float64
    fields = [('col0', np.int32)] + [(f'col{i}', np.float64) for i in range(1, n_cols)]
    rec_dtype = np.dtype(fields)

    mm = np.memmap(dat_path, dtype=rec_dtype, mode='r')
    regs_per_model = n_tr * n_theta * n_meds * n_freqs

    n_submodels = len(expanded_models)
    n_records_expected = n_submodels * regs_per_model
    if mm.shape[0] < n_records_expected:
        print(f"[F10 ERRO] .dat tem {mm.shape[0]} registros mas esperado >= {n_records_expected}")
        return

    # Extrai o tensor H(3,3) de um sub-modelo: lê os 18 float64 de Hxx..Hzz
    def extract_tensor(submodel_idx):
        # Offset do sub-modelo no .dat mesclado
        start = submodel_idx * regs_per_model
        end = start + regs_per_model
        slc = mm[start:end]
        # H shape: (nTR, ntheta, nmmax, nf, 9) complex
        H = np.zeros((n_tr, n_theta, n_meds, n_freqs, 9), dtype=np.complex128)
        # Estrutura do registro: i(1) zobs(1) rho_h(1) rho_v(1) Re(H1)(1) Im(H1)(1) ... Re(H9)(1) Im(H9)(1)
        # Total = 4 + 18 = 22 colunas — col4..col21 são Re/Im alternados
        # Ordem de leitura: (itr, kt, fi, jm) — matching Fortran writes_files:
        #   do k=1,nt  →  do j=1,nf (freq outer)  →  do i=1,nmeds(k) (pos inner)
        # Cada registro .dat contém TODOS os 9 componentes em col4..col21.
        # idx incrementa 1× por (itr, kt, fi, jm) — NÃO dentro do loop ic.
        idx = 0
        for itr in range(n_tr):
            for kt in range(n_theta):
                for fi in range(n_freqs):      # frequência — outer (j no Fortran)
                    for jm in range(n_meds):   # posição   — inner (i no Fortran)
                        for ic in range(9):
                            re = slc[idx][f'col{4 + 2*ic}']
                            im = slc[idx][f'col{5 + 2*ic}']
                            H[itr, kt, jm, fi, ic] = complex(re, im)
                        idx += 1  # 1 registro por (itr, kt, fi, jm) — fora do ic
        return H

    # Reagrupa por parent_id
    groups = {}  # parent_id -> {'nominal': H, 'perturbations': [...]}
    for sub_idx, submodel in enumerate(expanded_models):
        meta = submodel['jacobian_meta']
        pid = meta['parent_id']
        if pid not in groups:
            groups[pid] = {'nominal': None, 'perturbations': []}
        if meta['is_nominal']:
            groups[pid]['nominal'] = sub_idx
        else:
            groups[pid]['perturbations'].append((sub_idx, meta))

    # Determina n_max de camadas para alocação do array
    max_n = max(m['n_layers'] for m in expanded_models)

    # Aloca arrays de saída
    dJ_h = np.zeros((n_original_models, n_tr, n_theta, n_meds, n_freqs, 9, max_n),
                    dtype=np.complex128)
    dJ_v = np.zeros((n_original_models, n_tr, n_theta, n_meds, n_freqs, 9, max_n),
                    dtype=np.complex128)
    n_layers_arr = np.zeros(n_original_models, dtype=np.int32)
    deltas = np.zeros((n_original_models, max_n, 2), dtype=np.float64)
    parent_ids = np.zeros(n_original_models, dtype=np.int32)

    # Processa cada parent_id
    for pid, group in groups.items():
        if group['nominal'] is None:
            print(f"[F10 AVISO] parent_id={pid} sem nominal — pulando")
            continue

        parent_ids[pid] = pid
        nominal_model = expanded_models[group['nominal']]
        n_layers = nominal_model['n_layers']
        n_layers_arr[pid] = n_layers

        # Agrupa perturbações por (layer, component) → (H_plus, H_minus, delta)
        pert_dict = {}  # (layer, comp) -> {'plus': H, 'minus': H, 'delta': δ}
        for sub_idx, meta in group['perturbations']:
            key = (meta['layer'], meta['component'])
            if key not in pert_dict:
                pert_dict[key] = {'plus': None, 'minus': None, 'delta': meta['delta']}
            H_pert = extract_tensor(sub_idx)
            if meta['sign'] > 0:
                pert_dict[key]['plus'] = H_pert
            else:
                pert_dict[key]['minus'] = H_pert

        # Calcula J = (H_plus − H_minus) / (2 δ) para cada (layer, comp)
        for (layer, comp), data in pert_dict.items():
            if data['plus'] is None or data['minus'] is None:
                continue
            J = (data['plus'] - data['minus']) / (2.0 * data['delta'])
            if comp == 'h':
                dJ_h[pid, :, :, :, :, :, layer] = J
                deltas[pid, layer, 0] = data['delta']
            else:
                dJ_v[pid, :, :, :, :, :, layer] = J
                deltas[pid, layer, 1] = data['delta']

    # Salva .jac.npz
    np.savez_compressed(
        out_jac_path,
        dH_dRho_h=dJ_h,
        dH_dRho_v=dJ_v,
        n_layers_per_model=n_layers_arr,
        deltas=deltas,
        parent_ids=parent_ids,
    )
    print(f"[F10-B] Jacobiano salvo em {out_jac_path} | "
          f"shape=(N={n_original_models}, nTR={n_tr}, ntheta={n_theta}, "
          f"nmmax={n_meds}, nf={n_freqs}, 9, max_n={max_n})")


def read_jac_file(path):
    """
    Lê arquivo binário .jac v3 gerado pelo simulador Fortran (F10 Estratégia C).

    Formato stream unformatted (sem record markers Fortran):

      Header (escrito UMA vez por arquivo quando modelm == 1):
        magic        : 4 bytes  (ASCII b'JAC3')
        version      : int32    (= 3)
        nt           : int32    (ntheta — número de ângulos)
        nmmax        : int32    (nmeds máximo entre ângulos)
        nf           : int32    (número de frequências)
        n_components : int32    (= 9, constante)
        itr          : int32    (índice TR que gerou este arquivo, 1-based)
        nTR          : int32    (total de pares TR no modelo.in)
        nmaxmodel    : int32    (valor informativo — leitores contam até EOF)
        nmeds[nt]    : int32[nt] (medidas efetivas por ângulo — ≤ nmmax)

      Payload (repetido por modelo, até EOF):
        model_id     : int32    (identificador 1-based do modelo)
        n_layers     : int32    (número de camadas DESTE modelo — pode variar)
        dH_dRho_h    : (Σ nmeds) × nf × 9 × n_layers pares (Re float64, Im float64)
                       Ordem dos loops Fortran: (k, j, i, ic, layer)
                         k=ângulo, j=posição∈[1,nmeds(k)], i=freq, ic=comp, layer
        dH_dRho_v    : mesma estrutura que dH_dRho_h

    Compatibilidade v2:
        Também suporta arquivos v2 (magic b'JAC2') como fallback legado, desde
        que todos os modelos tenham o mesmo n_layers fixado no header.

    Estratégia de leitura v3 (duas passagens):
        1. Primeira passagem: caminha pelo arquivo contando modelos e registrando
           n_layers de cada um. Determina max_n_layers global.
        2. Segunda passagem: aloca arrays de saída com dimensão max_n_layers e
           preenche modelo-a-modelo, zero-pad onde n_layers_model < max_n_layers.

    Notas:
        - Apenas nmeds(k) posições por ângulo k são gravadas (não nmmax).
          Os slots j > nmeds(k) do array de saída permanecem como zero (pad).
        - Para nTR > 1 o Fortran gera arquivos separados: {filename}_TR{itr}.jac.
          Cada arquivo deve ser lido independentemente.

    Args:
        path (str | Path): Caminho para o arquivo .jac v3 (ou v2 legado).

    Returns:
        dict com chaves:
          'header'    : dict com nt, nmmax, nf, n_components, n_layers_max, itr,
                        nTR, nmaxmodel, nmeds (np.int32 shape (nt,))
          'dH_dRho_h' : np.complex128, shape (N, nt, nmmax, nf, 9, max_n_layers)
                        onde N é o número real de modelos lidos (via EOF)
                        zero-padded em j > nmeds(k) e layer > n_layers_model
          'dH_dRho_v' : mesma shape que dH_dRho_h
          'model_ids' : np.int32, shape (N,)
          'n_layers_per_model' : np.int32, shape (N,)

    Raises:
        ValueError: Magic inválido ou versão incompatível.
        EOFError:   Arquivo truncado no meio de um payload.

    Exemplo:
        >>> jac = read_jac_file('Inv0Dip0_1000.jac')
        >>> jac['header']['nmaxmodel']
        1000
        >>> jac['dH_dRho_h'].shape
        (1000, 1, 600, 2, 9, 30)   # max_n_layers = 30 (maior modelo)
        >>> jac['n_layers_per_model'][:3]
        array([ 7, 12,  5], dtype=int32)  # cada modelo pode ter tamanho diferente

    Ref: docs/reference/documentacao_simulador_fortran_otimizado.md §11.9.5
    """
    import struct

    with open(path, 'rb') as f:
        # ── Leitura do header ────────────────────────────────────────────────
        magic = f.read(4)
        if magic not in (b'JAC2', b'JAC3'):
            raise ValueError(
                f"Magic inválido: {magic!r} — esperado b'JAC2' ou b'JAC3'. "
                f"Verifique se o arquivo é .jac gerado por F10 Estratégia C.")

        version = struct.unpack('<i', f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(
                f"Versão incompatível: {version} — somente v2 ou v3 suportado.")

        if version == 3:
            # v3: 7 inteiros (sem n_layers)
            nt, nmmax, nf, n_comp, itr_hdr, nTR, nmaxmodel = \
                struct.unpack('<7i', f.read(28))
            n_layers_fixed = None  # determinado por-modelo
        else:
            # v2: 8 inteiros (n_layers fixo no header)
            nt, nmmax, nf, n_comp, n_layers_fixed, itr_hdr, nTR, nmaxmodel = \
                struct.unpack('<8i', f.read(32))

        # nmeds por ângulo (nt valores int32)
        nmeds = np.frombuffer(f.read(nt * 4), dtype=np.int32).copy()
        payload_start = f.tell()
        nmeds_sum = int(np.sum(nmeds))

        # ── Passagem 1: conta modelos e registra n_layers por modelo ─────────
        # Usa seek() para pular payloads sem lê-los — O(nmaxmodel) int32 reads.
        n_layers_list = []
        model_ids_list = []
        while True:
            raw_id = f.read(4)
            if len(raw_id) < 4:
                break  # EOF normal
            model_id = struct.unpack('<i', raw_id)[0]

            if version == 3:
                raw_nl = f.read(4)
                if len(raw_nl) < 4:
                    raise EOFError(
                        f"Arquivo truncado: n_layers faltando no modelo {len(n_layers_list)+1}")
                n_layers_m = struct.unpack('<i', raw_nl)[0]
            else:
                n_layers_m = n_layers_fixed

            # Tamanho do payload deste modelo (h + v)
            payload_bytes = 2 * (nmeds_sum * nf * n_comp * n_layers_m) * 16
            # Verifica se cabe antes de avançar
            cur_pos = f.tell()
            f.seek(0, 2)   # fim
            remaining = f.tell() - cur_pos
            f.seek(cur_pos)
            if remaining < payload_bytes:
                # Payload truncado — aborta a contagem neste modelo
                break
            f.seek(payload_bytes, 1)
            model_ids_list.append(model_id)
            n_layers_list.append(n_layers_m)

        n_models_actual = len(n_layers_list)
        if n_models_actual == 0:
            # Retorna estrutura vazia para arquivo válido mas sem payloads
            header = {
                'nt': nt, 'nmmax': nmmax, 'nf': nf, 'n_components': n_comp,
                'n_layers_max': 0, 'itr': itr_hdr, 'nTR': nTR,
                'nmaxmodel': nmaxmodel, 'nmeds': nmeds,
            }
            return {
                'header':     header,
                'dH_dRho_h':  np.zeros((0, nt, nmmax, nf, n_comp, 0), dtype=np.complex128),
                'dH_dRho_v':  np.zeros((0, nt, nmmax, nf, n_comp, 0), dtype=np.complex128),
                'model_ids':  np.zeros(0, dtype=np.int32),
                'n_layers_per_model': np.zeros(0, dtype=np.int32),
            }

        n_layers_per_model = np.array(n_layers_list, dtype=np.int32)
        max_n_layers = int(n_layers_per_model.max())

        header = {
            'nt': nt, 'nmmax': nmmax, 'nf': nf, 'n_components': n_comp,
            'n_layers_max': max_n_layers, 'itr': itr_hdr, 'nTR': nTR,
            'nmaxmodel': nmaxmodel, 'nmeds': nmeds,
        }

        # ── Alocação dos arrays de saída (zero-padded em n_layers_max) ───────
        dJ_h = np.zeros((n_models_actual, nt, nmmax, nf, n_comp, max_n_layers),
                        dtype=np.complex128)
        dJ_v = np.zeros_like(dJ_h)
        model_ids = np.array(model_ids_list, dtype=np.int32)

        offsets = np.concatenate(([0], np.cumsum(nmeds))).astype(np.int64)

        # ── Passagem 2: leitura real dos payloads ────────────────────────────
        f.seek(payload_start)
        for m in range(n_models_actual):
            # Consome model_id + n_layers (já lidos na passagem 1)
            f.read(4)  # model_id
            if version == 3:
                f.read(4)  # n_layers
            n_layers_m = int(n_layers_per_model[m])

            n_cplx = nmeds_sum * nf * n_comp * n_layers_m

            # dH_dRho_h: Re/Im intercalados
            raw_bytes = f.read(n_cplx * 16)
            if len(raw_bytes) < n_cplx * 16:
                raise EOFError(
                    f"Payload dH_dRho_h truncado no modelo {m+1}")
            raw_f64 = np.frombuffer(raw_bytes, dtype=np.float64)
            cplx_h = raw_f64[0::2] + 1j * raw_f64[1::2]

            # dH_dRho_v
            raw_bytes_v = f.read(n_cplx * 16)
            if len(raw_bytes_v) < n_cplx * 16:
                raise EOFError(
                    f"Payload dH_dRho_v truncado no modelo {m+1}")
            raw_f64_v = np.frombuffer(raw_bytes_v, dtype=np.float64)
            cplx_v = raw_f64_v[0::2] + 1j * raw_f64_v[1::2]

            # Distribui por ângulo k; layers > n_layers_m ficam zerados.
            for k in range(nt):
                nm_k = int(nmeds[k])
                if nm_k == 0:
                    continue
                i0 = int(offsets[k]) * nf * n_comp * n_layers_m
                cnt = nm_k * nf * n_comp * n_layers_m
                dJ_h[m, k, :nm_k, :, :, :n_layers_m] = \
                    cplx_h[i0: i0 + cnt].reshape(nm_k, nf, n_comp, n_layers_m)
                dJ_v[m, k, :nm_k, :, :, :n_layers_m] = \
                    cplx_v[i0: i0 + cnt].reshape(nm_k, nf, n_comp, n_layers_m)

    return {
        'header':              header,
        'dH_dRho_h':           dJ_h,
        'dH_dRho_v':           dJ_v,
        'model_ids':           model_ids,
        'n_layers_per_model':  n_layers_per_model,
    }


def merge_jac_files(chunks, output_path, total_models):
    """
    Mescla arquivos .jac v3 (ou v2 legado) de múltiplos workers em um único
    arquivo unificado.

    Cada worker gera um .jac parcial com os modelos do seu chunk. Esta função:
      1. Lê o header do primeiro chunk para obter as dimensões invariantes
         (nt, nmmax, nf, n_comp, nmeds).
      2. Escreve um header atualizado com nmaxmodel = n_modelos_reais_lidos
         (não confia no valor de chunk, que pode estar incorreto para workers
         que não começam em model_id=1).
      3. Copia sequencialmente os payloads — cada um com tamanho variável
         ditado pelo `n_layers` do próprio payload (formato v3).

    Estratégia v3: como cada modelo pode ter n_layers diferente, a cópia é
    streaming — lê (model_id, n_layers), calcula bytes do payload, copia.
    Em v2 legado, assume n_layers fixo do header do chunk.

    A ordem dos payloads no arquivo mesclado reflete a ordem dos workers
    (que é a mesma ordem crescente de model_id, já que chunks são gerados
    sequencialmente por model_start).

    Args:
        chunks (list[tuple]): Lista de (worker_id, filepath_absoluto) já
            ordenada por worker_id ascendente.
        output_path (str): Caminho de saída do arquivo .jac mesclado.
        total_models (int): Total de modelos ESPERADO no lote completo.
            Usado como metadado informativo no header; o valor REAL gravado
            é o número de modelos efetivamente lidos dos chunks.

    Raises:
        ValueError: Se o primeiro chunk tiver magic inválido.

    Ref: docs/reference/documentacao_simulador_fortran_otimizado.md §11.9.5
    """
    import struct

    if not chunks:
        raise ValueError("merge_jac_files: lista de chunks vazia.")

    # ── Lê header do primeiro chunk para obter dimensões invariantes ────────
    first_path = chunks[0][1]
    with open(first_path, 'rb') as f:
        magic_first = f.read(4)
        if magic_first not in (b'JAC2', b'JAC3'):
            raise ValueError(
                f"Magic inválido no primeiro chunk: {magic_first!r} — esperado b'JAC2' ou b'JAC3'.")
        version_first = struct.unpack('<i', f.read(4))[0]
        if version_first == 3:
            nt, nmmax, nf, n_comp, itr_hdr, nTR, _nmaxmodel_chunk = \
                struct.unpack('<7i', f.read(28))
        elif version_first == 2:
            nt, nmmax, nf, n_comp, _nlfix, itr_hdr, nTR, _nmaxmodel_chunk = \
                struct.unpack('<8i', f.read(32))
        else:
            raise ValueError(f"Versão não suportada: {version_first}")
        nmeds = np.frombuffer(f.read(nt * 4), dtype=np.int32).copy()

    nmeds_sum = int(np.sum(nmeds))

    # ── Escreve arquivo mesclado SEMPRE em formato v3 ───────────────────────
    # Mesmo que os chunks sejam v2, convertemos para v3 no merge — v3 é o
    # formato canônico. Cada payload ganha um campo n_layers explícito.
    n_total_written = 0
    with open(output_path, 'wb') as fout:
        # Placeholder header (nmaxmodel = total_models — atualizado depois se diferente)
        fout.write(b'JAC3')
        fout.write(struct.pack('<i', 3))
        fout.write(struct.pack('<7i',
                               nt, nmmax, nf, n_comp,
                               itr_hdr, nTR, total_models))
        fout.write(nmeds.astype(np.int32).tobytes())

        # Copia payloads de cada chunk em ordem
        for wid, fpath in chunks:
            n_copied = 0
            with open(fpath, 'rb') as fin:
                # Detecta versão deste chunk (pode diferir do primeiro)
                fin.read(4)  # descarta magic
                ver_c = struct.unpack('<i', fin.read(4))[0]
                if ver_c == 3:
                    fin.read(28)  # pula 7 ints
                    nl_fixed_chunk = None
                else:
                    _pack = struct.unpack('<8i', fin.read(32))
                    nl_fixed_chunk = _pack[4]
                fin.read(nt * 4)  # pula nmeds

                while True:
                    raw_id = fin.read(4)
                    if len(raw_id) < 4:
                        break  # EOF normal
                    model_id = struct.unpack('<i', raw_id)[0]

                    if ver_c == 3:
                        raw_nl = fin.read(4)
                        if len(raw_nl) < 4:
                            print(f"  [AVISO merge_jac] n_layers truncado em worker {wid}")
                            break
                        n_layers_m = struct.unpack('<i', raw_nl)[0]
                    else:
                        n_layers_m = nl_fixed_chunk

                    payload_bytes = 2 * (nmeds_sum * nf * n_comp * n_layers_m) * 16
                    payload = fin.read(payload_bytes)
                    if len(payload) < payload_bytes:
                        print(f"  [AVISO merge_jac] Payload truncado em worker {wid} "
                              f"(modelo {model_id}): {len(payload)}/{payload_bytes} bytes")
                        break

                    # Grava no formato v3 canônico: (model_id, n_layers, payload)
                    fout.write(struct.pack('<2i', model_id, n_layers_m))
                    fout.write(payload)
                    n_copied += 1

            n_total_written += n_copied
            print(f"     worker {wid}: {n_copied} modelo(s) copiado(s) de {Path(fpath).name}")

        # Atualiza nmaxmodel no header se o real diferiu do esperado
        if n_total_written != total_models:
            print(f"  [AVISO merge_jac] n_modelos lidos ({n_total_written}) ≠ "
                  f"total_models esperado ({total_models}) — corrigindo header.")
            fout.seek(4 + 4 + 6 * 4)   # posição de nmaxmodel no header v3
            fout.write(struct.pack('<i', n_total_written))

    out_size_kb = os.path.getsize(output_path) / 1024
    print(f"  [F10-C] .jac mesclado: {output_path} "
          f"({n_total_written} modelos, {out_size_kb:.1f} KB)")


def extract_model_in_header(model_in_path):
    """
    Extrator estático do prólogo de configuração primária física (`model.in`).

    Funcionamento:
        Filtra recursivamente da linha 1 até o terminador designado `!nome dos arquivos de saída`. 
        Tais linhas prescrevem ao precondicionador numérico do Fortran parâmetros intransigentes (ferramenta):
        - Frequências harmônicas.
        - Ângulos de mergulho (Dip / Anisotropia Relativa).
        - Espaçamentos T-R nativos para cálculo DOI (Profundidade de Investigação).
        Tais informações não devem sofrer metamorfose por perfil para que a ferramenta permaneça em equivalência.

    Args:
        model_in_path (str): URI absolutada ou relativa do arquivo template (ex: `model.in`).
    
    Returns:
        list[str]: Array vetorial de strings, mantendo inclusive os delimitadores `\n` originais para
                   posterior despejo no disco.
    """
    header_lines = []
    with open(model_in_path, 'r') as f:
        for line in f:
            header_lines.append(line)
            if '!nome dos arquivos de saída' in line:
                break
    return header_lines


def parse_model_in(filepath):
    """
    Decodificador semi-analítico das informações visuais indexadas do cabeçalho de simulação.

    Objetivo:
        Extrair os números absolutos operados na janela (Para Logs em terminal amigáveis no Console GUI).

    Args:
        filepath (str): Caminho local contendo a rota do arquivo estruturado (model.in).

    Returns:
        dict ou None: Um dicionário de mapeamento descrevendo os inputs geométricos da ferramenta de Logging.
                      Devolve None caso haja anomalia severa estrutural (ex: quebra de contrato numérico).
    """
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        idx = 0
        nf = int(lines[idx].split()[0]); idx += 1
        freqs = []
        for _ in range(nf):
            freqs.append(float(lines[idx].split()[0])); idx += 1
            
        ntheta = int(lines[idx].split()[0]); idx += 1
        angles = []
        for _ in range(ntheta):
            angles.append(float(lines[idx].split()[0])); idx += 1
            
        idx += 3 
        nTR = int(lines[idx].split()[0]); idx += 1
        trs = []
        for _ in range(nTR):
            trs.append(float(lines[idx].split()[0])); idx += 1
            
        return {'nf': nf, 'freqs': freqs, 'ntheta': ntheta, 'angles': angles, 'nTR': nTR, 'trs': trs}
    except Exception as e:
        return None


def validar_integridade(filepath):
    """
    Scanner C++ Integrado Preventivo contra Infs/NaNs com complexidade Assintótica `O(1)` de RAM.

    Conceito Matemático:
        No processo da solução recursiva das matrizes de Reflexão/Transmissão Eletromagnética nas interfaces,
        solvers diferenciais tendem a estourar assintotas gerando `NaN`. Identificar vazamentos em bases
        de dados textuais gigantescas afaga o sistema. Por isso o simulador despeja binários brutos cravados.

    Engenharia (Memmap):
        Configura o dtype de Numpy em mapeamento vetorial na placa de disco (memmap), transladando a stream 
        sequencial do Fortran (1 `Int32` para ID + 21 `Float64` para Tensões) em visualizações mascaradas super-rápidas. 
        Permite validar Petabytes em segundos avaliando blocos virtuais de 172 bytes.

    Args:
        filepath (str): Arquivo `.dat` final ou fragmento nativo cujo encoding de matriz será varrido.
    """
    try:
        # Assinatura contratual entre C/Fortran Compiler e a ponte Python. 172 Bytes/Frame:
        dtyp = np.dtype([('col0', np.int32)] + [('col{}'.format(i), np.float64) for i in range(1, 22)])
        
        # O Memory Map anexa o disco diretamente no ponteiro interno da CPython.
        mmap = np.memmap(filepath, dtype=dtyp, mode='r')
        
        has_error = False
        error_logs = []
        
        # Ignora a coluna [0] por se tratar de integer indexador natural. Pula diretamente aos campos elétricos complexos.
        for i in range(1, 22):
            col_name = f'col{i}'
            col_vuln = mmap[col_name]
            
            anomalia = np.isnan(col_vuln).any() or np.isinf(col_vuln).any()
            if anomalia:
                has_error = True
                error_logs.append(f"Detectado Vazamento Funcional Inf/NaN no Escopo do Tensor da Coluna: {col_name}")
                break 
                
        if has_error:
            print(f"    [!] AVISO CRÍTICO DE SOLVER: Falhas matemáticas numéricas (NaN/Inf) na Matriz Discreta de {filepath}")
            for err in error_logs:
                print(f"        -> {err}")
        else:
            print(f"    [+] CERTIFICADO DE INTEGRIDADE: Auditoria finalizada. Validação rigorosa reporta 0 falhas no Log Binário Mestre.")
            
        del mmap # Limpeza estrita e imediata do Buffer Virtual (desaloca referências ativas ao arquivo).
    except Exception as e:
        print(f"    [!] Colapso Operacional no Leitor Binário {filepath}: Diagnóstico Inválido. Razão: {e}")


def run_model_batch(args):
    """
    Fábrica Executora do Perímetro Assíncrono (Worker Sandbox).
    
    Finalidade:
        Esta rotina encarna a verdadeira Metamorfose Estocástica. Ela opera em um cordão umbilical 
        completamente desconectado da Main. Possui seu próprio executável `.x` manipulado clonado no escopo 
        da pasta `/tmp`, gerindo chamadas multithreading OpenMP que o Sistema Operacional dispacher.
        
    Execução e Lógica Dinâmica Tática:
        Em lugar da leitura rígida do `arquivo.in` e simples incrementação de contador da iteração passada, este motor:
        1. Resgata e grava o Cabeçalho Estático.
        2. Analisa o `geo_dict` do loop de Chunk para inferir topograficamente Resistividades Finais e Espessuras Variáveis.
        3. Realiza o Print-Out numérico para as linhas exclusivas do `model.in` local, instanciando uma infraestrutura unicamente 
           exclusiva para os solvers Fortran subjacentes engajarem sua marcha.
        4. Transfere sub-arquivos binários retornando o tracking da finalização para o `Future`.

    Args:
        args (tuple): Contrato empacotado que engloba (worker_id, chunk_models_list, header_lines, model_start, model_end, tatu_path, omp_threads)

    Returns:
        tuple: (ID Original do Worker Ativo, Offset de Início, Offset do Fim Sub-pacote, Array de tuplas dos DATs Fragmentados, Array das Tags OUT)
    """
    # F5/F7/F10: args estendidos com flags opcionais
    # Backward compat: len(args) == 7 (v7.0), 9 (v8/v9), 12 (v10 com F10)
    if len(args) == 12:
        (worker_id, chunk_models_list, header_lines, model_start, model_end,
         tatu_path, omp_threads,
         use_arb_freq, feature_tilted,
         use_jacobian, jacobian_method, jacobian_fd_step) = args
    elif len(args) == 9:
        (worker_id, chunk_models_list, header_lines, model_start, model_end,
         tatu_path, omp_threads, use_arb_freq, feature_tilted) = args
        use_jacobian = 0
        jacobian_method = 0
        jacobian_fd_step = 1e-4
    else:
        worker_id, chunk_models_list, header_lines, model_start, model_end, tatu_path, omp_threads = args
        use_arb_freq = 0
        feature_tilted = None
        use_jacobian = 0
        jacobian_method = 0
        jacobian_fd_step = 1e-4
    env = {**os.environ, 'OMP_NUM_THREADS': str(omp_threads)}
    
    header_text = "".join(header_lines)

    # Sub-diretório imutável auto-descartável mitigador de conflitos IOs.
    with tempfile.TemporaryDirectory(prefix=f'tatu_w{worker_id}_') as tmpdir:
        tatu_dst = os.path.join(tmpdir, 'tatu.x')
        shutil.copy2(tatu_path, tatu_dst)
        os.chmod(tatu_dst, 0o755)

        model_dst = os.path.join(tmpdir, 'model.in')

        # ITERAÇÃO DA METAMORFOSE: Compilando as Entropias para Texto Físico.
        for i, geo_dict in enumerate(chunk_models_list):
            current_model_index = model_start + i
            
            ncam = geo_dict['n_layers']
            rho_h = geo_dict['rho_h']
            rho_v = geo_dict['rho_v']
            thick = geo_dict['thicknesses']
            
            with open(model_dst, 'w') as f:
                # Topologia invariante.
                f.write(header_text)
                # Configuração Dinâmica da Interface de Domínio da Propagação:
                f.write(str(ncam) + '                !número de camadas\n')
                
                # Desenhando as Condutividades no Espaço Euclidiano `(horizontal, vertical)`
                for j in range(ncam):
                    rh = round(rho_h[j], 2)
                    rv = round(rho_v[j], 2)
                    if rv < rh: rv = rh
                    if j == 0:
                        f.write(f"{rh}    {rv}     !resistividades horizontal e vertical\n")
                    else:
                        f.write(f"{rh}    {rv}\n")
                
                # Desenhando as camadas delimitadoras (`thicknesses_n-2` fatias delimitadoras intrínsecas)
                for j in range(len(thick)):
                    t_val = round(thick[j], 2)
                    if j == 0:
                        f.write(f"{t_val}              !espessuras das n-2 camadas\n")
                    else:
                        f.write(f"{t_val}\n")
                
                # Trave Lógica do Motor Fortran para Inserções Sequenciais de Acumuladores (`current_id max_id`):
                f.write(f"{current_model_index} {model_end}         !modelo atual e o número máximo de modelos\n")

                # ═══════════════════════════════════════════════════════════════════
                # BLOCO DE FLAGS OPCIONAIS v10.0 — ORDEM CRÍTICA
                # ═══════════════════════════════════════════════════════════════════
                # RunAnisoOmp.f08 lê as linhas opcionais nesta ordem exata:
                #   F5  : use_arbitrary_freq
                #   F7  : use_tilted_antennas [+ n_tilted + (beta,phi)×n_tilted]
                #   F6  : use_compensation    [+ n_comp_pairs + (near,far)×n]
                #   FIL : filter_type
                #   F10 : use_jacobian        [+ jacobian_method + fd_step]
                #
                # Se F10 estiver habilitado, TODAS as linhas precedentes DEVEM ser
                # escritas — senão o Fortran interpreta F10 como F5/F6/filter_type,
                # resultando em comportamento indefinido. O guard EOF via iostat
                # ainda é válido: se paramos no meio de uma seção, apenas o default
                # das seções seguintes é aplicado.
                # ═══════════════════════════════════════════════════════════════════

                # F5 — Frequências arbitrárias
                f.write(f"{use_arb_freq}                 !F5: use_arbitrary_freq (0=desabilitado, 1=habilitado)\n")

                # F7 — Antenas inclinadas
                _use_tilted = 0
                _tilted_configs = []
                if feature_tilted and feature_tilted.get('use_tilted', 0) == 1:
                    _use_tilted = 1
                    _tilted_configs = feature_tilted.get('configs', [])
                f.write(f"{_use_tilted}                 !F7: use_tilted_antennas (0=desabilitado, 1=habilitado)\n")
                if _use_tilted == 1 and len(_tilted_configs) > 0:
                    f.write(f"{len(_tilted_configs)}                 !F7: n_tilted\n")
                    for _it_idx, (_beta_t, _phi_t) in enumerate(_tilted_configs):
                        f.write(f"{_beta_t}  {_phi_t}            !F7: beta({_it_idx+1}) phi({_it_idx+1})\n")

                # F10 — Sensibilidades ∂H/∂ρ (Jacobiano)
                #
                # Duas estratégias distintas:
                #   (B) jacobian_method == 0 → Python Workers orquestram sub-modelos
                #       perturbados. Cada worker roda Fortran sem Jacobiano (use_jacobian=0).
                #       O pós-processamento master calcula J = (H+ − H−)/(2δ).
                #   (C) jacobian_method == 1 → Fortran calcula J internamente via
                #       compute_jacobian_fd (OpenMP), gerando arquivos .jac adjacentes.
                #
                # Para Estratégia B escrevemos use_jacobian=0 (o Fortran não precisa
                # saber). Para Estratégia C escrevemos as três linhas F10 completas
                # PRECEDIDAS por F6=0 e filter_type=0 — a ordem de leitura do Fortran
                # em RunAnisoOmp.f08 é estrita:  F5 → F7 → F6 → filter_type → F10.
                # Omitir F6/filter_type faz o Fortran interpretar "1" do use_jacobian
                # como use_compensation e abortar a cadeia sem chegar em F10.
                if use_jacobian == 1 and jacobian_method == 1:
                    # F6 (use_compensation) — desabilitado em modo Jacobiano.
                    f.write(f"0                 !F6: use_compensation (0=desabilitado)\n")
                    # Filtro Adaptativo — Werthmuller (padrão, 201 pts)
                    f.write(f"0                 !Filtro: 0=Werthmuller (default)\n")
                    # F10 — três linhas obrigatórias
                    f.write(f"1                 !F10: use_jacobian (Estratégia C)\n")
                    f.write(f"{jacobian_method}                 !F10: jacobian_method (1=Fortran OpenMP)\n")
                    f.write(f"{jacobian_fd_step}          !F10: jacobian_fd_step\n")

            # Acionamento Ativo: O código fonte base reprocessa tudo embasado na Matemática Recriada acima.
            # timeout=120s por modelo (>60× o tempo esperado de ~0.06s) previne deadlock
            # de threads OpenMP em casos de divergência numérica ou falha de hardware.
            subprocess.run([tatu_dst], cwd=tmpdir, env=env,
                           capture_output=True, check=True, timeout=120)

        original_dir = os.getcwd()
        results_dat = []
        results_out = []
        results_jac = []

        # Salvamento Parametrizado. Protegendo as Saídas Brutas movendo-as prefixadas perante sobreposição:
        for f in Path(tmpdir).glob('*.dat'):
            new_name = f'w{worker_id}_{f.name}'
            shutil.move(str(f), os.path.join(original_dir, new_name))
            results_dat.append((f.name, new_name))

        for f in Path(tmpdir).glob('*.out'):
            new_name = f'w{worker_id}_{f.name}'
            shutil.move(str(f), os.path.join(original_dir, new_name))
            results_out.append((f.name, new_name))

        # F10 Estratégia C — Salva arquivos .jac gerados pelo Fortran.
        # CRÍTICO: sem este bloco os .jac são silenciosamente destruídos junto
        # com o TemporaryDirectory. Só move quando Strategy C está ativa.
        if use_jacobian == 1 and jacobian_method == 1:
            for f in Path(tmpdir).glob('*.jac'):
                new_name = f'w{worker_id}_{f.name}'
                shutil.move(str(f), os.path.join(original_dir, new_name))
                results_jac.append((f.name, new_name))

        return worker_id, model_start, model_end, results_dat, results_out, results_jac


# ==============================================================================
# ALGORITMO MESTRE (ENTRYPOINT)
# ==============================================================================
def _default_workers_and_omp():
    """
    Determina a config ótima (workers, omp_threads) para o sistema atual.

    Estratégia empiricamente validada (benchmark 3000 modelos, 16 cores lógicos):
      workers=8, omp=2 (16 total) → 210.346 mod/h  ← ÓTIMO
      workers=6, omp=2 (12 total) → 162.639 mod/h  (subutiliza)
      workers=4, omp=4 (16 total) → 167.624 mod/h  (overhead OpenMP)
      workers=2, omp=8 (16 total) → 142.983 mod/h  (poucos workers)

    Regra: workers × omp = cpu_count  AND  workers >= 4× omp
    (maximiza paralelismo de processos, OpenMP só acelera loops internos).

    Returns:
        (workers, omp_threads): tupla de inteiros, produto ≈ os.cpu_count().
    """
    ncpu = os.cpu_count() or 8
    # omp=2 é o sweet spot (OpenMP tem overhead não-trivial para loops curtos).
    # workers absorve o restante. Se ncpu é ímpar, arredonda para baixo.
    omp = 2
    workers = max(1, ncpu // omp)
    return workers, omp


def main():
    """
    Função Primordial que estabelece a Configuração Modular Dinâmica via Terminal.
    Faz parsing lógico de dependências e distribui o pipeline estocástico OMP, lidando com fusões e post-runs.
    """
    _def_w, _def_omp = _default_workers_and_omp()
    parser = argparse.ArgumentParser(description='Geosteering AI Simulator Orchestrator - Modulação Altamente Paralela e Entrópica.')
    parser.add_argument('--models', type=int, default=1000, help='Teto-alvo (Volumetria) Total de modelagens sintéticas geradas no Batch.')
    parser.add_argument('--workers', type=int, default=_def_w,
                        help=f'Paralelização Assíncrona Total (Núcleos virtuais simultâneos alocados p/ Sandbox). '
                             f'Default derivado de os.cpu_count(): {_def_w}.')
    parser.add_argument('--omp-threads', type=int, default=_def_omp,
                        help=f'Granularidade do Solver: Threads OpenMP delegadas por processo Worker. '
                             f'Default otimizado empiricamente: {_def_omp}.')
    parser.add_argument('--tatu', type=str, default='./tatu.x', help='Endereço Relativo do binário C++/Fortran (`tatu.x`).')
    parser.add_argument('--model-in', type=str, default='./model.in', help='Endereço Relativo do esqueleto descritor Topológico (`model.in`).')
    parser.add_argument('--no-concat', action='store_true', help='By-pass paramétrico: Preserva os Data-chunks descentralizados, suprimindo o Merge Binário.')
    # F5/F7 — Feature flags v8.0
    parser.add_argument('--use-arb-freq', type=int, default=0, choices=[0, 1],
                        help='F5: Frequências arbitrárias (0=desabilitado, 1=habilitado). Default: 0.')
    parser.add_argument('--use-tilted', type=int, default=0, choices=[0, 1],
                        help='F7: Antenas inclinadas (0=desabilitado, 1=habilitado). Default: 0.')
    parser.add_argument('--tilted-configs', type=str, default='',
                        help='F7: Configurações tilted "beta1,phi1;beta2,phi2" em graus. Ex: "45,0;30,90".')
    # F10 — Feature flags v10.0 (Sensibilidades ∂H/∂ρ — Jacobiano)
    parser.add_argument('--use-jacobian', type=int, default=0, choices=[0, 1],
                        help='F10: Cálculo do Jacobiano ∂H/∂ρ (0=desabilitado, 1=habilitado). Default: 0.')
    parser.add_argument('--jacobian-method', type=int, default=0, choices=[0, 1],
                        help='F10: Estratégia de cálculo — 0=Python Workers (B), 1=Fortran OpenMP interno (C). Default: 0.')
    parser.add_argument('--jacobian-fd-step', type=float, default=1e-4,
                        help='F10: ε relativo para diferenças finitas centradas. Default: 1e-4.')

    args = parser.parse_args()

    total = args.models
    n_workers = args.workers
    omp_t = args.omp_threads

    # ── Auditoria de balanceamento CPU: previne regressões de throughput ─────
    # Quando workers × omp_threads ≠ os.cpu_count(), o sistema fica sub ou
    # sobrecarregado — impactando throughput em até 30% (ex.: 162k vs 210k
    # mod/h no benchmark de 3000 modelos / 16 cores lógicos). O aviso ajuda o
    # usuário a encontrar a config ótima sem precisar fazer sweep empírico.
    _ncpu_sys = os.cpu_count() or 0
    _total_th = n_workers * omp_t
    if _ncpu_sys > 0:
        if _total_th < _ncpu_sys:
            print(f"[AVISO PERF] workers×omp = {_total_th} < cpu_count = {_ncpu_sys} "
                  f"— SUBUTILIZAÇÃO de {_ncpu_sys - _total_th} core(s). "
                  f"Config ótima sugerida: --workers {_ncpu_sys // 2} --omp-threads 2.")
        elif _total_th > _ncpu_sys:
            print(f"[AVISO PERF] workers×omp = {_total_th} > cpu_count = {_ncpu_sys} "
                  f"— OVERSUBSCRIPTION. Context switches degradarão throughput. "
                  f"Config ótima sugerida: --workers {_ncpu_sys // 2} --omp-threads 2.")

    # Validação antecipada de dependências críticas: falhar rápido antes de alocar
    # workers evita erros crípticos dentro do ProcessPoolExecutor.
    tatu_abs = os.path.abspath(args.tatu)
    if not os.path.isfile(tatu_abs):
        print(f"[FATAL] Binário tatu.x não encontrado: {tatu_abs}")
        print(f"        Compile com `make` em {os.path.dirname(tatu_abs)} e tente novamente.")
        sys.exit(1)
    if not os.path.isfile(args.model_in):
        print(f"[FATAL] Arquivo model.in não encontrado: {args.model_in}")
        sys.exit(1)

    # Setup Logístico e Front-End Mapeável
    print("="*80)
    print(" BATCH RUNNER ORQUESTRADOR METAMÓRFICO — GEOSTEERING AI EXPERT SYSTEM")
    print("="*80)
    
    info = parse_model_in(args.model_in)
    if info:
        print("[ANÁLISE DE TOPOLOGIA DE ENGENHARIA] Carregando Header Restrito do Model.in Base:")
        print(f"  -> Quantidade Frequencial  [nf]: {info['nf']} un | Assinaturas {info['freqs']} Hz")
        print(f"  -> Ângulos Ponto Dip     [ndip]: {info['ntheta']} rot | Disposições {info['angles']} Gr")
        print(f"  -> Tratos Emissores-Receptores : {info['nTR']} Pares | Profundas em {info['trs']} mts")
    else:
        print("[WARNING] Arquitetura de Header base corrompida, suprimindo metadados de Tool GUI.")

    header_lines_estaticas = extract_model_in_header(args.model_in)

    # Corrige o número de modelos embutido no filename do model.in.
    # fifthBuildTIVModels.py grava o filename com ntmodels=3000 hardcoded; ao
    # rodar batch_runner com --models diferente, substituímos o sufixo numérico
    # pelo valor real para que os arquivos .dat/.out reflitam o tamanho do batch.
    for _i, _line in enumerate(header_lines_estaticas):
        if '!nome dos arquivos de saída' in _line:
            _parts = _line.split('!')
            _fname = _parts[0].strip()
            _tokens = _fname.rsplit('_', 1)
            if len(_tokens) == 2 and _tokens[1].isdigit():
                _new_fname = f"{_tokens[0]}_{total}"
                header_lines_estaticas[_i] = f"{_new_fname}              !nome dos arquivos de saída\n"
                print(f"[FILENAME PATCH] {_fname} → {_new_fname}  (--models={total})")
            break

    # F5/F7 — Parse de feature flags
    use_arb_freq_flag = args.use_arb_freq
    feature_tilted_dict = None
    if args.use_tilted == 1 and args.tilted_configs:
        tilted_pairs = []
        for pair_str in args.tilted_configs.split(';'):
            parts = pair_str.strip().split(',')
            if len(parts) == 2:
                tilted_pairs.append((float(parts[0]), float(parts[1])))
        if tilted_pairs:
            feature_tilted_dict = {'use_tilted': 1, 'configs': tilted_pairs}
            print(f"[F7] Antenas inclinadas habilitadas: {len(tilted_pairs)} configuração(ões)")
            for idx_t, (b_t, p_t) in enumerate(tilted_pairs):
                print(f"     tilted({idx_t+1}): beta={b_t}° phi={p_t}°")
    if use_arb_freq_flag == 1:
        print(f"[F5] Frequências arbitrárias habilitadas (nf={info['nf'] if info else '?'})")

    # ── F10 — Parse de flags do Jacobiano ──────────────────────────────
    use_jacobian_flag  = args.use_jacobian
    jacobian_method    = args.jacobian_method
    jacobian_fd_step   = args.jacobian_fd_step
    if use_jacobian_flag == 1:
        if jacobian_method == 0:
            print(f"[F10] Jacobiano habilitado — Estratégia B (Python Workers, "
                  f"expansão 1+4n sub-modelos, fd_step={jacobian_fd_step:.1e})")
        else:
            print(f"[F10] Jacobiano habilitado — Estratégia C (Fortran OpenMP interno, "
                  f"fd_step={jacobian_fd_step:.1e})")

    # Injeta Criação Numérica e Entidade Monte-Carlo Assíncrona no Fluxo
    todos_modelos_gerados = fabricar_universos_aleatorios(total)

    # ── F10 Estratégia B — Expansão em sub-modelos perturbados ─────────
    # Esta expansão multiplica o número de "modelos" a rodar por (1 + 4n),
    # onde n é o número de camadas do modelo geológico. Cada sub-modelo é
    # rodado como se fosse um modelo normal pelo Fortran (sem conhecimento
    # do Jacobiano), e o pós-processamento reagrupa por parent_id.
    original_models_count = total
    if use_jacobian_flag == 1 and jacobian_method == 0:
        print(f"[F10-B] Expandindo {total} modelos em sub-modelos perturbados...")
        original_models = todos_modelos_gerados
        todos_modelos_gerados = expand_models_with_perturbations(
            original_models, delta_rel=jacobian_fd_step)
        n_submodels = len(todos_modelos_gerados)
        print(f"[F10-B] {total} modelos → {n_submodels} sub-modelos "
              f"(fator {n_submodels/total:.1f}×)")
        total = n_submodels  # atualiza total efetivo para chunking
        n_avg_layers = sum(m['n_layers'] for m in original_models) / len(original_models)
        print(f"[F10-B] Camadas médias: {n_avg_layers:.1f} → "
              f"custo por modelo original: 1 + 4×{n_avg_layers:.0f} ≈ "
              f"{1 + 4*n_avg_layers:.0f} chamadas do Fortran")
        
    print(f"\n[ENGAJAMENTO ASSÍNCRONO] Alocando Process Pool Paralelo para modelagem de {total} Perfis Estocásticos.")
    print(f"       -> {n_workers} Instâncias Operacionais Sand-Boxes Independentes limitadas a {omp_t} threads C/U.")
    print("-" * 80)

    # Lógica Algorítmica de Distribuição e Empilhamento O(n):
    chunk = total // n_workers
    remainder = total % n_workers
    batches = []
    start = 1
    for w in range(n_workers):
        # Repassa um diferencial de contagem residual para garantir somatórias perfeitas `N == M`
        end = start + chunk - 1 + (1 if w < remainder else 0)
        
        # Desmembra os perfis matemáticos por worker. Slice na Lista Mestre no Range Específico.
        chunk_dict_list = todos_modelos_gerados[start-1 : end]
        batches.append((w, chunk_dict_list, header_lines_estaticas, start, end, tatu_abs, omp_t,
                        use_arb_freq_flag, feature_tilted_dict,
                        use_jacobian_flag, jacobian_method, jacobian_fd_step))
        start = end + 1

    t0 = time.perf_counter()
    chunks_to_concat = {}

    # Pool de Execuções e Fechamento Místico de Contextos Python. Responde Eventos Concluidos em Tempos Ociosos.
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_model_batch, b): b for b in batches}
        for future in as_completed(futures):
            result = future.result()
            wid, ms, me, res_dat, res_out = result[:5]
            res_jac = result[5] if len(result) > 5 else []
            print(f'  [✓] Nó Analítico Worker-{wid}: Compilação Estrutural {ms}-{me} fechada e exportada.')
            if res_jac:
                print(f'       -> {len(res_jac)} arquivo(s) .jac (F10-C) movido(s).')

            # Ancoragem Base de Lixo para futura mesclagem indexada:
            for base, chunkf in res_dat + res_out + res_jac:
                if base not in chunks_to_concat:
                    chunks_to_concat[base] = []
                chunks_to_concat[base].append((wid, chunkf))

    elapsed = time.perf_counter() - t0
    throughput = total / (elapsed or 1) * 3600
    print("-" * 80)
    print(f"[THROUGHPUT MENSURADO] Dispersão Solucionada global em {elapsed:.1f}s — Rendimento: {throughput:.0f} Modelos/Hr")

    # Finalização Pós-Engenharia: Mesclagem Shutil Binária de Acoplamento Limpo.
    if not args.no_concat:
        print("\n[POST-PROCESS OPERATION] Inciando Protocolo de Concatenação Aderente...")
        for basename, chunks in chunks_to_concat.items():

            # Reorganiza topografias garantindo Profundidades Lineares Contínuas no Array.
            chunks.sort(key=lambda x: x[0])
            final_path = os.path.join(os.getcwd(), basename)
            if os.path.exists(final_path):
                 os.remove(final_path)

            if basename.endswith('.out'):
                # ═══════════════════════════════════════════════════════════════
                # TRATAMENTO ESPECIAL PARA .OUT — NÃO é concatenação binária!
                # ═══════════════════════════════════════════════════════════════
                # O arquivo .out é um cabeçalho TEXTO de metadados, não um stream
                # de registros. Cada worker gera um .out com seu próprio nmaxmodel
                # (o fim do SEU chunk), então concatenar byte-a-byte empilharia N
                # headers no arquivo final. A solução correta é:
                #   1. Ler um fragmento qualquer para extrair nmeds(1..nt)
                #   2. Descartar todos os fragmentos
                #   3. Escrever UM .out limpo em Python com nmaxmodel = total
                # Formato idêntico ao gerado por fifthBuildTIVModels.py:
                #   Linha 1: nt nf nmaxmodel
                #   Linha 2: angles(1..nt)
                #   Linha 3: freqs(1..nf)
                #   Linha 4: nmeds(1..nt)
                # ═══════════════════════════════════════════════════════════════
                print(f"  -> Gerando .out limpo (nmaxmodel={total}): {basename}")

                # Extrai nmeds do primeiro fragmento do worker 0 (todos têm o
                # mesmo nmeds pois tj/pmed/angles são invariantes entre workers).
                first_chunk_path = os.path.join(os.getcwd(), chunks[0][1])
                nmeds_from_frag = None
                try:
                    with open(first_chunk_path, 'r') as _ff:
                        _lines = [ln.strip() for ln in _ff if ln.strip()]
                    # Linha 4 (índice 3) do fragmento Fortran = nmeds por ângulo
                    if len(_lines) >= 4:
                        nmeds_from_frag = [int(x) for x in _lines[3].split()]
                except Exception as _e:
                    print(f"     [AVISO] Falha ao extrair nmeds do fragmento: {_e}")

                # Fallback: se não conseguiu ler, calcula a partir do model.in
                if nmeds_from_frag is None and info is not None:
                    try:
                        with open(args.model_in, 'r') as _fmi:
                            _all = [ln.strip() for ln in _fmi if ln.strip()]
                        # Após nf + ntheta vem h1, tj, pmed (h1 descartado)
                        _idx = 1 + info['nf'] + 1 + info['ntheta']
                        _idx += 1  # pula h1
                        _tj  = float(_all[_idx].split()[0]);     _idx += 1
                        _pm  = float(_all[_idx].split()[0])
                        import math as _math
                        nmeds_from_frag = [
                            _math.ceil(_tj / (_pm * _math.cos(_math.radians(a))))
                            for a in info['angles']
                        ]
                    except Exception as _e:
                        print(f"     [AVISO] Fallback model.in falhou: {_e}")
                        nmeds_from_frag = [0] * (info['ntheta'] if info else 1)

                # Descarta TODOS os fragmentos .out (não servem para concat).
                for wid, chunkf in chunks:
                    chunk_path = os.path.join(os.getcwd(), chunkf)
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)

                # Escreve .out limpo — formato Fortran list-directed REAL(8) / INTEGER
                # ┌────────────────────────────────────────────────────────────────────┐
                # │ Linha 1: inteiros em campo de 12 chars (%12d)                     │
                # │ Linha 2: ângulos  — float 18 chars (3 espaços + nº + 5 espaços)  │
                # │ Linha 3: freqs    — mesmo formato de float                        │
                # │ Linha 4: nmeds    — inteiro em campo de 12 chars (%12d)           │
                # │                                                                    │
                # │ Largura 18 do número = (dígitos antes ponto) + 1 (ponto) + dp    │
                # │   dp = 17 − dígitos_antes_ponto   (conserva 17 chars de dígitos) │
                # │   0.0     → dp=16 → '0.0000000000000000'  ✓                     │
                # │   20000.0 → dp=12 → '20000.000000000000'  ✓                     │
                # └────────────────────────────────────────────────────────────────────┘
                def _ff(x):
                    """Formata float no padrão Fortran REAL(8) list-directed (largura 18)."""
                    ax = abs(float(x))
                    k  = len(str(int(ax))) if ax >= 1.0 else 1  # dígitos antes do ponto
                    dp = max(0, 17 - k)
                    return f'   {float(x):.{dp}f}     '

                _nt = info['ntheta'] if info else 1
                _nf = info['nf']     if info else 1
                _ang = info['angles'] if info else [0.0]
                _frq = info['freqs']  if info else [0.0]
                with open(final_path, 'w') as _fo:
                    _fo.write(''.join(f'{v:12d}' for v in [_nt, _nf, total]) + '\n')
                    _fo.write(''.join(_ff(a) for a in _ang) + '\n')
                    _fo.write(''.join(_ff(f) for f in _frq) + '\n')
                    _fo.write(''.join(f'{n:12d}' for n in nmeds_from_frag) + '\n')
                print(f"     nt={_nt} nf={_nf} nmaxmodel={total} nmeds={nmeds_from_frag}")
                continue

            # ── Merge de .jac binários (F10 Estratégia C) ─────────────────────
            # Cada worker gera um .jac parcial. merge_jac_files lê o header
            # do primeiro chunk, descarta headers dos demais e concatena apenas
            # os payloads, escrevendo um .jac unificado com nmaxmodel=total.
            if basename.endswith('.jac'):
                print(f"  -> Mesclando {len(chunks)} fragmentos .jac → {basename}")
                try:
                    merge_jac_files(
                        chunks=[(wid, os.path.join(os.getcwd(), chunkf))
                                for wid, chunkf in chunks],
                        output_path=final_path,
                        total_models=total,
                    )
                    # Remove fragmentos após merge bem-sucedido
                    for _, chunkf in chunks:
                        chunk_path = os.path.join(os.getcwd(), chunkf)
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
                except Exception as _e:
                    print(f"  [AVISO] Falha ao mesclar .jac: {_e} — fragmentos preservados.")
                continue

            # ── Concatenação binária padrão (apenas para .dat) ─────────────────
            print(f"  -> Calibrando merge de {len(chunks)} sub-sinalizações particionais rumo ao Arquivo Base: {basename}")
            with open(final_path, 'wb') as outfile:
                # Bypass Completo Nativo em C do Modulo Shutil Operacional de IO garantindo FileBlock Swap Memory sem perdas
                for wid, chunkf in chunks:
                    chunk_path = os.path.join(os.getcwd(), chunkf)
                    with open(chunk_path, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
                    os.remove(chunk_path)

            # Barreira Protetora Numérica Restrita (Validador Memmap):
            if basename.endswith('.dat'):
                validar_integridade(final_path)
    else:
        print("\n[POST-PROCESS OPERATION] Sub-rotina de Supressão ativada (By-Pass Merge). Elementos Isolados resguardados.")

    # ──────────────────────────────────────────────────────────────────
    # F10 — Estratégia B: Pós-processamento do Jacobiano
    # ──────────────────────────────────────────────────────────────────
    # Após o merge dos .dat (cada sub-modelo gera regs_per_model registros),
    # reagrupamos por parent_id e calculamos J = (H+ − H−) / (2 δ).
    # Salva resultado em .jac.npz ao lado de cada .dat correspondente.
    if use_jacobian_flag == 1 and jacobian_method == 0 and not args.no_concat:
        print("\n[F10-B] Pós-processamento: reagrupando sub-modelos e calculando Jacobiano...")
        _nf_jac  = info['nf']     if info else 2
        _nt_jac  = info['ntheta'] if info else 1
        _ntr_jac = info['nTR']    if info else 1

        # ── Determinação dinâmica de nmeds por ângulo ──
        # Fonte de verdade: o arquivo .out gerado pelo Fortran contém nmeds(1..nt)
        # na 4ª linha. Caso não exista (por exemplo, se --no-concat for passado),
        # calculamos a partir de (tj, p_med, theta) usando ceil(tj/(p_med·cos(θ))).
        # Fallback final: 600 (valor histórico do geosteering padrão).
        def _nmeds_from_out_file(workdir: str, basename_dat: str) -> int:
            """Tenta ler nmeds a partir do .out correspondente ao .dat."""
            # Heurística: substitui ".dat" por ".out" no nome base.
            out_name = basename_dat.replace('.dat', '.out')
            out_path = os.path.join(workdir, out_name)
            if not os.path.isfile(out_path):
                # Tenta qualquer .out no diretório
                for candidate in os.listdir(workdir):
                    if candidate.endswith('.out'):
                        out_path = os.path.join(workdir, candidate)
                        break
                else:
                    return 0
            try:
                with open(out_path, 'r') as _fo:
                    _lines = [ln.strip() for ln in _fo if ln.strip()]
                # Linha 4 (índice 3): nmeds(1..nt)
                if len(_lines) >= 4:
                    nm_list = [int(x) for x in _lines[3].split()]
                    return max(nm_list) if nm_list else 0
            except Exception:
                return 0
            return 0

        def _nmeds_from_model_in() -> int:
            """Fallback: calcula nmeds a partir do model.in (tj, p_med, ângulos)."""
            if info is None:
                return 600
            try:
                with open(args.model_in, 'r') as _fmi:
                    _all = [ln.strip() for ln in _fmi if ln.strip()]
                _idx = 1 + info['nf'] + 1 + info['ntheta'] + 1  # pula h1
                _tj = float(_all[_idx].split()[0]); _idx += 1
                _pm = float(_all[_idx].split()[0])
                import math as _math
                _nms = [
                    _math.ceil(_tj / (_pm * _math.cos(_math.radians(a))))
                    for a in info['angles']
                ]
                return max(_nms) if _nms else 600
            except Exception:
                return 600

        for basename in chunks_to_concat.keys():
            if not basename.endswith('.dat'):
                continue
            dat_path = os.path.join(os.getcwd(), basename)
            jac_path = dat_path.replace('.dat', '.jac.npz')

            # Determina nmmax (nmeds máximo entre ângulos) dinamicamente
            _nmeds = _nmeds_from_out_file(os.getcwd(), basename)
            if _nmeds <= 0:
                _nmeds = _nmeds_from_model_in()
            if _nmeds <= 0:
                _nmeds = 600
            print(f"  [F10-B] {basename}: n_meds={_nmeds} (detectado dinamicamente)")

            try:
                compute_jacobian_from_perturbations(
                    dat_path=dat_path,
                    expanded_models=todos_modelos_gerados,
                    n_original_models=original_models_count,
                    out_jac_path=jac_path,
                    n_cols=22,
                    n_freqs=_nf_jac,
                    n_meds=_nmeds,
                    n_theta=_nt_jac,
                    n_tr=_ntr_jac,
                )
            except Exception as e:
                print(f"[F10-B ERRO] Falha ao calcular Jacobiano para {basename}: {e}")

    # ──────────────────────────────────────────────────────────────────
    # Resumo final dos arquivos Jacobiano gerados (F10-B ou F10-C)
    # ──────────────────────────────────────────────────────────────────
    # Mostra ao usuário exatamente onde os artefatos .jac/.jac.npz ficaram,
    # com caminho absoluto e tamanho. Facilita pipelines downstream (PINNs,
    # inversão Gauss-Newton) que precisam abrir esses arquivos.
    if use_jacobian_flag == 1 and not args.no_concat:
        jac_artifacts = []
        if jacobian_method == 1:
            # Estratégia C: arquivos .jac binários v3 gerados pelo Fortran
            jac_artifacts = sorted(Path(os.getcwd()).glob('*.jac'))
            label = "F10-C (Fortran OpenMP binário v3)"
        else:
            # Estratégia B: arquivos .jac.npz NumPy gerados pelo Python
            jac_artifacts = sorted(Path(os.getcwd()).glob('*.jac.npz'))
            label = "F10-B (Python Workers NumPy)"

        if jac_artifacts:
            print("")
            print(f"[{label}] Arquivos Jacobiano gerados:")
            for art in jac_artifacts:
                sz_kb = art.stat().st_size / 1024
                sz_mb = sz_kb / 1024
                sz_str = f"{sz_mb:.2f} MB" if sz_mb >= 1 else f"{sz_kb:.1f} KB"
                print(f"   -> {art}  ({sz_str})")
            print(f"   Diretório de saída: {os.getcwd()}")
        else:
            print(f"\n[AVISO {label}] Nenhum arquivo Jacobiano encontrado após execução.")

    print("="*80)
    print("Encerramento Executável com Resolução Sucesso. Datasets Inteligentes Subsequentes Disponíveis!")

if __name__ == '__main__':
    main()
